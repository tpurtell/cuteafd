//! Dense target adapter for the shared Qwen request lifecycle.
use super::engine::{DenseEngine, DenseMark, prefix::{DensePrefix, Placement}};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::scheduler::{self, Backend}, speculation::target::{AcceptedPrefix, BlockVerifyTarget, VerifyRequest, VerifyRow},
    token_io::DeviceLogits};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, MediaAdmission, MediaKeys, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily}};
use cuteafd_ffi::NativeLibrary;
use std::{cell::RefCell, path::Path};

#[derive(Default)]
pub(crate) struct DraftState;
impl scheduler::DraftState for DraftState {
    fn pending_rows(&self) -> usize { 0 }
    fn close(&mut self, _token: u32) {}
}

pub(crate) struct DenseBackend<'e, 'a> {
    pub engine: &'e RefCell<DenseEngine<'a>>,
    pub library: &'a NativeLibrary,
    profile: RefCell<[f64; 2]>,
    pending: RefCell<Option<DenseMark<'a>>>,
}

impl<'e, 'a> DenseBackend<'e, 'a> {
    pub fn new(engine: &'e RefCell<DenseEngine<'a>>, library: &'a NativeLibrary) -> Self {
        Self { engine, library, profile: RefCell::new([0.0; 2]), pending: RefCell::new(None) }
    }
}

impl<'e, 'a> Backend<'a> for DenseBackend<'e, 'a> {
    type Placement = Placement;
    type Draft = DraftState;
    type History = ();
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Prefix = DensePrefix<'e, 'a>;

    fn geometry(&self) -> scheduler::Geometry {
        let engine = self.engine.borrow();
        scheduler::Geometry { family: "qwen3_5", vocab_size: engine.cfg.vocab, max_context: engine.cfg.max_context,
            slots: engine.slot_count(), prefill_rows: 4096, decode_rows: engine.decode_rows(), full_prefill_logits: false }
    }
    fn library(&self) -> &'a NativeLibrary { self.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(args.prefix_partial == Toggle::Off, "dense Qwen supports exact prefix frontiers only");
        let family = DensePrefix::new(self.engine)?;
        let layout = family.layout();
        let shape = self.geometry();
        let args = &args.with_scoring(shape.decode_rows, shape.decode_rows, shape.vocab_size)?;
        let host = args.host_config(layout, shape.max_context, lanes)?.map(|config|
            CudaCopyEngine::registered_owned(self.library, family.snapshot_owners()).map(|copy| (config, copy))).transpose()?;
        let config = PrefixConfig { entries: args.prefix_cache_entries, mark_slots: family.slots(),
            keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
        let cache = PrefixCache::new(layout, config, host)?;
        Ok((family, cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty(), "dense Qwen vision is not yet qualified");
        let engine = self.engine.borrow();
        let keys = MediaKeys::new(&tokens, u32::try_from(engine.cfg.vocab)?, &[])?;
        let media = RequestMedia::new(Vec::new(), engine.cfg.hidden, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, Vec::new()))
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia,
        _cache: &mut EmbeddingCache, _snapshot: &Path) -> Result<()> {
        ensure!(prompt.job.media.is_empty() && media.spans().is_empty(), "dense target cannot accept unqualified media");
        Ok(())
    }
    fn coordinates(&self, _tokens: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        ensure!(job.media.is_empty() && media.spans().is_empty(), "dense text coordinates require no media");
        Ok(())
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement {
        Placement { units, slot: slot as usize, len: 0, state_len: 0 }
    }
    fn bind_request(&self, _placement: &mut Placement, _coordinates: (), _media: RequestMedia, _tokens: &[u32], _resume: usize) {}
    fn history(&self, _placement: &Placement) {}
    fn prefetch(&self, _tokens: &[u32], _next: &[u32]) {}
    fn require_live(&self) -> Result<()> {
        // A selection failure releases the requests without calling accept.
        // The next round boundary must retire that verify before reusing slots.
        if let Some(mark) = self.pending.borrow_mut().take() {
            self.engine.borrow_mut().restore(&mark)?;
        }
        self.engine.borrow().require_live()
    }
    fn log_table_stats(&self, _phase: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        _pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(self.pending.borrow().is_none(), "dense prefill during pending verify");
        let mut engine = self.engine.borrow_mut();
        ensure!(engine.slot_position(placement.slot)? == placement.len, "dense prefill placement/state frontier differs");
        engine.bind_slot(placement.slot, &placement.units)?;
        let output = engine.prefill_device(placement.slot, tokens, logit_rows)?;
        placement.len += tokens.len();
        placement.state_len = placement.len;
        Ok(Some(output))
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        _pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(self.pending.borrow().is_none(), "dense verify requires prior commit");
        let slots = sequences.iter().map(|(placement, _)| placement.slot).collect::<Vec<_>>();
        let mut engine = self.engine.borrow_mut();
        let mut rows = Vec::new();
        for (placement, tokens) in sequences.iter() {
            ensure!(engine.slot_position(placement.slot)? == placement.len, "dense verify placement/state frontier differs");
            engine.bind_slot(placement.slot, &placement.units)?;
            rows.extend(tokens.iter().enumerate().map(|(offset, &token)| VerifyRow {
                slot: placement.slot, position: placement.len + offset, token: Some(token) }));
        }
        let mark = engine.checkpoint(&slots)?;
        let output = match engine.verify(&mark, VerifyRequest { rows: &rows, tap_layers: &[] }) {
            Ok(output) => output.logits,
            Err(error) => { engine.restore(&mark)?; return Err(error); }
        };
        if speculative {
            *self.pending.borrow_mut() = Some(mark);
        } else {
            let accepted = sequences.iter().map(|(placement, tokens)| AcceptedPrefix {
                slot: placement.slot, rows: tokens.len() }).collect::<Vec<_>>();
            engine.commit(&mark, &accepted)?;
            for (placement, tokens) in sequences {
                placement.len += tokens.len();
                placement.state_len = placement.len;
            }
        }
        Ok(Some(output))
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Option<DeviceLogits>> {
        self.verify_device(sequences, speculative, pool)
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"causal_gqa", "rows":rows, "contexts":contexts})
    }
    fn draft_geometry(&self, _snapshot: &Path, _width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        anyhow::bail!("dense drafter adapter is not yet qualified")
    }
    fn arm_draft_probe(&self) -> Result<()> { anyhow::bail!("dense drafter adapter is not yet qualified") }
    fn prefill_draft(&self, _placement: &Placement, _start: usize, _chunk: &[u32], _next: Option<u32>, _draft: &mut DraftState) -> Result<()> {
        anyhow::bail!("dense drafter adapter is not yet qualified")
    }
    fn draft(&self, _sequences: &mut [scheduler::DraftSeq<'_, Placement, DraftState>], _timing: &mut scheduler::DraftTiming)
        -> Result<Vec<Vec<u32>>> { anyhow::bail!("dense drafter adapter is not yet qualified") }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, DraftState, ()>], speculative: bool, draft: bool) -> Result<()> {
        ensure!(!draft, "dense drafter adapter is not yet qualified");
        if !speculative {
            ensure!(self.pending.borrow().is_none(), "dense plain accept has pending speculative state");
            for sequence in verified {
                ensure!(sequence.placement.state_len == sequence.start + sequence.rows.len(),
                    "dense plain step did not commit every input");
            }
            return Ok(());
        }
        let mark = self.pending.borrow_mut().take().context("dense accept requires pending verify")?;
        let accepted = verified.iter().map(|sequence| AcceptedPrefix {
            slot: sequence.placement.slot, rows: sequence.kept.map_or(0, |(rows, _)| rows) }).collect::<Vec<_>>();
        let mut engine = self.engine.borrow_mut();
        if let Err(error) = engine.commit(&mark, &accepted) {
            engine.restore(&mark)?;
            return Err(error);
        }
        for sequence in verified {
            sequence.placement.state_len = engine.slot_position(sequence.placement.slot)?;
            if sequence.kept.is_none() { sequence.placement.len = sequence.placement.state_len; }
            ensure!(sequence.placement.len == sequence.placement.state_len, "dense accepted placement/state frontier differs");
        }
        Ok(())
    }
    fn probe_begin(&self, _rows: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
}

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[arg(long)]
    pub snapshot: std::path::PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: std::path::PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: std::path::PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long)]
    pub pool_tokens: Option<u64>,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long, default_value_t = 32768)]
    pub max_output: u32,
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub max_sequences: u8,
    #[arg(long)]
    pub model_id: Option<String>,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)]
    pub token_io: crate::shared::token_io::TokenIoArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    use cuteafd_api::openai::{chat::qwen4::QwenEncoding, ModelEncoding, ModelProfile, NativeLimits};
    use std::sync::{Arc, Mutex};
    let api = args.api.load()?;
    let snapshot = args.snapshot.clone();
    let encoding = Arc::new(QwenEncoding::from_snapshot(&snapshot)?);
    let eos = encoding.tokens().eos.clone();
    let model = args.model_id.clone().or_else(|| snapshot.ancestors().find_map(|path| {
        let name = path.file_name()?.to_str()?.strip_prefix("models--")?;
        let (organization, model) = name.split_once("--")?;
        Some(format!("{organization}/{model}"))
    })).context("dense serving requires --model-id for a snapshot without HF identity")?;
    let profile = ModelProfile::new(model, ModelEncoding::Qwen(encoding));
    ensure!(matches!(args.token_io.embed_placement, crate::shared::token_io::EmbedPlacement::Gpu),
        "dense target currently admits GPU token embeddings only");
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let engine_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| -> Result<()> {
            let header = cuteafd_loader::families::qwen3_5::Qwen35Checkpoint::read(&engine_args.snapshot)?;
            let checkpoint = cuteafd_loader::plan::checkpoint::Checkpoint::open(&engine_args.snapshot)?;
            // SAFETY: library remains loaded until programs, executor and prefix-copy owners drop.
            let library = unsafe { NativeLibrary::load(&engine_args.native_lib)? };
            library.cuda_set_device(engine_args.device)?;
            let programs = library.programs()?.with_manifest(&engine_args.manifest)?;
            let lanes = usize::from(engine_args.max_sequences);
            let engine = RefCell::new(DenseEngine::with_prefix(&library, &programs, &checkpoint,
                header.config, engine_args.pool_tokens.filter(|&tokens| tokens > 0), None, lanes, Some(engine_args.prefix.mark_rule(lanes)))?);
            let backend = DenseBackend::new(&engine, &library);
            scheduler::schedule(&backend, &engine_args.snapshot, &mut receive, &worker_stats, lanes,
                scheduler::Drafts::None, eos, engine_args.decode_share, &engine_args.prefix,
                engine_args.token_io.token_select, &mut text_media(), None, || {
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok(backend.geometry().max_context));
                    }
                    Ok(())
                })
        })();
        if let Some(ready) = ready.take() {
            let _ = ready.send(Err(anyhow::anyhow!("dense initialization failed: {:#}", result.as_ref().unwrap_err())));
        }
        result
    });
    let max_context = ready_rx.await.context("dense engine failed before readiness")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new(
        "qwen3_5", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<std::net::SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
    }
    Ok(())
}

pub(crate) fn text_media() -> MediaAdmission<scheduler::Prompt, crate::shared::qwen::Encoder> {
    MediaAdmission::new(EmbeddingCache::new(0), crate::shared::qwen::Encoder::Off, 16)
}
