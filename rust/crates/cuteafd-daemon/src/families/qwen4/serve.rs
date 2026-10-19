//! OpenAI-compatible API over the Qwen 3.8 Flash Next engine: continuous
//! batching with one prefill per admitted request and one decode-shaped step
//! for every active sequence, verifying native MTP drafts (`--mtp N`) under
//! the shared resource policy. History copies only extend a fully agreeing
//! neural window.
//!
//! Verify-by-replay: a step that verifies drafts runs speculatively (GDN and
//! PLE record each row's replay inputs and keep their state); the accepted
//! rows are then committed in one launch, the placement and n-gram history
//! rewound (K/V records past them are rewritten when those positions come
//! again). With MTP the kept rows' pre-mixer streams wait in the MTP stash
//! for the next cycle's draft step (see `speculate`).
//!
//! Prefix cache (`cuteafd_engine::prefix` over `super::prefix::Qwen4Prefix`):
//! admission restores the deepest retained snapshot whose tokens prefix the
//! prompt (shared units, the copied tail unit, the GDN/PLE state mark; the
//! n-gram history is recomputed from the tokens) and prefills only the rest;
//! a whole-prompt hit takes its first token from the retained logits. The
//! prompt is retained at prompt end (unless it was a whole hit), the
//! conversation at a normal finish (`Turn`; its kept rows are committed to
//! the state first), a prefill whose client left is parked at its last chunk;
//! a decode whose client left is not retained. A restored sequence's MTP
//! drafts once its own rows reach the stash. `prompt_cache_hit_tokens`
//! reports the restored rows; `/v1/stats` carries the cache's counters.
use super::engine::{Qwen4Engine, DECODE_ROWS};
use super::prefix::Qwen4Prefix;
use super::open;
use crate::shared::{console, probe, qwen::scheduler::{self, Backend, Drafts}, prefix::CudaCopyEngine,
    prefill_share::DecodeShareArgs};
pub(crate) use crate::shared::prefix::{PrefixArgs, Toggle};
use cuteafd_engine::{media::{EmbeddingCache, MediaAdmission}, prefix::{MarkArena, PrefixCache, PrefixConfig, PrefixFamily}};
use anyhow::{Context, Result};
use cuteafd_api::openai::{chat::qwen4::QwenEncoding, ModelEncoding, ModelProfile, NativeLimits, NativeRequest};
use std::{path::PathBuf, sync::{Arc, Mutex}, time::Duration};
use tokio::sync::mpsc;

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: super::EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Sequences decoding at once (each holds two state slots).
    #[arg(long, default_value_t = 4)]
    pub max_sequences: usize,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    /// With --mtp: verify exactly --mtp drafts per sequence where room allows
    /// instead of the adaptive plan.
    #[arg(long)]
    pub mtp_fixed: bool,
    #[command(flatten)]
    pub decode_share: DecodeShareArgs,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub console: console::ConsoleArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
    /// Resolved global vision policy, assigned before dispatch.
    #[arg(skip = cuteafd_loader::plan::MediaMode::Auto)]
    pub vision: cuteafd_loader::plan::MediaMode,
    /// Host embedding-cache quota; default min(8 GiB, 5% RAM).
    #[arg(long, value_parser = crate::shared::prefix::parse_bytes)]
    pub media_cache_bytes: Option<u64>,
    /// Admitted resident Spark encoder endpoints, in replica order.
    #[arg(long)]
    pub vision_peers: Option<String>,
    /// Shared planner admission hash (required with --vision-peers).
    #[arg(long, requires = "vision_peers")]
    pub encoder_plan_hash: Option<String>,
    /// Snapshot revision used in the encoder handshake.
    #[arg(long, requires = "vision_peers")]
    pub encoder_revision: Option<String>,
}

fn model_id(snapshot: &std::path::Path) -> Option<String> {
    snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    })
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    let api = args.api.load()?;
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let encoding = Arc::new(QwenEncoding::from_snapshot(&snapshot)?);
    let eos = encoding.tokens().eos.clone();
    let mut profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&snapshot)).context("model id")?,
        ModelEncoding::Qwen(encoding),
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    engine_args.slots = engine_args.slots.max(args.max_sequences);
    let (worker_stats, max_sequences, decode_share) = (stats.clone(), args.max_sequences, args.decode_share);
    let drafts = if engine_args.mtp > 0 {
        Drafts::Neural { depth: engine_args.mtp.min(DECODE_ROWS - 1), fixed: args.mtp_fixed }
    } else { Drafts::None };
    let prefix = args.prefix.clone();
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id, drafts, &eos)));
    let vision = args.vision;
    let remote = super::media::RemoteVision::from_args(&args)?;
    let media_cache_bytes = args.media_cache_bytes;
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, drafts, eos, decode_share, prefix, vision, media_cache_bytes, remote));
    let (max_context, media) = ready_rx.await.context("engine failed before it was ready")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    if let Some((preparer, health)) = media {
        profile = profile.with_loaded_vision(preparer);
        profile.vision_health = health;
    }
    cuteafd_bench::context::phase("engine loaded");
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, Duration::from_secs(25),
        hub.clone(), api.serve(profile.clone(), &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    // The ready ledger (placement gate): one report at readiness, before any request captures
    // lazy graphs or loads more code.
    crate::shared::memory_report::log("ready");
    tracing::info!(listen = %args.listen, model = %profile.id, "Qwen 3.8 Flash Next API is ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<std::net::SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
    }
    Ok(())
}

/// What the live console shows for Qwen 3.8 Flash Next.
fn console_layout(args: &ServeArgs, model: &str, drafts: Drafts, eos: &[u32]) -> console::Layout {
    use console::{Color::*, StepGroup};
    let mut layout = console::Layout::new("qwen4", model.into(), args.engine.snapshot.clone());
    let sparks = args.engine.peers.as_deref().map_or(0, |peers| peers.split(',').count());
    layout.hardware = console::hardware(1, sparks, args.engine.local_experts || args.engine.shared_only);
    layout.concurrency = args.max_sequences.min(DECODE_ROWS);
    layout.eos = eos.to_vec();
    layout.speculator = match drafts {
        Drafts::None => None,
        Drafts::Neural { depth, fixed } => Some(console::Speculator { name: "MTP".into(), positions: depth,
            policy: if fixed { format!("fixed {depth}") } else { format!("adaptive ≤ {depth}") } }),
    };
    layout.steps = vec![
        StepGroup::new("Decode step", "host clock", &[("round.cycle", "step", Target),
            ("draft", "draft plan + MTP / copy drafts", Accepted), ("verify", "verify pass + token selection", Target),
            ("emit", "accept + stream", Ink), ("commit", "commit kept rows + MTP stash", Accepted)]),
        StepGroup::new("Verify pass", "host clock, engine phases", &[
            ("gpu", "GPU until expert exchanges", Rtx), ("experts", "expert exchanges", Spark)]),
        StepGroup::layers(),
        StepGroup::admission(false),
    ];
    layout.layers = Some(console::Layers::host_clock());
    layout
}

type VisionReady = Option<(Arc<cuteafd_api::openai::media::MediaPreparer>, Option<Arc<std::sync::atomic::AtomicBool>>)>;

#[allow(clippy::too_many_arguments)]
fn serve_loop(mut args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<(usize, VisionReady)>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    draft: Drafts, eos: Vec<u32>, decode_share: DecodeShareArgs, prefix: PrefixArgs,
    vision: cuteafd_loader::plan::MediaMode, media_cache_bytes: Option<u64>, remote: Option<super::media::RemoteVision>) -> Result<()> {
    args.planner_graph_modes = Some((max_sequences.min(DECODE_ROWS), !matches!(draft, Drafts::None)));
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let (vision, prefix) = match super::media::ReadyVision::load(&args, &opened.library, vision, &prefix, media_cache_bytes, remote) {
        Ok(vision) => vision,
        Err(error) => { let _ = ready.send(Err(error)); return Ok(()); }
    };
    let preparer = vision.as_ref().map(|vision| vision.preparer.clone());
    let (encoder, bytes) = vision.map_or((crate::shared::qwen::Encoder::Off, 0), |vision| (vision.encoder, vision.cache_bytes));
    let health = encoder.health_handle();
    let mut media = MediaAdmission::new(EmbeddingCache::new(bytes), encoder, 16);
    if args.pool_tokens == 0 {
        let geometry = cuteafd_loader::serving_capacity::qwen_cache_geometry(&opened.cfg,
            opened.cfg.layers, args.mtp > 0, args.kv_format)?;
        let mark = geometry.ranks[0].retained_mark_bytes as usize;
        let slots = MarkArena::slots_for(max_sequences, prefix.prefix_cache_entries, mark,
            prefix.prefix_cache_mark_mib << 20);
        args.planner_prefix_bytes = Some(slots as u64 * mark as u64);
    }
    let mut ready = Some(ready);
    let result = opened.with_engine(&args, |engine| {
        anyhow::ensure!(engine.weights.layers.len() == engine.cfg.layers, "serve-qwen4 needs every layer");
        anyhow::ensure!(engine.experts().is_some(),
            "serve-qwen4 needs --peers (or --local-experts) for the routed experts");
        let spark = matches!(engine.experts(), Some(super::engine::Experts::Spark { .. }));
        console::layer_classes(engine.weights.layers.iter().map(|_| console::layer_class(false, spark)).collect());
        if matches!(engine.experts(), Some(super::engine::Experts::SharedOnly)) {
            tracing::warn!("serve-qwen4 --shared-only: replies do not match the model (plumbing and cache gates only)");
        }
        engine.warm_decode_graphs(max_sequences.min(DECODE_ROWS), !matches!(draft, Drafts::None))?;
        if let Ok(path) = std::env::var("CUTEAFD_QWEN4_PADDING_TOKENS") {
            let bytes = std::fs::read(&path).with_context(|| format!("Qwen padding tokens {path}"))?;
            anyhow::ensure!(bytes.len() % 4 == 0, "Qwen padding token file is not U32-aligned");
            let tokens: Vec<_> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
            engine.check_decode_padding(&tokens)?;
        }
        // Startup captures must be visible in the first API statistics snapshot.
        {
            let mut stats = stats.lock().map_err(|_| anyhow::anyhow!("Qwen serving stats lock poisoned"))?;
            *stats = serde_json::json!({});
            probe::graph_capture_stats(&mut stats);
        }
        scheduler::schedule(&Qwen4Backend { engine, opened: &opened }, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), draft, eos,
            decode_share, &prefix, args.token_io.token_select, &mut media, preparer.as_deref(), || {
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Ok((engine.max_context, preparer.clone().map(|p| (p, health.clone())))));
                }
                Ok(())
            })
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| (args.max_context, None)).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

/// The prefix cache over `engine` (always present: with zero entries it is the page allocator),
/// its mark arena sized for `lanes` decoding sequences.
fn prefix_cache<'e, 'a>(engine: &'e Qwen4Engine<'a>, args: &PrefixArgs, lanes: usize)
    -> Result<(Qwen4Prefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let args = &args.with_scoring(if engine.full_prefill_logits { engine.prefill_rows } else { 64 },
        engine.prefill_rows, engine.cfg.vocab_size)?;
    let entries = args.prefix_cache_entries;
    let budget = args.prefix_cache_mark_mib << 20;
    anyhow::ensure!(args.prefix_partial == Toggle::Off, "Qwen 3.8 Flash Next restores exact snapshots only (GDN state)");
    let family = Qwen4Prefix::new(engine, |mark| if entries == 0 { 0 } else { MarkArena::slots_for(lanes, entries, mark, budget) })?;
    let host = args.host_config(family.layout(), engine.max_context, lanes)?.map(|config|
        CudaCopyEngine::registered_owned(engine.library, family.snapshot_owners()).map(|copy| (config, copy)))
        .transpose()?;
    let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
    let layout = family.layout();
    let config = PrefixConfig { entries, mark_slots: family.slots(), keep_logits: true,
        min_tokens: args.prefix_cache_min_tokens };
    let cache = PrefixCache::new(layout, config, host)?;
    tracing::info!(entries, mark_slots = family.slots(), mark_bytes = family.mark_bytes(), page_bytes = layout.page_bytes,
        pages = layout.pages, page_rows = layout.page_rows, kv_format = %engine.kv_format, host_bytes, points = ?args.points(),
        "Qwen 3.8 Flash Next prefix cache");
    cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, layout.pages as u64,
        &format!("{} full-attention + BF16 index + GDN/PLE state", engine.kv_format), host_bytes);
    Ok((family, cache))
}


#[cfg(test)]
mod scheduler_tests {
    #[test]
    fn shared_draft_binding_preserves_config_geometry_and_history_prior() {
        use crate::shared::draft::resource::{self, LayerHome};
        use crate::shared::draft_policy::DraftHistory;
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(
            &cuteafd_loader::plan::testing::qwen4_config(4)).unwrap();
        let homes: Vec<_> = (0..cfg.layers).map(|index| if index < 2 {
            LayerHome::Local { slice_bytes: 4096. }
        } else { LayerHome::Remote { slice_bytes: 8192. } }).collect();
        let geometry = resource::geometry(&homes, cfg.experts, cfg.topk, super::DECODE_ROWS, 3,
            crate::shared::draft::copy::COPY_EXTENSION).unwrap();
        assert_eq!((geometry.layers.len(), usize::from(geometry.experts), usize::from(geometry.topk)),
            (cfg.layers, cfg.experts, cfg.topk));
        assert_eq!((geometry.max_requests, geometry.widths.clone(), geometry.max_positions),
            (super::DECODE_ROWS, vec![3], 35));
        assert!(geometry.layers.iter().all(|layer| layer.class.is_some()));
        let mut history = DraftHistory::default();
        history.observe(3, 2);
        let priors = DraftHistory::acceptance(&[&history], &[3]);
        assert_eq!(priors[0], history.pooled(3, std::iter::empty::<&DraftHistory>()));
    }

    #[test]
    fn shared_copy_trim_preserves_qwen_native_launch_limits() {
        for sequences in 1..=16 {
            for extra in 0..=64-sequences {
                let original: Vec<Vec<u32>> = (0..sequences).map(|i| {
                    (0..=extra / sequences + usize::from(i < extra % sequences))
                        .map(|j| (i * 100 + j) as u32).collect()
                }).collect();
                let real = original.iter().map(Vec::len).sum::<usize>();
                let limit = super::super::engine::copy_row_limit(real, sequences);
                let mut trimmed = original.clone();
                crate::shared::qwen::scheduler::trim_copy_rows(&mut trimmed, limit);
                assert_eq!(trimmed.iter().map(Vec::len).sum::<usize>(), limit);
                for (before, after) in original.iter().zip(&trimmed) {
                    assert!(!after.is_empty());
                    assert_eq!(after, &before[..after.len()]);
                }
            }
        }
    }

}

struct Qwen4Backend<'e, 'a> {
    engine: &'e Qwen4Engine<'a>,
    opened: &'e super::Opened,
}

impl scheduler::Placement for super::engine::Qwen4Placement {
    fn len(&self) -> usize { self.len }
    fn set_len(&mut self, len: usize) { self.len = len; }
    fn units(&self) -> &[u32] { &self.units }
    fn slot(&self) -> i32 { self.slot }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] { self.media.as_ref().map_or(&[], |media| media.spans()) }
}

impl<'e, 'a> Backend<'a> for Qwen4Backend<'e, 'a> {
    type Placement = super::engine::Qwen4Placement;
    type Draft = super::speculate::MtpSeq;
    type History = cuteafd_loader::families::qwen4::NgramHistory;
    type Coordinates = cuteafd_loader::families::qwen4::RopePositions;
    type Encoder = crate::shared::qwen::Encoder;
    type Prefix = Qwen4Prefix<'e, 'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "qwen4", vocab_size: self.engine.cfg.vocab_size,
            max_context: self.engine.max_context, slots: self.engine.slots, decode_rows: DECODE_ROWS,
            prefill_rows: self.engine.prefill_rows, full_prefill_logits: self.engine.full_prefill_logits }
    }
    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.engine.library }
    fn profile(&self) -> &std::cell::RefCell<[f64; 2]> { &self.engine.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        prefix_cache(self.engine, args, lanes)
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>)
        -> Result<(scheduler::Prompt, cuteafd_engine::media::RequestMedia, Vec<cuteafd_engine::media::EncodeJob>)> {
        super::media::prepare(job, tokens, &self.opened.checkpoint.config,
            self.engine.cfg.vocab_size, self.engine.cfg.hidden, self.engine.max_context)
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut cuteafd_engine::media::RequestMedia,
        cache: &mut EmbeddingCache, snapshot: &std::path::Path) -> Result<()> {
        super::media::probe_features(prompt, media, cache, snapshot)
    }
    fn coordinates(&self, tokens: &[u32], job: &NativeRequest, media: &cuteafd_engine::media::RequestMedia)
        -> Result<Self::Coordinates> {
        super::media::rope_positions(tokens, job, media, &self.opened.checkpoint.config)
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Self::Placement {
        Self::Placement::new(units, slot, super::engine::history_of(&self.engine.cfg, &[]))
    }
    fn bind_request(&self, placement: &mut Self::Placement, coordinates: Self::Coordinates,
        media: cuteafd_engine::media::RequestMedia, tokens: &[u32], resume: usize) {
        placement.rope = coordinates;
        placement.media = Some(media);
        placement.history = super::engine::history_of(&self.engine.cfg, &tokens[..resume]);
    }
    fn history(&self, placement: &Self::Placement) -> Self::History { placement.history.clone() }
    fn prefetch(&self, tokens: &[u32], next: &[u32]) {
        self.engine.prefetch_ple(&super::engine::history_of(&self.engine.cfg, tokens), next);
    }
    fn require_live(&self) -> Result<()> { self.engine.require_live() }
    fn log_table_stats(&self, phase: &str) { self.engine.log_table_stats(phase); }
    fn prefill_device(&self, placement: &mut Self::Placement, tokens: &[u32], rows: usize,
        _pool: &cuteafd_engine::prefix::RefPagePool)
        -> Result<Option<crate::shared::token_io::DeviceLogits>> {
        self.engine.prefill_device(placement, tokens, None, None, rows)
    }
    fn verify_device(&self, sequences: &mut [(&mut Self::Placement, &[u32])], speculative: bool,
        _pool: &cuteafd_engine::prefix::RefPagePool)
        -> Result<Option<crate::shared::token_io::DeviceLogits>> {
        self.engine.verify_device(sequences, speculative)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Self::Placement, &[u32])], speculative: bool,
        _pool: &cuteafd_engine::prefix::RefPagePool)
        -> Result<Option<crate::shared::token_io::DeviceLogits>> {
        self.engine.verify_device_ungraphed(sequences, speculative)
    }
    fn verify_bucket_rows(&self, rows: usize, speculative: bool, diagnostic: bool) -> usize {
        self.engine.verify_bucket_rows(rows, speculative, diagnostic)
    }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        self.engine.verify_attention_trace(contexts, rows)
    }
    fn draft_geometry(&self, snapshot: &std::path::Path, width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        self.engine.draft_geometry(snapshot, width)
    }
    fn arm_draft_probe(&self) -> Result<()> { self.engine.arm_draft_probe() }
    fn prefill_draft(&self, placement: &Self::Placement, start: usize, chunk: &[u32], next: Option<u32>, draft: &mut Self::Draft)
        -> Result<()> {
        super::speculate::prefill_chunk(self.engine, placement, start, chunk, next, draft)
    }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Self::Placement, Self::Draft>], timing: &mut scheduler::DraftTiming)
        -> Result<Vec<Vec<u32>>> {
        super::speculate::draft(self.engine, sequences, timing)
    }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Self::Placement, Self::Draft, Self::History>],
        speculative: bool, draft: bool) -> Result<()> {
        super::speculate::accept(self.engine, verified, speculative, draft)
    }
    fn probe_begin(&self, rows: usize) { self.engine.probe_begin(rows); }
    fn probe_end(&self) { self.engine.probe_end(); }
    fn probe_finish<T>(&self, observe: impl FnOnce(&crate::shared::draft::routes::RoundRoutes, &[Option<f64>]) -> T) -> Option<T> {
        self.engine.probe_finish(observe)
    }
}
