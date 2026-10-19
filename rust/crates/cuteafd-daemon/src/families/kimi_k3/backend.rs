//! Text-only, no-drafter adapter to the shared continuous scheduler.
use super::{engine::Engine, prefix::{KimiPrefix, Placement}};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::scheduler::{self, Backend}, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use std::{cell::{Cell, RefCell}, path::Path};

#[derive(Default)]
pub(crate) struct NoDraft;
impl scheduler::DraftState for NoDraft {
    fn pending_rows(&self) -> usize { 0 }
    fn close(&mut self, _token: u32) {}
}

pub(crate) struct KimiBackend<'e, 'a> {
    engine: &'e Engine<'a>,
    profile: RefCell<[f64; 2]>,
    geometry: scheduler::Geometry,
    steady_census: Cell<bool>,
}

impl<'e, 'a> KimiBackend<'e, 'a> {
    pub(crate) fn new(engine: &'e Engine<'a>) -> Result<Self> {
        Ok(Self { engine, profile: RefCell::new([0.0; 2]), geometry: engine.geometry()?,
            steady_census: Cell::new(false) })
    }
}

impl<'e, 'a> Backend<'a> for KimiBackend<'e, 'a> {
    type Placement = Placement;
    type Draft = NoDraft;
    type History = ();
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Prefix = KimiPrefix<'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "kimi_k3", vocab_size: self.geometry.vocab_size,
            max_context: self.geometry.max_context, slots: self.geometry.slots,
            prefill_rows: self.geometry.prefill_rows, decode_rows: self.geometry.decode_rows,
            full_prefill_logits: false }
    }
    fn message_starts(&self) -> &[&str] { &["<|open|>"] }
    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.engine.library() }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(args.prefix_partial == Toggle::Off, "Kimi restores exact KDA/MLA snapshots only");
        let args = args.with_scoring(self.geometry.decode_rows, self.geometry.prefill_rows, self.geometry.vocab_size)?;
        let family = self.engine.prefix_family()?;
        let layout = family.layout();
        let host = args.host_config(layout, self.geometry.max_context, lanes)?.map(|config|
            CudaCopyEngine::registered_owned(self.library(), family.snapshot_owners()).map(|copy| (config, copy)))
            .transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let config = PrefixConfig { entries: args.prefix_cache_entries, mark_slots: family.slots(),
            keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
        let cache = PrefixCache::new(layout, config, host)?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, layout.pages as u64,
            "BF16 owned MLA + exact TP2 KDA", host_bytes);
        Ok((family, cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty(), "Kimi serving currently supports text only");
        let keys = MediaKeys::new(&tokens, u32::try_from(self.geometry.vocab_size)?, &[])?;
        let media = RequestMedia::new(Vec::new(), 7168, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, Vec::new()))
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia,
        _cache: &mut EmbeddingCache, _snapshot: &Path) -> Result<()> {
        ensure!(prompt.job.media.is_empty() && media.spans().is_empty(), "Kimi media is not admitted");
        Ok(())
    }
    fn coordinates(&self, _tokens: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        ensure!(job.media.is_empty() && media.spans().is_empty(), "Kimi text coordinates require no media");
        Ok(())
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement {
        Placement { units, slot: slot as usize, len: 0, state_len: 0 }
    }
    fn bind_request(&self, _placement: &mut Placement, _coordinates: (), _media: RequestMedia,
        _tokens: &[u32], _resume: usize) {}
    fn history(&self, _placement: &Placement) {}
    fn prefetch(&self, _tokens: &[u32], _next: &[u32]) {}
    fn require_live(&self) -> Result<()> { self.engine.require_live() }
    fn log_table_stats(&self, _phase: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.engine.step_device(&mut [(placement, tokens)], pool, logit_rows.max(1), true)
            .map(|logits| (logit_rows > 0).then_some(logits))
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(!speculative, "Kimi speculative state commit is not qualified; use no drafter");
        let rows = sequences.iter().map(|(_, tokens)| tokens.len()).sum();
        self.engine.step_device(sequences, pool, rows, true).map(Some)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(!speculative, "Kimi speculative state commit is not qualified; use no drafter");
        let rows = sequences.iter().map(|(_, tokens)| tokens.len()).sum();
        self.engine.step_device(sequences, pool, rows, false).map(Some)
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"kimi_owned_mla_tp2_kda", "rows":rows, "contexts":contexts})
    }
    fn draft_geometry(&self, _snapshot: &Path, _width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        anyhow::bail!("Kimi drafter is not qualified")
    }
    fn arm_draft_probe(&self) -> Result<()> { anyhow::bail!("Kimi drafter is not qualified") }
    fn prefill_draft(&self, _placement: &Placement, _start: usize, _chunk: &[u32], _next: Option<u32>,
        _draft: &mut NoDraft) -> Result<()> { anyhow::bail!("Kimi drafter is not qualified") }
    fn draft(&self, _sequences: &mut [scheduler::DraftSeq<'_, Placement, NoDraft>], _timing: &mut scheduler::DraftTiming)
        -> Result<Vec<Vec<u32>>> { anyhow::bail!("Kimi drafter is not qualified") }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, NoDraft, ()>], speculative: bool, draft: bool)
        -> Result<()> {
        ensure!(!speculative && !draft, "Kimi requires no drafter");
        for sequence in verified {
            ensure!(sequence.placement.state_len == sequence.start + sequence.rows.len(),
                "Kimi non-speculative state did not commit all input rows");
        }
        // Shared token selection completed the producing stream before accept.
        // Record one steady census, never a per-step sampler or a compute drain.
        if !self.steady_census.replace(true) {
            let devices = self.engine.devices();
            crate::shared::inventory::RuntimeInventory::measure(self.library(), &devices,
                crate::shared::inventory::InventoryPhase::PostLoad, |_| Ok(()))?
                .record("kimi_k3-serving-steady", &[]);
        }
        Ok(())
    }
    fn probe_begin(&self, _rows: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
}
