//! Muse Spark on the shared scheduler (`qwen::scheduler::Backend`), text
//! only, no drafter yet.
//!
//! - Prefill: one sequence's chunk as row units on the shared pipeline
//!   (`MusePrefill`, units interleaved per layer, each in its own slot), then
//!   the frontier advances by the chunk; logits for its last row.
//! - Decode/verify: one step over the batch (`forward::step`, the decode bank),
//!   every row scored (at most `HEAD_ROWS` per call: `decode_rows`).
//! - Accept: rows past the accepted frontier stay written but invisible: the
//!   sliding rings keep one 64-row rollback step beyond the window
//!   (`cache::swa_ring`) and full-layer records past the frontier are rewritten
//!   by the next step, so publication is the frontier alone.
use super::{engine::RankResources, experts::{SparkExperts, WaveSegment, WaveStep}, forward, prefill::{MusePrefill, PrefillCall},
    prefix::{MusePrefix, Placement}, step::{StepMetadata, StepSequence}};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::scheduler::{self, Backend}, token_io::{DeviceLogits, TokenEmbedding}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::muse_spark::{lane::HEAD_ROWS, MuseSparkConfig};
use std::{cell::RefCell, ffi::c_void, path::Path};

/// No drafter yet: every step is plain decode.
#[derive(Debug, Default)]
pub(crate) struct NoDraft;
impl scheduler::DraftState for NoDraft {
    fn pending_rows(&self) -> usize { 0 }
    fn close(&mut self, _token: u32) {}
}

pub(crate) struct MuseBackend<'r, 'a> {
    cfg: &'r MuseSparkConfig,
    ranks: &'r [RankResources<'a>],
    streams: Vec<*mut c_void>,
    table: &'r TokenEmbedding<'a>,
    experts: RefCell<&'r mut dyn SparkExperts>,
    peer: Option<&'r super::peer::MusePeer<'a>>,
    library: &'a NativeLibrary,
    profile: RefCell<[f64; 2]>,
    /// Prefill units: rows and waves in flight; the sequence slots, context, pool pages.
    unit_rows: usize,
    waves: usize,
    slots: usize,
    context: usize,
    pages: usize,
    ring_rows: usize,
    /// Each rank's onboarded experts (empty: every layer on the Sparks).
    local: RefCell<Vec<super::onboard::OnboardedExperts<'a>>>,
}

impl<'r, 'a> MuseBackend<'r, 'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(cfg: &'r MuseSparkConfig, ranks: &'r [RankResources<'a>], streams: Vec<*mut c_void>,
        table: &'r TokenEmbedding<'a>, experts: &'r mut dyn SparkExperts, peer: Option<&'r super::peer::MusePeer<'a>>,
        unit_rows: usize, waves: usize, slots: usize, context: usize, pages: usize,
        local: Vec<super::onboard::OnboardedExperts<'a>>) -> Result<Self> {
        ensure!(!ranks.is_empty() && streams.len() == ranks.len() && (ranks.len() == 1) == peer.is_none()
            && (local.is_empty() || local.len() == ranks.len()) && table.vocab() == cfg.vocab_size && table.hidden() == cfg.hidden,
            "Muse backend geometry differs from its ranks, embedding or layers");
        let ring_rows = usize::try_from(cuteafd_loader::placement::families::muse_spark::ring_rows(cfg)?)?;
        Ok(Self { cfg, ranks, streams, table, experts: RefCell::new(experts), peer, library: ranks[0].device.library,
            profile: RefCell::new([0.0; 2]), unit_rows, waves, slots, context, pages, ring_rows,
            local: RefCell::new(local) })
    }

    fn head_rank(&self) -> usize { self.ranks.len() - 1 }

    fn logits(&self, buffer: cuteafd_ffi::CuteafdDeviceBuffer, rows: usize) -> DeviceLogits {
        DeviceLogits { ptr: buffer.ptr.cast_const(), rows, vocab: self.cfg.vocab_size, stride: self.cfg.vocab_size,
            stream: self.streams[self.head_rank()], greedy: None }
    }

    /// One decode/verify step over `sequences` (each from its committed length).
    fn step(&self, sequences: &[(&mut Placement, &[u32])], kind: cuteafd_transport::ExpertV2SourceKind)
        -> Result<DeviceLogits> {
        let rows = sequences.iter().map(|(_, tokens)| tokens.len()).sum::<usize>();
        ensure!((1..=HEAD_ROWS).contains(&rows), "Muse decode step of {rows} rows (at most {HEAD_ROWS})");
        let described = sequences.iter().map(|(placement, tokens)| StepSequence { slot: placement.slot,
            start: placement.len, rows: tokens.len(), pages: &placement.pages }).collect::<Vec<_>>();
        let meta = StepMetadata::build(&described, self.ring_rows, self.slots, self.unit_rows, false)?;
        let wave = WaveStep { segments: sequences.iter().map(|(placement, tokens)| WaveSegment {
            request: placement.slot as u64 + 1, rows: tokens.len() }).collect(), kind };
        let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
        let mut ends = self.peer.map(|peer| (0..2).map(|rank| peer.rank(rank)).collect::<Vec<_>>()).unwrap_or_default();
        let peers = ends.iter_mut().map(|end| end as &mut dyn super::device::PeerExchange).collect();
        let mut experts = self.experts.borrow_mut();
        let logits = forward::step(self.cfg, self.ranks, &self.streams, self.table, &tokens, &meta, false, wave,
            &mut **experts, peers, &mut self.local.borrow_mut(), 0..rows)?;
        Ok(self.logits(logits, rows))
    }
}

impl<'r, 'a> Backend<'a> for MuseBackend<'r, 'a> {
    type Placement = Placement;
    type Draft = NoDraft;
    type History = ();
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = MusePrefix<'r, 'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "muse_spark", vocab_size: self.cfg.vocab_size, max_context: self.context,
            slots: self.slots, prefill_rows: self.unit_rows * self.waves, decode_rows: HEAD_ROWS,
            full_prefill_logits: false }
    }
    /// Every Muse message opens with `<|start|>` (turn boundaries for prefix marks).
    fn message_starts(&self) -> &[&str] { &[cuteafd_api::openai::chat::muse::START] }
    fn library(&self) -> &'a NativeLibrary { self.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>, ()>)> {
        ensure!(args.prefix_partial == Toggle::Off, "Muse restores exact frontiers only");
        let family = MusePrefix::new(self.cfg, self.ranks, &self.streams, self.slots as u64, self.pages,
            args.mark_rule(lanes).slots(super::prefix::mark_bytes(self.cfg, self.ranks.len())?))?;
        let layout = family.layout();
        // The host tier copies the registered record pools and mark arenas.
        let host = args.host_config(layout, self.context, lanes)?.map(|config|
            CudaCopyEngine::registered_owned(self.library, family.snapshot_owners()).map(|copy| (config, copy)))
            .transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let config = PrefixConfig { entries: args.prefix_cache_entries, mark_slots: family.slots(), keep_logits: true,
            min_tokens: args.prefix_cache_min_tokens };
        let cache = PrefixCache::new(layout, config, host)?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, Some((layout.pages * layout.page_rows) as u64),
            layout.pages as u64, family.record_format(), host_bytes);
        Ok((family, cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty(), "Muse Spark media towers are not bound yet (text only)");
        let keys = MediaKeys::new(&tokens, u32::try_from(self.cfg.vocab_size)?, &[])?;
        let media = RequestMedia::new(Vec::new(), self.cfg.hidden, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, Vec::new()))
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia, _cache: &mut EmbeddingCache,
        _snapshot: &Path) -> Result<()> {
        ensure!(prompt.job.media.is_empty() && media.spans().is_empty(), "Muse Spark serves text only");
        Ok(())
    }
    fn coordinates(&self, _tokens: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        ensure!(job.media.is_empty() && media.spans().is_empty(), "Muse Spark serves text only");
        Ok(())
    }
    fn placement(&self, pages: Vec<u32>, slot: i32) -> Placement { Placement { slot: slot as usize, pages, len: 0 } }
    fn bind_request(&self, _placement: &mut Placement, _coordinates: (), _media: RequestMedia, _tokens: &[u32],
        _resume: usize) -> Result<()> { Ok(()) }
    fn history(&self, _placement: &Placement) {}
    fn require_live(&self) -> Result<()> { Ok(()) }
    fn log_table_stats(&self, _phase: &str) {}

    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize, _pool: &RefPagePool)
        -> Result<Option<DeviceLogits>> {
        ensure!(logit_rows <= 1, "Muse prefill scores its last row only");
        let units = crate::shared::prefill_pipeline::prefill_units(tokens.len(), self.unit_rows, self.waves, 64)?
            .into_iter().enumerate().map(|(index, rows)| crate::shared::prefill_pipeline::Unit { index, sequence: 0,
                segments: Vec::new(), rows, layers: 0..self.cfg.layers, slot: index % self.waves, barrier: false })
            .collect::<Vec<_>>();
        let mut ends = self.peer.map(|peer| (0..2).map(|rank| peer.rank(rank)).collect::<Vec<_>>()).unwrap_or_default();
        let peers = ends.iter_mut().map(|end| end as &mut dyn super::device::PeerExchange).collect();
        let mut experts = self.experts.borrow_mut();
        let mut local = self.local.borrow_mut();
        let call = PrefillCall { request: placement.slot as u64 + 1, slot: placement.slot, start: placement.len, tokens,
            pages: &placement.pages, ring_rows: self.ring_rows, slots: self.slots };
        let mut hooks = MusePrefill::new(self.cfg, self.ranks, self.streams.clone(), self.table, &mut **experts, peers,
            call, &mut local, units.len())?;
        crate::shared::prefill_pipeline::run_prefill_units(&mut hooks, &units, self.waves)?;
        let logits = hooks.logits.take();
        drop(hooks);
        placement.len += tokens.len();
        if logit_rows == 0 { return Ok(None); }
        Ok(Some(self.logits(logits.context("Muse prefill scored no last row")?, 1)))
    }

    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool, _pool: &RefPagePool)
        -> Result<Option<DeviceLogits>> {
        ensure!(!speculative, "Muse Spark has no drafter yet");
        self.step(sequences, cuteafd_transport::ExpertV2SourceKind::Decode).map(Some)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        _kind: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.verify_device(sequences, speculative, pool)
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind": "muse_gqa_swa2048_full", "rows": rows, "contexts": contexts})
    }
    fn draft_geometry(&self, _snapshot: &Path, _width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        anyhow::bail!("Muse Spark has no drafter yet")
    }
    fn arm_draft_probe(&self) -> Result<()> { anyhow::bail!("Muse Spark has no drafter yet") }
    fn prefill_draft(&self, _placement: &Placement, _start: usize, _chunk: &[u32], _next: Option<u32>,
        _draft: &mut NoDraft) -> Result<()> { Ok(()) }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Placement, NoDraft>], _timing: &mut scheduler::DraftTiming)
        -> Result<Vec<Vec<u32>>> {
        Ok(sequences.iter().map(|_| Vec::new()).collect())
    }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, NoDraft, ()>], speculative: bool) -> Result<()> {
        ensure!(!speculative, "Muse Spark has no drafter yet");
        for sequence in verified.iter_mut() {
            let kept = sequence.kept.map_or(0, |(rows, _)| rows);
            ensure!(kept <= sequence.rows.len(), "Muse accepted rows exceed the step");
            sequence.placement.len = sequence.start + kept;
        }
        Ok(())
    }
    fn probe_begin(&self, _rows: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
}

#[cfg(test)]
mod tests {
    /// Prefix turn marks key on the checkpoint's `<|start|>` (not Qwen's
    /// `<|im_start|>`), resolved from the served snapshot when present.
    #[test]
    fn message_starts_are_the_checkpoints() {
        let source = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("fn message_starts(&self) -> &[&str] { &[cuteafd_api::openai::chat::muse::START] }"));
        assert_eq!(cuteafd_api::openai::chat::muse::START, "<|start|>");
        let snapshots = std::path::Path::new(
            "/mnt/sparknest/hf-home/hub/models--meta-models--Muse-Spark-Open-816B-A42B/snapshots");
        let Some(Ok(snapshot)) = std::fs::read_dir(snapshots).ok().and_then(|mut entries| entries.next()) else { return };
        assert_eq!(crate::shared::prefix::marker_ids(&snapshot.path(), &["<|start|>"]).unwrap(), vec![200_022]);
    }
}
