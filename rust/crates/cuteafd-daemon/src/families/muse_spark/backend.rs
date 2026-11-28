//! Muse Spark on the shared scheduler (`qwen::scheduler::Backend`), text
//! only, with an optional DFlash2 drafter (`draft::MuseDraft`).
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
//! - Drafter: every step taps the head rank's hidden rows after the tapped
//!   layers; prefill commits the call's tail (its last `TAP_ROWS` rows) as
//!   context, accept commits each sequence's kept rows, and a draft is one
//!   block per sequence from its anchor (`valid_from` bounds the context a
//!   restored prefix does not have).
use super::{engine::RankResources, experts::{SparkExperts, WaveSegment, WaveStep}, forward, prefill::{MusePrefill, PrefillCall},
    prefix::{MusePrefix, Placement}, step::{StepMetadata, StepSequence}};
use crate::shared::{prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::scheduler::{self, Backend}, token_io::{DeviceLogits, TokenEmbedding}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::muse_spark::MuseSparkConfig;
use std::{cell::RefCell, ffi::c_void, path::Path};

/// Pack image/audio features with row offsets across the same sequence order as tokens.
fn step_media<'a>(groups: impl IntoIterator<Item = (Option<&'a RequestMedia>, usize, usize)>)
    -> Result<cuteafd_engine::media::MediaChunk> {
    let mut packed = cuteafd_engine::media::MediaChunk::default();
    let mut offset = 0u32;
    for (media, start, rows) in groups {
        let end = start.checked_add(rows).context("Muse media step extent overflow")?;
        if let Some(media) = media.filter(|media| media.needed(start, end).next().is_some()) {
            let mut chunk = cuteafd_engine::media::MediaChunk::default();
            media.write_chunk(start, end, &mut chunk)?;
            for index in chunk.indices {
                packed.indices.push(index.checked_add(offset).context("Muse media row overflow")?);
            }
            packed.features.extend(chunk.features);
        }
        offset = offset.checked_add(u32::try_from(rows)?).context("Muse packed step extent overflow")?;
    }
    Ok(packed)
}

/// The committed length after a verify step of `rows` from `len`: a plain
/// step commits every row (callers that never accept: probe scoring, cold
/// steps); a speculative one commits nothing until `accept`.
fn after_verify(len: usize, rows: usize, speculative: bool) -> usize { if speculative { len } else { len + rows } }

/// The length `accept` publishes: the step's start plus its kept rows (none
/// when a finished request keeps no state).
fn committed(start: usize, rows: usize, kept: Option<usize>) -> Result<usize> {
    let kept = kept.unwrap_or(0);
    ensure!(kept <= rows, "Muse accepted rows exceed the step");
    Ok(start + kept)
}

/// A drafted verify reaches the Sparks as MTP verify rows (each request's
/// rows stay under the BF16 cutoff: lossless); a plain step as decode.
pub(super) fn verify_kind(speculative: bool) -> cuteafd_transport::ExpertV2SourceKind {
    if speculative { cuteafd_transport::ExpertV2SourceKind::MtpVerify } else { cuteafd_transport::ExpertV2SourceKind::Decode }
}


/// A sequence's drafter state: the next block's anchor, whether its context
/// is current (prefilled or accepted through the frontier), and the first
/// position the drafter's context holds for it.
#[derive(Debug, Default)]
pub(crate) struct DraftState {
    anchor: Option<u32>,
    ready: bool,
    valid_from: usize,
    confidence: Vec<f32>,
}
impl scheduler::DraftState for DraftState {
    fn pending_rows(&self) -> usize { usize::from(self.ready && self.anchor.is_some()) }
    fn close(&mut self, token: u32) { self.anchor = Some(token); }
    /// A restored prefix has no drafter context: it starts cold at the frontier.
    fn restored(&mut self, frontier: usize) {
        self.ready = false;
        self.valid_from = frontier;
    }
}

pub(crate) struct MuseBackend<'r, 'a> {
    pub(crate) cfg: &'r MuseSparkConfig,
    pub(crate) ranks: &'r [RankResources<'a>],
    pub(crate) streams: Vec<*mut c_void>,
    table: &'r TokenEmbedding<'a>,
    experts: RefCell<&'r mut dyn SparkExperts>,
    peer: Option<&'r super::peer::MusePeer<'a>>,
    library: &'a NativeLibrary,
    profile: RefCell<[f64; 2]>,
    graphs: Vec<super::graphs::Graphs<'a>>,
    graph_enabled: std::cell::Cell<bool>,
    graph_cycles: RefCell<Option<crate::shared::decode_graph::device::EventCycles<'a>>>,
    graph_warming: std::cell::Cell<bool>,
    media_binding: super::media::Binding,
    /// Prefill units: rows and waves in flight; the sequence slots, context, pool pages.
    pub(crate) unit_rows: usize,
    packed_row_cap: Option<usize>,
    pub(crate) packed_fallback_reason: String,
    waves: usize,
    slots: usize,
    decode_rows: usize,
    context: usize,
    host_prefix_bytes: u64,
    pages: usize,
    ring_rows: usize,
    /// Each rank's onboarded experts (empty: every layer on the Sparks).
    local: RefCell<Vec<super::onboard::OnboardedExperts<'a>>>,
    /// The DFlash2 drafter on the head rank's stream (`--draft`).
    draft: Option<super::draft::MuseDraft<'a>>,
}

impl<'r, 'a> MuseBackend<'r, 'a> {
    fn prefill_with(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize, scoring: Option<crate::shared::probe::ScoringRows<'_>>)
        -> Result<Option<crate::shared::probe::ScoreLogits>> {
        crate::shared::experts::rtx::pair::submit(|| self.require_live(), || self.shutdown(), || {
            ensure!(scoring.is_some() || logit_rows <= 1, "Muse multi-row prefill requires a row consumer");
            ensure!(logit_rows <= tokens.len(), "Muse prefill logit rows exceed tokens");

            let units = crate::shared::prefill_pipeline::prefill_units(tokens.len(), self.unit_rows, self.waves, 64)?
                .into_iter().enumerate().map(|(index, rows)| crate::shared::prefill_pipeline::Unit { index, sequence: 0,
                    segments: Vec::new(), rows, layers: 0..self.cfg.layers, slot: index % self.waves, barrier: false })
                .collect::<Vec<_>>();
            let mut ends = self.peer.map(|peer| (0..2).map(|rank| peer.rank(rank)).collect::<Vec<_>>()).unwrap_or_default();
            let peers = ends.iter_mut().map(|end| end as &mut dyn super::device::PeerExchange).collect();
            let mut experts = self.experts.borrow_mut();
            let mut local = self.local.borrow_mut();
            let call = PrefillCall { packed: Vec::new(), request: placement.slot as u64 + 1, slot: placement.slot, start: placement.len, tokens,
                pages: &placement.pages, ring_rows: self.ring_rows, slots: self.slots, media: placement.media.as_ref(), logit_rows, scoring };
            let mut hooks = MusePrefill::new(self.cfg, self.ranks, self.streams.clone(), self.table, &mut **experts, peers,
                call, &mut local, self.draft.as_ref(), units.len())?;
            crate::shared::prefill_pipeline::run_prefill_units(&mut hooks, &units, self.waves)?;
            let logits = hooks.logits.take();
            drop(hooks);
            placement.len += tokens.len();
            if logit_rows == 0 { return Ok(None); }
            Ok(Some(if scoring.is_some() { crate::shared::probe::ScoreLogits::Recorded }
                else { crate::shared::probe::ScoreLogits::Device(self.logits(
                    logits.context("Muse prefill scored no rows")?, logit_rows)) }))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(cfg: &'r MuseSparkConfig, ranks: &'r [RankResources<'a>], streams: Vec<*mut c_void>,
        table: &'r TokenEmbedding<'a>, experts: &'r mut dyn SparkExperts, peer: Option<&'r super::peer::MusePeer<'a>>,
        unit_rows: usize, waves: usize, slots: usize, context: usize, pages: usize,
        local: Vec<super::onboard::OnboardedExperts<'a>>, draft: Option<super::draft::MuseDraft<'a>>,
        media_binding: super::media::Binding, packing: crate::shared::prefill_packing::PackingMode, host_prefix_bytes: u64) -> Result<Self> {
        ensure!(draft.as_ref().is_none_or(|draft| streams.last() == Some(&draft.stream)),
            "Muse drafter must run on the head rank's stream");
        ensure!(!ranks.is_empty() && streams.len() == ranks.len() && (ranks.len() == 1) == peer.is_none()
            && (local.is_empty() || local.len() == ranks.len()) && table.vocab() == cfg.vocab_size && table.hidden() == cfg.hidden,
            "Muse backend geometry differs from its ranks, embedding or layers");
        let decode_rows = ranks[0].decode_rows;
        ensure!(decode_rows > 0 && ranks.iter().all(|rank| rank.decode_rows == decode_rows),
            "Muse backend rank decode_rows differ");
        let ring_rows = usize::try_from(cuteafd_loader::placement::families::muse_spark::ring_rows(cfg)?)?;
        let (packed_row_cap, packed_fallback_reason) = if packing == crate::shared::prefill_packing::PackingMode::Exact {
            let mut cap = unit_rows;
            for rank in ranks { cap = cap.min(rank.programs.packed_cap()).min(rank.core.packed_cap()); }
            let mut reasons = Vec::new();
            if cap == 0 { reasons.push("coordinator program has no exact_row_cap attestation"); }
            // As in GLM, grouping/capacity does not attest native EXL3 arithmetic.
            if !local.is_empty() { reasons.push("onboarded EXL3 TP2/whole backend has no arithmetic-route attestation"); }
            reasons.push("Spark EXL3 backend has no arithmetic-route attestation");
            let reason = reasons.join("; ");
            tracing::info!(reason = %reason, "exact Muse packed prefill disabled; requests run alone");
            (Some(0), reason)
        } else { (None, String::new()) };
        Ok(Self { cfg, ranks, streams, table, experts: RefCell::new(experts), peer, library: ranks[0].device.library,
            profile: RefCell::new([0.0; 2]), graphs: Vec::new(), graph_cycles: RefCell::new(None), graph_enabled: std::cell::Cell::new(false), graph_warming: std::cell::Cell::new(false), media_binding, unit_rows, packed_row_cap, packed_fallback_reason, waves, slots, decode_rows, context, host_prefix_bytes, pages, ring_rows,
            local: RefCell::new(local), draft })
    }

    pub(crate) fn configure_graphs(&mut self, enabled: bool, onboard_layers: usize) -> Result<()> {
        if enabled {
            for ((local, rank), &stream) in self.local.borrow_mut().iter_mut().zip(self.ranks).zip(&self.streams) {
                rank.device.run(|| local.epilogue.initialize_identity(stream))?;
            }
            self.graphs = self.ranks.iter().enumerate().map(|(rank, resources)|
                super::graphs::Graphs::new(resources.device, rank, self.ranks.len(), onboard_layers)).collect::<Result<_>>()?;
        }
        if enabled {
            let streams = self.ranks.iter().zip(&self.streams).map(|(r, &s)| (r.device.id,s)).collect::<Vec<_>>();
            *self.graph_cycles.borrow_mut() = Some(crate::shared::decode_graph::device::EventCycles::new(self.library, &streams)?);
        }
        self.graph_enabled.set(enabled);
        if !enabled { return Ok(()); }
        // Every Muse segment is rank-local: `GRAPH_PREWARM=off` leaves them to
        // first-use capture (#2303).
        let Some(_startup) = crate::shared::decode_graph::sites::startup(
            super::graph_sites::spans("muse_spark.segment", false)) else { return Ok(()) };
        let Some(layout) = cuteafd_loader::serving_capacity::muse_graphs::layout(self.ranks.len(), onboard_layers) else {
            tracing::warn!(ranks = self.ranks.len(), onboard_layers,
                "Muse layout has no graph trace; bootstrap is empty and unpinned");
            return Ok(());
        };
        let bases = (0..self.ranks.len()).map(|rank|
            cuteafd_loader::serving_capacity::muse_graphs::policy(self.ranks.len(), onboard_layers, rank, true)
                .map(|policy| policy.base_bytes)).collect::<Result<Vec<_>, _>>()?;
        // Before the scheduler exists, page 0 and ring slot 0 are scratch. No
        // prefix, request or drafter context owns them; live admission overwrites
        // every visible row before reading it. The table pitch is fixed already.
        self.graph_warming.set(true);
        let warmed = (|| -> Result<()> {
            for entry in cuteafd_loader::serving_capacity::muse_graphs::selected(layout, &bases) {
                if entry.rows > self.unit_rows || entry.rows > self.decode_rows { continue; }
                let tokens = vec![1; entry.rows];
                let mut placement = Placement { slot: 0, pages: vec![0], len: 0, media: None };
                // Initialize shape-specific lazy native state before observing graph growth.
                self.graph_enabled.set(false);
                let touched = self.step(&[(&mut placement, &tokens)], cuteafd_transport::ExpertV2SourceKind::Decode);
                self.graph_enabled.set(true);
                touched?;
                self.diagnostic_sync()?;
                for rank in self.ranks {
                    crate::shared::decode_graph::device::observe_device(self.library, rank.device.id)?;
                }
                if entry.pin { for bank in &self.graphs { bank.pin_rows(entry.rows); } }
                self.step(&[(&mut placement, &tokens)], cuteafd_transport::ExpertV2SourceKind::Decode)?;
                self.diagnostic_sync()?;
            }
            Ok(())
        })();
        self.graph_warming.set(false);
        warmed
    }

    fn begin_graph_cycle(&self) -> Result<()> {
        if !self.graph_enabled.get() { return Ok(()) }
        if let Some(cycles) = self.graph_cycles.borrow_mut().as_mut() {
            // SAFETY: rank graph operands are persistent engine storage.
            unsafe { cycles.begin_persistent(|visit| { for bank in &self.graphs { bank.visit(visit)?; } Ok(()) })?; }
        }
        Ok(())
    }

    fn graph_completed(&self) -> Result<()> {
        if !self.graph_enabled.get() { return Ok(()) }
        let mut cycles = self.graph_cycles.borrow_mut();
        let Some(cycles) = cycles.as_mut() else { return Ok(()) };
        for bank in &self.graphs { bank.tick(); }
        // SAFETY: all rank graph work for this step has been submitted.
        unsafe { cycles.complete()?; }
        if self.graph_warming.get() {
            // Startup captures must be measured before the next geometry.
            let wait = std::time::Instant::now();
            self.diagnostic_sync()?;
            cycles.record_drain(wait.elapsed());
            cycles.poll()?;
        }
        Ok(())
    }

    pub(super) fn diagnostic_graph_width(&self) -> Option<usize> {
        self.graph_enabled.get().then(|| self.draft.as_ref().map_or(0, |draft| draft.block - 1))
    }

    pub(super) fn diagnostic_require_graph_rows(&self, rows: usize) -> Result<()> {
        ensure!(self.graph_enabled.get() && !self.graphs.is_empty()
            && self.graphs.iter().all(|bank| bank.retains_rows(rows)),
            "Muse graph oracle requires retained {rows}-row graphs on every rank; increase GPU_GRAPH_RESERVE");
        Ok(())
    }

    fn head_rank(&self) -> usize { self.ranks.len() - 1 }

    pub(super) fn diagnostic_sync(&self) -> Result<()> {
        for (rank, &stream) in self.ranks.iter().zip(&self.streams) {
            // SAFETY: the backend retains these streams and all their producers.
            rank.device.run(|| unsafe { self.library.cuda_stream_synchronize(stream) })?;
        }
        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        let result = super::engine::shutdown(&mut **self.experts.borrow_mut(), self.peer, self.ranks, &self.streams);
        // This backend owns queued local/draft storage; failed drainage
        // cannot return and allow those fields to be destroyed.
        crate::shared::decode_graph::fatal_drain(result, "Muse backend terminal teardown");
        Ok(())
    }

    /// The served verify with graphs bypassed: probe scoring, and the eager
    /// control arm of the graph oracles. Serving never takes this path.
    pub(super) fn verify_eager(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: cuteafd_transport::ExpertV2SourceKind) -> Result<Option<DeviceLogits>> {
        ensure!(!speculative || self.draft.is_some(), "Muse speculative verify without a drafter");
        let enabled = self.graph_enabled.replace(false);
        let result = self.verify_rows(sequences, speculative, kind);
        self.graph_enabled.set(enabled);
        result
    }

    fn verify_rows(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: cuteafd_transport::ExpertV2SourceKind) -> Result<Option<DeviceLogits>> {
        let logits = self.step(sequences, kind)?;
        for (placement, tokens) in sequences.iter_mut() { placement.len = after_verify(placement.len, tokens.len(), speculative); }
        Ok(Some(logits))
    }

    fn logits(&self, buffer: cuteafd_ffi::CuteafdDeviceBuffer, rows: usize) -> DeviceLogits {
        DeviceLogits { ptr: buffer.ptr.cast_const(), rows, vocab: self.cfg.vocab_size, stride: self.cfg.vocab_size,
            stream: self.streams[self.head_rank()], greedy: None }
    }

    /// One decode/verify step over `sequences` (each from its committed length).
    fn step(&self, sequences: &[(&mut Placement, &[u32])], kind: cuteafd_transport::ExpertV2SourceKind)
        -> Result<DeviceLogits> {
        crate::shared::experts::rtx::pair::submit(|| self.require_live(), || self.shutdown(), || {
            let rows = sequences.iter().map(|(_, tokens)| tokens.len()).sum::<usize>();
            ensure!((1..=self.decode_rows).contains(&rows), "Muse decode step of {rows} rows (at most {})", self.decode_rows);
            let described = sequences.iter().map(|(placement, tokens)| StepSequence { slot: placement.slot,
                start: placement.len, rows: tokens.len(), pages: &placement.pages }).collect::<Vec<_>>();
            let meta = StepMetadata::build(&described, self.ring_rows, self.slots, self.unit_rows, false)?
                .with_stride(self.ranks[0].step.tables.stride)?;
            let wave = WaveStep { segments: sequences.iter().map(|(placement, tokens)| WaveSegment {
                request: placement.slot as u64 + 1, rows: tokens.len() }).collect(), kind };
            let media = step_media(sequences.iter().map(|(placement, tokens)|
                (placement.media.as_ref(), placement.len, tokens.len())))?;
            let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
            let mut ends = self.peer.map(|peer| (0..2).map(|rank| peer.rank(rank)).collect::<Vec<_>>()).unwrap_or_default();
            let peers = ends.iter_mut().map(|end| end as &mut dyn super::device::PeerExchange).collect();
            let mut experts = self.experts.borrow_mut();
            let cycle = if self.graph_warming.get() {
                crate::shared::decode_graph::device::Cycle::Other
            } else if kind == cuteafd_transport::ExpertV2SourceKind::MtpVerify {
                crate::shared::decode_graph::device::Cycle::Verify
            } else { crate::shared::decode_graph::device::Cycle::Decode };
            for bank in &self.graphs { bank.cycle(cycle); }
            self.begin_graph_cycle()?;
            let logits = forward::step(self.cfg, self.ranks, &self.streams, self.table, &tokens, &media, &meta, false, wave,
                &mut **experts, peers, &mut self.local.borrow_mut(), self.draft.as_ref(),
                0..rows,
                if self.graph_enabled.get() { &self.graphs } else { &[] })?;
            self.graph_completed()?;
            Ok(self.logits(logits, rows))
        })
    }
}

impl<'r, 'a> Backend<'a> for MuseBackend<'r, 'a> {
    type Placement = Placement;
    type Draft = DraftState;
    type History = ();
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = MusePrefix<'r, 'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "muse_spark", vocab_size: self.cfg.vocab_size, max_context: self.context,
            slots: self.slots, prefill_rows: self.unit_rows * self.waves, decode_rows: self.decode_rows }
    }
    fn scoring_verify_capacity(&self) -> usize {
        scoring_verify_capacity(self.decode_rows)
    }
    /// Every Muse message opens with `<|start|>` (turn boundaries for prefix marks).
    fn message_starts(&self) -> &[&str] { &[cuteafd_api::openai::chat::muse::START] }
    fn library(&self) -> &'a NativeLibrary { self.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>, ()>)> {
        ensure!(args.prefix_partial == Toggle::Off, "Muse restores exact frontiers only");
        let family = MusePrefix::new(self.cfg, self.ranks, &self.streams, self.slots as u64, self.pages,
            args.mark_rule(lanes).slots(super::prefix::mark_bytes_with_format(self.cfg, self.ranks.len(), self.ranks[0].kv.format)?))?;
        let layout = family.layout();
        // The host tier copies the registered record pools and mark arenas.
        let mut admitted = args.clone();
        admitted.host_cache_bytes = crate::shared::prefix::budget::HostBudget::Bytes(self.host_prefix_bytes);
        let host = admitted.resolve_host_config(layout, self.context, lanes)?.map(|config|
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
        self.media_binding.prepare(job, tokens, self.cfg.vocab_size, self.cfg.hidden, self.context)
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia, cache: &mut EmbeddingCache,
        snapshot: &Path) -> Result<()> {
        self.media_binding.coordinates(&prompt.tokens, media)?;
        super::media::probe_features(prompt, media, cache, snapshot)
    }
    fn coordinates(&self, tokens: &[u32], _job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        self.media_binding.coordinates(tokens, media)
    }
    fn placement(&self, pages: Vec<u32>, slot: i32) -> Placement {
        Placement { slot: slot as usize, pages, len: 0, media: None }
    }
    fn bind_request(&self, placement: &mut Placement, _coordinates: (), media: RequestMedia, tokens: &[u32],
        resume: usize) -> Result<()> {
        self.media_binding.coordinates(tokens, &media)?;
        ensure!(resume <= tokens.len() && placement.len == resume, "Muse media bind frontier differs");
        placement.media = Some(media);
        Ok(())
    }
    fn history(&self, _placement: &Placement) {}
    fn require_live(&self) -> Result<()> {
        self.experts.borrow().require_live()?;
        self.peer.map_or(Ok(()), |peer| peer.exchange().require_live())
    }
    fn log_table_stats(&self, _phase: &str) {}

    fn packed_limits(&self) -> Option<crate::shared::prefill_packing::Limits> {
        Some(crate::shared::prefill_packing::Limits {
            // Exact mode is narrowed by the loaded coordinator/expert attestation.
            rows: self.unit_rows, chunk_rows: self.unit_rows, max_context: self.context,
            taps: self.draft.as_ref().map(|_| crate::shared::speculation::dflash::TAP_ROWS),
        })
    }
    fn coordinator_packed_cap(&self) -> Result<Option<usize>> {
        Ok(self.packed_row_cap)
    }
    fn prefill_packed(&self, sequences: &mut [scheduler::PrefillSeq<'_, Placement, DraftState>],
        _pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> crate::shared::prefill_packing::MemberOutcome)
        -> Result<Vec<Result<()>>> {
        use crate::shared::prefill_packing;
        crate::shared::experts::rtx::pair::submit(|| self.require_live(), || self.shutdown(), || {
            let limits = self.checked_packed_limits()?.context("Muse packing limits")?;
            let chunks = sequences.iter().map(|s| (s.placement.len, s.tokens.len())).collect::<Vec<_>>();
            let segments = prefill_packing::plan(&chunks, &limits)?;
            let mut tokens = Vec::new();
            let mut packed = Vec::new();
            for (sequence, &segment) in sequences.iter().zip(&segments) {
                ensure!(sequence.start == sequence.placement.len && sequence.placement.media.as_ref()
                    .is_none_or(|media| media.spans().is_empty()), "Muse packed member frontier/media");
                tokens.extend_from_slice(sequence.tokens);
                packed.push((super::step::StepSequence { slot: sequence.placement.slot, start: segment.start,
                    rows: segment.rows, pages: &sequence.placement.pages }, segment));
            }
            let units = vec![crate::shared::prefill_pipeline::Unit { index: 0, sequence: 0, segments: Vec::new(),
                rows: 0..tokens.len(), layers: 0..self.cfg.layers, slot: 0, barrier: false }];
            let mut ends = self.peer.map(|peer| (0..2).map(|rank| peer.rank(rank)).collect::<Vec<_>>()).unwrap_or_default();
            let peers = ends.iter_mut().map(|end| end as &mut dyn super::device::PeerExchange).collect();
            let mut experts = self.experts.borrow_mut();
            let mut local = self.local.borrow_mut();
            let call = PrefillCall { packed, request: 0, slot: 0, start: 0, tokens: &tokens, pages: &[],
                ring_rows: self.ring_rows, slots: self.slots, logit_rows: 0, media: None, scoring: None };
            let mut hooks = MusePrefill::new(self.cfg, self.ranks, self.streams.clone(), self.table, &mut **experts,
                peers, call, &mut local, self.draft.as_ref(), 1)?;
            // Eager launches only: changing segment geometry never captures a graph.
            crate::shared::prefill_pipeline::run_prefill_units(&mut hooks, &units, self.waves)?;
            drop(hooks);
            for (sequence, segment) in sequences.iter_mut().zip(&segments) {
                sequence.placement.len = segment.start + segment.rows;
                if sequence.neural {
                    let draft = self.draft.as_ref().context("Muse packed drafter not admitted")?;
                    draft.update_rows(sequence.placement.slot, segment.tap_offset,
                        segment.start + segment.rows - segment.tap_rows, segment.tap_rows)?;
                    if !sequence.draft.ready && sequence.draft.valid_from > segment.start { sequence.draft.valid_from = segment.start; }
                    if segment.start == 0 { sequence.draft.valid_from = 0; }
                    sequence.draft.ready = true;
                    sequence.draft.anchor = sequence.next;
                }
            }
            prefill_packing::finish_members(segments.len(), |i| {
                let rank = &self.ranks[self.head_rank()];
                let row = segments[i].end_row() - 1;
                let logits = rank.device.run(|| forward::head(self.cfg, rank, &rank.lane.views,
                    row..row + 1, self.streams[self.head_rank()]))?;
                Ok(self.logits(logits, 1))
            }, on_logits)
        })
    }
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize, _pool: &RefPagePool)
        -> Result<Option<DeviceLogits>> {
        match self.prefill_with(placement, tokens, logit_rows, None)? {
            Some(crate::shared::probe::ScoreLogits::Device(logits)) => Ok(Some(logits)),
            None => Ok(None),
            Some(_) => unreachable!("ordinary Muse prefill returns device logits"),
        }
    }
    fn streamed_prefill_scoring(&self) -> bool { true }
    fn prefill_scoring(&self, placement: &mut Placement, tokens: &[u32], logit_rows: usize,
        position: usize, probe: &crate::shared::probe::ProbeRef, _pool: &RefPagePool)
        -> Result<Option<crate::shared::probe::ScoreLogits>> {
        self.prefill_with(placement, tokens, logit_rows,
            (logit_rows > 1).then_some(crate::shared::probe::ScoringRows { consumer: probe, position }))
    }

    /// Scores every row of every sequence (an anchor and its drafts) from its
    /// committed length. Speculative rows commit through `accept` (the kept
    /// rows); a plain step commits all of its rows here, as the shared
    /// contract requires of the non-accepting callers (probe scoring, cold steps).
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool, _pool: &RefPagePool)
        -> Result<Option<DeviceLogits>> {
        ensure!(!speculative || self.draft.is_some(), "Muse speculative verify without a drafter");
        self.verify_rows(sequences, speculative, verify_kind(speculative))
    }
    /// Probe scoring rows are prefill-shaped teacher-forced rows.
    fn score_device(&self, sequences: &mut [(&mut Placement, &[u32])], _pool: &RefPagePool)
        -> Result<Option<DeviceLogits>> {
        self.verify_eager(sequences, false, cuteafd_transport::ExpertV2SourceKind::Prefill)
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> crate::shared::speculation_trace::Deferred {
        let contexts = contexts.to_vec();
        crate::shared::speculation_trace::Deferred::new(move || serde_json::json!({"kind": "muse_gqa_swa2048_full", "rows": rows, "contexts": contexts}))
    }
    /// The chunk's tail was tapped during its prefill call: commit it as context.
    fn prefill_draft(&self, placement: &Placement, start: usize, chunk: &[u32], next: Option<u32>,
        seq: &mut DraftState) -> Result<()> {
        crate::shared::experts::rtx::pair::submit(|| self.require_live(), || self.shutdown(), || {
            let draft = self.draft.as_ref().context("Muse drafter was not admitted")?;
            ensure!(!chunk.is_empty() && placement.len == start + chunk.len(),
                "Muse draft prefill differs from the committed target frontier");
            let rows = chunk.len().min(crate::shared::speculation::dflash::TAP_ROWS);
            draft.update(placement.slot, start + chunk.len() - rows, rows)?;
            if !seq.ready && seq.valid_from > start { seq.valid_from = start; }
            if start == 0 { seq.valid_from = 0; }
            seq.ready = true;
            seq.anchor = next;
            Ok(())
        })
    }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Placement, DraftState>], timing: &mut scheduler::DraftTiming, _selector: &mut crate::shared::token_io::TokenSelector<'_>)
        -> Result<Vec<Vec<u32>>> {
        crate::shared::experts::rtx::pair::submit(|| self.require_live(), || self.shutdown(), || {
            let draft = self.draft.as_ref().context("Muse drafter was not admitted")?;
            let started = std::time::Instant::now();
            let mut output = vec![Vec::new(); sequences.len()];
            for seq in sequences.iter_mut() { seq.seq.confidence.clear(); }
            let selected = sequences.iter().enumerate().filter(|(_, seq)| seq.depth > 0).map(|(index, seq)| {
                ensure!(seq.seq.ready && seq.depth < draft.block, "Muse draft continuation is cold or too wide");
                Ok((index, crate::shared::speculation::dflash::DraftSeq { slot: seq.placement.slot,
                    anchor: seq.seq.anchor.context("Muse draft anchor absent")?, position: seq.placement.len,
                    valid_from: seq.seq.valid_from }))
        }).collect::<Result<Vec<_>>>()?;
        let last = self.ranks.len() - 1;
        for batch in selected.chunks(draft.drafter.max_batch_sequences()) {
            let rows = batch.iter().map(|(_, seq)| *seq).collect::<Vec<_>>();
            let proposals = draft.draft(self.cfg, &self.ranks[last], self.table, &rows)?;
            ensure!(proposals.len() == batch.len(), "Muse draft output count differs from its batch");
            for ((index, _), proposal) in batch.iter().zip(proposals) {
                sequences[*index].seq.confidence = proposal.confidence;
                output[*index] = proposal.tokens.into_iter().take(sequences[*index].depth).collect();
            }
            timing.steps += 1;
        }
        timing.seconds += started.elapsed().as_secs_f64();
        Ok(output)
        })
    }
    fn draft_head_kind(&self) -> cuteafd_core::HeadScore {
        self.draft.as_ref().map_or(cuteafd_core::HeadScore::None, |d| d.drafter.score_source())
    }
    fn configure_draft_scores(&self, traced: bool, scored: bool) {
        if let Some(d) = &self.draft { d.drafter.set_selector_readback(traced, scored); }
    }
    fn draft_head<'d>(&self, draft: &'d DraftState) -> Option<&'d [f32]> { Some(&draft.confidence) }

    /// The frontier advances by the kept rows; a drafting sequence's kept rows
    /// (tapped by the verify step at its offset) become drafter context.
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, DraftState, ()>], speculative: bool) -> Result<()> {
        crate::shared::experts::rtx::pair::submit(|| self.require_live(), || self.shutdown(), || {
            ensure!(!speculative || self.draft.is_some(), "Muse speculative accept without a drafter");
            for sequence in verified.iter_mut() {
                sequence.placement.len = committed(sequence.start, sequence.rows.len(), sequence.kept.map(|(rows, _)| rows))?;
            }
            if let Some(draft) = &self.draft {
                for sequence in verified.iter_mut() {
                    let Some((keep, after)) = sequence.neural_kept() else { continue };
                    ensure!(keep > 0 && sequence.placement.len == sequence.start + keep,
                        "Muse draft accept differs from the committed target state");
                    draft.update_rows(sequence.placement.slot, sequence.first_row, sequence.start, keep)?;
                    sequence.seq.ready = true;
                    sequence.seq.anchor = Some(after);
                }
            }
            Ok(())
        })
    }
}

fn scoring_verify_capacity(decode_rows: usize) -> usize {
    decode_rows.min(cuteafd_loader::families::muse_spark::cache::SWA_RING_STEP as usize)
}

#[cfg(test)]
mod tests {
    #[test]
    fn wide_decode_rows_chunks_muse_scoring_within_swa_ring_step() -> anyhow::Result<()> {
        use crate::shared::probe::{score_plan, ScorePath};
        let capacity = super::scoring_verify_capacity(128);
        assert_eq!(capacity, 64);
        assert_eq!(super::scoring_verify_capacity(32), 32);

        let steps = score_plan(583, 64, 4096, capacity, None, ScorePath::Decode, false)?;
        let decode_steps: Vec<_> = steps.iter().filter(|step| step.path == ScorePath::Decode).collect();
        assert!(!decode_steps.is_empty());
        let pages: Vec<u32> = (0..16).collect();
        let ring_rows = 2048 + cuteafd_loader::families::muse_spark::cache::SWA_RING_STEP as usize;
        for step in decode_steps {
            assert!(step.tokens.len() <= 64, "Muse scoring chunk {} exceeds SWA_RING_STEP", step.tokens.len());
            super::StepMetadata::build(
                &[super::StepSequence { slot: 0, start: step.tokens.start, rows: step.tokens.len(), pages: &pages }],
                ring_rows,
                1,
                128,
                false,
            )?;
        }

        let err = score_plan(583, 64, 4096, capacity, Some(65), ScorePath::Decode, false).unwrap_err();
        assert!(err.to_string().contains("1..=64"), "{err:#}");
        let err = score_plan(583, 64, 4096, capacity, Some(128), ScorePath::Decode, false).unwrap_err();
        assert!(err.to_string().contains("1..=64"), "{err:#}");
        Ok(())
    }
    /// Prefix turn marks key on the checkpoint's `<|start|>` (not Qwen's
    /// `<|im_start|>`), resolved from the served snapshot when present.
    #[test]
    fn message_starts_are_the_checkpoints() {
        assert_eq!(cuteafd_api::openai::chat::muse::START, "<|start|>");
        let snapshots = std::path::PathBuf::from(std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME")
            .expect("CUTEAFD_DEPLOYMENT_TEST_HF_HOME is required"))
            .join("hub/models--meta-models--Muse-Spark-Open-816B-A42B/snapshots");
        let snapshot = std::fs::read_dir(snapshots).expect("installed Muse snapshots")
            .next().expect("installed Muse snapshot").expect("read installed Muse snapshot");
        assert_eq!(crate::shared::prefix::marker_ids(&snapshot.path(), &["<|start|>"]).unwrap(), vec![200_022]);
    }

    use super::*;

    #[test]
    fn packed_member_callback_failure_does_not_abort_its_partners() -> Result<()> {
        use crate::shared::{prefill_packing::finish_members, token_io::DeviceLogits};
        let mut consumed = Vec::new();
        let outcomes = finish_members(3, |_| Ok(DeviceLogits {
            ptr: std::ptr::null(), rows: 1, vocab: 1, stride: 1, stream: std::ptr::null_mut(), greedy: None,
        }), &mut |member, _| {
            consumed.push(member);
            Ok(if member == 1 { Err(anyhow::anyhow!("client cancelled")) } else { Ok(()) })
        })?;
        assert_eq!(consumed, [0, 1, 2]);
        assert!(outcomes[0].is_ok() && outcomes[1].is_err() && outcomes[2].is_ok());
        Ok(())
    }

    #[test]
    fn step_media_packs_partial_image_and_audio_spans_at_sequence_offsets() -> Result<()> {
        use cuteafd_core::{AudioKey, ImageKey, MediaKey, MediaSpan};
        fn ready(key: MediaKey, features: &[u8]) -> Result<RequestMedia> {
            let mut cache = EmbeddingCache::new(64);
            let _reservation = cache.reserve(key, features.len())?;
            let lease = cache.complete(key, std::sync::Arc::from(features))?;
            let mut media = RequestMedia::new(vec![MediaSpan { start: 2, len: 3, key }], 1, 8)?;
            media.attach(lease)?;
            Ok(media)
        }
        let image = ready(ImageKey([1; 32]).into(), &[10, 11, 20, 21, 30, 31])?;
        let audio = ready(AudioKey([1; 32]).into(), &[40, 41, 50, 51, 60, 61])?;
        let packed = step_media([(None, 0, 2), (Some(&image), 3, 3), (Some(&audio), 1, 4)])?;
        assert_eq!(packed.indices, [2, 3, 6, 7, 8]);
        assert_eq!(packed.features, [20, 21, 30, 31, 40, 41, 50, 51, 60, 61]);
        // A restored prefix may retain only span identities outside the current step.
        let cold = RequestMedia::new(image.spans().to_vec(), 1, 8)?;
        assert!(step_media([(Some(&cold), 5, 3)])?.indices.is_empty());
        assert!(step_media([(Some(&cold), 2, 1)]).is_err());
        assert!(step_media([(None, usize::MAX, 1)]).is_err());
        Ok(())
    }

    /// The shared scheduler's decode round, on the host: a step of an anchor
    /// and its drafts is verified, the round walks the rows setting the length
    /// to each committed row and stops at the first mismatch, then accepts.
    /// Drafts longer than one row (the round DFlash2 drives) are not refused,
    /// commit exactly the kept rows, and a plain step's chunks (probe scoring,
    /// cold steps) advance the frontier with no accept between them.
    #[test]
    fn verify_and_accept_follow_the_shared_round() -> anyhow::Result<()> {
        for (drafted, agree) in [(0usize, 0usize), (3, 3), (7, 2), (7, 0)] {
            let start = 1000usize;
            let rows = drafted + 1;
            let speculative = rows > 1;
            assert_eq!(verify_kind(speculative), if speculative { cuteafd_transport::ExpertV2SourceKind::MtpVerify }
                else { cuteafd_transport::ExpertV2SourceKind::Decode });
            let mut len = after_verify(start, rows, speculative);
            assert_eq!(len, if speculative { start } else { start + rows });
            // `decode_round`: rows 0..=j commit; a draft the target disagrees with ends the walk.
            let mut kept = 0;
            for j in 0..rows {
                len = start + j + 1;
                kept = j + 1;
                if j >= agree { break; }
            }
            assert_eq!(committed(start, rows, Some(kept))?, len, "drafted {drafted} agree {agree}");
            assert_eq!(kept, (agree + 1).min(rows));
        }
        assert!(committed(0, 3, Some(4)).is_err());
        assert_eq!(committed(10, 3, None)?, 10);
        // Probe scoring: chunks of a teacher-forced tail, no accept between them.
        let (mut len, mut written) = (500usize, Vec::new());
        for chunk in [64usize, 64, 17] {
            written.push(len);
            len = after_verify(len, chunk, false);
        }
        assert_eq!(written, [500, 564, 628], "each chunk writes past the last");
        Ok(())
    }
}

impl Drop for MuseBackend<'_, '_> {
    fn drop(&mut self) { let _ = self.shutdown(); }
}
