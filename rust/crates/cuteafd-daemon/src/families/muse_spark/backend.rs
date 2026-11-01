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
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs, Toggle},
    qwen::scheduler::{self, Backend}, token_io::{DeviceLogits, TokenEmbedding}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::muse_spark::{lane::HEAD_ROWS, MuseSparkConfig};
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
fn verify_kind(speculative: bool) -> cuteafd_transport::ExpertV2SourceKind {
    if speculative { cuteafd_transport::ExpertV2SourceKind::MtpVerify } else { cuteafd_transport::ExpertV2SourceKind::Decode }
}

/// Probe scoring rows are prefill-shaped teacher-forced rows; cold steps are decode.
fn ungraphed_kind(speculative: bool, kind: scheduler::UngraphedKind) -> cuteafd_transport::ExpertV2SourceKind {
    if speculative { return verify_kind(true); }
    match kind {
        scheduler::UngraphedKind::Scoring => cuteafd_transport::ExpertV2SourceKind::Prefill,
        scheduler::UngraphedKind::Decode => cuteafd_transport::ExpertV2SourceKind::Decode,
    }
}

/// What the draft policy prices a verify row with: every layer's expert
/// slice bytes (onboarded layers per RTX, Spark layers per rank) and the
/// installed groupings (the Sparks' bootstrap reports; the onboarded EXL3
/// executors' are read at use).
pub(crate) struct DraftPricing {
    pub homes: Vec<crate::shared::draft::resource::LayerHome>,
    pub grouping: Vec<cuteafd_transport::worker_selection::InstalledExpertGrouping>,
}

impl DraftPricing {
    /// From the checkpoint's EXL3 headers: an onboarded layer's slice is the
    /// GPU's (TP over the coordinator GPUs), a Spark layer's the rank's.
    pub(crate) fn new(cfg: &MuseSparkConfig, manifest: &cuteafd_loader::V41Exl3Manifest, onboarded: &[usize],
        gpus: usize, world: usize, grouping: Vec<cuteafd_transport::worker_selection::InstalledExpertGrouping>)
        -> Result<Self> {
        use crate::shared::draft::resource::{exl3_slice_bytes, LayerHome};
        let homes = (0..cfg.layers).map(|layer| Ok(if onboarded.contains(&layer) {
            LayerHome::Local { slice_bytes: exl3_slice_bytes(manifest, layer, gpus)? }
        } else { LayerHome::Remote { slice_bytes: exl3_slice_bytes(manifest, layer, world)? } }))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { homes, grouping })
    }
}

/// A sequence's drafter state: the next block's anchor, whether its context
/// is current (prefilled or accepted through the frontier), and the first
/// position the drafter's context holds for it.
#[derive(Debug, Default)]
pub(crate) struct DraftState {
    anchor: Option<u32>,
    ready: bool,
    valid_from: usize,
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
    cfg: &'r MuseSparkConfig,
    ranks: &'r [RankResources<'a>],
    streams: Vec<*mut c_void>,
    table: &'r TokenEmbedding<'a>,
    experts: RefCell<&'r mut dyn SparkExperts>,
    peer: Option<&'r super::peer::MusePeer<'a>>,
    library: &'a NativeLibrary,
    profile: RefCell<[f64; 2]>,
    media_binding: super::media::Binding,
    /// Prefill units: rows and waves in flight; the sequence slots, context, pool pages.
    unit_rows: usize,
    waves: usize,
    slots: usize,
    context: usize,
    pages: usize,
    ring_rows: usize,
    /// Each rank's onboarded experts (empty: every layer on the Sparks).
    local: RefCell<Vec<super::onboard::OnboardedExperts<'a>>>,
    /// The DFlash2 drafter on the head rank's stream (`--draft`).
    draft: Option<super::draft::MuseDraft<'a>>,
    /// The draft policy's expert pricing (`policy_geometry`), present with a drafter.
    pricing: Option<DraftPricing>,
    /// The verify rounds' routes and layer clock on the head rank (with a drafter).
    probe: RefCell<Option<crate::shared::memory::device::DeviceOwner<'a, crate::shared::draft::probe::RoundProbe<'a>>>>,
}

impl<'r, 'a> MuseBackend<'r, 'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(cfg: &'r MuseSparkConfig, ranks: &'r [RankResources<'a>], streams: Vec<*mut c_void>,
        table: &'r TokenEmbedding<'a>, experts: &'r mut dyn SparkExperts, peer: Option<&'r super::peer::MusePeer<'a>>,
        unit_rows: usize, waves: usize, slots: usize, context: usize, pages: usize,
        local: Vec<super::onboard::OnboardedExperts<'a>>, draft: Option<super::draft::MuseDraft<'a>>,
        pricing: Option<DraftPricing>, media_binding: super::media::Binding) -> Result<Self> {
        ensure!(draft.is_some() == pricing.is_some(), "Muse drafter and its policy pricing come together");
        // Every Muse layer is MoE; the probe rings the head rank's route ids per layer.
        let head = ranks.last().context("Muse backend without ranks")?.device;
        let probe = draft.as_ref().map(|_| head.own(|| crate::shared::draft::probe::RoundProbe::new(head.library,
            vec![false; cfg.layers], cfg.topk, HEAD_ROWS))).transpose()?;
        ensure!(draft.as_ref().is_none_or(|draft| streams.last() == Some(&draft.stream)),
            "Muse drafter must run on the head rank's stream");
        ensure!(!ranks.is_empty() && streams.len() == ranks.len() && (ranks.len() == 1) == peer.is_none()
            && (local.is_empty() || local.len() == ranks.len()) && table.vocab() == cfg.vocab_size && table.hidden() == cfg.hidden,
            "Muse backend geometry differs from its ranks, embedding or layers");
        let ring_rows = usize::try_from(cuteafd_loader::placement::families::muse_spark::ring_rows(cfg)?)?;
        Ok(Self { cfg, ranks, streams, table, experts: RefCell::new(experts), peer, library: ranks[0].device.library,
            profile: RefCell::new([0.0; 2]), media_binding, unit_rows, waves, slots, context, pages, ring_rows,
            local: RefCell::new(local), draft, pricing, probe: RefCell::new(probe) })
    }

    fn head_rank(&self) -> usize { self.ranks.len() - 1 }

    fn verify_rows(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: cuteafd_transport::ExpertV2SourceKind) -> Result<Option<DeviceLogits>> {
        let logits = self.step(sequences, kind)?;
        for (placement, tokens) in sequences.iter_mut() { placement.len = after_verify(placement.len, tokens.len(), speculative); }
        Ok(Some(logits))
    }

    /// The shared policy geometry over this placement's layer homes and the
    /// installed groupings at every verify row count up to `HEAD_ROWS`.
    fn policy_geometry(&self, width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        use crate::shared::{draft::resource::{self, LayerHome}, experts::rtx::RtxExpertLayer};
        let pricing = self.pricing.as_ref().context("Muse draft pricing absent")?;
        let local = self.local.borrow();
        resource::geometry_with_grouping(&pricing.homes, self.cfg.experts, self.cfg.topk, HEAD_ROWS, width, 0,
            |layer, rows| match pricing.homes[layer] {
                LayerHome::Local { .. } => {
                    let group = local.first().context("Muse onboarded layer without executors")?.executor.group_rows(rows)?;
                    ensure!(local.iter().all(|rank| rank.executor.group_rows(rows).ok() == Some(group)),
                        "Muse onboarded ranks disagree on grouping");
                    Ok(group)
                }
                _ => resource::worker_group_rows(&pricing.grouping, layer, rows),
            })
    }

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
        let media = step_media(sequences.iter().map(|(placement, tokens)|
            (placement.media.as_ref(), placement.len, tokens.len())))?;
        let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
        let mut ends = self.peer.map(|peer| (0..2).map(|rank| peer.rank(rank)).collect::<Vec<_>>()).unwrap_or_default();
        let peers = ends.iter_mut().map(|end| end as &mut dyn super::device::PeerExchange).collect();
        let mut experts = self.experts.borrow_mut();
        let mut probe = self.probe.borrow_mut();
        let logits = forward::step(self.cfg, self.ranks, &self.streams, self.table, &tokens, &media, &meta, false, wave,
            &mut **experts, peers, &mut self.local.borrow_mut(), self.draft.as_ref(),
            probe.as_mut().map(|owner| owner.get_mut()), 0..rows)?;
        Ok(self.logits(logits, rows))
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
        self.media_binding.prepare(job, tokens, self.cfg.vocab_size, self.cfg.hidden, self.context)
    }
    fn probe_features(&self, prompt: &scheduler::Prompt, media: &mut RequestMedia, _cache: &mut EmbeddingCache,
        _snapshot: &Path) -> Result<()> {
        self.media_binding.coordinates(&prompt.tokens, media)
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
            pages: &placement.pages, ring_rows: self.ring_rows, slots: self.slots, media: placement.media.as_ref() };
        let mut hooks = MusePrefill::new(self.cfg, self.ranks, self.streams.clone(), self.table, &mut **experts, peers,
            call, &mut local, self.draft.as_ref(), units.len())?;
        crate::shared::prefill_pipeline::run_prefill_units(&mut hooks, &units, self.waves)?;
        let logits = hooks.logits.take();
        drop(hooks);
        placement.len += tokens.len();
        if logit_rows == 0 { return Ok(None); }
        Ok(Some(self.logits(logits.context("Muse prefill scored no last row")?, 1)))
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
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: scheduler::UngraphedKind, _pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(!speculative || self.draft.is_some(), "Muse speculative verify without a drafter");
        self.verify_rows(sequences, speculative, ungraphed_kind(speculative, kind))
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind": "muse_gqa_swa2048_full", "rows": rows, "contexts": contexts})
    }
    fn draft_geometry(&self, _snapshot: &Path, width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        let draft = self.draft.as_ref().context("Muse drafter was not admitted")?;
        ensure!(width > 0 && width < draft.block, "Muse draft width {width} beyond the drafter's block {}", draft.block);
        self.policy_geometry(width)
    }
    fn arm_draft_probe(&self) -> Result<()> {
        ensure!(self.probe.borrow().is_some(), "Muse draft policy probe was not admitted");
        Ok(())
    }
    /// The chunk's tail was tapped during its prefill call: commit it as context.
    fn prefill_draft(&self, placement: &Placement, start: usize, chunk: &[u32], next: Option<u32>,
        seq: &mut DraftState) -> Result<()> {
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
    }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Placement, DraftState>], timing: &mut scheduler::DraftTiming)
        -> Result<Vec<Vec<u32>>> {
        let draft = self.draft.as_ref().context("Muse drafter was not admitted")?;
        let started = std::time::Instant::now();
        let mut output = vec![Vec::new(); sequences.len()];
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
                output[*index] = proposal.tokens.into_iter().take(sequences[*index].depth).collect();
            }
            timing.steps += 1;
        }
        timing.seconds += started.elapsed().as_secs_f64();
        Ok(output)
    }
    /// The frontier advances by the kept rows; a drafting sequence's kept rows
    /// (tapped by the verify step at its offset) become drafter context.
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, DraftState, ()>], speculative: bool) -> Result<()> {
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
    }
    fn probe_begin(&self, rows: usize) {
        if let Some(probe) = self.probe.borrow_mut().as_mut() { probe.get_mut().begin(rows); }
    }
    fn probe_end(&self) {
        if let Some(probe) = self.probe.borrow_mut().as_mut() { probe.get_mut().end(); }
    }
    fn probe_finish<T>(&self, observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> {
        let mut probe = self.probe.borrow_mut();
        let owner = probe.as_mut()?;
        let _scope = owner.device.enter().ok()?;
        let (routes, times) = owner.get_mut().finish()?;
        Some(observe(routes, &times))
    }
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

    use super::*;

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
        assert_eq!(ungraphed_kind(false, scheduler::UngraphedKind::Scoring), cuteafd_transport::ExpertV2SourceKind::Prefill);
        assert_eq!(ungraphed_kind(true, scheduler::UngraphedKind::Scoring), cuteafd_transport::ExpertV2SourceKind::MtpVerify);
        // The backend routes both verify entries through the bookkeeping above, never refusing a draft.
        let source = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        assert!(!source.contains("has no drafter yet"));
        let verify = source.split("fn verify_rows(").nth(1).unwrap().split("\n    }").next().unwrap();
        assert!(verify.contains("after_verify(placement.len, tokens.len(), speculative)"));
        Ok(())
    }
}
