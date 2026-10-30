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
use std::{cell::RefCell, path::Path, rc::Rc};
use crate::shared::{fp8_linear::{Fp8Rows, Fp8Scales}, speculation::{
    block_dspark::{self, DsparkDrafter}, dflash::{ContextRow, DraftSeq, TargetHead, TAP_ROWS}}};
use cuteafd_ffi::CuteafdDeviceBuffer;

fn draft_geometry(layers: usize, slots: usize, rows: usize, width: usize) -> cuteafd_core::PolicyGeometry {
    cuteafd_core::PolicyGeometry {
        layers: vec![cuteafd_core::LayerResource { class: None, slice_bytes: 0., group_rows: 1, timed: false }; layers],
        group_rows_by_rows: vec![vec![1; layers]; rows + 1], experts: 0, topk: 0, max_requests: slots,
        max_positions: rows, widths: vec![width], classes: vec![cuteafd_core::ResourceClass::rtx()], regimes: 1,
    }
}

#[derive(Default)]
struct DenseProbe { routes: Option<RoundRoutes> }
impl DenseProbe {
    fn arm(&mut self, layers: usize) { self.routes = Some(RoundRoutes::new(layers, 0)); }
    fn begin(&mut self, rows: usize) {
        if let Some(routes) = &mut self.routes {
            routes.begin(rows);
            for layer in 0..routes.layers() { routes.dense(layer).expect("admitted dense layer"); }
        }
    }
    fn finish<T>(&self, observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> {
        // Dense layers have no expert timings: the complete measured round
        // teaches the shared policy's bucket-local verifier residual instead.
        self.routes.as_ref().map(|routes| observe(routes, &[]))
    }
}

#[derive(Default)]
pub(crate) struct DraftState { anchor: Option<u32> }
impl scheduler::DraftState for DraftState {
    fn pending_rows(&self) -> usize { usize::from(self.anchor.is_some()) }
    fn close(&mut self, token: u32) { self.anchor = Some(token); }
}

pub(crate) struct DenseBackend<'e, 'a> {
    pub engine: &'e RefCell<DenseEngine<'a>>,
    pub library: &'a NativeLibrary,
    drafter: Option<Rc<DsparkDrafter<'a>>>,
    mtp: Option<Rc<super::engine::mtp::MtpCache<'a>>>,
    selector: RefCell<Option<crate::shared::token_io::TokenSelector<'a>>>,
    profile: RefCell<[f64; 2]>,
    pending: RefCell<Option<DenseMark<'a>>>,
    probe: RefCell<DenseProbe>,
    packing: crate::shared::prefill_packing::PackingMode,
}

impl<'e, 'a> DenseBackend<'e, 'a> {
    pub fn new(engine: &'e RefCell<DenseEngine<'a>>, library: &'a NativeLibrary) -> Self {
        Self { engine, library, drafter: None, mtp: None, selector: RefCell::new(None),
            profile: RefCell::new([0.0; 2]), pending: RefCell::new(None),
            probe: RefCell::new(DenseProbe::default()), packing: Default::default() }
    }
}

pub(crate) fn packed_check<'a>(engine: &RefCell<DenseEngine<'a>>, library: &'a NativeLibrary,
    tokens: &[u32], members: usize, mode: crate::shared::prefill_packing::PackingMode,
    windows: Option<&[cuteafd_bench::reference::Window]>, precision: &cuteafd_bench::fidelity_precision::Precision,
    mtp: Option<Rc<super::engine::mtp::MtpCache<'a>>>) -> Result<()> {
    let prompts = crate::shared::qwen::packed_check::prompts(tokens, members)?;
    let mut backend = DenseBackend::new(engine, library);
    backend.packing = mode;
    backend.mtp = mtp;
    let neural = backend.mtp.is_some();
    println!("dense packed check: native MTP {neural}");
    let limits = backend.checked_packed_limits()?.context("dense packed limits")?;
    let mut next_page = 0usize;
    let mut admit = |slot: usize, prompt: &[u32]| -> Result<Placement> {
        let pages = (prompt.len() + 1).div_ceil(64);
        let units = (next_page..next_page + pages).map(u32::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
        next_page += pages;
        engine.borrow_mut().reset_slot(slot)?;
        engine.borrow_mut().bind_slot(slot, &units)?;
        if let Some(mtp) = &backend.mtp {
            mtp.reset_context(slot)?;
            mtp.admit_pages(slot, prompt.len())?;
        }
        Ok(Placement { units, slot, len: 0, state_len: 0 })
    };
    let packed = prompts.iter().enumerate().map(|(i, p)| admit(i, p)).collect::<Result<Vec<_>>>()?;
    let alone = prompts.iter().enumerate().map(|(i, p)| admit(i + members, p)).collect::<Result<Vec<_>>>()?;
    let pool = cuteafd_engine::prefix::RefPagePool::new(engine.borrow().pool_tokens / 64, 64);
    crate::shared::qwen::packed_check::check::<_, DraftState>(library, &prompts, packed, alone, limits, neural,
        mode == crate::shared::prefill_packing::PackingMode::Exact,
        |s| backend.prefill_device(s.placement, s.tokens, 1, &pool)?.context("dense check needs head")?.row_host(library, 0),
        |s, callback| backend.prefill_packed(s, &pool, callback), |p| {
            let mut bytes = super::engine::prefix::packed_state(engine, p)?;
            if let Some(mtp) = &backend.mtp { bytes.extend(mtp.packed_state(p.slot, p.len)?); }
            Ok(bytes)
        })?;
    if mode == crate::shared::prefill_packing::PackingMode::Unrestricted {
        let windows = windows.context("dense packed check needs official windows")?;
        drop(admit);
        next_page = 0;
        let mut admit_reference = |slot: usize, w: &cuteafd_bench::reference::Window| {
            let pages = w.tokens.len().div_ceil(64);
            let units = (next_page..next_page + pages).map(u32::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
            next_page += pages;
            ensure!(next_page <= engine.borrow().pool_tokens / 64, "dense packed reference exhausts admitted pages");
            engine.borrow_mut().reset_slot(slot)?;
            engine.borrow_mut().bind_slot(slot, &units)?;
            if let Some(mtp) = &backend.mtp {
                mtp.reset_context(slot)?;
                mtp.admit_pages(slot, w.tokens.len())?;
            }
            Ok(Placement { units, slot, len: 0, state_len: 0 })
        };
        let placements = windows.iter().enumerate().map(|(i, w)| admit_reference(i, w)).collect::<Result<Vec<_>>>()?;
        let controls = windows.iter().enumerate().map(|(i, w)| admit_reference(i + members, w)).collect::<Result<Vec<_>>>()?;
        crate::shared::qwen::packed_check::reference::<_, DraftState>(library, windows, placements, controls, limits, neural, precision,
            |s| backend.prefill_device(s.placement, s.tokens, 1, &pool)?.context("dense reference needs head")?.row_host(library, 0),
            |s, callback| backend.prefill_packed(s, &pool, callback))?;
    }
    Ok(())
}

impl<'e, 'a> Backend<'a> for DenseBackend<'e, 'a> {
    type Placement = Placement;
    type Draft = DraftState;
    type History = ();
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = DensePrefix<'e, 'a>;

    fn geometry(&self) -> scheduler::Geometry {
        let engine = self.engine.borrow();
        scheduler::Geometry { family: "qwen3_5", vocab_size: engine.cfg.vocab, max_context: engine.cfg.max_context,
            slots: engine.slot_count(), prefill_rows: if self.drafter.is_some() { TAP_ROWS } else { 4096 }, decode_rows: engine.decode_rows(), full_prefill_logits: engine.full_prefill_logits() }
    }
    fn library(&self) -> &'a NativeLibrary { self.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(args.prefix_partial == Toggle::Off, "dense Qwen supports exact prefix frontiers only");
        let family = if let Some(mtp) = &self.mtp { DensePrefix::with_mtp(self.engine, mtp.clone())? }
            else { DensePrefix::with_draft(self.engine, self.drafter.clone())? };
        let layout = family.layout();
        let shape = self.geometry();
        let args = &args.with_scoring(if shape.full_prefill_logits { shape.prefill_rows } else { shape.decode_rows },
            shape.prefill_rows, shape.vocab_size)?;
        let host = args.host_config(layout, shape.max_context, lanes)?.map(|config|
            CudaCopyEngine::registered_retained(self.library, family.snapshot_owners(),
                self.drafter.iter().cloned().map(|draft| draft as Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner + 'a>)
                    .chain(self.mtp.iter().cloned().map(|mtp| mtp as Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner + 'a>))
                    .collect()).map(|copy| (config, copy))).transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let config = PrefixConfig { entries: args.prefix_cache_entries, mark_slots: family.slots(),
            keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
        let cache = PrefixCache::new_with_draft_pages(layout, config, host, family.draft_page_bytes())?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, Some((layout.pages * layout.page_rows) as u64), layout.pages as u64,
            family.record_format(), host_bytes);
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
    fn bind_request(&self, _placement: &mut Placement, _coordinates: (), _media: RequestMedia, _tokens: &[u32], _resume: usize) -> Result<()> { Ok(()) }
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
        let mut output = None;
        // Each tap tile is consumed before the target reuses its residual plane.
        let tile = if self.drafter.is_some() { TAP_ROWS } else { tokens.len() };
        for (index, chunk) in tokens.chunks(tile).enumerate() {
            let first = index * tile;
            let end = first + chunk.len();
            let head_first = tokens.len() - logit_rows;
            ensure!(logit_rows <= chunk.len() || head_first >= end,
                "dense tapped prefill requires bounded scoring rows");
            let last_layer = engine.cfg.attention.len() - 1;
            output = Some(engine.prefill_tapped(placement.slot, chunk,
                if end == tokens.len() { logit_rows } else { 0 }, |layer, hidden| {
                    if let Some(draft) = &self.drafter {
                        draft.tap_streams_at(layer, hidden.ptr, 1, 0, chunk.len(), 0)?;
                    }
                    if layer == last_layer {
                        if let Some(mtp) = &self.mtp { mtp.capture_hidden(hidden, chunk.len())?; }
                    }
                    Ok(())
                })?);
            if let Some(mtp) = &self.mtp {
                mtp.update_captured(&mut engine, placement.slot, placement.len + first, chunk)?;
            }
            if let Some(draft) = &self.drafter {
                let rows = (0..chunk.len()).map(|tap_row| ContextRow { tap_row, slot: placement.slot,
                    position: placement.len + first + tap_row }).collect::<Vec<_>>();
                draft.update(&rows)?;
            }
        }
        let output = output.context("empty dense prefill")?;
        placement.len += tokens.len();
        placement.state_len = placement.len;
        Ok((logit_rows > 0).then_some(output))
    }
    fn coordinator_packed_cap(&self) -> Result<Option<usize>> {
        if self.packing == crate::shared::prefill_packing::PackingMode::Exact {
            Ok(Some(self.engine.borrow().coordinator_packed_cap()?))
        } else { Ok(None) }
    }
    fn packed_limits(&self) -> Option<crate::shared::prefill_packing::Limits> {
        Some(crate::shared::prefill_packing::Limits {
            // The small and prefill FP8 FFNs have different projection arithmetic.
            rows: if self.packing == crate::shared::prefill_packing::PackingMode::Exact { 64 } else { cuteafd_loader::placement::families::qwen3_5::PREFILL_ROWS },
            chunk_rows: self.geometry().prefill_rows, max_context: self.engine.borrow().cfg.max_context,
            taps: self.drafter.as_ref().map(|_| TAP_ROWS),
        })
    }
    fn prefill_packed(&self, sequences: &mut [scheduler::PrefillSeq<'_, Placement, DraftState>],
        _pool: &cuteafd_engine::prefix::RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> crate::shared::prefill_packing::MemberOutcome)
        -> Result<Vec<Result<()>>> {
        ensure!(self.pending.borrow().is_none(), "dense packed prefill during pending verify");
        let limits = self.checked_packed_limits()?.context("dense packing limits")?;
        let mut engine = self.engine.borrow_mut();
        for s in sequences.iter() {
            ensure!(engine.slot_position(s.placement.slot)? == s.start && s.placement.len == s.start,
                "dense packed placement/state frontier differs");
            engine.bind_slot(s.placement.slot, &s.placement.units)?;
        }
        let inputs = sequences.iter().map(|s| (s.placement.slot, s.tokens)).collect::<Vec<_>>();
        let last_layer = engine.cfg.attention.len() - 1;
        let segments = engine.prefill_packed(&inputs, &limits, |layer, hidden, rows| {
            if let Some(draft) = &self.drafter { draft.tap_streams_at(layer, hidden.ptr, 1, 0, rows, 0)?; }
            if layer == last_layer {
                if let Some(mtp) = &self.mtp { mtp.capture_hidden(hidden, rows)?; }
            }
            Ok(())
        })?;
        for (s, segment) in sequences.iter_mut().zip(&segments) {
            s.placement.len = s.start + segment.rows;
            s.placement.state_len = s.placement.len;
        }
        let outcomes = crate::shared::prefill_packing::finish_members(sequences.len(),
            |i| engine.packed_head(segments[i].end_row() - 1), on_logits)?;
        for (s, segment) in sequences.iter_mut().zip(&segments) {
            if let Some(mtp) = &self.mtp {
                mtp.update_captured_at(&mut engine, s.placement.slot, s.start, s.tokens, segment.first)?;
            }
            if let Some(draft) = &self.drafter {
                let rows = (0..segment.tap_rows).map(|r| ContextRow { tap_row: segment.tap_offset + r,
                    slot: s.placement.slot, position: s.start + segment.rows - segment.tap_rows + r }).collect::<Vec<_>>();
                draft.update(&rows)?;
            }
            if s.neural { s.draft.anchor = s.next; }
        }
        Ok(outcomes)
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
        let mtp_layers = [engine.cfg.attention.len() - 1];
        let request = VerifyRequest { rows: &rows,
            tap_layers: if self.mtp.is_some() { &mtp_layers }
                else { self.drafter.as_ref().map_or(&[], |draft| draft.cfg.taps.as_slice()) } };
        let output = if speculative {
            let mark = engine.checkpoint(&slots)?;
            let output = match engine.verify(&mark, request) {
                Ok(output) => {
                    if let Some(draft) = &self.drafter {
                        // SAFETY: enqueue reads on the producer stream before commit invalidates taps.
                        unsafe { draft.capture_target_taps(output.taps, 0, rows.len(), 0)?; }
                    }
                    output.logits
                },
                Err(error) => { engine.restore(&mark)?; return Err(error); }
            };
            *self.pending.borrow_mut() = Some(mark);
            output
        } else {
            let output = engine.decode_device(&slots, request)?;
            if let Some(draft) = &self.drafter {
                // SAFETY: the target retains these taps until its next mutable operation.
                unsafe { draft.capture_target_taps(engine.verify_taps(), 0, rows.len(), 0)?; }
            }
            for (placement, tokens) in sequences {
                placement.len += tokens.len();
                placement.state_len = placement.len;
            }
            output
        };
        Ok(Some(output))
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        kind: scheduler::UngraphedKind, pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Option<DeviceLogits>> {
        let rows = sequences.iter().flat_map(|(placement, tokens)|
            tokens.iter().enumerate().map(|(offset, _)| (placement.slot, placement.len + offset))).collect::<Vec<_>>();
        let output = self.verify_device(sequences, speculative, pool)?;
        if kind == scheduler::UngraphedKind::Scoring {
            ensure!(!speculative, "dense scoring cannot leave speculative state");
            if let Some(mtp) = &self.mtp {
                let mut engine = self.engine.borrow_mut();
                let tap = engine.verify_taps()[0];
                let mut first = 0;
                for (placement, tokens) in sequences {
                    let width = engine.cfg.hidden * 2;
                    let hidden = CuteafdDeviceBuffer { ptr: tap.ptr.cast::<u8>().wrapping_add(first * width).cast_mut().cast(),
                        bytes: tokens.len() * width, device_id: self.library.cuda_get_device()?, ..Default::default() };
                    mtp.update(&mut engine, placement.slot, placement.len - tokens.len(), tokens, hidden)?;
                    first += tokens.len();
                }
            }
            if let Some(draft) = &self.drafter {
                draft.update(&rows.into_iter().enumerate().map(|(tap_row, (slot, position))|
                    ContextRow { tap_row, slot, position }).collect::<Vec<_>>())?;
            }
        }
        Ok(output)
    }
    fn verify_bucket_rows(&self, rows: usize, _speculative: bool, _diagnostic: bool) -> usize { rows }
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value {
        serde_json::json!({"kind":"causal_gqa", "rows":rows, "contexts":contexts})
    }
    fn draft_geometry(&self, _snapshot: &Path, width: usize) -> Result<cuteafd_core::PolicyGeometry> {
        if self.mtp.is_some() {
            ensure!((1..=3).contains(&width), "dense native MTP chain depth outside 1..=3");
            let engine = self.engine.borrow();
            return Ok(draft_geometry(engine.cfg.attention.len(), engine.slot_count(), engine.decode_rows(), width));
        }
        let draft = self.drafter.as_ref().context("dense serving has no drafter")?;
        ensure!(width <= draft.cfg.drafts(), "dense draft width exceeds checkpoint");
        let engine = self.engine.borrow();
        Ok(draft_geometry(engine.cfg.attention.len(), draft.max_sequences, engine.decode_rows(), width))
    }
    fn arm_draft_probe(&self) -> Result<()> {
        self.probe.borrow_mut().arm(self.engine.borrow().cfg.attention.len());
        Ok(())
    }
    fn prefill_draft(&self, _placement: &Placement, _start: usize, _chunk: &[u32], next: Option<u32>, draft: &mut DraftState) -> Result<()> {
        draft.anchor = next;
        Ok(())
    }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Placement, DraftState>], timing: &mut scheduler::DraftTiming)
        -> Result<Vec<Vec<u32>>> {
        if let Some(mtp) = &self.mtp {
            use crate::shared::token_io::{RowSelect, SelectBatch};
            let begin = std::time::Instant::now();
            let mut output = vec![Vec::new(); sequences.len()];
            let mut engine = self.engine.borrow_mut();
            let mut selector = self.selector.borrow_mut();
            let selector = selector.as_mut().context("dense MTP selector was not admitted")?;
            for (index, sequence) in sequences.iter().enumerate().filter(|(_, sequence)| sequence.depth > 0) {
                output[index] = mtp.draft_chain(&mut engine, sequence.placement.slot,
                    sequence.seq.anchor.context("dense MTP has no anchor")?, sequence.placement.len,
                    sequence.depth, |logits, step| {
                        let batch = SelectBatch { rows: vec![RowSelect { sampling: cuteafd_core::TargetSamplingParams::greedy(),
                            position: (sequence.placement.len + step) as u64, mask: None, logprob: false }] };
                        Ok(selector.select(&logits, &batch)?.into_iter().next()
                            .context("dense MTP selector returned no row")??.token)
                    })?;
            }
            timing.steps += 1;
            timing.seconds += begin.elapsed().as_secs_f64();
            return Ok(output);
        }
        let draft = self.drafter.as_ref().context("dense serving has no drafter")?;
        let mut proposed = vec![Vec::new(); sequences.len()];
        let selected = sequences.iter().enumerate().filter(|(_, sequence)| sequence.depth > 0).collect::<Vec<_>>();
        if selected.is_empty() { return Ok(proposed); }
        let rows = selected.iter().map(|(_, sequence)| Ok(DraftSeq {
            slot: sequence.placement.slot, anchor: sequence.seq.anchor.context("dense draft has no anchor")?,
            position: sequence.placement.len, valid_from: 0,
        })).collect::<Result<Vec<_>>>()?;
        let engine = self.engine.borrow();
        let start = std::time::Instant::now();
        let device_id = engine.target_stream().device.id;
        let head = TargetHead::Launch(Box::new(|hidden, logits, rows, stream| {
            ensure!(stream == engine.target_stream().raw, "dense draft head stream differs");
            for first in (0..rows).step_by(64) {
                let count = (rows - first).min(64);
                engine.target_head(CuteafdDeviceBuffer {
                    ptr: hidden.cast::<u8>().wrapping_add(first * engine.cfg.hidden * 2).cast_mut().cast(),
                    bytes: count * engine.cfg.hidden * 2, device_id, ..Default::default()
                }, CuteafdDeviceBuffer {
                    ptr: logits.wrapping_add(first * engine.cfg.vocab).cast(),
                    bytes: count * engine.cfg.vocab * 4, device_id, ..Default::default()
                }, count)?;
            }
            Ok(())
        }));
        let output = draft.draft_device(&rows, engine.token_embedding(), head)?;
        timing.steps += 1;
        timing.seconds += start.elapsed().as_secs_f64();
        for ((index, sequence), output) in selected.into_iter().zip(output) {
            proposed[index] = output.tokens.into_iter().take(sequence.depth).collect();
        }
        Ok(proposed)
    }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, DraftState, ()>], speculative: bool) -> Result<()> {
        if let Some(mtp) = &self.mtp {
            let mut engine = self.engine.borrow_mut();
            let tap = *engine.verify_taps().first().context("dense MTP target hidden tap missing")?;
            let width = engine.cfg.hidden * 2;
            for sequence in verified.iter_mut() {
                let count = if speculative { sequence.kept.map_or(0, |(rows, _)| rows) } else { sequence.rows.len() };
                if count > 0 {
                    let hidden = CuteafdDeviceBuffer { ptr: tap.ptr.cast::<u8>()
                        .wrapping_add(sequence.first_row * width).cast_mut().cast(), bytes: count * width,
                        device_id: self.library.cuda_get_device()?, ..Default::default() };
                    mtp.update(&mut engine, sequence.placement.slot, sequence.start, &sequence.rows[..count], hidden)?;
                }
                if let Some((_, token)) = sequence.kept { sequence.seq.anchor = Some(token); }
            }
        }
        if let Some(draft) = &self.drafter {
            let rows = verified.iter().flat_map(|sequence| {
                let count = if speculative { sequence.kept.map_or(0, |(rows, _)| rows) } else { sequence.rows.len() };
                (0..count).map(move |offset| ContextRow { tap_row: sequence.first_row + offset,
                    slot: sequence.placement.slot, position: sequence.start + offset })
            }).collect::<Vec<_>>();
            draft.update(&rows)?;
            for sequence in verified.iter_mut() {
                if let Some((_, token)) = sequence.kept { sequence.seq.anchor = Some(token); }
            }
        }
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
    fn cancel(&self, placement: &Placement) -> Result<()> {
        self.require_live()?;
        if let Some(mtp) = &self.mtp { mtp.discard_context(placement.slot)?; }
        if let Some(draft) = &self.drafter { draft.discard_context(placement.slot)?; }
        Ok(())
    }
    fn probe_begin(&self, rows: usize) { self.probe.borrow_mut().begin(rows); }
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> {
        self.probe.borrow().finish(observe)
    }
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
    /// Full-attention KV storage only; GDN recurrence remains FP32.
    #[arg(long, default_value = "fp8")]
    pub kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    /// Admit a full prefill batch of vocabulary logits for quality scoring.
    #[arg(long)]
    pub full_prefill_logits: bool,
    /// Bound packed rows to the singleton arithmetic route for exact validation.
    #[arg(long, value_enum, default_value_t = crate::shared::prefill_packing::PackingMode::default())]
    pub prefill_packing: crate::shared::prefill_packing::PackingMode,
    /// External checkpoint-native BF16 block dSpark; the launcher resolves its default.
    #[arg(long)]
    pub draft: Option<std::path::PathBuf>,
    #[arg(long)]
    pub draft_fixed: Option<usize>,
    /// Checkpoint-native one-layer MTP; external dSpark remains the default.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=3), conflicts_with = "draft")]
    pub mtp: u8,
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
            let reader = crate::shared::qwen::weights::WeightReader { library: &library, checkpoint: &checkpoint };
            let mtp_admission = (engine_args.mtp > 0).then(|| super::engine::mtp::Admission::new(
                &reader, &header.config, &programs)).transpose()?;
            let draft_config = engine_args.draft.as_ref().map(|snapshot| {
                block_dspark::check_checkpoint(snapshot, header.config.hidden, header.config.vocab, header.config.attention.len())?;
                let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
                super::engine::DrafterBytes::block_dspark(&config, lanes, library.sm_count()? as u64, header.config.max_context)
            }).transpose()?;
            ensure!(engine_args.draft_fixed.is_none() || engine_args.draft.is_some(), "--draft-fixed requires --draft");
            let engine = RefCell::new(DenseEngine::with_prefix(&library, &programs, &checkpoint,
                header.config, engine_args.pool_tokens.filter(|&tokens| tokens > 0),
                draft_config.or_else(|| mtp_admission.as_ref().map(|admission| admission.bytes)), lanes,
                Some(engine_args.prefix.mark_rule(lanes)), engine_args.kv_cache, engine_args.full_prefill_logits)?);
            let mut backend = DenseBackend::new(&engine, &library);
            backend.packing = engine_args.prefill_packing;
            let drafts = if let Some(snapshot) = &engine_args.draft {
                let target = engine.borrow();
                let config = block_dspark::DsparkConfig::read(snapshot)?;
                let weights = block_dspark::prefetch(snapshot).join().map_err(|_| anyhow::anyhow!("draft loader panicked"))??;
                let mask = target.token_embedding().host_rows(&[config.mask_token])?;
                let draft = Rc::new(DsparkDrafter::load_paged(&library, snapshot, weights, target.target_stream().raw,
                    lanes, lanes, mask, cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only,
                    Fp8Scales::Amax, Fp8Rows::W8a16, target.cfg.max_context, target.pool_tokens,
                    cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16)?);
                draft.prepare_workspace()?;
                let depth = engine_args.draft_fixed.unwrap_or(config.drafts());
                ensure!(depth <= config.drafts(), "--draft-fixed exceeds checkpoint draft block");
                backend.drafter = Some(draft);
                scheduler::Drafts::Neural { depth, fixed: engine_args.draft_fixed.is_some() }
            } else if let Some(admission) = mtp_admission {
                backend.mtp = Some(Rc::new(super::engine::mtp::MtpCache::load(&engine.borrow(), &reader, admission)?));
                backend.selector = RefCell::new(Some(crate::shared::token_io::TokenSelector::new(&library,
                    crate::shared::token_io::SelectPlacement::Device, engine.borrow().cfg.vocab, 1)?));
                scheduler::Drafts::Neural { depth: usize::from(engine_args.mtp), fixed: false }
            } else { scheduler::Drafts::None };
            scheduler::schedule(&backend, &engine_args.snapshot, &mut receive, &worker_stats, lanes,
                drafts, eos, engine_args.decode_share, &engine_args.prefix,
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
    let hub = crate::shared::console::hub(false, || console_layout(&args, &profile.id));
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

fn console_layout(args: &ServeArgs, model: &str) -> Result<crate::shared::console::Layout> {
    use crate::shared::console;
    let mut layout = console::Layout::new("qwen3_5", model.into(), args.snapshot.clone());
    layout.hardware = console::hardware(1, 0, true);
    layout.concurrency = usize::from(args.max_sequences);
    layout.speculator = args.draft.as_deref().map(|snapshot| {
        let depth = args.draft_fixed.unwrap_or(block_dspark::DsparkConfig::read(snapshot)?.drafts());
        Ok::<_, anyhow::Error>(console::Speculator { name: "dSpark".into(), positions: depth,
            policy: if args.draft_fixed.is_some() { format!("fixed {depth}") } else { format!("adaptive <= {depth}") } })
    }).transpose()?;
    if args.mtp > 0 {
        layout.speculator = Some(console::Speculator { name: "MTP".into(), positions: 1, policy: "adaptive <= 1".into() });
    }
    Ok(layout)
}

pub(crate) fn text_media() -> MediaAdmission<scheduler::Prompt, crate::shared::qwen::Encoder> {
    MediaAdmission::new(EmbeddingCache::new(0), crate::shared::qwen::Encoder::Off, 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        serve: ServeArgs,
    }

    #[test]
    fn dense_anchor_enables_drafting_after_prefill_or_restore() {
        use scheduler::DraftState as _;
        let mut state = DraftState::default();
        assert_eq!(state.pending_rows(), 0);
        state.close(42);
        assert_eq!(state.pending_rows(), 1);
        assert_eq!(state.anchor, Some(42));
    }

    #[test]
    #[ignore = "requires dense target/drafter checkpoints, matching native library and allocated GPU"]
    fn dense_dspark_prefix_frontiers_are_byte_exact() -> Result<()> {
        use cuteafd_loader::{families::qwen3_5::Qwen35Config, plan::checkpoint::Checkpoint};
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_DENSE_TEST_SNAPSHOT")?);
        let draft_snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_DENSE_TEST_DRAFT")?);
        let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_DENSE_TEST_MANIFEST")?);
        // SAFETY: the library outlives every stream, program and allocation owner.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        let cfg = Qwen35Config::read(&snapshot)?;
        let checkpoint = Checkpoint::open(&snapshot)?;
        let config = serde_json::from_slice(&std::fs::read(draft_snapshot.join("config.json"))?)?;
        let bytes = super::super::engine::DrafterBytes::block_dspark(&config, 4,
            library.sm_count()? as u64, cfg.max_context)?;
        let engine = RefCell::new(DenseEngine::with_prefix(&library, &programs, &checkpoint, cfg,
            Some(262144), Some(bytes), 4, Some(crate::shared::prefix::MarkRule {
                lanes: 4, entries: 4, budget_bytes: 1 << 30, host: false }),
            std::env::var("CUTEAFD_DENSE_TEST_KV_CACHE").unwrap_or_else(|_| "fp8".into()).parse()?, false)?);
        let draft = {
            let target = engine.borrow();
            let cfg = block_dspark::DsparkConfig::read(&draft_snapshot)?;
            let tensors = block_dspark::prefetch(&draft_snapshot).join()
                .map_err(|_| anyhow::anyhow!("draft loader panicked"))??;
            let mask = target.token_embedding().host_rows(&[cfg.mask_token])?;
            Rc::new(DsparkDrafter::load_paged(&library, &draft_snapshot, tensors, target.target_stream().raw,
                4, 4, mask, cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only,
                Fp8Scales::Amax, Fp8Rows::W8a16, target.cfg.max_context, target.pool_tokens,
                cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16)?)
        };
        draft.prepare_workspace()?;
        let family = DensePrefix::with_draft(&engine, Some(draft.clone()))?;
        let mut backend = DenseBackend::new(&engine, &library);
        backend.drafter = Some(draft);
        super::super::engine::prefix::check_backend_frontiers(&engine, &[1, 2, 3, 4], &backend, family, true)
    }

    #[test]
    #[ignore = "requires dense MTP checkpoint, matching native library and allocated GPU"]
    fn dense_mtp_serving_prefix_frontiers_are_byte_exact() -> Result<()> {
        use cuteafd_loader::{families::qwen3_5::Qwen35Config, plan::checkpoint::Checkpoint};
        use super::super::engine::mtp::{Admission, MtpCache};
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_DENSE_TEST_SNAPSHOT")?);
        let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_DENSE_TEST_MANIFEST")?);
        // SAFETY: the library outlives every stream, program and allocation owner.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        let cfg = Qwen35Config::read(&snapshot)?;
        let checkpoint = Checkpoint::open(&snapshot)?;
        let reader = crate::shared::qwen::weights::WeightReader { library: &library, checkpoint: &checkpoint };
        let admission = Admission::new(&reader, &cfg, &programs)?;
        let engine = RefCell::new(DenseEngine::with_prefix(&library, &programs, &checkpoint, cfg,
            Some(262144), Some(admission.bytes), 4, Some(crate::shared::prefix::MarkRule {
                lanes: 4, entries: 4, budget_bytes: 1 << 30, host: false }),
            std::env::var("CUTEAFD_DENSE_TEST_KV_CACHE").unwrap_or_else(|_| "fp8".into()).parse()?, false)?);
        let mtp = Rc::new(MtpCache::load(&engine.borrow(), &reader, admission)?);
        let family = DensePrefix::with_mtp(&engine, mtp.clone())?;
        let mut backend = DenseBackend::new(&engine, &library);
        backend.mtp = Some(mtp);
        backend.selector = RefCell::new(Some(crate::shared::token_io::TokenSelector::new(&library,
            crate::shared::token_io::SelectPlacement::Device, engine.borrow().cfg.vocab, 1)?));
        super::super::engine::prefix::check_backend_frontiers(&engine, &[1, 2, 3, 4], &backend, family, true)
    }

    #[test]
    fn native_mtp_is_opt_in_and_excludes_external_drafting() {
        let base = ["dense", "--snapshot", "checkpoint", "--native-lib", "native.so"];
        assert_eq!(Cli::try_parse_from(base).unwrap().serve.mtp, 0);
        for depth in ["1", "2", "3"] {
            assert_eq!(Cli::try_parse_from(base.into_iter().chain(["--mtp", depth])).unwrap().serve.mtp,
                depth.parse::<u8>().unwrap());
        }
        assert!(Cli::try_parse_from(base.into_iter().chain(["--mtp", "4"])).is_err());
        assert!(Cli::try_parse_from(base.into_iter().chain(["--mtp", "1", "--draft", "external"])).is_err());
    }

    #[test]
    fn dense_probe_observes_acceptance_and_costs_only_in_its_concurrency_bucket() {
        use crate::shared::draft::{clock::RoundTimes, evidence::DraftSource, resource};
        let mut policy = cuteafd_core::DraftPolicy::seeded(draft_geometry(64, 16, 64, 7),
            cuteafd_core::PolicySeed::weak(7)).unwrap();
        let mut probe = DenseProbe::default();
        assert!(probe.finish(|_, _| ()).is_none());
        probe.arm(64);
        for concurrency in [1, 4, 16] {
            let other = [1, 4, 16].into_iter().filter(|&n| n != concurrency)
                .map(|n| (n, policy.cost_snapshot(false, n), policy.calibration(n))).collect::<Vec<_>>();
            let before = policy.cost_snapshot(false, concurrency);
            let calibration = policy.calibration(concurrency);
            probe.begin(concurrency * 3);
            let requests = (0..concurrency).map(|id| resource::Verified { id: id as u64, rows: 3,
                accepted: 2, source: DraftSource::Neural, prior: Some(&[0.8, 0.8]), features: None,
                censor: None }).collect::<Vec<_>>();
            probe.finish(|routes, times| {
                assert!(routes.flat().is_empty());
                routes.complete().unwrap();
                resource::observe_drafted(&mut policy, &requests, routes, times,
                    RoundTimes { total_us: 2000 * concurrency as u64, draft_us: Some(300) },
                    7, None, Some(concurrency)).unwrap();
            }).unwrap();
            assert_ne!(policy.cost_snapshot(false, concurrency).round, before.round);
            assert_ne!(policy.cost_snapshot(false, concurrency).draft, before.draft);
            assert_ne!(policy.calibration(concurrency), calibration);
            for (n, costs, calibration) in other {
                assert_eq!(policy.cost_snapshot(false, n), costs);
                assert_eq!(policy.calibration(n), calibration);
            }
        }
        assert_eq!(policy.stats().rounds, 3);
        assert_eq!(policy.stats().verified_drafts, 42);
        assert_eq!(policy.stats().accepted_drafts, 21);
    }

    #[test]
    fn dense_defaults_to_compact_kv_without_full_prefill_arena() {
        let parsed = Cli::try_parse_from(["dense", "--snapshot", "checkpoint", "--native-lib", "native.so"])
            .unwrap().serve;
        assert_eq!(parsed.kv_cache, cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8);
        assert!(!parsed.full_prefill_logits);
    }

    #[test]
    fn full_prefill_logits_is_explicit_and_preserves_kv_selection() {
        for kv in ["bf16", "fp8"] {
            for full in [false, true] {
                let mut args = vec!["dense", "--snapshot", "checkpoint", "--native-lib", "native.so", "--kv-cache", kv];
                if full { args.push("--full-prefill-logits"); }
                let parsed = Cli::try_parse_from(args).unwrap().serve;
                assert_eq!(parsed.full_prefill_logits, full);
                assert_eq!(parsed.kv_cache.to_string(), kv);
                assert_eq!(parsed.max_sequences, 4);
            }
        }
    }
}
