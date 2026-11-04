//! Kimi's binding of the shared MLA DFlash executor to local or remote streams.
use super::weights::GlobalWeights;
use crate::shared::{draft::remote::{Prepared, RemoteBridge}, fp8_linear::{Fp8Rows, Fp8Scales},
    memory::device::{Allocation, DeviceOwner, Stream}, peer_split::{RankDevice, TerminalState},
    speculation::dflash::{ContextRow, DflashDrafter, Draft, DraftSeq, TargetHead, TAP_ROWS}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, programs::Programs};
use cuteafd_loader::{families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation},
    formats::dflash::{MlaDflashConfig, MlaDraftRuntimeLayout}, placement::Demand};
use std::{cell::{Cell, RefCell}, path::PathBuf, rc::Rc};

#[derive(Clone, Debug)]
pub(crate) struct Options {
    pub snapshot: PathBuf,
    pub fp8: bool,
    pub cache_format: cuteafd_core::MlaCacheFormat,
}

impl Options {
    pub(crate) fn representation(&self) -> GlmDraftRepresentation {
        if self.fp8 { GlmDraftRepresentation::Fp8Only } else { GlmDraftRepresentation::Bf16Only }
    }

    /// Exact verify widths with startup graphs: anchor plus a full draft block.
    pub(crate) fn verify_rows(&self) -> Result<usize> { Ok(MlaDflashConfig::read(&self.snapshot)?.block) }

    pub(crate) fn layout(&self, slots: usize, context: usize, sms: usize) -> Result<MlaDraftRuntimeLayout> {
        let cfg = MlaDflashConfig::read(&self.snapshot)?;
        cfg.runtime_layout(self.representation(), GlmDraftCapacity::new(slots, slots.min(8), cfg.block)?,
            context, slots.checked_mul(context).context("draft pool overflow")?, sms, Fp8Rows::W8a16.code() as u8, self.cache_format).map_err(Into::into)
    }
}

/// The target's tap staging holds one prefill call's `rows` (the lane rows).
pub(crate) fn bridge_demands(layout: &MlaDraftRuntimeLayout, rows: usize) -> Result<Vec<Demand>> {
    cuteafd_loader::families::kimi_k3::draft_bridge_demands(layout, rows)
}

/// Field order is intentional: the common executor drops before its borrowed
/// target weights, including on partial startup failure.
struct Owners<D, T> { draft: D, target: T }

/// Local drafting uses an ordinary retained stream, never a same-device peer exchange.
enum Bridge<'a> {
    Remote(RemoteBridge<'a>),
    Local { stream: Stream<'a>, terminal: Cell<TerminalState> },
}

impl<'a> Bridge<'a> {
    fn stream(&self) -> &Stream<'a> {
        match self { Self::Remote(bridge) => &bridge.stream, Self::Local { stream, .. } => stream }
    }
    fn require_live(&self) -> Result<()> {
        match self {
            Self::Remote(bridge) => bridge.require_live(),
            Self::Local { terminal, .. } => {
                ensure!(terminal.get() == TerminalState::Active, "Kimi local draft is terminal ({:?})", terminal.get());
                Ok(())
            }
        }
    }
    fn abort(&self) -> Result<()> {
        match self {
            Self::Remote(bridge) => bridge.publish_abort(),
            Self::Local { terminal, .. } => {
                if terminal.get() == TerminalState::Active { terminal.set(TerminalState::Aborting); }
                Ok(())
            }
        }
    }
    fn finish_request(&self) -> Result<()> {
        self.require_live()?;
        match self { Self::Remote(bridge) => bridge.finish_request(), Self::Local { .. } => Ok(()) }
    }
    fn copy_local(&self, destination: CuteafdDeviceBuffer, source: CuteafdDeviceBuffer,
        bytes: usize, producer: Option<&crate::shared::memory::device::Event<'a>>) -> Result<()> {
        self.require_live()?;
        let Self::Local { stream, .. } = self else { anyhow::bail!("Kimi local copy requires a local bridge") };
        ensure!(source.device_id == stream.device.id && destination.device_id == stream.device.id
            && bytes <= source.bytes && bytes <= destination.bytes,
            "Kimi local draft copy exceeds actual device owners");
        stream.device.run(|| {
            // SAFETY: the adapter retains both buffers, the event and this stream
            // through its terminal drain; readers follow this copy on the same stream.
            unsafe {
                if let Some(event) = producer { stream.device.library.cuda_stream_wait_event(stream.raw, event.raw)?; }
                stream.device.library.copy_d2d_async(destination, source, bytes, stream.raw)
            }
        })
    }
    fn synchronize(&self) -> Result<()> {
        match self { Self::Remote(bridge) => bridge.synchronize(), Self::Local { stream, .. } => stream.drain() }
    }
    fn finish_terminal(&self, drained: bool) {
        match self {
            Self::Remote(bridge) => bridge.finish_terminal(drained),
            Self::Local { terminal, .. } if terminal.get() == TerminalState::Aborting => {
                terminal.set(if drained { TerminalState::Drained } else { TerminalState::Retained });
            }
            Self::Local { .. } => {}
        }
    }
    fn checked<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        self.require_live()?;
        let result = work();
        if result.is_err() {
            crate::shared::decode_graph::fatal_drain(self.abort(), "Kimi draft abort publication");
        }
        result
    }
}

pub(crate) struct NativeDraft<'a> {
    owners: Owners<DeviceOwner<'a, DflashDrafter<'a>>, Option<Vec<Rc<GlobalWeights<'a>>>>>,
    taps: Allocation<'a>,
    ids: Allocation<'a>,
    input: Allocation<'a>,
    // Bridge controls and its stream outlive all common/target staging owners.
    bridge: Bridge<'a>,
    target_streams: Vec<Rc<Stream<'a>>>,
    producer: RefCell<crate::shared::memory::device::Event<'a>>,
    consumed: RefCell<crate::shared::memory::device::Event<'a>>,
    context_pending: Cell<bool>,
    window_marks: Option<Rc<Allocation<'a>>>,
    window_bytes: usize,
    programs: &'a Programs<'a>,
    /// The drafter owns a head program and scratch at its admitted row capacity;
    /// only the immutable target head weights are shared with the target lane.
    k6_head: Option<(crate::shared::dense_trellis::Projection<'a>, crate::shared::dense_trellis::ProjectionScratch<'a>)>,
    cache_format: cuteafd_core::MlaCacheFormat,
    marks: usize,
}

impl<'a> NativeDraft<'a> {
    pub(crate) fn load(admission: &mut super::admission::Admission<'a>,
        programs: &'a Programs<'a>, options: &Options, streams: [Rc<Stream<'a>>; 2],
        layout: &MlaDraftRuntimeLayout) -> Result<Self> {
        Self::load_for_ranks(admission, programs, options, streams.to_vec(), layout)
    }

    pub(crate) fn load_for_ranks(admission: &mut super::admission::Admission<'a>,
        programs: &'a Programs<'a>, options: &Options, streams: Vec<Rc<Stream<'a>>>,
        layout: &MlaDraftRuntimeLayout) -> Result<Self> {
        ensure!(options.cache_format == layout.cache_format(), "Kimi draft options/admission cache records differ");
        ensure!(matches!(streams.len(), 1 | 2)
            && (streams.len() == 1 || streams[0].device.id != streams[1].device.id),
            "Kimi draft requires one or two distinct physical owners");
        let head = streams.len() - 1;
        let ranks = streams.iter().map(|s| RankDevice { device: s.device.id, stream: s.raw }).collect::<Vec<_>>();
        let library = streams[0].device.library;
        let bridge = if streams.len() == 1 {
            Bridge::Local { stream: Stream::new(streams[0].device)?, terminal: Cell::new(TerminalState::Active) }
        } else {
            Bridge::Remote(RemoteBridge::new(Prepared::new(library, ranks[0], ranks[head], head,
                "kimi-remote-mla-dflash", crate::shared::draft::remote::decode_extent(
                    layout.capacity().block_rows, layout.target_hidden(), layout.target_vocab(), head)?)?))
        };
        // Recording preserves admitted allocation order; the shared group fills
        // these owners before target binding without a host payload read.
        let file = if crate::shared::experts::rtx::dense::is_recording() { Vec::new() }
            else { std::fs::read(options.snapshot.join("model.safetensors"))? };
        let draft = admission.allocate_draft(bridge.stream().device, programs, &options.snapshot, file,
            bridge.stream().raw, options.representation(), Fp8Scales::Amax, Fp8Rows::W8a16)?;
        let demands = cuteafd_loader::families::kimi_k3::draft_bridge_demands_for_ranks(
            layout, admission.lane_rows(), streams.len())?;
        let device = streams[0].device;
        let allocate = |name| -> Result<Allocation<'a>> {
            let demand = demands.iter().find(|d| d.group == name).context("Kimi draft bridge inventory missing")?;
            Allocation::new(device, usize::try_from(demand.bytes)?)
        };
        let marks = usize::try_from(admission.mark_selection.slots)?;
        let window_bytes = layout.window_mark_bytes()?;
        let window_marks = admission.allocate_draft_marks(streams[head].device)?;
        Ok(Self { window_marks, window_bytes, programs, k6_head: None, cache_format: layout.cache_format(), marks, owners: Owners { draft, target: None }, taps: allocate("DFlash bridge taps")?,
            ids: allocate("DFlash bridge input ids")?, input: allocate("DFlash bridge gathered input")?,
            bridge, producer: RefCell::new(crate::shared::memory::device::Event::new(device)?),
            consumed: RefCell::new(crate::shared::memory::device::Event::new(streams[head].device)?),
            context_pending: Cell::new(false), target_streams: streams })
    }

    pub(crate) fn cache_format(&self) -> cuteafd_core::MlaCacheFormat { self.cache_format }

    /// Validate the head rank, retaining a separate K6 program and scratch
    /// sized at the drafter's own admitted capacity.
    pub(crate) fn bind_target(&mut self, target: [Rc<GlobalWeights<'a>>; 2],
        head_programs: Rc<super::device::NativePrograms<'a>>) -> Result<()> {
        self.bind_target_ranks(target.to_vec(), head_programs)
    }

    pub(crate) fn bind_target_ranks(&mut self, target: Vec<Rc<GlobalWeights<'a>>>,
        head_programs: Rc<super::device::NativePrograms<'a>>) -> Result<()> {
        ensure!(self.owners.target.is_none() && target.len() == self.target_streams.len(),
            "Kimi draft target already bound or physical inventory differs");
        let head = target.len() - 1;
        for (rank, owner) in target.iter().enumerate() { owner.check_device(self.target_streams[rank].device)?; }
        let cfg = &self.owners.draft.cfg;
        ensure!(target[0].buffer("embedding")?.bytes == cfg.vocab * cfg.hidden * 2,
            "Kimi draft target embedding differs from admitted BF16 geometry");
        if let Some(weight) = &target[head].core_head {
            ensure!(weight.shape == [cfg.vocab, cfg.hidden], "Kimi draft K6 target head geometry differs");
            ensure!(head_programs.device.id == self.target_streams[head].device.id,
                "Kimi draft K6 head programs are not on the head rank");
            let rows = self.owners.draft.max_batch_sequences().checked_mul(cfg.block)
                .context("Kimi draft head row capacity overflow")?;
            let projection = self.target_streams[head].device.run(|| {
                crate::shared::dense_trellis::Projection::load_geometry_bits(self.programs,
                    cfg.hidden, cfg.vocab, 1, weight.bits, crate::shared::dense_trellis::codebook(weight),
                    rows, cuteafd_ffi::dense_trellis::TrellisDType::F32)
            })?;
            let scratch = {
                let _scope = cuteafd_ffi::memory_ledger::scope("drafter/kimi K6 head scratch");
                crate::shared::dense_trellis::ProjectionLayout::new(&projection.program, rows)?
                    .allocate(self.target_streams[head].device)?
            };
            self.k6_head = Some((projection, scratch));
        } else {
            ensure!(target[head].buffer("head")?.bytes == cfg.vocab * cfg.hidden * 2,
                "Kimi draft target head differs from admitted BF16 geometry");
        }
        self.owners.target = Some(target);
        self.owners.draft.device.run(|| self.owners.draft.prime_remote())?;
        // Prime the target-side ordered ID staging before readiness. This is an
        // ordered copy fence, not a compute drain; startup retires it explicitly.
        for count in 1..=self.owners.draft.max_batch_sequences() {
            self.gather(&vec![cfg.mask_token; count * cfg.block])?;
        }
        self.target_streams[0].drain()?;
        if matches!(self.bridge, Bridge::Local { .. }) { self.bridge.synchronize()?; }
        Ok(())
    }

    fn gather(&self, tokens: &[u32]) -> Result<()> {
        let draft = &self.owners.draft;
        ensure!(!tokens.is_empty() && tokens.len() * 4 <= self.ids.buffer.bytes
            && tokens.iter().all(|&id| (id as usize) < draft.cfg.vocab), "Kimi draft embedding IDs exceed admission");
        let weights = self.owners.target.as_ref().context("Kimi draft target is not bound")?;
        let table = weights[0].buffer("embedding")?;
        let ids = crate::shared::prefix::view(self.ids.buffer, 0, tokens.len() * 4)?;
        let output = crate::shared::prefix::view(self.input.buffer, 0, tokens.len() * draft.cfg.hidden * 2)?;
        let data = tokens.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>();
        let stream = match &self.bridge {
            Bridge::Remote(bridge) => bridge.exchange.stream(0),
            Bridge::Local { stream, .. } => stream.raw,
        };
        self.target_streams[0].device.run(|| {
            // SAFETY: all target operands and this exact stream survive the draft's drain.
            unsafe {
                draft.device.library.copy_h2d_ordered(self.ids.buffer, &data, stream)?;
                draft.device.library.cuda_embedding_lookup_bf16_async(table, ids, output, tokens.len(),
                    draft.cfg.vocab, draft.cfg.hidden, stream)
            }
        })
    }

    pub(crate) fn tap(&self, layer: usize, local: &mut super::device::LocalLane<'a>, rows: usize) -> Result<()> {
        self.tap_at(layer, local, rows, 0)
    }

    pub(crate) fn tap_at(&self, layer: usize, local: &mut super::device::LocalLane<'a>,
        rows: usize, to: usize) -> Result<()> {
        let Some(index) = self.owners.draft.tap_index(layer) else { return Ok(()) };
        let cfg = &self.owners.draft.cfg;
        let offset = tap_offset(to, rows, cfg.taps.len() * cfg.hidden, self.tap_rows())?;
        ensure!(Rc::ptr_eq(local.stream(), &self.target_streams[0]),
            "Kimi draft taps exceed staging or use the wrong producer stream");
        let hidden = local.terminal_residual(rows)?;
        local.stream().device.run(|| {
            // SAFETY: terminal_residual and this strided copy share the exact producer
            // stream; target staging survives every reader and is not graph-captured.
            unsafe { self.owners.draft.device.library.glm_dflash_tap(hidden.ptr.cast_const(),
                self.taps.buffer.ptr.wrapping_byte_add(offset), rows, cfg.hidden, cfg.taps.len() * cfg.hidden,
                index * cfg.hidden, local.stream().raw) }
        })
    }

    /// Join the previous context readers before the next target overwrites taps.
    pub(crate) fn prepare_target(&self) -> Result<()> {
        if self.context_pending.replace(false) {
            self.target_streams[0].device.run(|| {
                // SAFETY: this exact cross-rank event and both streams are retained.
                unsafe { self.owners.draft.device.library.cuda_stream_wait_event(
                    self.target_streams[0].raw, self.consumed.borrow().raw) }
            })?;
        }
        Ok(())
    }

    pub(crate) fn update(&self, rows: &[ContextRow]) -> Result<()> {
        for slice in context_slices(rows, self.tap_rows())? { self.update_slice(&slice)?; }
        Ok(())
    }

    /// Rows of the target's tap staging (one prefill call's lane rows).
    fn tap_rows(&self) -> usize {
        let cfg = &self.owners.draft.cfg;
        self.taps.buffer.bytes / (cfg.taps.len() * cfg.hidden * 2)
    }

    fn update_slice(&self, rows: &[ContextRow]) -> Result<()> {
        let first = rows.iter().map(|r| r.tap_row).min().context("empty Kimi context slice")?;
        let last = rows.iter().map(|r| r.tap_row).max().unwrap();
        let rebased = rows.iter().map(|r| ContextRow { tap_row: r.tap_row - first, ..*r }).collect::<Vec<_>>();
        let draft = &self.owners.draft;
        let pitch = draft.cfg.taps.len() * draft.cfg.hidden * 2;
        self.bridge.checked(|| draft.device.run(|| {
            draft.update_with(&rebased, || {
                self.producer.borrow_mut().record(&self.target_streams[0])?;
                match &self.bridge {
                    Bridge::Remote(bridge) => {
                        bridge.exchange.on(0, || {
                            // SAFETY: retained event joins this exact producer and consumer.
                            unsafe { draft.device.library.cuda_stream_wait_event(bridge.exchange.stream(0),
                                self.producer.borrow().raw) }
                        })?;
                        // SAFETY: checked source frontier and rebased destination fit their owners.
                        let plane = unsafe { cuteafd_ffi::peer_exchange::PeerPlane {
                            source: self.taps.buffer.ptr.cast::<u8>().add(first * pitch).cast(),
                            destination: draft.taps.buffer.ptr,
                            rows: (last - first + 1) as u64, row_bytes: pitch as u64,
                            source_pitch: pitch as u64, destination_pitch: pitch as u64,
                        } };
                        bridge.publish_context(&[plane])
                    }
                    Bridge::Local { .. } => {
                        let bytes = (last - first + 1) * pitch;
                        let source = crate::shared::prefix::view(self.taps.buffer, first * pitch, bytes)?;
                        self.bridge.copy_local(draft.taps.buffer, source, bytes, Some(&self.producer.borrow()))
                    }
                }
            })?;
            if let Bridge::Remote(bridge) = &self.bridge { bridge.acknowledge_context()?; }
            self.consumed.borrow_mut().record(self.bridge.stream())?;
            self.context_pending.set(true);
            Ok(())
        }))
    }

    pub(crate) fn reset(&self, slot: usize, origin: usize) -> Result<()> {
        self.bridge.finish_request()?;
        self.owners.draft.device.run(|| self.owners.draft.reset_context(slot, origin))
    }
    pub(crate) fn discard(&self, slot: usize) -> Result<()> {
        self.bridge.finish_request()?;
        self.owners.draft.device.run(|| self.owners.draft.discard_context(slot))
    }
    #[cfg(test)]
    pub(super) fn readback(&self, slot: usize, tap_rows: usize) -> Result<crate::shared::speculation::dflash::MlaReadback> {
        self.owners.draft.device.run(|| self.owners.draft.mla_readback(slot, tap_rows))
    }

    pub(crate) fn device(&self) -> crate::shared::memory::device::Device<'a> { self.owners.draft.device }
    pub(crate) fn common(&self) -> &DflashDrafter<'a> { &self.owners.draft }
    pub(crate) fn frontier(&self, slot: usize) -> Result<(usize, usize)> { self.common().mla_frontier(slot) }
    pub(crate) fn mark_owner(&self) -> Option<Rc<Allocation<'a>>> { self.window_marks.clone() }
    pub(crate) fn mark_bytes(&self) -> usize { self.window_bytes }
    pub(crate) fn mark_segments(&self, mark: usize) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!(mark < self.marks, "Kimi draft window mark outside admission");
        let window_marks = self.window_marks.as_ref().context("Kimi draft prefix marks are disabled")?;
        let windows = self.common().mla_window_segments(0)?;
        let mut offset = mark * self.window_bytes;
        windows.into_iter().map(|window| {
            let view = crate::shared::prefix::view(window_marks.buffer, offset, window.bytes)?;
            offset += window.bytes;
            Ok(view)
        }).collect()
    }
    pub(crate) fn move_mark(&self, mark: usize, slot: usize, capture: bool) -> Result<()> {
        let marks = self.mark_segments(mark)?;
        self.drain()?;
        self.owners.draft.device.run(|| self.common().copy_mla_windows(slot, &marks, capture))
    }

    pub(crate) fn max_batch_sequences(&self) -> usize { self.owners.draft.max_batch_sequences() }
    pub(crate) fn drafts(&self) -> usize { self.owners.draft.cfg.drafts() }
    pub(crate) fn require_live(&self) -> Result<()> { self.bridge.require_live() }
    pub(crate) fn abort(&self) -> Result<()> { self.bridge.abort() }
    pub(crate) fn drain(&self) -> Result<()> { self.bridge.synchronize() }
    pub(crate) fn finish_terminal(&self, drained: bool) { self.bridge.finish_terminal(drained); }

    pub(crate) fn propose(&self, sequences: &[DraftSeq]) -> Result<Vec<Draft>> {
        let draft = &self.owners.draft;
        let target = self.owners.target.as_ref().context("Kimi draft target is not bound")?;
        let head = target.len() - 1;
        // The reference is tied to this adapter's retained owner, not to a copied
        // buffer descriptor or a synthetic legacy allocation.
        self.bridge.checked(|| draft.device.run(|| {
            let input = |tokens: &[u32], output: CuteafdDeviceBuffer, stream| {
                ensure!(stream == self.bridge.stream().raw, "Kimi draft input stream differs from bridge");
                self.gather(tokens)?;
                let bytes = tokens.len() * draft.cfg.hidden * 2;
                match &self.bridge {
                    Bridge::Remote(bridge) => bridge.send_input(self.input.buffer.ptr.cast_const(), output.ptr, bytes),
                    Bridge::Local { .. } => self.bridge.copy_local(output, self.input.buffer, bytes, None),
                }
            };
            let head = match &self.k6_head {
                Some((projection, scratch)) => {
                    let weight = target[head].core_head.as_ref().context("Kimi draft K6 target head absent")?;
                    let device = self.target_streams[head].device.id;
                    let (hidden, vocab) = (draft.cfg.hidden, draft.cfg.vocab);
                    TargetHead::Launch(Box::new(move |x, logits, rows, stream| {
                        ensure!(stream == self.bridge.stream().raw, "Kimi draft K6 head stream differs from bridge");
                        let input = CuteafdDeviceBuffer { ptr: x.cast_mut(), bytes: rows * hidden * 2, device_id: device, flags: 0 };
                        let output = CuteafdDeviceBuffer { ptr: logits.cast(), bytes: rows * vocab * 4, device_id: device, flags: 0 };
                        // The drafter's own scratch, used only on this bridge stream; the
                        // target head weights are retained by the bound target owner.
                        projection.local(rows, input, weight, scratch, output, stream)
                    }))
                }
                None => TargetHead::Bf16Buffer(target[head].borrowed_buffer("head")?),
            };
            draft.draft_device_with(sequences, &input, head)
        }))
    }
}

impl crate::shared::prefix::cuda_copy::SnapshotOwner for NativeDraft<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<CuteafdDeviceBuffer>> { self.common().mla_snapshot_buffers() }
}

impl Drop for NativeDraft<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.drain(), "Kimi native draft teardown");
        for stream in &self.target_streams { stream.drain_or_abort("Kimi draft target borrowers"); }
    }
}

/// Bound the span, not just the number of accepted rows: rejected verify rows
/// can leave holes between requests in the target's tap staging.
fn context_slices(rows: &[ContextRow], tap_rows: usize) -> Result<Vec<Vec<ContextRow>>> {
    let mut rows = rows.to_vec();
    rows.sort_unstable_by_key(|r| r.tap_row);
    ensure!(rows.iter().all(|r| r.tap_row < tap_rows)
        && rows.windows(2).all(|pair| pair[0].tap_row != pair[1].tap_row),
        "Kimi context taps are duplicate or outside target staging");
    let mut slices: Vec<Vec<ContextRow>> = Vec::new();
    for row in rows {
        if slices.last().is_none_or(|slice| row.tap_row - slice[0].tap_row >= TAP_ROWS) {
            slices.push(Vec::new());
        }
        slices.last_mut().unwrap().push(row);
    }
    Ok(slices)
}

fn tap_offset(to: usize, rows: usize, stride: usize, tap_rows: usize) -> Result<usize> {
    ensure!(to.checked_add(rows).is_some_and(|end| end <= tap_rows),
        "Kimi draft tap destination exceeds staging");
    to.checked_mul(stride).and_then(|offset| offset.checked_mul(2))
        .context("Kimi draft tap offset overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Fixed lane rows for these tap fixtures, independent of serving defaults.
    const TARGET_TAP_ROWS: usize = 4096;

    #[test]
    fn subwave_taps_match_full_chunk_bytes() -> Result<()> {
        let (rows, hidden, taps) = (4096, 7, 3);
        let stride = taps * hidden;
        let mut full = vec![0u8; rows * stride * 2];
        let mut split = full.clone();
        for tap in 0..taps {
            let source = (0..rows * hidden * 2).map(|i| (i * 13 + tap) as u8).collect::<Vec<_>>();
            for row in 0..rows {
                let at = row * stride * 2 + tap * hidden * 2;
                full[at..at + hidden * 2].copy_from_slice(&source[row * hidden * 2..(row + 1) * hidden * 2]);
            }
            for (first, count) in [(0, 1408), (1408, 1408), (2816, 1280)] {
                let base = tap_offset(first, count, stride, TARGET_TAP_ROWS)?;
                for row in 0..count {
                    let at = base + row * stride * 2 + tap * hidden * 2;
                    let from = (first + row) * hidden * 2;
                    split[at..at + hidden * 2].copy_from_slice(&source[from..from + hidden * 2]);
                }
            }
        }
        assert_eq!(split, full);
        assert!(tap_offset(4095, 2, stride, TARGET_TAP_ROWS).is_err());
        assert!(tap_offset(8191, 1, stride, 8192).is_ok() && tap_offset(8191, 2, stride, 8192).is_err());
        assert!(tap_offset(usize::MAX, 1, stride, TARGET_TAP_ROWS).is_err());
        Ok(())
    }

    fn native_hosted_tail(dense: &[bool], csv: &str) -> Result<(usize, Vec<usize>)> {
        let hosted = csv.split(',').map(str::parse)
            .collect::<std::result::Result<Vec<usize>, _>>()?;
        let routed = dense.iter().enumerate().filter_map(|(layer, dense)| (!dense).then_some(layer))
            .collect::<Vec<_>>();
        let local = routed.len().checked_sub(hosted.len())
            .context("native proof worker list exceeds the routed layer count")?;
        ensure!(routed[local..] == hosted,
            "native proof workers must host an exact routed-layer suffix");
        Ok((local, hosted))
    }

    #[test]
    fn native_placement_derives_local_prefix_from_exact_worker_tail() -> Result<()> {
        let mut dense = vec![false; 93];
        dense[0] = true;
        let csv = (6..93).map(|layer| layer.to_string()).collect::<Vec<_>>().join(",");
        assert_eq!(native_hosted_tail(&dense, &csv)?, (5, (6..93).collect()));
        assert_eq!(native_hosted_tail(&[true, false, true, false, false], "3,4")?, (1, vec![3, 4]));
        for invalid in ["", "5,7,8", "91,91,92", "92,91", "0,91,92", "91,93", "91,92,"] {
            assert!(native_hosted_tail(&dense, invalid).is_err(), "accepted {invalid:?}");
        }
        assert!(native_hosted_tail(&[true, false], "0,1").is_err());
        Ok(())
    }

    #[test]
    fn wide_prefill_keeps_every_tap_in_two_common_slices() -> Result<()> {
        let rows = (0..TARGET_TAP_ROWS).map(|tap_row| ContextRow {
            tap_row, slot: 3, position: 8192 + tap_row }).collect::<Vec<_>>();
        let slices = context_slices(&rows, TARGET_TAP_ROWS)?;
        assert_eq!(slices.iter().map(Vec::len).collect::<Vec<_>>(), [TAP_ROWS, TAP_ROWS]);
        for (actual, expected) in slices.iter().flatten().zip(&rows) {
            assert_eq!((actual.tap_row, actual.slot, actual.position),
                (expected.tap_row, expected.slot, expected.position));
        }
        Ok(())
    }

    #[test]
    fn accepted_tap_holes_bound_span_not_row_count() -> Result<()> {
        let rows = [4095, 2047, 2048, 0].map(|tap_row| ContextRow {
            tap_row, slot: tap_row / TAP_ROWS, position: 17000 + tap_row });
        let slices = context_slices(&rows, TARGET_TAP_ROWS)?;
        assert_eq!(slices.iter().map(|slice| slice.iter().map(|r| r.tap_row)
            .collect::<Vec<_>>()).collect::<Vec<_>>(), [vec![0, 2047], vec![2048, 4095]]);
        for slice in &slices {
            let first = slice[0].tap_row;
            assert!(slice.iter().all(|row| row.tap_row - first < TAP_ROWS));
        }
        assert!(context_slices(&[], TARGET_TAP_ROWS)?.is_empty());
        assert!(context_slices(&[rows[0], rows[0]], TARGET_TAP_ROWS).is_err());
        assert!(context_slices(&[ContextRow { tap_row: TARGET_TAP_ROWS, ..rows[0] }], TARGET_TAP_ROWS).is_err());
        Ok(())
    }

    #[test]
    fn precision_is_one_explicit_resident_representation() {
        let options = Options { snapshot: PathBuf::new(), fp8: false, cache_format: Default::default() };
        assert_eq!(options.representation(), GlmDraftRepresentation::Bf16Only);
        assert_eq!(Options { fp8: true, ..options }.representation(), GlmDraftRepresentation::Fp8Only);
    }

    #[test]
    #[ignore = "requires sealed Kimi target/drafter, six Spark workers and CUTEAFD_KIMI_PUSH_DUMP via golden-run"]
    fn native_mla_push_dumps_each_requested_batch() -> Result<()> {
        let _ = tracing_subscriber::fmt().with_ansi(false)
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_test_writer().try_init();
        use crate::shared::qwen::scheduler::{self, Backend};
        use cuteafd_engine::{media::RequestMedia, prefix::{PrefixFamily, RefPagePool}};
        use super::super::{backend::{DraftState, KimiBackend}, prefix::Placement, startup};
        use std::sync::Arc;
        let env = |key| std::env::var(key).with_context(|| format!("Kimi push regression needs {key}"));
        let snapshot = PathBuf::from(env("CUTEAFD_GOLDEN_SNAPSHOT")?);
        let drafter = PathBuf::from(env("CUTEAFD_KIMI_DRAFT_SNAPSHOT")?);
        let output = PathBuf::from(env("CUTEAFD_KIMI_PUSH_DUMP")?);
        use crate::shared::golden_scores::push_regression as regression;
        let native = PathBuf::from(env("CUTEAFD_NATIVE_LIB")?);
        let manifest = PathBuf::from(env("CUTEAFD_PROGRAM_MANIFEST")?);
        let peers = env("CUTEAFD_GOLDEN_PEERS")?.split(',').map(str::parse)
            .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        ensure!(peers.len() == 6, "Kimi push regression requires six real Spark ranks");
        use crate::shared::experts::rtx::push::PushPolicy;
        ensure!(PushPolicy::for_snapshot(&snapshot)? == PushPolicy::Required
            && PushPolicy::for_snapshot(&drafter)? == PushPolicy::Required,
            "Kimi push regression requires target and drafter in a push namespace");
        let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&snapshot, false, false)?;
        let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let (local_layers, hosted_layers) = native_hosted_tail(&cfg.dense,
            &env("CUTEAFD_GOLDEN_EXPERT_LAYERS")?)?;
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
            .map_err(|_| anyhow::anyhow!("Kimi push regression requires an isolated test process"))?;
        startup::preflight(&manifest, &native)?;
        // SAFETY: golden-run supplies the sealed library; all borrowers drop before it.
        let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&native)? });
        let programs = library.programs()?.with_manifest(&manifest)?;
        let cache_format = cuteafd_core::MlaCacheFormat::Fp8;
        let loaded = startup::load(&library, &programs, &checkpoint, &cfg, &cfg, startup::Options {
            cache_format, devices: [0], native_lib: &native, peers,
            rtx_expert_layers: cuteafd_loader::placement::OnboardSetting::Layers(local_layers),
            spark_budget_gib: Some(110.), rows: 2048, waves: 2, prefill_rows: Some(1024),
            slots: 2, marks: 0, max_context: 8192, pool_tokens: Some(16384), graphs: false,
            ffn_mode: cuteafd_ffi::KimiFfnMode::default(), hosted_layers: Some(hosted_layers),
            draft: Some(Options { snapshot: drafter, fp8: true, cache_format }), verify_rows: 0,
        })?;
        let engine = &loaded.engine;
        let backend = KimiBackend::new(engine)?;
        let family = engine.prefix_family()?;
        let layout = family.layout();
        let mut pool = RefPagePool::new(layout.pages, layout.page_rows);
        let draft = engine.native_draft().context("Kimi push regression needs MLA DFlash")?;
        ensure!(draft.max_batch_sequences() >= 2, "Kimi push regression needs two admitted draft slots");
        std::fs::create_dir_all(&output)?;
        let prefix_error = |error: cuteafd_engine::prefix::BoxError| anyhow::anyhow!("{error}");
        for sequences in [1, 2] {
            let prompts = regression::prompts(sequences);
            let mut placements = Vec::new();
            let mut states = Vec::new();
            for (slot, prompt) in prompts.iter().enumerate() {
                let capacity = (prompt.len() + 16).div_ceil(layout.page_rows) * layout.page_rows;
                let mut placement = Placement { units: pool.alloc(capacity / layout.page_rows)?, slot, len: 0, state_len: 0 };
                family.bind(&mut placement).map_err(prefix_error)?;
                family.admit_draft_pages(&mut placement, capacity).map_err(prefix_error)?;
                backend.bind_request(&mut placement, (), RequestMedia::new(Vec::new(), cfg.hidden, prompt.len())?, prompt, 0)?;
                let mut state = DraftState::default();
                for (index, chunk) in prompt.chunks(1024).enumerate() {
                    backend.prefill_device(&mut placement, chunk, 1, &pool)?;
                    backend.prefill_draft(&placement, index * 1024, chunk, Some(42 + slot as u32), &mut state)?;
                }
                ensure!(draft.frontier(slot)? == (0, prompt.len()), "missing Kimi draft context slot {slot}");
                placements.push(placement);
                states.push(state);
            }
            let propose = |states: &mut [DraftState]| -> Result<Vec<Vec<u32>>> {
                let mut batch = placements.iter().zip(states).map(|(placement, seq)| scheduler::DraftSeq {
                    placement, seq, depth: draft.drafts() }).collect::<Vec<_>>();
                backend.draft(&mut batch, &mut scheduler::DraftTiming::default())
            };
            regression::draft_dump(&output, sequences, draft.drafts(), draft.common().cfg.block,
                cfg.vocab_size, || Ok((propose(&mut states)?, draft.common().last_logits(sequences)?)))?;
            // Fixed teacher-forced rows keep target comparison independent of draft agreement.
            let rows = regression::teacher_rows();
            let mut batch = placements.iter_mut().map(|placement| (placement, rows.as_slice())).collect::<Vec<_>>();
            let logits = backend.verify_device(&mut batch, true, &pool)?.context("Kimi verify logits absent")?;
            library.cuda_set_device(0)?;
            let values = logits.to_host(&library)?;
            regression::target_dump(&output, sequences, cfg.vocab_size, &values)?;
            for placement in &mut placements {
                backend.cancel(placement)?;
                family.drain().map_err(prefix_error)?;
                family.discard(placement).map_err(prefix_error)?;
                pool.release(&placement.units);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires sealed Kimi target, MLA drafter and six real Spark workers via golden-run --native-test"]
    fn native_mla_restored_cold_matches_fresh_and_uncached_target() -> Result<()> {
        use crate::shared::qwen::scheduler::{self, Backend};
        use cuteafd_engine::{media::RequestMedia, prefix::{MarkSlot, PrefixFamily, RefPagePool, TailCopy}};
        use super::super::{backend::{DraftState, KimiBackend}, prefix::Placement, startup};
        use std::sync::Arc;
        let env = |key| std::env::var(key).with_context(|| format!("native MLA proof needs {key}"));
        let snapshot = PathBuf::from(env("CUTEAFD_GOLDEN_SNAPSHOT")?);
        let drafter = PathBuf::from(env("CUTEAFD_KIMI_DRAFT_SNAPSHOT")?);
        let native = PathBuf::from(env("CUTEAFD_NATIVE_LIB")?);
        let manifest = PathBuf::from(env("CUTEAFD_PROGRAM_MANIFEST")?);
        let peers = env("CUTEAFD_GOLDEN_PEERS")?.split(',').map(str::parse)
            .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let cache_format = std::env::var("CUTEAFD_KIMI_KV_CACHE").unwrap_or_else(|_| "bf16".into()).parse().map_err(anyhow::Error::msg)?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&snapshot, false, false)?;
        let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let (local_layers, hosted_layers) = native_hosted_tail(&cfg.dense,
            &env("CUTEAFD_GOLDEN_EXPERT_LAYERS")?)?;
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
            .map_err(|_| anyhow::anyhow!("native MLA proof requires an isolated test process"))?;
        startup::preflight(&manifest, &native)?;
        // SAFETY: golden-run supplies the sealed library; all borrowers drop before it.
        let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&native)? });
        let programs = library.programs()?.with_manifest(&manifest)?;
        let loaded = startup::load(&library, &programs, &checkpoint, &cfg, &cfg, startup::Options {
            cache_format,
            devices: [0, 1], native_lib: &native, peers,
            rtx_expert_layers: cuteafd_loader::placement::OnboardSetting::Layers(local_layers), spark_budget_gib: None,
            rows: TARGET_TAP_ROWS, waves: 2, prefill_rows: None, slots: 1, marks: 1, max_context: 524_288,
            pool_tokens: Some(524_288), graphs: true, ffn_mode: cuteafd_ffi::KimiFfnMode::default(),
            hosted_layers: Some(hosted_layers),
            draft: Some(Options { snapshot: drafter, fp8: true, cache_format }), verify_rows: 0,
        })?;
        let engine = &loaded.engine;
        let backend = KimiBackend::new(engine)?;
        let family = engine.prefix_family()?;
        let layout = family.layout();
        let mut pool = RefPagePool::new(layout.pages, layout.page_rows);
        let draft = engine.native_draft().context("native proof drafter absent")?;
        let prefix_error = |error: cuteafd_engine::prefix::BoxError| anyhow::anyhow!("{error}");
        // Cross the 4096-row sliding ring and retain a partial full-cache page.
        let tokens = (0..4163).map(|i| 1000 + (i * 37 % 4096) as u32).collect::<Vec<_>>();
        let suffix = [2201, 2202, 2203];
        let extent = (tokens.len() + suffix.len() + 16).div_ceil(layout.page_rows);
        let allocate = |pool: &mut RefPagePool| -> Result<Placement> {
            Ok(Placement { units: pool.alloc(extent)?, slot: 0, len: 0, state_len: 0 })
        };
        let bind = |placement: &mut Placement, resume| -> Result<()> {
            backend.bind_request(placement, (), RequestMedia::new(Vec::new(), cfg.hidden, tokens.len())?,
                &tokens, resume)
        };
        let prefill = |placement: &mut Placement, seq: &mut DraftState, start, ids: &[u32], pool: &RefPagePool| -> Result<u32> {
            let mut next = None;
            for (index, chunk) in ids.chunks(TARGET_TAP_ROWS).enumerate() {
                let logits = backend.prefill_device(placement, chunk, 1, pool)?.context("native prefill logits absent")?;
                let values = engine.library().cuda_set_device(1).and_then(|_| logits.to_host(engine.library()))?;
                next = Some(values.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0 as u32);
                backend.prefill_draft(placement, start + index * TARGET_TAP_ROWS, chunk, next, seq)?;
            }
            next.context("native proof empty prefill")
        };
        // Retain real MLA pages and an exact KDA mark. The saved last page is never appended to.
        let mut saved = allocate(&mut pool)?;
        family.bind(&mut saved).map_err(prefix_error)?;
        family.admit_draft_pages(&mut saved, extent * layout.page_rows).map_err(prefix_error)?;
        bind(&mut saved, 0)?;
        let mut seq = DraftState::default();
        prefill(&mut saved, &mut seq, 0, &tokens, &pool)?;
        family.capture(MarkSlot(0), &saved, tokens.len()).map_err(prefix_error)?;
        let attachment = family.capture_draft_pages(&saved, tokens.len()).map_err(prefix_error)?
            .context("native warm proof did not capture draft pages")?;
        assert_eq!(family.capture_meta(&saved, tokens.len()).map_err(prefix_error)?.draft, Some((0, tokens.len())));
        family.drain().map_err(prefix_error)?;
        family.discard(&mut saved).map_err(prefix_error)?;
        let mut states = Vec::new();
        let mut proposals = Vec::new();
        let mut verifies = Vec::new();
        let mut common_verify = Vec::new();
        let mut warm_states = Vec::new();
        for arm in 0..4 {
            let mut placement = allocate(&mut pool)?;
            family.bind(&mut placement).map_err(prefix_error)?;
            let mut seq = DraftState::default();
            if arm == 0 || arm == 3 {
                for (index, (&from, &to)) in saved.units.iter().zip(&placement.units).enumerate() {
                    let rows = tokens.len().saturating_sub(index * layout.page_rows).min(layout.page_rows);
                    if rows > 0 { family.copy_rows(TailCopy { from, to, rows, owner: None }).map_err(prefix_error)?; }
                }
                family.restore(Some(MarkSlot(0)), &mut placement, tokens.len()).map_err(prefix_error)?;
                family.restore_draft_pages((arm == 3).then_some(&attachment), &mut placement, tokens.len())
                    .map_err(prefix_error)?;
                family.admit_draft_pages(&mut placement, extent * layout.page_rows).map_err(prefix_error)?;
                bind(&mut placement, tokens.len())?;
                backend.restore_draft(&placement, tokens.len(), &tokens, &mut seq)?;
            } else {
                family.admit_draft_pages(&mut placement, extent * layout.page_rows).map_err(prefix_error)?;
                bind(&mut placement, 0)?;
                prefill(&mut placement, &mut seq, 0, &tokens, &pool)?;
                if arm == 1 {
                    // Missing/revoked draft pages must erase the warm binding to cold(P).
                    family.restore_draft_pages(None, &mut placement, tokens.len()).map_err(prefix_error)?;
                    family.admit_draft_pages(&mut placement, extent * layout.page_rows).map_err(prefix_error)?;
                    bind(&mut placement, tokens.len())?;
                    seq = DraftState::default();
                }
            }
            if arm < 2 {
                let cold = draft.readback(0, 0)?;
                assert_eq!(cold.frontier, (tokens.len(), tokens.len()));
                assert!(cold.layers.iter().all(Vec::is_empty));
            }
            let anchor = prefill(&mut placement, &mut seq, tokens.len(), &suffix, &pool)?;
            if arm < 2 { states.push(draft.readback(0, suffix.len())?); }
            else { warm_states.push(draft.readback(0, suffix.len())?); }
            let proposed = backend.draft(&mut [scheduler::DraftSeq {
                placement: &placement, seq: &mut seq, depth: 7 }], &mut scheduler::DraftTiming::default())?.remove(0);
            assert_eq!(proposed.len(), 7);
            if arm == 0 { common_verify = std::iter::once(anchor).chain(proposed.iter().copied()).collect(); }
            else { assert_eq!(common_verify[0], anchor); }
            proposals.push(proposed);
            let start = placement.len;
            backend.probe_begin(common_verify.len());
            let logits = backend.verify_device(&mut [(&mut placement, &common_verify)], true, &pool)?
                .context("native verify logits absent")?;
            library.cuda_set_device(1)?;
            let values = logits.to_host(&library)?;
            backend.probe_end();
            verifies.push(values.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
            // Keep only the anchor: rejected provisional rows must not enter the next context.
            placement.len = start + 1;
            backend.accept(&mut [scheduler::Verified { placement: &mut placement, seq: &mut seq,
                start, history: (), rows: &common_verify, first_row: 0,
                kept: Some((1, common_verify[1])), finished: false, neural: true }], true)?;
            if arm < 2 { states.push(draft.readback(0, 1)?); }
            else { warm_states.push(draft.readback(0, 1)?); }
            backend.cancel(&placement)?;
            family.drain().map_err(prefix_error)?;
            family.discard(&mut placement).map_err(prefix_error)?;
            pool.release(&placement.units);
        }
        assert_eq!(states[0], states[2], "restored/fresh cold(P) suffix cache and tap bytes");
        assert_eq!(states[1], states[3], "restored/fresh cold(P) accepted-only cache and tap bytes");
        assert_eq!(proposals[0], proposals[1], "first seven draft tokens after cold(P)");
        assert_eq!(verifies[0], verifies[1], "restored versus fresh cold(P) target verification bits");
        assert_eq!(verifies[0], verifies[2], "both cold arms versus fresh no-snapshot target verification bits");
        assert_eq!(warm_states[0], warm_states[2], "warm restored/scratch suffix cache and tap bytes");
        assert_eq!(warm_states[1], warm_states[3], "warm restored/scratch accepted-only cache and tap bytes");
        assert_eq!(proposals[2], proposals[3], "warm restored/scratch proposals");
        assert_eq!(verifies[2], verifies[3], "warm restored/scratch target logits");
        family.release_draft_pages(&attachment).map_err(prefix_error)?;
        pool.release(&saved.units);
        Ok(())
    }

    #[test]
    fn common_drafter_drops_before_borrowed_target_owner() {
        struct Owner(&'static str, Rc<RefCell<Vec<&'static str>>>);
        impl Drop for Owner { fn drop(&mut self) { self.1.borrow_mut().push(self.0); } }
        let drops = Rc::new(RefCell::new(Vec::new()));
        drop(Owners { draft: Owner("draft", drops.clone()), target: Owner("head", drops.clone()) });
        assert_eq!(*drops.borrow(), ["draft", "head"]);
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn local_bridge_drains_without_peer_waits_and_closes_after_failure() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = crate::shared::memory::device::Device { library: &library, id: 0 };
        fixture.immediate_copies(&library)?;
        let bridge = Bridge::Local { stream: Stream::new(device)?, terminal: Cell::new(TerminalState::Active) };
        let target = Stream::new(device)?;
        let source = Allocation::new(device, 64)?;
        let destination = Allocation::new(device, 32)?;
        let bytes = (0u8..64).collect::<Vec<_>>();
        device.run(|| library.copy_h2d(source.buffer, &bytes))?;
        let mut producer = crate::shared::memory::device::Event::new(device)?;
        let mut consumed = crate::shared::memory::device::Event::new(device)?;
        let view = crate::shared::prefix::view(source.buffer, 16, 32)?;
        let before = fixture.events()?.len();
        producer.record(&target)?;
        bridge.checked(|| bridge.copy_local(destination.buffer, view, 32, Some(&producer)))?;
        consumed.record(bridge.stream())?;
        device.run(|| {
            // SAFETY: the test retains both streams and the ordering event through drain.
            unsafe { library.cuda_stream_wait_event(target.raw, consumed.raw) }
        })?;
        assert_eq!(&fixture.events()?[before..], "RWCRW");
        bridge.synchronize()?;
        let mut actual = vec![0; 32];
        device.run(|| library.copy_d2h(&mut actual, destination.buffer))?;
        assert_eq!(actual, bytes[16..48]);
        let before = fixture.events()?;
        let foreign = CuteafdDeviceBuffer { device_id: 1, ..view };
        assert!(bridge.copy_local(destination.buffer, foreign, 32, None).is_err());
        assert!(bridge.copy_local(destination.buffer, view, 33, None).is_err());
        assert_eq!(fixture.events()?, before, "invalid extents must not enqueue work");
        bridge.checked(|| Ok(()))?;
        bridge.finish_request()?;
        assert_eq!(bridge.stream().device.id, 0);
        let error: Result<()> = bridge.checked(|| anyhow::bail!("injected draft submission failure"));
        assert!(error.is_err());
        assert!(bridge.require_live().is_err());
        bridge.synchronize()?;
        bridge.finish_terminal(true);
        assert!(matches!(&bridge, Bridge::Local { terminal, .. } if terminal.get() == TerminalState::Drained));
        bridge.abort()?;
        assert!(bridge.checked(|| Ok(())).is_err());
        Ok(())
    }
}
