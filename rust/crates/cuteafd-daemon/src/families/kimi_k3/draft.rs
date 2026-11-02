//! Kimi's binding of the shared admitted MLA DFlash executor and remote bridge.
use super::weights::GlobalWeights;
use crate::shared::{draft::remote::{Prepared, RemoteBridge}, fp8_linear::{Fp8Rows, Fp8Scales},
    memory::device::{Allocation, DeviceOwner, Stream}, peer_split::RankDevice,
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

pub(crate) struct NativeDraft<'a> {
    owners: Owners<DeviceOwner<'a, DflashDrafter<'a>>, Option<[Rc<GlobalWeights<'a>>; 2]>>,
    taps: Allocation<'a>,
    ids: Allocation<'a>,
    input: Allocation<'a>,
    // Bridge controls and its stream outlive all common/target staging owners.
    bridge: RemoteBridge<'a>,
    target_streams: [Rc<Stream<'a>>; 2],
    producer: RefCell<crate::shared::memory::device::Event<'a>>,
    consumed: RefCell<crate::shared::memory::device::Event<'a>>,
    context_pending: Cell<bool>,
    window_marks: Option<Rc<Allocation<'a>>>,
    window_bytes: usize,
    /// A K6 target head (no BF16 operand to borrow): the target rank's K6
    /// programs and this drafter's own head scratch, used only on the bridge stream.
    k6_head: Option<(Rc<super::device::NativePrograms<'a>>, crate::shared::dense_trellis::ProjectionScratch<'a>)>,
    cache_format: cuteafd_core::MlaCacheFormat,
    marks: usize,
}

impl<'a> NativeDraft<'a> {
    pub(crate) fn load(admission: &mut super::admission::Admission<'a>,
        programs: &'a Programs<'a>, options: &Options, streams: [Rc<Stream<'a>>; 2],
        layout: &MlaDraftRuntimeLayout) -> Result<Self> {
        ensure!(options.cache_format == layout.cache_format(), "Kimi draft options/admission cache records differ");
        let ranks = streams.each_ref().map(|s| RankDevice { device: s.device.id, stream: s.raw });
        let library = streams[0].device.library;
        let prepared = Prepared::new(library, ranks[0], ranks[1], 1, "kimi-remote-mla-dflash")?;
        let file = std::fs::read(options.snapshot.join("model.safetensors"))?;
        let draft = admission.allocate_draft(prepared.stream.device, programs, &options.snapshot, file,
            prepared.stream.raw, options.representation(), Fp8Scales::Amax, Fp8Rows::W8a16)?;
        let demands = bridge_demands(layout, admission.lane_rows())?;
        let device = streams[0].device;
        let allocate = |name| -> Result<Allocation<'a>> {
            let demand = demands.iter().find(|d| d.group == name).context("Kimi draft bridge inventory missing")?;
            Allocation::new(device, usize::try_from(demand.bytes)?)
        };
        let marks = usize::try_from(admission.mark_selection.slots)?;
        let window_bytes = layout.window_mark_bytes()?;
        let window_marks = admission.allocate_draft_marks(streams[1].device)?;
        Ok(Self { window_marks, window_bytes, k6_head: None, cache_format: layout.cache_format(), marks, owners: Owners { draft, target: None }, taps: allocate("DFlash bridge taps")?,
            ids: allocate("DFlash bridge input ids")?, input: allocate("DFlash bridge gathered input")?,
            bridge: RemoteBridge::new(prepared), producer: RefCell::new(crate::shared::memory::device::Event::new(device)?),
            consumed: RefCell::new(crate::shared::memory::device::Event::new(streams[1].device)?),
            context_pending: Cell::new(false), target_streams: streams })
    }

    pub(crate) fn cache_format(&self) -> cuteafd_core::MlaCacheFormat { self.cache_format }

    /// `head_programs` are the head rank's native programs; with a K6 head the
    /// drafter projects through them into its own admitted head scratch.
    pub(crate) fn bind_target(&mut self, target: [Rc<GlobalWeights<'a>>; 2],
        head_programs: Rc<super::device::NativePrograms<'a>>) -> Result<()> {
        ensure!(self.owners.target.is_none(), "Kimi draft target already bound");
        for (rank, owner) in target.iter().enumerate() { owner.check_device(self.target_streams[rank].device)?; }
        let cfg = &self.owners.draft.cfg;
        ensure!(target[0].buffer("embedding")?.bytes == cfg.vocab * cfg.hidden * 2,
            "Kimi draft target embedding differs from admitted BF16 geometry");
        if let Some(head) = &target[1].core_head {
            ensure!(head.shape == [cfg.vocab, cfg.hidden], "Kimi draft K6 target head geometry differs");
            ensure!(head_programs.device.id == self.target_streams[1].device.id,
                "Kimi draft K6 head programs are not on the head rank");
            let projection = head_programs.core.as_ref().context("K6 head programs not resolved")?.projection(head)?;
            let rows = self.owners.draft.max_batch_sequences() * cfg.block;
            let scratch = {
                let _scope = cuteafd_ffi::memory_ledger::scope("drafter/kimi K6 head scratch");
                crate::shared::dense_trellis::ProjectionLayout::new(&projection.program, rows)?
                    .allocate(self.target_streams[1].device)?
            };
            self.k6_head = Some((head_programs, scratch));
        } else {
            ensure!(target[1].buffer("head")?.bytes == cfg.vocab * cfg.hidden * 2,
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
        self.target_streams[0].device.run(|| {
            // SAFETY: all target operands and this exact stream survive the draft's drain.
            unsafe {
                draft.device.library.copy_h2d_ordered(self.ids.buffer, &data, self.bridge.exchange.stream(0))?;
                draft.device.library.cuda_embedding_lookup_bf16_async(table, ids, output, tokens.len(),
                    draft.cfg.vocab, draft.cfg.hidden, self.bridge.exchange.stream(0))
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
                self.bridge.exchange.on(0, || {
                    // SAFETY: retained event joins this exact producer and consumer.
                    unsafe { draft.device.library.cuda_stream_wait_event(self.bridge.exchange.stream(0),
                        self.producer.borrow().raw) }
                })?;
                // SAFETY: checked source frontier and rebased destination fit their owners.
                let plane = unsafe { cuteafd_ffi::peer_exchange::PeerPlane {
                    source: self.taps.buffer.ptr.cast::<u8>().add(first * pitch).cast(),
                    destination: draft.taps.buffer.ptr,
                    rows: (last - first + 1) as u64, row_bytes: pitch as u64,
                    source_pitch: pitch as u64, destination_pitch: pitch as u64,
                } };
                self.bridge.publish_context(&[plane])
            })?;
            self.bridge.acknowledge_context()?;
            self.consumed.borrow_mut().record(&self.bridge.stream)?;
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
    pub(crate) fn abort(&self) -> Result<()> { self.bridge.publish_abort() }
    pub(crate) fn drain(&self) -> Result<()> { self.bridge.synchronize() }
    pub(crate) fn finish_terminal(&self, drained: bool) { self.bridge.finish_terminal(drained); }

    pub(crate) fn propose(&self, sequences: &[DraftSeq]) -> Result<Vec<Draft>> {
        let draft = &self.owners.draft;
        let target = self.owners.target.as_ref().context("Kimi draft target is not bound")?;
        // The reference is tied to this adapter's retained owner, not to a copied
        // buffer descriptor or a synthetic legacy allocation.
        self.bridge.checked(|| draft.device.run(|| {
            let input = |tokens: &[u32], output: CuteafdDeviceBuffer, stream| {
                ensure!(stream == self.bridge.stream.raw, "Kimi draft input stream differs from bridge");
                self.gather(tokens)?;
                self.bridge.send_input(self.input.buffer.ptr.cast_const(), output.ptr,
                    tokens.len() * draft.cfg.hidden * 2)
            };
            let head = match &self.k6_head {
                Some((programs, scratch)) => {
                    let weight = target[1].core_head.as_ref().context("Kimi draft K6 target head absent")?;
                    let projection = programs.core.as_ref().context("K6 head programs not resolved")?.projection(weight)?;
                    let device = self.target_streams[1].device.id;
                    let (hidden, vocab) = (draft.cfg.hidden, draft.cfg.vocab);
                    TargetHead::Launch(Box::new(move |x, logits, rows, stream| {
                        ensure!(stream == self.bridge.stream.raw, "Kimi draft K6 head stream differs from bridge");
                        let input = CuteafdDeviceBuffer { ptr: x.cast_mut(), bytes: rows * hidden * 2, device_id: device, flags: 0 };
                        let output = CuteafdDeviceBuffer { ptr: logits.cast(), bytes: rows * vocab * 4, device_id: device, flags: 0 };
                        // The drafter's own scratch, used only on this bridge stream; the
                        // target head weights are retained by the bound target owner.
                        projection.local(rows, input, weight, scratch, output, stream)
                    }))
                }
                None => TargetHead::Bf16Buffer(target[1].borrowed_buffer("head")?),
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
            rtx_expert_layers: cuteafd_loader::placement::Onboard::Layers(local_layers), spark_budget_gib: None,
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
    fn k6_target_head_projects_through_draft_owned_scratch_not_a_bf16_borrow() {
        let source = include_str!("draft.rs").split("\n#[cfg(test)]\nmod tests").next().unwrap();
        let bind = source.split("pub(crate) fn bind_target").nth(1).unwrap().split("fn gather").next().unwrap();
        // The BF16 head is required only when the target head is not K6.
        let (k6, bf16) = bind.split_once("} else {").unwrap();
        assert!(k6.contains("if let Some(head) = &target[1].core_head"));
        assert!(!k6.contains("buffer(\"head\")") && bf16.contains("buffer(\"head\")"));
        assert!(k6.contains("ProjectionLayout::new(&projection.program, rows)") && k6.contains(".allocate(self.target_streams[1].device)"));
        let propose = source.split("pub(crate) fn propose").nth(1).unwrap().split("impl crate::shared").next().unwrap();
        assert!(propose.contains("TargetHead::Launch") && propose.contains("stream == self.bridge.stream.raw"));
        assert!(propose.contains("None => TargetHead::Bf16Buffer(target[1].borrowed_buffer(\"head\")?)"));
    }
}
