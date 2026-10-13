//! One GPU1 DFlash2 owner; the target retains its single embedding and head.
use crate::families::glm5::dflash::{ContextRow, Draft, DraftSeq, GlmDrafter, ReplayDrafter, TargetHead, TAP_ROWS};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event, Stream};
use crate::shared::peer_split::{PeerExchange, RankDevice, TerminalState};
use crate::shared::token_io::TokenEmbedding;
use anyhow::{ensure, Result};
use cuteafd_ffi::peer_exchange::PeerPlane;
use cuteafd_ffi::programs::VocabularyHead;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::cell::Cell;
use std::ffi::c_void;

const TAPS: usize = 0;
const INPUT: usize = 1;
const HIDDEN: usize = 2;
const LOGITS: usize = 3;

fn tap_frontier(rows: &[ContextRow]) -> Result<Option<(usize, usize)>> {
    let Some(first) = rows.iter().map(|r| r.tap_row).min() else { return Ok(None) };
    let last = rows.iter().map(|r| r.tap_row).max().unwrap();
    ensure!(last < TAP_ROWS, "remote tap row {last} past {TAP_ROWS}");
    Ok(Some((first, last)))
}

pub(crate) struct Prepared<'a> {
    pub exchange: PeerExchange<'a>,
    pub stream: Stream<'a>,
    target_peer: RankDevice,
    target_ready: Event<'a>,
}

impl<'a> Prepared<'a> {
    pub fn new(library: &'a NativeLibrary, target: RankDevice, target_peer: RankDevice) -> Result<Self> {
        ensure!(target.device != target_peer.device && !target_peer.stream.is_null(),
            "remote draft requires the peer target producer stream");
        let device = Device { library, id: target_peer.device };
        let stream = Stream::new(device)?;
        let target_ready = Event { device, raw: device.run(|| library.cuda_event_create_ordering())? };
        let exchange = PeerExchange::new_abortable(library,
            [target, RankDevice { device: target_peer.device, stream: stream.raw }], 4, 0)?;
        exchange.trace_identity("glmf-remote-dflash:taps,input,hidden,logits");
        Ok(Self { exchange, stream, target_peer, target_ready })
    }

    /// Sampled before admission but charged only for the selected GPU1 candidate.
    pub fn sampled_bytes(&self) -> u64 { 4 * 256 + 256 + 256 }
}

struct TargetBuffers<'a> {
    taps: Allocation<'a>,
    ids: Allocation<'a>,
    input: Allocation<'a>,
    hidden: Allocation<'a>,
    logits: Allocation<'a>,
    head: VocabularyHead<'a>,
    _head_workspace: Allocation<'a>,
}

pub(crate) struct RemoteDflash<'a> {
    pub draft: DeviceOwner<'a, GlmDrafter<'a>>,
    target: DeviceOwner<'a, TargetBuffers<'a>>,
    exchange: PeerExchange<'a>,
    target_peer: RankDevice,
    target_ready: Event<'a>,
    // The drafter, exchange and event drop before the stream they borrow.
    stream: Stream<'a>,
    rows: usize,
    update_pending: Cell<bool>,
}

impl<'a> RemoteDflash<'a> {
    pub fn new(draft: DeviceOwner<'a, GlmDrafter<'a>>, prepared: Prepared<'a>) -> Result<Self> {
        let library = draft.device.library;
        ensure!(draft.device.id == prepared.stream.device.id, "remote draft owner disagrees with prepared stream");
        let c = &draft.cfg;
        let rows = draft.max_batch_sequences().checked_mul(c.block)
            .ok_or_else(|| anyhow::anyhow!("remote draft rows overflow"))?;
        let inventory = cuteafd_loader::placement::families::glm5_flash::remote_dflash_bridge_demands(
            c.hidden as u64, c.vocab as u64, c.taps.len() as u64, rows as u64)?;
        let device = Device { library, id: prepared.exchange.device(0) };
        let target = device.own(|| {
            let _scope = cuteafd_ffi::memory_ledger::scope("drafter/bridge");
            let alloc = |name| -> Result<Allocation<'a>> {
                let bytes = inventory.demands.iter().find(|d| d.group == name).unwrap().bytes;
                Allocation::new(device, usize::try_from(bytes)?)
            };
            let workspace = alloc("DFlash bridge head scratch")?;
            // SAFETY: this owner retains the scratch until after the head drops.
            let head = unsafe { library.vocabulary_head_rows(workspace.buffer.ptr,
                c.hidden as u32, rows as u32, c.vocab as u32)? };
            Ok(TargetBuffers { taps: alloc("DFlash bridge taps")?, ids: alloc("DFlash bridge input ids")?,
                input: alloc("DFlash bridge gathered input")?, hidden: alloc("DFlash bridge normalized hidden")?,
                logits: alloc("DFlash bridge logits")?, head, _head_workspace: workspace })
        })?;
        draft.device.run(|| draft.prepare_workspace())?;
        Ok(Self { draft, target, exchange: prepared.exchange, target_peer: prepared.target_peer,
            target_ready: prepared.target_ready, stream: prepared.stream, rows, update_pending: Cell::new(false) })
    }

    /// All draft and target-head first-use work finishes without peer waits.
    pub fn prime(&self, embedding: &TokenEmbedding<'_>, head: TargetHead<'_>) -> Result<()> {
        self.checked(|| {
            self.draft.device.run(|| self.draft.prime_remote())?;
            self.target.device.run(|| {
                let library = self.draft.device.library;
                library.cuda_zero_bytes(self.target.taps.buffer, self.target.taps.buffer.bytes)?;
                library.cuda_zero_bytes(self.target.hidden.buffer, self.target.hidden.buffer.bytes)?;
                for count in 1..=self.draft.max_batch_sequences() {
                    let rows = count * self.draft.cfg.block;
                    embedding.embed(&vec![self.draft.cfg.mask_token; rows], self.target.ids.buffer, 1,
                        self.target.input.buffer, self.exchange.stream(0))?;
                    self.launch_head(&head, rows, true)?;
                    self.launch_head(&head, rows, false)?;
                    // SAFETY: retire gathers before synchronous ids reuse on the
                    // next width; this target stream has no peer wait during priming.
                    unsafe { library.cuda_stream_synchronize(self.exchange.stream(0))?; }
                }
                Ok(())
            })
        })
    }

    fn launch_head(&self, head: &TargetHead<'_>, rows: usize, serving: bool) -> Result<()> {
        self.target.device.run(|| match head {
            TargetHead::Bf16(weight) if serving => {
                // SAFETY: target-owned weight and admitted scratch stay live on this stream.
                unsafe { crate::shared::draft::launch_bf16_head(&self.target.head,
                    self.target.hidden.buffer.ptr.cast(), weight.buffer.ptr.cast(),
                    self.target.logits.buffer.ptr.cast(), rows, self.exchange.stream(0)) }
            }
            TargetHead::Bf16(weight) => {
                // SAFETY: target head and admitted operands share the target stream.
                unsafe { crate::families::glm5::launch_draft_head(self.draft.device.library,
                    &self.target.head, self.target.hidden.buffer.ptr.cast_const(), weight.buffer.ptr.cast_const(),
                    self.target.logits.buffer.ptr.cast(), rows, self.draft.cfg.hidden, self.draft.cfg.vocab,
                    self.exchange.stream(0), self.draft.draft_head_mode()) }
            }
            TargetHead::Launch(launch) => launch(self.target.hidden.buffer.ptr.cast_const(),
                self.target.logits.buffer.ptr.cast(), rows, self.exchange.stream(0)),
        })
    }

    pub(crate) fn replay_write_ranges(&self) -> Vec<(i32, String, CuteafdDeviceBuffer)> {
        let mut ranges: Vec<_> = self.draft.replay_write_ranges().into_iter()
            .map(|(name, buffer)| (self.draft.device.id, name, buffer)).collect();
        for (name, buffer) in [("taps", &self.target.taps), ("ids", &self.target.ids),
            ("input", &self.target.input), ("hidden", &self.target.hidden),
            ("logits", &self.target.logits), ("head_workspace", &self.target._head_workspace)] {
            ranges.push((self.target.device.id, format!("bridge/{name}"), buffer.buffer));
        }
        ranges
    }

    pub fn require_live(&self) -> Result<()> { self.exchange.require_live() }

    pub fn publish_abort(&self) -> Result<()> {
        match self.exchange.terminal_state() {
            TerminalState::Aborting | TerminalState::Drained => Ok(()),
            _ => self.exchange.publish_abort(),
        }
    }

    /// No owner is released on an unproved drain; the engine's fatal-drain
    /// boundary also retains every target graph and peer-exchange operand.
    pub fn synchronize(&self) -> Result<()> {
        // Consume the final context acknowledgement at this drained boundary,
        // not only on the next update. Aborted waits also retire here.
        if self.update_pending.get() && self.exchange.terminal_state() == TerminalState::Active {
            self.exchange.wait(0, TAPS)?;
            self.target.device.run(|| {
                // SAFETY: the target stream and all context operands remain owned.
                unsafe { self.draft.device.library.cuda_stream_synchronize(self.exchange.stream(0)) }
            })?;
            self.update_pending.set(false);
        }
        self.stream.drain()?;
        if self.exchange.terminal_state() == TerminalState::Aborting { self.update_pending.set(false); }
        Ok(())
    }

    pub fn finish_terminal(&self, drained: bool) {
        if self.exchange.terminal_state() == TerminalState::Aborting {
            self.exchange.finish_terminal(drained);
        }
    }

    fn checked<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        self.require_live()?;
        let result = work();
        if result.is_err() {
            crate::shared::decode_graph::fatal_drain(self.publish_abort(), "DFlash bridge abort publication");
        }
        result
    }

    /// Target graph segments write only their own staging. Remote work is
    /// enqueued by `update`, never while a rank0 segment is being captured.
    pub fn tap_streams_at(&self, layer: usize, streams: *const c_void, hc: usize, first: usize,
        n: usize, to: usize) -> Result<()> {
        let Some(index) = self.draft.tap_index(layer) else { return Ok(()) };
        let c = &self.draft.cfg;
        let width = c.taps.len() * c.hidden;
        ensure!(to.checked_add(n).is_some_and(|end| end <= TAP_ROWS), "remote tap rows exceed {TAP_ROWS}");
        self.checked(|| self.exchange.on(0, || {
            // SAFETY: target rows are final on its stream; admitted local tap
            // storage outlives every captured graph and its queued reader.
            unsafe { self.draft.device.library.glm_dflash_tap_mean(
                streams.cast::<u8>().add(first * hc * c.hidden * 2).cast(),
                self.target.taps.buffer.ptr.cast::<u8>().add(to * width * 2).cast(),
                n, c.hidden, hc, width, index * c.hidden, self.exchange.stream(0)) }
        }))
    }

    pub(crate) fn validate_target_peer(&self, producer: RankDevice) -> Result<()> {
        ensure!(self.target_peer.device == producer.device && self.target_peer.stream == producer.stream
            && self.target_ready.device.id == producer.device
            && self.stream.device.id == producer.device && self.exchange.stream(1) == self.stream.raw,
            "resident remote peer waits must be ordered behind the cooperative target producer");
        Ok(())
    }

    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        let Some((first, last)) = tap_frontier(rows)? else { return Ok(()) };
        let pitch = self.draft.cfg.taps.len() * self.draft.cfg.hidden * 2;
        self.checked(|| {
            // Conservatively retire the paired bridge before publishing the
            // next context wave; metadata uploads themselves are stream-ordered.
            self.synchronize()?;
            self.draft.device.run(|| self.draft.update_with(rows, || {
                if self.update_pending.get() { self.exchange.wait(0, TAPS)?; }
                // SAFETY: the checked row frontier is inside both tap owners.
                let plane = unsafe { PeerPlane {
                    source: self.target.taps.buffer.ptr.cast::<u8>().add(first * pitch).cast(),
                    destination: self.draft.taps.buffer.ptr.cast::<u8>().add(first * pitch).cast(),
                    rows: (last - first + 1) as u64, row_bytes: pitch as u64,
                    source_pitch: pitch as u64, destination_pitch: pitch as u64,
                } };
                self.exchange.push_planes(0, TAPS, &[plane])?;
                // Both target ranks have submitted this prefill chunk or decode
                // round. A resident waiter must not steal the SM needed by the
                // peer target's full-grid cooperative EXL3 launch. Wait on its
                // completion event before starting the TAPS spinner, with no host
                // sync. GPU0 publishes TAPS N before its chunk-N+1 waits, so this
                // spinner retires before GPU1 needs its next cooperative grid.
                self.validate_target_peer(self.target_peer)?;
                self.stream.device.run(|| unsafe {
                    // SAFETY: both same-device streams and the reusable ordering
                    // event survive every queued update; the preceding wave drained.
                    self.draft.device.library.cuda_event_record(self.target_ready.raw, self.target_peer.stream)?;
                    self.draft.device.library.cuda_stream_wait_event(self.stream.raw, self.target_ready.raw)
                })?;
                self.exchange.wait(1, TAPS)
            }))?;
            // Acknowledgement means all context readers are finished, not just
            // that the incoming taps arrived. The next overwrite waits for it.
            self.exchange.signal(1, TAPS)?;
            self.update_pending.set(true);
            Ok(())
        })
    }

    pub fn put_taps(&self, rows: &[u8]) -> Result<()> {
        self.checked(|| {
            ensure!(rows.len() <= self.target.taps.buffer.bytes, "remote replay taps past staging capacity");
            self.synchronize()?;
            self.target.device.run(|| self.draft.device.library.copy_h2d(
                CuteafdDeviceBuffer { bytes: rows.len(), ..self.target.taps.buffer }, rows))
        })
    }

    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>,
        head: TargetHead<'_>) -> Result<Vec<Draft>> {
        self.draft_device_from(sequences, embedding, head, true)
    }

    pub fn draft_device_diagnostic(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>,
        head: TargetHead<'_>) -> Result<Vec<Draft>> {
        self.draft_device_from(sequences, embedding, head, false)
    }

    fn draft_device_from(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>,
        head: TargetHead<'_>, serving: bool) -> Result<Vec<Draft>> {
        self.checked(|| {
            let input = |ids: &[u32], out: CuteafdDeviceBuffer, stream: *mut c_void| {
                ensure!(stream == self.stream.raw && ids.len() <= self.rows, "remote draft input capacity/stream");
                self.target.device.run(|| embedding.embed(ids, self.target.ids.buffer, 1,
                    self.target.input.buffer, self.exchange.stream(0)))?;
                self.exchange.push_to(0, INPUT, self.target.input.buffer.ptr.cast_const(), out.ptr,
                    ids.len() * embedding.hidden() * 2)?;
                self.exchange.wait(1, INPUT)
            };
            let launch = |hidden: *const c_void, logits: *mut f32, rows: usize, stream: *mut c_void| {
                ensure!(stream == self.stream.raw && rows <= self.rows, "remote draft head capacity/stream");
                self.exchange.push_to(1, HIDDEN, hidden, self.target.hidden.buffer.ptr,
                    rows * self.draft.cfg.hidden * 2)?;
                self.exchange.wait(0, HIDDEN)?;
                self.launch_head(&head, rows, serving)?;
                self.exchange.push_to(0, LOGITS, self.target.logits.buffer.ptr.cast_const(), logits.cast(),
                    rows * self.draft.cfg.vocab * 4)?;
                self.exchange.wait(1, LOGITS)
            };
            self.draft.device.run(|| self.draft.draft_device_with(sequences, &input,
                TargetHead::Launch(Box::new(launch))))
        })
    }
}

impl Drop for RemoteDflash<'_> {
    fn drop(&mut self) {
        // No complete owner (including target graph operands) may release its
        // storage unless the engine and this extra compute stream drained.
        crate::shared::decode_graph::fatal_drain(self.synchronize(), "remote DFlash owner");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::peer_split::order::{self, Schedule};

    fn row(tap_row: usize) -> ContextRow { ContextRow { tap_row, slot: 0, position: tap_row } }

    #[test]
    fn tap_frontier_is_empty_or_bounded_and_keeps_holes() {
        assert_eq!(tap_frontier(&[]).unwrap(), None);
        assert_eq!(tap_frontier(&[row(5), row(2), row(5)]).unwrap(), Some((2, 5)));
        assert_eq!(tap_frontier(&[row(TAP_ROWS - 1)]).unwrap(), Some((TAP_ROWS - 1, TAP_ROWS - 1)));
        assert!(tap_frontier(&[row(TAP_ROWS)]).is_err());
    }

    fn update(schedule: &mut Schedule, pending: bool) {
        if pending { schedule.wait(0, "draft", TAPS, "previous context readers"); }
        schedule.push(0, "draft", TAPS, "target taps final");
        schedule.wait(1, "draft", TAPS, "context taps arrived");
        schedule.push(1, "draft", TAPS, "context readers finished");
    }

    fn draft(schedule: &mut Schedule) {
        schedule.push(0, "draft", INPUT, "target input gathered");
        schedule.wait(1, "draft", INPUT, "draft input arrived");
        schedule.push(1, "draft", HIDDEN, "draft hidden normalized");
        schedule.wait(0, "draft", HIDDEN, "target head input arrived");
        schedule.push(0, "draft", LOGITS, "target head finished");
        schedule.wait(1, "draft", LOGITS, "draft selector logits arrived");
    }

    #[test]
    fn repeated_context_and_head_bridges_have_no_wait_cycle() {
        for rounds in [1, 2, 8, 32] {
            let mut schedule = Schedule::default();
            for round in 0..rounds {
                update(&mut schedule, round > 0);
                // Target-only rounds still update the drafter's context.
                if round % 3 != 1 { draft(&mut schedule); }
            }
            schedule.wait(0, "draft", TAPS, "request boundary context readers");
            schedule.end_request();
            assert_eq!(order::check(&schedule), Ok(()));
        }
    }

    #[test]
    fn head_wait_before_input_publication_is_detected() {
        let mut schedule = Schedule::default();
        schedule.wait(0, "draft", HIDDEN, "head before input");
        draft(&mut schedule);
        assert!(order::check(&schedule).is_err());
    }

    /// Component transport/ownership gate, not a target-model quality gate.
    #[test]
    #[ignore = "requires both RTX GPUs, native CUDA library and explicit checkpoint paths"]
    fn remote_dflash_cuda_matches_local_and_drains_terminal_waits() -> Result<()> {
        use crate::families::glm5::dflash::{self, DflashConfig};
        use crate::shared::fp8_linear::{Fp8Rows, Fp8Scales};
        use crate::shared::memory::DeviceAllocation;
        use crate::shared::token_io::{EmbedPlacement, EmbedSource};
        use cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation;
        use cuteafd_loader::plan::checkpoint::Checkpoint;
        use std::os::unix::fs::FileExt;
        use std::path::PathBuf;

        let target_path = PathBuf::from(std::env::var("CUTEAFD_REMOTE_DFLASH_TARGET")?);
        let draft_path = PathBuf::from(std::env::var("CUTEAFD_REMOTE_DFLASH_CHECKPOINT")?);
        // SAFETY: this library outlives every fixture stream and device owner.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let device = Device { library: &library, id: 0 };
        let lead = Stream::new(device)?;
        let peer_target = Stream::new(Device { library: &library, id: 1 })?;
        let cfg = DflashConfig::read(&draft_path)?;
        let inventory = Checkpoint::open_for_role(&target_path, "remote-draft-component", |name|
            name.ends_with("embed_tokens.weight") || name == "lm_head.weight")?;
        let embedding_tensor = inventory.tensors.iter().find(|t| t.meta.name.ends_with("embed_tokens.weight"))
            .ok_or_else(|| anyhow::anyhow!("target embedding tensor missing"))?;
        let head_tensor = inventory.tensors.iter().find(|t| t.meta.name == "lm_head.weight")
            .ok_or_else(|| anyhow::anyhow!("target head tensor missing"))?;
        ensure!(head_tensor.meta.shape == [cfg.vocab, cfg.hidden]
            && head_tensor.meta.dtype == cuteafd_core::DType::Bf16, "component target head geometry/dtype");
        let (embedding, ()) = TokenEmbedding::load(&library, EmbedSource::new(&target_path,
            &embedding_tensor.shard, &embedding_tensor.meta, cfg.hidden)?, EmbedPlacement::Gpu, || Ok(()))?;
        let head = DeviceAllocation::new(&library, usize::try_from(head_tensor.meta.byte_length)?)?;
        let mut head_bytes = vec![0; head.buffer.bytes];
        std::fs::File::open(target_path.join(&head_tensor.shard))?
            .read_exact_at(&mut head_bytes, head_tensor.meta.byte_offset)?;
        library.copy_h2d(head.buffer, &head_bytes)?;
        drop(head_bytes);
        let file = dflash::prefetch(&draft_path).join().map_err(|_| anyhow::anyhow!("fixture draft read panicked"))??;
        let mask = embedding.host_rows(&[cfg.mask_token])?;
        for representation in [GlmDraftRepresentation::Fp8Only, GlmDraftRepresentation::Bf16Only] {
            let prepared = Prepared::new(&library, RankDevice { device: 0, stream: lead.raw },
                RankDevice { device: 1, stream: peer_target.raw })?;
            let local = GlmDrafter::load(&library, &draft_path, file.clone(), lead.raw, 4, 4,
                mask.clone(), true, representation, Fp8Scales::Pow2, Fp8Rows::W8a8)?;
            let draft_device = prepared.stream.device;
            let draft = draft_device.own(|| GlmDrafter::load(&library, &draft_path, file.clone(),
                prepared.stream.raw, 4, 4, mask.clone(), true, representation, Fp8Scales::Pow2, Fp8Rows::W8a8))?;
            let remote = RemoteDflash::new(draft, prepared)?;
            let result = (|| -> Result<()> {
                remote.prime(&embedding, TargetHead::Bf16(&head))?;
                let width = cfg.taps.len() * cfg.hidden;
                for round in 0..8 {
                    let count = [1, 2, 4, 2][round % 4];
                    let position = 8 + round;
                    // Nonzero BF16 taps and holes exercise frontier copies, ring
                    // reuse and sequence-count changes without a target executor.
                    let taps: Vec<u8> = (0..8 * width).flat_map(|i| {
                        let value = if (i + round) % 3 == 0 { 0x3d80u16 } else { 0xbd00u16 };
                        value.to_le_bytes()
                    }).collect();
                    let context: Vec<_> = (0..count).map(|slot| ContextRow {
                        tap_row: 1 + slot * 2, slot, position: position - 1 }).collect();
                    let sequences: Vec<_> = (0..count).map(|slot| DraftSeq { slot,
                        anchor: (1 + slot + round) as u32, position, valid_from: 7 }).collect();
                    local.put_taps(&taps)?;
                    local.update(&context)?;
                    lead.drain()?;
                    let expected = local.draft_device(&sequences, &embedding, TargetHead::Bf16(&head))?;
                    let expected_hidden = local.last_hidden(count)?;
                    remote.put_taps(&taps)?;
                    remote.update(&context)?;
                    let actual = remote.draft_device(&sequences, &embedding, TargetHead::Bf16(&head))?;
                    let actual_hidden = remote.last_hidden(count)?;
                    ensure!(expected_hidden == actual_hidden, "remote hidden mismatch {representation:?} round {round}");
                    for (expected, actual) in expected.iter().zip(&actual) {
                        ensure!(expected.tokens == actual.tokens, "remote token mismatch {representation:?} round {round}");
                        let bits = |draft: &Draft| draft.features.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>();
                        ensure!(bits(expected) == bits(actual), "remote selector mismatch {representation:?} round {round}");
                    }
                    lead.drain()?;
                }
                // Queue genuine unmatched CUDA waits on both sides, then fail
                // validation before metadata uploads or any blocking drain.
                remote.exchange.wait(0, HIDDEN)?;
                remote.exchange.wait(1, INPUT)?;
                for rank in 0..2 {
                    remote.exchange.on(rank, || {
                        // SAFETY: query observes live fixture-owned streams.
                        ensure!(!unsafe { library.cuda_stream_query(remote.exchange.stream(rank))? },
                            "fixture wait unexpectedly completed on rank {rank}");
                        Ok(())
                    })?;
                }
                ensure!(remote.put_taps(&vec![0; remote.target.taps.buffer.bytes + 1]).is_err(),
                    "invalid remote staging must fail");
                ensure!(remote.require_live().is_err(), "terminal remote bridge became reusable");
                lead.drain()?;
                remote.synchronize()?;
                remote.finish_terminal(true);
                ensure!(remote.exchange.terminal_state() == TerminalState::Drained,
                    "remote bridge did not finish terminal drainage");
                eprintln!("remote_dflash_component representation={representation:?} rounds=8 hidden_exact=true tokens_exact=true selector_exact=true terminal_drained=true target_quality=false");
                Ok(())
            })();
            if result.is_err() {
                crate::shared::decode_graph::fatal_drain(remote.publish_abort(), "remote component abort");
            }
            crate::shared::decode_graph::fatal_drain(lead.drain(), "remote component target");
            crate::shared::decode_graph::fatal_drain(remote.synchronize(), "remote component draft");
            result?;
        }
        Ok(())
    }
}

impl ReplayDrafter for RemoteDflash<'_> {
    fn block(&self) -> usize { self.draft.cfg.block }
    fn sequences(&self) -> usize { self.draft.max_batch_sequences() }
    fn resident_modes(&self) -> Vec<crate::families::glm5::dflash::ReplayMode> { self.draft.resident_modes() }
    fn context(&self, taps: &[u8], first: usize) -> Result<()> {
        let width = self.draft.cfg.taps.len() * self.draft.cfg.hidden * 2;
        ensure!(taps.len() % width == 0, "remote replay tap row width");
        self.put_taps(taps)?;
        self.update(&(0..taps.len() / width).map(|r| ContextRow { tap_row: r, slot: 0,
            position: first + r }).collect::<Vec<_>>())?;
        self.synchronize()
    }
    fn draft_tokens(&self, _: &[(usize, u32, usize)], _: &[u8], _: *const c_void) -> Result<Vec<Vec<u32>>> {
        anyhow::bail!("GPU1 DFlash replay requires the target-owned bridge callback")
    }
    fn tap_rows(&self) -> usize { TAP_ROWS }
    fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        self.draft.device.run(|| self.draft.last_hidden(sequences))
    }
}
