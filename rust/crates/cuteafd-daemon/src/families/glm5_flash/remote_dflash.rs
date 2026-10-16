//! One GPU1 DFlash2 owner; the target retains its single embedding and head.
//! GLM Flash's binding of the shared device-driven draft bridge
//! (`shared::draft::remote`): taps, input rows, hidden rows and logits.
use crate::families::glm5::dflash::{ContextRow, Draft, DraftSeq, GlmDrafter, ReplayDrafter, TargetHead, TAP_ROWS};
use crate::shared::draft::remote::{Prepared, RemoteBridge};
use crate::shared::memory::device::{Allocation, Device, DeviceOwner};
use crate::shared::peer_split::RankDevice;
use crate::shared::token_io::TokenEmbedding;
use anyhow::{ensure, Result};
use cuteafd_ffi::peer_exchange::PeerPlane;
use cuteafd_ffi::programs::VocabularyHead;
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::ffi::c_void;

pub(crate) const IDENTITY: &str = "glmf-remote-dflash:taps,input,hidden,logits";

fn tap_frontier(rows: &[ContextRow]) -> Result<Option<(usize, usize)>> {
    let Some(first) = rows.iter().map(|r| r.tap_row).min() else { return Ok(None) };
    let last = rows.iter().map(|r| r.tap_row).max().unwrap();
    ensure!(last < TAP_ROWS, "remote tap row {last} past {TAP_ROWS}");
    Ok(Some((first, last)))
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
    // The drafter drops before the bridge's stream it borrows.
    bridge: RemoteBridge<'a>,
    rows: usize,
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
        let bridge = RemoteBridge::new(prepared);
        let device = Device { library, id: bridge.target_device() };
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
        Ok(Self { draft, target, bridge, rows })
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
                        self.target.input.buffer, self.bridge.exchange.stream(0))?;
                    self.launch_head(&head, rows, true)?;
                    self.launch_head(&head, rows, false)?;
                    // SAFETY: retire gathers before synchronous ids reuse on the
                    // next width; this target stream has no peer wait during priming.
                    unsafe { library.cuda_stream_synchronize(self.bridge.exchange.stream(0))?; }
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
                    self.target.logits.buffer.ptr.cast(), rows, self.bridge.exchange.stream(0)) }
            }
            TargetHead::Bf16(weight) => {
                // SAFETY: target head and admitted operands share the target stream.
                unsafe { crate::families::glm5::launch_draft_head(self.draft.device.library,
                    &self.target.head, self.target.hidden.buffer.ptr.cast_const(), weight.buffer.ptr.cast_const(),
                    self.target.logits.buffer.ptr.cast(), rows, self.draft.cfg.hidden, self.draft.cfg.vocab,
                    self.bridge.exchange.stream(0), self.draft.draft_head_mode()) }
            }
            TargetHead::Launch(launch) => launch(self.target.hidden.buffer.ptr.cast_const(),
                self.target.logits.buffer.ptr.cast(), rows, self.bridge.exchange.stream(0)),
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

    pub fn require_live(&self) -> Result<()> { self.bridge.require_live() }

    pub fn publish_abort(&self) -> Result<()> { self.bridge.publish_abort() }

    pub fn synchronize(&self) -> Result<()> { self.bridge.synchronize() }

    pub fn finish_request(&self) -> Result<()> { self.bridge.finish_request() }

    pub fn finish_terminal(&self, drained: bool) { self.bridge.finish_terminal(drained) }

    fn checked<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> { self.bridge.checked(work) }

    /// Target graph segments write only their own staging. Remote work is
    /// enqueued by `update`, never while a rank0 segment is being captured.
    pub fn tap_streams_at(&self, layer: usize, streams: *const c_void, hc: usize, first: usize,
        n: usize, to: usize) -> Result<()> {
        let Some(index) = self.draft.tap_index(layer) else { return Ok(()) };
        let c = &self.draft.cfg;
        let width = c.taps.len() * c.hidden;
        ensure!(to.checked_add(n).is_some_and(|end| end <= TAP_ROWS), "remote tap rows exceed {TAP_ROWS}");
        self.checked(|| self.bridge.exchange.on(0, || {
            // SAFETY: target rows are final on its stream; admitted local tap
            // storage outlives every captured graph and its queued reader.
            unsafe { self.draft.device.library.glm_dflash_tap_mean(
                streams.cast::<u8>().add(first * hc * c.hidden * 2).cast(),
                self.target.taps.buffer.ptr.cast::<u8>().add(to * width * 2).cast(),
                n, c.hidden, hc, width, index * c.hidden, self.bridge.exchange.stream(0)) }
        }))
    }

    pub(crate) fn validate_target_peer(&self, producer: RankDevice) -> Result<()> {
        self.bridge.validate_target_peer(producer)
    }

    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        let Some((first, last)) = tap_frontier(rows)? else { return Ok(()) };
        let pitch = self.draft.cfg.taps.len() * self.draft.cfg.hidden * 2;
        self.checked(|| {
            self.draft.device.run(|| self.draft.update_with(rows, || {
                // SAFETY: the checked row frontier is inside both tap owners.
                let plane = unsafe { PeerPlane {
                    source: self.target.taps.buffer.ptr.cast::<u8>().add(first * pitch).cast(),
                    destination: self.draft.taps.buffer.ptr.cast::<u8>().add(first * pitch).cast(),
                    rows: (last - first + 1) as u64, row_bytes: pitch as u64,
                    source_pitch: pitch as u64, destination_pitch: pitch as u64,
                } };
                // Waits for the previous acknowledgement, pushes, and orders the
                // draft-side wait behind the cooperative peer target producer.
                self.bridge.publish_context(&[plane])
            }))?;
            self.bridge.acknowledge_context()
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
                ensure!(stream == self.bridge.stream.raw && ids.len() <= self.rows, "remote draft input capacity/stream");
                self.target.device.run(|| embedding.embed(ids, self.target.ids.buffer, 1,
                    self.target.input.buffer, self.bridge.exchange.stream(0)))?;
                self.bridge.send_input(self.target.input.buffer.ptr.cast_const(), out.ptr,
                    ids.len() * embedding.hidden() * 2)
            };
            let launch = |hidden: *const c_void, logits: *mut f32, rows: usize, stream: *mut c_void| {
                ensure!(stream == self.bridge.stream.raw && rows <= self.rows, "remote draft head capacity/stream");
                self.bridge.round_trip_head(hidden, self.target.hidden.buffer.ptr,
                    rows * self.draft.cfg.hidden * 2, || self.launch_head(&head, rows, serving),
                    self.target.logits.buffer.ptr.cast_const(), logits.cast(), rows * self.draft.cfg.vocab * 4)
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
    use crate::shared::draft::remote::{HIDDEN, INPUT};
    use crate::shared::memory::device::Stream;
    use crate::shared::peer_split::TerminalState;
    use cuteafd_ffi::NativeLibrary;

    fn row(tap_row: usize) -> ContextRow { ContextRow { tap_row, slot: 0, position: tap_row } }

    #[test]
    fn tap_frontier_is_empty_or_bounded_and_keeps_holes() {
        assert_eq!(tap_frontier(&[]).unwrap(), None);
        assert_eq!(tap_frontier(&[row(5), row(2), row(5)]).unwrap(), Some((2, 5)));
        assert_eq!(tap_frontier(&[row(TAP_ROWS - 1)]).unwrap(), Some((TAP_ROWS - 1, TAP_ROWS - 1)));
        assert!(tap_frontier(&[row(TAP_ROWS)]).is_err());
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
                RankDevice { device: 1, stream: peer_target.raw }, IDENTITY)?;
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
                remote.bridge.exchange.wait(0, HIDDEN)?;
                remote.bridge.exchange.wait(1, INPUT)?;
                for rank in 0..2 {
                    remote.bridge.exchange.on(rank, || {
                        // SAFETY: query observes live fixture-owned streams.
                        ensure!(!unsafe { library.cuda_stream_query(remote.bridge.exchange.stream(rank))? },
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
                ensure!(remote.bridge.exchange.terminal_state() == TerminalState::Drained,
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
