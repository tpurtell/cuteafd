//! Private dSpark inputs from the target pass, in exact flattened request order.
use crate::families::deepseek_v41::v41_backbone_cache::CacheBatch;
use crate::families::deepseek_v41::v41_backbone_router::ExpertRow;
use crate::families::deepseek_v41::v41_block::PreparedBlockInput;
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Result};
use std::rc::Rc;
use crate::shared::memory::row_window::{RowWindow, MAX_RESERVE_ROWS};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41AttentionOps};

const DFLY2_RECALL_LAYERS: [usize; 3] = [28, 32, 36];
const DFLY2_RECALL_ROW_BYTES: usize = 512 * 4;

#[derive(Default)]
struct Progress {
    batch: Option<u64>,
    next: usize,
}
impl Progress {
    fn check(&self, batch: u64, layer: usize) -> Result<()> {
        ensure!(
            self.batch == Some(batch) && self.next < 3 && layer == 37 + self.next,
            "target taps require this batch's layers 37, 38, 39 in order"
        );
        Ok(())
    }
    fn check_dfly2(&self, batch: u64, layer: usize) -> Result<()> {
        ensure!(
            self.batch == Some(batch) && self.next < 3 && layer == DFLY2_RECALL_LAYERS[self.next],
            "DFly2 recall taps require this batch's layers 28, 32, 36 in order"
        );
        Ok(())
    }
    fn ready(&self, batch: u64) -> Result<()> {
        ensure!(
            self.batch == Some(batch) && self.next == 3,
            "target taps incomplete or foreign"
        );
        Ok(())
    }
}

/// Shared taps: values is a row plane, retained across three hooks and read
/// by dSpark on another stream. No scratch or pinned staging. Batch progress
/// and row descriptors belong to each slot.
pub(crate) struct TargetTapStorage<'a> {
    stream: LoadStream<'a>,
    values: DeviceAllocation<'a>,
    rows: usize,
    unit: usize,
}
impl Drop for TargetTapStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: owned stream and values; callers drain external readers.
        crate::shared::decode_graph::fatal_drain(unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        }, "target tap storage");
    }
}
pub(crate) struct TargetTapWave<'a> {
    storage: Rc<TargetTapStorage<'a>>,
    window: RowWindow,
    ops: V41AttentionOps<'a>,
    capacity: usize,
    rows: Vec<ExpertRow>,
    progress: Progress,
    dfly2_mode: bool,
}
/// Borrowed proposed main context. Consume it before committing or discarding
/// the target pass; raw device views must not outlive this borrow.
pub(crate) struct TargetTaps<'a> {
    values: CuteafdDeviceBuffer,
    rows: &'a [ExpertRow],
    batch: u64,
}
impl TargetTaps<'_> {
    pub fn values(&self) -> CuteafdDeviceBuffer {
        self.values
    }
    pub fn rows(&self) -> &[ExpertRow] {
        self.rows
    }
    pub fn batch_identity(&self) -> u64 {
        self.batch
    }
}
impl<'a> TargetTapWave<'a> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid target tap capacity"
        );
        Ok(capacity * 30720)
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, budget: usize) -> Result<Self> {
        let bytes = Self::device_bytes(capacity)?;
        ensure!(bytes <= budget, "target taps exceed device budget");
        Self::on_storage(Self::storage(library, capacity, capacity)?)
    }
    pub(crate) fn storage(library: &'a NativeLibrary, rows: usize, unit: usize)
        -> Result<Rc<TargetTapStorage<'a>>> {
        ensure!((1..=MAX_RESERVE_ROWS).contains(&rows) && (1..=4096).contains(&unit)
            && unit <= rows, "invalid target tap storage dimensions");
        Ok(Rc::new(TargetTapStorage {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            values: DeviceAllocation::new(library, rows * 30720, Tag::workspace("v41/target-taps"))?, rows, unit,
        }))
    }
    pub(crate) fn on_storage(storage: Rc<TargetTapStorage<'a>>) -> Result<Self> {
        Ok(Self { ops: storage.stream.library.v41_attention_ops()?, capacity: storage.unit,
            rows: Vec::with_capacity(storage.unit), storage, window: RowWindow::NONE,
            progress: Progress::default(), dfly2_mode: false })
    }
    pub(crate) fn set_dfly2_mode(&mut self, enabled: bool) {
        self.dfly2_mode = enabled;
    }
    pub(crate) fn is_dfly2_mode(&self) -> bool {
        self.dfly2_mode
    }
    /// All readers of the old slot must drain before selecting its next unit.
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows
                && range.len() <= self.capacity, "target tap window outside storage or unit");
        }
        self.reset();
        self.window = window;
        Ok(())
    }
    pub(super) fn reset(&mut self) {
        self.progress = Progress::default();
        self.rows.clear();
    }
    pub(super) fn begin(&mut self, batch: &CacheBatch) -> Result<()> {
        self.reset();
        let rows = batch.expert_rows();
        ensure!(
            !rows.is_empty() && rows.len() <= self.capacity && self.window.admits(rows.len()),
            "target tap batch exceeds capacity"
        );
        self.rows = rows;
        self.progress.batch = Some(batch.identity());
        Ok(())
    }
    /// # Safety
    /// Input belongs to the target driver's current batch and is fully produced.
    /// This method drains its read before attention can overwrite the residual.
    pub(super) unsafe fn capture(
        &mut self,
        batch: &CacheBatch,
        input: &PreparedBlockInput<'_>,
    ) -> Result<()> {
        let queued = unsafe { self.enqueue_capture(batch, input) };
        let drained = if queued.is_err() { unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw) } }
            else { unsafe { crate::shared::memory::chain::finish(self.storage.stream.library, self.storage.stream.raw) } };
        if let Err(error) = queued.and(drained) { self.reset(); return Err(error); }
        self.progress.next += 1;
        Ok(())
    }
    /// # Safety
    /// Retain prepared input until the tap read completes or cancellation drains.
    pub(super) async unsafe fn capture_cooperative(&mut self, batch: &CacheBatch,
        input: &PreparedBlockInput<'_>) -> Result<()> {
        unsafe { self.enqueue_capture(batch, input)?; }
        if let Err(error) = unsafe { crate::shared::memory::chain::finish_cooperative(&self.storage.stream).await } {
            self.reset(); return Err(error);
        }
        self.progress.next += 1;
        Ok(())
    }
    unsafe fn enqueue_capture(&mut self, batch: &CacheBatch, input: &PreparedBlockInput<'_>) -> Result<()> {
        let result = (|| -> Result<()> {
            self.progress.check(batch.identity(), input.layer)?;
            ensure!(
                input.previous_binding().layer() + 1 == input.layer
                    && input.tokens.len() == self.rows.len()
                    && input
                        .tokens
                        .iter()
                        .zip(&self.rows)
                        .all(|(&p, row)| p == row.position)
                    && input.residual.bytes == self.rows.len() * 40960
                    && input.residual.device_id == self.window.apply(self.storage.values.buffer, 30720).device_id,
                "target tap prepared input differs from its batch"
            );
            #[cfg(test)]
            if let Some(dir) = std::env::var_os("CUTEAFD_TARGET_PASS_OUTPUT") {
                let dir = std::path::PathBuf::from(dir);
                std::fs::create_dir_all(&dir)?;
                let mut bytes = vec![0; input.residual.bytes];
                self.storage.stream.library.copy_d2h(&mut bytes, input.residual)?;
                std::fs::write(
                    dir.join(format!(
                        "batch{}-layer{}-tap-input.bin",
                        batch.identity(),
                        input.layer
                    )),
                    bytes,
                )?;
            }
            // SAFETY: prepared input is retained by the caller through drain.
            let launched = unsafe { self.enqueue_values(input.residual, input.layer) };
            if let Err(error) = launched {
                unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw)?; }
                return Err(error);
            }
            Ok(())
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }
    /// # Safety
    /// Residual and the selected output window remain live through the chain.
    unsafe fn enqueue_values(&self, residual: CuteafdDeviceBuffer, layer: usize) -> Result<()> {
        // SAFETY: input ownership is the caller's contract; values is a row plane.
        unsafe {
            crate::shared::memory::chain::join(self.storage.stream.library, self.storage.stream.raw)?;
            self.ops.tap(residual, self.window.apply(self.storage.values.buffer, 30720),
                self.rows.len() as u32, layer as u32, self.storage.stream.raw)
        }
    }
    pub(super) fn output(&self, batch: &CacheBatch) -> Result<TargetTaps<'_>> {
        self.progress.ready(batch.identity())?;
        let mut values = self.window.apply(self.storage.values.buffer, 30720);
        values.bytes = self.rows.len() * 30720;
        Ok(TargetTaps {
            values,
            rows: &self.rows,
            batch: batch.identity(),
        })
    }
    fn dfly2_stage_buffer(&self, stage: usize) -> Result<CuteafdDeviceBuffer> {
        let base = self.window.apply(self.storage.values.buffer, 30720);
        let stage_bytes = self.rows.len() * DFLY2_RECALL_ROW_BYTES;
        let offset = stage * stage_bytes;
        ensure!(
            offset + stage_bytes <= base.bytes,
            "DFly2 recall tap stage {stage} exceeds target tap window"
        );
        Ok(CuteafdDeviceBuffer {
            ptr: unsafe { base.ptr.cast::<u8>().add(offset).cast() },
            bytes: stage_bytes,
            ..base
        })
    }
    unsafe fn enqueue_dfly2_recall(
        &mut self,
        batch: &CacheBatch,
        layer: usize,
        selected: CuteafdDeviceBuffer,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            ensure!(self.dfly2_mode, "DFly2 recall capture requires dfly2_mode");
            self.progress.check_dfly2(batch.identity(), layer)?;
            let stage = self.progress.next;
            let dst = self.dfly2_stage_buffer(stage)?;
            ensure!(
                selected.bytes >= dst.bytes && selected.device_id == dst.device_id,
                "DFly2 recall selection buffer differs from target tap window ({} vs {}, dev {} vs {})",
                selected.bytes, dst.bytes, selected.device_id, dst.device_id
            );
            unsafe {
                crate::shared::memory::chain::join(self.storage.stream.library, self.storage.stream.raw)?;
                self.storage.stream.library.copy_d2d_async(dst, selected, dst.bytes, self.storage.stream.raw)?;
            }
            Ok(())
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }
    /// # Safety
    /// `selected` belongs to the current batch's completed decoder reindex for `layer ∈ {28, 32, 36}`.
    pub(super) unsafe fn capture_dfly2_recall(
        &mut self,
        batch: &CacheBatch,
        layer: usize,
        selected: CuteafdDeviceBuffer,
    ) -> Result<()> {
        let queued = unsafe { self.enqueue_dfly2_recall(batch, layer, selected) };
        let drained = if queued.is_err() {
            unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw) }
        } else {
            unsafe { crate::shared::memory::chain::finish(self.storage.stream.library, self.storage.stream.raw) }
        };
        if let Err(error) = queued.and(drained) {
            self.reset();
            return Err(error);
        }
        self.progress.next += 1;
        Ok(())
    }
    /// # Safety
    /// Same ownership contract as [`Self::capture_dfly2_recall`].
    pub(super) async unsafe fn capture_dfly2_recall_cooperative(
        &mut self,
        batch: &CacheBatch,
        layer: usize,
        selected: CuteafdDeviceBuffer,
    ) -> Result<()> {
        unsafe { self.enqueue_dfly2_recall(batch, layer, selected)?; }
        if let Err(error) = unsafe { crate::shared::memory::chain::finish_cooperative(&self.storage.stream).await } {
            self.reset();
            return Err(error);
        }
        self.progress.next += 1;
        Ok(())
    }
    pub(super) fn dfly2_recall_rows(&self, batch: &CacheBatch) -> Result<([CuteafdDeviceBuffer; 3], &[ExpertRow])> {
        ensure!(self.dfly2_mode, "DFly2 recall rows require dfly2_mode");
        self.progress.ready(batch.identity())?;
        Ok((
            [
                self.dfly2_stage_buffer(0)?,
                self.dfly2_stage_buffer(1)?,
                self.dfly2_stage_buffer(2)?,
            ],
            &self.rows,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::Progress;
    #[test]
    fn taps_require_complete_ordered_batch() {
        let mut progress = Progress {
            batch: Some(7),
            next: 0,
        };
        assert!(progress.ready(7).is_err());
        assert!(progress.check(8, 37).is_err());
        assert!(progress.check(7, 38).is_err());
        for layer in 37..40 {
            progress.check(7, layer).unwrap();
            progress.next += 1;
        }
        progress.ready(7).unwrap();
        assert!(progress.ready(8).is_err());
        assert!(progress.check(7, 39).is_err());
        progress = Progress::default();
        assert!(progress.ready(7).is_err());
    }
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and a GPU"]
    fn windowed_taps_match_private_runs_behind_a_gated_stream() -> Result<()> {
        // SAFETY: library outlives all streams and allocations in this test.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let (reserve, unit, rows) = (48, 16, 13);
        let storage = TargetTapWave::storage(&library, reserve, unit)?;
        let mut slots = [TargetTapWave::on_storage(storage.clone())?,
            TargetTapWave::on_storage(storage.clone())?];
        let sources = (0..6).map(|seed| -> Result<DeviceAllocation<'_>> {
            let allocation = DeviceAllocation::new(&library, rows * 40960, crate::shared::memory::TEST)?;
            let host: Vec<u8> = (0..rows * 20480).flat_map(|i| {
                let value = ((i * 7 + seed * 11) % 29) as f32 / 32.0 - 0.4;
                ((value.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            library.copy_h2d(allocation.buffer, &host)?;
            Ok(allocation)
        }).collect::<Result<Vec<_>>>()?;
        let row = |position| ExpertRow { request_id: 42, position,
            kind: cuteafd_transport::ExpertV2SourceKind::Prefill };
        for (seed, slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::rows(seed * unit..seed * unit + rows))?;
            slot.rows = (0..rows).map(|i| row(i as u64)).collect();
        }
        // Warm the exact taps on these streams before any gate: first-use CUDA
        // module loading may synchronize the context. Poison after warm-up.
        for layer in 37..40 {
            for seed in 0..2 {
                // SAFETY: initialized sources and selected planes stay owned.
                unsafe { slots[seed].enqueue_values(sources[(layer - 37) * 2 + seed].buffer, layer)?;
                    crate::shared::memory::chain::finish(&library, storage.stream.raw)?; }
            }
        }
        library.copy_h2d(storage.values.buffer, &vec![0xa5; storage.values.buffer.bytes])?;
        let chain = crate::shared::memory::chain::StageChain::new(&library)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        runtime.block_on(chain.handle().scope(async {
            // SAFETY: shared stream remains live until gate release and drain.
            let gate = unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&library, storage.stream.raw)? };
            for layer in 37..40 {
                for seed in 0..2 {
                    // SAFETY: all input/output allocations are retained until drain.
                    unsafe { slots[seed].enqueue_values(sources[(layer - 37) * 2 + seed].buffer, layer)?;
                        crate::shared::memory::chain::finish(&library, storage.stream.raw)?; }
                }
            }
            drop(gate);
            anyhow::Ok(())
        }))?;
        chain.drain()?;
        let mut entire = vec![0; storage.values.buffer.bytes];
        library.copy_d2h(&mut entire, storage.values.buffer)?;
        for seed in 0..2 {
            let mut private = TargetTapWave::new(&library, unit, usize::MAX)?;
            private.rows = (0..rows).map(|i| row(i as u64)).collect();
            for layer in 37..40 {
                // SAFETY: private inputs are initialized and retained through drain.
                unsafe { private.enqueue_values(sources[(layer - 37) * 2 + seed].buffer, layer)?;
                    crate::shared::memory::chain::finish(&library, private.storage.stream.raw)?; }
            }
            let mut expected = vec![0; rows * 30720];
            library.copy_d2h(&mut expected, private.storage.values.buffer)?;
            assert_eq!(entire[seed * unit * 30720..(seed * unit + rows) * 30720], expected);
        }
        for gap in [rows..unit, unit + rows..reserve] {
            assert!(entire[gap.start * 30720..gap.end * 30720].iter().all(|&v| v == 0xa5),
                "tap wrote outside selected windows");
        }
        Ok(())
    }
}
