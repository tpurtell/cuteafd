//! Unit-owned Engram row windows, serial layer gathers and H2D source fences.
pub(crate) mod layer;
pub(crate) mod placement;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use crate::shared::memory::row_window::{RowWindow, ScratchGuard, MAX_RESERVE_ROWS};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{EngramGatherLease, EngramGatherPoll, EngramGatherTicket, EngramGatherView, EngramPipeline, EngramWave};
use std::rc::Rc;

#[cfg(test)]
thread_local! {
    static SKIP_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Every buffer persists across the upload and gate hooks: these are row planes,
/// not shared scratch. Slots select disjoint windows for the whole unit.
pub(crate) struct EngramRowsStorage<'a> {
    stream: LoadStream<'a>,
    weights: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    embeddings: DeviceAllocation<'a>,
    text_mask: DeviceAllocation<'a>,
    rows: usize,
    unit: usize,
}
pub(crate) struct EngramDeviceRows<'a> {
    storage: Rc<EngramRowsStorage<'a>>,
    stream: LoadStream<'a>,
    consume_stream: LoadStream<'a>,
    /// Per unit, never shared between units. Protects L1 -> L14 and slot reuse.
    guard: ScratchGuard<'a>,
    post_gate: *mut std::ffi::c_void,
    gate_recorded: std::cell::Cell<bool>,
    completed: [*mut std::ffi::c_void; 2],
    staged: [Option<usize>; 2],
    window: RowWindow,
    /// Only borrowed fixture uploads need a copy; production gathers are pinned.
    staging: Option<crate::shared::memory::LayerStaging<'a>>,
    library: &'a NativeLibrary,
    capacity: usize,
    ready: Option<(usize, usize)>,
}
pub(crate) struct EngramDeviceView {
    pub embeddings: CuteafdDeviceBuffer,
    pub text_mask: CuteafdDeviceBuffer,
    pub rows: usize,
    pub layer_index: usize,
}
pub(crate) enum EngramUploadPoll { Pending, Cancelled, Ready(EngramDeviceView) }
impl<'a> EngramDeviceRows<'a> {
    pub(crate) fn library(&self) -> &'a NativeLibrary { self.library }
    pub fn device_bytes(rows: usize) -> Result<usize> {
        ensure!((1..=MAX_RESERVE_ROWS).contains(&rows), "invalid Engram device rows");
        Ok(rows * cuteafd_loader::ENGRAM_DEVICE_ROW_BYTES)
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, available_bytes: usize) -> Result<Self> {
        ensure!(Self::device_bytes(capacity)? <= available_bytes, "Engram rows exceed device budget");
        Self::on_storage(Self::storage(library, capacity, capacity)?)
    }
    pub(crate) fn storage(library: &'a NativeLibrary, rows: usize, unit: usize) -> Result<Rc<EngramRowsStorage<'a>>> {
        ensure!((1..=MAX_RESERVE_ROWS).contains(&rows) && (1..=4096).contains(&unit) && unit <= rows,
            "invalid Engram row storage dimensions");
        Ok(Rc::new(EngramRowsStorage {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            weights: DeviceAllocation::new(library, rows * 24 * 256)?,
            scales: DeviceAllocation::new(library, rows * 24 * 8)?,
            embeddings: DeviceAllocation::new(library, rows * 24 * 512)?,
            text_mask: DeviceAllocation::new(library, rows)?, rows, unit,
        }))
    }
    pub(crate) fn on_storage(storage: Rc<EngramRowsStorage<'a>>) -> Result<Self> {
        let library = storage.stream.library;
        let capacity = storage.unit;
        Ok(Self { storage, library, capacity, window: RowWindow::NONE, ready: None,
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            consume_stream: LoadStream { library, raw: library.cuda_stream_create()? },
            guard: ScratchGuard::new(library, &[library.cuda_get_device()?])?,
            post_gate: library.cuda_event_create_ordering()?, gate_recorded: std::cell::Cell::new(false),
            completed: [library.cuda_event_create_ordering()?, library.cuda_event_create_ordering()?],
            staged: [None; 2], staging: None })
    }
    /// Caller drained the previous occupant's entire chain before selecting rows.
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows && range.len() <= self.capacity,
                "Engram row window outside storage or unit");
        }
        self.reset()?;
        self.window = window;
        Ok(())
    }
    pub(crate) fn reset(&mut self) -> Result<()> {
        self.synchronize()?;
        self.staged = [None; 2]; self.ready = None; self.gate_recorded.set(false);
        Ok(())
    }
    /// Record immediately after the gate joined this unit's chain. Never park
    /// copy-engine work behind this event: four parked L14 copies block a fresh
    /// L1 copy across streams at the real 2048-row k4 shape. Existing layer polls
    /// observe completion before taking the ready gather lease from its wave.
    pub(crate) fn release_scratch(&self) -> Result<()> {
        self.guard.release()?;
        // SAFETY: own event on this gate's GPU; the chain retains its producer.
        self.gate_recorded.set(unsafe { crate::shared::memory::chain::record_into(self.library, self.post_gate)? });
        Ok(())
    }
    pub(crate) fn can_enqueue(&self, layer: usize) -> Result<bool> {
        ensure!(layer < 2, "invalid Engram layer");
        if !self.gate_recorded.get() { return Ok(true); }
        // SAFETY: own event; recording and querying stay on the CUDA thread.
        Ok(unsafe { self.library.cuda_event_query(self.post_gate)? })
    }
    pub(crate) async fn wait_enqueue(&self, layer: usize, timeout: std::time::Duration) -> Result<()> {
        tokio::time::timeout(timeout, async {
            let mut backoff = cuteafd_transport::wait_policy::Backoff::new();
            while !self.can_enqueue(layer)? { tokio::task::yield_now().await; backoff.pause(); }
            anyhow::Ok(())
        }).await.context("Engram post-gate completion timed out")?
    }
    #[cfg(test)]
    pub(crate) fn test_writer(&self, skip: bool) -> *mut std::ffi::c_void {
        SKIP_ACQUIRE.with(|value| value.set(skip)); self.stream.raw
    }
    unsafe fn acquire_scratch(&self) -> Result<()> {
        #[cfg(test)]
        if SKIP_ACQUIRE.with(|skip| skip.get()) { return Ok(()); }
        // SAFETY: the stream is this unit's live upload stream.
        unsafe { self.guard.acquire(self.stream.raw) }
    }
    fn plane(&self, buffer: CuteafdDeviceBuffer, stride: usize) -> CuteafdDeviceBuffer {
        self.window.apply(buffer, stride)
    }
    fn gathered_buffers(&self, gathered: &EngramGatherView<'_>) -> (CuteafdDeviceBuffer, CuteafdDeviceBuffer) {
        let mut weights = self.plane(self.storage.weights.buffer, 24 * 256);
        let mut scales = self.plane(self.storage.scales.buffer, 24 * 8);
        if gathered.encoding == cuteafd_loader::EngramEncoding::Nvfp4 {
            scales = weights;
            scales.ptr = scales.ptr.cast::<u8>().wrapping_add(gathered.weights.len()).cast();
        }
        weights.bytes = gathered.weights.len(); scales.bytes = gathered.scales.len();
        (weights, scales)
    }
    fn validate(&self, gathered: &EngramGatherView<'_>) -> Result<()> {
        ensure!(gathered.rows > 0 && gathered.rows <= self.capacity && self.window.admits(gathered.rows)
            && gathered.layer_index < 2, "invalid gathered Engram dimensions");
        ensure!(gathered.weights.len() == gathered.rows * 24 * gathered.encoding.weight_bytes()
            && gathered.scales.len() == gathered.rows * 24 * gathered.encoding.scale_bytes()
            && gathered.text_mask.len() == gathered.rows, "invalid gathered Engram storage");
        Ok(())
    }
    /// Queue copies only: the source's event is recorded before dequantization.
    unsafe fn enqueue_copies(&self, gathered: &EngramGatherView<'_>, staging: cuteafd_ffi::CuteafdHostBuffer) -> Result<()> {
        let (weights, scales) = self.gathered_buffers(gathered);
        let mask = self.plane(self.storage.text_mask.buffer, 1);
        let base = staging.bytes / (24 * (256 + 8) + 1);
        // Staging stores fixed-capacity FP8 planes. FP4 scales live in weights.
        let scale_offset = if gathered.encoding == cuteafd_loader::EngramEncoding::Fp8 {
            base * 24 * 256
        } else { base * 24 * 128 };
        let planes = [(weights, 0, gathered.weights.len()),
            (scales, scale_offset, gathered.scales.len()), (mask, base * 24 * (256 + 8), gathered.rows)]
            .map(|(destination, offset, bytes)| crate::shared::token_table::CopyPlane { destination, offset, bytes });
        // SAFETY: the unit owns its destination window and retains the source lease.
        unsafe { crate::shared::token_table::enqueue(self.library, staging, &planes, self.stream.raw)?; }
        Ok(())
    }
    unsafe fn dequantize(&self, gathered: &EngramGatherView<'_>) -> Result<()> {
        let (weights, scales) = self.gathered_buffers(gathered);
        let embeddings = self.plane(self.storage.embeddings.buffer, 24 * 512);
        let rows = i32::try_from(gathered.rows * 24)?;
        // SAFETY: all input copies precede this kernel on the unit's stream.
        unsafe { match gathered.encoding {
            cuteafd_loader::EngramEncoding::Fp8 => self.library.cuda_engram_dequant_bf16_async(weights, scales, embeddings, rows, self.stream.raw),
            cuteafd_loader::EngramEncoding::Nvfp4 => self.library.cuda_engram_nvfp4_dequant_bf16_async(weights, scales, gathered.global_scale, embeddings, rows, self.stream.raw),
        } }
    }
    /// Early upload does not join/publish the unit chain. Only its matching
    /// gate waits for this event, so L14 cannot serialize layers 2..13.
    pub fn enqueue_lease(&mut self, mut lease: EngramGatherLease, wave: &EngramWave) -> Result<()> {
        let result = self.enqueue_source(&mut lease);
        wave.retain_copy(lease);
        result
    }
    fn enqueue_source(&mut self, lease: &mut EngramGatherLease) -> Result<()> {
        let result = (|| -> Result<()> {
            let gathered = lease.view()?;
            self.validate(&gathered)?;
            ensure!(self.can_enqueue(gathered.layer_index)?, "Engram copy cannot precede post-gate completion");
            let staging = lease.host_buffer().context("production Engram gather must be pinned")?;
            // SAFETY: post-gate event is unit-owned; the lease is retained below.
            unsafe { self.acquire_scratch()?; }
            // SAFETY: retained source and unit-owned disjoint destination.
            let copied = unsafe { self.enqueue_copies(&gathered, staging) };
            let (rows, layer) = (gathered.rows, gathered.layer_index);
            // SAFETY: record even after a partial-copy error: earlier reads
            // may have been queued and must not lose their source fence.
            unsafe { lease.copied_on(self.stream.raw)?; }
            copied?;
            // SAFETY: input and output windows remain owned by this slot.
            unsafe { self.dequantize(&lease.view()?)?; self.library.cuda_event_record(self.completed[layer], self.stream.raw)?; }
            self.staged[layer] = Some(rows);
            Ok(())
        })();
        if result.is_err() { crate::shared::decode_graph::fatal_drain(self.synchronize(), "failed Engram upload"); }
        result
    }
    pub fn staged(&self, layer: usize) -> bool { self.staged[layer].is_some() }
    pub fn consume(&mut self, layer: usize) -> Result<EngramDeviceView> {
        let rows = self.staged[layer].context("Engram layer upload absent")?;
        // SAFETY: event belongs to this slot and includes its dequantization.
        unsafe {
            crate::shared::memory::chain::join(self.library, self.consume_stream.raw)?;
            self.library.cuda_stream_wait_event(self.consume_stream.raw, self.completed[layer])?;
            crate::shared::memory::chain::finish(self.library, self.consume_stream.raw)?;
        }
        self.ready = Some((rows, layer)); self.view()
    }
    pub fn poll_wave(&mut self, pipeline: &EngramPipeline, wave: &mut EngramWave,
        histories: &[&cuteafd_core::EngramHistory], layer: usize) -> Result<EngramUploadPoll> {
        wave.reclaim_copies()?;
        if self.staged(layer) { return Ok(EngramUploadPoll::Ready(self.consume(layer)?)); }
        if !self.can_enqueue(layer)? { return Ok(EngramUploadPoll::Pending); }
        Ok(match pipeline.poll(wave, histories, layer)? {
            EngramGatherPoll::Pending => EngramUploadPoll::Pending,
            EngramGatherPoll::Cancelled => EngramUploadPoll::Cancelled,
            EngramGatherPoll::Ready(lease) => {
                self.enqueue_lease(lease, wave)?;
                EngramUploadPoll::Ready(self.consume(layer)?)
            }
        })
    }
    pub fn poll_upload(&mut self, ticket: &mut EngramGatherTicket) -> Result<EngramUploadPoll> {
        Ok(match ticket.poll()? {
            EngramGatherPoll::Pending => EngramUploadPoll::Pending,
            EngramGatherPoll::Cancelled => EngramUploadPoll::Cancelled,
            EngramGatherPoll::Ready(lease) => EngramUploadPoll::Ready(self.upload(&lease.view()?)?),
        })
    }
    /// Borrowed fixture/reference path; production directly uploads the pinned lease.
    pub fn upload(&mut self, gathered: &EngramGatherView<'_>) -> Result<EngramDeviceView> {
        self.validate(gathered)?;
        if self.staging.is_none() { self.staging = Some(crate::shared::memory::LayerStaging::new(
            self.library, cuteafd_loader::EngramBatchStaging::payload_bytes(self.capacity), 2)?); }
        let (host, bytes) = self.staging.as_mut().unwrap().region(gathered.layer_index);
        let scale = if gathered.encoding == cuteafd_loader::EngramEncoding::Fp8 { self.capacity * 24 * 256 }
            else { self.capacity * 24 * 128 };
        bytes[..gathered.weights.len()].copy_from_slice(gathered.weights);
        bytes[scale..scale + gathered.scales.len()].copy_from_slice(gathered.scales);
        let mask = self.capacity * 24 * (256 + 8);
        bytes[mask..mask + gathered.rows].copy_from_slice(gathered.text_mask);
        // SAFETY: borrowed source is copied to slot-owned staging retained until drain.
        unsafe {
            crate::shared::memory::chain::join(self.library, self.stream.raw)?;
            self.acquire_scratch()?; self.enqueue_copies(gathered, host)?; self.dequantize(gathered)?;
            self.library.cuda_event_record(self.completed[gathered.layer_index], self.stream.raw)?;
        }
        self.staged[gathered.layer_index] = Some(gathered.rows);
        self.consume(gathered.layer_index)
    }
    pub async fn upload_cooperative(&mut self, gathered: &EngramGatherView<'_>) -> Result<EngramDeviceView> {
        let view = self.upload(gathered)?;
        if !crate::shared::memory::chain::deferred() { self.stream.wait().await?; }
        Ok(view)
    }
    #[cfg(test)]
    pub async fn check_cooperative(&mut self, gathered: &EngramGatherView<'_>) -> Result<()> {
        let view = self.upload_cooperative(gathered).await?;
        ensure!(view.rows == gathered.rows, "Engram row mismatch"); Ok(())
    }
    pub fn view(&self) -> Result<EngramDeviceView> {
        let (rows, layer_index) = self.ready.context("Engram device rows are not complete")?;
        let mut embeddings = self.plane(self.storage.embeddings.buffer, 24 * 512); embeddings.bytes = rows * 24 * 512;
        let mut text_mask = self.plane(self.storage.text_mask.buffer, 1); text_mask.bytes = rows;
        Ok(EngramDeviceView { embeddings, text_mask, rows, layer_index })
    }
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: this owner retains the stream.
        unsafe {
            self.library.cuda_stream_synchronize(self.stream.raw)?;
            self.library.cuda_stream_synchronize(self.consume_stream.raw)
        }
    }
}
impl Drop for EngramDeviceRows<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.synchronize(), "Engram unit upload");
        for event in self.completed.into_iter().chain(std::iter::once(self.post_gate)) {
            // SAFETY: own events and all producers have drained.
            unsafe { let _ = self.library.cuda_event_destroy(event); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gathered<'b>(weights: &'b [u8], scales: &'b [u8], layer: usize) -> EngramGatherView<'b> {
        EngramGatherView { weights, scales, text_mask: &[1], rows: 1,
            layer_index: layer, encoding: cuteafd_loader::EngramEncoding::Fp8, global_scale: 1.0 }
    }
    fn queue_layer(owner: &mut EngramDeviceRows<'_>, source: &mut cuteafd_ffi::PinnedBytes, layer: usize) -> Result<()> {
        let rows = source.bytes().len() / cuteafd_loader::EngramBatchStaging::payload_bytes(1);
        source.bytes_mut()[..rows * 24 * 256].fill(if layer == 0 { 0x38 } else { 0x40 });
        source.bytes_mut()[rows * 24 * 256..rows * 24 * (256 + 8)].fill(127);
        source.bytes_mut()[rows * 24 * (256 + 8)..].fill(1);
        let weights = vec![if layer == 0 { 0x38 } else { 0x40 }; rows * 24 * 256];
        let scales = vec![127; rows * 24 * 8];
        let text_mask = vec![1; rows];
        let view = EngramGatherView { weights: &weights, scales: &scales, text_mask: &text_mask, rows,
            layer_index: layer, encoding: cuteafd_loader::EngramEncoding::Fp8, global_scale: 1.0 };
        owner.validate(&view)?;
        // SAFETY: test owns source/streams/windows until every chain drains.
        unsafe {
            owner.acquire_scratch()?;
            owner.enqueue_copies(&view, source.buffer())?;
            source.copied_on(owner.stream.raw)?;
            owner.dequantize(&view)?;
            owner.library.cuda_event_record(owner.completed[layer], owner.stream.raw)?;
        }
        owner.staged[layer] = Some(rows);
        Ok(())
    }
    fn wait_event(lib: &NativeLibrary, event: *mut std::ffi::c_void, timeout: std::time::Duration) -> Result<bool> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            // SAFETY: caller retains the event throughout this bounded wait.
            if unsafe { lib.cuda_event_query(event)? } { return Ok(true); }
            if std::time::Instant::now() >= deadline { return Ok(false); }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    fn layer14_ordering(lib: &NativeLibrary, guarded: bool, held: bool) -> Result<()> {
        use crate::shared::memory::chain::StageChain;
        let mut owner = EngramDeviceRows::new(lib, 1, EngramDeviceRows::device_bytes(1)?)?;
        let mut source = lib.pinned_bytes(cuteafd_loader::EngramBatchStaging::payload_bytes(1))?;
        queue_layer(&mut owner, &mut source, 0)?;
        owner.synchronize()?;
        assert!(source.copy_ready()?);
        let read = LoadStream { library: lib, raw: lib.cuda_stream_create()? };
        let observed = DeviceAllocation::new(lib, 24 * 512)?;
        let chain = StageChain::new(lib)?;
        chain.set_device_order(true);
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // Declared after every stream owner so assertion unwinding releases it first.
        // SAFETY: read stream remains live through gate drop.
        let gate = if held { Some(unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(lib, read.raw)? }) } else { None };
        runtime.block_on(chain.handle().scope(async {
            let view = owner.consume(0)?;
            // This read is the gate's buffer consumer, behind the controlled hold.
            // SAFETY: all owners remain live through chain and copy drains.
            unsafe {
                crate::shared::memory::chain::join(lib, read.raw)?;
                lib.copy_d2d_async(observed.buffer, view.embeddings, observed.buffer.bytes, read.raw)?;
                crate::shared::memory::chain::finish(lib, read.raw)?;
            }
            owner.release_scratch()
        }))?;
        owner.test_writer(!guarded);
        let queued = queue_layer(&mut owner, &mut source, 1);
        owner.test_writer(false);
        queued?;
        let done = wait_event(lib, owner.completed[1], std::time::Duration::from_millis(if held && guarded { 100 } else { 3000 }))?;
        if held { assert_eq!(done, !guarded, "L14 completion while L1 gate held, guarded={guarded}"); }
        else { assert!(done, "ungated exact-call warmup"); }
        drop(gate);
        chain.drain()?; owner.synchronize()?;
        assert!(source.copy_ready()?);
        let mut bytes = vec![0; observed.buffer.bytes];
        lib.copy_d2h(&mut bytes, observed.buffer)?;
        let expected = [0x80, 0x3f].repeat(24 * 256);
        if guarded || !held { assert_eq!(bytes, expected, "L1 gate read clobbered"); }
        else { assert_ne!(bytes, expected, "skip-wait control failed to clobber"); }
        Ok(())
    }
    fn consume_does_not_join_copy(lib: &NativeLibrary, wrong_stream: bool, held: bool) -> Result<()> {
        use crate::shared::memory::chain::StageChain;
        let mut owner = EngramDeviceRows::new(lib, 1, EngramDeviceRows::device_bytes(1)?)?;
        let mut source = lib.pinned_bytes(cuteafd_loader::EngramBatchStaging::payload_bytes(1))?;
        queue_layer(&mut owner, &mut source, 0)?;
        owner.synchronize()?; assert!(source.copy_ready()?);
        let hold = LoadStream { library: lib, raw: lib.cuda_stream_create()? };
        let read = LoadStream { library: lib, raw: lib.cuda_stream_create()? };
        let observed = DeviceAllocation::new(lib, 24 * 512)?;
        let fence = lib.cuda_event_create_ordering()?;
        let done = lib.cuda_event_create_ordering()?;
        let chain = StageChain::new(lib)?; chain.set_device_order(true);
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // SAFETY: hold stream remains live, and gate drop always releases pending work.
        let gate = if held { Some(unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(lib, hold.raw)? }) } else { None };
        // SAFETY: the recorded-but-pending event deliberately holds L14's copy.
        unsafe { lib.cuda_event_record(fence, hold.raw)?; lib.cuda_stream_wait_event(owner.stream.raw, fence)?; }
        queue_layer(&mut owner, &mut source, 1)?;
        runtime.block_on(chain.handle().scope(async {
            if wrong_stream {
                // Negative control: the L1 consume join queues behind held L14.
                // SAFETY: deliberate ordering defect, all storage remains retained.
                unsafe {
                    crate::shared::memory::chain::join(lib, owner.stream.raw)?;
                    lib.cuda_stream_wait_event(owner.stream.raw, owner.completed[0])?;
                    crate::shared::memory::chain::finish(lib, owner.stream.raw)?;
                }
            } else { owner.consume(0)?; }
            // SAFETY: read is a unit-chain consumer with retained output storage.
            unsafe {
                crate::shared::memory::chain::join(lib, read.raw)?;
                lib.copy_d2d_async(observed.buffer,
                    owner.plane(owner.storage.embeddings.buffer, 24 * 512), observed.buffer.bytes, read.raw)?;
                lib.cuda_event_record(done, read.raw)?;
                crate::shared::memory::chain::finish(lib, read.raw)?;
            }
            anyhow::Ok(())
        }))?;
        let completed = wait_event(lib, done, std::time::Duration::from_millis(if held && wrong_stream { 100 } else { 3000 }))?;
        assert_eq!(completed, !held || !wrong_stream, "L1 consume join, wrong_stream={wrong_stream}");
        if held {
            // SAFETY: live event recorded on the held upload stream.
            assert!(!unsafe { lib.cuda_event_query(owner.completed[1])? }, "L14 must remain pending");
            if !wrong_stream {
                let mut bytes = vec![0; observed.buffer.bytes];
                lib.copy_d2h(&mut bytes, observed.buffer)?;
                assert_eq!(bytes, [0x80, 0x3f].repeat(24 * 256), "L1 gate must read its own rows");
            }
        }
        drop(gate); chain.drain()?; owner.synchronize()?;
        assert!(source.copy_ready()?);
        // SAFETY: every event producer and waiter has completed.
        unsafe { lib.cuda_event_destroy(fence)?; lib.cuda_event_destroy(done)?; }
        drop(observed);
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA native library, runs via golden-run --native-test"]
    fn engram_serial_upload_gate_and_consume_stream_controls() -> Result<()> {
        // SAFETY: golden-run supplies the sealed matching native library.
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        for device in 0..2 {
            lib.cuda_set_device(device)?;
            layer14_ordering(&lib, true, false)?;
            consume_does_not_join_copy(&lib, false, false)?;
            for repeat in 0..3 {
                layer14_ordering(&lib, true, true)?;
                layer14_ordering(&lib, false, true)?;
                consume_does_not_join_copy(&lib, false, true)?;
                consume_does_not_join_copy(&lib, true, true)?;
                eprintln!("PASS Engram GPU {device} repeat {repeat}: L14 event guard, skip-wait clobber, independent L1 consume, copy-stream join blocks");
            }
        }
        Ok(())
    }

    fn parked_l14_probe(lib: &NativeLibrary, k: usize, held: bool) -> Result<bool> {
        let rows = 8192usize.div_ceil(k).div_ceil(64) * 64;
        let mut owners = (0..k + 1).map(|_| EngramDeviceRows::new(lib, rows,
            EngramDeviceRows::device_bytes(rows)?)).collect::<Result<Vec<_>>>()?;
        let mut sources = (0..k + 1).map(|_| lib.pinned_bytes(
            cuteafd_loader::EngramBatchStaging::payload_bytes(rows))).collect::<Result<Vec<_>>>()?;
        let hold = LoadStream { library: lib, raw: lib.cuda_stream_create()? };
        let fence = lib.cuda_event_create_ordering()?;
        // The watchdog also covers a driver blocking inside cudaMemcpyAsync.
        // SAFETY: hold and all copy destinations outlive the controlled gate.
        let gate = if held { Some(unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(lib, hold.raw)? }) } else { None };
        let (stop, stopped) = std::sync::mpsc::channel();
        let released = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watchdog = gate.as_ref().map(|gate| {
            let release = gate.release_handle(); let fired = released.clone();
            std::thread::spawn(move || {
                if stopped.recv_timeout(std::time::Duration::from_secs(3)).is_err() {
                    fired.store(true, std::sync::atomic::Ordering::Release);
                    release.store(true, std::sync::atomic::Ordering::Release);
                }
            })
        });
        // SAFETY: all producers, waiters, sources and destinations remain live.
        unsafe {
            lib.cuda_event_record(fence, hold.raw)?;
            for (owner, source) in owners.iter_mut().zip(&mut sources).take(k) {
                lib.cuda_stream_wait_event(owner.stream.raw, fence)?;
                queue_layer(owner, source, 1)?;
            }
        }
        queue_layer(&mut owners[k], &mut sources[k], 0)?;
        let fresh_done = wait_event(lib, owners[k].completed[0], std::time::Duration::from_millis(300))?;
        let before_release = !released.load(std::sync::atomic::Ordering::Acquire);
        let mut all_parked = held;
        for owner in owners.iter().take(k) {
            // SAFETY: live event on a held upload stream.
            all_parked &= !unsafe { lib.cuda_event_query(owner.completed[1])? };
        }
        let _ = stop.send(());
        drop(gate);
        if let Some(watchdog) = watchdog { watchdog.join().expect("copy probe watchdog"); }
        for owner in &owners { owner.synchronize()?; }
        // SAFETY: all producers and consumers drained.
        unsafe { lib.cuda_event_destroy(fence)?; }
        Ok(fresh_done && before_release && (!held || all_parked))
    }
    fn host_triggered_l14(lib: &NativeLibrary, k: usize, held: bool) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let rows = 8192usize.div_ceil(k).div_ceil(64) * 64;
        let batch = std::sync::Arc::new(cuteafd_core::EngramHistory::new(0)?
            .prepare(0, &vec![Some(7); rows], rows)?);
        let weights = tempfile::NamedTempFile::new()?;
        let scales = tempfile::NamedTempFile::new()?;
        let count = cuteafd_core::ENGRAM_ROWS[1];
        weights.as_file().set_len(count * 256)?; scales.as_file().set_len(count * 8)?;
        let addresses = batch.hashes().iter().flat_map(|hashes| hashes[1]).collect::<std::collections::BTreeSet<_>>();
        for address in addresses {
            weights.as_file().write_all_at(&[0x40; 256], address * 256)?;
            scales.as_file().write_all_at(&[127; 8], address * 8)?;
        }
        let map = |file: &tempfile::NamedTempFile, width| {
            // SAFETY: private immutable sparse fixture files outlive every gather.
            unsafe { cuteafd_loader::MappedTable::single(file.path(), 0, count,
                cuteafd_loader::RowFormat { dtype: cuteafd_core::DType::U8, width, row_bytes: width }) }
        };
        let table = std::sync::Arc::new(cuteafd_loader::EngramTable::new(map(&weights, 256)?, map(&scales, 8)?)?);
        let gatherer = cuteafd_loader::EngramGatherer::new_pinned(lib,
            cuteafd_loader::EngramStagingShape::new(rows, k)?)?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_time().build()?;
        let mut leases = Vec::new();
        for _ in 0..k {
            let mut ticket = gatherer.try_submit(table.clone(), std::slice::from_ref(&batch), 1)?.context("gather admission")?;
            match runtime.block_on(ticket.wait())? {
                EngramGatherPoll::Ready(lease) => leases.push(lease),
                _ => anyhow::bail!("gather fixture cancelled"),
            }
        }
        let mut owners = (0..k + 1).map(|_| EngramDeviceRows::new(lib, rows,
            EngramDeviceRows::device_bytes(rows)?)).collect::<Result<Vec<_>>>()?;
        let mut source = lib.pinned_bytes(cuteafd_loader::EngramBatchStaging::payload_bytes(rows))?;
        let holds = (0..k).map(|_| Ok(LoadStream { library: lib, raw: lib.cuda_stream_create()? }))
            .collect::<Result<Vec<_>>>()?;
        let mut gates = Vec::new();
        for (owner, hold) in owners.iter().zip(&holds) {
            // SAFETY: each held producer and its post-gate event survive all copies.
            if held { gates.push(unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(lib, hold.raw)? }); }
            // This is the production post-L1-gate event; no copy has been queued.
            unsafe { lib.cuda_event_record(owner.post_gate, hold.raw)?; }
            owner.gate_recorded.set(true);
            if !held {
                // SAFETY: this warmup producer has no controlled hold.
                unsafe { lib.cuda_stream_synchronize(hold.raw)?; }
            }
        }
        for owner in owners.iter().take(k) { assert_eq!(owner.can_enqueue(1)?, !held); }
        queue_layer(&mut owners[k], &mut source, 0)?;
        assert!(wait_event(lib, owners[k].completed[0], std::time::Duration::from_secs(3))?,
            "fresh L1 must complete while every L14 lease is held on the host");
        let mut bytes = vec![0; rows * 24 * 512];
        lib.copy_d2h(&mut bytes, owners[k].storage.embeddings.buffer)?;
        assert_eq!(bytes, [0x80, 0x3f].repeat(rows * 24 * 256), "fresh L1 bytes");
        for index in 0..k {
            if held {
                assert!(!owners[index].can_enqueue(1)?);
                // Release exactly this unit; other L14 leases remain host-only.
                gates[index].release_handle().store(true, std::sync::atomic::Ordering::Release);
            }
            runtime.block_on(owners[index].wait_enqueue(1, std::time::Duration::from_secs(3)))?;
            owners[index].enqueue_source(&mut leases[index])?;
            owners[index].synchronize()?;
            lib.copy_d2h(&mut bytes, owners[index].storage.embeddings.buffer)?;
            assert_eq!(bytes, [0, 0x40].repeat(rows * 24 * 256), "L14 bytes unit {index}");
            for pending in index + 1..k { assert_eq!(owners[pending].can_enqueue(1)?, !held); }
        }
        drop(gates);
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA native library, runs via golden-run --native-test"]
    fn engram_host_triggered_l14_avoids_parked_copy_stall() -> Result<()> {
        // SAFETY: golden-run supplies the sealed matching native library.
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        for device in 0..2 {
            lib.cuda_set_device(device)?;
            for k in 2..=4 {
                assert!(parked_l14_probe(&lib, k, false)?, "ungated exact-call warmup");
                host_triggered_l14(&lib, k, false)?;
                for repeat in 0..3 {
                    let progress = parked_l14_probe(&lib, k, true)?;
                    assert_eq!(progress, k < 4, "parked-copy negative control GPU {device} k={k}");
                    host_triggered_l14(&lib, k, true)?;
                    eprintln!("PASS Engram host-triggered GPU {device} k={k} repeat={repeat}: fresh L1, host-held L14 leases, exact L14 bytes; parked control fresh L1={progress}");
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA native library and real packed PLE fixture with CPU reference"]
    fn nvfp4_ple_upload_matches_cpu_reference() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_PLE_FIXTURE")?);
        let fixtures: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for device in 0..2 {
            library.cuda_set_device(device)?;
            let mut owner = EngramDeviceRows::new(&library, 16, EngramDeviceRows::device_bytes(16)?)?;
            let pointers = (owner.storage.weights.buffer.ptr, owner.storage.scales.buffer.ptr, owner.storage.embeddings.buffer.ptr);
            for _ in 0..2 {
                for (index, fixture) in fixtures.as_array().context("fixture array")?.iter().enumerate() {
                    let layer = fixture["layer"].as_u64().context("layer")?;
                    let weights = std::fs::read(root.join(format!("layer{layer}-weights.bin")))?;
                    let scales = std::fs::read(root.join(format!("layer{layer}-scales.bin")))?;
                    let expected = std::fs::read(root.join(format!("layer{layer}-expected.bin")))?;
                    let mask: Vec<u8> = fixture["text_mask"].as_array().context("mask")?.iter()
                        .map(|v| v.as_u64().unwrap() as u8).collect();
                    // Reuse the owner at smaller and larger live sizes; FP4 scale
                    // placement depends on live rows, not allocation capacity.
                    for rows in [1, mask.len(), 3, mask.len()] {
                        let gathered = EngramGatherView {
                            weights: &weights[..rows*24*128], scales: &scales[..rows*24*16],
                            text_mask: &mask[..rows], rows, layer_index: index,
                            encoding: cuteafd_loader::EngramEncoding::Nvfp4,
                            global_scale: fixture["global_scale"].as_f64().context("global scale")? as f32,
                        };
                        for cooperative in [false, true] {
                            let view = if cooperative { runtime.block_on(owner.upload_cooperative(&gathered))? }
                                else { owner.upload(&gathered)? };
                            let mut actual = vec![0; view.embeddings.bytes];
                            library.copy_d2h(&mut actual, view.embeddings)?;
                            assert_eq!(actual, expected[..rows*24*512], "GPU {device}, layer {layer}, rows {rows}");
                            let mut actual_mask = vec![0; rows];
                            library.copy_d2h(&mut actual_mask, view.text_mask)?;
                            assert_eq!(actual_mask, mask[..rows]);
                        }
                        runtime.block_on(owner.check_cooperative(&gathered))?;
                        assert_eq!(pointers, (owner.storage.weights.buffer.ptr, owner.storage.scales.buffer.ptr, owner.storage.embeddings.buffer.ptr));
                    }
                    // Reuse the same allocation with the original FP8 format.
                    let weights = vec![0x38; 24*256];
                    let scales = vec![127; 24*8];
                    let gathered = EngramGatherView {
                        weights: &weights, scales: &scales, text_mask: &[1], rows: 1,
                        layer_index: index, encoding: cuteafd_loader::EngramEncoding::Fp8,
                        global_scale: 1.0,
                    };
                    let view = runtime.block_on(owner.upload_cooperative(&gathered))?;
                    let mut actual = vec![0; view.embeddings.bytes];
                    library.copy_d2h(&mut actual, view.embeddings)?;
                    assert!(actual.chunks_exact(2).all(|v| v == [0x80, 0x3f]));
                }
            }
        }
        Ok(())
    }
}
