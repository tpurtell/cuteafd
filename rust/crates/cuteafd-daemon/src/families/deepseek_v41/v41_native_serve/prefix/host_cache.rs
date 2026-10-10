//! Host snapshot cache binding: the CUDA copy engine over the FFI and the glue between the
//! engine's retained snapshots (`Saved`) and `cuteafd_hostcache::HostCache`. Design of record:
//! recipes `dsv41-flash-tp4-engram/research/afd-hostcache-design.md` §4.6. Everything here runs
//! on the scheduler thread; the GPU copies asynchronously on two dedicated streams.
//!
//! A retained snapshot's device bytes are its compressor source pages (each page's rows in four
//! device buffers), its backbone tail in an arena slot and its dSpark rings. Everything else in
//! a `Saved` is small host data and travels as the cache payload (`HostSaved`), so restoring
//! is: allocate fresh pages and arena slots, copy the bytes back, rebuild the `Saved` from parts
//! and let the engine's own restore logic run unchanged.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::BackbonePrefix;
use crate::families::deepseek_v41::v41_compressor::CompressorPrefix;
use crate::families::deepseek_v41::v41_dspark_cache::DsparkPrefix;
use crate::shared::memory::SnapshotStorage;
use crate::families::deepseek_v41::v41_window::WindowPrefix;
use cuteafd_core::EngramHistory;
use cuteafd_ffi::{CopyMechanism, CudaRuntime, CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_hostcache::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget,
    StoreOutcome, StoreTicket,
};
use cuteafd_hostcache::config::Config;
use cuteafd_hostcache::copy::{CopyEngine, DeviceRange, Event, Stream};
use cuteafd_hostcache::metrics::Snapshot as MetricsSnapshot;
use cuteafd_hostcache::pool::{HostChunk, HostRange, Layout, PinnedMemory};
use cuteafd_hostcache::snapshot::{DevicePageId, Hit, SnapshotMeta};
use cuteafd_hostcache::COMPRESSORS;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::time::{Duration, Instant};

/// The engine-side descriptors of a snapshot: everything in a `Saved` that is not device bytes.
/// `images` pins the snapshot's image key ids for as long as the host copy can be looked up.
pub(super) struct HostSaved {
    session: Option<String>,
    images: ImageKeys,
    history: EngramHistory,
    next: TokenScores,
    owner: u64,
    end: u64,
    /// `WindowPrefix` parts per backbone window.
    windows: Vec<(u64, u64, u64)>,
    /// Per compressor: owner, end, page count, rows.
    sources: Vec<(u64, u64, usize, usize)>,
    /// Per dSpark window: owner, end, ring bytes.
    draft: Option<Vec<(u64, u64, usize)>>,
}

fn range(buffer: CuteafdDeviceBuffer) -> DeviceRange {
    DeviceRange {
        addr: buffer.ptr as u64,
        bytes: buffer.bytes,
    }
}

mod regions;
use regions::Regions;
use crate::shared::memory::device::{Allocation, Device};
use std::rc::Rc;

/// One device-owned stream and its nonblocking event probe.
struct StreamState<'a> {
    device: Device<'a>,
    raw: *mut c_void,
    probe: *mut c_void,
    pending: VecDeque<(u64, *mut c_void)>,
    probing: Option<u64>,
}
impl<'a> StreamState<'a> {
    fn new(device: Device<'a>) -> Result<Self> {
        device.run(|| {
            let raw = device.library.cuda_stream_create()?;
            let probe = match device.library.cuda_stream_create() {
                Ok(probe) => probe,
                Err(error) => {
                    // SAFETY: this newly-created stream has no submitted work.
                    let _ = unsafe { device.library.cuda_stream_destroy(raw) };
                    return Err(error);
                }
            };
            Ok(Self { device, raw, probe, pending: VecDeque::new(), probing: None })
        })
    }
    fn synchronize(&self) -> Result<()> {
        self.device.run(|| {
            // SAFETY: this state owns the stream on the selected device.
            unsafe { self.device.library.cuda_stream_synchronize(self.raw) }
        })
    }
    fn completed(&mut self, id: u64) -> Result<bool> {
        let Some(&(_, event)) = self.pending.iter().find(|&&(pending, _)| pending == id) else {
            return Ok(true);
        };
        let _device = self.device.enter()?;
        let library = self.device.library;
        if self.probing != Some(id) {
            // SAFETY: both handles belong to this live state and its device.
            unsafe { library.cuda_stream_wait_event(self.probe, event)?; }
            self.probing = Some(id);
        }
        // SAFETY: the probe is live and its owning device is current.
        if !unsafe { library.cuda_stream_query(self.probe)? } { return Ok(false); }
        while let Some(&(pending, event)) = self.pending.front() {
            // SAFETY: this state's events up through `id` have completed.
            unsafe { library.cuda_event_destroy(event)?; }
            self.pending.pop_front();
            if pending == id { break; }
        }
        self.probing = None;
        Ok(true)
    }
}
impl Drop for StreamState<'_> {
    fn drop(&mut self) {
        let result = self.device.run(|| -> Result<()> {
            // SAFETY: the state owns these streams and events. Drain before
            // destroying them; stream destruction alone is asynchronous.
            unsafe {
                self.device.library.cuda_stream_synchronize(self.raw)?;
                self.device.library.cuda_stream_synchronize(self.probe)?;
                for &(_, event) in &self.pending { self.device.library.cuda_event_destroy(event)?; }
                self.device.library.cuda_stream_destroy(self.probe)?;
                self.device.library.cuda_stream_destroy(self.raw)?;
            }
            Ok(())
        });
        if let Err(error) = result { tracing::error!(%error, "destroying host-cache copy stream"); }
    }
}

pub(crate) struct CudaCopyEngine<'a> {
    library: &'a NativeLibrary,
    template: CuteafdDeviceBuffer,
    regions: Option<Regions>,
    chunks: Vec<Option<CuteafdHostBuffer>>,
    /// Two logical queues, with one physical queue per registered device.
    streams: Vec<[StreamState<'a>; 2]>,
    next_event: u64,
    started: Instant,
    runtime: Option<CudaRuntime>,
    submissions: u64,
    batch_dsts: Vec<*mut c_void>,
    batch_srcs: Vec<*const c_void>,
    batch_sizes: Vec<usize>,
    ordered: Vec<(HostRange, DeviceRange)>,
    routed: Vec<(usize, HostRange, CuteafdDeviceBuffer)>,
    /// Registered allocation owners outlive every stream and pinned chunk.
    /// A failed drain deliberately retains these references through shutdown.
    owners: Vec<Rc<Allocation<'a>>>,
}

impl<'a> CudaCopyEngine<'a> {
    /// Legacy single-stream-device mode. Peer-addressed ranges retain their
    /// existing behavior; registered families select the allocation owner.
    pub fn new(library: &'a NativeLibrary, template: CuteafdDeviceBuffer) -> Result<Self> {
        Self::create(library, template, None)
    }
    /// Register all live snapshot allocations before serving. Registrations
    /// carry no ownership: allocations outlive the cache and its queued copies.
    pub fn registered(library: &'a NativeLibrary, buffers: &[CuteafdDeviceBuffer]) -> Result<Self> {
        let regions = Regions::new(buffers)?;
        Self::create(library, buffers[0], Some(regions))
    }
    /// Keep the actual allocation owners, including family-owned mark arenas,
    /// alive until all host copies drain, even if the family/engine shuts down.
    pub fn registered_owned(library: &'a NativeLibrary, owners: Vec<Rc<Allocation<'a>>>) -> Result<Self> {
        ensure!(owners.iter().all(|owner| std::ptr::eq(owner.device.library, library)),
            "snapshot allocations belong to a different native library");
        let buffers: Vec<_> = owners.iter().map(|owner| owner.buffer).collect();
        let mut engine = Self::registered(library, &buffers)?;
        engine.owners = owners;
        Ok(engine)
    }
    fn create(library: &'a NativeLibrary, template: CuteafdDeviceBuffer, regions: Option<Regions>) -> Result<Self> {
        let ids = regions.as_ref().map(|r| r.devices.clone()).unwrap_or_else(|| vec![template.device_id]);
        let mut streams = Vec::with_capacity(ids.len());
        for id in ids {
            let device = Device { library, id };
            streams.push([StreamState::new(device)?, StreamState::new(device)?]);
        }
        let runtime = CudaRuntime::load();
        let mechanism = if runtime.is_some() { CopyMechanism::MemcpyBatch } else { CopyMechanism::Merged1d };
        tracing::info!(target: "cuteafd::host_cache", mechanism = mechanism.name(), devices = streams.len(),
            runtime_version = runtime.as_ref().map(CudaRuntime::version).unwrap_or(0),
            "host snapshot cache copy mechanism");
        Ok(Self { library, template, regions, chunks: Vec::new(), streams, next_event: 0,
            started: Instant::now(), runtime, submissions: 0, batch_dsts: Vec::new(),
            batch_srcs: Vec::new(), batch_sizes: Vec::new(), ordered: Vec::new(), routed: Vec::new(), owners: Vec::new() })
    }
    fn resolve_host(chunks: &[Option<CuteafdHostBuffer>], range: HostRange) -> Result<CuteafdHostBuffer> {
        let chunk = chunks.get(range.chunk as usize).and_then(Option::as_ref).context("host cache chunk released")?;
        ensure!(range.offset.checked_add(range.bytes).is_some_and(|end| end <= chunk.bytes),
            "host range outside its chunk");
        Ok(CuteafdHostBuffer { ptr: chunk.ptr.cast::<u8>().wrapping_add(range.offset).cast(),
            bytes: range.bytes, flags: chunk.flags })
    }
    /// Exceptional cleanup may exceed the cache's normal wait budget. Every
    /// physical queue is attempted before any storage can be released.
    pub fn synchronize(&mut self, stream: Stream) -> Result<()> {
        let mut first_error = None;
        for rank in &self.streams {
            if let Err(error) = rank[stream as usize].synchronize() {
                if first_error.is_none() { first_error = Some(error); }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Component-only queue identities; callers keep this cache alive.
    #[cfg(test)]
    pub(crate) fn terminal_fixture_streams(&self) -> Vec<(i32, *mut c_void)> {
        self.streams.iter().flat_map(|rank| rank.iter().map(|state| (state.device.id, state.raw))).collect()
    }

    /// Component fixture holds every event until all queues retire.
    #[cfg(test)]
    pub(crate) fn terminal_fixture_wait(&self, events: &[(i32, *mut c_void)]) -> Result<()> {
        for rank in &self.streams {
            for state in rank {
                let event = events.iter().find(|&&(device, _)| device == state.device.id)
                    .context("missing copy-owner fixture event")?.1;
                state.device.run(|| {
                    // SAFETY: the test holds this event and every queued owner
                    // through both compute and copy drainage on this device.
                    unsafe { self.library.cuda_stream_wait_event(state.raw, event) }
                })?;
            }
        }
        Ok(())
    }
    fn issue_many(&mut self, stream: Stream, copies: &[(HostRange, DeviceRange)], d2h: bool) -> Result<()> {
        self.routed.clear();
        // Validate the complete plan, including ownership and all host extents,
        // before the first device receives work.
        for &(host, device) in copies {
            ensure!(host.bytes == device.bytes, "copy length mismatch");
            Self::resolve_host(&self.chunks, host)?;
            if let Some(regions) = &self.regions {
                regions.route(host, device, &mut self.routed)?;
            } else if device.bytes != 0 {
                self.routed.push((0, host, CuteafdDeviceBuffer { ptr: device.addr as *mut c_void,
                    bytes: device.bytes, ..self.template }));
            }
        }
        for rank in 0..self.streams.len() {
            if !self.routed.iter().any(|&(owner, _, _)| owner == rank) { continue; }
            let state = &self.streams[rank][stream as usize];
            let _device = state.device.enter()?;
            self.batch_dsts.clear();
            self.batch_srcs.clear();
            self.batch_sizes.clear();
            for &(_, host, device) in self.routed.iter().filter(|&&(owner, _, _)| owner == rank) {
                let host = Self::resolve_host(&self.chunks, host)?;
                if self.runtime.is_some() {
                    let (dst, src) = if d2h { (host.ptr, device.ptr.cast_const()) }
                        else { (device.ptr, host.ptr.cast_const()) };
                    self.batch_dsts.push(dst);
                    self.batch_srcs.push(src);
                    self.batch_sizes.push(device.bytes);
                } else {
                    // SAFETY: registered extents are live on this stream's
                    // device; the host allocation remains held until completion.
                    unsafe {
                        if d2h { self.library.copy_d2h_host_buffer_async(host, device, device.bytes, state.raw)?; }
                        else { self.library.copy_host_buffer_h2d_async(device, host, device.bytes, state.raw)?; }
                    }
                    self.submissions += 1;
                }
            }
            if let Some(runtime) = &self.runtime {
                // SAFETY: the entire batch was validated, grouped by device,
                // and its staging arrays stay live through this submission.
                unsafe { runtime.memcpy_batch_async(&self.batch_dsts, &self.batch_srcs, &self.batch_sizes, state.raw)?; }
                self.submissions += 1;
            }
        }
        Ok(())
    }
}
impl PinnedMemory for CudaCopyEngine<'_> {
    fn allocate_chunk(&mut self, bytes: usize) -> Result<HostChunk> {
        let buffer = self.library.alloc_host_buffer(bytes)?;
        self.chunks.push(Some(buffer));
        Ok(HostChunk { id: (self.chunks.len() - 1) as u32, bytes })
    }
    fn release_chunk(&mut self, chunk: HostChunk) -> Result<()> {
        self.synchronize(Stream::Store)?;
        self.synchronize(Stream::Restore)?;
        let mut buffer = self.chunks.get_mut(chunk.id as usize).and_then(Option::take)
            .context("host cache chunk already released")?;
        self.library.free_host_buffer(&mut buffer)
    }
}
impl CopyEngine for CudaCopyEngine<'_> {
    fn d2h(&mut self, stream: Stream, src: DeviceRange, dst: HostRange) -> Result<()> {
        self.issue_many(stream, &[(dst, src)], true)
    }
    fn h2d(&mut self, stream: Stream, src: HostRange, dst: DeviceRange) -> Result<()> {
        self.issue_many(stream, &[(src, dst)], false)
    }
    fn d2h_many(&mut self, stream: Stream, copies: &[(DeviceRange, HostRange)]) -> Result<()> {
        self.ordered.clear();
        self.ordered.extend(copies.iter().map(|&(device, host)| (host, device)));
        let ordered = std::mem::take(&mut self.ordered);
        let result = self.issue_many(stream, &ordered, true);
        self.ordered = ordered;
        result
    }
    fn h2d_many(&mut self, stream: Stream, copies: &[(HostRange, DeviceRange)]) -> Result<()> {
        self.issue_many(stream, copies, false)
    }
    fn submission_count(&self) -> u64 { self.submissions }
    fn record(&mut self, stream: Stream) -> Result<Event> {
        self.next_event += 1;
        let id = self.next_event;
        for rank in &mut self.streams {
            let state = &mut rank[stream as usize];
            let _device = state.device.enter()?;
            let event = self.library.cuda_event_create()?;
            // SAFETY: the event and stream belong to the selected device.
            if let Err(error) = unsafe { self.library.cuda_event_record(event, state.raw) } {
                // SAFETY: failed recording left no consumer of this new event.
                let _ = unsafe { self.library.cuda_event_destroy(event) };
                return Err(error);
            }
            state.pending.push_back((id, event));
        }
        Ok(Event(id))
    }
    fn completed(&mut self, event: Event) -> Result<bool> {
        let mut complete = true;
        for rank in &mut self.streams {
            for state in rank {
                complete &= state.completed(event.0)?;
            }
        }
        Ok(complete)
    }
    fn wait(&mut self, event: Event, budget_ns: u64) -> Result<bool> {
        let deadline = self.now_ns().saturating_add(budget_ns);
        loop {
            if self.completed(event)? { return Ok(true); }
            let remaining = deadline.saturating_sub(self.now_ns());
            if remaining == 0 { return Ok(false); }
            std::thread::sleep(Duration::from_nanos(remaining.min(20_000)));
        }
    }
    fn release_barrier(&mut self, stream: Stream) -> Result<()> { self.synchronize(stream) }
    fn now_ns(&self) -> u64 { self.started.elapsed().as_nanos() as u64 }
}
impl Drop for CudaCopyEngine<'_> {
    fn drop(&mut self) {
        let store = self.synchronize(Stream::Store);
        let restore = self.synchronize(Stream::Restore);
        if let Err(error) = store.and(restore) {
            // Descriptors and page refs do not own memory. Retain the strong
            // allocation references too, so family/engine drop cannot free it.
            std::mem::forget(std::mem::take(&mut self.owners));
            tracing::error!(%error, "host cache drain failed; retaining pinned and device copy storage");
            return;
        }
        for buffer in self.chunks.iter_mut().flatten() {
            let _ = self.library.free_host_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod cuda_tests {
    use super::*;
    use crate::shared::memory::device::{Allocation, Device};

    fn queued_rank_release(drop_engine: bool) -> Result<()> {
        use crate::shared::memory::device::{Event as DeviceEvent, Stream as DeviceStream};
        use cuteafd_ffi::test_support::CudaStreamGate;
        use std::sync::atomic::Ordering;
        use std::time::Duration;
        // SAFETY: all CUDA owners are destroyed before the loaded library.
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        let allocations = devices.map(|device| Allocation::new(device, 4096)).into_iter().collect::<Result<Vec<_>>>()?;
        let buffers: Vec<_> = allocations.iter().map(|allocation| allocation.buffer).collect();
        for round in 1..=2u8 {
            let originals: Vec<_> = (0..2).map(|rank| vec![round * 29 + rank * 47; 4096]).collect();
            for (allocation, original) in allocations.iter().zip(&originals) {
                allocation.device.run(|| lib.copy_h2d(allocation.buffer, original))?;
            }
            lib.cuda_set_device(0)?;
            let engine = CudaCopyEngine::registered(&lib, &buffers)?;
            let config = Config { bytes: 1 << 20, chunk_bytes: 1 << 16, min_tokens: 1,
                copy_budget_ns: 0, ..Config::default() };
            let mut cache = HostCache::with_rule(config, Layout::family(4096, 4096, 0), engine,
                cuteafd_core::prefix::ReuseRule::EXACT,
                cuteafd_hostcache::snapshot::EvictionOrder::LeastRecent)?;
            let producer = DeviceStream::new(devices[1])?;
            // SAFETY: producer outlives its gate; an independent OS thread
            // releases the callback without scheduler or CUDA progress.
            let gate = devices[1].run(|| unsafe { CudaStreamGate::new(&lib, producer.raw) })?;
            let mut ready = DeviceEvent::new(devices[1])?;
            ready.record(&producer)?;
            let copy_raw = cache.engine_mut().streams[1][Stream::Store as usize].raw;
            devices[1].run(|| {
                // SAFETY: rank1 copy waits for a live rank1 producer event.
                unsafe { lib.cuda_stream_wait_event(copy_raw, ready.raw) }
            })?;
            let half = |buffer: CuteafdDeviceBuffer, offset| DeviceRange {
                addr: buffer.ptr as u64 + offset, bytes: 2048 };
            let mut pages: [Vec<DevicePage>; COMPRESSORS] = Default::default();
            pages[0].push(DevicePage { id: DevicePageId { compressor: 0, page: 0, generation: round as u32 },
                segments: vec![half(buffers[0], 0), half(buffers[1], 0)] });
            let snapshot = DeviceSnapshot { meta: SnapshotMeta { kind: cuteafd_hostcache::SnapshotKind::Prompt,
                tokens: vec![round as u32], end: 1, has_draft: false }, pages,
                tail: vec![half(buffers[0], 2048), half(buffers[1], 2048)], draft: None, scores: Vec::new() };
            let StoreOutcome::Issued(ticket) = cache.store(&snapshot, ()) else { anyhow::bail!("queued store was skipped") };
            let event = cache.engine_mut().record(Stream::Store)?;
            assert!(!cache.engine_mut().completed(event)?, "rank0 completion hid pending rank1 copies");
            let release = gate.release_handle();
            std::thread::scope(|threads| -> Result<()> {
                let (begin, wait) = std::sync::mpsc::channel();
                threads.spawn(move || {
                    let _ = wait.recv_timeout(Duration::from_secs(2));
                    std::thread::sleep(Duration::from_millis(30));
                    release.store(true, Ordering::Release);
                });
                begin.send(())?;
                if drop_engine {
                    drop(cache);
                } else {
                    assert_eq!(cache.before_device_evict(Some(ticket)), EvictDecision::DroppedUncached);
                    assert!(cache.engine_mut().completed(event)?);
                    // Reuse the source immediately; no extra drain may hide an
                    // early release at the device-cache eviction boundary.
                }
                let done = devices[1].run(|| {
                    // SAFETY: producer remains live through this assertion.
                    unsafe { lib.cuda_stream_query(producer.raw) }
                })?;
                assert!(done, "copy storage was released before its independent producer completed");
                assert_eq!(lib.cuda_get_device()?, 0);
                for allocation in &allocations {
                    allocation.device.run(|| lib.copy_h2d(allocation.buffer, &[0xee; 4096]))?;
                }
                Ok(())
            })?;
            drop(gate);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and two CUDA devices"]
    fn queued_rank_store_timeout_drains_before_pages_and_marks_are_reused() -> Result<()> {
        queued_rank_release(false)
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and two CUDA devices"]
    fn queued_rank_copy_engine_drop_drains_before_pinned_storage_is_freed() -> Result<()> {
        queued_rank_release(true)
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and two CUDA devices"]
    fn registered_owners_survive_family_drop_until_copy_engine_drains() -> Result<()> {
        use crate::shared::memory::device::{Event as DeviceEvent, Stream as DeviceStream};
        use cuteafd_ffi::test_support::CudaStreamGate;
        use std::sync::atomic::Ordering;
        // SAFETY: every CUDA owner is destroyed before this loaded library.
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let owners = (0..2).map(|id| Allocation::new(Device { library: &lib, id }, 4096).map(Rc::new))
            .collect::<Result<Vec<_>>>()?;
        let weak: Vec<_> = owners.iter().map(Rc::downgrade).collect();
        lib.cuda_set_device(0)?;
        let mut engine = CudaCopyEngine::registered_owned(&lib, owners.clone())?;
        let chunk = engine.allocate_chunk(8192)?;
        let device = Device { library: &lib, id: 1 };
        let producer = DeviceStream::new(device)?;
        // SAFETY: this independent producer stream outlives both its gate and the copy engine.
        let gate = device.run(|| unsafe { CudaStreamGate::new(&lib, producer.raw) })?;
        let mut ready = DeviceEvent::new(device)?;
        ready.record(&producer)?;
        device.run(|| {
            // SAFETY: the event and copy stream belong to this live device.
            unsafe { lib.cuda_stream_wait_event(engine.streams[1][Stream::Store as usize].raw, ready.raw) }
        })?;
        for (rank, owner) in owners.iter().enumerate() {
            engine.d2h(Stream::Store, range(owner.buffer), HostRange { chunk: chunk.id, offset: rank * 4096, bytes: 4096 })?;
        }
        drop(owners); // Model/family ownership has ended; only the copy engine remains.
        assert!(weak.iter().all(|owner| owner.upgrade().is_some()));
        let release = gate.release_handle();
        std::thread::scope(|threads| {
            threads.spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                release.store(true, Ordering::Release);
            });
            drop(engine);
        });
        drop(gate);
        assert!(weak.iter().all(|owner| owner.upgrade().is_none()));
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA devices"]
    fn registered_rank_pages_and_marks_restore_exactly_with_reused_storage() -> Result<()> {
        // SAFETY: all CUDA owners are destroyed before the loaded library.
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let mut allocations = Vec::new();
        for id in 0..2 {
            let device = Device { library: &lib, id };
            allocations.push(Allocation::new(device, 4096)?);
            allocations.push(Allocation::new(device, 2048)?);
        }
        let buffers: Vec<_> = allocations.iter().map(|allocation| allocation.buffer).collect();
        // No peer-access setup: every host copy must use its owning GPU.
        for fallback in [false, true] {
            lib.cuda_set_device(1)?;
            let mut engine = CudaCopyEngine::registered(&lib, &buffers)?;
            assert!(engine.runtime.is_some(), "CUDA 13 batch-copy runtime is required for this gate");
            if fallback { engine.runtime = None; }
            assert_eq!(lib.cuda_get_device()?, 1);
            let config = Config { bytes: 1 << 20, chunk_bytes: 1 << 16, min_tokens: 1,
                ..Config::default() };
            let mut cache = HostCache::with_rule(config, Layout::family(8192, 4096, 0), engine,
                cuteafd_core::prefix::ReuseRule::EXACT,
                cuteafd_hostcache::snapshot::EvictionOrder::LeastRecent)?;
            for round in 1..=3u32 {
                let expected: Vec<Vec<u8>> = buffers.iter().enumerate().map(|(index, buffer)| {
                    (0..buffer.bytes).map(|byte| (byte as u32 * 19 + round * 23 + index as u32 * 61) as u8).collect()
                }).collect();
                for (allocation, bytes) in allocations.iter().zip(&expected) {
                    allocation.device.run(|| lib.copy_h2d(allocation.buffer, bytes))?;
                }
                let mut pages: [Vec<DevicePage>; COMPRESSORS] = Default::default();
                pages[0].push(DevicePage { id: DevicePageId { compressor: 0, page: 0, generation: round },
                    segments: vec![range(buffers[0]), range(buffers[2])] });
                let snapshot = DeviceSnapshot { meta: SnapshotMeta { kind: cuteafd_hostcache::SnapshotKind::Prompt,
                    tokens: vec![round], end: 1, has_draft: false }, pages,
                    tail: vec![range(buffers[1]), range(buffers[3])], draft: None, scores: Vec::new() };
                let StoreOutcome::Issued(ticket) = cache.store(&snapshot, ()) else { anyhow::bail!("store was skipped") };
                assert_ne!(cache.before_device_evict(Some(ticket)), EvictDecision::Held);
                assert_eq!(lib.cuda_get_device()?, 1);
                for allocation in &allocations {
                    allocation.device.run(|| lib.copy_h2d(allocation.buffer, &vec![0; allocation.buffer.bytes]))?;
                }
                let hit = cache.lookup(&[round]).context("completed rank snapshot is missing")?;
                let mut target = RestoreTarget { pages: snapshot.pages.clone(), tail: snapshot.tail.clone(),
                    draft: None, scores: Vec::new() };
                target.pages[0][0].id.generation += 100;
                ensure!(matches!(cache.restore(hit.key, &target), RestoreOutcome::Done { .. }), "restore did not finish");
                assert_eq!(lib.cuda_get_device()?, 1);
                for (allocation, expected) in allocations.iter().zip(&expected) {
                    let mut actual = vec![0; expected.len()];
                    allocation.device.run(|| lib.copy_d2h(&mut actual, allocation.buffer))?;
                    assert_eq!(&actual, expected, "rank {} round {round} fallback={fallback}", allocation.device.id);
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA devices"]
    fn native_host_cache_copies_both_gpus() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let devices = [Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 }];
        for device in devices { device.run(|| lib.cuda_enable_peer(1-device.id))?; }
        let buffers = devices.map(|device| Allocation::new(device, 4096))
            .into_iter().collect::<Result<Vec<_>>>()?;
        let originals = [vec![17u8; 4096], vec![93u8; 4096]];
        let mut engine = CudaCopyEngine::new(&lib, buffers[0].buffer)?;
        let chunk = engine.allocate_chunk(8192)?;
        let hosts = [0, 4096].map(|offset| HostRange { chunk: chunk.id, offset, bytes: 4096 });
        // Cover the runtime-selected batch path and the older-runtime 1D fallback.
        for fallback in [false, true] {
            if fallback { engine.runtime = None; }
            for (buffer, original) in buffers.iter().zip(&originals) {
                buffer.device.run(|| lib.copy_h2d(buffer.buffer, original))?;
            }
            let copies: Vec<_> = buffers.iter().zip(hosts).map(|(b, h)| (range(b.buffer), h)).collect();
            engine.d2h_many(Stream::Store, &copies)?;
            engine.synchronize(Stream::Store)?;
            for buffer in &buffers {
                buffer.device.run(|| lib.copy_h2d(buffer.buffer, &[0; 4096]))?;
            }
            let copies: Vec<_> = copies.into_iter().map(|(d, h)| (h, d)).collect();
            engine.h2d_many(Stream::Restore, &copies)?;
            let event = engine.record(Stream::Restore)?;
            engine.synchronize(Stream::Restore)?;
            assert!(engine.completed(event)?);
            for (buffer, original) in buffers.iter().zip(&originals) {
                let mut actual = vec![0u8; 4096];
                buffer.device.run(|| lib.copy_d2h(&mut actual, buffer.buffer))?;
                assert_eq!(&actual, original, "GPU {} fallback={fallback}", buffer.device.id);
            }
        }
        engine.release_chunk(chunk)?;
        Ok(())
    }
}

/// The glue: builds `DeviceSnapshot`s from `Saved`s and `Saved`s from restored bytes.
pub(crate) struct HostCacheBinding<'a> {
    cache: HostCache<CudaCopyEngine<'a>, HostSaved>,
}

impl<'a> HostCacheBinding<'a> {
    /// `None` when the cache is disabled: the engine's paths stay untouched.
    pub fn new(
        library: &'a NativeLibrary,
        config: Config,
        template: CuteafdDeviceBuffer,
    ) -> Result<Option<Self>> {
        config.validate()?;
        if !config.enabled() {
            return Ok(None);
        }
        let engine = CudaCopyEngine::new(library, template)?;
        let cache = HostCache::with_rule(config, Layout::engine(0), engine,
            cuteafd_core::prefix::ReuseRule::V41, cuteafd_hostcache::snapshot::EvictionOrder::LeastRecent)?;
        tracing::info!(target: "cuteafd::host_cache", config = ?cache.config(), "host snapshot cache enabled");
        Ok(Some(Self { cache }))
    }

    fn describe(
        &self,
        kind: SnapshotKind,
        keys: &[u32],
        saved: &Saved<'a>,
        requests: &Requests<'a>,
    ) -> Result<(DeviceSnapshot, HostSaved)> {
        let (backbone, history) = saved.target.parts();
        let (owner, end, tail, windows, sources) = backbone.parts();
        let caches = requests.cache().sources();
        let mut pages: [Vec<DevicePage>; COMPRESSORS] = Default::default();
        let mut source_parts = Vec::with_capacity(COMPRESSORS);
        for (c, prefix) in sources.iter().enumerate() {
            let (source_owner, source_end, source) = prefix.parts();
            let cache = caches[c].get().source_cache();
            pages[c] = source
                .pages()
                .iter()
                .map(|&page| DevicePage {
                    id: DevicePageId {
                        compressor: c as u8,
                        page,
                        generation: cache.page_generation(page),
                    },
                    segments: cache.page_segments(page).into_iter().map(range).collect(),
                })
                .collect();
            source_parts.push((
                source_owner,
                source_end,
                source.pages().len(),
                source.rows(),
            ));
        }
        let draft = saved.draft.as_ref().map(|d| {
            d.parts()
                .iter()
                .map(|p| p.parts())
                .map(|(o, e, ring)| ((o, e, ring.buffer.bytes), range(ring.buffer)))
                .unzip::<_, _, Vec<_>, Vec<_>>()
        });
        let (draft_parts, draft_ranges) = match draft {
            Some((parts, ranges)) => (Some(parts), Some(ranges)),
            None => (None, None),
        };
        let snapshot = DeviceSnapshot {
            meta: SnapshotMeta {
                kind,
                tokens: keys.to_vec(),
                end: end as u32,
                has_draft: draft_ranges.is_some(),
            },
            pages,
            tail: vec![range(tail.buffer)],
            draft: draft_ranges,
            scores: vec![],
        };
        let payload = HostSaved {
            session: saved.session.clone(),
            images: saved._images.through(end as usize),
            history: history.fork()?,
            next: saved.next.clone(),
            owner,
            end,
            windows: windows.iter().map(WindowPrefix::parts).collect(),
            sources: source_parts,
            draft: draft_parts,
        };
        Ok((snapshot, payload))
    }

    /// Issue the write-behind copy of a freshly retained snapshot; the ticket lives in the `Saved`.
    pub(super) fn store(
        &mut self,
        kind: SnapshotKind,
        keys: &[u32],
        saved: &Saved<'a>,
        requests: &Requests<'a>,
    ) -> Result<Option<StoreTicket>> {
        let (snapshot, payload) = self.describe(kind, keys, saved, requests)?;
        Ok(match self.cache.store(&snapshot, payload) {
            StoreOutcome::Issued(ticket) | StoreOutcome::Deferred(ticket) => Some(ticket),
            StoreOutcome::Skipped(_) => None,
        })
    }
    /// The key-space tokens a resident host snapshot is keyed by.
    pub(super) fn snapshot_tokens(&self, key: cuteafd_hostcache::snapshot::Key) -> Option<Vec<u32>> {
        self.cache.snapshot_tokens(key).map(<[u32]>::to_vec)
    }
    pub(super) fn tick(&mut self) {
        self.cache.tick();
    }
    /// A host hit whose restore could not be carried out (for example no device pages for it):
    /// counted with the copy failures so `/v1/stats` shows every abandoned restore; the request
    /// prefills instead.
    pub(super) fn count_abandoned_restore(&mut self) {
        self.cache.metrics_mut().get_mut().restore_failures += 1;
    }
    /// The engine is dropping a `Saved`: let its copy finish within budget or count the loss.
    pub(super) fn before_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        self.cache.before_device_evict(ticket)
    }
    pub(super) fn release_barrier(&mut self) -> Result<()> {
        let store = self.cache.engine_mut().synchronize(Stream::Store);
        let restore = self.cache.engine_mut().synchronize(Stream::Restore);
        store.and(restore)
    }
    /// Delegates to [`HostCache::prefill_hold`]; the full contract lives there. The
    /// scheduler's wrapper observes the store stream (`tick`) on every prefill chunk
    /// regardless of `store_pace_ns` before calling this hold.
    pub(super) fn prefill_hold(&mut self) -> anyhow::Result<()> {
        self.cache.prefill_hold()
    }
    /// The effective configuration, exported with the metrics.
    pub(super) fn config(&self) -> &Config {
        self.cache.config()
    }
    pub(super) fn lookup(&mut self, keys: &[u32]) -> Option<Hit> {
        self.cache.lookup(keys)
    }
    /// Rebuild a `Saved` from the host copy. `Ok(None)` when the restore timed out or failed (the
    /// caller prefills); allocations are released only after the restore stream drained.
    pub(super) fn restore<C: crate::families::deepseek_v41::v41_native_serve::speculative::DraftChain<'a>>(
        &mut self,
        hit: &Hit,
        requests: &Requests<'a>,
        draft: Option<&DraftRuntime<'_, 'a, C>>,
    ) -> Result<Option<Saved<'a>>> {
        // Take what the rebuilt `Saved` needs out of the payload before the mutable restore call.
        let (owner, end, windows, source_parts, draft_parts, history, next, images, session) = {
            let payload = self
                .cache
                .payload(hit.key)
                .context("host cache hit without payload")?;
            (
                payload.owner,
                payload.end,
                payload.windows.clone(),
                payload.sources.clone(),
                payload.draft.clone(),
                payload.history.fork()?,
                payload.next.clone(),
                payload.images.through(payload.end as usize),
                payload.session.clone(),
            )
        };
        let backbone = requests.cache();
        let caches = backbone.sources();
        ensure!(
            draft_parts.is_some() == draft.is_some(),
            "host snapshot execution mode differs"
        );
        let mut sources = Vec::with_capacity(COMPRESSORS);
        let mut pages: [Vec<DevicePage>; COMPRESSORS] = Default::default();
        for (c, &(owner, end, count, rows)) in source_parts.iter().enumerate() {
            let cache = caches[c].get().source_cache();
            let source = cache.allocate_prefix(count, rows)?;
            pages[c] = source
                .pages()
                .iter()
                .map(|&page| DevicePage {
                    id: DevicePageId {
                        compressor: c as u8,
                        page,
                        generation: cache.page_generation(page),
                    },
                    segments: cache.page_segments(page).into_iter().map(range).collect(),
                })
                .collect();
            sources.push(CompressorPrefix::from_parts(owner, end, source));
        }
        let tail = SnapshotStorage::new(
            backbone.prefix_library(),
            BackbonePrefix::device_bytes(),
            backbone.prefix_pool(),
        )?;
        let rings = match (&draft_parts, draft) {
            (Some(parts), Some(runtime)) => Some(
                parts
                    .iter()
                    .zip(runtime.windows())
                    .map(|(&(_, _, bytes), window)| {
                        window.device().run(||
                            SnapshotStorage::new(window.library(), bytes, window.prefix_pool()))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            _ => None,
        };
        let target = RestoreTarget {
            pages,
            tail: vec![range(tail.buffer)],
            draft: rings
                .as_ref()
                .map(|rings| rings.iter().map(|r| range(r.buffer)).collect()),
            scores: vec![],
        };
        match self.cache.restore(hit.key, &target) {
            RestoreOutcome::Done { .. } => {}
            outcome => {
                tracing::warn!(target: "cuteafd::host_cache", ?outcome, "host restore did not complete; prefilling");
                if let Err(error) = self.cache.engine_mut().synchronize(Stream::Restore) {
                    // A failed release barrier must not return pages/arena
                    // slots while an upload may still write them.
                    std::mem::forget((sources, tail, rings));
                    return Err(error);
                }
                return Ok(None);
            }
        }
        for (cache,prefix) in caches.iter().zip(&sources) {
            // RestoreOutcome::Done means the host upload completed. Publish FP4
            // replica pages before the rebuilt prefix can enter the GPU cache.
            unsafe { cache.get().source_cache().publish_restored_prefix(prefix.parts().2)?; }
        }
        let windows = windows
            .iter()
            .map(|&(o, e, b)| WindowPrefix::from_parts(o, e, b))
            .collect();
        let target_prefix = RequestPrefix::from_parts(
            BackbonePrefix::from_parts(owner, end, tail, windows, sources),
            history,
        );
        let draft = match (draft_parts.as_ref(), rings) {
            (Some(parts), Some(rings)) => Some(DraftPrefix::from_parts(backbone.prefix_library(),
                parts
                    .iter()
                    .zip(rings)
                    .map(|(&(o, e, _), ring)| DsparkPrefix::from_parts(o, e, ring))
                    .collect(),
            )?),
            _ => None,
        };
        Ok(Some(Saved {
            session,
            _images: images,
            target: target_prefix,
            draft,
            next,
            ticket: None,
        }))
    }
    pub(super) fn metrics(&self) -> MetricsSnapshot {
        self.cache.metrics()
    }
}
