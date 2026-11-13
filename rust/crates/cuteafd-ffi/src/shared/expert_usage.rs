//! Per-engine device counters and one independent, sleeping snapshot worker.
use crate::*;
use anyhow::ensure;
use cuteafd_core::expert_usage::{device_owner, Geometry, Reader, Snapshot};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

type Tally =
    unsafe extern "C" fn(*const u32, *mut u64, u32, u32, u32, u32, *const i32, *mut c_void) -> i32;
type SnapshotFn = unsafe extern "C" fn(*mut u64, *mut u64, u32, *mut c_void) -> i32;
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(5);
const QUERY_INTERVAL: Duration = Duration::from_millis(10);

/// Recipe owners use logical ranks; pointer/stream lookup uses CUDA ordinals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UsageDevice {
    pub logical: u8,
    pub cuda: i32,
}

/// A retained backing, not a reconstructed allocation-history record.
#[derive(Clone, Copy, Debug)]
pub enum UsageStorage {
    Device(CuteafdDeviceBuffer),
    Pinned(CuteafdHostBuffer),
}

#[derive(Clone, Debug)]
pub struct UsageAllocation {
    pub owner: String,
    pub device: UsageDevice,
    pub storage: UsageStorage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsagePhase {
    Prefill,
    Decode,
}

/// Only the canonical router calls this descriptor, with its real row count.
/// Replicated TP2 routers pass `canonical = false` and enqueue no work.
#[derive(Clone, Copy)]
pub struct RouteTally<'a> {
    owner: &'a RawOwner,
    layer: usize,
    phase: UsagePhase,
}
impl RouteTally<'_> {
    /// # Safety
    /// `ids` has `rows * topk` U32 entries on the owner's current CUDA device,
    /// ordered before this call on `stream`; drain graphs before dropping usage.
    /// Optional `real_rows` points at a live device I32 count for padded graphs.
    pub unsafe fn record(
        &self,
        ids: *const c_void,
        rows: usize,
        real_rows: Option<*const i32>,
        mask: u32,
        canonical: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        if !canonical || rows == 0 {
            return Ok(());
        }
        ensure!(
            !ids.is_null() && real_rows.is_none_or(|p| !p.is_null()),
            "null usage route operand"
        );
        let g = &self.owner.geometry;
        let offset =
            (usize::from(self.phase == UsagePhase::Decode) * g.layers + self.layer) * g.experts;
        // SAFETY: descriptor construction validates layer, allocation has both phase tables.
        let counts = unsafe { self.owner.counters.ptr.cast::<u64>().add(offset) };
        // SAFETY: device pointers/stream lifetime are the caller's contract, ABI is fixed above.
        let status = unsafe {
            (self.owner.tally)(
                ids.cast(),
                counts,
                u32::try_from(rows)?,
                u32::try_from(g.topk)?,
                u32::try_from(g.experts)?,
                mask,
                real_rows.unwrap_or(std::ptr::null()),
                stream,
            )
        };
        ensure!(status == 0, "expert usage tally failed: {status}");
        Ok(())
    }
}

struct RawOwner {
    _module: Arc<Library>,
    geometry: Arc<Geometry>,
    counters: CuteafdDeviceBuffer,
    snapshot: CuteafdDeviceBuffer,
    host: CuteafdHostBuffer,
    stream: *mut c_void,
    event: *mut c_void,
    device: UsageDevice,
    tally: Tally,
    read: SnapshotFn,
    set_device: unsafe extern "C" fn(i32) -> CuteafdStatus,
    free_device: FreeDeviceBufferFn,
    free_host: FreeHostBufferFn,
    destroy_stream: CudaStreamDestroyFn,
    copy: CopyD2HAsyncFn,
    record_event: CudaEventRecordFn,
    query_event: CudaEventQueryFn,
    destroy_event: CudaEventDestroyFn,
}
// SAFETY: pointers remain owned until the worker joins. Counter access is atomic;
// only the worker touches snapshot/host storage, on its private device streams.
unsafe impl Send for RawOwner {}
// SAFETY: serving only reads immutable metadata and atomically adds device counters.
unsafe impl Sync for RawOwner {}
impl RawOwner {
    fn new(library: &NativeLibrary, geometry: Arc<Geometry>, device: UsageDevice) -> Result<Self> {
        library.cuda_set_device(device.cuda)?;
        let bytes = geometry
            .table_bytes()
            .context("usage table size overflow")?;
        // SAFETY: native ABI is fixed; the owner retains the module for all symbols.
        let mut raw = Self {
            _module: Arc::clone(&library.lib),
            geometry,
            counters: Default::default(),
            snapshot: Default::default(),
            host: Default::default(),
            stream: std::ptr::null_mut(),
            event: std::ptr::null_mut(),
            device,
            tally: unsafe { *library.lib.get(b"cuteafd_expert_usage_tally")? },
            read: unsafe { *library.lib.get(b"cuteafd_expert_usage_snapshot")? },
            set_device: library.cuteafd_cuda_set_device()?,
            free_device: library.cuteafd_free_device_buffer()?,
            free_host: library.cuteafd_free_host_buffer()?,
            destroy_stream: library.cuteafd_cuda_stream_destroy()?,
            copy: library.cuteafd_copy_d2h_async()?,
            record_event: library.cuteafd_cuda_event_record()?,
            query_event: library.cuteafd_cuda_event_query()?,
            destroy_event: library.cuteafd_cuda_event_destroy()?,
        };
        {
            let owner = device_owner(device.logical, "counters");
            let _owner = memory_ledger::owner(&owner);
            raw.counters = library.alloc_device_buffer(bytes, memory_ledger::tag(memory_ledger::MemoryKind::Workspace, &owner))?;
            library.cuda_zero_bytes(raw.counters, bytes)?;
        }
        {
            let owner = device_owner(device.logical, "snapshot");
            let _owner = memory_ledger::owner(&owner);
            raw.snapshot = library.alloc_device_buffer(bytes, memory_ledger::tag(memory_ledger::MemoryKind::Staging, &owner))?;
        }
        {
            let owner = device_owner(device.logical, "pinned");
            let _owner = memory_ledger::owner(&owner);
            raw.host = library.alloc_host_buffer(bytes, memory_ledger::tag(memory_ledger::MemoryKind::Staging, &owner))?;
        }
        raw.stream = library.cuda_stream_create()?;
        raw.event = library.cuda_event_create_ordering()?;
        Ok(raw)
    }
    fn begin_snapshot(&self) -> Result<()> {
        // SAFETY: all symbols belong to the retained module, buffers are owned
        // here, and each private snapshot stream has exactly one worker.
        unsafe {
            ensure!(
                (self.set_device)(self.device.cuda) == 0,
                "usage snapshot device selection failed"
            );
            ensure!(
                (self.read)(
                    self.counters.ptr.cast(),
                    self.snapshot.ptr.cast(),
                    u32::try_from(self.geometry.cells().unwrap() * 2)?,
                    self.stream
                ) == 0,
                "usage snapshot kernel failed"
            );
            ensure!(
                (self.copy)(self.host.ptr, self.snapshot, self.host.bytes, self.stream) == 0,
                "usage snapshot D2H failed"
            );
            ensure!(
                (self.record_event)(self.event, self.stream) == 0,
                "usage snapshot event recording failed"
            );
        }
        Ok(())
    }
    fn merge_snapshot(&self, merged: &mut Snapshot, deadline: Instant) -> Result<()> {
        // SAFETY: this worker owns the event and selects its CUDA device before querying.
        unsafe {
            ensure!(
                (self.set_device)(self.device.cuda) == 0,
                "usage snapshot device selection failed"
            );
            wait_ready(deadline, || {
                let mut ready = 0;
                ensure!(
                    (self.query_event)(self.event, &mut ready) == 0,
                    "usage snapshot event query failed"
                );
                ensure!(ready == 0 || ready == 1, "invalid usage event query result");
                Ok(ready == 1)
            })?;
            let cells = self.geometry.cells().unwrap();
            // Only the completed pinned aggregate matrices reach the host; never route IDs.
            let words = std::slice::from_raw_parts(self.host.ptr.cast::<u64>(), 2 * cells);
            merge_counts(merged, words)
        }
    }
}

fn wait_ready(deadline: Instant, mut query: impl FnMut() -> Result<bool>) -> Result<()> {
    loop {
        if query()? {
            return Ok(());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        ensure!(
            !remaining.is_zero(),
            "expert usage snapshot timed out; retained snapshot is stale"
        );
        std::thread::sleep(QUERY_INTERVAL.min(remaining));
    }
}

fn merge_counts(merged: &mut Snapshot, words: &[u64]) -> Result<()> {
    let cells = merged.prefill.len();
    ensure!(
        merged.decode.len() == cells && words.len() == cells * 2,
        "invalid usage merge extent"
    );
    for (target, &count) in merged
        .prefill
        .iter_mut()
        .chain(&mut merged.decode)
        .zip(words)
    {
        *target = target
            .checked_add(count)
            .context("expert usage merged counter overflow")?;
    }
    Ok(())
}

fn validate_devices(devices: &[UsageDevice]) -> bool {
    !devices.is_empty()
        && devices.iter().enumerate().all(|(i, device)| {
            device.cuda >= 0
                && devices[..i]
                    .iter()
                    .all(|old| old.logical != device.logical && old.cuda != device.cuda)
        })
}

impl Drop for RawOwner {
    fn drop(&mut self) {
        // SAFETY: worker joined and engine drained all counter users. On snapshot
        // failure the worker retains the whole owner set instead of reaching Drop.
        unsafe {
            if (self.set_device)(self.device.cuda) != 0 {
                std::mem::forget(Arc::clone(&self._module));
                return;
            }
            if !self.stream.is_null() {
                let _ = (self.destroy_stream)(self.stream);
            }
            if !self.event.is_null() {
                let _ = (self.destroy_event)(self.event);
            }
            for buffer in [&mut self.snapshot, &mut self.counters] {
                if !buffer.ptr.is_null() {
                    memory_ledger::record_free(buffer.ptr as usize);
                    let _ = (self.free_device)(buffer);
                }
            }
            if !self.host.ptr.is_null() {
                memory_ledger::record_free(self.host.ptr as usize);
                let _ = (self.free_host)(&mut self.host);
            }
        }
    }
}

struct RestoreDevice<'a> {
    library: &'a NativeLibrary,
    device: i32,
}
impl Drop for RestoreDevice<'_> {
    fn drop(&mut self) {
        let _ = self.library.cuda_set_device(self.device);
    }
}

pub struct ExpertUsage<'a> {
    library: &'a NativeLibrary,
    raw: Option<Arc<Vec<RawOwner>>>,
    stop: mpsc::Sender<()>,
    worker: Option<JoinHandle<()>>,
    reader: Reader,
}
impl<'a> ExpertUsage<'a> {
    /// `devices` includes every possible canonical router producer, not TP replicas.
    /// The caller's current device is restored, including on construction failure.
    pub fn new(
        library: &'a NativeLibrary,
        geometry: Geometry,
        devices: &[UsageDevice],
        reader: Reader,
    ) -> Result<Self> {
        reader.require_enabled().map_err(anyhow::Error::msg)?;
        ensure!(geometry.validate(), "invalid expert usage geometry");
        ensure!(
            validate_devices(devices),
            "usage devices must have unique logical ranks and CUDA ordinals"
        );
        ensure!(
            geometry.cells().unwrap() * 2 <= u32::MAX as usize,
            "usage geometry exceeds native indexing"
        );
        let _scope = memory_ledger::scope("expert-usage");
        let geometry = Arc::new(geometry);
        let current = library.cuda_get_device()?;
        let _restore = RestoreDevice {
            library,
            device: current,
        };
        let owners = devices
            .iter()
            .map(|&device| RawOwner::new(library, Arc::clone(&geometry), device))
            .collect::<Result<Vec<_>>>();
        // A partial collection is dropped on error before restoring this thread's device.
        let restored = library.cuda_set_device(current);
        let raw = Arc::new(owners?);
        restored?;
        reader
            .initialize((*geometry).clone())
            .map_err(anyhow::Error::msg)?;
        let (stop, receive) = mpsc::channel();
        let work = Arc::clone(&raw);
        let output = reader.clone();
        let worker = std::thread::Builder::new()
            .name("expert-usage".into())
            .stack_size(cuteafd_core::expert_usage::WORKER_STACK_BYTES)
            .spawn(move || {
                let start = Instant::now();
                while matches!(receive.recv_timeout(Duration::from_secs(1)), Err(mpsc::RecvTimeoutError::Timeout)) {
                    let result = (|| -> Result<()> {
                        let deadline = Instant::now() + SNAPSHOT_TIMEOUT;
                        for device in work.iter() { device.begin_snapshot()?; }
                        let cells = geometry.cells().unwrap();
                        let mut merged = Snapshot { elapsed_s: 0.0, prefill: vec![0; cells], decode: vec![0; cells] };
                        for device in work.iter() { device.merge_snapshot(&mut merged, deadline)?; }
                        merged.elapsed_s = start.elapsed().as_secs_f64();
                        output.publish(merged).map_err(anyhow::Error::msg)
                    })();
                    if let Err(error) = result {
                        tracing::warn!(error = %error, "expert usage snapshot stopped; retained history is stale; serving continues");
                        output.fail(format!("{error:#}"));
                        // A timeout/error may leave queued work. Retain all buffers and
                        // the DSO permanently rather than block serving or free live storage.
                        std::mem::forget(work);
                        return;
                    }
                }
            })?;
        Ok(Self {
            library,
            raw: Some(raw),
            stop,
            worker: Some(worker),
            reader,
        })
    }
    pub fn reader(&self) -> Reader {
        self.reader.clone()
    }
    /// Physical counter/snapshot/pinned owners, with logical and CUDA identities.
    /// Host history, worker stack and metadata are separate host-budget charges,
    /// not a single backing that this inventory could truthfully report.
    /// Descriptors borrow no ownership; their buffers stay live only with `self`.
    pub fn allocation_receipts(&self) -> Vec<UsageAllocation> {
        self.raw.as_ref().into_iter().flat_map(|owners| owners.iter()).flat_map(|raw| {
            [
                ("counters", UsageStorage::Device(raw.counters)),
                ("snapshot", UsageStorage::Device(raw.snapshot)),
                ("pinned", UsageStorage::Pinned(raw.host)),
            ].map(|(kind, storage)| UsageAllocation {
                owner: device_owner(raw.device.logical, kind), device: raw.device, storage,
            })
        }).collect()
    }
    /// No CUDA query/device switch: safe to resolve inside a captured router path.
    pub fn routes(&self, device: i32, layer: usize, phase: UsagePhase) -> Result<RouteTally<'_>> {
        let raw = self
            .raw
            .as_ref()
            .unwrap()
            .iter()
            .find(|raw| raw.device.cuda == device)
            .with_context(|| format!("CUDA device {device} has no canonical usage counters"))?;
        ensure!(
            raw.geometry.routed_layers.binary_search(&layer).is_ok(),
            "usage layer {layer} is not routed"
        );
        Ok(RouteTally {
            owner: raw,
            layer,
            phase,
        })
    }
}
impl Drop for ExpertUsage<'_> {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let current = self.library.cuda_get_device().ok();
        if let Some(raw) = self.raw.take() {
            if self.library.is_quarantined_after_failed_drain() {
                std::mem::forget(raw);
            } else {
                drop(raw);
            }
        }
        if let Some(device) = current {
            let _ = self.library.cuda_set_device(device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_usage_receipts_are_canonical_physical_backings_on_reversed_devices() -> Result<()> {
        use crate::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        for cuda in [0, 1] { fixture.configure_memory(&library, cuda, 1 << 30)?; }
        let geometry = Geometry { layers: 4, experts: 8, topk: 2, routed_layers: vec![1, 3] };
        let devices = [UsageDevice { logical: 0, cuda: 1 }, UsageDevice { logical: 1, cuda: 0 }];
        for devices in [&devices[..1], &devices[..]] {
            library.cuda_set_device(0)?;
            let usage = ExpertUsage::new(&library, geometry.clone(), devices, Reader::default())?;
            assert_eq!(library.cuda_get_device()?, 0, "constructor restores the caller's device");
            let receipts = usage.allocation_receipts();
            assert_eq!(receipts.len(), 3 * devices.len());
            let mut addresses = std::collections::BTreeSet::new();
            for (raw, receipts) in usage.raw.as_ref().unwrap().iter().zip(receipts.chunks_exact(3)) {
                for (receipt, (kind, address)) in receipts.iter().zip([
                    ("counters", raw.counters.ptr as usize),
                    ("snapshot", raw.snapshot.ptr as usize),
                    ("pinned", raw.host.ptr as usize),
                ]) {
                    assert_eq!(receipt.owner, device_owner(raw.device.logical, kind));
                    assert_eq!(receipt.device, raw.device);
                    let (actual, bytes) = match receipt.storage {
                        UsageStorage::Device(buffer) => {
                            assert_ne!(kind, "pinned");
                            assert_eq!(buffer.device_id, raw.device.cuda);
                            (buffer.ptr as usize, buffer.bytes)
                        }
                        UsageStorage::Pinned(buffer) => {
                            assert_eq!(kind, "pinned");
                            assert_ne!(buffer.flags & CUTEAFD_HOST_BUFFER_FLAG_PINNED, 0);
                            (buffer.ptr as usize, buffer.bytes)
                        }
                    };
                    assert_eq!(actual, address);
                    assert_eq!(bytes, geometry.table_bytes().unwrap());
                    assert!(actual != 0 && addresses.insert(actual), "each receipt names one distinct live backing");
                }
                library.cuda_set_device(raw.device.cuda)?;
                assert_eq!(library.cuda_memory_info()?.0, (1 << 30) - 2 * geometry.table_bytes().unwrap());
            }
            library.cuda_set_device(0)?;
            drop(usage);
            assert_eq!(library.cuda_get_device()?, 0, "retirement restores the caller's device");
            for device in devices {
                library.cuda_set_device(device.cuda)?;
                assert_eq!(library.cuda_memory_info()?, (1 << 30, 1 << 30));
            }
        }
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn usage_fixture_tallies_and_snapshots_real_rows_and_rejects_invalid_arguments() -> Result<()> {
        use crate::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        let geometry = Arc::new(Geometry { layers: 2, experts: 4, topk: 2, routed_layers: vec![1] });
        let raw = RawOwner::new(&library, Arc::clone(&geometry), UsageDevice { logical: 1, cuda: 0 })?;
        let ids = [0u32, 1, 0x10001, 3, 2, 2];
        let real_rows = 2;
        // SAFETY: fixture operands are live host-backed CPU buffers and the stream is retained by raw.
        unsafe {
            assert_eq!((raw.tally)(ids.as_ptr(), raw.counters.ptr.cast(), 3, 2, 4, 0xffff,
                &real_rows, raw.stream), 0);
            assert_ne!((raw.tally)(ids.as_ptr(), raw.counters.ptr.cast(), 1, 0, 4, 0xffff,
                std::ptr::null(), raw.stream), 0);
            assert_ne!((raw.tally)(ids.as_ptr(), raw.counters.ptr.cast(), u32::MAX, 2, 4, 0xffff,
                std::ptr::null(), raw.stream), 0);
            assert_ne!((raw.tally)(ids.as_ptr(), std::ptr::null_mut(), 1, 2, 4, 0xffff,
                std::ptr::null(), raw.stream), 0);
            assert_ne!((raw.read)(raw.counters.ptr.cast(), raw.snapshot.ptr.cast(), 0, raw.stream), 0);
            assert_ne!((raw.read)(raw.counters.ptr.cast(), raw.snapshot.ptr.cast(), 4,
                usize::MAX as *mut c_void), 0);
        }
        raw.begin_snapshot()?;
        let mut merged = Snapshot { elapsed_s: 0.0, prefill: vec![0; 8], decode: vec![0; 8] };
        raw.merge_snapshot(&mut merged, Instant::now() + SNAPSHOT_TIMEOUT)?;
        assert_eq!(merged.prefill, [1, 2, 0, 1, 0, 0, 0, 0]);
        assert_eq!(merged.decode, [0; 8]);
        Ok(())
    }

    #[test]
    fn merged_device_matrices_are_exact_and_overflow_is_rejected() {
        let mut merged = Snapshot {
            elapsed_s: 1.0,
            prefill: vec![0; 2],
            decode: vec![0; 2],
        };
        merge_counts(&mut merged, &[1, 0, 0, 3]).unwrap();
        merge_counts(&mut merged, &[0, 2, 4, 0]).unwrap();
        assert_eq!(merged.prefill, [1, 2]);
        assert_eq!(merged.decode, [4, 3]);
        assert!(merge_counts(&mut merged, &[1]).is_err());
        assert!(merge_counts(&mut merged, &[u64::MAX, 0, 0, 0]).is_err());
    }
    #[test]
    fn snapshot_wait_is_bounded_and_query_errors_propagate() {
        assert!(wait_ready(Instant::now(), || Ok(false))
            .unwrap_err()
            .to_string()
            .contains("timed out"));
        assert!(wait_ready(Instant::now(), || Ok(true)).is_ok());
        assert!(wait_ready(Instant::now(), || anyhow::bail!("failed event")).is_err());
    }
    #[test]
    fn device_ids_are_explicit_unique_and_within_metadata_budget() {
        assert!(validate_devices(&[UsageDevice {
            logical: 0,
            cuda: 1
        }]));
        assert!(validate_devices(&[
            UsageDevice {
                logical: 0,
                cuda: 1
            },
            UsageDevice {
                logical: 1,
                cuda: 0
            }
        ]));
        assert!(!validate_devices(&[]));
        assert!(!validate_devices(&[UsageDevice {
            logical: 0,
            cuda: -1
        }]));
        assert!(!validate_devices(&[
            UsageDevice {
                logical: 0,
                cuda: 0
            },
            UsageDevice {
                logical: 1,
                cuda: 0
            }
        ]));
        assert!(!validate_devices(&[
            UsageDevice {
                logical: 0,
                cuda: 0
            },
            UsageDevice {
                logical: 0,
                cuda: 1
            }
        ]));
        assert!(
            std::mem::size_of::<RawOwner>() <= cuteafd_core::expert_usage::DEVICE_METADATA_BYTES
        );
    }
}
