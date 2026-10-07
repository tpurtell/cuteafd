//! Device side of mapped lookup tables (`cuteafd_loader::MappedTable`): the
//! placement option, a pinned staging ring with stream-ordered H2D uploads,
//! and the gather/prefetch/stats bundle a family engine holds per table.
//!
//! A step's rows are known when its inputs are: `begin` gathers them on the
//! host pool into the next pinned ring slot while the engine enqueues the GPU
//! work that precedes their use (embedding, earlier layers), `finish` waits
//! (stall time) and copies them into the step's device rows on the engine
//! stream; a slot is refilled only after its copy's event completed. Kernels
//! read the gathered rows (`[rows, row_bytes]`, request order) from device
//! memory. Rows known a step ahead (the next prefill chunk) are advised to
//! the page cache by the table's prefetcher.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_loader::{GatherPool, HotRowCache, MappedTable, PendingGather, TablePrefetcher, TableStatsSnapshot};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::memory::HostAllocation;

/// Where a large per-token lookup table lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TablePlacement {
    /// Memory-mapped checkpoint (page cache); each step's rows gathered on
    /// host threads and uploaded. Startup reads only metadata.
    Mapped,
    /// The whole table read into pinned, device-mapped host memory at
    /// startup; kernels read rows over PCIe.
    #[value(alias = "host")]
    HostPreload,
    /// The whole table copied to GPU memory.
    Device,
}

/// Mapped-table options a family engine takes.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct MappedTableArgs {
    /// Host threads gathering mapped rows (and reading a preloaded table).
    #[arg(long = "table-threads", alias = "ple-threads", default_value_t = 16)]
    pub threads: usize,
    /// Hot-row cache in MiB (rows that survive page-cache eviction); 0 is off.
    #[arg(long = "table-cache-mib", default_value_t = 0)]
    pub cache_mib: usize,
    /// Advise the page cache about rows known ahead of their step (the next
    /// prefill chunk).
    #[arg(long = "table-prefetch", default_value_t = true, action = clap::ArgAction::Set)]
    pub prefetch: bool,
    /// Read the whole table into the page cache in the background after
    /// startup (not pinned; the kernel may evict it). Serving starts at once;
    /// gathers that fault before the warm-up reaches their rows read on demand.
    #[arg(long = "table-warm", default_value_t = true, action = clap::ArgAction::Set)]
    pub warm: bool,
    /// Warm-up read rate in MiB/s (0: unpaced); paced so startup weight
    /// streaming and demand faults keep most of the disk.
    #[arg(long = "table-warm-mib-s", default_value_t = 1024)]
    pub warm_mib_s: u64,
}

struct RingSlot<'a> {
    host: HostAllocation<'a>,
    /// Records the slot's last upload; reuse waits for it.
    event: *mut c_void,
    pending: bool,
}

/// Pinned host slots for stream-ordered uploads of gathered rows: a slot is
/// filled on the host, copied on a stream, and refilled only after the copy's
/// event completed.
pub(crate) struct PinnedRowRing<'a> {
    library: &'a NativeLibrary,
    slots: Vec<RingSlot<'a>>,
    next: usize,
}

impl<'a> PinnedRowRing<'a> {
    pub fn new(library: &'a NativeLibrary, slot_bytes: usize, slots: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("mapped-table");
        ensure!(slot_bytes > 0 && slots > 0, "an empty pinned row ring");
        let slots = (0..slots)
            .map(|_| -> Result<RingSlot<'a>> {
                Ok(RingSlot { host: HostAllocation::new(library, slot_bytes)?, event: library.cuda_event_create_ordering()?,
                    pending: false })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { library, slots, next: 0 })
    }

    pub fn slot_bytes(&self) -> usize {
        self.slots[0].host.buffer.bytes
    }

    /// The next slot, once its previous upload completed: (index, pinned bytes).
    pub fn acquire(&mut self) -> Result<(usize, *mut u8)> {
        let index = self.next;
        self.next = (index + 1) % self.slots.len();
        let slot = &mut self.slots[index];
        if slot.pending {
            // SAFETY: the event was created by this ring and recorded after the slot's copy.
            unsafe { self.library.cuda_event_synchronize(slot.event)? };
            slot.pending = false;
        }
        Ok((index, slot.host.buffer.ptr.cast()))
    }

    /// Queues the copy of `bytes` of slot `index` into `dst` on `stream`.
    ///
    /// # Safety
    /// The slot's bytes must be filled (no writer left), `stream` must be a
    /// live stream of the current device, and `dst` must stay allocated until
    /// the copy completes (stream order).
    pub unsafe fn submit(&mut self, index: usize, dst: CuteafdDeviceBuffer, bytes: usize, stream: *mut c_void)
        -> Result<()> {
        ensure!(bytes <= self.slot_bytes() && bytes <= dst.bytes, "{bytes} gathered bytes exceed the ring or device rows");
        let slot = &mut self.slots[index];
        if bytes == 0 {
            return Ok(());
        }
        let source = CuteafdHostBuffer { bytes, ..slot.host.buffer };
        // SAFETY: per this function's contract; the event guards the slot's reuse.
        unsafe {
            self.library.copy_host_buffer_h2d_async(dst, source, bytes, stream)?;
            self.library.cuda_event_record(slot.event, stream)?;
        }
        slot.pending = true;
        Ok(())
    }
}

impl Drop for PinnedRowRing<'_> {
    fn drop(&mut self) {
        for slot in &self.slots {
            // SAFETY: each event belongs to this ring; waiting first keeps the
            // pinned slot alive until its copy completes.
            unsafe {
                if slot.pending {
                    if let Err(error) = self.library.cuda_event_synchronize(slot.event) {
                        tracing::error!(%error, "draining a mapped-table upload");
                    }
                }
                if let Err(error) = self.library.cuda_event_destroy(slot.event) {
                    tracing::error!(%error, "destroying a mapped-table upload event");
                }
            }
        }
    }
}

/// A step's rows gathering into a ring slot; [`MappedTableDevice::finish`]
/// queues their upload. Dropping it waits for the gather (no upload).
pub(crate) struct PendingRows {
    gather: Option<PendingGather>,
    slot: usize,
    bytes: usize,
    decode: bool,
}

/// One mapped table with its gather pool, optional hot-row cache and
/// prefetcher, and the staging ring that uploads each step's rows.
pub(crate) struct MappedTableDevice<'a> {
    name: &'static str,
    table: Arc<MappedTable>,
    pool: Arc<GatherPool>,
    cache: Option<Arc<HotRowCache>>,
    prefetcher: Option<TablePrefetcher>,
    ring: RefCell<PinnedRowRing<'a>>,
    logged: Cell<TableStatsSnapshot>,
    warm: Option<(Arc<AtomicBool>, std::thread::JoinHandle<()>)>,
}

impl Drop for MappedTableDevice<'_> {
    fn drop(&mut self) {
        if let Some((stop, worker)) = self.warm.take() {
            stop.store(true, Ordering::Relaxed);
            let _ = worker.join();
        }
    }
}

impl<'a> MappedTableDevice<'a> {
    /// `max_rows` bounds one step's gather (the ring slot size).
    pub fn new(library: &'a NativeLibrary, name: &'static str, table: MappedTable, args: &MappedTableArgs,
        max_rows: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("mapped-table");
        table.name_stats(name);
        let row_bytes = table.row_bytes();
        let prefetcher = if args.prefetch {
            // Prefill chunks and every decode row's next token; a row spans at most two pages.
            Some(TablePrefetcher::new(&format!("{name}-prefetch"), 16, max_rows, 2 * max_rows)?)
        } else {
            None
        };
        let table = Arc::new(table);
        let warm = if args.warm {
            let stop = Arc::new(AtomicBool::new(false));
            let (reader, stopping) = (table.clone(), stop.clone());
            let rate = (args.warm_mib_s > 0).then_some(args.warm_mib_s << 20);
            let worker = std::thread::Builder::new().name(format!("{name}-warm")).spawn(move || {
                let started = Instant::now();
                match reader.warm(&stopping, rate) {
                    Ok(bytes) => tracing::info!(table = name, gib = bytes as f64 / (1u64 << 30) as f64,
                        elapsed_ms = started.elapsed().as_millis() as u64,
                        stopped = stopping.load(Ordering::Relaxed), "mapped table warm"),
                    Err(error) => tracing::warn!(table = name, %error, "mapped table warm-up failed"),
                }
            })?;
            Some((stop, worker))
        } else {
            None
        };
        Ok(Self {
            name,
            pool: Arc::new(GatherPool::new(&format!("{name}-gather"), args.threads)?),
            cache: HotRowCache::new(row_bytes, args.cache_mib << 20).map(Arc::new),
            prefetcher,
            ring: RefCell::new(PinnedRowRing::new(library, max_rows * row_bytes, 2)?),
            logged: Cell::new(table.stats().snapshot()),
            table,
            warm,
        })
    }

    pub fn max_rows(&self) -> usize {
        self.ring.borrow().slot_bytes() / self.table.row_bytes()
    }

    fn rows(ids: &[i64]) -> Result<Vec<u64>> {
        ids.iter().map(|&id| u64::try_from(id).context("negative table row")).collect()
    }

    /// Starts gathering `ids` (in order, duplicates kept) into the next ring
    /// slot on the gather pool; the caller enqueues GPU work that precedes
    /// the rows' use, then [`Self::finish`]es.
    pub fn begin(&self, ids: &[i64], decode: bool) -> Result<PendingRows> {
        ensure!(ids.len() <= self.max_rows(), "{} {} rows exceed the step's {}", ids.len(), self.name, self.max_rows());
        let rows = Self::rows(ids)?;
        let bytes = rows.len() * self.table.row_bytes();
        let (slot, host) = self.ring.borrow_mut().acquire()?;
        // SAFETY: the slot holds `bytes`; the ring hands it out again only
        // after this PendingRows is finished (its copy event) or dropped (the
        // gather waited), and nothing else touches it meanwhile.
        let gather = unsafe { self.pool.spawn_gather(self.table.clone(), rows, host, bytes, self.cache.clone()) };
        Ok(PendingRows { gather: Some(gather), slot, bytes, decode })
    }

    /// Waits for the gather (the wait counts as stall) and queues the rows'
    /// copy into `dst` on `stream`; returns the uploaded bytes.
    ///
    /// # Safety
    /// As [`PinnedRowRing::submit`] for `dst` and `stream`.
    pub unsafe fn finish(&self, mut pending: PendingRows, dst: CuteafdDeviceBuffer, stream: *mut c_void)
        -> Result<usize> {
        let waited = Instant::now();
        pending.gather.take().context("rows already finished")?.wait()?;
        let elapsed = waited.elapsed();
        self.table.stats().record_stall(elapsed);
        if pending.decode { self.table.stats().record_decode_stall(elapsed); }
        // SAFETY: the gather completed; dst/stream per this function's contract.
        unsafe { self.ring.borrow_mut().submit(pending.slot, dst, pending.bytes, stream)? };
        Ok(pending.bytes)
    }

    /// Best-effort page-cache advice for rows a later step will gather
    /// (truncated to one step's rows; dropped when the queue is full).
    pub fn prefetch(&self, ids: &[i64]) -> Result<()> {
        let Some(prefetcher) = &self.prefetcher else { return Ok(()) };
        if ids.is_empty() {
            return Ok(());
        }
        if !prefetcher.submit_detached(self.table.clone(), &Self::rows(ids)?)? {
            self.table.stats().record_prefetch_dropped();
        }
        Ok(())
    }

    /// Logs the stats since the previous call (`event` names the interval).
    pub fn log_interval(&self, event: &str) {
        let now = self.table.stats().snapshot();
        let d = now.since(&self.logged.replace(now));
        if d.rows == 0 && d.prefetch_jobs == 0 {
            return;
        }
        tracing::info!(table = self.name, event, rows = d.rows, gathers = d.gathers, mib = d.bytes as f64 / (1 << 20) as f64,
            gather_ms = d.gather_ns as f64 / 1e6, stall_ms = d.stall_ns as f64 / 1e6, major_faults = d.major_faults,
            minor_faults = d.minor_faults, resident = d.resident_rate(), prefetch_pages = d.prefetch_pages,
            prefetch_dropped = d.prefetch_dropped, cache_hit = d.cache_hit_rate().unwrap_or(f64::NAN),
            "mapped table");
    }
}
