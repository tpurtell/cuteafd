//! Token-table adapter: rows read ahead into the table's bounded row cache
//! (from admission for prompts, as soon as decode tokens are known), each step's
//! gather a cache lookup plus copy into pinned staging, and a host-triggered
//! H2D at the consumer.
use anyhow::{ensure, Context, Result};
use super::memory::device::{Device, Stream};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, PinnedBytes};
use cuteafd_loader::{GatherFailure, GatherPoll, GatherTicket, MappedTable, TableStatsSnapshot};
use cuteafd_loader::formats::mapped_table::ReadPriority;
use cuteafd_loader::formats::mapped_table::lookahead::{LookaheadHandle, RowSource as AheadSource, UnitPlan};
use cuteafd_loader::formats::mapped_table::token_table::{TokenTable, TokenTableLookahead};
use cuteafd_loader::shared::token_table::TablePipeline;
use std::{cell::{Cell, RefCell}, ffi::c_void, sync::{Arc, Mutex}, time::Duration};
pub(crate) trait RowSource: Send + Sync {
    fn units(&self) -> usize;
    fn rows(&self, unit: usize) -> &[i64];
}
impl RowSource for Vec<i64> {
    fn units(&self) -> usize { 1 }
    fn rows(&self, unit: usize) -> &[i64] { assert_eq!(unit, 0); self }
}
pub(crate) struct RowBatch { source: Arc<dyn RowSource>, begun: Mutex<Vec<bool>> }
#[derive(Clone)]
struct RowJob { batch: Arc<RowBatch>, unit: usize }
struct RowSlot { bytes: PinnedBytes }
type Pipeline = TablePipeline<RowSlot, RowJob>;
type Ticket = GatherTicket<RowSlot, RowJob, anyhow::Error>;
pub(crate) struct PendingRows { ticket: Option<Ticket>, job: RowJob, pipeline: Arc<Pipeline>, bytes: usize }

struct Destination { pointer: usize, consumed: *mut c_void, recorded: bool }
pub(crate) struct MappedTableDevice<'a> {
    library: &'a NativeLibrary,
    device: i32,
    name: &'static str,
    lookahead: TokenTableLookahead,
    prefill: Arc<Pipeline>,
    decode: Arc<Pipeline>,
    prefill_rows: usize,
    decode_rows: usize,
    prefill_slots: usize,
    host_workspace_bytes: usize,
    cache_bytes: usize,
    logged: Cell<TableStatsSnapshot>,
    destinations: RefCell<Vec<Destination>>,
    active_destination: Cell<usize>,
    upload: Stream<'a>,
    landed: *mut c_void,
}
impl<'a> MappedTableDevice<'a> {
    /// `prefill_rows`/`decode_rows` are table rows per staged prefill unit and
    /// decode step; `prefill_slots` units run in flight; `rows_per_token`
    /// table rows per token size the row cache.
    pub fn new(library: &'a NativeLibrary, name: &'static str, table: MappedTable,
        prefill_rows: usize, prefill_slots: usize, decode_rows: usize, rows_per_token: usize) -> Result<Self> {
        table.prepare_reads();
        let _scope = cuteafd_ffi::memory_ledger::scope("mapped-table");
        let staging = cuteafd_loader::formats::mapped_table::MappedStaging::new(prefill_rows, prefill_slots, decode_rows, table.row_bytes())?;
        let shares = TokenTable::shares(prefill_rows / rows_per_token.max(1), prefill_slots,
            decode_rows / rows_per_token.max(1), rows_per_token);
        let cache_bytes = TokenTable::cache_bytes(table.row_bytes(), shares);
        super::memory::budget::admit_host_memory(0, &[("staging/mapped-table", staging.total_bytes as u64),
            ("mapped-table/row-cache", cache_bytes as u64)], 0, 0)?;
        let device = library.cuda_get_device()?;
        table.name_stats(name);
        let logged = Cell::new(table.stats().snapshot());
        let table = TokenTable::new(table, shares, rows_per_token)?;
        let pipeline = |rows: usize, slots: usize, suffix: &str, priority: ReadPriority| -> Result<Arc<Pipeline>> {
            let table = Arc::clone(&table);
            let storage = (0..slots).map(|_| Ok(RowSlot { bytes: library.pinned_bytes(rows * table.row_bytes())? }))
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(TablePipeline::new(&format!("{name}-{suffix}"), storage, move |slot: &mut RowSlot, job: &RowJob| {
                let ids = job.batch.source.rows(job.unit);
                let rows = ids.iter().map(|&id| u64::try_from(id)).collect::<std::result::Result<Vec<_>, _>>()?;
                let bytes = rows.len() * table.row_bytes();
                table.gather(&rows, &mut slot.bytes.bytes_mut()[..bytes], priority)
            }, |slot| Ok(slot.bytes.copy_ready()?))?))
        };
        let prefill = pipeline(prefill_rows, 2 * prefill_slots, "prefill", ReadPriority::Prefill)?;
        let decode = pipeline(decode_rows, 2, "decode", ReadPriority::Decode)?;
        let lookahead = TokenTableLookahead::new(name, table)?;
        Ok(Self { library, device, name, logged, lookahead,
            prefill, decode, prefill_rows, decode_rows, prefill_slots, host_workspace_bytes: staging.workspace_bytes,
            cache_bytes,
            destinations: RefCell::new(Vec::new()), active_destination: Cell::new(0),
            upload: Stream::new(Device { library, id: device })?,
            landed: library.cuda_event_create_ordering()? })
    }
    pub fn host_workspace_bytes(&self) -> usize { self.host_workspace_bytes }
    /// The row cache's host bytes (admitted at construction).
    pub fn cache_bytes(&self) -> usize { self.cache_bytes }
    /// Prefill units staged ahead of the one running.
    pub fn depth(&self) -> usize { self.prefill_slots }
    /// Read a prompt's rows (`source`, from absolute position `start`) into the
    /// row cache ahead of its gathers, in its prefill units.
    pub fn start_lookahead(&self, tag: u64, start: u64, source: Box<dyn AheadSource>, units: UnitPlan) -> LookaheadHandle {
        self.lookahead.start(tag, start, source, units)
    }
    /// Read a decode step's rows into the row cache now. Never blocks.
    pub fn prefetch_decode(&self, ids: &[i64]) -> Result<()> {
        let rows = ids.iter().map(|&id| u64::try_from(id).context("negative table row")).collect::<Result<Vec<_>>>()?;
        self.lookahead.decode(rows);
        Ok(())
    }
    pub fn batch(&self, source: Arc<dyn RowSource>) -> Arc<RowBatch> {
        Arc::new(RowBatch { begun: Mutex::new(vec![false; source.units()]), source })
    }
    pub fn begin_unit(&self, batch: &Arc<RowBatch>, unit: usize, decode: bool) -> Result<PendingRows> {
        let _device = Device { library: self.library, id: self.device }.enter()?;
        let mut begun = batch.begun.lock().unwrap_or_else(|p| p.into_inner());
        let entry = begun.get_mut(unit).context("invalid table unit")?;
        ensure!(!*entry, "table unit already begun");
        let ids = batch.source.rows(unit);
        let capacity = if decode { self.decode_rows } else { self.prefill_rows };
        let table = self.lookahead.table();
        ensure!(ids.len() <= capacity && ids.iter().all(|&id| id >= 0 && (id as u64) < table.table().rows()), "invalid table rows");
        let pipeline = Arc::clone(if decode { &self.decode } else { &self.prefill });
        let job = RowJob { batch: Arc::clone(batch), unit };
        let ticket = pipeline.try_submit(job.clone(), true)?;
        *entry = true;
        Ok(PendingRows { ticket, job, pipeline, bytes: ids.len() * table.row_bytes() })
    }
    pub fn begin(&self, source: Arc<dyn RowSource>, decode: bool) -> Result<PendingRows> {
        self.begin_unit(&self.batch(source), 0, decode)
    }
    /// Called after independent preceding-layer work is enqueued, immediately
    /// before PLE. No copy-engine operation waits behind a host readiness flag.
    /// # Safety
    /// dst and stream are on this device and remain live until consumption.
    pub unsafe fn finish(&self, mut pending: PendingRows, dst: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<usize> {
        let _device = Device { library: self.library, id: self.device }.enter()?;
        ensure!(!self.library.is_quarantined_after_failed_drain(), "token-table device is quarantined");
        ensure!(dst.device_id == self.device && dst.bytes >= pending.bytes, "token-table destination geometry");
        let started = std::time::Instant::now();
        let mut ticket = match pending.ticket.take() {
            Some(ticket) => ticket,
            None => pending.pipeline.submit_timeout(pending.job.clone(), true, Duration::from_secs(120))?,
        };
        let mut lease = match ticket.wait_timeout(Duration::from_secs(120)) {
            Ok(GatherPoll::Ready(lease)) => lease,
            Ok(_) => anyhow::bail!("token-table gather cancelled"),
            Err(GatherFailure::Job(error)) => return Err(error),
            Err(GatherFailure::Worker(error)) => return Err(error.into()),
        };
        self.lookahead.table().table().stats().record_stall(started.elapsed());
        let slot = lease.slot_mut().context("empty table lease")?;
        let mut destinations = self.destinations.borrow_mut();
        let index = match destinations.iter().position(|entry| entry.pointer == dst.ptr as usize) {
            Some(index) => index,
            None => { destinations.push(Destination { pointer: dst.ptr as usize,
                consumed: self.library.cuda_event_create_ordering()?, recorded: false }); destinations.len() - 1 }
        };
        let destination = &destinations[index];
        if destination.recorded {
            let deadline = std::time::Instant::now() + Duration::from_secs(120);
            // Only the previous PLE reader, never unrelated execution-stream work.
            // SAFETY: destination owns this event on the current device.
            while !unsafe { self.library.cuda_event_query(destination.consumed)? } {
                ensure!(std::time::Instant::now() < deadline, "token-table consumer gate timed out");
                std::thread::sleep(Duration::from_micros(50));
            }
        }
        self.active_destination.set(index);
        // SAFETY: the lease owns the pinned source, and remains retained through
        // copied_on even when a partially enqueued copy fails.
        let copied = unsafe { super::token_table::enqueue(self.library, slot.bytes.buffer(),
            &[super::token_table::CopyPlane { destination: dst, offset: 0, bytes: pending.bytes }], self.upload.raw) };
        // SAFETY: this stream contains all reads from the source.
        let fenced = unsafe { slot.bytes.copied_on(self.upload.raw) };
        pending.pipeline.retain_copy(lease);
        if copied.is_err() || fenced.is_err() {
            // SAFETY: failed enqueues must drain before destination teardown.
            if unsafe { self.library.cuda_stream_synchronize(self.upload.raw) }.is_err() { self.library.quarantine_module_after_failed_drain(); }
        }
        copied?; fenced?;
        // SAFETY: upload is already issued on a stream with no pending gate.
        // The consuming stream alone joins its landing, after earlier work.
        let landed = unsafe {
            self.library.cuda_event_record(self.landed, self.upload.raw)
                .and_then(|()| self.library.cuda_stream_wait_event(stream, self.landed))
        };
        if landed.is_err() && self.upload.drain().is_err() {
            self.library.quarantine_module_after_failed_drain();
        }
        landed?;
        Ok(pending.bytes)
    }
    /// Record immediately after the PLE reader (or its captured segment).
    /// # Safety
    /// stream contains the reader of the most recently landed destination.
    pub unsafe fn consumed(&self, stream: *mut c_void) -> Result<()> {
        let _device = Device { library: self.library, id: self.device }.enter()?;
        let mut destinations = self.destinations.borrow_mut();
        let destination = destinations.get_mut(self.active_destination.get()).context("PLE consumption without landing")?;
        // SAFETY: event and stream belong to this device and outlive queued work.
        if let Err(error) = unsafe { self.library.cuda_event_record(destination.consumed, stream) } {
            // An unproved reader must never permit another destination overwrite.
            self.library.quarantine_module_after_failed_drain();
            return Err(error);
        }
        destination.recorded = true;
        Ok(())
    }
    pub fn log_interval(&self, event: &str) {
        let now = self.lookahead.table().table().stats().snapshot();
        let d = now.since(&self.logged.replace(now));
        if d.cache_hit_rows + d.cache_wait_rows + d.lookahead_rows + d.stall_prefill_fallback_rows
            + d.stall_decode_fallback_rows == 0 { return; }
        // Gather side: rows the gathers read themselves (fallback) or waited for
        // another reader to fill (pending); cache_wait_rows also counts the
        // look-ahead finding a gather ahead of it. Device side: rows read from
        // the table and their summed service time.
        tracing::info!(table = self.name, event, cache_hit_rows = d.cache_hit_rows, cache_wait_rows = d.cache_wait_rows,
            lookahead_rows = d.lookahead_rows, prefill_fallback_rows = d.stall_prefill_fallback_rows,
            prefill_pending_rows = d.stall_prefill_pending_rows, decode_fallback_rows = d.stall_decode_fallback_rows,
            decode_pending_rows = d.stall_decode_pending_rows, device_rows = d.page_misses,
            device_ms = d.miss_ns as f64 / 1e6, stall_ms = d.stall_ns as f64 / 1e6, "mapped table");
    }
}

impl Drop for MappedTableDevice<'_> {
    fn drop(&mut self) {
        let Ok(_device) = (Device { library: self.library, id: self.device }).enter() else {
            self.library.quarantine_module_after_failed_drain();
            return;
        };
        let drained = (|| -> Result<()> {
            // SAFETY: owner retains upload stream and all destination events.
            unsafe {
                self.library.cuda_stream_synchronize(self.upload.raw)?;
                for destination in self.destinations.get_mut().iter().filter(|d| d.recorded) {
                    self.library.cuda_event_synchronize(destination.consumed)?;
                }
            }
            Ok(())
        })();
        if drained.is_err() { self.library.quarantine_module_after_failed_drain(); return; }
        // SAFETY: all producers/readers have drained.
        unsafe {
            let _ = self.library.cuda_event_destroy(self.landed);
            for destination in self.destinations.get_mut() { let _ = self.library.cuda_event_destroy(destination.consumed); }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn qwen_host_admission_retains_mapped_worker_workspace() {
        let source = include_str!("mapped_table.rs");
        let constructor = source.split("pub fn new(library:").nth(1).unwrap()
            .split("pub fn host_workspace_bytes").next().unwrap();
        assert!(constructor.contains("host_workspace_bytes: staging.workspace_bytes"));
        let engine = include_str!("../families/qwen4/engine.rs");
        let demands = engine.split("fn future_host_workspaces(&self").nth(1).unwrap()
            .split("/// Size every unit-table upload").next().unwrap();
        assert!(demands.contains("ple.mapped().host_workspace_bytes() as u64"));
        assert!(demands.contains("staging/mapped-table-worker"));
    }

    #[test]
    fn prompts_read_ahead_from_admission_and_gathers_go_through_the_row_cache() {
        let scheduler = include_str!("qwen/scheduler.rs");
        let admit = scheduler.split("// Family table rows for the whole prompt are read from admission").nth(1).unwrap();
        assert!(admit.find("engine.read_ahead(&placement, &tokens, resume, &plan.chunks)").unwrap()
            < admit.find("prefills.push(Prefill {").unwrap());
        assert!(!scheduler.contains("engine.prefetch("), "per-chunk bursts are gone");
        let source = include_str!("mapped_table.rs");
        let new = source.split("pub fn new(library:").nth(1).unwrap().split("pub fn host_workspace_bytes").next().unwrap();
        assert!(new.contains("table.gather(&rows, &mut slot.bytes.bytes_mut()[..bytes], priority)"));
        assert!(new.contains("(\"mapped-table/row-cache\", cache_bytes as u64)"));
    }

    #[test]
    fn landing_gates_previous_consumer_not_execution_tail() {
        let source = include_str!("mapped_table.rs");
        let finish = source.split("pub unsafe fn finish(").nth(1).unwrap().split("pub unsafe fn consumed(").next().unwrap();
        assert!(finish.contains("cuda_event_query(destination.consumed)"));
        assert!(!finish.contains("cuda_stream_synchronize(stream)"));
        assert!(finish.contains("cuda_stream_wait_event(stream, self.landed)"));
        let engine = include_str!("../families/qwen4/engine.rs");
        let ple = engine.split("    fn ple(").nth(1).unwrap().split("    fn w8(").next().unwrap();
        assert!(ple.contains("table.mapped().consumed(self.execution_stream())"));
    }
}
