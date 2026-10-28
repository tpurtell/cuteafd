//! Token-table adapter: early bounded gather, host-triggered H2D at the consumer.
use anyhow::{ensure, Context, Result};
use super::memory::device::{Device, Stream};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, PinnedBytes};
use cuteafd_loader::{GatherFailure, GatherPoll, GatherTicket, MappedTable, TablePrefetcher, TableStatsSnapshot};
use cuteafd_loader::shared::token_table::{RowGather, TablePipeline};
use std::{cell::{Cell, RefCell}, ffi::c_void, sync::{Arc, Mutex}, time::Duration};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct MappedTableArgs {
    /// Host threads for large mapped-row gathers; small gathers run on the owner.
    #[arg(long = "table-threads", alias = "ple-threads", default_value_t = 16)]
    pub threads: usize,
    #[arg(long = "table-prefetch", default_value_t = true, action = clap::ArgAction::Set)]
    pub prefetch: bool,
}
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
    table: Arc<MappedTable>,
    prefetcher: Option<TablePrefetcher>,
    prefill: Arc<Pipeline>,
    decode: Arc<Pipeline>,
    prefill_rows: usize,
    decode_rows: usize,
    prefill_slots: usize,
    logged: Cell<TableStatsSnapshot>,
    ahead: RefCell<Vec<(Vec<i64>, PendingRows)>>,
    destinations: RefCell<Vec<Destination>>,
    active_destination: Cell<usize>,
    upload: Stream<'a>,
    landed: *mut c_void,
}
impl<'a> MappedTableDevice<'a> {
    pub fn new(library: &'a NativeLibrary, name: &'static str, table: MappedTable, args: &MappedTableArgs,
        prefill_rows: usize, prefill_slots: usize, decode_rows: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("mapped-table");
        let staging = cuteafd_loader::formats::mapped_table::MappedStaging::new(prefill_rows, prefill_slots, decode_rows, table.row_bytes())?;
        super::memory::budget::admit_host_memory(0, &[("staging/mapped-table", staging.total_bytes as u64)], 0, 0)?;
        let device = library.cuda_get_device()?;
        table.name_stats(name);
        let table = Arc::new(table);
        let pipeline = |rows: usize, slots: usize, suffix: &str| -> Result<Arc<Pipeline>> {
            let gather = RowGather::new(&format!("{name}-{suffix}-pages"), args.threads)?;
            let table = Arc::clone(&table);
            let storage = (0..slots).map(|_| Ok(RowSlot { bytes: library.pinned_bytes(rows * table.row_bytes())? }))
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(TablePipeline::new(&format!("{name}-{suffix}"), storage, move |slot: &mut RowSlot, job: &RowJob| {
                let ids = job.batch.source.rows(job.unit);
                let rows = ids.iter().map(|&id| u64::try_from(id)).collect::<std::result::Result<Vec<_>, _>>()?;
                let bytes = rows.len() * table.row_bytes();
                gather.gather(&table, &rows, &mut slot.bytes.bytes_mut()[..bytes])
            }, |slot| Ok(slot.bytes.copy_ready()?))?))
        };
        let prefill = pipeline(prefill_rows, 2 * prefill_slots, "prefill")?;
        let decode = pipeline(decode_rows, 2, "decode")?;
        let max_rows = prefill_rows.max(decode_rows);
        let prefetcher = if args.prefetch { Some(TablePrefetcher::new(&format!("{name}-prefetch"), 16, max_rows, 2 * max_rows)?) } else { None };
        Ok(Self { library, device, name, logged: Cell::new(table.stats().snapshot()), table, prefetcher,
            prefill, decode, prefill_rows, decode_rows, prefill_slots, ahead: RefCell::new(Vec::new()),
            destinations: RefCell::new(Vec::new()), active_destination: Cell::new(0),
            upload: Stream::new(Device { library, id: device })?,
            landed: library.cuda_event_create_ordering()? })
    }
    pub fn lookahead(&self) -> usize { self.prefill_slots }
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
        ensure!(ids.len() <= capacity && ids.iter().all(|&id| id >= 0 && (id as u64) < self.table.rows()), "invalid table rows");
        if !decode {
            let mut ahead = self.ahead.borrow_mut();
            if let Some(index) = ahead.iter().position(|(rows, _)| rows == ids) {
                *entry = true;
                return Ok(ahead.remove(index).1);
            }
        }
        let pipeline = Arc::clone(if decode { &self.decode } else { &self.prefill });
        let job = RowJob { batch: Arc::clone(batch), unit };
        let mut ticket = pipeline.try_submit(job.clone(), true)?;
        // Interleaved requests may leave both bursts cached for another history.
        // Demand must displace unused lookahead rather than wait on its tickets.
        while ticket.is_none() && !decode && !self.ahead.borrow().is_empty() {
            self.ahead.borrow_mut().remove(0);
            ticket = pipeline.try_submit(job.clone(), true)?;
        }
        *entry = true;
        Ok(PendingRows { ticket, job, pipeline, bytes: ids.len() * self.table.row_bytes() })
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
        self.table.stats().record_stall(started.elapsed());
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
    pub fn prefetch(&self, ids: &[i64]) -> Result<()> {
        let _device = Device { library: self.library, id: self.device }.enter()?;
        if ids.is_empty() { return Ok(()); }
        // Immutable ID equality makes this safe across interleaved requests and
        // prefix restores. Stale lookahead is cancelled, never published as state.
        // The scheduler calls this for the next chunk before executing this one.
        if ids.len() > self.decode_rows {
            for unit in ids.chunks(self.prefill_rows).take(self.prefill_slots) {
                if self.ahead.borrow().iter().any(|(rows, _)| rows == unit) { continue; }
                // The scheduler announces the following chunk before consuming
                // this one's cached rows. Retain both bursts through that handoff.
                if self.ahead.borrow().len() >= 2 * self.prefill_slots { self.ahead.borrow_mut().remove(0); }
                let batch = self.batch(Arc::new(unit.to_vec()));
                let job = RowJob { batch, unit: 0 };
                if let Some(ticket) = self.prefill.try_submit(job.clone(), false)? {
                    self.ahead.borrow_mut().push((unit.to_vec(), PendingRows { ticket: Some(ticket), job, pipeline: Arc::clone(&self.prefill),
                        bytes: unit.len() * self.table.row_bytes() }));
                }
            }
        }
        let Some(prefetcher) = &self.prefetcher else { return Ok(()) };
        let rows = ids.iter().map(|&id| u64::try_from(id).context("negative table row")).collect::<Result<Vec<_>>>()?;
        if !prefetcher.submit_detached(self.table.clone(), &rows)? { self.table.stats().record_prefetch_dropped(); }
        Ok(())
    }
    pub fn log_interval(&self, event: &str) {
        let now = self.table.stats().snapshot();
        let d = now.since(&self.logged.replace(now));
        if d.rows == 0 && d.prefetch_jobs == 0 { return; }
        tracing::info!(table = self.name, event, rows = d.rows, gathers = d.gathers,
            gather_ms = d.gather_ns as f64 / 1e6, stall_ms = d.stall_ns as f64 / 1e6, "mapped table");
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
    fn scheduler_lookahead_is_gathered_before_current_chunk_and_cannot_starve_demand() {
        let scheduler = include_str!("qwen/scheduler.rs");
        let round = scheduler.split("let chunk = &p.tokens[p.done..end];").nth(1).unwrap();
        assert!(round.find("engine.prefetch(&p.tokens[..end], &p.tokens[end..next])").unwrap()
            < round.find("engine.prefill_device(&mut p.placement, chunk").unwrap());
        let source = include_str!("mapped_table.rs");
        let prefetch = source.split("pub fn prefetch(&self").nth(1).unwrap().split("pub fn log_interval").next().unwrap();
        assert!(prefetch.contains("2 * self.prefill_slots"), "both current and next bursts survive scheduler handoff");
        assert!(prefetch.contains("self.prefill.try_submit(job.clone(), false)"));
        let begin = source.split("pub fn begin_unit(&self").nth(1).unwrap().split("pub fn begin(&self").next().unwrap();
        assert!(begin.contains("while ticket.is_none() && !decode && !self.ahead.borrow().is_empty()"));
        assert!(begin.contains("self.ahead.borrow_mut().remove(0)"));
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
