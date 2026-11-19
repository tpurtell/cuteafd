//! Optional, reusable GPU events for the native expert boundary.
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

/// Always-on expert-side counters.
///
/// The per-request `Spark expert compute` debug line wrote a line per request to syslog when
/// `CUTEAFD_EXPERT_TIMING=1`. These counters carry the same quantities — execute time, the GPU
/// gap (pace wait), waves and rows — for a few relaxed atomic adds per unit, and the expert
/// serve loop prints them on ONE bounded line per second instead of one line per request.
///
/// **Basis.** The CUDA-event measurement is the cost, so it stays behind its existing gate
/// (`CUTEAFD_EXPERT_TIMING` / the `cuteafd::expert_timing` debug target); what changes is that
/// enabling it now fills counters and one line/second instead of one line/request. The counters
/// themselves are four relaxed `fetch_add`s at ~20 ns each = ~80 ns per unit, against a unit's
/// own GPU compute floor of >= 100 us — <= 0.08 %, and they allocate nothing.
pub(crate) mod expert_stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    static EXECUTE_NS: AtomicU64 = AtomicU64::new(0);
    static PACE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
    static WAVES: AtomicU64 = AtomicU64::new(0);
    static ROWS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn record_unit(compute_us: f64, gpu_gap_us: Option<f64>, rows: u64) {
        EXECUTE_NS.fetch_add((compute_us.max(0.0) * 1000.0) as u64, Ordering::Relaxed);
        if let Some(gap) = gpu_gap_us {
            PACE_WAIT_NS.fetch_add((gap.max(0.0) * 1000.0) as u64, Ordering::Relaxed);
        }
        WAVES.fetch_add(1, Ordering::Relaxed);
        ROWS.fetch_add(rows, Ordering::Relaxed);
    }

    pub(crate) fn snapshot() -> serde_json::Value {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        serde_json::json!({
            "expert_ns_total": {"execute": get(&EXECUTE_NS), "pace_wait": get(&PACE_WAIT_NS)},
            "expert_waves_total": get(&WAVES),
            "expert_rows_total": get(&ROWS),
        })
    }
}

pub(crate) struct Event<'a> {
    library: &'a NativeLibrary,
    raw: *mut c_void,
}
impl<'a> Event<'a> {
    fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self {
            library,
            raw: library.cuda_event_create()?,
        })
    }

    fn ordering(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self { library, raw: library.cuda_event_create_ordering()? })
    }
}
impl Drop for Event<'_> {
    fn drop(&mut self) {
        if let Err(error) = unsafe { self.library.cuda_event_destroy(self.raw) } {
            tracing::error!(%error, "destroying expert timing event");
        }
    }
}

/// The owner stream's last completed unit end, shared by every adapter that
/// queues on that stream (FP8 weight groups), so the idle gap follows the
/// stream rather than one group's queue.
pub(crate) type PaceClock<'a> = std::rc::Rc<std::cell::RefCell<Option<std::rc::Rc<Event<'a>>>>>;

/// One unit's measured pace on its owner stream (#762): busy from its
/// arrival (before its input copies) to its fence, idle from the previous
/// unit's fence on the same stream to its arrival (zero for the stream's
/// first unit, or when it was queued behind its predecessor).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnitPace { pub busy_us: u32, pub idle_us: u32 }

/// Prefill units: per staging slot an arrival and a fence. A fence is shared
/// with the pace clock, which retains the newest completed one; a slot whose
/// fence the clock still holds takes the spare before it is recorded again.
struct UnitEvents<'a> {
    events: Vec<(Event<'a>, std::rc::Rc<Event<'a>>)>,
    /// Always exactly one spare fence.
    spare: std::rc::Rc<Event<'a>>,
    /// Slot whose arrival was recorded and whose fence is not yet queued.
    arrived: Option<usize>,
    clock: PaceClock<'a>,
}

enum Completion<'a> {
    SingleFlight(*mut c_void),
    UnitEvents(UnitEvents<'a>),
}

/// Decode has one retained unit, so stream idleness is its completion fence
/// (decode owners report no pace). Prefill retains independent unit fences so
/// an earlier reply need not wait for later work, and records each unit's
/// arrival before its input copies, so its pace is read from the completed
/// events after its fence (no added wait). All events are allocated before
/// serving; hot-path submission never allocates.
pub(crate) struct PendingUnits<'a> {
    library: &'a NativeLibrary,
    completion: Completion<'a>,
    free: std::collections::VecDeque<usize>,
    reserved: [bool; cuteafd_core::spark_executions::MAX_BUFFERED_PREFILL_UNITS as usize],
    queued: std::collections::VecDeque<(usize, (u64, u32, u32, u64))>,
}
impl<'a> PendingUnits<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, capacity: usize) -> Result<Self> {
        Self::for_owner(library, capacity, cuteafd_core::ExecutionOwner::Prefill)
    }

    pub(crate) fn for_owner(library: &'a NativeLibrary, capacity: usize,
        owner: cuteafd_core::ExecutionOwner) -> Result<Self> {
        Self::with_clock(library, capacity, owner, Default::default())
    }

    /// A queue whose prefill idle gaps follow `clock` (one per owner stream).
    pub(crate) fn with_clock(library: &'a NativeLibrary, capacity: usize, owner: cuteafd_core::ExecutionOwner,
        clock: PaceClock<'a>) -> Result<Self> {
        anyhow::ensure!((1..=4).contains(&capacity), "invalid expert unit queue capacity");
        let completion = match owner {
            cuteafd_core::ExecutionOwner::Decode(_) => {
                anyhow::ensure!(capacity == 1, "decode completion requires a single retained unit");
                Completion::SingleFlight(std::ptr::null_mut())
            },
            cuteafd_core::ExecutionOwner::Prefill => Completion::UnitEvents(UnitEvents {
                events: (0..capacity).map(|_| Ok((Event::new(library)?, std::rc::Rc::new(Event::new(library)?))))
                    .collect::<Result<_>>()?,
                spare: std::rc::Rc::new(Event::new(library)?), arrived: None, clock }),
        };
        Ok(Self { library, completion,
            free: (0..capacity).collect(), reserved: [false; cuteafd_core::spark_executions::MAX_BUFFERED_PREFILL_UNITS as usize], queued: std::collections::VecDeque::with_capacity(capacity) })
    }
    /// The staging slot the next unit uses.
    pub(crate) fn next_slot(&self) -> Result<usize> {
        self.free.front().copied().ok_or_else(|| anyhow::anyhow!("expert unit queue exhausted"))
    }
    pub(crate) fn reserve(&mut self) -> Result<usize> {
        let slot = self.free.pop_front().ok_or_else(|| anyhow::anyhow!("expert unit queue exhausted"))?;
        self.reserved[slot] = true;
        Ok(slot)
    }
    pub(crate) fn cancel_reserved(&mut self, slot: usize) -> Result<()> {
        anyhow::ensure!(self.reserved.get(slot) == Some(&true), "unit is not staged");
        self.reserved[slot] = false;
        self.free.push_back(slot);
        Ok(())
    }
    /// # Safety
    /// The reserved slot and stream remain live through record_reserved/pop.
    pub(crate) unsafe fn arrive_reserved(&mut self, slot: usize, stream: *mut c_void) -> Result<()> {
        anyhow::ensure!(self.reserved.get(slot) == Some(&true), "unit is not staged");
        let Completion::UnitEvents(units) = &self.completion else { anyhow::bail!("decode cannot stage"); };
        // SAFETY: the reservation exclusively owns this event.
        unsafe { self.library.cuda_event_record(units.events[slot].0.raw, stream) }
    }
    /// # Safety
    /// Called only after the shared launch on its stream. The staged slot is
    /// retained until this member fence; cancellation is no longer permitted.
    pub(crate) unsafe fn record_reserved(&mut self, slot: usize, identity: (u64, u32, u32, u64),
        stream: *mut c_void) -> Result<()> {
        anyhow::ensure!(self.reserved.get(slot) == Some(&true), "unit is not staged");
        let Completion::UnitEvents(units) = &mut self.completion else {
            anyhow::bail!("decode cannot stage units");
        };
        if std::rc::Rc::strong_count(&units.events[slot].1) > 1 {
            std::mem::swap(&mut units.events[slot].1, &mut units.spare);
        }
        // SAFETY: the reserved slot owns both events until publication.
        unsafe {
            self.library.cuda_event_record(units.events[slot].1.raw, stream)?;
        }
        self.reserved[slot] = false;
        self.queued.push_back((slot, identity));
        Ok(())
    }
    pub(crate) fn is_empty(&self) -> bool { self.queued.is_empty() && !self.reserved.contains(&true) }
    pub(crate) fn len(&self) -> usize { self.queued.len() }
    /// Marks the next prefill unit's arrival on `stream`, before its input
    /// copies (decode: nothing).
    /// # Safety
    /// `stream` is the owner stream the unit's fence is later recorded on.
    pub(crate) unsafe fn arrive(&mut self, stream: *mut c_void) -> Result<()> {
        let slot = self.next_slot()?;
        if let Completion::UnitEvents(units) = &mut self.completion {
            // SAFETY: this slot's events are exclusively reserved until the unit is popped.
            unsafe { self.library.cuda_event_record(units.events[slot].0.raw, stream)?; }
            units.arrived = Some(slot);
        }
        Ok(())
    }
    /// # Safety
    /// The stream and every borrowed request/reply owner remain live until pop.
    pub(crate) unsafe fn record(&mut self, identity: (u64, u32, u32, u64), stream: *mut c_void) -> Result<()> {
        let slot = self.next_slot()?;
        match &mut self.completion {
            Completion::SingleFlight(owner_stream) => *owner_stream = stream,
            Completion::UnitEvents(units) => {
                anyhow::ensure!(units.arrived.take() == Some(slot), "expert unit fence without its arrival");
                // The clock may still hold this slot's previous fence (it is the
                // stream's newest completed end): give the slot the spare instead.
                if std::rc::Rc::strong_count(&units.events[slot].1) > 1 {
                    std::mem::swap(&mut units.events[slot].1, &mut units.spare);
                }
                // SAFETY: this event is exclusively reserved until the unit is popped.
                unsafe { self.library.cuda_event_record(units.events[slot].1.raw, stream)?; }
            },
        }
        self.free.pop_front();
        self.queued.push_back((slot, identity));
        Ok(())
    }
    pub(crate) fn ready(&self) -> Result<bool> {
        let Some((slot, _)) = self.queued.front() else { return Ok(!self.reserved.contains(&true)) };
        // SAFETY: the queued unit retains its owner stream or exclusive event.
        unsafe { match &self.completion {
            Completion::SingleFlight(stream) => self.library.cuda_stream_query(*stream),
            Completion::UnitEvents(units) => self.library.cuda_event_query(units.events[*slot].1.raw),
        } }
    }
    /// The oldest unit's identity and measured pace (decode: zero), once its fence completed.
    pub(crate) fn pop(&mut self) -> Result<Option<((u64, u32, u32, u64), UnitPace)>> {
        anyhow::ensure!(self.ready()?, "expert unit completed before its fence");
        let Some((slot, identity)) = self.queued.pop_front() else { return Ok(None) };
        let pace = match &self.completion {
            Completion::SingleFlight(_) => UnitPace { busy_us: 0, idle_us: 0 },
            Completion::UnitEvents(units) => {
                let (arrived, end) = &units.events[slot];
                let elapsed_us = |from: *mut c_void, to: *mut c_void| -> Result<u32> {
                    // SAFETY: both events were recorded on this owner stream and have
                    // completed: the fence is ready and every earlier record precedes it.
                    let ms = unsafe { self.library.cuda_event_elapsed_ms(from, to)? };
                    Ok((f64::from(ms).max(0.0) * 1000.0).min(f64::from(u32::MAX)) as u32)
                };
                let busy_us = elapsed_us(arrived.raw, end.raw)?;
                let idle_us = match units.clock.borrow().as_ref() {
                    Some(previous) => elapsed_us(previous.raw, arrived.raw)?,
                    None => 0,
                };
                *units.clock.borrow_mut() = Some(end.clone());
                UnitPace { busy_us, idle_us }
            }
        };
        self.free.push_back(slot);
        Ok(Some((identity, pace)))
    }
}

fn safe_host(host: &str) -> String {
    let name: String = host.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
        c
    } else { '_' }).collect();
    if name.is_empty() { "unknown".into() } else { name }
}

fn enabled(value: Option<&str>) -> bool { value == Some("1") }

#[derive(serde::Serialize)]
struct WorkerSample<'a> {
    event: &'static str,
    host: &'a str,
    rank: usize,
    request_id: u64,
    layer: u32,
    rows: u32,
    executor_id: u64,
    format: &'a str,
    completed_unix_ns: u64,
    compute_us: f32,
    previous_request_id: Option<u64>,
    previous_layer: Option<u32>,
    gpu_gap_us: Option<f32>,
}
impl WorkerSample<'_> {
    fn write_to(&self, output: &mut impl std::io::Write) -> Result<()> {
        serde_json::to_writer(&mut *output, self)?;
        output.write_all(b"\n")?;
        output.flush()?;
        Ok(())
    }
}

struct WorkerSpans<'a> {
    free: std::collections::VecDeque<usize>,
    recording: Option<usize>,
    queued: std::collections::VecDeque<usize>,
    previous_end: Option<std::rc::Rc<Event<'a>>>,
}

/// Startup-only diagnostics shared by every Spark weight format.
/// The measured pipeline includes device routing and output reduction, not host copies.
pub(crate) struct WorkerTiming<'a> {
    events: Vec<[std::rc::Rc<Event<'a>>; 2]>,
    state: std::cell::RefCell<WorkerSpans<'a>>,
    previous: Option<(u64, u32)>,
    rank: usize,
    host: String,
    output: Option<std::io::BufWriter<std::fs::File>>,
}
impl<'a> WorkerTiming<'a> {
    pub(crate) fn from_env(library: &'a NativeLibrary, rank: usize) -> Result<Option<Self>> {
        if !enabled(std::env::var("CUTEAFD_EXPERT_TIMING").ok().as_deref()) {
            return Ok(None);
        }
        let host = std::env::var("CUTEAFD_PROFILE_HOST").or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "unknown".into());
        let host = safe_host(&host);
        let output = std::env::var_os("CUTEAFD_PROFILE_DIR").map(|directory| {
            let path = std::path::Path::new(&directory).join(format!("expert-timing-{host}-rank{rank}.jsonl"));
            std::fs::OpenOptions::new().create(true).append(true).open(path).map(std::io::BufWriter::new)
        }).transpose()?;
        Ok(Some(Self::new(library, rank, host, output)?))
    }

    fn new(library: &'a NativeLibrary, rank: usize, host: String,
        output: Option<std::io::BufWriter<std::fs::File>>) -> Result<Self> {
        // One extra span retains the preceding end while up to four units queue.
        let count = usize::from(cuteafd_transport::worker_selection::MAX_BUFFERED_PREFILL_UNITS) + 1;
        let events = (0..count).map(|_| Ok([std::rc::Rc::new(Event::new(library)?),
            std::rc::Rc::new(Event::new(library)?)] )).collect::<Result<_>>()?;
        Ok(Self { events, state: std::cell::RefCell::new(WorkerSpans {
            free: (0..count).collect(), recording: None, queued: Default::default(), previous_end: None }),
            previous: None, rank, host, output })
    }

    /// The worker retains its stream and inputs until the existing completion barrier.
    pub(crate) unsafe fn record(&self, index: usize, stream: *mut c_void) -> Result<()> {
        let mut state = self.state.borrow_mut();
        let slot = if index == 0 {
            anyhow::ensure!(state.recording.is_none(), "expert timing span already recording");
            let slot = state.free.pop_front().ok_or_else(|| anyhow::anyhow!("expert timing queue exhausted"))?;
            state.recording = Some(slot);
            slot
        } else {
            state.recording.ok_or_else(|| anyhow::anyhow!("expert timing end without start"))?
        };
        let event = &self.events[slot][usize::from(index != 0)];
        // SAFETY: this event pair remains reserved through this unit's publication.
        unsafe { event.library.cuda_event_record(event.raw, stream)?; }
        if index != 0 { state.recording = None; state.queued.push_back(slot); }
        Ok(())
    }

    /// Called after this unit's completion fence; never adds a GPU wait.
    pub(crate) unsafe fn publish(&mut self, request_id: u64, layer: u32, rows: u32, executor_id: u64,
        format: &str) -> Result<()> {
        let mut state = self.state.borrow_mut();
        let slot = *state.queued.front().ok_or_else(|| anyhow::anyhow!("no completed expert timing span"))?;
        let [start, end] = &self.events[slot];
        // SAFETY: the caller completed this unit's event, not necessarily its stream.
        let compute_us = unsafe { start.library.cuda_event_elapsed_ms(start.raw, end.raw)? } * 1000.0;
        let gpu_gap_us = state.previous_end.as_ref().map(|previous_end| {
            // SAFETY: both retained records completed before this publication.
            unsafe { start.library.cuda_event_elapsed_ms(previous_end.raw, start.raw) }
                .map(|ms| ms * 1000.0)
        }).transpose()?;
        let unix_ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos().min(u64::MAX as u128) as u64;
        // The same quantities as counters, always (no per-request log line).
        expert_stats::record_unit(compute_us as f64, gpu_gap_us.map(f64::from), rows as u64);
        if let Some(output) = &mut self.output {
            WorkerSample { event: "expert_compute", host: &self.host, rank: self.rank,
                request_id, layer, rows, executor_id, format, completed_unix_ns: unix_ns,
                compute_us, previous_request_id: self.previous.map(|p| p.0),
                previous_layer: self.previous.map(|p| p.1), gpu_gap_us }.write_to(output)?;
        }
        // The per-request `Spark expert compute` debug line is gone: the counters above carry
        // it, and the serve loop prints them once a second.
        self.previous = Some((request_id, layer));
        state.previous_end = Some(end.clone());
        state.queued.pop_front();
        state.free.push_back(slot);
        Ok(())
    }
}

#[cfg(test)]
mod worker_tests {
    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn decode_stream_completion_preserves_credit_identity_and_timing_without_unit_events() -> anyhow::Result<()> {
        use super::*;
        use cuteafd_core::ExecutionOwner;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 20)?;
        fixture.queued_experts(&library)?;
        let stream = crate::shared::memory::LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let allocation = crate::shared::memory::DeviceAllocation::new(&library, 1, crate::shared::memory::TEST)?;
        let input = [13u8; 1];
        let mut pending = PendingUnits::for_owner(&library, 1, ExecutionOwner::Decode(1))?;
        assert!(matches!(&pending.completion, Completion::SingleFlight(_)));
        assert!(PendingUnits::for_owner(&library, 2, ExecutionOwner::Decode(0)).is_err());
        let prefill = PendingUnits::for_owner(&library, 1, ExecutionOwner::Prefill)?;
        assert!(matches!(&prefill.completion, Completion::UnitEvents(units) if units.events.len() == 1));
        let path = fixture.directory().join("decode-timing.jsonl");
        let output = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let mut timing = WorkerTiming::new(&library, 0, "fixture".into(), Some(output))?;
        for request_id in 0..3 {
            assert!(pending.ready()? && pending.is_empty());
            // SAFETY: the source, buffers, stream and timing records outlive completion.
            unsafe {
                timing.record(0, stream.raw)?;
                library.copy_h2d_async(allocation.buffer, &input, stream.raw)?;
                timing.record(1, stream.raw)?;
                pending.record((request_id, 20, 24, 27), stream.raw)?;
            }
            assert_eq!(pending.len(), 1);
            assert!(pending.next_slot().is_err() && !pending.ready()?);
            assert!(pending.pop().is_err());
            fixture.advance_queued_experts(&library, 1)?;
            assert!(pending.ready()?);
            let (identity, pace) = pending.pop()?.unwrap();
            assert_eq!(identity, (request_id, 20, 24, 27));
            // Decode owners report no pace (#762: prefill owner only).
            assert_eq!(pace, UnitPace { busy_us: 0, idle_us: 0 });
            assert_eq!(pending.next_slot()?, 0);
            // SAFETY: the exclusive owner stream is idle after both timing records.
            unsafe { timing.publish(identity.0, identity.1, identity.2, identity.3, "native")?; }
        }
        let records = std::fs::read_to_string(path)?.lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(records.len(), 3);
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record["request_id"], index as u64);
            assert!((record["compute_us"].as_f64().unwrap() - 1.0).abs() < 0.001);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn staged_descriptor_slots_cannot_publish_or_reuse_before_group_fences() -> anyhow::Result<()> {
        use super::*;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 20)?;
        fixture.queued_experts(&library)?;
        let stream = crate::shared::memory::LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let allocation = crate::shared::memory::DeviceAllocation::new(&library, 1, crate::shared::memory::TEST)?;
        let mut pending = PendingUnits::new(&library, 2)?;
        let first = pending.reserve()?;
        let cancelled = pending.reserve()?;
        assert!(!pending.ready()? && pending.next_slot().is_err());
        assert!(pending.pop().is_err());
        pending.cancel_reserved(cancelled)?;
        assert_eq!(pending.next_slot()?, cancelled);
        let second = pending.reserve()?;
        let input = [42u8];
        // SAFETY: fixture allocations and staging survive all queued copies and fences.
        unsafe {
            pending.arrive_reserved(first, stream.raw)?;
            pending.arrive_reserved(second, stream.raw)?;
            library.copy_h2d_async(allocation.buffer, &input, stream.raw)?;
            pending.record_reserved(first, (11, 1, 2752, 0), stream.raw)?;
            pending.record_reserved(second, (12, 1, 2752, 0), stream.raw)?;
        }
        assert!(pending.next_slot().is_err());
        assert!(pending.cancel_reserved(first).is_err());
        assert!(!pending.ready()? && pending.pop().is_err());
        fixture.advance_queued_experts(&library, 1)?;
        assert_eq!(pending.pop()?.unwrap().0.0, 11);
        assert_eq!(pending.pop()?.unwrap().0.0, 12);
        assert_eq!(pending.next_slot()?, first);
        Ok(())
    }

    #[test]
    fn typed_owner_selects_completion_for_every_spark_backend() {
        let local = include_str!("../service/local.rs");
        assert!(local.contains("extent.staging_rows as usize, extent.owner)?"));
        for source in [include_str!("../execution.rs"), include_str!("../exl3/worker.rs"),
            include_str!("../service/exlr8.rs")] {
            assert!(source.contains("PendingUnits::for_owner(self.library, units, owner)?"));
        }
        // FP8 weight groups on one owner stream share its pace clock.
        assert!(include_str!("../fp8/worker.rs")
            .contains("PendingUnits::with_clock(self.library, units, owner,\n            self.clock.clone())?"));
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn queued_spans_keep_fifo_identities_and_gpu_gaps_across_event_reuse() -> anyhow::Result<()> {
        use super::*;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        library.cuda_set_device(7)?;
        fixture.configure_memory(&library, 7, 1 << 20)?;
        fixture.queued_experts(&library)?;
        let path = fixture.directory().join("timing.jsonl");
        let output = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let mut timing = WorkerTiming::new(&library, 0, "fixture".into(), Some(output))?;
        let input = [13u8; 1];
        let allocation = crate::shared::memory::DeviceAllocation::new(&library, 1, crate::shared::memory::TEST)?;
        let stream = crate::shared::memory::LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let mut pending = PendingUnits::new(&library, 4)?;
        for round in 0..3 {
            for unit in 0..4 {
                // SAFETY: the source and destination outlive this stream drain.
                unsafe {
                    pending.arrive(stream.raw)?;
                    library.copy_h2d_async(allocation.buffer, &input, stream.raw)?;
                    timing.record(0, stream.raw)?;
                    library.copy_h2d_async(allocation.buffer, &input, stream.raw)?;
                    timing.record(1, stream.raw)?;
                    pending.record((round * 4 + unit, 3, 1, 0), stream.raw)?;
                }
            }
            assert_eq!(pending.len(), 4);
            assert!(pending.next_slot().is_err());
            for unit in 0..4 {
                assert!(!pending.ready()?);
                fixture.advance_queued_experts(&library, 2)?;
                let (identity, pace) = pending.pop()?.unwrap();
                assert_eq!(identity.0, round * 4 + unit);
                // Two queued copies per unit, back to back on the stream.
                assert!(pace.busy_us > 0 && pace.idle_us <= pace.busy_us, "{pace:?}");
                // SAFETY: the unit's fence is ready while later spans may be queued.
                unsafe { timing.publish(identity.0, identity.1, identity.2, identity.3, "fp8_moe")?; }
            }
        }
        let rows = std::fs::read_to_string(path)?.lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(rows.len(), 12);
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row["request_id"], index as u64);
            assert!((row["compute_us"].as_f64().unwrap() - 1.0).abs() < 0.001);
            if index == 0 { assert!(row["gpu_gap_us"].is_null()); }
            else {
                assert_eq!(row["previous_request_id"], index as u64 - 1);
                assert!((row["gpu_gap_us"].as_f64().unwrap() - 1.0).abs() < 0.001);
            }
        }
        Ok(())
    }

    #[test]
    fn worker_spans_emit_the_same_jsonl_contract_for_exlr8_and_existing_formats() -> anyhow::Result<()> {
        let mut output = Vec::new();
        for format in ["native", "exl3", "fp8_moe", "exlr8"] {
            super::WorkerSample { event: "expert_compute", host: "worker", rank: 2,
                request_id: 17, layer: 3, rows: 65, executor_id: 23, format,
                completed_unix_ns: 29, compute_us: 31.5, previous_request_id: Some(16),
                previous_layer: Some(3), gpu_gap_us: Some(2.5) }.write_to(&mut output)?;
        }
        let text = String::from_utf8(output)?;
        assert!(text.ends_with('\n'));
        let records = text.lines().map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(records.len(), 4);
        for (record, format) in records.iter().zip(["native", "exl3", "fp8_moe", "exlr8"]) {
            assert_eq!(record, &serde_json::json!({"event": "expert_compute", "host": "worker",
                "rank": 2, "request_id": 17, "layer": 3, "rows": 65, "executor_id": 23,
                "format": format, "completed_unix_ns": 29, "compute_us": 31.5,
                "previous_request_id": 16, "previous_layer": 3, "gpu_gap_us": 2.5}));
        }
        Ok(())
    }

    #[test]
    fn timing_requires_explicit_one() {
        assert!(super::enabled(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true")] {
            assert!(!super::enabled(value));
        }
    }

    #[test]
    fn host_cannot_escape_profile_directory() {
        assert_eq!(super::safe_host("ostrich"), "ostrich");
        assert_eq!(super::safe_host("../host/a"), "___host_a");
        assert_eq!(super::safe_host(""), "unknown");
    }
}

/// The wave owns these events and drains its stream before dropping them.
pub(super) struct ExpertTiming<'a> {
    events: [Event<'a>; 3],
}
impl<'a> ExpertTiming<'a> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self {
            events: [
                Event::new(library)?,
                Event::new(library)?,
                Event::new(library)?,
            ],
        })
    }
    pub unsafe fn record(&self, index: usize, stream: *mut c_void) -> Result<()> {
        let event = &self.events[index];
        unsafe { event.library.cuda_event_record(event.raw, stream) }
    }
    /// Call only after the owning stream's existing completion barrier.
    pub unsafe fn elapsed_us(&self) -> Result<(f32, f32)> {
        let [start, kernel_end, compact_end] = &self.events;
        Ok(unsafe {
            (
                1000.0
                    * start
                        .library
                        .cuda_event_elapsed_ms(start.raw, kernel_end.raw)?,
                1000.0
                    * start
                        .library
                        .cuda_event_elapsed_ms(kernel_end.raw, compact_end.raw)?,
            )
        })
    }
}

#[cfg(test)]
mod expert_stats_tests {
    use super::expert_stats;

    /// The counters are process-global atomics; the delta tests must not race each other.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // These two tests are counter-level, not behavioural, and that is deliberate: the only
    // caller of `record_unit` is `WorkerTiming::publish`, which reads CUDA events through a
    // real `NativeLibrary`. The crate's only fake library
    // (`cuteafd_ffi::native_library_lifetime_fixture::Fixture`) needs an allocated CPU build
    // slot and an NVMe fixture directory, so every test that drives `publish` is `#[ignore]`d
    // and cannot run in CI. There is no GPU-free seam to drive, so the unit conversion
    // (microseconds in, nanoseconds out) is what is asserted here.

    /// One unit moves execute time, pace wait, waves and rows — the quantities the
    /// per-request `Spark expert compute` log line used to carry.
    #[test]
    fn a_unit_moves_every_expert_counter() {
        let _guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let before = expert_stats::snapshot();
        expert_stats::record_unit(250.0, Some(40.0), 12);
        let after = expert_stats::snapshot();
        assert_eq!(after["expert_ns_total"]["execute"].as_u64().unwrap()
            - before["expert_ns_total"]["execute"].as_u64().unwrap(), 250_000);
        assert_eq!(after["expert_ns_total"]["pace_wait"].as_u64().unwrap()
            - before["expert_ns_total"]["pace_wait"].as_u64().unwrap(), 40_000);
        assert_eq!(after["expert_waves_total"].as_u64().unwrap()
            - before["expert_waves_total"].as_u64().unwrap(), 1);
        assert_eq!(after["expert_rows_total"].as_u64().unwrap()
            - before["expert_rows_total"].as_u64().unwrap(), 12);
    }

    /// A unit with no previous fence has no GPU gap: pace wait stays put, the rest still moves.
    #[test]
    fn a_first_unit_contributes_no_pace_wait() {
        let _guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let before = expert_stats::snapshot();
        expert_stats::record_unit(100.0, None, 4);
        let after = expert_stats::snapshot();
        assert_eq!(after["expert_ns_total"]["pace_wait"].as_u64().unwrap()
            - before["expert_ns_total"]["pace_wait"].as_u64().unwrap(), 0);
        assert_eq!(after["expert_waves_total"].as_u64().unwrap()
            - before["expert_waves_total"].as_u64().unwrap(), 1);
    }
}
