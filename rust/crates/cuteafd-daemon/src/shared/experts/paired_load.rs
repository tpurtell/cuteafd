//! Shared, bounded pinned projection reads for the two RTX weight loaders.
use crate::shared::memory::{device::{Device, DeviceOwner}, HostAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer};
use cuteafd_loader::{OfficialV41Catalog, V41Exl3TensorSlice};

pub(crate) struct Projection {
    pub name: String,
    pub offset: usize,
    pub bytes: usize,
    pub slices: [V41Exl3TensorSlice; 2],
}

impl Projection {
    pub fn new(catalog: &OfficialV41Catalog, name: String, offset: usize,
        slices: [V41Exl3TensorSlice; 2]) -> Result<Self> {
        let bytes = usize::try_from(catalog.tensor(&name)?.metadata.byte_length)?;
        V41Exl3TensorSlice::validate_pair(slices, bytes)?;
        Ok(Self { name, offset, bytes, slices })
    }

    pub fn read(&self, catalog: &OfficialV41Catalog, destination: &mut [u8]) -> Result<usize> {
        let mut read = 0;
        if let Some(prefix) = self.name.strip_suffix(".trellis") {
            let mut mcg = [0; 4];
            read += catalog.read_projection_once(&format!("{prefix}.mcg"), &mut mcg)?;
            ensure!(u32::from_le_bytes(mcg) == 0xcbac1fed,
                "unsupported EXL3 MCG multiplier for {prefix}");
        }
        read += catalog.read_projection_once(&self.name, destination)?;
        if self.name.ends_with(".suh") || self.name.ends_with(".svh") {
            ensure!(destination.chunks_exact(2)
                .all(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7c00 != 0x7c00),
                "non-finite EXL3 rotation in {}", self.name);
        }
        Ok(read)
    }

    pub fn upload(&self, device: Device<'_>, rank: usize, host: CuteafdHostBuffer,
        destination: CuteafdDeviceBuffer, stream: &LoadStream<'_>) -> Result<()> {
        let slice = self.slices[rank];
        let offset = self.offset + slice.column_start_bytes;
        // SAFETY: validated slices fit this retained pinned bank; the stream is
        // drained on the owning GPU before either bank or destination is freed.
        device.run(|| unsafe {
            let mut source = host;
            source.ptr = source.ptr.cast::<u8>().add(offset).cast();
            source.bytes -= offset;
            device.library.copy_host_buffer_h2d_2d_async(destination,
                slice.selected_row_bytes, source, slice.source_row_bytes,
                slice.selected_row_bytes, slice.rows, stream.raw)
        })
    }
}

fn admit_banks(lanes: usize, bytes: usize, admitted: usize) -> Result<usize> {
    ensure!(lanes > 0 && bytes > 0, "empty paired staging bank");
    let allocated = bytes.checked_mul(lanes).and_then(|n| n.checked_mul(2))
        .context("paired pinned staging overflow")?;
    ensure!(allocated <= admitted, "paired double-buffer staging {allocated} exceeds admitted {admitted}");
    Ok(allocated)
}

pub(crate) fn banks<'a>(device: Device<'a>, lanes: usize, bytes: usize,
    admitted: usize) -> Result<Vec<Vec<HostAllocation<'a>>>> {
    admit_banks(lanes, bytes, admitted)?;
    device.run(|| (0..2).map(|_| (0..lanes)
        .map(|_| HostAllocation::new(device.library, bytes)).collect()).collect())
}

pub(crate) fn streams<'a>(devices: [Device<'a>; 2]) -> Result<[DeviceOwner<'a, LoadStream<'a>>; 2]> {
    let make = |device: Device<'a>| device.own(|| Ok(LoadStream {
        library: device.library, raw: device.library.cuda_stream_create()?,
    }));
    Ok([make(devices[0])?, make(devices[1])?])
}

pub(crate) struct Fences<'a> {
    events: [[crate::shared::memory::device::Event<'a>; 2]; 2],
    queued: [bool; 2],
}
impl<'a> Fences<'a> {
    pub fn new(devices: [Device<'a>; 2]) -> Result<Self> {
        use crate::shared::memory::device::Event;
        Ok(Self { events: [[Event::new(devices[0])?, Event::new(devices[1])?],
            [Event::new(devices[0])?, Event::new(devices[1])?]], queued: [false; 2] })
    }
    pub fn reuse(&self, bank: usize) -> Result<()> {
        if !self.queued[bank] { return Ok(()); }
        let results = self.events[bank].each_ref().map(|event| event.device.run(|| {
            // SAFETY: both events mark the last upload consuming this bank.
            unsafe { event.device.library.cuda_event_synchronize(event.raw) }
        }));
        let [left, right] = results;
        left.and(right)
    }
    pub fn record(&mut self, bank: usize, streams: &[DeviceOwner<'_, LoadStream<'_>>; 2]) -> Result<()> {
        for (event, stream) in self.events[bank].iter().zip(streams) {
            event.device.run(|| {
                // SAFETY: stream and event share the same owning device.
                unsafe { event.device.library.cuda_event_record(event.raw, stream.raw) }
            })?;
        }
        self.queued[bank] = true;
        Ok(())
    }
}

pub(crate) fn trace_overlap(streams: &[DeviceOwner<'_, LoadStream<'_>>; 2], phase: &str,
    layer: usize, first: usize) -> Result<()> {
    if std::env::var("CUTEAFD_TP2_LOAD_TRACE").as_deref() != Ok("1") { return Ok(()); }
    let busy = streams.each_ref().map(|stream| stream.device.run(|| {
        // SAFETY: diagnostic nonblocking query of the retained rank copy stream.
        unsafe { Ok(!stream.library.cuda_stream_query(stream.raw)?) }
    }));
    let [left, right] = busy;
    tracing::info!(layer, first, phase, gpu0_busy = left?, gpu1_busy = right?,
        "TP2 loader stream overlap probe");
    Ok(())
}

pub(crate) fn drain(streams: &[DeviceOwner<'_, LoadStream<'_>>; 2]) -> Result<()> {
    // Attempt both drains even if one fails; owners also drain during unwind.
    let results = streams.each_ref().map(|stream| stream.device.run(|| {
        // SAFETY: stream owner retains its stream until both uploads finish.
        unsafe { stream.library.cuda_stream_synchronize(stream.raw) }
    }));
    let [left, right] = results;
    left.and(right)
}

fn join_readers(readers: Vec<std::thread::ScopedJoinHandle<'_, Result<usize>>>) -> Result<usize> {
    let results: Vec<Result<usize>> = readers.into_iter().map(|reader| reader.join()
        .map_err(|_| anyhow::anyhow!("paired projection reader panicked"))
        .and_then(|result| result)).collect();
    results.into_iter().try_fold(0, |total, result| Ok(total + result?))
}

fn projection_slots<'a>(plan: &'a [Projection], mut bytes: &'a mut [u8])
    -> Result<Vec<(&'a Projection, &'a mut [u8])>> {
    let mut cursor = 0;
    let mut slots = Vec::with_capacity(plan.len());
    for job in plan {
        ensure!(job.offset >= cursor, "overlapping paired projection staging");
        let gap = job.offset - cursor;
        ensure!(gap <= bytes.len() && job.bytes <= bytes.len() - gap,
            "paired projection staging exceeds bank");
        let (_, remainder) = bytes.split_at_mut(gap);
        let (destination, remainder) = remainder.split_at_mut(job.bytes);
        slots.push((job, destination));
        bytes = remainder;
        cursor = job.offset + job.bytes;
    }
    Ok(slots)
}

pub(crate) fn read_jobs<T: Send>(jobs: Vec<T>, workers: usize,
    read: impl Fn(T) -> Result<usize> + Sync) -> Result<usize> {
    use std::sync::Mutex;
    let workers = workers.min(jobs.len());
    ensure!(workers > 0, "empty paired projection reader queue");
    let queue = Mutex::new(jobs);
    std::thread::scope(|scope| {
        let readers = (0..workers).map(|_| scope.spawn(|| {
            let mut bytes = 0;
            loop {
                // Release the queue lock before I/O. Each job owns an exclusive
                // bank slice; failure/panic still joins every worker before reuse.
                let job = queue.lock().map_err(|_| anyhow::anyhow!("poisoned projection queue"))?.pop();
                let Some(job) = job else { return Ok(bytes); };
                bytes += read(job)?;
            }
        })).collect();
        join_readers(readers)
    })
}

pub(crate) fn read_group(catalog: &OfficialV41Catalog, plans: &[Vec<Projection>],
    hosts: &mut [HostAllocation<'_>]) -> Result<usize> {
    let mut jobs = Vec::new();
    for (plan, host) in plans.iter().zip(hosts) {
        jobs.extend(projection_slots(plan, host.bytes_mut())?);
    }
    // Match the old two-rank storage queue depth without duplicate reads or
    // extra staging. Large trellis jobs go first; rotations fill tail capacity.
    jobs.sort_unstable_by_key(|(job, _)| job.bytes);
    read_jobs(jobs, super::layer::EXPERT_READ_LANES * 2,
        |(job, destination)| job.read(catalog, destination))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn double_banks_fit_existing_pair_admission() {
        // Native: two ranks x 16 half projections = two banks x 8 full.
        assert_eq!(admit_banks(8, 2048, 2 * 16 * 1024).unwrap(), 32768);
        // EXL3: both ranks already admitted two banks x 16 half projections.
        assert_eq!(admit_banks(16, 2048, 2 * 2 * 16 * 1024).unwrap(), 65536);
        assert!(admit_banks(8, 2048, 32767).is_err());
        assert!(admit_banks(16, usize::MAX, usize::MAX).is_err());
        assert!(admit_banks(0, 1, 0).is_err());
    }
    #[test]
    fn projection_queue_restores_two_rank_depth_and_reads_each_job_once() -> Result<()> {
        use std::sync::{Barrier, atomic::{AtomicUsize, Ordering}};
        let barrier = Barrier::new(32);
        let counts: Vec<_> = (0..144).map(|_| AtomicUsize::new(0)).collect();
        let started = AtomicUsize::new(0);
        let bytes = read_jobs((0..144).collect(), 32, |job| {
            if started.fetch_add(1, Ordering::SeqCst) < 32 { barrier.wait(); }
            counts[job].fetch_add(1, Ordering::SeqCst);
            Ok(job + 1)
        })?;
        assert_eq!(bytes, (1usize..=144).sum::<usize>());
        assert!(counts.iter().all(|count| count.load(Ordering::SeqCst) == 1));
        Ok(())
    }

    #[test]
    fn projection_slots_are_disjoint_and_preserve_padding() -> Result<()> {
        let slice = V41Exl3TensorSlice { rows: 1, source_row_bytes: 4,
            column_start_bytes: 0, selected_row_bytes: 4 };
        let plans = [Projection { name: "a".into(), offset: 0, bytes: 4, slices: [slice; 2] },
            Projection { name: "b".into(), offset: 8, bytes: 4, slices: [slice; 2] }];
        let mut bank = [7; 16];
        for (job, destination) in projection_slots(&plans, &mut bank)? {
            destination.fill(if job.offset == 0 { 1 } else { 2 });
        }
        assert_eq!(bank, [1,1,1,1,7,7,7,7,2,2,2,2,7,7,7,7]);
        assert!(projection_slots(&plans, &mut [0; 11]).is_err());
        let overlap = [Projection { name: "a".into(), offset: 0, bytes: 8, slices: [slice; 2] },
            Projection { name: "b".into(), offset: 4, bytes: 4, slices: [slice; 2] }];
        assert!(projection_slots(&overlap, &mut bank).is_err());
        Ok(())
    }

    #[test]
    fn reader_error_or_panic_joins_every_partner() {
        use std::sync::{Barrier, atomic::{AtomicUsize, Ordering}};
        for panic in [false, true] {
            let joined = AtomicUsize::new(0);
            let barrier = Barrier::new(2);
            std::thread::scope(|scope| {
                let left = scope.spawn(|| {
                    barrier.wait();
                    if panic { panic!("injected projection reader panic"); }
                    Err(anyhow::anyhow!("injected projection read error"))
                });
                let right = scope.spawn(|| {
                    barrier.wait();
                    joined.fetch_add(1, Ordering::SeqCst);
                    Ok(7)
                });
                assert!(join_readers(vec![left, right]).is_err());
                assert_eq!(joined.load(Ordering::SeqCst), 1);
            });
        }
    }
}
