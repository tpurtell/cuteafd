//! Buffered io_uring reads. Borrowed file/buffer pointers never escape a drained batch.
use super::{invalid, MappedRows, MappedTableError, Result};
use io_uring::{opcode, types, IoUring};
use std::cell::RefCell;
use std::os::fd::AsRawFd;
use std::sync::atomic::Ordering;
use std::time::Instant;

const MAX_BATCH: usize = 32768;
thread_local! { static RING: RefCell<Option<IoUring>> = const { RefCell::new(None) }; }

#[derive(Default, Debug, Clone, Copy)]
pub(super) struct ReadStats {
    pub resident_copy_rows: u64,
    pub late_major_faults: u64,
    pub unclassified_rows: u64,
    pub unclassified_bytes: u64,
    pub completion_histogram: [u64; 7],
    pub hits: u64,
    pub misses: u64,
    pub hit_bytes: u64,
    pub miss_bytes: u64,
    pub device_bytes_estimate: u64,
    pub miss_histogram: [u64; 7],
    pub miss_ns: u64,
    pub miss_max_ns: u64,
    pub nowait_ns: u64,
    pub nowait_max_ns: u64,
    pub nowait_batches: u64,
}

pub(super) fn tier(ns: u64) -> usize {
    [5_000, 10_000, 50_000, 200_000, 1_000_000, 10_000_000].partition_point(|&limit| ns >= limit)
}

pub(super) struct Read<'a> {
    pub part: &'a MappedRows,
    pub row: u64,
    pub slot: usize,
}

fn io(context: &str, source: std::io::Error) -> MappedTableError {
    MappedTableError::Io {
        context: context.into(),
        source,
    }
}

/// The guard drains every queued SQE, including on errors and unwinding. Ring
/// teardown alone is insufficient: kernel cancellation can outlive close(2).
struct Batch<'a> {
    ring: &'a mut IoUring,
    outstanding: usize,
}
impl Batch<'_> {
    fn push(&mut self, entry: io_uring::squeue::Entry) -> Result<()> {
        // SAFETY: callers validate disjoint destinations and live files before
        // queueing. This guard waits for all completions before either can drop.
        unsafe { self.ring.submission().push(&entry) }
            .map_err(|_| invalid("io_uring queue capacity exceeded"))?;
        self.outstanding += 1;
        Ok(())
    }
    fn next(&mut self) -> Result<io_uring::cqueue::Entry> {
        loop {
            if let Some(entry) = self.ring.completion().next() {
                self.outstanding -= 1;
                return Ok(entry);
            }
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(io("waiting for mapped-table reads", error)),
            }
        }
    }
    fn submit(&mut self) -> Result<()> {
        loop {
            match self.ring.submit() {
                Ok(_) => return Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(io("submitting mapped-table reads", error)),
            }
        }
    }
}
impl Drop for Batch<'_> {
    fn drop(&mut self) {
        while self.outstanding > 0 {
            // Remaining queued entries may not yet have been submitted. Draining
            // also handles partial submit and operation errors without freeing
            // storage that a kernel worker could still write.
            if let Err(error) = self.next() {
                tracing::error!(%error, "cannot drain borrowed io_uring buffers; refusing unsafe teardown");
                std::process::abort();
            }
        }
    }
}

fn entry(
    read: &Read<'_>,
    base: *mut u8,
    width: usize,
    nowait: bool,
    asynchronous: bool,
    done: usize,
    index: usize,
) -> Result<io_uring::squeue::Entry> {
    let offset = read.part.file_offset + read.row * width as u64 + done as u64;
    let at = read.slot * width + done;
    // SAFETY: gather validates every slot and row before queueing; the exclusive
    // output borrow and Batch guard outlive every kernel read into this slice.
    let ptr = unsafe { base.add(at) };
    Ok(opcode::Read::new(
        types::Fd(read.part.file.as_raw_fd()),
        ptr,
        u32::try_from(width - done)
            .map_err(|_| invalid("io_uring row exceeds signed CQE length"))?,
    )
    .offset(offset)
    .rw_flags(if nowait { libc::RWF_NOWAIT as _ } else { 0 })
    .build()
    .flags(if asynchronous {
        io_uring::squeue::Flags::ASYNC
    } else {
        io_uring::squeue::Flags::empty()
    })
    .user_data(index as u64))
}

fn new_ring(entries: u32) -> Result<IoUring> {
    let mut ring = IoUring::new(entries).map_err(|error| io("io_uring_setup", error))?;
    // Test enter/seccomp in each worker before exposing borrowed destinations.
    // Closing a failed NOP-only ring cannot leave a buffer write in flight.
    let nop = opcode::Nop::new().build();
    // SAFETY: NOP has no borrowed file or memory pointers.
    unsafe { ring.submission().push(&nop) }.map_err(|_| invalid("io_uring probe queue full"))?;
    loop {
        match ring.submit_and_wait(1) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io("io_uring_enter probe", error)),
            Ok(_) => break,
        }
    }
    let result = ring
        .completion()
        .next()
        .ok_or_else(|| invalid("missing io_uring probe completion"))?
        .result();
    if result < 0 {
        return Err(io(
            "io_uring NOP probe",
            std::io::Error::from_raw_os_error(-result),
        ));
    }
    Ok(ring)
}

/// Checks both ring setup (including seccomp) and this file's buffered NOWAIT
/// support. EOPNOTSUPP/EINVAL must not be silently counted as cache misses.
pub(super) fn probe(part: &MappedRows) -> Result<bool> {
    let mut ring = new_ring(2)?;
    let mut byte = [0];
    let sqe = opcode::Read::new(types::Fd(part.file.as_raw_fd()), byte.as_mut_ptr(), 1)
        .offset(part.file_offset)
        .rw_flags(libc::RWF_NOWAIT as _)
        .build();
    let mut batch = Batch {
        ring: &mut ring,
        outstanding: 0,
    };
    batch.push(sqe)?;
    batch.submit()?;
    let result = batch.next()?.result();
    if result == -libc::EOPNOTSUPP || result == -libc::EINVAL {
        // A filesystem lacking NOWAIT can still support ordinary buffered Read.
        let sqe = opcode::Read::new(types::Fd(part.file.as_raw_fd()), byte.as_mut_ptr(), 1)
            .offset(part.file_offset)
            .build();
        batch.push(sqe)?;
        batch.submit()?;
        let result = batch.next()?.result();
        if result != 1 {
            return Err(io(
                "buffered io_uring probe",
                if result < 0 {
                    std::io::Error::from_raw_os_error(-result)
                } else {
                    std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "empty immutable table")
                },
            ));
        }
        return Ok(false);
    }
    if result < 0 && result != -libc::EAGAIN {
        return Err(io(
            "buffered RWF_NOWAIT probe",
            std::io::Error::from_raw_os_error(-result),
        ));
    }
    Ok(true)
}

/// All NOWAIT rows are submitted together, then all misses together. Large
/// prefill batches use bounded 32K-row waves (decode fits in a single wave).
pub(super) fn gather(
    reads: &[Read<'_>],
    output: &mut [u8],
    width: usize,
    resident: &[Option<bool>],
) -> Result<ReadStats> {
    if width > i32::MAX as usize {
        return Err(invalid("io_uring row exceeds signed CQE length"));
    }
    if reads.len() != resident.len() {
        return Err(invalid("residency row count mismatch"));
    }
    let mut slots = std::collections::HashSet::with_capacity(reads.len());
    for read in reads {
        read.part.row_offset(read.row)?;
        if read
            .slot
            .checked_add(1)
            .and_then(|slot| slot.checked_mul(width))
            .is_none_or(|end| end > output.len())
            || !slots.insert(read.slot)
        {
            return Err(invalid("invalid or overlapping io_uring row destinations"));
        }
    }
    let base = output.as_mut_ptr();
    RING.with(|cached| {
        let mut cached = cached.borrow_mut();
        let entries = reads.len().clamp(2, MAX_BATCH).next_power_of_two() as u32;
        if cached
            .as_ref()
            .is_none_or(|ring| ring.params().sq_entries() < entries)
        {
            *cached = Some(new_ring(entries)?);
        }
        let ring = cached.as_mut().expect("ring initialized");
        let mut stats = ReadStats::default();
        for (reads, resident) in reads.chunks(MAX_BATCH).zip(resident.chunks(MAX_BATCH)) {
            let nowait: Vec<_> = reads
                .iter()
                .map(|read| read.part.nowait.load(Ordering::Relaxed))
                .collect();
            let mut nowait_left = nowait.iter().filter(|&&enabled| enabled).count();
            let mut batch = Batch {
                ring,
                outstanding: 0,
            };
            let mut retry = Vec::with_capacity(reads.len());
            let mut done = vec![0usize; reads.len()];
            let mut missed = vec![false; reads.len()];
            let mut error = None;
            for (index, read) in reads.iter().enumerate() {
                // Known misses go straight to kernel workers; warm buffered hits
                // may finish inline. Neither mode creates a managed row cache.
                batch.push(entry(
                    read,
                    base,
                    width,
                    nowait[index],
                    !nowait[index] && resident[index] == Some(false),
                    0,
                    index,
                )?)?;
            }
            let started = Instant::now();
            batch.submit()?;
            for _ in 0..reads.len() {
                let cqe = batch.next()?;
                let index = cqe.user_data() as usize;
                let result = cqe.result();
                if nowait[index] {
                    nowait_left -= 1;
                    if nowait_left == 0 {
                        let elapsed = started.elapsed().as_nanos() as u64;
                        stats.nowait_ns += elapsed;
                        stats.nowait_max_ns = stats.nowait_max_ns.max(elapsed);
                        stats.nowait_batches += 1;
                    }
                }
                if result < 0 && !(nowait[index] && result == -libc::EAGAIN) {
                    error = Some(io(
                        "initial mapped-table read",
                        std::io::Error::from_raw_os_error(-result),
                    ));
                    continue;
                }
                let hit = if nowait[index] {
                    Some(result == width as i32)
                } else {
                    resident[index]
                };
                if hit == Some(true) {
                    stats.hits += 1;
                    stats.hit_bytes += width as u64;
                } else if hit == Some(false) {
                    missed[index] = true;
                    stats.misses += 1;
                    stats.miss_bytes += width as u64;
                    let read = &reads[index];
                    let start = read.part.file_offset + read.row * width as u64;
                    let page = read.part.page_bytes as u64;
                    stats.device_bytes_estimate +=
                        ((start + width as u64 - 1) / page - start / page + 1) * page;
                }
                if result == width as i32 {
                    if hit == Some(false) {
                        record_miss(&mut stats, started);
                    } else if hit.is_none() {
                        record_unclassified(&mut stats, started, width);
                    }
                } else if !nowait[index] && result == 0 {
                    error = Some(io(
                        "blocking mapped-table read",
                        std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "short immutable table",
                        ),
                    ));
                } else {
                    // Restart a short NOWAIT row; continue a short buffered row.
                    done[index] = if nowait[index] { 0 } else { result as usize };
                    retry.push(index);
                }
            }
            if let Some(error) = error {
                return Err(error);
            }
            for &index in &retry {
                batch.push(entry(
                    &reads[index],
                    base,
                    width,
                    false,
                    true,
                    done[index],
                    index,
                )?)?;
            }
            if !retry.is_empty() {
                batch.submit()?;
            }
            while batch.outstanding > 0 {
                let cqe = batch.next()?;
                let index = cqe.user_data() as usize;
                let result = cqe.result();
                if result <= 0 {
                    error = Some(io(
                        "blocking mapped-table read",
                        if result == 0 {
                            std::io::Error::new(
                                std::io::ErrorKind::UnexpectedEof,
                                "short immutable table",
                            )
                        } else {
                            std::io::Error::from_raw_os_error(-result)
                        },
                    ));
                    continue;
                }
                done[index] += result as usize;
                if done[index] < width {
                    batch.push(entry(
                        &reads[index],
                        base,
                        width,
                        false,
                        true,
                        done[index],
                        index,
                    )?)?;
                    batch.submit()?;
                } else if missed[index] {
                    record_miss(&mut stats, started);
                } else if !nowait[index] && resident[index].is_none() {
                    record_unclassified(&mut stats, started, width);
                }
            }
            if let Some(error) = error {
                return Err(error);
            }
        }
        Ok(stats)
    })
}

fn record_unclassified(stats: &mut ReadStats, started: Instant, width: usize) {
    stats.unclassified_rows += 1;
    stats.unclassified_bytes += width as u64;
    stats.completion_histogram[tier(started.elapsed().as_nanos() as u64)] += 1;
}

fn record_miss(stats: &mut ReadStats, started: Instant) {
    let elapsed = started.elapsed().as_nanos() as u64;
    stats.miss_ns += elapsed;
    stats.miss_max_ns = stats.miss_max_ns.max(elapsed);
    stats.miss_histogram[tier(elapsed)] += 1;
}

/// mincore is a residency snapshot, not proof that a later read avoids I/O.
/// Query only requested pages, coalesced into bounded runs (not a table scan).
pub(super) fn resident(
    part: &MappedRows,
    pages: &[(usize, usize)],
    result: &mut [bool],
    residency: &mut Vec<u8>,
) -> Result<()> {
    let mut at = 0;
    while at < pages.len() {
        let mut end = at + 1;
        while end < pages.len() {
            let span = pages[end].1 - pages[at].1 + 1;
            // At most 16 MiB on 4K-page hosts, with at least 1/8 selected pages:
            // dense rows amortize syscalls without scanning sparse table gaps.
            if span > 4096 || span > (end - at + 1) * 8 {
                break;
            }
            end += 1;
        }
        let first = pages[at].1;
        let span = pages[end - 1].1 - first + 1;
        residency.resize(span, 0);
        let start = first * part.page_bytes;
        let len = ((pages[end - 1].1 + 1) * part.page_bytes).min(part.mapped_len) - start;
        // SAFETY: requested pages and the bounded span between them belong to
        // this live immutable mapping. The reused vector has one byte per page.
        if unsafe {
            libc::mincore(
                part.base.as_ptr().add(start).cast(),
                len,
                residency.as_mut_ptr(),
            )
        } != 0
        {
            return Err(super::os_error("querying mapped-table residency"));
        }
        for index in at..end {
            result[index] = residency[pages[index].1 - first] & 1 != 0;
        }
        at = end;
    }
    Ok(())
}

pub(super) fn is_fuse(file: &std::fs::File) -> bool {
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: fstatfs initializes the sized struct on success; fd stays live.
    if unsafe { libc::fstatfs(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
        return false;
    }
    // SAFETY: initialized by the successful call above.
    unsafe { stat.assume_init() }.f_type == 0x6573_5546
}

#[cfg(test)]
pub(super) fn dontneed(part: &MappedRows) -> i32 {
    // SAFETY: fadvise only attempts eviction of this live, immutable fd's
    // validated mapped payload range, not other files or the global cache.
    unsafe {
        libc::posix_fadvise(
            part.file.as_raw_fd(),
            part.file_offset as i64,
            (part.rows * part.row_bytes as u64) as i64,
            libc::POSIX_FADV_DONTNEED,
        )
    }
}

pub(super) fn random(file: &std::fs::File) -> Result<()> {
    // SAFETY: fadvise only changes the kernel's advice for this live file.
    let result = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
    if result != 0 {
        return Err(io(
            "setting buffered random table access",
            std::io::Error::from_raw_os_error(result),
        ));
    }
    Ok(())
}
