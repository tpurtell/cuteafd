//! Caller-side residency classification and process-wide buffered miss reads.
use super::{invalid, reactor, MappedRows, MappedTableError, ReadPriority, Result};
use io_uring::{opcode, IoUring};
use std::os::fd::AsRawFd;

pub(super) const MAX_BATCH: usize = 2048;
pub(super) const IOWQ_MAX_WORKERS: u32 = 1024;
pub(super) fn worker_limit() -> Result<u32> {
    match std::env::var("CUTEAFD_DIAG_TABLE_READ_DEPTH") {
        Ok(value) => match value.as_str() {
            "1024" => Ok(1024), "2048" => Ok(2048),
            _ => Err(invalid("CUTEAFD_DIAG_TABLE_READ_DEPTH must be 1024 or 2048")),
        },
        Err(std::env::VarError::NotPresent) => Ok(IOWQ_MAX_WORKERS),
        Err(_) => Err(invalid("invalid CUTEAFD_DIAG_TABLE_READ_DEPTH")),
    }
}
#[cfg(test)]
thread_local! { static SETUP_FAILURE: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(super) fn fail_next_setup(errno: i32) { SETUP_FAILURE.with(|failure| failure.set(Some(errno))); }
fn setup_error() -> Result<()> {
    #[cfg(test)]
    if let Some(errno) = SETUP_FAILURE.with(|failure| failure.take()) {
        return Err(io("injected table reactor failure", std::io::Error::from_raw_os_error(errno)));
    }
    Ok(())
}
#[derive(Default, Debug, Clone, Copy)]
pub(super) struct ReadStats {
    pub resident_copy_rows: u64,
    pub late_major_faults: u64,
    pub unclassified_rows: u64,
    pub unclassified_bytes: u64,
    pub unclassified_batches: u64,
    pub unclassified_batch_ns: u64,
    pub unclassified_batch_max_ns: u64,
    pub unclassified_batch_histogram: [u64; 7],
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
    MappedTableError::Io { context: context.into(), source }
}

fn setup_with(mut setup: impl FnMut(u32) -> std::io::Result<IoUring>) -> Result<IoUring> {
    let mut entries = MAX_BATCH as u32;
    loop {
        match setup(entries) {
            Ok(ring) => return Ok(ring),
            Err(error) if error.raw_os_error() == Some(libc::ENOMEM) && entries > 64 => {
                entries /= 2;
                tracing::warn!(entries, "retrying smaller table reactor ring after ENOMEM");
            },
            Err(error) => return Err(io("io_uring_setup", error)),
        }
    }
}
pub(super) fn new_ring() -> Result<IoUring> {
    let mut ring = setup_with(IoUring::new)?;
    let nop = opcode::Nop::new().build();
    // SAFETY: NOP holds no file or buffer pointers.
    unsafe { ring.submission().push(&nop) }.map_err(|_| invalid("io_uring probe queue full"))?;
    loop {
        match ring.submit_and_wait(1) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(io("io_uring_enter probe", error)),
            Ok(_) => break,
        }
    }
    let result = ring.completion().next().ok_or_else(|| invalid("missing io_uring probe completion"))?.result();
    if result < 0 { return Err(io("io_uring NOP probe", std::io::Error::from_raw_os_error(-result))); }
    let limit = worker_limit()?;
    let mut workers = [limit; 2];
    ring.submitter().register_iowq_max_workers(&mut workers)
        .map_err(|error| io("io_uring IOWQ_MAX_WORKERS", error))?;
    tracing::info!(sq_entries = ring.params().sq_entries(), iowq_max_workers = limit,
        buffer_budget_bytes = 16 * reactor::BUFFER_BYTES, "process-wide table read reactor ready");
    Ok(ring)
}

/// Allocate/probe once on the owner thread before model readiness. No transient
/// per-file probe rings or NOWAIT reads: all cache hits are caller-side copies.
pub(super) fn probe(_part: &MappedRows) -> Result<bool> {
    setup_error()?;
    let owner = reactor::global()?;
    if !owner.ready() { return Err(invalid("table reactor setup unavailable; batches will retry")); }
    Ok(false)
}

pub(super) fn start(reads: Vec<reactor::Read>, width: usize, priority: ReadPriority,
    window: usize, wait: bool) -> Result<reactor::ReadStream> {
    setup_error()?;
    reactor::global()?.start(reads, width, priority, window, wait)
}

pub(super) fn gather(reads: &[Read<'_>], output: &mut [u8], width: usize,
    resident: &[Option<bool>]) -> Result<ReadStats> {
    gather_priority(reads, output, width, resident, ReadPriority::Prefill)
}
pub(super) fn gather_priority(reads: &[Read<'_>], output: &mut [u8], width: usize,
    resident: &[Option<bool>], priority: ReadPriority) -> Result<ReadStats> {
    if width == 0 || width > i32::MAX as usize || reads.len() != resident.len() {
        return Err(invalid("invalid table reactor row width or residency count"));
    }
    let mut slots = std::collections::HashSet::with_capacity(reads.len());
    for read in reads {
        read.part.row_offset(read.row)?;
        if read.slot.checked_add(1).and_then(|slot| slot.checked_mul(width)).is_none_or(|end| end > output.len())
            || !slots.insert(read.slot) { return Err(invalid("invalid or overlapping table reactor destinations")); }
    }
    if reads.is_empty() { return Ok(ReadStats::default()); }
    setup_error()?;
    let mut stats = reactor::global()?.read(reads, output, width, priority)?;
    for (read, resident) in reads.iter().zip(resident) {
        if *resident == Some(true) { stats.hits += 1; stats.hit_bytes += width as u64; }
        else if *resident == Some(false) {
            stats.misses += 1;
            stats.miss_bytes += width as u64;
            stats.device_bytes_estimate += read.part.pages(read.row)?.count() as u64 * read.part.page_bytes as u64;
        } else {
            stats.unclassified_rows += 1;
            stats.unclassified_bytes += width as u64;
        }
    }
    if stats.unclassified_rows > 0 {
        stats.unclassified_batches = 1;
        stats.unclassified_batch_ns = stats.miss_max_ns;
        stats.unclassified_batch_max_ns = stats.miss_max_ns;
        stats.unclassified_batch_histogram[tier(stats.miss_max_ns)] = 1;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn setup_halves_enomem_to_floor_and_does_not_retry_other_errors() {
        let mut sizes = Vec::new();
        assert!(setup_with(|size| { sizes.push(size); Err(std::io::Error::from_raw_os_error(libc::ENOMEM)) }).is_err());
        assert_eq!(sizes, [2048, 1024, 512, 256, 128, 64]);
        sizes.clear();
        assert!(setup_with(|size| { sizes.push(size); Err(std::io::Error::from_raw_os_error(libc::EPERM)) }).is_err());
        assert_eq!(sizes, [2048]);
    }
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
