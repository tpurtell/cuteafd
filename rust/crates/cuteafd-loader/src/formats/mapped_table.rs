//! Page-cache-backed per-token lookup tables: DeepSeek V4.1 engram, Qwen 3.8 PLE
//! n-grams, and any other checkpoint table too large to preload.
//!
//! A table is one or more safetensors tensors (parts) viewed as one row space.
//! Parts are mapped read-only and page-cache backed with `MADV_RANDOM` (no
//! sequential read-ahead across a hash table). Buffered io_uring reads use
//! POSIX_FADV_RANDOM and NOWAIT, retrying misses without O_DIRECT.
//! Rows reach the GPU in three steps, each bounded:
//!
//! - prefetch: as soon as row ids are known (a prefill chunk ahead, a wave's
//!   hashes before its layer), [`TablePrefetcher`] advises the touched pages
//!   (`MADV_WILLNEED`) on a background worker with a bounded queue and page
//!   budget; a full queue drops the advice, never blocks the caller;
//! - gather: rows are copied in request order into caller storage, either on
//!   a [`GatherPool`] (parallel page faults, the caller waits) or on a
//!   [`GatherWorker`] (request-owned tickets and recycled staging slots);
//! - upload: the daemon's pinned staging and async H2D (`shared::mapped_table`).
//!
//! Page cache is the host cache: under memory pressure the kernel evicts table
//! pages and the next gather brings them back. [`TableStats`] distinguishes
//! NOWAIT hits from mmap residency snapshots, not just major faults. [`HotRowCache`] optionally pins the hottest rows by budget.
use cuteafd_core::DType;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Component, Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, Weak};

mod uring;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::SafetensorsTensorMetadata;

#[derive(Debug, thiserror::Error)]
pub enum MappedTableError {
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid mapped table: {0}")]
    Invalid(String),
    #[error("mapped row {row} exceeds table rows {rows}")]
    RowRange { row: u64, rows: u64 },
    #[error("prefetch batch exceeds page budget {0}")]
    PageBudget(usize),
    #[error("{0}")]
    Stopped(&'static str),
}

pub type Result<T, E = MappedTableError> = std::result::Result<T, E>;

fn invalid(message: impl Into<String>) -> MappedTableError {
    MappedTableError::Invalid(message.into())
}

fn os_error(context: impl Into<String>) -> MappedTableError {
    MappedTableError::Io { context: context.into(), source: std::io::Error::last_os_error() }
}

fn page_bytes() -> Result<usize> {
    // SAFETY: sysconf has no memory-safety preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return Err(invalid("cannot determine system page size"));
    }
    Ok(page as usize)
}

/// One immutable checkpoint tensor viewed as fixed-width byte rows.
///
/// The caller must keep the underlying file immutable for the mapping lifetime.
/// Truncating a mapped checkpoint can cause SIGBUS even with a read-only mapping.
pub struct MappedRows {
    base: NonNull<u8>,
    mapped_len: usize,
    data_offset: usize,
    rows: u64,
    row_bytes: usize,
    page_bytes: usize,
    file: File,
    file_offset: u64,
    nowait: AtomicBool,
    fuse: bool,
}

// SAFETY: Only immutable access is exposed; munmap runs after the final owner drops.
unsafe impl Send for MappedRows {}
// SAFETY: as above; concurrent readers only copy out of a read-only mapping.
unsafe impl Sync for MappedRows {}

impl MappedRows {
    /// Map a tensor payload whose offset is absolute within a checkpoint shard.
    ///
    /// # Safety
    /// The file must not be modified or truncated while this mapping exists.
    pub unsafe fn open(path: &Path, offset: u64, rows: u64, row_bytes: usize) -> Result<Self> {
        if rows == 0 || row_bytes == 0 {
            return Err(invalid("mapped tensor dimensions must be nonzero"));
        }
        let payload = rows.checked_mul(row_bytes as u64).ok_or_else(|| invalid("mapped tensor size overflow"))?;
        let end = offset.checked_add(payload).ok_or_else(|| invalid("mapped tensor end overflow"))?;
        let file = File::open(path).map_err(|source| MappedTableError::Io {
            context: format!("opening {}", path.display()),
            source,
        })?;
        let length = file
            .metadata()
            .map_err(|source| MappedTableError::Io { context: format!("reading {}", path.display()), source })?
            .len();
        if end > length {
            return Err(invalid("mapped tensor exceeds checkpoint shard length"));
        }
        let page_bytes = page_bytes()?;
        let aligned = offset / page_bytes as u64 * page_bytes as u64;
        let data_offset = usize::try_from(offset - aligned).map_err(|_| invalid("mapped offset overflow"))?;
        let mapped_len = usize::try_from(end - aligned).map_err(|_| invalid("mapped length overflow"))?;
        if mapped_len > isize::MAX as usize {
            return Err(invalid("mapped tensor exceeds addressable slice size"));
        }
        let file_offset = libc::off_t::try_from(aligned).map_err(|_| invalid("mapped offset overflow"))?;
        uring::random(&file)?;
        // SAFETY: a fresh read-only private mapping of a validated file range.
        let raw = unsafe {
            libc::mmap(std::ptr::null_mut(), mapped_len, libc::PROT_READ, libc::MAP_PRIVATE, file.as_raw_fd(),
                file_offset)
        };
        if raw == libc::MAP_FAILED {
            return Err(os_error("mapping checkpoint tensor"));
        }
        // mmap with a null hint on supported Linux hosts returns a non-null mapping.
        let Some(base) = NonNull::new(raw.cast::<u8>()) else {
            // SAFETY: raw/mapped_len come from the successful mmap above.
            unsafe {
                libc::munmap(raw, mapped_len);
            }
            return Err(invalid("checkpoint mapping returned a null address"));
        };
        let result = Self { base, mapped_len, data_offset, rows, row_bytes, page_bytes, file_offset: offset, nowait: AtomicBool::new(true), fuse: uring::is_fuse(&file), file };
        // Avoid kernel sequential read-ahead across a hundreds-of-GB hash table.
        // SAFETY: advice over exactly the mapping created above.
        if unsafe { libc::madvise(raw, mapped_len, libc::MADV_RANDOM) } != 0 {
            return Err(os_error("setting random checkpoint access"));
        }
        Ok(result)
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    fn row_offset(&self, row: u64) -> Result<usize> {
        if row >= self.rows {
            return Err(MappedTableError::RowRange { row, rows: self.rows });
        }
        // Use checked 64-bit arithmetic before conversion, including for high row IDs.
        let offset = row.checked_mul(self.row_bytes as u64).ok_or_else(|| invalid("row offset overflow"))?;
        self.data_offset
            .checked_add(usize::try_from(offset).map_err(|_| invalid("row offset overflow"))?)
            .ok_or_else(|| invalid("mapped row offset overflow"))
    }

    fn row(&self, row: u64) -> Result<&[u8]> {
        let offset = self.row_offset(row)?;
        // SAFETY: row_offset bounds the row inside the live read-only mapping.
        Ok(unsafe { std::slice::from_raw_parts(self.base.as_ptr().add(offset), self.row_bytes) })
    }

    /// Copy selected rows in input order, retaining duplicates and allocating no staging memory.
    pub fn gather_into(&self, rows: &[u64], output: &mut [u8]) -> Result<()> {
        let size = rows.len().checked_mul(self.row_bytes).ok_or_else(|| invalid("gather size overflow"))?;
        if output.len() != size {
            return Err(invalid("gather output size does not match requested rows"));
        }
        // Validate the whole batch before modifying any output.
        for &row in rows {
            self.row_offset(row)?;
        }
        for (&row, destination) in rows.iter().zip(output.chunks_exact_mut(self.row_bytes)) {
            destination.copy_from_slice(self.row(row)?);
        }
        Ok(())
    }

    fn pages(&self, row: u64) -> Result<std::ops::RangeInclusive<usize>> {
        let start = self.row_offset(row)?;
        Ok(start / self.page_bytes..=(start + self.row_bytes - 1) / self.page_bytes)
    }

    /// Advise coalesced runs of page indices; never includes unrelated gaps.
    fn advise_pages(&self, pages: impl Iterator<Item = usize>) -> Result<()> {
        let mut pages = pages.peekable();
        while let Some(first) = pages.next() {
            let mut last = first;
            while pages.peek().is_some_and(|next| *next == last + 1) {
                last = pages.next().unwrap_or(last);
            }
            let start = first * self.page_bytes;
            let end = ((last + 1) * self.page_bytes).min(self.mapped_len);
            // SAFETY: start..end lies inside this mapping (pages come from row_offset).
            if unsafe { libc::madvise(self.base.as_ptr().add(start).cast(), end - start, libc::MADV_WILLNEED) } != 0 {
                return Err(os_error("prefetching checkpoint rows"));
            }
        }
        Ok(())
    }

    /// Read `len` mapped bytes from `start` into the page cache and map them
    /// (no later faults): async read-ahead in [`WARM_ADVICE_BYTES`] pieces (one
    /// large WILLNEED is capped by the kernel's read-ahead window), then
    /// `MADV_POPULATE_READ`, which waits for them. Without POPULATE_READ
    /// (Linux < 5.14) the read-ahead alone is issued.
    fn warm_range(&self, start: usize, len: usize) -> Result<()> {
        let end = (start + len).min(self.mapped_len);
        let mut at = start;
        while at < end {
            let piece = WARM_ADVICE_BYTES.min(end - at);
            // SAFETY: at..at+piece lies inside this mapping.
            if unsafe { libc::madvise(self.base.as_ptr().add(at).cast(), piece, libc::MADV_WILLNEED) } != 0 {
                return Err(os_error("warming checkpoint rows"));
            }
            at += piece;
        }
        // SAFETY: start..end lies inside this read-only mapping; populating reads only.
        if unsafe { libc::madvise(self.base.as_ptr().add(start).cast(), end - start, libc::MADV_POPULATE_READ) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::EINVAL) {
                return Err(MappedTableError::Io { context: "populating checkpoint rows".into(), source: error });
            }
        }
        Ok(())
    }

    /// Advise only the deduplicated pages touched by this batch, with a hard page budget.
    ///
    /// WILLNEED starts OS read-ahead but does not guarantee residency or completion;
    /// gather_into remains the synchronization point for rows that fault in late.
    /// Call this on an I/O worker, not on a CUDA replay thread.
    pub fn prefetch(&self, rows: &[u64], max_pages: usize) -> Result<usize> {
        let mut pages = BTreeSet::new();
        for &row in rows {
            for page in self.pages(row)? {
                pages.insert(page);
                if pages.len() > max_pages {
                    return Err(MappedTableError::PageBudget(max_pages));
                }
            }
        }
        self.advise_pages(pages.iter().copied())?;
        Ok(pages.len())
    }
}

impl Drop for MappedRows {
    fn drop(&mut self) {
        // SAFETY: base/mapped_len describe the mapping this value owns.
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.mapped_len);
        }
    }
}

/// Read-ahead advice size while warming (larger advice is truncated).
const WARM_ADVICE_BYTES: usize = 128 << 10;
/// One warming window: bounded in-flight reads per warming thread.
const WARM_WINDOW_BYTES: usize = 8 << 20;

/// Where one part of a table lives: an absolute byte offset inside a shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TablePart {
    pub path: PathBuf,
    pub offset: u64,
    pub rows: u64,
}

/// What one row holds: `width` elements of `dtype` in `row_bytes` bytes
/// (packed formats such as NVFP4 store two elements per byte as `U8`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowFormat {
    pub dtype: DType,
    pub width: usize,
    pub row_bytes: usize,
}

impl RowFormat {
    /// The row format of a plain 2-D tensor `[rows, width]` of `dtype`.
    pub fn of(dtype: DType, width: usize) -> Result<Self> {
        let element = crate::dtype_byte_width(&dtype).map_err(|error| invalid(error.to_string()))?;
        Ok(Self { dtype, width, row_bytes: element * width })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableBackend { Uring, Mmap, MincoreRouted }
impl std::fmt::Display for TableBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { Self::Uring => "uring", Self::Mmap => "mmap", Self::MincoreRouted => "mincore-routed" })
    }
}
impl std::str::FromStr for TableBackend {
    type Err = MappedTableError;
    fn from_str(value: &str) -> Result<Self> {
        match value { "uring" => Ok(Self::Uring), "mmap" => Ok(Self::Mmap), "mincore-routed" => Ok(Self::MincoreRouted),
            _ => Err(invalid("table backend must be uring, mmap or mincore-routed")) }
    }
}
static BACKEND: Mutex<Option<TableBackend>> = Mutex::new(None);
impl TableBackend {
    /// CLI override takes precedence over CUTEAFD_TABLE_BACKEND. mmap remains
    /// the default until identical-config hardware qualification passes.
    pub fn set_override(backend: Self) { *BACKEND.lock().unwrap_or_else(|p| p.into_inner()) = Some(backend); }
    fn configured() -> Self {
        if let Some(backend) = *BACKEND.lock().unwrap_or_else(|p| p.into_inner()) { return backend; }
        match std::env::var("CUTEAFD_TABLE_BACKEND") {
            Ok(value) => match value.parse() {
                Ok(backend) => backend,
                Err(error) => { tracing::warn!(%error, "invalid mapped-table override; using mmap"); Self::Mmap }
            },
            Err(_) => Self::Mmap,
        }
    }
}
static TABLES: Mutex<Vec<(String, Weak<TableStats>)>> = Mutex::new(Vec::new());

/// Captured after model readiness, before the HTTP router starts. Request-time
/// reads upgrade weak counter references and load atomics only: no table,
/// gather, interval, or device-counter mutexes and no filesystem I/O.
#[derive(Clone, Default)]
pub struct MappedTableStatsReader {
    tables: Vec<(String, Weak<TableStats>)>,
}
impl MappedTableStatsReader {
    pub fn registered() -> Self {
        Self { tables: TABLES.lock().unwrap_or_else(|p| p.into_inner()).clone() }
    }
    pub fn snapshot(&self) -> Vec<NamedTableStats> {
        self.tables.iter().filter_map(|(name, stats)| {
            let stats = stats.upgrade()?;
            Some(NamedTableStats { name: name.clone(), backend: stats.backend().to_string(),
                accounting: stats.accounting(), cumulative: stats.snapshot(), interval: None,
                host_wide_device_reads: Vec::new(), host_wide_device_read_interval: None })
        }).collect()
    }
}

#[derive(Debug, serde::Serialize)]
pub struct NamedTableStats {
    pub name: String,
    pub backend: String,
    pub accounting: &'static str,
    pub cumulative: TableStatsSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval: Option<TableStatsSnapshot>,
    /// Backing device counters include all host readers, not just this table.
    pub host_wide_device_reads: Vec<DeviceReads>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_wide_device_read_interval: Option<Vec<DeviceReads>>,
}
#[derive(Debug, serde::Serialize)]
pub struct DeviceReads { pub device: String, pub bytes: u64 }
pub fn mapped_table_stats() -> Vec<NamedTableStats> { named_stats(false) }
pub fn mapped_table_stats_with_intervals() -> Vec<NamedTableStats> { named_stats(true) }
fn named_stats(interval: bool) -> Vec<NamedTableStats> {
    let mut registry = TABLES.lock().unwrap_or_else(|p| p.into_inner());
    registry.retain(|(_, stats)| stats.strong_count() > 0);
    registry.iter().filter_map(|(name, stats)| stats.upgrade().map(|stats| {
        let devices: Vec<_> = stats.devices.lock().unwrap_or_else(|p| p.into_inner()).iter().filter_map(|path| {
            let contents = std::fs::read_to_string(path).ok()?;
            let sectors = contents.split_whitespace().nth(2)?.parse::<u64>().ok()?;
            let device = Path::new(path).parent()?.file_name()?.to_string_lossy().into_owned();
            Some(DeviceReads { device, bytes: sectors.saturating_mul(512) })
        }).collect();
        let device_interval = interval.then(|| {
            let mut previous = stats.previous_devices.lock().unwrap_or_else(|p| p.into_inner());
            devices.iter().map(|device| {
                let before = previous.insert(device.device.clone(), device.bytes).unwrap_or(device.bytes);
                DeviceReads { device: device.device.clone(), bytes: device.bytes.saturating_sub(before) }
            }).collect()
        });
        NamedTableStats { name: name.clone(), backend: stats.backend().to_string(),
            accounting: stats.accounting(), cumulative: stats.snapshot(),
            interval: interval.then(|| stats.interval()), host_wide_device_reads: devices,
            host_wide_device_read_interval: device_interval }

    })).collect()
}

#[derive(Default)]
struct ResidencyWorkspace {
    pages: Vec<(usize, usize)>,
    resident: Vec<bool>,
    mincore: Vec<u8>,
}
thread_local! {
    static RESIDENCY: RefCell<ResidencyWorkspace> = RefCell::new(ResidencyWorkspace::default());
}

/// A row space over one or more mapped parts, with its stats.
pub struct MappedTable {
    parts: Vec<MappedRows>,
    /// First global row of each part, then the total.
    starts: Vec<u64>,
    /// Rows of every part but possibly the last (division lookup), else 0.
    uniform: u64,
    format: RowFormat,
    stats: Arc<TableStats>,
}

impl MappedTable {
    /// Map `parts` in row order.
    ///
    /// # Safety
    /// Every referenced file must stay immutable while the table is alive.
    pub unsafe fn open(parts: &[TablePart], format: RowFormat) -> Result<Self> {
        if parts.is_empty() || format.row_bytes == 0 || format.width == 0 {
            return Err(invalid("a mapped table needs rows and at least one part"));
        }
        let mut mapped = Vec::with_capacity(parts.len());
        let mut starts = Vec::with_capacity(parts.len() + 1);
        let mut total = 0u64;
        for part in parts {
            starts.push(total);
            // SAFETY: forwarded from this function's contract.
            mapped.push(unsafe { MappedRows::open(&part.path, part.offset, part.rows, format.row_bytes)? });
            total = total.checked_add(part.rows).ok_or_else(|| invalid("table row count overflow"))?;
        }
        starts.push(total);
        let first = parts[0].rows;
        let uniform = if parts[..parts.len() - 1].iter().all(|p| p.rows == first) && parts.last().is_some_and(|p| p.rows <= first) {
            first
        } else {
            0
        };
        let table = Self { parts: mapped, starts, uniform, format, stats: Arc::new(TableStats::default()) };
        {
            use std::os::unix::fs::MetadataExt;
            let mut devices = table.stats.devices.lock().unwrap_or_else(|p| p.into_inner());
            for part in &table.parts {
                if let Ok(meta) = part.file.metadata() {
                    let path = format!("/sys/dev/block/{}:{}", libc::major(meta.dev()), libc::minor(meta.dev()));
                    if let Ok(path) = std::fs::canonicalize(path) {
                        let path = path.join("stat").to_string_lossy().into_owned();
                        if !devices.contains(&path) { devices.push(path); }
                    }
                }
            }
        }
        table.stats.approximate.store(table.parts.iter().any(|part| part.fuse), Ordering::Relaxed);
        let off = match std::env::var("CUTEAFD_TABLE_ACCOUNTING").ok().as_deref() {
            None | Some("full") => false,
            Some("off") => true,
            Some(value) => { tracing::warn!(value, "invalid table accounting; using full residency"); false }
        };
        table.stats.accounting_off.store(off, Ordering::Relaxed);
        table.select_backend(TableBackend::configured());
        Ok(table)
    }

    /// One tensor as a table (V4.1 engram weights and scales).
    ///
    /// # Safety
    /// The file must stay immutable while the table is alive.
    pub unsafe fn single(path: &Path, offset: u64, rows: u64, format: RowFormat) -> Result<Self> {
        // SAFETY: forwarded from this function's contract.
        unsafe { Self::open(&[TablePart { path: path.to_owned(), offset, rows }], format) }
    }

    /// Checkpoint tensors (shard file relative to `snapshot`, header) as the
    /// parts of one table, in the given order: 2-D, one dtype and width, the
    /// payload exactly `rows x row_bytes`, shard paths inside the snapshot.
    ///
    /// # Safety
    /// Every referenced shard must stay immutable while the table is alive.
    pub unsafe fn from_tensors(snapshot: &Path, tensors: &[(&str, &SafetensorsTensorMetadata)]) -> Result<Self> {
        let (_, first) = tensors.first().ok_or_else(|| invalid("no table tensors"))?;
        if first.shape.len() != 2 {
            return Err(invalid(format!("table tensor {} is not 2-D", first.name)));
        }
        let format = RowFormat::of(first.dtype.clone(), first.shape[1])?;
        let mut parts = Vec::with_capacity(tensors.len());
        for (shard, meta) in tensors {
            if meta.dtype != format.dtype || meta.shape.len() != 2 || meta.shape[1] != format.width {
                return Err(invalid(format!("table tensor {} differs in dtype or width from {}", meta.name, first.name)));
            }
            let rows = meta.shape[0] as u64;
            if Some(meta.byte_length) != rows.checked_mul(format.row_bytes as u64) {
                return Err(invalid(format!("table tensor {} payload is not rows x row bytes", meta.name)));
            }
            if shard.is_empty() || !Path::new(shard).components().all(|part| matches!(part, Component::Normal(_))) {
                return Err(invalid(format!("table tensor {} has an invalid shard path", meta.name)));
            }
            parts.push(TablePart { path: snapshot.join(shard), offset: meta.byte_offset, rows });
        }
        // SAFETY: forwarded from this function's contract.
        unsafe { Self::open(&parts, format) }
    }

    pub fn rows(&self) -> u64 {
        self.starts[self.parts.len()]
    }
    pub fn row_bytes(&self) -> usize {
        self.format.row_bytes
    }
    pub fn format(&self) -> &RowFormat {
        &self.format
    }
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }
    /// Mapped payload bytes (not resident bytes).
    pub fn bytes(&self) -> u64 {
        self.rows() * self.format.row_bytes as u64
    }
    pub fn stats(&self) -> &TableStats {
        &self.stats
    }

    fn locate(&self, row: u64) -> Result<(usize, u64)> {
        if row >= self.rows() {
            return Err(MappedTableError::RowRange { row, rows: self.rows() });
        }
        if self.parts.len() == 1 {
            return Ok((0, row));
        }
        let part = if self.uniform > 0 {
            (row / self.uniform) as usize
        } else {
            self.starts.partition_point(|&start| start <= row) - 1
        };
        Ok((part, row - self.starts[part]))
    }

    /// One row's mapped bytes (faults it in when it is not resident).
    pub fn row(&self, row: u64) -> Result<&[u8]> {
        let (part, local) = self.locate(row)?;
        self.parts[part].row(local)
    }

    /// Copy selected rows in input order, retaining duplicates and allocating
    /// no staging memory; the whole batch is validated before any write.
    pub fn gather_into(&self, rows: &[u64], output: &mut [u8]) -> Result<()> {
        let size = rows.len().checked_mul(self.row_bytes()).ok_or_else(|| invalid("gather size overflow"))?;
        if output.len() != size {
            return Err(invalid("gather output size does not match requested rows"));
        }
        for &row in rows {
            self.locate(row)?;
        }
        let slots: Vec<_> = (0..rows.len()).collect();
        self.gather_slots(rows, &slots, output)?;
        Ok(())
    }

    /// Gather into distinct row slots (for deduplicated engram rows). The
    /// destination layout is shared by both backends; all rows are validated
    /// before any write, and every io_uring batch drains before returning.
    pub fn gather_slots(&self, rows: &[u64], slots: &[usize], output: &mut [u8]) -> Result<()> {
        if rows.len() != slots.len() { return Err(invalid("gather row/slot counts differ")); }
        let width = self.row_bytes();
        let mut unique = BTreeSet::new();
        let reads: Vec<_> = rows.iter().zip(slots).map(|(&row, &slot)| {
            let (part, local) = self.locate(row)?;
            if slot.checked_add(1).and_then(|n| n.checked_mul(width)).is_none_or(|n| n > output.len())
                || !unique.insert(slot) { return Err(invalid("invalid or overlapping gather slots")); }
            Ok(uring::Read { part: &self.parts[part], row: local, slot })
        }).collect::<Result<_>>()?;
        if self.backend() != TableBackend::Mmap {
            match self.uring_reads(rows, &reads, output, false) {
                Ok(stats) => { self.stats.record_reads(stats, false); return Ok(()); }
                Err(error) => self.uring_fallback(&error),
            }
        }
        self.record_residency(rows)?;
        for read in reads {
            output[read.slot * width..(read.slot + 1) * width].copy_from_slice(read.part.row(read.row)?);
        }
        Ok(())
    }

    /// mincore snapshots every requested page before copying any rows. Pages
    /// are deduplicated and consecutive runs coalesced; sparse rows cost one
    /// bounded query per page, never a scan of the hundreds-of-GB table.
    fn residency(&self, rows: &[u64], prefetch: bool) -> Result<Vec<bool>> {
        let started = Instant::now();
        let hits = RESIDENCY.with(|workspace| {
            let mut workspace = workspace.borrow_mut();
            let ResidencyWorkspace { pages, resident, mincore } = &mut *workspace;
            pages.clear();
            for &row in rows {
                let (part, local) = self.locate(row)?;
                pages.extend(self.parts[part].pages(local)?.map(|page| (part, page)));
            }
            pages.sort_unstable();
            pages.dedup();
            resident.resize(pages.len(), false);
            let mut at = 0;
            while at < pages.len() {
                let part = pages[at].0;
                let end = at + pages[at..].partition_point(|&(index, _)| index == part);
                uring::resident(&self.parts[part], &pages[at..end], &mut resident[at..end], mincore)?;
                at = end;
            }
            rows.iter().map(|&row| {
                let (part, local) = self.locate(row)?;
                Ok(self.parts[part].pages(local)?.all(|page| {
                    resident[pages.binary_search(&(part, page)).expect("requested page collected")]
                }))
            }).collect::<Result<Vec<_>>>()
        })?;
        (if prefetch { &self.stats.prefetch_residency_ns } else { &self.stats.residency_ns })
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(hits)
    }

    fn record_residency(&self, rows: &[u64]) -> Result<()> {
        let hits = self.residency(rows, false)?;
        let mut stats = uring::ReadStats::default();
        for (&row, hit) in rows.iter().zip(hits) {
            if hit {
                stats.hits += 1;
                stats.hit_bytes += self.row_bytes() as u64;
            } else {
                let (part, local) = self.locate(row)?;
                stats.misses += 1;
                stats.miss_bytes += self.row_bytes() as u64;
                stats.device_bytes_estimate += self.parts[part].pages(local)?.count() as u64 * self.parts[part].page_bytes as u64;
            }
        }
        self.stats.record_reads(stats, false);
        Ok(())
    }

    fn uring_reads(&self, rows: &[u64], reads: &[uring::Read<'_>], output: &mut [u8], prefetch: bool) -> Result<uring::ReadStats> {
        let routed = self.backend() == TableBackend::MincoreRouted;
        let unsupported: Vec<_> = reads.iter().enumerate().filter_map(|(i, read)|
            (!read.part.nowait.load(Ordering::Relaxed)).then_some(i)).collect();
        let selected: Vec<_> = unsupported.iter().map(|&i| rows[i]).collect();
        let mut hits = vec![None; reads.len()];
        if !selected.is_empty() && (routed || !self.stats.accounting_off.load(Ordering::Relaxed)) {
            for (index, hit) in unsupported.into_iter().zip(self.residency(&selected, prefetch)?) { hits[index] = Some(hit); }
        }
        if routed && hits.contains(&Some(true)) {
            // Unsupported-file resident rows use mmap; NOWAIT-capable files
            // retain exact completion-based classification in this mode too.
            let mut pending = Vec::with_capacity(reads.len());
            let mut pending_hits = Vec::with_capacity(reads.len());
            let before = thread_faults();
            let mut copied = 0;
            for (read, hit) in reads.iter().zip(hits) {
                if hit == Some(true) {
                    let width = self.row_bytes();
                    output[read.slot * width..(read.slot + 1) * width].copy_from_slice(read.part.row(read.row)?);
                    copied += 1;
                } else {
                    pending.push(uring::Read { part: read.part, row: read.row, slot: read.slot });
                    pending_hits.push(hit);
                }
            }
            // Minor faults can merely populate PTEs for cached pages. Major
            // faults during these copies witness late faults, not evicted rows.
            let late = thread_faults_since(before)[1].max(0) as u64;
            let mut stats = if pending.is_empty() { uring::ReadStats::default() }
                else { uring::gather(&pending, output, self.row_bytes(), &pending_hits)? };
            stats.hits += copied;
            stats.hit_bytes += copied * self.row_bytes() as u64;
            stats.resident_copy_rows += copied;
            stats.late_major_faults += late;
            return Ok(stats);
        }
        uring::gather(reads, output, self.row_bytes(), &hits)
    }

    fn uring_fallback(&self, error: &MappedTableError) {
        // A demand/prefetch worker can fail concurrently with another worker.
        // Only the first transition warns; subsequent batches use mmap.
        if self.stats.uring.swap(false, Ordering::Relaxed) {
            tracing::warn!(%error, "mapped table io_uring unavailable; falling back to mmap");
        }
    }

    pub fn backend(&self) -> TableBackend {
        self.stats.backend()
    }

    /// Select an explicit backend, retaining mmap only if ring setup or Read
    /// is unsupported. Files without NOWAIT use bounded mincore accounting.
    /// Called before any workers start.
    pub fn select_backend(&self, requested: TableBackend) {
        self.select_with_probe(requested, || self.parts.iter().try_for_each(|part| {
            part.nowait.store(uring::probe(part)?, Ordering::Relaxed);
            Ok(())
        }));
    }
    fn select_with_probe(&self, requested: TableBackend, probe: impl FnOnce() -> Result<()>) {
        let result = if requested != TableBackend::Mmap { probe() } else { Ok(()) };
        let available = result.is_ok();
        if let Err(error) = result { tracing::warn!(%error, "io_uring unavailable (kernel/filesystem/seccomp); using mmap"); }
        self.stats.routed.store(requested == TableBackend::MincoreRouted && available, Ordering::Relaxed);
        self.stats.uring.store(requested != TableBackend::Mmap && available, Ordering::Relaxed);
        self.stats.nowait.store(self.parts.iter().all(|part| part.nowait.load(Ordering::Relaxed)), Ordering::Relaxed);
        tracing::info!(backend = %self.backend(), requested = %requested,
            max_ring_entries = uring::MAX_BATCH,
            accounting = self.stats.accounting(),
            reason = if !available { "io_uring unavailable" } else if requested == TableBackend::Mmap { "explicit override or qualification default" }
                else if requested == TableBackend::MincoreRouted && !self.stats.nowait.load(Ordering::Relaxed) { "explicit override; resident memcpy and concurrent buffered misses where NOWAIT unsupported" }
                else if !self.stats.nowait.load(Ordering::Relaxed) { "explicit override; buffered reads with mincore where NOWAIT unsupported" } else { "explicit override; NOWAIT supported" },
            locality_note = if self.stats.approximate.load(Ordering::Relaxed) { "FUSE-mounted tables use approximate residency accounting; a local ext4/xfs copy with NOWAIT gives exact hit/miss counts" } else { "NOWAIT capability is probed per file" },
            "mapped table backend");
    }

    /// Register a weak stats reference; endpoint reads neither own nor touch
    /// mapped storage. Entries disappear when their table is released.
    pub fn name_stats(&self, name: impl Into<String>) {
        let mut registry = TABLES.lock().unwrap_or_else(|p| p.into_inner());
        registry.retain(|(_, stats)| stats.strong_count() > 0);
        let weak = Arc::downgrade(&self.stats);
        registry.retain(|(_, stats)| !stats.ptr_eq(&weak));
        registry.push((name.into(), weak));
    }

    /// Reads the whole table into the page cache and maps it, window by
    /// window, until done or `stop` is set; returns the bytes warmed. For
    /// tables that fit in RAM beside everything else: it turns later gathers'
    /// major faults into hits without pinning memory (the kernel may still
    /// evict the pages under pressure). Run it on a background thread.
    /// `bytes_per_second` paces the reads so demand reads (other weights,
    /// this table's own gathers) keep most of the device's bandwidth.
    pub fn warm(&self, stop: &AtomicBool, bytes_per_second: Option<u64>) -> Result<u64> {
        let started = Instant::now();
        let mut warmed = 0u64;
        for part in &self.parts {
            let mut at = part.data_offset / part.page_bytes * part.page_bytes;
            while at < part.mapped_len {
                if stop.load(Ordering::Relaxed) {
                    return Ok(warmed);
                }
                let len = WARM_WINDOW_BYTES.min(part.mapped_len - at);
                part.warm_range(at, len)?;
                warmed += len as u64;
                at += len;
                if let Some(rate) = bytes_per_second.filter(|&rate| rate > 0) {
                    let due = Duration::from_secs_f64(warmed as f64 / rate as f64);
                    if let Some(wait) = due.checked_sub(started.elapsed()) {
                        thread::sleep(wait);
                    }
                }
            }
        }
        Ok(warmed)
    }

    /// Advise the deduplicated pages a batch touches, with a hard page budget
    /// over all parts; returns the pages advised. Call from an I/O worker.
    pub fn prefetch(&self, rows: &[u64], max_pages: usize) -> Result<usize> {
        let mut pages = BTreeSet::new();
        for &row in rows {
            let (part, local) = self.locate(row)?;
            for page in self.parts[part].pages(local)? {
                pages.insert((part, page));
                if pages.len() > max_pages {
                    return Err(MappedTableError::PageBudget(max_pages));
                }
            }
        }
        if self.backend() != TableBackend::Mmap && !rows.is_empty() {
            let mut output = vec![0; rows.len().checked_mul(self.row_bytes()).ok_or_else(|| invalid("prefetch size overflow"))?];
            let reads: Vec<_> = rows.iter().enumerate().map(|(slot, &row)| {
                let (part, local) = self.locate(row)?;
                Ok(uring::Read { part: &self.parts[part], row: local, slot })
            }).collect::<Result<_>>()?;
            match self.uring_reads(rows, &reads, &mut output, true) {
                Ok(stats) => { self.stats.record_reads(stats, true); return Ok(pages.len()); }
                Err(error) => self.uring_fallback(&error),
            }
        }
        let count = pages.len();
        let mut pages = pages.into_iter().peekable();
        while let Some((part, page)) = pages.next() {
            let mut run = vec![page];
            while let Some(&(next_part, next)) = pages.peek() {
                if next_part != part {
                    break;
                }
                run.push(next);
                pages.next();
            }
            self.parts[part].advise_pages(run.into_iter())?;
        }
        Ok(count)
    }
}

/// Page faults and block reads of the calling thread since `before`
/// (minor, major, input blocks); -1 when the OS query failed.
pub fn thread_faults_since(before: Option<[i64; 3]>) -> [i64; 3] {
    before.zip(thread_faults()).map(|(a, b)| [b[0] - a[0], b[1] - a[1], b[2] - a[2]]).unwrap_or([-1; 3])
}

/// The calling thread's (minor faults, major faults, input blocks).
pub fn thread_faults() -> Option<[i64; 3]> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the complete output on success.
    if unsafe { libc::getrusage(libc::RUSAGE_THREAD, usage.as_mut_ptr()) } == 0 {
        // SAFETY: initialized by the successful call above.
        let usage = unsafe { usage.assume_init() };
        Some([usage.ru_minflt, usage.ru_majflt, usage.ru_inblock])
    } else {
        None
    }
}

/// Cumulative counters of one table; updated per batch, never per row.
#[derive(Default)]
pub struct TableStats {
    gathers: AtomicU64,
    rows: AtomicU64,
    bytes: AtomicU64,
    gather_ns: AtomicU64,
    stall_ns: AtomicU64,
    minor_faults: AtomicU64,
    major_faults: AtomicU64,
    input_blocks: AtomicU64,
    prefetch_jobs: AtomicU64,
    prefetch_pages: AtomicU64,
    prefetch_dropped: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    uring: AtomicBool,
    routed: AtomicBool,
    nowait: AtomicBool,
    approximate: AtomicBool,
    accounting_off: AtomicBool,
    previous: Mutex<TableStatsSnapshot>,
    interval_max: [AtomicU64; 6],
    devices: Mutex<Vec<String>>,
    previous_devices: Mutex<HashMap<String, u64>>,
    miss_histogram: [AtomicU64; 7],
    unclassified_rows: AtomicU64,
    unclassified_bytes: AtomicU64,
    unclassified_batches: AtomicU64,
    unclassified_batch_ns: AtomicU64,
    unclassified_batch_max_ns: AtomicU64,
    unclassified_batch_histogram: [AtomicU64; 7],
    prefetch_miss_histogram: [AtomicU64; 7],
    prefetch_unclassified_rows: AtomicU64,
    prefetch_unclassified_bytes: AtomicU64,
    prefetch_unclassified_batches: AtomicU64,
    prefetch_unclassified_batch_ns: AtomicU64,
    prefetch_unclassified_batch_max_ns: AtomicU64,
    prefetch_unclassified_batch_histogram: [AtomicU64; 7],
    page_hits: AtomicU64,
    page_misses: AtomicU64,
    page_hit_bytes: AtomicU64,
    miss_request_bytes: AtomicU64,
    miss_ns: AtomicU64,
    miss_max_ns: AtomicU64,
    nowait_ns: AtomicU64,
    nowait_max_ns: AtomicU64,
    nowait_batches: AtomicU64,
    residency_ns: AtomicU64,
    resident_copy_rows: AtomicU64,
    late_major_faults: AtomicU64,
    decode_stall_ns: AtomicU64,
    decode_steps: AtomicU64,
    device_bytes_estimate: AtomicU64,
    prefetch_device_bytes_estimate: AtomicU64,
    consumer_steps: AtomicU64,
    prefetch_hits: AtomicU64,
    prefetch_misses: AtomicU64,
    prefetch_hit_bytes: AtomicU64,
    prefetch_miss_request_bytes: AtomicU64,
    prefetch_miss_ns: AtomicU64,
    prefetch_miss_max_ns: AtomicU64,
    prefetch_nowait_ns: AtomicU64,
    prefetch_nowait_max_ns: AtomicU64,
    prefetch_nowait_batches: AtomicU64,
    prefetch_residency_ns: AtomicU64,
    prefetch_resident_copy_rows: AtomicU64,
    prefetch_late_major_faults: AtomicU64,

}

/// A point-in-time copy of [`TableStats`]; `since` gives an interval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TableStatsSnapshot {
    pub gathers: u64,
    pub rows: u64,
    pub bytes: u64,
    pub gather_ns: u64,
    /// Time a consumer waited for rows that were not ready.
    pub stall_ns: u64,
    pub minor_faults: u64,
    /// Page-cache misses that read the device (disk or network).
    pub major_faults: u64,
    pub input_blocks: u64,
    pub prefetch_jobs: u64,
    pub prefetch_pages: u64,
    /// Prefetch batches refused by a full queue.
    pub prefetch_dropped: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// Classified misses timed from initial batch submission through completion/retry.
    pub miss_histogram: [u64; 7],
    pub unclassified_rows: u64,
    pub unclassified_bytes: u64,
    #[serde(default)]
    pub unclassified_batches: u64,
    #[serde(default)]
    pub unclassified_batch_ns: u64,
    #[serde(default)]
    pub unclassified_batch_max_ns: u64,
    /// Submission-to-final-reap makespan tiers, in bounded waves, not rows.
    #[serde(default)]
    pub unclassified_batch_histogram: [u64; 7],
    pub prefetch_miss_histogram: [u64; 7],
    pub prefetch_unclassified_rows: u64,
    pub prefetch_unclassified_bytes: u64,
    #[serde(default)]
    pub prefetch_unclassified_batches: u64,
    #[serde(default)]
    pub prefetch_unclassified_batch_ns: u64,
    #[serde(default)]
    pub prefetch_unclassified_batch_max_ns: u64,
    /// Submission-to-final-reap makespan tiers, in bounded waves, not rows.
    #[serde(default)]
    pub prefetch_unclassified_batch_histogram: [u64; 7],
    pub page_hits: u64,
    pub page_misses: u64,
    pub page_hit_bytes: u64,
    pub miss_request_bytes: u64,
    pub miss_ns: u64,
    pub miss_max_ns: u64,
    pub nowait_ns: u64,
    pub nowait_max_ns: u64,
    pub nowait_batches: u64,
    pub residency_ns: u64,
    #[serde(default)]
    pub resident_copy_rows: u64,
    /// Thread-scoped major faults during resident copies; not missed rows.
    #[serde(default)]
    pub late_major_faults: u64,
    pub decode_stall_ns: u64,
    pub decode_steps: u64,
    /// Miss-row page spans (may double-count shared pages), not measured physical I/O.
    pub device_bytes_estimate: u64,
    /// Miss-row page spans (may double-count shared pages), not measured physical I/O.
    pub prefetch_device_bytes_estimate: u64,
    pub consumer_steps: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub prefetch_hit_bytes: u64,
    pub prefetch_miss_request_bytes: u64,
    pub prefetch_miss_ns: u64,
    pub prefetch_miss_max_ns: u64,
    pub prefetch_nowait_ns: u64,
    pub prefetch_nowait_max_ns: u64,
    pub prefetch_nowait_batches: u64,
    pub prefetch_residency_ns: u64,
    #[serde(default)]
    pub prefetch_resident_copy_rows: u64,
    #[serde(default)]
    pub prefetch_late_major_faults: u64,

}

impl TableStats {
    fn backend(&self) -> TableBackend {
        if !self.uring.load(Ordering::Relaxed) { TableBackend::Mmap }
        else if self.routed.load(Ordering::Relaxed) { TableBackend::MincoreRouted }
        else { TableBackend::Uring }
    }
    fn accounting(&self) -> &'static str {
        if self.uring.load(Ordering::Relaxed) && self.nowait.load(Ordering::Relaxed) { "nowait-exact" }
        else if self.backend() == TableBackend::MincoreRouted {
            if self.approximate.load(Ordering::Relaxed) { "fuse: mincore-routed (approx)" } else { "mincore-routed" }
        }
        else if self.uring.load(Ordering::Relaxed) && self.accounting_off.load(Ordering::Relaxed) { "unclassified batch makespan (mincore off)" }
        else if self.approximate.load(Ordering::Relaxed) { "fuse: mincore (approx)" }
        else { "mincore" }
    }

    /// One gather batch: rows and bytes copied, time on the gathering threads
    /// and their fault deltas (negative deltas, from failed queries, are skipped).
    pub fn record_gather(&self, rows: usize, bytes: usize, elapsed: Duration, faults: [i64; 3]) {
        self.gathers.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(rows as u64, Ordering::Relaxed);
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        self.gather_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        self.record_faults(faults);
    }
    pub fn record_faults(&self, faults: [i64; 3]) {
        for (counter, value) in [&self.minor_faults, &self.major_faults, &self.input_blocks].into_iter().zip(faults) {
            if value > 0 {
                counter.fetch_add(value as u64, Ordering::Relaxed);
            }
        }
    }
    pub fn record_stall(&self, elapsed: Duration) {
        self.consumer_steps.fetch_add(1, Ordering::Relaxed);
        self.stall_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }
    pub fn record_decode_stall(&self, elapsed: Duration) {
        self.decode_steps.fetch_add(1, Ordering::Relaxed);
        self.decode_stall_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }
    pub fn record_prefetch(&self, pages: usize) {
        self.prefetch_jobs.fetch_add(1, Ordering::Relaxed);
        self.prefetch_pages.fetch_add(pages as u64, Ordering::Relaxed);
    }
    pub fn record_prefetch_dropped(&self) {
        self.prefetch_dropped.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_cache(&self, hits: usize, misses: usize) {
        self.cache_hits.fetch_add(hits as u64, Ordering::Relaxed);
        self.cache_misses.fetch_add(misses as u64, Ordering::Relaxed);
    }
    fn record_reads(&self, reads: uring::ReadStats, prefetch: bool) {
        let (copied, late) = if prefetch { (&self.prefetch_resident_copy_rows, &self.prefetch_late_major_faults) }
            else { (&self.resident_copy_rows, &self.late_major_faults) };
        copied.fetch_add(reads.resident_copy_rows, Ordering::Relaxed);
        late.fetch_add(reads.late_major_faults, Ordering::Relaxed);
        let (rows, bytes, completions) = if prefetch {
            (&self.prefetch_unclassified_rows, &self.prefetch_unclassified_bytes, &self.prefetch_unclassified_batch_histogram)
        } else { (&self.unclassified_rows, &self.unclassified_bytes, &self.unclassified_batch_histogram) };
        rows.fetch_add(reads.unclassified_rows, Ordering::Relaxed);
        bytes.fetch_add(reads.unclassified_bytes, Ordering::Relaxed);
        let (batches, ns, max) = if prefetch {
            (&self.prefetch_unclassified_batches, &self.prefetch_unclassified_batch_ns, &self.prefetch_unclassified_batch_max_ns)
        } else { (&self.unclassified_batches, &self.unclassified_batch_ns, &self.unclassified_batch_max_ns) };
        batches.fetch_add(reads.unclassified_batches, Ordering::Relaxed);
        ns.fetch_add(reads.unclassified_batch_ns, Ordering::Relaxed);
        max.fetch_max(reads.unclassified_batch_max_ns, Ordering::Relaxed);
        self.interval_max[if prefetch { 5 } else { 4 }].fetch_max(reads.unclassified_batch_max_ns, Ordering::Relaxed);
        for (counter, value) in completions.iter().zip(reads.unclassified_batch_histogram) { counter.fetch_add(value, Ordering::Relaxed); }
        let counters = if prefetch {
            [&self.prefetch_hits, &self.prefetch_misses, &self.prefetch_hit_bytes, &self.prefetch_miss_request_bytes,
             &self.prefetch_miss_ns, &self.prefetch_nowait_ns, &self.prefetch_nowait_batches]
        } else {
            [&self.page_hits, &self.page_misses, &self.page_hit_bytes, &self.miss_request_bytes,
             &self.miss_ns, &self.nowait_ns, &self.nowait_batches]
        };
        let device = if prefetch { &self.prefetch_device_bytes_estimate } else { &self.device_bytes_estimate };
        device.fetch_add(reads.device_bytes_estimate, Ordering::Relaxed);
        for (counter, value) in counters.into_iter().zip([reads.hits, reads.misses, reads.hit_bytes, reads.miss_bytes,
            reads.miss_ns, reads.nowait_ns, reads.nowait_batches]) { counter.fetch_add(value, Ordering::Relaxed); }
        let (histogram, max, nowait_max) = if prefetch {
            (&self.prefetch_miss_histogram, &self.prefetch_miss_max_ns, &self.prefetch_nowait_max_ns)
        } else { (&self.miss_histogram, &self.miss_max_ns, &self.nowait_max_ns) };
        for (counter, value) in histogram.iter().zip(reads.miss_histogram) { counter.fetch_add(value, Ordering::Relaxed); }
        let base = if prefetch { 2 } else { 0 };
        self.interval_max[base].fetch_max(reads.miss_max_ns, Ordering::Relaxed);
        self.interval_max[base + 1].fetch_max(reads.nowait_max_ns, Ordering::Relaxed);
        max.fetch_max(reads.miss_max_ns, Ordering::Relaxed);
        nowait_max.fetch_max(reads.nowait_max_ns, Ordering::Relaxed);
    }
    /// Interval maxima are reset only by the serving stats publisher. Ordinary
    /// readers (console and benches) use the non-destructive cumulative view.
    fn interval(&self) -> TableStatsSnapshot {
        let mut before = self.previous.lock().unwrap_or_else(|p| p.into_inner());
        let now = self.snapshot();
        let mut delta = now.since(&before);
        delta.miss_max_ns = self.interval_max[0].swap(0, Ordering::Relaxed);
        delta.nowait_max_ns = self.interval_max[1].swap(0, Ordering::Relaxed);
        delta.prefetch_miss_max_ns = self.interval_max[2].swap(0, Ordering::Relaxed);
        delta.prefetch_nowait_max_ns = self.interval_max[3].swap(0, Ordering::Relaxed);
        delta.unclassified_batch_max_ns = self.interval_max[4].swap(0, Ordering::Relaxed);
        delta.prefetch_unclassified_batch_max_ns = self.interval_max[5].swap(0, Ordering::Relaxed);
        *before = now;
        delta
    }
    pub fn snapshot(&self) -> TableStatsSnapshot {
        let get = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        TableStatsSnapshot {
            gathers: get(&self.gathers),
            rows: get(&self.rows),
            bytes: get(&self.bytes),
            gather_ns: get(&self.gather_ns),
            stall_ns: get(&self.stall_ns),
            minor_faults: get(&self.minor_faults),
            major_faults: get(&self.major_faults),
            input_blocks: get(&self.input_blocks),
            prefetch_jobs: get(&self.prefetch_jobs),
            prefetch_pages: get(&self.prefetch_pages),
            prefetch_dropped: get(&self.prefetch_dropped),
            cache_hits: get(&self.cache_hits),
            cache_misses: get(&self.cache_misses),
            miss_histogram: std::array::from_fn(|i| get(&self.miss_histogram[i])),
            unclassified_rows: get(&self.unclassified_rows),
            unclassified_bytes: get(&self.unclassified_bytes),
            unclassified_batches: get(&self.unclassified_batches),
            unclassified_batch_ns: get(&self.unclassified_batch_ns),
            unclassified_batch_max_ns: get(&self.unclassified_batch_max_ns),
            unclassified_batch_histogram: std::array::from_fn(|i| get(&self.unclassified_batch_histogram[i])),
            prefetch_miss_histogram: std::array::from_fn(|i| get(&self.prefetch_miss_histogram[i])),
            prefetch_unclassified_rows: get(&self.prefetch_unclassified_rows),
            prefetch_unclassified_bytes: get(&self.prefetch_unclassified_bytes),
            prefetch_unclassified_batches: get(&self.prefetch_unclassified_batches),
            prefetch_unclassified_batch_ns: get(&self.prefetch_unclassified_batch_ns),
            prefetch_unclassified_batch_max_ns: get(&self.prefetch_unclassified_batch_max_ns),
            prefetch_unclassified_batch_histogram: std::array::from_fn(|i| get(&self.prefetch_unclassified_batch_histogram[i])),
            page_hits: get(&self.page_hits),
            page_misses: get(&self.page_misses),
            page_hit_bytes: get(&self.page_hit_bytes),
            miss_request_bytes: get(&self.miss_request_bytes),
            miss_ns: get(&self.miss_ns),
            miss_max_ns: get(&self.miss_max_ns),
            nowait_ns: get(&self.nowait_ns),
            nowait_max_ns: get(&self.nowait_max_ns),
            nowait_batches: get(&self.nowait_batches),
            residency_ns: get(&self.residency_ns),
            resident_copy_rows: get(&self.resident_copy_rows),
            late_major_faults: get(&self.late_major_faults),
            decode_stall_ns: get(&self.decode_stall_ns),
            decode_steps: get(&self.decode_steps),
            device_bytes_estimate: get(&self.device_bytes_estimate),
            prefetch_device_bytes_estimate: get(&self.prefetch_device_bytes_estimate),
            consumer_steps: get(&self.consumer_steps),
            prefetch_hits: get(&self.prefetch_hits),
            prefetch_misses: get(&self.prefetch_misses),
            prefetch_hit_bytes: get(&self.prefetch_hit_bytes),
            prefetch_miss_request_bytes: get(&self.prefetch_miss_request_bytes),
            prefetch_miss_ns: get(&self.prefetch_miss_ns),
            prefetch_miss_max_ns: get(&self.prefetch_miss_max_ns),
            prefetch_nowait_ns: get(&self.prefetch_nowait_ns),
            prefetch_nowait_max_ns: get(&self.prefetch_nowait_max_ns),
            prefetch_nowait_batches: get(&self.prefetch_nowait_batches),
            prefetch_residency_ns: get(&self.prefetch_residency_ns),
            prefetch_resident_copy_rows: get(&self.prefetch_resident_copy_rows),
            prefetch_late_major_faults: get(&self.prefetch_late_major_faults),

        }
    }
}

impl TableStatsSnapshot {
    pub fn since(&self, earlier: &Self) -> Self {
        let d = |a: u64, b: u64| a.saturating_sub(b);
        Self {
            gathers: d(self.gathers, earlier.gathers),
            rows: d(self.rows, earlier.rows),
            bytes: d(self.bytes, earlier.bytes),
            gather_ns: d(self.gather_ns, earlier.gather_ns),
            stall_ns: d(self.stall_ns, earlier.stall_ns),
            minor_faults: d(self.minor_faults, earlier.minor_faults),
            major_faults: d(self.major_faults, earlier.major_faults),
            input_blocks: d(self.input_blocks, earlier.input_blocks),
            prefetch_jobs: d(self.prefetch_jobs, earlier.prefetch_jobs),
            prefetch_pages: d(self.prefetch_pages, earlier.prefetch_pages),
            prefetch_dropped: d(self.prefetch_dropped, earlier.prefetch_dropped),
            cache_hits: d(self.cache_hits, earlier.cache_hits),
            cache_misses: d(self.cache_misses, earlier.cache_misses),
            miss_histogram: std::array::from_fn(|i| d(self.miss_histogram[i], earlier.miss_histogram[i])),
            unclassified_rows: d(self.unclassified_rows, earlier.unclassified_rows),
            unclassified_bytes: d(self.unclassified_bytes, earlier.unclassified_bytes),
            unclassified_batches: d(self.unclassified_batches, earlier.unclassified_batches),
            unclassified_batch_ns: d(self.unclassified_batch_ns, earlier.unclassified_batch_ns),
            unclassified_batch_max_ns: self.unclassified_batch_max_ns, // cumulative maximum, not an interval maximum
            unclassified_batch_histogram: std::array::from_fn(|i| d(self.unclassified_batch_histogram[i], earlier.unclassified_batch_histogram[i])),
            prefetch_miss_histogram: std::array::from_fn(|i| d(self.prefetch_miss_histogram[i], earlier.prefetch_miss_histogram[i])),
            prefetch_unclassified_rows: d(self.prefetch_unclassified_rows, earlier.prefetch_unclassified_rows),
            prefetch_unclassified_bytes: d(self.prefetch_unclassified_bytes, earlier.prefetch_unclassified_bytes),
            prefetch_unclassified_batches: d(self.prefetch_unclassified_batches, earlier.prefetch_unclassified_batches),
            prefetch_unclassified_batch_ns: d(self.prefetch_unclassified_batch_ns, earlier.prefetch_unclassified_batch_ns),
            prefetch_unclassified_batch_max_ns: self.prefetch_unclassified_batch_max_ns, // cumulative maximum, not an interval maximum
            prefetch_unclassified_batch_histogram: std::array::from_fn(|i| d(self.prefetch_unclassified_batch_histogram[i], earlier.prefetch_unclassified_batch_histogram[i])),
            page_hits: d(self.page_hits, earlier.page_hits),
            page_misses: d(self.page_misses, earlier.page_misses),
            page_hit_bytes: d(self.page_hit_bytes, earlier.page_hit_bytes),
            miss_request_bytes: d(self.miss_request_bytes, earlier.miss_request_bytes),
            miss_ns: d(self.miss_ns, earlier.miss_ns),
            miss_max_ns: self.miss_max_ns, // cumulative maximum, not an interval maximum
            nowait_ns: d(self.nowait_ns, earlier.nowait_ns),
            nowait_max_ns: self.nowait_max_ns, // cumulative maximum, not an interval maximum
            nowait_batches: d(self.nowait_batches, earlier.nowait_batches),
            residency_ns: d(self.residency_ns, earlier.residency_ns),
            resident_copy_rows: d(self.resident_copy_rows, earlier.resident_copy_rows),
            late_major_faults: d(self.late_major_faults, earlier.late_major_faults),
            decode_stall_ns: d(self.decode_stall_ns, earlier.decode_stall_ns),
            decode_steps: d(self.decode_steps, earlier.decode_steps),
            device_bytes_estimate: d(self.device_bytes_estimate, earlier.device_bytes_estimate),
            prefetch_device_bytes_estimate: d(self.prefetch_device_bytes_estimate, earlier.prefetch_device_bytes_estimate),
            consumer_steps: d(self.consumer_steps, earlier.consumer_steps),
            prefetch_hits: d(self.prefetch_hits, earlier.prefetch_hits),
            prefetch_misses: d(self.prefetch_misses, earlier.prefetch_misses),
            prefetch_hit_bytes: d(self.prefetch_hit_bytes, earlier.prefetch_hit_bytes),
            prefetch_miss_request_bytes: d(self.prefetch_miss_request_bytes, earlier.prefetch_miss_request_bytes),
            prefetch_miss_ns: d(self.prefetch_miss_ns, earlier.prefetch_miss_ns),
            prefetch_miss_max_ns: self.prefetch_miss_max_ns, // cumulative maximum, not an interval maximum
            prefetch_nowait_ns: d(self.prefetch_nowait_ns, earlier.prefetch_nowait_ns),
            prefetch_nowait_max_ns: self.prefetch_nowait_max_ns, // cumulative maximum, not an interval maximum
            prefetch_nowait_batches: d(self.prefetch_nowait_batches, earlier.prefetch_nowait_batches),
            prefetch_residency_ns: d(self.prefetch_residency_ns, earlier.prefetch_residency_ns),
            prefetch_resident_copy_rows: d(self.prefetch_resident_copy_rows, earlier.prefetch_resident_copy_rows),
            prefetch_late_major_faults: d(self.prefetch_late_major_faults, earlier.prefetch_late_major_faults),

        }
    }
    /// Rows served without a major fault (page cache or hot-row cache), 0..1.
    pub fn resident_rate(&self) -> f64 {
        let rows = self.page_hits + self.page_misses;
        if rows == 0 { return 1.0; }
        self.page_hits as f64 / rows as f64
    }
    pub fn cache_hit_rate(&self) -> Option<f64> {
        let total = self.cache_hits + self.cache_misses;
        (total > 0).then(|| self.cache_hits as f64 / total as f64)
    }
}

/// Anything whose rows can be advised as a unit (one table, or V4.1's paired
/// weight and scale tables).
pub trait AdviseRows: Send + Sync + 'static {
    fn row_count(&self) -> u64;
    /// Pages advised per member table, or None once `cancelled` is set
    /// between tables. Advice already issued to the OS is harmless.
    fn advise(&self, rows: &[u64], max_pages: usize, cancelled: &AtomicBool) -> Result<Option<Vec<usize>>>;
}

impl AdviseRows for MappedTable {
    fn row_count(&self) -> u64 {
        self.rows()
    }
    fn advise(&self, rows: &[u64], max_pages: usize, cancelled: &AtomicBool) -> Result<Option<Vec<usize>>> {
        if cancelled.load(Ordering::Acquire) {
            return Ok(None);
        }
        let pages = self.prefetch(rows, max_pages)?;
        self.stats.record_prefetch(pages);
        Ok(Some(vec![pages]))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TablePrefetchOutcome {
    Cancelled,
    /// Pages advised per member table.
    Advised(Vec<usize>),
}

struct PrefetchJob {
    table: Arc<dyn AdviseRows>,
    rows: Vec<u64>,
    cancelled: Arc<AtomicBool>,
    completion: Option<mpsc::SyncSender<Result<TablePrefetchOutcome>>>,
}

/// Request-owned completion, independent of recycled scheduler slot numbers.
/// Dropping it cancels queued work; advice already issued to the OS is harmless.
pub struct TablePrefetchTicket {
    cancelled: Arc<AtomicBool>,
    completion: mpsc::Receiver<Result<TablePrefetchOutcome>>,
}

impl TablePrefetchTicket {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn wait(&self) -> Result<TablePrefetchOutcome> {
        self.completion.recv().map_err(|_| MappedTableError::Stopped("table prefetch worker stopped"))?
    }
}

impl Drop for TablePrefetchTicket {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// One worker and a bounded queue; submission never waits for disk or queue
/// space. The row cap bounds job memory independently of page deduplication.
pub struct TablePrefetcher {
    sender: Option<mpsc::SyncSender<PrefetchJob>>,
    worker: Option<JoinHandle<()>>,
    max_rows: usize,
}

impl TablePrefetcher {
    pub fn new(name: &str, queue_depth: usize, max_rows: usize, max_pages_per_table: usize) -> Result<Self> {
        if queue_depth == 0 || max_rows == 0 || max_pages_per_table == 0 {
            return Err(invalid("table prefetch capacities must be nonzero"));
        }
        let (sender, receiver) = mpsc::sync_channel::<PrefetchJob>(queue_depth);
        let worker = thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let result = job
                        .table
                        .advise(&job.rows, max_pages_per_table, &job.cancelled)
                        .map(|pages| pages.map_or(TablePrefetchOutcome::Cancelled, TablePrefetchOutcome::Advised));
                    if let Some(completion) = job.completion {
                        let _ = completion.send(result);
                    } else if let Err(error) = result {
                        tracing::debug!(%error, "table prefetch advice failed");
                    }
                }
            })
            .map_err(|source| MappedTableError::Io { context: "starting table prefetch worker".into(), source })?;
        Ok(Self { sender: Some(sender), worker: Some(worker), max_rows })
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    fn job(&self, table: &Arc<dyn AdviseRows>, rows: &[u64]) -> Result<()> {
        if rows.len() > self.max_rows {
            return Err(invalid("table prefetch exceeds row capacity"));
        }
        if let Some(&row) = rows.iter().find(|&&row| row >= table.row_count()) {
            return Err(MappedTableError::RowRange { row, rows: table.row_count() });
        }
        Ok(())
    }

    fn send(&self, job: PrefetchJob) -> Result<bool> {
        match self.sender.as_ref().ok_or(MappedTableError::Stopped("table prefetcher stopped"))?.try_send(job) {
            Ok(()) => Ok(true),
            Err(mpsc::TrySendError::Full(_)) => Ok(false),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(MappedTableError::Stopped("table prefetch worker disconnected")),
        }
    }

    /// None means backpressure: the caller may gather on demand or retry later.
    pub fn try_submit(&self, table: Arc<dyn AdviseRows>, rows: &[u64]) -> Result<Option<TablePrefetchTicket>> {
        self.job(&table, rows)?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let (completion, receiver) = mpsc::sync_channel(1);
        let job = PrefetchJob { table, rows: rows.to_vec(), cancelled: cancelled.clone(), completion: Some(completion) };
        Ok(self.send(job)?.then_some(TablePrefetchTicket { cancelled, completion: receiver }))
    }

    /// Fire-and-forget advice (rows past the row cap are truncated); false
    /// when the queue was full and the advice was dropped.
    pub fn submit_detached(&self, table: Arc<dyn AdviseRows>, rows: &[u64]) -> Result<bool> {
        let rows = &rows[..rows.len().min(self.max_rows)];
        self.job(&table, rows)?;
        let job = PrefetchJob { table, rows: rows.to_vec(), cancelled: Arc::new(AtomicBool::new(false)), completion: None };
        self.send(job)
    }
}

impl Drop for TablePrefetcher {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// What a [`GatherPool`] gather did.
#[derive(Debug, Clone, Copy, Default)]
pub struct GatherReport {
    pub rows: usize,
    pub elapsed: Duration,
    pub major_faults: i64,
    pub cache_hits: usize,
}

/// A dedicated thread pool for synchronous gathers: concurrent page faults
/// keep cold rows from serializing on one thread.
pub struct GatherPool {
    pool: rayon::ThreadPool,
    threads: usize,
}

impl GatherPool {
    pub fn new(name: &str, threads: usize) -> Result<Self> {
        let threads = threads.max(1);
        let prefix = name.to_owned();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |index| format!("{prefix}-{index}"))
            .build()
            .map_err(|error| invalid(format!("starting the {name} gather pool: {error}")))?;
        Ok(Self { pool, threads })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Copy `rows` of `table` into `output` in order (duplicates kept), hits
    /// from `cache` first; the batch is validated before any write.
    pub fn gather(&self, table: &MappedTable, rows: &[u64], output: &mut [u8], cache: Option<&HotRowCache>)
        -> Result<GatherReport> {
        use rayon::prelude::*;
        let started = Instant::now();
        let row_bytes = table.row_bytes();
        if Some(output.len()) != rows.len().checked_mul(row_bytes) {
            return Err(invalid("gather output size does not match requested rows"));
        }
        for &row in rows {
            table.locate(row)?;
        }
        if let Some(cache) = cache {
            if cache.row_bytes() != row_bytes {
                return Err(invalid("hot-row cache row size differs from its table"));
            }
        }
        let (hits, misses) = match cache {
            Some(cache) => cache.lookup(rows, output),
            None => (0, (0..rows.len()).collect()),
        };
        let faults = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
        // Several chunks per thread balance uneven fault latency.
        let chunk = misses.len().div_ceil(self.threads * 4).max(1);
        let base = output.as_mut_ptr() as usize;
        let output_len = output.len();
        if table.backend() != TableBackend::Mmap {
            let selected: Vec<_> = misses.iter().map(|&index| rows[index]).collect();
            table.gather_slots(&selected, &misses, output)?;
        } else {
        let selected: Vec<_> = misses.iter().map(|&index| rows[index]).collect();
        table.record_residency(&selected)?;
        self.pool.install(|| {
            misses.par_chunks(chunk).try_for_each(|indices| -> Result<()> {
                let before = thread_faults();
                for &index in indices {
                    let source = table.row(rows[index])?;
                    debug_assert!((index + 1) * row_bytes <= output_len);
                    // SAFETY: `misses` holds distinct indices, so every task
                    // writes disjoint row slots of `output`, which outlives the
                    // scoped `install`; bounds were validated above.
                    let destination = unsafe {
                        std::slice::from_raw_parts_mut((base as *mut u8).add(index * row_bytes), row_bytes)
                    };
                    destination.copy_from_slice(source);
                }
                for (total, value) in faults.iter().zip(thread_faults_since(before)) {
                    if value > 0 {
                        total.fetch_add(value as u64, Ordering::Relaxed);
                    }
                }
                Ok(())
            })
        })?;
        }
        if let Some(cache) = cache {
            cache.insert(misses.iter().map(|&index| (rows[index], &output[index * row_bytes..(index + 1) * row_bytes])));
            table.stats.record_cache(hits, misses.len());
        }
        let elapsed = started.elapsed();
        let faults = faults.map(|value| value.load(Ordering::Relaxed) as i64);
        table.stats.record_gather(rows.len(), output.len(), elapsed, faults);
        Ok(GatherReport { rows: rows.len(), elapsed, major_faults: faults[1], cache_hits: hits })
    }
}

/// A gather running on a [`GatherPool`]; dropping it waits for completion
/// (the output storage must outlive the copy).
pub struct PendingGather {
    done: mpsc::Receiver<Result<GatherReport>>,
    finished: bool,
}

impl PendingGather {
    /// Blocks until the rows are in the output storage.
    pub fn wait(mut self) -> Result<GatherReport> {
        self.finished = true;
        self.done.recv().map_err(|_| MappedTableError::Stopped("gather pool task stopped"))?
    }
}

impl Drop for PendingGather {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.done.recv();
        }
    }
}

impl GatherPool {
    /// [`Self::gather`] on the pool, returning at once: the caller overlaps
    /// other work (enqueueing GPU work that precedes the rows' use) and waits
    /// on the returned handle.
    ///
    /// # Safety
    /// `output..output + len` must stay valid, and must be neither read nor
    /// written by anything else, until the handle is waited on or dropped.
    pub unsafe fn spawn_gather(self: &Arc<Self>, table: Arc<MappedTable>, rows: Vec<u64>, output: *mut u8, len: usize,
        cache: Option<Arc<HotRowCache>>) -> PendingGather {
        let (sender, done) = mpsc::sync_channel(1);
        let pool = Arc::clone(self);
        let output = output as usize;
        self.pool.spawn(move || {
            // SAFETY: valid and exclusive until the handle observes completion (contract above).
            let out = unsafe { std::slice::from_raw_parts_mut(output as *mut u8, len) };
            let _ = sender.send(pool.gather(&table, &rows, out, cache.as_deref()));
        });
        PendingGather { done, finished: false }
    }
}

/// A budgeted LRU of hot rows that survives page-cache eviction (useful when
/// a table is larger than RAM or shares it with other work). Off by default:
/// the page cache already caches rows while memory allows.
pub struct HotRowCache {
    inner: Mutex<Lru>,
    row_bytes: usize,
}

const NIL: u32 = u32::MAX;

struct Lru {
    slots: HashMap<u64, u32>,
    keys: Vec<u64>,
    prev: Vec<u32>,
    next: Vec<u32>,
    head: u32,
    tail: u32,
    data: Vec<u8>,
    capacity: usize,
}

impl Lru {
    fn unlink(&mut self, slot: u32) {
        let (p, n) = (self.prev[slot as usize], self.next[slot as usize]);
        if p == NIL { self.head = n } else { self.next[p as usize] = n }
        if n == NIL { self.tail = p } else { self.prev[n as usize] = p }
    }
    fn push_front(&mut self, slot: u32) {
        self.prev[slot as usize] = NIL;
        self.next[slot as usize] = self.head;
        if self.head != NIL {
            self.prev[self.head as usize] = slot;
        }
        self.head = slot;
        if self.tail == NIL {
            self.tail = slot;
        }
    }
}

impl HotRowCache {
    /// None when `budget_bytes` holds no row.
    pub fn new(row_bytes: usize, budget_bytes: usize) -> Option<Self> {
        let capacity = (budget_bytes / row_bytes.max(1)).min(NIL as usize - 1);
        (capacity > 0 && row_bytes > 0).then(|| Self {
            inner: Mutex::new(Lru {
                slots: HashMap::with_capacity(capacity),
                keys: Vec::with_capacity(capacity),
                prev: Vec::with_capacity(capacity),
                next: Vec::with_capacity(capacity),
                head: NIL,
                tail: NIL,
                data: Vec::new(),
                capacity,
            }),
            row_bytes,
        })
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map_or(0, |lru| lru.slots.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy cached rows into their output slots; returns hits and the
    /// indices (into `rows`) still to read.
    fn lookup(&self, rows: &[u64], output: &mut [u8]) -> (usize, Vec<usize>) {
        let Ok(mut lru) = self.inner.lock() else { return (0, (0..rows.len()).collect()) };
        let mut misses = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            match lru.slots.get(row).copied() {
                Some(slot) => {
                    let at = slot as usize * self.row_bytes;
                    output[index * self.row_bytes..(index + 1) * self.row_bytes]
                        .copy_from_slice(&lru.data[at..at + self.row_bytes]);
                    lru.unlink(slot);
                    lru.push_front(slot);
                }
                None => misses.push(index),
            }
        }
        (rows.len() - misses.len(), misses)
    }

    fn insert<'r>(&self, rows: impl Iterator<Item = (u64, &'r [u8])>) {
        let Ok(mut lru) = self.inner.lock() else { return };
        for (row, bytes) in rows {
            if lru.slots.contains_key(&row) {
                continue;
            }
            let slot = if lru.keys.len() < lru.capacity {
                let slot = lru.keys.len() as u32;
                lru.keys.push(row);
                lru.prev.push(NIL);
                lru.next.push(NIL);
                lru.data.extend_from_slice(bytes);
                slot
            } else {
                let slot = lru.tail;
                lru.unlink(slot);
                let old = lru.keys[slot as usize];
                lru.slots.remove(&old);
                lru.keys[slot as usize] = row;
                let at = slot as usize * self.row_bytes;
                lru.data[at..at + self.row_bytes].copy_from_slice(bytes);
                slot
            };
            lru.slots.insert(row, slot);
            lru.push_front(slot);
        }
    }
}

/// Recorded only when the job was submitted with timing (trace enabled).
#[derive(Debug, Clone, Copy)]
pub struct GatherTiming {
    pub queued: Duration,
    pub gather: Duration,
    pub completed: Instant,
    /// Fault/I/O counts are -1 when the OS counter query was unavailable.
    pub minor_faults: i64,
    pub major_faults: i64,
    pub input_blocks: i64,
}

/// Holds a staging slot until its consumer (a synchronous GPU upload) is done
/// with it. Dropping a ready result returns the slot to the pool without blocking.
pub struct GatherLease<S: Send + 'static, J> {
    slot: Option<S>,
    recycler: mpsc::SyncSender<S>,
    job: J,
    timing: Option<GatherTiming>,
}

impl<S: Send + 'static, J> GatherLease<S, J> {
    pub fn slot(&self) -> Option<&S> {
        self.slot.as_ref()
    }
    pub fn job(&self) -> &J {
        &self.job
    }
    pub fn timing(&self) -> Option<&GatherTiming> {
        self.timing.as_ref()
    }
}

impl<S: Send + 'static, J> Drop for GatherLease<S, J> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            // Full/disconnected means the pool is no longer usable; dropping is safe.
            let _ = self.recycler.try_send(slot);
        }
    }
}

/// A gather job failed in the worker, or the worker itself stopped.
#[derive(Debug)]
pub enum GatherFailure<E> {
    Job(E),
    Worker(MappedTableError),
}

pub enum GatherPoll<L> {
    Pending,
    Cancelled,
    Ready(L),
}

type Completion<S, J, E> = mpsc::Receiver<std::result::Result<Option<GatherLease<S, J>>, E>>;

/// Independent of scheduler slot reuse; drop cancels queued/in-progress work.
/// An in-progress OS page fault cannot be interrupted, but its result is discarded.
pub struct GatherTicket<S: Send + 'static, J, E> {
    cancelled: Arc<AtomicBool>,
    completion: Option<Completion<S, J, E>>,
    consumed: bool,
}

impl<S: Send + 'static, J, E> GatherTicket<S, J, E> {
    pub fn cancel(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.completion.take();
    }
    /// Nonblocking polling for a CUDA or scheduler thread; consume at most once.
    pub fn poll(&mut self) -> std::result::Result<GatherPoll<GatherLease<S, J>>, GatherFailure<E>> {
        if self.consumed {
            return Err(GatherFailure::Worker(MappedTableError::Stopped("gather completion already consumed")));
        }
        if self.cancelled.load(Ordering::Acquire) {
            self.consumed = true;
            return Ok(GatherPoll::Cancelled);
        }
        let completion = self
            .completion
            .as_ref()
            .ok_or(GatherFailure::Worker(MappedTableError::Stopped("gather completion receiver is closed")))?;
        match completion.try_recv() {
            Ok(result) => {
                self.consumed = true;
                self.completion.take();
                match result.map_err(GatherFailure::Job)? {
                    Some(lease) => Ok(GatherPoll::Ready(lease)),
                    None => Ok(GatherPoll::Cancelled),
                }
            }
            Err(mpsc::TryRecvError::Empty) => Ok(GatherPoll::Pending),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.consumed = true;
                self.completion.take();
                Err(GatherFailure::Worker(MappedTableError::Stopped("gather worker stopped before completion")))
            }
        }
    }
}

impl<S: Send + 'static, J, E> Drop for GatherTicket<S, J, E> {
    fn drop(&mut self) {
        self.cancel();
    }
}

struct WorkerJob<S: Send + 'static, J, E> {
    submitted: Option<Instant>,
    lease: GatherLease<S, J>,
    cancelled: Arc<AtomicBool>,
    completion: mpsc::SyncSender<std::result::Result<Option<GatherLease<S, J>>, E>>,
}

/// Bounded background gathers into a fixed pool of reusable staging slots:
/// the pool bounds queued, in-flight and completed-but-unconsumed storage, and
/// submission never waits (None is backpressure).
pub struct GatherWorker<S: Send + 'static, J: Send + 'static, E: Send + 'static> {
    sender: Option<mpsc::SyncSender<WorkerJob<S, J, E>>>,
    worker: Option<JoinHandle<()>>,
    pool: Mutex<mpsc::Receiver<S>>,
    recycler: mpsc::SyncSender<S>,
    shutdown: Arc<AtomicBool>,
}

impl<S: Send + 'static, J: Send + 'static, E: Send + 'static> GatherWorker<S, J, E> {
    /// `run` fills a slot for a job on the worker thread.
    pub fn new(name: &str, slots: Vec<S>, mut run: impl FnMut(&mut S, &J) -> std::result::Result<(), E> + Send + 'static)
        -> Result<Self> {
        if slots.is_empty() {
            return Err(invalid("a gather worker needs at least one staging slot"));
        }
        let count = slots.len();
        let (recycler, pool) = mpsc::sync_channel(count);
        for slot in slots {
            recycler.try_send(slot).map_err(|_| invalid("initializing the gather staging pool failed"))?;
        }
        let (sender, jobs) = mpsc::sync_channel::<WorkerJob<S, J, E>>(count);
        let shutdown = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&shutdown);
        let worker = thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                while let Ok(mut job) = jobs.recv() {
                    if stopping.load(Ordering::Acquire) || job.cancelled.load(Ordering::Acquire) {
                        let _ = job.completion.send(Ok(None));
                        continue;
                    }
                    let before = job.submitted.and_then(|_| thread_faults());
                    let started = job.submitted.map(|_| Instant::now());
                    let lease = &mut job.lease;
                    let result = match lease.slot.as_mut() {
                        Some(slot) => run(slot, &lease.job),
                        None => {
                            let _ = job.completion.send(Ok(None));
                            continue;
                        }
                    };
                    if let (Some(submitted), Some(started)) = (job.submitted, started) {
                        let completed = Instant::now();
                        let faults = thread_faults_since(before);
                        job.lease.timing = Some(GatherTiming {
                            queued: started.duration_since(submitted),
                            gather: completed.duration_since(started),
                            completed,
                            minor_faults: faults[0],
                            major_faults: faults[1],
                            input_blocks: faults[2],
                        });
                    }
                    let result = if stopping.load(Ordering::Acquire) || job.cancelled.load(Ordering::Acquire) {
                        Ok(None)
                    } else {
                        result.map(|()| Some(job.lease))
                    };
                    let _ = job.completion.send(result);
                }
            })
            .map_err(|source| MappedTableError::Io { context: format!("starting {name}"), source })?;
        Ok(Self { sender: Some(sender), worker: Some(worker), pool: Mutex::new(pool), recycler, shutdown })
    }

    /// Submit early, when row ids become available. None is bounded
    /// backpressure; this performs no mapped reads or blocking wait.
    /// `timed` records [`GatherTiming`] on the lease.
    pub fn try_submit(&self, job: J, timed: bool) -> Result<Option<GatherTicket<S, J, E>>> {
        let slot = match self.pool.try_lock() {
            Ok(pool) => match pool.try_recv() {
                Ok(slot) => slot,
                Err(mpsc::TryRecvError::Empty) => return Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => return Err(MappedTableError::Stopped("gather staging pool stopped")),
            },
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MappedTableError::Stopped("gather staging pool poisoned")),
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        let (completion, receive) = mpsc::sync_channel(1);
        let job = WorkerJob {
            submitted: timed.then(Instant::now),
            lease: GatherLease { slot: Some(slot), recycler: self.recycler.clone(), job, timing: None },
            cancelled: Arc::clone(&cancelled),
            completion,
        };
        match self.sender.as_ref().ok_or(MappedTableError::Stopped("gather worker stopped"))?.try_send(job) {
            Ok(()) => Ok(Some(GatherTicket { cancelled, completion: Some(receive), consumed: false })),
            Err(mpsc::TrySendError::Full(_)) => Ok(None),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(MappedTableError::Stopped("gather worker stopped")),
        }
    }
}

impl<S: Send + 'static, J: Send + 'static, E: Send + 'static> Drop for GatherWorker<S, J, E> {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                tracing::error!("gather worker panicked during shutdown");
            }
        }
    }
}

#[cfg(test)]
mod tests;
