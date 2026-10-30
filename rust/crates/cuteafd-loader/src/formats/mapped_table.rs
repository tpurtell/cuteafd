//! Page-cache-backed per-token lookup tables: DeepSeek V4.1 engram, Qwen 3.8 PLE
//! n-grams, and any other checkpoint table too large to preload.
//!
//! A table is one or more safetensors tensors (parts) viewed as one row space.
//! Parts are mapped read-only and page-cache backed with `MADV_RANDOM` (no
//! sequential read-ahead across a hash table). Buffered io_uring reads use
//! POSIX_FADV_RANDOM; a process-wide reactor reads only mincore misses without O_DIRECT.
//! Rows reach the GPU in three steps, each bounded:
//!
//! - prefetch: as soon as row ids are known (a prefill chunk ahead, a wave's
//!   hashes before its layer), [`TablePrefetcher`] reads the touched rows with
//!   the hybrid on a background worker; unavailable-ring fallback uses
//!   `MADV_WILLNEED`. Queue, rows and pages are bounded; a full queue drops work;
//! - gather: rows are copied in request order into caller storage, either on
//!   a [`GatherPool`] (parallel page faults, the caller waits) or on a
//!   [`GatherWorker`] (request-owned tickets and recycled staging slots);
//! - upload: the daemon's pinned staging and async H2D (`shared::mapped_table`).
//!
//! Page cache is the host cache: under memory pressure the kernel evicts table
//! pages and the next gather brings them back. [`TableStats`] distinguishes
//! caller-side residency snapshots from completed miss reads, not just major faults.
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
mod reactor;
pub mod row_cache;

/// Explicit priority from the decode/prefill pipeline, never inferred from row count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReadPriority { DecodeFirst, Decode, PrefillFirst, #[default] Prefill }
impl ReadPriority {
    pub(crate) fn is_decode(self) -> bool { matches!(self, Self::DecodeFirst | Self::Decode) }
    pub(crate) fn first(self) -> Self {
        if self.is_decode() { Self::DecodeFirst } else { Self::PrefillFirst }
    }
}
/// Conservative process-wide admission: 16 MiB miss buffers plus descriptors,
/// SQ/CQ storage and owner scratch. Charged once per process, not per table.
pub const TABLE_REACTOR_WORKSPACE_BYTES: usize = 18 << 20;

/// Setup retries retain the process-wide reservation after initialization.
pub fn table_reactor_started() -> bool { reactor::started() }

/// One process-wide reservation regardless of table groups or device owners.
pub fn table_reactor_workspace_bytes(tables: usize) -> usize {
    if tables > 0 { TABLE_REACTOR_WORKSPACE_BYTES } else { 0 }
}
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

/// Bytes per positioned payload read (scale preload, page-cache warm).
pub(crate) const PAYLOAD_CHUNK_BYTES: usize = 16 << 20;

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

/// Fixed pinned staging shared by runtime allocation and offline admission.
/// Each slot has cache-line-separated ready and consumed counter words.
#[derive(Debug, Clone, Copy)]
pub struct MappedStaging {
    prefill_bytes: usize,
    prefill_slots: usize,
    decode_bytes: usize,
    pub workspace_bytes: usize,
    pub total_bytes: usize,
}
impl MappedStaging {

    pub fn new(prefill_rows: usize, prefill_slots: usize, decode_rows: usize, row_bytes: usize) -> Result<Self> {
        if prefill_rows == 0 || decode_rows == 0 || row_bytes == 0 || !(1..=4).contains(&prefill_slots) {
            return Err(invalid("invalid mapped-table staging geometry"));
        }
        let overflow = || invalid("mapped-table staging overflow");
        let prefill_bytes = prefill_rows.checked_mul(row_bytes).ok_or_else(overflow)?;
        let decode_bytes = decode_rows.checked_mul(row_bytes).ok_or_else(overflow)?;
        let prefill_slots = crate::shared::token_table::TableShape::new(prefill_rows, prefill_slots, 1)
            .map_err(|_| invalid("invalid token-table shape"))?.slots().map_err(|_| overflow())?;
        // Bound row IDs, validation/read descriptors and backend scratch for
        // the independent prefill/decode workers; no GPU handshake counters.
        let workspace_bytes = prefill_rows.checked_add(decode_rows).and_then(|n| n.checked_mul(1024))
            .and_then(|n| n.checked_add(2 << 20)).ok_or_else(overflow)?;
        let total_bytes = prefill_bytes.checked_mul(prefill_slots)
            .and_then(|n| decode_bytes.checked_mul(2).and_then(|d| n.checked_add(d))).and_then(|n| n.checked_add(workspace_bytes)).ok_or_else(overflow)?;
        Ok(Self { prefill_bytes, prefill_slots, decode_bytes, workspace_bytes, total_bytes })
    }

    pub fn slot_bytes(self) -> impl Iterator<Item = usize> {
        std::iter::repeat_n(self.prefill_bytes, self.prefill_slots).chain(std::iter::repeat_n(self.decode_bytes, 2))
    }
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
    file: Arc<File>,
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
        let result = Self { base, mapped_len, data_offset, rows, row_bytes, page_bytes, file_offset: offset, nowait: AtomicBool::new(true), fuse: uring::is_fuse(&file), file: Arc::new(file) };
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

    fn payload_bytes(&self) -> u64 {
        self.rows * self.row_bytes as u64
    }

    /// Positioned read at a payload-relative offset through this part's own
    /// descriptor; the mapping is not touched.
    fn read_payload(&self, offset: u64, buffer: &mut [u8]) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.file.read_exact_at(buffer, self.file_offset + offset)
    }

    /// Release the page cache of a payload range already copied elsewhere.
    /// Advice only: a page still mapped somewhere stays, and a refusal leaves
    /// ordinary reclaimable cache.
    fn drop_cached(&self, offset: u64, length: usize) {
        // SAFETY: fadvise only asks the kernel to drop clean cached pages of
        // this live descriptor's range; no memory is accessed.
        unsafe {
            libc::posix_fadvise(self.file.as_raw_fd(), (self.file_offset + offset) as libc::off_t,
                length as libc::off_t, libc::POSIX_FADV_DONTNEED);
        }
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

/// A preloaded table's bytes in private anonymous memory. The kernel supplies
/// zero pages on first touch, so a preload pays no zeroing pass before its
/// reads overwrite every byte, and an allocation failure is an error rather
/// than an abort.
struct ResidentBytes {
    base: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is owned and written only through `&mut self` before a
// table publishes it; afterwards every access is a read.
unsafe impl Send for ResidentBytes {}
// SAFETY: as above; shared access only reads.
unsafe impl Sync for ResidentBytes {}

impl ResidentBytes {
    fn new(len: usize) -> Result<Self> {
        if len == 0 || len > isize::MAX as usize {
            return Err(invalid("invalid table preload size"));
        }
        // SAFETY: a fresh private anonymous mapping; no existing memory is touched.
        let raw = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        if raw == libc::MAP_FAILED {
            return Err(os_error("allocating table preload"));
        }
        let Some(base) = NonNull::new(raw.cast::<u8>()) else {
            // SAFETY: raw/len come from the successful mmap above.
            unsafe {
                libc::munmap(raw, len);
            }
            return Err(invalid("table preload mapping returned a null address"));
        };
        Ok(Self { base, len })
    }
}

impl std::ops::Deref for ResidentBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: the live, zero-initialized mapping this value owns.
        unsafe { std::slice::from_raw_parts(self.base.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for ResidentBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in deref; `&mut self` makes this the only access.
        unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr(), self.len) }
    }
}

impl Drop for ResidentBytes {
    fn drop(&mut self) {
        // SAFETY: base/len describe the mapping this value owns.
        unsafe {
            libc::munmap(self.base.as_ptr().cast(), self.len);
        }
    }
}

/// One positioned payload read: `length` bytes at `offset` within a part,
/// into `destination` (a preload) or a reader's scratch buffer (a warm).
struct PayloadChunk<'a> {
    part: usize,
    offset: u64,
    length: usize,
    destination: Option<&'a mut [u8]>,
}

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

/// What one [`MappedTable::warm`] pass read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WarmReport {
    /// Payload bytes read; below [`MappedTable::warm_bytes`] after a stop.
    pub bytes: u64,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableBackend { Mmap, MincoreRouted, Resident }
impl std::fmt::Display for TableBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { Self::Mmap => "mmap-fallback", Self::MincoreRouted => "mincore-routed", Self::Resident => "resident" })
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
    resident: Option<ResidentBytes>,
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
        let table = Self { parts: mapped, resident: None, starts, uniform, format, stats: Arc::new(TableStats::default()) };
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
    pub(crate) fn shared_stats(&self) -> Arc<TableStats> {
        Arc::clone(&self.stats)
    }

    /// Hold every table in host memory, all or none: read each payload once on
    /// `readers` threads ([`Self::read_chunks`]) into the table's own anonymous
    /// memory, releasing the page cache behind every chunk so the host never
    /// holds the bytes twice, then publish all of them together. A failure
    /// leaves every table mapped.
    pub fn preload_all(tables: &mut [&mut Self], readers: usize) -> Result<()> {
        Self::preload_chunked(tables, readers, PAYLOAD_CHUNK_BYTES)
    }

    fn preload_chunked(tables: &mut [&mut Self], readers: usize, chunk: usize) -> Result<()> {
        let stop = AtomicBool::new(false);
        let buffers = tables.iter().map(|table| {
            let size = usize::try_from(table.bytes()).map_err(|_| invalid("preload size overflow"))?;
            let mut bytes = ResidentBytes::new(size)?;
            table.read_chunks(table.payload_chunks(chunk, Some(&mut bytes[..])), readers, &stop, &|_: u64| (),
                true, "preload")?;
            // Pages shared by two chunks' edges stay cached until every chunk is read.
            for part in &table.parts {
                part.drop_cached(0, part.payload_bytes() as usize);
            }
            Ok(bytes)
        }).collect::<Result<Vec<_>>>()?;
        for (table, bytes) in tables.iter_mut().zip(buffers) {
            table.resident = Some(bytes);
            table.stats.resident.store(true, Ordering::Relaxed);
            table.stats.devices.lock().unwrap_or_else(|p| p.into_inner()).clear();
        }
        Ok(())
    }

    /// Bytes a [`Self::warm`] pass reads: the mapped payload, or zero once resident.
    pub fn warm_bytes(&self) -> u64 {
        if self.resident.is_some() { 0 } else { self.bytes() }
    }

    /// Advisory page-cache warm: read every part's payload once in
    /// [`PAYLOAD_CHUNK_BYTES`] reads on `readers` threads
    /// ([`Self::read_chunks`]), checking `stop` between chunks; `progress`
    /// receives each finished chunk's bytes on the reader threads. The pages
    /// stay ordinary page cache the kernel may evict. The mapping, its
    /// `MADV_RANDOM` advice and every gather path are untouched. A resident
    /// table returns at once with zero bytes.
    pub fn warm(&self, readers: usize, stop: &AtomicBool, progress: impl Fn(u64) + Sync) -> Result<WarmReport> {
        let started = Instant::now();
        if self.resident.is_some() {
            return Ok(WarmReport::default());
        }
        let bytes = self.read_chunks(self.payload_chunks(PAYLOAD_CHUNK_BYTES, None), readers, stop, &progress,
            false, "warm")?;
        Ok(WarmReport { bytes, elapsed: started.elapsed() })
    }

    /// Every part's payload as `chunk`-byte reads, in order. With `buffer`
    /// (the table's bytes, parts back to back), each read lands in its own
    /// disjoint slice of it.
    fn payload_chunks<'a>(&self, chunk: usize, buffer: Option<&'a mut [u8]>) -> Vec<PayloadChunk<'a>> {
        let mut chunks = Vec::new();
        let mut rest = buffer;
        for (part, rows) in self.parts.iter().enumerate() {
            // A mapped payload fits in usize: MappedRows::open bounds it by isize::MAX.
            let size = rows.payload_bytes() as usize;
            match rest.take().map(|bytes| bytes.split_at_mut(size)) {
                Some((mine, tail)) => {
                    rest = Some(tail);
                    chunks.extend(mine.chunks_mut(chunk).enumerate().map(|(index, destination)| PayloadChunk {
                        part, offset: (index * chunk) as u64, length: destination.len(), destination: Some(destination) }));
                }
                None => chunks.extend((0..size).step_by(chunk).map(|offset| PayloadChunk {
                    part, offset: offset as u64, length: (size - offset).min(chunk), destination: None })),
            }
        }
        chunks
    }

    /// Read `chunks` once on `readers` scoped threads named `table-{what}-N`,
    /// each taking the next chunk from a shared queue and checking `stop`
    /// between chunks. Reads use each part's own descriptor (a FUSE reopen
    /// without keep-cache would drop the inode's cached pages), whose
    /// `POSIX_FADV_RANDOM` keeps every read to its own chunk. `progress`
    /// receives each finished chunk's bytes on its reader thread; with
    /// `drop_cached` the chunk's page cache is released once copied. Returns
    /// the bytes read, fewer than the chunks hold after a stop.
    fn read_chunks(&self, chunks: Vec<PayloadChunk<'_>>, readers: usize, stop: &AtomicBool,
        progress: &(impl Fn(u64) + Sync), drop_cached: bool, what: &'static str) -> Result<u64> {
        let scratch = chunks.iter().filter(|chunk| chunk.destination.is_none()).map(|chunk| chunk.length).max()
            .unwrap_or(0);
        let readers = readers.max(1).min(chunks.len());
        let queue = Mutex::new(chunks.into_iter());
        let failed = AtomicBool::new(false);
        let read = || -> Result<u64> {
            let mut buffer = vec![0; scratch];
            let mut bytes = 0;
            while !stop.load(Ordering::Acquire) && !failed.load(Ordering::Relaxed) {
                let Some(chunk) = queue.lock().unwrap_or_else(|p| p.into_inner()).next() else { break };
                let part = &self.parts[chunk.part];
                let target = match chunk.destination {
                    Some(destination) => destination,
                    None => &mut buffer[..chunk.length],
                };
                if let Err(source) = part.read_payload(chunk.offset, target) {
                    failed.store(true, Ordering::Relaxed);
                    return Err(MappedTableError::Io { context: format!("mapped table {what} read"), source });
                }
                if drop_cached {
                    part.drop_cached(chunk.offset, chunk.length);
                }
                bytes += chunk.length as u64;
                progress(chunk.length as u64);
            }
            Ok(bytes)
        };
        let results = thread::scope(|scope| {
            let mut results = Vec::with_capacity(readers);
            let mut handles = Vec::with_capacity(readers);
            for index in 0..readers {
                match thread::Builder::new().name(format!("table-{what}-{index}")).spawn_scoped(scope, &read) {
                    Ok(handle) => handles.push(handle),
                    Err(source) => {
                        failed.store(true, Ordering::Relaxed);
                        results.push(Err(MappedTableError::Io { context: format!("starting mapped table {what} reader"), source }));
                        break;
                    }
                }
            }
            results.extend(handles.into_iter().map(|handle| handle.join()
                .unwrap_or(Err(MappedTableError::Stopped("mapped table reader panicked")))));
            results
        });
        results.into_iter().sum::<Result<u64>>()
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
        if let Some(bytes) = &self.resident {
            let offset = row as usize * self.row_bytes();
            return Ok(&bytes[offset..offset + self.row_bytes()]);
        }
        self.parts[part].row(local)
    }

    /// Copy selected rows in input order, retaining duplicates and allocating
    /// no staging memory; the whole batch is validated before any write.
    pub fn gather_into(&self, rows: &[u64], output: &mut [u8]) -> Result<()> {
        self.gather_into_priority(rows, output, ReadPriority::Prefill)
    }
    pub fn gather_into_priority(&self, rows: &[u64], output: &mut [u8], priority: ReadPriority) -> Result<()> {
        let size = rows.len().checked_mul(self.row_bytes()).ok_or_else(|| invalid("gather size overflow"))?;
        if output.len() != size {
            return Err(invalid("gather output size does not match requested rows"));
        }
        for &row in rows {
            self.locate(row)?;
        }
        let slots: Vec<_> = (0..rows.len()).collect();
        self.gather_slots_priority(rows, &slots, output, priority)?;
        Ok(())
    }

    /// Gather into distinct row slots (for deduplicated engram rows). The
    /// destination layout is shared by both backends; all rows are validated
    /// before any write, and every io_uring batch drains before returning.
    pub fn gather_slots(&self, rows: &[u64], slots: &[usize], output: &mut [u8]) -> Result<()> {
        self.gather_slots_priority(rows, slots, output, ReadPriority::Prefill)
    }
    pub fn gather_slots_priority(&self, rows: &[u64], slots: &[usize], output: &mut [u8], priority: ReadPriority) -> Result<()> {
        let pending = self.prepare_gather(rows, slots, output, priority, 2)?;
        pending.finish(self, output)
    }

    /// Copy hits now and submit only misses. The returned ticket owns its I/O
    /// buffers/files, not the destination, so another table can start first.
    pub(crate) fn prepare_gather(&self, rows: &[u64], slots: &[usize], output: &mut [u8],
        priority: ReadPriority, window: usize) -> Result<PreparedTableGather> {
        if rows.len() != slots.len() { return Err(invalid("gather row/slot counts differ")); }
        let width = self.row_bytes();
        let mut unique = BTreeSet::new();
        let locations = rows.iter().zip(slots).map(|(&row, &slot)| {
            let (part, local) = self.locate(row)?;
            if slot.checked_add(1).and_then(|n| n.checked_mul(width)).is_none_or(|n| n > output.len())
                || !unique.insert(slot) { return Err(invalid("invalid or overlapping gather slots")); }
            Ok((part, local, slot))
        }).collect::<Result<Vec<_>>>()?;
        let mut prepared = PreparedTableGather { misses: Vec::new(), stream: None,
            stats: Default::default(), output_len: output.len(), width };
        if self.resident.is_some() {
            for (&row, &slot) in rows.iter().zip(slots) {
                output[slot * width..(slot + 1) * width].copy_from_slice(self.row(row)?);
            }
            return Ok(prepared);
        }
        let hits = self.residency(rows, false)?;
        let before = thread_faults();
        for ((part, row, slot), hit) in locations.into_iter().zip(hits) {
            if hit {
                output[slot * width..(slot + 1) * width].copy_from_slice(self.parts[part].row(row)?);
                prepared.stats.hits += 1;
                prepared.stats.hit_bytes += width as u64;
            } else {
                prepared.misses.push((part, row, slot));
                prepared.stats.misses += 1;
                prepared.stats.miss_bytes += width as u64;
                prepared.stats.device_bytes_estimate += self.parts[part].pages(row)?.count() as u64 * self.parts[part].page_bytes as u64;
            }
        }
        prepared.stats.resident_copy_rows = prepared.stats.hits;
        prepared.stats.late_major_faults = thread_faults_since(before)[1].max(0) as u64;
        if !prepared.misses.is_empty() {
            let reads = prepared.misses.iter().map(|&(part, row, slot)| reactor::Read {
                file: Arc::clone(&self.parts[part].file), offset: self.parts[part].file_offset + row * width as u64, slot }).collect();
            match uring::start(reads, width, priority, window) {
                Ok(stream) => prepared.stream = Some(stream),
                Err(error) => self.uring_fallback(&error),
            }
        }
        Ok(prepared)
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

    fn uring_reads(&self, rows: &[u64], reads: &[uring::Read<'_>], output: &mut [u8], prefetch: bool,
        priority: ReadPriority) -> Result<uring::ReadStats> {
        // All residency snapshots and cache-hit copies stay on the caller.
        // mincore is intentionally approximate, including on NOWAIT-capable files.
        let hits = self.residency(rows, prefetch)?;
        let mut pending = Vec::with_capacity(reads.len());
        let before = thread_faults();
        let mut stats = uring::ReadStats::default();
        for (read, hit) in reads.iter().zip(hits) {
            if hit {
                let width = self.row_bytes();
                output[read.slot * width..(read.slot + 1) * width].copy_from_slice(read.part.row(read.row)?);
                stats.hits += 1;
                stats.hit_bytes += width as u64;
            } else { pending.push(uring::Read { part: read.part, row: read.row, slot: read.slot }); }
        }
        stats.resident_copy_rows = stats.hits;
        stats.late_major_faults = thread_faults_since(before)[1].max(0) as u64;
        if !pending.is_empty() {
            let misses = match uring::gather_priority(&pending, output, self.row_bytes(), &vec![Some(false); pending.len()], priority) {
                Ok(misses) => misses,
                Err(error) => {
                    self.uring_fallback(&error);
                    let mut misses = uring::ReadStats::default();
                    for read in &pending {
                        let width = self.row_bytes();
                        output[read.slot * width..(read.slot + 1) * width].copy_from_slice(read.part.row(read.row)?);
                        misses.misses += 1;
                        misses.miss_bytes += width as u64;
                        misses.device_bytes_estimate += read.part.pages(read.row)?.count() as u64 * read.part.page_bytes as u64;
                    }
                    misses
                },
            };
            stats.misses = misses.misses;
            stats.miss_bytes = misses.miss_bytes;
            stats.device_bytes_estimate = misses.device_bytes_estimate;
            stats.miss_histogram = misses.miss_histogram;
            stats.miss_ns = misses.miss_ns;
            stats.miss_max_ns = misses.miss_max_ns;
        }
        Ok(stats)
    }

    fn uring_fallback(&self, error: &MappedTableError) {
        // Only this batch falls back. The shared owner retries a failed ring
        // with bounded backoff instead of permanently disabling a table.
        tracing::warn!(%error, "mapped table read batch unavailable; using mmap for this batch");
    }

    fn backend(&self) -> TableBackend {
        self.stats.backend()
    }

    /// Finalize after preload admission, before readiness or gather workers.
    /// Fully private tables need neither a ring nor its process-wide workspace.
    pub fn prepare_reads(&self) {
        if self.resident.is_none() { self.initialize_reads(); }
    }

    fn initialize_reads(&self) {
        self.initialize_with_probe(|| self.parts.iter().try_for_each(|part| {
            part.nowait.store(uring::probe(part)?, Ordering::Relaxed);
            Ok(())
        }));
    }
    fn initialize_with_probe(&self, probe: impl FnOnce() -> Result<()>) {
        let result = probe();
        let available = result.is_ok();
        if let Err(error) = result { tracing::warn!(%error, "io_uring unavailable (kernel/filesystem/seccomp); using mmap fallback"); }
        self.stats.nowait.store(self.parts.iter().all(|part| part.nowait.load(Ordering::Relaxed)), Ordering::Relaxed);
        tracing::info!(backend = %self.backend(), max_ring_entries = uring::MAX_BATCH,
            iowq_max_workers = uring::IOWQ_MAX_WORKERS,
            accounting = self.stats.accounting(),
            reason = if !available { "reactor setup pending; recoverable batch fallback" }
                else { "caller resident memcpy and process-wide miss reactor" },
            locality_note = if self.stats.approximate.load(Ordering::Relaxed) { "FUSE residency is approximate and does not count physical device reads" } else { "mincore residency is a snapshot, not a physical-read counter" },
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

    /// Advise the deduplicated pages a batch touches, with a hard page budget
    /// over all parts; returns the pages advised. Call from an I/O worker.
    pub fn prefetch(&self, rows: &[u64], max_pages: usize) -> Result<usize> {
        if self.resident.is_some() {
            for &row in rows { self.locate(row)?; }
            return Ok(0);
        }
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
        if !rows.is_empty() {
            let mut output = vec![0; rows.len().checked_mul(self.row_bytes()).ok_or_else(|| invalid("prefetch size overflow"))?];
            let reads: Vec<_> = rows.iter().enumerate().map(|(slot, &row)| {
                let (part, local) = self.locate(row)?;
                Ok(uring::Read { part: &self.parts[part], row: local, slot })
            }).collect::<Result<_>>()?;
            match self.uring_reads(rows, &reads, &mut output, true, ReadPriority::Prefill) {
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
    resident: AtomicBool,
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
    nowait: AtomicBool,
    approximate: AtomicBool,
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
    pub(crate) cache_hit_rows: AtomicU64,
    pub(crate) cache_wait_rows: AtomicU64,
    pub(crate) cache_denied_rows: AtomicU64,
    pub(crate) cache_evictions: AtomicU64,
    pub(crate) lookahead_rows: AtomicU64,
    pub(crate) lookahead_pieces: AtomicU64,
    pub(crate) stall_prefill_events: AtomicU64,
    pub(crate) stall_prefill_ns: AtomicU64,
    pub(crate) stall_prefill_fallback_rows: AtomicU64,
    pub(crate) stall_prefill_pending_rows: AtomicU64,
    pub(crate) stall_decode_events: AtomicU64,
    pub(crate) stall_decode_ns: AtomicU64,
    pub(crate) stall_decode_fallback_rows: AtomicU64,
    pub(crate) stall_decode_pending_rows: AtomicU64,
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
    /// Row-cache lookups already published when a gather needed them.
    #[serde(default)]
    pub cache_hit_rows: u64,
    /// Row-cache lookups still being filled by another reader when a gather needed them.
    #[serde(default)]
    pub cache_wait_rows: u64,
    /// Rows a class could not reserve (its share was full); read uncached.
    #[serde(default)]
    pub cache_denied_rows: u64,
    /// Published rows evicted to make room.
    #[serde(default)]
    pub cache_evictions: u64,
    /// Rows the look-ahead read into the row cache.
    #[serde(default)]
    pub lookahead_rows: u64,
    /// Look-ahead pieces (both tables of a few hundred tokens) completed.
    #[serde(default)]
    pub lookahead_pieces: u64,
    /// Prefill consumers that needed a layer's rows before they were ready (Engram stalls).
    #[serde(default)]
    pub stall_prefill_events: u64,
    /// Prefill consumer time spent waiting for those rows.
    #[serde(default)]
    pub stall_prefill_ns: u64,
    /// Rows prefill gathers read themselves: the look-ahead had not fetched them.
    #[serde(default)]
    pub stall_prefill_fallback_rows: u64,
    /// Rows prefill gathers waited for: queued by the look-ahead or another gather, not yet landed.
    #[serde(default)]
    pub stall_prefill_pending_rows: u64,
    /// Decode consumers that needed a layer's rows before they were ready (Engram stalls).
    #[serde(default)]
    pub stall_decode_events: u64,
    /// Decode consumer time spent waiting for those rows.
    #[serde(default)]
    pub stall_decode_ns: u64,
    /// Rows decode gathers read themselves (not in the row cache).
    #[serde(default)]
    pub stall_decode_fallback_rows: u64,
    /// Rows decode gathers waited for while another reader filled them.
    #[serde(default)]
    pub stall_decode_pending_rows: u64,
}

impl TableStats {
    fn backend(&self) -> TableBackend {
        if self.resident.load(Ordering::Relaxed) { TableBackend::Resident }
        else if !reactor::ready() { TableBackend::Mmap }
        else { TableBackend::MincoreRouted }
    }
    fn accounting(&self) -> &'static str {
        if self.resident.load(Ordering::Relaxed) { "resident; no page-in" }
        else if self.backend() == TableBackend::MincoreRouted {
            if self.approximate.load(Ordering::Relaxed) { "fuse: mincore-routed (approx)" } else { "mincore-routed" }
        }
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
    /// A consumer that needed a layer's rows before they were ready (an Engram
    /// stall), split by phase.
    pub fn record_consumer_stall(&self, decode: bool, elapsed: Duration) {
        let (events, ns) = if decode { (&self.stall_decode_events, &self.stall_decode_ns) }
            else { (&self.stall_prefill_events, &self.stall_prefill_ns) };
        events.fetch_add(1, Ordering::Relaxed);
        ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }
    /// One gather's rows not already in the row cache: `fallback` it read itself,
    /// `pending` it waited for while another reader filled them.
    pub fn record_gather_misses(&self, decode: bool, fallback: usize, pending: usize) {
        let (read, waited) = if decode { (&self.stall_decode_fallback_rows, &self.stall_decode_pending_rows) }
            else { (&self.stall_prefill_fallback_rows, &self.stall_prefill_pending_rows) };
        read.fetch_add(fallback as u64, Ordering::Relaxed);
        waited.fetch_add(pending as u64, Ordering::Relaxed);
    }
    pub fn record_lookahead(&self, rows: usize) {
        self.lookahead_pieces.fetch_add(1, Ordering::Relaxed);
        self.lookahead_rows.fetch_add(rows as u64, Ordering::Relaxed);
    }
    pub fn record_prefetch(&self, pages: usize) {
        self.prefetch_jobs.fetch_add(1, Ordering::Relaxed);
        self.prefetch_pages.fetch_add(pages as u64, Ordering::Relaxed);
    }
    pub fn record_prefetch_dropped(&self) {
        self.prefetch_dropped.fetch_add(1, Ordering::Relaxed);
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
            cache_hit_rows: get(&self.cache_hit_rows),
            cache_wait_rows: get(&self.cache_wait_rows),
            cache_denied_rows: get(&self.cache_denied_rows),
            cache_evictions: get(&self.cache_evictions),
            lookahead_rows: get(&self.lookahead_rows),
            lookahead_pieces: get(&self.lookahead_pieces),
            stall_prefill_events: get(&self.stall_prefill_events),
            stall_prefill_ns: get(&self.stall_prefill_ns),
            stall_prefill_fallback_rows: get(&self.stall_prefill_fallback_rows),
            stall_prefill_pending_rows: get(&self.stall_prefill_pending_rows),
            stall_decode_events: get(&self.stall_decode_events),
            stall_decode_ns: get(&self.stall_decode_ns),
            stall_decode_fallback_rows: get(&self.stall_decode_fallback_rows),
            stall_decode_pending_rows: get(&self.stall_decode_pending_rows),
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
            cache_hit_rows: d(self.cache_hit_rows, earlier.cache_hit_rows),
            cache_wait_rows: d(self.cache_wait_rows, earlier.cache_wait_rows),
            cache_denied_rows: d(self.cache_denied_rows, earlier.cache_denied_rows),
            cache_evictions: d(self.cache_evictions, earlier.cache_evictions),
            lookahead_rows: d(self.lookahead_rows, earlier.lookahead_rows),
            lookahead_pieces: d(self.lookahead_pieces, earlier.lookahead_pieces),
            stall_prefill_events: d(self.stall_prefill_events, earlier.stall_prefill_events),
            stall_prefill_ns: d(self.stall_prefill_ns, earlier.stall_prefill_ns),
            stall_prefill_fallback_rows: d(self.stall_prefill_fallback_rows, earlier.stall_prefill_fallback_rows),
            stall_prefill_pending_rows: d(self.stall_prefill_pending_rows, earlier.stall_prefill_pending_rows),
            stall_decode_events: d(self.stall_decode_events, earlier.stall_decode_events),
            stall_decode_ns: d(self.stall_decode_ns, earlier.stall_decode_ns),
            stall_decode_fallback_rows: d(self.stall_decode_fallback_rows, earlier.stall_decode_fallback_rows),
            stall_decode_pending_rows: d(self.stall_decode_pending_rows, earlier.stall_decode_pending_rows),
        }
    }
    /// Fraction classified resident by NOWAIT or an approximate mincore snapshot.
    pub fn resident_rate(&self) -> f64 {
        let rows = self.page_hits + self.page_misses;
        if rows == 0 { return 1.0; }
        self.page_hits as f64 / rows as f64
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

    /// Copy rows in order (duplicates kept); validate the batch before writing.
    pub fn gather(&self, table: &MappedTable, rows: &[u64], output: &mut [u8])
        -> Result<GatherReport> {
        if Some(output.len()) != rows.len().checked_mul(table.row_bytes()) {
            return Err(invalid("gather output size does not match requested rows"));
        }
        let slots: Vec<_> = (0..rows.len()).collect();
        let report = self.gather_slots(table, rows, &slots, output)?;
        table.stats.record_gather(rows.len(), output.len(), report.elapsed, [0, report.major_faults, 0]);
        Ok(report)
    }

    /// Parallel reads into validated, disjoint canonical row slots.
    pub fn gather_slots(&self, table: &MappedTable, rows: &[u64], slots: &[usize], output: &mut [u8]) -> Result<GatherReport> {
        use rayon::prelude::*;
        let started = Instant::now();
        let row_bytes = table.row_bytes();
        let routed = table.resident.is_some() || reactor::global().is_ok_and(|owner| {
            if let Err(error) = owner.wake() { tracing::warn!(%error, "cannot wake table reactor recovery"); }
            owner.ready()
        });
        if routed {
            table.gather_slots(rows, slots, output)?;
            return Ok(GatherReport { rows: rows.len(), elapsed: started.elapsed(), major_faults: 0 });
        }
        if rows.len() != slots.len() { return Err(invalid("gather row/slot count mismatch")); }
        let mut unique = BTreeSet::new();
        for (&row, &slot) in rows.iter().zip(slots) {
            table.locate(row)?;
            if slot.checked_add(1).and_then(|n| n.checked_mul(row_bytes)).is_none_or(|n| n > output.len()) || !unique.insert(slot) {
                return Err(invalid("invalid or overlapping gather slots"));
            }
        }
        let faults = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
        // Several chunks per thread balance uneven fault latency.
        let chunk = slots.len().div_ceil(self.threads * 4).max(1);
        let base = output.as_mut_ptr() as usize;
        let output_len = output.len();
        {
            table.record_residency(rows)?;
            self.pool.install(|| {
                rows.par_chunks(chunk).zip(slots.par_chunks(chunk)).try_for_each(|(addresses, indices)| -> Result<()> {
                    let before = thread_faults();
                    for (&row, &index) in addresses.iter().zip(indices) {
                        let source = table.row(row)?;
                        debug_assert!((index + 1) * row_bytes <= output_len);
                        // SAFETY: `slots` holds distinct indices, so every task
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
        let elapsed = started.elapsed();
        let faults = faults.map(|value| value.load(Ordering::Relaxed) as i64);
        Ok(GatherReport { rows: rows.len(), elapsed, major_faults: faults[1] })
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
    pub unsafe fn spawn_gather(self: &Arc<Self>, table: Arc<MappedTable>, rows: Vec<u64>, output: *mut u8, len: usize) -> PendingGather {
        let (sender, done) = mpsc::sync_channel(1);
        let pool = Arc::clone(self);
        let output = output as usize;
        self.pool.spawn(move || {
            // SAFETY: valid and exclusive until the handle observes completion (contract above).
            let out = unsafe { std::slice::from_raw_parts_mut(output as *mut u8, len) };
            let _ = sender.send(pool.gather(&table, &rows, out));
        });
        PendingGather { done, finished: false }
    }
}

/// Split-phase gather with no borrowed destination or mapping pointer. The
/// table owner supplies the same immutable table and destination at completion.
pub(crate) struct PreparedTableGather {
    misses: Vec<(usize, u64, usize)>, stream: Option<reactor::ReadStream>,
    stats: uring::ReadStats, output_len: usize, width: usize,
}
impl PreparedTableGather {
    /// Rows that were not resident when the gather started.
    pub(crate) fn misses(&self) -> usize { self.stats.misses as usize }
    pub(crate) fn step(&mut self, table: &MappedTable, output: &mut [u8]) -> Result<bool> {
        if output.len() != self.output_len || table.row_bytes() != self.width {
            return Err(invalid("prepared gather destination changed"));
        }
        if let Some(stream) = &mut self.stream {
            if let Err(error) = stream.step(output) {
                // Drop drains submitted reads before mmap fallback or storage reuse.
                drop(self.stream.take());
                table.uring_fallback(&error);
            } else if !stream.finished() { return Ok(false); }
            else {
                let stats = stream.stats();
                self.stats.miss_ns = stats.miss_ns;
                self.stats.miss_max_ns = stats.miss_max_ns;
                self.stats.miss_histogram = stats.miss_histogram;
                table.stats.record_reads(self.stats, false);
                self.misses.clear();
                self.stream = None;
                self.stats = Default::default();
                return Ok(true);
            }
        }
        for &(part, row, slot) in &self.misses {
            output[slot * self.width..(slot + 1) * self.width].copy_from_slice(table.parts[part].row(row)?);
        }
        table.stats.record_reads(self.stats, false);
        self.misses.clear();
        self.stats = Default::default();
        Ok(true)
    }
    pub(crate) fn finish(mut self, table: &MappedTable, output: &mut [u8]) -> Result<()> {
        while !self.step(table, output)? {}
        Ok(())
    }
    /// Land whatever has completed without waiting for the device; true once
    /// every row is in `output`.
    pub(crate) fn poll(&mut self, table: &MappedTable, output: &mut [u8]) -> Result<bool> {
        if let Some(stream) = &mut self.stream {
            let failed = loop {
                if stream.finished() { break None; }
                match stream.try_step(output) {
                    Ok(true) => continue,
                    Ok(false) => return Ok(false),
                    Err(error) => break Some(error),
                }
            };
            if let Some(error) = failed {
                // As in `step`: drain the stream, then copy every miss from the mapping.
                drop(self.stream.take());
                table.uring_fallback(&error);
            }
        }
        // Finished: record its stats, or copy the misses after a failure.
        self.step(table, output)
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
    recycler: Arc<GatherSlots<S>>,
    job: J,
    timing: Option<GatherTiming>,
}

impl<S: Send + 'static, J> GatherLease<S, J> {
    pub fn slot(&self) -> Option<&S> {
        self.slot.as_ref()
    }
    pub fn slot_mut(&mut self) -> Option<&mut S> { self.slot.as_mut() }
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
            self.recycler.free.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(slot);
            self.recycler.admission.notify();
        }
    }
}

struct GatherSlots<S> {
    free: Mutex<Vec<S>>,
    admission: GatherAdmission,
    stopped: AtomicBool,
}
struct GatherWorkerExit<S: Send + 'static, J, E> {
    slots: Arc<GatherSlots<S>>,
    queue: GatherQueue<S, J, E>,
}
impl<S: Send + 'static, J, E> Drop for GatherWorkerExit<S, J, E> {
    fn drop(&mut self) {
        self.slots.stopped.store(true, Ordering::Release);
        let (lock, ready) = &*self.queue;
        let rejected = {
            let mut queues = lock.lock().unwrap_or_else(|p| p.into_inner());
            queues.closed = true;
            let mut rejected = std::mem::take(&mut queues.priority);
            rejected.append(&mut queues.normal);
            rejected
        };
        // Completion senders wake queued tickets on unwind; recycle leases
        // outside the queue lock so admission callbacks cannot re-enter it.
        drop(rejected);
        ready.notify_all();
        self.slots.admission.notify();
    }
}

#[derive(Default)]
struct GatherAdmission(Mutex<Vec<(Arc<()>, std::task::Waker)>>);
impl GatherAdmission {
    fn waiter(&self) -> GatherAdmissionWait<'_> {
        GatherAdmissionWait { admission: self, key: Arc::new(()) }
    }
    fn notify(&self) {
        let waiters = std::mem::take(&mut *self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        for (_, wake) in waiters { wake.wake(); }
    }
}

/// A registration belongs to its future, not its executor task: two futures
/// may share one waker, and dropping either must not remove the other's wake.
struct GatherAdmissionWait<'a> {
    admission: &'a GatherAdmission,
    key: Arc<()>,
}
impl GatherAdmissionWait<'_> {
    fn register(&self, wake: &std::task::Waker) {
        let mut waiters = self.admission.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, old)) = waiters.iter_mut().find(|(key, _)| Arc::ptr_eq(key, &self.key)) {
            old.clone_from(wake);
        } else {
            waiters.push((Arc::clone(&self.key), wake.clone()));
        }
    }
}
impl Drop for GatherAdmissionWait<'_> {
    fn drop(&mut self) {
        let mut waiters = self.admission.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(index) = waiters.iter().position(|(key, _)| Arc::ptr_eq(key, &self.key)) {
            waiters.swap_remove(index);
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

type GatherResult<S, J, E> = std::result::Result<Option<GatherLease<S, J>>, E>;
type Completion<S, J, E> = mpsc::Receiver<GatherResult<S, J, E>>;
type GatherWake = Arc<Mutex<Option<std::task::Waker>>>;

struct GatherCompletionSender<S: Send + 'static, J, E> {
    sender: Option<mpsc::SyncSender<GatherResult<S, J, E>>>,
    wake: GatherWake,
}

impl<S: Send + 'static, J, E> GatherCompletionSender<S, J, E> {
    fn send(self, result: GatherResult<S, J, E>) {
        if let Some(sender) = &self.sender { let _ = sender.send(result); }
        // Drop disconnects the channel before waking, including on worker panic.
    }
}

impl<S: Send + 'static, J, E> Drop for GatherCompletionSender<S, J, E> {
    fn drop(&mut self) {
        self.sender.take();
        let wake = self.wake.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
        if let Some(wake) = wake { wake.wake(); }
    }
}

/// Independent of scheduler slot reuse; drop cancels queued/in-progress work.
/// An in-progress OS page fault cannot be interrupted, but its result is discarded.
pub struct GatherTicket<S: Send + 'static, J, E> {
    cancelled: Arc<AtomicBool>,
    completion: Option<Completion<S, J, E>>,
    wake: GatherWake,
    consumed: bool,
}

impl<S: Send + 'static, J, E> GatherTicket<S, J, E> {
    pub fn cancel(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.completion.take();
        self.wake.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
    }

    /// Await a worker notification without polling or occupying an executor thread.
    /// Resolves to Ready, Cancelled or an error, never Pending. Dropping this
    /// future leaves the ticket usable; only a terminal result consumes it.
    pub async fn wait(&mut self) -> std::result::Result<GatherPoll<GatherLease<S, J>>, GatherFailure<E>> {
        std::future::poll_fn(|cx| {
            // Register before checking the channel: a concurrent completion either
            // wakes this task or is observed by poll, so no notification is lost.
            *self.wake.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(cx.waker().clone());
            match self.poll() {
                Ok(GatherPoll::Pending) => std::task::Poll::Pending,
                result => {
                    self.wake.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
                    std::task::Poll::Ready(result)
                }
            }
        }).await
    }

    /// Synchronous issuers sleep at the consuming layer, never dispatch another
    /// worker or spin while pages arrive. A timeout cancels this generation.
    pub fn wait_timeout(&mut self, timeout: Duration) -> std::result::Result<GatherPoll<GatherLease<S, J>>, GatherFailure<E>> {
        if self.consumed || self.cancelled.load(Ordering::Acquire) { return self.poll(); }
        let result = self.completion.as_ref().ok_or(GatherFailure::Worker(
            MappedTableError::Stopped("gather completion receiver is closed")))?.recv_timeout(timeout);
        self.consumed = true;
        self.completion.take();
        match result {
            Ok(result) => match result.map_err(GatherFailure::Job)? {
                Some(lease) => Ok(GatherPoll::Ready(lease)), None => Ok(GatherPoll::Cancelled),
            },
            Err(_) => { self.cancel(); Err(GatherFailure::Worker(MappedTableError::Stopped("gather wait timed out or worker stopped"))) }
        }
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
    completion: GatherCompletionSender<S, J, E>,
}

/// Pending jobs: priority first, each FIFO (the worker never runs a normal
/// job while a priority one is queued).
struct GatherQueues<S: Send + 'static, J, E> {
    priority: std::collections::VecDeque<WorkerJob<S, J, E>>,
    normal: std::collections::VecDeque<WorkerJob<S, J, E>>,
    closed: bool,
}
type GatherQueue<S, J, E> = Arc<(Mutex<GatherQueues<S, J, E>>, std::sync::Condvar)>;

struct GatherPermit(Arc<(Mutex<usize>, std::sync::Condvar)>);
impl GatherPermit {
    fn acquire(state: Arc<(Mutex<usize>, std::sync::Condvar)>) -> Self {
        let (lock, available) = &*state;
        let mut free = lock.lock().unwrap_or_else(|p| p.into_inner());
        while *free == 0 { free = available.wait(free).unwrap_or_else(|p| p.into_inner()); }
        *free -= 1;
        drop(free);
        Self(state)
    }
}
impl Drop for GatherPermit {
    fn drop(&mut self) {
        *self.0.0.lock().unwrap_or_else(|p| p.into_inner()) += 1;
        self.0.1.notify_one();
    }
}

/// Bounded background gathers into a fixed pool of reusable staging slots:
/// the pool bounds queued, in-flight and completed-but-unconsumed storage, and
/// submission never waits (None is backpressure). Priority submissions run
/// before every queued normal one.
pub struct GatherWorker<S: Send + 'static, J: Send + 'static, E: Send + 'static> {
    queue: GatherQueue<S, J, E>,
    worker: Option<JoinHandle<()>>,
    finishers: Vec<JoinHandle<()>>,
    slots: Arc<GatherSlots<S>>,
    shutdown: Arc<AtomicBool>,
}

impl<S: Send + 'static, J: Send + 'static, E: Send + 'static> GatherWorker<S, J, E> {
    /// `run` fills a slot for a job on the worker thread.
    pub fn new(name: &str, slots: Vec<S>, mut run: impl FnMut(&mut S, &J) -> std::result::Result<(), E> + Send + 'static)
        -> Result<Self> {
        if slots.is_empty() {
            return Err(invalid("a gather worker needs at least one staging slot"));
        }
        let slots = Arc::new(GatherSlots { free: Mutex::new(slots), admission: GatherAdmission::default(),
            stopped: AtomicBool::new(false) });
        let queue: GatherQueue<S, J, E> = Arc::new((Mutex::new(GatherQueues {
            priority: Default::default(), normal: Default::default(), closed: false }), std::sync::Condvar::new()));
        let jobs = Arc::clone(&queue);
        let shutdown = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&shutdown);
        let worker_slots = Arc::clone(&slots);
        let exit_queue = Arc::clone(&queue);
        let next = move || -> Option<WorkerJob<S, J, E>> {
            let (lock, ready) = &*jobs;
            let mut queues = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            loop {
                if let Some(job) = queues.priority.pop_front().or_else(|| queues.normal.pop_front()) { return Some(job); }
                if queues.closed { return None; }
                queues = ready.wait(queues).unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        };
        let worker = thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let _exit = GatherWorkerExit { slots: worker_slots, queue: exit_queue };
                while let Some(mut job) = next() {
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
        Ok(Self { queue, worker: Some(worker), finishers: Vec::new(), slots, shutdown })
    }

    /// One ordered preparation owner and two persistent completion workers.
    /// The permit is held from preparation through completion, so at most two
    /// jobs may own miss windows, independent of queued staging capacity.
    pub(crate) fn split<P: Send + 'static>(name: &str, staging: Vec<S>,
        mut prepare: impl FnMut(&mut S, &J) -> std::result::Result<P, E> + Send + 'static,
        finish: impl Fn(&mut S, &J, P) -> std::result::Result<(), E> + Send + Sync + 'static) -> Result<Self> {
        if staging.is_empty() { return Err(invalid("a gather worker needs at least one staging slot")); }
        let slots = Arc::new(GatherSlots { free: Mutex::new(staging), admission: GatherAdmission::default(),
            stopped: AtomicBool::new(false) });
        let queue: GatherQueue<S, J, E> = Arc::new((Mutex::new(GatherQueues {
            priority: Default::default(), normal: Default::default(), closed: false }), std::sync::Condvar::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let permits = Arc::new((Mutex::new(2usize), std::sync::Condvar::new()));
        let (send, receive) = mpsc::sync_channel::<(WorkerJob<S, J, E>, P, GatherPermit, Option<Instant>)>(2);
        let receive = Arc::new(Mutex::new(receive));
        let finish = Arc::new(finish);
        let mut finishers = Vec::new();
        for index in 0..2 {
            let (receive, finish, stopping) = (Arc::clone(&receive), Arc::clone(&finish), Arc::clone(&shutdown));
            finishers.push(thread::Builder::new().name(format!("{name}-finish-{index}")).spawn(move || {
                loop {
                    let item = receive.lock().unwrap_or_else(|p| p.into_inner()).recv();
                    let Ok((mut job, prepared, _permit, started)) = item else { break; };
                    let before = job.submitted.and_then(|_| thread_faults());
                    let result = if stopping.load(Ordering::Acquire) || job.cancelled.load(Ordering::Acquire) {
                        drop(prepared);
                        let _ = job.completion.send(Ok(None));
                        continue;
                    } else { finish(job.lease.slot.as_mut().expect("admitted staging"), &job.lease.job, prepared) };
                    if let (Some(submitted), Some(started)) = (job.submitted, started) {
                        let completed = Instant::now();
                        let faults = thread_faults_since(before);
                        job.lease.timing = Some(GatherTiming { queued: started.duration_since(submitted),
                            gather: completed.duration_since(started), completed,
                            minor_faults: faults[0], major_faults: faults[1], input_blocks: faults[2] });
                    }
                    let result = if stopping.load(Ordering::Acquire) || job.cancelled.load(Ordering::Acquire) {
                        Ok(None)
                    } else { result.map(|()| Some(job.lease)) };
                    let _ = job.completion.send(result);
                }
            }).map_err(|source| MappedTableError::Io { context: format!("starting {name} finisher"), source })?);
        }
        let (jobs, worker_slots, stopping) = (Arc::clone(&queue), Arc::clone(&slots), Arc::clone(&shutdown));
        let exit_queue = Arc::clone(&queue);
        let worker = thread::Builder::new().name(name.into()).spawn(move || {
            let _exit = GatherWorkerExit { slots: worker_slots, queue: exit_queue };
            loop {
                // Keep queued jobs visible to priority admission while both
                // finishers own their permits; choose only after one returns.
                let permit = GatherPermit::acquire(Arc::clone(&permits));
                let mut job = {
                    let (lock, ready) = &*jobs;
                    let mut queues = lock.lock().unwrap_or_else(|p| p.into_inner());
                    loop {
                        if let Some(job) = queues.priority.pop_front().or_else(|| queues.normal.pop_front()) { break job; }
                        if queues.closed { return; }
                        queues = ready.wait(queues).unwrap_or_else(|p| p.into_inner());
                    }
                };
                if stopping.load(Ordering::Acquire) || job.cancelled.load(Ordering::Acquire) {
                    let _ = job.completion.send(Ok(None));
                    continue;
                }
                let started = job.submitted.map(|_| Instant::now());
                // Fault counters are per-thread; finishers measure their own
                // work rather than subtracting the preparation owner's values.
                match prepare(job.lease.slot.as_mut().expect("admitted staging"), &job.lease.job) {
                    Ok(prepared) => {
                        if send.send((job, prepared, permit, started)).is_err() { return; }
                    },
                    Err(error) => { let _ = job.completion.send(Err(error)); },
                }
            }
        }).map_err(|source| MappedTableError::Io { context: format!("starting {name} preparation"), source })?;
        Ok(Self { queue, worker: Some(worker), finishers, slots, shutdown })
    }

    /// Submit early, when row ids become available. None is bounded
    /// backpressure; this performs no mapped reads or blocking wait.
    /// `timed` records [`GatherTiming`] on the lease.
    pub fn try_submit(&self, job: J, timed: bool) -> Result<Option<GatherTicket<S, J, E>>> {
        self.try_submit_with(job, timed, false)
    }
    /// [`Self::try_submit`]; `priority` runs it before every queued normal job.
    pub fn try_submit_with(&self, job: J, timed: bool, priority: bool) -> Result<Option<GatherTicket<S, J, E>>> {
        if self.slots.stopped.load(Ordering::Acquire) { return Err(MappedTableError::Stopped("gather worker stopped")); }
        let slot = match self.slots.free.try_lock() {
            Ok(mut free) => match free.pop() { Some(slot) => slot, None => return Ok(None) },
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MappedTableError::Stopped("gather staging pool poisoned")),
        };
        self.submit_slot(slot, job, timed, priority).map(Some)
    }

    /// Snapshot admission capacity without reserving a slot. CUDA-backed
    /// consumers use this to distinguish event stalls from worker-only waits.
    pub fn capacity_available(&self) -> Result<bool> {
        self.slots.free.lock().map(|free| !free.is_empty())
            .map_err(|_| MappedTableError::Stopped("gather staging pool poisoned"))
    }

    /// Wait until a released lease makes admission possible. This is a
    /// notification, not a reservation: callers retry try_submit after waking.
    pub async fn wait_for_capacity(&self) -> Result<()> {
        let waiter = self.slots.admission.waiter();
        std::future::poll_fn(|cx| {
            waiter.register(cx.waker());
            if self.slots.stopped.load(Ordering::Acquire) {
                return std::task::Poll::Ready(Err(MappedTableError::Stopped("gather worker stopped")));
            }
            match self.slots.free.lock() {
                Ok(free) if free.is_empty() => std::task::Poll::Pending,
                Ok(_) => std::task::Poll::Ready(Ok(())),
                Err(_) => std::task::Poll::Ready(Err(MappedTableError::Stopped("gather staging pool poisoned"))),
            }
        }).await
    }

    /// Await admission without discarding the unsubmitted job or spinning.
    pub async fn submit(&self, job: J, timed: bool) -> Result<GatherTicket<S, J, E>> {
        self.submit_with(job, timed, false).await
    }
    /// [`Self::submit`] with [`Self::try_submit_with`]'s priority.
    pub async fn submit_with(&self, job: J, timed: bool, priority: bool) -> Result<GatherTicket<S, J, E>> {
        let waiter = self.slots.admission.waiter();
        let slot = std::future::poll_fn(|cx| {
            waiter.register(cx.waker());
            if self.slots.stopped.load(Ordering::Acquire) {
                return std::task::Poll::Ready(Err(MappedTableError::Stopped("gather worker stopped")));
            }
            match self.slots.free.lock() {
                Ok(mut free) => match free.pop() {
                    Some(slot) => std::task::Poll::Ready(Ok(slot)),
                    None => std::task::Poll::Pending,
                },
                Err(_) => std::task::Poll::Ready(Err(MappedTableError::Stopped("gather staging pool poisoned"))),
            }
        }).await?;
        self.submit_slot(slot, job, timed, priority)
    }

    fn submit_slot(&self, slot: S, job: J, timed: bool, priority: bool) -> Result<GatherTicket<S, J, E>> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let wake = Arc::new(Mutex::new(None));
        let (sender, receive) = mpsc::sync_channel(1);
        let job = WorkerJob {
            submitted: timed.then(Instant::now),
            lease: GatherLease { slot: Some(slot), recycler: Arc::clone(&self.slots), job, timing: None },
            cancelled: Arc::clone(&cancelled),
            completion: GatherCompletionSender { sender: Some(sender), wake: Arc::clone(&wake) },
        };
        // Acquiring one of N slots bounds the queue at N jobs.
        let (lock, ready) = &*self.queue;
        let mut queues = lock.lock().map_err(|_| MappedTableError::Stopped("gather queue poisoned"))?;
        if queues.closed { return Err(MappedTableError::Stopped("gather worker stopped")); }
        if priority { queues.priority.push_back(job) } else { queues.normal.push_back(job) }
        drop(queues);
        ready.notify_one();
        Ok(GatherTicket { cancelled, completion: Some(receive), wake, consumed: false })
    }
}

impl<S: Send + 'static, J: Send + 'static, E: Send + 'static> Drop for GatherWorker<S, J, E> {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        {
            let (lock, ready) = &*self.queue;
            lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).closed = true;
            ready.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                tracing::error!("gather worker panicked during shutdown");
            }
        }
        for worker in self.finishers.drain(..) {
            if worker.join().is_err() { tracing::error!("gather finisher panicked during shutdown"); }
        }
    }
}

#[cfg(test)]
mod tests;
