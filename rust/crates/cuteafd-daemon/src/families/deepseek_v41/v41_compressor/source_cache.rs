//! Paired index/FP4 KV pages of one compressed source. The prefix engine owns the pages
//! (`cuteafd_engine::prefix::RefPagePool`, one 512-token unit = one page of each ratio-two source
//! and two of the ratio-one source); a request slot's page table is bound once, at admission, to
//! the units of its whole lifetime (`bind`), so appends only publish rows inside bound pages:
//! no allocation, copy-on-write or reservation happens on the decode path. Rows past a slot's
//! committed length are its own; shared pages hold only rows below every sharer's length.
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41Kv};
use std::ffi::c_void;
use std::rc::Rc;
pub(crate) mod replica;

/// Capacity failure for one entry in the caller's ordered append transaction.
/// Binding/ownership failures deliberately use different error types.
#[derive(Debug)]
pub(crate) struct SourcePoolExhausted {
    pub work_index: usize,
    pub needed: usize,
    pub available: usize,
}
impl std::fmt::Display for SourcePoolExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compressed KV pool exhausted at work item {}: need {} pages, {} available",
            self.work_index, self.needed, self.available)
    }
}
impl std::error::Error for SourcePoolExhausted {}

pub(crate) const PAGE_ROWS: usize = 256;
const KV_VALUES: usize = V41Kv::COMPRESSED_VALUE_BYTES;
const KV_SCALES: usize = V41Kv::COMPRESSED_SCALE_BYTES;
pub(crate) const SOURCE_ROW_BYTES: usize = 68 + V41Kv::COMPRESSED_ROW_BYTES;
pub(crate) struct SourceCache<'a> {
    pub packed: DeviceAllocation<'a>,
    pub scales: DeviceAllocation<'a>,
    pub kv_values: DeviceAllocation<'a>,
    pub kv_scales: DeviceAllocation<'a>,
    pub capacity: usize,
    page_table: DeviceAllocation<'a>,
    lengths: DeviceAllocation<'a>,
    staging: HostAllocation<'a>,
    stride: usize,
    /// Physical pages bound to each slot, in row order; rows below `rows` are initialized.
    pages: [Vec<u32>; 16],
    rows: [usize; 16],
    writing: Rc<std::cell::Cell<u16>>,
    pub(super) replica: Option<replica::SourceCacheReplica<'a>>,
}
/// An append transaction over bound pages: the new committed length of each participating
/// slot. Its slots are marked writing from `reserve` until `apply` or drop.
pub(super) struct IndexPlan {
    lengths: Vec<(usize, u64)>,
    guard: Option<WriteGuard>,
}
struct WriteGuard {
    flags: Rc<std::cell::Cell<u16>>,
    mask: u16,
}
impl Drop for WriteGuard {
    fn drop(&mut self) {
        self.flags.set(self.flags.get() & !self.mask);
    }
}
/// Only rows below `rows` are initialized. Logical row r uses physical page
/// pages[r / 256], offset r % 256. Drain device consumers before mutating owner.
/// The borrow prevents host-side release/commit while this view is in use.
pub(crate) struct IndexCacheView<'a> {
    pub packed: CuteafdDeviceBuffer,
    pub scales: CuteafdDeviceBuffer,
    pub pages: &'a [u32],
    pub rows: usize,
    /// U32 physical page IDs, capacity entries; only ceil(rows/256) are valid.
    pub device_pages: CuteafdDeviceBuffer,
    /// U64 committed row count, published after accepted value/scale writes.
    pub device_rows: CuteafdDeviceBuffer,
}
/// FP4 K16 serving KV shares physical pages and publication with index keys.
pub(crate) struct KvCacheView<'a> {
    pub values: CuteafdDeviceBuffer,
    pub scales: CuteafdDeviceBuffer,
    pub pages: &'a [u32],
    pub rows: usize,
    pub device_pages: CuteafdDeviceBuffer,
    pub device_rows: CuteafdDeviceBuffer,
}
impl<'a> SourceCache<'a> {
    pub fn device_bytes(pages: usize, slots: usize) -> Result<usize> {
        ensure!(
            (1..=262144).contains(&pages),
            "invalid index pool page count"
        );
        ensure!((1..=16).contains(&slots), "invalid index slot count");
        Ok(pages * PAGE_ROWS * SOURCE_ROW_BYTES + slots * (pages.min(4096) * 4 + 8))
    }
    pub fn new(library: &'a NativeLibrary, pages: usize, slots: usize) -> Result<Self> {
        Self::device_bytes(pages, slots)?;
        let lengths = DeviceAllocation::new(library, slots * 8)?;
        library.copy_h2d(lengths.buffer, &vec![0; slots * 8])?;
        Ok(Self {
            page_table: DeviceAllocation::new(library, slots * pages.min(4096) * 4)?,
            lengths,
            staging: HostAllocation::new(library, slots * 8)?,
            stride: pages.min(4096),
            packed: DeviceAllocation::new(library, pages * PAGE_ROWS * 64)?,
            scales: DeviceAllocation::new(library, pages * PAGE_ROWS * 4)?,
            kv_values: DeviceAllocation::new(library, pages * PAGE_ROWS * KV_VALUES)?,
            kv_scales: DeviceAllocation::new(library, pages * PAGE_ROWS * KV_SCALES)?,
            capacity: pages * PAGE_ROWS,
            pages: std::array::from_fn(|_| vec![]),
            rows: [0; 16],
            writing: Default::default(),
            replica: None,
        })
    }
    /// Physical pages of this source.
    pub fn page_count(&self) -> usize {
        self.capacity / PAGE_ROWS
    }
    pub fn view(&self, slot: usize, rows: usize) -> IndexCacheView<'_> {
        IndexCacheView {
            packed: self.packed.buffer,
            scales: self.scales.buffer,
            pages: &self.pages[slot],
            rows,
            device_pages: slice(
                self.page_table.buffer,
                slot * self.stride * 4,
                self.stride * 4,
            ),
            device_rows: slice(self.lengths.buffer, slot * 8, 8),
        }
    }
    pub fn kv_view(&self, slot: usize, rows: usize) -> KvCacheView<'_> {
        let index = self.view(slot, rows);
        KvCacheView {
            values: self.kv_values.buffer,
            scales: self.kv_scales.buffer,
            pages: index.pages,
            rows: index.rows,
            device_pages: index.device_pages,
            device_rows: index.device_rows,
        }
    }
    pub fn ensure_idle(&self, slot: usize) -> Result<()> {
        ensure!(slot < self.slots() && self.writing.get() & (1 << slot) == 0,
            "compressed cache slot has a pending append");
        Ok(())
    }
    /// Forget a slot's pages; its owner (the prefix engine) releases them after every consumer
    /// drained.
    pub fn release(&mut self, slot: usize) -> Result<()> {
        self.ensure_idle(slot)?;
        self.pages[slot].clear();
        self.rows[slot] = 0;
        // The lease is revoked and consumers drained. reset/bind install replacement device
        // metadata before a new owner becomes usable.
        Ok(())
    }
    pub fn reset(&self, slot: usize) -> Result<()> {
        self.ensure_idle(slot)?;
        self.lengths
            .library
            .copy_h2d(slice(self.lengths.buffer, slot * 8, 8), &[0; 8])?;
        if let Some(replica)=&self.replica { replica.storage.install_metadata(slot,&[],0)?; }
        Ok(())
    }
    /// Bind `pages` (physical, in row order) to a fresh slot or extend its binding, and publish
    /// its initialized `rows` (pages the prefix engine forked hold them). Every page past the
    /// existing binding is new to the slot. Device page-table entries are written before the
    /// length that exposes them.
    pub fn bind(&mut self, slot: usize, pages: &[u32], rows: usize) -> Result<()> {
        self.ensure_idle(slot)?;
        let bound = self.pages[slot].len();
        ensure!(pages.len() >= bound && pages[..bound] == self.pages[slot][..] && pages.len() <= self.stride
            && pages.iter().all(|&p| (p as usize) < self.page_count())
            && rows <= pages.len() * PAGE_ROWS && rows >= self.rows[slot],
            "source page binding differs from the slot's pages or exceeds the pool");
        let library = self.lengths.library;
        let added: Vec<u8> = pages[bound..].iter().flat_map(|p| p.to_ne_bytes()).collect();
        if !added.is_empty() {
            library.copy_h2d(slice(self.page_table.buffer, (slot * self.stride + bound) * 4, added.len()), &added)?;
        }
        if rows != self.rows[slot] || bound == 0 {
            library.copy_h2d(slice(self.lengths.buffer, slot * 8, 8), &(rows as u64).to_ne_bytes())?;
        }
        if let Some(replica)=&self.replica {
            let all: Vec<u8> = pages.iter().flat_map(|p| p.to_ne_bytes()).collect();
            replica.storage.install_metadata(slot,&all,rows)?;
        }
        self.pages[slot] = pages.to_vec();
        self.rows[slot] = rows;
        Ok(())
    }
    /// Drop the unwritten tail of a slot's binding: `pages` is a prefix of its pages that still
    /// holds its committed `rows`. Device page-table entries past it are never read again.
    pub fn shrink(&mut self, slot: usize, pages: &[u32], rows: usize) -> Result<()> {
        self.ensure_idle(slot)?;
        ensure!(self.pages[slot].starts_with(pages) && rows == self.rows[slot] && rows <= pages.len() * PAGE_ROWS,
            "shrunk source binding must keep the slot's committed rows");
        self.pages[slot].truncate(pages.len());
        Ok(())
    }
    /// Pages bound to `slot` and its initialized rows.
    pub fn binding(&self, slot: usize) -> (&[u32], usize) {
        (&self.pages[slot], self.rows[slot])
    }
    /// The four device segments holding `page`'s rows (packed index, index scales, KV values,
    /// KV scales), in the order the host cache stores them.
    pub fn page_segments(&self, page: u32) -> [CuteafdDeviceBuffer; 4] {
        let rows = |buffer: CuteafdDeviceBuffer, bytes: usize| {
            slice(buffer, page as usize * PAGE_ROWS * bytes, PAGE_ROWS * bytes)
        };
        [
            rows(self.packed.buffer, 64),
            rows(self.scales.buffer, 4),
            rows(self.kv_values.buffer, KV_VALUES),
            rows(self.kv_scales.buffer, KV_SCALES),
        ]
    }
    /// # Safety
    /// Copy rows `[0, rows)` of physical page `from` into page `to` (a forked snapshot's partial
    /// tail) on `stream`. `to` belongs to no reader yet; keep both live until the stream drains.
    pub unsafe fn copy_page_rows(&self, from: u32, to: u32, rows: usize, stream: *mut c_void) -> Result<()> {
        ensure!(rows <= PAGE_ROWS && (from as usize) < self.page_count() && (to as usize) < self.page_count(),
            "source tail copy outside the pool");
        if rows == 0 { return Ok(()); }
        for (buffer, width) in [(self.packed.buffer, 64), (self.scales.buffer, 4),
            (self.kv_values.buffer, KV_VALUES), (self.kv_scales.buffer, KV_SCALES)] {
            let bytes = rows * width;
            unsafe {
                self.lengths.library.copy_d2d_async(
                    slice(buffer, to as usize * PAGE_ROWS * width, bytes),
                    slice(buffer, from as usize * PAGE_ROWS * width, bytes),
                    bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// Value/scales writes precede this call on stream. Drain the stream before
    /// reusing staging, releasing a slot or publishing the plan on the host.
    pub unsafe fn upload(&mut self, plan: &IndexPlan, stream: *mut c_void) -> Result<()> {
        self.validate_plan(plan)?;
        let (library, lengths) = (self.lengths.library, self.lengths.buffer);
        unsafe { upload_lengths(library, lengths, plan, self.staging.bytes_mut(), stream) }
    }
    /// # Safety
    /// Same publication ordering as upload. Staging belongs to the producer and
    /// remains pinned and untouched until this stream drains.
    pub unsafe fn upload_staged(&self, plan: &IndexPlan, staging: &mut [u8], stream: *mut c_void) -> Result<()> {
        self.validate_plan(plan)?;
        unsafe { upload_lengths(self.lengths.library, self.lengths.buffer, plan, staging, stream) }
    }
    pub fn validate_plan(&self, plan: &IndexPlan) -> Result<()> {
        let guard = plan.guard.as_ref().ok_or_else(|| anyhow::anyhow!("source plan not reserved"))?;
        ensure!(Rc::ptr_eq(&guard.flags, &self.writing)
            && self.writing.get() & guard.mask == guard.mask, "foreign or lost source reservation");
        Ok(())
    }
    /// Claim the append slots after validating every participant: each append stays inside its
    /// slot's bound pages. Disjoint plans may coexist and apply in either order. After queueing
    /// GPU writes, drain before applying or dropping the plan.
    pub fn reserve(&self, appends: &[(usize, usize, usize)]) -> Result<IndexPlan> {
        let slots = self.slots();
        let mut plan = plan_appends(&self.pages[..slots], &self.rows[..slots], self.writing.get(), appends)?;
        let mask = appends.iter().fold(0u16, |mask, &(slot, _, _)| mask | 1 << slot);
        self.writing.set(self.writing.get() | mask);
        plan.guard = Some(WriteGuard { flags: self.writing.clone(), mask });
        Ok(plan)
    }
    fn slots(&self) -> usize {
        self.lengths.buffer.bytes / 8
    }
    pub fn destination(&self, plan: &IndexPlan, slot: usize, row: usize) -> Result<u64> {
        ensure!(plan.lengths.iter().any(|&(s, end)| s == slot && (row as u64) < end), "index append outside its plan");
        let page = *self.pages[slot].get(row / PAGE_ROWS).context("index append outside bound pages")?;
        Ok(u64::from(page) * PAGE_ROWS as u64 + (row % PAGE_ROWS) as u64)
    }
    pub fn apply(&mut self, mut plan: IndexPlan) {
        let guard = plan.guard.take().expect("source plan is not reserved");
        assert!(Rc::ptr_eq(&self.writing, &guard.flags), "foreign source plan");
        for (slot, rows) in plan.lengths {
            self.rows[slot] = rows as usize;
        }
        drop(guard);
    }
}

/// Validate `appends` (`(slot, old rows, new rows)`) against the bound `tables`: an append
/// writes only rows past the slot's committed length and inside its bound pages.
fn plan_appends(tables: &[Vec<u32>], rows: &[usize], writing: u16, appends: &[(usize, usize, usize)])
    -> Result<IndexPlan> {
    let mut plan = IndexPlan { lengths: vec![], guard: None };
    let mut seen = [false; 16];
    for (position, &(slot, old, new)) in appends.iter().enumerate() {
        ensure!(
            slot < tables.len() && !seen[slot],
            "duplicate or invalid index slot"
        );
        ensure!(writing & (1 << slot) == 0, "compressed cache slot has a pending append");
        seen[slot] = true;
        ensure!(old == rows[slot] && old <= new && new <= 1048576, "index history binding differs");
        let bound = tables[slot].len() * PAGE_ROWS;
        ensure!(new <= bound, SourcePoolExhausted { work_index: position, needed: new.div_ceil(PAGE_ROWS) - tables[slot].len(),
            available: 0 });
        plan.lengths.push((slot, new as u64));
    }
    Ok(plan)
}

unsafe fn upload_lengths(library: &NativeLibrary, lengths: CuteafdDeviceBuffer, plan: &IndexPlan,
    staging: &mut [u8], stream: *mut c_void) -> Result<()> {
    ensure!(staging.len() >= plan.lengths.len() * 8, "source metadata staging too small");
    for (i, &(slot, rows)) in plan.lengths.iter().enumerate() {
        let staging = &mut staging[i * 8..i * 8 + 8];
        staging.copy_from_slice(&rows.to_ne_bytes());
        unsafe {
            library.copy_h2d_async(slice(lengths, slot * 8, 8), staging, stream)?;
        }
    }
    Ok(())
}

fn slice(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    CuteafdDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() },
        bytes,
        ..buffer
    }
}

#[cfg(test)]
mod high_pages;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::LoadStream;

    pub(super) fn append(
        cache: &mut SourceCache<'_>,
        slot: usize,
        old: usize,
        new: usize,
        value: u8,
        stream: *mut c_void,
    ) -> Result<()> {
        let plan = cache.reserve(&[(slot, old, new)])?;
        let library = cache.lengths.library;
        for row in old..new {
            let destination = cache.destination(&plan, slot, row)? as usize;
            for (buffer, width) in [
                (cache.packed.buffer, 64),
                (cache.scales.buffer, 4),
                (cache.kv_values.buffer, KV_VALUES),
                (cache.kv_scales.buffer, KV_SCALES),
            ] {
                library.copy_h2d(slice(buffer, destination * width, width), &vec![value; width])?;
            }
        }
        unsafe {
            cache.upload(&plan, stream)?;
            library.cuda_stream_synchronize(stream)?;
        }
        cache.apply(plan);
        Ok(())
    }

    pub(super) fn read(cache: &SourceCache<'_>, slot: usize, rows: usize) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for row in 0..rows {
            let physical =
                cache.pages[slot][row / PAGE_ROWS] as usize * PAGE_ROWS + row % PAGE_ROWS;
            for (buffer, width) in [
                (cache.packed.buffer, 64),
                (cache.scales.buffer, 4),
                (cache.kv_values.buffer, KV_VALUES),
                (cache.kv_scales.buffer, KV_SCALES),
            ] {
                let start = bytes.len();
                bytes.resize(start + width, 0);
                cache
                    .lengths
                    .library
                    .copy_d2h(&mut bytes[start..], slice(buffer, physical * width, width))?;
            }
        }
        Ok(bytes)
    }

    #[test]
    fn appends_stay_inside_bound_pages_and_never_overlap() {
        let tables = [vec![3, 7], vec![3, 9]];
        let plan = plan_appends(&tables, &[300, 256], 0, &[(0, 300, 512), (1, 256, 400)]).unwrap();
        assert_eq!(plan.lengths, [(0, 512), (1, 400)]);
        // Past the binding is pool pressure (the admission bound the lifetime); a wrong old
        // length, a duplicate or a writing slot is a binding error.
        let error = plan_appends(&tables, &[300, 256], 0, &[(0, 300, 513)]).err().unwrap();
        assert!(error.downcast_ref::<SourcePoolExhausted>().is_some());
        for (appends, writing) in [(vec![(0, 299, 300)], 0), (vec![(0, 300, 301), (0, 300, 301)], 0),
            (vec![(0, 300, 301)], 1), (vec![(2, 0, 1)], 0)] {
            let error = plan_appends(&tables, &[300, 256], writing, &appends).err().unwrap();
            assert!(error.downcast_ref::<SourcePoolExhausted>().is_none(), "{appends:?}");
        }
    }

    #[test]
    #[ignore = "requires a CUDA native library in CUTEAFD_NATIVE_LIB"]
    fn native_bound_sources_share_full_pages_and_copy_tails() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        let mut cache = SourceCache::new(&library, 6, 3)?;
        cache.bind(0, &[0, 1], 0)?;
        append(&mut cache, 0, 0, 270, 0x31, stream.raw)?;
        let original = read(&cache, 0, 270)?;
        assert!(original.iter().all(|&v| v == 0x31));
        // A fork at 270 rows: page 0 shared, the partial tail copied into page 2.
        unsafe {
            cache.copy_page_rows(1, 2, 14, stream.raw)?;
            library.cuda_stream_synchronize(stream.raw)?;
        }
        cache.bind(1, &[0, 2], 270)?;
        assert_eq!(read(&cache, 1, 270)?, original);
        append(&mut cache, 1, 270, 300, 0x72, stream.raw)?;
        assert_eq!(read(&cache, 0, 270)?, original, "the writer's rows are untouched");
        let branch = read(&cache, 1, 300)?;
        assert!(branch[270 * SOURCE_ROW_BYTES..].iter().all(|&v| v == 0x72));
        // Binding grows a slot's pages and never rewrites the bound prefix.
        cache.bind(1, &[0, 2, 4], 300)?;
        assert!(cache.bind(1, &[0, 5, 4], 300).is_err());
        assert!(cache.bind(1, &[0, 2, 4], 299).is_err());
        assert!(cache.bind(2, &[6], 0).is_err(), "page outside the pool");
        append(&mut cache, 1, 300, 600, 0x55, stream.raw)?;
        assert!(cache.reserve(&[(1, 600, 769)]).is_err());
        cache.release(0)?;
        cache.release(1)?;
        cache.reset(1)?;
        assert_eq!(cache.binding(1), (&[][..], 0));
        Ok(())
    }
}
