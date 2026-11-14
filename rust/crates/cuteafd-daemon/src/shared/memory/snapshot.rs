//! Bounded snapshot storage allocated once, before serving becomes ready.
use super::*;
use anyhow::{ensure, Context};
use std::{cell::RefCell, rc::Rc};
use crate::shared::memory::Tag;

/// Stream destruction drains before the queued source/storage owners are dropped.
pub(crate) struct SnapshotCopies<'a, T> {
    pub stream: LoadStream<'a>,
    pub pending: Option<T>,
}
impl<'a, T> SnapshotCopies<'a, T> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
        Ok(Self { stream: LoadStream { library, raw: library.cuda_stream_create()? }, pending: None })
    }
    pub fn ready(&self) -> Result<bool> {
        ensure!(self.pending.is_some(), "snapshot copy is not pending");
        unsafe { self.stream.library.cuda_stream_query(self.stream.raw) }
    }
    pub fn abort(&mut self) -> Result<()> {
        if self.pending.is_none() { return Ok(()); }
        let drained = unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) };
        self.pending = None;
        drained
    }
}

struct Arena<'a> {
    allocation: DeviceAllocation<'a>,
    stride: usize,
    free: RefCell<Vec<usize>>,
}

#[derive(Clone)]
pub(crate) struct SnapshotPool<'a>(Rc<Arena<'a>>);
impl<'a> SnapshotPool<'a> {
    pub fn new(library: &'a NativeLibrary, bytes: usize, slots: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
        ensure!(bytes > 0 && slots > 0, "empty snapshot arena");
        let stride = bytes.checked_add(255).context("snapshot stride overflow")? / 256 * 256;
        let allocation = DeviceAllocation::new(library,
            stride.checked_mul(slots).context("snapshot arena overflow")?, Tag::prefix("shared/prefix-snapshots"))?;
        Ok(Self(Rc::new(Arena { allocation, stride, free: RefCell::new((0..slots).rev().collect()) })))
    }
    pub fn device_bytes(&self) -> usize { self.0.allocation.buffer.bytes }
    pub(crate) fn allocation_receipts(&self, inventory: &mut super::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        inventory.backed_device(name, self.0.allocation.buffer, self.0.allocation.buffer)?;
        let free = self.0.free.try_borrow()?;
        inventory.host(&format!("{name}.free"), &free)?;
        Ok(())
    }
    pub fn take(&self, bytes: usize) -> Result<SnapshotStorage<'a>> {
        ensure!(bytes > 0 && bytes <= self.0.stride, "snapshot exceeds arena slot");
        let slot = self.0.free.borrow_mut().pop().context("snapshot arena exhausted")?;
        let mut buffer = self.0.allocation.buffer;
        buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(slot * self.0.stride).cast() };
        buffer.bytes = bytes;
        Ok(SnapshotStorage { buffer, _owned: None, pooled: Some((self.clone(), slot)) })
    }
}

/// The owner must drain every reader/writer before dropping this storage.
/// A pooled slot returns to the free list without calling the CUDA allocator.
pub(crate) struct SnapshotStorage<'a> {
    pub buffer: CuteafdDeviceBuffer,
    _owned: Option<DeviceAllocation<'a>>,
    pooled: Option<(SnapshotPool<'a>, usize)>,
}
impl<'a> SnapshotStorage<'a> {
    pub(crate) fn allocation_receipts(&self, inventory: &mut super::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        let backing = self.pooled.as_ref().map(|(pool, _)| pool.0.allocation.buffer)
            .or_else(|| self._owned.as_ref().map(|owner| owner.buffer))
            .context("snapshot storage has no actual owner")?;
        inventory.backed_device(name, self.buffer, backing)
    }
    pub fn new(library: &'a NativeLibrary, bytes: usize, pool: Option<&SnapshotPool<'a>>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
        if let Some(pool) = pool { return pool.take(bytes); }
        let owned = DeviceAllocation::new(library, bytes, Tag::prefix("shared/prefix-snapshot"))?;
        Ok(Self { buffer: owned.buffer, _owned: Some(owned), pooled: None })
    }
}
impl Drop for SnapshotStorage<'_> {
    fn drop(&mut self) {
        if let Some((pool, slot)) = self.pooled.take() { pool.0.free.borrow_mut().push(slot); }
    }
}

/// Byte-exact marks for a windowed drafter. The mark stores every physical
/// K/V plane and its canonical origin/frontier; restore never reruns arithmetic.
pub(crate) struct WindowSnapshots<'a> {
    copies: crate::shared::prefix::PrefixCopies<'a>,
    arena: Rc<device::Allocation<'a>>,
    planes: Vec<usize>,
    stride: usize,
    marks: usize,
}

impl<'a> WindowSnapshots<'a> {
    pub fn bytes(planes: &[usize], marks: usize) -> Result<(usize, usize)> {
        ensure!(marks > 0 && !planes.is_empty() && planes.iter().all(|&n| n > 0 && n % 16 == 0),
            "window snapshots require aligned nonempty planes and marks");
        let payload = planes.iter().try_fold(16usize, |sum, &n| sum.checked_add(n))
            .context("window snapshot payload overflow")?;
        let stride = payload.checked_add(255).context("window snapshot alignment overflow")? / 256 * 256;
        Ok((stride, stride.checked_mul(marks).context("window snapshot arena overflow")?))
    }

    /// The caller admits the exact arena returned by `bytes` before construction.
    pub fn new(stream: &device::Stream<'a>, planes: Vec<usize>, marks: usize) -> Result<Self> {
        let (stride, bytes) = Self::bytes(&planes, marks)?;
        let arena = Rc::new(device::Allocation::new(stream.device, bytes,
            Tag::prefix("shared/draft-window-snapshots"))?);
        let copies = crate::shared::prefix::PrefixCopies::new(stream.device.library,
            vec![vec![crate::shared::prefix::Reader { device: stream.device.id, stream: stream.raw }]])?;
        Ok(Self { copies, arena, planes, stride, marks })
    }

    pub fn backing(&self) -> Rc<device::Allocation<'a>> { Rc::clone(&self.arena) }
    pub fn mark_bytes(&self) -> usize { self.stride }
    pub fn inventory(&self, inventory: &mut receipts::Inventory<'_>, name: &str) -> Result<()> {
        self.arena.inventory(inventory, name)?;
        self.copies.allocation_receipts(inventory, &format!("{name}.copies"))?;
        inventory.host(&format!("{name}.planes"), &self.planes)
    }

    pub fn mark(&self, mark: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(mark < self.marks, "window snapshot mark outside admission");
        Ok(CuteafdDeviceBuffer { ptr: self.arena.buffer.ptr.wrapping_byte_add(mark * self.stride),
            bytes: self.stride, ..self.arena.buffer })
    }

    fn parts(&self, mark: usize, live: &[CuteafdDeviceBuffer], capture: bool)
        -> Result<Vec<(usize, CuteafdDeviceBuffer, CuteafdDeviceBuffer)>> {
        ensure!(live.len() == self.planes.len(), "window snapshot plane inventory differs");
        let saved = self.mark(mark)?;
        for (index, plane) in live.iter().enumerate() {
            let end = (plane.ptr as usize).checked_add(plane.bytes).context("window live plane overflow")?;
            ensure!(live[..index].iter().all(|other| (other.ptr as usize).checked_add(other.bytes)
                .is_some_and(|other_end| end <= other.ptr as usize || other_end <= plane.ptr as usize)),
                "window snapshot live planes overlap");
        }
        let mut offset = 16;
        let mut copies = Vec::with_capacity(live.len());
        for (&live, &bytes) in live.iter().zip(&self.planes) {
            ensure!(live.device_id == saved.device_id && !live.ptr.is_null() && live.bytes == bytes
                && live.ptr as usize % 16 == 0, "window snapshot live plane differs from admission");
            let part = CuteafdDeviceBuffer { ptr: saved.ptr.wrapping_byte_add(offset), bytes, ..saved };
            let a = self.arena.buffer.ptr as usize;
            let b = live.ptr as usize;
            ensure!(a.checked_add(self.arena.buffer.bytes).is_some_and(|end| end <= b)
                || b.checked_add(bytes).is_some_and(|end| end <= a),
                "window snapshot aliases live state");
            copies.push(if capture { (0, part, live) } else { (0, live, part) });
            offset += bytes;
        }
        Ok(copies)
    }

    /// # Safety
    /// Live planes belong to the ordered reader stream, do not overlap, and
    /// their owners survive `drain` and all subsequent restoration copies.
    pub unsafe fn capture(&self, mark: usize, live: &[CuteafdDeviceBuffer], origin: usize, frontier: usize) -> Result<()> {
        ensure!(origin <= frontier, "window snapshot frontier precedes origin");
        let copies = self.parts(mark, live, true)?;
        let mut metadata = [0u8; 16];
        metadata[..8].copy_from_slice(&u64::try_from(origin)?.to_le_bytes());
        metadata[8..].copy_from_slice(&u64::try_from(frontier)?.to_le_bytes());
        let saved = self.mark(mark)?;
        self.copies.ordered(0, |stream| {
            // SAFETY: the ordered uploader retains its host staging until the
            // copy stream retires; mark storage survives every queued consumer.
            unsafe { self.arena.device.library.copy_h2d_ordered(
                CuteafdDeviceBuffer { bytes: 16, ..saved }, &metadata, stream) }
        })?;
        // SAFETY: the caller retains all live planes; parts validated extents.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    /// # Safety
    /// The destination planes satisfy `capture`'s ownership/stream contract.
    /// The mark is initialized by capture or a completed host-tier restore.
    pub unsafe fn restore(&self, mark: usize, live: &[CuteafdDeviceBuffer], frontier: usize)
        -> Result<(usize, usize)> {
        let copies = self.parts(mark, live, false)?;
        self.copies.drain()?;
        let saved = self.mark(mark)?;
        let mut metadata = [0u8; 16];
        self.arena.device.run(|| self.arena.device.library.copy_d2h(&mut metadata,
            CuteafdDeviceBuffer { bytes: 16, ..saved }))?;
        let origin = usize::try_from(u64::from_le_bytes(metadata[..8].try_into()?))?;
        let end = usize::try_from(u64::from_le_bytes(metadata[8..].try_into()?))?;
        ensure!(origin <= end && end == frontier, "window snapshot metadata/frontier differs");
        // SAFETY: caller retains the initialized mark and destination planes.
        unsafe { self.copies.copy_d2d(&copies)?; }
        Ok((origin, end))
    }

    pub fn drain(&self) -> Result<()> { self.copies.drain() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_snapshot_restore_equals_cold_state_and_survives_slot_reuse() -> Result<()> {
        use device::{Allocation, Device, Stream};
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime};
        use cuteafd_core::memory_layout::Category;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        fixture.quiet_events(&library, true)?;
        let device = Device { library: &library, id: 1 };
        let stream = Stream::new(device)?;
        let planes = [Allocation::new(device, 128, TEST)?, Allocation::new(device, 256, TEST)?];
        let live = planes.iter().map(|plane| plane.buffer).collect::<Vec<_>>();
        let snapshots = WindowSnapshots::new(&stream, vec![128, 256], 2)?;
        // A cold window after wrap, with deterministic BF16/FP8 payload bytes.
        let cold = |frontier: usize, bytes: usize| {
            let rows = bytes / 16;
            let mut ring = vec![0u8; bytes];
            for position in 0..frontier {
                let at = position % rows * 16;
                for (column, value) in ring[at..at + 16].iter_mut().enumerate() {
                    *value = (position.wrapping_mul(37) + column * 11) as u8;
                }
            }
            ring
        };
        let expected = [cold(23, 128), cold(23, 256)];
        for (plane, bytes) in live.iter().zip(&expected) {
            device.run(|| library.copy_h2d(*plane, bytes))?;
        }
        // SAFETY: owned disjoint fixture planes survive the ordered copy owner.
        unsafe { snapshots.capture(0, &live, 0, 23)?; }
        snapshots.drain()?;
        for plane in &live { device.run(|| library.copy_h2d(*plane, &vec![197; plane.bytes]))?; }
        // Another request/snapshot must not overwrite the original mark.
        unsafe { snapshots.capture(1, &live, 23, 31)?; }
        snapshots.drain()?;
        assert!(unsafe { snapshots.restore(0, &live, 24) }.is_err());
        for plane in &live {
            let mut bytes = vec![0; plane.bytes];
            device.run(|| library.copy_d2h(&mut bytes, *plane))?;
            assert_eq!(bytes, vec![197; plane.bytes], "invalid restore must not mutate destinations");
        }
        assert_eq!(unsafe { snapshots.restore(0, &live, 23)? }, (0, 23));
        snapshots.drain()?;
        for (plane, cold) in live.iter().zip(&expected) {
            let mut restored = vec![0; plane.bytes];
            device.run(|| library.copy_d2h(&mut restored, *plane))?;
            assert_eq!(&restored, cold);
        }
        let before = fixture.events()?;
        let mut inventory = receipts::Inventory::new(CoordinatorDevices::new(&[1])?, "window",
            Lifetime::TARGET, Category::Prefix, "draft window marks");
        snapshots.inventory(&mut inventory, "marks")?;
        assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_none()
            && r.allocation.location == cuteafd_loader::placement::planner::Location::coordinator(0)).count(), 1);
        assert_eq!(fixture.events()?, before, "inventory must not enqueue native work");
        assert!(unsafe { snapshots.capture(2, &live, 0, 23) }.is_err());
        assert!(WindowSnapshots::bytes(&[usize::MAX - 15], 2).is_err());
        Ok(())
    }

    #[test]
    fn snapshot_census_charges_shared_backing_once_without_native_work() -> Result<()> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, Location};
        let (_directory, fixture) = super::super::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let pool = SnapshotPool::new(&library, 513, 3)?;
        let first = pool.take(513)?;
        let second = pool.take(129)?;
        let before = fixture.events()?;
        let physical = [0];
        let mut inventory = super::super::receipts::Inventory::new(CoordinatorDevices::new(&physical)?,
            "fixture", Lifetime::TARGET, Category::Prefix, "snapshot");
        pool.allocation_receipts(&mut inventory, "pool")?;
        first.allocation_receipts(&mut inventory, "first")?;
        second.allocation_receipts(&mut inventory, "second")?;
        let owners: Vec<_> = inventory.receipts.iter().filter(|r|
            r.allocation.location == Location::coordinator(0) && r.allocation.view.is_none()).collect();
        assert_eq!(owners.len(), 1);
        assert_eq!(owners[0].allocation.bytes, pool.device_bytes() as u64);
        for (name, actual) in [("first", first.buffer), ("second", second.buffer)] {
            let receipt = inventory.receipts.iter().find(|r| r.allocation.id == format!("fixture.{name}")).unwrap();
            assert_eq!((receipt.address, receipt.allocation.bytes), (actual.ptr as u64, actual.bytes as u64));
            assert_eq!(receipt.allocation.view.as_ref().unwrap().offset, actual.ptr as u64 - owners[0].address);
        }
        assert_eq!(fixture.events()?, before);
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_snapshot_arena_reuses_only_released_slots() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let pool = SnapshotPool::new(&lib, 513, 2)?;
        assert!(pool.take(769).is_err());
        let first = pool.take(513)?;
        let second = pool.take(129)?;
        assert_ne!(first.buffer.ptr, second.buffer.ptr);
        assert!(pool.take(1).is_err());
        lib.copy_h2d(first.buffer, &[17; 513])?;
        lib.copy_h2d(second.buffer, &[29; 129])?;
        let address = first.buffer.ptr;
        drop(first);
        let reused = pool.take(513)?;
        assert_eq!(reused.buffer.ptr, address);
        lib.copy_h2d(reused.buffer, &[41; 513])?;
        // Outstanding storage owns the arena even when its allocator is dropped.
        drop(pool);
        let mut actual = [0; 129];
        lib.copy_d2h(&mut actual, second.buffer)?;
        assert_eq!(actual, [29; 129]);
        Ok(())
    }
}
