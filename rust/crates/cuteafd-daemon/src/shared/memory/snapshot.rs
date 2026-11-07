//! Bounded snapshot storage allocated once, before serving becomes ready.
use super::*;
use anyhow::{ensure, Context};
use std::{cell::RefCell, rc::Rc};

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
        // Only free the lane once the drain is proven. A failed drain may still
        // have copies in flight; clearing `pending` would let the request be
        // released and its storage reused underneath them (finding 3).
        if drained.is_ok() { self.pending = None; }
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
            stride.checked_mul(slots).context("snapshot arena overflow")?)?;
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
        let owned = DeviceAllocation::new(library, bytes)?;
        Ok(Self { buffer: owned.buffer, _owned: Some(owned), pooled: None })
    }
}
impl Drop for SnapshotStorage<'_> {
    fn drop(&mut self) {
        if let Some((pool, slot)) = self.pooled.take() { pool.0.free.borrow_mut().push(slot); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
