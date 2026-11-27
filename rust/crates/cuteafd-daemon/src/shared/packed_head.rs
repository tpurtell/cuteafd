//! Exact 12-bit copies of BF16 vocabulary heads (`native/shared/cuda/packed_head.cu`; format in
//! `cuteafd_core::packed_head`): packed and verified at load from the resident BF16 head, then
//! applied by a tensor-core GEMV that reads the BF16 values exactly (both planes and the exception
//! table, FP32 sums) or, for drafters, the coarse plane alone (the traffic of an E4M3 copy).
use crate::shared::memory::DeviceAllocation;
use anyhow::{Context, Result};
use cuteafd_core::packed_head::{PackStatus, PackedHeadLayout, Region, STATUS_WORDS};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, PackedHeadPlanes};
use std::ffi::c_void;
use crate::shared::memory::Tag;

/// An `[n, k]` BF16 head in the exact 12-bit format: one allocation laid out by
/// [`PackedHeadLayout`] (planes, bases, exception table and the pack's status).
pub(crate) struct PackedHead<'a> {
    storage: DeviceAllocation<'a>,
    layout: PackedHeadLayout,
    status: std::cell::Cell<Option<PackStatus>>,
    pub n: usize,
    pub k: usize,
}

/// The native `planes` array of a head stored at `storage`.
fn planes(storage: CuteafdDeviceBuffer, layout: &PackedHeadLayout) -> PackedHeadPlanes {
    [Region::Coarse, Region::Fine, Region::Base, Region::RowPtr, Region::Columns, Region::Values]
        .map(|region| storage.ptr.wrapping_byte_add(layout.offset(region)))
}

fn status_buffer(storage: CuteafdDeviceBuffer, layout: &PackedHeadLayout) -> CuteafdDeviceBuffer {
    CuteafdDeviceBuffer { ptr: storage.ptr.wrapping_byte_add(layout.offset(Region::Status)), bytes: STATUS_WORDS * 8,
        ..storage }
}

impl<'a> PackedHead<'a> {
    /// Packs the live BF16 weight `w` [n, k] on `stream`, drains it and checks the pack's own
    /// bit-for-bit verification. A failed drain quarantines the destination and the native
    /// module ([`NativeLibrary::is_quarantined_after_failed_drain`]); the caller must then also
    /// retain the source.
    pub fn pack(library: &'a NativeLibrary, w: *const c_void, n: usize, k: usize, stream: *mut c_void)
        -> Result<Self> {
        Self::pack_with(library, w, n, k, stream, |_, bytes| {
            crate::shared::memory::device::Allocation::new(crate::shared::memory::device::Device {
                library, id: library.cuda_get_device()? }, bytes, Tag::weights("shared/packed-head"))
        })
    }

    pub fn pack_with(library: &'a NativeLibrary, w: *const c_void, n: usize, k: usize, stream: *mut c_void,
        allocate: impl FnMut(&str, usize) -> Result<crate::shared::memory::device::Allocation<'a>>)
        -> Result<Self> {
        let head = Self::allocate_with(library, n, k, allocate)?;
        head.initialize(w, stream)?;
        Ok(head)
    }

    /// Reserve the final arena view before push delivery and workspace construction.
    /// Projections are refused until `initialize` has drained and verified the pack.
    pub fn allocate_with(library: &'a NativeLibrary, n: usize, k: usize,
        allocate: impl FnMut(&str, usize) -> Result<crate::shared::memory::device::Allocation<'a>>)
        -> Result<Self> {
        let layout = PackedHeadLayout::new(n, k)?;
        let device = crate::shared::memory::device::Device { library, id: library.cuda_get_device()? };
        let storage = DeviceAllocation::from_allocation(crate::shared::memory::weight_arena::checked_weight_allocation(
            device, "packed12", layout.total_bytes(), allocate)?);
        Ok(Self { storage, layout, status: std::cell::Cell::new(None), n, k })
    }

    /// Pack a delivered BF16 source once, retaining all operands through the drain.
    pub fn initialize(&self, w: *const c_void, stream: *mut c_void) -> Result<()> {
        anyhow::ensure!(self.status.get().is_none(), "packed head already initialized");
        let library = self.storage.library;
        anyhow::ensure!(library.cuda_get_device()? == self.storage.buffer.device_id,
            "packed head initialized on another device");
        let (storage, layout, n, k) = (&self.storage, &self.layout, self.n, self.k);
        let status = status_buffer(storage.buffer, layout);
        // SAFETY: `w` is a live [n, k] BF16 weight; the planes and status are regions of `storage`,
        // sized by `layout` for this shape and `layout.capacity` table entries.
        let launched = unsafe { library.packed_head_pack(w, &planes(storage.buffer, &layout), layout.capacity, n, k,
            status.ptr, stream) };
        // SAFETY: the source and `storage` stay owned until this drain proves the pack retired; a
        // launch error may follow queued work, so drain on that path too.
        if let Err(drain) = unsafe { library.cuda_stream_synchronize(stream) } {
            // Completion is unknown: retain the destination until process teardown.
            library.quarantine_module_after_failed_drain();
            let error = launched.err().unwrap_or_else(|| anyhow::anyhow!("packed head [{n}, {k}] pack"));
            return Err(error.context(format!(
                "packed head destination and native module quarantined after failed drain: {drain}")));
        }
        launched?;
        let mut words = [0u8; STATUS_WORDS * 8];
        // Quiescent: load-time one-shot status read immediately after `cuda_stream_synchronize(stream)` above.
        library.copy_d2h_quiescent(&mut words, status)?;
        let status = PackStatus::from_bytes(&words);
        status.check(layout.capacity).with_context(|| format!("packed head [{n}, {k}]"))?;
        self.status.set(Some(status));
        Ok(())
    }

    /// Device bytes of the head (the planner's [`PackedHeadLayout::bytes`]).
    pub fn bytes(&self) -> usize { self.layout.total_bytes() }

    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        self.storage.inventory(inventory, &format!("{name}.packed12"))
    }

    /// The head's one allocation (planes, table and status).
    pub fn buffer(&self) -> CuteafdDeviceBuffer { self.storage.buffer }

    /// Values outside their rows' windows, and the most in any row.
    pub fn exceptions(&self) -> (u64, u64) {
        self.status.get().map_or((0, 0), |status| (status.exceptions, status.max_row_exceptions))
    }

    /// `out` [rows, n] (BF16, or FP32 with `out_f32`) = `x` [rows, k] @ W^T: the exact BF16
    /// weights, or with `coarse` the coarse plane alone (mantissa bits 3..0 at their midpoint).
    ///
    /// # Safety
    /// `x` and `out` are live device buffers of those shapes; `scratch` is live device memory
    /// of at least `packed_head_workspace(rows)` bytes (as [`scratch`] sizes it).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn apply(&self, library: &NativeLibrary, x: *const c_void, out: *mut c_void, out_f32: bool, rows: usize,
        coarse: bool, scratch: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        anyhow::ensure!(self.status.get().is_some(), "packed head has not been initialized");
        // SAFETY: the caller's contract; the planes are this head's live regions.
        unsafe { library.packed_head_linear(x, &planes(self.storage.buffer, &self.layout), out, out_f32, rows, self.k,
            self.n, coarse, scratch.ptr, scratch.bytes, stream) }
    }
}

/// GEMV scratch for up to `rows` rows of every `(k, n)` head shape.
pub(crate) fn scratch<'a>(library: &'a NativeLibrary, rows: usize, shapes: &[(usize, usize)], tag: Tag)
    -> Result<DeviceAllocation<'a>> {
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace/packed-head");
    let bytes = shapes.iter().map(|&(k, n)| library.packed_head_workspace(rows, k, n))
        .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    DeviceAllocation::new(library, bytes.max(256), tag)
}

#[cfg(test)]
impl PackedHead<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        f(format!("{prefix}packed12"), self.storage.buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_pack_matches_reference_and_retains_backing() -> Result<()> {
        use crate::shared::memory::{device::{Allocation, Device}, weight_arena::DeviceWeightArena, receipts::Inventory};
        use cuteafd_core::{memory_layout::Category, packed_head::PackedReference};
        use cuteafd_loader::{serving_capacity::WeightArenaLayout, placement::planner::{CoordinatorDevices, Lifetime, Phase}};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 1 };
        let (n, k) = (32, 128);
        let mut values = (0..n*k).map(|i| (((i % 2) as u16) << 15) | ((120 + i % 6) as u16) << 7
            | (i % 128) as u16).collect::<Vec<_>>();
        for (index, bits) in [(0, 0), (k+7, 0x7f80), (2*k+9, 0xffc1)] { values[index] = bits; }
        let reference = PackedReference::pack(&values, n, k)?;
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let source = Allocation::new(device, bytes.len(), crate::shared::memory::TEST)?;
        device.run(|| library.copy_h2d(source.buffer, &bytes))?;
        let baseline = device.run(|| PackedHead::pack(&library, source.buffer.ptr, n, k, std::ptr::null_mut()))?;
        let layout = PackedHeadLayout::new(n,k)?;
        let arena = DeviceWeightArena::new(device, WeightArenaLayout::new("test/head", [("packed12".into(), layout.total_bytes())])?, crate::shared::memory::TEST_ARENA)?;
        let packed = device.run(|| PackedHead::allocate_with(&library, n, k,
            |name, bytes| arena.view(name, bytes)))?;
        assert!(packed.status.get().is_none());
        let scratch = DeviceAllocation::new(&library, 256, crate::shared::memory::TEST)?;
        // SAFETY: initialization is checked before operands; this must reject without dereferencing them.
        assert!(unsafe { packed.apply(&library, std::ptr::null(), std::ptr::null_mut(), true, 1,
            false, scratch.buffer, std::ptr::null_mut()) }.is_err());
        device.run(|| packed.initialize(source.buffer.ptr, std::ptr::null_mut()))?;
        assert!(device.run(|| packed.initialize(source.buffer.ptr, std::ptr::null_mut())).is_err());
        drop(arena);
        let expected = [(Region::Coarse, reference.coarse),
            (Region::Fine, reference.fine.iter().flat_map(|v| v.to_le_bytes()).collect()),
            (Region::Base, reference.base),
            (Region::RowPtr, reference.row_ptr.iter().flat_map(|v| v.to_le_bytes()).collect()),
            (Region::Columns, reference.columns.iter().flat_map(|v| v.to_le_bytes()).collect()),
            (Region::Values, reference.values.iter().flat_map(|v| v.to_le_bytes()).collect())];
        for (region, expected) in expected {
            for owner in [&baseline, &packed] {
                let mut actual = vec![0; expected.len()];
                device.run(|| library.copy_d2h_quiescent(&mut actual, CuteafdDeviceBuffer {
                    ptr: owner.storage.buffer.ptr.wrapping_byte_add(layout.offset(region)),
                    bytes: expected.len(), ..owner.storage.buffer }))?;
                assert_eq!(actual, expected, "{region:?}");
            }
        }
        assert_eq!(packed.exceptions(), (3,1));
        assert_eq!(packed.status, baseline.status);
        let mut inventory = Inventory::new(CoordinatorDevices::new(&[1])?, "head",
            Lifetime { first: Phase::Load, last: Phase::Growth }, Category::Weights, "head");
        packed.inventory(&mut inventory, "vocab")?;
        assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_none()).count(),1);
        assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_some()).count(),1);
        Ok(())
    }

    #[test]
    fn arena_failed_pack_drain_retains_destination_and_module() -> Result<()> {
        use crate::shared::memory::{device::{Allocation, Device}, weight_arena::DeviceWeightArena};
        use cuteafd_loader::serving_capacity::WeightArenaLayout;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.stage(&library)?;
        fixture.configure_pack(&library, 1, 1)?;
        let device = Device { library: &library, id: 0 };
        let source = Allocation::new(device, 16*128*2, crate::shared::memory::TEST)?;
        let arena = DeviceWeightArena::new(device, WeightArenaLayout::new("test/head/quarantine",
            [("packed12".into(), PackedHeadLayout::new(16,128)?.total_bytes())])?, crate::shared::memory::TEST_ARENA)?;
        let packed = PackedHead::pack_with(&library, source.buffer.ptr,16,128,std::ptr::null_mut(),
            |name, bytes| arena.view(name, bytes));
        assert!(packed.is_err());
        drop(packed); drop(arena);
        std::mem::forget(source);
        drop(library);
        assert!(fixture.resident()?);
        assert!(!fixture.events()?.contains('d'), "quarantined backing must not free");
        Ok(())
    }

    /// The planes array is the layout's first six regions, in the native order, and the
    /// status words follow the table.
    #[test]
    fn planes_follow_the_layout_regions() {
        let layout = PackedHeadLayout::new(64_640, 5120).unwrap();
        let storage = CuteafdDeviceBuffer { ptr: 0x1000_0000usize as *mut c_void, bytes: layout.total_bytes(),
            ..Default::default() };
        let planes = planes(storage, &layout);
        let offsets = planes.map(|p| p as usize - 0x1000_0000);
        assert_eq!(offsets, [Region::Coarse, Region::Fine, Region::Base, Region::RowPtr, Region::Columns,
            Region::Values].map(|region| layout.offset(region)));
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 64_640 * 5120);
        let status = status_buffer(storage, &layout);
        assert_eq!(status.ptr as usize - 0x1000_0000, layout.offset(Region::Status));
        assert_eq!(status.bytes, 32);
        assert!(layout.offset(Region::Status) + 32 <= layout.total_bytes());
    }
}
