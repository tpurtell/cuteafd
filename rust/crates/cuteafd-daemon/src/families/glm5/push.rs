//! GLM 5.3 coordinator dense operands as push-group ranks: each GPU's class-2
//! operand allocations form one logical address space (with gaps, so slot
//! fragments never merge across operands), the plan lands the same rectangles
//! the mount reads, and a destination maps fragments onto the allocations.
use super::landing::Record;
use crate::shared::memory::{device::Device, push_consumer::PushDestination,
    weight_arena::{Coverage, ExpectedCoverage}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::{push_load::builder::{Builder, CheckedPlan, Destination, Source}, OfficialV41Catalog};

/// One GPU's class-2 operands in its rank's logical space.
pub(crate) struct RankLayout {
    pub device: i32,
    /// (logical start, operand buffer, landed bytes, landings), ascending.
    pub operands: Vec<(u64, CuteafdDeviceBuffer, usize, Vec<super::landing::Land>)>,
    pub bytes: u64,
}

impl RankLayout {
    /// Records on `device`, in loader order, 256-byte aligned with at least one
    /// gap byte between operands.
    pub fn new(records: &[Record], device: i32) -> Result<Self> {
        let mut operands = Vec::new();
        let mut cursor = 0u64;
        for record in records.iter().filter(|r| r.device == device) {
            ensure!(record.bytes > 0, "empty pushed operand");
            let start = cursor.next_multiple_of(256);
            operands.push((start, record.buffer, record.bytes, record.lands.clone()));
            cursor = start + record.bytes as u64 + 1;
        }
        Ok(Self { device, operands, bytes: cursor })
    }

    /// Plan entries for group rank `rank`.
    pub fn append(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, rank: u32) -> Result<()> {
        (self.plan().append)(builder, catalog, rank)
    }

    /// This rank's checked entries alone, in a group of `world` ranks.
    pub fn checked(&self, catalog: &OfficialV41Catalog, root: &std::path::Path, world: usize, rank: usize)
        -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        *arenas.get_mut(rank).context("coordinator rank outside group")? = self.bytes;
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, catalog, u32::try_from(rank)?)?;
        Ok(builder.finish()?)
    }
}

/// Fragments land straight in the loader's operand allocations. Done requires
/// exactly the planned footprint; nothing reads these operands before commit.
pub(crate) struct OperandArena<'a> {
    device: Device<'a>,
    layout: RankLayout,
    expected: ExpectedCoverage,
    coverage: Coverage,
}

impl<'a> OperandArena<'a> {
    pub fn new(device: Device<'a>, layout: RankLayout, expected: ExpectedCoverage) -> Result<Self> {
        ensure!(layout.device == device.id, "operand layout belongs to another GPU");
        Ok(Self { device, layout, expected, coverage: Coverage::default() })
    }

    fn operand(&self, offset: u64, bytes: u64) -> Result<(CuteafdDeviceBuffer, usize)> {
        let index = self.layout.operands.partition_point(|(start, ..)| *start <= offset).checked_sub(1)
            .context("fragment before the first operand")?;
        let (start, buffer, landed, _) = &self.layout.operands[index];
        let at = usize::try_from(offset - start)?;
        ensure!(at.checked_add(usize::try_from(bytes)?).is_some_and(|end| end <= *landed),
            "fragment crosses an operand boundary");
        Ok((*buffer, at))
    }
}

impl PushDestination for OperandArena<'_> {
    type Delivered = ();
    fn belongs_to(&self, device: Device<'_>) -> bool {
        self.device.id == device.id && std::ptr::eq(self.device.library, device.library)
    }
    fn logical_bytes(&self) -> usize { self.layout.bytes as usize }
    fn device_bytes(&self) -> usize { self.layout.bytes as usize }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(bytes > 0 && self.expected.contains(offset, bytes), "fragment outside checked footprint");
        self.operand(offset, bytes).map(|_| ())
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        let (buffer, at) = self.operand(offset as u64, bytes as u64)?;
        // SAFETY: `at + bytes` lies inside this live operand allocation (checked above).
        Ok(CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(at).cast() }, bytes, ..buffer })
    }
    fn copy_validation(&mut self, _: u64, _: &[u8]) -> Result<()> {
        anyhow::bail!("coordinator operands have no host tail")
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        ensure!(self.expected.contains(offset, bytes), "completed fragment is outside checked plan footprint");
        self.coverage.insert(offset, bytes)
    }
    fn done(self) -> Result<()> {
        ensure!(self.expected.complete(&self.coverage), "rank Done before all planned operand bytes completed");
        Ok(())
    }
}

impl RankLayout {
    /// The rank's entries as a device-free plan the group driver appends, and
    /// where the TP2 arena starts after these operands.
    pub fn plan(&self) -> crate::shared::experts::rtx::push::DensePlan {
        let operands: Vec<(u64, Vec<super::landing::Land>)> =
            self.operands.iter().map(|(start, _, _, lands)| (*start, lands.clone())).collect();
        let base = self.bytes.next_multiple_of(256);
        crate::shared::experts::rtx::push::DensePlan { base, append: Box::new(move |builder, catalog, rank| {
            for (start, lands) in &operands {
                for land in lands {
                    let tensor = catalog.tensor(&land.tensor)?;
                    let source = Source { path: catalog.snapshot().join(&tensor.shard),
                        offset: tensor.metadata.byte_offset, bytes: tensor.metadata.byte_length };
                    let (_, row_bytes) = super::landing::geometry(catalog, &land.tensor)?;
                    builder.rows(&source, (land.rows.start * row_bytes + land.cols.start) as u64,
                        u32::try_from(land.rows.len())?, u32::try_from(land.width())?, row_bytes as u64,
                        land.width() as u64, Destination { rank, offset: start + land.offset as u64,
                            bytes: land.bytes() as u64 })?;
                }
            }
            Ok(())
        }) }
    }
}

/// Fills recorded class-2 operands through the mount (the whole push group
/// fell back), on each operand's GPU.
pub(crate) fn fill_by_mount(library: &cuteafd_ffi::NativeLibrary, catalog: &OfficialV41Catalog,
    records: &[Record]) -> Result<()> {
    for record in records {
        let bytes = super::landing::read(catalog, &record.lands, record.bytes)?;
        Device { library, id: record.device }.run(|| {
            crate::shared::memory::load_timing::upload(bytes.len(), || library.copy_h2d(
                CuteafdDeviceBuffer { bytes: record.bytes, ..record.buffer }, &bytes))
        })?;
    }
    Ok(())
}
