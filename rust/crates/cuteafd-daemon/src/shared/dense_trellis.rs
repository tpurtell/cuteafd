//! Shared native K6 resident weights and serialized-lane projection owners.
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_loader::{LoadedTensor, placement::Demand, formats::{
    dense_trellis::{LoadedDenseTrellis, TrellisSide}, exl3_storage::exl3_module,
    trellis_core::ProjectionScratchLayout}};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, programs::Programs, Bf16F32Add, F32Bf16Narrow,
    dense_trellis::{DenseTrellisBuffers, DenseTrellisDispatch, DenseTrellisPolicy,
        TrellisDType, GroupedWorkspace, GROUPS}};
use std::{collections::BTreeMap, ffi::c_void};
use crate::shared::memory::Tag;
use cuteafd_core::memory_layout::ArenaTag;

/// Native bytes remain compressed on device; admission precedes allocation.
/// The caller selects ownership from the shared layer placement before packing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CoreWeightLayout {
    pub stem: String, pub side: cuteafd_loader::formats::dense_trellis::TrellisSide,
    pub rank: usize, pub ranks: usize, pub shape: [usize;2], pub bytes: usize, pub bits: usize,
    pub codebook: cuteafd_loader::formats::exl3_storage::Exl3Codebook,
}
impl CoreWeightLayout {
    pub(crate) fn from_shared(layout: cuteafd_loader::formats::trellis_core::CoreWeightLayout) -> Self {
        Self { stem: layout.stem, side: layout.side, rank: layout.rank, ranks: layout.ranks,
            shape: layout.shape, bytes: layout.bytes, bits: layout.bits, codebook: layout.codebook }
    }
    pub(crate) fn new(projection: &cuteafd_loader::formats::trellis_core::CoreProjection,
        side: TrellisSide, rank: usize, ranks: usize) -> Result<Self> {
        Ok(Self::from_shared(cuteafd_loader::formats::trellis_core::CoreWeightLayout::new(
            projection, side, rank, ranks)?))
    }
    pub(crate) fn check(&self, loaded: &LoadedDenseTrellis) -> Result<()> {
        let shard = &loaded.shard;
        ensure!(loaded.codebook == self.codebook && shard.side == self.side && shard.shape == self.shape
            && shard.payload.len() == self.shape[0].checked_mul(self.shape[1]).and_then(|n|n.checked_mul(self.bits))
                .context("native trellis payload extent overflow")? / 8
            && shard.suh.len() == self.shape[1] * 2 && shard.svh.len() == self.shape[0] * 2,
            "native K6 packed shard differs from admitted operands");
        Ok(())
    }

    pub(crate) fn read(&self, mut read: impl FnMut(&str) -> Result<LoadedTensor>) -> Result<LoadedDenseTrellis> {
        let mut tensors = BTreeMap::new();
        for suffix in ["trellis", "suh", "svh", self.codebook.marker()] {
            let tensor = read(&format!("{}.{}", self.stem, suffix))?;
            tensors.insert(suffix, tensor);
        }
        let metadata = tensors.iter().map(|(&suffix, tensor)| (suffix, cuteafd_loader::SafetensorsTensorMetadata {
            name: tensor.info.name.clone(), dtype: tensor.info.dtype.clone(), shape: tensor.info.shape.clone(),
            byte_offset: tensor.info.byte_offset, byte_length: tensor.info.byte_length,
        })).collect::<BTreeMap<_, _>>();
        let suffixes = tensors.keys().copied().collect::<Vec<_>>();
        let module = exl3_module(&self.stem, &suffixes, |suffix| metadata.get(suffix))?;
        let loaded = self.pack(&module, |name| {
            let suffix = name.strip_prefix(&format!("{}.", self.stem)).context("native K6 source stem")?;
            tensors.remove(suffix).context("native K6 source member already consumed")
        })?;
        self.check(&loaded)?;
        Ok(loaded)
    }

    pub(crate) fn shared_layout(&self) -> cuteafd_loader::formats::trellis_core::CoreWeightLayout {
        cuteafd_loader::formats::trellis_core::CoreWeightLayout {
            stem: self.stem.clone(), side: self.side, rank: self.rank, ranks: self.ranks,
            shape: self.shape, bytes: self.bytes, bits: self.bits, codebook: self.codebook,
        }
    }

    /// Nonadopted families retain standalone requests until their arena migration.
    pub(crate) fn load_recorded<'a>(&self, device: Device<'a>, checkpoint: &cuteafd_loader::plan::Checkpoint,
        dense: &crate::shared::experts::rtx::dense::DenseLoad<'a>, stream: *mut c_void) -> Result<CoreWeights<'a>> {
        self.load_recorded_with(device, checkpoint, dense, stream,
            |name, bytes| crate::shared::memory::weight_arena::weight_allocation(device, None, name, bytes, ArenaTag::weights("shared/k6-core")))
    }

    fn output_start(&self, total: usize) -> Result<usize> {
        let local = self.shape[0];
        if local.checked_mul(self.ranks) == Some(total) {
            return Ok(self.rank * local);
        }
        ensure!(self.ranks == 2, "uneven output shard needs two ranks");
        let span = cuteafd_loader::shared::vocabulary::shard(total, 128, self.rank)?;
        ensure!(span.len() == local, "uneven output shard differs from admitted width");
        Ok(span.start)
    }

    /// The shared mount/push recipe; only final named storage is injected.
    pub(crate) fn load_recorded_with<'a>(&self, device: Device<'a>, checkpoint: &cuteafd_loader::plan::Checkpoint,
        dense: &crate::shared::experts::rtx::dense::DenseLoad<'a>, stream: *mut c_void,
        mut allocate: impl FnMut(&str, usize) -> Result<Allocation<'a>>) -> Result<CoreWeights<'a>> {
        use crate::shared::experts::rtx::dense::{Job, Operand, Rect, Step};
        let metadata = |name: &str| -> Result<&cuteafd_loader::SafetensorsTensorMetadata> {
            let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("checkpoint lacks {name}"))?;
            Ok(&checkpoint.tensors[at].meta)
        };
        let stem = &self.stem;
        let suffixes = ["trellis", "suh", "svh", self.codebook.marker()];
        let metadata = suffixes.iter().map(|&s| Ok((s, metadata(&format!("{stem}.{s}"))?.clone())))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let module = exl3_module(stem, &suffixes, |s| metadata.get(s))?;
        ensure!(module.codebook == self.codebook && module.bits == self.bits, "{stem}: trellis codebook/bits differ from admission");
        // The marker is four bytes: checked on the host while planning.
        let marker = crate::shared::checkpoint_read::read_tensor(checkpoint, &format!("{stem}.{}", self.codebook.marker()))?;
        ensure!(marker.bytes.as_slice() == self.codebook.multiplier().to_le_bytes(),
            "{stem}: unsupported {} marker payload", self.codebook.marker());
        let (n, k) = (module.output_features, module.input_features);
        let [local_n, local_k] = self.shape;
        let (n0, k0) = match self.side {
            TrellisSide::Output => (self.output_start(n)?, 0),
            TrellisSide::Input => (0, self.rank * local_k),
        };
        ensure!(match self.side { TrellisSide::Output => local_k == k,
            TrellisSide::Input => local_n == n && local_k * self.ranks == k }
            && local_n % 128 == 0 && local_k % 128 == 0, "{stem}: K6 shard geometry");
        let tile_bytes = 32 * module.bits;
        let payload = allocate(&format!("{stem}.trellis"), local_n * local_k * module.bits / 8)?;
        let k16 = local_k / 16;
        for (kr, source) in self.payload_rects(checkpoint, n)? {
            let rows = source.rows.len();
            dense.step(Step { device: device.id, rects: vec![source], jobs: vec![Job::TrellisTiles { rows, k0: kr, tiles: local_n / 16, tile_bytes,
                k16, input: self.side == TrellisSide::Input, dst: payload.buffer }] }, stream)?;
        }
        let mut vector = |suffix: &str, start: usize, count: usize| -> Result<Allocation<'a>> {
            let name = format!("{stem}.{suffix}");
            let r = Rect::tensor(checkpoint, &name, None, Some(start * 2..(start + count) * 2))?;
            let bytes = r.bytes();
            let allocation = allocate(&name, bytes)?;
            dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands: vec![(r, 0)], bytes })?;
            Ok(allocation)
        };
        let suh = vector("suh", k0, local_k)?;
        let svh = vector("svh", n0, local_n)?;
        ensure!(payload.buffer.bytes + suh.buffer.bytes + svh.buffer.bytes == self.bytes,
            "{stem}: K6 shard differs from admitted bytes");
        Ok(CoreWeights { payload, suh, svh, shape: self.shape, side: self.side, bits: self.bits, codebook: self.codebook })
    }

    /// Header-only source windows shared by recorded loading and admission tests.
    pub(crate) fn payload_rects(&self, checkpoint: &cuteafd_loader::plan::Checkpoint, outputs: usize)
        -> Result<Vec<(usize, crate::shared::experts::rtx::dense::Rect)>> {
        use crate::shared::experts::rtx::dense::{Rect, TRANSFORM_BYTES};
        let [local_n, local_k] = self.shape;
        let (n0, k0) = match self.side {
            TrellisSide::Output => (self.output_start(outputs)?, 0),
            TrellisSide::Input => (0, self.rank * local_k),
        };
        let tile_bytes = 32 * self.bits;
        let trellis = format!("{}.trellis", self.stem);
        // Checkpoint [K16][N16][tile]: each shard is a column window of K16 rows.
        let cols = (n0 / 16) * tile_bytes..(n0 + local_n) / 16 * tile_bytes;
        let per = (TRANSFORM_BYTES / cols.len()).max(1);
        (0..local_k / 16).step_by(per).map(|kr| {
            let rows = per.min(local_k / 16 - kr);
            let source = Rect::tensor(checkpoint, &trellis,
                Some(k0 / 16 + kr..k0 / 16 + kr + rows), Some(cols.clone()))?;
            ensure!(source.row_bytes == (outputs / 16) * tile_bytes, "{trellis}: trellis row extent");
            Ok((kr, source))
        }).collect()
    }

    pub(crate) fn demand(&self, device: u8) -> Demand {
        Demand::new(device,Category::Weights,format!("native K6 {}",self.stem),self.bytes as u64,Basis::Exact)
    }
    pub(crate) fn pack(&self,module:&cuteafd_loader::formats::exl3_storage::Exl3Module,
        read:impl FnMut(&str)->Result<LoadedTensor>) -> Result<cuteafd_loader::formats::dense_trellis::LoadedDenseTrellis> {
        let loaded=cuteafd_loader::formats::dense_trellis::load_native_shard(&self.stem,module,self.side,self.rank,self.ranks,read)?;
        ensure!(loaded.codebook==self.codebook && loaded.shard.shape==self.shape
            && loaded.shard.payload.len()+loaded.shard.suh.len()+loaded.shard.svh.len()==self.bytes,
            "native K6 packed shard differs from admitted geometry/bytes");
        self.check(&loaded)?;
        Ok(loaded)
    }
    pub(crate) fn upload<'a>(&self,device:Device<'a>,loaded:cuteafd_loader::formats::dense_trellis::LoadedDenseTrellis)
        -> Result<CoreWeights<'a>> {
        self.upload_with(device, loaded, |_, bytes| Allocation::new(device, bytes, Tag::weights("shared/k6-core")))
    }
    pub(crate) fn upload_with<'a>(&self,device:Device<'a>,loaded:LoadedDenseTrellis,
        mut allocate:impl FnMut(&str,usize)->Result<Allocation<'a>>) -> Result<CoreWeights<'a>> {
        self.check(&loaded)?;
        let shard=loaded.shard;
        ensure!(loaded.codebook==self.codebook && shard.side==self.side && shard.shape==self.shape
            && shard.payload.len()+shard.suh.len()+shard.svh.len()==self.bytes,
            "native K6 upload differs from admitted shard");
        let mut upload=|suffix:&str,bytes:&[u8]|->Result<Allocation<'a>> {
            let _scope=cuteafd_ffi::memory_ledger::scope("weights/native K6 core");
            let owner=crate::shared::memory::weight_arena::checked_weight_allocation(
                device,&format!("{}.{suffix}",self.stem),bytes.len(),&mut allocate)?;
            device.run(||device.library.copy_h2d(owner.buffer,bytes))?;
            Ok(owner)
        };
        Ok(CoreWeights{payload:upload("trellis",&shard.payload)?,suh:upload("suh",&shard.suh)?,svh:upload("svh",&shard.svh)?,
            shape:self.shape,side:self.side,bits:self.bits,codebook:self.codebook})
    }
}
pub(crate) struct CoreWeights<'a> {
    pub payload:Allocation<'a>, pub suh:Allocation<'a>, pub svh:Allocation<'a>,
    pub shape:[usize;2], pub side:cuteafd_loader::formats::dense_trellis::TrellisSide, pub bits:usize,
    pub codebook:cuteafd_loader::formats::exl3_storage::Exl3Codebook,
}


/// All mutable projection storage is private to a serialized execution lane.
/// Weight storage is separate and may be shared by lanes on the same device.
#[derive(Debug,Clone,PartialEq,Eq)]
pub(crate) struct ProjectionLayout { pub(crate) extents:BTreeMap<&'static str,usize>, pub(crate) rows:usize }
impl ProjectionLayout {
    pub(crate) fn merge(&mut self,other:&Self)->Result<()> {
        let mut layout=ProjectionScratchLayout {
            extents:self.extents.clone(), rows:self.rows };
        layout.merge(&ProjectionScratchLayout {
            extents:other.extents.clone(),rows:other.rows })?;
        self.extents=layout.extents;Ok(())
    }
    pub(crate) fn bytes(&self)->Result<usize> {
        ProjectionScratchLayout {
            extents:self.extents.clone(),rows:self.rows }.bytes()
    }
    pub(crate) fn new(program:&DenseTrellisDispatch<'_>,rows:usize)->Result<Self> {
        let p=program.policy();p.validate_rows(rows)?;
        Self::with_workspace(p.input,p.output,rows,program.workspace())
    }
    fn with_workspace(input:usize,output:usize,rows:usize,w:cuteafd_ffi::dense_trellis::DenseTrellisWorkspace)->Result<Self> {
        let layout=ProjectionScratchLayout::new(input,output,rows)?;
        ensure!([("rotated",w.rotated),("table",w.pointer_table),("scales",w.block_scales),
            ("partial",w.partial),("fused",w.fused),("accumulator",w.accumulator)].iter()
            .all(|(name,bytes)|layout.extents.get(name)==Some(bytes)),
            "K6 package/planner scratch mismatch");
        Ok(Self{extents:layout.extents,rows:layout.rows})
    }
    pub(crate) fn demand(&self,device:u8,name:&str)->Result<Demand> {
        Ok(Demand::new(device,Category::Workspace,name,self.bytes()? as u64,Basis::Exact))
    }
    pub(crate) fn allocate_packed<'a>(&self, device: Device<'a>, tag: Tag) -> Result<ProjectionScratch<'a>> {
        let layout = ProjectionScratchLayout { extents: self.extents.clone(), rows: self.rows };
        let storage = layout.storage()?;
        let packed = layout.packed()?;
        let backing = Allocation::new(device, packed.bytes, tag)?;
        let owner_regions: BTreeMap<&'static str, _> = storage.owners.keys().copied().zip(packed.regions).collect();
        let buffers = self.extents.iter().map(|(&name, &bytes)| {
            let (owner, offset) = if let Some(&offset) = storage.route_offsets.get(name) {
                ("route arena", offset)
            } else {
                (name, 0)
            };
            let region = &owner_regions[owner];
            Ok((name, crate::shared::prefix::view(backing.buffer, region.start + offset, bytes)?))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        if let Some(&table_bytes)=self.extents.get("table") {
            let pointer=if table_bytes>=GROUPS*8 {GroupedWorkspace::pointer_table(buffers["rotated"])?}
                else {(buffers["rotated"].ptr as u64).to_le_bytes().to_vec()};
            device.run(||device.library.copy_h2d(buffers["table"],&pointer))?;
        }
        Ok(ProjectionScratch { _owners: BTreeMap::from([("projection arena", backing)]), buffers, rows: self.rows })
    }
    pub(crate) fn allocate<'a>(&self,device:Device<'a>,tag:Tag)->Result<ProjectionScratch<'a>> {
        let _scope = cuteafd_ffi::memory_ledger::scope("workspace/native-projection");
        let storage=ProjectionScratchLayout {
            extents:self.extents.clone(),rows:self.rows }.storage()?;
        let mut owners=BTreeMap::new();
        for (&name,&bytes) in &storage.owners {owners.insert(name,Allocation::new(device,bytes,tag)?);}
        let mut buffers=BTreeMap::new();
        for (&name,&bytes) in &self.extents {
            let buffer=if let Some(&offset)=storage.route_offsets.get(name) {
                crate::shared::prefix::view(owners["route arena"].buffer,offset,bytes)?
            } else {owners[name].buffer};
            buffers.insert(name,buffer);
        }
        // Entry 0 is the standalone rotated input; a grouped launch reads one
        // fixed quarter per member, and quarter 0 starts at that same address.
        let pointer=if self.extents["table"]>=GROUPS*8 {GroupedWorkspace::pointer_table(buffers["rotated"])?}
            else {(buffers["rotated"].ptr as u64).to_le_bytes().to_vec()};
        device.run(||device.library.copy_h2d(buffers["table"],&pointer))?;
        Ok(ProjectionScratch{_owners:owners,buffers,rows:self.rows})
    }
}
pub(crate) struct ProjectionScratch<'a> {
    pub(crate) _owners:BTreeMap<&'static str,Allocation<'a>>,
    pub(crate) buffers:BTreeMap<&'static str,Buffer>,pub(crate) rows:usize,
}
impl ProjectionScratch<'_> {
    pub(crate) fn allocation_receipts(&self,
        inventory: &mut crate::shared::memory::receipts::Inventory<'_>, name: &str) -> Result<()> {
        for (&key, owner) in &self._owners {
            inventory.device(&format!("{name}.storage.{}", key.replace(' ', "_")), owner.buffer)?;
        }
        for (&key, &buffer) in &self.buffers {
            let start = buffer.ptr as usize;
            let end = start.checked_add(buffer.bytes).context("projection view overflow")?;
            let mut owners = self._owners.iter().filter(|(_, owner)| {
                buffer.device_id == owner.buffer.device_id && start >= owner.buffer.ptr as usize
                    && (owner.buffer.ptr as usize).checked_add(owner.buffer.bytes)
                        .is_some_and(|limit| end <= limit)
            });
            let (&backing, owner) = owners.next().context("projection view without backing")?;
            ensure!(owners.next().is_none(), "ambiguous projection backing");
            inventory.view(&format!("{name}.view.{key}"), buffer,
                &format!("{name}.storage.{}", backing.replace(' ', "_")), owner.buffer)?;
        }
        Ok(())
    }
    fn buffers(&self,input:Buffer,weights:&CoreWeights<'_>)->DenseTrellisBuffers {
        let get=|name|self.buffers[name];
        DenseTrellisBuffers {input,payload:weights.payload.buffer,suh:weights.suh.buffer,svh:weights.svh.buffer,
            rotated:get("rotated"),pointer_table:get("table"),scales:get("scales"),partial:get("partial"),
            fused:self.buffers.get("fused").copied().unwrap_or_else(||get("partial")),
            accumulator:get("accumulator")}
    }
}
/// Replace the producer's retained copy with one exact RNE narrowing launch.
/// Callers serialize the source accumulator and retain destination through drain.
pub(crate) fn retain_partial(narrow: &F32Bf16Narrow<'_>, source: Buffer, destination: Buffer,
    count: usize, stream: *mut c_void) -> Result<()> {
    validate_retained_partial(source, destination, count)?;
    // SAFETY: checked distinct typed extents and device; the serialized lane
    // orders the producer and retains both owners until this stream drains.
    unsafe { narrow.launch(source.ptr, destination.ptr, count, stream) }
}
fn validate_retained_partial(source: Buffer, destination: Buffer, count: usize) -> Result<()> {
    let input = count.checked_mul(4).filter(|&n| n > 0).context("K6 partial F32 extent overflow")?;
    let output = count.checked_mul(2).context("K6 partial BF16 extent overflow")?;
    ensure!(count <= (i64::MAX - 255) as usize, "K6 partial native extent overflow");
    ensure!(source.device_id >= 0 && destination.device_id == source.device_id,
        "K6 retained partial device mismatch");
    for (buffer, bytes, alignment) in [(source, input, 4), (destination, output, 2)] {
        ensure!(buffer.flags == 0 && !buffer.ptr.is_null() && buffer.ptr as usize % alignment == 0
            && buffer.bytes >= bytes && (buffer.ptr as usize).checked_add(bytes).is_some(),
            "K6 retained partial extent/alignment");
    }
    let src = source.ptr as usize;
    let dst = destination.ptr as usize;
    ensure!(src + input <= dst || dst + output <= src, "K6 retained partial aliases accumulator");
    Ok(())
}

#[cfg(test)]
mod retained_tests {
    use super::*;

    #[test]
    fn projection_census_uses_actual_backing_without_native_work() -> Result<()> {
        use crate::shared::memory::receipts::Inventory;
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, Location};
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let allocation = Allocation::new(Device { library: &library, id: 1 }, 1024, crate::shared::memory::TEST)?;
        let owner = allocation.buffer;
        let view = Buffer { ptr: (owner.ptr as usize + 128) as *mut c_void, bytes: 256, ..owner };
        let mut scratch = ProjectionScratch { _owners: BTreeMap::from([("workspace", allocation)]),
            buffers: BTreeMap::from([("partial", view)]), rows: 8 };
        library.cuda_set_device(0)?;
        let before = fixture.events()?;
        let physical = [1, 0];
        let devices = CoordinatorDevices::new(&physical)?;
        let mut inventory = Inventory::new(devices, "fixture", Lifetime::TARGET, Category::Workspace, "projection");
        scratch.allocation_receipts(&mut inventory, "core")?;
        assert_eq!(inventory.receipts.len(), 2);
        let backing = &inventory.receipts[0];
        assert_eq!((backing.address, backing.allocation.bytes, backing.allocation.location),
            (owner.ptr as u64, owner.bytes as u64, Location::coordinator(0)));
        let part = &inventory.receipts[1];
        assert_eq!((part.address, part.allocation.bytes), (view.ptr as u64, view.bytes as u64));
        let alias = part.allocation.view.as_ref().unwrap();
        assert_eq!((&alias.backing, alias.offset), (&backing.allocation.id, 128));
        for bad in [Buffer { bytes: 1024, ..view }, Buffer { device_id: 0, ..view }] {
            scratch.buffers.insert("partial", bad);
            let mut inventory = Inventory::new(devices, "invalid", Lifetime::TARGET, Category::Workspace, "projection");
            assert!(scratch.allocation_receipts(&mut inventory, "core").is_err());
        }
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(fixture.events()?, before, "census must not allocate, switch, synchronize or reset");
        Ok(())
    }

    #[test]
    fn packed_projection_retains_one_backing_and_initializes_pointer_views() -> Result<()> {
        use crate::shared::memory::receipts::Inventory;
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime};
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        for layout in [ProjectionScratchLayout::new(128, 256, 3)?,
            ProjectionScratchLayout::grouped(128, 256, 3, 4)?] {
            let projection = ProjectionLayout { extents: layout.extents.clone(), rows: layout.rows };
            let before = fixture.events()?.matches('D').count();
            let scratch = projection.allocate_packed(device, Tag::workspace("step5/test-projection"))?;
            assert_eq!(fixture.events()?.matches('D').count() - before, 1);
            assert_eq!(scratch._owners.len(), 1);
            let storage = layout.storage()?;
            let packed = layout.packed()?;
            let backing = scratch._owners["projection arena"].buffer;
            assert_eq!(backing.bytes, layout.packed_bytes()?);
            let owner_regions: BTreeMap<&'static str, _> = storage.owners.keys().copied().zip(&packed.regions).collect();
            for (&name, &bytes) in &layout.extents {
                let (owner, offset) = storage.route_offsets.get(name).map_or((name, 0), |&o| ("route arena", o));
                let region = owner_regions[owner];
                let view = scratch.buffers[name];
                assert_eq!(view.ptr as usize - backing.ptr as usize, region.start + offset);
                assert_eq!(view.bytes, bytes);
                assert_eq!(view.ptr as usize % 256, 0);
                assert!(region.end <= backing.bytes);
            }
            assert!(packed.regions.windows(2).all(|r| r[0].end <= r[1].start));
            let expected = if layout.extents["table"] >= GROUPS * 8 {
                GroupedWorkspace::pointer_table(scratch.buffers["rotated"])?
            } else { (scratch.buffers["rotated"].ptr as u64).to_le_bytes().to_vec() };
            let mut actual = vec![0; expected.len()];
            device.run(|| library.copy_d2h(&mut actual, scratch.buffers["table"]))?;
            assert_eq!(actual, expected);
            let mut inventory = Inventory::new(CoordinatorDevices::new(&[0])?, "packed",
                Lifetime::TARGET, Category::Workspace, "projection");
            scratch.allocation_receipts(&mut inventory, "core")?;
            assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_none()).count(), 1);
            assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_some()).count(), layout.extents.len());
            // Families that have not opted in still retain the original five owners.
            assert_eq!(projection.allocate(device, Tag::workspace("step5/test-projection"))?._owners.len(), 5);
        }
        Ok(())
    }

    fn draft_head_capacities(input:usize,output:usize,compiled_rows:usize)->Result<()> {
        use cuteafd_ffi::dense_trellis::DenseTrellisGeometry;
        let geometry=|rows| DenseTrellisGeometry { input,output,bits:6,side:1,rows,
            input_dtype:TrellisDType::Bf16,output_dtype:TrellisDType::F32 };
        let target=geometry(64).workspace()?;
        for rows in [8,16,64] {
            ensure!(rows<=compiled_rows,"sealed package cannot serve draft head rows");
            let own=geometry(rows).workspace()?;
            let layout=ProjectionLayout::with_workspace(input,output,rows,own)?;
            assert_eq!(layout.extents,ProjectionScratchLayout::new(input,output,rows)?.extents);
            if rows!=64 {
                assert!(ProjectionLayout::with_workspace(input,output,rows,target).is_err());
                assert!(layout.bytes()?<ProjectionScratchLayout::new(input,output,64)?.bytes()?);
            }
        }
        Ok(())
    }

    #[test]
    fn draft_head_program_capacity_matches_its_private_scratch()->Result<()> {
        // Kimi sealed PROGRAMS.json: N163840/K7168, package capacity 4096.
        draft_head_capacities(7168,163840,4096)
    }

    #[test]
    #[ignore = "requires CUTEAFD_PROGRAM_MANIFEST pointing to the sealed Kimi package; CPU only"]
    fn sealed_draft_head_program_capacity_matches_its_private_scratch()->Result<()> {
        let manifest:serde_json::Value=serde_json::from_slice(&std::fs::read(
            std::env::var_os("CUTEAFD_PROGRAM_MANIFEST").context("sealed manifest path")?)?)?;
        let policies=manifest["families"]["dense_trellis_k6"]["row_policies"].as_array().context("row policies")?;
        for policy in policies.iter().filter(|p|p["shape"]=="lm_head") {
            let name=format!("dense_trellis_k6_{}",policy["native_program"].as_str().context("head program")?);
            let program=manifest["programs"].as_array().context("programs")?.iter()
                .find(|p|p["name"]==name).context("sealed head program missing")?;
            let input=policy["input_features"].as_u64().context("head input")? as usize;
            let output=policy["output_features"].as_u64().context("head output")? as usize;
            let capacity=program["capacity_rows"].as_u64().context("head capacity")? as usize;
            assert_eq!(program["scratch_bytes_at_capacity"]["partial"].as_u64(),
                Some((capacity*input.div_ceil(1024)*output*4) as u64));
            draft_head_capacities(input,output,capacity)?;
        }
        ensure!(policies.iter().any(|p|p["shape"]=="lm_head"),"sealed package has no head");
        Ok(())
    }

    #[test]
    fn narrowing_requires_distinct_full_typed_storage() {
        let source = Buffer { ptr: 0x1000usize as *mut c_void, bytes: 64, device_id: 0, flags: 0 };
        let destination = Buffer { ptr: 0x2000usize as *mut c_void, bytes: 32, ..source };
        validate_retained_partial(source, destination, 16).unwrap();
        for count in [0, 17, usize::MAX] { assert!(validate_retained_partial(source, destination, count).is_err()); }
        for bad in [Buffer { bytes: 31, ..destination }, Buffer { device_id: 1, ..destination },
            Buffer { ptr: 0x103eusize as *mut c_void, ..destination }, Buffer { flags: 1, ..destination }] {
            assert!(validate_retained_partial(source, bad, 16).is_err());
        }
    }
}

pub(crate) struct Projection<'a> {pub program:DenseTrellisDispatch<'a>}
impl<'a> Projection<'a> {
    pub(crate) fn load(programs:&'a Programs<'a>,shape:&str,book:&str,rows:usize,output:TrellisDType)->Result<Self> {
        let policy=DenseTrellisPolicy::from_package(programs,shape,book)?;
        Ok(Self{program:DenseTrellisDispatch::load(programs,policy,rows,output)?})
    }
    pub(crate) fn load_geometry(programs:&'a Programs<'a>,input:usize,output:usize,side:usize,book:&str,
        rows:usize,dtype:TrellisDType)->Result<Self> {
        Self::load_geometry_bits(programs,input,output,side,6,book,rows,dtype)
    }
    /// The width-generic form: Step-5 MTP stages store K4 beside a K6 target core.
    pub(crate) fn load_geometry_bits(programs:&'a Programs<'a>,input:usize,output:usize,side:usize,bits:usize,
        book:&str,rows:usize,dtype:TrellisDType)->Result<Self> {
        let policy=DenseTrellisPolicy::from_package_geometry_bits(programs,input,output,side,bits,book)?;
        Ok(Self{program:DenseTrellisDispatch::load(programs,policy,rows,dtype)?})
    }
    pub(crate) fn core(&self,rows:usize,input:Buffer,weights:&CoreWeights<'_>,scratch:&ProjectionScratch<'_>,
        stream:*mut c_void)->Result<Buffer> {
        ensure!(rows>0 && rows<=scratch.rows,"K6 scratch live-row ceiling");
        let policy=self.program.policy();
        let side=match weights.side {cuteafd_loader::formats::dense_trellis::TrellisSide::Output=>1,
            cuteafd_loader::formats::dense_trellis::TrellisSide::Input=>2};
        let book=match weights.codebook {cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg=>"mcg",
            cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1=>"mul1"};
        ensure!(weights.shape==[policy.output,policy.input] && side==policy.side && book==policy.codebook
            && weights.bits==policy.bits,
            "trellis projection and resident weight geometry/width/codebook differ");
        let b=scratch.buffers(input,weights);self.program.core(rows,b,stream)?;Ok(b.accumulator)
    }
    /// The F32 core written straight into a caller-owned plane (a K-split
    /// partial that the caller exchanges and reduces before one egress).
    pub(crate) fn core_into(&self,rows:usize,input:Buffer,weights:&CoreWeights<'_>,scratch:&ProjectionScratch<'_>,
        accumulator:Buffer,stream:*mut c_void)->Result<()> {
        ensure!(rows>0 && rows<=scratch.rows,"trellis scratch live-row ceiling");
        let policy=self.program.policy();
        let side=match weights.side {cuteafd_loader::formats::dense_trellis::TrellisSide::Output=>1,
            cuteafd_loader::formats::dense_trellis::TrellisSide::Input=>2};
        ensure!(weights.shape==[policy.output,policy.input] && side==policy.side
            && codebook(weights)==policy.codebook && weights.bits==policy.bits
            && accumulator.bytes>=rows*policy.output*4,"trellis F32 core destination geometry/width/codebook");
        let b=DenseTrellisBuffers{accumulator,..scratch.buffers(input,weights)};
        self.program.core(rows,b,stream)?;Ok(())
    }
    pub(crate) fn local(&self,rows:usize,input:Buffer,weights:&CoreWeights<'_>,scratch:&ProjectionScratch<'_>,
        output:Buffer,stream:*mut c_void)->Result<()> {
        ensure!(rows>0 && rows<=scratch.rows,"K6 scratch live-row ceiling");
        let policy=self.program.policy();
        let side=match weights.side {cuteafd_loader::formats::dense_trellis::TrellisSide::Output=>1,
            cuteafd_loader::formats::dense_trellis::TrellisSide::Input=>2};
        ensure!(policy.side!=2,"K-split output requires F32 peer reduction before egress");
        ensure!(weights.shape==[policy.output,policy.input] && side==policy.side
            && codebook(weights)==policy.codebook && weights.bits==policy.bits,
            "trellis projection and resident weight geometry/width/codebook differ");
        let b=scratch.buffers(input,weights);
        self.program.local(rows,b,output,stream)?;Ok(())
    }
    pub(crate) fn reduced_f32(&self,add:&cuteafd_ffi::F32Add<'_>,rows:usize,rank0:Buffer,rank1:Buffer,reduced:Buffer,
        weights:&CoreWeights<'_>,output:Buffer,stream:*mut c_void)->Result<()> {
        self.program.reduce_f32_output(add,rows,rank0,rank1,reduced,weights.svh.buffer,output,stream)?;Ok(())
    }
    pub(crate) fn reduced(&self,add:&Bf16F32Add<'_>,rows:usize,rank0:Buffer,rank1:Buffer,reduced:Buffer,
        weights:&CoreWeights<'_>,output:Buffer,stream:*mut c_void)->Result<()> {
        self.program.reduce_output(add,rows,rank0,rank1,reduced,weights.svh.buffer,output,stream)?;Ok(())
    }
}


/// Target and drafter use independent scratch even though they share the head weights.
pub(crate) struct CoreHead<'a> {
    projection: Projection<'a>, scratch: ProjectionScratch<'a>, device: Device<'a>,
    stream: *mut c_void, rows: usize,
}
impl<'a> CoreHead<'a> {
    pub(crate) fn allocation_scratch(&self) -> &ProjectionScratch<'a> { &self.scratch }

    pub(crate) fn new(programs: &'a Programs<'a>, weights: &CoreWeights<'a>, rows: usize,
        device: Device<'a>, stream: *mut c_void,
        admitted: &cuteafd_loader::formats::trellis_core::ProjectionScratchLayout, tag: Tag) -> Result<Self> {
        ensure!(weights.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Output,
            "GLM K6 head must be whole on rank 0");
        let projection = device.run(|| Projection::load_geometry_bits(programs, weights.shape[1], weights.shape[0],
            1, weights.bits, codebook(weights), rows, TrellisDType::F32))?;
        let layout = ProjectionLayout::new(&projection.program, rows)?;
        ensure!(admitted.rows == layout.rows && admitted.extents == layout.extents,
            "GLM K6 head scratch differs from admission");
        let scratch = layout.allocate(device, tag)?;
        Ok(Self { projection, scratch, device, stream, rows })
    }

    pub(crate) fn launch(&self, weights: &CoreWeights<'_>, input: Buffer, output: Buffer,
        rows: usize, stream: *mut c_void) -> Result<()> {
        head_launch(self.rows, self.stream, rows, stream, || {
            self.device.run(|| self.projection.local(rows, input, weights, &self.scratch, output, stream))
        })
    }

    pub(crate) fn target<'s>(&'s self, weights: &'s CoreWeights<'_>) -> crate::shared::speculation::dflash::TargetHead<'s> {
        crate::shared::speculation::dflash::TargetHead::Launch(Box::new(move |input, logits, rows, stream| {
            let buffer = |ptr, bytes| Buffer { ptr, bytes, device_id: self.device.id, flags: 0 };
            self.launch(weights, buffer(input.cast_mut(), rows * weights.shape[1] * 2),
                buffer(logits.cast(), rows * weights.shape[0] * 4), rows, stream)
        }))
    }
}

/// Target decode/prefill may enqueue before either finishes, but CUDA FIFO
/// serializes their shared scratch only on this one bound stream.
pub(crate) fn head_launch<T>(capacity: usize, bound: *mut c_void, rows: usize, stream: *mut c_void,
    enqueue: impl FnOnce() -> Result<T>) -> Result<T> {
    ensure!((1..=capacity).contains(&rows) && stream == bound,
        "GLM K6 head rows/stream differ from admitted scratch");
    enqueue()
}


pub(crate) fn codebook(weights: &CoreWeights<'_>) -> &'static str {
    match weights.codebook {
        cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg => "mcg",
        cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1 => "mul1",
    }
}

#[cfg(test)]
mod upload_tests {
    use super::*;
    #[test]
    fn k4_and_k6_uploads_preserve_each_named_plane() -> Result<()> {
        use crate::shared::memory::{weight_arena::DeviceWeightArena, receipts::Inventory};
        use cuteafd_loader::{formats::{dense_trellis::NativeTrellisShard, exl3_storage::Exl3Codebook},
            serving_capacity::WeightArenaLayout, placement::planner::{CoordinatorDevices, Lifetime, Phase}};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library,1,1<<30)?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library,id:1 };
        for bits in [4,6,8] {
            let layout = CoreWeightLayout { stem: format!("model.k{bits}"), side: TrellisSide::Output,
                rank:0,ranks:1,shape:[256,128],bytes:256*128*bits/8+2*(256+128),bits,codebook:Exl3Codebook::Mcg };
            let requests = layout.shared_layout().arena_tensors(&layout.stem)?;
            let loaded = || LoadedDenseTrellis { codebook: layout.codebook, shard: NativeTrellisShard {
                shape:layout.shape,side:layout.side,payload:(0..256*128*bits/8).map(|i| (i as u8).wrapping_mul(37)).collect(),
                suh:(0..256).map(|i| (i as u8).wrapping_add(7)).collect(),
                svh:(0..512).map(|i| (i as u8).wrapping_mul(17)).collect() } };
            let baseline = layout.upload(device,loaded())?;
            let arena = DeviceWeightArena::new(device,WeightArenaLayout::new(format!("test/k{bits}"),requests.clone())?, crate::shared::memory::TEST_ARENA)?;
            let mut requested = Vec::new();
            let actual = layout.upload_with(device,loaded(),|name,bytes| {
                requested.push((name.to_owned(),bytes)); arena.view(name,bytes)
            })?;
            assert_eq!(requested,requests);
            drop(arena);
            let mut inventory = Inventory::new(CoordinatorDevices::new(&[1])?,"core",
                Lifetime { first:Phase::Load,last:Phase::Growth },Category::Weights,"core");
            for (name,a,b) in [("trellis",&actual.payload,&baseline.payload),("suh",&actual.suh,&baseline.suh),
                ("svh",&actual.svh,&baseline.svh)] {
                a.inventory(&mut inventory,name)?;
                let mut got=vec![0;a.buffer.bytes]; let mut expected=vec![0;b.buffer.bytes];
                device.run(|| { library.copy_d2h(&mut got,a.buffer)?; library.copy_d2h(&mut expected,b.buffer) })?;
                assert_eq!(got,expected,"K{bits} {name}");
            }
            assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_none()).count(),1);
            assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_some()).count(),3);
        }
        Ok(())
    }

    #[test]
    fn fused_k8_routes_decode_rows_to_single_launch_and_prefill_to_three() -> Result<()> {
        use cuteafd_ffi::{dense_trellis::fused_scratch_bytes, native_library_lifetime_fixture::Fixture};
        use cuteafd_loader::formats::{dense_trellis::TrellisSide, exl3_storage::Exl3Codebook};
        use serde_json::json;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        let (k, n, cap) = (128usize, 256usize, 129usize);
        let mk_entries = |side: usize, out_dtype: &str, with_fused: bool| -> (serde_json::Value, Vec<serde_json::Value>, Vec<u8>) {
            let fused_odt = if side == 2 { "raw_f32" } else { out_dtype };
            let fused_ptrs: Vec<&str> = if side == 2 {
                vec!["x", "payload", "suh", "scratch", "output"]
            } else {
                vec!["x", "payload", "suh", "svh", "scratch", "output"]
            };
            let mut entries = vec![
                json!({"name":"dense_trellis_k6_in","capacity_rows":4096,
                    "pointers":[{"name":"input"},{"name":"suh"},{"name":"rotated"},{"name":"input_block_scales"}],
                    "scalars":[{"type":"int32"},{"type":"int32"}],"scratch_bytes_at_capacity":{},
                    "geometry":{"features":k,"input_dtype":"bf16","output_dtype":"bf16","max_rows":4096}}),
                json!({"name":"dense_trellis_k6_core","capacity_rows":4096,
                    "pointers":[{"name":"rotated_inputs"},{"name":"payload"},{"name":"input_block_scales"},{"name":"partial"},{"name":"output"}],
                    "scalars":[{"type":"int32"},{"type":"int32"},{"type":"int32"}],
                    "scratch_bytes_at_capacity":{"partial":4096 * k.div_ceil(1024) * n * 4},
                    "geometry":{"abi_version":1,"input_features":k,"output_features":n,"bits":8,"codebook":"mul1",
                        "sharded_side":side,"rotation_count":1,"max_rows":4096,"split_k_blocks":8,
                        "splits":k.div_ceil(1024),"input_dtype":"bf16","ctas_per_sm":4}}),
                json!({"name":"dense_trellis_k6_out","capacity_rows":4096,
                    "pointers":[{"name":"input"},{"name":"svh"},{"name":"output"}],
                    "scalars":[{"type":"int32"},{"type":"int32"}],"scratch_bytes_at_capacity":{},
                    "geometry":{"features":n,"output_dtype":out_dtype,"max_rows":4096}}),
            ];
            let mut tags = vec![b'I', b'C', b'O'];
            let mut policy = json!({"shape":"fixture","input_features":k,"output_features":n,"sharded_side":side,
                "bits":8,"codebook":"mul1","native_input":"in","native_program":"core","output_program":"out"});
            if with_fused {
                policy["fused_program"] = json!("fused");
                entries.push(json!({"name":"dense_trellis_k6_fused","capacity_rows":cap,
                    "pointers":fused_ptrs.iter().map(|p| json!({"name":p})).collect::<Vec<_>>(),
                    "scalars":[{"type":"int32"},{"type":"int32"}],
                    "scratch_bytes_at_capacity":{"scratch":fused_scratch_bytes(cap, k, n).unwrap()},
                    "geometry":{"abi_version":1,"fused_transform":true,"input_features":k,"output_features":n,
                        "bits":8,"codebook":"mul1","sharded_side":side,"max_rows":cap,"split_k_blocks":8,
                        "splits":k.div_ceil(1024),"input_dtype":"bf16","output_dtype":fused_odt,"ctas_per_sm":4,
                        "exact_row_cap":cap,"decode_route_thresholds":[]}}));
                tags.push(b'F');
            }
            let manifest = json!({"families":{"dense_trellis_k6":{"abi_version":1,"row_policies":[policy]}},"programs":entries});
            (manifest, entries, tags)
        };
        let device = Device { library: &library, id: 0 };
        let live = 160usize;
        let input = Allocation::new(device, live * k * 2, crate::shared::memory::TEST)?;
        device.run(|| library.copy_h2d(input.buffer, &7u32.to_ne_bytes()))?;
        let out_bf16 = Allocation::new(device, live * n * 2, crate::shared::memory::TEST)?;
        let out_f32 = Allocation::new(device, live * n * 4, crate::shared::memory::TEST)?;
        for (side, out_dtype, with_fused) in [(1, "bf16", true), (1, "f32", true), (2, "bf16", true), (1, "f32", false)] {
            let (manifest, entries, tags) = mk_entries(side, out_dtype, with_fused);
            fixture.register_programs(&library, &entries)?;
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("PROGRAMS.json");
            std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
            let programs = library.programs()?.with_manifest(&path)?;
            let dtype = if out_dtype == "f32" { TrellisDType::F32 } else { TrellisDType::Bf16 };
            let projection = Projection::load_geometry_bits(&programs, k, n, side, 8, "mul1", live, dtype)?;
            assert_eq!(projection.program.has_fused(), with_fused);
            let mut scratch = ProjectionLayout::new(&projection.program, live)?.allocate_packed(device, crate::shared::memory::TEST)?;
            device.run(|| library.copy_h2d(scratch.buffers["accumulator"], &7u32.to_ne_bytes()))?;
            let table_word = (scratch.buffers["rotated"].ptr as usize as u32).max(1);
            for (idx, &tag) in tags.iter().enumerate() {
                let ready = if tag == b'C' { table_word } else { 7 };
                fixture.register_reply(&library, idx as u32, ready, tag)?;
            }
            for buf in scratch.buffers.values_mut() { buf.flags = 0; }
            let mut weights = CoreWeights {
                payload: Allocation::new(device, k * n, crate::shared::memory::TEST)?,
                suh: Allocation::new(device, k * 2, crate::shared::memory::TEST)?,
                svh: Allocation::new(device, n * 2, crate::shared::memory::TEST)?,
                shape: [n, k],
                side: if side == 2 { TrellisSide::Input } else { TrellisSide::Output },
                bits: 8,
                codebook: Exl3Codebook::Mul1,
            };
            let saved_flags = (weights.payload.buffer.flags, weights.suh.buffer.flags, weights.svh.buffer.flags);
            weights.payload.buffer.flags = 0;
            weights.suh.buffer.flags = 0;
            weights.svh.buffer.flags = 0;
            let clear = |b: Buffer| Buffer { flags: 0, ..b };
            for (rows, expect) in [
                (1usize, if !with_fused { "ICO" } else { "F" }),
                (64, if !with_fused { "ICO" } else { "F" }),
                (129, if !with_fused { "ICO" } else { "F" }),
                (130, if side == 2 { "IC" } else { "ICO" }),
            ] {
                let before = fixture.events()?.len();
                if side == 2 {
                    projection.core_into(rows, clear(input.buffer), &weights, &scratch, clear(out_f32.buffer), std::ptr::null_mut())?;
                } else {
                    let dst = if dtype == TrellisDType::F32 { out_f32.buffer } else { out_bf16.buffer };
                    projection.local(rows, clear(input.buffer), &weights, &scratch, clear(dst), std::ptr::null_mut())?;
                }
                let got: String = fixture.events()?[before..].chars().filter(|c| matches!(c, 'I'|'C'|'O'|'F')).collect();
                assert_eq!(got, expect, "side={side} out={out_dtype} fused={with_fused} rows={rows}");
            }
            (weights.payload.buffer.flags, weights.suh.buffer.flags, weights.svh.buffer.flags) = saved_flags;
        }
        Ok(())
    }
}
