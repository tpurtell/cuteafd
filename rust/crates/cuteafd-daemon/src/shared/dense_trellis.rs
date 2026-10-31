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
        self.check(&loaded)?;
        let shard=loaded.shard;
        ensure!(loaded.codebook==self.codebook && shard.side==self.side && shard.shape==self.shape
            && shard.payload.len()+shard.suh.len()+shard.svh.len()==self.bytes,
            "native K6 upload differs from admitted shard");
        let upload=|bytes:&[u8]|->Result<Allocation<'a>> {
            let _scope=cuteafd_ffi::memory_ledger::scope("weights/native K6 core");
            let owner=Allocation::new(device,bytes.len())?;
            device.run(||device.library.copy_h2d(owner.buffer,bytes))?;
            Ok(owner)
        };
        Ok(CoreWeights{payload:upload(&shard.payload)?,suh:upload(&shard.suh)?,svh:upload(&shard.svh)?,
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
        let layout=ProjectionScratchLayout::new(p.input,p.output,rows)?;
        let w=program.workspace();
        ensure!([("rotated",w.rotated),("table",w.pointer_table),("scales",w.block_scales),
            ("accumulator",w.accumulator)].iter()
            .all(|(name,bytes)|layout.extents.get(name)==Some(bytes))
            && layout.extents["partial"]==w.partial,"K6 package/planner scratch mismatch");
        Ok(Self{extents:layout.extents,rows:layout.rows})
    }
    pub(crate) fn demand(&self,device:u8,name:&str)->Result<Demand> {
        Ok(Demand::new(device,Category::Workspace,name,self.bytes()? as u64,Basis::Exact))
    }
    pub(crate) fn allocate<'a>(&self,device:Device<'a>)->Result<ProjectionScratch<'a>> {
        let storage=ProjectionScratchLayout {
            extents:self.extents.clone(),rows:self.rows }.storage()?;
        let mut owners=BTreeMap::new();
        for (&name,&bytes) in &storage.owners {owners.insert(name,Allocation::new(device,bytes)?);}
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
    fn buffers(&self,input:Buffer,weights:&CoreWeights<'_>)->DenseTrellisBuffers {
        let get=|name|self.buffers[name];
        DenseTrellisBuffers {input,payload:weights.payload.buffer,suh:weights.suh.buffer,svh:weights.svh.buffer,
            rotated:get("rotated"),pointer_table:get("table"),scales:get("scales"),partial:get("partial"),
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
        ensure!(weights.shape==[policy.output,policy.input] && weights.bits==policy.bits
            && accumulator.bytes>=rows*policy.output*4,"trellis F32 core destination geometry");
        let b=DenseTrellisBuffers{accumulator,..scratch.buffers(input,weights)};
        self.program.core(rows,b,stream)?;Ok(())
    }
    pub(crate) fn local(&self,rows:usize,input:Buffer,weights:&CoreWeights<'_>,scratch:&ProjectionScratch<'_>,
        output:Buffer,stream:*mut c_void)->Result<()> {
        ensure!(self.program.policy().side!=2,"K-split output requires F32 peer reduction before egress");
        let core=self.core(rows,input,weights,scratch,stream)?;
        self.program.output(rows,core,weights.svh.buffer,output,stream)?;Ok(())
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
    pub(crate) fn new(programs: &'a Programs<'a>, weights: &CoreWeights<'a>, rows: usize,
        device: Device<'a>, stream: *mut c_void,
        admitted: &cuteafd_loader::formats::trellis_core::ProjectionScratchLayout) -> Result<Self> {
        ensure!(weights.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Output,
            "GLM K6 head must be whole on rank 0");
        let projection = device.run(|| Projection::load_geometry_bits(programs, weights.shape[1], weights.shape[0],
            1, weights.bits, codebook(weights), rows, TrellisDType::F32))?;
        let layout = ProjectionLayout::new(&projection.program, rows)?;
        ensure!(admitted.rows == layout.rows && admitted.extents == layout.extents,
            "GLM K6 head scratch differs from admission");
        let scratch = layout.allocate(device)?;
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
