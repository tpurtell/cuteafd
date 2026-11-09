//! Output-channel FP8 projection shards with lane-owned exchange storage.
use crate::shared::memory::Tag;
use crate::shared::memory::{HostAllocation, device::{Allocation, Device, Event, Stream}};
use anyhow::{ensure, Context, Result};
use crate::shared::decode_graph::{CacheAttached, LayerGraphs};
use cuteafd_ffi::{CuteafdDeviceBuffer, V41Fp8Plan, V41PeerCopy};
use cuteafd_loader::OfficialV41Catalog;
use std::rc::Rc;
use crate::shared::memory::row_window::{RowWindow, ScratchGuard};
use cuteafd_core::memory_layout::ArenaTag;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind { QueryB, OutputB }
impl Kind {
    pub fn geometry(self) -> (usize, usize) {
        match self { Self::QueryB => (1280, 32768), Self::OutputB => (8192, 5120) }
    }
    fn name(self) -> &'static str { match self { Self::QueryB => "wq_b", Self::OutputB => "wo_b" } }
}

pub(crate) struct Weights<'a> {
    device: Device<'a>,
    layer: usize,
    rank: usize,
    kind: Kind,
    weight: Allocation<'a>,
    scales: Allocation<'a>,
}
impl<'a> Weights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        let name = format!("gpu{}.layers.{}.attn.{}.weight", self.device.id, self.layer, self.kind.name());
        self.weight.inventory(inventory, &name)?;
        self.scales.inventory(inventory, &cuteafd_loader::placement::families::deepseek_v41::weight_layout::packed_scale_name(&name))
    }
    pub fn device_bytes(kind: Kind) -> usize {
        let (k,n)=kind.geometry(); k*(n/2) + k*(n/2)/32
    }
    pub fn load_peak_device_bytes(kind: Kind) -> usize {
        let (k,n)=kind.geometry(); Self::device_bytes(kind) + k*(n/2)/1024
    }
    #[cfg(test)]
    pub fn load(device: Device<'a>, catalog: &OfficialV41Catalog, layer: usize,
        kind: Kind, rank: usize, budget: usize) -> Result<Self> {
        Self::load_in(device, catalog, layer, kind, rank, budget, None)
    }
    pub fn load_in(device: Device<'a>, catalog: &OfficialV41Catalog, layer: usize,
        kind: Kind, rank: usize, budget: usize,
        arena: Option<&crate::shared::memory::weight_arena::DeviceWeightArena<'a>>) -> Result<Self> {
        ensure!(layer<40 && rank<2 && budget>=Self::load_peak_device_bytes(kind),
            "invalid projection shard layer, rank or budget");
        let (k,n)=kind.geometry(); let n=n/2;
        let prefix=format!("layers.{layer}.attn.{}",kind.name());
        let weight_name=format!("{prefix}.weight"); let scale_name=format!("{prefix}.scale");
        ensure!(catalog.tensor(&weight_name)?.metadata.shape==[2*n,k]
            && catalog.tensor(&scale_name)?.metadata.shape==[2*n/32,k/32],
            "projection checkpoint geometry differs");
        let stream=Stream::new(device)?;
        let weight=crate::shared::memory::weight_arena::weight_allocation(device,arena,&weight_name,k*n, ArenaTag::weights("v41/target"))?;
        let scales=crate::shared::memory::weight_arena::weight_allocation(device,arena,
            &cuteafd_loader::placement::families::deepseek_v41::weight_layout::packed_scale_name(&weight_name),k*n/32, ArenaTag::weights("v41/target"))?;
        let raw_scales=Allocation::new(device,k*n/1024,Tag::staging("v41/projection-load"))?;
        let mut staging=HostAllocation::new(device.library,k*n,Tag::staging("v41/projection"))?;
        for (name,target) in [(&weight_name,&weight),(&scale_name,&raw_scales)] {
            let bytes=catalog.read_coordinator_tp2_into(name,0,rank,staging.bytes_mut(),&mut [])?;
            ensure!(bytes==target.buffer.bytes,"projection shard payload size differs");
            device.run(||device.library.copy_h2d(target.buffer,&staging.bytes_mut()[..bytes]))?;
        }
        let kernel=device.run(||device.library.v41_fp8_matrix_kernel(1,k as u32,n as u32))?;
        let queued=device.run(||unsafe { kernel.pack_scales(raw_scales.buffer,scales.buffer,stream.raw) });
        let drained=stream.drain();queued.and(drained)?;
        Ok(Self { device,layer,rank,kind,weight,scales })
    }
}

/// Guarded unit-cap inputs/outputs; gathered/query fork readers finish inside
/// the hook. GEMM scratch/alpha are plain scratch on each rank's shared stream.
#[cfg(test)]
thread_local! { static SKIP_STORAGE_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
fn acquire(guard: &ScratchGuard<'_>, stream: *mut std::ffi::c_void) -> Result<()> {
    #[cfg(test)]
    if SKIP_STORAGE_ACQUIRE.with(|skip|skip.get()) { return Ok(()); }
    // SAFETY: caller invokes on this live writer's device.
    unsafe { guard.acquire(stream) }
}
struct RankStorage<'a> {
    stream: Stream<'a>,
    plan: V41Fp8Plan<'a>,
    copy: V41PeerCopy<'a>,
    input: Allocation<'a>,
    output: Allocation<'a>,
    scratch: Allocation<'a>,
    alpha: Allocation<'a>,
    guard: ScratchGuard<'a>,
}
struct Rank<'w, 'a> {
    graphs: LayerGraphs<'w, 'a, Weights<'a>>,
    weights: &'w [Weights<'a>],
    storage: Rc<RankStorage<'a>>,
}
impl<'a> RankStorage<'a> {
    fn allocation_sizes(scratch: usize, kind: Kind, unit: u32) -> [usize; 4] {
        let (k,n) = kind.geometry(); [scratch,16,unit as usize*k*2,unit as usize*n]
    }
    fn new(device: Device<'a>, kind: Kind, capacity: u32, readers: &[i32]) -> Result<Rc<Self>> {
        let (k, n) = kind.geometry();
        let plan = device.run(|| device.library.v41_fp8_matrix_plan(capacity, k as u32, (n / 2) as u32))?;
        let stream = Stream::new(device)?;
        let [scratch_bytes, alpha_bytes, input_bytes, output_bytes] = Self::allocation_sizes(plan.info().scratch_bytes as usize, kind, capacity);
        let scratch = Allocation::new(device, scratch_bytes, Tag::workspace("v41/projection"))?;
        let alpha = Allocation::new(device, alpha_bytes, Tag::workspace("v41/projection"))?;
        let initialized = device.run(|| unsafe { plan.initialize_scratch(scratch.buffer, alpha.buffer, stream.raw) });
        initialized.and(stream.drain())?;
        Ok(Rc::new(Self { stream, plan, copy: device.run(|| device.library.v41_peer_copy())?,
            input: Allocation::new(device, input_bytes, Tag::workspace("v41/projection"))?,
            output: Allocation::new(device, output_bytes, Tag::workspace("v41/projection"))?, scratch, alpha,
            guard: ScratchGuard::new(device.library, readers)?,
        }))
    }
}
impl Drop for RankStorage<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("projection rank storage"); }
}
impl<'w,'a> Rank<'w,'a> {
    fn on_storage(weights: &'w [Weights<'a>], storage: Rc<RankStorage<'a>>) -> Result<Self> {
        let first=weights.first().context("projection shard weights absent")?;
        let device=first.device;
        ensure!(weights.len()<=40 && weights.iter().all(|w|w.kind==first.kind
            && w.rank==first.rank && w.device.id==device.id
            && std::ptr::eq(w.device.library,device.library)),"projection weight bank differs");
        for (index,w) in weights.iter().enumerate() {
            ensure!(!weights[..index].iter().any(|other|other.layer==w.layer),"duplicate projection layer");
        }
        ensure!(storage.stream.device.id == device.id, "projection rank storage device differs");
        let graphs=LayerGraphs::new(device.library, cuteafd_core::V41_VERIFY_ROWS as u32, "deepseek_v41");
        Ok(Self { graphs, weights, storage })
    }
}

impl<'w,'a> Rank<'w,'a> {
    unsafe fn enqueue(&self,weights:&Weights<'a>,rows:u32)->Result<()> {
        unsafe { self.storage.plan.launch(self.storage.input.buffer,weights.weight.buffer,weights.scales.buffer,
            self.storage.scratch.buffer,self.storage.alpha.buffer,self.storage.output.buffer,rows,self.storage.stream.raw) }
    }
    // Called only after the gathered result has completed, which also covers
    // the peer rank. Capture records the next invocation without repeating work.
    fn capture_ready(&mut self,layer:usize,rows:u32)->Result<()> {
        let weights=self.weights.iter().find(|w|w.layer==layer).context("projection layer absent")?;
        if !super::graph_policy::captures_shape(rows) || crate::shared::decode_graph::census::eager("layer_row")
            || self.graphs.get_shape(layer,weights,rows).is_some() { return Ok(()); }
        let device=self.storage.stream.device;
        let Some(ticket) = self.graphs.cache_ticket(&(layer,rows), super::graph_policy::pinned_shape(rows), None) else {
            super::graph_policy::note_miss(format!("projection_rank rows={rows}"), || "graph cache selected eager".into());
            return Ok(());
        };
        device.run(||unsafe {
            device.library.cuda_graph_begin_capture(self.storage.stream.raw)?;
            let queued=self.enqueue(weights,rows);
            let captured=device.library.cuda_graph_end_capture_topology(self.storage.stream.raw);
            match (queued,captured) {
                (Ok(()),Ok((graph, nodes, width, _spans, _))) => {
                    if let Err(error)=self.graphs.cache_insert(layer,weights,rows,graph,ticket,nodes as u64,width,true) {
                        device.library.cuda_graph_exec_destroy(graph)?;return Err(error);
                    }
                    Ok(())
                },
                (Err(error),Ok((graph, _, _, _, _))) => { device.library.cuda_graph_exec_destroy(graph)?;Err(error) },
                (Err(error),Err(_)) | (Ok(()),Err(error)) => Err(error),
            }
        })
    }
}
impl Drop for Rank<'_,'_> {
    fn drop(&mut self) {
        if let Err(error)=self.storage.stream.drain() { crate::shared::decode_graph::fatal_drain(Err(error), "projection rank"); }
        let device=self.storage.stream.device;
        if let Err(error)=device.run(||unsafe { self.graphs.clear() }) {
            tracing::error!(%error,"destroying projection rank graphs");
        }
    }
}

/// One TP2 projection storage shared by all slots. All device data are
/// guarded scratch at the unit cap; events/streams/plans live with storage.
pub(crate) struct Storage<'a> {
    ranks: [Rc<RankStorage<'a>>; 2],
    output: Allocation<'a>,
    peer_done: Event<'a>,
    input_ready: [Event<'a>; 2],
    guard: ScratchGuard<'a>,
    owner: usize,
    unit: u32,
    kind: Kind,
}
impl<'a> Storage<'a> {
    pub(crate) fn storage_bytes(library: &cuteafd_ffi::NativeLibrary, kind: Kind, unit: u32, owner: usize) -> Result<[usize; 2]> {
        Wave::device_bytes(library, kind, unit, owner)
    }
    pub(crate) fn new(devices: [Device<'a>; 2], kind: Kind, unit: u32, owner: usize) -> Result<Rc<Self>> {
        ensure!(owner < 2 && unit > 0 && unit <= 4096 && devices[0].id != devices[1].id,
            "invalid projection storage geometry");
        let readers = [devices[0].id, devices[1].id];
        for i in 0..2 { devices[i].run(|| devices[i].library.cuda_enable_peer(devices[1-i].id))?; }
        Ok(Rc::new(Self {
            ranks: [RankStorage::new(devices[0], kind, unit, &readers)?, RankStorage::new(devices[1], kind, unit, &readers)?],
            output: Allocation::new(devices[owner], unit as usize * kind.geometry().1 * 2, Tag::workspace("v41/projection"))?,
            peer_done: Event::new(devices[1-owner])?,
            input_ready: [Event::new(devices[0])?, Event::new(devices[1])?],
            guard: ScratchGuard::new(devices[0].library, &readers)?, owner, unit, kind,
        }))
    }
}
impl Drop for Storage<'_> {
    fn drop(&mut self) { for rank in &self.ranks { rank.stream.drain_or_abort("projection storage"); } }
}
pub(crate) struct Wave<'w,'a> {
    gather_graphs: LayerGraphs<'w,'a,Weights<'a>>,
    ranks: [Rank<'w,'a>;2],
    storage: Rc<Storage<'a>>,
    window: RowWindow,
    owner: usize,
    capacity: u32,
    kind: Kind,
}
struct Drain<'s,'w,'a> { wave: &'s mut Wave<'w,'a>, complete: bool }
impl Drop for Drain<'_,'_,'_> {
    fn drop(&mut self) {
        if !self.complete {
            if let Err(error)=self.wave.drain() { crate::shared::decode_graph::fatal_drain(Err(error), "projection unwind"); }
        }
    }
}
impl<'w,'a> Wave<'w,'a> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut dyn FnMut(i32, &'static str, bool, &mut dyn CacheAttached) -> Result<()>) -> Result<()> {
        let (rank_site, gather_site) = match self.kind {
            Kind::QueryB => ("v41.projection.query.rank", "v41.projection.query.gather"),
            Kind::OutputB => ("v41.projection.output.rank", "v41.projection.output.gather"),
        };
        for rank in &mut self.ranks {
            f(rank.storage.input.buffer.device_id, rank_site, true, &mut rank.graphs)?;
        }
        f(self.storage.output.buffer.device_id, gather_site, true, &mut self.gather_graphs)
    }

    pub fn kind(&self) -> Kind { self.kind }
    pub(crate) fn capacity(&self) -> u32 { self.capacity }
    pub fn output_device(&self) -> Device<'a> { self.storage.output.device }
    pub fn device_bytes(library: &cuteafd_ffi::NativeLibrary,kind: Kind,capacity: u32,owner: usize) -> Result<[usize;2]> {
        ensure!(owner<2 && (1..=4096).contains(&capacity),"invalid projection wave geometry");
        let (k,n)=kind.geometry();
        let plan=library.v41_fp8_matrix_plan_info(capacity,k as u32,(n/2) as u32)?;
        let half=RankStorage::allocation_sizes(plan.scratch_bytes as usize,kind,capacity).into_iter().sum::<usize>();
        let mut bytes=[half;2];bytes[owner]+=capacity as usize*n*2;Ok(bytes)
    }
    pub fn new(weights: [&'w [Weights<'a>];2],capacity: u32,owner: usize,budgets: [usize;2]) -> Result<Self> {
        let a=weights[0].first().context("projection rank 0 absent")?;
        let b=weights[1].first().context("projection rank 1 absent")?;
        ensure!(a.rank==0 && b.rank==1 && a.kind==b.kind && a.device.id!=b.device.id
            && std::ptr::eq(a.device.library,b.device.library),"projection ranks differ");
        ensure!(weights[0].len()==weights[1].len() && weights[0].iter().zip(weights[1]).all(|(a,b)|a.layer==b.layer),
            "projection rank layers differ");
        let needed=Self::device_bytes(a.device.library,a.kind,capacity,owner)?;
        ensure!(needed.iter().zip(budgets).all(|(&need,budget)|need<=budget),"projection workspace exceeds budget");
        Self::on_storage(weights, Storage::new([a.device, b.device], a.kind, capacity, owner)?)
    }
    pub(crate) fn on_storage(weights: [&'w [Weights<'a>]; 2], storage: Rc<Storage<'a>>) -> Result<Self> {
        let a = weights[0].first().context("projection rank 0 absent")?;
        let b = weights[1].first().context("projection rank 1 absent")?;
        ensure!(a.kind == storage.kind && b.kind == storage.kind && a.rank == 0 && b.rank == 1
            && a.device.id == storage.ranks[0].stream.device.id && b.device.id == storage.ranks[1].stream.device.id
            && std::ptr::eq(a.device.library, storage.ranks[0].stream.device.library)
            && std::ptr::eq(b.device.library, storage.ranks[1].stream.device.library),
            "projection weights/storage differ");
        ensure!(weights[0].len() == weights[1].len()
            && weights[0].iter().zip(weights[1]).all(|(a,b)| a.layer == b.layer),
            "projection rank layers differ");
        let gather_graphs=LayerGraphs::new(a.device.library, cuteafd_core::V41_VERIFY_ROWS as u32, "deepseek_v41");
        Ok(Self { gather_graphs,
            ranks: [Rank::on_storage(weights[0], storage.ranks[0].clone())?, Rank::on_storage(weights[1], storage.ranks[1].clone())?],
            owner: storage.owner, capacity: storage.unit, kind: storage.kind, storage, window: RowWindow::NONE })
    }
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        if let Some(range) = window.range() { ensure!(range.start < range.end && range.len() <= self.capacity as usize,
            "projection window exceeds unit"); }
        self.window = window; Ok(())
    }
    pub(crate) fn release_scratch(&self) -> Result<()> {
        for rank in &self.ranks { rank.storage.guard.release()?; }
        self.storage.guard.release()
    }
    unsafe fn enqueue_gather(&self,rows:u32)->Result<()> {
        let (_,n)=self.kind.geometry();let owner=&self.ranks[self.owner];
        for (i,rank) in self.ranks.iter().enumerate() {
            let mut dst=self.storage.output.buffer;
            dst.ptr=unsafe { dst.ptr.cast::<u8>().add(i*n).cast() };dst.bytes-=i*n;
            unsafe { owner.storage.copy.launch_rows(dst,rank.storage.output.buffer,n,rows as usize,n*2,n,owner.storage.stream.raw)?; }
        }
        Ok(())
    }
    fn capture_gather_ready(&mut self,rows:u32)->Result<()> {
        // The gather uses only this wave's fixed storage, independent of layer.
        let weights=&self.ranks[self.owner].weights[0];
        if !super::graph_policy::captures_shape(rows) || crate::shared::decode_graph::census::eager("layer_row")
            || self.gather_graphs.get_shape(0,weights,rows).is_some() { return Ok(()); }
        let stream=self.ranks[self.owner].storage.stream.raw;let device=self.storage.output.device;
        let Some(ticket) = self.gather_graphs.cache_ticket(&(0,rows), super::graph_policy::pinned_shape(rows), None) else {
            super::graph_policy::note_miss(format!("projection_gather rows={rows}"), || "graph cache selected eager".into());
            return Ok(());
        };
        device.run(||unsafe {
            device.library.cuda_graph_begin_capture(stream)?;
            let queued=self.enqueue_gather(rows);
            let captured=device.library.cuda_graph_end_capture_topology(stream);
            match (queued,captured) {
                (Ok(()),Ok((graph, nodes, width, _spans, _))) => {
                    if let Err(error)=self.gather_graphs.cache_insert(0,weights,rows,graph,ticket,nodes as u64,width,true) {
                        device.library.cuda_graph_exec_destroy(graph)?;return Err(error);
                    }
                    Ok(())
                },
                (Err(error),Ok((graph, _, _, _, _))) => { device.library.cuda_graph_exec_destroy(graph)?;Err(error) },
                (Err(error),Err(_)) | (Ok(()),Err(error)) => Err(error),
            }
        })
    }
    fn drain(&self) -> Result<()> {
        let first=self.ranks[0].storage.stream.drain();let second=self.ranks[1].storage.stream.drain();first.and(second)
    }
    /// # Safety
    /// Input production is complete. Keep input live/unchanged until completion
    /// or cancellation drains both ranks. Consume the result before reusing wave.
    pub async unsafe fn execute(&mut self,layer: usize,rows: u32,input: CuteafdDeviceBuffer) -> Result<CuteafdDeviceBuffer> {
        unsafe { self.execute_after(layer,rows,input,None,|output,_|Ok(output)).await }
    }
    /// Enqueue after an optional producer stream and append a same-device
    /// consumer before the one final cooperative wait.
    /// # Safety
    /// Producer belongs to input.device_id; absent producer means input is ready.
    /// Retain producer/input/consumer storage through completion or cancellation.
    /// Consumer queues only on the supplied stream and must not publish results
    /// before this future completes. It runs on the gathered output's device.
    pub async unsafe fn execute_after<T>(&mut self,layer: usize,rows: u32,input: CuteafdDeviceBuffer,
        producer: Option<*mut std::ffi::c_void>,
        consume: impl FnOnce(CuteafdDeviceBuffer,*mut std::ffi::c_void)->Result<T>) -> Result<T> {
        let (k,n)=self.kind.geometry();
        ensure!(rows>0 && rows<=self.capacity && self.window.admits(rows as usize) && input.bytes>=rows as usize*k*2
            && !input.ptr.is_null() && self.ranks.iter().any(|r|r.storage.stream.device.id==input.device_id),
            "projection input differs");
        let mut guard=Drain { wave:self,complete:false };let wave=&mut *guard.wave;
        let input_rank=wave.ranks.iter().position(|r|r.storage.stream.device.id==input.device_id).unwrap();
        if let Some(producer)=producer {
            wave.ranks[input_rank].storage.stream.device.run(||unsafe {
                wave.ranks[input_rank].storage.stream.device.library.cuda_event_record(wave.storage.input_ready[input_rank].raw,producer)
            })?;
        }
        for rank in &mut wave.ranks {
            let weights=rank.weights.iter().find(|w|w.layer==layer).context("projection layer absent")?;
            let graph = if wave.window.is_windowed() { None }
                else { rank.graphs.cached_launch(layer,weights,rows).map(|(raw,_)|raw) };
            rank.storage.stream.device.run(||unsafe {
                acquire(&rank.storage.guard, rank.storage.stream.raw)?;
                if producer.is_some() {
                    rank.storage.stream.device.library.cuda_stream_wait_event(rank.storage.stream.raw,wave.storage.input_ready[input_rank].raw)?;
                }
                rank.storage.copy.launch_rows(rank.storage.input.buffer,input,k*2,rows as usize,k*2,k*2,rank.storage.stream.raw)?;
                crate::shared::decode_graph::census::dispatch(rank.storage.stream.device.library, rank.storage.stream.raw,
                    c"layer_row", rows as usize, graph,
                    || rank.enqueue(weights,rows))
            })?;
        }
        let owner=&wave.ranks[wave.owner];let peer=&wave.ranks[1-wave.owner];
        peer.storage.stream.device.run(|| unsafe { peer.storage.stream.device.library.cuda_event_record(
            wave.storage.peer_done.raw, peer.storage.stream.raw) })?;
        let gather_graph = if wave.window.is_windowed() { None }
            else { wave.gather_graphs.cached_launch(0,&owner.weights[0],rows).map(|(raw,_)|raw) };
        owner.storage.stream.device.run(||unsafe {
            acquire(&wave.storage.guard, owner.storage.stream.raw)?;
            owner.storage.stream.device.library.cuda_stream_wait_event(owner.storage.stream.raw,wave.storage.peer_done.raw)?;
            crate::shared::decode_graph::census::dispatch(owner.storage.stream.device.library, owner.storage.stream.raw,
                c"layer_row", rows as usize, gather_graph,
                || wave.enqueue_gather(rows))
        })?;
        let output=CuteafdDeviceBuffer { bytes:rows as usize*n*2,..wave.storage.output.buffer };
        let result=owner.storage.stream.device.run(||consume(output,owner.storage.stream.raw))?;
        owner.storage.stream.complete().await?;
        if !wave.window.is_windowed() && !crate::shared::memory::chain::deferred() {
            for rank in &mut wave.ranks { rank.capture_ready(layer,rows)?; }
            wave.capture_gather_ready(rows)?;
        }
        guard.complete=true;Ok(result)
    }
}
impl Drop for Wave<'_,'_> {
    fn drop(&mut self) {
        if let Err(error)=self.drain() { crate::shared::decode_graph::fatal_drain(Err(error), "projection"); }
        let device=self.storage.output.device;
        if let Err(error)=device.run(||unsafe { self.gather_graphs.clear() }) {
            tracing::error!(%error,"destroying projection gather graphs");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
    #[test]
    #[ignore = "requires checkpoint, projection shard AOT and two CUDA GPUs"]
    fn checkpoint_dual_projection_lanes_match_full() -> Result<()> {
        let lib=unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices=[Device { library:&lib,id:0 },Device { library:&lib,id:1 }];
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        for kind in [Kind::QueryB,Kind::OutputB] {
            let (k,n)=kind.geometry();let capacity=16;
            let weights=[(0..2).map(|i|Weights::load(devices[0],&catalog,[2,20][i],kind,0,
                Weights::load_peak_device_bytes(kind))).collect::<Result<Vec<_>>>()?,
                (0..2).map(|i|Weights::load(devices[1],&catalog,[2,20][i],kind,1,
                Weights::load_peak_device_bytes(kind))).collect::<Result<Vec<_>>>()?];
            let banks=[weights[0].as_slice(),weights[1].as_slice()];
            assert!(Wave::new(banks,capacity,0,[0,0]).is_err());
            let mut lanes=[Wave::new(banks,capacity,0,Wave::device_bytes(&lib,kind,capacity,0)?)?,
                Wave::new(banks,capacity,1,Wave::device_bytes(&lib,kind,capacity,1)?)?];
            for lane in &mut lanes {
                lane.visit_graph_banks(&mut |device, name, spans, bank| bank.attach(device, 8 << 30,
                    crate::shared::decode_graph::device::Site { name,
                        cycle: crate::shared::decode_graph::device::Cycle::Other,
                        preference: crate::shared::decode_graph::device::Preference::Capture,
                        bootstrap: 2 << 20, basis: crate::shared::decode_graph::device::Basis::Bytes,
                        spans_devices: spans }))?;
            }
            let inputs=[Allocation::new(devices[0],capacity as usize*k*2, crate::shared::memory::TEST)?,
                Allocation::new(devices[1],capacity as usize*k*2, crate::shared::memory::TEST)?];
            let plan=devices[0].run(||lib.v41_fp8_matrix_plan(capacity,k as u32,n as u32))?;
            let stream=Stream::new(devices[0])?;
            let scratch=Allocation::new(devices[0],plan.info().scratch_bytes as usize, crate::shared::memory::TEST)?;
            let alpha=Allocation::new(devices[0],16, crate::shared::memory::TEST)?;
            let output=Allocation::new(devices[0],capacity as usize*n*2, crate::shared::memory::TEST)?;
            devices[0].run(||unsafe { plan.initialize_scratch(scratch.buffer,alpha.buffer,stream.raw) })?;
            stream.drain()?;
            for layer in [2,20] {
                let names=[format!("layers.{layer}.attn.{}.weight",kind.name()),
                    format!("layers.{layer}.attn.{}.scale",kind.name())];
                let full=devices[0].own(||NativeRtxTensors::load(&lib,&catalog,&names,
                    NativeRtxTensors::plan(&catalog,&names)?,1<<20))?;
                let scales=Allocation::new(devices[0],k*n/32, crate::shared::memory::TEST)?;
                devices[0].run(||unsafe { lib.v41_fp8_matrix_kernel(1,k as u32,n as u32)?
                    .pack_scales(full.get().get(&names[1])?,scales.buffer,stream.raw) })?;
                stream.drain()?;
                let mut first_graphs=None;
                for (rows,seed) in [(1,0),(6,7),(16,3),(1,5)] {
                    let host:Vec<u8>=(0..capacity as usize*k).flat_map(|i| {
                        let value=((i+seed)%31) as f32/32.0-0.5;
                        ((value.to_bits()>>16) as u16).to_ne_bytes()
                    }).collect();
                    for input in &inputs { input.device.run(||lib.copy_h2d(input.buffer,&host))?; }
                    devices[0].run(||unsafe { plan.launch(inputs[0].buffer,full.get().get(&names[0])?,scales.buffer,
                        scratch.buffer,alpha.buffer,output.buffer,rows,stream.raw) })?;
                    stream.drain()?;
                    let mut expected=vec![0;rows as usize*n*2];
                    devices[0].run(||lib.copy_d2h(&mut expected,CuteafdDeviceBuffer { bytes:rows as usize*n*2,..output.buffer }))?;
                    // Poll once and discard an unpublished submission, then reuse
                    // the same owner; its guard drains any work already issued.
                    runtime.block_on(async {
                        let mut pending=std::pin::pin!(unsafe { lanes[0].execute(layer,rows,inputs[0].buffer) });
                        std::future::poll_fn(|cx| {
                            use std::future::Future;
                            let _=pending.as_mut().poll(cx);std::task::Poll::Ready(())
                        }).await;
                    });
                    let [left,right]=&mut lanes;
                    let (a,b)=runtime.block_on(async { tokio::join!(
                        unsafe { left.execute(layer,rows,inputs[1].buffer) },
                        unsafe { right.execute(layer,rows,inputs[0].buffer) }) });
                    let graph_handles:Vec<_>=lanes.iter().flat_map(|lane| {
                        lane.ranks.iter().map(|rank| {
                            let weights=rank.weights.iter().find(|w|w.layer==layer).unwrap();
                            rank.graphs.get_shape(layer,weights,rows).unwrap().0
                        }).chain(std::iter::once(lane.gather_graphs.get_shape(0,
                            &lane.ranks[lane.owner].weights[0],rows).unwrap().0))
                    }).collect();
                    if rows==1 {
                        if let Some(first)=&first_graphs { assert_eq!(first,&graph_handles); }
                        else { first_graphs=Some(graph_handles); }
                    }
                    for (device,result) in devices.into_iter().zip([a?,b?]) {
                        let mut actual=vec![0;expected.len()];device.run(||lib.copy_d2h(&mut actual,result))?;
                        let mut max_error=0f32;
                        for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                            let decode=|v:&[u8]|f32::from_bits(u32::from(u16::from_ne_bytes([v[0],v[1]]))<<16);
                            let (a,b)=(decode(a),decode(b));
                            ensure!(a.is_finite() && b.is_finite() && (a-b).abs()<=1e-4,
                                "projection differs: {kind:?} layer={layer} rows={rows} {a} vs {b}");
                            max_error=max_error.max((a-b).abs());
                        }
                        eprintln!("projection owner {kind:?} layer={layer} rows={rows} gpu={} max_error={max_error}",device.id);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod storage_bytes_tests {
    use super::*;
    #[test]
    fn projection_storage_bytes_equal_rank_allocation_plan() {
        for kind in [Kind::QueryB,Kind::OutputB] {for unit in [4096,2752,2048] {
            let (k,n)=kind.geometry();
            let sizes=RankStorage::allocation_sizes(123,kind,unit);
            assert_eq!(sizes.into_iter().sum::<usize>(),123+16+unit as usize*(k*2+n));
        }}
    }
}
#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::shared::memory::chain::StageChain;
    fn run(kind: Kind, guarded: bool) -> Result<()> {
        let lib=unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        lib.cuda_set_device(0)?;
        let devices=[Device {library:&lib,id:0},Device {library:&lib,id:1}];
        let weights=[vec![Weights::load(devices[0],&catalog,2,kind,0,Weights::load_peak_device_bytes(kind))?],
            vec![Weights::load(devices[1],&catalog,2,kind,1,Weights::load_peak_device_bytes(kind))?]];
        let banks=[weights[0].as_slice(),weights[1].as_slice()];
        let (rows,unit)=(13u32,16u32);let (k,n)=kind.geometry();
        let inputs=[Allocation::new(devices[0],rows as usize*k*2, crate::shared::memory::TEST)?,Allocation::new(devices[0],rows as usize*k*2, crate::shared::memory::TEST)?];
        for (seed,input) in inputs.iter().enumerate() {
            let host:Vec<u8>=(0..rows as usize*k).flat_map(|i| {
                let f=((i*7+seed*11)%29) as f32/32.0-0.4;((f.to_bits()>>16) as u16).to_ne_bytes()
            }).collect();lib.copy_h2d(input.buffer,&host)?;
        }
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        let mut private=Wave::new(banks,unit,0,Wave::device_bytes(&lib,kind,unit,0)?)?;
        let expected=runtime.block_on(unsafe {private.execute(2,rows,inputs[0].buffer)})?;
        let mut want=vec![0;rows as usize*n*2];lib.copy_d2h(&mut want,expected)?;
        let storage=Storage::new(devices,kind,unit,0)?;
        let mut slots=[Wave::on_storage(banks,storage.clone())?,Wave::on_storage(banks,storage.clone())?];
        for (i,slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::within(i*unit as usize..i*unit as usize+rows as usize,48)?)?;
            runtime.block_on(unsafe {slot.execute(2,rows,inputs[i].buffer)})?;
        }
        let reader=Stream::new(devices[1])?;let saved=Allocation::new(devices[1],want.len(), crate::shared::memory::TEST)?;
        devices[1].run(||unsafe {lib.copy_peer_async(saved.buffer,storage.output.buffer,want.len(),reader.raw)})?;reader.drain()?;
        let chains=[StageChain::on_devices(&lib,&[0,1])?,StageChain::on_devices(&lib,&[0,1])?];
        let gate=devices[1].run(||unsafe {cuteafd_ffi::test_support::CudaStreamGate::new(&lib,reader.raw)})?;
        runtime.block_on(chains[0].handle().scope(async {
            let output=unsafe {slots[0].execute(2,rows,inputs[0].buffer).await?};
            devices[1].run(||unsafe {crate::shared::memory::chain::join(&lib,reader.raw)?;
                lib.copy_peer_async(saved.buffer,output,want.len(),reader.raw)?;
                crate::shared::memory::chain::finish(&lib,reader.raw)})?;
            slots[0].release_scratch()
        }))?;
        SKIP_STORAGE_ACQUIRE.with(|flag|flag.set(!guarded));
        let queued=runtime.block_on(chains[1].handle().scope(async {
            unsafe {slots[1].execute(2,rows,inputs[1].buffer).await?;} anyhow::Ok(())
        }));
        SKIP_STORAGE_ACQUIRE.with(|flag|flag.set(false));queued?;
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(if guarded {2} else {10});let mut done=false;
        while std::time::Instant::now()<deadline {
            if devices[0].run(||unsafe {lib.cuda_stream_query(storage.ranks[0].stream.raw)})? {done=true;break;}
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        drop(gate);for chain in &chains {chain.drain()?;}reader.drain()?;
        assert_eq!(done,!guarded,"projection B completion before A peer-reader release");
        let mut got=vec![0;want.len()];devices[1].run(||lib.copy_d2h(&mut got,saved.buffer))?;
        if guarded {assert_eq!(got,want);} else {assert_ne!(got,want,"negative control did not clobber gather output");}
        Ok(())
    }
    #[test]
    #[ignore = "requires projection AOT, snapshot and two GPUs"]
    fn tp2_projection_shared_scratch_preserves_gated_peer_reader() -> Result<()> {
        for _ in 0..3 {for kind in [Kind::QueryB,Kind::OutputB] {run(kind,true)?;}} Ok(())
    }
    #[test]
    #[ignore = "requires projection AOT, snapshot and two GPUs"]
    fn tp2_projection_without_guard_clobbers_gated_peer_reader() -> Result<()> {
        for _ in 0..3 {for kind in [Kind::QueryB,Kind::OutputB] {run(kind,false)?;}} Ok(())
    }
}
