//! Per-slot head-half waves on shared, guarded unit-cap attention storage.
use super::*;
use crate::shared::decode_graph::GraphPolicy;
use crate::shared::memory::Tag;
use crate::shared::memory::device::{Allocation, Device, DeviceOwner, Event};
use crate::families::deepseek_v41::v41_backbone_cache::{BackboneCache,CacheAttention,peer_inputs::PeerAttentionInputs};

/// Query/output, gather output, mirrored proposal/selection inputs and the
/// copied sink half are guarded unit scratch, consumed within one attention
/// hook. The half workspaces/descriptors are plain scratch on shared streams.
/// Host metadata, cold state and graph banks remain per slot. Mirrored source
/// bindings live on the shared PeerAttentionInputs, beside the shared arena:
/// A -> B -> A changes that one key on each slot switch and recopies A. The
/// guard orders the writes after all earlier readers, even with both slots in
/// flight. A repeated binding can reuse bytes only while that same arena still
/// carries it; no per-slot cache key survives another slot's overwrite.
#[cfg(test)]
thread_local! { static SKIP_DUAL_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
unsafe fn acquire_dual(guard: &crate::shared::memory::row_window::ScratchGuard<'_>, stream: *mut c_void) -> Result<()> {
    #[cfg(test)]
    if SKIP_DUAL_ACQUIRE.with(|skip| skip.get()) { return Ok(()); }
    // SAFETY: caller supplies a live writer stream on the current device.
    unsafe { guard.acquire(stream) }
}
pub(crate) struct DualAttentionStorage<'a> {
    halves: [std::rc::Rc<SparseAttentionStorage<'a, 32>>; 2],
    copies: [V41PeerCopy<'a>; 2],
    peer_done: Event<'a>,
    output: Allocation<'a>,
    inputs: PeerAttentionInputs<'a>,
    peer_sink: Allocation<'a>,
    guard: crate::shared::memory::row_window::ScratchGuard<'a>,
    capacity: usize,
}
impl<'a> DualAttentionStorage<'a> {
    pub(crate) fn storage_bytes(unit: usize) -> Result<[usize; 2]> {
        let half = SparseAttentionStorage::<'_, 32>::storage_bytes(unit)?;
        Ok([half + unit*64*1024, half + PeerAttentionInputs::device_bytes(unit)? + 128])
    }
    pub(crate) fn new(devices: [Device<'a>; 2], unit: usize) -> Result<std::rc::Rc<Self>> {
        ensure!(devices[0].id != devices[1].id && std::ptr::eq(devices[0].library, devices[1].library),
            "dual attention devices differ");
        Self::storage_bytes(unit)?;
        for i in 0..2 { devices[i].run(|| devices[i].library.cuda_enable_peer(devices[1-i].id))?; }
        let readers = [devices[0].id, devices[1].id];
        let inputs = PeerAttentionInputs::new(devices[0], devices[1], unit, PeerAttentionInputs::device_bytes(unit)?)?;
        Ok(std::rc::Rc::new(Self {
            halves: [devices[0].run(|| SparseAttentionStorage::new(devices[0].library, unit, &readers))?,
                devices[1].run(|| SparseAttentionStorage::new(devices[1].library, unit, &readers))?],
            copies: [devices[0].run(|| devices[0].library.v41_peer_copy())?,
                devices[1].run(|| devices[1].library.v41_peer_copy())?],
            peer_done: Event::new(devices[1])?,
            output: Allocation::new(devices[0], unit*64*1024, Tag::workspace("v41/dual-attention"))?, inputs,
            peer_sink: Allocation::new(devices[1], 128, Tag::transport("v41/dual-attention"))?,
            guard: crate::shared::memory::row_window::ScratchGuard::new(devices[0].library, &readers)?,
            capacity: unit,
        }))
    }
}
impl Drop for DualAttentionStorage<'_> {
    fn drop(&mut self) {
        for half in &self.halves {
            crate::shared::decode_graph::fatal_drain(unsafe {
                half.stream.library.cuda_set_device(half.query.buffer.device_id).and_then(|_|
                    half.stream.library.cuda_stream_synchronize(half.stream.raw))
            }, "dual attention storage");
        }
    }
}
pub(crate) struct DualAttentionWave<'a> {
    halves: [DeviceOwner<'a, CompactSparseAttentionWave<'a>>; 2],
    storage: std::rc::Rc<DualAttentionStorage<'a>>,
    window: crate::shared::memory::row_window::RowWindow,
    capacity: usize,
    pending: Option<OwnedSubmission>,
    tails: GraphBank<(usize, Vec<usize>), GraphOwner<'a, ()>>,
}
struct OwnedSubmission { cold:[bool;2],rows:usize,layer:usize }
impl<'a> DualAttentionWave<'a> {
    /// Budgets are in head-half order: the first device owns the gathered output.
    /// Includes private proposals, selection and sink storage on the peer.
    /// Committed cache replicas remain budgeted by the backbone bank.
    pub(crate) fn source_device(&self) -> Device<'a> { self.halves[0].device }
    pub(crate) fn capacity(&self) -> usize { self.capacity }
    pub fn device_bytes(capacity: usize) -> Result<[usize; 2]> {
        DualAttentionStorage::storage_bytes(capacity)
    }
    pub fn new(devices: [Device<'a>; 2], capacity: usize, budgets: [usize; 2]) -> Result<Self> {
        ensure!(Self::device_bytes(capacity)?.iter().zip(budgets).all(|(n,b)| *n<=b), "dual attention exceeds budget");
        Self::on_storage(DualAttentionStorage::new(devices, capacity)?)
    }
    pub(crate) fn on_storage(storage: std::rc::Rc<DualAttentionStorage<'a>>) -> Result<Self> {
        let devices = std::array::from_fn::<_, 2, _>(|i| Device {
            library: storage.halves[i].stream.library, id: storage.halves[i].query.buffer.device_id });
        Ok(Self {
            halves: [devices[0].own(|| CompactSparseAttentionWave::on_storage(storage.halves[0].clone()))?,
                devices[1].own(|| CompactSparseAttentionWave::on_storage(storage.halves[1].clone()))?],
            capacity: storage.capacity, storage, window: crate::shared::memory::row_window::RowWindow::NONE,
            pending: None, tails: GraphBank::with_family(GraphPolicy::Budgeted { bytes: None }, "deepseek_v41"),
        })
    }
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.pending.is_some(), "dual attention", "work is pending")?;
        for half in &mut self.halves { half.select_window(window.clone())?; }
        self.window = window; Ok(())
    }
    pub(crate) fn release_scratch(&self) -> Result<()> {
        for half in &self.halves { half.release_scratch()?; }
        self.storage.guard.release()
    }
    pub fn install_attention_plan(&mut self, plan: cuteafd_core::coordinator_programs::AttentionReductionPlan) -> Result<()> {
        for half in &mut self.halves {
            let device = half.device;
            device.run(|| half.get_mut().install_attention_plan(plan))?;
        }
        Ok(())
    }
    pub fn set_decode(&mut self, decode: bool) {
        for half in &mut self.halves { half.get_mut().set_decode(decode); }
    }
    pub fn reserve_decode_rows(&mut self, rows: usize) -> Result<()> {
        for half in &mut self.halves {
            let device=half.device;
            device.run(|| half.get_mut().reserve_decode_rows(rows))?;
        }
        Ok(())
    }
    pub fn enable_small_graph_shapes(&mut self) {
        for half in &mut self.halves { half.enable_small_graph_shapes(); }
    }
    pub fn drain(&mut self) -> Result<()> {
        // Always attempt both drains, even if the first device reports an error.
        let devices=[self.halves[0].device,self.halves[1].device];
        let first = devices[0].run(|| self.halves[0].drain_chain());
        let second = devices[1].run(|| self.halves[1].drain_chain());
        self.pending=None;
        first.and(second)
    }
    /// # Safety
    /// Same external lifetime contract as enqueue_cached. The enclosing lane
    /// must complete or drain this owner before releasing any input/consumer.
    pub unsafe fn enqueue_cached_owned(&mut self,query:&AttentionQueryOutput<'_>,sink:CuteafdDeviceBuffer,
        bank:&BackboneCache<'_>,cache:&CacheAttention<'_>,selection:Option<&IndexSelectionOutput<'_>>)->Result<()> {
        ensure!(self.pending.is_none(),"dual attention submission already pending");
        let mut submission=unsafe { self.enqueue_cached(query,sink,bank,cache,selection)? };
        let plan=OwnedSubmission { cold:submission.cold,rows:submission.rows,layer:submission.layer };
        submission.wave=None;
        drop(submission);
        self.pending=Some(plan);Ok(())
    }
    /// # Safety
    /// Preserve enqueue_cached_owned's owners and complete_then's consumer contract.
    pub async unsafe fn complete_owned_then<T>(&mut self,
        consume:impl FnOnce(QueuedSparseAttention,*mut c_void)->Result<T>)->Result<T> {
        let plan=self.pending.take().context("dual attention submission absent")?;
        let pending=PendingDualAttention { wave:Some(self),cold:plan.cold,rows:plan.rows,layer:plan.layer };
        unsafe { pending.complete_then(consume).await }
    }
    /// Complete this lane with a captured projection/FFN continuation. The
    /// enclosing backbone retains its weights and buffers until this wave drops.
    /// Cold tails warm once before capture; warm tails have only the existing
    /// final completion wait and never join another lane.
    /// # Safety
    /// Same ownership contract as complete_owned_then, plus retain all tail
    /// storage/weights until graph destruction and provide an exact identity.
    pub async unsafe fn complete_owned_tail(&mut self, tail: &mut dyn AttentionGraphTail) -> Result<()> {
        let plan=self.pending.as_ref().context("dual attention submission absent")?;
        let (layer,rows)=(plan.layer,plan.rows);
        let mut key=tail.identity();
        key.extend([rows,self.storage.output.buffer.ptr as usize]);
        let graph=if self.window.is_windowed() { None } else { self.tails.launch(&(layer,key.clone())).map(|graph|graph.raw) };
        let library=self.halves[0].device.library;
        let (attention,stream)=unsafe { self.complete_owned_then(|attention,stream| {
            tail.prepare(stream)?;
            let eager = crate::shared::decode_graph::census::eager("layer_row");
            if graph.is_some() && !eager { tail.replay_state()?; }
            crate::shared::decode_graph::census::dispatch(library, stream, c"layer_row", rows,
                graph, || tail.enqueue(&attention,stream))?;
            Ok((attention,stream))
        }).await? };
        if graph.is_some() || self.window.is_windowed() { return Ok(()); }
        // The warmup is complete. Re-establish host-side block state, capture
        // the same operations and replay them before publishing FFN output.
        let mut guard=PendingDualAttention { wave:Some(self),cold:[false;2],rows,layer };
        let wave=guard.wave.as_deref_mut().unwrap();
        let owner=wave.halves[0].device;
        owner.run(||unsafe {
            tail.restore_warmup()?;
            library.cuda_graph_begin_capture(stream)?;
            let queued=tail.enqueue(&attention,stream);
            let captured=library.cuda_graph_end_capture(stream);
            let graph=match (queued,captured) {
                (Ok(()),Ok(graph))=>graph,
                (Err(error),Ok(graph))=> { library.cuda_graph_exec_destroy(graph)?;return Err(error); },
                (Err(error),Err(_)) | (Ok(()),Err(error))=>return Err(error),
            };
            // Match the attention owner's bounded shape retention policy.
            let limit=wave.halves[0].graph_limit;
            if wave.tails.count(|(site,_)| *site==layer)>=limit {
                wave.tails.retire_oldest(|(site,_)| *site==layer);
                // The cold continuation above completed the entire owner chain.
                drop(wave.tails.drain_retired(||Ok(()))?);
            }
            // SAFETY: the enclosing lane retains tail weights and storage and
            // drains this owner before releasing them.
            let exec=GraphOwner::new(library,owner.id,graph,())?;
            tracing::debug!(target:"cuteafd::graph_capture",site="dual_attention_tail",
                layer,rows,device=owner.id,key=?key,bank=?(&wave.tails as *const _),
                "captured graph binding");
            wave.tails.insert((layer,key),exec,None);
            library.cuda_graph_launch(graph,stream)
        })?;
        owner.future(wave.halves[0].wait_chain()).await?;
        guard.wave=None;
        Ok(())
    }
    /// # Safety
    /// Original query/proposal/selection producers and committed replicas are
    /// complete. Retain bank, cache, query, selection and sink through returned
    /// completion or cancellation. This owner cannot be reused while consumers
    /// still read its previous gathered output.
    pub unsafe fn enqueue_cached<'s>(&'s mut self, query:&AttentionQueryOutput<'_>,
        sink:CuteafdDeviceBuffer, bank:&BackboneCache<'_>,cache:&CacheAttention<'_>,
        selection:Option<&IndexSelectionOutput<'_>>)->Result<PendingDualAttention<'s,'a>> {
        ensure!(self.pending.is_none(),"dual attention submission already pending");
        ensure!(query.rows>0 && query.rows<=self.capacity && sink.bytes>=256
            && sink.device_id==self.halves[0].device.id,"dual cached attention input differs");
        let mut pending=PendingDualAttention { wave:Some(self),cold:[false;2],rows:query.rows,layer:query.layer };
        // Guard precedes every copy; metadata views can drop on error only after
        // the underlying streams have drained through this pending owner.
        let wave=pending.wave.as_deref_mut().unwrap();
        let stream=wave.halves[1].chain_stream();
        let peer=wave.halves[1].device;
        peer.run(|| unsafe { wave.storage.guard.acquire(stream) })?;
        let mirrored=unsafe { wave.storage.inputs.enqueue(bank,cache,stream)? };
        let selected=selection.map(|s|unsafe { wave.storage.inputs.selection(s,stream) }).transpose()?;
        peer.run(||unsafe { wave.storage.copies[1].launch(wave.storage.peer_sink.buffer,slice(sink,128,128),128,stream) })?;
        let (original,original_count)=cache.attention_requests_fixed()?;
        let (requests,count)=mirrored.attention_requests();
        let request_slices=[&original[..original_count],&requests[..count]];
        let selections=[selection,selected.as_ref()];
        let sinks=[slice(sink,0,128),wave.storage.peer_sink.buffer];
        for i in 0..2 {
            let device=wave.halves[i].device;
            pending.cold[i]=device.run(||unsafe {
                wave.halves[i].enqueue_split_query_prepared(query,i*32,&wave.storage.copies[i],
                    sinks[i],request_slices[i],selections[i],true,None)
            })?.is_none();
        }
        Ok(pending)
    }
    /// # Safety
    /// Both devices' query/cache/proposal/selection producers have completed or
    /// are ordered before these waves' streams. Requests have identical logical
    /// rows and reference their respective local replicas. Sinks are the matching
    /// checkpoint head halves. Retain every external allocation through complete
    /// or drop of the returned guard; this includes deferred cold preparation.
    pub unsafe fn enqueue<'s>(&'s mut self, query: &AttentionQueryOutput<'_>,
        sinks: [CuteafdDeviceBuffer; 2], requests: [&[AttentionRequest<'_>]; 2],
        selections: [Option<&IndexSelectionOutput<'_>>; 2]) -> Result<PendingDualAttention<'s,'a>> {
        ensure!(self.pending.is_none(),"dual attention submission already pending");
        ensure!(query.rows>0 && query.rows<=self.capacity, "dual attention rows exceed capacity");
        ensure!(requests[0].len()==requests[1].len() && requests[0].iter().zip(requests[1]).all(|(a,b)|
            a.window.binding==b.window.binding && a.positions==b.positions
            && a.source.map(|s|s.binding())==b.source.map(|s|s.binding())),
            "dual attention proposal identities differ");
        let mut pending = PendingDualAttention { wave: Some(self), cold: [false;2],
            rows: query.rows, layer: query.layer };
        let wave = pending.wave.as_deref_mut().unwrap();
        for i in 0..2 {
            let device = wave.halves[i].device;
            pending.cold[i] = device.run(|| unsafe {
                wave.halves[i].enqueue_split_query_prepared(query,i*32,&wave.storage.copies[i],
                    sinks[i],requests[i],selections[i],true,None)
            })?.is_none();
        }
        Ok(pending)
    }
}

pub(crate) struct PendingDualAttention<'s,'a> {
    wave: Option<&'s mut DualAttentionWave<'a>>,
    cold: [bool;2],
    rows: usize,
    layer: usize,
}
impl PendingDualAttention<'_,'_> {
    pub async fn complete(self) -> Result<QueuedSparseAttention> {
        unsafe { self.complete_then(|output,_|Ok(output)).await }
    }
    /// Queue consumers after gathering and wait once for the entire lane chain.
    /// # Safety
    /// The callback enqueues only on the supplied stream, on the output device.
    /// External consumer allocations must remain alive through this future's
    /// completion or cancellation, including callback errors. The callback must
    /// not expose GPU results before completion and must not suspend.
    pub async unsafe fn complete_then<T>(mut self,
        consume:impl FnOnce(QueuedSparseAttention,*mut c_void)->Result<T>)->Result<T> {
        let wave = self.wave.as_deref_mut().unwrap();
        // Both warmups were submitted before either wait. Warm graph hits never
        // enter this branch and require no host rendezvous between the halves.
        for i in 0..2 {
            if self.cold[i] {
                let device = wave.halves[i].device;
                unsafe { device.future(wave.halves[i].finish_prepare(None)).await?; }
            }
        }
        let owner = wave.halves[0].device;
        let peer = wave.halves[1].device;
        let stream = wave.halves[0].chain_stream();
        let library = owner.library;
        owner.run(|| unsafe {
            acquire_dual(&wave.storage.guard,stream)?;
            wave.storage.copies[0].launch_rows(wave.storage.output.buffer,wave.halves[0].storage.output.buffer,
                32*1024,self.rows,64*1024,32*1024,stream)
        })?;
        peer.run(|| unsafe { library.cuda_event_record(wave.storage.peer_done.raw,wave.halves[1].chain_stream()) })?;
        owner.run(|| unsafe {
            library.cuda_stream_wait_event(stream,wave.storage.peer_done.raw)?;
            wave.storage.copies[0].launch_rows(slice(wave.storage.output.buffer,32*1024,wave.storage.output.buffer.bytes-32*1024),
                wave.halves[1].storage.output.buffer,32*1024,self.rows,64*1024,32*1024,stream)
        })?;
        let values = slice(wave.storage.output.buffer,0,self.rows*64*1024);
        let result=owner.run(||consume(QueuedSparseAttention { values,rows:self.rows,layer:self.layer },stream))?;
        owner.future(wave.halves[0].wait_chain()).await?;
        self.wave = None;
        Ok(result)
    }
}
impl Drop for PendingDualAttention<'_,'_> {
    fn drop(&mut self) {
        if let Some(wave) = self.wave.as_deref_mut() {
            if let Err(error) = wave.drain() { crate::shared::decode_graph::fatal_drain(Err(error), "dual attention cancellation"); }
        }
    }
}
impl Drop for DualAttentionWave<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain() { crate::shared::decode_graph::fatal_drain(Err(error), "dual attention"); }
        let owner=self.halves[0].device;
        self.tails.retire_all();
        if let Err(error)=owner.run(||self.tails.drain_retired(||Ok(())).map(drop)) {
            tracing::error!(%error,"destroying dual attention continuation graph");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires compact attention, SM peer copies and two CUDA GPUs"]
    fn dual_attention_gathers_both_head_halves() -> Result<()> {
        let library=unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        for owner in 0..2 {
            let devices=[Device { library:&library,id:owner },Device { library:&library,id:1-owner }];
            let mut wave=DualAttentionWave::new(devices,4,DualAttentionWave::device_bytes(4)?)?;
            let mut buffers=Vec::new();
            for (i,device) in devices.into_iter().enumerate() {
                let values=Allocation::new(device,128*512, crate::shared::memory::TEST)?;
                let scales=Allocation::new(device,128*16, crate::shared::memory::TEST)?;
                let end=Allocation::new(device,8, crate::shared::memory::TEST)?;
                let sink=Allocation::new(device,32*4, crate::shared::memory::TEST)?;
                device.run(|| {
                    library.copy_h2d(scales.buffer,&vec![127;128*16])?;
                    library.copy_h2d(end.buffer,&[0;8])?;
                    let s=if i==0 { 0f32 } else { 3f32.ln() };
                    library.copy_h2d(sink.buffer,&s.to_ne_bytes().repeat(32))?;
                    library.copy_h2d(wave.halves[i].storage.query.buffer,&vec![0;4*32*1024])?;
                    library.copy_h2d(wave.halves[i].storage.replay_begins.buffer,&[0;32])?;
                    let metadata:Vec<u8>=(0..4).flat_map(|pos|
                        [0u64,0,4,pos,0,0,0,0,0,0].into_iter().flat_map(u64::to_ne_bytes)).collect();
                    library.copy_h2d(wave.halves[i].storage.metadata.buffer,&metadata)
                })?;
                buffers.push((values,scales,end,sink));
            }
            for (encoded,value,cancel) in [(0x38,1f32,false),(0x40,2f32,true),(0x40,2f32,false),(0,0f32,false)] {
                for i in 0..2 {
                    let (values,scales,end,sink)=&buffers[i];
                    devices[i].run(|| {
                        library.copy_h2d(values.buffer,&vec![encoded;128*512])?;
                        let half=&wave.halves[i];
                        let window=V41SparseWindow { values:values.buffer,scales:scales.buffer,
                            proposals:values.buffer,proposal_scales:scales.buffer,end:end.buffer,
                            proposal_capacity:128,replay_begins:Some(half.storage.replay_begins.buffer) };
                        unsafe { half.storage.kernel.launch(half.storage.query.buffer,sink.buffer,half.storage.metadata.buffer,None,
                            &window,None,half.storage.output.buffer,4,0,None,half.chain_stream()) }
                    })?;
                }
                let pending=PendingDualAttention { wave:Some(&mut wave),cold:[false;2],rows:4,layer:0 };
                if cancel { drop(pending); continue; }
                let output=runtime.block_on(pending.complete())?;
                let mut actual=vec![0;output.values.bytes];
                devices[0].run(||library.copy_d2h(&mut actual,output.values))?;
                for row in 0..4 { for half in 0..2 {
                    let n=(row+1) as f32;
                    let expected=value*n/(n+if half==0 {1.0} else {3.0});
                    for bytes in actual[(row*64+half*32)*1024..(row*64+half*32+32)*1024].chunks_exact(2) {
                        let observed=f32::from_bits(u32::from(u16::from_ne_bytes([bytes[0],bytes[1]]))<<16);
                        assert!((observed-expected).abs()<0.005,"owner={owner} row={row} half={half}: {observed} != {expected}");
                    }
                }}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::shared::memory::{chain::StageChain, device::Stream};
    fn run(guarded: bool) -> Result<()> {
        let lib=unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let devices=[Device { library:&lib,id:0 },Device { library:&lib,id:1 }];
        let storage=DualAttentionStorage::new(devices,4)?;
        let mut slots=[DualAttentionWave::on_storage(storage.clone())?,DualAttentionWave::on_storage(storage.clone())?];
        for (i,slot) in slots.iter_mut().enumerate() {
            slot.select_window(crate::shared::memory::row_window::RowWindow::within(i*4..i*4+4,8)?)?;
        }
        let mut data=Vec::new();
        for device in devices {
            let values=[Allocation::new(device,128*512, crate::shared::memory::TEST)?,Allocation::new(device,128*512, crate::shared::memory::TEST)?];
            let scales=Allocation::new(device,128*16, crate::shared::memory::TEST)?;let end=Allocation::new(device,8, crate::shared::memory::TEST)?;
            let sink=Allocation::new(device,128, crate::shared::memory::TEST)?;
            device.run(|| {
                lib.copy_h2d(values[0].buffer,&vec![0x38;128*512])?;
                lib.copy_h2d(values[1].buffer,&vec![0x40;128*512])?;
                lib.copy_h2d(scales.buffer,&vec![127;128*16])?;
                lib.copy_h2d(end.buffer,&[0;8])?;lib.copy_h2d(sink.buffer,&[0;128])
            })?;data.push((values,scales,end,sink));
        }
        let metadata:Vec<u8>=(0..4).flat_map(|pos|
            [0u64,0,4,pos,0,0,0,0,0,0].into_iter().flat_map(u64::to_ne_bytes)).collect();
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        let enqueue=|wave:&mut DualAttentionWave<'_>,seed:usize|->Result<()> {
            for i in 0..2 {
                let half=&wave.halves[i];let stream=half.chain_stream();
                let (values,scales,end,sink)=&data[i];
                devices[i].run(||unsafe {
                    acquire_dual(&wave.storage.guard,stream)?;
                    half.acquire_scratch()?;
                    crate::shared::memory::chain::join(&lib,stream)?;
                    let window=V41SparseWindow {values:values[seed].buffer,scales:scales.buffer,
                        proposals:values[seed].buffer,proposal_scales:scales.buffer,end:end.buffer,
                        proposal_capacity:128,replay_begins:Some(half.storage.replay_begins.buffer)};
                    half.storage.kernel.launch(half.storage.query.buffer,sink.buffer,half.storage.metadata.buffer,None,
                        &window,None,half.storage.output.buffer,4,0,None,stream)
                })?;
            }
            Ok(())
        };
        let initialize=|wave:&DualAttentionWave<'_>|->Result<()> {
            for i in 0..2 {devices[i].run(|| {
                lib.copy_h2d(wave.halves[i].storage.query.buffer,&vec![0;4*32*1024])?;
                lib.copy_h2d(wave.halves[i].storage.metadata.buffer,&metadata)?;
                lib.copy_h2d(wave.halves[i].storage.replay_begins.buffer,&[0;32])
            })?;} Ok(())
        };
        let mut private=DualAttentionWave::new(devices,4,DualAttentionWave::device_bytes(4)?)?;
        initialize(&private)?;enqueue(&mut private,0)?;
        let output=runtime.block_on(PendingDualAttention {wave:Some(&mut private),cold:[false;2],rows:4,layer:0}.complete())?;
        let mut want=vec![0;4*64*1024];lib.copy_d2h(&mut want,output.values)?;
        initialize(&slots[0])?;
        for seed in 0..2 {
            enqueue(&mut slots[seed],seed)?;
            runtime.block_on(PendingDualAttention {wave:Some(&mut slots[seed]),cold:[false;2],rows:4,layer:0}.complete())?;
        }
        let reader=Stream::new(devices[1])?;let saved=Allocation::new(devices[1],want.len(), crate::shared::memory::TEST)?;
        devices[1].run(||unsafe {lib.copy_peer_async(saved.buffer,storage.output.buffer,want.len(),reader.raw)})?;reader.drain()?;
        let chains=[StageChain::on_devices(&lib,&[0,1])?,StageChain::on_devices(&lib,&[0,1])?];
        let gate=devices[1].run(||unsafe {cuteafd_ffi::test_support::CudaStreamGate::new(&lib,reader.raw)})?;
        runtime.block_on(chains[0].handle().scope(async {
            enqueue(&mut slots[0],0)?;
            let output=PendingDualAttention {wave:Some(&mut slots[0]),cold:[false;2],rows:4,layer:0}.complete().await?;
            devices[1].run(||unsafe {
                crate::shared::memory::chain::join(&lib,reader.raw)?;
                lib.copy_peer_async(saved.buffer,output.values,want.len(),reader.raw)?;
                crate::shared::memory::chain::finish(&lib,reader.raw)
            })?;slots[0].release_scratch()
        }))?;
        SKIP_DUAL_ACQUIRE.with(|s|s.set(!guarded));
        super::super::SKIP_STORAGE_ACQUIRE.with(|s|s.set(!guarded));
        let queued=runtime.block_on(chains[1].handle().scope(async {
            enqueue(&mut slots[1],1)?;
            PendingDualAttention {wave:Some(&mut slots[1]),cold:[false;2],rows:4,layer:0}.complete().await?;
            anyhow::Ok(())
        }));
        SKIP_DUAL_ACQUIRE.with(|s|s.set(false));super::super::SKIP_STORAGE_ACQUIRE.with(|s|s.set(false));queued?;
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(if guarded {2} else {10});let mut done=false;
        while std::time::Instant::now()<deadline {
            if devices[0].run(||unsafe {lib.cuda_stream_query(storage.halves[0].stream.raw)})? {done=true;break;}
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        devices[1].run(|| {drop(gate);Ok(())})?;
        for chain in &chains {chain.drain()?;}reader.drain()?;
        assert_eq!(done,!guarded,"dual B completion before A reader release");
        let mut got=vec![0;want.len()];devices[1].run(||lib.copy_d2h(&mut got,saved.buffer))?;
        if guarded {assert_eq!(got,want);}else {assert_ne!(got,want,"unguarded dual gather did not clobber");}
        Ok(())
    }
    #[test]
    #[ignore = "requires compact sparse attention and two GPUs"]
    fn dual_shared_arena_preserves_gated_peer_reader()->Result<()> {for _ in 0..3 {run(true)?;} Ok(())}
    #[test]
    #[ignore = "requires compact sparse attention and two GPUs"]
    fn dual_without_guards_clobbers_gated_peer_reader()->Result<()> {for _ in 0..3 {run(false)?;} Ok(())}
    #[test]
    fn dual_storage_bytes_equal_planned_allocations()->Result<()> {
        for unit in [4096,2752,2048] {
            let half=SparseAttentionStorage::<'_,32>::allocation_sizes(unit)?.into_iter().sum::<usize>();
            assert_eq!(DualAttentionStorage::storage_bytes(unit)?,
                [half+unit*65536,half+unit*(528+4*288+2048)+128]);
        } Ok(())
    }
}
