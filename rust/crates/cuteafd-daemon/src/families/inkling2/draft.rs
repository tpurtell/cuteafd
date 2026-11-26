//! Inkling binding of shared recurrent speculation, exact local marks and FIR
//! verifier storage. No scheduling, sampling policy or private exchange here.
use super::{draft_taps::TapBuffer,head::Head,layers::Kernels,verify::Verify,weights::ProjectionWeight};
use crate::shared::{memory::device::Stream,speculation::{recurrent::Drafter,recurrent_forward::{Rank,ProgramsForRank},
    recurrent_weights::Weights,recurrent_cache::{ContextBank,ContextState},recurrent_snapshots::Marks}};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::{programs::Programs,CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::{families::inkling2::draft::DraftLayout,placement::planner::ResolvedPlan};
use std::{cell::RefCell,rc::Rc};

fn snapshot_buffers(config:&cuteafd_loader::formats::block_dspark::recurrent::RecurrentBlockConfig,
    buffers:&std::collections::BTreeMap<String,crate::shared::memory::device::Allocation<'_>>,
    marks:Option<Buffer>)->Result<Vec<Buffer>>{
    let mut output=marks.into_iter().collect::<Vec<_>>();
    for (layer,policy) in config.context.iter().enumerate(){
        if matches!(policy,cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full){
            for plane in ["k","v"]{output.push(buffers.get(&format!("cache.layer.{layer}.{plane}"))
                .context("Inkling snapshot full KV owner missing")?.buffer);}
        }
    }Ok(output)
}

pub(crate) struct Binding<'a>{
    pub drafter:Rc<Drafter<'a>>,
    pub marks:Vec<Marks<'a>>,
    pub verify:Vec<Verify<'a>>,
    pub prefill:Vec<TapBuffer<'a>>,
    pub decode:TapBuffer<'a>,
    pub layout:DraftLayout,
}
impl crate::shared::prefix::cuda_copy::SnapshotOwner for Binding<'_>{
    fn snapshot_buffers(&self)->Result<Vec<Buffer>>{
        let mut output=Vec::new();
        for (index, rank) in self.ranks().enumerate() {
            output.extend(snapshot_buffers(&self.layout.config,&rank.buffers,
                self.marks.get(index).map(|marks|marks.backing().buffer))?);
        }
        Ok(output)
    }
}
impl<'a> Binding<'a>{
    pub fn ranks(&self) -> impl Iterator<Item=&Rank<'a>> {
        std::iter::once(&self.drafter.lead).chain(self.drafter.split.as_ref().map(|split| &**split.peer))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new(programs:&'a Programs<'a>,streams:Vec<Rc<Stream<'a>>>,weights:Vec<Weights<'a>>,layout:DraftLayout,
        admitted:&ResolvedPlan,target_head:ProjectionWeight<'_, 'a>,head:&Head<'a>,target_kernels:&Kernels<'a>,
        peer:Option<super::head::PeerHead<'_, 'a>>,
        prefill_rows:usize,waves:usize,decode_rows:usize,mark_slots:usize)->Result<Self>{
        let count=streams.len();
        ensure!(matches!(count,1|2) && weights.len()==count && layout.weights.len()==count && layout.runtime.len()==count,
            "Inkling target and draft rank counts differ");
        let check=|id:&str,bytes:usize,rank:usize|->Result<()>{
            let owner=admitted.facts.allocations.iter().find(|owner|owner.id==id).with_context(||format!("unadmitted drafter owner {id}"))?;
            ensure!(owner.bytes==bytes as u64 && owner.location==cuteafd_loader::placement::planner::Location::coordinator(rank as u8),
                "drafter runtime owner differs from admission {id}");Ok(())
        };
        let mut ranks=Vec::new();
        for (index,(stream,weights)) in streams.iter().zip(weights).enumerate(){
            let runtime=&layout.runtime[index];
            check(layout.weights[index].arena.id(),layout.weights[index].arena.bytes(),index)?;
            for (name,bytes) in &runtime.buffers{check(&format!("inkling2.draft.{index}.{name}"),*bytes,index)?;}
            if mark_slots>0{check(&format!("inkling2.draft.{index}.marks"),runtime.window_mark_bytes*mark_slots,index)?;}
            ranks.push(stream.device.run(||{
                let kernels=ProgramsForRank::load(programs,&weights,&layout.config,runtime,stream.device)?;
                let rank=Rank::allocate(weights,Rc::clone(stream),layout.config.clone(),runtime.clone(),kernels)?;
                rank.initialize_derived()?;
                if index == runtime.rank.lead {
                    if let Some(peer) = peer {
                        fold_split(programs,&rank,head,target_kernels,target_head,peer)?;
                    } else { match target_head {
                        ProjectionWeight::Bf16(w) => rank.fold_head(w.buffer)?,
                        weight => {
                            let (stored, stage, transpose) = head.compact_stages()?;
                            let h = rank.config.transformer.hidden;
                            rank.fold_compact_head(programs, stored, stage, transpose, |input, output, rows|
                                target_kernels.project_compact(stream, input, weight, output, rows, h, stored))?;
                        }
                    } }
                }
                stream.drain()?;
                Ok(rank)
            })?);
        }
        for (name,bytes) in &layout.target{check(&format!("inkling2.draft.target.{name}"),*bytes,0)?;}
        for (name,bytes) in &layout.peer{check(&format!("inkling2.draft.peer.{name}"),*bytes,1)?;}
        for rank in 0..count {
            for (name,bytes) in &layout.exchange {check(&format!("inkling2.draft.{rank}.{name}"),*bytes,rank)?;}
        }
        let runtime=&layout.runtime[0];
        let context=ContextBank{state:RefCell::new(ContextState::new(runtime.slots,
            usize::try_from(admitted.served_context())?,runtime.cache.target_tokens,0x494e4b4d5450)?)};
        let mut ranks=ranks.into_iter();
        let lead=ranks.next().context("Inkling lead draft rank absent")?;
        let drafter=Rc::new(if let Some(peer)=ranks.next(){Drafter::pair(lead,peer,context)?}else{Drafter::whole(lead,context)?});
        let local=std::iter::once(&drafter.lead).chain(drafter.split.as_ref().map(|split|&**split.peer));
        let marks=if mark_slots==0{Vec::new()}else{
            local.map(|rank|rank.stream.device.run(||Marks::new(rank,mark_slots))).collect::<Result<_>>()?
        };
        let width=9216+2048/count;
        let verify=streams.iter().map(|stream|stream.device.run(||Verify::new(programs,Rc::clone(stream),runtime.slots,width,decode_rows)))
            .collect::<Result<Vec<_>>>()?;
        let stream=&streams[0];
        let prefill=(0..waves).map(|_|TapBuffer::new(Rc::clone(stream),&layout.config,prefill_rows)).collect::<Result<Vec<_>>>()?;
        let decode=TapBuffer::new(Rc::clone(stream),&layout.config,decode_rows)?;
        stream.drain()?;
        Ok(Self{drafter,marks,verify,prefill,decode,layout})
    }
    pub fn select(&self, head:&Head<'a>, kernels:&Kernels<'a>, weight:ProjectionWeight<'_, 'a>,
        peer:Option<super::head::PeerHead<'_, 'a>>, sequences:usize, anchors:Buffer) -> Result<Buffer> {
        let rank=&self.drafter.lead;
        if let Some(peer) = peer {
            return select_split(rank,head,kernels,weight,peer,sequences,anchors);
        }
        if let ProjectionWeight::Bf16(w)=weight { return rank.recurrent_tokens(sequences,anchors,w.buffer); }
        let (stored, stage, narrowed)=head.compact_stages()?;
        let (h,n)=(rank.config.transformer.hidden,rank.config.padded_vocab);
        let stream=&rank.stream;
        let lib=stream.device.library;
        rank.recurrent_tokens_measured(sequences,anchors,Some(64),|input,output,rows| {
            ensure!(n<=stored && rows*stored*2<=narrowed.bytes && rows*n*2<=output.bytes,
                "compact recurrent head staging extent");
            kernels.project_compact(stream,input,weight,stage,rows,h,stored)?;
            // SAFETY: these idle verifier stages are disjoint and exclusively
            // borrowed on the same ordered stream; only physical padding is removed.
            unsafe {
                kernels.narrow.launch(stage.ptr,narrowed.ptr,rows*stored,stream.raw)?;
                lib.copy_d2d_2d_async(output,n*2,narrowed,stored*2,n*2,rows,stream.raw)
            }
        },|_,_|Ok(()))
    }

    pub fn taps(&self,decode:bool,unit:usize)->Result<&TapBuffer<'a>>{
        if decode{Ok(&self.decode)}else{self.prefill.get(unit).context("drafter prefill tap unit outside admission")}
    }
    pub fn update(&self,taps:&TapBuffer<'a>,slot:usize,start:usize,first:usize,rows:usize)->Result<()> {
        let pitch=self.layout.config.taps.len()*self.layout.config.transformer.hidden*2;
        for offset in (0..rows).step_by(self.layout.runtime[0].context_rows){
            let count=(rows-offset).min(self.layout.runtime[0].context_rows);
            let position=(first+offset)*pitch;
            let output=taps.output.buffer;
            ensure!(position.checked_add(count*pitch).is_some_and(|end|end<=output.bytes),"drafter target taps outside owner");
            self.drafter.append_context(slot,start+offset,
                Buffer{ptr:output.ptr.wrapping_byte_add(position),bytes:count*pitch,..output},count)?;
        }Ok(())
    }
}

fn fold_split<'a>(programs:&'a Programs<'a>, rank:&Rank<'a>, head:&Head<'a>, kernels:&Kernels<'a>,
    lead:ProjectionWeight<'_, 'a>, peer:super::head::PeerHead<'_, 'a>)->Result<()> {
    let peer_weight=peer.target.projection("model.llm.unembed.weight")?;
    let (h,n)=(rank.config.transformer.hidden,rank.config.padded_vocab);
    let left=cuteafd_loader::shared::vocabulary::shard(n,128,0)?.len();
    let stream=head.stream();
    let lib=stream.device.library;
    let handoff=head.handoff().context("Inkling draft split handoff missing")?;
    let peer_stream=head.peer_stream().context("Inkling draft peer stream missing")?;
    match (lead,peer_weight) {
        (ProjectionWeight::Bf16(a),ProjectionWeight::Bf16(b)) => {
            let stage=head.local_stage()?;
            for (first,width,source) in [(0,left,a.buffer),(left,n-left,b.buffer)] {
                for row in (0..width).step_by(2048) {
                    let count=(width-row).min(2048);
                    let weights=super::cache::view(source,row*h*2,count*h*2)?;
                    let weights=if first==0 { weights } else {
                        let output=super::cache::view(stage,0,count*h*2)?;
                        // SAFETY: startup is serialized, both arenas persist through
                        // fold, and no online work can mutate or reuse this tile.
                        unsafe { lib.copy_peer_async(output,weights,weights.bytes,stream.raw)?; }
                        output
                    };
                    rank.fold_bf16_rows(weights,first+row,count)?;
                }
            }
            Ok(())
        }
        (a,b) => {
            let (stored,stage,transpose)=head.compact_stages()?;
            let remote=head.peer_weight().context("Inkling compact peer width missing")?;
            let (remote_input,remote_stage)=head.peer_stages()?;
            rank.fold_compact_shard(programs,0,left,stored,stage,transpose,|input,output,rows|
                kernels.project_compact(stream,input,a,output,rows,h,stored))?;
            rank.fold_compact_shard(programs,left,n-left,remote,stage,transpose,|input,output,rows| {
                handoff.project(input,remote_input,|| Ok(()),
                    || peer.kernels.project_compact(peer_stream,remote_input,b,remote_stage,rows,h,remote),
                    || handoff.copy_columns(output,remote_stage,rows,remote*4,remote*4,remote*4))
            })
        }
    }
}

fn select_split<'a>(rank:&Rank<'a>, head:&Head<'a>, kernels:&Kernels<'a>, weight:ProjectionWeight<'_, 'a>,
    peer:super::head::PeerHead<'_, 'a>, sequences:usize, anchors:Buffer)->Result<Buffer> {
    use cuteafd_ffi::draft_projection::DraftProjections;
    let (h,n)=(rank.config.transformer.hidden,rank.config.padded_vocab);
    let left=cuteafd_loader::shared::vocabulary::shard(n,128,0)?.len();
    let right=n-left;
    let handoff=head.handoff().context("Inkling draft split handoff missing")?;
    let peer_stream=head.peer_stream().context("Inkling draft peer stream missing")?;
    let stream=head.stream();
    let (remote_input,remote_stage)=head.peer_stages()?;
    let peer_weight=peer.target.projection("model.llm.unembed.weight")?;
    let local=head.local_stage()?;
    rank.recurrent_tokens_measured(sequences,anchors,Some(64),|input,out,rows| {
        let projection=|remote:bool| -> Result<(Buffer,usize)> {
            let (stream,kernels,weight,input,stage,width,programs):(_,_,_,_,_,_,Option<&DraftProjections<'_>>)=if remote {
                (peer_stream,peer.kernels,peer_weight,remote_input,remote_stage,right,peer.kernels.draft_head.as_ref())
            } else { (stream,kernels,weight,input,local,left,kernels.draft_head.as_ref()) };
            if let ProjectionWeight::Bf16(w)=weight {
                // SAFETY: rank-local selected weight shard and input/stage outlive the launch.
                unsafe { programs.context("Inkling recurrent BF16 vocabulary shard not bound")?
                    .launch_checked(input.ptr,w.buffer.ptr,stage,rows,h,width,stream.raw)?; }
                Ok((stage,width))
            } else {
                let (stored,raw)=if remote { (head.peer_weight().unwrap(),remote_stage) }
                    else { let (stored,b,_)=head.compact_stages()?; (stored,b) };
                // Match the whole head's single F32 to BF16 narrowing before gathering.
                kernels.project_compact(stream,input,weight,raw,rows,h,stored)?;
                let dst=if remote { head.peer_draft_output()? } else { stage };
                // SAFETY: F32 and BF16 stages are disjoint and stream-owned.
                unsafe { kernels.narrow.launch(raw.ptr,dst.ptr,rows*stored,stream.raw)?; }
                Ok((dst,stored))
            }
        };
        let a=std::cell::Cell::new(None);
        let b=std::cell::Cell::new(None);
        handoff.project(input,remote_input,|| {a.set(Some(projection(false)?));Ok(())},
            || {b.set(Some(projection(true)?));Ok(())},|| {
                let (a,ap)=a.get().context("Inkling draft local shard missing")?;
                let (b,bp)=b.get().context("Inkling draft peer shard missing")?;
                handoff.copy_columns(out,a,rows,n*2,ap*2,left*2)?;
                handoff.copy_columns(super::cache::view(out,left*2,out.bytes-left*2)?,b,rows,n*2,bp*2,right*2)
            })
    },|_,_| Ok(()))
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::shared::{memory::{device::{Allocation,Device},Tag},prefix::{CudaCopyEngine,cuda_copy::{range,SnapshotOwner}},
        speculation::block_dspark::WindowSnapshots};
    use cuteafd_hostcache::{copy::{CopyEngine,Stream as CopyStream},pool::{PinnedMemory,HostRange}};
    struct SnapshotFixture<'a>{config:cuteafd_loader::formats::block_dspark::recurrent::RecurrentBlockConfig,
        buffers:std::collections::BTreeMap<String,Allocation<'a>>,marks:WindowSnapshots<'a>}
    impl SnapshotOwner for SnapshotFixture<'_>{fn snapshot_buffers(&self)->Result<Vec<Buffer>>{
        snapshot_buffers(&self.config,&self.buffers,Some(self.marks.backing().buffer))
    }}
    #[test]
    fn full_context_attachment_is_host_stored_and_restored_byte_exact()->Result<()> {
        use cuteafd_engine::prefix::{DraftPagePool, SnapshotKind};
        use cuteafd_hostcache::{cache::{DevicePage, DeviceSnapshot, HostCache, RestoreOutcome,
            RestoreTarget, SkipReason, StoreOutcome}, config::Config, pool::Layout,
            snapshot::{DevicePageId, SnapshotMeta}, DRAFT_PAGE_BANK};
        let fixture=cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library=fixture.load()?;fixture.immediate_copies(&library)?;
        let device=Device{library:&library,id:0};let stream=Rc::new(Stream::new(device)?);
        let config=cuteafd_loader::formats::block_dspark::recurrent::RecurrentBlockConfig::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        let mut buffers=std::collections::BTreeMap::new();
        for (layer,policy) in config.context.iter().enumerate(){if matches!(policy,
            cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full){
            for plane in ["k","v"]{buffers.insert(format!("cache.layer.{layer}.{plane}"),
                Allocation::new(device,2*64*32,Tag::drafter_kv("test/host-full-pages"))?);}
        }}
        let owner=Rc::new(SnapshotFixture{config,buffers,marks:WindowSnapshots::new(&stream,vec![256],2)?});
        let mut pool=DraftPagePool::new(128,1,128,77)?;
        let binding=pool.bind(0,0)?;
        let reservation=pool.reserve(binding,0,64)?;
        pool.commit(reservation,64)?;
        let (saved,copies)=pool.capture(binding)?;
        assert!(copies.is_empty());
        let attachment=pool.attachment(saved)?;
        let mark=owner.marks.mark(0)?;
        let full=owner.buffers.values().map(|buffer|Buffer{bytes:64*32,..buffer.buffer}).collect::<Vec<_>>();
        let originals=full.iter().enumerate().map(|(i,b)|vec![i as u8+31;b.bytes]).collect::<Vec<_>>();
        for (buffer,bytes) in full.iter().zip(&originals){library.copy_h2d(*buffer,bytes)?;}
        let mark_bytes=vec![73;mark.bytes];library.copy_h2d(mark,&mark_bytes)?;
        let mut pages=Default::default();
        let bank:&mut [Vec<DevicePage>;cuteafd_hostcache::PAGE_BANKS]=&mut pages;
        bank[DRAFT_PAGE_BANK]=attachment.pages.iter().map(|page|DevicePage{
            id:DevicePageId{compressor:DRAFT_PAGE_BANK as u8,page:page.index,generation:page.generation},
            segments:full.iter().copied().map(range).collect()}).collect();
        let snapshot=DeviceSnapshot{meta:SnapshotMeta{kind:SnapshotKind::Prompt,tokens:vec![1;64],end:64,has_draft:false},
            pages,tail:vec![range(mark)],draft:None,scores:vec![]};
        let config=Config{bytes:1<<20,chunk_bytes:8192,min_tokens:1,..Default::default()};
        let engine=||CudaCopyEngine::registered_retained(&library,vec![],vec![owner.clone()]);
        let layout=Layout::family(1,mark.bytes,0);
        let mut broken=HostCache::new(config.clone(),layout,engine()?)?;
        assert!(matches!(broken.store(&snapshot,attachment.clone()),StoreOutcome::Skipped(SkipReason::Malformed)));
        let mut host=HostCache::new(config,layout.with_draft_pages(full.iter().map(|b|b.bytes).sum()),engine()?)?;
        let ticket=match host.store(&snapshot,attachment.clone()) {
            StoreOutcome::Issued(ticket)=>ticket,other=>anyhow::bail!("full draft attachment was not stored: {other:?}"),
        };
        host.engine_mut().synchronize(CopyStream::Store)?;
        assert!(host.tick().completed.contains(&ticket));
        assert_eq!(host.metrics().stores_completed,1);
        for buffer in &full{library.copy_h2d(*buffer,&vec![0;buffer.bytes])?;}
        library.copy_h2d(mark,&vec![0;mark.bytes])?;
        let hit=host.lookup(&snapshot.meta.tokens).context("host attachment missing")?;
        assert!(matches!(host.restore(hit.key,&RestoreTarget{pages:snapshot.pages.clone(),tail:snapshot.tail.clone(),
            draft:None,scores:vec![]}),RestoreOutcome::Done{..}));
        for (buffer,expected) in full.iter().zip(&originals){let mut actual=vec![0;buffer.bytes];library.copy_d2h_quiescent(&mut actual,*buffer)?;assert_eq!(&actual,expected);}
        let mut actual=vec![0;mark.bytes];library.copy_d2h_quiescent(&mut actual,mark)?;assert_eq!(actual,mark_bytes);
        pool.check_attachment(&attachment)?;
        Ok(())
    }

    #[test]
    fn registered_draft_marks_and_full_pages_restore_exact_bytes()->Result<()>{
        let fixture=cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library=fixture.load()?;fixture.immediate_copies(&library)?;
        let device=Device{library:&library,id:0};let stream=Rc::new(Stream::new(device)?);
        let config=cuteafd_loader::formats::block_dspark::recurrent::RecurrentBlockConfig::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        let mut buffers=std::collections::BTreeMap::new();
        for (layer,policy) in config.context.iter().enumerate(){if matches!(policy,
            cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full){
            for plane in ["k","v"]{buffers.insert(format!("cache.layer.{layer}.{plane}"),
                Allocation::new(device,2*64*32,Tag::drafter_kv("test/full-pages"))?);}
        }}
        let owner=Rc::new(SnapshotFixture{config,buffers,marks:WindowSnapshots::new(&stream,vec![256],2)?});
        let registered=owner.snapshot_buffers()?;
        let contains=|segment:Buffer|registered.iter().any(|buffer|buffer.ptr as usize<=segment.ptr as usize
            && segment.ptr as usize+segment.bytes<=buffer.ptr as usize+buffer.bytes);
        let mut segments=(0..2).map(|slot|owner.marks.mark(slot)).collect::<Result<Vec<_>>>()?;
        for buffer in owner.buffers.values(){for page in 0..2{segments.push(Buffer{
            ptr:buffer.buffer.ptr.wrapping_byte_add(page*64*32),bytes:64*32,..buffer.buffer});}}
        assert!(segments.iter().copied().all(contains));
        let originals=segments.iter().enumerate().map(|(index,segment)|vec![index as u8+17;segment.bytes]).collect::<Vec<_>>();
        for (segment,bytes) in segments.iter().zip(&originals){library.copy_h2d(*segment,bytes)?;}
        let mut engine=CudaCopyEngine::registered_retained(&library,vec![],vec![owner.clone()])?;
        let chunk=engine.allocate_chunk(segments.iter().map(|segment|segment.bytes).sum())?;
        let mut offset=0;
        let copies=segments.iter().map(|segment|{let host=HostRange{chunk:chunk.id,offset,bytes:segment.bytes};offset+=segment.bytes;(range(*segment),host)}).collect::<Vec<_>>();
        engine.d2h_many(CopyStream::Store,&copies)?;engine.synchronize(CopyStream::Store)?;
        for segment in &segments{library.copy_h2d(*segment,&vec![0;segment.bytes])?;}
        engine.h2d_many(CopyStream::Restore,&copies.iter().map(|(device,host)|(*host,*device)).collect::<Vec<_>>())?;
        engine.synchronize(CopyStream::Restore)?;
        for (segment,expected) in segments.iter().zip(&originals){let mut actual=vec![0;segment.bytes];library.copy_d2h_quiescent(&mut actual,*segment)?;assert_eq!(&actual,expected);}
        let weak=Rc::downgrade(&owner);drop(owner);assert!(weak.upgrade().is_some());
        engine.release_chunk(chunk)?;drop(engine);assert!(weak.upgrade().is_none());Ok(())
    }
}
