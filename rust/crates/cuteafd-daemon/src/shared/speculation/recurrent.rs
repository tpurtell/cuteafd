//! One target-independent recurrent block drafter over shared cache/TP2 owners.
use super::{recurrent_forward::Rank,recurrent_cache::ContextBank,dflash::{Draft,DraftSeq},gqa_tp2::GqaHeadSplit};
use anyhow::{ensure,Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use std::{cell::Cell,rc::Rc};

pub(crate) struct Drafter<'a> {
    // Every producer drains in Drop before rank/peer/cache storage retires.
    pub lead:Rank<'a>,
    pub split:Option<GqaHeadSplit<'a,Rank<'a>>>,
    pub context:ContextBank,
    live:Cell<bool>,
    /// Read each drafted token's candidate probability back (policy or trace).
    scores:Cell<bool>,
}
impl<'a> Drafter<'a> {
    pub fn set_score_readback(&self,enabled:bool){ self.scores.set(enabled); }
    pub fn whole(lead:Rank<'a>,context:ContextBank)->Result<Self>{
        ensure!(lead.layout.rank.ranks==1,"whole recurrent constructor requires one rank");
        Ok(Self{lead,split:None,context,live:Cell::new(true),scores:Cell::new(false)})
    }
    pub fn pair(lead:Rank<'a>,peer:Rank<'a>,context:ContextBank)->Result<Self>{
        use crate::shared::{memory::Tag,peer_split::{PeerExchange,RankDevice}};
        ensure!(lead.layout.rank.ranks==2 && lead.layout.rank.rank==0 && lead.layout.rank.lead==0
            && peer.layout.rank.ranks==2 && peer.layout.rank.rank==1 && peer.layout.rank.lead==0,
            "recurrent TP2 needs lead rank 0 and peer rank 1");
        let streams=[&lead.stream,&peer.stream];
        let endpoints=streams.map(|s|RankDevice{device:s.device.id,stream:s.raw});
        let bytes=lead.buffer("step.partial")?.bytes;
        let slot = lead.buffers.get("step.partial_f32").map_or(bytes, |owner| owner.buffer.bytes);
        ensure!(peer.buffers.get("step.partial_f32").map_or(bytes, |owner| owner.buffer.bytes) == slot,
            "recurrent TP2 partial storage differs");
        let exchange=Rc::new(lead.stream.device.run(||PeerExchange::new_abortable(
            lead.stream.device.library,endpoints,1,slot,slot))?);
        let device=peer.stream.device;
        let split=GqaHeadSplit::new(device.own(||Ok(Box::new(peer)))?,exchange,0,bytes,
            [Tag::drafter_workspace("recurrent/tp2-reduce");2])?;
        Ok(Self{lead,split:Some(split),context,live:Cell::new(true),scores:Cell::new(false)})
    }
    pub fn drain(&self)->Result<()> {
        self.lead.stream.drain()?;
        if let Some(split)=&self.split{split.peer.stream.drain()?;}
        Ok(())
    }
    pub fn append_context(&self,slot:usize,start:usize,features:Buffer,rows:usize)->Result<()> {
        self.context.append_ranked(&self.lead,self.split.as_ref(),slot,start,features,rows)
    }
    pub fn append_context_batch(&self,features:Buffer,spans:&[super::recurrent_cache::ContextSpan])->Result<()> {
        self.context.append_batch_ranked(&self.lead,self.split.as_ref(),features,spans)
    }
    pub fn bind_slot(&self,slot:usize,origin:usize,admit_end:Option<usize>)->Result<()> {
        self.drain()?;
        let admit=admit_end.map(|end|(end,self.lead.config.transformer.block));
        self.context.state.borrow_mut().bind_slot(slot,origin,admit)?;
        Ok(())
    }
    pub fn reset_context(&self,slot:usize,origin:usize)->Result<()> {
        self.bind_slot(slot,origin,None)
    }
    pub fn frontier(&self,slot:usize)->Result<(usize,usize)>{
        let state=self.context.state.borrow();Ok(state.pool.frontier(state.binding(slot)?)?)
    }
    pub fn draft(&self,sequences:&[DraftSeq],embedding:Buffer,embed_norm:Buffer,select:impl FnOnce(usize,Buffer)->Result<Buffer>)->Result<Vec<Draft>> {
        ensure!(self.live.get(),"recurrent drafter is terminal");
        let result=(||{
            let pages=self.context.state.borrow().pages(sequences)?;
            let anchors=sequences.iter().map(|seq|seq.anchor.to_le_bytes()).flatten().collect::<Vec<_>>();
            let lib=self.lead.stream.device.library;let stream=self.lead.stream.raw;
            // SAFETY: permanent metadata owner and stream retain staging; no
            // host hop occurs for recurrent state, features or intermediate IDs.
            unsafe{self.lead.upload("step.ids",&anchors)?;}
            self.lead.initialize_block(self.lead.buffer("step.ids")?,embedding,embed_norm,sequences.len())?;
            for (index,seq) in sequences.iter().enumerate(){
                let source=self.lead.buffer("cache.last_features")?;
                let destination=self.lead.buffer("head.f_rnn")?;
                let width=self.lead.config.transformer.hidden*2;
                // SAFETY: exact persistent last-feature row copied into packed
                // live head sequence order, with both owners retained.
                unsafe{lib.copy_d2d_async(Buffer{ptr:destination.ptr.wrapping_byte_add(index*width),bytes:width,..destination},
                    Buffer{ptr:source.ptr.wrapping_byte_add(seq.slot*width),bytes:width,..source},width,stream)?;}
            }
            match &self.split {
                Some(split)=>super::recurrent_tp2::forward(&self.lead,split,sequences,&pages)?,
                None=>self.lead.forward_whole(sequences,&pages)?,
            }
            let selected=select(sequences.len(),self.lead.buffer("step.ids")?)?;
            self.lead.stream.drain()?;
            let count=sequences.len()*self.lead.config.predictions()*4;
            let mut bytes=vec![0;count];
            self.lead.stream.copy_d2h(&mut bytes,Buffer{bytes:count,..selected})?;
            let tokens=bytes.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            let predictions=self.lead.config.predictions();
            // Candidate probability per drafted token (exp of the argmax log-softmax),
            // read only while the policy or a speculation trace uses it.
            let confidence=if self.scores.get() {
                self.lead.stream.copy_d2h(&mut bytes,Buffer{bytes:count,..self.lead.buffer("head.logprob")?})?;
                bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap()).exp()).collect::<Vec<_>>()
            } else { Vec::new() };
            Ok(tokens.chunks_exact(predictions).enumerate().map(|(index,tokens)|Draft{tokens:tokens.to_vec(),features:vec![],
                confidence:confidence.get(index*predictions..(index+1)*predictions).map_or_else(Vec::new,<[f32]>::to_vec)}).collect())
        })();
        if result.is_err(){
            self.live.set(false);
            if let Some(split)=&self.split {
                let drained=(||{split.exchange.publish_abort()?;split.exchange.drain_compute()})();
                split.exchange.finish_terminal(drained.is_ok());
                crate::shared::decode_graph::fatal_drain(drained,"recurrent drafter failed peer step");
            }
            self.lead.stream.drain_or_abort("recurrent drafter failed step");
        }
        result
    }
}
/// The engine's terminal path owns a TP2 drafter's peer waits like its own (#2543).
impl crate::shared::peer_split::TerminalOwner for Drafter<'_> {
    fn publish_abort(&self) -> Result<()> {
        self.live.set(false);
        self.split.as_ref().map_or(Ok(()), |split| split.exchange.publish_abort())
    }
    fn drain_terminal(&self) -> Result<()> {
        if let Some(split) = &self.split { crate::shared::peer_split::TerminalOwner::drain_terminal(&*split.exchange)?; }
        self.drain()
    }
    fn finish_terminal(&self, drained: bool) {
        if let Some(split) = &self.split { crate::shared::peer_split::TerminalOwner::finish_terminal(&*split.exchange, drained); }
    }
}
impl Drop for Drafter<'_> {
    fn drop(&mut self){
        self.live.set(false);
        self.lead.stream.drain_or_abort("recurrent drafter retirement");
        if let Some(split)=&self.split{split.peer.stream.drain_or_abort("recurrent peer retirement");}
    }
}
