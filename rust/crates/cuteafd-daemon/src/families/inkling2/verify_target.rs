//! Inkling adapter of the shared checkpoint/verify/accepted-prefix contract.
//! The shared scheduler owns selection; this adapter commits FIR histories once.
use super::{cache::Placement,engine::Execution,verify::{Submitted,TargetMark}};
use crate::shared::{prefill_pipeline::{Unit,UnitSegment},speculation::target::{AcceptedPrefix,
    BlockVerifyTarget,HiddenTap,TapSite,VerifyOutput,VerifyRequest}};
use anyhow::{ensure,Context,Result};

pub(crate) struct Target<'e,'a>{
    pub execution:&'e mut Execution<'a>,
    pub placements:&'e mut [Placement],
    taps:Vec<HiddenTap>,
}
impl<'e,'a> Target<'e,'a>{
    pub fn new(execution:&'e mut Execution<'a>,placements:&'e mut [Placement])->Self{
        Self{execution,placements,taps:Vec::new()}
    }
}
impl<'a> BlockVerifyTarget for Target<'_,'a>{
    type Mark=TargetMark<'a>;
    fn checkpoint(&mut self,slots:&[usize])->Result<Self::Mark>{
        self.execution.require_live()?;
        ensure!(slots.len()==self.placements.len() && slots.iter().zip(self.placements.iter()).all(|(&slot,p)|
            slot==p.slot && p.len==p.state_len),"Inkling checkpoint differs from committed placements");
        let draft=self.execution.draft.as_ref().context("Inkling speculation was not admitted")?;
        let starts=self.placements.iter().map(|p|(p.slot,p.len)).collect::<Vec<_>>();
        let conv=draft.verify.iter().zip(self.execution.request_ranks()).map(|(verify,requests)|
            requests.decode.stream.device.run(||verify.checkpoint(requests.cache.conv_backing(),&starts)))
            .collect::<Result<Vec<_>>>()?;
        Ok(TargetMark{conv,placements:self.placements.to_vec()})
    }
    fn verify<'e>(&'e mut self,mark:&Self::Mark,request:VerifyRequest<'_>)->Result<VerifyOutput<'e>>{
        request.validate(self.execution.cfg.layers,self.execution.cfg.unpadded_vocab_size,&mark.conv.first().context("Inkling verify lead mark missing")?.slots)?;
        let draft=self.execution.draft.as_ref().context("Inkling drafter missing")?.clone();
        ensure!(request.tap_sites==draft.layout.config.taps && request.rows.len()<=self.execution.requests.decode.layout.rows
            && request.rows.iter().all(|row|row.token.is_some()),"Inkling verify requires admitted native sites and active packed rows");
        let mut tokens=Vec::<Vec<u32>>::new();let mut submitted=Vec::new();let mut segments=Vec::new();
        let mut first=0;
        let decode_rows=self.execution.requests.decode.layout.rows;
        for (index,p) in mark.placements.iter().enumerate(){
            let rows=request.rows.iter().filter(|row|row.slot==p.slot).collect::<Vec<_>>();
            ensure!((1..=decode_rows).contains(&rows.len()),"Inkling verify member exceeds decode rows");
            let source=rows.iter().map(|row|row.token.unwrap()).collect::<Vec<_>>();
            ensure!(request.rows[first..first+rows.len()].iter().all(|row|row.slot==p.slot),"Inkling verify members must be contiguous");
            segments.push(UnitSegment{sequence:index,rows:first..first+rows.len(),tokens:p.len..p.len+rows.len(),
                prompt_end:false,snapshot:false,head:true});
            submitted.push(Submitted{slot:p.slot,start:p.len,first,rows:rows.len()});first+=rows.len();tokens.push(source);
        }
        ensure!(first==request.rows.len(),"Inkling verify rows include unmarked slots");
        let unit=Unit{index:0,sequence:0,rows:0..first,layers:0..self.execution.cfg.layers,slot:0,barrier:false,segments};
        for (verify,requests) in draft.verify.iter().zip(self.execution.request_ranks()) {
            requests.decode.set_replay(Some((verify.replay.buffer,requests.cache.layout.conv.width)))?;
        }
        let sources=tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let result=self.execution.verify(self.placements,&sources,&unit,1);
        for requests in self.execution.request_ranks(){requests.decode.set_replay(None)?;}
        let logits=match result{Ok(logits)=>logits,Err(error)=>{self.restore(mark)?;return Err(error);}};
        for (verify,mark) in draft.verify.iter().zip(&mark.conv){verify.submitted(mark,submitted.clone())?;}
        let output=draft.decode.output.buffer;let h=self.execution.cfg.hidden;
        self.taps=request.tap_sites.iter().enumerate().map(|(index,&site)|HiddenTap{
            site,ptr:output.ptr.wrapping_byte_add(index*h*2),rows:first,hidden:h,stride:request.tap_sites.len()*h,
            stream:self.execution.requests.decode.stream.raw}).collect();
        Ok(VerifyOutput{logits,taps:&self.taps})
    }
    fn restore(&mut self,mark:&Self::Mark)->Result<()>{
        self.execution.restore_verify(mark)?;
        for (p,saved) in self.placements.iter_mut().zip(&mark.placements){*p=saved.clone();}
        Ok(())
    }
    fn commit(&mut self,mark:&Self::Mark,accepted:&[AcceptedPrefix])->Result<()>{
        let draft=self.execution.draft.as_ref().context("Inkling drafter missing")?.clone();
        let mut rank_frontiers=Vec::new();
        for ((verify,mark),requests) in draft.verify.iter().zip(&mark.conv).zip(self.execution.request_ranks()) {
            rank_frontiers.push(requests.decode.stream.device.run(||verify.commit(mark,requests.cache.conv_backing(),accepted))?);
        }
        let frontiers=rank_frontiers.first().context("Inkling accepted lead frontiers missing")?;
        ensure!(rank_frontiers.iter().all(|other|other==frontiers),"Inkling TP2 accepted frontiers differ");
        for requests in self.execution.request_ranks(){requests.decode.stream.drain()?;}
        self.execution.require_live()?;
        crate::shared::spark_waves::healthy()?;
        for (p,saved) in self.placements.iter_mut().zip(&mark.placements){
            let frontier=frontiers.iter().find(|(slot,_)|*slot==p.slot).context("Inkling accepted frontier omitted")?.1;
            p.len=frontier;p.state_len=frontier;
            ensure!(frontier>=saved.len,"Inkling accepted prefix precedes mark");
        }
        Ok(())
    }
}
