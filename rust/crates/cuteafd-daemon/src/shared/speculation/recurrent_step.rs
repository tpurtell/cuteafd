//! Block orchestration without a family-private scheduler or transport loop.
use super::recurrent_forward::Rank;
use anyhow::{ensure,Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy;

fn bytes<T:Copy>(value:&[T])->&[u8] {
    // SAFETY: plain initialized numeric metadata is borrowed until the native
    // ordered upload retains its own pinned copy.
    unsafe{std::slice::from_raw_parts(value.as_ptr().cast(),std::mem::size_of_val(value))}
}
fn bf16(value:f32)->u16 {
    let bits=value.to_bits();
    ((bits.wrapping_add(0x7fff+((bits>>16)&1)))>>16) as u16
}

impl Rank<'_> {
    /// Run only once after recorded push commit; sources are not execution-ready
    /// until the caller has joined this stream before publishing readiness.
    pub fn initialize_derived(&self)->Result<()> {
        let c=&self.config.transformer;let lib=self.stream.device.library;
        let count=(c.layers+1)*self.config.taps.len();
        let mut source=vec![0u8;count*2];
        lib.copy_d2h_quiescent(&mut source,self.weights.tensors.fuse.buffer)?;
        let values=source.chunks_exact(2).map(|v|f32::from_bits(u32::from(u16::from_le_bytes([v[0],v[1]]))<<16)).collect::<Vec<_>>();
        let alpha=values.chunks_exact(self.config.taps.len()).flat_map(|row| {
            let max=row.iter().copied().fold(f32::NEG_INFINITY,f32::max);
            let values=row.iter().map(|value|(*value-max).exp()).collect::<Vec<_>>();
            let sum=values.iter().sum::<f32>();
            values.into_iter().map(move |value|bf16(value/sum))
        }).collect::<Vec<_>>();
        ensure!(alpha.len()==count,"recurrent fuse softmax row inventory differs");
        lib.copy_h2d(self.weights.derived("derived.fuse_alpha")?.buffer,bytes(&alpha))?;
        let theta=c.rope_theta as f32;
        let inverse=(0..c.head_dim/2).map(|index|1.0f32/theta.powf((2*index) as f32/c.head_dim as f32)).collect::<Vec<_>>();
        lib.copy_h2d(self.buffer("rotary.frequencies")?,bytes(&inverse))?;
        Ok(())
    }

    pub fn initialize_block(&self,anchors:Buffer,target_embedding:Buffer,target_embed_norm:Buffer,
        sequences:usize)->Result<()> {
        let c=&self.config.transformer;let lib=self.stream.device.library;let stream=self.stream.raw;
        ensure!(sequences>0 && sequences<=self.layout.sequences,"recurrent block sequence count exceeds admission");
        // SAFETY: device token ids reference the borrowed native target embedding;
        // normalized anchors and materialized residual are admitted disjoint planes.
        unsafe{lib.cuda_embedding_lookup_bf16_async(target_embedding,anchors,self.buffer("head.hidden")?,
            sequences,self.config.padded_vocab,c.hidden,stream)?;
            lib.cuda_ds4_rmsnorm_bf16_rne_async(self.buffer("head.hidden")?,target_embed_norm,
                self.buffer("step.anchors")?,i32::try_from(sequences)?,i32::try_from(c.hidden)?,c.rms_norm_eps as f32,stream)?;
            self.programs.init.launch(&[self.buffer("step.anchors")?,self.weights.tensors.mask_embedding.buffer,
                self.buffer("step.residual")?],sequences,c.rms_norm_eps as f32,0,stream)?;}
        self.norm_first(sequences*c.block)
    }

    /// Per-layer lengths are exact: all draft queries see the same committed
    /// context window, and noncausal block attention sees every draft position.
    pub fn attention_metadata(&self,index:usize,sequences:&[super::dflash::DraftSeq],pages:&[Vec<i32>])->Result<usize> {
        let c=&self.config.transformer;let lib=self.stream.device.library;
        ensure!(!sequences.is_empty() && sequences.len()<=self.layout.sequences && pages.len()==sequences.len(),
            "recurrent metadata sequence inventory differs");
        let mut table=vec![0i32;3*sequences.len()];
        let mut positions=Vec::with_capacity(sequences.len()*c.block);
        let width=self.layout.cache.page_table_bytes as usize/self.layout.slots/4;
        let mut page_table=vec![0i32;sequences.len()*width];
        let mut max_context=0usize;
        for (row,seq) in sequences.iter().enumerate() {
            ensure!(seq.slot<self.layout.slots && seq.valid_from<=seq.position
                && seq.position.checked_add(c.block).is_some_and(|end|end<=c.max_position),"recurrent sequence frontier outside checkpoint");
            let full=seq.position-seq.valid_from;
            let visible=match self.config.context[index] {ContextPolicy::Full=>full,
                ContextPolicy::BlockAnchoredWindow(window)=>full.min(window)};
            table[row]=i32::try_from(seq.slot)?;
            table[sequences.len()+row]=i32::try_from(visible)?;
            table[2*sequences.len()+row]=i32::try_from(match self.config.context[index]{
                ContextPolicy::Full=>full,ContextPolicy::BlockAnchoredWindow(_)=>seq.position})?;
            positions.extend((seq.position..seq.position+c.block).map(|position|position as i64));
            if matches!(self.config.context[index],ContextPolicy::Full) {
                ensure!(pages[row].len()>=full.div_ceil(64) && pages[row].len()<=width
                    && pages[row].iter().all(|&page|page>=0 && (page as usize)<self.layout.cache.pages),
                    "recurrent full context page table incomplete or out of pool");
                page_table[row*width..row*width+pages[row].len()].copy_from_slice(&pages[row]);
            }
            max_context=max_context.max(visible);
        }
        // SAFETY: staging owns copied numeric metadata until each event; the
        // permanent destinations are ordered after previous attention readers.
        unsafe{self.upload("step.tables",bytes(&table))?;
            self.upload("step.positions",bytes(&positions))?;
            if matches!(self.config.context[index],ContextPolicy::Full){
                self.upload("cache.tables",bytes(&page_table))?;}}
        Ok(max_context+c.block)
    }

    pub fn forward_whole(&self,sequences:&[super::dflash::DraftSeq],pages:&[Vec<i32>])->Result<()> {
        ensure!(self.layout.rank.ranks==1,"whole recurrent body cannot skip TP reductions");
        let count=sequences.len();
        for index in 0..self.config.transformer.layers {
            let keys=self.attention_metadata(index,sequences,pages)?;
            self.attention_partial(index,count,keys)?;
            self.ffn_partial(index,count,self.buffer("step.partial")?)?;
            self.finish_layer(index,count,self.buffer("step.partial")?)?;
        }
        Ok(())
    }
}
