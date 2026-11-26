//! Context filling is shared by prefill, kept verification rows and replay.
//! Separate context projections never read the drafter's self K/V weights.
use super::recurrent_forward::Rank;
use anyhow::{ensure,Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;

fn bytes<T:Copy>(items:&[T])->&[u8] {
    // SAFETY: callers supply plain numeric metadata, whose initialized byte
    // representation is stable for the duration of the ordered staging copy.
    unsafe{std::slice::from_raw_parts(items.as_ptr().cast(),std::mem::size_of_val(items))}
}
fn view(buffer:Buffer,offset:usize,bytes:usize)->Result<Buffer>{
    ensure!(bytes>0 && offset.checked_add(bytes).is_some_and(|end|end<=buffer.bytes),"recurrent context view extent");
    Ok(Buffer{ptr:buffer.ptr.wrapping_byte_add(offset),bytes,..buffer})
}

impl Rank<'_> {
    /// destinations[layer] are physical rows in that layer's rank-local plane.
    /// A caller holds a shared page-pool reservation until this stream drains,
    /// then publishes only the accepted prefix. No rejected row is published.
    /// When `single_sequence` is true, rows overwritten inside a single sequence's
    /// local ring may be skipped; batched multi-sequence appends project all rows.
    pub fn store_context(&self,features:Buffer,positions:&[i64],destinations:&[Vec<i64>],single_sequence:bool)->Result<()> {
        let c=&self.config.transformer;let rank=self.layout.rank;
        let rows=positions.len();let lib=self.stream.device.library;let stream=self.stream.raw;
        ensure!(rows>0 && rows<=self.layout.context_rows && destinations.len()==c.layers
            && destinations.iter().all(|rows_|rows_.len()==rows)
            && positions.iter().all(|&p|p>=0 && (p as usize)<c.max_position),"recurrent context metadata differs");
        ensure!(features.bytes>=rows*self.config.taps.len()*c.hidden*2,"recurrent target tap extent differs");
        let total_rows=rows;
        let kv=rank.kv_heads*c.head_dim;
        for index in 0..c.layers {
            // Context projections are independent per position: for a single
            // sequence, rows overwritten in its local ring need not be computed.
            let first=if single_sequence {
                match self.config.context[index] {
                    cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::BlockAnchoredWindow(window)=>total_rows.saturating_sub(window),
                    cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full=>0,
                }
            } else { 0 };
            let rows=total_rows-first;
            let pitch=self.config.taps.len()*c.hidden*2;
            let features=view(features,first*pitch,rows*pitch)?;
            // SAFETY: ordered upload retains metadata until its copy completes;
            // permanent destinations survive every queued context consumer.
            unsafe{self.upload("context.positions",bytes(&positions[first..]))?;}
            let layer=&self.weights.tensors.layers[index];
            let alpha=self.weights.derived("derived.fuse_alpha")?.buffer;
            let alpha=view(alpha,index*self.config.taps.len()*2,self.config.taps.len()*2)?;
            // SAFETY: layer mixture uses raw ordered target taps; outputs are
            // retained disjoint BF16 planes, gamma and alpha immutable weights.
            unsafe{self.programs.mix.launch(&[features,alpha,layer.dflare_norm.buffer,self.buffer("context.mixed")?],
                rows,c.rms_norm_eps as f32,0,stream)?;}
            let names=["k_proj_ctx","v_proj_ctx"].map(|p|format!("layers.{index}.self_attn.{p}.weight"));
            self.programs.parts(&names,&layer.context_kv,self.buffer("context.mixed")?,self.buffer("context.kv")?,
                rows,c.hidden,&[kv,kv],lib,self.buffer("projection.bf16")?,stream)?;
            // q_out is unused for the q_heads=0 context route, but pass a valid
            // pointer as required by the uniform native pointer ABI.
            unsafe{self.programs.context_rotary.launch(&[self.buffer("context.kv")?,layer.q_norm.buffer,
                layer.context_k_norm.buffer,self.buffer("context.positions")?,self.buffer("rotary.frequencies")?,
                self.buffer("step.query")?,self.buffer("context.key")?,self.buffer("context.value")?],
                rows,c.rms_norm_eps as f32,0,stream)?;}
            let capacity=self.layout.cache.layer_bytes[index]/self.layout.cache.record_bytes;
            let active=&destinations[index][first..];
            ensure!(active.iter().all(|&slot|slot>=0 && (slot as usize)<capacity)
                && active.iter().copied().collect::<std::collections::BTreeSet<_>>().len()==rows,
                "recurrent KV store destinations alias or lie outside admitted plane");
            // SAFETY: host plan checked every slot against the layer's physical
            // capacity; typed store checks row sources and stream retains cache.
            unsafe{self.upload("context.destinations",bytes(active))?;
                self.programs.store.launch(&[self.buffer("context.key")?,self.buffer("context.value")?,
                    self.buffer("context.destinations")?,self.buffer(&format!("cache.layer.{index}.k"))?,
                    self.buffer(&format!("cache.layer.{index}.v"))?],rows,c.rms_norm_eps as f32,0,stream)?;}
        }
        Ok(())
    }

    /// Refresh one sequence's head feature from its last committed raw taps.
    pub fn recurrent_feature(&self,last_taps:Buffer,output:Buffer)->Result<()> {
        let c=&self.config.transformer;
        let head=self.weights.tensors.head.as_ref().ok_or_else(||anyhow::anyhow!("recurrent feature requires lead head"))?;
        let alpha=self.weights.derived("derived.fuse_alpha")?.buffer;
        let row=view(alpha,c.layers*self.config.taps.len()*2,self.config.taps.len()*2)?;
        // SAFETY: one complete raw feature row and immutable final mixture;
        // output is a distinct persistent per-slot BF16 feature record.
        unsafe{self.programs.mix.launch(&[last_taps,row,head.dflare_norm.buffer,output],1,c.rms_norm_eps as f32,0,self.stream.raw)}
    }

    /// Refresh a batch of distinct sequences' head features from packed last-tap rows
    /// (`[slots.len(), taps, hidden]` BF16) using one batched `mix` launch.
    pub fn recurrent_features(&self,last_taps:Buffer,slots:&[usize])->Result<()> {
        ensure!(!slots.is_empty() && slots.len()<=self.layout.context_rows,"recurrent feature batch exceeds admission");
        let c=&self.config.transformer;
        let width=c.hidden*2;
        let f=self.buffer("cache.last_features")?;
        if slots.len()==1 {
            let slot=slots[0];
            ensure!(slot<self.layout.slots,"recurrent feature slot exceeds admission");
            return self.recurrent_feature(last_taps,view(f,slot*width,width)?);
        }
        let mut seen=std::collections::BTreeSet::new();
        ensure!(slots.iter().all(|&slot|slot<self.layout.slots && seen.insert(slot)),
            "recurrent feature batch aliases or exceeds slots");
        let head=self.weights.tensors.head.as_ref().ok_or_else(||anyhow::anyhow!("recurrent feature requires lead head"))?;
        let alpha=self.weights.derived("derived.fuse_alpha")?.buffer;
        let row=view(alpha,c.layers*self.config.taps.len()*2,self.config.taps.len()*2)?;
        let mixed=self.buffer("context.mixed")?;
        // SAFETY: packed last-tap rows and immutable head mixture produce disjoint
        // per-row BF16 features in `context.mixed`, then scatter into persistent slots.
        unsafe{self.programs.mix.launch(&[last_taps,row,head.dflare_norm.buffer,mixed],
            slots.len(),c.rms_norm_eps as f32,0,self.stream.raw)?;}
        let lib=self.stream.device.library;
        let mut i=0;
        while i<slots.len() {
            let start_slot=slots[i];
            let mut run=1;
            while i+run<slots.len() && slots[i+run]==start_slot+run {
                run+=1;
            }
            let bytes=run*width;
            // SAFETY: `context.mixed` and `cache.last_features` are disjoint admitted
            // planes; stream order guarantees `mix` completes before scatter copies.
            unsafe{lib.copy_d2d_async(view(f,start_slot*width,bytes)?,
                view(mixed,i*width,bytes)?,bytes,self.stream.raw)?;}
            i+=run;
        }
        Ok(())
    }
}
