//! Shared page-pool reservations for mixed full/window block-drafter context.
use super::{recurrent_forward::Rank,dflash::DraftSeq};
use anyhow::{ensure,Context,Result};
use cuteafd_engine::prefix::{DraftBinding,DraftPagePool,DraftReservation};
use cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use std::cell::RefCell;

pub(crate) struct ContextState {
    pub pool:DraftPagePool,
    bindings:Vec<Option<DraftBinding>>,
    slots:usize,
    context:usize,
}
impl ContextState {
    pub fn new(slots:usize,context:usize,target_tokens:usize,identity:u64)->Result<Self>{
        Ok(Self{pool:DraftPagePool::new(target_tokens,slots,context,identity)?,
            bindings:vec![None;slots],slots,context})
    }
    pub fn reset(&mut self,slot:usize,origin:usize)->Result<()> {
        ensure!(slot<self.slots && origin<=self.context,"recurrent context reset outside admission");
        if let Some(binding)=self.bindings[slot].take(){self.pool.release(binding)?;}
        self.bindings[slot]=Some(self.pool.bind(slot,origin)?);Ok(())
    }
    /// Shared slot-binding entry point for both serving (`PrefixFamily::bind`) and
    /// golden/oracle runners: resets/binds `slot` at `origin` and, when `admit` is
    /// `Some((end, block))`, admits draft pages through `end` for the given block size.
    pub fn bind_slot(&mut self,slot:usize,origin:usize,admit:Option<(usize,usize)>)->Result<DraftBinding> {
        self.reset(slot,origin)?;
        let binding=self.binding(slot)?;
        if let Some((end,block))=admit {
            self.pool.admit(binding,end,block)?;
        }
        Ok(binding)
    }
    pub fn quarantine(&mut self,slot:usize) {
        if let Some(binding)=self.bindings.get(slot).copied().flatten() {
            self.pool.quarantine(binding).expect("live recurrent binding belongs to its pool");
        }
    }
    pub fn install(&mut self,slot:usize,binding:DraftBinding)->Result<()> {
        ensure!(slot<self.slots,"recurrent restored binding slot exceeds admission");
        self.bindings[slot]=Some(binding);Ok(())
    }
    pub fn binding(&self,slot:usize)->Result<DraftBinding>{
        self.bindings.get(slot).copied().flatten().context("unbound recurrent context slot")
    }
    pub fn pages(&self,sequences:&[DraftSeq])->Result<Vec<Vec<i32>>>{
        sequences.iter().map(|seq|{
            let binding=self.binding(seq.slot)?;
            let (origin,frontier)=self.pool.frontier(binding)?;
            ensure!((origin,frontier)==(seq.valid_from,seq.position),"recurrent draft differs from committed context frontier");
            Ok(self.pool.committed_page_table(binding)?)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quarantined_recurrent_destinations_survive_normal_cleanup() -> Result<()> {
        let mut state=ContextState::new(2,1024,8*64,88)?;
        state.reset(0,0)?;
        let binding=state.binding(0)?;
        let pending=state.pool.reserve(binding,0,3*64)?;
        let destinations=state.pool.destinations(pending)?;
        state.quarantine(0);
        assert!(state.reset(0,0).is_err(),"quarantined slot must never be rebound");
        assert_eq!(state.pool.free(),5);
        state.reset(1,0)?;
        let next=state.pool.reserve(state.binding(1)?,0,5*64)?;
        assert!(state.pool.destinations(next)?.iter().all(|row|!destinations.contains(row)));
        assert_eq!(state.pool.free(),0);
        Ok(())
    }

    #[test]
    fn bind_slot_binds_every_slot_and_rebinds_without_leaking_pages() -> Result<()> {
        // Exercise both Inkling 2 Mini (4 slots, block=8, 48-visit golden width-invariance/graph-parity)
        // and Inkling 2 Small (16 slots, block=8, width-invariance) recurrent page-pool geometries.
        for (slots, context, target_tokens, block, end, seed) in [
            (4usize, 4096usize, 32 * 64, 8usize, 127usize, 2001u64),
            (16usize, 4096usize, 64 * 64, 8usize, 72usize, 2002u64),
        ] {
            let mut state = ContextState::new(slots, context, target_tokens, seed)?;
            let initial_free = state.pool.free();
            // Unbound slots must fail with the exact guard error seen when a golden skips binding.
            for slot in 0..slots {
                let err = state.binding(slot).unwrap_err();
                assert!(err.to_string().contains("unbound recurrent context slot"), "{err:#}");
            }
            // Binding every slot through the shared entry point admits draft pages for prefill.
            for slot in 0..slots {
                let binding = state.bind_slot(slot, 0, Some((end, block)))?;
                assert_eq!(state.binding(slot)?, binding);
                let reservation = state.pool.reserve(binding, 0, end)?;
                state.pool.commit(reservation, end)?;
                assert_eq!(state.pool.frontier(binding)?, (0, end));
            }
            // Repeatedly re-binding a slot (width-invariance & graph-parity replay loops)
            // releases its prior draft pages cleanly without leaking pool capacity.
            let free_after_bind = state.pool.free();
            for _ in 0..6 {
                let binding = state.bind_slot(1, 0, Some((end, block)))?;
                let reservation = state.pool.reserve(binding, 0, end)?;
                state.pool.commit(reservation, end)?;
            }
            assert_eq!(state.pool.free(), free_after_bind);
            // Releasing every slot returns all pool pages.
            for slot in 0..slots {
                state.bind_slot(slot, 0, None)?;
            }
            assert_eq!(state.pool.free(), initial_free);
        }
        Ok(())
    }

    #[test]
    fn golden_and_serve_share_recurrent_slot_binding_across_families() {
        // Inkling 2 Mini: serve (`MiniPrefix::bind`) and golden (`inkling2_mini/golden.rs`)
        // must both go through `MiniPrefix::bind_slot` so drafter context binding never diverges.
        let mini_cache = include_str!("../../families/inkling2_mini/cache.rs");
        let mini_golden = include_str!("../../families/inkling2_mini/golden.rs");
        assert!(mini_cache.contains("pub fn bind_slot("), "MiniPrefix must expose shared bind_slot");
        let mini_serve_bind = mini_cache.split("fn bind(&self, placement: &mut Placement)").nth(1)
            .expect("MiniPrefix::bind").split("fn discard(").next().expect("MiniPrefix::bind body");
        assert!(mini_serve_bind.contains("self.bind_slot(placement, None)"),
            "Mini serve bind must delegate to MiniPrefix::bind_slot");
        assert!(mini_golden.contains("MiniPrefix::for_execution(&execution)"),
            "Mini golden must build MiniPrefix::for_execution");
        assert!(mini_golden.contains("prefix.bind_slot("),
            "Mini golden must bind slots via shared MiniPrefix::bind_slot");
        assert!(!mini_golden.contains("requests.cache.reset(") && !mini_golden.contains("drafter.reset_context("),
            "Mini golden must not bypass MiniPrefix::bind_slot with raw reset calls");

        // Inkling 2 Small: serve (`InklingPrefix::bind`) and width-invariance golden
        // (`inkling2/width_invariance.rs`) must both go through `InklingPrefix::bind_slot`.
        let small_cache = include_str!("../../families/inkling2/cache.rs");
        let small_wi = include_str!("../../families/inkling2/width_invariance.rs");
        assert!(small_cache.contains("pub fn bind_slot("), "InklingPrefix must expose shared bind_slot");
        let small_serve_bind = small_cache.split("fn bind(&self, placement: &mut Placement)").nth(1)
            .expect("InklingPrefix::bind").split("fn discard(").next().expect("InklingPrefix::bind body");
        assert!(small_serve_bind.contains("self.bind_slot(placement, None)"),
            "Inkling 2 Small serve bind must delegate to InklingPrefix::bind_slot");
        assert!(small_wi.contains("InklingPrefix::for_execution(&execution)"),
            "Inkling 2 Small width-invariance golden must build InklingPrefix::for_execution");
        assert!(small_wi.contains("prefix.bind_slot("),
            "Inkling 2 Small width-invariance golden must bind slots via shared InklingPrefix::bind_slot");
        assert!(!small_wi.contains("requests.cache.reset(") && !small_wi.contains("drafter.reset_context(")
                && !small_wi.contains("drafter.admit_pages("),
            "Inkling 2 Small width-invariance golden must not duplicate bind/admit outside InklingPrefix::bind_slot");

        // Audited sibling families: GLM Flash KDA and Kimi K3 goldens do not use `recurrent_cache`
        // (if either adopts `recurrent_cache` in the future, this assertion flags it for shared bind_slot wiring).
        for (name, src) in [
            ("glm5_flash/mod.rs", include_str!("../../families/glm5_flash/mod.rs")),
            ("glm5_flash/speculate.rs", include_str!("../../families/glm5_flash/speculate.rs")),
            ("kimi_k3/golden.rs", include_str!("../../families/kimi_k3/golden.rs")),
        ] {
            assert!(!src.contains("recurrent_cache"),
                "{name} adopted recurrent_cache; wire it to the shared bind_slot entry point and cover it here");
        }
    }

    #[test]
    fn batched_reservations_roll_back_atomically_on_later_span_failure() -> Result<()> {
        let mut state = ContextState::new(4, 1024, 8 * 64, 404)?;
        for slot in 0..4 {
            state.bind_slot(slot, 0, None)?;
        }
        let initial_free = state.pool.free();
        let spans = [
            ContextSpan { slot: 0, start: 0, first_row: 0, rows: 65 }, // allocates 2 pages
            ContextSpan { slot: 1, start: 0, first_row: 65, rows: 65 }, // allocates 2 pages
            ContextSpan { slot: 2, start: 5, first_row: 130, rows: 1 }, // bad frontier (expected 0)
        ];
        let mut reserved = Vec::new();
        let outcome = (|| -> Result<()> {
            for span in spans {
                let binding = state.binding(span.slot)?;
                let (origin, frontier) = state.pool.frontier(binding)?;
                ensure!(
                    frontier == span.start
                        && span.start.checked_add(span.rows).is_some_and(|end| end <= state.context)
                        && origin <= span.start,
                    "bad frontier"
                );
                let res = state.pool.reserve(binding, span.start, span.rows)?;
                reserved.push(res);
            }
            Ok(())
        })();
        assert!(outcome.is_err());
        assert_eq!(reserved.len(), 2);
        for res in reserved {
            state.pool.commit(res, 0)?;
        }
        assert_eq!(state.pool.free(), initial_free, "failed multi-span reserve leaked pages");
        for slot in 0..4 {
            assert_eq!(state.pool.frontier(state.binding(slot)?)?, (0, 0));
        }
        Ok(())
    }

    #[test]
    fn buffers_overlap_detects_aliasing_and_disjoint_ranges() {
        let base = Buffer { ptr: 0x1000 as *mut libc::c_void, bytes: 256, device_id: 0, flags: 0 };
        let inside = Buffer { ptr: 0x1080 as *mut libc::c_void, bytes: 64, device_id: 0, flags: 0 };
        let adjacent = Buffer { ptr: 0x1100 as *mut libc::c_void, bytes: 64, device_id: 0, flags: 0 };
        let other_dev = Buffer { ptr: 0x1000 as *mut libc::c_void, bytes: 256, device_id: 1, flags: 0 };
        assert!(buffers_overlap(base, inside));
        assert!(buffers_overlap(inside, base));
        assert!(!buffers_overlap(base, adjacent));
        assert!(!buffers_overlap(adjacent, base));
        assert!(!buffers_overlap(base, other_dev));
    }
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) struct ContextSpan {
    pub slot:usize,
    pub start:usize,
    pub first_row:usize,
    pub rows:usize,
}

fn buffers_overlap(a:Buffer,b:Buffer)->bool {
    if a.device_id!=b.device_id || a.bytes==0 || b.bytes==0 { return false; }
    let (a0,b0)=(a.ptr as usize,b.ptr as usize);
    let (a1,b1)=(a0.saturating_add(a.bytes),b0.saturating_add(b.bytes));
    a0<b1 && b0<a1
}

pub(crate) struct ContextBank { pub state:RefCell<ContextState> }
impl ContextBank {
    /// Context inputs are one contiguous causal sequence span. The caller
    /// tiles prefill and packed members through this shared operation.
    pub fn append_ranked(&self,rank:&Rank<'_>,split:Option<&super::gqa_tp2::GqaHeadSplit<'_,Rank<'_>>>,
        slot:usize,start:usize,features:Buffer,rows:usize)->Result<()> {
        ensure!(rows>0 && rows<=rank.layout.context_rows,"recurrent context append row capacity");
        let (reservation,physical)={
            let mut state=self.state.borrow_mut();let binding=state.binding(slot)?;
            let (origin,frontier)=state.pool.frontier(binding)?;
            ensure!(frontier==start && start.checked_add(rows).is_some_and(|end|end<=state.context)
                && origin<=start,"recurrent context append is not canonical");
            let reservation=state.pool.reserve(binding,start,rows)?;
            (reservation,state.pool.destinations(reservation)?)
        };
        let positions=(start..start+rows).map(|p|p as i64).collect::<Vec<_>>();
        let destinations=rank.config.context.iter().map(|policy|match policy {
            ContextPolicy::Full=>physical.clone(),ContextPolicy::BlockAnchoredWindow(window)=>
                (start..start+rows).map(|p|(slot*window+p%window) as i64).collect(),
        }).collect::<Vec<_>>();
        let result=(||{
            if let Some(split)=split {
                use crate::shared::peer_split::DIRECT;
                let input=split.peer.buffer("context.taps")?;
                let bytes=rows*rank.config.taps.len()*rank.config.transformer.hidden*2;
                ensure!(features.bytes>=bytes && input.bytes>=bytes,"recurrent peer feature extent differs");
                split.exchange.push_to(split.lead,DIRECT,features.ptr,input.ptr,bytes)?;
                split.exchange.wait(1-split.lead,DIRECT)?;
                split.peer.stream.device.run(||split.peer.store_context(input,&positions,&destinations,true))?;
            }
            rank.store_context(features,&positions,&destinations,true)?;
            if rank.layout.rank.rank==rank.layout.rank.lead {
                let width=rank.config.transformer.hidden*2;
                let taps=rank.config.taps.len()*width;
                let last=Buffer{ptr:features.ptr.wrapping_byte_add((rows-1)*taps),bytes:taps,..features};
                let f=rank.buffer("cache.last_features")?;
                rank.recurrent_feature(last,Buffer{ptr:f.ptr.wrapping_byte_add(slot*width),bytes:width,..f})?;
            }
            rank.stream.drain()?;
            if let Some(split)=split{split.peer.stream.drain()?;}
            self.state.borrow_mut().pool.commit(reservation,rows)?;Ok(())
        })();
        if result.is_err(){
            // A failed producer may still read/write the provisional pages.
            // Never return them to the pool until native stream retirement.
            if let Some(split)=split {
                let drained=(||{split.exchange.publish_abort()?;split.exchange.drain_compute()})();
                split.exchange.finish_terminal(drained.is_ok());
                crate::shared::decode_graph::fatal_drain(drained,"recurrent failed peer context append");
            }
            rank.stream.drain_or_abort("recurrent failed context append");
            self.state.borrow_mut().pool.commit(reservation,0)?;
        }
        result
    }

    /// Batched multi-sequence context append with one stream drain across all
    /// spans. Preserves exact per-slot destinations, RoPE positions and head
    /// features without per-request host syncs.
    pub fn append_batch_ranked(&self,rank:&Rank<'_>,split:Option<&super::gqa_tp2::GqaHeadSplit<'_,Rank<'_>>>,
        features:Buffer,spans:&[ContextSpan])->Result<()> {
        if spans.is_empty() { return Ok(()); }
        if let &[span] = spans {
            let pitch=rank.config.taps.len()*rank.config.transformer.hidden*2;
            let end_row=span.first_row.checked_add(span.rows).context("recurrent context feature row overflow")?;
            ensure!(end_row.checked_mul(pitch).is_some_and(|end|end<=features.bytes),
                "recurrent context feature span outside owner");
            let slice=Buffer{ptr:features.ptr.wrapping_byte_add(span.first_row*pitch),bytes:span.rows*pitch,..features};
            return self.append_ranked(rank,split,span.slot,span.start,slice,span.rows);
        }
        let mut total_rows=0usize;
        let mut seen_slots=std::collections::BTreeSet::new();
        let pitch=rank.config.taps.len()*rank.config.transformer.hidden*2;
        let min_window=rank.config.context.iter().filter_map(|p|match p {
            ContextPolicy::BlockAnchoredWindow(w)=>Some(*w),ContextPolicy::Full=>None,
        }).min();
        for span in spans {
            ensure!(span.rows>0,"recurrent context append span is empty");
            ensure!(seen_slots.insert(span.slot),"recurrent context append aliases a slot");
            if spans.len()>1 {
                if let Some(window)=min_window {
                    ensure!(span.rows<=window,"batched recurrent context span exceeds local window");
                }
            }
            let end_row=span.first_row.checked_add(span.rows).context("recurrent context feature row overflow")?;
            ensure!(end_row.checked_mul(pitch).is_some_and(|end|end<=features.bytes),
                "recurrent context feature span outside owner");
            total_rows=total_rows.checked_add(span.rows).context("recurrent context batch row overflow")?;
        }
        ensure!(total_rows>0 && total_rows<=rank.layout.context_rows,"recurrent context append row capacity");
        let mut reservations=Vec::with_capacity(spans.len());
        let mut physicals=Vec::with_capacity(spans.len());
        {
            let mut state=self.state.borrow_mut();
            let reserve_all=(||->Result<()>{
                for span in spans {
                    let binding=state.binding(span.slot)?;
                    let (origin,frontier)=state.pool.frontier(binding)?;
                    ensure!(frontier==span.start && span.start.checked_add(span.rows).is_some_and(|end|end<=state.context)
                        && origin<=span.start,"recurrent context append is not canonical");
                    let reservation=state.pool.reserve(binding,span.start,span.rows)?;
                    let physical=state.pool.destinations(reservation)?;
                    reservations.push((reservation,span.rows));
                    physicals.push(physical);
                }
                Ok(())
            })();
            if let Err(error)=reserve_all {
                for (reservation,_) in reservations { let _=state.pool.commit(reservation,0); }
                return Err(error);
            }
        }
        let mut positions=Vec::with_capacity(total_rows);
        for span in spans { positions.extend((span.start..span.start+span.rows).map(|p|p as i64)); }
        let destinations=rank.config.context.iter().map(|policy|match policy {
            ContextPolicy::Full=>physicals.iter().flat_map(|rows|rows.iter().copied()).collect(),
            ContextPolicy::BlockAnchoredWindow(window)=>{
                let mut dst=Vec::with_capacity(total_rows);
                for span in spans {
                    dst.extend((span.start..span.start+span.rows).map(|p|(span.slot*window+p%window) as i64));
                }
                dst
            }
        }).collect::<Vec<_>>();
        let first_row=spans[0].first_row;
        let contiguous=spans.iter().scan(first_row,|expected,span|{
            let ok=span.first_row==*expected;
            *expected=span.first_row.saturating_add(span.rows);
            Some(ok)
        }).all(|ok|ok);
        let single_sequence=spans.len()==1;
        let result=(||{
            let lib=rank.stream.device.library;
            let packed_features=if contiguous {
                Buffer{ptr:features.ptr.wrapping_byte_add(first_row*pitch),bytes:total_rows*pitch,..features}
            } else {
                let stage=rank.buffer("context.taps")?;
                ensure!(!buffers_overlap(features,stage),"batched recurrent context source aliases context.taps");
                let mut dst_row=0usize;
                let mut i=0usize;
                while i<spans.len() {
                    let src_first=spans[i].first_row;
                    let mut run_rows=spans[i].rows;
                    let mut j=i+1;
                    while j<spans.len() && spans[j].first_row==src_first+run_rows {
                        run_rows+=spans[j].rows;
                        j+=1;
                    }
                    let bytes=run_rows*pitch;
                    // SAFETY: disjoint tap source and admitted context.taps staging on this stream.
                    unsafe{lib.copy_d2d_async(
                        Buffer{ptr:stage.ptr.wrapping_byte_add(dst_row*pitch),bytes,..stage},
                        Buffer{ptr:features.ptr.wrapping_byte_add(src_first*pitch),bytes,..features},
                        bytes,rank.stream.raw)?;}
                    dst_row+=run_rows;
                    i=j;
                }
                Buffer{bytes:total_rows*pitch,..stage}
            };
            if let Some(split)=split {
                use crate::shared::peer_split::DIRECT;
                let input=split.peer.buffer("context.taps")?;
                let bytes=total_rows*pitch;
                ensure!(packed_features.bytes>=bytes && input.bytes>=bytes,"recurrent peer feature extent differs");
                split.exchange.push_to(split.lead,DIRECT,packed_features.ptr,input.ptr,bytes)?;
                split.exchange.wait(1-split.lead,DIRECT)?;
                split.peer.stream.device.run(||split.peer.store_context(input,&positions,&destinations,single_sequence))?;
            }
            rank.store_context(packed_features,&positions,&destinations,single_sequence)?;
            if rank.layout.rank.rank==rank.layout.rank.lead {
                if single_sequence {
                    let width=rank.config.transformer.hidden*2;
                    let last=Buffer{ptr:packed_features.ptr.wrapping_byte_add((total_rows-1)*pitch),bytes:pitch,..packed_features};
                    let f=rank.buffer("cache.last_features")?;
                    rank.recurrent_feature(last,Buffer{ptr:f.ptr.wrapping_byte_add(spans[0].slot*width),bytes:width,..f})?;
                } else {
                    let slots=spans.iter().map(|s|s.slot).collect::<Vec<_>>();
                    let last_taps=if spans.iter().all(|s|s.rows==1) {
                        packed_features
                    } else {
                        let stage=rank.buffer("context.taps")?;
                        ensure!(!buffers_overlap(features,stage),"batched recurrent last-feature gather aliases context.taps");
                        for (index,span) in spans.iter().enumerate() {
                            let src_row=span.first_row+span.rows-1;
                            // SAFETY: `store_context` already consumed `context.taps` on this stream,
                            // and `features` is disjoint from `context.taps`.
                            unsafe{lib.copy_d2d_async(
                                Buffer{ptr:stage.ptr.wrapping_byte_add(index*pitch),bytes:pitch,..stage},
                                Buffer{ptr:features.ptr.wrapping_byte_add(src_row*pitch),bytes:pitch,..features},
                                pitch,rank.stream.raw)?;}
                        }
                        Buffer{bytes:spans.len()*pitch,..stage}
                    };
                    rank.recurrent_features(last_taps,&slots)?;
                }
            }
            rank.stream.drain()?;
            if let Some(split)=split{split.peer.stream.drain()?;}
            let mut state=self.state.borrow_mut();
            for &(reservation,rows) in &reservations {
                state.pool.commit(reservation,rows)?;
            }
            Ok(())
        })();
        if result.is_err(){
            // A failed producer may still read/write the provisional pages.
            // Never return them to the pool until native stream retirement.
            if let Some(split)=split {
                let drained=(||{split.exchange.publish_abort()?;split.exchange.drain_compute()})();
                split.exchange.finish_terminal(drained.is_ok());
                crate::shared::decode_graph::fatal_drain(drained,"recurrent failed peer context append");
            }
            rank.stream.drain_or_abort("recurrent failed context append");
            let mut state=self.state.borrow_mut();
            for &(reservation,_) in &reservations {
                let _=state.pool.commit(reservation,0);
            }
        }
        result
    }
}
