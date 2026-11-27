//! Exact accepted-prefix FIR-history commit, shared with hybrid verification.
//! Payloads are F32 bit patterns; no arithmetic or checkpoint reconstruction.
use crate::{programs::{Program,Programs,Scalar},CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure,Context,Result};
use std::ffi::c_void;

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Layout {
    pub slots:usize,
    pub width:usize,
    pub replay_rows:usize,
    pub lag_stride:usize,
    pub layer_stride:usize,
    pub slot_stride:usize,
    pub replay_row_stride:usize,
    pub replay_layer_stride:usize,
}
impl Layout {
    pub fn contiguous(slots:usize,width:usize)->Result<Self>{
        Self::contiguous_rows(slots,width,64)
    }
    pub fn contiguous_rows(slots:usize,width:usize,replay_rows:usize)->Result<Self>{
        ensure!(slots>0 && width>0 && width<=11264 && matches!(replay_rows,64|128),"FIR bank width/slots/rows outside export");
        let layer_stride=width.checked_mul(3).context("FIR layer stride overflow")?;
        Ok(Self{slots,width,replay_rows,lag_stride:width,layer_stride,
            slot_stride:layer_stride.checked_mul(48).context("FIR slot stride overflow")?,
            replay_row_stride:width,replay_layer_stride:width.checked_mul(replay_rows).context("FIR replay stride overflow")?})
    }
    pub fn history_bytes(self)->Result<usize>{
        self.slots.checked_mul(self.slot_stride).and_then(|n|n.checked_mul(4)).context("FIR history extent overflow")
    }
    pub fn replay_bytes(self)->Result<usize>{
        self.replay_layer_stride.checked_mul(48).and_then(|n|n.checked_mul(4)).context("FIR replay extent overflow")
    }
}

pub struct Commit<'a>{program:Program<'a>,layout:Layout,capacity:usize}
impl<'a> Commit<'a>{
    pub fn load(programs:&'a Programs<'a>,layout:Layout)->Result<Self>{
        Self::load_capacity(programs,layout,layout.replay_rows)
    }
    pub fn load_capacity(programs:&'a Programs<'a>,layout:Layout,capacity:usize)->Result<Self>{
        ensure!(matches!(capacity,64|128) && capacity<=layout.replay_rows
            && layout.replay_layer_stride>=capacity.checked_mul(layout.replay_row_stride).context("FIR commit stride overflow")?,
            "FIR commit capacity differs from replay layout");
        let name=if capacity==64{"fir_history_commit"}else{"fir_history_commit_m128"};
        let program=programs.program(name,&["checkpoint","replay","table","live"])?;
        let spec=program.spec();
        ensure!(spec.capacity_rows as usize==capacity && spec.scalar_kinds==b"iiilllll" && spec.scratch.is_empty(),
            "FIR commit manifest ABI/capacity differs");
        Ok(Self{program,layout,capacity})
    }
    pub fn capacity(&self)->usize{self.capacity}
    /// Metadata validation precedes upload and every mutation. Segments and
    /// destination slots are unique and contain only actually verified inputs.
    pub fn table(&self,entries:&[(usize,usize,usize)],available:&[(usize,usize)])->Result<Vec<i32>>{
        build_table(self.layout,self.capacity,entries,available)
    }
    /// # Safety
    /// Table was checked by `table` and uploaded before this stream reaches the
    /// launch. Source planes and disjoint live backing survive commit/drain.
    pub unsafe fn launch(&self,checkpoint:Buffer,replay:Buffer,table:Buffer,live:Buffer,
        sequences:usize,stream:*mut c_void)->Result<()> {
        let l=self.layout;
        ensure!((1..=self.capacity).contains(&sequences),"FIR commit live sequence count");
        let spans=[(checkpoint,l.history_bytes()?),(replay,l.replay_bytes()?),(table,sequences*12),(live,l.history_bytes()?)];
        for (buffer,bytes) in spans {
            ensure!(!buffer.ptr.is_null() && buffer.device_id==live.device_id && buffer.bytes>=bytes
                && buffer.ptr as usize%4==0 && (buffer.ptr as usize).checked_add(bytes).is_some(),"FIR commit operand extent/device/alignment");
        }
        let out=live.ptr as usize;
        for (source,bytes) in spans[..3].iter().copied(){
            let from=source.ptr as usize;
            ensure!(out+l.history_bytes()?<=from || from+bytes<=out,"FIR commit source aliases live output");
        }
        let scalars=[Scalar::I32(i32::try_from(sequences)?),Scalar::I32(i32::try_from(l.slots)?),Scalar::I32(i32::try_from(l.width)?),
            Scalar::I64(i64::try_from(l.lag_stride)?),Scalar::I64(i64::try_from(l.layer_stride)?),
            Scalar::I64(i64::try_from(l.slot_stride)?),Scalar::I64(i64::try_from(l.replay_row_stride)?),Scalar::I64(i64::try_from(l.replay_layer_stride)?)];
        // SAFETY: validated exact physical planes; caller retains initialized
        // checked metadata and all storage through ordered stream completion.
        unsafe{self.program.launch(&[checkpoint.ptr,replay.ptr,table.ptr,live.ptr],&scalars,stream)}
    }
}

fn build_table(layout:Layout,cap:usize,entries:&[(usize,usize,usize)],available:&[(usize,usize)])->Result<Vec<i32>>{
    ensure!(cap<=layout.replay_rows,"FIR commit capacity exceeds replay layout");
    ensure!(!entries.is_empty() && entries.len()<=cap && entries.len()==available.len(),"FIR commit sequence inventory differs");
    let mut slots=std::collections::BTreeSet::new();
    let mut table=vec![0;entries.len()*3];
    for (index,(&(slot,first,keep),&(start,count))) in entries.iter().zip(available).enumerate(){
        ensure!(slot<layout.slots && slots.insert(slot) && first==start && count>0 && count<=cap
            && keep<=count && first.checked_add(count).is_some_and(|end|end<=cap),"FIR commit outside staged segment or aliases slot");
        // The lag-3 FIR commit kernel caps its `keep` scalar at 8 while `replay` holds `cap`
        // rows (64 or 128). For `keep > 3`, all three lags read exclusively from the final
        // three accepted replay rows `first + keep - 3 .. first + keep`, which is bit-identical
        // to passing `(first + keep - 3, 3)`.
        let (table_first,table_keep)=if keep>3 {(first+keep-3,3)} else {(first,keep)};
        table[index]=i32::try_from(slot)?;
        table[entries.len()+index]=i32::try_from(table_first)?;
        table[2*entries.len()+index]=i32::try_from(table_keep)?;
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_128_layout_supports_64_and_128_row_commit_strides() -> Result<()> {
        let layout = Layout::contiguous_rows(64, 6656, 128)?;
        assert_eq!(layout.replay_rows, 128);
        assert_eq!(layout.replay_row_stride, 6656);
        assert_eq!(layout.replay_layer_stride, 128 * 6656);
        assert!(layout.replay_layer_stride >= 64 * layout.replay_row_stride);
        assert!(layout.replay_layer_stride >= 128 * layout.replay_row_stride);
        assert_eq!(layout.replay_bytes()?, 48 * 128 * 6656 * 4);
        Ok(())
    }

    /// Simulate the exact `fir_history_commit` kernel guard and lag-3 gather for one sequence.
    fn simulate_kernel(cap:usize,prior:[u32;3],replay:&[u32],table_first:i32,table_keep:i32)->Option<[u32;3]>{
        let (first,keep)=(usize::try_from(table_first).ok()?,usize::try_from(table_keep).ok()?);
        // Mirror `fir_history.py:58-59` exactly, including `keep <= 8`.
        if first>cap || keep>8 || first+keep>cap {return None;}
        let mut live=[0u32;3];
        for lag in 0..3 {
            live[lag]=if keep>lag {replay[first+keep-1-lag]} else {prior[lag-keep]};
        }
        Some(live)
    }

    #[test]
    fn copy_extended_accepts_reproduce_exact_lag3_fir_state_under_kernel_keep_guard()->Result<()>{
        let prior=[900,901,902];
        for (replay_rows,cap) in [(64usize,64usize),(128,64),(128,128)]{
            let layout=Layout::contiguous_rows(16,6656,replay_rows)?;
            let replay: Vec<u32> = (1000..1000 + replay_rows as u32).collect();
            for (first,count) in [(0,1),(0,8),(0,40),(0,cap),(7,40),(cap-40,40),(cap-9,9)]{
                for keep in 0..=count {
                    let table=build_table(layout,cap,&[(3,first,keep)],&[(first,count)])?;
                    let live=simulate_kernel(cap,prior,&replay,table[1],table[2])
                        .context("kernel guard rejected encoded table entry")?;
                    let mut serial=prior.to_vec();
                    for &tok in &replay[first..first+keep]{serial.insert(0,tok);serial.truncate(3);}
                    assert_eq!(live.as_slice(),serial.as_slice(),"replay={replay_rows} cap={cap} first={first} count={count} keep={keep}");
                }
            }
            assert!(build_table(layout,cap,&[(0,0,cap+1)],&[(0,cap)]).is_err());
            assert!(build_table(layout,cap,&[(0,1,cap)],&[(1,cap)]).is_err());
            assert!(build_table(layout,cap,&[(0,0,1),(0,1,1)],&[(0,1),(1,1)]).is_err());
        }
        Ok(())
    }
}
