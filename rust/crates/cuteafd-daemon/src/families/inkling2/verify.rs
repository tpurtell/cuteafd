//! Accepted-prefix FIR verification owners on the shared target contract.
//! Checkpoint and replay are one independently admitted mutable storage set.
use crate::shared::{memory::{device::{Allocation,Stream},Tag},qwen::MarkLayout};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::{fir_history::{Commit,Layout},programs::Programs,CuteafdDeviceBuffer as Buffer};
use std::{cell::{Cell,RefCell},rc::Rc,sync::atomic::{AtomicU64,Ordering}};

static NEXT_OWNER:AtomicU64=AtomicU64::new(1);
pub(crate) struct Mark<'a>{owner:u64,generation:u64,checkpoint:Rc<Allocation<'a>>,pub slots:Vec<(usize,usize)>}
pub(crate) struct TargetMark<'a>{pub conv:Vec<Mark<'a>>,pub placements:Vec<super::cache::Placement>}
#[derive(Clone)]
pub(crate) struct Submitted{pub slot:usize,pub start:usize,pub first:usize,pub rows:usize}
pub(crate) struct Verify<'a>{
 pub checkpoint:Rc<Allocation<'a>>,pub replay:Allocation<'a>,table:Allocation<'a>,
 commit:Commit<'a>,layout:Layout,mark_layout:MarkLayout,stream:Rc<Stream<'a>>,
 decode_rows:usize,owner:u64,generation:Cell<u64>,pending:RefCell<Option<Vec<Submitted>>>,
}
impl<'a> Verify<'a>{
 pub fn new(programs:&'a Programs<'a>,stream:Rc<Stream<'a>>,slots:usize,width:usize,decode_rows:usize)->Result<Self>{
  let replay_rows=cuteafd_loader::families::inkling2::execution::replay_capacity(decode_rows);
  let layout=Layout::contiguous_rows(slots,width,replay_rows)?;
  let commit=Commit::load(programs,layout)?;
  let mark_layout=MarkLayout::new(&[(stream.device.id,layout.history_bytes()?)])?;
  let checkpoint=Rc::new(Allocation::new(stream.device,layout.history_bytes()?,Tag::kv("inkling2/verify-conv-checkpoint"))?);
  let replay=Allocation::new(stream.device,layout.replay_bytes()?,Tag::workspace("inkling2/verify-conv-replay"))?;
  // The attention peer produces only its K/V replay channels. Its unused
  // hidden FIR lanes remain exact zeros across every accepted-prefix commit.
  if width<11264{stream.device.library.cuda_zero_bytes(replay.buffer,replay.buffer.bytes)?;}
  let table=Allocation::new(stream.device,replay_rows*12,Tag::workspace("inkling2/verify-conv-table"))?;
  let owner=NEXT_OWNER.fetch_update(Ordering::Relaxed,Ordering::Relaxed,|id|id.checked_add(1))
   .map_err(|_|anyhow::anyhow!("Inkling verify owner identity exhausted"))?;
  Ok(Self{checkpoint,replay,table,commit,layout,mark_layout,stream,decode_rows,owner,generation:Cell::new(0),pending:Default::default()})
 }
 pub fn checkpoint(&self,live:Buffer,slots:&[(usize,usize)])->Result<Mark<'a>>{
  ensure!(self.pending.borrow().is_none() && !slots.is_empty(),"Inkling verify requires committed state");
  let mut unique=std::collections::BTreeSet::new();
  ensure!(slots.iter().all(|&(slot,_)|slot<self.layout.slots && unique.insert(slot)),"Inkling verify slot aliases or exceeds admission");
  let generation=self.generation.get().checked_add(1).context("Inkling verify generation exhausted")?;self.generation.set(generation);
  ensure!(live.bytes==self.layout.history_bytes()? && self.mark_layout.bytes==live.bytes,"Inkling verify live backing differs from admitted mark");
  // SAFETY: complete contiguous live FIR state and checkpoint are disjoint,
  // all readers/producers are ordered on this same retained target stream.
  unsafe{crate::shared::memory::device::copy_device_ordered(self.stream.device.library,self.checkpoint.buffer,live,live.bytes,self.stream.raw)?;}
  Ok(Mark{owner:self.owner,generation,checkpoint:Rc::clone(&self.checkpoint),slots:slots.to_vec()})
 }
 fn validate(&self,mark:&Mark<'_>)->Result<()>{ensure!(mark.owner==self.owner && mark.generation==self.generation.get()
  && Rc::ptr_eq(&mark.checkpoint,&self.checkpoint),"stale or foreign Inkling verify mark");Ok(())}
 pub fn submitted(&self,mark:&Mark<'_>,rows:Vec<Submitted>)->Result<()>{
  self.validate(mark)?;ensure!(self.pending.borrow().is_none() && rows.len()==mark.slots.len(),"Inkling pending verify inventory differs");
  let mut total=0;
  for (row,&(slot,start)) in rows.iter().zip(&mark.slots){
   ensure!(row.slot==slot && row.start==start && row.first==total && (1..=self.decode_rows).contains(&row.rows),"Inkling verified receipt differs from marked slots");
   total+=row.rows;
  }ensure!(total<=self.decode_rows,"Inkling verified rows exceed bounded replay");
  *self.pending.borrow_mut()=Some(rows);Ok(())
 }
 pub fn commit(&self,mark:&Mark<'_>,live:Buffer,accepted:&[crate::shared::speculation::target::AcceptedPrefix])->Result<Vec<(usize,usize)>>{
  self.validate(mark)?;
  let pending=self.pending.borrow();let submitted=pending.as_ref().context("Inkling accepted commit lacks successful verify")?;
  ensure!(accepted.len()==submitted.len(),"Inkling accepted slot count differs");
  let mut entries=Vec::with_capacity(accepted.len());let mut available=Vec::with_capacity(accepted.len());let mut frontiers=Vec::new();
  for row in submitted{
   let keep=accepted.iter().find(|entry|entry.slot==row.slot).context("Inkling accepted slot omitted")?.rows;
   ensure!(accepted.iter().filter(|entry|entry.slot==row.slot).count()==1,"Inkling accepted slot repeated");
   entries.push((row.slot,row.first,keep));available.push((row.first,row.rows));frontiers.push((row.slot,row.start+keep));
  }
  let table=self.commit.table(&entries,&available)?;
  let bytes=table.iter().flat_map(|word|word.to_le_bytes()).collect::<Vec<_>>();
  // SAFETY: one checked small host metadata upload, followed by one FIR
  // gather across every sequence/layer/channel, no per-sequence launch loop.
  unsafe{self.stream.device.library.copy_h2d_ordered(self.table.buffer,&bytes,self.stream.raw)?;
   self.commit.launch(self.checkpoint.buffer,self.replay.buffer,self.table.buffer,live,accepted.len(),self.stream.raw)?;}
  drop(pending);self.pending.borrow_mut().take();self.generation.set(self.generation.get().checked_add(1).context("verify generation overflow")?);
  Ok(frontiers)
 }
 pub fn restore(&self,mark:&Mark<'_>,live:Buffer)->Result<Vec<(usize,usize)>>{
  self.validate(mark)?;
  // SAFETY: checkpoint precedes verification, all restored state belongs to
  // this generation; append-only KV/rings beyond logical frontiers are dead.
  unsafe{crate::shared::memory::device::copy_device_ordered(self.stream.device.library,live,self.checkpoint.buffer,live.bytes,self.stream.raw)?;}
  self.pending.borrow_mut().take();Ok(mark.slots.clone())
 }
}
impl Drop for Verify<'_>{fn drop(&mut self){self.stream.drain_or_abort("Inkling FIR verify retirement");}}
