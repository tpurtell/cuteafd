//! Persistent request state, separate from in-flight row scratch. Shared prefix
//! copy fences join every reader before snapshots, restores and retirement.
use crate::shared::{memory::{device::{Allocation, Stream}, receipts::Inventory},
    prefix::{PrefixCopies, Reader}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::inkling2::execution::CacheLayout;
use std::{collections::BTreeMap, rc::Rc};

pub(crate) fn view(buffer: Buffer, offset: usize, bytes: usize) -> Result<Buffer> {
    ensure!(bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "Inkling cache view exceeds backing");
    Ok(Buffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

pub(crate) struct CacheBank<'a> {
    // Copies drain while all readers and their storage still exist.
    copies: PrefixCopies<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    active: Allocation<'a>,
    conv: Allocation<'a>,
    marks: Option<Allocation<'a>>,
    pages: BTreeMap<usize, Allocation<'a>>,
    pub layout: CacheLayout,
    mark_slots: usize,
}
impl<'a> CacheBank<'a> {
    pub fn require_reader(&self, stream: &Stream<'_>) -> Result<()> {
        ensure!(self.readers.iter().any(|reader| reader.raw == stream.raw
            && reader.device.id == stream.device.id
            && std::ptr::eq(reader.device.library, stream.device.library)),
            "Inkling arithmetic stream is not fenced by persistent cache");
        Ok(())
    }
    /// All requests, including tails, are admitted before construction.
    pub fn new(readers: Vec<Rc<Stream<'a>>>, layout: CacheLayout, mark_slots: usize) -> Result<Self> {
        let first = readers.first().context("Inkling cache requires readers")?;
        let device = first.device;
        ensure!(readers.iter().all(|stream| stream.device.id == device.id
            && std::ptr::eq(stream.device.library, device.library)), "Inkling cache reader devices differ");
        let active = Allocation::new(device, layout.active_bytes()?, Tag::kv("inkling2/sequence-state"))?;
        let conv = Allocation::new(device, layout.conv.bytes()?, Tag::kv("inkling2/convolution-state"))?;
        let marks = (mark_slots > 0).then(|| Allocation::new(device, layout.retained_bytes(mark_slots)?,
            Tag::new(cuteafd_core::memory_layout::MemoryKind::Prefix, "inkling2/prefix-marks"))).transpose()?;
        let pages = layout.global_layers.iter().map(|&layer| Ok((layer,
            Allocation::new(device, layout.layer_page_bytes()?, Tag::kv("inkling2/global-pages"))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let copies = PrefixCopies::new(device.library, vec![readers.iter().map(|stream|
            Reader { device: stream.device.id, stream: stream.raw }).collect()])?;
        let bank = Self { copies, readers, active, conv, marks, pages, layout, mark_slots };
        bank.copies.ordered(0, |stream| {
            // SAFETY: all allocations succeeded before initialization; the bank
            // retains every destination and joins readers on both sides.
            unsafe {
                device.library.cuda_zero_bytes_async(bank.active.buffer, bank.active.buffer.bytes, stream)?;
                device.library.cuda_zero_bytes_async(bank.conv.buffer, bank.conv.buffer.bytes, stream)?;
            }
            Ok(())
        })?;
        Ok(bank)
    }
    pub fn field(&self, sequence: usize, layer: usize, name: &str) -> Result<Buffer> {
        if name == "ring" {
            let span = self.layout.ring_field(sequence, layer)?;
            return view(self.active.buffer, span.start, span.len());
        }
        let (span, _) = self.layout.conv.field(sequence, layer, name)?;
        view(self.conv.buffer, span.start, span.len())
    }
    pub fn field_bank(&self, layer: usize, name: &str) -> Result<Buffer> {
        let first = self.field(0, layer, name)?;
        let stride = if name == "ring" { self.layout.ring_sequence_bytes }
            else { self.layout.conv.slot_stride*4 };
        let backing = if name == "ring" { self.active.buffer } else { self.conv.buffer };
        view(backing, first.ptr as usize-backing.ptr as usize,
            (self.layout.sequences-1)*stride + first.bytes)
    }
    pub fn global(&self, layer: usize) -> Result<Buffer> {
        self.pages.get(&layer).map(|owner| owner.buffer).context("Inkling layer has no global KV pages")
    }
    pub fn page(&self, layer: usize, page: usize) -> Result<Buffer> {
        ensure!(page < self.layout.pool_pages, "Inkling physical page outside pool");
        view(self.global(layer)?, page*self.layout.page_bytes, self.layout.page_bytes)
    }
    fn rings_for_sequence(&self, sequence: usize) -> Result<Buffer> {
        ensure!(sequence < self.layout.sequences, "Inkling sequence outside cache");
        view(self.active.buffer, sequence*self.layout.ring_sequence_bytes, self.layout.ring_sequence_bytes)
    }
    pub fn mark(&self, slot: usize) -> Result<Buffer> {
        ensure!(slot < self.mark_slots, "Inkling mark outside arena");
        view(self.marks.as_ref().context("Inkling mark arena absent")?.buffer,
            slot*self.layout.sequence_bytes, self.layout.sequence_bytes)
    }
    pub fn reset(&self, sequence: usize) -> Result<()> {
        let rings = self.rings_for_sequence(sequence)?;
        let conv = self.conv_for_sequence(sequence)?;
        self.copies.ordered(0, |stream| {
            // SAFETY: copy fences join every arithmetic reader before overwrite.
            unsafe {
                self.readers[0].device.library.cuda_zero_bytes_async(rings, rings.bytes, stream)?;
                self.readers[0].device.library.cuda_zero_bytes_async(conv, conv.bytes, stream)
            }
        })
    }
    pub fn conv_for_sequence(&self, sequence: usize) -> Result<Buffer> {
        ensure!(sequence < self.layout.sequences, "Inkling convolution sequence outside cache");
        let bytes = self.layout.conv.slot_stride*4;
        view(self.conv.buffer, sequence*bytes, bytes)
    }
    pub fn conv_backing(&self) -> Buffer { self.conv.buffer }
    fn state_copy(&self, sequence: usize, slot: usize, capture: bool) -> Result<()> {
        let mark = self.mark(slot)?;
        let rings = self.rings_for_sequence(sequence)?;
        let conv = self.conv_for_sequence(sequence)?;
        let saved_rings = view(mark, 0, rings.bytes)?;
        let saved_conv = view(mark, rings.bytes, conv.bytes)?;
        let copies = if capture { vec![(0, saved_rings, rings), (0, saved_conv, conv)] }
            else { vec![(0, rings, saved_rings), (0, conv, saved_conv)] };
        // SAFETY: exact disjoint ring/convolution views share registered readers;
        // one prefix fence orders both backings before snapshot publication.
        unsafe { self.copies.copy_d2d(&copies) }
    }
    pub fn capture(&self, sequence: usize, slot: usize) -> Result<()> { self.state_copy(sequence, slot, true) }
    pub fn restore(&self, slot: usize, sequence: usize) -> Result<()> { self.state_copy(sequence, slot, false) }
    fn copy(&self, dst: Buffer, src: Buffer) -> Result<()> {
        ensure!(dst.bytes == src.bytes && dst.ptr != src.ptr, "Inkling cache copy extent/alias");
        // SAFETY: checked disjoint sequence/mark regions survive all fenced
        // copies; PrefixCopies orders every registered consumer on both sides.
        unsafe { self.copies.copy_d2d(&[(0, dst, src)]) }
    }
    pub fn copy_page(&self, from: usize, to: usize) -> Result<()> {
        self.copy_page_rows(from, to, 64)
    }
    pub fn copy_page_rows(&self, from: usize, to: usize, rows: usize) -> Result<()> {
        ensure!(from != to && (1..=64).contains(&rows), "Inkling page tail copy aliases or exceeds page");
        let bytes = self.layout.page_bytes / 64 * rows;
        let copies = self.layout.global_layers.iter().map(|&layer|
            Ok((0, view(self.page(layer, to)?, 0, bytes)?, view(self.page(layer, from)?, 0, bytes)?)))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: physical page IDs are distinct; the retained bank owns every
        // source/destination until the shared prefix fence drains.
        unsafe { self.copies.copy_d2d(&copies) }
    }
    pub fn mark_slots(&self) -> usize { self.mark_slots }
    pub fn drain(&self) -> Result<()> { self.copies.drain() }
    pub fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        let location = inventory.location(self.readers[0].device.id)?;
        let cuteafd_loader::placement::planner::Device::Coordinator(gpu) = location.device
            else { anyhow::bail!("Inkling cache is not coordinator-owned"); };
        self.active.inventory(inventory, &format!("inkling2.cache.{gpu}.active"))?;
        self.conv.inventory(inventory, &format!("inkling2.cache.{gpu}.conv"))?;
        if let Some(marks) = &self.marks { marks.inventory(inventory, &format!("inkling2.cache.{gpu}.marks"))?; }
        for (layer, owner) in &self.pages { owner.inventory(inventory, &format!("inkling2.kv.{layer}.{gpu}.records"))?; }
        self.copies.allocation_receipts(inventory, &format!("inkling2.cache.{gpu}.copies"))
    }
}
/// Logical pages and frontiers belong to the shared scheduler. Mutable model
/// bytes stay in the bank and are never reconstructed from a continuation.
#[derive(Clone, Debug)]
pub(crate) struct Placement {
    pub units: Vec<u32>,
    pub slot: usize,
    pub len: usize,
    pub state_len: usize,
    pub media: Option<cuteafd_engine::media::RequestMedia>,
}
impl crate::shared::qwen::scheduler::Placement for Placement {
    fn len(&self) -> usize { self.len }
    fn set_len(&mut self, len: usize) { self.len = len; }
    fn units(&self) -> &[u32] { &self.units }
    fn slot(&self) -> i32 { self.slot as i32 }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] { self.media.as_ref().map_or(&[], |media| media.spans()) }
}

#[derive(Clone)]
pub(crate) struct InklingPrefix<'a> { bank:Rc<CacheBank<'a>>, peer:Option<Rc<CacheBank<'a>>>, draft:Option<Rc<super::draft::Binding<'a>>> }
impl<'a> InklingPrefix<'a> {
    pub fn new(bank: Rc<CacheBank<'a>>) -> Self { Self {bank,peer:None,draft:None} }
    pub fn for_execution(execution: &super::engine::Execution<'a>) -> Self {
        Self { bank: execution.requests.cache.clone(), peer: execution.peer.as_ref().map(|peer| peer.requests.cache.clone()),
            draft: execution.draft.clone() }
    }
    fn banks(&self) -> impl Iterator<Item = &Rc<CacheBank<'a>>> {
        std::iter::once(&self.bank).chain(self.peer.as_ref())
    }
    pub fn snapshot_owners(&self)->Vec<Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner+'a>>{
        let mut owners:Vec<Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner+'a>>=self.banks().map(|bank| bank.clone() as Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner>).collect();
        if let Some(draft)=&self.draft{owners.push(draft.clone());}owners
    }
    fn validate(&self, placement: &Placement, len: usize) -> Result<()> {
        ensure!(placement.slot < self.bank.layout.sequences && len.div_ceil(64) <= placement.units.len()
            && placement.units.iter().all(|&page| (page as usize) < self.bank.layout.pool_pages),
            "Inkling prefix placement outside admitted state/pages");
        ensure!(placement.units.iter().copied().collect::<std::collections::BTreeSet<_>>().len()
            == placement.units.len(), "Inkling prefix placement aliases physical pages");
        Ok(())
    }
    /// Shared request-slot binding entry point for both serving (`PrefixFamily::bind`)
    /// and golden/width-invariance runners: validates `placement`, resets every rank's
    /// target cache bank, binds the recurrent drafter context slot when attached, and
    /// optionally admits draft pages up to `admit_capacity`.
    pub fn bind_slot(&self, placement: &mut Placement, admit_capacity: Option<usize>) -> Result<()> {
        self.validate(placement, placement.len)?;
        for bank in self.banks() {
            bank.reset(placement.slot)?;
            bank.drain()?;
        }
        if let Some(draft) = &self.draft {
            draft.drafter.bind_slot(placement.slot, 0, admit_capacity)?;
        }
        placement.state_len = 0;
        Ok(())
    }
}
impl cuteafd_engine::prefix::PrefixFamily for InklingPrefix<'_> {
    type Placement = Placement;
    fn record_format(&self) -> &'static str { self.bank.layout.kv_format.record_format() }
    fn layout(&self) -> cuteafd_engine::prefix::FamilyLayout {
        use cuteafd_engine::prefix::{FamilyLayout, MarkStore, ReuseRule};
        FamilyLayout { page_rows: 64, pages: self.bank.layout.pool_pages,
            page_bytes: self.banks().map(|bank| bank.layout.page_bytes * bank.layout.global_layers.len()).sum(),
            mark_bytes: self.banks().map(|bank| bank.layout.sequence_bytes).sum::<usize>()+self.draft.as_ref().map_or(0,|draft|draft.marks.iter().map(|marks|marks.bytes()).sum()), draft_bytes: 0,
            rule: ReuseRule::EXACT, mark_store: MarkStore::Arena, page_owners: Default::default() }
    }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.units }
    fn draft_page_bytes(&self)->usize {self.draft.as_ref().map_or(0,|draft|draft.drafter.page_bytes())}
    fn quarantine_draft_pages(&self,placement:&Placement) {
        if let Some(draft)=&self.draft {draft.drafter.quarantine_pages(placement.slot);}
    }
    fn draft_demand(&self,placement:&Placement)->Option<cuteafd_engine::reserve::DraftDemand> {
        self.draft.as_ref()?.drafter.reservation(placement.slot)
    }
    fn draft_reservable(&self)->Option<usize> {Some(self.draft.as_ref()?.drafter.reservable())}
    fn draft_lifetime_rows(&self)->Option<usize> {Some(self.draft.as_ref()?.drafter.lifetime_rows())}
    fn draft_admission(&self,origin:usize)->Option<cuteafd_engine::reserve::DraftDemand> {
        Some(self.draft.as_ref()?.drafter.admission(origin))
    }
    fn admit_draft_pages(&self,p:&mut Placement,capacity:usize)->Result<(),cuteafd_engine::prefix::BoxError>{
        if let Some(draft)=&self.draft{draft.drafter.admit_pages(p.slot,capacity)?;}Ok(())
    }
    fn draft_pages_valid(&self,a:&cuteafd_engine::prefix::DraftPageAttachment)->bool{self.draft.as_ref().is_some_and(|d|d.drafter.pages_valid(a))}
    fn draft_pages_revocable(&self,a:&cuteafd_engine::prefix::DraftPageAttachment)->bool{self.draft.as_ref().is_some_and(|d|d.drafter.pages_revocable(a))}
    fn retain_draft_pages(&self,a:&cuteafd_engine::prefix::DraftPageAttachment)->Result<(),cuteafd_engine::prefix::BoxError>{
        self.draft.as_ref().context("Inkling prefix has no drafter")?.drafter.retain_pages(a)?;Ok(())
    }
    fn capture_draft_pages(&self,p:&Placement,len:usize)->Result<Option<cuteafd_engine::prefix::DraftPageAttachment>,cuteafd_engine::prefix::BoxError>{
        Ok(self.draft.as_ref().map(|draft|draft.drafter.capture_pages(p.slot,len)).transpose()?)
    }
    fn release_draft_pages(&self,a:&cuteafd_engine::prefix::DraftPageAttachment)->Result<(),cuteafd_engine::prefix::BoxError>{
        self.draft.as_ref().context("Inkling prefix has no drafter")?.drafter.release_pages(a)?;Ok(())
    }
    fn restore_draft_pages(&self,a:Option<&cuteafd_engine::prefix::DraftPageAttachment>,p:&mut Placement,frontier:usize)->Result<(),cuteafd_engine::prefix::BoxError>{
        if let Some(draft)=&self.draft{
            if let Some(a)=a{draft.drafter.restore_pages(p.slot,a,frontier)?;}
            else{if frontier!=0{return Err("Inkling draft restore needs exact full-context attachment".into());}draft.drafter.reset_context(p.slot,0)?;}
        }Ok(())
    }
    fn promote_draft_pages(&self,a:&cuteafd_engine::prefix::DraftPageAttachment)->Result<cuteafd_engine::prefix::DraftPageAttachment,cuteafd_engine::prefix::BoxError>{
        Ok(self.draft.as_ref().context("Inkling prefix has no drafter")?.drafter.promote_pages(a)?)
    }
    fn draft_page_segments(&self,page:cuteafd_engine::prefix::DraftPageId)->Vec<cuteafd_hostcache::copy::DeviceRange>{
        self.draft.as_ref().map_or_else(Vec::new,|d|d.drafter.page_segments(page).expect("admitted draft page")
            .into_iter().map(crate::shared::prefix::cuda_copy::range).collect())
    }
    fn bind(&self, placement: &mut Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.bind_slot(placement, None)?;
        Ok(())
    }
    fn discard(&self, placement: &mut Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        for bank in self.banks() { bank.reset(placement.slot)?; }
        if let Some(draft)=&self.draft{draft.drafter.reset_context(placement.slot,0)?;}
        placement.state_len = 0;
        Ok(())
    }
    fn extend_pages(&self, placement: &mut Placement, pages: &[u32]) -> Result<(), cuteafd_engine::prefix::BoxError> {
        let mut extended = placement.clone();
        extended.units.extend_from_slice(pages);
        self.validate(&extended, extended.len)?;
        placement.units = extended.units;
        Ok(())
    }
    fn commit_point(&self, placement: &Placement) -> usize { placement.len.min(placement.state_len) }
    fn capture(&self, slot: cuteafd_engine::prefix::MarkSlot, placement: &Placement, len: usize)
        -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        if len != placement.len || len != placement.state_len {
            return Err("Inkling snapshot frontier does not match committed causal state".into());
        }
        for bank in self.banks() { bank.capture(placement.slot,slot.0 as usize)?; }
        if let Some(draft)=&self.draft{
            let (origin,frontier)=draft.drafter.frontier(placement.slot)?;
            if frontier!=len || draft.marks.len()!=draft.ranks().count(){return Err("draft snapshot frontier or ranks differ from target".into());}
            for (rank,marks) in draft.ranks().zip(&draft.marks){
                rank.stream.device.run(||marks.capture(rank,placement.slot,slot.0 as usize,origin,frontier))?;
            }
        }
        Ok(())
    }
    fn restore(&self, mark: Option<cuteafd_engine::prefix::MarkSlot>, placement: &mut Placement, len: usize)
        -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        let slot = mark.context("Inkling exact restore requires its SWA/convolution mark")?;
        for bank in self.banks() { bank.restore(slot.0 as usize, placement.slot)?; }
        if let Some(draft)=&self.draft{
            if draft.marks.len()!=draft.ranks().count(){return Err("draft snapshot rank marks missing".into());}
            for (rank,marks) in draft.ranks().zip(&draft.marks){
                rank.stream.device.run(||marks.restore(rank,placement.slot,slot.0 as usize,len))?;
            }
        }
        placement.len = len;
        placement.state_len = len;
        Ok(())
    }
    fn copy_rows(&self, copy: cuteafd_engine::prefix::TailCopy) -> Result<(), cuteafd_engine::prefix::BoxError> {
        if copy.owner.is_some() { return Err("Inkling whole-rank pages have no parity owner".into()); }
        for bank in self.banks() { bank.copy_page_rows(copy.from as usize, copy.to as usize, copy.rows)?; }
        Ok(())
    }
    fn drain(&self) -> Result<(), cuteafd_engine::prefix::BoxError> {
        for bank in self.banks() { bank.drain()?; }
        if let Some(draft)=&self.draft{for marks in &draft.marks{marks.drain()?;}draft.drafter.drain()?;}
        Ok(())
    }
    fn page_segments(&self, page: u32) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        self.banks().flat_map(|bank| bank.layout.global_layers.iter().filter_map(move |&layer| bank.page(layer, page as usize).ok()))
            .map(crate::shared::prefix::cuda_copy::range).collect()
    }
    fn mark_segments(&self, slot: cuteafd_engine::prefix::MarkSlot) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        let mut segments=self.banks().filter_map(|bank| bank.mark(slot.0 as usize).ok()).map(crate::shared::prefix::cuda_copy::range).collect::<Vec<_>>();
        if let Some(draft)=&self.draft{segments.extend(draft.marks.iter().filter_map(|marks|marks.mark(slot.0 as usize).ok()).map(crate::shared::prefix::cuda_copy::range));}
        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::Device;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::families::inkling2::{execution::UnitLayout, Inkling2Config};

    #[test]
    fn shared_cache_restores_prompt_and_turn_marks_without_reconstruction() -> Result<()> {
        use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
        use crate::shared::prefix::CudaCopyEngine;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let reader = Rc::new(Stream::new(device)?);
        let mut cfg = Inkling2Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        cfg.layers = 2;
        cfg.local = vec![true, false];
        let bank = Rc::new(CacheBank::new(vec![reader], CacheLayout::new(&cfg, 2, 8)?, 3)?);
        let family = InklingPrefix::new(bank.clone());
        let mut cache: PrefixCache<CudaCopyEngine<'_>> = PrefixCache::new(family.layout(),
            PrefixConfig { entries: 3, mark_slots: 3, keep_logits: true, min_tokens: 1 }, None)?;
        let tokens = (0..65).collect::<Vec<u32>>();
        let mut source = cache.admit(&family, &tokens, 128, false,
            |units| Placement { media: None, units, slot: 0, len: 0, state_len: 0 })?.placement;
        let state = bank.rings_for_sequence(0)?;
        let prompt = vec![0x17; state.bytes];
        library.copy_h2d(state, &prompt)?;
        for &page in &source.units {
            let buffer = bank.page(1, page as usize)?;
            library.copy_h2d(buffer, &vec![page as u8 + 0x30; buffer.bytes])?;
        }
        source.len = tokens.len(); source.state_len = tokens.len();
        assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens, &source, After::from_logits(&[1., 0.], true))?);
        cache.release(&family, &source.units)?;
        let restored = cache.admit(&family, &tokens, 128, false,
            |units| Placement { media: None, units, slot: 1, len: 0, state_len: 0 })?;
        assert_eq!(restored.resume, tokens.len());
        let mut observed = vec![0; state.bytes];
        library.copy_d2h_quiescent(&mut observed, bank.rings_for_sequence(1)?)?;
        assert_eq!(observed, prompt);
        let mut continued = restored.placement;
        let turn = vec![0x6b; state.bytes];
        library.copy_h2d(bank.rings_for_sequence(1)?, &turn)?;
        let mut turn_tokens = tokens.clone(); turn_tokens.push(91);
        continued.len = turn_tokens.len(); continued.state_len = turn_tokens.len();
        assert!(cache.capture(&family, SnapshotKind::Turn, &turn_tokens, &continued, After::from_logits(&[0., 1.], true))?);
        cache.release(&family, &continued.units)?;
        let restored = cache.admit(&family, &turn_tokens, 128, false,
            |units| Placement { media: None, units, slot: 0, len: 0, state_len: 0 })?;
        assert_eq!(restored.resume, turn_tokens.len());
        library.copy_d2h_quiescent(&mut observed, bank.rings_for_sequence(0)?)?;
        assert_eq!(observed, turn);
        cache.release(&family, &restored.placement.units)?;
        let restored = cache.admit(&family, &tokens, 128, false,
            |units| Placement { media: None, units, slot: 1, len: 0, state_len: 0 })?;
        assert_eq!(restored.resume, tokens.len());
        library.copy_d2h_quiescent(&mut observed, bank.rings_for_sequence(1)?)?;
        assert_eq!(observed, prompt, "turn divergence must not mutate the prompt mark");
        cache.release(&family, &restored.placement.units)?;
        Ok(())
    }

    /// #2342: a restore that fails after the engine bound the placement must roll
    /// the slot back, cold-bind, and leave state a plain cold prefill of the same
    /// prompt reaches: no restored tail row or recurrent byte survives.
    #[test]
    fn failed_restore_fallback_is_a_cold_prefill_and_binds_byte_identically() -> Result<()> {
        use crate::shared::prefix::CudaCopyEngine;
        use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let reader = Rc::new(Stream::new(device)?);
        let mut cfg = Inkling2Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        cfg.layers = 2;
        cfg.local = vec![true, false];
        let bank = Rc::new(CacheBank::new(vec![reader], CacheLayout::new(&cfg, 3, 8)?, 3)?);
        let family = InklingPrefix::new(bank.clone());
        let mut cache: PrefixCache<CudaCopyEngine<'_>> = PrefixCache::new(family.layout(),
            PrefixConfig { entries: 3, mark_slots: 3, keep_logits: true, min_tokens: 1 }, None)?;
        let tokens = (0..65).collect::<Vec<u32>>();
        let mut source = cache.admit(&family, &tokens, 128, false,
            |units| Placement { media: None, units, slot: 0, len: 0, state_len: 0 })?.placement;
        prefill(&library, &bank, &mut source, &tokens, 0x51)?;
        assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens, &source,
            After::from_logits(&[1.0, 0.0], true))?);
        cache.release(&family, &source.units)?;

        // The restore attempt binds slot 1 with one unit missing: the family's
        // restore rejects the frontier after bind, so the engine must drain,
        // discard the partial placement, and cold-bind a fresh one.
        let builds = std::cell::Cell::new(0);
        let mut fallback = cache.admit(&family, &tokens, 128, false, |mut units| {
            if builds.replace(builds.get() + 1) == 0 { units.truncate(1); }
            Placement { media: None, units, slot: 1, len: 0, state_len: 0 }
        })?;
        assert_eq!((fallback.resume, cache.stats().restore_failures, builds.get()), (0, 1, 2));
        assert_eq!((fallback.placement.len, fallback.placement.state_len), (0, 0));
        for buffer in [bank.rings_for_sequence(1)?, bank.conv_for_sequence(1)?] {
            let mut state = vec![0; buffer.bytes];
            library.copy_d2h_quiescent(&mut state, buffer)?;
            assert!(state.iter().all(|&byte| byte == 0), "cold bind must reset the abandoned slot");
        }

        let mut cold = cache.admit_cold(&family, tokens.len(), 128,
            |units| Placement { media: None, units, slot: 2, len: 0, state_len: 0 })?;
        prefill(&library, &bank, &mut fallback.placement, &tokens, 0x51)?;
        prefill(&library, &bank, &mut cold.placement, &tokens, 0x51)?;
        assert_eq!(fallback.placement.units.len(), cold.placement.units.len());
        for (index, (&left_page, &right_page)) in fallback.placement.units.iter()
            .zip(&cold.placement.units).enumerate() {
            let rows = tokens.len().saturating_sub(index * 64).min(64);
            if rows == 0 { break; }
            let bytes = bank.layout.page_bytes / 64 * rows;
            for &layer in &bank.layout.global_layers {
                let mut left = vec![0; bytes];
                let mut right = vec![0; bytes];
                library.copy_d2h_quiescent(&mut left, view(bank.page(layer, left_page as usize)?, 0, bytes)?)?;
                library.copy_d2h_quiescent(&mut right, view(bank.page(layer, right_page as usize)?, 0, bytes)?)?;
                assert_eq!(left, right, "page {index} layer {layer}: the fallback is not a cold prefill");
            }
        }
        cache.release(&family, &fallback.placement.units)?;
        cache.release(&family, &cold.placement.units)?;
        Ok(())
    }

    /// Writes the rows a prefill would write from `placement.len` onward, then the
    /// recurrent state the whole prompt reaches. A cold fallback that resumed past
    /// zero leaves its restored rows untouched and differs from a plain cold arm.
    fn prefill(library: &cuteafd_ffi::NativeLibrary, bank: &CacheBank<'_>, placement: &mut Placement,
        tokens: &[u32], seed: u8) -> Result<()> {
        let start = placement.len;
        for (index, &page) in placement.units.iter().enumerate() {
            let first = index * 64;
            if first >= tokens.len() { break; }
            let end = (first + 64).min(tokens.len());
            if end <= start { continue; }
            let row_first = start.max(first) - first;
            let rows = end - start.max(first);
            let row_bytes = bank.layout.page_bytes / 64;
            let bytes = (0..rows * row_bytes)
                .map(|i| seed.wrapping_add((first + i / row_bytes * 31 + i % row_bytes) as u8)).collect::<Vec<_>>();
            for &layer in &bank.layout.global_layers {
                library.copy_h2d(view(bank.page(layer, page as usize)?, row_first * row_bytes,
                    rows * row_bytes)?, &bytes)?;
            }
        }
        for buffer in [bank.rings_for_sequence(placement.slot)?, bank.conv_for_sequence(placement.slot)?] {
            library.copy_h2d(buffer, &vec![seed.wrapping_add(0x9d); buffer.bytes])?;
        }
        placement.len = tokens.len();
        placement.state_len = tokens.len();
        Ok(())
    }

    #[test]
    fn target_and_draft_banks_share_combined_budget_and_last_slot_bounds() -> Result<()> {
        use crate::shared::speculation::block_dspark::WindowSnapshots;
        use cuteafd_loader::families::inkling2::draft::prefix_mark_slots;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let mut cfg = Inkling2Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        cfg.layers = 2; cfg.local = vec![true, false];
        let planes = vec![4096];
        let (draft_bytes, _) = WindowSnapshots::bytes(&planes, 1)?;
        for host in [false, true] {
            let reader = Rc::new(Stream::new(device)?);
            let layout = CacheLayout::new(&cfg, 2, 1)?;
            let budget = 3 * (layout.sequence_bytes + draft_bytes) as u64;
            let slots = prefix_mark_slots(2, 4, layout.sequence_bytes as u64, draft_bytes as u64, budget, host)?;
            assert_eq!(slots, if host { 2 } else { 3 });
            let target = CacheBank::new(vec![reader.clone()], layout, slots)?;
            let draft = WindowSnapshots::new(&reader, planes.clone(), slots)?;
            assert_eq!(target.mark_slots(), draft.backing().buffer.bytes / draft.mark_bytes());
            let live = Allocation::new(device, planes[0], Tag::drafter_kv("test/inkling-draft-plane"))?;
            library.copy_h2d(live.buffer, &vec![0x5a; planes[0]])?;
            target.capture(0, slots - 1)?;
            // SAFETY: fixture-backed, disjoint live/mark allocations remain alive
            // through both registered copy fences and the final drain.
            unsafe { draft.capture(slots - 1, &[live.buffer], 0, 64)?; }
            target.drain()?; draft.drain()?;
            assert!(target.mark(slots).is_err());
            assert!(draft.mark(slots).is_err());
            let mut saved = vec![0; draft.mark_bytes()];
            library.copy_d2h_quiescent(&mut saved, draft.mark(slots - 1)?)?;
            assert!(saved[16..16 + planes[0]].iter().all(|&byte| byte == 0x5a));
        }
        Ok(())
    }

    #[test]
    fn persistent_cache_marks_copy_exact_bytes_and_fence_only_registered_readers() -> Result<()> {
        cache_mark_oracle(cuteafd_loader::families::inkling2::InklingKvCache::Bf16)
    }

    #[test]
    fn fp8_cache_marks_and_tail_scales_restore_byte_exactly() -> Result<()> {
        cache_mark_oracle(cuteafd_loader::families::inkling2::InklingKvCache::Fp8)
    }

    fn cache_mark_oracle(format: cuteafd_loader::families::inkling2::InklingKvCache) -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let reader = Rc::new(Stream::new(device)?);
        let mut cfg = Inkling2Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        cfg.layers = 2;
        cfg.local = vec![true, false];
        let cache = CacheBank::new(vec![reader.clone()], CacheLayout::with_format(&cfg, 2, 3, format)?, 1)?;
        let state = cache.rings_for_sequence(0)?;
        let pattern = (0..state.bytes).map(|byte| (byte.wrapping_mul(17) % 251) as u8).collect::<Vec<_>>();
        let conv = cache.conv_for_sequence(0)?;
        let conv_pattern = (0..conv.bytes).map(|byte| (byte.wrapping_mul(31).wrapping_add(7)%251) as u8).collect::<Vec<_>>();
        library.copy_h2d(state, &pattern)?;
        library.copy_h2d(conv, &conv_pattern)?;
        cache.capture(0, 0)?;
        cache.reset(0)?;
        cache.restore(0, 1)?;
        cache.drain()?;
        let mut restored = vec![0; pattern.len()];
        library.copy_d2h_quiescent(&mut restored, cache.rings_for_sequence(1)?)?;
        assert_eq!(restored, pattern);
        library.copy_d2h_quiescent(&mut restored, cache.rings_for_sequence(0)?)?;
        assert!(restored.iter().all(|&byte| byte == 0));
        let mut restored_conv = vec![0; conv_pattern.len()];
        library.copy_d2h_quiescent(&mut restored_conv, cache.conv_for_sequence(1)?)?;
        assert_eq!(restored_conv, conv_pattern, "all contiguous FIR state bytes restore exactly");
        library.copy_d2h_quiescent(&mut restored_conv, cache.conv_for_sequence(0)?)?;
        assert!(restored_conv.iter().all(|&byte|byte==0));
        let page = cache.page(1, 0)?;
        let page_pattern = vec![0x5a; page.bytes];
        library.copy_h2d(page, &page_pattern)?;
        cache.copy_page(0, 2)?;
        cache.drain()?;
        let mut restored_page = vec![0; page.bytes];
        library.copy_d2h_quiescent(&mut restored_page, cache.page(1, 2)?)?;
        assert_eq!(restored_page, page_pattern);
        let unit = super::super::owners::UnitOwner::new(reader, UnitLayout::serving(&cfg, 64, 256)?)?;
        assert!(unit.bind(&cache, 1, 0).is_ok());
        use super::super::owners::UnitBuffers;
        let segment = unit.segment(&cache, 1, 0, 7..19)?;
        for (name, stride) in [("partial", 4608*4), ("q", 6144*4), ("relative", 48*1024*2),
            ("cache_slots", 8), ("first", 4), ("expert_output", 4608*2)] {
            let whole = unit.buffer(name)?;
            let sliced = segment.buffer(name)?;
            assert_eq!(sliced.ptr as usize - whole.ptr as usize, 7*stride, "{name}");
            assert_eq!(sliced.bytes, 12*stride, "{name}");
        }
        let arena = segment.buffer("residual_arena")?;
        assert_eq!(arena.ptr as usize - unit.buffer("residual_arena")?.ptr as usize, 7*4608*4);
        assert_eq!(segment.layout().rows, 64, "completed-block pitch must remain the admitted unit width");
        assert_eq!(segment.buffer("ring")?.ptr, cache.field(1, 0, "ring")?.ptr);
        assert!(unit.segment(&cache, 0, 0, 63..65).is_err());
        assert!(unit.bind(&cache, 2, 0).is_err());
        assert!(unit.bind(&cache, 0, 2).is_err());
        let stranger = Rc::new(Stream::new(device)?);
        assert!(cache.require_reader(&stranger).is_err());
        let unit = super::super::owners::UnitOwner::new(stranger, UnitLayout::serving(&cfg, 64, 256)?)?;
        assert!(unit.bind(&cache, 0, 0).is_err());
        assert!(cache.field(0, 1, "ring").is_err());
        assert!(cache.copy_page(0, 0).is_err());
        assert!(cache.copy_page(0, 3).is_err());
        use cuteafd_engine::prefix::{MarkSlot, PrefixFamily, TailCopy};
        let cache = Rc::new(cache);
        let family = InklingPrefix::new(cache.clone());
        let mut source = Placement { media: None, units: vec![0, 1], slot: 0, len: 65, state_len: 65 };
        library.copy_h2d(cache.rings_for_sequence(0)?, &pattern)?;
        family.capture(MarkSlot(0), &source, 65).map_err(|e| anyhow::anyhow!("{e}"))?;
        source.state_len = 64;
        assert!(family.capture(MarkSlot(0), &source, 65).is_err());
        let mut destination = Placement { media: None, units: vec![0, 2], slot: 1, len: 0, state_len: 0 };
        family.bind(&mut destination).map_err(|e| anyhow::anyhow!("{e}"))?;
        assert!(family.restore(None, &mut destination, 65).is_err());
        family.restore(Some(MarkSlot(0)), &mut destination, 65).map_err(|e| anyhow::anyhow!("{e}"))?;
        family.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!((destination.len, destination.state_len), (65, 65));
        library.copy_d2h_quiescent(&mut restored, cache.rings_for_sequence(1)?)?;
        assert_eq!(restored, pattern);
        let sentinel = vec![0x33; page.bytes];
        library.copy_h2d(cache.page(1, 2)?, &sentinel)?;
        family.copy_rows(TailCopy { from: 0, to: 2, rows: 1, owner: None })
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        family.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
        library.copy_d2h_quiescent(&mut restored_page, cache.page(1, 2)?)?;
        let record_bytes = format.record_bytes(cfg.kv_heads, cfg.head_dim);
        assert_eq!(&restored_page[..record_bytes], &page_pattern[..record_bytes]);
        assert_eq!(&restored_page[record_bytes..], &sentinel[record_bytes..]);
        assert_eq!(family.page_segments(0).len(), 1);
        assert_eq!(family.mark_segments(MarkSlot(0)).len(), 1);
        assert!(family.extend_pages(&mut destination, &[0]).is_err());
        family.discard(&mut destination).map_err(|e| anyhow::anyhow!("{e}"))?;
        assert_eq!(destination.state_len, 0);
        Ok(())
    }

}

impl crate::shared::prefix::cuda_copy::SnapshotOwner for CacheBank<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<Buffer>> {
        Ok([self.active.buffer,self.conv.buffer].into_iter().chain(self.marks.iter().map(|owner| owner.buffer))
            .chain(self.pages.values().map(|owner| owner.buffer)).collect())
    }
}

impl Drop for CacheBank<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.copies.drain(), "Inkling persistent cache retirement");
    }
}
