//! Exact page-reference snapshots: no recurrent state, private marks or replay.
use super::cache::{view, PageBank, Placement, PAGE_PLANES, PAGE_ROWS};
use crate::shared::{memory::device::Allocation, prefix::{cuda_copy::range, PrefixCopies, Reader}};
use anyhow::{ensure, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily,
    ReuseRule, TailCopy};
use cuteafd_hostcache::copy::DeviceRange;
use std::rc::Rc;

struct Inner<'a> {
    // Copy streams drain before the retained page planes or their readers retire.
    copies: PrefixCopies<'a>,
    banks: Vec<Rc<PageBank<'a>>>,
    /// The first prefill unit's residual plane, fully overwritten by embedding
    /// at begin_unit before arithmetic reads it. Not a table, length or KV byte.
    scratch: Rc<Allocation<'a>>,
    layout: FamilyLayout,
}

#[derive(Clone)]
pub(crate) struct MiniMaxPrefix<'a>(Rc<Inner<'a>>);

impl<'a> MiniMaxPrefix<'a> {
    pub fn new(banks: Vec<Rc<PageBank<'a>>>, residual_scratch: Rc<Allocation<'a>>) -> Result<Self> {
        ensure!(matches!(banks.len(), 1 | 2), "MiniMax prefix requires one or two physical ranks");
        let first = &banks[0];
        let library = first.stream.device.library;
        let mut devices = std::collections::BTreeSet::new();
        let mut layers = std::collections::BTreeSet::new();
        for bank in &banks {
            ensure!(devices.insert(bank.stream.device.id)
                && std::ptr::eq(library, bank.stream.device.library)
                && bank.layout.pool_pages == first.layout.pool_pages
                && bank.layout.requests == first.layout.requests && bank.rope == first.rope
                && bank.model_layers == first.model_layers,
                "MiniMax prefix rank geometry or device differs");
            ensure!(bank.layers.clone().all(|layer| layers.insert(layer)),
                "MiniMax prefix ranks replicate a layer's KV");
        }
        ensure!(layers.iter().copied().eq(0..first.model_layers),
            "MiniMax prefix rank ranges do not cover the complete model");
        ensure!(residual_scratch.buffer.bytes >= 1
            && residual_scratch.buffer.device_id == first.stream.device.id
            && std::ptr::eq(residual_scratch.device.library, library),
            "MiniMax host-tail scratch differs from rank zero");
        let page_bytes = layers.len().checked_mul(PAGE_PLANES.iter().map(|&(_, bytes)| bytes).sum())
            .ok_or_else(|| anyhow::anyhow!("MiniMax prefix page extent overflow"))?;
        let layout = FamilyLayout { page_rows: PAGE_ROWS, pages: first.layout.pool_pages as usize,
            page_bytes, mark_bytes: 0, draft_bytes: 0, rule: ReuseRule::paged(PAGE_ROWS),
            mark_store: MarkStore::Arena, page_owners: Default::default() };
        let copies = PrefixCopies::new(library, banks.iter().map(|bank| bank.readers().iter()
            .map(|reader| Reader { device: reader.device.id, stream: reader.raw }).collect()).collect())?;
        for (rank, bank) in banks.iter().enumerate() {
            copies.ordered(rank, |stream| bank.reserve_tables(stream))?;
        }
        Ok(Self(Rc::new(Inner { copies, banks, scratch: residual_scratch, layout })))
    }

    pub fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.0.banks.iter().flat_map(|bank| bank.owners()).chain([Rc::clone(&self.0.scratch)]).collect()
    }

    fn validate(&self, placement: &Placement, len: usize) -> Result<()> {
        for bank in &self.0.banks { bank.validate(placement, len)?; }
        Ok(())
    }

    fn install(&self, placement: &Placement) -> Result<()> {
        self.validate(placement, placement.len)?;
        for (rank, bank) in self.0.banks.iter().enumerate() {
            self.0.copies.ordered(rank, |stream| bank.install(placement, stream))?;
        }
        Ok(())
    }
}

impl PrefixFamily for MiniMaxPrefix<'_> {
    type Placement = Placement;

    fn layout(&self) -> FamilyLayout { self.0.layout }
    fn record_format(&self) -> &'static str { "minimax_msa_fp8_kv_index_v1" }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.units }
    fn commit_point(&self, placement: &Placement) -> usize { placement.len }
    fn bind(&self, placement: &mut Placement) -> Result<(), BoxError> { Ok(self.install(placement)?) }
    fn discard(&self, placement: &mut Placement) -> Result<(), BoxError> {
        self.restore(None, placement, 0)
    }
    fn extend_pages(&self, placement: &mut Placement, pages: &[u32]) -> Result<(), BoxError> {
        let mut next = placement.clone();
        next.units.extend_from_slice(pages);
        self.install(&next)?;
        *placement = next;
        Ok(())
    }
    fn capture(&self, _: MarkSlot, placement: &Placement, len: usize) -> Result<(), BoxError> {
        self.validate(placement, len)?;
        ensure!(len == placement.len, "MiniMax capture is not at the committed frontier");
        Ok(())
    }
    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        ensure!(mark.is_none(), "MiniMax paged snapshots have no positional mark");
        self.validate(placement, len)?;
        let next = Placement { len, ..placement.clone() };
        self.install(&next)?;
        placement.len = len;
        Ok(())
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        ensure!(copy.rows > 0 && copy.rows <= PAGE_ROWS && copy.owner.is_none()
            && copy.from != copy.to, "MiniMax invalid uniform-pool tail copy");
        let mut copies = Vec::new();
        for (rank, bank) in self.0.banks.iter().enumerate() {
            for layer in bank.layers.clone() {
                for (plane, _) in PAGE_PLANES {
                    // K/V are head-major, and index keys/scales are separate
                    // subplanes. Copy the entire physical page, not rows*stride.
                    copies.push((rank, bank.page(layer, plane, copy.to)?,
                        bank.page(layer, plane, copy.from)?));
                }
            }
        }
        // SAFETY: distinct pool page IDs give disjoint destinations. The banks
        // are retained and copy streams fence every registered reader.
        unsafe { self.0.copies.copy_d2d(&copies)?; }
        Ok(())
    }
    fn drain(&self) -> Result<(), BoxError> { Ok(self.0.copies.drain()?) }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        if page as usize >= self.0.layout.pages { return Vec::new(); }
        self.0.banks.iter().flat_map(|bank| bank.layers.clone().flat_map(move |layer|
            PAGE_PLANES.into_iter().map(move |(plane, _)|
                range(bank.page(layer, plane, page).expect("validated MiniMax page"))))).collect()
    }
    fn mark_segments(&self, _: MarkSlot) -> Vec<DeviceRange> { Vec::new() }
    fn host_tail(&self) -> Vec<DeviceRange> {
        vec![range(view(self.0.scratch.buffer, 0, 1).expect("validated MiniMax scratch"))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{memory::device::{Device, Stream}, prefix::CudaCopyEngine};
    use cuteafd_core::memory_layout::Tag;
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, SnapshotKind};
    use cuteafd_ffi::{native_library_lifetime_fixture::Fixture, NativeLibrary};
    use cuteafd_loader::families::minimax_m3::{MiniMaxM3Config, execution::{MsaCache, MsaRope}};

    fn cfg() -> MiniMaxM3Config {
        let mut cfg = MiniMaxM3Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/minimax_m3/fixtures/config.json")).unwrap()).unwrap();
        cfg.layers = 2;
        cfg
    }
    fn family(library: &NativeLibrary, pages: u32) -> Result<MiniMaxPrefix<'_>> {
        let cfg = cfg();
        let mut banks = Vec::new();
        for rank in 0..2 {
            let stream = Rc::new(Stream::new(Device { library, id: rank })?);
            banks.push(Rc::new(PageBank::new(vec![stream], &cfg,
                MsaCache::new(1, pages, 4, 256)?, MsaRope::new(&cfg, 256)?,
                rank as usize..rank as usize + 1)?));
        }
        let scratch = Rc::new(Allocation::new(banks[0].stream.device, 256, Tag::workspace("minimax/unit"))?);
        MiniMaxPrefix::new(banks, scratch)
    }
    fn build(units: Vec<u32>, slot: usize) -> Placement { Placement { units, slot, len: 0 } }
    fn read_page(family: &MiniMaxPrefix<'_>, page: u32) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        for bank in &family.0.banks {
            for layer in bank.layers.clone() {
                for (plane, _) in PAGE_PLANES {
                    let buffer = bank.page(layer, plane, page)?;
                    let mut bytes = vec![0; buffer.bytes];
                    bank.stream.device.run(|| bank.stream.device.library.copy_d2h(&mut bytes, buffer))?;
                    out.push(bytes);
                }
            }
        }
        Ok(out)
    }
    fn write_page(family: &MiniMaxPrefix<'_>, page: u32, salt: usize) -> Result<()> {
        for bank in &family.0.banks {
            for layer in bank.layers.clone() {
                for (plane, stride) in PAGE_PLANES {
                    let bytes = (0..stride).map(|i| ((i * 31 + salt * 13 + layer * 7) % 251) as u8).collect::<Vec<_>>();
                    bank.stream.device.run(|| bank.stream.device.library.copy_h2d(
                        bank.page(layer, plane, page)?, &bytes))?;
                }
            }
        }
        Ok(())
    }
    fn cache(family: &MiniMaxPrefix<'_>) -> Result<PrefixCache<CudaCopyEngine<'static>>> {
        Ok(PrefixCache::new(family.layout(), PrefixConfig {
            entries: 8, mark_slots: 0, keep_logits: true, min_tokens: 1 }, None)?)
    }

    #[test]
    fn prompt_and_turn_restore_partial_pages_byte_identically_on_both_ranks() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 0, 32 << 20)?;
        fixture.configure_memory(&library, 1, 32 << 20)?;
        let family = family(&library, 20)?;
        let mut cache = cache(&family)?;
        let prompt: Vec<u32> = (0..65).collect();
        let turn: Vec<u32> = (0..129).collect();
        let mut original = cache.admit(&family, &prompt, 192, false, |pages| build(pages, 0))?.placement;
        for (i, &page) in original.units.iter().enumerate() { write_page(&family, page, i)?; }
        original.len = 65;
        let first = read_page(&family, original.units[0])?;
        let prompt_tail = read_page(&family, original.units[1])?;
        assert!(cache.capture(&family, SnapshotKind::Prompt, &prompt, &original,
            After::from_logits(&[0.0, 1.0], true))?);
        // A turn changes every byte of the live tail, including head 3 and the
        // separate index scales, after the prompt snapshot privately copied it.
        write_page(&family, original.units[1], 91)?;
        original.len = 129;
        let turn_middle = read_page(&family, original.units[1])?;
        let turn_tail = read_page(&family, original.units[2])?;
        assert!(cache.capture(&family, SnapshotKind::Turn, &turn, &original,
            After::from_logits(&[2.0, 1.0], true))?);
        write_page(&family, original.units[2], 93)?;
        cache.release(&family, &original.units)?;

        let hit = cache.admit(&family, &prompt, 192, false, |pages| build(pages, 1))?;
        assert_eq!(hit.resume, 65);
        assert_eq!(hit.source.as_ref().unwrap().kind, SnapshotKind::Prompt);
        assert!(hit.source.as_ref().unwrap().plan.exact());
        assert_eq!(read_page(&family, hit.placement.units[0])?, first);
        assert_eq!(read_page(&family, hit.placement.units[1])?, prompt_tail);
        let sibling = cache.admit(&family, &prompt, 192, false, |pages| build(pages, 2))?;
        assert_ne!(hit.placement.units[1], sibling.placement.units[1]);
        write_page(&family, hit.placement.units[1], 121)?;
        assert_eq!(read_page(&family, sibling.placement.units[1])?, prompt_tail);
        cache.release(&family, &hit.placement.units)?;
        cache.release(&family, &sibling.placement.units)?;

        let hit = cache.admit(&family, &turn, 192, false, |pages| build(pages, 1))?;
        assert_eq!(hit.resume, 129);
        assert_eq!(hit.source.as_ref().unwrap().kind, SnapshotKind::Turn);
        assert_eq!(read_page(&family, hit.placement.units[1])?, turn_middle);
        assert_eq!(read_page(&family, hit.placement.units[2])?, turn_tail);
        cache.release(&family, &hit.placement.units)?;
        let mut branch = turn.clone(); branch[128] = 9999;
        let partial = cache.admit(&family, &branch, 192, false, |pages| build(pages, 3))?;
        assert_eq!(partial.resume, 128);
        assert_eq!(read_page(&family, partial.placement.units[1])?, turn_middle);
        cache.release(&family, &partial.placement.units)?;
        cache.clear(&family)?;
        assert_eq!(cache.pool().used(), 0);
        drop(cache); drop(family);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        for id in [0, 1] {
            assert_eq!(Device { library: &library, id }.run(|| library.cuda_physical_memory_info())?.0, 32 << 20);
        }
        Ok(())
    }
}
