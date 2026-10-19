//! Exact KDA frontiers and placement-owned MLA pages for the shared prefix cache.
use super::{mla::{CacheBank, PREFIX_REUSE}, state::KdaBank};
use crate::shared::{memory::device::{Allocation, Event, Stream}, prefix::cuda_copy::range};
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, TailCopy};
use cuteafd_ffi::{CuteafdDeviceBuffer, MLA_PAGE_ROWS, MLA_RECORD_BYTES};
use cuteafd_hostcache::copy::DeviceRange;
use cuteafd_loader::families::kimi_k3::config::{KimiAttention, KimiK3Config};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
pub(crate) struct Placement {
    pub units: Vec<u32>,
    pub slot: usize,
    pub len: usize,
    pub state_len: usize,
}

impl crate::shared::qwen::scheduler::Placement for Placement {
    fn len(&self) -> usize { self.len }
    fn set_len(&mut self, len: usize) { self.len = len; }
    fn units(&self) -> &[u32] { &self.units }
    fn slot(&self) -> i32 { self.slot as i32 }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] { &[] }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Frontier { units: Vec<u32>, len: usize }

struct Joins<'a> { produced: Vec<Event<'a>>, landed: Event<'a> }

struct Inner<'a> {
    kda: [Rc<KdaBank<'a>>; 2],
    mla: [Rc<CacheBank<'a>>; 2],
    cfg: KimiK3Config,
    context: usize,
    layout: FamilyLayout,
    joins: [RefCell<Joins<'a>>; 2],
    live: RefCell<Vec<Option<Frontier>>>,
}

impl Drop for Inner<'_> {
    fn drop(&mut self) {
        for bank in &self.kda {
            for stream in bank.readers() { stream.drain_or_abort("Kimi prefix reader teardown"); }
            bank.stream.drain_or_abort("Kimi prefix copy teardown");
        }
    }
}

#[derive(Clone)]
pub(crate) struct KimiPrefix<'a>(Rc<Inner<'a>>);

impl<'a> KimiPrefix<'a> {
    pub(crate) fn new(cfg: &KimiK3Config, context: usize,
        kda: [Rc<KdaBank<'a>>; 2], mla: [Rc<CacheBank<'a>>; 2]) -> Result<Self> {
        ensure!((1..=1_048_576).contains(&context) && kda[0].slots() == kda[1].slots()
            && kda[0].marks() == kda[1].marks() && mla[0].pages() == mla[1].pages(),
            "Kimi prefix bank/context extents differ");
        ensure!(kda[0].stream.device.id != kda[1].stream.device.id
            && std::ptr::eq(kda[0].stream.device.library, kda[1].stream.device.library),
            "Kimi prefix requires exact TP2 native owners");
        let mut page_bytes = 0usize;
        for rank in 0..2 {
            ensure!(kda[rank].stream.device.id == mla[rank].stream.device.id
                && std::ptr::eq(kda[rank].stream.device.library, mla[rank].stream.device.library),
                "Kimi prefix KDA/MLA device owners differ");
            for (lane, reader) in kda[rank].readers().iter().enumerate() { mla[rank].check_reader(lane, reader)?; }
            for (layer, kind) in cfg.attention.iter().enumerate() {
                if *kind == KimiAttention::Kda { kda[rank].slot(layer, 0)?; }
                else if mla[rank].layer(layer).is_ok() { page_bytes += MLA_PAGE_ROWS * MLA_RECORD_BYTES; }
            }
        }
        ensure!(page_bytes > 0, "Kimi prefix has no resident MLA pages");
        let joins = |rank: usize| -> Result<_> {
            let device = kda[rank].stream.device;
            Ok(RefCell::new(Joins { produced: kda[rank].readers().iter()
                .map(|_| Event::new(device)).collect::<Result<_>>()?, landed: Event::new(device)? }))
        };
        let joins = [joins(0)?, joins(1)?];
        let layout = FamilyLayout { page_rows: MLA_PAGE_ROWS, pages: mla[0].pages(), page_bytes,
            mark_bytes: kda[0].mark_bytes() + kda[1].mark_bytes(), draft_bytes: 0,
            rule: PREFIX_REUSE, mark_store: MarkStore::Arena, page_owners: Default::default() };
        let live = RefCell::new(vec![None; kda[0].slots()]);
        Ok(Self(Rc::new(Inner { kda, mla, cfg: cfg.clone(), context, layout, joins, live })))
    }

    pub(crate) fn slots(&self) -> usize { self.0.kda[0].marks() }

    pub(crate) fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.0.mla.iter().flat_map(|bank| bank.owners())
            .chain(self.0.kda.iter().map(|bank| bank.mark_owner())).collect()
    }

    fn validate(&self, placement: &Placement, len: usize) -> Result<()> {
        ensure!(placement.slot < self.0.kda[0].slots() && len <= self.0.context
            && len.div_ceil(MLA_PAGE_ROWS) <= placement.units.len(), "Kimi prefix slot/context/pages");
        let mut unique = std::collections::BTreeSet::new();
        ensure!(placement.units.iter().all(|&page| (page as usize) < self.0.layout.pages && unique.insert(page)),
            "Kimi prefix pages outside pool or repeated");
        Ok(())
    }

    pub(crate) fn check_append(&self, placement: &Placement, rows: usize) -> Result<()> {
        let end = placement.len.checked_add(rows).context("Kimi prefix frontier overflow")?;
        self.validate(placement, end)?;
        let live = self.0.live.borrow();
        let frontier = live[placement.slot].as_ref().context("Kimi prefix slot is not bound")?;
        ensure!(rows > 0 && frontier.len == placement.len && placement.state_len == placement.len
            && frontier.units == placement.units, "Kimi submission differs from bound frontier");
        Ok(())
    }

    /// Publish only after a complete non-speculative pass or exact accepted-state commit.
    pub(crate) fn commit(&self, placement: &mut Placement, start: usize, rows: usize) -> Result<()> {
        let end = start.checked_add(rows).context("Kimi prefix frontier overflow")?;
        self.validate(placement, end)?;
        let mut live = self.0.live.borrow_mut();
        let frontier = live[placement.slot].as_mut().context("Kimi prefix slot is not bound")?;
        ensure!(rows > 0 && frontier.len == start && frontier.units == placement.units,
            "Kimi prefix commit differs from bound frontier");
        frontier.len = end;
        placement.len = end;
        placement.state_len = end;
        Ok(())
    }

    fn state_buffers(&self, rank: usize, slot: usize) -> Result<Vec<CuteafdDeviceBuffer>> {
        self.0.cfg.attention.iter().enumerate().filter(|(_, kind)| **kind == KimiAttention::Kda)
            .map(|(layer, _)| self.0.kda[rank].slot(layer, slot)).collect::<Result<Vec<_>>>()
            .map(|pairs| pairs.into_iter().flatten().collect())
    }

    fn mark_buffers(&self, rank: usize, mark: MarkSlot) -> Result<Vec<CuteafdDeviceBuffer>> {
        self.0.cfg.attention.iter().enumerate().filter(|(_, kind)| **kind == KimiAttention::Kda)
            .map(|(layer, _)| self.0.kda[rank].mark(layer, mark.0 as usize)).collect::<Result<Vec<_>>>()
            .map(|pairs| pairs.into_iter().flatten().collect())
    }

    fn page_buffers(&self, rank: usize, page: u32, rows: usize) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!((page as usize) < self.0.layout.pages && (1..=MLA_PAGE_ROWS).contains(&rows),
            "Kimi prefix page copy outside admission");
        self.0.cfg.attention.iter().enumerate().filter(|(layer, kind)|
            **kind == KimiAttention::Mla && self.0.mla[rank].layer(*layer).is_ok())
            .map(|(layer, _)| self.0.mla[rank].page(layer, page as usize, rows)).collect()
    }

    fn ordered(&self, rank: usize, copies: impl FnOnce(&Stream<'a>) -> Result<()>) -> Result<()> {
        let bank = &self.0.kda[rank];
        let copy = &bank.stream;
        let mut joins = self.0.joins[rank].borrow_mut();
        // Ordered copy fence, not a compute drain. Join every exact producer,
        // then make every possible consumer wait before reusing the bank.
        for (event, reader) in joins.produced.iter_mut().zip(bank.readers()) {
            event.record(reader)?;
            copy.device.run(|| {
                // SAFETY: the bank and event owners outlive all queued waits.
                unsafe { copy.device.library.cuda_stream_wait_event(copy.raw, event.raw) }
            })?;
        }
        let result = copy.device.run(|| copies(copy));
        joins.landed.record(copy)?;
        for reader in bank.readers() {
            reader.device.run(|| {
                // SAFETY: the retained reader joins this exact completed copy frontier.
                unsafe { reader.device.library.cuda_stream_wait_event(reader.raw, joins.landed.raw) }
            })?;
        }
        result
    }

    fn move_mark(&self, mark: MarkSlot, slot: usize, capture: bool) -> Result<()> {
        let plans = (0..2).map(|rank| Ok((self.state_buffers(rank, slot)?, self.mark_buffers(rank, mark)?)))
            .collect::<Result<Vec<_>>>()?;
        for (rank, (state, marks)) in plans.into_iter().enumerate() {
            self.ordered(rank, |stream| {
                for (live, saved) in state.into_iter().zip(marks) {
                    let (dst, src) = if capture { (saved, live) } else { (live, saved) };
                    // SAFETY: both exact views are retained by this prefix owner through drain.
                    unsafe { stream.device.library.copy_d2d_async(dst, src, src.bytes, stream.raw)?; }
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

impl PrefixFamily for KimiPrefix<'_> {
    type Placement = Placement;
    fn record_format(&self) -> &'static str { "kimi_k3_bf16_mla_tp2_kda_f32_v1" }
    fn layout(&self) -> FamilyLayout { self.0.layout }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.units }
    fn commit_point(&self, placement: &Placement) -> usize {
        self.0.live.borrow().get(placement.slot).and_then(Option::as_ref)
            .filter(|frontier| frontier.units == placement.units && frontier.len == placement.len
                && frontier.len == placement.state_len)
            .map_or(0, |frontier| frontier.len)
    }
    fn bind(&self, placement: &mut Placement) -> Result<(), BoxError> {
        self.validate(placement, 0)?;
        // The scheduler recycles slots only after releasing/draining their pages.
        for rank in 0..2 {
            let regions = self.state_buffers(rank, placement.slot)?;
            self.ordered(rank, |stream| {
                for region in regions {
                    // SAFETY: this slot is private, and all readers join this initialization.
                    unsafe { stream.device.library.cuda_zero_bytes_async(region, region.bytes, stream.raw)?; }
                }
                Ok(())
            })?;
        }
        self.0.live.borrow_mut()[placement.slot] = Some(Frontier { units: placement.units.clone(), len: 0 });
        placement.len = 0;
        placement.state_len = 0;
        Ok(())
    }
    fn discard(&self, placement: &mut Placement) -> Result<(), BoxError> {
        if placement.slot >= self.0.kda[0].slots() { return Err("Kimi prefix discard slot".into()); }
        self.0.live.borrow_mut()[placement.slot] = None;
        placement.len = 0;
        placement.state_len = 0;
        Ok(())
    }
    fn capture(&self, mark: MarkSlot, placement: &Placement, len: usize) -> Result<(), BoxError> {
        self.validate(placement, len)?;
        if !(len > 0 && len == placement.len && len == placement.state_len
            && self.commit_point(placement) == len) {
            return Err("Kimi prefix capture requires exact committed frontier".into());
        }
        Ok(self.move_mark(mark, placement.slot, true)?)
    }
    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        self.validate(placement, len)?;
        let mark = mark.context("Kimi prefix requires exact KDA marks; partial replay is unsupported")?;
        if !(len > 0 && self.0.live.borrow()[placement.slot].as_ref()
            .is_some_and(|frontier| frontier.units == placement.units && frontier.len == 0)) {
            return Err("Kimi prefix restore requires a fresh bound placement".into());
        }
        self.move_mark(mark, placement.slot, false)?;
        self.0.live.borrow_mut()[placement.slot] = Some(Frontier { units: placement.units.clone(), len });
        placement.len = len;
        placement.state_len = len;
        Ok(())
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        if copy.owner.is_some() || copy.from == copy.to {
            return Err("Kimi prefix COW requires distinct uniform pages".into());
        }
        let plans = (0..2).map(|rank| Ok((self.page_buffers(rank, copy.from, copy.rows)?,
            self.page_buffers(rank, copy.to, copy.rows)?))).collect::<Result<Vec<_>>>()?;
        for (rank, (source, destination)) in plans.into_iter().enumerate() {
            self.ordered(rank, |stream| {
                for (src, dst) in source.into_iter().zip(destination) {
                    // SAFETY: the shared page pool retains source and private destination through drain.
                    unsafe { stream.device.library.copy_d2d_async(dst, src, src.bytes, stream.raw)?; }
                }
                Ok(())
            })?;
        }
        Ok(())
    }
    fn drain(&self) -> Result<(), BoxError> {
        for rank in 0..2 {
            self.ordered(rank, |_| Ok(()))?;
            self.0.kda[rank].stream.drain()?;
        }
        Ok(())
    }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        (0..2).flat_map(|rank| self.page_buffers(rank, page, MLA_PAGE_ROWS).unwrap_or_default()).map(range).collect()
    }
    fn mark_segments(&self, mark: MarkSlot) -> Vec<DeviceRange> {
        (0..2).flat_map(|rank| self.mark_buffers(rank, mark).unwrap_or_default()).map(range).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::Device;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::placement::AttentionPlacement;

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn exact_adapter_restores_both_kda_shards_and_full_partial_mla_pages() -> Result<()> {
        for heads in [false, true] {
            let fixture = Fixture::build()?;
            let library = fixture.load()?;
            let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
            cfg.layers = 3;
            cfg.attention = vec![KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Mla];
            cfg.dense.truncate(3);
            let mut inputs = super::super::admission::tests::inputs(&cfg);
            inputs.attention = heads.then_some(AttentionPlacement::Heads);
            let mut placement = super::super::admission::admit(inputs)?.placement;
            placement.pool_tokens = 256;
            let bank = |rank| -> Result<_> {
                let device = Device { library: &library, id: rank as i32 };
                let readers = (0..2).map(|_| Stream::new(device).map(Rc::new)).collect::<Result<Vec<_>>>()?;
                let kda = super::super::state::KdaLayout::new(&cfg, 2, 1, 2)?.allocate(device, readers.clone())?;
                let mla = super::super::mla::CacheLayout::new(&cfg, &placement, rank)?.allocate(device, readers)?;
                Ok((Rc::new(kda), Rc::new(mla)))
            };
            let (k0, m0) = bank(0)?;
            let (k1, m1) = bank(1)?;
            let family = KimiPrefix::new(&cfg, 256, [k0, k1], [m0, m1])?;
            let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
                let mut bytes = vec![0; buffer.bytes];
                library.copy_d2h(&mut bytes, buffer)?;
                Ok(bytes)
            };
            let fill = |buffer: CuteafdDeviceBuffer, seed: usize| -> Result<Vec<u8>> {
                let bytes = (0..buffer.bytes).map(|i| (i.wrapping_mul(31) + seed) as u8).collect::<Vec<_>>();
                library.copy_h2d(buffer, &bytes)?;
                Ok(bytes)
            };
            let boxed = |error: BoxError| anyhow::anyhow!("Kimi prefix fixture: {error}");
            let mut original = Placement { units: vec![0, 1], slot: 0, len: 0, state_len: 0 };
            family.bind(&mut original).map_err(boxed)?;
            family.check_append(&original, 65)?;
            assert!(family.check_append(&original, 0).is_err());
            let mut wrong = original.clone();
            wrong.state_len = 1;
            assert!(family.check_append(&wrong, 1).is_err());
            wrong = original.clone();
            wrong.units.reverse();
            assert!(family.check_append(&wrong, 1).is_err());
            let mut expected = Vec::new();
            for rank in 0..2 {
                for (i, region) in family.state_buffers(rank, 0)?.into_iter().enumerate() {
                    expected.push(fill(region, rank * 47 + i + 11)?);
                }
                for page in 0..2 {
                    for (i, region) in family.page_buffers(rank, page, 64)?.into_iter().enumerate() {
                        fill(region, rank * 71 + i + page as usize * 17)?;
                    }
                }
            }
            family.commit(&mut original, 0, 65)?;
            assert_eq!(family.commit_point(&original), 65);
            let mut stale = original.clone();
            stale.len = 64;
            stale.state_len = 64;
            assert_eq!(family.commit_point(&stale), 0, "a later KDA frontier cannot certify an earlier prefix");
            assert!(family.capture(MarkSlot(0), &stale, 64).is_err());
            assert!(family.commit(&mut stale, 64, 1).is_err());
            assert!(family.check_append(&stale, 1).is_err());
            family.check_append(&original, 1)?;
            let before = fixture.events()?.len();
            family.capture(MarkSlot(0), &original, 65).map_err(boxed)?;
            let events = fixture.events()?;
            assert!(!events[before..].contains('S'), "capture must use event joins, not compute drains");
            assert_eq!(events[before..].matches('W').count(), 8);
            assert!(family.capture(MarkSlot(0), &original, 64).is_err());
            assert!(family.capture(MarkSlot(1), &original, 65).is_err());
            for rank in 0..2 {
                for region in family.state_buffers(rank, 0)? { fill(region, 222)?; }
                for page in 2..4 {
                    for region in family.page_buffers(rank, page, 64)? { fill(region, 199)?; }
                }
            }
            for (from, to, rows) in [(0, 2, 64), (1, 3, 1)] {
                family.copy_rows(TailCopy { from, to, rows, owner: None }).map_err(boxed)?;
                for rank in 0..2 {
                    for (src, dst) in family.page_buffers(rank, from, 64)?.into_iter()
                        .zip(family.page_buffers(rank, to, 64)?) {
                        let source = read(src)?;
                        let destination = read(dst)?;
                        let copied = rows * MLA_RECORD_BYTES;
                        assert_eq!(source[..copied], destination[..copied]);
                        if rows == 1 {
                            let sentinel = (0..dst.bytes).map(|i| (i.wrapping_mul(31) + 199) as u8).collect::<Vec<_>>();
                            assert_eq!(destination[copied..], sentinel[copied..]);
                        }
                    }
                }
            }
            let mut restored = Placement { units: vec![2, 3], slot: 1, len: 0, state_len: 0 };
            family.bind(&mut restored).map_err(boxed)?;
            assert!(family.restore(None, &mut restored, 65).is_err());
            family.restore(Some(MarkSlot(0)), &mut restored, 65).map_err(boxed)?;
            let before = fixture.events()?.len();
            family.drain().map_err(boxed)?;
            let events = fixture.events()?;
            assert_eq!(events[before..].matches('W').count(), 8,
                "page release must join every exact reader even without a pending capture");
            assert_eq!(events[before..].matches('S').count(), 2);
            let actual = (0..2).flat_map(|rank| family.state_buffers(rank, 1).unwrap()).map(read).collect::<Result<Vec<_>>>()?;
            assert_eq!(actual, expected, "both KDA state and convolution shards must restore byte-exactly");
            assert_eq!(family.commit_point(&restored), 65);
            assert_eq!(family.page_segments(0).iter().map(|range| range.bytes).sum::<usize>(), family.layout().page_bytes);
            assert_eq!(family.mark_segments(MarkSlot(0)).iter().map(|range| range.bytes).sum::<usize>(), family.layout().mark_bytes);
            assert_eq!(family.layout().page_bytes, 2 * if heads { 2 } else { 1 } * 64 * MLA_RECORD_BYTES);
            fixture.configure_pack(&library, 1, 0)?;
            let before = fixture.events()?.len();
            assert!(family.capture(MarkSlot(0), &original, 65).is_err());
            let events = fixture.events()?;
            assert!(!events[before..].contains('S'));
            assert_eq!(events[before..].matches('W').count(), 4,
                "a failed copy must still order its queued work before every consumer");
            fixture.configure_pack(&library, 0, 0)?;
            family.drain().map_err(boxed)?;
            // Shared scheduler releases pages before recycling its recurrent slot.
            family.bind(&mut original).map_err(boxed)?;
            assert_eq!(family.commit_point(&original), 0);
            family.discard(&mut restored).map_err(boxed)?;
            assert_eq!(family.commit_point(&restored), 0);
        }
        Ok(())
    }
}
