//! Exact KDA frontiers and placement-owned MLA pages for the shared prefix cache.
use super::{mla::{CacheBank, PREFIX_REUSE}, state::KdaBank};
use crate::shared::{memory::device::Allocation, prefix::{cuda_copy::range, PrefixCopies, Reader}};
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, TailCopy};
use cuteafd_ffi::{CuteafdDeviceBuffer, MLA_PAGE_ROWS};
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SnapshotMeta {
    pub draft: Option<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Frontier { units: Vec<u32>, len: usize }

struct Inner<'a> {
    kda: [Rc<KdaBank<'a>>; 2],
    mla: [Rc<CacheBank<'a>>; 2],
    cfg: KimiK3Config,
    context: usize,
    layout: FamilyLayout,
    /// Per rank: the prefix copy stream, fenced against that rank's lane streams.
    copies: PrefixCopies<'a>,
    live: RefCell<Vec<Option<Frontier>>>,
    draft: Option<Rc<super::draft::NativeDraft<'a>>>,
}

impl Drop for Inner<'_> {
    fn drop(&mut self) {
        // The copy streams drain in `PrefixCopies`' own drop; the readers before the banks go.
        for bank in &self.kda {
            for stream in bank.readers() { stream.drain_or_abort("Kimi prefix reader teardown"); }
        }
    }
}

#[derive(Clone)]
pub(crate) struct KimiPrefix<'a>(Rc<Inner<'a>>);

impl<'a> KimiPrefix<'a> {
    pub(crate) fn new(cfg: &KimiK3Config, context: usize,
        kda: [Rc<KdaBank<'a>>; 2], mla: [Rc<CacheBank<'a>>; 2]) -> Result<Self> {
        ensure!((1..=1_048_576).contains(&context) && kda[0].slots() == kda[1].slots()
            && kda[0].marks() == kda[1].marks() && mla[0].pages() == mla[1].pages() && mla[0].format() == mla[1].format(),
            "Kimi prefix bank/context extents differ");
        ensure!(kda[0].stream.device.id != kda[1].stream.device.id
            && std::ptr::eq(kda[0].stream.device.library, kda[1].stream.device.library),
            "Kimi prefix requires exact TP2 native owners");
        let mut resident = [0; 2];
        for rank in 0..2 {
            ensure!(kda[rank].stream.device.id == mla[rank].stream.device.id
                && std::ptr::eq(kda[rank].stream.device.library, mla[rank].stream.device.library),
                "Kimi prefix KDA/MLA device owners differ");
            for (lane, reader) in kda[rank].readers().iter().enumerate() { mla[rank].check_reader(lane, reader)?; }
            for (layer, kind) in cfg.attention.iter().enumerate() {
                if *kind == KimiAttention::Kda { kda[rank].slot(layer, 0)?; }
                else if mla[rank].layer(layer).is_ok() { resident[rank] += 1; }
            }
        }
        let geometry = cuteafd_loader::families::kimi_k3::kda::cache_geometry(cfg, resident, mla[0].format())?;
        let page_bytes = geometry.ranks.iter().map(|rank| rank.persistent_unit_bytes as usize).sum();
        let mark_bytes = geometry.ranks.iter().map(|rank| rank.retained_mark_bytes as usize).sum();
        ensure!(mark_bytes == kda[0].mark_bytes() + kda[1].mark_bytes(), "Kimi prefix KDA shard geometry differs");
        ensure!(page_bytes > 0, "Kimi prefix has no resident MLA pages");
        let copies = PrefixCopies::new(kda[0].stream.device.library, kda.iter().map(|bank| bank.readers().iter()
            .map(|reader| Reader { device: reader.device.id, stream: reader.raw }).collect()).collect())?;
        let layout = FamilyLayout { page_rows: MLA_PAGE_ROWS, pages: mla[0].pages(), page_bytes,
            mark_bytes, draft_bytes: 0,
            rule: PREFIX_REUSE, mark_store: MarkStore::Arena, page_owners: Default::default() };
        let live = RefCell::new(vec![None; kda[0].slots()]);
        Ok(Self(Rc::new(Inner { kda, mla, cfg: cfg.clone(), context, layout, copies, live, draft: None })))
    }

    pub(crate) fn with_draft(mut self, draft: Option<Rc<super::draft::NativeDraft<'a>>>) -> Result<Self> {
        let inner = Rc::get_mut(&mut self.0).context("Kimi prefix drafter must bind before publication")?;
        if let Some(draft) = &draft {
            ensure!(draft.cache_format() == inner.mla[0].format(), "Kimi target/draft cache records differ");
            let bytes = draft.mark_segments(0)?.iter().map(|buffer| buffer.bytes).sum::<usize>();
            inner.layout.mark_bytes = inner.layout.mark_bytes.checked_add(bytes).context("Kimi complete mark overflow")?;
        }
        inner.draft = draft;
        Ok(self)
    }
    pub(crate) fn snapshot_retained(&self) -> Vec<Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner + 'a>> {
        self.0.draft.iter().cloned().map(|draft| draft as Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner + 'a>).collect()
    }

    pub(crate) fn slots(&self) -> usize { self.0.kda[0].marks() }

    pub(crate) fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.0.mla.iter().flat_map(|bank| bank.owners())
            .chain(self.0.kda.iter().map(|bank| bank.mark_owner()))
            .chain(self.0.draft.iter().map(|draft| draft.mark_owner())).collect()
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

    /// Accept observes the original state frontier, not the scheduler's emitted
    /// token length. Validate all members before any accepted replay is queued.
    pub(crate) fn check_provisional(&self, placement: &Placement, start: usize, rows: usize) -> Result<()> {
        let original = Placement { len: start, ..placement.clone() };
        self.check_append(&original, rows)
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

    fn move_mark(&self, mark: MarkSlot, slot: usize, capture: bool) -> Result<()> {
        let plans = (0..2).map(|rank| Ok((self.state_buffers(rank, slot)?, self.mark_buffers(rank, mark)?)))
            .collect::<Result<Vec<_>>>()?;
        let copies: Vec<_> = plans.into_iter().enumerate().flat_map(|(rank, (state, marks))| state.into_iter()
            .zip(marks).map(move |(live, saved)| if capture { (rank, saved, live) } else { (rank, live, saved) }))
            .collect();
        // SAFETY: both exact views are retained by this prefix owner through drain.
        unsafe { self.0.copies.copy_d2d(&copies)?; }
        if let Some(draft) = &self.0.draft { draft.move_mark(mark.0 as usize, slot, capture)?; }
        Ok(())
    }
}

impl PrefixFamily<SnapshotMeta> for KimiPrefix<'_> {
    type Placement = Placement;
    fn draft_page_bytes(&self) -> usize { self.0.draft.as_ref().map_or(0, |draft| draft.common().mla_draft_page_bytes()) }
    fn admit_draft_pages(&self, placement: &mut Placement, capacity: usize) -> Result<(), BoxError> {
        if let Some(draft) = &self.0.draft {
            let _scope = draft.device().enter()?;
            draft.common().admit_mla_pages(placement.slot, capacity)?;
        }
        Ok(())
    }
    fn draft_pages_valid(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        self.0.draft.as_ref().is_some_and(|draft| draft.common().mla_pages_valid(attachment))
    }
    fn draft_pages_revocable(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        self.0.draft.as_ref().is_some_and(|draft| draft.common().mla_pages_revocable(attachment))
    }
    fn retain_draft_pages(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<(), BoxError> {
        let draft = self.0.draft.as_ref().context("Kimi draft attachment without executor")?;
        Ok(draft.common().retain_mla_pages(attachment)?)
    }
    fn capture_draft_pages(&self, placement: &Placement, len: usize)
        -> Result<Option<cuteafd_engine::prefix::DraftPageAttachment>, BoxError> {
        let Some(draft) = &self.0.draft else { return Ok(None) };
        if draft.frontier(placement.slot)? != (0, len) { return Ok(None); }
        let _scope = draft.device().enter()?;
        draft.common().capture_mla_pages(placement.slot, len, len).map(Some)
    }
    fn release_draft_pages(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<(), BoxError> {
        let draft = self.0.draft.as_ref().context("Kimi draft attachment without executor")?;
        let _scope = draft.device().enter()?;
        Ok(draft.common().release_mla_pages(attachment)?)
    }
    fn promote_draft_pages(&self, saved: &cuteafd_engine::prefix::DraftPageAttachment)
        -> Result<cuteafd_engine::prefix::DraftPageAttachment, BoxError> {
        self.0.draft.as_ref().context("Kimi draft attachment without executor")?.common().promote_mla_pages(saved)
    }
    fn draft_page_segments(&self, page: cuteafd_engine::prefix::DraftPageId) -> Vec<DeviceRange> {
        self.0.draft.as_ref().and_then(|draft| draft.common().mla_page_segments(page).ok()).unwrap_or_default()
    }
    fn restore_draft_pages(&self, attachment: Option<&cuteafd_engine::prefix::DraftPageAttachment>,
        placement: &mut Placement, frontier: usize) -> Result<(), BoxError> {
        let Some(draft) = &self.0.draft else { return Ok(()) };
        if let Some(attachment) = attachment {
            if attachment.origin != 0 || attachment.frontier != frontier { return Err("Kimi noncanonical draft snapshot".into()); }
            draft.drain()?;
            let _scope = draft.device().enter()?;
            draft.common().restore_mla_pages(placement.slot, attachment, frontier)
        } else { Ok(draft.reset(placement.slot, frontier)?) }
    }
    fn capture_meta(&self, placement: &Placement, len: usize) -> Result<SnapshotMeta, BoxError> {
        let draft = self.0.draft.as_ref().map(|draft| draft.frontier(placement.slot)).transpose()?;
        Ok(SnapshotMeta { draft: draft.filter(|frontier| *frontier == (0, len)) })
    }

    fn record_format(&self) -> &'static str {
        match self.0.mla[0].format() {
            cuteafd_core::MlaCacheFormat::Bf16 => "kimi_k3_bf16_mla_tp2_kda_f32_v1",
            cuteafd_core::MlaCacheFormat::Fp8 => "kimi_k3_ds_fp8_mla_tp2_kda_f32_v1",
        }
    }
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
            let library = self.0.kda[rank].stream.device.library;
            self.0.copies.ordered(rank, |stream| {
                for region in regions {
                    // SAFETY: this slot is private, and all readers join this initialization.
                    unsafe { library.cuda_zero_bytes_async(region, region.bytes, stream)?; }
                }
                Ok(())
            })?;
        }
        if let Some(draft) = &self.0.draft { draft.reset(placement.slot, 0)?; }
        self.0.live.borrow_mut()[placement.slot] = Some(Frontier { units: placement.units.clone(), len: 0 });
        placement.len = 0;
        placement.state_len = 0;
        Ok(())
    }
    fn discard(&self, placement: &mut Placement) -> Result<(), BoxError> {
        if placement.slot >= self.0.kda[0].slots() { return Err("Kimi prefix discard slot".into()); }
        if let Some(draft) = &self.0.draft { draft.discard(placement.slot)?; }
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
        let copies: Vec<_> = plans.into_iter().enumerate().flat_map(|(rank, (source, destination))|
            source.into_iter().zip(destination).map(move |(src, dst)| (rank, dst, src))).collect();
        // SAFETY: the shared page pool retains source and private destination through drain.
        Ok(unsafe { self.0.copies.copy_d2d(&copies) }?)
    }
    fn drain(&self) -> Result<(), BoxError> {
        if let Some(draft) = &self.0.draft { draft.drain()?; }
        Ok(self.0.copies.drain()?)
    }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        (0..2).flat_map(|rank| self.page_buffers(rank, page, MLA_PAGE_ROWS).unwrap_or_default()).map(range).collect()
    }
    fn mark_segments(&self, mark: MarkSlot) -> Vec<DeviceRange> {
        (0..2).flat_map(|rank| self.mark_buffers(rank, mark).unwrap_or_default())
            .chain(self.0.draft.iter().flat_map(|draft| draft.mark_segments(mark.0 as usize).unwrap_or_default()))
            .map(range).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::{Device, Stream};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::placement::AttentionPlacement;

    #[test]
    fn planner_equals_daemon_owned_and_head_split_host_layout() -> Result<()> {
        use cuteafd_loader::{plan::{checkpoint::Checkpoint, family}, serving_capacity::{CacheOptions, KvPlacement}};
        let checkpoint = Checkpoint { snapshot: Default::default(),
            config: cuteafd_loader::plan::testing::kimi_k3_config(), quantize_config: None,
            weight_map: cuteafd_loader::plan::testing::kimi_k3_tensors(false).into_iter()
                .map(|(name, ..)| (name, "fixture.safetensors".into())).collect(),
            tensors: vec![], missing_shards: vec![], shard_bytes: 0 };
        let model = family::detect(&checkpoint).context("Kimi family")?.open(&checkpoint)?;
        let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
        for (attention, cache_format) in [AttentionPlacement::Layers, AttentionPlacement::Heads].into_iter()
            .flat_map(|attention| [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8]
                .map(|format| (attention, format))) {
            let planned = model.cache_geometry(CacheOptions { coordinator_ranks: 2, mla_kv: cache_format,
                attention_placement: Some(attention), ..Default::default() })?.context("host layout")?;
            let mut input = super::super::admission::tests::inputs(&cfg);
            input.attention = Some(attention);
            input.cache_format = cache_format;
            let admitted = super::super::admission::admit(input)?;
            let executor = cuteafd_loader::placement::families::executor("kimi_k3")
                .context("Kimi serve executor registry entry")?;
            executor.check(&admitted.placement)?;
            assert_eq!(executor.attention, AttentionPlacement::Layers);
            assert_eq!(planned.logical_unit_rows, MLA_PAGE_ROWS as u64);
            assert_eq!(planned.placement, KvPlacement::PartitionedLayers,
                "host copies every physical owner, including both head-split MLA copies");
            for rank in 0..2 {
                let mla = super::super::mla::CacheLayout::new(&cfg, &admitted.placement, rank, cache_format)?;
                let kda = super::super::state::KdaLayout::new(&cfg, 2, 2, 2)?;
                let units = admitted.placement.pool_tokens / MLA_PAGE_ROWS as u64;
                assert_eq!(planned.ranks[rank].persistent_unit_bytes, mla.bytes() as u64 / units);
                assert_eq!(planned.ranks[rank].retained_mark_bytes, kda.mark_bytes() as u64);
                assert_eq!(planned.ranks[rank].active_state_per_sequence_bytes, kda.mark_bytes() as u64);
            }
            let pages = planned.ranks.iter().map(|r| r.persistent_unit_bytes).sum::<u64>();
            assert_eq!(pages, (24 * 64 * cache_format.record_bytes()) as u64 * if attention == AttentionPlacement::Heads { 2 } else { 1 });
            assert_eq!(planned.ranks.iter().map(|r| r.retained_mark_bytes).sum::<u64>(), 449_372_160);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn host_spill_restores_exact_prompt_turn_and_resident_page_frontiers() -> Result<()> {
        use crate::shared::prefix::CudaCopyEngine;
        use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, SnapshotKind};
        use cuteafd_hostcache::{config::Config, copy::Stream as CopyStream};

        for (heads, context, cache_format) in [(false, 262_144), (true, 524_288)].into_iter()
            .flat_map(|(heads, context)| [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8]
                .map(|format| (heads, context, format))) {
            let fixture = Fixture::build()?;
            let library = fixture.load()?;
            fixture.immediate_copies(&library)?;
            let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
            cfg.layers = 3;
            cfg.attention = vec![KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Mla];
            cfg.dense.truncate(3);
            let mut inputs = super::super::admission::tests::inputs(&cfg);
            inputs.attention = heads.then_some(AttentionPlacement::Heads);
            inputs.cache_format = cache_format;
            for layouts in &mut inputs.lane_layouts {
                for layout in layouts {
                    *layout = super::super::device::LaneLayout::new(
                        &KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?, inputs.rows, (inputs.rows).min(4096), context,
                        super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] })?;
                }
            }
            let mut placement = super::super::admission::admit(inputs)?.placement;
            placement.pool_tokens = 512;
            let bank = |rank| -> Result<_> {
                let device = Device { library: &library, id: rank as i32 };
                let readers = (0..2).map(|_| Stream::new(device).map(Rc::new)).collect::<Result<Vec<_>>>()?;
                let kda = super::super::state::KdaLayout::new(&cfg, 2, 1, 2)?.allocate(device, readers.clone())?;
                let mla = super::super::mla::CacheLayout::new(&cfg, &placement, rank, cache_format)?.allocate(device, readers)?;
                Ok((Rc::new(kda), Rc::new(mla)))
            };
            let (k0, m0) = bank(0)?;
            let (k1, m1) = bank(1)?;
            let family = KimiPrefix::new(&cfg, context, [k0, k1], [m0, m1])?;
            let boxed = |error: BoxError| anyhow::anyhow!("Kimi host fixture: {error}");
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
            let state = |slot| -> Result<Vec<Vec<u8>>> {
                (0..2).flat_map(|rank| family.state_buffers(rank, slot).unwrap()).map(read).collect()
            };
            let pages = |p: &Placement| -> Result<Vec<Vec<u8>>> {
                let mut result = Vec::new();
                for rank in 0..2 {
                    for (index, &page) in p.units.iter().take(p.len.div_ceil(64)).enumerate() {
                        let rows = (p.len - index * 64).min(64);
                        for buffer in family.page_buffers(rank, page, rows)? { result.push(read(buffer)?); }
                    }
                }
                Ok(result)
            };
            let make_cache = || -> Result<PrefixCache<CudaCopyEngine<'_>, SnapshotMeta>> {
                let layout = family.layout();
                let chunk = layout.mark_bytes.max(layout.page_bytes) as u64;
                let copy = CudaCopyEngine::registered_owned(&library, family.snapshot_owners())?;
                Ok(PrefixCache::new(layout, PrefixConfig { entries: 8, mark_slots: family.slots(),
                    keep_logits: true, min_tokens: 1 }, Some((Config { bytes: chunk * 6, chunk_bytes: chunk,
                        min_tokens: 1, ..Default::default() }, copy)))?)
            };
            let finish_store = |cache: &mut PrefixCache<CudaCopyEngine<'_>, SnapshotMeta>| -> Result<()> {
                cache.host_engine_mut().context("host tier missing")?.synchronize(CopyStream::Store)?;
                cache.tick();
                Ok(())
            };
            let build = |slot| move |units| Placement { units, slot, len: 0, state_len: 0 };
            let mut cache = make_cache()?;
            let tokens = (100..230).collect::<Vec<u32>>();
            let mut original = cache.admit(&family, &tokens[..65], 131, false, build(0))?.placement;
            let mut expected = Vec::new();
            for (end, kind) in [(65, SnapshotKind::Prompt), (130, SnapshotKind::Turn)] {
                let start = original.len;
                for rank in 0..2 {
                    for (i, buffer) in family.state_buffers(rank, 0)?.into_iter().enumerate() {
                        fill(buffer, end + rank * 47 + i)?;
                    }
                    // Do not rewrite shared full pages while the prompt snapshot
                    // is retained: only the newly appended rows change at turn end.
                    for (index, &page) in original.units.iter().enumerate() {
                        for buffer in family.page_buffers(rank, page, 64)? {
                            let first = start.saturating_sub(index * 64).min(64);
                            let last = end.saturating_sub(index * 64).min(64);
                            if first < last {
                                let offset = first * cache_format.record_bytes();
                                let bytes = (last - first) * cache_format.record_bytes();
                                let view = crate::shared::prefix::view(buffer, offset, bytes)?;
                                fill(view, end + rank * 71 + index * 17)?;
                            }
                        }
                    }
                }
                family.commit(&mut original, start, end - start)?;
                expected.push((end, state(0)?, pages(&original)?));
                assert!(cache.capture(&family, kind, &tokens[..end], &original,
                    After::from_logits(&[1.0, 2.0], true))?);
                finish_store(&mut cache)?;
            }
            cache.release(&family, &original.units)?;
            cache.clear(&family)?;
            // Every former device byte is poisoned before either host restore.
            for rank in 0..2 {
                for buffer in family.state_buffers(rank, 0)?.into_iter()
                    .chain(family.mark_buffers(rank, MarkSlot(0))?) { fill(buffer, 249)?; }
                for page in 0..family.layout().pages as u32 {
                    for buffer in family.page_buffers(rank, page, 64)? { fill(buffer, 251)?; }
                }
            }
            for (end, expected_state, expected_pages) in expected {
                let hit = cache.admit(&family, &tokens[..end], end + 1, false, build(1))?;
                assert_eq!(hit.resume, end);
                assert!(hit.source.as_ref().is_some_and(|source| source.host));
                let mut restored = hit.placement;
                family.drain().map_err(boxed)?;
                assert_eq!(state(1)?, expected_state, "both TP2 state/conv shards at {end}");
                assert_eq!(pages(&restored)?, expected_pages, "owned/split MLA prefix and partial tail at {end}");
                family.check_append(&restored, 1)?;
                for rank in 0..2 {
                    for buffer in family.state_buffers(rank, 1)? { fill(buffer, 243)?; }
                    for buffer in family.page_buffers(rank, restored.units[end / 64], 64)? {
                        fill(crate::shared::prefix::view(buffer, (end % 64) * cache_format.record_bytes(),
                            cache_format.record_bytes())?, 241)?;
                    }
                }
                family.commit(&mut restored, end, 1)?;
                assert_eq!(family.commit_point(&restored), end + 1);
                // Appending mutates the fork's tail and recurrent state, never
                // the snapshot's exact frontier or its private partial page.
                let again = cache.admit(&family, &tokens[..end], end + 1, false, build(0))?;
                assert_eq!(again.resume, end);
                family.drain().map_err(boxed)?;
                assert_eq!(state(0)?, expected_state);
                assert_eq!(pages(&again.placement)?, expected_pages);
                cache.release(&family, &again.placement.units)?;
                cache.release(&family, &restored.units)?;
                cache.clear(&family)?;
            }
            drop(cache);

            // Keep a full-page snapshot resident. The shared working-set policy
            // can spill only its mark; the page lease must remain identical.
            let mut cache = make_cache()?;
            let tokens = (500..564).collect::<Vec<u32>>();
            let mut original = cache.admit(&family, &tokens, 65, false, build(0))?.placement;
            for rank in 0..2 {
                for buffer in family.state_buffers(rank, 0)? { fill(buffer, rank + 41)?; }
                for buffer in family.page_buffers(rank, original.units[0], 64)? { fill(buffer, rank + 81)?; }
            }
            family.commit(&mut original, 0, 64)?;
            let expected_state = state(0)?;
            let expected_pages = pages(&original)?;
            assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens, &original,
                After::from_logits(&[3.0, 1.0], true))?);
            finish_store(&mut cache)?;
            for rank in 0..2 {
                for buffer in family.state_buffers(rank, 0)? { fill(buffer, 247)?; }
            }
            let hit = cache.admit(&family, &tokens, 65, false, build(1))?;
            assert_eq!(hit.resume, 64);
            assert_eq!(hit.placement.units[0], original.units[0], "resident full page must not be replaced");
            family.drain().map_err(boxed)?;
            assert_eq!(state(1)?, expected_state);
            assert_eq!(pages(&hit.placement)?, expected_pages);
            family.check_append(&hit.placement, 1)?;
            cache.release(&family, &hit.placement.units)?;
            cache.release(&family, &original.units)?;
            cache.clear(&family)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR; CPU-only fake DSO"]
    fn exact_adapter_restores_both_kda_shards_and_full_partial_mla_pages() -> Result<()> {
        for (heads, cache_format) in [false, true].into_iter()
            .flat_map(|heads| [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8]
                .map(|format| (heads, format))) {
            let fixture = Fixture::build()?;
            let library = fixture.load()?;
            fixture.immediate_copies(&library)?;
            let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
            cfg.layers = 3;
            cfg.attention = vec![KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Mla];
            cfg.dense.truncate(3);
            let mut inputs = super::super::admission::tests::inputs(&cfg);
            inputs.attention = heads.then_some(AttentionPlacement::Heads);
            inputs.cache_format = cache_format;
            let mut placement = super::super::admission::admit(inputs)?.placement;
            placement.pool_tokens = 256;
            let bank = |rank| -> Result<_> {
                let device = Device { library: &library, id: rank as i32 };
                let readers = (0..2).map(|_| Stream::new(device).map(Rc::new)).collect::<Result<Vec<_>>>()?;
                let kda = super::super::state::KdaLayout::new(&cfg, 2, 1, 2)?.allocate(device, readers.clone())?;
                let mla = super::super::mla::CacheLayout::new(&cfg, &placement, rank, cache_format)?.allocate(device, readers)?;
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
                        let copied = rows * cache_format.record_bytes();
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
            assert_eq!(family.layout().page_bytes, 2 * if heads { 2 } else { 1 } * 64 * cache_format.record_bytes());
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
