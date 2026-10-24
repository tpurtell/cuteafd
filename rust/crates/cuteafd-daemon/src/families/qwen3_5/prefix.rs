//! Exact dense Qwen frontiers: plain 64-row GQA pages plus FP32 GDN state.
use super::*;
use crate::shared::{prefix::view, qwen::MarkLayout};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, ReuseRule, TailCopy};
use cuteafd_hostcache::copy::DeviceRange;
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

pub(crate) struct DensePrefix<'e, 'a> {
    engine: &'e RefCell<DenseEngine<'a>>,
    layout: FamilyLayout,
    mark: MarkLayout,
    drafter: Option<Rc<crate::shared::speculation::block_dspark::DsparkDrafter<'a>>>,
    mtp: Option<Rc<super::mtp::MtpCache<'a>>>,
    /// The prefix copy stream, fenced against the target stream (the drafter's and the MTP
    /// chain's copies run on that stream too).
    copies: crate::shared::prefix::PrefixCopies<'a>,
}

impl<'e, 'a> DensePrefix<'e, 'a> {
    pub fn new(engine: &'e RefCell<DenseEngine<'a>>) -> Result<Self> {
        Self::with_draft(engine, None)
    }

    pub fn with_draft(engine: &'e RefCell<DenseEngine<'a>>,
        drafter: Option<Rc<crate::shared::speculation::block_dspark::DsparkDrafter<'a>>>) -> Result<Self> {
        let target = engine.borrow();
        let regions = Self::state(&target, 0);
        let mark = MarkLayout::new(&regions.iter().map(|region| (region.device_id, region.bytes)).collect::<Vec<_>>())?;
        ensure!(mark.bytes > 0 && !target.kv.iter().flatten().collect::<Vec<_>>().is_empty(),
            "dense prefix requires recurrent and full-attention layers");
        ensure!(target.prefix_arena.as_ref().map_or(0, |arena| arena.buffer.bytes)
            == target.prefix_marks * mark.bytes, "dense prefix marks differ from admission");
        let layout = FamilyLayout { page_rows: 64, pages: target.pool_tokens / 64,
            page_bytes: 64 * target.cfg.full_kv_bytes_per_token(target.kv_format),
            mark_bytes: mark.bytes, draft_bytes: 0, rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena, page_owners: Default::default() };
        let copies = crate::shared::prefix::PrefixCopies::new(target.stream.device.library,
            vec![vec![crate::shared::prefix::Reader { device: target.stream.device.id, stream: target.stream.raw }]])?;
        drop(target);
        Ok(Self { engine, layout, mark, drafter, mtp: None, copies })
    }

    pub(crate) fn with_mtp(engine: &'e RefCell<DenseEngine<'a>>, mtp: Rc<super::mtp::MtpCache<'a>>) -> Result<Self> {
        let mut family = Self::new(engine)?;
        family.mtp = Some(mtp);
        Ok(family)
    }

    pub fn slots(&self) -> usize { self.engine.borrow().prefix_marks }

    pub fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        let target = self.engine.borrow();
        target.kv.iter().flatten().cloned().chain(target.prefix_arena.iter().cloned()).collect()
    }

    fn state(target: &DenseEngine<'_>, slot: usize) -> Vec<CuteafdDeviceBuffer> {
        (0..target.homes.gdn_layers[0]).flat_map(|ordinal| target.bank.slot_regions(ordinal, slot)).collect()
    }

    fn mark_buffers(&self, slot: MarkSlot) -> Result<Vec<CuteafdDeviceBuffer>> {
        let target = self.engine.borrow();
        ensure!((slot.0 as usize) < target.prefix_marks, "dense prefix mark outside admitted arena");
        let arena = target.prefix_arena.as_ref().context("dense prefix arena disabled")?;
        self.mark.regions.iter().map(|&(_, offset, bytes)|
            view(arena.buffer, slot.0 as usize * self.mark.bytes + offset, bytes)).collect()
    }

    fn move_mark(&self, mark: MarkSlot, state: usize, capture: bool) -> Result<()> {
        let marks = self.mark_buffers(mark)?;
        let target = self.engine.borrow();
        ensure!(state < target.slots.len(), "dense prefix state outside admitted slots");
        let copies: Vec<_> = Self::state(&target, state).into_iter().zip(marks)
            .map(|(live, saved)| if capture { (0, saved, live) } else { (0, live, saved) }).collect();
        // SAFETY: admitted arena and live bank outlive the shared prefix cache's drain.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    fn page_buffers(&self, page: u32, rows: usize) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!((page as usize) < self.layout.pages && rows <= 64, "dense prefix page/rows outside admitted pool");
        let target = self.engine.borrow();
        let record = target.kv_format.record_bytes(target.cfg.kv_heads, target.cfg.head_dim);
        target.kv.iter().flatten().map(|pool| view(pool.buffer, page as usize * 64 * record, rows * record)).collect()
    }
}

/// Qualification through the real prefix adapter, including a partial-page
/// copy-on-write tail and a restored frontier on different physical KV pages.
pub(crate) fn check(engine: &RefCell<DenseEngine<'_>>, tokens: &[u32]) -> Result<()> {
    ensure!(!tokens.is_empty(), "dense prefix check requires tokens");
    let family = DensePrefix::new(engine)?;
    ensure!(family.slots() > 0 && family.layout.pages >= 6, "dense prefix check requires admitted marks and six pages");
    let continuation = tokens.iter().copied().cycle().take(3).collect::<Vec<_>>();
    let state = || -> Result<Vec<Vec<u8>>> {
        let target = engine.borrow();
        target.stream.drain()?;
        DensePrefix::state(&target, 0).into_iter().map(|region| {
            let mut bytes = vec![0; region.bytes];
            target.stream.device.library.copy_d2h(&mut bytes, region)?;
            Ok(bytes)
        }).collect()
    };
    let boxed = |error: BoxError| anyhow::anyhow!("dense prefix check: {error}");
    for len in [63usize, 64, 65, 127, 128, 129] {
        let prompt = tokens.iter().copied().cycle().take(len).collect::<Vec<_>>();
        let mut original = Placement { units: vec![0, 1, 2], slot: 0, len: 0, state_len: 0 };
        family.bind(&mut original).map_err(boxed)?;
        engine.borrow_mut().prefill_device(0, &prompt, 1)?;
        original.len = len;
        original.state_len = len;
        family.capture(MarkSlot(0), &original, len).map_err(boxed)?;
        family.drain().map_err(boxed)?;
        let saved = state()?;
        let library = engine.borrow().stream.device.library;
        let expected = engine.borrow_mut().prefill_device(0, &continuation, 3)?.to_host(library)?;
        let expected_state = state()?;
        family.discard(&mut original).map_err(boxed)?;
        for page in 0..len.div_ceil(64) {
            family.copy_rows(TailCopy { from: page as u32, to: page as u32 + 3,
                rows: (len - page * 64).min(64), owner: None }).map_err(boxed)?;
        }
        let mut restored = Placement { units: vec![3, 4, 5], slot: 0, len: 0, state_len: 0 };
        family.restore(Some(MarkSlot(0)), &mut restored, len).map_err(boxed)?;
        ensure!(state()? == saved, "dense cached prefix at {len} changed recurrence/convolution bytes");
        let actual = engine.borrow_mut().prefill_device(0, &continuation, 3)?.to_host(library)?;
        ensure!(actual.iter().map(|v| v.to_bits()).eq(expected.iter().map(|v| v.to_bits()))
            && state()? == expected_state, "dense cached prefix at {len} changed continuation logits or state");
        family.discard(&mut restored).map_err(boxed)?;
        family.drain().map_err(boxed)?;
        println!("dense prefix adapter: exact {len}-row state and continuation on copied full/partial KV pages byte-exact");
    }
    check_direct_frontiers(engine, tokens)?;
    Ok(())
}

fn check_direct_frontiers(engine: &RefCell<DenseEngine<'_>>, tokens: &[u32]) -> Result<()> {
    let family = DensePrefix::new(engine)?;
    ensure!(family.slots() >= 2, "dense direct prefix check requires prompt and turn marks");
    let library = engine.borrow().stream.device.library;
    let backend = super::super::serve::DenseBackend::new(engine, library);
    check_backend_frontiers(engine, tokens, &backend, family, false)
}

pub(crate) fn check_backend_frontiers(engine: &RefCell<DenseEngine<'_>>, tokens: &[u32],
    backend: &super::super::serve::DenseBackend<'_, '_>, family: DensePrefix<'_, '_>, speculative: bool) -> Result<()> {
    use crate::shared::qwen::scheduler::{Backend, Verified};
    let library = engine.borrow().stream.device.library;
    let pool = cuteafd_engine::prefix::RefPagePool::new(family.layout.pages, 64);
    let boxed = |error: BoxError| anyhow::anyhow!("dense direct prefix check: {error}");
    let snapshot = |placement: &Placement| -> Result<Vec<Vec<u8>>> {
        let target = engine.borrow();
        target.stream.drain()?;
        let mut regions = DensePrefix::state(&target, placement.slot);
        drop(target);
        for (index, &page) in placement.units.iter().enumerate().take(placement.len.div_ceil(64)) {
            regions.extend(family.page_buffers(page, (placement.len - index * 64).min(64))?);
        }
        let mut bytes = regions.into_iter().map(|region| {
            let mut bytes = vec![0; region.bytes];
            library.copy_d2h(&mut bytes, region)?;
            Ok(bytes)
        }).collect::<Result<Vec<_>>>()?;
        if let Some(saved) = family.capture_draft_pages(placement, placement.len).map_err(boxed)? {
            for (index, &page) in saved.pages.iter().enumerate() {
                let initialized = saved.frontier - usize::from(family.mtp.is_some());
                let rows = initialized.saturating_sub(index * 64).min(64);
                for (part, segment) in family.draft_page_segments(page).into_iter().enumerate() {
                    // Native MTP owns only the final page's pending-hidden sidecar.
                    if family.mtp.is_some() && part > 0 && index + 1 != saved.pages.len() { continue; }
                    let segment_bytes = if family.mtp.is_some() && part > 0 { segment.bytes }
                        else { segment.bytes / 64 * rows };
                    let mut data = vec![0; segment_bytes];
                    library.copy_d2h(&mut data, CuteafdDeviceBuffer { ptr: segment.addr as *mut _,
                        bytes: segment_bytes, device_id: library.cuda_get_device()?, ..Default::default() })?;
                    bytes.push(data);
                }
            }
            family.release_draft_pages(&saved).map_err(boxed)?;
        }
        Ok(bytes)
    };
    let decode = |placement: &mut Placement, inputs: &[u32]| -> Result<Vec<f32>> {
        if family.mtp.is_some() {
            use crate::shared::qwen::scheduler::{DraftSeq, DraftState, DraftTiming};
            let mut state = super::super::serve::DraftState::default();
            state.close(inputs[0]);
            let mut previous = Vec::new();
            for depth in 1..=3 {
                let proposed = backend.draft(&mut [DraftSeq { placement, seq: &mut state, depth }],
                    &mut DraftTiming::default())?;
                ensure!(proposed.len() == 1 && proposed[0].len() == depth
                    && proposed[0].starts_with(&previous), "native serving recursive draft differs by depth");
                previous = proposed[0].clone();
            }
        }
        let start = placement.len;
        let logits = backend.verify_device(&mut [(placement, inputs)], speculative, &pool)?
            .context("dense direct prefix check requires logits")?.to_host(library)?;
        let mut draft = super::super::serve::DraftState::default();
        if speculative { placement.len = start + inputs.len(); }
        backend.accept(&mut [Verified { placement, seq: &mut draft, start, history: (), rows: inputs,
            first_row: 0, kept: Some((inputs.len(), 42)), finished: false, neural: speculative }], speculative)?;
        Ok(logits)
    };
    let continuation = tokens.iter().copied().cycle().take(3).collect::<Vec<_>>();
    for len in [63usize, 65, 129] {
        let prompt = tokens.iter().copied().cycle().take(len).collect::<Vec<_>>();
        let mut placement = Placement { units: vec![0, 1, 2], slot: 0, len: 0, state_len: 0 };
        family.bind(&mut placement).map_err(boxed)?;
        family.admit_draft_pages(&mut placement, len + 2 * continuation.len()).map_err(boxed)?;
        backend.prefill_device(&mut placement, &prompt, 1, &pool)?;
        family.capture(MarkSlot(0), &placement, len).map_err(boxed)?;
        let prompt_draft = family.capture_draft_pages(&placement, len).map_err(boxed)?;
        family.drain().map_err(boxed)?;
        let prompt_bytes = snapshot(&placement)?;
        let first = decode(&mut placement, &continuation)?;
        let turn_len = placement.len;
        family.capture(MarkSlot(1), &placement, turn_len).map_err(boxed)?;
        let turn_draft = family.capture_draft_pages(&placement, turn_len).map_err(boxed)?;
        family.drain().map_err(boxed)?;
        let turn_bytes = snapshot(&placement)?;
        let second = decode(&mut placement, &continuation)?;
        for (mark, frontier, saved, expected, draft) in [(MarkSlot(0), len, &prompt_bytes, &first, prompt_draft.as_ref()),
            (MarkSlot(1), turn_len, &turn_bytes, &second, turn_draft.as_ref())] {
            family.discard(&mut placement).map_err(boxed)?;
            family.restore(Some(mark), &mut placement, frontier).map_err(boxed)?;
            family.restore_draft_pages(draft, &mut placement, frontier).map_err(boxed)?;
            family.admit_draft_pages(&mut placement, frontier + continuation.len()).map_err(boxed)?;
            ensure!(snapshot(&placement)? == *saved,
                "dense direct decode changed retained frontier {frontier} state/KV bytes");
            let actual = decode(&mut placement, &continuation)?;
            ensure!(actual.iter().map(|v| v.to_bits()).eq(expected.iter().map(|v| v.to_bits())),
                "dense direct decode changed retained frontier {frontier} continuation logits");
        }
        family.discard(&mut placement).map_err(boxed)?;
        for saved in [prompt_draft, turn_draft].into_iter().flatten() {
            family.release_draft_pages(&saved).map_err(boxed)?;
        }
        println!("dense prefix adapter: retained prompt {len} and turn {turn_len} target/draft state/KV/logits byte-exact (speculative={speculative})");
    }
    Ok(())
}

impl PrefixFamily for DensePrefix<'_, '_> {
    type Placement = Placement;

    fn draft_page_bytes(&self) -> usize {
        self.mtp.as_ref().map_or_else(|| self.drafter.as_ref().map_or(0, |draft| draft.page_bytes()), |mtp| mtp.page_bytes())
    }
    fn admit_draft_pages(&self, placement: &mut Placement, capacity: usize) -> Result<(), BoxError> {
        if let Some(mtp) = &self.mtp { mtp.admit_pages(placement.slot, capacity)?; }
        if let Some(draft) = &self.drafter { draft.admit_pages(placement.slot, capacity)?; }
        Ok(())
    }
    fn draft_pages_valid(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        self.mtp.as_ref().is_some_and(|mtp| mtp.pages_valid(attachment))
            || self.drafter.as_ref().is_some_and(|draft| draft.pages_valid(attachment))
    }
    fn draft_pages_revocable(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        self.mtp.as_ref().is_some_and(|mtp| mtp.pages_revocable(attachment))
            || self.drafter.as_ref().is_some_and(|draft| draft.pages_revocable(attachment))
    }
    fn retain_draft_pages(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<(), BoxError> {
        if let Some(mtp) = &self.mtp { mtp.retain_pages(attachment)?; return Ok(()); }
        self.drafter.as_ref().context("dense prefix has no drafter")?.retain_pages(attachment)?;
        Ok(())
    }
    fn capture_draft_pages(&self, placement: &Placement, len: usize)
        -> Result<Option<cuteafd_engine::prefix::DraftPageAttachment>, BoxError> {
        if let Some(mtp) = &self.mtp { return Ok(Some(mtp.capture_pages(placement.slot, len)?)); }
        self.drafter.as_ref().map(|draft| draft.capture_pages(placement.slot, len, len))
            .transpose().map_err(Into::into)
    }
    fn release_draft_pages(&self, attachment: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<(), BoxError> {
        if let Some(mtp) = &self.mtp { mtp.release_pages(attachment)?; return Ok(()); }
        self.drafter.as_ref().context("dense prefix has no drafter")?.release_pages(attachment)?;
        Ok(())
    }
    fn restore_draft_pages(&self, attachment: Option<&cuteafd_engine::prefix::DraftPageAttachment>,
        placement: &mut Placement, frontier: usize) -> Result<(), BoxError> {
        if let Some(mtp) = &self.mtp {
            if let Some(attachment) = attachment { mtp.restore_pages(placement.slot, attachment, frontier)?; }
            else {
                if frontier != 0 { return Err("dense MTP requires an exact prefix attachment".into()); }
                mtp.reset_context(placement.slot)?;
            }
        }
        if let Some(draft) = &self.drafter {
            if let Some(attachment) = attachment {
                if attachment.origin != 0 || attachment.frontier != frontier {
                    return Err("dense draft prefix differs from target".into());
                }
                draft.restore_pages(placement.slot, attachment, frontier)?;
            } else {
                if frontier != 0 { return Err("dense draft requires an exact full-context prefix attachment".into()); }
                draft.reset_context(placement.slot, 0)?;
            }
        }
        Ok(())
    }
    fn promote_draft_pages(&self, saved: &cuteafd_engine::prefix::DraftPageAttachment)
        -> Result<cuteafd_engine::prefix::DraftPageAttachment, BoxError> {
        if let Some(mtp) = &self.mtp { return Ok(mtp.promote_pages(saved)?); }
        Ok(self.drafter.as_ref().context("dense prefix has no drafter")?.promote_pages(saved)?)
    }
    fn draft_page_segments(&self, page: cuteafd_engine::prefix::DraftPageId) -> Vec<DeviceRange> {
        if let Some(mtp) = &self.mtp { return mtp.page_segments(page).expect("admitted dense MTP page"); }
        self.drafter.as_ref().map_or_else(Vec::new, |draft| draft.page_segments(page).expect("admitted dense draft page"))
    }

    fn record_format(&self) -> &'static str {
        match self.engine.borrow().kv_format {
            cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16 => "qwen_dense_bf16_gqa_gdn_f32_v1",
            cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 => "qwen_dense_e4m3_head_f32_gqa_gdn_f32_v1",
        }
    }
    fn layout(&self) -> FamilyLayout { self.layout }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.units }
    fn commit_point(&self, placement: &Placement) -> usize { placement.len.min(placement.state_len) }

    fn bind(&self, placement: &mut Placement) -> Result<(), BoxError> {
        let mut target = self.engine.borrow_mut();
        target.reset_slot(placement.slot)?;
        target.bind_slot(placement.slot, &placement.units)?;
        if let Some(mtp) = &self.mtp { mtp.reset_context(placement.slot)?; }
        if let Some(draft) = &self.drafter { draft.reset_context(placement.slot, 0)?; }
        placement.state_len = 0;
        Ok(())
    }

    fn discard(&self, placement: &mut Placement) -> Result<(), BoxError> {
        self.engine.borrow_mut().reset_slot(placement.slot)?;
        if let Some(mtp) = &self.mtp { mtp.discard_context(placement.slot)?; }
        if let Some(draft) = &self.drafter { draft.discard_context(placement.slot)?; }
        placement.len = 0;
        placement.state_len = 0;
        Ok(())
    }

    fn capture(&self, slot: MarkSlot, placement: &Placement, len: usize) -> Result<(), BoxError> {
        let target = self.engine.borrow();
        if !(placement.slot < target.slots.len() && len == placement.len && len == placement.state_len
            && len == target.slots[placement.slot].position && placement.units == target.slots[placement.slot].pages) {
            return Err("dense prefix capture requires exact committed frontier".into());
        }
        if !target.healthy || target.verification.pending.is_some() {
            return Err("dense prefix capture during uncommitted verify".into());
        }
        drop(target);
        self.move_mark(slot, placement.slot, true)?;
        Ok(())
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        let mark = mark.context("dense prefix restores require exact GDN marks")?;
        if len == 0 || len > self.engine.borrow().cfg.max_context || len.div_ceil(64) > placement.units.len() {
            return Err("dense prefix restore outside assigned pages/context".into());
        }
        // Validate the mark before changing any live state.
        self.mark_buffers(mark)?;
        {
            let mut target = self.engine.borrow_mut();
            target.bind_slot(placement.slot, &placement.units)?;
            target.healthy = false;
        }
        self.move_mark(mark, placement.slot, false)?;
        let mut target = self.engine.borrow_mut();
        // The restored state is ordered before the target stream's next pass by the copy fence.
        target.slots[placement.slot].position = len;
        target.position = target.slots[0].position;
        target.healthy = true;
        placement.len = len;
        placement.state_len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let source = self.page_buffers(copy.from, copy.rows)?;
        let destination = self.page_buffers(copy.to, copy.rows)?;
        let copies: Vec<_> = destination.into_iter().zip(source).map(|(dst, src)| (0, dst, src)).collect();
        // SAFETY: shared pool owns both page views until this family drains.
        Ok(unsafe { self.copies.copy_d2d(&copies) }?)
    }

    fn drain(&self) -> Result<(), BoxError> { Ok(self.copies.drain()?) }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.page_buffers(page, 64).unwrap_or_default().into_iter()
            .map(|buffer| DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes }).collect()
    }
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.mark_buffers(slot).unwrap_or_default().into_iter()
            .map(|buffer| DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes }).collect()
    }
}
