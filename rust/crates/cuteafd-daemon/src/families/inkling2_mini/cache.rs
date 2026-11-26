//! Persistent 48-slot request state and exact prefix cache for Inkling 2 Mini (TP1).
//!
//! 8 global paged KV planes (slots 5, 11, 17, 23, 29, 35, 41, 47), 40 local SWA-1024
//! append rings, and 48 contiguous 3-lag FP32 short-conv histories (width 6656).
use crate::shared::{
    memory::{
        device::{Allocation, Stream},
        receipts::Inventory,
    },
    prefix::{PrefixCopies, Reader},
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::inkling2_mini::execution::MiniCacheLayout;
use std::{collections::BTreeMap, rc::Rc};

pub(crate) fn view(buffer: Buffer, offset: usize, bytes: usize) -> Result<Buffer> {
    ensure!(
        bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "Inkling 2 Mini cache view exceeds backing"
    );
    Ok(Buffer {
        ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(),
        bytes,
        ..buffer
    })
}

pub(crate) struct CacheBank<'a> {
    copies: PrefixCopies<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    active: Allocation<'a>,
    conv: Allocation<'a>,
    marks: Option<Allocation<'a>>,
    pages: BTreeMap<usize, Allocation<'a>>,
    pub layout: MiniCacheLayout,
    mark_slots: usize,
}

impl<'a> CacheBank<'a> {
    pub fn require_reader(&self, stream: &Stream<'_>) -> Result<()> {
        ensure!(
            self.readers.iter().any(|reader| {
                reader.raw == stream.raw
                    && reader.device.id == stream.device.id
                    && std::ptr::eq(reader.device.library, stream.device.library)
            }),
            "Inkling 2 Mini arithmetic stream is not fenced by persistent cache"
        );
        Ok(())
    }

    pub fn new(readers: Vec<Rc<Stream<'a>>>, layout: MiniCacheLayout, mark_slots: usize) -> Result<Self> {
        let first = readers.first().context("Inkling 2 Mini cache requires readers")?;
        let device = first.device;
        ensure!(
            readers
                .iter()
                .all(|s| s.device.id == device.id && std::ptr::eq(s.device.library, device.library)),
            "Inkling 2 Mini cache reader devices differ"
        );
        let active = Allocation::new(device, layout.active_bytes()?, Tag::kv("inkling2_mini/sequence-state"))?;
        let conv = Allocation::new(device, layout.conv.bytes()?, Tag::kv("inkling2_mini/convolution-state"))?;
        let marks = (mark_slots > 0)
            .then(|| {
                Allocation::new(
                    device,
                    layout.retained_bytes(mark_slots)?,
                    Tag::new(cuteafd_core::memory_layout::MemoryKind::Prefix, "inkling2_mini/prefix-marks"),
                )
            })
            .transpose()?;
        let pages = layout
            .global_slots
            .iter()
            .map(|&slot| {
                Ok((
                    slot,
                    Allocation::new(device, layout.layer_page_bytes()?, Tag::kv("inkling2_mini/global-pages"))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let copies = PrefixCopies::new(
            device.library,
            vec![readers
                .iter()
                .map(|s| Reader {
                    device: s.device.id,
                    stream: s.raw,
                })
                .collect()],
        )?;
        let bank = Self {
            copies,
            readers,
            active,
            conv,
            marks,
            pages,
            layout,
            mark_slots,
        };
        bank.copies.ordered(0, |stream| {
            // SAFETY: fresh disjoint allocations owned by bank; fences join readers.
            unsafe {
                device
                    .library
                    .cuda_zero_bytes_async(bank.active.buffer, bank.active.buffer.bytes, stream)?;
                device
                    .library
                    .cuda_zero_bytes_async(bank.conv.buffer, bank.conv.buffer.bytes, stream)?;
            }
            Ok(())
        })?;
        Ok(bank)
    }

    pub fn field(&self, sequence: usize, cache_slot: usize, name: &str) -> Result<Buffer> {
        if name == "ring" {
            let span = self.layout.ring_field(sequence, cache_slot)?;
            return view(self.active.buffer, span.start, span.len());
        }
        let (span, _) = self.layout.conv.field(sequence, cache_slot, name)?;
        view(self.conv.buffer, span.start, span.len())
    }

    pub fn field_bank(&self, cache_slot: usize, name: &str) -> Result<Buffer> {
        let first = self.field(0, cache_slot, name)?;
        let stride = if name == "ring" {
            self.layout.ring_sequence_bytes
        } else {
            self.layout.conv.slot_stride * 4
        };
        let backing = if name == "ring" {
            self.active.buffer
        } else {
            self.conv.buffer
        };
        view(
            backing,
            first.ptr as usize - backing.ptr as usize,
            (self.layout.sequences - 1) * stride + first.bytes,
        )
    }

    pub fn global(&self, cache_slot: usize) -> Result<Buffer> {
        self.pages
            .get(&cache_slot)
            .map(|owner| owner.buffer)
            .with_context(|| format!("Inkling 2 Mini cache slot {cache_slot} has no global KV pages"))
    }

    pub fn page(&self, cache_slot: usize, page: usize) -> Result<Buffer> {
        ensure!(page < self.layout.pool_pages, "Inkling 2 Mini physical page outside pool");
        view(self.global(cache_slot)?, page * self.layout.page_bytes, self.layout.page_bytes)
    }

    fn rings_for_sequence(&self, sequence: usize) -> Result<Buffer> {
        ensure!(sequence < self.layout.sequences, "Inkling 2 Mini sequence outside cache");
        view(
            self.active.buffer,
            sequence * self.layout.ring_sequence_bytes,
            self.layout.ring_sequence_bytes,
        )
    }

    pub fn mark(&self, slot: usize) -> Result<Buffer> {
        ensure!(slot < self.mark_slots, "Inkling 2 Mini mark outside arena");
        view(
            self.marks.as_ref().context("Inkling 2 Mini mark arena absent")?.buffer,
            slot * self.layout.sequence_bytes,
            self.layout.sequence_bytes,
        )
    }

    pub fn conv_for_sequence(&self, sequence: usize) -> Result<Buffer> {
        ensure!(sequence < self.layout.sequences, "Inkling 2 Mini conv sequence outside cache");
        let bytes = self.layout.conv.slot_stride * 4;
        view(self.conv.buffer, sequence * bytes, bytes)
    }

    pub fn conv_backing(&self) -> Buffer {
        self.conv.buffer
    }

    pub fn reset(&self, sequence: usize) -> Result<()> {
        let rings = self.rings_for_sequence(sequence)?;
        let conv = self.conv_for_sequence(sequence)?;
        self.copies.ordered(0, |stream| {
            // SAFETY: copy fences join every arithmetic reader before overwrite.
            unsafe {
                self.readers[0]
                    .device
                    .library
                    .cuda_zero_bytes_async(rings, rings.bytes, stream)?;
                self.readers[0]
                    .device
                    .library
                    .cuda_zero_bytes_async(conv, conv.bytes, stream)
            }
        })
    }

    fn state_copy(&self, sequence: usize, slot: usize, capture: bool) -> Result<()> {
        let mark = self.mark(slot)?;
        let rings = self.rings_for_sequence(sequence)?;
        let conv = self.conv_for_sequence(sequence)?;
        let saved_rings = view(mark, 0, rings.bytes)?;
        let saved_conv = view(mark, rings.bytes, conv.bytes)?;
        let copies = if capture {
            vec![(0, saved_rings, rings), (0, saved_conv, conv)]
        } else {
            vec![(0, rings, saved_rings), (0, conv, saved_conv)]
        };
        // SAFETY: exact disjoint ring/convolution views share registered readers.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    pub fn capture(&self, sequence: usize, slot: usize) -> Result<()> {
        self.state_copy(sequence, slot, true)
    }

    pub fn restore(&self, slot: usize, sequence: usize) -> Result<()> {
        self.state_copy(sequence, slot, false)
    }

    pub fn copy_page_rows(&self, from: usize, to: usize, rows: usize) -> Result<()> {
        ensure!(
            from != to && (1..=64).contains(&rows),
            "Inkling 2 Mini page tail copy aliases or exceeds page"
        );
        let bytes = self.layout.page_bytes / 64 * rows;
        let copies = self
            .layout
            .global_slots
            .iter()
            .map(|&slot| {
                Ok((
                    0,
                    view(self.page(slot, to)?, 0, bytes)?,
                    view(self.page(slot, from)?, 0, bytes)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: physical page IDs are distinct; retained bank owns every buffer.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    pub fn mark_slots(&self) -> usize {
        self.mark_slots
    }

    pub fn drain(&self) -> Result<()> {
        self.copies.drain()
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        self.active.inventory(inventory, "inkling2_mini.cache.0.active")?;
        self.conv.inventory(inventory, "inkling2_mini.cache.0.conv")?;
        if let Some(marks) = &self.marks {
            marks.inventory(inventory, "inkling2_mini.cache.0.marks")?;
        }
        for (slot, owner) in &self.pages {
            owner.inventory(inventory, &format!("inkling2_mini.kv.{slot}.0.records"))?;
        }
        self.copies.allocation_receipts(inventory, "inkling2_mini.cache.0.copies")
    }
}

impl crate::shared::prefix::cuda_copy::SnapshotOwner for CacheBank<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<Buffer>> {
        Ok([self.active.buffer, self.conv.buffer]
            .into_iter()
            .chain(self.marks.iter().map(|owner| owner.buffer))
            .chain(self.pages.values().map(|owner| owner.buffer))
            .collect())
    }
}

impl Drop for CacheBank<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.copies.drain(), "Inkling 2 Mini persistent cache retirement");
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Placement {
    pub units: Vec<u32>,
    pub slot: usize,
    pub len: usize,
    pub state_len: usize,
    pub media: Option<cuteafd_engine::media::RequestMedia>,
}

impl crate::shared::qwen::scheduler::Placement for Placement {
    fn len(&self) -> usize {
        self.len
    }
    fn set_len(&mut self, len: usize) {
        self.len = len;
    }
    fn units(&self) -> &[u32] {
        &self.units
    }
    fn slot(&self) -> i32 {
        self.slot as i32
    }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] {
        self.media.as_ref().map_or(&[], |media| media.spans())
    }
}

#[derive(Clone)]
pub(crate) struct MiniPrefix<'a> {
    bank: Rc<CacheBank<'a>>,
    draft: Option<Rc<super::draft::Binding<'a>>>,
}

impl<'a> MiniPrefix<'a> {
    pub fn for_execution(execution: &super::engine::Execution<'a>) -> Self {
        Self {
            bank: execution.requests.cache.clone(),
            draft: execution.draft.clone(),
        }
    }

    pub fn snapshot_owners(&self) -> Vec<Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner + 'a>> {
        let mut owners: Vec<Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner + 'a>> =
            vec![self.bank.clone() as Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner>];
        if let Some(draft) = &self.draft {
            owners.push(draft.clone());
        }
        owners
    }

    fn validate(&self, placement: &Placement, len: usize) -> Result<()> {
        ensure!(
            placement.slot < self.bank.layout.sequences
                && len.div_ceil(64) <= placement.units.len()
                && placement
                    .units
                    .iter()
                    .all(|&page| (page as usize) < self.bank.layout.pool_pages),
            "Inkling 2 Mini prefix placement outside admitted state/pages"
        );
        ensure!(
            placement.units.iter().copied().collect::<std::collections::BTreeSet<_>>().len() == placement.units.len(),
            "Inkling 2 Mini prefix placement aliases physical pages"
        );
        Ok(())
    }

    /// Shared request-slot binding entry point for both serving (`PrefixFamily::bind`)
    /// and golden/oracle runners: validates `placement`, resets the 48-slot target
    /// cache bank, binds the recurrent drafter context slot when a drafter is engaged,
    /// and optionally admits draft pages up to `admit_capacity`.
    pub fn bind_slot(&self, placement: &mut Placement, admit_capacity: Option<usize>) -> Result<()> {
        self.validate(placement, placement.len)?;
        self.bank.reset(placement.slot)?;
        self.bank.drain()?;
        if let Some(draft) = &self.draft {
            draft.bind_slot(placement.slot, 0, admit_capacity)?;
        }
        placement.state_len = 0;
        Ok(())
    }
}

impl cuteafd_engine::prefix::PrefixFamily for MiniPrefix<'_> {
    type Placement = Placement;

    fn record_format(&self) -> &'static str {
        self.bank.layout.kv_format.record_format()
    }

    fn layout(&self) -> cuteafd_engine::prefix::FamilyLayout {
        use cuteafd_engine::prefix::{FamilyLayout, MarkStore, ReuseRule};
        FamilyLayout {
            page_rows: 64,
            pages: self.bank.layout.pool_pages,
            page_bytes: self.bank.layout.page_bytes * self.bank.layout.global_slots.len(),
            mark_bytes: self.bank.layout.sequence_bytes
                + self
                    .draft
                    .as_ref()
                    .map_or(0, |draft| draft.mark_bytes()),
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena,
            page_owners: Default::default(),
        }
    }

    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] {
        &placement.units
    }
    fn draft_page_bytes(&self) -> usize {
        self.draft.as_ref().and_then(|d| d.drafter.as_ref()).map_or(0, |d| d.page_bytes())
    }
    fn quarantine_draft_pages(&self, placement: &Placement) {
        if let Some(draft) = self.draft.as_ref().and_then(|d| d.drafter.as_ref()) {
            draft.quarantine_pages(placement.slot);
        }
    }
    fn draft_demand(&self, placement: &Placement) -> Option<cuteafd_engine::reserve::DraftDemand> {
        self.draft.as_ref()?.drafter.as_ref()?.reservation(placement.slot)
    }
    fn draft_reservable(&self) -> Option<usize> {
        Some(self.draft.as_ref()?.drafter.as_ref()?.reservable())
    }
    fn draft_lifetime_rows(&self) -> Option<usize> {
        Some(self.draft.as_ref()?.drafter.as_ref()?.lifetime_rows())
    }
    fn draft_admission(&self, origin: usize) -> Option<cuteafd_engine::reserve::DraftDemand> {
        Some(self.draft.as_ref()?.drafter.as_ref()?.admission(origin))
    }
    fn admit_draft_pages(&self, p: &mut Placement, capacity: usize) -> Result<(), cuteafd_engine::prefix::BoxError> {
        if let Some(draft) = self.draft.as_ref().and_then(|d| d.drafter.as_ref()) {
            draft.admit_pages(p.slot, capacity)?;
        }
        Ok(())
    }
    fn draft_pages_valid(&self, a: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        self.draft.as_ref().and_then(|d| d.drafter.as_ref()).is_some_and(|d| d.pages_valid(a))
    }
    fn draft_pages_revocable(&self, a: &cuteafd_engine::prefix::DraftPageAttachment) -> bool {
        self.draft.as_ref().and_then(|d| d.drafter.as_ref()).is_some_and(|d| d.pages_revocable(a))
    }
    fn retain_draft_pages(&self, a: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.draft
            .as_ref()
            .context("Inkling 2 Mini prefix has no drafter")?
            .drafter.as_ref().context("Mini recurrent pages absent")?
            .retain_pages(a)?;
        Ok(())
    }
    fn capture_draft_pages(
        &self,
        p: &Placement,
        len: usize,
    ) -> Result<Option<cuteafd_engine::prefix::DraftPageAttachment>, cuteafd_engine::prefix::BoxError> {
        Ok(self
            .draft
            .as_ref()
            .and_then(|d| d.drafter.as_ref())
            .map(|draft| draft.capture_pages(p.slot, len))
            .transpose()?)
    }
    fn release_draft_pages(&self, a: &cuteafd_engine::prefix::DraftPageAttachment) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.draft
            .as_ref()
            .context("Inkling 2 Mini prefix has no drafter")?
            .drafter.as_ref().context("Mini recurrent pages absent")?
            .release_pages(a)?;
        Ok(())
    }
    fn restore_draft_pages(
        &self,
        a: Option<&cuteafd_engine::prefix::DraftPageAttachment>,
        p: &mut Placement,
        frontier: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        if let Some(draft) = self.draft.as_ref().and_then(|d| d.drafter.as_ref()) {
            if let Some(a) = a {
                draft.restore_pages(p.slot, a, frontier)?;
            } else {
                if frontier != 0 {
                    return Err("Inkling 2 Mini draft restore needs exact full-context attachment".into());
                }
                draft.reset_context(p.slot, 0)?;
            }
        }
        Ok(())
    }
    fn promote_draft_pages(
        &self,
        a: &cuteafd_engine::prefix::DraftPageAttachment,
    ) -> Result<cuteafd_engine::prefix::DraftPageAttachment, cuteafd_engine::prefix::BoxError> {
        Ok(self
            .draft
            .as_ref()
            .context("Inkling 2 Mini prefix has no drafter")?
            .drafter.as_ref().context("Mini recurrent pages absent")?
            .promote_pages(a)?)
    }
    fn draft_page_segments(&self, page: cuteafd_engine::prefix::DraftPageId) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        self.draft.as_ref().and_then(|d| d.drafter.as_ref()).map_or_else(Vec::new, |d| {
            d.page_segments(page)
                .expect("admitted draft page")
                .into_iter()
                .map(crate::shared::prefix::cuda_copy::range)
                .collect()
        })
    }

    fn bind(&self, placement: &mut Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.bind_slot(placement, None)?;
        Ok(())
    }

    fn discard(&self, placement: &mut Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.bank.reset(placement.slot)?;
        if let Some(draft) = &self.draft {
            draft.bind_slot(placement.slot, 0, None)?;
        }
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

    fn commit_point(&self, placement: &Placement) -> usize {
        placement.len.min(placement.state_len)
    }

    fn capture(
        &self,
        slot: cuteafd_engine::prefix::MarkSlot,
        placement: &Placement,
        len: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        if len != placement.len || len != placement.state_len {
            return Err("Inkling 2 Mini snapshot frontier differs from committed causal state".into());
        }
        self.bank.capture(placement.slot, slot.0 as usize)?;
        if let Some(draft) = &self.draft {
            draft.capture(placement.slot, slot.0 as usize, len)?;
        }
        Ok(())
    }

    fn restore(
        &self,
        mark: Option<cuteafd_engine::prefix::MarkSlot>,
        placement: &mut Placement,
        len: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        let slot = mark.context("Inkling 2 Mini exact restore requires its 48-slot SWA/conv mark")?;
        self.bank.restore(slot.0 as usize, placement.slot)?;
        if let Some(draft) = &self.draft {
            draft.restore(placement.slot, slot.0 as usize, len)?;
        }
        placement.len = len;
        placement.state_len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: cuteafd_engine::prefix::TailCopy) -> Result<(), cuteafd_engine::prefix::BoxError> {
        if copy.owner.is_some() {
            return Err("Inkling 2 Mini TP1 pages have no parity owner".into());
        }
        self.bank
            .copy_page_rows(copy.from as usize, copy.to as usize, copy.rows)?;
        Ok(())
    }

    fn drain(&self) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.bank.drain()?;
        if let Some(draft) = &self.draft {
            draft.drain()?;
        }
        Ok(())
    }

    fn page_segments(&self, page: u32) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        self.bank
            .layout
            .global_slots
            .iter()
            .filter_map(|&slot| self.bank.page(slot, page as usize).ok())
            .map(crate::shared::prefix::cuda_copy::range)
            .collect()
    }

    fn mark_segments(&self, slot: cuteafd_engine::prefix::MarkSlot) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        let mut segments = self
            .bank
            .mark(slot.0 as usize)
            .ok()
            .into_iter()
            .map(crate::shared::prefix::cuda_copy::range)
            .collect::<Vec<_>>();
        if let Some(draft) = &self.draft {
            segments.extend(draft.mark_buffers(slot.0 as usize).into_iter().map(crate::shared::prefix::cuda_copy::range));
        }
        segments
    }
}
