//! Muse Spark as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! Paged state: the full (NoPE) layers' K|V records, 64 rows per page; one
//! page index spans every full layer (on each GPU its own KV heads). Full
//! pages are shared by reference (every sequence appends past its own
//! length), so only a partial tail page is copied.
//!
//! Mark: the sliding layers keep an append ring per sequence (`window + 64`
//! rows, `cache::swa_ring`); a sequence continuing at `P` reads ring rows
//! `P - window .. P`, so a mark keeps each sliding layer's visible window in
//! position order (`MuseStateLayout::capture`), and a restore puts the rows
//! back into the same ring slots a prefill to `P` writes
//! (`MuseStateLayout::restore`; its host model proves restore == fresh
//! prefill on the visible window). Restores are exact (`ReuseRule::EXACT`).
//! The ring holds exactly the window plus one rollback step, so a mark is
//! taken at the committed frontier (`capture_reach` 0).
//!
//! Every copy runs on the prefix copy stream of its rank, fenced against that
//! rank's engine stream.
use super::engine::RankResources;
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, ReuseRule, TailCopy};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::copy::DeviceRange;
use cuteafd_loader::families::muse_spark::{cache::{MuseCacheLayout, MuseStateLayout, StateAt, PAGE_ROWS},
    MuseAttention, MuseSparkConfig};
use std::rc::Rc;

/// A sequence's placement: its slot (rings), pages and committed length.
#[derive(Debug, Clone, Default)]
pub(crate) struct Placement {
    pub slot: usize,
    pub pages: Vec<u32>,
    pub len: usize,
}

pub(crate) struct MusePrefix<'r, 'a> {
    library: &'a cuteafd_ffi::NativeLibrary,
    ranks: &'r [RankResources<'a>],
    states: Vec<MuseStateLayout>,
    /// Per rank: the full layers' record pools and the record bytes.
    full: Vec<(usize, CuteafdDeviceBuffer, usize)>,
    arenas: Vec<Rc<Allocation<'a>>>,
    slots: usize,
    pages: usize,
    copies: crate::shared::prefix::PrefixCopies<'a>,
}

fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    crate::shared::prefix::view(buffer, offset, bytes)
}

impl<'r, 'a> MusePrefix<'r, 'a> {
    /// `slots`: device mark slots (the shared prefix rule; admission charged
    /// `slots x mark bytes` per GPU); `sequences`: the ring slots; `pages`: the pool.
    pub(crate) fn new(cfg: &MuseSparkConfig, ranks: &'r [RankResources<'a>], streams: &[*mut std::ffi::c_void],
        sequences: u64, pages: usize, slots: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix");
        ensure!(!ranks.is_empty() && streams.len() == ranks.len(), "Muse prefix ranks and streams");
        let library = ranks[0].device.library;
        let cache = MuseCacheLayout::new(cfg, ranks.len())?;
        let state = MuseStateLayout::new(cfg, &cache, sequences)?;
        let record = usize::try_from(cache.record_bytes)?;
        let mut full = Vec::new();
        for (rank, resources) in ranks.iter().enumerate() {
            for (layer, kind) in cfg.attention.iter().enumerate() {
                if *kind == MuseAttention::Full {
                    if let super::device::LayerKv::Paged { cache } = resources.kv.layers[layer] { full.push((rank, cache, record)); }
                }
            }
        }
        let mark = usize::try_from(state.mark_bytes())?;
        let arenas = if slots > 0 {
            ranks.iter().map(|r| Allocation::new(Device { library, id: r.device.id }, (slots * mark).max(256)).map(Rc::new))
                .collect::<Result<Vec<_>>>()?
        } else { Vec::new() };
        let copies = crate::shared::prefix::PrefixCopies::new(library, ranks.iter().zip(streams).map(|(r, &stream)|
            Ok(vec![crate::shared::prefix::Reader { device: r.device.id, stream }])).collect::<Result<_>>()?)?;
        Ok(Self { library, ranks, states: vec![state; ranks.len()], full, arenas, slots, pages, copies })
    }

    /// Mark copies of `placement` at `len` into or out of `slot`, on every rank.
    fn move_mark(&self, slot: MarkSlot, placement: &Placement, len: usize, capture: bool) -> Result<()> {
        ensure!(!self.arenas.is_empty() && (slot.0 as usize) < self.slots, "Muse mark slot {} of {}", slot.0, self.slots);
        let mut copies = Vec::new();
        for (rank, state) in self.states.iter().enumerate() {
            let plan = if capture { state.capture(placement.slot as u64, len as u64, slot.0 as u64)? }
                else { state.restore(placement.slot as u64, len as u64, slot.0 as u64)? };
            let (rings, arena) = (self.ranks[rank].kv.rings.buffer, self.arenas[rank].buffer);
            let at = |place: StateAt, bytes: u64| -> Result<CuteafdDeviceBuffer> {
                match place {
                    StateAt::Ring(offset) => view(rings, usize::try_from(offset)?, usize::try_from(bytes)?),
                    StateAt::Mark(offset) => view(arena, usize::try_from(offset)?, usize::try_from(bytes)?),
                }
            };
            for copy in plan { copies.push((rank, at(copy.destination, copy.bytes)?, at(copy.source, copy.bytes)?)); }
        }
        // SAFETY: every view lies inside a live allocation of its rank; the cache drains
        // the copy streams before it hands any of that storage to someone else.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    pub(crate) fn slots(&self) -> usize { if self.arenas.is_empty() { 0 } else { self.slots } }

    pub(crate) fn page_bytes(&self) -> usize { self.full.iter().map(|(_, _, record)| PAGE_ROWS as usize * record).sum() }

    /// Stable allocations the host snapshot tier may copy: every rank's record
    /// pool and mark arena.
    pub(crate) fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> { self.arenas.clone() }
}

impl PrefixFamily for MusePrefix<'_, '_> {
    type Placement = Placement;

    fn record_format(&self) -> &'static str { "muse_bf16_kv_v1" }

    fn layout(&self) -> FamilyLayout {
        FamilyLayout { page_rows: PAGE_ROWS as usize, pages: self.pages, page_bytes: self.page_bytes(),
            mark_bytes: self.states.iter().map(|s| s.mark_bytes() as usize).sum(), draft_bytes: 0,
            rule: ReuseRule::EXACT, mark_store: MarkStore::Arena, page_owners: Default::default() }
    }

    fn capture_reach(&self) -> usize { 0 }

    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.pages }

    fn commit_point(&self, placement: &Placement) -> usize { placement.len }

    fn capture(&self, slot: MarkSlot, placement: &Placement, len: usize) -> Result<(), BoxError> {
        if len != placement.len {
            return Err(format!("Muse marks are taken at the committed frontier {}, not {len}", placement.len).into());
        }
        Ok(self.move_mark(slot, placement, len, true)?)
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        if len.div_ceil(PAGE_ROWS as usize) > placement.pages.len() {
            return Err(format!("restore of {len} rows into {} pages", placement.pages.len()).into());
        }
        match mark {
            Some(slot) => self.move_mark(slot, placement, len, false)?,
            None if len == 0 => {}
            None => return Err("Muse restores exact snapshots with their positional mark".into()),
        }
        placement.len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let page = PAGE_ROWS as usize;
        let copies = self.full.iter().map(|&(rank, buffer, record)| Ok((rank,
            view(buffer, copy.to as usize * page * record, copy.rows * record)?,
            view(buffer, copy.from as usize * page * record, copy.rows * record)?))).collect::<Result<Vec<_>>>()?;
        // SAFETY: as `move_mark`.
        Ok(unsafe { self.copies.copy_d2d(&copies) }?)
    }

    fn drain(&self) -> Result<(), BoxError> { Ok(self.copies.drain()?) }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.full.iter().map(|&(_, buffer, record)| {
            let bytes = PAGE_ROWS as usize * record;
            DeviceRange { addr: buffer.ptr as u64 + (page as usize * bytes) as u64, bytes }
        }).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.arenas.iter().zip(&self.states).map(|(arena, state)| {
            let bytes = state.mark_bytes() as usize;
            DeviceRange { addr: arena.buffer.ptr as u64 + (slot.0 as usize * bytes) as u64, bytes }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    /// The mark arena each rank allocates is the placement's "exact sliding
    /// marks" charge: `slots x (sliding layers x window x record / ranks)`.
    #[test]
    fn mark_arena_equals_the_charge() -> anyhow::Result<()> {
        use cuteafd_loader::families::muse_spark::{cache::{MuseCacheLayout, MuseStateLayout}, MuseSparkConfig};
        let cfg = MuseSparkConfig::from_hf(&cuteafd_loader::plan::testing::muse_spark_config())?;
        for ranks in [1usize, 2] {
            let state = MuseStateLayout::new(&cfg, &MuseCacheLayout::new(&cfg, ranks)?, 16)?;
            let charged = cuteafd_loader::placement::families::muse_spark::mark_bytes(&cfg, 1) / ranks as u64;
            assert_eq!(state.mark_bytes(), charged, "ranks {ranks}");
        }
        Ok(())
    }
}
