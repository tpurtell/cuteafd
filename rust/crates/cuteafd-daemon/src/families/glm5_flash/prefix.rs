//! GLM 5.3 Flash as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! Paged state, one 256-row allocation unit per page index: on each of the 11 MLA layers the
//! unit's four 64-row MLA pages of FP8 latent records (528 B per row) and, with `--index-cache
//! keys`, DSA token keys (512 B per row), and the pool-key page of the same index (64 pools of
//! 4 tokens: 64 x 128 E4M3 then 64 FP32 scales) = 3.02 MB per unit (11.8 KB per token), or
//! 1.58 MB (6.17 KB per token) with the compact index cache. Units are refcounted: full units
//! are shared, nobody writes them again (every sequence appends past its own length), and a
//! partial tail unit is copied (its rows and its complete pools).
//!
//! Mark: the KDA layers' recurrent state is per sequence and overwritten by every step, so a
//! snapshot copies it whole: every KDA slot region the engine hands out (`slot_regions_on`:
//! each KDA layer's state `[64, 128, 128]`, FP32 or with `--kda-state bf16` BF16, and conv
//! window, the last three q/k/v inputs, and with the compact index cache every MLA layer's index
//! tail, the key | gate rows of its open pool, 11 x 1,552 B), 34 layers = 140.8 MiB with an FP32
//! state, 72.8 MiB with a BF16 one. A restore copies the mark back into the new
//! sequence's own KDA slot and maps its pool pages; nothing else of the state is positional.
//! `--prefix-marks` picks where marks live:
//! - `arena` (default): two in-flight staging marks with the host tier enabled; without it,
//!   the legacy lane/entry/budget rule retains positional marks on device;
//! - `pool`: whole units of the KV pool (49 per mark at FP32 state and token keys), taken at
//!   capture and evicted like any snapshot's rows. Each rank keeps its part of a mark in its own
//!   copy of the mark's units, laid out buffer by buffer over the units ([`mark_runs`]), and a
//!   capture or restore is one batch of copies between those runs and the slot regions
//!   ([`gather_scatter`], `cudaMemcpyBatchAsync` where the runtime has it). Whatever the slot
//!   regions hold (BF16 state, index tails) and whatever a unit's buffers are (with or without
//!   token keys), the mark follows them. Unit 0 is never handed out (to a mark or to rows): the
//!   decode sparse MLA reads its first record, slot 0, for every masked candidate and weights it
//!   by zero, so a mark's bytes there (an E4M3 NaN, an arbitrary FP32 scale) would turn every
//!   decode row with a masked candidate into NaN (0 x NaN = NaN). Reserved, it stays zeroed
//!   (`GLMF_POOL_MARK_RESERVED_UNITS`, allocated beside the admitted pool).
//! The capture point must be where the KDA state is: `kda_len` (a speculative verify leaves
//! the state behind the placement until its kept rows are committed), so `capture_reach` is 0.
//!
//! Restores are exact frontiers only (`ReuseRule::EXACT`): recurrent state exists at the
//! points it was captured and nowhere else. The DFlash2 drafter is not captured: a restored
//! sequence drafts cold with `valid_from` at the restore point.
//!
//! Every copy is enqueued on the engine stream, in order with the forward passes, and `drain`
//! synchronizes it. Under a head split both GPUs hold identical copies of the paged state (the
//! replicated MLA projection and indexer write them) and each its own KDA heads' state: page
//! copies run on each GPU's stream, a mark holds both GPUs' halves (an arena per GPU, or each
//! GPU's copy of the mark's units), and the host tier routes each range through its owning GPU.
use super::engine::{GlmfEngine, GlmfPlacement, IndexCache, KdaState, PagedLayer, KEY_BYTES, KPOOL, PAGE_ROWS,
    RECORD_BYTES, UNIT_PAGES, UNIT_ROWS};
use crate::shared::memory::device::{Allocation, Device};
use std::rc::Rc;
use crate::shared::prefix::MarkRule;
use crate::shared::prefix::view;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, ReuseRule, TailCopy};
use cuteafd_ffi::{CudaRuntime, CuteafdDeviceBuffer};
use cuteafd_hostcache::copy::DeviceRange;
use std::ffi::c_void;

/// Pool-key page: 64 pools x 128 E4M3, then 64 FP32 scales.
const POOL_KEY_BYTES: usize = 128;
/// Leading units pool marks keep out of every allocation (unit 0: the decode sparse MLA's
/// stand-in record for masked candidates).
const RESERVED_UNITS: usize = cuteafd_loader::serving_capacity::GLMF_POOL_MARK_RESERVED_UNITS as usize;
const POOL_SCALES: usize = PAGE_ROWS * POOL_KEY_BYTES;
const POOL_PAGE_BYTES: usize = PAGE_ROWS * (POOL_KEY_BYTES + 4);

/// Where prefix-cache snapshots keep their KDA state marks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum PrefixMarks {
    /// A device arena of 2C + 2 marks beside the KV pool.
    Arena,
    /// Units of the KV pool, taken at capture and evicted like any snapshot's rows.
    Pool,
}

impl PrefixMarks {
    /// Where marks live with `entries` retained snapshots per bank. Without entries no mark is
    /// ever taken, so there is no store: pool marks keep no unit back and need no room in the
    /// pool, as an arena of no entries holds no mark.
    pub fn with_entries(self, entries: usize) -> Self {
        if entries == 0 { Self::Arena } else { self }
    }
}

/// The prefix mark arena a command allocates on every GPU (a head split's GPUs each their part
/// of every mark), which either KV admission reserves before the pool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ArenaMarks {
    /// No arena: pool marks, or no prefix cache.
    #[default]
    None,
    /// A fixed count (`glmf-golden --resume-at` takes two).
    Slots(usize),
    /// serve-glmf's prefix cache: the arena rule with its knobs, over one mark.
    Rule(MarkRule),
}

impl ArenaMarks {
    /// Slots for marks of `mark_bytes` (every rank's part of one mark). `prefix_cache` sizes the
    /// arena with this from the engine's own slot regions.
    pub fn slots(self, mark_bytes: usize) -> usize {
        match self {
            Self::None => 0,
            Self::Slots(slots) => slots,
            Self::Rule(rule) => rule.slots(mark_bytes),
        }
    }

    /// Slots on the layout an engine serves, from the checkpoint alone, before the engine
    /// exists: either KV admission reserves this count. `index_cache` must be the cache the
    /// engine builds, not the one requested (a head split keeps the token keys, whose marks are
    /// smaller than compact ones, so a byte budget can hold one mark more): its marks are then
    /// the engine's slot regions, and this count is the arena `prefix_cache` allocates.
    pub fn slots_on(self, cfg: &cuteafd_loader::families::glm5_flash::GlmNextConfig, layers: usize,
        index_cache: IndexCache, kda_state: KdaState) -> Result<usize> {
        self.slots_placed(cfg, layers, index_cache, kda_state, false)
    }

    /// [`Self::slots_on`] for an engine whose head split runs context attention (`context`): each
    /// GPU's part of a mark is its KDA heads' state and every (replicated) index tail.
    pub fn slots_placed(self, cfg: &cuteafd_loader::families::glm5_flash::GlmNextConfig, layers: usize,
        index_cache: IndexCache, kda_state: KdaState, context: bool) -> Result<usize> {
        let Self::Rule(_) = self else { return Ok(self.slots(0)) };
        let geometry = if context {
            cuteafd_loader::serving_capacity::glm_flash_placed_cache_geometry_rows(cfg, layers, 2, index_cache.into(),
                kda_state.bytes() as u64, super::engine::DECODE_ROWS as u64,
                cuteafd_loader::placement::AttentionPlacement::Context, 0)?
        } else {
            cuteafd_loader::serving_capacity::glm_flash_rank_cache_geometry(cfg, layers, 1, index_cache.into(),
                kda_state.bytes() as u64)?
        };
        let mark: u64 = geometry.ranks.iter().map(|rank| rank.retained_mark_bytes).sum();
        ensure!(mark > 0, "GLM 5.3 Flash cache geometry without a rank");
        Ok(self.slots(usize::try_from(mark)?))
    }
}

pub(crate) struct GlmfPrefix<'e, 'a> {
    engine: &'e GlmfEngine<'a>,
    /// Per rank and MLA layer: records, token keys (`--index-cache keys`), pool keys.
    paged: Vec<(usize, PagedLayer)>,
    mark_bytes: usize,
    page_bytes: usize,
    /// Per rank: its part of every mark (its KDA heads' state) and the arena of those parts.
    arenas: Vec<(usize, Option<Rc<Allocation<'a>>>)>,
    slots: usize,
    /// Pool units of one mark (`PrefixMarks::Pool`; 0 with an arena).
    mark_units: usize,
    /// The loaded CUDA runtime's batched copy, which moves a pool mark in one call per rank
    /// (without it, one copy per segment).
    runtime: Option<CudaRuntime>,
}

impl<'e, 'a> GlmfPrefix<'e, 'a> {
    /// The family over `engine`'s buffers, its marks in a device arena of `slots(mark_bytes)`
    /// marks or in pool units.
    pub(crate) fn finish_request(&self) -> Result<()> { self.engine.finish_request() }

    pub fn new(engine: &'e GlmfEngine<'a>, marks: PrefixMarks, slots: impl FnOnce(usize) -> usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix");
        let paged: Vec<_> = (0..engine.ranks())
            .flat_map(|rank| engine.paged_buffers_on(rank).into_iter().map(move |buffers| (rank, buffers))).collect();
        let units = engine.local_units();
        for (_, layer) in &paged {
            ensure!(layer.records.bytes >= units * UNIT_ROWS * RECORD_BYTES
                && layer.keys.is_none_or(|keys| keys.bytes >= units * UNIT_ROWS * KEY_BYTES)
                && layer.pools.bytes >= units * POOL_PAGE_BYTES, "MLA cache buffers smaller than the units");
        }
        let parts: Vec<usize> = (0..engine.ranks())
            .map(|rank| engine.slot_regions_on(rank, 0).iter().map(|r| r.bytes).sum()).collect();
        let mark_bytes = parts.iter().sum();
        let page_bytes = paged.iter().filter(|(rank, _)| !(engine.context_kv() || engine.replicated_kv()) || *rank == 0)
            .map(|(_, layer)| UNIT_ROWS * (RECORD_BYTES + layer.keys.map_or(0, |_| KEY_BYTES)) + POOL_PAGE_BYTES).sum::<usize>()
            + (0..engine.ranks()).filter(|&rank| !(engine.context_kv() || engine.replicated_kv()) || rank == 0)
                .map(|rank| engine.pool_mark_unit(rank, 0).map(|b| b.map_or(0, |b| b.bytes)))
                .collect::<Result<Vec<_>>>()?.into_iter().sum::<usize>();
        let (slots, mark_units) = match marks {
            PrefixMarks::Arena => (slots(mark_bytes), 0),
            // Every rank's part fills whole units of its own copy of the pool.
            PrefixMarks::Pool => (0, parts.iter().enumerate().map(|(rank, &part)| -> Result<usize> {
                if part == 0 { return Ok(0); }
                let unit: usize = unit_ranges(&paged, rank, 0)?.iter()
                    .chain(engine.pool_mark_unit(rank, 0)?.iter()).map(|b| b.bytes).sum();
                ensure!(unit > 0, "GLM 5.3 Flash rank {rank} has no admitted pool-mark backing");
                Ok(part.div_ceil(unit))
            }).try_fold(0, |units, rank| rank.map(|rank| units.max(rank)))?),
        };
        ensure!(mark_units == 0 || !engine.context_kv(), "context attention keeps prefix marks in the arena");
        let reserved = if mark_units > 0 { RESERVED_UNITS } else { 0 };
        ensure!(mark_units + reserved < engine.pool_pages.max(1), "a pool mark of {mark_units} units and {reserved} \
            reserved do not fit a pool of {} units", engine.pool_pages);
        let arenas = parts.into_iter().enumerate().map(|(rank, part)| -> Result<_> {
            let arena = if slots > 0 { Some(engine.on(rank, || {
                let device = Device { library: engine.library, id: engine.library.cuda_get_device()? };
                Allocation::new(device, slots * part).map(Rc::new)
            })?) }
                else { None };
            Ok((part, arena))
        }).collect::<Result<_>>()?;
        let runtime = if mark_units > 0 { CudaRuntime::load() } else { None };
        if mark_units > 0 {
            tracing::info!(mark_bytes, mark_units,
                copy = if runtime.is_some() { "cuda-memcpy-batch" } else { "per-segment" },
                "GLM 5.3 Flash prefix marks in pool units");
        }
        Ok(Self { engine, paged, mark_bytes, page_bytes, arenas, slots, mark_units, runtime })
    }

    pub fn mark_bytes(&self) -> usize {
        self.mark_bytes
    }

    pub fn page_bytes(&self) -> usize {
        self.page_bytes
    }

    /// Whether units are partitioned by parity (context attention).
    fn parity(&self) -> bool {
        self.engine.context_kv()
    }

    /// The GPU holding unit `unit`'s rows and its local unit there.
    fn locate(&self, unit: u32) -> (usize, u32) {
        if self.parity() { ((unit & 1) as usize, unit >> 1) } else { (0, unit) }
    }

    /// Every allocation the host tier copies (both GPUs' pools under context), for the copy
    /// engine's routing by owning device.
    pub fn buffers(&self) -> Vec<CuteafdDeviceBuffer> {
        self.paged.iter().flat_map(|(_, layer)| std::iter::once(layer.records).chain(layer.keys).chain([layer.pools]))
            .chain(self.engine.pool_mark_buffers()).collect()
    }

    pub fn arena_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.arenas.iter().filter_map(|(_, arena)| arena.clone()).collect()
    }

    /// Arena marks (none when marks live in pool units).
    pub fn slots(&self) -> usize {
        if self.arenas.iter().all(|(_, arena)| arena.is_some()) { self.slots } else { 0 }
    }

    /// Pool units of one mark (0: arena marks).
    pub fn mark_units(&self) -> usize {
        self.mark_units
    }

    /// A copy on rank `rank`'s stream.
    fn copy(&self, rank: usize, dst: CuteafdDeviceBuffer, src: CuteafdDeviceBuffer) -> Result<()> {
        debug_assert_eq!(dst.bytes, src.bytes);
        // SAFETY: both views lie inside live engine allocations of that rank's GPU (checked by
        // `view`); the copy is ordered on its stream with every forward pass that reads or
        // writes them.
        self.engine.on(rank, || unsafe {
            self.engine.library.copy_d2d_async(dst, src, src.bytes, self.engine.stream_of(rank))
        })
    }

    /// One unit's views on rank `rank`, per MLA layer: records, token keys (`--index-cache keys`),
    /// pool keys.
    fn unit_layers_on(&self, rank: usize, unit: u32) -> Result<Vec<PagedLayer>> {
        unit_layers(&self.paged, rank, unit)
    }

    /// The device ranges of one unit on rank `rank`, per MLA layer: records, token keys
    /// (`--index-cache keys`), pool keys.
    fn unit_ranges_on(&self, rank: usize, unit: u32) -> Result<Vec<CuteafdDeviceBuffer>> {
        let mut ranges = unit_ranges(&self.paged, rank, unit)?;
        ranges.extend(self.engine.pool_mark_unit(rank, unit)?);
        Ok(ranges)
    }

    /// Where rank `rank`'s part of the pool mark in `pages` lies ([`mark_runs`] over the units).
    fn mark_runs_on(&self, rank: usize, pages: &[u32]) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!(self.mark_units > 0 && pages.len() == self.mark_units, "a pool mark of {} units, not {}",
            self.mark_units, pages.len());
        let part = self.arenas.get(rank).map(|(part, _)| *part).context("no such rank")?;
        if part == 0 { return Ok(Vec::new()); }
        let units = pages.iter().map(|&unit| self.unit_ranges_on(rank, unit)).collect::<Result<Vec<_>>>()?;
        let template = *units[0].first().context("units without buffers")?;
        let spans: Vec<Vec<Span>> = units.iter().map(|unit| unit.iter().map(|b| (b.ptr as usize, b.bytes)).collect())
            .collect();
        let runs = mark_runs(&spans, part);
        ensure!(runs.iter().map(|&(_, bytes)| bytes).sum::<usize>() == part, "{} units hold less than a mark",
            pages.len());
        Ok(runs.into_iter().map(|(addr, bytes)| CuteafdDeviceBuffer { ptr: addr as *mut c_void, bytes, ..template })
            .collect())
    }

    /// Copy every KDA slot region of slot `kda` into the pool mark in `pages`, or back, on each
    /// rank's stream: one batch of copies per rank.
    fn move_mark_pages(&self, pages: &[u32], kda: i32, capture: bool) -> Result<()> {
        ensure!(kda >= 0 && (kda as usize) < self.engine.slots, "KDA slot {kda} of {}", self.engine.slots);
        for rank in 0..self.engine.ranks() {
            let regions: Vec<Span> = self.engine.slot_regions_on(rank, kda as usize).iter()
                .map(|r| (r.ptr as usize, r.bytes)).collect();
            let runs = self.mark_runs_on(rank, pages)?;
            if regions.is_empty() { continue; }
            let template = *runs.first().context("nonempty owner state has no pool-mark runs")?;
            let runs: Vec<Span> = runs.iter().map(|b| (b.ptr as usize, b.bytes)).collect();
            let copies = if capture { gather_scatter(&regions, &runs) } else { gather_scatter(&runs, &regions) };
            ensure!(copies.iter().map(|c| c.bytes).sum::<usize>() == regions.iter().map(|&(_, b)| b).sum::<usize>(),
                "the pool mark does not cover the KDA state");
            self.copy_batch(rank, template, &copies)?;
        }
        Ok(())
    }

    /// `copies` on rank `rank`'s stream (`template`: one of that rank's buffers).
    fn copy_batch(&self, rank: usize, template: CuteafdDeviceBuffer, copies: &[SpanCopy]) -> Result<()> {
        let Some(runtime) = &self.runtime else {
            for copy in copies {
                let at = |addr: usize| CuteafdDeviceBuffer { ptr: addr as *mut c_void, bytes: copy.bytes, ..template };
                self.copy(rank, at(copy.dst), at(copy.src))?;
            }
            return Ok(());
        };
        let dsts: Vec<*mut c_void> = copies.iter().map(|c| c.dst as *mut c_void).collect();
        let srcs: Vec<*const c_void> = copies.iter().map(|c| c.src as *const c_void).collect();
        let sizes: Vec<usize> = copies.iter().map(|c| c.bytes).collect();
        // SAFETY: every range lies inside a live engine allocation of that rank's GPU (the slot
        // regions, and the mark's units checked by `view`); the destinations are disjoint (a
        // mark's own units, or one KDA slot's regions); the batch is ordered on the rank's
        // stream with every forward pass that reads or writes them, and the cache drains that
        // stream before it hands the units or the slot to anyone else.
        self.engine.on(rank, || unsafe {
            runtime.memcpy_batch_async(&dsts, &srcs, &sizes, self.engine.stream_of(rank))
        })
    }

    /// The pool mark in `pages`, rank by rank (checks).
    fn mark_pages_host(&self, pages: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for rank in 0..self.engine.ranks() {
            for run in self.mark_runs_on(rank, pages)? {
                out.extend(download(self.engine, rank, run)?);
            }
        }
        Ok(out)
    }

    /// Diagnostic (`--resume-poison-unit0`): fills record slot 0 (unit 0's first MLA record) of
    /// every MLA layer on every rank with `byte`, after the streams drained. With pool marks unit 0
    /// is reserved and nothing else reads or writes it, so whatever reaches the logits through it
    /// came from a kernel's stand-in for masked candidates.
    fn fill_stand_in(&self, byte: u8) -> Result<()> {
        ensure!(self.mark_units > 0, "the stand-in poison needs pool marks (unit 0 reserved)");
        self.engine.synchronize()?;
        let bytes = vec![byte; RECORD_BYTES];
        for &(rank, layer) in &self.paged {
            let slot0 = view(layer.records, 0, RECORD_BYTES)?;
            self.engine.on(rank, || self.engine.library.copy_h2d(slot0, &bytes))?;
        }
        Ok(())
    }

    /// Mark `slot`'s part on each rank: (rank, its arena range).
    fn mark_parts(&self, slot: MarkSlot) -> Result<Vec<(usize, CuteafdDeviceBuffer)>> {
        ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
        self.arenas.iter().enumerate().map(|(rank, (part, arena))| {
            let arena = arena.as_ref().map(|a| a.buffer).ok_or_else(|| anyhow::anyhow!("no mark arena"))?;
            Ok((rank, view(arena, slot.0 as usize * part, *part)?))
        }).collect()
    }

    /// Copy every KDA layer's state of KDA slot `kda` to or from mark `slot` (each rank its heads).
    fn move_mark(&self, slot: MarkSlot, kda: i32, capture: bool) -> Result<()> {
        ensure!(kda >= 0 && (kda as usize) < self.engine.slots, "KDA slot {kda} of {}", self.engine.slots);
        for (rank, part) in self.mark_parts(slot)? {
            let mut offset = 0;
            for region in self.engine.slot_regions_on(rank, kda as usize) {
                let mark = view(part, offset, region.bytes)?;
                if capture {
                    self.copy(rank, mark, region)?;
                } else {
                    self.copy(rank, region, mark)?;
                }
                offset += region.bytes;
            }
        }
        Ok(())
    }

    /// Mark `slot`'s bytes, rank by rank (checks).
    fn mark_host(&self, slot: MarkSlot) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for (rank, part) in self.mark_parts(slot)? {
            out.extend(download(self.engine, rank, part)?);
        }
        Ok(out)
    }
}

impl PrefixFamily for GlmfPrefix<'_, '_> {
    type Placement = GlmfPlacement;

    fn layout(&self) -> FamilyLayout {
        FamilyLayout {
            page_rows: UNIT_ROWS,
            pages: self.engine.pool_pages,
            page_bytes: self.page_bytes(),
            mark_bytes: self.mark_bytes,
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
            mark_store: if self.mark_units > 0 { MarkStore::Pool { pages: self.mark_units, reserved: RESERVED_UNITS } }
                else { MarkStore::Arena },
            page_owners: if self.parity() { cuteafd_engine::prefix::PageOwners::Parity }
                else { cuteafd_engine::prefix::PageOwners::Uniform },
        }
    }

    fn pages<'p>(&self, placement: &'p GlmfPlacement) -> &'p [u32] {
        &placement.units
    }

    fn commit_point(&self, placement: &GlmfPlacement) -> usize {
        placement.kda_len.min(placement.len)
    }

    fn capture(&self, slot: MarkSlot, placement: &GlmfPlacement, len: usize) -> Result<(), BoxError> {
        capture_point(placement, len)?;
        Ok(self.move_mark(slot, placement.slot, true)?)
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut GlmfPlacement, len: usize) -> Result<(), BoxError> {
        let Some(slot) = mark else {
            return Err("GLM 5.3 Flash restores exact snapshots with their KDA state".into());
        };
        restore_point(placement, len)?;
        self.engine.map_pools(placement)?;
        self.move_mark(slot, placement.slot, false)?;
        placement.len = len;
        placement.kda_len = len;
        Ok(())
    }

    fn capture_pages(&self, pages: &[u32], placement: &GlmfPlacement, len: usize) -> Result<(), BoxError> {
        capture_point(placement, len)?;
        Ok(self.move_mark_pages(pages, placement.slot, true)?)
    }

    fn restore_pages(&self, pages: &[u32], placement: &mut GlmfPlacement, len: usize) -> Result<(), BoxError> {
        restore_point(placement, len)?;
        self.engine.map_pools(placement)?;
        self.move_mark_pages(pages, placement.slot, false)?;
        placement.len = len;
        placement.kda_len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let rows = UNIT_PAGES * PAGE_ROWS;
        let pools = copy.rows / KPOOL;
        // Under heads every GPU holds the unit; under context only its owner, at its local unit.
        let ((from_rank, from), (to_rank, to)) = (self.locate(copy.from), self.locate(copy.to));
        if from_rank != to_rank || copy.owner.map(usize::from).is_some_and(|o| o != from_rank)
            || copy.owner.is_some() != self.parity() {
            return Err(format!("tail copy {copy:?} crosses GPUs").into());
        }
        let copy = TailCopy { from, to, ..copy };
        for &(rank, layer) in self.paged.iter().filter(|(rank, _)| !self.parity() || *rank == from_rank) {
            let pool_keys = layer.pools;
            for (buffer, row) in std::iter::once((layer.records, RECORD_BYTES)).chain(layer.keys.map(|k| (k, KEY_BYTES))) {
                self.copy(rank, view(buffer, copy.to as usize * rows * row, copy.rows * row)?,
                    view(buffer, copy.from as usize * rows * row, copy.rows * row)?)?;
            }
            // The complete pools: their E4M3 keys, then their scales.
            if pools > 0 {
                let (from, to) = (copy.from as usize * POOL_PAGE_BYTES, copy.to as usize * POOL_PAGE_BYTES);
                self.copy(rank, view(pool_keys, to, pools * POOL_KEY_BYTES)?,
                    view(pool_keys, from, pools * POOL_KEY_BYTES)?)?;
                self.copy(rank, view(pool_keys, to + POOL_SCALES, pools * 4)?,
                    view(pool_keys, from + POOL_SCALES, pools * 4)?)?;
            }
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> {
        Ok(self.engine.synchronize()?)
    }

    fn page_device(&self, page: u32) -> Option<u8> {
        self.parity().then_some((page & 1) as u8)
    }

    /// Heads store one identical copy; context stores its owner; layers store distinct owners.
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        let (owner, unit) = self.locate(page);
        (0..self.engine.ranks()).filter(|&rank| if self.parity() { rank == owner }
                else { !self.engine.replicated_kv() || rank == 0 })
            .flat_map(|rank| self.unit_ranges_on(rank, unit).unwrap_or_default())
            .map(|b| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }).collect()
    }

    fn page_replicas(&self, page: u32) -> Vec<(DeviceRange, DeviceRange)> {
        if !self.engine.replicated_kv() { return Vec::new(); }
        let left = self.unit_ranges_on(0, page).expect("admitted page");
        let right = self.unit_ranges_on(1, page).expect("admitted page replica");
        replica_ranges(&left, &right)
    }

    fn copy_replicas(&self, copies: &[(DeviceRange, DeviceRange)]) -> Result<(), BoxError> {
        self.engine.on(1, || {
            for &(src, dst) in copies {
                let template = self.paged.iter().find(|(rank, _)| *rank == 1).context("missing replica owner")?.1.records;
                let source = CuteafdDeviceBuffer { ptr: src.addr as *mut c_void, bytes: src.bytes,
                    device_id: self.engine.device, ..template };
                let target = CuteafdDeviceBuffer { ptr: dst.addr as *mut c_void, bytes: dst.bytes, ..template };
                // SAFETY: registered page views stay live until the destination stream drains.
                unsafe { self.engine.library.copy_d2d_async(target, source, src.bytes, self.engine.stream_of(1))?; }
            }
            Ok(())
        }).map_err(Into::into)
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.mark_parts(slot).unwrap_or_default().into_iter()
            .map(|(_, b)| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }).collect()
    }

    fn mark_page_segments(&self, pages: &[u32]) -> Result<Vec<DeviceRange>, BoxError> {
        let mut out = Vec::new();
        for rank in 0..self.engine.ranks() {
            out.extend(self.mark_runs_on(rank, pages)?.into_iter()
                .map(|b| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }));
        }
        Ok(out)
    }
}

/// A capture is taken where the KDA state is: the placement's length, every row committed.
fn capture_point(placement: &GlmfPlacement, len: usize) -> Result<(), BoxError> {
    if len != placement.len || len != placement.kda_len {
        return Err(format!("capture at {len}: the placement holds {} rows, its KDA state {}", placement.len,
            placement.kda_len).into());
    }
    Ok(())
}

fn restore_point(placement: &GlmfPlacement, len: usize) -> Result<(), BoxError> {
    if len == 0 || len.div_ceil(UNIT_ROWS) > placement.units.len() {
        return Err(format!("restore of {len} rows into {} units", placement.units.len()).into());
    }
    Ok(())
}

/// `unit`'s views on rank `rank`, per MLA layer: records, token keys (`--index-cache keys`),
/// pool keys.
fn unit_layers(paged: &[(usize, PagedLayer)], rank: usize, unit: u32) -> Result<Vec<PagedLayer>> {
    let unit = unit as usize;
    let rows = UNIT_PAGES * PAGE_ROWS;
    paged.iter().filter(|(r, _)| *r == rank).map(|(_, layer)| Ok(PagedLayer {
        records: view(layer.records, unit * rows * RECORD_BYTES, rows * RECORD_BYTES)?,
        keys: layer.keys.map(|keys| view(keys, unit * rows * KEY_BYTES, rows * KEY_BYTES)).transpose()?,
        pools: view(layer.pools, unit * POOL_PAGE_BYTES, POOL_PAGE_BYTES)?,
    })).collect()
}

/// The device ranges of `unit` on rank `rank`, per MLA layer: records, token keys
/// (`--index-cache keys`), pool keys. A pool mark lies in these segments ([`mark_runs`]).
fn unit_ranges(paged: &[(usize, PagedLayer)], rank: usize, unit: u32) -> Result<Vec<CuteafdDeviceBuffer>> {
    Ok(unit_layers(paged, rank, unit)?.into_iter()
        .flat_map(|layer| std::iter::once(layer.records).chain(layer.keys).chain([layer.pools])).collect())
}

fn replica_ranges(left: &[CuteafdDeviceBuffer], right: &[CuteafdDeviceBuffer]) -> Vec<(DeviceRange, DeviceRange)> {
    assert_eq!(left.len(), right.len(), "replica segment count");
    left.iter().zip(right).map(|(src, dst)| {
        assert_eq!(src.bytes, dst.bytes, "replica segment extent");
        (DeviceRange { addr: src.ptr as u64, bytes: src.bytes },
            DeviceRange { addr: dst.ptr as u64, bytes: dst.bytes })
    }).collect()
}

/// A device byte range: address, bytes.
type Span = (usize, usize);

/// One device copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SpanCopy {
    dst: usize,
    src: usize,
    bytes: usize,
}

/// Where a mark of `part` bytes lies in its units (`units[u]`: unit u's segments, every unit
/// alike): segment k of every unit in page order before segment k + 1, so one buffer's
/// segments of consecutive units are adjacent and merge, cut off after `part` bytes.
fn mark_runs(units: &[Vec<Span>], part: usize) -> Vec<Span> {
    let mut runs: Vec<Span> = Vec::new();
    let mut left = part;
    for k in 0..units.first().map_or(0, Vec::len) {
        for unit in units {
            if left == 0 {
                return runs;
            }
            let (addr, bytes) = unit[k];
            let take = bytes.min(left);
            match runs.last_mut() {
                Some((at, run)) if *at + *run == addr => *run += take,
                _ => runs.push((addr, take)),
            }
            left -= take;
        }
    }
    runs
}

/// The copies that lay the bytes of `from` (its spans concatenated) onto `to` (concatenated
/// likewise), as far as both reach: one per overlap of a source and a destination span, merged
/// where both sides continue.
fn gather_scatter(from: &[Span], to: &[Span]) -> Vec<SpanCopy> {
    let mut copies: Vec<SpanCopy> = Vec::new();
    let (mut i, mut j, mut a, mut b) = (0, 0, 0, 0);
    while i < from.len() && j < to.len() {
        let ((src, src_bytes), (dst, dst_bytes)) = (from[i], to[j]);
        let bytes = (src_bytes - a).min(dst_bytes - b);
        if bytes > 0 {
            let (src, dst) = (src + a, dst + b);
            match copies.last_mut() {
                Some(last) if last.src + last.bytes == src && last.dst + last.bytes == dst => last.bytes += bytes,
                _ => copies.push(SpanCopy { dst, src, bytes }),
            }
        }
        a += bytes;
        b += bytes;
        if a == src_bytes {
            (i, a) = (i + 1, 0);
        }
        if b == dst_bytes {
            (j, b) = (j + 1, 0);
        }
    }
    copies
}

/// One continued prefill: every layer's output digest, every row's logits digest and argmax,
/// the last row, and how many logits were not finite (NaN or infinite: never in a sound run).
pub(crate) struct SuffixRun {
    pub layers: Vec<u64>,
    pub logits: u64,
    pub argmax: Vec<usize>,
    pub last: Vec<f32>,
    pub nonfinite: usize,
}

/// Logits that are NaN or infinite.
fn nonfinite(values: &[f32]) -> usize {
    values.iter().filter(|v| !v.is_finite()).count()
}

/// Prefill `tokens` into `placement` in chunks of `chunk` rows, digesting each layer's streams
/// and (with `logits`) every row's logits.
pub(crate) fn prefill_digest(engine: &GlmfEngine<'_>, placement: &mut GlmfPlacement, tokens: &[u32], chunk: usize,
    logits: bool) -> Result<SuffixRun> {
    use std::hash::{Hash, Hasher};
    let vocab = engine.cfg.vocab_size;
    let mut hashers: Vec<std::collections::hash_map::DefaultHasher> = Vec::new();
    let mut logit_hash = std::collections::hash_map::DefaultHasher::new();
    let (mut argmax, mut last, mut bad) = (Vec::new(), Vec::new(), 0);
    for part in tokens.chunks(chunk) {
        let mut on_layer = |layer: usize, rows: &[u8]| -> Result<()> {
            if hashers.len() <= layer {
                hashers.resize_with(layer + 1, Default::default);
            }
            rows.hash(&mut hashers[layer]);
            Ok(())
        };
        let out = engine.prefill_forced(placement, part, Some(&mut on_layer), None, logits)?
            .ok_or_else(|| anyhow::anyhow!("the resume check needs every layer"))?;
        if logits {
            for values in out.chunks_exact(vocab) {
                values.iter().for_each(|v| v.to_bits().hash(&mut logit_hash));
                argmax.push(values.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i));
            }
            bad += nonfinite(&out);
            last = out[out.len() - vocab..].to_vec();
        }
    }
    Ok(SuffixRun { layers: hashers.iter().map(Hasher::finish).collect(), logits: logit_hash.finish(), argmax, last,
        nonfinite: bad })
}

/// `range` of rank `rank`'s GPU, after every stream drained (retiring every write to it).
fn download(engine: &GlmfEngine<'_>, rank: usize, range: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
    engine.synchronize()?;
    let mut bytes = vec![0u8; range.bytes];
    engine.on(rank, || engine.library.copy_d2h(&mut bytes, range))?;
    Ok(bytes)
}

/// Every paged byte of `placement`'s first `len` rows (records and, with `--index-cache keys`,
/// token keys of each row, pool keys and scales of each complete pool), MLA layer by layer, in
/// position order.
pub(super) fn paged_rows(family: &GlmfPrefix<'_, '_>, placement: &GlmfPlacement, len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let pools = len / KPOOL;
    for (u, &unit) in placement.units.iter().enumerate().take(len.div_ceil(UNIT_ROWS)) {
        let rows = (len - u * UNIT_ROWS).min(UNIT_ROWS);
        let unit_pools = (pools.saturating_sub(u * PAGE_ROWS)).min(PAGE_ROWS);
        let (owner, local) = family.locate(unit);
        for rank in (0..family.engine.ranks()).filter(|&rank| !family.parity() || rank == owner) {
            let unit = if family.parity() { local } else { unit };
            for layer in family.unit_layers_on(rank, unit)? {
                out.extend_from_slice(&download(family.engine, rank, layer.records)?[..rows * RECORD_BYTES]);
                if let Some(keys) = layer.keys {
                    out.extend_from_slice(&download(family.engine, rank, keys)?[..rows * KEY_BYTES]);
                }
                let page = download(family.engine, rank, layer.pools)?;
                out.extend_from_slice(&page[..unit_pools * POOL_KEY_BYTES]);
                out.extend_from_slice(&page[POOL_SCALES..POOL_SCALES + unit_pools * 4]);
            }
        }
    }
    Ok(out)
}

/// Exercise production workspace reuse without selection, downloads, or a
/// barrier between chunks. A separate run drains after each call as the oracle.
/// Both runs reuse the same placement, so only queue ordering changes.
pub(crate) fn upload_check(engine: &GlmfEngine<'_>, tokens: &[u32], chunk: usize) -> Result<()> {
    use super::engine::Allocator;
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--upload-check needs every layer");
    let chunk = chunk.clamp(1, engine.prefill_capacity());
    ensure!(tokens.len() > 2 * chunk && tokens.len() + 31 <= engine.max_context,
        "--upload-check needs at least three chunks and room for 31 decode rows");
    let family = GlmfPrefix::new(engine, PrefixMarks::Pool, |_| 0)?;
    let mut allocator = Allocator::with_reserved(engine.pages, engine.slots, family.layout().mark_store.reserved());
    let mut placement = allocator.admit(tokens.len() + 31)?;
    let mut expected = None;
    for barrier in [true, false] {
        engine.synchronize()?;
        placement.len = 0;
        placement.kda_len = 0;
        for part in tokens.chunks(chunk) {
            engine.prefill_device(&mut placement, part)?.context("upload check prefill logits")?;
            if barrier { engine.synchronize()?; }
        }
        let prefill = (engine.slot_state(placement.slot)?, paged_rows(&family, &placement, placement.len)?);
        for token in tokens.iter().cycle().take(31) {
            engine.verify_device(&mut [(&mut placement, 1)], &[*token], false)?
                .context("upload check decode logits")?;
            if barrier { engine.synchronize()?; }
        }
        let replay = (engine.slot_state(placement.slot)?, paged_rows(&family, &placement, placement.len)?);
        if let Some((reference_prefill, reference_replay)) = &expected {
            ensure!(reference_prefill == &prefill, "queued multi-chunk prefill differs from drained state");
            ensure!(reference_replay == &replay, "queued decode replay differs from drained state");
        } else { expected = Some((prefill, replay)); }
    }
    allocator.release(placement);
    println!("upload check: {} tokens in {} chunks, 31 decode rows: state and paged bytes identical without barriers",
        tokens.len(), tokens.len().div_ceil(chunk));
    Ok(())
}

/// glmf-golden --resume-at P: prefill `tokens[..P]` into sequence A, capture its snapshot (shared
/// units, the copied tail unit, the KDA mark), restore it into sequence B (its own KDA slot and
/// pool-page mapping), continue both with the same chunks and `decode` greedy single-row steps,
/// and compare every layer's rows, every logit, every paged row, the KDA state and the mark
/// round trip byte for byte (a restore must be exact). Every logit must also be finite: equal NaNs
/// in A and B compare identical. A straight prefill without the boundary at P is reported too
/// (informational: chunking changes may round differently).
/// With `cold`, B is prefilled from scratch on its own units instead (no restore): the floor of
/// what the kernels themselves vary. The check runs `repeat` times on fresh sequences (every
/// attempt must be identical: the DSA top-k is deterministic, ties going to the lower index).
/// `marks` picks where the two marks live: arena slots, or pool units taken beside the
/// sequences' own units (so their rows must come through the mark's units untouched too), with
/// unit 0 reserved as in serving.
/// `poison` (diagnostic, pool marks): before the continuations, fill the reserved unit's record
/// slot 0 with 0xFF (NaN as E4M3 and as FP32) in every MLA layer, and report the non-finite
/// logits instead of failing on them. Nothing but a kernel's stand-in for masked candidates reads
/// that record: non-finite decode logits show the decode sparse MLA reads it (and multiplies it
/// by zero) for masked candidates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resume_check(engine: &GlmfEngine<'_>, tokens: &[u32], at: usize, n: usize, chunk: usize, decode: usize,
    cold: bool, repeat: usize, marks: PrefixMarks, poison: bool) -> Result<()> {
    use super::engine::Allocator;
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--resume-at needs every layer");
    ensure!(engine.full_prefill_logits, "--resume-at needs every prefill row's logits");
    ensure!(at > 0 && at < n && n <= tokens.len(), "--resume-at {at} must lie inside the {n} prefilled tokens");
    ensure!(!poison || marks == PrefixMarks::Pool,
        "--resume-poison-unit0 needs --prefix-marks pool (unit 0 reserved, so nothing else reads or writes it)");
    let chunk = chunk.clamp(1, engine.prefill_rows);
    let embed = &tokens[..n];
    let family = GlmfPrefix::new(engine, marks, |_| 2)?;
    let reserved = family.layout().mark_store.reserved();
    let mut allocator = if reserved == 0 { Allocator::for_engine(engine) }
        else { Allocator::with_reserved(engine.pages, engine.slots, reserved) };
    let err = |e: BoxError| anyhow::anyhow!("{e}");
    // Marks 0 and 1: arena slots, or two marks of pool units.
    let pool_marks = match marks {
        PrefixMarks::Arena => None,
        PrefixMarks::Pool => Some([allocator.take_units(family.mark_units())?,
            allocator.take_units(family.mark_units())?]),
    };
    let capture = |mark: usize, placement: &GlmfPlacement| match &pool_marks {
        Some(pages) => family.capture_pages(&pages[mark], placement, at),
        None => family.capture(MarkSlot(mark as u32), placement, at),
    }.map_err(err);
    let restore = |mark: usize, placement: &mut GlmfPlacement| match &pool_marks {
        Some(pages) => family.restore_pages(&pages[mark], placement, at),
        None => family.restore(Some(MarkSlot(mark as u32)), placement, at),
    }.map_err(err);
    let mark_host = |mark: usize| match &pool_marks {
        Some(pages) => family.mark_pages_host(&pages[mark]),
        None => family.mark_host(MarkSlot(mark as u32)),
    };
    let (mut identical, mut marks_identical, mut last, mut bad_total) = (0, 0, None, 0);
    for attempt in 1..=repeat.max(1) {
    // Every step names where it failed (an expert exchange refusing NaN routes, a missing unit).
    let phase = |what: &str| format!("--resume-at {at}, attempt {attempt}: {what}");
    // A: prefill [0, P), capture, continue in place.
    let mut a = allocator.admit(n + decode).with_context(|| phase("admitting A"))?;
    prefill_digest(engine, &mut a, &embed[..at], chunk, false).with_context(|| phase("A's prefill to P"))?;
    family.drain().map_err(err)?;
    let started = std::time::Instant::now();
    capture(0, &a).with_context(|| phase("capturing A's mark"))?;
    // B: a second sequence restored from the snapshot (shared full units, its own tail and KDA slot).
    let mut b = if cold {
        // The floor: B prefilled cold on its own units (what placement alone changes).
        let mut b = allocator.admit(n + decode).with_context(|| phase("admitting B"))?;
        prefill_digest(engine, &mut b, &embed[..at], chunk, false).with_context(|| phase("B's cold prefill to P"))?;
        capture(1, &b).with_context(|| phase("capturing B's mark"))?;
        restore(1, &mut b).with_context(|| phase("restoring B from its own mark"))?;
        b
    } else {
        let (mut b, copy) = allocator.fork(&a, at, n + decode).with_context(|| phase("forking B from A"))?;
        if let Some(copy) = copy {
            family.copy_rows(copy).map_err(err).with_context(|| phase("copying B's tail unit"))?;
        }
        restore(0, &mut b).with_context(|| phase("restoring B from A's mark"))?;
        b
    };
    family.drain().map_err(err)?;
    let restore_ms = started.elapsed().as_secs_f64() * 1e3;
    // The restored KDA state reads back exactly as the captured one.
    capture(1, &b).with_context(|| phase("capturing B's restored state"))?;
    let mark_equal = mark_host(0)? == mark_host(1)?;
    // The state at P (every paged row and the KDA state): what a restore must reproduce.
    let state_at = paged_rows(&family, &a, at)? == paged_rows(&family, &b, at)?
        && engine.slot_state(a.slot)? == engine.slot_state(b.slot)?;
    if poison {
        family.fill_stand_in(0xFF).with_context(|| phase("poisoning the reserved record slot 0"))?;
    }
    let straight = prefill_digest(engine, &mut a, &embed[at..], chunk, true)
        .with_context(|| phase("A's continuation"))?;
    let restored = prefill_digest(engine, &mut b, &embed[at..], chunk, true)
        .with_context(|| phase("B's continuation"))?;
    let first_layer = straight.layers.iter().zip(&restored.layers).position(|(x, y)| x != y);
    let logits_equal = straight.logits == restored.logits;
    let max_diff = straight.last.iter().zip(&restored.last).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    // Greedy single-row decode steps from both (decode graphs, the restored pool mapping).
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i as u32);
    let (mut next_a, mut next_b) = (argmax(&straight.last), argmax(&restored.last));
    let mut decode_equal = next_a == next_b;
    let (mut decode_bad_a, mut decode_bad_b) = (0, 0);
    for step in 1..=decode {
        let la = engine.verify(&mut [(&mut a, 1)], &[next_a], None)
            .with_context(|| phase(&format!("A's decode step {step}")))?.context("decode logits")?;
        let lb = engine.verify(&mut [(&mut b, 1)], &[next_b], None)
            .with_context(|| phase(&format!("B's decode step {step}")))?.context("decode logits")?;
        decode_equal &= la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits());
        (decode_bad_a, decode_bad_b) = (decode_bad_a + nonfinite(&la), decode_bad_b + nonfinite(&lb));
        (next_a, next_b) = (argmax(&la), argmax(&lb));
    }
    if poison {
        family.fill_stand_in(0).with_context(|| phase("clearing the reserved record slot 0"))?;
    }
    let len = a.len;
    let paged_equal = paged_rows(&family, &a, len)? == paged_rows(&family, &b, len)?;
    let state_equal = engine.slot_state(a.slot)? == engine.slot_state(b.slot)?;
    let bad = straight.nonfinite + restored.nonfinite + decode_bad_a + decode_bad_b;
    let store = if pool_marks.is_some() { format!("in {} pool units, unit 0 reserved", family.mark_units()) }
        else { "in an arena slot".into() };
    println!("resume at {at} of {n} (chunks of {chunk}, {decode} decode steps): state at {at} {} | layers {} | logits \
        {} (last row max |diff| {max_diff:.3e}) | decode {} | paged rows 0..{len} {} | KDA state {} | mark round trip \
        {} ({} B {store}), capture+restore {restore_ms:.1} ms | non-finite logits: prefill {} + {}, decode {} + {}{}",
        if state_at { "identical" } else { "DIFFERS" },
        first_layer.map_or("identical".to_string(), |l| format!("differ from layer {l}")),
        if logits_equal { "identical" } else { "DIFFER" }, if decode_equal { "identical" } else { "DIFFERS" },
        if paged_equal { "identical" } else { "DIFFER" }, if state_equal { "identical" } else { "DIFFERS" },
        if mark_equal { "identical" } else { "DIFFERS" }, family.mark_bytes(), straight.nonfinite, restored.nonfinite,
        decode_bad_a, decode_bad_b, if poison { " [record slot 0 poisoned]" } else { "" });
    allocator.release(b);
    allocator.release(a);
    identical += usize::from(first_layer.is_none() && logits_equal && decode_equal && paged_equal && state_equal
        && mark_equal && (poison || bad == 0));
    marks_identical += usize::from(mark_equal && state_at);
    bad_total += bad;
    last = Some(restored);
    }
    let restored = last.context("no attempt")?;
    let repeat = repeat.max(1);
    // C: one prefill with no boundary at P (chunking changes may round differently; informational).
    let mut c = allocator.admit(n)?;
    let whole = prefill_digest(engine, &mut c, embed, chunk, true)
        .with_context(|| format!("--resume-at {at}: the straight prefill without the boundary"))?;
    let x = &whole.argmax[at..];
    let agree = x.iter().zip(&restored.argmax).filter(|(p, q)| p == q).count();
    let last_equal = whole.last.iter().zip(&restored.last).all(|(p, q)| p.to_bits() == q.to_bits());
    println!("vs one straight prefill without the boundary at {at}: suffix top-1 agreement {agree}/{}, last row \
        logits {}, non-finite logits {}", x.len(), if last_equal { "identical" } else { "differ" }, whole.nonfinite);
    allocator.release(c);
    for pages in pool_marks.iter().flatten() {
        allocator.release_units(pages);
    }
    println!("resume at {at} of {n} (chunks of {chunk}): {identical}/{repeat} attempts byte-identical, mark round trip \
        and state at {at} identical in {marks_identical}/{repeat}{}", if cold { " [cold floor: B prefilled, not restored]" } else { "" });
    if poison {
        println!("record slot 0 poisoned with 0xFF in every MLA layer: {bad_total} non-finite logits across the \
            attempts (non-zero: a kernel reads it for masked candidates; the prefill path should stay finite)");
    }
    ensure!(marks_identical == repeat, "the restored state differs from the captured one");
    ensure!(poison || (bad_total == 0 && whole.nonfinite == 0), "{} non-finite logits (A, B and the straight prefill)",
        bad_total + whole.nonfinite);
    ensure!(identical == repeat, "the restored sequence's continuation differs from the straight one");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::engine::{Allocator, GlmfPlacement};

    #[test]
    fn heads_host_round_trip_keeps_one_page_copy_and_both_rank_marks_exact() {
        use cuteafd_hostcache::{cache::{DevicePage, DeviceSnapshot, HostCache, RestoreOutcome, RestoreTarget},
            config::Config, copy::{CopyModel, DeviceRange, StubCopyEngine}, pool::Layout,
            snapshot::{DevicePageId, SnapshotMeta}};
        use cuteafd_core::prefix::{ReuseRule, SnapshotKind};
        use cuteafd_ffi::CuteafdDeviceBuffer;
        let rows = super::UNIT_ROWS;
        let sizes = [rows * super::RECORD_BYTES, rows * super::KEY_BYTES, super::POOL_PAGE_BYTES];
        let unit: usize = sizes.iter().sum();
        let gpu_bytes = unit * 2 + 64;
        let mut memory = StubCopyEngine::new(CopyModel::default(), gpu_bytes * 2, 1 << 26);
        let buffer = |gpu: usize, offset: usize, bytes: usize| CuteafdDeviceBuffer {
            ptr: (gpu * gpu_bytes + offset) as *mut std::ffi::c_void, bytes, device_id: gpu as i32, flags: 0 };
        let paged = (0..2).map(|rank| (rank, super::PagedLayer {
            records: buffer(rank, 0, sizes[0] * 2),
            keys: Some(buffer(rank, sizes[0] * 2, sizes[1] * 2)),
            pools: buffer(rank, (sizes[0] + sizes[1]) * 2, sizes[2] * 2),
        })).collect::<Vec<_>>();
        let left = super::unit_ranges(&paged, 0, 1).unwrap();
        let right = super::unit_ranges(&paged, 1, 1).unwrap();
        let replicas = super::replica_ranges(&left, &right);
        let page = left.iter().map(|b| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }).collect::<Vec<_>>();
        assert_eq!(page.iter().map(|r| r.bytes).sum::<usize>(), unit);
        let marks = (0..2).map(|gpu| DeviceRange { addr: (gpu * gpu_bytes + unit * 2) as u64, bytes: 17 })
            .collect::<Vec<_>>();
        let layout = Layout { page: unit, tail: 34, draft: 0, scores: 0 };
        let config = Config { bytes: 1 << 26, chunk_bytes: 1 << 22, min_tokens: 1, ..Default::default() };
        for (src, dst) in &replicas {
            memory.write_device(*src, &vec![5; src.bytes]);
            memory.write_device(*dst, &vec![5; dst.bytes]);
        }
        for (gpu, mark) in marks.iter().enumerate() { memory.write_device(*mark, &vec![11 + gpu as u8; mark.bytes]); }
        let mut host = HostCache::with_rule(config, layout, memory, ReuseRule::EXACT,
            cuteafd_hostcache::snapshot::EvictionOrder::LeastRecent).unwrap();
        for (kind, tokens) in [(SnapshotKind::Prompt, vec![1, 2]), (SnapshotKind::Turn, vec![1, 2, 3])] {
            let mut pages: [Vec<DevicePage>; cuteafd_hostcache::COMPRESSORS] = Default::default();
            pages[0].push(DevicePage { id: DevicePageId { compressor: 0, page: 1, generation: 1 }, segments: page.clone() });
            let snapshot = DeviceSnapshot { meta: SnapshotMeta { kind, end: tokens.len() as u32, tokens: tokens.clone(),
                has_draft: false }, pages: pages.clone(), tail: marks.clone(), draft: None, scores: vec![] };
            assert!(matches!(host.store(&snapshot, ()), cuteafd_hostcache::cache::StoreOutcome::Issued(_)));
            host.engine_mut().advance(100_000_000);
            let report = host.tick();
            let key = report.committed[0].1;
            for (src, dst) in &replicas {
                host.engine_mut().write_device(*src, &vec![0; src.bytes]);
                host.engine_mut().write_device(*dst, &vec![0; dst.bytes]);
            }
            for mark in &marks { host.engine_mut().write_device(*mark, &vec![0; mark.bytes]); }
            let target = RestoreTarget { pages, tail: marks.clone(), draft: None, scores: vec![] };
            assert!(matches!(host.restore(key, &target), RestoreOutcome::Done { .. }));
            for (src, dst) in &replicas {
                let bytes = host.engine_mut().read_device(*src);
                host.engine_mut().write_device(*dst, &bytes);
                assert_eq!(bytes, vec![5; src.bytes]);
                assert_eq!(host.engine_mut().read_device(*dst), bytes);
            }
            for (gpu, mark) in marks.iter().enumerate() {
                assert_eq!(host.engine_mut().read_device(*mark), vec![11 + gpu as u8; mark.bytes]);
                host.engine_mut().write_device(*mark, &vec![0; mark.bytes]);
            }
            let before = host.metrics().restore_bytes;
            assert!(matches!(host.restore_mark(key, marks.clone()), RestoreOutcome::Done { .. }));
            assert_eq!(host.metrics().restore_bytes - before, 34, "resident KV is not recopied");
            for (gpu, mark) in marks.iter().enumerate() {
                assert_eq!(host.engine_mut().read_device(*mark), vec![11 + gpu as u8; mark.bytes]);
            }
        }
    }

    /// GLM-5.3-Flash's text config: 34 KDA layers (64 heads of 128) and 11 MLA layers.
    fn glm53_flash() -> cuteafd_loader::families::glm5_flash::GlmNextConfig {
        let types: Vec<&str> = (0..45)
            .map(|l| if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect();
        let mlp: Vec<&str> = (0..45).map(|l| if l < 3 { "dense" } else { "sparse" }).collect();
        cuteafd_loader::families::glm5_flash::GlmNextConfig::from_hf(&serde_json::json!({
            "model_type": "glm5_next", "text_config": {
                "model_type": "glm5_next_text", "vocab_size": 154880, "hidden_size": 4096, "num_hidden_layers": 45,
                "layer_types": types, "mlp_layer_types": mlp, "intermediate_size": 12288, "n_routed_experts": 288,
                "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "routed_scaling_factor": 2.5,
                "swiglu_limit": 10.0, "rms_norm_eps": 1e-5, "hc_mult": 4, "mla_use_nope": true,
                "qk_rope_head_dim": 0, "num_attention_heads": 64, "q_lora_rank": 1536, "kv_lora_rank": 512,
                "qk_nope_head_dim": 256, "v_head_dim": 256, "index_topk": 2048, "index_kpool": 4,
                "eos_token_id": [154820, 154827, 154829],
                "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                                       "gate_lower_bound": -5.0}}})).unwrap()
    }

    /// The KV admission reserves the engine's own state: the planner's mark and replay bytes
    /// are the engine's slot regions and replay records (per GPU of a head split too, with the
    /// compact index cache's tails and key | gate records, and with a BF16 KDA state), and the
    /// arena slots it reserves are the ones `prefix_cache` allocates, for every state: 2C + 2 for
    /// 147.6 MB FP32 marks, 28 BF16 marks in the 2 GiB budget below 14 lanes. The arenas' bytes
    /// are the `prefix` ledger scopes a 32 GB card measured at 8 and 16 sequences.
    #[test]
    fn the_planner_reserves_the_marks_and_replay_records_the_engine_allocates() {
        use super::super::engine::{kda_layer_bytes, IndexCache, KdaState, DECODE_ROWS, KEY_BYTES, TAIL_BYTES};
        use crate::shared::prefix::PrefixArgs;
        use clap::Parser;
        use cuteafd_loader::families::glm5_flash::GlmNextAttention;
        use cuteafd_loader::serving_capacity::{glm_flash_rank_cache_geometry, GlmfIndexCache};
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            prefix: PrefixArgs,
        }
        let cfg = glm53_flash();
        let kda = cfg.attention.iter().filter(|&&a| a == GlmNextAttention::Kda).count();
        let mla = cfg.attention.iter().filter(|&&a| a == GlmNextAttention::Mla).count();
        for (ranks, state) in [(1, KdaState::F32), (2, KdaState::F32), (1, KdaState::Bf16)] {
            let geometry = glm_flash_rank_cache_geometry(&cfg, cfg.layers, ranks, GlmfIndexCache::Keys,
                state.bytes() as u64).unwrap();
            let (state, conv, replay) = kda_layer_bytes(&cfg, cfg.kda_heads / ranks, state, DECODE_ROWS);
            let rank = &geometry.ranks[0];
            assert_eq!((rank.retained_mark_bytes, rank.active_state_per_sequence_bytes),
                ((kda * (state + conv)) as u64, (kda * (state + conv)) as u64));
            assert_eq!(rank.speculative_replay_bytes, (kda * replay) as u64);
        }
        // The compact index cache (one GPU): `Caches::new` adds every MLA layer's tail to a slot's
        // regions (so to every mark) and a 64-row key | gate record per MLA layer.
        for state in [KdaState::F32, KdaState::Bf16] {
            let compact = glm_flash_rank_cache_geometry(&cfg, cfg.layers, 1, GlmfIndexCache::Compact,
                state.bytes() as u64).unwrap();
            let (state, conv, replay) = kda_layer_bytes(&cfg, cfg.kda_heads, state, DECODE_ROWS);
            let rank = &compact.ranks[0];
            let slot = kda * (state + conv) + mla * TAIL_BYTES;
            assert_eq!((rank.retained_mark_bytes, rank.active_state_per_sequence_bytes), (slot as u64, slot as u64));
            assert_eq!(rank.speculative_replay_bytes, (kda * replay + mla * 64 * KEY_BYTES) as u64);
        }
        let (state, conv, replay) = kda_layer_bytes(&cfg, cfg.kda_heads, KdaState::F32, DECODE_ROWS);
        let mark = kda * (state + conv);
        let bf16 = kda * (kda_layer_bytes(&cfg, cfg.kda_heads, KdaState::Bf16, DECODE_ROWS).0 + conv);
        assert_eq!((mark, bf16, kda * replay, mla * 64 * KEY_BYTES), (147_619_840, 76_316_672, 321_421_312, 360_448));
        let prefix = Cli::parse_from(["serve"]).prefix;
        for (lanes, f32_slots, bf16_slots) in [(4, 14, 28), (8, 18, 28), (16, 34, 34), (64, 130, 130)] {
            for (index, state, mark, slots) in [(IndexCache::Keys, KdaState::F32, mark, f32_slots),
                (IndexCache::Compact, KdaState::F32, mark + mla * TAIL_BYTES, f32_slots),
                (IndexCache::Keys, KdaState::Bf16, bf16, bf16_slots),
                (IndexCache::Compact, KdaState::Bf16, bf16 + mla * TAIL_BYTES, bf16_slots)] {
                let planned = super::ArenaMarks::Rule(prefix.mark_rule(lanes)).slots_on(&cfg, cfg.layers, index, state)
                    .unwrap();
                let allocated = cuteafd_engine::prefix::MarkArena::slots_for(lanes, prefix.prefix_cache_entries, mark,
                    prefix.prefix_cache_mark_mib << 20, prefix.host_cache_bytes.enabled());
                assert_eq!((planned, allocated), (slots, slots), "{lanes} lanes, {index:?}, {state:?}");
            }
        }
        let hosted = Cli::parse_from(["serve", "--host-cache-bytes", "auto"]).prefix;
        for lanes in [1, 16, 64] {
            for (index, state) in [(IndexCache::Keys, KdaState::F32), (IndexCache::Compact, KdaState::Bf16)] {
                let bytes = glm_flash_rank_cache_geometry(&cfg, cfg.layers, 1, index.into(), state.bytes() as u64)
                    .unwrap().ranks[0].retained_mark_bytes;
                let planned = super::ArenaMarks::Rule(hosted.mark_rule(lanes))
                    .slots_on(&cfg, cfg.layers, index, state).unwrap();
                assert_eq!(planned as u64, cuteafd_core::prefix::mark_slots_with_host(lanes as u64,
                    hosted.prefix_cache_entries as u64, bytes, (hosted.prefix_cache_mark_mib as u64) << 20, true));
                assert_eq!(planned, 2);
            }
        }
        // What either admission now reserves is the arena the ledger measured as `prefix`: FP32 at 8
        // and 16 sequences, compact at 16, BF16 at 8 and 16 (a flat 18-mark reserve left 16 FP32
        // marks, 2,361,917,440 B, out at 16 sequences).
        let arena = |lanes, index: IndexCache, state: KdaState| {
            let mark = glm_flash_rank_cache_geometry(&cfg, cfg.layers, 1, index.into(), state.bytes() as u64).unwrap()
                .ranks[0].retained_mark_bytes as usize;
            super::ArenaMarks::Rule(prefix.mark_rule(lanes)).slots_on(&cfg, cfg.layers, index, state).unwrap() * mark
        };
        assert_eq!(arena(8, IndexCache::Keys, KdaState::F32), 2_657_157_120);
        assert_eq!(arena(16, IndexCache::Keys, KdaState::F32), 5_019_074_560);
        assert_eq!(arena(16, IndexCache::Compact, KdaState::F32), 5_019_655_008);
        assert_eq!(arena(8, IndexCache::Keys, KdaState::Bf16), 2_136_866_816);
        assert_eq!(arena(16, IndexCache::Keys, KdaState::Bf16), 2_594_766_848);
        assert_eq!(arena(16, IndexCache::Keys, KdaState::F32) - 18 * mark, 2_361_917_440);
        let off = PrefixArgs { prefix_cache_entries: 0, ..prefix };
        assert_eq!(super::ArenaMarks::Rule(off.mark_rule(16))
            .slots_on(&cfg, cfg.layers, IndexCache::Keys, KdaState::F32).unwrap(), 0);
    }

    /// The engine's own mark (`slot_regions_on` summed over its GPUs, as `GlmfPrefix::new` sums
    /// them): every KDA layer's state and conv window over each GPU's heads, and with the compact
    /// index cache (one GPU) every MLA layer's index tail.
    fn engine_mark(cfg: &cuteafd_loader::families::glm5_flash::GlmNextConfig, ranks: usize,
        index: super::IndexCache, state: super::KdaState) -> (Vec<usize>, usize) {
        use cuteafd_loader::families::glm5_flash::GlmNextAttention;
        let kda = cfg.attention.iter().filter(|&&a| a == GlmNextAttention::Kda).count();
        let mla = cfg.attention.iter().filter(|&&a| a == GlmNextAttention::Mla).count();
        let (state, conv, _) = super::super::engine::kda_layer_bytes(cfg, cfg.kda_heads / ranks, state, super::super::engine::DECODE_ROWS);
        let tails = if index == super::IndexCache::Compact { mla * super::super::engine::TAIL_BYTES } else { 0 };
        let parts = vec![kda * (state + conv) + tails; ranks];
        let mark = parts.iter().sum();
        (parts, mark)
    }

    #[test]
    fn whole_mla_owner_marks_equal_admission_on_both_first_owners() {
        use super::super::engine::{self, IndexCache, KdaState};
        use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
        use cuteafd_loader::placement::AttentionPlacement;
        use cuteafd_loader::serving_capacity::glm_flash_placed_cache_geometry_rows;
        for layers in [2, 6] {
        let cfg = GlmNextConfig::from_hf(&cuteafd_loader::plan::testing::glm5_flash_config(layers)).unwrap();
        for index in [IndexCache::Keys, IndexCache::Compact] {
            for first_owner in [0, 1] {
                let geometry = glm_flash_placed_cache_geometry_rows(&cfg, cfg.layers, 2, index.into(),
                    KdaState::F32.bytes() as u64, engine::DECODE_ROWS as u64,
                    AttentionPlacement::Layers, first_owner).unwrap();
                let (state, conv, _) = engine::kda_layer_bytes(&cfg, cfg.kda_heads / 2,
                    KdaState::F32, engine::DECODE_ROWS);
                let mut parts = [0; 2];
                let mut owner = first_owner;
                for kind in &cfg.attention {
                    match kind {
                        GlmNextAttention::Kda => parts.iter_mut().for_each(|part| *part += state + conv),
                        GlmNextAttention::Mla => {
                            if index == IndexCache::Compact { parts[owner] += engine::TAIL_BYTES; }
                            owner ^= 1;
                        }
                    }
                }
                assert_eq!(geometry.ranks.iter().map(|rank| rank.retained_mark_bytes).collect::<Vec<_>>(),
                    parts.map(|part| part as u64));
                let total = parts.iter().sum::<usize>();
                // Whole MLA ownership partitions the tails; KDA partitions heads, so
                // aggregate mark size stays identical to the existing single-rank rule.
                let budget = 14 * total as u64;
                let arena = super::ArenaMarks::Rule(crate::shared::prefix::MarkRule {
                    lanes: 5, entries: 20, budget_bytes: budget, host: false });
                assert_eq!(arena.slots_placed(&cfg, cfg.layers, index, KdaState::F32, false).unwrap(),
                    arena.slots(total));
                assert_eq!(arena.slots(total), 14);
            }
        }
        }
    }

    /// The prefix mark arena under a head split with `--index-cache compact`: the engine keeps
    /// the token keys (`served_index_cache`), so its marks have no index tails and are 17,072 B
    /// smaller. With a 1,971 MiB budget, at most 5 lanes and at least 6 entries, the budget holds
    /// 14 of them but only 13 compact marks: the admission counted on the requested cache
    /// reserved 13 and `prefix_cache` refused its 14 after allocation. Counted on the layout the
    /// engine serves, both sides are 14, and each GPU reserves its half of every mark.
    #[test]
    fn a_head_split_counts_its_mark_arena_on_the_token_keys_it_serves() {
        use super::super::engine::{IndexCache, KdaState};
        use super::super::served_index_cache;
        use cuteafd_loader::serving_capacity::glm_flash_rank_cache_geometry;
        let cfg = glm53_flash();
        let served = served_index_cache(IndexCache::Compact, true, cuteafd_loader::placement::AttentionPlacement::Heads);
        assert_eq!((served, served_index_cache(IndexCache::Compact, false, cuteafd_loader::placement::AttentionPlacement::Heads)), (IndexCache::Keys, IndexCache::Compact));
        let (parts, mark) = engine_mark(&cfg, 2, served, KdaState::F32);
        let compact = engine_mark(&cfg, 1, IndexCache::Compact, KdaState::F32).1;
        assert_eq!((parts[0], mark, compact), (73_809_920, 147_619_840, 147_636_912));
        let budget = 1971u64 << 20;
        assert!(14 * mark as u64 <= budget && 14 * compact as u64 > budget);
        for lanes in 1..=5 {
            for entries in [6, 7, 20, 64] {
                let arena = super::ArenaMarks::Rule(crate::shared::prefix::MarkRule { lanes, entries,
                    budget_bytes: budget, host: false });
                let planned = arena.slots_on(&cfg, cfg.layers, served, KdaState::F32).unwrap();
                assert_eq!((planned, arena.slots(mark)), (14, 14), "{lanes} lanes, {entries} entries");
                // The count on the cache requested: one short of the arena allocated.
                assert_eq!(arena.slots_on(&cfg, cfg.layers, IndexCache::Compact, KdaState::F32).unwrap(), 13);
                // Each GPU's planned reserve is its part of every mark: the arena it allocates.
                let split = glm_flash_rank_cache_geometry(&cfg, cfg.layers, 2, served.into(), 4).unwrap();
                for (rank, part) in split.ranks.iter().zip(&parts) {
                    assert_eq!(planned as u64 * rank.retained_mark_bytes, (14 * part) as u64);
                }
            }
        }
    }

    /// Either KV admission's arena count (on the checkpoint's geometry of the layout the engine
    /// serves) is the count `prefix_cache` sizes over the engine's own marks, and its bytes on
    /// every GPU the arena's, for a head split or one GPU, either index cache requested, FP32 and
    /// BF16 state, and budgets that do and do not bind.
    #[test]
    fn planned_mark_arena_is_the_runtime_arena_on_every_layout() {
        use super::super::engine::{IndexCache, KdaState};
        use cuteafd_loader::serving_capacity::glm_flash_rank_cache_geometry;
        let cfg = glm53_flash();
        let mut binding = 0;
        for split in [false, true] {
            for requested in [IndexCache::Keys, IndexCache::Compact] {
                let served = super::super::served_index_cache(requested, split, cuteafd_loader::placement::AttentionPlacement::Heads);
                let ranks = if split { 2 } else { 1 };
                for state in [KdaState::F32, KdaState::Bf16] {
                    let (parts, mark) = engine_mark(&cfg, ranks, served, state);
                    let geometry = glm_flash_rank_cache_geometry(&cfg, cfg.layers, ranks, served.into(),
                        state.bytes() as u64).unwrap();
                    for mib in [0, 1000, 1971, 2048, 4096] {
                        for lanes in [1, 4, 5, 8, 16, 64] {
                            for entries in [0, 6, 20] {
                                for host in [false, true] {
                                    let rule = crate::shared::prefix::MarkRule { lanes, entries, budget_bytes: mib << 20, host };
                                    let arena = super::ArenaMarks::Rule(rule);
                                    let planned = arena.slots_on(&cfg, cfg.layers, served, state).unwrap();
                                    let allocated = arena.slots(mark);
                                    assert_eq!(planned, allocated, "split {split}, {requested:?}, {state:?}, {mib} MiB, \
                                        {lanes} lanes, {entries} entries, host {host}");
                                    for (rank, part) in geometry.ranks.iter().zip(&parts) {
                                        assert_eq!(planned as u64 * rank.retained_mark_bytes, (allocated * part) as u64);
                                    }
                                    if host { assert_eq!(planned, if entries == 0 { 0 } else { 2 }); }
                                    binding += usize::from(!host && entries > 0 && planned < 2 * entries + 2
                                        && planned > 2 * lanes + 2);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Budgets that bind between the lane floor and the entries' wish (the rounding at stake).
        assert!(binding > 0);
        // A fixed count, or none, stands on any layout.
        assert_eq!(super::ArenaMarks::Slots(2).slots_on(&cfg, cfg.layers, IndexCache::Compact, KdaState::F32)
            .unwrap(), 2);
        assert_eq!(super::ArenaMarks::None.slots_on(&cfg, cfg.layers, IndexCache::Keys, KdaState::F32).unwrap(), 0);
    }

    /// Without entries no mark is taken: pool marks need no room in the pool and keep no unit
    /// back, and the arena rule sizes no arena.
    #[test]
    fn no_entries_take_no_prefix_marks() {
        use super::PrefixMarks;
        assert_eq!(PrefixMarks::Pool.with_entries(0), PrefixMarks::Arena);
        assert_eq!(PrefixMarks::Pool.with_entries(1), PrefixMarks::Pool);
        assert_eq!(PrefixMarks::Arena.with_entries(0), PrefixMarks::Arena);
        let rule = crate::shared::prefix::MarkRule { lanes: 16, entries: 0, budget_bytes: 2 << 30, host: false };
        assert_eq!(super::ArenaMarks::Rule(rule).slots(147_619_840), 0);
    }

    /// Pool marks off the GPU: a mark laid out buffer by buffer over its units round trips
    /// byte-exactly between scattered KDA slot regions, writes no byte outside its runs, and its
    /// copies merge over consecutive units.
    #[test]
    fn pool_marks_gather_and_scatter_slot_regions_exactly() {
        use super::{gather_scatter, mark_runs, Span, SpanCopy};
        // Two MLA layers' buffers of 6 units (records 40 B, keys 24 B, pool keys 6 B per unit),
        // separate allocations, then the KDA state and conv pools of 3 layers x 2 slots (50 B and
        // 9 B per region), as `Caches::new` lays them out: a unit holds 140 B, a mark 177 B, so
        // 2 units per mark.
        let (units, slots, kda) = (6, 2, 3);
        let mut buffers = Vec::new();
        let mut at = 0;
        for _ in 0..2 {
            for bytes in [40, 24, 6] {
                buffers.push((at, bytes));
                at += units * bytes + 16;
            }
        }
        let unit = |u: usize| -> Vec<Span> { buffers.iter().map(|&(base, bytes)| (base + u * bytes, bytes)).collect() };
        let (state, conv) = (at, at + kda * slots * 50);
        let regions = |slot: usize| -> Vec<Span> {
            (0..kda).map(|k| (state + (k * slots + slot) * 50, 50))
                .chain((0..kda).map(|k| (conv + (k * slots + slot) * 9, 9))).collect()
        };
        let end = conv + kda * slots * 9;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let memory: Vec<u8> = (0..end)
            .map(|_| { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed as u8 }).collect();
        let bytes_of = |memory: &[u8], spans: &[Span]| -> Vec<u8> {
            spans.iter().flat_map(|&(at, n)| memory[at..at + n].to_vec()).collect()
        };
        let apply = |memory: &mut Vec<u8>, copies: &[SpanCopy]| {
            for c in copies {
                memory.copy_within(c.src..c.src + c.bytes, c.dst);
            }
        };
        let part = 177;
        assert_eq!(regions(0).iter().map(|&(_, n)| n).sum::<usize>(), part);
        for (pages, runs_expected) in [([2usize, 3], 4), ([0, 5], 7)] {
            let runs = mark_runs(&pages.iter().map(|&u| unit(u)).collect::<Vec<_>>(), part);
            assert_eq!((runs.len(), runs.iter().map(|&(_, n)| n).sum::<usize>()), (runs_expected, part), "{pages:?}");
            // Capture slot 0 into the mark, then restore it into slot 1.
            let mut after = memory.clone();
            let capture = gather_scatter(&regions(0), &runs);
            apply(&mut after, &capture);
            assert_eq!(bytes_of(&after, &runs), bytes_of(&memory, &regions(0)));
            let restore = gather_scatter(&runs, &regions(1));
            assert_eq!(restore.iter().map(|c| c.bytes).sum::<usize>(), part);
            apply(&mut after, &restore);
            assert_eq!(bytes_of(&after, &regions(1)), bytes_of(&memory, &regions(0)), "{pages:?}");
            // Nothing else moved: every byte outside the runs and slot 1 is as it was.
            let written: Vec<Span> = runs.iter().copied().chain(regions(1)).collect();
            for (i, (x, y)) in memory.iter().zip(&after).enumerate() {
                if !written.iter().any(|&(at, n)| (at..at + n).contains(&i)) {
                    assert_eq!(x, y, "byte {i} outside the mark and slot 1 changed ({pages:?})");
                }
            }
        }
        // Consecutive units merge each buffer's segments: fewer copies than scattered units.
        let consecutive = gather_scatter(&regions(0), &mark_runs(&[unit(2), unit(3)], part)).len();
        let scattered = gather_scatter(&regions(0), &mark_runs(&[unit(0), unit(5)], part)).len();
        assert!(consecutive < scattered, "{consecutive} vs {scattered}");
        // One unit holds less than a mark (the family refuses that, never truncates the state).
        assert_eq!(mark_runs(&[unit(1)], part).iter().map(|&(_, n)| n).sum::<usize>(), 140);
    }

    #[test]
    fn pool_marks_keep_kda_only_and_mla_owners_byte_exact() {
        use super::{gather_scatter, mark_runs};
        // Run both all-MLA-on-one-rank directions and a split MLA inventory.
        for mla in [[0usize, 2], [2, 0], [1, 1]] {
            for (rank, layers) in mla.into_iter().enumerate() {
                let part = 59 + layers * 7; // recurrent state + conv + compact tails
                let backing = if layers == 0 { part } else { layers * 40 };
                let mut memory = (0..1024).map(|i| ((i * 37 + rank * 61) % 251) as u8).collect::<Vec<_>>();
                let original = memory.clone();
                let from = [(600, 50), (700, 9), (750, layers * 7)];
                let to = [(800, 50), (900, 9), (950, layers * 7)];
                let units = vec![vec![(backing, backing)], vec![(3 * backing, backing)]];
                let runs = mark_runs(&units, part);
                assert_eq!(runs.iter().map(|&(_, n)| n).sum::<usize>(), part);
                for copy in gather_scatter(&from, &runs) {
                    memory.copy_within(copy.src..copy.src + copy.bytes, copy.dst);
                }
                for copy in gather_scatter(&runs, &to) {
                    memory.copy_within(copy.src..copy.src + copy.bytes, copy.dst);
                }
                for ((src, len), (dst, _)) in from.into_iter().zip(to) {
                    assert_eq!(memory[dst..dst + len], original[src..src + len], "rank {rank}, MLA {mla:?}");
                }
                for i in 0..memory.len() {
                    if !runs.iter().chain(&to).any(|&(at, len)| (at..at + len).contains(&i)) {
                        assert_eq!(memory[i], original[i]);
                    }
                }
            }
        }
    }

    /// Pool marks keep unit 0 out of every allocation: the decode sparse MLA reads its first record
    /// (slot 0) for every masked candidate and weights it by zero, so a mark's bytes there would
    /// make every decode row with a masked candidate NaN. The golden harness's allocator takes its
    /// marks first, as in the failing `--resume-at` runs, and still never hands out unit 0; the
    /// marks' runs never start at record slot 0 of any MLA layer.
    #[test]
    fn pool_marks_never_take_the_stand_in_unit() {
        use super::{mark_runs, Span, RESERVED_UNITS};
        assert_eq!(RESERVED_UNITS, 1);
        let (pages, slots, mark_units) = (512 * 4, 4, 49);
        let mut allocator = Allocator::with_reserved(pages, slots, RESERVED_UNITS);
        let marks = [allocator.take_units(mark_units).unwrap(), allocator.take_units(mark_units).unwrap()];
        let a = allocator.admit(5008).unwrap();
        let (b, _) = allocator.fork(&a, 2600, 5008).unwrap();
        let c = allocator.admit(5000).unwrap();
        for units in marks.iter().chain([&a.units, &b.units, &c.units]) {
            assert!(!units.contains(&0), "{units:?}");
        }
        assert_eq!(marks[0][0], 1);
        // Unit u's segments in a fake address space: records of every unit first (264 B each),
        // then pool keys (8 B each), as two MLA layers' buffers would lie.
        let unit = |u: u32| -> Vec<Span> { vec![(u as usize * 264, 264), (1 << 20 | u as usize * 8, 8)] };
        let runs = mark_runs(&marks[0].iter().map(|&u| unit(u)).collect::<Vec<_>>(), 1000);
        assert!(runs.iter().all(|&(at, _)| at != 0 && at != 1 << 20), "{runs:?}");
        // Unreserved, the same allocation puts the first mark on unit 0 (the failing runs).
        let mut unreserved = Allocator::new(pages, slots);
        assert_eq!(unreserved.take_units(mark_units).unwrap()[0], 0);
    }

    #[test]
    fn units_expand_to_mla_and_pool_pages_and_forks_share_whole_units() {
        let p = GlmfPlacement::new(vec![2, 0], 1);
        assert_eq!(p.pages, vec![8, 9, 10, 11, 0, 1, 2, 3]);
        assert_eq!(p.pool_pages, vec![2, 0]);
        // Position 300: the second unit (index 0), its MLA page 0 (page 0), row 44; pool 74 (row 10
        // of pool page 0) completes at 299.
        assert_eq!(p.record(300).unwrap(), 44);
        assert_eq!(p.pool_slot(299).unwrap(), 10);
        assert_eq!(p.pool_slot(300).unwrap(), -1);
        assert_eq!(p.pool_slot(255).unwrap(), 2 * 64 + 63);
        let mut allocator = Allocator::new(4 * 8, 3);
        let a = allocator.admit(600).unwrap();
        assert_eq!(a.units.len(), 3);
        let (b, copy) = allocator.fork(&a, 300, 900).unwrap();
        assert_eq!((b.units.len(), b.units[0]), (4, a.units[0]));
        let copy = copy.unwrap();
        assert_eq!((copy.from, copy.to, copy.rows), (a.units[1], b.units[1], 44));
        assert_ne!(a.slot, b.slot);
        allocator.release(b);
        allocator.release(a);
        assert_eq!(allocator.admit(8 * 256).unwrap().units.len(), 8);
    }
}
