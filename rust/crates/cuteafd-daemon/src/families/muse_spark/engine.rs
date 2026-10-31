//! One coordinator rank's resident Muse state, loaded after admission in the
//! order admission charged it (weights, KV, the rotary table, the lane, the
//! K6 scratch), every owner bound to the rank's device. The rank's bound
//! programs come from the caller's per-device `Programs` (which own the
//! loaded modules and outlive this state). `RankResources::device` lends the
//! device layer (`RankDevice`) its view for one step.
use super::{core::CorePrograms, device::{RankDevice, RankPrograms, StepPlanes, Tp2Ops}, lane::Lane, step::StepMetadata,
    storage::RankKv, weights::{load_rank, LayerWeights}};
use crate::shared::memory::device::Allocation;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::muse_spark::lane::StepTables;
use crate::shared::{memory::device::Device, rope::{RotaryTables, Table}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::Programs;
use cuteafd_loader::families::muse_spark::{cache::MuseCacheLayout, lane::LanePlanes, MuseSparkConfig};
use cuteafd_loader::placement::{families::muse_spark::{self as shared, MuseRequest}, Placement};
use std::{ffi::c_void, rc::Rc};

/// What one rank's load needs from startup.
pub(crate) struct RankLoad<'c> {
    pub cfg: &'c MuseSparkConfig,
    pub checkpoint: &'c cuteafd_loader::plan::Checkpoint,
    pub built: &'c MuseRequest,
    pub placement: &'c Placement,
    pub rank: usize,
    /// Active sequence slots (each owns its sliding rings).
    pub slots: u64,
    /// Prefill unit rows and waves in flight (the lane's span).
    pub unit_rows: usize,
    pub waves: usize,
    /// The context the rotary table was admitted at (the request's `context`).
    pub context: u64,
}

pub(crate) struct RankResources<'a> {
    pub device: Device<'a>,
    pub programs: RankPrograms<'a>,
    pub core: CorePrograms<'a>,
    pub layers: Vec<LayerWeights<'a>>,
    pub kv: RankKv<'a>,
    pub lane: Lane<'a>,
    rope: Rc<Table<'a>>,
    _rotary: RotaryTables<'a>,
    pub tp2: Option<Tp2Ops<'a>>,
    pub ranks: usize,
    pub rank: usize,
    pub wire_grid: crate::shared::launch_grid::Fp8QuantizeGrid,
    pub step: StepStorage<'a>,
}

/// The step metadata planes beyond the lane (`StepTables`, as admitted).
pub(crate) struct StepStorage<'a> {
    owner: Allocation<'a>,
    pub tables: StepTables,
}

fn at(base: Buffer, offset: usize, bytes: usize) -> Buffer {
    Buffer { ptr: base.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..base }
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: plain-old-data integers, read as their in-memory bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl<'a> StepStorage<'a> {
    pub(crate) fn allocate(device: Device<'a>, tables: StepTables) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("workspace/muse step metadata");
        Ok(Self { owner: Allocation::new(device, tables.bytes())?, tables })
    }

    /// (page_slots, step_slots, page_table) whole planes.
    fn planes(&self) -> [Buffer; 3] {
        let slots = self.tables.slots_bytes().next_multiple_of(cuteafd_loader::families::muse_spark::lane::ALIGN);
        [at(self.owner.buffer, 0, self.tables.slots_bytes()), at(self.owner.buffer, slots, self.tables.slots_bytes()),
            at(self.owner.buffer, 2 * slots, self.tables.table_bytes())]
    }

    /// Upload one step's metadata on `stream` (ordered after earlier readers of
    /// these planes, outside capture) and return the planes the layers read.
    pub(crate) fn upload(&self, library: &cuteafd_ffi::NativeLibrary, lane: &super::lane::LaneViews,
        meta: &StepMetadata, prefill: bool, stream: *mut c_void) -> Result<StepPlanes> {
        let rows = meta.positions.len();
        ensure!(rows <= self.tables.rows && meta.page_table.len() * 4 <= self.tables.table_bytes()
            && meta.table_stride <= self.tables.stride, "Muse step of {rows} rows exceeds its admitted metadata");
        let [page_slots, step_slots, page_table] = self.planes();
        let (positions, ring_slots, seq_first) = (lane.rows("positions", rows)?, lane.rows("ring_slots", rows)?,
            lane.rows("seq_first", rows)?);
        for (destination, source) in [(lane.plane("positions")?, bytes_of(&meta.positions)),
            (lane.plane("ring_slots")?, bytes_of(&meta.ring_slots)), (lane.plane("seq_first")?, bytes_of(&meta.seq_first)),
            (page_slots, bytes_of(&meta.page_slots)), (step_slots, bytes_of(&meta.step_slots())),
            (page_table, bytes_of(&meta.page_table))] {
            // SAFETY: fixed admitted destinations of this device; every reader is
            // ordered on `stream`; the ordered upload retains its own staging.
            unsafe { library.copy_h2d_ordered(destination, source, stream)?; }
        }
        let planes = StepPlanes { rows, positions, ring_slots, seq_first,
            page_slots: Buffer { bytes: rows * 8, ..page_slots }, step_slots: Buffer { bytes: rows * 8, ..step_slots },
            page_table: Buffer { bytes: meta.page_table.len() * 4, ..page_table }, table_stride: meta.table_stride,
            prefill };
        planes.check(meta)?;
        Ok(planes)
    }
}

impl<'a> RankResources<'a> {
    pub(crate) fn load(device: Device<'a>, programs: &'a Programs<'a>, load: RankLoad<'_>) -> Result<Self> {
        let RankLoad { cfg, checkpoint, built, placement, rank, slots, unit_rows, waves, context } = load;
        let tables = shared::step_tables(unit_rows, usize::try_from(slots)?, context)?;
        let ranks = built.layouts.first().map_or(0, Vec::len);
        ensure!(rank < ranks && placement.items.len() == ranks, "Muse rank {rank} of {ranks} placement ranks");
        let library = device.library;
        // Programs before any allocation: a missing export refuses here.
        let bound = device.run(|| RankPrograms::bind(programs, ranks))?;
        let core = CorePrograms::load(programs, device, built.projections.get(rank).context("Muse rank projections")?)?;
        let layers = load_rank(device, cfg, checkpoint, &built.layouts, rank)?;
        let kv = RankKv::allocate(device, cfg, &MuseCacheLayout::new(cfg, ranks)?, placement, rank, slots)?;
        let plan = shared::rope_tables(cfg, context)?;
        let recipe = *plan.recipes.first().context("Muse rotary plan has no recipe")?;
        let rotary = RotaryTables::load(device, programs, plan, "", u64::MAX, &[])?;
        let rope = rotary.get(recipe)?;
        let lane = Lane::allocate(device, &LanePlanes::with_waves(cfg, ranks, unit_rows, waves)?)?;
        let tp2 = (ranks == 2).then(|| -> Result<Tp2Ops<'a>> {
            Ok(Tp2Ops { add: library.bf16_f32_add()?, narrow: library.f32_bf16_narrow()? })
        }).transpose()?;
        let step = StepStorage::allocate(device, tables)?;
        let wire_grid = crate::shared::launch_grid::Fp8QuantizeGrid::new(device.run(|| library.sm_count())?, None)?;
        Ok(Self { device, programs: bound, core, layers, kv, lane, rope, _rotary: rotary, tp2, ranks, rank, wire_grid, step })
    }

    /// The device layer's view of this rank for one step on `stream`.
    pub(crate) fn device<'w>(&'w self, cfg: &'w MuseSparkConfig, stream: *mut c_void) -> RankDevice<'a, 'w> {
        RankDevice { library: self.device.library, cfg, programs: &self.programs, core: &self.core,
            layers: &self.layers, lane: &self.lane.views, rope: self.rope.buffers(), kv: &self.kv.layers,
            ranks: self.ranks, rank: self.rank, tp2: self.tp2.as_ref(), wire_grid: self.wire_grid, stream }
    }
}
