//! Exact storage layout for the current dense-summary Step-5 kernel ABI.
//! This is cache storage, not serving admission: executable workspace, graph,
//! transport and weight owners must contribute their actual reservations too.
use super::{CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry};
use crate::{families::qwen4::Qwen4KvCache, plan::families::step5::Config};

pub const PAGE_ROWS: u64 = 64;
pub const REGION_ROWS: u64 = 8;
pub const PROXY_DIM: u64 = 256;
pub const SUMMARY_PAGE_BYTES: u64 = PAGE_ROWS / REGION_ROWS * PROXY_DIM;

/// Bytes of completed summaries in the first `rows` token rows of a page.
/// The unfinished region is represented by the committed BF16 pending bank.
pub fn summary_prefix_bytes(rows: u64) -> u64 {
    debug_assert!(rows <= PAGE_ROWS);
    rows / REGION_ROWS * PROXY_DIM
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSpan {
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotCopy {
    pub active: ByteSpan,
    pub mark: ByteSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlidingState {
    pub layer: usize,
    pub mtp: bool,
    pub query_heads: usize,
    pub key: ByteSpan,
    pub value: ByteSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionState {
    pub layer: usize,
    pub query_heads: usize,
    /// One K or V plane's bytes per physical page on this layer's owner.
    pub page_bytes: u64,
    /// BF16 [8,256], including deterministic unused slots, on each owner.
    pub pending_key: ByteSpan,
    pub pending_z: ByteSpan,
    pub next_key: ByteSpan,
    pub next_z: ByteSpan,
    /// Published complete-region summaries, colocated with owned CSA attention.
    /// The diagnostic head-split layout retains its lead-rank-only table.
    pub summaries: Option<ByteSpan>,
}

impl RegionState {
    pub fn summary_page_bytes(&self) -> u64 {
        if self.summaries.is_some() { SUMMARY_PAGE_BYTES } else { 0 }
    }

    pub fn persistent_page_bytes(&self) -> u64 {
        2 * self.page_bytes + self.summary_page_bytes()
    }
}

/// Active storage is component-major: component.offset * sequences, followed
/// by contiguous request-sized records. Marks are slot-major, with independent
/// offsets for canonical committed windows (no proposal slack or CSA summaries).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankState {
    pub sliding: Vec<SlidingState>,
    pub regions: Vec<RegionState>,
    pub length: ByteSpan,
    pub next_length: ByteSpan,
    pub mtp_lengths: Vec<ByteSpan>,
    /// Target BF16 rows consumed by the first predictor.
    pub mtp_hidden: Option<ByteSpan>,
    /// Each predictor's own F32 residual history, never target-hidden aliases.
    pub mtp_stage_hidden: Vec<ByteSpan>,
    pub bytes: u64,
    pub mark_bytes: u64,
    mark_components: Vec<(ByteSpan, ByteSpan)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step5CacheLayout {
    pub ranks: Vec<RankState>,
    pub geometry: FamilyCacheGeometry,
    pub context: u64,
    pub kv_format: Qwen4KvCache,
    pub mtp_hidden_rows: u64,
    /// Target SWA rings: [`swa_ring`] of one prefill unit.
    pub ring: cuteafd_core::prefix::AppendRing,
    /// Predictor SWA rings: [`swa_ring`] of one prefill unit too. A predictor
    /// appends at most one unit per call (`catch_up` chunks by unit capacity,
    /// the draft appends `depth + 1` rows), never the whole wave.
    pub mtp_ring: cuteafd_core::prefix::AppendRing,
    /// Predictor hidden history: `window + stages + 1` (the teacher shift
    /// through every stage) plus one whole step, page aligned. Target hidden
    /// rows are published for a complete prefill step at once, so this ring
    /// keeps the step rows; `None` without MTP.
    pub mtp_history: Option<cuteafd_core::prefix::AppendRing>,
    /// Rows one step may carry (the prefill reserve): staging and step admission.
    pub step_rows: u64,
    window: u64,
}

/// The SWA ring: `window + rows a kernel reads in one call`, page aligned.
/// A target unit's K/V are staged into the ring before its attention reads
/// them (`Step::stage`, then `Step::attend`, in order on one lane stream), so
/// the ring holds the window plus the rows of one unit. Units of a sequence
/// stage, attend and commit in order on that stream, so a whole multi-unit
/// step never occupies the ring at once: sizing by the prefill reserve only
/// wastes memory. Predictors run one unit per call for the same reason. The
/// cache layout, staging, step metadata (ring page tables) and the planner
/// all read [`cuteafd_core::prefix::swa_ring`].
pub fn swa_ring(window: u64, unit_rows: u64) -> Option<cuteafd_core::prefix::AppendRing> {
    cuteafd_core::prefix::swa_ring(window, unit_rows, PAGE_ROWS)
}

fn invalid(what: &'static str) -> CacheGeometryError {
    CacheGeometryError::Unsupported { family: "step5", what }
}

fn mul(values: &[u64]) -> Result<u64, CacheGeometryError> {
    values.iter().try_fold(1u64, |a, b| a.checked_mul(*b)
        .ok_or(CacheGeometryError::Overflow("Step-5 cache storage")))
}

impl RankState {
    fn reserve(&mut self, bytes: u64) -> Result<ByteSpan, CacheGeometryError> {
        let span = ByteSpan { offset: self.bytes, bytes };
        self.bytes = self.bytes.checked_add(bytes).and_then(|end| end.checked_add(255))
            .map(|end| end & !255)
            .ok_or(CacheGeometryError::Overflow("Step-5 sequence state"))?;
        Ok(span)
    }

    fn reserve_mark(&mut self, active: ByteSpan, bytes: u64) -> Result<(), CacheGeometryError> {
        let mark = ByteSpan { offset: self.mark_bytes, bytes };
        self.mark_bytes = self.mark_bytes.checked_add(bytes).and_then(|end| end.checked_add(255))
            .map(|end| end & !255)
            .ok_or(CacheGeometryError::Overflow("Step-5 canonical mark"))?;
        self.mark_components.push((active, mark));
        Ok(())
    }

    pub fn mark_component(&self, active: ByteSpan) -> Result<ByteSpan, CacheGeometryError> {
        self.mark_components.iter().find_map(|&(component, mark)| (component == active).then_some(mark))
            .ok_or_else(|| invalid("component is not positional mark state"))
    }

    fn ring(&mut self, layer: usize, mtp: bool, query_heads: usize, bytes: u64) -> Result<(), CacheGeometryError> {
        let key = self.reserve(bytes)?;
        let value = self.reserve(bytes)?;
        self.sliding.push(SlidingState { layer, mtp, query_heads, key, value });
        Ok(())
    }
}

impl Step5CacheLayout {
    /// Selected K/V, BF16 pending K/Z, E4M3 dense summaries; recurrence stays F32.
    /// `unit_rows` sizes every SWA ring ([`swa_ring`]: target and predictor);
    /// `step_rows` (at least one unit) sizes staging and the target hidden
    /// history, which one step publishes for later steps.
    pub fn new(cfg: &Config, ranks: usize, context: u64, mtp: usize, unit_rows: u64, step_rows: u64,
        kv_format: Qwen4KvCache) -> Result<Self, CacheGeometryError> {
        Self::build(cfg, ranks, context, mtp, unit_rows, step_rows, kv_format, None)
    }

    /// The serving candidate uses the same validated ownership as its weights.
    /// This storage contract alone does not admit execution or serving.
    pub fn for_execution(cfg: &Config, ownership: &crate::plan::families::step5::ownership::ExecutionOwnership,
        context: u64, mtp: usize, unit_rows: u64, step_rows: u64, kv_format: Qwen4KvCache) -> Result<Self, CacheGeometryError> {
        ownership.validate_config(cfg).map_err(|_| invalid("ownership/checkpoint geometry"))?;
        Self::build(cfg, ownership.ranks(), context, mtp, unit_rows, step_rows, kv_format, Some(ownership))
    }

    #[allow(clippy::too_many_arguments)]
    fn build(cfg: &Config, ranks: usize, context: u64, mtp: usize, unit_rows: u64, step_rows: u64,
        kv_format: Qwen4KvCache, ownership: Option<&crate::plan::families::step5::ownership::ExecutionOwnership>) -> Result<Self, CacheGeometryError> {
        if unit_rows == 0 || unit_rows > step_rows {
            return Err(invalid("prefill unit outside the step rows"));
        }
        if ![1, 2].contains(&ranks) || cfg.groups % ranks != 0 || cfg.heads % ranks != 0
            || (cfg.window, cfg.groups, cfg.dim, cfg.hidden) != (512, 4, 192, 4096)
            || cfg.full.len() != cfg.layers + cfg.mtp || cfg.layers != 92
            || mtp > cfg.mtp || cfg.full[cfg.layers..].iter().any(|full| *full)
        {
            return Err(invalid("native coordinator or MTP geometry"));
        }
        if context == 0 || context > cfg.max_context {
            return Err(invalid("context outside checkpoint extent"));
        }
        let capacity = context.div_ceil(REGION_ROWS);
        let mtp_owner = ownership.map_or(0, |o| usize::from(o.execution().spec().head_gpu));
        let window = cfg.window as u64;
        let ring = swa_ring(window, unit_rows).ok_or_else(|| invalid("append-only ring step or extent"))?;
        // A predictor appends at most one unit (a catch-up chunk or the
        // triangular draft), so its rings follow the target's: window + unit.
        let mtp_ring = swa_ring(window, unit_rows)
            .ok_or_else(|| invalid("predictor ring step or extent"))?;
        // Preserve the old frontier while appending a complete uncommitted wave,
        // including the teacher shift through all predictor stages. Target
        // hidden rows are published for a whole step at once, so this ring
        // keeps `step_rows`, unlike the predictor K/V rings above.
        let mtp_history = if mtp == 0 { None } else {
            let history_window = window.checked_add(mtp as u64 + 1)
                .ok_or(CacheGeometryError::Overflow("Step-5 MTP history window"))?;
            Some(cuteafd_core::prefix::swa_ring(history_window, step_rows, PAGE_ROWS)
                .ok_or_else(|| invalid("MTP append-only history extent"))?)
        };
        let mtp_hidden_rows = mtp_history.map_or(0, |ring| ring.rows);
        let mut states = Vec::with_capacity(ranks);
        let mut costs = Vec::with_capacity(ranks);
        for rank in 0..ranks {
            let mut state = RankState {
                sliding: Vec::new(), regions: Vec::new(),
                length: ByteSpan { offset: 0, bytes: 0 },
                next_length: ByteSpan { offset: 0, bytes: 0 },
                mtp_lengths: Vec::new(), mtp_hidden: None,
                mtp_stage_hidden: Vec::new(), bytes: 0, mark_bytes: 0, mark_components: Vec::new(),
            };
            state.length = state.reserve(8)?;
            state.next_length = state.reserve(8)?;
            let mut page_bytes = 0u64;
            for layer in 0..cfg.layers {
                let (heads, groups) = if let Some(ownership) = ownership {
                    let local = ownership.layer(layer, rank).map_err(|_| invalid("layer owner"))?;
                    (local.query_heads, local.kv_heads)
                } else { (cfg.heads / ranks, cfg.groups / ranks) };
                if heads == 0 { continue; }
                let record = kv_format.plane_record_bytes(groups, cfg.dim) as u64;
                if cfg.full[layer] {
                    let plane = mul(&[PAGE_ROWS, record])?;
                    let pending_key = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let pending_z = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let next_key = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let next_z = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let summaries = if ownership.is_some() || rank == 0 {
                        Some(state.reserve(mul(&[capacity, PROXY_DIM])?)?)
                    } else { None };
                    let region = RegionState { layer, query_heads: heads, page_bytes: plane,
                        pending_key, pending_z, next_key, next_z, summaries };
                    page_bytes = page_bytes.checked_add(region.persistent_page_bytes())
                        .ok_or(CacheGeometryError::Overflow("Step-5 page layers"))?;
                    state.regions.push(region);
                } else {
                    state.ring(layer, false, heads, mul(&[ring.rows, record])?)?;
                }
            }
            if rank == mtp_owner && mtp > 0 {
                for stage in 0..mtp {
                    state.ring(cfg.layers + stage, true, cfg.heads,
                        mul(&[mtp_ring.rows, kv_format.plane_record_bytes(cfg.groups, cfg.dim) as u64])?)?;
                    let length = state.reserve(8)?;
                    state.mtp_lengths.push(length);
                    let hidden = state.reserve(mul(&[mtp_hidden_rows, cfg.hidden as u64, 4])?)?;
                    state.mtp_stage_hidden.push(hidden);
                }
                state.mtp_hidden = Some(state.reserve(mul(&[mtp_hidden_rows, cfg.hidden as u64, 2])?)?);
            }
            // Fixed mark slots retain only the committed positional state. The
            // context-scaled complete regions belong to the shared CSA pages.
            state.reserve_mark(state.length, state.length.bytes)?;
            for component in state.mtp_lengths.clone() {
                state.reserve_mark(component, component.bytes)?;
            }
            for sliding in state.sliding.clone() {
                let layout = if sliding.mtp { mtp_ring } else { ring };
                for component in [sliding.key, sliding.value] {
                    state.reserve_mark(component, mul(&[component.bytes / layout.rows, layout.window])?)?;
                }
            }
            for region in state.regions.clone() {
                for component in [region.pending_key, region.pending_z] {
                    state.reserve_mark(component, component.bytes)?;
                }
            }
            if let Some(history) = mtp_history {
                for component in state.mtp_hidden.into_iter().chain(state.mtp_stage_hidden.clone()) {
                    state.reserve_mark(component, mul(&[component.bytes / history.rows, history.window])?)?;
                }
            }
            costs.push(RankCacheGeometry {
                persistent_unit_bytes: page_bytes,
                active_state_per_sequence_bytes: state.bytes,
                retained_mark_bytes: state.mark_bytes,
                // Staged proposals belong to the actual lane workspace, not committed state.
                ..Default::default()
            });
            states.push(state);
        }
        Ok(Self {
            ranks: states,
            geometry: FamilyCacheGeometry {
                logical_unit_rows: PAGE_ROWS,
                placement: if ranks == 1 { KvPlacement::SingleDevice }
                    else if ownership.is_some() { KvPlacement::PartitionedLayers }
                    else { KvPlacement::PartitionedHeads },
                ranks: costs,
            },
            context, kv_format, mtp_hidden_rows, ring, mtp_ring, mtp_history, step_rows, window,
        })
    }

    pub fn query_heads(&self, rank: usize, layer: usize) -> Result<usize, CacheGeometryError> {
        let state = self.ranks.get(rank).ok_or_else(|| invalid("attention rank"))?;
        state.sliding.iter().find(|s| !s.mtp && s.layer == layer).map(|s| s.query_heads)
            .or_else(|| state.regions.iter().find(|s| s.layer == layer).map(|s| s.query_heads))
            .ok_or_else(|| invalid("attention layer is not owned on rank"))
    }

    /// One compact producer pair per target lane, reused after stream-ordered
    /// copies into retained storage. Predictor staging is separate.
    pub fn target_pack_planes(&self, rank: usize) -> Result<[usize; 2], CacheGeometryError> {
        let state = self.ranks.get(rank).ok_or_else(|| invalid("attention rank"))?;
        let heads = state.sliding.iter().filter(|s| !s.mtp).map(|s| s.query_heads)
            .chain(state.regions.iter().map(|s| s.query_heads)).max().unwrap_or(0);
        let bytes = if self.kv_format == Qwen4KvCache::Fp8 && heads > 0 {
            mul(&[self.ring.step, self.kv_format.plane_record_bytes(heads / 16, 192) as u64])?
        } else { 0 };
        let bytes = usize::try_from(bytes).map_err(|_| invalid("producer staging extent"))?;
        Ok([bytes; 2])
    }

    /// Canonical marks contain only committed rows, never proposal slack.
    /// Predictor frontiers are actual stage-local lengths, not inferred target shifts.
    pub fn snapshot_spans(&self, rank: usize, frontier: u64, stages: &[u64]) -> Result<Vec<ByteSpan>, CacheGeometryError> {
        Ok(self.snapshot_parts(rank, frontier, stages)?.into_iter().map(|(component, offset, _, bytes)|
            ByteSpan { offset: component.offset + offset, bytes }).collect())
    }

    // (component, physical offset, canonical mark offset, bytes).
    fn snapshot_parts(&self, rank: usize, frontier: u64, stages: &[u64])
        -> Result<Vec<(ByteSpan, u64, u64, u64)>, CacheGeometryError> {
        if frontier > self.context { return Err(invalid("snapshot frontier outside admitted context")); }
        let state = self.ranks.get(rank).ok_or_else(|| invalid("snapshot rank"))?;
        if stages.len() != state.mtp_lengths.len() || stages.iter().enumerate().any(|(stage, &end)|
            end > frontier.saturating_sub(stage as u64 + 1)
                || (stage > 0 && end > stages[stage - 1])) {
            return Err(invalid("snapshot predictor frontier is not committed"));
        }
        let mut parts = vec![(state.length, 0, 0, state.length.bytes)];
        for ring in &state.sliding {
            let end = if ring.mtp { *stages.get(ring.layer - 92)
                .ok_or_else(|| invalid("snapshot predictor stage"))? } else { frontier };
            let layout = if ring.mtp { self.mtp_ring } else { self.ring };
            for component in [ring.key, ring.value] {
                let row = component.bytes / layout.rows;
                for run in layout.visible(end) {
                    parts.push((component, run.slot * row, run.offset * row, run.rows * row));
                }
            }
        }
        for region in &state.regions {
            for component in [region.pending_key, region.pending_z] {
                parts.push((component, 0, 0, component.bytes));
            }
        }
        for &component in &state.mtp_lengths {
            parts.push((component, 0, 0, component.bytes));
        }
        if let Some(target) = state.mtp_hidden {
            let history = self.mtp_history.ok_or_else(|| invalid("MTP hidden history ring"))?;
            for (component, end) in std::iter::once((target, frontier))
                .chain(state.mtp_stage_hidden.iter().copied().zip(stages.iter().copied())) {
                let row = component.bytes / history.rows;
                for run in history.visible(end) {
                    parts.push((component, run.slot * row, run.offset * row, run.rows * row));
                }
            }
        }
        Ok(parts)
    }

    /// Bind exact copies to active component-major request planes and slot-major
    /// marks. The kernel ABI has no sequence stride between a component's rows.
    pub fn snapshot_copies(&self, rank: usize, frontier: u64, stages: &[u64], sequences: u64, sequence: u64,
        slots: u64, slot: u64) -> Result<Vec<SnapshotCopy>, CacheGeometryError> {
        if sequences == 0 || sequence >= sequences || slot >= slots {
            return Err(invalid("snapshot sequence or mark slot"));
        }
        let state = self.ranks.get(rank).ok_or_else(|| invalid("snapshot rank"))?;
        let total_active = mul(&[state.bytes, sequences])?;
        let total_marks = mul(&[state.mark_bytes, slots])?;
        self.snapshot_parts(rank, frontier, stages)?.into_iter().map(|(component, offset, canonical, bytes)| {
            let active = mul(&[component.offset, sequences])?.checked_add(mul(&[sequence, component.bytes])?)
                .and_then(|at| at.checked_add(offset))
                .ok_or(CacheGeometryError::Overflow("Step-5 active copy offset"))?;
            let retained = state.mark_component(component)?;
            if canonical.checked_add(bytes).is_none_or(|end| end > retained.bytes) {
                return Err(invalid("snapshot canonical component extent"));
            }
            let mark = mul(&[slot, state.mark_bytes])?.checked_add(retained.offset)
                .and_then(|at| at.checked_add(canonical))
                .ok_or(CacheGeometryError::Overflow("Step-5 mark copy offset"))?;
            if active.checked_add(bytes).is_none_or(|end| end > total_active)
                || mark.checked_add(bytes).is_none_or(|end| end > total_marks) {
                return Err(invalid("snapshot byte extent"));
            }
            Ok(SnapshotCopy { active: ByteSpan { offset: active, bytes },
                mark: ByteSpan { offset: mark, bytes } })
        }).collect()
    }

    /// Pending buffers describe only the current frontier, even though rings
    /// retain extra rows. Capture exact frontiers, not reconstructed history.
    pub fn capture_reach(&self) -> u64 { 0 }

    pub fn window(&self) -> u64 { self.window }

    /// Partial CSA pages copy every retained K/V row, including the causal
    /// current region; summary state includes completed regions only.
    pub fn csa_tail_bytes(&self, rank: usize, rows: u64) -> Result<u64, CacheGeometryError> {
        if rows > PAGE_ROWS { return Err(invalid("CSA tail longer than a page")); }
        let geometry = self.geometry.ranks.get(rank).ok_or_else(|| invalid("CSA tail rank"))?;
        let state = &self.ranks[rank];
        let summary = state.regions.iter().filter(|r| r.summaries.is_some()).count() as u64;
        let kv = geometry.persistent_unit_bytes - summary * SUMMARY_PAGE_BYTES;
        Ok(mul(&[kv / PAGE_ROWS, rows])? + summary * summary_prefix_bytes(rows))
    }

    /// Retention follows the shared slot policy. Its active-lane floor can
    /// exceed the nominal mark budget, so charge every resulting slot.
    pub fn prefix_arena_bytes(&self, lanes: u64, entries: u64, budget: u64)
        -> Result<(u64, Vec<u64>), CacheGeometryError> {
        let mark = self.ranks.iter().try_fold(0u64, |sum, rank| sum.checked_add(rank.mark_bytes)
            .ok_or(CacheGeometryError::Overflow("Step-5 combined mark")))?;
        let slots = cuteafd_core::prefix::mark_slots_for(lanes, entries, mark, budget);
        let bytes = self.ranks.iter().map(|rank| mul(&[slots, rank.mark_bytes])).collect::<Result<_, _>>()?;
        Ok((slots, bytes))
    }
}
