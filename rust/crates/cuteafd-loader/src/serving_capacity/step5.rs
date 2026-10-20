//! Exact storage layout for the current dense-summary Step-5 kernel ABI.
//! This is cache storage, not serving admission: executable workspace, graph,
//! transport and weight owners must contribute their actual reservations too.
use super::{CacheGeometryError, FamilyCacheGeometry, KvPlacement, RankCacheGeometry};
use crate::plan::families::step5::Config;

pub const PAGE_ROWS: u64 = 64;
pub const REGION_ROWS: u64 = 8;
pub const PROXY_DIM: u64 = 256;

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

/// Component offsets in one mark slot. Active storage is component-major:
/// component.offset * sequences, followed by contiguous request-sized records.
/// Marks are slot-major, so snapshot copies use explicit source/destination offsets.
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step5CacheLayout {
    pub ranks: Vec<RankState>,
    pub geometry: FamilyCacheGeometry,
    pub context: u64,
    pub mtp_hidden_rows: u64,
    pub ring: cuteafd_core::prefix::AppendRing,
    window: u64,
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

    fn ring(&mut self, layer: usize, mtp: bool, query_heads: usize, bytes: u64) -> Result<(), CacheGeometryError> {
        let key = self.reserve(bytes)?;
        let value = self.reserve(bytes)?;
        self.sliding.push(SlidingState { layer, mtp, query_heads, key, value });
        Ok(())
    }
}

impl Step5CacheLayout {
    /// BF16 K/V, BF16 pending K/Z, E4M3-byte dense summaries. MTP is owned by
    /// rank 0 with all heads; target K/V is head-split. No 256-row MiMo limits.
    pub fn new(cfg: &Config, ranks: usize, context: u64, mtp: usize, max_step: u64) -> Result<Self, CacheGeometryError> {
        Self::build(cfg, ranks, context, mtp, max_step, None)
    }

    /// The serving candidate uses the same validated ownership as its weights.
    /// This storage contract alone does not admit execution or serving.
    pub fn for_execution(cfg: &Config, ownership: &crate::plan::families::step5::ownership::ExecutionOwnership,
        context: u64, mtp: usize, max_step: u64) -> Result<Self, CacheGeometryError> {
        ownership.validate_config(cfg).map_err(|_| invalid("ownership/checkpoint geometry"))?;
        Self::build(cfg, ownership.ranks(), context, mtp, max_step, Some(ownership))
    }

    fn build(cfg: &Config, ranks: usize, context: u64, mtp: usize, max_step: u64,
        ownership: Option<&crate::plan::families::step5::ownership::ExecutionOwnership>) -> Result<Self, CacheGeometryError> {
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
        let ring = cuteafd_core::prefix::AppendRing::new(window, max_step, PAGE_ROWS)
            .ok_or_else(|| invalid("append-only ring step or extent"))?;
        // Preserve the old frontier while appending a complete uncommitted wave,
        // including the teacher shift through all predictor stages.
        let mtp_hidden_rows = if mtp == 0 { 0 } else {
            let history_window = window.checked_add(mtp as u64 + 1)
                .ok_or(CacheGeometryError::Overflow("Step-5 MTP history window"))?;
            cuteafd_core::prefix::AppendRing::new(history_window, max_step, PAGE_ROWS)
                .ok_or_else(|| invalid("MTP append-only history extent"))?.rows
        };
        let mut states = Vec::with_capacity(ranks);
        let mut costs = Vec::with_capacity(ranks);
        for rank in 0..ranks {
            let mut state = RankState {
                sliding: Vec::new(), regions: Vec::new(),
                length: ByteSpan { offset: 0, bytes: 0 },
                next_length: ByteSpan { offset: 0, bytes: 0 },
                mtp_lengths: Vec::new(), mtp_hidden: None,
                mtp_stage_hidden: Vec::new(), bytes: 0,
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
                let record = mul(&[groups as u64, cfg.dim as u64, 2])?;
                if cfg.full[layer] {
                    let plane = mul(&[PAGE_ROWS, record])?;
                    page_bytes = page_bytes.checked_add(mul(&[plane, 2])?)
                        .ok_or(CacheGeometryError::Overflow("Step-5 page layers"))?;
                    let pending_key = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let pending_z = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let next_key = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let next_z = state.reserve(REGION_ROWS * PROXY_DIM * 2)?;
                    let summaries = if ownership.is_some() || rank == 0 {
                        Some(state.reserve(mul(&[capacity, PROXY_DIM])?)?)
                    } else { None };
                    state.regions.push(RegionState { layer, query_heads: heads, page_bytes: plane,
                        pending_key, pending_z, next_key, next_z, summaries });
                } else {
                    state.ring(layer, false, heads, mul(&[ring.rows, record])?)?;
                }
            }
            if rank == mtp_owner && mtp > 0 {
                for stage in 0..mtp {
                    state.ring(cfg.layers + stage, true, cfg.heads,
                        mul(&[ring.rows, cfg.groups as u64, cfg.dim as u64, 2])?)?;
                    let length = state.reserve(8)?;
                    state.mtp_lengths.push(length);
                    let hidden = state.reserve(mul(&[mtp_hidden_rows, cfg.hidden as u64, 4])?)?;
                    state.mtp_stage_hidden.push(hidden);
                }
                state.mtp_hidden = Some(state.reserve(mul(&[mtp_hidden_rows, cfg.hidden as u64, 2])?)?);
            }
            costs.push(RankCacheGeometry {
                persistent_unit_bytes: page_bytes,
                active_state_per_sequence_bytes: state.bytes,
                retained_mark_bytes: state.bytes,
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
            context, mtp_hidden_rows, ring, window,
        })
    }

    pub fn query_heads(&self, rank: usize, layer: usize) -> Result<usize, CacheGeometryError> {
        let state = self.ranks.get(rank).ok_or_else(|| invalid("attention rank"))?;
        state.sliding.iter().find(|s| !s.mtp && s.layer == layer).map(|s| s.query_heads)
            .or_else(|| state.regions.iter().find(|s| s.layer == layer).map(|s| s.query_heads))
            .ok_or_else(|| invalid("attention layer is not owned on rank"))
    }

    /// Canonical mark spans exclude target ring slack and the inactive pending
    /// bank. MTP state is retained whole until its independent executor exists.
    /// Only complete summaries are visible; unfinished K/V stays in CSA pages.
    pub fn snapshot_spans(&self, rank: usize, frontier: u64) -> Result<Vec<ByteSpan>, CacheGeometryError> {
        Ok(self.snapshot_parts(rank, frontier)?.into_iter().map(|(component, offset, _, bytes)|
            ByteSpan { offset: component.offset + offset, bytes }).collect())
    }

    // (component, physical offset, canonical mark offset, bytes).
    fn snapshot_parts(&self, rank: usize, frontier: u64)
        -> Result<Vec<(ByteSpan, u64, u64, u64)>, CacheGeometryError> {
        if frontier > self.context { return Err(invalid("snapshot frontier outside admitted context")); }
        let state = self.ranks.get(rank).ok_or_else(|| invalid("snapshot rank"))?;
        let mut parts = vec![(state.length, 0, 0, state.length.bytes)];
        for ring in &state.sliding {
            for component in [ring.key, ring.value] {
                if ring.mtp {
                    parts.push((component, 0, 0, component.bytes));
                } else {
                    let row = component.bytes / self.ring.rows;
                    for run in self.ring.visible(frontier) {
                        parts.push((component, run.slot * row, run.offset * row, run.rows * row));
                    }
                }
            }
        }
        for region in &state.regions {
            for component in [region.pending_key, region.pending_z] {
                parts.push((component, 0, 0, component.bytes));
            }
            if let Some(summary) = region.summaries {
                let bytes = mul(&[frontier / REGION_ROWS, PROXY_DIM])?;
                if bytes > 0 { parts.push((summary, 0, 0, bytes)); }
            }
        }
        for component in state.mtp_lengths.iter().copied().chain(state.mtp_hidden)
            .chain(state.mtp_stage_hidden.iter().copied()) {
            parts.push((component, 0, 0, component.bytes));
        }
        Ok(parts)
    }

    /// Bind exact copies to active component-major request planes and slot-major
    /// marks. The kernel ABI has no sequence stride between a component's rows.
    pub fn snapshot_copies(&self, rank: usize, frontier: u64, sequences: u64, sequence: u64,
        slots: u64, slot: u64) -> Result<Vec<SnapshotCopy>, CacheGeometryError> {
        if sequences == 0 || sequence >= sequences || slot >= slots {
            return Err(invalid("snapshot sequence or mark slot"));
        }
        let state = self.ranks.get(rank).ok_or_else(|| invalid("snapshot rank"))?;
        let total_active = mul(&[state.bytes, sequences])?;
        let total_marks = mul(&[state.bytes, slots])?;
        self.snapshot_parts(rank, frontier)?.into_iter().map(|(component, offset, canonical, bytes)| {
            let active = mul(&[component.offset, sequences])?.checked_add(mul(&[sequence, component.bytes])?)
                .and_then(|at| at.checked_add(offset))
                .ok_or(CacheGeometryError::Overflow("Step-5 active copy offset"))?;
            let mark = mul(&[slot, state.bytes])?.checked_add(component.offset)
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
    /// current region; summary state is copied separately by snapshot_spans.
    pub fn csa_tail_bytes(&self, rank: usize, rows: u64) -> Result<u64, CacheGeometryError> {
        if rows > PAGE_ROWS { return Err(invalid("CSA tail longer than a page")); }
        let geometry = self.geometry.ranks.get(rank).ok_or_else(|| invalid("CSA tail rank"))?;
        mul(&[geometry.persistent_unit_bytes / PAGE_ROWS, rows])
    }

    /// Retention follows the shared slot policy. Its active-lane floor can
    /// exceed the nominal mark budget, so charge every resulting slot.
    pub fn prefix_arena_bytes(&self, lanes: u64, entries: u64, budget: u64)
        -> Result<(u64, Vec<u64>), CacheGeometryError> {
        let mark = self.ranks.iter().try_fold(0u64, |sum, rank| sum.checked_add(rank.bytes)
            .ok_or(CacheGeometryError::Overflow("Step-5 combined mark")))?;
        let slots = cuteafd_core::prefix::mark_slots_for(lanes, entries, mark, budget);
        let bytes = self.ranks.iter().map(|rank| mul(&[slots, rank.bytes])).collect::<Result<_, _>>()?;
        Ok((slots, bytes))
    }
}
