//! CPU allocation layouts shared by Kimi checkpoint planning and native owners.
use super::config::{KimiAttention, KimiK3Config};
use crate::placement::{Demand, GraphSet, Lifetime};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeScratch {
    pub kda: usize,
    pub mla: [usize; 3],
}

impl NativeScratch {
    pub fn combine(kda: [usize; 2], mla: impl IntoIterator<Item = [usize; 3]>) -> Result<Self> {
        let mut scratch = Self { kda: *kda.iter().max().unwrap(), mla: [0; 3] };
        for requirements in mla {
            for (current, required) in scratch.mla.iter_mut().zip(requirements) {
                *current = (*current).max(required);
            }
        }
        scratch.bytes()?;
        Ok(scratch)
    }

    pub fn bytes(&self) -> Result<usize> {
        self.mla.iter().try_fold(self.kda, |sum, bytes| sum.checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("Kimi native scratch extent overflow")))
    }
}

/// Most rows one kernel launch (one prefill unit) handles.
pub const UNIT_ROWS_MAX: usize = 4096;
/// Largest shared live span: three page-aligned units of 2752 rows.
/// Each lane allocates its exact admitted span, never this validation ceiling.
pub const LANE_ROWS_MAX: usize = (crate::serving_capacity::prefill_waves::RESERVE_ROWS + 64) as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLaneLayout {
    /// Lane rows: the call's row planes, each unit owning its rows of them.
    pub rows: usize,
    /// Unit rows: hook-local scratch (the phase arena and K6 stages)
    /// serves one unit's kernel launch at a time.
    pub unit: usize,
    pub max_context: usize,
    pub operands: BTreeMap<&'static str, usize>,
    pub arena_offsets: BTreeMap<&'static str, usize>,
    pub arena_bytes: usize,
}

pub const SCRATCH_PHASES: [&[&str]; 3] = [
    &["KDA normalized", "KDA scratch"],
    &["MLA query", "MLA q residual", "MLA attention", "MLA LSE",
        "MLA producer scratch", "MLA attention scratch", "MLA output scratch"],
    &["gate-up", "activated"],
];
/// Live operands at consecutive rank-local kernel boundaries. Projection scratch
/// is a separate owner; all these phases execute on the captured lane stream.
pub const CORE_PHASES: &[&[&str]] = &[
    &["K6 q", "K6 k", "K6 v", "K6 gate"],
    &["K6 q", "K6 k", "K6 v", "K6 gate", "K6 normalized", "K6 KDA scratch"],
    &["K6 normalized"],
    &["K6 q_a", "K6 kv_a", "MLA q residual", "K6 MLA scratch"],
    &["K6 q_b", "MLA q residual", "MLA query"],
    &["MLA query", "MLA attention", "MLA LSE", "MLA attention scratch"],
    &["MLA attention", "K6 gate", "K6 normalized"],
    &["K6 FFN gate", "K6 FFN up"],
    &["K6 FFN gate", "K6 FFN up", "gate-up"],
    &["gate-up", "activated"],
    &["activated"],
];
pub const CORE_MEET_PLANES: [&str; 3] = ["K6 attention partial", "K6 FFN partial", "K6 reduced"];
pub const ARENA_ALIGNMENT: usize = 1024;
pub const ATTENTION_METADATA: [&str; 6] = ["KDA slots", "KDA first", "MLA slots",
    "MLA page table", "MLA lengths", "MLA query offsets"];
/// Consumed entirely on the lane stream inside route/latent_up. Neither peer
/// exchange nor a pending Spark wave retains these intermediates.
pub const HOOK_SCRATCH: [&str; 3] = ["router logits", "latent input", "latent normalized"];

impl NativeLaneLayout {
    fn metadata_banks_for(rows: usize, unit: usize) -> usize {
        if rows <= 64 { return 1; }
        rows.div_ceil(unit)
    }

    pub fn metadata_banks(&self) -> usize { Self::metadata_banks_for(self.rows, self.unit) }

    /// Each bank has the decode-capable extent; unit tails can be <=64 rows.
    pub fn metadata_extent(&self, name: &str, bank: usize) -> Result<(usize, usize)> {
        ensure!(ATTENTION_METADATA.contains(&name) && bank < self.metadata_banks(),
            "Kimi attention metadata bank outside admission");
        let bytes = self.operands.get(name).context("Kimi metadata plane absent")?;
        ensure!(bytes % self.metadata_banks() == 0, "Kimi metadata bank stride");
        let stride = bytes / self.metadata_banks();
        ensure!(stride >= 16 && stride % 16 == 0, "Kimi metadata bank alignment");
        Ok((bank * stride, stride))
    }

    /// `rows` lane rows hold units of at most `unit` rows; one-unit lanes pass `unit == rows`.
    pub fn new(cfg: &KimiK3Config, rows: usize, unit: usize, max_context: usize,
        scratch: NativeScratch) -> Result<Self> {
        cfg.release_geometry()?;
        ensure!((1..=LANE_ROWS_MAX).contains(&rows) && (1..=UNIT_ROWS_MAX.min(rows)).contains(&unit)
            && (1..=1_048_576).contains(&max_context), "Kimi lane row/unit/context capacity");
        let mut operands = BTreeMap::new();
        let mut plane = |name, shape: &[usize]| -> Result<()> {
            let bytes = shape.iter().try_fold(1usize, |n, size| n.checked_mul(*size)
                .ok_or_else(|| anyhow::anyhow!("Kimi lane {name} extent overflow")))?;
            ensure!(operands.insert(name, bytes.max(16)).is_none(), "duplicate Kimi lane operand");
            Ok(())
        };
        let hidden = [rows, cfg.hidden, 2];
        plane("residual arena", &[9, rows, cfg.hidden, 2])?;
        for name in ["delta", "normalized", "attention output", "ffn output", "shared output", "routed output", "combined output"] {
            plane(name, &hidden)?;
        }
        let intermediate = (cfg.dense_intermediate / 2).max(cfg.moe_intermediate * cfg.shared_experts / 2);
        plane("gate-up", &[unit, 2, intermediate, 2])?;
        plane("activated", &[unit, intermediate, 2])?;
        plane("expert combined", &[rows, cfg.latent, 2])?;
        for name in ["latent input", "latent normalized"] { plane(name, &[unit, cfg.latent, 2])?; }
        plane("router logits", &[unit, cfg.experts, 4])?;
        for name in ["route ids", "route weights"] { plane(name, &[rows, cfg.topk, 4])?; }
        plane("expert wire", &[rows, (3584 + 3584 / 32)])?;
        // Phase-arena operands serve one unit's rows. Metadata has one bank
        // per possible call unit (sized below), never rewritten per layer.
        plane("KDA normalized", &[unit, 48, 128, 2])?;
        for name in ["KDA slots", "KDA first"] { plane(name, &[unit, 4])?; }
        plane("MLA slots", &[unit, 8])?;
        plane("MLA query", &[unit, cfg.heads, 576, 2])?;
        plane("MLA q residual", &[unit, cfg.q_lora_rank, 2])?;
        plane("MLA attention", &[unit, cfg.heads, 512, 2])?;
        plane("MLA LSE", &[unit, cfg.heads, 4])?;
        // Decode/verify permits one request per row, at most the 64-row
        // specialization. Wide prefill is one sequence. Cache pages live outside.
        let requests = unit.min(64);
        plane("MLA page table", &[requests, max_context.div_ceil(64), 4])?;
        plane("MLA lengths", &[requests, 4])?;
        plane("MLA query offsets", &[requests + 1, 4])?;
        plane("KDA scratch", &[scratch.kda])?;
        for (name, bytes) in ["MLA producer scratch", "MLA attention scratch", "MLA output scratch"]
            .into_iter().zip(scratch.mla) { plane(name, &[bytes])?; }
        // A call stages each unit once. Metadata survives all its layers, so
        // in-flight units require disjoint banks, including a <=64-row tail.
        // Native kernels require every bank pointer, not just its base, aligned.
        let banks = Self::metadata_banks_for(rows, unit);
        for name in ATTENTION_METADATA {
            let bytes = operands.get_mut(name).expect("attention metadata plane");
            let stride = bytes.checked_add(15).context("Kimi metadata alignment overflow")? / 16 * 16;
            *bytes = stride.checked_mul(banks).context("Kimi metadata banks overflow")?;
        }
        let mut arena_offsets = BTreeMap::new();
        let mut arena_bytes = 0;
        for phase in SCRATCH_PHASES {
            let mut end = 0usize;
            for &name in phase {
                let offset = end.checked_add(ARENA_ALIGNMENT - 1)
                    .map(|n| n / ARENA_ALIGNMENT * ARENA_ALIGNMENT)
                    .ok_or_else(|| anyhow::anyhow!("Kimi phase alignment overflow"))?;
                ensure!(arena_offsets.insert(name, offset).is_none(), "duplicate Kimi arena operand");
                end = offset.checked_add(operands[name])
                    .ok_or_else(|| anyhow::anyhow!("Kimi phase extent overflow"))?;
            }
            arena_bytes = arena_bytes.max(end);
        }
        let layout = Self { rows, unit, max_context, operands, arena_offsets, arena_bytes };
        layout.bytes()?;
        Ok(layout)
    }

    /// Preserve every BF16 offset. Place transient K6 operands into holes whose
    /// previous reader has completed on the same lane stream. Peer meet outputs
    /// remain separate allocations because the exchange stream reads them.
    pub fn with_core_stages(mut self, stages: BTreeMap<&'static str, usize>) -> Result<Self> {
        let mut transient = stages.iter().filter(|(name, _)| !CORE_MEET_PLANES.contains(name))
            .map(|(&name, &bytes)| (name, bytes)).collect::<Vec<_>>();
        transient.sort_by_key(|&(name, bytes)| (std::cmp::Reverse(bytes), name));
        ensure!(stages.keys().all(|name| !self.operands.contains_key(name)), "duplicate Kimi core arena operand");
        self.operands.extend(stages);
        let phases = self.core_phases();
        for (name, bytes) in transient {
            let conflicts = phases.iter().map(Vec::as_slice).chain(SCRATCH_PHASES)
                .filter(|phase| phase.contains(&name)).flat_map(|phase| phase.iter().copied())
                .filter(|other| *other != name && self.arena_offsets.contains_key(other))
                .collect::<std::collections::BTreeSet<_>>();
            ensure!(CORE_PHASES.iter().any(|phase| phase.contains(&name)), "Kimi stage lacks a live phase: {name}");
            let mut candidates = vec![0];
            for other in &conflicts {
                candidates.push(self.arena_offsets[other].checked_add(self.operands[other])
                    .context("Kimi core arena candidate overflow")?);
            }
            candidates.sort_unstable();
            let mut selected = None;
            for start in candidates {
                let start = start.checked_add(ARENA_ALIGNMENT - 1)
                    .map(|n| n / ARENA_ALIGNMENT * ARENA_ALIGNMENT).context("Kimi core arena alignment overflow")?;
                let end = start.checked_add(bytes).context("Kimi core arena extent overflow")?;
                if conflicts.iter().all(|other| end <= self.arena_offsets[other]
                    || start >= self.arena_offsets[other] + self.operands[other]) {
                    selected = Some((start, end)); break;
                }
            }
            let (start, end) = selected.context("Kimi core arena placement absent")?;
            self.arena_offsets.insert(name, start);
            self.arena_bytes = self.arena_bytes.max(end);
        }
        self.bytes()?;
        Ok(self)
    }

    /// Resolve optional larger split scratch to the fused scratch fallback used
    /// by the runtime. Every listed operand is live through that kernel boundary.
    pub fn core_phases(&self) -> Vec<Vec<&'static str>> {
        CORE_PHASES.iter().map(|phase| phase.iter().map(|&name| match name {
            "K6 KDA scratch" if !self.operands.contains_key(name) => "KDA scratch",
            "K6 MLA scratch" if !self.operands.contains_key(name) => "MLA producer scratch",
            _ => name,
        }).collect()).collect()
    }

    pub fn bytes(&self) -> Result<usize> {
        self.operands.iter().filter(|(name, _)| !self.arena_offsets.contains_key(*name))
            .try_fold(self.arena_bytes, |sum, (_, bytes)| sum.checked_add(*bytes).context("Kimi lane extent overflow"))
    }
}

impl NativeScratch {
    pub fn from_manifest(manifest: &serde_json::Value) -> Result<Self> {
        let scratch = |name: &str| -> Result<usize> {
            let found = manifest["programs"].as_array().context("Kimi program inventory missing")?
                .iter().filter(|p| p["name"] == name).collect::<Vec<_>>();
            ensure!(found.len() == 1, "Kimi requires one {name} scratch inventory");
            usize::try_from(found[0]["scratch_bytes_at_capacity"]["scratch"].as_u64()
                .with_context(|| format!("Kimi missing {name} scratch extent"))?).context("Kimi scratch overflow")
        };
        let mut mla = Vec::new();
        for suffix in ["", "_h96"] {
            for (mode, rows) in [("decode", 64), ("verify", 64), ("extend", 4096)] {
                mla.push([scratch(&format!("kimik3_mla_producer{suffix}_m{rows}"))?,
                    scratch(&format!("kimik3_dense_mla_{mode}{suffix}_m{rows}"))?,
                    scratch(&format!("kimik3_mla_output{suffix}_m{rows}"))?]);
            }
        }
        Self::combine([scratch("kimik3_kda_m64")?, scratch("kimik3_kda_m4096")?], mla)
    }
}

pub const GRAPH_RESERVE: &str = "Kimi startup graph reserve";
pub const GRAPH_ROWS: [usize; 5] = [1, 4, 16, 64, 4096];

/// Speculative verify rows replay at their exact width (no padding: every
/// row writes KDA replay records and MLA cache slots). `verify` is the
/// drafter's block (anchor + drafts), 0 without speculation.
pub fn verify_graph_rows(verify: usize, capacity: usize) -> std::ops::RangeInclusive<usize> {
    2..=verify.min(64).min(capacity)
}

/// Every captured row count: the plain buckets plus each exact verify width.
/// Plain attention (and multi-sequence KDA) keys stay on the plain buckets; the
/// added widths capture speculative single-sequence attention, residual and FFN.
pub fn graph_rows(verify: usize, capacity: usize) -> Vec<usize> {
    let mut rows = GRAPH_ROWS.into_iter().chain(verify_graph_rows(verify, capacity))
        .filter(|&r| r <= capacity).collect::<Vec<_>>();
    rows.sort_unstable();
    rows.dedup();
    rows
}

pub fn graphs(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    resident: impl Fn(usize, usize) -> bool) -> Result<GraphSet> {
    ensure!((1..=2).contains(&lanes) && (1..=UNIT_ROWS_MAX).contains(&capacity)
        && (1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots) && verify <= 64, "Kimi graph geometry");
    let mut counts = [0; 2];
    for (rank, count) in counts.iter_mut().enumerate() {
        for rows in graph_rows(verify, capacity) {
            *count += (3 * cfg.layers + usize::from(rank == 1)) as u64;
            for layer in 0..cfg.layers {
                if resident(rank, layer) {
                    if GRAPH_ROWS.contains(&rows) {
                        *count += 1 + u64::from(cfg.attention[layer] == KimiAttention::Kda
                            && (2..=slots).contains(&rows));
                    }
                    *count += u64::from(verify_graph_rows(verify, capacity).contains(&rows));
                }
            }
        }
        *count *= lanes as u64;
    }
    Ok(GraphSet::new(&counts, 256 << 10, 10, 64 << 20,
        counts.iter().sum(), Lifetime::Startup))
}

pub fn graph_demand(set: &GraphSet, rank: usize) -> Demand {
    let mut demand = set.demand(rank);
    demand.group = GRAPH_RESERVE.into();
    demand.basis = Basis::Estimated;
    demand
}

pub fn kda_demands(cfg: &KimiK3Config, slots: usize, marks: usize, lanes: usize,
    rank: u8) -> Result<[Demand; 4]> {
    kda_demands_for_layers(cfg.kda_layers(), slots, marks, lanes, rank)
}

pub fn kda_demands_for_layers(layers: usize, slots: usize, marks: usize, lanes: usize,
    rank: u8) -> Result<[Demand; 4]> {
    ensure!((1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots) && (1..=2).contains(&lanes)
        && rank < 2, "Kimi KDA state geometry");
    let state = 48 * 128 * 128 * 4 + 3 * 3 * 6144 * 2;
    let replay = 64 * 48 * (3 * 128 * 4 + 4 + 3 * 128 * 2);
    Ok([(Category::Kv, "Kimi KDA active state", layers * slots * state),
        (Category::Workspace, "Kimi KDA replay", layers * lanes * replay),
        (Category::Prefix, "Kimi exact KDA marks", layers * marks * state),
        (Category::Workspace, "Kimi KDA commit tables", lanes * slots * 3 * 4)]
        .map(|(category, group, bytes)| Demand::new(rank, category, group, bytes as u64, Basis::Exact)))
}

pub fn io_extents(vocab: usize, rank: usize, rows: usize, core_head: bool) -> [usize; 3] {
    if rank == 0 { [rows * 4, 0, 0] }
    else { [0, rows.min(64) * vocab * 4, if core_head { 0 } else { 4 << 20 }] }
}

/// Return slots for the prefill units, from every admitted rank: `shape.waves`
/// slots of `shape.unit_rows` each.
pub fn spark_demand(shape: crate::serving_capacity::PrefillShape, lane: usize, ranks: usize) -> Demand {
    Demand::new(1, Category::Workspace, format!("Kimi Spark lane {lane} latent intake"),
        crate::serving_capacity::prefill_return_bytes(shape.waves, shape.unit_rows, ranks as u64, 3584), Basis::Exact)
}

/// Row planes hold exactly the admitted units; scratch and return slots serve
/// one unit each. Diagnostic callers normalize their shape before allocation.
pub fn prefill_shape_with(rows: usize, waves: usize, unit: Option<usize>)
    -> Result<crate::serving_capacity::PrefillShape> {
    use crate::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides};
    ensure!((1..=crate::serving_capacity::prefill_waves::MAX_WAVES as usize).contains(&waves)
        && (1..=LANE_ROWS_MAX).contains(&rows) && rows >= 64 * waves,
        "Kimi lane row/wave capacity: {rows} rows x {waves} waves");
    let unit = unit.unwrap_or(rows / waves) as u64;
    let plan = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
        unit_rows: Some(unit), waves: Some(waves as u64), span_rows: Some(rows as u64), reserve_rows: None,
    })?;
    Ok(crate::serving_capacity::PrefillShape { unit_rows: plan.unit_rows, waves: plan.waves })
}

pub fn exchange_demand(rows: usize, rank: u8, progress: usize) -> Result<Demand> {
    let recv = super::core::exchange_recv_bytes(rows)?;
    let bytes = recv + 256 + 256 + (((crate::placement::HOP_SLOTS + 1) * 16) as usize).max(256) + progress;
    Ok(Demand::new(rank, Category::Workspace, "Kimi peer/abort/hop controls", bytes as u64, Basis::Exact))
}

pub fn code_demand(rank: usize, pending: u64, cublas_bytes: u64, lanes: usize) -> Result<Demand> {
    ensure!(rank < 2 && (1..=2).contains(&lanes), "Kimi code allowance rank/lane geometry");
    let bytes = pending.checked_add(if rank == 1 { lanes as u64 * cublas_bytes } else { 0 })
        .context("Kimi future code/handle allowance overflow")?;
    Ok(Demand::new(rank as u8, Category::Workspace, "Kimi future loaded code and private head handles", bytes, Basis::Formula))
}

/// Exact FirstCapacity TP2 owner, using the shared EXL3 capacity arenas. A
/// TP2 FFN completes inside one unit's hook: `rows` is the unit.
pub fn expert_workspace(package: &std::path::Path, rows: usize) -> Result<u64> {
    let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
        .capacities(rows)?;
    let manifests = capacities.into_iter().map(|capacity| -> Result<serde_json::Value> {
        let path = package.join(format!("m{capacity}/v41_exl3.json"));
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)
            .with_context(|| format!("read {}", path.display()))?)?;
        ensure!(manifest["hidden"] == 3584 && manifest["capacity"] == capacity,
            "Kimi TP2 expert workspace geometry/capacity mismatch");
        Ok(manifest)
    }).collect::<Result<Vec<_>>>()?;
    Ok(crate::serving_capacity::exl3_workspace_bytes(&manifests, true)? + (rows * 3584 * 4) as u64)
}

pub fn expert_package(manifest: &std::path::Path) -> Result<std::path::PathBuf> {
    let parent = manifest.parent().context("Kimi manifest has no parent")?;
    [parent.join("exl3/exl3-kimik3-k23/rtx-tp2"),
        parent.join("../lib/exl3/exl3-kimik3-k23/rtx-tp2")].into_iter()
        .find(|path| path.is_dir()).context("Kimi installed TP2 expert workspace manifests missing")
}

/// Lane row planes plus `shape.waves` return slots of `shape.unit_rows` each.
pub fn baseline_demands(cfg: &KimiK3Config, manifest: &serde_json::Value,
    rows: usize, shape: crate::serving_capacity::PrefillShape, context: usize, slots: usize, verify: usize,
    core: &super::core::CoreInventory, heads: bool, ranks: usize) -> Result<Vec<Demand>> {
    let shape = shape.validate()?;
    let unit = usize::try_from(shape.unit_rows)?;
    prefill_shape_with(rows, usize::try_from(shape.waves)?, Some(unit))?;
    let layout = NativeLaneLayout::new(cfg, rows, unit, context, NativeScratch::from_manifest(manifest)?)?;
    let owners = crate::placement::attention::alternating_owners(
        cfg.attention.iter().map(|kind| (*kind == KimiAttention::Mla, None)), 0)?;
    let graphs = graphs(cfg, 1, unit, slots, verify, |rank, layer| cfg.attention[layer] == KimiAttention::Kda
        || heads || rank == usize::from(owners[layer]))?;
    let mut demands = vec![spark_demand(shape, 0, ranks)];
    for rank in 0..2 {
        demands.push(Demand::new(rank as u8, Category::Workspace, "Kimi lane 0 native operands",
            layout.bytes()? as u64, Basis::Exact));
        demands.push(Demand::new(rank as u8, Category::Workspace, "Kimi lane 0 token IO",
            io_extents(cfg.vocab_size, rank, rows, core.get_weight("language_model.lm_head").is_some())
                .into_iter().sum::<usize>() as u64, Basis::Exact));
        demands.push(graph_demand(&graphs, rank));
        demands.push(exchange_demand(unit, rank as u8, 0)?);
        // Active state and retained marks are charged by the caller's mark policy.
        demands.extend(kda_demands(cfg, slots, 1, 1, rank as u8)?.into_iter()
            .filter(|d| d.category == Category::Workspace));
    }
    Ok(demands)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> KimiK3Config {
        KimiK3Config::from_hf(&crate::plan::testing::kimi_k3_config()).unwrap()
    }

    #[test]
    fn two_unit_lane_doubles_row_planes_and_keeps_unit_scratch() -> Result<()> {
        let cfg = cfg();
        let scratch = NativeScratch { kda: 1024, mla: [512, 0, 1024] };
        let one = NativeLaneLayout::new(&cfg, 4096, 4096, 1_048_576, scratch)?;
        let two = NativeLaneLayout::new(&cfg, 8192, 4096, 1_048_576, scratch)?;
        assert_eq!(two.arena_bytes, one.arena_bytes, "hook-local scratch serves one unit");
        assert_eq!(two.arena_offsets, one.arena_offsets);
        for name in ["residual arena", "delta", "normalized", "ffn output", "expert combined",
            "route ids", "expert wire"] {
            assert_eq!(two.operands[name], 2 * one.operands[name], "{name} spans the lane");
        }
        for name in ["gate-up", "activated", "KDA normalized", "MLA query", "MLA attention"]
            .into_iter().chain(HOOK_SCRATCH) {
            assert_eq!(two.operands[name], one.operands[name], "{name} is one unit's");
        }
        for name in ATTENTION_METADATA {
            assert_eq!(two.metadata_extent(name, 0)?.1, one.metadata_extent(name, 0)?.1);
            assert_eq!(two.operands[name], 2 * one.operands[name], "{name} has two unit banks");
        }
        // Units never exceed the kernels' 4096 rows nor the admitted lane.
        for (rows, unit) in [(LANE_ROWS_MAX + 1, 4096), (8192, 4097), (2048, 4096), (4096, 0)] {
            assert!(NativeLaneLayout::new(&cfg, rows, unit, 1_048_576, scratch).is_err(), "{rows}/{unit}");
        }
        Ok(())
    }

    #[test]
    fn shared_wave_shapes_allocate_exact_row_planes_and_unit_scratch() -> Result<()> {
        use crate::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides};
        let cfg = cfg();
        let scratch = NativeScratch { kda: 1024, mla: [512, 0, 1024] };
        for (waves, unit, rows) in [(2, 4096, 8192), (3, 2752, 8256), (4, 2048, 8192)] {
            let plan = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
                waves: Some(waves), ..Default::default()
            })?;
            assert_eq!((plan.unit_rows, plan.span_rows), (unit, rows));
            let shape = prefill_shape_with(rows as usize, waves as usize, Some(unit as usize))?;
            assert_eq!(shape, plan.units());
            let one = NativeLaneLayout::new(&cfg, unit as usize, unit as usize, 1_048_576, scratch)?;
            let lane = NativeLaneLayout::new(&cfg, rows as usize, unit as usize, 1_048_576, scratch)?;
            assert_eq!(lane.metadata_banks(), waves as usize, "banks follow the admitted depth");
            assert_eq!(lane.arena_bytes, one.arena_bytes);
            for name in ["residual arena", "delta", "expert combined", "route ids", "expert wire"] {
                assert_eq!(lane.operands[name], waves as usize * one.operands[name], "{name}");
            }
            for name in HOOK_SCRATCH { assert_eq!(lane.operands[name], one.operands[name], "{name}"); }
            assert_eq!(lane.metadata_extent("KDA slots", 0)?.1, unit as usize * 4);
            assert_eq!(lane.metadata_extent("MLA slots", 0)?.1, unit as usize * 8);
            for ranks in 1..=8 {
                assert_eq!(spark_demand(shape, 0, ranks).bytes, rows * ranks as u64 * 3584 * 2);
            }
            assert!(prefill_shape_with(rows as usize - 64, waves as usize, Some(unit as usize)).is_err());
        }
        Ok(())
    }

    #[test]
    fn hook_scratch_is_unit_sized_but_pending_wave_planes_keep_the_span() -> Result<()> {
        let cfg = cfg();
        let layout = NativeLaneLayout::new(&cfg, 8256, 2752, 1_048_576,
            NativeScratch { kda: 16, mla: [16; 3] })?;
        for name in HOOK_SCRATCH {
            let stride = if name == "router logits" { cfg.experts * 4 } else { cfg.latent * 2 };
            assert_eq!(layout.operands[name], layout.unit * stride);
            assert_ne!(layout.operands[name], layout.rows * stride, "negative control: span-sized scratch");
        }
        assert_eq!(layout.operands["expert combined"], layout.rows * cfg.latent * 2);
        for name in ["normalized", "shared output", "routed output", "delta"] {
            assert_eq!(layout.operands[name], layout.rows * cfg.hidden * 2);
        }
        Ok(())
    }

    #[test]
    fn metadata_banks_are_disjoint_and_reject_undersized_or_unknown_planes() -> Result<()> {
        let layout = NativeLaneLayout::new(&cfg(), 8256, 2752, 1_048_576,
            NativeScratch { kda: 16, mla: [16; 3] })?;
        assert_eq!(layout.metadata_banks(), 3);
        assert_ne!(layout.metadata_banks(), 4, "negative control: no maximum-depth padding");
        for rows in [1, 16, 64] {
            let decode = NativeLaneLayout::new(&cfg(), rows, rows, 1_048_576,
                NativeScratch { kda: 16, mla: [16; 3] })?;
            assert_eq!(decode.metadata_banks(), 1, "decode rows {rows}");
        }
        for name in ATTENTION_METADATA {
            let mut end = 0;
            for bank in 0..layout.metadata_banks() {
                let (offset, bytes) = layout.metadata_extent(name, bank)?;
                assert_eq!(offset, end);
                assert!(bytes >= 16);
                assert_eq!(offset % 16, 0, "{name} bank {bank} offset");
                assert_eq!(bytes % 16, 0, "{name} bank {bank} stride");
                end += bytes;
            }
            assert_eq!(end, layout.operands[name]);
            assert!(layout.metadata_extent(name, layout.metadata_banks()).is_err());
        }
        assert!(layout.metadata_extent("delta", 0).is_err());
        let mut bad = layout.clone();
        *bad.operands.get_mut("KDA slots").unwrap() -= 1;
        assert!(bad.metadata_extent("KDA slots", 0).is_err());
        Ok(())
    }

    #[test]
    fn every_metadata_bank_is_aligned_even_for_odd_unit_and_context_extents() -> Result<()> {
        for (rows, unit, context) in [(768, 256, 1_048_576), (8256, 2752, 1_048_576),
            (195, 65, 65), (195, 65, 129), (9, 3, 65), (51, 17, 129)] {
            let layout = NativeLaneLayout::new(&cfg(), rows, unit, context,
                NativeScratch { kda: 16, mla: [16; 3] })?;
            let requests = unit.min(64);
            for (name, raw) in [("KDA slots", unit * 4), ("KDA first", unit * 4),
                ("MLA slots", unit * 8), ("MLA page table", requests * context.div_ceil(64) * 4),
                ("MLA lengths", requests * 4), ("MLA query offsets", (requests + 1) * 4)] {
                let stride = raw.max(16).div_ceil(16) * 16;
                assert_eq!(layout.operands[name], stride * layout.metadata_banks());
                for bank in 0..layout.metadata_banks() {
                    assert_eq!(layout.metadata_extent(name, bank)?, (bank * stride, stride));
                    assert_eq!((bank * stride) % 16, 0, "{name} bank {bank}");
                }
            }
        }
        // Negative control: the former 65-i32 stride misaligned banks 1 and 2.
        let mut old = NativeLaneLayout::new(&cfg(), 768, 256, 1_048_576,
            NativeScratch { kda: 16, mla: [16; 3] })?;
        let old_stride = 65 * 4;
        old.operands.insert("MLA query offsets", old_stride * old.metadata_banks());
        assert_eq!(old_stride, 260);
        assert_ne!(old_stride % 16, 0);
        for bank in 0..old.metadata_banks() {
            assert!(old.metadata_extent("MLA query offsets", bank).is_err());
        }
        Ok(())
    }

    #[test]
    fn prefill_shapes_derive_units_up_to_4096() -> Result<()> {
        let unit = |rows, waves, explicit| prefill_shape_with(rows, waves, explicit).map(|s| s.unit_rows);
        assert_eq!(unit(8192, 2, None)?, 4096);
        assert_eq!(unit(8192, 2, Some(4096))?, 4096);
        assert_eq!(unit(4096, 2, None)?, 2048);
        assert_eq!(unit(4096, 1, None)?, 4096);
        assert!(unit(8192, 1, None).is_err(), "a unit is at most 4096 rows");
        assert!(unit(8192, 2, Some(8192)).is_err(), "a unit is at most 4096 rows");
        assert!(unit(4096, 2, Some(4096)).is_err(), "the lane holds both units");
        assert!(unit(8192, 2, Some(2048)).is_err(), "the lane holds exactly its live units");
        assert!(unit(4097, 2, None).is_err(), "the lane cannot hide a rounded tail");
        assert!(unit(8256, 2, Some(2048)).is_err(), "the reserve of 2048-row slots is 8192 rows");
        Ok(())
    }
}
