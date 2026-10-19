//! Native coordinator inventory passed to the shared startup solver, before
//! weights, recurrent state or cache owners allocate. This is not the header
//! lower bound exposed by `plan` and does not admit Spark workers by itself.
use super::{device::{Lane, LaneLayout}, weights::{LayerLayout, LayerWeights, PackedLayer}};
use crate::shared::memory::device::Device;
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_ffi::{KimiKda, MLA_PAGE_ROWS, MLA_RECORD_BYTES};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config}, placement::*};

const MODES: &[LayerMode] = &[LayerMode::HeadSplit,
    LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }];
const EXECUTOR: ExecutorModes = ExecutorModes { family: "kimi_k3", modes: MODES,
    hops: true, attention: AttentionPlacement::Layers };
const FLOOR: u64 = 262_144;

/// All demands are still future to the two post-module CUDA samples. The
/// engine supplies its native programs' scratch, token I/O, lane workspaces,
/// exchange, graphs and page tables, rather than estimated family allowances.
pub(crate) struct Inputs<'a> {
    pub cfg: &'a KimiK3Config,
    pub gpus: [GpuBudget; 2],
    pub devices: [i32; 2],
    pub peer_access: bool,
    pub spark_ranks: usize,
    pub attention: Option<AttentionPlacement>,
    pub requested_pool: Option<u64>,
    pub state_slots: usize,
    pub prefix_marks: usize,
    pub rows: usize,
    pub lanes: usize,
    pub lane_layouts: Vec<[LaneLayout; 2]>,
    pub fixed: Vec<Demand>,
    pub startup: Vec<Demand>,
    pub pool_bounded: Vec<BoundedDemand>,
}

pub(crate) struct Admission {
    pub placement: Placement,
    layouts: Vec<[LayerLayout; 2]>,
    devices: [i32; 2],
    lanes: u64,
    lane_layouts: Vec<[LaneLayout; 2]>,
    lane_allocated: Vec<[bool; 2]>,
}

/// The normal Spark lane consumes these selected layer IDs; selection still
/// goes through shared `SparkAdmission` before any transport connects.
impl Admission {
    pub(crate) fn working_set(&self) -> SparkWorkingSet {
        SparkWorkingSet { rtx_layers: vec![], spark_layers: self.placement.layers.iter().enumerate()
            .filter_map(|(layer, assignment)| (assignment.experts == ExpertHome::Spark).then_some(layer)).collect(),
            spark_ranks: 6, prefill_lanes: self.lanes }
    }

    /// No caller-supplied byte count: upload uses the immutable native layout
    /// reconciled against the shared solver's accepted weight ledger.
    pub(crate) fn upload<'a>(&self, device: Device<'a>, rank: usize, layer: usize,
        packed: PackedLayer) -> Result<LayerWeights<'a>> {
        ensure!(rank < 2, "Kimi upload rank outside TP2");
        ensure!(device.id == self.devices[rank], "Kimi upload device differs from admitted rank");
        let layout = &self.layouts.get(layer).context("Kimi upload layer outside admission")?[rank];
        LayerWeights::upload(device, packed, layout, layout.bytes()?)
    }

    pub(crate) fn allocate_lane<'a>(&mut self, device: Device<'a>, rank: usize, lane: usize) -> Result<Lane<'a>> {
        let layout = self.check_lane(rank, lane, device.id)?;
        let allocated = layout.allocate(device)?;
        self.lane_allocated[lane][rank] = true;
        Ok(allocated)
    }

    fn check_lane(&self, rank: usize, lane: usize, device: i32) -> Result<&LaneLayout> {
        ensure!(rank < 2 && device == self.devices[rank] && lane < self.lanes as usize,
            "Kimi lane allocation differs from admitted rank/device/index");
        ensure!(!self.lane_allocated[lane][rank], "Kimi admitted lane already allocated");
        let layout = &self.lane_layouts[lane][rank];
        let demand = layout.demand(rank as u8, lane)?;
        let charges = self.placement.items[rank].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
        ensure!(charges.len() == 1 && charges[0].category == demand.category
            && charges[0].basis == Basis::Exact && charges[0].bytes == demand.bytes,
            "Kimi lane allocation differs from accepted operand charge");
        Ok(layout)
    }
}

pub(crate) fn admit(inputs: Inputs<'_>) -> Result<Admission> {
    let cfg = inputs.cfg;
    let devices = inputs.devices;
    let lanes = inputs.lanes as u64;
    let lane_layouts = inputs.lane_layouts.clone();
    let request = request(inputs)?;
    let placement = solve(&request)?;
    EXECUTOR.check(&placement)?;
    let layouts = placement.layers.iter().enumerate().map(|(layer, assignment)|
        LayerLayout::pair(cfg, layer, assignment.mode)).collect::<Result<Vec<_>>>()?;
    for rank in 0..2 {
        let native = layouts.iter().try_fold(0u64, |sum, pair| sum.checked_add(pair[rank].bytes()? as u64)
            .context("Kimi admitted native weight sum overflow"))?;
        let charged = placement.items[rank].iter().filter(|item|
            item.category == Category::Weights && (item.group == "layers" || item.group.starts_with("Kimi layer ")))
            .try_fold(0u64, |sum, item| sum.checked_add(item.bytes).context("Kimi weight charge overflow"))?;
        ensure!(native == charged, "Kimi rank {rank} native weights differ from shared admission: {native} != {charged}");
    }
    let lane_allocated = vec![[false; 2]; lane_layouts.len()];
    Ok(Admission { placement, layouts, devices, lanes, lane_layouts, lane_allocated })
}

fn request(i: Inputs<'_>) -> Result<PlacementRequest> {
    let cfg = i.cfg;
    ensure!(cfg.layers > 0 && cfg.layers == cfg.attention.len() && cfg.layers == cfg.dense.len(),
        "Kimi admission layer inventory differs from checkpoint");
    ensure!(i.spark_ranks == 6 && i.peer_access, "Kimi native admission requires TP2 peer access and six Spark ranks");
    ensure!(i.devices.iter().all(|device| *device >= 0) && i.devices[0] != i.devices[1],
        "Kimi admitted devices must be distinct CUDA ordinals");
    ensure!(i.attention != Some(AttentionPlacement::Context), "Kimi context attention has no query/LSE exchange ABI");
    ensure!((1..=64).contains(&i.state_slots) && i.prefix_marks > 0 && (1..=4096).contains(&i.rows)
        && (1..=2).contains(&i.lanes), "Kimi native admission slot/prefix/row/lane geometry");
    ensure!(cfg.kda_heads == 96 && cfg.kda_head_dim == 128 && cfg.kda_conv == 4
        && cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi native state/cache geometry");
    ensure!(i.requested_pool.filter(|n| *n > 0).is_none_or(|n| n >= FLOOR),
        "Kimi pool cannot be below the 256K agentic floor");
    ensure!(i.gpus.iter().all(|gpu| matches!(gpu.baseline, Baseline::Measured { .. })),
        "Kimi runtime admission needs post-module CUDA samples");
    ensure!(i.fixed.iter().chain(&i.startup).all(|d| d.gpu < 2 && d.basis != Basis::Estimated)
        && i.pool_bounded.iter().all(|d| d.gpu < 2 && d.basis != Basis::Estimated),
        "Kimi runtime admission cannot use estimated or out-of-rank demands");
    ensure!(i.lane_layouts.len() == i.lanes && i.lane_layouts.iter().flatten().all(|layout| layout.rows() == i.rows),
        "Kimi lane operand layouts differ from executor geometry");
    let mut fixed = i.fixed;
    for (lane, pair) in i.lane_layouts.iter().enumerate() {
        for (rank, layout) in pair.iter().enumerate() {
            let demand = layout.demand(rank as u8, lane)?;
            ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
                "Kimi lane operand charge supplied twice");
            fixed.push(demand);
        }
    }
    let state = KimiKda::state_bytes(i.state_slots)?.checked_add(KimiKda::conv_state_bytes(i.state_slots)?)
        .context("Kimi state extent overflow")? as u64;
    let mark = KimiKda::state_bytes(1)?.checked_add(KimiKda::conv_state_bytes(1)?)
        .and_then(|n| n.checked_mul(i.prefix_marks)).context("Kimi prefix extent overflow")? as u64;
    let mut layers = Vec::with_capacity(cfg.layers);
    for layer in 0..cfg.layers {
        let split = LayerLayout::pair(cfg, layer, LayerMode::HeadSplit)?;
        let mut split_attention = [0; 2];
        // Whole describes attention residency, not the split FFN's residency.
        // Keep these always-resident native operands on both ranks, as GLM's
        // existing fixed target-weight inventory does for the same mode.
        for rank in 0..2 {
            let attention = split[rank].attention_bytes()?;
            split_attention[rank] = attention as u64;
            fixed.push(Demand::new(rank as u8, Category::Weights, format!("Kimi layer {layer} TP2 FFN/residual/router/latent"),
                (split[rank].bytes()? - attention) as u64, Basis::Exact));
        }
        let mla = cfg.attention[layer] == KimiAttention::Mla;
        let whole = if mla { LayerLayout::pair(cfg, layer, MODES[1])?[0].attention_bytes()? as u64 } else { 0 };
        let mut extra = vec![];
        if !mla {
            for (category, group, bytes) in [(Category::Kv, "Kimi KDA active state", state),
                (Category::Workspace, "Kimi KDA replay", KimiKda::replay_bytes() as u64),
                (Category::Prefix, "Kimi exact KDA marks", mark)] {
                extra.push(LayerExtraDemand::new(category, group, bytes, LayerResidency::Replicated, Basis::Exact));
            }
        }
        let unit = if mla { (MLA_PAGE_ROWS * MLA_RECORD_BYTES) as u64 } else { 0 };
        layers.push(LayerDemand { kind: if mla { AttentionClass::Mla } else { AttentionClass::Kda },
            weights: ModeBytes { whole, split: split_attention },
            kv_unit: KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2], unit_bytes_context: None },
            fixed_bytes: ModeBytes::default(), extra, context_indexer: false, colocate: None,
            // RTX routed-expert onboarding is deliberately not executable yet.
            experts: (!cfg.dense[layer]).then_some(ExpertCost { whole: Bytes2::default(),
                half: [Bytes2::default(); 2], tp2: false, spark_ok: true }),
            modes: if mla { MODES.to_vec() } else { vec![LayerMode::HeadSplit] } });
    }
    let mut pool = PoolPolicy::resolve(&i.gpus.map(|gpu| gpu.capacity_bytes), FLOOR, i.requested_pool, MLA_PAGE_ROWS as u64, true);
    pool.ceiling = pool.target.max(pool.requested.unwrap_or(0));
    Ok(PlacementRequest { inventory: Inventory { gpus: i.gpus.to_vec(), spark_ranks: i.spark_ranks, peer_access: i.peer_access },
        attention_placement: i.attention, context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool, layers, pool_overhead: vec![0; 2], pool_bounded: i.pool_bounded, fixed, startup: i.startup, movables: vec![],
        expert_workspace: 0, tp2_workspace: [0; 2], onboard: Onboard::Layers(0), expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit], ownership: LayerOwnership::Alternating,
            attention: vec![(AttentionClass::Mla, AttentionPlacement::Layers), (AttentionClass::Kda, AttentionPlacement::Heads)],
            ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: (cfg.hidden * 2) as u64, rows: i.rows as u64,
            lanes: i.lanes as u64, entry_gpu: 0, head_gpu: 1, terminal: TerminalHome::Replicated }, executor: EXECUTOR })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> KimiK3Config {
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        cfg.layers = 5;
        cfg.attention = vec![KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Kda];
        cfg.dense.truncate(5);
        cfg
    }

    fn inputs(cfg: &KimiK3Config) -> Inputs<'_> {
        Inputs { cfg, gpus: [GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 2 << 30,
            baseline: Baseline::Measured { free_bytes: 96 << 30 } }; 2], devices: [0, 1], peer_access: true, spark_ranks: 6,
            attention: None, requested_pool: None, state_slots: 2, prefix_marks: 2, rows: 4096, lanes: 2,
            lane_layouts: vec![std::array::from_fn(|_| LaneLayout::new(
                &KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap(), 4096, 1_048_576,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }).unwrap()); 2],
            fixed: (0..2).map(|rank| Demand::new(rank, Category::Workspace, "native engine buffers", 1 << 20, Basis::Exact)).collect(),
            startup: vec![], pool_bounded: vec![] }
    }

    #[test]
    fn owned_mla_charges_nonowner_ffn_and_exact_native_weights() -> Result<()> {
        let cfg = cfg();
        let admitted = admit(inputs(&cfg))?;
        assert_eq!(admitted.placement.pool_tokens, 2 << 20);
        for (layer, owner) in [(1, 0), (3, 1)] {
            assert_eq!(admitted.placement.layers[layer].mode, LayerMode::Whole { gpu: owner, ffn: FfnMode::Split });
            let nonowner = usize::from(1 - owner);
            assert_eq!(admitted.layouts[layer][nonowner].attention_bytes()?, 0);
            let bytes = admitted.layouts[layer][nonowner].bytes()? as u64;
            assert!(bytes > 0);
            let group = format!("Kimi layer {layer} TP2 FFN/residual/router/latent");
            assert_eq!(admitted.placement.items[nonowner].iter().find(|item| item.group == group).unwrap().bytes, bytes);
        }
        let working = admitted.working_set();
        assert!(working.rtx_layers.is_empty());
        assert_eq!(working.spark_layers, cfg.dense.iter().enumerate().filter_map(|(layer, dense)| (!dense).then_some(layer)).collect::<Vec<_>>());
        assert_eq!(working.spark_ranks, 6);
        Ok(())
    }

    #[test]
    fn explicit_heads_preserves_ffns_and_replicates_mla_cache() -> Result<()> {
        let cfg = cfg();
        let owned = admit(inputs(&cfg))?;
        let mut i = inputs(&cfg);
        i.attention = Some(AttentionPlacement::Heads);
        let heads = admit(i)?;
        assert!(heads.placement.layers.iter().all(|layer| layer.mode == LayerMode::HeadSplit));
        let records = |admitted: &Admission| admitted.placement.items.iter().flatten()
            .filter(|item| item.group == "records").map(|item| item.bytes).sum::<u64>();
        assert_eq!(records(&heads), 2 * records(&owned));
        for rank in 0..2 {
            for category in [Category::Kv, Category::Workspace, Category::Prefix] {
                let state = |a: &Admission| a.placement.items[rank].iter().filter(|item|
                    item.category == category && item.group.starts_with("Kimi")).map(|item| item.bytes).sum::<u64>();
                assert_eq!(state(&heads), state(&owned));
            }
        }
        Ok(())
    }

    #[test]
    fn lane_allocations_require_immutable_accepted_rank_extents() -> Result<()> {
        let cfg = cfg();
        let mut admitted = admit(inputs(&cfg))?;
        let expected = admitted.lane_layouts[0][0].bytes()?;
        assert_eq!(admitted.check_lane(0, 0, 0)?.bytes()?, expected);
        for (rank, lane, device) in [(2, 0, 0), (0, 2, 0), (0, 0, 1), (1, 0, -1)] {
            assert!(admitted.check_lane(rank, lane, device).is_err());
        }
        admitted.lane_allocated[0][0] = true;
        assert!(admitted.check_lane(0, 0, 0).is_err());
        admitted.lane_allocated[0][0] = false;
        let group = admitted.lane_layouts[0][0].demand(0, 0)?.group;
        admitted.placement.items[0].iter_mut().find(|item| item.group == group).unwrap().bytes -= 1;
        assert!(admitted.check_lane(0, 0, 0).is_err());
        let mut i = inputs(&cfg);
        i.fixed.push(i.lane_layouts[0][0].demand(0, 0)?);
        assert!(admit(i).is_err());
        let mut i = inputs(&cfg);
        i.lane_layouts.pop();
        assert!(admit(i).is_err());
        Ok(())
    }

    #[test]
    fn refuses_unsupported_or_unmeasured_runtime_inventory() {
        let cfg = cfg();
        for case in 0..10 {
            let mut i = inputs(&cfg);
            match case {
                0 => i.spark_ranks = 4,
                1 => i.peer_access = false,
                2 => i.attention = Some(AttentionPlacement::Context),
                3 => i.requested_pool = Some(FLOOR - 1),
                4 => i.prefix_marks = 0,
                5 => i.gpus[0].baseline = Baseline::Planned { context_bytes: 0, loaded_bytes: 0 },
                6 => i.fixed[0].basis = Basis::Estimated,
                7 => i.fixed[0].gpu = 2,
                8 => i.devices = [1, 1],
                _ => i.devices[0] = -1,
            }
            assert!(admit(i).is_err(), "case {case}");
        }
        let mut i = inputs(&cfg);
        i.requested_pool = Some(2 << 20);
        i.gpus[1].baseline = Baseline::Measured { free_bytes: 2 << 30 };
        assert!(admit(i).is_err());
    }
}
