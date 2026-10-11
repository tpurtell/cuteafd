//! Qwen's shared admission request. Weights, PLE and resident expert arenas
//! are already in the baseline; every allocation after that sample is named
//! here. Startup graphs depend on the admitted pool, so resolve their shape
//! before publishing the final placement, without taking another CUDA sample.
use crate::families::qwen4::Qwen4Attention;
use crate::placement::*;
use crate::serving_capacity::qwen_graphs::{qwen_admission, qwen_startup_graphs, QwenAdmissionInputs, QWEN_UNIT_ROWS};
use cuteafd_core::memory_layout::{Basis, Category};

pub const DEFAULT_MARK_SLOTS: u64 = 18;
pub const LAZY_GRAPH_BYTES: u64 = (1 << 30) / 2;
pub const SPARK_WORKSPACE_BYTES: u64 = 56 * (1 << 30) / 100;
pub const SPARK_RING_BYTES: u64 = 78 * (1 << 30) / 100;

/// A dual Qwen executor requires nonempty whole-width ownership on both ends.
/// Do not silently turn the shared solver's legal all-on-one-rank cut into a
/// private cut heuristic or a partly idle dual executor.
pub fn check_dual_layer_owners(owners: &[usize]) -> anyhow::Result<()> {
    anyhow::ensure!(owners.first() == Some(&0) && owners.last() == Some(&1)
        && owners.iter().all(|&owner| owner <= 1)
        && owners.windows(2).filter(|pair| pair[0] != pair[1]).count() == 1,
        "Qwen dual attention needs nonempty owners and one contiguous owner-zero to owner-one cutover");
    Ok(())
}

pub struct QwenInputs<'a> {
    pub admission: QwenAdmissionInputs<'a>,
    pub capacity_bytes: u64,
    pub baseline: Baseline,
    pub pending_code_bytes: u64,
    pub max_context: u64,
    pub requested_pool: Option<u64>,
    pub spark_ranks: usize,
    /// None means lazy diagnostic graphs, not the serving startup set.
    pub startup_graph_modes: Option<(usize, bool)>,
}

pub fn request(inputs: &QwenInputs<'_>, graph_bytes: u64) -> anyhow::Result<PlacementRequest> {
    let admission = qwen_admission(&inputs.admission)?;
    let unit_rows = QWEN_UNIT_ROWS as u64;
    let whole = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
    let mut previous = 0;
    let mut layers = Vec::new();
    for (index, &kind) in inputs.admission.cfg.attention[..inputs.admission.layers].iter().enumerate() {
        let geometry = crate::serving_capacity::qwen_cache_geometry(inputs.admission.cfg, index + 1, false, inputs.admission.kv_format)?;
        let bytes = geometry.ranks[0].persistent_unit_bytes;
        layers.push(LayerDemand {
            kind: if kind == Qwen4Attention::Full { AttentionClass::Gqa } else { AttentionClass::Gdn },
            weights: ModeBytes::default(),
            kv_unit: KvDemand { unit_bytes_whole: bytes - previous, ..Default::default() },
            fixed_bytes: ModeBytes::default(), context_indexer: false, colocate: None,
            experts: None,
            modes: vec![whole],
        });
        previous = bytes;
    }
    let records: u64 = layers.iter().map(|l| l.kv_unit.unit_bytes_whole).sum();
    // Includes optional MTP records, pool metadata, the prefill/decode page
    // tables, and the existing per-token rounding of those table bytes.
    let overhead = admission.per_token.checked_mul(unit_rows).and_then(|n| n.checked_sub(records))
        .ok_or_else(|| anyhow::anyhow!("Qwen pool overhead overflow"))?;
    let mut fixed = admission.items.iter().map(|&(category, group, bytes)|
        Demand::new(0, category, group, bytes, Basis::Formula)).collect::<Vec<_>>();
    fixed.push(Demand::new(0, Category::Runtime, "startup graphs", graph_bytes, Basis::Formula));
    if inputs.pending_code_bytes > 0 {
        fixed.push(Demand::new(0, Category::Runtime, "loaded code", inputs.pending_code_bytes, Basis::Formula));
    }
    Ok(PlacementRequest {
        inventory: Inventory { gpus: vec![GpuBudget { capacity_bytes: inputs.capacity_bytes,
            headroom_bytes: admission.headroom, baseline: inputs.baseline }],
            spark_ranks: inputs.spark_ranks, peer_access: false },
        attention_placement: Some(AttentionPlacement::Heads),
        context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool: PoolPolicy::resolve(&[inputs.capacity_bytes], inputs.max_context, inputs.requested_pool,
            unit_rows, inputs.spark_ranks == 0),
        layers, pool_overhead: vec![overhead], fixed, movables: Vec::new(),
        expert_workspace: 0, tp2_workspace: [0; 2], onboard: Onboard::Auto, expert_gpus: 0,
        policy: LayerPolicy { default: vec![whole], by_kind: Vec::new() },
        hops: HopSpec { row_bytes: inputs.admission.cfg.hc_count as u64 * inputs.admission.cfg.hidden as u64 * 2,
            rows: inputs.admission.prefill_rows.max(64), lanes: 1, entry_gpu: 0, head_gpu: 0 },
        executor: super::QWEN4,
    })
}

/// Solve using one baseline; descend through the pool-dependent graph sets
/// exactly as the former graph-aware admission did. An explicit pool stays
/// strict. No state is allocated until this converges.
pub fn placement(inputs: &QwenInputs<'_>) -> anyhow::Result<Placement> {
    let Some((sequences, speculation)) = inputs.startup_graph_modes else {
        let mut result = solve(&request(inputs, LAZY_GRAPH_BYTES)?)?;
        for item in &mut result.items[0] {
            if item.group == "startup graphs" { item.group = "graph growth".into(); }
        }
        return Ok(result);
    };
    anyhow::ensure!(sequences <= 16, "Qwen startup graphs support at most 16 concurrent sequences");
    let mut tokens = solve(&request(inputs, 0)?)?.pool_tokens;
    loop {
        let graphs = qwen_startup_graphs(usize::try_from(inputs.max_context)?, usize::try_from(tokens)?,
            inputs.admission.cfg.dense_context(), sequences, speculation, inputs.admission.layers)
            .ok_or_else(|| anyhow::anyhow!("Qwen graph count overflow"))?;
        let mut req = request(inputs, graphs.bytes(0))?;
        req.pool.target = req.pool.target.min(tokens);
        let mut result = solve(&req)?;
        if result.pool_tokens == tokens {
            // Keep the ready ledger's measured set separate from its growth
            // margin, while both remain charged during admission.
            result.items[0].retain(|i| i.group != "startup graphs");
            result.items[0].extend(graphs.items(0));
            return Ok(result);
        }
        anyhow::ensure!(result.pool_tokens < tokens, "Qwen graph admission did not descend");
        tokens = result.pool_tokens;
    }
}

/// Admission contract for the private whole-owner executor. Unlike the legacy
/// single-owner path, layer weights and routed halves are future allocations;
/// both baselines must be sampled before loading them. The public selector
/// stays single-owner until safety and numerical gates enable this caller.
pub struct QwenDualInputs<'a> {
    pub admission: QwenAdmissionInputs<'a>,
    pub gpus: [GpuBudget; 2],
    pub layer_weights: &'a [u64],
    pub layer_experts: &'a [Option<ExpertCost>],
    /// Head, MTP weights/experts, embedding, PLE table and exact transport
    /// buffers. Head/MTP are owner-only groups, not paired rank groups.
    pub fixed: Vec<Demand>,
    pub tp2_workspace: [u64; 2],
    pub pending_code_bytes: [u64; 2],
    pub mark_slots: u64,
    pub max_context: u64,
    pub requested_pool: Option<u64>,
    pub spark_ranks: usize,
    pub onboard: Onboard,
    pub startup_graph_modes: Option<(usize, bool)>,
}

const DUAL_WHOLE: [LayerMode; 2] = [LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
    LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }];

pub fn dual_request(inputs: &QwenDualInputs<'_>, graph_bytes: [u64; 2]) -> anyhow::Result<PlacementRequest> {
    let a = &inputs.admission;
    anyhow::ensure!(a.layers > 1 && a.layers <= a.cfg.layers
        && inputs.layer_weights.len() == a.layers && inputs.layer_experts.len() == a.layers,
        "Qwen dual demands must name every backbone layer");
    anyhow::ensure!(a.future_expert_bytes == 0, "Qwen dual experts must be solver-owned TP2 halves");
    anyhow::ensure!(inputs.layer_experts.iter().flatten().all(|cost| cost.tp2),
        "Qwen dual routed experts require TP2 packages");
    let mut layers = Vec::with_capacity(a.layers);
    let mut previous = crate::serving_capacity::RankCacheGeometry::default();
    for index in 0..a.layers {
        let current = crate::serving_capacity::qwen_cache_geometry(a.cfg, index + 1, false, a.kv_format)?.ranks[0];
        let state = (current.active_state_per_sequence_bytes - previous.active_state_per_sequence_bytes)
            .checked_mul(a.slots).and_then(|n| n.checked_add(current.speculative_replay_bytes - previous.speculative_replay_bytes))
            .and_then(|n| (current.retained_mark_bytes - previous.retained_mark_bytes).checked_mul(inputs.mark_slots)
                .and_then(|marks| n.checked_add(marks)))
            .ok_or_else(|| anyhow::anyhow!("Qwen owner state overflow"))?;
        layers.push(LayerDemand {
            kind: if a.cfg.attention[index] == Qwen4Attention::Full { AttentionClass::Gqa } else { AttentionClass::Gdn },
            weights: ModeBytes::replicated(inputs.layer_weights[index]),
            kv_unit: KvDemand { unit_bytes_whole: current.persistent_unit_bytes - previous.persistent_unit_bytes,
                ..Default::default() },
            fixed_bytes: ModeBytes::replicated(state), context_indexer: false, colocate: None,
            experts: inputs.layer_experts[index], modes: DUAL_WHOLE.to_vec(),
        });
        previous = current;
    }
    let complete = crate::serving_capacity::qwen_cache_geometry(a.cfg, a.layers, a.mtp, a.kv_format)?.ranks[0];
    // Both workspaces allocate full-width page tables; owner0 alone owns pool
    // metadata, MTP records/pending rows and the deferred MTP id buffer.
    let tables = (1 + 64) * 5 * 4;
    let mut overhead = [tables, tables];
    overhead[0] += complete.persistent_unit_bytes - previous.persistent_unit_bytes + complete.pool_metadata_unit_bytes;
    // Preserve single-owner per-token rounding; do not hide rounded table bytes.
    for bytes in &mut overhead { *bytes = bytes.div_ceil(QWEN_UNIT_ROWS as u64) * QWEN_UNIT_ROWS as u64; }
    let mut fixed = inputs.fixed.clone();
    let admitted = qwen_admission(a)?;
    for gpu in 0..2 {
        fixed.extend(admitted.items.iter().filter(|(_, group, _)| matches!(*group, "decode step" | "prefill step"))
            .map(|&(category, group, bytes)| Demand::new(gpu as u8, category, group, bytes, Basis::Formula)));
        fixed.push(Demand::new(gpu as u8, Category::Kv, "owner commit tables", 3 * 64 * 4, Basis::Formula));
        fixed.push(Demand::new(gpu as u8, Category::Runtime, "startup graphs", graph_bytes[gpu], Basis::Formula));
        fixed.push(Demand::new(gpu as u8, Category::Runtime, "loaded code", inputs.pending_code_bytes[gpu], Basis::Formula));
    }
    let pending = (complete.active_state_per_sequence_bytes - previous.active_state_per_sequence_bytes)
        .checked_mul(a.slots).and_then(|n| n.checked_add(complete.fixed_state_bytes - 3 * 64 * 4))
        .ok_or_else(|| anyhow::anyhow!("Qwen MTP state overflow"))?;
    fixed.push(Demand::new(0, Category::Kv, "owner0 MTP pending", pending, Basis::Formula));
    if a.full_prefill_logits > 0 {
        fixed.push(Demand::new(0, Category::Workspace, "probe prefill logits", a.full_prefill_logits, Basis::Formula));
    }
    let capacities = inputs.gpus.map(|g| g.capacity_bytes);
    let mut pool = PoolPolicy::resolve(&capacities, inputs.max_context, inputs.requested_pool,
        QWEN_UNIT_ROWS as u64, inputs.spark_ranks == 0);
    pool.ceiling = pool.target;
    Ok(PlacementRequest {
        inventory: Inventory { gpus: inputs.gpus.to_vec(), spark_ranks: inputs.spark_ranks, peer_access: true },
        attention_placement: Some(AttentionPlacement::Layers), context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool, layers, pool_overhead: overhead.to_vec(), fixed, movables: Vec::new(), expert_workspace: 0,
        tp2_workspace: inputs.tp2_workspace, onboard: inputs.onboard, expert_gpus: 2,
        policy: LayerPolicy { default: DUAL_WHOLE.to_vec(), by_kind: Vec::new() },
        hops: HopSpec { row_bytes: a.cfg.hc_width() as u64 * 2, rows: a.prefill_rows.max(64), lanes: 1,
            entry_gpu: 0, head_gpu: 0 },
        executor: ExecutorModes { family: "qwen4", modes: &DUAL_WHOLE, hops: true },
    })
}

/// Converge the owner-partitioned startup graph inventory and pool against the
/// same pre-allocation baselines. Refuse an empty chain or a graph/owner cycle;
/// neither condition is permission to invent a family-private boundary.
pub fn dual_placement(inputs: &QwenDualInputs<'_>) -> anyhow::Result<Placement> {
    use crate::serving_capacity::qwen_graphs::qwen_startup_graphs_placed;
    let owners = |p: &Placement| -> anyhow::Result<Vec<usize>> {
        p.layers.iter().map(|layer| match layer.mode {
            LayerMode::Whole { gpu, ffn: FfnMode::Owner } => Ok(usize::from(gpu)),
            _ => anyhow::bail!("Qwen dual requires whole owner layers"),
        }).collect()
    };
    let Some((sequences, speculation)) = inputs.startup_graph_modes else {
        let mut result = solve(&dual_request(inputs, [LAZY_GRAPH_BYTES; 2])?)?;
        check_dual_layer_owners(&owners(&result)?)?;
        for rank in &mut result.items {
            for item in rank { if item.group == "startup graphs" { item.group = "graph growth".into(); } }
        }
        return Ok(result);
    };
    anyhow::ensure!(sequences <= 16, "Qwen startup graphs support at most 16 concurrent sequences");
    let mut result = solve(&dual_request(inputs, [0; 2])?)?;
    let mut visited = std::collections::BTreeSet::new();
    loop {
        let layer_owners = owners(&result)?;
        check_dual_layer_owners(&layer_owners)?;
        anyhow::ensure!(visited.insert((result.pool_tokens, layer_owners.clone())),
            "Qwen graph admission owner/pool cycle");
        let graphs = qwen_startup_graphs_placed(usize::try_from(inputs.max_context)?,
            usize::try_from(result.pool_tokens)?, inputs.admission.cfg.dense_context(), sequences, speculation, &layer_owners)
            .ok_or_else(|| anyhow::anyhow!("Qwen owner graph count overflow"))?;
        let mut req = dual_request(inputs, [graphs.bytes(0), graphs.bytes(1)])?;
        req.pool.target = req.pool.target.min(result.pool_tokens);
        req.pool.ceiling = req.pool.ceiling.min(result.pool_tokens);
        let mut next = solve(&req)?;
        if next.pool_tokens == result.pool_tokens && owners(&next)? == layer_owners {
            for gpu in 0..2 {
                next.items[gpu].retain(|item| item.group != "startup graphs");
                next.items[gpu].extend(graphs.items(gpu));
            }
            return Ok(next);
        }
        anyhow::ensure!(next.pool_tokens <= result.pool_tokens, "Qwen graph admission did not descend");
        result = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen4::{Qwen4Config, Qwen4KvCache};
    use crate::serving_capacity::qwen_graphs::qwen_graph_pool;

    #[test]
    fn planner_equals_runtime_qwen4_dual_contract() {
        use crate::serving_capacity::qwen_graphs::qwen_cache_geometry_placed;
        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        let weights = vec![100 << 20; 48];
        let experts = vec![Some(ExpertCost { whole: Bytes2::default(),
            half: [Bytes2 { resident: 600 << 20, staging: 64 << 20 }, Bytes2 { resident: 400 << 20, staging: 32 << 20 }],
            tp2: true, spark_ok: true }); 48];
        for kv_format in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
            let inputs = |measured| QwenDualInputs {
                admission: QwenAdmissionInputs { cfg: &cfg, layers: 48, mtp: false, kv_format, manifest: None,
                    prefill_rows: 4096, slots: 16, mark_bytes: 0, full_prefill_logits: 0,
                    ple: None, future_expert_bytes: 0, headroom: 3 << 30 },
                gpus: [0, 1].map(|_| GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 3 << 30,
                    baseline: if measured { Baseline::Measured { free_bytes: (96 << 30) - (512 << 20) } }
                        else { Baseline::Planned { context_bytes: 512 << 20, loaded_bytes: 0 } } }),
                layer_weights: &weights, layer_experts: &experts,
                fixed: vec![Demand::new(0, Category::Weights, "owner0 head", 1 << 30, Basis::Exact)],
                tp2_workspace: [128 << 20, 96 << 20], pending_code_bytes: [64 << 20; 2], mark_slots: 18,
                max_context: 131072, requested_pool: Some(2 << 20), spark_ranks: 0, onboard: Onboard::Auto,
                startup_graph_modes: Some((16, true)),
            };
            let planned = dual_placement(&inputs(false)).unwrap();
            let measured = dual_placement(&inputs(true)).unwrap();
            assert_eq!(planned, measured);
            assert_eq!(planned.pool_tokens, 2 << 20);
            assert!(planned.layers.iter().all(|layer| layer.experts == ExpertHome::RtxTp2));
            let owners = planned.layers.iter().map(|layer| match layer.mode {
                LayerMode::Whole { gpu, .. } => usize::from(gpu), _ => unreachable!(),
            }).collect::<Vec<_>>();
            check_dual_layer_owners(&owners).unwrap();
            assert_eq!(planned.hops.len(), 2);
            let req = dual_request(&inputs(false), [0; 2]).unwrap();
            let geometry = qwen_cache_geometry_placed(&cfg, &owners, false, kv_format).unwrap();
            for gpu in 0..2 {
                let state = req.layers.iter().zip(&owners).filter(|(_, owner)| **owner == gpu)
                    .map(|(layer, _)| layer.fixed_bytes.whole).sum::<u64>();
                let rank = geometry.ranks[gpu];
                assert_eq!(state, rank.active_state_per_sequence_bytes * 16 + rank.retained_mark_bytes * 18
                    + rank.speculative_replay_bytes);
                let records = req.layers.iter().zip(&owners).filter(|(_, owner)| **owner == gpu)
                    .map(|(layer, _)| layer.kv_unit.unit_bytes_whole).sum::<u64>();
                let tables = 65 * 5 * 4;
                assert_eq!(records + req.pool_overhead[gpu],
                    (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes + tables).div_ceil(256) * 256);
            }
        }
    }

    #[test]
    fn planner_equals_runtime_qwen4() {
        let mut config = crate::plan::testing::qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let cfg = Qwen4Config::from_hf(&config).unwrap();
        for (total, kv_format) in [32_u64 << 30, 101_973_491_712].into_iter().flat_map(|total|
            [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8].map(|format| (total, format))) {
            for (expert_bytes, mtp) in [(45_u64 << 30, false), (50_u64 << 30, true)] {
                for requested in [None, Some(32768), Some(2097152)] {
                    let context = 131072;
                    let context_bytes = 586_416_128;
                    let inputs = |baseline| QwenInputs {
                        admission: QwenAdmissionInputs { cfg: &cfg, layers: 48, mtp, kv_format, manifest: None,
                            prefill_rows: 4096, slots: 16, mark_bytes: 256 << 20, full_prefill_logits: 0,
                            ple: Some((160, true)), future_expert_bytes: 0, headroom: 3 << 30 },
                        capacity_bytes: total, baseline, pending_code_bytes: 0, max_context: context,
                        requested_pool: requested, spark_ranks: 0, startup_graph_modes: Some((16, true)),
                    };
                    let planned = inputs(Baseline::Planned { context_bytes, loaded_bytes: expert_bytes });
                    let measured = inputs(Baseline::Measured {
                        free_bytes: total.saturating_sub(context_bytes + expert_bytes) });
                    match (placement(&planned), placement(&measured)) {
                        (Ok(p), Ok(r)) => {
                            assert_eq!(p, r);
                            let admission = qwen_admission(&planned.admission).unwrap();
                            let policy = PoolPolicy::resolve(&[total], context, requested, 256, true);
                            let old = qwen_graph_pool(total - context_bytes - expert_bytes, admission.fixed(),
                                admission.per_token, policy.target, requested, |tokens|
                                    qwen_startup_graphs(context as usize, tokens as usize, cfg.dense_context(), 16, true, 48)).unwrap();
                            assert_eq!(p.pool_tokens, old.0);
                            assert_eq!(p.items[0].iter().map(|i| i.bytes).sum::<u64>(),
                                admission.fixed() - admission.headroom + old.1.bytes(0) + old.0 * admission.per_token);
                        }
                        (Err(p), Err(r)) => assert_eq!(p.to_string(), r.to_string()),
                        other => panic!("planner/runtime disagree: {other:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn qwen_pool_demands_match_geometry_and_lazily_loaded_experts() {
        let mut config = crate::plan::testing::qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let cfg = Qwen4Config::from_hf(&config).unwrap();
        for (mtp, kv_format) in [false, true].into_iter().flat_map(|mtp|
            [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8].map(|format| (mtp, format))) {
            let inputs = QwenInputs {
                admission: QwenAdmissionInputs { cfg: &cfg, layers: 48, mtp, kv_format, manifest: None,
                    prefill_rows: 4096, slots: 16, mark_bytes: 256 << 20, full_prefill_logits: 0,
                    ple: None, future_expert_bytes: 10 << 30, headroom: 3 << 30 },
                capacity_bytes: 96 << 30, baseline: Baseline::Measured { free_bytes: 70 << 30 },
                pending_code_bytes: 780_221_844, max_context: 131072, requested_pool: None,
                spark_ranks: 0, startup_graph_modes: None,
            };
            let req = request(&inputs, LAZY_GRAPH_BYTES).unwrap();
            let admission = qwen_admission(&inputs.admission).unwrap();
            assert_eq!(req.layers.iter().map(|l| l.kv_unit.unit_bytes_whole).sum::<u64>() + req.pool_overhead[0],
                admission.per_token * 256);
            assert!(req.fixed.iter().any(|d| d.group == "lazy EXL3 window" && d.bytes == 10 << 30));
            assert!(req.fixed.iter().any(|d| d.group == "loaded code" && d.bytes == 780_221_844));
        }
    }
}
