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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen4::{Qwen4Config, Qwen4KvCache};
    use crate::serving_capacity::qwen_graphs::qwen_graph_pool;

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
