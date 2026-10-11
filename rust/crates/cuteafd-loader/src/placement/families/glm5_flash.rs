//! GLM Flash admission before weights: planner and runtime differ only in baseline.
use crate::families::glm5_flash::{GlmNextAttention, GlmNextConfig, resident::{GlmfRepresentation, GlmfResidentRank}};
use crate::serving_capacity::{self as cache, glmf_graphs};
use crate::placement::*;
use cuteafd_core::memory_layout::{Basis, Category};

#[derive(Debug, Clone)]
pub struct GlmfInputs<'a> {
    pub cfg: &'a GlmNextConfig,
    pub layers: usize,
    /// Sample after programs, before any weights; planned baseline charges context/code only.
    pub gpus: Vec<(u64, Baseline)>,
    /// Code not present at the baseline sample: LoadedCode minus already-sampled code, per rank.
    pub pending_code: Vec<u64>,
    pub headroom_bytes: u64,
    pub spark_ranks: usize,
    pub prefill_lanes: u64,
    pub prefill_rows: u64,
    pub decode_rows: u64,
    pub partial_bytes: u64,
    pub max_context: u64,
    pub sequences: u64,
    pub state_slots: u64,
    pub mark_slots: u64,
    pub pool_marks: bool,
    pub index: cache::GlmfIndexCache,
    pub kda_state_bytes: u64,
    pub shared_replay: bool,
    pub representation: GlmfRepresentation,
    pub resident: Vec<GlmfResidentRank>,
    /// Exact gate weight + FP32 bias of every selected routed layer, replicated on rank 1.
    pub router_replica_bytes: u64,
    pub workspace: Vec<u64>,
    pub graphs: GraphSet,
    /// Routed layers in backbone order, not including dense layers.
    pub experts: Vec<ExpertCost>,
    pub expert_workspace: u64,
    pub tp2_workspace: [u64; 2],
    /// Entire DFlash allocation, including package scratch; embedding stays host mapped.
    pub drafter_bytes: u64,
    pub requested_pool: Option<u64>,
    pub onboard: Onboard,
    pub full_prefill_logits: u64,
}

pub fn default_onboard() -> Onboard { Onboard::Auto }

pub fn request(i: &GlmfInputs<'_>) -> Result<PlacementRequest, PlacementError> {
    let ranks = i.gpus.len();
    if ![1, 2].contains(&ranks) || i.layers == 0 || i.layers > i.cfg.layers
        || i.resident.len() != ranks || i.workspace.len() != ranks || i.graphs.ranks.len() != ranks
        || i.pending_code.len() != ranks || ![2, 4].contains(&i.partial_bytes) {
        return Err(PlacementError::Inventory("GLM Flash rank/layer inventory"));
    }
    if ranks == 2 && i.decode_rows != cache::GLMF_DECODE_ROWS {
        return Err(PlacementError::Inventory("GLM Flash head split needs 64 decode rows"));
    }
    if i.shared_replay && (ranks != 1 || i.spark_ranks == 0 || i.requested_pool.is_some()) {
        return Err(PlacementError::Inventory("GLM Flash shared replay needs single-GPU automatic Spark admission"));
    }
    let geometry = cache::glm_flash_rank_cache_geometry_rows(i.cfg, i.layers, ranks, i.index,
        i.kda_state_bytes, i.decode_rows).map_err(|_| PlacementError::Inventory("GLM Flash cache geometry"))?;
    let routed = i.cfg.dense[..i.layers].iter().filter(|&&d| !d).count();
    if i.experts.len() != routed {
        return Err(PlacementError::Inventory("GLM Flash routed expert inventory"));
    }
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(PlacementError::Overflow("GLM Flash fixed demand"));
    let mut fixed = Vec::new();
    let mut expert = i.experts.iter();
    let mut layers = Vec::new();
    for layer in 0..i.layers {
        let mla = i.cfg.attention[layer] == GlmNextAttention::Mla;
        let unit = if mla { 256 * (528 + if i.index == cache::GlmfIndexCache::Keys { 512 } else { 0 }) + 64 * 132 } else { 0 };
        layers.push(LayerDemand { kind: if mla { AttentionClass::Dsa } else { AttentionClass::Kda },
            weights: ModeBytes::default(), kv_unit: ModeBytes::replicated(unit),
            experts: if i.cfg.dense[layer] { None } else { expert.next().copied() },
            modes: if ranks == 2 { vec![LayerMode::HeadSplit] } else { vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] } });
    }
    let mla = i.cfg.attention[..i.layers].iter().filter(|&&a| a == GlmNextAttention::Mla).count() as u64;
    let tails = if i.index == cache::GlmfIndexCache::Compact { mla * cache::GLMF_INDEX_TAIL_BYTES } else { 0 };
    let index_replay = if tails > 0 { mla * i.decode_rows * 512 } else { 0 };
    for (rank, g) in geometry.ranks.iter().enumerate() {
        let gpu = rank as u8;
        fixed.push(Demand::new(gpu, Category::Runtime, "pending code", i.pending_code.get(rank).copied().unwrap_or(0), Basis::Calibrated));
        fixed.push(Demand::new(gpu, Category::Embedding, "embedding", i.resident[rank].embedding, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Weights, "target weights", i.resident[rank].weights, Basis::Exact));
        if rank == 1 {
            fixed.push(Demand::new(gpu, Category::Weights, "router replica", i.router_replica_bytes, Basis::Exact));
        }
        let replay = if i.shared_replay { 0 } else { g.speculative_replay_bytes - index_replay };
        fixed.push(Demand::new(gpu, Category::Kv, "KDA state and replay",
            mul(g.active_state_per_sequence_bytes - tails, i.state_slots)? + replay, Basis::Formula));
        fixed.push(Demand::new(gpu, Category::Kv, "DSA index tails and replay",
            mul(tails, i.state_slots)? + index_replay, Basis::Formula));
        fixed.push(Demand::new(gpu, Category::Kv, "commit tables", g.fixed_state_bytes, Basis::Formula));
        if !i.pool_marks {
            fixed.push(Demand::new(gpu, Category::Prefix, "marks", mul(g.retained_mark_bytes, i.mark_slots)?, Basis::Formula));
        } else {
            fixed.push(Demand::new(gpu, Category::Prefix, "reserved units",
                (g.persistent_unit_bytes + g.pool_metadata_unit_bytes) * cache::GLMF_POOL_MARK_RESERVED_UNITS, Basis::Formula));
        }
        fixed.push(Demand::new(gpu, Category::Workspace, "steps", i.workspace[rank], Basis::Formula));
        fixed.extend(i.graphs.items(rank).into_iter().map(|item|
            Demand::new(gpu, item.category, item.group, item.bytes, item.basis)));
        if ranks == 2 {
            fixed.push(Demand::new(gpu, Category::Transport, "peer exchange",
                glmf_graphs::peer_exchange_bytes(i.prefill_lanes, i.prefill_rows.max(i.decode_rows),
                    i.cfg.hidden as u64, i.partial_bytes, i.representation.output_shard), Basis::Formula));
            let slots = 2 * i.prefill_lanes;
            fixed.push(Demand::new(gpu, Category::Transport, "route exchange",
                slots * (i.prefill_rows.max(i.decode_rows) * i.cfg.topk as u64 * 8).max(256)
                    + ((slots + 1) * 16).max(256), Basis::Formula));
        }
        if rank == 0 {
            if i.decode_rows > cache::GLMF_DECODE_ROWS {
                fixed.push(Demand::new(gpu, Category::Workspace, "wide decode selector",
                    cache::glmf_selector_bytes(i.decode_rows, i.cfg.vocab_size as u64)
                        - cache::glmf_selector_bytes(cache::GLMF_DECODE_ROWS, i.cfg.vocab_size as u64), Basis::Formula));
            }
            fixed.push(Demand::new(gpu, Category::Transport, "Spark intake",
                cache::glmf_spark_intake_bytes(i.prefill_lanes, i.spark_ranks as u64,
                    cache::glmf_expert_rows(i.prefill_rows, i.decode_rows), i.cfg.hidden as u64), Basis::Formula));
            fixed.push(Demand::new(gpu, Category::Workspace, "probe prefill logits", i.full_prefill_logits, Basis::Formula));
        }
    }
    Ok(PlacementRequest {
        inventory: Inventory { gpus: i.gpus.iter().map(|&(capacity_bytes, baseline)| GpuBudget {
            capacity_bytes, headroom_bytes: i.headroom_bytes, baseline }).collect(), spark_ranks: i.spark_ranks, peer_access: ranks == 2 },
        pool: PoolPolicy::resolve(&i.gpus.iter().map(|g| g.0).collect::<Vec<_>>(), i.max_context,
            i.requested_pool, geometry.logical_unit_rows, i.spark_ranks == 0),
        layers, pool_overhead: geometry.ranks.iter().map(|g| g.pool_metadata_unit_bytes).collect(), fixed,
        movables: if i.drafter_bytes == 0 { vec![] } else { vec![Movable { id: MovableId::Drafter,
            parts: vec![Bytes2 { resident: i.drafter_bytes, staging: 0 }], allowed: vec![0], expert_arena: false }] },
        expert_workspace: i.expert_workspace, tp2_workspace: i.tp2_workspace, onboard: i.onboard,
        expert_gpus: usize::from(ranks == 1 || i.experts.iter().all(|cost| cost.tp2)),
        policy: LayerPolicy { default: if ranks == 2 { vec![LayerMode::HeadSplit] } else {
            vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] }, by_kind: vec![] },
        hops: HopSpec { row_bytes: 4 * i.cfg.hidden as u64 * 2, rows: i.prefill_rows.max(i.decode_rows),
            lanes: i.prefill_lanes, entry_gpu: 0, head_gpu: 0 }, executor: super::GLM5_FLASH,
    })
}

/// Exact header-only layer residency for the EXL3 and fp8moe loaders.
pub fn expert_costs(catalog: &crate::OfficialV41Catalog, tp2: bool) -> anyhow::Result<Vec<ExpertCost>> {
    use crate::formats::fp8_experts::{Fp8Projection, Slicing};
    let shape = catalog.routed_experts();
    let cost = |layer, tp, rank| -> anyhow::Result<Bytes2> {
        let resident = if let Some(exl3) = catalog.exl3() {
            exl3.residency(crate::V41Exl3Layer::Backbone(layer), tp, rank)?.device_arena_layout()?.1 as u64
        } else if let Some(fp8) = catalog.fp8() {
            Fp8Projection::ALL.iter().try_fold(0u64, |sum, &p| -> anyhow::Result<u64> {
                let (w, _) = fp8.slice_bytes_with(p, tp, rank, Slicing::Blocks(128))?;
                Ok(sum + (shape.experts * w).max(256) as u64 + fp8.scale_region_bytes_with(p, tp, rank, Slicing::Blocks(128))?.max(256) as u64)
            })?
        } else { anyhow::bail!("GLM Flash routed experts need EXL3 or fp8moe") };
        Ok(Bytes2 { resident, staging: 0 })
    };
    (shape.first_layer..shape.layers).map(|layer| Ok(ExpertCost { whole: cost(layer, 1, 0)?,
        half: if tp2 { [cost(layer, 2, 0)?, cost(layer, 2, 1)?] } else { [Bytes2::default(); 2] },
        tp2, spark_ok: true })).collect()
}

/// Exact compiled package arenas; missing packages fail closed, never become zero scratch.
pub fn expert_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize) -> anyhow::Result<u64> {
    anyhow::ensure!([1, 2].contains(&tp), "GLM Flash RTX experts need TP1 or TP2");
    anyhow::ensure!((1..=4096).contains(&rows), "GLM Flash expert capacity must be 1..=4096 rows");
    let shape = catalog.routed_experts();
    let lib = inventory::image_lib(manifest);
    if let Some(exl3) = catalog.exl3() {
        let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
        let stem = format!("exl3-glmf-k{tiers}");
        let parent = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json")).parent().unwrap();
        let root = [parent.join("exl3").join(&stem), lib.join("exl3").join(&stem)]
            .into_iter().find(|p| p.join(format!("rtx-tp{tp}")).is_dir())
            .ok_or_else(|| anyhow::anyhow!("missing {stem}/rtx-tp{tp} package"))?;
        let manifests = inventory::exl3_capacities(rows).into_iter().map(|capacity| -> anyhow::Result<serde_json::Value> {
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join(format!("rtx-tp{tp}/m{capacity}/v41_exl3.json")))?)?;
            anyhow::ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
                && value["intermediate"].as_u64() == Some((shape.intermediate / tp) as u64)
                && value["experts"].as_u64() == Some(shape.experts as u64), "GLM Flash EXL3 package geometry mismatch");
            Ok(value)
        }).collect::<anyhow::Result<Vec<_>>>()?;
        return Ok(cache::exl3_workspace_bytes(&manifests, true)? + rows.max(1) * shape.hidden as u64 * if tp == 2 { 4 } else { 2 });
    }
    let fp8 = catalog.fp8().ok_or_else(|| anyhow::anyhow!("GLM Flash fp8moe catalog missing"))?;
    let a4 = "fp8-glmf-nvfp4a4";
    let name = if fp8.format() == crate::formats::fp8_experts::ExpertFormat::Nvfp4 && lib.join("fp8").join(a4).is_dir() {
        a4.to_string()
    } else { format!("fp8-glmf{}", fp8.format().package_suffix()) };
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(lib.join("fp8").join(&name).join("manifest.json"))?)?;
    let scratch = inventory::fp8moe_scratch_bytes(&value, &format!("tp{tp}"), rows)
        .ok_or_else(|| anyhow::anyhow!("missing GLM Flash tp{tp} scratch for {rows} rows"))?;
    let bf16 = lib.join("fp8").join(format!("{name}-bf16"));
    let sibling = if bf16.join(format!("tp{tp}")).is_dir() {
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(bf16.join("manifest.json"))?)?;
        inventory::fp8moe_scratch_bytes(&manifest, &format!("tp{tp}"), rows)
            .ok_or_else(|| anyhow::anyhow!("missing BF16-input GLM Flash scratch"))?
    } else { 0 };
    Ok(scratch.max(sibling) + if tp == 2 { rows * shape.hidden as u64 * 2 } else { 0 })
}

/// Rebuild startup graphs at the admitted pool rather than a pre-admission guess.
/// Explicit graph budgets retain their growth-only lifetime.
pub fn solve_with_graphs(i: &GlmfInputs<'_>, target: u64, automatic_context: bool,
    sms: usize, rtx_experts: bool) -> Result<(Placement, GraphSet), PlacementError> {
    let mut candidate = i.requested_pool.unwrap_or(target.max(i.max_context));
    if i.requested_pool.is_none() && (i.onboard.layers(i.experts.len()).is_some() || i.spark_ranks == 0) {
        let mut req = request(i)?;
        if !rtx_experts { req.expert_gpus = 0; }
        candidate = solve(&req)?.pool_tokens;
    }
    loop {
        let mut trial = i.clone();
        if i.graphs.lifetime == Lifetime::Startup {
            let context = glmf_graphs::admitted_graph_context(i.max_context as usize, candidate as usize, automatic_context);
            trial.graphs = startup_graphs(i.cfg, i.layers, context, candidate as usize, i.sequences as usize,
                i.drafter_bytes > 0, i.gpus.len(), i.decode_rows as usize, sms);
        }
        let mut req = request(&trial)?;
        if !rtx_experts { req.expert_gpus = 0; }
        req.pool.target = req.pool.target.max(target).min(candidate);
        req.pool.ceiling = candidate;
        let placed = solve(&req)?;
        if i.requested_pool.is_some() || i.graphs.lifetime != Lifetime::Startup || placed.pool_tokens >= candidate {
            return Ok((placed, trial.graphs));
        }
        candidate = placed.pool_tokens;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn startup_graphs(cfg: &GlmNextConfig, layers: usize, context: usize, pool: usize, sequences: usize,
    speculation: bool, gpus: usize, decode_rows: usize, sms: usize) -> GraphSet {
    let counts = glmf_graphs::serving_graph_counts(context, pool, cfg.dense_context(), sequences, speculation,
        layers, gpus == 2, &glmf_graphs::DecodeBuckets::new(glmf_graphs::verify_budget(decode_rows, sms)));
    let ranks = counts.iter().enumerate().map(|(rank, &count)| {
        let role = if gpus == 1 { 0 } else { rank + 1 };
        let measured = glmf_graphs::measured_graph_bytes(count, role);
        let margin = (measured * glmf_graphs::GRAPH_MARGIN_PERCENT).div_ceil(100)
            .max(glmf_graphs::GRAPH_RANK_MARGIN_BYTES);
        GraphRank { executables: count, bytes: measured + margin, margin }
    }).collect();
    GraphSet { ranks, shapes: counts[0] / (layers + 1) as u64, lifetime: Lifetime::Startup }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(cfg: &GlmNextConfig, ranks: usize, capacity: u64) -> GlmfInputs<'_> {
        let context = ArchContext::coordinator(capacity, None).context_bytes;
        GlmfInputs {
            cfg, layers: cfg.layers,
            gpus: (0..ranks).map(|rank| (capacity, Baseline::Planned {
                context_bytes: context + loaded_code("glmf", "exl3", ranks == 2, rank as u8).unwrap().bytes,
                loaded_bytes: 0 })).collect(), pending_code: vec![0; ranks], headroom_bytes: 1 << 30,
            spark_ranks: 4, prefill_lanes: 2, prefill_rows: 4096, decode_rows: 64, partial_bytes: 2,
            max_context: 262_144, sequences: 8, state_slots: 8, mark_slots: 16, pool_marks: false,
            index: cache::GlmfIndexCache::Keys, kda_state_bytes: 4, shared_replay: false,
            representation: GlmfRepresentation::default(),
            resident: (0..ranks).map(|rank| GlmfResidentRank {
                embedding: if rank == 0 { 1 << 30 } else { 0 }, weights: 5 << 30 }).collect(),
            router_replica_bytes: 64 << 20, workspace: vec![3 << 30; ranks],
            graphs: startup_graphs(cfg, cfg.layers, 262_144, if capacity <= 34 << 30 { 1 << 20 } else { 2 << 20 },
                8, true, ranks, 64, 188),
            experts: cfg.dense.iter().filter(|&&dense| !dense).map(|_| ExpertCost {
                whole: Bytes2 { resident: 2 << 30, staging: 0 },
                half: [Bytes2 { resident: 1 << 30, staging: 0 }; 2], tp2: ranks == 2, spark_ok: true }).collect(),
            expert_workspace: 128 << 20, tp2_workspace: [128 << 20; 2], drafter_bytes: 3 << 30,
            requested_pool: None, onboard: default_onboard(), full_prefill_logits: 0,
        }
    }

    #[test]
    fn planner_equals_runtime_glm5_flash() {
        let mut config = crate::plan::testing::glm5_flash_config(45);
        config["text_config"]["layer_types"] = serde_json::json!((0..45).map(|l|
            if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        for (ranks, capacity) in [(1, 32 << 30), (1, inventory::PRO_TOTAL_BYTES), (2, inventory::PRO_TOTAL_BYTES)] {
            let planned = inputs(&cfg, ranks, capacity);
            let planned_request = request(&planned).unwrap();
            assert_eq!(planned_request.layers.iter().filter(|l| l.kind == AttentionClass::Dsa).count(), 11);
            let p = solve(&planned_request).unwrap();
            let mut measured = planned.clone();
            for rank in 0..ranks {
                let context = ArchContext::coordinator(capacity, None).context_bytes;
                let code = loaded_code("glmf", "exl3", ranks == 2, rank as u8).unwrap();
                let sample = context + code.bytes / 3;
                measured.gpus[rank].1 = Baseline::Measured { free_bytes: capacity - sample };
                measured.pending_code[rank] = code.pending(sample, context);
            }
            let r = solve(&request(&measured).unwrap()).unwrap();
            assert_eq!(p.pool_tokens, r.pool_tokens);
            assert_eq!(p.onboard_layers, r.onboard_layers);
            assert_eq!(p.layers, r.layers);
            assert_eq!(p.tp2, r.tp2);
            for rank in 0..ranks {
                let bytes = |placed: &Placement, group: &str| placed.items[rank].iter()
                    .filter(|i| i.group == group).map(|i| i.bytes).sum::<u64>();
                for group in ["graphs", inventory::GRAPH_GROWTH, "records"] {
                    assert_eq!(bytes(&p, group), bytes(&r, group));
                }
                assert_eq!(bytes(&p, "graphs"), planned.graphs.at_ready(rank));
                assert_eq!(bytes(&p, inventory::GRAPH_GROWTH), planned.graphs.growth(rank));
                let charged = |placed: &Placement| placed.items[rank].iter().map(|i| i.bytes).sum::<u64>();
                let baseline = |input: &GlmfInputs<'_>| capacity - GpuBudget {
                    capacity_bytes: capacity, headroom_bytes: 0, baseline: input.gpus[rank].1 }.available();
                assert_eq!(baseline(&planned) + charged(&p), baseline(&measured) + charged(&r));
            }
            if ranks == 2 {
                assert!(p.tp2.is_some());
                assert!(!p.layers.iter().any(|l| matches!(l.experts, ExpertHome::RtxWhole { .. })));
            }
        }
    }

    #[test]
    fn glm5_flash_header_costs_and_fp8_package_scratch_are_static() {
        use crate::plan::testing::{glm5_flash_config, glm5_flash_tensors, write_snapshot};
        use crate::formats::fp8_experts::{Fp8Projection, Slicing};
        let dir = tempfile::tempdir().unwrap();
        let config = glm5_flash_config(2);
        write_snapshot(dir.path(), &config, &glm5_flash_tensors(&config), None);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let fp8 = catalog.fp8().unwrap();
        let costs = expert_costs(&catalog, true).unwrap();
        let exact = |tp, rank| Fp8Projection::ALL.iter().map(|&proj| {
            let (bytes, _) = fp8.slice_bytes_with(proj, tp, rank, Slicing::Blocks(128)).unwrap();
            (bytes * fp8.shape().experts).max(256) as u64
                + fp8.scale_region_bytes_with(proj, tp, rank, Slicing::Blocks(128)).unwrap().max(256) as u64
        }).sum::<u64>();
        assert_eq!(costs[0].whole.resident, exact(1, 0));
        assert_eq!(costs[0].half.map(|c| c.resident), [exact(2, 0), exact(2, 1)]);
        let share = dir.path().join("share");
        let package = dir.path().join("lib/fp8/fp8-glmf");
        std::fs::create_dir_all(&share).unwrap();
        std::fs::create_dir_all(&package).unwrap();
        let path = share.join("PROGRAMS.json");
        assert!(expert_workspace(&catalog, Some(&path), 4096, 2).is_err());
        std::fs::write(package.join("manifest.json"), serde_json::json!({"layouts": {
            "tp1": {"capacities":[{"capacity":4096,"scratch_bytes":12345}]},
            "tp2": {"capacities":[{"capacity":4096,"scratch_bytes":67890}]}}}).to_string()).unwrap();
        assert_eq!(expert_workspace(&catalog, Some(&path), 4096, 1).unwrap(), 12345);
        assert_eq!(expert_workspace(&catalog, Some(&path), 4096, 2).unwrap(), 67890 + 4096 * 4096 * 2);
        assert!(expert_workspace(&catalog, Some(&path), 4097, 2).is_err());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
        assert_eq!(crate::families::glm5_flash::resident::router_replica_bytes(&checkpoint, &cfg, 2).unwrap(),
            288 * 4096 * 2 + 288 * 4);
    }

    #[test]
    fn glm5_flash_graph_inventory_rebuilds_at_the_smaller_pool() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let input = inputs(&cfg, 1, 32 << 30);
        let (placed, graphs) = solve_with_graphs(&input, 1 << 20, true, 170, true).unwrap();
        assert!(placed.pool_tokens < 1 << 20);
        let context = glmf_graphs::admitted_graph_context(input.max_context as usize, placed.pool_tokens as usize, true);
        let expected = startup_graphs(&cfg, cfg.layers, context, placed.pool_tokens as usize,
            input.sequences as usize, true, 1, 64, 170);
        assert_eq!(graphs.ranks, expected.ranks);
    }

    #[test]
    fn glm5_flash_split_without_tp2_cannot_place_whole_experts() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        for expert in &mut input.experts { expert.tp2 = false; }
        let placed = solve(&request(&input).unwrap()).unwrap();
        assert_eq!(placed.onboard_layers, 0);
        input.onboard = Onboard::Layers(1);
        assert!(solve(&request(&input).unwrap()).is_err());
    }
}
