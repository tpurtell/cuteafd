//! MiMo's planner consumes the same allocation descriptions as runtime admission.
use super::*;
use crate::families::mimo_v2::{admission::*, capacity::*, resident::MimoResidentOptions,
    projection::MimoProjectionRepresentation as R, weight_policy::*, *};
use anyhow::{Context, Result};

pub(super) fn profile(checkpoint: &super::super::Checkpoint, options: &LayoutOptions,
    ranks: usize, spark_ranks: usize, rows: u64, concurrency: u64, context: u64,
    manifest: Option<&serde_json::Value>) -> Result<MimoCapacityProfiles> {
    let cfg = MimoV2Config::from_hf(&checkpoint.config)?;
    let qualified = default_policy(checkpoint, &cfg) == MimoDefaultPolicy::Fp8;
    let source = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .map(|t| if t.meta.dtype == cuteafd_core::DType::Bf16 && !qualified { R::Bf16 } else { R::Fp8 });
    let output_formats: std::collections::BTreeMap<_, _> = (0..cfg.layers).filter_map(|layer| {
        let name = format!("model.layers.{layer}.self_attn.o_proj.weight");
        source(&name).map(|r| (name, r))
    }).collect();
    let fp8_output = output_formats.values().any(|&r| r == R::Fp8);
    let lanes = resolve_transport_lanes(spark_ranks > 0, cfg.program_family()?,
        if options.prefill_lanes > 0 { Some(options.prefill_lanes.to_string()) }
        else { std::env::var("CUTEAFD_MIMO_PREFILL_LANES").ok() }.as_deref())?;
    let rings = options.mimo_rings.max(concurrency);
    let cache = crate::serving_capacity::mimo_cache_geometry(&cfg, cfg.layers, ranks, MimoKvCache::Int8, 0)?;
    let dir = checkpoint.snapshot.join("dflash");
    let draft = if dir.join("dflash_draft_model.safetensors").is_file() {
        Some(draft_config::DflashConfig::read(&dir)?)
    } else { None };
    let draft_mark = if options.mimo_prefix_draft {
        if let Some(draft) = &draft { draft_representation::mimo_draft_mark_bytes(draft.layers as u64, draft.kv_width() as u64)? }
        else if dir.join("config.json").is_file() {
            let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
            super::mimo_draft_prefix_bytes(&config, 1, rings).map_err(anyhow::Error::msg)?.0
        } else { 0 }
    } else { 0 };
    let mark = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum::<u64>() + draft_mark;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_for(
        concurrency, options.mimo_prefix_entries, mark, options.mimo_prefix_mark_bytes));
    let graphs = decode_graph::MimoDecodeGraphPlan::new(cfg.layers, ranks, 64, options.mimo_decode_graphs)?;
    let output = if options.full_prefill_logits { MimoPrefillOutput::AllRows } else { MimoPrefillOutput::LastRow };
    let mut runtime = Vec::new();
    for rank in 0..ranks {
        let mut additional = Vec::new();
        additional.extend(transport_reservations(&cfg, ranks, rank, rows as usize, lanes, spark_ranks)?);
        if rank == 0 {
            if spark_ranks == 0 {
                let lib = crate::placement::inventory::image_lib(options.workspace_manifest.as_deref());
                if let Some(path) = &options.mimo_expert_manifest {
                    let value = serde_json::from_slice(&std::fs::read(path)?)?;
                    let bytes = crate::placement::inventory::fp8moe_scratch_bytes(&value, "tp1", rows.max(64))
                        .context("MiMo local expert manifest lacks selected capacity")?;
                    additional.push(cuteafd_core::serving_capacity::MemoryReservation {
                        name: "experts.local_scratch".into(), bytes });
                } else if let Ok(catalog) = crate::read_expert_catalog(&checkpoint.snapshot) {
                    if let Some(tensors) = catalog.fp8() {
                        if let Some((_, bytes)) = crate::placement::inventory::fp8moe_package_scratch(
                            &lib, cfg.program_family()?, tensors.format(), rows.max(64)) {
                            additional.push(cuteafd_core::serving_capacity::MemoryReservation {
                                name: "experts.local_scratch".into(), bytes });
                        }
                    }
                }
            }
            additional.extend(sampling_reservations(cfg.vocab_size)?);
            if let Some(draft) = &draft {
                let sequences = options.draft_sequences.max(concurrency);
                let slots = draft_representation::mimo_draft_context_slots(sequences, rings, options.draft_context_slots);
                let capacity = draft_representation::MimoDraftCapacity::new(slots as usize, sequences as usize, draft.block)?;
                let headers = crate::read_safetensors_metadata(&dir.join("dflash_draft_model.safetensors"))?;
                let costs = draft_reservations_with(draft, &headers,
                    if options.mimo_draft_bf16 { draft_representation::MimoDraftRepresentation::Bf16Only }
                        else { draft_representation::MimoDraftRepresentation::Fp8Only }, capacity,
                    |shape| draft_scratch_bytes(shape, options.physical_sms.unwrap_or_else(||
                        crate::placement::inventory::ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms) as usize))?;
                additional.extend(costs.steady);
                if prefill_lane_taps(lanes, rows as usize, 0, output) {
                    additional.push(cuteafd_core::serving_capacity::MemoryReservation {
                        name: "draft.prefill_first_lane_taps".into(), bytes: draft.prefill_lane_tap_bytes()? as u64 });
                }
            }
            additional.extend(draft_prefix_reservations(options.mimo_prefix_draft, draft_mark, marks, rings)?);
        }
        let workspaces = workspace_options(&cfg, ranks, rank, rows as usize, context as usize,
            lanes, spark_ranks > 0, MimoKvCache::Int8, 0,
            if options.full_prefill_logits { MimoPrefillOutput::AllRows } else { MimoPrefillOutput::LastRow },
            fp8_output, |name| {
                let Some(manifest) = manifest else { return Ok(0); };
                let program = manifest["programs"].as_array().context("PROGRAMS.json lacks programs")?
                    .iter().find(|p| p["name"].as_str() == Some(name))
                    .with_context(|| format!("MiMo requires program {name}"))?;
                Ok(program["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0))
            })?;
        additional.extend(graphs.reservations(rank, decode_graph::MIMO_GRAPH_EXEC_BOUND_BYTES,
            decode_graph::MIMO_GRAPH_DRIVER_MARGIN_BYTES)?);
        debug_assert_eq!(graphs.graph_set(decode_graph::MIMO_GRAPH_EXEC_BOUND_BYTES,
            decode_graph::MIMO_GRAPH_DRIVER_MARGIN_BYTES).bytes(rank), additional.iter()
                .filter(|r| r.name.starts_with("runtime.decode_graph")).map(|r| r.bytes).sum::<u64>());
        runtime.push(MimoRankRuntime { device: rank as u32, workspaces,
            loading_additional: Vec::new(), additional });
    }
    let host_embedding = (options.host_embedding || (!options.force_gpu_embedding && options.rtx_bytes[0] <= 32 * GIB))
        && checkpoint.require_untied_embedding("model.embed_tokens.weight").is_ok();
    Ok(mimo_capacity_profiles(checkpoint, &cfg, MimoResidentOptions {
        layers: cfg.layers, coordinator_ranks: ranks, checkpoint_tp: checkpoint_tp(&checkpoint.snapshot)?,
        native_mtp_layers: 0, gpu_embedding: !host_embedding, head_format: source("lm_head.weight").unwrap_or(R::Bf16), output_formats,
    }, &MimoCapacityOptions { checkpoint_max_context_tokens:
        crate::serving_capacity::checkpoint_context_limit(&checkpoint.config)?
            .context("MiMo checkpoint lacks max_position_embeddings")?,
        max_context_tokens: context, rings, mark_slots: marks, kv_cache: MimoKvCache::Int8,
        ranks: runtime, host_prefix_bytes: 0 })?)
}

pub(super) fn apply(checkpoint: &super::super::Checkpoint, options: &LayoutOptions,
    devices: &mut [DeviceLayout], spark_ranks: usize, rows: u64, concurrency: u64,
    context: u64, manifest: Option<&serde_json::Value>) -> Result<u64> {
    let profiles = profile(checkpoint, options, devices.len(), spark_ranks, rows, concurrency, context, manifest)?;
    let probe = spark_intake_probe_bytes(spark_ranks > 0, std::env::var("CUTEAFD_SPARK_INTAKE").ok().as_deref())?;
    for (rank, device) in devices.iter_mut().enumerate() {
        device.capacity_bytes = options.rtx_bytes[rank].saturating_sub(headroom_bytes(
            options.rtx_bytes[rank], rank == 0 && probe > 0).max(options.headroom_bytes));
    }
    for (device, rank) in devices.iter_mut().zip(&profiles.steady.devices) {
        device.items.retain(|i| matches!(i.category, Category::Tables | Category::Reserved | Category::Experts | Category::Runtime)
            || i.group.starts_with("vision") || i.group.starts_with("audio"));
        for cost in &rank.reservations {
            let name = cost.name.as_str();
            let category = reservation_category(name);
            let group = match name { "state.active_rings" => "state", "prefix.exact_mark_arena" => "marks", "prefix.dflash_context_marks" => "DFlash context marks",
                "prefix.dflash_valid_floor_transfer" => "DFlash valid-floor transfer", _ => name };
            device.items.push(Item::new(category, group, "", cost.bytes, Basis::Formula));
        }
    }
    let unit = profiles.steady.pool_unit_rows;
    let target = if options.rtx_bytes.iter().any(|&total| total <= 32 * GIB) { (1 << 20).max(context) }
        else { cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS.max(context) };
    let units = options.pool_tokens.filter(|&n| n > 0).map(|n| n.div_ceil(unit)).unwrap_or_else(|| {
        devices.iter().zip(&profiles.steady.devices).map(|(d, r)|
            (d.free_bytes().max(0) as u64) / r.pool_unit_bytes.max(1)).min().unwrap_or(0).min(target.div_ceil(unit))
    });
    let cfg = MimoV2Config::from_hf(&checkpoint.config)?;
    let cache = crate::serving_capacity::mimo_cache_geometry(&cfg, cfg.layers, devices.len(), MimoKvCache::Int8, 0)?;
    for ((device, rank), storage) in devices.iter_mut().zip(&profiles.steady.devices).zip(&cache.ranks) {
        let kv_unit = storage.persistent_unit_bytes + storage.pool_metadata_unit_bytes;
        device.items.push(Item::new(Category::Kv, "records", "", units * kv_unit, Basis::Formula));
        device.items.push(Item::new(Category::Workspace, "pool page tables", "", units * (rank.pool_unit_bytes - kv_unit), Basis::Formula));
        device.kv_tokens = units * unit;
    }
    Ok(units * unit)
}

pub(super) fn reservation_category(name: &str) -> Category {
    if name.contains("embed_tokens") { Category::Embedding }
    else if name == "draft.fp8_scratch" { Category::Workspace }
    else if name.starts_with("draft.") { Category::Drafter }
    else if name.starts_with("workspace.") || name.starts_with("decode.") || name.starts_with("prefill") || name.starts_with("sampling.")
        || name == "transport.spark_intake_planes" || name == "state.prefill_kv_wide" { Category::Workspace }
    else if name.starts_with("transport.") { Category::Transport }
    else if name.starts_with("prefix.") { Category::Prefix }
    else if name.starts_with("state.") || name.starts_with("context.") { Category::Kv }
    else if name == "experts.local_scratch" { Category::Workspace }
    else if name.starts_with("experts.") { Category::Experts }
    else if name.starts_with("runtime.") || name.starts_with("graphs.") { Category::Runtime }
    else { Category::Weights }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{checkpoint::Checkpoint, testing::*};

    #[test]
    fn planner_equals_runtime_admission_flash_pro_one_two_rtx() {
        for pro in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let mut config = if pro { mimo_pro_config() } else { mimo_flash_mopd_config() };
            config["max_position_embeddings"] = serde_json::json!(1 << 20);
            write_snapshot(dir.path(), &config, &if pro { mimo_pro_tensors() } else { mimo_flash_mopd_tensors() }, Some(if pro { 8 } else { 4 }));
            let checkpoint = Checkpoint::open(dir.path()).unwrap();
            let cfg = MimoV2Config::from_hf(&config).unwrap();
            for ranks in [1, 2] {
                for graphs in [false, true] {
                    let opts = LayoutOptions { rtx_bytes: vec![95 * GIB; ranks], concurrency: 4,
                        pool_tokens: Some(32768), mimo_decode_graphs: graphs, ..Default::default() };
                    let planned = profile(&checkpoint, &opts, ranks, 4, 4096, 4, 32768, None).unwrap();
                    let lanes = resolve_transport_lanes(true, cfg.program_family().unwrap(), None).unwrap();
                    let cache = crate::serving_capacity::mimo_cache_geometry(&cfg, cfg.layers, ranks, MimoKvCache::Int8, 0).unwrap();
                    let mark_bytes = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum();
                    let slots = cuteafd_core::prefix::mark_slots_for(4, opts.mimo_prefix_entries, mark_bytes, opts.mimo_prefix_mark_bytes);
                    let graph = decode_graph::MimoDecodeGraphPlan::new(cfg.layers, ranks, 64, graphs).unwrap();
                    let runtime = (0..ranks).map(|rank| {
                        let mut additional = transport_reservations(&cfg, ranks, rank, 4096, lanes, 4).unwrap();
                        if rank == 0 { additional.extend(sampling_reservations(cfg.vocab_size).unwrap()); }
                        additional.extend(graph.reservations(rank, decode_graph::MIMO_GRAPH_EXEC_BOUND_BYTES,
                            decode_graph::MIMO_GRAPH_DRIVER_MARGIN_BYTES).unwrap());
                        MimoRankRuntime { device: rank as u32, loading_additional: vec![], additional,
                            workspaces: workspace_options(&cfg, ranks, rank, 4096, 32768, lanes, true,
                                MimoKvCache::Int8, 0, MimoPrefillOutput::LastRow, pro, |_| Ok(0)).unwrap() }
                    }).collect();
                    let admitted = mimo_capacity_profiles(&checkpoint, &cfg, MimoResidentOptions {
                        layers: cfg.layers, coordinator_ranks: ranks, checkpoint_tp: if pro { 8 } else { 4 },
                        native_mtp_layers: 0, gpu_embedding: true, head_format: if pro { R::Fp8 } else { R::Bf16 },
                        output_formats: (0..cfg.layers).map(|l| (format!("model.layers.{l}.self_attn.o_proj.weight"),
                            if pro { R::Fp8 } else { R::Bf16 })).collect(),
                    }, &MimoCapacityOptions { checkpoint_max_context_tokens: 1 << 20, max_context_tokens: 32768,
                        rings: 16, mark_slots: slots, kv_cache: MimoKvCache::Int8, ranks: runtime, host_prefix_bytes: 0 }).unwrap();
                    let mut devices: Vec<_> = (0..ranks).map(|rank| DeviceLayout {
                        kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: 95 * GIB,
                        items: vec![], kv_tokens: 0,
                    }).collect();
                    assert_eq!(apply(&checkpoint, &opts, &mut devices, 4, 4096, 4, 32768, None).unwrap(), 32768);
                    for rank in 0..ranks {
                        let total: u64 = admitted.steady.devices[rank].reservations.iter().map(|r| r.bytes).sum::<u64>()
                            + (32768 / admitted.steady.pool_unit_rows) * admitted.steady.devices[rank].pool_unit_bytes;
                        assert_eq!(devices[rank].used_bytes(), total, "mapped planner items equal admission total");
                        assert_eq!(planned.steady.devices[rank].reservations, admitted.steady.devices[rank].reservations,
                            "pro={pro}, ranks={ranks}, graphs={graphs}, rank={rank}");
                        assert_eq!(planned.steady.devices[rank].pool_unit_bytes, admitted.steady.devices[rank].pool_unit_bytes);
                    }
                }
            }
        }
    }

    #[test]
    fn recorded_inventory_pool_matches_runtime_admission() {
        use cuteafd_core::serving_capacity::{CapacityPolicy, DeviceMemory, resolve_capacity_with_startup_peaks};
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_flash_mopd_config(), &mimo_flash_mopd_tensors(), Some(4));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        for (total, baseline) in [(101_973_491_712u64, 586_416_128u64),
            (101_970_345_984, 586_416_128), (34_144_990_003, 596_901_888)] {
            let baseline = baseline + 35_651_584;
            let opts = LayoutOptions { rtx_bytes: vec![total], headroom_bytes: 0, concurrency: 4, ..Default::default() };
            let mut profile = profile(&checkpoint, &opts, 1, 2, 4096, 4, 1 << 20, None).unwrap();
            let cublas = crate::placement::inventory::ArchContext::for_device("sm_120", total).cublas_bytes;
            profile.steady.devices[0].reservations.push(cuteafd_core::serving_capacity::MemoryReservation {
                name: "runtime.cublas_first_gemm".into(), bytes: cublas,
            });
            let policy = CapacityPolicy { concurrency: 4, small_card_headroom: true,
                target_pool_tokens: if total <= 32 * GIB { 1 << 20 } else { 2 << 20 },
                max_context_tokens: Some(1 << 20), ..Default::default() };
            let admitted = resolve_capacity_with_startup_peaks(policy, &profile.steady,
                &[DeviceMemory { device: 0, total_bytes: total, baseline_free_bytes: total - baseline }],
                &[(0, 64 << 20)]).unwrap();
            let mut devices = vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0, capacity_bytes: total,
                items: vec![Item::new(Category::Runtime, "context+modules", "", baseline + cublas, Basis::Exact)], kv_tokens: 0 }];
            let pool = apply(&checkpoint, &opts, &mut devices, 2, 4096, 4, 1 << 20, None).unwrap();
            assert_eq!(pool, admitted.allocated_gpu_kv_tokens, "recorded total {total}");
            assert_eq!(devices[0].capacity_bytes, total - headroom_bytes(total, true));
        }
    }

    #[test]
    fn backend_slots_and_scope_categories_match_allocators() {
        assert_eq!(peer_slots(resolve_transport_lanes(false, "mimop", None).unwrap()), 8);
        assert_eq!(peer_slots(resolve_transport_lanes(true, "mimop", None).unwrap()), 12);
        assert_eq!(reservation_category("context.rope_tables"), Category::Kv);
        assert_eq!(reservation_category("state.prefill_kv_wide"), Category::Workspace);
        assert_eq!(reservation_category("prefill_first_lane.scratch"), Category::Workspace);
        assert_eq!(reservation_category("decode.page_table"), Category::Workspace);
        assert_eq!(reservation_category("draft.fp8_scratch"), Category::Workspace);
    }
}
