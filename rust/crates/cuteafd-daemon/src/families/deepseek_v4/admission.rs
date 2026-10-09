//! Opt-in V4 capacity resolution, after coordinator weights and before caches.
use anyhow::{ensure, Context, Result};
use cuteafd_core::serving_capacity::{resolve_capacity, CapacityPolicy, CapacityProfile,
    ContextLimits, DeviceCosts, DeviceMemory, MemoryReservation, ResolvedCapacity};
use cuteafd_loader::serving_capacity::FamilyCacheGeometry;

pub(crate) struct Shape {
    pub sequences: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub max_context: usize,
    pub reserve_bytes: u64,
    pub prefix_bytes: Vec<u64>,
    pub workspace_bytes: Option<Vec<u64>>,
    pub peer_bytes: u64,
}

/// Modules and weights already appear in each sample's non-engine usage.
/// Reserve future workspaces, graph growth, cache carry, prefix marks and
/// local experts separately. Pool-sized lane page tables constrain the same
/// logical pool as the compressed records on every KV-owning GPU.
pub(crate) fn profile(geometry: &FamilyCacheGeometry, memory: &[DeviceMemory], shape: &Shape,
    local_bytes: &[u64]) -> Result<CapacityProfile> {
    ensure!(geometry.ranks.len() == memory.len() && shape.prefix_bytes.len() == memory.len(),
        "V4 admission needs a cache/prefix cost for each device");
    let costs = cuteafd_loader::plan::layout::family_costs("deepseek_v4");
    let mut devices = Vec::new();
    for (rank, (cache, sample)) in geometry.ranks.iter().zip(memory).enumerate() {
        let role = if memory.len() == 1 { 0 } else if rank == 0 { 1 } else { 2 };
        let reservation = |name: &str, bytes| MemoryReservation { name: name.into(), bytes };
        let table_bytes = (shape.prefill_rows as u64 * super::engine::PREFILL_LANES as u64
            + shape.decode_rows as u64).checked_mul(4).context("V4 pool lane table overflow")?;
        let state = cache.active_state_per_sequence_bytes.checked_mul(shape.sequences as u64)
            .and_then(|n| n.checked_add(cache.fixed_state_bytes))
            .and_then(|n| n.checked_add(cache.speculative_replay_bytes))
            .and_then(|n| table_bytes.checked_mul(shape.sequences as u64).and_then(|tables| n.checked_add(tables)))
            .context("V4 state overflow")?;
        let context = cache.context_table_bytes_per_token.checked_mul(shape.max_context as u64)
            .context("V4 context tables overflow")?;
        let workspace = shape.workspace_bytes.as_ref().and_then(|ranks| ranks.get(rank)).copied()
            .unwrap_or(costs.workspace_bytes[role] * shape.prefill_rows as u64 / 4096);
        let headroom = cuteafd_loader::serving_capacity::deepseek_v4_headroom_bytes(
            sample.total_bytes, shape.reserve_bytes, workspace, costs.graph_bytes[role]);
        let mut reservations = vec![
            reservation("active window/compressor state and partial units", state),
            reservation("RoPE context tables", context),
            reservation("prefix mark arena", shape.prefix_bytes[rank]),
            reservation("workspaces", workspace),
            reservation("future graphs", costs.graph_bytes[role]),
            reservation("workspace and headroom reserve", headroom),
        ];
        if memory.len() == 2 { reservations.push(reservation("peer exchange", shape.peer_bytes)); }
        reservations.push(reservation("RTX experts and loading peak", local_bytes.get(rank).copied().unwrap_or(0)));
        let pool_unit_bytes = cache.persistent_unit_bytes.checked_add(cache.pool_metadata_unit_bytes)
            .and_then(|n| n.checked_add(table_bytes)).context("V4 pool unit overflow")?;
        devices.push(DeviceCosts { device: sample.device, reservations, pool_unit_bytes });
    }
    Ok(CapacityProfile { context: ContextLimits { checkpoint_max_tokens: shape.max_context as u64,
        compiled_index_max_tokens: Some(shape.max_context as u64) }, pool_unit_rows: geometry.logical_unit_rows,
        devices, host_prefix_bytes: 0 })
}

#[cfg(test)]
pub(crate) fn resolve(profile: &CapacityProfile, memory: &[DeviceMemory], sequences: usize) -> Result<ResolvedCapacity> {
    resolve_pool(profile, memory, sequences, None)
}

pub(crate) fn resolve_pool(profile: &CapacityProfile, memory: &[DeviceMemory], sequences: usize,
    pool_tokens: Option<u64>) -> Result<ResolvedCapacity> {
    let capacity = resolve_capacity(CapacityPolicy { concurrency: u32::try_from(sequences)?,
        gpu_occupancy_percent: 100, pool_tokens, ..Default::default() }, profile, memory)?;
    ensure!(capacity.allocated_gpu_kv_tokens > 0, "V4 automatic pool has no allocation unit that fits");
    Ok(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::serving_capacity::{KvPlacement, RankCacheGeometry};

    #[test]
    fn runtime_and_planner_share_small_card_headroom() {
        use cuteafd_core::serving_capacity::{GpuMemoryBudget, SMALL_CARD_HEADROOM_BYTES};
        let geometry = FamilyCacheGeometry { logical_unit_rows: 256, placement: KvPlacement::SingleDevice,
            ranks: vec![RankCacheGeometry { persistent_unit_bytes: 1 << 20, ..Default::default() }] };
        let shape = Shape { sequences: 8, prefill_rows: 4096, decode_rows: 64, max_context: 1 << 20,
            reserve_bytes: 10 << 30, prefix_bytes: vec![0], workspace_bytes: Some(vec![4 << 30]), peer_bytes: 0 };
        let graph = cuteafd_loader::plan::layout::family_costs("deepseek_v4").graph_bytes[0];
        for gib in [31.8, 95.5] {
            let total = GpuMemoryBudget::from_gib(gib).unwrap().0;
            let memory = [DeviceMemory { device: 0, total_bytes: total, baseline_free_bytes: total }];
            let profile = profile(&geometry, &memory, &shape, &[]).unwrap();
            let headroom = profile.devices[0].reservations.iter()
                .find(|r| r.name == "workspace and headroom reserve").unwrap().bytes;
            assert_eq!(headroom, cuteafd_loader::serving_capacity::deepseek_v4_headroom_bytes(
                total, shape.reserve_bytes, 4 << 30, graph));
            assert_eq!(headroom, if gib == 31.8 { SMALL_CARD_HEADROOM_BYTES }
                else { shape.reserve_bytes - (4 << 30) - graph });
        }
    }

    #[test]
    fn planner_equals_runtime_pool_and_per_gpu_layers() {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::{plan::{plan, testing::write_v4_snapshot, ExpertPlacement, PlanOptions, layout::LayoutOptions},
            serving_capacity::{compiled_c128_width, deepseek_v4_cache_geometry, deepseek_v4_expert_cost,
                deepseek_v4_expert_exchange_bytes, deepseek_v4_native_workspace, deepseek_v4_peer_exchange_bytes,
                deepseek_v4_placement, deepseek_v4_workspace_geometry, deepseek_v4_workspace_scratch}};
        use serde_json::json;
        let dir = tempfile::tempdir().unwrap();
        write_v4_snapshot(dir.path());
        let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(dir.path(), 0).unwrap();
        let catalog = cuteafd_loader::read_expert_catalog(dir.path()).unwrap();
        let weights = (0..cfg.n_layers).map(|layer| deepseek_v4_expert_cost(&catalog, layer, false).unwrap())
            .collect::<Vec<_>>();
        let manifest = json!({"capacities": {"prefill_rows": 4096, "decode_rows": 64, "max_context": 1048576},
            "programs": [{"family": "dsv4f", "name": "dsv4f_sparse_mla_decode_c128_m64", "params": {"indexed_width": 8192}},
                {"name": "dsv4f_index_topk_decode_m64", "scratch_bytes_at_capacity": {"scratch": 8653824}},
                {"name": "dsv4f_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 558007296}}]});
        let path = dir.path().join("PROGRAMS.json");
        std::fs::write(&path, manifest.to_string()).unwrap();
        for budgets in [vec![24 << 30], vec![16 << 30, 24 << 30]] {
            for context in [16384, 1048576] {
                let planned = plan(dir.path(), &PlanOptions { placement: ExpertPlacement::from_spark_ranks(2),
                    layout: Some(LayoutOptions { rtx_bytes: budgets.clone(), context_tokens: context,
                        workspace_manifest: Some(path.clone()), ..Default::default() }), ..Default::default() }).unwrap();
                assert!(planned.placement_supported, "{:?}", planned.hints);
                let layout = planned.memory_layout.unwrap();
                let geometry = deepseek_v4_cache_geometry(&cfg, budgets.len(), 4096, 0).unwrap();
                let mark = geometry.ranks.iter().map(|r| r.retained_mark_bytes).sum::<u64>();
                let slots = cuteafd_engine::prefix::MarkArena::slots_for(8, 64, mark as usize, 2 << 30);
                let scratch = deepseek_v4_workspace_scratch(&manifest, "dsv4f", budgets.len() == 2, 4096, 64).unwrap();
                let workspace = deepseek_v4_workspace_geometry(&cfg, 4096, 64,
                    compiled_c128_width(&manifest, "dsv4f").unwrap() * 128, budgets.len(), scratch).unwrap();
                // Simulated CUDA samples include only already-resident weights/modules.
                let memory = layout.devices.iter().zip(&budgets).map(|(d, &total)| {
                    let loaded: u64 = d.items.iter().filter(|i| matches!(i.category,
                        Category::Weights | Category::Embedding | Category::Drafter)
                        || i.group == "context+modules").map(|i| i.bytes).sum();
                    DeviceMemory { device: d.index, total_bytes: total, baseline_free_bytes: total - loaded }
                }).collect::<Vec<_>>();
                let shape = Shape { sequences: 8, prefill_rows: 4096, decode_rows: 64, max_context: context as usize,
                    reserve_bytes: 10 << 30, prefix_bytes: geometry.ranks.iter().map(|r| r.retained_mark_bytes * slots as u64).collect(),
                    workspace_bytes: Some(workspace.iter().enumerate().map(|(rank, r)| r.fixed_device_bytes
                        + if rank == 0 { 2 * 2 * 4096 * cfg.dim as u64 * 2 } else { 0 }
                        + if budgets.len() == 2 { deepseek_v4_expert_exchange_bytes(cfg.dim as u64, 6, 4096, 64, rank).unwrap() } else { 0 }).collect()),
                    peer_bytes: deepseek_v4_peer_exchange_bytes(cfg.dim as u64, 4096, 64).unwrap() };
                let profile = profile(&geometry, &memory, &shape, &[]).unwrap();
                let available = profile.devices.iter().zip(&memory).map(|(d, m)| m.baseline_free_bytes
                    - d.reservations.iter().map(|r| r.bytes).sum::<u64>()).collect::<Vec<_>>();
                let runtime = deepseek_v4_placement(&available, &budgets,
                    &profile.devices.iter().map(|d| d.pool_unit_bytes).collect::<Vec<_>>(), geometry.logical_unit_rows,
                    None, None, 0, &weights, &[], deepseek_v4_native_workspace(cfg.dim as u64,
                        cfg.moe_inter_dim as u64, cfg.n_routed_experts as u64, 6, 4096).unwrap(), None).unwrap();
                assert_eq!(layout.pool_tokens, runtime.pool_tokens);
                for (rank, range) in runtime.ranks.iter().enumerate() {
                    let item = layout.devices[rank].items.iter().find(|i| i.group.starts_with("resident routed layers ")).unwrap();
                    assert_eq!(item.bytes, range.peak_bytes);
                    assert_eq!(item.group, format!("resident routed layers {}..{}", range.first, range.first + range.layers));
                }
                if budgets.len() == 2 { assert!(runtime.ranks[1].layers > 0); }
            }
        }
    }

    #[test]
    fn peer_pool_metadata_and_prefix_reservation_bound_auto_capacity() {
        let geometry = FamilyCacheGeometry { logical_unit_rows: 256, placement: KvPlacement::Replicated,
            ranks: vec![RankCacheGeometry { persistent_unit_bytes: 1 << 20,
                active_state_per_sequence_bytes: 1024, context_table_bytes_per_token: 512,
                ..Default::default() }; 2] };
        let memory: Vec<_> = [0, 7].into_iter().map(|device| DeviceMemory { device,
            total_bytes: 96 << 30, baseline_free_bytes: 16 << 30 }).collect();
        let shape = Shape { sequences: 8, prefill_rows: 4096, decode_rows: 64, max_context: 262144,
            reserve_bytes: 3 << 30, prefix_bytes: vec![0, 8 << 30], workspace_bytes: None, peer_bytes: 1 << 28 };
        let profile = profile(&geometry, &memory, &shape, &[1 << 30, 0]).unwrap();
        assert_eq!(profile.devices[0].pool_unit_bytes, (1 << 20) + (8192 + 64) * 4);
        let resolved = resolve(&profile, &memory, 8).unwrap();
        assert!(resolved.allocated_gpu_kv_tokens < cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS);
        assert_eq!(resolved.allocated_gpu_kv_tokens % 256, 0);
        assert_eq!(resolve_pool(&profile, &memory, 8, Some(257)).unwrap().allocated_gpu_kv_tokens, 512);
        assert!(resolve_pool(&profile, &memory, 8, Some(resolved.allocated_gpu_kv_tokens + 256)).is_err());
        let mut exhausted = memory.clone();
        exhausted[1].baseline_free_bytes = 1 << 30;
        assert!(resolve(&profile, &exhausted, 8).is_err());
    }
}
