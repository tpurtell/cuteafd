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
    local_bytes: u64) -> Result<CapacityProfile> {
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
        if rank == 0 { reservations.push(reservation("RTX experts and loading peak", local_bytes)); }
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
            let profile = profile(&geometry, &memory, &shape, 0).unwrap();
            let headroom = profile.devices[0].reservations.iter()
                .find(|r| r.name == "workspace and headroom reserve").unwrap().bytes;
            assert_eq!(headroom, cuteafd_loader::serving_capacity::deepseek_v4_headroom_bytes(
                total, shape.reserve_bytes, 4 << 30, graph));
            assert_eq!(headroom, if gib == 31.8 { SMALL_CARD_HEADROOM_BYTES }
                else { shape.reserve_bytes - (4 << 30) - graph });
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
        let profile = profile(&geometry, &memory, &shape, 1 << 30).unwrap();
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
