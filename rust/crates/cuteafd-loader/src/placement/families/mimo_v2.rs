//! MiMo admission over its exact allocation profile. Both planner and runtime
//! supply the same retained representations; only the GPU baseline differs.
use crate::families::mimo_v2::{admission::headroom_bytes, MimoAttention, MimoKvCache, MimoV2Config};
use crate::placement::*;
use crate::serving_capacity::{mimo_cache_geometry, CacheGeometryError};
use cuteafd_core::memory_layout::{Basis, Category, Item};
use cuteafd_core::serving_capacity::{CapacityProfile, MemoryReservation, ResolvedCapacity, ResolvedDeviceCapacity};

#[derive(Clone)]
pub struct MimoInputs<'a> {
    /// True only when the operator left context automatic.
    pub automatic_context: bool,
    pub cfg: &'a MimoV2Config,
    pub profile: &'a CapacityProfile,
    pub layers: usize,
    pub kv: MimoKvCache,
    pub mtp_stages: usize,
    /// Physical totals and baselines, in logical rank order. MiMo samples
    /// before weights, so every retained representation is still a demand.
    pub gpus: Vec<(u64, Baseline)>,
    pub headroom_floor: u64,
    pub spark_ranks: usize,
    pub startup_probe_bytes: u64,
    pub context: u64,
    pub requested_pool: Option<u64>,
    /// Automatic pools under a fixed onboard stop at the target (no growth to free memory).
    pub pool_ceiling_at_target: bool,
    pub onboard: Onboard,
    /// Backbone order (including dense layers), exact whole/TP2 package costs.
    pub experts: Vec<Option<ExpertCost>>,
    pub expert_workspace: u64,
    pub tp2_workspace: [u64; 2],
    pub rows: u64,
    pub lanes: u64,
}

pub struct MimoSolution {
    pub request: PlacementRequest,
    pub placement: Placement,
    pub working_set: SparkWorkingSet,
}

impl<'a> MimoInputs<'a> {
    pub fn selected<'b>(&'b self, no_spark: &'b CapacityProfile, working_set: &SparkWorkingSet) -> MimoInputs<'b> {
        let remote = working_set.includes(WorkingSetComponent::SparkExperts);
        MimoInputs { profile: if remote { self.profile } else { no_spark },
            spark_ranks: working_set.selected_ranks().len(),
            lanes: if remote { working_set.prefill_lanes } else { 1 },
            startup_probe_bytes: if remote { self.startup_probe_bytes } else { 0 },
            ..self.clone() }
    }
}

/// Evaluate each contiguous RTX prefix with the fixed storage its Spark suffix
/// actually executes. Nothing is refunded after admission or recomputed by workers.
pub fn solve_working_set(inputs: &MimoInputs<'_>, no_spark: &CapacityProfile)
    -> Result<MimoSolution, CacheGeometryError> {
    let routed = inputs.experts.iter().enumerate().filter_map(|(layer, cost)| cost.map(|_| layer))
        .collect::<Vec<_>>();
    let counts = match inputs.onboard.layers(routed.len()) {
        Some(n) => vec![n],
        None if inputs.spark_ranks == 0 => vec![routed.len()],
        None => (0..=routed.len()).collect(),
    };
    let mut best: Option<MimoSolution> = None;
    let mut failure = None;
    for n in counts {
        if n > 0 && inputs.expert_workspace == 0 && inputs.tp2_workspace.iter().all(|&bytes| bytes == 0) {
            if failure.is_none() {
                failure = Some(PlacementError::Inventory("MiMo local experts need a matching BF16-input package").into());
            }
            continue;
        }
        let spark_layers = routed[n..].to_vec();
        if !spark_layers.is_empty() && inputs.spark_ranks == 0 {
            failure = Some(PlacementError::SparkFree { layers: routed.len(), placed: n }.into());
            continue;
        }
        let remote = !spark_layers.is_empty();
        let working_set = SparkWorkingSet { rtx_layers: routed[..n].to_vec(), spark_layers,
            spark_ranks: if remote { inputs.spark_ranks } else { 0 },
            prefill_lanes: if remote { inputs.lanes } else { 1 } };
        let selected = inputs.selected(no_spark, &working_set);
        let result = (|| {
            let mut request = request(&selected)?;
            request.onboard = Onboard::Layers(n);
            match inputs.onboard {
                Onboard::Auto if inputs.spark_ranks > 0 => {
                    request.pool.ceiling = request.pool.target;
                    // Spark Auto may clamp serving context to a smaller pool;
                    // fixed candidates must not introduce a new context floor.
                    if remote && request.pool.requested.is_none() { request.pool.floor = 0; }
                }
                Onboard::ExpertsFirst { pool_floor } => {
                    request.pool.floor = request.pool.floor.max(request.pool.requested.unwrap_or(pool_floor));
                    request.pool.ceiling = request.pool.target;
                    // Preserve the shared solver's remote zero-prefix fallback
                    // when context is larger than the experts-first floor.
                    if remote && n == 0 && request.pool.requested.is_none() {
                        request.pool.floor = pool_floor;
                    }
                }
                _ => {}
            }
            if inputs.pool_ceiling_at_target && request.pool.requested.is_none() {
                request.pool.ceiling = request.pool.target;
            }
            let placement = solve(&request)?;
            let actual = placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
                (assignment.experts == ExpertHome::Spark).then_some(layer)).collect::<Vec<_>>();
            let local = placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
                matches!(assignment.experts, ExpertHome::RtxTp2 | ExpertHome::RtxWhole { .. })
                    .then_some(layer)).collect::<Vec<_>>();
            if actual != working_set.spark_layers || local != working_set.rtx_layers
                || placement.onboard_layers != n {
                return Err(PlacementError::Inventory("MiMo solved prefix disagrees with its working set").into());
            }
            Ok::<_, CacheGeometryError>(MimoSolution { request, placement, working_set })
        })();
        match result {
            Ok(candidate) => {
                let score = |solution: &MimoSolution| match inputs.onboard {
                    Onboard::ExpertsFirst { .. } => (solution.placement.onboard_layers as u64,
                        solution.placement.pool_tokens),
                    _ => (solution.placement.pool_tokens.min(solution.request.pool.target),
                        solution.placement.onboard_layers as u64),
                };
                if best.as_ref().is_none_or(|current| score(&candidate) > score(current)) {
                    best = Some(candidate);
                }
            }
            Err(error) => if failure.is_none() { failure = Some(error); },
        }
    }
    best.ok_or_else(|| failure.unwrap_or_else(|| PlacementError::Inventory("MiMo has no feasible working set").into()))
}

/// Exact native weight/scale bytes, shared with the allocation path.
pub fn expert_layer_bytes(tensors: &crate::formats::fp8_experts::Fp8ExpertTensors,
    tp: usize, rank: usize, slicing: crate::formats::fp8_experts::Slicing) -> anyhow::Result<u64> {
    use crate::formats::fp8_experts::Fp8Projection;
    Fp8Projection::ALL.iter().try_fold(0u64, |total, &projection| {
        let (weights, _) = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
        let scales = tensors.scale_region_bytes_with(projection, tp, rank, slicing)?;
        total.checked_add((tensors.shape().experts as u64).checked_mul(weights as u64)
            .ok_or_else(|| anyhow::anyhow!("MiMo expert weight size overflow"))?)
            .and_then(|bytes| bytes.checked_add(scales as u64))
            .ok_or_else(|| anyhow::anyhow!("MiMo expert layer size overflow"))
    })
}

/// Select the same BF16-input package and whole-block slices as the runtime,
/// using only package manifests (no CUDA or native-library initialization).
pub fn package_directory(lib: &std::path::Path, cfg: &MimoV2Config,
    format: crate::formats::fp8_experts::ExpertFormat, tp: usize) -> Result<std::path::PathBuf, anyhow::Error> {
    let wire = spark_package_directory(lib, cfg, format, tp)?;
    let package = wire.parent().ok_or_else(|| anyhow::anyhow!("MiMo package parent"))?;
    let sibling = package.with_file_name(format!("{}-bf16", package.file_name()
        .and_then(|name| name.to_str()).ok_or_else(|| anyhow::anyhow!("MiMo package name"))?));
    Ok(if sibling.join(format!("tp{tp}")).is_dir() { sibling.join(format!("tp{tp}")) } else { wire })
}

/// Spark workers select their wire package before the optional BF16 sibling.
pub fn spark_package_directory(lib: &std::path::Path, cfg: &MimoV2Config,
    format: crate::formats::fp8_experts::ExpertFormat, tp: usize) -> Result<std::path::PathBuf, anyhow::Error> {
    use crate::formats::fp8_experts::ExpertFormat;
    let family = match (cfg.program_family()?, format) {
        ("mimo", ExpertFormat::Mxfp4) => "mimof",
        (family, _) => family,
    };
    let root = lib.join("fp8");
    let a4 = root.join(format!("fp8-{family}-nvfp4a4"));
    let package = if format == ExpertFormat::Nvfp4
        && std::env::var("CUTEAFD_NVFP4_ACTIVATIONS").as_deref() != Ok("a16") && a4.join(format!("tp{tp}")).is_dir() { a4 }
        else { root.join(format!("fp8-{family}{}", format.package_suffix())) };
    Ok(package.join(format!("tp{tp}")))
}

/// A missing implicit manifest makes local onboarding ineligible. A malformed
/// or explicitly selected manifest is an unsupported package, never a fallback.
pub fn package_manifest(directory: &std::path::Path, explicit: Option<&std::path::Path>)
    -> anyhow::Result<Option<serde_json::Value>> {
    let path = explicit.map(std::path::Path::to_path_buf).unwrap_or_else(||
        directory.parent().unwrap_or(directory).join("manifest.json"));
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).map_err(|error|
            anyhow::anyhow!("MiMo expert manifest {}: {error}", path.display()))?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && explicit.is_none() => Ok(None),
        Err(error) => Err(anyhow::anyhow!("MiMo expert manifest {}: {error}", path.display())),
    }
}

pub fn exact_package(directory: &std::path::Path,
    tensors: &crate::formats::fp8_experts::Fp8ExpertTensors, tp: usize, rank: usize)
    -> (std::path::PathBuf, crate::formats::fp8_experts::Slicing) {
    crate::formats::fp8_experts::exact_layout_directory(directory, tensors, tp, rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn route_exchange_matches_native_slot_alignment_and_rejects_overflow() {
        for rows in [1, 3, 64, 4096] {
            for lanes in [1, 2, 3] {
                let payload = (rows * 8 * 4u64).div_ceil(16) * 32;
                assert_eq!(route_exchange_bytes(rows, 8, lanes).unwrap(),
                    lanes * 2 * payload.max(256) + ((lanes * 2 + 1) * 16).max(256) + 256);
            }
        }
        for (rows, topk, lanes) in [(u64::MAX, 8, 3), (64, u64::MAX, 3), (64, 8, u64::MAX)] {
            assert!(route_exchange_bytes(rows, topk, lanes).is_err());
        }
        assert_eq!(cuteafd_core::memory_layout::Category::of_scope("peer-split"), Category::Transport);
    }
    #[test]
    fn selected_spark_working_set_drives_every_fixed_demand_and_policy() {
        use cuteafd_core::serving_capacity::{ContextLimits, DeviceCosts};
        let mut cfg = MimoV2Config::from_hf(&crate::plan::testing::mimo_pro_config()).unwrap();
        cfg.layers = 4;
        cfg.dense = vec![true, false, false, false];
        cfg.attention = vec![MimoAttention::Full; 4];
        for ranks in [1, 2] {
            let cache = mimo_cache_geometry(&cfg, 4, ranks, MimoKvCache::Int8, 0).unwrap();
            let profile = |spark| CapacityProfile {
                context: ContextLimits { checkpoint_max_tokens: 1 << 20, compiled_index_max_tokens: None },
                pool_unit_rows: cache.logical_unit_rows, host_prefix_bytes: 0,
                devices: cache.ranks.iter().enumerate().map(|(rank, storage)| DeviceCosts {
                    device: rank as u32, pool_unit_bytes: storage.persistent_unit_bytes + 256,
                    reservations: vec![MemoryReservation { name: if spark { "transport.spark_intake_planes" }
                        else { "workspace.local" }.into(), bytes: if spark { 2 << 30 } else { 1 << 30 } }],
                }).collect(),
            };
            let spark = profile(true);
            let local = profile(false);
            let cost = ExpertCost { whole: Bytes2 { resident: 1 << 30, staging: 0 },
                half: [Bytes2 { resident: 512 << 20, staging: 0 }; 2], tp2: ranks == 2, spark_ok: true };
            let inputs = MimoInputs { automatic_context: false, cfg: &cfg, profile: &spark, layers: 4, kv: MimoKvCache::Int8,
                mtp_stages: 0, gpus: vec![(96 << 30, Baseline::Measured { free_bytes: 90 << 30 }); ranks],
                headroom_floor: 0, spark_ranks: 6, startup_probe_bytes: 64 << 20, context: 256 << 10,
                requested_pool: Some(2 << 20), pool_ceiling_at_target: false, onboard: Onboard::Auto,
                experts: vec![None, Some(cost), Some(cost), Some(cost)], expert_workspace: if ranks == 1 { 1 << 20 } else { 0 },
                tp2_workspace: if ranks == 2 { [1 << 20; 2] } else { [0; 2] }, rows: 4096, lanes: 3 };
            let all = solve_working_set(&inputs, &local).unwrap();
            assert_eq!(all.placement.onboard_layers, 3);
            assert_eq!(all.working_set, SparkWorkingSet { rtx_layers: vec![1, 2, 3],
                spark_layers: vec![], spark_ranks: 0, prefill_lanes: 1 });
            assert!(all.working_set.includes(WorkingSetComponent::RtxExperts));
            assert!(!all.working_set.includes(WorkingSetComponent::SparkExperts));
            assert!(all.working_set.selected_ranks().is_empty());
            assert_eq!(all.request.fixed[0].group, "workspace.local");
            assert_eq!(all.request.inventory.gpus[0].headroom_bytes, headroom_bytes(96 << 30, false));
            let tight = MimoInputs { gpus: all.placement.items.iter().enumerate().map(|(rank, items)|
                (96 << 30, Baseline::Measured { free_bytes: items.iter().map(|item| item.bytes).sum::<u64>()
                    + all.request.inventory.gpus[rank].headroom_bytes + (512 << 20) })).collect(), ..inputs.clone() };
            assert_eq!(solve_working_set(&tight, &local).unwrap().placement.onboard_layers, 3,
                "the local-only branch fits without charging unused configured Spark storage");
            let mut overcharged = request(&tight).unwrap();
            overcharged.onboard = Onboard::Layers(3);
            assert!(solve(&overcharged).is_err(), "configured-peer costs would incorrectly reject all-local");
            let selected = inputs.selected(&local, &all.working_set);
            let configured_only = inputs.selected(&local, &SparkWorkingSet { spark_ranks: 6,
                prefill_lanes: 3, ..SparkWorkingSet::default() });
            assert_eq!((configured_only.spark_ranks, configured_only.lanes,
                configured_only.startup_probe_bytes), (0, 1, 0));
            assert!(std::ptr::eq(configured_only.profile, &local));
            assert_eq!(resolved_capacity(&selected, &all.request, &all.placement, 16, 16).unwrap().devices[0]
                .reservations[0].name, "workspace.local");
            for onboard in [Onboard::Layers(2), Onboard::Fraction(2.0 / 3.0)] {
                let partial = solve_working_set(&MimoInputs { onboard, ..inputs.clone() }, &local).unwrap();
                assert_eq!(partial.working_set.rtx_layers, vec![1, 2]);
                assert_eq!(partial.working_set.spark_layers, vec![3]);
                assert!(partial.working_set.includes(WorkingSetComponent::RtxExperts));
                assert!(partial.working_set.includes(WorkingSetComponent::SparkExperts));
                assert_eq!(partial.working_set.selected_ranks(), 0..6);
                assert_eq!((partial.working_set.spark_ranks, partial.working_set.prefill_lanes), (6, 3));
                assert_eq!(partial.request.fixed[0].group, "transport.spark_intake_planes");
                assert_eq!(partial.request.inventory.gpus[0].headroom_bytes, (96u64 << 30)
                    - (96u64 << 30) * 97 / 100 + (64 << 20));
            }
            for onboard in [Onboard::Layers(3), Onboard::Fraction(1.0), Onboard::ExpertsFirst { pool_floor: 262144 }] {
                let result = solve_working_set(&MimoInputs { onboard, ..inputs.clone() }, &local).unwrap();
                assert_eq!(result.placement.onboard_layers, 3);
                assert_eq!(result.working_set.spark_ranks, 0);
            }
            let fallback = solve_working_set(&MimoInputs { expert_workspace: 0, tp2_workspace: [0; 2],
                requested_pool: None, ..inputs.clone() }, &local).unwrap();
            assert_eq!(fallback.placement.onboard_layers, 0);
            assert!(fallback.working_set.rtx_layers.is_empty());
            assert!(!fallback.working_set.includes(WorkingSetComponent::RtxExperts));
            assert!(fallback.working_set.includes(WorkingSetComponent::SparkExperts));
            assert_eq!(fallback.working_set.spark_layers, vec![1, 2, 3]);
            assert_eq!(fallback.placement.pool_tokens, 2 << 20);
            let constrained = MimoInputs { context: 1 << 20, requested_pool: None,
                expert_workspace: 0, tp2_workspace: [0; 2],
                gpus: vec![(96 << 30, Baseline::Measured { free_bytes: 5 << 30 }); ranks], ..inputs.clone() };
            let shrunk = solve_working_set(&constrained, &local).unwrap();
            assert!(shrunk.placement.pool_tokens < constrained.context);
            assert!(shrunk.placement.pool_tokens > 0);
            assert_eq!(shrunk.request.pool.floor, 0);
            assert!(solve_working_set(&MimoInputs { requested_pool: Some(1 << 20),
                ..constrained }, &local).is_err(), "an explicit remote pool must never shrink");
            // Fixed zero onboarding: an automatic pool is capped at the target but
            // shrinks to what fits; an explicit pool stays strict; below the floor fails.
            let zero = |free: u64, requested_pool: Option<u64>| solve_working_set(&MimoInputs {
                onboard: Onboard::Layers(0), pool_ceiling_at_target: true, requested_pool, context: 256 << 10,
                expert_workspace: 0, tp2_workspace: [0; 2],
                gpus: vec![(96 << 30, Baseline::Measured { free_bytes: free }); ranks], ..inputs.clone() }, &local);
            let roomy = zero(90 << 30, None).unwrap().placement.pool_tokens;
            assert_eq!(roomy, 2 << 20, "automatic pool is capped at the target");
            let mut free = 90u64 << 30;
            let fitted = loop {
                free -= 1 << 30;
                let pool = zero(free, None).map(|s| s.placement.pool_tokens);
                if pool.as_ref().is_ok_and(|&p| p < 2 << 20) { break (free, pool.unwrap()); }
                assert!(pool.is_ok(), "no shrunk pool between the target and the floor");
            };
            assert!(fitted.1 >= 256 << 10 && fitted.1 < 2 << 20);
            assert!(matches!(zero(fitted.0, Some(2 << 20)).err(), Some(CacheGeometryError::ResidentTensor { what, .. })
                if what.starts_with("explicit KV pool of")), "an explicit pool stays strict (PoolDoesNotFit)");
            assert!(zero(1 << 30, None).is_err(), "below the floor still fails");
            assert!(solve_working_set(&MimoInputs { spark_ranks: 0, onboard: Onboard::Layers(2),
                ..inputs.clone() }, &local).is_err());
        }
    }

    #[test]
    fn drafter_movable_places_exactly_like_the_former_fixed_demands() {
        use cuteafd_core::serving_capacity::{ContextLimits, DeviceCosts};
        let mut cfg = MimoV2Config::from_hf(&crate::plan::testing::mimo_pro_config()).unwrap();
        cfg.layers = 4;
        cfg.dense = vec![true, false, false, false];
        cfg.attention = vec![MimoAttention::Full; 4];
        for ranks in [1, 2] {
            let cache = mimo_cache_geometry(&cfg, 4, ranks, MimoKvCache::Int8, 0).unwrap();
            let mut drafts = vec![("draft.weights.fp8_values", 1400u64 << 20), ("draft.fp8_scratch", 16 << 20),
                ("draft.layer0.k_ring", 32 << 20), ("workspace.local", 1 << 30)];
            let profile = CapacityProfile {
                context: ContextLimits { checkpoint_max_tokens: 1 << 20, compiled_index_max_tokens: None },
                pool_unit_rows: cache.logical_unit_rows, host_prefix_bytes: 0,
                devices: cache.ranks.iter().enumerate().map(|(rank, storage)| DeviceCosts {
                    device: rank as u32, pool_unit_bytes: storage.persistent_unit_bytes + 256,
                    reservations: drafts.iter().filter(|(name, _)| rank == 0 || !name.starts_with("draft."))
                        .map(|&(name, bytes)| MemoryReservation { name: name.into(), bytes }).collect(),
                }).collect(),
            };
            let cost = ExpertCost { whole: Bytes2 { resident: 1 << 30, staging: 0 },
                half: [Bytes2 { resident: 512 << 20, staging: 0 }; 2], tp2: ranks == 2, spark_ok: true };
            for onboard in [Onboard::Auto, Onboard::Layers(2)] {
                let inputs = MimoInputs { automatic_context: false, cfg: &cfg, profile: &profile, layers: 4, kv: MimoKvCache::Int8,
                    mtp_stages: 0, gpus: vec![(96 << 30, Baseline::Measured { free_bytes: 90 << 30 }); ranks],
                    headroom_floor: 0, spark_ranks: 6, startup_probe_bytes: 64 << 20, context: 256 << 10,
                    requested_pool: None, pool_ceiling_at_target: false, onboard,
                    experts: vec![None, Some(cost), Some(cost), Some(cost)],
                    expert_workspace: if ranks == 1 { 1 << 20 } else { 0 },
                    tp2_workspace: if ranks == 2 { [1 << 20; 2] } else { [0; 2] }, rows: 4096, lanes: 3 };
                let req = request(&inputs).unwrap();
                assert_eq!(req.movables.len(), 1);
                assert!(req.fixed.iter().all(|d| !d.group.starts_with("draft.")));
                assert_eq!(req.movables[0].conditional[0].demands.iter()
                    .map(|d| (d.category, d.group.as_str())).collect::<Vec<_>>(),
                    [(Category::Drafter, "draft.weights.fp8_values"), (Category::Workspace, "draft.fp8_scratch"),
                        (Category::Drafter, "draft.layer0.k_ring")]);
                crate::placement::draft::assert_same_as_fixed(&req);
            }
            drafts.retain(|(name, _)| !name.starts_with("draft."));
        }
    }

    #[test]
    fn implicit_missing_explicit_and_malformed_package_contract() {
        let dir = tempfile::tempdir().unwrap();
        let layout = dir.path().join("tp2");
        assert!(package_manifest(&layout, None).unwrap().is_none());
        assert!(package_manifest(&layout, Some(&dir.path().join("missing.json"))).is_err());
        let manifest = dir.path().join("manifest.json");
        std::fs::write(&manifest, "not json").unwrap();
        assert!(package_manifest(&layout, None).is_err());
        std::fs::write(&manifest, "{}").unwrap();
        assert!(package_manifest(&layout, None).unwrap().is_some());
    }
}

/// Ready route exchange includes receive slots, control and abort words on each rank.
pub fn route_exchange_bytes(rows: u64, topk: u64, lanes: u64) -> anyhow::Result<u64> {
    let section = rows.checked_mul(topk).and_then(|bytes| bytes.checked_mul(4))
        .ok_or_else(|| anyhow::anyhow!("MiMo route exchange overflow"))?;
    let slots = lanes.checked_mul(2).ok_or_else(|| anyhow::anyhow!("MiMo route slots overflow"))?;
    let slot = section.div_ceil(16).checked_mul(32).ok_or_else(|| anyhow::anyhow!("MiMo route slot overflow"))?;
    let control = slots.checked_add(1).and_then(|words| words.checked_mul(16))
        .ok_or_else(|| anyhow::anyhow!("MiMo route control overflow"))?;
    slots.checked_mul(slot.max(256)).and_then(|bytes| bytes.checked_add(control.max(256)))
        .and_then(|bytes| bytes.checked_add(256)).ok_or_else(|| anyhow::anyhow!("MiMo route exchange bytes overflow"))
}

/// Keep the allocation names in the ready ledger; category mapping is shared
/// rather than independently guessed by planner and runtime.
pub fn reservation_category(name: &str) -> Category {
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

pub fn reservation_group(name: &str) -> &str {
    match name {
        "state.active_rings" => "state",
        "prefix.exact_mark_arena" => "marks",
        "prefix.dflash_context_marks" => "DFlash context marks",
        "prefix.dflash_valid_floor_transfer" => "DFlash valid-floor transfer",
        _ => name,
    }
}

pub fn request(inputs: &MimoInputs<'_>) -> Result<PlacementRequest, CacheGeometryError> {
    let ranks = inputs.gpus.len();
    let profile = inputs.profile;
    let layers = inputs.layers;
    if !(1..=2).contains(&ranks) || profile.devices.len() != ranks || inputs.experts.len() != layers {
        return Err(PlacementError::Inventory("MiMo ranks, profile and layer costs disagree").into());
    }
    let cache = mimo_cache_geometry(inputs.cfg, layers, ranks, inputs.kv, inputs.mtp_stages)?;
    if profile.pool_unit_rows != cache.logical_unit_rows {
        return Err(PlacementError::Inventory("MiMo pool quantum").into());
    }
    let mut fixed = Vec::new();
    let mut drafter = Vec::new();
    let mut pool_overhead = Vec::new();
    for (rank, costs) in profile.devices.iter().enumerate() {
        for reservation in &costs.reservations {
            let demand = Demand::new(rank as u8, reservation_category(&reservation.name),
                reservation_group(&reservation.name), reservation.bytes, Basis::Formula);
            // The external drafter's own allocations form the shared drafter Movable.
            if reservation.name.starts_with("draft.") { drafter.push(demand) } else { fixed.push(demand) }
        }
        pool_overhead.push(costs.pool_unit_bytes.checked_sub(cache.ranks[rank].persistent_unit_bytes)
            .ok_or(PlacementError::Inventory("MiMo pool tables omit layer records"))?);
    }
    let mut tp2_workspace = inputs.tp2_workspace;
    if ranks == 2 && tp2_workspace.iter().any(|&bytes| bytes > 0) {
        let bytes = route_exchange_bytes(inputs.rows.max(64), inputs.cfg.topk as u64, inputs.lanes)
            .map_err(|_| PlacementError::Overflow("MiMo route exchange"))?;
        for workspace in &mut tp2_workspace {
            *workspace = workspace.checked_add(bytes).ok_or(PlacementError::Overflow("MiMo TP2 workspace"))?;
        }
    }
    let routed = inputs.experts.iter().filter(|cost| cost.is_some()).count();
    let local_package = inputs.expert_workspace > 0 || inputs.tp2_workspace.iter().any(|&bytes| bytes > 0);
    let mut pool = PoolPolicy::resolve(&inputs.gpus.iter().map(|g| g.0).collect::<Vec<_>>(), inputs.context,
        inputs.requested_pool, profile.pool_unit_rows, inputs.spark_ranks == 0)
        .with_automatic_context(inputs.automatic_context);
    let onboard = if routed > 0 && !local_package {
        if inputs.spark_ranks == 0 || !(inputs.onboard == Onboard::Auto || inputs.onboard.layers(routed) == Some(0)) {
            return Err(PlacementError::Inventory("MiMo local experts need a matching BF16-input package").into());
        }
        // Zero eligible local layers must not turn auto's bounded pool into a
        // fixed-onboard pool that consumes every remaining byte.
        if inputs.onboard == Onboard::Auto {
            pool.ceiling = pool.target;
            if pool.requested.is_none() { pool.floor = 0; }
        }
        Onboard::Layers(0)
    } else { inputs.onboard };
    let whole = inputs.cfg.head_split(1).map_err(|_| PlacementError::Inventory("MiMo whole heads"))?;
    let split = inputs.cfg.head_split(2).map_err(|_| PlacementError::Inventory("MiMo split heads"))?;
    let demands = (0..layers).map(|layer| {
        let full = inputs.cfg.attention[layer] == MimoAttention::Full;
        let bytes = |share: &MimoV2Config| {
            if full { profile.pool_unit_rows * share.record_bytes(MimoAttention::Full, inputs.kv) as u64 } else { 0 }
        };
        LayerDemand {
            kind: if full { AttentionClass::Gqa } else { AttentionClass::Swa },
            // Exact representations (including replicated routers and the lead head) are
            // already in the profile; later ownership executors must split them.
            weights: ModeBytes::default(),
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            kv_unit: KvDemand::from(ModeBytes { whole: bytes(&whole), split: [bytes(&split); 2] }),
            experts: inputs.experts[layer],
            modes: if ranks == 2 { vec![LayerMode::HeadSplit] }
                else { vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] },
        }
    }).collect();
    // DFlash reads GPU0's taps and borrows its head; the shared Movable keeps it
    // there until a measured bridge binding allows the other GPU.
    if drafter.iter().any(|d| d.gpu != 0) {
        return Err(PlacementError::Inventory("MiMo drafter allocations outside the head GPU").into());
    }
    let movables = if drafter.is_empty() { Vec::new() } else {
        vec![draft::DrafterDemand::itemized(drafter, 0).movable(ranks)?]
    };
    let gpus = inputs.gpus.iter().enumerate().map(|(rank, &(capacity_bytes, baseline))| GpuBudget {
        capacity_bytes, baseline,
        headroom_bytes: headroom_bytes(capacity_bytes, false)
            .max(if rank == 0 { inputs.startup_probe_bytes.saturating_add(capacity_bytes - capacity_bytes * 97 / 100) } else { 0 })
            .max(inputs.headroom_floor),
    }).collect();
    Ok(PlacementRequest {
        pool_bounded: Vec::new(),
        inventory: Inventory { gpus, spark_ranks: inputs.spark_ranks, peer_access: ranks == 2 },
        pool,
        layers: demands, pool_overhead, fixed, startup: Vec::new(), movables,
        attention_placement: None, context_buffers: Default::default(), layers_first_gpu: 0,
        expert_workspace: inputs.expert_workspace, tp2_workspace,
        minimum_expert_layers: 0,
        onboard, expert_gpus: 1, policy: LayerPolicy::default(),
        hops: HopSpec { storage: crate::placement::HopStorage::ReceiveSlots, row_bytes: inputs.cfg.hidden as u64 * 2, rows: inputs.rows, lanes: inputs.lanes,
            entry_gpu: 0, head_gpu: 0, terminal: if ranks == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } },
        executor: super::MIMO_V2,
    })
}

/// Split the admitted arena into ready residency and transport categories.
/// The solver reserves route storage only when it opens a TP2 arena.
pub fn expert_items(inputs: &MimoInputs<'_>, placement: &Placement, rank: usize) -> Result<Vec<Item>, PlacementError> {
    let mut items = placement.items[rank].iter().filter(|item| item.format == "tp2" || item.format == "local")
        .cloned().collect::<Vec<_>>();
    if placement.tp2.is_some() {
        let route = route_exchange_bytes(inputs.rows.max(64), inputs.cfg.topk as u64, inputs.lanes)
            .map_err(|_| PlacementError::Overflow("MiMo route exchange"))?;
        let arena = items.iter_mut().find(|item| item.format == "tp2")
            .ok_or(PlacementError::Inventory("MiMo TP2 arena item"))?;
        arena.bytes = arena.bytes.checked_sub(route).ok_or(PlacementError::Inventory("MiMo TP2 route reservation"))?;
        items.push(Item::new(Category::Transport, "TP2 canonical routes", "", route, Basis::Formula));
    }
    Ok(items)
}

/// Preserve the capacity interface consumed by context/prefix startup, without
/// a second pool solve. Startup load and post-KV packing checks remain separate.
pub fn resolved_capacity(inputs: &MimoInputs<'_>, request: &PlacementRequest, placement: &Placement,
    concurrency: u32, rings: u32) -> Result<ResolvedCapacity, PlacementError> {
    if concurrency == 0 || rings < concurrency || inputs.context == 0 {
        return Err(PlacementError::Inventory("MiMo concurrency, rings or context"));
    }
    let units = placement.pool_tokens / inputs.profile.pool_unit_rows;
    let mut feasible_units = u64::MAX;
    let mut devices = Vec::new();
    for (rank, costs) in inputs.profile.devices.iter().enumerate() {
        let gpu = request.inventory.gpus[rank];
        let budget = gpu.available();
        let mut reservations = costs.reservations.clone();
        reservations.extend(expert_items(inputs, placement, rank)?.into_iter().map(|item| MemoryReservation {
            name: if item.group == "TP2 canonical routes" { "transport.tp2_routes".into() }
                else { format!("experts.{}", item.group) }, bytes: item.bytes }));
        let reserved = reservations.iter().try_fold(0u64, |n, r| n.checked_add(r.bytes))
            .ok_or(PlacementError::Overflow("MiMo reservations"))?;
        let pool_bytes = units.checked_mul(costs.pool_unit_bytes).ok_or(PlacementError::Overflow("MiMo pool bytes"))?;
        let used = reserved.checked_add(pool_bytes).ok_or(PlacementError::Overflow("MiMo engine bytes"))?;
        let unused = budget.checked_sub(used).ok_or(PlacementError::Mandatory { gpu: rank as u8, what: "MiMo resolved capacity".into() })?;
        if costs.pool_unit_bytes > 0 { feasible_units = feasible_units.min((budget - reserved) / costs.pool_unit_bytes); }
        let non_engine_bytes = match gpu.baseline {
            Baseline::Measured { free_bytes } => gpu.capacity_bytes.saturating_sub(free_bytes),
            Baseline::Planned { context_bytes, loaded_bytes } => context_bytes + loaded_bytes,
        };
        devices.push(ResolvedDeviceCapacity { device: costs.device, total_bytes: gpu.capacity_bytes,
            non_engine_bytes, engine_budget_bytes: budget, reservations, reserved_bytes: reserved,
            pool_bytes, unused_budget_bytes: unused });
    }
    let feasible = feasible_units.checked_mul(inputs.profile.pool_unit_rows).ok_or(PlacementError::Overflow("MiMo feasible pool"))?;
    let floor = inputs.requested_pool.filter(|&n| n > 0).unwrap_or(request.pool.target);
    Ok(ResolvedCapacity {
        concurrency, state_slots: rings,
        checkpoint_max_context_tokens: inputs.profile.context.checkpoint_max_tokens,
        compiled_index_max_context_tokens: inputs.profile.context.compiled_index_max_tokens,
        effective_max_context_tokens: inputs.context, requested_kv_floor_tokens: floor,
        feasible_gpu_kv_tokens: feasible, allocated_gpu_kv_tokens: placement.pool_tokens,
        requested_floor_fits_hardware: feasible >= floor, requested_floor_allocated: placement.pool_tokens >= floor,
        requested_floor_shortfall_tokens: floor.saturating_sub(placement.pool_tokens),
        active_max_context_sequences: (units / inputs.context.div_ceil(inputs.profile.pool_unit_rows))
            .min(concurrency as u64) as u32,
        host_prefix_bytes: inputs.profile.host_prefix_bytes, explicit_pool_override: request.pool.requested.is_some(), devices,
    })
}
