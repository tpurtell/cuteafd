//! MiMo physical facts. Both planner and runtime declare the same owners and
//! retained representations; only the GPU baseline differs.
pub mod admission;

pub use admission::route_exchange_bytes;

use crate::families::mimo_v2::MimoV2Config;
use crate::placement::{PlacementError, ResidentWeights};
use cuteafd_core::memory_layout::Category;

/// Item category for one profile reservation or pool owner.
pub fn reservation_category(name: &str) -> Category {
    if name.starts_with("mimo.") && name.ends_with(".tail") { Category::Runtime }
    else if name.contains("embed_tokens") { Category::Embedding }
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

/// Item group: physical families that keep their own ledger row.
pub fn reservation_group(name: &str) -> &str {
    match name {
        "state.active_rings" => "state",
        "prefix.exact_mark_arena" => "marks",
        "prefix.dflash_context_marks" => "DFlash context marks",
        "prefix.dflash_valid_floor_transfer" => "DFlash valid-floor transfer",
        _ => name,
    }
}

/// Already-loaded tower owners follow physical CUDA ordinals, not rank numbers.
/// Their physical storage is in the baseline; only checkpoint weights are added.
pub fn add_local_media_weights(weights: &mut [ResidentWeights], devices: &[i32],
    media: Option<(i32, ResidentWeights)>, limit: ResidentWeights) -> Result<(), PlacementError> {
    if weights.len() != devices.len() {
        return Err(PlacementError::Inventory("MiMo media owner/rank geometry"));
    }
    if let Some((device, media)) = media {
        if let Some(rank) = devices.iter().position(|&ordinal| ordinal == device) {
            weights[rank] = weights[rank].checked_add(media)?;
        } else {
            let gpu = u8::try_from(device)
                .map_err(|_| PlacementError::Inventory("MiMo media CUDA ordinal"))?;
            media.check(gpu, limit)?;
        }
    }
    Ok(())
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
    let package = lib.join("fp8").join(format.package_name(family));
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
    use crate::families::mimo_v2::{MimoAttention, MimoKvFormats};
    use crate::placement::planner::Pin;
    use crate::placement::{Bytes2, ExpertCost, Onboard};
    use crate::serving_capacity::mimo_cache_geometry;
    use cuteafd_core::serving_capacity::{
        CapacityProfile, ContextLimits, DeviceCosts, MemoryReservation,
    };

    #[test]
    fn media_resources_follow_physical_ordinals_and_check_independent_owners() {
        for devices in [[7, 2], [2, 7]] {
            for rank in 0..2 {
                let mut weights = [ResidentWeights(11), ResidentWeights(22)];
                add_local_media_weights(&mut weights, &devices,
                    Some((devices[rank], ResidentWeights(123))), ResidentWeights(256)).unwrap();
                assert_eq!(weights[rank].0, [11, 22][rank] + 123);
                assert_eq!(weights[1 - rank].0, [11, 22][1 - rank]);
            }
            let mut weights = [ResidentWeights(11), ResidentWeights(22)];
            add_local_media_weights(&mut weights, &devices, None, ResidentWeights(256)).unwrap();
            add_local_media_weights(&mut weights, &devices, Some((9, ResidentWeights(256))), ResidentWeights(256)).unwrap();
            assert_eq!(weights, [ResidentWeights(11), ResidentWeights(22)]);
            assert_eq!(add_local_media_weights(&mut weights, &devices,
                Some((9, ResidentWeights(257))), ResidentWeights(256)),
                Err(PlacementError::ResidentWeightLimit { gpu: 9, required: 257, limit: 256 }));
        }
        assert!(add_local_media_weights(&mut [ResidentWeights(u64::MAX)], &[7],
            Some((7, ResidentWeights(1))), ResidentWeights(u64::MAX)).is_err());
        assert!(add_local_media_weights(&mut [], &[7], None, ResidentWeights(1)).is_err());
    }

    #[test]
    fn route_exchange_matches_native_slot_alignment_and_rejects_overflow() {
        for rows in [1, 3, 64, 4096] {
            for lanes in [1, 2, 3] {
                let payload = (rows * 8 * 4u64).div_ceil(16) * 32;
                assert_eq!(route_exchange_bytes(rows, 8, lanes).unwrap(),
                    lanes * 2 * payload.max(256) + 2 * 64 * 8 * 8 + ((lanes * 2 + 3) * 16).max(256) + 256);
            }
        }
        for (rows, topk, lanes) in [(u64::MAX, 8, 3), (64, u64::MAX, 3), (64, 8, u64::MAX)] {
            assert!(route_exchange_bytes(rows, topk, lanes).is_err());
        }
        assert_eq!(cuteafd_core::memory_layout::Tag::transport("test/peer-split").kind.category(), Category::Transport);
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

    /// Four backbone layers, one routed slot: enough to exercise the contract
    /// without a checkpoint, at MiMo's real pool quantum.
    fn fixture(ranks: usize, context: u64) -> (MimoV2Config, CapacityProfile, CapacityProfile) {
        let mut cfg = MimoV2Config::from_hf(&crate::plan::testing::mimo_pro_config()).unwrap();
        cfg.layers = 4;
        cfg.dense = vec![true, false, false, false];
        cfg.attention = vec![MimoAttention::Full; 4];
        let cache = mimo_cache_geometry(&cfg, 4, ranks, MimoKvFormats::default(), 0).unwrap();
        let profile = CapacityProfile {
            context: ContextLimits { checkpoint_max_tokens: 1 << 20, compiled_index_max_tokens: None },
            pool_unit_rows: cache.logical_unit_rows,
            host_prefix_bytes: 0,
            devices: cache.ranks.iter().enumerate().map(|(rank, storage)| DeviceCosts {
                device: rank as u32,
                pool_unit_bytes: storage.persistent_unit_bytes + storage.pool_metadata_unit_bytes + 256,
                reservations: vec![MemoryReservation { name: format!("state.rank{rank}"),
                    bytes: 64 << 20 }], reserve_bytes: 0,
            }).collect(),
        };
        let _ = context;
        (cfg, profile.clone(), profile)
    }

    fn phase_profiles(profile: &CapacityProfile) -> crate::families::mimo_v2::capacity::MimoCapacityProfiles {
        crate::families::mimo_v2::capacity::MimoCapacityProfiles { weight_arenas: vec![vec![]; profile.devices.len()], resident_weights: vec![ResidentWeights(0); profile.devices.len()],
            loading: profile.devices.iter().map(|costs| DeviceCosts { device: costs.device, reservations: vec![], pool_unit_bytes: 0, reserve_bytes: 0 }).collect(),
            post_target_kv: profile.devices.clone(), steady: profile.clone() }
    }

    fn input<'a>(cfg: &'a MimoV2Config, spark: &'a CapacityProfile, no_spark: &'a CapacityProfile,
        ranks: usize, context: u64) -> admission::ProductionInput<'a> {
        let cost = ExpertCost { whole: Bytes2 { resident: 1 << 30, staging: 0 },
            half: [Bytes2 { resident: 512 << 20, staging: 0 }; 2], tp2: ranks == 2,
            whole_supported: true, spark_ok: true };
        admission::ProductionInput {
            expert_usage: false,
            blas_head: crate::placement::planner::serving::BlasHead::Native, draft_head: false,
            cfg, layers: 4, kv: MimoKvFormats::default(), mtp_stages: 0, context,
            maximum_pool_tokens: admission::maximum_pool_tokens(&[96 << 30; 2][..ranks], context,
                None, spark.pool_unit_rows),
            prefill: crate::placement::planner::PrefillShape { rows: 4096, waves: 3 },
            lanes: 3, embedding: crate::placement::planner::EmbeddingHome::Device(0),
            gpus: ranks, spark_ranks: 6,
            experts: vec![None, Some(cost), Some(cost), Some(cost)],
            expert_workspace: if ranks == 1 { 1 << 20 } else { 0 },
            tp2_workspace: if ranks == 2 { [1 << 20; 2] } else { [0; 2] },
            spark: phase_profiles(spark), no_spark: phase_profiles(no_spark),
            post_kv_additional: vec![vec![]; ranks], draft_packing: vec![vec![]; ranks],
            host: admission::HostOwners { expert_input: Default::default(), output: crate::families::mimo_v2::MimoPrefillOutput::LastRow,
                serving: true, scoring_rows: 64, diagnostic_rows: 256, prefix_bytes: 0, media_bytes: 0, headroom: 0 },
            spark_reply_formats: Default::default(),
            resident_weights: vec![ResidentWeights(11 << 30); ranks],
            operation: crate::placement::planner::Operation::Serving,
            budgets: admission::production_budgets(&vec![(96 << 30, 96 << 30); ranks],
                &[0; 2][..ranks], vec![crate::placement::planner::Baseline::Forecast {
                    occupied: 0, source: "unit test".into(), assumed: false }; ranks],
                (128 << 30, 128 << 30), 108 << 30, 6).unwrap(),
        }
    }

    #[test]
    fn arena_charges_replace_payloads_once_across_phases_and_onboarding() {
        use crate::families::mimo_v2::capacity::MimoWeightArena;
        use crate::placement::planner::{self, ComponentKind, Lifetime, Location, Phase, Recipe};
        use crate::serving_capacity::WeightArenaLayout;
        let (cfg, spark, no_spark) = fixture(2, 4096);
        let mut facts = input(&cfg, &spark, &no_spark, 2, 4096);
        let arenas = (0..2).map(|rank| vec![
            MimoWeightArena { kind: cuteafd_core::memory_layout::MemoryKind::Weights, layout: WeightArenaLayout::new(format!("mimo.target.rank{rank}"),
                [("payload".into(), (3 << 20) + 1), ("norm".into(), 17)]).unwrap(), lifetime: Lifetime::RESIDENT },
            MimoWeightArena { kind: cuteafd_core::memory_layout::MemoryKind::DrafterWeights, layout: WeightArenaLayout::new(format!("mimo.dflash.rank{rank}"),
                [("draft.payload".into(), (5 << 20) + 1)]).unwrap(),
                lifetime: Lifetime { first: Phase::DraftLoad, last: Phase::Growth } },
        ]).collect::<Vec<_>>();
        for profiles in [&mut facts.spark, &mut facts.no_spark] {
            profiles.weight_arenas = arenas.clone();
            for (rank, arenas) in arenas.iter().enumerate() {
                for arena in arenas {
                    let rows = arena.reservations().unwrap();
                    profiles.steady.devices[rank].reservations.extend(rows.clone());
                    if arena.lifetime == Lifetime::RESIDENT {
                        profiles.loading[rank].reservations.extend(rows.clone());
                        profiles.post_target_kv[rank].reservations.extend(rows);
                    }
                }
            }
        }
        let prepared = facts.prepare().unwrap();
        let request = admission::production_request(4096, Some(4096), Onboard::Layers(1),
            prepared.recipe.prefill, planner::Operation::Serving, ResidentWeights(u64::MAX), Default::default());
        let contract = prepared.recipe.prepare(&request).unwrap();
        let mut roots = Vec::new();
        for component in &contract.components {
            if let ComponentKind::Allocation { template, .. } = &component.kind {
                assert!(!template.id.contains(".mimo."), "no phase-specific duplicate arena rows");
                if template.group == "weight arena" {
                    assert_eq!(component.activation, planner::Activation::always());
                    assert!(template.view.is_none());
                    roots.push(template);
                }
            }
        }
        assert_eq!(roots.len(), 4);
        for (rank, arenas) in arenas.iter().enumerate() {
            for arena in arenas {
                let root = roots.iter().find(|root| root.id == arena.layout.id()).unwrap();
                assert_eq!(root.location, Location::coordinator(rank as u8));
                assert_eq!(root.bytes, arena.layout.bytes() as u64);
                assert_eq!(root.lifetime, arena.lifetime);
                assert_eq!(root.category, arena.kind.category());
                let tails = contract.components.iter().filter_map(|component| match &component.kind {
                    ComponentKind::AllocationGranularity(tail) if tail.id == format!("{}.tail", root.id) => Some(tail),
                    _ => None,
                }).collect::<Vec<_>>();
                assert_eq!(tails.len(), 1);
                assert_eq!(tails[0].backings[0].backing, root.id);
                assert_eq!(tails[0].requests(), [root.bytes]);
                assert_eq!(tails[0].lifetime, root.lifetime);
                assert_eq!(tails[0].bytes().unwrap(), arena.reservations().unwrap()[1].bytes);
            }
        }
        let admitted = prepared.admit(request).unwrap();
        for rank in 0..2 {
            let claimed = admitted.solved.facts.resident_weights.iter()
                .filter(|claim| claim.location == Location::coordinator(rank)).map(|claim| claim.weights.0).sum::<u64>();
            let expert = admitted.solved.facts.resident_weights.iter()
                .filter(|claim| claim.location == Location::coordinator(rank) && claim.id.contains("experts.arena"))
                .map(|claim| claim.weights.0).sum::<u64>();
            assert_eq!(claimed - expert, 11 << 30, "arena claims replace the old aggregate");
        }
    }

    #[test]
    fn expert_usage_prices_one_canonical_producer_even_with_tp2() {
        use crate::placement::planner::{ComponentKind, Location, Operation, Recipe};
        for ranks in [1, 2] {
            let (cfg, spark, no_spark) = fixture(ranks, 4096);
            let off = input(&cfg, &spark, &no_spark, ranks, 4096).prepare().unwrap();
            assert!(off.recipe.expert_usage.is_none());
            let mut facts = input(&cfg, &spark, &no_spark, ranks, 4096);
            facts.expert_usage = true;
            let on = facts.prepare().unwrap();
            let geometry = on.recipe.expert_usage.as_ref().unwrap();
            assert_eq!(geometry.routed_layers, [1, 2, 3]);
            assert_eq!((geometry.layers, geometry.experts, geometry.topk), (4, cfg.experts, cfg.topk));
            let request = admission::production_request(4096, Some(4096), Onboard::Layers(1),
                on.recipe.prefill, Operation::Serving, ResidentWeights(u64::MAX), Default::default());
            let contract = on.recipe.prepare(&request).unwrap();
            let mut owners = Vec::new();
            let mut unchanged = Vec::new();
            for component in contract.components {
                if let ComponentKind::Allocation { template, .. } = &component.kind {
                    if template.id.starts_with("expert-usage.") {
                        assert_eq!(component.bytes.at(1).unwrap(), component.bytes.at(on.recipe.maximum_pages).unwrap());
                        owners.push(template.clone());
                        continue;
                    }
                }
                unchanged.push(component);
            }
            assert_eq!(format!("{unchanged:?}"), format!("{:?}", off.recipe.prepare(&request).unwrap().components));
            assert_eq!(owners.len(), 4);
            let table = geometry.table_bytes().unwrap() as u64;
            for (suffix, location, bytes) in [
                ("gpu0.counters", Location::coordinator(0), table),
                ("gpu0.snapshot", Location::coordinator(0), table),
                ("gpu0.pinned", Location::PINNED, table),
                ("history", Location::HOST, geometry.host_bytes(1).unwrap() as u64),
            ] {
                let owner = owners.iter().find(|owner| owner.id == format!("expert-usage.{suffix}")).unwrap();
                assert_eq!((owner.location, owner.bytes), (location, bytes));
            }
        }
    }

    #[test]
    fn expert_usage_geometry_follows_backbone_not_expert_costs() {
        let (mut cfg, spark, no_spark) = fixture(1, 4096);
        let mut facts = input(&cfg, &spark, &no_spark, 1, 4096);
        facts.expert_usage = true;
        facts.experts.fill(None);
        let prepared = facts.prepare().unwrap();
        assert!(prepared.recipe.routed.is_empty());
        assert_eq!(prepared.recipe.expert_usage.unwrap().routed_layers, [1, 2, 3]);
        cfg.dense.fill(true);
        let mut facts = input(&cfg, &spark, &no_spark, 1, 4096);
        facts.expert_usage = true;
        facts.experts.fill(None);
        assert!(facts.prepare().unwrap().recipe.expert_usage.is_none());
    }

    #[test]
    fn blas_capacity_receipt_matches_ready_owners_and_pre_body_preparation() {
        use crate::placement::planner::{Location, Phase, serving::BlasHead};
        for ranks in [1, 2] {
            for head in [BlasHead::Native, BlasHead::TensorLt, BlasHead::Gemm] {
                for draft in [false, true] {
                    let (cfg, spark, no_spark) = fixture(ranks, 4096);
                    let mut facts = input(&cfg, &spark, &no_spark, ranks, 4096);
                    facts.blas_head = head;
                    facts.draft_head = draft;
                    let prepared = facts.prepare().unwrap();
                    let request = admission::production_request(4096, Some(4096), Onboard::Layers(0),
                        prepared.recipe.prefill, crate::placement::planner::Operation::Serving,
                        ResidentWeights(u64::MAX), Default::default());
                    let admitted = prepared.admit(request).unwrap();
                    let capacity = admission::resolved_capacity(&admitted, 1, 1).unwrap();
                    let solved = &admitted.solved;
                    for rank in 0..ranks {
                        let allowances = |phase| solved.facts.allowances.iter().filter(move |owner|
                            owner.location == Location::coordinator(rank as u8)
                                && owner.lifetime.contains(phase, solved.facts.phase_order));
                        // The CUDA compute stream exists from the first load
                        // copy, so its driver stream slab is resident in every
                        // phase. It is the only allowance charged before Ready;
                        // the old opaque post-Ready growth charge is gone.
                        let pre_ready = allowances(Phase::DraftLoad).collect::<Vec<_>>();
                        assert_eq!(pre_ready.len(), 1);
                        assert!(pre_ready[0].source.starts_with("mimo/driver-stream-slabs"), "{:?}", pre_ready[0].source);
                        assert_eq!(pre_ready[0].bytes, crate::placement::planner::cuda_stream_slab_bytes(
                            admission::startup_peak_live_streams()).unwrap());
                        let gemm = allowances(Phase::Ready).filter(|owner| owner.source.starts_with("cuBLAS handles:"))
                            .map(|owner| owner.bytes).sum::<u64>();
                        assert_eq!(gemm, u64::from(rank == 0 && (head.gemm() || (head.bf16() && draft))) * (64 << 20));
                        for owner in allowances(Phase::Ready) {
                            let receipts = capacity.devices[rank].reservations.iter()
                                .filter(|item| item.name == owner.source).collect::<Vec<_>>();
                            assert_eq!(receipts.len(), 1);
                            assert_eq!(receipts[0].bytes, owner.bytes);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_and_unset_pool_admit_identically() {
        let (cfg, spark, no_spark) = fixture(1, 256 << 10);
        let context = 256 << 10;
        let prepared = input(&cfg, &spark, &no_spark, 1, context).prepare().unwrap();
        let request = |pool| admission::production_request(context, pool, Onboard::Auto,
            prepared.recipe.prefill, crate::placement::planner::Operation::Serving,
            ResidentWeights(u64::MAX), Default::default());
        let unset = prepared.admit(request(None)).unwrap();
        let zero = prepared.admit(request(Some(0))).unwrap();
        assert!(matches!(unset.solved.request.pool, Pin::Auto));
        assert!(matches!(zero.solved.request.pool, Pin::Auto));
        assert_eq!(unset.placement.pool_tokens, zero.placement.pool_tokens);
        assert_eq!(unset.spark.rtx_layers, zero.spark.rtx_layers);
        assert_eq!(unset.placement.items, zero.placement.items);
    }

    #[test]
    fn host_tables_match_admitted_curves_and_share_embedding_budget() {
        let (cfg, spark, no_spark) = fixture(2, 4096);
        let mut facts = input(&cfg, &spark, &no_spark, 2, 4096);
        facts.embedding = crate::placement::planner::EmbeddingHome::HostPinned;
        let prepared = facts.prepare().unwrap();
        let recipe = &prepared.recipe;
        let components = recipe.host.components(recipe, 1).unwrap();
        for pages in [1, 2, 63, 64, 65, 66, 1024, recipe.maximum_pages] {
            for (component, (name, bytes)) in components.iter().zip(recipe.host.demands(recipe, pages, 1).unwrap()) {
                assert_eq!(component.bytes.at(pages).unwrap(), bytes, "{name} pages={pages}");
            }
        }
        let request = admission::production_request(4096, Some(4096), Onboard::Layers(1), recipe.prefill,
            crate::placement::planner::Operation::Serving, ResidentWeights(u64::MAX), Default::default());
        let admitted = prepared.admit(request).unwrap();
        let host = admitted.solved.items(crate::placement::planner::Location::PINNED).unwrap();
        assert!(host.iter().any(|item| item.category == Category::Embedding && item.bytes > 0));
        assert!(admitted.solved.phases.iter().any(|row| row.budget == "rtx0"
            && row.phase == crate::placement::planner::Phase::Load));
    }

    #[test]
    fn explicit_reply_formats_are_preserved_by_the_same_contract() {
        use cuteafd_core::spark_reply::{SparkReplyFormat::*, SparkReplyFormats};
        let (cfg, spark, no_spark) = fixture(1, 4096);
        for prefill in [Bf16, Fp8, Nvfp4] {
            for decode in [Bf16, Fp8, Nvfp4] {
                let formats = SparkReplyFormats { prefill: Some(prefill), decode: Some(decode) };
                let mut facts = input(&cfg, &spark, &no_spark, 1, 4096);
                facts.spark_reply_formats = formats;
                let prepared = facts.prepare().unwrap();
                let request = admission::production_request(4096, Some(4096), Onboard::Layers(0), prepared.recipe.prefill,
                    crate::placement::planner::Operation::Serving, ResidentWeights(u64::MAX), formats);
                let admitted = prepared.admit(request).unwrap();
                assert_eq!(admitted.solved.decisions.spark_reply_formats, Some(formats));
            }
        }
    }

    #[test]
    fn unaffordable_card_returns_a_typed_refusal() {
        let (cfg, spark, no_spark) = fixture(1, 256 << 10);
        let context = 256 << 10;
        let mut prepared = input(&cfg, &spark, &no_spark, 1, context);
        prepared.budgets = admission::production_budgets(&[(1 << 30, 1 << 30)], &[0],
            vec![crate::placement::planner::Baseline::Forecast { occupied: 0,
                source: "unit test".into(), assumed: false }], (128 << 30, 128 << 30),
            108 << 30, 6).unwrap();
        let prepared = prepared.prepare().unwrap();
        let request = admission::production_request(context, None, Onboard::Auto,
            prepared.recipe.prefill, crate::placement::planner::Operation::Serving,
            ResidentWeights(u64::MAX), Default::default());
        let refused = prepared.admit(request).err().expect("a 1 GiB card cannot hold the context floor");
        assert!(matches!(refused, crate::placement::planner::Refusal::Capacity { .. }
            | crate::placement::planner::Refusal::FullContextShortfall { .. }
            | crate::placement::planner::Refusal::Unsupported), "unexpected refusal {refused}");
    }
}
