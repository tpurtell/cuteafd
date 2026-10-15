//! V4.1 startup ownership guards; all pool/expert decisions use placement::solve.
pub(crate) use crate::shared::memory::budget::LocalLayers;
use anyhow::{Context, Result, ensure};
use cuteafd_loader::placement::*;

pub(super) const RUNTIME_HEADROOM: usize = 2 << 30;
pub(super) const EXPERT_SETUP_HEADROOM: usize = 64 << 20;
const SMALL_CARD_SAMPLE_MARGIN: usize = 32 << 20;

pub(super) fn graph_reserve_bytes(total: usize, dual_gpu: Option<usize>) -> usize {
    assert!(dual_gpu.is_none_or(|gpu| gpu < 2));
    families::deepseek_v41::graph_reserve(total as u64, if dual_gpu.is_some() { 2 } else { 1 })
        as usize
}

pub(super) fn embedding_placement(
    explicit: Option<crate::shared::memory::EmbedPlacement>,
    total: usize,
) -> crate::shared::memory::EmbedPlacement {
    explicit.unwrap_or(if total <= 32usize << 30 {
        crate::shared::memory::EmbedPlacement::Host
    } else {
        crate::shared::memory::EmbedPlacement::Gpu
    })
}

pub(super) fn measured_pool_memory(lib: &cuteafd_ffi::NativeLibrary) -> Result<(usize, usize)> {
    let (free, total) = lib.cuda_memory_info()?;
    if total > 32usize << 30 {
        return Ok((free, total));
    }
    let device = lib.cuda_get_device()?;
    let live = cuteafd_ffi::memory_ledger::snapshot()
        .total(cuteafd_ffi::memory_ledger::Space::Device, device);
    let occupied = total - free;
    // Empirical small-card reserve, not a per-executable model: official V4.1 Flash,
    // 31.8 GiB/C16 grew 1,107,296,256 bytes with six shapes, and 838,860,800
    // with eight shapes plus bounded index selection, ready -> ten-batch plateau.
    // The 2 GiB envelope also covers untracked driver/allocator growth;
    // other shape sets still require their own warmed margin gate.
    let graph_budget = std::env::var("CUTEAFD_V41_GRAPH_BUDGET_MIB")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2048)
        .checked_mul(1 << 20)
        .context("V4.1 graph budget overflow")?;
    let admitted_free = free
        .checked_sub(SMALL_CARD_SAMPLE_MARGIN)
        .and_then(|bytes| bytes.checked_sub(graph_budget))
        .context("32 GB V4.1 startup leaves no room for the measured-pool and graph margins")?;
    tracing::info!(
        occupied_bytes = occupied,
        tracked_owner_bytes = live,
        measured_context_module_allocator_bytes = occupied.saturating_sub(live),
        sample_margin_bytes = SMALL_CARD_SAMPLE_MARGIN,
        graph_budget_bytes = graph_budget,
        admitted_free_bytes = admitted_free,
        "V4.1 measured CUDA occupancy charged before pool sizing"
    );
    Ok((admitted_free, total))
}

pub(super) fn startup_phase(
    lib: &cuteafd_ffi::NativeLibrary,
    phase: &str,
    device_total: usize,
) -> Result<()> {
    if device_total <= 32usize << 30 {
        let (free, total) = lib.cuda_memory_info()?;
        let device = lib.cuda_get_device()?;
        let ledger = cuteafd_ffi::memory_ledger::snapshot();
        let space = cuteafd_ffi::memory_ledger::Space::Device;
        tracing::info!(
            phase,
            occupied_bytes = total - free,
            tracked_live_bytes = ledger.total(space, device),
            tracked_peak_bytes = ledger.peak.get(&(space, device)).copied().unwrap_or(0),
            "V4.1 small-card startup phase high-water ledger"
        );
    }
    Ok(())
}

/// Startup admission runs before any weight payload or device owner.
/// Loading-only owners are charged at their sequential phase, not retained in the KV ledger.
pub(super) fn admit_startup(
    lib: &cuteafd_ffi::NativeLibrary,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    args: &crate::cli::NativeServeArgs,
) -> Result<()> {
    use super::*;
    let (_, total) = lib.cuda_memory_info()?;
    let capacity = cuteafd_core::coordinator_programs::v41_live_rows(args.prefill_batch_tokens)
        .context("invalid V4.1 prefill batch")?;
    let rows = capacity as usize;
    let head_rows = cuteafd_core::V41_VERIFY_ROWS;
    let mut owners: Vec<(&str, usize)> = vec![
        (
            "backbone weights/load bound",
            BackboneLaneWeights::device_bytes(lib, catalog)?,
        ),
        (
            "cache producer weights",
            CacheProducerWeights::device_bytes(lib, catalog)?,
        ),
        (
            "index weights",
            IndexLaneWeights::device_bytes(lib, catalog)?,
        ),
        (
            "embedding",
            if embedding_placement(args.embedding_placement, total)
                == crate::shared::memory::EmbedPlacement::Gpu
            {
                NativeRtxTensors::plan(catalog, &["embed.weight".into()])?
            } else {
                0
            },
        ),
        (
            "vocabulary resident",
            VocabularyHead::resident_bytes(catalog)?,
        ),
        ("head weights", TargetHeadWeights::device_bytes(catalog)?),
        (
            "engram weights",
            EngramLayerWeights::device_bytes(lib, catalog, 0)?
                + EngramLayerWeights::device_bytes(lib, catalog, 1)?,
        ),
        (
            "vision",
            if local_vision_owner(
                cuteafd_api::openai::vision_input_enabled(),
                args.vision_peers.is_some(),
            ) {
                crate::families::deepseek_v41::v41_vision::VisionRuntime::device_bytes(
                    catalog, 9216,
                )?
            } else {
                0
            },
        ),
    ];
    let weight_bytes = owners.iter().take(4).try_fold(0usize, |n, (_, b)| {
        n.checked_add(*b).context("startup weights overflow")
    })?;
    let engram_bytes = owners
        .iter()
        .find(|(name, _)| *name == "engram weights")
        .unwrap()
        .1;
    let vocabulary_peak = VocabularyHead::plan(catalog)?;
    let lane = [
        TargetEmbeddingWave::device_bytes(rows)?,
        BackboneLane::workspace_bytes(lib, capacity)?
            .into_iter()
            .sum(),
        IndexLane::workspace_bytes(lib, capacity)?.into_iter().sum(),
        BackboneExecution::workspace_bytes(lib, capacity)?,
        EngramDeviceRows::device_bytes(rows)?,
        EngramGate::device_bytes(lib, rows)? * 2,
        TargetHeadWave::device_bytes(head_rows)?,
        crate::families::deepseek_v41::v41_target_pass::TargetTapWave::device_bytes(rows)?,
    ]
    .into_iter()
    .try_fold(0usize, |n, b| {
        n.checked_add(b).context("target lane admission overflow")
    })?;
    let mut loading_peaks = vec![(
        "vocabulary packing",
        weight_bytes
            .checked_add(engram_bytes)
            .and_then(|n| n.checked_add(lane))
            .and_then(|n| n.checked_add(vocabulary_peak))
            .context("vocabulary phase overflow")?,
    )];
    owners.push((
        "target/prefill lanes",
        lane.checked_mul(2).context("lane admission overflow")?,
    ));
    if crate::families::deepseek_v41::v41_tensors::fp8_head()
        == crate::families::deepseek_v41::v41_tensors::Fp8Head::All
    {
        owners.push((
            "target vocabulary FP8 scratch",
            lib.fp8_w8a16_workspace(head_rows, 5120, 129280)?
                .max(256)
                .checked_mul(2)
                .context("target FP8 scratch overflow")?,
        ));
    }
    owners.push((
        "target/prefill transports",
        NativeTp4Wave::device_bytes_for(capacity, args.peers.len())?
            .checked_mul(2)
            .context("transport admission overflow")?,
    ));
    if args.dspark {
        let tiers = catalog.exl3().map(|m| m.decoder_tiers()).unwrap_or(&[]);
        let directory = crate::families::deepseek_v41::v41_experts::exl3::aot_layout_directory(
            &args.native_lib,
            tiers,
            "dspark",
        );
        let (weights, runtime, staging) =
            crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::serving_bytes(
                lib,
                catalog,
                capacity,
                args.concurrency,
                if args.dspark_draft_limit > 5 { 7 } else { 5 },
                Some(&directory),
            )?;
        // Expert staging drains before auxiliary tensors and serving waves are allocated.
        let before_draft = owners
            .iter()
            .filter(|(name, _)| *name != "vision")
            .try_fold(0usize, |n, (_, b)| {
                n.checked_add(*b).context("pre-draft owners overflow")
            })?;
        loading_peaks.push((
            "dSpark loading",
            before_draft
                .checked_add(weights)
                .and_then(|n| n.checked_add(staging))
                .context("dSpark phase overflow")?,
        ));
        owners.extend([
            ("dSpark weights", weights),
            ("dSpark lanes/windows", runtime),
        ]);
    }
    ensure!(
        args.prefix_cache_entries <= 128,
        "invalid retained-turn limit"
    );
    let snapshot_slots = if args.prefix_cache_entries == 0 {
        0
    } else {
        2 * args.prefix_cache_entries as usize + 2
    };
    let snapshots = snapshot_slots
        * (crate::families::deepseek_v41::v41_backbone_cache::BackboneMark::BYTES
            + if args.dspark { 3 * 128 * 528 } else { 0 });
    owners.push(("snapshot arenas", snapshots));
    let fixed = owners.iter().try_fold(0usize, |n, (_, b)| {
        n.checked_add(*b).context("startup fixed-owner overflow")
    })?;
    let peak = startup_peak(fixed, &loading_peaks)?;
    // Geometry queries initialize native modules; sample their real CUDA cost
    // now, before reading weights, rather than using a card/driver constant.
    let (free, total) = measured_pool_memory(lib)?;
    ensure!(
        peak <= free,
        "V4.1 startup refused before weights: peak {peak} bytes, free {free}, phases {loading_peaks:?}, owners {owners:?}; use --prefill-batch-tokens 256 or a larger coordinator"
    );
    let after = free.checked_sub(fixed).with_context(|| format!("V4.1 fixed startup owners need {fixed} bytes, free {free}; set --prefill-batch-tokens 256, --rtx-expert-layers 0 or use a larger coordinator; owners {owners:?}"))?;
    let expert_rows = cuteafd_core::coordinator_programs::v41_aot_rows(capacity)
        .context("invalid local expert capacity")?;
    let experts = if args.rtx_expert_layers == LocalLayers::Count(0) {
        Vec::new()
    } else {
        expert_demands(lib, catalog, expert_rows, 1)?
    };
    let workspace = local_workspace(lib, catalog, args, expert_rows)?;
    let pool = resolve(args, &[(after, total)], experts, workspace, &[0], 0)
        .with_context(|| format!("V4.1 startup refused before weights: fixed-owner bound {fixed}, owners {owners:?}; use --prefill-batch-tokens 256 or increase coordinator memory"))?;
    tracing::info!(fixed_owner_bound_bytes=fixed, startup_peak_bound_bytes=peak, loading_phases=?loading_peaks, owners=?owners, admitted_cache_bytes=?pool.cache_bytes,
        "V4.1 complete startup admission before weight loads");
    Ok(())
}

// Remote vision reserves its tower on the Spark, never in the RTX startup bound.
fn local_vision_owner(enabled: bool, remote: bool) -> bool {
    enabled && !remote
}

fn startup_peak(fixed: usize, loading_phases: &[(&str, usize)]) -> Result<usize> {
    Ok(loading_phases
        .iter()
        .map(|(_, bytes)| *bytes)
        .max()
        .unwrap_or(0)
        .max(fixed))
}

#[cfg(test)]
mod startup_tests {
    #[test]
    fn verifier_extent_matches_heads_sparse_admission_and_planner() -> anyhow::Result<()> {
        use crate::families::deepseek_v41::graph_policy;
        use crate::families::deepseek_v41::v41_target_head::{
            TargetHeadWave, distributed_target::DistributedTargetHead,
        };
        let rows = cuteafd_core::V41_VERIFY_ROWS;
        assert_eq!(rows, 64);
        assert_eq!(graph_policy::MAX_DECODE_ROWS as usize, rows);
        assert_eq!(cuteafd_core::MAX_DSPARK_PROPOSALS, 7);
        assert!(graph_policy::profile_fixed_shapes(32usize << 30, Some("1,6,64"))?.is_some());
        assert!(graph_policy::profile_fixed_shapes(32usize << 30, Some("1,6,65")).is_err());
        let sparse_growth = cuteafd_ffi::V41SparseAttention::split_scratch_bytes(rows, 10)?
            + cuteafd_ffi::V41SparseAttention::batch_descriptor_bytes(rows)?
            - cuteafd_ffi::V41SparseAttention::split_scratch_bytes(48, 10)?
            - cuteafd_ffi::V41SparseAttention::batch_descriptor_bytes(48)?;
        let single = 2
            * (TargetHeadWave::device_bytes(rows)? - TargetHeadWave::device_bytes(48)?
                + sparse_growth);
        assert_eq!(
            single as u64,
            cuteafd_loader::placement::families::deepseek_v41::verify_workspace_growth(1, 0)
        );
        let wide = DistributedTargetHead::device_bytes(rows, 64640)?;
        let narrow = DistributedTargetHead::device_bytes(48, 64640)?;
        for rank in 0..2 {
            assert_eq!(
                (2 * (wide[rank] - narrow[rank] + sparse_growth)) as u64,
                cuteafd_loader::placement::families::deepseek_v41::verify_workspace_growth(2, rank)
            );
        }
        Ok(())
    }

    #[test]
    fn startup_vision_bound_only_charges_enabled_rtx_fallback() {
        assert!(super::local_vision_owner(true, false));
        assert!(!super::local_vision_owner(true, true));
        assert!(!super::local_vision_owner(false, false));
        assert!(!super::local_vision_owner(false, true));
    }
    #[test]
    fn embedding_profile_preserves_explicit_overrides_and_pro_default() {
        use crate::shared::memory::EmbedPlacement::{Gpu, Host};
        for total in [31usize << 30, 32usize << 30, 96usize << 30] {
            assert_eq!(super::embedding_placement(Some(Gpu), total), Gpu);
            assert_eq!(super::embedding_placement(Some(Host), total), Host);
            assert_eq!(
                super::embedding_placement(None, total),
                if total <= 32usize << 30 { Host } else { Gpu }
            );
        }
    }
    #[test]
    fn sequential_loading_peaks_do_not_shrink_the_serving_pool() {
        let fixed = 26usize << 30;
        let phases = [
            ("head packing", 20usize << 30),
            ("draft staging", 27usize << 30),
        ];
        assert_eq!(super::startup_peak(fixed, &phases).unwrap(), 27usize << 30);
        assert_eq!(super::startup_peak(fixed, &[]).unwrap(), fixed);
        // Retained residency, not sum of loading phases, is subtracted before KV.
        assert_eq!((32usize << 30) - fixed, 6usize << 30);
    }
}

/// One shared decision before cache or conditional routed weights are allocated.
/// `memory` is the context/module inventory less subsequent live owners.
pub(super) struct Admitted {
    pub pages: [usize; 4],
    pub cache_bytes: Vec<usize>,
    pub global_bytes: usize,
    pub reservation_bytes: Vec<usize>,
    pub placement: Placement,
}

pub(super) fn local_workspace(
    lib: &cuteafd_ffi::NativeLibrary,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    args: &crate::cli::NativeServeArgs,
    rows: u32,
) -> Result<u64> {
    if args.rtx_expert_layers == LocalLayers::Count(0) {
        return Ok(0);
    }
    use crate::families::deepseek_v41::v41_experts::local::LocalExpertWave;
    let lane = if let Some(exl3) = catalog.exl3() {
        let directory = crate::families::deepseek_v41::v41_experts::exl3::aot_layout_directory(
            &args.native_lib,
            exl3.decoder_tiers(),
            "rtx-tp1",
        );
        LocalExpertWave::exl3_device_bytes(&directory, rows)?
    } else {
        LocalExpertWave::device_bytes_for(lib, rows, catalog.nvfp4().is_some())?
    };
    Ok(lane
        .checked_mul(2)
        .context("local serving lanes overflow")? as u64)
}

pub(super) fn resolve(
    args: &crate::cli::NativeServeArgs,
    memory: &[(usize, usize)],
    experts: Vec<ExpertCost>,
    workspace: u64,
    deferred: &[u64],
    minimum: usize,
) -> Result<Admitted> {
    use cuteafd_core::memory_layout::{Basis, Category};
    let ranks = memory.len();
    ensure!(
        matches!(ranks, 1 | 2) && deferred.len() == ranks && minimum <= experts.len(),
        "invalid V4.1 admission inventory"
    );
    ensure!(
        (1..=1_048_576).contains(&args.max_context_tokens),
        "invalid V4.1 context limit"
    );
    if let LocalLayers::Count(count) = args.rtx_expert_layers {
        ensure!(
            count <= experts.len(),
            "explicit V4.1 prefix exceeds available expert inventory"
        );
    }
    let automatic = args.pool_tokens == Some(0);
    let mut ceilings = Vec::new();
    let gpus = memory
        .iter()
        .enumerate()
        .map(|(gpu, &(free, total))| {
            ensure!(
                total > 0 && free <= total,
                "invalid GPU {gpu} memory sample"
            );
            let mut ceiling = args
                .memory_reservation
                .map(|r| r.bytes(total))
                .transpose()?
                .unwrap_or(total);
            let small = total <= 32usize << 30;
            if small {
                ceiling = ceiling.min(cuteafd_core::serving_capacity::admission_ceiling(
                    total as u64,
                    97,
                    cuteafd_core::serving_capacity::small_card_headroom_bytes(total as u64),
                )? as usize);
            } else if automatic {
                ceiling = ceiling.min((total as u128 * 97 / 100) as usize);
            }
            ceilings.push(ceiling);
            let runtime =
                families::deepseek_v41::runtime_reserve(total as u64, ranks, automatic) as usize;
            let headroom =
                total - ceiling + runtime + graph_reserve_bytes(total, (ranks == 2).then_some(gpu));
            Ok(GpuBudget {
                capacity_bytes: total as u64,
                headroom_bytes: headroom as u64,
                baseline: Baseline::Measured {
                    free_bytes: free as u64,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let cards = memory.iter().map(|m| m.1 as u64).collect::<Vec<_>>();
    let tails = args.concurrency as u64 + 2 * args.prefix_cache_entries as u64;
    let mut pool = PoolPolicy::resolve(
        &cards,
        args.max_context_tokens as u64,
        args.pool_tokens,
        512,
        args.peers.is_empty(),
    );
    if let Some(bytes) = args.kv_pool_size.filter(|_| args.pool_tokens.is_none()) {
        let groups = bytes.0 as u64 / (5 * 256 * 356);
        ensure!(
            groups > tails && groups <= 131072,
            "explicit KV byte pool is outside physical/admission limits"
        );
        pool.requested = Some((groups - tails) * 512);
        pool.floor = (groups - tails) * 512;
    }
    if pool.requested.is_none() {
        pool.floor = pool.floor.max(
            (args.max_context_tokens as u64).div_ceil(512) * 512 + args.concurrency as u64 * 512,
        );
        pool.target = pool.target.max(pool.floor);
    }
    pool.ceiling = pool.requested.unwrap_or(pool.target).div_ceil(512) * 512;
    let onboard = match args.rtx_expert_layers {
        LocalLayers::Auto => Onboard::Auto,
        LocalLayers::Count(n) => Onboard::Layers(n),
    };
    let mut fixed = Vec::new();
    for gpu in 0..ranks {
        if deferred[gpu] > 0 {
            fixed.push(Demand::new(
                gpu as u8,
                Category::Transport,
                "deferred serving lanes and setup",
                deferred[gpu],
                Basis::Exact,
            ));
        }
    }
    let inventory = Inventory {
        gpus,
        spark_ranks: args.peers.len(),
        peer_access: ranks == 2,
    };
    let mut req = families::deepseek_v41::request(
        inventory,
        pool,
        vec![0; ranks],
        experts,
        workspace,
        onboard,
    );
    req.fixed = fixed;
    req.minimum_expert_layers = minimum;
    let config = serde_json::from_slice(&std::fs::read(args.snapshot.join("config.json"))?)?;
    families::deepseek_v41::cache_demands(
        &mut req,
        &config,
        args.concurrency as u64,
        args.prefix_cache_entries as u64,
        args.tp2_attention,
    )?;
    let placement = solve(&req)?;
    families::deepseek_v41::validate_pool_floor(&req.pool, &placement)?;
    ensure!(
        placement.onboard_layers >= minimum,
        "mandatory V4.1 expert prefix does not fit"
    );
    let groups = usize::try_from(placement.pool_tokens.div_ceil(512) + tails)?;
    let exact = cuteafd_loader::serving_capacity::deepseek_v41_cache_bytes(
        &config,
        ranks,
        args.concurrency as u64,
        groups as u64,
        args.tp2_attention,
    )?;
    for (gpu, &bytes) in exact.iter().enumerate() {
        ensure!(
            placement.items[gpu]
                .iter()
                .filter(|i| i.category == Category::Kv)
                .map(|i| i.bytes)
                .sum::<u64>()
                == bytes,
            "shared V4.1 cache demand differs from allocator"
        );
    }
    Ok(Admitted {
        pages: [groups, groups, groups, 2 * groups],
        cache_bytes: exact
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<_, _>>()?,
        global_bytes: groups * (5 * 256 * 356),
        reservation_bytes: ceilings,
        placement,
    })
}

/// Native info, not an independently rounded padding formula, determines every
/// resident pack. The checkpoint header supplies staging and EXL3 arena layout.
pub(super) fn expert_demands(
    lib: &cuteafd_ffi::NativeLibrary,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    rows: u32,
    ranks: usize,
) -> Result<Vec<ExpertCost>> {
    let stored = if catalog.exl3().is_some() {
        0
    } else {
        let info = match (ranks, catalog.nvfp4().is_some()) {
            (1, false) => lib.v41_local_expert_info(rows)?,
            (1, true) => lib.v41_nvfp4_local_expert_info(rows)?,
            (2, false) => lib.v41_tp2_expert_info(1)?,
            (2, true) => lib.v41_nvfp4_tp2_expert_info(1)?,
            _ => anyhow::bail!("invalid V4.1 expert owners"),
        };
        info.kernel_intermediate as u64
    };
    (0..40)
        .map(|layer| {
            let mut cost = ExpertCost {
                whole: Bytes2::default(),
                half: [Bytes2::default(); 2],
                tp2: ranks == 2,
                spark_ok: true,
            };
            if ranks == 1 {
                cost.whole = families::deepseek_v41::expert_cost(catalog, layer, 1, 0, stored)?;
            } else {
                for rank in 0..2 {
                    cost.half[rank] =
                        families::deepseek_v41::expert_cost(catalog, layer, 2, rank, stored)?;
                }
            }
            Ok(cost)
        })
        .collect()
}

/// Charge all occupancy arriving after the context/module inventory, including
/// lazy module growth, not just the tracked weight allocations.
pub(super) fn admission_memory(
    lib: &cuteafd_ffi::NativeLibrary,
    inventory: &crate::shared::inventory::RuntimeInventory,
) -> Result<Vec<(usize, usize)>> {
    let current = lib.cuda_get_device()?;
    let sampled = (|| -> Result<Vec<_>> {
        inventory
            .gpus
            .iter()
            .map(|g| {
                lib.cuda_set_device(g.device)?;
                measured_pool_memory(lib)
            })
            .collect()
    })();
    lib.cuda_set_device(current)?;
    let sampled = sampled?;
    let loaded = inventory
        .gpus
        .iter()
        .zip(&sampled)
        .map(|(g, (free, _))| g.free_bytes.saturating_sub(*free as u64))
        .collect::<Vec<_>>();
    let baselines = inventory.baselines(&loaded);
    let free = baselines
        .iter()
        .map(|(_, baseline)| match baseline {
            Baseline::Measured { free_bytes } => *free_bytes,
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    inventory.record("deepseek_v41", &free);
    baselines
        .into_iter()
        .map(|(total, baseline)| match baseline {
            Baseline::Measured { free_bytes } => {
                Ok((usize::try_from(free_bytes)?, usize::try_from(total)?))
            }
            _ => unreachable!(),
        })
        .collect()
}

/// Retain both independent lane contexts while RuntimeInventory samples CUDA.
/// Weight and scratch allocations happen only after the resulting admission.
pub(super) fn preload_exl3<'a>(
    lib: &'a cuteafd_ffi::NativeLibrary, directory: &std::path::Path,
    capacity: u32, ranks: usize,
) -> Result<[Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>; 2]> {
    use crate::shared::experts::exl3::execution::{Exl3InputFormat, Exl3Modules};
    let capacities = families::deepseek_v41::expert_capacities(capacity as u64, ranks == 1)?;
    let mut lanes = [Vec::new(), Vec::new()];
    for lane in &mut lanes {
        for &capacity in &capacities {
            // SAFETY: operator-selected pinned native exports, owned on the current device.
            lane.push(unsafe { Exl3Modules::load(lib, &directory.join(format!("m{capacity}")), Exl3InputFormat::Fp8K32)? });
        }
    }
    Ok(lanes)
}

/// Bind target modules before the inventory sample and first immutable weight
/// upload. EXL3 contexts are retained separately by `preload_exl3`.
pub(super) fn preload(
    lib: &cuteafd_ffi::NativeLibrary,
    rows: u32,
    ranks: usize,
    nvfp4: bool,
    exl3: bool,
) -> Result<()> {
    use super::*;
    if ranks == 1 {
        BackboneLane::workspace_bytes(lib, rows)?;
    } else {
        BackboneLane::placed_workspace_bytes(lib, rows)?;
    }
    IndexLane::workspace_bytes(lib, rows)?;
    for &(k, n) in &[(5120, 1152), (1152, 5120)] {
        if ranks == 2 {
            lib.v41_fp8_matrix_plan(rows, k, n)?;
        }
    }
    if !exl3 {
        let capacity = if ranks == 1 {
            cuteafd_core::coordinator_programs::v41_aot_rows(rows).context("expert capacity")?
        } else {
            rows
        };
        for c in families::deepseek_v41::expert_capacities(capacity as u64, ranks == 1)? {
            match (ranks, nvfp4) {
                (1, false) => {
                    lib.v41_local_expert_kernel(c as u32)?;
                }
                (1, true) => {
                    lib.v41_nvfp4_local_expert_kernel(c as u32)?;
                }
                (2, false) => {
                    lib.v41_tp2_expert_kernel(c as u32)?;
                }
                (2, true) => {
                    lib.v41_nvfp4_tp2_expert_kernel(c as u32)?;
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use clap::Parser;

    fn args(directory: &std::path::Path) -> crate::cli::NativeServeArgs {
        let cli = crate::cli::Cli::try_parse_from([
            "cuteafd",
            "serve-native",
            "--snapshot",
            directory.to_str().unwrap(),
            "--native-lib",
            "/unused.so",
            "--peers",
            "127.0.0.1:19441",
        ])
        .unwrap();
        let crate::cli::Commands::ServeNative(args) = cli.command else {
            unreachable!()
        };
        args
    }

    #[test]
    fn planner_equals_runtime_deepseek_v41() -> Result<()> {
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::plan::{
            ExpertPlacement, MediaMode, PlanOptions, layout::LayoutOptions, plan,
        };
        use serde_json::json;
        for nvfp4 in [false, true] {
            let dir = tempfile::tempdir()?;
            if nvfp4 {
                cuteafd_loader::plan::testing::write_v41_nvfp4_snapshot(dir.path());
            } else {
                cuteafd_loader::plan::testing::write_v41_snapshot(dir.path());
            }
            let share = dir.path().join("share");
            std::fs::create_dir_all(share.join("V41_EXPERT_WORKSPACES"))?;
            let programs = share.join("PROGRAMS.json");
            std::fs::write(&programs, b"{}")?;
            for ranks in [1, 2] {
                let role = if ranks == 1 {
                    "rtx_backbone"
                } else {
                    "rtx_tp2"
                };
                let manifest = json!({"role":role, "geometry": {"hidden":5120,
                "intermediate":2304/ranks,"kernel_intermediate":if ranks==1 {2304}else{1152},
                "experts":384,"topk":6}, "variants":([1,16,80,256,1024,4096].map(|c|
                    json!({"capacity_rows":c,"core_scratch_nbytes":if c==16 {99999}else{c*7}})))});
                std::fs::write(
                    share.join("V41_EXPERT_WORKSPACES").join(format!(
                        "{}_{role}.json",
                        if nvfp4 { "nvfp4" } else { "mxfp4" }
                    )),
                    serde_json::to_vec(&manifest)?,
                )?;
            }
            let variants = [(5120, 1152), (1152, 5120)]
                .into_iter()
                .flat_map(|(k, n)| {
                    [1, 16, 80, 256, 1024, 4096].map(|c| {
                        json!({"capacity":c,"input_dim":k,
                "output_dim":n,"scratch_bytes":c+1})
                    })
                })
                .collect::<Vec<_>>();
            std::fs::write(
                share.join("V41_FP8_AOT.json"),
                serde_json::to_vec(&json!({"variants":variants}))?,
            )?;
            let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
            for ranks in [1, 2] {
                for (onboard, pool_tokens, concurrency, turns) in [
                    (Onboard::Auto, None, 16, 20),
                    (Onboard::Layers(3), Some(0), 16, 20),
                    (Onboard::Layers(3), Some(1 << 20), 1, 0),
                ] {
                    let total = 96u64 << 30;
                    let report = plan(
                        dir.path(),
                        &PlanOptions {
                            placement: ExpertPlacement::from_spark_ranks(4),
                            vision: MediaMode::Off,
                            layout: Some(LayoutOptions {
                                rtx_bytes: vec![total; ranks],
                                prefill_rows: 256,
                                context_tokens: 262_144,
                                concurrency,
                                prefix_slots: Some(if turns == 0 { 0 } else { 2 * turns + 2 }),
                                pool_tokens,
                                native_mtp_layers: 0,
                                workspace_manifest: Some(programs.clone()),
                                onboard: Some(onboard),
                                ..Default::default()
                            }),
                            ..Default::default()
                        },
                    )?;
                    let layout = report.memory_layout.context("missing V4.1 layout")?;
                    ensure!(
                        report.placement_supported,
                        "planner refused {ranks} {onboard:?}: {:?} {:?}",
                        layout.notes,
                        report.hints
                    );
                    let mut args = args(dir.path());
                    args.rtx_gpus = ranks as u32;
                    args.max_context_tokens = 262_144;
                    args.pool_tokens = pool_tokens;
                    args.concurrency = concurrency as u32;
                    args.prefix_cache_entries = turns as u32;
                    args.rtx_expert_layers = match onboard {
                        Onboard::Auto => LocalLayers::Auto,
                        Onboard::Layers(n) => LocalLayers::Count(n),
                        _ => unreachable!(),
                    };
                    // Reconstruct the admission sample after the planner's retained
                    // owners, excluding pool/expert allocations and lazy graphs.
                    let memory = layout
                        .devices
                        .iter()
                        .take(ranks)
                        .map(|device| {
                            let loaded = device
                                .items
                                .iter()
                                .filter(|i| {
                                    !matches!(i.category, Category::Kv | Category::Experts)
                                        && i.group != "graph allowance"
                                })
                                .map(|i| i.bytes)
                                .sum::<u64>();
                            ((total - loaded) as usize, total as usize)
                        })
                        .collect::<Vec<_>>();
                    // Linked native info supplies these extents at serving startup.
                    // The fixture's published manifest carries the same values;
                    // build runtime demands independently of expert_inventory.
                    let stored = if ranks == 1 { 2304 } else { 1152 };
                    let experts = (0..40)
                        .map(|layer| {
                            let mut cost = ExpertCost {
                                whole: Bytes2::default(),
                                half: [Bytes2::default(); 2],
                                tp2: ranks == 2,
                                spark_ok: true,
                            };
                            if ranks == 1 {
                                cost.whole = families::deepseek_v41::expert_cost(
                                    &catalog, layer, 1, 0, stored,
                                )?;
                            } else {
                                for rank in 0..2 {
                                    cost.half[rank] = families::deepseek_v41::expert_cost(
                                        &catalog, layer, 2, rank, stored,
                                    )?;
                                }
                            }
                            Ok(cost)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut workspace = families::deepseek_v41::NativeWorkspace {
                        scratch_bytes: 99999,
                        kernel_intermediate: stored,
                    }
                    .lane_bytes(256, 5120, 6, ranks == 2, nvfp4)?;
                    if ranks == 2 {
                        let fp8 = serde_json::from_slice(&std::fs::read(
                            share.join("V41_FP8_AOT.json"),
                        )?)?;
                        workspace += families::deepseek_v41::shared_workspace(&fp8, 256)?
                            + 256 * (10240 + 5280 + 24 + 24 + 10240);
                    }
                    let admitted = resolve(
                        &args,
                        &memory,
                        experts,
                        2 * workspace,
                        &vec![0; ranks],
                        usize::from(ranks == 2),
                    )?;
                    assert_eq!(
                        layout.pool_tokens, admitted.placement.pool_tokens,
                        "{ranks} {onboard:?} pool"
                    );
                    for (gpu, items) in admitted.placement.items.iter().enumerate() {
                        for item in items {
                            assert!(
                                layout.devices[gpu].items.contains(item),
                                "{ranks} {onboard:?} gpu{gpu}: {item:?}; planned {:?}",
                                layout.devices[gpu]
                                    .items
                                    .iter()
                                    .filter(|i| i.category == item.category)
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                    assert_eq!(admitted.placement.layers.len(), 40);
                    for (layer, assigned) in admitted.placement.layers.iter().enumerate() {
                        assert_eq!(
                            assigned.mode,
                            LayerMode::Whole {
                                gpu: u8::from(ranks == 2 && layer >= 20),
                                ffn: FfnMode::Owner
                            }
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn logical_pool_excludes_snapshot_tails_and_charges_each_cache_owner() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
            )),
        )?;
        for (ranks, replicas) in [(1, false), (2, false), (2, true)] {
            let mut args = args(dir.path());
            args.rtx_gpus = ranks as u32;
            args.tp2_attention = replicas;
            args.rtx_expert_layers = LocalLayers::Count(0);
            let memory = vec![(80usize << 30, 96usize << 30); ranks];
            let admitted = resolve(&args, &memory, vec![], 0, &vec![0; ranks], 0)?;
            assert_eq!(admitted.placement.pool_tokens, 2 << 20);
            assert_eq!(admitted.pages, [4152, 4152, 4152, 8304]);
            args.kv_pool_size = Some(crate::shared::memory::budget::ByteSize(
                (4152 * 5 * 256 * 356) + 1,
            ));
            let exact = resolve(&args, &memory, vec![], 0, &vec![0; ranks], 0)?;
            assert_eq!(exact.pages, admitted.pages);
            assert_eq!(exact.cache_bytes, admitted.cache_bytes);
            args.kv_pool_size = None;
            args.pool_tokens = Some(67_108_864);
            assert!(resolve(&args, &memory, vec![], 0, &vec![0; ranks], 0).is_err());
            args.pool_tokens = Some(2 << 20);
            assert!(
                resolve(
                    &args,
                    &vec![(1usize << 30, 96usize << 30); ranks],
                    vec![],
                    0,
                    &vec![0; ranks],
                    0
                )
                .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn strict_pool_respects_each_owner_deferred_bytes_and_reservation_rounding() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
            )),
        )?;
        let total = 96usize << 30;
        for (ranks, replicas) in [(1, false), (2, false), (2, true)] {
            let mut args = args(dir.path());
            args.rtx_gpus = ranks as u32;
            args.tp2_attention = replicas;
            args.rtx_expert_layers = LocalLayers::Count(0);
            args.pool_tokens = Some(1 << 20);
            args.memory_reservation = Some("80.000001GiB".parse()?);
            let deferred = vec![64u64 << 20; ranks];
            let roomy = vec![(80usize << 30, total); ranks];
            let admitted = resolve(&args, &roomy, vec![], 0, &deferred, 0)?;
            let ceiling = args.memory_reservation.unwrap().bytes(total)?;
            let exact = admitted
                .cache_bytes
                .iter()
                .enumerate()
                .map(|(gpu, &bytes)| {
                    let reserve = total - ceiling
                        + families::deepseek_v41::runtime_reserve(total as u64, ranks, false)
                            as usize
                        + graph_reserve_bytes(total, (ranks == 2).then_some(gpu));
                    (bytes + reserve + deferred[gpu] as usize, total)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                resolve(&args, &exact, vec![], 0, &deferred, 0)?.pages,
                admitted.pages
            );
            for gpu in 0..ranks {
                let mut short = exact.clone();
                short[gpu].0 -= 1;
                assert!(
                    resolve(&args, &short, vec![], 0, &deferred, 0).is_err(),
                    "rank {gpu} cannot borrow the other owner's free bytes"
                );
            }
            // Snapshot arenas are already live in the measured sample; they
            // reduce fit, never the logical-to-physical tail conversion.
            let occupied =
                50 * crate::families::deepseek_v41::v41_backbone_cache::BackboneMark::BYTES;
            let with_snapshots = roomy
                .iter()
                .map(|&(free, total)| (free - occupied, total))
                .collect::<Vec<_>>();
            assert_eq!(
                resolve(&args, &with_snapshots, vec![], 0, &deferred, 0)?.pages,
                admitted.pages
            );
            args.kv_pool_size = Some(crate::shared::memory::budget::ByteSize(1 << 20));
            args.pool_tokens = None;
            assert!(resolve(&args, &roomy, vec![], 0, &deferred, 0).is_err());
        }
        Ok(())
    }

    #[test]
    fn invalid_admission_shapes_and_explicit_prefixes_refuse() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
            )),
        )?;
        let mut args = args(dir.path());
        args.rtx_expert_layers = LocalLayers::Count(3);
        let expert = ExpertCost {
            whole: Bytes2 {
                resident: 1 << 30,
                staging: 20 << 20,
            },
            half: [Bytes2::default(); 2],
            tp2: false,
            spark_ok: true,
        };
        let memory = [(80usize << 30, 96usize << 30)];
        let admitted = resolve(&args, &memory, vec![expert; 40], 600 << 20, &[0], 0)?;
        assert_eq!(admitted.placement.onboard_layers, 3);
        assert!(resolve(&args, &memory, vec![expert; 2], 600 << 20, &[0], 0).is_err());
        args.rtx_expert_layers = LocalLayers::Count(0);
        for context in [0, 1_048_577, u32::MAX] {
            args.max_context_tokens = context;
            assert!(resolve(&args, &memory, vec![], 0, &[0], 0).is_err());
        }
        args.max_context_tokens = 262_144;
        for concurrency in [0, 17, u32::MAX] {
            args.concurrency = concurrency;
            assert!(resolve(&args, &memory, vec![], 0, &[0], 0).is_err());
        }
        args.concurrency = 16;
        args.prefix_cache_entries = 129;
        assert!(resolve(&args, &memory, vec![], 0, &[0], 0).is_err());
        args.prefix_cache_entries = 20;
        assert!(resolve(&args, &[(97usize << 30, 96usize << 30)], vec![], 0, &[0], 0).is_err());
        assert!(resolve(&args, &[(0, 0)], vec![], 0, &[0], 0).is_err());
        Ok(())
    }

    #[test]
    fn graph_reserves_and_small_card_logical_target_remain_distinct() -> Result<()> {
        assert_eq!(graph_reserve_bytes(96usize << 30, None), 3264 << 20);
        for rank in [0, 1] {
            assert_eq!(graph_reserve_bytes(96usize << 30, Some(rank)), 1472 << 20);
        }
        assert_eq!(graph_reserve_bytes(32usize << 30, None), 0);
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
            )),
        )?;
        let mut args = args(dir.path());
        args.rtx_expert_layers = LocalLayers::Count(0);
        args.max_context_tokens = 262_144;
        let admitted = resolve(&args, &[(30usize << 30, 32usize << 30)], vec![], 0, &[0], 0)?;
        assert_eq!(admitted.placement.pool_tokens, 1 << 20);
        assert_eq!(admitted.pages[0], 2048 + 56);
        args.memory_reservation = Some(crate::shared::memory::budget::Reservation::Percent(
            1_000_000,
        ));
        assert!(resolve(&args, &[(30usize << 30, 32usize << 30)], vec![], 0, &[0], 0).is_err());
        Ok(())
    }
}
