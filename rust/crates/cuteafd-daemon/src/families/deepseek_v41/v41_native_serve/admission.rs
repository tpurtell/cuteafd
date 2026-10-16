//! V4.1 startup ownership guards; all pool/expert decisions use placement::solve.
pub(crate) use crate::shared::memory::budget::LocalLayers;
use anyhow::{Context, Result, ensure};
use cuteafd_loader::placement::*;

pub(super) const RUNTIME_HEADROOM: usize = 2 << 30;
pub(super) const EXPERT_SETUP_HEADROOM: usize = 64 << 20;

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
    // The inventory is the raw CUDA observation. Graph/runtime growth and
    // allocator margins are explicit solver demands, not fabricated occupancy.
    Ok(lib.cuda_memory_info()?)
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
    use cuteafd_core::memory_layout::{Basis, Category};
    use families::deepseek_v41::{owners, resident};
    let config = serde_json::from_slice(&std::fs::read(catalog.snapshot().join("config.json"))?)?;
    let retained = resident::resident_headers(&config,
        catalog.tensors().iter().map(|tensor| &tensor.metadata), 1, false)
        .context("unsupported V4.1 native backbone residency")?;
    let weights = retained.into_iter().next().unwrap().into_iter()
        .filter(|item| item.category != Category::Embedding
            || embedding_placement(args.embedding_placement, total)
                == crate::shared::memory::EmbedPlacement::Gpu)
        .map(|item| Demand::new(0, item.category, item.group, item.bytes, item.basis))
        .collect::<Vec<_>>();
    let manifest = owners::native_manifest(&args.native_lib, "V41_FP8_AOT.json")?;
    let programs = owners::NativePrograms::new(&manifest, u32::try_from(lib.sm_count()?)?)?;
    let mut target = programs.target(u64::from(capacity), 1,
        crate::families::deepseek_v41::v41_tensors::fp8_head()
            == crate::families::deepseek_v41::v41_tensors::Fp8Head::All,
        BackboneExecution::window_batch_enabled())?;
    // Test builds retain the materialized query reference buffer; production
    // intentionally has no such owner.
    #[cfg(test)]
    for lane in 0..2 {
        target.push(Demand::new(0, Category::Workspace,
            format!("lane {lane} test query reference"), u64::from(capacity) * 65536, Basis::Formula));
    }
    let mut startup = vec![owners::vocabulary_load_phase(&weights, &target,
        VocabularyHead::plan(catalog)? as u64)?];
    let mut fixed_demands = weights;
    fixed_demands.append(&mut target);
    fixed_demands.push(Demand::new(0, Category::Transport, "target/prefill transports",
        2 * owners::transport_bytes(u64::from(capacity), args.peers.len())?, Basis::Formula));
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    let draft = if args.dspark {
        let tiers = catalog.exl3().map(|m| m.decoder_tiers()).unwrap_or(&[]);
        let directory = crate::families::deepseek_v41::v41_experts::exl3::aot_layout_directory(
            &args.native_lib, tiers, "dspark");
        let (weights, runtime, staging) =
            crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::serving_bytes(
                lib, catalog, capacity, args.concurrency,
                if args.dspark_draft_limit > 5 { 7 } else { 5 }, Some(&directory))?;
        let before_draft = fixed_demands.iter().try_fold(0u64, |n, d|
            n.checked_add(d.bytes).context("pre-draft owners overflow"))?;
        startup.push(Demand::new(0, Category::Staging, "dSpark loading",
            before_draft.checked_add(weights as u64).and_then(|n| n.checked_add(staging as u64))
                .context("dSpark phase overflow")?, Basis::Formula));
        let prefixes = resident::prefix_bytes(u64::from(args.prefix_cache_entries), true, 1)[0]
            - resident::prefix_bytes(u64::from(args.prefix_cache_entries), false, 1)[0];
        Some(families::deepseek_v41::DraftInventory {
            resident: weights as u64,
            runtime: vec![Demand::new(0, Category::Drafter, "dSpark lanes/windows",
                runtime as u64, Basis::Exact)],
            marks: prefixes, peer_half: 0, staging: [staging as u64, 0],
        })
    } else { None };
    if local_vision_owner(cuteafd_api::openai::vision_input_enabled(), args.vision_peers.is_some()) {
        fixed_demands.push(Demand::new(0, Category::Workspace, "vision weights/scratch",
            crate::families::deepseek_v41::v41_vision::VisionRuntime::device_bytes(catalog, 9216)? as u64,
            Basis::Formula));
    }
    fixed_demands.push(Demand::new(0, Category::Prefix, "target snapshot arenas",
        resident::prefix_bytes(u64::from(args.prefix_cache_entries), false, 1)[0], Basis::Formula));
    let fixed = fixed_demands.iter().try_fold(0u64, |n, d|
        n.checked_add(d.bytes).context("startup fixed-owner overflow"))?;
    let expert_rows = cuteafd_core::coordinator_programs::v41_aot_rows(capacity)
        .context("invalid local expert capacity")?;
    let experts = if args.rtx_expert_layers == LocalLayers::Count(0) {
        Vec::new()
    } else {
        expert_demands(lib, catalog, expert_rows, 1)?
    };
    let workspace = local_workspace(lib, catalog, args, expert_rows)?;
    // Native geometry queries may initialize modules. Observe that CUDA cost
    // only after every query, still before payloads or retained device owners.
    let (free, total) = measured_pool_memory(lib)?;
    let pool = resolve_with_owners(args, &[(free, total)], experts, workspace, &[0], 0,
        &fixed_demands, &startup, draft.as_ref())
        .with_context(|| format!("V4.1 startup refused before weights: fixed owners {fixed}; reduce the prefill batch or increase coordinator memory"))?;
    tracing::info!(fixed_owner_bytes=fixed, loading_phases=?startup, owners=?fixed_demands,
        draft=?draft, admitted_cache_bytes=?pool.cache_bytes,
        "V4.1 allocation-owner startup admission before weight loads");
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

/// With `draft`, the sample must precede all draft weights, runtime buffers and
/// the draft-mark arena. `None` is for samples that already contain those owners.
pub(super) fn resolve(
    args: &crate::cli::NativeServeArgs,
    memory: &[(usize, usize)],
    experts: Vec<ExpertCost>,
    workspace: u64,
    deferred: &[u64],
    minimum: usize,
    draft: Option<&families::deepseek_v41::DraftInventory>,
) -> Result<Admitted> {
    resolve_with_owners(args, memory, experts, workspace, deferred, minimum, &[], &[], draft)
}

/// `startup` contains full sequential peaks above `memory`, never staging-only
/// deltas. Draft owners supplied separately must not also appear in `owners`.
pub(super) fn resolve_with_owners(
    args: &crate::cli::NativeServeArgs,
    memory: &[(usize, usize)],
    experts: Vec<ExpertCost>,
    workspace: u64,
    deferred: &[u64],
    minimum: usize,
    owners: &[Demand],
    startup: &[Demand],
    draft: Option<&families::deepseek_v41::DraftInventory>,
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
            let headroom = total - ceiling;
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
    let onboard = args.rtx_expert_layers.onboard();
    let mut fixed = owners.to_vec();
    for gpu in 0..ranks {
        fixed.extend(families::deepseek_v41::growth_demands(memory[gpu].1 as u64, ranks,
            automatic, gpu as u8));
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
    req.startup = startup.to_vec();
    if let Some(draft) = draft {
        req.movables.push(families::deepseek_v41::drafter(draft, ranks)?);
    }
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
    use crate::shared::experts::exl3::execution::{Exl3InputFormat, Exl3Modules, Exl3ReductionPlan};
    let capacities = families::deepseek_v41::exl3_capacities(capacity as u64, ranks == 1)?;
    let directories: Vec<_> = capacities.iter().map(|capacity| directory.join(format!("m{capacity}"))).collect();
    Exl3ReductionPlan::new(&directories)?;
    let mut lanes = [Vec::new(), Vec::new()];
    for lane in &mut lanes {
        for directory in &directories {
            // SAFETY: operator-selected pinned native exports, owned on the current device.
            lane.push(unsafe { Exl3Modules::load(lib, directory, Exl3InputFormat::Fp8K32)? });
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
    if !nvfp4 && !exl3 {
        let capacity = if ranks == 1 {
            cuteafd_core::coordinator_programs::v41_aot_rows(rows).context("expert capacity")?
        } else {
            rows
        };
        let infos = families::deepseek_v41::expert_capacities(capacity as u64, ranks == 1)?
            .into_iter()
            .map(|capacity| {
                match ranks {
                    1 => lib.v41_local_expert_info(capacity as u32),
                    2 => lib.v41_tp2_expert_info(capacity as u32),
                    _ => anyhow::bail!("invalid V4.1 expert owners"),
                }
                .with_context(|| format!("V4.1 MXFP4 {ranks}-GPU m{capacity} export"))
            })
            .collect::<Result<Vec<_>>>()?;
        crate::shared::experts::ExpertReductionPlan::native(lib, infos)?;
    }
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
                "output_dim":n,"label":format!("shared_{k}_{n}_m{c}"),"scratch_bytes":c+1})
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
                        workspace += families::deepseek_v41::shared_workspace(&fp8, 256, 188)?
                            + 256 * (10240 + 5280 + 24 + 24 + 10240);
                    }
                    let admitted = resolve(
                        &args,
                        &memory,
                        experts,
                        2 * workspace,
                        &vec![0; ranks],
                        usize::from(ranks == 2), None,
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
    fn max_admits_experts_first_through_the_shared_solver() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        )))?;
        for ranks in [1, 2] {
            let mut args = args(dir.path());
            args.rtx_gpus = ranks as u32;
            args.max_context_tokens = 262144;
            args.rtx_expert_layers = "max".parse()?;
            let experts = vec![ExpertCost {
                whole: Bytes2 { resident: 1 << 30, staging: 64 << 20 },
                half: [Bytes2 { resident: 1 << 30, staging: 64 << 20 }; 2],
                tp2: ranks == 2,
                spark_ok: true,
            }; 40];
            let memory = vec![(40usize << 30, 96usize << 30); ranks];
            let max = resolve(&args, &memory, experts.clone(), 64 << 20,
                &vec![0; ranks], usize::from(ranks == 2), None)?;
            assert!(max.placement.onboard_layers >= 1);
            assert!(max.placement.pool_tokens >= EXPERTS_FIRST_POOL_FLOOR);
            args.rtx_expert_layers = LocalLayers::Auto;
            let auto = resolve(&args, &memory, experts, 64 << 20,
                &vec![0; ranks], usize::from(ranks == 2), None)?;
            assert!(max.placement.onboard_layers > auto.placement.onboard_layers);
            assert!(max.placement.pool_tokens < auto.placement.pool_tokens);
            assert_eq!(max.placement.attention_placement,
                if ranks == 2 { AttentionPlacement::Layers } else { AttentionPlacement::Heads });
        }
        Ok(())
    }

    #[test]
    fn pre_draft_admission_charges_weights_runtime_and_marks_once() -> Result<()> {
        use cuteafd_core::memory_layout::{Basis, Category};
        use families::deepseek_v41::DraftInventory;
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        )))?;
        for ranks in [1, 2] {
            let mut args = args(dir.path());
            args.rtx_gpus = ranks as u32;
            args.max_context_tokens = 512;
            args.pool_tokens = Some(512);
            args.rtx_expert_layers = LocalLayers::Count(0);
            let home = (ranks - 1) as u8;
            let draft = DraftInventory {
                resident: 8 << 30,
                runtime: (0..ranks).map(|gpu| Demand::new(gpu as u8,
                    Category::Workspace, "draft runtime", 64 << 20, Basis::Exact)).collect(),
                marks: 32 << 20,
                peer_half: if ranks == 2 { 512 << 20 } else { 0 },
                // Staging is not a retained owner or a complete startup phase.
                staging: [100 << 30; 2],
                loading_experts: [0; 2],
                loader_peak: [0; 2],
            };
            let memory = vec![(80usize << 30, 96usize << 30); ranks];
            let moved = resolve(&args, &memory, vec![], 0, &vec![0; ranks], 0, Some(&draft))?;
            assert_eq!(draft::home(&moved.placement), Some(home));
            assert_eq!(moved.placement.items[home as usize].iter()
                .filter(|item| item.group == "dSpark weights").map(|item| item.bytes).sum::<u64>(), draft.resident);
            assert_eq!(moved.placement.items[home as usize].iter()
                .filter(|item| item.group == "dSpark marks").map(|item| item.bytes).sum::<u64>(), draft.marks);
            for gpu in 0..ranks {
                assert_eq!(moved.placement.items[gpu].iter().filter(|item| item.group == "draft runtime")
                    .map(|item| item.bytes).sum::<u64>(), 64 << 20);
            }
            let movable = families::deepseek_v41::drafter(&draft, ranks)?;
            let fixed = &movable.conditional.iter().find(|c| c.placement_gpu == home).unwrap().demands;
            let original = resolve_with_owners(&args, &memory, vec![], 0, &vec![0; ranks], 0, fixed, &[], None)?;
            assert_eq!(moved.cache_bytes, original.cache_bytes);
            for gpu in 0..ranks {
                let total = |items: &[cuteafd_core::memory_layout::Item]| items.iter().map(|i| i.bytes).sum::<u64>();
                assert_eq!(total(&moved.placement.items[gpu]), total(&original.placement.items[gpu]));
            }
            // A pre-draft sample does not yet contain these bytes. Omitting
            // the movable would incorrectly admit this constrained inventory.
            let constrained = vec![(8usize << 30, 96usize << 30); ranks];
            assert!(resolve(&args, &constrained, vec![], 0, &vec![0; ranks], 0, None).is_ok());
            assert!(resolve(&args, &constrained, vec![], 0, &vec![0; ranks], 0, Some(&draft)).is_err());
            // Only the caller's full coexistence peak gates the load phase.
            let peak = [Demand::new(home, Category::Staging, "full draft load phase",
                81 << 30, Basis::Exact)];
            assert!(resolve_with_owners(&args, &memory, vec![], 0, &vec![0; ranks], 0,
                &[], &peak, Some(&draft)).is_err());
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
            let admitted = resolve(&args, &memory, vec![], 0, &vec![0; ranks], 0, None)?;
            assert_eq!(admitted.placement.pool_tokens, 2 << 20);
            assert_eq!(admitted.pages, [4152, 4152, 4152, 8304]);
            args.kv_pool_size = Some(crate::shared::memory::budget::ByteSize(
                (4152 * 5 * 256 * 356) + 1,
            ));
            let exact = resolve(&args, &memory, vec![], 0, &vec![0; ranks], 0, None)?;
            assert_eq!(exact.pages, admitted.pages);
            assert_eq!(exact.cache_bytes, admitted.cache_bytes);
            args.kv_pool_size = None;
            args.pool_tokens = Some(67_108_864);
            assert!(resolve(&args, &memory, vec![], 0, &vec![0; ranks], 0, None).is_err());
            args.pool_tokens = Some(2 << 20);
            assert!(
                resolve(
                    &args,
                    &vec![(1usize << 30, 96usize << 30); ranks],
                    vec![],
                    0,
                    &vec![0; ranks],
                    0, None
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
            let admitted = resolve(&args, &roomy, vec![], 0, &deferred, 0, None)?;
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
                resolve(&args, &exact, vec![], 0, &deferred, 0, None)?.pages,
                admitted.pages
            );
            for gpu in 0..ranks {
                let mut short = exact.clone();
                short[gpu].0 -= 1;
                assert!(
                    resolve(&args, &short, vec![], 0, &deferred, 0, None).is_err(),
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
                resolve(&args, &with_snapshots, vec![], 0, &deferred, 0, None)?.pages,
                admitted.pages
            );
            args.kv_pool_size = Some(crate::shared::memory::budget::ByteSize(1 << 20));
            args.pool_tokens = None;
            assert!(resolve(&args, &roomy, vec![], 0, &deferred, 0, None).is_err());
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
        let admitted = resolve(&args, &memory, vec![expert; 40], 600 << 20, &[0], 0, None)?;
        assert_eq!(admitted.placement.onboard_layers, 3);
        assert!(resolve(&args, &memory, vec![expert; 2], 600 << 20, &[0], 0, None).is_err());
        args.rtx_expert_layers = LocalLayers::Count(0);
        for context in [0, 1_048_577, u32::MAX] {
            args.max_context_tokens = context;
            assert!(resolve(&args, &memory, vec![], 0, &[0], 0, None).is_err());
        }
        args.max_context_tokens = 262_144;
        for concurrency in [0, 17, u32::MAX] {
            args.concurrency = concurrency;
            assert!(resolve(&args, &memory, vec![], 0, &[0], 0, None).is_err());
        }
        args.concurrency = 16;
        args.prefix_cache_entries = 129;
        assert!(resolve(&args, &memory, vec![], 0, &[0], 0, None).is_err());
        args.prefix_cache_entries = 20;
        assert!(resolve(&args, &[(97usize << 30, 96usize << 30)], vec![], 0, &[0], 0, None).is_err());
        assert!(resolve(&args, &[(0, 0)], vec![], 0, &[0], 0, None).is_err());
        Ok(())
    }

    #[test]
    fn graph_reserves_and_small_card_logical_target_remain_distinct() -> Result<()> {
        assert_eq!(graph_reserve_bytes(96usize << 30, None), 3264 << 20);
        for rank in [0, 1] {
            assert_eq!(graph_reserve_bytes(96usize << 30, Some(rank)), 1472 << 20);
        }
        assert_eq!(graph_reserve_bytes(32usize << 30, None), 2048 << 20);
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
        let admitted = resolve(&args, &[(30usize << 30, 32usize << 30)], vec![], 0, &[0], 0, None)?;
        assert_eq!(admitted.placement.pool_tokens, 1 << 20);
        assert_eq!(admitted.pages[0], 2048 + 56);
        args.memory_reservation = Some(crate::shared::memory::budget::Reservation::Percent(
            1_000_000,
        ));
        assert!(resolve(&args, &[(30usize << 30, 32usize << 30)], vec![], 0, &[0], 0, None).is_err());
        Ok(())
    }
}
