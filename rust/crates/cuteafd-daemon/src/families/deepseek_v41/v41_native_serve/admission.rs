//! V4.1 startup ownership guards; all pool/expert decisions use placement::solve.
pub(crate) use crate::shared::memory::budget::LocalLayers;
use anyhow::{Context, Result, ensure};
use cuteafd_loader::placement::*;

fn spark_model(catalog: &cuteafd_loader::OfficialV41Catalog, capacity: u32, waves: u8, bf16: bool)
    -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    let decode = super::DECODE_LANE_ROWS;
    use cuteafd_core::{SparkEndpoint, SparkIngress};
    use cuteafd_loader::plan::spark_executions::{admit_family, SparkBackend};
    let ingress = if bf16 { SparkIngress::Bf16 } else { SparkIngress::Fp8K32 };
    // The shared ladder: the unit's covering export, `waves` units buffered.
    let executions = admit_family("deepseek_v41", &SparkBackend::for_catalog(catalog)?.exports(), None,
        u64::from(capacity), u64::from(waves))?;
    // The two decode lanes' links at the decode width and the one prefill
    // link at the unit (its `waves` units are the prefill owner's buffered units).
    let model = cuteafd_transport::worker_selection::ModelAdmission::new(
        vec![SparkEndpoint { ring_depth: None, capacity: decode.min(capacity), ingress },
             SparkEndpoint { ring_depth: None, capacity: decode.min(capacity), ingress },
             SparkEndpoint { ring_depth: None, capacity, ingress }],
        crate::shared::placement::spark_rings(64 << 20)?, executions)?;
    model.validate(cuteafd_core::ExpertGeometry::DEEPSEEK_V41)?;
    Ok(model)
}

pub(super) fn start_sparks(args: &crate::cli::NativeServeArgs,
    catalog: &cuteafd_loader::OfficialV41Catalog, local_layers: usize,
    capacity: u32,
    topology: Option<cuteafd_transport::expert::SparkTopology>)
    -> Result<Option<crate::shared::placement::PendingSparkAdmission>> {
    plan_sparks(args, catalog, local_layers, capacity, topology)?
        .map(|admission| admission.start(&args.peers)).transpose()
}

/// The Spark admission (headers only), before any worker is contacted.
pub(super) fn plan_sparks(args: &crate::cli::NativeServeArgs,
    catalog: &cuteafd_loader::OfficialV41Catalog, local_layers: usize,
    capacity: u32,
    topology: Option<cuteafd_transport::expert::SparkTopology>)
    -> Result<Option<crate::shared::placement::SparkAdmission>> {
    let geometry = catalog.routed_experts().geometry()?;
    ensure!(local_layers <= geometry.layers as usize, "V4.1 local boundary exceeds checkpoint");
    if args.peers.is_empty() {
        ensure!(local_layers == geometry.layers as usize, "V4.1 remote layers require Spark peers");
        return Ok(None);
    }
    let slots = super::AdmittedPrefill::resolve(args)?.waves;
    spark_admission(&args.snapshot, catalog, local_layers, args.peers.len(), capacity, slots, topology).map(Some)
}

/// The Spark admission for `ranks` workers serving layers `local_layers..`,
/// the decode lanes' and prefill link's endpoints at `capacity` (the unit),
/// `slots` prefill units buffered: what serve (and the GPU fixtures) start
/// before any transport connects.
pub(crate) fn spark_admission(snapshot: &std::path::Path, catalog: &cuteafd_loader::OfficialV41Catalog,
    local_layers: usize, ranks: usize, capacity: u32, slots: usize,
    topology: Option<cuteafd_transport::expert::SparkTopology>) -> Result<crate::shared::placement::SparkAdmission> {
    let geometry = catalog.routed_experts().geometry()?;
    let working = SparkWorkingSet { spark_layers: (local_layers..geometry.layers as usize).collect(),
        rtx_layers: (0..local_layers).collect(), spark_ranks: ranks, prefill_lanes: slots as u64 };
    crate::shared::placement::SparkAdmission::plan(snapshot, geometry, Some(catalog), &working, ranks, topology,
        spark_model(catalog, capacity, u8::try_from(slots)?, catalog.nvfp4().is_some())?)
}

#[cfg(test)]
mod spark_tests {
    use super::*;

    #[test]
    fn local_load_overlaps_spark_load_before_transport_connects() {
        for (source, load, connect) in [
            (include_str!("../v41_native_serve.rs"), "ExpertWeights::load(", "transport.bootstrap("),
            (include_str!("distributed.rs"), "tp2_ffn::load_routed_weights(", "super::spark_transport("),
        ] {
            let start = source.find("admission::start_sparks(").or_else(|| source.find("plan.start_with(")).unwrap();
            let load = source[start..].find(load).unwrap() + start;
            let wait = source.find("pending.wait()?").unwrap();
            let connect = source[wait..].find(connect).unwrap() + wait;
            assert!(start < load && load < wait && wait < connect);
        }
    }

    #[test]
    fn planner_and_serve_admit_identical_spark_owners() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_v41_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let backend = cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?;
        for live in [256u32, 2048, 4096] {
            for waves in [1u8, 2] {
                // The planner's resolved unit and waves, as serve admits them.
                let options = cuteafd_loader::plan::layout::LayoutOptions { prefill_unit_rows: Some(u64::from(live)),
                    prefill_waves: u64::from(waves), ..Default::default() };
                assert_eq!(spark_model(&catalog, live, waves, false)?.executions,
                    cuteafd_loader::plan::layout::planned_spark_executions("deepseek_v41", backend, None, &options)?);
            }
        }
        Ok(())
    }

    /// The GPU fixture's admission (endpoints at the 64-row verify width, two
    /// 64-row prefill units) passes the same validation the workers run.
    #[test]
    fn the_fixture_admission_shape_validates() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_v41_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        let model = spark_model(&catalog, cuteafd_core::V41_VERIFY_ROWS as u32, 2, false)?;
        model.validate(cuteafd_core::ExpertGeometry::DEEPSEEK_V41)?;
        assert!(spark_model(&catalog, 16, 2, false).is_err(),
            "16-row endpoints cannot admit the 64-row decode owners");
        Ok(())
    }

    #[test]
    fn endpoint_contract_covers_the_decode_and_prefill_links() -> Result<()> {
        let root = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_v41_snapshot(root.path());
        let catalog = cuteafd_loader::read_expert_catalog(root.path())?;
        for (live, compiled) in [(256, 256), (2048, 4096), (4096, 4096)] {
            for bf16 in [false, true] {
                let model = spark_model(&catalog, live, 2, bf16)?;
                assert_eq!((model.capacity(), model.compiled_capacity()), (live, compiled));
                // Two decode lanes, one prefill owner holding the two units in flight.
                assert_eq!(model.executions.decode, [cuteafd_core::DecodeExecution {
                    capacity: cuteafd_core::V41_VERIFY_ROWS as u32, compiled_capacity: 80 }; 2]);
                assert_eq!(model.executions.prefill, cuteafd_core::PrefillExecution { capacity: live,
                    compiled_capacity: compiled, unit_rows: live, buffered_units: 2 });
                // The decode lanes' links at the decode width (P3), the prefill link at the unit.
                let ingress = if bf16 { cuteafd_core::SparkIngress::Bf16 } else { cuteafd_core::SparkIngress::Fp8K32 };
                assert_eq!(model.endpoints.iter().map(|e| (e.capacity, e.ingress)).collect::<Vec<_>>(),
                    [(80.min(live), ingress), (80.min(live), ingress), (live, ingress)]);
            }
        }
        Ok(())
    }
}

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
    if std::env::var("CUTEAFD_STARTUP_MEMORY_PROBE").as_deref() == Ok("1") {
        crate::shared::inventory::startup_probe(lib, &[lib.cuda_get_device()?], phase)?;
    }
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

/// Runs `build` with its device allocations owned by `"{scope}/{component}"`
/// (ledger owner tag and default category). Category scopes nested inside
/// (`sampler`, `workspace/fp8-linear`) keep the owner, so the plan == runtime
/// check sums every byte the component allocates and reports each component.
pub(super) fn scoped<T>(scope: &str, component: &str, build: impl FnOnce() -> Result<T>) -> Result<T> {
    let label = format!("{scope}/{component}");
    let _owner = cuteafd_ffi::memory_ledger::owner(&label);
    let _scope = cuteafd_ffi::memory_ledger::scope_owned(&label);
    build()
}

/// Plan == runtime (WP6): the decode lanes' and prefill slots' device
/// allocations on each of `gpus` (their ledger owners) are exactly what the
/// planner charges (`NativePrograms::target`). A retained temporary or an
/// unmodelled owner fails startup instead of silently shrinking the pool.
pub(super) fn verify_target_charges(lib: &cuteafd_ffi::NativeLibrary, native_lib: &std::path::Path, gpus: &[i32],
    decode_rows: u32, reserve: u64, unit: u32, slots: usize, concurrency: u32) -> Result<()> {
    use cuteafd_ffi::memory_ledger::{snapshot, Space};
    let manifest = families::deepseek_v41::owners::native_manifest(native_lib, "V41_FP8_AOT.json")?;
    let programs = families::deepseek_v41::owners::NativePrograms::new(&manifest, u32::try_from(lib.sm_count()?)?)?;
    let head_mode = crate::families::deepseek_v41::v41_tensors::fp8_head();
    let window_batch = gpus.len() == 1 && super::BackboneExecution::window_batch_enabled();
    let ranks = gpus.len();
    let lanes = programs.decode_lanes(u64::from(decode_rows), ranks, head_mode, window_batch)?;
    let prefill = programs.prefill_slots(reserve, u64::from(unit), ranks, slots, u64::from(concurrency),
        head_mode, window_batch)?;
    let ledger = snapshot();
    for (rank, &gpu) in gpus.iter().enumerate() {
        let owners = ledger.by_owner(Space::Device, gpu);
        for (scope, demands, prefix) in [(super::DECODE_LANES_SCOPE, &lanes, "lane "), (super::PREFILL_SLOTS_SCOPE, &prefill, "prefill slots ")] {
            // Live bytes per component (`scoped`), and the plan's per group
            // (decode lanes' groups summed over both lanes).
            let mut live = std::collections::BTreeMap::<String, u64>::new();
            // Engram rows and gates are charged apart but one owner allocates both on two RTX.
            let component = |name: &str| if name.starts_with("engram") { "engram".to_string() } else { name.to_string() };
            for (label, &bytes) in owners.iter().filter(|(label, _)| label.starts_with(scope)) {
                let name = label.strip_prefix(scope).unwrap_or(label).trim_start_matches('/');
                *live.entry(if name.is_empty() { "(unscoped)".into() } else { component(name) }).or_default() += bytes as u64;
            }
            let mut plan = std::collections::BTreeMap::<String, u64>::new();
            for demand in demands.iter().filter(|d| usize::from(d.gpu) == rank) {
                let group = demand.group.strip_prefix(prefix).unwrap_or(&demand.group);
                let group = if prefix == "lane " { group.split_once(' ').map_or(group, |(_, g)| g) } else { group };
                // The decode lanes' window batch table belongs to their producers.
                let group = if group == "window batch table" { "cache producers" } else { group };
                *plan.entry(component(group)).or_default() += demand.bytes;
            }
            let (live_total, plan_total) = (live.values().sum::<u64>(), plan.values().sum::<u64>());
            let differing: Vec<_> = live.keys().chain(plan.keys()).collect::<std::collections::BTreeSet<_>>().into_iter()
                .filter_map(|group| { let (l, p) = (live.get(group).copied().unwrap_or(0), plan.get(group).copied().unwrap_or(0));
                    (l != p).then(|| format!("{group}: live {l} plan {p} ({:+})", l as i64 - p as i64)) }).collect();
            tracing::info!(gpu, scope, live = live_total, plan = plan_total, ?differing, "V4.1 target owners: plan == runtime");
            ensure!(live_total == plan_total && differing.is_empty(),
                "V4.1 {scope} on GPU {gpu}: allocated {live_total} B, planned {plan_total} B ({:+} B); {}",
                live_total as i64 - plan_total as i64, differing.join("; "));
        }
    }
    Ok(())
}

/// Plan == runtime for the coordinator expert workspaces (WP6): the planner's
/// charge (`expert_inventory`: slot arenas at the unit, decode lanes at their
/// width) equals what serve admits and allocates (`exact`, from the native
/// library's own plans and the TP2 owner policy).
pub(super) fn verify_expert_charges(catalog: &cuteafd_loader::OfficialV41Catalog, native_lib: &std::path::Path,
    sms: u32, ranks: usize, unit: u32, slots: usize, reserve: u64, exact: u64) -> Result<()> {
    let directory = native_lib.parent().context("native library has no directory")?;
    let programs = [directory.join("PROGRAMS.json"), directory.join("../share/PROGRAMS.json"),
        directory.join("../../share/PROGRAMS.json")].into_iter().find(|path| path.is_file())
        .context("missing PROGRAMS.json beside the native library or in its share directory")?;
    let (_, planned) = families::deepseek_v41::expert_inventory(catalog, &programs, u64::from(unit), ranks, sms,
        slots, reserve, u64::from(super::DECODE_LANE_ROWS))?;
    tracing::info!(planned, exact, ranks, unit, slots, reserve, "V4.1 expert workspaces: plan == runtime");
    ensure!(planned == exact, "V4.1 expert workspaces: planned {planned} B per GPU, serve admits {exact} B ({:+} B)",
        exact as i64 - planned as i64);
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
    // Every owner serve allocates: two decode lanes, the prefill slots' storage.
    let shape = super::AdmittedPrefill::resolve(args)?;
    let passes = shape.waves;
    let mut target = programs.target(u64::from(super::DECODE_LANE_ROWS), shape.reserve_rows() as u64,
        u64::from(capacity), 1, passes, u64::from(args.concurrency),
        crate::families::deepseek_v41::v41_tensors::fp8_head(),
        BackboneExecution::window_batch_enabled())?;
    validate_engram_rows(&target, shape.reserve_rows(), super::DECODE_LANE_ROWS as usize)?;
    // Test builds retain the materialized query reference buffer; production
    // intentionally has no such owner.
    #[cfg(test)]
    for lane in 0..2 {
        target.push(Demand::new(0, Category::Workspace,
            format!("lane {lane} test query reference"), u64::from(capacity) * 65536, Basis::Formula));
    }
    let mut startup = vec![owners::vocabulary_load_phase(&weights, &target,
        VocabularyHead::plan(catalog)? as u64)?];
    ensure!(args.prefix_cache_entries <= 128, "invalid retained-turn limit");
    let slots = cuteafd_core::prefix::mark_slots_with_host(0, u64::from(args.prefix_cache_entries), 1, u64::MAX, args.host_cache_bytes.enabled());
    let mut target_owners = owners::TargetOwners::new(weights, target, 1,
        owners::transports_bytes(u64::from(super::DECODE_LANE_ROWS), u64::from(capacity), args.peers.len(), passes)?, slots)?;
    let draft = if args.dspark {
        let mut draft =
            crate::families::deepseek_v41::v41_experts::dspark::DsparkWeights::single_inventory(
                lib, catalog, capacity, args.concurrency,
                if args.dspark_draft_limit > 5 { 7 } else { 5 })?;
        startup.extend(target_owners.draft_load_phases(&draft)?);
        draft.marks = resident::prefix_arena_bytes(slots, true, 1)[0]
            - resident::prefix_arena_bytes(slots, false, 1)[0];
        Some(draft)
    } else { None };
    if local_vision_owner(cuteafd_api::openai::vision_input_enabled(), args.vision_peers.is_some()) {
        target_owners.add_vision(Demand::new(0, Category::Workspace, "vision weights/scratch",
            crate::families::deepseek_v41::v41_vision::VisionRuntime::device_bytes(catalog, 9216)? as u64,
            Basis::Formula))?;
    }
    let fixed_demands = target_owners.fixed;
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

/// The Engram upload planes are formula owners, separate from the gates:
/// the prefill slots' one shared row plane over the `reserve` rows (each
/// in-flight unit's rows a window of it, never unit scratch) and each decode
/// lane's private planes at `decode_rows`, all on the gate GPU (GPU0).
fn validate_engram_rows(target: &[Demand], reserve: usize, decode_rows: usize) -> Result<()> {
    use cuteafd_core::memory_layout::Basis;
    let row_bytes = cuteafd_loader::ENGRAM_DEVICE_ROW_BYTES as u64;
    let rows = target.iter().filter(|d| d.group.ends_with(" engram rows")).collect::<Vec<_>>();
    let slots = rows.iter().filter(|d| d.group == "prefill slots engram rows").collect::<Vec<_>>();
    let lanes = rows.iter().filter(|d| d.group.starts_with("lane ")).collect::<Vec<_>>();
    ensure!(rows.iter().all(|d| d.gpu == 0 && d.basis == Basis::Formula)
        && rows.len() == 3 && slots.len() == 1 && lanes.len() == 2
        && slots[0].bytes == reserve as u64 * row_bytes
        && lanes.iter().all(|d| d.bytes == decode_rows as u64 * row_bytes),
        "V4.1 Engram upload admission differs from the reserve row plane and the decode lanes' rows: {:?}",
        rows.iter().map(|d| (d.group.as_str(), d.gpu, d.bytes)).collect::<Vec<_>>());
    Ok(())
}

// Remote vision reserves its tower on the Spark, never in the RTX startup bound.
fn local_vision_owner(enabled: bool, remote: bool) -> bool {
    enabled && !remote
}

#[cfg(test)]
mod startup_tests {
    /// The planner's own target list (decode lanes + prefill slot set) passes
    /// the Engram check at every pinned depth, and the check still refuses a
    /// missing, mis-sized or misplaced owner.
    #[test]
    fn engram_row_admission_matches_the_slot_set_and_decode_lanes() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let manifest = cuteafd_loader::plan::testing::write_v41_workspaces(dir.path());
        let fp8: serde_json::Value = serde_json::from_slice(&std::fs::read(
            manifest.parent().unwrap().join("V41_FP8_AOT.json"))?)?;
        let programs = super::families::deepseek_v41::owners::NativePrograms::new(&fp8, 188)?;
        for k in 2..=4u64 {
            let unit = cuteafd_loader::serving_capacity::prefill_waves::unit_rows(k);
            let reserve = (k * unit) as usize;
            let target = programs.target(80, reserve as u64, unit, 1, k as usize, 16,
                super::families::deepseek_v41::resident::Fp8Head::Off, true)?;
            super::validate_engram_rows(&target, reserve, 80)?;
            assert!(super::validate_engram_rows(&target, reserve - 64, 80).is_err(), "reserve mismatch");
            let missing: Vec<_> = target.iter().filter(|d| d.group != "lane 1 engram rows").cloned().collect();
            assert!(super::validate_engram_rows(&missing, reserve, 80).is_err(), "missing decode lane");
            let mut moved = target.clone();
            moved.iter_mut().find(|d| d.group == "prefill slots engram rows").unwrap().gpu = 1;
            assert!(super::validate_engram_rows(&moved, reserve, 80).is_err(), "off the gate GPU");
        }
        Ok(())
    }

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
    let lane = |rows: u32| LocalExpertWave::device_bytes_for(lib, rows, catalog.nvfp4().is_some());
    // One local expert wave per prefill slot at the unit, one per decode lane
    // at decode width.
    let slots = super::AdmittedPrefill::resolve(args)?.waves;
    Ok((lane(rows)?.checked_mul(slots).context("local serving lanes overflow")?
        + 2 * lane(super::DECODE_LANE_ROWS)?) as u64)
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
        pool = PoolPolicy::resolve(&cards, args.max_context_tokens as u64,
            Some((groups - tails) * 512), 512, args.peers.is_empty());
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
    let stored = {
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

/// Bind target modules before the inventory sample and first immutable weight upload.
pub(super) fn preload(
    lib: &cuteafd_ffi::NativeLibrary,
    rows: u32,
    ranks: usize,
    nvfp4: bool,
) -> Result<()> {
    use super::*;
    if !nvfp4 {
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
    {
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

    /// The planner's transport charge is exactly what serve allocates: two
    /// decode lanes' linked waves, every prefill slot's reduce output (slot 0
    /// holds the link) and the one prefill link's intake over its slots.
    #[test]
    fn transport_charge_equals_serve_allocation() -> anyhow::Result<()> {
        use crate::families::deepseek_v41::v41_experts::coordinator::{NativeTp4Wave, V41Link};
        for (rows, ranks) in [(1024u32, 3usize), (1408, 3), (2752, 4), (2752, 6), (4096, 4)] {
            for slots in 2..=4usize {
                let reserve = slots as u64 * u64::from(rows);
                let runtime = 2 * NativeTp4Wave::device_bytes_for(crate::families::deepseek_v41::v41_native_serve::DECODE_LANE_ROWS, ranks)?
                    + slots * rows as usize * 5120 * 2
                    + V41Link::prefill_device_bytes(reserve as usize, ranks)?;
                assert_eq!(cuteafd_loader::placement::families::deepseek_v41::owners::transports_bytes(u64::from(crate::families::deepseek_v41::v41_native_serve::DECODE_LANE_ROWS), u64::from(rows), ranks, slots)?, runtime as u64,
                    "{slots} x {rows} rows on {ranks} ranks");
            }
        }
        Ok(())
    }
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
    fn v41_admission_contract_synthetic_inventory() -> Result<()> {
        use cuteafd_core::memory_layout::{Basis, Category};
        use families::deepseek_v41::{owners, resident};
        for nvfp4 in [false, true] {
            let dir = tempfile::tempdir()?;
            if nvfp4 {
                cuteafd_loader::plan::testing::write_v41_nvfp4_snapshot(dir.path());
            } else {
                cuteafd_loader::plan::testing::write_v41_snapshot(dir.path());
            }
            let manifest = cuteafd_loader::plan::testing::write_v41_workspaces(dir.path());
            let fp8 = serde_json::from_slice(&std::fs::read(
                manifest.parent().unwrap().join("V41_FP8_AOT.json"))?)?;
            let programs = owners::NativePrograms::new(&fp8, 188)?;
            let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
            let config = serde_json::from_slice(&std::fs::read(dir.path().join("config.json"))?)?;
            for ranks in [1, 2] {
                let weights = resident::resident_headers(&config,
                    catalog.tensors().iter().map(|tensor| &tensor.metadata), ranks, false)
                    .context("synthetic native backbone residency")?;
                let weights = weights.into_iter().enumerate().flat_map(|(gpu, items)|
                    items.into_iter().map(move |item| Demand::new(gpu as u8, item.category,
                        item.group, item.bytes, item.basis))).collect::<Vec<_>>();
                let target = programs.target(80, 512, 256, ranks, 2, 16, resident::Fp8Head::All, ranks == 1)?;
                // The startup validators accept the planner's own target list.
                if ranks == 1 { validate_engram_rows(&target, 512, 80)?; }
                let fixed = owners::TargetOwners::new(weights, target, ranks,
                    2 * owners::transport_bytes(256, 4)?, 42)?.fixed;
                let (experts, workspace) = families::deepseek_v41::expert_inventory(
                    &catalog, &manifest, 256, ranks, 188, 2, 512, 80)?;
                // Deliberately synthetic pre-weight inventory, independent of
                // planner output. This is not a CUDA sample or ready-parity gate.
                let memory = vec![(80usize << 30, 96usize << 30); ranks];
                for (onboard, pool_tokens, concurrency, turns) in [
                    (LocalLayers::Auto, None, 16, 20),
                    (LocalLayers::Count(3), Some(0), 16, 20),
                    (LocalLayers::Count(3), Some(1 << 20), 1, 0),
                ] {
                    let mut args = args(dir.path());
                    args.peers = vec![args.peers[0].clone(); 4];
                    args.rtx_gpus = ranks as u32;
                    args.max_context_tokens = 262_144;
                    args.pool_tokens = pool_tokens;
                    args.concurrency = concurrency;
                    args.prefix_cache_entries = turns;
                    args.rtx_expert_layers = onboard;
                    let admitted = resolve_with_owners(&args, &memory, experts.clone(), workspace,
                        &vec![0; ranks], usize::from(ranks == 2), &fixed, &[], None)?;
                    assert!(admitted.placement.pool_tokens >= 262_144);
                    if let Some(pool) = pool_tokens.filter(|&n| n > 0) {
                        assert_eq!(admitted.placement.pool_tokens, pool);
                    }
                    for owner in &fixed {
                        let expected = fixed.iter().filter(|d| d.gpu == owner.gpu
                            && d.category == owner.category && d.group == owner.group)
                            .map(|d| d.bytes).sum::<u64>();
                        let actual = admitted.placement.items[usize::from(owner.gpu)].iter()
                            .filter(|item| item.category == owner.category && item.group == owner.group)
                            .map(|item| item.bytes).sum::<u64>();
                        assert_eq!(actual, expected, "{owner:?}");
                        assert!(admitted.placement.items[usize::from(owner.gpu)].iter()
                            .filter(|item| item.category == owner.category && item.group == owner.group)
                            .all(|item| item.lifetime == owner.lifetime));
                    }
                    assert_eq!(admitted.placement.attention_placement,
                        if ranks == 2 { AttentionPlacement::Layers } else { AttentionPlacement::Heads });
                    assert_eq!(admitted.placement.layers.len(), 40);
                    for (layer, assigned) in admitted.placement.layers.iter().enumerate() {
                        assert_eq!(assigned.mode, LayerMode::Whole {
                            gpu: u8::from(ranks == 2 && layer >= 20), ffn: FfnMode::Owner });
                    }
                    let over = [Demand::new(0, Category::Staging, "synthetic load overflow",
                        (80 << 30) + 1, Basis::Exact)];
                    assert!(resolve_with_owners(&args, &memory, experts.clone(), workspace,
                        &vec![0; ranks], usize::from(ranks == 2), &fixed, &over, None).is_err());
                    let mut missing_owner = fixed.clone();
                    missing_owner.push(Demand::new(0, Category::Workspace,
                        "synthetic owner overflow", 80 << 30, Basis::Exact));
                    assert!(resolve_with_owners(&args, &memory, experts.clone(), workspace,
                        &vec![0; ranks], usize::from(ranks == 2), &missing_owner, &[], None).is_err());
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
    fn explicit_byte_pool_must_cover_fixed_context_after_snapshot_tails() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"
        )))?;
        let mut args = args(dir.path());
        args.max_context_tokens = 131_072;
        args.rtx_expert_layers = LocalLayers::Count(0);
        let tails = args.concurrency as usize + 2 * args.prefix_cache_entries as usize;
        let memory = [(80usize << 30, 96usize << 30)];
        for tokens in [32_768, 131_072] {
            args.kv_pool_size = Some(crate::shared::memory::budget::ByteSize(
                (tokens / 512 + tails) * 5 * 256 * 356));
            let result = resolve(&args, &memory, vec![], 0, &[0], 0, None);
            if tokens < args.max_context_tokens as usize {
                let error = result.err().context("small explicit byte pool was admitted")?;
                assert_eq!(error.downcast_ref::<PlacementError>(), Some(&PlacementError::RequestedBelowContext {
                    requested: tokens as u64, floor: args.max_context_tokens as u64,
                }));
                assert!(error.to_string().contains("KV_POOL_SIZE"));
            } else {
                assert_eq!(result?.placement.pool_tokens, tokens as u64);
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

#[cfg(test)]
mod plan_runtime_breakdown {
    /// Prints the decode lane's planner groups beside the runtime owners'
    /// own byte functions at decode width (CUTEAFD_NATIVE_LIB: a WIP or image
    /// library; no device work).
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB"]
    fn decode_lane_groups_against_runtime_byte_functions() -> anyhow::Result<()> {
        use crate::families::deepseek_v41::*;
        let path = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIB")?);
        // SAFETY: trusted native library; only plan metadata is read.
        let lib = unsafe { cuteafd_ffi::NativeLibrary::load(&path)? };
        let manifest = super::families::deepseek_v41::owners::native_manifest(&path, "V41_FP8_AOT.json")?;
        let programs = super::families::deepseek_v41::owners::NativePrograms::new(&manifest, 188)?;
        let rows = 80u32;
        let plan = programs.decode_lanes(rows as u64, 1, super::families::deepseek_v41::resident::Fp8Head::Off, true)?;
        for d in plan.iter().filter(|d| d.group.starts_with("lane 0 ")) { eprintln!("PLAN {} {}", d.group, d.bytes); }
        let r = rows as usize;
        let lane: usize = v41_backbone_lane::BackboneLane::workspace_bytes(&lib, rows)?.into_iter().sum();
        let index: usize = v41_index_lane::IndexLane::workspace_bytes(&lib, rows)?.into_iter().sum();
        let producers = v41_backbone_execution::BackboneExecution::workspace_bytes(&lib, rows)?;
        let gates = 2 * v41_engram::layer::EngramGate::device_bytes(&lib, r)?;
        let head = v41_target_head::TargetHeadWave::device_bytes(cuteafd_core::V41_VERIFY_ROWS)?;
        for (name, bytes) in [("backbone", lane), ("index", index), ("cache producers+batch", producers),
            ("embedding", v41_target_embedding::TargetEmbeddingWave::device_bytes(r)?),
            ("engram rows", v41_engram::EngramDeviceRows::device_bytes(r)?), ("engram gates", gates),
            ("target head", head), ("target taps", v41_target_pass::TargetTapWave::device_bytes(r)?)] {
            eprintln!("RUNTIME {name} {bytes}");
        }
        Ok(())
    }
}
