//! Qwen automatic KV admission after weights/PLE and exact expert ownership are established.
use super::EngineArgs;
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::qwen4::Qwen4Config;
use cuteafd_loader::serving_capacity::qwen_cache_geometry;

use cuteafd_loader::serving_capacity::qwen_graphs::QwenAdmissionInputs;
use cuteafd_loader::placement::{families::qwen4, Baseline};

/// The shared admission inputs (`serving_capacity::qwen_graphs::qwen_admission`) at serve's arguments.
#[allow(clippy::too_many_arguments)]
pub(super) fn inputs<'a>(args: &EngineArgs, cfg: &'a Qwen4Config, layers: usize, mtp: bool,
    manifest: Option<&'a serde_json::Value>, ple: Option<(u64, bool)>, future_expert_bytes: u64) -> Result<QwenAdmissionInputs<'a>> {
    let geometry = qwen_cache_geometry(cfg, layers, mtp, args.kv_format)?;
    let marks = args.planner_prefix_bytes.unwrap_or(geometry.ranks[0].retained_mark_bytes * qwen4::DEFAULT_MARK_SLOTS);
    let logits = if args.full_prefill_logits { cuteafd_loader::plan::layout::full_prefill_logits_bytes(
        "qwen4", args.prefill_rows as u64, cfg.vocab_size as u64) } else { 0 };
    Ok(QwenAdmissionInputs { cfg, layers, mtp, kv_format: args.kv_format, manifest, prefill_rows: args.prefill_rows as u64,
        slots: args.slots as u64, mark_bytes: marks, full_prefill_logits: logits, ple, future_expert_bytes,
        headroom: cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes.max(3 << 30) })
}

/// Qwen's KV pool from one sample of free memory after weights, the PLE table and resident experts:
/// the shared admission's fixed items and, with startup graphs, the largest pool whose own graph set
/// fits beside them.
pub(super) fn pool_tokens(library: &NativeLibrary, args: &EngineArgs, inputs: &QwenAdmissionInputs<'_>) -> Result<usize> {
    let experts = if args.peers.is_some() { "none" } else if inputs.future_expert_bytes > 0 { "exl3" } else { "fp8" };
    let pending = crate::shared::inventory::pending_code(library, args.device, 0, false, "qwen4", experts)?;
    let enabled = super::engine::startup_graphs_enabled(
        std::env::var("CUTEAFD_QWEN4_GRAPHS").ok().as_deref(),
        std::env::var("CUTEAFD_QWEN4_STARTUP_GRAPHS").ok().as_deref());
    let current = library.cuda_get_device()?;
    library.cuda_set_device(args.device)?;
    let sample = library.cuda_memory_info();
    library.cuda_set_device(current)?;
    let (available, total) = sample?;
    let admission = QwenAdmissionInputs { cfg: inputs.cfg, layers: inputs.layers, mtp: inputs.mtp, kv_format: inputs.kv_format,
        manifest: inputs.manifest, prefill_rows: inputs.prefill_rows, slots: inputs.slots,
        mark_bytes: inputs.mark_bytes, full_prefill_logits: inputs.full_prefill_logits,
        ple: inputs.ple, future_expert_bytes: inputs.future_expert_bytes, headroom: inputs.headroom };
    let placement = qwen4::placement(&qwen4::QwenInputs { admission, capacity_bytes: total as u64,
        baseline: Baseline::Measured { free_bytes: available as u64 }, pending_code_bytes: pending,
        max_context: args.max_context as u64, requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        spark_ranks: usize::from(args.peers.is_some()),
        startup_graph_modes: args.planner_graph_modes.filter(|_| enabled) })?;
    tracing::info!(pool_tokens = placement.pool_tokens, "Qwen shared solver admission before allocation");
    Ok(usize::try_from(placement.pool_tokens)?)
}

/// Static rank packages, selected before either device allocates experts. The
/// loader manifest plan and backend DSO ABI must agree on widths and bytes.
pub(super) struct DualExpertPlan {
    pub packages: [std::path::PathBuf; 2],
    pub costs: Vec<Option<cuteafd_loader::placement::ExpertCost>>,
    pub workspace: [u64; 2],
    pub transport: Vec<cuteafd_loader::placement::Demand>,
    pub wire: bool,
}

pub(super) fn dual_experts(args: &EngineArgs, catalog: &cuteafd_loader::OfficialV41Catalog,
    layers: usize) -> Result<DualExpertPlan> {
    use anyhow::{ensure, Context};
    use crate::shared::experts::rtx::{exl3::Exl3Tp2, fp8moe::Fp8MoeTp2};
    use cuteafd_loader::formats::fp8_experts::Slicing;
    let rows = args.prefill_rows.max(super::engine::DECODE_ROWS);
    let costs = qwen4::dual_expert_costs(catalog, layers)?;
    let wire = catalog.exl3().is_some();
    let (packages, backend_workspace) = if let Some(exl3) = catalog.exl3() {
        let name = crate::shared::experts::exl3::package_name("qwen4", exl3.decoder_tiers());
        let root = args.native_lib.parent().unwrap_or(std::path::Path::new(".")).join("exl3").join(name);
        let packages = [0, 1].map(|rank| root.join(format!("rtx-tp2-rank{rank}")));
        let workspace = [Exl3Tp2::workspace_bytes_for(&packages[0], catalog.routed_experts().hidden, rows)?,
            Exl3Tp2::workspace_bytes_for(&packages[1], catalog.routed_experts().hidden, rows)?];
        (packages, workspace)
    } else {
        let tensors = catalog.fp8().context("Qwen dual experts need EXL3/FP8/NVFP4")?;
        let directory = args.fp8_package.clone().unwrap_or_else(||
            crate::shared::experts::fp8::package_directory(&args.native_lib, 2, tensors.format()));
        let plans = [Fp8MoeTp2::plan(tensors, &directory, 0, rows)?,
            Fp8MoeTp2::plan(tensors, &directory, 1, rows)?];
        for rank in 0..2 {
            ensure!(plans[rank].slicing == Slicing::Blocks(128),
                "Qwen dual experts require complete-H128 exact rank packages; padded TP2 is unsupported");
            for cost in costs.iter().flatten() {
                ensure!(cost.half[rank].resident == plans[rank].resident_layer_bytes as u64
                    && cost.half[rank].staging == plans[rank].staging_bytes as u64,
                    "Qwen TP2 header residency disagrees with backend rank {rank}");
            }
        }
        let workspace = plans.each_ref().map(|plan| plan.workspace_bytes);
        (plans.map(|plan| plan.directory), workspace)
    };
    let workspace = qwen4::dual_expert_workspace(catalog,
        packages.each_ref().map(|p| p.as_path()), rows as u64)?;
    for rank in 0..2 {
        ensure!(workspace[rank] == backend_workspace[rank] as u64,
            "Qwen TP2 manifest workspace disagrees with backend rank {rank}");
    }
    let shape = catalog.routed_experts();
    let transport = qwen4::QwenTp2Rows::new(shape.hidden as u64, shape.topk as u64, rows as u64, wire)?.demands()?;
    Ok(DualExpertPlan { packages, costs, workspace, transport, wire })
}

#[cfg(test)]
mod tests {
    use cuteafd_loader::serving_capacity::qwen_graphs::*;

    #[test]
    fn dual_exl3_packages_match_shared_manifest_plan_before_cuda() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::testing;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let dir = tempfile::tempdir()?;
        let (tensors, metadata) = testing::qwen4_exl3(1, 4);
        testing::write_snapshot(dir.path(), &testing::qwen4_config(1), &tensors, None);
        testing::write_quantize_config(dir.path(), &metadata);
        let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
        let mut args = Cli::try_parse_from(["serve", "--snapshot", dir.path().to_str().unwrap(),
            "--native-lib", "/absent/native.so", "--prefill-rows", "16"])?.engine;
        args.native_lib = dir.path().join("libcuteafd_native.so");
        // Qwen admits at least the 64-row decode shape, hence m80 as well.
        let package = crate::shared::experts::exl3::package_name("qwen4", catalog.exl3().unwrap().decoder_tiers());
        for (rank, width) in [384, 256].into_iter().enumerate() {
            for capacity in [1, 16, 80] {
                let path = dir.path().join("exl3").join(&package)
                    .join(format!("rtx-tp2-rank{rank}/m{capacity}"));
                std::fs::create_dir_all(&path)?;
                let manifest = serde_json::json!({"hidden": 2560, "intermediate": width, "experts": 512,
                    "top_k": 10, "capacity": capacity, "output_dtype": "fp32", "trellis_lut": {"bytes": 32},
                    "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * width,
                        "dtype": "f16", "zero_on_create": false},
                        "state": {"allocation": "state", "bytes": 16, "zero_on_create": true}}});
                std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&manifest)?)?;
            }
        }
        let plan = super::dual_experts(&args, &catalog, 1)?;
        assert!(plan.wire);
        assert_eq!(plan.costs.len(), 1);
        assert!(plan.costs[0].unwrap().half[0].resident > plan.costs[0].unwrap().half[1].resident);
        assert_eq!(plan.workspace, [80 * 384 + 3 * (32 + 16) + 97 * 2560 * 2 + 64 * 2560 * 4,
            80 * 256 + 3 * (32 + 16) + 97 * 2560 * 2 + 64 * 2560 * 4]);
        std::fs::remove_file(plan.packages[1].join("m80/v41_exl3.json"))?;
        assert!(super::dual_experts(&args, &catalog, 1).is_err());
        Ok(())
    }

    fn set(count: u64) -> Option<cuteafd_loader::placement::GraphSet> {
        Some(cuteafd_loader::placement::GraphSet::new(&[count], QWEN_GRAPH_BYTES_PER_GRAPH, QWEN_GRAPH_MARGIN_PERCENT,
            QWEN_GRAPH_MARGIN_BYTES, 0, cuteafd_loader::placement::Lifetime::Startup))
    }

    #[test]
    fn planner_equals_runtime_qwen4() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::{plan, testing, ExpertPlacement, PlanOptions};
        use cuteafd_loader::plan::layout::LayoutOptions;
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        use cuteafd_loader::placement::{families::qwen4, Baseline};
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let snapshot = tempfile::tempdir()?;
        testing::write_snapshot(snapshot.path(), &testing::qwen4_config(48), &[], None);
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::read(snapshot.path())?;
        for (capacity, kv_format) in [32_u64 << 30, 101_973_491_712].into_iter().flat_map(|capacity|
            [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8].map(|format| (capacity, format))) {
            for pool in [None, Some(32768)] {
                let report = plan(snapshot.path(), &PlanOptions {
                    placement: ExpertPlacement::Local,
                    layout: Some(LayoutOptions { rtx_bytes: vec![capacity], context_tokens: 131072,
                        pool_tokens: pool, qwen_kv: kv_format, concurrency: 8, state_slots: Some(8), prefix_slots: Some(0),
                        ..Default::default() }), ..Default::default()
                })?;
                let layout = report.memory_layout.unwrap();
                let device = &layout.devices[0];
                let baseline = device.items.iter().filter(|i| !matches!(i.group.as_str(), "records" | "state" |
                    "marks" | "graphs" | "graph growth" | "decode step" | "prefill step"))
                    .map(|i| i.bytes).sum::<u64>();
                let mut args = Cli::try_parse_from(["serve", "--snapshot", snapshot.path().to_str().unwrap(),
                    "--native-lib", "/nonexistent/lib.so", "--max-context", "131072", "--slots", "8",
                    "--pool-tokens", &pool.unwrap_or(0).to_string(), "--kv-format", &kv_format.to_string()])?.engine;
                args.planner_prefix_bytes = Some(0);
                args.planner_graph_modes = Some((8, true));
                let inputs = super::inputs(&args, &cfg, 48, false, None, None, 0)?;
                let placement = qwen4::placement(&qwen4::QwenInputs { admission: inputs, capacity_bytes: capacity,
                    baseline: Baseline::Measured { free_bytes: capacity - baseline }, pending_code_bytes: 0,
                    max_context: 131072, requested_pool: pool, spark_ranks: 0,
                    startup_graph_modes: args.planner_graph_modes })?;
                assert_eq!(placement.pool_tokens, layout.pool_tokens);
                for item in &placement.items[0] {
                    assert!(device.items.contains(item), "runtime item absent in plan: {item:?}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn auto_admission_subtracts_measured_graph_reserve_and_separate_headroom() {
        let (available, headroom, per_token) = (4_u64 << 30, 1_u64 << 30, 4096);
        let (tokens, graphs) = qwen_graph_pool(available, headroom, per_token, 2_097_152, None, |_| set(12397)).unwrap();
        let reserve = graphs.bytes(0);
        assert_eq!(reserve, 12397 * QWEN_GRAPH_BYTES_PER_GRAPH + QWEN_GRAPH_MARGIN_BYTES);
        assert_eq!(tokens, (available - headroom - reserve) / per_token / 256 * 256);
        assert!(tokens * per_token + headroom + reserve <= available);
    }

    #[test]
    fn admission_recounts_geometry_after_pool_shrinks_and_rejects_fixed_overcommit() {
        let count = |tokens| set(if tokens >= 32768 { 21560 } else { 7840 });
        let (tokens, graphs) = qwen_graph_pool(4 << 30, 512 << 20, 65536, 65536, None, count).unwrap();
        assert!(tokens < 32768);
        assert_eq!(graphs.ranks[0].executables, 7840);
        assert!(tokens * 65536 + (512 << 20) + graphs.bytes(0) <= 4 << 30);
        assert!(qwen_graph_pool(4 << 30, 512 << 20, 65536, 65536, Some(32768), count).is_err());
    }

    #[test]
    fn margin_and_fixed_pool_rounding_do_not_spend_headroom() {
        let measured = 21560 * QWEN_GRAPH_BYTES_PER_GRAPH;
        assert_eq!(set(21560).unwrap().bytes(0), measured + measured.div_ceil(10));
        let (tokens, graphs) = qwen_graph_pool(8 << 30, 3 << 30, 28422, 2097152, Some(73727), |_| set(12397)).unwrap();
        assert_eq!(tokens, 73728);
        assert!(tokens * 28422 + graphs.bytes(0) + (3 << 30) <= 8 << 30);
        assert!(qwen_graph_pool(256 << 20, 0, 28422, 2097152, None, |_| set(12397)).is_err());
    }

    #[test]
    fn graph_counts_match_actual_pool_context_layers_and_serving_modes() {
        use super::super::engine::serving_graph_count;
        assert_eq!(serving_graph_count(8192, 73728, 2051, 16, true, 48).unwrap(), 12397);
        assert_eq!(serving_graph_count(8192, 2097152, 2051, 16, true, 48).unwrap(), 12397);
        assert_eq!(serving_graph_count(8192, 73728, 2051, 16, false, 48).unwrap(), 4508);
        assert_eq!(serving_graph_count(32768, 32768, 2051, 16, true, 48).unwrap(), 21560);
        assert!(serving_graph_count(32768, 4096, 2051, 16, true, 48).unwrap() < 21560);
        assert_eq!(qwen_startup_graphs(8192, 73728, 2051, 16, true, 48).unwrap().ranks[0].executables, 12397);
    }
}
