//! Qwen automatic KV admission after weights/PLE and exact expert ownership are established.
use super::EngineArgs;
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::qwen4::Qwen4Config;
use cuteafd_loader::serving_capacity::qwen_cache_geometry;

use cuteafd_loader::serving_capacity::qwen_graphs::QwenAdmissionInputs;
use cuteafd_loader::placement::{families::qwen4, Baseline};

pub(super) fn spark_model(rows: usize, bf16: bool)
    -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    let capacity = u32::try_from(rows)?;
    let ingress = if bf16 { cuteafd_core::SparkIngress::Bf16 }
        else { cuteafd_core::SparkIngress::Fp8K32 };
    Ok(cuteafd_transport::worker_selection::ModelAdmission {
        capacity, compiled_capacity: capacity,
        endpoints: vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity, ingress, execution_lane: 0 }],
        rings: crate::shared::placement::spark_rings(64 << 20)?,
    })
}

/// The shared admission inputs (`serving_capacity::qwen_graphs::qwen_admission`) at serve's arguments.
#[allow(clippy::too_many_arguments)]
pub(super) fn inputs<'a>(args: &EngineArgs, cfg: &'a Qwen4Config, layers: usize, mtp: bool,
    manifest: Option<&'a serde_json::Value>, ple: Option<(u64, bool)>, future_expert_bytes: u64) -> Result<QwenAdmissionInputs<'a>> {
    let geometry = qwen_cache_geometry(cfg, layers, mtp, args.kv_format)?;
    let marks = args.planner_prefix_bytes.unwrap_or(geometry.ranks[0].retained_mark_bytes * qwen4::DEFAULT_MARK_SLOTS);
    let logits = if args.full_prefill_logits { cuteafd_loader::plan::layout::full_prefill_logits_bytes(
        "qwen4", args.prefill_rows as u64, cfg.vocab_size as u64) } else { 0 };
    Ok(QwenAdmissionInputs { prefill_shape: args.prefill_shape()?, cfg, layers, mtp, kv_format: args.kv_format, manifest, prefill_rows: args.prefill_rows as u64,
        slots: args.slots as u64, mark_bytes: marks, full_prefill_logits: logits, ple, future_expert_bytes,
        headroom: cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes.max(3 << 30) })
}

pub(super) use cuteafd_loader::families::qwen4::StepTableBytes;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct HostWorkspaceBytes {
    pub router: u64,
    pub staging: u64,
    pub ordered: u64,
    pub logits: u64,
}

impl HostWorkspaceBytes {
    #[allow(clippy::too_many_arguments)]
    pub fn new(cfg: &Qwen4Config, rows: usize, decode: bool, pages: usize, pool_pages: usize,
        spark: bool, bf16: bool, ordered: bool, ple_ids: bool, logits: usize) -> Result<Self> {
        use anyhow::Context;
        let mul = |a: usize, b: usize| a.checked_mul(b).context("Qwen host workspace size overflow");
        let add = |a: usize, b: usize| a.checked_add(b).context("Qwen host workspace size overflow");
        let table_rows = if decode { rows } else { 1 };
        let router = if spark { mul(rows, add(mul(cfg.topk, 8)?,
            if bf16 { mul(cfg.hidden, 2)? } else { add(cfg.hidden, cfg.hidden / 32)? })?)? } else { 256 };
        let row = add(8 * 3 + 4 * 4 + 24, add(mul(cfg.ple_rows(), 8)?, mul(5, mul(cfg.hidden, 2)?)?)?)?;
        let staging = add(add(256, mul(rows, row)?)?, mul(mul(table_rows, add(pages, pool_pages)?)?, 4)?)?;
        Ok(Self { router: router as u64, staging: staging as u64,
            ordered: if ordered { StepTableBytes::new(rows, decode, pages, pool_pages, cfg.ple_rows())?
                .ordered_bytes(ple_ids)? } else { 0 },
            logits: mul(mul(logits, cfg.vocab_size)?, 4)? as u64 })
    }
}

/// Admission follows the selected executor, not unused configured peers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExpertSource {
    LocalFp8,
    LocalExl3,
    Spark { ranks: usize },
    None,
}

impl ExpertSource {
    pub(super) fn selected(experts: Option<&super::engine::Experts<'_>>) -> Result<Self> {
        Ok(match experts {
            Some(super::engine::Experts::Local(_)) => Self::LocalFp8,
            Some(super::engine::Experts::LocalExl3(_)) => Self::LocalExl3,
            Some(super::engine::Experts::Spark { transport, .. }) => Self::Spark { ranks: transport.borrow().world_size() },
            Some(super::engine::Experts::SharedOnly) | None => Self::None,
            Some(super::engine::Experts::Tp2 { .. }) => anyhow::bail!("Qwen TP2 needs dual admission"),
        })
    }

    fn code_key(self) -> &'static str {
        match self { Self::LocalFp8 => "fp8", Self::LocalExl3 => "exl3", Self::Spark { .. } | Self::None => "none" }
    }

    fn spark_ranks(self) -> usize {
        match self { Self::Spark { ranks } => ranks, _ => 0 }
    }
}

/// Qwen's KV pool from one sample of free memory after weights, the PLE table and resident experts:
/// the shared admission's fixed items and, with startup graphs, the largest pool whose own graph set
/// fits beside them.
pub(super) fn pool_tokens(library: &NativeLibrary, args: &EngineArgs, inputs: &QwenAdmissionInputs<'_>,
    source: ExpertSource, automatic_context: bool) -> Result<usize> {
    let pending = crate::shared::inventory::pending_code(library, args.device, 0, false, "qwen4", source.code_key(),
        &inputs.manifest.map(|m| cuteafd_loader::placement::ProgramSet::from_manifest(m, &["qwen4"]))
            .unwrap_or_else(|| cuteafd_loader::placement::ProgramSet::from_names(&["qwen4"], [])))?;
    let enabled = super::engine::startup_graphs_enabled(
        std::env::var("CUTEAFD_QWEN4_GRAPHS").ok().as_deref(),
        std::env::var("CUTEAFD_QWEN4_STARTUP_GRAPHS").ok().as_deref());
    let current = library.cuda_get_device()?;
    library.cuda_set_device(args.device)?;
    let sample = library.cuda_memory_info();
    library.cuda_set_device(current)?;
    let (available, total) = sample?;
    let admission = QwenAdmissionInputs { prefill_shape: inputs.prefill_shape, cfg: inputs.cfg, layers: inputs.layers, mtp: inputs.mtp, kv_format: inputs.kv_format,
        manifest: inputs.manifest, prefill_rows: inputs.prefill_rows, slots: inputs.slots,
        mark_bytes: inputs.mark_bytes, full_prefill_logits: inputs.full_prefill_logits,
        ple: inputs.ple, future_expert_bytes: inputs.future_expert_bytes, headroom: inputs.headroom };
    let placement = qwen4::placement(&placement_inputs(args, admission, source,
        (available as u64, total as u64), pending, enabled, automatic_context))?;
    tracing::info!(pool_tokens = placement.pool_tokens, "Qwen shared solver admission before allocation");
    Ok(usize::try_from(placement.pool_tokens)?)
}

fn placement_inputs<'a>(args: &EngineArgs, admission: QwenAdmissionInputs<'a>, source: ExpertSource,
    (available, total): (u64, u64), pending: u64, graphs_enabled: bool, automatic_context: bool) -> qwen4::QwenInputs<'a> {
    qwen4::QwenInputs { automatic_context: automatic_context, admission, capacity_bytes: total,
        baseline: Baseline::Measured { free_bytes: available }, pending_code_bytes: pending,
        max_context: args.max_context as u64, requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        spark_ranks: source.spark_ranks(),
        startup_graph_modes: args.planner_graph_modes.filter(|_| graphs_enabled) }
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

/// Checkpoint-native whole-layer storage plus entry-owner operands, planned
/// before allocations. PLE follows its layer; draft experts remain separate.
pub(super) struct DualWeightPlan {
    pub resident: cuteafd_loader::families::qwen4::resident::Qwen4CheckpointResident,
    pub fixed: Vec<cuteafd_loader::placement::Demand>,
    pub layer_extra: Vec<Vec<cuteafd_loader::placement::LayerExtraDemand>>,
}

pub(super) fn dual_weights(args: &EngineArgs, checkpoint: &cuteafd_loader::plan::Checkpoint,
    cfg: &Qwen4Config, layers: usize, mtp: bool) -> Result<DualWeightPlan> {
    use cuteafd_loader::families::qwen4::resident::{checkpoint_resident_bytes, Qwen4Representation};
    let selected = Qwen4Representation { fp8_projections: args.fp8_decode, fp8_head: args.mtp_fp8_head };
    let resident = checkpoint_resident_bytes(checkpoint, cfg, layers, mtp, selected)?;
    let mut fixed = qwen4::dual_owner0_weights(&resident,
        args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Host);
    fixed.extend(qwen4::dual_projection_load_staging(cfg, layers, mtp, selected));
    let layer_extra = qwen4::dual_ple_demands(cfg, layers, &resident)?;
    Ok(DualWeightPlan { resident, fixed, layer_extra })
}

#[cfg(test)]
mod tests {
    use cuteafd_loader::serving_capacity::qwen_graphs::*;

    #[test]
    fn host_tables_match_full_destination_extents_and_minimum_allocations() -> anyhow::Result<()> {
        for rows in [1usize, 64, 512, 4096] {
            for decode in [false, true] {
                let tables = super::StepTableBytes::new(rows, decode, 32768, 8192, 9)?;
                let table_rows = if decode { rows } else { 1 };
                let expected: usize = [rows * 8, rows * 12, rows * 12, rows * 8,
                    rows * 4, rows * 4, rows * 8, rows * 4,
                    table_rows * 32768 * 4, table_rows * 8192 * 4, rows * 4]
                    .into_iter().map(|n| n.max(256)).sum();
                assert_eq!(tables.ordered_bytes(false)?, expected as u64);
                assert_eq!(tables.ordered_bytes(true)?, (expected + (rows * 9 * 8).max(256)) as u64);
            }
        }
        assert!(super::StepTableBytes::new(usize::MAX, false, 1, 1, 1).is_err());
        Ok(())
    }

    #[test]
    fn public_planner_ordered_tables_equal_runtime_at_forced_32gib() -> anyhow::Result<()> {
        use cuteafd_loader::plan::{plan, testing, ExpertPlacement, PlanOptions};
        use cuteafd_loader::plan::layout::LayoutOptions;
        use cuteafd_loader::serving_capacity::qwen_graphs::{qwen_pool_pages, QWEN_UNIT_PAGES};
        for mapped_ple in [false, true] {
            let dir = tempfile::tempdir()?;
            let mut config = testing::qwen4_config(4);
            if mapped_ple {
                config["text_config"]["ple_layer_ids"] = serde_json::json!([2]);
                config["text_config"]["ple_embed_dim"] = 2560.into();
                config["text_config"]["ngram_size"] = 3.into();
                config["text_config"]["heads_per_ngram"] = 8.into();
            }
            let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(&config)?;
            let mut tensors = testing::qwen4_coordinator(&config, false);
            if mapped_ple {
                tensors.push(testing::t("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                    "F8_E4M3", &[100, 160]));
            }
            testing::write_snapshot(dir.path(), &config, &tensors, None);
            for waves in [1, 2] {
                let report = plan(dir.path(), &PlanOptions {
                    placement: ExpertPlacement::Sparks { ranks: 4 },
                    layout: Some(LayoutOptions { rtx_bytes: vec![32 << 30], prefill_rows: 4096,
                        prefill_waves: waves, context_tokens: 32768, prefix_slots: Some(0),
                        host_cache_bytes: Some(0), host_memory: Some((128 << 30, 128 << 30)),
                        ..Default::default() }), ..Default::default()
                })?;
                assert!(report.solver_placement.is_some(), "{}", cuteafd_loader::plan::render(&report));
                assert_eq!((report.prefill_unit_rows, report.prefill_waves, report.prefill_span_rows),
                    (Some(4096), Some(waves), Some(4096 * waves)));
                let pool = usize::try_from(report.memory_layout.as_ref().unwrap().pool_tokens)?;
                let pages = qwen_pool_pages(pool).unwrap().max(QWEN_UNIT_PAGES);
                let runtime = super::HostWorkspaceBytes::new(&cfg, 4096, false, pages, pages / QWEN_UNIT_PAGES,
                    true, false, waves > 1, !mapped_ple, 1)?;
                let items: Vec<_> = report.host_ledger.as_ref().unwrap().items.iter()
                    .filter(|item| item.group == "staging/ordered-tables").collect();
                assert_eq!(items.len(), usize::from(waves > 1));
                assert_eq!(items.iter().map(|item| item.bytes).sum::<u64>(), runtime.ordered,
                    "4096x{waves}, mapped PLE {mapped_ple}");
            }
        }
        Ok(())
    }

    #[test]
    fn host_workspace_charges_all_lazy_consumers_for_serial_and_spark_owners() -> anyhow::Result<()> {
        use super::HostWorkspaceBytes;
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(
            &cuteafd_loader::plan::testing::qwen4_config(48))?;
        let (pages, pools) = (32768usize, 8192usize);
        for rows in [64usize, 512, 4096] {
            for decode in [false, true] {
                for spark in [false, true] {
                    for bf16 in [false, true] {
                        let host = HostWorkspaceBytes::new(&cfg, rows, decode, pages, pools,
                            spark, bf16, !decode && spark, false, if decode { 64 } else { 1 })?;
                        let table_rows = if decode { rows } else { 1 };
                        assert_eq!(host.staging as usize, 256 + rows * (8 * 3 + 4 * 4 + 24
                            + cfg.ple_rows() * 8 + 5 * cfg.hidden * 2) + table_rows * (pages + pools) * 4);
                        assert_eq!(host.router as usize, if spark { rows * (cfg.topk * 8
                            + if bf16 { cfg.hidden * 2 } else { cfg.hidden + cfg.hidden / 32 }) } else { 256 });
                        assert_eq!(host.logits as usize, (if decode { 64 } else { 1 }) * cfg.vocab_size * 4);
                        assert_eq!(host.ordered > 0, !decode && spark);
                    }
                }
            }
        }
        let full = HostWorkspaceBytes::new(&cfg, 4096, false, pages, pools, true, true, true, false, 8192)?;
        assert_eq!(full.logits, 8192 * cfg.vocab_size as u64 * 4);
        assert!(HostWorkspaceBytes::new(&cfg, usize::MAX, false, pages, pools,
            true, true, true, false, 1).is_err());
        Ok(())
    }

    #[test]
    fn spark_contract_matches_single_link_capacity_and_checkpoint_ingress() -> anyhow::Result<()> {
        for rows in [64, 2048, 4096] {
            for bf16 in [false, true] {
                let model = super::spark_model(rows, bf16)?;
                model.validate(cuteafd_core::ExpertGeometry::QWEN4)?;
                assert_eq!((model.capacity, model.compiled_capacity), (rows as u32, rows as u32));
                assert_eq!(model.endpoints.len(), 1);
                assert_eq!(model.endpoints[0].capacity, rows as u32);
                assert_eq!(model.endpoints[0].ingress, if bf16 { cuteafd_core::SparkIngress::Bf16 }
                    else { cuteafd_core::SparkIngress::Fp8K32 });
            }
        }
        Ok(())
    }

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
                    "top_k": 10, "capacity": capacity, "output_dtype": "fp32",
                    "tile": [64, 128, 64, 128], "trellis_lut": {"bytes": 32},
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

    #[test]
    fn large_ple_default_stays_mapped_and_charges_only_rows_and_scale() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_core::memory_layout::Category;
        use cuteafd_loader::families::qwen4::{resident::Qwen4CheckpointResident, Qwen4Config, Qwen4KvCache};
        use cuteafd_loader::placement::{families::qwen4, Baseline, GpuBudget, Onboard};
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let argv = ["serve", "--snapshot", "/absent/checkpoint", "--native-lib", "/absent/native.so"];
        let args = Cli::try_parse_from(argv)?.engine;
        assert!(args.table.prefetch);
        let mut config = cuteafd_loader::plan::testing::qwen4_config(48);
        config["text_config"]["ple_layer_ids"] = serde_json::json!([24]);
        config["text_config"]["ple_embed_dim"] = serde_json::json!(2560);
        config["text_config"]["ngram_size"] = serde_json::json!(3);
        config["text_config"]["heads_per_ngram"] = serde_json::json!(8);
        let cfg = Qwen4Config::from_hf(&config)?;
        // The BF16 table is about 95.4 GiB; headers alone must not promote it.
        let resident = Qwen4CheckpointResident {
            ple_table_bytes: (954 * (1u64 << 30)).div_ceil(10), ple_row_bytes: 160 * 2,
            ..Default::default()
        };
        let extra = qwen4::dual_ple_demands(&cfg, 48, &resident)?;
        let weights = vec![100 << 20; 48];
        let experts = vec![None; 48];
        // Even enough capacity for the full table must retain mapped residency.
        for capacity in [96u64 << 30, 192u64 << 30] {
            for onboard in [Onboard::Auto, Onboard::ExpertsFirst { pool_floor: 2 << 20 }] {
                let mut inputs = qwen4::QwenDualInputs { automatic_context: false,
                    owner_manifests: [None; 2],
                    admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 48, mtp: false,
                        kv_format: Qwen4KvCache::Bf16, manifest: None, prefill_rows: 4096,
                        slots: 16, mark_bytes: 0, full_prefill_logits: 0,
                        ple: Some((resident.ple_row_bytes, false)), future_expert_bytes: 0, headroom: 3 << 30 },
                    gpus: [GpuBudget { capacity_bytes: capacity, headroom_bytes: 3 << 30,
                        baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: 0 } }; 2],
                    layer_weights: &weights, layer_experts: &experts, layer_extra: &extra,
                    fixed: Vec::new(), tp2_workspace: [0; 2], pending_code_bytes: [0; 2], mark_slots: 18,
                    max_context: 131072, requested_pool: Some(2 << 20), spark_ranks: 0, onboard,
                    startup_graph_modes: None,
                };
                let mapped = qwen4::dual_request(&inputs, [0; 2])?;
                assert!(mapped.movables.is_empty());
                inputs.admission.ple = None;
                let direct = qwen4::dual_request(&inputs, [0; 2])?;
                for (group, rows) in [("decode step", 64), ("prefill step", 4096)] {
                    for gpu in 0..2 {
                        let bytes = |req: &cuteafd_loader::placement::PlacementRequest| req.fixed.iter()
                            .find(|d| d.gpu == gpu && d.group == group).unwrap().bytes;
                        assert_eq!(bytes(&mapped) - bytes(&direct),
                            rows * cfg.ple_rows() as u64 * (8 + resident.ple_row_bytes));
                    }
                }
                let planned = cuteafd_loader::placement::solve(&mapped)?;
                let mut measured = mapped.clone();
                for gpu in &mut measured.inventory.gpus {
                    gpu.baseline = Baseline::Measured { free_bytes: capacity };
                }
                assert_eq!(planned, cuteafd_loader::placement::solve(&measured)?);
                assert_eq!(planned.pool_tokens, 2 << 20);
                assert_eq!(planned.items.iter().flatten().filter(|i| i.category == Category::Tables)
                    .map(|i| (i.group.as_str(), i.bytes)).collect::<Vec<_>>(), vec![("PLE scale", 256)]);
                assert!(!planned.items.iter().flatten().any(|i| i.bytes == resident.ple_table_bytes));
            }
        }
        for (option, value) in [("--table-placement", "mapped"), ("--ple", "mapped"), ("--table-cache-mib", "0")] {
            assert!(Cli::try_parse_from(argv.into_iter().chain([option, value])).is_err());
        }
        Ok(())
    }

    #[test]
    fn dual_weight_headers_match_selected_operands_before_cuda() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::testing;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let dir = tempfile::tempdir()?;
        let mut config = testing::qwen4_config(2);
        config["text_config"]["ple_layer_ids"] = serde_json::json!([2]);
        config["text_config"]["ple_embed_dim"] = serde_json::json!(2560);
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(&config)?;
        let mut tensors = Vec::new();
        for layer in 0..2 {
            let prefix = format!("model.language_model.layers.{layer}.linear_attn");
            for (name, rows, cols) in [("in_proj_qkv", 10240, 2560), ("in_proj_z", 6144, 2560),
                ("in_proj_b", 48, 2560), ("in_proj_a", 48, 2560), ("out_proj", 2560, 6144)] {
                tensors.push(testing::t(format!("{prefix}.{name}.weight"), "BF16", &[rows, cols]));
            }
        }
        tensors.extend([
            testing::t("model.language_model.hyper_connection_mixer.hc_norm.weight", "BF16", &[2560]),
            testing::t("model.language_model.embed_tokens.weight", "BF16", &[64, 2560]),
            testing::t("lm_head.weight", "BF16", &[64, 2560]),
            testing::t("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                "F8_E4M3", &[100, 160]),
        ]);
        testing::write_snapshot(dir.path(), &config, &tensors, None);
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(dir.path())?;
        let mut args = Cli::try_parse_from(["serve", "--snapshot", dir.path().to_str().unwrap(),
            "--native-lib", "/absent/native.so"])?.engine;
        for projections in [false, true] {
            for host in [false, true] {
                args.fp8_decode = projections;
                args.token_io.embed_placement = if host { crate::shared::token_io::EmbedPlacement::Host }
                    else { crate::shared::token_io::EmbedPlacement::Gpu };
                let plan = super::dual_weights(&args, &checkpoint, &cfg, 2, false)?;
                assert_eq!(plan.resident.layer_bytes.len(), 2);
                assert_eq!(plan.resident.target_bytes, plan.resident.entry_bytes
                    + plan.resident.layer_bytes.iter().sum::<u64>());
                let staging = if projections { 84_377_600 } else { 0 };
                assert_eq!(plan.fixed.iter().map(|d| d.bytes).sum::<u64>(), plan.resident.entry_bytes
                    + plan.resident.head_bytes + if host { 0 } else { plan.resident.embedding_bytes }
                    + 2 * staging);
                let load = plan.fixed.iter().filter(|d| d.category == cuteafd_core::memory_layout::Category::Staging)
                    .collect::<Vec<_>>();
                assert_eq!(load.len(), if projections { 2 } else { 0 });
                for (gpu, demand) in load.iter().enumerate() {
                    assert_eq!((demand.gpu, demand.bytes), (gpu as u8, staging));
                    assert_eq!(demand.group, "FP8 projection load staging");
                }
                assert!(plan.fixed.iter().filter(|d| d.category != cuteafd_core::memory_layout::Category::Staging)
                    .all(|d| d.gpu == 0));
                let plan = super::dual_weights(&args, &checkpoint, &cfg, 2, false)?;
                assert_eq!(plan.layer_extra.len(), 2);
                assert!(plan.layer_extra[0].is_empty());
                assert_eq!(plan.layer_extra[1].iter().map(|d| d.bytes).sum::<u64>(), 256);
                assert!(plan.layer_extra[1].iter().all(|d| d.category == cuteafd_core::memory_layout::Category::Tables
                    && d.residency == cuteafd_loader::placement::LayerResidency::Owner));
            }
        }
        Ok(())
    }

    #[test]
    fn public_dual_planner_equals_runtime_before_allocation() -> anyhow::Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::{plan, testing, ExpertPlacement, PlanOptions};
        use cuteafd_loader::plan::layout::LayoutOptions;
        use cuteafd_loader::placement::{families::qwen4, Baseline, GpuBudget, Onboard};
        #[derive(Parser)]
        struct Cli { #[command(flatten)] engine: super::super::EngineArgs }
        for mtp in [false, true] {
        let dir = tempfile::tempdir()?;
        let (mut tensors, mut metadata) = testing::qwen4_exl3(4, 4);
        for layer in 0..4 {
            let prefix = format!("model.language_model.layers.{layer}");
            let projections = if layer == 3 {
                vec![("self_attn.q_proj", 12288, 2560), ("self_attn.k_proj", 512, 2560),
                    ("self_attn.v_proj", 512, 2560), ("self_attn.indexer.index_qk_proj", 640, 2560),
                    ("self_attn.o_proj", 2560, 6144)]
            } else {
                vec![("linear_attn.in_proj_qkv", 10240, 2560), ("linear_attn.in_proj_z", 6144, 2560),
                    ("linear_attn.in_proj_b", 48, 2560), ("linear_attn.in_proj_a", 48, 2560),
                    ("linear_attn.out_proj", 2560, 6144)]
            };
            for (name, rows, cols) in projections {
                tensors.push(testing::t(format!("{prefix}.{name}.weight"), "BF16", &[rows, cols]));
            }
        }
        tensors.extend([testing::t("model.language_model.embed_tokens.weight", "BF16", &[64, 2560]),
            testing::t("lm_head.weight", "BF16", &[64, 2560])]);
        let mut config = testing::qwen4_config(4);
        if mtp {
            config["text_config"]["mtp_num_hidden_layers"] = 1.into();
            let (draft_tensors, draft_metadata) = testing::qwen4_exl3(1, 4);
            for (name, dtype, shape) in draft_tensors {
                tensors.push((name.replace("model.language_model.layers.0", "mtp.layers.0"), dtype, shape));
            }
            for (name, storage) in draft_metadata["tensor_storage"].as_object().unwrap() {
                let name = name.replace("model.language_model.layers.0", "mtp.layers.0");
                let mut storage = storage.clone();
                let stored = storage["stored_tensors"].as_object().unwrap().iter().map(|(key, value)|
                    (key.replace("model.language_model.layers.0", "mtp.layers.0"), value.clone())).collect();
                storage["stored_tensors"] = serde_json::Value::Object(stored);
                metadata["tensor_storage"][name] = storage;
            }
            for (name, rows, cols) in [("q_proj", 12288, 2560), ("k_proj", 512, 2560),
                ("v_proj", 512, 2560), ("indexer.index_qk_proj", 640, 2560), ("o_proj", 2560, 6144)] {
                tensors.push(testing::t(format!("mtp.layers.0.self_attn.{name}.weight"), "BF16", &[rows, cols]));
            }
        }
        testing::write_snapshot(dir.path(), &config, &tensors, None);
        testing::write_quantize_config(dir.path(), &metadata);
        let share = dir.path().join("share");
        std::fs::create_dir_all(&share)?;
        let manifest_path = share.join("PROGRAMS.json");
        let manifest = serde_json::json!({"programs": [{"name": "qwen4_sparse_gqa_kv_fp8_m64",
            "scratch_bytes_at_capacity": {"scratch": 8192}, "geometry": {
                "attention_plan_capacity_rows": 64, "attention_plan_scratch": [
                    {"rows_cap": 1, "scratch": 4096}, {"rows_cap": 8, "scratch": 2048},
                    {"rows_cap": 64, "scratch": 1024}]}}]});
        let mut manifest = manifest;
        let recipe = manifest["programs"][0].clone();
        manifest["programs"][0]["card_plans"] = serde_json::json!([
            {"sm_count": 170, "scratch_bytes_at_capacity": {"scratch": 16384},
                "geometry": {"attention_plan_capacity_rows": 64, "attention_plan_scratch": [
                    {"rows_cap": 1, "scratch": 8192}, {"rows_cap": 8, "scratch": 4096},
                    {"rows_cap": 64, "scratch": 2048}]}},
            {"sm_count": 188, "scratch_bytes_at_capacity": recipe["scratch_bytes_at_capacity"],
                "geometry": recipe["geometry"]}]);
        std::fs::write(&manifest_path, serde_json::to_vec(&manifest)?)?;
        let lib = dir.path().join("lib/libcuteafd_native.so");
        let catalog = cuteafd_loader::read_expert_catalog(dir.path())?;
        let packages = qwen4::dual_packages(&catalog, &lib)?;
        for (rank, width) in [384, 256].into_iter().enumerate() {
            for capacity in [1, 16, 80] {
                let path = packages[rank].join(format!("m{capacity}"));
                std::fs::create_dir_all(&path)?;
                let m = serde_json::json!({"hidden": 2560, "intermediate": width, "experts": 512,
                    "top_k": 10, "capacity": capacity, "output_dtype": "fp32",
                    "tile": [64, 128, 64, 128], "trellis_lut": {"bytes": 32},
                    "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * width,
                        "dtype": "f16", "zero_on_create": false}}});
                std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&m)?)?;
            }
        }
        if mtp {
            let draft_package = packages[0].parent().unwrap().join("rtx-tp1");
            for capacity in [1, 16, 80] {
                let path = draft_package.join(format!("m{capacity}"));
                std::fs::create_dir_all(&path)?;
                let m = serde_json::json!({"hidden": 2560, "intermediate": 640, "experts": 512,
                    "top_k": 10, "capacity": capacity, "tile": [64, 128, 64, 128],
                    "trellis_lut": {"bytes": 32},
                    "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * 640,
                        "dtype": "f16", "zero_on_create": false}}});
                std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&m)?)?;
            }
        }
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(dir.path())?;
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::read(dir.path())?;
        let mut args = Cli::try_parse_from(["serve", "--snapshot", dir.path().to_str().unwrap(),
            "--native-lib", lib.to_str().unwrap(), "--local-experts", "--split-device", "1",
            "--prefill-rows", "16", "--max-context", "131072", "--pool-tokens", "0"])?.engine;
        args.mtp = if mtp { 3 } else { 0 };
        args.planner_prefix_bytes = Some(0);
        args.planner_graph_modes = Some((8, true));
        for (host, fp8_projections, fp8_head, attention_rows) in [(false, false, true, None),
            (true, false, true, None), (false, true, true, None), (false, false, false, None),
            (false, false, true, Some(64))] {
            for capacities in [[101_973_491_712; 2], [32 << 30, 101_973_491_712], [101_973_491_712, 32 << 30]] {
            args.attention_plan_rows = attention_rows;
            let reduction = super::super::attention_plan(&args, mtp)?;
            let manifests = qwen4::dual_program_manifests(&manifest,
                capacities.map(|bytes| cuteafd_loader::placement::ArchContext::for_device("sm_120", bytes).sms), reduction)?;
            args.fp8_decode = fp8_projections;
            args.mtp_fp8_head = fp8_head;
            args.token_io.embed_placement = if host { crate::shared::token_io::EmbedPlacement::Host }
                else { crate::shared::token_io::EmbedPlacement::Gpu };
            let report = plan(dir.path(), &PlanOptions { placement: ExpertPlacement::Local,
                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: capacities.to_vec(),
                    context_tokens: 131072, prefill_rows: 16, concurrency: 8, state_slots: Some(8),
                    prefix_slots: Some(0), native_mtp_layers: args.mtp, host_embedding: host,
                    qwen_attention_plan_rows: attention_rows,
                    qwen_representation: cuteafd_loader::families::qwen4::resident::Qwen4Representation { fp8_projections, fp8_head },
                    workspace_manifest: Some(manifest_path.clone()), ..Default::default() }), ..Default::default() })?;
            assert!(report.placement_supported, "{:?}", report.hints);
            assert_eq!(report.attention_placement, cuteafd_loader::placement::AttentionPlacement::Layers);
            let layout = report.memory_layout.unwrap();
            let weights = super::dual_weights(&args, &checkpoint, &cfg, 4, mtp)?;
            let experts = super::dual_experts(&args, &catalog, 4)?;
            let mut fixed = weights.fixed;
            fixed.extend(experts.transport);
            if mtp { fixed.extend(qwen4::dual_draft_demands(&catalog, &lib, 16)?); }
            let admission = super::inputs(&args, &cfg, 4, mtp, Some(&manifests[0]), None, 0)?;
            let runtime = qwen4::dual_placement(&qwen4::QwenDualInputs { automatic_context: false, owner_manifests: manifests.each_ref().map(Some), admission,
                gpus: [0, 1].map(|rank| GpuBudget { capacity_bytes: capacities[rank],
                    headroom_bytes: 3 << 30, baseline: Baseline::Measured { free_bytes: capacities[rank]
                        - layout.devices[rank].items.iter().filter(|item| item.group == "context+modules")
                            .map(|item| item.bytes).sum::<u64>() } }),
                layer_weights: &weights.resident.layer_bytes, layer_experts: &experts.costs,
                layer_extra: &weights.layer_extra, fixed, tp2_workspace: experts.workspace,
                pending_code_bytes: [0; 2], mark_slots: 0, max_context: 131072, requested_pool: None,
                spark_ranks: 0, onboard: Onboard::Auto, startup_graph_modes: args.planner_graph_modes })?;
            assert_eq!(runtime.pool_tokens, if capacities.iter().any(|&bytes| bytes <= 32 << 30) { 1 << 20 } else { 2 << 20 });
            assert_eq!(runtime.pool_tokens, layout.pool_tokens);
            assert_eq!(runtime.tp2.unwrap().layers, 4);
            assert!(layout.notes.contains(&format!("placement: {}", runtime.summary())));
            for rank in 0..2 {
                let planned = layout.devices[rank].items.iter().filter(|item| item.group != "context+modules")
                    .cloned().collect::<Vec<_>>();
                assert_eq!(runtime.items[rank], planned, "rank {rank}, MTP={mtp}, attention={attention_rows:?}");
            }
        }
        }
        std::fs::remove_file(packages[1].join("m80/v41_exl3.json"))?;
        assert!(super::dual_experts(&args, &catalog, 4).is_err());
        }
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
        // The coordinator headers serve-qwen4 loads: the planner charges their exact
        // resident inventory and refuses a layout it cannot derive from them.
        let config = testing::qwen4_config(48);
        testing::write_snapshot(snapshot.path(), &config, &testing::qwen4_coordinator(&config, false), None);
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::read(snapshot.path())?;
        for (capacity, kv_format) in [32_u64 << 30, 101_973_491_712].into_iter().flat_map(|capacity|
            [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8].map(|format| (capacity, format))) {
            for pool in [None, Some(32768)] {
                let context = pool.unwrap_or(131072);
                let report = plan(snapshot.path(), &PlanOptions {
                    placement: ExpertPlacement::Local,
                    layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![capacity], context_tokens: context,
                        pool_tokens: pool, qwen_kv: kv_format, concurrency: 8, state_slots: Some(8), prefix_slots: Some(0),
                        ..Default::default() }), ..Default::default()
                })?;
                assert!(report.placement_supported, "{}", cuteafd_loader::plan::render(&report));
                let layout = report.memory_layout.unwrap();
                let device = &layout.devices[0];
                assert!(device.items.iter().any(|i| i.group == "target incl PLE projections" && i.bytes > 0));
                let baseline = device.items.iter().filter(|i| !matches!(i.group.as_str(), "records" | "state" |
                    "marks" | "graphs" | "graph growth" | "decode step" | "prefill step"))
                    .map(|i| i.bytes).sum::<u64>();
                let mut args = Cli::try_parse_from(["serve", "--snapshot", snapshot.path().to_str().unwrap(),
                    "--native-lib", "/nonexistent/lib.so", "--max-context", &context.to_string(), "--slots", "8",
                    "--pool-tokens", &pool.unwrap_or(0).to_string(), "--kv-cache", &kv_format.to_string()])?.engine;
                args.planner_prefix_bytes = Some(0);
                args.planner_graph_modes = Some((8, true));
                let inputs = super::inputs(&args, &cfg, 48, false, None, None, 0)?;
                let placement = qwen4::placement(&qwen4::QwenInputs { automatic_context: false, admission: inputs, capacity_bytes: capacity,
                    baseline: Baseline::Measured { free_bytes: capacity - baseline }, pending_code_bytes: 0,
                    max_context: context, requested_pool: pool, spark_ranks: 0,
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
    fn selected_local_source_ignores_configured_peers_for_admission() -> anyhow::Result<()> {
        use super::{placement_inputs, ExpertSource};
        use clap::Parser;
        use cuteafd_loader::placement::families::qwen4;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            engine: super::super::EngineArgs,
        }
        let cfg = cuteafd_loader::families::qwen4::Qwen4Config::from_hf(
            &cuteafd_loader::plan::testing::qwen4_config(48))?;
        let local = Cli::try_parse_from(["serve", "--snapshot", "/absent/checkpoint", "--native-lib", "/absent/native.so",
            "--local-experts", "--max-context", "32768"])?.engine;
        let mut with_peers = local.clone();
        with_peers.peers = Some("127.0.0.1:1234,127.0.0.1:1235".into());
        let input = |args, source, pending| -> anyhow::Result<_> {
            Ok(placement_inputs(args, super::inputs(args, &cfg, 48, false, None, None, 0)?,
                source, (96 << 30, 96 << 30), pending, false, false))
        };
        for (source, key) in [(ExpertSource::LocalFp8, "fp8"), (ExpertSource::LocalExl3, "exl3")] {
            assert_eq!(source.code_key(), key);
            assert_eq!(source.spark_ranks(), 0);
            let pending = cuteafd_loader::placement::loaded_code("qwen4", source.code_key(), false, 0)
                .unwrap().pending(0, 0);
            let plain = input(&local, source, pending)?;
            let peers = input(&with_peers, source, pending)?;
            assert_eq!(peers.pending_code_bytes, pending);
            assert_eq!(peers.spark_ranks, 0);
            assert_eq!(qwen4::request(&plain, 0)?, qwen4::request(&peers, 0)?);
            assert_eq!(qwen4::placement(&plain)?, qwen4::placement(&peers)?);
        }
        let spark = ExpertSource::Spark { ranks: 4 };
        assert_eq!(spark.code_key(), "none");
        // The selected transport's actual world, not the configured two peers.
        assert_eq!(input(&with_peers, spark, 0)?.spark_ranks, 4);
        // Runtime's measured baseline already includes the selected transport.
        let request = qwen4::request(&input(&with_peers, spark, 0)?, 0)?;
        assert!(matches!(request.inventory.gpus[0].baseline, cuteafd_loader::placement::Baseline::Measured { .. }));
        assert!(request.fixed.iter().all(|d| d.group != "Spark intake planes"));
        assert!(qwen4::request(&input(&local, ExpertSource::LocalExl3, 0)?, 0)?.fixed.iter()
            .all(|d| d.group != "Spark intake planes"));
        assert_eq!(ExpertSource::selected(None)?, ExpertSource::None);
        let shared = super::super::engine::Experts::SharedOnly;
        assert_eq!(ExpertSource::selected(Some(&shared))?, ExpertSource::None);
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
