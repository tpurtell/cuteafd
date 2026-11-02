//! GLM 5.x (glm_moe_dsa) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
use crate::shared::speculation::dflash;
pub(crate) mod engine;
pub(crate) mod prefix;
mod packed_check;
pub(crate) mod serve;
pub(crate) mod landing;
pub(crate) mod native_dense;
pub(crate) mod push;
pub(crate) mod weights;
pub(crate) mod core_runtime;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::glm5::GlmDsaConfig;
use crate::shared::spark_intake::SparkLink;

use std::ffi::c_void;
use std::path::PathBuf;
use std::time::Instant;

/// One enum for the engine flag and `cuteafd plan`, so the planner charges what admission charges.
pub(crate) use cuteafd_loader::plan::exlr8::PrefillActivations as Exlr8PrefillActivations;

/// What every GLM command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Shared GQA drafter-ring storage (independent of weight precision).
    #[arg(long, default_value = "fp8")]
    pub draft_kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
    /// Checkpoint snapshot (the EXL3 publication carries the dense weights).
    #[arg(long)]
    pub snapshot: PathBuf,
    /// Published native EXLR8 expert manifest (for example, up1).
    #[arg(long)]
    pub exlr8_manifest: Option<String>,
    /// Native EXLR8 trellis prefill activations; decode/verify always keep A16.
    #[arg(long, value_parser = crate::cli::exlr8_prefill_activations_parser(),
        default_value_t = Exlr8PrefillActivations::A16)]
    pub exlr8_prefill_activations: Exlr8PrefillActivations,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Split every layer's attention heads (q_b, kv_b, o_proj; 32 of 64 each)
    /// and the dense / shared-expert MLPs over --device and this second GPU,
    /// with the small latent projection, latent cache and DSA indexer
    /// replicated; the partial sums meet over peer memory. Router, routed
    /// experts, LM head and drafter stay on --device.
    #[arg(long)]
    pub split_device: Option<i32>,
    /// Run only the first N layers (dense layers need no experts).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Longest sequence; 0 selects checkpoint full, bounded by compiled support and the admitted pool.
    #[arg(long, default_value_t = 0)]
    pub max_context: usize,
    /// Tokens the shared latent/index cache pool holds across sequences; 0
    /// sizes it from what every GPU has left after weights and the planner's
    /// workspace, graph and drafter costs (up to the common 2M-token target).
    #[arg(long, default_value_t = 262_144)]
    pub pool_tokens: usize,
    /// Per-unit rows; 0 selects the shared topology policy.
    #[arg(long, default_value_t = 0)]
    pub prefill_rows: usize,
    /// Shared prefill span; 0 resolves to waves times per-unit rows.
    #[arg(long, default_value_t = 0)]
    pub prefill_span_rows: usize,
    /// Prefill units in flight (0 = shared topology policy, 1 = serial).
    #[arg(long, default_value_t = 0,
        value_parser = clap::value_parser!(u8).range(0..=4))]
    pub prefill_waves: u8,
    /// Pack queued short prompts; exact preserves loaded expert arithmetic routes.
    #[arg(long, value_enum)]
    pub prefill_packing: Option<crate::shared::prefill_packing::PackingMode>,
    /// Pool-first by default; local requests require an exported RTX backend.
    #[arg(long, env = "RTX_EXPERT_LAYERS", default_value = "auto")]
    pub rtx_expert_layers: cuteafd_loader::placement::Onboard,
    /// Spark expert ranks in TP order (HOST:PORT,...), for MoE layers.
    #[arg(long)]
    pub peers: Option<String>,
    /// DFlash2 drafter snapshot (incoai/GLM-5.3-DFlash2); taps the target
    /// layers and drafts on the coordinator GPU.
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Maximum members of one draft batch, independent of context slots.
    #[arg(long, default_value_t = cuteafd_loader::families::glm5::draft_representation::DEFAULT_DRAFT_SEQUENCES)]
    pub draft_sequences: usize,
    /// Context slots (default max(20, draft_sequences)); target head is shared.
    #[arg(long)]
    pub draft_context_slots: Option<usize>,
    /// Explicit calibration-free E4M3 quantization of own drafter GEMMs.
    /// Unset/true: E4M3 single copy (measured faster); false keeps checkpoint BF16.
    #[arg(long, action = clap::ArgAction::Set)]
    pub draft_fp8: Option<bool>,
    /// Scale rule of the FP8 copies made from BF16 weights at load: amax /
    /// 448, the smallest power of two >= it (pow2), or per block whichever of
    /// the two leaves the smaller error (best).
    #[arg(long, value_enum, default_value_t = crate::shared::fp8_linear::Fp8Scales::Amax)]
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    #[command(flatten)]
    pub l2: crate::shared::l2_prefetch::L2PrefetchArgs,
    /// Keep every prefill row's logits (glm-golden --nll; 2.5 GiB at 4096 rows).
    #[arg(long, hide = true)]
    pub full_prefill_logits: bool,
    /// Benchmarks only: MoE layers skip the Spark exchange (routed experts
    /// contribute zero) but keep the route/wire download, the host sync, the
    /// shared expert, the plane uploads and the reduce, timing the
    /// coordinator alone. Outputs are not the model's.
    #[arg(long, hide = true)]
    pub skip_routed_experts: bool,
    /// Prefill projections over the checkpoint's FP8 weights run W8A16 (the
    /// weights widened to BF16 in-kernel: the former BF16 prefill bit for bit)
    /// instead of W8A8 (E4M3 activations per row and 128-K block with FP32
    /// scales, the official FP8 release's served numerics).
    #[arg(long)]
    pub prefill_w8a16: bool,
    #[command(flatten)]
    pub token_io: crate::shared::token_io::TokenIoArgs,
    /// Attention over the two GPUs: auto (the recorded default: context where this build
    /// carries its programs), context (KV pages split by parity, a q/candidate/partial
    /// exchange per layer), or heads (replicated latent, head-split attention).
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto")]
    pub attention_placement: String,
    /// Numerical decode/verify reduction plan rows; automatic uses the active drafter width.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLAN_ROWS")]
    pub attention_plan_rows: Option<usize>,
}

fn resolve_prefill(args: &mut EngineArgs) -> Result<()> {
    use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides, MAX_WAVES};
    let ranks = args.peers.as_deref().map_or(0, |peers| peers.split(',').count());
    let layers = cuteafd_loader::placement::attention::parse(&args.attention_placement)
        .map_err(anyhow::Error::msg)? == Some(cuteafd_loader::placement::AttentionPlacement::Layers);
    let family = cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::glm::GLM5);
    let shape = prefill_shape(ranks, PrefillLimits {
        max_waves: if ranks == 0 { 1 } else if layers { cuteafd_loader::placement::LAYER_PIPELINE_MAX_LANES } else { MAX_WAVES },
        ..family
    }, PrefillOverrides {
        unit_rows: (args.prefill_rows > 0).then_some(args.prefill_rows as u64),
        waves: (args.prefill_waves > 0).then_some(u64::from(args.prefill_waves)),
        span_rows: (args.prefill_span_rows > 0).then_some(args.prefill_span_rows as u64),
        reserve_rows: None,
    })?;
    args.prefill_rows = usize::try_from(shape.unit_rows)?;
    args.prefill_waves = u8::try_from(shape.waves)?;
    args.prefill_span_rows = usize::try_from(shape.span_rows)?;
    Ok(())
}

fn native_activation(args: &EngineArgs, first_moe: usize, layers: usize, serving: bool) -> Result<()> {
    use cuteafd_loader::placement::AttentionPlacement;
    ensure!(!args.skip_routed_experts, "EXLR8 requires real Spark experts");
    let peers = args.peers.as_deref().context("EXLR8 requires Spark peers")?;
    ensure!(!peers.is_empty() && peers.split(',').all(|peer| !peer.trim().is_empty()),
        "EXLR8 requires nonempty Spark peers");
    let partial = args.layers.is_some_and(|count| count > first_moe && count < layers);
    if !serving && partial {
        ensure!(args.split_device.is_none_or(|device| device != args.device),
            "EXLR8 partial execution requires distinct coordinators");
        return Ok(());
    }
    ensure!(matches!(args.exlr8_manifest.as_deref(), Some("up1" | "bit325")),
        "EXLR8 full execution currently admits up1 or bit325 only");
    ensure!(args.split_device.is_none_or(|device| device != args.device)
        && (1..=8).contains(&peers.split(',').count()) && args.layers.is_none_or(|count| count == layers),
        "EXLR8 full execution requires one or two distinct coordinators, 1..=8 Spark ranks and all backbone layers");
    let attention = cuteafd_loader::placement::attention::parse(&args.attention_placement)
        .map_err(anyhow::Error::msg)?;
    let expected = if args.split_device.is_some() { AttentionPlacement::Context } else { AttentionPlacement::Heads };
    ensure!(attention.is_none_or(|placement| placement == expected),
        "EXLR8 full execution requires {expected} attention placement for the selected coordinators");
    Ok(())
}

/// Bytes one coordinator GPU admits for native dense execution before any allocation: decode
/// scratch plus the prefill scratch every slot shares and, with A8 prefill, its MX8 records.
/// `cuteafd plan --layout` charges the same bytes (`plan::layout::glm5::native_layout`).
fn native_dense_charge(cfg: &GlmDsaConfig, gpus: usize, args: &EngineArgs) -> u64 {
    cuteafd_loader::plan::exlr8::dense::lane_charge_bytes(cfg, gpus, engine::DECODE_ROWS, args.prefill_rows,
        args.exlr8_prefill_activations == Exlr8PrefillActivations::A8) as u64
}

use cuteafd_loader::plan::spark_executions::SparkBackend;

/// The decode link plus one prefill link carrying every wave in flight, over
/// typed owners from the shared admission (one prefill scratch at the ladder
/// maximum, waves only buffer per-unit I/O).
fn spark_model(backend: SparkBackend, waves: usize, unit_rows: u32, bf16: bool)
    -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    ensure!(waves > 0, "GLM Spark admission requires at least one prefill wave");
    let native = backend == SparkBackend::Exlr8;
    let ingress = if native { cuteafd_core::SparkIngress::Fp8K32 }
        else if bf16 { cuteafd_core::SparkIngress::Bf16 } else { cuteafd_core::SparkIngress::Fp8K32 };
    let executions = cuteafd_loader::plan::spark_executions::admit_family("glm5", &backend.exports(), None,
        u64::from(unit_rows), waves as u64)?;
    let rings = crate::shared::placement::spark_rings(if native {
        cuteafd_transport::expert::native_frame_budget(cuteafd_core::ExpertGeometry::GLM5, executions.capacity())?
    } else { 64 << 20 })?;
    let model = cuteafd_transport::worker_selection::ModelAdmission::new(
        vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity: engine::DECODE_ROWS as u32, ingress, partial_reply_bytes: None },
            cuteafd_core::SparkEndpoint { ring_depth: None, capacity: unit_rows, ingress, partial_reply_bytes: None }], rings, executions)?;
    validate_spark_endpoint_count(model.endpoints.len(), waves)?;
    Ok(model)
}

fn validate_spark_endpoint_count(endpoints: usize, waves: usize) -> Result<()> {
    ensure!(endpoints == cuteafd_loader::plan::layout::spark_endpoints("glm5", waves as u64),
        "GLM Spark endpoint count differs from the planner");
    Ok(())
}

#[cfg(test)]
mod draft_cli_tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn width_diagnostics_fail_closed_and_detect_live_route_changes() {
        let tap = || engine::WidthTap { layer: 3, x: vec![0; 4], routes: vec![0; 16], delta: vec![0; 4] };
        assert!(validate_width_taps(&[], &[3], 2, 1, 1).is_err());
        assert!(validate_width_taps(&[tap()], &[4], 2, 1, 1).is_err());
        assert!(validate_width_taps(&[tap()], &[3], 1, 1, 1).is_err());
        assert!(validate_width_taps(&[tap()], &[3], 2, 1, 1).is_ok());
        let base = engine::WidthTap { layer: 3, x: vec![0; 2], routes: vec![0; 8], delta: vec![0; 2] };
        assert!(tap_split(&[tap()], std::slice::from_ref(&base), 1, 1, 1).is_empty());
        for (offset, label) in [(4, "ids differ"), (12, "weights differ")] {
            let mut changed = tap();
            changed.routes[offset] = 1;
            assert!(tap_split(&[changed], std::slice::from_ref(&base), 1, 1, 1).contains(label));
        }
        let source = include_str!("engine.rs");
        let tap = source.split("fn tap_width(").nth(1).unwrap().split("fn finish_width_tap(").next().unwrap();
        assert!(tap.find("if self.width_taps.borrow().is_none()").unwrap() < tap.find("self.download(").unwrap());
        assert!(tap.contains("self.download(&w.route_ids,") && tap.contains("self.download(&w.route_weights,"));
        assert!(!tap.contains("router_host"));
    }

    #[derive(Parser)]
    struct Parse {
        #[command(flatten)]
        engine: EngineArgs,
    }


    fn planned(family: &str, backend: cuteafd_loader::plan::spark_executions::SparkBackend, decode: Option<u32>,
        unit: u64, waves: u64) -> cuteafd_core::SparkExecutions {
        let options = cuteafd_loader::plan::layout::LayoutOptions { prefill_unit_rows: Some(unit),
            prefill_waves: waves, ..Default::default() };
        cuteafd_loader::plan::layout::planned_spark_executions(family, backend, decode, &options).unwrap()
    }

    #[test]
    fn planner_and_serve_admit_identical_spark_owners() -> Result<()> {
        let k4 = SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest);
        for backend in [SparkBackend::Fp8, SparkBackend::Exlr8, k4] {
            for (waves, unit) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048)] {
                assert_eq!(spark_model(backend, waves, unit, false)?.executions,
                    planned("glm5", backend, None, u64::from(unit), waves as u64), "{backend:?} k{waves}");
            }
        }
        Ok(())
    }

    #[test]
    fn spark_contract_matches_decode_link_and_parallel_prefill_lanes() -> Result<()> {
        for (waves, unit) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048)] {
            for bf16 in [false, true] {
                let model = spark_model(SparkBackend::Fp8, waves, unit, bf16)?;
                model.validate(cuteafd_core::ExpertGeometry::GLM5)?;
                // The decode link and one prefill link carrying every wave in flight.
                assert_eq!(model.endpoints.len(), 2);
                assert_eq!(model.endpoints[0].capacity, engine::DECODE_ROWS as u32);
                assert_eq!(model.endpoints[1].capacity, unit);
                assert!(model.endpoints.iter().all(|endpoint| endpoint.ingress == if bf16 { cuteafd_core::SparkIngress::Bf16 }
                        else { cuteafd_core::SparkIngress::Fp8K32 }));
                // One prefill owner at the selected unit's covering export; k only buffers units.
                assert_eq!(model.executions.prefill, cuteafd_core::PrefillExecution { capacity: unit,
                    compiled_capacity: 4096, unit_rows: unit, buffered_units: waves as u8 });
                assert_eq!(model.executions.decode, [cuteafd_core::DecodeExecution {
                    capacity: engine::DECODE_ROWS as u32, compiled_capacity: 80 }]);
            }
        }
        for waves in 1..=4 {
            let native = spark_model(SparkBackend::Exlr8, waves, 4096, false)?;
            native.validate(cuteafd_core::ExpertGeometry::GLM5)?;
            assert_eq!(native.endpoints.len(), 2);
            assert_eq!(native.endpoints[0].capacity, engine::DECODE_ROWS as u32);
            assert_eq!(native.endpoints[1].capacity, 4096);
            assert!(native.endpoints.iter().all(|endpoint|
                endpoint.ingress == cuteafd_core::SparkIngress::Fp8K32));
            assert_eq!(native.executions.decode[0].compiled_capacity,
                cuteafd_core::expert_geometry::EXLR8_EXECUTION_ROWS as u32);
            assert_eq!(native.rings.max_frame_bytes,
                cuteafd_transport::expert::native_frame_budget(cuteafd_core::ExpertGeometry::GLM5, native.capacity())?);
        }
        assert!(spark_model(SparkBackend::Fp8, 0, 4096, false).is_err());
        assert!(validate_spark_endpoint_count(1, 1).is_err());
        assert!(validate_spark_endpoint_count(3, 3).is_err());
        assert!(validate_spark_endpoint_count(2, 1).is_ok());
        let source = include_str!("mod.rs");
        let links = source.rsplit("let mut transport = selected_peers.map(").next().unwrap()
            .split("let runtime = tokio::runtime::Builder").next().unwrap();
        let check = |body: &str| body.contains("engine::DECODE_ROWS as u32, config.clone(), row_bytes")
            && body.contains("warm(engine::DECODE_ROWS, true)")
            && body.contains("warm(args.prefill_rows, false)")
            && !body.contains("selected_peers.filter(");
        assert!(check(links));
        assert!(!check(&links.replace("engine::DECODE_ROWS as u32", "4096")));
        assert!(!check(&links.replace("match selected_peers {", "match selected_peers.filter(|_| prefill_lanes > 1) {")));
        Ok(())
    }

    #[test]
    fn layers_prefill_uses_shared_topology_and_pinned_waves() {
        use cuteafd_loader::serving_capacity::prefill_waves::{unit_rows, PrefillError, MAX_WAVES};
        for ranks in 0..=8 {
            let peers = (0..ranks).map(|rank| format!("worker{rank}:1")).collect::<Vec<_>>().join(",");
            for requested in std::iter::once(None).chain((1..=MAX_WAVES).map(Some)) {
                let wave_flag = requested.map(|waves| waves.to_string());
                let mut argv = vec!["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                    "--attention-placement", "layers", "--split-device", "1"];
                if ranks > 0 { argv.extend(["--peers", peers.as_str()]); }
                if let Some(value) = &wave_flag { argv.extend(["--prefill-waves", value.as_str()]); }
                let mut args = Parse::try_parse_from(argv).unwrap().engine;
                let waves = requested.unwrap_or_else(|| cuteafd_loader::serving_capacity::prefill_waves::depth(ranks, Default::default()));
                let result = resolve_prefill(&mut args);
                if ranks == 0 && waves > 1 {
                    assert!(matches!(result.unwrap_err().downcast_ref::<PrefillError>(),
                        Some(PrefillError::ExecutorCapacity { max_waves: 1, .. })));
                } else {
                    result.unwrap();
                    let rows = unit_rows(waves);
                    assert_eq!((args.prefill_rows as u64, u64::from(args.prefill_waves), args.prefill_span_rows as u64),
                        (rows, waves, rows * waves), "ranks={ranks}, requested={requested:?}");
                }
            }
        }
    }

    /// TJ 2026-10-28: an unpinned GLM 5 deployment with Sparks runs three
    /// units of the reserve; an explicit depth pins; a resolved deployment's
    /// shape passed back resolves to itself.
    #[test]
    fn unpinned_prefill_runs_three_units_and_pins_stay() {
        let peers = (0..6).map(|rank| format!("worker{rank}:1")).collect::<Vec<_>>().join(",");
        let resolve = |extra: &[&str]| {
            let mut argv = vec!["test", "--snapshot", "/checkpoint", "--native-lib", "/native", "--peers", peers.as_str()];
            argv.extend_from_slice(extra);
            let mut args = Parse::try_parse_from(argv).unwrap().engine;
            resolve_prefill(&mut args).unwrap();
            (args.prefill_rows, args.prefill_waves, args.prefill_span_rows)
        };
        assert_eq!(resolve(&[]), (2752, 3, 8256));
        assert_eq!(resolve(&["--prefill-waves", "2"]), (4096, 2, 8192));
        assert_eq!(resolve(&["--prefill-waves", "4"]), (2048, 4, 8192));
        assert_eq!(resolve(&["--attention-placement", "layers", "--split-device", "1"]), (2752, 3, 8256));
        assert_eq!(resolve(&["--prefill-rows", "2752", "--prefill-waves", "3", "--prefill-span-rows", "8256"]),
            (2752, 3, 8256));
    }

    #[test]
    fn packed_prefill_mode_cli_and_golden_defaults() {
        use crate::shared::prefill_packing::PackingMode;
        let parse = |extra: &[&str]| {
            let mut argv = vec!["test", "--snapshot", "/checkpoint", "--native-lib", "/native"];
            argv.extend_from_slice(extra);
            Parse::try_parse_from(argv).map(|p| p.engine.prefill_packing)
        };
        assert_eq!(parse(&[]).unwrap().unwrap_or_default(), PackingMode::Unrestricted);
        let source = include_str!("mod.rs");
        let golden = source.split(concat!("fn ", "golden(mut args:")).nth(1).unwrap()
            .split(concat!("fn ", "golden_run(")).next().unwrap();
        assert!(golden.contains("args.engine.prefill_packing.unwrap_or_default()"));
        assert!(!golden.contains("PackingMode::resolve"));
        assert_eq!(parse(&["--prefill-packing", "exact"]).unwrap(), Some(PackingMode::Exact));
        assert_eq!(parse(&["--prefill-packing", "unrestricted"]).unwrap(), Some(PackingMode::Unrestricted));
        assert!(parse(&["--prefill-packing", "off"]).is_err());
    }

    #[test]
    fn golden_rejects_invalid_unit_geometry_before_opening_checkpoint() {
        #[derive(Parser)]
        struct Golden { #[command(flatten)] args: GoldenArgs }
        for rows in ["32", "255", "257"] {
            let args = Golden::try_parse_from(["test", "--snapshot", "/missing-checkpoint",
                "--native-lib", "/missing-library", "--golden", "/missing-fixture",
                "--prefill-rows", rows, "--nll"]).unwrap().args;
            let error = golden(args).unwrap_err().to_string();
            assert!(error.contains("unsupported prefill shape"), "{error}");
        }
    }

    #[test]
    fn native_prefill_activation_cli_defaults_to_a16() {
        let parse = |extra: &[&str]| {
            let mut argv = vec!["test", "--snapshot", "/native", "--native-lib", "/library"];
            argv.extend_from_slice(extra);
            Parse::try_parse_from(argv).map(|p| p.engine.exlr8_prefill_activations)
        };
        assert_eq!(parse(&[]).unwrap(), Exlr8PrefillActivations::A16);
        assert_eq!(parse(&["--exlr8-prefill-activations", "a16"]).unwrap(), Exlr8PrefillActivations::A16);
        assert_eq!(parse(&["--exlr8-prefill-activations", "a8"]).unwrap(), Exlr8PrefillActivations::A8);
        assert!(parse(&["--exlr8-prefill-activations", "fp8"]).is_err());

        #[derive(Parser)]
        struct ExpertDaemon { #[command(flatten)] args: crate::cli::NativeExpertDaemonArgs }
        let parse_expertd = |extra: &[&str]| {
            let mut argv = vec!["test", "--snapshot", "/native", "--native-lib", "/library",
                "--rank", "0", "--device-budget-bytes", "1"];
            argv.extend_from_slice(extra);
            ExpertDaemon::try_parse_from(argv).map(|p| p.args.exlr8_prefill_activations)
        };
        assert_eq!(parse_expertd(&[]).unwrap(), Exlr8PrefillActivations::A16);
        assert_eq!(parse_expertd(&["--exlr8-prefill-activations", "a16"]).unwrap(), Exlr8PrefillActivations::A16);
        assert_eq!(parse_expertd(&["--exlr8-prefill-activations", "a8"]).unwrap(), Exlr8PrefillActivations::A8);
        assert!(parse_expertd(&["--exlr8-prefill-activations", "fp8"]).is_err());
    }

    /// `cuteafd plan --layout` and `serve-glm` admission charge the same native dense bytes for the same
    /// geometry: the planner's per-GPU "native dense scratch" item equals `native_dense_charge`.
    #[test]
    fn planner_charge_equals_admission_charge_for_the_same_geometry() {
        use cuteafd_loader::plan::exlr8::PrefillActivations;
        const GROUP: &str = "native dense scratch and prefill staging";
        let dir = tempfile::tempdir().unwrap();
        cuteafd_loader::plan::testing::exlr8::glm_snapshot(dir.path());
        let manifest = cuteafd_loader::plan::testing::glm5_native_programs();
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(dir.path()).unwrap();
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
        for (gpus, peers, split) in [(1usize, "a:1,b:1,c:1,d:1", None), (2, "a:1,b:1,c:1,d:1,e:1,f:1", Some("1"))] {
            for (rows, waves) in [("4096", "1"), ("4096", "2"), ("2048", "3"), ("1024", "4")] {
                let engine = |activations: &str| {
                    let mut argv = vec!["test", "--snapshot", "/native", "--native-lib", "/library", "--peers", peers,
                        "--prefill-rows", rows, "--prefill-waves", waves, "--exlr8-prefill-activations", activations];
                    if let Some(device) = split { argv.extend(["--split-device", device]); }
                    Parse::try_parse_from(argv).unwrap().engine
                };
                let mut admitted = Vec::new();
                for activations in ["a16", "a8"] {
                    let engine = engine(activations);
                    let charge = native_dense_charge(&cfg, gpus, &engine);
                    let layout = cuteafd_loader::plan::layout::LayoutOptions {
                        rtx_bytes: vec![96 << 30; gpus], prefill_rows: engine.prefill_rows as u64,
                        prefill_lanes: u64::from(engine.prefill_waves), prefill_waves: u64::from(engine.prefill_waves),
                        exlr8_prefill_activations: activations.parse::<PrefillActivations>().unwrap(), context_tokens: 65536,
                        onboard: Some(cuteafd_loader::placement::Onboard::Layers(0)), glm5_drafter_disabled: true,
                        host_memory: Some((128 << 30, 128 << 30)), host_cache_bytes: Some(0), ..Default::default() };
                    let (memory, _) = cuteafd_loader::plan::layout::glm5::native_layout(&checkpoint, &layout,
                        &vec![256; gpus], 256, peers.split(',').count(), &manifest, None).unwrap();
                    assert_eq!(memory.devices.len(), gpus);
                    for device in &memory.devices {
                        let planned: u64 = device.items.iter().filter(|item| item.group == GROUP).map(|item| item.bytes).sum();
                        assert_eq!(planned, charge, "{gpus} RTX/{rows} rows/{waves} waves/{activations}");
                    }
                    admitted.push(charge);
                }
                // Every slot shares one prefill scratch and its records (#762).
                let (heads, inter) = (cfg.heads / gpus, cfg.dense_intermediate / gpus);
                let records = cuteafd_loader::plan::exlr8::dense::a8_scratch_bytes(&cfg, heads, inter, rows.parse().unwrap());
                assert_eq!(admitted[1] - admitted[0], records as u64, "{gpus} RTX/{rows}/{waves}");
            }
        }
    }

    #[test]
    fn native_expert_selection_reaches_golden() {
        #[derive(Parser)]
        struct Golden { #[command(flatten)] args: GoldenArgs }
        let args = Golden::try_parse_from(["test", "--snapshot", "/native", "--native-lib", "/library",
            "--golden", "/golden", "--exlr8-manifest", "up1"])
            .unwrap().args.engine;
        assert_eq!(args.exlr8_manifest.as_deref(), Some("up1"));
    }

    #[test]
    fn native_full_activation_is_narrow_and_preserves_component_goldens() {
        let args = Parse::try_parse_from(["test", "--snapshot", "/native", "--native-lib", "/library",
            "--exlr8-manifest", "up1", "--split-device", "1",
            "--peers", "a:1,b:1,c:1,d:1", "--attention-placement", "context", "--rtx-expert-layers", "0"])
            .unwrap().engine;
        for manifest in ["up1", "bit325"] {
            let mut candidate = args.clone();
            candidate.exlr8_manifest = Some(manifest.into());
            assert!(native_activation(&candidate, 3, 78, true).is_ok());
            candidate.attention_placement = "auto".into();
            assert!(native_activation(&candidate, 3, 78, false).is_ok());
        }
        for mode in ["heads", "auto"] {
            let mut one = args.clone();
            one.split_device = None;
            one.attention_placement = mode.into();
            assert!(native_activation(&one, 3, 78, true).is_ok());
            assert!(native_activation(&one, 3, 78, false).is_ok());
        }
        for ranks in 1..=8 {
            let mut candidate = args.clone();
            candidate.peers = Some((0..ranks).map(|rank| format!("worker{rank}:1")).collect::<Vec<_>>().join(","));
            assert!(native_activation(&candidate, 3, 78, true).is_ok(), "TP{ranks}");
        }
        let mut drafting = args.clone();
        drafting.draft = Some("/draft".into());
        assert!(native_activation(&drafting, 3, 78, true).is_ok());
        let mut onboard = args.clone();
        onboard.rtx_expert_layers = cuteafd_loader::placement::Onboard::Layers(1);
        assert!(native_activation(&onboard, 3, 78, true).is_ok());
        for bad in 0..8 {
            let mut candidate = args.clone();
            match bad {
                0 => candidate.exlr8_manifest = Some("home".into()),
                1 => candidate.split_device = None,
                2 => candidate.split_device = Some(0),
                3 => candidate.peers = Some("a:1,b:1,c:1,d:1,e:1,f:1,g:1,h:1,i:1".into()),
                4 => candidate.layers = Some(4),
                5 => candidate.attention_placement = "heads".into(),
                6 => candidate.exlr8_manifest = None,
                _ => candidate.skip_routed_experts = true,
            }
            assert!(native_activation(&candidate, 3, 78, true).is_err(), "case {bad}");
        }
        let mut component = args;
        component.split_device = None;
        component.layers = Some(4);
        assert!(native_activation(&component, 3, 78, false).is_ok());
        assert!(native_activation(&component, 3, 78, true).is_err());

    }

    #[test]
    fn drafter_defaults_are_checkpoint_bf16_with_independent_capacities() {
        let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native"])
            .unwrap().engine;
        assert_eq!(parsed.draft_fp8, None);
        assert_eq!(parsed.draft_sequences, 16);
        assert_eq!(parsed.draft_context_slots.unwrap_or(20.max(parsed.draft_sequences)), 20);
    }

    #[test]
    fn attention_plan_override_is_permanent_and_validated_against_numerical_domain() {
        let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--attention-plan-rows", "8"]).unwrap().engine;
        assert_eq!(parsed.attention_plan_rows, Some(8));
        for value in ["0", "65"] {
            let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--attention-plan-rows", value]).unwrap().engine;
            assert!(cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
                engine::DECODE_ROWS, None, parsed.attention_plan_rows).is_err());
        }
    }

    #[test]
    fn explicit_fp8_option_and_context_batch_limits_are_forwarded() {
        for (value, expected) in [("true", true), ("false", false)] {
            let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--draft-fp8", value, "--draft-context-slots", "20", "--draft-sequences", "16"])
                .unwrap().engine;
            assert_eq!(parsed.draft_fp8, Some(expected));
            assert_eq!((parsed.draft_context_slots, parsed.draft_sequences), (Some(20), 16));
        }
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--draft-fp8", "auto"]).is_err());
    }
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/families/glm5/golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Prefill only the first N tokens, then feed the rest through decode
    /// steps of --step-rows rows (teacher-forced), comparing their rows.
    #[arg(long)]
    pub prefill: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub step_rows: usize,
    /// Compare only logits during decode (decode then runs its captured graphs).
    #[arg(long)]
    pub no_layer_compare: bool,
    /// With --draft: after the prefill, decode N greedy tokens one row per
    /// step, drafting before each, and report how many drafts the target
    /// reproduced (without it, drafts are scored against tokens.bin).
    #[arg(long)]
    pub generate: Option<usize>,
    /// With --draft: run only the drafter on the golden taps and compare with
    /// python/reference/families/glm5/dflash2/reference.py's output directory.
    #[arg(long)]
    pub draft_oracle: Option<PathBuf>,
    /// With --draft: replay the drafter alone on the golden taps, drafting
    /// after every token from this position on, BF16 and FP8 (acceptance
    /// against the text and the golden greedy picks, draft time).
    #[arg(long)]
    pub draft_replay: Option<usize>,
    /// Teacher-force this many copies of the sequence together (prefilled to
    /// staggered lengths), --step-rows rows each per step, and compare every
    /// copy's logits with the golden logits.
    #[arg(long)]
    pub sequences: Option<usize>,
    /// Compare packed short prompts with standalone passes, including every live cache byte.
    #[arg(long)]
    pub packed_check: Option<usize>,
    /// Score every prefill row's logits against the golden (mean NLL, top-1,
    /// KL); the prefill runs without layer downloads (Spark lanes).
    #[arg(long)]
    pub nll: bool,
    /// With --nll: also write tokens.bin and every row's logits.bin (F32) to
    /// this directory, a golden for A/B runs between builds or numerics (e.g.
    /// W8A8 against --prefill-w8a16 with --skip-routed-experts).
    #[arg(long, hide = true)]
    pub save_logits: Option<PathBuf>,
    /// Time this many prefills of --bench-prefill-tokens tokens (the golden
    /// prompt repeated, in chunks of the prefill capacity) on fresh sequences.
    #[arg(long, default_value_t = 0)]
    pub bench_prefill: usize,
    #[arg(long, default_value_t = 4096)]
    pub bench_prefill_tokens: usize,
    /// Time teacher-forced verify steps of 1, 2, 4, ... up to N rows of one
    /// sequence after --prefill tokens (default 512) of the golden prompt
    /// (median of 7 after 2 warm-ups; the sequence rewinds between steps),
    /// and with --draft the drafter step for 1, 2, 4, ... sequences.
    #[arg(long)]
    pub bench_verify: Option<usize>,
    /// With --bench-verify: time only these row counts (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub bench_rows: Vec<usize>,
    /// With --bench-verify and --draft: time only these sequence counts.
    #[arg(long, value_delimiter = ',')]
    pub bench_sequences: Vec<usize>,
    /// With --bench-verify: write each row count's last logits (F32) and,
    /// with --draft, each sequence count's drafts to this directory (A/B
    /// numerics checks between builds).
    #[arg(long)]
    pub bench_dump: Option<PathBuf>,
    /// Row-width invariance after --prefill tokens (default 512): verify the same
    /// teacher-forced rows at each of these widths (comma-separated; 1 runs first and
    /// again last as a determinism check), rewinding between widths, and compare every
    /// shared row's per-layer residual and logits bitwise against width 1's. Reports the
    /// first layer that differs for each (width, row). Layer downloads run the steps
    /// uncaptured, on the same kernels.
    #[arg(long, value_delimiter = ',')]
    pub width_invariance: Vec<usize>,
    /// With --bench-verify: prefill this many tokens (the golden prompt
    /// repeated) instead of --prefill, for long-context step costs.
    #[arg(long)]
    pub bench_context: Option<usize>,
    /// Prefix-cache restore check at token P: prefill the first P tokens (of --prefill, default
    /// all), fork their pages into a second sequence (shared full pages, the copied tail page),
    /// continue both (prefill in --prefill-chunk rows, then --resume-decode greedy steps) and
    /// compare every layer's rows, the logits and the paged rows byte for byte.
    #[arg(long)]
    pub resume_at: Option<usize>,
    /// Prefill chunk rows of --resume-at (default the engine's prefill rows).
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    #[arg(long, default_value_t = 4)]
    pub resume_decode: usize,
    /// With --resume-at: prefill the second sequence cold on its own pages instead of
    /// restoring it (the floor: what the kernels themselves vary).
    #[arg(long, hide = true)]
    pub resume_cold: bool,
    /// --resume-at attempts on fresh sequences (all must be byte-identical).
    #[arg(long, default_value_t = 1)]
    pub resume_repeat: usize,
    /// Token I/O gate after the golden prompt (--prefill tokens of it), then
    /// stop: the resident embedding table against the shard, device against
    /// host greedy selection over this many decode steps, device against host sampling.
    #[arg(long)]
    pub token_check: Option<usize>,
}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub snapshot: PathBuf,
    pub catalog: cuteafd_loader::OfficialV41Catalog,
    /// An EXLR8 checkpoint: its own dense records run natively.
    pub native: bool,
    pub cfg: GlmDsaConfig,
    pub library: std::sync::Arc<NativeLibrary>,
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    let mut catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
    cuteafd_core::set_expert_geometry(catalog.routed_experts().geometry()?)
        .map_err(|g| anyhow::anyhow!("geometry already {g:?}"))?;
    let cfg = GlmDsaConfig::read(&args.snapshot)?;
    let native = catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8;
    if native {
        ensure!(args.exlr8_manifest.is_some(), "native EXLR8 requires --exlr8-manifest");
        native_activation(args, cfg.first_moe_layer, cfg.layers, false)?;
    } else {
        ensure!(args.exlr8_manifest.is_none()
            && args.exlr8_prefill_activations == Exlr8PrefillActivations::A16,
            "EXLR8 options require a native EXLR8 checkpoint");
    }
    if let Some(snapshot) = &args.draft {
        // A native EXLR8 target lends its row-FP8 head through a launcher.
        if !native { dflash::check_snapshot_target_head(&args.snapshot, cfg.hidden, cfg.vocab_size, false)?; }
        dflash::check_checkpoint(snapshot, args.draft_fp8, args.draft_context_slots, args.draft_sequences)?;
    }
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&args.native_lib) }?);
    library.cuda_set_device(args.device)?;
    Ok(Opened { snapshot: args.snapshot.clone(), catalog, native, cfg, library })
}

impl Opened {
    /// Builds the engine (and the Spark transport when peers are given) and
    /// hands them to `body`.
    pub fn with_engine<T>(&self, args: &EngineArgs,
        body: impl FnOnce(&engine::GlmEngine<'_>, Option<&mut SparkLink<'_>>, &tokio::runtime::Runtime) -> Result<T>)
        -> Result<T> {
        let automatic_context = args.max_context == 0;
        let mut context_args = args.clone();
        resolve_prefill(&mut context_args)?;
        context_args.max_context = crate::shared::context::checkpoint_context(
            &args.snapshot, &args.manifest, "glm5", args.max_context)?;
        let args = &context_args;
        let native = self.native;
        let dense_catalog = &self.catalog;
        ensure!(!args.skip_routed_experts || args.peers.is_none(),
            "--skip-routed-experts is a coordinator-only diagnostic; do not pass --peers");
        let programs = self.library.programs()?.with_manifest(&args.manifest)?;
        programs.capacities().require_context("glm5", args.max_context)?;
        let provisional_drafter = args.draft.as_deref().map(dflash::DflashConfig::read).transpose()?;
        if let Some(cfg) = &provisional_drafter {
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "the DFlash2 drafter does not fit this target");
        }
        let attention_plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
            engine::DECODE_ROWS, provisional_drafter.as_ref().map(|cfg| cfg.drafts()), args.attention_plan_rows)?;
        let legacy_manifest = args.manifest.with_file_name("DSV4_PROGRAMS.json");
        let manifest_path = if !args.manifest.exists()
            && args.manifest.file_name() == Some("PROGRAMS.json".as_ref()) && legacy_manifest.exists() {
            &legacy_manifest
        } else { &args.manifest };
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest_path)?)?;
        // GLM 5.3's own programs (`glm_*`, and the head split's `glm2_*`), sampled as the runtime
        // inventory; the image's other families are never launched here.
        let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: "glm",
            split_family: args.split_device.map(|_| "glm2") };
        let programs = programs.with_attention_plan(attention_plan, |name| selected.contains(name))?;
        let stream = self.library.cuda_stream_create()?;
        // The head split's second GPU and its stream (load kernels, then the engine's).
        // A head split needs its share's programs (`glm2`) in this build.
        let split_device = match args.split_device {
            Some(device) if programs.spec("glm2_o_m64").is_ok() => Some(device),
            Some(device) => {
                tracing::info!(device, "no head-split programs (glm2) in this build; serving from --device alone");
                None
            }
            None => None,
        };
        let split_device = crate::shared::peer_split::probed_device(&self.library, args.device, split_device)?;
        ensure!(!native || args.split_device.is_none() || split_device.is_some(),
            "EXLR8 two-coordinator execution cannot fall back to one GPU");
        let peer_stream = match split_device {
            Some(device) => {
                ensure!(device != args.device, "--split-device must differ from --device");
                self.library.cuda_enable_peer(device)?;
                self.library.cuda_set_device(device)?;
                let stream = self.library.cuda_enable_peer(args.device).and_then(|()| self.library.cuda_stream_create());
                self.library.cuda_set_device(args.device)?;
                Some((device, stream?))
            }
            None => None,
        };
        let devices: Vec<_> = std::iter::once(args.device).chain(split_device).collect();
        use cuteafd_loader::families::glm5::DensePrograms;
        let dense_programs = if native { DensePrograms::Exlr8 } else { DensePrograms::Fp8 };
        // Preserve EXLR8's eager split stages on one GPU. Their ABI still
        // comes from this checkpoint, never from unrelated K6 exports.
        let inventory = crate::shared::inventory::RuntimeInventory::measure(&self.library, &devices, crate::shared::inventory::InventoryPhase::PreWeight, |_| {
            let (loaded, skipped) = programs.load_matching(|name| selected.contains(name)
                || (native && (cuteafd_core::coordinator_programs::program_namespace(name) == Some("exlr8")
                    || dense_programs.stage(name, true))))?;
            tracing::info!(loaded, skipped, "GLM programs loaded");
            Ok(())
        })?;
        let draft_file = args.draft.as_deref().map(dflash::prefetch);
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let requested = cuteafd_loader::placement::attention::parse(&args.attention_placement)
            .map_err(anyhow::Error::msg)?;
        ensure!(requested != Some(cuteafd_loader::placement::AttentionPlacement::Layers) || devices.len() == 2,
            "GLM layers attention requires two coordinator GPUs");
        let owners = (requested == Some(cuteafd_loader::placement::AttentionPlacement::Layers))
            .then(|| cuteafd_loader::placement::families::glm5::attention_owners(&self.cfg, layers)).transpose()?;
        use cuteafd_loader::placement::{families::glm5 as glm, Baseline};
        let checkpoint = (!native).then(|| cuteafd_loader::plan::Checkpoint::coordinator(dense_catalog.snapshot(), false, false))
            .transpose()?;
        // The EXLR8 checkpoint's own dense records, per coordinator rank.
        let native_records = if native {
            ensure!(owners.is_none(), "EXLR8 native dense runs head or context attention, not whole-layer owners");
            let records = cuteafd_loader::plan::exlr8::dense::DenseRecord::inventory(&self.snapshot)?;
            Some(records)
        } else { None };
        let native_plans = if native {
            (0..devices.len()).map(|rank| engine::exlr8::NativePlan::new(&self.snapshot,
                args.exlr8_manifest.as_deref().context("EXLR8 manifest")?,
                self.cfg.first_moe_layer, layers, &programs, devices.len(), rank))
                .collect::<Result<Vec<_>>>()?
        } else { Vec::new() };
        for plan in &native_plans { plan.admit_host()?; }
        let native_workspace = native_plans.iter().map(|plan| plan.workspace as u64).max().unwrap_or(0);
        let host_embedding = args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Host;
        ensure!(!native || !host_embedding, "the EXLR8 row-FP8 embedding is device-resident (its owner holds the head too)");
        let groups = match (&checkpoint, &native_records) {
            (Some(checkpoint), _) => {
                if host_embedding { checkpoint.require_untied_embedding("model.embed_tokens.weight")?; }
                cuteafd_loader::plan::layout::glm5::resident_owned(checkpoint, devices.len(), layers, owners.as_deref())?
            }
            (None, Some(records)) => (0..devices.len()).map(|rank| cuteafd_loader::plan::exlr8::dense::resident_native(
                records, devices.len(), rank, layers, self.cfg.heads, self.cfg.qk_nope_head_dim, self.cfg.v_head_dim,
                self.cfg.kv_lora_rank).map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?,
            (None, None) => anyhow::bail!("GLM coordinator operands need a checkpoint"),
        };
        let mut reserved_weights = groups.into_iter().map(|groups|
                groups.into_iter().filter(|(group, _, _)| (!host_embedding || group != "embedding")
                    && (!native || group != "shared_expert"))
                    .try_fold(0u64, |sum, (_, _, bytes)| sum.checked_add(bytes)
                        .context("GLM coordinator weight reservation overflow")))
                .collect::<Result<Vec<_>>>()?;
        for (reserved, plan) in reserved_weights.iter_mut().zip(&native_plans) {
            *reserved = reserved.checked_add(plan.resident).context("EXLR8 coordinator reservation overflow")?;
        }
        let spark_bf16 = checkpoint.as_ref().is_some_and(|c| c.spark_input_is_bf16());
        // Native dense: decode plus every prefill lane's scratch and opt-in A8 records.
        let native_dense_charge: Vec<u64> = if native {
            vec![native_dense_charge(&self.cfg, devices.len(), args); devices.len()]
        } else { Vec::new() };
        let gpus = inventory.gpus.iter().zip(&reserved_weights).map(|(gpu, &bytes)| -> Result<_> {
            let free_bytes = gpu.free_bytes.checked_sub(bytes)
                .context("GLM coordinator header-derived weights exceed pre-weight free memory")?;
            Ok((gpu.total_bytes, Baseline::Measured { free_bytes }))
        }).collect::<Result<Vec<_>>>()?;
        inventory.record("glm5", &gpus.iter().map(|(_, b)| match b {
            Baseline::Measured { free_bytes } => *free_bytes, _ => unreachable!(),
        }).collect::<Vec<_>>());
        let scratch = |decode| -> Result<_> {
            let lookup = |name: &str| programs.spec(name).ok().map(|p| p.scratch.get("scratch").copied().unwrap_or(0));
            let mut scratch = glm::step_scratch(&self.cfg, devices.len() == 2, decode, lookup)?;
            if owners.is_some() { scratch.programs = scratch.programs.max(glm::step_scratch(&self.cfg, false, decode, lookup)?.programs); }
            Ok(scratch)
        };
        let drafter_bytes = args.draft.as_deref().map(|snapshot| glm::drafter_bytes(snapshot,
            args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences,
            self.library.sm_count()? as u64, args.draft_fp8 == Some(false), args.draft_kv_cache)).transpose()?.unwrap_or(0);
        let drafter_staging = args.draft.as_deref().map(|snapshot| glm::drafter_staging(snapshot,
            args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences,
            args.draft_fp8 == Some(false))).transpose()?.unwrap_or(0);
        let spark_ranks = args.peers.as_deref().map_or(0, |p| p.split(',').count());
        // Token-split attention: the build's programs (GPU0's table; both GPUs load the same).
        let names: Vec<&str> = programs.names().collect();
        let mut context = (devices.len() == 2).then(|| glm::context_programs(names.iter().copied(),
            |name| programs.spec(name).ok().map(|p| p.scratch.get("scratch").copied().unwrap_or(0)))).flatten();
        if let Some(context) = &mut context {
            let selected_scratch = devices.iter().map(|&device| -> Result<u64> {
                self.library.cuda_set_device(device)?;
                let sms = self.library.sm_count()?;
                [0, self.cfg.heads / 2].into_iter().map(|begin| {
                    let name = cuteafd_core::coordinator_programs::attention_partial_program(
                        &manifest, "glm", begin, engine::DECODE_ROWS, sms, &attention_plan)?;
                    Ok(programs.spec(&name)?.scratch.get("scratch").copied().unwrap_or(0))
                }).collect::<Result<Vec<_>>>().map(|sizes| sizes.into_iter().max().unwrap_or(0))
            }).collect::<Result<Vec<_>>>();
            self.library.cuda_set_device(args.device)?;
            context.partial_scratch = selected_scratch?.into_iter().max().unwrap_or(0);
        }
        // Every per-slot owner covers the waves in flight.
        let prefill_lanes = usize::from(args.prefill_waves);
        ensure!(owners.is_none() || prefill_lanes <= cuteafd_loader::placement::LAYER_PIPELINE_MAX_LANES as usize,
            "layer-owned prefill lacks capacity for {prefill_lanes} waves");
        let core = checkpoint.as_ref().map(|checkpoint| -> Result<_> {
            let core = cuteafd_loader::families::glm5::core::CoreInventory::from_checkpoint(&self.cfg, checkpoint)?;
            if core.projections.is_empty() { return Ok(None); }
            let draft_rows = args.draft.as_deref().map(|snapshot| -> Result<_> {
                let config = cuteafd_loader::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
                let (_, block) = cuteafd_loader::families::glm5::draft_representation::draft_geometry(&config)?;
                args.draft_sequences.checked_mul(usize::try_from(block)?).context("GLM K6 draft head rows")
            }).transpose()?;
            Ok(Some(cuteafd_loader::families::glm5::core::CoreCharges::from_package(&self.cfg,
                &core, layers, devices.len(), owners.as_deref(), args.prefill_rows, prefill_lanes,
                draft_rows, &manifest)?))
        }).transpose()?.flatten();
        let native_dense_charge = if let Some(core) = &core { core.bytes(self.cfg.hidden)? }
            else { native_dense_charge };
        let code_set = core.as_ref().map(|core| core.program_set(&manifest, devices.len() == 2))
            .unwrap_or_else(|| dense_programs.program_set(&manifest, true));
        let pending_code = devices.iter().enumerate().map(|(rank, &device)|
            crate::shared::inventory::pending_code(&self.library, device, rank, devices.len() == 2,
                if core.is_some() { "glm-k6" } else { "glm" }, "*", &code_set)).collect::<Result<Vec<_>>>()?;
        let exchange_f32 = glm::local_exchange_f32()?;
        let local = glm::select_local_inventory(&self.catalog, manifest_path, layers, devices.len(),
            args.prefill_rows as u64, prefill_lanes as u64, exchange_f32, args.rtx_expert_layers,
            args.skip_routed_experts, args.exlr8_manifest.as_deref()
                .map(|selected| (selected, args.exlr8_prefill_activations)))?;
        if let Some(local) = &local {
            let max_rows = args.prefill_rows.max(engine::DECODE_ROWS);
            match local.format {
                glm::GlmLocalFormat::Exl3 => ensure!(crate::shared::experts::rtx::exl3::Exl3Tp2::workspace_bytes_for(
                    &local.package, self.cfg.hidden, max_rows)? as u64 == local.backend_workspace,
                    "GLM EXL3 planner/runtime backend workspace mismatch"),
                glm::GlmLocalFormat::Exlr8 => ensure!(
                    crate::shared::experts::rtx::exlr8::Exlr8Tp2Weights::workspace_bytes(
                        &programs, max_rows, args.exlr8_prefill_activations)? as u64 == local.backend_workspace,
                    "GLM EXLR8 planner/runtime backend workspace mismatch"),
                glm::GlmLocalFormat::Nvfp4 { .. } => {
                    let tensors = self.catalog.fp8().context("NVFP4 tensors")?;
                    for rank in 0..2 {
                        let plan = crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::plan(tensors, &local.package, rank, max_rows)?;
                        ensure!(plan.workspace_bytes as u64 == local.backend_workspace,
                            "GLM NVFP4 planner/runtime backend workspace mismatch");
                    }
                }
            }
        }
        let tp2_workspace = local.as_ref().map_or([0; 2], |l| l.workspace());
        let serial_workspace = local.as_ref().map_or([0; 2], |l| l.workspace_for(&self.cfg,
            args.prefill_rows as u64, 1, exchange_f32));
        let inputs = glm::GlmInputs { cfg: &self.cfg, layers, gpus, headroom_bytes: 2 << 30,
            spark_ranks, native_experts: native, native_workspace, native_dense: native_dense_charge.clone(), native_head: core.as_ref().is_some_and(|c| c.ranks[0].target_head.is_some()), skip_routed_experts: args.skip_routed_experts,
            prefill_rows: args.prefill_rows as u64,
            prefill_lanes: prefill_lanes as u64,
            max_context: args.max_context as u64, automatic_context, scratch: Some([scratch(true)?, scratch(false)?]),
            drafter_bytes, drafter_staging, pending_code,
            experts: local.as_ref().map_or_else(Vec::new, |l| l.experts.clone()),
            expert_workspace: 0, tp2_workspace, requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
            onboard: if args.skip_routed_experts { cuteafd_loader::placement::Onboard::Layers(0) }
                else { args.rtx_expert_layers }, full_prefill_logits: args.full_prefill_logits, context };
        let (placement, working) = glm::solve_working_set(&inputs, serial_workspace, requested)?;
        ensure!(!native || devices.len() != 2
            || placement.attention_placement == cuteafd_loader::placement::AttentionPlacement::Context,
            "EXLR8 two-coordinator admission requires context attention programs");
        let prefill_lanes = usize::try_from(working.spark.prefill_lanes)?;
        let selected_peers = if working.spark.spark_ranks > 0 { args.peers.as_deref() } else { None };
        cuteafd_loader::placement::families::GLM5.check(&placement)?;
        tracing::info!(placement = %placement.summary(), "GLM admission");
        let tp2_range = placement.tp2.map(|p| p.first..p.first + p.layers);
        let max_rows = args.prefill_rows.max(engine::DECODE_ROWS);
        let tp2_devices = || -> Result<[crate::shared::memory::device::Device<'_>; 2]> {
            Ok([crate::shared::memory::device::Device { library: &self.library, id: args.device },
                crate::shared::memory::device::Device { library: &self.library, id: split_device.context("TP2 peer")? }])
        };
        let tp2_budgets = || -> Result<[usize; 2]> {
            let budgets = placement.tp2.context("TP2 budget")?.peak_bytes;
            Ok([usize::try_from(budgets[0])?, usize::try_from(budgets[1])?])
        };
        let loader = weights::GlmLoader { library: &self.library, catalog: dense_catalog, stream, device: args.device,
            peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
                .collect(), push: None };
        let mut coordinator_push = None;
        let mut native_push = None;
        let native_push_experts = || -> Result<_> {
            tp2_range.as_ref().map(|range| -> Result<_> {
                let workspace = crate::shared::experts::rtx::exlr8::Exlr8Tp2Weights::workspace_bytes(
                    &programs, max_rows, args.exlr8_prefill_activations)?;
                let budgets = tp2_budgets()?.map(|b| b.saturating_sub(workspace));
                Ok(crate::shared::experts::rtx::push::CoordinatorExperts::Exlr8Tp2 {
                    manifest: args.exlr8_manifest.clone().context("EXLR8 manifest")?,
                    layers: range.clone().collect(), budgets: budgets.to_vec() })
            }).transpose()
        };
        let admission = if let Some(peers) = args.peers.as_deref() {
            let peers = peers.split(',').map(str::parse)
                .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let admission = crate::shared::placement::SparkAdmission::plan_manifest(&self.snapshot,
                self.catalog.routed_experts().geometry()?, Some(&self.catalog), &working.spark,
                peers.len(), None, spark_model(SparkBackend::for_catalog(&self.catalog)?, prefill_lanes, u32::try_from(args.prefill_rows.max(engine::DECODE_ROWS))?, spark_bf16)?,
                args.exlr8_manifest.as_deref())?;
            // The coordinator GPUs' EXL3 TP2 halves join the Spark push group as
            // two more ranks; planning failure leaves them on the mount.
            // The coordinator GPUs' EXL3 TP2 halves and dense operands join the
            // Spark push group as two more ranks, planned from headers before
            // membership: a planning error offers no coordinator ranks (the
            // coordinator loads by mount, the Sparks still push).
            let ranks = match (&tp2_range, local.as_ref().map(|l| &l.format)) {
                (Some(range), Some(glm::GlmLocalFormat::Exl3)) if admission.coordinator_pushes() => (|| -> Result<_> {
                    let local = local.as_ref().context("TP2 local inventory")?;
                    let workspace = crate::shared::experts::rtx::exl3::Exl3Tp2::workspace_bytes_for(
                        &local.package, self.cfg.hidden, max_rows)?;
                    let budgets = tp2_budgets()?.map(|b| b.saturating_sub(workspace));
                    let devices = tp2_devices()?;
                    let sink = std::rc::Rc::new(landing::PushSink::default());
                    let planning = weights::GlmLoader { push: Some(sink.clone()), ..loader.clone() };
                    let model = {
                        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                        planning.model(&self.cfg, layers, owners.as_deref(), native)?
                    };
                    let (records, deferred) = sink.take();
                    let dense = push::plan_ranks(&self.snapshot, records.clone(), devices, peers.len())?;
                    let prepared = crate::shared::experts::rtx::push::CoordinatorPush::prepare(&self.snapshot,
                        Some(&self.catalog), 2, Some(crate::shared::experts::rtx::push::CoordinatorExperts::Tp2 { layers: range.clone().collect(), budgets: budgets.to_vec() }))?;
                    Ok((prepared, (model, records, deferred, dense)))
                })().map_err(|error| tracing::info!(%error, "coordinator TP2 push unavailable; mount")).ok(),
                _ => None,
            };
            let ranks = ranks.map(|((push, streams, ranks), planned)| {
                coordinator_push = Some((push, peers.len(), planned));
                (streams, ranks)
            });
            // Native dense operands and onboarded EXLR8 halves share each GPU's
            // coordinator rank. Onboarded experts have no post-push mount loader.
            let ranks = match (ranks, &native_records) {
                (None, Some(_)) if admission.coordinator_pushes() => {
                    let experts = native_push_experts()?;
                    let (push, streams, ranks) = crate::shared::experts::rtx::push::CoordinatorPush::prepare(
                        &self.snapshot, Some(&self.catalog), devices.len(), experts)?;
                    native_push = Some((push, peers.len()));
                    Some((streams, ranks))
                }
                (ranks, _) => ranks,
            };
            Some(admission.start_with(&peers, ranks)?)
        } else { None };
        // All selected routed layers may be onboarded in a partial model.
        // Idle Spark peers acknowledge identity while these destinations push alone.
        if native_records.is_some() && native_push.is_none() {
            let push = crate::shared::experts::rtx::push::CoordinatorPush::prepare_alone(
                &self.snapshot, Some(&self.catalog), devices.len(), native_push_experts()?)?;
            native_push = Some((push, 0));
        }
        let started = Instant::now();
        // The admitted TP2 expert halves are charged before the serving pool;
        // their bytes land inside the Spark admission window (pushed, or through
        // the mount). Execution (modules, workspaces, routes) binds after the engine exists.
        let tp2_weights = |range: std::ops::Range<usize>, pushed| -> Result<crate::shared::experts::rtx::Tp2Weights<'_>> {
            let pushed: Option<Vec<crate::shared::experts::rtx::push::PushedExperts<'_>>> = pushed;
            let local = local.as_ref().context("TP2 placement without local package inventory")?;
            let (devices, budgets) = (tp2_devices()?, tp2_budgets()?);
            use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy::CanonicalDecode};
            let geometry = cuteafd_core::expert_geometry();
            Ok(match local.format {
                glm::GlmLocalFormat::Exl3 => crate::shared::experts::rtx::Tp2Weights::Exl3(match pushed {
                    Some(pushed) => Exl3Tp2::pushed_pair_weights(devices, &self.catalog, geometry,
                        [&local.package; 2], range.collect(), max_rows, budgets, CanonicalDecode, pushed.into_iter().map(|r| r.exl3()).collect::<Result<Vec<_>>>()?
                            .try_into().map_err(|_| anyhow::anyhow!("two EXL3 coordinator GPUs"))?)?,
                    None => Exl3Tp2::load_pair_weights(devices, &self.catalog, geometry,
                        [&local.package; 2], range.collect(), max_rows, budgets, CanonicalDecode)?,
                }),
                glm::GlmLocalFormat::Exlr8 => crate::shared::experts::rtx::Tp2Weights::Exlr8(
                    crate::shared::experts::rtx::exlr8::Exlr8Tp2Weights::pushed_pair(devices,
                        &programs, range.collect(), max_rows, budgets, args.exlr8_prefill_activations,
                        pushed.context("onboarded EXLR8 experts require a committed push group")?
                            .into_iter().map(|r| r.exlr8()).collect::<Result<Vec<_>>>()?
                            .try_into().map_err(|_| anyhow::anyhow!("two EXLR8 coordinator GPUs"))?)?),
                glm::GlmLocalFormat::Nvfp4 { .. } => crate::shared::experts::rtx::Tp2Weights::Fp8(
                    crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::load_pair(devices,
                        self.catalog.fp8().context("NVFP4 tensors")?, &local.package,
                        range.collect(), max_rows, budgets)?),
            })
        };
        let (embedding, ((model, mut shares), tp2)) = crate::shared::placement::load_while_admitting(admission, || {
            if let Some(records) = &native_records {
                let loaded = self.load_native(args, records, layers, &programs, &manifest, &loader, &devices, stream,
                    peer_stream, native_push.take())?;
                let tp2 = tp2_range.clone().map(|range| tp2_weights(range, loaded.2)).transpose()?;
                return Ok((loaded.0, (loaded.1, tp2)));
            }
            let (embedding, (model, pushed)) = crate::shared::token_io::TokenEmbedding::load(&self.library,
                self.embed_source()?, args.token_io.embed_placement, || {
                let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                let Some((push, sparks, (model, records, deferred, dense))) = coordinator_push.take() else {
                    return Ok((loader.model(&self.cfg, layers, owners.as_deref(), native)?, None));
                };
                // The planned dense operands join the same ranks as the TP2 halves:
                // receive both, then run the transforms of landed bytes.
                let devices = tp2_devices()?;
                let (pushed, ()) = push.receive_with(devices.to_vec(), Some(&self.catalog), sparks, Some(Ok(dense)), || ());
                let pushed = pushed?;
                let rank_stream = |device: i32| if device == args.device { stream }
                    else { peer_stream.map_or(stream, |(_, s)| s) };
                if pushed.is_none() {
                    crate::shared::experts::rtx::dense::fill_by_mount(&self.library, &records, &[], rank_stream)?;
                }
                for job in &deferred {
                    crate::shared::memory::device::Device { library: &self.library, id: job.device }
                        .run(|| job.run(&self.library, rank_stream(job.device)))?;
                }
                tracing::info!(pushed = pushed.is_some(), operands = records.len(),
                    "GLM coordinator dense operands landed");
                Ok((model, pushed))
            })?;
            let tp2 = tp2_range.clone().map(|range| tp2_weights(range, pushed)).transpose()?;
            Ok((embedding, (model, tp2)))
        })?;
        let native_owners = native_plans.into_iter().zip(&devices).map(|(plan, &device)| {
            self.library.cuda_set_device(device)?;
            engine::exlr8::NativeExperts::load(&self.library, plan)
        }).collect::<Result<Vec<_>>>()?;
        self.library.cuda_set_device(args.device)?;
        let bytes = model.layers.iter().map(|layer| layer.bytes()).sum::<usize>()
            + model.norm.buffer.bytes + model.head.bytes();
        let embedding_bytes = if host_embedding || native { 0 } else { embedding.vocab() * embedding.hidden() * 2 };
        let actual_weights = std::iter::once((bytes + embedding_bytes) as u64)
            .chain(shares.iter().map(|rank| rank.iter().map(|layer| layer.bytes() as u64).sum()))
            .enumerate().map(|(rank, bytes)| bytes
                + native_owners.get(rank).map_or(0, |owner| owner.resident_bytes() as u64))
            .collect::<Vec<_>>();
        ensure!(actual_weights == reserved_weights,
            "GLM loaded coordinator owners {actual_weights:?} differ from header reservation {reserved_weights:?}");
        tracing::info!(layers, elapsed_ms = started.elapsed().as_millis() as u64,
            resident_bytes = ?actual_weights,
            "GLM coordinator weights validated against pre-weight reservation after Spark admission (elapsed includes wait)");
        let context_kv = placement.attention_placement == cuteafd_loader::placement::AttentionPlacement::Context;
        ensure!(!context_kv || peer_stream.is_some(), "context attention needs the head split's second GPU");
        cuteafd_bench::context::set_resolved("attention-placement", &placement.attention_summary());
        cuteafd_bench::context::set_rtx_expert_layers(placement.onboard_layers);
        cuteafd_bench::context::set_resolved("pool-tokens", &placement.pool_tokens.to_string());
        let pool_tokens = usize::try_from(placement.pool_tokens)?;
        let max_context = crate::shared::context::pool_context("glm5", args.max_context, automatic_context, pool_tokens, 256)?;
        let pages = pool_tokens.div_ceil(engine::PAGE_ROWS);
        // Pack before allocating the serving pool and expert arenas: the load-phase
        // admission peak is above the same resident target-weight baseline.
        let drafter = if let Some(snapshot) = &args.draft {
            let started = Instant::now();
            let cfg = dflash::DflashConfig::read(snapshot)?;
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "the DFlash2 drafter does not fit this target");
            let mask = embedding.host_rows(&[cfg.mask_token])?;
            tracing::info!(mask_source = "target mask_token row", mask_token = cfg.mask_token, "drafter mask source");
            let file = draft_file.context("drafter prefetch")?.join()
                .map_err(|_| anyhow::anyhow!("drafter prefetch panicked"))??;
            let representation = cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation
                ::from_fp8_option(args.draft_fp8);
            let drafter = dflash::DflashDrafter::load(&self.library, snapshot, file, stream,
                args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences,
                mask, false, representation, args.fp8_scales, crate::shared::fp8_linear::Fp8Rows::W8a16, args.draft_kv_cache)?;
            // Admission includes the shared drafter's maximum serving workspace.
            drafter.prepare_workspace()?;
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64,
                draft_head = "tensor", "DFlash2 drafter resident");
            Some(drafter)
        } else { None };
        let depth = crate::shared::prefill_depth::PrefillDepth::new(prefill_lanes, args.prefill_rows)?;
        let mut engine = engine::GlmEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            max_context, args.prefill_rows, pages, embedding, depth, context.filter(|_| context_kv))?;
        engine.max_pool_tokens_no_onboard = Some(placement.max_pool_tokens_no_onboard);
        engine.exlr8_prefill_activations = args.exlr8_prefill_activations;
        engine.native_experts = native_owners;
        engine.spark_bf16 = spark_bf16;
        engine.full_prefill_logits = args.full_prefill_logits;
        engine.prefill_span_rows = args.prefill_span_rows;
        engine.attention_plan = attention_plan;
        if tp2_range.is_some() && exchange_f32 {
            engine.set_tp2_exchange(crate::shared::experts::rtx::ExchangeDtype::F32);
        }
        if let Some((device, stream)) = peer_stream {
            let mut peer_layers = shares.pop().context("head-split shares")?;
            if let Some(range) = &tp2_range { loader.peer_routers(&mut peer_layers, range.clone())?; }
            engine.attach_peer(device, stream, peer_layers)?;
        }
        if owners.is_some() { engine.bind_layer_execution(&placement)?; }
        engine.bind_core(core)?;
        if native {
            // Bind each GPU before capture; all prefill scratch is lane-local.
            for rank in 0..engine.ranks() {
                let bound = engine.on(rank, || engine::native_dense::NativePrograms::bind(&programs, &manifest))?;
                engine.native_dense.push(Some(bound));
            }
        }
        if let Some(tp2) = tp2 {
            let local = local.as_ref().context("TP2 placement without local package inventory")?;
            let ranks = tp2.execution()?;
            for rank in &ranks { ensure!(rank.workspace_bytes() as u64 == local.backend_workspace,
                "GLM loaded TP2 workspace differs from admission"); }
            engine.install_tp2(ranks)?;
        }
        if context_kv {
            // The solver charged staging over the compiled extent it was given.
            engine.attach_context(args.max_context, &manifest)?;
        }
        engine.prefill_w8a8 = !args.prefill_w8a16;
        if args.skip_routed_experts {
            engine.skip_routed_experts()?;
        }
        engine.drafter = drafter;
        engine.bind_draft_core_head()?;
        engine.attention_plan = attention_plan.validate_loaded(engine.drafter.as_ref().map(|d| d.cfg.drafts()))?;
        tracing::info!(plan_rows = engine.attention_plan.rows(),
            active_drafts = ?engine.attention_plan.active_drafts(), "immutable attention reduction plan");
        let ranks = |peers: &str| -> Result<(Vec<std::net::SocketAddr>, cuteafd_transport::expert::SparkWorld)> {
            let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let world = cuteafd_transport::expert::SparkWorld::canonical(peers.len())?;
            Ok((peers, world))
        };
        let config = cuteafd_transport::TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(),
            timeout: std::time::Duration::from_secs(120), max_frame_bytes: if native { cuteafd_transport::expert::native_frame_budget(cuteafd_core::ExpertGeometry::GLM5, u32::try_from(args.prefill_rows.max(engine::DECODE_ROWS))?)? } else { 64 << 20 } };
        let row_bytes = self.cfg.hidden * 2;
        crate::shared::memory_report::release_load_staging(&self.library);
        // Connect every rank and register full-size buffers now: the first
        // request otherwise pays seconds of connection setup.
        let (h, topk, experts, layer) =
            (self.cfg.hidden, self.cfg.topk, self.cfg.experts,
                working.spark.spark_layers.first().copied().unwrap_or(self.cfg.first_moe_layer) as u32);
        let warm = move |rows: usize, decode: bool| -> Result<cuteafd_transport::ExpertProtocolV2Request> {
            let routes = (0..rows * topk).map(|i| cuteafd_transport::ExpertProtocolV2RouteEntry {
                row_index: (i / topk) as u32, expert_id: (i % experts) as u32, gate_weight: 0.0,
            }).collect();
            let mut request = cuteafd_transport::ExpertProtocolV2Request::new(1, 17, layer,
                h as u32, if engine.spark_bf16 { cuteafd_transport::ExpertV2Dtype::Bf16 }
                    else { cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32 },
                (0..rows as u32).map(|row| cuteafd_transport::ExpertProtocolV2RowDescriptor {
                    row_id: u64::from(row), source_kind: if decode { cuteafd_transport::ExpertV2SourceKind::Decode }
                        else { cuteafd_transport::ExpertV2SourceKind::Prefill },
                    source_request_id: 1, token_position: u64::from(row), route_offset: row * topk as u32,
                    route_count: topk as u32,
                }).collect(),
                routes, vec![0; rows * (if engine.spark_bf16 { h * 2 } else { h + h / 32 })])?;
            request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            Ok(request)
        };
        let mut transport = selected_peers.map(|peers| -> Result<SparkLink<'_>> {
            let (peers, world) = ranks(peers)?;
            let mut link = SparkLink::new(&self.library, world, &peers, engine::DECODE_ROWS as u32, config.clone(), row_bytes)?;
            // The bootstrap opens every rank before any wave does, so the
            // installed grouping is readable from here on. The warm-up below
            // then reuses these sessions.
            link.bootstrap(&warm(engine::DECODE_ROWS, true)?)?;
            Ok(link)
        }).transpose()?;
        // One prefill transport with `prefill_lanes` waves in flight (PLAN
        // "Prefill contract"): each in-flight unit lands in its own return slot.
        let mut prefill_link = match selected_peers {
            Some(peers) => {
                let (peers, world) = ranks(peers)?;
                let mut link = SparkLink::new_waves(&self.library, world, &peers, u32::try_from(args.prefill_rows)?, config.clone(), row_bytes,
                    prefill_lanes, u32::try_from(args.prefill_rows)?)?;
                link.bootstrap(&warm(args.prefill_rows, false)?)?;
                Some(link)
            }
            None => None,
        };
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        let started = Instant::now();
        for (link, decode) in transport.iter_mut().map(|link| (link, true))
            .chain(prefill_link.iter_mut().map(|link| (link, false))) {
            // A full-width wave through each link's exchange (rings and the
            // Sparks' first-use costs), landed and drained on the stream.
            let rows = link.capacity() as usize;
            let exchange = link.exchange()?;
            let ticket = exchange.send_request(warm(rows, decode)?)?;
            exchange.land_drained(ticket, stream)?;
        }
        if transport.is_some() {
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, prefill_waves = prefill_lanes,
                "Spark expert transports warm");
        }
        *engine.prefill_link.borrow_mut() = prefill_link;
        engine.prefill_runtime = Some(tokio::runtime::Builder::new_current_thread().enable_all().build()?);
        engine.configure_packing(args.prefill_packing.unwrap_or_default())?;
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::GLM_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
            if engine.ranks() > 1 {
                engine.attach_peer_l2(budget)?;
            }
        }
        engine.prime_tp2()?;
        // Every decode graph shape serving replays, captured now (CUTEAFD_GLM_WARM_GRAPHS=0 skips it).
        if engine.weights.layers.len() == self.cfg.layers
            && std::env::var("CUTEAFD_GLM_WARM_GRAPHS").map_or(true, |v| v != "0")
            && (transport.is_some() || args.skip_routed_experts || engine.has_tp2()) {
            engine.warm_decode_graphs(transport.as_mut().map(|t| (t, &runtime)))?;
        }
        engine.check_tp2_routes(transport.as_mut().map(|t| (t, &runtime)))?;
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        let result = body(&engine, transport.as_mut(), &runtime);
        drop(engine);
        drop(transport);
        // SAFETY: the engine that used the streams is gone.
        unsafe {
            self.library.cuda_stream_destroy(stream)?;
            if let Some((device, stream)) = peer_stream {
                self.library.cuda_set_device(device)?;
                let destroyed = self.library.cuda_stream_destroy(stream);
                self.library.cuda_set_device(args.device)?;
                destroyed?;
            }
        }
        result
    }
}


impl Opened {
    /// An EXLR8 checkpoint's own dense operands on every coordinator rank: the
    /// rank-0 row-FP8 vocabulary owner (head and embedding), then each layer's
    /// native operands through the dense push ranks when the group offered them
    /// (`push`), else the mount.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn load_native<'s>(&'s self, args: &EngineArgs, records: &[cuteafd_loader::plan::exlr8::dense::DenseRecord],
        layers: usize, programs: &'s cuteafd_ffi::programs::Programs<'s>, _manifest: &serde_json::Value,
        loader: &weights::GlmLoader<'s>, devices: &[i32], stream: *mut c_void, peer_stream: Option<(i32, *mut c_void)>,
        push: Option<(crate::shared::experts::rtx::push::CoordinatorPush, usize)>)
        -> Result<(crate::shared::token_io::TokenEmbedding<'s>, (weights::GlmWeights<'s>, Vec<Vec<weights::GlmLayer<'s>>>),
            Option<Vec<crate::shared::experts::rtx::push::PushedExperts<'s>>>)> {
        use crate::shared::token_io::native::{NativeVocabulary, VocabularyPlan};
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
        let world = devices.len();
        let started = Instant::now();
        self.library.cuda_set_device(devices[0])?;
        let (_, available) = cuteafd_loader::formats::mapped_preload::host_memory()?;
        let available = usize::try_from(available)?;
        let vocabulary = NativeVocabulary::load(&self.library, programs,
            VocabularyPlan::new(records, world, 0, self.cfg.hidden, self.cfg.vocab_size)?, available, 0, 0)?
            .context("rank 0 owns the native vocabulary")?;
        let vocabulary = std::rc::Rc::new(vocabulary);
        let source = self.native_embed_source()?;
        let embedding = crate::shared::token_io::TokenEmbedding::row_fp8(&self.library, source, vocabulary.clone())?;
        let per_rank = (0..world).map(|rank| native_dense::NativeDenseRecords::new(&self.snapshot, layers, world, rank))
            .collect::<Result<Vec<_>>>()?;
        let rank_stream = |device: i32| if device == args.device { stream } else { peer_stream.map_or(stream, |(_, s)| s) };
        let ((model, shares), pushed) = match push {
            Some((push, sparks)) => {
                let load = crate::shared::experts::rtx::dense::DenseLoad::push(&self.library);
                let model = loader.native_model(&self.cfg, layers, &per_rank, &load, vocabulary.clone())?;
                let records = load.sink.as_ref().context("push sink")?.take_all();
                let dense_devices: Vec<_> = devices.iter().map(|&id| crate::shared::memory::device::Device {
                    library: &self.library, id }).collect();
                let planned = crate::shared::experts::rtx::dense::plan_records(&self.snapshot, records.clone(),
                    &dense_devices, sparks);
                let (pushed, ()) = push.receive_with(dense_devices, Some(&self.catalog), sparks, Some(planned), || ());
                let pushed = pushed?;
                if pushed.is_none() {
                    crate::shared::experts::rtx::dense::fill_records_by_mount(&self.library, &records, rank_stream)?;
                }
                tracing::info!(pushed = pushed.is_some(), operands = records.operands.len(), steps = records.steps.len(),
                    "GLM native dense operands landed");
                (model, pushed)
            }
            None => {
                let load = crate::shared::experts::rtx::dense::DenseLoad::mount(&self.library);
                (loader.native_model(&self.cfg, layers, &per_rank, &load, vocabulary.clone())?, None)
            }
        };
        self.library.cuda_set_device(args.device)?;
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "GLM native dense weights resident");
        Ok((embedding, (model, shares), pushed))
    }

    /// The row-FP8 embedding's identity (vocabulary file, payload offset).
    fn native_embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let records = cuteafd_loader::plan::exlr8::dense::DenseRecord::inventory(&self.snapshot)?;
        let record = records.iter().find(|r| r.tensor == "model.embed_tokens.weight").context("native embedding")?;
        let payload = record.components.iter().find(|c| c.suffix == "fp8").context("native embedding payload")?;
        Ok(crate::shared::token_io::EmbedSource { path: payload.source.path.clone(), offset: payload.source.offset,
            vocab: record.shape[0], hidden: record.shape[1], snapshot: None, tensor_name: record.tensor.clone() })
    }

    /// The checkpoint's `model.embed_tokens.weight` (BF16 [vocab, hidden]).
    fn embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let catalog = &self.catalog;
        let tensor = catalog.tensor("model.embed_tokens.weight")?;
        crate::shared::token_io::EmbedSource::new(catalog.snapshot(), &tensor.shard, &tensor.metadata,
            self.cfg.hidden)
    }
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn similarity(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE), diff.sqrt() / nb.sqrt().max(f64::MIN_POSITIVE))
}

pub(crate) async fn run_golden(mut args: GoldenArgs) -> Result<()> {
    args.engine.full_prefill_logits |= args.nll || args.resume_at.is_some();
    tokio::task::spawn_blocking(move || golden(args)).await?
}

/// Mean KL(golden || engine) of logits rows, in float64.
fn mean_kl(logits: &[f32], golden: &[f32], vocab: usize) -> f64 {
    let log_softmax = |l: &[f32]| -> Vec<f64> {
        let top = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = top + l.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln();
        l.iter().map(|&x| x as f64 - lse).collect()
    };
    let rows = logits.len() / vocab;
    (0..rows).map(|r| {
        let (p, q) = (log_softmax(&golden[r * vocab..][..vocab]), log_softmax(&logits[r * vocab..][..vocab]));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum::<f64>()
    }).sum::<f64>() / rows.max(1) as f64
}

/// --nll: score one retained prefill unit per call, downloading its logits
/// before the next call reuses the workspace: mean NLL, top-1 agreement, KL.
fn nll_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, mut transport: Option<&mut SparkLink<'_>>,
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let vocab = opened.cfg.vocab_size;
    let mut placement = engine::PageAllocator::new(engine).admit(tokens.len())?;
    let started = Instant::now();
    let mut logits = Vec::new();
    let rows = crate::shared::probe::scoring_prefill_capacity(true, engine.prefill_rows, engine.prefill_capacity());
    for chunk in tokens.chunks(rows) {
        logits.extend(engine.prefill_rows_logits(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)),
            None, chunk.len())?.context("the prefill needs every layer")?);
    }
    let seconds = started.elapsed().as_secs_f64();
    if let Some(dir) = &args.save_logits {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("tokens.bin"), tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())?;
        std::fs::write(dir.join("logits.bin"), logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    let nll_of = |l: &[f32], next: u32| {
        let top = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        top + l.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln() - l[next as usize] as f64
    };
    let Ok(golden) = std::fs::read(args.golden.join("logits.bin")) else {
        // No golden logits (a token file alone): the engine's NLL of the text and a digest of
        // every logit (bitwise A/B between builds and runs).
        use std::hash::{Hash, Hasher};
        let mut digest = std::collections::hash_map::DefaultHasher::new();
        logits.iter().for_each(|v| v.to_bits().hash(&mut digest));
        let nll: f64 = (0..tokens.len() - 1).map(|r| nll_of(&logits[r * vocab..][..vocab], tokens[r + 1])).sum();
        println!("prefill logits: {} tokens in {seconds:.2} s | mean NLL engine {:.4} | logits digest {:016x}",
            tokens.len(), nll / (tokens.len() - 1) as f64, digest.finish());
        return Ok(());
    };
    let golden: Vec<f32> = golden.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    let rows = tokens.len();
    let (mut agree, mut nll, mut golden_nll) = (0usize, 0f64, 0f64);
    for r in 0..rows {
        let (ours, theirs) = (&logits[r * vocab..][..vocab], &golden[r * vocab..][..vocab]);
        agree += usize::from(argmax(ours) == argmax(theirs));
        if let Some(&next) = tokens.get(r + 1) {
            nll += nll_of(ours, next);
            golden_nll += nll_of(theirs, next);
        }
    }
    println!("prefill logits: {rows} tokens in {seconds:.2} s | top-1 agreement {:.2}% | mean NLL engine {:.4} golden \
        {:.4} | mean KL(golden||engine) {:.5}", 100.0 * agree as f64 / rows as f64, nll / (rows - 1) as f64,
        golden_nll / (rows - 1) as f64, mean_kl(&logits, &golden[..logits.len()], vocab));
    Ok(())
}

/// --bench-verify: verify-step cost by rows (the DFlash2 policy's step
/// table) with the host's phase split, then the drafter step by sequences.
/// One verify step of `rows` teacher-forced rows from `start`: every layer's residual
/// rows (BF16, the real rows only) and the logits, then the sequence rewinds.
fn hooked_verify(engine: &engine::GlmEngine<'_>, placement: &mut engine::GlmPlacement, start: usize,
    tokens: &[u32], mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime)
    -> Result<(Vec<Vec<u8>>, Vec<f32>, Vec<engine::WidthTap>)> {
    let rows = tokens.len();
    let row_bytes = engine.cfg.hidden * 2;
    let mut layers = Vec::new();
    let mut hook = |_: usize, h: &[u8]| -> Result<()> {
        layers.push(h[..rows * row_bytes].to_vec());
        Ok(())
    };
    placement.len = start;
    *engine.width_taps.borrow_mut() = Some(Vec::new());
    let logits = engine.verify(&mut [(placement, rows)], tokens, transport.as_deref_mut().map(|t| (t, runtime)),
        Some(&mut hook));
    let taps = engine.width_taps.borrow_mut().take().unwrap_or_default();
    placement.len = start;
    let logits = logits?.context("width invariance needs every layer")?;
    let expected = engine.weights.layers.iter().enumerate().filter_map(|(i, layer)| (!layer.dense).then_some(i))
        .collect::<Vec<_>>();
    validate_width_taps(&taps, &expected, rows, engine.cfg.hidden, engine.cfg.topk)?;
    Ok((layers, logits, taps))
}

fn validate_width_taps(taps: &[engine::WidthTap], expected: &[usize], rows: usize, hidden: usize, topk: usize) -> Result<()> {
    ensure!(taps.len() == expected.len() && !expected.is_empty(), "width diagnostics missed MoE layer taps");
    for (tap, &layer) in taps.iter().zip(expected) {
        ensure!(tap.layer == layer && tap.x.len() == rows * hidden * 2 && tap.delta.len() == tap.x.len()
            && tap.routes.len() == rows * topk * 8, "incomplete width diagnostic tap for MoE layer {layer}");
    }
    Ok(())
}

/// Where a differing row first departs inside its first differing MoE layer:
/// the MoE input (attention and norm), the staged routes (router), or the
/// routed + shared output (Spark experts, exchange and reduce).
fn tap_split(taps: &[engine::WidthTap], reference: &[engine::WidthTap], row: usize, hidden: usize, topk: usize) -> String {
    let rows = |tap: &engine::WidthTap| tap.routes.len() / (topk * 8);
    for (tap, base) in taps.iter().zip(reference) {
        let x = tap.x[row * hidden * 2..(row + 1) * hidden * 2] != base.x[..hidden * 2];
        let (n, m) = (rows(tap), rows(base));
        let route = |t: &engine::WidthTap, n: usize, r: usize| (t.routes[r * topk * 4..(r + 1) * topk * 4].to_vec(),
            t.routes[n * topk * 4 + r * topk * 4..n * topk * 4 + (r + 1) * topk * 4].to_vec());
        let (ids, weights) = route(tap, n, row);
        let (base_ids, base_weights) = route(base, m, 0);
        let delta = tap.delta[row * hidden * 2..(row + 1) * hidden * 2] != base.delta[..hidden * 2];
        if x || ids != base_ids || weights != base_weights || delta {
            return format!(" [MoE layer {}: input {} ids {} weights {} output {}]", tap.layer,
                if x { "differs" } else { "same" }, if ids != base_ids { "differ" } else { "same" },
                if weights != base_weights { "differ" } else { "same" }, if delta { "differs" } else { "same" });
        }
    }
    String::new()
}

/// [`GoldenArgs::width_invariance`].
fn width_invariance(args: &GoldenArgs, engine: &engine::GlmEngine<'_>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let max_rows = *args.width_invariance.iter().max().unwrap();
    ensure!(args.width_invariance.iter().all(|&w| (1..=engine::DECODE_ROWS).contains(&w)),
        "--width-invariance takes widths 1..={}", engine::DECODE_ROWS);
    ensure!(transport.is_some() || engine.skip_routed()
        || engine.weights.layers.iter().enumerate().all(|(i, layer)| layer.dense
            || engine.tp2_layers().is_some_and(|layers| layers.contains(&i))),
        "--width-invariance needs Spark peers, local experts, or --skip-routed-experts");
    let prefill = args.prefill.unwrap_or(512);
    let tokens: Vec<u32> = tokens.iter().copied().cycle().take(prefill + max_rows).collect();
    let mut placement = engine::PageAllocator::new(engine).admit(prefill + max_rows)?;
    for chunk in tokens[..prefill].chunks(engine.prefill_capacity()) {
        engine.prefill(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)), None)?;
    }
    let start = placement.len;
    let row_bytes = engine.cfg.hidden * 2;
    let vocab = engine.cfg.vocab_size;
    // Width 1 at every row position is the reference: a decode step of that row alone
    // after the rows before it (teacher-forced, so each sees the same history).
    let mut reference: Vec<(Vec<Vec<u8>>, Vec<f32>, Vec<engine::WidthTap>)> = Vec::new();
    for r in 0..max_rows {
        // Rows before r enter the cache through width-1 steps too, so every width's row r
        // sees byte-identical history.
        for k in 0..r {
            placement.len = start + k;
            engine.verify(&mut [(&mut placement, 1)], &tokens[start + k..start + k + 1],
                transport.as_deref_mut().map(|t| (t, runtime)), None)?;
        }
        let (layers, logits, taps) = hooked_verify(engine, &mut placement, start + r, &tokens[start + r..start + r + 1],
            transport.as_deref_mut(), runtime)?;
        reference.push((layers, logits, taps));
    }
    println!("width invariance after {prefill} tokens: rows compared bitwise against width-1 decode at the same \
        position and history (first differing layer per row; L = logits)");
    let mut widths = vec![1];
    widths.extend(args.width_invariance.iter().copied().filter(|&w| w > 1));
    widths.push(1);
    let mut clean = true;
    for (pass, &width) in widths.iter().enumerate() {
        let (layers, logits, taps) = hooked_verify(engine, &mut placement, start, &tokens[start..start + width],
            transport.as_deref_mut(), runtime)?;
        let mut differing = Vec::new();
        for r in 0..width {
            let (ref_layers, ref_logits, ref_taps) = &reference[r];
            let layer = (0..layers.len()).find(|&i| layers[i][r * row_bytes..(r + 1) * row_bytes]
                != ref_layers[i][..row_bytes]);
            let logits_differ = logits[r * vocab..(r + 1) * vocab].iter().map(|v| v.to_bits())
                .ne(ref_logits[..vocab].iter().map(|v| v.to_bits()));
            let split = tap_split(&taps, ref_taps, r, engine.cfg.hidden, engine.cfg.topk);
            if layer.is_some() || logits_differ || !split.is_empty() {
                let max = logits[r * vocab..(r + 1) * vocab].iter().zip(&ref_logits[..vocab])
                    .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                differing.push(format!("row {r}: layer {} (max |dlogit| {max:.3e}){}",
                    layer.map_or("L".to_string(), |i| i.to_string()),
                    split));
            }
        }
        clean &= differing.is_empty();
        println!("  width {width:2}{}: {}", if pass + 1 == widths.len() { " (repeat)" } else { "" },
            if differing.is_empty() { "bitwise identical".to_string() } else { differing.join("; ") });
    }
    println!("width invariance: {}", if clean { "PASS" } else { "FAIL" });
    ensure!(clean, "width invariance failed");
    Ok(())
}

fn bench_verify(args: &GoldenArgs, engine: &engine::GlmEngine<'_>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime, max_rows: usize) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    ensure!(max_rows >= 1 && max_rows <= engine::DECODE_ROWS, "--bench-verify takes 1..={} rows", engine::DECODE_ROWS);
    ensure!(transport.is_some() || engine.skip_routed(), "--bench-verify needs Spark peers or --skip-routed-experts");
    let prefill = args.bench_context.or(args.prefill).unwrap_or(512);
    let tokens: Vec<u32> = tokens.iter().copied().cycle().take(prefill + max_rows).collect();
    let mut placement = engine::PageAllocator::new(engine).admit(prefill + max_rows)?;
    for chunk in tokens[..prefill].chunks(engine.prefill_capacity()) {
        engine.prefill(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)), None)?;
    }
    let start = placement.len;
    println!("verify cost after {prefill} tokens (teacher-forced, one sequence, median of 7; host phases per step: \
        GPU wait before each exchange, exchange, logits download){}",
        if engine.skip_routed() { " [routed experts skipped]" } else { "" });
    let mut counts: Vec<usize> = (0..).map(|i| 1usize << i).take_while(|&r| r < max_rows).chain([max_rows]).collect();
    if !args.bench_rows.is_empty() {
        ensure!(args.bench_rows.iter().all(|&r| r >= 1 && r <= max_rows), "--bench-rows past --bench-verify");
        counts = args.bench_rows.clone();
    }
    for rows in counts {
        let mut times = Vec::new();
        for round in 0..9 {
            placement.len = start;
            if round == 2 {
                *engine.profile.borrow_mut() = [0.0; 3];
            }
            let started = Instant::now();
            let logits = engine.verify(&mut [(&mut placement, rows)], &tokens[start..start + rows],
                transport.as_deref_mut().map(|t| (t, runtime)), None)?;
            if round >= 2 {
                times.push(started.elapsed().as_secs_f64() * 1e3);
            }
            if let (Some(dir), Some(logits), 8) = (&args.bench_dump, logits, round) {
                std::fs::create_dir_all(dir)?;
                std::fs::write(dir.join(format!("logits-{rows}.bin")),
                    logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
        let phases = std::mem::take(&mut *engine.profile.borrow_mut());
        let n = times.len() as f64;
        times.sort_by(f64::total_cmp);
        println!("  {rows} rows: {:.2} ms (min {:.2}) | gpu wait {:.2} exchange {:.2} logits {:.2} ms", times[times.len() / 2],
            times[0], 1e3 * phases[0] / n, 1e3 * phases[1] / n, 1e3 * phases[2] / n);
    }
    if let Some(drafter) = engine.drafter.as_ref() {
        // The drafter reads its ring context of the prefilled taps; anchors at the prefill's end.
        let n = start.min(dflash::TAP_ROWS);
        drafter.update(&(0..n).map(|r| dflash::ContextRow { tap_row: r, slot: 0, position: start - n + r })
            .collect::<Vec<_>>())?;
        println!("drafter step (anchors at position {start}, median of 7):");
        let max_batch = dflash::ReplayDrafter::sequences(drafter);
        let mut counts: Vec<usize> = (0..).map(|i| 1usize << i).take_while(|&s| s < max_batch)
            .chain([max_batch]).collect();
        if !args.bench_sequences.is_empty() {
            ensure!(args.bench_sequences.iter().all(|&s| s >= 1 && s <= max_batch), "--bench-sequences past the draft batch capacity");
            counts = args.bench_sequences.clone();
        }
        for sequences in counts {
            let seqs: Vec<dflash::DraftSeq> = (0..sequences)
                .map(|slot| dflash::DraftSeq { slot, anchor: tokens[start], position: start, valid_from: 0 }).collect();
            let mut times = Vec::new();
            for round in 0..9 {
                let started = Instant::now();
                let drafts = drafter.draft_device_diagnostic(&seqs, &engine.embedding,
                    engine.draft_target_head()?)?;
                if round >= 2 {
                    times.push(started.elapsed().as_secs_f64() * 1e3);
                }
                if let (Some(dir), 8) = (&args.bench_dump, round) {
                    std::fs::write(dir.join(format!("drafts-{sequences}.txt")),
                        drafts.iter().map(|d| format!("{:?}\n", d.tokens)).collect::<String>())?;
                }
            }
            times.sort_by(f64::total_cmp);
            println!("  {sequences} sequences: {:.2} ms (min {:.2})", times[times.len() / 2], times[0]);
        }
    }
    Ok(())
}

/// --bench-prefill: fresh sequences of --bench-prefill-tokens tokens.
fn bench_prefill(args: &GoldenArgs, engine: &engine::GlmEngine<'_>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let n = args.bench_prefill_tokens;
    let long: Vec<u32> = tokens.iter().copied().cycle().take(n).collect();
    let mut allocator = engine::PageAllocator::new(engine);
    let mut times = Vec::new();
    for round in 0..=args.bench_prefill {
        if round == 1 {
            *engine.profile.borrow_mut() = [0.0; 3];
        }
        let mut placement = allocator.admit(n)?;
        let started = Instant::now();
        for chunk in long.chunks(engine.prefill_capacity()) {
            engine.prefill(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)), None)?;
        }
        // Round 0 warms the workspaces.
        if round > 0 {
            times.push(started.elapsed().as_secs_f64());
        }
        allocator.release(placement);
    }
    times.sort_by(f64::total_cmp);
    let (median, runs) = (times[times.len() / 2], times.len() as f64);
    let phases = *engine.profile.borrow();
    let host = *engine.exchange_host.borrow();
    println!("prefill bench host per prefill: request build + post {:.1} ms, partial copies {:.1} ms (measured runs \
        and warm-up; build {:.1} ms, post {:.1} ms)", 1e3 * host[0] / (runs + 1.0), 1e3 * host[1] / (runs + 1.0),
        1e3 * host[2] / (runs + 1.0), 1e3 * host[3] / (runs + 1.0));
    println!("prefill bench: {n} tokens, median {:.1} ms ({:.0} tok/s), min {:.1} ms; per prefill GPU wait {:.1} ms, \
        Spark wait {:.1} ms; runs {:?} ms", 1e3 * median, n as f64 / median, 1e3 * times[0], 1e3 * phases[0] / runs,
        1e3 * phases[1] / runs, times.iter().map(|t| (t * 1e3).round() as u32).collect::<Vec<_>>());
    Ok(())
}

fn golden(mut args: GoldenArgs) -> Result<()> {
    args.engine.prefill_packing = Some(args.engine.prefill_packing.unwrap_or_default());
    // Reject invalid explicit geometry before checkpoint I/O.
    resolve_prefill(&mut args.engine)?;
    // Validate the scoring call before opening CUDA or loading any weights.
    let rows = args.engine.prefill_rows;
    crate::shared::prefill_pipeline::retained_logits_units(rows, rows,
        usize::from(args.engine.prefill_waves), 256,
        if args.nll || args.resume_at.is_some() { rows } else { 1 },
        if args.engine.full_prefill_logits { args.engine.prefill_rows } else { engine::DECODE_ROWS })?;
    let opened = open(&args.engine)?;
    opened.with_engine(&args.engine, |engine, transport, runtime| golden_run(&args, &opened, engine, transport, runtime))
}

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, mut transport: Option<&mut SparkLink<'_>>,
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    if let Some(sequences) = args.packed_check {
        let tokens = std::fs::read(args.golden.join("tokens.bin"))?.chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        return packed_check::check(engine, &tokens, &args.golden, sequences,
            args.engine.prefill_packing.unwrap_or_default(), transport, runtime);
    }
    if let Some(dir) = &args.draft_oracle {
        return draft_oracle(args, opened, engine, dir);
    }
    if let Some(start) = args.draft_replay {
        let drafter = engine.drafter.as_ref().context("--draft-replay needs --draft")?;
        let (tokens, greedy) = dflash::golden_sequence(&args.golden, opened.cfg.vocab_size)?;
        let hidden = opened.cfg.hidden;
        let layers: Vec<Vec<u8>> = drafter.cfg.taps.iter()
            .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
        let row = hidden * 2;
        let taps = |first: usize, n: usize| -> Result<Vec<u8>> {
            let mut taps = vec![0u8; n * layers.len() * row];
            for r in 0..n {
                for (i, layer) in layers.iter().enumerate() {
                    taps[(r * layers.len() + i) * row..][..row].copy_from_slice(&layer[(first + r) * row..][..row]);
                }
            }
            Ok(taps)
        };
        return dflash::replay(drafter, &tokens, &greedy, &taps, &|t| engine.embedding.host_rows(t),
            engine.weights.head.bf16()?.buffer.ptr, start);
    }
    if let Some(rows) = args.bench_verify {
        return bench_verify(args, engine, transport, runtime, rows);
    }
    if !args.width_invariance.is_empty() {
        return width_invariance(args, engine, transport, runtime);
    }
    if let Some(at) = args.resume_at {
        let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
            .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return prefix::resume_check(engine, &tokens, at, n,
            args.prefill_chunk.unwrap_or(engine.prefill_rows), args.resume_decode, args.resume_cold,
            args.resume_repeat, transport.map(|t| (t, runtime)));
    }
    if let Some(steps) = args.token_check {
        return token_check(args, opened, engine, transport, runtime, steps);
    }
    if engine.drafter.is_some() {
        return draft_run(args, opened, engine, transport, runtime);
    }
    if args.nll || args.bench_prefill > 0 {
        ensure!(transport.is_some() || engine.skip_routed() || engine.has_tp2(),
            "--nll and --bench-prefill need Spark peers, local TP2 or --skip-routed-experts");
        if args.nll {
            nll_run(args, opened, engine, transport.as_deref_mut(), runtime)?;
        }
        if args.bench_prefill > 0 {
            bench_prefill(args, engine, transport, runtime)?;
        }
        return Ok(());
    }
    if let Some(copies) = args.sequences {
        return multi_run(args, opened, engine, copies, transport, runtime);
    }
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut placement = engine::PageAllocator::new(engine).admit(tokens.len())?;
    let row = cfg.hidden * 2;
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    let started = Instant::now();
    // Rows [first, first + n) of layer `layer`'s golden output.
    let compare = |layer: usize, first: usize, stream: &[u8], worst: &mut Vec<f64>| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        if let Ok(golden) = std::fs::read(&path) {
            ensure!(golden.len() >= (first * row + stream.len()), "golden layer {layer} is short");
            let golden = &golden[first * row..][..stream.len()];
            let (cosine, rel) = similarity(&bf16s(stream), &bf16s(golden));
            if worst.len() <= layer {
                worst.resize(layer + 1, 1.0);
            }
            worst[layer] = worst[layer].min(cosine);
            if first == 0 {
                println!("layer {layer:2}: cosine {cosine:.6} rel_l2 {rel:.3e}");
            }
        }
        Ok(())
    };
    let mut worst = Vec::new();
    let logits = engine.prefill(&mut placement, &tokens[..prefill], transport.as_deref_mut().map(|t| (t, runtime)),
        Some(&mut |layer, stream| compare(layer, 0, stream, &mut worst)))?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut decode_worst = Vec::new();
    let mut decode_logits: Vec<f32> = Vec::new();
    let mut position = prefill;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        let mut layer_compare = |layer: usize, stream: &[u8]| compare(layer, first, stream, &mut decode_worst);
        let on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
            if args.no_layer_compare { None } else { Some(&mut layer_compare) };
        if let Some(logits) = engine.verify(&mut [(&mut placement, n)], &tokens[position..position + n],
            transport.as_deref_mut().map(|t| (t, runtime)), on_layer)? {
            decode_logits.extend(logits);
        }
        position += n;
    }
    if prefill < tokens.len() {
        println!("decode: {} rows in steps of {} in {:.2} s; worst row-block cosine per layer {:?}", tokens.len() - prefill,
            args.step_rows, started.elapsed().as_secs_f64(),
            decode_worst.iter().map(|c| format!("{c:.6}")).collect::<Vec<_>>());
    }
    if !decode_logits.is_empty() {
        let golden: Vec<f32> = std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let vocab = cfg.vocab_size;
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        let (mut agree, mut next_ok, mut golden_next) = (0usize, 0usize, 0usize);
        let rows = decode_logits.len() / vocab;
        for r in 0..rows {
            let (ours, theirs) = (&decode_logits[r * vocab..][..vocab], &golden[(prefill + r) * vocab..][..vocab]);
            agree += usize::from(argmax(ours) == argmax(theirs));
            if let Some(&next) = tokens.get(prefill + r + 1) {
                next_ok += usize::from(argmax(ours) == next as usize);
                golden_next += usize::from(argmax(theirs) == next as usize);
            }
        }
        println!("decode logits: top-1 agreement {:.1}% over {rows} rows | next-token accuracy engine {:.1}% golden {:.1}%",
            100.0 * agree as f64 / rows as f64, 100.0 * next_ok as f64 / (rows - 1).max(1) as f64,
            100.0 * golden_next as f64 / (rows - 1).max(1) as f64);
        use std::hash::{Hash, Hasher};
        let mut digest = std::collections::hash_map::DefaultHasher::new();
        decode_logits.iter().for_each(|v| v.to_bits().hash(&mut digest));
        println!("decode logits: mean KL(golden||engine) {:.5} | logits digest {:016x}",
            mean_kl(&decode_logits, &golden[prefill * vocab..][..decode_logits.len()], vocab), digest.finish());
        // A/B between builds or attention placements: every decode row's logits (F32).
        if let Some(dir) = &args.save_logits {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("decode-logits.bin"),
                decode_logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
    }
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s");
    if let Some(logits) = logits {
        let golden: Vec<f32> = std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let last = &golden[(prefill - 1) * cfg.vocab_size..][..cfg.vocab_size];
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).context("empty");
        let (cosine, _) = similarity(&logits, last);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(&logits)?, argmax(last)?);
    }
    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

/// Prefills the golden prompt, then decodes one row per step (teacher-forced
/// on tokens.bin, or greedy with --generate), drafting with DFlash2 before
/// every step; reports the accepted prefix per step against the sequence.
fn draft_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, mut transport: Option<&mut SparkLink<'_>>,
    runtime: &tokio::runtime::Runtime) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft")?;
    let mut sequence: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let prefill = args.prefill.unwrap_or(sequence.len() / 2).min(sequence.len() - 1);
    let end = match args.generate {
        Some(n) => {
            sequence.truncate(prefill);
            prefill + n
        }
        None => sequence.len(),
    };
    let mut placement = engine::PageAllocator::new(engine).admit(end + 1)?;
    let started = Instant::now();
    let logits = engine.prefill(&mut placement, &sequence[..prefill],
        transport.as_deref_mut().map(|t| (t, runtime)), None)?.context("drafting needs every target layer")?;
    let n = prefill.min(dflash::TAP_ROWS);
    drafter.update(&(0..n).map(|r| dflash::ContextRow { tap_row: r, slot: 0, position: prefill - n + r })
        .collect::<Vec<_>>())?;
    if args.generate.is_some() {
        sequence.push(argmax(&logits));
    }
    println!("prefill: {prefill} tokens in {:.2} s", started.elapsed().as_secs_f64());
    let (mut drafts, mut draft_seconds) = (Vec::new(), 0f64);
    let started = Instant::now();
    for position in prefill..end {
        let anchor = sequence[position];
        let timer = Instant::now();
        let draft = drafter.draft_device_diagnostic(&[dflash::DraftSeq { slot: 0, anchor, position, valid_from: 0 }],
            &engine.embedding, engine.draft_target_head()?)?;
        draft_seconds += timer.elapsed().as_secs_f64();
        drafts.push((position, draft.into_iter().next().context("draft")?));
        let logits = engine.verify(&mut [(&mut placement, 1)], &[anchor], transport.as_deref_mut().map(|t| (t, runtime)),
            None)?.context("decode needs every layer")?;
        drafter.update(&[dflash::ContextRow { tap_row: 0, slot: 0, position }])?;
        if args.generate.is_some() {
            sequence.push(argmax(&logits));
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let block = drafter.cfg.drafts();
    let mut histogram = vec![0usize; block + 1];
    let mut first_rank = [0usize; 16];
    for (position, draft) in &drafts {
        let truth = &sequence[position + 1..];
        if truth.len() < block {
            continue;
        }
        let accepted = draft.tokens.iter().zip(truth).take_while(|(d, t)| d == t).count();
        histogram[accepted] += 1;
        first_rank[draft.features[0][3] as usize] += 1;
    }
    let steps: usize = histogram.iter().sum();
    let accepted: usize = histogram.iter().enumerate().map(|(i, c)| i * c).sum();
    println!("drafts: {steps} steps, {accepted} drafted tokens accepted as a prefix ({:.2} per step, {:.1}% of {}), \
        histogram {histogram:?}, first-draft selector rank {first_rank:?}, {:.2} ms/draft, {:.1} ms/step",
        accepted as f64 / steps.max(1) as f64, 100.0 * accepted as f64 / (steps * block).max(1) as f64, steps * block,
        draft_seconds * 1e3 / drafts.len().max(1) as f64, seconds * 1e3 / drafts.len().max(1) as f64);
    if args.generate.is_some() {
        let text = cuteafd_loader::LoadedTokenizer::from_snapshot(&opened.snapshot)?
            .decode_ids(&sequence[prefill..], false).map(|d| d.text).unwrap_or_default();
        println!("generated: {text:?}");
    }
    Ok(())
}

/// Runs the drafter alone on the golden taps at reference.py's anchor
/// positions and compares tokens, selector features and final-norm rows.
fn draft_oracle(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, dir: &std::path::Path) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-oracle needs --draft")?;
    let (hidden, block, drafts_per) = (opened.cfg.hidden, drafter.cfg.block, drafter.cfg.drafts());
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let positions: Vec<usize> = meta["positions"].as_array().context("positions")?.iter()
        .map(|p| p.as_u64().map(|p| p as usize).context("position")).collect::<Result<_>>()?;
    let words = |name: &str| -> Result<Vec<u32>> {
        Ok(std::fs::read(dir.join(name))?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let (ref_tokens, ref_features) = (words("drafts.bin")?, words("features.bin")?);
    let ref_hidden = bf16s(&std::fs::read(dir.join("hidden.bin"))?);
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let layers: Vec<Vec<u8>> = drafter.cfg.taps.iter()
        .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
    let row = hidden * 2;
    let width = layers.len() * row;
    let (mut done, mut exact, mut first, mut matched, mut worst, mut feature_error) = (0usize, 0, 0, 0, 1f64, 0f64);
    let mut draft_seconds = 0f64;
    for (index, &position) in positions.iter().enumerate() {
        // Context rows done..position through the tap buffer.
        while done < position {
            let n = (position - done).min(dflash::TAP_ROWS);
            let mut taps = vec![0u8; n * width];
            for r in 0..n {
                for (i, layer) in layers.iter().enumerate() {
                    taps[r * width + i * row..][..row].copy_from_slice(&layer[(done + r) * row..][..row]);
                }
            }
            drafter.put_taps(&taps)?;
            drafter.update(&(0..n).map(|r| dflash::ContextRow { tap_row: r, slot: 0, position: done + r })
                .collect::<Vec<_>>())?;
            // SAFETY: the oracle owns the stream; a following chunk reuses
            // the taps and context metadata read by this update.
            unsafe { engine.library.cuda_stream_synchronize(engine.stream)? };
            done += n;
        }
        let anchor = tokens[position];
        let timer = Instant::now();
        let draft = drafter.draft_device_diagnostic(&[dflash::DraftSeq { slot: 0, anchor, position, valid_from: 0 }],
            &engine.embedding, engine.draft_target_head()?)?.remove(0);
        draft_seconds += timer.elapsed().as_secs_f64();
        let reference = &ref_tokens[index * drafts_per..][..drafts_per];
        exact += usize::from(draft.tokens == reference);
        first += usize::from(draft.tokens[0] == reference[0]);
        matched += draft.tokens.iter().zip(reference).take_while(|(a, b)| a == b).count();
        let (cosine, _) = similarity(&bf16s(&drafter.last_hidden(1)?), &ref_hidden[index * block * hidden..][..block * hidden]);
        worst = worst.min(cosine);
        if draft.tokens[0] == reference[0] {
            let theirs = f32::from_bits(ref_features[index * drafts_per * 4]);
            feature_error = feature_error.max(f64::from((draft.features[0][0] - theirs).abs()));
        }
        if draft.tokens != reference {
            println!("position {position}: engine {:?} reference {reference:?}", draft.tokens);
        }
    }
    let n = positions.len();
    println!("draft oracle: {n} anchors, identical drafts {exact}/{n}, first draft {first}/{n}, matching prefix \
        {:.2} of {drafts_per}, worst final-norm cosine {worst:.6}, first-margin max error {feature_error:.4}, \
        {:.2} ms/draft", matched as f64 / n as f64, draft_seconds * 1e3 / n as f64);
    Ok(())
}

/// Several copies of the golden sequence verified in the same decode steps.
fn multi_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, copies: usize,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime) -> Result<()> {
    let vocab = opened.cfg.vocab_size;
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let golden: Vec<f32> = std::fs::read(args.golden.join("logits.bin"))?
        .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let prefill = args.prefill.unwrap_or(512);
    let mut pages = engine::PageAllocator::new(engine);
    let mut placements = Vec::new();
    for copy in 0..copies {
        let mut placement = pages.admit(tokens.len())?;
        let n = prefill - 37 * copy;
        engine.prefill(&mut placement, &tokens[..n],
            transport.as_deref_mut().map(|t| (t, runtime)), None)?;
        placements.push(placement);
    }
    let (mut agree, mut total) = (vec![0usize; copies], vec![0usize; copies]);
    loop {
        let counts: Vec<usize> = placements.iter().map(|p| args.step_rows.min(tokens.len() - p.len)).collect();
        if counts.contains(&0) {
            break;
        }
        let mut rows = Vec::new();
        for (p, &n) in placements.iter().zip(&counts) {
            rows.extend_from_slice(&tokens[p.len..p.len + n]);
        }
        let starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
        let mut step: Vec<(&mut engine::GlmPlacement, usize)> = placements.iter_mut().zip(&counts).map(|(p, &n)| (p, n)).collect();
        let logits = engine.verify(&mut step, &rows,
            transport.as_deref_mut().map(|t| (t, runtime)), None)?.context("decode needs every layer")?;
        let mut offset = 0;
        for (copy, (&start, &n)) in starts.iter().zip(&counts).enumerate() {
            for r in 0..n {
                let ours = &logits[(offset + r) * vocab..][..vocab];
                let theirs = &golden[(start + r) * vocab..][..vocab];
                agree[copy] += usize::from(argmax(ours) == argmax(theirs));
                total[copy] += 1;
            }
            offset += n;
        }
    }
    println!("multi: {copies} copies, {} rows each per step: top-1 agreement per copy {:?}", args.step_rows,
        agree.iter().zip(&total).map(|(a, t)| format!("{:.1}% of {t}", 100.0 * *a as f64 / *t as f64)).collect::<Vec<_>>());
    Ok(())
}

/// `--token-check N`: prefill the golden prompt (--prefill tokens of it), then
/// [`crate::shared::token_io::gate`] over N greedy decode steps.
fn token_check(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime, steps: usize) -> Result<()> {
    ensure!(transport.is_some() || engine.skip_routed(), "--token-check needs Spark peers or --skip-routed-experts");
    let mut tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(p) = args.prefill {
        tokens.truncate(p);
    }
    let mut allocator = engine::PageAllocator::new(engine);
    let mut placement = allocator.admit(tokens.len() + steps + 1)?;
    let mut last = None;
    for chunk in tokens.chunks(engine.prefill_capacity()) {
        last = engine.prefill_device(&mut placement, chunk, transport.as_deref_mut().map(|t| (t, runtime)), 1)?;
    }
    let last = last.context("--token-check needs every layer")?.row_host(&opened.library, 0)?;
    let first = cuteafd_core::TargetSamplingParams::greedy().select_token(&last, None, 0)? as u32;
    let result = crate::shared::token_io::gate(&opened.library, &engine.embedding, first, steps, |token| {
        engine.verify_device(&mut [(&mut placement, 1)], &[token], transport.as_deref_mut().map(|t| (t, runtime)))?
            .context("decode needs every layer")
    });
    allocator.release(placement);
    result
}
