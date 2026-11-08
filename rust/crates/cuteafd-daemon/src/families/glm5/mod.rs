//! GLM 5.x (glm_moe_dsa) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
use crate::shared::speculation::dflash;
pub(crate) mod engine;
pub(crate) mod prefix;
mod packed_check;
pub(crate) mod serve;
pub(crate) mod ml4_media;
pub(crate) mod landing;
pub(crate) mod push;
pub(crate) mod weights;
pub(crate) mod core_runtime;
pub(crate) mod drafter;

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
    #[command(flatten)]
    pub(crate) attention: crate::cli::AttentionArgs,
    /// Count executed expert routes; --expert-usage=false disables telemetry.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set,
        num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    pub expert_usage: bool,
    #[arg(skip)]
    pub expert_usage_reader: cuteafd_core::expert_usage::Reader,
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
    /// Served context: auto fits the admitted pool; a positive count pins it.
    #[arg(long = "max-context-tokens", default_value = "auto", value_parser = crate::shared::context::parse_tokens::<usize>)]
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
    /// Pool-first by default; `all` pins every routed-expert layer of the
    /// target model and `N` pins exactly N. Local requests require an
    /// exported RTX backend.
    #[arg(long, env = "RTX_EXPERT_LAYERS", default_value = "auto")]
    pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    /// Spark expert ranks in TP order (HOST:PORT,...), for MoE layers.
    #[arg(long)]
    pub peers: Option<String>,
    /// DFlash2 drafter snapshot (incoai/GLM-5.3-DFlash2); taps the target
    /// layers and drafts on the coordinator GPU.
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Mistral Large 4: draft with the checkpoint's bundled mtp_dspark (params.json
    /// mtp; mtp_dspark.* tensors) on the shared MLA block-draft path.
    #[arg(long, conflicts_with = "draft")]
    pub dspark: bool,
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
    /// Prefix-cache marks of a window-ring drafter (mtp_dspark): each holds one slot's draft
    /// windows; serve sets it from the prefix mark rule (0: no retained draft windows).
    #[arg(long, hide = true, default_value_t = 0)]
    pub draft_window_marks: usize,
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
    cuteafd_bench::context::ResolvedSettings::prefill(shape).publish();
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
    ensure!(attention.is_none_or(|placement| placement == AttentionPlacement::Heads
        || (args.split_device.is_some() && matches!(placement, AttentionPlacement::Context | AttentionPlacement::Layers))),
        "EXLR8 full execution requires heads or two-GPU context/layers attention placement");
    Ok(())
}

/// Layouts one coordinator GPU admits for native dense execution before any allocation: decode
/// scratch plus the prefill scratch every slot shares and, with A8 prefill, its MX8 records.
/// `cuteafd plan --layout` charges the same bytes (`plan::layout::glm5::native_layout`).
fn native_dense_layouts(cfg: &GlmDsaConfig, records: &[cuteafd_loader::plan::exlr8::dense::DenseRecord],
    plan: &cuteafd_loader::plan::exlr8::dense::DensePlan, args: &EngineArgs)
    -> Result<[cuteafd_loader::families::glm5::native::ScratchLayout; 2]> {
    cuteafd_loader::families::glm5::native::scratch_layouts(cfg, records, plan, engine::DECODE_ROWS,
        args.prefill_rows, args.exlr8_prefill_activations == Exlr8PrefillActivations::A8)
}

use cuteafd_loader::plan::spark_executions::SparkBackend;

/// The decode link plus one prefill link carrying every wave in flight, over
/// typed owners from the shared admission (one prefill scratch at the ladder
/// maximum, waves only buffer per-unit I/O).
fn spark_model(family: &str, backend: SparkBackend, waves: usize, unit_rows: u32, bf16: bool)
    -> Result<cuteafd_transport::worker_selection::ModelAdmission> {
    ensure!(waves > 0, "GLM Spark admission requires at least one prefill wave");
    let native = backend == SparkBackend::Exlr8;
    let ingress = if native { cuteafd_core::SparkIngress::Fp8K32 }
        else if bf16 { cuteafd_core::SparkIngress::Bf16 } else { cuteafd_core::SparkIngress::Fp8K32 };
    let executions = cuteafd_loader::plan::spark_executions::admit_family(family, &backend.exports(), None,
        u64::from(unit_rows), waves as u64)?;
    let rings = crate::shared::placement::spark_rings(if native {
        cuteafd_transport::expert::native_frame_budget(cuteafd_core::ExpertGeometry::GLM5, executions.capacity())?
    } else { 64 << 20 })?;
    let model = cuteafd_transport::worker_selection::ModelAdmission::new(
        vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity: engine::DECODE_ROWS as u32, ingress, partial_reply_bytes: None },
            cuteafd_core::SparkEndpoint { ring_depth: None, capacity: unit_rows, ingress, partial_reply_bytes: None }], rings, executions)?;
    validate_spark_endpoint_count(family, model.endpoints.len(), waves)?;
    Ok(model)
}

fn validate_spark_endpoint_count(family: &str, endpoints: usize, waves: usize) -> Result<()> {
    ensure!(endpoints == cuteafd_loader::plan::layout::spark_endpoints(family, waves as u64),
        "GLM Spark endpoint count differs from the planner");
    Ok(())
}

#[cfg(test)]
mod draft_cli_tests {
    use super::*;

    #[test]
    fn featureless_drafts_have_no_selector_rank() {
        let mut ranks = [0; 16];
        record_selector_rank(&mut ranks, &[]);
        assert_eq!(ranks, [0; 16]);
        record_selector_rank(&mut ranks, &[[0., 0., 0., 3.]]);
        assert_eq!(ranks[3], 1);
        record_selector_rank(&mut ranks, &[[0., 0., 0., 16.]]);
        assert_eq!(ranks.iter().sum::<usize>(), 1);
    }

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
                assert_eq!(spark_model("glm5", backend, waves, unit, false)?.executions,
                    planned("glm5", backend, None, u64::from(unit), waves as u64), "{backend:?} k{waves}");
            }
        }
        // Mistral Large 4 runs this engine under its own descriptor.
        for (waves, unit) in [(1, 4096), (3, 2752)] {
            assert_eq!(spark_model("mistral_large4", k4, waves, unit, false)?.executions,
                planned("mistral_large4", k4, None, u64::from(unit), waves as u64), "ml4 k{waves}");
        }
        Ok(())
    }

    #[test]
    fn spark_contract_matches_decode_link_and_parallel_prefill_lanes() -> Result<()> {
        for (waves, unit) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048)] {
            for bf16 in [false, true] {
                let model = spark_model("glm5", SparkBackend::Fp8, waves, unit, bf16)?;
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
            let native = spark_model("glm5", SparkBackend::Exlr8, waves, 4096, false)?;
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
        assert!(spark_model("glm5", SparkBackend::Fp8, 0, 4096, false).is_err());
        assert!(validate_spark_endpoint_count("glm5", 1, 1).is_err());
        assert!(validate_spark_endpoint_count("glm5", 3, 3).is_err());
        assert!(validate_spark_endpoint_count("glm5", 2, 1).is_ok());
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
            crate::commands::bench::assert_resolved_family_seam("glm5",
                &["max-context-tokens", "pool-tokens", "rtx-expert-layers", "draft-window-marks", "attention-plan-rows"], || {
                    crate::commands::bench::fixture_prefix().resolve_entries(16);
                    serve::resolve_serving_capacity(&mut args, 16);
                    resolve_prefill(&mut args)?;
                    spark_model("glm5", SparkBackend::Fp8, usize::from(args.prefill_waves), args.prefill_rows as u32, true)?;
                    Ok(())
                }).unwrap();
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
    fn bench_verify_mla_tail_binds_and_populates_every_requested_slot() -> Result<()> {
        use cuteafd_loader::formats::{dflash::{MlaDflashConfig, MlaDraftCacheState,
            MlaDraftCacheRequest}, mla::MlaGeometry, block_dspark::DsparkRotaryPlan};
        let cfg = MlaDflashConfig { mla: MlaGeometry { hidden: 128, heads: 2, q_rank: 0,
            kv_rank: 128, nope: 64, rope: 64, value: 128 }, intermediate: 128, eps: 1e-5,
            block: 3, group: 0, selector_rank: 512, selector_top_k: 0, taps: vec![0, 1],
            vocab: 16, mask_token: 0, max_context: 32768, windows: vec![Some(2048); 2],
            rotary: DsparkRotaryPlan { inv_freq: vec![1.; 32], attention_factor: 1., softmax_factor: 1. },
            kind: cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark, markov_rank: 512 };
        let state = std::cell::RefCell::new(MlaDraftCacheState::new(
            cfg.cache_layout(3, 32768, cuteafd_core::MlaCacheFormat::Bf16)?)?);
        for start in [17, dflash::TAP_ROWS + 19] {
            for sequences in [1, 3, 2] {
                let origin = bench_verify_draft_context(start, sequences,
                    |slot, origin| Ok(state.borrow_mut().reset(slot, origin)?),
                    |rows| {
                        let slot = rows[0].slot;
                        let first = rows[0].position;
                        for (offset, row) in rows.iter().enumerate() {
                            assert_eq!((row.slot, row.tap_row, row.position), (slot, offset, first + offset));
                        }
                        let mut state = state.borrow_mut();
                        for layer in 0..2 { state.context_rows(layer, slot, first, rows.len())?; }
                        Ok(state.commit_context(slot, first, rows.len())?)
                    })?;
                assert_eq!(origin, start.saturating_sub(dflash::TAP_ROWS));
                let requests = (0..sequences).map(|slot| {
                    assert_eq!(state.borrow().frontier(slot).unwrap(), (origin, start));
                    MlaDraftCacheRequest { slot, position: start, valid_from: origin }
                }).collect::<Vec<_>>();
                for layer in 0..2 { state.borrow().block_batches(layer, &requests)?; }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires full ML4, CUDA and Spark peers; CUTEAFD_ML4_BENCH_VERIFY_ARGS is a JSON golden argv list"]
    fn bench_verify_mla_drafter_writes_each_requested_batch() -> Result<()> {
        #[derive(Parser)]
        struct Golden { #[command(flatten)] args: GoldenArgs }
        let argv: Vec<String> = serde_json::from_str(&std::env::var("CUTEAFD_ML4_BENCH_VERIFY_ARGS")?)?;
        let mut args = Golden::try_parse_from(std::iter::once("ml4-bench-verify".to_owned()).chain(argv))?.args;
        ensure!(args.engine.dspark && args.engine.peers.is_some() && !args.engine.skip_routed_experts,
            "MLA bench regression requires the bundled ML4 drafter and live experts");
        let output = tempfile::tempdir()?;
        args.bench_verify = Some(1);
        args.bench_rows = vec![1];
        args.bench_sequences = vec![1, 2];
        args.bench_context = Some(dflash::TAP_ROWS + 19);
        args.bench_dump = Some(output.path().to_owned());
        golden(args)?;
        ensure!(!std::fs::read(output.path().join("logits-1.bin"))?.is_empty(), "missing target logits");
        for sequences in [1, 2] {
            let drafts = std::fs::read_to_string(output.path().join(format!("drafts-{sequences}.txt")))?;
            ensure!(drafts.lines().count() == sequences, "missing requested MLA draft slots");
        }
        Ok(())
    }

    #[test]
    fn golden_layer_dump_round_trips_bf16_rows() -> Result<()> {
        let root = tempfile::tempdir()?;
        let output = root.path().join("layers");
        let stream: Vec<u8> = [1.0f32, -2.5, 0.0, 16.0].into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()).collect();
        dump_golden_layer(&output, 3, &stream)?;
        let loaded = std::fs::read(output.join("layer03.bin"))?;
        assert_eq!(loaded, stream);
        assert_eq!(bf16s(&loaded), vec![1.0, -2.5, 0.0, 16.0]);
        let (cosine, relative) = similarity(&bf16s(&loaded), &bf16s(&stream));
        assert!(cosine > 0.999999);
        assert_eq!(relative, 0.0);
        assert!(dump_golden_layer(&output, 4, &[0]).is_err());
        assert!(!output.join("layer04.bin").exists());
        Ok(())
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
    /// geometry: the planner's per-GPU item equals the admitted native scratch layouts.
    #[test]
    fn planner_charge_equals_admission_charge_for_the_same_geometry() {
        use cuteafd_loader::plan::exlr8::PrefillActivations;
        cuteafd_loader::placement::families::glm5::admission::allow_synthetic_graphs();
        const GROUP: &str = "GLM target hooks";
        let dir = tempfile::tempdir().unwrap();
        cuteafd_loader::plan::testing::exlr8::glm_admission_snapshot(dir.path());
        let manifest = cuteafd_loader::plan::testing::exlr8::glm_admission_programs();
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(dir.path()).unwrap();
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config).unwrap();
        let records = cuteafd_loader::plan::exlr8::dense::DenseRecord::inventory(dir.path()).unwrap();
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
                    let charge = (0..gpus).map(|rank| {
                        let plan = cuteafd_loader::plan::exlr8::dense::DensePlan::for_rank(&records, gpus, rank).unwrap();
                        cuteafd_loader::families::glm5::native::scratch_bytes(
                            &native_dense_layouts(&cfg, &records, &plan, &engine).unwrap()).unwrap()
                    }).collect::<Vec<_>>();
                    // A serving layout allocates its prefill lanes on the first live
                    // prefill and charges them at Growth; scoring runs that prefill
                    // before Ready, so its items hold every lane the admission
                    // charge names. Compare the scoring view.
                    let layout = cuteafd_loader::plan::layout::LayoutOptions {
                        full_prefill_logits: true,
                        rtx_bytes: vec![96 << 30; gpus], prefill_rows: engine.prefill_rows as u64,
                        prefill_lanes: u64::from(engine.prefill_waves), prefill_waves: u64::from(engine.prefill_waves),
                        exlr8_prefill_activations: activations.parse::<PrefillActivations>().unwrap(), context_tokens: 65536,
                        onboard: Some(cuteafd_loader::placement::OnboardSetting::Layers(0)), glm5_drafter_disabled: true,
                        host_memory: Some((128 << 30, 128 << 30)), host_cache_bytes: Some(0), ..Default::default() };
                    let (memory, _) = cuteafd_loader::plan::layout::glm5::native_layout(&checkpoint, &layout,
                        peers.split(',').count(), &manifest, "k4").unwrap();
                    assert_eq!(memory.devices.len(), gpus);
                    for (rank, device) in memory.devices.iter().enumerate() {
                        let planned: u64 = device.items.iter().filter(|item| item.group == GROUP).map(|item| item.bytes).sum();
                        assert_eq!(planned, charge[rank], "{gpus} RTX/{rows} rows/{waves} waves/{activations}");
                    }
                    admitted.push(charge);
                }
                // Every slot shares one prefill scratch and its records (#762).
                for rank in 0..gpus {
                    let plan = cuteafd_loader::plan::exlr8::dense::DensePlan::for_rank(&records, gpus, rank).unwrap();
                    let layouts = native_dense_layouts(&cfg, &records, &plan, &engine("a8")).unwrap();
                    assert_eq!(admitted[1][rank] - admitted[0][rank], layouts[1].linear.records.unwrap() as u64,
                        "{gpus} RTX/{rows}/{waves}");
                }
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
            for mode in ["auto", "heads", "layers"] {
                candidate.attention_placement = mode.into();
                assert!(native_activation(&candidate, 3, 78, true).is_ok());
                assert!(native_activation(&candidate, 3, 78, false).is_ok());
            }
        }
        for mode in ["heads", "auto"] {
            let mut one = args.clone();
            one.split_device = None;
            one.attention_placement = mode.into();
            assert!(native_activation(&one, 3, 78, true).is_ok());
            assert!(native_activation(&one, 3, 78, false).is_ok());
        }
        let mut one_layers = args.clone();
        one_layers.split_device = None;
        one_layers.attention_placement = "layers".into();
        assert!(native_activation(&one_layers, 3, 78, true).is_err());
        for ranks in 1..=8 {
            let mut candidate = args.clone();
            candidate.peers = Some((0..ranks).map(|rank| format!("worker{rank}:1")).collect::<Vec<_>>().join(","));
            assert!(native_activation(&candidate, 3, 78, true).is_ok(), "TP{ranks}");
        }
        let mut drafting = args.clone();
        drafting.draft = Some("/draft".into());
        assert!(native_activation(&drafting, 3, 78, true).is_ok());
        let mut onboard = args.clone();
        onboard.rtx_expert_layers = cuteafd_loader::placement::OnboardSetting::Layers(1);
        assert!(native_activation(&onboard, 3, 78, true).is_ok());
        for bad in 0..8 {
            let mut candidate = args.clone();
            match bad {
                0 => candidate.exlr8_manifest = Some("home".into()),
                1 => candidate.split_device = None,
                2 => candidate.split_device = Some(0),
                3 => candidate.peers = Some("a:1,b:1,c:1,d:1,e:1,f:1,g:1,h:1,i:1".into()),
                4 => candidate.layers = Some(4),
                5 => candidate.attention_placement = "invalid".into(),
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
    fn expert_usage_defaults_on_and_explicit_off_or_skip_disables() {
        for (flag, requested) in [(None, true), (Some("--expert-usage"), true),
            (Some("--expert-usage=true"), true), (Some("--expert-usage=false"), false)] {
            for skip in [false, true] {
                let mut argv = vec!["test", "--snapshot", "/checkpoint", "--native-lib", "/native"];
                argv.extend(flag);
                if skip { argv.push("--skip-routed-experts"); }
                let args = Parse::try_parse_from(argv).unwrap().engine;
                assert_eq!(args.expert_usage, requested);
                assert_eq!(args.expert_usage_reader.configure(args.expert_usage,
                    args.skip_routed_experts).unwrap(), requested && !skip);
            }
        }
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--expert-usage=invalid"]).is_err());
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
    /// Write each prefill layer's BF16 residual rows as layerNN.bin for --golden comparisons.
    #[arg(long)]
    pub layer_dump: Option<PathBuf>,
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
    pub view: cuteafd_loader::plan::checkpoint::CheckpointView,
    pub catalog: cuteafd_loader::OfficialV41Catalog,
    /// An EXLR8 checkpoint: its own dense records run natively.
    pub native: bool,
    pub cfg: GlmDsaConfig,
    pub library: std::sync::Arc<NativeLibrary>,
    startup_audit: std::cell::RefCell<Option<cuteafd_ffi::memory_ledger::AllocationAudit>>,
}

pub(crate) fn validate_selected_attention(view: &cuteafd_loader::plan::checkpoint::CheckpointView,
    cfg: &GlmDsaConfig) -> Result<()> {
    if let Some(ml4) = cfg.variant.ml4() {
        let checkpoint = cuteafd_loader::plan::Checkpoint::open_view_for_role(view,
            "Mistral Large 4 selected attention", |name| name.contains(".attention.wq.")
                || name.contains(".attention.wo."))?;
        cuteafd_loader::families::mistral_large4::attention::validate(&checkpoint, ml4)?;
    } else {
        let checkpoint = cuteafd_loader::plan::Checkpoint::inventory_view(view)?;
        cuteafd_loader::families::glm5::core::validate_native_attention(cfg, &checkpoint)?;
    }
    Ok(())
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    let startup_audit = crate::shared::inventory::startup_probe_enabled()
        .then(cuteafd_ffi::memory_ledger::AllocationAudit::begin);
    let view = args.attention.selected_view.clone().map(Ok)
        .unwrap_or_else(|| args.attention.select(&args.snapshot))?;
    args.attention.validate_view(&view)?;
    let mut catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
    let draft_ownership = catalog.bundled_dspark_ownership(args.dspark)?;
    catalog.select_draft(draft_ownership)?;
    cuteafd_core::set_expert_geometry(catalog.routed_experts().geometry()?)
        .map_err(|g| anyhow::anyhow!("geometry already {g:?}"))?;
    let cfg = GlmDsaConfig::from_hf(&view.config()?)?;
    validate_selected_attention(&view, &cfg)?;
    cuteafd_bench::context::set_resolved("attention-bundle-fingerprint", view.fingerprint());
    cuteafd_bench::context::set_family(cfg.family_id());
    let native = catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8;
    if native {
        ensure!(args.exlr8_manifest.is_some(), "native EXLR8 requires --exlr8-manifest");
        native_activation(args, cfg.first_moe_layer, cfg.layers, false)?;
    } else {
        ensure!(args.exlr8_manifest.is_none()
            && args.exlr8_prefill_activations == Exlr8PrefillActivations::A16,
            "EXLR8 options require a native EXLR8 checkpoint");
    }
    if args.dspark {
        dflash::check_view_target_head(&view, cfg.head(), cfg.hidden, cfg.vocab_size, false)?;
    }
    if let Some(snapshot) = &args.draft {
        // A native EXLR8 target lends its row-FP8 head through a launcher.
        if !native { dflash::check_view_target_head(&view, cfg.head(), cfg.hidden, cfg.vocab_size, false)?; }
        dflash::check_checkpoint(snapshot, args.draft_fp8, args.draft_context_slots, args.draft_sequences)?;
    }
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&args.native_lib) }?);
    library.cuda_set_device(args.device)?;
    let opened = Opened { snapshot: args.snapshot.clone(), view, catalog, native, cfg, library,
        startup_audit: std::cell::RefCell::new(startup_audit) };
    opened.startup_probe(&std::iter::once(args.device).chain(args.split_device).collect::<Vec<_>>(),
        "glm/open", None, false)?;
    Ok(opened)
}

struct ResolvedEngineArgs {
    args: EngineArgs,
    context: cuteafd_loader::placement::planner::Context,
}

impl ResolvedEngineArgs {
    fn new(args: &EngineArgs) -> Result<Self> {
        let requested_context = (args.max_context > 0).then_some(args.max_context as u64);
        let mut resolved = args.clone();
        resolve_prefill(&mut resolved)?;
        resolved.max_context = crate::shared::context::checkpoint_context(
            &args.snapshot, &args.manifest, "glm5", args.max_context)?;
        Ok(Self { context: cuteafd_loader::placement::planner::Context::resolve(resolved.max_context as u64, requested_context), args: resolved })
    }
}

impl Opened {
    fn startup_probe(&self, devices: &[i32], stage: &str,
        census: Option<&crate::shared::memory::receipts::Census>, finish: bool) -> Result<()> {
        use crate::shared::inventory::{startup_probe, StartupEvidence};
        if !crate::shared::inventory::startup_probe_enabled() { return Ok(()); }
        if finish {
            let history = self.startup_audit.borrow_mut().take().map(|audit| audit.finish());
            startup_probe(&self.library, devices, stage,
                StartupEvidence { audit: None, history: history.as_ref(), census })
        } else {
            let audit = self.startup_audit.borrow();
            startup_probe(&self.library, devices, stage,
                StartupEvidence { audit: audit.as_ref(), history: None, census })
        }
    }

    /// Builds the engine (and the Spark transport when peers are given) and
    /// hands them to `body`.
    pub fn with_engine<T>(&self, args: &EngineArgs,
        body: impl FnOnce(&engine::GlmEngine<'_>, Option<&mut SparkLink<'_>>, &tokio::runtime::Runtime) -> Result<T>)
        -> Result<T> {
        self.with_resolved_engine(&ResolvedEngineArgs::new(args)?, body)
    }

    fn with_resolved_engine<T>(&self, resolved: &ResolvedEngineArgs,
        body: impl FnOnce(&engine::GlmEngine<'_>, Option<&mut SparkLink<'_>>, &tokio::runtime::Runtime) -> Result<T>)
        -> Result<T> {
        self.with_resolved_engine_media(resolved, Default::default(), None, (None, 0), body)
    }
    fn with_resolved_engine_media<T>(&self, resolved: &ResolvedEngineArgs,
        media: crate::shared::experts::rtx::dense::Records,
        serving: Option<(&crate::shared::prefix::PrefixArgs, usize)>,
        media_owners: (Option<(i32, u64, u64)>, u64),
        body: impl FnOnce(&engine::GlmEngine<'_>, Option<&mut SparkLink<'_>>, &tokio::runtime::Runtime) -> Result<T>) -> Result<T> {
        let ResolvedEngineArgs { args, context: serving_context } = resolved;
        let expert_usage = args.expert_usage_reader.configure(args.expert_usage, args.skip_routed_experts)
            .map_err(anyhow::Error::msg)?;
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
        // The bundled mtp_dspark drafts its whole block (anchor-as-first).
        let ml4_draft = args.dspark.then(|| cuteafd_loader::placement::families::glm5::ml4_draft_layout(&args.snapshot,
            args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences, args.max_context,
            self.library.sm_count()? as usize)).transpose()?;
        if ml4_draft.is_some() {
            cuteafd_ffi::mla_draft::Ml4ContextProjection::validate(&programs)?;
        }
        let draft_width = provisional_drafter.as_ref().map(|cfg| cfg.drafts())
            .or(ml4_draft.as_ref().map(|layout| layout.capacity().block_rows / layout.capacity().max_batch_sequences));
        let attention_plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
            engine::DECODE_ROWS, draft_width, args.attention_plan_rows)?;
        let legacy_manifest = args.manifest.with_file_name("DSV4_PROGRAMS.json");
        let manifest_path = if !args.manifest.exists()
            && args.manifest.file_name() == Some("PROGRAMS.json".as_ref()) && legacy_manifest.exists() {
            &legacy_manifest
        } else { &args.manifest };
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest_path)?)?;
        // GLM 5.3's own programs (`glm_*`, and the head split's `glm2_*`), sampled as the runtime
        // inventory; the image's other families are never launched here.
        // Mistral Large 4 runs its own `ml4_*` programs (whole attention on each GPU).
        let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: self.cfg.program_family(),
            split_family: args.split_device.and(self.cfg.split_program_family()) };
        let programs = programs.with_attention_plan(attention_plan, |name| selected.contains(name))?;
        let stream = self.library.cuda_stream_create()?;
        // The head split's second GPU and its stream (load kernels, then the engine's).
        // A head split needs its share's programs (`glm2`) in this build.
        let split_device = match args.split_device {
            // Mistral Large 4 pairs GPUs as whole-layer owners without its share programs.
            Some(device) if !self.cfg.variant.is_glm() => Some(device),
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
        self.startup_probe(&devices, "glm/streams-before-programs", None, false)?;
        use cuteafd_loader::families::glm5::DensePrograms;
        let dense_programs = if native { DensePrograms::Exlr8 } else { DensePrograms::Fp8 };
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let requested = cuteafd_loader::placement::attention::parse(&args.attention_placement)
            .map_err(anyhow::Error::msg)?;
        let context = if devices.len() == 2 && !matches!(requested,
            Some(cuteafd_loader::placement::AttentionPlacement::Heads | cuteafd_loader::placement::AttentionPlacement::Layers)) {
            let sms = devices.iter().map(|&device| crate::shared::peer_split::on_device(
                &self.library, device, || self.library.sm_count())).collect::<Result<Vec<_>>>()?;
            cuteafd_loader::placement::families::glm5::manifest_context_for_cards(
                &self.cfg, &manifest, &sms, attention_plan)?
        } else { None };
        let requested = cuteafd_loader::placement::families::glm5::resolve_context_attention(
            &self.cfg, requested, devices.len(), context.is_some(), !native)?;
        ensure!(requested != Some(cuteafd_loader::placement::AttentionPlacement::Layers) || devices.len() == 2,
            "GLM layers attention requires two coordinator GPUs");
        let owners = (requested == Some(cuteafd_loader::placement::AttentionPlacement::Layers))
            .then(|| cuteafd_loader::placement::families::glm5::attention_owners(&self.cfg, layers)).transpose()?;
        let whole_mla = owners.as_ref().map(|owners| owners.iter().enumerate()
            .map(|(layer, &rank)| (layer, usize::from(rank))).collect()).unwrap_or_default();
        let native_records = native.then(|| cuteafd_loader::plan::exlr8::dense::DenseRecord::inventory(&self.snapshot)).transpose()?;
        let native_dense_records = if native {
            (0..devices.len()).map(|rank| cuteafd_loader::plan::exlr8::dense::requests::NativeDenseRecords::new(
                &self.snapshot, layers, devices.len(), rank, &whole_mla))
                .collect::<Result<Vec<_>>>()?
        } else { Vec::new() };
        let native_selected = if let Some(records) = &native_records {
            let config = serde_json::from_slice(&std::fs::read(self.snapshot.join("config.json"))?)?;
            let selection = cuteafd_loader::plan::exlr8::dense::DenseSelection::from_hf(&config)?;
            let plans = native_dense_records.iter().map(|rank| rank.plan.clone()).collect::<Vec<_>>();
            cuteafd_loader::plan::exlr8::dense::programs::resolve_activations(&manifest, &config,
                records, layers, &plans, Some(args.exlr8_prefill_activations), Exlr8PrefillActivations::default())?;
            native_dense_records.iter().map(|rank| cuteafd_loader::plan::exlr8::dense::programs::select(
                &programs, &manifest, &selection, records, layers, &rank.plan,
                args.exlr8_prefill_activations == Exlr8PrefillActivations::A8)).collect::<Result<Vec<_>>>()?
        } else { Vec::new() };
        let mut native_load: std::collections::BTreeSet<String> = native_selected.iter().flatten().cloned().collect();
        if native {
            native_load.extend(cuteafd_loader::plan::exlr8::dense::programs::vocabulary(
                &programs, self.cfg.hidden, self.cfg.vocab_size)?);
        }
        // Preserve native primitive split stages, but load only selected dense
        // backbone geometry and the separately owned vocabulary.
        let inventory = crate::shared::inventory::RuntimeInventory::measure(&self.library, &devices, crate::shared::inventory::InventoryPhase::PreWeight, |_| {
            let (loaded, skipped) = programs.load_matching(|name| selected.contains(name)
                || (native && (native_load.contains(name) || dense_programs.stage(name, true)
                    || (cuteafd_core::coordinator_programs::program_namespace(name) == Some("exlr8")
                        && !name.starts_with("exlr8_dense_")))))?;
            tracing::info!(loaded, skipped, "GLM programs loaded");
            Ok(())
        })?;
        self.startup_probe(&devices, "glm/programs-before-weights", None, false)?;
        let mut draft_file = None;
        use cuteafd_loader::placement::{families::glm5 as glm, planner::*};
        use glm::{admission::*, local_physical::*};
        let checkpoint = (!native).then(|| {
            let view = dense_catalog.selected_view().context("GLM dense catalog has no selected view")?;
            cuteafd_loader::plan::Checkpoint::coordinator_view(view, false, false)
        }).transpose()?;
        // The EXLR8 checkpoint's own dense records, per coordinator rank.
        let native_plans = if native {
            (0..devices.len()).map(|rank| crate::shared::experts::exlr8::NativePlan::new(&self.snapshot,
                args.exlr8_manifest.as_deref().context("EXLR8 manifest")?,
                self.cfg.first_moe_layer, layers, &programs, cuteafd_ffi::exlr8::KernelFamily::Glm,
                devices.len(), rank))
                .collect::<Result<Vec<_>>>()?
        } else { Vec::new() };
        // Shared admission below charges native loader owners and the host tier together.
        let host_embedding = args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Host;
        ensure!(!native || !host_embedding, "the EXLR8 row-FP8 embedding is device-resident (its owner holds the head too)");
        if let Some(checkpoint) = &checkpoint {
            if host_embedding { checkpoint.require_untied_embedding(self.cfg.embedding())?; }
        }
        let spark_bf16 = checkpoint.as_ref().is_some_and(|c| c.spark_input_is_bf16());
        // Native dense: decode plus every prefill lane's scratch and opt-in A8 records.
        let native_scratch = native_dense_records.iter().map(|owned|
            native_dense_layouts(&self.cfg, native_records.as_ref().context("native records")?, &owned.plan, args))
            .collect::<Result<Vec<_>>>()?;
        if !native_scratch.is_empty() {
            use cuteafd_loader::plan::exlr8::dense::{DenseSelection, scratch::LinearScratchLayout};
            let config = serde_json::from_slice(&std::fs::read(self.snapshot.join("config.json"))?)?;
            let selection = DenseSelection::from_hf(&config)?;
            let records = native_records.as_ref().context("native records")?;
            for (owned, layouts) in native_dense_records.iter().zip(&native_scratch) {
                for (layout, rows, a8) in [(&layouts[0], engine::DECODE_ROWS, false),
                    (&layouts[1], args.prefill_rows, args.exlr8_prefill_activations == Exlr8PrefillActivations::A8)] {
                    ensure!(layout.linear == LinearScratchLayout::from_manifest(&manifest, &selection,
                        records, layers, &owned.plan, rows, a8)?, "admitted native scratch differs from exported route");
                }
            }
        }
        inventory.record("glm5", &inventory.gpus.iter().map(|gpu| gpu.free_bytes).collect::<Vec<_>>());
        let cards = devices.iter().map(|&device| crate::shared::peer_split::on_device(&self.library, device, || {
            let (_, physical) = self.library.cuda_allocatable_memory_info()?;
            Ok((physical as u64, cuteafd_ffi::coordinator_gpu_budget().map_or(physical as u64, |budget| budget.0)))
        })).collect::<Result<Vec<_>>>()?;
        let scratch = |decode| -> Result<_> {
            let lookup = |name: &str| programs.spec(name).ok().map(|p| p.scratch.get("scratch").copied().unwrap_or(0));
            let mut scratch = glm::step_scratch(&self.cfg, devices.len() == 2, decode, lookup)?;
            if owners.is_some() { scratch.programs = scratch.programs.max(glm::step_scratch(&self.cfg, false, decode, lookup)?.programs); }
            Ok(scratch)
        };
        let draft_facts = match (&ml4_draft, args.draft.as_deref(), &provisional_drafter) {
            (Some(layout), _, _) => DraftFacts::ml4(layout)?,
            (None, Some(snapshot), Some(draft)) => DraftFacts::gqa(snapshot,
                cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::from_fp8_option(args.draft_fp8),
                cuteafd_loader::families::glm5::draft_representation::GlmDraftCapacity::new(
                    args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences, draft.block)?,
                self.library.sm_count()? as usize, 0, args.draft_kv_cache)?,
            _ => DraftFacts::default(),
        };
        let spark_ranks = args.peers.as_deref().map_or(0, |p| p.split(',').count());
        // Every per-slot owner covers the waves in flight.
        let prefill_lanes = usize::from(args.prefill_waves);
        ensure!(owners.is_none() || prefill_lanes <= cuteafd_loader::placement::LAYER_PIPELINE_MAX_LANES as usize,
            "layer-owned prefill lacks capacity for {prefill_lanes} waves");
        let core = checkpoint.as_ref().map(|checkpoint| -> Result<_> {
            let core = cuteafd_loader::families::glm5::core::CoreInventory::from_checkpoint(&self.cfg, checkpoint)?;
            if core.projections.is_empty() { return Ok(None); }
            // Draft rows the K6 target head scores per step: DFlash2's sequences x block, or
            // mtp_dspark's whole admitted block (the planner's drafter_admission rows).
            let draft_rows = match (args.draft.as_deref(), &ml4_draft) {
                (Some(snapshot), _) => Some({
                    let config = cuteafd_loader::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
                    let (_, block) = cuteafd_loader::families::glm5::draft_representation::draft_geometry(&config)?;
                    args.draft_sequences.checked_mul(usize::try_from(block)?).context("GLM K6 draft head rows")?
                }),
                (None, Some(layout)) => Some(layout.capacity().block_rows),
                (None, None) => None,
            };
            Ok(Some(cuteafd_loader::families::glm5::core::CoreCharges::from_package(&self.cfg,
                &core, layers, devices.len(), owners.as_deref(), args.prefill_rows, prefill_lanes,
                draft_rows, &manifest)?))
        }).transpose()?.flatten();
        // Context/program and graph bytes come from maintained recipe figures.
        let exchange_f32 = glm::local_exchange_f32()?;
        let onboard = args.rtx_expert_layers.resolve(self.catalog.routed_experts().routed_layers());
        let local = glm::select_local_inventory(&self.cfg, &self.catalog, manifest_path, layers, devices.len(),
            args.prefill_rows as u64, prefill_lanes as u64, exchange_f32, onboard,
            args.skip_routed_experts, args.exlr8_manifest.as_deref()
                .map(|selected| (selected, args.exlr8_prefill_activations)))?;
        if let Some(local) = &local {
            let max_rows = args.prefill_rows.max(engine::DECODE_ROWS);
            match local.format {
                glm::GlmLocalFormat::Exl3 => ensure!(crate::shared::experts::rtx::exl3::Exl3Tp2::workspace_bytes_for(
                    &local.package, self.cfg.hidden, max_rows)? as u64 == local.backend_workspace,
                    "GLM EXL3 planner/runtime backend workspace mismatch"),
                glm::GlmLocalFormat::Exlr8 => ensure!(
                    crate::shared::experts::rtx::exlr8::Exlr8RtxWeights::workspace_bytes_with_limit(
                        &programs, cuteafd_ffi::exlr8::KernelFamily::Glm, max_rows, local.unit_limit,
                        args.exlr8_prefill_activations)? as u64 == local.backend_workspace,
                    "GLM EXLR8 planner/runtime backend workspace mismatch"),
                glm::GlmLocalFormat::Nvfp4 { .. } => {
                    let tensors = self.catalog.fp8().context("NVFP4 tensors")?;
                    for rank in 0..2 {
                        let plan = crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::plan(tensors, &local.package, &[tensors.shape().first_layer], rank, max_rows)?;
                        ensure!(plan.workspace_bytes as u64 == local.backend_workspace,
                            "GLM NVFP4 planner/runtime backend workspace mismatch");
                    }
                }
            }
        }
        let attention = requested.unwrap_or(if devices.len() == 2 {
            cuteafd_loader::placement::AttentionPlacement::Context
        } else { cuteafd_loader::placement::AttentionPlacement::Heads });
        let page_tokens = if attention == cuteafd_loader::placement::AttentionPlacement::Context { 128 } else { 64 };
        let requested_pool = cuteafd_loader::placement::PoolPolicy::normalize_request(Some(args.pool_tokens as u64));
        let maximum_pool_tokens = cuteafd_loader::placement::PoolPolicy::resolve(
            &cards.iter().map(|&(_, ceiling)| ceiling).collect::<Vec<_>>(), args.max_context as u64,
            requested_pool, page_tokens).maximum_tokens();
        let maximum_pages = maximum_pool_tokens.div_ceil(page_tokens);
        let prefill = PrefillShape { rows: args.prefill_rows as u64, waves: prefill_lanes as u64 };
        let operation = if args.full_prefill_logits { Operation::Scoring } else { Operation::Serving };
        let push_policy = crate::shared::experts::rtx::push::PushPolicy::for_snapshot(&self.snapshot)?;
        let storage = match push_policy { crate::shared::experts::rtx::push::PushPolicy::Mount => LocalStorage::Mount,
            crate::shared::experts::rtx::push::PushPolicy::Required => LocalStorage::Push };
        let target = if let Some(checkpoint) = &checkpoint {
            TargetFacts::checkpoint(checkpoint, layers, devices.len(), owners.as_deref(), cuteafd_loader::families::glm5::native_bf16(),
                &[if host_embedding { EmbeddingHome::HostPinned } else { EmbeddingHome::Device(0) }], core.as_ref())?
        } else {
            let config = serde_json::from_slice(&std::fs::read(self.snapshot.join("config.json"))?)?;
            let selection = cuteafd_loader::plan::exlr8::dense::DenseSelection::from_hf(&config)?;
            let vocabulary = (0..devices.len()).map(|rank| crate::shared::token_io::native::VocabularyPlan::new(
                native_records.as_ref().context("native records")?, &selection, devices.len(), rank,
                self.cfg.hidden, self.cfg.vocab_size).map(|plan| plan.dense_plan().clone())).collect::<Result<Vec<_>>>()?;
            let shared = native_plans.iter().map(|plan| plan.residency().to_vec()).collect::<Vec<_>>();
            let workspace = cuteafd_loader::plan::exlr8::shared_execution_workspace(&manifest, cuteafd_ffi::exlr8::KernelFamily::Glm)?;
            TargetFacts::native(&selection, layers, &native_dense_records.iter().map(|owned| owned.plan.clone()).collect::<Vec<_>>(),
                &vocabulary, &shared, &native_scratch, &workspace, storage)?
        };
        let arena_layouts = target.weights[0].1.coordinator_arenas()?;
        let native_load = target.native_load.clone();
        let reserved_weights = target.device_weight_bytes(if host_embedding { EmbeddingHome::HostPinned }
            else { EmbeddingHome::Device(0) })?;
        let physical = LocalPhysicalInput { layers: (self.cfg.first_moe_layer..layers).collect(),
            max_rows: args.prefill_rows.max(engine::DECODE_ROWS), lanes: prefill_lanes, storage,
            route_checks: true, peer_diagnostics: std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1"),
            activations: args.exlr8_prefill_activations };
        let metadata = args.exlr8_manifest.as_deref().map(|selected|
            cuteafd_loader::plan::exlr8::residency::MetadataInput::open(&self.snapshot, selected)).transpose()?;
        let source = local.as_ref().map(|local| LocalPhysicalSource::read(&self.cfg, &self.catalog, local,
            &physical, metadata.as_ref())).transpose()?;
        let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
        let cache = if let Some(ml4) = self.cfg.variant.ml4() {
            cuteafd_loader::serving_capacity::mistral_large4_cache_geometry(ml4, 1)?
        } else { cuteafd_loader::serving_capacity::glm_cache_geometry(&self.cfg, layers, 1)? };
        let mark_bytes = ml4_draft.as_ref().map(glm::draft_window_mark_bytes).transpose()?.unwrap_or(0);
        let host_config = if devices.len() == 2 && attention == cuteafd_loader::placement::AttentionPlacement::Heads { None }
            else { serving.map(|(prefix, concurrency)| prefix.resolve_host_config(cuteafd_engine::prefix::FamilyLayout {
                page_rows: 64, pages: maximum_pool_tokens as usize / 64,
                page_bytes: cache.ranks[0].persistent_unit_bytes as usize, mark_bytes, draft_bytes: 0,
                rule: cuteafd_engine::prefix::ReuseRule::EXACT, mark_store: cuteafd_engine::prefix::MarkStore::Arena,
                page_owners: if attention == cuteafd_loader::placement::AttentionPlacement::Context {
                    cuteafd_engine::prefix::PageOwners::Parity } else { cuteafd_engine::prefix::PageOwners::Uniform },
            }, args.max_context, concurrency)).transpose()?.flatten() };
        let push_headers = checkpoint.is_none().then(|| cuteafd_loader::plan::Checkpoint::coordinator_view(
            self.catalog.selected_view().context("GLM push selected view")?, false, args.dspark)).transpose()?;
        let push_headers = checkpoint.as_ref().or(push_headers.as_ref()).context("GLM selected push headers")?;
        let largest_native = native_records.iter().flatten().flat_map(|record| &record.components)
            .map(|component| component.source.bytes as u64).max().unwrap_or(0);
        let admitted_push_host = push_host_envelope(&push_headers, &self.cfg, layers, devices.len(),
            Some(&self.catalog), metadata.as_ref(), largest_native)?;
        let mut additional = Vec::new();
        if let Some(envelope) = &admitted_push_host {
            additional.extend(envelope.allocations("glm.load.push")?.into_iter().map(|owner| {
                let bytes = ByteCurve::fixed(owner.bytes, maximum_pages);
                Component::allocation(owner, bytes, false, maximum_pages)
            }));
        }
        let mut blas = Vec::new();
        if let Some((device, weights, total)) = media_owners.0 {
            let rank = devices.iter().position(|&id| id == device).context("vision device absent from GLM placement")?;
            let location = Location::coordinator(rank as u8);
            blas.push(serving::BlasOwners { name: "GLM local vision".into(), process: "coordinator".into(),
                thread: "glm/blas/media".into(), location, gemm: true, lt: false,
                lifetime: Lifetime::RESIDENT, activation: Activation::always() });
            let owner = Allocation::buffer("glm.encoder.weights", location, weights,
                Lifetime::RESIDENT, cuteafd_core::memory_layout::Category::Weights, "bundled encoder weights");
            additional.push(Component::resident_weight(ResidentWeight::from_allocation(&owner,
                cuteafd_loader::placement::ResidentWeights(weights)), ByteCurve::fixed(weights, maximum_pages)));
            additional.push(Component::allocation(owner, ByteCurve::fixed(weights, maximum_pages), false, maximum_pages));
            let scratch = total.checked_sub(weights).context("vision receipt smaller than weights")?;
            if scratch > 0 {
                let owner = Allocation::buffer("glm.encoder.scratch", location, scratch,
                    Lifetime::RESIDENT, cuteafd_core::memory_layout::Category::Workspace, "bundled encoder scratch");
                additional.push(Component::allocation(owner, ByteCurve::fixed(scratch, maximum_pages), false, maximum_pages));
            }
        }
        if media_owners.1 > 0 {
            let owner = Allocation::buffer("glm.media.cache", Location::HOST, media_owners.1,
                Lifetime::TARGET, cuteafd_core::memory_layout::Category::Embedding, "media embedding cache");
            additional.push(Component::allocation(owner, ByteCurve::fixed(media_owners.1, maximum_pages), false, maximum_pages));
        }
        if serving.is_some() {
            use cuteafd_loader::placement::planner::serving::{ServingOwners, SelectorOwners, HostPrefixOwners};
            additional.extend(ServingOwners { selector: SelectorOwners { location: Location::coordinator(0),
                capacity: engine::DECODE_ROWS, vocab: self.cfg.vocab_size, device_sampling: true },
                prefix: HostPrefixOwners { quota_bytes: host_config.as_ref().map_or(0, |config| config.bytes),
                    chunk_bytes: host_config.as_ref().map_or(0, |config| config.chunk_bytes) },
            }.components(maximum_pages)?);
        }
        let frame_bytes = if native { cuteafd_transport::expert::native_frame_budget(cuteafd_core::ExpertGeometry::GLM5,
            u32::try_from(args.prefill_rows.max(engine::DECODE_ROWS))?)? } else { 64 << 20 };
        let (decode_link, prefill_link) = production_links(&self.cfg, spark_ranks, prefill,
            if !native && spark_bf16 { cuteafd_core::SparkIngress::Bf16 } else { cuteafd_core::SparkIngress::Fp8K32 },
            cuteafd_core::SparkRings { max_frame_bytes: frame_bytes, depth: 8, slot_bytes: 8 << 20, alignment: 4096 },
            cuteafd_core::spark_reply::requested())?;
        let budgets = production_budgets(&cards, cuteafd_loader::placement::reserve::gpu("glm5"), host,
            serving.map_or(8 << 30, |(prefix, _)| prefix.host_cache_headroom_bytes).max(host.0 / 10),
            &vec![cuteafd_loader::placement::reserve::spark_cap(); spark_ranks])?;
        let request = production_request(*serving_context, requested_pool,
            if args.skip_routed_experts { cuteafd_loader::placement::Onboard::Layers(0) } else { onboard },
            prefill, operation, crate::shared::memory::budget::resident_weight_limit(), cuteafd_core::spark_reply::requested());
        let mut admitted = ProductionInput { expert_usage, cfg: &self.cfg, layers, context: args.max_context as u64,
            maximum_pool_tokens, attention, gpus: devices.len(), prefill,
            blas_head: crate::shared::target_head::TargetHead::configured().blas_head(!native
                && !core.as_ref().is_some_and(|core| core.ranks[0].target_head.is_some())), blas, target,
            draft: draft_facts.clone().with_window_marks(mark_bytes.checked_mul(args.draft_window_marks)
                .context("GLM draft marks overflow")? as u64),
            scratch: [scratch(true)?, scratch(false)?], context_scratch: context,
            exchange_element_bytes: if exchange_f32 { 4 } else { 2 }, peer_diagnostics: physical.peer_diagnostics,
            local: source.as_ref().map(|source| (source, physical)), spark_ranks,
            prefix_host: host_config.is_some(), decode_link, prefill_link,
            skip_routed_experts: args.skip_routed_experts, operation, sync_upload_bytes: 0,
            owners: additional, budgets }.prepare()?.admit(request)?;
        cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "glm5");
        {
            use sha2::{Digest, Sha256};
            let hash = |path: &std::path::Path| -> Result<String> {
                let mut file = std::fs::File::open(path)?;
                let mut digest = Sha256::new();
                std::io::copy(&mut file, &mut digest)?;
                Ok(format!("{:x}", digest.finalize()))
            };
            let programs_hash = hash(manifest_path)?;
            let native_hash = hash(&args.native_lib)?;
            let gpu_classes = inventory.gpus.iter().map(|gpu| format!("{}/{}/{}", gpu.arch,
                gpu.sms, gpu.total_bytes)).collect::<Vec<_>>().join(",");
            let drivers = inventory.gpus.iter().map(|gpu| gpu.driver.as_str()).collect::<Vec<_>>().join(",");
            let layout = format!("snapshot/{}; draft/{:?}; context/{}; C{:?}; host-prefix/{}; replies/{:?}",
                self.snapshot.display(), args.draft, args.max_context, serving.map(|(_, c)| c),
                host_config.as_ref().map_or(0, |config| config.bytes), cuteafd_core::spark_reply::requested());
            if let Some(recommendation) = admitted.census_recommendation(cuteafd_loader::placement::inventory::FigureIdentity {
                declared_graphs: "", programs_sha256: &programs_hash, native_sha256: &native_hash,
                toolchain: env!("CUTEAFD_BUILD_TOOLCHAIN"), gpu_class: &gpu_classes, driver: &drivers, layout: &layout,
            })? { tracing::warn!("{recommendation}"); }
        }
        if let Some(note) = admitted.load_overlap_note() { tracing::info!("{note}"); }
        let placement = admitted.placement.clone();
        let working = &admitted.spark;
        if native {
            use cuteafd_loader::plan::family::Family;
            cuteafd_loader::plan::families::glm::GLM5.exlr8_execution()
                .check(devices.len(), placement.onboard_layers > 0)?;
        }
        let prefill_lanes = usize::try_from(working.prefill_lanes)?;
        let selected_peers = if !working.spark_layers.is_empty() { args.peers.as_deref() } else { None };
        cuteafd_loader::placement::families::GLM5.check(&placement)?;
        tracing::info!(placement = %placement.summary(), "GLM admission");
        let draft_home = match cuteafd_loader::placement::draft::require_home(&placement, "glm5", &[0, 1])? {
            None | Some(0) => DraftHome::Local,
            Some(_) => {
                let (device, stream) = peer_stream.context("admitted remote drafter needs the second GPU")?;
                let rows = match (&ml4_draft, &provisional_drafter) {
                    (Some(layout), _) => layout.capacity().block_rows,
                    (None, Some(draft)) => args.draft_sequences.checked_mul(draft.block).context("GLM draft rows")?,
                    (None, None) => anyhow::bail!("admitted remote drafter without a drafter"),
                };
                tracing::info!(device, rows, "GLM drafter admitted on GPU1 through the shared remote bridge");
                DraftHome::Peer { device, stream, rows, bridge: draft_facts.bridge().to_vec() }
            }
        };
        let tp2_range = placement.tp2.map(|p| p.first..p.first + p.layers);
        let whole_range = (devices.len() == 1).then(|| placement.expert_ranges.first().copied())
            .flatten().filter(|range| range.layers > 0);
        let max_rows = args.prefill_rows.max(engine::DECODE_ROWS);
        let tp2_devices = || -> Result<[crate::shared::memory::device::Device<'_>; 2]> {
            Ok([crate::shared::memory::device::Device { library: &self.library, id: args.device },
                crate::shared::memory::device::Device { library: &self.library, id: split_device.context("TP2 peer")? }])
        };
        let tp2_budgets = || -> Result<[usize; 2]> {
            let budgets = placement.tp2.context("TP2 budget")?.peak_bytes;
            Ok([usize::try_from(budgets[0])?, usize::try_from(budgets[1])?])
        };
        let arenas = arena_layouts.into_iter().zip(&devices).map(|(layout, &id)| {
            let device = crate::shared::memory::device::Device { library: &self.library, id };
            Ok((id, crate::shared::memory::weight_arena::DeviceWeightArena::new(device, layout, cuteafd_core::memory_layout::ArenaTag::weights("glm/target"))?))
        }).collect::<Result<std::collections::BTreeMap<_, _>>>()?;
        let loader = weights::GlmLoader { library: &self.library, catalog: dense_catalog, stream, device: args.device,
            peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
                .collect(), arenas: Some(std::rc::Rc::new(arenas)), allocation_prefix: String::new(), push: None };
        let mut coordinator_push = None;
        let mut native_push = None;
        let native_push_experts = || -> Result<_> {
            let selected = if let Some(range) = &tp2_range {
                Some((range.clone().collect::<Vec<_>>(), tp2_budgets()?.to_vec()))
            } else { whole_range.map(|range| ((range.first..range.first + range.layers).collect(),
                vec![range.peak_bytes as usize])) };
            selected.map(|(layers, budgets)| -> Result<_> {
                let workspace = usize::try_from(local.as_ref().context("native local inventory")?.backend_workspace)?;
                Ok(crate::shared::experts::rtx::push::CoordinatorExperts::Exlr8 {
                    manifest: args.exlr8_manifest.clone().context("EXLR8 manifest")?, layers,
                    budgets: budgets.into_iter().map(|bytes| bytes.saturating_sub(workspace)).collect() })
            }).transpose()
        };
        let dense_devices = devices.iter().map(|&id| crate::shared::memory::device::Device {
            library: &self.library, id }).collect::<Vec<_>>();
        let dense_push_experts = || -> Result<_> {
            if let Some(range) = whole_range {
                let local = local.as_ref().context("whole local inventory")?;
                ensure!(local.format == glm::GlmLocalFormat::Exl3, "whole dense push requires EXL3");
                return Ok(Some(crate::shared::experts::rtx::push::CoordinatorExperts::Whole {
                    owners: vec![crate::shared::experts::rtx::push::WholeOwner { rank: 0,
                        selections: (range.first..range.first + range.layers).map(|layer|
                            crate::shared::experts::layer::ExpertLayer::BackboneFull { layer }).collect(),
                        budget: usize::try_from(range.peak_bytes.checked_sub(local.backend_workspace)
                            .context("whole workspace exceeds admission")?)? }],
                }));
            }
            tp2_range.as_ref().map(|range| -> Result<_> {
                let local = local.as_ref().context("TP2 local inventory")?;
                let workspace = match local.format {
                    glm::GlmLocalFormat::Exl3 => crate::shared::experts::rtx::exl3::Exl3Tp2::workspace_bytes_for(
                        &local.package, self.cfg.hidden, max_rows)?,
                    glm::GlmLocalFormat::Nvfp4 { .. } => crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::plan(
                        self.catalog.fp8().context("NVFP4 tensors")?, &local.package, &range.clone().collect::<Vec<_>>(), 0, max_rows)?.workspace_bytes,
                    glm::GlmLocalFormat::Exlr8 => anyhow::bail!("EXLR8 experts require native dense planning"),
                };
                Ok(crate::shared::experts::rtx::push::CoordinatorExperts::Tp2 {
                    layers: range.clone().collect(),
                    budgets: tp2_budgets()?.map(|b| b.saturating_sub(workspace)).to_vec() })
            }).transpose()
        };
        let mut plan_dense = |sparks: usize| -> Result<_> {
            let sink = std::rc::Rc::new(landing::PushSink::default());
            let planning = weights::GlmLoader { push: Some(sink.clone()), ..loader.clone() };
            let ((embedding, model, drafter), mut records) = crate::shared::experts::rtx::dense::recording(|| {
                let (embedding, model) = crate::shared::token_io::TokenEmbedding::load_with(&self.library,
                    self.embed_source()?, args.token_io.embed_placement, || {
                        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                        planning.model(&self.cfg, layers, owners.as_deref(), native)
                    }, |_, bytes| planning.allocate_owner("embedding", bytes))?;
                let drafter = self.load_drafter(args, &programs, &embedding, stream, &ml4_draft, &mut draft_file, &draft_home, &model.0.head)?;
                Ok((embedding, model, drafter))
            })?;
            let (dense_records, deferred) = sink.take();
            records.extend(dense_records);
            records.extend(media.clone());
            let dense = push::plan_ranks(&self.snapshot, records.clone(), &dense_devices, sparks)?;
            Ok((embedding, model, drafter, records, deferred, dense))
        };
        let admission = if let Some(peers) = selected_peers {
            let peers = peers.split(',').map(str::parse)
                .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let admission = crate::shared::placement::SparkAdmission::plan_view(&self.view,
                self.catalog.routed_experts().geometry()?, Some(&self.catalog), &working,
                peers.len(), spark_model(self.cfg.family_id(), SparkBackend::for_catalog(&self.catalog)?, prefill_lanes, u32::try_from(args.prefill_rows.max(engine::DECODE_ROWS))?, spark_bf16)?,
                args.exlr8_manifest.as_deref())?;
            // Plan coordinator membership before offers; a Required failure
            // refuses startup rather than silently reading through the mount.
            let ranks = if !native && admission.coordinator_pushes() {
                push_policy.plan(|| {
                    let experts = dense_push_experts()?;
                    let planned = plan_dense(peers.len())?;
                    let prepared = crate::shared::experts::rtx::push::CoordinatorPush::prepare_view_admitted(
                        self.catalog.selected_view().context("GLM push selected view")?, Some(&self.catalog), devices.len(), experts,
                        admitted_push_host.clone().context("GLM push host receipt")?)?;
                    Ok((prepared, planned))
                }).context("GLM coordinator dense push planning")?
            } else { None };
            let ranks = ranks.map(|((push, streams, ranks), planned)| {
                coordinator_push = Some((push, peers.len(), planned));
                (streams, ranks)
            });
            // Native dense operands and onboarded EXLR8 halves share each GPU's
            // coordinator rank. Onboarded experts have no post-push mount loader.
            let ranks = match (ranks, &native_records) {
                (None, Some(_)) if admission.coordinator_pushes() => {
                    let planned = push_policy.plan(|| {
                        let experts = native_push_experts()?;
                        crate::shared::experts::rtx::push::CoordinatorPush::prepare_view_admitted(
                            self.catalog.selected_view().context("GLM push selected view")?, Some(&self.catalog), devices.len(), experts,
                            admitted_push_host.clone().context("GLM push host receipt")?)
                    }).context("GLM native coordinator push planning")?;
                    planned.map(|(push, streams, ranks)| {
                        native_push = Some((push, peers.len()));
                        (streams, ranks)
                    })
                }
                (ranks, _) => ranks,
            };
            Some(admission.start_with(&peers, ranks)?)
        } else { None };
        // Dense-only TP1/TP2 ranks push even without onboarded experts or active Sparks.
        if !native && coordinator_push.is_none() {
            coordinator_push = push_policy.plan(|| {
                let experts = dense_push_experts()?;
                let planned = plan_dense(0)?;
                let push = crate::shared::experts::rtx::push::CoordinatorPush::prepare_alone_view_admitted(
                    self.catalog.selected_view().context("GLM push selected view")?, Some(&self.catalog), devices.len(), experts,
                    admitted_push_host.clone().context("GLM push host receipt")?)?;
                Ok((push, 0, planned))
            }).context("GLM coordinator dense push planning")?;
        }
        // All selected routed layers may be onboarded in a partial model.
        // Idle Spark peers acknowledge identity while these destinations push alone.
        if native_records.is_some() && native_push.is_none() {
            native_push = push_policy.plan(|| {
                let push = crate::shared::experts::rtx::push::CoordinatorPush::prepare_alone_view_admitted(
                    self.catalog.selected_view().context("GLM push selected view")?, Some(&self.catalog), devices.len(), native_push_experts()?,
                    admitted_push_host.clone().context("GLM push host receipt")?)?;
                Ok((push, 0))
            }).context("GLM native coordinator push planning")?;
        }
        if coordinator_push.is_none() && native_push.is_none() {
            draft_file = args.draft.as_deref().map(dflash::prefetch);
        }
        let mut planned_drafter = None;
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
                    crate::shared::experts::rtx::exlr8::Exlr8RtxWeights::pushed_pair(devices,
                        &programs, range.collect(), max_rows, budgets, args.exlr8_prefill_activations,
                        pushed.context("onboarded EXLR8 experts require a committed push group")?
                            .into_iter().map(|r| r.exlr8()).collect::<Result<Vec<_>>>()?
                            .try_into().map_err(|_| anyhow::anyhow!("two EXLR8 coordinator GPUs"))?)?),
                glm::GlmLocalFormat::Nvfp4 { .. } => crate::shared::experts::rtx::Tp2Weights::Fp8(match pushed {
                    Some(pushed) => crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::pushed_pair(devices,
                        self.catalog.fp8().context("NVFP4 tensors")?, &local.package,
                        range.collect(), max_rows, budgets,
                        pushed.into_iter().map(|rank| rank.fp8()).collect::<Result<Vec<_>>>()?
                            .try_into().map_err(|_| anyhow::anyhow!("two NVFP4 coordinator GPUs"))?)?,
                    None => crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::load_pair(devices,
                        self.catalog.fp8().context("NVFP4 tensors")?, &local.package,
                        range.collect(), max_rows, budgets)?,
                }),
            })
        };
        let mut whole_weights = None;
        let mut exl3_whole_weights = None;
        let (embedding, ((model, mut shares), tp2)) = crate::shared::placement::load_while_admitting(admission, || {
            if let Some(records) = &native_records {
                let loaded = self.load_native(args, native_load.as_ref().context("native loader admission absent")?, records, &native_dense_records, owners.as_deref(), layers, &programs, &manifest, &loader, &devices, stream,
                    peer_stream, native_push.take(), &media, &ml4_draft, &mut draft_file, &draft_home)?;
                planned_drafter = loaded.3;
                let tp2 = if let Some(range) = tp2_range.clone() {
                    Some(tp2_weights(range, loaded.2)?)
                } else {
                    if let Some(range) = whole_range {
                        let mut pushed = loaded.2.context("whole EXLR8 requires a committed push group")?;
                        ensure!(pushed.len() == 1, "one EXLR8 coordinator GPU required");
                        whole_weights = Some(crate::shared::experts::rtx::exlr8::Exlr8RtxWeights::pushed_rank(
                            dense_devices[0], &programs, (range.first..range.first + range.layers).collect(),
                            max_rows, usize::try_from(range.peak_bytes)?, args.exlr8_prefill_activations,
                            1, 0, pushed.pop().unwrap().exlr8()?)?);
                    }
                    None
                };
                return Ok((loaded.0, (loaded.1, tp2)));
            }
            let (embedding, model, pushed) = if let Some((push, sparks, (embedding, model, drafter, records, deferred, dense))) = coordinator_push.take() {
                // Dense operands and any expert halves commit before either is used.
                let (pushed, ()) = push.receive_with(dense_devices.clone(), Some(&self.catalog), sparks, Some(Ok(dense)), || ());
                let pushed = push_policy.committed(pushed).context("GLM coordinator dense push delivery")?;
                let rank_stream = |device: i32| if device == args.device { stream }
                    else { peer_stream.map_or(stream, |(_, s)| s) };
                for job in &deferred {
                    crate::shared::memory::device::Device { library: &self.library, id: job.device }
                        .run(|| job.run(&self.library, rank_stream(job.device)))?;
                }
                tracing::info!(pushed = pushed.is_some(), operands = records.operands.len(),
                    "GLM coordinator dense operands landed");
                planned_drafter = Some(drafter);
                (embedding, model, pushed)
            } else {
                let (embedding, model) = crate::shared::token_io::TokenEmbedding::load_with(&self.library,
                    self.embed_source()?, args.token_io.embed_placement, || {
                        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                        loader.model(&self.cfg, layers, owners.as_deref(), native)
                    }, |_, bytes| loader.allocate_owner("embedding", bytes))?;
                (embedding, model, None)
            };
            let tp2 = if let Some(range) = tp2_range.clone() {
                Some(tp2_weights(range, pushed)?)
            } else {
                if let Some(range) = whole_range {
                    use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy::CanonicalDecode};
                    let local = local.as_ref().context("whole local inventory")?;
                    let layers = (range.first..range.first + range.layers).collect();
                    let budget = usize::try_from(range.peak_bytes)?;
                    exl3_whole_weights = Some(match pushed {
                        Some(mut pushed) => {
                            ensure!(pushed.len() == 1, "one whole EXL3 coordinator rank required");
                            Exl3Tp2::pushed_whole_weights(dense_devices[0], &self.catalog, &local.package,
                                layers, max_rows, budget, CanonicalDecode, pushed.pop().unwrap().exl3()?)?
                        }
                        None => Exl3Tp2::load_whole_weights(dense_devices[0], &self.catalog, &local.package,
                            layers, max_rows, budget, CanonicalDecode)?,
                    });
                }
                None
            };
            Ok((embedding, (model, tp2)))
        })?;
        let native_owners = native_plans.into_iter().zip(&devices).map(|(plan, &device)| {
            self.library.cuda_set_device(device)?;
            crate::shared::experts::exlr8::NativeExperts::load_admitted(&self.library, plan,
                native_load.as_ref().context("native shared admission absent")?)
        }).collect::<Result<Vec<_>>>()?;
        self.library.cuda_set_device(args.device)?;
        let bytes = model.layers.iter().map(|layer| layer.bytes()).sum::<usize>()
            + model.norm.buffer.bytes + model.head.bytes();
        let embedding_bytes = embedding.device_owner_bytes();
        let actual_weights = std::iter::once((bytes + embedding_bytes) as u64)
            .chain(shares.iter().map(|rank| rank.iter().map(|layer| layer.bytes() as u64).sum()))
            .enumerate().map(|(rank, bytes)| bytes
                + native_owners.get(rank).map_or(0, |owner| owner.resident_bytes() as u64))
            .collect::<Vec<_>>();
        ensure!(actual_weights == reserved_weights,
            "GLM loaded coordinator owners {actual_weights:?} differ from admitted target owners {reserved_weights:?}");
        tracing::info!(layers, elapsed_ms = started.elapsed().as_millis() as u64,
            resident_bytes = ?actual_weights,
            "GLM coordinator weights validated against pre-weight reservation after Spark admission (elapsed includes wait)");
        self.startup_probe(&devices, "glm/weights-before-draft-pool", None, false)?;
        let context_kv = placement.attention_placement == cuteafd_loader::placement::AttentionPlacement::Context;
        ensure!(!context_kv || peer_stream.is_some(), "context attention needs the head split's second GPU");
        cuteafd_bench::context::set_resolved("attention-placement", &placement.attention_summary());
        cuteafd_bench::context::set_rtx_expert_layers(placement.onboard_layers);
        cuteafd_bench::context::set_resolved("pool-tokens", &placement.pool_tokens.to_string());
        let pool_tokens = usize::try_from(placement.pool_tokens)?;
        let max_context = crate::shared::context::pool_context("glm5", *serving_context, pool_tokens)?;
        let pages = pool_tokens.div_ceil(engine::PAGE_ROWS);
        // Pack before allocating the serving pool and expert arenas: the load-phase
        // admission peak is above the same resident target-weight baseline.
        let drafter = match planned_drafter {
            Some(drafter) => drafter,
            None => self.load_drafter(args, &programs, &embedding, stream, &ml4_draft, &mut draft_file, &draft_home, &model.head)?,
        };
        if let Some(drafter) = &drafter { drafter.prepare_workspace()?; }
        self.startup_probe(&devices, "glm/draft-before-kv", None, false)?;
        let depth = crate::shared::prefill_depth::PrefillDepth::new(prefill_lanes, args.prefill_rows)?;
        let mut engine = engine::GlmEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            max_context, args.prefill_rows, pages, embedding, depth, context.filter(|_| context_kv))?;
        engine.initialize_usage(admitted.expert_usage.clone(), args.expert_usage_reader.clone())?;
        engine.max_pool_tokens_no_onboard = placement.max_pool_tokens_no_onboard;
        engine.admitted_host_config = host_config;
        engine.exlr8_prefill_activations = args.exlr8_prefill_activations;
        engine.native_experts = native_owners;
        engine.native_scratch = native_scratch;
        engine.spark_bf16 = spark_bf16;
        engine.full_prefill_logits = args.full_prefill_logits;
        engine.prefill_span_rows = args.prefill_span_rows;
        engine.attention_plan = attention_plan;
        if tp2_range.is_some() && exchange_f32 {
            engine.set_tp2_exchange(crate::shared::experts::rtx::ExchangeDtype::F32);
        }
        if let Some((device, stream)) = peer_stream {
            let mut peer_layers = shares.pop().context("head-split shares")?;
            if let Some(range) = &tp2_range { loader.peer_routers(&self.cfg, &mut peer_layers, range.clone())?; }
            engine.attach_peer(device, stream, peer_layers)?;
        }
        if owners.is_some() { engine.bind_layer_execution(&placement)?; }
        engine.bind_core(core)?;
        if native {
            // The same owned plans selected programs before load and drove landing.
            for (rank, selected) in native_selected.iter().enumerate() {
                let bound = engine.on(rank, || engine::native_dense::NativePrograms::bind(&programs, &manifest, selected))?;
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
        if let Some(weights) = exl3_whole_weights {
            let backend = weights.execution()?;
            ensure!(crate::shared::experts::rtx::RtxExpertLayer::workspace_bytes(&backend) as u64
                == local.as_ref().context("whole local inventory")?.backend_workspace,
                "GLM whole EXL3 backend workspace differs from admission");
            engine.install_whole(Box::new(backend))?;
        }
        if let Some(weights) = whole_weights {
            let backend = weights.execution()?;
            ensure!(crate::shared::experts::rtx::RtxExpertLayer::workspace_bytes(&backend) as u64
                == local.as_ref().context("whole local inventory")?.backend_workspace,
                "GLM whole backend workspace differs from admission");
            engine.install_whole(Box::new(backend))?;
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
        engine.draft_window_marks = if engine.drafter.as_ref().is_some_and(|d| d.mla_windows_only())
            { args.draft_window_marks } else { 0 };
        cuteafd_bench::context::set_resolved("draft-window-marks", &engine.draft_window_marks.to_string());
        engine.bind_draft_core_head()?;
        if let (Some(drafter), Some((device, stream))) = (&engine.drafter, peer_stream) {
            drafter.bind_target(crate::shared::peer_split::RankDevice { device, stream }, &engine.embedding,
                engine.draft_target_head()?)?;
        }
        engine.attention_plan = attention_plan.validate_loaded(engine.drafter.as_ref().map(|d| d.cfg.drafts()))?;
            cuteafd_bench::context::set_resolved("attention-plan-rows", &engine.attention_plan.rows().to_string());
        tracing::info!(plan_rows = engine.attention_plan.rows(),
            active_drafts = ?engine.attention_plan.active_drafts(), "immutable attention reduction plan");
        let ranks = |peers: &str| -> Result<(Vec<std::net::SocketAddr>, cuteafd_transport::expert::SparkWorld)> {
            let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let world = cuteafd_transport::expert::SparkWorld::canonical(peers.len())?;
            Ok((peers, world))
        };
        let config = cuteafd_transport::TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(),
            timeout: std::time::Duration::from_secs(120), max_frame_bytes: if native { cuteafd_transport::expert::native_frame_budget(cuteafd_core::ExpertGeometry::GLM5, u32::try_from(args.prefill_rows.max(engine::DECODE_ROWS))?)? } else { 64 << 20 } };
        crate::shared::memory_report::release_load_staging(&self.library)?;
        self.startup_probe(&devices, "glm/pool-before-transport", None, false)?;
        // Connect every rank and register full-size buffers now: the first
        // request otherwise pays seconds of connection setup.
        let session_shape = |rows: usize| -> Result<cuteafd_transport::SparkSessionShape> {
            let dtype = if engine.spark_bf16 { cuteafd_transport::ExpertV2Dtype::Bf16 }
                else { cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32 };
            let mut shape = cuteafd_transport::SparkSessionShape::new(rows as u32,
                self.cfg.hidden as u32, self.cfg.topk as u32, dtype)?;
            shape.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            Ok(shape)
        };
        let mut transport = selected_peers.map(|peers| -> Result<SparkLink<'_>> {
            let (peers, world) = ranks(peers)?;
            let mut link = SparkLink::prepared(&self.library, world, &peers, config.clone(),
                admitted.decode_link.clone().context("admitted decode transport")?)?;
            // The bootstrap opens every rank before any wave does, so the
            // installed grouping is readable from here on, without executing experts.
            link.bootstrap(&session_shape(engine::DECODE_ROWS)?)?;
            Ok(link)
        }).transpose()?;
        // One prefill transport with `prefill_lanes` waves in flight (PLAN
        // "Prefill contract"): each in-flight unit lands in its own return slot.
        let mut prefill_link = match selected_peers {
            Some(peers) => {
                let (peers, world) = ranks(peers)?;
                let mut link = SparkLink::prepared(&self.library, world, &peers, config.clone(),
                    admitted.prefill_link.clone().context("admitted prefill transport")?)?;
                link.bootstrap(&session_shape(args.prefill_rows)?)?;
                Some(link)
            }
            None => None,
        };
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        if let Some(&layer) = working.spark_layers.first() {
            let started = Instant::now();
            for (link, kind) in transport.iter_mut().map(|link| (link, cuteafd_transport::ExpertV2SourceKind::Decode))
                .chain(prefill_link.iter_mut().map(|link| (link, cuteafd_transport::ExpertV2SourceKind::Prefill))) {
                link.warm_experts(&session_shape(link.capacity() as usize)?, layer, self.cfg.experts, kind)?;
            }
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "Spark expert transports warm");
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
        self.startup_probe(&devices, "glm/transport-before-graphs", None, false)?;
        // Every decode graph shape serving replays, captured now (CUTEAFD_GLM_WARM_GRAPHS=0 skips it).
        if engine.weights.layers.len() == self.cfg.layers
            && std::env::var("CUTEAFD_GLM_WARM_GRAPHS").map_or(true, |v| v != "0")
            && (transport.is_some() || args.skip_routed_experts || engine.has_local()) {
            let captures = engine.warm_decode_graphs(transport.as_mut().map(|t| (t, &runtime)))
                .with_context(|| format!(
                    "GLM startup graph capture (owner glm.graphs rank 0..{}) failed; free device memory \
                     with GPU_MEMORY_CAP or GPU_RESERVE", engine.ranks()))?;
            let declared = admitted.graphs.declared();
            let measured = captures.iter().map(|capture|
                cuteafd_loader::placement::inventory::CapturedGraphRank::measure(capture.rank as u8,
                    &declared[capture.rank], &capture.keys, capture.before, capture.after)
                    .map_err(anyhow::Error::msg)).collect::<Result<Vec<_>>>()?;
            // A successful capture is never refused: an excess over the estimate
            // is reported by the reconcile and serving continues (#1605).
            let reconciled = cuteafd_loader::placement::planner::serving::reconcile_startup_graphs(
                &mut admitted.solved, "glm.graphs", &admitted.graphs.set(),
                admitted.graphs.fingerprints(), &measured).map_err(|error| anyhow::anyhow!(
                    "{error}; the startup graph receipt does not match this plan"))?;
            for (rank, (row, live)) in admitted.graphs.ranks().iter().zip(&reconciled.ranks).enumerate() {
                tracing::info!(rank, executables = live.executables, measured = live.bytes,
                    plan = row.bytes, "GLM startup graph capture reconciled");
            }
            if let Some(note) = admitted.graphs.capture_note(&reconciled) { tracing::warn!("{note}"); }
            cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "glm5 captured");
        }
        engine.check_tp2_routes(transport.as_mut().map(|t| (t, &runtime)))?;
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        self.startup_probe(&devices, "glm/graphs-before-serving", None, false)?;
        engine.usage_active.set(true);
        let result = body(&engine, transport.as_mut(), &runtime);
        // Non-serving commands do not pass schedule's readiness boundary.
        drop(self.startup_audit.borrow_mut().take());
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


/// The drafter's admitted home: beside the target (GPU0), or GPU1 through the
/// shared device-driven bridge with the admitted bridge owners.
pub(crate) enum DraftHome {
    Local,
    Peer { device: i32, stream: *mut c_void, rows: usize, bridge: Vec<cuteafd_loader::placement::Demand> },
}

impl Opened {
    /// The admitted single drafter copy. A GPU1 `home` loads the common executor
    /// on the bridge's own GPU1 stream; GPU0 keeps the embedding and head.
    #[allow(clippy::too_many_arguments)]
    fn load_drafter<'s>(&'s self, args: &EngineArgs, programs: &'s cuteafd_ffi::programs::Programs<'s>,
        embedding: &crate::shared::token_io::TokenEmbedding<'s>, stream: *mut c_void,
        ml4_draft: &Option<cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout>,
        draft_file: &mut Option<std::thread::JoinHandle<std::io::Result<Vec<u8>>>>,
        home: &DraftHome, head: &weights::GlmHead<'_>) -> Result<Option<drafter::GlmDrafter<'s>>> {
        let prepared = match home {
            DraftHome::Local => None,
            DraftHome::Peer { device, stream: peer_stream, rows, .. } => Some(crate::shared::draft::remote::Prepared::new(
                &self.library, crate::shared::peer_split::RankDevice { device: args.device, stream },
                crate::shared::peer_split::RankDevice { device: *device, stream: *peer_stream }, 0, drafter::IDENTITY,
                crate::shared::draft::remote::decode_extent(*rows, self.cfg.hidden, self.cfg.vocab_size, 0)?)?),
        };
        let draft_stream = prepared.as_ref().map_or(stream, |prepared| prepared.stream.raw);
        let draft_device = crate::shared::memory::device::Device { library: &self.library,
            id: prepared.as_ref().map_or(args.device, |prepared| prepared.stream.device.id) };
        let target_projection = dflash::TargetProjection::from_bf16(matches!(head, weights::GlmHead::Bf16(_)));
        let draft_projection = if prepared.is_some() { dflash::TargetProjection::Native } else { target_projection };
        let common = draft_device.own(|| self.load_common_drafter(args, programs, draft_projection,
            embedding, draft_stream, ml4_draft, draft_file))?;
        let Some(common) = common.into_inner() else { return Ok(None) };
        if let DraftHome::Peer { rows, bridge, .. } = home {
            let c = &common.cfg;
            let built = cuteafd_loader::placement::draft::remote_dflash_bridge_demands(c.hidden as u64,
                c.vocab as u64, c.taps.len() as u64, (common.max_batch_sequences() * c.block) as u64, 0)?.demands;
            ensure!(*rows == common.max_batch_sequences() * c.block && built == *bridge,
                "GLM remote draft bridge differs from its admitted owners");
            // Pooled draft pages route through the target's prefix copies; no GLM
            // drafter has them (DFlash2 rings, mtp_dspark windows), so refuse one.
            ensure!(common.mla_draft_page_bytes() == 0, "a remote GLM drafter cannot pool MLA draft pages");
        }
        Ok(Some(match prepared {
            None => drafter::GlmDrafter::Local(common),
            Some(prepared) => drafter::GlmDrafter::Remote(crate::shared::draft::remote_dflash::RemoteDflash::new(
                draft_device.own(|| Ok(common))?, prepared, target_projection)?),
        }))
    }

    fn load_common_drafter<'s>(&'s self, args: &EngineArgs, programs: &'s cuteafd_ffi::programs::Programs<'s>,
        target_projection: dflash::TargetProjection,
        embedding: &crate::shared::token_io::TokenEmbedding<'s>, stream: *mut c_void,
        ml4_draft: &Option<cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout>,
        draft_file: &mut Option<std::thread::JoinHandle<std::io::Result<Vec<u8>>>>)
        -> Result<Option<dflash::DflashDrafter<'s>>> {
        Ok(if let Some(snapshot) = &args.draft {
            let started = Instant::now();
            let cfg = dflash::DflashConfig::read(snapshot)?;
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "the DFlash2 drafter does not fit this target");
            let mask = embedding.host_rows(&[cfg.mask_token])?;
            tracing::info!(mask_source = "target mask_token row", mask_token = cfg.mask_token, "drafter mask source");
            let file = if crate::shared::experts::rtx::dense::is_recording() {
                Vec::new()
            } else {
                draft_file.take().unwrap_or_else(|| dflash::prefetch(snapshot)).join()
                    .map_err(|_| anyhow::anyhow!("drafter prefetch panicked"))??
            };
            let representation = cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation
                ::from_fp8_option(args.draft_fp8);
            let device = crate::shared::memory::device::Device { library: &self.library,
                id: self.library.cuda_get_device()? };
            let arena = crate::shared::memory::weight_arena::DeviceWeightArena::new(device,
                cuteafd_loader::serving_capacity::WeightArenaLayout::new(format!("glm/draft/{}", device.id),
                    cuteafd_loader::families::glm5::draft_physical::gqa_checkpoint_arena_requests(
                        snapshot, representation, 0, 1)?)?,
                cuteafd_core::memory_layout::ArenaTag::drafter_weights("glm/draft"))?;
            let drafter = dflash::DflashDrafter::load_with(&self.library, programs, target_projection, snapshot, file, stream,
                args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences,
                mask, false, representation, args.fp8_scales, crate::shared::fp8_linear::Fp8Rows::W8a16, args.draft_kv_cache,
                |id, name, bytes| {
                    ensure!(id == device.id, "GLM draft arena device differs from load binder");
                    arena.view(name, bytes)
                })?;
            // Priming reads the weights; recorded owners prime only after commit.
            if !crate::shared::experts::rtx::dense::is_recording() { drafter.prepare_workspace()?; }
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64,
                draft_head = "tensor", "DFlash2 drafter resident");
            Some(drafter)
        } else if let Some(layout) = ml4_draft.clone() {
            let started = Instant::now();
            let ml4 = self.cfg.variant.ml4().context("mtp_dspark needs the Mistral Large 4 config")?;
            let params = cuteafd_loader::plan::checkpoint::read_json(&args.snapshot.join("params.json"))?;
            let dspark = cuteafd_loader::families::mistral_large4::dspark::Ml4DsparkConfig::from_params(&params, ml4.layers)?;
            let native = cuteafd_loader::formats::dflash::MlaDflashConfig::ml4(&dspark, ml4)?;
            let catalog = &self.catalog;
            let source = |name: &str, shape: &[usize]| -> Result<crate::shared::experts::rtx::dense::Rect> {
                let tensor = catalog.tensor(name)?;
                let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d)).context("mtp_dspark source extent overflow")?;
                ensure!(tensor.metadata.dtype == cuteafd_core::DType::Bf16 && tensor.metadata.shape == shape
                    && tensor.metadata.byte_length == bytes as u64,
                    "{name}: expected BF16 {shape:?}, found {:?} {:?}", tensor.metadata.dtype, tensor.metadata.shape);
                let rows = if shape.len() >= 2 { shape[0] } else { 1 };
                ensure!(rows > 0 && bytes > 0 && bytes % rows == 0, "{name}: invalid drafter rows");
                Ok(crate::shared::experts::rtx::dense::Rect { name: name.into(), path: catalog.snapshot().join(&tensor.shard),
                    base: tensor.metadata.byte_offset, row_bytes: bytes / rows, rows: 0..rows, cols: 0..bytes / rows })
            };
            let device = crate::shared::memory::device::Device { library: &self.library,
                id: self.library.cuda_get_device()? };
            let arena = crate::shared::memory::weight_arena::DeviceWeightArena::new(device,
                cuteafd_loader::serving_capacity::WeightArenaLayout::new(format!("glm/draft/{}", device.id),
                    layout.immutable_arena_requests())?,
                cuteafd_core::memory_layout::ArenaTag::drafter_weights("glm/draft"))?;
            let drafter = dflash::DflashDrafter::load_ml4_with(&self.library, programs, target_projection, native, ml4,
                ml4.logit_softcap as f32, stream, layout, &source, |id, name, bytes| {
                    ensure!(id == device.id, "ML4 draft arena device differs from load binder");
                    arena.view(name, bytes)
                })?;
            if !crate::shared::experts::rtx::dense::is_recording() { drafter.prepare_workspace()?; }
            tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "mtp_dspark drafter resident");
            Some(drafter)
        } else { None })
    }

    /// Native vocabulary, dense operands and drafter share the existing group.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn load_native<'s>(&'s self, args: &EngineArgs, admitted: &cuteafd_loader::placement::families::glm5::admission::NativeLoad, records: &[cuteafd_loader::plan::exlr8::dense::DenseRecord],
        per_rank: &[cuteafd_loader::plan::exlr8::dense::requests::NativeDenseRecords], owners: Option<&[u8]>, layers: usize, programs: &'s cuteafd_ffi::programs::Programs<'s>, _manifest: &serde_json::Value,
        loader: &weights::GlmLoader<'s>, devices: &[i32], stream: *mut c_void, _peer_stream: Option<(i32, *mut c_void)>,
        push: Option<(crate::shared::experts::rtx::push::CoordinatorPush, usize)>,
        media: &crate::shared::experts::rtx::dense::Records,
        ml4_draft: &Option<cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout>,
        draft_file: &mut Option<std::thread::JoinHandle<std::io::Result<Vec<u8>>>>, draft_home: &DraftHome)
        -> Result<(crate::shared::token_io::TokenEmbedding<'s>, (weights::GlmWeights<'s>, Vec<Vec<weights::GlmLayer<'s>>>),
            Option<Vec<crate::shared::experts::rtx::push::PushedExperts<'s>>>, Option<Option<drafter::GlmDrafter<'s>>>)> {
        use crate::shared::token_io::native::{NativeVocabulary, VocabularyPlan};
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
        let world = devices.len();
        let started = Instant::now();
        self.library.cuda_set_device(devices[0])?;
        ensure!(per_rank.len() == world, "native loader requires one owned plan per coordinator");
        let config = self.view.config()?;
        let selection = cuteafd_loader::plan::exlr8::dense::DenseSelection::from_hf(&config)?;
        let recorder = crate::shared::experts::rtx::dense::Recorder::start(push.is_some())?;
        let vocabulary_plan = VocabularyPlan::new(records, &selection, world, 0, self.cfg.hidden, self.cfg.vocab_size)?;
        let staging_bytes = vocabulary_plan.staging_bytes() as u64;
        admitted.validate_vocabulary(staging_bytes)?;
        let vocabulary = NativeVocabulary::load_prepared_with(&self.library, programs,
            vocabulary_plan, staging_bytes, devices[0], |name, bytes| {
                let (table, suffix) = name.split_once('.').context("native vocabulary plane key")?;
                let suffix = match suffix { "payload" => "fp8", "scales" => "scale",
                    _ => anyhow::bail!("unexpected native vocabulary plane {name}") };
                loader.allocate_owner(&format!("vocabulary.{table}.{suffix}"), bytes)
            })?.context("rank 0 owns the native vocabulary")?;
        let vocabulary = std::rc::Rc::new(vocabulary);
        let source = self.native_embed_source()?;
        let embedding = crate::shared::token_io::TokenEmbedding::row_fp8_with(&self.library, source,
            vocabulary.clone(), |_, bytes| loader.allocate_owner("embedding", bytes))?;
        let mut planned_drafter = None;
        let ((model, shares), pushed) = match push {
            Some((push, sparks)) => {
                let load = crate::shared::experts::rtx::dense::DenseLoad::push(&self.library);
                let model = loader.native_model(&self.cfg, layers, &per_rank, owners, &load, vocabulary.clone())?;
                planned_drafter = Some(self.load_drafter(args, programs, &embedding, stream, ml4_draft, draft_file, draft_home, &model.0.head)?);
                let mut records = recorder.context("native push recorder")?.finish();
                records.extend(load.sink.as_ref().context("push sink")?.take_all());
                records.extend(media.clone());
                let dense_devices: Vec<_> = devices.iter().map(|&id| crate::shared::memory::device::Device {
                    library: &self.library, id }).collect();
                let planned = crate::shared::experts::rtx::dense::plan_records(&self.snapshot, records.clone(),
                    &dense_devices, sparks);
                let (pushed, ()) = push.receive_with(dense_devices, Some(&self.catalog), sparks, Some(planned), || ());
                let pushed = crate::shared::experts::rtx::push::PushPolicy::for_snapshot(&self.snapshot)?
                    .committed(pushed).context("GLM native coordinator push delivery")?;
                tracing::info!(pushed = pushed.is_some(), operands = records.operands.len(), steps = records.steps.len(),
                    "GLM native dense operands landed");
                (model, pushed)
            }
            None => {
                let load = crate::shared::experts::rtx::dense::DenseLoad::mount(&self.library);
                (loader.native_model(&self.cfg, layers, &per_rank, owners, &load, vocabulary.clone())?, None)
            }
        };
        self.library.cuda_set_device(args.device)?;
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "GLM native dense weights resident");
        Ok((embedding, (model, shares), pushed, planned_drafter))
    }

    /// The row-FP8 embedding's identity (vocabulary file, payload offset).
    fn native_embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let records = cuteafd_loader::plan::exlr8::dense::DenseRecord::inventory(&self.snapshot)?;
        let record = records.iter().find(|r| r.tensor == "model.embed_tokens.weight").context("native embedding")?;
        let payload = record.components.iter().find(|c| c.suffix == "fp8").context("native embedding payload")?;
        Ok(crate::shared::token_io::EmbedSource { path: payload.source.path.clone(), offset: payload.source.offset,
            vocab: record.shape[0], hidden: record.shape[1], snapshot: None, tensor_name: record.tensor.clone() })
    }

    /// The checkpoint's own embedding tensor (BF16 [vocab, hidden]).
    fn embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let catalog = &self.catalog;
        let tensor = catalog.tensor(self.cfg.embedding())?;
        crate::shared::token_io::EmbedSource::new(catalog.snapshot(), &tensor.shard, &tensor.metadata,
            self.cfg.hidden)
    }
}

fn dump_golden_layer(dir: &std::path::Path, layer: usize, stream: &[u8]) -> Result<()> {
    ensure!(stream.len() % 2 == 0, "golden layer dump must contain complete BF16 elements");
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("layer{layer:02}.bin")), stream)?;
    Ok(())
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

fn write_attention_identity(dir: &std::path::Path, view: &cuteafd_loader::plan::checkpoint::CheckpointView) -> Result<()> {
    std::fs::write(dir.join("attention.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "mode": view.mode(), "bundle_fingerprint": view.fingerprint(),
        "config": view.config_name(), "index": view.index_name(),
        "fp8_recipe": (view.mode() == cuteafd_loader::plan::checkpoint::AttentionMode::Fp8Compact)
            .then_some(cuteafd_loader::plan::checkpoint::view::FP8_RECIPE),
    }))?)?;
    Ok(())
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
        write_attention_identity(dir, &opened.view)?;
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
            || engine.local_layers().is_some_and(|layers| layers.contains(&i))),
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

fn bench_verify_draft_context(start: usize, sequences: usize,
    mut reset: impl FnMut(usize, usize) -> Result<()>,
    mut update: impl FnMut(&[dflash::ContextRow]) -> Result<()>) -> Result<usize> {
    let rows = start.min(dflash::TAP_ROWS);
    let origin = start - rows;
    for slot in 0..sequences {
        reset(slot, origin)?;
        if rows > 0 {
            update(&(0..rows).map(|tap_row| dflash::ContextRow {
                tap_row, slot, position: origin + tap_row,
            }).collect::<Vec<_>>())?;
        }
    }
    Ok(origin)
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
        println!("drafter step (anchors at position {start}, median of 7):");
        let max_batch = dflash::ReplayDrafter::sequences(drafter);
        let mut counts: Vec<usize> = (0..).map(|i| 1usize << i).take_while(|&s| s < max_batch)
            .chain([max_batch]).collect();
        if !args.bench_sequences.is_empty() {
            ensure!(args.bench_sequences.iter().all(|&s| s >= 1 && s <= max_batch), "--bench-sequences past the draft batch capacity");
            counts = args.bench_sequences.clone();
        }
        for sequences in counts {
            // Startup retires MLA bindings; each batch reads the same retained tap tail.
            let origin = bench_verify_draft_context(start, sequences,
                |slot, origin| drafter.reset_context(slot, origin), |rows| drafter.update(rows))?;
            let seqs: Vec<dflash::DraftSeq> = (0..sequences)
                .map(|slot| dflash::DraftSeq { slot, anchor: tokens[start], position: start, valid_from: origin }).collect();
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
        ensure!(transport.is_some() || engine.skip_routed() || engine.has_local(),
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
        Some(&mut |layer, stream| {
            if let Some(dir) = &args.layer_dump { dump_golden_layer(dir, layer, stream)?; }
            compare(layer, 0, stream, &mut worst)
        }))?;
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
            write_attention_identity(dir, &opened.view)?;
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
        record_selector_rank(&mut first_rank, &draft.features);
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

fn record_selector_rank(histogram: &mut [usize; 16], features: &[[f32; 4]]) {
    if let Some(rank) = features.first().and_then(|f| histogram.get_mut(f[3] as usize)) { *rank += 1; }
}

/// Runs the drafter alone on the golden taps at reference.py's anchor
/// positions and compares tokens, selector features and final-norm rows.
fn draft_oracle(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmEngine<'_>, dir: &std::path::Path) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-oracle needs --draft")?;
    anyhow::ensure!(!drafter.token_only(), "--draft-oracle requires a selector-feature drafter; bundled dSpark has no selector oracle");
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
