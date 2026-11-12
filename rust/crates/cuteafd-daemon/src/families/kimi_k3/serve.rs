//! Native text serving through the shared scheduler, sampler and exact prefix cache.
use super::{backend::KimiBackend, startup};
use crate::shared::{prefix::PrefixArgs, qwen::scheduler::{self, Backend}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::kimi::KimiEncoding, ModelEncoding, ModelProfile, NativeLimits, NativeRequest};
use std::{net::SocketAddr, path::PathBuf, sync::{Arc, Mutex}};
use cuteafd_loader::families::kimi_k3::COMPILED_CONTEXT;

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub(crate) attention: crate::cli::CoreViewArgs,
    /// Target and drafter share the exact selected MLA record.
    #[arg(long, default_value = "fp8")] pub kv_cache: cuteafd_core::MlaCacheFormat,
    #[arg(long)] pub snapshot: PathBuf,
    /// Aggregate executed routed-expert selections on each canonical producer GPU.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set,
        num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    pub expert_usage: bool,
    #[arg(skip)] pub expert_usage_reader: cuteafd_core::expert_usage::Reader,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)] pub device: i32,
    /// Optional distinct physical peer; absent serves the full model on --device.
    #[arg(long)] pub split_device: Option<i32>,
    /// MLA auto uses context on two devices, whole ownership on one; KDA stays head-split on a pair.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto",
        value_parser = ["auto", "heads", "context", "layers"])]
    pub attention_placement: String,
    #[arg(long, value_delimiter = ',')] pub peers: Vec<SocketAddr>,
    /// Auto admission grows beyond the selected context floor while memory allows.
    #[arg(long)] pub pool_tokens: Option<u64>,
    /// Served context: auto fits the model maximum and admitted pool; a number pins it.
    #[arg(long = "max-context-tokens", value_parser = crate::shared::context::parse_tokens::<usize>)]
    pub max_context: Option<usize>,
    /// RTX expert layers: auto, all (every routed-expert layer of the target
    /// model), or an exact N.
    #[arg(long, default_value = "auto")] pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    #[arg(long, default_value = "0.0.0.0:8000")] pub listen: String,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub max_sequences: u8,
    #[arg(long)] pub model_id: Option<String>,
    /// Opt in to a checkpoint-native MLA DFlash2 drafter.
    #[arg(long)] pub draft: Option<PathBuf>,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)] pub draft_fp8: bool,
    /// Split the native draft cache and attention over both physical GPUs.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)] pub tp2_dflash: bool,
    /// Fixed draft depth; absent lets the shared policy choose up to the checkpoint width.
    #[command(flatten)]
    pub draft_length: crate::shared::draft::length::DraftLength,
    /// Prefill unit rows, at most 4096; unset follows the shared wave policy.
    /// The lane holds the exact shared shape (8256 rows for three waves).
    #[arg(long)] pub prefill_rows: Option<usize>,
    /// Prefill units in flight on one transport; unset follows the shared six-Spark policy.
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub prefill_waves: Option<u8>,
    /// Admit every prefill row's logits for diagnostic fidelity scoring.
    #[arg(long)] pub full_prefill_logits: bool,
    /// Pack already queued prompts, or run unknown arithmetic routes alone in exact mode.
    #[arg(long, value_enum, default_value = "unrestricted")]
    pub prefill_packing: crate::shared::prefill_packing::PackingMode,
    #[command(flatten)] pub prefix: PrefixArgs,
    #[command(flatten)] pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)] pub loop_abort: crate::shared::loop_abort::LoopAbortArgs,
    #[command(flatten)] pub token_io: crate::shared::token_io::TokenIoArgs,
    #[command(flatten)] pub api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    /// `admitted` is the engine's resolved context extent (the explicit
    /// --max-context-tokens, else the admitted automatic extent).
    fn api_limits(&self, admitted: usize) -> Result<NativeLimits> {
        self.prefix.reserve.limits(self.max_context.filter(|&tokens| tokens > 0).unwrap_or(admitted).min(admitted))
    }

    pub(super) fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        let plan = prefill_shape(self.peers.len(), cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::kimi::KIMI_K3), PrefillOverrides {
            unit_rows: self.prefill_rows.map(|rows| rows as u64),
            waves: self.prefill_waves.map(u64::from), span_rows: None, reserve_rows: None,
        })?;
        cuteafd_bench::context::ResolvedSettings::prefill(plan).publish();
        Ok(plan)
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.tp2_dflash || (self.draft.is_some() && self.split_device.is_some()),
            "Kimi TP2 DFlash requires a drafter and two coordinator devices");
        ensure!(self.device >= 0 && self.split_device.is_none_or(|peer| peer >= 0 && peer != self.device),
            "Kimi serving requires one device or two distinct coordinator devices");
        ensure!(self.peers.len() <= 8 && self.peers.iter().collect::<std::collections::HashSet<_>>().len() == self.peers.len(),
            "Kimi serving accepts at most eight distinct Spark endpoints");
        ensure!(self.max_context.is_none_or(|context| context == 0 || (1..=COMPILED_CONTEXT).contains(&context)),
            "Kimi context exceeds the compiled 1M extent");
        self.prefill()?;
        ensure!(self.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Kimi serving requires exact prefix frontiers");
        ensure!(matches!(self.token_io.embed_placement, crate::shared::token_io::EmbedPlacement::Gpu),
            "Kimi serving admits GPU token embeddings only");
        Ok(())
    }
}

fn load_target<'a, const RANKS: usize>(args: &ServeArgs,
    library: &'a Arc<cuteafd_ffi::NativeLibrary>, programs: &'a cuteafd_ffi::programs::Programs<'a>,
    checkpoint: &cuteafd_loader::plan::Checkpoint,
    cfg: &cuteafd_loader::families::kimi_k3::config::KimiK3Config, devices: [i32; RANKS],
    slots: usize, marks: usize, prefill: cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan)
    -> Result<startup::Loaded<'a>> {
    let draft = args.draft.as_ref().map(|snapshot| super::draft::Options {
        snapshot: snapshot.clone(), fp8: args.draft_fp8, tp2_dflash: args.tp2_dflash, cache_format: args.kv_cache });
    startup::load(library, programs, checkpoint, cfg, cfg, startup::Options { role: startup::Role::Serve,
        expert_usage: args.expert_usage, expert_usage_reader: args.expert_usage_reader.clone(),
        cache_format: args.kv_cache, devices, native_lib: &args.native_lib, manifest: &args.manifest,
        attention: cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?,
        peers: args.peers.clone(), rtx_expert_layers: args.rtx_expert_layers,
        rows: usize::try_from(prefill.span_rows)?, waves: usize::try_from(prefill.waves)?,
        prefill_rows: Some(usize::try_from(prefill.unit_rows)?), slots, marks,
        max_context: args.max_context.filter(|&tokens| tokens > 0).unwrap_or(cfg.max_position),
        context: cuteafd_loader::placement::planner::Context::resolve(cfg.max_position as u64, args.max_context.map(|n| n as u64)),
        pool_tokens: args.pool_tokens.filter(|&tokens| tokens > 0), graphs: true,
        full_prefill_logits: args.full_prefill_logits,
        ffn_mode: cuteafd_ffi::KimiFfnMode::default(), hosted_layers: None, draft, verify_rows: 0, prefix: Some(&args.prefix),
    })
}

pub(crate) async fn run(mut args: ServeArgs) -> Result<()> {
    args.draft_length.report_inactive(args.draft.is_some());
    cuteafd_core::concurrency::validate("kimi_k3", args.max_sequences as u64)?;
    args.prefix.resolve_entries(args.max_sequences as usize);
    args.validate()?;
    let api = args.api.load()?;
    let encoding = Arc::new(KimiEncoding::from_snapshot(&args.snapshot)?);
    let eos = encoding.eos_token_ids();
    let model = args.model_id.clone().or_else(|| args.snapshot.ancestors().find_map(|path| {
        let name = path.file_name()?.to_str()?.strip_prefix("models--")?;
        let (organization, model) = name.split_once("--")?;
        Some(format!("{organization}/{model}"))
    })).context("Kimi serving requires --model-id for a snapshot without HF identity")?;
    let mut profile = ModelProfile::new(model, ModelEncoding::Kimi(encoding));
    profile.expert_usage = args.expert_usage_reader.clone();
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(cuteafd_core::concurrency::queue_depth(args.max_sequences as u64) as usize);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let worker_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| -> Result<()> {
            cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
                .map_err(|_| anyhow::anyhow!("Kimi serving needs an isolated process"))?;
            let view = worker_args.attention.selected_view.clone().map(Ok)
                .unwrap_or_else(|| worker_args.attention.select(&worker_args.snapshot))?;
            worker_args.attention.validate_view(&view)?;
            let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator_view(&view, false, false)?;
            let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(&checkpoint.config)?;
            cfg.release_geometry()?;
            cuteafd_bench::context::set_family(cfg.family_id());
            let view = checkpoint.view.as_ref()
                .context("Kimi serving requires the immutable selected checkpoint view")?;
            use cuteafd_loader::plan::checkpoint::{view::FP8_RECIPE, CoreView};
            let mode = match view.mode() {
                CoreView::Original => "original", CoreView::K6Compact => "k6-compact", CoreView::K8Compact => "k8-compact",
                CoreView::Fp8Compact => "fp8-compact",
            };
            let recipe = if view.mode() == CoreView::Fp8Compact { FP8_RECIPE } else { "none" };
            cuteafd_bench::context::set_resolved("attention-mode", mode);
            cuteafd_bench::context::set_resolved("attention-bundle-fingerprint", view.fingerprint());
            cuteafd_bench::context::set_resolved("attention-fp8-recipe", recipe);
            tracing::info!(core_view = mode, bundle = view.fingerprint(), fp8_recipe = recipe,
                "selected immutable attention bundle");
            let concurrency = usize::from(worker_args.max_sequences);
            let slots = cuteafd_core::concurrency::state_slots(concurrency as u64) as usize;
            let prefill = worker_args.prefill()?;
            let physical_ranks = 1 + usize::from(worker_args.split_device.is_some());
            let mark_bytes = super::state::KdaLayout::for_ranks(&cfg, slots, 1, 1, physical_ranks)?.mark_bytes()
                .checked_mul(physical_ranks).context("Kimi prefix mark extent overflow")?;
            let marks = worker_args.prefix.mark_rule(concurrency).slots(mark_bytes);
            startup::preflight(&worker_args.manifest, &worker_args.native_lib)?;
            // SAFETY: every native owner drops on this worker before the library.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;
            let loaded = match worker_args.split_device {
                Some(peer) => load_target(&worker_args, &library, &programs, &checkpoint, &cfg,
                    [worker_args.device, peer], slots, marks, prefill)?,
                None => load_target(&worker_args, &library, &programs, &checkpoint, &cfg,
                    [worker_args.device], slots, marks, prefill)?,
            };
            let physical_devices = std::iter::once(worker_args.device).chain(worker_args.split_device).collect::<Vec<_>>();
            crate::shared::memory_report::release_load_staging(&library)?;
            loaded.engine.configure_packing(worker_args.prefill_packing)?;
            let backend = KimiBackend::new(&loaded.engine)?;
            cuteafd_bench::context::set_resolved("prefill-scoring", "streamed");
            // The shared selector allocates on the calling thread's device and
            // consumes logits on the terminal rank's exact producing stream.
            library.cuda_set_device(worker_args.split_device.unwrap_or(worker_args.device))?;
            scheduler::schedule(&backend, &worker_args.snapshot, &mut receive, &worker_stats, concurrency,
                match loaded.engine.native_draft() {
                    Some(draft) => scheduler::Drafts::Neural {
                        depth: draft.drafts() },
                    None => scheduler::Drafts::None,
                }, worker_args.draft_length, eos, worker_args.decode_share, &worker_args.prefix,
                worker_args.loop_abort.config(), worker_args.token_io.token_select,
                &mut super::super::qwen3_5::serve::text_media(), None, || {
                    loaded.engine.require_live()?;
                    crate::shared::inventory::RuntimeInventory::measure(&library,
                        &physical_devices,
                        crate::shared::inventory::InventoryPhase::PostLoad, |_| Ok(()))?
                        .record("kimi_k3-serving-loaded", &loaded.admission_free);
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok(backend.geometry().max_context));
                    }
                    Ok(())
                })
        })();
        crate::shared::serve::forward_initialization(result, ready.is_some(), "Kimi initialization failed", |error| {
            ready.take().expect("pending readiness").send(Err(error)).map_err(|result| match result { Err(error) => error, Ok(_) => unreachable!("sent an error") })
        })
    });
    let compiled_context = ready_rx.await.context("Kimi worker exited before readiness")??;
    let limits = args.api_limits(compiled_context)?;
    tracing::info!(api_max_context = limits.context(), compiled_max_context = compiled_context,
        "Kimi API and compiled context limits");
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new(
        "kimi_k3", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    let mut worker = Some(worker);
    tokio::select! {
        served = crate::shared::api::serve_http(listener, api.app(router, hub), "kimi_k3") => served?,
        _ = crate::shared::api::watch_scheduler(&mut worker) => unreachable!(),
    }
    crate::shared::api::finish_scheduler(&mut worker, "kimi_k3").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn args(extra: &[&str]) -> Result<ServeArgs, clap::Error> {
        let mut argv = vec!["cuteafd", "serve-kimi", "--snapshot", "/checkpoint", "--native-lib", "/native.so"];
        argv.extend_from_slice(extra);
        let cli = crate::cli::Cli::try_parse_from(argv)?;
        let crate::cli::Commands::ServeKimi(args) = cli.command else { panic!("Kimi command") };
        Ok(args)
    }

    #[test]
    fn expert_usage_off_and_skip_have_no_reader_or_native_owners() -> Result<()> {
        assert!(args(&[])?.expert_usage);
        let off = args(&["--expert-usage=false"])?;
        assert!(!off.expert_usage);
        assert_eq!(off.expert_usage_reader.snapshot()["enabled"], false);
        assert!(args(&["--expert-usage"])?.expert_usage);
        assert!(!off.expert_usage_reader.configure(true, true).map_err(anyhow::Error::msg)?);
        assert!(off.expert_usage_reader.require_enabled().is_err());
        Ok(())
    }

    #[test]
    fn core_view_flags_are_explicit_and_reject_removed_flags() -> Result<()> {
        use cuteafd_loader::plan::checkpoint::CoreView;
        assert_eq!(args(&[])?.attention.mode(), CoreView::Original);
        assert_eq!(args(&["--core-view", "k6-compact"])?.attention.mode(), CoreView::K6Compact);
        assert_eq!(args(&["--core-view", "k8-compact"])?.attention.mode(), CoreView::K8Compact);
        assert_eq!(args(&["--core-view", "fp8-compact"])?.attention.mode(), CoreView::Fp8Compact);
        for flag in ["--compact-attention", "--fp8-compact-attention"] { assert!(args(&[flag]).is_err()); }
        Ok(())
    }

    #[test]
    fn attention_selector_defaults_auto_and_preserves_explicit_modes() -> Result<()> {
        assert_eq!(args(&[])?.attention_placement, "auto");
        for value in ["auto", "heads", "context", "layers"] {
            assert_eq!(args(&["--attention-placement", value])?.attention_placement, value);
        }
        assert!(args(&["--attention-placement", "invalid"]).is_err());
        Ok(())
    }

    #[test]
    fn native_draft_is_opt_in_with_explicit_precision_and_fixed_length() -> Result<()> {
        let defaults = args(&[])?;
        assert!(defaults.draft.is_none());
        assert!(!defaults.draft_fp8);
        assert!(!defaults.tp2_dflash);
        assert!(defaults.draft_length.draft_length.is_none());
        for fp8 in ["false", "true"] {
            let selected = args(&["--draft", "/drafter", "--draft-fp8", fp8, "--draft-length", "7"])?;
            assert_eq!(selected.draft.as_deref(), Some(std::path::Path::new("/drafter")));
            assert_eq!(selected.draft_fp8, fp8 == "true");
            assert_eq!(selected.draft_length.draft_length, Some(7));
        }
        for enabled in ["false", "true"] {
            assert_eq!(args(&["--tp2-dflash", enabled])?.tp2_dflash, enabled == "true");
        }
        assert!(args(&["--tp2-dflash"]).is_err());
        assert!(args(&["--draft-fp8"]).is_err());
        assert!(args(&["--draft-length", "1"]).is_ok());
        for depth in ["0", "invalid"] {
            assert!(args(&["--draft", "/drafter", "--draft-length", depth]).is_err());
        }
        Ok(())
    }

    #[test]
    fn context_draft_validation_requires_drafter_and_physical_peer() -> Result<()> {
        let base = ["--tp2-dflash", "true", "--peers", "127.0.0.1:8001"];
        assert!(args(&base)?.validate().is_err());
        let mut selected = base.to_vec();
        selected.extend(["--draft", "/drafter"]);
        assert!(args(&selected)?.validate().is_err());
        selected.extend(["--split-device", "1"]);
        args(&selected)?.validate()?;
        Ok(())
    }

    #[test]
    fn full_prefill_logits_is_explicit_and_preserves_shape() -> Result<()> {
        let ordinary = args(&[])?;
        let scoring = args(&["--full-prefill-logits"])?;
        assert!(!ordinary.full_prefill_logits);
        assert!(scoring.full_prefill_logits);
        assert_eq!(ordinary.prefill()?, scoring.prefill()?);
        Ok(())
    }

    #[test]
    fn queued_packing_defaults_unrestricted_and_accepts_exact() -> Result<()> {
        use crate::shared::prefill_packing::PackingMode;
        assert_eq!(args(&[])?.prefill_packing, PackingMode::Unrestricted);
        assert_eq!(args(&["--prefill-packing", "exact"])?.prefill_packing, PackingMode::Exact);
        assert!(args(&["--prefill-packing", "invalid"]).is_err());
        Ok(())
    }

    #[test]
    fn shared_prefill_policy_matches_planner_and_runtime_admission() -> Result<()> {
        use cuteafd_loader::{families::kimi_k3::allocation, plan::{self, layout::LayoutOptions,
            testing, ExpertPlacement, PlanOptions}};
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let plan_input = cuteafd_loader::plan::testing::BudgetedPlanInput::open(snapshot.path())?;
        let peers = "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4,127.0.0.1:5,127.0.0.1:6";
        for (extra, expected) in [
            // Unpinned: the fixed three units of the reserve (TJ 2026-10-28).
            (vec![], (2752, 3, 8256)),
            // Units are the 8192-row reserve divided by k.
            (vec!["--prefill-waves", "2"], (4096, 2, 8192)),
            (vec!["--prefill-rows", "4096", "--prefill-waves", "2"], (4096, 2, 8192)),
            (vec!["--prefill-waves", "3"], (2752, 3, 8256)),
            (vec!["--prefill-waves", "4"], (2048, 4, 8192)),
            (vec!["--prefill-rows", "1024", "--prefill-waves", "2"], (1024, 2, 2048)),
        ] {
            let mut argv = vec!["--peers", peers];
            argv.extend(extra);
            let args = args(&argv)?;
            args.validate()?;
            let shape = args.prefill()?;
            assert_eq!((shape.unit_rows, shape.waves, shape.span_rows), expected);
            assert_eq!(allocation::prefill_shape_with(shape.span_rows as usize, shape.waves as usize,
                Some(shape.unit_rows as usize))?, cuteafd_loader::serving_capacity::PrefillShape {
                    unit_rows: shape.unit_rows, waves: shape.waves });
            let options = PlanOptions { placement: ExpertPlacement::Sparks { ranks: 6 },
                layout: Some(LayoutOptions { rtx_bytes: vec![96 << 30; 2], host_memory: Some((128 << 30, 128 << 30)),
                    host_cache_bytes: Some(0), prefill_rows: args.prefill_rows.unwrap_or(0) as u64,
                    prefill_lanes: args.prefill_waves.map_or(0, u64::from), ..Default::default() }),
                ..Default::default() };
            let report = plan_input.plan(&options)?;
            assert_eq!((report.prefill_unit_rows, report.prefill_waves, report.prefill_span_rows),
                (Some(shape.unit_rows), Some(shape.waves), Some(shape.span_rows)));
        }
        for extra in [vec!["--prefill-rows", "0"], vec!["--prefill-rows", "2753"],
            vec!["--prefill-rows", "4097"], vec!["--prefill-rows", "4096", "--prefill-waves", "4"]] {
            let mut argv = vec!["--peers", peers];
            argv.extend(extra);
            assert!(args(&argv)?.validate().is_err());
        }
        assert!(args(&["--prefill-lanes", "3"]).is_err(), "removed alias stays removed");
        assert!(args(&["--kv-cache", "int4"]).is_err(), "Kimi MLA supports only BF16/FP8 records");
        Ok(())
    }

    #[test]
    fn production_defaults_and_preflight_reject_unsupported_geometry() -> Result<()> {
        let peers = "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4,127.0.0.1:5,127.0.0.1:6";
        let defaults = args(&["--peers", peers])?;
        defaults.validate()?;
        // Automatic context is independent of core weight precision.
        assert_eq!(defaults.max_context, None);
        assert_eq!(args(&["--peers", peers, "--max-context-tokens", "auto"])?.max_context, Some(0));
        assert_eq!(args(&["--peers", peers, "--max-context-tokens", "524288"])?.max_context, Some(524_288));
        assert_eq!(defaults.kv_cache, cuteafd_core::MlaCacheFormat::Fp8);
        assert_eq!(cuteafd_loader::plan::layout::LayoutOptions::default().mla_kv, defaults.kv_cache);
        assert_eq!(args(&["--peers", peers, "--kv-cache", "bf16"])?.kv_cache, cuteafd_core::MlaCacheFormat::Bf16);
        assert_eq!(args(&["--peers", peers, "--kv-cache", "fp8"])?.kv_cache, cuteafd_core::MlaCacheFormat::Fp8);
        assert!(args(&["--peers", peers, "--kv-cache", "int8"]).is_err());
        assert_eq!(defaults.rtx_expert_layers, cuteafd_loader::placement::OnboardSetting::Auto);
        assert_eq!(defaults.max_sequences, 16);
        assert!(args(&["--peers", peers, "--native-k6"]).is_err());
        assert_eq!(defaults.pool_tokens, None);
        assert_eq!(defaults.api_limits(524_288)?.context(), 524_288);
        assert_eq!(defaults.api_limits(COMPILED_CONTEXT)?.context(), COMPILED_CONTEXT as u32);
        args(&["--peers", peers, "--max-context-tokens", "1048576"])?.validate()?;
        let limited = args(&["--peers", peers, "--max-context-tokens", "262144"])?;
        limited.validate()?;
        let limits = limited.api_limits(COMPILED_CONTEXT)?;
        assert_eq!(limits.context(), 262144);
        assert_eq!(limits.output_for_prompt(262143, 32768)?, 1);
        assert!(limits.output_for_prompt(262144, 1).is_err());
        assert_eq!(args(&["--peers", peers, "--rtx-expert-layers", "3"])?.rtx_expert_layers,
            cuteafd_loader::placement::OnboardSetting::Layers(3));
        assert_eq!(args(&["--peers", peers, "--rtx-expert-layers", "all"])?.rtx_expert_layers,
            cuteafd_loader::placement::OnboardSetting::All);
        args(&["--peers", peers, "--rtx-expert-layers", "auto"])?.validate()?;
        // The Spark cap is the global --spark-memory-cap, not a serve-kimi flag.
        assert!(args(&["--peers", peers, "--spark-budget-gib", "108"]).is_err());
        args(&[])?.validate()?;
        for extra in [["--split-device", "0"], ["--max-context-tokens", "1048577"],
            ["--embedding-placement", "host"], ["--prefix-partial", "on"]] {
            assert!(args(&["--peers", peers, extra[0], extra[1]]).is_err()
                || args(&["--peers", peers, extra[0], extra[1]])?.validate().is_err());
        }
        assert!(!defaults.prefix.host_cache_bytes.enabled(), "host retention remains opt-in");
        for budget in ["0", "auto", "1", "32GiB"] {
            args(&["--peers", peers, "--host-cache-bytes", budget])?.validate()?;
        }
        assert!(args(&["--max-sequences", "129"]).is_err());
        assert!(cuteafd_core::concurrency::validate("kimi_k3", 65).is_err());
        assert!(args(&["--speculator", "mtp"]).is_err());
        Ok(())
    }
}
