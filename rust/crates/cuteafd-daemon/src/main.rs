use anyhow::{Context, Result};
use clap::{CommandFactory, FromArgMatches};
use serde::Serialize;
use std::io::IsTerminal;
use std::process::Command;

// Optional allocator instrumentation (2026-10-26): `--features jemalloc`
// replaces glibc so the coordinator's host heap can be measured with jemalloc's
// stats (live vs active vs mapped).  The default build keeps glibc + malloc_tune.
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod cli;
mod commands;
mod families;
mod shared;
use cli::{Cli, Commands};
use commands::bench_rdma::run_bench_rdma;
use commands::bench_rdma_ring::run_bench_rdma_ring;
use commands::doctor::run_doctor;
use commands::plan::run_plan;
use commands::transport_capabilities::run_transport_capabilities;

#[derive(Debug, Serialize)]
pub(crate) struct Probe {
    pub(crate) ok: bool,
    pub(crate) output: String,
}

fn resolve_vision(
    family: Option<cuteafd_loader::plan::MediaMode>,
    initial: Option<cuteafd_loader::plan::MediaMode>,
) -> cuteafd_loader::plan::MediaMode {
    family.or(initial).unwrap_or(cuteafd_loader::plan::MediaMode::Auto)
}

fn resolve_qwen_vision(
    family: Option<cuteafd_loader::plan::MediaMode>,
    initial: Option<cuteafd_loader::plan::MediaMode>,
) -> cuteafd_loader::plan::MediaMode {
    family.or(initial).unwrap_or(cuteafd_loader::plan::MediaMode::Auto)
}

/// Borrow each media-capable command's options once; resolution and assignment
/// must not diverge between the global API policy and the serving owner.
struct ServeMedia<'a> {
    snapshot: &'a std::path::Path,
    vision: &'a mut cuteafd_loader::plan::MediaMode,
    audio: Option<&'a mut cuteafd_loader::plan::MediaMode>,
}

fn serve_media(command: &mut Commands) -> Option<ServeMedia<'_>> {
    match command {
        Commands::ServeInkling2(args) => Some(ServeMedia { snapshot: &args.snapshot,
            vision: &mut args.vision.vision, audio: None }),
        Commands::ServeStep5(args) => Some(ServeMedia { snapshot: &args.snapshot,
            vision: &mut args.vision.vision, audio: None }),
        Commands::ServeMimo(args) => Some(ServeMedia { snapshot: &args.engine.snapshot,
            vision: &mut args.vision, audio: Some(&mut args.audio) }),
        Commands::ServeKimi(args) => Some(ServeMedia { snapshot: &args.snapshot,
            vision: &mut args.media.vision, audio: None }),
        Commands::ServeMinimax(args) => Some(ServeMedia { snapshot: &args.snapshot,
            vision: &mut args.media.vision, audio: None }),
        Commands::ServeMuse(args) => Some(ServeMedia { snapshot: &args.snapshot,
            vision: &mut args.vision, audio: Some(&mut args.audio) }),
        Commands::ServeQwen4(args) => Some(ServeMedia { snapshot: &args.engine.snapshot,
            vision: &mut args.vision, audio: None }),
        Commands::ServeGlm(args) => Some(ServeMedia { snapshot: &args.engine.snapshot,
            vision: &mut args.vision, audio: None }),
        Commands::ServeGlmf(args) => Some(ServeMedia { snapshot: &args.engine.snapshot,
            vision: &mut args.vision, audio: None }),
        _ => None,
    }
}

/// Every qualified serve/golden entry stores the bundle before media or engine setup.
fn command_attention(command: &mut Commands) -> Option<(&std::path::Path, &mut cli::CoreViewArgs)> {
    match command {
        Commands::ServeGlmf(args) => Some((&args.engine.snapshot, &mut args.engine.attention)),
        Commands::GlmfGolden(args) => Some((&args.engine.snapshot, &mut args.engine.attention)),
        Commands::ServeGlm(args) => Some((&args.engine.snapshot, &mut args.engine.attention)),
        Commands::GlmGolden(args) => Some((&args.engine.snapshot, &mut args.engine.attention)),
        Commands::ServeKimi(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::KimiGolden(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::MinimaxGolden(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::Inkling2Golden(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::Hy4fGolden(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::ServeHy4f(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::ServeInkling2(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::ServeMinimax(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::ServeStep5(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::Step5Golden(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::ServeMuse(args) => Some((&args.snapshot, &mut args.attention)),
        Commands::MuseGolden(args) => Some((&args.engine.snapshot, &mut args.engine.attention)),
        _ => None,
    }
}

fn select_command_attention(command: &mut Commands,
    selected: Option<cuteafd_loader::plan::checkpoint::CheckpointView>) -> Result<()> {
    if let Some((snapshot, attention)) = command_attention(command) {
        let view = selected.map(Ok).unwrap_or_else(|| attention.select(snapshot))?;
        attention.validate_view(&view)?;
        attention.selected_view = Some(view);
    }
    Ok(())
}

fn resolve_command_attention_precision(command: &mut Commands) -> Result<()> {
    let (id, requested) = match command {
        Commands::ServeNative(args) => ("deepseek_v41", &mut args.attention_precision),
        Commands::V41Golden(args) => ("deepseek_v41", &mut args.serve.attention_precision),
        Commands::ServeDsv4(args) => ("deepseek_v4", &mut args.engine.attention_precision),
        Commands::Dsv4Golden(args) => ("deepseek_v4", &mut args.engine.attention_precision),
        Commands::ServeGlm(args) => ("glm5", &mut args.engine.attention.attention_precision),
        Commands::GlmGolden(args) => ("glm5", &mut args.engine.attention.attention_precision),
        Commands::ServeGlmf(args) => ("glm5_flash", &mut args.engine.attention.attention_precision),
        Commands::GlmfGolden(args) => ("glm5_flash", &mut args.engine.attention.attention_precision),
        Commands::ServeMimo(args) => ("mimo_v2", &mut args.engine.attention_precision),
        Commands::MimoGolden(args) => ("mimo_v2", &mut args.engine.attention_precision),
        Commands::ServeQwen4(args) => ("qwen4", &mut args.engine.attention_precision),
        Commands::Qwen4Golden(args) => ("qwen4", &mut args.engine.attention_precision),
        Commands::ServeQwen35(args) => ("qwen3_5", &mut args.attention_precision),
        Commands::Qwen35Golden(args) => ("qwen3_5", &mut args.attention_precision),
        Commands::ServeKimi(args) => ("kimi_k3", &mut args.attention.attention_precision),
        Commands::KimiGolden(args) => ("kimi_k3", &mut args.attention.attention_precision),
        Commands::ServeStep5(args) => ("step5", &mut args.attention.attention_precision),
        Commands::Step5Golden(args) => ("step5", &mut args.attention.attention_precision),
        Commands::ServeMinimax(args) => ("minimax_m3", &mut args.attention.attention_precision),
        Commands::MinimaxGolden(args) => ("minimax_m3", &mut args.attention.attention_precision),
        Commands::ServeMuse(args) => ("muse_spark", &mut args.attention.attention_precision),
        Commands::MuseGolden(args) => ("muse_spark", &mut args.engine.attention.attention_precision),
        Commands::ServeInkling2(args) => ("inkling2", &mut args.attention.attention_precision),
        Commands::Inkling2Golden(args) => ("inkling2", &mut args.attention.attention_precision),
        Commands::Hy4fGolden(args) => ("hy4_flash", &mut args.attention.attention_precision),
        Commands::ServeHy4f(args) => ("hy4_flash", &mut args.attention.attention_precision),
        _ => return Ok(()),
    };
    let family = cuteafd_loader::plan::family::registry().iter().copied()
        .find(|family| family.id() == id).context("missing attention family descriptor")?;
    let resolved = cuteafd_loader::plan::attention_precision::resolve(family, *requested)?;
    tracing::info!(family = id, requested = %requested, effective = %resolved.mode,
        reason = %resolved.reason, "resolved attention precision");
    *requested = resolved.mode;
    cuteafd_bench::context::set_resolved("attention-precision", resolved.mode.name());
    Ok(())
}

fn resolve_serve_media(command: &mut Commands, vision: cuteafd_loader::plan::MediaMode,
    audio: cuteafd_loader::plan::MediaMode) -> Result<(cuteafd_loader::plan::MediaMode, cuteafd_loader::plan::MediaMode)> {
    use cuteafd_loader::plan::MediaMode;
    // Once selected, a missing view is an error, never permission to reopen defaults.
    let selected = command_attention(command).map(|(_, attention)| {
        attention.selected_view.as_ref().context("media requires its selected checkpoint view")?.config()
    }).transpose()?;
    if matches!(command, Commands::ServeQwen35(_)) {
        // No dense Qwen tower is qualified: automatic media serves text, explicit placement refuses.
        anyhow::ensure!(matches!(vision, MediaMode::Auto | MediaMode::Off) && matches!(audio, MediaMode::Auto | MediaMode::Off),
            "dense Qwen has no qualified media execution; use --vision auto|off and --audio auto|off");
        return Ok((MediaMode::Off, MediaMode::Off));
    }
    let Some(options) = serve_media(command) else {
        anyhow::ensure!(matches!(audio, MediaMode::Auto | MediaMode::Off),
            "explicit audio placement requires a media-capable serve command");
        return Ok((vision, MediaMode::Off));
    };
    let config = match selected {
        Some(config) => config,
        None if vision != MediaMode::Off || audio != MediaMode::Off =>
            cuteafd_loader::read_json_file(&options.snapshot.join("config.json"))?,
        None => serde_json::Value::Null,
    };
    let vision = if vision == MediaMode::Off { vision } else {
        match config.get("vision_config").filter(|value| !value.is_null()) {
            Some(config) => {
                anyhow::ensure!(config.is_object(), "vision_config must be an object");
                vision
            }
            None if vision == MediaMode::Auto => MediaMode::Off,
            None => anyhow::bail!("checkpoint has no bundled vision tower"),
        }
    };
    let audio = cuteafd_loader::plan::resolve_audio_config(audio, options.snapshot, &config)?;
    anyhow::ensure!(options.audio.is_some() || audio == MediaMode::Off,
        "this serve command has no bundled audio execution owner");
    *options.vision = vision;
    if let Some(target) = options.audio { *target = audio; }
    Ok((vision, audio))
}

#[cfg(test)]
mod media_defaults_tests {
    use super::{resolve_qwen_vision, resolve_vision};
    use cuteafd_loader::plan::MediaMode;

    #[test]
    fn glm_golden_help_lists_logit_dump_for_sealed_runner() {
        use clap::CommandFactory;
        let mut command = super::cli::Cli::command();
        let golden = command.find_subcommand_mut("glm-golden").unwrap();
        assert!(golden.render_long_help().to_string().contains("--save-logits"));
    }

    #[test]
    fn mapped_table_backend_selector_is_removed() {
        use clap::Parser;
        for backend in ["mmap", "uring", "mincore-routed"] {
            let error = super::cli::Cli::try_parse_from([
                "cuteafd", "--table-backend", backend, "plan", "/model",
            ]).unwrap_err();
            assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
        }
    }

    #[test]
    fn native_flash_serving_cli_accepts_a16_and_refuses_a8() {
        use clap::Parser;
        use super::cli::{Cli, Commands};
        use cuteafd_loader::plan::exlr8::PrefillActivations;
        let base = ["cuteafd", "serve-glmf", "--snapshot", "/model", "--native-lib", "/native.so",
            "--exlr8-manifest", "k4"];
        for extra in [vec![], vec!["--exlr8-prefill-activations", "a16"]] {
            let cli = Cli::try_parse_from(base.into_iter().chain(extra)).unwrap();
            let Commands::ServeGlmf(args) = cli.command else { panic!("expected Flash serving"); };
            assert_eq!(args.engine.exlr8_prefill_activations, PrefillActivations::A16);
        }
        for value in ["a8", "auto", "unknown"] {
            let error = Cli::try_parse_from(base.into_iter()
                .chain(["--exlr8-prefill-activations", value])).unwrap_err();
            assert_eq!(error.kind(), clap::error::ErrorKind::ValueValidation);
            assert!(error.to_string().contains("native Flash EXLR8 prefill is A16"), "{error}");
        }
    }

    #[test]
    fn absent_daemon_image_cap_keeps_native_family_defaults() {
        use clap::Parser;
        use cuteafd_loader::media::{ImageFamily,ProcessorConfig};
        let cli=super::cli::Cli::try_parse_from(["cuteafd","plan","/model"]).unwrap();
        assert_eq!(cli.max_image_tokens,None);
        for (family,expected) in [(ImageFamily::Kimi,16698),(ImageFamily::MiniMax,576)] {
            let config=ProcessorConfig::for_family(family);
            assert_eq!(config.resolved_span_cap(cli.max_image_tokens.map(|n|n as usize)).unwrap(),expected);
        }
        let explicit=super::cli::Cli::try_parse_from(["cuteafd","--max-image-tokens","512","plan","/model"]).unwrap();
        for family in [ImageFamily::Kimi,ImageFamily::MiniMax] {
            assert_eq!(ProcessorConfig::for_family(family).resolved_span_cap(explicit.max_image_tokens.map(|n|n as usize)).unwrap(),512);
        }
    }

    #[test]
    fn planner_audio_defaults_auto_and_preserves_global_off() {
        use clap::Parser;
        let cli = super::cli::Cli::try_parse_from(["cuteafd", "plan", "/model"]).unwrap();
        let super::cli::Commands::Plan(args) = cli.command else { panic!("expected planner"); };
        assert_eq!(args.audio, MediaMode::Auto);
        let cli = super::cli::Cli::try_parse_from(["cuteafd", "plan", "/model", "--audio", "off"]).unwrap();
        assert_eq!(cli.audio, Some(MediaMode::Off));
    }

    #[test]
    fn qualified_qwen_vision_defaults_auto_and_preserves_overrides() {
        assert_eq!(resolve_qwen_vision(None, None), MediaMode::Auto);
        assert_eq!(resolve_qwen_vision(None, Some(MediaMode::Off)), MediaMode::Off);
        assert_eq!(resolve_qwen_vision(None, Some(MediaMode::Auto)), MediaMode::Auto);
        assert_eq!(resolve_qwen_vision(Some(MediaMode::Off), Some(MediaMode::Auto)), MediaMode::Off);
        assert_eq!(resolve_qwen_vision(Some(MediaMode::Spark(Some(0))), None), MediaMode::Spark(Some(0)));
    }

    #[test]
    fn qwen_direct_cli_defaults_auto_and_resolves_explicit_off() {
        use clap::Parser;
        use super::cli::{Cli, Commands};
        let base = ["cuteafd", "serve-qwen4", "--snapshot", "/model", "--native-lib", "/native.so"];
        let defaults = Cli::try_parse_from(base).unwrap();
        let Commands::ServeQwen4(args) = defaults.command else { panic!("expected Qwen serving"); };
        assert_eq!(args.vision, MediaMode::Auto);
        assert_eq!(resolve_qwen_vision(defaults.vision, None), MediaMode::Auto);
        let off = Cli::try_parse_from(base.into_iter().chain(["--vision", "off"])).unwrap();
        assert_eq!(resolve_qwen_vision(off.vision, None), MediaMode::Off);
    }

    fn media_command(name: &str, snapshot: &std::path::Path) -> super::Commands {
        use clap::Parser;
        let mut command = super::cli::Cli::try_parse_from(["cuteafd", name, "--snapshot", snapshot.to_str().unwrap(),
            "--native-lib", "/native.so"]).unwrap().command;
        super::select_command_attention(&mut command, None).unwrap();
        command
    }

    #[test]
    fn core_views_have_explicit_cli_and_no_old_flags() -> anyhow::Result<()> {
        use clap::Parser;
        for view in ["original", "k6-compact", "k8-compact", "fp8-compact"] {
            super::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--core-view", view])?;
        }
        assert!(super::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--core-view", "compact"]).is_err());
        Ok(())
    }

    #[test]
    fn checkpoint_media_resolution_is_uniform_for_mimo_muse_and_text_only_qwen() -> anyhow::Result<()> {
        use cuteafd_loader::plan::{testing, resolve_audio};
        let bundled = tempfile::tempdir()?;
        testing::write_muse_media_snapshot(bundled.path(), &[]);
        let text = tempfile::tempdir()?;
        std::fs::write(text.path().join("config.json"), r#"{"model_type":"qwen4_exp"}"#)?;
        for name in ["serve-mimo", "serve-muse", "serve-qwen4", "serve-glmf"] {
            for (snapshot, present) in [(bundled.path(), true), (text.path(), false)] {
                for mode in [MediaMode::Auto, MediaMode::Rtx(Some(0)), MediaMode::Spark(Some(0)), MediaMode::Off] {
                    let mut command = media_command(name, snapshot);
                    let result = super::resolve_serve_media(&mut command, mode, mode);
                    let owns_audio = super::serve_media(&mut command).unwrap().audio.is_some();
                    if (present && owns_audio) || mode == MediaMode::Off || (!present && mode == MediaMode::Auto) {
                        let expected = if present { mode } else { MediaMode::Off };
                        assert_eq!(result?, (expected, expected), "{name}: {mode:?}");
                        let options = super::serve_media(&mut command).unwrap();
                        assert_eq!(*options.vision, expected);
                        if let Some(audio) = options.audio { assert_eq!(*audio, expected); }
                    } else {
                        assert!(result.is_err(), "{name}: {mode:?} must not silently disable explicit media");
                    }
                }
                // A vision-only command can still serve its bundled image tower.
                let mut command = media_command(name, snapshot);
                let expected = if present { MediaMode::Auto } else { MediaMode::Off };
                assert_eq!(super::resolve_serve_media(&mut command, MediaMode::Auto, MediaMode::Off)?,
                    (expected, MediaMode::Off));
                if owns_audio_command(name) {
                    assert_eq!(resolve_audio(MediaMode::Auto, snapshot)?,
                        super::resolve_serve_media(&mut command, MediaMode::Off, MediaMode::Auto)?.1);
                }
            }
        }
        Ok(())
    }

    fn owns_audio_command(name: &str) -> bool { matches!(name, "serve-mimo" | "serve-muse") }

    #[test]
    fn dense_qwen_never_enables_media_input_and_refuses_explicit_placement() -> anyhow::Result<()> {
        use clap::Parser;
        let mut command = super::cli::Cli::try_parse_from(["cuteafd", "serve-qwen35", "--snapshot", "/not-read",
            "--native-lib", "/native.so"])?.command;
        for vision in [MediaMode::Auto, MediaMode::Off] {
            for audio in [MediaMode::Auto, MediaMode::Off] {
                assert_eq!(super::resolve_serve_media(&mut command, vision, audio)?, (MediaMode::Off, MediaMode::Off));
            }
        }
        for explicit in [MediaMode::Rtx(Some(0)), MediaMode::Spark(Some(0))] {
            assert!(super::resolve_serve_media(&mut command, explicit, MediaMode::Off).is_err());
            assert!(super::resolve_serve_media(&mut command, MediaMode::Off, explicit).is_err());
        }
        Ok(())
    }

    #[test]
    fn declared_mimo_rvq_errors_and_off_bypass_match_shared_resolver() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        for audio_config in [serde_json::json!({"audio_channels":20}),
            serde_json::json!({"model_type":"unknown_audio"})] {
            std::fs::write(dir.path().join("config.json"),
                serde_json::json!({"model_type":"mimo_v2", "hidden_size":4096,
                    "vision_config":{}, "audio_config":audio_config}).to_string())?;
            for mode in [MediaMode::Auto, MediaMode::Rtx(Some(0)), MediaMode::Spark(Some(0))] {
                let expected = cuteafd_loader::plan::resolve_audio(mode, dir.path()).unwrap_err().to_string();
                for name in ["serve-mimo", "serve-muse"] {
                    let mut command = media_command(name, dir.path());
                    let actual = super::resolve_serve_media(&mut command, MediaMode::Off, mode).unwrap_err();
                    assert_eq!(actual.to_string(), expected);
                }
            }
            let mut command = media_command("serve-mimo", dir.path());
            assert_eq!(super::resolve_serve_media(&mut command, MediaMode::Off, MediaMode::Off)?,
                (MediaMode::Off, MediaMode::Off));
        }
        Ok(())
    }

    #[test]
    fn qualified_vision_defaults_auto_and_preserves_explicit_off() {
        assert_eq!(resolve_vision(None, None), MediaMode::Auto);
        assert_eq!(resolve_vision(None, Some(MediaMode::Off)), MediaMode::Off);
        assert_eq!(resolve_vision(Some(MediaMode::Off), Some(MediaMode::Auto)), MediaMode::Off);
        assert_eq!(resolve_vision(Some(MediaMode::Spark(Some(0))), None), MediaMode::Spark(Some(0)));
    }
}

fn main() -> Result<()> {
    // Before any thread, tracing or CUDA: exec the AVX-512 build when this host
    // has AVX-512 and the image carries it (cuteafd_ffi::host_isa).
    let host_isa = cuteafd_ffi::host_isa::startup()?;
    // Before any worker thread: pin glibc's mmap threshold so a request's multi-megabyte
    // transients go back to the kernel on free instead of fragmenting the arenas
    // (shared::malloc_tune; CUTEAFD_MALLOC_TUNE=off keeps glibc's dynamic threshold).
    // Apply unconditionally: under the jemalloc build this still tunes glibc, which the
    // prebuilt native library (libcuteafd_native.so) uses.
    let malloc = shared::malloc_tune::apply_from_env().to_string();
    tracing_subscriber::fmt()
        .with_ansi(std::io::stderr().is_terminal())
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    tracing::info!("{host_isa}");
    tracing::info!("{malloc}");
    // Opt-in only (CUTEAFD_DIAG_STACK_ON_TIMEOUT=1): its handler is installed before any worker thread.
    shared::stack_dump::install();
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let result = shared::memory_report::finish(runtime.block_on(run()));
    shared::decode_graph::device::shutdown_traces();
    shared::speculation_trace::shutdown();
    cuteafd_ffi::memory_ledger::fault::shutdown();
    // Blocking CUDA workers cannot be cancelled; a stuck peer must not hang exit.
    runtime.shutdown_timeout(std::time::Duration::from_secs(1));
    result
}

async fn run() -> Result<()> {

    let parse = |matches: clap::ArgMatches| {

        match Cli::from_arg_matches(&matches) {
            Ok(mut cli) => {
                if let Some(samples) = &cli.spark_memory_samples {
                    if let Err(error) = cuteafd_loader::serving_capacity::spark_memory::install_samples(samples) {
                        clap::Error::raw(clap::error::ErrorKind::ValueValidation,
                            format!("invalid --spark-memory-samples: {error}")).exit();
                    }
                }
                if let Err(error) = cli::resolve_tp2_dflash(&mut cli.command, &matches) {
                    clap::Error::raw(clap::error::ErrorKind::ValueValidation, error.to_string()).exit();
                }
                if let Commands::Plan(args) = &mut cli.command {
                    args.spark_memory_samples = cli.spark_memory_samples.clone();
                    if let Some((_, options)) = matches.subcommand() {
                        args.explicit_settings = options.ids().filter(|id| matches!(options.value_source(id.as_str()),
                            Some(clap::parser::ValueSource::CommandLine | clap::parser::ValueSource::EnvVariable)))
                            .map(|id| id.to_string()).collect();
                    }
                }
                (cli.command, matches, cli.memory, cli.vision, cli.audio, cli.max_image_tokens, cli.image_url_fetch,
                    cuteafd_transport::wait_policy::WaitPolicy { wait: cli.host_wait,
                        idle_park: std::time::Duration::from_secs_f64(cli.idle_park_seconds) }, cuteafd_core::spark_reply::SparkReplyFormats {
                            prefill: cli.spark_prefill_reply_format, decode: cli.spark_decode_reply_format },
                    cli.coordinator_weight_budget_gib, cli.graph_disable_low_mem,
                    (cli.graph_no_prewarm, cli.graph_donor_always, cli.graph_donor_cold_lag))
            },
            Err(error) => error.exit(),
        }
    };
    let (command, matches, initial_budget, initial_vision, initial_audio, initial_image_cap, initial_fetch, initial_wait, initial_reply, initial_weight_budget, initial_graph_floor, initial_graph_policy) =
        parse(Cli::command().get_matches());
    let mut selected_view = None;
    // `serve` and `golden` pick the family and stand for its own command.
    let (mut command, matches, family_budget, family_vision, family_audio, family_image_cap, family_fetch, wait_policy, family_reply, family_weight_budget, family_graph_floor, family_graph_policy) = match command {
        Commands::Serve(args) => match commands::family::resolve(commands::family::Kind::Serve, args)? {
            Some(resolved) => { selected_view = resolved.view; parse(Cli::command().get_matches_from(resolved.argv)) },
            None => return Ok(()),
        },
        Commands::Golden(args) => match commands::family::resolve(commands::family::Kind::Golden, args)? {
            Some(resolved) => { selected_view = resolved.view; parse(Cli::command().get_matches_from(resolved.argv)) },
            None => return Ok(()),
        },
        command => (command, matches, initial_budget, initial_vision, initial_audio, initial_image_cap, initial_fetch,
            initial_wait, initial_reply, initial_weight_budget, initial_graph_floor.clone(), initial_graph_policy),
    };
    // Spanning graph sites always prewarm, so `off` is safe on every layout (#2303).
    let no_prewarm = initial_graph_policy.0 || family_graph_policy.0;
    // Qualify pressure-only donor reuse at Flash's existing exact-row reserve.
    let (donor_always, donor_cold_lag) = if matches!(&command, Commands::ServeGlmf(_) | Commands::GlmfGolden(_)) {
        (false, 0)
    } else {
        (initial_graph_policy.1 || family_graph_policy.1, initial_graph_policy.2.max(family_graph_policy.2))
    };
    shared::decode_graph::configure(no_prewarm, donor_always, donor_cold_lag)?;
    if let Some((role, directory)) = shared::memory_report::dump_location(&matches) {
        let host = std::env::var("CUTEAFD_PROFILE_HOST").unwrap_or_else(|_| {
            std::env::var("HOSTNAME").map(|name| format!("container {name}")).unwrap_or_else(|_| "unknown".into())
        });
        if let Err(error) = cuteafd_ffi::memory_ledger::fault::initialize(&directory, &host, &role) {
            tracing::error!(%error, directory = %directory.display(), "memory failure dump unavailable: cannot preopen spool");
        }
    } else if matches.subcommand_name().is_some_and(|name| name == "expertd-native" || name.starts_with("serve")) {
        tracing::warn!("memory failure dump unavailable: no configured instance usage directory or profile output directory");
    }
    select_command_attention(&mut command, selected_view)?;
    resolve_command_attention_precision(&mut command)?;
    cuteafd_core::spark_reply::set_requested(cuteafd_core::spark_reply::SparkReplyFormats {
        prefill: family_reply.prefill.or(initial_reply.prefill),
        decode: family_reply.decode.or(initial_reply.decode),
    }).map_err(anyhow::Error::msg)?;
    // Read by the coordinator and every Spark worker from the same keys.
    cuteafd_transport::wait_policy::set_wait_policy(wait_policy)?;
    if let Some((_, options)) = matches.subcommand() {
        if let Ok(Some(path)) = options.try_get_one::<std::path::PathBuf>("native_lib") {
            cuteafd_transport::set_verbs_host_native_library_path(path.clone())?;
        }
    }
    let vision = if matches!(&command, Commands::ServeQwen4(_)) {
        resolve_qwen_vision(family_vision, initial_vision)
    } else {
        resolve_vision(family_vision, initial_vision)
    };
    // GLM 5.3 Flash: automatic vision serves text when the checkpoint's own chat template cannot
    // mark images. Planner-resolved peers are an explicit placement and keep serving's refusal.
    let vision = match &mut command {
        Commands::ServeGlmf(args) => {
            let view = args.engine.attention.selected_view.as_ref().context("selected Flash view")?;
            let (vision, note) = if args.vision_peers.is_none() {
                families::glm5_flash::serve::auto_vision_view(vision, &view, args.chat_template_from.as_deref())?
            } else { (vision, None) };
            if let Some(note) = note { tracing::warn!("{note}"); }
            vision
        }
        _ => vision,
    };
    let requested_audio = family_audio.or(initial_audio).unwrap_or(cuteafd_loader::plan::MediaMode::Auto);
    let (vision, audio) = if let Commands::Plan(args) = &mut command {
        args.vision = vision;
        args.audio = requested_audio;
        (vision, requested_audio)
    } else {
        resolve_serve_media(&mut command, vision, requested_audio)?
    };
    cuteafd_api::openai::set_media_input_policy(vision != cuteafd_loader::plan::MediaMode::Off,
        audio != cuteafd_loader::plan::MediaMode::Off);
    cuteafd_api::openai::media::set_preparation_policy(
        family_image_cap.or(initial_image_cap).map(|cap| cap as usize),
        family_fetch.or(initial_fetch).unwrap_or(cuteafd_api::openai::media::ImageUrlFetch::Public))?;
    let weight_budget_gib = family_weight_budget.or(initial_weight_budget);
    if let Commands::Plan(args) = &mut command {
        args.coordinator_weight_budget_gib = weight_budget_gib;
    } else if let Some(gib) = weight_budget_gib {
        let bytes = cuteafd_loader::plan::budget_bytes("--coordinator-weight-budget-gib", gib)?;
        shared::memory::budget::set_resident_weight_limit(cuteafd_loader::placement::ResidentWeights(bytes))?;
    }
    let memory = family_budget.or(initial_budget);
    let policy = memory.policy();
    policy.validate()?;
    if let Commands::Plan(args) = &mut command {
        args.memory = memory;
    } else if let Some(cap) = policy.gpu_cap {
        anyhow::ensure!(matches!(&command, Commands::ServeNative(_) | Commands::ServeMimo(_)
            | Commands::ServeQwen4(_) | Commands::ServeQwen35(_) | Commands::ServeKimi(_) | Commands::ServeMinimax(_) | Commands::ServeInkling2(_) | Commands::ServeHy4f(_) | Commands::ServeStep5(_) | Commands::ServeMuse(_) | Commands::ServeGlmf(_) | Commands::ServeGlm(_)
            | Commands::ServeDsv4(_) | Commands::V41Golden(_) | Commands::Dsv4Golden(_) | Commands::GlmGolden(_)
            | Commands::MimoGolden(_) | Commands::GlmfGolden(_) | Commands::Qwen4Golden(_) | Commands::Qwen35Golden(_) | Commands::Step5Golden(_) | Commands::KimiGolden(_) | Commands::MinimaxGolden(_) | Commands::Inkling2Golden(_) | Commands::Hy4fGolden(_) | Commands::MuseGolden(_)),
            "--gpu-memory-cap applies only to coordinator serve/golden and plan commands");
        cuteafd_ffi::set_coordinator_gpu_budget(cap)?;
        tracing::info!(bytes = cap, "installed per-GPU memory cap; SM count and L2 unchanged");
    }
    policy.install().map_err(anyhow::Error::msg)?;
    let carveout = memory.spark_carveout != Some(cli::SparkCarveout::Off);
    cuteafd_ffi::attic::set_enabled(carveout);
    cuteafd_loader::serving_capacity::spark_memory::set_carveout_enabled(carveout);
    // Resolve the environment only after generic serve/golden's family reparse:
    // an environment fallback must not overwrite an explicit outer CLI value.
    let graph_floor = family_graph_floor.or(initial_graph_floor).or(
        std::env::var("CUTEAFD_GRAPH_DISABLE_LOW_MEM").map(Some).or_else(|error| match error {
            std::env::VarError::NotPresent => Ok(None),
            error => Err(error),
        })?);
    shared::decode_graph::device::set_physical_floor(graph_floor.clone())?;
    shared::memory_report::register_inputs(&matches, &policy, weight_budget_gib);
    // A serve command's resolved options, for the server's benchmark reports.
    commands::bench::capture(&matches, memory, graph_floor.as_deref())?;
    // Memory ledger reports for the long-running roles (device use by category).
    match &command {
        Commands::Expertd(_) => {
            shared::memory_report::monitor("expertd", std::time::Duration::from_secs(10));
            shared::thermal::monitor("expertd");
        }
        Commands::ServeNative(_) | Commands::ServeMimo(_) | Commands::ServeQwen4(_) | Commands::ServeQwen35(_) | Commands::ServeKimi(_) | Commands::ServeMinimax(_) | Commands::ServeInkling2(_) | Commands::ServeHy4f(_) | Commands::ServeStep5(_) | Commands::ServeMuse(_) | Commands::ServeGlmf(_)
        | Commands::ServeGlm(_) | Commands::ServeDsv4(_) => {
            shared::host_heap::start();
            tracing::info!(target: "cuteafd::fabric", source = "sparknest", status = cuteafd_transport::traffic::storage_status(), "local storage traffic source");
            shared::memory_report::monitor("coordinator", std::time::Duration::from_secs(10));
            shared::thermal::monitor("coordinator");
        }
        _ => {}
    }
    match command {
        Commands::Serve(_) | Commands::Golden(_) => unreachable!("resolved to a family command above"),
        Commands::Gateway(args) => commands::gateway::run(args).await,
        Commands::Usage(args) => commands::usage::run(args).await,
        Commands::Doctor(args) => run_doctor(args),
        Commands::Plan(args) => run_plan(args),
        Commands::ProgramsCheck(args) => commands::programs_check::run(args),
        Commands::V41Golden(args) => families::deepseek_v41::v41_golden::run_golden(args).await,
        Commands::Dsv4Golden(args) => families::deepseek_v4::run_golden(args).await,
        Commands::GlmGolden(args) => families::glm5::run_golden(args).await,
        Commands::MimoGolden(args) => families::mimo_v2::run_golden(args).await,
        Commands::ServeMimo(args) => families::mimo_v2::serve::run_serve(args).await,
        Commands::GlmfGolden(args) => families::glm5_flash::run_golden(args).await,
        Commands::Qwen4Golden(args) => families::qwen4::run_golden(args).await,
        Commands::Qwen35Golden(args) => families::qwen3_5::run_golden(args).await,
        Commands::Step5Golden(args) => families::step5::golden::run_golden(args).await,
        Commands::KimiGolden(args) => families::kimi_k3::golden::run_golden(args).await,
        Commands::MinimaxGolden(args) => families::minimax_m3::golden::run_golden(args).await,
        Commands::Inkling2Golden(args) => families::inkling2::golden::run_golden(args).await,
        Commands::Hy4fGolden(args) => families::hy4_flash::golden::run_golden(args).await,
        Commands::ServeHy4f(args) => families::hy4_flash::serve::run(args).await,
        Commands::ServeInkling2(args) => families::inkling2::serve::run(args).await,
        Commands::ServeMinimax(args) => families::minimax_m3::serve::run(args).await,
        Commands::ServeKimi(args) => families::kimi_k3::serve::run(args).await,
        Commands::ServeStep5(args) => families::step5::serve::run_serve(args).await,
        Commands::ServeMuse(args) => families::muse_spark::serve::run(args).await,
        Commands::MuseGolden(args) => families::muse_spark::packed_check::run(args).await,
        Commands::ServeQwen35(args) => families::qwen3_5::serve::run_serve(args).await,
        Commands::ServeQwen4(args) => families::qwen4::serve::run_serve(args).await,
        Commands::ServeGlmf(args) => families::glm5_flash::serve::run_serve(args).await,
        Commands::ServeGlm(args) => families::glm5::serve::run_serve(args).await,
        Commands::ServeDsv4(args) => families::deepseek_v4::serve::run_serve(args).await,
        Commands::Fabric(args) => {
            let report = cuteafd_transport::fabric::discover()?;
            let landings = shared::spark_intake::fabric_probe(args.native_lib.as_deref(), args.device,
                args.landing_warmup, args.landing_repeats, args.landing_peer);
            let landing = &landings[0];
            let p2p = args.p2p.then(|| shared::peer_probe::run(args.native_lib.as_deref(), &args.p2p_devices,
                &args.p2p_bytes));
            if args.json {
                let mut value = serde_json::to_value(&report)?;
                value["gpu_landing"] = serde_json::to_value(landing)?;
                value["gpu_landing_repeats"] = serde_json::to_value(&landings)?;
                if let Some(p2p) = &p2p {
                    value["p2p"] = serde_json::to_value(p2p)?;
                }
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                for port in &report.ports {
                    println!(
                        "{} port {}: {} {} {:.0} Gb/s, PCIe {}, netdev {}, RoCE v2 {:?}, subnets {:?}",
                        port.device,
                        port.port,
                        if port.active { "active" } else { "down" },
                        port.link_layer,
                        port.link_gbps,
                        port.pci.as_ref().map_or("?".into(), |pci| format!("{} GT/s x{} ({:.0} Gb/s)", pci.gts, pci.width, pci.gbps())),
                        port.netdev.as_deref().unwrap_or("-"),
                        port.roce_v2.iter().map(|(_, address)| address).collect::<Vec<_>>(),
                        port.subnets,
                    );
                }
                println!("{}", report.summary());
                for landing in &landings {
                    println!("{}", landing.summary());
                }
                if let Some(p2p) = &p2p {
                    print!("{}", p2p.table());
                }
            }
            Ok(())
        }
        Commands::Expertd(args) => shared::experts::service::run(args).await,
        Commands::ServeNative(args) => families::deepseek_v41::v41_native_serve::run(args).await,
        Commands::Bench(args) => tokio::task::spawn_blocking(move || commands::bench::run(args)).await?,
        Commands::BenchRdma(args) => run_bench_rdma(args),
        Commands::BenchRdmaRing(args) => run_bench_rdma_ring(args),
        Commands::TransportCapabilities(args) => run_transport_capabilities(args),
    }
}

pub(crate) fn command_probe(program: &str, args: &[&str]) -> Probe {
    match Command::new(program).args(args).output() {
        Ok(output) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            Probe {
                ok: output.status.success(),
                output: text.trim().to_owned(),
            }
        }
        Err(err) => Probe {
            ok: false,
            output: err.to_string(),
        },
    }
}
