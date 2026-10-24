use anyhow::Result;
use clap::{CommandFactory, FromArgMatches};
use serde::Serialize;
use std::process::Command;

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

#[cfg(test)]
mod media_defaults_tests {
    use super::{resolve_qwen_vision, resolve_vision};
    use cuteafd_loader::plan::MediaMode;

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
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    tracing::info!("{host_isa}");
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(run())
}

async fn run() -> Result<()> {

    let parse = |matches: clap::ArgMatches| {
        match Cli::from_arg_matches(&matches) {
            Ok(cli) => (cli.command, matches, cli.coordinator_gpu_budget_gib, cli.vision, cli.audio, cli.max_image_tokens, cli.image_url_fetch,
                cuteafd_transport::SparkChunking { request: cli.spark_request_chunk_bytes.0, reply: cli.spark_reply_chunk_bytes.0 }),
            Err(error) => error.exit(),
        }
    };
    let (command, matches, initial_budget, initial_vision, initial_audio, initial_image_cap, initial_fetch, initial_chunking) =
        parse(Cli::command().get_matches());
    for replacement in cli::deprecated_budget_flags(std::env::args()) {
        eprintln!("warning: deprecated coordinator budget flag; use {replacement}");
    }
    // `serve` and `golden` pick the family and stand for its own command.
    let (mut command, matches, family_budget, family_vision, family_audio, family_image_cap, family_fetch, chunking) = match command {
        Commands::Serve(args) => match commands::family::argv(commands::family::Kind::Serve, args)? {
            Some(argv) => parse(Cli::command().get_matches_from(argv)),
            None => return Ok(()),
        },
        Commands::Golden(args) => match commands::family::argv(commands::family::Kind::Golden, args)? {
            Some(argv) => parse(Cli::command().get_matches_from(argv)),
            None => return Ok(()),
        },
        command => (command, matches, initial_budget, initial_vision, initial_audio, initial_image_cap, initial_fetch,
            initial_chunking),
    };
    // TEMPORARY: remove after PFC test (TJ 2026-10-25). Coordinators send
    // these to every worker; a worker uses only the values it receives.
    cuteafd_transport::set_spark_chunking(chunking)?;
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
    let vision = match &command {
        Commands::ServeGlmf(args) if args.vision_peers.is_none() => {
            let (vision, note) = families::glm5_flash::serve::auto_vision(vision, &args.engine.snapshot,
                args.chat_template_from.as_deref())?;
            if let Some(note) = note { tracing::warn!("{note}"); }
            vision
        }
        _ => vision,
    };
    let requested_audio = family_audio.or(initial_audio).unwrap_or(cuteafd_loader::plan::MediaMode::Auto);
    let audio = match &command {
        Commands::Plan(_) => requested_audio,
        Commands::ServeMimo(args) => cuteafd_loader::plan::resolve_audio(requested_audio, &args.engine.snapshot)?,
        _ => match requested_audio {
            cuteafd_loader::plan::MediaMode::Auto | cuteafd_loader::plan::MediaMode::Off => cuteafd_loader::plan::MediaMode::Off,
            _ => anyhow::bail!("explicit audio placement requires a qualified MiMo audio checkpoint"),
        },
    };
    if let Commands::Plan(args) = &mut command { args.vision = vision; args.audio = audio; }
    if let Commands::ServeMimo(args) = &mut command { args.vision = vision; args.audio = audio; }
    if let Commands::ServeGlmf(args) = &mut command { args.vision = vision; }
    if let Commands::ServeQwen4(args) = &mut command {
        args.vision = vision;
    }
    cuteafd_api::openai::set_media_input_policy(vision != cuteafd_loader::plan::MediaMode::Off,
        audio != cuteafd_loader::plan::MediaMode::Off);
    cuteafd_api::openai::media::set_preparation_policy(
        family_image_cap.or(initial_image_cap).unwrap_or(4096) as usize,
        family_fetch.or(initial_fetch).unwrap_or(cuteafd_api::openai::media::ImageUrlFetch::Public))?;
    let coordinator_budget_gib = family_budget.or(initial_budget);
    if let Commands::Plan(args) = &mut command {
        args.coordinator_gpu_budget_gib = coordinator_budget_gib;
    } else if let Some(gib) = coordinator_budget_gib {
        anyhow::ensure!(matches!(&command, Commands::ServeNative(_) | Commands::ServeMimo(_)
            | Commands::ServeQwen4(_) | Commands::ServeQwen35(_) | Commands::ServeKimi(_) | Commands::ServeStep5(_) | Commands::ServeGlmf(_) | Commands::ServeGlm(_)
            | Commands::ServeDsv4(_) | Commands::V41Golden(_) | Commands::Dsv4Golden(_) | Commands::GlmGolden(_)
            | Commands::MimoGolden(_) | Commands::GlmfGolden(_) | Commands::Qwen4Golden(_) | Commands::Qwen35Golden(_) | Commands::Step5Golden(_) | Commands::KimiGolden(_)),
            "--coordinator-gpu-budget-gib applies only to coordinator serve/golden and plan commands");
        let budget = cuteafd_core::serving_capacity::GpuMemoryBudget::from_gib(gib)?;
        cuteafd_ffi::set_coordinator_gpu_budget(budget.0)?;
        tracing::info!(gib, bytes = budget.0, "installed per-GPU coordinator memory budget; SM count and L2 unchanged");
    }
    // A serve command's resolved options, for the server's benchmark reports.
    commands::bench::capture(&matches, coordinator_budget_gib)?;
    // Memory ledger reports for the long-running roles (device use by category).
    match &command {
        Commands::Expertd(_) => {
            shared::memory_report::monitor("expertd", std::time::Duration::from_secs(10));
            shared::thermal::monitor("expertd");
        }
        Commands::ServeNative(_) | Commands::ServeMimo(_) | Commands::ServeQwen4(_) | Commands::ServeQwen35(_) | Commands::ServeKimi(_) | Commands::ServeStep5(_) | Commands::ServeGlmf(_)
        | Commands::ServeGlm(_) | Commands::ServeDsv4(_) => {
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
        Commands::ServeKimi(args) => families::kimi_k3::serve::run(args).await,
        Commands::ServeStep5(args) => families::step5::serve::run_serve(args).await,
        Commands::ServeQwen35(args) => families::qwen3_5::serve::run_serve(args).await,
        Commands::ServeQwen4(args) => families::qwen4::serve::run_serve(args).await,
        Commands::ServeGlmf(args) => families::glm5_flash::serve::run_serve(args).await,
        Commands::ServeGlm(args) => families::glm5::serve::run_serve(args).await,
        Commands::ServeDsv4(args) => families::deepseek_v4::serve::run_serve(args).await,
        Commands::Fabric(args) => {
            let report = cuteafd_transport::fabric::discover()?;
            let landings = shared::spark_intake::fabric_probe(args.native_lib.as_deref(), args.device,
                args.landing_warmup, args.landing_repeats);
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
