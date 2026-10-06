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
use commands::expert_probe::run_expert_probe;
use commands::plan::run_plan;
use commands::transport_capabilities::run_transport_capabilities;

#[derive(Debug, Serialize)]
pub(crate) struct Probe {
    pub(crate) ok: bool,
    pub(crate) output: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let parse = |matches: clap::ArgMatches| {
        match Cli::from_arg_matches(&matches) {
            Ok(cli) => (cli.command, matches, cli.coordinator_gpu_budget_gib, cli.vision, cli.audio, cli.max_image_tokens, cli.image_url_fetch),
            Err(error) => error.exit(),
        }
    };
    let (command, matches, initial_budget, initial_vision, initial_audio, initial_image_cap, initial_fetch) = parse(Cli::command().get_matches());
    // `serve` and `golden` pick the family and stand for its own command.
    let (mut command, matches, family_budget, family_vision, family_audio, family_image_cap, family_fetch) = match command {
        Commands::Serve(args) => match commands::family::argv(commands::family::Kind::Serve, args)? {
            Some(argv) => parse(Cli::command().get_matches_from(argv)),
            None => return Ok(()),
        },
        Commands::Golden(args) => match commands::family::argv(commands::family::Kind::Golden, args)? {
            Some(argv) => parse(Cli::command().get_matches_from(argv)),
            None => return Ok(()),
        },
        command => (command, matches, initial_budget, initial_vision, initial_audio, initial_image_cap, initial_fetch),
    };
    let vision = family_vision.or(initial_vision).unwrap_or(cuteafd_loader::plan::MediaMode::Auto);
    let audio = family_audio.or(initial_audio).unwrap_or(cuteafd_loader::plan::MediaMode::Off);
    if let Commands::Plan(args) = &mut command { args.vision = vision; args.audio = audio; }
    if let Commands::ServeMimo(args) = &mut command { args.vision = vision; }
    if let Commands::ServeQwen4(args) = &mut command {
        args.vision = family_vision.or(initial_vision).unwrap_or(cuteafd_loader::plan::MediaMode::Off);
    }
    cuteafd_api::openai::set_media_input_policy(vision != cuteafd_loader::plan::MediaMode::Off,
        audio != cuteafd_loader::plan::MediaMode::Off);
    cuteafd_api::openai::media::set_preparation_policy(
        family_image_cap.or(initial_image_cap).unwrap_or(4096) as usize,
        family_fetch.or(initial_fetch).unwrap_or(cuteafd_api::openai::media::ImageUrlFetch::Public))?;
    let coordinator_budget_gib = family_budget.or(initial_budget);
    if let Some(gib) = coordinator_budget_gib {
        anyhow::ensure!(matches!(&command, Commands::ServeNative(_) | Commands::ServeMimo(_)
            | Commands::ServeQwen4(_) | Commands::ServeGlmf(_) | Commands::ServeGlm(_)
            | Commands::ServeDsv4(_) | Commands::Dsv4Golden(_) | Commands::GlmGolden(_)
            | Commands::MimoGolden(_) | Commands::GlmfGolden(_) | Commands::Qwen4Golden(_)),
            "--coordinator-gpu-budget-gib applies only to coordinator serve/golden commands; plan uses --layout --rtx-budget-gib");
        let budget = cuteafd_core::serving_capacity::GpuMemoryBudget::from_gib(gib)?;
        cuteafd_ffi::set_coordinator_gpu_budget(budget.0)?;
        tracing::info!(gib, bytes = budget.0, "installed per-GPU coordinator memory budget; SM count and L2 unchanged");
    }
    // A serve command's resolved options, for the server's benchmark reports.
    commands::bench::capture(&matches, coordinator_budget_gib);
    // Memory ledger reports for the long-running roles (device use by category).
    match &command {
        Commands::Expertd(_) => shared::memory_report::monitor("expertd", std::time::Duration::from_secs(10)),
        Commands::ServeNative(_) | Commands::ServeMimo(_) | Commands::ServeQwen4(_) | Commands::ServeGlmf(_)
        | Commands::ServeGlm(_) | Commands::ServeDsv4(_) =>
            shared::memory_report::monitor("coordinator", std::time::Duration::from_secs(10)),
        _ => {}
    }
    match command {
        Commands::Serve(_) | Commands::Golden(_) => unreachable!("resolved to a family command above"),
        Commands::Doctor(args) => run_doctor(args),
        Commands::Plan(args) => run_plan(args),
        Commands::ExpertProbe(args) => run_expert_probe(args).await,
        Commands::Dsv4Golden(args) => families::deepseek_v4::run_golden(args).await,
        Commands::GlmGolden(args) => families::glm5::run_golden(args).await,
        Commands::MimoGolden(args) => families::mimo_v2::run_golden(args).await,
        Commands::ServeMimo(args) => families::mimo_v2::serve::run_serve(args).await,
        Commands::GlmfGolden(args) => families::glm5_flash::run_golden(args).await,
        Commands::Qwen4Golden(args) => families::qwen4::run_golden(args).await,
        Commands::ServeQwen4(args) => families::qwen4::serve::run_serve(args).await,
        Commands::ServeGlmf(args) => families::glm5_flash::serve::run_serve(args).await,
        Commands::ServeGlm(args) => families::glm5::serve::run_serve(args).await,
        Commands::ServeDsv4(args) => families::deepseek_v4::serve::run_serve(args).await,
        Commands::Fabric(args) => {
            let report = cuteafd_transport::fabric::discover()?;
            let landing = shared::spark_intake::fabric_probe(args.native_lib.as_deref(), args.device);
            let p2p = args.p2p.then(|| shared::peer_probe::run(args.native_lib.as_deref(), &args.p2p_devices,
                &args.p2p_bytes));
            if args.json {
                let mut value = serde_json::to_value(&report)?;
                value["gpu_landing"] = serde_json::to_value(&landing)?;
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
                println!("{}", landing.summary());
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
