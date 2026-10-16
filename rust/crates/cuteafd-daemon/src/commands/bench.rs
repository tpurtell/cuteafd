//! `cuteafd bench`: run a benchmark profile on a server (the runner inside it
//! does the work), `bench publish` for the README table and benchmarks index,
//! `bench smoke --matrix` for release prep. Also the capture of a serve
//! command's resolved options for the server's own benchmark reports.
use anyhow::Result;
use clap::{ArgMatches, CommandFactory};
use cuteafd_bench::report::Setting;
use std::path::PathBuf;

#[derive(Debug, clap::Args)]
#[command(args_conflicts_with_subcommands = true)]
pub(crate) struct BenchArgs {
    #[command(subcommand)]
    pub(crate) action: Option<BenchAction>,
    #[command(flatten)]
    pub(crate) run: RunArgs,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum BenchAction {
    /// Multi-window fidelity probes and paired non-inferiority gates.
    Fidelity(cuteafd_bench::fidelity_cli::Args),
    /// Rebuild the root README's results table and benchmarks/README.md from
    /// the reports under benchmarks/<family>/<date>-<profile>-<hardware>/.
    Publish(PublishArgs),
    /// Release smoke over a matrix of launches (./run.sh per entry).
    Smoke(cuteafd_bench::smoke::SmokeArgs),
    /// Re-render the exports of a saved report.json (no server needed).
    Export {
        /// The report.json to render.
        #[arg(long)]
        json: PathBuf,
        /// Output directory (default: benchmarks/<family>/<date>-<profile>-<checkpoint>-<hardware>/ under --root).
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value = ".")]
        root: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "svg,png,json")]
        export: Vec<String>,
    },
    /// Cancel the server's active run.
    Cancel {
        #[arg(long, default_value = "http://127.0.0.1:8000")]
        url: String,
        #[arg(long, env = "CUTEAFD_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
    },
}

#[derive(Debug, clap::Args)]
pub(crate) struct RunArgs {
    /// The server to benchmark.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub(crate) url: String,
    /// A built-in (share, smoke, speed, daily, quality, quant, long, reasoning,
    /// full) or saved profile. The baseline always runs first (once per server).
    #[arg(long, conflicts_with = "panels")]
    pub(crate) profile: Option<String>,
    /// Collect leased fabric samples (profiles enable this by default).
    #[arg(long, action = clap::ArgAction::Set)]
    pub(crate) fabric: Option<bool>,
    /// Explicit panels instead of a profile, e.g. tool_eval,prefill.
    #[arg(long, value_delimiter = ',')]
    pub(crate) panels: Option<Vec<String>>,
    /// Passes per panel, e.g. tool_eval=3 (the only way to repeat a panel).
    #[arg(long)]
    pub(crate) passes: Option<String>,
    /// Cases for the spec_lossless_set panel (JSON: [{id, prompt, nonce}]); default: the shipped
    /// scripts/bench/fixtures/spec-lossless-set-v1.json.
    #[arg(long, value_name = "FILE")]
    pub(crate) lossless_set: Option<PathBuf>,
    /// Dump full-vocabulary log-prob rows of both arms for these spec_lossless_set cases over
    /// generated tokens START..=END: CASE[,CASE...]:START-END. Needs a server launched with
    /// PROBE_DUMP_ROOT; the panel's pass names the run directory and its manifest.json.
    #[arg(long, value_name = "CASES:START-END", value_parser = cuteafd_bench::panels::lossless::RowDump::parse)]
    pub(crate) lossless_dump: Option<cuteafd_bench::panels::lossless::RowDump>,
    /// Shared base context lengths for context_load (256..1048576 tokens).
    #[arg(long, value_parser = cuteafd_bench::panels::context_load::parse_lengths_arg)]
    pub(crate) context_lengths: Option<String>,
    /// Decode concurrency levels for context_load (distinct values in 1..256).
    #[arg(long, value_parser = cuteafd_bench::panels::context_load::parse_levels_arg)]
    pub(crate) context_levels: Option<String>,
    /// Exports to write: svg (report, card, panels), png (report, card), json.
    #[arg(long, value_delimiter = ',', default_value = "svg,png,json")]
    pub(crate) export: Vec<String>,
    /// Output directory (default benchmarks/<family>/<date>-<profile>-<hardware>/ under --root).
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
    #[arg(long, default_value = ".")]
    pub(crate) root: PathBuf,
    /// Bearer key for the server's benchmark controls.
    #[arg(long, env = "CUTEAFD_API_KEY", hide_env_values = true)]
    pub(crate) api_key: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct PublishArgs {
    /// Repository root (holds README.md and benchmarks/).
    #[arg(long, default_value = ".")]
    pub(crate) root: PathBuf,
    /// The published matrix: which checkpoint is current in each reference
    /// column, and why a cell publishes nothing else. Defaults to
    /// `<root>/benchmarks/current-matrix.json`.
    #[arg(long)]
    pub(crate) matrix: Option<PathBuf>,
    /// Report directories just placed (checked); every placed report is indexed.
    pub(crate) dirs: Vec<PathBuf>,
}

pub(crate) fn run(args: BenchArgs) -> Result<()> {
    match args.action {
        Some(BenchAction::Fidelity(args)) => {
            match cuteafd_bench::fidelity_cli::execute(args) {
                Ok(true) => Ok(()),
                Ok(false) => std::process::exit(3),
                Err(error) => { eprintln!("fidelity: {error:#}"); std::process::exit(1); }
            }
        }
        Some(BenchAction::Publish(publish)) => {
            let matrix = publish.matrix.map_or_else(
                || publish.root.join(cuteafd_bench::publish::MATRIX_FILE), |path| path);
            let (readme, index, count) =
                cuteafd_bench::publish::publish(&publish.root, &publish.dirs, &matrix)?;
            eprintln!("{count} reports: updated {} and {}", readme.display(), index.display());
            Ok(())
        }
        Some(BenchAction::Smoke(smoke)) => cuteafd_bench::smoke::run(smoke),
        Some(BenchAction::Cancel { url, api_key }) => cuteafd_bench::cli::cancel(&url, &api_key),
        Some(BenchAction::Export { json, out, root, export }) => {
            let report: cuteafd_bench::report::Report = serde_json::from_str(&std::fs::read_to_string(&json)?)?;
            // The exports re-render under the rule in force now (an nvidia NVFP4
            // card's absolute fidelity is informational, not a gate), so the
            // report is not reconciled here.
            let out = out.unwrap_or_else(|| cuteafd_bench::cli::default_dir(&root, &report));
            for path in cuteafd_bench::cli::write_exports(&report, &out, &export)? {
                eprintln!("wrote {}", path.display());
            }
            Ok(())
        }
        None => {
            let run = args.run;
            let passes = match &run.passes {
                Some(text) => cuteafd_bench::profiles::parse_passes(text).map_err(anyhow::Error::msg)?,
                None => Vec::new(),
            };
            let lossless_set = run.lossless_set.as_deref().map(cuteafd_bench::panels::lossless::CaseSet::load).transpose()?;
            let options = cuteafd_bench::cli::RunOptions { url: run.url, profile: run.profile, panels: run.panels,
                passes, baseline: None, fabric: run.fabric, export: run.export, out: run.out, label: None, root: run.root, api_key: run.api_key, quiet: false, deadline: None,
                lossless_set, lossless_dump: run.lossless_dump, context_lengths: run.context_lengths.as_deref().map(cuteafd_bench::sweep::parse_lengths).transpose().map_err(anyhow::Error::msg)?,
                context_levels: run.context_levels.as_deref().map(cuteafd_bench::sweep::parse_levels).transpose().map_err(anyhow::Error::msg)? };
            let (report, dir) = cuteafd_bench::cli::run(&options)?;
            println!("{}", dir.display());
            if report.quality_failed() {
                anyhow::bail!("the quality gate failed (exports are watermarked)");
            }
            Ok(())
        }
    }
}

/// Serve commands and their family ids.
fn family_of(command: &str) -> Option<&'static str> {
    crate::commands::family::FAMILIES.iter().find(|(_, serve, _)| *serve == command).map(|(id, _, _)| *id)
}

/// Records a serve command's resolved options (value, default, source) for
/// the benchmark; other commands record nothing.
pub(crate) fn capture(matches: &ArgMatches, coordinator_budget_gib: Option<f64>) -> Result<()> {
    if let Some(context) = serve_context(matches, coordinator_budget_gib)? {
        cuteafd_bench::context::set(context);
    }
    Ok(())
}

/// The serve command's resolved settings, without publishing them. Tests read
/// this directly: `capture` stores one process-wide context, so a test that
/// called it would replace whatever another test was asserting.
fn serve_context(matches: &ArgMatches, coordinator_budget_gib: Option<f64>)
    -> Result<Option<cuteafd_bench::context::ServerContext>> {
    let Some((name, sub)) = matches.subcommand() else { return Ok(None) };
    let Some(family) = family_of(name) else { return Ok(None) };
    let command = crate::cli::Cli::command();
    let Some(definition) = command.find_subcommand(name) else { return Ok(None) };
    let mut settings: Vec<Setting> = Vec::new();
    let mut snapshot = None;
    for arg in definition.get_arguments() {
        let id = arg.get_id().as_str();
        if matches!(id, "help" | "version" | "coordinator_gpu_budget_gib") {
            continue;
        }
        let value = sub.get_raw(id).map(|values| values.map(|v| v.to_string_lossy().into_owned())
            .collect::<Vec<_>>().join(","));
        let source = match sub.value_source(id) {
            Some(clap::parser::ValueSource::CommandLine) => "cli",
            Some(clap::parser::ValueSource::EnvVariable) => "env",
            Some(clap::parser::ValueSource::DefaultValue) => "default",
            _ => "unset",
        };
        if value.is_none() && source == "unset" {
            continue;
        }
        let mut defaults: Vec<String> = arg.get_default_values().iter().map(|v| v.to_string_lossy().into_owned())
            .collect();
        if defaults.is_empty() {
            // Flags default through their action.
            match arg.get_action() {
                clap::ArgAction::SetTrue => defaults.push("false".into()),
                clap::ArgAction::SetFalse => defaults.push("true".into()),
                _ => {}
            }
        }
        let name = arg.get_long().map(str::to_string).unwrap_or_else(|| id.replace('_', "-"));
        if name == "snapshot" {
            snapshot = value.as_ref().map(PathBuf::from);
        }
        settings.push(Setting { name, value, default: (!defaults.is_empty()).then(|| defaults.join(",")),
            source: source.to_string() });
    }
    // Global options are not in the unbuilt family definition. Pass the
    // resolved ceiling explicitly so generic serve's reparse retains it too.
    if let Some(gib) = coordinator_budget_gib {
        settings.push(Setting { name: "coordinator-gpu-budget-gib".into(), value: Some(gib.to_string()),
            default: None, source: "cli".into() });
    }
    // The parsed flag already applies CUTEAFD_ATTENTION_PLACEMENT (cli over env over
    // default). Report that resolution once; the launch spelling is the same option.
    let requested = sub.try_get_one::<String>("attention_placement").ok().flatten().map(String::as_str)
        .map(str::to_owned)
        .or_else(|| std::env::var("CUTEAFD_ATTENTION_PLACEMENT").ok())
        .unwrap_or_else(|| "auto".into());
    // Reject an unrunnable placement before loading CUDA or starting workers.
    let mode = attention_mode(family, sub, &requested)?;
    let executor = cuteafd_loader::placement::families::executor(family).expect("serve family executor");
    settings.retain(|setting| setting.name != "attention-placement");
    settings.push(Setting { name: "attention-placement".into(), value: Some(mode.to_string()),
        default: Some(executor.attention_default().to_string()), source: "resolved".into() });
    settings.extend(cuteafd_bench::context::env_settings().into_iter()
        .filter(|setting| setting.name != "CUTEAFD_ATTENTION_PLACEMENT"));
    Ok(Some(cuteafd_bench::context::ServerContext { command: name.to_string(),
        family: Some(family.to_string()), snapshot, settings }))
}

/// The attention placement `requested` resolves to for a serve command: two coordinator GPUs
/// are generic serve's `--rtx-gpus 2` or a family's `--split-device`.
fn attention_mode(family: &str, sub: &ArgMatches, requested: &str)
    -> Result<cuteafd_loader::placement::AttentionPlacement> {
    let requested = cuteafd_loader::placement::attention::parse(requested).map_err(anyhow::Error::msg)?;
    let executor = cuteafd_loader::placement::families::executor(family).expect("serve family executor");
    let peer = sub.try_get_one::<u32>("rtx_gpus").ok().flatten().is_some_and(|&gpus| gpus == 2)
        || sub.try_get_one::<i32>("split_device").ok().flatten().is_some();
    Ok(executor.check_attention(requested, if peer { 2 } else { 1 }, peer)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::FromArgMatches;

    #[test]
    fn context_load_arguments_parse_and_validate() {
        let args = |lengths: &str, levels: &str| crate::cli::Cli::command().try_get_matches_from([
            "cuteafd", "bench", "--panels", "context_load", "--context-lengths", lengths, "--context-levels", levels]);
        let parsed = args("32768,131072", "1,8,16").unwrap();
        let bench = BenchArgs::from_arg_matches(parsed.subcommand().unwrap().1).unwrap();
        assert_eq!(bench.run.context_lengths.as_deref(), Some("32768,131072"));
        assert_eq!(bench.run.context_levels.as_deref(), Some("1,8,16"));
        for (l, c) in [("255", "1"), ("1048577", "1"), ("abc", "1"), ("32768", "0"), ("32768", "8,8")] {
            assert!(args(l, c).is_err());
        }
    }

    #[test]
    fn a_split_device_admits_two_gpu_attention_placements() {
        use cuteafd_loader::placement::AttentionPlacement::{Context, Heads};
        let serve = |extra: &[&str]| {
            let argv = ["cuteafd", "serve-glm", "--snapshot", "/s", "--native-lib", "/l"].into_iter()
                .chain(extra.iter().copied()).collect::<Vec<_>>();
            crate::cli::Cli::command().try_get_matches_from(argv).unwrap()
        };
        let split = serve(&["--split-device", "1"]);
        let (_, sub) = split.subcommand().unwrap();
        assert_eq!(attention_mode("glm5", sub, "context").unwrap(), Context);
        assert_eq!(attention_mode("glm5", sub, "auto").unwrap(), Context);
        assert_eq!(attention_mode("glm5", sub, "heads").unwrap(), Heads);
        let one = serve(&[]);
        let (_, sub) = one.subcommand().unwrap();
        assert!(attention_mode("glm5", sub, "context").is_err());
        assert_eq!(attention_mode("glm5", sub, "auto").unwrap(), Heads);
    }

    #[test]
    fn serve_options_are_captured_with_defaults_and_sources() {
        let argv = ["cuteafd", "--coordinator-gpu-budget-gib", "32", "serve-qwen4", "--snapshot", "/hub/models--Qwen--Q/snapshots/abc", "--native-lib",
            "/opt/lib.so", "--max-sequences", "2"];
        let matches = crate::cli::Cli::command().try_get_matches_from(argv).unwrap();
        let cli = crate::cli::Cli::from_arg_matches(&matches).unwrap();
        capture(&matches, cli.coordinator_gpu_budget_gib).unwrap();
        let context = cuteafd_bench::context::get();
        assert_eq!(context.family.as_deref(), Some("qwen4"));
        assert_eq!(context.snapshot.as_deref(), Some(std::path::Path::new("/hub/models--Qwen--Q/snapshots/abc")));
        let find = |name: &str| context.settings.iter().find(|s| s.name == name).cloned()
            .unwrap_or_else(|| panic!("{name} missing"));
        let budget = find("coordinator-gpu-budget-gib");
        assert_eq!((budget.value.as_deref(), budget.default.as_deref(), budget.source.as_str()),
            (Some("32"), None, "cli"));
        assert_eq!(context.settings.iter().filter(|s| s.name == budget.name).count(), 1);
        let sequences = find("max-sequences");
        assert_eq!((sequences.value.as_deref(), sequences.default.as_deref(), sequences.source.as_str()),
            (Some("2"), Some("4"), "cli"));
        assert!(sequences.differs());
        let output = find("max-output");
        assert_eq!(output.source, "default");
        assert!(!output.differs());
        assert!(!context.settings.iter().any(|s| s.name == "no-copy-drafts"));
    }

    #[test]
    fn attention_placement_is_reported_once_as_the_resolved_value() {
        let serve = |extra: &[&str]| {
            let argv = ["cuteafd", "serve-glmf", "--snapshot", "/s", "--native-lib", "/l"].into_iter()
                .chain(extra.iter().copied()).collect::<Vec<_>>();
            crate::cli::Cli::command().try_get_matches_from(argv).unwrap()
        };
        let placement = |context: &cuteafd_bench::context::ServerContext| context.settings.iter()
            .filter(|s| s.name == "attention-placement" || s.name == "CUTEAFD_ATTENTION_PLACEMENT")
            .cloned().collect::<Vec<_>>();
        let split = serve_context(&serve(&["--split-device", "1", "--attention-placement", "context"]), None)
            .unwrap().expect("serve context");
        let placed = placement(&split);
        assert_eq!(placed.len(), 1, "{placed:?}");
        assert_eq!((placed[0].value.as_deref(), placed[0].default.as_deref(), placed[0].source.as_str()),
            (Some("context"), Some("context"), "resolved"));
        let one = serve_context(&serve(&[]), None).unwrap().expect("serve context");
        let placed = placement(&one);
        assert_eq!(placed.len(), 1, "{placed:?}");
        assert_eq!(placed[0].value.as_deref(), Some("heads"));
    }
}
