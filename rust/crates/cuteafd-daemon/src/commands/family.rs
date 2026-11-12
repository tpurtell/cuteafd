//! One `serve` and one `golden` for every family. The family comes from the
//! checkpoint (`cuteafd_loader::plan::family::detect` on `--snapshot`) or from
//! `--family`; the remaining arguments go to that family's command, which keeps
//! its own options (`cuteafd serve --family glm5 --help`). The per-family
//! commands (`serve-glm`, `glm-golden`, ...) stay as hidden aliases.
use anyhow::{bail, Context, Result};
use clap::CommandFactory;
use std::path::Path;

/// Family id, its serve command and its golden comparison (if any).
pub(crate) const FAMILIES: &[(&str, &str, Option<&str>)] = &[
    ("deepseek_v41", "serve-native", Some("v41-golden")),
    ("deepseek_v4", "serve-dsv4", Some("dsv4-golden")),
    ("glm5", "serve-glm", Some("glm-golden")),
    ("glm5_flash", "serve-glmf", Some("glmf-golden")),
    ("mimo_v2", "serve-mimo", Some("mimo-golden")),
    ("qwen4", "serve-qwen4", Some("qwen4-golden")),
    ("qwen3_5", "serve-qwen35", Some("qwen35-golden")),
    ("kimi_k3", "serve-kimi", Some("kimi-golden")),
    ("step5", "serve-step5", Some("step5-golden")),
    ("muse_spark", "serve-muse", Some("muse-golden")),
];

#[derive(Debug, clap::Args)]
#[command(disable_help_flag = true)]
pub(crate) struct FamilyArgs {
    /// Family id: deepseek_v41, deepseek_v4, glm5, glm5_flash, mimo_v2, qwen4, qwen3_5, kimi_k3, step5 or muse_spark.
    /// Detected from --snapshot's config.json when omitted.
    #[arg(long)]
    pub(crate) family: Option<String>,
    /// The family command's options; `--family ID --help` lists them.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub(crate) args: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Serve,
    Golden,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Serve => "serve",
            Kind::Golden => "golden",
        }
    }
}

fn snapshot_arg(args: &[String]) -> Option<&str> {
    args.iter().enumerate().find_map(|(i, arg)| match arg.strip_prefix("--snapshot") {
        Some("") => args.get(i + 1).map(String::as_str),
        Some(rest) => rest.strip_prefix('='),
        None => None,
    })
}

/// The family id of the checkpoint at `snapshot`.
pub(crate) fn detect(snapshot: &Path) -> Result<&'static str> {
    detect_view(&cuteafd_loader::plan::checkpoint::CheckpointView::select(snapshot,
        cuteafd_loader::plan::checkpoint::CoreView::Original)?)
}

fn detect_view(view: &cuteafd_loader::plan::checkpoint::CheckpointView) -> Result<&'static str> {
    let snapshot = view.snapshot();
    let checkpoint = cuteafd_loader::plan::Checkpoint::inventory_view(view)
        .with_context(|| format!("reading {}", snapshot.display()))?;
    let family = cuteafd_loader::plan::family::detect(&checkpoint)
        .with_context(|| format!("{}: no supported family (run `cuteafd plan` on it)", snapshot.display()))?;
    Ok(family.id())
}

/// The argv of the family command this `serve`/`golden` invocation stands for,
/// or None after printing help.
pub(crate) struct ResolvedFamily {
    pub argv: Vec<String>,
    pub view: Option<cuteafd_loader::plan::checkpoint::CheckpointView>,
}

pub(crate) fn resolve(kind: Kind, args: FamilyArgs) -> Result<Option<ResolvedFamily>> {
    let mut core_view = std::env::var("CUTEAFD_CORE_VIEW").unwrap_or_else(|_| "original".into());
    for (i, arg) in args.args.iter().enumerate() {
        if arg == "--core-view" { core_view = args.args.get(i + 1).context("--core-view requires a value")?.clone(); }
        else if let Some(value) = arg.strip_prefix("--core-view=") { core_view = value.into(); }
    }
    let attention = crate::cli::CoreViewArgs { core_view: core_view.parse()?, ..Default::default() };
    let wants_help = args.args.iter().any(|a| a == "--help" || a == "-h");
    // Detection and the eventual family command consume this same immutable bundle.
    let view = snapshot_arg(&args.args).filter(|_| args.family.is_none())
        .map(|snapshot| attention.select(Path::new(snapshot))).transpose()?;
    if let Some(view) = &view { attention.validate_view(view)?; }
    let family = match (&args.family, snapshot_arg(&args.args)) {
        (Some(family), _) => family.clone(),
        (None, Some(_)) => detect_view(view.as_ref().context("checkpoint view required for family detection")?)?.to_string(),
        (None, None) if wants_help || args.args.is_empty() => {
            let mut cli = crate::cli::Cli::command();
            cli.find_subcommand_mut(kind.name()).expect("subcommand").print_long_help()?;
            println!("\nFamilies:");
            for (id, serve, golden) in FAMILIES {
                let command = match kind {
                    Kind::Serve => Some(*serve),
                    Kind::Golden => *golden,
                };
                if let Some(command) = command {
                    println!("  {id:<14} {command}");
                }
            }
            return Ok(None);
        }
        (None, None) => bail!("cuteafd {}: pass --snapshot DIR (the family is read from it) or --family ID", kind.name()),
    };
    if family == "minimax_m3" {
        anyhow::ensure!(kind == Kind::Golden,
            "MiniMax serving remains refused until full-model qualification");
        return Ok(Some(ResolvedFamily { argv: ["cuteafd", "minimax-golden"].into_iter()
            .map(String::from).chain(args.args).collect(), view }));
    }
    let Some(&(_, serve, golden)) = FAMILIES.iter().find(|(id, _, _)| *id == family) else {
        bail!("unknown family {family}; known: {}",
            FAMILIES.iter().map(|(id, ..)| *id).collect::<Vec<_>>().join(", "));
    };
    let command = match kind {
        Kind::Serve => serve,
        Kind::Golden => golden.with_context(|| format!("family {family} has no golden comparison"))?,
    };
    tracing::info!(family, command, "cuteafd {}", kind.name());
    Ok(Some(ResolvedFamily { argv: ["cuteafd", command].into_iter().map(String::from).chain(args.args).collect(), view }))
}

#[cfg(test)]
fn argv(kind: Kind, args: FamilyArgs) -> Result<Option<Vec<String>>> {
    resolve(kind, args).map(|result| result.map(|resolved| resolved.argv))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(family: Option<&str>, rest: &[&str]) -> FamilyArgs {
        FamilyArgs { family: family.map(String::from), args: rest.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn family_flag_selects_the_command_and_forwards_everything_else() {
        let argv = argv(Kind::Serve, args(Some("glm5_flash"), &["--port", "9"])).unwrap().unwrap();
        assert_eq!(argv, ["cuteafd", "serve-glmf", "--port", "9"]);
        let argv = argv_golden("qwen4");
        assert_eq!(argv[1], "qwen4-golden");
        assert_eq!(argv_golden("deepseek_v41")[1], "v41-golden");
        assert_eq!(argv_golden("step5")[1], "step5-golden");
        assert_eq!(argv_golden("qwen3_5")[1], "qwen35-golden");
        assert_eq!(super::argv(Kind::Serve, args(Some("qwen3_5"), &["--snapshot", "/m"]))
            .unwrap().unwrap(), ["cuteafd", "serve-qwen35", "--snapshot", "/m"]);
        assert_eq!(super::argv(Kind::Serve, args(Some("step5"), &["--snapshot", "/m"]))
            .unwrap().unwrap(), ["cuteafd", "serve-step5", "--snapshot", "/m"]);
        assert!(super::argv(Kind::Serve, args(Some("nope"), &[])).is_err());
    }

    fn argv_golden(family: &str) -> Vec<String> {
        super::argv(Kind::Golden, args(Some(family), &["--golden", "/g"])).unwrap().unwrap()
    }

    #[test]
    fn memory_policy_is_global_and_survives_family_forwarding() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "--gpu-memory-cap", "32GiB", "--family-reserve", "1GiB",
            "serve", "--family", "qwen4", "--snapshot", "/m"]).unwrap();
        assert_eq!((cli.memory.gpu_memory_cap, cli.memory.family_reserve), (Some(32 << 30), Some(1 << 30)));
        let crate::cli::Commands::Serve(args) = cli.command else { panic!("serve") };
        let argv = argv(Kind::Serve, args).unwrap().unwrap();
        let mut argv = argv;
        argv.extend(["--native-lib".into(), "/lib.so".into(), "--gpu-memory-cap".into(), "32GiB".into()]);
        let forwarded = crate::cli::Cli::try_parse_from(argv).unwrap();
        assert_eq!(forwarded.memory.gpu_memory_cap, Some(32 << 30));
        for (_, serve, _) in FAMILIES {
            let mut argv = vec!["cuteafd", *serve, "--snapshot", "/m", "--native-lib", "/lib.so",
                "--gpu-memory-cap", "32GiB"];
            if matches!(*serve, "serve-native" | "serve-mimo" | "serve-glm" | "serve-dsv4") {
                argv.extend(["--peers", "127.0.0.1:9,127.0.0.1:10"]);
            }
            if matches!(*serve, "serve-step5" | "serve-muse") {
                argv.extend(["--peers", "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4"]);
            }
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            assert_eq!(cli.memory.gpu_memory_cap, Some(32 << 30), "{serve}");
        }
    }

    #[test]
    fn resident_weight_budget_survives_generic_family_forwarding() {
        use clap::Parser;
        for (command, kind) in [("serve", Kind::Serve), ("golden", Kind::Golden)] {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "--coordinator-weight-budget-gib", "73",
                command, "--family", "kimi_k3", "--snapshot", "/m"]).unwrap();
            let initial = cli.coordinator_weight_budget_gib;
            let args = match cli.command {
                crate::cli::Commands::Serve(args) | crate::cli::Commands::Golden(args) => args,
                _ => panic!("generic family command"),
            };
            let mut forwarded = argv(kind, args).unwrap().unwrap();
            forwarded.extend(["--native-lib".into(), "/lib.so".into()]);
            if command == "golden" {
                forwarded.extend(["--nll".into(), "--golden".into(), "/g".into(),
                    "--save-logits".into(), "/logits".into()]);
            }
            let family = crate::cli::Cli::try_parse_from(forwarded.clone()).unwrap();
            assert_eq!(family.coordinator_weight_budget_gib.or(initial), Some(73.0));
            forwarded.extend(["--coordinator-weight-budget-gib".into(), "71".into()]);
            let family = crate::cli::Cli::try_parse_from(forwarded).unwrap();
            assert_eq!(family.coordinator_weight_budget_gib.or(initial), Some(71.0));
        }
    }

    #[test]
    fn snapshot_is_found_in_either_spelling() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(snapshot_arg(&a(&["--port", "1", "--snapshot", "/x"])), Some("/x"));
        assert_eq!(snapshot_arg(&a(&["--snapshot=/y"])), Some("/y"));
        assert_eq!(snapshot_arg(&a(&["--snapshots", "/z"])), None);
    }

    #[test]
    fn every_family_command_parses() {
        let cli = crate::cli::Cli::command();
        for (_, serve, golden) in FAMILIES {
            assert!(cli.find_subcommand(serve).is_some(), "{serve}");
            if let Some(golden) = golden {
                assert!(cli.find_subcommand(golden).is_some(), "{golden}");
            }
        }
    }
}
