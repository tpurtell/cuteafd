//! `cuteafd plan`: inspect a checkpoint and report what this build can serve.
use anyhow::{Context, Result};
use cuteafd_loader::plan::{budget_bytes, plan, plan_preferred, render, ExpertPlacement, PlanError, PlanOptions};
use cuteafd_loader::{default_hf_home, resolve_snapshot_at_revision};
use std::path::PathBuf;

use crate::cli::PlanArgs;

/// The planning options `args` name, validated before any checkpoint is read.
fn options(args: &PlanArgs) -> Result<PlanOptions, PlanError> {
    if !args.layout && args.vision_replicas != 1 {
        return Err(PlanError::InvalidOption { option: "--vision-replicas", reason: "requires --layout to describe replica inventory".into() });
    }
    let options = PlanOptions {
        vision: args.vision,
        audio: args.audio,
        placement: ExpertPlacement::from_spark_ranks(args.spark_ranks.unwrap_or(4)),
        spark_budget_bytes: budget_bytes("--spark-budget-gib", args.spark_budget_gib)?,
        coordinator_budget_bytes: budget_bytes("--coordinator-weight-budget-gib", args.coordinator_weight_budget_gib)?
            .min(if args.layout || args.coordinator_gpu_budget_gib.is_some() {
                budget_bytes("--coordinator-gpu-budget-gib", args.coordinator_gpu_budget_gib.unwrap_or(95.5))?
            } else { u64::MAX }),
        layout: args.layout.then(|| -> Result<_, PlanError> {
            if !(1..=2).contains(&args.rtx) {
                return Err(PlanError::InvalidOption { option: "--rtx", reason: "1 or 2 coordinator GPUs".into() });
            }
            Ok(cuteafd_loader::plan::layout::LayoutOptions {
                rtx_bytes: vec![budget_bytes("--coordinator-gpu-budget-gib", args.coordinator_gpu_budget_gib.unwrap_or(95.5))?; args.rtx],
                spark_allocation_budget_bytes: Some(budget_bytes("--spark-budget-gib", args.spark_budget_gib)?),
                pool_tokens: args.pool_tokens,
                vision_replicas: args.vision_replicas as usize,
                host_embedding: args.embedding_placement == Some(crate::shared::token_io::EmbedPlacement::Host),
                force_gpu_embedding: args.embedding_placement == Some(crate::shared::token_io::EmbedPlacement::Gpu),
                local_expert_layers: args.local_expert_layers,
                context_tokens: args.context_tokens,
                prefill_rows: args.prefill_rows,
                full_prefill_logits: args.full_prefill_logits,
                prefill_lanes: args.prefill_lanes,
                glmf_decode_rows: args.decode_rows,
                headroom_bytes: budget_bytes("--headroom-gib", args.headroom_gib)?,
                graph_budget_bytes: args.graph_budget_mib.map(|mib| mib << 20),
                glmf_pool_marks: args.prefix_marks == crate::families::glm5_flash::prefix::PrefixMarks::Pool,
                glmf_shared_replay: args.replay_records == crate::families::glm5_flash::engine::ReplayRecords::Shared,
                concurrency: args.concurrency,
                state_slots: args.state_slots,
                glmf_mark_lanes: args.mark_lanes,
                prefix_slots: args.prefix_slots,
                mimo_prefix_entries: args.prefix_cache_entries,
                mimo_prefix_mark_bytes: args.prefix_cache_mark_mib.checked_mul(1 << 20)
                    .ok_or_else(|| PlanError::InvalidOption { option: "--prefix-cache-mark-mib", reason: "byte budget overflows".into() })?,
                mimo_prefix_draft: args.mimo_prefix_draft,
                mimo_rings: args.mimo_rings,
                draft_sequences: args.draft_sequences,
                draft_context_slots: args.draft_context_slots,
                native_mtp_layers: args.native_mtp_layers,
                workspace_manifest: args.workspace_manifest.clone(),
                drafter_bytes: if args.drafter_gib > 0.0 { budget_bytes("--drafter-gib", args.drafter_gib)? } else { 0 },
                ..Default::default()
            })
        }).transpose()?,
    };
    options.validate()?;
    Ok(options)
}

pub(crate) fn run_plan(args: PlanArgs) -> Result<()> {
    let options = options(&args)?;
    let snapshot = if PathBuf::from(&args.model).is_dir() {
        PathBuf::from(&args.model)
    } else {
        let hf_home = args.hf_home.clone().unwrap_or_else(default_hf_home);
        let resolved = match resolve_snapshot_at_revision(&args.model, Some(&hf_home), args.revision.as_deref()) {
            Ok(resolved) => resolved,
            // A stale main ref: describe the only snapshot present (serving still refuses it).
            Err(error) if args.revision.is_none() => {
                let snapshots = cuteafd_loader::model_cache_dir(&hf_home, &args.model).join("snapshots");
                let present: Vec<PathBuf> = std::fs::read_dir(&snapshots).map(|entries| {
                    entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect()
                }).unwrap_or_default();
                let [only] = present.as_slice() else { return Err(error) };
                eprintln!("note: {error:#}; describing {}", only.display());
                let revision = only.file_name().and_then(|n| n.to_str()).context("snapshot name")?.to_owned();
                resolve_snapshot_at_revision(&args.model, Some(&hf_home), Some(&revision))?
            }
            Err(error) => return Err(error),
        };
        resolved
            .snapshot_path
            .with_context(|| format!("no snapshot of {} under {}", args.model, hf_home.display()))?
    };
    if args.files || args.fetch || args.role.is_some() {
        use cuteafd_loader::plan::files::{manifest_roles, ReadRole};
        let parse = |name: &str| ReadRole::parse(name, options.placement, args.include_speculator);
        let roles = if let Some(layout) = &args.file_layout {
            anyhow::ensure!(args.role.is_none(), "--role and --file-layout are mutually exclusive");
            let hosts: std::collections::BTreeMap<String, Vec<String>> =
                serde_json::from_reader(std::fs::File::open(layout)?)?;
            if let Some(host) = &args.host {
                hosts.get(host).with_context(|| format!("host {host} is absent from {}", layout.display()))?
                    .iter().map(|name| parse(name)).collect::<Result<Vec<_>>>()?
            } else {
                anyhow::ensure!(!args.fetch, "--file-layout --fetch needs --host");
                let manifests = hosts.iter().map(|(host, names)| {
                    let roles = names.iter().map(|name| parse(name)).collect::<Result<Vec<_>>>()?;
                    Ok((host.clone(), manifest_roles(&snapshot, &roles)?))
                }).collect::<Result<std::collections::BTreeMap<_, _>>>()?;
                anyhow::ensure!(args.json, "all-host inventory needs --json; select --host for a plain file list");
                println!("{}", serde_json::to_string_pretty(&manifests)?);
                return Ok(());
            }
        } else {
            anyhow::ensure!(args.host.is_none() || args.fetch, "--host needs --file-layout or --fetch");
            vec![parse(args.role.as_deref().unwrap_or("coordinator"))?]
        };
        let manifest = manifest_roles(&snapshot, &roles)?;
        if args.fetch {
            transfer(&snapshot, &manifest, &args)?;
        } else if args.files {
            if args.json { println!("{}", serde_json::to_string_pretty(&manifest)?); }
            else { for file in &manifest.files { println!("{}", file.path); } }
        } else {
            let mut headers = 0;
            for role in &roles { headers += cuteafd_loader::plan::files::open_role(&snapshot, *role)?.tensors.len(); }
            if args.json { println!("{}", serde_json::to_string_pretty(&manifest)?); }
            else { println!("role {} HEADERS READY: {headers} tensor headers, {} required files",
                manifest.role, manifest.files.len()); }
        }
        return Ok(());
    }
    let report = if args.spark_ranks.is_some() { plan(&snapshot, &options)? }
        else { plan_preferred(&snapshot, &options)? };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render(&report));
        if let Some(layout) = &report.memory_layout {
            print!("\nmemory layout (planner)\n{}", layout.render());
        }
    }
    if args.require_ready && !report.executable() {
        anyhow::bail!("{} is not servable by this build", args.model);
    }
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn transfer(snapshot: &std::path::Path, manifest: &cuteafd_loader::plan::files::FileManifest,
    args: &PlanArgs) -> Result<()> {
    use anyhow::ensure;
    use std::io::Write;
    use std::process::Command;
    let destination = args.destination.as_ref().context("--fetch needs --destination")?;
    ensure!(destination.is_absolute(), "destination must be an absolute snapshot directory");
    let host = args.host.as_deref();
    if let Some(host) = host {
        ensure!(!host.starts_with('-') && host.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-@".contains(&c)),
            "invalid SSH host {host}");
    }
    let available = |program: &str| Command::new("sh").args(["-c", &format!("command -v {program} >/dev/null")])
        .status().is_ok_and(|s| s.success());
    let remote_rdma = host.is_none() || Command::new("ssh").args([
        host.unwrap_or_default(), "command -v rdmasync >/dev/null"])
        .status().is_ok_and(|s| s.success());
    let program = if host.is_some() && available("rdmasync") && remote_rdma { "rdmasync" } else {
        eprintln!("warning: rdmasync unavailable on both ends; using rsync over SSH/local transport");
        "rsync"
    };
    let mut selected = Vec::new();
    let mut bytes = 0u64;
    for file in &manifest.files {
        ensure!(!file.path.contains(['\n', '\r']) && !std::path::Path::new(&file.path).is_absolute()
            && std::path::Path::new(&file.path).components().all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe file-list path {}", file.path);
        let source_bytes = file.bytes.with_context(|| format!("source {} lacks {}", snapshot.display(), file.path))?;
        let target = destination.join(&file.path);
        let existing = if let Some(host) = host {
            let output = Command::new("ssh").args([host, &format!("stat -Lc %s -- {} 2>/dev/null || true",
                shell_quote(&target.display().to_string()))]).output()?;
            ensure!(output.status.success(), "host {host}: failed to inspect {}", file.path);
            String::from_utf8(output.stdout)?.trim().parse::<u64>().ok()
        } else { target.metadata().ok().filter(|m| m.is_file()).map(|m| m.len()) };
        if args.force || existing != Some(source_bytes) {
            bytes = bytes.checked_add(source_bytes).context("transfer bytes overflow")?;
            selected.push(file);
        }
    }
    let list = selected.iter().map(|file| format!("{}\n", file.path)).collect::<String>();
    let src = format!("{}/", snapshot.display());
    let dest = host.map_or_else(|| format!("{}/", destination.display()),
        |host| format!("{host}:{}/", destination.display()));
    let list_arg = "--files-from=-";
    let mut command = Command::new(program);
    // -L materializes HF blob symlinks as ordinary snapshot files accepted by
    // every loader. Never retain links pointing outside the destination cache.
    command.args(["-aL", "--info=progress2", "--protect-args", list_arg]);
    if args.force { command.arg("--ignore-times"); }
    else { command.arg("--size-only"); }
    command.args([&src, &dest]);
    eprintln!("host {}: {} files, {bytes} bytes", host.unwrap_or("local"), selected.len());
    if args.dry_run {
        let rendered = std::iter::once(program.to_string()).chain(command.get_args().map(|a| shell_quote(&a.to_string_lossy())))
            .collect::<Vec<_>>().join(" ");
        println!("{rendered}");
        for file in &selected { println!("{}\t{}", file.bytes.unwrap_or_default(), file.path); }
        return Ok(());
    }
    if host.is_none() { std::fs::create_dir_all(destination)?; }
    if let Some(host) = host {
        let status = Command::new("ssh").args([host, &format!("mkdir -p -- {}", shell_quote(&destination.display().to_string()))]).status()?;
        ensure!(status.success(), "host {host}: cannot create snapshot destination");
    }
    let mut child = command.stdin(std::process::Stdio::piped()).spawn()
        .with_context(|| format!("starting {program} for host {}", host.unwrap_or("local")))?;
    child.stdin.take().context("transfer list stdin")?.write_all(list.as_bytes())?;
    let status = child.wait()?;
    ensure!(status.success(), "host {} transfer failed ({status}); files: {:?}", host.unwrap_or("local"),
        selected.iter().map(|f| &f.path).collect::<Vec<_>>());
    for file in &manifest.files {
        let target = destination.join(&file.path);
        let actual = if let Some(host) = host {
            let output = Command::new("ssh").args([host, &format!("stat -Lc %s -- {}", shell_quote(&target.display().to_string()))]).output()?;
            ensure!(output.status.success(), "host {host}: missing {} after copy", file.path);
            String::from_utf8(output.stdout)?.trim().parse::<u64>()?
        } else { target.metadata().with_context(|| format!("local: missing {} after copy", file.path))?.len() };
        ensure!(Some(actual) == file.bytes, "host {}: {} size {actual}, expected {:?}", host.unwrap_or("local"), file.path, file.bytes);
    }
    if destination.parent().and_then(|p| p.file_name()).is_some_and(|name| name == "snapshots") {
        let refs = destination.parent().and_then(|p| p.parent()).context("HF cache root")?.join("refs");
        let revision = destination.file_name().context("snapshot revision")?.to_string_lossy();
        if let Some(host) = host {
            let reference = shell_quote(&refs.join("main").display().to_string());
            let script = format!("mkdir -p -- {} && {{ test -e {reference} || (set -C; printf '%s\\n' {} > {reference}); }}",
                shell_quote(&refs.display().to_string()), shell_quote(&revision));
            ensure!(Command::new("ssh").args([host, &script]).status()?.success(), "host {host}: cannot create refs/main");
        } else {
            std::fs::create_dir_all(&refs)?;
            match std::fs::OpenOptions::new().write(true).create_new(true).open(refs.join("main")) {
                Ok(mut file) => writeln!(file, "{revision}")?,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::plan::testing::{mimo_flash_config, mimo_flash_tensors, mimo_pro_config, mimo_pro_tensors,
        write_snapshot};

    fn args(model: &std::path::Path, spark_ranks: usize, require_ready: bool) -> PlanArgs {
        PlanArgs {
            vision_replicas: 1,
            vision: cuteafd_loader::plan::MediaMode::Auto,
            audio: cuteafd_loader::plan::MediaMode::Off,
            model: model.display().to_string(),
            embedding_placement: Some(crate::shared::token_io::EmbedPlacement::Gpu),
            revision: None,
            hf_home: None,
            spark_ranks: Some(spark_ranks),
            spark_budget_gib: 100.0,
            coordinator_weight_budget_gib: 80.0,
            json: true,
            files: false, role: None, host: None, file_layout: None, fetch: false, destination: None,
            dry_run: false, force: false, include_speculator: false,
            require_ready,
            layout: true,
            rtx: 2,
            coordinator_gpu_budget_gib: None,
            pool_tokens: None,
            drafter_gib: 0.0,
            local_expert_layers: None,
            context_tokens: 262144,
            prefill_rows: 4096,
            full_prefill_logits: false,
            prefill_lanes: 0,
            decode_rows: 64,
            headroom_gib: 2.0,
            graph_budget_mib: None,
            replay_records: crate::families::glm5_flash::engine::ReplayRecords::Own,
            concurrency: 8,
            state_slots: None,
            mark_lanes: None,
            prefix_slots: None,
            prefix_cache_entries: 20,
            prefix_cache_mark_mib: 2048,
            mimo_prefix_draft: false,
            mimo_rings: 16,
            draft_sequences: 4,
            draft_context_slots: None,
            prefix_marks: crate::families::glm5_flash::prefix::PrefixMarks::Arena,
            native_mtp_layers: 3,
            workspace_manifest: None,
        }
    }

    #[test]
    fn files_cli_and_local_transfer_skip_matching_sizes_without_deleting() {
        use clap::Parser;
        let source = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        std::fs::write(source.path().join("config.json"), b"config").unwrap();
        std::fs::write(destination.path().join("keep.txt"), b"keep").unwrap();
        let manifest = cuteafd_loader::plan::files::FileManifest {
            role: "coordinator".into(), snapshot: source.path().display().to_string(),
            repo_id: None, revision: None, files: vec![cuteafd_loader::plan::files::RequiredFile {
                path: "config.json".into(), bytes: Some(6) }], total_bytes: Some(6) };
        let mut args = args(source.path(), 4, false);
        args.fetch = true;
        args.destination = Some(destination.path().into());
        if std::process::Command::new("rsync").arg("--version").output().is_ok() {
            transfer(source.path(), &manifest, &args).unwrap();
            assert_eq!(std::fs::read(destination.path().join("config.json")).unwrap(), b"config");
            std::fs::write(destination.path().join("config.json"), b"custom").unwrap();
            transfer(source.path(), &manifest, &args).unwrap();
            assert_eq!(std::fs::read(destination.path().join("config.json")).unwrap(), b"custom");
            args.force = true;
            transfer(source.path(), &manifest, &args).unwrap();
            assert_eq!(std::fs::read(destination.path().join("config.json")).unwrap(), b"config");
            assert_eq!(std::fs::read(destination.path().join("keep.txt")).unwrap(), b"keep");
        }
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--files",
            "--role", "spark0", "--spark-ranks", "4"]).unwrap();
        let crate::cli::Commands::Plan(args) = cli.command else { unreachable!() };
        assert!(args.files && args.role.as_deref() == Some("spark0"));
    }

    #[test]
    fn planner_threads_warm_draft_context_override() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout",
            "--mimo-prefix-draft", "--rings", "20", "--draft-sequences", "8",
            "--draft-context-slots", "25"]).unwrap();
        let crate::cli::Commands::Plan(args) = cli.command else { unreachable!() };
        let layout = options(&args).unwrap().layout.unwrap();
        assert!(layout.mimo_prefix_draft);
        assert_eq!(layout.mimo_rings, 20);
        assert_eq!(layout.draft_sequences, 8);
        assert_eq!(layout.draft_context_slots, Some(25));
    }

    #[test]
    fn media_modes_parse_on_every_command_and_invalid_modes_fail() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--vision", "off", "--audio", "auto"]).unwrap();
        assert_eq!(cli.vision, Some(cuteafd_loader::plan::MediaMode::Off));
        assert_eq!(cli.audio, Some(cuteafd_loader::plan::MediaMode::Auto));
        assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--vision", "bad"]).is_err());
    }

    #[test]
    fn planner_and_glm_flash_share_auto_default_and_explicit_off() {
        use clap::Parser;
        use cuteafd_loader::plan::MediaMode;
        for command in ["plan", "serve-glmf"] {
            for mode in [None, Some("off")] {
                let mut argv = vec!["cuteafd", command, "--snapshot", "/not-read", "--native-lib", "/not-loaded"];
                if command == "plan" { argv = vec!["cuteafd", command, "/not-read"]; }
                if let Some(mode) = mode { argv.extend(["--vision", mode]); }
                let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
                let expected = if mode.is_some() { MediaMode::Off } else { MediaMode::Auto };
                assert_eq!(crate::resolve_vision(cli.vision, None), expected);
                match cli.command {
                    crate::cli::Commands::Plan(args) => assert_eq!(args.vision, MediaMode::Auto),
                    crate::cli::Commands::ServeGlmf(args) => assert_eq!(args.vision, MediaMode::Auto),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn replicas_require_an_explicit_layout_inventory() {
        let mut request = args(std::path::Path::new("/not-read"), 4, false);
        request.layout = false;
        request.vision_replicas = 2;
        assert!(matches!(options(&request), Err(PlanError::InvalidOption { option: "--vision-replicas", .. })));
        request.layout = true;
        assert_eq!(options(&request).unwrap().layout.unwrap().vision_replicas, 2);
    }

    #[test]
    fn rtx_budget_flag_and_legacy_alias_bound_every_layout_gpu() {
        use clap::Parser;
        for flag in ["--coordinator-gpu-budget-gib", "--rtx-budget-gib", "--rtx-gib"] {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout", "--rtx", "2", flag, "32"])
                .unwrap();
            let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan") };
            args.coordinator_gpu_budget_gib = cli.coordinator_gpu_budget_gib;
            let options = options(&args).unwrap();
            assert_eq!(options.layout.unwrap().rtx_bytes, vec![32 << 30; 2]);
            assert_eq!(options.coordinator_budget_bytes, 32 << 30);
        }
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read"]).unwrap();
        let crate::cli::Commands::Plan(plan_args) = cli.command else { panic!("plan") };
        assert!(options(&plan_args).unwrap().layout.is_none());
        for gib in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let dir = tempfile::tempdir().unwrap();
            let error = options(&PlanArgs { coordinator_gpu_budget_gib: Some(gib), ..args(dir.path(), 4, false) }).unwrap_err();
            assert!(matches!(error, PlanError::InvalidOption { option: "--coordinator-gpu-budget-gib", .. }));
        }
    }

    #[test]
    fn unified_budget_matches_aliases_and_serving_capacity() {
        use clap::Parser;
        let parse = |flag: &str, before: bool| {
            let argv = if before {
                vec!["cuteafd", flag, "31.8", "plan", "/not-read", "--layout"]
            } else {
                vec!["cuteafd", "plan", "/not-read", "--layout", flag, "31.8"]
            };
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan") };
            args.coordinator_gpu_budget_gib = cli.coordinator_gpu_budget_gib;
            options(&args).unwrap()
        };
        let expected = parse("--coordinator-gpu-budget-gib", false);
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let expected_report = serde_json::to_value(plan(snapshot.path(), &expected).unwrap()).unwrap();
        for flag in ["--coordinator-gpu-budget-gib", "--rtx-budget-gib", "--rtx-gib"] {
            for before in [false, true] {
                let actual = parse(flag, before);
                assert_eq!(serde_json::to_value(plan(snapshot.path(), &actual).unwrap()).unwrap(), expected_report);
                assert_eq!(actual.layout.unwrap().rtx_bytes, expected.layout.as_ref().unwrap().rtx_bytes);
                assert_eq!(actual.coordinator_budget_bytes, expected.coordinator_budget_bytes);
            }
        }
        let serving = cuteafd_core::serving_capacity::GpuMemoryBudget::from_gib(31.8).unwrap();
        assert_eq!(expected.layout.unwrap().rtx_bytes, vec![serving.0]);
        assert_eq!(crate::cli::deprecated_budget_flags(["--rtx-gib=31.8", "--rtx-budget-gib", "31.8"]),
            vec!["--coordinator-gpu-budget-gib"]);
        assert!(crate::cli::deprecated_budget_flags(["--coordinator-gpu-budget-gib", "31.8"]).is_empty());
    }

    #[test]
    fn weight_budget_remains_a_distinct_cap_with_deprecated_alias() {
        use clap::Parser;
        for flag in ["--coordinator-weight-budget-gib", "--coordinator-budget-gib"] {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout", flag, "10"]).unwrap();
            let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan") };
            let options = options(&args).unwrap();
            assert_eq!(options.coordinator_budget_bytes, 10 << 30);
            assert_eq!(options.layout.unwrap().rtx_bytes, vec![budget_bytes("budget", 95.5).unwrap()]);
        }
        assert_eq!(crate::cli::deprecated_budget_flags(["--coordinator-budget-gib", "10"]),
            vec!["--coordinator-weight-budget-gib"]);
    }

    #[test]
    fn spark_encoder_admission_honors_runtime_budget() {
        let mut request = args(std::path::Path::new("/not-read"), 4, false);
        request.spark_budget_gib = 82.0;
        let options = options(&request).unwrap();
        assert_eq!(options.spark_budget_bytes, 82 << 30);
        assert_eq!(options.layout.unwrap().spark_allocation_budget_bytes, Some(82 << 30));
    }

    #[test]
    fn glm_flash_prefix_marks_reach_the_layout() {
        use clap::Parser;
        let parse = |extra: &[&str]| crate::cli::Cli::try_parse_from(
            ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied()));
        for (extra, pool) in [(&[][..], false), (&["--prefix-marks", "arena"][..], false),
            (&["--prefix-marks", "pool"][..], true)] {
            let crate::cli::Commands::Plan(args) = parse(extra).unwrap().command else { panic!("plan") };
            assert_eq!(options(&args).unwrap().layout.unwrap().glmf_pool_marks, pool);
        }
        assert!(parse(&["--prefix-marks", "host"]).is_err());
    }

    /// The flags the launcher's encoder placement plan passes for GLM 5.3 Flash size its layout as
    /// serve-glmf allocates: its recurrent state holds max(--slots 8, --max-sequences) slots
    /// (`serve.rs`, `engine_args.slots.max(args.max_sequences)`) and its mark arena counts
    /// min(--max-sequences, DECODE_ROWS = 64) lanes (`serve.rs`, `prefix.mark_rule(lanes)`), over the
    /// entries and mark budget it gets. The launcher passes those counts as --state-slots and
    /// --mark-lanes beside --concurrency; without them the planner derives both from --concurrency.
    #[test]
    fn glm_flash_serving_knobs_reach_the_layout() {
        use clap::Parser;
        let mut config = cuteafd_loader::plan::testing::glm5_flash_config(45);
        config["text_config"]["layer_types"] = serde_json::json!((0..45)
            .map(|l| if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
        let cfg = cuteafd_loader::families::glm5_flash::GlmNextConfig::from_hf(&config).unwrap();
        let rank = cuteafd_loader::serving_capacity::glm_flash_cache_geometry(&cfg, 45).unwrap().ranks[0];
        // The standard 45-layer FP32-state geometry: a mark and a sequence's state are 147,619,840 B.
        let (mark, per_sequence) = (rank.retained_mark_bytes, rank.active_state_per_sequence_bytes);
        assert_eq!((mark, per_sequence), (147_619_840, 147_619_840));
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &config, &[], None);
        let model = snapshot.path().display().to_string();
        let layout = |extra: &[&str]| {
            let argv = ["cuteafd", "plan", model.as_str(), "--layout", "--spark-ranks", "4", "--rtx", "1",
                "--rtx-gib", "96", "--pool-tokens", "0"];
            let cli = crate::cli::Cli::try_parse_from(argv.into_iter().chain(extra.iter().copied())).unwrap();
            let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan") };
            options(&args).unwrap()
        };
        // (marks, state) bytes on the GPU.
        let planned = |extra: &[&str]| {
            let memory = plan(std::path::Path::new(&model), &layout(extra)).unwrap().memory_layout.unwrap();
            let group = |name: &str| memory.devices[0].items.iter().filter(|i| i.group == name).map(|i| i.bytes)
                .sum::<u64>();
            (group("marks"), group("state"))
        };
        let state = |slots: u64| rank.fixed_state_bytes + per_sequence * slots + rank.speculative_replay_bytes;
        // serve-glmf's own counts for `sequences`, `entries` and a mark budget (MiB).
        let served = |sequences: u64, entries: u64, mib: u64| {
            let (slots, lanes) = (sequences.max(8), sequences.min(64));
            (cuteafd_core::prefix::mark_slots_for(lanes, entries, mark, mib << 20) * mark, state(slots))
        };
        // The launcher's arguments for CONCURRENCY, PREFIX_CACHE_ENTRIES and PREFIX_CACHE_MARK_MIB.
        let launcher = |sequences: u64, entries: u64, mib: Option<u64>| {
            let mut argv = vec!["--concurrency".to_string(), sequences.to_string(), "--state-slots".into(),
                sequences.max(8).to_string(), "--mark-lanes".into(), sequences.min(64).to_string(),
                "--prefix-cache-entries".into(), entries.to_string()];
            if let Some(mib) = mib { argv.extend(["--prefix-cache-mark-mib".into(), mib.to_string()]); }
            argv
        };
        for (sequences, entries, mib) in [(1, 0, None), (1, 20, None), (5, 6, Some(1971)), (8, 20, None),
            (16, 20, None), (16, 6, Some(1971)), (64, 20, None), (65, 20, None), (128, 20, None), (128, 0, None)] {
            let argv = launcher(sequences, entries, mib);
            let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
            assert_eq!(planned(&argv), served(sequences, entries, mib.unwrap_or(2048)), "{argv:?}");
        }
        // The boundaries, against the planner's own rule over --concurrency alone (C + 2 state slots,
        // C lanes): C1 without entries took 3 slots where serve-glmf keeps 8 (738,099,200 B short), C16
        // took 18 for 16 (295,239,680 B over), and C128 took 258 marks for 130 (18,895,339,520 B over).
        assert_eq!(planned(&launcher(1, 0, None).iter().map(String::as_str).collect::<Vec<_>>()), (0, state(8)));
        assert_eq!(planned(&["--concurrency", "1", "--prefix-cache-entries", "0"]), (0, state(3)));
        assert_eq!(state(8) - state(3), 738_099_200);
        assert_eq!(planned(&launcher(16, 20, None).iter().map(String::as_str).collect::<Vec<_>>()),
            (34 * mark, state(16)));
        assert_eq!(planned(&["--concurrency", "16"]), (34 * mark, state(18)));
        assert_eq!(state(18) - state(16), 295_239_680);
        assert_eq!(planned(&launcher(128, 20, None).iter().map(String::as_str).collect::<Vec<_>>()),
            (130 * mark, state(128)));
        assert_eq!(planned(&["--concurrency", "128"]).0, 258 * mark);
        assert_eq!((258 - 130) * mark, 18_895_339_520);
        // With no keys set the launcher passes none of them: the planner's defaults (8 sequences,
        // 20 entries, 2,048 MiB) give serve-glmf's 18 marks, and work/p0's C + 2 = 10 state slots.
        assert_eq!(planned(&[]), (18 * mark, state(10)));
        // The other knobs serve-glmf gets reach the layout as given.
        let custom = layout(&["--concurrency", "16", "--state-slots", "16", "--mark-lanes", "16",
            "--prefix-cache-entries", "6", "--prefix-cache-mark-mib", "1971", "--replay-records", "shared"]);
        let knobs = custom.layout.as_ref().unwrap();
        assert_eq!((knobs.concurrency, knobs.state_slots, knobs.glmf_mark_lanes, knobs.mimo_prefix_entries,
            knobs.mimo_prefix_mark_bytes, knobs.glmf_shared_replay), (16, Some(16), Some(16), 6, 1971 << 20, true));
        for zero in ["--state-slots", "--mark-lanes"] {
            assert!(crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/not-read", "--layout", zero, "0"]).is_err());
        }
    }

    #[test]
    fn diagnostic_prefill_flag_is_forwarded_to_layout() {
        use clap::Parser;
        for enabled in [false, true] {
            let mut argv = vec!["cuteafd", "plan", "/not-read", "--layout"];
            if enabled { argv.push("--full-prefill-logits"); }
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Commands::Plan(args) = cli.command else { panic!("plan") };
            assert_eq!(options(&args).unwrap().layout.unwrap().full_prefill_logits, enabled);
        }
    }

    #[test]
    fn layout_refuses_full_storage_shortfall_even_when_weights_fit() {
        let snapshot = tempfile::tempdir().unwrap();
        write_snapshot(snapshot.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let tiny = PlanArgs { coordinator_gpu_budget_gib: Some(2.0), ..args(snapshot.path(), 4, true) };
        let mut weight_only = options(&tiny).unwrap();
        weight_only.layout = None;
        assert!(plan(snapshot.path(), &weight_only).unwrap().executable(), "weight inventory fits");
        let report = plan(snapshot.path(), &options(&tiny).unwrap()).unwrap();
        assert!(!report.fits && !report.executable());
        assert!(report.hints.iter().any(|h| h.what.contains("full memory layout")
            && h.what.contains("shortfall") && h.what.contains("bytes")));
        let error = run_plan(tiny).unwrap_err();
        assert!(error.to_string().contains("is not servable"), "{error:#}");
    }

    #[test]
    fn require_ready_exits_by_verdict_and_options_fail_typed() {
        let flash = tempfile::tempdir().unwrap();
        write_snapshot(flash.path(), &mimo_flash_config(), &mimo_flash_tensors(), Some(1));
        let pro = tempfile::tempdir().unwrap();
        write_snapshot(pro.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
        // Complete inventories at packaged worlds are servable.
        run_plan(args(flash.path(), 4, true)).unwrap();
        run_plan(args(pro.path(), 6, true)).unwrap();
        run_plan(args(pro.path(), 0, true)).unwrap();
        // V2.6 Pro has no tp4 MXFP4 layout: a verdict, an error only with --require-ready.
        run_plan(args(pro.path(), 4, false)).unwrap();
        let error = run_plan(args(pro.path(), 4, true)).unwrap_err();
        assert!(error.to_string().contains("is not servable by this build"), "{error:#}");
        // Options that describe no deployment fail before the checkpoint is read.
        for (ranks, budget) in [(5, 100.0), (8, 100.0), (4, f64::NAN), (4, 0.0), (4, -3.0)] {
            let error = run_plan(PlanArgs { spark_budget_gib: budget, ..args(flash.path(), ranks, false) }).unwrap_err();
            assert!(matches!(error.downcast_ref::<PlanError>(), Some(PlanError::InvalidOption { .. })),
                "{ranks} ranks, {budget} GiB: {error:#}");
        }
    }
}
