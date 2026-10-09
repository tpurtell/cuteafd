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
                glmf_index: args.index_cache.into(),
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

pub(crate) fn run_plan(mut args: PlanArgs) -> Result<()> {
    if args.files || args.fetch || args.role.is_some() {
        let cfg = launch_config(args.config.as_deref())?;
        if args.revision.is_none() && !PathBuf::from(&args.model).is_dir() {
            args.revision = cfg.get("MODEL_REVISION").filter(|v| !v.is_empty()).cloned();
        }
    }
    let mut options = options(&args)?;
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
    if let Some(layout) = &mut options.layout {
        let config = std::fs::File::open(snapshot.join("config.json")).ok()
            .and_then(|file| serde_json::from_reader::<_, serde_json::Value>(file).ok());
        if config.as_ref().and_then(|value| value.get("model_type"))
            .and_then(serde_json::Value::as_str) == Some("deepseek_v41") {
            // Match run.sh's supported worker capacities without changing other families.
            layout.spark_capacity_rows = if args.prefill_rows == 0 { 4096 }
                else if args.prefill_rows <= 80 { 80 }
                else if args.prefill_rows <= 256 { 256 }
                else if args.prefill_rows <= 1024 { 1024 }
                else { 4096 };
        }
    }
    if args.files || args.fetch || args.role.is_some() {
        use cuteafd_loader::plan::files::{manifest_roles, ReadRole};
        let args = file_args(args, &snapshot)?;
        let placement = ExpertPlacement::from_spark_ranks(args.spark_ranks.unwrap_or(4));
        let parse = |name: &str| {
            anyhow::ensure!(!(args.no_speculator && name == "drafter"), "--no-speculator conflicts with drafter role");
            ReadRole::parse(name, placement,
                args.include_speculator || matches!(args.speculator.as_str(), "mtp" | "dspark"))
        };
        let complete_roles = |mut roles: Vec<ReadRole>| {
            if roles.iter().any(|r| matches!(r, ReadRole::Coordinator { .. })) {
                if args.vision != cuteafd_loader::plan::MediaMode::Off { roles.push(ReadRole::Vision); }
                if args.audio != cuteafd_loader::plan::MediaMode::Off { roles.push(ReadRole::Audio); }
            }
            roles
        };
        let roles = if let Some(layout) = &args.file_layout {
            anyhow::ensure!(args.role.is_none(), "--role and --file-layout are mutually exclusive");
            let hosts: std::collections::BTreeMap<String, Vec<String>> =
                serde_json::from_reader(std::fs::File::open(layout)?)?;
            if let Some(host) = &args.host {
                hosts.get(host).with_context(|| format!("host {host} is absent from {}", layout.display()))?
                    .iter().map(|name| parse(name)).collect::<Result<Vec<_>>>()?
            } else {
                let manifests = hosts.iter().map(|(host, names)| {
                    let roles = complete_roles(names.iter().map(|name| parse(name)).collect::<Result<Vec<_>>>()?);
                    let mut manifest = manifest_roles(&snapshot, &roles)?;
                    attach_snapshots(&mut manifest, &roles, &args)?;
                    Ok((host.clone(), manifest))
                }).collect::<Result<std::collections::BTreeMap<_, _>>>()?;
                if args.fetch {
                    let hosts: Vec<_> = manifests.into_iter().collect();
                    for chunk in hosts.chunks(args.fetch_parallel as usize) {
                        std::thread::scope(|scope| -> Result<()> {
                            let handles: Vec<_> = chunk.iter().map(|(host, manifest)| {
                                let mut host_args = args.clone();
                                host_args.host = Some(host.clone());
                                let snapshot = &snapshot;
                                scope.spawn(move || transfer(snapshot, manifest, &host_args))
                            }).collect();
                            for handle in handles { handle.join().map_err(|_| anyhow::anyhow!("host transfer thread panicked"))??; }
                            Ok(())
                        })?;
                    }
                    eprintln!("layout: {} host transfers complete", hosts.len());
                    return Ok(());
                }
                anyhow::ensure!(args.json, "all-host inventory needs --json; select --host for a plain file list");
                println!("{}", serde_json::to_string_pretty(&manifests)?);
                return Ok(());
            }
        } else {
            anyhow::ensure!(args.host.is_none() || args.fetch, "--host needs --file-layout or --fetch");
            vec![parse(args.role.as_deref().unwrap_or("coordinator"))?]
        };
        let roles = complete_roles(roles);
        let mut manifest = manifest_roles(&snapshot, &roles)?;
        attach_snapshots(&mut manifest, &roles, &args)?;
        if args.fetch {
            transfer(&snapshot, &manifest, &args)?;
        } else if args.files {
            if args.json { println!("{}", serde_json::to_string_pretty(&manifest)?); }
            else {
                anyhow::ensure!(manifest.additional_snapshots.is_empty(), "multiple snapshot roots need --json; inventory each root separately for a plain rsync list");
                for file in &manifest.files { println!("{}", file.path); }
            }
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

fn launch_config(path: Option<&std::path::Path>) -> Result<std::collections::BTreeMap<String, String>> {
    let Some(path) = path else { return Ok(Default::default()) };
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(text.lines().filter_map(|line| {
        let (key, value) = line.split_once('=')?;
        (!key.is_empty() && key.bytes().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_'))
            .then(|| (key.to_owned(), value.to_owned()))
    }).collect())
}

/// Resolve the release defaults or the launcher's plain KEY=VALUE config.
/// Never execute a shell configuration while constructing a file inventory.
fn file_args(mut args: PlanArgs, snapshot: &std::path::Path) -> Result<PlanArgs> {
    use cuteafd_loader::plan::MediaMode;
    let inventory = cuteafd_loader::plan::checkpoint::Checkpoint::inventory(snapshot)?;
    let family = cuteafd_loader::plan::family::detect(&inventory).context("unsupported checkpoint family")?.id();
    let cfg = launch_config(args.config.as_deref())?;
    if let Some(model) = cfg.get("MODEL_ID") {
        if !PathBuf::from(&args.model).is_dir() {
            anyhow::ensure!(model == &args.model, "--config MODEL_ID={model} differs from {}", args.model);
        }
    }
    if args.spark_ranks.is_none() {
        if cfg.get("EXPERT_BACKEND").is_some_and(|v| v == "local") { args.spark_ranks = Some(0); }
        else if let Some(ranks) = cfg.get("SPARK_COUNT") { args.spark_ranks = Some(ranks.parse().context("SPARK_COUNT")?); }
    }
    let default = match family {
        "glm5" | "glm5_flash" => "dflash2",
        "mimo_v2" if snapshot.join("dflash").is_dir() => "dflash2",
        "mimo_v2" | "qwen4" => "mtp",
        "deepseek_v4" | "deepseek_v41" => "dspark",
        _ => "off",
    };
    if args.speculator == "auto" {
        args.speculator = cfg.get("SPECULATOR").cloned().unwrap_or_else(|| {
            if cfg.get("DFLASH").is_some_and(|v| v == "on") || cfg.get("DRAFT_MODEL_ID").is_some_and(|v| !v.is_empty()) { "dflash2".into() }
            else if let Some(depth) = cfg.get("MTP") { if depth == "0" { "off" } else { "mtp" }.into() }
            else if let Some(mode) = cfg.get("DSPARK") { if mode == "on" { "dspark" } else { "off" }.into() }
            else if family == "mimo_v2" && cfg.get("DFLASH").is_some_and(|v| v == "off") { "off".into() }
            else { default.into() }
        });
    }
    if args.speculator == "auto" { args.speculator = default.into(); }
    anyhow::ensure!(["off", "mtp", "dspark", "dflash2"].contains(&args.speculator.as_str()), "unknown SPECULATOR={}", args.speculator);
    anyhow::ensure!(!(args.no_speculator && args.include_speculator), "--no-speculator conflicts with --include-speculator");
    if args.no_speculator {
        anyhow::ensure!(args.drafter_snapshot.is_none() && args.role.as_deref() != Some("drafter"), "--no-speculator conflicts with drafter inputs/role");
        args.speculator = "off".into();
    } else if (args.include_speculator || args.drafter_snapshot.is_some()) && args.speculator == "off" {
        args.speculator = default.into();
        anyhow::ensure!(args.speculator != "off", "this family has no default speculator");
    }
    let enabled = match (family, args.speculator.as_str()) {
        (_, "off") | ("glm5", "dflash2") | ("glm5_flash", "dflash2" | "dspark")
        | ("mimo_v2", "dflash2" | "mtp") | ("qwen4", "mtp")
        | ("deepseek_v4" | "deepseek_v41", "dspark") => true,
        _ => false,
    };
    anyhow::ensure!(enabled, "SPECULATOR={} does not apply to {family}", args.speculator);
    let needs_drafter = if let Some(layout) = &args.file_layout {
        let hosts: std::collections::BTreeMap<String, Vec<String>> =
            serde_json::from_reader(std::fs::File::open(layout)?)?;
        let roles = if let Some(host) = &args.host {
            hosts.get(host).with_context(|| format!("host {host} is absent from {}", layout.display()))?.clone()
        } else { hosts.into_values().flatten().collect() };
        roles.iter().any(|r| matches!(r.as_str(), "coordinator" | "rtx0" | "rtx1" | "drafter"))
    } else {
        args.role.as_deref().is_none_or(|r| matches!(r, "coordinator" | "rtx0" | "rtx1" | "drafter"))
    };
    if needs_drafter && args.drafter_snapshot.is_none() && matches!(args.speculator.as_str(), "dflash2" | "dspark") {
        let model = cfg.get("SPECULATOR_MODEL_ID").or_else(|| cfg.get("DRAFT_MODEL_ID"))
            .filter(|v| !v.is_empty()).map(String::as_str).or(match (family, args.speculator.as_str()) {
                ("glm5", "dflash2") => Some("incoai/GLM-5.3-DFlash2"),
                ("glm5_flash", "dflash2") => Some("incoai/GLM-5.3-Flash-DFlash2"),
                ("glm5_flash", "dspark") => Some("RedHatAI/GLM-5.3-Flash-speculator.dspark-preview"),
                _ => None,
            });
        if let Some(model) = model {
            let hf_home = args.hf_home.clone().unwrap_or_else(default_hf_home);
            let revision = cfg.get("SPECULATOR_MODEL_REVISION").or_else(|| cfg.get("DRAFT_MODEL_REVISION"))
                .filter(|v| !v.is_empty()).map(String::as_str);
            args.drafter_snapshot = Some(resolve_snapshot_at_revision(model, Some(&hf_home), revision)?
                .snapshot_path.with_context(|| format!("required speculator {model} is not downloaded; use --no-speculator only when serving without it"))?);
        }
    }
    if let Some(mode) = cfg.get("VISION") { args.vision = mode.parse().map_err(anyhow::Error::msg)?; }
    else if !matches!(family, "mimo_v2" | "glm5_flash" | "qwen4") && args.vision == MediaMode::Auto {
        args.vision = MediaMode::Off;
    }
    if let Some(mode) = cfg.get("AUDIO") { args.audio = mode.parse().map_err(anyhow::Error::msg)?; }
    args.audio = cuteafd_loader::plan::resolve_audio(args.audio, snapshot)?;
    Ok(args)
}

fn attach_snapshots(manifest: &mut cuteafd_loader::plan::files::FileManifest,
    roles: &[cuteafd_loader::plan::files::ReadRole], args: &PlanArgs) -> Result<()> {
    use cuteafd_loader::plan::files::{manifest_standalone, ReadRole};
    let coordinator = roles.iter().any(|r| matches!(r, ReadRole::Coordinator { .. }));
    if (coordinator || roles.contains(&ReadRole::Drafter)) && args.speculator == "dflash2"
        && args.drafter_snapshot.is_none() {
        let root = PathBuf::from(&manifest.snapshot);
        cuteafd_loader::plan::files::include_directory(manifest, &root, "dflash")?;
    }
    let configured = [
        ("drafter", &args.drafter_snapshot, coordinator && args.speculator != "off"
            || roles.iter().any(|r| matches!(r, ReadRole::Drafter | ReadRole::Coordinator { speculator: true, .. }))),
        ("vision", &args.vision_snapshot, roles.contains(&ReadRole::Vision)),
        ("audio", &args.audio_snapshot, roles.contains(&ReadRole::Audio)),
    ];
    for (role, path, enabled) in configured {
        if let Some(path) = path.as_ref().filter(|_| enabled) { manifest.additional_snapshots.push(manifest_standalone(path, role)?); }
    }
    if coordinator {
        let root = PathBuf::from(&manifest.snapshot);
        let inventory = cuteafd_loader::plan::checkpoint::Checkpoint::inventory(&root)?;
        if cuteafd_loader::plan::family::detect(&inventory).is_some_and(|f| f.id() == "glm5_flash") {
            let cfg = launch_config(args.config.as_deref())?;
            let model = cfg.get("GLM5_FLASH_FP8_MODEL_ID").or_else(|| cfg.get("GLMF_FP8_MODEL_ID"))
                .filter(|v| !v.is_empty()).map(String::as_str).unwrap_or("zai-org/GLM-5.3-Flash");
            if model != "off" {
                let hf_home = args.hf_home.clone().unwrap_or_else(default_hf_home);
                let revision = cfg.get("GLM5_FLASH_FP8_MODEL_REVISION").or_else(|| cfg.get("GLMF_FP8_MODEL_REVISION"))
                    .filter(|v| !v.is_empty()).map(String::as_str);
                let source = resolve_snapshot_at_revision(model, Some(&hf_home), revision)?.snapshot_path
                    .with_context(|| format!("required GLM Flash FP8 source {model} is not downloaded"))?;
                if source != root {
                    let mut fp8 = cuteafd_loader::plan::files::manifest(&source,
                        ReadRole::Coordinator { local_experts: false, speculator: false })?;
                    fp8.role = "fp8 projections".into();
                    manifest.additional_snapshots.push(fp8);
                }
            }
        }
    }
    manifest.total_bytes = manifest.additional_snapshots.iter().fold(manifest.total_bytes, |sum, repo|
        sum.and_then(|sum| sum.checked_add(repo.total_bytes?)));
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// One SSH session for the entire file inventory, preserving input order.
fn remote_sizes(host: &str, root: &std::path::Path, files: &[cuteafd_loader::plan::files::RequiredFile]) -> Result<Vec<Option<u64>>> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("ssh").args([host,
        "while IFS= read -r path; do if test -f \"$path\"; then stat -Lc %s -- \"$path\" || exit 1; else printf '%s\\n' '-'; fi; done"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
    let list = files.iter().map(|file| format!("{}\n", root.join(&file.path).display())).collect::<String>();
    child.stdin.take().context("SSH size list stdin")?.write_all(list.as_bytes())?;
    let output = child.wait_with_output()?;
    anyhow::ensure!(output.status.success(), "host {host}: inspecting file sizes failed");
    parse_remote_sizes(host, &String::from_utf8(output.stdout)?, files.len())
}

fn parse_remote_sizes(host: &str, output: &str, count: usize) -> Result<Vec<Option<u64>>> {
    let sizes = output.lines().map(|line| {
        if line == "-" { Ok(None) } else { Ok(Some(line.parse::<u64>()?)) }
    }).collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(sizes.len() == count, "host {host}: expected {count} size results, got {}", sizes.len());
    Ok(sizes)
}

fn source_location(value: Option<&str>, snapshot: &std::path::Path) -> Result<(Option<String>, PathBuf)> {
    let value = value.map(str::to_owned).unwrap_or_else(|| snapshot.display().to_string());
    let (host, path) = if let Some((host, path)) = value.split_once(':') {
        anyhow::ensure!(!host.is_empty() && !host.starts_with('-') && host.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-@".contains(&c)), "invalid source host {host}");
        (Some(host.to_owned()), PathBuf::from(path))
    } else { (None, PathBuf::from(value)) };
    anyhow::ensure!(path.is_absolute() && !path.to_string_lossy().contains(['\n', '\r']), "source needs an absolute snapshot path");
    Ok((host, path))
}

fn transfer(snapshot: &std::path::Path, manifest: &cuteafd_loader::plan::files::FileManifest,
    args: &PlanArgs) -> Result<()> {
    use anyhow::ensure;
    use std::io::Write;
    use std::process::Command;
    ensure!(manifest.additional_snapshots.is_empty(), "fetch separate snapshot roots individually; additional repos are listed in --files --json");
    let automatic = args.source.as_deref() == Some("auto");
    let (source_host, source_path) = source_location(if automatic { None } else { args.source.as_deref() }, snapshot)?;
    if automatic {
        use std::process::Command;
        let local = Command::new("hostname").arg("-s").output().ok()
            .and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_owned());
        let placement = manifest.repo_id.as_ref().and_then(|repo| {
            let selector = format!("hf:{repo}{}", manifest.revision.as_ref().map(|r| format!("@{r}")).unwrap_or_default());
            Command::new("nest").args(["where", &selector, "--json"]).output().ok()
                .filter(|o| o.status.success()).and_then(|o| serde_json::from_slice::<serde_json::Value>(&o.stdout).ok())
        });
        let sealed = snapshot.starts_with("/mnt/sparknest") && placement.as_ref()
            .and_then(|v| v["hosts"].as_array()).is_some_and(|hosts| hosts.iter().any(|host|
                host["ready"] == true && host["host"].as_str() == local.as_deref()));
        if sealed { eprintln!("source auto: sealed local sparknest copy at {}", snapshot.display()); }
        else if snapshot.starts_with("/mnt/sparknest") {
            eprintln!("source auto: no sealed local copy; streaming from sparknest at {}", snapshot.display());
        } else { eprintln!("source auto: using local snapshot {} (no sparknest discovery)", snapshot.display()); }
    }
    let destination = args.destination.as_ref().context("--fetch needs --destination")?;
    ensure!(destination.is_absolute(), "destination must be an absolute snapshot directory");
    let host = args.host.as_deref();
    if let Some(host) = host {
        ensure!(!host.starts_with('-') && host.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-@".contains(&c)),
            "invalid SSH host {host}");
    }
    let available = |program: &str| Command::new("sh").args(["-c", &format!("command -v {program} >/dev/null")])
        .status().is_ok_and(|s| s.success());
    let installed_remote = |host: &str| Command::new("ssh").args([host, "command -v rdmasync >/dev/null"])
        .status().is_ok_and(|s| s.success());
    let remote_rdma = host.is_none_or(installed_remote) && source_host.as_deref().is_none_or(installed_remote);
    let program = if (host.is_some() || source_host.is_some()) && available("rdmasync") && remote_rdma { "rdmasync" } else {
        eprintln!("warning: rdmasync unavailable on both ends; using rsync over SSH/local transport");
        "rsync"
    };
    ensure!(!destination.to_string_lossy().contains(['\n', '\r']), "destination contains a line break");
    for file in &manifest.files {
        ensure!(!file.path.contains(['\n', '\r']) && !std::path::Path::new(&file.path).is_absolute()
            && std::path::Path::new(&file.path).components().all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe file-list path {}", file.path);
    }
    let source_sizes = if let Some(host) = source_host.as_deref() { remote_sizes(host, &source_path, &manifest.files)? }
        else { manifest.files.iter().map(|file| source_path.join(&file.path).metadata().ok()
            .filter(|m| m.is_file()).map(|m| m.len())).collect() };
    for (file, actual) in manifest.files.iter().zip(&source_sizes) {
        ensure!(actual.is_some(), "source host {} lacks {}", source_host.as_deref().unwrap_or("local"), file.path);
        if let Some(expected) = file.bytes { ensure!(*actual == Some(expected), "source {} size {:?}, expected {expected}", file.path, actual); }
    }
    let existing_sizes = if let Some(host) = host { remote_sizes(host, destination, &manifest.files)? }
        else { manifest.files.iter().map(|file| destination.join(&file.path).metadata().ok()
            .filter(|m| m.is_file()).map(|m| m.len())).collect() };
    let mut selected = Vec::new();
    let mut bytes = 0u64;
    for ((file, existing), source_bytes) in manifest.files.iter().zip(existing_sizes).zip(&source_sizes) {
        let source_bytes = source_bytes.context("source size missing after validation")?;
        if args.force || existing != Some(source_bytes) {
            bytes = bytes.checked_add(source_bytes).context("transfer bytes overflow")?;
            selected.push(file);
        }
    }
    let list = selected.iter().map(|file| format!("{}\n", file.path)).collect::<String>();
    let src = if host.is_none() { source_host.as_deref().map_or_else(|| format!("{}/", source_path.display()),
        |peer| format!("{peer}:{}/", source_path.display())) } else { format!("{}/", source_path.display()) };
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
    // rsync cannot copy between two remote endpoints. Run it on the source
    // peer, which pushes to the selected target using its normal SSH access.
    if let Some(peer) = source_host.as_deref().filter(|_| host.is_some()) {
        let remote = std::iter::once(program.to_string()).chain(command.get_args().map(|a| shell_quote(&a.to_string_lossy())))
            .collect::<Vec<_>>().join(" ");
        command = Command::new("ssh");
        if args.forward_agent { command.arg("-A"); }
        command.args([peer, &remote]);
    }
    eprintln!("host {}: {} files, {bytes} bytes", host.unwrap_or("local"), selected.len());
    if args.dry_run {
        let rendered = std::iter::once(command.get_program().to_string_lossy().into_owned()).chain(command.get_args().map(|a| shell_quote(&a.to_string_lossy())))
            .collect::<Vec<_>>().join(" ");
        println!("{rendered}");
        for file in &selected {
            let size = manifest.files.iter().position(|entry| entry.path == file.path)
                .and_then(|index| source_sizes[index]).context("selected source size")?;
            println!("{size}\t{}", file.path);
        }
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
    let actual_sizes = if let Some(host) = host { remote_sizes(host, destination, &manifest.files)? }
        else { manifest.files.iter().map(|file| destination.join(&file.path).metadata().ok()
            .filter(|m| m.is_file()).map(|m| m.len())).collect() };
    for ((file, actual), expected) in manifest.files.iter().zip(actual_sizes).zip(source_sizes) {
        ensure!(actual == expected && actual.is_some(), "host {}: {} size {actual:?}, expected {expected:?}", host.unwrap_or("local"), file.path);
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
            files: false, role: None, host: None, file_layout: None, fetch_parallel: 2, fetch: false, destination: None, source: None, forward_agent: false,
            dry_run: false, force: false, include_speculator: false,
            config: None, speculator: "auto".into(), no_speculator: false,
            drafter_snapshot: None, vision_snapshot: None, audio_snapshot: None,
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
            index_cache: crate::families::glm5_flash::engine::IndexCache::Keys,
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
    fn standalone_repos_are_inventory_only_when_the_host_runs_that_role() {
        use cuteafd_loader::plan::files::{manifest_roles, ReadRole};
        let main = tempfile::tempdir().unwrap();
        write_snapshot(main.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let draft = tempfile::tempdir().unwrap();
        write_snapshot(draft.path(), &serde_json::json!({"model_type":"external_drafter"}),
            &[cuteafd_loader::plan::testing::t("draft.weight", "BF16", &[2])], None);
        let mut args = args(main.path(), 4, false);
        args.drafter_snapshot = Some(draft.path().into());
        let spark = ReadRole::Spark { rank: 0, world: 4 };
        let mut manifest = manifest_roles(main.path(), &[spark]).unwrap();
        attach_snapshots(&mut manifest, &[spark], &args).unwrap();
        assert!(manifest.additional_snapshots.is_empty());
        let coordinator = ReadRole::Coordinator { local_experts: false, speculator: true };
        let mut manifest = manifest_roles(main.path(), &[coordinator]).unwrap();
        attach_snapshots(&mut manifest, &[coordinator], &args).unwrap();
        assert_eq!(manifest.additional_snapshots.len(), 1);
        assert!(manifest.additional_snapshots[0].files.iter().any(|f| f.path == "model-00001-of-00001.safetensors"));
    }

    #[test]
    fn auto_speculators_equal_the_sanitized_release_configs() {
        use cuteafd_loader::plan::testing::*;
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../scripts/fixtures/release-configs");
        let v41 = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cuteafd-loader/src/families/deepseek_v41/official-v41-config.json"))).unwrap();
        let v4 = serde_json::json!({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"});
        for (name, config, expected) in [
            ("glm53-exl3", glm5_config(), "dflash2"),
            ("glm53f-exl3", glm5_flash_config(2), "dflash2"),
            ("mimo26-flash", mimo_flash_mopd_config(), "dflash2"),
            ("qwen38-exl3", qwen4_config(4), "mtp"),
            ("v4-flash", v4, "dspark"), ("v41-flash", v41, "dspark"),
        ] {
            let target = tempfile::tempdir().unwrap();
            write_snapshot(target.path(), &config, &[t("unused.weight", "BF16", &[2])], None);
            if name == "mimo26-flash" { std::fs::create_dir(target.path().join("dflash")).unwrap(); }
            let draft = tempfile::tempdir().unwrap();
            let mut auto = args(target.path(), 4, false);
            // Resolve policy independently of the machine's HF cache.
            if name.starts_with("glm") { auto.drafter_snapshot = Some(draft.path().into()); }
            let automatic = file_args(auto.clone(), target.path()).unwrap();
            auto.config = Some(root.join(format!("{name}.config")));
            let configured = file_args(auto, target.path()).unwrap();
            assert_eq!(automatic.speculator, expected, "{name}");
            assert_eq!(automatic.speculator, configured.speculator, "{name}");
            assert_eq!(automatic.drafter_snapshot, configured.drafter_snapshot);
        }
    }

    #[test]
    fn glm_flash_includes_the_configured_fp8_projection_source() {
        use cuteafd_loader::plan::testing::*;
        let hf = tempfile::tempdir().unwrap();
        let source = hf.path().join("hub/models--fixture--fp8/snapshots/pinned");
        std::fs::create_dir_all(&source).unwrap();
        write_snapshot(&source, &glm5_flash_config(2), &[t("lm_head.weight", "BF16", &[2])], None);
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &glm5_flash_config(2), &[t("lm_head.weight", "BF16", &[2])], None);
        let config = target.path().join("launch.config");
        std::fs::write(&config, "SPECULATOR=off\nGLM5_FLASH_FP8_MODEL_ID=fixture/fp8\nGLM5_FLASH_FP8_MODEL_REVISION=pinned\n").unwrap();
        let mut input = args(target.path(), 4, false);
        input.config = Some(config);
        input.hf_home = Some(hf.path().into());
        let effective = file_args(input, target.path()).unwrap();
        let roles = [cuteafd_loader::plan::files::ReadRole::Coordinator { local_experts: false, speculator: false }];
        let mut manifest = cuteafd_loader::plan::files::manifest_roles(target.path(), &roles).unwrap();
        attach_snapshots(&mut manifest, &roles, &effective).unwrap();
        assert_eq!(manifest.additional_snapshots.len(), 1);
        assert_eq!(manifest.additional_snapshots[0].snapshot, source.display().to_string());
        assert_eq!(manifest.additional_snapshots[0].role, "fp8 projections");
    }

    #[test]
    fn configured_off_is_explicit_but_include_speculator_cannot_be_silenced() {
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let config = target.path().join("serving.config");
        std::fs::write(&config, "SPECULATOR=off\nVISION=off\nAUDIO=off\n").unwrap();
        let mut input = args(target.path(), 4, false);
        input.config = Some(config);
        let off = file_args(input.clone(), target.path()).unwrap();
        assert_eq!(off.speculator, "off");
        input.include_speculator = true;
        assert_eq!(file_args(input.clone(), target.path()).unwrap().speculator, "mtp");
        input.no_speculator = true;
        assert!(file_args(input, target.path()).is_err());
        std::fs::create_dir(target.path().join("dflash")).unwrap();
        let legacy = target.path().join("legacy.config");
        std::fs::write(&legacy, "DFLASH=off\n").unwrap();
        let mut input = args(target.path(), 4, false);
        input.config = Some(legacy);
        assert_eq!(file_args(input, target.path()).unwrap().speculator, "off");
    }

    #[test]
    fn config_local_experts_and_spark_only_layout_match_launcher_requirements() {
        use cuteafd_loader::plan::testing::*;
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let config = target.path().join("launch.config");
        std::fs::write(&config, "EXPERT_BACKEND=local\nSPARK_COUNT=4\nSPECULATOR=off\n").unwrap();
        let mut input = args(target.path(), 4, false);
        input.spark_ranks = None;
        input.config = Some(config);
        let effective = file_args(input, target.path()).unwrap();
        assert_eq!(effective.spark_ranks, Some(0));
        let role = cuteafd_loader::plan::files::ReadRole::parse("coordinator",
            ExpertPlacement::from_spark_ranks(effective.spark_ranks.unwrap()), false).unwrap();
        let manifest = cuteafd_loader::plan::files::manifest(target.path(), role).unwrap();
        assert!(manifest.files.iter().any(|f| f.path.ends_with(".safetensors")));

        let glm = tempfile::tempdir().unwrap();
        write_snapshot(glm.path(), &glm5_config(), &[t("unused.weight", "BF16", &[2])], None);
        let layout = glm.path().join("layout.json");
        std::fs::write(&layout, r#"{"spark0":["spark0"],"head":["coordinator"]}"#).unwrap();
        let mut input = args(glm.path(), 4, false);
        input.file_layout = Some(layout);
        input.host = Some("spark0".into());
        let empty_cache = tempfile::tempdir().unwrap();
        input.hf_home = Some(empty_cache.path().into());
        assert!(file_args(input, glm.path()).unwrap().drafter_snapshot.is_none());
    }

    #[test]
    fn no_speculator_rejects_drafter_in_a_file_layout() {
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let layout = target.path().join("layout.json");
        std::fs::write(&layout, r#"{"worker":["drafter"]}"#).unwrap();
        let mut input = args(target.path(), 4, false);
        input.files = true;
        input.no_speculator = true;
        input.file_layout = Some(layout);
        assert!(run_plan(input).unwrap_err().to_string().contains("conflicts with drafter role"));
    }

    #[test]
    fn bundled_inventory_contains_every_drafter_input_and_off_excludes_it() {
        let target = tempfile::tempdir().unwrap();
        write_snapshot(target.path(), &mimo_flash_config(), &mimo_flash_tensors(), None);
        let draft = target.path().join("dflash");
        std::fs::create_dir(&draft).unwrap();
        for name in ["config.json", "mask_embedding.pt", "model.safetensors.index.json", "dflash_draft_model.safetensors", "dflash.py"] {
            std::fs::write(draft.join(name), b"fixture").unwrap();
        }
        let automatic = file_args(args(target.path(), 4, false), target.path()).unwrap();
        let roles = [cuteafd_loader::plan::files::ReadRole::Coordinator { local_experts: false, speculator: false }];
        let mut manifest = cuteafd_loader::plan::files::manifest_roles(target.path(), &roles).unwrap();
        attach_snapshots(&mut manifest, &roles, &automatic).unwrap();
        for entry in std::fs::read_dir(&draft).unwrap() {
            let relative = format!("dflash/{}", entry.unwrap().file_name().to_str().unwrap());
            assert!(manifest.files.iter().any(|f| f.path == relative), "loader input missing: {relative}");
        }
        let mut input = args(target.path(), 4, false);
        input.no_speculator = true;
        let off = file_args(input, target.path()).unwrap();
        let mut manifest = cuteafd_loader::plan::files::manifest_roles(target.path(), &roles).unwrap();
        attach_snapshots(&mut manifest, &roles, &off).unwrap();
        assert!(!manifest.files.iter().any(|f| f.path.starts_with("dflash/")));
    }

    #[test]
    fn peer_source_paths_are_absolute_and_hosts_cannot_supply_options() {
        assert_eq!(source_location(Some("worker:/models/snapshot"), std::path::Path::new("/unused")).unwrap(),
            (Some("worker".into()), PathBuf::from("/models/snapshot")));
        assert!(source_location(Some("-oProxyCommand=bad:/snapshot"), std::path::Path::new("/unused")).is_err());
        assert!(source_location(Some("worker:relative"), std::path::Path::new("/unused")).is_err());
        assert!(source_location(Some("worker:/line\nbreak"), std::path::Path::new("/unused")).is_err());
    }

    #[test]
    fn batched_size_results_preserve_order_and_reject_partial_or_invalid_output() {
        assert_eq!(parse_remote_sizes("peer", "17\n-\n0\n", 3).unwrap(), vec![Some(17), None, Some(0)]);
        assert!(parse_remote_sizes("peer", "17\n", 2).unwrap_err().to_string().contains("peer"));
        assert!(parse_remote_sizes("peer", "banner\n17\n", 2).is_err());
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
                path: "config.json".into(), bytes: Some(6) }], total_bytes: Some(6), additional_snapshots: Vec::new() };
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
    fn glm_flash_index_cache_reaches_the_layout() {
        use clap::Parser;
        use cuteafd_loader::serving_capacity::GlmfIndexCache;
        let parse = |extra: &[&str]| crate::cli::Cli::try_parse_from(
            ["cuteafd", "plan", "/not-read", "--layout"].into_iter().chain(extra.iter().copied()));
        for (extra, index) in [(&[][..], GlmfIndexCache::Keys),
            (&["--index-cache", "keys"][..], GlmfIndexCache::Keys),
            (&["--index-cache", "compact"][..], GlmfIndexCache::Compact)] {
            let crate::cli::Commands::Plan(args) = parse(extra).unwrap().command else { panic!("plan") };
            assert_eq!(options(&args).unwrap().layout.unwrap().glmf_index, index);
        }
        assert!(parse(&["--index-cache", "auto"]).is_err());
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
        // Implicit state slots now follow serving's max(8, C), including C1 and C16.
        // Explicit mark lanes still cap C128 at 64 instead of the planner's uncapped C lanes.
        assert_eq!(planned(&launcher(1, 0, None).iter().map(String::as_str).collect::<Vec<_>>()), (0, state(8)));
        assert_eq!(planned(&["--concurrency", "1", "--prefix-cache-entries", "0"]), (0, state(8)));
        assert_eq!(planned(&launcher(16, 20, None).iter().map(String::as_str).collect::<Vec<_>>()),
            (34 * mark, state(16)));
        assert_eq!(planned(&["--concurrency", "16"]), (34 * mark, state(16)));
        assert_eq!(planned(&launcher(128, 20, None).iter().map(String::as_str).collect::<Vec<_>>()),
            (130 * mark, state(128)));
        assert_eq!(planned(&["--concurrency", "128"]).0, 258 * mark);
        assert_eq!((258 - 130) * mark, 18_895_339_520);
        // Unset launch keys now plan serving's defaults: 18 marks and 8 state slots.
        assert_eq!(planned(&[]), (18 * mark, state(8)));
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
