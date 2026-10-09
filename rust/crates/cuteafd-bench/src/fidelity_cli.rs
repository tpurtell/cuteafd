//! Remote fidelity probes and local paired comparison, without a teacher service.
use crate::fidelity::{compare, compare_full, Run};
use crate::reference::{Fidelity, Reference};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub action: Action,
}
#[derive(Debug, clap::Subcommand)]
pub enum Action {
    /// Score a pinned family reference on the served engine (cold, drafts off).
    Run(RunArgs),
    /// Gate a full precision decision on both decode and prefill with fixed margins.
    CompareFull {
        #[arg(long)]
        a_decode: PathBuf,
        #[arg(long)]
        b_decode: PathBuf,
        #[arg(long)]
        a_prefill: PathBuf,
        #[arg(long)]
        b_prefill: PathBuf,
        #[arg(long, default_value_t = 5000)]
        bootstrap: usize,
        #[arg(long, default_value_t = 20260829)]
        seed: u64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Compare candidate A to checkpoint-precision baseline B, paired by position.
    Compare {
        a: PathBuf,
        b: PathBuf,
        /// Select a path from an automatic Standard report (also accepts standalone Run files).
        #[arg(long, value_parser = ["decode", "prefill"], default_value = "decode")]
        score_path: String,
        #[arg(long)]
        top1_margin: Option<f64>,
        #[arg(long)]
        kl_margin: Option<f64>,
        #[arg(long, default_value_t = 5000)]
        bootstrap: usize,
        #[arg(long, default_value_t = 20260829)]
        seed: u64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}
#[derive(Debug, Clone, clap::Args)]
pub struct RunArgs {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub url: String,
    #[arg(long, value_parser = ["quick", "standard", "full"], default_value = "quick")]
    pub tier: String,
    #[arg(long)]
    pub arm: String,
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long)]
    pub reference: Option<PathBuf>,
    /// Root of hash-sealed first-party image fixtures for media windows.
    #[arg(long)]
    pub media_root: Option<PathBuf>,
    /// Public HF dataset repository; the verified publication is the full-tier default.
    #[arg(long, num_args = 0..=1, default_missing_value = crate::fidelity_dataset::REPOSITORY,
        conflicts_with_all = ["reference", "rows"])]
    pub dataset: Option<String>,
    /// Immutable dataset commit (defaults to the checksum-verified publication).
    #[arg(long, conflicts_with_all = ["reference", "rows"])]
    pub dataset_revision: Option<String>,
    /// Published config (otherwise select the served checkpoint's verified default).
    #[arg(long)]
    pub dataset_config: Option<String>,
    #[arg(long)]
    pub dataset_cache: Option<PathBuf>,
    /// Full-reference directory (rows.json and sealed f16 files), visible to this client.
    #[arg(long, env = "CUTEAFD_FIDELITY_ROWS")]
    pub rows: Option<PathBuf>,
    /// New directory on server-local NVMe, also visible to this client for scoring.
    #[arg(long)]
    pub dump_dir: Option<PathBuf>,
    /// Kernel shape to score; Quick uses decode, Standard automatically scores both admitted paths.
    #[arg(long, value_parser = ["decode", "prefill"], default_value = "decode")]
    pub score_path: String,
    #[arg(long)]
    pub verify_rows: Option<usize>,
    #[arg(long, env = "CUTEAFD_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
}

#[cfg(test)]
fn dataset_source<'a>(args: &'a RunArgs, model: &str) -> Result<Option<(&'a str, &'a str, &'a str)>> {
    dataset_source_with(args, crate::fidelity_dataset::default_publication(model))
}

fn dataset_source_with<'a>(args: &'a RunArgs, publication: Option<(&'a str, &'a str)>) -> Result<Option<(&'a str, &'a str, &'a str)>> {
    let default_full = args.reference.is_none() && args.rows.is_none();
    let repo = args.dataset.as_deref().or(default_full.then_some(crate::fidelity_dataset::REPOSITORY));
    ensure!(args.dataset_revision.is_none() || repo.is_some(),
        "--dataset-revision needs --dataset or the dataset default");
    let Some(repo) = repo else { return Ok(None); };
    let config = args.dataset_config.as_deref().or(publication.map(|p| p.1))
        .context("no published family default; use --dataset-config and --dataset-revision, or --reference")?;
    let commit = args.dataset_revision.as_deref().or_else(|| {
        publication.filter(|p| repo == crate::fidelity_dataset::REPOSITORY && config == p.1).map(|p| p.0)
    }).context("explicit repository/config requires --dataset-revision")?;
    Ok(Some((repo, commit, config)))
}

fn validate_served_reference(reference: &Reference, model: &str, dataset: bool) -> Result<()> {
    let model_level = dataset && crate::fidelity_dataset::same_base_checkpoint(&reference.checkpoint, model);
    ensure!(model_level || reference.models.iter().any(|pattern| crate::reference::glob(pattern, model)),
        "reference does not match served model");
    ensure!(model_level || reference.windows.is_empty() || reference.checkpoint == model,
        "reference checkpoint differs from served checkpoint");
    Ok(())
}

fn request(agent: &ureq::Agent, url: &str, key: &Option<String>, body: &Value) -> Result<Value> {
    let mut request = agent.post(url);
    if let Some(key) = key { request = request.set("authorization", &format!("Bearer {key}")); }
    match request.send_json(body) {
        Ok(response) => Ok(response.into_json()?),
        Err(ureq::Error::Status(code, response)) => bail!("HTTP {code}: {}", response.into_string().unwrap_or_default()),
        Err(error) => Err(error.into()),
    }
}

fn prepare_dump_dir(args: &RunArgs) -> Result<()> {
    if let Some(dir) = &args.dump_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("cannot create --dump-dir {}", dir.display()))?;
        tempfile::NamedTempFile::new_in(dir)
            .with_context(|| format!("--dump-dir {} is not writable", dir.display()))?;
    }
    Ok(())
}

pub fn run(args: &RunArgs) -> Result<Vec<Run>> {
    prepare_dump_dir(args)?;
    ensure!(matches!(args.tier.as_str(), "quick" | "standard" | "full"), "unknown tier");
    ensure!(args.tier == "full" || args.score_path == "decode", "Quick and automatic Standard start with decode");
    let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(900)).build();
    let base = args.url.trim_end_matches('/');
    let models: Value = agent.get(&format!("{base}/v1/models")).call()?.into_json()?;
    let served = models["data"][0]["id"].as_str().context("served checkpoint id")?;
    let status: Value = agent.get(&format!("{base}/v1/bench/status")).call()?.into_json()?;
    let model = status["checkpoint"].as_str().unwrap_or(served);
    let mut model_record = models["data"][0].clone();
    model_record["full_prefill_logits"] = status["full_prefill_logits"].clone();
    model_record["snapshot"] = status["snapshot"].clone();
    let mut runs = vec![run_with(args, model, &model_record, |body| {
        request(&agent, &format!("{base}/v1/bench/probe"), &args.api_key, &body)
    }, |_, _, _| {})?];
    if args.tier == "standard" && model_record["full_prefill_logits"] == true {
        let mut prefill_args = args.clone();
        prefill_args.score_path = "prefill".into();
        runs.push(run_with(&prefill_args, model, &model_record, |body| {
            request(&agent, &format!("{base}/v1/bench/probe"), &args.api_key, &body)
        }, |_, _, _| {})?);
        ensure!(runs[0].engine == runs[1].engine && runs[0].settings == runs[1].settings
            && runs[0].dataset == runs[1].dataset && runs[0].reference_sha256 == runs[1].reference_sha256,
            "Standard scoring paths use different references or server settings");
    }
    let mut output = if args.tier == "standard" { crate::panels::fidelity::record(&runs[0], None) }
        else { json!(runs[0]) };
    if let Some(prefill) = runs.get(1) {
        for (key, value) in crate::panels::fidelity::record(prefill, None).as_object().unwrap() {
            if key.starts_with("prefill") { output[key] = value.clone(); }
        }
    }
    std::fs::write(&args.out, serde_json::to_vec_pretty(&output)?)?;
    Ok(runs)
}

/// Shared runner for remote CLI probes and the dashboard's already-held bench slot.
pub fn run_with(args: &RunArgs, model: &str, model_record: &Value,
    mut probe_request: impl FnMut(Value) -> Result<Value>,
    mut progress: impl FnMut(usize, usize, &Run)) -> Result<Run> {
    prepare_dump_dir(args)?;
    ensure!(matches!(args.tier.as_str(), "quick" | "standard" | "full"), "unknown tier");
    ensure!(args.tier != "quick" || args.score_path == "decode", "Quick must be decode-shaped");
    let agent = ureq::AgentBuilder::new().timeout_read(Duration::from_secs(120)).build();
    let explicit = args.dataset_config.is_some() || args.dataset_revision.is_some() || args.reference.is_some();
    let snapshot = model_record["snapshot"].as_str().map(std::path::Path::new);
    let resolved = crate::fidelity_match::resolve(model, snapshot);
    // Explicit dataset choices override automatic ancestry, including a rejected card.
    let mut selection = if explicit { resolved.ok() } else { Some(resolved?) };
    let publication = selection.as_ref().map(|s| (s.revision.as_str(), s.config.as_str()))
        .or_else(|| explicit.then(|| crate::fidelity_dataset::default_publication(model)).flatten());
    let (reference, digest, mut dataset_identity) = if let Some((repo, commit, config)) = dataset_source_with(args, publication)? {
        crate::fidelity_dataset::ensure_valid_publication(repo, commit, config)?;
        let cache = args.dataset_cache.clone().unwrap_or_else(|| PathBuf::from(
            std::env::var_os("HOME").unwrap_or_default()).join(".cache/cuteafd/fidelity"));
        let (reference, digest, identity) = crate::fidelity_dataset::download(&agent, &cache, repo, commit, config)?;
        (reference, digest, Some(identity))
    } else if let Some(path) = &args.reference {
        let bytes = std::fs::read(path)?;
        (serde_json::from_slice::<Reference>(&bytes)?, format!("{:x}", Sha256::digest(&bytes)), None)
    } else {
        bail!("no reference source; use a published dataset or --reference");
    };
    let ancestry_match = selection.as_ref().is_some_and(|s|
        dataset_identity.as_ref().is_some_and(|d| d["repository"] == crate::fidelity_dataset::REPOSITORY
            && d["config"] == s.config && d["revision"] == s.revision));
    if !ancestry_match { validate_served_reference(&reference, model, dataset_identity.is_some())?; }
    if explicit {
        let root = dataset_identity.as_ref().filter(|d| d["repository"] == crate::fidelity_dataset::REPOSITORY)
            .and_then(|d| crate::fidelity_match::PUBLICATIONS.iter()
            .find(|p| d["config"] == p.config && d["revision"] == p.revision))
            .map(|p| p.root).unwrap_or(&reference.checkpoint).to_owned();
        selection = Some(crate::fidelity_match::Resolution { reference_match: "explicit".into(),
            reference_root: root, text_checkpoint: Some(reference.checkpoint.clone()),
            resolved_chain: selection.map(|s| s.resolved_chain).unwrap_or_default(),
            revision: dataset_identity.as_ref().and_then(|d| d["revision"].as_str()).unwrap_or("").into(),
            config: dataset_identity.as_ref().and_then(|d| d["config"].as_str()).unwrap_or("").into() });
    }
    if let Some(selection) = &mut selection { selection.text_checkpoint = Some(reference.checkpoint.clone()); }
    let standard = args.tier == "standard";
    let admitted = model_record["full_prefill_logits"] == true;
    let windows = if standard {
        ensure!(dataset_identity.is_some(), "Standard-v2 requires a sealed published dataset");
        crate::fidelity_dataset::standard_windows(&reference, &args.score_path, admitted)?
    } else if args.tier == "quick" && dataset_identity.is_some() {
        crate::fidelity_dataset::quick_windows(&reference)?
    } else { reference.selected_windows(args.tier != "quick")? };
    if standard {
        dataset_identity.as_mut().unwrap()["standard_subset"] = json!({
            "version": crate::fidelity_dataset::STANDARD_TIER,
            "sha256": crate::fidelity_dataset::STANDARD_SPLIT_SHA256,
            "mode": if admitted { "32 decode / 32 prefill" } else { crate::fidelity_dataset::STANDARD_FALLBACK },
            "decode": crate::fidelity_dataset::STANDARD_DECODE,
            "prefill": crate::fidelity_dataset::STANDARD_PREFILL,
        });
    }
    let standard_balance = standard.then(|| crate::fidelity_dataset::standard_balance(&reference));
    if let Some(limit) = model_record["max_context"].as_u64() {
        ensure!(windows.iter().all(|w| w.tokens.len() as u64 <= limit), "fidelity windows exceed server context ({limit})");
    }
    let media_payloads: Vec<_> = windows.iter().map(|window| {
        if window.media.is_empty() { return Ok(Vec::new()); }
        let root = args.media_root.as_ref().context("media windows require --media-root")?;
        crate::reference::media_probe_payload(window, model_record, root)
    }).collect::<Result<_>>()?;
    if args.tier != "quick" || dataset_identity.is_some() {
        ensure!(args.dump_dir.is_some(), "qualified dataset scoring needs --dump-dir on server-local NVMe");
    }
    let rows = if args.tier != "quick" && dataset_identity.is_none() {
        let dir = args.rows.as_ref().context("full tier needs --rows / CUTEAFD_FIDELITY_ROWS")?;
        ensure!(args.dump_dir.is_some(), "full tier needs --dump-dir on server-local NVMe");
        let rows = crate::fidelity_rows::manifest(dir, &reference.checkpoint, &reference.set_sha256, reference.vocab)?;
        crate::fidelity_rows::coverage(&rows, &windows)?;
        Some(rows)
    } else { None };
    if args.tier == "quick" {
        if let Some(identity) = &mut dataset_identity {
            identity["quick_subset"] = json!({"version":"bench-v1", "windows":crate::fidelity_dataset::QUICK_WINDOWS});
        }
    }
    let started = Instant::now();
    let mut records = Vec::new();
    let (mut missing, mut engine, mut settings) = (0usize, String::new(), Value::Null);
    let mut shape = String::new();
    let make_run = |records: Vec<crate::reference::Position>, missing, engine: String, settings: Value, shape: String| {
        let mut score = Fidelity::from_records(records); score.missing = missing;
        Run { schema: "cuteafd.fidelity.run/2".into(), arm: args.arm.clone(), checkpoint: model.into(),
            set_sha256: if reference.set_sha256.is_empty() { digest.clone() } else { reference.set_sha256.clone() },
            reference_sha256: digest.clone(), tier: if standard { crate::fidelity_dataset::STANDARD_TIER.into() } else { args.tier.clone() }, path_shape: shape,
            kl_kind: if dataset_identity.is_some() { "qualified-top1024-plus-tail" }
                else if rows.is_some() { "full-vocabulary" } else { "top32-plus-tail" }.into(),
            dataset: dataset_identity.clone(), reference_selection: selection.clone(), standard_balance: standard_balance.clone(), verify_rows: args.verify_rows, engine, settings,
            seconds: started.elapsed().as_secs_f64(), score, floor_top1: reference.expect.top1_min,
            floor_kl: reference.expect.kl_max, tripwire_expect: reference.expect.tripwires.clone() }
    };
    for (i, window) in windows.iter().enumerate() {
        let end = window.positions.last().context("empty window")?.pos + 1;
        let dump = args.dump_dir.as_ref().map(|d| d.join(if standard {
            format!("window-{}-{i:03}", args.score_path)
        } else { format!("window-{i:03}") }));
        let mut spec = json!({"prompt_ids": window.tokens[..end], "score_from": window.score_from,
            "top_k": 32, "want": window.want(), "cold": true, "no_speculation": true,
            "score_path": args.score_path});
        if !media_payloads[i].is_empty() { spec["media"] = json!(media_payloads[i]); }
        if args.tier != "quick" || dataset_identity.is_some() { spec["dump_rows"] = json!(dump); }
        if let Some(width) = args.verify_rows { spec["verify_rows"] = json!(width); }
        let response = probe_request(json!({"body": {"messages": [{"role": "user", "content": "fidelity probe"}], "max_tokens": 1,
                "temperature": 0}, "spec": spec}))?;
        crate::reference::verify_media_echo(window, &response["probe"])?;
        let probe: cuteafd_api::openai::probe::ProbeRecord = serde_json::from_value(response["probe"].clone())?;
        if let Some(error) = &probe.error { bail!("window {}: {error}", window.id); }
        ensure!(probe.engine.is_some() && probe.cold && probe.no_speculation && probe.cached_tokens == 0,
            "engine did not honor cold, drafts-off scoring");
        ensure!(probe.prompt_ids == window.tokens[..end], "engine ran different prompt tokens");
        let checkpoint = response["server"]["snapshot"].as_str().and_then(crate::report::hub_repo)
            .unwrap_or_else(|| response["server"]["model"].as_str().unwrap_or("").to_string());
        ensure!(checkpoint == model, "checkpoint changed during scoring");
        let this_engine = probe.engine.clone().unwrap();
        let actual_path = probe.score_path.as_deref().context("engine did not report its scoring path")?;
        ensure!(actual_path == args.score_path, "engine did not honor requested scoring path");
        let this_shape = match actual_path {
            "prefill" => "prefill-shaped",
            "decode" => "decode-shaped",
            _ => bail!("unsupported reported scoring path: {actual_path}"),
        };
        if i == 0 {
            engine = this_engine; settings = response["server"].clone(); shape = this_shape.into();
        } else {
            ensure!(engine == this_engine && settings == response["server"] && shape == this_shape,
                "engine build/settings changed during scoring");
        }
        let mut f = window.score(&probe.rows);
        if let Some(rows) = &rows {
            crate::fidelity_rows::score(args.rows.as_ref().unwrap(), rows, window, dump.as_ref().unwrap(), &mut f)?;
        }
        if dataset_identity.is_some() {
            crate::fidelity_rows::score_compact(reference.vocab, window, dump.as_ref().unwrap(), &mut f)?;
        }
        missing += f.missing;
        eprintln!("{}: {} rows, top1 {:.2}%, KL {:.6}", window.id, f.positions, 100.0 * f.top1, f.kl);
        records.extend(f.records);
        progress(i + 1, windows.len(), &make_run(records.clone(), missing, engine.clone(), settings.clone(), shape.clone()));
    }
    Ok(make_run(records, missing, engine, settings, shape))
}

fn run_pass(run: &Run) -> bool {
    // Explicit schema-1 compact references are diagnostic context-only probes,
    // not a published assistant-row fidelity gate. Preserve their sanity verdict.
    if run.dataset.is_none() && run.tier == "quick" && run.kl_kind == "top32-plus-tail"
        && run.score.records.iter().all(|row| row.role == "ctx") {
        return run.score.positions > 0 && run.score.missing == 0 && run.score.non_finite == 0;
    }
    crate::fidelity::verdict(run).pass
}

/// Exit 0 pass / 3 gate failure / 1 error is handled at the daemon edge.
pub fn execute(args: Args) -> Result<bool> {
    match args.action {
        Action::Run(args) => {
            let runs = run(&args)?;
            for run in &runs {
                eprintln!("{}: {} ({}) rows in {:.2}s; {} / {}", run.arm, run.score.positions,
                    run.score.missing, run.seconds, run.path_shape, run.kl_kind);
                if let Some(label) = crate::fidelity::verdict(run).label { eprintln!("{label}"); }
            }
            Ok(runs.iter().all(run_pass))
        }
        Action::CompareFull { a_decode, b_decode, a_prefill, b_prefill, bootstrap, seed, out } => {
            let load = |path: PathBuf| -> Result<Run> {
                Ok(serde_json::from_reader(std::fs::File::open(path)?)?)
            };
            let comparison = compare_full(&load(a_decode)?, &load(b_decode)?,
                &load(a_prefill)?, &load(b_prefill)?, bootstrap, seed)?;
            let text = serde_json::to_string_pretty(&comparison)?;
            if let Some(path) = out { std::fs::write(path, &text)?; }
            println!("{text}");
            eprintln!("Both full-tier statistical paths checked; separate agentic replay remains required.");
            Ok(comparison.pass)
        }
        Action::Compare { a, b, score_path, top1_margin, kl_margin, bootstrap, seed, out } => {
            let load = |path: PathBuf| -> Result<Run> {
                let value: Value = serde_json::from_reader(std::fs::File::open(path)?)?;
                let value = if value["schema"] == "cuteafd.fidelity.run/2" { value } else { value[&score_path].clone() };
                serde_json::from_value(value).context("missing selected scoring path")
            };
            let a = load(a)?;
            let b = load(b)?;
            let margin = if a.tier == "quick" { 0.01 } else { 0.005 };
            let comparison = compare(&a, &b, top1_margin.unwrap_or(margin), kl_margin.unwrap_or(margin), bootstrap, seed)?;
            let text = serde_json::to_string_pretty(&comparison)?;
            if let Some(path) = out { std::fs::write(path, &text)?; }
            println!("{text}");
            eprintln!("This is a {}-only gate. Precision defaults require both decode and prefill full-tier results.", a.path_shape);
            Ok(comparison.pass)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        args: Args,
    }

    fn parse(extra: &[&str]) -> RunArgs {
        let mut argv = vec!["fidelity", "run", "--arm", "baseline", "--out", "run.json"];
        argv.extend_from_slice(extra);
        let Action::Run(args) = Cli::try_parse_from(argv).unwrap().args.action else { panic!("run action") };
        args
    }

    #[test]
    fn explicit_dataset_overrides_rejected_ancestry_but_defaults_fail_before_network() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("README.md"), "---\nbase_model: [broken\n---\n").unwrap();
        std::fs::write(temp.path().join("config.json"), "{}").unwrap();
        let model_record = json!({"snapshot":temp.path()});
        let args = parse(&[]);
        let error = run_with(&args, "test/quant", &model_record, |_| panic!("unexpected probe"), |_, _, _| {}).unwrap_err();
        assert!(error.to_string().contains("malformed model-card YAML"));
        let args = parse(&["--dataset-config", "explicit", "--dataset-revision", "not-immutable"]);
        let error = run_with(&args, "test/quant", &model_record, |_| panic!("unexpected probe"), |_, _, _| {}).unwrap_err();
        assert!(error.to_string().contains("immutable lowercase 40-hex"), "{error}");
        assert_eq!(dataset_source_with(&args, None).unwrap().unwrap().2, "explicit");
    }

    #[test]
    fn dump_parent_is_created_without_creating_or_replacing_window_leaves() {
        let temp = tempfile::tempdir().unwrap();
        let dump = temp.path().join("new/arm/dump");
        let args = parse(&["--dump-dir", dump.to_str().unwrap()]);
        prepare_dump_dir(&args).unwrap();
        assert!(dump.is_dir());
        assert_eq!(std::fs::read_dir(&dump).unwrap().count(), 0);
        let leaf = dump.join("window-000");
        std::fs::create_dir(&leaf).unwrap();
        std::fs::write(leaf.join("sealed"), b"unchanged").unwrap();
        prepare_dump_dir(&args).unwrap();
        assert_eq!(std::fs::read(leaf.join("sealed")).unwrap(), b"unchanged");
    }

    #[test]
    fn unusable_dump_parent_fails_with_path_before_any_probe_or_network_request() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("not-a-directory");
        std::fs::write(&file, b"keep").unwrap();
        let dump = file.join("dump");
        let args = parse(&["--dump-dir", dump.to_str().unwrap(), "--url", "http://127.0.0.1:1"]);
        let mut requests = 0;
        let error = run_with(&args, "test", &json!({}), |_| {
            requests += 1;
            bail!("unexpected probe")
        }, |_, _, _| {}).unwrap_err();
        assert!(error.to_string().contains(dump.to_str().unwrap()));
        assert_eq!(requests, 0);
        assert!(run(&args).unwrap_err().to_string().contains(dump.to_str().unwrap()));
        assert_eq!(std::fs::read(file).unwrap(), b"keep");
    }

    #[test]
    fn explicit_context_only_quick_references_keep_diagnostic_sanity_verdict() {
        let score = Fidelity::from_records(vec![crate::reference::Position {
            window: "legacy".into(), block: "E".into(), bucket: "0-2K".into(), role: "ctx".into(),
            position: 1, agree: true, confident: true, top3_contained: true, agree_text: true,
            finite: true, kl: 0.0, nll: 0.5, ref_nll: 0.5, argmax: 1, reference_argmax: 1,
        }]);
        let mut run = Run { schema: "cuteafd.fidelity.run/2".into(), arm: "test".into(),
            checkpoint: "test".into(), set_sha256: "set".into(), reference_sha256: "ref".into(),
            tier: "quick".into(), path_shape: "decode-shaped".into(), kl_kind: "top32-plus-tail".into(),
            verify_rows: None, dataset: None, reference_selection: None, standard_balance: None, engine: "test".into(), settings: json!({}), seconds: 0.0,
            score, floor_top1: 0.9, floor_kl: 0.06, tripwire_expect: None };
        assert!(run_pass(&run));
        run.score.missing = 1; assert!(!run_pass(&run));
        run.score.missing = 0; run.score.non_finite = 1; assert!(!run_pass(&run));
        run.score.non_finite = 0; run.dataset = Some(json!({})); assert!(!run_pass(&run));
    }

    #[test]
    fn standard_parses_and_uses_the_same_published_default() {
        let standard = parse(&["--tier", "standard"]);
        assert_eq!(standard.tier,"standard");
        assert_eq!(dataset_source(&standard,"Qwen/Qwen3.8-Flash-Next-FP8").unwrap(),
            dataset_source(&parse(&["--tier","full"]),"Qwen/Qwen3.8-Flash-Next-FP8").unwrap());
    }

    #[test]
    fn paired_cli_accepts_both_standard_report_paths() {
        let report = crate::sample::full_report();
        let panel = report.panel("fidelity").unwrap().latest().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("standard.json");
        std::fs::write(&path,serde_json::to_vec(panel).unwrap()).unwrap();
        for score_path in ["decode","prefill"] {
            let args = Cli::try_parse_from(["fidelity","compare",path.to_str().unwrap(),path.to_str().unwrap(),
                "--score-path",score_path,"--bootstrap","100"]).unwrap().args;
            assert!(execute(args).unwrap());
        }
    }

    #[test]
    fn verified_full_default_preserves_quick_and_local_sources() {
        let full = parse(&["--tier", "full"]);
        assert_eq!(dataset_source(&full, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::REVISION, crate::fidelity_dataset::CONFIG)));
        assert_eq!(dataset_source(&parse(&[]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), dataset_source(&full, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap());
        assert_eq!(dataset_source(&parse(&["--tier", "full", "--reference", "reference.json"]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), None);
        assert_eq!(dataset_source(&parse(&["--tier", "full", "--rows", "rows"]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), None);
        assert_eq!(dataset_source(&parse(&["--tier", "full", "--dataset"]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), dataset_source(&full, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap());
    }

    #[test]
    fn flash_publication_default_is_family_specific_and_overrides_fail_closed() {
        let args = parse(&["--tier", "full"]);
        assert_eq!(dataset_source(&args, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").unwrap(),
            Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::FLASH_REVISION,
                crate::fidelity_dataset::FLASH_CONFIG)));
        assert!(dataset_source(&args, "unknown/model").is_err());
        assert_eq!(dataset_source(&parse(&[]), "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").unwrap(), dataset_source(&args, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").unwrap());
        let explicit = parse(&["--tier", "full", "--dataset", "other/repo"]);
        assert!(dataset_source(&explicit, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").is_err());
        let explicit = parse(&["--tier", "full", "--dataset-config", "other-config"]);
        assert!(dataset_source(&explicit, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").is_err());
    }

    #[test]
    fn mimo_pro_publication_is_checkpoint_specific_for_every_tier() {
        let model = "XiaomiMiMo/MiMo-V2.6-Pro-MOPD";
        for tier in ["quick", "standard", "full"] {
            let args = parse(&["--tier", tier]);
            assert_eq!(dataset_source(&args, model).unwrap(),
                Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::MIMO_PRO_REVISION,
                    crate::fidelity_dataset::MIMO_PRO_CONFIG)));
            assert_ne!(dataset_source(&args, model).unwrap(),
                dataset_source(&args, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").unwrap());
            for other in ["mimo_v2", "XiaomiMiMo/MiMo-V2.6-Pro-RL", "XiaomiMiMo/MiMo-V2.6-Pro-MOPD-speculator"] {
                assert!(dataset_source(&args, other).is_err(), "{other}");
            }
        }
        for local_flag in ["--reference", "--rows"] {
            assert_eq!(dataset_source(&parse(&[local_flag, "local"]), model).unwrap(), None);
        }
        assert!(dataset_source(&parse(&["--dataset", "other/repo"]), model).is_err());
        assert!(dataset_source(&parse(&["--dataset-config", "other-config"]), model).is_err());
        let media = parse(&["--tier", "full", "--dataset-config", crate::fidelity_dataset::MIMO_PRO_MEDIA_CONFIG,
            "--dataset-revision", crate::fidelity_dataset::MIMO_PRO_MEDIA_REVISION, "--media-root", "fixtures"]);
        assert_eq!(dataset_source(&media, model).unwrap(), Some((crate::fidelity_dataset::REPOSITORY,
            crate::fidelity_dataset::MIMO_PRO_MEDIA_REVISION, crate::fidelity_dataset::MIMO_PRO_MEDIA_CONFIG)));
    }

    #[test]
    fn glm_flash_publication_default_matches_the_base_and_quants() {
        let model = "zai-org/GLM-5.3-Flash";
        let full = parse(&["--tier", "full"]);
        assert_eq!(dataset_source(&full, model).unwrap(),
            Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::GLMF_REVISION,
                crate::fidelity_dataset::GLMF_CONFIG)));
        assert_eq!(dataset_source(&parse(&[]), model).unwrap(), dataset_source(&full, model).unwrap());
        for local_flag in ["--reference", "--rows"] {
            assert_eq!(dataset_source(&parse(&["--tier", "full", local_flag, "local"]), model).unwrap(), None);
        }
        for quant in ["zai-org/GLM-5.3-Flash-BF16", "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1",
            "wrldsuksgo2mars--GLM-5.3-Flash-EXL3-K3.25-v1"] {
            assert_eq!(dataset_source(&full, quant).unwrap(), dataset_source(&full, model).unwrap());
        }
        for other in ["zai-org/GLM-5.3-Flashlight", "RedHatAI/GLM-5.3-Flash-speculator.dspark-preview"] {
            assert!(dataset_source(&full, other).is_err());
        }
        assert!(dataset_source(&parse(&["--tier", "full", "--dataset", "other/repo"]), model).is_err());
        assert!(dataset_source(&parse(&["--tier", "full", "--dataset-config", "other-config"]), model).is_err());
    }

    #[test]
    fn qwen_publication_default_matches_the_base_and_quants() {
        let model = "Qwen/Qwen3.8-Flash-Next-FP8";
        let full = parse(&["--tier", "full"]);
        assert_eq!(dataset_source(&full, model).unwrap(),
            Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::QWEN_REVISION,
                crate::fidelity_dataset::QWEN_CONFIG)));
        assert_eq!(dataset_source(&parse(&[]), model).unwrap(), dataset_source(&full, model).unwrap());
        for local_flag in ["--reference", "--rows"] {
            assert_eq!(dataset_source(&parse(&["--tier", "full", local_flag, "local"]), model).unwrap(), None);
        }
        for quant in ["Qwen/Qwen3.8-Flash-Next", "wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1",
            "wrldsuksgo2mars--Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1"] {
            assert_eq!(dataset_source(&full, quant).unwrap(), dataset_source(&full, model).unwrap());
        }
        for other in ["Qwen/Qwen3.8-Flash", "Qwen/Qwen3.8-Flash-NextGen", "Qwen/Qwen3.8-Flash-Next-speculator"] {
            assert!(dataset_source(&full, other).is_err());
        }
        assert!(dataset_source(&parse(&["--tier", "full", "--dataset", "other/repo"]), model).is_err());
        assert!(dataset_source(&parse(&["--tier", "full", "--dataset-config", "other-config"]), model).is_err());
    }

    #[test]
    fn v4_flash_publication_default_matches_the_pinned_base_and_quants() {
        let model = "deepseek-ai/DeepSeek-V4-Flash-0731";
        let full = parse(&["--tier", "full"]);
        assert_eq!(dataset_source(&full, model).unwrap(),
            Some((crate::fidelity_dataset::REPOSITORY, crate::fidelity_dataset::V4FLASH_REVISION,
                crate::fidelity_dataset::V4FLASH_CONFIG)));
        assert_eq!(dataset_source(&parse(&[]), model).unwrap(), dataset_source(&full, model).unwrap());
        for local_flag in ["--reference", "--rows"] {
            assert_eq!(dataset_source(&parse(&["--tier", "full", local_flag, "local"]), model).unwrap(), None);
        }
        for quant in ["wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1",
            "wrldsuksgo2mars--DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1",
            "wrldsuksgo2mars/DeepSeek-V4-Flash-0731-NVFP4-v1"] {
            assert_eq!(dataset_source(&full, quant).unwrap(), dataset_source(&full, model).unwrap());
        }
        for other in ["deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro-0731",
            "deepseek-ai/DeepSeek-V4.1-Flashlight", "deepseek-ai/DeepSeek-V4-Flash-07310",
            "RedHatAI/DeepSeek-V4-Flash-0731-speculator", "other/DeepSeek-V4-Flash-0731-DFlash2"] {
            assert!(dataset_source(&full, other).is_err(), "{other}");
        }
        assert!(dataset_source(&parse(&["--tier", "full", "--dataset", "other/repo"]), model).is_err());
        assert!(dataset_source(&parse(&["--tier", "full", "--dataset-config", "other-config"]), model).is_err());
        let reference: Reference = serde_json::from_value(json!({"name":"test", "models":[model],
            "checkpoint":model, "vocab":129280, "expect":{"top1_min":0.94,"kl_max":0.04}})).unwrap();
        let quant = "wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1";
        assert!(validate_served_reference(&reference, quant, true).is_ok());
        assert!(validate_served_reference(&reference, quant, false).is_err());
        assert!(validate_served_reference(&reference, "deepseek-ai/DeepSeek-V4-Pro-0731", true).is_err());
        assert!(validate_served_reference(&reference, "deepseek-ai/DeepSeek-V4.1-Flash", true).is_err());
    }

    #[test]
    fn published_model_level_reference_accepts_quants_but_local_sources_remain_exact() {
        let qwen: Reference = serde_json::from_value(json!({"name":"test", "models":["Qwen/Qwen3.8-Flash-Next-FP8"],
            "checkpoint":"Qwen/Qwen3.8-Flash-Next-FP8", "vocab":248320,
            "expect":{"top1_min":0.94,"kl_max":0.04}})).unwrap();
        let quant = "wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1";
        assert!(validate_served_reference(&qwen, quant, true).is_ok());
        assert!(validate_served_reference(&qwen, quant, false).is_err());
        assert!(validate_served_reference(&qwen, "zai-org/GLM-5.3-Flash", true).is_err());
        let glm: Reference = serde_json::from_value(json!({"name":"test", "models":["zai-org/GLM-5.3-Flash"],
            "checkpoint":"zai-org/GLM-5.3-Flash", "vocab":154880,
            "expect":{"top1_min":0.93,"kl_max":0.05}})).unwrap();
        assert!(validate_served_reference(&glm, "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1", true).is_ok());
        assert!(validate_served_reference(&glm, quant, true).is_err());
    }

    #[test]
    fn explicit_repo_and_revision_override_only_the_dataset_source() {
        let args = parse(&["--tier", "full", "--dataset", "other/repo", "--dataset-revision", "1111111111111111111111111111111111111111"]);
        assert_eq!(dataset_source(&args, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), Some(("other/repo", "1111111111111111111111111111111111111111", crate::fidelity_dataset::CONFIG)));
        let args = parse(&["--tier", "full", "--dataset-revision", "2222222222222222222222222222222222222222"]);
        assert_eq!(dataset_source(&args, "deepseek-ai/DeepSeek-V4.1-Flash").unwrap(), Some((crate::fidelity_dataset::REPOSITORY, "2222222222222222222222222222222222222222", crate::fidelity_dataset::CONFIG)));
        assert_eq!(dataset_source(&parse(&["--dataset-revision", "main"]), "deepseek-ai/DeepSeek-V4.1-Flash").unwrap().unwrap().1, "main");
        for local_flag in ["--reference", "--rows"] {
            assert!(Cli::try_parse_from(["fidelity", "run", "--arm", "baseline", "--out", "run.json",
                "--tier", "full", "--dataset", crate::fidelity_dataset::REPOSITORY, local_flag, "local"]).is_err());
        }
    }
}
