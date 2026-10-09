//! `cuteafd bench smoke --matrix FILE`: release prep. For each matrix entry
//! (family, checkpoint/quant, hardware, launcher options) it writes a config,
//! launches it with ./run.sh, waits for readiness, runs the Release smoke
//! profile through the server's own runner, exports the report, tears the
//! launch down and moves on, past failures. Entries on disjoint hardware run
//! side by side (taking the shared lock files for the hardware they use);
//! finished entries are remembered per build so an interrupted matrix resumes.
use crate::report::Report;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, clap::Args)]
pub struct SmokeArgs {
    /// The matrix (JSON): {"build", "profile", "defaults": {...}, "entries": [{...}]}.
    #[arg(long)]
    pub matrix: PathBuf,
    /// Repository root: ./run.sh lives here and reports go to <out-root>/benchmarks/....
    #[arg(long, default_value = ".")]
    pub repo: PathBuf,
    /// Where `benchmarks/` is written (default: --repo).
    #[arg(long)]
    pub out_root: Option<PathBuf>,
    /// State, generated configs and logs (resume reads it).
    #[arg(long, default_value = "~/.cache/cuteafd/bench/smoke")]
    pub state: String,
    /// Entries run at once on disjoint hardware.
    #[arg(long, default_value_t = 2)]
    pub parallel: usize,
    /// Build label the entries run (default: the matrix's "build", else the repo HEAD).
    #[arg(long)]
    pub build: Option<String>,
    /// Run only these entries (comma-separated names).
    #[arg(long, value_delimiter = ',')]
    pub only: Option<Vec<String>>,
    /// Re-run entries already done for this build.
    #[arg(long)]
    pub force: bool,
    /// Print the plan and exit.
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Matrix {
    #[serde(default)]
    pub build: Option<String>,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub defaults: Entry,
    pub entries: Vec<Entry>,
}

/// One launch. Unset fields come from the matrix defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    #[serde(default)]
    pub name: String,
    /// Family id (deepseek_v41 launches through run.sh's own path and runs alone).
    #[serde(default)]
    pub family: Option<String>,
    /// Base config file (relative to --repo).
    #[serde(default)]
    pub config: Option<String>,
    /// Config keys set on top of the base (MODEL_ID, SPARK_COUNT, ADDR, ...).
    #[serde(default)]
    pub set: BTreeMap<String, String>,
    /// Extra ./run.sh arguments (e.g. ["--wip", "bench"]).
    #[serde(default)]
    pub run_args: Option<Vec<String>>,
    /// Coordinator GPUs (host indices) and Spark hosts the launch uses.
    #[serde(default)]
    pub gpus: Option<Vec<u32>>,
    #[serde(default)]
    pub sparks: Option<Vec<String>>,
    /// Seconds allowed for the launch to become ready.
    #[serde(default)]
    pub timeout_s: Option<u64>,
    /// Seconds allowed for the benchmark run once ready (default 900).
    #[serde(default)]
    pub run_timeout_s: Option<u64>,
    /// Runs with nothing else at once (default: deepseek_v41 entries).
    #[serde(default)]
    pub exclusive: Option<bool>,
    /// A profile name, or `panels:a,b` for those panels alone.
    #[serde(default)]
    pub profile: Option<String>,
}

impl Entry {
    fn merged(&self, defaults: &Entry) -> Entry {
        let mut set = defaults.set.clone();
        set.extend(self.set.clone());
        Entry {
            name: self.name.clone(),
            family: self.family.clone().or_else(|| defaults.family.clone()),
            config: self.config.clone().or_else(|| defaults.config.clone()),
            set,
            run_args: self.run_args.clone().or_else(|| defaults.run_args.clone()),
            gpus: self.gpus.clone().or_else(|| defaults.gpus.clone()),
            sparks: self.sparks.clone().or_else(|| defaults.sparks.clone()),
            timeout_s: self.timeout_s.or(defaults.timeout_s),
            run_timeout_s: self.run_timeout_s.or(defaults.run_timeout_s),
            exclusive: self.exclusive.or(defaults.exclusive),
            profile: self.profile.clone().or_else(|| defaults.profile.clone()),
        }
    }

    fn exclusive(&self) -> bool {
        self.exclusive.unwrap_or(self.family.as_deref() == Some("deepseek_v41"))
    }

    /// API port from ADDR (default 8000).
    fn port(&self, base: &HashMap<String, String>) -> u16 {
        let addr = self.set.get("ADDR").or_else(|| base.get("ADDR")).cloned().unwrap_or_else(|| "0.0.0.0:8000".into());
        addr.rsplit(':').next().and_then(|p| p.parse().ok()).unwrap_or(8000)
    }

    /// A key of everything that defines the launch (resume compares it).
    fn key(&self) -> String {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(serde_json::to_vec(self).unwrap_or_default());
        digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Outcome {
    pub key: String,
    pub build: String,
    pub status: String,
    #[serde(default)]
    pub dir: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub launch_s: Option<f64>,
    #[serde(default)]
    pub readiness_s: Option<f64>,
    #[serde(default)]
    pub code_tok_s: Option<f64>,
    #[serde(default)]
    pub prefill_tok_s: Option<f64>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub total_s: Option<f64>,
    pub finished: String,
}

fn expand(path: &str) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into())).join(rest),
        None => PathBuf::from(path),
    }
}

/// KEY=VALUE lines of a config file.
fn read_config(path: &Path) -> Result<HashMap<String, String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    Ok(text.lines().filter_map(|line| {
        let (key, value) = line.split_once('=')?;
        let key = key.trim();
        (!key.is_empty() && key.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .then(|| (key.to_string(), value.trim().to_string()))
    }).collect())
}

/// Lock order: Spark pool, GPU0, GPU1, then the two out-of-pool hosts.
fn locks_for(entry: &Entry) -> Vec<&'static str> {
    let gpus = entry.gpus.clone().unwrap_or_else(|| vec![0]);
    let mut locks = Vec::new();
    if gpus.contains(&0) || !entry.sparks.clone().unwrap_or_default().is_empty() || entry.exclusive() {
        locks.push("sparks.lock");
    }
    if gpus.contains(&0) || entry.exclusive() {
        locks.push("gpu0.lock");
    }
    if gpus.contains(&1) || entry.exclusive() {
        locks.push("gpu1.lock");
    }
    for (host, lock) in [("rhea", "rhea.lock"), ("moa", "moa.lock")] {
        if entry.sparks.as_ref().is_some_and(|hosts| hosts.iter().any(|h| h == host)) {
            locks.push(lock);
        }
    }
    locks
}

fn disjoint(a: &Entry, b: &Entry, base: &HashMap<String, String>) -> bool {
    if a.exclusive() || b.exclusive() || a.port(base) == b.port(base) {
        return false;
    }
    let (ga, gb) = (a.gpus.clone().unwrap_or_else(|| vec![0]), b.gpus.clone().unwrap_or_else(|| vec![0]));
    let (sa, sb) = (a.sparks.clone().unwrap_or_default(), b.sparks.clone().unwrap_or_default());
    !ga.iter().any(|g| gb.contains(g)) && !sa.iter().any(|s| sb.contains(s))
}

/// Lock files this process holds, with how many running entries need each.
#[derive(Default)]
struct Locks {
    held: HashMap<&'static str, (File, usize)>,
}

impl Locks {
    /// Takes `names`, waiting in line like every other agent (flock), when
    /// nothing of ours runs; with entries running, only what is free now.
    fn take(&mut self, names: &[&'static str], wait: bool) -> Result<bool> {
        if !wait {
            return self.try_take(names);
        }
        let dir = expand("~/.cache/cuteafd");
        std::fs::create_dir_all(&dir)?;
        let mut ordered: Vec<&'static str> = names.to_vec();
        ordered.sort_by_key(|n| match *n {
            "sparks.lock" => 0, "gpu0.lock" => 1, "gpu1.lock" => 2,
            "rhea.lock" => 3, "moa.lock" => 4, _ => 5,
        });
        let deadline = Instant::now() + Duration::from_secs(1800);
        for name in ordered {
            if self.held.contains_key(name) {
                continue;
            }
            let file = File::options().create(true).append(true).open(dir.join(name))?;
            loop {
                match file.try_lock() {
                    Ok(()) => break,
                    Err(std::fs::TryLockError::WouldBlock) if Instant::now() < deadline =>
                        std::thread::sleep(Duration::from_millis(500)),
                    Err(error) => bail!("timed out or failed waiting for {name}: {error}"),
                }
            }
            self.held.insert(name, (file, 0));
        }
        for name in names {
            if let Some((_, count)) = self.held.get_mut(name) {
                *count += 1;
            }
        }
        Ok(true)
    }

    fn try_take(&mut self, names: &[&'static str]) -> Result<bool> {
        let dir = expand("~/.cache/cuteafd");
        std::fs::create_dir_all(&dir)?;
        let mut taken = Vec::new();
        for &name in names {
            if self.held.contains_key(name) {
                continue;
            }
            let file = File::options().create(true).append(true).open(dir.join(name))?;
            match file.try_lock() {
                Ok(()) => taken.push((name, file)),
                Err(_) => {
                    // Give back what this attempt took; another agent holds the hardware.
                    for (_, file) in taken {
                        let _ = file.unlock();
                    }
                    return Ok(false);
                }
            }
        }
        for (name, file) in taken {
            self.held.insert(name, (file, 0));
        }
        for name in names {
            if let Some((_, count)) = self.held.get_mut(name) {
                *count += 1;
            }
        }
        Ok(true)
    }

    fn release(&mut self, names: &[&'static str]) {
        for name in names {
            let drop_it = match self.held.get_mut(name) {
                Some((_, count)) => {
                    *count = count.saturating_sub(1);
                    *count == 0
                }
                None => false,
            };
            if drop_it {
                if let Some((file, _)) = self.held.remove(name) {
                    let _ = file.unlock();
                }
            }
        }
    }
}

/// The build label entries are expected to run.
fn expected_build(args: &SmokeArgs, matrix: &Matrix) -> String {
    if let Some(build) = args.build.clone().or_else(|| matrix.build.clone()) {
        return build;
    }
    let git = |a: &[&str]| Command::new("git").arg("-C").arg(&args.repo).args(a).output().ok()
        .filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    match git(&["rev-parse", "HEAD"]) {
        Some(head) => {
            let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
            format!("{}{}", &head[..head.len().min(12)], if dirty { "+dirty" } else { "" })
        }
        None => "unknown".into(),
    }
}

struct Paths {
    state: PathBuf,
    configs: PathBuf,
    logs: PathBuf,
}

fn load_state(path: &Path) -> BTreeMap<String, Outcome> {
    std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn save_state(path: &Path, state: &BTreeMap<String, Outcome>) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

pub fn run(args: SmokeArgs) -> Result<()> {
    let matrix: Matrix = serde_json::from_str(&std::fs::read_to_string(&args.matrix)
        .with_context(|| format!("{}", args.matrix.display()))?).context("parsing the matrix")?;
    let repo = args.repo.canonicalize().context("--repo")?;
    // ./run.sh runs inside --repo: every path handed to it must be absolute.
    let out_root = std::path::absolute(args.out_root.clone().unwrap_or_else(|| repo.clone())).context("--out-root")?;
    let state_dir = std::path::absolute(expand(&args.state)).context("--state")?;
    let paths = Paths { state: state_dir.join("state.json"), configs: state_dir.join("configs"), logs: state_dir.join("logs") };
    std::fs::create_dir_all(&paths.configs)?;
    std::fs::create_dir_all(&paths.logs)?;
    let build = expected_build(&args, &matrix);
    let mut names = std::collections::HashSet::new();
    let mut entries = Vec::new();
    for raw in &matrix.entries {
        let entry = raw.merged(&matrix.defaults);
        if entry.name.is_empty() || !entry.name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) {
            bail!("matrix entry names must be non-empty [A-Za-z0-9._-]: {:?}", entry.name);
        }
        if !names.insert(entry.name.clone()) {
            bail!("duplicate matrix entry {}", entry.name);
        }
        if args.only.as_ref().is_some_and(|only| !only.contains(&entry.name)) {
            continue;
        }
        entries.push(entry);
    }
    let mut state = load_state(&paths.state);
    let pending: Vec<Entry> = entries.iter().filter(|e| {
        let done = state.get(&e.name).is_some_and(|o| o.status == "done" && o.key == e.key() && o.build == build);
        if done && !args.force {
            eprintln!("skip {} (done for build {build})", e.name);
        }
        args.force || !done
    }).cloned().collect();
    eprintln!("{} entries, {} to run, build {build}, up to {} at once", entries.len(), pending.len(), args.parallel);
    if args.dry_run {
        for e in &pending {
            eprintln!("  {} family={} gpus={:?} sparks={:?} locks={:?} set={:?}", e.name,
                e.family.as_deref().unwrap_or("?"), e.gpus, e.sparks, locks_for(e), e.set);
        }
        return Ok(());
    }
    let profile = matrix.profile.clone().unwrap_or_else(|| "smoke".into());
    let mut queue: std::collections::VecDeque<Entry> = pending.into();
    let mut running: Vec<Entry> = Vec::new();
    let mut locks = Locks::default();
    let (done_tx, done_rx) = mpsc::channel::<(String, Outcome)>();
    let started = Instant::now();
    let mut waiting_note = Instant::now() - Duration::from_secs(600);
    while !queue.is_empty() || !running.is_empty() {
        // Start whatever fits beside the running entries.
        let mut started_one = false;
        if running.len() < args.parallel.max(1) {
            let base_of = |e: &Entry| e.config.as_ref().map(|c| repo.join(c)).and_then(|p| read_config(&p).ok())
                .unwrap_or_default();
            if let Some(index) = queue.iter().position(|e| running.iter().all(|r| disjoint(e, r, &base_of(e)))) {
                let entry = queue[index].clone();
                let needed = locks_for(&entry);
                if running.is_empty() {
                    eprintln!("{}: waiting in line for {needed:?}", entry.name);
                }
                if locks.take(&needed, running.is_empty())? {
                    queue.remove(index);
                    running.push(entry.clone());
                    started_one = true;
                    let (tx, repo, out_root, profile, build) = (done_tx.clone(), repo.clone(), out_root.clone(),
                        entry.profile.clone().unwrap_or_else(|| profile.clone()), build.clone());
                    let (configs, logs) = (paths.configs.clone(), paths.logs.clone());
                    let parallel = args.parallel > 1;
                    std::thread::spawn(move || {
                        let outcome = run_entry(&entry, &repo, &out_root, &configs, &logs, &profile, &build, parallel);
                        let _ = tx.send((entry.name.clone(), outcome));
                    });
                } else if waiting_note.elapsed() > Duration::from_secs(120) {
                    eprintln!("waiting for {needed:?} (held by another agent) for {}", entry.name);
                    waiting_note = Instant::now();
                }
            }
        }
        if started_one {
            continue;
        }
        match done_rx.recv_timeout(Duration::from_secs(5)) {
            Ok((name, outcome)) => {
                if let Some(index) = running.iter().position(|e| e.name == name) {
                    let entry = running.remove(index);
                    locks.release(&locks_for(&entry));
                }
                eprintln!("{name}: {}{}", outcome.status, outcome.error.as_deref().map(|e| format!(" ({e})")).unwrap_or_default());
                state.insert(name, outcome);
                save_state(&paths.state, &state)?;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let summary = summary(&entries, &state, &build, started.elapsed().as_secs_f64());
    std::fs::write(state_dir.join("summary.md"), &summary)?;
    println!("{summary}");
    let failed = entries.iter().filter(|e| state.get(&e.name).is_none_or(|o| o.status != "done")).count();
    if failed > 0 {
        bail!("{failed} of {} entries did not finish (see {})", entries.len(), state_dir.join("summary.md").display());
    }
    Ok(())
}

fn summary(entries: &[Entry], state: &BTreeMap<String, Outcome>, build: &str, seconds: f64) -> String {
    let mut out = format!("# Release smoke · build {build} · {}\n\n", crate::render::seconds(seconds));
    out.push_str("| Entry | Status | Ready | C1 code | 8K prefill | Quality | Report |\n");
    out.push_str("| --- | --- | ---: | ---: | ---: | --- | --- |\n");
    for e in entries {
        let Some(o) = state.get(&e.name) else {
            out.push_str(&format!("| {} | not run | | | | | |\n", e.name));
            continue;
        };
        let opt = |v: Option<f64>, f: fn(f64) -> String| v.map(f).unwrap_or_default();
        out.push_str(&format!("| {} | {}{} | {} | {} | {} | {} | {} |\n", e.name, o.status,
            o.error.as_deref().map(|e| format!(": {}", e.chars().take(80).collect::<String>())).unwrap_or_default(),
            opt(o.launch_s, crate::render::seconds), opt(o.code_tok_s, crate::render::rate),
            opt(o.prefill_tok_s, crate::render::rate), o.quality.clone().unwrap_or_default(),
            o.dir.clone().unwrap_or_default()));
    }
    out
}

/// A config for the entry: the base file plus the entry's keys (later lines win).
fn write_config(entry: &Entry, repo: &Path, configs: &Path, parallel: bool) -> Result<PathBuf> {
    let base = entry.config.as_ref().map(|c| repo.join(c)).unwrap_or_else(|| repo.join("cuteafd.config"));
    let mut text = std::fs::read_to_string(&base).with_context(|| format!("{}", base.display()))?;
    text.push_str(&format!("\n# bench smoke entry {}\n", entry.name));
    let mut set = entry.set.clone();
    if parallel && entry.family.as_deref() != Some("deepseek_v41") {
        // Distinct coordinator containers for launches side by side.
        set.entry("INSTANCE".into()).or_insert_with(|| entry.name.clone());
    }
    for (key, value) in &set {
        if matches!(key.as_str(), "BENCH_SIMULATED" | "BENCH_HARDWARE_CLASS") {
            continue;
        }
        text.push_str(&format!("{key}={value}\n"));
    }
    let path = configs.join(format!("{}.config", entry.name));
    std::fs::write(&path, text)?;
    Ok(path)
}

/// The path of an open log file (Linux: through /proc).
fn log_path(log: &File) -> std::io::Result<PathBuf> {
    use std::os::fd::AsRawFd;
    std::fs::read_link(format!("/proc/self/fd/{}", log.as_raw_fd()))
}

/// Removes the entry's coordinator and expert containers.
fn teardown(entry: &Entry, config: &HashMap<String, String>, log: &mut File) {
    let instance = config.get("INSTANCE").filter(|v| !v.is_empty());
    let coordinator = match instance {
        Some(instance) => format!("cuteafd-coordinator-{instance}"),
        None => "cuteafd-coordinator".into(),
    };
    // The coordinator's own log goes next to the entry's (failures and warnings outlive the container).
    if let Ok(path) = log_path(log) {
        if let Ok(file) = File::create(path.with_extension("coordinator.log")) {
            let _ = Command::new("docker").args(["logs", &coordinator]).stdout(file.try_clone().map(Stdio::from)
                .unwrap_or(Stdio::null())).stderr(Stdio::from(file)).status();
        }
    }
    let _ = Command::new("docker").args(["rm", "-f", &coordinator]).stdout(Stdio::null()).stderr(Stdio::null()).status();
    let port = config.get("EXPERT_PORT").cloned().unwrap_or_else(|| "19441".into());
    let count: usize = config.get("SPARK_COUNT").and_then(|v| v.parse().ok()).unwrap_or(0);
    for rank in 0..count {
        let Some(host) = config.get(&format!("SPARK_{rank}_HOST")) else { continue };
        let worker = format!("cuteafd-spark-expert-{host}-{port}");
        if let Ok(path) = log_path(log) {
            if let Ok(file) = File::create(path.with_extension(format!("{host}.log"))) {
                let _ = Command::new("ssh").args(["-o", "BatchMode=yes", host, "docker", "logs", &worker])
                    .stdout(file.try_clone().map(Stdio::from).unwrap_or(Stdio::null()))
                    .stderr(Stdio::from(file)).status();
            }
        }
        let script = format!("ids=$(docker ps -aq --filter name=^{worker}$); \
            [ -z \"$ids\" ] || docker rm -f $ids >/dev/null 2>&1 || true");
        let _ = Command::new("ssh").args(["-o", "BatchMode=yes", host, &script]).status();
    }
    let _ = writeln!(log, "teardown {} ({coordinator})", entry.name);
}

#[allow(clippy::too_many_arguments)]
fn run_entry(entry: &Entry, repo: &Path, out_root: &Path, configs: &Path, logs: &Path, profile: &str, build: &str,
    parallel: bool) -> Outcome {
    let started = Instant::now();
    let mut outcome = Outcome { key: entry.key(), build: build.to_string(), status: "failed".into(),
        finished: String::new(), ..Outcome::default() };
    let log_path = logs.join(format!("{}.log", entry.name));
    let mut log = match File::create(&log_path) {
        Ok(file) => file,
        Err(error) => {
            outcome.error = Some(format!("log {}: {error}", log_path.display()));
            return outcome;
        }
    };
    let result = (|| -> Result<Report> {
        let config_path = write_config(entry, repo, configs, parallel)?;
        let config = read_config(&config_path)?;
        let port = entry.port(&config);
        let mut command = Command::new(repo.join("run.sh"));
        command.arg("--config").arg(&config_path).arg("--restart").args(entry.run_args.clone().unwrap_or_default())
            .current_dir(repo).stdout(log.try_clone()?).stderr(log.try_clone()?);
        let launch = Instant::now();
        let mut child = command.spawn().context("starting ./run.sh")?;
        let timeout = Duration::from_secs(entry.timeout_s.unwrap_or(900));
        let url = format!("http://127.0.0.1:{port}");
        // run.sh waits for readiness itself; poll /health too in case it returns early.
        loop {
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    bail!("./run.sh exited {status} (log {})", log_path.display());
                }
                if health(&url) {
                    break;
                }
            }
            if launch.elapsed() > timeout {
                let _ = child.kill();
                bail!("not ready within {} s", timeout.as_secs());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        outcome.launch_s = Some(launch.elapsed().as_secs_f64());
        // `panels:a,b` runs those panels instead of a profile.
        let (profile, panels) = match profile.strip_prefix("panels:") {
            Some(list) => (None, Some(list.split(',').map(str::to_string).collect())),
            None => (Some(profile.to_string()), None),
        };
        let options = crate::cli::RunOptions { url, profile, panels, passes: vec![],
            export: vec!["svg".into(), "card".into(), "json".into()], out: None, label: Some(entry.name.clone()),
            root: out_root.to_path_buf(),
            api_key: std::env::var("CUTEAFD_API_KEY").ok(), quiet: true,
            deadline: Some(Duration::from_secs(entry.run_timeout_s.unwrap_or(900))) };
        let (mut report, dir) = crate::cli::run(&options)?;
        // Publication markers are runner metadata, not serving defaults. This
        // also works against images built before these markers existed.
        for (key, name) in [("BENCH_SIMULATED", "simulated"), ("BENCH_HARDWARE_CLASS", "hardware.class")] {
            if let Some(value) = entry.set.get(key).filter(|v| !v.is_empty()) {
                report.server.configuration.settings.retain(|s| s.name != name);
                report.server.configuration.settings.push(crate::report::Setting {
                    name: name.into(), value: Some(value.clone()), default: None, source: "publication".into(),
                });
            }
        }
        crate::cli::write_exports(&report, &dir, &options.export)?;
        outcome.dir = Some(dir.display().to_string());
        Ok(report)
    })();
    match result {
        Ok(report) => {
            outcome.status = "done".into();
            outcome.readiness_s = report.server.readiness_s;
            if let Some(b) = &report.baseline {
                outcome.code_tok_s = b.card.decode_of("code").map(|d| d.tok_s);
                outcome.prefill_tok_s = b.card.prefill.as_ref().map(|p| p.tok_s);
                outcome.quality = Some(if report.quality_failed() { format!("FAILED: {}", crate::render::bodies::quality_line(b)) }
                    else { b.quality.badge() });
            }
            outcome.build = build.to_string();
            let _ = writeln!(log, "server build {}", report.server.build.label());
            // A server that died after finishing the run still fails the entry's serving check.
            let running = Command::new("docker").args(["inspect", "-f", "{{.State.Running}}", &coordinator_name(entry, parallel)])
                .output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim() == "true");
            if running == Some(false) {
                outcome.error = Some("the server exited during or after the run (see its coordinator log)".into());
                let _ = writeln!(log, "warning: the coordinator is not running any more");
            }
        }
        Err(error) => {
            outcome.error = Some(format!("{error:#}"));
            let _ = writeln!(log, "error: {error:#}");
            // The coordinator's last lines explain most failures.
            let tail = Command::new("docker").args(["logs", "--tail", "40", &coordinator_name(entry, parallel)]).output();
            if let Ok(tail) = tail {
                let _ = log.write_all(&tail.stdout);
                let _ = log.write_all(&tail.stderr);
            }
        }
    }
    if let Ok(config) = read_config(&configs.join(format!("{}.config", entry.name))) {
        teardown(entry, &config, &mut log);
    }
    outcome.total_s = Some(started.elapsed().as_secs_f64());
    outcome.finished = crate::report::now_rfc3339();
    outcome
}

fn coordinator_name(entry: &Entry, parallel: bool) -> String {
    match entry.set.get("INSTANCE") {
        Some(instance) => format!("cuteafd-coordinator-{instance}"),
        None if parallel && entry.family.as_deref() != Some("deepseek_v41") => format!("cuteafd-coordinator-{}", entry.name),
        None => "cuteafd-coordinator".into(),
    }
}

fn health(url: &str) -> bool {
    ureq::AgentBuilder::new().timeout(Duration::from_secs(5)).build().get(&format!("{url}/health")).call().is_ok()
}

/// The matrix entries as JSON (for `--dry-run` tooling).
pub fn describe(matrix: &Matrix) -> Value {
    serde_json::to_value(matrix).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, gpus: &[u32], sparks: &[&str], port: u16) -> Entry {
        Entry { name: name.into(), family: Some("glm5_flash".into()), gpus: Some(gpus.to_vec()),
            sparks: Some(sparks.iter().map(|s| s.to_string()).collect()),
            set: [("ADDR".to_string(), format!("0.0.0.0:{port}"))].into(), ..Entry::default() }
    }

    #[test]
    fn disjoint_hardware_runs_together() {
        let base = HashMap::new();
        let a = entry("a", &[0], &["ostrich", "dodo", "emu", "kiwi"], 8000);
        let b = entry("b", &[1], &["rhea", "moa"], 8001);
        let c = entry("c", &[1], &[], 8000);
        assert!(disjoint(&a, &b, &base));
        assert!(!disjoint(&a, &c, &base), "same port");
        assert!(!disjoint(&b, &entry("d", &[1], &[], 8002), &base), "same GPU");
        let v41 = Entry { family: Some("deepseek_v41".into()), ..entry("v", &[1], &[], 8009) };
        assert!(!disjoint(&a, &v41, &base), "V4.1 runs alone");
        assert_eq!(locks_for(&a), vec!["sparks.lock", "gpu0.lock"]);
        assert_eq!(locks_for(&b), vec!["sparks.lock", "gpu1.lock", "rhea.lock", "moa.lock"]);
        assert_eq!(locks_for(&entry("max", &[0, 1], &["ostrich", "rhea", "moa"], 8003)),
            vec!["sparks.lock", "gpu0.lock", "gpu1.lock", "rhea.lock", "moa.lock"]);
        assert_eq!(locks_for(&c), vec!["gpu1.lock"]);
    }

    #[test]
    fn entries_merge_defaults_and_keys_change_with_settings() {
        let defaults = Entry { config: Some("base.config".into()), set: [("CONCURRENCY".to_string(), "8".to_string())].into(),
            timeout_s: Some(600), ..Entry::default() };
        let mut e = entry("a", &[1], &[], 8001);
        let merged = e.merged(&defaults);
        assert_eq!(merged.config.as_deref(), Some("base.config"));
        assert_eq!(merged.set.get("CONCURRENCY").map(String::as_str), Some("8"));
        let key = merged.key();
        e.set.insert("MODEL_ID".into(), "x/y".into());
        assert_ne!(e.merged(&defaults).key(), key);
    }

    #[test]
    fn configs_append_overrides() {
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(repo.path().join("cuteafd.config"), "MODEL_ID=a/b\nADDR=0.0.0.0:8000\n").unwrap();
        let mut e = entry("one", &[1], &[], 8123);
        e.set.insert("BENCH_SIMULATED".into(), "5090".into());
        e.set.insert("BENCH_HARDWARE_CLASS".into(), "5090".into());
        let path = write_config(&e, repo.path(), repo.path(), true).unwrap();
        let config = read_config(&path).unwrap();
        assert_eq!(config.get("ADDR").map(String::as_str), Some("0.0.0.0:8123"));
        assert_eq!(config.get("INSTANCE").map(String::as_str), Some("one"));
        assert!(!config.contains_key("BENCH_SIMULATED"));
        assert!(!config.contains_key("BENCH_HARDWARE_CLASS"));
        assert_eq!(config.get("MODEL_ID").map(String::as_str), Some("a/b"));
    }
}
