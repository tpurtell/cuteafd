//! `cuteafd bench smoke --matrix FILE`: release prep. For each matrix entry
//! (family, checkpoint/quant, hardware, launcher options) it writes a config,
//! launches it with ./run.sh, waits for readiness, runs the Release smoke
//! profile through the server's own runner, exports the report, tears the
//! launch down and moves on, past failures. Entries on disjoint hardware run
//! side by side (taking the shared lock files for the hardware they use);
//! finished entries are remembered per build so an interrupted matrix resumes.
//! SIGTERM, SIGINT or SIGHUP cancels the run: every launched entry's launcher
//! is stopped and its containers removed (bounded, see [`CANCEL_TEARDOWN_S`])
//! before the process exits non-zero and its hardware locks are released.
use crate::report::Report;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::os::unix::process::CommandExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
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
    /// Historical exports only; never scheduled as launches.
    #[serde(default)]
    pub retired: Vec<crate::publish::Retired>,
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
    /// Collect fabric telemetry; release cards enable it unless explicitly disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fabric: Option<bool>,
    /// Optional argv hooks, run only while smoke owns this card's locks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precheck: Option<Hook>,
    /// Bounded setup after acquiring hardware locks, before the launcher starts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prelaunch: Option<Hook>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observer: Option<Hook>,
    /// Client-side code-decode sweep after the profile (`[1, 16]`), written to
    /// `sweep.json` beside the report; see `crate::sweep`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub levels: Option<Vec<usize>>,
    /// Re-measure C1 after the widest sweep level (post-load C1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_load_c1: Option<bool>,
    /// Make that post-load C1 re-send the levels C1's exact prompts (needs
    /// `PREFIX_CACHE_ENTRIES=0`: the sweep fails if a repeat hits the cache).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_load_repeat: Option<bool>,
    /// Cold prefill at these prompt lengths, in the same sweep (`[8192]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_lengths: Option<Vec<u64>>,
    /// Output tokens per sweep decode request (default `crate::sweep::TOKENS`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep_tokens: Option<u64>,
    /// `false` runs explicit panels without the server's baseline (basic card
    /// + quick quality); default true. Needs a server build that honours it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<bool>,
    /// A measurement window around the timed part of the sweep (needs levels or
    /// prefill_lengths), e.g. a profiler's capture range.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<Window>,
    /// A `spec_lossless_set` case file (absolute, or relative to --repo); default: the shipped set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lossless_set: Option<String>,
    /// Opt-in full-vocabulary row dumps for selected lossless cases (needs PROBE_DUMP_ROOT).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lossless_dump: Option<crate::panels::lossless::RowDump>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_lengths: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_levels: Option<Vec<usize>>,
}

/// `open` runs once after the sweep's first warm-up, just before its first
/// timed request; `close` runs when the sweep ends (passed or failed) and again
/// before teardown if it has not succeeded yet. Both run to completion while
/// the server is up; a failure fails the entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Window {
    pub open: Hook,
    pub close: Hook,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Hook {
    pub argv: Vec<String>,
    pub timeout_s: u64,
    /// Grace after the observer stop signal, still bounded by the total timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_timeout_s: Option<u64>,
    /// Observer errors are diagnostic unless explicitly required.
    #[serde(default)]
    pub required: bool,
    /// Optional bounded cleanup for external resources (e.g. sampler containers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cleanup: Option<Vec<String>>,
}

// Each hook gets a process group so subprocesses cannot survive lock release.
struct HookChild {
    child: Child,
    deadline: Instant,
    finish_timeout: Duration,
    stop: PathBuf,
    cleanup: Option<Vec<String>>,
    inflight: Arc<Inflight>,
}

impl HookChild {
    fn spawn(hook: &Hook, url: &str, config: &Path, log: &Path, kind: &str, inflight: &Arc<Inflight>) -> Result<Self> {
        if hook.argv.is_empty() || hook.timeout_s == 0 || hook.timeout_s > 7200 {
            bail!("{kind} hook needs argv and timeout_s in 1..=7200");
        }
        let finish_timeout = Duration::from_secs(hook.finish_timeout_s.unwrap_or(5));
        if finish_timeout.is_zero() || finish_timeout.as_secs() > 7200 {
            bail!("{kind} hook finish_timeout_s must be in 1..=7200");
        }
        let stop = log.with_extension(format!("{kind}.stop"));
        let _ = std::fs::remove_file(&stop);
        let output = File::create(log.with_extension(format!("{kind}.log")))?;
        // Registered under the lock a cancellation snapshots, so none is missed.
        let mut hooks = inflight.hooks.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(signal) = inflight.cancelled() {
            bail!("{kind} hook not started: {}", Cancelled(signal));
        }
        let child = Command::new("timeout").args(["--kill-after=1s", &format!("{}s", hook.timeout_s)])
            .args(&hook.argv)
            .env("CUTEAFD_SMOKE_URL", url).env("CUTEAFD_SMOKE_CONFIG", config)
            .env("CUTEAFD_SMOKE_STOP", &stop)
            .stdout(output.try_clone()?).stderr(output).process_group(0).spawn()
            .with_context(|| format!("starting {kind} hook"))?;
        hooks.insert(child.id(), hook.cleanup.clone());
        drop(hooks);
        Ok(Self { child, deadline: Instant::now() + Duration::from_secs(hook.timeout_s), finish_timeout, stop,
            cleanup: hook.cleanup.clone(), inflight: inflight.clone() })
    }

    fn wait(&mut self) -> Result<()> {
        loop {
            if let Some(status) = self.child.try_wait()? {
                if !status.success() { bail!("hook exited {status}"); }
                return Ok(());
            }
            if Instant::now() >= self.deadline { bail!("hook timed out"); }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(status) = self.child.try_wait()? {
            if !status.success() { bail!("observer exited {status}"); }
            return Ok(());
        }
        if Instant::now() >= self.deadline { bail!("observer timed out"); }
        std::fs::write(&self.stop, b"stop\n")?;
        self.deadline = self.deadline.min(Instant::now() + self.finish_timeout);
        self.wait()
    }
}

impl Drop for HookChild {
    fn drop(&mut self) {
        // Unregistered before it is reaped: a cancellation never signals a reused pid.
        self.inflight.hooks.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.child.id());
        // Kill the group, including an exited hook's descendants, before locks release.
        kill_group(self.child.id(), "KILL");
        let _ = self.child.wait();
        hook_cleanup(&self.cleanup);
    }
}

/// A hook's bounded cleanup command (idempotent: a cancellation may run it too).
fn hook_cleanup(cleanup: &Option<Vec<String>>) {
    if let Some(argv) = cleanup.as_ref().filter(|argv| !argv.is_empty()) {
        let _ = Command::new("timeout").args(["--kill-after=1s", "10s"]).args(argv)
            .stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
}

/// Sends SIG (a `kill` name, e.g. TERM) to process group `pgid`.
fn kill_group(pgid: u32, signal: &str) {
    let _ = Command::new("/bin/kill").args([&format!("-{signal}"), "--", &format!("-{pgid}")])
        .stdout(Stdio::null()).stderr(Stdio::null()).status();
}

/// A smoke run stopped by SIGTERM, SIGINT or SIGHUP (the signal number). The
/// CLI exits 128 + the signal once every launched entry is torn down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("cancelled by {}", signal_name(*.0))]
pub struct Cancelled(pub i32);

fn signal_name(signal: i32) -> String {
    match signal {
        signal_hook::consts::SIGTERM => "SIGTERM".into(),
        signal_hook::consts::SIGINT => "SIGINT".into(),
        signal_hook::consts::SIGHUP => "SIGHUP".into(),
        other => format!("signal {other}"),
    }
}

/// Wall-clock bound (seconds) on a cancelled run's cleanup: hooks killed,
/// launchers stopped, then every launched entry's coordinator and Spark
/// workers removed, all in parallel and each command bounded. It stays under
/// systemd's default TimeoutStopSec (90 s) and hwlock.py's stop grace (80 s),
/// so the hardware locks are released only after the containers are gone.
const CANCEL_TEARDOWN_S: u64 = 60;
/// How long a signalled launcher (./run.sh) may take to exit before SIGKILL.
const LAUNCHER_GRACE_S: u64 = 5;
/// Bounds (seconds) on each cancelled-teardown step: `docker inspect`, the
/// graceful `docker stop -t`, `docker logs` and `docker rm -f`; `timeout`
/// adds [`KILL_AFTER_S`] to each.
const INSPECT_S: u64 = 5;
const STOP_S: u64 = 15;
const LOGS_S: u64 = 10;
const RM_S: u64 = 10;
const KILL_AFTER_S: u64 = 1;
// The coordinator's steps run one after another (a worker's take less) after the launcher stops.
const _: () = assert!(LAUNCHER_GRACE_S + INSPECT_S + (STOP_S + 5) + LOGS_S + RM_S + 4 * KILL_AFTER_S <= CANCEL_TEARDOWN_S);

/// Programs teardown runs (tests substitute logging fakes).
#[derive(Debug, Clone)]
struct Tools {
    docker: PathBuf,
    ssh: PathBuf,
}

impl Default for Tools {
    fn default() -> Self {
        Self { docker: "docker".into(), ssh: "ssh".into() }
    }
}

/// An entry whose launcher started: its containers exist until teardown.
struct Launch {
    launcher: Child,
    config: PathBuf,
    log: PathBuf,
}

/// What a cancellation must undo, shared by the scheduler and entry threads:
/// the signal that cancelled the run (0 while running), launched entries (an
/// entry stays here until its own teardown finished) and live hook process
/// groups with their cleanup commands.
#[derive(Default)]
struct Inflight {
    signal: Arc<AtomicUsize>,
    launches: Mutex<BTreeMap<String, Launch>>,
    hooks: Mutex<BTreeMap<u32, Option<Vec<String>>>>,
    tools: Tools,
}

impl Inflight {
    /// Routes SIGTERM, SIGINT and SIGHUP to the cancel flag (the first one
    /// wins; repeats, e.g. systemd's cgroup-wide SIGTERM plus a parent
    /// forwarding its own, change nothing).
    fn install(&self) -> Result<()> {
        use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
        self.register(&[SIGTERM, SIGINT, SIGHUP])
    }

    fn register(&self, signals: &[i32]) -> Result<()> {
        for &signal in signals {
            // A plain flag, never the default action: a repeated signal must not
            // cut a running teardown short (systemd's own SIGKILL bounds it).
            signal_hook::flag::register_usize(signal, self.signal.clone(), signal as usize)
                .with_context(|| format!("installing the {} handler", signal_name(signal)))?;
        }
        Ok(())
    }

    fn cancelled(&self) -> Option<i32> {
        match self.signal.load(Ordering::SeqCst) {
            0 => None,
            signal => Some(signal as i32),
        }
    }

    #[cfg(test)]
    fn cancel(&self, signal: i32) {
        let _ = self.signal.compare_exchange(0, signal as usize, Ordering::SeqCst, Ordering::SeqCst);
    }

    /// Starts `command` as entry `name`'s launcher, in its own process group
    /// so a cancellation can stop it and everything it started. Refused once
    /// cancelled: a launch never begins after the cancellation's snapshot.
    fn launch(&self, name: &str, command: &mut Command, config: &Path, log: &Path) -> Result<()> {
        let mut launches = self.launches.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(signal) = self.cancelled() {
            return Err(Cancelled(signal).into());
        }
        let launcher = command.process_group(0).spawn().context("starting ./run.sh")?;
        launches.insert(name.into(), Launch { launcher, config: config.into(), log: log.into() });
        Ok(())
    }

    /// The launcher's exit status, once it exited.
    fn launcher_status(&self, name: &str) -> Result<Option<std::process::ExitStatus>> {
        let mut launches = self.launches.lock().unwrap_or_else(|e| e.into_inner());
        let launch = launches.get_mut(name).context("launcher not registered")?;
        Ok(launch.launcher.try_wait()?)
    }

    /// Stops entry `name`'s launcher and its process group (TERM, then KILL).
    fn stop_launcher(&self, name: &str) {
        let pgid = {
            let mut launches = self.launches.lock().unwrap_or_else(|e| e.into_inner());
            let Some(launch) = launches.get_mut(name) else { return };
            if !matches!(launch.launcher.try_wait(), Ok(None)) {
                return;
            }
            launch.launcher.id()
        };
        kill_group(pgid, "TERM");
        let deadline = Instant::now() + Duration::from_secs(LAUNCHER_GRACE_S);
        while Instant::now() < deadline && matches!(self.launcher_status(name), Ok(None)) {
            std::thread::sleep(Duration::from_millis(50));
        }
        kill_group(pgid, "KILL");
        if let Some(launch) = self.launches.lock().unwrap_or_else(|e| e.into_inner()).get_mut(name) {
            let _ = launch.launcher.wait();
        }
    }

    /// The entry's own teardown finished: nothing of it is left to undo.
    fn finished(&self, name: &str) {
        self.launches.lock().unwrap_or_else(|e| e.into_inner()).remove(name);
    }

    /// Undoes everything in flight, bounded by [`CANCEL_TEARDOWN`]: kills hook
    /// groups (running their cleanups), stops launchers, then tears down every
    /// launched entry in parallel. Returns the entries torn down.
    fn cancel_all(&self) -> Vec<String> {
        let hooks: Vec<(u32, Option<Vec<String>>)> = self.hooks.lock().unwrap_or_else(|e| e.into_inner())
            .iter().map(|(pid, cleanup)| (*pid, cleanup.clone())).collect();
        let names: Vec<(String, PathBuf, PathBuf)> = self.launches.lock().unwrap_or_else(|e| e.into_inner())
            .iter().map(|(name, launch)| (name.clone(), launch.config.clone(), launch.log.clone())).collect();
        std::thread::scope(|scope| {
            for (pid, cleanup) in &hooks {
                scope.spawn(move || {
                    kill_group(*pid, "KILL");
                    hook_cleanup(cleanup);
                });
            }
            for (name, config, log) in &names {
                scope.spawn(move || {
                    self.stop_launcher(name);
                    let mut log = match File::options().create(true).append(true).open(log) {
                        Ok(file) => file,
                        Err(_) => match File::create("/dev/null") { Ok(file) => file, Err(_) => return },
                    };
                    let _ = writeln!(log, "cancelled: tearing down {name}");
                    match read_config(config) {
                        Ok(config) => teardown(name, &config, &mut log, &self.tools, true),
                        Err(error) => { let _ = writeln!(log, "cancel teardown: {error:#}"); }
                    }
                });
            }
        });
        names.into_iter().map(|(name, _, _)| name).collect()
    }
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
            fabric: self.fabric.or(defaults.fabric),
            precheck: self.precheck.clone().or_else(|| defaults.precheck.clone()),
            prelaunch: self.prelaunch.clone().or_else(|| defaults.prelaunch.clone()),
            observer: self.observer.clone().or_else(|| defaults.observer.clone()),
            levels: self.levels.clone().or_else(|| defaults.levels.clone()),
            post_load_c1: self.post_load_c1.or(defaults.post_load_c1),
            post_load_repeat: self.post_load_repeat.or(defaults.post_load_repeat),
            prefill_lengths: self.prefill_lengths.clone().or_else(|| defaults.prefill_lengths.clone()),
            sweep_tokens: self.sweep_tokens.or(defaults.sweep_tokens),
            baseline: self.baseline.or(defaults.baseline),
            window: self.window.clone().or_else(|| defaults.window.clone()),
            lossless_set: self.lossless_set.clone().or_else(|| defaults.lossless_set.clone()),
            lossless_dump: self.lossless_dump.clone().or_else(|| defaults.lossless_dump.clone()),
            context_lengths: self.context_lengths.clone().or_else(|| defaults.context_lengths.clone()),
            context_levels: self.context_levels.clone().or_else(|| defaults.context_levels.clone()),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hooks: Option<BTreeMap<String, String>>,
    /// The client-side sweep, when the entry asked for one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep: Option<crate::sweep::Sweep>,
    #[serde(default)]
    pub fabric: Vec<crate::fabric::Phase>,
    #[serde(default)]
    pub fabric_caveats: Vec<String>,
    #[serde(default)]
    pub ab_verdict: String,
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

/// First-come turn order for the hardware locks, the same protocol as
/// scripts/hw/hwlock.py: numbered tickets in `<lock dir>/hwq`, each flocked by
/// its live owner; a ticket whose flock is free is stale and removed.
mod fair {
    use anyhow::Result;
    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};

    pub(super) struct Ticket {
        queue: PathBuf,
        seq: u64,
        path: PathBuf,
        locks: Vec<String>,
        _file: File,
    }

    /// Whether this process runs inside a live `hwlock.py --gate` ticket
    /// (`CUTEAFD_HW_GATE_TICKETS`), which already holds this job's turn.
    pub(super) fn inside_gate(dir: &Path) -> bool {
        let queue = dir.join("hwq");
        std::env::var("CUTEAFD_HW_GATE_TICKETS").unwrap_or_default().split(':')
            .filter_map(|s| s.parse::<u64>().ok())
            .any(|seq| live(&queue, seq))
    }

    fn live(queue: &Path, seq: u64) -> bool {
        let prefix = format!("{seq:012}.");
        std::fs::read_dir(queue).into_iter().flatten().flatten().any(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.starts_with(&prefix) && name.ends_with(".ticket")
                && File::open(entry.path()).is_ok_and(|f| matches!(f.try_lock(), Err(std::fs::TryLockError::WouldBlock)))
        })
    }

    impl Ticket {
        pub(super) fn take(dir: &Path, locks: &[&str]) -> Result<Self> {
            let queue = dir.join("hwq");
            std::fs::create_dir_all(&queue)?;
            let seq = {
                let guard = File::options().create(true).append(true).open(queue.join(".seq.lock"))?;
                guard.lock()?;
                let counter = queue.join("seq");
                let next = std::fs::read_to_string(&counter).ok().and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0) + 1;
                std::fs::write(&counter, next.to_string())?;
                next
            };
            let locks: Vec<String> = locks.iter().map(|n| n.trim_end_matches(".lock").to_string()).collect();
            let pid = std::process::id();
            let staged = queue.join(format!(".{seq:012}.{pid}.tmp"));
            let mut file = File::create(&staged)?;
            // Locked before it becomes visible, so it is never mistaken for stale.
            file.lock()?;
            file.write_all(serde_json::json!({"locks": locks, "pid": pid, "argv": ["cuteafd", "bench", "smoke"]}).to_string().as_bytes())?;
            let path = queue.join(format!("{seq:012}.{pid}.ticket"));
            std::fs::rename(&staged, &path)?;
            Ok(Self { queue, seq, path, locks, _file: file })
        }

        /// Whether a live, older ticket wants any of our locks (prunes stale ones).
        /// Tickets of `hwlock.py --gate` ancestors (`CUTEAFD_HW_GATE_TICKETS`)
        /// hold the turn for this process, so they never block it.
        pub(super) fn older_conflict(&self) -> Result<bool> {
            let ancestors: Vec<u64> = std::env::var("CUTEAFD_HW_GATE_TICKETS").unwrap_or_default()
                .split(':').filter_map(|s| s.parse().ok()).collect();
            self.older_conflict_except(&ancestors)
        }

        fn older_conflict_except(&self, ancestors: &[u64]) -> Result<bool> {
            let mut names: Vec<_> = std::fs::read_dir(&self.queue)?.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".ticket")).collect();
            names.sort();
            for name in names {
                let Some(seq) = name.split('.').next().and_then(|s| s.parse::<u64>().ok()) else { continue };
                let path = self.queue.join(&name);
                let Ok(mut other) = File::open(&path) else { continue };
                match other.try_lock() {
                    Ok(()) => { let _ = std::fs::remove_file(&path); continue; } // owner gone
                    Err(std::fs::TryLockError::WouldBlock) => {}
                    Err(_) => continue,
                }
                if seq >= self.seq || ancestors.contains(&seq) {
                    continue;
                }
                let mut text = String::new();
                let _ = other.read_to_string(&mut text);
                let theirs: Vec<String> = serde_json::from_str::<serde_json::Value>(&text).ok()
                    .and_then(|v| v["locks"].as_array().cloned()).unwrap_or_default()
                    .into_iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
                if theirs.iter().any(|l| self.locks.contains(l)) {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }

    impl Drop for Ticket {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Ticket;

        #[test]
        fn older_live_ticket_blocks_and_stale_ones_are_pruned() {
            let dir = tempfile::tempdir().unwrap();
            let older = Ticket::take(dir.path(), &["gpu0.lock", "gpu1.lock"]).unwrap();
            let newer = Ticket::take(dir.path(), &["gpu1.lock"]).unwrap();
            let disjoint = Ticket::take(dir.path(), &["sparks.lock"]).unwrap();
            assert!(newer.older_conflict().unwrap());
            assert!(!disjoint.older_conflict().unwrap());
            assert!(!older.older_conflict().unwrap());
            drop(older);
            assert!(!newer.older_conflict().unwrap());
            let stale = dir.path().join("hwq").join("000000000001.4242.ticket");
            std::fs::write(&stale, r#"{"locks":["gpu1"]}"#).unwrap();
            assert!(!newer.older_conflict().unwrap());
            assert!(!stale.exists());
        }

        #[test]
        fn a_child_of_a_live_gate_takes_no_ticket_of_its_own() {
            let dir = tempfile::tempdir().unwrap();
            let gate = Ticket::take(dir.path(), &["sparks.lock", "gpu0.lock"]).unwrap();
            assert!(super::live(&dir.path().join("hwq"), gate.seq));
            assert!(!super::live(&dir.path().join("hwq"), gate.seq + 1));
            drop(gate);
            assert!(!super::live(&dir.path().join("hwq"), 1));
        }

        #[test]
        fn a_gated_parent_ticket_does_not_block_its_child() {
            let dir = tempfile::tempdir().unwrap();
            let gate = Ticket::take(dir.path(), &["sparks.lock", "gpu0.lock", "gpu1.lock"]).unwrap();
            let child = Ticket::take(dir.path(), &["sparks.lock", "gpu1.lock"]).unwrap();
            assert!(child.older_conflict_except(&[]).unwrap());
            assert!(!child.older_conflict_except(&[gate.seq]).unwrap());
            // Another job's older ticket still blocks the child.
            let other = Ticket::take(dir.path(), &["gpu1.lock"]).unwrap();
            let late = Ticket::take(dir.path(), &["gpu1.lock"]).unwrap();
            assert!(late.older_conflict_except(&[gate.seq]).unwrap());
            drop(other);
        }
    }
}

/// Lock order: Spark pool, GPU0, GPU1, then the two out-of-pool hosts.
fn locks_for(entry: &Entry) -> Vec<&'static str> {
    let gpus = entry.gpus.clone().unwrap_or_else(|| vec![0]);
    let mut locks = Vec::new();
    let pool = entry.sparks.as_ref().is_some_and(|hosts| hosts.iter().any(|h| matches!(h.as_str(), "ostrich" | "dodo" | "emu" | "kiwi")));
    if gpus.contains(&0) || pool || entry.exclusive() {
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
    fn take(&mut self, names: &[&'static str], wait: bool, cancelled: &dyn Fn() -> Option<i32>) -> Result<bool> {
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
        // A deep shared queue holds the pool for hours, and a fair-queue ticket holds no
        // lock while it waits: default to the same 6 h bound as golden-run and hwlock.py.
        // CUTEAFD_SMOKE_LOCK_WAIT_S, then CUTEAFD_HW_LOCK_WAIT_S, override it.
        let wait = ["CUTEAFD_SMOKE_LOCK_WAIT_S", "CUTEAFD_HW_LOCK_WAIT_S"].iter()
            .find_map(|k| std::env::var(k).ok().and_then(|s| s.parse().ok())).unwrap_or(21600);
        let deadline = Instant::now() + Duration::from_secs(wait);
        // A parent `hwlock.py` already holding a lock for us names it here; taking it
        // again would deadlock on our own parent.
        let parent: Vec<String> = std::env::var("CUTEAFD_HW_LOCKS_HELD").unwrap_or_default()
            .split(',').filter(|s| !s.is_empty()).map(|s| format!("{}.lock", s.trim_end_matches(".lock"))).collect();
        let wanted: Vec<&'static str> = ordered.iter().copied()
            .filter(|n| !self.held.contains_key(n) && !parent.iter().any(|p| p == n)).collect();
        if wanted.is_empty() {
            return Ok(true);
        }
        // Fair turn order shared with scripts/hw/hwlock.py: a waiter takes its
        // locks only when no older live ticket wants any of them. Inside an
        // `hwlock.py --gate`, the gate already won this job's turn: a new,
        // younger ticket would queue behind jobs that themselves wait on the
        // gate (a deadlock), so a gated child only waits for the flocks.
        let ticket = if fair::inside_gate(&dir) { None } else { Some(fair::Ticket::take(&dir, &wanted)?) };
        loop {
            if !ticket.as_ref().map_or(Ok(false), |t| t.older_conflict())? {
                let mut taken = Vec::new();
                for &name in &wanted {
                    let file = File::options().create(true).append(true).open(dir.join(name))?;
                    match file.try_lock() {
                        Ok(()) => taken.push((name, file)),
                        Err(std::fs::TryLockError::WouldBlock) => break,
                        Err(error) => bail!("locking {name}: {error}"),
                    }
                }
                if taken.len() == wanted.len() {
                    for (name, file) in taken {
                        self.held.insert(name, (file, 0));
                    }
                    break;
                }
            }
            if let Some(signal) = cancelled() {
                return Err(Cancelled(signal).into());
            }
            if Instant::now() >= deadline {
                bail!("timed out after {wait} s waiting for a fair turn on {}", wanted.join(","));
            }
            std::thread::sleep(Duration::from_secs(2));
        }
        drop(ticket);
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

/// `post_load_c1` runs after the widest level; `post_load_repeat` replays the levels C1.
fn check_post_load(entry: &Entry) -> Result<()> {
    if entry.post_load_c1 == Some(true) && entry.levels.is_none() {
        bail!("{}: post_load_c1 needs levels (it runs after the widest one)", entry.name);
    }
    if entry.post_load_repeat == Some(true) && entry.post_load_c1 != Some(true) {
        bail!("{}: post_load_repeat needs post_load_c1 (it replays the levels C1 there)", entry.name);
    }
    Ok(())
}

pub fn run(args: SmokeArgs) -> Result<()> {
    let matrix: Matrix = serde_json::from_str(&std::fs::read_to_string(&args.matrix)
        .with_context(|| format!("{}", args.matrix.display()))?).context("parsing the matrix")?;
    crate::publish::validate_retired(&matrix.retired)?;
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
        crate::profiles::validate_selection(entry.profile.as_deref().or(matrix.profile.as_deref()).unwrap_or("smoke"))
            .map_err(anyhow::Error::msg)?;
        if let Some(levels) = &entry.levels {
            let text = levels.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
            crate::sweep::parse_levels(&text).map_err(|e| anyhow::anyhow!("{}: {e}", entry.name))?;
        }
        if let Some(lengths) = &entry.prefill_lengths {
            let text = lengths.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
            crate::sweep::parse_lengths(&text).map_err(|e| anyhow::anyhow!("{}: {e}", entry.name))?;
        }
        check_post_load(&entry)?;
        if entry.window.is_some() && entry.levels.is_none() && entry.prefill_lengths.is_none() {
            bail!("{}: window needs levels or prefill_lengths (it frames the sweep's timed requests)", entry.name);
        }
        let set = entry.lossless_set.as_ref().map(|path| crate::panels::lossless::CaseSet::load(&args.repo.join(path))
            .with_context(|| format!("{}: lossless_set", entry.name))).transpose()?;
        if let Some(dump) = &entry.lossless_dump {
            dump.validate(&set.unwrap_or_else(crate::panels::lossless::CaseSet::default_set))
                .map_err(|e| anyhow::anyhow!("{}: lossless_dump: {e}", entry.name))?;
        }
        let defaults = crate::panels::context_load::Options::default();
        crate::panels::context_load::Options {
            lengths: entry.context_lengths.clone().unwrap_or(defaults.lengths),
            levels: entry.context_levels.clone().unwrap_or(defaults.levels),
        }.validate().map_err(|e| anyhow::anyhow!("{}: {e}", entry.name))?;
        entries.push(entry);
    }
    // Benchmark controls and inference on the launched servers take an API key:
    // an entry's own API_KEY_FILE is used unchanged, else one private key per
    // state directory (created on the first real run, never printed).
    let smoke_key = assign_api_keys(&mut entries, &state_dir);
    if !args.dry_run && entries.iter().any(|e| e.set.get("API_KEY_FILE").is_some_and(|f| Path::new(f) == smoke_key)) {
        ensure_key_file(&smoke_key)?;
        eprintln!("smoke API key file: {}", smoke_key.display());
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
            eprintln!("  {} api_key_file={} family={} gpus={:?} sparks={:?} locks={:?} levels={:?} post_load_c1={} post_load_repeat={} prefill_lengths={:?} sweep_tokens={} baseline={} window={} lossless_set={} lossless_dump={:?} context_lengths={:?} context_levels={:?} fabric={} worker_stop={} set={:?}",
                e.name, e.set.get("API_KEY_FILE").map_or("-", String::as_str), e.family.as_deref().unwrap_or("?"), e.gpus, e.sparks, locks_for(e), e.levels,
                e.post_load_c1.unwrap_or(false), e.post_load_repeat.unwrap_or(false), e.prefill_lengths,
                e.sweep_tokens.unwrap_or(crate::sweep::TOKENS), e.baseline.unwrap_or(true), e.window.is_some(),
                e.lossless_set.as_deref().unwrap_or("default"), e.lossless_dump, e.context_lengths, e.context_levels, e.fabric.unwrap_or(true),
                WORKER_STOP, e.set);
        }
        return Ok(());
    }
    let profile = matrix.profile.clone().unwrap_or_else(|| "smoke".into());
    let inflight = Arc::new(Inflight::default());
    inflight.install()?;
    let mut queue: std::collections::VecDeque<Entry> = pending.into();
    let mut running: Vec<Entry> = Vec::new();
    let mut locks = Locks::default();
    let (done_tx, done_rx) = mpsc::channel::<(String, Outcome)>();
    let started = Instant::now();
    let mut waiting_note = Instant::now() - Duration::from_secs(600);
    while !queue.is_empty() || !running.is_empty() {
        if let Some(signal) = inflight.cancelled() {
            return cancel_run(&inflight, signal, &running, &mut state, &paths, &state_dir, &entries, &build, started);
        }
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
                if locks.take(&needed, running.is_empty(), &|| inflight.cancelled())? {
                    queue.remove(index);
                    running.push(entry.clone());
                    started_one = true;
                    let (tx, repo, out_root, profile, build) = (done_tx.clone(), repo.clone(), out_root.clone(),
                        entry.profile.clone().unwrap_or_else(|| profile.clone()), build.clone());
                    let (configs, logs) = (paths.configs.clone(), paths.logs.clone());
                    let parallel = args.parallel > 1;
                    let inflight = inflight.clone();
                    std::thread::spawn(move || {
                        let outcome = run_entry(&entry, &repo, &out_root, &configs, &logs, &profile, &build, parallel, &inflight);
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
        match done_rx.recv_timeout(Duration::from_secs(1)) {
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
    if let Some(signal) = inflight.cancelled() {
        return cancel_run(&inflight, signal, &running, &mut state, &paths, &state_dir, &entries, &build, started);
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

/// A signal cancelled the run: tear down everything launched (bounded), record
/// the running entries as cancelled and return [`Cancelled`]. The caller's
/// locks are released only after this returns, once the containers are gone.
#[allow(clippy::too_many_arguments)]
fn cancel_run(inflight: &Inflight, signal: i32, running: &[Entry], state: &mut BTreeMap<String, Outcome>, paths: &Paths,
    state_dir: &Path, entries: &[Entry], build: &str, started: Instant) -> Result<()> {
    let reason = Cancelled(signal);
    eprintln!("smoke: {reason}; tearing down {} running entr{}", running.len(), if running.len() == 1 { "y" } else { "ies" });
    let torn = inflight.cancel_all();
    for entry in running {
        let error = if torn.contains(&entry.name) { format!("{reason}; its containers were torn down") } else { reason.to_string() };
        state.insert(entry.name.clone(), Outcome { key: entry.key(), build: build.to_string(), status: "cancelled".into(),
            error: Some(error), finished: crate::report::now_rfc3339(), ..Outcome::default() });
    }
    let _ = save_state(&paths.state, state);
    let summary = summary(entries, state, build, started.elapsed().as_secs_f64());
    let _ = std::fs::write(state_dir.join("summary.md"), &summary);
    println!("{summary}");
    Err(reason.into())
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
    for e in entries {
        if let Some(o) = state.get(&e.name) {
            for phase in &o.fabric {
                out.push_str(&format!("\nFabric {} / {} ({}): NIC / engine / SparkNest / headers estimate (engine lower bound) / residual bytes (tx+rx per host; ACK/retry/skew remain residual)\n", e.name, phase.name, if phase.timed { "timed" } else { "envelope" }));
                for p in &phase.ports {
                    let pair = |a: Option<u64>, b: Option<u64>| a.zip(b).map(|(a,b)| a.saturating_add(b).to_string()).unwrap_or_else(|| "unknown".into());
                    out.push_str(&format!("- {} {}/{}: {} / {} / {} / {} / {}\n", p.host, p.device, p.port,
                        pair(p.sent_bytes,p.received_bytes), pair(p.engine_sent_bytes,p.engine_received_bytes),
                        pair(p.sparknest_sent_bytes,p.sparknest_received_bytes), pair(p.estimated_header_sent_bytes,p.estimated_header_received_bytes),
                        pair(p.foreign_sent_bytes,p.foreign_received_bytes)));
                }
            }
            for caveat in &o.fabric_caveats { out.push_str(&format!("\nA/B caveat {}: {caveat}\n",e.name)); }
        }
    }
    out
}

/// Every entry serves an API key: its own API_KEY_FILE, else `<state>/api-key`
/// (returned). Keys only; the file is created by [`ensure_key_file`].
fn assign_api_keys(entries: &mut [Entry], state_dir: &Path) -> PathBuf {
    let smoke_key = state_dir.join("api-key");
    for entry in entries {
        entry.set.entry("API_KEY_FILE".into()).or_insert_with(|| smoke_key.display().to_string());
    }
    smoke_key
}

/// A private (0600) random key at `path`, created once and then reused; refuses a
/// symlink or a file other users can read.
fn ensure_key_file(path: &Path) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    match std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path) {
        Ok(mut file) => {
            let key = format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple());
            file.write_all(format!("{key}\n").as_bytes())?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).with_context(|| format!("{}", path.display())),
    }
    let meta = std::fs::symlink_metadata(path)?;
    if meta.file_type().is_symlink() || meta.permissions().mode() & 0o077 != 0 {
        bail!("{} must be a private (0600) regular file", path.display());
    }
    Ok(())
}

/// A config for the entry: the base file plus the entry's keys (later lines win).
fn write_config(entry: &Entry, repo: &Path, configs: &Path, parallel: bool) -> Result<PathBuf> {
    let base = entry.config.as_ref().map(|c| repo.join(c)).unwrap_or_else(|| repo.join("cuteafd.config"));
    let mut text = std::fs::read_to_string(&base).with_context(|| format!("{}", base.display()))?;
    text.push_str(&format!("\n# bench smoke entry {}\n", entry.name));
    let mut set = entry.set.clone();
    // Smoke drives the run through /v1/bench, which the launcher serves only on request.
    set.insert("ENABLE_BENCH".into(), "on".into());
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

/// `program args`, under `timeout` when bounded (a cancellation must finish
/// within [`CANCEL_TEARDOWN_S`]); unbounded otherwise, since a profiled
/// container's own stop timeout may be long.
fn tool(program: &Path, args: &[&str], bound: Option<u64>) -> Command {
    let mut command = match bound {
        Some(seconds) => {
            let mut command = Command::new("timeout");
            command.args([format!("--kill-after={KILL_AFTER_S}s"), format!("{seconds}s")]).arg(program);
            command
        }
        None => Command::new(program),
    };
    command.args(args);
    command
}

/// Removes the entry's coordinator and expert containers; the Spark workers in
/// parallel with the coordinator. `cancelled` bounds every step (see
/// [`CANCEL_TEARDOWN_S`]), including a graceful stop's own timeout.
fn teardown(name: &str, config: &HashMap<String, String>, log: &mut File, tools: &Tools, cancelled: bool) {
    let bound = |seconds: u64| cancelled.then_some(seconds);
    let stop_timeout = cancelled.then_some(STOP_S);
    let instance = config.get("INSTANCE").filter(|v| !v.is_empty());
    let coordinator = match instance {
        Some(instance) => format!("cuteafd-coordinator-{instance}"),
        None => "cuteafd-coordinator".into(),
    };
    let base = log_path(log).ok();
    let port = config.get("EXPERT_PORT").cloned().unwrap_or_else(|| "19441".into());
    let count: usize = config.get("SPARK_COUNT").and_then(|v| v.parse().ok()).unwrap_or(0);
    let hosts: Vec<&String> = (0..count).filter_map(|rank| config.get(&format!("SPARK_{rank}_HOST"))).collect();
    let ssh = |host: &str, remote: &[&str], seconds: u64| {
        let mut args = vec!["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host];
        args.extend_from_slice(remote);
        tool(&tools.ssh, &args, bound(seconds))
    };
    std::thread::scope(|scope| {
        for host in &hosts {
            let (base, port, ssh) = (&base, &port, &ssh);
            scope.spawn(move || {
                let worker = format!("cuteafd-spark-expert-{host}-{port}");
                if let Some(file) = base.as_ref().and_then(|p| File::create(p.with_extension(format!("{host}.log"))).ok()) {
                    let _ = ssh(host, &["docker", "logs", &worker], LOGS_S)
                        .stdout(file.try_clone().map(Stdio::from).unwrap_or(Stdio::null()))
                        .stderr(Stdio::from(file)).status();
                }
                let _ = ssh(host, &[&worker_teardown_script(&worker, stop_timeout)], STOP_S + RM_S + 5)
                    .stdout(Stdio::null()).stderr(Stdio::null()).status();
            });
        }
        // A launcher that labels its coordinator cuteafd.graceful-stop=1 (a process
        // that writes output at exit, e.g. under a profiler) gets `docker stop`, with
        // the container's own stop signal and timeout, before the forced removal.
        let label = tool(&tools.docker, &["inspect", "-f", "{{index .Config.Labels \"cuteafd.graceful-stop\"}}", &coordinator], bound(INSPECT_S))
            .stderr(Stdio::null()).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
        if label.as_deref() == Some("1") {
            let timeout = stop_timeout.map(|t| t.to_string());
            let mut args = vec!["stop"];
            if let Some(timeout) = &timeout {
                args.extend(["-t", timeout]);
            }
            args.push(&coordinator);
            let stopped = tool(&tools.docker, &args, bound(STOP_S + 5)).stdout(Stdio::null()).stderr(Stdio::null()).status();
            let _ = writeln!(log, "graceful stop {coordinator}: {}", stopped.map(|s| s.to_string()).unwrap_or_else(|e| e.to_string()));
        }
        // The coordinator's own log goes next to the entry's (failures and warnings outlive the container).
        if let Some(file) = base.as_ref().and_then(|p| File::create(p.with_extension("coordinator.log")).ok()) {
            let _ = tool(&tools.docker, &["logs", &coordinator], bound(LOGS_S)).stdout(file.try_clone().map(Stdio::from)
                .unwrap_or(Stdio::null())).stderr(Stdio::from(file)).status();
        }
        let _ = tool(&tools.docker, &["rm", "-f", &coordinator], bound(RM_S)).stdout(Stdio::null()).stderr(Stdio::null()).status();
    });
    let _ = writeln!(log, "teardown {name} ({coordinator})");
}

/// Advertised by `--dry-run` (`worker_stop=graceful`): wip-cards refuses a
/// PROFILE_SPARK card on an older binary that would `rm -f` a profiled worker.
const WORKER_STOP: &str = "graceful";

/// Removes one Spark worker. A worker labelled cuteafd.graceful-stop=1 (a
/// profiled worker, PROFILE_SPARK) first gets `docker stop`, with its own stop
/// signal and timeout (or `stop_timeout` seconds, on a cancellation), so the
/// profiler writes its report before the removal.
fn worker_teardown_script(worker: &str, stop_timeout: Option<u64>) -> String {
    let stop = stop_timeout.map(|t| format!("-t {t} ")).unwrap_or_default();
    format!("ids=$(docker ps -aq --filter name=^{worker}$); \
        if [ -n \"$ids\" ] && [ \"$(docker inspect -f '{{{{index .Config.Labels \"cuteafd.graceful-stop\"}}}}' {worker} 2>/dev/null)\" = 1 ]; then \
        docker stop {stop}{worker} >/dev/null 2>&1 || true; fi; \
        [ -z \"$ids\" ] || docker rm -f $ids >/dev/null 2>&1 || true")
}

#[allow(clippy::too_many_arguments)]
fn run_entry(entry: &Entry, repo: &Path, out_root: &Path, configs: &Path, logs: &Path, profile: &str, build: &str,
    parallel: bool, inflight: &Arc<Inflight>) -> Outcome {
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
    let mut observer = None;
    let mut launcher_started = false;
    let mut hook_results = BTreeMap::new();
    // The window's state: opened and not yet closed successfully.
    let window_open = std::cell::Cell::new(false);
    let window_results = std::cell::RefCell::new(BTreeMap::<String, String>::new());
    let window_hook = |kind: &str, hook: &Hook, url: &str, config: &Path| -> Result<()> {
        let checked = HookChild::spawn(hook, url, config, &log_path, kind, inflight).and_then(|mut child| child.wait());
        window_results.borrow_mut().insert(kind.into(), checked.as_ref().map(|_| "pass".into())
            .unwrap_or_else(|error| format!("{error:#}")));
        checked.with_context(|| format!("{kind} hook"))
    };
    let close_window = |url: &str, config: &Path| -> Result<()> {
        match entry.window.as_ref().filter(|_| window_open.get()) {
            Some(window) => {
                window_hook("window-close", &window.close, url, config)?;
                window_open.set(false);
                Ok(())
            }
            None => Ok(()),
        }
    };
    let result = (|| -> Result<Option<Report>> {
        let config_path = write_config(entry, repo, configs, parallel)?;
        let config = read_config(&config_path)?;
        let port = entry.port(&config);
        let mut command = Command::new(repo.join("run.sh"));
        command.arg("--config").arg(&config_path).arg("--restart").args(entry.run_args.clone().unwrap_or_default())
            .current_dir(repo).stdin(Stdio::null()).stdout(log.try_clone()?).stderr(log.try_clone()?);
        let url = format!("http://127.0.0.1:{port}");
        if let Some(hook) = &entry.prelaunch {
            let checked = (|| -> Result<()> {
                let mut child = HookChild::spawn(hook, &url, &config_path, &log_path, "prelaunch", inflight)?;
                child.wait()
            })();
            hook_results.insert("prelaunch".into(), checked.as_ref().map(|_| "pass".into())
                .unwrap_or_else(|error| format!("{error:#}")));
            checked.context("prelaunch hook failed before launcher")?;
        }
        let launch = Instant::now();
        if let Some(hook) = &entry.observer {
            match HookChild::spawn(hook, &url, &config_path, &log_path, "observer", inflight) {
                Ok(child) => observer = Some(child),
                Err(error) => {
                    hook_results.insert("observer".into(), format!("{error:#}"));
                    if hook.required { return Err(error.context("required observer hook")); }
                }
            }
        }
        // Registered before it starts: a cancellation tears down what it launches.
        inflight.launch(&entry.name, &mut command, &config_path, &log_path)?;
        launcher_started = true;
        let timeout = Duration::from_secs(entry.timeout_s.unwrap_or(900));
        // run.sh waits for readiness itself; poll /health too in case it returns early.
        loop {
            if let Some(signal) = inflight.cancelled() {
                return Err(Cancelled(signal).into());
            }
            if let Some(status) = inflight.launcher_status(&entry.name)? {
                if !status.success() {
                    bail!("./run.sh exited {status} (log {})", log_path.display());
                }
                if health(&url) {
                    break;
                }
            }
            if launch.elapsed() > timeout {
                inflight.stop_launcher(&entry.name);
                bail!("not ready within {} s", timeout.as_secs());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        outcome.launch_s = Some(launch.elapsed().as_secs_f64());
        if let Some(hook) = &entry.precheck {
            let checked = (|| -> Result<()> {
                let mut child = HookChild::spawn(hook, &url, &config_path, &log_path, "precheck", inflight)?;
                child.wait()
            })();
            hook_results.insert("precheck".into(), checked.as_ref().map(|_| "pass".into())
                .unwrap_or_else(|error| format!("{error:#}")));
            checked.context("precheck hook failed before workload")?;
        }
        // `panels:a,b` runs those panels instead of a profile; a bare `panels:`
        // runs no server-side profile at all (the sweep alone, when set).
        let (profile, panels): (Option<String>, Option<Vec<String>>) = match profile.strip_prefix("panels:") {
            Some(list) => (None, Some(list.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect())),
            None => (Some(profile.to_string()), None),
        };
        // An explicit CUTEAFD_API_KEY wins; otherwise the key file this launch serves.
        let api_key = std::env::var("CUTEAFD_API_KEY").ok().or_else(|| config.get("API_KEY_FILE")
            .and_then(|file| std::fs::read_to_string(file).ok()).map(|key| key.trim().to_string()).filter(|k| !k.is_empty()));
        let sweeping = entry.levels.is_some() || entry.prefill_lengths.is_some();
        let run_sweep = || -> Result<crate::sweep::Sweep> {
            let open: crate::sweep::OnTimed = entry.window.as_ref().map(|window| -> Box<dyn FnOnce() -> Result<()> + '_> {
                Box::new(|| {
                    window_open.set(true);
                    window_hook("window-open", &window.open, &url, &config_path)
                })
            });
            let sweep = crate::sweep::run_windowed(&url, api_key.as_deref(), entry.levels.as_deref().unwrap_or(&[]),
                entry.post_load_c1.unwrap_or(false), entry.post_load_repeat.unwrap_or(false), entry.prefill_lengths.as_deref().unwrap_or(&[]), entry.fabric.unwrap_or(true),
                entry.sweep_tokens.unwrap_or(crate::sweep::TOKENS), open);
            let closed = close_window(&url, &config_path);
            let sweep = sweep?;
            closed?;
            Ok(sweep)
        };
        if sweeping && panels.as_ref().is_some_and(Vec::is_empty) {
            let sweep = run_sweep()?;
            let dir = out_root.join("sweep-only").join(&entry.name);
            std::fs::create_dir_all(&dir)?;
            std::fs::write(dir.join("sweep.json"), serde_json::to_vec_pretty(&sweep)?)?;
            outcome.dir = Some(dir.display().to_string());
            outcome.fabric.extend(sweep.points.iter().filter_map(|p| p.fabric.as_ref())
                .chain(sweep.prefill.iter().filter_map(|p| p.fabric.as_ref()))
                .filter_map(|p| serde_json::from_value::<crate::fabric::Phase>(p.clone()).ok()));
            outcome.sweep = Some(sweep);
            return Ok(None);
        }
        let options = crate::cli::RunOptions { url: url.clone(), profile, panels, passes: vec![], baseline: entry.baseline, fabric: Some(entry.fabric.unwrap_or(true)),
            export: vec!["svg".into(), "card".into(), "json".into()], out: None, label: Some(entry.name.clone()),
            root: out_root.to_path_buf(),
            api_key: api_key.clone(), quiet: true,
            deadline: Some(Duration::from_secs(entry.run_timeout_s.unwrap_or(900))),
            lossless_set: entry.lossless_set.as_ref().map(|p| crate::panels::lossless::CaseSet::load(&repo.join(p)))
                .transpose()?, lossless_dump: entry.lossless_dump.clone(), context_lengths: entry.context_lengths.clone(), context_levels: entry.context_levels.clone() };
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
        if sweeping {
            let sweep = run_sweep()?;
            std::fs::write(dir.join("sweep.json"), serde_json::to_vec_pretty(&sweep)?)?;
            report.fabric.extend(sweep.points.iter().filter_map(|p| p.fabric.as_ref())
                .chain(sweep.prefill.iter().filter_map(|p| p.fabric.as_ref()))
                .filter_map(|p| serde_json::from_value::<crate::fabric::Phase>(p.clone()).ok()));
            crate::cli::write_exports(&report, &dir, &options.export)?;
            outcome.sweep = Some(sweep);
        }
        Ok(Some(report))
    })();
    // A window still open (its close failed, or the sweep stopped between the
    // two) gets one more close while the server is still up.
    if launcher_started && window_open.get() {
        if let Ok(config_path) = write_config(entry, repo, configs, parallel) {
            let url = format!("http://127.0.0.1:{}", read_config(&config_path).map(|c| entry.port(&c)).unwrap_or(8000));
            if let Err(error) = close_window(&url, &config_path) {
                let _ = writeln!(log, "warning: window close before teardown failed: {error:#}");
            }
        }
    }
    hook_results.extend(window_results.into_inner());
    if let Some(mut child) = observer.take() {
        let observed = child.finish();
        hook_results.insert("observer".into(), observed.as_ref().map(|_| "pass".into())
            .unwrap_or_else(|error| format!("{error:#}")));
        if let Err(error) = observed {
            if entry.observer.as_ref().is_some_and(|h| h.required) {
                outcome.error = Some(format!("required observer hook failed: {error:#}"));
            }
        }
    }
    if !hook_results.is_empty() {
        outcome.hooks = Some(hook_results);
        // Failure diagnostics exist even when no benchmark report was produced.
        let _ = std::fs::write(log_path.with_extension("hooks.json"), serde_json::to_vec_pretty(&outcome.hooks).unwrap_or_default());
    }
    match result {
        Ok(None) => {
            outcome.status = if outcome.error.is_some() { "failed" } else { "done" }.into();
            outcome.code_tok_s = outcome.sweep.as_ref().and_then(|s| s.rate(1, "levels"));
            let running = Command::new(&inflight.tools.docker).args(["inspect", "-f", "{{.State.Running}}", &coordinator_name(entry, parallel)])
                .output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim() == "true");
            if running == Some(false) {
                outcome.error = Some("the server exited during or after the run (see its coordinator log)".into());
                outcome.status = "failed".into();
            }
        }
        Ok(Some(report)) => {
            outcome.status = if outcome.error.is_some() { "failed" } else { "done" }.into();
            if let (Some(hooks), Some(dir)) = (&outcome.hooks, &outcome.dir) {
                let path = Path::new(dir).join("report.json");
                if let Ok(mut value) = serde_json::to_value(&report) {
                    value["smoke_hooks"] = serde_json::to_value(hooks).unwrap_or_default();
                    let _ = std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap_or_default());
                }
            }
            outcome.fabric.extend(report.fabric.iter().cloned());
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
            let running = Command::new(&inflight.tools.docker).args(["inspect", "-f", "{{.State.Running}}", &coordinator_name(entry, parallel)])
                .output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim() == "true");
            if running == Some(false) {
                outcome.error = Some("the server exited during or after the run (see its coordinator log)".into());
                let _ = writeln!(log, "warning: the coordinator is not running any more");
            }
        }
        Err(error) => {
            outcome.error = Some(format!("{error:#}"));
            let _ = writeln!(log, "error: {error:#}");
            // Only collect serving diagnostics after this entry launched containers
            // (not on a cancellation: the bounded teardown collects the logs).
            if launcher_started && inflight.cancelled().is_none() {
                let tail = Command::new(&inflight.tools.docker).args(["logs", "--tail", "40", &coordinator_name(entry, parallel)]).output();
                if let Ok(tail) = tail {
                    let _ = log.write_all(&tail.stdout);
                    let _ = log.write_all(&tail.stderr);
                }
            }
        }
    }
    if outcome.hooks.is_some() && outcome.dir.is_none() {
        let failure = serde_json::json!({"entry": entry.name, "status": outcome.status,
            "error": outcome.error, "smoke_hooks": outcome.hooks});
        let _ = std::fs::write(log_path.with_extension("report.json"), serde_json::to_vec_pretty(&failure).unwrap_or_default());
    }
    // A rejected prelaunch hook never owns any serving containers.
    // A cancelled run's teardown belongs to the scheduler (Inflight::cancel_all,
    // bounded, all entries at once): the entry stays registered for it. A
    // cancellation arriving during this teardown repeats it there, bounded.
    if launcher_started && inflight.cancelled().is_none() {
        inflight.stop_launcher(&entry.name);
        if let Ok(config) = read_config(&configs.join(format!("{}.config", entry.name))) {
            teardown(&entry.name, &config, &mut log, &inflight.tools, false);
        }
        inflight.finished(&entry.name);
    }
    if let Some(signal) = inflight.cancelled().filter(|_| outcome.status != "done") {
        outcome.status = "cancelled".into();
        outcome.error.get_or_insert_with(|| Cancelled(signal).to_string());
    }
    outcome.fabric_caveats = crate::fabric::caveats(&outcome.fabric);
    outcome.ab_verdict = if outcome.fabric_caveats.is_empty() { "eligible" } else { "rerun" }.into();
    for caveat in &outcome.fabric_caveats { let _ = writeln!(log, "fabric A/B caveat: {caveat}"); }
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
    #[test]
    fn release_smoke_entries_all_serve_an_api_key() {
        use super::*;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        for matrix in ["scripts/bench/release-smoke-v3.json", "scripts/bench/release-smoke-v4.json", "scripts/bench/release-smoke.example.json"] {
            let matrix: Matrix = serde_json::from_str(&std::fs::read_to_string(root.join(matrix)).unwrap()).unwrap();
            let mut entries: Vec<Entry> = matrix.entries.iter().map(|e| e.merged(&matrix.defaults)).collect();
            let state = tempfile::tempdir().unwrap();
            let key = assign_api_keys(&mut entries, state.path());
            assert!(!entries.is_empty());
            for entry in &entries {
                assert_eq!(entry.set.get("API_KEY_FILE").map(PathBuf::from), Some(key.clone()), "{}", entry.name);
            }
            // An entry's own key file is kept.
            let mut own = entries[0].clone();
            own.set.insert("API_KEY_FILE".into(), "/operator/key".into());
            assign_api_keys(std::slice::from_mut(&mut own), state.path());
            assert_eq!(own.set["API_KEY_FILE"], "/operator/key");
            ensure_key_file(&key).unwrap();
            let first = std::fs::read_to_string(&key).unwrap();
            ensure_key_file(&key).unwrap();
            assert_eq!(std::fs::read_to_string(&key).unwrap(), first, "reused, not rotated");
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&key).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }

    #[test]
    fn post_load_repeat_needs_post_load_c1_and_merges() {
        let named = |name: &str| Entry { name: name.into(), levels: Some(vec![1, 16]), ..Entry::default() };
        let bad = Entry { post_load_repeat: Some(true), ..named("a") };
        assert!(check_post_load(&bad).unwrap_err().to_string().contains("needs post_load_c1"));
        let good = Entry { post_load_c1: Some(true), post_load_repeat: Some(true), ..named("a") };
        check_post_load(&good).unwrap();
        check_post_load(&named("a")).unwrap();
        let defaults = Entry { post_load_repeat: Some(true), ..Entry::default() };
        assert_eq!(Entry::default().merged(&defaults).post_load_repeat, Some(true));
        let own = Entry { post_load_repeat: Some(false), ..Entry::default() };
        assert_eq!(own.merged(&defaults).post_load_repeat, Some(false));
    }

    use super::*;

    fn entry(name: &str, gpus: &[u32], sparks: &[&str], port: u16) -> Entry {
        Entry { name: name.into(), family: Some("glm5_flash".into()), gpus: Some(gpus.to_vec()),
            sparks: Some(sparks.iter().map(|s| s.to_string()).collect()),
            set: [("ADDR".to_string(), format!("0.0.0.0:{port}"))].into(), ..Entry::default() }
    }

    #[test]
    fn context_options_merge_and_roundtrip() {
        let defaults = Entry { context_lengths: Some(vec![32768, 131072]), context_levels: Some(vec![1, 8, 16]), ..Entry::default() };
        let own = Entry { context_levels: Some(vec![1, 8]), ..Entry::default() };
        let merged = own.merged(&defaults);
        assert_eq!(merged.context_lengths, defaults.context_lengths);
        assert_eq!(merged.context_levels, Some(vec![1, 8]));
        let encoded = serde_json::to_value(&merged).unwrap();
        assert_eq!(encoded["context_lengths"], serde_json::json!([32768, 131072]));
        assert_eq!(serde_json::from_value::<Entry>(encoded).unwrap(), merged);
    }

    #[test]
    fn absent_hooks_preserve_serialized_keys_and_outputs() {
        let original = serde_json::json!({"name":"legacy","family":null,"config":null,"set":{},
            "run_args":null,"gpus":null,"sparks":null,"timeout_s":null,"run_timeout_s":null,
            "exclusive":null,"profile":null});
        let entry: Entry = serde_json::from_value(original.clone()).unwrap();
        assert_eq!(serde_json::to_value(&entry).unwrap(), original);
        assert_eq!(entry.merged(&Entry::default()), entry);
        let outcome = Outcome::default();
        assert!(serde_json::to_value(outcome).unwrap().get("hooks").is_none());
    }

    #[test]
    fn prelaunch_hook_merges_and_changes_resume_key() {
        let hook = Hook { argv: vec!["true".into()], timeout_s: 2, required: true,
            cleanup: None, finish_timeout_s: None };
        let original = entry("table", &[1], &[], 8000);
        let defaults = Entry { prelaunch: Some(hook.clone()), ..Entry::default() };
        let merged = original.merged(&defaults);
        assert_eq!(merged.prelaunch, Some(hook));
        assert_ne!(original.key(), merged.key());
        assert_eq!(serde_json::from_value::<Entry>(serde_json::to_value(&merged).unwrap()).unwrap(), merged);
        let override_hook = Hook { argv: vec!["false".into()], ..merged.prelaunch.clone().unwrap() };
        let override_entry = Entry { prelaunch: Some(override_hook.clone()), ..original };
        assert_eq!(override_entry.merged(&defaults).prelaunch, Some(override_hook));
    }

    #[test]
    fn worker_teardown_stops_a_labelled_worker_before_removing_it() {
        let dir = std::env::temp_dir().join(format!("cuteafd-teardown-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let docker = dir.join("docker");
        // A fake docker: logs each call; `ps` finds the worker, `inspect` prints LABEL.
        std::fs::write(&docker, "#!/bin/sh\necho \"$*\" >>\"$LOG\"\ncase \"$1\" in ps) echo id1;; inspect) printf '%s\\n' \"$LABEL\";; esac\n").unwrap();
        std::fs::set_permissions(&docker, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        for (label, stopped, bound) in [("1", true, None), ("", false, None), ("1", true, Some(15))] {
            let script = worker_teardown_script("cuteafd-spark-expert-kiwi-19441", bound);
            let log = dir.join(format!("log-{stopped}-{bound:?}"));
            let status = Command::new("sh").arg("-c").arg(&script)
                .env("PATH", format!("{}:/usr/bin:/bin", dir.display())).env("LOG", &log).env("LABEL", label)
                .status().unwrap();
            assert!(status.success());
            let calls = std::fs::read_to_string(&log).unwrap();
            let lines: Vec<&str> = calls.lines().collect();
            assert!(lines.iter().any(|l| l.starts_with("inspect -f {{index .Config.Labels \"cuteafd.graceful-stop\"}} cuteafd-spark-expert-kiwi-19441")), "{calls}");
            let stop = match bound { Some(t) => format!("stop -t {t} cuteafd-spark-expert-kiwi-19441"), None => "stop cuteafd-spark-expert-kiwi-19441".into() };
            assert_eq!(lines.iter().any(|l| *l == stop), stopped, "{calls}");
            assert_eq!(lines.last(), Some(&"rm -f id1"), "{calls}");
            if stopped {
                let stop = lines.iter().position(|l| l.starts_with("stop ")).unwrap();
                assert!(stop < lines.len() - 1);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Fakes for teardown: `docker` and `ssh` append their argv to `log`.
    fn fake_tools(dir: &Path, log: &Path) -> Tools {
        for name in ["docker", "ssh"] {
            let path = dir.join(name);
            std::fs::write(&path, format!("#!/bin/sh\necho \"{name} $*\" >>'{}'\n", log.display())).unwrap();
            std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        }
        Tools { docker: dir.join("docker"), ssh: dir.join("ssh") }
    }

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !done() {
            assert!(Instant::now() < deadline, "{what}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn a_signal_tears_down_the_launched_entry_before_smoke_returns() {
        let root = tempfile::tempdir().unwrap();
        let (configs, logs) = (root.path().join("configs"), root.path().join("logs"));
        std::fs::create_dir_all(&configs).unwrap();
        std::fs::create_dir_all(&logs).unwrap();
        let calls = root.path().join("calls.log");
        // A launcher that never becomes ready (and records that it was stopped).
        let launcher_pid = root.path().join("launcher.pid");
        std::fs::write(root.path().join("run.sh"), format!(
            "#!/bin/sh\necho $$ >'{}'\ntrap 'echo launcher stopped >>\"{}\"; exit 143' TERM\nwhile :; do sleep 0.05; done\n",
            launcher_pid.display(), calls.display())).unwrap();
        std::fs::set_permissions(root.path().join("run.sh"), std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        std::fs::write(root.path().join("base.config"),
            "ADDR=127.0.0.1:9\nINSTANCE=cancel-test\nEXPERT_PORT=19441\nSPARK_COUNT=2\nSPARK_0_HOST=kiwi\nSPARK_1_HOST=rhea\n").unwrap();
        let entry = Entry { name: "cancel".into(), config: Some("base.config".into()), ..Entry::default() };
        let inflight = Arc::new(Inflight { tools: fake_tools(root.path(), &calls), ..Inflight::default() });
        inflight.register(&[signal_hook::consts::SIGUSR2]).unwrap();
        let worker = {
            let (entry, repo, configs, logs, inflight) = (entry.clone(), root.path().to_path_buf(), configs.clone(), logs.clone(), inflight.clone());
            std::thread::spawn(move || run_entry(&entry, &repo, &repo, &configs, &logs, "smoke", "test", false, &inflight))
        };
        wait_until("launcher registered", || launcher_pid.exists() && inflight.launches.lock().unwrap().contains_key("cancel"));
        // A real signal through the installed handler, as systemd's stop sends one
        // (SIGUSR2 stands in for SIGTERM, which would reach the whole test binary).
        signal_hook::low_level::raise(signal_hook::consts::SIGUSR2).unwrap();
        wait_until("cancel flag", || inflight.cancelled().is_some());
        let outcome = worker.join().unwrap();
        assert_eq!(outcome.status, "cancelled");
        assert!(outcome.error.as_deref().unwrap().contains("cancelled by signal"), "{:?}", outcome.error);
        // The entry left its teardown to the scheduler and is still registered for it.
        assert!(inflight.launches.lock().unwrap().contains_key("cancel"));
        assert!(!std::fs::read_to_string(&calls).unwrap_or_default().contains("rm -f"));
        let started = Instant::now();
        assert_eq!(inflight.cancel_all(), vec!["cancel".to_string()]);
        assert!(started.elapsed() < Duration::from_secs(CANCEL_TEARDOWN_S));
        let calls = std::fs::read_to_string(&calls).unwrap();
        assert!(calls.contains("launcher stopped"), "{calls}");
        assert!(calls.contains("docker rm -f cuteafd-coordinator-cancel-test"), "{calls}");
        for host in ["kiwi", "rhea"] {
            assert!(calls.lines().any(|l| l.starts_with("ssh ") && l.contains(host) && l.contains("docker rm -f") && l.contains(&format!("cuteafd-spark-expert-{host}-19441"))), "{calls}");
        }
        assert!(std::fs::read_to_string(logs.join("cancel.log")).unwrap().contains("teardown cancel (cuteafd-coordinator-cancel-test)"));
        // A cancelled registry refuses new launches and hooks.
        let mut command = Command::new("true");
        assert!(inflight.launch("late", &mut command, &configs, &logs).unwrap_err().downcast_ref::<Cancelled>().is_some());
        let hook = Hook { argv: vec!["true".into()], timeout_s: 2, required: false, cleanup: None, finish_timeout_s: None };
        assert!(HookChild::spawn(&hook, "http://127.0.0.1", &logs.join("x.log"), &logs.join("x.log"), "precheck", &inflight).is_err());
    }

    #[test]
    fn cancel_all_kills_hook_groups_and_runs_their_cleanup() {
        let root = tempfile::tempdir().unwrap();
        let log = root.path().join("hook.log");
        let cleaned = root.path().join("cleaned");
        let hook = Hook { argv: vec!["sleep".into(), "30".into()], timeout_s: 60, required: false,
            cleanup: Some(vec!["touch".into(), cleaned.display().to_string()]), finish_timeout_s: None };
        let inflight = Arc::new(Inflight::default());
        let mut child = HookChild::spawn(&hook, "http://127.0.0.1", &log, &log, "observer", &inflight).unwrap();
        inflight.cancel(signal_hook::consts::SIGTERM);
        let started = Instant::now();
        assert!(inflight.cancel_all().is_empty());
        assert!(cleaned.exists());
        assert!(child.wait().is_err(), "the hook was killed");
        assert!(started.elapsed() < Duration::from_secs(10));
        drop(child);
        assert!(inflight.hooks.lock().unwrap().is_empty());
    }

    #[test]
    fn window_merges_round_trips_and_absent_keeps_legacy_shape() {
        let hook = |cmd: &str| Hook { argv: vec![cmd.into()], timeout_s: 60, required: true, cleanup: None, finish_timeout_s: None };
        let window = Window { open: hook("open"), close: hook("close") };
        let defaults = Entry { window: Some(window.clone()), ..Entry::default() };
        let merged = Entry { name: "a".into(), ..Entry::default() }.merged(&defaults);
        assert_eq!(merged.window, Some(window.clone()));
        let value = serde_json::to_value(&merged).unwrap();
        assert_eq!(value["window"]["open"]["argv"][0], "open");
        let back: Entry = serde_json::from_value(value).unwrap();
        assert_eq!(back.window, Some(window));
        assert_ne!(back.key(), Entry { name: "a".into(), ..Entry::default() }.key());
        assert!(serde_json::to_value(Entry::default()).unwrap().get("window").is_none());
    }

    #[test]
    fn failed_prelaunch_hook_never_starts_launcher() {
        let root = tempfile::tempdir().unwrap();
        let configs = root.path().join("configs");
        let logs = root.path().join("logs");
        std::fs::create_dir_all(&configs).unwrap();
        std::fs::create_dir_all(&logs).unwrap();
        let config = root.path().join("base.config");
        std::fs::write(&config, "SPARK_COUNT=0\nINSTANCE=smoke-prelaunch-negative-test\n").unwrap();
        let e = Entry { name: "cold".into(), config: Some(config.display().to_string()),
            prelaunch: Some(Hook { argv: vec!["false".into()], timeout_s: 2, required: true,
                cleanup: None, finish_timeout_s: None }), ..Entry::default() };
        // No run.sh exists: the hook failure must precede any spawn attempt.
        let outcome = run_entry(&e, root.path(), root.path(), &configs, &logs, "smoke", "test", false, &Arc::new(Inflight::default()));
        assert_eq!(outcome.status, "failed");
        assert!(outcome.error.unwrap().contains("prelaunch hook failed before launcher"));
        assert!(outcome.launch_s.is_none());
        assert!(outcome.hooks.unwrap()["prelaunch"].contains("exited"));
        assert!(!std::fs::read_to_string(logs.join("cold.log")).unwrap().contains("teardown"));
    }

    #[test]
    fn hooks_are_bounded_and_fail_clearly() {
        let root = std::env::temp_dir().join(format!("cuteafd-smoke-hook-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let log = root.join("test.log");
        let hook = Hook { argv: vec!["false".into()], timeout_s: 2, required: false, cleanup: None, finish_timeout_s: None };
        let inflight = Arc::new(Inflight::default());
        let mut child = HookChild::spawn(&hook, "http://127.0.0.1", &log, &log, "precheck", &inflight).unwrap();
        assert!(child.wait().unwrap_err().to_string().contains("exited"));
        drop(child);
        let hook = Hook { argv: vec!["sleep".into(), "20".into()], timeout_s: 1, required: false, cleanup: None, finish_timeout_s: None };
        let mut child = HookChild::spawn(&hook, "http://127.0.0.1", &log, &log, "observer", &inflight).unwrap();
        let started = Instant::now();
        assert!(child.wait().is_err());
        drop(child);
        assert!(started.elapsed() < Duration::from_secs(3));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn observer_finish_grace_is_configurable_but_cannot_extend_total_deadline() {
        let root = std::env::temp_dir().join(format!("cuteafd-smoke-finish-hook-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let log = root.join("test.log");
        let mut hook: Hook = serde_json::from_value(serde_json::json!({
            "argv": ["sleep", "20"], "timeout_s": 30, "finish_timeout_s": 300
        })).unwrap();
        let inflight = Arc::new(Inflight::default());
        let mut child = HookChild::spawn(&hook, "http://127.0.0.1", &log, &log, "observer", &inflight).unwrap();
        assert_eq!(child.finish_timeout, Duration::from_secs(300));
        child.deadline = Instant::now() + Duration::from_millis(100);
        let total_deadline = child.deadline;
        assert!(child.finish().unwrap_err().to_string().contains("timed out"));
        assert_eq!(child.deadline, total_deadline);
        assert!(child.stop.exists());
        drop(child);
        for invalid in [0, 7201] {
            hook.finish_timeout_s = Some(invalid);
            assert!(HookChild::spawn(&hook, "http://127.0.0.1", &log, &log, "observer", &inflight).is_err());
        }
        let _ = std::fs::remove_dir_all(root);
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
        assert_eq!(locks_for(&b), vec!["gpu1.lock", "rhea.lock", "moa.lock"]);
        assert!(!locks_for(&a).iter().any(|lock| locks_for(&b).contains(lock)));
        assert!(!a.sparks.as_ref().unwrap().iter().any(|host| b.sparks.as_ref().unwrap().contains(host)));
        assert_eq!(locks_for(&entry("max", &[0, 1], &["ostrich", "rhea", "moa"], 8003)),
            vec!["sparks.lock", "gpu0.lock", "gpu1.lock", "rhea.lock", "moa.lock"]);
        assert_eq!(locks_for(&c), vec!["gpu1.lock"]);
    }

    #[test]
    fn pool_lock_narrowing_changes_only_gpu1_out_of_pool() {
        for gpus in [vec![0], vec![1], vec![0, 1]] {
            for hosts in [vec![], vec!["ostrich", "dodo", "emu", "kiwi"], vec!["rhea"], vec!["moa"], vec!["rhea", "moa"], vec!["ostrich", "rhea"]] {
                for exclusive in [false, true] {
                    let mut e = entry("test", &gpus, &hosts, 8000);
                    e.exclusive = Some(exclusive);
                    let old_pool = gpus.contains(&0) || !hosts.is_empty() || exclusive;
                    let new_pool = locks_for(&e).contains(&"sparks.lock");
                    if old_pool != new_pool {
                        assert_eq!(gpus, vec![1]);
                        assert!(!exclusive && !hosts.is_empty());
                        assert!(hosts.iter().all(|h| matches!(*h, "rhea" | "moa")));
                    }
                }
            }
        }
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
        assert_eq!(config.get("ENABLE_BENCH").map(String::as_str), Some("on"));
    }
}
