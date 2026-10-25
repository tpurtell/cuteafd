//! The benchmark runner inside the server: one run at a time on its own
//! thread, driving the server's own API over loopback. While a run is active
//! every other inference request is refused (`lockout`), progress and partial
//! results stream to `/v1/bench/events`, and the report lands in SQLite.
use crate::client::Client;
use crate::panels::{self, Ctx, Progress, Rates};
use crate::profiles::{self, Profile};
use crate::report::{
    now_rfc3339, Baseline, PanelResult, PanelStatus, PlannedPanel, Report, RunStatus, ServerInfo, SCHEMA,
};
use crate::store::{self, Store};
use cuteafd_api::openai::ConsoleHub;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

/// What a client asks to run.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RunRequest {
    /// A built-in or saved profile name.
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub fabric: Option<bool>,
    /// Explicit panels (a custom run); overrides the profile's list.
    #[serde(default)]
    pub panels: Option<Vec<String>>,
    /// `panel -> passes` overrides.
    #[serde(default)]
    pub passes: HashMap<String, u32>,
    /// `false`: run explicit `panels` without the baseline (basic card + quick
    /// quality) when none is cached; panels then price themselves from default
    /// rates. Ignored for profiles; default true.
    #[serde(default)]
    pub baseline: Option<bool>,
    /// Cases for `spec_lossless_set` (default: the shipped set).
    #[serde(default)]
    pub lossless_set: Option<panels::lossless::SetRequest>,
    /// Opt-in row dumps for selected `spec_lossless_set` cases.
    #[serde(default)]
    pub lossless_dump: Option<panels::lossless::RowDump>,
    #[serde(default)]
    pub context_lengths: Option<Vec<u64>>,
    #[serde(default)]
    pub context_levels: Option<Vec<usize>>,
    #[serde(default)]
    pub context_suffix: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum StartError {
    #[error("a benchmark is already running ({0})")]
    Busy(String),
    #[error("unknown profile {0}")]
    UnknownProfile(String),
    #[error("the server is not ready")]
    NotReady,
    #[error("the server is stopping ({0})")]
    Stopping(String),
    #[error("invalid lossless set: {0}")]
    InvalidSet(String),
    #[error("invalid context-load options: {0}")]
    InvalidContext(String),
}

/// What the progress ticker knows about the panel now running: enough to price
/// the whole run (`remaining`/`total`) and this panel on its own.
#[derive(Debug, Clone)]
struct Plan {
    run: String,
    panel: String,
    /// Estimated seconds for one pass of this panel.
    estimate: f64,
    /// Whole-run seconds still to run.
    remaining: f64,
    /// Whole-run seconds at the start.
    total: f64,
    /// The pass now running (1-based) and how many this panel runs.
    pass: u32,
    passes: u32,
    /// When the running pass started.
    started: Instant,
}

impl Default for Plan {
    fn default() -> Self {
        Self { run: String::new(), panel: String::new(), estimate: 0.0, remaining: 0.0, total: 1.0,
            pass: 1, passes: 1, started: Instant::now() }
    }
}

/// Seconds left in the running panel: the pass's measured pace once it has
/// enough data to extrapolate from, else its estimate; later passes at the
/// estimate.
fn panel_eta_s(estimate: f64, pass: u32, passes: u32, fraction: f64, elapsed: f64) -> f64 {
    let pass_left = if fraction > 0.1 && elapsed > 0.0 {
        elapsed * (1.0 - fraction) / fraction
    } else {
        estimate * (1.0 - fraction)
    };
    (pass_left + estimate * f64::from(passes.saturating_sub(pass))).max(0.0)
}

/// The run holding the lock.
#[derive(Clone)]
pub struct ActiveRun {
    pub id: String,
    pub token: String,
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
    pub eta_s: f64,
    pub panel: String,
    pub fraction: f64,
}

impl std::fmt::Debug for ActiveRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveRun").field("id", &self.id).field("token", &"<redacted>").finish_non_exhaustive()
    }
}

/// Allows console token text while a run holds the server; clearing on drop
/// covers every exit path, cancellation and panics included.
struct BenchText(Arc<ConsoleHub>);

impl BenchText {
    /// `None` when no console is attached (a bench without the API server).
    fn enable(bench: &Bench) -> Option<Self> {
        let console = bench.console.get()?.clone();
        console.set_bench_active(true);
        Some(Self(console))
    }
}

impl Drop for BenchText {
    fn drop(&mut self) {
        self.0.set_bench_active(false);
    }
}

pub struct Bench {
    store: Mutex<Store>,
    active: Mutex<Option<ActiveRun>>,
    live: Mutex<HashMap<String, Arc<Mutex<Report>>>>,
    baselines: Mutex<HashMap<String, Baseline>>,
    info: Mutex<Option<ServerInfo>>,
    events: broadcast::Sender<Arc<str>>,
    /// The live console of the in-process server, attached at mount. A run
    /// allows token text on it: the lockout makes the run's own prompts the
    /// only requests in.
    console: OnceLock<Arc<ConsoleHub>>,
    /// Benchmark controls require the configured API key or the console cookie.
    pub api_key: OnceLock<cuteafd_api::openai::auth::ApiKey>,
    pub(crate) console_gate: OnceLock<cuteafd_api::console_gate::ConsoleGate>,
    /// Whether benchmark requests are kept in the usage history (attached at mount).
    pub(crate) usage: OnceLock<UsageToggle>,
    /// Why the server is stopping, once it is (see `crate::stopping`).
    stopping: OnceLock<String>,
}

/// The usage store's `record_bench` setting, as the bench page toggles it.
#[derive(Clone)]
pub struct UsageToggle {
    pub get: Arc<dyn Fn() -> bool + Send + Sync>,
    pub set: Arc<dyn Fn(bool) -> Result<(), String> + Send + Sync>,
}

impl Bench {
    pub fn new(store: Store) -> Arc<Self> {
        let (events, _) = broadcast::channel(512);
        Arc::new(Self {
            store: Mutex::new(store),
            active: Mutex::new(None),
            live: Mutex::new(HashMap::new()),
            baselines: Mutex::new(HashMap::new()),
            info: Mutex::new(None),
            events,
            console: OnceLock::new(),
            api_key: OnceLock::new(),
            console_gate: OnceLock::new(),
            usage: OnceLock::new(),
            stopping: OnceLock::new(),
        })
    }

    pub fn set_api_key(&self, key: cuteafd_api::openai::auth::ApiKey) {
        let _ = self.api_key.set(key);
    }

    /// The console unlock cookie also authorizes the bench page's controls.
    pub fn set_console_gate(&self, gate: cuteafd_api::console_gate::ConsoleGate) {
        let _ = self.console_gate.set(gate);
    }

    pub fn accepts_internal(&self, token: &str) -> bool {
        self.active().is_some_and(|active|
            cuteafd_api::openai::auth::constant_time_eq(active.token.as_bytes(), token.as_bytes()))
    }

    /// The usage history's benchmark switch; called once when the bench is mounted.
    pub fn set_usage(&self, toggle: UsageToggle) {
        let _ = self.usage.set(toggle);
    }

    /// The server's live console; called once when the bench is mounted.
    pub fn set_console(&self, console: Arc<ConsoleHub>) {
        let _ = self.console.set(console);
    }

    /// Learns the served model and this server's configuration once, off the
    /// request path, so the page's fingerprint, its accumulated view and panel
    /// availability are right before any run. Serving calls it when the bench
    /// is mounted; it waits for the API to answer (bounded) and is a no-op
    /// once a run has learned the same.
    pub fn discover(&self) {
        let mut last = None;
        for attempt in 0..60 {
            if self.info.lock().is_ok_and(|i| i.is_some()) { return; }
            if let Some(base) = crate::context::loopback() {
                let mut client = Client::new(&base, None, Arc::new(AtomicBool::new(false))).with_api_key(self.api_key.get().cloned());
                match client.discover() {
                    Ok(_) => { self.server_info(&client.model); return; }
                    Err(error) => last = Some(error),
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(if attempt < 10 { 200 } else { 1000 }));
        }
        if let Some(error) = last { tracing::debug!(%error, "bench: server discovery deferred to the first run"); }
    }

    /// The process-wide instance (SQLite under `store::default_dir()`, in memory if unwritable).
    pub fn global() -> Arc<Self> {
        static BENCH: OnceLock<Arc<Bench>> = OnceLock::new();
        BENCH.get_or_init(|| {
            let dir = store::default_dir();
            let store = Store::open(&dir).unwrap_or_else(|error| {
                tracing::warn!(%error, dir = %dir.display(), "benchmark history is not persisted");
                Store::memory().expect("in-memory SQLite")
            });
            // Re-derive summaries stored under an older verdict rule on a second
            // connection, after start: readiness never waits on the history.
            if store.stale().unwrap_or(false) {
                let _ = std::thread::Builder::new().name("bench-backfill".into()).spawn(move || {
                    if let Err(error) = Store::open(&dir).and_then(|s| s.backfill()) {
                        tracing::warn!(%error, "bench store: summary backfill failed");
                    }
                });
            }
            Self::new(store)
        }).clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Arc<str>> {
        self.events.subscribe()
    }

    fn emit(&self, event: Value) {
        let _ = self.events.send(Arc::from(event.to_string()));
    }

    pub fn active(&self) -> Option<ActiveRun> {
        self.active.lock().ok()?.clone()
    }

    /// For the lockout: Some(retry-after seconds) when a run holds the server
    /// and `token` is not its token.
    pub fn locked(&self, token: Option<&str>) -> Option<u64> {
        let active = self.active()?;
        if token.is_some_and(|token| cuteafd_api::openai::auth::constant_time_eq(active.token.as_bytes(), token.as_bytes())) {
            return None;
        }
        Some((active.eta_s.ceil() as u64).clamp(5, 3600))
    }

    pub fn store<T>(&self, f: impl FnOnce(&Store) -> anyhow::Result<T>) -> anyhow::Result<T> {
        let store = self.store.lock().map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
        f(&store)
    }

    pub fn profiles(&self) -> Vec<Profile> {
        let mut all = profiles::builtin();
        all.extend(self.store(|s| s.profiles()).unwrap_or_default());
        all
    }

    /// A live or stored report.
    pub fn report(&self, id: &str) -> Option<Report> {
        if let Some(live) = self.live.lock().ok()?.get(id) {
            return live.lock().ok().map(|r| r.clone());
        }
        self.store(|s| s.load(id)).ok().flatten()
    }

    /// One panel of a live or stored run; a stored one is extracted on its own.
    pub fn panel(&self, id: &str, panel: &str) -> Option<crate::report::PanelResult> {
        if let Some(live) = self.live.lock().ok()?.get(id) {
            return live.lock().ok()?.panel(panel).cloned();
        }
        self.store(|s| s.panel(id, panel)).ok().flatten()
    }

    /// Every panel's newest result on `fingerprint`, each tagged with the run
    /// it came from, under the newest baseline; the active run's live panels
    /// replace their own sections. Reads only the runs that contribute.
    pub fn merged(&self, fingerprint: &str) -> Option<(Report, Option<(String, String)>)> {
        let sources = self.store(|s| s.sources(fingerprint)).ok()?;
        let active = self.active().filter(|a| self.report(&a.id).is_some_and(|r| r.fingerprint == fingerprint));
        let live = active.as_ref().and_then(|a| self.report(&a.id));
        let newest = live.clone().or_else(|| sources.newest.as_deref().and_then(|id| self.report(id)))?;
        let mut cache: HashMap<String, Report> = HashMap::new();
        let mut fetch = |id: &str| -> Option<Report> {
            if !cache.contains_key(id) { cache.insert(id.to_string(), self.report(id)?); }
            cache.get(id).cloned()
        };
        let mut merged = newest.clone();
        // The merged view keeps the newest run's id: server, hardware,
        // configuration and fabric render from it; panels carry their own run.
        merged.panels = Vec::new();
        // The live run's own panels first (running, done or queued), then every
        // other panel's newest stored result.
        if let Some(live) = &live {
            for panel in &live.panels {
                if panel.status == crate::report::PanelStatus::Pending && panel.passes.is_empty()
                    && sources.panels.iter().any(|(id, _)| id == &panel.id) { continue; }
                let mut panel = panel.clone();
                panel.run = Some(live.id.clone()); panel.run_created = Some(live.created.clone());
                merged.panels.push(panel);
            }
        }
        for (panel_id, run_id) in &sources.panels {
            if merged.panels.iter().any(|p| &p.id == panel_id) { continue; }
            let Some(report) = fetch(run_id) else { continue };
            if let Some(panel) = report.panel(panel_id) {
                let mut panel = panel.clone();
                panel.run = Some(report.id.clone()); panel.run_created = Some(report.created.clone());
                merged.panels.push(panel);
            }
        }
        let baseline_run = live.as_ref().filter(|l| l.baseline.is_some()).map(|l| l.id.clone()).or(sources.baseline.clone());
        let baseline = baseline_run.as_deref().and_then(&mut fetch);
        merged.baseline = baseline.as_ref().and_then(|r| r.baseline.clone());
        merged.plan = Vec::new();
        Some((merged, baseline.map(|r| (r.id, r.created))))
    }

    /// The newest report (live first).
    pub fn latest(&self) -> Option<Report> {
        if let Some(active) = self.active() {
            return self.report(&active.id);
        }
        let id = self.store(|s| s.list(1)).ok()?.into_iter().next()?.id;
        self.report(&id)
    }

    /// Stores an imported `report.json`; returns its id.
    pub fn import(&self, mut report: Report) -> anyhow::Result<String> {
        anyhow::ensure!(report.schema.starts_with("cuteafd.bench.report/"), "not a cuteafd bench report");
        if self.report(&report.id).is_some() {
            report.id = format!("{}-import-{}", report.id, &uuid::Uuid::new_v4().simple().to_string()[..6]);
        }
        report.status = RunStatus::Imported;
        self.store(|s| s.save(&report))?;
        Ok(report.id)
    }

    fn server_info(&self, model: &str) -> ServerInfo {
        let mut slot = self.info.lock().expect("info lock");
        if let Some(info) = slot.as_ref().filter(|i| i.model == model) {
            return info.clone();
        }
        let info = crate::server::server_info(model);
        *slot = Some(info.clone());
        info
    }

    /// Rates for estimates: this lifetime's baseline, else a stored one, else defaults.
    fn rates(&self, fingerprint: Option<&str>) -> Rates {
        let Some(fingerprint) = fingerprint else { return Rates::default() };
        if let Some(b) = self.baselines.lock().ok().and_then(|b| b.get(fingerprint).cloned()) {
            return Rates::from_baseline(&b);
        }
        self.store(|s| s.latest_baseline(fingerprint)).ok().flatten()
            .map(|b| Rates::from_baseline(&b)).unwrap_or_default()
    }

    /// The panel catalog with per-pass estimates on this server.
    pub fn catalog(&self) -> Value {
        let info = self.info.lock().ok().and_then(|i| i.clone()).unwrap_or_default();
        let fingerprint = (!info.model.is_empty()).then(|| crate::server::fingerprint(&info));
        let rates = self.rates(fingerprint.as_deref());
        let has_baseline = fingerprint.as_ref().is_some_and(|f| self.baselines.lock().is_ok_and(|b| b.contains_key(f)));
        let mut panels = vec![json!({"id": "baseline", "title": "Basic card + quick quality", "mandatory": true,
            "description": "C1 decode on code, prose and JSON (thinking off), C8 code aggregate, 8K prefill and TTFT; logit fidelity, \
                prefix-cache restore exactness, lossless speculation, template round trip, C1 vs C4.",
            "estimate_s": if has_baseline { 0.0 } else { crate::baseline::estimate_s(&rates, &info) },
            "done": has_baseline})];
        for panel in panels::catalog() {
            let (span, min_width) = crate::render::layout::span(panel.id());
            panels.push(json!({"id": panel.id(), "title": panel.title(), "description": panel.description(),
                "span": span, "min_width": min_width,
                "estimate_s": panel.estimate_s(&rates, &info), "unavailable": panel.unavailable(&info),
                "always": panels::ALWAYS.contains(&panel.id())}));
        }
        json!({"panels": panels, "rates": rates})
    }

    pub fn status(&self) -> Value {
        let active = self.active();
        let info = self.info.lock().ok().and_then(|i| i.clone());
        let fingerprint = info.as_ref().map(crate::server::fingerprint);
        let baseline = fingerprint.as_ref().and_then(|f| self.baselines.lock().ok()?.get(f).cloned());
        json!({
            "active": active.as_ref().map(|a| json!({"id": a.id, "panel": a.panel, "fraction": a.fraction,
                "eta_s": a.eta_s, "elapsed_s": a.started.elapsed().as_secs_f64()})),
            "readiness_s": crate::context::readiness_s(),
            "full_prefill_logits": crate::fidelity_dataset::prefill_admitted(&crate::context::get().settings),
            "snapshot": crate::context::get().snapshot,
            "settings": crate::context::get().settings,
            "fingerprint": fingerprint,
            "model": info.as_ref().map(|i| i.model.clone()),
            "checkpoint": crate::context::get().snapshot.as_ref().and_then(|p| crate::report::hub_repo(&p.to_string_lossy()))
                .or_else(|| info.as_ref().map(|i| i.checkpoint())),
            "baseline": baseline.as_ref().map(|b| json!({"run": b.run_id, "quality": b.quality.status,
                "badge": b.quality.badge()})),
            "quality_failed": baseline.as_ref().is_some_and(|b| b.quality.status == crate::report::CheckStatus::Fail),
            "auth": "console unlock or API key",
        })
    }

    /// The server is shutting down: refuse new runs and stop the active one,
    /// which then records `reason` instead of the connection errors that follow.
    pub fn stopping(&self, reason: &str) {
        let _ = self.stopping.set(reason.to_string());
        if let Some(active) = self.active() {
            tracing::warn!(run = %active.id, reason, "server stopping: interrupting the running benchmark");
            active.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn interrupted(&self) -> Option<String> {
        self.stopping.get().map(|reason| format!("interrupted: the server stopped ({reason}) during this run"))
    }

    pub fn cancel(&self, id: &str) -> bool {
        match self.active() {
            Some(active) if active.id == id => {
                active.cancel.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    /// One remote probe is registered inside the serving process, where the registry lives.
    /// It holds the same exclusive inference slot as a benchmark, including on errors.
    pub fn probe(self: &Arc<Self>, body: Value, mut spec: cuteafd_api::openai::probe::ProbeSpec) -> anyhow::Result<Value> {
        anyhow::ensure!(body.is_object(), "probe body must be a chat object");
        if spec.dump_rows.is_some() {
            let root = std::env::var_os("CUTEAFD_PROBE_DUMP_ROOT")
                .ok_or_else(|| anyhow::anyhow!("remote row dumps require CUTEAFD_PROBE_DUMP_ROOT"))?;
            spec.constrain_dump_root(std::path::Path::new(&root))?;
        }
        let base = crate::context::loopback().ok_or(StartError::NotReady)?;
        let active = ActiveRun { id: uuid::Uuid::new_v4().simple().to_string(),
            token: uuid::Uuid::new_v4().simple().to_string(), cancel: Arc::new(AtomicBool::new(false)),
            started: Instant::now(), eta_s: 900.0, panel: "fidelity probe".into(), fraction: 0.0 };
        {
            let mut slot = self.active.lock().map_err(|_| anyhow::anyhow!("active lock poisoned"))?;
            if let Some(current) = slot.as_ref() { return Err(StartError::Busy(current.id.clone()).into()); }
            *slot = Some(active.clone());
        }
        struct Release(Arc<Bench>, String);
        impl Drop for Release {
            fn drop(&mut self) {
                if let Ok(mut slot) = self.0.active.lock() {
                    if slot.as_ref().is_some_and(|a| a.id == self.1) { *slot = None; }
                }
            }
        }
        let _release = Release(self.clone(), active.id.clone());
        let mut client = Client::new(&base, Some(active.token), active.cancel).with_api_key(self.api_key.get().cloned());
        client.discover()?;
        let model = client.model.clone();
        let chat = client.chat(body, Some(spec))?;
        let mut value = serde_json::to_value(chat)?;
        let context = crate::context::get();
        value["server"] = json!({"model": model, "family": context.family,
            "snapshot": context.snapshot, "settings": context.settings, "build": crate::server::build_info()});
        Ok(value)
    }

    /// Starts a run; returns its id.
    pub fn start(self: &Arc<Self>, request: RunRequest) -> Result<String, StartError> {
        if let Some(reason) = self.stopping.get() { return Err(StartError::Stopping(reason.clone())); }
        let base = crate::context::loopback().ok_or(StartError::NotReady)?;
        let lossless_set = request.lossless_set.clone().map(|set| set.validated()).transpose()
            .map_err(StartError::InvalidSet)?;
        if let Some(dump) = &request.lossless_dump {
            dump.validate(lossless_set.as_ref().unwrap_or(&panels::lossless::CaseSet::default_set()))
                .and_then(|()| panels::lossless::RowDump::root().map(drop)).map_err(StartError::InvalidSet)?;
        }
        let lossless_dump = request.lossless_dump.clone();
        let defaults = panels::context_load::Options::default();
        let context_load = panels::context_load::Options {
            lengths: request.context_lengths.clone().unwrap_or(defaults.lengths),
            levels: request.context_levels.clone().unwrap_or(defaults.levels),
            suffix_tokens: request.context_suffix.unwrap_or(defaults.suffix_tokens),
        };
        context_load.validate().map_err(StartError::InvalidContext)?;
        let (name, planned) = match (&request.panels, &request.profile) {
            (Some(panels), _) => ("custom".to_string(),
                panels.iter().map(|id| PlannedPanel { id: id.clone(), passes: 1 }).collect::<Vec<_>>()),
            (None, Some(name)) => {
                let profile = self.profiles().into_iter().find(|p| &p.name == name)
                    .ok_or_else(|| StartError::UnknownProfile(name.clone()))?;
                (profile.name, profile.panels)
            }
            (None, None) => ("share".to_string(), Vec::new()),
        };
        let fabric = request.fabric.unwrap_or_else(|| request.panels.is_none()
            && self.profiles().iter().any(|p| p.name == name && p.fabric));
        let passes: Vec<(String, u32)> = request.passes.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let (plan, dropped) = profiles::resolve(&planned, &passes);
        let skip_baseline = request.panels.is_some() && request.baseline == Some(false);
        let id = uuid::Uuid::new_v4().simple().to_string();
        let active = ActiveRun { id: id.clone(), token: uuid::Uuid::new_v4().simple().to_string(),
            cancel: Arc::new(AtomicBool::new(false)), started: Instant::now(), eta_s: 0.0, panel: "starting".into(),
            fraction: 0.0 };
        {
            let mut slot = self.active.lock().expect("active lock");
            if let Some(current) = slot.as_ref() {
                return Err(StartError::Busy(current.id.clone()));
            }
            *slot = Some(active.clone());
        }
        let report = Arc::new(Mutex::new(Report {
            schema: SCHEMA.into(), prompt_content: Some(crate::text::PROMPT_CONTENT.into()), id: id.clone(), created: now_rfc3339(), finished: None, status: RunStatus::Running,
            profile: name, plan: plan.clone(), server: ServerInfo::default(), fingerprint: String::new(), baseline: None,
            panels: plan.iter().map(|p| PanelResult { id: p.id.clone(),
                title: panels::find(&p.id).map_or(p.id.clone(), |panel| panel.title().to_string()),
                ..PanelResult::default() }).collect(),
            fabric: Vec::new(),
            error: (!dropped.is_empty()).then(|| format!("panels not in this build: {}", dropped.join(", "))),
        }));
        // Publish the starting run before its id can reach a polling client.
        self.live.lock().expect("live lock").insert(id.clone(), report.clone());
        let bench = self.clone();
        // The run is active from here (the lockout is already refusing other
        // clients), so its console text is allowed until it retires.
        let text = BenchText::enable(self);
        std::thread::Builder::new().name("cuteafd-bench".into()).spawn(move || {
            bench.execute(active, base, plan, report, text, skip_baseline, lossless_set, lossless_dump, context_load, fabric);
        }).expect("spawn the benchmark thread");
        Ok(id)
    }

    fn update_active(&self, f: impl FnOnce(&mut ActiveRun)) {
        if let Ok(mut slot) = self.active.lock() {
            if let Some(active) = slot.as_mut() {
                f(active);
            }
        }
    }

    fn execute(self: Arc<Self>, active: ActiveRun, base: String, plan: Vec<PlannedPanel>,
        report: Arc<Mutex<Report>>, text: Option<BenchText>, skip_baseline: bool,
        lossless_set: Option<panels::lossless::CaseSet>, lossless_dump: Option<panels::lossless::RowDump>,
        context_load: panels::context_load::Options, fabric: bool) {
        let id = active.id.clone();
        let progress = Progress::default();
        let done = Arc::new(AtomicBool::new(false));
        let ticker = {
            let (bench, progress, done, report, base) = (self.clone(), progress.clone(), done.clone(), report.clone(),
                base.clone());
            std::thread::Builder::new().name("cuteafd-bench-tick".into())
                .spawn(move || bench.tick(&report, &progress, &done, &base, fabric)).ok()
        };
        let outcome = self.run_plan(&active, &base, &report, &progress, &plan, skip_baseline, lossless_set.as_ref(),
            lossless_dump.as_ref(), &context_load, fabric);
        {
            let mut r = report.lock().expect("report lock");
            r.finished = Some(now_rfc3339());
            let interrupted = self.interrupted();
            match outcome {
                Ok(()) if interrupted.is_none() => r.status = RunStatus::Done,
                _ if interrupted.is_some() => {
                    r.status = RunStatus::Cancelled;
                    r.error = interrupted.clone();
                }
                Ok(()) => r.status = RunStatus::Done,
                Err(error) if active.cancel.load(Ordering::Relaxed)
                    || error.downcast_ref::<crate::client::Cancelled>().is_some() => r.status = RunStatus::Cancelled,
                Err(error) => {
                    r.status = RunStatus::Failed;
                    r.error = Some(format!("{error:#}"));
                }
            }
            for panel in r.panels.iter_mut().filter(|p| matches!(p.status, PanelStatus::Pending | PanelStatus::Running)) {
                if panel.status == PanelStatus::Running && interrupted.is_some() { panel.error = interrupted.clone(); }
                panel.status = PanelStatus::Cancelled;
            }
            if let Err(error) = self.store(|s| s.save(&r)) {
                tracing::warn!(%error, "benchmark report not stored");
            }
            tracing::info!(run = %r.id, status = ?r.status, "benchmark finished");
        }
        done.store(true, Ordering::Relaxed);
        if let Some(ticker) = ticker {
            let _ = ticker.join();
        }
        if let Ok(mut plan) = self.plan_state().lock() {
            *plan = Plan::default();
        }
        // Console text ends before the lock lifts: a run admitted in the gap
        // would otherwise have its own override cleared by this run's guard.
        drop(text);
        *self.active.lock().expect("active lock") = None;
        let snapshot = report.lock().expect("report lock").clone();
        self.emit(json!({"type": "report", "run": id, "report": snapshot}));
        self.emit(json!({"type": "status", "status": self.status()}));
        // Finished runs are served from the store from now on.
        self.live.lock().expect("live lock").remove(&id);
    }

    fn run_plan(&self, active: &ActiveRun, base: &str, report: &Arc<Mutex<Report>>, progress: &Progress,
        plan: &[PlannedPanel], skip_baseline: bool, lossless_set: Option<&panels::lossless::CaseSet>,
        lossless_dump: Option<&panels::lossless::RowDump>, context_load: &panels::context_load::Options, fabric: bool)
        -> anyhow::Result<()> {
        let mut client = Client::new(base, Some(active.token.clone()), active.cancel.clone()).with_api_key(self.api_key.get().cloned()).with_fabric(fabric);
        let record = client.discover()?;
        let max_context = record["max_context_tokens"].as_u64().unwrap_or(8192);
        let max_output = record["max_output_tokens"].as_u64().unwrap_or(4096);
        let info = self.server_info(&client.model);
        let fingerprint = crate::server::fingerprint(&info);
        let run_mark = client.thermal_mark();
        {
            let mut r = report.lock().expect("report lock");
            r.server = info.clone();
            r.fingerprint = fingerprint.clone();
        }
        self.emit(json!({"type": "status", "status": self.status()}));
        let baseline = self.baselines.lock().ok().and_then(|b| b.get(&fingerprint).cloned());
        let mut rates = self.rates(Some(&fingerprint));
        // Estimates for the ETA: (panel, seconds per pass, passes).
        let estimates: Vec<(String, f64, u32)> = plan.iter().filter_map(|p| panels::find(&p.id)
            .map(|panel| (p.id.clone(), panel.estimate_s(&rates, &info), p.passes))).collect();
        let mut remaining: f64 = estimates.iter().map(|(_, s, n)| s * f64::from(*n)).sum::<f64>()
            + if baseline.is_none() && !skip_baseline { crate::baseline::estimate_s(&rates, &info) } else { 0.0 };
        let total = remaining.max(1.0);
        let baseline = match baseline {
            Some(mut b) => { b.fabric.clear(); Some(b) },
            // Explicit panels alone: no basic card, no quick quality.
            None if skip_baseline => None,
            None => {
                let estimate = crate::baseline::estimate_s(&rates, &info);
                self.begin(active, "baseline", estimate, remaining, total, 1, 1, progress);
                let thermal_mark = client.thermal_mark();
                let mut b = crate::baseline::run(&client, &info, progress, &active.id, &fingerprint, max_context,
                    max_output)?;
                b.thermal = client.thermal_since(thermal_mark);
                remaining -= estimate;
                if b.quality.status != crate::report::CheckStatus::Pending {
                    self.baselines.lock().expect("baselines lock").insert(fingerprint.clone(), b.clone());
                }
                rates = Rates::from_baseline(&b);
                Some(b)
            }
        };
        {
            let mut r = report.lock().expect("report lock");
            r.fabric.extend(baseline.iter().flat_map(|b| b.fabric.iter().cloned()));
            r.baseline = baseline.clone();
            progress.finish_baseline();
        }
        Self::record_power(&client, run_mark, report);
        self.emit(json!({"type": "report", "run": active.id, "report": report.lock().expect("report lock").clone()}));
        for (index, planned) in plan.iter().enumerate() {
            let Some(panel) = panels::find(&planned.id) else { continue };
            let estimate = panel.estimate_s(&rates, &info);
            if let Some(reason) = panel.unavailable(&info) {
                let mut r = report.lock().expect("report lock");
                r.panels[index].status = PanelStatus::Unsupported;
                r.panels[index].error = Some(reason);
                remaining -= estimate * f64::from(planned.passes);
                continue;
            }
            // This panel's earlier passes on the server, newest run first;
            // SQLite extracts the one panel, never whole reports.
            let earlier: Vec<Value> = self.store(|s| s.panel_history(&fingerprint, panel.id(), 50)).unwrap_or_default()
                .into_iter().flat_map(|p| p.passes).collect();
            {
                let mut r = report.lock().expect("report lock");
                r.panels[index].status = PanelStatus::Running;
                r.panels[index].started = Some(now_rfc3339());
                r.panels[index].history = earlier.clone();
            }
            let started = Instant::now();
            for pass in 1..=planned.passes {
                client.check()?;
                self.begin(active, panel.id(), estimate, remaining, total, pass, planned.passes, progress);
                let ctx = Ctx { client: &client, info: &info, baseline: baseline.as_ref(), rates, progress, pass,
                    history: &earlier, max_context, max_output, lossless_set, lossless_dump, context_load };
                let table_before = panels::common::mapped_counters(&client);
                let thermal_mark = client.thermal_mark();
                let fabric_mark = client.fabric_mark();
                let outcome = panel.run(&ctx);
                let fabric = client.fabric_since(fabric_mark, &format!("{}/pass{pass}", panel.id()), false);
                let thermal = client.thermal_since(thermal_mark);
                let table_after = panels::common::mapped_counters(&client);
                remaining -= estimate;
                let mut r = report.lock().expect("report lock");
                match outcome {
                    Ok(mut value) => {
                        let tables = panels::common::mapped_interval(&table_before, &table_after);
                        if !tables.is_empty() {
                            value["mapped_tables"] = json!(tables);
                            value["mapped_tables_scope"] = json!("whole panel pass, including panel-internal priming");
                        }
                        if let Some(phase) = fabric {
                            value["fabric"] = json!(phase);
                            r.fabric.push(phase);
                        }
                        fn collect_fabric(value: &Value, phases: &mut Vec<crate::fabric::Phase>) {
                            match value {
                                Value::Object(fields) => for (key, value) in fields {
                                    if key == "fabric" || key.ends_with("_fabric") {
                                        if let Ok(phase) = serde_json::from_value::<crate::fabric::Phase>(value.clone()) {
                                            if phase.timed { phases.push(phase); }
                                        }
                                    } else { collect_fabric(value, phases); }
                                },
                                Value::Array(values) => for value in values { collect_fabric(value, phases); },
                                _ => {}
                            }
                        }
                        collect_fabric(&value, &mut r.fabric);
                        // GPU throttle report over the whole pass (panel-internal warm-ups included).
                        if let (Some(thermal), true) = (thermal, value.is_object()) { value["thermal"] = thermal; }
                        r.panels[index].passes.push(value);
                        r.panels[index].partial = None;
                    }
                    Err(error) => {
                        if error.downcast_ref::<crate::client::Cancelled>().is_some() || self.stopping.get().is_some() {
                            if let Some(reason) = self.interrupted() { r.panels[index].error = Some(reason); }
                            r.panels[index].status = PanelStatus::Cancelled;
                            return Err(error);
                        }
                        r.panels[index].status = PanelStatus::Failed;
                        r.panels[index].error = Some(format!("{error:#}"));
                        break;
                    }
                }
                drop(r);
                Self::record_power(&client, run_mark, report);
                self.emit(json!({"type": "report", "run": active.id, "report": report.lock().expect("report lock").clone()}));
            }
            let mut r = report.lock().expect("report lock");
            if r.panels[index].status == PanelStatus::Running {
                r.panels[index].status = PanelStatus::Done;
            }
            r.panels[index].finished = Some(now_rfc3339());
            r.panels[index].seconds = started.elapsed().as_secs_f64();
            if let Err(error) = self.store(|s| s.save(&r)) {
                tracing::warn!(%error, "benchmark report not stored");
            }
        }
        Ok(())
    }

    /// Record the GPU power limits the run actually got (whole-run throttle window, so the
    /// label reflects what gpu-powerd or the static cap delivered, not a one-time read).
    fn record_power(client: &Client, mark: Option<f64>, report: &Arc<Mutex<Report>>) {
        let Some(window) = client.thermal_since(mark) else { return };
        if let Some(power) = crate::report::Power::from_json(&[&window]) {
            report.lock().expect("report lock").server.hardware.power = Some(power);
        }
    }

    /// A panel (or the baseline) starts a pass: reset its progress and the ETA base.
    fn begin(&self, active: &ActiveRun, panel: &str, estimate: f64, remaining: f64, total: f64,
        pass: u32, passes: u32, progress: &Progress) {
        progress.reset();
        if let Ok(mut plan) = self.plan_state().lock() {
            *plan = Plan { run: active.id.clone(), panel: panel.to_string(), estimate, remaining, total,
                pass, passes, started: Instant::now() };
        }
        self.update_active(|a| a.panel = panel.to_string());
    }

    fn plan_state(&self) -> &'static Mutex<Plan> {
        static STATE: OnceLock<Mutex<Plan>> = OnceLock::new();
        STATE.get_or_init(|| Mutex::new(Plan::default()))
    }

    /// Fold a pass's partial into the shared report. A baseline partial never replaces the
    /// final baseline: the ticker can deliver one after `run_plan` installed the result.
    /// Call with the report lock held, so the check and the write are atomic against the install.
    fn apply_partial(r: &mut Report, progress: &Progress, panel: &str, partial: &Value) {
        if panel == "baseline" {
            if progress.baseline_final() {
                return;
            }
            if let Ok(b) = serde_json::from_value::<Baseline>(partial.clone()) {
                r.baseline = Some(b);
            }
        } else if let Some(p) = r.panels.iter_mut().find(|p| p.id == panel) {
            p.partial = Some(partial.clone());
        }
    }

    /// Once a second while a run is active: progress, ETA, the live strip, partial results.
    fn tick(&self, report: &Arc<Mutex<Report>>, progress: &Progress, done: &AtomicBool, base: &str, fabric: bool) {
        let stats_client = Client::new(base, None, Arc::new(AtomicBool::new(false))).with_api_key(self.api_key.get().cloned()).with_fabric(fabric);
        let mut last: Option<(Instant, f64)> = None;
        let mut revision = u64::MAX;
        let mut fabric_before = None;
        cuteafd_transport::traffic::background_priority();
        while !done.load(Ordering::Relaxed) {
            let state = progress.get();
            let plan = self.plan_state().lock().map(|s| s.clone()).unwrap_or_default();
            let eta = (plan.remaining - plan.estimate * state.fraction).max(0.0);
            let fraction = (1.0 - eta / plan.total.max(1.0)).clamp(0.0, 1.0);
            self.update_active(|a| {
                a.eta_s = eta;
                a.fraction = fraction;
            });
            // Output tokens per second and active requests from the server's own counters.
            let (mut tok_s, mut active_requests) = (None, None);
            if let Ok(stats) = stats_client.stats() {
                let generated = stats["generated_tokens"].as_f64().or_else(|| stats["totals"]["output_tokens"].as_f64());
                if let Some(generated) = generated {
                    let now = Instant::now();
                    if let Some((then, before)) = last {
                        let dt = now.duration_since(then).as_secs_f64();
                        if dt > 0.0 {
                            tok_s = Some(((generated - before) / dt).max(0.0));
                        }
                    }
                    last = Some((now, generated));
                }
                active_requests = stats["active"].as_u64().or_else(|| {
                    let t = &stats["totals"];
                    Some(t["requests_admitted"].as_u64()?.saturating_sub(t["requests_retired"].as_u64()?))
                });
            }
            if fabric {
                if let Some(after) = stats_client.fabric() {
                    if let Some(before) = fabric_before.as_ref() {
                        let interval = crate::fabric::interval(&plan.panel, false, before, &after);
                        self.emit(json!({"type": "fabric", "run": plan.run, "interval": interval}));
                    }
                    fabric_before = Some(after);
                } else {
                    self.emit(json!({"type": "fabric", "run": plan.run, "interval": {"ports": []}}));
                    fabric_before = None;
                }
            }
            let title = if plan.panel == "baseline" { "Basic card + quick quality".to_string() }
                else { panels::find(&plan.panel).map_or(plan.panel.clone(), |p| p.title().to_string()) };
            if plan.run.is_empty() {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
            let panel_eta = panel_eta_s(plan.estimate, plan.pass, plan.passes, state.fraction,
                plan.started.elapsed().as_secs_f64());
            self.emit(json!({"type": "progress", "run": plan.run, "panel": plan.panel, "title": title,
                "label": state.label, "panel_fraction": state.fraction, "panel_eta_s": panel_eta,
                "fraction": fraction, "eta_s": eta, "live": {"tok_s": tok_s, "active": active_requests}}));
            if state.revision != revision {
                revision = state.revision;
                if let Some(partial) = state.partial {
                    if let Ok(mut r) = report.lock() {
                        Self::apply_partial(&mut r, progress, &plan.panel, &partial);
                    }
                    self.emit(json!({"type": "partial", "run": plan.run, "panel": plan.panel, "value": partial}));
                }
            }
            std::thread::sleep(Duration::from_millis(1000));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::panel_eta_s;

    fn close(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

    #[test]
    fn stale_baseline_partial_never_replaces_final_baseline() {
        let mut report = crate::sample::report(false);
        let final_checks = report.baseline.as_ref().unwrap().quality.checks.len();
        let mut stale = report.baseline.clone().unwrap();
        stale.quality.checks.truncate(final_checks - 1);
        let partial = serde_json::to_value(&stale).unwrap();
        let progress = crate::panels::Progress::default();
        // Before the install a partial applies; after it, a late partial is dropped.
        super::Bench::apply_partial(&mut report, &progress, "baseline", &partial);
        assert_eq!(report.baseline.as_ref().unwrap().quality.checks.len(), final_checks - 1);
        report.baseline = Some(serde_json::from_value(serde_json::to_value(crate::sample::report(false).baseline).unwrap()).unwrap());
        progress.finish_baseline();
        super::Bench::apply_partial(&mut report, &progress, "baseline", &partial);
        assert_eq!(report.baseline.as_ref().unwrap().quality.checks.len(), final_checks);
    }

    #[test]
    fn internal_token_is_private_full_length_and_only_active() {
        let bench = super::Bench::new(crate::store::Store::memory().unwrap());
        assert!(!bench.accepts_internal("private-run-token"));
        let active = super::ActiveRun { id: "run".into(), token: "private-run-token".into(),
            cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            started: std::time::Instant::now(), eta_s: 10.0, panel: "basic".into(), fraction: 0.0 };
        assert!(!format!("{active:?}").contains("private-run-token"));
        *bench.active.lock().unwrap() = Some(active);
        assert!(bench.accepts_internal("private-run-token"));
        assert_eq!(bench.locked(Some("private-run-token")), None);
        for wrong in ["", "private-run-toke", "private-run-token-tail"] {
            assert!(!bench.accepts_internal(wrong));
            assert!(bench.locked(Some(wrong)).is_some());
        }
        assert!(!bench.status().to_string().contains("private-run-token"));
    }

    #[test]
    fn fidelity_catalog_defers_unknown_server_until_basic_card_discovery() {
        let directory = tempfile::tempdir().unwrap();
        let bench = super::Bench::new(crate::store::Store::open(directory.path()).unwrap());
        let catalog = bench.catalog();
        for id in ["fidelity", "fidelity_full"] {
            let panel = catalog["panels"].as_array().unwrap().iter().find(|p| p["id"] == id).unwrap();
            let reason = panel["unavailable"].as_str().unwrap();
            assert!(reason.contains("Discovering server configuration"));
            assert!(!reason.contains("no verified"));
        }
        let baseline = catalog["panels"].as_array().unwrap().iter().find(|p| p["id"] == "baseline").unwrap();
        assert!(baseline["unavailable"].is_null());
    }

    #[test]
    fn panel_eta_uses_the_estimate_before_a_pass_has_data() {
        // Three passes of 100 s, a fifth through the first: the estimate, and nothing measured yet.
        assert!(close(panel_eta_s(100.0, 1, 3, 0.2, 0.0), 80.0 + 200.0));
    }

    #[test]
    fn panel_eta_follows_the_measured_pace_once_a_pass_is_under_way() {
        // 25 s into a pass that is half done: 25 s left, plus the later passes.
        assert!(close(panel_eta_s(100.0, 1, 2, 0.5, 25.0), 25.0 + 100.0));
        // The last pass has no later passes to price in.
        assert!(close(panel_eta_s(100.0, 2, 2, 0.25, 30.0), 90.0));
    }

    #[test]
    fn panel_eta_never_goes_negative() {
        assert!(close(panel_eta_s(0.0, 1, 1, 1.0, 10.0), 0.0));
    }
}
