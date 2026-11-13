//! What the server tells the benchmark about itself: its family, resolved
//! options with their defaults, snapshot, start and readiness times, and the
//! loopback address the runner drives. The daemon fills this in once at
//! startup; everything here is cheap to read.
use crate::report::Setting;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

#[derive(Debug, Clone, Default)]
pub struct ServerContext {
    /// The serve command that runs (`serve-native`, `serve-glm`, ...).
    pub command: String,
    pub family: Option<String>,
    pub snapshot: Option<PathBuf>,
    pub settings: Vec<Setting>,
    pub supports_thinking_off: Option<bool>,
    /// Argument definitions whose launch values are not runtime facts.
    pub derived: Vec<String>,
}

impl ServerContext {
    /// Fail closed when a startup-normalized option has not been published.
    pub fn validate_resolved(&self) -> Result<(), Vec<String>> {
        let missing: Vec<_> = self.derived.iter().filter(|name| !self.settings.iter()
            .any(|s| &s.name == *name && s.source == "resolved" && s.value.is_some())).cloned().collect();
        if missing.is_empty() { Ok(()) } else { Err(missing) }
    }

    /// Require runtime evidence even for an explicitly supplied launch value.
    pub fn require_resolved(&mut self, name: &str) {
        if !self.derived.iter().any(|n| n == name) { self.derived.push(name.into()); }
        for setting in self.settings.iter_mut().filter(|s| s.name == name && s.source != "resolved") {
            setting.source = "unresolved".into();
            setting.value = None;
        }
        if !self.settings.iter().any(|s| s.name == name) {
            self.settings.push(Setting { name: name.into(), source: "unresolved".into(), ..Default::default() });
        }
    }

    /// No report caller may substitute a launch request for missing runtime evidence.
    pub fn report_settings(&self) -> Vec<Setting> {
        let mut context = self.clone();
        if let Err(missing) = context.validate_resolved() {
            for name in missing { context.require_resolved(&name); }
        }
        context.settings
    }

    pub fn draft_mode(&self) -> &str {
        self.settings.iter().find(|setting| setting.name == "draft-mode" && setting.source == "resolved")
            .and_then(|setting| setting.value.as_deref()).unwrap_or_else(||
                if crate::server::speculator(self.family.as_deref(), &self.settings).is_some() { "adaptive" }
                else { "no speculation" })
    }

    pub fn resolve(&mut self, name: &str, value: &str) {
        apply_resolved(&mut self.settings, name, value);
    }

    /// The setting does not apply to what served: drop every record of it, launch-time ones
    /// included, so no report shows a value the server never had.
    pub fn withdraw(&mut self, name: &str) {
        self.settings.retain(|setting| setting.name != name);
        self.derived.retain(|derived| derived != name);
    }

    /// Publish the canonical identity from the engine's validated target configuration.
    pub fn set_family(&mut self, family: &str) {
        self.family = Some(family.to_owned());
    }
}

/// The engine calls this after loading its target config, independently of its serve alias.
pub fn set_family(family: &str) {
    if let Ok(mut slot) = state().context.lock() { slot.set_family(family); }
}

/// What the engine resolved for its device KV pool and host prefix cache.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct KvCapacity {
    pub tokens: Option<u64>,
    /// Pool ceiling without RTX-onboarded experts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    pub pages: Option<u64>,
    /// KV record format, e.g. `FP8 latent + index` or `int8 full + BF16 SWA`.
    pub format: Option<String>,
    /// Pinned host prefix-cache bytes (0: off).
    pub host_bytes: Option<u64>,
}

struct State {
    kv: Mutex<Option<KvCapacity>>,
    mapped: Mutex<Option<cuteafd_loader::formats::mapped_preload::MappedPlacement>>,
    context: Mutex<ServerContext>,
    phases: Mutex<Vec<(String, SystemTime)>>,
    started: SystemTime,
    ready: OnceLock<SystemTime>,
    listen: OnceLock<SocketAddr>,
}

impl State {
    fn new() -> Self { Self {
        context: Mutex::new(ServerContext::default()),
        kv: Mutex::new(None),
        mapped: Mutex::new(None),
        phases: Mutex::new(Vec::new()),
        started: process_start().unwrap_or_else(SystemTime::now),
        ready: OnceLock::new(),
        listen: OnceLock::new(),
    } }
}

fn state() -> &'static State {
    static STATE: OnceLock<State> = OnceLock::new();
    STATE.get_or_init(State::new)
}

/// Records the serve command's resolved settings (called once from `main`).
pub fn set(context: ServerContext) {
    let _ = state();
    if let Ok(mut slot) = state().context.lock() {
        *slot = context;
    }
}

/// Records a value the engine resolved at admission (the resolved onboard
/// and KV pool), replacing every launch-time setting of that name so the
/// card's configuration panel shows what actually served, once.
/// Replaces every copy of `name` with one resolved record. The first copy's
/// default is kept; a name that was not present is appended.
fn apply_resolved(settings: &mut Vec<Setting>, name: &str, value: &str) {
    let mut resolved = Setting { name: name.into(), value: Some(value.into()), default: None, source: "resolved".into() };
    let mut replaced = false;
    settings.retain_mut(|existing| {
        if existing.name != resolved.name { return true; }
        if replaced { return false; }
        resolved.default = existing.default.clone();
        *existing = resolved.clone();
        replaced = true;
        true
    });
    if !replaced { settings.push(resolved); }
}

#[cfg(feature = "test-support")]
thread_local! {
    static SEAM_SETTINGS: std::cell::RefCell<Option<ResolvedSettings>> = const { std::cell::RefCell::new(None) };
}

/// Capture the production publication calls on this CPU fixture's thread.
#[cfg(feature = "test-support")]
pub fn capture_resolved<T>(seam: impl FnOnce() -> T) -> (T, ResolvedSettings) {
    struct Reset;
    impl Drop for Reset { fn drop(&mut self) { SEAM_SETTINGS.with(|slot| { slot.borrow_mut().take(); }); } }
    SEAM_SETTINGS.with(|slot| {
        assert!(slot.borrow().is_none(), "nested resolution capture");
        *slot.borrow_mut() = Some(ResolvedSettings::default());
    });
    let _reset = Reset;
    let result = seam();
    let settings = SEAM_SETTINGS.with(|slot| slot.borrow_mut().take().unwrap());
    (result, settings)
}

pub fn set_resolved(name: &str, value: &str) {
    #[cfg(feature = "test-support")]
    if SEAM_SETTINGS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(settings) = slot.as_mut() {
            settings.0.push((name.into(), value.into()));
            true
        } else { false }
    }) { return; }
    publish_resolved(state(), name, value);
}

fn publish_resolved(state: &State, name: &str, value: &str) {
    if let Ok(mut slot) = state.context.lock() {
        if state.ready.get().is_some() {
            tracing::error!(setting = name, "resolved settings are sealed at readiness; publication rejected");
            drop(slot);
            debug_assert!(false, "resolved setting {name} published after readiness");
            return;
        }
        apply_resolved(&mut slot.settings, name, value);
    }
}

/// Withdraw a setting that does not apply to the served checkpoint (see `ServerContext::withdraw`).
pub fn withdraw(name: &str) {
    let state = state();
    if let Ok(mut slot) = state.context.lock() {
        if state.ready.get().is_some() {
            tracing::error!(setting = name, "resolved settings are sealed at readiness; withdrawal rejected");
            drop(slot);
            debug_assert!(false, "setting {name} withdrawn after readiness");
            return;
        }
        slot.withdraw(name);
    }
}

/// Publish the accepted expert placement, including zero onboarded layers.
pub fn set_rtx_expert_layers(layers: usize) {
    set_resolved("rtx-expert-layers", &layers.to_string());
    if get().derived.iter().any(|name| name == "onboard") { set_resolved("onboard", &layers.to_string()); }
}

/// Runtime facts emitted by a normalization/admission seam. Kept local until
/// publication so CPU fixtures can validate them without process-global state.
#[derive(Debug, Default)]
pub struct ResolvedSettings(pub Vec<(String, String)>);
impl ResolvedSettings {
    pub fn value(mut self, name: &str, value: impl ToString) -> Self {
        self.0.push((name.into(), value.to_string()));
        self
    }
    /// A seam test exercises the very records the serving path publishes.
    #[cfg(any(test, feature = "test-support"))]
    pub fn assert_resolved(&self) {
        let mut context = ServerContext::default();
        for (name, _) in &self.0 { context.require_resolved(name); }
        assert!(context.validate_resolved().is_err(), "fixture must require runtime facts");
        self.apply(&mut context);
        assert_eq!(context.validate_resolved(), Ok(()));
        assert!(context.report_settings().iter().all(|setting| setting.source == "resolved"));
    }

    pub fn apply(&self, context: &mut ServerContext) {
        for (name, value) in &self.0 { context.resolve(name, value); }
    }
    pub fn publish(&self) {
        for (name, value) in &self.0 { set_resolved(name, value); }
    }
    pub fn prefill(plan: cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan) -> Self {
        Self::default().value("prefill-rows", plan.unit_rows).value("prefill-waves", plan.waves)
            .value("prefill-span-rows", plan.span_rows)
    }
    pub fn placement(placement: &cuteafd_loader::placement::Placement) -> Self {
        Self::default().value("pool-tokens", placement.pool_tokens)
            .value("rtx-expert-layers", placement.onboard_layers).value("onboard", placement.onboard_layers)
            .value("attention-placement", placement.attention_summary())
    }
}

pub fn set_thinking_off(supported: bool) {
    if let Ok(mut slot) = state().context.lock() { slot.supports_thinking_off = Some(supported); }
}

pub fn get() -> ServerContext {
    let mut context = state().context.lock().map(|c| c.clone()).unwrap_or_default();
    if let Err(missing) = context.validate_resolved() {
        for name in missing { context.require_resolved(&name); }
    }
    context
}

/// The engine's resolved device KV pool (tokens, pages, record format) and host cache bytes.
pub fn set_kv(tokens: u64, max_tokens: Option<u64>, pages: u64, format: &str, host_bytes: u64) {
    set_resolved("pool-tokens", &tokens.to_string());
    set_resolved("kv-format", format);
    if let Ok(mut slot) = state().kv.lock() {
        *slot = Some(KvCapacity { tokens: Some(tokens), max_tokens, pages: Some(pages), format: Some(format.to_string()),
            host_bytes: Some(host_bytes) });
    }
}

pub fn kv() -> Option<KvCapacity> {
    state().kv.lock().ok().and_then(|k| k.clone())
}

/// A named startup milestone (the startup panel's Gantt).
pub fn phase(name: &str) {
    if let Ok(mut phases) = state().phases.lock() {
        phases.push((name.to_string(), SystemTime::now()));
    }
}

/// Startup milestones as seconds after process start.
pub fn phases() -> Vec<(String, f64)> {
    let start = state().started;
    state().phases.lock().map(|p| p.iter().map(|(n, t)| (n.clone(),
        t.duration_since(start).map(|d| d.as_secs_f64()).unwrap_or(0.0))).collect()).unwrap_or_default()
}

/// The API accepts requests on `listen` from now on.
pub fn mark_ready(listen: SocketAddr) {
    if let Err(missing) = get().validate_resolved() {
        for setting in missing { tracing::warn!(%setting, "derived report setting was not resolved at readiness"); }
    }
    let placement = get().settings.iter().find(|s| s.name == "attention-placement").and_then(|s| s.value.clone());
    if let Some(placement) = placement { tracing::info!(attention_placement = placement, "ready attention placement"); }
    phase("API listening");
    seal(state(), listen);
}

fn seal(state: &State, listen: SocketAddr) {
    // Serialize the readiness boundary with the last admitted setting.
    let _context = state.context.lock().unwrap_or_else(|p| p.into_inner());
    let _ = state.ready.set(SystemTime::now());
    let _ = state.listen.set(listen);
}

pub fn started() -> SystemTime {
    state().started
}

/// Seconds from process start to readiness, once ready.
pub fn readiness_s() -> Option<f64> {
    let ready = state().ready.get()?;
    ready.duration_since(state().started).ok().map(|d| d.as_secs_f64())
}

/// Where the runner reaches this server: the listen port on loopback.
pub fn loopback() -> Option<String> {
    let listen = state().listen.get()?;
    let host = if listen.is_ipv6() && !listen.ip().is_unspecified() && !listen.ip().is_loopback() {
        format!("[{}]", listen.ip())
    } else if listen.ip().is_unspecified() || listen.ip().is_loopback() {
        if listen.is_ipv6() { "[::1]".to_string() } else { "127.0.0.1".to_string() }
    } else {
        listen.ip().to_string()
    };
    Some(format!("http://{host}:{}", listen.port()))
}

/// The process start time from /proc (Linux), so readiness counts the whole load.
fn process_start() -> Option<SystemTime> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Field 22 (starttime, clock ticks since boot) follows the parenthesized command.
    let after = stat.rsplit_once(')')?.1;
    let ticks: u64 = after.split_whitespace().nth(19)?.parse().ok()?;
    let uptime: f64 = std::fs::read_to_string("/proc/uptime").ok()?.split_whitespace().next()?.parse().ok()?;
    let hz = 100.0; // _SC_CLK_TCK is 100 on every Linux target this runs on.
    let since_start = uptime - ticks as f64 / hz;
    SystemTime::now().checked_sub(std::time::Duration::from_secs_f64(since_start.max(0.0)))
}

/// Settings from environment variables that change engine behaviour
/// (`CUTEAFD_*`), excluding identity, paths and secrets.
pub fn env_settings() -> Vec<Setting> {
    const IDENTITY: [&str; 16] = ["CUTEAFD_ENGINE_COMMIT", "CUTEAFD_SPARKINFER_COMMIT", "CUTEAFD_RELEASE_VERSION",
        "CUTEAFD_ROLE", "CUTEAFD_CUDA_ARCH", "CUTEAFD_RELEASE_CONFIG_SHA256", "CUTEAFD_CONSOLE_REVISION",
        "CUTEAFD_NATIVE_LIB", "CUTEAFD_CONSOLE_PAGE", "CUTEAFD_TARGET_PLATFORM", "CUTEAFD_IMAGE",
        "CUTEAFD_GIT_REMOTE", "CUTEAFD_PYTHON", "CUTEAFD_SPARK_TP_ROLES", "CUTEAFD_CONSOLE_TEXT", "CUTEAFD_API_KEY"];
    // Launchers pass these spellings for "the engine's default".
    let neutral = |v: &str| matches!(v, "" | "auto" | "default" | "false" | "off" | "0");
    let mut out: Vec<Setting> = std::env::vars()
        .filter(|(name, _)| name.starts_with("CUTEAFD_"))
        .filter(|(name, _)| !IDENTITY.contains(&name.as_str()) && !name.starts_with("CUTEAFD_BENCH")
            && !name.starts_with("CUTEAFD_RELEASE_"))
        .filter(|(name, _)| !["KEY", "TOKEN", "SECRET", "PASSWORD"].iter().any(|s| name.contains(s)))
        .filter(|(name, _)| !name.ends_with("_DIR") && !name.ends_with("_PATH") && !name.ends_with("_TRACE"))
        .map(|(name, value)| {
            let default = neutral(&value).then(|| value.clone());
            Setting { name, value: Some(value), default, source: "env".into() }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Option names that describe the deployment rather than a tuning choice:
/// never shown as chips (they are in the hardware and configuration panels).
pub const DEPLOYMENT: [&str; 16] = ["snapshot", "native-lib", "native_lib", "peers", "listen", "model-id",
    "model_id", "hf-home", "hf_home", "device", "api-key-file", "usage-dir", "console-secret-file", "enable-bench",
    "CUTEAFD_CONSOLE_LINK_COMMAND", "CUTEAFD_FIDELITY_CACHE"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_derived_values_fail_closed_for_every_launch_source() {
        for source in ["default", "cli", "env", "unset"] {
            let mut context = ServerContext { settings: vec![Setting {
                name: "slots".into(), value: Some("8".into()), default: Some("8".into()), source: source.into(),
            }], ..Default::default() };
            context.require_resolved("slots");
            context.require_resolved("index-cache");
            assert_eq!(context.validate_resolved().unwrap_err(), ["slots", "index-cache"]);
            let settings = context.report_settings();
            assert!(settings.iter().all(|s| s.source == "unresolved" && s.value.is_none()));
            ResolvedSettings::default().value("slots", 16).value("index-cache", "compact").apply(&mut context);
            assert_eq!(context.validate_resolved(), Ok(()));
            assert_eq!(context.settings[0].value.as_deref(), Some("16"));
            assert_eq!(context.settings[0].default.as_deref(), Some("8"));
        }
    }

    #[test]
    fn withdrawn_settings_leave_no_launch_or_resolved_record() {
        let launch = |name: &str| Setting { name: name.into(), value: Some("original".into()),
            default: Some("original".into()), source: "default".into() };
        let mut context = ServerContext { settings: vec![launch("core-view"), launch("slots")], ..Default::default() };
        context.resolve("core-view", "original");
        context.require_resolved("core-view-fingerprint");
        context.withdraw("core-view");
        context.withdraw("core-view-fingerprint");
        assert_eq!(context.validate_resolved(), Ok(()));
        let names: Vec<_> = context.report_settings().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["slots"]);
    }

    #[test]
    fn configuration_panel_labels_non_derived_defaults_as_launch_defaults() {
        assert!(include_str!("../assets/bench.html").contains("s.source === 'default' ? 'launch default' : s.source"));
    }

    #[test]
    fn kv_capacity_maximum_is_optional_in_old_records() {
        let old = r#"{"tokens":1800000,"pages":28125,"format":"FP8","host_bytes":0}"#;
        let mut capacity: KvCapacity = serde_json::from_str(old).unwrap();
        assert_eq!(capacity.max_tokens, None);
        assert!(!serde_json::to_string(&capacity).unwrap().contains("max_tokens"));
        capacity.max_tokens = Some(2_100_000);
        let json = serde_json::to_string(&capacity).unwrap();
        assert_eq!(serde_json::from_str::<KvCapacity>(&json).unwrap(), capacity);
    }

    #[test]
    fn process_start_is_in_the_past() {
        let start = process_start().expect("linux /proc");
        assert!(start <= SystemTime::now());
    }

    #[test]
    fn loopback_rewrites_unspecified_hosts() {
        mark_ready("0.0.0.0:8123".parse().unwrap());
        assert_eq!(loopback().as_deref(), Some("http://127.0.0.1:8123"));
        assert!(readiness_s().is_some());
    }

    #[test]
    fn admitted_attention_overrides_launch_default_before_ready() {
        let state = State::new();
        state.context.lock().unwrap().settings.push(Setting {
            name: "attention-placement".into(), value: Some("layers".into()),
            default: Some("layers".into()), source: "resolved".into(),
        });
        publish_resolved(&state, "attention-placement", "context");
        seal(&state, "127.0.0.1:8123".parse().unwrap());
        let context = state.context.lock().unwrap().clone();
        let settings = context.settings.iter().filter(|s| s.name == "attention-placement").collect::<Vec<_>>();
        assert_eq!(settings.len(), 1);
        assert_eq!(settings[0].value.as_deref(), Some("context"));
        assert_eq!(settings[0].source, "resolved");
    }

    #[test]
    fn every_post_ready_publication_is_rejected_without_mutating_identity() {
        for name in ["pool-tokens", "route-source", "runtime-counter", "new-field"] {
            let state = State::new();
            publish_resolved(&state, name, "before");
            seal(&state, "127.0.0.1:8123".parse().unwrap());
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                publish_resolved(&state, name, "after");
            }));
            assert_eq!(rejected.is_err(), cfg!(debug_assertions));
            let context = state.context.lock().unwrap();
            assert_eq!(context.settings.len(), 1);
            assert_eq!(context.settings[0].value.as_deref(), Some("before"));
        }
    }

    #[test]
    fn resolved_setting_replaces_every_copy_of_the_name() {
        let mut settings = vec![
            Setting { name: "attention-placement".into(), value: Some("auto".into()),
                default: Some("auto".into()), source: "env".into() },
            Setting { name: "concurrency".into(), value: Some("4".into()), default: None, source: "cli".into() },
            Setting { name: "attention-placement".into(), value: Some("context".into()),
                default: Some("context".into()), source: "resolved".into() },
        ];
        apply_resolved(&mut settings, "attention-placement", "context");
        let placed: Vec<_> = settings.iter().filter(|s| s.name == "attention-placement").collect();
        assert_eq!(placed.len(), 1, "{settings:?}");
        assert_eq!((placed[0].value.as_deref(), placed[0].default.as_deref(), placed[0].source.as_str()),
            (Some("context"), Some("auto"), "resolved"));
        assert!(settings.iter().any(|s| s.name == "concurrency"));
        apply_resolved(&mut settings, "pool-tokens", "1024");
        assert_eq!(settings.iter().filter(|s| s.name == "pool-tokens").count(), 1);
    }
}

/// Resolved resident/mapped table owners, published only after preload admission.
pub fn set_mapped_placement(placement: cuteafd_loader::formats::mapped_preload::MappedPlacement) {
    if let Ok(mut slot) = state().mapped.lock() { *slot = Some(placement); }
}
pub fn mapped_placement() -> Option<cuteafd_loader::formats::mapped_preload::MappedPlacement> {
    state().mapped.lock().ok().and_then(|slot| slot.clone())
}
