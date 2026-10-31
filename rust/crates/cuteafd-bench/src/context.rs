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

fn state() -> &'static State {
    static STATE: OnceLock<State> = OnceLock::new();
    STATE.get_or_init(|| State {
        context: Mutex::new(ServerContext::default()),
        kv: Mutex::new(None),
        mapped: Mutex::new(None),
        phases: Mutex::new(Vec::new()),
        started: process_start().unwrap_or_else(SystemTime::now),
        ready: OnceLock::new(),
        listen: OnceLock::new(),
    })
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

pub fn set_resolved(name: &str, value: &str) {
    if let Ok(mut slot) = state().context.lock() {
        apply_resolved(&mut slot.settings, name, value);
    }
}

/// Publish the accepted expert placement, including zero onboarded layers.
pub fn set_rtx_expert_layers(layers: usize) {
    set_resolved("rtx-expert-layers", &layers.to_string());
}

pub fn set_thinking_off(supported: bool) {
    if let Ok(mut slot) = state().context.lock() { slot.supports_thinking_off = Some(supported); }
}

pub fn get() -> ServerContext {
    state().context.lock().map(|c| c.clone()).unwrap_or_default()
}

/// The engine's resolved device KV pool (tokens, pages, record format) and host cache bytes.
pub fn set_kv(tokens: u64, max_tokens: Option<u64>, pages: u64, format: &str, host_bytes: u64) {
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
    let placement = get().settings.iter().find(|s| s.name == "attention-placement").and_then(|s| s.value.clone());
    if let Some(placement) = placement { tracing::info!(attention_placement = placement, "ready attention placement"); }
    phase("API listening");
    let _ = state().ready.set(SystemTime::now());
    let _ = state().listen.set(listen);
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
