//! The report record: what a run measured, on what, and how it was configured.
//! One JSON document per run; the dashboard, the exports and `bench publish`
//! all read this shape, and a saved `report.json` loads back into the view.
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SCHEMA: &str = "cuteafd.bench.report/1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Queued,
    Running,
    Done,
    Cancelled,
    Failed,
    /// Loaded from a saved `report.json`.
    Imported,
}

impl RunStatus {
    pub fn finished(self) -> bool {
        !matches!(self, Self::Queued | Self::Running)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema: String,
    /// Absent in historical reports that used synthetic prompts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_content: Option<String>,
    pub id: String,
    /// RFC 3339 UTC.
    pub created: String,
    #[serde(default)]
    pub finished: Option<String>,
    pub status: RunStatus,
    /// Profile name (`share`, `speed`, ..., or a saved custom profile).
    pub profile: String,
    /// Panels in run order with their pass counts.
    pub plan: Vec<PlannedPanel>,
    pub server: ServerInfo,
    pub fingerprint: String,
    #[serde(default)]
    pub baseline: Option<Baseline>,
    #[serde(default)]
    pub panels: Vec<PanelResult>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fabric: Vec<crate::fabric::Phase>,
}

impl Report {
    /// A planner-only rejection has no measured baseline or performance values.
    pub fn no_fit_reason(&self) -> Option<&str> {
        (self.baseline.is_none() && self.status == RunStatus::Failed)
            .then(|| self.server.setting("qualification.no-fit"))
            .flatten()
            .filter(|reason| !reason.is_empty())
    }

    /// True when the baseline's quality gate failed: every view and export of
    /// this report carries the warning.
    pub fn quality_failed(&self) -> bool {
        self.baseline.as_ref().is_some_and(|b| b.quality.status == CheckStatus::Fail)
    }

    pub fn panel(&self, id: &str) -> Option<&PanelResult> {
        self.panels.iter().find(|p| p.id == id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedPanel {
    pub id: String,
    pub passes: u32,
}

impl ServerInfo {
    /// The checkpoint repository the server loaded (`org/name` from a Hugging Face
    /// snapshot path), else the served model name: quants of one model share a
    /// served name but not a checkpoint.
    pub fn checkpoint(&self) -> String {
        hub_repo(self.configuration.snapshot.as_deref().unwrap_or("")).unwrap_or_else(|| self.model.clone())
    }
}

#[cfg(test)]
mod checkpoint_tests {
    #[test]
    fn checkpoint_comes_from_the_snapshot_path() {
        let mut s = super::ServerInfo { model: "zai-org/GLM-5.3".into(), ..Default::default() };
        assert_eq!(s.checkpoint(), "zai-org/GLM-5.3");
        s.configuration.snapshot = Some("/root/.cache/huggingface/hub/models--nvidia--GLM-5.3-NVFP4/snapshots/e3b8".into());
        assert_eq!(s.checkpoint(), "nvidia/GLM-5.3-NVFP4");
    }
}

/// What served the run: model, build, hardware, resolved configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerInfo {
    pub model: String,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    pub build: BuildInfo,
    pub hardware: Hardware,
    pub configuration: Configuration,
    /// Seconds from process start to the API accepting requests.
    #[serde(default)]
    pub readiness_s: Option<f64>,
    /// RFC 3339 UTC of the server's start.
    #[serde(default)]
    pub started: Option<String>,
}

impl ServerInfo {
    pub fn setting(&self, name: &str) -> Option<&str> {
        self.configuration.settings.iter().find(|s| s.name == name)
            .and_then(|s| s.value.as_deref())
    }

    /// The chart and basic sweep stop at C8 on cards with at most 32 GiB per GPU.
    pub fn card_levels(&self) -> &'static [usize] {
        let used: Vec<_> = self.hardware.gpus.iter().filter(|g| g.used).collect();
        let gpus = if used.is_empty() { self.hardware.gpus.iter().collect() } else { used };
        let small = self.column_5090() || self.coordinator_budget().is_some_and(|gib| gib <= 32.0)
            || gpus.iter().any(|gpu| gpu.memory_mib.is_some_and(|mib| mib <= 32 * 1024));
        if small { &[1, 4, 8] } else { &[1, 4, 8, 16] }
    }

    pub fn coordinator_budget(&self) -> Option<f64> {
        self.setting("coordinator-gpu-budget-gib")?.parse::<f64>().ok()
            .filter(|gib| gib.is_finite() && *gib > 0.0)
    }

    pub fn simulated_5090(&self) -> bool {
        self.setting("simulated") == Some("5090")
    }

    pub fn column_5090(&self) -> bool {
        self.simulated_5090() || self.setting("hardware.class") == Some("5090")
    }

    pub fn hardware_line(&self) -> String {
        if self.simulated_5090() {
            let sms = self.hardware.gpus.iter().find(|g| g.used)
                .or_else(|| self.hardware.gpus.first()).and_then(|g| g.sm_count);
            let sms = sms.map(|v| format!(" ({v} SMs)")).unwrap_or_default();
            let cap = self.coordinator_budget().map(|g| format!(" capped at {g} GiB"))
                .unwrap_or_else(|| " (memory cap unknown)".into());
            let sparks = if self.hardware.sparks.is_empty() { String::new() }
                else { format!(" + {} Spark", self.hardware.sparks.len()) };
            format!("simulated 5090: RTX PRO 6000{sms}{cap}{sparks}")
        } else {
            let mut line = self.hardware.line();
            if let Some(gib) = self.coordinator_budget() {
                line.push_str(&format!(" · {gib} GiB budget"));
            }
            line
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct BuildInfo {
    /// Crate version of the binary.
    pub version: String,
    /// Release tag, when built from one.
    #[serde(default)]
    pub release: Option<String>,
    /// Container image (name:tag or digest), when known.
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub remote: Option<String>,
    #[serde(default)]
    pub commit: Option<String>,
    #[serde(default)]
    pub dirty: Option<bool>,
}

impl BuildInfo {
    /// One line for footers and the README: the release tag, else
    /// `<commit12>[+dirty]`, else the crate version.
    pub fn label(&self) -> String {
        if let Some(release) = &self.release {
            return release.clone();
        }
        match &self.commit {
            Some(commit) => format!("{}{}", &commit[..commit.len().min(12)],
                if self.dirty == Some(true) { "+dirty" } else { "" }),
            None => format!("v{}", self.version),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Hardware {
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub gpus: Vec<Gpu>,
    #[serde(default)]
    pub driver: Option<String>,
    #[serde(default)]
    pub cuda: Option<String>,
    #[serde(default)]
    pub sparks: Vec<Spark>,
    #[serde(default)]
    pub fabric: Vec<FabricPort>,
    /// The rail plan's one-line summary.
    #[serde(default)]
    pub rails: Option<String>,
    /// What the used GPUs' power limits actually were over the timed windows (absent in old
    /// reports and without a throttle sampler).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<Power>,
}

/// Measured power limits of the used GPUs over a run, summarized from the engine's
/// throttle windows (`/v1/thermal`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Power {
    /// `gpu-powerd` when a governor state was present in the windows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub governor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_w: Option<f64>,
    pub gpus: Vec<PowerGpu>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PowerGpu {
    pub index: u32,
    /// Time-weighted mean power limit over the windows.
    pub mean_limit_w: f64,
    pub min_limit_w: f64,
    pub max_limit_w: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_draw_w: Option<f64>,
    /// The governor's dominant plan bound (`max`, `budget`, `model`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound: Option<String>,
    /// Measured seconds with a software power cap or hardware power brake.
    #[serde(default)]
    pub cap_seconds: f64,
}

fn collect_windows<'a>(value: &'a Value, out: &mut Vec<&'a Value>) {
    match value {
        Value::Object(map) => {
            if map.contains_key("from_s") && map.get("gpus").is_some_and(Value::is_array) {
                out.push(value);
            } else {
                map.values().for_each(|v| collect_windows(v, out));
            }
        }
        Value::Array(items) => items.iter().for_each(|v| collect_windows(v, out)),
        _ => {}
    }
}

impl Power {
    /// Merge throttle windows (any JSON containing `thermal::window` objects) into one summary,
    /// weighting each window by its seconds. `None` when no window carries a limit.
    pub fn from_json(roots: &[&Value]) -> Option<Power> {
        let mut windows = Vec::new();
        roots.iter().for_each(|root| collect_windows(root, &mut windows));
        // index -> (weight, limit*weight, draw weight, draw*weight, min, max, bound seconds)
        #[derive(Default)]
        struct Acc { w: f64, limit: f64, dw: f64, draw: f64, min: Option<f64>, max: Option<f64>, bounds: Vec<(String, f64)>, cap_seconds: f64 }
        let mut accs: std::collections::BTreeMap<u32, Acc> = Default::default();
        let (mut governor, mut budget) = (false, None);
        for window in windows {
            let weight = window["seconds"].as_f64().unwrap_or(0.0).max(0.001);
            if window.get("governor").is_some_and(|g| !g.is_null()) {
                governor = true;
                budget = window["governor_budget_w"].as_f64().or(budget);
            }
            for gpu in window["gpus"].as_array().into_iter().flatten() {
                let (Some(index), Some(mean)) = (gpu["index"].as_u64(), gpu["power_limit_w"]["mean"].as_f64()
                    .or_else(|| gpu["power_limit_w"]["max"].as_f64())) else { continue };
                let acc = accs.entry(index as u32).or_default();
                acc.cap_seconds += ["sw_power_cap", "hw_power_brake"].iter()
                    .filter_map(|reason| gpu["throttle_s"][reason].as_f64()).filter(|s| s.is_finite() && *s > 0.0).sum::<f64>();
                acc.w += weight;
                acc.limit += mean * weight;
                if let Some(draw) = gpu["power_w"]["mean"].as_f64() { acc.dw += weight; acc.draw += draw * weight; }
                let lo = gpu["power_limit_w"]["min"].as_f64().unwrap_or(mean);
                let hi = gpu["power_limit_w"]["max"].as_f64().unwrap_or(mean);
                acc.min = Some(acc.min.map_or(lo, |m| m.min(lo)));
                acc.max = Some(acc.max.map_or(hi, |m| m.max(hi)));
                if let Some(bound) = gpu["governor_bound"].as_str() {
                    match acc.bounds.iter_mut().find(|(b, _)| b == bound) {
                        Some(entry) => entry.1 += weight,
                        None => acc.bounds.push((bound.to_string(), weight)),
                    }
                }
            }
        }
        if accs.is_empty() { return None; }
        let gpus = accs.into_iter().map(|(index, a)| PowerGpu {
            index, mean_limit_w: a.limit / a.w, min_limit_w: a.min.unwrap_or_default(), max_limit_w: a.max.unwrap_or_default(),
            mean_draw_w: (a.dw > 0.0).then(|| a.draw / a.dw),
            cap_seconds: a.cap_seconds,
            bound: a.bounds.into_iter().max_by(|x, y| x.1.total_cmp(&y.1)).map(|(b, _)| b),
        }).collect();
        Some(Power { governor: governor.then(|| "gpu-powerd".to_string()), budget_w: budget, gpus })
    }

    pub fn capped(&self) -> bool {
        self.gpus.iter().any(|gpu| gpu.cap_seconds > 0.0)
    }

    /// `gpu-powerd 450/450 W` or `325 W`, plus ` (380–600)` when a limit moved by more than 10 W.
    pub fn label(&self) -> String {
        let round5 = |w: f64| (w / 5.0).round() * 5.0;
        let means: Vec<f64> = self.gpus.iter().map(|g| round5(g.mean_limit_w)).collect();
        let shown: Vec<f64> = if self.governor.is_none() && means.windows(2).all(|p| p[0] == p[1]) {
            means.iter().take(1).copied().collect()
        } else { means };
        let mut text = shown.iter().map(|w| format!("{w:.0}")).collect::<Vec<_>>().join("/");
        text.push_str(" W");
        if let Some(name) = &self.governor { text = format!("{name} {text}"); }
        let lo = self.gpus.iter().map(|g| g.min_limit_w).fold(f64::INFINITY, f64::min);
        let hi = self.gpus.iter().map(|g| g.max_limit_w).fold(f64::NEG_INFINITY, f64::max);
        if self.gpus.iter().any(|g| g.max_limit_w - g.min_limit_w > 10.0) {
            text.push_str(&format!(" ({:.0}–{:.0})", round5(lo), round5(hi)));
        }
        text
    }
}

impl Hardware {
    /// GPUs the server actually uses (all visible ones when unknown).
    pub fn used_gpus(&self) -> usize {
        let used = self.gpus.iter().filter(|g| g.used).count();
        if used == 0 { self.gpus.len() } else { used }
    }

    /// `1rtx-4spark`, `2rtx`, ... — the directory slug of `benchmarks/`.
    pub fn slug(&self) -> String {
        let gpus = self.used_gpus();
        let mut slug = format!("{gpus}rtx");
        if !self.sparks.is_empty() {
            slug.push_str(&format!("-{}spark", self.sparks.len()));
        }
        slug
    }

    /// `1× RTX PRO 6000 + 4× Spark` — one line for cards and tables.
    pub fn line(&self) -> String {
        let used: Vec<&Gpu> = {
            let used: Vec<&Gpu> = self.gpus.iter().filter(|g| g.used).collect();
            if used.is_empty() { self.gpus.iter().collect() } else { used }
        };
        let mut parts = Vec::new();
        if let Some(first) = used.first() {
            let mut part = format!("{}× {}", used.len(), short_gpu_name(&first.name));
            if let Some(power) = &self.power {
                part.push_str(&format!(" @ {}", power.label()));
            } else if let Some(cap) = first.power_limit_w {
                part.push_str(&format!(" @ {cap:.0} W"));
            }
            parts.push(part);
        }
        if !self.sparks.is_empty() {
            parts.push(format!("{}× DGX Spark", self.sparks.len()));
        }
        if parts.is_empty() { "unknown hardware".into() } else { parts.join(" + ") }
    }

    /// Reference class per AGENTS.md: the natural minimum is one RTX, the
    /// maximum two RTX with four or six Sparks.
    pub fn class(&self) -> HardwareClass {
        match (self.used_gpus(), self.sparks.len()) {
            (1, _) => HardwareClass::Minimum,
            (2, 4 | 6) => HardwareClass::Maximum,
            _ => HardwareClass::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareClass {
    Minimum,
    Maximum,
    Other,
}

/// "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" -> "RTX PRO 6000".
pub fn short_gpu_name(name: &str) -> String {
    let name = name.trim_start_matches("NVIDIA ").trim();
    if let Some(index) = name.find(" Blackwell") {
        return name[..index].to_string();
    }
    name.to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    #[serde(default)]
    pub uuid: Option<String>,
    #[serde(default)]
    pub memory_mib: Option<u64>,
    #[serde(default)]
    pub power_limit_w: Option<f64>,
    #[serde(default)]
    pub power_max_w: Option<f64>,
    #[serde(default)]
    pub sm_count: Option<u32>,
    #[serde(default)]
    pub compute_cap: Option<String>,
    #[serde(default)]
    pub pcie: Option<String>,
    /// Whether this server runs on it (from CUDA_VISIBLE_DEVICES / settings).
    #[serde(default)]
    pub used: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Spark {
    pub address: String,
    #[serde(default)]
    pub name: Option<String>,
    /// TP rank order of the expert service.
    pub rank: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FabricPort {
    pub device: String,
    pub port: u32,
    pub active: bool,
    pub link_gbps: f64,
    #[serde(default)]
    pub pcie: Option<String>,
    #[serde(default)]
    pub netdev: Option<String>,
    #[serde(default)]
    pub subnets: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Configuration {
    /// Renderer capability; absent preserves requests to older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_thinking_off: Option<bool>,
    #[serde(default)]
    pub snapshot: Option<String>,
    /// Storage formats per tensor group (from `cuteafd plan`).
    #[serde(default)]
    pub quant: Vec<QuantGroup>,
    #[serde(default)]
    pub speculator: Option<String>,
    /// One line, e.g. `TP4 × 1 Spark group, experts layers 0-18 on RTX`.
    #[serde(default)]
    pub layout: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapped_placement: Option<cuteafd_loader::formats::mapped_preload::MappedPlacement>,
    /// Every resolved option with its default.
    #[serde(default)]
    pub settings: Vec<Setting>,
}

impl Configuration {
    pub fn supports_thinking_off(&self) -> bool { self.supports_thinking_off != Some(false) }

    pub fn thinking_label(&self) -> &'static str {
        if self.supports_thinking_off() { "thinking off" } else { "thinking always on (template)" }
    }

    /// Options whose value differs from the default (the chips).
    pub fn non_default(&self) -> impl Iterator<Item = &Setting> {
        self.settings.iter().filter(|s| s.differs())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct QuantGroup {
    pub group: String,
    pub formats: Vec<String>,
}

/// One resolved option. `default: None` means the option has no default
/// (unset unless given); environment overrides carry `source: "env"`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Setting {
    pub name: String,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub default: Option<String>,
    /// `default`, `cli`, `env`, or `resolved` (decided at admission).
    pub source: String,
}

impl Setting {
    pub fn differs(&self) -> bool {
        self.source != "default" && self.value != self.default
    }

    /// `name=value` for a chip.
    pub fn chip(&self) -> String {
        match &self.value {
            Some(value) if value == "true" => self.name.clone(),
            Some(value) => format!("{}={}", self.name, hub_repo(value).unwrap_or_else(|| value.clone())),
            None => format!("{}=∅", self.name),
        }
    }
}

/// `org/name` for a Hugging Face cache path (`.../models--org--name/snapshots/rev`).
pub fn hub_repo(path: &str) -> Option<String> {
    path.split('/').find_map(|part| part.strip_prefix("models--"))
        .and_then(|repo| repo.split_once("--").map(|(org, name)| format!("{org}/{name}")))
}

impl Configuration {
    /// The speculator label with a snapshot revision in it replaced by the repository
    /// a setting loads it from ("DFlash2 (425aa6…)" → "DFlash2 (incoai/GLM-5.3-DFlash2)").
    pub fn speculator_label(&self) -> String {
        let mut label = self.speculator.clone().unwrap_or_else(|| "none".into());
        for setting in &self.settings {
            let Some(value) = &setting.value else { continue };
            let (Some(repo), Some(revision)) = (hub_repo(value), value.rsplit('/').next()) else { continue };
            if revision.len() >= 7 && label.contains(revision) {
                label = label.replace(revision, &repo);
            }
        }
        label
    }
}

/// The mandatory baseline: once per server lifetime and fingerprint, shared
/// by every run on it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub fingerprint: String,
    /// The run that measured it.
    pub run_id: String,
    pub created: String,
    pub card: BasicCard,
    pub quality: Quality,
    /// Wall seconds the baseline took.
    #[serde(default)]
    pub seconds: f64,
    /// GPU throttle report over the whole baseline run (`/v1/thermal`; absent without one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<Value>,
    /// Timed measurements only; warm-ups and quality probes are not attributed here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fabric: Vec<crate::fabric::Phase>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasicCard {
    /// Actual request mode; always-thinking templates must not be compared as off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// C1 decode per content type, thinking off.
    pub decode: Vec<ContentRate>,
    /// Decode sweep, thinking off; aggregate over each batch's decode interval.
    /// Each point names its prompt set ([`ConcurrentRate::prompts`]).
    #[serde(default)]
    pub concurrency: Vec<ConcurrentRate>,
    #[serde(default)]
    pub prefill: Option<PrefillRate>,
    /// Server tokens per reference token from the corpus warm-up fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_token_ratio: Option<f64>,
    /// Seconds of the untimed warm-up requests (first-use loads, graphs, tables).
    #[serde(default)]
    pub warmup_s: Option<f64>,
    /// The serving capacity the numbers were measured under.
    #[serde(default)]
    pub capacity: Option<Capacity>,
}

/// KV pool, request and context limits, host prefix cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Capacity {
    #[serde(default)]
    pub kv_tokens: Option<u64>,
    /// Pool ceiling without RTX-onboarded experts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub kv_pages: Option<u64>,
    #[serde(default)]
    pub kv_format: Option<String>,
    /// Requests decoding at once.
    #[serde(default)]
    pub max_requests: Option<u64>,
    #[serde(default)]
    pub max_context: Option<u64>,
    #[serde(default)]
    pub max_output: Option<u64>,
    /// Pinned host prefix-cache bytes (0: off).
    #[serde(default)]
    pub host_cache_bytes: Option<u64>,
}

/// Token counts in K/M: 412K, 14.7M.
pub fn short_tokens(v: u64) -> String {
    // Powers-of-two style sizes read best in binary units (32768 -> 32K).
    if v >= 1 << 20 && v % (1 << 20) == 0 {
        return format!("{}M", v >> 20);
    }
    if (1024..1 << 20).contains(&v) && v % 1024 == 0 {
        return format!("{}K", v >> 10);
    }
    match v {
        v if v >= 1_000_000 => format!("{:.1}M", v as f64 / 1e6).replace(".0M", "M"),
        v if v >= 10_000 => format!("{}K", (v as f64 / 1e3).round()),
        v if v >= 1_000 => format!("{:.1}K", v as f64 / 1e3).replace(".0K", "K"),
        v => v.to_string(),
    }
}

fn bytes_label(v: u64) -> String {
    if v == 0 { "off".into() } else if v >= 1 << 30 { format!("{:.0} GiB", v as f64 / (1u64 << 30) as f64) }
    else { format!("{:.0} MiB", v as f64 / (1u64 << 20) as f64) }
}

impl Capacity {
    /// From a report's resolved options, for records without a measured capacity.
    pub fn from_settings(info: &ServerInfo) -> Self {
        let get = |names: &[&str]| info.configuration.settings.iter().find(|s| names.contains(&s.name.as_str()))
            .and_then(|s| s.value.clone());
        let number = |names: &[&str]| get(names).and_then(|v| v.parse::<u64>().ok());
        Self {
            kv_tokens: number(&["pool-tokens"]),
            max_tokens: None,
            kv_pages: None,
            kv_format: None,
            max_requests: number(&["concurrency", "max-sequences"]),
            max_context: number(&["max-context", "max-context-tokens"]),
            max_output: number(&["max-output", "max-output-tokens"]),
            host_cache_bytes: number(&["host-cache-bytes"]),
        }
    }

    /// `KV 412K tok · int8 full · 16 req max · 128K ctx · host cache 64 GiB`.
    pub fn line(&self) -> String {
        let mut parts = Vec::new();
        if let Some(tokens) = self.kv_tokens.filter(|&tokens| tokens > 0) {
            let mut kv = format!("KV {} tok", short_tokens(tokens));
            if let Some(max) = self.max_tokens.filter(|&max| max > tokens) {
                kv.push_str(&format!(" (max {})", short_tokens(max)));
            }
            parts.push(kv);
        }
        if let Some(format) = &self.kv_format {
            parts.push(format.clone());
        }
        if let Some(n) = self.max_requests {
            parts.push(format!("{n} req max"));
        }
        if let Some(n) = self.max_context {
            parts.push(format!("{} ctx", short_tokens(n)));
        }
        if let Some(n) = self.max_output {
            parts.push(format!("{} out", short_tokens(n)));
        }
        if let Some(bytes) = self.host_cache_bytes {
            parts.push(format!("host cache {}", bytes_label(bytes)));
        }
        parts.join(" · ")
    }

    /// `412K tok / 16 req` for the README table.
    pub fn compact(&self) -> String {
        match (self.kv_tokens, self.max_requests) {
            (Some(t), Some(r)) => format!("{} tok / {r} req", short_tokens(t)),
            (Some(t), None) => format!("{} tok", short_tokens(t)),
            (None, Some(r)) => format!("{r} req"),
            _ => "—".into(),
        }
    }
}

impl Report {
    /// Resolved onboard count belongs beside capacity, not among tuning chips.
    /// Hardware-only cards have no onboarding trade-off, including dense models.
    pub fn capacity_line(&self) -> String {
        let mut capacity = self.capacity();
        let placement = if self.server.hardware.sparks.is_empty() {
            capacity.max_tokens = None;
            Some("all RTX".to_string())
        } else {
            self.server.configuration.settings.iter()
                .find(|s| s.name == "rtx-expert-layers" && s.source == "resolved")
                .and_then(|s| s.value.as_ref()?.parse::<usize>().ok())
                .map(|count| format!("{count} RTX expert layers"))
        };
        let line = capacity.line();
        let Some(placement) = placement else { return line };
        let mut parts: Vec<_> = line.split(" · ").filter(|part| !part.is_empty()).map(str::to_owned).collect();
        // Keep placement visible even if a verbose KV format needs truncating.
        parts.insert(usize::from(capacity.kv_tokens.is_some_and(|tokens| tokens > 0)), placement);
        parts.join(" · ")
    }

    /// The capacity of the baseline, else what the options say.
    pub fn capacity(&self) -> Capacity {
        self.baseline.as_ref().and_then(|b| b.card.capacity.clone())
            .unwrap_or_else(|| Capacity::from_settings(&self.server))
    }
}

impl BasicCard {
    pub fn concurrent_at(&self, width: usize) -> Option<&ConcurrentRate> {
        self.concurrency.iter().find(|point| point.width == width)
    }

    pub fn decode_of(&self, content: &str) -> Option<&ContentRate> {
        self.decode.iter().find(|d| d.content == content)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentRate {
    pub content: String,
    /// Median over runs of decode tokens per second after the first token.
    pub tok_s: f64,
    #[serde(default)]
    pub runs: Vec<StreamTiming>,
    /// Draft acceptance reported by the server for these runs, when it speculates.
    #[serde(default)]
    pub acceptance: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcurrentRate {
    pub width: usize,
    pub aggregate_tok_s: f64,
    pub per_stream_median_tok_s: f64,
    /// First emitted token to last emitted token across all streams.
    pub decode_s: f64,
    pub runs: Vec<ConcurrentTiming>,
    /// Wall seconds of the untimed batch at the same width.
    pub warmup_s: f64,
    /// The prompts behind the point: `code` (the C1 code decode) or a versioned
    /// distinct set (`distinct-v1`: each stream its own style and topic).
    /// Absent on earlier reports, whose waves sent the code prompt to every
    /// stream; those share experts and drafts and are not comparable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<String>,
}

impl ConcurrentRate {
    /// How the point's streams were prompted, for captions and tables.
    pub fn prompt_label(&self) -> &str {
        match self.prompts.as_deref() {
            None => "one code prompt",
            Some("code") => "code",
            Some(_) => "distinct prompts",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcurrentTiming {
    /// Send offset on the batch's clock, needed to reproduce the aggregate rate.
    pub sent_s: f64,
    pub timing: StreamTiming,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefillRate {
    pub prompt_tokens: u64,
    /// Median prompt tokens per second of time to first token.
    pub tok_s: f64,
    pub ttft_s: f64,
    #[serde(default)]
    pub runs: Vec<StreamTiming>,
}

/// One streamed request as the client saw it.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StreamTiming {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// Seconds from send to the first generated token.
    pub ttft_s: f64,
    /// Seconds from send to the last byte.
    pub total_s: f64,
    /// Seconds from the first to the last generated token.
    pub decode_s: f64,
    /// Seconds from send to the end of the reasoning section, when there was one.
    #[serde(default)]
    pub reasoning_end_s: Option<f64>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

impl StreamTiming {
    /// Decode tokens per second after the first token.
    pub fn decode_tok_s(&self) -> f64 {
        let tokens = self.completion_tokens.saturating_sub(1) as f64;
        if self.decode_s > 0.0 && tokens > 0.0 { tokens / self.decode_s } else { 0.0 }
    }

    /// Uncached prompt tokens per second of time to first token.
    pub fn prefill_tok_s(&self) -> f64 {
        let tokens = self.prompt_tokens.saturating_sub(self.cached_tokens) as f64;
        if self.ttft_s > 0.0 { tokens / self.ttft_s } else { 0.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Fail,
    /// Measured, not gated.
    Info,
    /// Not applicable to this server (no speculator, no reference file).
    Skipped,
    /// The server lacks the hook the check needs.
    Unsupported,
    #[default]
    Pending,
}

impl CheckStatus {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Pass => "✓",
            Self::Fail => "✗",
            Self::Info => "i",
            Self::Skipped | Self::Unsupported => "–",
            Self::Pending => "…",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Quality {
    /// Fail if any gated check failed; Pass if every gated check that ran passed.
    pub status: CheckStatus,
    pub checks: Vec<Check>,
}

impl Quality {
    pub fn check(&self, id: &str) -> Option<&Check> {
        self.checks.iter().find(|c| c.id == id)
    }

    /// Recompute `status` from the checks.
    pub fn settle(&mut self) {
        let gated = || self.checks.iter().filter(|c| !matches!(c.status, CheckStatus::Info));
        self.status = if gated().any(|c| c.status == CheckStatus::Fail) {
            CheckStatus::Fail
        } else if gated().any(|c| c.status == CheckStatus::Pending) {
            CheckStatus::Pending
        } else if gated().any(|c| c.status == CheckStatus::Pass) {
            CheckStatus::Pass
        } else {
            CheckStatus::Skipped
        };
    }

    /// The share-card badge, e.g. `KL 0.058 · top-1 89.9% ✓ exact cache`. An
    /// informational fidelity keeps its numbers with the reason it is not
    /// gated: `KL 0.039 · top-1 93.7% (info: calibrated W4A4) ✓ exact cache`.
    pub fn badge(&self) -> String {
        let mut parts = Vec::new();
        if let Some(fidelity) = self.check("fidelity")
            .filter(|c| matches!(c.status, CheckStatus::Pass | CheckStatus::Fail | CheckStatus::Info)) {
            if let (Some(kl), Some(top1)) = (fidelity.metric("kl"), fidelity.metric("top1")) {
                let mark = if fidelity.status == CheckStatus::Fail { "✗ " } else { "" };
                let note = match (&fidelity.note, fidelity.status) {
                    (Some(note), CheckStatus::Info) => format!(" (info: {note})"),
                    (None, CheckStatus::Info) => " (info)".into(),
                    _ => String::new(),
                };
                parts.push(format!("{mark}KL {kl:.3} · top-1 {:.1}%{note}", top1 * 100.0));
            }
        }
        let mut tail = Vec::new();
        if let Some(cache) = self.check("cache_exact") {
            match cache.status {
                CheckStatus::Pass => tail.push("✓ exact cache".to_string()),
                CheckStatus::Fail => tail.push("✗ cache mismatch".to_string()),
                _ => {}
            }
        }
        if let Some(spec) = self.check("spec_lossless") {
            match spec.status {
                CheckStatus::Pass => tail.push("✓ lossless spec".to_string()),
                CheckStatus::Fail => tail.push("✗ spec diverges".to_string()),
                _ => {}
            }
        }
        let mut badge = parts.join(" · ");
        if !tail.is_empty() {
            if !badge.is_empty() {
                badge.push(' ');
            }
            badge.push_str(&tail.join(" "));
        }
        if badge.is_empty() {
            badge = match self.status {
                CheckStatus::Pass => "quality ✓".into(),
                CheckStatus::Fail => "QUALITY GATE FAILED".into(),
                _ => "quality not verified".into(),
            };
        }
        badge
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Check {
    pub id: String,
    pub title: String,
    pub status: CheckStatus,
    /// One line: what was found.
    pub summary: String,
    /// Why an informational check is not gated (`calibrated W4A4`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Named numbers (kl, top1, nll, ...).
    #[serde(default)]
    pub metrics: serde_json::Map<String, Value>,
    #[serde(default)]
    pub seconds: f64,
}

impl Check {
    pub fn new(id: &str, title: &str) -> Self {
        Self { id: id.into(), title: title.into(), ..Self::default() }
    }

    pub fn metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).and_then(Value::as_f64)
    }

    pub fn set(&mut self, name: &str, value: impl Into<Value>) {
        self.metrics.insert(name.into(), value.into());
    }

    /// The summary and, when the check is informational, the reason it is not
    /// gated: `KL 0.039 · top-1 93.7% (info: calibrated W4A4)`.
    pub fn detail(&self) -> String {
        match (&self.note, self.status) {
            (Some(note), CheckStatus::Info) => format!("{} (info: {note})", self.summary),
            _ => self.summary.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PanelStatus {
    #[default]
    Pending,
    Running,
    Done,
    Failed,
    Cancelled,
    /// Not runnable on this server (e.g. a tool missing from the image).
    Unsupported,
}

/// One panel's results over its passes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PanelResult {
    pub id: String,
    pub title: String,
    pub status: PanelStatus,
    /// Each pass's panel-specific record.
    #[serde(default)]
    pub passes: Vec<Value>,
    #[serde(default)]
    pub started: Option<String>,
    #[serde(default)]
    pub finished: Option<String>,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub error: Option<String>,
    /// Earlier runs on the same fingerprint the chart aggregates (tool eval).
    #[serde(default)]
    pub history: Vec<Value>,
    /// The running pass's record so far (charts fill progressively).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial: Option<Value>,
    /// In an accumulated view: the run this result came from, and when.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_created: Option<String>,
}

impl PanelResult {
    /// The newest record: the running pass's partial one, else the last pass.
    pub fn latest(&self) -> Option<&Value> {
        self.partial.as_ref().or_else(|| self.passes.last())
    }
}

/// RFC 3339 UTC with seconds, from the system clock.
pub fn now_rfc3339() -> String {
    rfc3339(std::time::SystemTime::now())
}

pub fn rfc3339(time: std::time::SystemTime) -> String {
    let secs = time.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Howard Hinnant's days-from-civil inverse.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn configuration_preserves_resolved_mapped_placement() {
        use cuteafd_loader::formats::mapped_preload::MappedPlacement;
        let mut configuration = Configuration::default();
        let old = serde_json::to_value(&configuration).unwrap();
        assert!(old.get("mapped_placement").is_none());
        assert_eq!(serde_json::from_value::<Configuration>(old).unwrap(), configuration);
        for cap in [0, 64] {
            let placement = MappedPlacement::resolve(4, 2, 64, cap, 128);
            configuration.layout = Some(placement.label());
            configuration.mapped_placement = Some(placement.clone());
            let encoded = serde_json::to_value(&configuration).unwrap();
            assert_eq!(encoded["mapped_placement"]["preloaded_bytes"], placement.preloaded_bytes);
            assert_eq!(serde_json::from_value::<Configuration>(encoded).unwrap(), configuration);
        }
    }

    #[test]
    fn rfc3339_formats_known_instants() {
        let at = |s: u64| rfc3339(std::time::UNIX_EPOCH + std::time::Duration::from_secs(s));
        assert_eq!(at(0), "1970-01-01T00:00:00Z");
        assert_eq!(at(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(at(1_790_000_000), "2026-09-21T14:13:20Z");
    }

    #[test]
    fn hardware_slug_and_class() {
        let gpu = |used| Gpu { name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition".into(), used,
            power_limit_w: Some(325.0), ..Gpu::default() };
        let spark = |rank| Spark { address: format!("10.55.0.{}", rank + 1), rank, name: None };
        let one = Hardware { gpus: vec![gpu(true), gpu(false)], sparks: (0..4).map(spark).collect(), ..Hardware::default() };
        assert_eq!(one.slug(), "1rtx-4spark");
        assert_eq!(one.class(), HardwareClass::Minimum);
        assert_eq!(one.line(), "1× RTX PRO 6000 @ 325 W + 4× DGX Spark");
        let two = Hardware { gpus: vec![gpu(true), gpu(true)], sparks: (0..6).map(spark).collect(), ..Hardware::default() };
        assert_eq!(two.slug(), "2rtx-6spark");
        assert_eq!(two.class(), HardwareClass::Maximum);
        let local = Hardware { gpus: vec![gpu(true)], ..Hardware::default() };
        assert_eq!(local.slug(), "1rtx");
    }

    fn window(seconds: f64, gov: bool, gpus: &[(u32, f64, f64, f64, &str)]) -> Value {
        let gpus: Vec<Value> = gpus.iter().map(|&(i, mean, lo, hi, bound)| json!({
            "index": i, "governor_bound": if gov { json!(bound) } else { Value::Null },
            "power_w": {"mean": mean - 20.0},
            "power_limit_w": {"mean": mean, "min": lo, "max": hi}})).collect();
        let mut w = json!({"from_s": 0.0, "to_s": seconds, "seconds": seconds, "gpus": gpus});
        if gov { w["governor"] = json!({"budget_w": 900.0}); w["governor_budget_w"] = json!(900.0); }
        w
    }

    fn labelled(windows: &[Value], gpus: usize) -> String {
        let mut hw = Hardware { gpus: (0..gpus).map(|i| Gpu { index: i as u32, name: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition".into(),
            used: true, power_limit_w: Some(325.0), ..Gpu::default() }).collect(), ..Hardware::default() };
        hw.power = Power::from_json(&windows.iter().collect::<Vec<_>>());
        hw.line()
    }

    #[test]
    fn power_cap_preserves_short_throttled_windows_without_inventing_caps() {
        let mut cap = window(1.0, true, &[(0, 325.0, 325.0, 325.0, "budget")]);
        cap["gpus"][0]["throttle_s"] = json!({"sw_power_cap": 0.25, "hw_power_brake": 0.0});
        let clean = window(90.0, true, &[(0, 325.0, 325.0, 325.0, "budget")]);
        assert!(!Power::from_json(&[&clean]).unwrap().capped());
        let power = Power::from_json(&[&clean, &cap]).unwrap();
        assert!(power.capped());
        assert_eq!(power.gpus[0].cap_seconds, 0.25);
    }

    #[test]
    fn power_label_formats() {
        // No governor: the plain form with the mean limit.
        let w = window(10.0, false, &[(0, 325.0, 325.0, 325.0, ""), (1, 325.0, 325.0, 325.0, "")]);
        assert_eq!(labelled(&[w], 2), "2× RTX PRO 6000 @ 325 W");
        // gpu-powerd at the ceiling.
        let w = window(10.0, true, &[(0, 450.0, 450.0, 450.0, "budget"), (1, 450.0, 450.0, 450.0, "budget")]);
        assert_eq!(labelled(&[w], 2), "2× RTX PRO 6000 @ gpu-powerd 450/450 W");
        // One card throttled: the limit moved more than 10 W.
        let w = window(10.0, true, &[(0, 522.0, 380.0, 600.0, "model"), (1, 408.0, 405.0, 410.0, "budget")]);
        assert_eq!(labelled(&[w], 2), "2× RTX PRO 6000 @ gpu-powerd 520/410 W (380–600)");
        // A single card, two windows weighted by seconds.
        let a = window(30.0, true, &[(0, 600.0, 600.0, 600.0, "max")]);
        let b = window(10.0, true, &[(0, 600.0, 600.0, 600.0, "max")]);
        assert_eq!(labelled(&[a, b], 1), "1× RTX PRO 6000 @ gpu-powerd 600 W");
    }

    #[test]
    fn power_merge_weights_windows_and_old_reports_render_plain() {
        let a = window(30.0, true, &[(0, 600.0, 600.0, 600.0, "max")]);
        let b = window(10.0, true, &[(0, 400.0, 400.0, 400.0, "budget")]);
        let p = Power::from_json(&[&json!({"x": [a, b]})]).unwrap();
        assert_eq!(p.gpus[0].mean_limit_w, 550.0);
        assert_eq!(p.gpus[0].bound.as_deref(), Some("max"));
        assert_eq!(p.budget_w, Some(900.0));
        let old: Hardware = serde_json::from_value(json!({"gpus": [{"index": 0, "name": "RTX PRO 6000",
            "power_limit_w": 325.0, "used": true}]})).unwrap();
        assert_eq!(old.line(), "1× RTX PRO 6000 @ 325 W");
    }

    #[test]
    fn quality_settles_and_badges() {
        let mut quality = Quality::default();
        let mut fidelity = Check::new("fidelity", "Logit fidelity");
        fidelity.status = CheckStatus::Pass;
        fidelity.set("kl", 0.0581);
        fidelity.set("top1", 0.899);
        let mut cache = Check::new("cache_exact", "Prefix-cache restore");
        cache.status = CheckStatus::Pass;
        let mut c4 = Check::new("c1_c4", "C1 vs C4");
        c4.status = CheckStatus::Info;
        quality.checks = vec![fidelity, cache, c4];
        quality.settle();
        assert_eq!(quality.status, CheckStatus::Pass);
        assert_eq!(quality.badge(), "KL 0.058 · top-1 89.9% ✓ exact cache");
        quality.checks[1].status = CheckStatus::Fail;
        quality.settle();
        assert_eq!(quality.status, CheckStatus::Fail);
    }
}
