//! Benchmark panels: each runs one pass against the server through the
//! loopback client and returns its record (a JSON value the renderers read).
use crate::client::Client;
use crate::report::{Baseline, ServerInfo};
use serde::Serialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

pub mod agentic;
pub mod common;
pub mod fidelity;
pub mod info;
pub mod names;
pub mod quality;
pub mod reasoning;
pub mod speed;
pub mod tools;

/// Rates that turn a panel's workload into seconds: the baseline's when it
/// ran, a conservative guess before.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Rates {
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub measured: bool,
}

impl Default for Rates {
    fn default() -> Self {
        Self { decode_tok_s: 30.0, prefill_tok_s: 1500.0, measured: false }
    }
}

impl Rates {
    pub fn from_baseline(baseline: &Baseline) -> Self {
        let decode = baseline.card.decode_of("code").map(|d| d.tok_s).filter(|v| *v > 0.0);
        let prefill = baseline.card.prefill.as_ref().map(|p| p.tok_s).filter(|v| *v > 0.0);
        let fallback = Self::default();
        Self { decode_tok_s: decode.unwrap_or(fallback.decode_tok_s),
            prefill_tok_s: prefill.unwrap_or(fallback.prefill_tok_s), measured: decode.is_some() }
    }

    /// Seconds to prefill `prompt` tokens and decode `output` tokens, plus a
    /// fixed per-request overhead.
    pub fn seconds(&self, prompt: f64, output: f64) -> f64 {
        0.15 + prompt / self.prefill_tok_s + output / self.decode_tok_s
    }
}

/// Where a running pass reports how far it is and what it has so far.
#[derive(Clone, Default)]
pub struct Progress {
    state: Arc<Mutex<ProgressState>>,
}

#[derive(Debug, Clone, Default)]
pub struct ProgressState {
    pub fraction: f64,
    pub label: String,
    /// The pass's partial record (charts fill progressively).
    pub partial: Option<Value>,
    pub revision: u64,
}

impl Progress {
    pub fn step(&self, fraction: f64, label: impl Into<String>) {
        if let Ok(mut s) = self.state.lock() {
            s.fraction = fraction.clamp(0.0, 1.0);
            s.label = label.into();
            s.revision += 1;
        }
    }

    pub fn partial(&self, value: Value) {
        if let Ok(mut s) = self.state.lock() {
            s.partial = Some(value);
            s.revision += 1;
        }
    }

    pub fn get(&self) -> ProgressState {
        self.state.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn reset(&self) {
        if let Ok(mut s) = self.state.lock() {
            *s = ProgressState { revision: s.revision + 1, ..ProgressState::default() };
        }
    }
}

/// What a pass gets to work with.
pub struct Ctx<'a> {
    pub client: &'a Client,
    pub info: &'a ServerInfo,
    pub baseline: Option<&'a Baseline>,
    pub rates: Rates,
    pub progress: &'a Progress,
    /// 1-based pass number of this panel in the run.
    pub pass: u32,
    /// Earlier passes on the same fingerprint (panels that accumulate).
    pub history: &'a [Value],
    /// The server's request limits (`/v1/models`).
    pub max_context: u64,
    pub max_output: u64,
}

pub trait Panel: Send + Sync {
    fn id(&self) -> &'static str;
    fn title(&self) -> &'static str;
    fn description(&self) -> &'static str;
    /// Expected seconds for one pass.
    fn estimate_s(&self, rates: &Rates, info: &ServerInfo) -> f64;
    /// Why this server cannot run the panel, if it cannot.
    fn unavailable(&self, _info: &ServerInfo) -> Option<String> {
        None
    }
    fn run(&self, ctx: &Ctx<'_>) -> anyhow::Result<Value>;
}

/// Every panel this build runs, in display order.
pub fn catalog() -> Vec<&'static dyn Panel> {
    vec![&info::HARDWARE, &info::CONFIGURATION, &speed::DECODE_CONTENT, &speed::CONCURRENCY, &speed::DRAFT_MIX, &speed::PREFILL,
        &speed::RETAINED, &speed::PREFIX_CACHE, &agentic::AGENTIC, &quality::STRUCTURED, &quality::NEEDLE, &quality::IFEVAL,
        &fidelity::STANDARD, &fidelity::FULL, &quality::CODE, &quality::MATH, &reasoning::REASONING, &tools::TOOL_EVAL, &info::STARTUP]
}

pub fn find(id: &str) -> Option<&'static dyn Panel> {
    catalog().into_iter().find(|p| p.id() == id)
}

/// Panels always shown in a report (and exports) whether or not a profile ticks them.
pub const ALWAYS: [&str; 2] = ["hardware", "configuration"];
