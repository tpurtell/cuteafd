//! The launch description of a checkpoint: its family and the decoder layers
//! whose routed experts the Spark ranks serve, derived from the same
//! canonical configuration readers the runtimes use.
//! `scripts/lib/checkpoint-family.py` prints the same line for the launch
//! scripts (which run without a cuteafd binary); both are checked against
//! `tests/fixtures/launch-families.json`.
use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::Value;

use crate::families::{glm5::GlmDsaConfig, glm5_flash::GlmNextConfig, mimo_v2::MimoV2Config, qwen4::Qwen4Config};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LaunchDescription {
    pub family: &'static str,
    pub model_type: String,
    /// First decoder layer with routed experts.
    pub first_expert_layer: usize,
    /// Last decoder layer the Spark ranks serve; `None` through the end
    /// (MTP/nextn layers included).
    pub last_expert_layer: Option<usize>,
}

impl LaunchDescription {
    /// `FAMILY MODEL_TYPE FIRST LAST` (LAST -1: through the end), as
    /// checkpoint-family.py prints it.
    pub fn line(&self) -> String {
        let last = self.last_expert_layer.map_or("-1".to_string(), |l| l.to_string());
        format!("{} {} {} {last}", self.family, self.model_type, self.first_expert_layer)
    }
}

/// The launch description of `config.json`.
pub fn describe(config: &Value) -> Result<LaunchDescription> {
    let model_type = config.get("model_type").and_then(Value::as_str).unwrap_or("?").to_owned();
    let (family, first, last) = match model_type.as_str() {
        "deepseek_v41" => ("deepseek_v41", 0, None),
        "deepseek_v4" => ("deepseek_v4", 0, None),
        "glm_moe_dsa" => ("glm5", GlmDsaConfig::from_hf(config)?.first_moe_layer, None),
        "glm5_next" => {
            let cfg = GlmNextConfig::from_hf(config)?;
            let first = cfg.dense.iter().position(|dense| !dense).unwrap_or(cfg.layers);
            // serve-glmf does not run the MTP layer: its experts stay off the Sparks.
            ("glm5_flash", first, Some(cfg.layers - 1))
        }
        "mimo_v2_flash" | "mimo_v2" => {
            let cfg = MimoV2Config::from_hf(config)?;
            ("mimo_v2", cfg.dense.iter().position(|dense| !dense).unwrap_or(cfg.layers), None)
        }
        "qwen4_exp" => {
            let cfg = Qwen4Config::from_hf(config)?;
            ("qwen4", 0, Some(cfg.layers - 1))
        }
        "qwen3_5" => {
            let cfg = crate::families::qwen3_5::Qwen35Config::from_hf(config)?;
            ("qwen3_5", cfg.attention.len(), Some(cfg.attention.len() - 1))
        }
        "step5_vl" | "step5" => {
            let (first, last) = step5_routed(text_config(config))?;
            ("step5", first, Some(last))
        }
        "kimi_k3" => {
            let (first, last) = kimi_routed(text_config(config))?;
            ("kimi_k3", first, Some(last))
        }
        other => bail!("unsupported model_type {other:?}"),
    };
    Ok(LaunchDescription { family, model_type, first_expert_layer: first, last_expert_layer: last })
}

fn text_config(config: &Value) -> &Value {
    config.get("text_config").unwrap_or(config)
}

/// Step-5's routed layers: exactly 3..=90 of its 92 decoder layers, from
/// `moe_layer_list` (the full reader is `plan::families::step5::Config`).
fn step5_routed(text: &Value) -> Result<(usize, usize)> {
    let routed = text.get("moe_layer_list").and_then(Value::as_array)
        .map(|layers| layers.iter().map(Value::as_u64).collect::<Option<Vec<_>>>());
    let mut routed = match (text.get("model_type").and_then(Value::as_str), text.get("num_hidden_layers").and_then(Value::as_u64), routed) {
        (Some("step5"), Some(92), Some(Some(routed))) => routed,
        _ => bail!("unsupported Step-5 routed layer map"),
    };
    routed.sort_unstable();
    if routed != (3..91).collect::<Vec<u64>>() {
        bail!("unsupported Step-5 routed layer map");
    }
    Ok((3, 90))
}

/// Kimi K3's routed layers: a dense prefix (`first_k_dense_replace`), then every
/// layer routed (`moe_layer_freq` 1), through the last decoder layer.
fn kimi_routed(text: &Value) -> Result<(usize, usize)> {
    let layers = match text.get("num_hidden_layers").and_then(Value::as_u64) {
        Some(layers) if layers > 0 => layers as usize,
        _ => bail!("config lacks num_hidden_layers"),
    };
    let first = match text.get("first_k_dense_replace") {
        None => 0,
        Some(value) => match value.as_u64() {
            Some(first) if value.is_u64() && (first as usize) < layers => first as usize,
            _ => bail!("unsupported kimi_k3 MoE layer pattern"),
        },
    };
    if text.get("moe_layer_freq").is_some_and(|freq| freq.as_u64() != Some(1) || !freq.is_u64()) {
        bail!("unsupported kimi_k3 MoE layer pattern");
    }
    Ok((first, layers - 1))
}
