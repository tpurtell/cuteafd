//! Fidelity bars follow routed-expert precision, not dense/head storage.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unknown routed-expert quantization format: {0}")]
    UnknownFormat(String),
    #[error("invalid routed-expert precision metadata: {0}")]
    Invalid(String),
    #[error("cannot read checkpoint precision metadata: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid checkpoint precision JSON: {0}")]
    Json(#[from] serde_json::Error),
}
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Tier {
    Strict,
    Mid,
    Low,
}
impl Tier {
    pub fn bars(self) -> (f64, f64) {
        match self {
            Self::Strict => (0.95, 0.03),
            Self::Mid => (0.93, 0.05),
            Self::Low => (0.90, 0.07),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Mid => "3 to <6 bpw",
            Self::Low => "<3 bpw",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Precision {
    pub expert_bpw: f64,
    pub format: String,
    pub tier: Tier,
    pub tier_top1_min: f64,
    pub tier_kl_max: f64,
    pub reference_root_format: bool,
    pub source: String,
}
impl Precision {
    pub fn new(bpw: f64, format: &str, reference_root_format: bool, source: &str) -> Result<Self> {
        if !bpw.is_finite() || bpw <= 0.0 || bpw > 64.0 {
            return Err(Error::Invalid(format!("invalid bpw {bpw}")));
        }
        let tier = if reference_root_format || bpw >= 6.0 {
            Tier::Strict
        } else if bpw >= 3.0 {
            Tier::Mid
        } else {
            Tier::Low
        };
        let (tier_top1_min, tier_kl_max) = tier.bars();
        Ok(Self {
            expert_bpw: bpw,
            format: format.into(),
            tier,
            tier_top1_min,
            tier_kl_max,
            reference_root_format,
            source: source.into(),
        })
    }
    pub fn validate(&self) -> Result<()> {
        let expected = Self::new(
            self.expert_bpw,
            &self.format,
            self.reference_root_format,
            &self.source,
        )?;
        if self.tier != expected.tier
            || self.tier_top1_min != expected.tier_top1_min
            || self.tier_kl_max != expected.tier_kl_max
        {
            return Err(Error::Invalid(
                "tier disagrees with bpw/reference-root identity".into(),
            ));
        }
        Ok(())
    }
    pub fn label(&self) -> String {
        let bpw = if self.expert_bpw.fract() == 0.0 {
            format!("{:.1}", self.expert_bpw)
        } else {
            format!("{:.4}", self.expert_bpw)
                .trim_end_matches('0')
                .to_owned()
        };
        if self.reference_root_format {
            format!("{bpw} bpw, reference-root format -> strict tier")
        } else {
            format!("tier: {} experts ({bpw} bpw)", self.tier.name())
        }
    }
}

fn number(v: &Value) -> Result<f64> {
    if let Some(n) = v.as_f64() {
        return Ok(n);
    }
    if let Some(s) = v.as_str() {
        let parse = |s: &str| {
            s.parse::<f64>()
                .map_err(|_| Error::Invalid(format!("invalid bpw {v}")))
        };
        return if let Some((a, b)) = s.split_once('/') {
            Ok(parse(a)? / parse(b)?)
        } else {
            parse(s)
        };
    }
    Err(Error::Invalid(format!("invalid bpw {v}")))
}

fn target_bpw(v: &Value, values: &mut Vec<f64>) -> Result<()> {
    match v {
        Value::Object(map) => {
            for (key, value) in map {
                // Draft/head precision must not choose the target routed-expert tier.
                if matches!(
                    key.as_str(),
                    "mtp" | "draft" | "head" | "non_routed_quantization"
                ) {
                    continue;
                }
                if key == "target_bpw" {
                    values.push(number(value)?);
                } else {
                    target_bpw(value, values)?;
                }
            }
        }
        Value::Array(a) => {
            for value in a {
                target_bpw(value, values)?;
            }
        }
        _ => (),
    }
    Ok(())
}

fn config_precision(config: &Value) -> Result<Precision> {
    let q = config
        .get("text_config")
        .and_then(|t| t.get("quantization_config"))
        .filter(|q| !q.is_null())
        .or_else(|| config.get("quantization_config").filter(|q| !q.is_null()));
    let Some(q) = q else {
        return Precision::new(16.0, "bf16", false, "config.json: unquantized");
    };
    let method = q["quant_method"]
        .as_str()
        .or_else(|| q["format"].as_str())
        .or_else(|| q["method"].as_str())
        .unwrap_or(if q["quant_algo"] == "NVFP4" {
            "modelopt"
        } else {
            "unspecified"
        })
        .to_ascii_lowercase();
    if let Some(algo) = q["quant_algo"].as_str() {
        if !matches!(algo, "NVFP4" | "FP8" | "MIXED_PRECISION") {
            return Err(Error::UnknownFormat(format!("{method}: quant_algo {algo}")));
        }
    }
    if let Some(algo) = q["moe_quant_algo"].as_str() {
        if !matches!(algo, "NVFP4" | "FP8") {
            return Err(Error::UnknownFormat(format!(
                "{method}: moe_quant_algo {algo}"
            )));
        }
    }
    for key in ["store_dtype", "expert_dtype"] {
        if let Some(dtype) = q[key].as_str() {
            if !matches!(
                dtype,
                "mxfp4" | "fp4" | "fp8" | "mxfp8" | "float8_e4m3fn" | "float8_e4m3fnuz"
            ) {
                return Err(Error::UnknownFormat(format!("{method}: {key} {dtype}")));
            }
        }
    }
    let (bpw, format) = match method.as_str() {
        "exl3" => {
            let mut targets = Vec::new();
            target_bpw(&q["meta"], &mut targets)?;
            let bpw = targets
                .first()
                .copied()
                .map_or_else(|| number(&q["bits"]), Ok)?;
            if targets
                .iter()
                .any(|v| !v.is_finite() || (v - bpw).abs() > 1e-9)
            {
                return Err(Error::Invalid("conflicting EXL3 target_bpw values".into()));
            }
            (bpw, "exl3")
        }
        "fp8" | "mxfp8" | "mxfp4" | "modelopt" | "compressed-tensors" => {
            if q["moe_quant_algo"].as_str() == Some("NVFP4")
                || q["quant_algo"].as_str() == Some("NVFP4")
            {
                (4.0, "nvfp4")
            } else if let Some(groups) = q["config_groups"].as_object() {
                let routed = |g: &&Value| {
                    g["targets"].as_array().is_some_and(|a| {
                        a.iter()
                            .any(|t| t.as_str().is_some_and(|s| s.contains("experts")))
                    })
                };
                let mut selected: Vec<_> = groups.values().filter(routed).collect();
                if selected.is_empty() {
                    selected = groups.values().collect();
                }
                // MTP groups are auxiliary; explicit target groups take precedence.
                let layers = config["text_config"]["num_hidden_layers"]
                    .as_u64()
                    .or_else(|| config["num_hidden_layers"].as_u64());
                selected.retain(|g| {
                    !g["targets"].as_array().is_some_and(|a| {
                        !a.is_empty()
                            && a.iter().all(|t| {
                                t.as_str().is_some_and(|s| {
                                    s.contains("mtp")
                                        || layers
                                            .is_some_and(|l| s.contains(&format!("layers.{l}.")))
                                })
                            })
                    })
                });
                let widths = selected
                    .iter()
                    .map(|g| {
                        if g["weights"]["type"].as_str() != Some("float") {
                            return Err(Error::UnknownFormat(format!(
                                "{method}: {}",
                                g["weights"]
                            )));
                        }
                        number(&g["weights"]["num_bits"])
                    })
                    .collect::<Result<Vec<_>>>()?;
                if widths.iter().any(|w| !matches!(w, 4.0 | 8.0)) {
                    return Err(Error::UnknownFormat(format!(
                        "{method}: weight widths {widths:?}"
                    )));
                }
                let width = widths
                    .first()
                    .copied()
                    .ok_or_else(|| Error::Invalid("empty quantization groups".into()))?;
                if widths.iter().any(|w| *w != width) {
                    return Err(Error::UnknownFormat(format!(
                        "{method}: mixed routed weight widths {widths:?}"
                    )));
                }
                match width {
                    4.0 => (4.0, "nvfp4"),
                    8.0 => (8.0, "fp8"),
                    _ => return Err(Error::UnknownFormat(format!("{method}: {width}-bit"))),
                }
            } else if method == "mxfp4"
                || q["store_dtype"].as_str() == Some("mxfp4")
                || q["expert_dtype"].as_str() == Some("fp4")
            {
                (4.0, "mxfp4")
            } else if matches!(method.as_str(), "fp8" | "mxfp8") {
                (8.0, "fp8")
            } else {
                return Err(Error::UnknownFormat(format!(
                    "{method}: {}",
                    q["quant_algo"]
                )));
            }
        }
        _ => return Err(Error::UnknownFormat(method)),
    };
    Precision::new(
        bpw,
        format,
        false,
        "config.json: quantization_config routed experts",
    )
}

fn manifest_precision(snapshot: &Path, config: &Value, name: &str) -> Result<Precision> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-_".contains(&b))
    {
        return Err(Error::Invalid(
            "EXLR8 manifest must be a published name".into(),
        ));
    }
    let filename = format!("manifests/exlr8-manifest-{name}.json");
    let m: Value = serde_json::from_slice(&std::fs::read(snapshot.join(&filename))?)?;
    if m["format"] != "exlr8-v2-manifest" || m["name"] != name {
        return Err(Error::UnknownFormat(format!(
            "EXLR8 manifest {} / {}",
            m["format"], m["name"]
        )));
    }
    let layers = config["num_hidden_layers"]
        .as_u64()
        .ok_or_else(|| Error::Invalid("EXLR8 num_hidden_layers missing".into()))?;
    let mut sum = 0.0;
    let mut count = 0;
    let entries = m["layers"]
        .as_object()
        .ok_or_else(|| Error::Invalid("EXLR8 layers missing".into()))?;
    for (layer, values) in entries {
        let layer: u64 = layer
            .parse()
            .map_err(|_| Error::Invalid("EXLR8 layer id".into()))?;
        if layer >= layers {
            continue;
        }
        for projection in ["gate", "up", "down"] {
            let rows = values[projection]
                .as_array()
                .ok_or_else(|| Error::Invalid(format!("EXLR8 {projection} missing")))?;
            for row in rows {
                let widths = if projection == "down" {
                    vec![row]
                } else {
                    row.as_array()
                        .ok_or_else(|| Error::Invalid("EXLR8 unit widths missing".into()))?
                        .iter()
                        .collect()
                };
                // Each gate/up unit spans the same weights as one down unit.
                let weight = if projection == "down" {
                    values["gate"][0]
                        .as_array()
                        .map(Vec::len)
                        .filter(|n| *n > 0)
                        .ok_or_else(|| Error::Invalid("EXLR8 empty units".into()))?
                } else {
                    1
                };
                for width in widths {
                    let k = number(width)?;
                    if !matches!(k, 2.0 | 3.0 | 4.0) {
                        return Err(Error::UnknownFormat(format!("EXLR8 K{k}")));
                    }
                    sum += k * weight as f64;
                    count += weight;
                }
            }
        }
    }
    Precision::new(sum / count as f64, "exlr8", false, &filename)
}

/// Resolve the active serving manifest from launch settings, never a filename guess.
pub fn selected_manifest(settings: &Value) -> Option<&str> {
    settings
        .as_array()?
        .iter()
        .find(|s| s["name"] == "exlr8-manifest")?["value"]
        .as_str()
        .filter(|s| !s.is_empty())
}

pub fn resolve(
    model: &str,
    snapshot: Option<&Path>,
    settings: &Value,
    reference_root: &str,
    selection: Option<&crate::fidelity_match::Resolution>,
) -> Result<Precision> {
    let snapshot = crate::fidelity_match::served_snapshot_path(model, snapshot);
    resolve_in(
        model,
        snapshot.as_deref(),
        settings,
        reference_root,
        selection,
        &crate::fidelity_match::hub(),
    )
}

fn resolve_in(
    model: &str,
    snapshot: Option<&Path>,
    settings: &Value,
    reference_root: &str,
    selection: Option<&crate::fidelity_match::Resolution>,
    hub: &Path,
) -> Result<Precision> {
    let path = snapshot
        .map(Path::to_path_buf)
        .or_else(|| crate::fidelity_match::local_snapshot(hub, model))
        .ok_or_else(|| Error::Invalid(format!("no checkpoint config available for {model}")))?;
    let config: Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let mut p = if let Some(name) = selected_manifest(settings) {
        manifest_precision(&path, &config, name)?
    } else if path.join("manifests").is_dir() {
        return Err(Error::Invalid(
            "EXLR8 active exlr8-manifest setting missing".into(),
        ));
    } else {
        config_precision(&config)?
    };
    let identity = model == reference_root;
    let mapped = selection.is_some_and(|s| {
        s.reference_root == reference_root
            && s.resolved_chain.iter().any(|chain| {
                chain.first().is_some_and(|id| id == model)
                    && chain.last().is_some_and(|id| id == reference_root)
            })
    });
    let root_format = if !identity && mapped {
        crate::fidelity_match::local_snapshot(hub, reference_root)
            .and_then(|root| std::fs::read(root.join("config.json")).ok())
            .map(|bytes| -> Result<bool> {
                let root = config_precision(&serde_json::from_slice::<Value>(&bytes)?)?;
                Ok(root.format == p.format && (root.expert_bpw - p.expert_bpw).abs() < 1e-9)
            })
            .transpose()?
            .unwrap_or(false)
    } else {
        false
    };
    p.reference_root_format = identity || root_format;
    Precision::new(p.expert_bpw, &p.format, p.reference_root_format, &p.source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tier_boundaries_and_invalid_widths() {
        for (bpw, tier) in [
            (2.999, Tier::Low),
            (3.0, Tier::Mid),
            (5.999, Tier::Mid),
            (6.0, Tier::Strict),
            (16.0, Tier::Strict),
        ] {
            assert_eq!(
                Precision::new(bpw, "test", false, "test").unwrap().tier,
                tier
            );
        }
        for bpw in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(Precision::new(bpw, "test", false, "test").is_err());
        }
    }

    #[test]
    fn metadata_targets_routed_experts_and_rejects_unknown_formats() {
        let p = config_precision(
            &json!({"text_config":{"quantization_config":{"quant_method":"exl3","bits":3,
            "meta":{"allocation":{"target_bpw":"13/4"},"mtp":{"target_bpw":4}}}}}),
        )
        .unwrap();
        assert_eq!(p.expert_bpw, 3.25);
        let p = config_precision(
            &json!({"quantization_config":{"quant_method":"modelopt", "config_groups":{
            "dense":{"targets":["lm_head"],"weights":{"type":"float","num_bits":8}},
            "routed":{"targets":["mlp.experts"],"weights":{"type":"float","num_bits":4}}}}}),
        )
        .unwrap();
        assert_eq!(p.expert_bpw, 4.0);
        assert!(
            matches!(config_precision(&json!({"quantization_config":{"quant_method":"mystery"}})), Err(Error::UnknownFormat(s)) if s == "mystery")
        );
        assert!(config_precision(
            &json!({"quantization_config":{"quant_method":"exl3","bits":3,"meta":{
            "a":{"target_bpw":"13/4"},"b":{"target_bpw":"17/4"}}}})
        )
        .is_err());
        for q in [
            json!({"quant_method":"fp8", "store_dtype":"future-fp2"}),
            json!({"quant_method":"fp8", "moe_quant_algo":"future-fp2"}),
            json!({"quant_method":"compressed-tensors", "config_groups":{
                "routed":{"targets":["mlp.experts"],"weights":{"type":"int","num_bits":4}}}}),
        ] {
            assert!(matches!(
                config_precision(&json!({"quantization_config":q})),
                Err(Error::UnknownFormat(_))
            ));
        }
        assert_eq!(config_precision(&json!({})).unwrap().expert_bpw, 16.0);
    }

    #[test]
    fn reference_root_is_strict_at_honest_four_bpw_but_requantization_is_not() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            json!({"quantization_config":{"quant_method":"fp8","expert_dtype":"fp4"}}).to_string(),
        )
        .unwrap();
        let p = resolve(
            "test/root",
            Some(root.path()),
            &json!([]),
            "test/root",
            None,
        )
        .unwrap();
        assert_eq!(p.expert_bpw, 4.0);
        assert_eq!(p.tier, Tier::Strict);
        assert!(p.reference_root_format);
        assert!(p.label().contains("4.0 bpw, reference-root format"));
        std::fs::write(
            root.path().join("config.json"),
            json!({"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}})
                .to_string(),
        )
        .unwrap();
        let p = resolve(
            "test/requant",
            Some(root.path()),
            &json!([]),
            "test/root",
            None,
        )
        .unwrap();
        assert_eq!(p.expert_bpw, 4.0);
        assert_eq!(p.tier, Tier::Mid);
        assert!(!p.reference_root_format);
    }

    #[test]
    fn verified_ancestry_with_root_format_is_strict_but_nvfp4_is_not() {
        let hub = tempfile::tempdir().unwrap();
        let root = hub.path().join("models--test--root/snapshots/aaaa");
        let served = hub.path().join("alias");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&served).unwrap();
        let q =
            serde_json::json!({"quantization_config":{"quant_method":"fp8","store_dtype":"mxfp4"}});
        std::fs::write(root.join("config.json"), q.to_string()).unwrap();
        std::fs::write(served.join("config.json"), q.to_string()).unwrap();
        let selection = crate::fidelity_match::Resolution {
            reference_match: "base_model".into(),
            reference_root: "test/root".into(),
            text_checkpoint: None,
            resolved_chain: vec![vec!["test/alias".into(), "test/root".into()]],
            revision: String::new(),
            config: String::new(),
        };
        let p = resolve_in(
            "test/alias",
            Some(&served),
            &json!([]),
            "test/root",
            Some(&selection),
            hub.path(),
        )
        .unwrap();
        assert_eq!(p.expert_bpw, 4.0);
        assert_eq!(p.tier, Tier::Strict);
        std::fs::write(
            served.join("config.json"),
            json!({"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}})
                .to_string(),
        )
        .unwrap();
        let p = resolve_in(
            "test/alias",
            Some(&served),
            &json!([]),
            "test/root",
            Some(&selection),
            hub.path(),
        )
        .unwrap();
        assert_eq!(p.tier, Tier::Mid);
        assert!(!p.reference_root_format);
    }

    #[test]
    fn manifest_selection_is_required_and_unknown_manifest_format_is_typed() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("manifests")).unwrap();
        std::fs::write(temp.path().join("config.json"), "{\"num_hidden_layers\":2}").unwrap();
        assert!(
            matches!(resolve("test/exlr8", Some(temp.path()), &json!([]), "test/root", None), Err(Error::Invalid(s)) if s.contains("active"))
        );
        std::fs::write(
            temp.path().join("manifests/exlr8-manifest-home.json"),
            json!({"format":"future-exlr8","name":"home"}).to_string(),
        )
        .unwrap();
        assert!(
            matches!(resolve("test/exlr8", Some(temp.path()), &json!([{"name":"exlr8-manifest","value":"home"}]), "test/root", None), Err(Error::UnknownFormat(s)) if s.contains("future-exlr8"))
        );
    }

    #[test]
    fn active_manifest_weights_projection_units_and_excludes_shared_and_mtp() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("manifests")).unwrap();
        let config = json!({"num_hidden_layers":2});
        let manifest = json!({"format":"exlr8-v2-manifest","name":"test","layers":{
            "1":{"gate":[[2,4]],"up":[[3,3]],"down":[4],"shared":{"gate":[2]}},
            "2":{"gate":[[2,2]],"up":[[2,2]],"down":[2]}}});
        std::fs::write(
            temp.path().join("manifests/exlr8-manifest-test.json"),
            manifest.to_string(),
        )
        .unwrap();
        let p = manifest_precision(temp.path(), &config, "test").unwrap();
        assert_eq!(p.expert_bpw, 20.0 / 6.0);
        assert_eq!(p.tier, Tier::Mid);
        assert!(manifest_precision(temp.path(), &config, "../test").is_err());
    }

    #[test]
    fn cached_checkpoint_metadata_when_available() {
        let hub = crate::fidelity_match::hub();
        for (model, bpw, root) in [
            ("wrldsuksgo2mars/Kimi-K3-EXL3-K2", 2.0, "moonshotai/Kimi-K3"),
            (
                "wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1",
                2.0,
                "deepseek-ai/DeepSeek-V4-Pro-0813",
            ),
            (
                "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1",
                3.25,
                "zai-org/GLM-5.3-Flash-BF16",
            ),
            (
                "wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-v1",
                4.25,
                "Qwen/Qwen3.8-Flash-Next",
            ),
            (
                "nvidia/DeepSeek-V4.1-Flash-NVFP4",
                4.0,
                "deepseek-ai/DeepSeek-V4.1-Flash",
            ),
            (
                "deepseek-ai/DeepSeek-V4.1-Flash",
                4.0,
                "deepseek-ai/DeepSeek-V4.1-Flash",
            ),
            (
                "deepseek-ai/DeepSeek-V4-Flash-0731",
                8.0,
                "deepseek-ai/DeepSeek-V4-Flash-0731",
            ),
        ] {
            let Some(path) = crate::fidelity_match::local_snapshot(&hub, model) else {
                continue;
            };
            let p = resolve(model, Some(&path), &json!([]), root, None).unwrap();
            assert_eq!(p.expert_bpw, bpw, "{model}");
            assert_eq!(
                p.tier,
                if model == root {
                    Tier::Strict
                } else if bpw < 3.0 {
                    Tier::Low
                } else {
                    Tier::Mid
                }
            );
        }
        let Some(path) =
            crate::fidelity_match::local_snapshot(&hub, "kindlingai/glm-5.3-exlr8-k2-k3-k4")
        else {
            return;
        };
        for (name, expected) in [("home", 3.25), ("up1", 4.0)] {
            if !path
                .join(format!("manifests/exlr8-manifest-{name}.json"))
                .exists()
            {
                continue;
            }
            let p = resolve(
                "kindlingai/glm-5.3-exlr8-k2-k3-k4",
                Some(&path),
                &json!([{"name":"exlr8-manifest","value":name}]),
                "zai-org/GLM-5.3-BF16",
                None,
            )
            .unwrap();
            assert_eq!(p.expert_bpw, expected, "{name}");
        }
    }
}
