//! Optional EXL3 publication metadata cross-checks. Tensor headers own storage;
//! declared core, vocabulary and MTP widths only verify that derived inventory.
use super::exl3_storage::{check_quantization_config, Exl3StorageMap};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

fn head(name: &str) -> bool {
    name == "lm_head" || name.ends_with(".lm_head")
}

pub(crate) fn routed(name: &str) -> bool {
    name.split_once(".experts.").is_some_and(|(_, tail)|
        tail.split_once('.').is_some_and(|(expert, _)| expert.parse::<usize>().is_ok()))
}

fn mtp(name: &str, backbone_layers: usize) -> bool {
    name.starts_with("mtp.") || name.contains(".mtp.") || name.split_once(".layers.")
        .and_then(|(_, tail)| tail.split('.').next()?.parse::<usize>().ok())
        .is_some_and(|layer| layer >= backbone_layers)
}

fn layer_of(name: &str) -> Option<usize> {
    name.split_once(".layers.").and_then(|(_, tail)| tail.split('.').next()?.parse().ok())
}

fn bits(key: &str, value: &Value, max: usize) -> Result<usize> {
    Ok(value.as_f64().filter(|v| v.fract() == 0.0 && (1.0..=max as f64).contains(v))
        .with_context(|| format!("EXL3 {key} must be an integer in 1..{max}"))? as usize)
}

pub(crate) fn check_header_widths(
    quant: &Value, derived: &Exl3StorageMap, backbone_layers: usize,
) -> Result<()> {
    for key in ["core_bits", "head_bits", "mtp_bits"] {
        let Some(value) = quant.get(key) else { continue };
        let width = bits(key, value, if key == "head_bits" { 16 } else { 8 })?;
        let mut modules = derived.iter().filter(|(name, _)| match key {
            "mtp_bits" => mtp(name, backbone_layers),
            "head_bits" => head(name),
            _ => !routed(name) && !mtp(name, backbone_layers)
                && !(head(name) && quant.get("head_bits").is_some()),
        }).peekable();
        // A native BF16 head has no EXL3 header entry.
        ensure!(modules.peek().is_some() || key == "head_bits" && width == 16,
            "EXL3 {key}={width} but the headers carry no matching EXL3 modules");
        for (name, module) in modules {
            ensure!(module.bits == width,
                "EXL3 {key}={width} disagrees with the headers at {name} (K{})", module.bits);
        }
    }
    // Routed experts: `expert_bits_by_layer` overrides `expert_bits` per target layer.
    if quant.get("expert_bits").is_some() || quant.get("expert_bits_by_layer").is_some() {
        let base = quant.get("expert_bits").map(|v| bits("expert_bits", v, 8)).transpose()?;
        let by_layer = match quant.get("expert_bits_by_layer") {
            None => BTreeMap::new(),
            Some(map) => map.as_object().context("EXL3 expert_bits_by_layer must be an object")?.iter()
                .map(|(layer, v)| Ok((layer.parse::<usize>().with_context(||
                    format!("EXL3 expert_bits_by_layer key {layer} is not a layer"))?, bits("expert_bits_by_layer", v, 8)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        };
        for (name, module) in derived.iter().filter(|(name, _)| routed(name) && !mtp(name, backbone_layers)) {
            let layer = layer_of(name);
            let declared = layer.and_then(|l| by_layer.get(&l).copied()).or(base);
            if let Some(width) = declared {
                ensure!(module.bits == width,
                    "EXL3 routed width K{width} disagrees with the headers at {name} (K{})", module.bits);
            }
        }
    }
    if let Some(value) = quant.get("lm_head") {
        let declared = value.as_bool().context("EXL3 lm_head must be boolean")?;
        ensure!(declared == derived.keys().any(|name| head(name)),
            "EXL3 lm_head={declared} disagrees with the vocabulary headers");
    }
    Ok(())
}

pub(crate) fn check_gptqmodel_manifest(
    compact: &Value, manifest: &Value, derived: &Exl3StorageMap, backbone_layers: usize,
) -> Result<()> {
    for field in ["quant_method", "method", "format", "checkpoint_format"] {
        ensure!(manifest.get(field).and_then(Value::as_str) == Some("exl3"),
            "EXL3 manifest requires {field}=exl3");
    }
    // out_scales describes whether the quantizer folded scales into svh.
    ensure!(manifest["codebook"] == "mcg"
        && matches!(manifest["out_scales"].as_str(), Some("never" | "auto"))
        && manifest["group_size"] == -1 && manifest["desc_act"] == false
        && manifest.get("pack_dtype").is_none_or(|v| v == "int32")
        && manifest.get("sym").is_none_or(|v| v == true)
        && manifest.get("calibration").is_none_or(Value::is_object),
        "unsupported EXL3 storage contract");
    ensure!(manifest["bits"].as_f64().is_some_and(|v| v.fract() == 0.0 && (2.0..=5.0).contains(&v)),
        "EXL3 base bits must be an integer in 2..5");
    let object = manifest.as_object().context("EXL3 manifest must be an object")?;
    for key in object.keys() {
        ensure!(matches!(key.as_str(),
            "bits" | "checkpoint_format" | "codebook" | "core_bits" | "head_bits" | "mtp_bits"
            | "desc_act" | "format" | "group_size" | "lm_head" | "meta" | "method" | "module_include"
            | "out_scales" | "pack_dtype" | "quant_method" | "tensor_storage" | "per_module_metadata"
            | "sym" | "calibration" | "mcg_multiplier" | "expert_bits" | "expert_bits_by_layer"
            | "kept_tensors"), "EXL3 manifest has unexpected key {key}");
    }
    if let Some(multiplier) = manifest.get("mcg_multiplier") {
        ensure!(multiplier.as_u64() == Some(super::exl3_storage::EXL3_MCG_MULTIPLIER),
            "EXL3 mcg_multiplier {multiplier} (the decoders implement 0xCBAC1FED)");
    }
    if let Some(kept) = manifest.get("kept_tensors") {
        ensure!(kept.is_object(), "EXL3 kept_tensors must map tensor names to native dtypes/shapes");
    }
    check_header_widths(manifest, derived, backbone_layers)?;
    let compact = compact.as_object().context("missing compact EXL3 config")?;
    ensure!(compact.get("quant_method") == Some(&Value::from("exl3")), "config must declare EXL3 quantization");
    for (key, value) in compact {
        if key == "per_module_metadata" {
            ensure!(value == "quantize_config.json:tensor_storage",
                "EXL3 per_module_metadata must name quantize_config.json:tensor_storage, not {value}");
            continue;
        }
        ensure!(manifest.get(key) == Some(value), "compact/full EXL3 metadata disagree at {key}");
    }
    if let Some(pointer) = manifest.get("per_module_metadata") {
        ensure!(pointer == "quantize_config.json:tensor_storage",
            "EXL3 per_module_metadata must name quantize_config.json:tensor_storage, not {pointer}");
    }
    Ok(())
}

pub(crate) fn check_exllamav3_manifest(compact: &Value, manifest: &Value) -> Result<()> {
    let manifest = manifest.as_object().context("EXL3 storage map must be an object")?;
    let compact_object = compact.as_object().context("missing compact EXL3 config")?;
    for key in compact_object.keys().chain(manifest.keys()) {
        ensure!(key == "tensor_storage" || manifest.get(key) == compact_object.get(key),
            "compact/full EXL3 metadata disagree at {key}");
    }
    ensure!(!compact_object.contains_key("tensor_storage"), "config.json carries an inline tensor_storage");
    check_quantization_config(compact)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::exl3_storage::{Exl3Codebook, Exl3Module};
    use std::collections::BTreeMap;

    fn storage() -> Exl3StorageMap {
        let module = |bits| Exl3Module { codebook: Exl3Codebook::Mcg, bits,
            input_features: 128, output_features: 128, tensors: BTreeMap::new() };
        BTreeMap::from([
            ("model.layers.0.self_attn.q_a_proj".into(), module(6)),
            ("model.layers.1.mlp.shared_experts.down_proj".into(), module(6)),
            ("model.layers.1.mlp.experts.0.gate_proj".into(), module(4)),
            ("model.layers.2.self_attn.q_a_proj".into(), module(4)),
            ("model.layers.2.mlp.experts.0.gate_proj".into(), module(4)),
            ("model.layers.2.eh_proj".into(), module(4)),
            ("lm_head".into(), module(6)),
        ])
    }

    fn publication() -> (Value, Value) {
        let compact = serde_json::json!({"quant_method":"exl3", "bits":4.0,
            "core_bits":6.0, "head_bits":6.0, "mtp_bits":4.0, "checkpoint_format":"exl3",
            "format":"exl3", "per_module_metadata":"quantize_config.json:tensor_storage"});
        let mut manifest = compact.clone();
        manifest.as_object_mut().unwrap().extend(serde_json::json!({"codebook":"mcg", "method":"exl3",
            "desc_act":false, "group_size":-1, "lm_head":true, "out_scales":"auto", "sym":true,
            "calibration":{"samples":1426, "tokens":1097737}, "meta":{}, "module_include":[],
            "tensor_storage":true}).as_object().unwrap().clone());
        (compact, manifest)
    }

    #[test]
    fn published_glm_k6_core_k4_mtp_key_set_checks_header_widths() {
        let (compact, manifest) = publication();
        let storage = storage();
        check_gptqmodel_manifest(&compact, &manifest, &storage, 2).unwrap();
        check_quantization_config(&compact).unwrap();
        check_header_widths(&compact, &storage, 2).unwrap();
        for (stem, key) in [
            ("model.layers.0.self_attn.q_a_proj", "core_bits"),
            ("model.layers.1.mlp.shared_experts.down_proj", "core_bits"),
            ("model.layers.2.self_attn.q_a_proj", "mtp_bits"),
            ("model.layers.2.mlp.experts.0.gate_proj", "mtp_bits"),
            ("model.layers.2.eh_proj", "mtp_bits"),
            ("lm_head", "head_bits"),
        ] {
            let mut wrong = storage.clone();
            wrong.get_mut(stem).unwrap().bits = 3;
            let error = check_gptqmodel_manifest(&compact, &manifest, &wrong, 2).unwrap_err().to_string();
            assert!(error.contains(stem) && error.contains(key), "{error}");
            assert!(check_header_widths(&compact, &wrong, 2).is_err());
        }
    }

    #[test]
    fn mtp_prefixes_and_backbone_boundary_select_only_mtp() {
        for stem in ["mtp.0.eh_proj", "mtp.layers.0.eh_proj", "model.mtp.layers.0.eh_proj",
            "language_model.model.layers.2.eh_proj", "model.language_model.layers.2.eh_proj"] {
            let mut storage = storage();
            let module = storage.remove("model.layers.2.eh_proj").unwrap();
            storage.insert(stem.into(), module);
            check_header_widths(&publication().0, &storage, 2).unwrap();
        }
    }

    #[test]
    fn mixed_width_contract_rejects_metadata_drift() {
        let (compact, manifest) = publication();
        for key in ["core_bits", "head_bits", "mtp_bits"] {
            for value in [Value::from(0), Value::from(6.5), Value::from("6"), Value::Null] {
                let mut wrong = manifest.clone();
                wrong[key] = value;
                assert!(check_gptqmodel_manifest(&compact, &wrong, &storage(), 2).is_err());
            }
            let mut wrong = compact.clone();
            wrong[key] = Value::from(3.0);
            assert!(check_gptqmodel_manifest(&wrong, &manifest, &storage(), 2).is_err());
        }
        for (key, value) in [("lm_head", Value::from(false)), ("lm_head", Value::from("true")),
            ("sym", Value::from(false)), ("calibration", Value::Null),
            ("per_module_metadata", Value::from("other.json:tensor_storage"))] {
            let mut wrong = manifest.clone();
            wrong[key] = value;
            assert!(check_gptqmodel_manifest(&compact, &wrong, &storage(), 2).is_err(), "accepted {key}");
        }
        let mut missing_head = storage();
        missing_head.remove("lm_head");
        assert!(check_header_widths(&manifest, &missing_head, 2).is_err());
        let mut native_head = manifest;
        native_head["lm_head"] = Value::from(false);
        native_head["head_bits"] = Value::from(16);
        check_header_widths(&native_head, &missing_head, 2).unwrap();
    }
}
