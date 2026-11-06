//! One immutable attention bundle, selected before inventory or payload reads.
use super::read_json;
#[cfg(test)]
use super::HF_INDEX;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const COMPACT_CONFIG: &str = "config.compact-attention.json";
pub const COMPACT_INDEX: &str = "model.compact-attention.safetensors.index.json";
pub const COMPACT_QUANT: &str = "quantize_config.compact-attention.json";
/// Changing the quantizer invalidates only FP8 attention identities.
pub const FP8_RECIPE: &str = "e4m3-block128-rne-v1";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionMode {
    /// The checkpoint's default bundle; HQA publications retain native attention.
    #[default]
    Hqa,
    /// The complete alternate bundle, preserving the parent's quantized attention.
    Compact,
    /// The default bundle with deterministic load-time FP8 attention conversion.
    Fp8Compact,
}

#[derive(Debug, Clone)]
pub struct CheckpointView {
    snapshot: PathBuf,
    mode: AttentionMode,
    config_name: String,
    index_name: String,
    quant_names: Vec<String>,
    fingerprint: String,
}

fn check_references(value: &Value, quant_names: &[String], index: &str) -> Result<()> {
    match value {
        Value::Object(object) => for (key, value) in object {
            if key == "per_module_metadata" {
                let reference = value.as_str().context("per_module_metadata must name a metadata file")?;
                let (file, field) = reference.split_once(':').context("per_module_metadata must name file:field")?;
                ensure!(quant_names.iter().any(|name| name == file) && field == "tensor_storage",
                    "attention bundle references unselected quantization metadata {reference}");
            }
            if key == "transformers_weights" && !value.is_null() {
                ensure!(value.as_str() == Some(index), "attention bundle references unselected index {value}");
            }
            check_references(value, quant_names, index)?;
        },
        Value::Array(values) => for value in values { check_references(value, quant_names, index)?; },
        _ => (),
    }
    Ok(())
}

impl CheckpointView {
    pub fn select(snapshot: &Path, mode: AttentionMode) -> Result<Self> {
        Self::select_recipe(snapshot, mode, FP8_RECIPE)
    }

    fn select_recipe(snapshot: &Path, mode: AttentionMode, recipe: &str) -> Result<Self> {
        let config_name = if mode == AttentionMode::Compact { COMPACT_CONFIG } else { "config.json" };
        let config = read_json(&snapshot.join(config_name))
            .with_context(|| format!("selecting {mode:?} attention bundle: {config_name}"))?;
        let index_name = if mode == AttentionMode::Compact { COMPACT_INDEX.to_owned() }
            else { super::index_name_from_config(&config)? };
        let quant_names: Vec<String> = if mode == AttentionMode::Compact {
            for name in [COMPACT_INDEX, COMPACT_QUANT] {
                ensure!(snapshot.join(name).is_file(), "incomplete compact attention bundle: missing {name}");
            }
            vec![COMPACT_QUANT.into()]
        } else {
            ["quantize_config.json", "quantization_config.json", "hf_quant_config.json"].into_iter()
                .filter(|name| snapshot.join(name).is_file()).map(str::to_owned).collect()
        };
        let allowed_quant = if mode == AttentionMode::Compact { quant_names.clone() } else {
            ["quantize_config.json", "quantization_config.json", "hf_quant_config.json"].map(str::to_owned).to_vec()
        };
        check_references(&config, &allowed_quant, &index_name)?;
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-attention-view-v1\0");
        hash.update(match mode { AttentionMode::Hqa => b"hqa".as_slice(),
            AttentionMode::Compact => b"compact", AttentionMode::Fp8Compact => b"fp8-compact" });
        if mode == AttentionMode::Fp8Compact { hash.update(recipe.as_bytes()); }
        for name in std::iter::once(config_name).chain(quant_names.iter().map(String::as_str)) {
            let path = snapshot.join(name);
            // Storage maps are large; parse their compact fields with the shared
            // streaming visitor, and bind the full original bytes by digest.
            let value = if name == config_name { config.clone() } else {
                crate::formats::exl3_storage::read_storage_metadata(&path, super::MAX_JSON_BYTES, None)?
            };
            check_references(&value, &allowed_quant, &index_name)?;
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(std::fs::read(path)?);
        }
        if snapshot.join(&index_name).is_file() {
            let index = read_json(&snapshot.join(&index_name))?;
            ensure!(index["weight_map"].is_object(), "attention bundle index has no weight_map");
            hash.update(index_name.as_bytes());
            hash.update(serde_json::to_vec(&index)?);
        } else {
            ensure!(mode != AttentionMode::Compact, "compact attention requires {COMPACT_INDEX}");
        }
        Ok(Self { snapshot: snapshot.into(), mode, config_name: config_name.into(), index_name,
            quant_names, fingerprint: format!("{:x}", hash.finalize()) })
    }

    pub fn snapshot(&self) -> &Path { &self.snapshot }
    pub fn mode(&self) -> AttentionMode { self.mode }
    pub fn config_name(&self) -> &str { &self.config_name }
    pub fn index_name(&self) -> &str { &self.index_name }
    pub fn quant_names(&self) -> &[String] { &self.quant_names }
    pub fn fingerprint(&self) -> &str { &self.fingerprint }
    pub fn config(&self) -> Result<Value> { read_json(&self.snapshot.join(&self.config_name)) }
    pub fn index_path(&self) -> PathBuf { self.snapshot.join(&self.index_name) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bundle(root: &Path, compact: bool) {
        let (config, index, quant) = if compact { (COMPACT_CONFIG, COMPACT_INDEX, COMPACT_QUANT) }
            else { ("config.json", HF_INDEX, "quantize_config.json") };
        std::fs::write(root.join(config), json!({"quantization_config": {
            "per_module_metadata": format!("{quant}:tensor_storage")}}).to_string()).unwrap();
        std::fs::write(root.join(index), json!({"weight_map": {"a.weight": "a.safetensors"}}).to_string()).unwrap();
        std::fs::write(root.join(quant), "{}").unwrap();
    }

    #[test]
    fn bundle_is_complete_and_never_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        bundle(dir.path(), false);
        assert!(CheckpointView::select(dir.path(), AttentionMode::Compact).is_err());
        bundle(dir.path(), true);
        let view = CheckpointView::select(dir.path(), AttentionMode::Compact).unwrap();
        assert_eq!(view.index_name(), COMPACT_INDEX);
        assert_eq!(view.quant_names(), &[COMPACT_QUANT]);
        std::fs::remove_file(dir.path().join(COMPACT_QUANT)).unwrap();
        assert!(CheckpointView::select(dir.path(), AttentionMode::Compact).is_err());
        bundle(dir.path(), true);
        std::fs::copy(dir.path().join("config.json"), dir.path().join(COMPACT_CONFIG)).unwrap();
        assert!(CheckpointView::select(dir.path(), AttentionMode::Compact).unwrap_err().to_string()
            .contains("unselected quantization"));
    }

    #[test]
    fn identities_separate_modes_and_only_fp8_depends_on_recipe() {
        let dir = tempfile::tempdir().unwrap();
        bundle(dir.path(), false);
        bundle(dir.path(), true);
        let identities = [AttentionMode::Hqa, AttentionMode::Compact, AttentionMode::Fp8Compact]
            .map(|mode| CheckpointView::select(dir.path(), mode).unwrap().fingerprint);
        assert_ne!(identities[0], identities[1]);
        assert_ne!(identities[0], identities[2]);
        assert_ne!(identities[1], identities[2]);
        for mode in [AttentionMode::Hqa, AttentionMode::Compact, AttentionMode::Fp8Compact] {
            let old = CheckpointView::select_recipe(dir.path(), mode, "recipe-a").unwrap();
            let new = CheckpointView::select_recipe(dir.path(), mode, "recipe-b").unwrap();
            assert_eq!(old.fingerprint == new.fingerprint, mode != AttentionMode::Fp8Compact);
        }
    }
}
