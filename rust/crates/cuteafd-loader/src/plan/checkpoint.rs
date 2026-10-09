//! Family-independent view of a Hugging Face checkpoint: configuration JSON,
//! the safetensors index and every shard header. Nothing here reads tensor data.
use crate::{read_safetensors_metadata, SafetensorsTensorMetadata};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;

/// One tensor as the checkpoint stores it.
#[derive(Debug, Clone)]
pub struct CheckpointTensor {
    pub shard: String,
    pub meta: SafetensorsTensorMetadata,
}

#[derive(Debug)]
pub struct Checkpoint {
    pub snapshot: PathBuf,
    /// `config.json`, verbatim.
    pub config: Value,
    /// A separate `quantize_config.json` / `quantization_config.json`, when present.
    pub quantize_config: Option<Value>,
    /// Authoritative inventory, including tensors whose shards are not local.
    pub weight_map: BTreeMap<String, String>,
    /// Headers read for this role, sorted by name.
    pub tensors: Vec<CheckpointTensor>,
    /// Shards named by the index that are not readable (absent or incomplete).
    pub missing_shards: Vec<String>,
    pub shard_bytes: u64,
}

pub fn read_json(path: &Path) -> Result<Value> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_JSON_BYTES,
        "{} exceeds {MAX_JSON_BYTES} bytes",
        path.display()
    );
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

impl Checkpoint {
    /// Reads the configuration, the index (or a single `model.safetensors`) and
    /// every shard header. Missing shards are recorded rather than fatal so a
    /// partially downloaded checkpoint can still be planned.
    pub fn open(snapshot: &Path) -> Result<Self> {
        Self::open_filtered(snapshot, None, |_| true)
    }

    /// Coordinator ownership uses the planner's family classification, not
    /// whichever headers happen to be readable on this host.
    pub fn coordinator(snapshot: &Path, local_experts: bool, speculator: bool) -> Result<Self> {
        super::files::open_role(snapshot, super::files::ReadRole::Coordinator { local_experts, speculator })
    }

    /// Reads only the index and configuration. Presence never depends on headers.
    pub fn inventory(snapshot: &Path) -> Result<Self> {
        Self::open_filtered(snapshot, None, |_| false)
    }

    /// A runtime role declares the tensors it reads before any shard is opened.
    /// Every selected index entry must have a readable, matching header.
    pub fn open_for_role(snapshot: &Path, role: &str, needed: impl Fn(&str) -> bool) -> Result<Self> {
        Self::open_filtered(snapshot, Some(role), needed)
    }

    fn open_filtered(snapshot: &Path, role: Option<&str>, needed: impl Fn(&str) -> bool) -> Result<Self> {
        let config = read_json(&snapshot.join("config.json"))?;
        let quantize_config = ["quantize_config.json", "quantization_config.json"]
            .iter()
            .map(|name| snapshot.join(name))
            .find(|path| path.is_file())
            .map(|path| read_json(&path))
            .transpose()?;
        let index_path = snapshot.join("model.safetensors.index.json");
        let weight_map: BTreeMap<String, String> = if index_path.is_file() {
            let index = read_json(&index_path)?;
            serde_json::from_value(
                index
                    .get("weight_map")
                    .cloned()
                    .context("index has no weight_map")?,
            )
            .context("decoding weight_map")?
        } else {
            let single = snapshot.join("model.safetensors");
            ensure!(
                single.is_file(),
                "{} has neither model.safetensors.index.json nor model.safetensors",
                snapshot.display()
            );
            read_safetensors_metadata(&single)?
                .into_iter()
                .map(|meta| (meta.name, "model.safetensors".to_owned()))
                .collect()
        };
        let shards: BTreeSet<&String> = weight_map.iter()
            .filter(|(name, _)| needed(name)).map(|(_, shard)| shard).collect();
        let mut tensors = Vec::with_capacity(weight_map.len());
        let mut missing_shards = Vec::new();
        let mut shard_bytes = 0u64;
        for shard in shards {
            let path = snapshot.join(shard);
            let tensor = weight_map.iter().find(|(name, file)| *file == shard && needed(name))
                .map(|(name, _)| name.as_str()).context("selected shard has no tensor")?;
            let requirement = || format!("role {} needs tensor {tensor} in shard {shard} at snapshot {}",
                role.unwrap_or("planner"), snapshot.display());
            let metadata = match path.metadata() {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => {
                    ensure!(role.is_none(), "{}: absent or unreadable", requirement());
                    missing_shards.push(shard.clone());
                    continue;
                }
            };
            let headers = match read_safetensors_metadata(&path) {
                Ok(headers) => headers,
                Err(error) => {
                    ensure!(role.is_none(), "{}: {error:#}", requirement());
                    missing_shards.push(shard.clone());
                    continue;
                }
            };
            if let Some(role) = role {
                let header_names: BTreeSet<_> = headers.iter().map(|meta| meta.name.as_str()).collect();
                for (name, file) in &weight_map {
                    if file == shard && needed(name) {
                        ensure!(header_names.contains(name.as_str()),
                            "role {role} needs tensor {name} in shard {shard} at snapshot {}: header is missing",
                            snapshot.display());
                    }
                }
            }
            shard_bytes += metadata.len();
            for meta in headers {
                if weight_map.get(&meta.name) == Some(shard) && needed(&meta.name) {
                    tensors.push(CheckpointTensor {
                        shard: shard.clone(),
                        meta,
                    });
                }
            }
        }
        tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        Ok(Self {
            snapshot: snapshot.to_path_buf(),
            config,
            quantize_config,
            weight_map,
            tensors,
            missing_shards,
            shard_bytes,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        // In-memory fixtures predating the HF index use their complete headers.
        let from_headers = self.weight_map.is_empty();
        self.weight_map.keys().map(String::as_str)
            .chain(self.tensors.iter().filter(move |_| from_headers).map(|t| t.meta.name.as_str()))
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        if !self.weight_map.is_empty() { return self.weight_map.contains_key(name); }
        self.tensors.binary_search_by(|tensor| tensor.meta.name.as_str().cmp(name)).is_ok()
    }

    /// Explicit requirements also reject tensors absent from the index itself.
    pub fn open_required(snapshot: &Path, role: &str, names: &BTreeSet<String>) -> Result<Self> {
        let inventory = Self::inventory(snapshot)?;
        for name in names {
            ensure!(inventory.contains_tensor(name),
                "role {role} needs tensor {name} at snapshot {}: absent from weight_map", snapshot.display());
        }
        Self::open_for_role(snapshot, role, |name| names.contains(name))
    }

    /// The text model configuration: `text_config` when present, else the root.
    pub fn text_config(&self) -> &Value {
        self.config
            .get("text_config")
            .filter(|value| value.is_object())
            .unwrap_or(&self.config)
    }

    /// `quantization_config` from config.json, else the external file.
    pub fn quantization(&self) -> Option<&Value> {
        self.config
            .get("quantization_config")
            .or_else(|| self.text_config().get("quantization_config"))
            .or(self.quantize_config.as_ref())
    }

    pub fn architectures(&self) -> Vec<String> {
        self.config
            .get("architectures")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn model_type(&self) -> Option<&str> {
        self.config.get("model_type").and_then(Value::as_str)
    }

    /// Host placement cannot free a tensor also used by the vocabulary head.
    /// Require a separate head in addition to the config's tying declaration.
    pub fn require_untied_embedding(&self, name: &str) -> Result<()> {
        ensure!(self.text_config().get("tie_word_embeddings").and_then(Value::as_bool) != Some(true)
            && self.config.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true),
            "{name}: host embedding is ineligible: tie_word_embeddings=true");
        let embedding = self.tensors.iter().find(|t| t.meta.name == name)
            .with_context(|| format!("host embedding tensor {name} is missing"))?;
        let heads: Vec<_> = self.tensors.iter().filter(|t|
            t.meta.name == "head.weight" || t.meta.name == "lm_head.weight"
                || t.meta.name.ends_with(".lm_head.weight")).collect();
        ensure!(!heads.is_empty(), "{name}: host embedding is ineligible: no separate LM head tensor");
        for head in heads {
            let same_file = head.shard == embedding.shard
                || std::fs::canonicalize(self.snapshot.join(&head.shard)).ok().zip(
                    std::fs::canonicalize(self.snapshot.join(&embedding.shard)).ok())
                    .is_some_and(|(a, b)| a == b);
            let overlaps = head.meta.byte_offset < embedding.meta.byte_offset.saturating_add(embedding.meta.byte_length)
                && embedding.meta.byte_offset < head.meta.byte_offset.saturating_add(head.meta.byte_length);
            ensure!(!(same_file && overlaps),
                "{name}: host embedding is ineligible: LM head {} aliases its storage", head.meta.name);
        }
        Ok(())
    }
}

/// Reads an integer config field, checking `text_config` first.
pub fn usize_field(config: &Value, key: &str) -> Result<usize> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("config field {key} is missing or not an unsigned integer"))
}

pub fn opt_usize_field(config: &Value, key: &str) -> Option<usize> {
    config.get(key).and_then(Value::as_u64).map(|value| value as usize)
}

#[cfg(test)]
mod sliced_tests {
    use super::*;
    use crate::plan::testing::{t, write_safetensors};
    use serde_json::json;

    #[test]
    fn role_headers_and_index_presence_are_independent() {
        let dir = tempfile::tempdir().unwrap();
        let names = ["lm_head.weight", "model.layers.1.mlp.experts.7.gate_proj.weight",
            "model.mtp.layers.0.eh_proj.weight", "model.visual.weight"];
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let map: BTreeMap<_, _> = names.iter().enumerate().map(|(i, name)|
            (name.to_string(), format!("shard{i}.safetensors"))).collect();
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map": map})).unwrap()).unwrap();
        for (i, name) in names.iter().enumerate() {
            write_safetensors(&dir.path().join(format!("shard{i}.safetensors")), &[t(*name, "BF16", &[2])]);
        }
        let full = Checkpoint::open(dir.path()).unwrap();
        std::fs::remove_file(dir.path().join("shard1.safetensors")).unwrap();
        std::fs::write(dir.path().join("shard3.safetensors"), b"corrupt").unwrap();
        let local = |name: &str| name == names[0] || name == names[2];
        let sliced = Checkpoint::open_for_role(dir.path(), "coordinator", local).unwrap();
        assert_eq!(full.tensor_names().collect::<Vec<_>>(), sliced.tensor_names().collect::<Vec<_>>());
        assert_eq!(full.tensors.iter().filter(|t| local(&t.meta.name)).map(|t| &t.meta).collect::<Vec<_>>(),
            sliced.tensors.iter().map(|t| &t.meta).collect::<Vec<_>>());
        assert!(sliced.contains_tensor(names[1]));
        assert!(sliced.contains_tensor(names[3]));
        let error = Checkpoint::open_for_role(dir.path(), "spark0", |name| name == names[1]).unwrap_err().to_string();
        for expected in ["spark0", names[1], "shard1.safetensors", dir.path().to_str().unwrap()] {
            assert!(error.contains(expected), "{error}");
        }
        assert!(Checkpoint::open_for_role(dir.path(), "vision", |name| name == names[3]).is_err());
        std::fs::remove_file(dir.path().join("shard0.safetensors")).unwrap();
        let error = Checkpoint::open_for_role(dir.path(), "coordinator", local).unwrap_err().to_string();
        assert!(error.contains(names[0]) && error.contains("shard0.safetensors"));
    }

    #[test]
    fn selected_tensor_missing_from_header_is_named() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"head.weight":"head.safetensors"}}"#).unwrap();
        write_safetensors(&dir.path().join("head.safetensors"), &[t("other.weight", "BF16", &[2])]);
        let error = Checkpoint::open_for_role(dir.path(), "coordinator", |_| true).unwrap_err().to_string();
        assert!(error.contains("head.weight") && error.contains("head.safetensors"));
        let required = BTreeSet::from(["absent.weight".to_string()]);
        assert!(Checkpoint::open_required(dir.path(), "coordinator", &required).unwrap_err()
            .to_string().contains("absent from weight_map"));
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    use crate::SafetensorsTensorMetadata;
    use cuteafd_core::DType;

    fn fixture() -> Checkpoint {
        let t = |name: &str, offset| CheckpointTensor { shard: "model.safetensors".into(),
            meta: SafetensorsTensorMetadata { name: name.into(), dtype: DType::Bf16,
                shape: vec![64, 128], byte_offset: offset, byte_length: 16384 } };
        Checkpoint { snapshot: Default::default(), config: serde_json::json!({"tie_word_embeddings":false}),
            quantize_config: None, weight_map: Default::default(), missing_shards: vec![], shard_bytes: 0,
            tensors: vec![t("model.embed_tokens.weight", 0), t("lm_head.weight", 16384)] }
    }

    #[test]
    fn host_embedding_requires_untied_distinct_storage() {
        let mut c = fixture();
        assert!(c.require_untied_embedding("model.embed_tokens.weight").is_ok());
        c.config["tie_word_embeddings"] = Value::Bool(true);
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("tie_word_embeddings"));
        c.config = serde_json::json!({"text_config":{"tie_word_embeddings":true}});
        assert!(c.require_untied_embedding("model.embed_tokens.weight").is_err());
        c.config = serde_json::json!({});
        c.tensors[1].meta.byte_offset = 8192;
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("aliases"));
        c.tensors.pop();
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("no separate LM head"));
    }
}
