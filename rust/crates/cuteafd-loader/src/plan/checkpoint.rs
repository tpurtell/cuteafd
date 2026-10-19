//! Family-independent view of a Hugging Face checkpoint: configuration JSON,
//! the safetensors index and every shard header. Nothing here reads tensor data.
use crate::{read_safetensors_metadata, SafetensorsTensorMetadata};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Bounds a metadata read, not tensor data. Kimi K3 EXL3 K2's index names
/// 991,812 tensors (~100 MiB) and its GPTQModel storage map is ~141 MiB.
const MAX_JSON_BYTES: u64 = 512 * 1024 * 1024;

pub mod identity;

/// Storage discovery is shared by inventories, catalogs and bootstrap identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat { HfIndexed, HfSingle, Exlr8 }

impl CheckpointFormat {
    pub fn detect(snapshot: &Path) -> Result<Self> {
        if super::exlr8::recognizes(snapshot)? { return Ok(Self::Exlr8); }
        if snapshot.join("model.safetensors.index.json").is_file() { return Ok(Self::HfIndexed); }
        ensure!(snapshot.join("model.safetensors").is_file(),
            "checkpoint has no recognized tensor inventory");
        Ok(Self::HfSingle)
    }
}

/// Arithmetic declared by the checkpoint, independent of optional load optimizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleNumerics { CheckpointPrecision, W4A4, W8A8, W8CheckpointActivations }

fn module_match(pattern: &str, module: &str) -> bool {
    if !pattern.contains('*') {
        return module == pattern || module.strip_prefix(pattern).is_some_and(|s| s.starts_with('.'));
    }
    let mut remainder = module;
    let parts: Vec<_> = pattern.split('*').collect();
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() { continue; }
        let Some(at) = remainder.find(part) else { return false; };
        if i == 0 && at != 0 { return false; }
        remainder = &remainder[at + part.len()..];
    }
    pattern.ends_with('*') || remainder.is_empty()
}

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
    /// Standard external quantization metadata, without the separately validated tensor_storage map.
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

    /// Reads the index/configuration plus scale-free declared-A4 weight headers
    /// to distinguish unquantized modules from incomplete packed storage.
    pub fn inventory(snapshot: &Path) -> Result<Self> {
        let mut checkpoint = Self::open_filtered(snapshot, None, |_| false)?;
        let mut unscaled = BTreeMap::<String, BTreeSet<String>>::new();
        for (name, shard) in &checkpoint.weight_map {
            let Some(module) = name.strip_suffix(".weight") else { continue };
            if checkpoint.declared_numerics(module)? == ModuleNumerics::W4A4
                && ["weight_scale", "weight_scale_2", "input_scale"].iter()
                    .all(|suffix| !checkpoint.contains_tensor(&format!("{module}.{suffix}"))) {
                unscaled.entry(shard.clone()).or_default().insert(name.clone());
            }
        }
        // Only anomalous unscaled declarations need headers; group by shard so
        // the many BF16 nextn experts do not reopen the same header per module.
        for (shard, names) in unscaled {
            for meta in read_safetensors_metadata(&snapshot.join(&shard))? {
                if names.contains(&meta.name) {
                    checkpoint.tensors.push(CheckpointTensor { shard:shard.clone(), meta });
                }
            }
        }
        checkpoint.tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        Ok(checkpoint)
    }

    /// A runtime role declares the tensors it reads before any shard is opened.
    /// Every selected index entry must have a readable, matching header.
    pub fn open_for_role(snapshot: &Path, role: &str, needed: impl Fn(&str) -> bool) -> Result<Self> {
        Self::open_filtered(snapshot, Some(role), needed)
    }

    fn open_filtered(snapshot: &Path, role: Option<&str>, needed: impl Fn(&str) -> bool) -> Result<Self> {
        let config = read_json(&snapshot.join("config.json"))?;
        if CheckpointFormat::detect(snapshot)? == CheckpointFormat::Exlr8 {
            return super::exlr8::inventory(snapshot, config, needed);
        }
        let quantize_config = ["quantize_config.json", "quantization_config.json", "hf_quant_config.json"]
            .iter()
            .map(|name| snapshot.join(name))
            .find(|path| path.is_file())
            .map(|path| crate::formats::exl3_storage::read_storage_metadata(&path, MAX_JSON_BYTES, None))
            .transpose()?;
        let index_path = snapshot.join("model.safetensors.index.json");
        let weight_map: BTreeMap<String, String> = if index_path.is_file() {
            #[derive(serde::Deserialize)]
            struct Index { weight_map: BTreeMap<String, String> }
            let file = File::open(&index_path)?;
            ensure!(file.metadata()?.len() <= MAX_JSON_BYTES, "index exceeds metadata size limit");
            let mut input = std::io::BufReader::new(file).take(MAX_JSON_BYTES + 1);
            let index: Index = serde_json::from_reader(&mut input).context("decoding weight_map")?;
            ensure!(MAX_JSON_BYTES + 1 - input.limit() <= MAX_JSON_BYTES, "index exceeds metadata size limit");
            index.weight_map
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

    /// Select each module's declared arithmetic. Ignore rules take precedence;
    /// class-wide Linear rules apply only to linear weights, not arbitrary tensors.
    pub fn module_numerics(&self, module: &str) -> Result<ModuleNumerics> {
        let stem = module.strip_suffix(".weight").unwrap_or(module);
        let weight = format!("{stem}.weight");
        let dtype = self.tensors.binary_search_by(|tensor| tensor.meta.name.cmp(&weight))
            .ok().map(|at| self.tensors[at].meta.dtype.clone());
        match dtype {
            Some(dtype) => self.module_numerics_with_header(stem, dtype),
            None => self.declared_numerics(stem),
        }
    }

    /// Complete unquantized BF16 headers override an overbroad ModelOpt declaration;
    /// partial quantization is never treated as an unquantized checkpoint module.
    pub fn module_numerics_with_header(&self, module: &str, dtype: cuteafd_core::DType) -> Result<ModuleNumerics> {
        let module = module.strip_suffix(".weight").unwrap_or(module);
        let declared = self.declared_numerics(module)?;
        if declared != ModuleNumerics::W4A4 { return Ok(declared); }
        let scales = ["weight_scale", "weight_scale_2", "input_scale"]
            .map(|suffix| self.contains_tensor(&format!("{module}.{suffix}")));
        if dtype == cuteafd_core::DType::Bf16 && scales.iter().all(|present| !present) {
            static WARNED: std::sync::OnceLock<std::sync::Mutex<BTreeSet<(PathBuf, String)>>> = std::sync::OnceLock::new();
            let mut warned = WARNED.get_or_init(Default::default).lock()
                .map_err(|_| anyhow::anyhow!("checkpoint numerics warning registry poisoned"))?;
            if warned.insert((self.snapshot.clone(), module.to_owned())) {
                tracing::warn!(module, "quantization config/header disagreement: declared W4A4 but BF16 header has no quantization scales; preserving checkpoint precision");
            }
            return Ok(ModuleNumerics::CheckpointPrecision);
        }
        for (suffix, present) in ["weight_scale", "weight_scale_2", "input_scale"].into_iter().zip(scales) {
            ensure!(present, "{module}.{suffix}: declared W4A4 has partial or invalid NVFP4 storage ({dtype:?}); missing scale tensor");
        }
        ensure!(matches!(dtype, cuteafd_core::DType::U8 | cuteafd_core::DType::I8),
            "{module}.weight: declared W4A4 has partial or invalid NVFP4 storage ({dtype:?})");
        Ok(declared)
    }

    fn declared_numerics(&self, module: &str) -> Result<ModuleNumerics> {
        let module = module.strip_suffix(".weight").unwrap_or(module);
        // Stacked MoE projections are parameters of the MoE, not child Linear
        // modules. In particular, a router-module wildcard must not accidentally
        // match the lexical gate_proj parameter stem. Keep the original stem for
        // storage/scale validation and projection-target declarations below.
        let owner = module.rsplit_once('.').and_then(|(parent, parameter)| {
            (parent.ends_with(".moe") && matches!(parameter, "gate_proj" | "up_proj" | "down_proj"))
                .then_some(parent)
        }).unwrap_or(module);
        self.declared_numerics_for_owner(module, owner)
    }

    fn declared_numerics_for_owner(&self, module: &str, owner: &str) -> Result<ModuleNumerics> {
        let Some(root) = self.quantization() else { return Ok(ModuleNumerics::CheckpointPrecision) };
        let q = root.get("quantization").unwrap_or(root);
        if root.get("quant_method").and_then(Value::as_str) != Some("modelopt")
            && root.pointer("/producer/name").and_then(Value::as_str) != Some("modelopt") {
            return Ok(ModuleNumerics::CheckpointPrecision);
        }
        if ["ignore", "exclude_modules"].iter().any(|key| q.get(key).and_then(Value::as_array).is_some_and(|rules|
            rules.iter().filter_map(Value::as_str).any(|rule| module_match(rule, owner)))) {
            return Ok(ModuleNumerics::CheckpointPrecision);
        }
        let mut selected = None;
        if let Some(groups) = q.get("config_groups").and_then(Value::as_object) {
            for group in groups.values() {
                let applies = group.get("targets").and_then(Value::as_array).is_some_and(|targets|
                    targets.iter().filter_map(Value::as_str).any(|target|
                        module_match(target, module) || (target == "Linear" &&
                            (module.contains(".experts.") || module.ends_with("_proj") ||
                                module.ends_with(".w1") || module.ends_with(".w2") || module.ends_with(".w3")))));
                if !applies { continue; }
                let w = group.pointer("/weights/num_bits").and_then(Value::as_u64);
                let a = group.pointer("/input_activations/num_bits").and_then(Value::as_u64);
                let numerics = match (w, a) {
                    (Some(4), Some(4)) => {
                        ensure!(group.pointer("/weights/group_size").and_then(Value::as_u64) == Some(16)
                            && group.pointer("/input_activations/group_size").and_then(Value::as_u64) == Some(16),
                            "{module}: W4A4 requires declared K16 scales");
                        ModuleNumerics::W4A4
                    }
                    (Some(8), Some(8)) => ModuleNumerics::W8A8,
                    (Some(8), None) => ModuleNumerics::W8CheckpointActivations,
                    _ => anyhow::bail!("{module}: unsupported declared weight/activation bits {w:?}/{a:?}"),
                };
                ensure!(selected.is_none_or(|old| old == numerics), "{module}: conflicting quantization groups");
                selected = Some(numerics);
            }
        }
        if let Some(numerics) = selected { return Ok(numerics); }
        if let Some(layers) = q.get("quantized_layers").and_then(Value::as_object) {
            let declaration = layers.iter().filter(|(name, _)| module_match(name, module))
                .max_by_key(|(name, _)| name.len()).map(|(_, value)| value);
            if let Some(declaration) = declaration {
                return match declaration.get("quant_algo").and_then(Value::as_str) {
                    Some("NVFP4") => Ok(ModuleNumerics::W4A4),
                    Some("FP8") => Ok(ModuleNumerics::W8A8),
                    other => anyhow::bail!("{module}: unsupported declared quantization {other:?}"),
                };
            }
        }
        // Older standard ModelOpt exports declare a checkpoint-wide algorithm.
        if (module.contains(".experts.") || module.ends_with("_proj"))
            && q.get("quant_algo").and_then(Value::as_str) == Some("NVFP4") {
            return Ok(ModuleNumerics::W4A4);
        }
        Ok(ModuleNumerics::CheckpointPrecision)
    }

    pub fn declared_module_numerics(&self) -> Result<BTreeMap<String, ModuleNumerics>> {
        self.tensor_names().filter_map(|name| name.strip_suffix(".weight"))
            .map(|module| Ok((module.to_owned(), self.module_numerics(module)?))).collect()
    }

    /// Native NVFP4 experts quantize original BF16 activations on the worker.
    /// Use the complete index, never a role's local expert/MTP header subset.
    pub fn spark_input_is_bf16(&self) -> bool {
        let layers = opt_usize_field(self.text_config(), "num_hidden_layers");
        self.tensor_names().any(|name| {
            let Some(stem) = name.strip_suffix(".input_scale") else { return false };
            if !stem.contains(".experts.") || !self.contains_tensor(&format!("{stem}.weight_scale_2")) {
                return false;
            }
            // Legacy fixtures/undeclared checkpoints retain header detection.
            // Declared modules use the same selector as runtime and plan.
            if self.quantization().is_some_and(|root| {
                let q = root.get("quantization").unwrap_or(root);
                q.get("config_groups").is_some() || q.get("quantized_layers").is_some()
                    || q.get("quant_algo").is_some()
            }) && self.module_numerics(stem).ok() != Some(ModuleNumerics::W4A4) {
                return false;
            }
            let parts: Vec<_> = stem.split('.').collect();
            if parts.iter().any(|part| matches!(*part, "mtp" | "nextn")) { return false; }
            let layer = parts.iter().position(|part| *part == "layers")
                .and_then(|at| parts.get(at + 1)).and_then(|value| value.parse::<usize>().ok());
            layer.is_some_and(|layer| layers.is_none_or(|count| layer < count))
        })
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
    fn spark_input_agrees_between_coordinator_and_worker_for_every_family_and_format() {
        for prefix in ["layers.0.ffn", "model.layers.0.mlp", "model.language_model.layers.0.mlp"] {
            for family in ["deepseek_v41", "deepseek_v4", "glm5", "glm5_flash", "qwen4", "mimo_v2"] {
                for format in ["nvfp4", "mxfp4", "fp8", "exl3", "bf16"] {
                    let dir = tempfile::tempdir().unwrap();
                    std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&json!({
                        "model_type": family, "num_hidden_layers": 1
                    })).unwrap()).unwrap();
                    let mut tensors = vec![t("lm_head.weight", "BF16", &[2]),
                        t(format!("{prefix}.experts.0.w1.weight"), "U8", &[2])];
                    if format == "nvfp4" {
                        tensors.extend([t(format!("{prefix}.experts.0.w1.input_scale"), "F32", &[1]),
                            t(format!("{prefix}.experts.0.w1.weight_scale_2"), "F32", &[1])]);
                    }
                    // Dense NVFP4 and a differently quantized MTP never select the backbone wire.
                    for stem in ["model.layers.0.mlp.gate_proj", "model.layers.1.mlp.experts.0.w1",
                        "model.mtp.layers.0.mlp.experts.0.w1", "mtp.0.ffn.experts.0.w1"] {
                        tensors.extend([t(format!("{stem}.input_scale"), "F32", &[1]),
                            t(format!("{stem}.weight_scale_2"), "F32", &[1])]);
                    }
                    write_safetensors(&dir.path().join("model.safetensors"), &tensors);
                    let coordinator = Checkpoint::open_for_role(dir.path(), "coordinator", |name| name == "lm_head.weight").unwrap();
                    let worker = Checkpoint::inventory(dir.path()).unwrap();
                    assert_eq!(coordinator.tensors.len(), 1);
                    assert_eq!(coordinator.spark_input_is_bf16(), format == "nvfp4", "{family}/{format}/{prefix}");
                    assert_eq!(coordinator.spark_input_is_bf16(), worker.spark_input_is_bf16());
                }
            }
        }
    }

    #[test]
    fn declared_numerics_respect_ignore_and_mixed_groups() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&json!({
            "quantization_config": {"quant_method":"modelopt", "ignore":["*.shared_experts.*"],
                "config_groups": {
                    "a4":{"targets":["model.layers.0.mlp.experts"],
                        "weights":{"num_bits":4,"group_size":16},
                        "input_activations":{"num_bits":4,"group_size":16}},
                    "a8":{"targets":["mtp.layers.0.mlp.experts"],
                        "weights":{"num_bits":8},"input_activations":{"num_bits":8}}
                }}
        })).unwrap()).unwrap();
        write_safetensors(&dir.path().join("model.safetensors"), &[t("head.weight", "BF16", &[2])]);
        let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.experts.17.gate_proj.weight").unwrap(), ModuleNumerics::W4A4);
        assert_eq!(checkpoint.module_numerics("mtp.layers.0.mlp.experts.3.down_proj").unwrap(), ModuleNumerics::W8A8);
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.shared_experts.gate_proj").unwrap(), ModuleNumerics::CheckpointPrecision);
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.experts_extra.gate_proj").unwrap(), ModuleNumerics::CheckpointPrecision);
    }

    #[test]
    fn modelopt_hf_metadata_and_legacy_algorithm_respect_exclusions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        write_safetensors(&dir.path().join("model.safetensors"), &[t("head.weight", "BF16", &[2])]);
        std::fs::write(dir.path().join("hf_quant_config.json"), serde_json::to_vec(&json!({
            "producer": {"name":"modelopt"}, "quantization": {
                "quant_algo":"NVFP4", "exclude_modules":["*.shared_experts.*"]
            }
        })).unwrap()).unwrap();
        let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.experts.3.gate_proj").unwrap(), ModuleNumerics::W4A4);
        assert_eq!(checkpoint.module_numerics("model.layers.0.mlp.shared_experts.gate_proj").unwrap(), ModuleNumerics::CheckpointPrecision);
        assert_eq!(checkpoint.module_numerics("head").unwrap(), ModuleNumerics::CheckpointPrecision);
    }

    #[test]
    fn glmf_nextn_bf16_header_preserves_precision_but_partial_nvfp4_fails() {
        // NVIDIA GLM-5.3-Flash-NVFP4 layer 45 is BF16 [2048,4096],
        // absent from both ignore lists despite the class-wide Linear declaration.
        let stem = "model.language_model.layers.45.mlp.experts.65.gate_proj";
        let mut checkpoint = Checkpoint { snapshot:PathBuf::new(), config:Value::Null,
            quantize_config:None, weight_map:BTreeMap::new(), tensors:Vec::new(),
            missing_shards:Vec::new(), shard_bytes:0 };
        checkpoint.config = json!({"text_config":{"num_hidden_layers":45,"num_nextn_predict_layers":1},
            "quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4",
                "ignore":["model.language_model.layers.44.self_attn*"],
                "config_groups":{"group_0":{"targets":["Linear"],
                    "weights":{"num_bits":4,"group_size":16},
                    "input_activations":{"num_bits":4,"group_size":16}}}}});
        checkpoint.tensors = vec![CheckpointTensor { shard:"model-00002-of-00033.safetensors".into(),
            meta: crate::SafetensorsTensorMetadata { name:format!("{stem}.weight"), dtype:cuteafd_core::DType::Bf16,
                shape:vec![2048,4096], byte_offset:0, byte_length:2048*4096*2 } }];
        checkpoint.weight_map.clear();
        assert_eq!(checkpoint.module_numerics(stem).unwrap(), ModuleNumerics::CheckpointPrecision);
        checkpoint.tensors[0].meta.dtype = cuteafd_core::DType::U8;
        checkpoint.tensors[0].meta.shape = vec![2048,2048];
        assert!(checkpoint.module_numerics(stem).unwrap_err().to_string().contains("partial or invalid"));
        checkpoint.tensors[0].meta.dtype = cuteafd_core::DType::Bf16;
        checkpoint.weight_map.insert(format!("{stem}.weight_scale"), "scales.safetensors".into());
        assert!(checkpoint.module_numerics(stem).is_err());
        checkpoint.tensors[0].meta.dtype = cuteafd_core::DType::U8;
        for suffix in ["weight_scale", "weight_scale_2", "input_scale"] {
            checkpoint.weight_map.insert(format!("{stem}.{suffix}"), "scales.safetensors".into());
        }
        assert_eq!(checkpoint.module_numerics(stem).unwrap(), ModuleNumerics::W4A4);
    }

    #[test]
    fn nested_declared_numerics_control_bf16_ingress() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let stem = "model.layers.0.mlp.experts.0.gate_proj";
        write_safetensors(&dir.path().join("model.safetensors"), &[
            t(format!("{stem}.weight"), "U8", &[1]),
            t(format!("{stem}.input_scale"), "F32", &[1]),
            t(format!("{stem}.weight_scale_2"), "F32", &[1]),
        ]);
        for (bits, bf16) in [(4, true), (8, false)] {
            std::fs::write(dir.path().join("hf_quant_config.json"), serde_json::to_vec(&json!({
                "producer":{"name":"modelopt"}, "quantization":{"config_groups":{
                    "experts":{"targets":[stem],
                        "weights":{"num_bits":bits,"group_size":16},
                        "input_activations":{"num_bits":bits,"group_size":16}}
                }}
            })).unwrap()).unwrap();
            assert_eq!(Checkpoint::inventory(dir.path()).unwrap().spark_input_is_bf16(), bf16);
        }
    }

    #[test]
    fn real_nvidia_configs_select_declared_module_numerics() {
        // Optional local inventory gate: no tensor payload reads or GPU needed.
        let Some(root) = std::env::var_os("CUTEAFD_NVFP4_CONFIG_ROOT") else { return };
        let root = Path::new(&root);
        for (model, routed, preserved, a8) in [
            ("GLM-5.3-NVFP4", "model.layers.3.mlp.experts.0.gate_proj", "model.layers.3.mlp.shared_experts.gate_proj", Some("model.layers.0.mlp.gate_proj")),
            ("GLM-5.3-Flash-NVFP4", "model.language_model.layers.3.mlp.experts.0.gate_proj", "model.language_model.layers.3.mlp.shared_experts.gate_proj", None),
            ("DeepSeek-V4.1-Flash-NVFP4", "layers.0.ffn.experts.0.w1", "layers.0.ffn.shared_experts.w1", None),
            ("Qwen3.8-Flash-Next-NVFP4", "model.language_model.layers.0.mlp.experts.0.gate_proj", "model.language_model.layers.0.mlp.shared_expert.gate_proj", Some("mtp.layers.0.mlp.experts.0.gate_proj")),
        ] {
            let snapshots = root.join(format!("models--nvidia--{model}/snapshots"));
            let path = std::fs::read_dir(snapshots).unwrap().map(|entry| entry.unwrap().path())
                .find(|path| path.join("config.json").is_file()).unwrap();
            let checkpoint = Checkpoint::inventory(&path).unwrap();
            assert_eq!(checkpoint.module_numerics(routed).unwrap(), ModuleNumerics::W4A4, "{model}");
            assert_eq!(checkpoint.module_numerics(preserved).unwrap(), ModuleNumerics::CheckpointPrecision, "{model}");
            if let Some(module) = a8 {
                assert_eq!(checkpoint.module_numerics(module).unwrap(), ModuleNumerics::W8A8, "{model}");
            }
            if model == "GLM-5.3-Flash-NVFP4" {
                let nextn = "model.language_model.layers.45.mlp.experts.65.gate_proj";
                assert_eq!(checkpoint.module_numerics(nextn).unwrap(), ModuleNumerics::CheckpointPrecision);
                let header = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{nextn}.weight")).unwrap();
                assert_eq!(header.meta.dtype, cuteafd_core::DType::Bf16);
                assert_eq!(header.meta.shape, [2048,4096]);
            }
            assert!(checkpoint.spark_input_is_bf16(), "{model}");
            let numerics = checkpoint.declared_module_numerics().unwrap();
            assert!(numerics.values().any(|value| *value == ModuleNumerics::W4A4), "{model}");
            // Compare every indexed weight, not just representative modules:
            // legacy per-expert exports must keep exactly the same ignore and
            // quantized sets under module ownership resolution.
            for module in numerics.keys() {
                assert_eq!(checkpoint.declared_numerics(module).unwrap(),
                    checkpoint.declared_numerics_for_owner(module, module).unwrap(),
                    "changed legacy declaration for {model}/{module}");
            }
            let report = super::super::plan(&path, &super::super::PlanOptions {
                placement: super::super::ExpertPlacement::Sparks { ranks: 6 },
                layout: Some(Default::default()), ..Default::default()
            }).unwrap();
            assert!(report.config_error.is_none(), "{model}: {:?}", report.config_error);
            eprintln!("{model}: {} numerics entries unchanged; live planner accepted config", numerics.len());
        }
    }

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
