//! Index-derived role requirements. File inventories never infer missing headers.
use super::{checkpoint::Checkpoint, family, Component, ExpertPlacement};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadRole {
    Coordinator { local_experts: bool, speculator: bool },
    Spark { rank: usize, world: usize },
    Vision,
    Audio,
    Drafter,
}

impl ReadRole {
    pub fn parse(value: &str, placement: ExpertPlacement, speculator: bool) -> Result<Self> {
        match value {
            "coordinator" | "rtx0" | "rtx1" => Ok(Self::Coordinator {
                local_experts: placement == ExpertPlacement::Local, speculator }),
            "vision" => Ok(Self::Vision),
            "audio" => Ok(Self::Audio),
            "drafter" => Ok(Self::Drafter),
            _ => {
                let rank = value.strip_prefix("spark").context("role must be coordinator, rtx0, rtx1, sparkN, vision, audio or drafter")?
                    .parse::<usize>().context("Spark rank must be an integer")?;
                let world = placement.spark_ranks();
                ensure!(rank < world, "spark{rank} is outside TP{world}");
                Ok(Self::Spark { rank, world })
            }
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Coordinator { .. } => "coordinator".into(),
            Self::Spark { rank, world } => format!("spark{rank} (TP{world})"),
            Self::Vision => "vision".into(), Self::Audio => "audio".into(), Self::Drafter => "drafter".into(),
        }
    }
}

pub fn required_tensors(checkpoint: &Checkpoint, role: ReadRole) -> Result<BTreeSet<String>> {
    let family = family::detect(checkpoint).context("checkpoint family is not supported")?;
    let model = family.open(checkpoint)?;
    Ok(checkpoint.tensor_names().filter(|name| {
        let Some(tensor) = family.classify(model.spec(), name) else { return false };
        match role {
            ReadRole::Coordinator { local_experts, speculator } => match tensor.component {
                Component::RoutedExpert => local_experts,
                Component::Speculator | Component::SpeculatorExpert => speculator,
                Component::Vision | Component::Audio => false,
                _ => true,
            },
            // TP slices the intermediate axis; every rank reads each expert's
            // shard, not a disjoint set of experts.
            ReadRole::Spark { .. } => tensor.component == Component::RoutedExpert,
            ReadRole::Vision => tensor.component == Component::Vision,
            ReadRole::Audio => tensor.component == Component::Audio,
            ReadRole::Drafter => matches!(tensor.component, Component::Speculator | Component::SpeculatorExpert),
        }
    }).map(str::to_owned).collect())
}

pub fn open_role(snapshot: &Path, role: ReadRole) -> Result<Checkpoint> {
    let inventory = Checkpoint::inventory(snapshot)?;
    let names = required_tensors(&inventory, role)?;
    Checkpoint::open_required(snapshot, &role.label(), &names)
}

#[derive(Debug, Serialize)]
pub struct RequiredFile {
    pub path: String,
    pub bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct FileManifest {
    pub role: String,
    pub snapshot: String,
    pub repo_id: Option<String>,
    pub revision: Option<String>,
    pub files: Vec<RequiredFile>,
    pub total_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub additional_snapshots: Vec<FileManifest>,
}

pub fn manifest(snapshot: &Path, role: ReadRole) -> Result<FileManifest> {
    manifest_roles(snapshot, &[role])
}

/// A host's file list is the union of the processes it runs, never a shard split.
pub fn manifest_roles(snapshot: &Path, roles: &[ReadRole]) -> Result<FileManifest> {
    ensure!(!roles.is_empty(), "a file inventory needs at least one role");
    let inventory = Checkpoint::inventory(snapshot)?;
    let mut tensors = BTreeSet::new();
    for role in roles { tensors.extend(required_tensors(&inventory, *role)?); }
    let mut files: BTreeSet<String> = tensors.iter().filter_map(|name| inventory.weight_map.get(name).cloned()).collect();
    if roles.iter().any(|r| matches!(r, ReadRole::Drafter | ReadRole::Coordinator { speculator: true, .. })) {
        // Native speculators read calibrated confidence files by numerical key.
        for entry in std::fs::read_dir(snapshot)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.path().is_file() && name.starts_with("draft-confidence.") && name.ends_with(".json") {
                files.insert(name);
            }
        }
    }
    if roles.contains(&ReadRole::Audio) && snapshot.join("audio_tokenizer").is_dir() {
        collect_directory(snapshot, &snapshot.join("audio_tokenizer"), &mut files)?;
    }
    manifest_from_files(snapshot, roles.iter().map(|r| r.label()).collect::<Vec<_>>().join(", "), files)
}

/// Standalone drafters/encoders have their own config and naming convention;
/// their complete index is required, without pretending they are text families.
pub fn manifest_standalone(snapshot: &Path, role: &str) -> Result<FileManifest> {
    // A standalone model's own loaders may open trained vectors or other
    // auxiliary payloads, not just entries in a safetensors weight map.
    let mut files = BTreeSet::new();
    collect_directory(snapshot, snapshot, &mut files)?;
    manifest_from_files(snapshot, role.into(), files)
}

pub fn include_directory(manifest: &mut FileManifest, snapshot: &Path, directory: &str) -> Result<()> {
    let mut files = manifest.files.iter().map(|f| f.path.clone()).collect();
    collect_directory(snapshot, &snapshot.join(directory), &mut files)?;
    *manifest = manifest_from_files(snapshot, manifest.role.clone(), files)?;
    Ok(())
}

fn collect_directory(root: &Path, directory: &Path, files: &mut BTreeSet<String>) -> Result<()> {
    for entry in std::fs::read_dir(directory).with_context(|| format!("inventory {}", directory.display()))? {
        let path = entry?.path();
        if path.is_dir() { collect_directory(root, &path, files)?; }
        else if path.is_file() {
            files.insert(path.strip_prefix(root)?.to_str().context("non-UTF8 inventory path")?.to_owned());
        }
    }
    Ok(())
}

fn manifest_from_files(snapshot: &Path, role: String, mut files: BTreeSet<String>) -> Result<FileManifest> {
    // Named inputs opened by checkpoint, tokenizer, template and processor
    // loaders. Weight shards remain role-filtered by the checkpoint index.
    for name in ["config.json", "model.safetensors.index.json", "generation_config.json",
        "quantize_config.json", "quantization_config.json", "hf_quant_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "tokenizer.model", "spiece.model", "merges.txt", "vocab.txt", "vocab.json",
        "chat_template.jinja", "preprocessor_config.json", "processor_config.json",
        "chat_template.json", "inference/config.json"] {
        if snapshot.join(name).is_file() { files.insert(name.into()); }
    }
    let files: Vec<_> = files.into_iter().map(|path| {
        let bytes = std::fs::metadata(snapshot.join(&path)).ok().filter(|m| m.is_file()).map(|m| m.len());
        RequiredFile { path, bytes }
    }).collect();
    let total_bytes = files.iter().try_fold(0u64, |total, file| total.checked_add(file.bytes?));
    let cache_repo = snapshot.parent().filter(|p| p.file_name().is_some_and(|n| n == "snapshots"))
        .and_then(|p| p.parent());
    let repo_id = cache_repo.and_then(|p| p.file_name()).and_then(|n| n.to_str())
        .and_then(|name| name.strip_prefix("models--")).map(|name| name.replace("--", "/"));
    let revision = cache_repo.and_then(|_| snapshot.file_name()).and_then(|n| n.to_str()).map(str::to_owned);
    Ok(FileManifest { role, snapshot: snapshot.display().to_string(), repo_id, revision, files, total_bytes,
        additional_snapshots: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{mimo_flash_config, mimo_flash_tensors, write_safetensors};

    #[test]
    fn all_family_loader_inputs_are_in_the_role_inventory() {
        use crate::plan::testing::*;
        let v41: serde_json::Value = serde_json::from_str(include_str!("../families/deepseek_v41/official-v41-config.json")).unwrap();
        let mut v4 = v41.clone();
        v4["architectures"] = serde_json::json!(["DeepseekV4ForCausalLM"]);
        v4["model_type"] = serde_json::json!("deepseek_v4");
        // Use real family geometry fixtures; each role's filtered loader must
        // open only shards the manifest lists, including native speculation.
        for config in [mimo_flash_config(), glm5_config(), glm5_flash_config(2), qwen4_config(4), v41, v4] {
            let dir = tempfile::tempdir().unwrap();
            write_snapshot(dir.path(), &config, &[t("lm_head.weight", "BF16", &[2]), t("norm.weight", "BF16", &[2])], None);
            if config["model_type"] == "deepseek_v4" {
                std::fs::create_dir(dir.path().join("inference")).unwrap();
                std::fs::write(dir.path().join("inference/config.json"), serde_json::json!({
                    "vocab_size": 64, "dim": 128, "moe_inter_dim": 128, "n_layers": 2,
                    "n_heads": 2, "n_routed_experts": 4, "n_shared_experts": 1,
                    "n_activated_experts": 2, "score_func": "sqrtsoftplus", "route_scale": 1.5,
                    "swiglu_limit": 10.0, "q_lora_rank": 128, "head_dim": 128, "rope_head_dim": 64,
                    "o_groups": 1, "o_lora_rank": 128, "window_size": 128, "compress_ratios": [0,4],
                    "compress_rope_theta": 160000, "original_seq_len": 65536, "rope_theta": 10000,
                    "rope_factor": 16, "beta_fast": 32, "beta_slow": 1,
                    "index_n_heads": 2, "index_head_dim": 128, "index_topk": 512,
                    "hc_mult": 4, "hc_sinkhorn_iters": 20
                }).to_string()).unwrap();
            }
            for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"] {
                std::fs::write(dir.path().join(name), b"{}").unwrap();
            }
            let roles = [ReadRole::Coordinator { local_experts: true, speculator: true }, ReadRole::Vision, ReadRole::Audio];
            let manifest = manifest_roles(dir.path(), &roles).unwrap();
            let listed: BTreeSet<_> = manifest.files.iter().map(|f| f.path.as_str()).collect();
            for role in roles {
                for tensor in open_role(dir.path(), role).unwrap().tensors {
                    assert!(listed.contains(tensor.shard.as_str()), "loader input missing: {}", tensor.shard);
                }
            }
            for name in ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"] {
                assert!(listed.contains(name), "loader input missing: {name}");
            }
        }
    }

    #[test]
    fn standalone_glm_and_glmf_include_all_auxiliary_inputs() {
        for architecture in ["DFlash2DraftModel", "DSparkDraftModel"] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("config.json"), serde_json::json!({"architectures":[architecture]}).to_string()).unwrap();
            std::fs::write(dir.path().join("model.safetensors"), b"fixture").unwrap();
            std::fs::write(dir.path().join("draft-confidence.fp8-w8a16-r1.json"), b"{}").unwrap();
            let manifest = manifest_standalone(dir.path(), "drafter").unwrap();
            for entry in std::fs::read_dir(dir.path()).unwrap() {
                let name = entry.unwrap().file_name().to_string_lossy().into_owned();
                assert!(manifest.files.iter().any(|f| f.path == name), "loader input missing: {name}");
            }
        }
    }

    #[test]
    fn mimo_coordinator_ignores_expert_and_disabled_tower_shards() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&mimo_flash_config()).unwrap()).unwrap();
        let tensors = mimo_flash_tensors();
        let mut map = BTreeMap::new();
        let mut coordinator = Vec::new();
        let mut experts = Vec::new();
        for tensor in tensors {
            let shard = if tensor.0.contains(".mlp.experts.") {
                experts.push(tensor.clone()); "experts.safetensors"
            } else { coordinator.push(tensor.clone()); "coordinator.safetensors" };
            map.insert(tensor.0, shard);
        }
        map.insert("model.mtp.layers.0.eh_proj.weight".into(), "mtp.safetensors");
        map.insert("model.visual.weight".into(), "vision.safetensors");
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap()).unwrap();
        write_safetensors(&dir.path().join("coordinator.safetensors"), &coordinator);
        let role = ReadRole::Coordinator { local_experts: false, speculator: false };
        let opened = open_role(dir.path(), role).unwrap();
        assert!(opened.contains_tensor("model.mtp.layers.0.eh_proj.weight"));
        let manifest = manifest(dir.path(), role).unwrap();
        assert!(manifest.files.iter().any(|f| f.path == "coordinator.safetensors"));
        assert!(!manifest.files.iter().any(|f| f.path == "experts.safetensors" || f.path == "mtp.safetensors"));
        let mtp = open_role(dir.path(), ReadRole::Coordinator { local_experts: false, speculator: true })
            .unwrap_err().to_string();
        assert!(mtp.contains("model.mtp.layers.0.eh_proj.weight") && mtp.contains("mtp.safetensors"));
        write_safetensors(&dir.path().join("experts.safetensors"), &experts);
        let spark_role = ReadRole::Spark { rank: 0, world: 4 };
        let spark = open_role(dir.path(), spark_role).unwrap();
        assert_eq!(spark.tensors.len(), experts.len());
        let union = manifest_roles(dir.path(), &[role, spark_role]).unwrap();
        let listed: BTreeSet<_> = union.files.iter().map(|f| f.path.as_str()).collect();
        assert!(listed.contains("coordinator.safetensors") && listed.contains("experts.safetensors"));
        assert!(!listed.contains("mtp.safetensors") && !listed.contains("vision.safetensors"));
        for tensor in opened.tensors.iter().chain(&spark.tensors) {
            assert!(listed.contains(tensor.shard.as_str()), "loader opened an unlisted shard");
        }
        assert!(listed.contains("config.json") && listed.contains("model.safetensors.index.json"));
        std::fs::remove_file(dir.path().join("coordinator.safetensors")).unwrap();
        assert!(open_role(dir.path(), ReadRole::Spark { rank: 3, world: 4 }).is_ok());
        let error = open_role(dir.path(), role).unwrap_err().to_string();
        assert!(error.contains("coordinator.safetensors") && error.contains("coordinator"));
    }
}

pub fn shards(checkpoint: &Checkpoint, names: &BTreeSet<String>) -> BTreeMap<String, Vec<String>> {
    let mut result: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in names {
        if let Some(shard) = checkpoint.weight_map.get(name) {
            result.entry(shard.clone()).or_default().push(name.clone());
        }
    }
    result
}
