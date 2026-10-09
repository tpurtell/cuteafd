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
    pub files: Vec<RequiredFile>,
    pub total_bytes: Option<u64>,
}

pub fn manifest(snapshot: &Path, role: ReadRole) -> Result<FileManifest> {
    let inventory = Checkpoint::inventory(snapshot)?;
    let tensors = required_tensors(&inventory, role)?;
    let mut files: BTreeSet<String> = tensors.iter().filter_map(|name| inventory.weight_map.get(name).cloned()).collect();
    for entry in std::fs::read_dir(snapshot)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().is_file() && (name.ends_with(".json") || name.ends_with(".jinja")
            || name.ends_with(".model") || name == "merges.txt" || name == "vocab.txt") {
            files.insert(name);
        }
    }
    let files: Vec<_> = files.into_iter().map(|path| {
        let bytes = std::fs::metadata(snapshot.join(&path)).ok().filter(|m| m.is_file()).map(|m| m.len());
        RequiredFile { path, bytes }
    }).collect();
    let total_bytes = files.iter().try_fold(0u64, |total, file| total.checked_add(file.bytes?));
    Ok(FileManifest { role: role.label(), snapshot: snapshot.display().to_string(), files, total_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{mimo_flash_config, mimo_flash_tensors, write_safetensors};

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
        let spark = open_role(dir.path(), ReadRole::Spark { rank: 0, world: 4 }).unwrap();
        assert_eq!(spark.tensors.len(), experts.len());
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
