//! Startup identity from the checkpoint's own inventories, never local paths.
use super::{CheckpointFormat, MAX_JSON_BYTES};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestIdentity {
    pub name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointIdentity {
    pub fingerprint: String,
    pub manifest: Option<ManifestIdentity>,
}

fn hash_file(hash: &mut Sha256, snapshot: &Path, name: &str, header_only: bool) -> Result<()> {
    let mut file =
        File::open(snapshot.join(name)).with_context(|| format!("checkpoint identity {name}"))?;
    let length = if header_only {
        let mut size = [0; 8];
        file.read_exact(&mut size)?;
        let length = u64::from_le_bytes(size);
        ensure!(
            length
                .checked_add(8)
                .is_some_and(|end| end <= file.metadata().map(|m| m.len()).unwrap_or(0)),
            "truncated checkpoint identity header {name}"
        );
        length
    } else {
        file.metadata()?.len()
    };
    ensure!(
        length <= MAX_JSON_BYTES,
        "checkpoint identity {name} exceeds header limit"
    );
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update(length.to_le_bytes());
    let mut limited = (&mut file).take(length);
    let mut buffer = [0; 64 * 1024];
    let mut read = 0;
    loop {
        let count = limited.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        read += count as u64;
        hash.update(&buffer[..count]);
    }
    ensure!(
        read == length,
        "checkpoint identity {name} changed while reading"
    );
    if !header_only {
        ensure!(
            file.metadata()?.len() == length,
            "checkpoint identity {name} changed while reading"
        );
    }
    Ok(())
}

impl CheckpointIdentity {
    /// Native formats require an explicit selection; an HF publication cannot
    /// silently accept a native manifest. This is called by both bootstrap peers.
    pub fn from_snapshot(snapshot: &Path, manifest: Option<&str>) -> Result<Self> {
        let format = CheckpointFormat::detect(snapshot)?;
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-checkpoint-identity-v2\0");
        hash_file(&mut hash, snapshot, "config.json", false)?;
        let manifest = match format {
            CheckpointFormat::HfIndexed | CheckpointFormat::HfSingle => {
                ensure!(
                    manifest.is_none(),
                    "EXLR8 manifest requires an EXLR8 checkpoint"
                );
                let (name, header) = if format == CheckpointFormat::HfIndexed {
                    (super::index_name(snapshot)?, false)
                } else {
                    ("model.safetensors".to_owned(), true)
                };
                hash_file(&mut hash, snapshot, &name, header)?;
                None
            }
            CheckpointFormat::Exlr8 => {
                let name = manifest.context("EXLR8 bootstrap requires an explicit manifest")?;
                for path in super::super::exlr8::catalog_headers(snapshot)? {
                    let relative = path
                        .strip_prefix(snapshot)?
                        .to_str()
                        .context("non-UTF8 native header path")?;
                    hash_file(&mut hash, snapshot, relative, true)?;
                }
                let (_, sha256) = super::super::exlr8::manifest_identity(snapshot, name)?;
                Some(ManifestIdentity {
                    name: name.into(),
                    sha256,
                })
            }
        };
        Ok(Self {
            fingerprint: format!("{:x}", hash.finalize()),
            manifest,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn header(path: &Path, layer: &str) {
        let bytes = serde_json::to_vec(&json!({"__metadata__": {
            "format":"exlr8-v2", "scheme":"exlr8-e4m3-v2", "layer":layer},
            "signs.in":{"dtype":"I8", "shape":[1], "data_offsets":[0,1]}}))
        .unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.write_all(&[1]).unwrap();
    }

    fn native(root: &Path, reverse: bool) {
        std::fs::create_dir_all(root.join("dense")).unwrap();
        std::fs::create_dir_all(root.join("experts")).unwrap();
        std::fs::create_dir_all(root.join("manifests")).unwrap();
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        let mut names = [
            "dense/exlr8-dense-000.safetensors",
            "experts/exlr8-layer-003.index.safetensors",
        ];
        if reverse {
            names.reverse();
        }
        for name in names {
            header(&root.join(name), "3");
        }
        for name in ["up1", "bit325"] {
            std::fs::write(
                root.join(format!("manifests/exlr8-manifest-{name}.json")),
                serde_json::to_vec(
                    &json!({"format":"exlr8-v2-manifest", "name":name, "set_id":"a".repeat(64)}),
                )
                .unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    fn exlr8_identity_is_path_and_enumeration_independent_but_manifest_bound() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        native(a.path(), false);
        native(b.path(), true);
        let first = CheckpointIdentity::from_snapshot(a.path(), Some("up1")).unwrap();
        assert_eq!(
            first,
            CheckpointIdentity::from_snapshot(b.path(), Some("up1")).unwrap()
        );
        let other = CheckpointIdentity::from_snapshot(a.path(), Some("bit325")).unwrap();
        assert_eq!(first.fingerprint, other.fingerprint);
        assert_ne!(first.manifest, other.manifest);
        let path = a.path().join("manifests/exlr8-manifest-up1.json");
        let mut file = std::fs::OpenOptions::new().append(true).open(path).unwrap();
        file.write_all(b"\n").unwrap();
        let changed = CheckpointIdentity::from_snapshot(a.path(), Some("up1")).unwrap();
        assert_ne!(first.manifest, changed.manifest);
        assert!(CheckpointIdentity::from_snapshot(a.path(), None).is_err());
        assert!(CheckpointIdentity::from_snapshot(a.path(), Some("../up1")).is_err());
        header(&b.path().join("dense/exlr8-dense-000.safetensors"), "4");
        assert_ne!(
            first.fingerprint,
            CheckpointIdentity::from_snapshot(b.path(), Some("up1"))
                .unwrap()
                .fingerprint
        );
    }

    #[test]
    fn exlr8_inventory_keeps_native_names_without_hf_index() {
        let root = tempfile::tempdir().unwrap();
        native(root.path(), false);
        let inventory = super::super::Checkpoint::inventory(root.path()).unwrap();
        assert_eq!(inventory.weight_map.len(), 2);
        assert!(inventory.tensors.is_empty());
        assert!(inventory.contains_tensor("experts/exlr8-layer-003.index.safetensors::signs.in"));
        let open = super::super::Checkpoint::open(root.path()).unwrap();
        assert_eq!(open.tensors.len(), 2);
        assert!(!root.path().join("model.safetensors.index.json").exists());
    }

    #[test]
    fn exlr8_catalog_detects_mtp_storage_and_requires_external_drafter() {
        use crate::plan::spec::DraftOwnership;
        let root = tempfile::tempdir().unwrap();
        native(root.path(), false);
        let mut config = crate::plan::testing::glm5_config();
        config["num_nextn_predict_layers"] = json!(1);
        config
            .as_object_mut()
            .unwrap()
            .remove("quantization_config");
        std::fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        std::fs::remove_file(
            root.path()
                .join("experts/exlr8-layer-003.index.safetensors"),
        )
        .unwrap();
        header(
            &root
                .path()
                .join("experts/exlr8-layer-001.index.safetensors"),
            "1",
        );
        header(
            &root
                .path()
                .join("experts/exlr8-layer-002.index.safetensors"),
            "2",
        );
        let mut catalog = crate::read_expert_catalog(root.path()).unwrap();
        assert_eq!(catalog.format(), CheckpointFormat::Exlr8);
        assert_eq!(catalog.routed_experts().draft_stages, 1);
        assert_eq!(catalog.routed_experts().draft_experts, 256);
        assert_eq!(
            catalog
                .select_draft(DraftOwnership::NativeMtp)
                .unwrap_err()
                .to_string(),
            "EXLR8 MTP drafter not wired; use the default GLM DFlash2 external drafter"
        );
        catalog
            .select_draft(DraftOwnership::selected("glm5", "dflash2"))
            .unwrap();
        assert_eq!(catalog.routed_experts().draft_stages, 0);
        assert_eq!(catalog.routed_experts().draft_experts, 0);
        std::fs::remove_file(
            root.path()
                .join("experts/exlr8-layer-002.index.safetensors"),
        )
        .unwrap();
        assert!(crate::read_expert_catalog(root.path()).is_err());
    }

    #[test]
    fn exlr8_native_bootstrap_mounted_checkpoint_when_supplied() {
        let Some(snapshot) = std::env::var_os("CUTEAFD_EXLR8_TEST_SNAPSHOT") else {
            return;
        };
        let root = Path::new(&snapshot);
        let mut catalog = crate::read_expert_catalog(root).unwrap();
        assert_eq!(catalog.format(), CheckpointFormat::Exlr8);
        assert_eq!(catalog.routed_experts().layers, 78);
        assert_eq!(catalog.routed_experts().draft_stages, 1);
        assert_eq!(catalog.routed_experts().draft_experts, 256);
        catalog
            .select_draft(crate::plan::spec::DraftOwnership::None)
            .unwrap();
        assert_eq!(catalog.routed_experts().draft_stages, 0);
        let up1 = CheckpointIdentity::from_snapshot(root, Some("up1")).unwrap();
        for name in ["bit325", "home"] {
            let other = CheckpointIdentity::from_snapshot(root, Some(name)).unwrap();
            assert_eq!(up1.fingerprint, other.fingerprint);
            assert_ne!(up1.manifest, other.manifest);
        }
    }

    #[test]
    fn hf_single_identity_reads_header_not_payload_and_refuses_native_manifest() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), b"{}").unwrap();
        crate::plan::testing::write_safetensors(
            &root.path().join("model.safetensors"),
            &[crate::plan::testing::t("weight", "BF16", &[2])],
        );
        let first = CheckpointIdentity::from_snapshot(root.path(), None).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(root.path().join("model.safetensors"))
            .unwrap();
        file.write_all(&[1; 8]).unwrap();
        assert_eq!(
            first,
            CheckpointIdentity::from_snapshot(root.path(), None).unwrap()
        );
        assert!(CheckpointIdentity::from_snapshot(root.path(), Some("up1")).is_err());
    }
}
