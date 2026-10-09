//! Measured single-copy defaults for MiMo checkpoints (2026-10-03, one RTX at
//! the natural minimum, checkpoint -> FP8): V2 Flash C4 code 108.5 -> 125.1 tok/s,
//! KL 0.103 -> 0.092; V2.6 Pro C4 71.5 -> 87.3, KL 0.026 -> 0.028. The target head
//! and attention O projections and a BF16 DFlash drafter convert to E4M3 at
//! load into their only resident copy; native FP8 QKV/FFN are unchanged.
//! `--weight-policy checkpoint` and explicit per-weight options take precedence.
use crate::plan::checkpoint::Checkpoint;
use super::MimoV2Config;
use super::projection::{MimoProjectionLayout, MimoProjectionLayoutError, MimoProjectionRepresentation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoDefaultPolicy { Checkpoint, Fp8 }

/// Aggregate physical head/O storage across coordinator ranks; this does not
/// include other target weights, optional drafters, workspaces or caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualifiedProjectionMemory {
    pub source_bytes: u64,
    pub resident_bytes: u64,
    /// One drained source block, not the sum of every converted matrix.
    pub max_load_staging: u64,
}

pub fn qualified_projection_memory(cfg: &MimoV2Config) -> Result<QualifiedProjectionMemory, MimoProjectionLayoutError> {
    let head = |format| MimoProjectionLayout::new(cfg.vocab_size as u64, cfg.hidden as u64, format, 1);
    let output = |format| MimoProjectionLayout::new(cfg.hidden as u64,
        (cfg.heads as u64).checked_mul(cfg.v_head_dim as u64).ok_or(MimoProjectionLayoutError)?, format, 2);
    let source_head = head(MimoProjectionRepresentation::Bf16)?;
    let source_output = output(MimoProjectionRepresentation::Bf16)?;
    let resident_head = head(MimoProjectionRepresentation::Fp8)?;
    let resident_output = output(MimoProjectionRepresentation::Fp8)?;
    let aggregate = |head: MimoProjectionLayout, output: MimoProjectionLayout| {
        let head_bytes = head.resident_bytes()?;
        output.resident_bytes()?.checked_mul(cfg.layers as u64)
            .and_then(|bytes| bytes.checked_add(head_bytes))
            .ok_or(MimoProjectionLayoutError)
    };
    Ok(QualifiedProjectionMemory {
        source_bytes: aggregate(source_head, source_output)?,
        resident_bytes: aggregate(resident_head, resident_output)?,
        max_load_staging: resident_head.max_load_staging.max(resident_output.max_load_staging),
    })
}

/// FP8 when every target head/O matrix has a 128-wide K grid (the single-copy
/// FP8 consumers' layout); otherwise the checkpoint formats.
pub fn default_policy(checkpoint: &Checkpoint, cfg: &MimoV2Config) -> MimoDefaultPolicy {
    // The V2.6 Flash checkpoint has not qualified lossy target conversions.
    if matches!(cfg.program_family().ok(), Some("mimof" | "mimof2")) {
        return MimoDefaultPolicy::Checkpoint;
    }
    let names = std::iter::once("lm_head.weight".to_string())
        .chain((0..cfg.layers).map(|layer| format!("model.layers.{layer}.self_attn.o_proj.weight")));
    for name in names {
        let Ok(index) = checkpoint.tensors.binary_search_by(|tensor| tensor.meta.name.cmp(&name)) else {
            return MimoDefaultPolicy::Checkpoint;
        };
        let shape = &checkpoint.tensors[index].meta.shape;
        if shape.len() != 2 || shape[1] % 128 != 0 { return MimoDefaultPolicy::Checkpoint; }
    }
    MimoDefaultPolicy::Fp8
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::plan::checkpoint::CheckpointTensor;
    use crate::SafetensorsTensorMetadata;
    use cuteafd_core::DType;
    use serde_json::Value;

    pub fn fixture(snapshot: &str) -> (Checkpoint, MimoV2Config, Value, Vec<SafetensorsTensorMetadata>) {
        let value: Value = serde_json::from_str(include_str!("../../../tests/fixtures/mimo-qualified-pro.json")).unwrap();
        let metadata = |key: &str| -> Vec<SafetensorsTensorMetadata> {
            value[key].as_object().unwrap().iter().map(|(name, tensor)| SafetensorsTensorMetadata {
                name: name.clone(), dtype: DType::from_safetensors(tensor["dtype"].as_str().unwrap()),
                shape: tensor["shape"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as usize).collect(),
                byte_offset: 0, byte_length: 0,
            }).collect()
        };
        let mut tensors: Vec<_> = metadata("target_headers").into_iter().map(|meta| CheckpointTensor { shard: "test.safetensors".into(), meta }).collect();
        tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        let cfg = MimoV2Config::from_hf(&value["target_config"]).unwrap();
        let checkpoint = Checkpoint { snapshot: snapshot.into(), config: value["target_config"].clone(),
            quantize_config: None, weight_map: Default::default(), tensors, missing_shards: Vec::new(), shard_bytes: 0 };
        (checkpoint, cfg, value["draft_config"].clone(), metadata("draft_headers"))
    }

    #[test]
    fn mimo_checkpoints_default_to_fp8_head_and_output() {
        for snapshot in ["/arbitrarily-renamed-local-copy", "/hf/models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/rev"] {
            let (checkpoint, cfg, _, _) = fixture(snapshot);
            assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Fp8);
            let spec = crate::plan::families::mimo::spec_from(&cfg, &checkpoint);
            assert!(spec.notes.iter().any(|note| note.contains("default: single-copy FP8")));
        }
    }

    #[test]
    fn flash_mopd_does_not_inherit_qualified_lossy_target_defaults() {
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::write_snapshot(dir.path(), &crate::plan::testing::mimo_flash_mopd_config(),
            &crate::plan::testing::mimo_flash_mopd_tensors(), Some(4));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
        assert_eq!(default_policy(&checkpoint, &cfg.head_split(2).unwrap()), MimoDefaultPolicy::Checkpoint);
    }

    #[test]
    fn selected_projection_report_uses_physical_layout_and_one_loading_stage() {
        let (_, cfg, _, _) = fixture("/local");
        let memory = qualified_projection_memory(&cfg).unwrap();
        assert_eq!(memory.source_bytes, 15_967_715_328);
        assert_eq!(memory.resident_bytes, 8_453_554_176);
        assert_eq!(memory.max_load_staging, 33_554_432);
    }

    #[test]
    fn missing_or_unaligned_projection_keeps_checkpoint_formats() {
        let (mut checkpoint, cfg, _, _) = fixture("/local");
        let at = checkpoint.tensors.iter().position(|t| t.meta.name == "lm_head.weight").unwrap();
        checkpoint.tensors[at].meta.shape[1] = 6100;
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
        checkpoint.tensors.remove(at);
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
    }
}
