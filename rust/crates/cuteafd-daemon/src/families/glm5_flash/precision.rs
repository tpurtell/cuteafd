//! Header-only admission of the FP8 block consumers' inputs (MLA, dense and
//! shared-expert projections): checkpoint E4M3 with FP32 128x128 scales, or
//! BF16 quantized to those blocks at load (the FP8 copy is the only resident one).
use cuteafd_core::DType;
use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use cuteafd_loader::plan::checkpoint::Checkpoint;

#[derive(Debug, thiserror::Error)]
pub(crate) enum ProjectionInputError {
    #[error("GLMF checkpoint has no selected projection tensor {name}")]
    Missing { name: String },
    #[error("{name}: GLMF {consumer} reads {expected:?}, found {actual:?}")]
    Dtype { name: String, consumer: &'static str, expected: DType, actual: DType },
    #[error("{name}: GLMF {consumer} reads checkpoint E4M3 with FP32 128x128 scales or BF16 (quantized to those blocks at load), found {actual:?}; add a consumer for this source format")]
    Source { name: String, consumer: &'static str, actual: DType },
    #[error("{name}: GLMF {consumer} quantizes BF16 to 128x128 blocks and needs whole blocks, found {actual:?}")]
    Blocks { name: String, consumer: &'static str, actual: Vec<usize> },
    #[error("{name}: GLMF {consumer} expected shape {expected:?}, found {actual:?}")]
    Shape { name: String, consumer: &'static str, expected: Vec<usize>, actual: Vec<usize> },
    #[error("{name}: GLMF {consumer} expected {expected} bytes, header declares {actual}")]
    Bytes { name: String, consumer: &'static str, expected: u64, actual: u64 },
    #[error("{name}: GLMF projection geometry overflows its storage byte count")]
    Overflow { name: String },
}

fn require(checkpoint: &Checkpoint, name: String, dtype: DType, shape: Vec<usize>,
    width: u64, consumer: &'static str) -> Result<(), ProjectionInputError> {
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.cmp(&name))
        .map_err(|_| ProjectionInputError::Missing { name: name.clone() })?;
    let tensor = &checkpoint.tensors[at].meta;
    if tensor.dtype != dtype {
        return Err(ProjectionInputError::Dtype { name, consumer, expected: dtype, actual: tensor.dtype.clone() });
    }
    if tensor.shape != shape {
        return Err(ProjectionInputError::Shape { name, consumer, expected: shape, actual: tensor.shape.clone() });
    }
    let expected = shape.iter().try_fold(width, |bytes, &extent| bytes.checked_mul(extent as u64))
        .ok_or_else(|| ProjectionInputError::Overflow { name: name.clone() })?;
    if tensor.byte_length != expected {
        return Err(ProjectionInputError::Bytes { name, consumer, expected, actual: tensor.byte_length });
    }
    Ok(())
}

fn block(checkpoint: &Checkpoint, name: String, rows: usize, cols: usize,
    consumer: &'static str) -> Result<(), ProjectionInputError> {
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.cmp(&name))
        .map_err(|_| ProjectionInputError::Missing { name: name.clone() })?;
    match checkpoint.tensors[at].meta.dtype {
        DType::F8E4M3 => {
            require(checkpoint, name.clone(), DType::F8E4M3, vec![rows, cols], 1, consumer)?;
            require(checkpoint, format!("{name}_scale_inv"), DType::F32,
                vec![rows.div_ceil(128), cols.div_ceil(128)], 4, consumer)
        }
        DType::Bf16 => {
            require(checkpoint, name.clone(), DType::Bf16, vec![rows, cols], 2, consumer)?;
            if rows % 128 != 0 || cols % 128 != 0 {
                return Err(ProjectionInputError::Blocks { name, consumer, actual: vec![rows, cols] });
            }
            Ok(())
        }
        ref actual => Err(ProjectionInputError::Source { name, consumer, actual: actual.clone() }),
    }
}

fn scalar(checkpoint: &Checkpoint, name: String) -> Result<(), ProjectionInputError> {
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.cmp(&name))
        .map_err(|_| ProjectionInputError::Missing { name: name.clone() })?;
    // The existing loader accepts scalar or singleton-shaped FP32 storage,
    // with exactly one FP32 value; a vector is not a scale scalar.
    let shape = checkpoint.tensors[at].meta.shape.clone();
    if shape.iter().any(|&extent| extent != 1) {
        return Err(ProjectionInputError::Shape { name, consumer: "NVFP4 dense scalar",
            expected: vec![], actual: shape });
    }
    require(checkpoint, name, DType::F32, shape, 4, "NVFP4 dense MLP")
}

/// The MLA and dense/shared FFN block programs consume one E4M3
/// representation: the checkpoint's own blocks, or BF16 quantized at load.
/// A selected side checkpoint is authoritative, with no fallback to the primary.
/// Existing ModelOpt packed dense MLPs retain their native NVFP4 route.
pub(crate) fn check_projection_inputs(primary: &Checkpoint, fp8_source: Option<&Checkpoint>,
    cfg: &GlmNextConfig, layers: usize) -> Result<(), ProjectionInputError> {
    let source = fp8_source.unwrap_or(primary);
    let h = cfg.hidden;
    for layer in 0..layers.min(cfg.layers) {
        let prefix = format!("model.language_model.layers.{layer}");
        if cfg.attention[layer] == GlmNextAttention::Mla {
            let q_width = cfg.heads.checked_mul(cfg.qk_nope_dim)
                .ok_or_else(|| ProjectionInputError::Overflow { name: format!("{prefix}.self_attn.q_b_proj.weight") })?;
            let v_width = cfg.heads.checked_mul(cfg.v_head_dim)
                .ok_or_else(|| ProjectionInputError::Overflow { name: format!("{prefix}.self_attn.o_proj.weight") })?;
            for (suffix, rows, cols) in [
                ("q_a_proj.weight", cfg.q_lora_rank, h),
                ("kv_a_proj_with_mqa.weight", cfg.kv_lora_rank, h),
                ("q_b_proj.weight", q_width, cfg.q_lora_rank),
                ("o_proj.weight", h, v_width),
            ] {
                block(source, format!("{prefix}.self_attn.{suffix}"), rows, cols, "MLA FP8 block projection")?;
            }
        }
        let dense = cfg.dense[layer];
        let mlp = if dense { format!("{prefix}.mlp") } else { format!("{prefix}.mlp.shared_experts") };
        let inter = if dense { cfg.dense_intermediate } else { cfg.moe_intermediate };
        let packed_dense = dense && primary.tensors.iter().any(|t|
            t.meta.name == format!("{mlp}.gate_proj.weight") && t.meta.dtype == DType::U8);
        for (projection, rows, cols) in [("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)] {
            let name = format!("{mlp}.{projection}");
            if packed_dense {
                require(primary, format!("{name}.weight"), DType::U8, vec![rows, cols / 2], 1, "NVFP4 dense MLP")?;
                require(primary, format!("{name}.weight_scale"), DType::F8E4M3, vec![rows, cols / 16], 1, "NVFP4 dense MLP")?;
                for scalar in ["weight_scale_2", "input_scale"] {
                    self::scalar(primary, format!("{name}.{scalar}"))?;
                }
            } else {
                block(source, format!("{name}.weight"), rows, cols, "dense/shared FP8 block MLP")?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::{plan::checkpoint::CheckpointTensor, SafetensorsTensorMetadata};

    fn input(name: String, dtype: DType, shape: Vec<usize>, width: u64) -> CheckpointTensor {
        CheckpointTensor { shard: "no-payload.safetensors".into(), meta: SafetensorsTensorMetadata {
            name, dtype, byte_offset: 0, byte_length: shape.iter().map(|&n| n as u64).product::<u64>() * width,
            shape,
        } }
    }

    fn fixture() -> (Checkpoint, GlmNextConfig) {
        let cfg = GlmNextConfig {
            vocab_size: 154880, hidden: 4096, layers: 2,
            attention: vec![GlmNextAttention::Kda, GlmNextAttention::Mla], dense: vec![true, false],
            dense_intermediate: 12288, experts: 288, topk: 8, moe_intermediate: 2048,
            routed_scale: 2.5, swiglu_limit: 10.0, rms_norm_eps: 1e-5, hc_mult: 4,
            kda_heads: 64, kda_head_dim: 128, heads: 64, q_lora_rank: 1536, kv_lora_rank: 512,
            qk_nope_dim: 256, v_head_dim: 256, index_topk: 2048, index_kpool: 4, eos: vec![],
        };
        let mut tensors = Vec::new();
        let mut add = |name: String, rows: usize, cols: usize| {
            tensors.push(input(name.clone(), DType::F8E4M3, vec![rows, cols], 1));
            tensors.push(input(format!("{name}_scale_inv"), DType::F32,
                vec![rows.div_ceil(128), cols.div_ceil(128)], 4));
        };
        for (suffix, rows, cols) in [
            ("q_a_proj.weight", 1536, 4096), ("kv_a_proj_with_mqa.weight", 512, 4096),
            ("q_b_proj.weight", 16384, 1536), ("o_proj.weight", 4096, 16384),
        ] { add(format!("model.language_model.layers.1.self_attn.{suffix}"), rows, cols); }
        for (layer, mlp, inter) in [(0, "mlp", 12288), (1, "mlp.shared_experts", 2048)] {
            for (projection, rows, cols) in [("gate_proj", inter, 4096), ("up_proj", inter, 4096),
                ("down_proj", 4096, inter)] {
                add(format!("model.language_model.layers.{layer}.{mlp}.{projection}.weight"), rows, cols);
            }
        }
        tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        (Checkpoint { snapshot: "/no-payload".into(), config: serde_json::Value::Null, quantize_config: None, weight_map: Default::default(),
            tensors, missing_shards: vec![], shard_bytes: 0 }, cfg)
    }

    fn widen_sources(checkpoint: &mut Checkpoint) {
        for tensor in &mut checkpoint.tensors {
            if tensor.meta.dtype == DType::F8E4M3 {
                tensor.meta.dtype = DType::Bf16;
                tensor.meta.byte_length *= 2;
            }
        }
    }

    #[test]
    fn direct_cli_native_fp8_needs_no_conversion_or_side_checkpoint() {
        let (checkpoint, cfg) = fixture();
        check_projection_inputs(&checkpoint, None, &cfg, cfg.layers).unwrap();
    }

    #[test]
    fn direct_cli_bf16_is_admitted_for_block_quantization_at_load() {
        let (mut checkpoint, cfg) = fixture();
        widen_sources(&mut checkpoint);
        check_projection_inputs(&checkpoint, None, &cfg, cfg.layers).unwrap();
    }

    #[test]
    fn other_source_formats_and_partial_bf16_blocks_are_named() {
        let (mut checkpoint, cfg) = fixture();
        widen_sources(&mut checkpoint);
        let name = "model.language_model.layers.0.mlp.gate_proj.weight";
        checkpoint.tensors.iter_mut().find(|t| t.meta.name == name).unwrap().meta.dtype = DType::F16;
        let error = check_projection_inputs(&checkpoint, None, &cfg, cfg.layers).unwrap_err();
        assert!(matches!(error, ProjectionInputError::Source { ref name, actual: DType::F16, .. }
            if name == "model.language_model.layers.0.mlp.gate_proj.weight"), "{error}");
        let (mut checkpoint, mut cfg) = fixture();
        widen_sources(&mut checkpoint);
        cfg.moe_intermediate = 2000;
        for t in &mut checkpoint.tensors {
            if t.meta.name.contains("shared_experts") {
                let s = &mut t.meta.shape;
                *s = s.iter().map(|&n| if n == 2048 { 2000 } else { n }).collect();
                t.meta.byte_length = s.iter().map(|&n| n as u64).product::<u64>() * 2;
            }
        }
        assert!(matches!(check_projection_inputs(&checkpoint, None, &cfg, cfg.layers),
            Err(ProjectionInputError::Blocks { .. })));
    }

    #[test]
    fn explicit_native_source_is_authoritative_without_a_primary_fallback() {
        let (mut primary, cfg) = fixture();
        let (native, _) = fixture();
        widen_sources(&mut primary);
        check_projection_inputs(&primary, Some(&native), &cfg, cfg.layers).unwrap();
        // The selected side checkpoint is read even when the primary is native.
        let (mut selected, _) = fixture();
        for t in &mut selected.tensors {
            if t.meta.dtype == DType::F8E4M3 { t.meta.dtype = DType::F16; t.meta.byte_length *= 2; }
        }
        let (native_primary, _) = fixture();
        assert!(matches!(check_projection_inputs(&native_primary, Some(&selected), &cfg, cfg.layers),
            Err(ProjectionInputError::Source { actual: DType::F16, .. })));
    }

    #[test]
    fn zero_layer_drafter_diagnostics_do_not_admit_unused_target_projections() {
        let (mut checkpoint, cfg) = fixture();
        widen_sources(&mut checkpoint);
        check_projection_inputs(&checkpoint, None, &cfg, 0).unwrap();
    }

    #[test]
    fn native_fp8_requires_named_matching_fp32_scale_headers() {
        let (mut checkpoint, cfg) = fixture();
        let name = "model.language_model.layers.1.self_attn.q_b_proj.weight_scale_inv";
        checkpoint.tensors.iter_mut().find(|t| t.meta.name == name).unwrap().meta.dtype = DType::Bf16;
        assert!(matches!(check_projection_inputs(&checkpoint, None, &cfg, cfg.layers),
            Err(ProjectionInputError::Dtype { ref name, actual: DType::Bf16, .. }) if name.ends_with("_scale_inv")));
        checkpoint.tensors.retain(|t| t.meta.name != name);
        assert!(matches!(check_projection_inputs(&checkpoint, None, &cfg, cfg.layers),
            Err(ProjectionInputError::Missing { name: missing }) if missing == name));
    }

    #[test]
    fn native_projection_rejects_same_size_transpose_or_truncated_storage() {
        let (mut checkpoint, cfg) = fixture();
        let name = "model.language_model.layers.1.self_attn.q_a_proj.weight";
        let tensor = &mut checkpoint.tensors.iter_mut().find(|t| t.meta.name == name).unwrap().meta;
        tensor.shape = vec![4096, 1536];
        assert!(matches!(check_projection_inputs(&checkpoint, None, &cfg, cfg.layers),
            Err(ProjectionInputError::Shape { .. })));
        let (mut checkpoint, _) = fixture();
        checkpoint.tensors.iter_mut().find(|t| t.meta.name == name).unwrap().meta.byte_length -= 1;
        assert!(matches!(check_projection_inputs(&checkpoint, None, &cfg, cfg.layers),
            Err(ProjectionInputError::Bytes { .. })));
    }

    #[test]
    fn existing_nvfp4_dense_uses_primary_checkpoint_even_with_native_fp8_side() {
        let (mut primary, cfg) = fixture();
        let (native, _) = fixture();
        primary.tensors.retain(|t| !t.meta.name.starts_with("model.language_model.layers.0.mlp."));
        for (projection, rows, cols) in [("gate_proj", 12288, 4096), ("up_proj", 12288, 4096),
            ("down_proj", 4096, 12288)] {
            let p = format!("model.language_model.layers.0.mlp.{projection}");
            primary.tensors.push(input(format!("{p}.weight"), DType::U8, vec![rows, cols / 2], 1));
            primary.tensors.push(input(format!("{p}.weight_scale"), DType::F8E4M3, vec![rows, cols / 16], 1));
            primary.tensors.push(input(format!("{p}.weight_scale_2"), DType::F32, vec![1], 4));
            primary.tensors.push(input(format!("{p}.input_scale"), DType::F32, vec![], 4));
        }
        primary.tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        check_projection_inputs(&primary, None, &cfg, cfg.layers).unwrap();
        check_projection_inputs(&primary, Some(&native), &cfg, cfg.layers).unwrap();
    }

    #[test]
    fn nvfp4_scalar_rejects_a_vector_even_when_its_byte_count_matches() {
        let (mut checkpoint, _) = fixture();
        let name = "model.language_model.layers.0.mlp.gate_proj.weight_scale_2";
        checkpoint.tensors.push(input(name.into(), DType::F32, vec![2], 4));
        checkpoint.tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        assert!(matches!(scalar(&checkpoint, name.into()),
            Err(ProjectionInputError::Shape { .. })));
        let tensor = &mut checkpoint.tensors.iter_mut().find(|t| t.meta.name == name).unwrap().meta;
        tensor.shape = vec![1, 1];
        tensor.byte_length = 4;
        scalar(&checkpoint, name.into()).unwrap();
        checkpoint.tensors.iter_mut().find(|t| t.meta.name == name).unwrap().meta.byte_length = 8;
        assert!(matches!(scalar(&checkpoint, name.into()),
            Err(ProjectionInputError::Bytes { expected: 4, .. })));
    }
}
