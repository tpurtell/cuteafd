//! Header-only admission for the KDA programs' checkpoint weight inputs.
use cuteafd_core::DType;
use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use cuteafd_loader::plan::checkpoint::Checkpoint;

#[derive(Debug, thiserror::Error)]
pub(crate) enum KdaInputError {
    #[error("GLMF KDA programs support hidden=4096, heads=64, head_dim=128; found hidden={hidden}, heads={heads}, head_dim={head_dim}; add a KDA exporter for this geometry")]
    Geometry {
        hidden: usize,
        heads: usize,
        head_dim: usize,
    },
    #[error("GLMF KDA checkpoint has no tensor {name}")]
    Missing { name: String },
    #[error("{name}: GLMF KDA requires checkpoint BF16 weights, found {dtype:?}; add a checkpoint-native KDA kernel/exporter for this format instead of widening or quantizing it implicitly")]
    Dtype { name: String, dtype: DType },
    #[error("{name}: GLMF KDA expected BF16 shape {expected:?}, found {actual:?}")]
    Shape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("{name}: GLMF KDA BF16 shape requires {expected} bytes, header declares {actual}")]
    Bytes {
        name: String,
        expected: u64,
        actual: u64,
    },
}

/// The KDA programs read the checkpoint's BF16 weights: as-is, or quantized
/// at load to the only resident FP8 in/out projections (--kda-fp8). Native
/// FP8 KDA source weights have no consumer. Validate only the layers this
/// invocation loads, before any native module.
pub(crate) fn check_kda_inputs(
    checkpoint: &Checkpoint,
    cfg: &GlmNextConfig,
    layers: usize,
) -> Result<(), KdaInputError> {
    let active: Vec<usize> = cfg
        .attention
        .iter()
        .take(layers)
        .enumerate()
        .filter_map(|(layer, kind)| (*kind == GlmNextAttention::Kda).then_some(layer))
        .collect();
    if active.is_empty() {
        return Ok(());
    }
    if (cfg.hidden, cfg.kda_heads, cfg.kda_head_dim) != (4096, 64, 128) {
        return Err(KdaInputError::Geometry {
            hidden: cfg.hidden,
            heads: cfg.kda_heads,
            head_dim: cfg.kda_head_dim,
        });
    }
    let (h, d, rank) = (
        cfg.hidden,
        cfg.kda_heads * cfg.kda_head_dim,
        cfg.kda_head_dim,
    );
    for layer in active {
        for (suffix, shape) in [
            ("q_proj.weight", vec![d, h]),
            ("k_proj.weight", vec![d, h]),
            ("v_proj.weight", vec![d, h]),
            ("f_a_proj.weight", vec![rank, h]),
            ("g_a_proj.weight", vec![rank, h]),
            ("b_proj.weight", vec![cfg.kda_heads, h]),
            ("f_b_proj.weight", vec![d, rank]),
            ("g_b_proj.weight", vec![d, rank]),
            ("o_proj.weight", vec![h, d]),
            ("o_norm.weight", vec![rank]),
        ] {
            let name = format!("model.language_model.layers.{layer}.self_attn.{suffix}");
            let at = checkpoint
                .tensors
                .binary_search_by(|t| t.meta.name.cmp(&name))
                .map_err(|_| KdaInputError::Missing { name: name.clone() })?;
            let tensor = &checkpoint.tensors[at].meta;
            if tensor.dtype != DType::Bf16 {
                return Err(KdaInputError::Dtype {
                    name,
                    dtype: tensor.dtype.clone(),
                });
            }
            if tensor.shape != shape {
                return Err(KdaInputError::Shape {
                    name,
                    expected: shape,
                    actual: tensor.shape.clone(),
                });
            }
            let expected = shape.iter().map(|&n| n as u64).product::<u64>() * 2;
            if tensor.byte_length != expected {
                return Err(KdaInputError::Bytes {
                    name,
                    expected,
                    actual: tensor.byte_length,
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::{plan::checkpoint::CheckpointTensor, SafetensorsTensorMetadata};

    fn fixture() -> (Checkpoint, GlmNextConfig) {
        let cfg = GlmNextConfig {
            vocab_size: 154880,
            hidden: 4096,
            layers: 2,
            attention: vec![GlmNextAttention::Kda, GlmNextAttention::Mla],
            dense: vec![false; 2],
            dense_intermediate: 12288,
            experts: 288,
            topk: 8,
            moe_intermediate: 2048,
            routed_scale: 2.5,
            swiglu_limit: 10.0,
            rms_norm_eps: 1e-5,
            hc_mult: 4,
            kda_heads: 64,
            kda_head_dim: 128,
            heads: 64,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_dim: 256,
            v_head_dim: 256,
            index_topk: 2048,
            index_kpool: 4,
            eos: vec![],
        };
        // Actual official Flash KDA header shapes; no payload or GPU is needed.
        let mut tensors: Vec<_> = [
            ("q_proj.weight", vec![8192, 4096]),
            ("k_proj.weight", vec![8192, 4096]),
            ("v_proj.weight", vec![8192, 4096]),
            ("f_a_proj.weight", vec![128, 4096]),
            ("g_a_proj.weight", vec![128, 4096]),
            ("b_proj.weight", vec![64, 4096]),
            ("f_b_proj.weight", vec![8192, 128]),
            ("g_b_proj.weight", vec![8192, 128]),
            ("o_proj.weight", vec![4096, 8192]),
            ("o_norm.weight", vec![128]),
        ]
        .into_iter()
        .map(|(suffix, shape)| CheckpointTensor {
            shard: "unused.safetensors".into(),
            meta: SafetensorsTensorMetadata {
                name: format!("model.language_model.layers.0.self_attn.{suffix}"),
                dtype: DType::Bf16,
                byte_offset: 0,
                byte_length: shape.iter().map(|&n| n as u64).product::<u64>() * 2,
                shape,
            },
        })
        .collect();
        tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        (
            Checkpoint {
                snapshot: "/unused".into(),
                config: serde_json::Value::Null,
                quantize_config: None, weight_map: Default::default(),
                tensors,
                missing_shards: vec![],
                shard_bytes: 0,
            },
            cfg,
        )
    }

    fn tensor<'a>(
        checkpoint: &'a mut Checkpoint,
        suffix: &str,
    ) -> &'a mut SafetensorsTensorMetadata {
        &mut checkpoint
            .tensors
            .iter_mut()
            .find(|t| t.meta.name.ends_with(suffix))
            .unwrap()
            .meta
    }

    #[test]
    fn supported_bf16_headers_need_no_tensor_reads() {
        let (checkpoint, cfg) = fixture();
        check_kda_inputs(&checkpoint, &cfg, cfg.layers).unwrap();
    }

    #[test]
    fn native_fp8_kda_is_named_unsupported_before_widening() {
        let (mut checkpoint, cfg) = fixture();
        tensor(&mut checkpoint, "q_proj.weight").dtype = DType::F8E4M3;
        let error = check_kda_inputs(&checkpoint, &cfg, cfg.layers).unwrap_err();
        assert!(
            matches!(error, KdaInputError::Dtype { ref name, dtype: DType::F8E4M3 } if name.ends_with("q_proj.weight"))
        );
        assert!(error
            .to_string()
            .contains("checkpoint-native KDA kernel/exporter"));
    }

    #[test]
    fn equal_width_f16_cannot_be_reinterpreted_as_bf16() {
        let (mut checkpoint, cfg) = fixture();
        tensor(&mut checkpoint, "o_proj.weight").dtype = DType::F16;
        assert!(matches!(
            check_kda_inputs(&checkpoint, &cfg, cfg.layers),
            Err(KdaInputError::Dtype {
                dtype: DType::F16,
                ..
            })
        ));
    }

    #[test]
    fn transposed_shape_with_equal_bytes_is_rejected() {
        let (mut checkpoint, cfg) = fixture();
        tensor(&mut checkpoint, "q_proj.weight").shape = vec![4096, 8192];
        assert!(matches!(
            check_kda_inputs(&checkpoint, &cfg, cfg.layers),
            Err(KdaInputError::Shape { .. })
        ));
    }

    #[test]
    fn incomplete_bf16_storage_is_rejected() {
        let (mut checkpoint, cfg) = fixture();
        tensor(&mut checkpoint, "o_proj.weight").byte_length -= 2;
        assert!(matches!(
            check_kda_inputs(&checkpoint, &cfg, cfg.layers),
            Err(KdaInputError::Bytes { .. })
        ));
    }

    #[test]
    fn missing_projection_names_the_checkpoint_input() {
        let (mut checkpoint, cfg) = fixture();
        checkpoint
            .tensors
            .retain(|t| !t.meta.name.ends_with("g_b_proj.weight"));
        assert!(matches!(check_kda_inputs(&checkpoint, &cfg, cfg.layers),
            Err(KdaInputError::Missing { name }) if name.ends_with("g_b_proj.weight")));
    }

    #[test]
    fn unsupported_geometry_requires_a_new_exporter() {
        let (checkpoint, mut cfg) = fixture();
        cfg.kda_heads = 32;
        assert!(matches!(
            check_kda_inputs(&checkpoint, &cfg, cfg.layers),
            Err(KdaInputError::Geometry { .. })
        ));
    }

    #[test]
    fn unloaded_layers_do_not_reject_unread_inputs() {
        let (mut checkpoint, cfg) = fixture();
        tensor(&mut checkpoint, "q_proj.weight").dtype = DType::F8E4M3;
        check_kda_inputs(&checkpoint, &cfg, 0).unwrap();
    }
}
