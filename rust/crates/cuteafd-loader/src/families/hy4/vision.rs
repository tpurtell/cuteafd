//! Bundled Hy4 40-layer ViT (`vit.*`) facts from the checkpoint's own config and
//! headers, reusing the shared Hy4 M-RoPE and `smart_resize` geometry (#2550).
//!
//! Geometry (`tencent/Hy4` / `wrldsuksgo2mars/Hy4-EXL3-K4-MC`):
//! - 40 pre-LN SigLIP2 blocks: `hidden=1536`, `heads=16` (`head_dim=96`), `intermediate=6144`
//! - patch-16 raster conv + bias, learned 128x128+1 position table (`scale_factor=(h+0.1)/128`)
//! - separate biased Q/K/V/O, exact GELU MLP, final RMSNorm (`eps=1e-5`)
//! - 2x2/2 conv (`1536 -> 6144`) + GELU + 1x1 conv (`6144 -> 6144`) + `image_newline` (`[6144]`).
use crate::families::hy4_flash::vision as shared_vision;
use crate::SafetensorsTensorMetadata;
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::DType;
use serde_json::Value;
use std::collections::BTreeMap;

pub use shared_vision::{
    smart_resize, span_rows, ImageSpan, PackedTensor, PackedVision, RopePositions, MAX_IMAGE_TOKENS,
    MAX_MERGED_ROWS, MAX_PIXELS, MIN_PIXELS, PATCH, POSITION_EDGE,
};

pub const HEAD_DIM: usize = 96;
pub const HIDDEN: usize = 1536;
pub const LAYERS: usize = 40;
pub const HEADS: usize = 16;
pub const INTERMEDIATE: usize = 6144;
pub const OUTPUT: usize = 6144;
pub const MERGER_HIDDEN: usize = 6144;

/// Parse and validate Hy4's bundled 40-layer ViT contract into the shared
/// `VisionContract` descriptor (identical layout and M-RoPE semantics to Hy4-flash).
pub fn vision_contract(config: &Value) -> Result<shared_vision::VisionContract> {
    let v = config
        .get("vision_config")
        .filter(|v| v.is_object())
        .context("Hy4 checkpoint has no bundled vision_config")?;
    let int = |key: &str| {
        v[key]
            .as_u64()
            .with_context(|| format!("Hy4 vision_config lacks {key}"))
    };
    let fixed = [
        ("hidden_size", HIDDEN as u64),
        ("num_hidden_layers", LAYERS as u64),
        ("num_attention_heads", HEADS as u64),
        ("num_key_value_heads", HEADS as u64),
        ("intermediate_size", INTERMEDIATE as u64),
        ("out_hidden_size", OUTPUT as u64),
        ("patch_size", PATCH as u64),
        ("spatial_merge_size", 2),
        ("temporal_patch_size", 1),
        ("spatial_patch_size", 1),
        ("num_channels", 3),
        ("max_image_size", 2048),
        ("cat_extra_token", 0),
        ("learnable_mlp_pooling_size", 0),
    ];
    for (key, want) in fixed {
        ensure!(
            int(key)? == want,
            "Hy4 vision_config.{key}={}, need {want}; add a tower exporter/kernel",
            v[key]
        );
    }
    let flag = |key: &str, want: bool| {
        let value = v.get(key).and_then(Value::as_bool);
        ensure!(
            value.is_none_or(|value| value == want),
            "Hy4 vision_config.{key} must be {want}"
        );
        Ok(())
    };
    flag("use_2d_rope", false)?;
    flag("use_image_newline_token", true)?;
    flag("perceive_post_norm", false)?;
    flag("vision_full_attention", false)?;
    ensure!(
        v["hidden_act"] == "gelu"
            && v["interpolate_mode"] == "bilinear"
            && v["projector_type"] == "rmsnorm_mlp_adapter",
        "unsupported Hy4 vision activation/interpolation/projector"
    );
    ensure!(
        v.get("perceive_intermediate_size")
            .is_none_or(Value::is_null),
        "Hy4 vision merger hidden width must be 4 x hidden"
    );
    let eps = v["rms_norm_eps"]
        .as_f64()
        .context("Hy4 vision rms_norm_eps")?;
    ensure!(
        eps == 1e-5,
        "Hy4 vision LayerNorm/RMSNorm epsilon {eps}, need 1e-5"
    );
    let text = &config["text_config"];
    ensure!(
        text["hidden_size"] == OUTPUT as u64,
        "Hy4 vision output width differs from text hidden_size"
    );
    let id = |key: &str| {
        config[key]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .with_context(|| format!("Hy4 config lacks {key}"))
    };
    let (image_token, image_start, image_end) = (
        id("image_token_id")?,
        id("image_start_token_id")?,
        id("image_end_token_id")?,
    );
    if let Some(vis_start) = config.get("vision_start_token_id").and_then(Value::as_u64) {
        ensure!(
            vis_start == u64::from(image_start),
            "Hy4 vision/image start ids differ"
        );
    }
    let rope = text
        .get("rope_parameters")
        .or_else(|| config.get("rope_parameters"))
        .filter(|r| r.is_object())
        .context("Hy4 text rope_parameters missing")?;
    ensure!(
        rope["mrope_interleaved"] == true,
        "Hy4 text rope must be interleaved M-RoPE"
    );
    let s: Vec<usize> = serde_json::from_value(rope["mrope_section"].clone())
        .context("Hy4 text mrope_section")?;
    let mrope_section: [usize; 3] = s
        .try_into()
        .map_err(|_| anyhow::anyhow!("Hy4 mrope_section needs three axes"))?;
    ensure!(
        mrope_section == [12, 10, 10] && text["qk_rope_head_dim"] == 64,
        "unsupported Hy4 mrope_section {mrope_section:?}; export an interleaved M-RoPE producer for it"
    );
    Ok(shared_vision::VisionContract {
        hidden: HIDDEN,
        layers: LAYERS,
        heads: HEADS,
        intermediate: INTERMEDIATE,
        output: OUTPUT,
        merger_hidden: MERGER_HIDDEN,
        eps: eps as f32,
        image_token,
        image_start,
        image_end,
        mrope_section,
    })
}

/// Validate every `vit.*` tensor header against `contract` and lay them out 256-byte aligned.
pub fn pack<'a>(
    contract: &shared_vision::VisionContract,
    tensors: impl Iterator<Item = &'a SafetensorsTensorMetadata>,
) -> Result<PackedVision> {
    let mut catalog = BTreeMap::new();
    for tensor in tensors.filter(|t| shared_vision::VisionContract::is_tensor(&t.name)) {
        ensure!(
            catalog.insert(tensor.name.as_str(), tensor).is_none(),
            "duplicate Hy4 vision tensor {}",
            tensor.name
        );
    }
    let expected = contract.tensor_shapes();
    if catalog.len() != expected.len() {
        let known: std::collections::BTreeSet<_> =
            expected.iter().map(|(n, _)| n.as_str()).collect();
        let extra: Vec<_> = catalog.keys().filter(|n| !known.contains(*n)).collect();
        bail!(
            "Hy4 vision needs {} vit.* tensors, found {} (unexpected {extra:?}); add a tower exporter/kernel",
            expected.len(),
            catalog.len()
        );
    }
    let mut packed = PackedVision {
        tensors: Vec::with_capacity(expected.len()),
        arena_bytes: 0,
    };
    for (name, shape) in expected {
        let tensor = *catalog
            .get(name.as_str())
            .with_context(|| format!("missing Hy4 vision tensor {name}"))?;
        let bytes = shape
            .iter()
            .try_fold(2u64, |n, &d| n.checked_mul(d as u64))
            .context("Hy4 vision extent overflow")?;
        ensure!(
            tensor.dtype == DType::Bf16
                && tensor.shape == shape
                && tensor.byte_length == bytes
                && tensor.byte_offset.checked_add(bytes).is_some(),
            "{name}: {:?} {:?}; need BF16 {shape:?}; add a tower exporter/kernel",
            tensor.dtype,
            tensor.shape
        );
        let destination = packed
            .arena_bytes
            .checked_add(255)
            .context("Hy4 vision alignment overflow")?
            / 256
            * 256;
        packed.arena_bytes = destination
            .checked_add(bytes)
            .context("Hy4 vision arena overflow")?;
        packed.tensors.push(PackedTensor {
            metadata: tensor.clone(),
            destination,
        });
    }
    for layer in 0..contract.layers {
        for kind in ["weight", "bias"] {
            let at = |part: &str| {
                packed.offset(&format!("vit.layers.{layer}.self_attn.{part}_proj.{kind}"))
            };
            let width = if kind == "weight" {
                (contract.hidden * contract.hidden * 2) as u64
            } else {
                (contract.hidden * 2) as u64
            };
            let (q, k, v) = (at("q")?, at("k")?, at("v")?);
            ensure!(
                k == q + width && v == k + width,
                "Hy4 layer {layer} Q/K/V {kind}s are not contiguous"
            );
        }
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hy4_config() -> Value {
        serde_json::json!({
            "model_type": "hy_v4_vl",
            "image_token_id": 120120,
            "image_start_token_id": 120118,
            "image_end_token_id": 120119,
            "text_config": {
                "hidden_size": 6144,
                "qk_rope_head_dim": 64,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [12, 10, 10],
                    "rope_theta": 10000000,
                    "rope_type": "default"
                }
            },
            "vision_config": {
                "hidden_size": 1536,
                "num_hidden_layers": 40,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "intermediate_size": 6144,
                "out_hidden_size": 6144,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "spatial_patch_size": 1,
                "num_channels": 3,
                "max_image_size": 2048,
                "cat_extra_token": 0,
                "learnable_mlp_pooling_size": 0,
                "hidden_act": "gelu",
                "interpolate_mode": "bilinear",
                "projector_type": "rmsnorm_mlp_adapter",
                "rms_norm_eps": 1e-5,
                "perceive_post_norm": false,
                "vision_full_attention": false
            }
        })
    }

    #[test]
    fn hy4_40_layer_vision_contract_and_pack() -> Result<()> {
        let c = vision_contract(&hy4_config())?;
        assert_eq!((c.hidden, c.layers, c.heads, c.intermediate, c.output), (1536, 40, 16, 6144, 6144));
        let shapes = c.tensor_shapes();
        assert_eq!(shapes.len(), 649);
        let mut offset = 0u64;
        let metas: Vec<SafetensorsTensorMetadata> = shapes
            .into_iter()
            .map(|(name, shape)| {
                let len = shape.iter().product::<usize>() as u64 * 2;
                let meta = SafetensorsTensorMetadata {
                    name,
                    dtype: DType::Bf16,
                    shape,
                    byte_offset: offset,
                    byte_length: len,
                };
                offset += len;
                meta
            })
            .collect();
        let packed = pack(&c, metas.iter())?;
        assert_eq!(packed.tensors.len(), 649);
        assert!(packed.arena_bytes >= offset);
        Ok(())
    }
}
