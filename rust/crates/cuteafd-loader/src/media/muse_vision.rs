//! Header-native Muse vision storage and its resident CUDA arena contract.
use cuteafd_core::DType;
use crate::SafetensorsTensorMetadata;
use serde_json::Value;

pub const MAX_TOKENS: u64 = 4096;

pub fn validate_config(cfg: &Value) -> Result<Vec<u32>, String> {
    let v = &cfg["vision_config"];
    for (field, expected) in [("hidden_size",1536), ("intermediate_size",8960),
        ("num_attention_heads",16), ("num_hidden_layers",50), ("patch_size",14),
        ("patch_temporal",2), ("merge_size",2), ("pos_emb_height",32), ("pos_emb_width",32)] {
        if v[field].as_u64() != Some(expected) {
            return Err(format!("Muse vision_config.{field} needs {expected}; add an exporter/kernel"));
        }
    }
    if cfg["text_config"]["hidden_size"] != 8192 || cfg["out_hidden_size"] != 6144
        || cfg["projector_hidden_size"] != 4096 || cfg["projector_hidden_act"] != "gelu"
        || v["hidden_act"] != "gelu" || v["layer_norm_eps"].as_f64() != Some(1e-5)
        || v["rope_parameters"]["rope_type"] != "axial"
        || v["rope_parameters"]["rope_theta"].as_f64() != Some(10000.0) {
        return Err("unsupported Muse vision numerics/adapter".into());
    }
    let layers = v["layer_types"].as_array().filter(|a| a.len() == 50)
        .ok_or("Muse vision needs 50 attention layer types")?;
    layers.iter().enumerate().map(|(i, kind)| match kind.as_str() {
        Some("window_attention") => Ok(32), Some("full_attention") => Ok(0),
        _ => Err(format!("Muse vision layer {i} attention {kind}")),
    }).collect()
}

/// Packing order matches the native QKV concatenation; no tensor conversion.
pub fn tensor_shapes() -> Vec<(String, Vec<usize>)> {
    let mut out = Vec::with_capacity(809);
    let mut put = |name: &str, shape: &[usize]| out.push((name.to_owned(), shape.to_vec()));
    put("model.vision_tower.patch_embedder.patch_embedding.weight", &[1536,1176]);
    put("model.vision_tower.patch_embedder.position_embedding_table.weight", &[1024,1536]);
    put("model.vision_tower.ln_pre.weight", &[1536]);
    put("model.vision_tower.ln_pre.bias", &[1536]);
    for i in 0..50 {
        let mut field = |suffix: &str, shape: &[usize]| put(&format!("model.vision_tower.layers.{i}.{suffix}"), shape);
        for p in ["q", "k", "v"] { field(&format!("attn.{p}_proj.weight"), &[1536,1536]); }
        for p in ["q", "k", "v"] { field(&format!("attn.{p}_proj.bias"), &[1536]); }
        for (name, shape) in [("attn.proj.weight", vec![1536,1536]), ("attn.proj.bias", vec![1536]),
            ("mlp.fc1.weight", vec![8960,1536]), ("mlp.fc1.bias", vec![8960]),
            ("mlp.fc2.weight", vec![1536,8960]), ("mlp.fc2.bias", vec![1536]),
            ("norm1.weight", vec![1536]), ("norm2.weight", vec![1536]),
            ("norm1.bias", vec![1536]), ("norm2.bias", vec![1536])] { field(name, &shape); }
    }
    put("model.vision_tower.ln_post.weight", &[1536]);
    put("model.vision_tower.ln_post.bias", &[1536]);
    put("model.vision_adapter.fc1.weight", &[4096,6144]);
    put("model.vision_adapter.fc2.weight", &[4096,4096]);
    put("model.vision_projection.weight", &[8192,4096]);
    out
}

pub fn resident_weights<'a>(cfg: &Value, tensors: impl Iterator<Item = &'a SafetensorsTensorMetadata>) -> Result<u64, String> {
    validate_config(cfg)?;
    let tensors: std::collections::BTreeMap<_, _> = tensors.filter(|t| is_tensor(&t.name))
        .map(|t| (t.name.as_str(), t)).collect();
    let expected = tensor_shapes();
    if tensors.len() != expected.len() {
        return Err(format!("Muse vision needs {} tensors, found {}", expected.len(), tensors.len()));
    }
    let mut bytes = 0u64;
    for (name, shape) in expected {
        let t = tensors.get(name.as_str()).ok_or_else(|| format!("missing {name}"))?;
        let extent = (shape.iter().product::<usize>() * 2) as u64;
        if t.dtype != DType::Bf16 || t.shape != shape || t.byte_length != extent {
            return Err(format!("{name} needs BF16 {shape:?}"));
        }
        bytes = bytes.div_ceil(256) * 256 + extent;
    }
    Ok(bytes.div_ceil(256) * 256 + 24 * 4)
}

pub fn is_tensor(name: &str) -> bool {
    ["model.vision_tower.", "model.vision_adapter.", "model.vision_projection."].iter()
        .any(|prefix| name.starts_with(prefix))
}

/// Exactly scratch_muse, including fixed-capacity attention LSE and BLAS workspace.
pub fn scratch_bytes(tokens: u64) -> Result<u64, String> {
    if !(1..=MAX_TOKENS).contains(&tokens) { return Err("Muse vision capacity must be 1..=4096".into()); }
    let n = tokens * 4;
    Ok([n*1536*2, n*1536*2, n*1536*2, n*1176*2, n*4608*2,
        n*1536*2, n*1536*2, n*1536*2, n*1536*2, n*1536*2, n*8960*2,
        tokens*4096*2, tokens*8192*2, n*96*2, n*96*2, n*14*14*3, 768*4,
        n*4, n*4, n*4*4, n*4*4, 513*4, 16384*16*4, 4*1024*1024]
        .into_iter().map(|b| b.div_ceil(256)*256).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Value {
        serde_json::json!({"text_config":{"hidden_size":8192}, "out_hidden_size":6144,
            "projector_hidden_size":4096, "projector_hidden_act":"gelu", "vision_config":{
            "hidden_size":1536, "intermediate_size":8960, "num_attention_heads":16,
            "num_hidden_layers":50, "patch_size":14, "patch_temporal":2, "merge_size":2,
            "pos_emb_height":32, "pos_emb_width":32, "hidden_act":"gelu", "layer_norm_eps":1e-5,
            "rope_parameters":{"rope_type":"axial", "rope_theta":10000.0},
            "layer_types":vec!["window_attention";50]}})
    }
    #[test]
    fn header_native_weights_refuse_missing_extra_or_converted_tensors() {
        let cfg = config();
        let mut headers: Vec<_> = tensor_shapes().into_iter().map(|(name, shape)| {
            SafetensorsTensorMetadata { name, dtype:DType::Bf16,
                byte_length:(shape.iter().product::<usize>()*2) as u64, shape, byte_offset:0 }
        }).collect();
        assert_eq!(resident_weights(&cfg, headers.iter()).unwrap(), 3_856_274_528);
        assert!(resident_weights(&cfg, headers.iter().skip(1)).is_err());
        headers[0].dtype = DType::F32;
        assert!(resident_weights(&cfg, headers.iter()).is_err());
        headers[0].dtype = DType::Bf16;
        let mut extra = headers[0].clone();
        extra.name = "model.vision_adapter.extra.weight".into();
        headers.push(extra);
        assert!(resident_weights(&cfg, headers.iter()).is_err());
        let mut wrong = cfg;
        wrong["vision_config"]["patch_temporal"] = 1.into();
        assert!(validate_config(&wrong).is_err());
    }
    #[test]
    fn native_bf16_storage_and_bounded_arena() {
        let shapes = tensor_shapes();
        assert_eq!(shapes.len(), 809);
        let bytes: usize = shapes.iter().map(|(_, shape)| shape.iter().product::<usize>() * 2).sum();
        assert_eq!(bytes, 3_856_274_432);
        assert_eq!(bytes % 256, 0);
        assert_eq!(scratch_bytes(4096).unwrap(), 1_008_276_736);
        assert!(scratch_bytes(0).is_err());
        assert!(scratch_bytes(4097).is_err());
    }
}
