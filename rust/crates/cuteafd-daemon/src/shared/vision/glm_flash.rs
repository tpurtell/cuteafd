//! The officially bundled GLM 5.3 Flash tower; header-only until admission.
use super::{Result, TensorRead, TowerSpec, VisionError};
use cuteafd_core::DType;
use cuteafd_ffi::vision::{VisionBlock, VisionSpec, NO_VISION_OFFSET};
use cuteafd_loader::read_safetensors_metadata;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Component, Path},
};

fn validate(config: &serde_json::Value, max_tokens: usize) -> Result<()> {
    let v = &config["vision_config"];
    for (name, expected) in [
        ("depth", 24),
        ("hidden_size", 1024),
        ("intermediate_size", 4096),
        ("num_heads", 16),
        ("out_hidden_size", 4096),
        ("patch_size", 14),
        ("temporal_patch_size", 2),
        ("spatial_merge_size", 2),
        ("projection_intermediate_size", 10240),
        ("in_channels", 3),
    ] {
        if v[name].as_u64() != Some(expected) {
            return Err(VisionError::Unsupported(format!(
                "vision_config.{name}={}; need {expected}; add a GLM tower kernel",
                v[name]
            )));
        }
    }
    if config["model_type"] != "glm5_next"
        || config["text_config"]["hidden_size"] != 4096
        || v["attention_bias"] != true
        || v["hidden_act"] != "silu"
        || v["rms_norm_eps"].as_f64() != Some(1e-5)
        || v["swiglu_limit"].as_f64() != Some(10.0)
        || max_tokens == 0
        || max_tokens > 4096
    {
        return Err(VisionError::Unsupported(
            "GLM Flash vision/LM geometry or image capacity".into(),
        ));
    }
    if let Some(rope) = v.get("rope_parameters") {
        if rope["rope_type"] != "axial" || rope["rope_theta"].as_f64() != Some(10000.0) {
            return Err(VisionError::Unsupported(
                "GLM tower needs axial RoPE theta=10000".into(),
            ));
        }
    }
    Ok(())
}

impl TowerSpec {
    pub fn glm_flash(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_reader(File::open(snapshot.join("config.json"))?)?;
        validate(&config, max_tokens)?;
        let index = snapshot.join("model.safetensors.index.json");
        let mut files = BTreeSet::new();
        if index.exists() {
            let index: serde_json::Value = serde_json::from_reader(File::open(index)?)?;
            let map = index["weight_map"]
                .as_object()
                .ok_or_else(|| VisionError::Unsupported("safetensors weight_map absent".into()))?;
            for (name, file) in map {
                if !name.starts_with("model.visual.") {
                    continue;
                }
                let file = file
                    .as_str()
                    .ok_or_else(|| VisionError::Unsupported(format!("invalid shard for {name}")))?;
                let path = Path::new(file);
                if path.is_absolute()
                    || path
                        .components()
                        .any(|c| !matches!(c, Component::Normal(_)))
                {
                    return Err(VisionError::Unsupported(format!(
                        "invalid shard path {file}"
                    )));
                }
                files.insert(snapshot.join(path));
            }
        } else {
            for entry in std::fs::read_dir(snapshot)? {
                let path = entry?.path();
                if path.extension().is_some_and(|s| s == "safetensors") {
                    files.insert(path);
                }
            }
        }
        let mut tensors = BTreeMap::new();
        for path in files {
            for metadata in read_safetensors_metadata(&path)
                .map_err(|e| VisionError::Unsupported(e.to_string()))?
            {
                if metadata.name.starts_with("model.visual.") {
                    let name = metadata.name.clone();
                    if tensors
                        .insert(name.clone(), (path.clone(), metadata))
                        .is_some()
                    {
                        return Err(VisionError::Unsupported(format!("duplicate {name}")));
                    }
                }
            }
        }
        let mut reads = Vec::new();
        let mut cursor = 0usize;
        let mut put = |name: &str, shape: &[usize], vector: bool, align: bool| -> Result<u64> {
            let full = format!("model.visual.{name}");
            let (path, metadata) = tensors
                .get(&full)
                .ok_or_else(|| VisionError::Unsupported(format!("missing {full}")))?;
            if metadata.dtype != DType::Bf16
                || metadata.shape != shape
                || metadata.byte_length != (shape.iter().product::<usize>() * 2) as u64
            {
                return Err(VisionError::Unsupported(format!(
                    "{full}: {:?} {:?}; need BF16 {shape:?}",
                    metadata.dtype, metadata.shape
                )));
            }
            if align {
                cursor = (cursor + 255) & !255;
            }
            let destination = cursor;
            cursor += metadata.byte_length as usize * if vector { 2 } else { 1 };
            reads.push(TensorRead {
                path: path.clone(),
                metadata: metadata.clone(),
                destination,
                vector,
            });
            Ok(destination as u64)
        };
        let mut native = VisionSpec {
            abi_version: 2,
            reserved: 3,
            max_tokens: max_tokens as u32,
            output_width: 4096,
            hidden: 1024,
            depth: 24,
            heads: 16,
            kv_heads: 16,
            head_dim: 64,
            intermediate: 4096,
            patch_size: 14,
            merger_width: 4096,
            norm_eps: 1e-5,
            patch_bias: NO_VISION_OFFSET,
            pos_embed: NO_VISION_OFFSET,
            merger_norm_bias: NO_VISION_OFFSET,
            merger_fc1_bias: NO_VISION_OFFSET,
            merger_fc2_bias: NO_VISION_OFFSET,
            merger_extra: [NO_VISION_OFFSET; 8],
            norm1_bias: [NO_VISION_OFFSET; 28],
            norm2_bias: [NO_VISION_OFFSET; 28],
            q_norm: [NO_VISION_OFFSET; 28],
            k_norm: [NO_VISION_OFFSET; 28],
            ..Default::default()
        };
        native.patch = put(
            "patch_embed.proj.weight",
            &[1024, 3, 2, 14, 14],
            false,
            true,
        )?;
        native.patch_bias = put("patch_embed.proj.bias", &[1024], true, true)?;
        for i in 0..24 {
            let mut field = |suffix: &str, shape: &[usize], vector: bool, align: bool| {
                put(&format!("blocks.{i}.{suffix}"), shape, vector, align)
            };
            let block = &mut native.blocks[i];
            *block = VisionBlock {
                qkv: field("attn.qkv.weight", &[3072, 1024], false, true)?,
                qkv_bias: field("attn.qkv.bias", &[3072], true, true)?,
                proj: field("attn.proj.weight", &[1024, 1024], false, true)?,
                proj_bias: field("attn.proj.bias", &[1024], true, true)?,
                gate_up: field("mlp.gate_proj.weight", &[4096, 1024], false, true)?,
                key0_bias: NO_VISION_OFFSET,
                ..Default::default()
            };
            field("mlp.up_proj.weight", &[4096, 1024], false, false)?;
            block.gate_up_bias = field("mlp.gate_proj.bias", &[4096], true, true)?;
            field("mlp.up_proj.bias", &[4096], true, false)?;
            block.down = field("mlp.down_proj.weight", &[1024, 4096], false, true)?;
            block.down_bias = field("mlp.down_proj.bias", &[1024], true, true)?;
            block.norm1 = field("norm1.weight", &[1024], true, true)?;
            block.norm2 = field("norm2.weight", &[1024], true, true)?;
            native.q_norm[i] = field("attn.q_norm.weight", &[64], true, true)?;
            native.k_norm[i] = field("attn.k_norm.weight", &[64], true, true)?;
        }
        native.merger_norm = put("post_layernorm.weight", &[1024], true, true)?;
        native.merger_extra[0] = put("downsample.weight", &[4096, 1024, 2, 2], false, true)?;
        native.merger_extra[1] = put("downsample.bias", &[4096], true, true)?;
        native.merger_extra[2] = put("merger.proj.weight", &[4096, 4096], false, true)?;
        native.merger_extra[3] = put("merger.post_projection_norm.weight", &[4096], true, true)?;
        native.merger_norm_bias = put("merger.post_projection_norm.bias", &[4096], true, true)?;
        native.merger_extra[4] = put("merger.gate_proj.weight", &[10240, 4096], false, true)?;
        put("merger.up_proj.weight", &[10240, 4096], false, false)?;
        native.merger_extra[6] = put("merger.down_proj.weight", &[4096, 10240], false, true)?;
        drop(put);
        if reads.len() != tensors.len() {
            return Err(VisionError::Unsupported(
                "unmapped GLM vision tensors; update tower loader".into(),
            ));
        }
        cursor = (cursor + 255) & !255;
        native.inv_freq = cursor as u64;
        native.weight_bytes = (cursor + 64) as u64;
        Ok(Self { native, reads })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> serde_json::Value {
        serde_json::json!({"model_type":"glm5_next","text_config":{"hidden_size":4096},"vision_config":{
            "depth":24,"hidden_size":1024,"intermediate_size":4096,"num_heads":16,
            "out_hidden_size":4096,"patch_size":14,"temporal_patch_size":2,"spatial_merge_size":2,
            "projection_intermediate_size":10240,"in_channels":3,"attention_bias":true,
            "hidden_act":"silu","rms_norm_eps":1e-5,"swiglu_limit":10.0}})
    }
    #[test]
    fn rejects_unknown_geometry_before_tensor_reads() {
        assert!(validate(&config(), 4096).is_ok());
        for cap in [0, 4097] {
            assert!(validate(&config(), cap).is_err());
        }
        for (name, value) in [
            ("num_heads", serde_json::json!(8)),
            ("patch_size", serde_json::json!(16)),
            ("rms_norm_eps", serde_json::json!(1e-6)),
            ("swiglu_limit", serde_json::json!(0.0)),
        ] {
            let mut cfg = config();
            cfg["vision_config"][name] = value;
            assert!(validate(&cfg, 4096).is_err(), "{name}");
        }
    }
    #[test]
    #[ignore = "requires an installed official GLM HF snapshot"]
    fn installed_snapshot_is_header_only_and_has_347_tensors() {
        let path = std::env::var("CUTEAFD_GLM_VISION_SNAPSHOT").expect("snapshot path");
        let spec = TowerSpec::from_snapshot(Path::new(&path), 4096).unwrap();
        assert_eq!(spec.reads.len(), 347);
        assert_eq!(spec.native.hidden, 1024);
        assert_eq!(spec.native.output_width, 4096);
        assert_eq!(spec.native.blocks[0].key0_bias, NO_VISION_OFFSET);
        assert_eq!(
            spec.image_family(),
            cuteafd_loader::media::ImageFamily::GlmFlash
        );
    }
    #[test]
    fn header_plan_fuses_gate_up_and_does_not_open_text_shards() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config()).unwrap(),
        )
        .unwrap();
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        let mut add = |name: String, shape: Vec<usize>| {
            let start = end;
            end += shape.iter().product::<usize>() as u64 * 2;
            header.insert(
                format!("model.visual.{name}"),
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,end]}),
            );
        };
        add("patch_embed.proj.weight".into(), vec![1024, 3, 2, 14, 14]);
        add("patch_embed.proj.bias".into(), vec![1024]);
        for i in 0..24 {
            for (name, shape) in [
                ("attn.qkv.weight", vec![3072, 1024]),
                ("attn.qkv.bias", vec![3072]),
                ("attn.proj.weight", vec![1024, 1024]),
                ("attn.proj.bias", vec![1024]),
                ("attn.q_norm.weight", vec![64]),
                ("attn.k_norm.weight", vec![64]),
                ("norm1.weight", vec![1024]),
                ("norm2.weight", vec![1024]),
                ("mlp.gate_proj.weight", vec![4096, 1024]),
                ("mlp.up_proj.weight", vec![4096, 1024]),
                ("mlp.gate_proj.bias", vec![4096]),
                ("mlp.up_proj.bias", vec![4096]),
                ("mlp.down_proj.weight", vec![1024, 4096]),
                ("mlp.down_proj.bias", vec![1024]),
            ] {
                add(format!("blocks.{i}.{name}"), shape);
            }
        }
        for (name, shape) in [
            ("post_layernorm.weight", vec![1024]),
            ("downsample.weight", vec![4096, 1024, 2, 2]),
            ("downsample.bias", vec![4096]),
            ("merger.proj.weight", vec![4096, 4096]),
            ("merger.post_projection_norm.weight", vec![4096]),
            ("merger.post_projection_norm.bias", vec![4096]),
            ("merger.gate_proj.weight", vec![10240, 4096]),
            ("merger.up_proj.weight", vec![10240, 4096]),
            ("merger.down_proj.weight", vec![4096, 10240]),
        ] {
            add(name.into(), shape);
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(dir.path().join("vision.safetensors")).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + end).unwrap();
        let mut map = serde_json::Map::new();
        for name in header.keys() {
            map.insert(name.clone(), serde_json::json!("vision.safetensors"));
        }
        map.insert(
            "model.language_model.layers.0.unread".into(),
            serde_json::json!("missing-text.safetensors"),
        );
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap(),
        )
        .unwrap();
        let spec = TowerSpec::from_snapshot(dir.path(), 4096).unwrap();
        assert_eq!(
            spec.image_family(),
            cuteafd_loader::media::ImageFamily::GlmFlash
        );
        assert_eq!(spec.reads.len(), 347);
        assert_eq!(spec.native.reserved, 3);
        for i in 0..24 {
            let read = |suffix| {
                spec.reads
                    .iter()
                    .find(|r| r.metadata.name == format!("model.visual.blocks.{i}.mlp.{suffix}"))
                    .unwrap()
            };
            let gate = read("gate_proj.weight");
            let up = read("up_proj.weight");
            assert_eq!(
                up.destination,
                gate.destination + gate.metadata.byte_length as usize
            );
            assert_eq!(spec.native.blocks[i].gate_up, gate.destination as u64);
        }
        assert!(spec.native.weight_bytes > end);
    }
}
