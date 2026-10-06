//! Qwen 3.8 Flash Next's header-only, biased LayerNorm/GELU tower plan.
use super::*;

impl TowerSpec {
    pub fn qwen(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_reader(File::open(snapshot.join("config.json"))?)?;
        let v = &cfg["vision_config"];
        for (name, want) in [
            ("depth", 27), ("hidden_size", 1152), ("intermediate_size", 4304),
            ("num_heads", 16), ("num_position_embeddings", 2304),
            ("out_hidden_size", 2560), ("patch_size", 16),
            ("temporal_patch_size", 2), ("spatial_merge_size", 2),
        ] {
            if v[name].as_u64() != Some(want) {
                return Err(VisionError::Unsupported(format!(
                    "vision_config.{name}={}; need {want}; add a Qwen tower exporter/kernel", v[name]
                )));
            }
        }
        if cfg["model_type"] != "qwen4_exp"
            || cfg["text_config"]["hidden_size"] != 2560
            || cfg["text_config"]["hc_count"] != 4
            || v["hidden_act"] != "gelu_pytorch_tanh"
            || !v["deepstack_visual_indexes"].as_array().is_some_and(Vec::is_empty)
            || !(1..=4096).contains(&max_tokens)
        {
            return Err(VisionError::Unsupported("Qwen tower config/merger geometry or image capacity".into()));
        }
        let tensors = tower_catalog(snapshot, "model.visual.")?;
        let mut reads = Vec::new();
        let mut cursor = 0usize;
        let mut put = |name: &str, shape: &[usize], vector: bool| -> Result<u64> {
            let full = format!("model.visual.{name}");
            let (path, metadata) = tensors.get(&full)
                .ok_or_else(|| VisionError::Unsupported(format!("missing {full}")))?;
            if metadata.dtype != DType::Bf16 || metadata.shape != shape
                || metadata.byte_length != (shape.iter().product::<usize>() * 2) as u64
            {
                return Err(VisionError::Unsupported(format!(
                    "{full}: {:?} {:?}; need BF16 {shape:?}", metadata.dtype, metadata.shape
                )));
            }
            cursor = (cursor + 255) & !255;
            let destination = cursor;
            cursor += metadata.byte_length as usize * if vector { 2 } else { 1 };
            reads.push(TensorRead { path: path.clone(), metadata: metadata.clone(), destination, vector });
            Ok(destination as u64)
        };
        let mut native = VisionSpec {
            abi_version: 2, reserved: 2, max_tokens: max_tokens as u32, output_width: 2560,
            hidden: 1152, depth: 27, heads: 16, kv_heads: 16, head_dim: 72,
            intermediate: 4304, patch_size: 16, merger_width: 4608, norm_eps: 1e-6,
            merger_extra: [NO_VISION_OFFSET; 8], q_norm: [NO_VISION_OFFSET; 28],
            k_norm: [NO_VISION_OFFSET; 28], ..Default::default()
        };
        native.patch = put("patch_embed.proj.weight", &[1152, 3, 2, 16, 16], false)?;
        native.patch_bias = put("patch_embed.proj.bias", &[1152], true)?;
        // Position table stays BF16 just like the official learned embedding.
        native.pos_embed = put("pos_embed.weight", &[2304, 1152], false)?;
        for i in 0..27 {
            let mut field = |suffix: &str, shape: &[usize], vector: bool| {
                put(&format!("blocks.{i}.{suffix}"), shape, vector)
            };
            native.blocks[i] = VisionBlock {
                qkv: field("attn.qkv.weight", &[3456, 1152], false)?,
                qkv_bias: field("attn.qkv.bias", &[3456], true)?,
                proj: field("attn.proj.weight", &[1152, 1152], false)?,
                proj_bias: field("attn.proj.bias", &[1152], true)?,
                gate_up: field("mlp.linear_fc1.weight", &[4304, 1152], false)?,
                gate_up_bias: field("mlp.linear_fc1.bias", &[4304], true)?,
                down: field("mlp.linear_fc2.weight", &[1152, 4304], false)?,
                down_bias: field("mlp.linear_fc2.bias", &[1152], true)?,
                norm1: field("norm1.weight", &[1152], true)?,
                norm2: field("norm2.weight", &[1152], true)?,
                key0_bias: NO_VISION_OFFSET, window: 0, column_order: 0,
            };
            native.norm1_bias[i] = field("norm1.bias", &[1152], true)?;
            native.norm2_bias[i] = field("norm2.bias", &[1152], true)?;
        }
        native.merger_norm = put("merger.norm.weight", &[1152], true)?;
        native.merger_norm_bias = put("merger.norm.bias", &[1152], true)?;
        native.merger_fc1 = put("merger.linear_fc1.weight", &[4608, 4608], false)?;
        native.merger_fc1_bias = put("merger.linear_fc1.bias", &[4608], true)?;
        native.merger_fc2 = put("merger.linear_fc2.weight", &[2560, 4608], false)?;
        native.merger_fc2_bias = put("merger.linear_fc2.bias", &[2560], true)?;
        drop(put);
        if reads.len() != tensors.len() {
            let known: BTreeSet<_> = reads.iter().map(|r| r.metadata.name.as_str()).collect();
            let unexpected: Vec<_> = tensors.keys().filter(|name| !known.contains(name.as_str())).collect();
            return Err(VisionError::Unsupported(format!("unexpected Qwen tower tensors {unexpected:?}; add a tower exporter/kernel")));
        }
        cursor = (cursor + 255) & !255;
        native.inv_freq = cursor as u64;
        native.weight_bytes = (cursor + 18 * 4) as u64;
        Ok(Self { native, reads })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn snapshot() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({"model_type":"qwen4_exp","text_config":{"hidden_size":2560,"hc_count":4},"vision_config":{
            "depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,
            "num_position_embeddings":2304,"out_hidden_size":2560,"patch_size":16,
            "temporal_patch_size":2,"spatial_merge_size":2,"hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[]}});
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
        let mut end = 0u64;
        let mut header = serde_json::Map::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let start = end;
            end += shape.iter().product::<usize>() as u64 * 2;
            header.insert(format!("model.visual.{name}"), serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,end]}));
        };
        add("patch_embed.proj.weight".into(), vec![1152,3,2,16,16]);
        add("patch_embed.proj.bias".into(), vec![1152]);
        add("pos_embed.weight".into(), vec![2304,1152]);
        for i in 0..27 {
            for (name, shape) in [
                ("attn.qkv.weight",vec![3456,1152]), ("attn.qkv.bias",vec![3456]),
                ("attn.proj.weight",vec![1152,1152]), ("attn.proj.bias",vec![1152]),
                ("mlp.linear_fc1.weight",vec![4304,1152]), ("mlp.linear_fc1.bias",vec![4304]),
                ("mlp.linear_fc2.weight",vec![1152,4304]), ("mlp.linear_fc2.bias",vec![1152]),
                ("norm1.weight",vec![1152]), ("norm1.bias",vec![1152]),
                ("norm2.weight",vec![1152]), ("norm2.bias",vec![1152]),
            ] { add(format!("blocks.{i}.{name}"), shape); }
        }
        for (name, shape) in [
            ("merger.norm.weight",vec![1152]), ("merger.norm.bias",vec![1152]),
            ("merger.linear_fc1.weight",vec![4608,4608]), ("merger.linear_fc1.bias",vec![4608]),
            ("merger.linear_fc2.weight",vec![2560,4608]), ("merger.linear_fc2.bias",vec![2560]),
        ] { add(name.into(), shape); }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut f = File::create(dir.path().join("vision.safetensors")).unwrap();
        f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
        f.set_len(8 + bytes.len() as u64 + end).unwrap();
        let mut map = serde_json::Map::new();
        for name in header.keys() { map.insert(name.clone(), serde_json::json!("vision.safetensors")); }
        map.insert("model.language_model.unread.weight".into(), serde_json::json!("absent-lm.safetensors"));
        std::fs::write(dir.path().join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap()).unwrap();
        dir
    }
    #[test]
    fn qwen_headers_match_biased_head72_tower_without_lm_reads() {
        let dir = snapshot();
        let spec = TowerSpec::from_snapshot(dir.path(), 4096).unwrap();
        assert_eq!(spec.image_family(), cuteafd_loader::media::ImageFamily::Qwen);
        assert_eq!(spec.reads.len(), 333);
        assert_eq!(spec.native.head_dim, 72);
        assert_eq!(spec.native.merger_width, 4608);
        assert_eq!(spec.native.weight_bytes % 256, 72);
        assert_ne!(spec.encoder_id("revision", 120), spec.encoder_id("revision", 121));
        assert!(spec.reads.iter().all(|r| r.destination % 256 == 0));
        assert!(TowerSpec::qwen(dir.path(), 4097).is_err());
        let p = dir.path().join("config.json");
        let mut cfg: serde_json::Value = serde_json::from_reader(File::open(&p).unwrap()).unwrap();
        cfg["vision_config"]["intermediate_size"] = 4305.into();
        std::fs::write(&p, serde_json::to_vec(&cfg).unwrap()).unwrap();
        assert!(TowerSpec::qwen(dir.path(), 4096).unwrap_err().to_string().contains("intermediate_size"));
    }
    #[test]
    fn qwen_rejects_unknown_shards_and_inconsistent_norm_shape() {
        let dir = snapshot();
        let p = dir.path().join("model.safetensors.index.json");
        let mut index: serde_json::Value = serde_json::from_reader(File::open(&p).unwrap()).unwrap();
        index["weight_map"]["model.visual.extra.weight"] = "../outside.safetensors".into();
        std::fs::write(&p, serde_json::to_vec(&index).unwrap()).unwrap();
        assert!(TowerSpec::qwen(dir.path(), 1).unwrap_err().to_string().contains("shard path"));
    }
}
