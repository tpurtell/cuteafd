//! Official Muse BF16 tower packing; no LM tensor payload is read.
use super::*;

impl TowerSpec {
    pub fn muse(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let cfg: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        let v = &cfg["vision_config"];
        for (field, expected) in [("hidden_size",1536), ("intermediate_size",8960),
            ("num_attention_heads",16), ("num_hidden_layers",50), ("patch_size",14),
            ("patch_temporal",2), ("merge_size",2), ("pos_emb_height",32), ("pos_emb_width",32)] {
            if v[field].as_u64() != Some(expected) {
                return Err(VisionError::Unsupported(format!("vision_config.{field}={}; need {expected}; add a Muse exporter/kernel", v[field])));
            }
        }
        let layers = v["layer_types"].as_array().filter(|a| a.len()==50)
            .ok_or_else(|| VisionError::Unsupported("Muse needs 50 attention layer types".into()))?;
        if !(1..=4096).contains(&max_tokens) || cfg["text_config"]["hidden_size"] != 8192
            || cfg["out_hidden_size"] != 6144 || cfg["projector_hidden_size"] != 4096
            || cfg["projector_hidden_act"] != "gelu" || v["hidden_act"] != "gelu"
            || v["layer_norm_eps"].as_f64() != Some(1e-5)
            || v["rope_parameters"]["rope_type"] != "axial"
            || v["rope_parameters"]["rope_theta"].as_f64() != Some(10000.0) {
            return Err(VisionError::Unsupported("unsupported Muse vision numerics/adapter/capacity".into()));
        }
        let mut tensors = tower_catalog(snapshot, "model.vision_tower.")?;
        tensors.extend(tower_catalog(snapshot, "model.vision_adapter.")?);
        tensors.extend(tower_catalog(snapshot, "model.vision_projection.")?);
        let mut reads = Vec::new();
        let mut cursor = 0usize;
        let mut put = |name: &str, shape: &[usize]| -> Result<u64> {
            let (path, metadata) = tensors.get(name)
                .ok_or_else(|| VisionError::Unsupported(format!("missing {name}")))?;
            if metadata.dtype != DType::Bf16 || metadata.shape != shape
                || metadata.byte_length != (shape.iter().product::<usize>() * 2) as u64 {
                return Err(VisionError::Unsupported(format!("{name}: {:?} {:?}; need BF16 {shape:?}", metadata.dtype, metadata.shape)));
            }
            cursor = (cursor+255)&!255;
            let destination = cursor;
            cursor += metadata.byte_length as usize;
            reads.push(TensorRead { path:path.clone(), metadata:metadata.clone(), destination, vector:false });
            Ok(destination as u64)
        };
        let mut native = VisionSpec { abi_version:2, reserved:5, max_tokens:max_tokens as u32,
            output_width:8192, hidden:1536, depth:50, heads:16, kv_heads:16, head_dim:96,
            intermediate:8960, patch_size:14, merger_width:6144, norm_eps:1e-5,
            patch_bias:NO_VISION_OFFSET, merger_fc1_bias:NO_VISION_OFFSET,
            merger_fc2_bias:NO_VISION_OFFSET, merger_extra:[NO_VISION_OFFSET;8],
            ..Default::default() }.with_depth(50);
        native.patch = put("model.vision_tower.patch_embedder.patch_embedding.weight", &[1536,1176])?;
        native.pos_embed = put("model.vision_tower.patch_embedder.position_embedding_table.weight", &[1024,1536])?;
        native.merger_extra[0] = put("model.vision_tower.ln_pre.weight", &[1536])?;
        native.merger_extra[1] = put("model.vision_tower.ln_pre.bias", &[1536])?;
        for (i, kind) in layers.iter().enumerate() {
            let mut field = |suffix: &str, shape: &[usize]| put(&format!("model.vision_tower.layers.{i}.{suffix}"), shape);
            // BF16 Q/K/V matrices and biases are concatenated without changing
            // their native values; each extent is already 256-byte aligned.
            let q = field("attn.q_proj.weight", &[1536,1536])?;
            let k = field("attn.k_proj.weight", &[1536,1536])?;
            let val = field("attn.v_proj.weight", &[1536,1536])?;
            if k != q+1536*1536*2 || val != k+1536*1536*2 {
                return Err(VisionError::Unsupported("Muse QKV packed extent".into()));
            }
            let bias = field("attn.q_proj.bias", &[1536])?;
            let kb = field("attn.k_proj.bias", &[1536])?;
            let vb = field("attn.v_proj.bias", &[1536])?;
            if kb != bias+3072 || vb != kb+3072 {
                return Err(VisionError::Unsupported("Muse QKV bias packed extent".into()));
            }
            native.blocks[i] = VisionBlock { qkv:q, qkv_bias:bias,
                proj:field("attn.proj.weight", &[1536,1536])?, proj_bias:field("attn.proj.bias", &[1536])?,
                gate_up:field("mlp.fc1.weight", &[8960,1536])?, gate_up_bias:field("mlp.fc1.bias", &[8960])?,
                down:field("mlp.fc2.weight", &[1536,8960])?, down_bias:field("mlp.fc2.bias", &[1536])?,
                norm1:field("norm1.weight", &[1536])?, norm2:field("norm2.weight", &[1536])?,
                key0_bias:NO_VISION_OFFSET, window:match kind.as_str() {
                    Some("window_attention")=>32, Some("full_attention")=>0,
                    _=>return Err(VisionError::Unsupported(format!("Muse layer {i} attention {kind}"))),
                }, column_order:0 };
            native.norm1_bias[i] = field("norm1.bias", &[1536])?;
            native.norm2_bias[i] = field("norm2.bias", &[1536])?;
            native.q_norm[i] = NO_VISION_OFFSET;
            native.k_norm[i] = NO_VISION_OFFSET;
        }
        native.merger_norm = put("model.vision_tower.ln_post.weight", &[1536])?;
        native.merger_norm_bias = put("model.vision_tower.ln_post.bias", &[1536])?;
        native.merger_fc1 = put("model.vision_adapter.fc1.weight", &[4096,6144])?;
        native.merger_fc2 = put("model.vision_adapter.fc2.weight", &[4096,4096])?;
        native.merger_extra[2] = put("model.vision_projection.weight", &[8192,4096])?;
        drop(put);
        if reads.len() != tensors.len() {
            let known:BTreeSet<_> = reads.iter().map(|r| r.metadata.name.as_str()).collect();
            let unexpected:Vec<_> = tensors.keys().filter(|n| !known.contains(n.as_str())).collect();
            return Err(VisionError::Unsupported(format!("unexpected Muse tower tensors {unexpected:?}")));
        }
        cursor = (cursor+255)&!255;
        native.inv_freq = cursor as u64;
        native.weight_bytes = (cursor+24*4) as u64;
        Ok(Self { native, reads })
    }
}
