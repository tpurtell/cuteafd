//! Mistral Large 4's officially bundled Pixtral tower (consolidated naming);
//! header-only until admission.
//!
//! Official arithmetic (transformers `pixtral` + `mistral_large4`, BF16):
//! patch_conv (no bias) -> ln_pre RMS -> 30 x [attention_norm RMS, wq/wk/wv,
//! axial RoPE, full attention, wo, residual; ffn_norm RMS, SiLU(w1) * w3, w2,
//! residual] -> pre_mm_projector_norm RMS -> 2x2 unfold merger (8192 -> 2048)
//! -> w_in (2048 -> 6144), GELU, w_out (6144 -> 6144). Every RMS norm is
//! Pixtral's (eps 1e-5, rounded to BF16 before gamma); nothing carries a bias.
//!
//! The LM span is Mistral's own layout: per merged row, `[IMG]` x w then
//! `[IMG_BREAK]`; the last token is `[IMG_END]`. The tower writes the
//! checkpoint's own `[IMG_BREAK]`/`[IMG_END]` embedding rows (copied from
//! `tok_embeddings`) into those positions, so the injected span is exactly
//! what the LM's embedding gather would produce for the marker tokens.
use super::{Result, TensorRead, TowerSpec, VisionError};
pub use cuteafd_loader::families::mistral_large4::vision::*;
use cuteafd_core::DType;
use cuteafd_ffi::vision::{VisionBlock, VisionSpec, NO_VISION_OFFSET};
use cuteafd_loader::{read_safetensors_metadata, SafetensorsTensorMetadata};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component, Path, PathBuf},
};

fn catalog(snapshot: &Path) -> Result<BTreeMap<String, (PathBuf, SafetensorsTensorMetadata)>> {
    let index: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("consolidated.safetensors.index.json"))?;
    let map = index["weight_map"].as_object()
        .ok_or_else(|| VisionError::Unsupported("consolidated weight_map absent".into()))?;
    let wanted = |name: &str| ["vision_encoder.", "patch_merger.", "pre_mm_projector_norm.",
        "vision_language_adapter."].iter().any(|p| name.starts_with(p)) || name == "tok_embeddings.weight";
    let mut files = BTreeSet::new();
    for (name, file) in map.iter().filter(|(name, _)| wanted(name)) {
        let file = file.as_str().ok_or_else(|| VisionError::Unsupported(format!("invalid shard for {name}")))?;
        let path = Path::new(file);
        if path.is_absolute() || path.components().any(|c| !matches!(c, Component::Normal(_))) {
            return Err(VisionError::Unsupported(format!("invalid shard path {file}")));
        }
        files.insert(snapshot.join(path));
    }
    let mut tensors = BTreeMap::new();
    for path in files {
        for metadata in read_safetensors_metadata(&path).map_err(|e| VisionError::Unsupported(e.to_string()))? {
            if wanted(&metadata.name) {
                let name = metadata.name.clone();
                if tensors.insert(name.clone(), (path.clone(), metadata)).is_some() {
                    return Err(VisionError::Unsupported(format!("duplicate {name}")));
                }
            }
        }
    }
    Ok(tensors)
}

impl TowerSpec {
    pub fn mistral_large4(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let config: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        validate(&config, max_tokens).map_err(VisionError::Unsupported)?;
        let markers = Markers::from_snapshot(snapshot).map_err(VisionError::Unsupported)?;
        let tensors = catalog(snapshot)?;
        let mut reads = Vec::new();
        let mut cursor = 0usize;
        let mut put = |name: &str, shape: &[usize], vector: bool| -> Result<u64> {
            let (path, metadata) = tensors.get(name).ok_or_else(|| VisionError::Unsupported(format!("missing {name}")))?;
            if metadata.dtype != DType::Bf16 || metadata.shape != shape
                || metadata.byte_length != (shape.iter().product::<usize>() * 2) as u64 {
                return Err(VisionError::Unsupported(format!("{name}: {:?} {:?}; need BF16 {shape:?}",
                    metadata.dtype, metadata.shape)));
            }
            cursor = (cursor + 255) & !255;
            let destination = cursor;
            cursor += metadata.byte_length as usize * if vector { 2 } else { 1 };
            reads.push(TensorRead { path: path.clone(), metadata: metadata.clone(), destination, vector });
            Ok(destination as u64)
        };
        let mut native = VisionSpec {
            abi_version: 2, reserved: RESERVED, max_tokens: max_tokens as u32, output_width: LM_HIDDEN as u32,
            hidden: HIDDEN as u32, heads: HEADS as u32, kv_heads: HEADS as u32, head_dim: HEAD_DIM as u32,
            intermediate: INTERMEDIATE as u32, patch_size: PATCH as u32, merger_width: MERGED as u32, norm_eps: 1e-5,
            patch_bias: NO_VISION_OFFSET, pos_embed: NO_VISION_OFFSET, merger_norm_bias: NO_VISION_OFFSET,
            merger_fc1_bias: NO_VISION_OFFSET, merger_fc2_bias: NO_VISION_OFFSET,
            merger_extra: [NO_VISION_OFFSET; 8],
            ..Default::default()
        }.with_depth(DEPTH);
        for layer in [&mut native.norm1_bias, &mut native.norm2_bias, &mut native.q_norm, &mut native.k_norm] {
            layer.fill(NO_VISION_OFFSET);
        }
        native.patch = put("vision_encoder.patch_conv.weight", &[HIDDEN, 3, PATCH, PATCH], false)?;
        // merger_extra: [ln_pre, merger, w_in, w_out, [IMG_BREAK] row, [IMG_END] row].
        native.merger_extra[0] = put("vision_encoder.ln_pre.weight", &[HIDDEN], true)?;
        for i in 0..DEPTH {
            let p = format!("vision_encoder.transformer.layers.{i}");
            // wq | wk | wv are adjacent so one GEMM produces the fused [q | k | v].
            let qkv = put(&format!("{p}.attention.wq.weight"), &[HIDDEN, HIDDEN], false)?;
            for name in ["wk", "wv"] {
                let at = put(&format!("{p}.attention.{name}.weight"), &[HIDDEN, HIDDEN], false)?;
                debug_assert_eq!(at as usize % 256, 0);
            }
            let proj = put(&format!("{p}.attention.wo.weight"), &[HIDDEN, HIDDEN], false)?;
            // w1 | w3 adjacent: one GEMM produces [gate | up].
            let gate_up = put(&format!("{p}.feed_forward.w1.weight"), &[INTERMEDIATE, HIDDEN], false)?;
            put(&format!("{p}.feed_forward.w3.weight"), &[INTERMEDIATE, HIDDEN], false)?;
            let down = put(&format!("{p}.feed_forward.w2.weight"), &[HIDDEN, INTERMEDIATE], false)?;
            let norm1 = put(&format!("{p}.attention_norm.weight"), &[HIDDEN], true)?;
            let norm2 = put(&format!("{p}.ffn_norm.weight"), &[HIDDEN], true)?;
            native.blocks[i] = VisionBlock { qkv, qkv_bias: NO_VISION_OFFSET, proj, proj_bias: NO_VISION_OFFSET,
                gate_up, gate_up_bias: NO_VISION_OFFSET, down, down_bias: NO_VISION_OFFSET, norm1, norm2,
                key0_bias: NO_VISION_OFFSET, window: 0, column_order: 0 };
        }
        native.merger_norm = put("pre_mm_projector_norm.weight", &[HIDDEN], true)?;
        native.merger_extra[1] = put("patch_merger.merging_layer.weight", &[HIDDEN, MERGED], false)?;
        native.merger_fc1 = put("vision_language_adapter.w_in.weight", &[LM_HIDDEN, HIDDEN], false)?;
        native.merger_fc2 = put("vision_language_adapter.w_out.weight", &[LM_HIDDEN, LM_HIDDEN], false)?;
        drop(put);
        let tower = reads.len();
        if tower != tower_tensors().len()
            || tensors.keys().filter(|name| *name != "tok_embeddings.weight").count() != tower {
            return Err(VisionError::Unsupported("unmapped Mistral vision tensors; update tower loader".into()));
        }
        // The two marker rows of the BF16 embedding table, byte for byte.
        let (path, embedding) = tensors.get("tok_embeddings.weight")
            .ok_or_else(|| VisionError::Unsupported("missing tok_embeddings.weight".into()))?;
        let vocab = *embedding.shape.first().unwrap_or(&0);
        if embedding.dtype != DType::Bf16 || embedding.shape != [vocab, LM_HIDDEN]
            || markers.row_break as usize >= vocab || markers.end as usize >= vocab {
            return Err(VisionError::Unsupported("tok_embeddings must be BF16 [vocab, 6144] holding the markers".into()));
        }
        let row = (LM_HIDDEN * 2) as u64;
        for (slot, id) in [(4, markers.row_break), (5, markers.end)] {
            cursor = (cursor + 255) & !255;
            let mut metadata = embedding.clone();
            metadata.name = format!("tok_embeddings.weight[{id}]");
            metadata.byte_offset += u64::from(id) * row;
            metadata.byte_length = row;
            metadata.shape = vec![LM_HIDDEN];
            native.merger_extra[slot] = cursor as u64;
            reads.push(TensorRead { path: path.clone(), metadata, destination: cursor, vector: false });
            cursor += row as usize;
        }
        cursor = (cursor + 255) & !255;
        native.inv_freq = cursor as u64;
        native.weight_bytes = (cursor + (HEAD_DIM / 2) * 4) as u64;
        debug_assert_eq!(native.weight_bytes, resident_weight_bytes());
        Ok(Self { native, reads })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_plan_reads_only_the_tower_and_two_marker_rows() {
        use cuteafd_loader::plan::testing;
        let dir = tempfile::tempdir().unwrap();
        testing::write_mistral_large4_snapshot(dir.path(), Some(&[1]), true);
        let spec = TowerSpec::from_snapshot(dir.path(), MAX_MERGED_TOKENS).unwrap();
        assert_eq!(spec.native.reserved, RESERVED);
        assert_eq!(spec.reads.len(), tower_tensors().len() + 2);
        assert_eq!(spec.native.weight_bytes, resident_weight_bytes());
        assert!(spec.reads.iter().all(|r| !r.metadata.name.starts_with("layers.")));
        let markers = Markers::from_snapshot(dir.path()).unwrap();
        assert_eq!(markers, Markers { image: 10, row_break: 12, end: 13 });
        let row = |slot: usize| spec.reads.iter().find(|r| r.destination as u64 == spec.native.merger_extra[slot]).unwrap();
        let embedding = spec.reads.iter().find(|r| r.metadata.name.starts_with("tok_embeddings.weight[12]")).unwrap();
        assert_eq!(row(4).metadata.byte_offset, embedding.metadata.byte_offset);
        assert_eq!(row(5).metadata.byte_length, 6144 * 2);
        for (i, block) in spec.native.blocks.iter().enumerate() {
            let at = |name: &str| spec.reads.iter().find(|r| r.metadata.name
                == format!("vision_encoder.transformer.layers.{i}.{name}.weight")).unwrap().destination as u64;
            assert_eq!(block.qkv, at("attention.wq"));
            assert_eq!(at("attention.wk"), block.qkv + 2048 * 2048 * 2);
            assert_eq!(at("attention.wv"), block.qkv + 2 * 2048 * 2048 * 2);
            assert_eq!(at("feed_forward.w3"), block.gate_up + 6144 * 2048 * 2);
        }
        assert_eq!(spec.image_family(), cuteafd_loader::media::ImageFamily::Mistral);
    }
}
