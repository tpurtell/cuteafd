//! The CPU oracle against the official model: a tiny checkpoint (2 layers,
//! sliding RoPE + full NoPE, window shorter than the prompt, top-2 of 6
//! experts) written and scored by `python/reference/families/muse_spark/
//! engine_fixture.py` through the pinned transformers modules.
use super::block::forward;
use super::reference::{Layer, Matrix, Reference};
use anyhow::{Context, Result};
use cuteafd_core::DType;
use cuteafd_loader::families::muse_spark::MuseSparkConfig;
use std::collections::BTreeMap;

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/muse_spark/fixtures")
}

fn load() -> Result<BTreeMap<String, (Vec<usize>, Vec<f32>)>> {
    let path = fixtures().join("model.safetensors");
    let bytes = std::fs::read(&path)?;
    let mut out = BTreeMap::new();
    for meta in cuteafd_loader::read_safetensors_metadata(&path)? {
        let raw = &bytes[meta.byte_offset as usize..(meta.byte_offset + meta.byte_length) as usize];
        let values: Vec<f32> = match meta.dtype {
            DType::Bf16 => raw.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
            DType::F32 => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            other => anyhow::bail!("{}: unexpected fixture dtype {other:?}", meta.name),
        };
        out.insert(meta.name.clone(), (meta.shape.clone(), values));
    }
    Ok(out)
}

fn reference(cfg: &MuseSparkConfig) -> Result<Reference> {
    let mut tensors = load()?;
    let mut take = |name: &str| tensors.remove(name).with_context(|| format!("fixture lacks {name}"));
    let mut matrix = |name: &str| -> Result<Matrix> {
        let (shape, w) = take(name)?;
        Ok(Matrix { n: shape[0], k: shape[1], w })
    };
    let p = "model.language_model.";
    let embedding = matrix(&format!("{p}embed_tokens.weight"))?;
    let head = matrix("lm_head.weight")?;
    let mut layers = Vec::new();
    for layer in 0..cfg.layers {
        let b = format!("{p}layers.{layer}.");
        let mut vector = |name: &str| -> Result<Vec<f32>> { Ok(tensors_take(&mut tensors, &format!("{b}{name}"))?.1) };
        let input_norm = vector("input_layernorm.weight")?;
        let pre_ffn_norm = vector("pre_feedforward_layernorm.weight")?;
        let post_ffn_norm = vector("post_feedforward_layernorm.weight")?;
        let attention_gate = vector("post_attention_residual_gate.gate")?;
        let ffn_gate = vector("post_feedforward_residual_gate.gate")?;
        let pre_expert_norm = vector("mlp.pre_expert_norm.weight")?;
        let post_expert_norm = vector("mlp.experts.post_expert_norm.weight")?;
        let bias = vector("mlp.gate.e_score_correction_bias")?;
        let mut matrix = |name: &str| -> Result<Matrix> {
            let (shape, w) = tensors_take(&mut tensors, &format!("{b}{name}"))?;
            Ok(Matrix { n: shape[0], k: shape[1], w })
        };
        let (q, k, v, o, g) = (matrix("self_attn.q_proj.weight")?, matrix("self_attn.k_proj.weight")?,
            matrix("self_attn.v_proj.weight")?, matrix("self_attn.o_proj.weight")?, matrix("self_attn.gate_proj.weight")?);
        let (router, pre_expert, post_expert) = (matrix("mlp.gate.weight")?, matrix("mlp.pre_expert_proj.weight")?,
            matrix("mlp.post_expert_proj.weight")?);
        let (gu_shape, gu) = tensors_take(&mut tensors, &format!("{b}mlp.experts.gate_up_proj"))?;
        let (down_shape, down) = tensors_take(&mut tensors, &format!("{b}mlp.experts.down_proj"))?;
        let split = |values: &[f32], shape: &[usize]| values.chunks_exact(shape[1] * shape[2]).map(<[f32]>::to_vec).collect();
        layers.push(Layer { input_norm, pre_ffn_norm, post_ffn_norm, attention_gate, ffn_gate, q, k, v, o, g,
            router, bias, pre_expert, pre_expert_norm, post_expert, post_expert_norm,
            gate_up: split(&gu, &gu_shape), down: split(&down, &down_shape) });
    }
    let norm = tensors_take(&mut tensors, &format!("{p}norm.weight"))?.1;
    Ok(Reference::new(cfg.clone(), embedding, layers, norm, head))
}

fn tensors_take(tensors: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    tensors.remove(name).with_context(|| format!("fixture lacks {name}"))
}

#[test]
fn cpu_oracle_matches_the_official_model_on_a_tiny_checkpoint() -> Result<()> {
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures().join("official.json"))?)?;
    let cfg = MuseSparkConfig::from_hf(&record["config"])?;
    let tokens: Vec<u32> = serde_json::from_value(record["tokens"].clone())?;
    let official: Vec<Vec<f32>> = serde_json::from_value(record["logits"].clone())?;
    let mut ops = reference(&cfg)?;
    let logits = forward(&cfg, &mut ops, &tokens, cfg.layers)?;
    let mut worst = 0f32;
    let mut top1 = 0;
    for (row, (ours, theirs)) in logits.iter().zip(&official).enumerate() {
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        top1 += usize::from(argmax(ours) == argmax(theirs));
        for (a, b) in ours.iter().zip(theirs) { worst = worst.max((a - b).abs()); }
        eprintln!("row {row}: max |diff| {:.4}", ours.iter().zip(theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
    }
    eprintln!("muse oracle vs official: max |logit diff| {worst:.4}, top-1 {top1}/{}", tokens.len());
    // Logits live in [-20, 20] (softcap 20). Measured 0.014: BF16 accumulation
    // order inside the official CPU matmuls; a semantic slip moves logits by > 0.1.
    assert!(worst <= 0.03, "max logit difference {worst}");
    assert_eq!(top1, tokens.len());
    Ok(())
}
