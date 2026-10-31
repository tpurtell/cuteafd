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
    let ulps = logits.iter().zip(&official).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| {
        let ulp = f32::from_bits(y.abs().max(1e-30).to_bits() & 0x7f80_0000) * 2f32.powi(-7);
        ((x - y).abs() / ulp).round() as usize
    })).fold(vec![0usize; 8], |mut h, u| { h[u.min(7)] += 1; h });
    eprintln!("muse oracle vs official: logit diff in BF16 ulps histogram [0..6, 7+] {ulps:?}");
    // Logits span about +-10 at this geometry (hidden 512, head_dim 64). The
    // official CPU eager model accumulates its BF16 linears in its own order
    // (measured 0.246 max, most within 6 BF16 ulps); a semantic slip (a missed
    // norm, a wrong rounding point or scale) moves logits by whole units.
    assert!(worst <= 0.4, "max logit difference {worst}");
    assert_eq!(top1, tokens.len());
    Ok(())
}

/// Online (flash-style) softmax attention, the native kernels' arithmetic
/// class, still agrees with the official model's top-1 everywhere.
#[test]
fn online_softmax_attention_stays_within_the_stated_tolerance() -> Result<()> {
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures().join("official.json"))?)?;
    let cfg = MuseSparkConfig::from_hf(&record["config"])?;
    let tokens: Vec<u32> = serde_json::from_value(record["tokens"].clone())?;
    let official: Vec<Vec<f32>> = serde_json::from_value(record["logits"].clone())?;
    for tile in [1, 2, 4, 64] {
        let mut ops = reference(&cfg)?;
        ops.online_tile = Some(tile);
        let logits = forward(&cfg, &mut ops, &tokens, cfg.layers)?;
        let (worst, top1) = compare(&logits, &official);
        eprintln!("online softmax tile {tile}: max |logit diff| {worst:.4}, top-1 {top1}/{}", tokens.len());
        assert!(worst <= 0.4, "tile {tile}: {worst}");
        assert_eq!(top1, tokens.len());
    }
    Ok(())
}

fn fixture_logits(ops: &mut Reference, cfg: &MuseSparkConfig) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<u32>)> {
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures().join("official.json"))?)?;
    let tokens: Vec<u32> = serde_json::from_value(record["tokens"].clone())?;
    let official: Vec<Vec<f32>> = serde_json::from_value(record["logits"].clone())?;
    Ok((forward(cfg, ops, &tokens, cfg.layers)?, official, tokens))
}

fn compare(ours: &[Vec<f32>], official: &[Vec<f32>]) -> (f32, usize) {
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let top1 = ours.iter().zip(official).filter(|(a, b)| argmax(a) == argmax(b)).count();
    let worst = ours.iter().zip(official).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0f32, f32::max);
    (worst, top1)
}

/// The exchange is lossless: injecting the oracle's own dumped attention
/// reproduces its logits bit for bit.
#[test]
fn attention_dump_then_inject_reproduces_the_oracle() -> Result<()> {
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures().join("official.json"))?)?;
    let cfg = MuseSparkConfig::from_hf(&record["config"])?;
    let dir = tempfile::tempdir()?;
    let mut dumped = reference(&cfg)?;
    dumped.attention_dump = Some(dir.path().to_path_buf());
    let (first, ..) = fixture_logits(&mut dumped, &cfg)?;
    for layer in 0..cfg.layers {
        for name in ["q", "k", "v", "out"] {
            assert!(dir.path().join(format!("layer{layer}.{name}.bin")).is_file());
        }
    }
    let mut injected = reference(&cfg)?;
    injected.attention_inject = Some(dir.path().to_path_buf());
    let (second, ..) = fixture_logits(&mut injected, &cfg)?;
    assert_eq!(first, second);
    Ok(())
}

/// Engine-level gate for a native attention kernel (Sol kernel agent):
/// 1. `CUTEAFD_MUSE_ATTENTION_DUMP=DIR cargo test ... muse_spark_attention_exchange`
///    writes each layer's kernel inputs (`layer{L}.{q,k,v}.bin`, BF16 [9, heads, 64],
///    q/k after norm + RoPE, rows at positions 0..9 of one sequence) and the eager
///    oracle's own output (`layer{L}.out.bin`, BF16 [9, 8 * 64], before o_norm);
/// 2. run the kernel on those inputs (layer 0 sliding window 5 with RoPE, layer 1
///    full NoPE, 8 query / 2 KV heads of 64, scale 0.125) and write its outputs
///    to `KDIR/layer{L}.out.bin` in the same layout;
/// 3. `CUTEAFD_MUSE_ATTENTION_INJECT=KDIR cargo test ...` runs the rest of the
///    model through the oracle and gates against the eager oracle (identical
///    pipeline elsewhere): max |logit diff| <= 1.5x the floor, the FP32-online-softmax
///    oracle vs the eager oracle on this fixture (measured here, max over key tiles
///    1/4/64; 0.1523 at this fixture, bar 0.2285), and top-1 equal on every
///    row whose eager margin exceeds 0.5. The official model's numbers are
///    printed for information. The per-element bar (atol 0.002 / rtol 0.02 vs
///    the dumped `out`) is the kernel agent's own check.
#[test]
fn muse_spark_attention_exchange() -> Result<()> {
    let (dump, inject) = (std::env::var_os("CUTEAFD_MUSE_ATTENTION_DUMP"), std::env::var_os("CUTEAFD_MUSE_ATTENTION_INJECT"));
    if dump.is_none() && inject.is_none() { return Ok(()); }
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures().join("official.json"))?)?;
    let cfg = MuseSparkConfig::from_hf(&record["config"])?;
    let mut eager = reference(&cfg)?;
    let (base, official, tokens) = fixture_logits(&mut eager, &cfg)?;
    let mut ops = reference(&cfg)?;
    ops.attention_dump = dump.map(std::path::PathBuf::from);
    ops.attention_inject = inject.clone().map(std::path::PathBuf::from);
    let (ours, ..) = fixture_logits(&mut ops, &cfg)?;
    let (worst, top1) = compare(&ours, &base);
    let mut floor = 0f32;
    for tile in [1, 4, 64] {
        let mut online = reference(&cfg)?;
        online.online_tile = Some(tile);
        floor = floor.max(compare(&fixture_logits(&mut online, &cfg)?.0, &base).0);
    }
    let bar = 1.5 * floor;
    eprintln!("muse attention exchange floor (FP32 online vs eager): {floor:.4}; bar {bar:.4}");
    let (official_worst, official_top1) = compare(&ours, &official);
    eprintln!("muse attention exchange vs eager oracle: max |logit diff| {worst:.4}, top-1 {top1}/{}", tokens.len());
    eprintln!("muse attention exchange vs official model: max |logit diff| {official_worst:.4}, top-1 {official_top1}/{}",
        tokens.len());
    if inject.is_some() {
        assert!(worst <= bar, "max logit difference vs the eager oracle {worst} (bar {bar:.4} = 1.5 x floor {floor:.4})");
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        for (row, (ours, base)) in ours.iter().zip(&base).enumerate() {
            let mut sorted = base.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            if sorted[0] - sorted[1] > 0.5 {
                assert_eq!(argmax(ours), argmax(base), "row {row}: top-1 flipped at margin {}", sorted[0] - sorted[1]);
            }
        }
    }
    Ok(())
}

/// Online softmax (the kernels' arithmetic class) against the eager oracle on
/// the same pipeline: the attention-only difference the exchange gate bounds.
#[test]
fn online_softmax_against_the_eager_oracle() -> Result<()> {
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures().join("official.json"))?)?;
    let cfg = MuseSparkConfig::from_hf(&record["config"])?;
    let mut eager = reference(&cfg)?;
    let (base, _, tokens) = fixture_logits(&mut eager, &cfg)?;
    for tile in [1, 4, 64] {
        let mut ops = reference(&cfg)?;
        ops.online_tile = Some(tile);
        let (ours, ..) = fixture_logits(&mut ops, &cfg)?;
        let (worst, top1) = compare(&ours, &base);
        eprintln!("online tile {tile} vs eager oracle: max |logit diff| {worst:.4}, top-1 {top1}/{}", tokens.len());
    }
    Ok(())
}

/// Semantic check independent of rounding: the oracle with every BF16 rounding
/// disabled against the official model run in FP32 (`fixture_precision.py`
/// writes `official_f32.json`). BF16 noise vanishes from both sides, so any
/// remaining gap is a semantic difference or FP32-vs-oracle reduction order.
#[test]
fn oracle_without_bf16_rounding_matches_the_official_fp32_model() -> Result<()> {
    let path = fixtures().join("official_f32.json");
    if !path.is_file() { return Ok(()); }
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures().join("official.json"))?)?;
    let cfg = MuseSparkConfig::from_hf(&record["config"])?;
    let tokens: Vec<u32> = serde_json::from_value(record["tokens"].clone())?;
    let exact: Vec<Vec<f32>> = serde_json::from_value(serde_json::from_slice::<serde_json::Value>(&std::fs::read(&path)?)?["logits"].clone())?;
    let mut ops = reference(&cfg)?;
    let logits = cuteafd_loader::families::muse_spark::reference::without_bf16_rounding(||
        forward(&cfg, &mut ops, &tokens, cfg.layers))?;
    let (worst, top1) = compare(&logits, &exact);
    eprintln!("muse oracle (no BF16 rounding) vs official FP32: max |logit diff| {worst:.6}, top-1 {top1}/{}", tokens.len());
    assert!(worst <= 1e-4, "semantic gap {worst}");
    Ok(())
}
