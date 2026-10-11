//! Resident GLM Flash operands: the loader's concatenations, conversions and rank ownership.
use super::{GlmNextAttention, GlmNextConfig};
use crate::plan::Checkpoint;
use cuteafd_core::DType;

#[derive(Debug, Clone, Copy)]
pub struct GlmfRepresentation {
    pub kda_fp8: bool,
    pub fp8_head: bool,
    pub output_shard: bool,
}
impl Default for GlmfRepresentation {
    fn default() -> Self { Self { kda_fp8: true, fp8_head: true, output_shard: true } }
}

#[derive(Debug, Clone, Default)]
pub struct GlmfResidentRank {
    pub embedding: u64,
    pub weights: u64,
}

/// No tensor data is read. FP8 block scales and FP32 auxiliaries are counted as uploaded, not
/// as the source checkpoint stores them. Routed experts, media and external drafts are separate.
pub fn resident_weights(checkpoint: &Checkpoint, cfg: &GlmNextConfig, layers: usize, ranks: usize,
    repr: GlmfRepresentation) -> Result<Vec<GlmfResidentRank>, String> {
    if !(1..=2).contains(&ranks) { return Err("GLM Flash weights require one or two ranks".into()); }
    let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .ok_or_else(|| format!("GLM Flash resident operand missing: {name}"));
    let elements = |name: &str| -> Result<u64, String> {
        tensor(name)?.meta.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d as u64)
            .ok_or_else(|| format!("GLM Flash operand overflow: {name}")))
    };
    let mut out = vec![GlmfResidentRank::default(); ranks];
    out[0].embedding = tensor("model.language_model.embed_tokens.weight")?.meta.byte_length.max(256);
    let fp8 = |names: &[String], split: bool, row128: bool| -> Result<u64, String> {
        let mut values = 0;
        let mut scales = 0;
        for name in names {
            let t = tensor(name)?;
            let shape = &t.meta.shape;
            if shape.len() != 2 { return Err(format!("GLM Flash FP8 operand is not a matrix: {name}")); }
            let n = shape[0] as u64;
            let k = shape[1] as u64;
            values += n * k;
            scales += if row128 { n * k.div_ceil(128) * 4 } else { n.div_ceil(128) * k.div_ceil(128) * 4 };
        }
        let divisor = if split { ranks as u64 } else { 1 };
        Ok((values / divisor).max(256) + (scales / divisor).max(256))
    };
    out[0].weights = tensor("model.language_model.norm.weight")?.meta.byte_length.max(256)
        + if repr.fp8_head { fp8(&["lm_head.weight".into()], false, true)? }
            else { tensor("lm_head.weight")?.meta.byte_length.max(256) };
    for layer in 0..layers.min(cfg.layers) {
        let p = format!("model.language_model.layers.{layer}");
        let a = |s: &str| format!("{p}.self_attn.{s}");
        let one = |s: &str| -> Result<u64, String> { Ok(tensor(s)?.meta.byte_length.max(256)) };
        let f32 = |names: &[String], split: bool| -> Result<u64, String> {
            let n = names.iter().map(|name| elements(name)).collect::<Result<Vec<_>, _>>()?.into_iter().sum::<u64>();
            Ok((n * 4 / if split { ranks as u64 } else { 1 }).max(256))
        };
        for rank in 0..ranks {
            let mut bytes = 0;
            for site in ["attn", "ffn"] {
                for suffix in ["fn", "scale", "base"] { bytes += f32(&[format!("{p}.hc_{site}_{suffix}")], false)?; }
            }
            bytes += one(&format!("{p}.input_layernorm.weight"))? + one(&format!("{p}.post_attention_layernorm.weight"))?;
            match cfg.attention[layer] {
                GlmNextAttention::Kda => {
                    let inputs: Vec<_> = ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj"]
                        .iter().map(|s| a(&format!("{s}.weight"))).collect();
                    let mut input_values = 0;
                    for (i, name) in inputs.iter().enumerate() {
                        input_values += elements(name)? / if ranks > 1 && !matches!(i, 3 | 4) { ranks as u64 } else { 1 };
                    }
                    bytes += if repr.kda_fp8 { input_values.max(256) + (input_values / 128 * 4).max(256) }
                        else { (input_values * 2).max(256) };
                    let output_split = ranks > 1 && !(repr.kda_fp8 && repr.output_shard);
                    bytes += if repr.kda_fp8 { fp8(&[a("o_proj.weight")], output_split, true)? }
                        else { (elements(&a("o_proj.weight"))? * 2 / ranks as u64).max(256) };
                    bytes += ((elements(&a("f_b_proj.weight"))? + elements(&a("g_b_proj.weight"))?) * 2 / ranks as u64).max(256);
                    bytes += f32(&["q", "k", "v"].map(|s| a(&format!("{s}_conv1d.weight"))), ranks > 1)?;
                    bytes += f32(&[a("A_log")], ranks > 1)? + f32(&[a("dt_bias")], ranks > 1)?;
                    bytes += one(&a("o_norm.weight"))?;
                }
                GlmNextAttention::Mla => {
                    bytes += one(&a("q_a_layernorm.weight"))? + one(&a("kv_a_layernorm.weight"))?;
                    // kv_b is absorbed into BF16 UK and UV, divided by heads.
                    bytes += (cfg.heads * cfg.qk_nope_dim * cfg.kv_lora_rank * 2 / ranks) as u64;
                    bytes += (cfg.heads * cfg.v_head_dim * cfg.kv_lora_rank * 2 / ranks) as u64;
                    bytes += fp8(&[a("q_a_proj.weight"), a("kv_a_proj_with_mqa.weight")], false, false)?;
                    bytes += fp8(&[a("q_b_proj.weight")], ranks > 1, false)? + fp8(&[a("o_proj.weight")], ranks > 1, false)?;
                    bytes += one(&a("indexer.wq_b.weight"))?;
                    bytes += ["wk.weight", "weights_proj.weight", "index_kpool_compress_gate"].iter()
                        .map(|s| elements(&a(&format!("indexer.{s}")))).collect::<Result<Vec<_>, _>>()?.into_iter().sum::<u64>() * 2;
                    for s in ["k_norm.weight", "k_norm.bias", "index_kpool_compress_ape"] { bytes += one(&a(&format!("indexer.{s}")))?; }
                }
            }
            let mlp = if cfg.dense[layer] { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
            let gate = format!("{mlp}.gate_proj.weight");
            if cfg.dense[layer] && tensor(&gate)?.meta.dtype == DType::U8 {
                if rank == 0 {
                    for s in ["gate_proj", "up_proj", "down_proj"] {
                        bytes += one(&format!("{mlp}.{s}.weight"))?;
                        bytes += (tensor(&format!("{mlp}.{s}.weight_scale"))?.meta.byte_length + 8).max(256);
                    }
                }
            } else {
                bytes += fp8(&[gate, format!("{mlp}.up_proj.weight")], ranks > 1, false)?;
                bytes += fp8(&[format!("{mlp}.down_proj.weight")], ranks > 1, false)?;
            }
            if !cfg.dense[layer] && rank == 0 {
                bytes += one(&format!("{p}.mlp.gate.weight"))?;
                bytes += f32(&[format!("{p}.mlp.gate.e_score_correction_bias")], false)?;
            }
            out[rank].weights += bytes;
        }
    }
    Ok(out)
}

/// Router operands newly replicated on GPU1 by the TP2 executor, counted before loading.
pub fn router_replica_bytes(checkpoint: &Checkpoint, cfg: &GlmNextConfig, layers: usize) -> Result<u64, String> {
    let mut bytes = 0u64;
    for layer in 0..layers.min(cfg.layers) {
        if cfg.dense[layer] { continue; }
        let prefix = format!("model.language_model.layers.{layer}.mlp.gate");
        for (suffix, f32) in [("weight", false), ("e_score_correction_bias", true)] {
            let name = format!("{prefix}.{suffix}");
            let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == name)
                .ok_or_else(|| format!("GLM Flash router operand missing: {name}"))?;
            let resident = if f32 { tensor.meta.shape.iter().product::<usize>() as u64 * 4 }
                else { tensor.meta.byte_length };
            bytes = bytes.checked_add(resident.max(256)).ok_or_else(|| "GLM Flash router bytes overflow".to_string())?;
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_fixture_charges_borrowable_head_only_on_lead() {
        use crate::plan::checkpoint::CheckpointTensor;
        let config = crate::plan::testing::glm5_flash_config(2);
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let tensors = [("model.language_model.embed_tokens.weight", vec![64, 4096]),
            ("model.language_model.norm.weight", vec![4096]), ("lm_head.weight", vec![64, 4096])]
            .into_iter().map(|(name, shape)| CheckpointTensor { shard: "fixture".into(),
                meta: crate::SafetensorsTensorMetadata { name: name.into(), dtype: DType::Bf16,
                    byte_length: shape.iter().product::<usize>() as u64 * 2, shape, byte_offset: 0 } }).collect();
        let checkpoint = Checkpoint { snapshot: "fixture".into(), config, quantize_config: None,
            weight_map: Default::default(), tensors, missing_shards: vec![], shard_bytes: 0 };
        // Zero target layers isolates the head representation and ownership contract.
        let single = resident_weights(&checkpoint, &cfg, 0, 1, Default::default()).unwrap();
        let split = resident_weights(&checkpoint, &cfg, 0, 2, Default::default()).unwrap();
        assert_eq!(single[0].embedding, 524_288);
        assert_eq!(single[0].weights, 278_528); // norm BF16 + head FP8 values and row128 FP32 scales.
        assert_eq!(split[0].weights, single[0].weights);
        assert_eq!((split[1].embedding, split[1].weights), (0, 0));
        assert!(resident_weights(&checkpoint, &cfg, 2, 1, Default::default()).unwrap_err().contains("resident operand missing"));
    }

    #[test]
    fn real_exl3_and_fp8_headers_match_rc3_uploaded_operands() {
        let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
            .unwrap_or_else(|| "/mnt/sparknest/hf-home".into());
        for model in ["wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1", "zai-org/GLM-5.3-Flash"] {
            let Ok(resolved) = crate::resolve_snapshot_at_revision(model, Some(&home), None) else { continue };
            let Some(path) = resolved.snapshot_path else { continue };
            let checkpoint = Checkpoint::open(&path).unwrap();
            let cfg = GlmNextConfig::from_hf(&checkpoint.config).unwrap();
            let single = resident_weights(&checkpoint, &cfg, cfg.layers, 1, Default::default()).unwrap();
            assert_eq!((single[0].embedding, single[0].weights), (1_268_776_960, 8_956_184_320), "{model}");
            let split = resident_weights(&checkpoint, &cfg, cfg.layers, 2, Default::default()).unwrap();
            assert_eq!((split[0].weights, split[1].weights), (5_660_947_200, 4_907_587_072), "{model}");
            assert_eq!(split[1].embedding, 0);
        }
    }
}
