//! Native dense BF16/E4M3 operands with checkpoint-format scales and BF16 gates.
use crate::shared::memory::device::Allocation;
use crate::shared::qwen::weights::WeightReader;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_loader::families::qwen::HybridAttention;
use cuteafd_loader::families::qwen3_5::Qwen35Config;
use std::{collections::HashMap, ffi::c_void};

const PREFIX: &str = "model.language_model.";

pub(crate) struct DenseLayer<'a> {
    pub(crate) attention: HybridAttention,
    operands: HashMap<&'static str, Allocation<'a>>,
}

impl DenseLayer<'_> {
    pub(crate) fn ptr(&self, name: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(name).with_context(|| format!("dense Qwen layer has no operand {name}"))?.buffer.ptr)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct DenseWeights<'a> {
    pub(crate) layers: Vec<DenseLayer<'a>>,
    pub(crate) norm: Allocation<'a>,
    pub(crate) head: Allocation<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage { Bf16, F32, NativeNorm, Fp8, Scales }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DenseFormat { Bf16, Fp8 }

impl DenseFormat {
    pub(crate) fn detect(reader: &WeightReader<'_>) -> Result<Self> {
        match reader.tensor(&format!("{PREFIX}layers.0.mlp.gate_proj.weight"))?.meta.dtype {
            DType::Bf16 => Ok(Self::Bf16),
            DType::F8E4M3 => Ok(Self::Fp8),
            ref dtype => anyhow::bail!("unsupported dense projection format {dtype:?}"),
        }
    }

    pub(crate) fn fp8(self) -> bool { self == Self::Fp8 }
}

#[derive(Debug, PartialEq, Eq)]
struct Operand {
    name: &'static str,
    tensors: Vec<String>,
    storage: Storage,
}

fn layer_operands(prefix: &str, attention: HybridAttention) -> Vec<Operand> {
    let mut out = Vec::new();
    let mut add = |name, tensors: &[&str], storage| out.push(Operand { name,
        tensors: tensors.iter().map(|t| format!("{prefix}.{t}")).collect(), storage });
    for (operand, tensor) in [("attn.norm", "input_layernorm.weight"), ("ffn.norm", "post_attention_layernorm.weight"),
        ("ffn.w_down", "mlp.down_proj.weight")] {
        add(operand, &[tensor], Storage::Bf16);
    }
    add("ffn.w_gate_up", &["mlp.gate_proj.weight", "mlp.up_proj.weight"], Storage::Bf16);
    match attention {
        HybridAttention::Gdn => {
            add("w_in", &["linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"], Storage::Bf16);
            for (operand, tensor) in [("conv_w", "conv1d.weight"), ("a_log", "A_log"), ("dt_bias", "dt_bias")] {
                add(operand, &[&format!("linear_attn.{tensor}")], Storage::F32);
            }
            add("norm_w", &["linear_attn.norm.weight"], Storage::NativeNorm);
            add("w_out", &["linear_attn.out_proj.weight"], Storage::Bf16);
        }
        HybridAttention::Full => {
            add("w_in", &["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"], Storage::Bf16);
            for (operand, tensor) in [("q_norm", "q_norm.weight"), ("k_norm", "k_norm.weight"), ("w_o", "o_proj.weight")] {
                add(operand, &[&format!("self_attn.{tensor}")], Storage::Bf16);
            }
        }
    }
    out
}

fn native_operands(prefix: &str, attention: HybridAttention, format: DenseFormat) -> Vec<Operand> {
    let specs = layer_operands(prefix, attention);
    if !format.fp8() { return specs; }
    let mut out = Vec::new();
    for mut spec in specs {
        let projection = matches!(spec.name, "w_in" | "w_out" | "w_o" | "ffn.w_gate_up" | "ffn.w_down");
        if !projection { out.push(spec); continue; }
        if attention == HybridAttention::Gdn && spec.name == "w_in" {
            let gates = spec.tensors.split_off(2);
            out.push(Operand { name: "w_ba", tensors: gates, storage: Storage::Bf16 });
        }
        let (weight, scales) = match spec.name {
            "w_in" if attention == HybridAttention::Gdn => ("w_qkvz_fp8", "w_qkvz_scale"),
            "w_in" => ("w_in_fp8", "w_in_scale"),
            "w_out" => ("w_out_fp8", "w_out_scale"),
            "w_o" => ("w_o_fp8", "w_o_scale"),
            "ffn.w_gate_up" => ("ffn.w_gate_up_fp8", "ffn.w_gate_up_scale"),
            "ffn.w_down" => ("ffn.w_down_fp8", "ffn.w_down_scale"),
            _ => unreachable!(),
        };
        out.push(Operand { name: scales, tensors: spec.tensors.iter()
            .map(|name| format!("{}_scale_inv", name)).collect(), storage: Storage::Scales });
        out.push(Operand { name: weight, tensors: spec.tensors, storage: Storage::Fp8 });
    }
    out
}

fn packed_native_layout(tensors: &[(Vec<u8>, DType, Vec<usize>)], storage: Storage) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut cols = None;
    for (bytes, dtype, shape) in tensors {
        ensure!(shape.len() == 2 && shape[0] > 0 && shape[1] > 0, "native projection requires nonempty matrices");
        ensure!(cols.is_none_or(|previous| previous == shape[1]), "native packed matrix columns differ");
        cols = Some(shape[1]);
        let elements = shape[0].checked_mul(shape[1]).context("native matrix size overflow")?;
        let width = match (storage, dtype) {
            (Storage::Fp8, DType::F8E4M3) => 1,
            (Storage::Scales, DType::Bf16) => 2,
            (Storage::Scales, DType::F32) => 4,
            _ => anyhow::bail!("native matrix dtype disagrees with operand storage"),
        };
        ensure!(bytes.len() == elements.checked_mul(width).context("native matrix byte size overflow")?,
            "native matrix payload disagrees with shape");
        if storage == Storage::Scales && *dtype == DType::Bf16 {
            out.extend(bytes.chunks_exact(2).flat_map(|v|
                (u32::from(u16::from_le_bytes([v[0], v[1]])) << 16).to_le_bytes()));
        } else { out.extend_from_slice(bytes); }
    }
    ensure!(!out.is_empty(), "empty native matrix packing");
    Ok(out)
}

pub(crate) struct DenseWeightBytes {
    pub(crate) layers: Vec<u64>,
    pub(crate) norm: u64,
    pub(crate) head: u64,
    pub(crate) embedding: u64,
}

impl<'a> WeightReader<'a> {
    /// Header-only native allocation inventory, including widened tiny tensors.
    pub(crate) fn dense_bytes(&self, cfg: &Qwen35Config, norm_dtype: DType, format: DenseFormat) -> Result<DenseWeightBytes> {
        ensure!(matches!(norm_dtype, DType::Bf16 | DType::F32), "dense Qwen GDN norm ABI must be BF16 or FP32");
        let specs: Vec<_> = cfg.attention.iter().enumerate().map(|(layer, &kind)|
            native_operands(&format!("{PREFIX}layers.{layer}"), kind, format)).collect();
        for spec in specs.iter().flatten() {
            for name in &spec.tensors {
                let dtype = &self.tensor(name)?.meta.dtype;
                let expected = match spec.storage { Storage::NativeNorm => &norm_dtype,
                    Storage::Fp8 => &DType::F8E4M3, _ => &DType::Bf16 };
                ensure!(dtype == expected || (matches!(spec.storage, Storage::F32 | Storage::Scales) && *dtype == DType::F32),
                    "{name}: dense export expects {expected:?}, found {dtype:?}");
                if matches!(spec.storage, Storage::Fp8 | Storage::Scales) {
                    let meta = &self.tensor(name)?.meta;
                    ensure!(meta.shape.len() == 2, "{name}: native FP8 operands must be matrices");
                    if spec.storage == Storage::Fp8 {
                        ensure!(meta.shape[0] % 128 == 0 && meta.shape[1] % 128 == 0,
                            "{name}: packed native FP8 operands require whole 128x128 blocks");
                    }
                }
            }
        }
        let norm_name = format!("{PREFIX}norm.weight");
        let head_name = if cfg.tied_embeddings { format!("{PREFIX}embed_tokens.weight") } else { "lm_head.weight".into() };
        for name in [&norm_name, &head_name] {
            ensure!(self.tensor(name)?.meta.dtype == DType::Bf16, "{name}: expected BF16");
        }
        let layers = specs.iter().map(|operands| operands.iter().try_fold(0u64, |total, spec| {
            let bytes = spec.tensors.iter().try_fold(0u64, |total, name| {
                let tensor = self.tensor(name)?;
                let bytes = if matches!(spec.storage, Storage::F32 | Storage::Scales) && tensor.meta.dtype == DType::Bf16 {
                    tensor.meta.byte_length.checked_mul(2).context("dense widened weight size overflow")?
                } else { tensor.meta.byte_length };
                total.checked_add(bytes).context("dense packed weight size overflow")
            })?.max(256);
            total.checked_add(bytes).context("dense layer weight size overflow")
        })).collect::<Result<_>>()?;
        let embedding = self.tensor(&format!("{PREFIX}embed_tokens.weight"))?;
        ensure!(embedding.meta.dtype == DType::Bf16, "dense embedding must be native BF16");
        Ok(DenseWeightBytes { layers, norm: self.tensor(&norm_name)?.meta.byte_length.max(256),
            head: self.tensor(&head_name)?.meta.byte_length.max(256), embedding: embedding.meta.byte_length })
    }

    /// Validate every header before uploading any weights.
    pub(crate) fn dense_native(&self, cfg: &Qwen35Config, norm_dtype: DType, format: DenseFormat) -> Result<DenseWeights<'a>> {
        self.dense_bytes(cfg, norm_dtype, format)?;
        let norm_name = format!("{PREFIX}norm.weight");
        let head_name = if cfg.tied_embeddings { format!("{PREFIX}embed_tokens.weight") } else { "lm_head.weight".into() };
        let specs = cfg.attention.iter().enumerate().map(|(layer, &kind)|
            native_operands(&format!("{PREFIX}layers.{layer}"), kind, format));
        let layers = cfg.attention.iter().zip(specs).map(|(&attention, specs)| {
            let operands = specs.into_iter().map(|spec| {
                let weight = match spec.storage {
                    Storage::Bf16 if spec.tensors.len() > 1 => self.rows(&spec.tensors, 0)?,
                    Storage::Bf16 => self.bf16(&spec.tensors[0])?,
                    Storage::F32 => self.f32(&spec.tensors[0])?,
                    Storage::Fp8 | Storage::Scales => {
                        let tensors = spec.tensors.iter().map(|name| self.raw(name)).collect::<Result<Vec<_>>>()?;
                        self.upload(&packed_native_layout(&tensors, spec.storage)?)?
                    }
                    Storage::NativeNorm => {
                        let (bytes, _, _) = self.raw(&spec.tensors[0])?;
                        self.upload(&bytes)?
                    }
                };
                Ok((spec.name, weight))
            }).collect::<Result<_>>()?;
            Ok(DenseLayer { attention, operands })
        }).collect::<Result<_>>()?;
        Ok(DenseWeights { layers, norm: self.bf16(&norm_name)?, head: self.bf16(&head_name)? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_operands_keep_raw_weight_bytes_exact_scales_and_bf16_gates() -> Result<()> {
        let specs = native_operands("layer", HybridAttention::Gdn, DenseFormat::Fp8);
        let find = |name| specs.iter().find(|spec| spec.name == name).unwrap();
        assert_eq!(find("w_ba").tensors, ["layer.linear_attn.in_proj_b.weight", "layer.linear_attn.in_proj_a.weight"]);
        assert_eq!(find("w_ba").storage, Storage::Bf16);
        assert_eq!(find("w_qkvz_scale").tensors, ["layer.linear_attn.in_proj_qkv.weight_scale_inv",
            "layer.linear_attn.in_proj_z.weight_scale_inv"]);
        assert!(specs.iter().all(|spec| !matches!(spec.name, "w_in" | "w_out")));
        assert_eq!(packed_native_layout(&[(vec![1, 2], DType::F8E4M3, vec![1, 2]),
            (vec![3, 4], DType::F8E4M3, vec![1, 2])], Storage::Fp8)?, vec![1, 2, 3, 4]);
        let bits = [0x3f80u16, 0x8000, 0x0001, 0x7f7f];
        let bytes = bits.iter().flat_map(|value| value.to_le_bytes()).collect();
        let widened = packed_native_layout(&[(bytes, DType::Bf16, vec![2, 2])], Storage::Scales)?;
        assert_eq!(widened, bits.iter().flat_map(|&value| (u32::from(value) << 16).to_le_bytes()).collect::<Vec<_>>());
        assert!(packed_native_layout(&[(vec![0; 2], DType::Bf16, vec![1, 1])], Storage::Fp8).is_err());
        assert!(packed_native_layout(&[(vec![0], DType::F8E4M3, vec![1, 2])], Storage::Fp8).is_err());
        Ok(())
    }

    #[test]
    fn dense_operands_keep_native_projection_order_and_norm_precision() {
        for kind in [HybridAttention::Gdn, HybridAttention::Full] {
            let specs = layer_operands("layer", kind);
            let names = |operand: &str| specs.iter().find(|s| s.name == operand).unwrap().tensors.clone();
            assert_eq!(names("ffn.w_gate_up"), ["layer.mlp.gate_proj.weight", "layer.mlp.up_proj.weight"]);
            if kind == HybridAttention::Gdn {
                assert_eq!(names("w_in"), ["layer.linear_attn.in_proj_qkv.weight", "layer.linear_attn.in_proj_z.weight",
                    "layer.linear_attn.in_proj_b.weight", "layer.linear_attn.in_proj_a.weight"]);
                assert_eq!(specs.iter().find(|s| s.name == "norm_w").unwrap().storage, Storage::NativeNorm);
                assert_eq!(specs.iter().filter(|s| s.storage == Storage::F32).count(), 3);
            } else {
                assert_eq!(names("w_in"), ["layer.self_attn.q_proj.weight", "layer.self_attn.k_proj.weight", "layer.self_attn.v_proj.weight"]);
            }
            assert!(specs.iter().all(|s| !s.name.contains("expert") && !s.name.contains("hc") && !s.name.contains("index")));
        }
    }
}
