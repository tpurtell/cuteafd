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

fn validate_native_matrices<'t>(tensors: impl IntoIterator<Item = (&'t DType, &'t [usize], u64)>,
    storage: Storage) -> Result<()> {
    let mut cols = None;
    for (dtype, shape, bytes) in tensors {
        ensure!(shape.len() == 2 && shape[0] > 0 && shape[1] > 0, "native projection requires nonempty matrices");
        ensure!(cols.is_none_or(|previous| previous == shape[1]), "native packed matrix columns differ");
        cols = Some(shape[1]);
        let width = match (storage, dtype) {
            (Storage::Fp8, DType::F8E4M3) => 1,
            (Storage::Scales, DType::Bf16) => 2,
            (Storage::Scales, DType::F32) => 4,
            _ => anyhow::bail!("native matrix dtype disagrees with operand storage"),
        };
        let size = shape[0].checked_mul(shape[1]).and_then(|n| n.checked_mul(width))
            .context("native matrix byte size overflow")?;
        ensure!(u64::try_from(size)? == bytes, "native matrix payload disagrees with shape");
    }
    ensure!(cols.is_some(), "empty native matrix packing");
    Ok(())
}

#[cfg(test)]
fn packed_native_layout(tensors: &[(Vec<u8>, DType, Vec<usize>)], storage: Storage) -> Result<Vec<u8>> {
    validate_native_matrices(tensors.iter().map(|(bytes, dtype, shape)|
        (dtype, shape.as_slice(), bytes.len() as u64)), storage)?;
    let mut out = Vec::new();
    for (bytes, dtype, _) in tensors {
        if storage == Storage::Scales && *dtype == DType::Bf16 {
            out.extend(bytes.chunks_exact(2).flat_map(|v|
                (u32::from(u16::from_le_bytes([v[0], v[1]])) << 16).to_le_bytes()));
        } else { out.extend_from_slice(bytes); }
    }
    ensure!(!out.is_empty(), "empty native matrix packing");
    Ok(out)
}

impl<'a> WeightReader<'a> {
    pub(crate) fn dense_bytes(&self, cfg: &Qwen35Config, norm_dtype: DType, _format: DenseFormat)
        -> Result<cuteafd_loader::placement::families::qwen3_5::NativeWeights> {
        cuteafd_loader::placement::families::qwen3_5::NativeWeights::native(self.checkpoint, cfg, norm_dtype)
    }

    /// Validate every header before uploading any weights.
    pub(crate) fn dense_native(&self, cfg: &Qwen35Config, norm_dtype: DType, format: DenseFormat) -> Result<DenseWeights<'a>> {
        self.dense_bytes(cfg, norm_dtype, format)?;
        let norm_name = format!("{PREFIX}norm.weight");
        let head_name = if cfg.tied_embeddings { format!("{PREFIX}embed_tokens.weight") } else { "lm_head.weight".into() };
        let layers = cfg.attention.iter().enumerate().map(|(index, &kind)|
            self.dense_layer(&format!("{PREFIX}layers.{index}"), kind, format)).collect::<Result<_>>()?;
        Ok(DenseWeights { layers, norm: self.bf16(&norm_name)?, head: self.bf16(&head_name)? })
    }

    pub(crate) fn dense_layer_bytes(&self, prefix: &str, attention: HybridAttention, format: DenseFormat) -> Result<u64> {
        native_operands(prefix, attention, format).into_iter().try_fold(0u64, |total, spec| {
            let bytes = spec.tensors.iter().try_fold(0u64, |bytes, name| {
                let tensor = &self.tensor(name)?.meta;
                ensure!(match spec.storage {
                    Storage::Bf16 => tensor.dtype == DType::Bf16,
                    Storage::F32 | Storage::Scales => matches!(tensor.dtype, DType::Bf16 | DType::F32),
                    Storage::Fp8 => tensor.dtype == DType::F8E4M3,
                    Storage::NativeNorm => matches!(tensor.dtype, DType::Bf16 | DType::F32),
                }, "{name}: native dense operand dtype differs from admitted projection format");
                let widen = matches!(spec.storage, Storage::F32 | Storage::Scales) && tensor.dtype == DType::Bf16;
                Ok::<_, anyhow::Error>(bytes + tensor.byte_length * if widen { 2 } else { 1 })
            })?;
            Ok(total + bytes.max(256))
        })
    }

    /// MTP and target blocks share the checkpoint-native projection packing.
    pub(crate) fn dense_layer(&self, prefix: &str, attention: HybridAttention, format: DenseFormat) -> Result<DenseLayer<'a>> {
        let operands = native_operands(prefix, attention, format).into_iter().map(|spec| {
                let weight = match spec.storage {
                    Storage::Bf16 if spec.tensors.len() > 1 => self.rows(&spec.tensors, 0)?,
                    Storage::Bf16 => self.bf16(&spec.tensors[0])?,
                    Storage::F32 => self.f32(&spec.tensors[0])?,
                    Storage::Fp8 | Storage::Scales => {
                        let headers = spec.tensors.iter().map(|name| self.tensor(name)).collect::<Result<Vec<_>>>()?;
                        validate_native_matrices(headers.iter().map(|tensor|
                            (&tensor.meta.dtype, tensor.meta.shape.as_slice(), tensor.meta.byte_length)), spec.storage)?;
                        self.native(&spec.tensors, spec.storage == Storage::Scales)?
                    },
                    Storage::NativeNorm => self.native(&spec.tensors, false)?,
                };
            Ok((spec.name, weight))
        }).collect::<Result<_>>()?;
        Ok(DenseLayer { attention, operands })
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
