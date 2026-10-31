//! Checkpoint-native packing of one Muse Spark layer for one coordinator rank,
//! checked against the shared layout the planner charged
//! (`cuteafd_loader::families::muse_spark::weight_layout`), then uploaded.
//! Head-split attention takes this rank's query/key/value/gate rows and its
//! o_proj columns; everything else is whole. The residual gates are packed as
//! F32 (alpha, beta) from the contract's host function, so the device kernel
//! only applies them.
use crate::shared::{dense_trellis::{CoreWeightLayout, CoreWeights}, memory::device::{Allocation, Device}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::{families::muse_spark::{core::PREFIX, reference::residual_gate_coefficients,
    weight_layout::LayerLayout, MuseSparkConfig}, formats::dense_trellis::LoadedDenseTrellis, LoadedTensor};
use std::collections::BTreeMap;

/// One rank's packed host operands of one layer.
pub(crate) struct PackedLayer {
    pub(crate) operands: BTreeMap<&'static str, Vec<u8>>,
    pub(crate) core: BTreeMap<&'static str, LoadedDenseTrellis>,
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect()
}

/// Rows `[first, first + count)` of a row-major BF16 `[n, k]` matrix.
fn rows(bytes: &[u8], k: usize, first: usize, count: usize) -> Vec<u8> {
    bytes[first * k * 2..(first + count) * k * 2].to_vec()
}

/// Columns `[first, first + count)` of a row-major BF16 `[n, k]` matrix.
fn columns(bytes: &[u8], n: usize, k: usize, first: usize, count: usize) -> Vec<u8> {
    (0..n).flat_map(|row| bytes[(row * k + first) * 2..(row * k + first + count) * 2].iter().copied()).collect()
}

impl PackedLayer {
    /// `read` returns a whole checkpoint tensor (one read per tensor per rank).
    pub(crate) fn pack(cfg: &MuseSparkConfig, layer: usize, rank: usize, ranks: usize, layout: &LayerLayout,
        core_layouts: &BTreeMap<&'static str, CoreWeightLayout>,
        mut read: impl FnMut(&str) -> Result<LoadedTensor>) -> Result<Self> {
        ensure!(rank < ranks && matches!(ranks, 1 | 2), "Muse rank {rank} of {ranks}");
        let root = format!("{PREFIX}{layer}");
        let (h, d) = (cfg.hidden, cfg.head_dim);
        let (q, kv) = (cfg.heads * d / ranks, cfg.kv_heads * d / ranks);
        let mut operands = BTreeMap::new();
        let mut core = BTreeMap::new();
        let tensor = |read: &mut dyn FnMut(&str) -> Result<LoadedTensor>, stem: &str, dtype: DType| -> Result<Vec<u8>> {
            let name = format!("{root}.{stem}");
            let loaded = read(&name)?;
            ensure!(loaded.info.dtype == dtype, "{name}: {:?}, expected {dtype:?}", loaded.info.dtype);
            Ok(loaded.bytes)
        };
        let half = cfg.expert_hidden / ranks;
        for (name, stem, outputs, split_rows) in [("attn.q", "self_attn.q_proj", q, true), ("attn.k", "self_attn.k_proj", kv, true),
            ("attn.v", "self_attn.v_proj", kv, true), ("attn.gate", "self_attn.gate_proj", q, true),
            ("moe.pre_expert", "mlp.pre_expert_proj", half, true),
            ("moe.post_expert", "mlp.post_expert_proj", h, false), ("attn.o", "self_attn.o_proj", h, false)] {
            if let Some(core_layout) = core_layouts.get(name) {
                core.insert(name, core_layout.read(&mut |member: &str| read(member))?);
                continue;
            }
            let whole = tensor(&mut read, &format!("{stem}.weight"), DType::Bf16)?;
            let packed = if name == "attn.o" {
                // o_proj [h, heads*d]: this rank's heads are its input columns.
                columns(&whole, h, cfg.heads * d, rank * q, q)
            } else if name == "moe.post_expert" {
                // post_expert_proj [h, E]: this rank's half of the expert width (K-split).
                columns(&whole, h, cfg.expert_hidden, rank * half, half)
            } else if split_rows {
                rows(&whole, h, rank * outputs, outputs)
            } else { whole };
            operands.insert(name, packed);
        }
        operands.insert("moe.router", tensor(&mut read, "mlp.gate.weight", DType::F32)?);
        operands.insert("moe.bias", tensor(&mut read, "mlp.gate.e_score_correction_bias", DType::F32)?);
        operands.insert("moe.pre_expert_norm", tensor(&mut read, "mlp.pre_expert_norm.weight", DType::Bf16)?);
        // The routed experts' shared post_expert_norm: the coordinator combine applies it.
        operands.insert("moe.post_expert_norm", tensor(&mut read, "mlp.experts.post_expert_norm.weight", DType::Bf16)?);
        for (name, stem) in [("norm.input", "input_layernorm"), ("norm.pre_ffn", "pre_feedforward_layernorm"),
            ("norm.post_ffn", "post_feedforward_layernorm")] {
            operands.insert(name, tensor(&mut read, &format!("{stem}.weight"), DType::Bf16)?);
        }
        for (site, stem) in [("attention", "post_attention_residual_gate"), ("ffn", "post_feedforward_residual_gate")] {
            let gate = bf16_values(&tensor(&mut read, &format!("{stem}.gate"), DType::Bf16)?);
            let coefficients = residual_gate_coefficients(&gate, cfg.residual_gate_temperature as f32);
            let alpha = coefficients.iter().flat_map(|(a, _)| a.to_le_bytes()).collect();
            let beta = coefficients.iter().flat_map(|(_, b)| b.to_le_bytes()).collect();
            operands.insert(if site == "attention" { "gate.attention.alpha" } else { "gate.ffn.alpha" }, alpha);
            operands.insert(if site == "attention" { "gate.attention.beta" } else { "gate.ffn.beta" }, beta);
        }
        let packed = Self { operands, core };
        packed.check(layout)?;
        Ok(packed)
    }

    /// Exactly the operands and extents the planner charged for this rank.
    pub(crate) fn check(&self, layout: &LayerLayout) -> Result<()> {
        ensure!(self.operands.len() == layout.operands.len() && self.core.len() == layout.core.len(),
            "Muse packed operand inventory differs from the admitted layout");
        for (name, bytes) in &layout.operands {
            ensure!(self.operands.get(name).is_some_and(|data| data.len() == *bytes),
                "Muse packed operand {name} differs from the admitted layout");
        }
        for (name, shared) in &layout.core {
            let loaded = self.core.get(name).context("Muse packed K6 operand absent")?;
            CoreWeightLayout::from_shared(shared.clone()).check(loaded)?;
        }
        Ok(())
    }
}

/// One rank's resident layer weights.
pub(crate) struct LayerWeights<'a> {
    operands: BTreeMap<&'static str, Allocation<'a>>,
    core: BTreeMap<&'static str, CoreWeights<'a>>,
}

impl<'a> LayerWeights<'a> {
    /// `charged_bytes`: this rank's layer demand accepted by placement admission.
    pub(crate) fn upload(device: Device<'a>, packed: PackedLayer, layout: &LayerLayout, charged_bytes: usize) -> Result<Self> {
        packed.check(layout)?;
        ensure!(layout.bytes()? == charged_bytes, "Muse layer layout differs from the admitted weight bytes");
        let operands = packed.operands.into_iter().map(|(name, bytes)| {
            let _scope = cuteafd_ffi::memory_ledger::scope("weights/muse layer weights");
            let allocation = Allocation::new(device, bytes.len())?;
            device.run(|| device.library.copy_h2d(allocation.buffer, &bytes))?;
            Ok((name, allocation))
        }).collect::<Result<_>>()?;
        let core = packed.core.into_iter().map(|(name, loaded)| {
            let shared = layout.core.get(name).context("Muse K6 layout absent")?;
            CoreWeightLayout::from_shared(shared.clone()).upload(device, loaded).map(|weights| (name, weights))
        }).collect::<Result<_>>()?;
        Ok(Self { operands, core })
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self.operands.get(name).with_context(|| format!("Muse layer has no resident operand {name}"))?.buffer)
    }

    pub(crate) fn core(&self, name: &str) -> Option<&CoreWeights<'a>> { self.core.get(name) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack both ranks of the engine fixture's layer (BF16, head-split) and the
    /// whole form: extents match the shared layout, the halves reassemble the
    /// whole matrices, and the gate coefficients are the contract's.
    #[test]
    fn fixture_layer_packs_to_the_shared_layout() -> Result<()> {
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/muse_spark/fixtures");
        let record: serde_json::Value = serde_json::from_slice(&std::fs::read(fixtures.join("official.json"))?)?;
        let cfg = MuseSparkConfig::from_hf(&record["config"])?;
        let path = fixtures.join("model.safetensors");
        let bytes = std::fs::read(&path)?;
        let headers = cuteafd_loader::read_safetensors_metadata(&path)?;
        let read = |name: &str| -> Result<LoadedTensor> {
            let meta = headers.iter().find(|m| m.name == name).with_context(|| format!("fixture lacks {name}"))?;
            Ok(LoadedTensor { info: cuteafd_core::TensorInfo { name: name.into(), file: "model.safetensors".into(),
                dtype: meta.dtype.clone(), shape: meta.shape.clone(), byte_offset: meta.byte_offset,
                byte_length: meta.byte_length, role: cuteafd_core::TensorRole::Other, layer_id: None, expert_id: None,
                is_quantization_metadata: false }, source_path: path.clone(),
                bytes: bytes[meta.byte_offset as usize..(meta.byte_offset + meta.byte_length) as usize].to_vec(),
                elapsed_micros: 0, sha256: String::new() })
        };
        let no_core = cuteafd_loader::families::muse_spark::core::CoreInventory::default();
        let whole_layout = LayerLayout::ranks(&cfg, 0, 1, &no_core)?.remove(0);
        let whole = PackedLayer::pack(&cfg, 0, 0, 1, &whole_layout, &BTreeMap::new(), read)?;
        let pair_layout = LayerLayout::ranks(&cfg, 0, 2, &no_core)?;
        let pair = (0..2).map(|rank| PackedLayer::pack(&cfg, 0, rank, 2, &pair_layout[rank], &BTreeMap::new(), read))
            .collect::<Result<Vec<_>>>()?;
        for name in ["attn.q", "attn.k", "attn.v", "attn.gate"] {
            let joined: Vec<u8> = pair[0].operands[name].iter().chain(&pair[1].operands[name]).copied().collect();
            assert_eq!(joined, whole.operands[name], "{name} halves are its row blocks");
        }
        // o_proj columns: each rank's half of every row.
        let (h, half) = (cfg.hidden, cfg.heads * cfg.head_dim / 2);
        for row in [0, h / 2, h - 1] {
            let whole_row = &whole.operands["attn.o"][row * 2 * half * 2..(row + 1) * 2 * half * 2];
            assert_eq!(&pair[0].operands["attn.o"][row * half * 2..(row + 1) * half * 2], &whole_row[..half * 2]);
            assert_eq!(&pair[1].operands["attn.o"][row * half * 2..(row + 1) * half * 2], &whole_row[half * 2..]);
        }
        assert_eq!(pair[0].operands["moe.router"], whole.operands["moe.router"]);
        let gate = bf16_values(&read("model.language_model.layers.0.post_attention_residual_gate.gate")?.bytes);
        let expected = residual_gate_coefficients(&gate, cfg.residual_gate_temperature as f32);
        let alpha: Vec<f32> = whole.operands["gate.attention.alpha"].chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(alpha, expected.iter().map(|(a, _)| *a).collect::<Vec<_>>());
        Ok(())
    }
}
