//! Checkpoint-native packing of one Muse Spark layer for one coordinator rank,
//! checked against the shared layout the planner charged
//! (`cuteafd_loader::families::muse_spark::weight_layout`), then filled by the
//! shared bounded mount/push records. The host packer remains the test oracle.
//! Head-split attention takes this rank's query/key/value/gate rows and its
//! o_proj columns; everything else is whole. The residual gates are packed as
//! F32 (alpha, beta) from the contract's host function, so the device kernel
//! only applies them.
use crate::shared::{dense_trellis::{CoreWeightLayout, CoreWeights}, memory::device::{Allocation, Device}};
use anyhow::{ensure, Context, Result};
use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step};
use cuteafd_core::DType;
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::families::muse_spark::{core::PREFIX, weight_layout::LayerLayout, MuseSparkConfig};
#[cfg(test)]
use cuteafd_loader::{families::muse_spark::reference::residual_gate_coefficients,
    formats::dense_trellis::LoadedDenseTrellis, LoadedTensor};
use std::collections::BTreeMap;

/// One rank's packed host operands of one layer.
#[cfg(test)]
pub(crate) struct PackedLayer {
    pub(crate) operands: BTreeMap<&'static str, Vec<u8>>,
    pub(crate) core: BTreeMap<&'static str, LoadedDenseTrellis>,
}

#[cfg(test)]
fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect()
}

/// Rows `[first, first + count)` of a row-major BF16 `[n, k]` matrix.
#[cfg(test)]
fn rows(bytes: &[u8], k: usize, first: usize, count: usize) -> Vec<u8> {
    bytes[first * k * 2..(first + count) * k * 2].to_vec()
}

/// Columns `[first, first + count)` of a row-major BF16 `[n, k]` matrix.
#[cfg(test)]
fn columns(bytes: &[u8], n: usize, k: usize, first: usize, count: usize) -> Vec<u8> {
    (0..n).flat_map(|row| bytes[(row * k + first) * 2..(row * k + first + count) * 2].iter().copied()).collect()
}

#[cfg(test)]
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

fn checked_rect(checkpoint: &cuteafd_loader::plan::Checkpoint, name: &str, dtype: DType,
    rows: Option<std::ops::Range<usize>>, cols: Option<std::ops::Range<usize>>) -> Result<Rect> {
    let rect = Rect::tensor(checkpoint, name, rows, cols)?;
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| anyhow::anyhow!("Muse checkpoint lacks {name}"))?;
    ensure!(checkpoint.tensors[at].meta.dtype == dtype, "Muse {name}: expected {dtype:?}");
    Ok(rect)
}

/// Exact plain operand ownership, independent of device allocation or payload I/O.
fn layer_rects(checkpoint: &cuteafd_loader::plan::Checkpoint, cfg: &MuseSparkConfig,
    layer: usize, rank: usize, ranks: usize, layout: &LayerLayout) -> Result<BTreeMap<&'static str, Rect>> {
    ensure!(matches!(ranks, 1 | 2) && rank < ranks, "Muse dense rank geometry");
    let root = format!("{PREFIX}{layer}");
    let mut rects = BTreeMap::new();
    for (name, stem, n, k, row_split) in [
        ("attn.q", "self_attn.q_proj", cfg.heads * cfg.head_dim, cfg.hidden, true),
        ("attn.k", "self_attn.k_proj", cfg.kv_heads * cfg.head_dim, cfg.hidden, true),
        ("attn.v", "self_attn.v_proj", cfg.kv_heads * cfg.head_dim, cfg.hidden, true),
        ("attn.gate", "self_attn.gate_proj", cfg.heads * cfg.head_dim, cfg.hidden, true),
        ("attn.o", "self_attn.o_proj", cfg.hidden, cfg.heads * cfg.head_dim, false),
        ("moe.pre_expert", "mlp.pre_expert_proj", cfg.expert_hidden, cfg.hidden, true),
        ("moe.post_expert", "mlp.post_expert_proj", cfg.hidden, cfg.expert_hidden, false),
    ] {
        if layout.core.contains_key(name) { continue; }
        let (rows, cols) = if row_split { (Some(rank * n / ranks..(rank + 1) * n / ranks), None) }
            else { (None, Some(rank * k * 2 / ranks..(rank + 1) * k * 2 / ranks)) };
        let tensor_name = format!("{root}.{stem}.weight");
        let rect = checked_rect(checkpoint, &tensor_name, DType::Bf16, rows, cols)?;
        ensure!(rect.row_bytes == k * 2, "Muse {tensor_name}: matrix row extent");
        rects.insert(name, rect);
    }
    for (name, stem, dtype) in [
        ("moe.router", "mlp.gate.weight", DType::F32),
        ("moe.bias", "mlp.gate.e_score_correction_bias", DType::F32),
        ("moe.pre_expert_norm", "mlp.pre_expert_norm.weight", DType::Bf16),
        ("moe.post_expert_norm", "mlp.experts.post_expert_norm.weight", DType::Bf16),
        ("norm.input", "input_layernorm.weight", DType::Bf16),
        ("norm.pre_ffn", "pre_feedforward_layernorm.weight", DType::Bf16),
        ("norm.post_ffn", "post_feedforward_layernorm.weight", DType::Bf16),
    ] {
        rects.insert(name, checked_rect(checkpoint, &format!("{root}.{stem}"), dtype, None, None)?);
    }
    ensure!(rects.iter().all(|(name, rect)| layout.operands.get(name) == Some(&rect.bytes())),
        "Muse dense rectangles differ from admitted operands");
    Ok(rects)
}

fn allocate<'a>(device: Device<'a>, bytes: usize) -> Result<Allocation<'a>> {
    let _scope = cuteafd_ffi::memory_ledger::scope("weights/muse dense");
    Allocation::new(device, bytes)
}

fn land<'a>(device: Device<'a>, dense: &DenseLoad<'a>, rect: Rect) -> Result<Allocation<'a>> {
    let bytes = rect.bytes();
    let allocation = allocate(device, bytes)?;
    dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands: vec![(rect, 0)], bytes })?;
    Ok(allocation)
}

/// One rank's resident layer weights.
pub(crate) struct LayerWeights<'a> {
    operands: BTreeMap<&'static str, Allocation<'a>>,
    core: BTreeMap<&'static str, CoreWeights<'a>>,
}

impl<'a> LayerWeights<'a> {
    fn load(device: Device<'a>, checkpoint: &cuteafd_loader::plan::Checkpoint, cfg: &MuseSparkConfig,
        layer: usize, rank: usize, ranks: usize, layout: &LayerLayout, dense: &DenseLoad<'a>) -> Result<Self> {
        let recipe = layer_rects(checkpoint, cfg, layer, rank, ranks, layout)?;
        let mut operands = BTreeMap::new();
        for (name, rect) in recipe {
            operands.insert(name, land(device, dense, rect)?);
        }
        for (stem, alpha, beta) in [
            ("post_attention_residual_gate", "gate.attention.alpha", "gate.attention.beta"),
            ("post_feedforward_residual_gate", "gate.ffn.alpha", "gate.ffn.beta"),
        ] {
            let rect = checked_rect(checkpoint, &format!("{PREFIX}{layer}.{stem}.gate"), DType::Bf16,
                None, None)?;
            ensure!(rect.bytes() == cfg.hidden * 2, "Muse residual gate source extent");
            let a = allocate(device, layout.operands[alpha])?;
            let b = allocate(device, layout.operands[beta])?;
            dense.step(Step { device: device.id, rects: vec![rect], jobs: vec![Job::ResidualGate {
                temperature: cfg.residual_gate_temperature as f32, alpha: a.buffer, beta: b.buffer,
            }] }, std::ptr::null_mut())?;
            operands.insert(alpha, a);
            operands.insert(beta, b);
        }
        let core = layout.core.iter().map(|(&name, shared)| {
            CoreWeightLayout::from_shared(shared.clone()).load_recorded(device, checkpoint, dense,
                std::ptr::null_mut()).map(|weights| (name, weights))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        ensure!(operands.len() == layout.operands.len() && operands.iter().all(|(name, allocation)|
            layout.operands.get(name) == Some(&allocation.buffer.bytes)), "Muse recorded operand inventory");
        Ok(Self { operands, core })
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self.operands.get(name).with_context(|| format!("Muse layer has no resident operand {name}"))?.buffer)
    }

    pub(crate) fn core(&self, name: &str) -> Option<&CoreWeights<'a>> { self.core.get(name) }
}

/// One rank's share of the global weights (`GlobalLayout`) beyond the token
/// embedding (the shared `TokenEmbedding` on the entry rank): the final norm
/// on every rank and the K6 head on the head rank (the last).
pub(crate) struct GlobalWeights<'a> {
    pub final_norm: Allocation<'a>,
    pub head: Option<CoreWeights<'a>>,
}

impl<'a> GlobalWeights<'a> {
    pub(crate) fn load(device: Device<'a>, checkpoint: &cuteafd_loader::plan::Checkpoint,
        layout: &cuteafd_loader::families::muse_spark::weight_layout::GlobalLayout, rank: usize, ranks: usize)
        -> Result<Self> {
        let root = PREFIX.trim_end_matches("layers.");
        let name = format!("{root}norm.weight");
        let dense = DenseLoad::mount(device.library);
        let rect = checked_rect(checkpoint, &name, DType::Bf16, None, None)?;
        ensure!(rect.bytes() == layout.final_norm, "Muse final norm differs from admitted bytes");
        let final_norm = land(device, &dense, rect)?;
        let head = if rank + 1 == ranks {
            let core = CoreWeightLayout::from_shared(layout.head.clone().context("Muse serves the K6 head")?);
            Some(core.load_recorded(device, checkpoint, &dense, std::ptr::null_mut())?)
        } else { None };
        Ok(Self { final_norm, head })
    }
}

/// Every layer of one coordinator rank: read through the shared checkpoint
/// reader, packed to the admitted layout and uploaded. `charged[layer]` is the
/// rank's layer bytes admission accepted (`LayerLayout::bytes`).
pub(crate) fn load_rank<'a>(device: Device<'a>, cfg: &MuseSparkConfig, checkpoint: &cuteafd_loader::plan::Checkpoint,
    layouts: &[Vec<LayerLayout>], rank: usize) -> Result<Vec<LayerWeights<'a>>> {
    let ranks = layouts.first().map_or(0, Vec::len);
    ensure!(layouts.len() == cfg.layers && rank < ranks, "Muse rank {rank} of {ranks} over {} layers", layouts.len());
    let dense = DenseLoad::mount(device.library);
    layouts.iter().enumerate().map(|(layer, per_rank)| {
        let layout = &per_rank[rank];
        LayerWeights::load(device, checkpoint, cfg, layer, rank, ranks, layout, &dense)
            .with_context(|| format!("Muse layer {layer} rank {rank}"))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_push_recipes_match_host_packing_on_one_and_two_ranks() -> Result<()> {
        use cuteafd_loader::plan::{Checkpoint, checkpoint::CheckpointTensor};
        use cuteafd_loader::push_load::builder::Builder;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/muse_spark/fixtures");
        let record: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("official.json"))?)?;
        let cfg = MuseSparkConfig::from_hf(&record["config"])?;
        let headers = cuteafd_loader::read_safetensors_metadata(&root.join("model.safetensors"))?;
        let mut checkpoint = Checkpoint { snapshot: root, config: record["config"].clone(), quantize_config: None,
            weight_map: headers.iter().map(|m| (m.name.clone(), "model.safetensors".into())).collect(),
            tensors: headers.into_iter().map(|meta| CheckpointTensor { shard: "model.safetensors".into(), meta }).collect(),
            missing_shards: vec![], shard_bytes: 0 };
        checkpoint.tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        let inventory = cuteafd_loader::families::muse_spark::core::CoreInventory::default();
        for ranks in [1, 2] {
            for layer in 0..cfg.layers {
                let layouts = LayerLayout::ranks(&cfg, layer, ranks, &inventory)?;
                for rank in 0..ranks {
                    let read = |name: &str| crate::shared::checkpoint_read::read_tensor(&checkpoint, name);
                    let expected = PackedLayer::pack(&cfg, layer, rank, ranks, &layouts[rank], &BTreeMap::new(), read)?;
                    let rects = layer_rects(&checkpoint, &cfg, layer, rank, ranks, &layouts[rank])?;
                    let bytes = rects.values().map(Rect::bytes).sum::<usize>();
                    let mut builder = Builder::new(&checkpoint.snapshot, vec![bytes as u64; ranks]);
                    let mut offset = 0;
                    for (name, rect) in &rects {
                        let mut actual = vec![0; rect.bytes()];
                        rect.read(&mut actual)?;
                        assert_eq!(actual, expected.operands[name], "layer {layer} rank {rank}: {name}");
                        crate::shared::experts::rtx::dense::append(rect, &mut builder, rank as u32, offset)?;
                        offset += rect.bytes() as u64;
                    }
                    builder.finish()?;
                    for (stem, a, b) in [
                        ("post_attention_residual_gate", "gate.attention.alpha", "gate.attention.beta"),
                        ("post_feedforward_residual_gate", "gate.ffn.alpha", "gate.ffn.beta"),
                    ] {
                        let gate = read(&format!("{PREFIX}{layer}.{stem}.gate"))?;
                        let (alpha, beta) = crate::shared::experts::rtx::dense::residual_gate_bytes(
                            &gate.bytes, cfg.residual_gate_temperature as f32)?;
                        assert_eq!(alpha, expected.operands[a]);
                        assert_eq!(beta, expected.operands[b]);
                    }
                    assert_eq!(rects.len() + 4, layouts[rank].operands.len());
                }
            }
        }
        // Metadata-only planning rejects a corrupt index instead of reading a different shard.
        checkpoint.weight_map.insert(format!("{PREFIX}0.self_attn.q_proj.weight"), "wrong.safetensors".into());
        let layout = LayerLayout::ranks(&cfg, 0, 1, &inventory)?.remove(0);
        assert!(layer_rects(&checkpoint, &cfg, 0, 0, 1, &layout).is_err());
        Ok(())
    }

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
