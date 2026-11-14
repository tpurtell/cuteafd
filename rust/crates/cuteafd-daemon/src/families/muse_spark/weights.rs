//! Checkpoint-native packing of one Muse Spark layer for one coordinator rank,
//! checked against the shared layout the planner charged
//! (`cuteafd_loader::families::muse_spark::weight_layout`), then filled by the
//! shared bounded mount/push records. The host packer remains the test oracle.
//! Head-split attention takes this rank's query/key/value/gate rows and its
//! o_proj columns; everything else is whole. The residual gates are packed as
//! F32 (alpha, beta) from the contract's host function, so the device kernel
//! only applies them.
use crate::shared::{dense_trellis::{CoreWeightLayout, CoreWeights},
    memory::{device::{Allocation, Device, DeviceOwner}, weight_arena::DeviceWeightArena, DeviceAllocation},
    native_attention::{attention_records, AttentionRecord, NativeAttentionWeights}};
use anyhow::{ensure, Context, Result};
use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step};
use cuteafd_core::DType;
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::families::muse_spark::{core::PREFIX, weight_layout::LayerLayout, MuseSparkConfig};
#[cfg(test)]
use cuteafd_loader::{families::muse_spark::reference::residual_gate_coefficients,
    formats::dense_trellis::LoadedDenseTrellis, LoadedTensor};
use std::collections::BTreeMap;
use cuteafd_core::memory_layout::ArenaTag;

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
        if layout.core.contains_key(name) || layout.native.contains_key(name) { continue; }
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

fn allocate<'a>(device: Device<'a>, arena: &DeviceWeightArena<'a>, name: &str, bytes: usize) -> Result<Allocation<'a>> {
    crate::shared::memory::weight_arena::weight_allocation(device, Some(arena), name, bytes, ArenaTag::weights("muse/target"))
}

fn land<'a>(device: Device<'a>, arena: &DeviceWeightArena<'a>, name: &str,
    dense: &DenseLoad<'a>, rect: Rect) -> Result<Allocation<'a>> {
    let bytes = rect.bytes();
    let allocation = allocate(device, arena, name, bytes)?;
    dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands: vec![(rect, 0)], bytes })?;
    Ok(allocation)
}

/// One rank's resident layer weights.
pub(crate) struct LayerWeights<'a> {
    operands: BTreeMap<&'static str, Allocation<'a>>,
    core: BTreeMap<&'static str, CoreWeights<'a>>,
    native: BTreeMap<&'static str, NativeWeights<'a>>,
}

pub(crate) type NativeWeights<'a> = DeviceOwner<'a, NativeAttentionWeights<'a>>;

fn allocate_native<'a>(device: Device<'a>, arena: &DeviceWeightArena<'a>, layout: &cuteafd_loader::formats::native_attention::NativeLayout)
    -> Result<NativeWeights<'a>> {
    device.own(|| {
        let _scope = cuteafd_ffi::memory_ledger::scope("weights/muse native attention");
        let [n, k] = layout.shape;
        let values = DeviceAllocation::from_allocation(allocate(device, arena, &format!("{}.values", layout.stem),
            n * k * if layout.fp8 { 1 } else { 2 })?);
        let scales = layout.fp8.then(|| allocate(device, arena, &format!("{}.scales", layout.stem),
            n.div_ceil(128) * k.div_ceil(128) * 4).map(DeviceAllocation::from_allocation)).transpose()?;
        Ok(NativeAttentionWeights { layout: layout.clone(), values, scales })
    })
}

impl<'a> LayerWeights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>, prefix: &str)
        -> Result<()> {
        for (name, owner) in &self.operands { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        for (name, owner) in &self.core {
            for (suffix, allocation) in [("trellis", &owner.payload), ("suh", &owner.suh), ("svh", &owner.svh)] {
                allocation.inventory(inventory, &format!("{prefix}.{name}.{suffix}"))?;
            }
        }
        for (name, owner) in &self.native {
            owner.values.inventory(inventory, &format!("{prefix}.{name}.values"))?;
            if let Some(scales) = &owner.scales { scales.inventory(inventory, &format!("{prefix}.{name}.scales"))?; }
        }
        Ok(())
    }

    fn load(device: Device<'a>, checkpoint: &cuteafd_loader::plan::Checkpoint, cfg: &MuseSparkConfig,
        layer: usize, rank: usize, ranks: usize, layout: &LayerLayout, dense: &DenseLoad<'a>,
        native_load: &DenseLoad<'a>, arena: &DeviceWeightArena<'a>) -> Result<Self> {
        let recipe = layer_rects(checkpoint, cfg, layer, rank, ranks, layout)?;
        let mut operands = BTreeMap::new();
        for (name, rect) in recipe {
            operands.insert(name, land(device, arena, &format!("layers.{layer}.{name}"), dense, rect)?);
        }
        for (stem, alpha, beta) in [
            ("post_attention_residual_gate", "gate.attention.alpha", "gate.attention.beta"),
            ("post_feedforward_residual_gate", "gate.ffn.alpha", "gate.ffn.beta"),
        ] {
            let rect = checked_rect(checkpoint, &format!("{PREFIX}{layer}.{stem}.gate"), DType::Bf16,
                None, None)?;
            ensure!(rect.bytes() == cfg.hidden * 2, "Muse residual gate source extent");
            let a = allocate(device, arena, &format!("layers.{layer}.{alpha}"), layout.operands[alpha])?;
            let b = allocate(device, arena, &format!("layers.{layer}.{beta}"), layout.operands[beta])?;
            dense.step(Step { device: device.id, rects: vec![rect], jobs: vec![Job::ResidualGate {
                temperature: cfg.residual_gate_temperature as f32, alpha: a.buffer, beta: b.buffer,
            }] }, std::ptr::null_mut())?;
            operands.insert(alpha, a);
            operands.insert(beta, b);
        }
        let core = layout.core.iter().map(|(&name, shared)| {
            CoreWeightLayout::from_shared(shared.clone()).load_recorded_with(device, checkpoint, dense,
                std::ptr::null_mut(), |name, bytes| allocate(device, arena, name, bytes)).map(|weights| (name, weights))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        ensure!(operands.len() == layout.operands.len() && operands.iter().all(|(name, allocation)|
            layout.operands.get(name) == Some(&allocation.buffer.bytes)), "Muse recorded operand inventory");
        let native = layout.native.iter().map(|(&name, layout)| {
            let weights = allocate_native(device, arena, layout)?;
            let selected = checkpoint.projection(&layout.stem)?;
            for record in attention_records(&checkpoint.snapshot, &selected, checkpoint.projection_view(&layout.stem), layout, cuteafd_loader::formats::native_attention::Fp8Role::exact("projection", true),
                crate::shared::fp8_linear::Fp8Scales::Amax, weights.values.buffer,
                weights.scales.as_ref().map(|scales| scales.buffer))? {
                match record {
                    AttentionRecord::Operand(operand) => native_load.operand(operand)?,
                    AttentionRecord::Step(step) => native_load.step(step, std::ptr::null_mut())?,
                }
            }
            Ok((name, weights))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self { operands, core, native })
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self.operands.get(name).with_context(|| format!("Muse layer has no resident operand {name}"))?.buffer)
    }

    pub(crate) fn core(&self, name: &str) -> Option<&CoreWeights<'a>> { self.core.get(name) }

    pub(crate) fn native(&self, name: &str) -> Option<&NativeWeights<'a>> { self.native.get(name) }
}

/// One rank's share of the global weights (`GlobalLayout`) beyond the token
/// embedding (the shared `TokenEmbedding` on the entry rank): the final norm
/// on every rank and the selected BF16/trellis head on the last rank.
pub(crate) struct GlobalWeights<'a> {
    pub final_norm: Allocation<'a>,
    pub head: Option<CoreWeights<'a>>,
    pub head_bf16: Option<Allocation<'a>>,
}

impl<'a> GlobalWeights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        self.final_norm.inventory(inventory, "global.final_norm")?;
        if let Some(head) = &self.head {
            for (suffix, owner) in [("trellis", &head.payload), ("suh", &head.suh), ("svh", &head.svh)] {
                owner.inventory(inventory, &format!("global.head.{suffix}"))?;
            }
        }
        if let Some(head) = &self.head_bf16 { head.inventory(inventory, "global.head")?; }
        Ok(())
    }

    pub(crate) fn load(device: Device<'a>, checkpoint: &cuteafd_loader::plan::Checkpoint,
        layout: &cuteafd_loader::families::muse_spark::weight_layout::GlobalLayout, rank: usize, ranks: usize,
        arena: &DeviceWeightArena<'a>)
        -> Result<Self> {
        let root = PREFIX.trim_end_matches("layers.");
        let name = format!("{root}norm.weight");
        let dense = DenseLoad::mount(device.library);
        let rect = checked_rect(checkpoint, &name, DType::Bf16, None, None)?;
        ensure!(rect.bytes() == layout.final_norm, "Muse final norm differs from admitted bytes");
        let final_norm = land(device, arena, "global.final_norm", &dense, rect)?;
        let (head, head_bf16) = if rank + 1 == ranks {
            if let Some(core) = &layout.head {
                let core = CoreWeightLayout::from_shared(core.clone());
                (Some(core.load_recorded_with(device, checkpoint, &dense, std::ptr::null_mut(),
                    |name, bytes| allocate(device, arena, name, bytes))?), None)
            } else {
                let rect = checked_rect(checkpoint, "lm_head.weight", DType::Bf16, None, None)?;
                ensure!(rect.bytes() == layout.head_bf16, "Muse BF16 head differs from admitted bytes");
                (None, Some(land(device, arena, "global.head", &dense, rect)?))
            }
        } else { (None, None) };
        Ok(Self { final_norm, head, head_bf16 })
    }
}

/// Every layer of one coordinator rank: read through the shared checkpoint
/// reader, packed to the admitted layout and uploaded. `charged[layer]` is the
/// rank's layer bytes admission accepted (`LayerLayout::bytes`).
pub(crate) fn load_rank<'a>(device: Device<'a>, cfg: &MuseSparkConfig, checkpoint: &cuteafd_loader::plan::Checkpoint,
    layouts: &[Vec<LayerLayout>], rank: usize, arena: &DeviceWeightArena<'a>) -> Result<Vec<LayerWeights<'a>>> {
    let ranks = layouts.first().map_or(0, Vec::len);
    ensure!(layouts.len() == cfg.layers && rank < ranks, "Muse rank {rank} of {ranks} over {} layers", layouts.len());
    let dense = DenseLoad::mount(device.library);
    let native_load = DenseLoad::mount(device.library);
    native_load.admit_mount(cuteafd_loader::families::muse_spark::weight_layout::native_mount_load(
        layouts.iter().flat_map(|per_rank| per_rank[rank].native.values()))?)?;
    layouts.iter().enumerate().map(|(layer, per_rank)| {
        let layout = &per_rank[rank];
        LayerWeights::load(device, checkpoint, cfg, layer, rank, ranks, layout, &dense, &native_load, arena)
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
        let mut checkpoint = Checkpoint { view: None, snapshot: root, config: record["config"].clone(), quantize_config: None,
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

    #[test]
    fn arena_layer_contents_match_unchanged_host_oracle() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::plan::{Checkpoint, checkpoint::CheckpointTensor};
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        for device in 0..2 { fixture.configure_memory(&library, device, 32 << 20)?; }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/muse_spark/fixtures");
        let record: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("official.json"))?)?;
        let cfg = MuseSparkConfig::from_hf(&record["config"])?;
        let headers = cuteafd_loader::read_safetensors_metadata(&root.join("model.safetensors"))?;
        let mut checkpoint = Checkpoint { view: None, snapshot: root, config: record["config"].clone(), quantize_config: None,
            weight_map: headers.iter().map(|m| (m.name.clone(), "model.safetensors".into())).collect(),
            tensors: headers.into_iter().map(|meta| CheckpointTensor { shard: "model.safetensors".into(), meta }).collect(),
            missing_shards: vec![], shard_bytes: 0 };
        checkpoint.tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        let inventory = cuteafd_loader::families::muse_spark::core::CoreInventory::default();
        let global = cuteafd_loader::families::muse_spark::weight_layout::GlobalLayout::new(&cfg, &inventory)?;
        for ranks in [1, 2] {
            let layouts = (0..cfg.layers).map(|layer| LayerLayout::ranks(&cfg, layer, ranks, &inventory))
                .collect::<Result<Vec<_>>>()?;
            for rank in 0..ranks {
                let device = Device { library: &library, id: rank as i32 };
                let local = layouts.iter().map(|per_rank| &per_rank[rank]).collect::<Vec<_>>();
                let arena = DeviceWeightArena::new(device,
                    cuteafd_loader::families::muse_spark::weight_layout::coordinator_arena(rank, ranks, &global, &local)?, crate::shared::memory::TEST_ARENA)?;
                let (embedding, weights) = if rank == 0 {
                    let tensor = checkpoint.tensors.iter().find(|tensor|
                        tensor.meta.name == "model.language_model.embed_tokens.weight").context("fixture embedding")?;
                    let source = crate::shared::token_io::EmbedSource::new(&checkpoint.snapshot, &tensor.shard,
                        &tensor.meta, cfg.hidden)?;
                    let (embedding, weights) = device.run(|| crate::shared::token_io::TokenEmbedding::load_with(
                        &library, source, crate::shared::memory::EmbedPlacement::Gpu,
                        || load_rank(device, &cfg, &checkpoint, &layouts, rank, &arena),
                        |name, bytes| allocate(device, &arena, name, bytes)))?;
                    (Some(embedding), weights)
                } else { (None, load_rank(device, &cfg, &checkpoint, &layouts, rank, &arena)?) };
                let globals = GlobalWeights::load(device, &checkpoint, &global, rank, ranks, &arena)?;
                assert!(globals.head.is_none());
                assert_eq!(globals.head_bf16.is_some(), rank + 1 == ranks);
                if let Some(head) = &globals.head_bf16 {
                    let expected = crate::shared::checkpoint_read::read_tensor(&checkpoint, "lm_head.weight")?;
                    let mut actual = vec![0; head.buffer.bytes];
                    device.run(|| library.copy_d2h(&mut actual, head.buffer))?;
                    assert_eq!(actual, expected.bytes);
                    assert_eq!(head.weight_backing().context("BF16 head lost arena")?.buffer.bytes, arena.layout().bytes());
                }
                let backing_bytes = arena.layout().bytes();
                drop(arena);
                let device_ids = (0..ranks as i32).collect::<Vec<_>>();
                let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&device_ids)?;
                let mut receipts = crate::shared::memory::receipts::Inventory::new(devices, "muse",
                    cuteafd_loader::placement::planner::Lifetime::RESIDENT, cuteafd_core::memory_layout::Category::Weights,
                    "Muse target weights");
                for (layer, weights) in weights.iter().enumerate() {
                    let expected = PackedLayer::pack(&cfg, layer, rank, ranks, &layouts[layer][rank], &BTreeMap::new(),
                        |name| crate::shared::checkpoint_read::read_tensor(&checkpoint, name))?;
                    weights.inventory(&mut receipts, &format!("layers.{layer}"))?;
                    for (name, bytes) in &expected.operands {
                        let owner = &weights.operands[name];
                        assert_eq!(owner.weight_backing().context("Muse view lost backing")?.buffer.bytes, backing_bytes);
                        let mut actual = vec![0; bytes.len()];
                        device.run(|| library.copy_d2h(&mut actual, owner.buffer))?;
                        assert_eq!(&actual, bytes, "TP{ranks} rank{rank} layer{layer} {name}");
                    }
                }
                globals.inventory(&mut receipts)?;
                let mut census = crate::shared::memory::receipts::Census::default();
                census.owners(receipts.receipts);
                if let Some(embedding) = &embedding {
                    assert_eq!(embedding.verify_resident()?, global.embedding);
                    census.owners(embedding.allocation_receipts(devices, "muse.embedding")?);
                }
                census.normalize(cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT)?;
                assert_eq!(census.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).count(), 1);
            }
        }
        assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
        Ok(())
    }

    #[test]
    fn arena_k6_and_native_contents_match_original_loads() -> Result<()> {
        use cuteafd_loader::{formats::{dense_trellis::TrellisSide, exl3_storage::Exl3Codebook,
            trellis_core::{CoreProjection, CoreSplit}}, plan::{testing, checkpoint::CoreView}};
        use std::io::{Seek, SeekFrom, Write};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        for device in 0..2 { fixture.configure_memory(&library, device, 128 << 20)?; }
        let dir = tempfile::tempdir()?;
        let q = format!("{PREFIX}0.self_attn.q_proj");
        let o = format!("{PREFIX}0.self_attn.o_proj");
        let head = cuteafd_loader::families::muse_spark::core::HEAD;
        let bf16 = format!("{PREFIX}0.self_attn.k_proj");
        let fp8 = format!("{PREFIX}0.self_attn.v_proj");
        let mut tensors = Vec::new();
        for stem in [&*q, &*o, head] { tensors.extend(testing::exl3(stem, 256, 256, 6)); }
        tensors.push(testing::t(format!("{bf16}.weight"), "BF16", &[256, 256]));
        tensors.extend(testing::fp8(&fp8, 256, 256, None));
        testing::write_snapshot(dir.path(), &serde_json::json!({}), &tensors, None);
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(dir.path())?;
        let mut file = std::fs::OpenOptions::new().write(true)
            .open(dir.path().join("model-00001-of-00001.safetensors"))?;
        for tensor in &checkpoint.tensors {
            let bytes = if tensor.meta.name.ends_with(".mcg") { Exl3Codebook::Mcg.multiplier().to_le_bytes().to_vec() }
                else if tensor.meta.dtype == DType::Bf16 {
                    (0..tensor.meta.byte_length / 2).flat_map(|i| {
                        let value = ((i % 23) as f32 - 11.0) / 16.0;
                        ((value.to_bits() >> 16) as u16).to_le_bytes()
                    }).collect()
                } else { (0..tensor.meta.byte_length).map(|i| (i.wrapping_mul(17) % 251) as u8).collect() };
            file.seek(SeekFrom::Start(tensor.meta.byte_offset))?;
            file.write_all(&bytes)?;
        }
        drop(file);
        for ranks in [1, 2] {
            for rank in 0..ranks {
                let device = Device { library: &library, id: rank as i32 };
                for compact in [false, true] {
                    let projection = |stem: &str, split, side, local_rank, local_ranks| {
                        let projection = CoreProjection { stem: stem.into(), bits: 6, codebook: Exl3Codebook::Mcg,
                            shape: [if stem == head { 193 } else { 256 }, 256], stored_outputs: 256,
                            split, source_bytes: 0 };
                        cuteafd_loader::formats::trellis_core::CoreWeightLayout::new(&projection, side, local_rank, local_ranks)
                    };
                    let cores = BTreeMap::from([
                        ("attn.q", projection(&q, CoreSplit::Output, TrellisSide::Output, rank, ranks)?),
                        ("attn.o", projection(&o, CoreSplit::Input, TrellisSide::Input, rank, ranks)?),
                    ]);
                    let native = BTreeMap::from([
                        ("attn.k", cuteafd_loader::families::muse_spark::core::NativeLayout { stem: bf16.clone(),
                            shape: [256 / ranks, 256], fp8: compact, rank, ranks, split: CoreSplit::Output }),
                        ("attn.v", cuteafd_loader::families::muse_spark::core::NativeLayout { stem: fp8.clone(),
                            shape: [256, 256 / ranks], fp8: true, rank, ranks, split: CoreSplit::Input }),
                    ]);
                    let layer = LayerLayout { operands: BTreeMap::new(), core: cores, native };
                    let global = cuteafd_loader::families::muse_spark::weight_layout::GlobalLayout {
                        embedding: 512, final_norm: 512, head: Some(projection(head, CoreSplit::Whole,
                            TrellisSide::Output, 0, 1)?), head_bf16: 0 };
                    let arena = DeviceWeightArena::new(device,
                        cuteafd_loader::families::muse_spark::weight_layout::coordinator_arena(rank, ranks, &global, &[&layer])?, crate::shared::memory::TEST_ARENA)?;
                    let dense = DenseLoad::mount(&library);
                    for shared in layer.core.values().chain((rank + 1 == ranks).then_some(global.head.as_ref().unwrap())) {
                        let layout = CoreWeightLayout::from_shared(shared.clone());
                        let expected = layout.read(|name| crate::shared::checkpoint_read::read_tensor(&checkpoint, name))?;
                        let actual = layout.load_recorded_with(device, &checkpoint, &dense, std::ptr::null_mut(),
                            |name, bytes| allocate(device, &arena, name, bytes))?;
                        for (owner, bytes) in [(&actual.payload, &expected.shard.payload),
                            (&actual.suh, &expected.shard.suh), (&actual.svh, &expected.shard.svh)] {
                            let mut landed = vec![0; bytes.len()];
                            device.run(|| library.copy_d2h(&mut landed, owner.buffer))?;
                            assert_eq!(&landed, bytes, "TP{ranks} rank{rank} {}", shared.stem);
                        }
                    }
                    for layout in layer.native.values() {
                        let actual = allocate_native(device, &arena, layout)?;
                        let expected = device.own(|| Ok(NativeAttentionWeights { layout: layout.clone(),
                            values: DeviceAllocation::new(&library, actual.values.buffer.bytes, crate::shared::memory::TEST)?,
                            scales: actual.scales.as_ref().map(|s| DeviceAllocation::new(&library, s.buffer.bytes, crate::shared::memory::TEST)).transpose()? }))?;
                        let selected = checkpoint.projection(&layout.stem)?;
                        let mode = if compact && layout.stem == bf16 { CoreView::Fp8Compact } else { CoreView::Original };
                        for owner in [&actual, &expected] {
                            device.run(|| {
                                for record in attention_records(dir.path(), &selected, mode, layout, cuteafd_loader::formats::native_attention::Fp8Role::exact("projection", true),
                                    crate::shared::fp8_linear::Fp8Scales::Amax, owner.values.buffer,
                                    owner.scales.as_ref().map(|s| s.buffer))? {
                                    match record {
                                        AttentionRecord::Operand(operand) => dense.operand(operand)?,
                                        AttentionRecord::Step(step) => dense.step(step, std::ptr::null_mut())?,
                                    }
                                }
                                Ok(())
                            })?;
                        }
                        for (actual, expected) in std::iter::once((&actual.values, &expected.values))
                            .chain(actual.scales.iter().zip(expected.scales.iter())) {
                            let mut a = vec![0; actual.buffer.bytes];
                            let mut b = vec![0; expected.buffer.bytes];
                            device.run(|| { library.copy_d2h(&mut a, actual.buffer)?; library.copy_d2h(&mut b, expected.buffer) })?;
                            assert_eq!(a, b, "native TP{ranks} rank{rank} {} compact={compact}", layout.stem);
                        }
                    }
                    assert_eq!(fixture.device_free_mismatch(&library)?.0, 0, "native owner drops");
                    drop(arena);
                    assert_eq!(fixture.device_free_mismatch(&library)?.0, 0, "last arena root drop");
                    drop(dense);
                    assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1),
                        "dense mount scratch drop TP{ranks} rank{rank} compact={compact}");
                }
            }
        }
        assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
        Ok(())
    }

    #[test]
    fn shared_native_tp2_owners_destroy_values_and_scales_on_their_device() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::formats::{native_attention::NativeLayout, trellis_core::CoreSplit};
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.strict_device_free(&library, true)?;
        for fp8 in [false, true] {
            let owners = (0..2).map(|rank| {
                let device = Device { library: &library, id: rank };
                let layout = NativeLayout { stem: format!("o.{rank}"), shape: [128, 128], fp8,
                    rank: rank as usize, ranks: 2, split: CoreSplit::Input };
                let mut tensors = vec![(format!("{}.values", layout.stem), 128 * 128 * if fp8 { 1 } else { 2 })];
                if fp8 { tensors.push((format!("{}.scales", layout.stem), 4)); }
                let arena = DeviceWeightArena::new(device,
                    cuteafd_loader::serving_capacity::WeightArenaLayout::new(format!("muse-test/{rank}"), tensors)?, crate::shared::memory::TEST_ARENA)?;
                allocate_native(device, &arena, &layout)
            }).collect::<Result<Vec<_>>>()?;
            for owner in owners {
                let rank = owner.device.id;
                assert_eq!(owner.values.buffer.device_id, rank);
                assert_eq!(owner.scales.as_ref().map(|scales| scales.buffer.device_id), fp8.then_some(rank));
                library.cuda_set_device(1 - rank)?;
                drop(owner);
                assert_eq!(library.cuda_get_device()?, 1 - rank);
            }
        }
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    #[test]
    fn native_attention_records_match_mode_residency_and_admitted_load() -> Result<()> {
        use cuteafd_loader::{families::muse_spark::core::{CoreSplit, NativeLayout},
            plan::{checkpoint::{CoreView, CheckpointView}, testing, Checkpoint}};
        use crate::shared::native_attention::{attention_records, AttentionRecord};
        let dir = tempfile::tempdir()?;
        let stem = format!("{PREFIX}0.self_attn.o_proj");
        let mut config = testing::muse_spark_config();
        config["quantization_config"] = serde_json::json!({"quant_method":"exl3"});
        testing::write_mc_snapshot(dir.path(), &config,
            &[testing::t(format!("{stem}.weight"), "BF16", &[8192, 8192])],
            &BTreeMap::from([(format!("{stem}.weight"), vec![8192, 8192])]));
        for mode in [CoreView::Original, CoreView::Fp8Compact] {
            let view = CheckpointView::select(dir.path(), mode)?;
            let checkpoint = Checkpoint::open_view(&view)?;
            let selected = checkpoint.projection(&stem)?;
            for ranks in [1, 2] {
                for rank in 0..ranks {
                    let k = 8192 / ranks;
                    let fp8 = mode == CoreView::Fp8Compact;
                    let layout = NativeLayout { stem: stem.clone(), shape: [8192, k], fp8,
                        split: CoreSplit::Input, rank, ranks };
                    let values = CuteafdDeviceBuffer { ptr: 0x1000_0000usize as *mut _,
                        bytes: 8192 * k * if fp8 { 1 } else { 2 }, device_id: rank as i32, flags: 0 };
                    let scales = fp8.then_some(CuteafdDeviceBuffer { ptr: 0x8000_0000usize as *mut _,
                        bytes: 64 * (k / 128) * 4, ..values });
                    let receipt = cuteafd_loader::families::muse_spark::weight_layout::native_mount_load([&layout])?;
                    let records = attention_records(dir.path(), &selected, mode, &layout, cuteafd_loader::formats::native_attention::Fp8Role::exact("projection", true),
                        crate::shared::fp8_linear::Fp8Scales::Amax, values, scales)?;
                    // Conversion windows are whole 128-row scale blocks, bounded above
                    // by STEP_BYTES; smaller shared transform windows are also valid.
                    assert!(!records.is_empty());
                    if !fp8 { assert_eq!(records.len(), 1); }
                    let mut converted_rows = 0;
                    for record in records {
                        match record {
                            AttentionRecord::Operand(operand) => {
                                assert!(!fp8);
                                receipt.validate(operand.mount_load()?)?;
                                assert!(operand.lands.iter().all(|(rect, _)| rect.name == format!("{stem}.weight")));
                            }
                            AttentionRecord::Step(step) => {
                                assert!(fp8);
                                let rows = step.rects.iter().map(|rect| rect.rows.len()).sum::<usize>();
                                assert_eq!(rows % 128, 0);
                                assert!(rows * k * 2 <= cuteafd_loader::shared::dense_read::STEP_BYTES);
                                converted_rows += rows;
                                receipt.validate(step.mount_load()?)?;
                                assert!(step.rects.iter().all(|rect| rect.name == format!("{stem}.weight")));
                                assert!(step.jobs.iter().all(|job| matches!(job, Job::Quant {
                                    layout: crate::shared::fp8_load_quant::Layout::Block, .. })));
                            }
                        }
                    }
                    if fp8 { assert_eq!(converted_rows, 8192); }
                }
            }
        }
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
