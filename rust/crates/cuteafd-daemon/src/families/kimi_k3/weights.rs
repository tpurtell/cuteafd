//! Checkpoint-native layer packing and device owners for the admitted W9 engine.
//! Pack one layer at a time; allocation starts only after the caller's shared
//! placement admission and an exact comparison with that layer's charged bytes.
use crate::shared::{memory::device::{Allocation, Device}, placement::layers::AttentionShare};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, memory_layout::{Basis, Category}};
use cuteafd_ffi::{CuteafdDeviceBuffer, KimiKdaWeights, KimiMlaWeights};
use cuteafd_loader::{families::kimi_k3::{config::{KimiAttention, KimiK3Config}, kda, mla,
    w8::{self, ResidualWeightSite}}, formats::mla::MlaOperand, kda::{KdaOperand, KdaTensor}, placement::{Demand, LayerMode}, LoadedTensor};
use std::collections::BTreeMap;
use cuteafd_loader::families::kimi_k3::core::CoreInventory;
use cuteafd_loader::formats::dense_trellis::{LoadedDenseTrellis, TrellisSide};
use crate::shared::dense_trellis::{CoreWeightLayout, CoreWeights};

/// One canonical front end per routed ordinal, independent of attention placement.
pub(crate) use cuteafd_loader::families::kimi_k3::weight_layout::routed_owner;

/// Reads only the coordinator's header-selected tensors through the standard
/// checkpoint inventory (the shared reader). Routed payloads remain on the Spark loading path.
pub(crate) fn read_tensor(checkpoint: &cuteafd_loader::plan::Checkpoint, name: &str) -> Result<LoadedTensor> {
    crate::shared::checkpoint_read::read_tensor(checkpoint, name)
}

/// Global tensors follow the shared entry/head ranks: one embedding on rank 0,
/// one untied full vocabulary head and output mix/norm on rank 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GlobalLayout {
    rank: usize,
    hidden: usize,
    vocab: usize,
    operands: BTreeMap<&'static str, usize>,
    core_head: Option<CoreWeightLayout>,
}

impl GlobalLayout {
    pub(crate) fn pair(cfg: &KimiK3Config) -> Result<[Self; 2]> {
        Self::pair_with_core(cfg, &CoreInventory::default())
    }
    pub(crate) fn pair_with_core(cfg: &KimiK3Config, inventory: &CoreInventory) -> Result<[Self; 2]> {
        Ok(cuteafd_loader::families::kimi_k3::weight_layout::GlobalLayout::pair_with_core(cfg, inventory)?
            .map(|layout| Self { rank: layout.rank, hidden: layout.hidden, vocab: layout.vocab,
                operands: layout.operands, core_head: layout.core_head.map(CoreWeightLayout::from_shared) }))
    }

    pub(crate) fn bytes(&self) -> Result<usize> {
        self.operands.values().copied().chain(self.core_head.iter().map(|head|head.bytes))
            .try_fold(0usize, |sum, bytes| sum.checked_add(bytes)
            .context("Kimi global weight extent overflow"))
    }

    pub(crate) fn demand(&self) -> Result<Demand> {
        Ok(Demand::new(self.rank as u8, Category::Weights, "Kimi global weights", self.bytes()? as u64, Basis::Exact))
    }

    pub(crate) fn pack(&self, cfg: &KimiK3Config, mut read: impl FnMut(&str) -> Result<LoadedTensor>) -> Result<PackedGlobal> {
        ensure!(self.rank<2,"Kimi global rank outside admission");
        let mut expected=Self::pair(cfg)?;
        if let Some(head)=&self.core_head {
            ensure!(self.rank==1 && head.stem=="language_model.lm_head" && head.shape==[self.vocab,self.hidden]
                && head.side==TrellisSide::Output && head.rank==0 && head.ranks==1,"Kimi global K6 head geometry");
            expected[1].operands.remove("head");
            expected[1].core_head=Some(head.clone());
        }
        ensure!(expected[self.rank] == *self, "Kimi global packing differs from admitted geometry");
        let mut operands=BTreeMap::new();
        let core_head=if let Some(head)=&self.core_head {Some(head.read(&mut read)?)} else {
            let name = if self.rank == 0 { "language_model.model.embed_tokens.weight" } else { "language_model.lm_head.weight" };
            let tensor=read(name)?;
            check(&tensor,DType::Bf16,&[self.vocab,self.hidden])?;
            operands.insert(if self.rank==0 {"embedding"} else {"head"},tensor.bytes);
            None
        };
        if self.rank == 1 {
            let output = w8::load_residual(cfg, ResidualWeightSite::Output, &mut read)?;
            operands.insert("output.score", output.score);
            operands.insert("output.norm", output.norm.bytes);
        }
        let packed = PackedGlobal { rank: self.rank, operands, core_head };
        self.check(&packed)?;
        Ok(packed)
    }

    fn check(&self, packed: &PackedGlobal) -> Result<()> {
        ensure!(packed.rank == self.rank && packed.operands.len() == self.operands.len()
            && self.operands.iter().all(|(name, bytes)| packed.operands.get(name).is_some_and(|data| data.len() == *bytes)),
            "Kimi packed global operands differ from admission");
        ensure!(self.core_head.is_some()==packed.core_head.is_some(),"Kimi packed head format differs from admission");
        if let (Some(layout),Some(head))=(&self.core_head,&packed.core_head) {layout.check(head)?;}
        Ok(())
    }

    pub(super) fn upload<'a>(&self, device: Device<'a>, packed: PackedGlobal) -> Result<GlobalWeights<'a>> {
        self.check(&packed)?;
        let operands = packed.operands.into_iter().map(|(name, bytes)| {
            let _scope = cuteafd_ffi::memory_ledger::scope("weights/kimi global weights");
            let allocation = Allocation::new(device, bytes.len())?;
            ensure!(allocation.buffer.bytes == bytes.len() && allocation.buffer.device_id == device.id,
                "Kimi global allocation differs from admitted extent/device");
            device.run(|| device.library.copy_h2d(allocation.buffer, &bytes))?;
            Ok((name, allocation))
        }).collect::<Result<_>>()?;
        let core_head=packed.core_head.map(|head|self.core_head.as_ref().unwrap().upload(device,head)).transpose()?;
        Ok(GlobalWeights { operands, core_head })
    }
}

pub(crate) struct PackedGlobal {
    rank: usize,
    operands: BTreeMap<&'static str, Vec<u8>>,
    core_head: Option<LoadedDenseTrellis>,
}

pub(crate) struct GlobalWeights<'a> {
    operands: BTreeMap<&'static str, Allocation<'a>>,
    pub(crate) core_head: Option<CoreWeights<'a>>,
}

impl GlobalWeights<'_> {
    pub(super) fn check_device(&self, device: Device<'_>) -> Result<()> {
        ensure!(self.operands.values().chain(self.core_head.iter().flat_map(|weights|
            [&weights.payload,&weights.suh,&weights.svh])).all(|allocation| allocation.device.id == device.id
            && std::ptr::eq(allocation.device.library, device.library)), "Kimi global weight owner differs from IO device");
        Ok(())
    }

    pub(crate) fn borrowed_buffer(&self, name: &str) -> Result<&CuteafdDeviceBuffer> {
        Ok(&self.operands.get(name).with_context(|| format!("Kimi global operand {name} absent on this rank"))?.buffer)
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self.operands.get(name).with_context(|| format!("Kimi global operand {name} absent on this rank"))?.buffer)
    }
}

/// Exact resident operand extents, computed without reading tensor payloads.
/// Shared startup admission charges these bytes; packing independently checks
/// every operand before any device allocation can begin.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LayerLayout {
    operands: BTreeMap<&'static str, usize>,
    pub(crate) attention: AttentionShare,
    core: BTreeMap<&'static str, CoreWeightLayout>,
}

impl LayerLayout {
    pub(crate) fn pair(cfg: &KimiK3Config, layer: usize, mode: LayerMode) -> Result<[Self; 2]> {
        Self::pair_with_core(cfg, layer, mode, &CoreInventory::default())
    }
    pub(crate) fn pair_with_core(cfg: &KimiK3Config, layer: usize, mode: LayerMode,
        inventory: &CoreInventory) -> Result<[Self; 2]> {
        use cuteafd_loader::families::kimi_k3::weight_layout::{LayerLayout as Shared, WeightAttention};
        Ok(Shared::pair_with_core(cfg, layer, mode, inventory)?.map(|layout| Self {
            operands: layout.operands,
            attention: match layout.attention { WeightAttention::Absent => AttentionShare::Absent,
                WeightAttention::Whole => AttentionShare::Whole, WeightAttention::Split => AttentionShare::Split },
            core: layout.core.into_iter().map(|(name, layout)| (name, CoreWeightLayout::from_shared(layout))).collect(),
        }))
    }

    pub(crate) fn core_layouts(&self) -> impl Iterator<Item=&CoreWeightLayout> { self.core.values() }

    pub(crate) fn bytes(&self) -> Result<usize> {
        self.operands.values().copied().chain(self.core.values().map(|layout| layout.bytes))
            .try_fold(0usize, |n, size| n.checked_add(size).context("Kimi layer admission extent overflow"))
    }

    pub(super) fn attention_bytes(&self) -> Result<usize> {
        self.operands.iter().filter(|(name, _)| name.starts_with("mla.") || name.starts_with("kda."))
            .map(|(_, bytes)| *bytes).chain(self.core.iter().filter(|(name, _)|
                name.starts_with("mla.") || name.starts_with("kda.")).map(|(_, layout)| layout.bytes))
            .try_fold(0usize, |n, bytes| n.checked_add(bytes).context("Kimi attention extent overflow"))
    }

    fn check(&self, packed: &PackedLayer) -> Result<()> {
        ensure!(self.attention == packed.attention && self.operands.len() == packed.operands.len()
            && self.core.len() == packed.core.len(),
            "Kimi packed operand inventory differs from native admission");
        for (name, bytes) in &self.operands {
            ensure!(packed.operands.get(name).is_some_and(|data| data.len() == *bytes),
                "Kimi packed operand {name} differs from native admission");
        }
        for (name, layout) in &self.core {
            layout.check(packed.core.get(name).context("Kimi packed core operand absent")?)?;
        }
        Ok(())
    }
}

fn extent(dimensions: &[usize]) -> Result<usize> {
    ensure!(dimensions.iter().all(|n| *n > 0), "Kimi weight extent is empty");
    dimensions.iter().try_fold(1usize, |n, dim| n.checked_mul(*dim)).context("Kimi weight extent overflow")
}

/// Byte counts describe the packed operands, not the checkpoint's pre-pack
/// representation (the residual score product, for example, is F32).
pub(crate) struct PackedLayer {
    operands: BTreeMap<&'static str, Vec<u8>>,
    attention: AttentionShare,
    core: BTreeMap<&'static str, LoadedDenseTrellis>,
}

impl PackedLayer {
    pub(crate) fn bytes(&self) -> Result<usize> {
        self.operands.values().map(Vec::len).chain(self.core.values().map(|loaded|
            loaded.shard.payload.len() + loaded.shard.suh.len() + loaded.shard.svh.len()))
            .try_fold(0usize, |total, bytes| total.checked_add(bytes).context("Kimi packed layer size overflow"))
    }

    /// Both ranks are packed together so a source operand is read once, rather
    /// than rereading every tensor separately for each coordinator GPU.
    pub(crate) fn pair(cfg: &KimiK3Config, layer: usize, mode: LayerMode,
        mut read: impl FnMut(&str) -> Result<LoadedTensor>) -> Result<[Self; 2]> {
        Self::pair_with_core(cfg, layer, mode, &CoreInventory::default(), &mut read)
    }

    pub(crate) fn pair_with_core(cfg: &KimiK3Config, layer: usize, mode: LayerMode,
        inventory: &CoreInventory, mut read: impl FnMut(&str) -> Result<LoadedTensor>) -> Result<[Self; 2]> {
        let layouts = LayerLayout::pair_with_core(cfg, layer, mode, inventory)?;
        let kind = cfg.attention[layer];
        let shares = layouts.each_ref().map(|layout| layout.attention);
        let mut pair = std::array::from_fn(|rank| Self {
            operands: BTreeMap::new(), attention: shares[rank], core: BTreeMap::new() });
        for (site, score, norm) in [
            (ResidualWeightSite::Attention(layer), "attention.score", "attention.norm"),
            (ResidualWeightSite::Ffn(layer), "ffn.score", "ffn.norm"),
        ] {
            let weights = w8::load_residual(cfg, site, &mut read)?;
            for rank in &mut pair {
                rank.operands.insert(score, weights.score.clone());
                rank.operands.insert(norm, weights.norm.bytes.clone());
            }
        }
        if layouts[0].operands.contains_key("ffn.gate_up") {
            for (rank, slice) in pair.iter_mut().zip(w8::pack_ffn(cfg, layer, 2, &mut read)?) {
                rank.operands.insert("ffn.gate_up", slice.gate_up);
                rank.operands.insert("ffn.down", slice.down);
            }
        } else {
            let root = format!("language_model.model.layers.{layer}.block_sparse_moe.shared_experts");
            let width = cfg.moe_intermediate * cfg.shared_experts;
            for (name, leaf, shape, side) in [
                ("ffn.gate", "gate_proj", [width, cfg.hidden], TrellisSide::Output),
                ("ffn.up", "up_proj", [width, cfg.hidden], TrellisSide::Output),
                ("ffn.down", "down_proj", [cfg.hidden, width], TrellisSide::Input),
            ] { pack_projection_pair(&layouts, &mut pair, name, &format!("{root}.{leaf}"), shape, side, &[0, 1], &mut read)?; }
        }
        if !cfg.dense[layer] {
            let owner=routed_owner(cfg,layer)?;
            let root=format!("language_model.model.layers.{layer}.block_sparse_moe");
            if layouts[owner].core.contains_key("latent.down") {
                for (name,leaf,shape) in [("latent.down","routed_expert_down_proj",[cfg.latent,cfg.hidden]),
                    ("latent.up","routed_expert_up_proj",[cfg.hidden,cfg.latent])] {
                    pack_projection_pair(&layouts,&mut pair,name,&format!("{root}.{leaf}"),shape,TrellisSide::Output,
                        &[owner],&mut read)?;
                }
                let norm=read(&format!("{root}.routed_expert_norm.weight"))?;
                check(&norm,DType::Bf16,&[cfg.latent])?;
                pair[owner].operands.insert("latent.norm",norm.bytes);
            } else {
                let latent=w8::load_latent(cfg,layer,&mut read)?;
                for (name,bytes) in [("latent.down",latent.down.bytes),("latent.norm",latent.norm.bytes),
                    ("latent.up",latent.up.bytes)] {pair[owner].operands.insert(name,bytes);}
            }
            let router=read(&format!("{root}.gate.weight"))?;
            let bias=read(&format!("{root}.gate.e_score_correction_bias"))?;
            check(&router,DType::Bf16,&[cfg.experts,cfg.hidden])?;
            check(&bias,DType::F32,&[cfg.experts])?;
            pair[owner].operands.insert("router.weight",router.bytes);
            pair[owner].operands.insert("router.bias",bias.bytes);
        }
        match kind {
            KimiAttention::Kda => {
                for (operand, name) in [(KdaOperand::WIn, "kda.input"), (KdaOperand::WFg, "kda.forget"),
                    (KdaOperand::ConvW, "kda.conv"), (KdaOperand::ALog, "kda.a_log"),
                    (KdaOperand::DtBias, "kda.dt_bias"), (KdaOperand::ONorm, "kda.norm"), (KdaOperand::WO, "kda.output")] {
                    if !layouts[0].operands.contains_key(name) || (operand == KdaOperand::WO && !layouts[0].operands.contains_key("kda.input")) { continue; }
                    let slices = kda::pack(cfg, layer, operand, 2, |name| {
                        let tensor = read(name)?;
                        Ok(KdaTensor { dtype: tensor.info.dtype, shape: tensor.info.shape, bytes: tensor.bytes })
                    })?;
                    for (rank, slice) in pair.iter_mut().zip(slices) { rank.operands.insert(name, slice.bytes); }
                }
                if !layouts[0].operands.contains_key("kda.input") {
                    let root = format!("language_model.model.layers.{layer}.self_attn");
                    let width = cfg.kda_heads * cfg.kda_head_dim;
                    let fa = read(&format!("{root}.f_a_proj.weight"))?;
                    let beta = read(&format!("{root}.b_proj.weight"))?;
                    check(&fa, DType::Bf16, &[cfg.kda_head_dim, cfg.hidden])?;
                    check(&beta, DType::Bf16, &[cfg.kda_heads, cfg.hidden])?;
                    let share = beta.bytes.len() / 2;
                    for (rank, packed) in pair.iter_mut().enumerate() {
                        let mut fab = fa.bytes.clone();
                        fab.extend_from_slice(&beta.bytes[rank * share..(rank + 1) * share]);
                        packed.operands.insert("kda.fab", fab);
                    }
                    for (name, leaf, shape, side) in [
                        ("kda.q", "q_proj", [width, cfg.hidden], TrellisSide::Output),
                        ("kda.k", "k_proj", [width, cfg.hidden], TrellisSide::Output),
                        ("kda.v", "v_proj", [width, cfg.hidden], TrellisSide::Output),
                        ("kda.gate", "g_proj", [width, cfg.hidden], TrellisSide::Output),
                        ("kda.output", "o_proj", [cfg.hidden, width], TrellisSide::Input),
                    ] { pack_projection_pair(&layouts, &mut pair, name, &format!("{root}.{leaf}"), shape, side, &[0, 1], &mut read)?; }
                }
            }
            KimiAttention::Mla => {
                let owners: Vec<_> = shares.iter().enumerate().filter(|(_, share)| share.present())
                    .map(|(rank, _)| rank).collect();
                for (operand, name) in [(MlaOperand::QkvA, "mla.qkv_a"), (MlaOperand::QNorm, "mla.q_norm"),
                    (MlaOperand::KvNorm, "mla.kv_norm"), (MlaOperand::QB, "mla.q_b"), (MlaOperand::Uk, "mla.uk"),
                    (MlaOperand::Uv, "mla.uv"), (MlaOperand::Gate, "mla.gate"), (MlaOperand::Output, "mla.output")] {
                    if !layouts[owners[0]].operands.contains_key(name)
                        || ([MlaOperand::QB, MlaOperand::Gate, MlaOperand::Output].contains(&operand)
                            && !layouts[owners[0]].operands.contains_key("mla.qkv_a")) { continue; }
                    let slices = mla::pack(cfg, layer, operand, owners.len(), &mut read)?;
                    for (&owner, slice) in owners.iter().zip(slices) { pair[owner].operands.insert(name, slice.bytes); }
                }
                if !layouts[owners[0]].operands.contains_key("mla.qkv_a") {
                    let root = format!("language_model.model.layers.{layer}.self_attn");
                    for (name, leaf, shape, side, replicated) in [
                        ("mla.q_a", "q_a_proj", [cfg.q_lora_rank, cfg.hidden], TrellisSide::Output, true),
                        ("mla.kv_a", "kv_a_proj_with_mqa", [cfg.kv_lora_rank + cfg.qk_rope_dim, cfg.hidden], TrellisSide::Output, true),
                        ("mla.q_b", "q_b_proj", [cfg.heads * (cfg.qk_nope_dim + cfg.qk_rope_dim), cfg.q_lora_rank], TrellisSide::Output, false),
                        ("mla.gate", "g_proj", [cfg.heads * cfg.v_head_dim, cfg.hidden], TrellisSide::Output, false),
                        ("mla.output", "o_proj", [cfg.hidden, cfg.heads * cfg.v_head_dim], TrellisSide::Input, false),
                    ] {
                        if replicated {
                            for &owner in &owners {
                                pack_projection_pair(&layouts, &mut pair, name, &format!("{root}.{leaf}"), shape, side, &[owner], &mut read)?;
                            }
                        } else { pack_projection_pair(&layouts, &mut pair, name, &format!("{root}.{leaf}"), shape, side, &owners, &mut read)?; }
                    }
                }
            }
        }
        for (layout, packed) in layouts.iter().zip(&pair) { layout.check(packed)?; }
        Ok(pair)
    }
}

fn pack_projection_pair(layouts: &[LayerLayout; 2], packed: &mut [PackedLayer; 2], name: &'static str,
    stem: &str, shape: [usize; 2], side: TrellisSide, owners: &[usize],
    read: &mut impl FnMut(&str) -> Result<LoadedTensor>) -> Result<()> {
    ensure!(!owners.is_empty() && owners.iter().all(|&rank| rank < 2), "Kimi projection owners");
    if let Some(layout) = layouts[owners[0]].core.get(name) {
        let mut tensors = BTreeMap::new();
        for suffix in ["trellis", "suh", "svh", layout.codebook.marker()] {
            tensors.insert(format!("{stem}.{suffix}"), read(&format!("{stem}.{suffix}"))?);
        }
        for &owner in owners {
            packed[owner].core.insert(name, layouts[owner].core[name].read(|name|
                tensors.get(name).cloned().context("Kimi K6 source member absent"))?);
        }
    } else {
        let tensor = read(&format!("{stem}.weight"))?;
        check(&tensor, DType::Bf16, &shape)?;
        let [n, k] = shape;
        let ranks = owners.len();
        ensure!(!owners.is_empty() && match side { TrellisSide::Output => n % ranks == 0,
            TrellisSide::Input => k % ranks == 0 }, "Kimi BF16 mixed projection shard geometry");
        for (rank, &owner) in owners.iter().enumerate() {
            let bytes = match side {
                TrellisSide::Output => {
                    let share = tensor.bytes.len() / ranks;
                    tensor.bytes[rank * share..(rank + 1) * share].to_vec()
                }
                TrellisSide::Input => {
                    let share = k * 2 / ranks;
                    tensor.bytes.chunks_exact(k * 2).flat_map(|row|
                        row[rank * share..(rank + 1) * share].iter().copied()).collect()
                }
            };
            packed[owner].operands.insert(name, bytes);
        }
    }
    Ok(())
}

fn check(tensor: &LoadedTensor, dtype: DType, shape: &[usize]) -> Result<()> {
    let element = if dtype == DType::F32 { 4usize } else { 2 };
    let bytes = shape.iter().try_fold(element, |size, &dim| size.checked_mul(dim))
        .context("Kimi tensor extent overflow")?;
    ensure!(tensor.info.dtype == dtype && tensor.info.shape == shape && tensor.bytes.len() == bytes,
        "Kimi tensor {} expected {dtype:?} {shape:?} with {bytes} bytes", tensor.info.name);
    Ok(())
}

pub(crate) struct LayerWeights<'a> {
    operands: BTreeMap<&'static str, Allocation<'a>>,
    pub(crate) attention: AttentionShare,
    core: BTreeMap<&'static str, CoreWeights<'a>>,
}

impl<'a> LayerWeights<'a> {
    pub(super) fn check_device(&self, device: Device<'_>) -> Result<()> {
        ensure!(self.operands.values().chain(self.core.values().flat_map(|weights|
            [&weights.payload, &weights.suh, &weights.svh])).all(|allocation| allocation.device.id == device.id
            && std::ptr::eq(allocation.device.library, device.library)), "Kimi layer weight owner differs from lane device");
        Ok(())
    }

    /// `charged_bytes` is the exact weight demand accepted by shared startup
    /// admission, not available CUDA bytes or an independent family budget.
    /// Uploads are synchronous; no queued work can outlive error cleanup here.
    pub(crate) fn upload(device: Device<'a>, packed: PackedLayer, layout: &LayerLayout,
        charged_bytes: usize) -> Result<Self> {
        layout.check(&packed)?;
        ensure!(layout.bytes()? == charged_bytes, "Kimi layer layout differs from admitted weight bytes");
        let operands = packed.operands.into_iter().map(|(name, bytes)| {
            ensure!(!bytes.is_empty(), "Kimi operand {name} is empty");
            let _scope = cuteafd_ffi::memory_ledger::scope("weights/kimi layer weights");
            let allocation = Allocation::new(device, bytes.len())?;
            device.run(|| device.library.copy_h2d(allocation.buffer, &bytes))?;
            Ok((name, allocation))
        }).collect::<Result<_>>()?;
        let core = packed.core.into_iter().map(|(name, loaded)|
            Ok((name, layout.core[name].upload(device, loaded)?))).collect::<Result<_>>()?;
        Ok(Self { operands, attention: packed.attention, core })
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self.operands.get(name).with_context(|| format!("Kimi layer has no resident operand {name}"))?.buffer)
    }

    pub(crate) fn core(&self,name:&str)->Result<&CoreWeights<'a>> {
        self.core.get(name).with_context(||format!("Kimi layer has no resident K6 projection {name}"))
    }
    pub(crate) fn is_core(&self,name:&str)->bool { self.core.contains_key(name) }

    pub(crate) fn kda(&self) -> Result<KimiKdaWeights> {
        ensure!(self.attention == AttentionShare::Split, "Kimi KDA weights are not TP2 split");
        Ok(KimiKdaWeights { input: self.buffer("kda.input")?, forget: self.buffer("kda.forget")?,
            convolution: self.buffer("kda.conv")?, a_log: self.buffer("kda.a_log")?,
            dt_bias: self.buffer("kda.dt_bias")?, norm: self.buffer("kda.norm")?, output: self.buffer("kda.output")? })
    }

    pub(crate) fn mla(&self) -> Result<KimiMlaWeights> {
        ensure!(self.attention.present(), "Kimi MLA weights absent on this nonowner");
        Ok(KimiMlaWeights { qkv_a: self.buffer("mla.qkv_a")?, q_a_norm: self.buffer("mla.q_norm")?,
            kv_a_norm: self.buffer("mla.kv_norm")?, q_b: self.buffer("mla.q_b")?, uk: self.buffer("mla.uk")?,
            uv: self.buffer("mla.uv")?, gate: self.buffer("mla.gate")?, output: self.buffer("mla.output")? })
    }
}

#[path = "records.rs"]
pub(crate) mod records;

#[cfg(test)]
#[path = "coordinator_golden_tests.rs"]
pub(crate) mod coordinator_golden_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::{TensorInfo, TensorRole};
    use cuteafd_loader::placement::FfnMode;

    fn cfg() -> KimiK3Config {
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        cfg.hidden = 4; cfg.heads = 4; cfg.q_lora_rank = 3; cfg.kv_lora_rank = 2;
        cfg.qk_nope_dim = 2; cfg.qk_rope_dim = 1; cfg.v_head_dim = 1;
        cfg.dense_intermediate = 8; cfg.latent = 2; cfg.experts = 4;
        cfg.moe_intermediate = 4; cfg.shared_experts = 2;
        cfg
    }

    fn tensor(name: &str) -> Result<LoadedTensor> {
        let (dtype, shape) = if name.ends_with("e_score_correction_bias") { (DType::F32, vec![4]) }
        else { (DType::Bf16, if name.ends_with("q_a_proj.weight") { vec![3, 4] }
            else if name.ends_with("q_a_layernorm.weight") { vec![3] }
            else if name.ends_with("kv_a_layernorm.weight") { vec![2] }
            else if name.ends_with("kv_a_proj_with_mqa.weight") { vec![3, 4] }
            else if name.ends_with("q_b_proj.weight") { vec![12, 3] }
            else if name.ends_with("kv_b_proj.weight") { vec![12, 2] }
            else if name.ends_with("g_proj.weight") || name.ends_with("o_proj.weight") { vec![4, 4] }
            else if name.ends_with("routed_expert_down_proj.weight") { vec![2, 4] }
            else if name.ends_with("routed_expert_up_proj.weight") { vec![4, 2] }
            else if name.ends_with("routed_expert_norm.weight") { vec![2] }
            else if name.ends_with("gate.weight") { vec![4, 4] }
            else if name.ends_with("down_proj.weight") { vec![4, 8] }
            else if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") { vec![8, 4] }
            else if name.ends_with("_res_proj.weight") { vec![1, 4] }
            else { vec![4] }) };
        let count: usize = shape.iter().product();
        let bytes = if dtype == DType::F32 { vec![0; count * 4] }
            else { std::iter::repeat_n(0x3f80u16, count).flat_map(u16::to_le_bytes).collect() };
        Ok(LoadedTensor { info: TensorInfo { name: name.into(), file: "test".into(), dtype, shape,
            byte_offset: 0, byte_length: bytes.len() as u64, role: TensorRole::Other, layer_id: None,
            expert_id: None, is_quantization_metadata: false }, source_path: "test".into(), bytes,
            elapsed_micros: 0, sha256: String::new() })
    }

    #[test]
    fn checkpoint_reader_uses_header_offsets_and_refuses_index_mismatch() -> Result<()> {
        use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("weights.safetensors"), [9, 8, 7, 0x80, 0x3f, 0, 0x40])?;
        let name = "language_model.model.norm.weight";
        let mut checkpoint = Checkpoint { snapshot: directory.path().into(), config: serde_json::json!({}),
            quantize_config: None, weight_map: BTreeMap::from([(name.into(), "weights.safetensors".into())]),
            tensors: vec![CheckpointTensor { shard: "weights.safetensors".into(),
                meta: cuteafd_loader::SafetensorsTensorMetadata { name: name.into(), dtype: DType::Bf16,
                    shape: vec![2], byte_offset: 3, byte_length: 4 } }], missing_shards: vec![], shard_bytes: 7 };
        let loaded = read_tensor(&checkpoint, name)?;
        assert_eq!(loaded.bytes, [0x80, 0x3f, 0, 0x40]);
        assert_eq!(loaded.info.shape, [2]);
        assert_eq!(loaded.info.byte_offset, 3);
        assert!(read_tensor(&checkpoint, "absent").is_err());
        checkpoint.weight_map.insert(name.into(), "another.safetensors".into());
        assert!(read_tensor(&checkpoint, name).is_err());
        checkpoint.weight_map.insert(name.into(), "weights.safetensors".into());
        checkpoint.tensors[0].meta.byte_length += 1;
        assert!(read_tensor(&checkpoint, name).is_err());
        Ok(())
    }

    #[test]
    fn global_native_extents_follow_entry_and_terminal_without_extra_copies() -> Result<()> {
        let mut cfg = cfg();
        cfg.vocab_size = 8;
        let pair = GlobalLayout::pair(&cfg)?;
        assert_eq!(pair[0].operands, BTreeMap::from([("embedding", 64)]));
        assert_eq!(pair[1].operands, BTreeMap::from([("head", 64), ("output.score", 16), ("output.norm", 8)]));
        let mut names = Vec::new();
        for (rank, layout) in pair.iter().enumerate() {
            let packed = layout.pack(&cfg, |name| {
                names.push(name.to_string());
                let mut value = tensor(name)?;
                if name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight") {
                    value.info.shape = vec![8, 4];
                    value.bytes = std::iter::repeat_n(0x3f80u16, 32).flat_map(u16::to_le_bytes).collect();
                    value.info.byte_length = value.bytes.len() as u64;
                }
                Ok(value)
            })?;
            layout.check(&packed)?;
            assert_eq!(layout.bytes()?, packed.operands.values().map(Vec::len).sum::<usize>());
            let demand = layout.demand()?;
            assert_eq!(demand.gpu, rank as u8);
            assert_eq!(demand.category, Category::Weights);
            assert_eq!(demand.basis, Basis::Exact);
            assert_eq!(demand.bytes, layout.bytes()? as u64);
            if rank == 1 {
                let scores = packed.operands["output.score"].chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>();
                assert_eq!(scores, vec![1.; 4]);
            }
            assert!(pair[1 - rank].check(&packed).is_err());
        }
        assert_eq!(names, ["language_model.model.embed_tokens.weight", "language_model.lm_head.weight",
            "language_model.model.output_attn_res_norm.weight", "language_model.model.output_attn_res_proj.weight",
            "language_model.model.norm.weight"]);
        for invalid in 0..3 {
            let mut broken = cfg.clone();
            match invalid { 0 => broken.tie_embeddings = true, 1 => broken.hidden = usize::MAX, _ => broken.vocab_size = 0 }
            assert!(GlobalLayout::pair(&broken).is_err());
        }
        Ok(())
    }

    #[test]
    fn global_packing_rejects_bad_native_operands_even_with_equal_total_bytes() -> Result<()> {
        let mut cfg = cfg();
        cfg.vocab_size = 1;
        let pair = GlobalLayout::pair(&cfg)?;
        for case in 0..3 {
            assert!(pair[0].pack(&cfg, |name| {
                let mut value = tensor(name)?;
                value.info.shape = vec![1, 4];
                match case { 0 => value.info.dtype = DType::F32, 1 => value.info.shape = vec![4, 1], _ => { value.bytes.pop(); } }
                Ok(value)
            }).is_err());
        }
        let mut packed = pair[1].pack(&cfg, |name| {
            let mut value = tensor(name)?;
            if name.ends_with("lm_head.weight") { value.info.shape = vec![1, 4]; }
            Ok(value)
        })?;
        packed.operands.get_mut("head").unwrap().pop();
        packed.operands.get_mut("output.norm").unwrap().push(0);
        assert_eq!(pair[1].bytes()?, packed.operands.values().map(Vec::len).sum::<usize>());
        assert!(pair[1].check(&packed).is_err());
        Ok(())
    }

    #[test]
    fn routed_frontends_follow_routed_ordinal_not_attention_or_dense_gaps() -> Result<()> {
        let mut cfg = cfg();
        cfg.layers = 7;
        cfg.attention = vec![KimiAttention::Mla; 7];
        cfg.dense = vec![true, false, true, false, false, true, false];
        for (layer, owner) in [(1, 0), (3, 1), (4, 0), (6, 1)] {
            assert_eq!(routed_owner(&cfg, layer)?, owner);
            for mode in [LayerMode::HeadSplit,
                LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
                LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }] {
                let layouts = LayerLayout::pair(&cfg, layer, mode)?;
                let packed = PackedLayer::pair(&cfg, layer, mode, tensor)?;
                for rank in 0..2 {
                    layouts[rank].check(&packed[rank])?;
                    for key in ["router.weight", "router.bias", "latent.down", "latent.norm", "latent.up"] {
                        assert_eq!(packed[rank].operands.contains_key(key), rank == owner);
                        assert_eq!(layouts[rank].operands.contains_key(key), rank == owner);
                    }
                }
            }
        }
        for layer in [0, 2, 5, 7, usize::MAX] { assert!(routed_owner(&cfg, layer).is_err()); }
        Ok(())
    }

    #[test]
    fn owned_mla_keeps_tp_ffn_and_only_one_canonical_moe_frontend() -> Result<()> {
        let cfg = cfg();
        for owner in 0..2 {
            let pair = PackedLayer::pair(&cfg, 3, LayerMode::Whole { gpu: owner, ffn: FfnMode::Split }, tensor)?;
            let full = &pair[usize::from(owner)];
            let absent = &pair[usize::from(1 - owner)];
            assert_eq!(full.attention, AttentionShare::Whole);
            assert_eq!(absent.attention, AttentionShare::Absent);
            assert!(!absent.operands.keys().any(|key| key.starts_with("mla.")));
            assert_eq!(full.operands["mla.q_b"].len(), 12 * 3 * 2);
            assert_eq!(full.operands["mla.uk"].len(), 4 * 2 * 2 * 2);
            for key in ["ffn.gate_up", "ffn.down", "attention.score", "ffn.norm"] {
                assert_eq!(full.operands[key], absent.operands[key]);
            }
            let frontend = routed_owner(&cfg, 3)?;
            for key in ["router.weight", "router.bias", "latent.down", "latent.norm", "latent.up"] {
                assert!(pair[frontend].operands.contains_key(key));
                assert!(!pair[1 - frontend].operands.contains_key(key));
            }
            assert_eq!(absent.operands["attention.score"].len(), 4 * 4);
            assert_eq!(absent.operands["ffn.gate_up"].len(), 2 * 4 * 4 * 2);
            assert_eq!(full.bytes()?, full.operands.values().map(Vec::len).sum::<usize>());
        }
        Ok(())
    }

    #[test]
    fn kda_native_f32_operands_and_column_output_use_shared_tp2_packer() -> Result<()> {
        let mut cfg = cfg();
        cfg.hidden = 128; cfg.kda_heads = 16; cfg.kda_head_dim = 128;
        let width = cfg.kda_heads * cfg.kda_head_dim;
        let pair = PackedLayer::pair(&cfg, 0, LayerMode::HeadSplit, |name| {
            let (dtype, shape) = if let Some(leaf) = name.split("self_attn.").nth(1) {
                match leaf {
                    "q_proj.weight" | "k_proj.weight" | "v_proj.weight" | "g_proj.weight" =>
                        (DType::Bf16, vec![width, cfg.hidden]),
                    "f_a_proj.weight" => (DType::Bf16, vec![128, cfg.hidden]),
                    "f_b_proj.weight" => (DType::Bf16, vec![width, 128]),
                    "b_proj.weight" => (DType::Bf16, vec![cfg.kda_heads, cfg.hidden]),
                    "q_conv1d.weight" | "k_conv1d.weight" | "v_conv1d.weight" =>
                        (DType::F32, vec![width, 1, 4]),
                    "A_log" => (DType::F32, vec![cfg.kda_heads]),
                    "dt_bias" => (DType::F32, vec![width]),
                    "o_norm.weight" => (DType::F32, vec![128]),
                    "o_proj.weight" => (DType::Bf16, vec![cfg.hidden, width]),
                    _ => anyhow::bail!("unexpected KDA tensor {name}"),
                }
            } else if name.ends_with("down_proj.weight") { (DType::Bf16, vec![cfg.hidden, 8]) }
            else if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") {
                (DType::Bf16, vec![8, cfg.hidden])
            } else if name.ends_with("_res_proj.weight") { (DType::Bf16, vec![1, cfg.hidden]) }
            else { (DType::Bf16, vec![cfg.hidden]) };
            let mut value = tensor("test.weight")?;
            value.info.name = name.into();
            value.info.dtype = dtype.clone();
            value.info.shape = shape.clone();
            value.bytes = vec![0; shape.iter().product::<usize>() * if dtype == DType::F32 { 4 } else { 2 }];
            value.info.byte_length = value.bytes.len() as u64;
            Ok(value)
        })?;
        for rank in pair {
            assert_eq!(rank.attention, AttentionShare::Split);
            assert_eq!(rank.operands["kda.norm"].len(), 128 * 4);
            assert_eq!(rank.operands["kda.a_log"].len(), 8 * 4);
            assert_eq!(rank.operands["kda.dt_bias"].len(), width / 2 * 4);
            assert_eq!(rank.operands["kda.output"].len(), cfg.hidden * width / 2 * 2);
            assert!(!rank.operands.keys().any(|key| key.starts_with("latent.") || key.starts_with("router.")));
        }
        Ok(())
    }

    #[test]
    fn admitted_extents_match_packing_and_reject_equal_total_operand_swaps() -> Result<()> {
        let cfg = cfg();
        let mode = LayerMode::Whole { gpu: 1, ffn: FfnMode::Split };
        let layouts = LayerLayout::pair(&cfg, 3, mode)?;
        let mut packed = PackedLayer::pair(&cfg, 3, mode, tensor)?;
        for (layout, packed) in layouts.iter().zip(&packed) {
            assert_eq!(layout.bytes()?, packed.bytes()?);
            layout.check(packed)?;
        }
        let before = packed[0].bytes()?;
        packed[0].operands.get_mut("ffn.gate_up").unwrap().pop();
        packed[0].operands.get_mut("ffn.down").unwrap().push(0);
        assert_eq!(before, packed[0].bytes()?);
        assert!(layouts[0].check(&packed[0]).is_err());
        for change in 0..3 {
            let mut invalid = cfg.clone();
            match change {
                0 => invalid.hidden = usize::MAX,
                1 => invalid.shared_experts = usize::MAX,
                _ => invalid.moe_intermediate = 3,
            }
            if change == 2 { invalid.shared_experts = 1; }
            assert!(PackedLayer::pair(&invalid, 3, mode, |_| unreachable!()).is_err());
        }
        Ok(())
    }

    #[test]
    fn explicit_head_split_and_invalid_source_or_placement_fail_closed() -> Result<()> {
        let cfg = cfg();
        let pair = PackedLayer::pair(&cfg, 3, LayerMode::HeadSplit, tensor)?;
        assert!(pair.iter().all(|rank| rank.attention == AttentionShare::Split));
        assert_eq!(pair[0].operands["mla.q_b"].len(), 6 * 3 * 2);
        assert!(PackedLayer::pair(&cfg, 3, LayerMode::ContextSplit, |_| unreachable!()).is_err());
        assert!(PackedLayer::pair(&cfg, 3, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
            |_| unreachable!()).is_err());
        assert!(PackedLayer::pair(&cfg, 0, LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
            |_| unreachable!()).is_err());
        assert!(PackedLayer::pair(&cfg, cfg.layers, LayerMode::HeadSplit, |_| unreachable!()).is_err());
        assert!(PackedLayer::pair(&cfg, 3, LayerMode::HeadSplit, |name| {
            let mut value = tensor(name)?;
            if name.ends_with("e_score_correction_bias") { value.bytes.pop(); }
            Ok(value)
        }).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod core_owner_tests {
    use super::*;
    use cuteafd_loader::{families::kimi_k3::core::{CoreProjection,CoreSplit},
        formats::{dense_trellis::TrellisSide,exl3_storage::Exl3Codebook}};
    fn fixture_config() -> KimiK3Config {
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        cfg.hidden = 256;
        cfg.heads = 4;
        cfg.kda_heads = 16;
        cfg.kda_head_dim = 128;
        cfg.q_lora_rank = 128;
        cfg.kv_lora_rank = 64;
        cfg.qk_nope_dim = 64;
        cfg.qk_rope_dim = 64;
        cfg.v_head_dim = 128;
        cfg.latent = 128;
        cfg.experts = 4;
        cfg.moe_intermediate = 128;
        cfg.shared_experts = 2;
        cfg.dense_intermediate = 256;
        cfg
    }

    fn inventory(layer: usize, leaves: &[(&str, [usize; 2])]) -> CoreInventory {
        CoreInventory { projections: leaves.iter().map(|(leaf, shape)| {
            let stem = format!("language_model.model.layers.{layer}.{leaf}");
            let projection = CoreProjection { stem: stem.clone(), bits: 6, codebook: Exl3Codebook::Mul1,
                shape: *shape, stored_outputs: shape[0], split: cuteafd_loader::families::kimi_k3::core::split(&stem).unwrap(),
                source_bytes: 0 };
            (stem, projection)
        }).collect() }
    }

    #[test]
    fn mixed_kda_fallback_preserves_fused_attention_and_native_shared_charges() -> Result<()> {
        let cfg = fixture_config();
        let core = inventory(1, &[("block_sparse_moe.shared_experts.gate_proj", [256, 256]),
            ("block_sparse_moe.shared_experts.up_proj", [256, 256]),
            ("block_sparse_moe.shared_experts.down_proj", [256, 256])]);
        let original = LayerLayout::pair(&cfg, 1, LayerMode::HeadSplit)?;
        let mixed = LayerLayout::pair_with_core(&cfg, 1, LayerMode::HeadSplit, &core)?;
        for rank in 0..2 {
            assert_eq!(mixed[rank].attention_bytes()?, original[rank].attention_bytes()?);
            for name in ["kda.input", "kda.output", "kda.forget", "kda.conv", "kda.norm"] {
                assert_eq!(mixed[rank].operands[name], original[rank].operands[name]);
            }
            assert!(!mixed[rank].operands.contains_key("ffn.gate_up"));
            assert_eq!(mixed[rank].core.len(), 3);
            let removed = original[rank].operands["ffn.gate_up"] + original[rank].operands["ffn.down"];
            let native = mixed[rank].core.values().map(|layout| layout.bytes).sum::<usize>();
            assert_eq!(mixed[rank].bytes()?, original[rank].bytes()? - removed + native);
            assert_eq!(mixed[rank].core["ffn.down"].side, TrellisSide::Input);
        }
        assert_eq!(LayerLayout::pair_with_core(&cfg, 1, LayerMode::HeadSplit, &CoreInventory::default())?, original);
        Ok(())
    }

    #[test]
    fn mla_native_padding_and_ownership_leave_absorbed_kv_b_unchanged() -> Result<()> {
        use cuteafd_loader::placement::FfnMode;
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let mut core = inventory(3, &[("self_attn.kv_a_proj_with_mqa", [576, 7168]),
            ("self_attn.o_proj", [7168, 12288])]);
        core.projections.get_mut("language_model.model.layers.3.self_attn.kv_a_proj_with_mqa").unwrap().stored_outputs = 640;
        for mode in [LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
            LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }] {
            let baseline = LayerLayout::pair(&cfg, 3, mode)?;
            let mixed = LayerLayout::pair_with_core(&cfg, 3, mode, &core)?;
            for rank in 0..2 {
                if !mixed[rank].attention.present() { assert!(mixed[rank].core.is_empty()); continue; }
                assert_eq!(mixed[rank].core["mla.kv_a"].shape, [640, 7168]);
                assert_eq!(mixed[rank].operands["mla.uk"], baseline[rank].operands["mla.uk"]);
                assert_eq!(mixed[rank].operands["mla.uv"], baseline[rank].operands["mla.uv"]);
                assert!(!mixed[rank].operands.contains_key("mla.qkv_a"));
                assert_eq!(mixed[rank].core["mla.output"].side,
                    if mode == LayerMode::HeadSplit { TrellisSide::Input } else { TrellisSide::Output });
            }
        }
        Ok(())
    }

    fn mixed_tensor(cfg: &KimiK3Config, inventory: &CoreInventory, name: &str) -> Result<LoadedTensor> {
        use cuteafd_core::{TensorInfo, TensorRole};
        let (stem, suffix) = name.rsplit_once('.').context("fixture tensor suffix")?;
        let (dtype, shape, bytes) = if let Some(projection) = inventory.get_weight(stem) {
            let [n, k] = [projection.stored_outputs, projection.shape[1]];
            match suffix {
                "trellis" => (DType::I16, vec![k / 16, n / 16, 96], vec![0; n * k * 6 / 8]),
                "suh" => (DType::F16, vec![k], vec![0; k * 2]),
                "svh" => (DType::F16, vec![n], vec![0; n * 2]),
                value if value == projection.codebook.marker() =>
                    (DType::I32, vec![], projection.codebook.multiplier().to_le_bytes().to_vec()),
                _ => anyhow::bail!("compressed fixture read as BF16: {name}"),
            }
        } else {
            let leaf = name.split("self_attn.").nth(1).unwrap_or("");
            let d = cfg.kda_heads * cfg.kda_head_dim;
            let (dtype, shape) = match leaf {
                "q_proj.weight" | "k_proj.weight" | "v_proj.weight" => (DType::Bf16, vec![d, cfg.hidden]),
                "f_a_proj.weight" => (DType::Bf16, vec![cfg.kda_head_dim, cfg.hidden]),
                "f_b_proj.weight" => (DType::Bf16, vec![d, cfg.kda_head_dim]),
                "b_proj.weight" => (DType::Bf16, vec![cfg.kda_heads, cfg.hidden]),
                "q_conv1d.weight" | "k_conv1d.weight" | "v_conv1d.weight" => (DType::F32, vec![d, 1, cfg.kda_conv]),
                "A_log" => (DType::F32, vec![cfg.kda_heads]),
                "dt_bias" => (DType::F32, vec![d]),
                "o_norm.weight" => (DType::F32, vec![cfg.kda_head_dim]),
                "q_a_proj.weight" => (DType::Bf16, vec![cfg.q_lora_rank, cfg.hidden]),
                "kv_a_proj_with_mqa.weight" => (DType::Bf16, vec![cfg.kv_lora_rank + cfg.qk_rope_dim, cfg.hidden]),
                "q_a_layernorm.weight" => (DType::Bf16, vec![cfg.q_lora_rank]),
                "kv_a_layernorm.weight" => (DType::Bf16, vec![cfg.kv_lora_rank]),
                "q_b_proj.weight" => (DType::Bf16, vec![cfg.heads * (cfg.qk_nope_dim + cfg.qk_rope_dim), cfg.q_lora_rank]),
                "kv_b_proj.weight" => (DType::Bf16, vec![cfg.heads * (cfg.qk_nope_dim + cfg.v_head_dim), cfg.kv_lora_rank]),
                "g_proj.weight" => (DType::Bf16, vec![if name.contains("layers.3.") {cfg.heads * cfg.v_head_dim} else {d}, cfg.hidden]),
                "o_proj.weight" => (DType::Bf16, vec![cfg.hidden, if name.contains("layers.3.") {cfg.heads * cfg.v_head_dim} else {d}]),
                _ if name.ends_with("routed_expert_down_proj.weight") => (DType::Bf16, vec![cfg.latent, cfg.hidden]),
                _ if name.ends_with("routed_expert_up_proj.weight") => (DType::Bf16, vec![cfg.hidden, cfg.latent]),
                _ if name.ends_with("routed_expert_norm.weight") => (DType::Bf16, vec![cfg.latent]),
                _ if name.ends_with("gate.weight") => (DType::Bf16, vec![cfg.experts, cfg.hidden]),
                _ if name.ends_with("e_score_correction_bias") => (DType::F32, vec![cfg.experts]),
                _ if name.ends_with("_res_proj.weight") => (DType::Bf16, vec![1, cfg.hidden]),
                _ if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") =>
                    (DType::Bf16, vec![cfg.moe_intermediate * cfg.shared_experts, cfg.hidden]),
                _ if name.ends_with("down_proj.weight") => (DType::Bf16, vec![cfg.hidden, cfg.moe_intermediate * cfg.shared_experts]),
                _ => (DType::Bf16, vec![cfg.hidden]),
            };
            let bytes = vec![0; shape.iter().product::<usize>() * if dtype == DType::F32 {4} else {2}];
            (dtype, shape, bytes)
        };
        Ok(LoadedTensor { info: TensorInfo { name: name.into(), file: "test".into(), dtype, shape,
            byte_offset: 0, byte_length: bytes.len() as u64, role: TensorRole::Other, layer_id: None,
            expert_id: None, is_quantization_metadata: false }, source_path: "test".into(), bytes,
            elapsed_micros: 0, sha256: String::new() })
    }

    #[test]
    fn mixed_layer_pack_matches_each_admitted_operand_without_bf16_shadows() -> Result<()> {
        use cuteafd_loader::placement::FfnMode;
        let cfg = fixture_config();
        for layer in [1, 3] {
            let mut leaves = vec![("block_sparse_moe.shared_experts.gate_proj", [256, 256]),
                ("block_sparse_moe.shared_experts.down_proj", [256, 256])];
            if layer == 1 {
                leaves.push(("self_attn.q_proj", [2048, 256]));
            } else {
                leaves.push(("self_attn.kv_a_proj_with_mqa", [128, 256]));
                leaves.push(("self_attn.o_proj", [256, 512]));
            }
            let core = inventory(layer, &leaves);
            let modes = if layer == 1 { vec![LayerMode::HeadSplit] } else { vec![LayerMode::HeadSplit,
                LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }] };
            for mode in modes {
                let layouts = LayerLayout::pair_with_core(&cfg, layer, mode, &core)?;
                let packed = PackedLayer::pair_with_core(&cfg, layer, mode, &core, |name| mixed_tensor(&cfg, &core, name))?;
                for rank in 0..2 {
                    layouts[rank].check(&packed[rank])?;
                    assert_eq!(layouts[rank].bytes()?, packed[rank].bytes()?);
                    assert!(packed[rank].core.contains_key("ffn.gate"));
                    assert!(packed[rank].operands.contains_key("ffn.up"));
                    assert!(!packed[rank].operands.contains_key("ffn.gate"));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn native_latent_pair_is_whole_on_routed_owner_with_norm_and_router_bf16() -> Result<()> {
        let cfg=fixture_config();
        for layer in [1,2,3] {
            let core=inventory(layer,&[("block_sparse_moe.routed_expert_down_proj",[cfg.latent,cfg.hidden]),
                ("block_sparse_moe.routed_expert_up_proj",[cfg.hidden,cfg.latent])]);
            core.validate_runtime_groups(&cfg)?;
            let layouts=LayerLayout::pair_with_core(&cfg,layer,LayerMode::HeadSplit,&core)?;
            let packed=PackedLayer::pair_with_core(&cfg,layer,LayerMode::HeadSplit,&core,
                |name|mixed_tensor(&cfg,&core,name))?;
            let owner=routed_owner(&cfg,layer)?;
            for rank in 0..2 {
                layouts[rank].check(&packed[rank])?;
                assert_eq!(layouts[rank].bytes()?,packed[rank].bytes()?);
                assert_eq!(packed[rank].core.len(),if rank==owner {2}else{0});
                for name in ["latent.down","latent.up"] {assert!(!packed[rank].operands.contains_key(name));}
            }
            for name in ["latent.down","latent.up"] {
                let projection=&layouts[owner].core[name];
                assert_eq!((projection.rank,projection.ranks,projection.side),(0,1,TrellisSide::Output));
            }
            for name in ["latent.norm","router.weight","router.bias"] {
                assert!(packed[owner].operands.contains_key(name));
                assert!(!packed[1-owner].operands.contains_key(name));
            }
        }
        Ok(())
    }

    #[test]
    fn native_head_is_terminal_only_and_packs_no_bf16_shadow() -> Result<()> {
        let mut cfg=fixture_config();cfg.vocab_size=128;
        let stem="language_model.lm_head".to_owned();
        let core=CoreInventory{projections:BTreeMap::from([(stem.clone(),CoreProjection{
            stem,bits:6,codebook:Exl3Codebook::Mul1,shape:[cfg.vocab_size,cfg.hidden],
            stored_outputs:cfg.vocab_size,split:CoreSplit::Whole,source_bytes:0})])};
        let baseline=GlobalLayout::pair(&cfg)?;
        let layouts=GlobalLayout::pair_with_core(&cfg,&core)?;
        assert_eq!(layouts[0],baseline[0]);
        assert!(!layouts[1].operands.contains_key("head"));
        let head=layouts[1].core_head.as_ref().unwrap();
        assert_eq!((head.rank,head.ranks,head.side),(0,1,TrellisSide::Output));
        assert_eq!(layouts[1].bytes()?,baseline[1].bytes()?-cfg.vocab_size*cfg.hidden*2+head.bytes);
        let mut names=Vec::new();
        let mut packed=layouts[1].pack(&cfg,|name|{
            names.push(name.to_owned());mixed_tensor(&cfg,&core,name)
        })?;
        assert!(!names.iter().any(|name|name=="language_model.lm_head.weight"));
        assert_eq!(names.iter().filter(|name|name.starts_with("language_model.lm_head.")).count(),4);
        assert_eq!(packed.operands.len(),2);
        assert!(packed.core_head.is_some());
        packed.core_head.as_mut().unwrap().shard.svh.pop();
        assert!(layouts[1].check(&packed).is_err());
        Ok(())
    }

    #[test]
    fn native_member_checks_reject_compensating_vector_extent_errors() -> Result<()> {
        use cuteafd_loader::formats::dense_trellis::{DenseTrellisCheckpoint, LoadedDenseTrellis};
        let projection = inventory(1, &[("self_attn.q_proj", [256, 256])]);
        let p = projection.projections.values().next().unwrap();
        let layout = CoreWeightLayout::new(p, TrellisSide::Output, 0, 2)?;
        let payload = vec![0; 256 * 256 * 6 / 8];
        let vector = vec![0; 256 * 2];
        let source = DenseTrellisCheckpoint { shape: [256, 256], bits: 6, payload: &payload, suh: &vector, svh: &vector };
        let mut loaded = LoadedDenseTrellis { codebook: p.codebook, shard: source.native_shard(TrellisSide::Output, 0, 2)? };
        layout.check(&loaded)?;
        loaded.shard.suh.pop();
        loaded.shard.svh.push(0);
        assert_eq!(loaded.shard.payload.len() + loaded.shard.suh.len() + loaded.shard.svh.len(), layout.bytes);
        assert!(layout.check(&loaded).is_err());
        Ok(())
    }

    #[test]
    fn mixed_projection_packing_never_reads_a_bf16_shadow() -> Result<()> {
        use cuteafd_core::{TensorInfo, TensorRole};
        let cfg = fixture_config();
        let core = inventory(1, &[("block_sparse_moe.shared_experts.gate_proj", [256, 256])]);
        let layouts = LayerLayout::pair_with_core(&cfg, 1, LayerMode::HeadSplit, &core)?;
        let mut packed = std::array::from_fn(|rank| PackedLayer { operands: BTreeMap::new(),
            core: BTreeMap::new(), attention: layouts[rank].attention });
        let stem = "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj";
        let mut names = Vec::new();
        pack_projection_pair(&layouts, &mut packed, "ffn.gate", stem, [256, 256], TrellisSide::Output,
            &[0, 1], &mut |name: &str| {
                names.push(name.to_owned());
                let (dtype, shape, bytes) = match name.rsplit_once('.').unwrap().1 {
                    "trellis" => (DType::I16, vec![16, 16, 96], vec![0; 256 * 256 * 6 / 8]),
                    "suh" | "svh" => (DType::F16, vec![256], vec![0; 512]),
                    "mul1" => (DType::I32, vec![], Exl3Codebook::Mul1.multiplier().to_le_bytes().to_vec()),
                    _ => anyhow::bail!("unexpected BF16 shadow read {name}"),
                };
                Ok(LoadedTensor { info: TensorInfo { name: name.into(), file: "test".into(), dtype, shape,
                    byte_offset: 0, byte_length: bytes.len() as u64, role: TensorRole::Other, layer_id: None,
                    expert_id: None, is_quantization_metadata: false }, source_path: "test".into(), bytes,
                    elapsed_micros: 0, sha256: String::new() })
            })?;
        assert_eq!(names.len(), 4);
        assert!(names.iter().all(|name| !name.ends_with(".weight")));
        for rank in 0..2 { layouts[rank].core["ffn.gate"].check(&packed[rank].core["ffn.gate"])?; }
        Ok(())
    }

    #[test]
    fn native_owner_charge_matches_planner_and_storage_side() {
        for (split,side) in [(CoreSplit::Output,TrellisSide::Output),(CoreSplit::Input,TrellisSide::Input)] {
            let p=CoreProjection{stem:"projection".into(),bits:6,codebook:Exl3Codebook::Mul1,
                shape:[12288,7168],stored_outputs:12288,split,source_bytes:0};
            for rank in 0..2 {
                let layout=CoreWeightLayout::new(&p,side,rank,2).unwrap();
                assert_eq!(layout.bytes as u64,p.resident_bytes(2).unwrap());
                assert_eq!(layout.shape,if side==TrellisSide::Output {[6144,7168]}else{[12288,3584]});
            }
            assert!(CoreWeightLayout::new(&p,side,2,2).is_err());
            assert!(CoreWeightLayout::new(&p,side,0,3).is_err());
        }
        let p=CoreProjection{stem:"kv_a".into(),bits:6,codebook:Exl3Codebook::Mcg,
            shape:[576,7168],stored_outputs:640,split:CoreSplit::Whole,source_bytes:0};
        let layout=CoreWeightLayout::new(&p,TrellisSide::Output,0,1).unwrap();
        assert_eq!(layout.shape,[640,7168]);assert_eq!(layout.bytes as u64,p.resident_bytes(1).unwrap());
    }
}
