//! Packed device weight extents shared by checkpoint admission and runtime owners.
use crate::formats::trellis_core::CoreWeightLayout;
use super::{config::{KimiAttention, KimiK3Config}, core::CoreInventory, kda};
use crate::{placement::{Demand, LayerMode, ResidentWeights}, formats::dense_trellis::TrellisSide};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightAttention { Absent, Whole, Split }
impl WeightAttention {
    fn bind(mode: LayerMode, ranks: usize) -> Result<Vec<Self>> {
        ensure!(ranks == 2, "Kimi weight layout requires TP2");
        match mode {
            LayerMode::HeadSplit | LayerMode::ContextSplit => Ok(vec![Self::Split; ranks]),
            LayerMode::Whole { gpu, .. } if usize::from(gpu) < ranks => {
                let mut shares = vec![Self::Absent; ranks];
                shares[usize::from(gpu)] = Self::Whole;
                Ok(shares)
            }
            _ => anyhow::bail!("Kimi attention weight owner outside TP2"),
        }
    }
    pub fn present(self) -> bool { self != Self::Absent }
}
pub fn routed_owner(cfg: &KimiK3Config, layer: usize) -> Result<usize> {
    ensure!(cfg.dense.get(layer) == Some(&false), "Kimi front end requires a routed layer");
    Ok(cfg.dense[..layer].iter().filter(|&&dense| !dense).count() % 2)
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalLayout {
    pub rank: usize, pub hidden: usize, pub vocab: usize,
    pub operands: BTreeMap<&'static str, usize>, pub core_head: Option<CoreWeightLayout>,
}
impl GlobalLayout {
    pub fn pair(cfg: &KimiK3Config) -> Result<[Self; 2]> {
        ensure!(!cfg.tie_embeddings, "Kimi global weights require the checkpoint's untied head");
        let matrix = extent(&[cfg.vocab_size, cfg.hidden, 2])?;
        Ok([Self { rank: 0, hidden: cfg.hidden, vocab: cfg.vocab_size,
                core_head: None, operands: BTreeMap::from([("embedding", matrix)]) },
            Self { rank: 1, hidden: cfg.hidden, vocab: cfg.vocab_size,
                core_head: None, operands: BTreeMap::from([("head", matrix), ("output.score", extent(&[cfg.hidden, 4])?),
                    ("output.norm", extent(&[cfg.hidden, 2])?)]) }])
    }

    pub fn pair_with_core(cfg: &KimiK3Config, inventory: &CoreInventory) -> Result<[Self; 2]> {
        let mut pair=Self::pair(cfg)?;
        if let Some(head)=inventory.get_weight("language_model.lm_head") {
            ensure!(head.shape==[cfg.vocab_size,cfg.hidden],"Kimi K6 head inventory geometry");
            pair[1].operands.remove("head");
            pair[1].core_head=Some(CoreWeightLayout::new(head,TrellisSide::Output,0,1)?);
        }
        Ok(pair)
    }

    pub fn bytes(&self) -> Result<usize> {
        self.operands.values().copied().chain(self.core_head.iter().map(|head|head.bytes))
            .try_fold(0usize, |sum, bytes| sum.checked_add(bytes)
            .context("Kimi global weight extent overflow"))
    }

    pub fn demand(&self) -> Result<Demand> {
        let bytes = self.bytes()? as u64;
        Ok(Demand::new(self.rank as u8, Category::Weights, "Kimi global weights", bytes, Basis::Exact)
            .with_resident_weights(ResidentWeights(bytes)))
    }

}
#[derive(Debug, PartialEq, Eq)]
pub struct LayerLayout {
    pub operands: BTreeMap<&'static str, usize>,
    pub attention: WeightAttention,
    pub core: BTreeMap<&'static str, CoreWeightLayout>,
}
impl LayerLayout {
    pub fn pair(cfg: &KimiK3Config, layer: usize, mode: LayerMode) -> Result<[Self; 2]> {
        let kind = *cfg.attention.get(layer).context("Kimi attention layer out of range")?;
        ensure!(cfg.layers == cfg.attention.len() && cfg.dense.len() == cfg.layers,
            "Kimi dense/attention inventories differ");
        ensure!(mode != LayerMode::ContextSplit, "Kimi context attention has no query/LSE exchange ABI");
        ensure!(!matches!(mode, LayerMode::Whole { ffn: crate::placement::FfnMode::Owner, .. }),
            "Kimi FFNs require TP2 intermediate slices");
        ensure!(kind != KimiAttention::Kda || mode == LayerMode::HeadSplit,
            "Kimi KDA requires TP2 head-split placement");
        let shares = WeightAttention::bind(mode, 2)?;
        let intermediate = if cfg.dense[layer] { cfg.dense_intermediate }
            else { extent(&[cfg.moe_intermediate, cfg.shared_experts])? };
        ensure!(intermediate > 0 && intermediate % 2 == 0, "Kimi FFN intermediate does not divide TP2");
        let local = intermediate / 2;
        let sum = |a: usize, b: usize| a.checked_add(b).context("Kimi weight extent overflow");
        let mut pair = std::array::from_fn(|rank| Self { operands: BTreeMap::new(), attention: shares[rank], core: BTreeMap::new() });
        for (index, rank) in pair.iter_mut().enumerate() {
            for name in ["attention.score", "ffn.score"] { rank.insert(name, &[cfg.hidden, 4])?; }
            for name in ["attention.norm", "ffn.norm"] { rank.insert(name, &[cfg.hidden, 2])?; }
            rank.insert("ffn.gate_up", &[2, local, cfg.hidden, 2])?;
            rank.insert("ffn.down", &[cfg.hidden, local, 2])?;
            if !cfg.dense[layer] && index == routed_owner(cfg, layer)? {
                for name in ["latent.down", "latent.up"] { rank.insert(name, &[cfg.latent, cfg.hidden, 2])?; }
                rank.insert("latent.norm", &[cfg.latent, 2])?;
                rank.insert("router.weight", &[cfg.experts, cfg.hidden, 2])?;
                rank.insert("router.bias", &[cfg.experts, 4])?;
            }
            match kind {
                KimiAttention::Kda => {
                    kda::geometry(cfg).validate(2)?;
                    let heads = cfg.kda_heads / 2;
                    let width = extent(&[heads, cfg.kda_head_dim])?;
                    let input = sum(sum(extent(&[4, width])?, cfg.kda_head_dim)?, heads)?;
                    rank.insert("kda.input", &[input, cfg.hidden, 2])?;
                    rank.insert("kda.forget", &[width, cfg.kda_head_dim, 2])?;
                    rank.insert("kda.conv", &[3, width, cfg.kda_conv, 4])?;
                    rank.insert("kda.a_log", &[heads, 4])?;
                    rank.insert("kda.dt_bias", &[width, 4])?;
                    rank.insert("kda.norm", &[cfg.kda_head_dim, 4])?;
                    rank.insert("kda.output", &[cfg.hidden, width, 2])?;
                }
                KimiAttention::Mla if rank.attention.present() => {
                    ensure!(cfg.heads > 0 && cfg.heads % 2 == 0, "Kimi MLA heads do not divide TP2");
                    let heads = if rank.attention == WeightAttention::Whole { cfg.heads } else { cfg.heads / 2 };
                    let input = sum(cfg.q_lora_rank, sum(cfg.kv_lora_rank, cfg.qk_rope_dim)?)?;
                    rank.insert("mla.qkv_a", &[input, cfg.hidden, 2])?;
                    rank.insert("mla.q_norm", &[cfg.q_lora_rank, 2])?;
                    rank.insert("mla.kv_norm", &[cfg.kv_lora_rank, 2])?;
                    rank.insert("mla.q_b", &[heads, sum(cfg.qk_nope_dim, cfg.qk_rope_dim)?, cfg.q_lora_rank, 2])?;
                    rank.insert("mla.uk", &[heads, cfg.kv_lora_rank, cfg.qk_nope_dim, 2])?;
                    rank.insert("mla.uv", &[heads, cfg.v_head_dim, cfg.kv_lora_rank, 2])?;
                    rank.insert("mla.gate", &[heads, cfg.v_head_dim, cfg.hidden, 2])?;
                    rank.insert("mla.output", &[cfg.hidden, heads, cfg.v_head_dim, 2])?;
                }
                KimiAttention::Mla => {}
            }
        }
        Ok(pair)
    }

    /// The BF16-only layout remains exactly the original fused layout. A mixed
    /// group stores individual BF16 matrices beside its native K6 projections;
    /// it never allocates a dequantized shadow of a compressed source tensor.
    pub fn pair_with_core(cfg: &KimiK3Config, layer: usize, mode: LayerMode,
        inventory: &CoreInventory) -> Result<[Self; 2]> {
        let mut pair = Self::pair(cfg, layer, mode)?;
        let root = format!("language_model.model.layers.{layer}");
        let shared = format!("{root}.block_sparse_moe.shared_experts");
        let attn = format!("{root}.self_attn");
        let native = |stem: &str| inventory.get_weight(stem).is_some();
        let ffn_split = !cfg.dense[layer] && ["gate_proj", "up_proj", "down_proj"].iter()
            .any(|leaf| native(&format!("{shared}.{leaf}")));
        let attention_split = ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "q_a_proj",
            "q_b_proj", "kv_a_proj_with_mqa"].iter().any(|leaf| native(&format!("{attn}.{leaf}")));
        let attention_ranks = pair.iter().filter(|layout| layout.attention.present()).count();
        for (rank, layout) in pair.iter_mut().enumerate() {
            if ffn_split {
                layout.operands.remove("ffn.gate_up");
                layout.operands.remove("ffn.down");
                let width = cfg.moe_intermediate * cfg.shared_experts;
                for (name, leaf, shape, side) in [
                    ("ffn.gate", "gate_proj", [width, cfg.hidden], TrellisSide::Output),
                    ("ffn.up", "up_proj", [width, cfg.hidden], TrellisSide::Output),
                    ("ffn.down", "down_proj", [cfg.hidden, width], TrellisSide::Input),
                ] { layout.projection(name, &format!("{shared}.{leaf}"), shape, side, rank, 2, inventory)?; }
            }
            if !cfg.dense[layer] && rank==routed_owner(cfg,layer)? {
                let root=format!("{root}.block_sparse_moe");
                for (name,leaf,shape) in [("latent.down","routed_expert_down_proj",[cfg.latent,cfg.hidden]),
                    ("latent.up","routed_expert_up_proj",[cfg.hidden,cfg.latent])] {
                    if native(&format!("{root}.{leaf}")) {
                        layout.operands.remove(name);
                        layout.projection(name,&format!("{root}.{leaf}"),shape,TrellisSide::Output,0,1,inventory)?;
                    }
                }
            }
            if !attention_split || !layout.attention.present() { continue; }
            let shard = if attention_ranks == 1 { 0 } else { rank };
            match cfg.attention[layer] {
                KimiAttention::Kda => {
                    layout.operands.remove("kda.input");
                    layout.operands.remove("kda.output");
                    layout.insert("kda.fab", &[cfg.kda_head_dim + cfg.kda_heads / 2, cfg.hidden, 2])?;
                    let width = cfg.kda_heads * cfg.kda_head_dim;
                    for (name, leaf) in [("kda.q", "q_proj"), ("kda.k", "k_proj"),
                        ("kda.v", "v_proj"), ("kda.gate", "g_proj")] {
                        layout.projection(name, &format!("{attn}.{leaf}"), [width, cfg.hidden],
                            TrellisSide::Output, shard, 2, inventory)?;
                    }
                    layout.projection("kda.output", &format!("{attn}.o_proj"), [cfg.hidden, width],
                        TrellisSide::Input, shard, 2, inventory)?;
                }
                KimiAttention::Mla => {
                    for name in ["mla.qkv_a", "mla.q_b", "mla.gate", "mla.output"] { layout.operands.remove(name); }
                    for (name, leaf, shape, side, count, index) in [
                        ("mla.q_a", "q_a_proj", [cfg.q_lora_rank, cfg.hidden], TrellisSide::Output, 1, 0),
                        ("mla.kv_a", "kv_a_proj_with_mqa", [cfg.kv_lora_rank + cfg.qk_rope_dim, cfg.hidden], TrellisSide::Output, 1, 0),
                        ("mla.q_b", "q_b_proj", [cfg.heads * (cfg.qk_nope_dim + cfg.qk_rope_dim), cfg.q_lora_rank], TrellisSide::Output, attention_ranks, shard),
                        ("mla.gate", "g_proj", [cfg.heads * cfg.v_head_dim, cfg.hidden], TrellisSide::Output, attention_ranks, shard),
                        ("mla.output", "o_proj", [cfg.hidden, cfg.heads * cfg.v_head_dim], TrellisSide::Input, attention_ranks, shard),
                    ] { layout.projection(name, &format!("{attn}.{leaf}"), shape, side, index, count, inventory)?; }
                }
            }
        }
        Ok(pair)
    }

    pub fn core_layouts(&self) -> impl Iterator<Item=&CoreWeightLayout> { self.core.values() }

    fn projection(&mut self, name: &'static str, stem: &str, shape: [usize; 2], side: TrellisSide,
        rank: usize, ranks: usize, inventory: &CoreInventory) -> Result<()> {
        if let Some(projection) = inventory.get_weight(stem) {
            ensure!(projection.shape == shape, "Kimi {stem} core inventory differs from layer geometry");
            let storage = if ranks == 1 { TrellisSide::Output } else { side };
            self.core.insert(name, CoreWeightLayout::new(projection, storage, rank, ranks)?);
        } else {
            let [mut n, mut k] = shape;
            match side { TrellisSide::Output => n /= ranks, TrellisSide::Input => k /= ranks }
            self.insert(name, &[n, k, 2])?;
        }
        Ok(())
    }

    fn insert(&mut self, name: &'static str, dimensions: &[usize]) -> Result<()> {
        ensure!(self.operands.insert(name, extent(dimensions)?).is_none(), "duplicate Kimi operand {name}");
        Ok(())
    }

    pub fn bytes(&self) -> Result<usize> {
        self.operands.values().copied().chain(self.core.values().map(|layout| layout.bytes))
            .try_fold(0usize, |n, size| n.checked_add(size).context("Kimi layer admission extent overflow"))
    }

    pub fn attention_bytes(&self) -> Result<usize> {
        self.operands.iter().filter(|(name, _)| name.starts_with("mla.") || name.starts_with("kda."))
            .map(|(_, bytes)| *bytes).chain(self.core.iter().filter(|(name, _)|
                name.starts_with("mla.") || name.starts_with("kda.")).map(|(_, layout)| layout.bytes))
            .try_fold(0usize, |n, bytes| n.checked_add(bytes).context("Kimi attention extent overflow"))
    }

}
fn extent(dimensions: &[usize]) -> Result<usize> {
    ensure!(dimensions.iter().all(|n| *n > 0), "Kimi weight extent is empty");
    dimensions.iter().try_fold(1usize, |n, dim| n.checked_mul(*dim)).context("Kimi weight extent overflow")
}

