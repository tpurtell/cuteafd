//! Per-rank coordinator weight layout of one Muse Spark layer and the
//! globals: the operand inventory and byte extents the device packs and the
//! planner charges (one definition, so plan == runtime). Attention heads split
//! across a head-split pair (64 query / 8 KV heads per GPU); everything else
//! is whole on each GPU. K6 projections keep their payload + H128 vectors;
//! the residual gates are packed as host-computed F32 alpha/beta.
use super::{core::{CoreInventory, CoreSplit, PREFIX}, MuseSparkConfig};
use crate::formats::{dense_trellis::TrellisSide, trellis_core::CoreWeightLayout};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// One rank's operands of one layer: plain operand bytes by name, and K6 core
/// layouts by name. Names are the device binding names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerLayout {
    pub operands: BTreeMap<&'static str, usize>,
    pub core: BTreeMap<&'static str, CoreWeightLayout>,
}

impl LayerLayout {
    pub fn bytes(&self) -> Result<usize> {
        self.operands.values().copied().chain(self.core.values().map(|c| c.bytes))
            .try_fold(0usize, |n, b| n.checked_add(b).context("Muse layer weight extent overflow"))
    }

    /// Bytes of the head-split attention operands on this rank (query, key,
    /// value, output and gate projections): what the solver charges per mode.
    pub fn attention_bytes(&self) -> Result<usize> {
        self.operands.iter().filter(|(n, _)| n.starts_with("attn.")).map(|(_, b)| *b)
            .chain(self.core.iter().filter(|(n, _)| n.starts_with("attn.")).map(|(_, c)| c.bytes))
            .try_fold(0usize, |n, b| n.checked_add(b).context("Muse attention extent overflow"))
    }

    /// The layout of every rank (one RTX: `ranks` 1; a head-split pair: 2).
    pub fn ranks(cfg: &MuseSparkConfig, layer: usize, ranks: usize, core: &CoreInventory) -> Result<Vec<Self>> {
        ensure!(layer < cfg.layers && matches!(ranks, 1 | 2) && cfg.heads % ranks == 0 && cfg.kv_heads % ranks == 0,
            "Muse layer layout: layer {layer} of {}, ranks {ranks}", cfg.layers);
        let (h, d, e) = (cfg.hidden, cfg.head_dim, cfg.expert_hidden);
        let (q, kv) = (cfg.heads * d / ranks, cfg.kv_heads * d / ranks);
        let root = format!("{PREFIX}{layer}");
        (0..ranks).map(|rank| {
            let mut operands = BTreeMap::new();
            let mut cores = BTreeMap::new();
            let mut matrix = |name: &'static str, stem: &str, n: usize, k: usize, split: Option<CoreSplit>| -> Result<()> {
                match core.projections.get(&format!("{root}.{stem}")) {
                    Some(projection) => {
                        let side = match (split, ranks) {
                            (Some(CoreSplit::Input), 2) => TrellisSide::Input,
                            _ => TrellisSide::Output,
                        };
                        let (r, w) = if split.is_some() && ranks == 2 { (rank, 2) } else { (0, 1) };
                        cores.insert(name, CoreWeightLayout::new(projection, side, r, w)?);
                    }
                    None => { operands.insert(name, n * k * 2); }
                }
                Ok(())
            };
            // Head-split attention (TP2 output units, o_proj input units).
            matrix("attn.q", "self_attn.q_proj", q, h, Some(CoreSplit::Output))?;
            matrix("attn.k", "self_attn.k_proj", kv, h, Some(CoreSplit::Output))?;
            matrix("attn.v", "self_attn.v_proj", kv, h, Some(CoreSplit::Output))?;
            matrix("attn.o", "self_attn.o_proj", h, q, Some(CoreSplit::Input))?;
            matrix("attn.gate", "self_attn.gate_proj", q, h, Some(CoreSplit::Output))?;
            // Whole on every rank: the MoE frontend/backend, router and vectors.
            matrix("moe.pre_expert", "mlp.pre_expert_proj", e, h, None)?;
            matrix("moe.post_expert", "mlp.post_expert_proj", h, e, None)?;
            operands.insert("moe.router", cfg.experts * h * 4);
            operands.insert("moe.bias", cfg.experts * 4);
            operands.insert("moe.pre_expert_norm", e * 2);
            for norm in ["norm.input", "norm.pre_ffn", "norm.post_ffn"] { operands.insert(norm, h * 2); }
            // Residual gates as host-computed F32 (alpha, beta).
            for gate in ["gate.attention.alpha", "gate.attention.beta", "gate.ffn.alpha", "gate.ffn.beta"] {
                operands.insert(gate, h * 4);
            }
            Ok(Self { operands, core: cores })
        }).collect()
    }
}

/// Globals: the embedding (on the entry GPU), the final norm (each GPU) and the
/// head (on the head GPU; K6, stored N128-padded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalLayout {
    pub embedding: usize,
    pub final_norm: usize,
    pub head: Option<CoreWeightLayout>,
    pub head_bf16: usize,
}

impl GlobalLayout {
    pub fn new(cfg: &MuseSparkConfig, core: &CoreInventory) -> Result<Self> {
        let head = core.get_weight(super::core::HEAD).map(|p| CoreWeightLayout::new(p, TrellisSide::Output, 0, 1))
            .transpose()?;
        Ok(Self { embedding: cfg.vocab_size * cfg.hidden * 2, final_norm: cfg.hidden * 2,
            head_bf16: if head.is_some() { 0 } else { cfg.vocab_size * cfg.hidden * 2 }, head })
    }

    pub fn head_bytes(&self) -> usize { self.head.as_ref().map_or(self.head_bf16, |h| h.bytes) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_layer_layout_on_one_and_two_gpus() {
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::write_muse_spark_snapshot(dir.path(), &[]);
        let checkpoint = crate::plan::Checkpoint::inventory(dir.path()).unwrap();
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config).unwrap();
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint).unwrap();
        let whole = LayerLayout::ranks(&cfg, 3, 1, &core).unwrap().remove(0);
        let pair = LayerLayout::ranks(&cfg, 3, 2, &core).unwrap();
        // K6 q/k/v/o and pre/post expert; gate_proj BF16 in this publication.
        assert_eq!(whole.core.len(), 6);
        assert_eq!(whole.operands["attn.gate"], 8192 * 8192 * 2);
        assert_eq!(pair[0].operands["attn.gate"], 4096 * 8192 * 2);
        // Head-split attention halves across the pair; whole operands repeat.
        for rank in &pair {
            assert_eq!(rank.core["attn.q"].shape, [4096, 8192]);
            assert_eq!(rank.core["attn.o"].shape, [8192, 4096]);
            assert_eq!(rank.core["moe.pre_expert"], whole.core["moe.pre_expert"]);
            assert_eq!(rank.operands["moe.router"], 256 * 8192 * 4);
        }
        let pair_total = pair[0].attention_bytes().unwrap() + pair[1].attention_bytes().unwrap();
        // Split halves sum to the whole payload plus one replicated input/output vector set.
        assert!(pair_total >= whole.attention_bytes().unwrap());
        assert!(pair_total - whole.attention_bytes().unwrap() < 1 << 20);
        let global = GlobalLayout::new(&cfg, &core).unwrap();
        assert_eq!(global.head.as_ref().unwrap().shape, [202112, 8192]);
    }
}
