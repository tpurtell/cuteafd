//! Native stage2 arithmetic geometry and immutable program/owner contracts.
//! No scheduling or placement decisions live here.
use super::Inkling2Config;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitLayout {
    pub rows: usize,
    pub blocks: usize,
    pub buffers: BTreeMap<String, usize>,
}

impl UnitLayout {
    pub fn new(cfg: &Inkling2Config, rows: usize, context: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && rows <= context && context <= cfg.max_position,
            "Inkling arithmetic rows/context outside checkpoint/export");
        ensure!((cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.d_rel, cfg.sconv_kernel, cfg.res_block)
            == (4608, 48, 8, 128, 16, 4, 6), "Inkling native stage2 requires released Small geometry");
        let blocks = cfg.layers.div_ceil(cfg.res_block) + 1;
        let mut buffers = BTreeMap::new();
        let mut add = |name: &str, width: usize, bytes: usize| -> Result<()> {
            let extent = rows.checked_mul(width).and_then(|n| n.checked_mul(bytes))
                .context("Inkling unit extent overflow")?;
            ensure!(buffers.insert(name.into(), extent).is_none(), "duplicate Inkling arithmetic owner");
            Ok(())
        };
        for name in ["partial", "attention", "mlp", "branch", "mixed", "projection_input"] {
            add(name, cfg.hidden, if name == "projection_input" { 2 } else { 4 })?;
        }
        add("residual_arena", blocks * cfg.hidden, 4)?;
        add("global_mixes", 3 * cfg.hidden, 4)?;
        for (name, width, bytes) in [("q", 6144, 4), ("r", 768, 4), ("k", 1024, 4), ("v", 1024, 4),
            ("q_clipped", 6144, 4), ("r_clipped", 768, 4), ("k_clipped", 1024, 4), ("v_norm", 1024, 4),
            ("q_bf16", 6144, 2), ("relative_source", 768, 2), ("relative_input", 48*64, 2), ("relative", 48 * 1024, 2),
            ("relative_f32", 48*1024, 4), ("conv_k", 1024, 4), ("conv_v", 1024, 4),
            ("gate", 6144, 4), ("sinks", 48, 4), ("attended", 6144, 2), ("gated", 6144, 2),
            ("kv_step", 2048, 2), ("k_dynamic", 32, 4), ("v_dynamic", 32, 4),
            ("route_logits", cfg.experts, 4), ("route_ids", cfg.topk, 4), ("route_weights", cfg.topk, 4),
            ("expert_wire", cfg.hidden + cfg.hidden/32, 1), ("expert_output", cfg.hidden, 2),
            ("positions", 1, 8), ("first", 1, 4), ("ring_slots", 1, 8), ("cache_slots", 1, 8)] {
            add(name, width, bytes)?;
        }
        buffers.insert("attention_scratch".into(), 64 * 32 * 48 * (128*4+4));
        buffers.insert("ring".into(), 1088 * 4096);
        buffers.insert("global_cache".into(), context.div_ceil(64).checked_mul(64*4096).context("Inkling cache overflow")?);
        buffers.insert("page_table".into(), context.div_ceil(64)*4);
        // Cold golden reuses one layer's four conv histories serially. Serving
        // history belongs to each layer's cache owner, not this scratch layout.
        for (name, width) in [("attn_conv_history", 4608), ("mlp_conv_history", 4608),
            ("k_conv_history", 1024), ("v_conv_history", 1024)] {
            buffers.insert(name.into(), 3*width*4);
        }
        Ok(Self { rows, blocks, buffers })
    }
    pub fn bytes(&self) -> Result<usize> {
        self.buffers.values().try_fold(0usize, |sum, bytes| sum.checked_add(*bytes).context("Inkling owner sum overflow"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualSite {
    /// Completed blocks present before attention, including the embedding block.
    pub attention_candidates: usize,
    pub mlp_candidates: usize,
    pub queries: usize,
    pub starts_block: bool,
}
impl ResidualSite {
    pub fn layer(cfg: &Inkling2Config, layer: usize) -> Result<Self> {
        ensure!(layer < cfg.layers, "Inkling residual layer outside checkpoint");
        let completed = layer.div_ceil(cfg.res_block);
        let starts_block = layer % cfg.res_block == 0;
        Ok(Self { attention_candidates: completed+1,
            mlp_candidates: completed+1+usize::from(starts_block),
            queries: if cfg.local[layer] { 1 } else { 3 }, starts_block })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> Inkling2Config {
        Inkling2Config::from_hf(&serde_json::from_str(include_str!("fixtures/config.json")).unwrap()).unwrap()
    }
    #[test]
    fn six_layer_residual_boundaries_preserve_embedding_block() {
        let cfg = cfg();
        for (layer, attn, mlp, global) in [(0, 1, 2, false), (1, 2, 2, false), (5, 2, 2, true),
            (6, 2, 3, false), (7, 3, 3, false), (47, 9, 9, true)] {
            let site = ResidualSite::layer(&cfg, layer).unwrap();
            assert_eq!((site.attention_candidates, site.mlp_candidates, site.queries),
                (attn, mlp, if global { 3 } else { 1 }));
        }
    }
    #[test]
    fn every_unit_owner_has_checked_named_extent() {
        let cfg = cfg();
        for rows in [1, 64, 2048, 2752, 4096] {
            let layout = UnitLayout::new(&cfg, rows, 8192).unwrap();
            assert_eq!(layout.buffers["residual_arena"], 9*rows*4608*4);
            assert_eq!(layout.buffers["global_cache"], 8192*4096);
            assert_eq!(layout.buffers["global_mixes"], 3*rows*4608*4);
            assert_eq!(layout.buffers["expert_wire"], rows*(4608+144));
            assert!(layout.bytes().unwrap()>0);
        }
        for rows in [0, 4097, usize::MAX] { assert!(UnitLayout::new(&cfg, rows, 8192).is_err()); }
    }
}
