//! One coordinator rank's lane planes: every activation a layer step reads or
//! writes, at the lane's row capacity, as one exact extent the planner charges
//! and the device allocates (plan == runtime). Prefill units in flight share
//! the lane; the Spark return planes are charged separately (`prefill_return_bytes`).
//!
//! Head-split TP2: q/k/v, attention output and gate hold this rank's heads; the
//! residual stream and everything outside attention is whole on each GPU.
use super::MuseSparkConfig;
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// Plane name -> bytes, in a fixed order (BTreeMap) so the device's offsets
/// are deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanePlanes {
    pub rows: usize,
    pub planes: BTreeMap<&'static str, usize>,
}

pub const ALIGN: usize = 256;

impl LanePlanes {
    /// `rows`: the lane's capacity (one prefill unit, or the decode/verify width).
    pub fn new(cfg: &MuseSparkConfig, ranks: usize, rows: usize) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2) && (1..=4096).contains(&rows), "Muse lane geometry");
        let (h, e, d) = (cfg.hidden, cfg.expert_hidden, cfg.head_dim);
        let (q, kv) = (cfg.heads * d / ranks, cfg.kv_heads * d / ranks);
        let k = cfg.topk;
        let wire = e + e / 32;
        let bf16 = |cols: usize| rows * cols * 2;
        let f32 = |cols: usize| rows * cols * 4;
        let mut planes = BTreeMap::from([
            // Residual stream, its normed copies and the gated residual outputs.
            ("hidden", bf16(h)), ("normed", bf16(h)), ("residual", bf16(h)),
            // Attention: raw projections, normed/RoPEd q and k, the step's K|V records,
            // attention output, gate, gated output and the o_proj result.
            ("q_raw", bf16(q)), ("k_raw", bf16(kv)), ("q", bf16(q)), ("k", bf16(kv)),
            ("kv_step", bf16(2 * kv)), ("attention", bf16(q)), ("gate", bf16(q)), ("gated", bf16(q)),
            ("attended", bf16(h)),
            // MoE: router logits, selected routes, the expert-width stream and its wire,
            // the combined experts and the post-projection.
            ("router_logits", f32(cfg.experts)), ("route_ids", rows * k * 4), ("route_weights", f32(k)),
            ("expert_input", bf16(e)), ("expert_wire", rows * wire), ("combined", bf16(e)), ("moe", bf16(h)),
            // Step metadata.
            ("positions", rows * 8), ("ring_slots", rows * 8), ("seq_first", rows * 4),
        ]);
        if ranks == 2 {
            // The TP2 o_proj F32 partial lands from the peer.
            planes.insert("o_peer", f32(h));
        }
        let lane = Self { rows, planes };
        lane.bytes()?;
        Ok(lane)
    }

    /// Byte offsets of each plane in one allocation, each `ALIGN`-aligned.
    pub fn offsets(&self) -> Result<BTreeMap<&'static str, (usize, usize)>> {
        let mut at = 0usize;
        let mut out = BTreeMap::new();
        for (&name, &bytes) in &self.planes {
            out.insert(name, (at, bytes));
            at = at.checked_add(bytes.next_multiple_of(ALIGN)).context("Muse lane plane overflow")?;
        }
        Ok(out)
    }

    pub fn bytes(&self) -> Result<usize> {
        self.planes.values().try_fold(0usize, |n, b| n.checked_add(b.next_multiple_of(ALIGN)))
            .context("Muse lane bytes overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_lane_planes() -> Result<()> {
        let cfg = MuseSparkConfig::from_hf(&crate::plan::testing::muse_spark_config())?;
        let whole = LanePlanes::new(&cfg, 1, 2752)?;
        let pair = LanePlanes::new(&cfg, 2, 2752)?;
        assert_eq!(whole.planes["q"], 2752 * 8192 * 2);
        assert_eq!(pair.planes["q"] * 2, whole.planes["q"]);
        assert_eq!(pair.planes["kv_step"], 2752 * 8 * 64 * 2 * 2);
        assert_eq!(whole.planes["expert_wire"], 2752 * 4224);
        assert!(!whole.planes.contains_key("o_peer") && pair.planes.contains_key("o_peer"));
        let offsets = pair.offsets()?;
        assert!(offsets.values().all(|(at, _)| at % ALIGN == 0));
        let end = offsets.values().map(|(at, n)| at + n).max().unwrap();
        assert!(end <= pair.bytes()?);
        assert!(LanePlanes::new(&cfg, 3, 64).is_err() && LanePlanes::new(&cfg, 1, 4097).is_err());
        Ok(())
    }
}
