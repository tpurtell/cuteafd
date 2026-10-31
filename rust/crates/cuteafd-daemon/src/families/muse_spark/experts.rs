//! The Spark expert path behind one engine contract. The engine hands a
//! layer's expert-width rows and their routes to [`SparkExperts::send`] and,
//! after [`SparkExperts::land`], reads the combined expert output: one E-wide
//! BF16 row per row in the lane's `combined` plane (each routed expert's
//! output post_expert_normed, weighted and summed over the top-k). How the
//! Sparks hold experts, what they reply and how the coordinator combines it
//! are the implementation's; the engine sees none of it.
//!
//! The Muse residency is TP (the EP study, `ep_layouts.py`, approved
//! 2026-10-31): every Spark holds an intermediate slice of every expert and
//! replies with its partial down-proj output per routed (row, expert) as
//! NVFP4 (E2M1 with an E4M3 scale per 16, the hardware cvt); the coordinator
//! program `muse_tp_combine` dequantizes, sums the ranks per expert, applies
//! post_expert_norm and the routing weight, and sums the top-k into `combined`.
//! FP8 (E4M3, UE8M0 per 32) is the documented fallback wire.
use anyhow::{ensure, Result};

/// The wire of one rank's per-(row, expert) partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartialWire {
    /// E2M1 values with an E4M3 scale per 16 elements (4.5 bits per element).
    Nvfp4,
    /// E4M3 values with a UE8M0 scale per 32 elements (8.25 bits per element).
    Fp8,
}

impl PartialWire {
    /// Bytes of one E-wide partial row on the wire (values, then scales).
    pub(crate) fn row_bytes(self, width: usize) -> Result<usize> {
        Ok(match self {
            Self::Nvfp4 => { ensure!(width % 32 == 0, "NVFP4 partial width"); width / 2 + width / 16 }
            Self::Fp8 => { ensure!(width % 32 == 0, "FP8 partial width"); width + width / 32 }
        })
    }
}

/// The partial replies of one wave: `ranks` planes of `rows * topk` partial
/// rows, landed on the coordinator for `muse_tp_combine`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PartialReplies {
    pub wire: PartialWire,
    pub ranks: usize,
    pub topk: usize,
    pub width: usize,
}

impl PartialReplies {
    /// Reply payload bytes one rank sends for `rows` rows.
    pub(crate) fn rank_bytes(&self, rows: usize) -> Result<usize> {
        ensure!(rows > 0 && self.ranks > 0 && self.topk > 0, "Muse partial reply geometry");
        Ok(rows * self.topk * self.wire.row_bytes(self.width)?)
    }

    /// Coordinator landing bytes for `rows` rows (every rank's partials).
    pub(crate) fn landing_bytes(&self, rows: usize) -> Result<usize> {
        Ok(self.rank_bytes(rows)? * self.ranks)
    }
}

/// The combine launches one wave needs: one per non-empty region, by format
/// (`muse_tp_combine_{bf16,nvfp4,fp8}`), each with its region's row count. A
/// region with no rows is skipped here, never launched with zero rows.
pub(crate) fn combine_launches(plan: &cuteafd_loader::families::muse_spark::replies::ReplyPlan)
    -> Vec<(&'static str, usize)> {
    use cuteafd_loader::families::muse_spark::replies::PartialFormat;
    let wide = match plan.wide { PartialFormat::Bf16 => "tp_combine_bf16", PartialFormat::Nvfp4 => "tp_combine_nvfp4",
        PartialFormat::Fp8 => "tp_combine_fp8" };
    [("tp_combine_bf16", plan.bf16_rows.len()), (wide, plan.wide_rows.len())]
        .into_iter().filter(|(_, rows)| *rows > 0).collect()
}

/// One wave in flight: the layer and rows it carries, in send order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Wave { pub layer: usize, pub rows: usize, pub sequence: u64 }

/// The engine's view of the Spark experts. `send` enqueues the expert-width
/// rows' wire quantization and the routes' copies on the lane stream and posts
/// the request once they land (the host never waits on the GPU); `land` makes
/// the lane stream wait for the wave's replies and leaves the combined expert
/// output in the lane's `combined` plane. Waves land in send order.
pub(crate) trait SparkExperts {
    fn send(&mut self, layer: usize, rows: usize) -> Result<Wave>;
    fn land(&mut self, wave: Wave) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The study's reply sizes: NVFP4 partials are top-k x 4.5-bit rows per
    /// rank (2.25x one E-wide BF16 row at k = 8); FP8 is the 8.25-bit fallback.
    #[test]
    fn partial_reply_extents_match_the_layout_study() -> Result<()> {
        let nvfp4 = PartialReplies { wire: PartialWire::Nvfp4, ranks: 6, topk: 8, width: 4096 };
        assert_eq!(nvfp4.rank_bytes(1)?, 8 * (2048 + 256));
        assert_eq!(nvfp4.rank_bytes(1)? as f64 / (4096.0 * 2.0), 2.25);
        assert_eq!(nvfp4.landing_bytes(2752)?, 6 * 2752 * 8 * 2304);
        let fp8 = PartialReplies { wire: PartialWire::Fp8, ..nvfp4 };
        assert_eq!(fp8.rank_bytes(2752)?, 2752 * 8 * 4224);
        assert!(PartialWire::Nvfp4.row_bytes(4100).is_err());
        assert!(nvfp4.rank_bytes(0).is_err());
        Ok(())
    }

    #[test]
    fn empty_regions_are_never_launched() -> Result<()> {
        use cuteafd_loader::families::muse_spark::replies::{Cutoff, ReplyPlan, WideFormat};
        let cutoff = Cutoff::with_wide(64, 8, WideFormat::Nvfp4)?;
        assert_eq!(combine_launches(&ReplyPlan::new(cutoff, &[1, 8])), [("tp_combine_bf16", 9)]);
        assert_eq!(combine_launches(&ReplyPlan::new(cutoff, &[2752])), [("tp_combine_nvfp4", 2752)]);
        assert_eq!(combine_launches(&ReplyPlan::new(cutoff, &[1, 2000])),
            [("tp_combine_bf16", 1), ("tp_combine_nvfp4", 2000)]);
        let fp8 = Cutoff::with_wide(64, 8, WideFormat::Fp8)?;
        assert_eq!(combine_launches(&ReplyPlan::new(fp8, &[100])), [("tp_combine_fp8", 100)]);
        assert!(combine_launches(&ReplyPlan::new(cutoff, &[])).is_empty());
        Ok(())
    }
}
