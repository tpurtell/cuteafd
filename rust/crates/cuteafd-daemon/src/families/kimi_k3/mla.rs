//! Family caller for W7's native NoPE MLA chain. W9 supplies the owning
//! allocations, launch metadata and TP2 partial-sum exchange.
use anyhow::{ensure, Result};
use cuteafd_core::prefix::ReuseRule;
use cuteafd_ffi::{programs::Programs, KimiMla, KimiMlaBatch, KimiMlaMode, KimiMlaWeights, MLA_RECORD_BYTES};
use cuteafd_loader::families::kimi_k3::config::{KimiAttention, KimiK3Config};
use std::ffi::c_void;

/// MLA pages alone could resume at a page boundary, but KDA recurrent state
/// belongs to the captured frontier. The whole family must restore exactly.
pub(crate) const PREFIX_REUSE: ReuseRule = ReuseRule::EXACT;

pub(crate) struct Mla<'a> {
    native: KimiMla<'a>,
    layers: Vec<KimiAttention>,
}

impl<'a> Mla<'a> {
    pub(crate) fn load(cfg: &KimiK3Config, programs: &'a Programs<'a>, heads: usize,
                       capacity: usize, mode: KimiMlaMode) -> Result<Self> {
        ensure!(cfg.hidden == 7168 && cfg.heads == 96 && cfg.q_lora_rank == 1536
            && cfg.kv_lora_rank == 512 && cfg.qk_nope_dim == 128 && cfg.qk_rope_dim == 64
            && cfg.v_head_dim == 128, "unsupported Kimi native MLA geometry");
        ensure!(cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi MLA cache accounting mismatch");
        Ok(Self { native: KimiMla::load(programs, heads, capacity, mode)?, layers: cfg.attention.clone() })
    }

    pub(crate) fn scratch_bytes(&self) -> [usize; 3] { self.native.scratch_bytes() }
    pub(crate) fn max_context(&self) -> usize { self.native.max_context() }
    pub(crate) fn splits(&self) -> usize { self.native.splits() }

    /// Output is head-local at 48 heads and complete at 96 heads. This call
    /// never applies the TP2 meet twice or substitutes an ungated projection.
    pub(crate) fn forward(&self, layer: usize, weights: KimiMlaWeights,
                          batch: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        ensure!(self.layers.get(layer) == Some(&KimiAttention::Mla), "layer {layer} is not Kimi MLA");
        self.native.launch(weights, batch, stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_cache_bytes_and_recurrent_prefix_policy() {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        assert_eq!(cfg.mla_record_values() * 2, MLA_RECORD_BYTES);
        assert_eq!(cfg.mla_layers() * MLA_RECORD_BYTES, 27 * 1024);
        assert!(!PREFIX_REUSE.partial());
        assert_eq!(PREFIX_REUSE.skipped(65, 65), 65);
        assert_eq!(PREFIX_REUSE.skipped(64, 65), 0);
        assert_eq!(PREFIX_REUSE.skipped(65, 128), 0);
    }
}
