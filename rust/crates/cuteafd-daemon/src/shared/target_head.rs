//! How every family's target logits run through a BF16 vocabulary head.
//!
//! One route for all families (GLM 5.3, GLM 5.3 Flash, MiMo, Qwen, DeepSeek V4 and V4.1): a BF16
//! head's logits run as a BF16 tensor-core GEMM with FP32 accumulation and FP32 logits at every
//! row count ([`TargetHead::Tensor`], the default). Their reference implementations compute the
//! head as BF16 products with FP32 sums. On five real heads (RTX PRO 6000,
//! final-norm-shaped rows), the fixed route is byte-identical for batches of
//! 1..64 versus serial rows, with FP64-reference mean KL at most 1.46e-11 nats.
//! It is faster than pedantic at all tested shapes (1, 4, 8, 16 and 64 rows);
//! `native/tests/target_head_selftest.py` also checks capacity and graph stability.
//!
//! [`TargetHead::Exact`] is a diagnostic: the pedantic FP32 cuBLAS GEMM (FP32 products and sums on
//! CUDA cores) at every row count. FP8 heads run their own programs and ignore the setting.
//! Drafters keep their own route (`shared::draft`); only target logits come through here.
//!
//! Selected by `CUTEAFD_TARGET_HEAD` (`tensor` or `exact`; launcher key `TARGET_HEAD`), read once
//! per process before the first launch: decode graphs capture the head's launch.
//! The tensor route pins one cuBLASLt algorithm/tile with split-K disabled at load;
//! decode and verify must not choose different reductions based on row count.

use anyhow::{bail, Result};
use cuteafd_ffi::programs::VocabularyHead;
use std::ffi::c_void;

/// How target logits run through a BF16 head.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TargetHead {
    /// BF16 tensor cores, FP32 accumulation and logits, one read of the head, at every row count.
    #[default]
    Tensor,
    /// Diagnostic: the pedantic FP32 cuBLAS GEMM on CUDA cores at every row count.
    Exact,
}

impl TargetHead {
    /// `CUTEAFD_TARGET_HEAD` (unset: [`TargetHead::Tensor`]).
    pub fn parse(value: Option<&str>) -> Result<Self> {
        Ok(match value.map(str::trim) {
            None | Some("" | "tensor") => Self::Tensor,
            Some("exact") => Self::Exact,
            Some(other) => bail!("CUTEAFD_TARGET_HEAD must be tensor or exact, not {other:?}"),
        })
    }

    /// The process's setting, read once. An invalid value is reported and keeps the default.
    pub fn configured() -> Self {
        static MODE: std::sync::OnceLock<TargetHead> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| {
            let value = std::env::var("CUTEAFD_TARGET_HEAD").ok();
            let mode = Self::parse(value.as_deref()).unwrap_or_else(|error| {
                tracing::warn!(%error, "keeping the tensor-core target head");
                Self::Tensor
            });
            tracing::info!(target_head = ?mode, "BF16 target vocabulary heads");
            mode
        })
    }

    /// Whether this setting takes tensor cores (every row count, or none).
    pub fn tensor_cores(self) -> bool {
        self == Self::Tensor
    }
}

/// Target logits `[rows, vocab]` (FP32) of BF16 rows `x [rows, width]` through the BF16 head
/// `weight [vocab, width]` under `mode`.
///
/// # Safety
/// `x`, `weight` and `logits` are live device buffers of these shapes on `stream`'s device until
/// it drains, and `head` was created for this width, vocabulary and at least `rows` rows.
pub(crate) unsafe fn launch(mode: TargetHead, head: &VocabularyHead<'_>, x: *const c_void,
    weight: *const c_void, logits: *mut f32, rows: usize, stream: *mut c_void) -> Result<()> {
    let rows = u32::try_from(rows).map_err(|_| anyhow::anyhow!("target head rows exceed u32"))?;
    // SAFETY: the caller's contract.
    unsafe {
        if mode.tensor_cores() {
            head.launch_tensor_invariant(x.cast(), weight.cast(), logits, rows, stream)
        } else {
            head.launch(x.cast(), weight.cast(), logits, rows, stream)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset and `tensor` take tensor cores; `exact` never does; anything else is refused.
    #[test]
    fn the_target_head_defaults_to_tensor_cores_and_keeps_exact_as_a_diagnostic() {
        assert_eq!(TargetHead::default(), TargetHead::Tensor);
        for value in [None, Some(""), Some("tensor"), Some(" tensor ")] {
            assert_eq!(TargetHead::parse(value).unwrap(), TargetHead::Tensor, "{value:?}");
        }
        assert_eq!(TargetHead::parse(Some("exact")).unwrap(), TargetHead::Exact);
        for value in ["fp8", "pedantic", "Tensor"] {
            assert!(TargetHead::parse(Some(value)).is_err(), "{value}");
        }
        assert!(TargetHead::Tensor.tensor_cores());
        assert!(!TargetHead::Exact.tensor_cores());
    }

    /// Every family's BF16 target logits take this route (GLM 5.3, GLM 5.3 Flash, MiMo, Qwen,
    /// DeepSeek V4, V4.1 single-GPU and split vocabulary heads), and none launches its BF16 target
    /// head on the pedantic GEMM directly.
    #[test]
    fn every_family_target_head_takes_the_shared_route() {
        let sources = [
            ("glm5/engine.rs", include_str!("../families/glm5/engine.rs"), "shared::target_head::launch("),
            ("glm5_flash/engine.rs", include_str!("../families/glm5_flash/engine.rs"), "shared::target_head::launch("),
            ("mimo_v2/head.rs", include_str!("../families/mimo_v2/head.rs"), "shared::target_head::launch("),
            ("qwen4/engine.rs", include_str!("../families/qwen4/engine.rs"), "shared::target_head::launch("),
            ("deepseek_v4/engine.rs", include_str!("../families/deepseek_v4/engine.rs"), "shared::target_head::launch("),
            ("deepseek_v41/v41_target_head.rs", include_str!("../families/deepseek_v41/v41_target_head.rs"),
                "project_target_vocabulary("),
            ("deepseek_v41/v41_target_head/distributed.rs",
                include_str!("../families/deepseek_v41/v41_target_head/distributed.rs"), "project_target_vocabulary("),
        ];
        for (name, source, route) in sources {
            assert!(source.contains(route), "{name} does not take the shared target-head route");
        }
        let v41 = include_str!("../families/deepseek_v41/v41_tensors.rs");
        let target = &v41[v41.find("fn project_target_vocabulary").unwrap()..];
        assert!(target.contains("TargetHead::configured().tensor_cores()") && target.contains("launch_tensor_invariant("));
        // The old direct pedantic launches of the BF16 target heads are gone.
        for (name, source, old) in [
            ("glm5/engine.rs", include_str!("../families/glm5/engine.rs"), "super::launch_head(self.library"),
            ("qwen4/engine.rs", include_str!("../families/qwen4/engine.rs"), "w.head.launch(x.cast()"),
            ("deepseek_v4/engine.rs", include_str!("../families/deepseek_v4/engine.rs"), "?.launch(w.y.buffer"),
        ] {
            assert!(!source.contains(old), "{name} still launches its target head directly");
        }
    }
}
