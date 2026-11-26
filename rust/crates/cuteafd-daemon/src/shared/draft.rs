//! Speculative drafting shared by every family's serve loop.
//!
//! [`history`] is the one draft policy: per-position acceptance history with
//! exploration (`cuteafd_core::DraftHistory`), its launch configuration
//! ([`length`]), trace records and stats. A drafter declares its native block
//! and, where it has a trained confidence head, a raw score per position;
//! nothing else. [`copy`] finds the history copies the policy may verify past
//! an agreeing neural draft. [`clock`] and [`timing`] are profiling only; the
//! policy reads no device signal.
pub(crate) mod clock;
pub(crate) mod copy;
pub(crate) mod dfly2;
pub(crate) mod history;
pub(crate) mod length;
pub(crate) mod remote;
pub(crate) mod remote_dflash;
pub(crate) mod report;
pub(crate) mod timing;

/// Explicit device timing profiles.
pub(crate) fn timing_profile() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| ["CUTEAFD_DRAFT_TIMING", "CUTEAFD_PROTOCOL_V2_TCP_TIMING"].iter()
        .any(|name| std::env::var(name).is_ok_and(|v| matches!(v.as_str(), "1" | "on" | "true"))))
}

/// BF16 drafter proposals use tensor cores at every row count, borrowing the
/// target's only resident head and the drafter's admitted handle/workspace.
/// Target and diagnostic projections keep their own arithmetic.
///
/// # Safety
/// `input` [rows, hidden] BF16, `weight` [vocab, hidden] BF16 and `logits`
/// [rows, vocab] FP32 are live on `stream`'s device until it drains. `head`
/// was created for this geometry and at least `rows` rows.
pub(crate) unsafe fn launch_bf16_head(head: &cuteafd_ffi::programs::VocabularyHead<'_>,
    input: *const u16, weight: *const u16, logits: *mut f32, rows: usize,
    stream: *mut std::ffi::c_void) -> anyhow::Result<()> {
    let rows = u32::try_from(rows).map_err(|_| anyhow::anyhow!("drafter head row count exceeds u32"))?;
    // SAFETY: the caller retains the checked geometry, buffers and workspace.
    unsafe { head.launch_tensor_op(input, weight, logits, rows, stream) }
}
