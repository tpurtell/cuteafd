//! Engine plumbing for the resource-priced draft policy
//! (`cuteafd_core::draft_policy`), shared by every family's serve loop.
//!
//! A round gives the policy three signals, each taken where the engine already
//! has the data:
//! - [`clock::LayerClock`]: per-layer device time from CUDA timing events
//!   recorded between graph launches, read once after the round's own sync;
//! - [`routes::RoundRoutes`]: the round's routed expert ids per layer and
//!   verifier row, copied from host staging (Spark layers) or a device ring
//!   (local layers);
//! - [`clock::RoundClock`]: host round and draft-call boundaries.
//!
//! [`evidence`] names where each request's drafts came from and what the
//! drafter said about them, with the censor reasons that keep unverified
//! positions out of the acceptance evidence. [`binding`] turns one completed
//! round into the policy's observation and owns the per-lane state a serve loop
//! carries between selection and observation.
pub(crate) mod binding;
pub(crate) mod clock;
pub(crate) mod copy;
pub(crate) mod evidence;
pub(crate) mod probe;
pub(crate) mod remote;
pub(crate) mod resource;
pub(crate) mod routes;
pub(crate) mod runtime;
pub(crate) mod timing;

/// Apply the opt-in length rules: `CUTEAFD_DRAFT_SURVIVAL_LENGTHS` (launcher
/// key `DRAFT_SURVIVAL_LENGTHS`) verifies drafts while their calibrated
/// survival is at least one half, with width, skip and copy actions still
/// resource priced; `CUTEAFD_DRAFT_POLICY` (launcher key `DRAFT_POLICY`,
/// `shared` by default) selects the experimental simple rules
/// (`cuteafd_core::DraftMode`).
pub(crate) fn apply_length_rule(policy: &mut cuteafd_core::DraftPolicy) {
    let survival = super::draft_policy::enabled("CUTEAFD_DRAFT_SURVIVAL_LENGTHS");
    if survival { tracing::info!("draft lengths follow calibrated survival >= 0.5"); }
    policy.set_survival_lengths(survival);
    policy.set_draft_mode(draft_mode());
}

/// The configured draft mode; an unknown value keeps the shared policy.
pub(crate) fn draft_mode() -> cuteafd_core::DraftMode {
    static MODE: std::sync::OnceLock<cuteafd_core::DraftMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let value = std::env::var("CUTEAFD_DRAFT_POLICY").unwrap_or_default();
        let mode = cuteafd_core::DraftMode::parse(&value).unwrap_or_else(|| {
            tracing::warn!(value, "unknown CUTEAFD_DRAFT_POLICY; using the shared policy");
            cuteafd_core::DraftMode::Shared
        });
        if mode != cuteafd_core::DraftMode::Shared { tracing::info!(mode = mode.label(), "simple draft policy selected"); }
        mode
    })
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
