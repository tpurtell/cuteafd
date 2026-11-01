//! Loop abort: the shared serving-path wiring for [`RepeatGuard`].
//!
//! Every family's serve loop builds one [`LoopAbort`] per admitted request from
//! the live [`LoopAbortConfig`], feeds it each generated token id, and treats a
//! `true` from [`LoopAbort::push`] exactly like the `max_tokens` finish: flush
//! the decoder, send `Finish { Length }`, and retire the row in the same round.
//! The abort also sets `usage.finish_detail = "repetition"` and bumps the
//! `loop_aborts` console total.
//!
//! The knobs are generic (`--loop-abort-run`, `CUTEAFD_LOOP_ABORT_RUN`, ...) so
//! every family shares them; the rule itself lives in [`super::repeat_guard`].

use clap::Args;
use cuteafd_api::openai::FinishDetail;

use super::repeat_guard::{LoopHit, RepeatGuard};

/// The four loop-abort knobs, flattened into every family's serve args.
#[derive(Debug, Clone, Copy, Args)]
pub(crate) struct LoopAbortArgs {
    /// Loop abort: end a response with finish_reason=length once this many
    /// consecutive 32-token shingles are an eighth-or-later verbatim copy of
    /// earlier output (rule R). 0 = off.
    #[arg(long, default_value_t = 512, env = "CUTEAFD_LOOP_ABORT_RUN",
        value_parser = clap::value_parser!(u32).range(0..=65536))]
    pub loop_abort_run: u32,

    /// Loop abort: how many verbatim copies of a block count as a loop (rule R).
    /// 8 = the shipped patience.
    #[arg(long, default_value_t = 8, env = "CUTEAFD_LOOP_ABORT_COPIES",
        value_parser = clap::value_parser!(u32).range(2..=64))]
    pub loop_abort_copies: u32,

    /// Loop abort: the fast short-cycle rule (F) fires once the copies after the
    /// first span at least this many tokens. 0 = off.
    #[arg(long, default_value_t = 384, env = "CUTEAFD_LOOP_FAST_SPAN",
        value_parser = clap::value_parser!(u32).range(0..=65536))]
    pub loop_fast_span: u32,

    /// Loop abort: how many identical copies of a short unit count as a loop
    /// (rule F). 8 = the shipped patience.
    #[arg(long, default_value_t = 8, env = "CUTEAFD_LOOP_FAST_COPIES",
        value_parser = clap::value_parser!(u32).range(2..=64))]
    pub loop_fast_copies: u32,
}

impl Default for LoopAbortArgs {
    fn default() -> Self {
        Self { loop_abort_run: 512, loop_abort_copies: 8, loop_fast_span: 384, loop_fast_copies: 8 }
    }
}

impl LoopAbortArgs {
    /// The resolved, copyable config a serve loop hands to each request.
    pub(crate) fn config(&self) -> LoopAbortConfig {
        LoopAbortConfig { run: self.loop_abort_run, copies: self.loop_abort_copies,
            fast_span: self.loop_fast_span, fast_copies: self.loop_fast_copies }
    }
}

/// The resolved loop-abort knobs. `Copy`, so a serve loop can build one guard
/// per request without borrowing the args.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LoopAbortConfig {
    pub(crate) run: u32,
    pub(crate) copies: u32,
    pub(crate) fast_span: u32,
    pub(crate) fast_copies: u32,
}

impl Default for LoopAbortConfig {
    fn default() -> Self {
        LoopAbortArgs::default().config()
    }
}

impl LoopAbortConfig {
    /// Build the per-request guard, sharing the request's finish detail.
    pub(crate) fn guard(&self, detail: FinishDetail, request_id: u64) -> LoopAbort {
        LoopAbort {
            guard: RepeatGuard::new(self.run, self.copies, self.fast_span, self.fast_copies),
            detail,
            request_id,
        }
    }
}

/// One request's loop guard: the detector plus the request-local finish detail.
pub(crate) struct LoopAbort {
    guard: RepeatGuard,
    detail: FinishDetail,
    request_id: u64,
}

impl LoopAbort {
    /// Feed one generated token. Returns `true` when the reply must be aborted
    /// now; the finish detail and the console total are updated as a side effect.
    pub(crate) fn push(&mut self, token: u32) -> bool {
        let Some(hit) = self.guard.push(token) else { return false };
        self.detail.set_repetition();
        super::console::totals::loop_abort();
        let (rule, period) = match hit {
            LoopHit::Repeat { period } => ("repeat", Some(period)),
            LoopHit::Fast { period } => ("fast", Some(period as usize)),
        };
        tracing::info!(target: "cuteafd::loop_abort", request_id = self.request_id,
            rule, period, run = self.guard.consecutive(), period_hint = ?self.guard.period_hint(),
            last_ids = ?self.guard.last_ids(),
            "loop abort: repeated block; finishing with length");
        true
    }

    /// The request-local finish detail, for the response renderer.
    pub(crate) fn detail(&self) -> &FinishDetail { &self.detail }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_config_never_aborts() {
        let config = LoopAbortConfig { run: 0, copies: 8, fast_span: 0, fast_copies: 8 };
        let detail = FinishDetail::default();
        let mut abort = config.guard(detail.clone(), 1);
        for token in 0..10_000u32 {
            assert!(!abort.push(token));
        }
        assert_eq!(detail.name(), None);
    }

    #[test]
    fn abort_sets_the_request_local_detail() {
        let config = LoopAbortConfig::default();
        let detail = FinishDetail::default();
        let other = FinishDetail::default();
        let mut abort = config.guard(detail.clone(), 7);
        let unit: Vec<u32> = (2..602).collect();
        let mut aborted = false;
        for _ in 0..9 {
            for &token in &unit {
                if abort.push(token) { aborted = true; break; }
            }
            if aborted { break; }
        }
        assert!(aborted, "the shipped guard aborts a nine-copy loop");
        assert_eq!(detail.name(), Some("repetition"));
        assert_eq!(other.name(), None, "the detail is request-local");
        assert_eq!(abort.detail().name(), Some("repetition"));
    }
}
