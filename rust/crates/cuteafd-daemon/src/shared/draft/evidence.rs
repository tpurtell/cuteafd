//! Where a request's drafts came from and what the drafter said about them.
//!
//! Every drafted position carries one piece of evidence, which the policy
//! turns into a conditional acceptance probability: a prior per (family,
//! drafter, numerics) key maps evidence to `p0`, then the policy's online
//! per-position Platt scaling calibrates it. Only positions whose
//! predecessors were all accepted carry evidence; a verification that stopped
//! for a reason other than a miss is censored, never booked as one.

/// Who proposed a request's verifier rows this round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DraftSource {
    /// The family's neural drafter (dSpark, DFlash, MTP).
    Neural,
    /// A copy span from the request's own history. Its rows train the cost
    /// fits and route history, never neural evidence.
    Copy,
    /// No drafts: the anchor alone (short history, exhausted budget, probe).
    Undrafted,
}

/// DFlash2 selector features of one drafted position: its candidate's margin
/// over the runner-up, its probability, the candidate distribution's entropy,
/// and its unary rank. No probability of its own until a keyed prior maps it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SelectorFeatures {
    pub margin: f32,
    pub p_top: f32,
    pub entropy: f32,
    pub rank: f32,
}

/// What the drafter reported per drafted position, verified or not.
#[allow(dead_code, reason = "D3/D4 bind the MTP and copy evidence")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Evidence<'a> {
    /// A trained confidence head's sigmoid per position (dSpark on V4.1 and
    /// GLM Flash). Its prior is the identity.
    Head(&'a [f64]),
    /// DFlash2's selector features per position and the keyed `SelectorFit`
    /// prior's probabilities of them. The binding evaluates the prior, since
    /// its first feature is the request's own history rate.
    Selector { features: &'a [SelectorFeatures], prior: &'a [f64] },
    /// Nothing but the request's past outcomes (MTP).
    History,
    /// A copy span of `match_len` matched tokens.
    Copy { match_len: usize },
}

impl<'a> Evidence<'a> {
    /// The prior's per-position probability `p0`, where this evidence has a
    /// prior today: a head's sigmoid, or the selector fit's output. `History`
    /// and `Copy` priors arrive with their first bindings (D3/D4); until then
    /// they carry no neural evidence.
    pub fn prior(self) -> Option<&'a [f64]> {
        match self {
            Self::Head(probabilities) | Self::Selector { prior: probabilities, .. } => Some(probabilities),
            Self::History | Self::Copy { .. } => None,
        }
    }
}

/// Why a request's verification stopped before a miss could be observed.
/// V4.1 truncates by grammar and budget before verification and books stops
/// as before; GLM Flash (D2) censors stops on accepted drafts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Censor {
    /// The request emitted its stop token on an accepted draft.
    Eos,
    /// The request reached its output limit.
    OutputLimit,
    /// The request was cancelled before its round completed.
    Cancelled,
}

/// A trained head's logit as its sigmoid probability.
pub(crate) fn sigmoid(x: f32) -> f64 {
    let x = f64::from(x);
    if x >= 0. { 1. / (1. + (-x).exp()) } else { x.exp() / (1. + x.exp()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_and_selector_evidence_have_priors() {
        let p = [0.9, 0.5];
        assert_eq!(Evidence::Head(&p).prior(), Some(&p[..]));
        let features = [SelectorFeatures { margin: 1., p_top: 0.8, entropy: 0.3, rank: 0. }];
        let fitted = [0.7];
        assert_eq!(Evidence::Selector { features: &features, prior: &fitted }.prior(), Some(&fitted[..]));
        assert_eq!(Evidence::History.prior(), None);
        assert_eq!(Evidence::Copy { match_len: 8 }.prior(), None);
        assert_eq!((sigmoid(0.), sigmoid(40.), sigmoid(-40.) > 0.), (0.5, 1., true));
    }
}
