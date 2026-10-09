//! Likelihood-ranked verify rows (`serve-glmf --verify-policy chain`), after glm53f-afd's
//! verify-length policy (hughmadden/glm53f-afd v1.1.0, MIT: `crates/glm53f-coordinator/src/spec.rs`).
//!
//! The default policy (`cost`) gives every sequence the same room under the verify budget
//! (`GlmfEngine::verify_rows`: 64 rows, or the GPU's whole sparse MLA waves with `--decode-rows
//! 128`; `verify_rows / sequences - 1` drafts, 3 at 16 sequences and 64 rows) and lets the cost
//! model choose a depth within it. `chain` spends the budget by draft likelihood instead: each
//! sequence's drafts are cut where the product of the drafter's probabilities through the draft
//! (the chain's own estimate that the draft is kept) falls below tau, 0.7; when the step's rows
//! still exceed the budget, the least likely drafts across all sequences are dropped first. A
//! confident sequence verifies up to its seven drafts while a doubtful one verifies few, so the
//! rows go to the drafts most likely to be kept.
//!
//! It changes only which drafts a step verifies: verification decides what is kept. Given the
//! same target logits both policies emit the same tokens (`serve::verify_policy_tests`); on the
//! GPU a different verify shape can round a near-tie the other way, so chain need not be
//! output-identical to cost, while every emitted greedy token is its own verified row's argmax.
//!
//! The drafter's probability of a draft is its selector's softmax over the position's 16
//! candidates (DFlash2: the `best probability` feature, as glm53f-afd's `conf`), or a dSpark
//! confidence head's predicted acceptance. A copy-window token the drafter also proposed keeps the
//! drafter's probability; any other copied token counts [`COPY_TOKEN_P`].

/// The default chain cut (`--spec-tau`): on GLM-5.3-Flash with DFlash2, 0.5 and 0.7 were equal or
/// faster than 0.3 on code, prose and counting in glm53f-afd, and 0.7 verifies fewer drafts.
pub(crate) const DEFAULT_TAU: f64 = 0.7;

/// The likelihood the row budget gives a copied token the drafter did not propose (glm53f-afd's
/// `copy::TOKEN_P`: the measured 94% for the next token of a copy backed by 8 or more tokens).
pub(crate) const COPY_TOKEN_P: f32 = 0.94;

/// Which drafts a speculative step verifies (`--verify-policy`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum VerifyPolicy {
    /// Every sequence the same room (`verify_rows / sequences - 1` drafts), the cost model's depth
    /// within it.
    #[default]
    Cost,
    /// Each sequence's drafts cut at `--spec-tau` cumulative probability, then the least likely
    /// drafts across sequences dropped first until the step fits the verify budget.
    Chain,
}

impl VerifyPolicy {
    /// Drafts each of `sequences` may add after its next token under a budget of `verify_rows`
    /// rows: the even share (cost), or the whole budget, cut by [`budget`] after drafting (chain).
    pub(crate) fn room(self, verify_rows: usize, sequences: usize) -> usize {
        match self {
            VerifyPolicy::Cost => (verify_rows / sequences.max(1)).max(1) - 1,
            VerifyPolicy::Chain => verify_rows.max(1) - 1,
        }
    }
}

/// A drafter probability as a likelihood: clamped to [0, 1], 0 when it is not a number.
fn likelihood(p: f32) -> f64 {
    let p = f64::from(p);
    if p.is_finite() { p.clamp(0.0, 1.0) } else { 0.0 }
}

/// Drafts one sequence verifies under the chain cut: while the product of the drafter's
/// probabilities through the draft stays at or above `tau`, at most `cap` and at most the drafts
/// it has.
pub(crate) fn chain_length(probs: &[f32], cap: usize, tau: f64) -> usize {
    let cap = cap.min(probs.len());
    let mut kept = 1.0f64;
    for (j, &p) in probs.iter().enumerate().take(cap) {
        kept *= likelihood(p);
        if kept < tau {
            return j;
        }
    }
    cap
}

/// At most `max_rows` verify rows in all (each sequence's window is its next token plus its
/// drafts): when `drafts` (per sequence) come to more, drafts are dropped from the least likely
/// up, likelihood being the product of the drafter's probabilities through the draft. A sequence
/// always keeps its first row and its rows stay a prefix; ties drop the deeper draft, then the
/// later sequence. At or under the budget (and with `max_rows` 0) nothing changes.
pub(crate) fn budget(probs: &[Vec<f32>], drafts: &[usize], max_rows: usize) -> Vec<usize> {
    let rows: usize = drafts.iter().map(|k| k + 1).sum();
    if max_rows == 0 || rows <= max_rows {
        return drafts.to_vec();
    }
    // Every draft verified: (P(kept through it), depth, sequence).
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (i, &k) in drafts.iter().enumerate() {
        let mut kept = 1.0f64;
        for j in 0..k {
            kept *= likelihood(probs.get(i).and_then(|p| p.get(j)).copied().unwrap_or(0.0));
            candidates.push((kept, j + 1, i));
        }
    }
    // The least likely first. A sequence's own drafts come deepest first (the product never grows
    // with depth), so each drop is the last row of its window.
    candidates.sort_by(|x, y| x.0.total_cmp(&y.0).then(y.1.cmp(&x.1)).then(y.2.cmp(&x.2)));
    let mut out = drafts.to_vec();
    let mut over = rows - max_rows.max(drafts.len());
    for (_, depth, i) in candidates {
        if over == 0 {
            break;
        }
        if depth == out[i] {
            out[i] -= 1;
            over -= 1;
        }
    }
    out
}

/// The probabilities of a copy-window proposal: where it matches the drafter's own `drafted`
/// tokens, the drafter's `probs`; past the first mismatch (or without a draft), [`COPY_TOKEN_P`].
pub(crate) fn copy_probs(copy: &[u32], drafted: &[u32], probs: &[f32]) -> Vec<f32> {
    let agree = copy.iter().zip(drafted).take_while(|(c, d)| c == d).count();
    copy.iter().enumerate().map(|(j, _)| if j < agree { probs.get(j).copied().unwrap_or(COPY_TOKEN_P) } else {
        COPY_TOKEN_P }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_cuts_where_the_product_drops_below_tau() {
        let p = [0.9f32, 0.8, 0.5, 0.9];
        // 0.9, 0.72, 0.36, 0.324: at tau 0.3 all four; at 0.4 the third falls below.
        assert_eq!(chain_length(&p, 7, 0.3), 4);
        assert_eq!(chain_length(&p, 7, 0.4), 2);
        assert_eq!(chain_length(&p, 1, 0.3), 1, "the cap binds");
        assert_eq!(chain_length(&[], 7, 0.3), 0);
        // The default keeps the first two (0.9, 0.72).
        assert_eq!(chain_length(&p, 7, DEFAULT_TAU), 2);
        // A first draft below tau verifies none; out-of-range probabilities clamp.
        assert_eq!(chain_length(&[0.5, 1.0], 7, DEFAULT_TAU), 0);
        assert_eq!(chain_length(&[1.5, -0.2], 7, DEFAULT_TAU), 1);
        assert_eq!(chain_length(&[f32::NAN, 0.9], 7, DEFAULT_TAU), 0);
    }

    #[test]
    fn a_product_exactly_at_tau_is_kept_and_every_sequence_at_tau_cuts_alike() {
        // The cut is strict: a product equal to tau keeps its draft (exact binary fractions).
        assert_eq!(chain_length(&[0.75, 1.0, 0.5], 7, 0.75), 2);
        assert_eq!(chain_length(&[0.5, 0.5], 7, 0.25), 2);
        // An f32 0.7 sits just below the f64 default.
        assert_eq!(chain_length(&[0.7], 7, DEFAULT_TAU), 0);
        // Sixteen sequences whose second draft lands on tau (0.875 x f32 0.8 = 0.70000001, then
        // x 0.99 below): each verifies two, and the 48 rows are left as the chains cut them.
        let at_tau = vec![0.875f32, 0.8, 0.99, 0.99];
        let probs = vec![at_tau; 16];
        let ks: Vec<usize> = probs.iter().map(|p| chain_length(p, 7, DEFAULT_TAU)).collect();
        assert_eq!(ks, vec![2; 16]);
        assert_eq!(budget(&probs, &ks, 64), ks);
        // Held to 40 rows, the eight equally likely second drafts go from the last sequence back.
        assert_eq!(budget(&probs, &ks, 40), (0..16).map(|i| if i < 8 { 2 } else { 1 }).collect::<Vec<_>>());
    }

    #[test]
    fn the_budget_keeps_the_most_likely_drafts() {
        let hi = vec![0.99f32, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93];
        let mid = vec![0.9f32; 7];
        let lo = vec![0.8f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let probs = [hi.clone(), mid.clone(), lo.clone()];
        let ks = [7usize, 7, 7];
        // At or under the budget, and without one, nothing changes.
        assert_eq!(budget(&probs, &ks, 24), ks.to_vec());
        assert_eq!(budget(&probs, &ks, 100), ks.to_vec());
        assert_eq!(budget(&probs, &ks, 0), ks.to_vec());
        // One row over: the least likely draft goes (lo's 7th, 0.8 x 0.5^6).
        assert_eq!(budget(&probs, &ks, 23), vec![7, 7, 6]);
        // lo's drafts after its first are the least likely, then mid's from the back, while hi
        // keeps all seven.
        assert_eq!(budget(&probs, &ks, 18), vec![7, 7, 1]);
        assert_eq!(budget(&probs, &ks, 16), vec![7, 5, 1]);
        // Down to the anchors: every sequence keeps its first row, however small the budget.
        assert_eq!(budget(&probs, &ks, 3), vec![0, 0, 0]);
        assert_eq!(budget(&probs, &ks, 1), vec![0, 0, 0]);
        // Each sequence's rows stay a prefix, and the lengths never grow.
        for max in 1..=24 {
            let b = budget(&probs, &ks, max);
            let rows: usize = b.iter().map(|k| k + 1).sum();
            assert!(rows <= max.max(3) && b.iter().zip(&ks).all(|(x, y)| x <= y), "{max}: {b:?}");
        }
        // The chain cut, then the budget: a light step is the chain's own.
        let chain: Vec<usize> = probs.iter().map(|p| chain_length(p, 7, DEFAULT_TAU)).collect();
        assert_eq!(chain, vec![7, 3, 1]);
        assert_eq!(budget(&probs, &chain, 64), chain);
        // 14 rows into 8: mid's 3rd (0.729), hi's 7th (0.750), lo's 1st (0.800), hi's 6th
        // (0.807), mid's 2nd (0.810) and hi's 5th (0.858) go.
        assert_eq!(budget(&probs, &chain, 8), vec![4, 1, 0]);
    }

    #[test]
    fn ties_drop_the_deeper_draft_then_the_later_sequence() {
        let mid = vec![0.9f32; 7];
        assert_eq!(budget(&[mid.clone(), mid.clone()], &[2, 2], 5), vec![2, 1]);
        assert_eq!(budget(&[vec![1.0; 3], vec![1.0; 3]], &[3, 3], 6), vec![2, 2]);
        assert_eq!(budget(&[vec![1.0; 3], vec![1.0; 3]], &[3, 3], 7), vec![3, 2]);
    }

    #[test]
    fn a_single_sequence_is_held_to_the_budget_alone() {
        let p = vec![0.95f32; 7];
        assert_eq!(budget(std::slice::from_ref(&p), &[7], 64), vec![7]);
        assert_eq!(budget(std::slice::from_ref(&p), &[7], 5), vec![4]);
        assert_eq!(budget(std::slice::from_ref(&p), &[7], 1), vec![0]);
    }

    #[test]
    fn sixteen_sequences_spend_sixty_four_rows_on_the_likeliest_drafts() {
        // Half the sequences confident, half doubtful, seven drafts each after the chain cut at
        // tau 0: 128 rows. The budget keeps 64: the doubtful ones fall to their anchors first.
        let sure = vec![0.98f32; 7];
        let doubt = vec![0.6f32; 7];
        let probs: Vec<Vec<f32>> = (0..16).map(|i| if i % 2 == 0 { sure.clone() } else { doubt.clone() }).collect();
        let ks = budget(&probs, &[7; 16], 64);
        assert_eq!(ks.iter().map(|k| k + 1).sum::<usize>(), 64);
        assert!(ks.iter().enumerate().all(|(i, &k)| if i % 2 == 0 { k >= 3 } else { k <= 1 }), "{ks:?}");
        // The even split it replaces: room = 64 / 16 - 1 = 3 drafts each, whatever the likelihood.
        let even: f64 = probs.iter().map(|p| p[..3].iter().scan(1.0f64, |q, &x| {
            *q *= f64::from(x);
            Some(*q)
        }).sum::<f64>()).sum();
        let ranked: f64 = probs.iter().zip(&ks).map(|(p, &k)| p[..k].iter().scan(1.0f64, |q, &x| {
            *q *= f64::from(x);
            Some(*q)
        }).sum::<f64>()).sum();
        assert!(ranked > even, "expected kept drafts: ranked {ranked:.2} against even {even:.2}");
    }

    #[test]
    fn room_is_the_even_share_under_cost_and_the_whole_budget_under_chain() {
        // 64 rows: 3 drafts each at 16 sequences, 63 for one; 127 rows (an RTX 5090's whole sparse
        // MLA waves with --decode-rows 128): 6 each at 16.
        assert_eq!([1, 2, 9, 16, 64, 65].map(|s| VerifyPolicy::Cost.room(64, s)), [63, 31, 6, 3, 0, 0]);
        assert_eq!([1, 16, 17].map(|s| VerifyPolicy::Cost.room(127, s)), [126, 6, 6]);
        assert_eq!([1, 16, 64].map(|s| VerifyPolicy::Chain.room(64, s)), [63; 3]);
        assert_eq!([1, 16].map(|s| VerifyPolicy::Chain.room(127, s)), [126; 2]);
    }

    #[test]
    fn a_wide_step_is_held_to_its_verify_rows() {
        // Sixteen sequences with seven equally likely drafts: 128 rows, one over 127 (the last
        // sequence's last draft goes), 64 over 64.
        let probs = vec![vec![0.9f32; 7]; 16];
        let wide = budget(&probs, &[7; 16], 127);
        assert_eq!(wide, (0..16).map(|i| if i == 15 { 6 } else { 7 }).collect::<Vec<_>>());
        let narrow = budget(&probs, &[7; 16], 64);
        assert_eq!(narrow.iter().map(|k| k + 1).sum::<usize>(), 64);
        assert!(narrow.iter().all(|&k| k == 3), "{narrow:?}");
    }

    #[test]
    fn copied_tokens_keep_the_drafters_probability_where_it_agrees() {
        let probs = [0.9f32, 0.8, 0.7];
        assert_eq!(copy_probs(&[5, 6, 7, 8], &[5, 6, 9], &probs), vec![0.9, 0.8, COPY_TOKEN_P, COPY_TOKEN_P]);
        assert_eq!(copy_probs(&[5, 6], &[], &[]), vec![COPY_TOKEN_P; 2]);
        assert_eq!(copy_probs(&[], &[1, 2], &probs), Vec::<f32>::new());
    }
}
