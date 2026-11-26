//! Greedy verification of an emitted anchor and a lane-admitted draft prefix.
//! Neural width is independent of the target verifier row budget.
pub const MAX_DSPARK_PROPOSALS: usize = 7;

/// Total target rows per V4.1 verifier lane, including all request anchors.
pub const V41_VERIFY_ROWS: usize = 64;
/// Compiled target decode covers; Python/CMake exports read V41_VERIFY_ROWS.
pub const V41_DECODE_CAPACITIES: [usize; 3] = [1, 16, V41_VERIFY_ROWS];
/// Legacy drafter cover for 16 request sequences flattened at width five.
/// This is not a target verifier capacity, even when a lane has fewer requests.
pub const V41_DRAFT_LEGACY_ROWS: usize = 80;
/// Thirty-two lane members at width seven produce 224 rows, covered by 256.
pub const V41_DRAFT_WIDE_ROWS: usize = 256;
pub const V41_TARGET_CAPACITIES: [u64; 6] = [1, 16, V41_VERIFY_ROWS as u64, 256, 1024, 4096];

/// Persistent request identities are shared by the two decode lanes.
pub const V41_MAX_REQUESTS: usize = crate::concurrency::MAX_CONCURRENCY as usize;
/// Fixed native drafter request capacity; target lanes can admit more anchors.
pub const V41_MAX_DRAFT_MEMBERS: usize = V41_MAX_REQUESTS.div_ceil(2);
/// Total request capacity divided across the resolved target decode lanes.
pub const fn v41_lane_members(lanes: usize) -> usize { V41_MAX_REQUESTS.div_ceil(lanes) }
/// Physical draft request capacity, independent of target membership.
pub const fn v41_draft_members(members: usize, width: usize) -> usize {
    let cap = V41_DRAFT_WIDE_ROWS / width;
    let cap = if cap < V41_MAX_DRAFT_MEMBERS { cap } else { V41_MAX_DRAFT_MEMBERS };
    if members < cap { members } else { cap }
}
/// Vocabulary heads cover the compiled target verify-row budget (256).
pub const V41_VOCAB_ROWS: usize = V41_MAX_DRAFT_MEMBERS * (MAX_DSPARK_PROPOSALS + 1);
pub const V41_WRITE_DESCRIPTOR_BYTES: usize = V41_MAX_REQUESTS * 24;
pub const V41_READ_DESCRIPTOR_BYTES: usize = V41_MAX_REQUESTS * 8;

#[derive(Debug, PartialEq, Eq)]
pub struct GreedyVerification {
    /// Input rows to publish, including the anchor. A correction/bonus token
    /// has not been evaluated as input and must remain the next pending anchor.
    pub accepted_inputs: u32,
    /// Newly generated tokens; excludes the previously emitted anchor.
    pub emitted: Vec<u32>,
    pub eos: bool,
    pub length_limit: bool,
}

/// `inputs` is anchor followed by drafts. `target_next[i]` is the greedy target
/// prediction after input row i, using causal verification of the same prefix.
/// The anchor must already have been emitted, and must not be EOS.
pub fn verify_dspark_greedy(
    inputs: &[u32],
    target_next: &[u32],
    eos: u32,
    remaining: usize,
) -> Result<GreedyVerification, &'static str> {
    verify_dspark_greedy_vocab(inputs, target_next, Some(eos), remaining, 129280)
}

/// Vocabulary-parametric greedy verification for checkpoint-defined targets.
/// `None` disables EOS stopping for fixed-length numerical qualification.
pub fn verify_dspark_greedy_vocab(inputs: &[u32], target_next: &[u32], eos: Option<u32>,
    remaining: usize, vocab: usize) -> Result<GreedyVerification, &'static str> {
    if inputs.is_empty() || inputs.len() != target_next.len() {
        return Err("verification requires nonempty corresponding target rows");
    }
    if remaining == 0 || eos == Some(inputs[0]) {
        return Err("completed request cannot verify another anchor");
    }
    if vocab == 0 || eos.is_some_and(|id| id as usize >= vocab)
        || inputs.iter().chain(target_next).any(|&id| id as usize >= vocab) {
        return Err("verification token outside target vocabulary");
    }
    let mut result = GreedyVerification {
        accepted_inputs: 1,
        emitted: Vec::with_capacity(inputs.len()),
        eos: false,
        length_limit: false,
    };
    for (i, &token) in target_next.iter().enumerate() {
        let matched = inputs.get(i + 1) == Some(&token);
        result.emitted.push(token);
        if matched {
            result.accepted_inputs += 1;
        }
        result.eos = eos == Some(token);
        result.length_limit = result.emitted.len() == remaining;
        if !matched || result.eos || result.length_limit {
            break;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_vocabulary_supports_large_ids_without_changing_legacy_results() {
        let result = verify_dspark_greedy_vocab(&[248077, 200001], &[200001, 200002], None, 8, 248320).unwrap();
        assert_eq!(result.accepted_inputs, 2);
        assert_eq!(result.emitted, [200001, 200002]);
        assert!(!result.eos);
        for mismatch in 0..3 {
            let inputs = [10, 11, 12];
            let mut next = [11, 12, 13];
            next[mismatch] = 99;
            assert_eq!(verify_dspark_greedy(&inputs, &next, 1, 8),
                verify_dspark_greedy_vocab(&inputs, &next, Some(1), 8, 129280));
        }
        assert!(verify_dspark_greedy_vocab(&[248320], &[0], None, 1, 248320).is_err());
        assert!(verify_dspark_greedy_vocab(&[0], &[0], Some(248320), 1, 248320).is_err());
        assert!(verify_dspark_greedy_vocab(&[0], &[0], None, 1, 0).is_err());
    }

    #[test]
    fn seven_drafts_correct_at_every_position_and_stop() {
        let inputs = [10, 11, 12, 13, 14, 15, 16, 17];
        let full = [11, 12, 13, 14, 15, 16, 17, 18];
        for mismatch in 0..8 {
            let mut target = full;
            target[mismatch] = 99;
            let result = verify_dspark_greedy(&inputs, &target, 1, 100).unwrap();
            assert_eq!(result.accepted_inputs, mismatch as u32 + 1);
            assert_eq!(result.emitted, target[..=mismatch]);
        }
        let bonus = verify_dspark_greedy(&inputs, &full, 1, 100).unwrap();
        assert_eq!(bonus.accepted_inputs, 8);
        assert_eq!(bonus.emitted, full);
        for remaining in 1..=8 {
            let result = verify_dspark_greedy(&inputs, &full, 1, remaining).unwrap();
            assert_eq!(result.emitted, full[..remaining]);
            assert_eq!(result.accepted_inputs as usize, (remaining + 1).min(8));
            assert!(result.length_limit);
        }
        for eos_position in 0..8 {
            let mut target = full;
            target[eos_position] = 1;
            let result = verify_dspark_greedy(&inputs, &target, 1, 100).unwrap();
            assert_eq!(result.emitted, target[..=eos_position]);
            assert!(result.eos);
        }
        assert!(verify_dspark_greedy(&[10; 9], &[11; 9], 1, 100).is_ok());
    }
    #[test]
    fn long_copy_tails_correct_at_every_position() {
        let inputs: Vec<_> = (10..48).collect();
        let full: Vec<_> = (11..49).collect();
        for mismatch in 0..inputs.len() {
            let mut target = full.clone();
            target[mismatch] = 99;
            let result = verify_dspark_greedy(&inputs, &target, 1, 100).unwrap();
            assert_eq!(result.accepted_inputs as usize, mismatch + 1);
            assert_eq!(result.emitted, target[..=mismatch]);
        }
        let result = verify_dspark_greedy(&inputs, &full, 1, 100).unwrap();
        assert_eq!(result.accepted_inputs as usize, inputs.len());
        assert_eq!(result.emitted, full);
    }

    #[test]
    fn greedy_prefix_correction_bonus_and_stopping() {
        let inputs = [10, 11, 12, 13, 14, 15];
        for mismatch in 0..6 {
            let mut target = [11, 12, 13, 14, 15, 16];
            target[mismatch] = 99;
            let result = verify_dspark_greedy(&inputs, &target, 1, 100).unwrap();
            assert_eq!(result.accepted_inputs, mismatch as u32 + 1);
            assert_eq!(result.emitted, target[..=mismatch]);
            assert!(!result.eos && !result.length_limit);
        }
        let result = verify_dspark_greedy(&inputs, &[11, 12, 13, 14, 15, 16], 1, 100).unwrap();
        assert_eq!(result.accepted_inputs, 6);
        assert_eq!(result.emitted, [11, 12, 13, 14, 15, 16]);
        let stopped = verify_dspark_greedy(&inputs, &[11, 12, 13, 14, 15, 16], 1, 2).unwrap();
        assert_eq!(stopped.accepted_inputs, 3);
        assert_eq!(stopped.emitted, [11, 12]);
        assert!(stopped.length_limit);
        for draft in [1, 12] {
            let stopped = verify_dspark_greedy(&[10, draft, 13], &[1, 13, 14], 1, 100).unwrap();
            assert_eq!(stopped.emitted, [1]);
            assert!(stopped.eos);
            assert_eq!(stopped.accepted_inputs, if draft == 1 { 2 } else { 1 });
        }
        assert!(verify_dspark_greedy(&[1], &[10], 1, 1).is_err());
        assert!(verify_dspark_greedy(&[10], &[11], 1, 0).is_err());
        assert!(verify_dspark_greedy(&[10], &[], 1, 1).is_err());
        assert!(verify_dspark_greedy(&[10], &[129280], 1, 1).is_err());
    }
}
