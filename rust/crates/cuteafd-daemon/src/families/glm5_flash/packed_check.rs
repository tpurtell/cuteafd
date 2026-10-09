//! glmf-golden --packed-check N: a packed prefill against each sequence's own pass. N sequences
//! of mixed lengths, cut from the golden prompt (one resumed from a prefilled prefix, so its chunk
//! starts mid-sequence; one past the dense context, so its pool top-k runs), prefill in one
//! packed pass (`GlmfEngine::prefill_packed`) and, on fresh units and slots of the same engine,
//! each in its own pass (`prefill_device`, the path a lone prompt takes). Each sequence's
//! last-row logits, its KDA state (with everything `slot_state` carries) and every paged byte of
//! its units must agree bit for bit; the verdict, the largest logit difference and both arms'
//! times are printed per sequence.
use super::engine::{Allocator, GlmfEngine, GlmfPlacement};
use super::lane_check::unit_bytes;
use anyhow::{ensure, Context, Result};

/// Chunk lengths of the check's sequences, in order: short prompts under and over the 64-row
/// chunked-KDA cut, the 160-row router GEMV and the 384-row mHC TF32 cut, a page and a pool page
/// apart, a single row, the longest chunk two Spark lanes leave in one pass, and a resumed one.
const LENGTHS: [usize; 10] = [29, 40, 7, 64, 65, 255, 1, 384, 511, 100];
/// The resumed sequence (the last): this many tokens prefilled alone first in both arms, so its
/// chunk starts mid-page past the 2,051-token dense context and its pool top-k runs.
const RESUMED: (usize, usize) = (9, 2100);

/// Sequence `i`'s tokens: `len` of the golden prompt from a per-sequence offset, wrapping.
fn sequence(tokens: &[u32], i: usize, len: usize) -> Vec<u32> {
    let offset = (i * 97) % tokens.len();
    (0..len).map(|j| tokens[(offset + j) % tokens.len()]).collect()
}

pub(crate) fn packed_check(engine: &GlmfEngine<'_>, tokens: &[u32], sequences: usize) -> Result<()> {
    ensure!(engine.packs_prefill(), "--packed-check needs one GPU and every layer");
    ensure!(!tokens.is_empty() && (2..=LENGTHS.len()).contains(&sequences),
        "--packed-check takes 2 to {} sequences of a non-empty golden prompt", LENGTHS.len());
    let limits = engine.packed_limits();
    // The longest prefix of the lengths that fits one pass (the drafter's taps included); the
    // resumed sequence is always the last.
    let mut lengths: Vec<usize> = LENGTHS[..sequences - 1].to_vec();
    lengths.push(LENGTHS[RESUMED.0]);
    let resumed = lengths.len() - 1;
    let prefix = |i: usize| if i == resumed { RESUMED.1 } else { 0 };
    let chunks: Vec<(usize, usize)> = lengths.iter().enumerate().map(|(i, &len)| (prefix(i), len)).collect();
    ensure!(super::packing::fits(&chunks, &limits), "--packed-check's {} sequences ({chunks:?}) do not fit one pass \
        ({limits:?}): take fewer", lengths.len());
    let n = lengths.len();
    let prompts: Vec<Vec<u32>> = lengths.iter().enumerate().map(|(i, &len)| sequence(tokens, i, prefix(i) + len))
        .collect();
    let mut allocator = Allocator::new(engine.pages, engine.slots);
    let admit = |allocator: &mut Allocator, prompt: &[u32]| allocator.admit(prompt.len() + 1);
    let mut packed: Vec<GlmfPlacement> = prompts.iter().map(|p| admit(&mut allocator, p)).collect::<Result<_>>()?;
    let mut alone: Vec<GlmfPlacement> = prompts.iter().map(|p| admit(&mut allocator, p)).collect::<Result<_>>()?;
    // The resumed sequence's prefix, alone in both arms (in chunks of the engine's capacity).
    for placement in [&mut packed[resumed], &mut alone[resumed]] {
        for part in prompts[resumed][..RESUMED.1].chunks(engine.prefill_capacity()) {
            engine.prefill_device(placement, part)?;
        }
    }
    engine.synchronize()?;
    let chunk = |i: usize| &prompts[i][prompts[i].len() - lengths[i]..];
    // Arm A: one packed pass.
    let started = std::time::Instant::now();
    let mut packed_logits = vec![Vec::new(); n];
    {
        let mut segments: Vec<(&mut GlmfPlacement, &[u32])> = packed.iter_mut().enumerate()
            .map(|(i, placement)| (placement, chunk(i))).collect();
        engine.prefill_packed(&mut segments, &mut |i, logits| {
            packed_logits[i] = logits.row_host(engine.library, 0)?;
            Ok(Ok(()))
        })?;
    }
    engine.synchronize()?;
    let packed_ms = 1e3 * started.elapsed().as_secs_f64();
    // Arm B: each sequence's own pass.
    let mut alone_logits = Vec::with_capacity(n);
    let mut alone_ms = Vec::with_capacity(n);
    for (i, placement) in alone.iter_mut().enumerate() {
        let started = std::time::Instant::now();
        let logits = engine.prefill_device(placement, chunk(i))?.context("--packed-check needs the head")?;
        alone_logits.push(logits.row_host(engine.library, logits.rows - 1)?);
        alone_ms.push(1e3 * started.elapsed().as_secs_f64());
    }
    let verdict = |same: bool| if same { "identical" } else { "DIFFER" };
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    let mut all = true;
    for i in 0..n {
        let (a, b) = (&packed_logits[i], &alone_logits[i]);
        let logits = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let state = engine.slot_state(packed[i].slot)? == engine.slot_state(alone[i].slot)?;
        let paged = unit_bytes(engine, &packed[i])? == unit_bytes(engine, &alone[i])?;
        all &= logits && state && paged;
        println!("packed check: sequence {i}: {} rows at position {}: logits {} (max |diff| {worst:.3e}, argmax {} / {}), \
            KDA state {}, paged units {} (alone {:.1} ms)", lengths[i], prompts[i].len() - lengths[i], verdict(logits),
            argmax(a), argmax(b), verdict(state), verdict(paged), alone_ms[i]);
    }
    println!("packed check: {n} sequences, {} rows: one packed pass {packed_ms:.1} ms, their own passes {:.1} ms; {}",
        lengths.iter().sum::<usize>(), alone_ms.iter().sum::<f64>(),
        if all { "every sequence identical" } else { "a sequence DIFFERS" });
    for placement in packed.into_iter().chain(alone) {
        allocator.release(placement);
    }
    ensure!(all, "a packed sequence differs from its own pass");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences_wrap_the_golden_prompt_from_their_own_offsets() {
        let tokens: Vec<u32> = (0..100).collect();
        assert_eq!(sequence(&tokens, 0, 3), [0, 1, 2]);
        assert_eq!(sequence(&tokens, 1, 5), [97, 98, 99, 0, 1]);
        assert_eq!(sequence(&tokens, 2, 2), [94, 95]);
        assert_eq!(sequence(&tokens, 3, 300).len(), 300);
    }

    #[test]
    fn the_default_lengths_cross_every_row_cut_and_fit_one_pass() {
        use super::super::packing::{fits, plan, Limits};
        // Two Spark lanes of 4,096 rows with a drafter: chunks up to 511 rows, 2,048 tap rows.
        let limits = Limits { rows: 4096, chunk_rows: 511, table_pages: 2048, table_pool_pages: 512,
            taps: Some(2048), dense_context: 2051, max_context: 131_072 };
        let chunks: Vec<(usize, usize)> = LENGTHS.iter().enumerate().map(|(i, &len)|
            (if i == RESUMED.0 { RESUMED.1 } else { 0 }, len)).collect();
        // Every row count a per-sequence program switches at (KDA 64, router 160, mHC 384), from
        // each side, and a single row.
        for cut in [64, 160, 384] {
            assert!(LENGTHS.iter().any(|&l| l <= cut) && LENGTHS.iter().any(|&l| l > cut), "{cut}");
        }
        assert!(LENGTHS.contains(&1));
        assert!(fits(&chunks, &limits));
        // The resumed sequence alone runs the pool top-k, from mid-page.
        let segments = plan(&chunks, &limits).unwrap();
        assert_eq!(segments.iter().map(|s| s.long).collect::<Vec<_>>(), [false; 9].into_iter().chain([true])
            .collect::<Vec<_>>());
        assert_eq!((segments[9].start % 64, segments[9].first), (52, 1356));
    }
}
