//! Speculation checks for GLM 5.3 Flash (glmf-golden): the DFlash2 drafter
//! against the torch reference, drafts against the target, KDA
//! verify-by-replay against serial steps, and the verify-step cost by rows.
use super::engine::{Allocator, GlmfEngine, GlmfPlacement};
use super::{bf16s, similarity, GoldenArgs, Opened};
use crate::families::glm5::dflash::{ContextRow, DraftSeq, TAP_ROWS};
use anyhow::{ensure, Context, Result};
use std::time::Instant;

fn tokens(args: &GoldenArgs) -> Result<Vec<u32>> {
    Ok(std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

/// BF16 mean of the four mHC streams of `rows` rows of a golden layer
/// ([T, 4, hidden]): FP32 sum in stream order, then the division, as the
/// engine's tap kernel.
fn stream_mean(layer: &[u8], first: usize, rows: usize, hidden: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * hidden * 2);
    let value = |i: usize| f32::from_bits(u32::from(u16::from_le_bytes([layer[2 * i], layer[2 * i + 1]])) << 16);
    for r in first..first + rows {
        for c in 0..hidden {
            let mut sum = 0f32;
            for k in 0..4 {
                sum += value((r * 4 + k) * hidden + c);
            }
            // Round to nearest even (the tap kernel's __float2bfloat16; no NaNs here).
            let bits = (sum / 4.0).to_bits();
            out.extend_from_slice(&(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_le_bytes());
        }
    }
    out
}

/// Prefills `tokens` in prefill-row chunks, feeding each chunk's tapped tail to
/// the drafter's ring `slot`; returns the last row's logits.
fn prefill_with_taps(engine: &GlmfEngine<'_>, placement: &mut GlmfPlacement, tokens: &[u32],
    slot: usize) -> Result<Vec<f32>> {
    let mut logits = None;
    for chunk in tokens.chunks(engine.prefill_capacity()) {
        let start = placement.len;
        logits = engine.prefill(placement, chunk, None)?;
        if let Some(drafter) = &engine.drafter {
            let n = chunk.len().min(TAP_ROWS);
            let first = start + chunk.len() - n;
            drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot, position: first + r }).collect::<Vec<_>>())?;
        }
    }
    logits.context("the prefill needs every layer")
}

/// Runs the drafter alone on the golden taps at reference.py's anchor
/// positions and compares tokens, selector features and final-norm rows.
pub(super) fn draft_oracle(args: &GoldenArgs, opened: &Opened, engine: &GlmfEngine<'_>, dir: &std::path::Path)
    -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-oracle needs --draft")?;
    let (hidden, block, drafts_per) = (opened.cfg.hidden, drafter.block(), drafter.drafts());
    let dspark = matches!(drafter, super::dspark::Drafter::Dspark(_));
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let positions: Vec<usize> = meta["positions"].as_array().context("positions")?.iter()
        .map(|p| p.as_u64().map(|p| p as usize).context("position")).collect::<Result<_>>()?;
    let words = |name: &str| -> Result<Vec<u32>> {
        Ok(std::fs::read(dir.join(name))?.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let ref_tokens = words("drafts.bin")?;
    // DFlash2: selector features [N, 7, 4]; dSpark: confidence [N, 8].
    let ref_features = words(if dspark { "confidence.bin" } else { "features.bin" })?;
    let ref_hidden = bf16s(&std::fs::read(dir.join("hidden.bin"))?);
    let tokens = tokens(args)?;
    let layers: Vec<Vec<u8>> = drafter.taps().iter()
        .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
    let row = hidden * 2;
    let width = layers.len() * row;
    let (mut done, mut exact, mut first, mut matched, mut worst, mut feature_error) = (0usize, 0, 0, 0, 1f64, 0f64);
    let mut draft_seconds = 0f64;
    for (index, &position) in positions.iter().enumerate() {
        while done < position {
            let n = (position - done).min(TAP_ROWS);
            let means: Vec<Vec<u8>> = layers.iter().map(|layer| stream_mean(layer, done, n, hidden)).collect();
            let mut taps = vec![0u8; n * width];
            for r in 0..n {
                for (i, mean) in means.iter().enumerate() {
                    taps[r * width + i * row..][..row].copy_from_slice(&mean[r * row..][..row]);
                }
            }
            drafter.put_taps(&taps)?;
            drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: done + r }).collect::<Vec<_>>())?;
            // SAFETY: the oracle owns the stream; a following chunk reuses
            // the taps and context metadata read by this update.
            unsafe { engine.library.cuda_stream_synchronize(engine.stream)? };
            done += n;
        }
        let anchor = tokens[position];
        let timer = Instant::now();
        let draft = drafter.draft_device(&[DraftSeq { slot: 0, anchor, position, valid_from: 0 }],
            &engine.embedding, engine.draft_head())?.remove(0);
        draft_seconds += timer.elapsed().as_secs_f64();
        let reference = &ref_tokens[index * drafts_per..][..drafts_per];
        exact += usize::from(draft.tokens == reference);
        first += usize::from(draft.tokens[0] == reference[0]);
        matched += draft.tokens.iter().zip(reference).take_while(|(a, b)| a == b).count();
        let (cosine, _) = similarity(&bf16s(&drafter.last_hidden(1)?), &ref_hidden[index * block * hidden..][..block * hidden]);
        worst = worst.min(cosine);
        if dspark {
            // Confidence of the rows whose previous token agrees (the Markov embedding is the same).
            for k in 0..drafts_per {
                if k > 0 && draft.tokens[k - 1] != reference[k - 1] {
                    break;
                }
                let theirs = f32::from_bits(ref_features[index * drafts_per + k]);
                feature_error = feature_error.max(f64::from((draft.confidence[k] - theirs).abs()));
            }
        } else if draft.tokens[0] == reference[0] {
            let theirs = f32::from_bits(ref_features[index * drafts_per * 4]);
            feature_error = feature_error.max(f64::from((draft.features[0][0] - theirs).abs()));
        }
        if draft.tokens != reference {
            println!("position {position}: engine {:?} reference {reference:?}", draft.tokens);
        }
    }
    let n = positions.len();
    println!("draft oracle ({}): {n} anchors, identical drafts {exact}/{n}, first draft {first}/{n}, matching prefix \
        {:.2} of {drafts_per}, worst final-norm cosine {worst:.6}, {} max error {feature_error:.4}, {:.2} ms/draft",
        drafter.name(), matched as f64 / n as f64, if dspark { "confidence" } else { "first-margin" },
        draft_seconds * 1e3 / n as f64);
    Ok(())
}

/// [`crate::families::glm5::dflash::replay`] on the golden taps (mHC stream means).
pub(super) fn draft_replay(args: &GoldenArgs, opened: &Opened, engine: &GlmfEngine<'_>, start: usize) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-replay needs --draft")?;
    let (tokens, greedy) = crate::families::glm5::dflash::golden_sequence(&args.golden, opened.cfg.vocab_size)?;
    let hidden = opened.cfg.hidden;
    let layers: Vec<Vec<u8>> = drafter.taps().iter()
        .map(|l| std::fs::read(args.golden.join(format!("layer{l:02}.bin")))).collect::<std::io::Result<_>>()?;
    let row = hidden * 2;
    let taps = |first: usize, n: usize| -> Result<Vec<u8>> {
        let means: Vec<Vec<u8>> = layers.iter().map(|layer| stream_mean(layer, first, n, hidden)).collect();
        let mut taps = vec![0u8; n * layers.len() * row];
        for r in 0..n {
            for (i, mean) in means.iter().enumerate() {
                taps[(r * layers.len() + i) * row..][..row].copy_from_slice(&mean[r * row..][..row]);
            }
        }
        Ok(taps)
    };
    crate::families::glm5::dflash::replay(drafter.replay(), &tokens, &greedy, &taps, &|t| engine.embedding.host_rows(t),
        engine.weights.head.bf16().context("--draft-replay borrows the target BF16 head; the FP8-only head \
            (--fp8-head) has no BF16 copy to replay against")?.buffer.ptr, start)
}

/// Prefills the golden prompt's first --prefill tokens, then decodes one row
/// per step (teacher-forced on tokens.bin, or greedy with --generate),
/// drafting with DFlash2 before every step; reports the accepted prefix per
/// step against the sequence.
pub(super) fn draft_run(args: &GoldenArgs, opened: &Opened, engine: &GlmfEngine<'_>) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft")?;
    let mut sequence = tokens(args)?;
    let prefill = args.prefill.unwrap_or(sequence.len() / 2).min(sequence.len() - 1);
    let end = match args.generate {
        Some(n) if n > 0 => {
            sequence.truncate(prefill);
            prefill + n
        }
        _ => sequence.len(),
    };
    let greedy = end > sequence.len();
    let mut placement = Allocator::new(engine.pages, engine.slots).admit(end + 1)?;
    let started = Instant::now();
    let logits = prefill_with_taps(engine, &mut placement, &sequence[..prefill], 0)?;
    if greedy {
        sequence.push(argmax(&logits));
    }
    println!("prefill: {prefill} tokens in {:.2} s", started.elapsed().as_secs_f64());
    let (mut drafts, mut draft_seconds) = (Vec::new(), 0f64);
    let started = Instant::now();
    for position in prefill..end {
        let anchor = sequence[position];
        let timer = Instant::now();
        let draft = drafter.draft_device(&[DraftSeq { slot: 0, anchor, position, valid_from: 0 }], &engine.embedding,
            engine.draft_head())?;
        draft_seconds += timer.elapsed().as_secs_f64();
        drafts.push((position, draft.into_iter().next().context("draft")?));
        let logits = engine.verify(&mut [(&mut placement, 1)], &[anchor], None)?.context("decode needs every layer")?;
        drafter.update(&[ContextRow { tap_row: 0, slot: 0, position }])?;
        if greedy {
            sequence.push(argmax(&logits));
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let block = drafter.drafts();
    let mut histogram = vec![0usize; block + 1];
    let mut first_rank = [0usize; 16];
    for (position, draft) in &drafts {
        let truth = &sequence[position + 1..];
        if truth.len() < block {
            continue;
        }
        let accepted = draft.tokens.iter().zip(truth).take_while(|(d, t)| d == t).count();
        histogram[accepted] += 1;
        first_rank[draft.features[0][3] as usize] += 1;
    }
    let steps: usize = histogram.iter().sum();
    let accepted: usize = histogram.iter().enumerate().map(|(i, c)| i * c).sum();
    println!("drafts: {steps} steps, {accepted} drafted tokens accepted as a prefix ({:.2} per step, {:.1}% of {}), \
        histogram {histogram:?}, first-draft selector rank {first_rank:?}, {:.2} ms/draft, {:.1} ms/step",
        accepted as f64 / steps.max(1) as f64, 100.0 * accepted as f64 / (steps * block).max(1) as f64, steps * block,
        draft_seconds * 1e3 / drafts.len().max(1) as f64, seconds * 1e3 / drafts.len().max(1) as f64);
    if greedy {
        let text = cuteafd_loader::LoadedTokenizer::from_snapshot(&opened.checkpoint.snapshot)?
            .decode_ids(&sequence[prefill..], false).map(|d| d.text).unwrap_or_default();
        println!("generated: {text:?}");
    }
    Ok(())
}

/// Element `i` of a recurrent state of `element`-byte values (FP32 or BF16).
fn state_value(state: &[u8], i: usize, element: usize) -> f32 {
    if element == 4 {
        f32::from_le_bytes(state[i * 4..i * 4 + 4].try_into().unwrap())
    } else {
        f32::from_bits(u32::from(u16::from_le_bytes(state[i * 2..i * 2 + 2].try_into().unwrap())) << 16)
    }
}

/// Bytes that differ and the largest difference over the recurrent part (`recurrent_bytes` of
/// `element`-byte values) of two slot-state copies.
fn state_delta(a: &[u8], b: &[u8], recurrent_bytes: usize, element: usize) -> (usize, f32) {
    let differ = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let worst = (0..recurrent_bytes / element)
        .map(|i| (state_value(a, i, element) - state_value(b, i, element)).abs()).fold(0f32, f32::max);
    (differ, worst)
}

fn replay_bounds(tokens: usize, rows: usize, prefill: Option<usize>) -> Result<usize> {
    ensure!(rows > 0 && rows <= super::engine::DECODE_ROWS && rows < tokens,
        "--replay-check needs 1..={} rows and at least one prefill token", super::engine::DECODE_ROWS);
    let prefill = prefill.unwrap_or(64).min(tokens - rows);
    ensure!(prefill > 0, "--replay-check needs at least one prefill token");
    Ok(prefill)
}

fn finite_state(state: &[u8], recurrent_bytes: usize, element: usize) -> Result<()> {
    ensure!(recurrent_bytes <= state.len() && matches!(element, 2 | 4) && recurrent_bytes % element == 0,
        "invalid recurrent state layout");
    for i in 0..recurrent_bytes / element {
        ensure!(state_value(state, i, element).is_finite(), "non-finite KDA recurrent word {i}");
    }
    for (i, word) in state[recurrent_bytes..].chunks_exact(2).enumerate() {
        ensure!(f32::from_bits(u32::from(u16::from_le_bytes(word.try_into().unwrap())) << 16).is_finite(),
            "non-finite KDA convolution word {i}");
    }
    Ok(())
}

fn finite_logits(logits: &[f32]) -> Result<()> {
    ensure!(logits.iter().all(|v| v.is_finite()), "non-finite replay-check logits");
    Ok(())
}

fn exact_logits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// One fixed-token serial/wide trace; no greedy feedback or rejected suffix
/// changes. Per-layer router inputs isolate where geometry drift grows.
pub(super) fn geometry_trace(args: &GoldenArgs, engine: &GlmfEngine<'_>, dir: &std::path::Path) -> Result<()> {
    let sequence = tokens(args)?;
    let rows = args.step_rows;
    ensure!(rows > 1, "--geometry-trace needs --step-rows > 1");
    let prefill = replay_bounds(sequence.len(), rows, args.prefill)?;
    let mut allocator = Allocator::new(engine.pages, engine.slots);
    let mut serial = allocator.admit(prefill + rows)?;
    let mut wide = allocator.admit(prefill + rows)?;
    engine.prefill(&mut serial, &sequence[..prefill], None)?;
    engine.prefill(&mut wide, &sequence[..prefill], None)?;
    let mut serial_logits = Vec::new();
    for j in 0..rows {
        let mut snapshot = |layer, streams: &[u8]| engine.trace_decode_layer(layer, 1, streams,
            &dir.join(format!("serial/row{j:02}/layer{layer:02}")));
        serial_logits.extend(engine.verify_trace(&mut [(&mut serial, 1)], &sequence[prefill + j..prefill + j + 1],
            &mut snapshot, &dir.join(format!("serial/row{j:02}")))?.context("geometry trace needs every layer")?);
    }
    let mut snapshot = |layer, streams: &[u8]| engine.trace_decode_layer(layer, rows, streams,
        &dir.join(format!("wide/layer{layer:02}")));
    let wide_logits = engine.verify_trace(&mut [(&mut wide, rows)], &sequence[prefill..prefill + rows],
        &mut snapshot, &dir.join("wide"))?.context("geometry trace needs every layer")?;
    finite_logits(&serial_logits)?;
    finite_logits(&wide_logits)?;
    for (name, logits) in [("serial-logits.bin", &serial_logits), ("wide-logits.bin", &wide_logits)] {
        std::fs::write(dir.join(name), logits.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
    }
    println!("geometry trace: {rows} rows after {prefill} fixed tokens, KL(serial || wide) {:.6} nat; {}",
        super::mean_kl(&wide_logits, &serial_logits, 0, engine.cfg.vocab_size), dir.display());
    Ok(())
}

/// See `GoldenArgs::replay_check`.
pub(super) fn replay_check(args: &GoldenArgs, engine: &GlmfEngine<'_>, rows: usize) -> Result<()> {
    let sequence = tokens(args)?;
    let prefill = replay_bounds(sequence.len(), rows, args.prefill)?;
    ensure!(engine.slots >= 4, "--replay-check needs --slots >= 4");
    let embed = &sequence[prefill..prefill + rows];
    let family = super::prefix::GlmfPrefix::new(engine, super::prefix::PrefixMarks::Arena, |_| 0)?;
    let kda_layers = engine.weights.layers.iter()
        .filter(|l| l.attention == cuteafd_loader::families::glm5_flash::GlmNextAttention::Kda).count();
    // The recurrent part of a slot's state (FP32 or BF16, `--kda-state`), then its conv windows.
    let element = engine.kda_state.bytes();
    let recurrent_bytes = kda_layers * engine.cfg.kda_heads * 128 * 128 * element;
    let allocator = std::cell::RefCell::new(Allocator::new(engine.pages, engine.slots));
    let fresh = || -> Result<GlmfPlacement> {
        let mut placement = allocator.borrow_mut().admit(prefill + rows + 1)?;
        engine.prefill(&mut placement, &sequence[..prefill], None)?;
        Ok(placement)
    };
    let max_logit = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    for keep in 1..=rows {
        let mut serial = fresh()?;
        let mut serial_logits = Vec::new();
        for j in 0..keep {
            if let Some(l) = engine.verify(&mut [(&mut serial, 1)], &embed[j..j + 1], None)? {
                serial_logits.push(l);
            }
        }
        let mut spec = fresh()?;
        let start = spec.len;
        let initial = engine.slot_state(spec.slot)?;
        finite_state(&initial, recurrent_bytes, element)?;
        let logits = engine.verify_spec(&mut [(&mut spec, rows)], embed)?.context("replay check needs every layer")?;
        finite_logits(&logits)?;
        ensure!(engine.slot_state(spec.slot)? == initial, "{rows}-row speculative verify changed uncommitted KDA state");
        engine.commit(&[(spec.slot, 0, keep)])?;
        spec.len = start + keep;
        spec.kda_len = spec.len;
        let retained = engine.slot_state(spec.slot)?;
        finite_state(&retained, recurrent_bytes, element)?;
        let (differ, worst) = state_delta(&engine.slot_state(serial.slot)?, &retained, recurrent_bytes, element);
        let logit = (0..serial_logits.len()).map(|j|
            max_logit(&logits[j * logits.len() / rows..][..logits.len() / rows], &serial_logits[j])).fold(0f32, f32::max);
        let serial_flat: Vec<f32> = serial_logits.iter().flatten().copied().collect();
        finite_logits(&serial_flat)?;
        let vocab = engine.cfg.vocab_size;
        let kl = super::mean_kl(&logits[..keep * vocab], &serial_flat, 0, vocab);
        println!("keep {keep}/{rows}: speculative verify + commit vs {keep} serial steps: {differ} state bytes differ \
            (max |delta| {worst:.3e}); kept-row logits max |delta| {logit:.3e}, mean KL(serial || verify) \
            {kl:.6} nat (geometry diagnostic)");
        // Keep the execution geometry fixed. A later rejected token may change
        // neither an earlier logit nor the committed recurrent/paged state.
        let mut changed = embed.to_vec();
        for token in &mut changed[keep..] {
            *token = (*token + 1) % engine.cfg.vocab_size as u32;
        }
        let mut alternate = fresh()?;
        let alternate_logits = engine.verify_spec(&mut [(&mut alternate, rows)], &changed)?
            .context("replay check needs every layer")?;
        finite_logits(&alternate_logits)?;
        engine.commit(&[(alternate.slot, 0, keep)])?;
        alternate.len = start + keep;
        alternate.kda_len = alternate.len;
        let alternate_state = engine.slot_state(alternate.slot)?;
        finite_state(&alternate_state, recurrent_bytes, element)?;
        let (differ, worst) = state_delta(&retained, &alternate_state, recurrent_bytes, element);
        ensure!(exact_logits(&logits[..keep * vocab], &alternate_logits[..keep * vocab]),
            "keep {keep}/{rows}: rejected suffix changed kept-row logits (same geometry)");
        ensure!(differ == 0, "keep {keep}/{rows}: rejected suffix changed {differ} committed KDA state bytes \
            (max |delta| {worst:.3e}, first byte {:?})", retained.iter().zip(&alternate_state).position(|(a, b)| a != b));
        ensure!(super::prefix::paged_rows(&family, &spec, spec.len)?
            == super::prefix::paged_rows(&family, &alternate, alternate.len)?,
            "keep {keep}/{rows}: rejected suffix changed committed MLA cache rows");
        if keep == rows {
            let mut plain = fresh()?;
            let plain_logits = engine.verify(&mut [(&mut plain, rows)], embed, None)?
                .context("replay check needs every layer")?;
            finite_logits(&plain_logits)?;
            let plain_state = engine.slot_state(plain.slot)?;
            finite_state(&plain_state, recurrent_bytes, element)?;
            let (differ, worst) = state_delta(&plain_state, &retained, recurrent_bytes, element);
            println!("keep {keep}/{rows}: speculative verify + commit vs a plain {rows}-row verify: {differ} state bytes \
                differ (max |delta| {worst:.3e})");
            ensure!(differ == 0 && exact_logits(&plain_logits, &logits),
                "full {rows}-row replay commit differs from plain verify (same geometry)");
            allocator.borrow_mut().release(plain);
        }
        let next = [sequence.get(prefill + keep).copied().unwrap_or(embed[keep - 1])];
        let a = engine.verify(&mut [(&mut spec, 1)], &next, None)?.context("replay check needs every layer")?;
        let b = engine.verify(&mut [(&mut alternate, 1)], &next, None)?.context("replay check needs every layer")?;
        finite_logits(&a)?;
        finite_logits(&b)?;
        let (a_state, b_state) = (engine.slot_state(spec.slot)?, engine.slot_state(alternate.slot)?);
        finite_state(&a_state, recurrent_bytes, element)?;
        finite_state(&b_state, recurrent_bytes, element)?;
        ensure!(exact_logits(&a, &b) && a_state == b_state
            && super::prefix::paged_rows(&family, &spec, spec.len)?
                == super::prefix::paged_rows(&family, &alternate, alternate.len)?,
            "keep {keep}/{rows}: rejected suffix changed continuation logits/state");
        println!("keep {keep}/{rows}: rejected-suffix causality exact (kept logits, KDA state, MLA rows, continuation)");
        allocator.borrow_mut().release(serial);
        allocator.borrow_mut().release(spec);
        allocator.borrow_mut().release(alternate);
    }
    // Cost: plain N-row verify vs speculative verify + commit, same position.
    let mut placement = fresh()?;
    let backup = fresh()?;
    let start = placement.len;
    let mut time = |spec: bool| -> Result<f64> {
        let mut times = Vec::new();
        for _ in 0..7 {
            placement.len = start;
            placement.kda_len = start;
            engine.copy_slot(backup.slot, placement.slot)?;
            // SAFETY: the engine owns this stream; the untimed restore must
            // finish before measuring a step from the same recurrent state.
            unsafe { engine.library.cuda_stream_synchronize(engine.stream)? };
            let timer = Instant::now();
            if spec {
                engine.verify_spec(&mut [(&mut placement, rows)], embed)?;
                engine.commit(&[(placement.slot, 0, rows)])?;
                // SAFETY: the engine owns this stream.
                unsafe { engine.library.cuda_stream_synchronize(engine.stream)? };
            } else {
                engine.verify(&mut [(&mut placement, rows)], embed, None)?;
            }
            times.push(timer.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        Ok(times[times.len() / 2] * 1e3)
    };
    let (plain, spec) = (time(false)?, time(true)?);
    let (plain2, spec2) = (time(false)?, time(true)?);
    println!("{rows}-row step: plain {plain:.2} / {plain2:.2} ms, speculative + commit {spec:.2} / {spec2:.2} ms");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn replay_check_rejects_invalid_bounds_before_subtracting() {
        assert!(super::replay_bounds(10, 11, None).is_err());
        assert!(super::replay_bounds(0, 0, None).is_err());
        assert!(super::replay_bounds(10, 0, None).is_err());
        assert!(super::replay_bounds(10, 10, None).is_err());
        assert!(super::replay_bounds(100, super::super::engine::DECODE_ROWS + 1, None).is_err());
        assert!(super::replay_bounds(10, 4, Some(0)).is_err());
        assert_eq!(super::replay_bounds(10, 4, Some(20)).unwrap(), 6);
    }

    #[test]
    fn replay_check_does_not_accept_non_finite_state_or_logits() {
        assert!(super::finite_state(&f32::NAN.to_le_bytes(), 4, 4).is_err());
        assert!(super::finite_state(&0x7f80u16.to_le_bytes(), 0, 4).is_err());
        assert!(super::finite_logits(&[f32::INFINITY]).is_err());
        assert!(super::finite_state(&1f32.to_le_bytes(), 4, 4).is_ok());
        // A BF16 recurrent state: its words are BF16 too.
        assert!(super::finite_state(&[0x7f80u16.to_le_bytes(), 0x3f80u16.to_le_bytes()].concat(), 2, 2).is_err());
        assert!(super::finite_state(&[0x3f80u16.to_le_bytes(), 0x3f80u16.to_le_bytes()].concat(), 2, 2).is_ok());
        let (a, b) = ([0x3f80u16.to_le_bytes(), 0u16.to_le_bytes()].concat(), [0x4000u16.to_le_bytes(), 0u16.to_le_bytes()].concat());
        assert_eq!(super::state_delta(&a, &b, 2, 2), (2, 1.0));
        assert!(super::finite_logits(&[1.0, -2.0]).is_ok());
        assert!(!super::exact_logits(&[0.0], &[-0.0]));
    }
}

/// See `GoldenArgs::bench_verify`.
pub(super) fn bench_verify(args: &GoldenArgs, engine: &GlmfEngine<'_>, max_rows: usize) -> Result<()> {
    let sequence = tokens(args)?;
    let count = args.bench_sequences.max(1);
    let prefill = args.prefill.unwrap_or(256).min(sequence.len() - max_rows - count);
    let mut allocator = Allocator::new(engine.pages, engine.slots);
    // Distinct sequences: sequence i starts i tokens later in the golden prompt.
    let mut placements = (0..count).map(|i| -> Result<GlmfPlacement> {
        let mut placement = allocator.admit(prefill + max_rows + 1)?;
        engine.prefill(&mut placement, &sequence[i..i + prefill], None)?;
        Ok(placement)
    }).collect::<Result<Vec<_>>>()?;
    let starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
    println!("verify cost, {count} distinct sequence(s) after {prefill} tokens (speculative steps, median of 7):");
    for rows in 1..=max_rows {
        if count * rows > super::engine::DECODE_ROWS {
            break;
        }
        let tokens: Vec<u32> = (0..count).flat_map(|i| sequence[i + prefill..i + prefill + rows].iter().copied()).collect();
        let mut times = Vec::new();
        *engine.profile.borrow_mut() = [0.0; 3];
        for round in 0..9 {
            for (placement, &start) in placements.iter_mut().zip(&starts) {
                placement.len = start;
            }
            let mut step: Vec<(&mut GlmfPlacement, usize)> = placements.iter_mut().map(|p| (p, rows)).collect();
            let timer = Instant::now();
            engine.verify_spec(&mut step, &tokens)?;
            if round >= 2 {
                times.push(timer.elapsed().as_secs_f64());
            } else {
                *engine.profile.borrow_mut() = [0.0; 3];
            }
        }
        times.sort_by(f64::total_cmp);
        let phases = std::mem::take(&mut *engine.profile.borrow_mut());
        let n = times.len() as f64;
        println!("  {rows} rows/sequence ({} rows): {:.2} ms (min {:.2}); GPU until exchanges {:.2} ms, Spark exchanges \
            {:.2} ms, head {:.2} ms", count * rows, 1e3 * times[times.len() / 2], 1e3 * times[0], 1e3 * phases[0] / n,
            1e3 * phases[1] / n, 1e3 * phases[2] / n);
    }
    Ok(())
}
