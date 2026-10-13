//! Speculation checks for GLM 5.3 Flash (glmf-golden): the DFlash2 drafter
//! against the torch reference, drafts against the target, KDA
//! verify-by-replay against serial steps, and the verify-step cost by rows.
use super::engine::{Allocator, GlmfEngine, GlmfPlacement};
use super::{bf16s, similarity, GoldenArgs, Opened};
use crate::families::glm5::dflash::{ContextRow, DraftSeq, RING, TAP_ROWS};
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
            // A following oracle chunk reuses taps and context metadata on
            // the selected drafter device, not necessarily the target stream.
            engine.synchronize()?;
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
    crate::families::glm5::dflash::replay_with(drafter.replay(), &tokens, &greedy, &taps,
        &|t| engine.embedding.host_rows(t), &|sequences, _| {
            Ok(drafter.draft_device(sequences, &engine.embedding, engine.draft_head())?.into_iter()
                .map(|draft| draft.tokens).collect())
        }, start)
}

/// A drafter setting: (--draft-head, --draft-linear).
type DraftMode = (crate::families::glm5::DraftHead, crate::shared::fp8_linear::Fp8Rows);

/// The setting [`draft_modes`] compares against and builds every context under: the defaults.
const REFERENCE_MODE: DraftMode =
    (crate::families::glm5::DraftHead::Exact, crate::shared::fp8_linear::Fp8Rows::W8a16);

/// The settings [`draft_modes`] compares: both heads under every linear mode an FP8 drafter's
/// scratch admits (`admitted`, its load's mode, and the modes before it), the head alone for a
/// BF16 drafter. The first is [`REFERENCE_MODE`].
fn draft_mode_settings(fp8: bool, admitted: crate::shared::fp8_linear::Fp8Rows) -> Vec<DraftMode> {
    use crate::families::glm5::DraftHead;
    use crate::shared::fp8_linear::Fp8Rows;
    [Fp8Rows::W8a16, Fp8Rows::Wide, Fp8Rows::W8a8].into_iter()
        .filter(|&linear| linear == Fp8Rows::W8a16 || (fp8 && linear <= admitted))
        .flat_map(|linear| [DraftHead::Exact, DraftHead::Tensor].map(|head| (head, linear)))
        .collect()
}

/// What [`draft_mode_schedule`] drives: the target and drafter on a GPU, a recorder in tests.
trait DraftModeSteps {
    /// Selects the drafter's head and linear modes for the calls that follow.
    fn set_mode(&mut self, mode: DraftMode) -> Result<()>;
    /// Prefills `tokens` from position 0, projecting their tapped tail into the ring.
    fn prefill(&mut self, tokens: &[u32]) -> Result<()>;
    /// One teacher-forced target step of `tokens` from position `first`, projecting every row
    /// into the ring.
    fn advance(&mut self, first: usize, tokens: &[u32]) -> Result<()>;
    /// One draft step of `sequences`; their drafted tokens.
    fn draft(&mut self, sequences: &[DraftSeq]) -> Result<Vec<Vec<u32>>>;
}

/// [`draft_mode_schedule`]'s counts. Per setting: drafts kept, identical to the first setting's,
/// identical to the anchors' own, and the step seconds.
struct DraftModeTally {
    anchors: usize,
    alone_kept: usize,
    settings: Vec<(usize, usize, usize, Vec<f64>)>,
}

/// [`draft_modes`]' schedule. Every context the ring holds is projected under
/// [`REFERENCE_MODE`]: the mode is set before the prefill and before every run of context
/// updates, so the anchors alone and every setting draft from the same ring, whichever setting
/// drafted last. Before each window of `sequences` anchors (positions p..p + N, up to `end`)
/// the anchors draft alone under the reference, then together under each setting in turn.
fn draft_mode_schedule(steps: &mut impl DraftModeSteps, tokens: &[u32], prefill: usize, end: usize,
    sequences: usize, step_rows: usize, settings: &[DraftMode]) -> Result<DraftModeTally> {
    ensure!(settings.first() == Some(&REFERENCE_MODE), "--draft-modes compares against {REFERENCE_MODE:?} first");
    let kept = |draft: &[u32], position: usize| draft.iter().zip(&tokens[position + 1..]).take_while(|(d, t)| d == t).count();
    let mut tally = DraftModeTally { anchors: 0, alone_kept: 0,
        settings: vec![(0usize, 0usize, 0usize, Vec::new()); settings.len()] };
    steps.set_mode(REFERENCE_MODE)?;
    steps.prefill(&tokens[..prefill])?;
    let (mut done, mut p) = (prefill, prefill);
    while p + sequences <= end {
        // The context of anchors p..p + N ends at p + N - 2; the last setting may still be set.
        steps.set_mode(REFERENCE_MODE)?;
        while done + 1 < p + sequences {
            let rows = (p + sequences - 1 - done).min(step_rows);
            steps.advance(done, &tokens[done..done + rows])?;
            done += rows;
        }
        let seqs: Vec<DraftSeq> = (p..p + sequences)
            .map(|q| DraftSeq { slot: 0, anchor: tokens[q], position: q, valid_from: 0 }).collect();
        let alone = seqs.iter().map(|seq| Ok(steps.draft(std::slice::from_ref(seq))?.remove(0)))
            .collect::<Result<Vec<_>>>()?;
        tally.alone_kept += alone.iter().enumerate().map(|(j, d)| kept(d, p + j)).sum::<usize>();
        let mut first: Option<Vec<Vec<u32>>> = None;
        for (&mode, total) in settings.iter().zip(&mut tally.settings) {
            steps.set_mode(mode)?;
            let timer = Instant::now();
            let wide = steps.draft(&seqs)?;
            total.3.push(timer.elapsed().as_secs_f64());
            total.0 += wide.iter().enumerate().map(|(j, d)| kept(d, p + j)).sum::<usize>();
            total.1 += wide.iter().zip(first.get_or_insert_with(|| wide.clone()).iter()).filter(|(a, b)| a == b).count();
            total.2 += wide.iter().zip(&alone).filter(|(a, b)| a == b).count();
        }
        tally.anchors += sequences;
        p += sequences;
    }
    Ok(tally)
}

/// [`DraftModeSteps`] on the target and its drafter (ring slot 0).
struct EngineDraftModes<'a, 'e> {
    engine: &'a GlmfEngine<'e>,
    drafter: &'a super::dspark::Drafter<'e>,
    placement: GlmfPlacement,
}

impl DraftModeSteps for EngineDraftModes<'_, '_> {
    fn set_mode(&mut self, (head, linear): DraftMode) -> Result<()> {
        self.drafter.set_draft_head(head);
        self.drafter.set_draft_linear(linear)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        prefill_with_taps(self.engine, &mut self.placement, tokens, 0).map(drop)
    }

    fn advance(&mut self, first: usize, tokens: &[u32]) -> Result<()> {
        let rows = tokens.len();
        self.engine.verify(&mut [(&mut self.placement, rows)], tokens, None)?
            .context("--draft-modes needs every layer")?;
        self.drafter.update(&(0..rows).map(|r| ContextRow { tap_row: r, slot: 0, position: first + r }).collect::<Vec<_>>())
    }

    fn draft(&mut self, sequences: &[DraftSeq]) -> Result<Vec<Vec<u32>>> {
        Ok(self.drafter.draft_device(sequences, &self.engine.embedding, self.engine.draft_head())?
            .into_iter().map(|d| d.tokens).collect())
    }
}

/// Draft-kernel A/B (`--draft-modes N`): after the --prefill tokens (default 1024), the golden
/// tokens go through the target in teacher-forced steps that tap the drafter's ring slot 0, and
/// before each window of N anchors (positions p..p + N) the window drafts in one step of N
/// sequences (8N rows; each anchor at its own position on the one ring) under every --draft-head
/// and --draft-linear setting the drafter admits, and each anchor alone (8 rows, the same bits
/// under every setting). The ring's context is projected under the default setting (exact head,
/// W8A16) throughout, so every setting drafts from the same context ([`draft_mode_schedule`]).
/// A window's context ends at p + N - 2 and an anchor at q reads only positions before q; windows
/// stop at the ring's length, so no entry an anchor reads has been overwritten. Prints, per
/// setting, the drafts kept as a prefix of the text, the share of drafts identical to the first
/// setting's and to the anchors' own, and the median step time.
pub(super) fn draft_modes(args: &GoldenArgs, engine: &GlmfEngine<'_>, sequences: usize) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft-modes needs --draft")?;
    ensure!((1..=drafter.max_batch_sequences()).contains(&sequences),
        "--draft-modes takes 1 to {} sequences (--draft-sequences)", drafter.max_batch_sequences());
    let tokens = tokens(args)?;
    let drafts = drafter.drafts();
    let prefill = args.prefill.unwrap_or(1024);
    // The last window's context ends at the ring's last position; its drafts score `drafts` tokens.
    let end = (RING + 1).min(tokens.len().saturating_sub(drafts));
    ensure!(prefill >= 1 && prefill + sequences <= end, "--draft-modes needs --prefill + {sequences} <= {end} \
        (the ring holds {RING} positions; tokens.bin has {} tokens)", tokens.len());
    let fp8 = drafter.replay().resident_modes().iter().any(|mode| mode.name == "FP8");
    let settings = draft_mode_settings(fp8, args.engine.draft_linear);
    let placement = Allocator::for_engine(engine).admit(end + 1)?;
    let mut steps = EngineDraftModes { engine, drafter, placement };
    let tally = draft_mode_schedule(&mut steps, &tokens, prefill, end, sequences, super::engine::DECODE_ROWS,
        &settings);
    // The load's settings again, whether or not the schedule finished.
    steps.set_mode((args.engine.draft_head, args.engine.draft_linear))?;
    let DraftModeTally { anchors, alone_kept, settings: mut totals } = tally?;
    let n = anchors.max(1) as f64;
    println!("draft modes ({} drafter, {}): {anchors} anchors from {prefill} in steps of {sequences} sequences \
        ({} rows), every context under the first setting; drafts kept as a prefix of the text, of {drafts}:",
        drafter.name(), if fp8 { "FP8" } else { "BF16" }, sequences * drafter.block());
    println!("  each anchor alone ({} rows): kept {:.3}", drafter.block(), alone_kept as f64 / n);
    for (&(head, linear), (kept, same, same_alone, times)) in settings.iter().zip(&mut totals) {
        times.sort_by(f64::total_cmp);
        println!("  head {head:?}, linear {linear:?}: kept {:.3}, identical to the first setting {:.2}%, to the \
            anchors' own {:.2}%, step median {:.3} ms (p10 {:.3}, p90 {:.3})", *kept as f64 / n,
            100.0 * *same as f64 / n, 100.0 * *same_alone as f64 / n, 1e3 * times[times.len() / 2],
            1e3 * times[times.len() / 10], 1e3 * times[times.len() * 9 / 10]);
    }
    Ok(())
}

/// Prefills the golden prompt's first --prefill tokens, then decodes one row
/// per step (teacher-forced on tokens.bin, or greedy with --generate),
/// drafting with DFlash2 before every step; reports the accepted prefix per
/// step against the sequence.
pub(super) fn draft_run(args: &GoldenArgs, opened: &Opened, engine: &GlmfEngine<'_>) -> Result<()> {
    let drafter = engine.drafter.as_ref().context("--draft")?;
    let golden = tokens(args)?;
    let mut sequence = golden.clone();
    let prefill = args.prefill.unwrap_or(sequence.len() / 2).min(sequence.len() - 1);
    let end = match args.generate {
        Some(n) if n > 0 => {
            sequence.truncate(prefill);
            prefill + n
        }
        _ => sequence.len(),
    };
    let greedy = end > sequence.len();
    let mut allocator = Allocator::for_engine(engine);
    let mut placement = allocator.admit(end + 1)?;
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
    engine.synchronize()?;
    allocator.release(placement);
    draft_prefill_bench(args, &golden, prefill, engine.prefill_capacity(), |prompt| {
        let mut fresh = allocator.admit(prompt.len())?;
        let started = Instant::now();
        // Serving leaves intermediate logits on device; a host download would
        // drain both target ranks before the context bridge and hide wait cycles.
        for chunk in prompt.chunks(engine.prefill_capacity()) {
            let start = fresh.len;
            engine.prefill_device(&mut fresh, chunk)?.context("prefill needs every layer")?;
            let n = chunk.len().min(TAP_ROWS);
            let first = start + chunk.len() - n;
            drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot: 0, position: first + r })
                .collect::<Vec<_>>())?;
        }
        engine.synchronize()?;
        let seconds = started.elapsed().as_secs_f64();
        allocator.release(fresh);
        Ok(seconds)
    })
}

fn draft_prefill_bench(args: &GoldenArgs, golden: &[u32], prefill: usize, capacity: usize,
    mut run: impl FnMut(&[u32]) -> Result<f64>) -> Result<()> {
    if args.bench_prefill == 0 { return Ok(()) }
    let n = args.bench_prefill_tokens.unwrap_or(prefill.min(capacity));
    ensure!(n > 0 && !golden.is_empty(), "--bench-prefill needs a nonempty prompt");
    let prompt: Vec<_> = golden.iter().copied().cycle().take(n).collect();
    let mut times = Vec::with_capacity(args.bench_prefill);
    for index in 0..args.bench_prefill {
        println!("post-draft prefill {}/{}, {n} tokens: starting", index + 1, args.bench_prefill);
        let seconds = run(&prompt)?;
        println!("post-draft prefill {}/{}, {n} tokens: completed in {seconds:.2} s",
            index + 1, args.bench_prefill);
        times.push(seconds);
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    println!("post-draft prefill bench: {n} tokens, median {:.1} ms ({:.0} tok/s), min {:.1} ms",
        1e3 * median, n as f64 / median, 1e3 * times[0]);
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

/// The prefill before a `rows`-row check of `tokens` golden tokens on an engine of `decode_rows`
/// decode rows (`--decode-rows`: up to 128 with the wide programs).
fn replay_bounds(tokens: usize, rows: usize, prefill: Option<usize>, decode_rows: usize) -> Result<usize> {
    ensure!(rows > 0 && rows <= decode_rows && rows < tokens,
        "--replay-check needs 1..={decode_rows} rows (--decode-rows) and at least one prefill token");
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
    let prefill = replay_bounds(sequence.len(), rows, args.prefill, engine.decode_rows)?;
    let mut allocator = Allocator::for_engine(engine);
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

#[derive(serde::Deserialize)]
struct ReplayRound {
    round: u64,
    spec: bool,
    request_ids: Vec<u64>,
    starts: Vec<usize>,
    verify_tokens: Vec<Vec<u32>>,
    committed_tokens: Vec<Vec<u32>>,
    neural_tokens: Vec<Option<Vec<u32>>>,
    finished: Vec<bool>,
}

fn parse_replay_rounds(text: &str, vocab: usize, decode_rows: usize) -> Result<Vec<ReplayRound>> {
    let mut rounds = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let round: ReplayRound = serde_json::from_str(line).context("invalid speculation replay trace")?;
        ensure!(round.request_ids.len() == 1 && round.starts.len() == 1 && round.verify_tokens.len() == 1
            && round.committed_tokens.len() == 1 && round.neural_tokens.len() == 1 && round.finished.len() == 1,
            "round {}: replay trace needs exactly one request", round.round);
        let (input, emitted) = (&round.verify_tokens[0], &round.committed_tokens[0]);
        ensure!(!input.is_empty() && input.len() <= decode_rows && !emitted.is_empty() && emitted.len() <= input.len(),
            "round {}: invalid verify/commit width", round.round);
        ensure!(input.iter().chain(emitted).all(|&id| (id as usize) < vocab),
            "round {}: token outside vocabulary", round.round);
        ensure!(input[1..emitted.len()] == emitted[..emitted.len() - 1],
            "round {}: accepted inputs do not match emissions", round.round);
        rounds.push(round);
    }
    ensure!(!rounds.is_empty(), "empty speculation replay trace");
    Ok(rounds)
}

/// The target, selector, commit and drafter context lifecycle from a cold single-request trace.
pub(super) fn private_replay_dir(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    match std::fs::symlink_metadata(path) {
        Ok(meta) => ensure!(meta.is_dir() && meta.permissions().mode() & 0o077 == 0,
            "replay dump must be a private, non-symlink directory"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::DirBuilder::new().mode(0o700).create(path)?;
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

pub(super) fn private_replay_write(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?.write_all(bytes)?;
    Ok(())
}

fn first_differing_byte(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

fn replay_round_valid(dump_continue: bool, geometry_exact: bool, serial_emissions_exact: bool,
    recorded_emissions_exact: bool) -> bool {
    recorded_emissions_exact && (dump_continue || (geometry_exact && serial_emissions_exact))
}

fn replay_rounds(args: &GoldenArgs, engine: &GlmfEngine<'_>, sequence: &[u32], prefill: usize,
    path: &std::path::Path) -> Result<()> {
    use crate::shared::token_io::{SelectBatch, SelectPlacement, TokenSelector};
    use cuteafd_core::TargetSamplingParams;
    let rounds = parse_replay_rounds(&std::fs::read_to_string(path)?, engine.cfg.vocab_size, engine.decode_rows)?;
    ensure!(engine.production_replay_ready() && engine.slots >= 2, "trace replay needs production graphs and two slots");
    if let Some(dir) = &args.replay_dump {
        private_replay_dir(dir)?;
        private_replay_write(&dir.join("owners.json"), &serde_json::to_vec_pretty(&engine.replay_owner_ranges()?)?)?;
    }
    let capacity = rounds.iter().map(|r| r.starts[0].checked_add(r.verify_tokens[0].len())
        .context("trace position overflow")).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap().max(prefill);
    ensure!(capacity <= engine.max_context, "trace exceeds admitted context");
    let mut allocator = Allocator::for_engine(engine);
    let mut target = allocator.admit(capacity)?;
    let mut serial = allocator.admit(capacity)?;
    let prompt_logits = prefill_with_taps(engine, &mut target, &sequence[..prefill], 0)?;
    let serial_prompt = engine.prefill(&mut serial, &sequence[..prefill], None)?
        .context("serial replay prefill needs every layer")?;
    finite_logits(&prompt_logits)?;
    finite_logits(&serial_prompt)?;
    ensure!(exact_logits(&prompt_logits, &serial_prompt)
        && argmax(&prompt_logits) == rounds[0].verify_tokens[0][0],
        "trace replay prompt logits or initial anchor diverged");
    let family = super::prefix::GlmfPrefix::new(engine, super::prefix::PrefixMarks::Arena, |_| 0)?;
    let mut selector = TokenSelector::new(engine.library, SelectPlacement::Device, engine.cfg.vocab_size,
        engine.decode_rows)?;
    let vocab = engine.cfg.vocab_size;
    let request = rounds[0].request_ids[0];
    let mut anchor = argmax(&prompt_logits);
    let snapshots = engine.enable_replay_snapshots()?;
    ensure!(!snapshots || args.replay_dump.is_some(), "replay snapshots need --replay-dump");
    for round in rounds {
        ensure!(round.request_ids[0] == request && round.starts[0] == target.len,
            "round {}: request or position changed (expected {})", round.round, target.len);
        let (input, emitted) = (&round.verify_tokens[0], &round.committed_tokens[0]);
        ensure!(input[0] == anchor, "round {}: anchor does not continue the preceding emission", round.round);
        let start = target.len;
        if args.replay_draft_sync { engine.replay_device_sync()?; }
        if round.neural_tokens[0].is_some() {
            let drafter = engine.drafter.as_ref().context("neural trace replay needs --draft")?;
            let draft = drafter.draft_device(&[DraftSeq { slot: 0, anchor: input[0], position: start, valid_from: 0 }],
                &engine.embedding, engine.draft_head())?.remove(0);
            println!("round {}: neural proposal matches trace {}", round.round,
                round.neural_tokens[0].as_ref().is_some_and(|tokens| *tokens == draft.tokens));
        }
        if snapshots {
            let dir = args.replay_dump.as_ref().unwrap().join(format!("round{}", round.round));
            private_replay_dir(&dir)?;
            engine.replay_snapshot_directory(dir.join("target"))?;
        }
        let logits = engine.verify_device(&mut [(&mut target, input.len())], input, round.spec)?
            .context("trace replay needs every layer")?;
        let mut batch = SelectBatch::default();
        batch.push_sequence(TargetSamplingParams::greedy(), None, input, start as u64 + 1)?;
        let selected = selector.select(&logits, &batch)?;
        let actual = logits.to_host(engine.library)?;
        finite_logits(&actual)?;
        let keep = emitted.len();
        let sampler_exact = selected.len() >= keep && selected.iter().take(keep).zip(actual.chunks_exact(vocab))
            .all(|(selected, row)| selected.as_ref().is_ok_and(|s| s.token == argmax(row)));
        ensure!(sampler_exact, "round {}: selector differs from downloaded logits", round.round);
        // Finished cold requests release without commit/update, but their emitted logits still gate.
        let committed = if round.finished[0] {
            None
        } else {
            if round.spec { engine.commit(&[(target.slot, 0, keep)])?; }
            target.len = start + keep;
            target.kda_len = target.len;
            let committed = engine.slot_state(target.slot)?;
            let committed_mla = args.replay_dump_continue.then(||
                super::prefix::paged_rows(&family, &target, target.len)).transpose()?;
            if args.replay_draft_sync { engine.replay_device_sync()?; }
            if let Some(drafter) = &engine.drafter {
                drafter.update(&(0..keep).map(|r| ContextRow { tap_row: r, slot: 0, position: start + r })
                    .collect::<Vec<_>>())?;
            }
            ensure!(engine.slot_state(target.slot)? == committed,
                "round {}: drafter context update changed target state", round.round);
            if let Some(before) = committed_mla {
                ensure!(super::prefix::paged_rows(&family, &target, target.len)? == before,
                    "round {}: drafter context update changed target MLA", round.round);
            }
            Some(committed)
        };
        let mut expected = Vec::new();
        for (row, &token) in input[..keep].iter().enumerate() {
            if snapshots {
                engine.replay_snapshot_directory(args.replay_dump.as_ref().unwrap()
                    .join(format!("round{}/serial{row:02}", round.round)))?;
            }
            expected.extend(engine.verify_device(&mut [(&mut serial, 1)], &[token], false)?
                .context("serial replay needs every layer")?.to_host(engine.library)?);
        }
        finite_logits(&expected)?;
        let logit_exact = exact_logits(&actual[..keep * vocab], &expected);
        let serial_state = committed.as_ref().map(|_| engine.slot_state(serial.slot)).transpose()?;
        let state_first = committed.as_ref().zip(serial_state.as_ref())
            .and_then(|(a, b)| first_differing_byte(a, b));
        let state_exact = state_first.is_none();
        let paged = if round.finished[0] { None } else {
            Some((super::prefix::paged_rows(&family, &target, target.len)?,
                super::prefix::paged_rows(&family, &serial, serial.len)?))
        };
        let paged_first = paged.as_ref().and_then(|(a, b)| first_differing_byte(a, b));
        let paged_exact = paged_first.is_none();
        if let Some(dir) = &args.replay_dump {
            let dir = dir.join(format!("round{}", round.round));
            private_replay_dir(&dir)?;
            for (name, logits) in [("target-logits.bin", &actual[..keep * vocab]), ("serial-logits.bin", expected.as_slice())] {
                private_replay_write(&dir.join(name), &logits.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
            }
            if let Some((target_state, serial_state)) = committed.as_ref().zip(serial_state.as_ref()) {
                private_replay_write(&dir.join("target-state.bin"), target_state)?;
                private_replay_write(&dir.join("serial-state.bin"), serial_state)?;
            }
            if let Some((target_paged, serial_paged)) = &paged {
                private_replay_write(&dir.join("target-mla.bin"), target_paged)?;
                private_replay_write(&dir.join("serial-mla.bin"), serial_paged)?;
            }
            private_replay_write(&dir.join("geometry.json"), &serde_json::to_vec(&serde_json::json!({
                "round": round.round, "start": start, "live": input.len(), "keep": keep,
                "bucket": engine.serving_decode_rows(input.len(), round.spec), "sync": args.replay_draft_sync,
                "state_first_byte": state_first, "mla_first_byte": paged_first }))?)?;
        }
        let first = actual[..keep * vocab].iter().zip(&expected).position(|(a,b)| a.to_bits() != b.to_bits());
        let emissions_exact = actual[..keep * vocab].chunks_exact(vocab).map(argmax).eq(emitted.iter().copied());
        let serial_emissions_exact = expected.chunks_exact(vocab).map(argmax).eq(emitted.iter().copied());
        println!("round {}: start {start}, live {}, bucket {}, keep {keep}: logits exact {logit_exact}, \
            first differing (row,vocab) {:?}; state exact {state_exact}, first byte {state_first:?}; \
            MLA exact {paged_exact}, first byte {paged_first:?}; \
            selector exact {sampler_exact}; recorded emissions exact {emissions_exact}; \
            serial emissions exact {serial_emissions_exact}", round.round, input.len(),
            engine.serving_decode_rows(input.len(), round.spec), first.map(|i| (i / vocab, i % vocab)));
        ensure!(replay_round_valid(args.replay_dump_continue, logit_exact && state_exact && paged_exact,
            serial_emissions_exact, emissions_exact), "round {}: first serving replay divergence", round.round);
        if args.replay_dump_continue {
            println!("round {}: dump-only diagnostic; serial-geometry comparison is informational, not a correctness PASS", round.round);
        }
        if round.finished[0] {
            println!("round {}: finished cold frontier; persistent state and MLA comparison skipped", round.round);
            break;
        }
        anchor = *emitted.last().unwrap();
    }
    Ok(())
}

/// See `GoldenArgs::replay_check`.
pub(super) fn replay_check(args: &GoldenArgs, engine: &GlmfEngine<'_>, rows: usize) -> Result<()> {
    let sequence = tokens(args)?;
    let prefill = replay_bounds(sequence.len(), rows, args.prefill, engine.decode_rows)?;
    if let Some(path) = &args.replay_trace {
        return replay_rounds(args, engine, &sequence, prefill, path);
    }
    ensure!(engine.slots >= 4, "--replay-check needs --slots >= 4");
    ensure!(!args.replay_production || engine.production_replay_ready(),
        "--replay-production needs captured startup decode graphs");
    let verify = |placement: &mut GlmfPlacement, count, input: &[u32], spec| -> Result<Vec<f32>> {
        let logits = if args.replay_production {
            engine.verify_device(&mut [(placement, count)], input, spec)?
                .map(|logits| logits.to_host(engine.library)).transpose()?
        } else if spec {
            engine.verify_spec(&mut [(placement, count)], input)?
        } else {
            engine.verify(&mut [(placement, count)], input, None)?
        };
        logits.context("replay check needs every layer")
    };
    if args.replay_production {
        println!("production replay: live rows {rows}, captured bucket {}; \
            expert policy uses live rows, copy-extension rows use the same keep boundary",
            engine.serving_decode_rows(rows, true));
    }
    let embed = &sequence[prefill..prefill + rows];
    let family = super::prefix::GlmfPrefix::new(engine, super::prefix::PrefixMarks::Arena, |_| 0)?;
    let kda_layers = engine.weights.layers.iter()
        .filter(|l| l.attention == cuteafd_loader::families::glm5_flash::GlmNextAttention::Kda).count();
    // The recurrent part of a slot's state (FP32 or BF16, `--kda-state`), then its conv windows.
    let element = engine.kda_state.bytes();
    let recurrent_bytes = kda_layers * engine.cfg.kda_heads * 128 * 128 * element;
    let allocator = std::cell::RefCell::new(Allocator::for_engine(engine));
    let fresh = || -> Result<GlmfPlacement> {
        let mut placement = allocator.borrow_mut().admit(prefill + rows + 1)?;
        engine.prefill(&mut placement, &sequence[..prefill], None)?;
        Ok(placement)
    };
    let max_logit = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let mut production_exact = true;
    for keep in 1..=rows {
        let mut serial = fresh()?;
        let mut serial_logits = Vec::new();
        for j in 0..keep {
            serial_logits.push(verify(&mut serial, 1, &embed[j..j + 1], false)?);
        }
        let mut spec = fresh()?;
        let start = spec.len;
        let initial = engine.slot_state(spec.slot)?;
        finite_state(&initial, recurrent_bytes, element)?;
        let logits = verify(&mut spec, rows, embed, true)?;
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
        if args.replay_production {
            let mut eager = fresh()?;
            let eager_initial = engine.slot_state(eager.slot)?;
            let eager_logits = engine.verify_spec(&mut [(&mut eager, rows)], embed)?
                .context("replay check needs every layer")?;
            finite_logits(&eager_logits)?;
            ensure!(engine.slot_state(eager.slot)? == eager_initial,
                "eager speculative verify changed uncommitted KDA state");
            engine.commit(&[(eager.slot, 0, keep)])?;
            eager.len = start + keep;
            eager.kda_len = eager.len;
            let eager_state = engine.slot_state(eager.slot)?;
            finite_state(&eager_state, recurrent_bytes, element)?;
            let (state_bytes, worst) = state_delta(&eager_state, &retained, recurrent_bytes, element);
            let first_state = eager_state.iter().zip(&retained).position(|(a, b)| a != b);
            let first_logit = eager_logits.iter().zip(&logits).position(|(a, b)| a.to_bits() != b.to_bits());
            let eager_paged = super::prefix::paged_rows(&family, &eager, eager.len)?;
            let production_paged = super::prefix::paged_rows(&family, &spec, spec.len)?;
            let paged_exact = eager_paged == production_paged;
            let first_paged = eager_paged.iter().zip(&production_paged).position(|(a, b)| a != b);
            let bucket = engine.serving_decode_rows(rows, true);
            println!("keep {keep}/{rows}: eager {rows} vs production bucket {bucket}: logits exact {}, \
                first differing (row, vocab) {:?}, max |delta| {:.3e}; committed KDA/conv/index-tail \
                state {state_bytes} bytes differ (first byte {first_state:?}, recurrent max |delta| \
                {worst:.3e}); committed MLA exact {paged_exact}, first byte {first_paged:?}",
                exact_logits(&eager_logits, &logits), first_logit.map(|i| (i / vocab, i % vocab)),
                max_logit(&eager_logits, &logits));
            production_exact &= exact_logits(&eager_logits, &logits) && eager_state == retained && paged_exact;
            allocator.borrow_mut().release(eager);
        }
        // Keep the execution geometry fixed. A later rejected token may change
        // neither an earlier logit nor the committed recurrent/paged state.
        let mut changed = embed.to_vec();
        for token in &mut changed[keep..] {
            *token = (*token + 1) % engine.cfg.vocab_size as u32;
        }
        let mut alternate = fresh()?;
        let alternate_logits = verify(&mut alternate, rows, &changed, true)?;
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
            let plain_logits = verify(&mut plain, rows, embed, false)?;
            finite_logits(&plain_logits)?;
            let plain_state = engine.slot_state(plain.slot)?;
            finite_state(&plain_state, recurrent_bytes, element)?;
            let (differ, worst) = state_delta(&plain_state, &retained, recurrent_bytes, element);
            println!("keep {keep}/{rows}: speculative verify + commit vs a plain {rows}-row verify: {differ} state bytes \
                differ (max |delta| {worst:.3e})");
            ensure!(differ == 0 && exact_logits(&plain_logits, &logits),
                "full {rows}-row replay commit differs from plain verify (same geometry)");
            let plain_paged = super::prefix::paged_rows(&family, &plain, plain.len)?;
            ensure!(plain_paged == super::prefix::paged_rows(&family, &spec, spec.len)?,
                "full {rows}-row replay commit differs from plain MLA cache rows");
            allocator.borrow_mut().release(plain);
            if args.replay_production {
                let mut eager_plain = fresh()?;
                let eager_logits = engine.verify(&mut [(&mut eager_plain, rows)], embed, None)?
                    .context("replay check needs every layer")?;
                finite_logits(&eager_logits)?;
                let eager_state = engine.slot_state(eager_plain.slot)?;
                finite_state(&eager_state, recurrent_bytes, element)?;
                let eager_paged = super::prefix::paged_rows(&family, &eager_plain, eager_plain.len)?;
                let state_bytes = state_delta(&eager_state, &plain_state, recurrent_bytes, element).0;
                println!("plain {rows}: eager vs production bucket {}: logits exact {}, max |delta| {:.3e}; \
                    committed state {state_bytes} bytes differ; MLA exact {}",
                    engine.serving_decode_rows(rows, false), exact_logits(&eager_logits, &plain_logits),
                    max_logit(&eager_logits, &plain_logits), eager_paged == plain_paged);
                production_exact &= exact_logits(&eager_logits, &plain_logits)
                    && eager_state == plain_state && eager_paged == plain_paged;
                allocator.borrow_mut().release(eager_plain);
            }
        }
        let next = [sequence.get(prefill + keep).copied().unwrap_or(embed[keep - 1])];
        let a = verify(&mut spec, 1, &next, false)?;
        let b = verify(&mut alternate, 1, &next, false)?;
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
    ensure!(production_exact, "eager {rows}-row verification differs from production padding; see per-keep diagnostics");
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
            verify(&mut placement, rows, embed, spec)?;
            if spec {
                engine.commit(&[(placement.slot, 0, rows)])?;
            }
            engine.synchronize()?;
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
    fn draft_prefill_options_run_requested_fresh_prompts() {
        use clap::Parser;
        #[derive(Parser)]
        struct Parse {
            #[command(flatten)]
            args: super::GoldenArgs,
        }
        let base = ["test", "--snapshot", "/checkpoint", "--native-lib", "/native", "--golden", "/golden",
            "--draft", "/drafter"];
        let mut args = Parse::try_parse_from(base.into_iter().chain([
            "--bench-prefill", "3", "--bench-prefill-tokens", "8"])).unwrap().args;
        assert!(args.engine.draft.is_some());
        let mut seen = Vec::new();
        super::draft_prefill_bench(&args, &[1, 2, 3], 2, 4, |prompt| {
            seen.push(prompt.to_vec());
            Ok(1.0)
        }).unwrap();
        assert_eq!(seen, vec![vec![1, 2, 3, 1, 2, 3, 1, 2]; 3]);
        args.bench_prefill_tokens = None;
        args.bench_prefill = 1;
        super::draft_prefill_bench(&args, &[1, 2, 3], 5, 2, |prompt| {
            assert_eq!(prompt, &[1, 2]);
            Ok(1.0)
        }).unwrap();
        args.bench_prefill = 0;
        super::draft_prefill_bench(&args, &[], 0, 0, |_| panic!("disabled benchmark ran")).unwrap();
        args.bench_prefill = 1;
        args.bench_prefill_tokens = Some(0);
        assert!(super::draft_prefill_bench(&args, &[1], 1, 1, |_| panic!("empty benchmark ran")).is_err());
        let source = include_str!("speculate.rs");
        let draft_run = source.split("pub(super) fn draft_run").nth(1).unwrap()
            .split("fn draft_prefill_bench").next().unwrap();
        assert!(draft_run.contains("draft_prefill_bench(args, &golden"));
        assert!(draft_run.find("allocator.release(placement)").unwrap()
            < draft_run.find("draft_prefill_bench(args, &golden").unwrap());
    }

    #[test]
    fn replay_check_rejects_invalid_bounds_before_subtracting() {
        use super::super::engine::{DECODE_ROWS, WIDE_DECODE_ROWS};
        assert!(super::replay_bounds(10, 11, None, DECODE_ROWS).is_err());
        assert!(super::replay_bounds(0, 0, None, DECODE_ROWS).is_err());
        assert!(super::replay_bounds(10, 0, None, DECODE_ROWS).is_err());
        assert!(super::replay_bounds(10, 10, None, DECODE_ROWS).is_err());
        assert!(super::replay_bounds(200, DECODE_ROWS + 1, None, DECODE_ROWS).is_err());
        assert!(super::replay_bounds(10, 4, Some(0), DECODE_ROWS).is_err());
        assert_eq!(super::replay_bounds(10, 4, Some(20), DECODE_ROWS).unwrap(), 6);
        // --decode-rows 128: up to 128 rows (`--replay-check 128`), with the 64-row default prefill.
        assert_eq!(super::replay_bounds(400, WIDE_DECODE_ROWS, None, WIDE_DECODE_ROWS).unwrap(), 64);
        assert!(super::replay_bounds(400, WIDE_DECODE_ROWS + 1, None, WIDE_DECODE_ROWS).is_err());
    }

    #[test]
    fn replay_dump_is_private_and_refuses_overwrites_and_symlinks() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("dump");
        super::private_replay_dir(&dir).unwrap();
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        let file = dir.join("state.bin");
        super::private_replay_write(&file, &[1, 2]).unwrap();
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
        assert!(super::private_replay_write(&file, &[3]).is_err());
        let link = root.path().join("link");
        symlink(&dir, &link).unwrap();
        assert!(super::private_replay_dir(&link).is_err());
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(super::private_replay_dir(&dir).is_err());
        assert_eq!(super::first_differing_byte(&[1, 2], &[1, 3]), Some(1));
        assert_eq!(super::first_differing_byte(&[1], &[1, 2]), Some(1));
        assert_eq!(super::first_differing_byte(&[1], &[1]), None);
    }

    #[test]
    fn replay_dump_continuation_never_waives_recorded_emissions() {
        for continue_dump in [false, true] {
            for geometry in [false, true] {
                for serial_emissions in [false, true] {
                    assert!(!super::replay_round_valid(continue_dump, geometry, serial_emissions, false));
                    assert_eq!(super::replay_round_valid(continue_dump, geometry, serial_emissions, true),
                        continue_dump || (geometry && serial_emissions));
                }
            }
        }
    }

    #[test]
    fn replay_round_trace_checks_commit_and_request_geometry() {
        let round = serde_json::json!({"round": 1, "spec": true, "request_ids": [9], "starts": [53],
            "verify_tokens": [[2, 3, 4]], "committed_tokens": [[3, 7]], "neural_tokens": [[3, 4]],
            "finished": [false]});
        let parsed = super::parse_replay_rounds(&round.to_string(), 8, 4).unwrap();
        assert_eq!(parsed[0].starts, [53]);
        for (key, value) in [("committed_tokens", serde_json::json!([[5, 7]])),
            ("request_ids", serde_json::json!([9, 10])), ("verify_tokens", serde_json::json!([[2, 3, 8]])),
            ("committed_tokens", serde_json::json!([[]]))] {
            let mut bad = round.clone();
            bad[key] = value;
            assert!(super::parse_replay_rounds(&bad.to_string(), 8, 4).is_err(), "{key}");
        }
        assert!(super::parse_replay_rounds("", 8, 4).is_err());
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
    let mut allocator = Allocator::for_engine(engine);
    // Distinct sequences: sequence i starts i tokens later in the golden prompt.
    let mut placements = (0..count).map(|i| -> Result<GlmfPlacement> {
        let mut placement = allocator.admit(prefill + max_rows + 1)?;
        engine.prefill(&mut placement, &sequence[i..i + prefill], None)?;
        Ok(placement)
    }).collect::<Result<Vec<_>>>()?;
    let starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
    println!("verify cost, {count} distinct sequence(s) after {prefill} tokens (speculative steps, median of 7):");
    // Rows per sequence of every timed step: 1..=max_rows each while the step fits --decode-rows, and with
    // the wide programs the verify budget's own step where it is not a multiple of the sequences (the
    // remainder's rows go to the first sequences: 15 x 8 + 7 rows at 16 sequences on an RTX 5090).
    let mut shapes: Vec<Vec<usize>> = (1..=max_rows).take_while(|&rows| count * rows <= engine.decode_rows)
        .map(|rows| vec![rows; count]).collect();
    let budget = engine.verify_rows;
    if budget > super::engine::DECODE_ROWS && budget % count != 0 && budget / count < max_rows {
        shapes.push((0..count).map(|i| budget / count + usize::from(i < budget % count)).collect());
    }
    for shape in shapes {
        let total: usize = shape.iter().sum();
        let tokens: Vec<u32> = shape.iter().enumerate()
            .flat_map(|(i, &rows)| sequence[i + prefill..i + prefill + rows].iter().copied()).collect();
        let mut times = Vec::new();
        *engine.profile.borrow_mut() = [0.0; 3];
        for round in 0..9 {
            for (placement, &start) in placements.iter_mut().zip(&starts) {
                placement.len = start;
            }
            let mut step: Vec<(&mut GlmfPlacement, usize)> = placements.iter_mut().zip(&shape)
                .map(|(p, &rows)| (p, rows)).collect();
            let timer = Instant::now();
            engine.verify_spec(&mut step, &tokens)?;
            if round >= 2 {
                times.push(timer.elapsed().as_secs_f64());
            } else {
                *engine.profile.borrow_mut() = [0.0; 3];
                engine.op_profile()?;
            }
        }
        times.sort_by(f64::total_cmp);
        let phases = std::mem::take(&mut *engine.profile.borrow_mut());
        let n = times.len() as f64;
        let per_sequence = if shape.iter().all(|&rows| rows == shape[0]) { shape[0].to_string() }
            else { format!("{}..{}", shape[count - 1], shape[0]) };
        println!("  {per_sequence} rows/sequence ({total} rows): {:.2} ms (min {:.2}); GPU until exchanges {:.2} ms, \
            Spark exchanges {:.2} ms, head {:.2} ms", 1e3 * times[times.len() / 2], 1e3 * times[0], 1e3 * phases[0] / n,
            1e3 * phases[1] / n, 1e3 * phases[2] / n);
        super::print_op_profile(engine, &format!("  verify ops per step, {total} rows"), times.len())?;
    }
    Ok(())
}

#[cfg(test)]
mod draft_mode_tests {
    use super::*;
    use crate::families::glm5::DraftHead;
    use crate::shared::fp8_linear::Fp8Rows;

    #[test]
    fn draft_modes_compare_both_heads_under_every_admitted_linear_mode() {
        let all = draft_mode_settings(true, Fp8Rows::W8a8);
        assert_eq!(all, [(DraftHead::Exact, Fp8Rows::W8a16), (DraftHead::Tensor, Fp8Rows::W8a16),
            (DraftHead::Exact, Fp8Rows::Wide), (DraftHead::Tensor, Fp8Rows::Wide),
            (DraftHead::Exact, Fp8Rows::W8a8), (DraftHead::Tensor, Fp8Rows::W8a8)]);
        assert_eq!(draft_mode_settings(true, Fp8Rows::Wide).len(), 4);
        assert_eq!(draft_mode_settings(true, Fp8Rows::W8a16), [(DraftHead::Exact, Fp8Rows::W8a16),
            (DraftHead::Tensor, Fp8Rows::W8a16)]);
        // A BF16 drafter has no FP8 GEMMs: the head alone.
        assert_eq!(draft_mode_settings(false, Fp8Rows::W8a8), draft_mode_settings(true, Fp8Rows::W8a16));
    }

    /// A phase of [`draft_mode_schedule`] and the drafter mode it ran under.
    #[derive(Debug, Clone, PartialEq)]
    enum Phase {
        Prefill { tokens: usize },
        Advance { first: usize, rows: usize },
        Draft { sequences: usize },
    }

    /// Records the mode active at every phase; drafts are the anchor repeated, so tallies count.
    struct Recorder {
        mode: DraftMode,
        log: Vec<(Phase, DraftMode)>,
    }

    impl DraftModeSteps for Recorder {
        fn set_mode(&mut self, mode: DraftMode) -> Result<()> {
            self.mode = mode;
            Ok(())
        }
        fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
            self.log.push((Phase::Prefill { tokens: tokens.len() }, self.mode));
            Ok(())
        }
        fn advance(&mut self, first: usize, tokens: &[u32]) -> Result<()> {
            self.log.push((Phase::Advance { first, rows: tokens.len() }, self.mode));
            Ok(())
        }
        fn draft(&mut self, sequences: &[DraftSeq]) -> Result<Vec<Vec<u32>>> {
            self.log.push((Phase::Draft { sequences: sequences.len() }, self.mode));
            Ok(sequences.iter().map(|s| vec![s.anchor; 7]).collect())
        }
    }

    /// The ring is filled under the reference (exact head, W8A16) only: before the first window
    /// the drafter is in its load's mode (here tensor / W8A8), and after each window's settings
    /// the last one (tensor / W8A8) would otherwise project the next window's context.
    #[test]
    fn draft_modes_build_every_context_under_the_reference_mode() {
        let settings = draft_mode_settings(true, Fp8Rows::W8a8);
        let mut steps = Recorder { mode: (DraftHead::Tensor, Fp8Rows::W8a8), log: Vec::new() };
        let tokens: Vec<u32> = (0..64).collect();
        // Prefill 10, windows of 3 anchors from 10 to 22, target steps of at most 2 rows.
        let tally = draft_mode_schedule(&mut steps, &tokens, 10, 22, 3, 2, &settings).unwrap();
        assert_eq!(tally.anchors, 12);
        assert_eq!(tally.settings.len(), settings.len());
        let mut log = steps.log.iter();
        assert_eq!(log.next(), Some(&(Phase::Prefill { tokens: 10 }, REFERENCE_MODE)));
        let mut done = 10;
        for p in (10..22).step_by(3) {
            // Context up to p + 1 (the window's last anchor reads up to p + 1), in steps of 2.
            while done + 1 < p + 3 {
                let rows = (p + 2 - done).min(2);
                assert_eq!(log.next(), Some(&(Phase::Advance { first: done, rows }, REFERENCE_MODE)), "window {p}");
                done += rows;
            }
            for _ in 0..3 {
                assert_eq!(log.next(), Some(&(Phase::Draft { sequences: 1 }, REFERENCE_MODE)), "window {p}");
            }
            for &mode in &settings {
                assert_eq!(log.next(), Some(&(Phase::Draft { sequences: 3 }, mode)), "window {p}");
            }
        }
        assert_eq!(log.next(), None);
        // Every phase that writes the ring ran under the reference.
        assert!(steps.log.iter().all(|(phase, mode)| matches!(phase, Phase::Draft { .. }) || *mode == REFERENCE_MODE));
        // A schedule whose first setting is not the reference is refused.
        assert!(draft_mode_schedule(&mut steps, &tokens, 10, 22, 3, 2, &settings[1..]).is_err());
    }
}
