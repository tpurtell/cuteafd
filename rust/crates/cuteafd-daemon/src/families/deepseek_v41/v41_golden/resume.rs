//! `v41-golden --resume-at`: the prefix cache's byte-exact restore gate, in process, through the
//! serving worker and its probe (no HTTP). Each case is a sequence of greedy, speculation-free
//! probed requests on golden token ids:
//! - **exact repeat**: `tokens[..P]` twice. The second must report `P` cached tokens and
//!   reproduce every recorded generated row byte for byte (the first token from the retained
//!   row, the rest decoded from the restored state).
//! - **short / long continuation**: the first request's prompt, its generated tokens, then a
//!   suffix of golden tokens shorter than 128 (exact mark restore) or at least 128 (encoder
//!   continuation). Served cached and cold; rows are compared byte for byte (informational:
//!   a suffix prefill chunks differently from a cold prefill of the whole prompt, so FP
//!   reordering may differ) and by KL.
//! - **partial branch**: `tokens[..P - 37]` plus a fresh suffix, a partial match of the first
//!   snapshot (approximate replay). Generated rows against cold: mean KL and top-1 agreement.
//!   The gate compares these against the same numbers from work/p0.
//! Odd and even `P` exercise the ratio-two compressor carry; the report names each case's
//! parity.
use super::*;

const RECORD: usize = 16;

struct Run {
    cached: usize,
    generated: Vec<u32>,
    hashes: Vec<String>,
    rows: Vec<f32>,
}

/// Serve `prompt` once and collect what its probe recorded.
fn run(queue: &tokio::sync::mpsc::Sender<NativeRequest>, prompt: &[u32], rows: usize, cold: bool) -> Result<Run> {
    let dump = tempdir()?;
    let spec = ProbeSpec {
        cold,
        no_speculation: true,
        prompt_ids: Some(prompt.to_vec()),
        record_rows: rows,
        dump_rows: Some(dump.join("rows")),
        top_k: 1,
        ..ProbeSpec::default()
    };
    let probe = Probe::new(spec);
    let (events, mut receive) = tokio::sync::mpsc::unbounded_channel();
    let request = NativeRequest {
        prompt: String::new(), constraint: None, images: Vec::new(), media: Vec::new(), audio: Vec::new(),
        max_tokens: rows, sampling: cuteafd_core::TargetSamplingParams::greedy(), stop_token_ids: Vec::new(),
        events, probe: Some(Arc::clone(&probe)), usage: None,
    };
    queue.blocking_send(request).map_err(|_| anyhow::anyhow!("V4.1 worker stopped"))?;
    let mut finished = false;
    while let Some(event) = receive.blocking_recv() {
        match event {
            Ok(InferenceChunk::Finish { .. }) => finished = true,
            Ok(_) => {}
            Err(failure) => anyhow::bail!("probed request failed: {failure}"),
        }
    }
    ensure!(finished, "probed request ended without finishing");
    let record = probe.record();
    ensure!(record.error.is_none(), "probe error: {}", record.error.unwrap_or_default());
    let mut rows_dump = read_dump(&dump.join("rows"))?;
    let _ = std::fs::remove_dir_all(&dump);
    let mut recorded: Vec<_> = record.rows.iter().map(|row| (row.position, row.hash.clone())).collect();
    recorded.sort();
    ensure!(recorded.iter().map(|r| r.0).eq(rows_dump.positions.iter().copied()), "probe rows and row dump differ");
    rows_dump.hashes = recorded.into_iter().map(|r| r.1).collect();
    Ok(Run { cached: record.cached_tokens, generated: record.generated, hashes: rows_dump.hashes, rows: rows_dump.logits })
}

/// Mean KL(reference || candidate) and top-1 agreement over two runs' rows (log-probabilities).
fn compare(reference: &Run, candidate: &Run) -> (f64, f64, usize) {
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    let (mut kl, mut agree, mut n) = (0f64, 0usize, 0usize);
    for (p, q) in reference.rows.chunks_exact(VOCAB).zip(candidate.rows.chunks_exact(VOCAB)) {
        kl += p.iter().zip(q).map(|(&p, &q)| (p as f64).exp() * (p as f64 - q as f64)).sum::<f64>();
        agree += usize::from(argmax(p) == argmax(q));
        n += 1;
    }
    (kl / n.max(1) as f64, agree as f64 / n.max(1) as f64, n)
}

pub(super) fn check(serve: crate::cli::NativeServeArgs, tokens: &[u32], at: &[usize], rows: usize) -> Result<()> {
    let rows = if rows == 0 { RECORD } else { rows.min(256) };
    let (queue, worker) = super::super::v41_native_serve::start_worker(serve)?;
    let result = (|| -> Result<bool> {
        let mut passed = true;
        for &p in at {
            ensure!(p > 40 && p + 200 < tokens.len(), "--resume-at {p} needs 40 < P and P + 200 < {} golden tokens",
                tokens.len());
            let parity = if p % 2 == 1 { "odd" } else { "even" };
            let prompt = &tokens[..p];
            let first = run(&queue, prompt, rows, false)?;
            let repeat = run(&queue, prompt, rows, false)?;
            let exact = repeat.cached == p && repeat.hashes == first.hashes && repeat.generated == first.generated;
            passed &= exact;
            println!("resume at {p} ({parity}): exact repeat cached {}/{p} | {} of {} rows byte-identical | {}",
                repeat.cached, repeat.hashes.iter().zip(&first.hashes).filter(|(a, b)| a == b).count(),
                first.hashes.len(), if exact { "PASS" } else { "FAIL" });
            // Continue the turn snapshot (prompt + generated) by a short and a long suffix.
            let mut turn = prompt.to_vec();
            turn.extend(&first.generated);
            for (name, suffix) in [("short", 37usize), ("long", 160)] {
                let mut next = turn.clone();
                next.extend(&tokens[p..p + suffix]);
                let cached = run(&queue, &next, rows, false)?;
                let cold = run(&queue, &next, rows, true)?;
                let (kl, top1, n) = compare(&cold, &cached);
                let identical = cached.hashes == cold.hashes;
                let reused = cached.cached >= turn.len();
                passed &= reused;
                println!("resume at {p} ({parity}): {name} continuation (+{suffix}) cached {}/{} (turn {}) | rows vs cold \
                    {} | KL {kl:.3e} top-1 {:.1}% over {n} rows | {}", cached.cached, next.len(), turn.len(),
                    if identical { "byte-identical" } else { "differ (suffix chunking; informational)" }, 100.0 * top1,
                    if reused { "PASS" } else { "FAIL: did not reuse the turn" });
            }
            // A branch inside the first prompt: a partial match, approximate replay.
            let mut branch = prompt[..p - 37].to_vec();
            branch.extend(&tokens[p + 1..p + 1 + 64]);
            let partial = run(&queue, &branch, rows, false)?;
            let cold = run(&queue, &branch, rows, true)?;
            let (kl, top1, n) = compare(&cold, &partial);
            println!("resume at {p} ({parity}): partial branch at {} cached {} | approximate replay vs cold: mean KL \
                {kl:.4e} | top-1 {:.1}% over {n} rows | NLL-proxy (partial) {:.4}", p - 37, partial.cached, 100.0 * top1,
                nll(&partial, &cold));
        }
        Ok(passed)
    })();
    drop(queue);
    let joined = worker.join();
    let passed = result?;
    joined?;
    ensure!(passed, "prefix restore check failed");
    println!("v41 golden --resume-at: PASS");
    Ok(())
}

/// Mean negative log-probability the candidate's rows give the reference's greedy tokens.
fn nll(candidate: &Run, reference: &Run) -> f64 {
    let n = candidate.rows.len() / VOCAB;
    let total: f64 = candidate.rows.chunks_exact(VOCAB).zip(&reference.generated)
        .map(|(row, &token)| -f64::from(row[token as usize])).sum();
    total / n.max(1) as f64
}
