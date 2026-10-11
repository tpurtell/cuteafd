//! `v41-golden` (`cuteafd golden --family deepseek_v41`): teacher-forced scoring of a
//! golden prompt through the serving worker's scoring probe. The worker runs as
//! `serve-native` does, with no HTTP listener; one cold, speculation-free probed
//! request scores every row from `--score-from`, then the worker drains and stops.
//! The golden directory holds `tokens.bin` (u32/i32 LE) and, optionally,
//! `logits.bin` (F32 `[T, vocab]`) from `python/reference/families/deepseek_v41/golden.py`.
//! The probe hashes every scored row's raw F32 logits (FNV-1a); the digest folds
//! those hashes in position order, so equal digests are byte-identical runs.
use super::v41_native_serve::scores::VOCAB;
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::probe::{Probe, ProbeSpec};
use cuteafd_api::openai::{InferenceChunk, NativeRequest};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub serve: crate::cli::NativeServeArgs,
    /// Directory with tokens.bin and (optionally) logits.bin from golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// First scored position; every row predicting `score_from..len` is recorded.
    #[arg(long, default_value_t = 1)]
    pub score_from: usize,
    /// Score only the first N golden tokens.
    #[arg(long)]
    pub tokens: Option<usize>,
    /// Scoring kernel shape: `prefill` (48-row prefill continuation, the legacy
    /// V4.1 path) or `decode` (teacher-forced verify waves of --verify-rows).
    #[arg(long, value_parser = ["prefill", "decode"], default_value = "prefill")]
    pub score_path: String,
    #[arg(long)]
    pub verify_rows: Option<usize>,
    /// Save the scored rows' F32 logits here (`[rows, vocab]`, little endian).
    #[arg(long)]
    pub save_logits: Option<PathBuf>,
    /// Prefix-cache restore check instead of scoring: for each P, serve `tokens[..P]` (retained
    /// as a prompt snapshot), repeat it (an exact hit: every generated row must be byte-identical),
    /// then continue it by a short (<128) and a long (>=128) suffix of the golden tokens (exact
    /// ancestor restores, compared with the same request served cold) and branch it at P - 37
    /// (a partial, approximate replay: generated rows' KL and top-1 against cold). Comma separated.
    #[arg(long, value_delimiter = ',')]
    pub resume_at: Vec<usize>,
    /// Generated rows recorded per request in --resume-at.
    #[arg(long, default_value_t = 16)]
    pub resume_rows: usize,
}

/// Rows the probe records, in position order: each row's log-probabilities
/// and the hash of its raw logits.
struct Rows {
    positions: Vec<usize>,
    logits: Vec<f32>,
    hashes: Vec<String>,
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || golden(args)).await?
}

fn read_tokens(args: &GoldenArgs) -> Result<Vec<u32>> {
    let bytes = std::fs::read(args.golden.join("tokens.bin"))
        .with_context(|| format!("reading {}", args.golden.join("tokens.bin").display()))?;
    ensure!(bytes.len() % 4 == 0, "golden tokens must be 4-byte ids");
    let mut tokens: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(n) = args.tokens { tokens.truncate(n); }
    ensure!(tokens.len() >= 2 && tokens.iter().all(|&t| (t as usize) < VOCAB), "invalid golden tokens");
    Ok(tokens)
}

fn golden(args: GoldenArgs) -> Result<()> {
    let tokens = read_tokens(&args)?;
    if !args.resume_at.is_empty() {
        return resume::check(args.serve, &tokens, &args.resume_at, args.resume_rows);
    }
    ensure!((1..tokens.len()).contains(&args.score_from), "--score-from must be inside the golden prompt");
    let GoldenArgs { serve, golden, score_from, score_path, verify_rows, save_logits, .. } = args;
    let dump = tempdir()?;
    let spec = ProbeSpec {
        cold: true,
        no_speculation: true,
        prompt_ids: Some(tokens.clone()),
        score_from: Some(score_from),
        score_path: Some(score_path.clone()),
        verify_rows,
        dump_rows: Some(dump.join("rows")),
        top_k: 1,
        ..ProbeSpec::default()
    };
    let started = std::time::Instant::now();
    let rows = score(serve, spec, &dump);
    let _ = std::fs::remove_dir_all(&dump);
    let rows = rows?;
    let seconds = started.elapsed().as_secs_f64();
    let expected: Vec<usize> = (score_from..tokens.len()).collect();
    ensure!(rows.positions == expected, "scored positions differ from {score_from}..{}", tokens.len());
    report(&Report { golden, score_path, save_logits }, &tokens, &rows, seconds)
}

/// The outputs `report` writes and where the reference lives.
struct Report {
    golden: PathBuf,
    score_path: String,
    save_logits: Option<PathBuf>,
}

/// A private scratch directory for the probe's row dump (the dump leaf must not exist).
fn tempdir() -> Result<PathBuf> {
    let base = std::env::temp_dir().join(format!("cuteafd-v41-golden-{}", std::process::id()));
    ensure!(!base.exists(), "{} already exists", base.display());
    std::fs::create_dir_all(&base)?;
    Ok(base)
}

/// Start the serving worker, submit one scoring request and collect its rows.
fn score(serve: crate::cli::NativeServeArgs, spec: ProbeSpec, dump: &std::path::Path) -> Result<Rows> {
    let probe = Probe::new(spec);
    let (queue, worker) = super::v41_native_serve::start_worker(serve)?;
    let result = (|| -> Result<()> {
        let (events, mut receive) = tokio::sync::mpsc::unbounded_channel();
        let request = NativeRequest {
            prompt: String::new(),
            constraint: None,
            images: Vec::new(),
            media: Vec::new(),
            audio: Vec::new(),
            max_tokens: 1,
            sampling: cuteafd_core::TargetSamplingParams::greedy(),
            stop_token_ids: Vec::new(),
            events,
            probe: Some(Arc::clone(&probe)),
            usage: None,
        };
        queue.blocking_send(request).map_err(|_| anyhow::anyhow!("V4.1 worker stopped before scoring"))?;
        let mut finished = false;
        while let Some(event) = receive.blocking_recv() {
            match event {
                Ok(InferenceChunk::Finish { .. }) => finished = true,
                Ok(_) => {}
                Err(failure) => anyhow::bail!("scoring request failed: {failure}"),
            }
        }
        ensure!(finished, "scoring request ended without finishing");
        Ok(())
    })();
    drop(queue);
    let joined = worker.join();
    result?;
    joined?;
    let record = probe.record();
    ensure!(record.error.is_none(), "probe error: {}", record.error.unwrap_or_default());
    ensure!(record.engine.as_deref() == Some("deepseek_v41") && record.cold && record.no_speculation
        && record.cached_tokens == 0, "the worker did not honour the cold, speculation-free scoring probe");
    let mut rows = read_dump(&dump.join("rows"))?;
    let mut recorded: Vec<_> = record.rows.iter().map(|row| (row.position, row.hash.clone())).collect();
    recorded.sort();
    ensure!(recorded.iter().map(|r| r.0).eq(rows.positions.iter().copied()), "probe rows and row dump differ");
    rows.hashes = recorded.into_iter().map(|r| r.1).collect();
    Ok(rows)
}

/// The probe's row dump stores log-softmax F32 rows; read them back in position order.
fn read_dump(directory: &std::path::Path) -> Result<Rows> {
    let manifest = std::fs::read_to_string(directory.join("manifest.jsonl"))?;
    let mut entries: Vec<(usize, PathBuf)> = Vec::new();
    for line in manifest.lines() {
        let row: serde_json::Value = serde_json::from_str(line)?;
        ensure!(row["vocab_size"].as_u64() == Some(VOCAB as u64), "dumped row vocabulary differs");
        entries.push((row["position"].as_u64().context("row position")? as usize,
            directory.join(row["file"].as_str().context("row file")?)));
    }
    entries.sort_by_key(|(position, _)| *position);
    let mut rows = Rows { positions: Vec::with_capacity(entries.len()), logits: Vec::with_capacity(entries.len() * VOCAB),
        hashes: Vec::new() };
    for (position, path) in entries {
        let bytes = std::fs::read(&path)?;
        let header = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
        let data = &bytes[8 + header..];
        ensure!(data.len() == VOCAB * 4, "dumped row {position} extent differs");
        rows.positions.push(position);
        rows.logits.extend(data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
    }
    Ok(rows)
}

fn report(args: &Report, tokens: &[u32], rows: &Rows, seconds: f64) -> Result<()> {
    let digest = cuteafd_api::openai::probe::fnv1a(rows.hashes.concat().as_bytes());
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    // Rows are log-probabilities, so the NLL is the negated entry of the next token.
    let (mut nll, mut next_ok) = (0f64, 0usize);
    for (row, &position) in rows.logits.chunks_exact(VOCAB).zip(&rows.positions) {
        let next = tokens[position] as usize;
        nll -= f64::from(row[next]);
        next_ok += usize::from(argmax(row) == next);
    }
    let n = rows.positions.len();
    println!("v41 golden: {} rows scored ({} path) in {seconds:.1} s | mean NLL engine {:.4} | next-token accuracy {:.1}% \
        | logits digest {digest:016x}", n, args.score_path, nll / n as f64, 100.0 * next_ok as f64 / n as f64);
    if let Ok(golden) = std::fs::read(args.golden.join("logits.bin")) {
        ensure!(golden.len() % (VOCAB * 4) == 0, "golden logits are not [T, vocab] F32");
        let golden: Vec<f32> = golden.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        // golden.py writes the row predicting token t at index t - 1.
        let (mut agree, mut kl, mut golden_nll) = (0usize, 0f64, 0f64);
        for (ours, &position) in rows.logits.chunks_exact(VOCAB).zip(&rows.positions) {
            let theirs = golden.get((position - 1) * VOCAB..position * VOCAB).context("golden logits are short")?;
            let top = theirs.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = top + theirs.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln();
            agree += usize::from(argmax(ours) == argmax(theirs));
            golden_nll += lse - theirs[tokens[position] as usize] as f64;
            kl += theirs.iter().zip(ours).map(|(&g, &q)| {
                let p = g as f64 - lse;
                p.exp() * (p - q as f64)
            }).sum::<f64>();
        }
        println!("v41 golden vs reference: top-1 agreement {:.2}% over {n} rows | mean NLL golden {:.4} \
            | mean KL(golden||engine) {:.5}", 100.0 * agree as f64 / n as f64, golden_nll / n as f64, kl / n as f64);
    }
    if let Some(path) = &args.save_logits {
        std::fs::write(path, rows.logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    Ok(())
}

#[path = "v41_golden/resume.rs"]
mod resume;

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn golden_takes_the_serving_options_and_defaults_to_the_prefill_path() {
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "v41-golden", "--snapshot", "/m",
            "--native-lib", "/lib.so", "--peers", "127.0.0.1:9,127.0.0.1:10,127.0.0.1:11,127.0.0.1:12",
            "--golden", "/g", "--score-from", "17", "--dspark"]).unwrap();
        let crate::cli::Commands::V41Golden(args) = cli.command else { panic!("v41-golden") };
        assert_eq!((args.score_from, args.score_path.as_str(), args.serve.peers.len()), (17, "prefill", 4));
        assert!(args.serve.dspark);
        assert!(crate::cli::Cli::try_parse_from(["cuteafd", "v41-golden", "--snapshot", "/m",
            "--native-lib", "/lib.so", "--golden", "/g", "--score-path", "other"]).is_err());
    }
}
