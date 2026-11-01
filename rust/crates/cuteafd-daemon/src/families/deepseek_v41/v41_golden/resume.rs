//! In-process exact-state gates through the serving worker. Hashes cover initialized
//! source rows, window rings, odd compressor carry and Engram history. Generated
//! logits alone are not evidence of byte-exact persistent state restoration.
use super::*;

const RECORD: usize = 16;

struct Run {
    cached: usize,
    generated: Vec<u32>,
    hashes: Vec<String>,
    rows: Vec<f32>,
    record: cuteafd_api::openai::probe::ProbeRecord,
}

enum Requests {
    Local(tokio::sync::mpsc::Sender<NativeRequest>),
    Remote(cuteafd_bench::client::Client),
}

const IMAGE_BYTES: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 8, 14, 14, 6, 0, 1, 246, 0, 250, 253, 142, 150, 138, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];
const IMAGE_URL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMIDg4GAAH2APr9jpaKAAAAAElFTkSuQmCC";

/// Serve `prompt` once and collect what its probe recorded. `output_dependent`
/// marks a prompt that extends the model's own generated tokens: matched A/B
/// pairs compare only input-determined prompts.
fn run(queue: &Requests, prompt: &[u32], rows: usize, cold: bool, output_dependent: bool,
    image: bool, cold_steps: Vec<cuteafd_api::openai::probe::ProbeColdStep>) -> Result<Run> {
    let dump = tempdir()?;
    let mut spec = ProbeSpec {
        cold,
        no_speculation: true,
        prompt_ids: Some(prompt.to_vec()),
        record_rows: rows,
        record_state: true,
        cold_steps,
        dump_rows: Some(dump.join("rows")),
        top_k: 1,
        output_dependent,
        ..ProbeSpec::default()
    };
    if let Requests::Remote(client) = queue {
        spec.dump_rows = None;
        spec.top_k = VOCAB;
        let content = if image { serde_json::json!([
            {"type": "text", "text": "prefix restore gate"},
            {"type": "image_url", "image_url": {"url": IMAGE_URL}}
        ]) } else { serde_json::json!("prefix restore gate") };
        let value = client.media_probe(serde_json::json!({"messages": [{"role": "user", "content": content}],
            "max_tokens": rows, "temperature": 0}), serde_json::to_value(spec)?)?;
        let mut record: cuteafd_api::openai::probe::ProbeRecord = serde_json::from_value(value["probe"].clone())?;
        ensure!(record.error.is_none(), "remote probe error: {:?}", record.error);
        record.rows.sort_by_key(|row| row.position);
        let mut logits = Vec::new();
        for row in &record.rows {
            ensure!(row.finite && row.top.len() == VOCAB, "remote probe lacks complete finite rows");
            let start = logits.len();
            logits.resize(start + VOCAB, f32::NEG_INFINITY);
            for &(token, probability) in &row.top { logits[start + token as usize] = probability; }
        }
        std::fs::remove_dir_all(&dump)?;
        return Ok(Run { cached: record.cached_tokens, generated: record.generated.clone(),
            hashes: record.rows.iter().map(|row| row.hash.clone()).collect(), rows: logits, record });
    }
    let probe = Probe::new(spec);
    let (events, mut receive) = tokio::sync::mpsc::unbounded_channel();
    let images = if image {
        vec![cuteafd_loader::V41Image::decode(IMAGE_BYTES)?]
    } else { Vec::new() };
    let request = NativeRequest {
        prefix_cache_source: Default::default(), finish_detail: Default::default(), literal_spans: Vec::new(), prompt: String::new(), constraint: None, images, media: Vec::new(), audio: Vec::new(),
        max_tokens: rows, sampling: cuteafd_core::TargetSamplingParams::greedy(), stop_token_ids: Vec::new(),
        events, probe: Some(Arc::clone(&probe)), usage: None,
    };
    let Requests::Local(queue) = queue else { unreachable!() };
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
    probe.wait_state()?;
    let record = probe.record();
    ensure!(record.error.is_none(), "probe error: {}", record.error.as_deref().unwrap_or_default());
    let mut rows_dump = read_dump(&dump.join("rows"))?;
    let _ = std::fs::remove_dir_all(&dump);
    let mut recorded: Vec<_> = record.rows.iter().map(|row| (row.position, row.hash.clone())).collect();
    recorded.sort();
    ensure!(recorded.iter().map(|r| r.0).eq(rows_dump.positions.iter().copied()), "probe rows and row dump differ");
    rows_dump.hashes = recorded.into_iter().map(|r| r.1).collect();
    Ok(Run { cached: record.cached_tokens, generated: record.generated.clone(), hashes: rows_dump.hashes, rows: rows_dump.logits, record })
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

fn state<'a>(run: &'a Run, kind: &str) -> Result<&'a cuteafd_api::openai::probe::ProbeState> {
    run.record.states.iter().find(|state| state.kind == kind)
        .with_context(|| format!("missing {kind} state diagnostic"))
}

fn exact(cases: &mut Vec<serde_json::Value>, a: &Run, a_kind: &str, b: &Run, b_kind: &str, label: &str) -> Result<()> {
    let (a, b) = (state(a, a_kind)?, state(b, b_kind)?);
    cases.push(serde_json::json!({"name": label, "required": true, "reference": a, "candidate": b,
        "pass": a.end == b.end && a.bytes == b.bytes && a.sha256 == b.sha256}));
    ensure!(a.end == b.end && a.bytes == b.bytes && a.sha256 == b.sha256,
        "{label}: persistent state differs at {} / {} ({} / {} bytes, {} / {})",
        a.end, b.end, a.bytes, b.bytes, a.sha256, b.sha256);
    println!("{label}: byte-exact state at {} ({} initialized bytes) PASS", a.end, a.bytes);
    Ok(())
}

/// Cold recomputation can vary with unordered Spark prefill reductions. Record
/// that noise separately; exact restore always compares with its captured state.
fn cold_evidence(cases: &mut Vec<serde_json::Value>, a: &Run, a_kind: &str,
    b: &Run, b_kind: &str, label: &str) -> Result<()> {
    let (a, b) = (state(a, a_kind)?, state(b, b_kind)?);
    let equal = a.end == b.end && a.bytes == b.bytes && a.sha256 == b.sha256;
    cases.push(serde_json::json!({"name": label, "required": false,
        "reference": a, "candidate": b, "equal": equal}));
    println!("{label}: {} (informational prefill reproducibility)",
        if equal { "byte-exact" } else { "cold recompute differs" });
    Ok(())
}

pub(super) fn check(serve: crate::cli::NativeServeArgs, tokens: &[u32], at: &[usize], rows: usize,
    require_host: bool, url: Option<&str>, output: Option<&std::path::Path>) -> Result<()> {
    use cuteafd_api::openai::probe::ProbeColdStep;
    let rows = if rows == 0 { RECORD } else { rows.clamp(2, 256) };
    let config = std::env::var_os("CUTEAFD_SMOKE_CONFIG").map(std::fs::read_to_string).transpose()?;
    let config_value = |key: &str| config.as_deref().and_then(|text| text.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        (name.trim() == key).then(|| value.trim())
    }));
    let entries = config_value("PREFIX_CACHE_ENTRIES").map(str::parse::<usize>).transpose()?
        .unwrap_or(serve.prefix_cache_entries as usize);
    ensure!(entries > 0, "--resume-at requires a device prefix cache");
    let (queue, worker) = if let Some(url) = url {
        use cuteafd_api::openai::auth::ApiKey;
        let key = if let Some(path) = serve.api.api_key_file.as_deref() {
            Some(ApiKey::from_file(path)?)
        } else if let Some(path) = config_value("API_KEY_FILE") {
            Some(ApiKey::from_file(std::path::Path::new(path))?)
        } else {
            std::env::var("CUTEAFD_API_KEY").ok().map(|key| ApiKey::new(&key)).transpose()
                .map_err(anyhow::Error::msg)?
        };
        let mut client = cuteafd_bench::client::Client::new(url, None,
            Arc::new(std::sync::atomic::AtomicBool::new(false))).with_api_key(key);
        client.discover()?;
        (Requests::Remote(client), None)
    } else {
        let (queue, worker) = super::super::v41_native_serve::start_worker(serve)?;
        (Requests::Local(queue), Some(worker))
    };
    let mut cases = Vec::new();
    let result = (|| -> Result<()> {
        let namespace_stride = entries.saturating_mul(2).saturating_add(3);
        ensure!(at.len().saturating_mul(2).saturating_mul(namespace_stride) < VOCAB - 1000,
            "too many resume cases for distinct pressure namespaces");
        for (case_index, &p) in at.iter().enumerate() {
            ensure!(p > 40 && p + 200 < tokens.len(), "--resume-at {p} needs 40 < P and P + 200 < {} golden tokens",
                tokens.len());
            for image in [false, true] {
                let mut prompt = tokens[..p].to_vec();
                // Each case starts cold even when a previous case retained a shorter ancestor.
                let namespace = 1000 + (case_index * 2 + usize::from(image)) * namespace_stride;
                prompt[0] = namespace as u32;
                if image { prompt[8] = cuteafd_loader::V41_IMAGE_TOKEN_ID; }
                let label = format!("resume P={p} {}", if image { "image" } else { "text" });
                let first = run(&queue, &prompt, rows, false, false, image, Vec::new())?;
                let cold = run(&queue, &prompt, rows, true, false, image, Vec::new())?;
                cold_evidence(&mut cases, &first, "prompt", &cold, "prompt", &format!("{label} cold same-chunk"))?;
                let repeat = run(&queue, &prompt, rows, false, false, image, Vec::new())?;
                ensure!(repeat.cached == first.record.prompt_ids.len(), "{label}: prompt cache miss");
                exact(&mut cases, &first, "prompt", &repeat, "restore", &format!("{label} prompt restore"))?;
                exact(&mut cases, &first, "prompt", &repeat, "prompt", &format!("{label} prompt remains restored"))?;
                let decode_exact = repeat.hashes == first.hashes && repeat.generated == first.generated;
                cases.push(serde_json::json!({"name": format!("{label} decode after restore"),
                    "required": true, "rows": repeat.hashes.len(), "pass": decode_exact}));
                ensure!(decode_exact && !repeat.hashes.is_empty(),
                    "{label}: exact repeat generated rows differ or were not recorded");

                let prompt_end = first.record.prompt_ids.len();
                let turn_end = state(&first, "turn")?.end;
                ensure!(turn_end >= prompt_end && turn_end <= prompt_end + first.generated.len(),
                    "{label}: committed turn frontier outside emitted tokens");
                let mut turn = prompt.clone();
                turn.extend(&first.generated[..turn_end - prompt_end]);
                let mut next = turn.clone();
                next.extend(&tokens[p..p + 37]);
                let cached = run(&queue, &next, rows, false, true, image, Vec::new())?;
                ensure!(cached.cached == turn_end, "{label}: did not reuse turn frontier {turn_end}");
                exact(&mut cases, &first, "turn", &cached, "restore", &format!("{label} turn restore"))?;
                let mut steps = vec![ProbeColdStep { end: prompt_end, decode: false }];
                steps.extend((prompt_end + 1..=turn_end).map(|end| ProbeColdStep { end, decode: true }));
                steps.push(ProbeColdStep { end: turn_end + 37, decode: false });
                let cold_turn = run(&queue, &next, rows, true, true, image, steps)?;
                cold_evidence(&mut cases, &first, "turn", &cold_turn, "checkpoint", &format!("{label} turn vs cold same-shape"))?;
                cold_evidence(&mut cases, &cached, "prompt", &cold_turn, "prompt", &format!("{label} continuation vs cold same-shape"))?;

                let mut long = turn;
                long.extend(&tokens[p..p + 160]);
                let continued = run(&queue, &long, rows, false, true, image, Vec::new())?;
                ensure!(continued.cached >= turn_end, "{label}: encoder continuation missed turn snapshot");
                println!("{label}: long continuation cached {}/{} PASS", continued.cached, continued.record.prompt_ids.len());

                if require_host {
                    // Capacity pressure through ordinary requests exercises the shared host tier;
                    // no private eviction or promotion hook bypasses engine admission.
                    for n in 0..entries.saturating_mul(2).saturating_add(2) {
                        let mut other = prompt.clone();
                        // Pressure leaves must not themselves hit a previous case's host bank.
                        other[0] = (namespace + n + 1) as u32;
                        run(&queue, &other, 2, false, false, image, Vec::new())?;
                    }
                    let promoted = run(&queue, &prompt, rows, false, false, image, Vec::new())?;
                    ensure!(promoted.record.provenance.as_ref().and_then(|v| v.get("prefix_restore"))
                        .and_then(|v| v.get("host")).and_then(|v| v.as_bool()) == Some(true),
                        "{label}: expected a host promotion, not a device hit or cold miss: {:?}",
                        promoted.record.provenance);
                    exact(&mut cases, &first, "prompt", &promoted, "restore", &format!("{label} host round trip"))?;
                }

                if !image {
                    let mut branch = prompt[..p - 37].to_vec();
                    branch.extend(&tokens[p + 1..p + 1 + 64]);
                    let partial = run(&queue, &branch, rows, false, false, false, Vec::new())?;
                    let cold = run(&queue, &branch, rows, true, false, false, Vec::new())?;
                    let (kl, top1, n) = compare(&cold, &partial);
                    ensure!(n > 0, "{label}: approximate replay recorded no rows");
                    cases.push(serde_json::json!({"name": format!("{label} approximate replay"),
                        "required": false, "cached_tokens": partial.cached, "rows": n,
                        "kl": kl, "top1": top1, "nll_proxy": nll(&partial, &cold),
                        "cold_nll_proxy": nll(&cold, &cold)}));
                    println!("{label}: partial branch cached {} | approximate replay mean KL {kl:.4e} | top-1 {:.1}% over {n} rows | NLL-proxy {:.4}",
                        partial.cached, 100.0 * top1, nll(&partial, &cold));
                }
            }
        }
        Ok(())
    })();
    drop(queue);
    let joined = worker.map(|worker| worker.join()).transpose();
    if let Some(output) = output {
        std::fs::create_dir_all(output)?;
        std::fs::write(output.join("result.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "pass": result.is_ok() && joined.is_ok(), "host_required": require_host, "cases": cases,
            "error": result.as_ref().err().map(|error| format!("{error:#}")),
        }))?)?;
    }
    result?;
    joined?;
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

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(hash: &str, end: usize, bytes: usize) -> Run {
        let mut record = cuteafd_api::openai::probe::ProbeRecord::default();
        record.states.push(cuteafd_api::openai::probe::ProbeState {
            kind: "prompt".into(), end, bytes, sha256: hash.into(),
        });
        Run { cached: 0, generated: Vec::new(), hashes: Vec::new(), rows: Vec::new(), record }
    }
    #[test]
    fn exact_state_requires_hash_frontier_and_initialized_extent() {
        let a = fixture("a", 129, 1024);
        let mut cases = Vec::new();
        assert!(exact(&mut cases, &a, "prompt", &a, "prompt", "same").is_ok());
        for b in [fixture("b", 129, 1024), fixture("a", 128, 1024), fixture("a", 129, 1023)] {
            assert!(exact(&mut cases, &a, "prompt", &b, "prompt", "different").is_err());
            assert_eq!(cases.last().unwrap()["pass"], false);
        }
        assert!(state(&a, "turn").is_err());
    }
    #[test]
    fn native_image_fixture_expands_through_bundled_geometry() {
        let image = cuteafd_loader::V41Image::decode(IMAGE_BYTES).unwrap();
        let prompt = cuteafd_loader::V41VisionPrompt::expand(&[1, cuteafd_loader::V41_IMAGE_TOKEN_ID, 2],
            vec![image], 4096).unwrap();
        assert!(prompt.tokens.len() > 3);
        assert_eq!(prompt.images.len(), 1);
    }
}
