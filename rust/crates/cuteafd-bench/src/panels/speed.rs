//! Speed panels: decode by content type, the concurrency sweep, the prefill
//! matrix, decode against retained context, prefix-cache behaviour.
use super::common::{self, aggregate, plain, sized_prompt, table, wave};
use super::{Ctx, Panel, Rates};
use crate::report::ServerInfo;
use crate::text::{filler, nonce};
use anyhow::{Context, Result};
use cuteafd_api::openai::probe::ProbeSpec;
use serde_json::{json, Value};

/// Eight kinds of output, thinking off.
pub const CONTENT: [(&str, &str); 8] = [
    ("code", "Write a complete Python module that implements a thread-safe LRU cache with per-entry TTL expiry, \
        a background janitor thread, hit/miss statistics and a full pytest test suite. Output only the code."),
    ("prose", "Write a long, richly detailed short story about a lighthouse keeper on a remote northern island who \
        finds a message in a bottle that seems to predict the weather. Use vivid description and dialogue."),
    ("json", "Output a JSON array of 40 fictional customer records with id, name, email, city, country, \
        signup_date, plan, monthly_spend and tags. Output only the JSON array."),
    ("math", "Solve step by step, showing every intermediate calculation: a train leaves at 9:14 travelling 87 km/h, \
        a second leaves the same station at 10:02 at 113 km/h on the same track. When and where does the second catch \
        the first? Then repeat the analysis for speeds 95 and 121 km/h."),
    ("chat", "hey! i'm planning a weekend trip to Lisbon with two friends, we like food, music and walking. any tips \
        on neighbourhoods, what to eat, and how to get around? keep it friendly and detailed."),
    ("translation", "Translate into French, keeping the tone: 'The committee met on Tuesday to review the budget. \
        After a long discussion, members agreed to postpone the purchase of new equipment until spring, citing rising \
        costs and uncertain demand. The chair thanked everyone and asked for written proposals by Friday.' Then \
        translate the same text into German and Spanish."),
    ("summary", "Summarize, in detail and with headings, the main causes, events and consequences of the industrial \
        revolution in Britain, covering technology, labour, cities, trade and politics."),
    ("table", "Produce a markdown table of 30 programming languages with columns: name, year, paradigm, typing, \
        main use, notable feature. Then add a short note under the table."),
];

const DECODE_TOKENS: u64 = 256;

/// Distinct code prompts per stream, shared by the sweep and basic card.
pub(crate) fn code_batch(client: &crate::client::Client, width: usize, tokens: u64)
    -> Vec<Result<common::Timed>> {
    let bodies = (0..width).map(|_| plain(&format!("[{}] {}", nonce(), CONTENT[0].1), tokens)).collect();
    wave(client, bodies)
}

pub struct DecodeContent;
pub struct Concurrency;
pub struct Prefill;
pub struct Retained;
pub struct PrefixCache;
pub struct PrefillShare;
pub static PREFILL_SHARE: PrefillShare = PrefillShare;
pub static DECODE_CONTENT: DecodeContent = DecodeContent;
pub static CONCURRENCY: Concurrency = Concurrency;
pub static PREFILL: Prefill = Prefill;
pub static RETAINED: Retained = Retained;
pub static PREFIX_CACHE: PrefixCache = PrefixCache;

impl Panel for DecodeContent {
    fn id(&self) -> &'static str { "decode_content" }
    fn title(&self) -> &'static str { "Decode by content" }
    fn description(&self) -> &'static str {
        "C1 decode tok/s on eight kinds of output (thinking off), with draft acceptance where the server reports it."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        8.0 * rates.seconds(120.0, DECODE_TOKENS as f64)
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let mut rows = Vec::new();
        for (i, (content, prompt)) in CONTENT.iter().enumerate() {
            ctx.progress.step(i as f64 / CONTENT.len() as f64, *content);
            let table_before = common::mapped_counters(ctx.client);
            let before = common::draft_counters(ctx.client);
            let chat = ctx.client.chat(plain(&format!("[{}] {prompt}", nonce()), DECODE_TOKENS), None)
                .with_context(|| format!("{content} decode"))?;
            let acceptance = common::acceptance(before, common::draft_counters(ctx.client));
            rows.push(json!({"content": content, "tok_s": chat.timing.decode_tok_s(),
                "tokens": chat.timing.completion_tokens, "ttft_s": chat.timing.ttft_s, "acceptance": acceptance,
                "mapped_tables": common::mapped_interval(&table_before, &common::mapped_counters(ctx.client))}));
            ctx.progress.partial(json!({"rows": rows}));
        }
        let table_rows = rows.iter().map(|r| vec![r["content"].clone(), r["tok_s"].clone(), r["tokens"].clone(),
            json!(r["ttft_s"].as_f64().unwrap_or(0.0) * 1e3), r["acceptance"].clone()]).collect();
        Ok(json!({"rows": rows, "table": table(&["content", "tok/s", "tokens", "TTFT ms", "acceptance"], table_rows)}))
    }
}

impl Concurrency {
    fn levels(info: &ServerInfo) -> Vec<usize> {
        let limit = common::concurrency(info).max(1);
        [1usize, 2, 4, 8, 16, 32].into_iter().filter(|&c| c <= limit.max(1)).collect()
    }
}

impl Panel for Concurrency {
    fn id(&self) -> &'static str { "concurrency" }
    fn title(&self) -> &'static str { "Concurrency sweep" }
    fn description(&self) -> &'static str {
        "Aggregate and per-request decode tok/s and TTFT from C1 to C32 (up to the server's concurrency), code \
         prompts, thinking off."
    }
    fn estimate_s(&self, rates: &Rates, info: &ServerInfo) -> f64 {
        Self::levels(info).iter().map(|&c| 2.0 * rates.seconds(120.0 * c as f64, 192.0) * (1.0 + 0.08 * c as f64)).sum()
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let levels = Self::levels(ctx.info);
        let mut points = Vec::new();
        for (i, &c) in levels.iter().enumerate() {
            ctx.progress.step(i as f64 / levels.len() as f64, format!("C{c}"));
            ctx.client.check()?;
            // Prime each width before measuring: graphs, workspaces and mapped tables.
            code_batch(ctx.client, c, 192).into_iter().collect::<Result<Vec<_>>>()
                .with_context(|| format!("C{c} warm-up"))?;
            ctx.client.check()?;
            let table_before = common::mapped_counters(ctx.client);
            let results = code_batch(ctx.client, c, 192);
            let errors = results.iter().filter(|r| r.is_err()).count();
            let ok: Vec<_> = results.into_iter().filter_map(Result::ok).collect();
            let mut per: Vec<f64> = ok.iter().map(|r| r.chat.timing.decode_tok_s()).collect();
            let mut ttft: Vec<f64> = ok.iter().map(|r| r.chat.timing.ttft_s).collect();
            points.push(json!({"c": c, "aggregate_tok_s": aggregate(&ok), "per_request_tok_s": common::median(&mut per),
                "ttft_s": common::median(&mut ttft), "errors": errors,
                "mapped_tables": common::mapped_interval(&table_before, &common::mapped_counters(ctx.client))}));
            ctx.progress.partial(json!({"points": points}));
        }
        let rows = points.iter().map(|p| vec![json!(format!("C{}", p["c"])), p["aggregate_tok_s"].clone(),
            p["per_request_tok_s"].clone(), json!(p["ttft_s"].as_f64().unwrap_or(0.0) * 1e3), p["errors"].clone()]).collect();
        Ok(json!({"points": points, "table": table(&["concurrency", "aggregate tok/s", "per request tok/s", "TTFT ms",
            "errors"], rows)}))
    }
}

impl Prefill {
    fn lengths(max_context: u64) -> Vec<u64> {
        common::doublings(1024, max_context.saturating_sub(512).min(131_072))
    }
}

impl Panel for Prefill {
    fn id(&self) -> &'static str { "prefill" }
    fn title(&self) -> &'static str { "Prefill matrix" }
    fn description(&self) -> &'static str {
        "TTFT and prefill tok/s from 1K to 128K tokens (up to the context limit), cold and with the prompt cached \
         except its last 256 tokens."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        Self::lengths(131_072).iter().map(|&l| rates.seconds(l as f64 * 1.15, 2.0) + 1.0).sum()
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let lengths = Self::lengths(ctx.max_context);
        let mut points = Vec::new();
        for (i, &target) in lengths.iter().enumerate() {
            ctx.progress.step(i as f64 / lengths.len() as f64, format!("{}K", target / 1024));
            let text = sized_prompt(ctx, 1000 + i as u64, target, "Reply with the single word OK.");
            // Cold: a fresh prompt; its ids come back for the cached run.
            let cold = ctx.client.chat(plain(&text, 1), Some(ProbeSpec { cold: true, ..ProbeSpec::default() }))?;
            let ids = cold.probe.as_ref().filter(|r| r.engine.is_some()).map(|r| r.prompt_ids.clone());
            let mut point = json!({"target": target, "prompt_tokens": cold.timing.prompt_tokens,
                "cold_ttft_s": cold.timing.ttft_s, "cold_tok_s": cold.timing.prefill_tok_s()});
            if let Some(ids) = ids.filter(|ids| ids.len() > 512) {
                // Cached: retain all but the last 256 tokens, then send the whole prompt.
                let cut = ids.len() - 256;
                ctx.client.chat(plain("prefix", 1), Some(ProbeSpec { prompt_ids: Some(ids[..cut].to_vec()),
                    ..ProbeSpec::default() }))?;
                let cached = ctx.client.chat(plain("prefix", 1), Some(ProbeSpec { prompt_ids: Some(ids.clone()),
                    ..ProbeSpec::default() }))?;
                point["cached_tokens"] = json!(cached.timing.cached_tokens);
                point["cached_ttft_s"] = json!(cached.timing.ttft_s);
            }
            points.push(point);
            ctx.progress.partial(json!({"points": points}));
        }
        let rows = points.iter().map(|p| vec![p["prompt_tokens"].clone(),
            json!(p["cold_ttft_s"].as_f64().unwrap_or(0.0) * 1e3), p["cold_tok_s"].clone(),
            p.get("cached_tokens").cloned().unwrap_or(Value::Null),
            json!(p["cached_ttft_s"].as_f64().map(|v| v * 1e3))]).collect();
        Ok(json!({"points": points, "table": table(&["prompt tokens", "cold TTFT ms", "cold tok/s", "cached tokens",
            "cached TTFT ms"], rows)}))
    }
}

impl Retained {
    fn contexts(max_context: u64) -> Vec<u64> {
        let mut out = vec![0];
        out.extend([4096u64, 16_384, 65_536, 131_072].into_iter().filter(|&c| c + 512 <= max_context));
        out
    }
}

impl Panel for Retained {
    fn id(&self) -> &'static str { "retained" }
    fn title(&self) -> &'static str { "Decode vs retained context" }
    fn description(&self) -> &'static str {
        "C1 decode tok/s after 0, 4K, 16K, 64K and 128K tokens of context (up to the limit), thinking off."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        Self::contexts(131_072 + 512).iter().map(|&c| rates.seconds(c as f64, 160.0) * 1.2).sum()
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let contexts = Self::contexts(ctx.max_context);
        let mut points = Vec::new();
        for (i, &context) in contexts.iter().enumerate() {
            ctx.progress.step(i as f64 / contexts.len() as f64, format!("{}K context", context / 1024));
            let question = format!("Ignore the notes. {}", CONTENT[1].1);
            let text = if context == 0 { format!("[{}] {question}", nonce()) }
                else { sized_prompt(ctx, 2000 + i as u64, context, &question) };
            let chat = ctx.client.chat(plain(&text, 160), None)?;
            points.push(json!({"context": context, "prompt_tokens": chat.timing.prompt_tokens,
                "decode_tok_s": chat.timing.decode_tok_s(), "ttft_s": chat.timing.ttft_s,
                "tokens": chat.timing.completion_tokens}));
            ctx.progress.partial(json!({"points": points}));
        }
        let rows = points.iter().map(|p| vec![p["prompt_tokens"].clone(), p["decode_tok_s"].clone(),
            json!(p["ttft_s"].as_f64().unwrap_or(0.0) * 1e3), p["tokens"].clone()]).collect();
        Ok(json!({"points": points, "table": table(&["context tokens", "decode tok/s", "TTFT ms", "tokens"], rows)}))
    }
}

impl Panel for PrefillShare {
    fn id(&self) -> &'static str { "prefill_share" }
    fn title(&self) -> &'static str { "Prefill-share exactness" }
    fn description(&self) -> &'static str {
        "Opt-in qualification: fixed C1/C4 streams with and without an 8K injection; first-row hashes and batching noise. Instrumented, not a speed measurement."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        12.0 * rates.seconds(120.0, 512.0) + 3.0 * rates.seconds(8192.0, 16.0)
    }
    fn unavailable(&self, info: &ServerInfo) -> Option<String> {
        (common::concurrency(info) < 5).then(|| "requires five admitted requests".into())
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        use crate::client::Chat;
        use cuteafd_api::openai::probe::ProbeRecord;
        use std::time::Duration;
        let deadline = std::time::Instant::now() + Duration::from_secs(1200);
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        runtime.block_on(async {
            let (_abort, abort) = tokio::sync::watch::channel(false);
            let spec = prefill_share_stream_spec;
            let bodies: Vec<_> = (0..4).map(|i| plain(&format!("prefill-share-fixed stream {i}. Write a complete Python AVL tree with insertion, deletion, traversal and 100 fully implemented unittest methods. Include all code; do not abbreviate.") , 512)).collect();
            let long = plain(&format!("prefill-share-fixed injection. {}\nIgnore the notes. Count from 1 to 20, separated by commas.", filler(0xdec0de, 6100)), 16);
            ctx.progress.step(0.02, "isolated prefill exactness control");
            let long_alone = ctx.client.chat_bounded(long.clone(), Some(prefill_share_long_spec()), deadline, abort.clone(), || {}).await?;
            ctx.progress.step(0.05, "fixed streams alone");
            let mut alone = Vec::new();
            for body in &bodies { alone.push(ctx.client.chat_bounded(body.clone(), Some(spec()), deadline, abort.clone(), || {}).await?); }
            ctx.progress.step(0.35, "C4 without injection");
            let (cohort, _) = prefill_share_wave(ctx.client, &bodies, None, deadline, Duration::from_secs(120)).await?;
            ctx.progress.step(0.55, "C1 with injection");
            let (c1, long_c1) = prefill_share_wave(ctx.client, &bodies[..1], Some(&long), deadline, Duration::from_secs(120)).await?;
            ctx.progress.step(0.75, "C4 with injection");
            let (mixed, long_c4) = prefill_share_wave(ctx.client, &bodies, Some(&long), deadline, Duration::from_secs(120)).await?;
            let record = |chat: &Chat| -> Result<ProbeRecord> {
                let record = chat.probe.as_ref().context("no probe record")?;
                anyhow::ensure!(record.engine.is_some() && record.error.is_none() && record.cold && record.cached_tokens == 0,
                    "cold probe not honoured");
                Ok(record.clone())
            };
            let comparisons = |a: &[Chat], b: &[Chat]| -> Result<Vec<Value>> {
                a.iter().zip(b).map(|(a, b)| crate::baseline::compare_probe_outputs(&record(a)?, &record(b)?)).collect()
            };
            let first_hash = |chat: &Chat| -> Result<String> {
                let record = record(chat)?;
                let row = record.rows.iter().find(|r| r.position == record.prompt_ids.len())
                    .context("injected prompt's first row missing")?;
                anyhow::ensure!(row.finite, "injected prompt's first row is nonfinite");
                Ok(row.hash.clone())
            };
            let long_c1 = long_c1.context("C1 injection missing")?;
            let long_c4 = long_c4.context("C4 injection missing")?;
            let control = first_hash(&long_alone)?;
            let c1_hash = first_hash(&long_c1)?;
            let c4_hash = first_hash(&long_c4)?;
            let qualification = prefill_share_qualification(control == c1_hash && control == c4_hash,
                &record(&long_c1)?, &record(&long_c4)?);
            Ok(json!({"scope": "fixed cold prompts; full-row hashes; common-prefix batching noise; instrumented",
                "prefill_exact": qualification["prefill_exact"],
                "qualification": qualification,
                "first_row_hashes": {"alone": control, "c1_injected": c1_hash, "c4_injected": c4_hash},
                "alone_vs_cohort": comparisons(&alone, &cohort)?,
                "cohort_vs_injected": comparisons(&cohort, &mixed)?,
                "alone_vs_c1_injected": comparisons(&alone[..1], &c1)?,
                "alone": alone, "cohort": cohort, "c1_injected": c1, "c4_injected": mixed,
                "long_alone": long_alone, "long_c1": long_c1, "long_c4": long_c4}))
        })
    }
}

fn prefill_share_stream_spec() -> ProbeSpec {
    ProbeSpec { cold: true, record_rows: 512, top_k: 32, ..ProbeSpec::default() }
}

fn prefill_share_long_spec() -> ProbeSpec {
    ProbeSpec { cold: true, record_first: true, top_k: 32, ..ProbeSpec::default() }
}

fn prefill_share_qualification(hashes_equal: bool,
    c1: &cuteafd_api::openai::probe::ProbeRecord, c4: &cuteafd_api::openai::probe::ProbeRecord) -> Value {
    let exercised = |r: &cuteafd_api::openai::probe::ProbeRecord|
        r.prefill_share.as_ref().is_some_and(|p| p.exercised());
    let exercised = exercised(c1) && exercised(c4);
    json!({"hashes_equal": hashes_equal, "exercised": exercised,
        "status": if !exercised { "not exercised" } else if hashes_equal { "exact" } else { "hash mismatch" },
        "prefill_exact": hashes_equal && exercised,
        "c1": c1.prefill_share, "c4": c4.prefill_share})
}

async fn prefill_share_wave(client: &crate::client::Client, bodies: &[Value], long: Option<&Value>,
    deadline: std::time::Instant, injection_wait: std::time::Duration,
) -> Result<(Vec<crate::client::Chat>, Option<crate::client::Chat>)> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let (abort, abort_rx) = tokio::sync::watch::channel(false);
    let (send, mut receive) = tokio::sync::mpsc::unbounded_channel();
    let finished: Vec<_> = bodies.iter().map(|_| AtomicBool::new(false)).collect();
    let workers = futures::future::join_all(bodies.iter().cloned().enumerate().map(|(index, body)| {
        let send = send.clone();
        let abort_rx = abort_rx.clone();
        let abort = &abort;
        let finished = &finished[index];
        async move {
            let mut count = 0;
            let result = client.chat_bounded(body, Some(prefill_share_stream_spec()), deadline, abort_rx, || {
                count += 1;
                if count == 12 { let _ = send.send((index, true)); }
            }).await;
            finished.store(true, Ordering::Release);
            let _ = send.send((index, false));
            if result.is_err() { let _ = abort.send(true); }
            result
        }
    }));
    drop(send);
    let injection = async {
        let result: Result<Option<crate::client::Chat>> = async {
            let Some(long) = long else { return Ok(None); };
            let ready = async {
                let mut ready = vec![false; bodies.len()];
                while !ready.iter().all(|r| *r) {
                    let (index, live) = receive.recv().await.context("streams ended before injection point")?;
                    anyhow::ensure!(live, "stream ended before every stream reached injection point");
                    ready[index] = true;
                }
                anyhow::ensure!(finished.iter().all(|done| !done.load(Ordering::Acquire)), "stream ended before injection");
                Ok::<_, anyhow::Error>(())
            };
            tokio::time::timeout_at((std::time::Instant::now() + injection_wait).min(deadline).into(), ready)
                .await.context("streams did not reach injection point before deadline")??;
            Ok(Some(client.chat_bounded(long.clone(), Some(prefill_share_long_spec()), deadline, abort_rx, || {}).await?))
        }.await;
        // Wake every body reader before waiting for worker results. No blocking
        // scoped threads remain to hold the benchmark lockout after failure.
        if result.is_err() { let _ = abort.send(true); }
        result
    };
    let (chats, injected) = tokio::join!(workers, injection);
    let injected = injected?;
    Ok((chats.into_iter().collect::<Result<Vec<_>>>()?, injected))
}

#[cfg(test)]
mod prefill_share_tests {
    use super::*;
    use cuteafd_api::openai::probe::{ProbePrefillShare, ProbeRecord};
    use std::sync::{Arc, atomic::AtomicBool};
    use std::time::{Duration, Instant};

    #[test]
    fn equal_hashes_cannot_qualify_exclusive_or_idle_prefill() {
        let mut record = ProbeRecord::default();
        for evidence in [None, Some(ProbePrefillShare { decode_share: 0.0, resumed_waves: 3,
            interleaved_decode_steps: 2, ..Default::default() }),
            Some(ProbePrefillShare { decode_share: 0.2, ..Default::default() }),
            Some(ProbePrefillShare { decode_share: 0.2, resumed_waves: 3, ..Default::default() })] {
            record.prefill_share = evidence;
            let result = prefill_share_qualification(true, &record, &record);
            assert_eq!(result["prefill_exact"], false);
            assert_eq!(result["status"], "not exercised");
        }
        record.prefill_share = Some(ProbePrefillShare { decode_share: 0.2,
            resumed_waves: 3, interleaved_decode_steps: 2, ..Default::default() });
        assert_eq!(prefill_share_qualification(true, &record, &record)["prefill_exact"], true);
        assert_eq!(prefill_share_qualification(false, &record, &record)["prefill_exact"], false);
        assert_eq!(prefill_share_qualification(true, &record, &ProbeRecord::default())["prefill_exact"], false);
    }

    async fn stalled_server(keepalives: bool) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(socket.read_u8().await.unwrap());
            }
            let headers = String::from_utf8(headers).unwrap();
            let length: usize = headers.lines().find_map(|line| line.split_once(':')
                .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| value.trim().parse().unwrap())).unwrap();
            let mut body = vec![0; length];
            socket.read_exact(&mut body).await.unwrap();
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").await.unwrap();
            let mut byte = [0];
            loop {
                tokio::select! {
                    read = socket.read(&mut byte) => {
                        match read {
                            Ok(0) => (),
                            Err(error) if matches!(error.kind(), std::io::ErrorKind::ConnectionReset
                                | std::io::ErrorKind::ConnectionAborted) => (),
                            other => panic!("reader not closed: {other:?}"),
                        }
                        break;
                    }
                    _ = tokio::time::sleep(Duration::from_millis(10)), if keepalives => {
                        if socket.write_all(b": keepalive\n\n").await.is_err() { break; }
                    }
                }
            }
        });
        (base, server)
    }

    #[tokio::test]
    async fn injection_timeout_closes_silent_and_keepalive_readers() {
        for keepalives in [false, true] {
            let (base, server) = stalled_server(keepalives).await;
            let client = crate::client::Client::new(&base, None, Arc::new(AtomicBool::new(false)));
            let started = Instant::now();
            let error = tokio::time::timeout(Duration::from_secs(2), prefill_share_wave(&client,
                &[plain("stream", 512)], Some(&plain("injection", 16)),
                started + Duration::from_secs(30), Duration::from_millis(100))).await.unwrap().unwrap_err();
            assert!(error.to_string().contains("injection point"), "{error:#}");
            assert!(started.elapsed() < Duration::from_secs(1));
            tokio::time::timeout(Duration::from_secs(1), server).await.unwrap().unwrap();
        }
    }

    #[tokio::test]
    async fn overall_deadline_closes_keepalive_reader_without_injection() {
        let (base, server) = stalled_server(true).await;
        let client = crate::client::Client::new(&base, None, Arc::new(AtomicBool::new(false)));
        let started = Instant::now();
        let error = tokio::time::timeout(Duration::from_secs(2), prefill_share_wave(&client,
            &[plain("stream", 512)], None, started + Duration::from_millis(100),
            Duration::from_secs(120))).await.unwrap().unwrap_err();
        assert!(error.to_string().contains("deadline"), "{error:#}");
        tokio::time::timeout(Duration::from_secs(1), server).await.unwrap().unwrap();
    }
}

impl Panel for PrefixCache {
    fn id(&self) -> &'static str { "prefix_cache" }
    fn title(&self) -> &'static str { "Prefix cache" }
    fn description(&self) -> &'static str {
        "A six-turn conversation: per turn the prompt, the tokens restored from the cache, TTFT against a cold \
         prefill of the same prompt."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        6.0 * (rates.seconds(1500.0, 64.0) + rates.seconds(3000.0, 1.0))
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let mut history = vec![json!({"role": "system", "content": format!("[{}] You are a concise assistant. \
            Reference notes follow.\n\n{}", nonce(), filler(4242, 700))})];
        let mut turns = Vec::new();
        for turn in 1..=6 {
            ctx.progress.step((turn - 1) as f64 / 6.0, format!("turn {turn}"));
            history.push(json!({"role": "user", "content": format!("Question {turn}: {} Answer in three sentences.",
                filler(turn as u64 * 77, 120))}));
            let body = json!({"messages": history, "max_tokens": 64, "temperature": 0, "thinking": {"type": "disabled"}});
            let chat = ctx.client.chat(body.clone(), None)?;
            let mut cold_body = body.clone();
            cold_body["max_tokens"] = json!(1);
            let cold = ctx.client.chat(cold_body, Some(ProbeSpec { cold: true, ..ProbeSpec::default() }))?;
            let supported = cold.probe.as_ref().is_some_and(|r| r.engine.is_some());
            turns.push(json!({"turn": turn, "prompt_tokens": chat.timing.prompt_tokens,
                "cached_tokens": chat.timing.cached_tokens, "ttft_s": chat.timing.ttft_s,
                "cold_ttft_s": supported.then_some(cold.timing.ttft_s)}));
            history.push(json!({"role": "assistant", "content": chat.content}));
            ctx.progress.partial(json!({"turns": turns}));
        }
        let rows = turns.iter().map(|t| vec![t["turn"].clone(), t["prompt_tokens"].clone(), t["cached_tokens"].clone(),
            json!(t["ttft_s"].as_f64().unwrap_or(0.0) * 1e3), json!(t["cold_ttft_s"].as_f64().map(|v| v * 1e3))]).collect();
        Ok(json!({"turns": turns, "table": table(&["turn", "prompt", "cached", "TTFT ms", "cold TTFT ms"], rows)}))
    }
}
