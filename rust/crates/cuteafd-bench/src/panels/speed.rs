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
pub struct DraftMix;
pub struct Prefill;
pub struct Retained;
pub struct PrefixCache;
pub static DECODE_CONTENT: DecodeContent = DecodeContent;
pub static CONCURRENCY: Concurrency = Concurrency;
pub static DRAFT_MIX: DraftMix = DraftMix;
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

/// The fixture module a copy-heavy request rewrites.
const REWRITE_SOURCE: &str = "ledger/money.py";
const MIX_TOKENS: u64 = 256;

impl Panel for DraftMix {
    fn id(&self) -> &'static str { "draft_mix" }
    fn title(&self) -> &'static str { "Draft mix" }
    fn description(&self) -> &'static str {
        "C16 aggregate decode tok/s on heterogeneous prompts (the eight content kinds, two waves), and C1 decode \
         tok/s and tokens per verification round rewriting a fixture module (copy-heavy), thinking off."
    }
    fn estimate_s(&self, rates: &Rates, _info: &ServerInfo) -> f64 {
        2.0 * rates.seconds(120.0 * 16.0, MIX_TOKENS as f64) * 2.3 + 2.0 * rates.seconds(1500.0, 1200.0)
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let width = common::concurrency(ctx.info).clamp(1, 16);
        let mut waves = Vec::new();
        for wave_index in 0..2 {
            ctx.progress.step(0.4 * wave_index as f64, format!("C{width} mixed, wave {}", wave_index + 1));
            let bodies = (0..width).map(|i| plain(&format!("[{}] {}", nonce(), CONTENT[(i + wave_index) % CONTENT.len()].1),
                MIX_TOKENS)).collect();
            let results = wave(ctx.client, bodies);
            let errors = results.iter().filter(|r| r.is_err()).count();
            let ok: Vec<_> = results.into_iter().filter_map(Result::ok).collect();
            let mut per: Vec<f64> = ok.iter().map(|r| r.chat.timing.decode_tok_s()).collect();
            waves.push(json!({"c": width, "aggregate_tok_s": aggregate(&ok), "per_request_tok_s": common::median(&mut per),
                "errors": errors}));
            ctx.progress.partial(json!({"waves": waves}));
        }
        let source = crate::panels::agentic::fixture(REWRITE_SOURCE).context("rewrite fixture missing")?;
        let mut rewrites = Vec::new();
        for pass in 0..2 {
            ctx.progress.step(0.8 + 0.1 * pass as f64, format!("rewrite {}", pass + 1));
            let prompt = format!("[{}] Rewrite this Python module exactly as it is, adding a one-line comment \
                `# reviewed` at the very top and changing nothing else. Output only the code.\n\n```python\n{source}```",
                nonce());
            let before = common::round_counters(ctx.client);
            let drafts = common::draft_counters(ctx.client);
            let chat = ctx.client.chat(plain(&prompt, 1600u64.min(ctx.max_output)), None).context("rewrite decode")?;
            let after = common::round_counters(ctx.client);
            let per_round = before.zip(after).and_then(|((r0, t0), (r1, t1))| (r1 > r0).then(|| (t1 - t0) / (r1 - r0)));
            rewrites.push(json!({"tok_s": chat.timing.decode_tok_s(), "tokens": chat.timing.completion_tokens,
                "tokens_per_round": per_round,
                "acceptance": common::acceptance(drafts, common::draft_counters(ctx.client))}));
            ctx.progress.partial(json!({"waves": waves, "rewrite": rewrites}));
        }
        let mut rows: Vec<Vec<Value>> = waves.iter().enumerate().map(|(i, w)| vec![json!(format!("C{} mixed {}", w["c"], i + 1)),
            w["aggregate_tok_s"].clone(), w["per_request_tok_s"].clone(), Value::Null]).collect();
        rows.extend(rewrites.iter().enumerate().map(|(i, r)| vec![json!(format!("C1 rewrite {}", i + 1)), r["tok_s"].clone(),
            r["tok_s"].clone(), r["tokens_per_round"].clone()]));
        Ok(json!({"waves": waves, "rewrite": rewrites,
            "table": table(&["workload", "aggregate tok/s", "per request tok/s", "tokens/round"], rows)}))
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
