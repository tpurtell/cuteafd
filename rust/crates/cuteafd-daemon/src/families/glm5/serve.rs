//! OpenAI-compatible API over the GLM engine: continuous batching with one
//! prefill per admitted request and one decode-shaped step for every active
//! sequence. With a DFlash2 drafter (--draft), every step first drafts after
//! each sequence's next token on the GPU and verifies as many drafts as the
//! adaptive policy (glm/dflash_policy.rs) prices worthwhile; copy-window
//! drafts extend a DFlash2 draft they agree with.
//!
//! Prefix cache (`cuteafd_engine::prefix` over `super::prefix::GlmPrefix`):
//! the pages are the whole state, so admission restores the deepest retained
//! snapshot whose tokens prefix the prompt by sharing its full pages and
//! copying its tail page, and prefills only the rest; a whole-prompt hit
//! takes its first token from the retained logits. The prompt is retained at
//! prompt end (unless it was a whole hit), the conversation at a normal
//! finish (`Turn`), a prefill whose client left is parked at its last chunk;
//! a decode whose client left is not retained. `prompt_cache_hit_tokens`
//! reports the restored rows; `/v1/stats` carries the cache's counters.
use super::dflash::{ContextRow, DraftSeq, TAP_ROWS};
use super::dflash_policy::{self, DraftHistory, Shape};
use super::engine::{GlmEngine, GlmPlacement, DECODE_ROWS};
use super::prefix::GlmPrefix;
use crate::families::deepseek_v41::v41_native_serve::prefix::CudaCopyEngine;
use crate::shared::prefix::{PrefixArgs, Toggle};
use crate::shared::probe;
use cuteafd_engine::prefix::{After, PointPlan, PointPolicy, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};

/// Most copy-window draft tokens verified per sequence and step.
const COPY_DRAFT: usize = 7;
use super::{open, Opened};
use crate::shared::token_io::{SelectBatch, SelectPlacement, TokenSelector};
use crate::shared::prefill_share::{Chunk, DecodeShareArgs};
use anyhow::{Context, Result};
use cuteafd_api::openai::chat::glm5::{GlmEncoding, GlmThinkingOff};
use cuteafd_api::openai::{
    InferenceChunk, InferenceFinishReason, ModelEncoding, ModelProfile, NativeFailure, NativeLimits, NativeRequest,
    PromptUsage,
};
use crate::shared::spark_intake::SparkLink;
use crate::shared::console;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: super::EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Sequences decoding at once (at most the decode programs' 64 rows).
    #[arg(long, default_value_t = 16)]
    pub max_sequences: usize,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    /// How a request that turns thinking off renders: empty (strictly off,
    /// an empty think block after the default Max effort) or low (the
    /// template's Low effort, think block open, a short plan as reasoning).
    #[arg(long, env = "CUTEAFD_GLM_THINKING_OFF", default_value = "empty")]
    pub thinking_off: GlmThinkingOff,
    /// Decode one token per step (no copy-window drafts).
    #[arg(long)]
    pub no_copy_drafts: bool,
    /// With --draft: verify exactly this many DFlash2 drafts per step
    /// (within the rows) instead of the adaptive policy.
    #[arg(long)]
    pub draft_fixed: Option<usize>,
    #[command(flatten)]
    pub decode_share: DecodeShareArgs,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub console: console::ConsoleArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
}

fn model_id(snapshot: &std::path::Path) -> Option<String> {
    snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    })
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    let api = args.api.load()?;
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let encoding = GlmEncoding::from_snapshot(&snapshot)?.with_thinking_off(args.thinking_off);
    let profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&snapshot)).context("model id")?,
        ModelEncoding::Glm(Arc::new(encoding)),
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    if engine_args.draft_context_slots.is_none() {
        engine_args.draft_context_slots = Some(20.max(engine_args.draft_sequences)
            .max(args.max_sequences.saturating_mul(5).div_ceil(4)));
    }
    let (worker_stats, max_sequences) = (stats.clone(), args.max_sequences);
    let policy = Policy { copy: if args.no_copy_drafts { 0 } else { COPY_DRAFT }, fixed: args.draft_fixed,
        decode_share: args.decode_share };
    let prefix = args.prefix.clone();
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id)));
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, policy, prefix));
    let max_context = ready_rx.await.context("engine failed before it was ready")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    cuteafd_bench::context::phase("engine loaded");
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, Duration::from_secs(25),
        hub.clone(), crate::shared::api::profile(profile.clone()));
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    tracing::info!(listen = %args.listen, model = %profile.id, "GLM API is ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<std::net::SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
    }
    Ok(())
}

/// What the live console shows for GLM 5.3.
fn console_layout(args: &ServeArgs, model: &str) -> console::Layout {
    use console::{Color::*, StepGroup};
    let mut layout = console::Layout::new("glm5", model.into(), args.engine.snapshot.clone());
    let sparks = args.engine.peers.as_deref().map_or(0, |peers| peers.split(',').count());
    layout.hardware = console::hardware(1 + usize::from(args.engine.split_device.is_some()), sparks, false);
    layout.split = args.engine.split_device.map(|_| "head split".into());
    layout.concurrency = args.max_sequences.min(DECODE_ROWS);
    let copy = if args.no_copy_drafts { "" } else { " · copy windows" };
    let policy = args.draft_fixed.map_or_else(|| "adaptive".to_string(), |n| format!("fixed {n}"));
    layout.speculator = match (&args.engine.draft, args.no_copy_drafts) {
        (Some(_), _) => Some(console::Speculator { name: "DFlash2".into(), positions: 8, policy: policy + copy }),
        (None, false) => Some(console::Speculator { name: "Copy window".into(), positions: COPY_DRAFT,
            policy: "adaptive length".into() }),
        (None, true) => None,
    };
    layout.steps = vec![
        StepGroup::new("Decode step", "host clock", &[("round.cycle", "step", Target),
            ("draft", "DFlash2 draft", Accepted), ("plan", "draft plan + copy windows", Ink),
            ("rows", "selection rows", Ink), ("verify", "verify pass + token selection", Target),
            ("emit", "accept + stream", Ink), ("update", "drafter context", Accepted)]),
        StepGroup::new("Verify pass", "host clock, engine phases", &[
            ("gpu", "GPU until expert exchanges", Rtx), ("experts", "Spark expert exchanges", Spark),
            ("head", "head + logits", Target)]),
        StepGroup::layers(),
        StepGroup::admission(false),
    ];
    layout.layers = Some(console::Layers::host_clock());
    layout
}

/// Speculation settings: copy-window draft cap (0 disables) and a fixed
/// DFlash2 draft count replacing the adaptive policy.
#[derive(Debug, Clone, Copy)]
struct Policy {
    copy: usize,
    fixed: Option<usize>,
    decode_share: DecodeShareArgs,
}

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<usize>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    policy: Policy, prefix: PrefixArgs) -> Result<()> {
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let mut ready = Some(ready);
    let result = opened.with_engine(&args, |engine, transport, runtime| {
        anyhow::ensure!(transport.is_some() || engine.skip_routed(), "serve-glm needs --peers for the routed experts");
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok(engine.max_context));
        }
        let ranks = args.peers.as_deref().map_or(4, |peers| peers.split(',').count());
        let spark = transport.is_some();
        console::layer_classes(engine.weights.layers.iter().map(|l| console::layer_class(l.dense, spark)).collect());
        schedule(engine, &opened, &mut receive, transport, runtime, &stats, max_sequences.min(DECODE_ROWS), policy,
            ranks, &prefix, args.token_io.token_select)
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| args.max_context).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

/// An admitted prompt waiting for its remaining prefill chunks.
struct Prefill<'a> {
    job: NativeRequest,
    constraint: Option<crate::shared::constraints::State<'a>>,
    tokens: Vec<u32>,
    /// Prompt tokens prefilled so far (from the prefix cache's restore point).
    done: usize,
    /// Rows restored from the prefix cache.
    resume: usize,
    /// Chunk ends and intermediate snapshot points (`cuteafd_engine::prefix::plan_points`).
    plan: PointPlan,
    /// Chunks prefilled so far.
    chunks: usize,
    /// The client left mid-prefill: its prefilled rows are parked as a prompt snapshot.
    cancelled: bool,
    placement: GlmPlacement,
    capacity: usize,
    slot: Option<usize>,
    /// The last row's logits on the host: a whole-prompt cache hit's retained
    /// row, or the prefill's last row downloaded for the prompt snapshot.
    logits: Option<Vec<f32>>,
    /// The first generated token, selected on the device after the last chunk.
    first: Option<u32>,
    started: Instant,
    /// Seconds in this prompt's chunks.
    busy: f64,
    id: u64,
    ticket: console::Ticket,
}

struct Active<'a> {
    job: NativeRequest,
    /// Serial number (the trace's request id).
    id: u64,
    prompt_tokens: usize,
    /// Prompt and generated tokens, for copy-window drafts.
    history: Vec<u32>,
    /// Current copy-draft length (halved after a fully rejected draft,
    /// doubled after a fully accepted one) and steps left before drafting
    /// resumes once it reached zero.
    draft_limit: usize,
    draft_pause: usize,
    /// DFlash2 ring slot and draft outcomes (None without a drafter slot).
    slot: Option<usize>,
    drafts: DraftHistory,
    copy_drafts: DraftHistory,
    /// Hash of `history` (identical sequences share it).
    digest: u64,
    /// Steps, DFlash2 drafts verified and accepted, copy drafts verified and accepted.
    counts: [usize; 5],
    constraint: Option<crate::shared::constraints::State<'a>>,
    placement: GlmPlacement,
    /// First position the DFlash2 drafter's context holds for this sequence
    /// (the prefix-cache restore point; the drafter starts cold there).
    draft_from: usize,
    /// The logit row that produced the last token, once the request finished
    /// normally (EOS or max_tokens with the client still there): what follows
    /// its `Turn` snapshot.
    turn: Option<Vec<f32>>,
    capacity: usize,
    next: u32,
    decoder: cuteafd_loader::StreamingTokenDecoder,
    generated: usize,
    buffered: usize,
    started: Instant,
    ticket: console::Ticket,
}

/// Commits a selected token to the request's grammar.
fn take(constraint: Option<&mut crate::shared::constraints::State<'_>>, selected: &crate::shared::token_io::RowResult)
    -> Result<u32> {
    let token = selected.as_ref().map_err(|e| anyhow::anyhow!("sampling: {e:?}"))?.token;
    if let Some(state) = constraint {
        state.accept(token)?;
    }
    Ok(token)
}

impl Active<'_> {
    /// Selects from a host logits row (a whole-prompt cache hit's retained row).
    fn select(&mut self, logits: &[f32]) -> Result<u32> {
        let position = self.placement.len as u64;
        let mask = match self.constraint.as_mut() {
            Some(state) => state.mask()?,
            None => None,
        };
        let token = self.job.sampling.select_token(logits, mask, position)
            .map_err(|e| anyhow::anyhow!("sampling: {e:?}"))? as u32;
        if let Some(state) = self.constraint.as_mut() {
            state.accept(token)?;
        }
        Ok(token)
    }

    fn send(&self, chunk: InferenceChunk) -> Result<()> {
        self.job.events.send(Ok(chunk)).map_err(|_| anyhow::anyhow!("client went away"))
    }

    /// Streams `token` (special tokens stay text for the GLM parser); returns
    /// true when the request is finished.
    fn emit(&mut self, token: u32) -> Result<bool> {
        crate::shared::probe::token(&self.job.probe, token);
        self.history.push(token);
        self.digest = digest(self.digest, token);
        self.generated += 1;
        self.buffered += 1;
        // A grammar that accepted one of its stop tokens has ended the request.
        let stop = self.job.stop_token_ids.contains(&token)
            || self.constraint.as_ref().is_some_and(|state| state.terminated());
        if !stop {
            if let Some(content) = self.decoder.step(token)? {
                self.send(InferenceChunk::Text { content, content_tokens: self.buffered })?;
                self.buffered = 0;
            }
        }
        let finish = if stop {
            Some(InferenceFinishReason::Stop)
        } else if self.generated >= self.job.max_tokens || self.placement.len + 1 >= self.capacity {
            Some(InferenceFinishReason::Length)
        } else {
            None
        };
        let Some(finish) = finish else {
            self.next = token;
            return Ok(false);
        };
        let content = self.decoder.finish()?.unwrap_or_default();
        if !content.is_empty() || self.buffered > 0 {
            self.send(InferenceChunk::Text { content, content_tokens: self.buffered })?;
        }
        self.ticket.finishing();
        self.send(InferenceChunk::Finish { finish_reason: finish })?;
        Ok(true)
    }
}

/// Per-cycle speculation trace (CUTEAFD_SPECULATION_TRACE=path, JSON lines): every
/// verify of a sequence (position, rows, committed rows, planned DFlash2
/// drafts, whether a copy window replaced them, the full DFlash2 draft and
/// its selector features, step ms) and every finished request's generated
/// tokens. At temperature 0 the output is the target's greedy sequence, so
/// the trace scores any draft-count policy offline.
struct Trace(std::io::BufWriter<std::fs::File>);

impl Trace {
    fn open() -> Result<Option<Self>> {
        let Some((var, path)) = crate::shared::draft_policy::speculation_trace_path("CUTEAFD_GLM_TRACE") else { return Ok(None) };
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            .with_context(|| format!("{var} {path}"))?;
        Ok(Some(Self(std::io::BufWriter::new(file))))
    }

    #[allow(clippy::too_many_arguments)]
    fn cycle(&mut self, id: u64, position: usize, rows: usize, committed: usize, planned: usize, copy: bool,
        draft: Option<&super::dflash::Draft>, verify_ms: f64) -> Result<()> {
        use std::io::Write;
        let line = serde_json::json!({"kind": "cycle", "id": id, "position": position, "rows": rows,
            "committed": committed, "planned": planned, "copy": copy, "verify_ms": verify_ms,
            "draft": draft.map(|d| &d.tokens), "features": draft.map(|d| &d.features)});
        writeln!(self.0, "{line}")?;
        Ok(())
    }

    fn done(&mut self, id: u64, prompt_tokens: usize, generated: &[u32]) -> Result<()> {
        use std::io::Write;
        writeln!(self.0, "{}", serde_json::json!({"kind": "done", "id": id, "prompt_tokens": prompt_tokens,
            "generated": generated}))?;
        self.0.flush()?;
        Ok(())
    }
}

const DIGEST_SEED: u64 = 0xcbf2_9ce4_8422_2325;

fn digest(state: u64, token: u32) -> u64 {
    (state ^ u64::from(token)).wrapping_mul(0x0100_0000_01b3)
}

/// Longest n-gram (from 8 down to 4 tokens) that ends the history and occurred
/// earlier; proposes up to `limit` tokens that followed its latest earlier
/// occurrence (a copy window). Exact: the verify step accepts only tokens the
/// model itself produces.
fn copy_drafts(history: &[u32], limit: usize) -> Vec<u32> {
    let len = history.len();
    for n in (4..=8).rev() {
        if len <= n {
            continue;
        }
        let tail = &history[len - n..];
        if let Some(start) = (0..len - n).rev().find(|&i| &history[i..i + n] == tail) {
            let from = start + n;
            return history[from..(from + limit).min(len)].to_vec();
        }
    }
    Vec::new()
}

/// The prefix cache over `engine` (always present: with zero entries it is the page allocator).
fn prefix_cache<'e, 'a>(engine: &'e GlmEngine<'a>, args: &PrefixArgs)
    -> Result<(GlmPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let entries = args.prefix_cache_entries;
    let family = GlmPrefix::new(engine, args.prefix_partial == Toggle::On)?;
    let template = engine.paged_buffers().first().map(|b| b.0).context("GLM has no layers")?;
    // The pinned host tier copies through one GPU's copy engine; a head split keeps its
    // (replicated) pages on both GPUs, so it keeps device-resident snapshots only.
    let host = if engine.ranks() > 1 {
        if args.host_cache_bytes.enabled() && args.prefix_cache_entries > 0 {
            tracing::warn!("GLM head split: the prefix cache's host tier is off (device-resident snapshots only)");
        }
        None
    } else {
        args.host_tier(engine.library, template, family.layout(), engine.max_context)?
    };
    let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
    let layout = family.layout();
    let config = PrefixConfig { entries, mark_slots: 0, keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
    let cache = PrefixCache::new(layout, config, host)?;
    tracing::info!(entries, page_bytes = layout.page_bytes, pages = layout.pages, host_bytes,
        rule = ?layout.rule, points = ?args.points(), "GLM prefix cache");
    cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, layout.pages as u64,
        &"FP8 MLA latent + DSA index".to_string(), host_bytes);
    Ok((family, cache))
}

/// Gives a finished or failed sequence's pages and drafter slot back.
fn release(family: &GlmPrefix<'_, '_>, cache: &mut PrefixCache<CudaCopyEngine<'_>>, slots: &mut Vec<usize>,
    placement: &GlmPlacement, slot: Option<usize>) {
    if let Err(error) = cache.release(family, &placement.pages) {
        tracing::error!(%error, "releasing a GLM sequence's pages");
    }
    slots.extend(slot);
}

/// Serving statistics for `/v1/stats`.
fn publish(stats: &Mutex<serde_json::Value>, requests: u64, generated: u64, active: usize, prefilling: usize,
    cache: &PrefixCache<CudaCopyEngine<'_>>) {
    if let Ok(mut stats) = stats.lock() {
        *stats = serde_json::json!({"requests": requests, "generated_tokens": generated, "active": active,
            "prefilling": prefilling, "prefix_cache": cache.stats()});
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule(engine: &GlmEngine<'_>, opened: &Opened, receive: &mut mpsc::Receiver<NativeRequest>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime, stats: &Mutex<serde_json::Value>,
    max_sequences: usize, policy: Policy, ranks: usize, prefix: &PrefixArgs, select: SelectPlacement) -> Result<()> {
    let (family, mut cache) = prefix_cache(engine, prefix)?;
    let markers = crate::shared::prefix::marker_ids(&opened.snapshot,
        &crate::families::glm5_flash::serve::MESSAGE_STARTS)?;
    let mut grammars = crate::shared::constraints::Compiler::with_vocab(
        &opened.library, opened.snapshot.join("tokenizer.json"), engine.cfg.vocab_size, engine.cfg.eos_tokens.clone());
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(&opened.snapshot)?;
    let drafter = engine.drafter.as_ref();
    let mut free_slots: Vec<usize> = drafter.map_or(Vec::new(), |d| (0..d.slots).rev().collect());
    // Measured at TP4; other Spark layouts scale its Spark share (TP6: 384 / 512).
    let widest = dflash_policy::widest_slice(engine.cfg.moe_intermediate, ranks);
    let table = dflash_policy::rescale_spark(&dflash_policy::K4_TP4_STEP_MS, dflash_policy::K4_TP4_GPU_MS, 512, widest);
    let mut cost = dflash_policy::step_cost(&table, DECODE_ROWS);
    let mut confidence = drafter.map(|d| d.confidence_policy("glm5", false)).transpose()?;
    let copy_policy = crate::shared::draft_policy::enabled("CUTEAFD_COPY_DRAFT_POLICY");
    let refine_confidence = crate::shared::draft_policy::enabled("CUTEAFD_DRAFT_CONFIDENCE");
    tracing::info!(copy_policy, refine_confidence, "shared draft policy experiments");
    let mut skip = dflash_policy::DraftSkip::default();
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total, mut admitted_total) = (0u64, 0u64, 0u64);
    let mut selector = TokenSelector::new(&opened.library, select, engine.cfg.vocab_size, DECODE_ROWS)?;
    // Host seconds per step phase since the last completed request: draft
    // (anchor rows + drafter), plan (policy + copy windows), embed (verify
    // rows), verify, select (sampling + streaming), drafter context update;
    // then steps and verified rows.
    let mut phases = [0f64; 8];
    let mut trace = Trace::open()?;
    let mut prefills = policy.decode_share.queue::<Prefill<'_>>()?;
    let mut kv_waiter = cuteafd_engine::prefix::DeferredAdmission::<NativeRequest>::default();
    loop {
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("expert wire unavailable until restart: {reason}");
        }
        while active.len() + prefills.len() < max_sequences {
            let busy = !active.is_empty() || !prefills.is_empty();
            let job = match kv_waiter.poll(cache.pool().free(), cache.pool().release_epoch(), busy, |job| job.events.is_closed()) {
                cuteafd_engine::prefix::AdmissionPoll::Blocked => break,
                cuteafd_engine::prefix::AdmissionPoll::Ready(job) => job,
                cuteafd_engine::prefix::AdmissionPoll::Empty => {
                    if !busy {
                        // Idle: publish the state the server waits in (captures and releases done).
                        cache.tick();
                        publish(stats, requests, generated_total, 0, 0, &cache);
                        console::gauges_now(|| console::Gauges::prefix_cache(&cache, 0, 0, 0));
                        match receive.blocking_recv() {
                            Some(job) => job,
                            None => return Ok(()),
                        }
                    } else {
                        match receive.try_recv() {
                            Ok(job) => job,
                            Err(_) => break,
                        }
                    }
                },
            };
            let reject = |job: &NativeRequest, message: String| {
                let _ = job.events.send(Err(NativeFailure::BadRequest(message)));
            };
            if let Err(error) = probe::validate_scoring(&job.probe, engine.full_prefill_logits) {
                reject(&job, format!("scoring: {error:#}"));
                continue;
            }
            let constraint = match job.constraint.as_ref().map(|spec| grammars.matcher(spec)).transpose() {
                Ok(constraint) => constraint,
                Err(error) => {
                    reject(&job, format!("{error:#}"));
                    continue;
                }
            };
            let tokens = probe::prompt_ids(&job.probe, || Ok(tokenizer.encode_text(&job.prompt, false)?.token_ids))?;
            let cold = probe::cold(&job.probe);
            if tokens.is_empty() || tokens.len() >= engine.max_context {
                reject(&job, format!("prompt of {} tokens is outside 1..{}", tokens.len(), engine.max_context));
                continue;
            }
            let capacity = (tokens.len() + job.max_tokens).min(engine.max_context);
            let slot = free_slots.pop();
            cache.tick();
            let admit_started = Instant::now();
            // Lookup and fork of the retained pages (byte-exact: the pages are the whole state).
            let lookup: &[u32] = if cold { &[] } else { &tokens };
            let admitted = match cache.admit(&family, lookup, capacity, true, |pages| GlmPlacement { pages, len: 0 }) {
                Ok(admitted) => admitted,
                Err(error) => {
                    free_slots.extend(slot);
                    // Running requests keep their pages pinned. Delay a request
                    // that fits alone instead of rejecting transient KV pressure.
                    match kv_waiter.defer(job, &error, busy, cache.pool().release_epoch()) {
                        Ok(()) => break,
                        Err(job) => reject(&job, format!("{error:#}")),
                    }
                    continue;
                }
            };
            admitted_total += 1;
            let resume = admitted.resume;
            probe::admitted(&job.probe, "glm5", &tokens, resume);
            let _ = job.events.send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: resume },
            }));
            if let Some(from) = probe::scoring(&job.probe) {
                // Teacher-forced scoring: every row's logits, no generation, nothing retained.
                let mut placement = admitted.placement;
                let mut state = (&mut placement, transport.as_deref_mut());
                // Diagnostic tails can be smaller than the pipeline's 512-row minimum.
                let prefill_rows = probe::scoring_prefill_capacity(engine.full_prefill_logits,
                    engine.prefill_rows, engine.prefill_capacity());
                let scored = probe::score(&opened.library, &job.probe, &tokens, from, prefill_rows,
                    DECODE_ROWS, probe::verify_rows(&job.probe), engine.full_prefill_logits, &mut state,
                    |(placement, transport), chunk, rows| {
                        let experts = transport.as_deref_mut().map(|t| (t, runtime));
                        if rows > 1 {
                            Ok(engine.prefill_rows_logits(placement, chunk, experts, None, rows)?
                                .map(|values| probe::ScoreLogits::Host { values, vocab: engine.cfg.vocab_size }))
                        } else {
                            Ok(engine.prefill_device(placement, chunk, experts)?.map(probe::ScoreLogits::Device))
                        }
                    },
                    |(placement, transport), chunk| engine.verify_device(&mut [(&mut **placement, chunk.len())], chunk,
                        transport.as_deref_mut().map(|t| (t, runtime)))?.context("scoring needs every layer"));
                match scored {
                    Ok(_) => { let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })); }
                    Err(error) => { let _ = job.events.send(Err(NativeFailure::Worker(format!("scoring: {error:#}")))); }
                }
                release(&family, &mut cache, &mut free_slots, &placement, slot);
                continue;
            }
            let ticket = console::admit(tokens.len(), resume, job.max_tokens, constraint.is_some(), job.images.len(),
                admit_started);
            if let Some(source) = admitted.source {
                tracing::info!(tokens = tokens.len(), resume, kind = ?source.kind, frontier = source.frontier,
                    host = source.host, partial = source.partial, "prefix cache hit");
            }
            // A cold probe keeps the same chunk plan (identical numerics); it only skips the captures.
            let plan = if cache.enabled() {
                cuteafd_engine::prefix::plan_points(resume, tokens.len(), engine.prefill_capacity(),
                    &crate::shared::prefix::boundaries(&tokens, &markers), family.capture_reach(),
                    prefix.prefix_cache_min_tokens, prefix.points())
            } else {
                cuteafd_engine::prefix::plan_points(resume, tokens.len(), engine.prefill_capacity(), &[], 0, 0,
                    PointPolicy { gap: 0, boundaries: 0, per_request: 0 })
            };
            // A whole-prompt hit brings its first token's logits: nothing to prefill.
            let logits = admitted.after.and_then(|after| after.logits).map(|logits| logits.to_vec());
            let plan = if logits.is_some() { PointPlan::default() } else { plan };
            prefills.push(Prefill { job, constraint, tokens, done: resume, resume, plan, chunks: 0, cancelled: false,
                placement: admitted.placement, capacity, slot, logits, first: None, started: Instant::now(), busy: 0.0,
                id: admitted_total, ticket });
        }
        if prefills.due(!active.is_empty()) {
            // One chunk (a lane wave) of each waiting prompt (whole prompts with --decode-share 0).
            let finished = prefills.round(|p| {
                if p.done == p.tokens.len() {
                    return Ok(Chunk::Done);
                }
                if p.job.events.is_closed() {
                    p.cancelled = true;
                    anyhow::bail!("client went away");
                }
                let timer = Instant::now();
                let end = p.plan.chunks.get(p.chunks).copied().unwrap_or(p.tokens.len());
                let chunk = &p.tokens[p.done..end];
                let start = p.placement.len;
                let logits = engine.prefill_device(&mut p.placement, chunk,
                    transport.as_deref_mut().map(|t| (t, runtime)))?;
                if end == p.tokens.len() {
                    // The first token, while this prompt's logits are the workspace's; the
                    // prompt snapshot keeps that row (downloaded only when the cache retains).
                    let logits = logits.context("prefill produced no logits")?;
                    let mut batch = SelectBatch::default();
                    batch.push_next(p.job.sampling, p.constraint.as_mut(), p.placement.len as u64)?;
                    let selected = selector.select(&logits, &batch)?;
                    p.first = Some(take(p.constraint.as_mut(), &selected[0])?);
                    if cache.enabled() && !probe::cold(&p.job.probe) {
                        p.logits = Some(logits.row_host(&opened.library, 0)?);
                    }
                    if probe::wants_first(&p.job.probe) {
                        probe::device_rows(&opened.library, &p.job.probe, &logits, 0, 1, p.tokens.len())?;
                    }
                }
                // The chunk's tapped tail becomes drafter context before the next step.
                if let (Some(drafter), Some(slot)) = (drafter, p.slot) {
                    let n = chunk.len().min(TAP_ROWS);
                    let first = start + chunk.len() - n;
                    drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot, position: first + r })
                        .collect::<Vec<_>>())?;
                }
                p.done += chunk.len();
                p.busy += timer.elapsed().as_secs_f64();
                p.ticket.prefill(chunk.len(), p.chunks, p.plan.chunks.len(), timer);
                // Intermediate snapshot points this chunk ends at (off unless configured).
                for &(_, point) in p.plan.points.iter().filter(|&&(chunk, _)| chunk == p.chunks && !probe::cold(&p.job.probe)) {
                    if let Err(error) = cache.capture(&family, SnapshotKind::Prompt, &p.tokens[..point], &p.placement,
                        After::default()) {
                        tracing::warn!("snapshot point {point} not retained: {error:#}");
                    }
                }
                p.chunks += 1;
                Ok(if p.done == p.tokens.len() { Chunk::Done } else { Chunk::More })
            });
            for (mut p, prefilled) in finished {
                let (slot, placement, resume) = (p.slot, p.placement.clone(), p.resume);
                if let Err(error) = &prefilled {
                    if p.cancelled {
                        p.ticket.cancel();
                        // The client left during the prefill: keep what it computed for a retry.
                        if placement.len > resume && !probe::cold(&p.job.probe) {
                            if let Err(error) = cache.park(&family, &p.tokens[..placement.len], &placement) {
                                tracing::warn!("parking a cancelled prefill: {error:#}");
                            }
                        }
                    } else {
                        tracing::warn!("prefill failed: {error:#}");
                        let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    }
                    release(&family, &mut cache, &mut free_slots, &placement, slot);
                    continue;
                }
                if p.first.is_none() && p.logits.is_none() {
                    tracing::warn!("prefill produced no logits");
                    release(&family, &mut cache, &mut free_slots, &placement, slot);
                    continue;
                }
                let logits = p.logits.clone();
                if resume < p.tokens.len() {
                    tracing::debug!(tokens = p.tokens.len(), cached = resume,
                        elapsed_ms = p.started.elapsed().as_millis() as u64, busy_ms = (1e3 * p.busy) as u64, "prefill");
                }
                // The prompt snapshot, taken once the first token is out (it only enqueues copies).
                let prompt = (resume < p.tokens.len()).then(|| p.tokens.clone());
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &GlmPlacement| {
                    if let (Some(prompt), Some(logits)) = (&prompt, &logits) {
                        if let Err(error) = cache.capture(&family, SnapshotKind::Prompt, prompt, placement,
                            After::from_logits(logits, true)) {
                            tracing::warn!("prompt snapshot not retained: {error:#}");
                        }
                    }
                };
                let job_events = p.job.events.clone();
                let admitted = (|| -> Result<Active<'_>> {
                    let mut request = Active {
                        id: p.id,
                        prompt_tokens: p.tokens.len(),
                        digest: p.tokens.iter().fold(DIGEST_SEED, |d, &t| digest(d, t)),
                        history: p.tokens,
                        draft_limit: policy.copy,
                        draft_pause: 0,
                        slot,
                        drafts: DraftHistory::default(),
                        copy_drafts: DraftHistory::default(),
                        counts: [0; 5],
                        decoder: cuteafd_loader::streaming_token_decoder(&opened.snapshot, false)?,
                        job: p.job, constraint: p.constraint, placement: p.placement, draft_from: resume, turn: None,
                        capacity: p.capacity, next: 0, generated: 0, buffered: 0, started: Instant::now(),
                        ticket: p.ticket,
                    };
                    request.next = match (p.first, &logits) {
                        (Some(first), _) => first,
                        (None, Some(logits)) => {
                            if probe::wants_first(&request.job.probe) {
                                probe::host_row(&request.job.probe, request.prompt_tokens, logits);
                            }
                            request.select(logits)?
                        }
                        (None, None) => anyhow::bail!("prefill produced no first token"),
                    };
                    Ok(request)
                })();
                match admitted {
                    Ok(mut request) => {
                        let token = request.next;
                        let emitted = request.emit(token);
                        request.ticket.first(token);
                        retain_prompt(&mut cache, &request.placement);
                        match &emitted {
                            Ok(false) => active.push(request),
                            // Finished at its first token: its turn is its prompt snapshot.
                            Ok(true) | Err(_) => {
                                if let Err(error) = &emitted {
                                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                                }
                                request.ticket.done(request.generated);
                                release(&family, &mut cache, &mut free_slots, &request.placement, request.slot)
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!("admission failed: {error:#}");
                        let _ = job_events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        retain_prompt(&mut cache, &placement);
                        release(&family, &mut cache, &mut free_slots, &placement, slot);
                    }
                }
            }
            prefills.settle(!active.is_empty());
        }
        if active.is_empty() {
            continue;
        }
        let cycle = Instant::now();
        let mut tally = console::Step::begin(0);
        let (phases_before, engine_before) = (phases, tally.live().then(|| *engine.profile.borrow()));
        // Rows each sequence may add after its next token.
        let room = (DECODE_ROWS / active.len()).max(1) - 1;
        let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else {
            room.min(a.job.max_tokens - a.generated - 1).min(a.capacity - a.placement.len - 1) }).collect();
        // DFlash2 drafts after every next token, then the policy's counts.
        let timer = Instant::now();
        let drafted: Vec<Option<super::dflash::Draft>> = match drafter {
            Some(drafter) if skip.drafts() && active.iter().any(|a| a.slot.is_some()) => {
                let seqs: Vec<(usize, DraftSeq)> = active.iter().enumerate()
                    .filter_map(|(i, a)| a.slot.map(|slot| (i, DraftSeq { slot, anchor: a.next, position: a.placement.len,
                        valid_from: a.draft_from })))
                    .take(drafter.max_batch_sequences())
                    .collect();
                let timer = Instant::now();
                let drafts = drafter.draft_device(&seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), &engine.embedding,
                    super::dflash::TargetHead::Bf16(&engine.weights.head));
                cost.observe_draft(active.len(), timer.elapsed().as_secs_f64() * 1e3);
                let mut out = vec![None; active.len()];
                match drafts {
                    Ok(drafts) => {
                        for ((i, _), draft) in seqs.into_iter().zip(drafts) {
                            out[i] = Some(draft);
                        }
                    }
                    // Drafts only speed decoding up; the step verifies the next tokens alone.
                    Err(error) => tracing::warn!("DFlash2 draft failed: {error:#}"),
                }
                out
            }
            _ => vec![None; active.len()],
        };
        phases[0] += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        // Identical sequences (same tokens at the same position) route alike
        // and draft alike: the policy prices and plans them as one group.
        let key = |a: &Active<'_>| (a.placement.len, a.digest);
        let priors: Vec<Vec<f64>> = active.iter().zip(&drafted).map(|(a, draft)| {
            confidence.as_ref().map_or(Vec::new(), |policy| policy.prior(&a.drafts,
                draft.as_ref().map(|d| d.features.as_slice()), draft.as_ref().map_or(0, |d| d.tokens.len())))
        }).collect();
        let rates: Vec<Vec<f64>> = priors.iter().map(|prior| confidence.as_ref()
            .map_or_else(|| prior.clone(), |policy| policy.apply(prior))).collect();
        let inputs: Vec<dflash_policy::PlanInput<'_>> = active.iter().enumerate().map(|(i, a)| dflash_policy::PlanInput {
            key: key(a), history: &a.drafts, features: drafted[i].as_ref().map(|d| d.features.as_slice()),
            confidence: None, rates: Some(&rates[i]), limit: limits[i],
        }).collect();
        let planned = dflash_policy::plan_counts(&inputs, policy.fixed, &cost);
        drop(inputs);
        skip.after(drafted.iter().any(Option::is_some) && policy.fixed.is_none(), planned.iter().all(|&n| n == 0));
        // Each sequence verifies its next token, then its DFlash2 drafts, or
        // a copy-window draft when it agrees with them and runs longer.
        let copy_choice = copy_policy.then(|| {
            let proposals = active.iter().enumerate().map(|(i, a)| {
            if !a.job.sampling.is_greedy() { return Vec::new(); }
            copy_drafts(&a.history, limits[i].min(a.draft_limit))
        }).collect::<Vec<_>>();
            let copy_rates: Vec<_> = active.iter().zip(&proposals).map(|(a, copy)| a.copy_drafts.conditional(copy.len())).collect();
            let inputs: Vec<_> = active.iter().enumerate().map(|(i, a)| crate::shared::draft_policy::CopyInput {
                key: key(a), neural: drafted[i].as_ref().map_or(&[], |d| &d.tokens[..planned[i]]),
                confidence: &rates[i], copy: &proposals[i], copy_confidence: &copy_rates[i],
            }).collect();
            let (lengths, used) = crate::shared::draft_policy::compete_copies(&inputs,
                drafted.iter().any(Option::is_some), 0, &cost);
            (proposals, lengths, used)
        });
        let mut used_copy = vec![false; active.len()];
        let sequences: Vec<Vec<u32>> = active.iter_mut().enumerate().map(|(i, a)| {
            if a.draft_pause > 0 {
                a.draft_pause -= 1;
                if a.draft_pause == 0 {
                    a.draft_limit = 1;
                }
            }
            let dflash: &[u32] = drafted[i].as_ref().map_or(&[], |d| &d.tokens[..planned[i]]);
            let full: &[u32] = drafted[i].as_ref().map_or(&[], |d| &d.tokens);
            // `emit` already appended `next` to the history.
            let copy = copy_drafts(&a.history, limits[i].min(a.draft_limit));
            let agrees = copy.iter().zip(full).take_while(|(c, d)| c == d).count() >= dflash.len();
            let draft = if let Some((copies, lengths, used)) = &copy_choice {
                used_copy[i] = used[i];
                if used[i] { copies[i][..lengths[i]].to_vec() } else { dflash.to_vec() }
            } else if copy.len() > dflash.len() && agrees {
                used_copy[i] = true;
                copy
            } else {
                dflash.to_vec()
            };
            let mut rows: Vec<u32> = std::iter::once(a.next).chain(draft).collect();
            // Drafts the grammar rejects could never be kept: verify none of them.
            if let Some(state) = a.constraint.as_ref() {
                state.truncate_proposal(&mut rows)?;
            }
            Ok(rows)
        }).collect::<Result<_>>()?;
        let mut poisoned: Vec<Option<String>> = vec![None; sequences.len()];
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let distinct_rows: usize = active.iter().zip(&sequences).map(|(a, rows)| (key(a), rows))
            .collect::<std::collections::HashSet<_>>().iter().map(|(_, rows)| rows.len()).sum();
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        let mut cycle_host = timer.elapsed().as_secs_f64();
        phases[1] += cycle_host;
        // Each row draws at the position after it, masked along its sequence's drafts.
        let timer = Instant::now();
        let mut batch = SelectBatch::default();
        for (i, ((a, rows), &start)) in active.iter().zip(&sequences).zip(&starts).enumerate() {
            // A grammar failure fails that sequence alone, after the step.
            poisoned[i] = batch.push_sequence_isolated(a.job.sampling, a.constraint.as_ref(), rows, start as u64 + 1);
        }
        phases[2] += timer.elapsed().as_secs_f64();
        cycle_host += timer.elapsed().as_secs_f64();
        let mut rows: Vec<(&mut GlmPlacement, usize)> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.len())).collect();
        let timer = Instant::now();
        let step = engine.verify_device(&mut rows, &tokens, transport.as_deref_mut().map(|t| (t, runtime)))
            .and_then(|logits| logits.context("decode needs every layer"))
            .and_then(|logits| Ok((selector.select(&logits, &batch)?, logits)));
        let (selected, logits) = match step {
            Ok(step) => step,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(&family, &mut cache, &mut free_slots, &request.placement, request.slot);
                }
                continue;
            }
        };
        let verify_ms = timer.elapsed().as_secs_f64() * 1e3;
        cost.observe_verify(Shape { rows: tokens.len(), distinct: distinct_rows, sequences: sequences.len() }, verify_ms);
        phases[3] += verify_ms / 1e3;
        phases[6] += 1.0;
        phases[7] += tokens.len() as f64;
        let timer = Instant::now();
        let mut offset = 0;
        let mut context = Vec::new();
        let draft_list = &drafted;
        let trace_ref = &mut trace;
        let before: Vec<usize> = active.iter().map(|a| a.history.len()).collect();
        // A normal finish's turn snapshot keeps the row that produced the last token.
        let retain_turn = cache.enabled();
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(starts).enumerate()
            .map(|(i, ((request, rows), start))| {
            if let Some(error) = poisoned[i].take() {
                let _ = request.job.events.send(Err(NativeFailure::Worker(error)));
                offset += rows.len();
                return true;
            }
            let mut finished = false;
            for j in 0..rows.len() {
                // Rows 0..=j are committed; the token row j produces is next.
                request.placement.len = start + j + 1;
                probe::decode_row(&opened.library, &request.job.probe, &logits, offset + j, request.generated,
                    request.history.len());
                match take(request.constraint.as_mut(), &selected[offset + j]).and_then(|t| Ok((t, request.emit(t)?))) {
                    Ok((token, done)) => {
                        finished = done;
                        if done {
                            // A normal finish (the client took the last chunk): the row that
                            // produced the last token follows the turn snapshot.
                            if retain_turn && !probe::cold(&request.job.probe) {
                                request.turn = logits.row_host(&opened.library, offset + j)
                                    .inspect_err(|error| tracing::warn!("turn logits: {error:#}")).ok();
                            }
                        }
                        if done || rows.get(j + 1) != Some(&token) {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        finished = true;
                        break;
                    }
                }
            }
            let committed = request.placement.len - start;
            if let Some(slot) = request.slot.filter(|_| !finished) {
                context.extend((0..committed).map(|r| ContextRow { tap_row: offset + r, slot, position: start + r }));
            }
            offset += rows.len();
            let (drafted, accepted) = (rows.len() - 1, committed - 1);
            request.counts[0] += 1;
            if used_copy[i] {
                request.copy_drafts.observe(drafted.min(accepted + usize::from(!finished)), accepted);
                request.counts[3] += drafted;
                request.counts[4] += accepted;
            } else {
                request.counts[1] += drafted;
                request.counts[2] += accepted;
            }
            if planned[i] > 0 && (!used_copy[i] || (!refine_confidence && !copy_policy)) {
                if let Some(policy) = confidence.as_mut().filter(|_| !used_copy[i]) {
                    policy.observe(&request.drafts, draft_list[i].as_ref().map(|d| d.features.as_slice()),
                        &priors[i], drafted, accepted, finished, request.ticket.id());
                }
                request.drafts.observe(if refine_confidence || copy_policy { drafted.min(accepted + usize::from(!finished)) } else { planned[i] }, accepted);
            }
            if let Some(trace) = trace_ref.as_mut() {
                if let Err(error) = trace.cycle(request.id, start, rows.len(), committed, planned[i], used_copy[i],
                    draft_list[i].as_ref(), verify_ms) {
                    tracing::warn!("trace: {error:#}");
                }
            }
            // Adapt the copy-draft length to how much of it the model reproduced.
            if used_copy[i] || drafter.is_none() {
                if drafted > 0 && accepted == 0 {
                    request.draft_limit /= 2;
                    if request.draft_limit == 0 {
                        request.draft_pause = 8;
                    }
                } else if drafted > 0 && accepted == drafted {
                    request.draft_limit = (request.draft_limit * 2).clamp(1, policy.copy.max(1));
                }
            }
            finished
        }).collect();
        phases[4] += timer.elapsed().as_secs_f64();
        cycle_host += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        if let Some(drafter) = drafter {
            drafter.update(&context)?;
        }
        phases[5] += timer.elapsed().as_secs_f64();
        cost.observe_host(active.len(), 1e3 * (cycle_host + timer.elapsed().as_secs_f64()));
        for (i, request) in active.iter().enumerate() {
            let proposal = if used_copy[i] { &sequences[i][1..] } else { drafted[i].as_ref().map_or(&[][..], |d| &d.tokens) };
            tally.member(&request.ticket, proposal, sequences[i].len() - 1, &request.history[before[i]..],
                request.constraint.is_some(), finished[i]);
        }
        tally.end(|| {
            let host = |i: usize| 1e6 * (phases[i] - phases_before[i]);
            let engine_now = *engine.profile.borrow();
            let gpu = |i: usize| engine_before.map_or(f64::NAN, |before| 1e6 * (engine_now[i] - before[i]));
            vec![("draft", host(0)), ("plan", host(1)), ("rows", host(2)), ("verify", host(3)), ("emit", host(4)),
                ("update", host(5)), ("gpu", gpu(0)), ("experts", gpu(1)), ("head", gpu(2))]
        });
        for index in (0..active.len()).rev() {
            if !finished[index] {
                continue;
            }
            let mut request = active.remove(index);
            request.ticket.done(request.generated);
            if let Some(trace) = trace.as_mut() {
                trace.done(request.id, request.prompt_tokens, &request.history[request.prompt_tokens..])?;
            }
            requests += 1;
            generated_total += request.generated as u64;
            let seconds = request.started.elapsed().as_secs_f64();
            let engine_phases = std::mem::take(&mut *engine.profile.borrow_mut());
            let host = std::mem::take(&mut phases);
            let [steps, dflash, dflash_ok, copy, copy_ok] = request.counts;
            let per_step = |s: f64| 1e3 * s / host[6].max(1.0);
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps, dflash, dflash_ok, copy, copy_ok, gpu_wait_s = engine_phases[0],
                experts_s = engine_phases[1], head_s = engine_phases[2], "request complete");
            tracing::info!(steps = host[6], rows_per_step = host[7] / host[6].max(1.0), draft_ms = per_step(host[0]),
                plan_ms = per_step(host[1]), embed_ms = per_step(host[2]), verify_ms = per_step(host[3]),
                select_ms = per_step(host[4]), update_ms = per_step(host[5]),
                "host step phases since the last completed request (ms per step)");
            if let Some(row) = &request.turn {
                // The conversation so far: every committed row (the last token is not in it).
                let rows = &request.history[..request.placement.len];
                if let Err(error) = cache.capture(&family, SnapshotKind::Turn, rows, &request.placement,
                    After::from_logits(row, true)) {
                    tracing::warn!("turn snapshot not retained: {error:#}");
                }
            }
            release(&family, &mut cache, &mut free_slots, &request.placement, request.slot);
        }
        cache.tick();
        publish(stats, requests, generated_total, active.len(), prefills.len(), &cache);
        console::gauges(|| console::Gauges::prefix_cache(&cache, active.len(), prefills.len(), receive.len() + kv_waiter.len()));
        prefills.stepped(cycle.elapsed().as_secs_f64());
    }
}
