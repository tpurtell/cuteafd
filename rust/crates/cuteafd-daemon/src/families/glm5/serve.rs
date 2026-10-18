//! OpenAI-compatible API over the GLM engine: continuous batching with one
//! prefill per admitted request and one decode-shaped step for every active
//! sequence. With a DFlash2 drafter (--draft), every step first drafts after
//! each sequence's next token on the GPU and verifies as many drafts as the
//! shared resource policy prices worthwhile. History copies extend only a
//! whole-window agreeing greedy neural draft, never replace it.
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
use crate::shared::draft_policy::DraftHistory;
use crate::shared::draft::evidence::{Censor, DraftSource};
use super::engine::{GlmEngine, GlmPlacement, DECODE_ROWS};
use super::prefix::GlmPrefix;
use crate::shared::prefix::CudaCopyEngine;
use crate::shared::prefix::{PrefixArgs, Toggle};
use crate::shared::probe;
use cuteafd_engine::prefix::{After, PointPlan, PointPolicy, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};

use super::{open, Opened};
use crate::shared::token_io::{SelectBatch, SelectPlacement, TokenSelector};
use crate::shared::prefill_share::{Chunk, DecodeShareArgs};
use anyhow::{ensure, Context, Result};
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
    if cuteafd_loader::plan::exlr8::recognizes(&args.engine.snapshot)? {
        let cfg = cuteafd_loader::families::glm5::GlmDsaConfig::read(&args.engine.snapshot)?;
        super::native_activation(&args.engine, cfg.first_moe_layer, cfg.layers, true)?;
    }
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
    let policy = Policy { fixed: args.draft_fixed,
        decode_share: args.decode_share };
    let prefix = args.prefix.clone();
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id)));
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, policy, prefix));
    let max_context = ready_rx.await.context("engine failed before it was ready")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    cuteafd_bench::context::phase("engine loaded");
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, Duration::from_secs(25),
        hub.clone(), api.serve(profile.clone(), &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    // The ready ledger (placement gate): one report at readiness, before any request captures
    // lazy graphs or loads more code.
    crate::shared::memory_report::log("ready");
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
    let policy = args.draft_fixed.map_or_else(|| "shared resource-priced + agreed copies".to_string(), |n| format!("fixed {n}"));
    layout.speculator = args.engine.draft.as_ref().map(|_| console::Speculator {
        name: "DFlash2".into(), positions: 8, policy });
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

/// A diagnostic fixed DFlash2 count, otherwise shared resource-priced selection.
#[derive(Debug, Clone, Copy)]
struct Policy {
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
        anyhow::ensure!(transport.is_some() || engine.skip_routed() || engine.has_tp2(), "serve-glm needs Spark peers, local TP2 or --skip-routed-experts");
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
    /// DFlash2 ring slot and draft outcomes (None without a drafter slot).
    slot: Option<usize>,
    drafts: DraftHistory,
    /// Steps and neural/agreed-tail drafts verified and accepted.
    counts: [usize; 3],
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

/// Fixed diagnostic counts still share one verifier budget, anchors included.
fn trim_budget(lengths: &mut [usize], budget: usize) -> bool {
    let mut rows = lengths.len() + lengths.iter().sum::<usize>();
    let trimmed = rows > budget;
    while rows > budget {
        let Some(index) = (0..lengths.len()).filter(|&i| lengths[i] > 0).max_by_key(|&i| lengths[i]) else { break };
        lengths[index] -= 1;
        rows -= 1;
    }
    trimmed
}

/// Stops censor an unjudged continuation, never a disagreeing draft token.
fn finish_censor(finished: bool, committed: usize, rows: &[u32], last: Option<u32>, stops: &[u32]) -> Option<Censor> {
    if !finished { return None; }
    let next = rows.get(committed);
    if next.is_some() && next != last.as_ref() { return None; }
    Some(if last.is_some_and(|token| stops.contains(&token)) { Censor::Eos } else { Censor::OutputLimit })
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
struct Trace(crate::shared::speculation_trace::Trace);

impl Trace {
    fn open() -> Result<Option<Self>> {
        Ok(crate::shared::speculation_trace::Trace::open()?.map(Self))
    }

    #[allow(clippy::too_many_arguments)]
    fn cycle(&mut self, id: u64, position: usize, rows: usize, committed: usize, planned: usize, copy: bool,
        draft: Option<&super::dflash::Draft>, verify_ms: f64) -> Result<()> {
        let line = serde_json::json!({"kind": "cycle", "id": id, "position": position, "rows": rows,
            "committed": committed, "planned": planned, "copy": copy, "verify_ms": verify_ms,
            "draft": draft.map(|d| &d.tokens), "features": draft.map(|d| &d.features)});
        self.0.write(&line)?;
        Ok(())
    }

    fn done(&mut self, id: u64, prompt_tokens: usize, generated: &[u32]) -> Result<()> {
        self.0.write(&serde_json::json!({"kind": "done", "id": id, "prompt_tokens": prompt_tokens,
            "generated": generated}))?;
        Ok(())
    }
}

/// The prefix cache over `engine` (always present: with zero entries it is the page allocator).
fn prefix_cache<'e, 'a>(engine: &'e GlmEngine<'a>, args: &PrefixArgs)
    -> Result<(GlmPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let entries = args.prefix_cache_entries;
    let family = GlmPrefix::new(engine, args.prefix_partial == Toggle::On)?;
    let template = engine.paged_buffers().first().map(|b| b.0).context("GLM has no layers")?;
    // The pinned host tier: one GPU's pages, or under context each page from its owner
    // (both GPUs' pools registered, so every copy runs on its owner's copy streams). A heads
    // split keeps replicated pages on both GPUs and stays device-resident.
    let host = if engine.ranks() > 1 && !engine.context_kv() && !engine.layers_kv() {
        if args.host_cache_bytes.enabled() && args.prefix_cache_entries > 0 {
            tracing::warn!("GLM heads split: the prefix cache's host tier is off (device-resident snapshots only)");
        }
        None
    } else if engine.ranks() > 1 {
        // SAFETY of the registration: the engine (and its pools) outlives the cache, which
        // `schedule` drops first.
        args.host_config(family.layout(), engine.max_context)?.map(|config|
            CudaCopyEngine::registered(engine.library, &family.buffers()).map(|copy| (config, copy))).transpose()?
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
            "prefilling": prefilling, "prefix_cache": cache.stats(),
            "draft_policy": crate::shared::draft::resource::snapshot()});
    }
}

/// Weak GLM 5.3 priors from its EXL3 K4/TP4 D4 run, not GLM Flash's fits.
/// C1 residual was 1355 us; drafter medians were 3634 us at C1 and 7288 us at C4.
fn draft_seed() -> cuteafd_core::PolicySeed {
    cuteafd_core::PolicySeed { round: [1250., 20., 100.], draft: [2400., 1220.], samples: 0.01,
        cold_drafts: 5, warm_rounds: 24, warm_error: 0.08 }
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
    use crate::shared::draft::{resource, runtime::SharedPolicy};
    let mut shared = drafter.map(|d| {
        let tp2 = engine.tp2_layers();
        let remote = if engine.weights.layers.iter().enumerate().any(|(layer, weights)|
            !weights.dense && !tp2.as_ref().is_some_and(|layers| layers.contains(&layer))) {
            transport.as_ref().context("GLM installed Spark transport missing")?.transport.installed_grouping()?
        } else { Vec::new() };
        let catalog = cuteafd_loader::read_expert_catalog(&opened.snapshot)?;
        let homes = engine.weights.layers.iter().enumerate().map(|(layer, weights)| {
            if weights.dense { return Ok(resource::LayerHome::Dense); }
            let local = tp2.as_ref().is_some_and(|layers| layers.contains(&layer));
            let tp = if local { 2 } else { remote.len() };
            ensure!(tp > 0, "GLM layer {layer} has no installed expert ranks");
            let slice_bytes = match (catalog.fp8(), catalog.exl3()) {
                (Some(_), _) => engine.draft_weight_read_bytes(layer, &remote)? as f64,
                (_, Some(manifest)) => resource::exl3_slice_bytes(manifest, layer, tp)?,
                _ => anyhow::bail!("GLM routed expert format cannot be priced"),
            };
            Ok(if local { resource::LayerHome::Local { slice_bytes } }
                else { resource::LayerHome::Remote { slice_bytes } })
        }).collect::<Result<Vec<_>>>()?;
        let geometry = resource::geometry_with_grouping(&homes, engine.cfg.experts, engine.cfg.topk, DECODE_ROWS,
            d.cfg.drafts(), resource::COPY_EXTENSION, |layer, rows| engine.draft_group_rows(layer, rows, &remote))?;
        engine.arm_draft_probe()?;
        SharedPolicy::with_geometry(geometry, resource::Prior::Selector(d.selector_fit("glm5", false)?), draft_seed())
    }).transpose()?;
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
        engine.require_live()?;
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("expert wire unavailable until restart: {reason}");
        }
        while active.len() + prefills.len() < max_sequences {
            let busy = !active.is_empty() || !prefills.is_empty();
            let job = match kv_waiter.poll(cache.pool().admission_free(), cache.pool().release_epoch(), busy, |job| job.events.is_closed()) {
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
            let tokens = probe::prompt_ids(&job.probe, || Ok(job.encode_prompt(&tokenizer)?))?;
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
                admit_started, job.usage.clone());
            if let (Some(usage), Some(session)) = (&job.usage, admitted.source.as_ref().and_then(|s| s.session.as_ref())) {
                usage.session(session.clone(), "prefix");
            }
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
                cuteafd_transport::route_log::step(|| vec![(p.id, chunk.len() as u32)]);
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
                    cache.capture_session(p.ticket.session());
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
                            cache.capture_session(p.ticket.session());
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
                let session = p.ticket.session();
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &GlmPlacement| {
                    if let (Some(prompt), Some(logits)) = (&prompt, &logits) {
                        cache.capture_session(session.clone());
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
                        history: p.tokens,
                        slot,
                        drafts: DraftHistory::default(),
                        counts: [0; 3],
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
        let mut round_clock = crate::shared::draft::clock::RoundClock::at(cycle);
        let mut drafted_width = 0;
        let mut drafted_requests = 0;
        let mut tally = console::Step::begin(0);
        let (phases_before, engine_before) = (phases, tally.live().then(|| *engine.profile.borrow()));
        // Rows each sequence may add after its next token.
        let room = DECODE_ROWS.saturating_sub(active.len());
        let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else {
            room.min(a.job.max_tokens - a.generated - 1).min(a.capacity - a.placement.len - 1) }).collect();
        // Skip/block decisions use the same resource prices as post-draft lengths.
        let eligible: Vec<_> = active.iter().zip(&limits).filter(|(a, limit)| a.slot.is_some() && **limit > 0)
            .map(|(a, &limit)| (a.ticket.id(), limit)).take(drafter.map_or(0, |d| d.max_batch_sequences())).collect();
        let run_draft = if eligible.is_empty() { false } else if policy.fixed.is_some()
            || active.iter().any(|a| a.drafts.cold() && eligible.iter().any(|&(id, _)| id == a.ticket.id())) { true } else {
            let shared = shared.as_mut().expect("eligible draft slots have a shared policy");
            let width = drafter.unwrap().cfg.drafts();
            // Undrafted members still contribute their anchors, routes and emitted tokens.
            let requests: Vec<_> = active.iter().map(|a| (a.ticket.id(),
                eligible.iter().find(|&&(id, _)| id == a.ticket.id()).map_or(0, |&(_, room)| room))).collect();
            resource::choose_block(&mut shared.policy, &requests, width, eligible.len(), DECODE_ROWS, false)?
        };
        let timer = Instant::now();
        let drafted: Vec<Option<super::dflash::Draft>> = match drafter {
            Some(drafter) if run_draft => {
                let seqs: Vec<(usize, DraftSeq)> = active.iter().enumerate()
                    .filter_map(|(i, a)| a.slot.filter(|_| limits[i] > 0).map(|slot| (i, DraftSeq { slot, anchor: a.next, position: a.placement.len,
                        valid_from: a.draft_from })))
                    .take(drafter.max_batch_sequences())
                    .collect();
                drafted_requests = seqs.len();
                let drafts = round_clock.draft(|| drafter.draft_device(&seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), &engine.embedding,
                    super::dflash::TargetHead::Bf16(&engine.weights.head)));
                drafted_width = drafter.cfg.drafts();

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
        let mut priors: Vec<_> = active.iter().zip(&drafted).map(|(a, draft)| match (&shared, draft) {
            (Some(shared), Some(d)) => shared.prior(&a.drafts,
                &resource::DraftEvidence::Selector(&d.features), 0),
            _ => Vec::new(),
        }).collect();
        let copies: Vec<_> = active.iter().zip(&drafted).zip(&limits).zip(&mut priors)
            .map(|(((a, draft), &limit), prior)| match (&shared, draft) {
                (Some(shared), Some(d)) if policy.fixed.is_none() => resource::copy_continuation(&shared.policy, active.len(),
                    a.ticket.id(), &a.history, &d.tokens, prior, limit, a.job.sampling.is_greedy()),
                _ => None,
            }).collect();
        let mut predicted = None;
        let mut planned = match (policy.fixed, shared.as_mut()) {
            (Some(fixed), _) => drafted.iter().zip(&limits).map(|(draft, &limit)|
                draft.as_ref().map_or(0, |d| fixed.min(limit).min(d.tokens.len()))).collect(),
            (None, Some(shared)) if drafted_width > 0 => {
                let candidates: Vec<_> = active.iter().zip(&priors).zip(&limits).map(|((a, prior), &limit)|
                    resource::Candidate { id: a.ticket.id(), prior, limit, selector: true, cold: a.drafts.cold() }).collect();
                let selection = resource::select_drafted(&mut shared.policy, &candidates, drafted_width, DECODE_ROWS, drafted_requests);
                predicted = selection.predicted;
                selection.lengths
            }
            _ => vec![0; active.len()],
        };
        if trim_budget(&mut planned, DECODE_ROWS) { predicted = None; }
        let sequences: Vec<Vec<u32>> = active.iter().enumerate().map(|(i, a)| {
            let full = copies[i].as_ref().map(|c| c.tokens.as_slice())
                .unwrap_or_else(|| drafted[i].as_ref().map_or(&[][..], |d| d.tokens.as_slice()));
            let draft = full.iter().take(planned[i]).copied();
            let mut rows: Vec<u32> = std::iter::once(a.next).chain(draft).collect();
            if let Some(state) = a.constraint.as_ref() { state.truncate_proposal(&mut rows)?; }
            Ok(rows)
        }).collect::<Result<_>>()?;
        let mut poisoned: Vec<Option<String>> = vec![None; sequences.len()];
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        phases[1] += timer.elapsed().as_secs_f64();
        // Each row draws at the position after it, masked along its sequence's drafts.
        let timer = Instant::now();
        let mut batch = SelectBatch::default();
        for (i, ((a, rows), &start)) in active.iter().zip(&sequences).zip(&starts).enumerate() {
            // A grammar failure fails that sequence alone, after the step.
            poisoned[i] = batch.push_sequence_isolated(a.job.sampling, a.constraint.as_ref(), rows, start as u64 + 1);
        }
        phases[2] += timer.elapsed().as_secs_f64();
        cuteafd_transport::route_log::step(|| active.iter().zip(&sequences).map(|(a, s)| (a.id, s.len() as u32)).collect());
        let mut rows: Vec<(&mut GlmPlacement, usize)> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.len())).collect();
        let timer = Instant::now();
        engine.probe_begin(tokens.len());
        let step = engine.verify_device(&mut rows, &tokens, transport.as_deref_mut().map(|t| (t, runtime)))
            .and_then(|logits| logits.context("decode needs every layer"))
            .and_then(|logits| Ok((selector.select(&logits, &batch)?, logits)));
        engine.probe_end();
        let (selected, logits) = match step {
            Ok(step) => step,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    if let Some(shared) = shared.as_mut() { shared.policy.release(request.ticket.id()); }
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(&family, &mut cache, &mut free_slots, &request.placement, request.slot);
                }
                continue;
            }
        };
        let verify_ms = timer.elapsed().as_secs_f64() * 1e3;
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
        let mut verified = vec![(0usize, 1usize, Some(Censor::Cancelled)); active.len()];
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).enumerate()
            .map(|(i, ((request, rows), start))| {
            if let Some(error) = poisoned[i].take() {
                let _ = request.job.events.send(Err(NativeFailure::Worker(error)));
                verified[i] = (rows.len(), 1, Some(Censor::Cancelled));
                offset += rows.len();
                return true;
            }
            let mut finished = false;
            let mut cancelled = false;
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
                        cancelled = true;
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
            let censor = if cancelled { Some(Censor::Cancelled) } else {
                finish_censor(finished, committed, rows, request.history.last().copied(), &request.job.stop_token_ids)
            };
            verified[i] = (rows.len(), committed, censor);
            request.counts[0] += 1;
            request.counts[1] += drafted;
            request.counts[2] += accepted;
            if planned[i] > 0 {
                let neural = drafted.min(copies[i].as_ref().map_or_else(
                    || draft_list[i].as_ref().map_or(0, |d| d.tokens.len()), |c| c.neural));
                request.drafts.observe(neural.min(accepted + usize::from(censor.is_none())), accepted.min(neural));
            }
            if let Some(trace) = trace_ref.as_mut() {
                let copy = copies[i].as_ref().is_some_and(|copy| copy.uses_copy(drafted));
                if let Err(error) = trace.cycle(request.id, *start, rows.len(), committed, planned[i], copy,
                    draft_list[i].as_ref(), verify_ms) {
                    tracing::warn!("trace: {error:#}");
                }
            }
            finished
        }).collect();
        phases[4] += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        if let Some(drafter) = drafter {
            drafter.update(&context)?;
        }
        phases[5] += timer.elapsed().as_secs_f64();
        if let Some(shared) = shared.as_mut() {
            let requests: Vec<_> = active.iter().enumerate().map(|(i, a)| {
                let (rows, accepted, censor) = verified[i];
                resource::Verified { id: a.ticket.id(), rows, accepted,
                    source: if rows > 1 && drafted[i].is_some() { DraftSource::Neural }
                        else { DraftSource::Undrafted },
                    prior: drafted[i].as_ref().map(|_| priors[i].as_slice()),
                    features: drafted[i].as_ref().map(|d| d.features.as_slice()),
                    censor }
            }).collect();
            let times = round_clock.observe();
            if let Some(Err(error)) = engine.probe_finish(|routes, layer_us| resource::observe_drafted(
                &mut shared.policy, &requests, routes, layer_us, times, drafted_width, predicted, Some(drafted_requests))) {
                tracing::debug!(%error, "GLM draft round not observed");
            }
            if shared.policy.stats().rounds % 32 == 1 {
                resource::publish(&shared.policy, shared.policy.stats().rounds, 0, None);
            }
        }
        for (i, request) in active.iter().enumerate() {
            let proposal = &sequences[i][1..];
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
            if let Some(shared) = shared.as_mut() { shared.policy.release(request.ticket.id()); }
            request.ticket.done(request.generated);
            if let Some(trace) = trace.as_mut() {
                trace.done(request.id, request.prompt_tokens, &request.history[request.prompt_tokens..])?;
            }
            requests += 1;
            generated_total += request.generated as u64;
            let seconds = request.started.elapsed().as_secs_f64();
            let engine_phases = std::mem::take(&mut *engine.profile.borrow_mut());
            let host = std::mem::take(&mut phases);
            let [steps, dflash, dflash_ok] = request.counts;
            let per_step = |s: f64| 1e3 * s / host[6].max(1.0);
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps, dflash, dflash_ok, gpu_wait_s = engine_phases[0],
                experts_s = engine_phases[1], head_s = engine_phases[2], "request complete");
            tracing::info!(steps = host[6], rows_per_step = host[7] / host[6].max(1.0), draft_ms = per_step(host[0]),
                plan_ms = per_step(host[1]), embed_ms = per_step(host[2]), verify_ms = per_step(host[3]),
                select_ms = per_step(host[4]), update_ms = per_step(host[5]),
                "host step phases since the last completed request (ms per step)");
            if let Some(row) = &request.turn {
                // The conversation so far: every committed row (the last token is not in it).
                let rows = &request.history[..request.placement.len];
                cache.capture_session(request.ticket.session());
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

#[cfg(test)]
mod draft_tests {
    use super::*;

    #[test]
    fn glm53_seed_uses_its_own_weak_drafter_prices() {
        let seed = draft_seed();
        assert_eq!(seed.samples, 0.01);
        assert_eq!(seed.cold_drafts, 5);
        assert!((seed.draft[0] + seed.draft[1] - 3634.).abs() < 50.);
        assert!((seed.draft[0] + 4. * seed.draft[1] - 7288.).abs() < 50.);
        assert_ne!(seed.draft, crate::shared::draft::resource::seed().draft);
    }

    #[test]
    fn stops_censor_only_unjudged_agreed_drafts() {
        assert_eq!(finish_censor(false, 2, &[1, 5, 9], Some(9), &[9]), None);
        assert_eq!(finish_censor(true, 2, &[1, 5, 9, 4], Some(9), &[9]), Some(Censor::Eos));
        assert_eq!(finish_censor(true, 2, &[1, 5, 7, 4], Some(9), &[9]), None);
        assert_eq!(finish_censor(true, 1, &[1, 5, 6], Some(5), &[9]), Some(Censor::OutputLimit));
        assert_eq!(finish_censor(true, 3, &[1, 5, 6], Some(8), &[9]), Some(Censor::OutputLimit));
    }

    #[test]
    fn fixed_counts_keep_every_anchor_inside_the_global_budget() {
        let mut lengths = [7; 16];
        assert!(trim_budget(&mut lengths, 64));
        assert_eq!(lengths.len() + lengths.iter().sum::<usize>(), 64);
        assert!(lengths.iter().all(|&length| length == 3));
        let mut lengths = [32, 0, 0, 0];
        assert!(!trim_budget(&mut lengths, 64));
        assert_eq!(lengths, [32, 0, 0, 0]);
    }
}
