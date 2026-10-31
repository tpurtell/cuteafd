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
use crate::shared::speculation::dflash::{ContextRow, DraftSeq, TAP_ROWS};
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
use cuteafd_api::openai::chat::{glm5::{GlmEncoding, GlmThinkingOff}, mistral::MistralEncoding};
use cuteafd_api::openai::{
    InferenceChunk, InferenceFinishReason, ModelEncoding, ModelProfile, NativeFailure, NativeLimits, NativeRequest,
    PromptUsage,
};
use crate::shared::spark_intake::SparkLink;
use crate::shared::console;
use crate::shared::serve::lanes::{self, LaneCommit, LaneLease, LaneSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use cuteafd_engine::media::{EmbeddingCache, MediaAdmission, MediaKeys, MediaPoll, MediaWaiter, RequestMedia};
use super::ml4_media;
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
    #[arg(long, default_value_t = cuteafd_loader::families::glm5::draft_representation::DEFAULT_MAX_SEQUENCES)]
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
    /// Resolved global vision policy, assigned before dispatch (Mistral Large 4's tower).
    #[arg(skip = cuteafd_loader::plan::MediaMode::Auto)]
    pub vision: cuteafd_loader::plan::MediaMode,
    /// Host embedding-cache quota; default min(8 GiB, 5% RAM).
    #[arg(long, value_parser = crate::shared::prefix::parse_bytes)]
    pub media_cache_bytes: Option<u64>,
}

fn model_id(snapshot: &std::path::Path) -> Option<String> {
    snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    })
}

/// The chat contract the checkpoint itself speaks: its tokenizer's control tokens
/// and its own template (Mistral instruct for Mistral Large 4), never the engine variant.
fn chat_encoding(snapshot: &std::path::Path, thinking_off: GlmThinkingOff) -> Result<ModelEncoding> {
    Ok(if MistralEncoding::recognizes(snapshot)? {
        ModelEncoding::Mistral(Arc::new(MistralEncoding::from_snapshot(snapshot)?))
    } else {
        ModelEncoding::Glm(Arc::new(GlmEncoding::from_snapshot(snapshot)?.with_thinking_off(thinking_off)))
    })
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    if cuteafd_loader::plan::exlr8::recognizes(&args.engine.snapshot)? {
        let cfg = cuteafd_loader::families::glm5::GlmDsaConfig::read(&args.engine.snapshot)?;
        super::native_activation(&args.engine, cfg.first_moe_layer, cfg.layers, true)?;
    }
    let api = args.api.load()?;
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&snapshot)).context("model id")?,
        chat_encoding(&snapshot, args.thinking_off)?,
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    if engine_args.draft_context_slots.is_none() {
        engine_args.draft_context_slots = Some(cuteafd_loader::families::glm5::draft_representation::serving_draft_context_slots(
            engine_args.draft_sequences, args.max_sequences, None));
    }
    // A window-ring drafter's prefix marks: one per arena slot, sized like the planner's.
    if engine_args.dspark && args.prefix.prefix_cache_entries > 0 {
        let layout = cuteafd_loader::placement::families::glm5::ml4_draft_layout(&engine_args.snapshot,
            engine_args.draft_context_slots.unwrap_or(20), engine_args.draft_sequences, engine_args.max_context, 1)?;
        engine_args.draft_window_marks = args.prefix.mark_rule(args.max_sequences)
            .slots(cuteafd_loader::placement::families::glm5::draft_window_mark_bytes(&layout)?);
    }
    let (worker_stats, max_sequences) = (stats.clone(), args.max_sequences);
    let policy = Policy { fixed: args.draft_fixed,
        decode_share: args.decode_share };
    let prefix = args.prefix.clone();
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id)));
    let (vision, media_cache_bytes) = (args.vision, args.media_cache_bytes);
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, policy, prefix, vision, media_cache_bytes));
    let (max_context, media) = ready_rx.await.context("engine failed before it was ready")??;
    let mut profile = profile;
    if let Some((preparer, health)) = media {
        profile = profile.with_loaded_vision(preparer);
        profile.vision_health = health;
    }
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

type VisionReady = Option<(Arc<cuteafd_api::openai::media::MediaPreparer>, Option<Arc<std::sync::atomic::AtomicBool>>)>;

#[allow(clippy::too_many_arguments)]
fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<(usize, VisionReady)>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    policy: Policy, prefix: PrefixArgs, vision: cuteafd_loader::plan::MediaMode, media_cache_bytes: Option<u64>) -> Result<()> {
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    // The tower loads before the LM so its admitted bytes are in the LM's live baseline.
    let (vision, prefix) = match ml4_media::ReadyVision::load(&args, &opened.library, vision, &prefix, media_cache_bytes) {
        Ok(vision) => vision,
        Err(error) => { let _ = ready.send(Err(error)); return Ok(()); }
    };
    let preparer = vision.as_ref().map(|v| v.preparer.clone());
    let expander = vision.as_ref().map(|v| v.expander);
    let (encoder, bytes) = vision.map_or((ml4_media::Encoder::Off, 0), |v| (v.encoder, v.cache_bytes));
    let health = encoder.health_handle();
    let mut media = MediaAdmission::new(EmbeddingCache::new(bytes), encoder, 16);
    let mut ready = Some(ready);
    let result = opened.with_engine(&args, |engine, transport, runtime| {
        anyhow::ensure!(transport.is_some() || engine.skip_routed() || engine.has_tp2(), "serve-glm needs Spark peers, local TP2 or --skip-routed-experts");
        let ranks = args.peers.as_deref().map_or(4, |peers| peers.split(',').count());
        let spark = transport.is_some();
        console::layer_classes(engine.weights.layers.iter().map(|l| console::layer_class(l.dense, spark)).collect());
        anyhow::ensure!(preparer.is_none() || media.encoder().available(), "vision encoder unavailable before readiness");
        schedule(engine, &opened, &mut receive, transport, runtime, &stats, max_sequences.min(DECODE_ROWS), policy,
            ranks, &prefix, args.token_io.token_select, &mut media, expander.as_ref(), || {
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Ok((engine.max_context, preparer.clone().map(|p| (p, health.clone())))));
                }
            })
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| (args.max_context, preparer.map(|p| (p, health))))
            .map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

/// A prompt ready for prefix-cache admission: its keyed tokens and features.
struct Admit<'a> {
    prompt: ml4_media::Prompt,
    media: RequestMedia,
    /// Kept across a KV deferral.
    constraint: Option<crate::shared::constraints::State<'a>>,
}

/// An admitted prompt waiting for its remaining prefill chunks.
struct Prefill<'a> {
    job: NativeRequest,
    constraint: Option<crate::shared::constraints::State<'a>>,
    tokens: Vec<u32>,
    /// The prefix cache's keyed copy of `tokens` (image rows keyed by their image; text: the tokens).
    keys: MediaKeys,
    /// The prompt's image features, injected where its chunks embed them.
    media: RequestMedia,
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
    /// The prompt's image spans (the turn snapshot keys its rows like the prompt's).
    spans: Vec<cuteafd_loader::media::MediaSpan>,
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
        draft: Option<&crate::shared::speculation::dflash::Draft>, verify_ms: f64) -> Result<()> {
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
fn prefix_cache<'e, 'a>(engine: &'e GlmEngine<'a>, args: &PrefixArgs, concurrency: usize)
    -> Result<(GlmPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let args = &args.with_scoring(if engine.full_prefill_logits { engine.prefill_capacity() } else { 64 },
        engine.prefill_rows, engine.cfg.vocab_size)?;
    let entries = args.prefix_cache_entries;
    let family = GlmPrefix::new(engine, args.prefix_partial == Toggle::On, engine.draft_window_marks)?;
    let template = engine.paged_buffers().iter().find_map(|(records, _)| records.map(|(buffer, _)| buffer))
        .context("GLM has no latent-owning layers")?;
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
        args.host_config(family.layout(), engine.max_context, concurrency)?.map(|config|
            CudaCopyEngine::registered(engine.library, &family.buffers()).map(|copy| (config, copy))).transpose()?
    } else {
        args.host_tier(engine.library, template, family.layout(), engine.max_context, concurrency)?
    };
    let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
    let layout = family.layout();
    // Arena slots only for a window-ring drafter's marks (the count admission charged).
    let mark_slots = if layout.mark_bytes > 0 { engine.draft_window_marks } else { 0 };
    let config = PrefixConfig { entries, mark_slots, keep_logits: true, min_tokens: args.prefix_cache_min_tokens };
    let cache = PrefixCache::new(layout, config, host)?;
    tracing::info!(entries, page_bytes = layout.page_bytes, pages = layout.pages, host_bytes,
        rule = ?layout.rule, points = ?args.points(), "GLM prefix cache");
    cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, engine.max_pool_tokens_no_onboard, layout.pages as u64,
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

/// A queued chunk eligible for immediate shared admission.
fn packed_chunk(p: &Prefill<'_>) -> Option<(usize, usize)> {
    // A prompt with images prefills alone, where its chunks inject the features.
    if p.job.events.is_closed() || p.done == p.tokens.len() || p.logits.is_some()
        || !p.media.spans().is_empty() { return None; }
    Some((p.placement.len, p.plan.chunks.get(p.chunks).copied().unwrap_or(p.tokens.len()) - p.done))
}

fn prefill_group(engine: &GlmEngine<'_>, family: &GlmPrefix<'_, '_>,
    cache: &mut PrefixCache<CudaCopyEngine<'_>>, selector: &mut TokenSelector<'_>, group: &mut [Prefill<'_>])
    -> Vec<Result<Chunk>> {
    let timer = Instant::now();
    let ends = group.iter().map(|p| p.plan.chunks.get(p.chunks).copied().unwrap_or(p.tokens.len())).collect::<Vec<_>>();
    let counts = group.iter().zip(&ends).map(|(p, end)| end - p.done).collect::<Vec<_>>();
    cuteafd_transport::route_log::step(|| group.iter().zip(&counts).map(|(p, &n)| (p.id, n as u32)).collect());
    let result = (|| -> Result<Vec<Result<()>>> {
        let mut sequences = Vec::new();
        let mut outputs = Vec::new();
        for (p, &end) in group.iter_mut().zip(&ends) {
            let Prefill { job, constraint, tokens, done, placement, first, logits, .. } = p;
            sequences.push((placement, &tokens[*done..end]));
            outputs.push((&*job, constraint, first, logits, tokens.len(), end));
        }
        let heads = outputs.iter().map(|(job, _, _, _, len, end)| (*job, end == len)).collect::<Vec<_>>();
        let (segments, outcomes) = engine.prefill_packed_selected(&mut sequences,
            &mut |i| heads[i].1 && !heads[i].0.events.is_closed(), &mut |i, logits| {
            let (job, constraint, first, retained, len, end) = &mut outputs[i];
            if job.events.is_closed() { return Ok(Err(anyhow::anyhow!("client went away"))); }
            if end != len { return Ok(Ok(())); }
            let mut batch = SelectBatch::default();
            if let Err(error) = batch.push_next(job.sampling, constraint.as_mut(), *end as u64) {
                return Ok(Err(error));
            }
            let selected = selector.select(&logits, &batch)?;
            match take(constraint.as_mut(), &selected[0]) {
                Ok(token) => **first = Some(token),
                Err(error) => return Ok(Err(error)),
            }
            if cache.enabled() && !probe::cold(&job.probe) { **retained = Some(logits.row_host(engine.library, 0)?); }
            if probe::wants_first(&job.probe) {
                probe::device_rows(engine.library, &job.probe, &logits, 0, 1, *len)?;
            }
            Ok(Ok(()))
        })?;
        drop((sequences, outputs));
        if let Some(drafter) = &engine.drafter {
            for (p, s) in group.iter().zip(&segments) {
                if let Some(slot) = p.slot {
                    drafter.update(&(0..s.tap_rows).map(|r| ContextRow { tap_row: s.tap_offset + r, slot,
                        position: s.start + s.rows - s.tap_rows + r }).collect::<Vec<_>>())?;
                }
            }
        }
        Ok(outcomes)
    })();
    let elapsed = timer.elapsed().as_secs_f64();
    for (((p, &end), &count), share) in group.iter_mut().zip(&ends).zip(&counts)
        .zip(crate::shared::prefill_packing::row_shares(&counts)) {
        p.done = end;
        p.cancelled = p.job.events.is_closed();
        p.busy += elapsed * share;
        p.ticket.prefill(count, p.chunks, p.plan.chunks.len(), timer);
    }
    let outcomes = crate::shared::prefill_packing::member_outcomes(group.len(), result);
    group.iter_mut().zip(outcomes).map(|(p, outcome)| {
        outcome?;
        for &(_, point) in p.plan.points.iter().filter(|&&(chunk, _)| chunk == p.chunks && !probe::cold(&p.job.probe)) {
            cache.capture_session(p.ticket.session());
            if let Err(error) = cache.capture_media(family, SnapshotKind::Prompt, &p.keys.tokens()[..point], p.keys.spans(),
                &p.placement, After::default()) {
                tracing::warn!("snapshot point {point} not retained: {error:#}");
            }
        }
        p.chunks += 1;
        Ok(if p.done == p.tokens.len() { Chunk::Done } else { Chunk::More })
    }).collect()
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

fn draft_homes(layers: impl IntoIterator<Item = (bool, bool, usize)>,
    exl3: Option<&cuteafd_loader::V41Exl3Manifest>,
    read_bytes: impl Fn(usize) -> Result<usize>) -> Result<Vec<crate::shared::draft::resource::LayerHome>> {
    use crate::shared::draft::resource::{self, LayerHome};
    layers.into_iter().enumerate().map(|(layer, (dense, local, tp))| {
        if dense { return Ok(LayerHome::Dense); }
        ensure!(tp > 0, "GLM layer {layer} has no installed expert ranks");
        // EXL3 workers retain manifest pricing; every other backend attests its
        // actual installed weight/scale extent, including native EXLR8.
        let slice_bytes = match exl3 {
            Some(manifest) => resource::exl3_slice_bytes(manifest, layer, tp)?,
            None => read_bytes(layer)? as f64,
        };
        Ok(if local { LayerHome::Local { slice_bytes } } else { LayerHome::Remote { slice_bytes } })
    }).collect()
}

#[allow(clippy::too_many_arguments)]
fn schedule(engine: &GlmEngine<'_>, opened: &Opened, receive: &mut mpsc::Receiver<NativeRequest>,
    mut transport: Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime, stats: &Mutex<serde_json::Value>,
    max_sequences: usize, policy: Policy, ranks: usize, prefix: &PrefixArgs, select: SelectPlacement,
    media: &mut MediaAdmission<ml4_media::Prompt, ml4_media::Encoder>,
    expander: Option<&cuteafd_loader::media::MistralSpanExpander>, ready: impl FnOnce()) -> Result<()> {
    let (family, mut cache) = prefix_cache(engine, prefix, max_sequences)?;
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
            transport.as_ref().context("GLM installed Spark transport missing")?.installed_grouping()?
        } else { Vec::new() };
        let catalog = cuteafd_loader::read_expert_catalog(&opened.snapshot)?;
        let homes = draft_homes(engine.weights.layers.iter().enumerate().map(|(layer, weights)| {
            let local = tp2.as_ref().is_some_and(|layers| layers.contains(&layer));
            (weights.dense, local, if local { 2 } else { remote.len() })
        }), catalog.exl3(), |layer| engine.draft_weight_read_bytes(layer, &remote))?;
        let geometry = resource::geometry_with_grouping(&homes, engine.cfg.experts, engine.cfg.topk, DECODE_ROWS,
            d.cfg.drafts(), resource::COPY_EXTENSION, |layer, rows| engine.draft_group_rows(layer, rows, &remote))?;
        engine.arm_draft_probe()?;
        let prior = if d.token_only() { resource::Prior::History }
            else { resource::Prior::Selector(d.selector_fit("glm5", false)?) };
        SharedPolicy::with_geometry(geometry, prior, draft_seed())
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
    // A request waits here for KV (and before that, with images, in `media` for its features).
    let mut kv_waiter = cuteafd_engine::prefix::DeferredAdmission::<Admit>::default();
    // GLM's only decode lane (`LaneSupport::One`), polled on this thread.
    let lanes = LaneSet::default();
    // Prefix capacity and all fallible scheduler initialization precede readiness.
    ready();
    loop {
        engine.require_live()?;
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("expert wire unavailable until restart: {reason}");
        }
        while active.len() + prefills.len() < max_sequences {
            let busy = !active.is_empty() || !prefills.is_empty();
            let admit = match kv_waiter.poll(cache.pool().admission_free(), cache.pool().release_epoch(), busy,
                |admit| admit.prompt.job.events.is_closed()) {
                cuteafd_engine::prefix::AdmissionPoll::Blocked => break,
                cuteafd_engine::prefix::AdmissionPoll::Ready(admit) => admit,
                cuteafd_engine::prefix::AdmissionPoll::Empty => match media.poll(|prompt| prompt.job.events.is_closed()) {
                    MediaPoll::Ready(ready) => {
                        let (prompt, request_media) = ready.into_parts();
                        Admit { prompt, media: request_media, constraint: None }
                    }
                    MediaPoll::Failed(prompt, error) => {
                        let _ = prompt.job.events.send(Err(ml4_media::failure(error)));
                        continue;
                    }
                    MediaPoll::Pending | MediaPoll::Empty => {
                        let job = if !busy && media.is_empty() {
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
                        };
                        let reject = |job: &NativeRequest, message: String| {
                            let _ = job.events.send(Err(NativeFailure::BadRequest(message)));
                        };
                        if let Err(error) = probe::validate_scoring(&job.probe, engine.full_prefill_logits) {
                            reject(&job, format!("scoring: {error:#}"));
                            continue;
                        }
                        if !job.media.is_empty() && probe::scoring(&job.probe).is_some() {
                            reject(&job, "teacher-forced scoring takes text prompts".into());
                            continue;
                        }
                        if !job.media.is_empty() && !media.encoder().available() {
                            let _ = job.events.send(Err(NativeFailure::Unavailable("vision encoder unavailable".into())));
                            continue;
                        }
                        let tokens = match probe::prompt_ids(&job.probe, || Ok(job.encode_prompt(&tokenizer)?)) {
                            Ok(tokens) => tokens,
                            Err(error) => { reject(&job, format!("{error:#}")); continue; }
                        };
                        let events = job.events.clone();
                        let (prompt, request_media, jobs) = match ml4_media::prepare(job, tokens, expander,
                            engine.cfg.vocab_size, engine.cfg.hidden, engine.max_context) {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                let _ = events.send(Err(NativeFailure::BadRequest(format!("{error:#}"))));
                                continue;
                            }
                        };
                        if jobs.is_empty() {
                            Admit { prompt, media: request_media, constraint: None }
                        } else {
                            // Encode first; the prefix lookup this waiter peeks only orders its encodes.
                            let resume = cache.peek_media(prompt.keys.tokens(), prompt.keys.spans(), true);
                            let waiter = match MediaWaiter::new(prompt, request_media, jobs, resume) {
                                Ok(waiter) => waiter,
                                Err(error) => { let _ = events.send(Err(ml4_media::failure(error))); continue; }
                            };
                            if let Err((_, error)) = media.enqueue(waiter) {
                                let _ = events.send(Err(ml4_media::failure(error)));
                            }
                            continue;
                        }
                    }
                },
            };
            let Admit { prompt: ml4_media::Prompt { job, tokens, keys }, media: request_media, constraint } = admit;
            let reject = |job: &NativeRequest, message: String| {
                let _ = job.events.send(Err(NativeFailure::BadRequest(message)));
            };
            let constraint = match constraint {
                Some(constraint) => Some(constraint),
                None => match job.constraint.as_ref().map(|spec| grammars.matcher(spec)).transpose() {
                    Ok(constraint) => constraint,
                    Err(error) => {
                        reject(&job, format!("{error:#}"));
                        continue;
                    }
                },
            };
            let cold = probe::cold(&job.probe);
            if tokens.is_empty() || tokens.len() >= engine.max_context {
                reject(&job, format!("prompt of {} tokens is outside 1..{}", tokens.len(), engine.max_context));
                continue;
            }
            let capacity = (tokens.len() + job.max_tokens).min(engine.max_context);
            let slot = free_slots.pop();
            cache.tick();
            let admit_started = Instant::now();
            // Lookup and fork of the retained pages (byte-exact: the pages are the whole state);
            // image rows match only the same image (keyed rows, verified identities).
            let (lookup, spans): (&[u32], &[cuteafd_loader::media::MediaSpan]) =
                if cold { (&[], &[]) } else { (keys.tokens(), keys.spans()) };
            let admitted = match cache.admit_media(&family, lookup, spans, capacity, true,
                |pages| GlmPlacement { pages, len: 0, draft_slot: slot }) {
                Ok(admitted) => admitted,
                Err(error) => {
                    free_slots.extend(slot);
                    // Running requests keep their pages pinned. Delay a request
                    // that fits alone instead of rejecting transient KV pressure.
                    let admit = Admit { prompt: ml4_media::Prompt { job, tokens, keys }, media: request_media, constraint };
                    match kv_waiter.defer(admit, &error, busy, cache.pool().release_epoch()) {
                        Ok(()) => break,
                        Err(admit) => reject(&admit.prompt.job, format!("{error:#}")),
                    }
                    continue;
                }
            };
            admitted_total += 1;
            let resume = admitted.resume;
            job.prefix_cache_source.set(resume, admitted.source.as_ref().is_some_and(|source| source.host));
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
                            Ok(engine.prefill_device(placement, chunk, experts, 1)?.map(probe::ScoreLogits::Device))
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
                cuteafd_engine::prefix::plan_media_points(resume, tokens.len(), engine.prefill_capacity(),
                    &crate::shared::prefix::boundaries(&tokens, &markers), family.capture_reach(),
                    prefix.prefix_cache_min_tokens, prefix.points(), keys.spans())
            } else {
                cuteafd_engine::prefix::plan_points(resume, tokens.len(), engine.prefill_capacity(), &[], 0, 0,
                    PointPolicy { gap: 0, boundaries: 0, per_request: 0 })
            };
            // A whole-prompt hit brings its first token's logits: nothing to prefill.
            let logits = admitted.after.and_then(|after| after.logits).map(|logits| logits.to_vec());
            let plan = if logits.is_some() { PointPlan::default() } else { plan };
            prefills.push(Prefill { job, constraint, tokens, keys, media: request_media, done: resume, resume, plan, chunks: 0, cancelled: false,
                placement: admitted.placement, capacity, slot, logits, first: None, started: Instant::now(), busy: 0.0,
                id: admitted_total, ticket });
        }
        if prefills.due(!active.is_empty()) {
            // One chunk (a lane wave) of each waiting prompt (whole prompts with --decode-share 0).
            let limits = engine.packed_limits();
            let finished = prefills.round_packed(limits.rows, DECODE_ROWS,
                |p| packed_chunk(p).map(|(_, rows)| rows),
                |group, next| crate::shared::prefill_packing::plan(&group.iter().chain([next])
                    .filter_map(packed_chunk).collect::<Vec<_>>(), &limits).is_ok(),
                |group| {
                    if group.len() > 1 && group.iter().all(|p| packed_chunk(p).is_some()) {
                        return prefill_group(engine, &family, &mut cache, &mut selector, group);
                    }
                    group.iter_mut().map(|p| {
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
                        let logits = engine.prefill_media_device(&mut p.placement, chunk,
                            transport.as_deref_mut().map(|t| (t, runtime)), usize::from(end == p.tokens.len()),
                            Some(&p.media))?;
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
                            if let Err(error) = cache.capture_media(&family, SnapshotKind::Prompt, &p.keys.tokens()[..point],
                                p.keys.spans(), &p.placement, After::default()) {
                                tracing::warn!("snapshot point {point} not retained: {error:#}");
                            }
                        }
                        p.chunks += 1;
                        Ok(if p.done == p.tokens.len() { Chunk::Done } else { Chunk::More })
                    }).collect()
                });
            for (mut p, prefilled) in finished {
                let (slot, placement, resume) = (p.slot, p.placement.clone(), p.resume);
                if p.cancelled || p.job.events.is_closed() {
                    p.ticket.cancel();
                    // A skipped final head is still successful computed KV.
                    // Park it before either the error or missing-logits path.
                    if placement.len > resume && !probe::cold(&p.job.probe) {
                        cache.capture_session(p.ticket.session());
                        if let Err(error) = cache.park_media(&family, &p.keys.tokens()[..placement.len], p.keys.spans(),
                            &placement) {
                            tracing::warn!("parking a cancelled prefill: {error:#}");
                        }
                    }
                    release(&family, &mut cache, &mut free_slots, &placement, slot);
                    continue;
                }
                if let Err(error) = &prefilled {
                    tracing::warn!("prefill failed: {error:#}");
                    let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
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
                let prompt = (resume < p.tokens.len()).then(|| p.keys.tokens().to_vec());
                let spans = p.keys.spans().to_vec();
                let session = p.ticket.session();
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &GlmPlacement| {
                    if let (Some(prompt), Some(logits)) = (&prompt, &logits) {
                        cache.capture_session(session.clone());
                        if let Err(error) = cache.capture_media(&family, SnapshotKind::Prompt, prompt, &spans, placement,
                            After::from_logits(logits, true)) {
                            tracing::warn!("prompt snapshot not retained: {error:#}");
                        }
                    }
                };
                let job_events = p.job.events.clone();
                let admitted = (|| -> Result<Active<'_>> {
                    let mut request = Active {
                        spans: spans.clone(),
                        id: p.id,
                        prompt_tokens: p.tokens.len(),
                        history: p.tokens,
                        slot,
                        drafts: DraftHistory::default(),
                        counts: [0; 3],
                        decoder: cuteafd_loader::streaming_token_decoder(&opened.snapshot, false)?,
                        job: p.job, constraint: p.constraint, placement: p.placement,
                        // An MLA drafter restored warm from its prefix mark holds history from 0.
                        draft_from: match (engine.drafter.as_ref(), slot) {
                            (Some(drafter), Some(slot)) if drafter.mla_windows_only() =>
                                drafter.mla_frontier(slot).map_or(resume, |(origin, _)| origin),
                            _ => resume,
                        }, turn: None,
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
        let Some(finished) = runtime.block_on(lanes.run_one(|lease| decode_round(lease, engine, opened, policy.fixed,
            &mut transport, runtime, &mut phases, &mut active, &mut shared, &mut selector, &family, &mut cache,
            &mut free_slots, &mut trace, cycle)))? else { continue };
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
                // Keyed like its prompt: image rows by their image identity.
                let keys = MediaKeys::new(&request.history, engine.cfg.vocab_size as u32, &request.spans)?;
                let rows = &keys.tokens()[..request.placement.len];
                cache.capture_session(request.ticket.session());
                if let Err(error) = cache.capture_media(&family, SnapshotKind::Turn, rows, &request.spans, &request.placement,
                    After::from_logits(row, true)) {
                    tracing::warn!("turn snapshot not retained: {error:#}");
                }
            }
            release(&family, &mut cache, &mut free_slots, &request.placement, request.slot);
        }
        cache.tick();
        publish(stats, requests, generated_total, active.len(), prefills.len(), &cache);
        console::gauges(|| console::Gauges::prefix_cache(&cache, active.len(), prefills.len(),
            receive.len() + kv_waiter.len() + media.len()));
        // Encodes in flight: wake for their completion without spinning.
        if active.is_empty() && prefills.is_empty() && !media.is_empty() { std::thread::park_timeout(Duration::from_millis(1)); }
        prefills.stepped(cycle.elapsed().as_secs_f64());
    }
}

/// The lane's commit. GLM keeps no recurrent target state (a verify's
/// rejected rows just set the sequence length back during emission), so the
/// commit is the DFlash2 drafter's context update for the committed rows,
/// enqueued on the drafter's stream and ordered before its next draft.
struct DraftContext<'d, 'a> {
    drafter: Option<&'d crate::shared::speculation::dflash::DflashDrafter<'a>>,
    context: Vec<ContextRow>,
}

impl LaneCommit for DraftContext<'_, '_> {
    fn begin(&mut self) -> Result<()> {
        self.drafter.map_or(Ok(()), |drafter| drafter.update(&self.context))
    }
    fn ready(&mut self) -> Result<bool> { Ok(true) }
    fn publish(&mut self) -> Result<()> { Ok(()) }
    fn abort_and_drain(&mut self) -> Result<()> { Ok(()) }
}

/// One round of lane 0 (GLM's only decode lane, `LaneSupport::One`): draft,
/// plan, verify on the lane's decode link, select, emit and commit. `None`
/// when the step failed and its requests were released.
#[allow(clippy::too_many_arguments)]
async fn decode_round(lease: LaneLease<'_>, engine: &GlmEngine<'_>, opened: &Opened, fixed: Option<usize>,
    transport: &mut Option<&mut SparkLink<'_>>, runtime: &tokio::runtime::Runtime, phases: &mut [f64; 8],
    active: &mut Vec<Active<'_>>, shared: &mut Option<crate::shared::draft::runtime::SharedPolicy>,
    selector: &mut TokenSelector<'_>, family: &GlmPrefix<'_, '_>, cache: &mut PrefixCache<CudaCopyEngine<'_>>,
    free_slots: &mut Vec<usize>, trace: &mut Option<Trace>, cycle: Instant) -> Result<Option<Vec<bool>>> {
    use crate::shared::draft::resource;
    let drafter = engine.drafter.as_ref();
    let mut round_clock = crate::shared::draft::clock::RoundClock::at(cycle);
    let mut drafted_width = 0;
    let mut drafted_requests = 0;
    let mut tally = console::Step::begin(lease.index());
    let (phases_before, engine_before) = (*phases, tally.live().then(|| *engine.profile.borrow()));
    // Rows each sequence may add after its next token.
    let room = DECODE_ROWS.saturating_sub(active.len());
    let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else {
        room.min(a.job.max_tokens - a.generated - 1).min(a.capacity - a.placement.len - 1) }).collect();
    // Skip/block decisions use the same resource prices as post-draft lengths.
    let eligible: Vec<_> = active.iter().zip(&limits).filter(|(a, limit)| a.slot.is_some() && **limit > 0)
        .map(|(a, &limit)| (a.ticket.id(), limit)).take(drafter.map_or(0, |d| d.max_batch_sequences())).collect();
    let run_draft = if eligible.is_empty() { false } else if fixed.is_some()
        || active.iter().any(|a| a.drafts.cold() && eligible.iter().any(|&(id, _)| id == a.ticket.id())) { true } else {
        let shared = shared.as_mut().expect("eligible draft slots have a shared policy");
        let width = drafter.unwrap().cfg.drafts();
        // Undrafted members still contribute their anchors, routes and emitted tokens.
        let requests: Vec<_> = active.iter().map(|a| (a.ticket.id(),
            eligible.iter().find(|&&(id, _)| id == a.ticket.id()).map_or(0, |&(_, room)| room))).collect();
        resource::choose_block(&mut shared.policy, &requests, width, eligible.len(), DECODE_ROWS, false)?
    };
    let timer = Instant::now();
    let drafted: Vec<Option<crate::shared::speculation::dflash::Draft>> = match drafter {
        Some(drafter) if run_draft => {
            let seqs: Vec<(usize, DraftSeq)> = active.iter().enumerate()
                .filter_map(|(i, a)| a.slot.filter(|_| limits[i] > 0).map(|slot| (i, DraftSeq { slot, anchor: a.next, position: a.placement.len,
                    valid_from: a.draft_from })))
                .take(drafter.max_batch_sequences())
                .collect();
            drafted_requests = seqs.len();
            let drafts = round_clock.draft(|| drafter.draft_device(&seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), &engine.embedding,
                engine.draft_target_head()?));
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
    let token_only = engine.drafter.as_ref().is_some_and(|d| d.token_only());
    let mut priors: Vec<_> = active.iter().enumerate().zip(&drafted).map(|((i, a), draft)| match (&shared, draft) {
        // A token-only drafter (mtp_dspark) is judged by the pooled request history.
        (Some(shared), Some(d)) if token_only => {
            let pooled = a.drafts.pooled(d.tokens.len(), active.iter().enumerate()
                .filter(|&(other, _)| other != i).map(|(_, other)| &other.drafts));
            shared.prior(&a.drafts, &resource::DraftEvidence::History(&pooled), 0)
        }
        (Some(shared), Some(d)) => shared.prior(&a.drafts,
            &resource::DraftEvidence::Selector(&d.features), 0),
        _ => Vec::new(),
    }).collect();
    let copies: Vec<_> = active.iter().zip(&drafted).zip(&limits).zip(&mut priors)
        .map(|(((a, draft), &limit), prior)| match (&shared, draft) {
            (Some(shared), Some(d)) if fixed.is_none() => resource::copy_continuation(&shared.policy, active.len(),
                a.ticket.id(), &a.history, &d.tokens, prior, limit, a.job.sampling.is_greedy()),
            _ => None,
        }).collect();
    let mut predicted = None;
    let mut planned = match (fixed, shared.as_mut()) {
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
                release(family, cache, free_slots, &request.placement, request.slot);
            }
            return Ok(None);
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
    let trace_ref = &mut *trace;
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
    lanes::commit(&mut DraftContext { drafter, context }).await?;
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
    Ok(Some(finished))
}

#[cfg(test)]
mod media_tests {
    /// Text prompts key and admit exactly as before: no spans, the keyed copy is the tokens, and
    /// the prefix calls are the media forms the text forms already delegate to.
    #[test]
    fn text_prompts_take_the_text_prefix_path() {
        let tokens = vec![3u32, 1, 4, 1, 5];
        let keys = cuteafd_engine::media::MediaKeys::new(&tokens, 10, &[]).unwrap();
        assert_eq!(keys.tokens(), tokens.as_slice());
        let policy = cuteafd_engine::prefix::PointPolicy { gap: 4, boundaries: 1, per_request: 2 };
        assert_eq!(cuteafd_engine::prefix::plan_media_points(0, 5, 2, &[2], 1, 1, policy, &[]),
            cuteafd_engine::prefix::plan_points(0, 5, 2, &[2], 1, 1, policy));
        let engine = include_str!("../../../../cuteafd-engine/src/prefix/cache.rs");
        assert!(engine.contains("self.admit_native(family, tokens, tokens, &[], capacity, sampled, build)"));
        assert!(engine.contains("self.capture_media(family, kind, tokens, &[], placement, after)"));
    }
}

#[cfg(test)]
mod lane_tests {
    #[test]
    fn final_chunk_cancellation_parks_before_error_or_missing_head() {
        let source = include_str!("serve.rs");
        let finish = source.split_once(concat!("for (mut p, pre", "filled) in finished {")).unwrap().1
            .split_once("let logits = p.logits.clone();").unwrap().0;
        let cancel = finish.find("if p.cancelled || p.job.events.is_closed()").unwrap();
        let park = finish.find("cache.park_media(").unwrap();
        let error = finish.find("if let Err(error) = &prefilled").unwrap();
        let missing_head = finish.find("prefill produced no logits").unwrap();
        assert!(cancel < park && park < error && error < missing_head);
        let cancelled = &finish[cancel..error];
        assert!(cancelled.contains("p.ticket.cancel()"));
        assert!(cancelled.contains("placement.len > resume"));
        assert!(cancelled.contains("release(&family") && cancelled.contains("continue;"));
        assert!(!cancelled.contains("prefilled.is_err()"));
    }

    #[test]
    fn decode_runs_as_lane_zero_of_a_shared_lane_set() {
        let source = include_str!("serve.rs");
        let schedule = source.split_once("\nfn schedule(").unwrap().1
            .split_once("\n/// The lane's commit.").unwrap().0;
        // One lane (`LaneSupport::One`): every decode round, and only it, runs on the set.
        assert_eq!(schedule.matches("lanes.run_one(|lease| decode_round(lease,").count(), 1);
        assert!(!schedule.contains("lanes.run(") && !schedule.contains("lanes.join("));
        assert!(!schedule.contains("engine.verify_device(&mut rows,"), "no private decode round remains");
        let round = source.split_once("\nasync fn decode_round").unwrap().1
            .split_once("\n#[cfg(test)]").unwrap().0;
        assert!(round.contains("console::Step::begin(lease.index())"));
        assert!(round.contains("lanes::commit(&mut DraftContext { drafter, context }).await?;"));
        assert!(!round.contains("drafter.update("), "the drafter commit goes through the lane commit");
    }
}

#[cfg(test)]
mod draft_tests {
    use super::*;
    use crate::shared::draft::{resource, runtime::SharedPolicy};
    use cuteafd_transport::worker_selection::{ExpertGroupingSelector, ExpertTrafficContract,
        InstalledExpertGrouping, InstalledExpertVariant, WorkerIdentity, WorkerSelection};

    fn installed_reports(bytes: Option<[usize; 2]>, conservative: bool) -> Vec<InstalledExpertGrouping> {
        let identity: WorkerIdentity = serde_json::from_value(serde_json::json!({
            "checkpoint": "0".repeat(64), "geometry": [6144, 256, 8, 2048, 78],
            "world": 2, "topology": [2, 1], "ownership_wire": false,
        })).unwrap();
        let loaded = WorkerSelection::new(identity, &(3..78).collect::<Vec<_>>()).unwrap();
        (0..2).map(|rank| {
            let contract = if conservative { ExpertTrafficContract::ConservativeRouteUpperBound }
                else { ExpertTrafficContract::ExactGrouping };
            let variant = InstalledExpertVariant::from_piecewise_queries(256, contract,
                |_| Ok(if conservative { 1 } else { 48 })).unwrap();
            let report = InstalledExpertGrouping::new(loaded.clone(), rank, 128,
                ExpertGroupingSelector::Fp8Nearest, vec![variant]).unwrap();
            bytes.map_or(report.clone(), |bytes| report.with_weight_read_bytes(bytes[rank]).unwrap())
        }).collect()
    }

    #[test]
    fn exlr8_dflash2_scheduler_constructs_from_installed_native_read_bytes() {
        let reports = installed_reports(Some([6_144_024, 9_216_024]), true);
        let homes = draft_homes((0..78).map(|layer| (layer < 3, false, reports.len())), None,
            |layer| resource::worker_weight_read_bytes(&reports, layer)).unwrap();
        let geometry = resource::geometry_with_grouping(&homes, 256, 8, DECODE_ROWS, 7,
            resource::COPY_EXTENSION, |layer, rows| resource::worker_group_rows(&reports, layer, rows)).unwrap();
        let scheduler = SharedPolicy::with_geometry(geometry,
            resource::Prior::Selector(crate::shared::draft_confidence::SelectorFit::default()), draft_seed()).unwrap();
        assert_eq!(scheduler.policy.geometry().widths, [7]);
        assert_eq!(scheduler.policy.geometry().layers.len(), 78);
        assert!(scheduler.policy.geometry().layers[..3].iter().all(|layer| layer.class.is_none()));
        for layer in &scheduler.policy.geometry().layers[3..] {
            assert_eq!(layer.slice_bytes.to_bits(), 9_216_024f64.to_bits());
            assert_eq!(layer.group_rows, 1);
        }
    }

    #[test]
    fn installed_fp8_mxfp4_nvfp4_prices_remain_bit_identical() {
        for (format, bytes) in [("fp8", [12_288_024, 9_216_024]),
            ("mxfp4", [6_291_456, 4_194_304]), ("nvfp4", [7_077_900, 4_718_600])] {
            let reports = installed_reports(Some(bytes), false);
            let homes = draft_homes((0..78).map(|layer| (layer < 3, layer == 3, 2)), None,
                |layer| resource::worker_weight_read_bytes(&reports, layer)).unwrap();
            let previous = (0..78).map(|layer| if layer < 3 { resource::LayerHome::Dense }
                else if layer == 3 { resource::LayerHome::Local { slice_bytes: bytes[0] as f64 } }
                else { resource::LayerHome::Remote { slice_bytes: bytes[0] as f64 } }).collect::<Vec<_>>();
            assert_eq!(homes, previous, "{format}");
            let geometry = |homes: &[_]| resource::geometry_with_grouping(homes, 256, 8, DECODE_ROWS, 7,
                resource::COPY_EXTENSION, |layer, rows| resource::worker_group_rows(&reports, layer, rows)).unwrap();
            assert_eq!(geometry(&homes), geometry(&previous), "{format}");
        }
    }

    #[test]
    fn missing_installed_read_bytes_is_typed_without_format_fallback() {
        let reports = installed_reports(None, true);
        let error = draft_homes((0..78).map(|layer| (layer < 3, false, 2)), None,
            |layer| resource::worker_weight_read_bytes(&reports, layer)).unwrap_err();
        assert_eq!(error.downcast_ref::<cuteafd_transport::worker_selection::MissingWeightReadBytes>(),
            Some(&cuteafd_transport::worker_selection::MissingWeightReadBytes { rank: 0 }));
        let error = resource::worker_weight_read_bytes(&[], 3).unwrap_err();
        assert_eq!(error.downcast_ref::<resource::MissingWorkerReadBytes>(),
            Some(&resource::MissingWorkerReadBytes { layer: 3 }));
    }

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

#[cfg(test)]
mod chat_encoding_tests {
    /// The served checkpoint picks its own chat contract (Mistral Large 4 → Mistral
    /// instruct; GLM 5.3 → GLM), whichever engine variant runs it.
    #[test]
    fn served_checkpoints_select_their_own_chat_encoding_when_present() {
        let hub = std::path::Path::new("/mnt/sparknest/hf-home/hub");
        for (snapshot, mistral) in [
            ("models--mistralai--Mistral-Large-4-1T-A52B/snapshots/f12dabdc0c1e99e7f7924461a6f308859b05af4d", true),
            ("models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1/snapshots/47af23347db743b4666d952e2eb48f2b01c3fede", false),
        ] {
            let path = hub.join(snapshot);
            if !path.exists() { continue; }
            let encoding = super::chat_encoding(&path, Default::default()).unwrap();
            assert_eq!(matches!(encoding, super::ModelEncoding::Mistral(_)), mistral, "{snapshot}");
            assert_eq!(matches!(encoding, super::ModelEncoding::Glm(_)), !mistral, "{snapshot}");
        }
    }
}
