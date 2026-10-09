//! OpenAI-compatible API over the GLM 5.3 Flash engine: continuous batching
//! with one prefill per admitted request and one decode-shaped step for
//! every active sequence. With a DFlash2 drafter (--draft), every step first
//! drafts after each sequence's next token on the GPU and verifies as many
//! drafts as the adaptive policy (glm/dflash_policy.rs, priced by the
//! measured Spark TP2 step cost) finds worthwhile; copy-window drafts extend
//! a DFlash2 draft they agree with (or stand alone without a drafter).
//!
//! KDA layers advance recurrent state that a rejected draft cannot simply
//! drop, so a step that verifies drafts runs speculatively
//! (`GlmfEngine::verify_spec`: the KDA state stays, each row's replay inputs
//! are recorded) and then commits every sequence's kept rows
//! (`GlmfEngine::commit`), which leaves the state serial steps would have.
//! MLA records past the kept length are rewritten by later steps.
//!
//! Prefix cache (`cuteafd_engine::prefix` over `super::prefix::GlmfPrefix`):
//! admission restores the deepest retained snapshot whose tokens prefix the
//! prompt (shared units, the copied tail unit, the KDA state mark) and
//! prefills only the rest; a whole-prompt hit takes its first token from the
//! retained logits. The prompt is retained at prompt end (unless it was a
//! whole hit), the conversation at a normal finish (`Turn`; its kept rows are
//! committed to the KDA state first), a prefill whose client left is parked
//! at its last chunk; a decode whose client left is not retained.
//! `prompt_cache_hit_tokens` reports the restored rows; `/v1/stats` carries
//! the cache's counters.
use super::engine::{GlmfEngine, GlmfPlacement, DECODE_ROWS, WIDE_DECODE_ROWS};
use super::prefix::{GlmfPrefix, PrefixMarks};
use cuteafd_engine::media::{EmbeddingCache, MediaAdmission, MediaPoll, MediaReady, MediaWaiter, RequestMedia, MediaKeys};
use crate::families::deepseek_v41::v41_native_serve::prefix::CudaCopyEngine;
use crate::shared::prefix::{PrefixArgs, Toggle};
use crate::shared::probe;
use crate::shared::console;
use cuteafd_engine::prefix::{After, PointPlan, PointPolicy, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
use crate::families::glm5::dflash::{ContextRow, Draft, DraftSeq, TAP_ROWS};
use crate::families::glm5::dflash_policy::{self, DraftHistory, Shape};
use super::{open, Opened};
use crate::shared::token_io::{RowResult, SelectBatch, SelectPlacement, TokenSelector};
use crate::shared::prefill_share::{add_phases, isolated_phases, Chunk, DecodeShareArgs};
use anyhow::{Context, Result};
use cuteafd_api::openai::chat::glm5::{GlmEncoding, GlmThinkingOff};
use cuteafd_api::openai::{
    InferenceChunk, InferenceFinishReason, ModelEncoding, ModelProfile, NativeFailure, NativeLimits, NativeRequest,
    PromptUsage,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Most copy-window draft tokens verified per sequence and step.
const COPY_DRAFT: usize = 7;

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: super::EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Sequences decoding at once (each holds two KDA state slots).
    #[arg(long, default_value_t = 4)]
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
    /// Resolved global vision policy, assigned before dispatch.
    #[arg(skip = cuteafd_loader::plan::MediaMode::Auto)]
    pub vision: cuteafd_loader::plan::MediaMode,
    /// Explicit vendor template source for vision: cached HF id or snapshot directory.
    #[arg(long)]
    pub chat_template_from: Option<String>,
    /// Host embedding-cache quota; default min(8 GiB, 5% RAM).
    #[arg(long, value_parser = crate::shared::prefix::parse_bytes)]
    pub media_cache_bytes: Option<u64>,
    /// Admitted resident Spark encoder endpoints, in replica order.
    #[arg(long)]
    pub vision_peers: Option<String>,
    /// Shared planner admission hash (required with --vision-peers).
    #[arg(long, requires = "vision_peers")]
    pub encoder_plan_hash: Option<String>,
    /// Snapshot revision used in the encoder handshake.
    #[arg(long, requires = "vision_peers")]
    pub encoder_revision: Option<String>,
    #[command(flatten)]
    pub console: console::ConsoleArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
}

/// Speculation settings: copy-window draft cap (0 disables) and a fixed
/// DFlash2 draft count replacing the adaptive policy.
#[derive(Debug, Clone, Copy)]
struct Policy {
    copy: usize,
    fixed: Option<usize>,
}

pub(crate) fn model_id(snapshot: &std::path::Path) -> Option<String> {
    snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    })
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    let api = args.api.load()?;
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let encoding = if super::media::vision_config(args.vision, &snapshot)?.is_none() {
        GlmEncoding::from_snapshot(&snapshot)?
    } else {
        GlmEncoding::from_snapshot_for_vision(&snapshot, args.chat_template_from.as_deref(), None)?
    }.with_thinking_off(args.thinking_off);
    if let Some(provenance) = encoding.template_provenance() {
        tracing::info!(chat_template = %serde_json::to_string(provenance)?, "GLM Flash chat template selected");
    }
    let mut profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&snapshot)).context("model id")?,
        ModelEncoding::Glm(Arc::new(encoding)),
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    engine_args.slots = engine_args.slots.max(args.max_sequences);
    (engine_args.draft_context_slots, engine_args.draft_sequences) = draft_capacity(args.max_sequences,
        engine_args.draft_sequences, engine_args.draft_context_slots);
    let (worker_stats, max_sequences, decode_share) = (stats.clone(), args.max_sequences, args.decode_share);
    let policy = Policy { copy: if args.no_copy_drafts { 0 } else { COPY_DRAFT }, fixed: args.draft_fixed };
    let prefix = args.prefix.clone();
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id)));
    let vision = args.vision;
    let remote = super::media::RemoteVision::from_args(&args)?;
    let media_cache_bytes = args.media_cache_bytes;
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, policy, decode_share, prefix, vision, media_cache_bytes, remote));
    let (max_context, media) = ready_rx.await.context("engine failed before it was ready")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    if let Some((preparer, health)) = media {
        profile = profile.with_loaded_vision(preparer);
        profile.vision_health = health;
    }
    cuteafd_bench::context::phase("engine loaded");
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, Duration::from_secs(25),
        hub.clone(), crate::shared::api::profile(profile.clone()));
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    tracing::info!(listen = %args.listen, model = %profile.id, "GLM 5.3 Flash API is ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<std::net::SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
    }
    Ok(())
}

/// What the live console shows for GLM 5.3 Flash.
fn console_layout(args: &ServeArgs, model: &str) -> console::Layout {
    use console::{Color::*, StepGroup};
    let mut layout = console::Layout::new("glm5_flash", model.into(), args.engine.snapshot.clone());
    let sparks = args.engine.peers.as_deref().map_or(0, |peers| peers.split(',').count());
    layout.hardware = console::hardware(1 + usize::from(args.engine.split_device.is_some()), sparks,
        args.engine.local_experts);
    layout.split = args.engine.split_device.map(|_| "head split".into());
    layout.concurrency = args.max_sequences.min(DECODE_ROWS);
    let copy = if args.no_copy_drafts { "" } else { " · copy windows" };
    let policy = args.draft_fixed.map_or_else(|| "adaptive".to_string(), |n| format!("fixed {n}"));
    let drafter = args.engine.draft.as_deref()
        .map(|snapshot| if super::dspark::is_dspark(snapshot) { "dSpark" } else { "DFlash2" });
    layout.speculator = match (drafter, args.no_copy_drafts) {
        (Some(name), _) => Some(console::Speculator { name: name.into(), positions: 8, policy: policy + copy }),
        (None, false) => Some(console::Speculator { name: "Copy window".into(), positions: COPY_DRAFT,
            policy: "adaptive length".into() }),
        (None, true) => None,
    };
    layout.steps = vec![
        StepGroup::new("Decode step", "host clock", &[("round.cycle", "step", Target),
            ("draft", "block draft", Accepted), ("plan", "draft plan + copy windows", Ink),
            ("verify", "verify pass + token selection", Target), ("emit", "accept + stream", Ink),
            ("commit", "KDA commit + drafter context", Accepted)]),
        StepGroup::new("Verify pass", "host clock, engine phases", &[
            ("gpu", "GPU until expert exchanges", Rtx), ("experts", "Spark expert exchanges", Spark),
            ("head", "head + logits", Target)]),
        StepGroup::layers(),
        StepGroup::admission(false),
    ];
    layout.layers = Some(console::Layers::host_clock());
    layout
}

type VisionReady = Option<(Arc<cuteafd_api::openai::media::MediaPreparer>, Option<Arc<std::sync::atomic::AtomicBool>>)>;

#[allow(clippy::too_many_arguments)]
fn serve_loop(mut args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<(usize, VisionReady)>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    policy: Policy, decode_share: DecodeShareArgs, prefix: PrefixArgs,
    vision: cuteafd_loader::plan::MediaMode, media_cache_bytes: Option<u64>, remote: Option<super::media::RemoteVision>) -> Result<()> {
    args.serving_graph_policy = Some((max_sequences.min(DECODE_ROWS), args.draft.is_some() || policy.copy > 0));
    let lanes = max_sequences.min(DECODE_ROWS);
    // Without entries no mark is taken: pool marks keep no unit back and need no room in the pool.
    let marks = args.prefix_marks.with_entries(prefix.prefix_cache_entries);
    if marks != args.prefix_marks {
        tracing::info!("GLM 5.3 Flash prefix cache off (no entries): no prefix marks, no reserved unit");
        args.prefix_marks = marks;
    }
    // Either KV admission reserves the mark arena `prefix_cache` will allocate (none: marks in pool units),
    // counted once the engine's layout is known (`with_engine`).
    args.mark_arena = match args.prefix_marks {
        PrefixMarks::Arena => super::prefix::ArenaMarks::Rule(prefix.mark_rule(lanes)),
        PrefixMarks::Pool => super::prefix::ArenaMarks::None,
    };
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let (vision, prefix) = match super::media::ReadyVision::load(&args, &opened.library, vision, &prefix, media_cache_bytes, remote) {
        Ok(vision) => vision,
        Err(error) => { let _ = ready.send(Err(error)); return Ok(()); }
    };
    let preparer = vision.as_ref().map(|vision| vision.preparer.clone());
    let (encoder, bytes) = vision.map_or((super::media::Encoder::Off, 0), |vision|
        (vision.encoder, vision.cache_bytes));
    let health = encoder.health_handle();
    let mut media = MediaAdmission::new(EmbeddingCache::new(bytes), encoder, 16);
    let mut ready = Some(ready);
    let result = opened.with_engine(&args, |engine| {
        anyhow::ensure!(engine.weights.layers.len() == engine.cfg.layers, "serve-glmf needs every layer");
        anyhow::ensure!(engine.experts().is_some(), "serve-glmf needs --peers (or --local-experts) for the routed experts");
        anyhow::ensure!(preparer.is_none() || media.encoder().available(), "vision encoder unavailable before readiness");
        engine.warm_decode_graphs(max_sequences.min(DECODE_ROWS), engine.drafter.is_some() || policy.copy > 0)?;
        let ranks = args.peers.as_deref().map(|peers| peers.split(',').count());
        let spark = matches!(engine.experts(), Some(super::engine::Experts::Spark { .. }));
        console::layer_classes(engine.weights.layers.iter().map(|l| console::layer_class(l.dense, spark)).collect());
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, lanes, policy, ranks, decode_share, &prefix,
            (args.prefix_marks, args.mark_arena), args.token_io.token_select, &mut media, preparer.as_deref(),
            || {
                // Ready once the prefix cache and the token selector exist: the ledger then lists
                // every start-up allocation.
                crate::shared::memory_report::log("glm5_flash ready");
                if let Some(ready) = ready.take() {
                    let _ = ready.send(Ok((engine.max_context, preparer.clone().map(|p| (p, health.clone())))));
                }
            })
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| (args.max_context, preparer.map(|p| (p, health)))).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

/// An admitted prompt waiting for its remaining prefill chunks.
struct Prefill<'a> {
    job: NativeRequest,
    constraint: Option<crate::shared::constraints::State<'a>>,
    tokens: Vec<u32>,
    keys: MediaKeys,
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
    placement: GlmfPlacement,
    capacity: usize,
    slot: Option<usize>,
    /// A whole-prompt prefix hit's retained logits (its first token is selected from them).
    logits: Option<Vec<f32>>,
    /// The first generated token, selected right after the last chunk, and
    /// that row's logits when a prompt snapshot will keep them.
    first: Option<u32>,
    prompt_row: Option<Vec<f32>>,
    started: Instant,
    /// Seconds in this prompt's chunks, and their engine phases.
    busy: f64,
    phases: [f64; 3],
    ticket: console::Ticket,
}

struct Active<'a> {
    job: NativeRequest,
    /// Prompt and generated tokens, for copy-window drafts.
    history: Vec<u32>,
    keys: MediaKeys,
    _media: RequestMedia,
    /// Current copy-draft length (halved after a fully rejected draft,
    /// doubled after a fully accepted one) and steps left before drafting
    /// resumes once it reached zero.
    draft_limit: usize,
    draft_pause: usize,
    /// DFlash2 ring slot and draft outcomes (None without a drafter slot).
    slot: Option<usize>,
    drafts: DraftHistory,
    /// Hash of `history` (identical sequences share it).
    digest: u64,
    /// Steps, DFlash2 drafts verified/accepted, copy drafts verified/accepted,
    /// and actual neural drafter forward calls for this sequence.
    counts: [usize; 6],
    constraint: Option<crate::shared::constraints::State<'a>>,
    placement: GlmfPlacement,
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
fn take(constraint: Option<&mut crate::shared::constraints::State<'_>>, selected: &RowResult) -> Result<u32> {
    let token = selected.as_ref().map_err(|e| anyhow::anyhow!("sampling: {e:?}"))?.token;
    if let Some(state) = constraint {
        state.accept(token)?;
    }
    Ok(token)
}

/// Selects (and commits) a token from host logits: a whole-prompt prefix hit's retained row.
fn select_host(constraint: Option<&mut crate::shared::constraints::State<'_>>,
    sampling: cuteafd_core::TargetSamplingParams, logits: &[f32], position: u64) -> Result<u32> {
    let mut constraint = constraint;
    let mask = match constraint.as_deref_mut() {
        Some(state) => state.mask()?.map(<[u32]>::to_vec),
        None => None,
    };
    let token = sampling.select_token(logits, mask.as_deref(), position)
        .map_err(|e| anyhow::anyhow!("sampling: {e:?}"))? as u32;
    if let Some(state) = constraint {
        state.accept(token)?;
    }
    Ok(token)
}

impl Active<'_> {
    fn send(&self, chunk: InferenceChunk) -> Result<()> {
        self.job.events.send(Ok(chunk)).map_err(|_| anyhow::anyhow!("client went away"))
    }

    /// Streams `token` (special tokens stay text for the GLM parser); returns
    /// true when the request is finished.
    fn emit(&mut self, token: u32) -> Result<bool> {
        probe::token(&self.job.probe, token);
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

const DIGEST_SEED: u64 = 0xcbf2_9ce4_8422_2325;

fn digest(state: u64, token: u32) -> u64 {
    (state ^ u64::from(token)).wrapping_mul(0x0100_0000_01b3)
}

fn media_digest(tokens: &[u32], spans: &[cuteafd_loader::media::MediaSpan]) -> u64 {
    let mut state = tokens.iter().fold(DIGEST_SEED, |d, &t| digest(d, t));
    // Radix hints truncate image identity; speculative grouping must see every key byte.
    for span in spans {
        for byte in span.start.to_le_bytes().into_iter().chain(span.len.to_le_bytes()).chain(*span.key.bytes()) {
            state = digest(state, u32::from(byte));
        }
    }
    state
}

fn cold_replay_plan(probe: &probe::ProbeRef, plan: PointPlan, retained_logits: bool) -> PointPlan {
    if let Some(probe) = probe.as_ref().filter(|p| !p.spec.cold_steps.is_empty()) {
        // Validation precedes admission; replay the source's exact prefill/decode boundaries.
        PointPlan { chunks: probe.spec.cold_steps.iter().map(|step| step.end).collect(), points: Vec::new() }
    } else if retained_logits { PointPlan::default() } else { plan }
}

fn cold_replay_decode(probe: &probe::ProbeRef, chunk: usize) -> bool {
    probe.as_ref().and_then(|p| p.spec.cold_steps.get(chunk)).is_some_and(|step| step.decode)
}

#[cfg(test)]
mod media_tests {
    #[test]
    fn verify_histogram_counts_actual_steps_and_existing_timing_boundary() {
        let mut stats = super::VerifyStats::default();
        assert_eq!(stats.snapshot()["rows"], serde_json::json!({}));
        stats.record(7, 8, true, 12.5);
        stats.record(8, 8, true, 10.0);
        stats.record(1, 1, false, 3.0);
        let json = stats.snapshot();
        assert_eq!(json["rows"], serde_json::json!({"1": 1, "7": 1, "8": 1}));
        assert_eq!(json["by_bucket"]["8"]["steps"], 2);
        assert_eq!(json["by_bucket"]["8"]["speculative_steps"], 2);
        assert_eq!(json["by_bucket"]["8"]["verify_ms_sum"], 22.5);
        assert_eq!(json["by_bucket"]["8"]["verify_ms_max"], 12.5);
        assert_eq!(json["by_real_rows"]["1"]["speculative_steps"], 0);
    }

    #[test]
    fn cold_replay_keeps_source_geometry_and_disables_snapshot_points() {
        use cuteafd_api::openai::probe::{Probe, ProbeColdStep, ProbeSpec};
        use cuteafd_engine::prefix::PointPlan;
        let spec = ProbeSpec { cold: true, no_speculation: true,
            cold_steps: vec![ProbeColdStep { end: 8, decode: false },
                ProbeColdStep { end: 9, decode: true }, ProbeColdStep { end: 12, decode: false }],
            ..Default::default() };
        spec.validate_cold_steps(12, 8, 2).unwrap();
        let probe = Some(Probe::new(spec));
        let ordinary = || PointPlan { chunks: vec![8, 12], points: vec![(0, 8)] };
        let plan = super::cold_replay_plan(&probe, ordinary(), false);
        assert_eq!(plan.chunks, [8, 9, 12]);
        assert!(plan.points.is_empty());
        assert!(!super::cold_replay_decode(&probe, 0));
        assert!(super::cold_replay_decode(&probe, 1));
        assert!(!super::cold_replay_decode(&probe, 2));
        assert!(!super::cold_replay_decode(&probe, 3));
        let plan = super::cold_replay_plan(&None, ordinary(), false);
        assert_eq!(plan.chunks, [8, 12]);
        assert_eq!(plan.points, [(0, 8)]);
        assert!(!super::cold_replay_decode(&None, 1));
        assert!(super::cold_replay_plan(&None, ordinary(), true).chunks.is_empty());
    }

    #[test]
    fn speculation_digest_keeps_text_and_binds_full_image_identity() {
        use cuteafd_loader::media::{ImageKey, MediaSpan};
        let tokens = [1, 4, 5, 6, 2];
        assert_eq!(super::media_digest(&tokens, &[]), tokens.iter().fold(super::DIGEST_SEED, |d, &t| super::digest(d, t)));
        let a = MediaSpan { start: 2, len: 1, key: ImageKey([0; 32]).into() };
        let mut b = a.clone(); let mut bytes = *b.key.bytes(); bytes[31] = 1; b.key = ImageKey(bytes).into();
        assert_ne!(super::media_digest(&tokens, &[a]), super::media_digest(&tokens, &[b]));
    }
}

/// Longest n-gram (from 8 down to 4 tokens) that ends the history and occurred
/// earlier; proposes up to `limit` tokens that followed its latest earlier
/// occurrence (a copy window). Exact: the verify step accepts only tokens the
/// model itself produces.
pub(crate) fn copy_drafts(history: &[u32], limit: usize) -> Vec<u32> {
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

/// The prefix cache over `engine` (always present: with zero entries it is the page allocator),
/// its marks in pool units or in the mark arena `arena`, sized by its rule over the engine's own
/// marks: the `engine.mark_slots` the KV admission reserved ([`super::prefix::ArenaMarks::slots_on`]).
fn prefix_cache<'e, 'a>(engine: &'e GlmfEngine<'a>, args: &PrefixArgs, (marks, arena): (PrefixMarks,
    super::prefix::ArenaMarks)) -> Result<(GlmfPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let entries = args.prefix_cache_entries;
    anyhow::ensure!(args.prefix_partial == Toggle::Off, "GLM 5.3 Flash restores exact snapshots only (KDA state)");
    let family = GlmfPrefix::new(engine, marks, |mark| arena.slots(mark))?;
    anyhow::ensure!(family.slots() == engine.mark_slots, "the prefix mark arena holds {} marks of {} B, the KV \
        admission reserved {}", family.slots(), family.mark_bytes(), engine.mark_slots);
    let template = engine.paged_buffers().first().map(|b| b.records).context("GLM 5.3 Flash has no MLA layer")?;
    // The pinned host tier copies through one GPU's copy engine; a head split keeps its pages
    // and marks on both GPUs, so it keeps device-resident snapshots only.
    let host = if engine.ranks() > 1 {
        if args.host_cache_bytes.enabled() && entries > 0 {
            tracing::warn!("GLM 5.3 Flash head split: the prefix cache's host tier is off (device-resident snapshots only)");
        }
        None
    } else {
        args.host_tier(engine.library, template, family.layout(), engine.max_context)?
    };
    let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
    let layout = family.layout();
    if marks == PrefixMarks::Pool && host.is_none() && entries > 0 {
        tracing::info!("GLM 5.3 Flash pool marks without the host tier: snapshots the pool evicts are dropped \
            (--host-cache-bytes auto keeps them in RAM)");
    }
    let config = PrefixConfig { entries, mark_slots: family.slots(), keep_logits: true,
        min_tokens: args.prefix_cache_min_tokens };
    let cache = PrefixCache::new(layout, config, host)?;
    // The pages requests and snapshots share: the pool, without the units pool marks reserve.
    let pages = cache.pool().capacity();
    tracing::info!(entries, mark_slots = family.slots(), mark_bytes = family.mark_bytes(),
        mark_store = ?layout.mark_store, page_bytes = layout.page_bytes, pages,
        reserved_pages = layout.mark_store.reserved(), page_rows = layout.page_rows, host_bytes,
        points = ?args.points(), "GLM 5.3 Flash prefix cache");
    cuteafd_bench::context::set_kv((pages * layout.page_rows) as u64, pages as u64,
        &"FP8 MLA latent + KDA state".to_string(), host_bytes);
    Ok((family, cache))
}

/// The drafter's context rings and the most sequences of one draft step, for `max_sequences`
/// sequences: the rings given, or one per sequence the scheduler can hold (it admits while
/// fewer than `max_sequences` are active or prefilling, and every finished, failed or scoring
/// sequence gives its ring back), with the draft batch at most the rings (it never holds more
/// sequences than are active). Each ring is 2,048 rows x 5 layers x K and V x 1,024 BF16 values
/// (41,943,040 B for DFlash2).
pub(crate) fn draft_capacity(max_sequences: usize, draft_sequences: usize, context_slots: Option<usize>)
    -> (Option<usize>, usize) {
    match context_slots {
        Some(slots) => (Some(slots), draft_sequences),
        None => {
            let slots = max_sequences.max(1);
            (Some(slots), draft_sequences.min(slots))
        }
    }
}

/// A drafter ring for a newly admitted sequence that drafts (`wanted`), counting the admissions
/// that found none (they decode without drafts until they finish).
fn take_ring(free: &mut Vec<usize>, wanted: bool, misses: &mut u64) -> Option<usize> {
    if !wanted {
        return None;
    }
    let slot = free.pop();
    if slot.is_none() {
        *misses += 1;
        tracing::warn!(misses = *misses, "a drafting request was admitted without a drafter ring: it decodes undrafted");
    }
    slot
}

/// Gives a finished or failed sequence's units, KDA slot and drafter slot back.
fn release(family: &GlmfPrefix<'_, '_>, cache: &mut PrefixCache<CudaCopyEngine<'_>>, kda: &mut Vec<i32>,
    slots: &mut Vec<usize>, placement: &GlmfPlacement, slot: Option<usize>) {
    if let Err(error) = cache.release(family, &placement.units) {
        tracing::error!(%error, "releasing a GLM 5.3 Flash sequence's units");
    }
    kda.push(placement.slot);
    slots.extend(slot);
}

#[derive(Clone, Copy, Default, serde::Serialize)]
struct VerifyBucket {
    steps: u64,
    speculative_steps: u64,
    verify_ms_sum: f64,
    verify_ms_max: f64,
}

/// Verify steps by real rows and by the rows they ran (their bucket), up to the wide programs' 128.
struct VerifyStats {
    real: [VerifyBucket; WIDE_DECODE_ROWS + 1],
    bucket: [VerifyBucket; WIDE_DECODE_ROWS + 1],
}

impl Default for VerifyStats {
    fn default() -> Self {
        Self { real: [VerifyBucket::default(); WIDE_DECODE_ROWS + 1],
            bucket: [VerifyBucket::default(); WIDE_DECODE_ROWS + 1] }
    }
}

impl VerifyStats {
    fn record(&mut self, rows: usize, bucket: usize, spec: bool, ms: f64) {
        for entry in [&mut self.real[rows], &mut self.bucket[bucket]] {
            entry.steps += 1;
            entry.speculative_steps += u64::from(spec);
            entry.verify_ms_sum += ms;
            entry.verify_ms_max = entry.verify_ms_max.max(ms);
        }
    }

    fn snapshot(&self) -> serde_json::Value {
        let timing = |entries: &[VerifyBucket]| entries.iter().enumerate().filter(|(_, v)| v.steps > 0)
            .map(|(rows, value)| (rows.to_string(), *value)).collect::<std::collections::BTreeMap<_, _>>();
        let rows = self.real.iter().enumerate().filter(|(_, v)| v.steps > 0)
            .map(|(rows, value)| (rows.to_string(), value.steps)).collect::<std::collections::BTreeMap<_, _>>();
        serde_json::json!({"rows": rows, "by_real_rows": timing(&self.real), "by_bucket": timing(&self.bucket),
            "timing_scope": "serving verify plus token selection host wall; existing synchronization boundary; excludes draft and commit; cumulative attempts including errors"})
    }
}

fn first_prefill_sample<T>(seen: &std::cell::Cell<[bool; 2]>, has_media: bool,
    sample: impl FnOnce() -> T) -> Option<T> {
    let index = usize::from(has_media);
    let mut kinds = seen.get();
    if kinds[index] { return None; }
    kinds[index] = true;
    seen.set(kinds);
    Some(sample())
}

#[cfg(test)]
mod memory_diagnostics_tests {
    use super::first_prefill_sample;
    use std::cell::Cell;

    #[test]
    fn samples_each_prefill_kind_once_even_when_query_is_unavailable() {
        let seen = Cell::new([false; 2]);
        let calls = Cell::new(0);
        for has_media in [false, false, true, true, false, true] {
            let before = calls.get();
            let sample = first_prefill_sample(&seen, has_media, || {
                calls.set(calls.get() + 1);
                None::<u64>
            });
            assert_eq!(sample.is_some(), calls.get() != before);
        }
        assert_eq!(calls.get(), 2);
        assert_eq!(seen.get(), [true, true]);
    }
}

fn prefill_memory_sample() -> serde_json::Value {
    use cuteafd_ffi::memory_ledger::{snapshot, Space};
    let runtime = cuteafd_ffi::memory_ledger::current_cuda_memory_snapshot();
    let ledger = snapshot();
    let device = runtime.as_ref().and_then(|r| r["device"].as_i64());
    let tracked = device.map(|device| ledger.total(Space::Device, device as i32)
        + ledger.total(Space::Managed, device as i32));
    serde_json::json!({"cuda_runtime": runtime, "tracked_device_and_managed_bytes": tracked})
}

/// Serving statistics for `/v1/stats`.
fn publish(stats: &Mutex<serde_json::Value>, requests: u64, generated: u64, active: usize, prefilling: usize,
    ring_misses: u64, cache: &PrefixCache<CudaCopyEngine<'_>>, media: &MediaAdmission<super::media::Prompt, super::media::Encoder>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>, verify: &VerifyStats,
    memory_boundaries: &std::cell::RefCell<Vec<serde_json::Value>>) {
    if let Ok(mut stats) = stats.lock() {
        *stats = serde_json::json!({"requests": requests, "generated_tokens": generated, "active": active,
            "prefilling": prefilling, "draft_ring_misses": ring_misses, "prefix_cache": cache.stats(),
            "verify": verify.snapshot(),
            "media": media.stats(cache.stats().media_key_collisions, preparer.map_or(0, |p| p.memo_hits()))});
        crate::shared::probe::graph_capture_stats(&mut stats);
        // Idle checkpoints only: never sample the process ledger on the decode hot path.
        if active == 0 && prefilling == 0 {
            use cuteafd_ffi::memory_ledger::{snapshot, Space};
            let ledger = snapshot();
            let devices: Vec<_> = ledger.devices().into_iter().map(|device| serde_json::json!({
                "device": device, "device_bytes": ledger.total(Space::Device, device),
                "managed_bytes": ledger.total(Space::Managed, device),
                "device_by_scope": ledger.by_scope(Space::Device, device),
                "managed_by_scope": ledger.by_scope(Space::Managed, device)})).collect();
            stats["memory_ledger"] = serde_json::json!({"devices": devices,
                "cuda_runtime": cuteafd_ffi::memory_ledger::current_cuda_memory_snapshot(),
                "first_prefill_boundaries": memory_boundaries.borrow().as_slice(),
                "pinned_bytes": ledger.rows.iter().filter(|r| r.key.space == Space::Pinned).map(|r| r.bytes).sum::<usize>(),
                "scope": "idle live allocations tracked through NativeLibrary; runtime contexts, modules, cuBLAS and graph executables excluded"});
        }
    }
}

/// Message starts of the GLM chat template: a snapshot right before one is a message boundary.
pub(crate) const MESSAGE_STARTS: [&str; 4] = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"];

#[allow(clippy::too_many_arguments)]
fn schedule(engine: &GlmfEngine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    policy: Policy, ranks: Option<usize>, decode_share: DecodeShareArgs, prefix: &PrefixArgs,
    marks: (PrefixMarks, super::prefix::ArenaMarks), select: SelectPlacement,
    media: &mut MediaAdmission<super::media::Prompt, super::media::Encoder>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>, ready: impl FnOnce())
    -> Result<()> {
    let (family, mut cache) = prefix_cache(engine, prefix, marks)?;
    // Made before the KV pool when start-up admitted it from measured memory.
    let mut selector = match engine.take_selector() {
        Some(selector) => selector,
        None => TokenSelector::new(&opened.library, select, engine.cfg.vocab_size, engine.decode_rows)?,
    };
    let markers = crate::shared::prefix::marker_ids(snapshot, &MESSAGE_STARTS)?;
    let mut free_kda: Vec<i32> = (0..engine.slots as i32).rev().collect();
    let mut grammars = crate::shared::constraints::Compiler::with_vocab(
        &opened.library, snapshot.join("tokenizer.json"), engine.cfg.vocab_size, engine.cfg.eos.clone());
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
    let drafter = engine.drafter.as_ref();
    let mut free_slots: Vec<usize> = drafter.map_or(Vec::new(), |d| (0..d.slots()).rev().collect());
    // Admissions that found no drafter ring (must stay 0: one ring per sequence the scheduler holds).
    let mut ring_misses = 0u64;
    // The TP2 table also prices TP4 as served (its observed ratio settles the
    // level); TP6 scales the Spark share by its widest slice against TP4's.
    let table = match ranks {
        Some(ranks) if ranks > 4 => dflash_policy::rescale_spark(&GLMF_TP2_STEP_MS, GLMF_TP2_GPU_MS, 512,
            dflash_policy::widest_slice(engine.cfg.moe_intermediate, ranks)),
        _ => GLMF_TP2_STEP_MS.to_vec(),
    };
    // A verify step schedules up to `verify_rows` rows (64, or the GPU's whole sparse MLA waves with
    // `--decode-rows 128`): the table, measured to 64 rows, extends past them and serving refits it.
    let verify_rows = engine.verify_rows;
    let mut cost = dflash_policy::step_cost(&table, verify_rows);
    let mut skip = dflash_policy::DraftSkip::default();
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total) = (0u64, 0u64);
    let mut verify_stats = VerifyStats::default();
    let memory_boundaries = std::cell::RefCell::new(Vec::<serde_json::Value>::new());
    let first_prefill_seen = std::cell::Cell::new([false; 2]);
    // Per request window: verify steps, and host seconds drafting, verifying
    // (engine step + commit) and selecting/streaming tokens.
    let (mut steps, mut draft_s, mut verify_s, mut emit_s) = (0u64, 0f64, 0f64, 0f64);
    let mut prefills = decode_share.queue::<Prefill<'_>>()?;
    let mut kv_waiter = cuteafd_engine::prefix::DeferredAdmission::<MediaReady<super::media::Prompt>>::default();
    publish(stats, requests, generated_total, 0, 0, ring_misses, &cache, media, preparer, &verify_stats,
        &memory_boundaries);
    ready();
    loop {
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("expert wire unavailable until restart: {reason}");
        }
        while active.len() + prefills.len() < max_sequences {
            let busy = !active.is_empty() || !prefills.is_empty();
            let ready = match kv_waiter.poll(cache.pool().free(), cache.pool().release_epoch(), busy,
                |ready| ready.job().job.events.is_closed()) {
                cuteafd_engine::prefix::AdmissionPoll::Blocked => break,
                cuteafd_engine::prefix::AdmissionPoll::Ready(ready) => ready,
                cuteafd_engine::prefix::AdmissionPoll::Empty => match media.poll(|prompt| prompt.job.events.is_closed()) {
                    MediaPoll::Ready(ready) => ready,
                    MediaPoll::Failed(prompt, error) => {
                        let _ = prompt.job.events.send(Err(super::media::failure(error)));
                        continue;
                    }
                    MediaPoll::Pending | MediaPoll::Empty => {
                        cache.tick();
                        publish(stats, requests, generated_total, active.len(), prefills.len(), ring_misses, &cache,
                            media, preparer, &verify_stats, &memory_boundaries);
                        let job = if !busy && media.is_empty() {
                            match receive.blocking_recv() { Some(job) => job, None => return Ok(()) }
                        } else {
                            match receive.try_recv() {
                                Ok(job) => job,
                                Err(_) => break,
                            }
                        };

                        if let Err(error) = probe::validate_scoring(&job.probe, engine.full_prefill_logits) {
                            let _ = job.events.send(Err(NativeFailure::BadRequest(format!("scoring: {error:#}"))));
                            continue;
                        }
                        if !job.media.is_empty() && !media.encoder().available() && !super::media::reference_probe(&job.probe) {
                            let _ = job.events.send(Err(NativeFailure::Unavailable("vision encoder unavailable".into())));
                            continue;
                        }
                        let tokens = match probe::prompt_ids(&job.probe,
                            || Ok(tokenizer.encode_text(&job.prompt, false)?.token_ids)) {
                            Ok(tokens) => tokens,
                            Err(error) => {
                                let _ = job.events.send(Err(NativeFailure::BadRequest(format!("prompt tokenization: {error:#}"))));
                                continue;
                            }
                        };
                        let events = job.events.clone();
                        let (prompt, mut request_media, jobs) = match super::media::prepare(job, tokens, &config,
                            engine.cfg.vocab_size, engine.cfg.hidden, engine.max_context) {
                            Ok(prepared) => prepared,
                            Err(error) => { let _ = events.send(Err(NativeFailure::BadRequest(format!("{error:#}")))); continue; }
                        };
                        if prompt.tokens.is_empty() || prompt.tokens.len() >= engine.max_context {
                            let _ = events.send(Err(NativeFailure::BadRequest(format!("prompt of {} tokens is outside 1..{}",
                                prompt.tokens.len(), engine.max_context))));
                            continue;
                        }
                        if let Err(error) = super::media::probe_features(&prompt, &mut request_media, &mut media.cache, snapshot) {
                            if let Some(probe) = &prompt.job.probe { probe.fail(format!("reference features: {error:#}")); }
                            let _ = events.send(Err(NativeFailure::BadRequest(format!("reference features: {error:#}"))));
                            drop(request_media); media.cache.prune_reservations();
                            continue;
                        }
                        let resume = if probe::cold(&prompt.job.probe) { 0 }
                            else { cache.peek_media(prompt.keys.tokens(), prompt.keys.spans(), true) };
                        let waiter = MediaWaiter::new(prompt, request_media, jobs, resume)?;
                        if let Err((_, error)) = media.enqueue(waiter) {
                            let _ = events.send(Err(super::media::failure(error)));
                        }
                        continue;
                    }
                },
            };
            if !ready.job().job.media.is_empty() && !media.encoder().available() && !super::media::reference_probe(&ready.job().job.probe) {
                let _ = ready.job().job.events.send(Err(NativeFailure::Unavailable("vision encoder unavailable".into())));
                continue;
            }
            let reject = |ready: &MediaReady<super::media::Prompt>, message: String| {
                let _ = ready.job().job.events.send(Err(NativeFailure::BadRequest(message)));
            };
            let constraint = match ready.job().job.constraint.as_ref().map(|spec| grammars.matcher(spec)).transpose() {
                Ok(constraint) => constraint,
                Err(error) => { reject(&ready, format!("{error:#}")); continue; }
            };
            if let Some(probe) = &ready.job().job.probe {
                if let Err(error) = probe.spec.validate_cold_steps(ready.job().tokens.len(), engine.prefill_capacity(), DECODE_ROWS) {
                    probe.fail(format!("cold replay: {error:#}"));
                    reject(&ready, format!("cold replay: {error:#}"));
                    continue;
                }
            }
            let cold = ready.cold() || probe::cold(&ready.job().job.probe);
            let capacity = (ready.job().tokens.len() + ready.job().job.max_tokens).min(engine.max_context);
            let Some(kda) = free_kda.pop() else {
                reject(&ready, "KDA state slots exhausted".into());
                continue;
            };
            // Disabled neural drafts need neither a ring slot nor context updates.
            let slot = take_ring(&mut free_slots, drafter.is_some() && !probe::no_speculation(&ready.job().job.probe)
                && policy.fixed != Some(0), &mut ring_misses);
            cache.tick();
            let admit_started = Instant::now();
            // Lookup, fork of the retained pages and restore of the mark (byte-exact).
            let build = |units| GlmfPlacement::new(units, kda);
            let admission = if cold { cache.admit_cold(&family, ready.job().tokens.len(), capacity, build) }
                else { cache.admit_media(&family, ready.job().keys.tokens(), ready.job().keys.spans(), capacity, true, build) };
            let admitted = match admission {
                Ok(admitted) => admitted,
                Err(error) => {
                    free_kda.push(kda);
                    free_slots.extend(slot);
                    // Running requests keep their pages pinned. Delay a request
                    // that fits alone instead of rejecting transient KV pressure.
                    match kv_waiter.defer(ready, &error, busy, cache.pool().release_epoch()) {
                        Ok(()) => break,
                        Err(ready) => reject(&ready, format!("{error:#}")),
                    }
                    continue;
                }
            };
            let resume = admitted.resume;
            let ready = match ready.reconcile(resume) {
                Ok(ready) => ready,
                Err(waiter) => {
                    release(&family, &mut cache, &mut free_kda, &mut free_slots, &admitted.placement, slot);
                    let events = waiter.job.job.events.clone();
                    if let Err((_, error)) = media.enqueue(waiter) {
                        let _ = events.send(Err(super::media::failure(error)));
                    }
                    continue;
                }
            };
            let (super::media::Prompt { job, tokens, keys }, request_media) = ready.into_parts();
            probe::admitted(&job.probe, "glm5_flash", &tokens, resume);
            let _ = job.events.send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: resume },
            }));
            if let Some(from) = probe::scoring(&job.probe) {
                // Teacher-forced scoring: every row's logits, no generation, nothing retained.
                let mut placement = admitted.placement;
                let scored = probe::score(&opened.library, &job.probe, &tokens, from, engine.prefill_capacity(),
                    DECODE_ROWS, probe::verify_rows(&job.probe), engine.full_prefill_logits, &mut placement,
                    |placement, chunk, rows| {
                        if rows > 1 {
                            engine.prefill_scoring_media(placement, chunk, &request_media)
                        } else {
                            Ok(engine.prefill_media_device(placement, chunk, &request_media)?.map(probe::ScoreLogits::Device))
                        }
                    },
                    |placement, chunk| engine.verify_media_device(&mut [(placement, chunk.len())], chunk, &request_media)?
                        .context("scoring needs every layer"));
                match scored {
                    Ok(_) => { let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })); }
                    Err(error) => { let _ = job.events.send(Err(NativeFailure::Worker(format!("scoring: {error:#}")))); }
                }
                release(&family, &mut cache, &mut free_kda, &mut free_slots, &placement, slot);
                continue;
            }
            let ticket = console::admit(tokens.len(), resume, job.max_tokens, constraint.is_some(), job.media.len(),
                admit_started);
            if let Some(source) = admitted.source {
                tracing::info!(tokens = tokens.len(), resume, kind = ?source.kind, frontier = source.frontier,
                    host = source.host, "prefix cache hit");
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
            let plan = cold_replay_plan(&job.probe, plan, logits.is_some());
            prefills.push(Prefill { job, constraint, tokens, keys, media: request_media, done: resume, resume, plan, chunks: 0, cancelled: false,
                placement: admitted.placement, capacity, slot, logits, first: None, prompt_row: None,
                started: Instant::now(), busy: 0.0, phases: [0.0; 3], ticket });
        }
        if prefills.due(!active.is_empty()) {
            let caching = cache.enabled();
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
                // One sample around the first text and first media prefill only;
                // no extra synchronization and no steady-state decode sampling.
                let media_index = usize::from(!p.media.spans().is_empty());
                let memory_before = first_prefill_sample(&first_prefill_seen, media_index != 0, prefill_memory_sample);
                let (result, phases) = isolated_phases(&engine.profile, || -> Result<()> {
                    let start = p.placement.len;
                    let logits = if cold_replay_decode(&p.job.probe, p.chunks) {
                        engine.verify_media_device(&mut [(&mut p.placement, chunk.len())], chunk, &p.media)?
                    } else {
                        engine.prefill_media_device(&mut p.placement, chunk, &p.media)?
                    };
                    if end == p.tokens.len() {
                        // The first token, while this prompt's logits are the workspace's.
                        let logits = logits.context("prefill produced no logits")?;
                        if caching && !probe::cold(&p.job.probe) && p.resume < p.tokens.len() {
                            p.prompt_row = Some(logits.row_host(&opened.library, 0)?);
                        }
                        if probe::wants_first(&p.job.probe) {
                            probe::device_rows(&opened.library, &p.job.probe, &logits, 0, 1, p.tokens.len())?;
                        }
                        let mut batch = SelectBatch::default();
                        batch.push_next(p.job.sampling, p.constraint.as_mut(), p.placement.len as u64)?;
                        let selected = selector.select(&logits, &batch)?;
                        p.first = Some(take(p.constraint.as_mut(), &selected[0])?);
                    }
                    // The chunk's tapped tail becomes drafter context before the next step.
                    if let (Some(drafter), Some(slot)) = (drafter, p.slot) {
                        let n = chunk.len().min(TAP_ROWS);
                        let first = start + chunk.len() - n;
                        drafter.update(&(0..n).map(|r| ContextRow { tap_row: r, slot, position: first + r })
                            .collect::<Vec<_>>())?;
                    }
                    Ok(())
                });
                if let Some(before) = memory_before {
                    let boundary = serde_json::json!({"kind": if media_index == 0 { "first-text-prefill" } else { "first-media-prefill" },
                        "rows": chunk.len(), "success": result.is_ok(), "before": before,
                        "after": prefill_memory_sample(),
                        "scope": "first prefill engine/selection/drafter work on serving thread; no extra synchronization; media encoder preparation precedes this boundary"});
                    tracing::info!(report = %boundary, "GLM Flash first prefill memory boundary");
                    memory_boundaries.borrow_mut().push(boundary);
                }
                p.done += chunk.len();
                add_phases(&mut p.phases, phases);
                p.busy += timer.elapsed().as_secs_f64();
                p.ticket.prefill(chunk.len(), p.chunks, p.plan.chunks.len(), timer);
                result?;
                // Intermediate snapshot points this chunk ends at (off unless configured).
                for &(_, point) in p.plan.points.iter().filter(|&&(chunk, _)| chunk == p.chunks && !probe::cold(&p.job.probe)) {
                    if let Err(error) = cache.capture_media(&family, SnapshotKind::Prompt, &p.keys.tokens()[..point], p.keys.spans(), &p.placement,
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
                            if let Err(error) = cache.park_media(&family, &p.keys.tokens()[..placement.len], p.keys.spans(), &placement) {
                                tracing::warn!("parking a cancelled prefill: {error:#}");
                            }
                        }
                    } else {
                        tracing::warn!("prefill failed: {error:#}");
                        let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    }
                    release(&family, &mut cache, &mut free_kda, &mut free_slots, &placement, slot);
                    continue;
                }
                // A whole-prompt hit selects from its retained logits; a prefill selected already.
                let first = match (p.first, p.logits.as_deref()) {
                    (Some(first), _) => Ok(first),
                    (None, Some(logits)) => select_host({
                        if probe::wants_first(&p.job.probe) {
                            probe::host_row(&p.job.probe, p.tokens.len(), logits);
                        }
                        p.constraint.as_mut() }, p.job.sampling, logits,
                        p.placement.len as u64),
                    (None, None) => Err(anyhow::anyhow!("prefill produced no logits")),
                };
                let first = match first {
                    Ok(first) => first,
                    Err(error) => {
                        tracing::warn!("{error:#}");
                        release(&family, &mut cache, &mut free_kda, &mut free_slots, &placement, slot);
                        continue;
                    }
                };
                let logits = p.prompt_row.take();
                if resume < p.tokens.len() {
                    tracing::info!(tokens = p.tokens.len(), cached = resume,
                        elapsed_ms = p.started.elapsed().as_millis() as u64, busy_ms = (1e3 * p.busy) as u64,
                        tok_s = (p.tokens.len() - resume) as f64 / p.busy, gpu_wait_ms = (1e3 * p.phases[0]) as u64,
                        experts_ms = (1e3 * p.phases[1]) as u64, head_ms = (1e3 * p.phases[2]) as u64, "prefill");
                }
                // The prompt snapshot, taken once the first token is out (it only enqueues copies).
                let prompt = (resume < p.tokens.len() && !probe::cold(&p.job.probe)).then(|| p.keys.tokens().to_vec());
                let spans = p.keys.spans().to_vec();
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &GlmfPlacement| {
                    if let (Some(prompt), Some(logits)) = (&prompt, &logits) {
                        if let Err(error) = cache.capture_media(&family, SnapshotKind::Prompt, prompt, &spans, placement,
                            After::from_logits(logits, true)) {
                            tracing::warn!("prompt snapshot not retained: {error:#}");
                        }
                    }
                };
                let job_events = p.job.events.clone();
                let admitted = (|| -> Result<Active<'_>> {
                    let mut request = Active {
                        digest: media_digest(&p.tokens, p.keys.spans()),
                        history: p.tokens,
                        keys: p.keys,
                        _media: p.media,
                        draft_limit: policy.copy,
                        draft_pause: 0,
                        slot,
                        drafts: DraftHistory::default(),
                        counts: [0; 6],
                        decoder: cuteafd_loader::streaming_token_decoder(snapshot, false)?,
                        job: p.job, constraint: p.constraint, placement: p.placement, draft_from: resume, turn: None,
                        capacity: p.capacity, next: 0, generated: 0, buffered: 0, started: Instant::now(),
                        ticket: p.ticket,
                    };
                    request.next = first;
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
                                release(&family, &mut cache, &mut free_kda, &mut free_slots, &request.placement,
                                    request.slot)
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!("admission failed: {error:#}");
                        let _ = job_events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        retain_prompt(&mut cache, &placement);
                        release(&family, &mut cache, &mut free_kda, &mut free_slots, &placement, slot);
                    }
                }
            }
            prefills.settle(!active.is_empty());
        }
        if active.is_empty() {
            if !media.is_empty() { std::thread::park_timeout(Duration::from_millis(1)); }
            continue;
        }
        let cycle = Instant::now();
        let mut tally = console::Step::begin(0);
        let (draft0, verify0, emit0) = (draft_s, verify_s, emit_s);
        let engine_before = tally.live().then(|| *engine.profile.borrow());
        // Rows each sequence may add after its next token: an even share of the verify budget.
        let room = (verify_rows / active.len()).max(1) - 1;
        let mut limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else {
            room.min(a.job.max_tokens - a.generated - 1).min(a.capacity - a.placement.len - 1) }).collect();
        // With the wide programs the rows the even share leaves over go to the first sequences that can
        // draft once more (127 rows at 16 sequences: 15 draft 7, one 6); 64 rows keep the even share.
        if engine.decode_rows > DECODE_ROWS {
            super::engine::hand_out_remainder(&mut limits, room, verify_rows, |i| {
                let a = &active[i];
                !probe::no_speculation(&a.job.probe)
                    && room < (a.job.max_tokens - a.generated - 1).min(a.capacity - a.placement.len - 1)
            });
        }
        // DFlash2 drafts after every next token, then the policy's counts.
        let timer = Instant::now();
        let drafted: Vec<Option<Draft>> = match drafter {
            Some(drafter) if skip.drafts() && active.iter().enumerate().any(|(i, a)| a.slot.is_some() && limits[i] > 0) => {
                let seqs: Vec<(usize, DraftSeq)> = active.iter().enumerate()
                    .filter_map(|(i, a)| a.slot.filter(|_| limits[i] > 0).map(|slot|
                        (i, DraftSeq { slot, anchor: a.next, position: a.placement.len, valid_from: a.draft_from })))
                    .take(drafter.max_batch_sequences())
                    .collect();
                for &(i, _) in &seqs {
                    active[i].counts[5] += 1;
                }
                let drafts = drafter.draft_device(&seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), &engine.embedding,
                    engine.draft_head());
                cost.observe_draft(active.len(), timer.elapsed().as_secs_f64() * 1e3);
                let mut out = vec![None; active.len()];
                match drafts {
                    Ok(drafts) => {
                        for ((i, _), draft) in seqs.into_iter().zip(drafts) {
                            out[i] = Some(draft);
                        }
                    }
                    // Drafts only speed decoding up; the step verifies the next tokens alone.
                    Err(error) => tracing::warn!("{} draft failed: {error:#}", drafter.name()),
                }
                out
            }
            _ => vec![None; active.len()],
        };
        draft_s += timer.elapsed().as_secs_f64();
        // Identical sequences (same tokens at the same position) route alike
        // and draft alike: the policy prices and plans them as one group.
        let key = |a: &Active<'_>| (a.placement.len, a.digest);
        let inputs: Vec<dflash_policy::PlanInput<'_>> = active.iter().enumerate().map(|(i, a)| dflash_policy::PlanInput {
            key: key(a), history: &a.drafts, features: drafted[i].as_ref().map(|d| d.features.as_slice()),
            confidence: drafted[i].as_ref().map(|d| d.confidence.as_slice()).filter(|c| !c.is_empty()),
            limit: limits[i],
        }).collect();
        let plan_timer = Instant::now();
        let planned = dflash_policy::plan_counts(&inputs, policy.fixed, &cost);
        drop(inputs);
        skip.after(drafted.iter().any(Option::is_some) && policy.fixed.is_none(), planned.iter().all(|&n| n == 0));
        // Each sequence verifies its next token, then its DFlash2 drafts, or
        // a copy-window draft when it agrees with them and runs longer.
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
            let draft = if copy.len() > dflash.len() && agrees {
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
        let plan_us = console::us(plan_timer);
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let distinct_rows: usize = active.iter().zip(&sequences).map(|(a, rows)| (key(a), rows))
            .collect::<std::collections::HashSet<_>>().iter().map(|(_, rows)| rows.len()).sum();
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        // A step with drafts runs speculatively and commits what it keeps.
        let spec = sequences.iter().any(|rows| rows.len() > 1);
        let mut rows: Vec<(&mut GlmfPlacement, usize)> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.len())).collect();
        steps += 1;
        let timer = Instant::now();
        let step = engine.verify_device(&mut rows, &tokens, spec).and_then(|logits| logits.context("decode needs every layer"))
            .and_then(|logits| {
                // Each row draws at the position after it, masked along its sequence's drafts.
                let mut batch = SelectBatch::default();
                for (i, ((a, rows), &start)) in active.iter().zip(&sequences).zip(&starts).enumerate() {
                    // A grammar failure fails that sequence alone, after the step.
                    poisoned[i] = batch.push_sequence_isolated(a.job.sampling, a.constraint.as_ref(), rows, start as u64 + 1);
                }
                Ok((selector.select(&logits, &batch)?, logits))
            });
        let step_ms = timer.elapsed().as_secs_f64() * 1e3;
        verify_stats.record(tokens.len(), engine.serving_decode_rows(tokens.len(), spec), spec, step_ms);
        verify_s += step_ms / 1e3;
        let (selected, logits) = match step {
            Ok(step) => step,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(&family, &mut cache, &mut free_kda, &mut free_slots, &request.placement, request.slot);
                }
                continue;
            }
        };
        cost.observe_verify(Shape { rows: tokens.len(), distinct: distinct_rows, sequences: sequences.len() }, step_ms);
        let timer = Instant::now();
        let mut offset = 0;
        let mut context = Vec::new();
        let mut commits = Vec::new();
        let mut kept = Vec::new();
        let caching = cache.enabled();
        let before: Vec<usize> = active.iter().map(|a| a.history.len()).collect();
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
                        if done && caching && !probe::cold(&request.job.probe) {
                            // A normal finish (the client took the last chunk): the row that
                            // produced the last token follows the turn snapshot.
                            request.turn = logits.row_host(&opened.library, offset + j).ok();
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
            // A turn snapshot captures the KDA state at the kept length: commit it too.
            if !finished || request.turn.is_some() {
                commits.push((request.placement.slot, offset, committed));
                kept.push(i);
            }
            if !finished {
                if let Some(slot) = request.slot {
                    context.extend((0..committed).map(|r| ContextRow { tap_row: offset + r, slot, position: start + r }));
                }
            }
            offset += rows.len();
            let (drafted, accepted) = (rows.len() - 1, committed - 1);
            request.counts[0] += 1;
            if used_copy[i] {
                request.counts[3] += drafted;
                request.counts[4] += accepted;
            } else {
                request.counts[1] += drafted;
                request.counts[2] += accepted;
            }
            if planned[i] > 0 {
                request.drafts.observe(planned[i], accepted);
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
        emit_s += timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        if spec {
            engine.commit(&commits)?;
            for &i in &kept {
                active[i].placement.kda_len = active[i].placement.len;
            }
        }
        if let Some(drafter) = drafter {
            drafter.update(&context)?;
        }
        verify_s += timer.elapsed().as_secs_f64();
        for (i, request) in active.iter().enumerate() {
            let proposal = if used_copy[i] { &sequences[i][1..] } else { drafted[i].as_ref().map_or(&[][..], |d| &d.tokens) };
            tally.member(&request.ticket, proposal, sequences[i].len() - 1, &request.history[before[i]..],
                request.constraint.is_some(), finished[i]);
        }
        tally.end(|| {
            let engine_now = *engine.profile.borrow();
            let gpu = |i: usize| engine_before.map_or(f64::NAN, |before| 1e6 * (engine_now[i] - before[i]));
            vec![("draft", 1e6 * (draft_s - draft0)), ("plan", plan_us), ("verify", 1e3 * step_ms),
                ("emit", 1e6 * (emit_s - emit0)), ("commit", 1e6 * (verify_s - verify0) - 1e3 * step_ms),
                ("gpu", gpu(0)), ("experts", gpu(1)), ("head", gpu(2))]
        });
        for index in (0..active.len()).rev() {
            if !finished[index] {
                continue;
            }
            let mut request = active.remove(index);
            request.ticket.done(request.generated);
            requests += 1;
            generated_total += request.generated as u64;
            let seconds = request.started.elapsed().as_secs_f64();
            let phases = std::mem::take(&mut *engine.profile.borrow_mut());
            let [steps_seen, dflash, dflash_ok, copy, copy_ok, draft_calls] = request.counts;
            let graphs = engine.graph_stats();
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps = steps_seen, all_steps = steps, dflash, dflash_ok, copy, copy_ok, draft_calls,
                draft_s, verify_s, emit_s, gpu_wait_s = phases[0], experts_s = phases[1], head_s = phases[2],
                graph_captures = graphs.stats.captures, graph_recaptures = graphs.stats.recaptures,
                graph_held = graphs.held, graph_shapes = graphs.shapes, graph_bytes = graphs.bytes,
                draft_ring_misses = ring_misses, "request complete");
            (steps, draft_s, verify_s, emit_s) = (0, 0.0, 0.0, 0.0);
            if let Some(row) = &request.turn {
                // The conversation so far: every committed row (the last token is not in it).
                let keys = MediaKeys::new(&request.history, engine.cfg.vocab_size as u32, request.keys.spans())?;
                let rows = &keys.tokens()[..request.placement.len];
                if let Err(error) = cache.capture_media(&family, SnapshotKind::Turn, rows, request.keys.spans(), &request.placement,
                    After::from_logits(row, true)) {
                    tracing::warn!("turn snapshot not retained: {error:#}");
                }
            }
            release(&family, &mut cache, &mut free_kda, &mut free_slots, &request.placement, request.slot);
        }
        cache.tick();
        publish(stats, requests, generated_total, active.len(), prefills.len(), ring_misses, &cache, media, preparer,
            &verify_stats, &memory_boundaries);
        console::gauges(|| console::Gauges::prefix_cache(&cache, active.len(), prefills.len(), receive.len() + kv_waiter.len() + media.len()));
        prefills.stepped(cycle.elapsed().as_secs_f64());
    }
}

/// Coordinator share of `GLMF_TP2_STEP_MS` at one row (GPU 6.7 ms of 18.8).
const GLMF_TP2_GPU_MS: f64 = 6.7;
/// GLM 5.3 Flash, 1 RTX PRO 6000 (325 W) + Spark TP2 (rhea, moa), recommended
/// FP8 decode config: speculative verify step ms by rows of one sequence
/// (glmf-golden --bench-verify 16 after 512 tokens, median of 7); past 16
/// rows extrapolated at the 12-16 slope (serving refits intercept and slope).
const GLMF_TP2_STEP_MS: [(usize, f64); 15] = [(1, 19.1), (2, 26.0), (3, 30.2), (4, 35.2), (5, 40.1), (6, 43.5),
    (7, 48.7), (8, 53.8), (10, 60.7), (12, 67.4), (16, 80.2), (24, 106.0), (32, 132.0), (48, 183.0), (64, 234.0)];

#[cfg(test)]
mod ring_tests {
    use super::{draft_capacity, take_ring};

    #[test]
    fn rings_default_to_one_per_sequence_with_the_draft_batch_within_them() {
        // 16 sequences: 16 rings (4 x 41,943,040 B fewer than the 20 of max(20, 16, 5C/4)).
        assert_eq!(draft_capacity(16, 16, None), (Some(16), 16));
        assert_eq!(draft_capacity(8, 16, None), (Some(8), 8));
        assert_eq!(draft_capacity(64, 16, None), (Some(64), 16));
        assert_eq!(draft_capacity(0, 16, None), (Some(1), 1));
        // Explicit rings stay as given.
        assert_eq!(draft_capacity(16, 16, Some(20)), (Some(20), 16));
        assert_eq!(draft_capacity(4, 16, Some(2)), (Some(2), 16));
    }

    #[test]
    fn admissions_without_a_ring_are_counted() {
        let (mut free, mut misses) = (vec![1usize, 0], 0u64);
        assert_eq!(take_ring(&mut free, true, &mut misses), Some(0));
        assert_eq!(take_ring(&mut free, false, &mut misses), None);
        assert_eq!(take_ring(&mut free, true, &mut misses), Some(1));
        assert_eq!(misses, 0);
        assert_eq!(take_ring(&mut free, true, &mut misses), None);
        assert_eq!(misses, 1);
        assert_eq!(take_ring(&mut free, false, &mut misses), None);
        assert_eq!(misses, 1);
    }
}
