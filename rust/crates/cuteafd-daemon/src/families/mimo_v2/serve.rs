//! OpenAI-compatible API over the MiMo V2 engine: continuous batching with
//! independent prefill chunks paired when their existing single-lane shapes
//! fit, and one decode-shaped step for every active sequence, verifying drafts.
//!
//! MiMo keeps no recurrent state: full layers write paged records and SWA
//! layers a 256-slot ring that outlives the 128-token window by more than a
//! step's rows, so a verify whose drafts are rejected just sets the
//! sequence length back.
//!
//! Prefix cache (`cuteafd_engine::prefix` over `super::prefix::MimoPrefix`):
//! admission looks the prompt up, restores the longest retained exact
//! frontier (shared pages, the copied tail page, the SWA/MTP mark) and
//! prefills only the rest; a whole-prompt hit takes its first token from the
//! retained logits. The prompt is retained at prompt end (a `Prompt`
//! snapshot, unless it was a whole hit), the conversation at a normal finish
//! (`Turn`, EOS or max_tokens with the client still there); a prefill whose
//! client left is parked at its last chunk boundary; a decode whose client
//! left is not retained. `prompt_cache_hit_tokens` reports the restored rows.
use super::dflash::{ContextRow, DraftSeq};
use super::mtp::MtpSeq;
use super::engine::{MimoEngine, MimoPlacement, DECODE_ROWS};
use super::prefix::MimoPrefix;
use super::serve_failures::FailureRecipients;
use crate::families::deepseek_v41::v41_native_serve::prefix::CudaCopyEngine;
use cuteafd_engine::media::{EmbeddingCache, MediaAdmission, MediaPoll, MediaReady, MediaWaiter, RequestMedia, MediaKeys};
use cuteafd_engine::prefix::{After, MarkArena, PointPolicy, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
use crate::families::glm5::dflash_policy::{self, CycleCost, DraftHistory, Group, Shape};
use super::{open, Opened};
use crate::shared::token_io::{SelectBatch, SelectPlacement, TokenSelector};
use crate::shared::prefill_share::{add_phases, isolated_phases, Chunk, DecodeShareArgs};
use anyhow::{Context, Result};
use cuteafd_api::openai::chat::qwen4::QwenEncoding;
use cuteafd_api::openai::{
    InferenceChunk, InferenceFinishReason, ModelEncoding, ModelProfile, NativeFailure, NativeLimits, NativeRequest,
    PromptUsage,
};
use crate::shared::console;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Most copy-window draft tokens verified per sequence and step.
const COPY_DRAFT: usize = 7;

#[path = "copy.rs"]
mod copy;

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: super::EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Sequences decoding at once (each holds an SWA ring).
    #[arg(long, default_value_t = 4)]
    pub max_sequences: usize,
    /// Bounded pending API requests (default at least the number of decoding slots).
    #[arg(long, env = "CUTEAFD_HTTP_QUEUE_DEPTH")]
    pub http_queue_depth: Option<usize>,
    /// Time a caller waits for a pending-queue permit.
    #[arg(long, env = "CUTEAFD_HTTP_QUEUE_WAIT_MS", default_value_t = 25_000)]
    pub http_queue_wait_ms: u64,
    /// Target seconds per long prefill chunk while another request decodes (experimental).
    #[arg(long, env = "CUTEAFD_MIMO_PREFILL_CHUNK_S")]
    pub prefill_chunk_s: Option<f64>,
    /// Cap automatic pinned memory at MiMo's RAM ceiling; enable with --host-cache-bytes auto.
    #[arg(long, default_value_t = false)]
    pub mimo_host_cache: bool,
    /// Wait for an identical in-flight prefill's snapshot before admitting an extension.
    #[arg(long, env = "CUTEAFD_MIMO_SNAPSHOT_WAIT", default_value_t = false)]
    pub mimo_snapshot_wait: bool,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    /// Decode one token per step (no copy-window drafts).
    #[arg(long)]
    pub no_copy_drafts: bool,
    /// Indexed eight-token greedy copy windows that replace neural drafts (experimental).
    #[arg(long, env = "CUTEAFD_MIMO_COPY_WINDOWS", default_value_t = false)]
    pub mimo_copy_windows: bool,
    /// With --draft: verify this many DFlash drafts per sequence every step
    /// instead of the adaptive plan.
    #[arg(long)]
    pub draft_fixed: Option<usize>,
    #[command(flatten)]
    pub decode_share: DecodeShareArgs,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub console: console::ConsoleArgs,
    /// Resolved global vision policy, assigned before dispatch.
    #[arg(skip = cuteafd_loader::plan::MediaMode::Off)]
    pub vision: cuteafd_loader::plan::MediaMode,
    #[arg(skip = cuteafd_loader::plan::MediaMode::Off)]
    pub audio: cuteafd_loader::plan::MediaMode,
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
    /// Independently admitted audio tower endpoints.
    #[arg(long)]
    pub audio_peers: Option<String>,
    #[arg(long, requires = "audio_peers")]
    pub audio_encoder_plan_hash: Option<String>,
    #[arg(long, requires = "audio_peers")]
    pub audio_encoder_revision: Option<String>,
    /// Exact SM121 backend/export identity reported by the reserved worker.
    #[arg(long, requires = "audio_peers")]
    pub audio_encoder_backend: Option<String>,
}

pub(crate) use crate::shared::prefix::{PrefixArgs, Toggle};
use crate::shared::probe;

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    anyhow::ensure!(args.prefill_chunk_s.is_none_or(|s| s.is_finite() && s > 0.0 && s <= 5.0),
        "--prefill-chunk-s must be finite and in (0, 5]");
    anyhow::ensure!(args.prefill_chunk_s.is_none() || args.decode_share.decode_share > 0.0,
        "--prefill-chunk-s requires a positive --decode-share");
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let limits = NativeLimits::new(args.engine.max_context as u32, args.max_output)?;
    // MiMo's template and tool calls follow Qwen3-Coder's XML (`<tool_call>
    // <function=NAME><parameter=KEY>VALUE</parameter>`), its reasoning `<think>`.
    let encoding = QwenEncoding::from_snapshot(&snapshot)?;
    let mut profile = ModelProfile::new(
        args.model_id.clone().or_else(|| crate::families::glm5_flash::serve::model_id(&snapshot)).context("model id")?,
        ModelEncoding::Qwen(Arc::new(encoding)),
    );
    anyhow::ensure!(args.max_sequences > 0 && args.max_sequences <= DECODE_ROWS,
        "--max-sequences must be in 1..={DECODE_ROWS}");
    let depth = args.http_queue_depth.unwrap_or(args.max_sequences.max(16));
    anyhow::ensure!(depth > 0, "--http-queue-depth must be positive");
    let (queue, receive) = mpsc::channel::<NativeRequest>(depth);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    engine_args.rings = engine_args.rings.max(args.max_sequences);
    let (worker_stats, max_sequences) = (stats.clone(), args.max_sequences);
    engine_args.draft_sequences = engine_args.draft_sequences.max(args.max_sequences);
    engine_args.draft_context_slots = Some(usize::try_from(
        cuteafd_loader::families::mimo_v2::draft_representation::mimo_draft_context_slots(
            engine_args.draft_sequences as u64, engine_args.rings as u64,
            engine_args.draft_context_slots.map(|n| n as u64)))?);
    let draft = Policy { copy: if args.no_copy_drafts { 0 } else { COPY_DRAFT }, fixed: args.draft_fixed,
        decode_share: args.decode_share, chunk_s: args.prefill_chunk_s, indexed_copy: args.mimo_copy_windows,
        snapshot_wait: args.mimo_snapshot_wait };
    let mut prefix = args.prefix.clone();
    prefix.mimo_host_cap = args.mimo_host_cache;
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id)));
    let vision = args.vision;
    let audio = args.audio;
    let remote = super::media::RemoteVision::from_args(&args)?;
    let remote_audio = super::media::RemoteAudio::from_args(&args)?;
    let media_cache_bytes = args.media_cache_bytes;
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, draft, prefix, vision, audio, media_cache_bytes, remote, remote_audio));
    if let Some((preparer, audio_preparer, health, audio_health)) = ready_rx.await.context("engine failed before it was ready")?? {
        if let Some(preparer) = preparer {
            profile = profile.with_loaded_vision(preparer);
            profile.vision_health = health.clone();
        }
        if let Some(preparer) = audio_preparer {
            profile = profile.with_loaded_audio(preparer, audio_health.context("audio owner health")?);
        }
    }
    cuteafd_bench::context::phase("engine loaded");
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, Duration::from_millis(args.http_queue_wait_ms),
        hub.clone(), profile.clone());
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    tracing::info!(listen = %args.listen, model = %profile.id, "MiMo V2 API is ready");
    tokio::select! {
        served = axum::serve(listener, cuteafd_bench::app(router, hub)
            .into_make_service_with_connect_info::<std::net::SocketAddr>()) => served?,
        finished = worker => finished??,
    }
    Ok(())
}

/// What the live console shows for MiMo V2 (Flash and V2.6 Pro).
fn console_layout(args: &ServeArgs, model: &str) -> console::Layout {
    use console::{Color::*, StepGroup};
    let mut layout = console::Layout::new("mimo_v2", model.into(), args.engine.snapshot.clone());
    let sparks = args.engine.peers.as_deref().map_or(0, |peers| peers.split(',').count());
    layout.hardware = console::hardware(1 + usize::from(args.engine.split_device.is_some()), sparks,
        args.engine.local_experts);
    layout.split = args.engine.split_device.map(|_| "head split".into());
    layout.concurrency = args.max_sequences.min(DECODE_ROWS);
    let copy = if args.no_copy_drafts { "" } else { " · copy windows" };
    let policy = args.draft_fixed.map_or_else(|| "adaptive".to_string(), |n| format!("fixed {n}"));
    let drafter = if args.engine.draft.is_some() { Some("DFlash") } else if args.engine.mtp > 0 { Some("MTP") } else { None };
    layout.speculator = match (drafter, args.no_copy_drafts) {
        (Some(name), _) => Some(console::Speculator { name: name.into(), positions: 8, policy: policy + copy }),
        (None, false) => Some(console::Speculator { name: "Copy window".into(), positions: COPY_DRAFT,
            policy: "adaptive length".into() }),
        (None, true) => None,
    };
    layout.steps = vec![
        StepGroup::new("Decode step", "host clock", &[("round.cycle", "step", Target),
            ("draft", "draft (DFlash / MTP)", Accepted), ("plan", "draft plan + copy windows", Ink),
            ("verify", "verify pass + token selection", Target), ("emit", "accept + stream + drafter context", Ink)]),
        StepGroup::new("Verify pass", "host clock, engine phases", &[
            ("gpu", "GPU until expert exchanges", Rtx), ("experts", "expert exchanges", Spark)]),
        StepGroup::layers(),
        StepGroup::admission(false),
    ];
    layout.layers = Some(console::Layers::host_clock());
    layout
}

/// Draft settings: copy-window draft length, fixed DFlash draft count.
#[derive(Debug, Clone, Copy)]
struct Policy {
    copy: usize,
    fixed: Option<usize>,
    decode_share: DecodeShareArgs,
    chunk_s: Option<f64>,
    indexed_copy: bool,
    snapshot_wait: bool,
}

/// Verify-step ms by rows for MiMo V2.6 Pro, RTX PRO 6000 (GPU0, 325 W) + six
/// Sparks (TP6 MXFP4, FP8 wire rows): teacher-forced decode steps of one
/// sequence (`mimo-golden --timing --prefill 1000 --step-rows N`, 571 rows).
/// The coordinator alone (`--skip-experts`) takes 18.2 / 19.2 / 23.3 / 33.8 /
/// 37.2 ms at 1 / 4 / 16 / 24 / 32 rows (E4M3 decode weights up to 32 rows)
/// and 56.7 ms at 48; the rest is the Spark exchange. Serving refits its
/// intercept and slope as it observes (sequences that share experts make rows
/// cheaper than one sequence's).
const PRO_TP6_STEP_MS: [(usize, f64); 9] = [(1, 31.6), (2, 39.5), (4, 54.1), (8, 77.1), (16, 114.4), (24, 166.7),
    (32, 197.8), (48, 258.3), (64, 304.4)];

type VisionReady = Option<(Option<Arc<cuteafd_api::openai::media::MediaPreparer>>,
    Option<Arc<cuteafd_api::openai::media::audio::AudioPreparer>>, Option<Arc<std::sync::atomic::AtomicBool>>, Option<Arc<std::sync::atomic::AtomicBool>>)>;

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<VisionReady>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    draft: Policy, prefix: PrefixArgs, vision: cuteafd_loader::plan::MediaMode, audio: cuteafd_loader::plan::MediaMode, media_cache_bytes: Option<u64>, remote: Option<super::media::RemoteVision>, remote_audio: Option<super::media::RemoteAudio>) -> Result<()> {
    let opened = match open(&args) {
        Ok(opened) => opened,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let (vision, prefix) = match super::media::ReadyVision::load(&args, &opened.library, vision, audio, &prefix, media_cache_bytes, remote, remote_audio) {
        Ok(vision) => vision,
        Err(error) => { let _ = ready.send(Err(error)); return Ok(()); }
    };
    let preparer = vision.as_ref().and_then(|vision| vision.preparer.clone());
    let audio_preparer = vision.as_ref().and_then(|vision| vision.audio_preparer.clone());
    let (encoder, bytes) = vision.map_or((super::media::Encoder::Off, 0), |vision|
        (vision.encoder, vision.cache_bytes));
    let health = encoder.health_handle(false);
    let audio_health = encoder.health_handle(true);
    let mut media = MediaAdmission::new(EmbeddingCache::new(bytes), encoder, 16);
    let mut ready = Some(ready);
    let result = opened.with_engine_reserved(&args, Some((&prefix, max_sequences)),
        if args.full_prefill_logits { cuteafd_loader::families::mimo_v2::MimoPrefillOutput::AllRows }
        else { cuteafd_loader::families::mimo_v2::MimoPrefillOutput::LastRow }, |engine, host_config| {
        anyhow::ensure!(engine.weights.layers.len() == engine.cfg.layers, "serve-mimo needs every layer");
        anyhow::ensure!(engine.has_experts(), "serve-mimo needs --peers (or --local-experts) for the routed experts");
        let spark = args.peers.is_some() && !args.local_experts;
        console::layer_classes(engine.weights.layers.iter().map(|l| console::layer_class(l.dense, spark)).collect());
        // Every decode/verify shape the scheduler can step (1..=64 rows) is captured before
        // the server is ready: serving replays them without capturing.
        let started = std::time::Instant::now();
        let graphs = engine.capture_decode_graphs(DECODE_ROWS)?;
        if graphs > 0 {
            tracing::info!(graphs, rows = DECODE_ROWS, elapsed_ms = started.elapsed().as_millis() as u64,
                "MiMo decode graphs captured");
        }
        anyhow::ensure!(preparer.is_none() || media.encoder().available_for(false), "vision encoder unavailable before readiness");
        anyhow::ensure!(audio_preparer.is_none() || media.encoder().available_for(true), "audio encoder unavailable before readiness");
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok((preparer.is_some() || audio_preparer.is_some()).then(|| (preparer.clone(), audio_preparer.clone(), health.clone(), audio_health.clone()))));
        }
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), draft, &prefix,
            args.token_io.token_select, host_config, &mut media, preparer.as_deref())
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| (preparer.is_some() || audio_preparer.is_some()).then(|| (preparer, audio_preparer, health, audio_health))).map_err(|e| anyhow::anyhow!("{e:#}")));
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
    plan: cuteafd_engine::prefix::PointPlan,
    /// Chunks prefilled so far.
    chunks: usize,
    paired_chunks: usize,
    /// The client left mid-prefill: its prefilled rows are parked as a prompt snapshot.
    cancelled: bool,
    placement: super::engine::MimoPlacement,
    capacity: usize,
    slot: Option<usize>,
    /// The last prompt row's logits: a whole-prompt hit's retained row, or
    /// (with the prefix cache on) the prefill's, for the prompt snapshot.
    logits: Option<Vec<f32>>,
    /// The first generated token, selected on the device after the last chunk.
    first: Option<u32>,
    started: Instant,
    /// Seconds in this prompt's chunks, and their engine phases.
    busy: f64,
    phases: [f64; 2],
    ticket: console::Ticket,
}

struct Active<'a> {
    job: NativeRequest,
    /// Prompt and generated tokens, for copy-window drafts.
    history: Vec<u32>,
    copy_index: copy::CopyIndex,
    keyed_history: Vec<u32>,
    media: RequestMedia,
    /// The sequence's DFlash ring slot (None: copy-window drafts only).
    slot: Option<usize>,
    /// Recent DFlash (proposed, accepted) outcomes for the adaptive plan.
    drafts: DraftHistory,
    /// Steps, neural drafts verified / accepted, copy drafts verified / accepted,
    /// and actual neural-drafter calls for this request.
    counts: [usize; 6],
    /// Current copy-draft length (halved after a fully rejected draft,
    /// doubled after a fully accepted one) and steps left before drafting
    /// resumes once it reached zero.
    draft_limit: usize,
    draft_pause: usize,
    constraint: Option<crate::shared::constraints::State<'a>>,
    placement: MimoPlacement,
    /// First position the DFlash drafter's context holds for this sequence
    /// (the prefix-cache restore point; the drafter starts cold there).
    draft_from: usize,
    /// The logit row that produced the last token, once the request finished
    /// normally (EOS or max_tokens with the client still there) and the prefix
    /// cache retains turns: what follows its `Turn` snapshot.
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
    /// Selects from host logits (a whole-prompt hit's retained row).
    fn select_host(&mut self, logits: &[f32]) -> Result<u32> {
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

    /// Streams `token` (special tokens stay text for the output parser);
    /// returns true when the request is finished.
    fn emit(&mut self, token: u32) -> Result<bool> {
        probe::token(&self.job.probe, token);
        self.history.push(token);
        self.keyed_history.push(token);
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

/// The prefix cache over `engine` (always present: with zero entries it is the page allocator).
pub(super) fn prefix_cache<'e, 'a>(engine: &'e MimoEngine<'a>, args: &PrefixArgs, concurrency: usize)
    -> Result<(MimoPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    prefix_cache_with(engine, args, concurrency, |layout| args.host_config(layout, engine.max_context))
}

fn prefix_cache_with<'e, 'a>(engine: &'e MimoEngine<'a>, args: &PrefixArgs, concurrency: usize,
    host_config: impl FnOnce(cuteafd_engine::prefix::FamilyLayout)
        -> Result<Option<cuteafd_hostcache::config::Config>>)
    -> Result<(MimoPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let entries = args.prefix_cache_entries;
    let budget = args.prefix_cache_mark_mib.checked_mul(1 << 20).context("MiMo mark budget overflows")?;
    let family = MimoPrefix::new_with_draft(engine, |mark| MarkArena::slots_for(concurrency, entries, mark, budget),
        args.prefix_partial == Toggle::On, args.mimo_prefix_draft)?;
    let layout = family.layout();
    // Serving consumes the quota resolved before loading; diagnostics may
    // resolve it here. Register all pools/mark arenas on their actual GPUs.
    let host = host_config(layout)?.map(|config| {
        CudaCopyEngine::registered_owned(engine.library, family.host_owners()).map(|copy| (config, copy))
    }).transpose()?;
    let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
    let config = PrefixConfig { entries, mark_slots: family.slots(), keep_logits: true,
        min_tokens: args.prefix_cache_min_tokens };
    let cache = PrefixCache::new(layout, config, host)?;
    tracing::info!(entries, mark_slots = family.slots(), mark_bytes = family.mark_bytes(), page_bytes = layout.page_bytes,
        pages = layout.pages, host_bytes, rule = ?layout.rule, points = ?args.points(),
        reach = family.capture_reach(), "MiMo prefix cache");
    cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, layout.pages as u64,
        &format!("{:?} full + BF16 SWA rings", engine.kv_cache()), host_bytes);
    Ok((family, cache))
}

/// Gives a finished or failed sequence's pages, ring and drafter slot back.
fn release(family: &MimoPrefix<'_, '_>, cache: &mut PrefixCache<CudaCopyEngine<'_>>, rings: &mut Vec<i32>,
    slots: &mut Vec<usize>, placement: &MimoPlacement, slot: Option<usize>) {
    if let Err(error) = cache.release(family, &placement.pages) {
        tracing::error!(%error, "releasing a MiMo sequence's pages");
    }
    rings.push(placement.ring);
    slots.extend(slot);
}

#[allow(clippy::too_many_arguments)]
fn schedule(engine: &MimoEngine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    policy: Policy, prefix: &PrefixArgs, select: SelectPlacement,
    host_config: Option<cuteafd_hostcache::config::Config>,
    media: &mut MediaAdmission<super::media::Prompt, super::media::Encoder>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>) -> Result<()> {
    let mut owners = super::serving_owners::ServingOwners::new(engine);
    let (family, cache) = prefix_cache_with(engine, prefix, max_sequences, |_| Ok(host_config))?;
    owners.prefix(family, cache);
    owners.selector(TokenSelector::new(engine.library, select, engine.cfg.vocab_size, DECODE_ROWS)?);
    schedule_inner(engine, opened, snapshot, receive, stats, max_sequences, policy, prefix, &mut owners, media, preparer)
}

#[allow(clippy::too_many_arguments)]
fn schedule_inner(engine: &MimoEngine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    policy: Policy, prefix: &PrefixArgs, owners: &mut super::serving_owners::ServingOwners<'_, '_>,
    media: &mut MediaAdmission<super::media::Prompt, super::media::Encoder>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>) -> Result<()> {
    let draft = policy.copy;
    let drafter = engine.drafter.as_ref();
    let mut free_slots: Vec<usize> = drafter.map_or(Vec::new(), |d| (0..d.slots).rev().collect());
    let mut cost = dflash_policy::step_cost(&PRO_TP6_STEP_MS, DECODE_ROWS);
    let mut skip = crate::families::glm5::dflash_policy::DraftSkip::default();
    let (family, cache, selector) = owners.parts();
    // Messages start with `<|im_start|>`: a snapshot right before one is a message boundary.
    let message_start = *QwenEncoding::from_snapshot(snapshot)?.tokens().turn_markers.first()
        .context("the chat template names no message-start token")?;
    let mut free_rings: Vec<i32> = (0..engine.rings as i32).rev().collect();
    let mut grammars = crate::shared::constraints::Compiler::with_vocab(
        &opened.library, snapshot.join("tokenizer.json"), engine.cfg.vocab_size, QwenEncoding::from_snapshot(snapshot)?.tokens().eos.clone());
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total) = (0u64, 0u64);
    // Verify steps since the last completed request, and host seconds in
    // them (engine) and in token selection + streaming.
    let mut steps = 0u64;
    let (mut verify_s, mut draft_s, mut emit_s) = (0f64, 0f64, 0f64);
    let mut seconds_per_row = 0.01f64;
    let mut prefills = policy.decode_share.queue::<Prefill<'_>>()?;
    let mut snapshot_waiter: Option<MediaReady<super::media::Prompt>> = None;
    let mut kv_waiter = cuteafd_engine::prefix::DeferredAdmission::<MediaReady<super::media::Prompt>>::default();
    let config = &opened.checkpoint.config;
    let mut failures = FailureRecipients::default();
    // Keep admitted request owners alive until a fatal cause has reached every
    // affected client, including requests waiting for prefill or KV admission.
    let result = (|| -> Result<()> { loop {
        while active.len() + prefills.len() < max_sequences {
            let busy = !active.is_empty() || !prefills.is_empty();
            if snapshot_waiter.as_ref().is_some_and(|ready| ready.job().job.events.is_closed()) {
                snapshot_waiter = None;
            }
            if snapshot_waiter.as_ref().is_some_and(|ready| prefills.iter().any(|p|
                !p.job.events.is_closed() && !probe::cold(&p.job.probe)
                && snapshot_extension(&p.keys, &ready.job().keys, prefix.prefix_cache_min_tokens))) { break; }
            let ready = if let Some(ready) = snapshot_waiter.take() { ready } else { match kv_waiter.poll(cache.pool().free(), cache.pool().release_epoch(), busy,
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
                        publish(stats, requests, generated_total, active.len(), prefills.len(), cache, media, preparer);
                        let job = if !busy && media.is_empty() {
                            match receive.blocking_recv() { Some(job) => job, None => return Ok(()) }
                        } else {
                            match receive.try_recv() {
                                Ok(job) => job,
                                Err(_) => break,
                            }
                        };
                        failures.watch(&job.events);
                        if let Err(error) = probe::validate_scoring(&job.probe, engine.full_prefill_logits()) {
                            let _ = job.events.send(Err(NativeFailure::BadRequest(format!("scoring: {error:#}"))));
                            continue;
                        }
                        if (!job.media.is_empty() && !media.encoder().available_for(false))
                            || (!job.audio.is_empty() && !media.encoder().available_for(true)) {
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
                        let (prompt, mut request_media, jobs) = match super::media::prepare(job, tokens, config,
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
            } };
            if (!ready.job().job.media.is_empty() && !media.encoder().available_for(false))
                || (!ready.job().job.audio.is_empty() && !media.encoder().available_for(true)) {
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
            if policy.snapshot_wait && cache.enabled() && !ready.cold() && !probe::cold(&ready.job().job.probe)
                && prefills.iter().any(|p| !p.job.events.is_closed() && !probe::cold(&p.job.probe)
                    && snapshot_extension(&p.keys, &ready.job().keys, prefix.prefix_cache_min_tokens)) {
                snapshot_waiter = Some(ready);
                break;
            }
            let cold = ready.cold() || probe::cold(&ready.job().job.probe);
            let capacity = (ready.job().tokens.len() + ready.job().job.max_tokens).min(engine.max_context);
            let Some(ring) = free_rings.pop() else {
                reject(&ready, "SWA rings exhausted".into());
                continue;
            };
            // Disabled neural drafts need neither a ring slot nor context updates.
            let slot = if probe::no_speculation(&ready.job().job.probe) || policy.fixed == Some(0) {
                None
            } else { free_slots.pop() };
            family.bind_drafter(ring, slot);
            cache.tick();
            let admit_started = Instant::now();
            // Lookup, fork of the retained pages and restore of the mark (byte-exact).
            let build = |pages| MimoPlacement { pages, ring, len: 0 };
            let admission = if cold { cache.admit_cold(family, ready.job().tokens.len(), capacity, build) }
                else { cache.admit_media(family, ready.job().keys.tokens(), ready.job().keys.spans(), capacity, true, build) };
            let admitted = match admission {
                Ok(admitted) => admitted,
                Err(error) => {
                    free_rings.push(ring);
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
                    release(family, cache, &mut free_rings, &mut free_slots, &admitted.placement, slot);
                    let events = waiter.job.job.events.clone();
                    if let Err((_, error)) = media.enqueue(waiter) {
                        let _ = events.send(Err(super::media::failure(error)));
                    }
                    continue;
                }
            };
            let (super::media::Prompt { job, tokens, keys }, request_media) = ready.into_parts();
            probe::admitted(&job.probe, "mimo_v2", &tokens, resume);
            let _ = job.events.send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: resume },
            }));
            if let Some(from) = probe::scoring(&job.probe) {
                // Teacher-forced scoring: every row's logits, no generation, nothing retained.
                let mut placement = admitted.placement;
                let scored = probe::score(engine.library, &job.probe, &tokens, from,
                    if engine.full_prefill_logits() { engine.prefill_rows } else { engine.prefill_capacity() },
                    DECODE_ROWS, probe::verify_rows(&job.probe), engine.full_prefill_logits(), &mut placement,
                    |placement, chunk, rows| Ok(engine.prefill_media_device(placement, chunk, rows > 1, None, None, Some(&request_media))?
                        .map(probe::ScoreLogits::Device)),
                    |placement, chunk| engine.verify_media_device(&mut [(placement, chunk.len())], chunk, None, Some(&request_media))?
                        .context("scoring needs every layer"));
                match scored {
                    Ok(_) => { let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })); }
                    Err(error) => {
                        if let Some(probe) = &job.probe { probe.fail(format!("scoring: {error:#}")); }
                        let _ = job.events.send(Err(NativeFailure::Worker(format!("scoring: {error:#}"))));
                        if engine.is_terminal() { return Err(error); }
                    }
                }
                release(family, cache, &mut free_rings, &mut free_slots, &placement, slot);
                continue;
            }
            let ticket = console::admit(tokens.len(), resume, job.max_tokens, constraint.is_some(), job.media.len(),
                admit_started);
            if let Some(source) = admitted.source {
                tracing::info!(tokens = tokens.len(), resume, kind = ?source.kind, frontier = source.frontier,
                    host = source.host, partial = source.partial, "prefix cache hit");
            }
            let boundaries = cuteafd_engine::prefix::message_boundaries(&tokens, message_start);
            // A cold probe keeps the same chunk plan (identical numerics); it only skips the captures.
            let plan = if cache.enabled() {
                cuteafd_engine::prefix::plan_media_points(resume, tokens.len(), engine.prefill_capacity(), &boundaries,
                    family.capture_reach(), prefix.prefix_cache_min_tokens, prefix.points(), keys.spans())
            } else {
                cuteafd_engine::prefix::plan_points(resume, tokens.len(), engine.prefill_capacity(), &[], 0, 0,
                    PointPolicy { gap: 0, boundaries: 0, per_request: 0 })
            };
            // A whole-prompt hit brings its first token's logits: nothing to prefill.
            let logits = admitted.after.and_then(|after| after.logits).map(|logits| logits.to_vec());
            let plan = if let Some(probe) = job.probe.as_ref().filter(|p| !p.spec.cold_steps.is_empty()) {
                // Rebuild from zero with the source's exact prefill/decode boundaries.
                cuteafd_engine::prefix::PointPlan {
                    chunks: probe.spec.cold_steps.iter().map(|step| step.end).collect(), points: Vec::new(),
                }
            } else if logits.is_some() { cuteafd_engine::prefix::PointPlan::default() } else { plan };
            prefills.push(Prefill { job, constraint, tokens, keys, media: request_media, done: resume, resume, plan, chunks: 0, paired_chunks: 0, cancelled: false,
                placement: admitted.placement, capacity, slot, logits, first: None, started: Instant::now(), busy: 0.0,
                phases: [0.0; 2], ticket });
        }
        if prefills.due(!active.is_empty()) {
            if !active.is_empty() {
                if let Some(target) = policy.chunk_s {
                    for p in prefills.iter_mut() {
                        if p.job.probe.is_none() {
                            split_timed_chunk(&mut p.plan, p.chunks, p.done,
                                timed_chunk_rows(target, seconds_per_row, p.done, engine.prefill_capacity()));
                        }
                    }
                }
            }
            // One chunk of each waiting prompt (whole prompts with --decode-share 0).
            let row_cost = seconds_per_row;
            let round_target = policy.chunk_s.filter(|_| !active.is_empty());
            let finished = prefills.round_pairs(|a, b| {
                let rows = [a, b].map(|p| p.plan.chunks.get(p.chunks).copied()
                    .unwrap_or(p.tokens.len()).saturating_sub(p.done));
                let replay = [&*a, &*b].iter().any(|p| p.job.probe.as_ref().is_some_and(|probe| !probe.spec.cold_steps.is_empty()));
                !replay && round_target.is_none_or(|target| rows.iter().sum::<usize>() as f64 * row_cost <= target)
                    && engine.can_prefill_pair(rows) && !a.job.events.is_closed() && !b.job.events.is_closed()
            }, |batch| {
                let rows: usize = batch.iter().map(|p| p.plan.chunks.get(p.chunks).copied()
                    .unwrap_or(p.tokens.len()).saturating_sub(p.done)).sum();
                let clock = Instant::now();
                let outcome = prefill_batch(engine, family, cache, selector, batch);
                if rows > 0 { seconds_per_row = (clock.elapsed().as_secs_f64() / rows as f64).max(seconds_per_row * 0.9); }
                outcome
            });
            if engine.is_terminal() {
                let mut primary = None;
                let mut affected = Vec::new();
                for (p, prefilled) in finished {
                    affected.push(p);
                    if let Err(error) = prefilled {
                        if primary.is_none() { primary = Some(error); }
                    }
                }
                let error = primary.unwrap_or_else(|| engine.terminal_error());
                for p in affected {
                    let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                }
                return Err(error);
            }
            for (mut p, prefilled) in finished {
                let (slot, placement, resume) = (p.slot, p.placement.clone(), p.resume);
                if let Err(error) = &prefilled {
                    if p.cancelled {
                        p.ticket.cancel();
                        // The client left during the prefill: keep what it computed for a retry.
                        if placement.len > resume && !probe::cold(&p.job.probe) {
                            if let Err(error) = cache.park_media(family, &p.keys.tokens()[..placement.len], p.keys.spans(), &placement) {
                                tracing::warn!("parking a cancelled prefill: {error:#}");
                            }
                        }
                    } else {
                        tracing::warn!("prefill failed: {error:#}");
                        let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    }
                    release(family, cache, &mut free_rings, &mut free_slots, &placement, slot);
                    continue;
                }
                if p.first.is_none() && p.logits.is_none() {
                    tracing::warn!("prefill produced no logits");
                    release(family, cache, &mut free_rings, &mut free_slots, &placement, slot);
                    continue;
                }
                let logits = p.logits.clone();
                if resume < p.tokens.len() {
                    tracing::info!(tokens = p.tokens.len(), cached = resume,
                        paired_chunks = p.paired_chunks,
                        elapsed_ms = p.started.elapsed().as_millis() as u64, busy_ms = (1e3 * p.busy) as u64,
                        tok_s = (p.tokens.len() - resume) as f64 / p.busy, gpu_wait_ms = (1e3 * p.phases[0]) as u64,
                        experts_ms = (1e3 * p.phases[1]) as u64, "prefill");
                }
                // The prompt snapshot, taken once the first token is out (it only enqueues copies).
                let prompt = (resume < p.tokens.len() && !probe::cold(&p.job.probe)).then(|| p.keys.tokens().to_vec());
                let spans = p.keys.spans().to_vec();
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &MimoPlacement| {
                    if let Some(prompt) = &prompt {
                        let after = logits.as_deref().map_or_else(After::default, |l| After::from_logits(l, true));
                        if let Err(error) = cache.capture_media(family, SnapshotKind::Prompt, prompt, &spans, placement, after) {
                            tracing::warn!("prompt snapshot not retained: {error:#}");
                        }
                    }
                };
                engine.mtp_reset(p.placement.ring as usize, p.placement.len);
                // Construction can fail after moving the request into its active
                // state. Keep its event sender until that error is reported.
                let admission_events = p.job.events.clone();
                let job_events = p.job.events.clone();
                let draft_from = family.draft_from(p.placement.ring);
                let admitted = (|| -> Result<Active<'_>> {
                    let mut request = Active {
                        slot,
                        drafts: DraftHistory::default(),
                        counts: [0; 6],
                        history: p.tokens,
                        copy_index: copy::CopyIndex::default(),
                        keyed_history: p.keys.tokens().to_vec(),
                        media: p.media,
                        draft_limit: draft,
                        draft_pause: 0,
                        decoder: cuteafd_loader::streaming_token_decoder(snapshot, false)?,
                        job: p.job, constraint: p.constraint, placement: p.placement, draft_from, turn: None,
                        capacity: p.capacity, next: 0, generated: 0, buffered: 0, started: Instant::now(),
                        ticket: p.ticket,
                    };
                    request.next = match p.first {
                        Some(token) => token,
                        None => {
                            let logits = logits.as_deref().context("no first-token logits")?;
                            if probe::wants_first(&request.job.probe) {
                                probe::host_row(&request.job.probe, request.history.len(), logits);
                            }
                            request.select_host(logits)?
                        }
                    };
                    Ok(request)
                })();
                match admitted {
                    Ok(mut request) => {
                        let token = request.next;
                        let emitted = request.emit(token);
                        if let Err(error) = &emitted {
                            let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        }
                        request.ticket.first(token);
                        retain_prompt(cache, &request.placement);
                        match &emitted {
                            Ok(false) => active.push(request),
                            // Finished at its first token: its turn is its prompt snapshot.
                            Ok(true) | Err(_) => {
                                if let Err(error) = &emitted {
                                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                                }
                                failures.finished(&request.job.events);
                                request.ticket.done(request.generated);
                                release(family, cache, &mut free_rings, &mut free_slots, &request.placement,
                                    request.slot)
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!("admission failed: {error:#}");
                        let _ = job_events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        let _ = admission_events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        if engine.is_terminal() { return Err(error); }
                        retain_prompt(cache, &placement);
                        release(family, cache, &mut free_rings, &mut free_slots, &placement, slot);
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
        let (draft0, emit0) = (draft_s, emit_s);
        let engine_before = tally.live().then(|| *engine.profile.borrow());
        // Rows each sequence may add after its next token.
        let room = (DECODE_ROWS / active.len()).max(1) - 1;
        let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else { room.min(a.job.max_tokens - a.generated - 1)
            .min(a.capacity - a.placement.len - 1) }).collect();
        // Greedy copy windows replace, rather than accompany, this sequence's neural draft.
        let copies: Vec<Vec<u32>> = active.iter_mut().enumerate().map(|(i, a)| {
            if a.draft_pause > 0 {
                a.draft_pause -= 1;
                if a.draft_pause == 0 { a.draft_limit = draft.min(1); }
            }
            if policy.indexed_copy && a.job.sampling.is_greedy() {
                a.copy_index.propose(&a.history, limits[i].min(draft))
            } else { Vec::new() }
        }).collect();
        // DFlash drafts after every next token (sequences with a ring slot),
        // then the adaptive plan's counts.
        let drafted: Vec<Option<super::dflash::Draft>> = match drafter {
            Some(drafter) if skip.drafts() && active.iter().enumerate().any(|(i, a)| a.slot.is_some() && limits[i] > 0 && copies[i].is_empty()) => {
                let seqs: Vec<(usize, DraftSeq)> = active.iter().enumerate()
                    .filter_map(|(i, a)| a.slot.filter(|_| limits[i] > 0 && copies[i].is_empty()).map(|slot| (i, DraftSeq { slot, anchor: a.next, position: a.placement.len,
                        valid_from: a.draft_from })))
                    .collect();
                for &(i, _) in &seqs {
                    active[i].counts[5] += 1;
                }
                let timer = Instant::now();
                let drafts = engine.submit(|| drafter.draft(&seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), &engine.embedding,
                    engine.head()));
                let ms = timer.elapsed().as_secs_f64() * 1e3;
                cost.observe_draft(ms);
                draft_s += ms / 1e3;
                let mut out = vec![None; active.len()];
                match drafts {
                    Ok(drafts) => {
                        for ((i, _), draft) in seqs.into_iter().zip(drafts) {
                            out[i] = Some(draft);
                        }
                    }
                    Err(error) => {
                        tracing::warn!("DFlash draft failed: {error:#}");
                        if engine.is_terminal() {
                            return Err(error);
                        }
                    }
                }
                out
            }
            // Native MTP drafts (MiMo V2 Flash; V2.6 Pro prefers DFlash).
            None if skip.drafts() && engine.mtp.is_some() && policy.fixed != Some(0)
                && limits.iter().any(|&limit| limit > 0) => {
                let indices: Vec<usize> = (0..active.len()).filter(|&i| {
                    if limits[i] == 0 || !copies[i].is_empty() { return false; }
                    let a = &active[i];
                    let start = a.placement.len.saturating_sub(engine.cfg.window + engine.mtp.as_ref().unwrap().stages.len());
                    if !a.media.ready(start, a.history.len()) {
                        // Restored image effects are in target state, not host embeddings.
                        // Keep MTP cold until its reconstruction window is past those rows.
                        engine.mtp_reset(a.placement.ring as usize, a.placement.len);
                        return false;
                    }
                    true
                }).collect();
                for &i in &indices {
                    active[i].counts[5] += 1;
                }
                let seqs: Vec<MtpSeq<'_>> = indices.iter().map(|&i| {
                    let a = &active[i];
                    MtpSeq { ring: a.placement.ring as usize, len: a.placement.len, tokens: &a.history, media: Some(&a.media) }
                }).collect();
                let stages = engine.mtp.as_ref().map_or(0, |m| m.stages.len());
                let timer = Instant::now();
                let drafts = engine.mtp_draft(&seqs, stages);
                let ms = timer.elapsed().as_secs_f64() * 1e3;
                cost.observe_draft(ms);
                draft_s += ms / 1e3;
                match drafts {
                    Ok(drafts) => {
                        let mut out = vec![None; active.len()];
                        for (i, tokens) in indices.into_iter().zip(drafts) {
                            let features = vec![[0.0, 1.0, 0.0, 0.0]; tokens.len()];
                            out[i] = Some(super::dflash::Draft { tokens, features });
                        }
                        out
                    },
                    Err(error) => {
                        tracing::warn!("MTP draft failed: {error:#}");
                        if engine.is_terminal() {
                            return Err(error);
                        }
                        vec![None; active.len()]
                    }
                }
            }
            _ => vec![None; active.len()],
        };
        let plan_timer = Instant::now();
        let planned = plan_drafts(&active, &drafted, &limits, policy.fixed, &cost);
        skip.after(drafted.iter().any(Option::is_some) && policy.fixed.is_none(), planned.iter().all(|&n| n == 0));
        // Indexed copies replace neural drafts; the legacy path only extends
        // an agreeing neural proposal. Both verify against the same target.
        let mut used_copy = vec![false; active.len()];
        let sequences: Vec<Vec<u32>> = active.iter_mut().enumerate().map(|(i, a)| {
            let dflash: &[u32] = drafted[i].as_ref().map_or(&[], |d| &d.tokens[..planned[i]]);
            let legacy = if policy.indexed_copy { Vec::new() } else {
                let copy = crate::families::glm5_flash::serve::copy_drafts(&a.history, limits[i].min(a.draft_limit));
                let full = drafted[i].as_ref().map_or(&[][..], |d| &d.tokens[..]);
                let agrees = copy.iter().zip(full).take_while(|(c, d)| c == d).count() >= dflash.len();
                if copy.len() > dflash.len() && agrees { copy } else { Vec::new() }
            };
            used_copy[i] = !copies[i].is_empty() || !legacy.is_empty();
            let draft = if !copies[i].is_empty() { &copies[i][..] }
                else if !legacy.is_empty() { &legacy[..] } else { dflash };
            let mut rows: Vec<u32> = std::iter::once(a.next).chain(draft.iter().copied()).collect();
            // Drafts the grammar rejects could never be kept: verify none of them.
            if let Some(state) = a.constraint.as_ref() {
                state.truncate_proposal(&mut rows)?;
            }
            Ok(rows)
        }).collect::<Result<_>>()?;
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let plan_us = console::us(plan_timer);
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        steps += 1;
        let timer = Instant::now();
        // A sequence whose grammar fails is failed alone after the step; an
        // error inside `submit` would end the engine.
        let mut poisoned: Vec<Option<String>> = vec![None; active.len()];
        let step = engine.submit(|| {
            let mut rows: Vec<(&mut MimoPlacement, usize)> = active.iter_mut().zip(&sequences)
                .map(|(a, s)| (&mut a.placement, s.len())).collect();
            let logits = engine.verify_device(&mut rows, &tokens, None)?.context("decode needs every layer")?;
            drop(rows);
            // Each row draws at the position after it, masked along its sequence's drafts.
            let mut batch = SelectBatch::default();
            for (i, ((a, rows), &start)) in active.iter().zip(&sequences).zip(&starts).enumerate() {
                poisoned[i] = batch.push_sequence_isolated(a.job.sampling, a.constraint.as_ref(), rows, start as u64 + 1);
            }
            Ok((selector.select(&logits, &batch)?, logits))
        });
        let step_s = timer.elapsed().as_secs_f64();
        verify_s += step_s;
        cost.observe_verify(Shape::plain(tokens.len(), sequences.len()), step_s * 1e3);
        let timer = Instant::now();
        let (selected, logits) = match step {
            Ok(step) => step,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                if engine.is_terminal() {
                    return Err(error);
                }
                for request in active.drain(..) {
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(family, cache, &mut free_rings, &mut free_slots, &request.placement, request.slot);
                }
                continue;
            }
        };
        let mut offset = 0;
        let mut context = Vec::new();
        let before: Vec<usize> = active.iter().map(|a| a.history.len()).collect();
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).enumerate()
            .map(|(i, ((request, rows), &start))| {
            if let Some(error) = poisoned[i].take() {
                let _ = request.job.events.send(Err(NativeFailure::Worker(error)));
                offset += rows.len();
                return true;
            }
            let mut finished = false;
            for j in 0..rows.len() {
                // Rows 0..=j are committed; the token row j produces is next.
                request.placement.len = start + j + 1;
                probe::decode_row(&engine.library, &request.job.probe, &logits, offset + j, request.generated,
                    request.history.len());
                match take(request.constraint.as_mut(), &selected[offset + j]).and_then(|t| Ok((t, request.emit(t)?))) {
                    Ok((token, done)) => {
                        finished = done;
                        if done && cache.enabled() && !probe::cold(&request.job.probe) {
                            // A normal finish (the client took the last chunk): the row that
                            // produced the last token follows the turn snapshot.
                            request.turn = logits.row_host(engine.library, offset + j).ok();
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
            if let Some(slot) = request.slot.filter(|_| !finished || (prefix.mimo_prefix_draft && request.turn.is_some())) {
                context.extend((0..committed).map(|r| ContextRow { tap_row: offset + r, slot, position: start + r }));
            }
            offset += rows.len();
            let (drafted, accepted) = (rows.len() - 1, committed - 1);
            if prefix.mimo_prefix_draft && request.counts[0] == 0 {
                tracing::info!(position = start, valid_from = request.draft_from, drafted, accepted,
                    copy = used_copy[i], neural_calls = request.counts[5], "first draft verification");
            }
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
            if used_copy[i] || (drafted > 0 && drafter.is_none() && engine.mtp.is_none()) {
                if drafted > 0 && accepted == 0 {
                    request.draft_limit /= 2;
                    if request.draft_limit == 0 {
                        request.draft_pause = 8;
                    }
                } else if drafted > 0 && accepted == drafted {
                    request.draft_limit = (request.draft_limit * 2).clamp(draft.min(1), draft);
                }
            }
            finished
        }).collect();
        for (request, &done) in active.iter().zip(&finished) {
            if done { failures.finished(&request.job.events); }
        }
        if let Some(drafter) = drafter.filter(|_| !context.is_empty()) {
            engine.submit(|| drafter.update(&context))?;
        }
        emit_s += timer.elapsed().as_secs_f64();
        for (i, request) in active.iter().enumerate() {
            let proposal = if used_copy[i] { &sequences[i][1..] } else { drafted[i].as_ref().map_or(&[][..], |d| &d.tokens) };
            tally.member(&request.ticket, proposal, sequences[i].len() - 1, &request.history[before[i]..],
                request.constraint.is_some(), finished[i]);
        }
        tally.end(|| {
            let engine_now = *engine.profile.borrow();
            let gpu = |i: usize| engine_before.map_or(f64::NAN, |before| 1e6 * (engine_now[i] - before[i]));
            vec![("draft", 1e6 * (draft_s - draft0)), ("plan", plan_us), ("verify", 1e6 * step_s),
                ("emit", 1e6 * (emit_s - emit0)), ("gpu", gpu(0)), ("experts", gpu(1))]
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
            let [cycles, dflash, dflash_ok, copy, copy_ok, draft_calls] = request.counts;
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps, verify_s, draft_s, emit_s, gpu_wait_s = phases[0],
                experts_s = phases[1], cycles, dflash, dflash_ok, copy, copy_ok, draft_calls, late_graphs = engine.late_captures(),
                "request complete");
            (steps, verify_s, draft_s, emit_s) = (0, 0.0, 0.0, 0.0);
            if let Some(row) = request.turn.as_ref().filter(|_| !probe::cold(&request.job.probe)) {
                // The conversation so far: every committed row (the last token is not in it).
                let rows = &request.keyed_history[..request.placement.len];
                if let Err(error) = cache.capture_media(family, SnapshotKind::Turn, rows, request.media.spans(), &request.placement,
                    After::from_logits(row, true)) {
                    tracing::warn!("turn snapshot not retained: {error:#}");
                }
            }
            release(family, cache, &mut free_rings, &mut free_slots, &request.placement, request.slot);
        }
        cache.tick();
        publish(stats, requests, generated_total, active.len(), prefills.len(), cache, media, preparer);
        console::gauges(|| console::Gauges::prefix_cache(cache, active.len(), prefills.len(),
            receive.len() + kv_waiter.len() + usize::from(snapshot_waiter.is_some())));
        prefills.stepped(cycle.elapsed().as_secs_f64());
    } })();
    if let Err(error) = &result {
        failures.fail_and_close(error, receive);
    }
    result
}

/// Serving statistics for `/v1/stats`.
fn publish(stats: &Mutex<serde_json::Value>, requests: u64, generated: u64, active: usize, prefilling: usize,
    cache: &PrefixCache<CudaCopyEngine<'_>>, media: &MediaAdmission<super::media::Prompt, super::media::Encoder>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>) {
    if let Ok(mut stats) = stats.lock() {
        *stats = serde_json::json!({"requests": requests, "generated_tokens": generated, "active": active,
            "prefilling": prefilling, "prefix_cache": cache.stats(),
            "media": media.stats(cache.stats().media_key_collisions, preparer.map_or(0, |p| p.memo_hits()))});
    }
}

#[cfg(test)]
mod port_tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        serve: ServeArgs,
    }

    #[test]
    fn port_knobs_default_off_and_sixteen_slots_are_configurable() {
        let default = Cli::parse_from(["serve", "--snapshot", "/model", "--native-lib", "/lib"]);
        let args = default.serve;
        assert_eq!(args.max_sequences, 4);
        assert_eq!(args.http_queue_depth, None);
        assert_eq!(args.http_queue_wait_ms, 25_000);
        assert!(!args.mimo_copy_windows && !args.prefix.mimo_prefix_draft && !args.mimo_snapshot_wait);
        assert!(!args.mimo_host_cache && args.prefill_chunk_s.is_none());
        let enabled = Cli::parse_from(["serve", "--snapshot", "/model", "--native-lib", "/lib",
            "--max-sequences", "16", "--http-queue-depth", "32", "--http-queue-wait-ms", "1234",
            "--mimo-copy-windows", "--mimo-prefix-draft", "--mimo-snapshot-wait",
            "--mimo-host-cache", "--host-cache-bytes", "0", "--prefill-chunk-s", "4"]);
        assert_eq!((enabled.serve.max_sequences, enabled.serve.http_queue_depth), (16, Some(32)));
        assert_eq!(enabled.serve.prefix.host_cache_bytes, crate::shared::prefix::HostBudget::Bytes(0));
        assert_eq!(enabled.serve.prefill_chunk_s, Some(4.0));
    }

    #[test]
    fn snapshot_extensions_require_retained_length_and_full_media_identity() {
        use cuteafd_engine::media::{ImageKey, MediaKeys, MediaSpan};
        let a = MediaSpan { start: 1, len: 1, key: ImageKey([0; 32]).into() };
        // These distinct SHA keys fold to the same radix hint.
        let mut collision = [0u8; 32];
        collision[0] = 1;
        collision[8..16].copy_from_slice(&1u64.rotate_right(13).to_le_bytes());
        let b = MediaSpan { key: ImageKey(collision).into(), ..a };
        let saved = MediaKeys::new(&[1, 2, 3], 100, &[a]).unwrap();
        let extension = MediaKeys::new(&[1, 2, 3, 4], 100, &[a]).unwrap();
        assert!(snapshot_extension(&saved, &extension, 3));
        assert!(!snapshot_extension(&saved, &extension, 4));
        let other = MediaKeys::new(&[1, 2, 3, 4], 100, &[b]).unwrap();
        assert_eq!(extension.tokens(), other.tokens());
        assert!(!snapshot_extension(&saved, &other, 3));
        let shorter = MediaKeys::new(&[1, 2], 100, &[a]).unwrap();
        assert!(!snapshot_extension(&saved, &shorter, 1));
    }

    #[test]
    fn timed_chunk_limits_long_positions_and_preserves_snapshot_ends() {
        assert_eq!(timed_chunk_rows(4.0, 0.001, 0, 4096), 1024);
        assert_eq!(timed_chunk_rows(4.0, 0.001, 65_536, 4096), 256);
        assert_eq!(timed_chunk_rows(4.0, 10.0, 65_536, 4096), 1);
        let mut plan = cuteafd_engine::prefix::PointPlan { chunks: vec![4096, 8192], points: vec![(0, 4096), (1, 8192)] };
        split_timed_chunk(&mut plan, 0, 0, 256);
        assert_eq!(plan.chunks, [256, 4096, 8192]);
        assert_eq!(plan.points, [(1, 4096), (2, 8192)]);
        split_timed_chunk(&mut plan, 1, 256, 3840);
        assert_eq!(plan.chunks, [256, 4096, 8192]);
    }
}

// Hugh Madden's mimo26f-afd v1.3.0 api.rs/scheduler.rs snapshot coalescing,
// with CuteAFD's full media identity check rather than a hashed-token match alone.
fn snapshot_extension(saved: &cuteafd_engine::media::MediaKeys,
    request: &cuteafd_engine::media::MediaKeys, min_tokens: usize) -> bool {
    saved.tokens().len() >= min_tokens && request.tokens().starts_with(saved.tokens())
        && cuteafd_engine::media::verify_media(saved.tokens().len(), saved.spans(), request.spans())
}

fn timed_chunk_rows(target: f64, seconds_per_row: f64, position: usize, capacity: usize) -> usize {
    // Full-attention cost grows with position; keep a safety margin for the next chunk.
    let rows = (0.75 * target / seconds_per_row.max(1e-6)) as usize;
    rows.max(1).min(capacity).min(if position >= 65_536 { 256 } else { 1024 })
}

fn split_timed_chunk(plan: &mut cuteafd_engine::prefix::PointPlan, chunk: usize, start: usize, rows: usize) {
    if let Some(&end) = plan.chunks.get(chunk) {
        if start + rows < end {
            plan.chunks.insert(chunk, start + rows);
            for (index, _) in &mut plan.points {
                if *index >= chunk { *index += 1; }
            }
        }
    }
}

/// DFlash draft counts: `fixed` (within each limit), or the adaptive plan
/// (glmrt v9's schedule, `dflash_policy::plan`) over the drafting sequences
/// with each one's history of conditional acceptance (the GLM selector
/// calibration does not apply to this drafter), priced with the sequences
/// that do not draft.
fn plan_drafts(active: &[Active<'_>], drafted: &[Option<super::dflash::Draft>], limits: &[usize],
    fixed: Option<usize>, cost: &CycleCost) -> Vec<usize> {
    let indices: Vec<usize> = (0..active.len()).filter(|&i| drafted[i].is_some()).collect();
    let mut counts = vec![0; active.len()];
    let width = |i: usize| drafted[i].as_ref().map_or(0, |d| d.tokens.len());
    if let Some(fixed) = fixed {
        for &i in &indices {
            counts[i] = fixed.min(limits[i]).min(width(i));
        }
        return counts;
    }
    if indices.is_empty() {
        return counts;
    }
    let groups: Vec<Group<'_>> = indices.iter().map(|&i| Group {
        history: &active[i].drafts,
        confidence: active[i].drafts.conditional(width(i)),
        room: limits[i],
        members: 1,
        informed: false,
    }).collect();
    let others = active.len() - indices.len();
    for (&i, n) in indices.iter().zip(dflash_policy::plan(&groups, (others, others), cost)) {
        counts[i] = n;
    }
    counts
}

#[allow(clippy::too_many_arguments)]
fn prefill_one(engine: &MimoEngine<'_>, family: &MimoPrefix<'_, '_>,
    cache: &mut PrefixCache<CudaCopyEngine<'_>>, selector: &mut TokenSelector<'_>, p: &mut Prefill<'_>)
    -> Result<Chunk> {
    if p.done == p.tokens.len() {
        return Ok(Chunk::Done);
    }
    if p.job.events.is_closed() {
        p.cancelled = true;
        anyhow::bail!("client went away");
    }
    let timer = Instant::now();
    let end = p.plan.chunks.get(p.chunks).copied().unwrap_or(p.tokens.len());
    let count = end - p.done;
    let retain = cache.enabled() && !probe::cold(&p.job.probe);
    let (result, phases) = isolated_phases(&engine.profile, || engine.submit(|| -> Result<()> {
        let start = p.placement.len;
        let decode = p.job.probe.as_ref().and_then(|probe| probe.spec.cold_steps.get(p.chunks))
            .is_some_and(|step| step.decode);
        let logits = if decode {
            engine.verify_media_device(&mut [(&mut p.placement, count)], &p.tokens[p.done..end], None, Some(&p.media))?
        } else {
            engine.prefill_media_device(&mut p.placement, &p.tokens[p.done..end], false, None, None, Some(&p.media))?
        };
        consume_prefill_chunk(engine, selector, p, start, end, logits, retain, None)?;
        Ok(())
    }));
    p.done += count;
    add_phases(&mut p.phases, phases);
    p.busy += timer.elapsed().as_secs_f64();
    p.ticket.prefill(count, p.chunks, p.plan.chunks.len(), timer);
    result?;
    complete_prefill_chunk(family, cache, p)
}

fn complete_prefill_chunk(family: &MimoPrefix<'_, '_>, cache: &mut PrefixCache<CudaCopyEngine<'_>>,
    p: &mut Prefill<'_>) -> Result<Chunk> {
    // Intermediate snapshot points this chunk reaches (off unless configured).
    for &(_, point) in p.plan.points.iter().filter(|&&(chunk, _)| chunk == p.chunks && !probe::cold(&p.job.probe)) {
        if let Err(error) = cache.capture_media(family, SnapshotKind::Prompt, &p.keys.tokens()[..point], p.keys.spans(), &p.placement,
            After::default()) {
            tracing::warn!("snapshot point {point} not retained: {error:#}");
        }
    }
    p.chunks += 1;
    Ok(if p.done == p.tokens.len() { Chunk::Done } else { Chunk::More })
}

#[allow(clippy::too_many_arguments)]
fn consume_prefill_chunk(engine: &MimoEngine<'_>, selector: &mut TokenSelector<'_>, p: &mut Prefill<'_>,
    start: usize, end: usize, logits: Option<crate::shared::token_io::DeviceLogits>, retain: bool,
    lane: Option<usize>) -> Result<()> {
    if end == p.tokens.len() {
        let logits = logits.context("prefill produced no logits")?;
        let mut batch = SelectBatch::default();
        batch.push_next(p.job.sampling, p.constraint.as_mut(), p.placement.len as u64)?;
        let selected = selector.select(&logits, &batch)?;
        p.first = Some(take(p.constraint.as_mut(), &selected[0])?);
        p.logits = if retain { Some(logits.row_host(engine.library, 0)?) } else { None };
        if probe::wants_first(&p.job.probe) {
            probe::device_rows(engine.library, &p.job.probe, &logits, 0, 1, p.tokens.len())?;
        }
    }
    if let (Some(drafter), Some(slot)) = (engine.drafter.as_ref(), p.slot) {
        let count = end - p.done;
        let n = count.min(super::dflash::TAP_ROWS);
        let rows = (0..n).map(|r| ContextRow { tap_row: r, slot, position: start + count - n + r })
            .collect::<Vec<_>>();
        match lane {
            Some(lane) => drafter.update_lane(lane, &rows)?,
            None => drafter.update(&rows)?,
        }
    }
    Ok(())
}

fn prefill_batch(engine: &MimoEngine<'_>, family: &MimoPrefix<'_, '_>,
    cache: &mut PrefixCache<CudaCopyEngine<'_>>, selector: &mut TokenSelector<'_>, prompts: &mut [Prefill<'_>])
    -> [Result<Chunk>; 2] {
    // Recheck cancellation immediately before submission. A cancelled neighbour
    // uses the existing independent cleanup; it never cancels the healthy one.
    if prompts.len() != 2 || prompts.iter().any(|p| p.job.events.is_closed() || p.done == p.tokens.len()) {
        let first = prefill_one(engine, family, cache, selector, &mut prompts[0]);
        let second = if prompts.len() == 2 {
            prefill_one(engine, family, cache, selector, &mut prompts[1])
        } else { Ok(Chunk::Done) };
        return [first, second];
    }
    let timer = Instant::now();
    let (a, b) = prompts.split_at_mut(1);
    let mut pair = [&mut a[0], &mut b[0]];
    let ends = pair.each_ref().map(|p| p.plan.chunks.get(p.chunks).copied().unwrap_or(p.tokens.len()));
    let starts = pair.each_ref().map(|p| p.placement.len);
    let counts = [ends[0] - pair[0].done, ends[1] - pair[1].done];
    let (result, phases) = isolated_phases(&engine.profile, || engine.submit(|| -> Result<()> {
        let [a, b] = &mut pair;
        let logits = engine.prefill_pair_media_device([
            (&mut a.placement, &a.tokens[a.done..ends[0]]),
            (&mut b.placement, &b.tokens[b.done..ends[1]]),
        ], [Some(&a.media), Some(&b.media)])?;
        // Each head/tap bank remains live until its own consumer has queued.
        // Context updates reuse scratch in stream order, with explicit drains.
        for (i, logits) in logits.into_iter().enumerate() {
            let retain = cache.enabled() && !probe::cold(&pair[i].job.probe);
            consume_prefill_chunk(engine, selector, pair[i], starts[i], ends[i], logits, retain, Some(i))?;
        }
        Ok(())
    }));
    // The pair's elapsed host interval includes transport/compute waits. Split
    // it equally only for additive bookkeeping, not as kernel attribution.
    let busy = timer.elapsed().as_secs_f64() / 2.0;
    for (i, p) in pair.into_iter().enumerate() {
        p.done += counts[i];
        p.paired_chunks += 1;
        p.busy += busy;
        add_phases(&mut p.phases, phases.map(|seconds| seconds / 2.0));
        p.ticket.prefill(counts[i], p.chunks, p.plan.chunks.len(), timer);
    }
    match result {
        Ok(()) => [complete_prefill_chunk(family, cache, &mut prompts[0]),
            complete_prefill_chunk(family, cache, &mut prompts[1])],
        Err(error) => [Err(error), Err(engine.terminal_error())],
    }
}
