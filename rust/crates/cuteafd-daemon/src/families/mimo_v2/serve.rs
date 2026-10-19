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
use crate::shared::prefix::CudaCopyEngine;
use cuteafd_engine::media::{EmbeddingCache, MediaAdmission, MediaPoll, MediaReady, MediaWaiter, RequestMedia, MediaKeys};
use cuteafd_engine::prefix::{After, MarkArena, PointPolicy, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
use crate::shared::draft_policy::DraftHistory;
use crate::shared::draft::{resource, clock::RoundClock, evidence::{Censor, DraftSource}};
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
    /// Bounded pending API requests (default 16).
    #[arg(long, env = "CUTEAFD_HTTP_QUEUE_DEPTH")]
    pub http_queue_depth: Option<usize>,
    /// Time a caller waits for a pending-queue permit.
    #[arg(long, env = "CUTEAFD_HTTP_QUEUE_WAIT_MS", default_value_t = 25_000)]
    pub http_queue_wait_ms: u64,
    // Keep these opt-in defaults; promote-or-delete decision: https://github.com/tpurtell/cuteafd/issues/403.
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
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
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

fn initial_reservation(prompt: usize, max_tokens: usize, context: usize) -> usize {
    prompt.saturating_add(max_tokens.clamp(1024, 8192)).saturating_add(64).min(context)
}

#[cfg(test)]
mod reservation_tests {
    use super::initial_reservation;

    #[test]
    fn initial_lease_clamps_output_and_context_without_overflow() {
        assert_eq!(initial_reservation(100, 1, 1 << 20), 1188);
        assert_eq!(initial_reservation(100, 4000, 1 << 20), 4164);
        assert_eq!(initial_reservation(100, 100_000, 1 << 20), 8356);
        assert_eq!(initial_reservation((1 << 20) - 100, 100_000, 1 << 20), 1 << 20);
        assert_eq!(initial_reservation(usize::MAX, usize::MAX, 1 << 20), 1 << 20);
    }
}

pub(crate) async fn run_serve(mut args: ServeArgs) -> Result<()> {
    anyhow::ensure!(args.prefill_chunk_s.is_none_or(|s| s.is_finite() && s > 0.0 && s <= 5.0),
        "--prefill-chunk-s must be finite and in (0, 5]");
    anyhow::ensure!(args.prefill_chunk_s.is_none() || args.decode_share.decode_share > 0.0,
        "--prefill-chunk-s requires a positive --decode-share");
    let api = args.api.load()?;
    let snapshot: PathBuf = args.engine.snapshot.clone();
    let effective_sequences = args.max_sequences.clamp(1, DECODE_ROWS);
    if effective_sequences != args.max_sequences {
        tracing::warn!(requested = args.max_sequences, effective = effective_sequences, "clamping MiMo decoding sequences");
        args.max_sequences = effective_sequences;
    }
    // MiMo's template and tool calls follow Qwen3-Coder's XML (`<tool_call>
    // <function=NAME><parameter=KEY>VALUE</parameter>`), its reasoning `<think>`.
    let encoding = QwenEncoding::from_snapshot(&snapshot)?;
    let mut profile = ModelProfile::new(
        args.model_id.clone().or_else(|| crate::families::glm5_flash::serve::model_id(&snapshot)).context("model id")?,
        ModelEncoding::Qwen(Arc::new(encoding)),
    );
    let depth = serving_queue_depth(args.http_queue_depth);
    anyhow::ensure!(depth > 0, "--http-queue-depth must be positive");
    let (queue, receive) = mpsc::channel::<NativeRequest>(depth);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    engine_args.rings = engine_args.rings.max(args.max_sequences);
    let (worker_stats, max_sequences) = (stats.clone(), args.max_sequences);
    engine_args.draft_sequences = engine_args.draft_sequences.max(args.max_sequences);
    engine_args.draft_context_slots = Some(serving_context_slots(
        engine_args.draft_sequences, engine_args.rings, engine_args.draft_context_slots, args.prefix.mimo_prefix_draft));
    let draft = Policy { fixed: args.draft_fixed,
        decode_share: args.decode_share, chunk_s: args.prefill_chunk_s,
        snapshot_wait: args.mimo_snapshot_wait };
    let prefix = args.prefix.clone();
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id)));
    let vision = args.vision;
    let audio = args.audio;
    let remote = super::media::RemoteVision::from_args(&args)?;
    let remote_audio = super::media::RemoteAudio::from_args(&args)?;
    let media_cache_bytes = args.media_cache_bytes;
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, draft, prefix, vision, audio, media_cache_bytes, remote, remote_audio));
    let (max_context, media) = ready_rx.await.context("engine failed before it was ready")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    if let Some((preparer, audio_preparer, health, audio_health)) = media {
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
        hub.clone(), api.serve(profile.clone(), &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    // The ready ledger (placement gate): one report at readiness, before any request captures
    // lazy graphs or loads more code.
    crate::shared::memory_report::log("ready");
    tracing::info!(listen = %args.listen, model = %profile.id, "MiMo V2 API is ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<std::net::SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
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
    let policy = args.draft_fixed.map_or_else(|| "adaptive".to_string(), |n| format!("fixed {n}"));
    let drafter = if args.engine.draft.is_some() { Some("DFlash") } else if args.engine.mtp > 0 { Some("MTP") } else { None };
    layout.speculator = drafter.map(|name| console::Speculator {
        name: name.into(), positions: 8, policy: policy + " · agreed copy extension" });
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

/// Neural draft settings and the shared resource-policy binding.
#[derive(Debug, Clone, Copy)]
struct Policy {
    fixed: Option<usize>,
    decode_share: DecodeShareArgs,
    chunk_s: Option<f64>,
    snapshot_wait: bool,
}

type VisionReady = Option<(Option<Arc<cuteafd_api::openai::media::MediaPreparer>>,
    Option<Arc<cuteafd_api::openai::media::audio::AudioPreparer>>, Option<Arc<std::sync::atomic::AtomicBool>>, Option<Arc<std::sync::atomic::AtomicBool>>)>;

fn serve_loop(args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<(usize, VisionReady)>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
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
        engine.check_tp2_routers()?;
        anyhow::ensure!(preparer.is_none() || media.encoder().available_for(false), "vision encoder unavailable before readiness");
        anyhow::ensure!(audio_preparer.is_none() || media.encoder().available_for(true), "audio encoder unavailable before readiness");
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok((engine.max_context, (preparer.is_some() || audio_preparer.is_some()).then(|| (preparer.clone(), audio_preparer.clone(), health.clone(), audio_health.clone())))));
        }
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), draft, &prefix,
            args.token_io.token_select, host_config, &mut media, preparer.as_deref())
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| (args.max_context, (preparer.is_some() || audio_preparer.is_some()).then(|| (preparer, audio_preparer, health, audio_health)))).map_err(|e| anyhow::anyhow!("{e:#}")));
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
    keyed_history: Vec<u32>,
    media: RequestMedia,
    /// The sequence's DFlash ring slot (None: no DFlash state).
    slot: Option<usize>,
    /// Recent DFlash (proposed, accepted) outcomes for the adaptive plan.
    drafts: DraftHistory,
    /// Steps, neural drafts verified / accepted, copy drafts verified / accepted,
    /// and actual neural-drafter calls for this request.
    counts: [usize; 6],
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
        self.finish(finish)?;
        Ok(true)
    }

    fn finish(&mut self, finish: InferenceFinishReason) -> Result<()> {
        let content = self.decoder.finish()?.unwrap_or_default();
        if !content.is_empty() || self.buffered > 0 {
            self.send(InferenceChunk::Text { content, content_tokens: self.buffered })?;
        }
        self.ticket.finishing();
        self.send(InferenceChunk::Finish { finish_reason: finish })
    }
}

/// The prefix cache over `engine` (always present: with zero entries it is the page allocator).
pub(super) fn prefix_cache<'e, 'a>(engine: &'e MimoEngine<'a>, args: &PrefixArgs, concurrency: usize)
    -> Result<(MimoPrefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let args = args.with_scoring(if engine.full_prefill_logits() { engine.prefill_rows } else { 64 },
        engine.prefill_rows, engine.cfg.vocab_size)?;
    prefix_cache_with(engine, &args, concurrency, |layout| args.host_config(layout, engine.max_context, concurrency))
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
    let scoring = args.with_scoring(if engine.full_prefill_logits() { engine.prefill_rows } else { 64 },
        engine.prefill_rows, engine.cfg.vocab_size)?;
    let host = host_config(layout)?.map(|config| {
        // Preflight resolved the quota before loading. Re-admit with all pinned
        // engine and transport owners now resident, without charging them twice.
        crate::shared::prefix::budget::admit(config.bytes, scoring.scoring_host_bytes,
            args.host_cache_headroom_bytes)?;
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
    let drafter = engine.drafter.as_ref();
    let mut free_slots: Vec<usize> = drafter.map_or(Vec::new(), |d| (0..d.slots).rev().collect());
    let head_numerics = match engine.head().weight {
        super::head::MimoHead::Bf16(_) => "head-bf16",
        super::head::MimoHead::Fp8 { .. } => "head-fp8",
    };
    let width = drafter.map_or_else(|| engine.mtp.as_ref().map_or(0, |m| m.stages.len()), |d| d.cfg.drafts());
    let mut shared = if width > 0 {
        let prior = match drafter {
            Some(d) => resource::Prior::Selector(d.confidence_prior(head_numerics)?),
            None => resource::Prior::History,
        };
        let mut shared = crate::shared::draft::runtime::SharedPolicy::with_geometry(
            engine.draft_geometry(width)?, prior, cuteafd_core::PolicySeed::weak(width))?;
        shared.policy.set_native_confidence_positions(width)?;
        engine.arm_draft_probe()?;
        resource::publish(&shared.policy, 0, 0, None);
        Some(shared)
    } else { None };
    let (mut observed_rounds, mut skipped_rounds) = (0u64, 0u64);

    let mut trace = crate::shared::speculation_trace::Trace::open()?;
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
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("expert wire unavailable until restart: {reason}");
        }
        while active.len() + prefills.len() < max_sequences {
            let busy = !active.is_empty() || !prefills.is_empty();
            if snapshot_waiter.as_ref().is_some_and(|ready| ready.job().job.events.is_closed()) {
                snapshot_waiter = None;
            }
            if snapshot_waiter.as_ref().is_some_and(|ready| prefills.iter().any(|p|
                !p.job.events.is_closed() && !probe::cold(&p.job.probe)
                && snapshot_extension(&p.keys, &ready.job().keys, prefix.prefix_cache_min_tokens))) { break; }
            let ready = if let Some(ready) = snapshot_waiter.take() { ready } else { match kv_waiter.poll(cache.pool().admission_free(), cache.pool().release_epoch(), busy,
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
                            || Ok(job.encode_prompt(&tokenizer)?)) {
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
            let capacity = ready.job().tokens.len().saturating_add(ready.job().job.max_tokens).min(engine.max_context);
            let initial_capacity = initial_reservation(ready.job().tokens.len(), ready.job().job.max_tokens, engine.max_context);
            let Some(ring) = free_rings.pop() else {
                reject(&ready, "SWA rings exhausted".into());
                continue;
            };
            // Disabled neural drafts need neither a ring slot nor context updates.
            let slot = if probe::no_speculation(&ready.job().job.probe) || policy.fixed == Some(0) {
                None
            } else { free_slots.pop() };
            if prefix.mimo_prefix_draft && engine.drafter.is_some() { family.bind_drafter(ring, slot); }
            cache.tick();
            let admit_started = Instant::now();
            // Lookup, fork of the retained pages and restore of the mark (byte-exact).
            let build = |pages| MimoPlacement { pages, ring, len: 0 };
            let admission = if cold { cache.admit_cold(family, ready.job().tokens.len(), initial_capacity, build) }
                else { cache.admit_media(family, ready.job().keys.tokens(), ready.job().keys.spans(), initial_capacity, true, build) };
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
                admit_started, job.usage.clone());
            if let (Some(usage), Some(session)) = (&job.usage, admitted.source.as_ref().and_then(|s| s.session.as_ref())) {
                usage.session(session.clone(), "prefix");
            }
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
                if policy.chunk_s.is_none() {
                    return prefill_batch(engine, family, cache, selector, batch);
                }
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
                            cache.capture_session(p.ticket.session());
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
                let session = p.ticket.session();
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &MimoPlacement| {
                    if let Some(prompt) = &prompt {
                        let after = logits.as_deref().map_or_else(After::default, |l| After::from_logits(l, true));
                        cache.capture_session(session.clone());
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
                let draft_from = if prefix.mimo_prefix_draft && engine.drafter.is_some() { family.draft_from(p.placement.ring) } else { resume };
                let admitted = (|| -> Result<Active<'_>> {
                    let mut request = Active {
                        slot,
                        drafts: DraftHistory::default(),
                        counts: [0; 6],
                        history: p.tokens,
                        keyed_history: p.keys.tokens().to_vec(),
                        media: p.media,
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
        // Grow only before submitting a round. Failed growth ends this sequence
        // cleanly without reallocating CUDA storage or stealing another live lease.
        for index in (0..active.len()).rev() {
            let request = &mut active[index];
            let rows = request.placement.len.saturating_add(64).min(request.capacity);
            if let Err(error) = cache.grow(family, &mut request.placement.pages, rows) {
                tracing::warn!(%error, "MiMo KV growth exhausted; finishing at committed frontier");
                let mut request = active.remove(index);
                if let Some(shared) = shared.as_mut() { shared.policy.release(request.ticket.id()); }
                let _ = request.finish(InferenceFinishReason::Length);
                requests += 1;
                generated_total += request.generated as u64;
                failures.finished(&request.job.events);
                request.ticket.done(request.generated);
                release(family, cache, &mut free_rings, &mut free_slots, &request.placement, request.slot);
            }
        }
        if active.is_empty() { continue; }
        let cycle = Instant::now();
        let mut round_clock = RoundClock::at(cycle);
        let mut drafted_requests = 0;
        let mut drafted_width = 0;
        let mut tally = console::Step::begin(0);
        let (draft0, emit0) = (draft_s, emit_s);
        let engine_before = tally.live().then(|| *engine.profile.borrow());
        // Rows each sequence may add after its next token.
        let room = DECODE_ROWS.saturating_sub(active.len());
        let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else { room.min(a.job.max_tokens - a.generated - 1)
            .min(a.capacity - a.placement.len - 1) }).collect();
        let ready: Vec<_> = active.iter().map(|a| match drafter {
            Some(_) => a.slot.is_some(),
            None => engine.mtp.as_ref().is_some_and(|mtp| {
                let start = a.placement.len.saturating_sub(engine.cfg.window + mtp.stages.len());
                if a.media.ready(start, a.history.len()) { true } else {
                    // Restored media is in target state, not host embeddings.
                    engine.mtp_reset(a.placement.ring as usize, a.placement.len);
                    false
                }
            }),
        }).collect();
        let draft_rooms = eligible_draft_rooms(&limits, &ready, width);
        let action_requests: Vec<_> = active.iter().zip(&limits).map(|(a, &room)| (a.ticket.id(), room)).collect();
        let run_draft = match shared.as_mut() {
            Some(shared) => select_draft_action(&mut shared.policy, &action_requests, &draft_rooms,
                width, DECODE_ROWS, policy.fixed)?,
            None => false,
        };
        // DFlash drafts after every next token (sequences with a ring slot),
        // then the adaptive plan's counts.
        let drafted: Vec<Option<super::dflash::Draft>> = match drafter {
            Some(drafter) if run_draft => {
                let seqs: Vec<(usize, DraftSeq)> = active.iter().enumerate()
                    .filter_map(|(i, a)| a.slot.filter(|_| draft_rooms[i] > 0).map(|slot| (i, DraftSeq { slot, anchor: a.next, position: a.placement.len,
                        valid_from: a.draft_from })))
                    .collect();
                for &(i, _) in &seqs {
                    active[i].counts[5] += 1;
                }
                let timer = Instant::now();
                drafted_requests = seqs.len();
                drafted_width = width;
                let drafts = round_clock.draft(|| engine.submit(|| drafter.draft(&seqs.iter().map(|(_, s)| *s).collect::<Vec<_>>(), &engine.embedding,
                    engine.head())));
                let ms = timer.elapsed().as_secs_f64() * 1e3;
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
            None if run_draft && engine.mtp.is_some() && policy.fixed != Some(0)
                && limits.iter().any(|&limit| limit > 0) => {
                let indices: Vec<usize> = (0..active.len()).filter(|&i| draft_rooms[i] > 0).collect();
                for &i in &indices {
                    active[i].counts[5] += 1;
                }
                let seqs: Vec<MtpSeq<'_>> = indices.iter().map(|&i| {
                    let a = &active[i];
                    MtpSeq { ring: a.placement.ring as usize, len: a.placement.len, tokens: &a.history, media: Some(&a.media) }
                }).collect();
                let stages = engine.mtp.as_ref().map_or(0, |m| m.stages.len());
                let timer = Instant::now();
                drafted_requests = seqs.len();
                drafted_width = stages;
                let drafts = round_clock.draft(|| engine.mtp_draft(&seqs, stages));
                let ms = timer.elapsed().as_secs_f64() * 1e3;
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
        let priors: Vec<Vec<f64>> = active.iter().enumerate().zip(&drafted).map(|((i, a), draft)| {
            let (Some(shared), Some(draft)) = (shared.as_ref(), draft.as_ref()) else { return Vec::new() };
            if drafter.is_some() {
                shared.prior(&a.drafts, &resource::DraftEvidence::Selector(&draft.features),
                    crate::shared::draft::copy::COPY_EXTENSION)
            } else {
                let pooled = a.drafts.pooled(draft.tokens.len(), active.iter().enumerate()
                    .filter(|&(other, _)| other != i).map(|(_, other)| &other.drafts));
                shared.prior(&a.drafts, &resource::DraftEvidence::History(&pooled),
                    crate::shared::draft::copy::COPY_EXTENSION)
            }
        }).collect();
        let copy_agreement = trace.as_ref().map(|_| active.iter().zip(&drafted).map(|(a, d)|
            d.as_ref().map_or(0, |d| crate::shared::draft::copy::agreement(&a.history, &d.tokens)))
            .collect::<Vec<_>>());
        let copy_windows = trace.as_ref().map(|_| active.iter().zip(&drafted).map(|(a, d)|
            crate::shared::draft::copy::trace_window(&a.history,
                d.as_ref().map_or(0, |d| d.tokens.len() + crate::shared::draft::copy::COPY_EXTENSION)))
            .collect::<Vec<_>>());
        let mut priors = priors;
        let copies: Vec<_> = active.iter().zip(&drafted).zip(&limits).zip(&mut priors)
            .map(|(((a, d), &limit), prior)| match (shared.as_ref(), d.as_ref()) {
                (Some(shared), Some(d)) if policy.fixed.is_none() => resource::copy_continuation(&shared.policy, active.len(),
                    a.ticket.id(), &a.history, &d.tokens, prior, limit, a.job.sampling.is_greedy()),
                _ => None,
            }).collect();
        for (i, prior) in priors.iter_mut().enumerate() {
            let available = copies[i].as_ref().map_or_else(
                || drafted[i].as_ref().map_or(0, |draft| draft.tokens.len()), |copy| copy.tokens.len());
            prior.truncate(available);
        }
        let mut predicted = None;
        let mut planned = if let Some(fixed) = policy.fixed {
            drafted.iter().zip(&limits).map(|(d, &limit)| d.as_ref().map_or(0, |d| fixed.min(limit).min(d.tokens.len()))).collect()
        } else if let Some(shared) = shared.as_mut().filter(|_| drafted_width > 0) {
            let candidates: Vec<_> = active.iter().zip(&priors).zip(&limits).map(|((a, prior), &limit)|
                resource::Candidate { id: a.ticket.id(), prior, limit, selector: false, cold: a.drafts.cold() }).collect();
            let selection = resource::select_drafted(&mut shared.policy, &candidates, drafted_width, DECODE_ROWS, drafted_requests);
            predicted = selection.predicted;
            selection.lengths
        } else { vec![0; active.len()] };
        if resource::trim_budget(&mut planned, DECODE_ROWS) { predicted = None; }
        if policy.fixed.is_some() {
            if let Some(shared) = shared.as_ref() {
                let candidates: Vec<_> = active.iter().zip(&priors).map(|(a, prior)|
                    cuteafd_core::DraftCandidate { id: a.ticket.id(), confidence: prior }).collect();
                crate::shared::draft::binding::trace_selection(&shared.policy, &candidates,
                    drafted_width, drafted_requests, DECODE_ROWS, &planned);
            }
        }
        let sequences: Vec<Vec<u32>> = active.iter().enumerate().map(|(i, a)| {
            let full = copies[i].as_ref().map(|c| c.tokens.as_slice())
                .unwrap_or_else(|| drafted[i].as_ref().map_or(&[][..], |d| d.tokens.as_slice()));
            let mut rows: Vec<u32> = std::iter::once(a.next).chain(full.iter().take(planned[i]).copied()).collect();
            if let Some(state) = a.constraint.as_ref() { state.truncate_proposal(&mut rows)?; }
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
        cuteafd_transport::route_log::step(|| active.iter().zip(&sequences).map(|(a, s)| (a.ticket.id(), s.len() as u32)).collect());
        engine.probe_begin(tokens.len());
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
        engine.probe_end();
        let step_s = timer.elapsed().as_secs_f64();
        verify_s += step_s;
        let timer = Instant::now();
        let (selected, logits) = match step {
            Ok(step) => step,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                if engine.is_terminal() {
                    return Err(error);
                }
                for request in active.drain(..) {
                    if let Some(shared) = shared.as_mut() { shared.policy.release(request.ticket.id()); }
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(family, cache, &mut free_rings, &mut free_slots, &request.placement, request.slot);
                }
                continue;
            }
        };
        let mut offset = 0;
        let mut context = Vec::new();
        let before: Vec<usize> = active.iter().map(|a| a.history.len()).collect();
        let mut verified = vec![(0usize, 1usize, Some(Censor::Cancelled)); active.len()];
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).enumerate()
            .map(|(i, ((request, rows), &start))| {
            if let Some(error) = poisoned[i].take() {
                let _ = request.job.events.send(Err(NativeFailure::Worker(error)));
                verified[i] = (rows.len(), 1, Some(Censor::Cancelled));
                offset += rows.len();
                return true;
            }
            let mut finished = false;
            let mut cancelled = false;
            let mut last = None;
            for j in 0..rows.len() {
                // Rows 0..=j are committed; the token row j produces is next.
                request.placement.len = start + j + 1;
                probe::decode_row(&engine.library, &request.job.probe, &logits, offset + j, request.generated,
                    request.history.len());
                match take(request.constraint.as_mut(), &selected[offset + j]).and_then(|t| Ok((t, request.emit(t)?))) {
                    Ok((token, done)) => {
                        last = Some(token);
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
                        cancelled = true;
                        break;
                    }
                }
            }
            let committed = request.placement.len - start;
            if let Some(slot) = request.slot.filter(|_| !finished || (prefix.mimo_prefix_draft && request.turn.is_some())) {
                context.extend((0..committed).map(|r| ContextRow { tap_row: offset + r, slot, position: start + r }));
            }
            offset += rows.len();
            let (proposed, accepted) = (rows.len() - 1, committed - 1);
            if prefix.mimo_prefix_draft && request.counts[0] == 0 {
                tracing::info!(position = start, valid_from = request.draft_from, drafted = proposed, accepted,
                    copy = false, neural_calls = request.counts[5], "first draft verification");
            }
            request.counts[0] += 1;
            let censor = if cancelled { Some(Censor::Cancelled) } else {
                resource::finish_censor(finished, committed, rows, last, &request.job.stop_token_ids) };
            verified[i] = (rows.len(), committed, censor);
            request.counts[1] += proposed;
            request.counts[2] += accepted;
            let neural = proposed.min(copies[i].as_ref().map_or_else(
                || drafted[i].as_ref().map_or(0, |d| d.tokens.len()), |c| c.neural));
            request.drafts.observe(neural.min(accepted + usize::from(censor.is_none())), accepted.min(neural));
            finished
        }).collect();
        for (request, &done) in active.iter().zip(&finished) {
            if done { failures.finished(&request.job.events); }
        }
        if let Some(drafter) = drafter.filter(|_| !context.is_empty()) {
            engine.submit(|| drafter.update(&context))?;
        }
        emit_s += timer.elapsed().as_secs_f64();
        if let Some(shared) = shared.as_mut() {
            let requests: Vec<_> = active.iter().enumerate().map(|(i, a)| {
                let (rows, accepted, censor) = verified[i];
                let neural = drafted[i].as_ref().map_or(0, |draft| draft.tokens.len())
                    .min(copies[i].as_ref().map_or(usize::MAX, |copy| copy.neural))
                    .min(rows.saturating_sub(1)).min(priors[i].len());
                resource::Verified { id: a.ticket.id(), rows, accepted,
                    source: if neural > 0 { DraftSource::Neural } else { DraftSource::Undrafted },
                    prior: (neural > 0).then_some(&priors[i][..neural]),
                    features: drafter.and_then(|_| drafted[i].as_ref().map(|d| &d.features[..neural])), censor }
            }).collect();
            let times = round_clock.observe();
            match engine.probe_finish(|routes, layer_us| resource::observe_drafted(&mut shared.policy,
                &requests, routes, layer_us, times, drafted_width, predicted, Some(drafted_requests))) {
                Some(Ok(())) => observed_rounds += 1,
                Some(Err(error)) => {
                    skipped_rounds += 1;
                    tracing::debug!(%error, "MiMo draft round not observed");
                }
                None => {
                    skipped_rounds += 1;
                    tracing::debug!("MiMo draft round probe incomplete");
                }
            }
            resource::publish(&shared.policy, observed_rounds, skipped_rounds, None);
        }
        if let Some(trace) = trace.as_mut() {
            let line = serde_json::json!({"rows": sequences.iter().map(Vec::len).sum::<usize>(),
                "draft_width": drafted_width, "drafted_requests": drafted_requests,
                "lengths": sequences.iter().map(|s| s.len()-1).collect::<Vec<_>>(),
                "accepted": verified.iter().map(|v| v.1.saturating_sub(1)).collect::<Vec<_>>(),
                "neural_lengths": drafted.iter().map(|d| d.as_ref().map_or(0, |d| d.tokens.len())).collect::<Vec<_>>(),
                "request_ids": active.iter().map(|a| a.ticket.id()).collect::<Vec<_>>(),
                "neural_tokens": drafted.iter().map(|d| d.as_ref().map(|d| &d.tokens)).collect::<Vec<_>>(),
                "copy_windows": copy_windows.as_ref().map(|windows| windows.iter().map(|(start, suffix, tokens)|
                    serde_json::json!({"start": start, "matched_suffix": suffix, "tokens": tokens})).collect::<Vec<_>>()),
                "committed_tokens": active.iter().zip(&before).map(|(a, &n)| &a.history[n..]).collect::<Vec<_>>(),
                "copy_agreement": copy_agreement,
                "copy_neural_prefix": copies.iter().map(|c| c.as_ref().map(|c| c.neural)).collect::<Vec<_>>(),
                "copy_lengths": copies.iter().map(|c| c.as_ref().map_or(0, |c| c.tokens.len())).collect::<Vec<_>>(),
                "prior": priors, "predicted_us": predicted, "policy": resource::snapshot()});
            if let Err(error) = trace.write(&line) {
                tracing::warn!(%error, "MiMo speculation trace write failed");
            }
        }
        for (i, request) in active.iter().enumerate() {
            let proposal = &sequences[i][1..];
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
            if let Some(shared) = shared.as_mut() { shared.policy.release(request.ticket.id()); }
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
                cache.capture_session(request.ticket.session());
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
        *stats = serde_json::json!({ "draft_policy": resource::snapshot(),"requests": requests, "generated_tokens": generated, "active": active,
            "prefilling": prefilling, "prefix_cache": cache.stats(),
            "media": media.stats(cache.stats().media_key_collisions, preparer.map_or(0, |p| p.memo_hits()))});
    }
}

fn eligible_draft_rooms(limits: &[usize], ready: &[bool], width: usize) -> Vec<usize> {
    limits.iter().zip(ready).map(|(&limit, &ready)| if ready { limit.min(width) } else { 0 }).collect()
}

fn select_draft_action(policy: &mut cuteafd_core::DraftPolicy, requests: &[(u64, usize)],
    rooms: &[usize], width: usize, max_rows: usize, fixed: Option<usize>) -> Result<bool> {
    let participants = rooms.iter().filter(|&&room| room > 0).count();
    let actions = [cuteafd_core::DraftAction { width: 0, draft_us: 0. }, cuteafd_core::DraftAction {
        width, draft_us: if participants > 0 {
            policy.draft_cost_drafted(false, requests.len(), participants, width)
        } else { 0. } }];
    let skip_rooms = vec![0; requests.len()];
    let action_rooms = [skip_rooms.as_slice(), rooms];
    let chosen = if fixed.is_none() {
        policy.choose_action_with_rooms(false, requests, &actions, &action_rooms, max_rows)?
    } else { fixed.map(|count| usize::from(count > 0)) };
    // None is the shared policy's cold-start result, not a history bypass.
    let run = participants > 0 && max_rows > requests.len() && chosen.is_none_or(|action| action == 1);
    crate::shared::draft::binding::trace_action(policy, false, requests, &actions,
        Some(&action_rooms), max_rows, chosen, usize::from(run));
    Ok(run)
}

fn serving_queue_depth(explicit: Option<usize>) -> usize { explicit.unwrap_or(16) }

fn serving_context_slots(sequences: usize, rings: usize, explicit: Option<usize>, warm: bool) -> usize {
    if warm { explicit.unwrap_or(sequences).max(sequences).max(rings) }
    else { explicit.unwrap_or(sequences).max(rings) }
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
    fn shared_draft_binding_preserves_config_geometry_and_frozen_prior() {
        use crate::shared::draft::resource::LayerHome;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/tests/fixtures/mimo-qualified-pro.json")).unwrap();
        let cfg = cuteafd_loader::families::mimo_v2::MimoV2Config::from_hf(&fixture["target_config"]).unwrap();
        let homes: Vec<_> = cfg.dense.iter().enumerate().map(|(index, &dense)| {
            if dense { LayerHome::Dense }
            else if index % 2 == 0 { LayerHome::Local { slice_bytes: 4096. } }
            else { LayerHome::Remote { slice_bytes: 8192. } }
        }).collect();
        let mut geometry = resource::geometry_with_grouping(&homes, cfg.experts, cfg.topk, DECODE_ROWS, 7,
            crate::shared::draft::copy::COPY_EXTENSION, |_, rows| Ok(if rows <= 16 { 16 } else { 48 })).unwrap();
        geometry.classes = vec![cuteafd_core::ResourceClass::rtx(), cuteafd_core::ResourceClass::spark()];
        assert_eq!(geometry.classes[1].prior, cuteafd_core::ResourceClass::spark().prior);
        assert_eq!(geometry.group_rows_by_rows[16][cfg.dense.iter().position(|&dense| !dense).unwrap()], 16);
        assert_eq!(geometry.group_rows_by_rows[17][cfg.dense.iter().position(|&dense| !dense).unwrap()], 48);
        assert_eq!((geometry.layers.len(), usize::from(geometry.experts), usize::from(geometry.topk)),
            (cfg.layers, cfg.experts, cfg.topk));
        assert_eq!((geometry.max_requests, geometry.widths.clone(), geometry.max_positions),
            (DECODE_ROWS, vec![7], 39));
        for (dense, layer) in cfg.dense.iter().zip(&geometry.layers) { assert_eq!(layer.class.is_none(), *dense); }
        let directory = tempfile::tempdir().unwrap();
        let key = "mimo_v2/dflash/bf16-bf16-linear-head-bf16-native-r1";
        let mut fit = crate::shared::draft_confidence::SelectorFit::default();
        fit.key = key.into();
        std::fs::write(directory.path().join("draft-confidence.bf16-bf16-linear-head-bf16-native-r1.json"),
            serde_json::to_vec(&fit).unwrap()).unwrap();
        let loaded = crate::shared::draft_confidence::SelectorFit::load(directory.path(), key).unwrap();
        assert_eq!(loaded.key, key);
        let mut history = DraftHistory::default();
        history.observe(7, 4);
        let features = [[1.0, 0.8, 0.5, 0.0]; 7];
        let expected = loaded.confidence(&history.conditional(7), &features).unwrap();
        let shared = crate::shared::draft::runtime::SharedPolicy::with_geometry(geometry.clone(),
            resource::Prior::Selector(loaded), cuteafd_core::PolicySeed::weak(7)).unwrap();
        assert_eq!(shared.prior(&history, &resource::DraftEvidence::Selector(&features), 0), expected);
        let mtp = crate::shared::draft::runtime::SharedPolicy::with_geometry(geometry,
            resource::Prior::History, cuteafd_core::PolicySeed::weak(7)).unwrap();
        let mut other = DraftHistory::default();
        other.observe(7, 7);
        let pooled = history.pooled(3, [&other].into_iter());
        assert_ne!(pooled, history.conditional(3));
        assert_eq!(mtp.prior(&history, &resource::DraftEvidence::History(&pooled), 0), pooled);
    }

    #[test]
    fn shared_action_reserves_excluded_anchors_and_uses_shared_cold_result() {
        let homes = [resource::LayerHome::Dense, resource::LayerHome::Local { slice_bytes: 4096. }];
        let geometry = resource::geometry_with_grouping(&homes, 384, 8, 64, 7,
            crate::shared::draft::copy::COPY_EXTENSION, |_, _| Ok(16)).unwrap();
        let mut shared = crate::shared::draft::runtime::SharedPolicy::with_geometry(geometry,
            resource::Prior::History, cuteafd_core::PolicySeed::weak(7)).unwrap();
        let limits = [7, 7, 0, 2];
        let rooms = eligible_draft_rooms(&limits, &[true, false, true, true], 7);
        assert_eq!(rooms, [7, 0, 0, 2]);
        let requests = [(10, 7), (11, 7), (12, 0), (13, 2)];
        assert!(select_draft_action(&mut shared.policy, &requests, &rooms, 7, 5, None).unwrap());
        assert!(!select_draft_action(&mut shared.policy, &requests, &rooms, 7, 4, None).unwrap());
        assert!(!select_draft_action(&mut shared.policy, &requests, &[0; 4], 7, 64, None).unwrap());
        assert!(!select_draft_action(&mut shared.policy, &requests, &rooms, 7, 64, Some(0)).unwrap());
        assert!(select_draft_action(&mut shared.policy, &requests, &rooms, 7, 64, Some(3)).unwrap());
        let rates = [0.99; 7];
        let candidates: Vec<_> = requests.iter().zip(&rooms).map(|(&(id, _), &limit)|
            resource::Candidate { id, prior: if limit > 0 { &rates } else { &[] },
                limit, selector: false, cold: false }).collect();
        let selected = resource::select_drafted(&mut shared.policy, &candidates, 7, 5, 2);
        assert_eq!(selected.lengths[1], 0);
        assert_eq!(selected.lengths[2], 0);
        assert_eq!(selected.lengths.iter().sum::<usize>(), 1);
        assert_eq!(requests.len() + selected.lengths.iter().sum::<usize>(), 5);
    }

    #[test]
    fn neural_evidence_stops_before_copy_and_agreeing_terminal_positions() {
        use crate::shared::draft::{clock::RoundTimes, routes::RoundRoutes};
        let homes = [resource::LayerHome::Dense, resource::LayerHome::Remote { slice_bytes: 4096. }];
        let geometry = resource::geometry_with_grouping(&homes, 384, 8, 64, 3,
            crate::shared::draft::copy::COPY_EXTENSION, |_, _| Ok(16)).unwrap();
        let mut shared = crate::shared::draft::runtime::SharedPolicy::with_geometry(geometry,
            resource::Prior::History, cuteafd_core::PolicySeed::weak(3)).unwrap();
        shared.policy.set_native_confidence_positions(3).unwrap();
        let mut routes = RoundRoutes::new(2, 8);
        routes.begin(7);
        routes.dense(0).unwrap();
        routes.push_narrow(1, &[0; 7 * 8]).unwrap();
        let requests = [resource::Verified { id: 1, rows: 6, accepted: 6,
            source: DraftSource::Neural, prior: Some(&[0.8, 0.7, 0.6]), features: None,
            censor: Some(Censor::OutputLimit) },
            resource::Verified { id: 2, rows: 1, accepted: 1,
                source: DraftSource::Undrafted, prior: None, features: None, censor: None }];
        resource::observe_drafted(&mut shared.policy, &requests, &routes, &[None, Some(1000.)],
            RoundTimes { total_us: 10_000, draft_us: Some(2000) }, 3, None, Some(1)).unwrap();
        assert_eq!(shared.policy.stats().verified_rows, 7);
        assert_eq!(&shared.policy.stats().position_reached[..3], &[1, 1, 1]);
        assert!(shared.policy.stats().position_reached[3..].iter().all(|&n| n == 0));
        let censored = [resource::Verified { id: 1, rows: 6, accepted: 2,
            source: DraftSource::Neural, prior: Some(&[0.8, 0.7, 0.6]), features: None,
            censor: Some(Censor::Eos) },
            resource::Verified { id: 2, rows: 1, accepted: 1,
                source: DraftSource::Undrafted, prior: None, features: None, censor: None }];
        resource::observe_drafted(&mut shared.policy, &censored, &routes, &[None, Some(1000.)],
            RoundTimes { total_us: 10_000, draft_us: Some(2000) }, 3, None, Some(1)).unwrap();
        assert_eq!(&shared.policy.stats().position_reached[..3], &[2, 1, 1]);
        assert_eq!(shared.policy.stats().verified_rows, 14);
    }

    #[test]
    fn port_knobs_default_off_and_sixteen_slots_are_configurable() {
        let default = Cli::parse_from(["serve", "--snapshot", "/model", "--native-lib", "/lib"]);
        let args = default.serve;
        assert_eq!(args.max_sequences, 4);
        assert_eq!(args.http_queue_depth, None);
        assert_eq!(args.http_queue_wait_ms, 25_000);
        assert!(!args.prefix.mimo_prefix_draft && !args.mimo_snapshot_wait);
        assert!(!args.mimo_host_cache && args.prefill_chunk_s.is_none());
        let enabled = Cli::parse_from(["serve", "--snapshot", "/model", "--native-lib", "/lib",
            "--max-sequences", "16", "--http-queue-depth", "32", "--http-queue-wait-ms", "1234",
            "--mimo-prefix-draft", "--mimo-snapshot-wait",
            "--mimo-host-cache", "--host-cache-bytes", "0", "--prefill-chunk-s", "4"]);
        assert_eq!((enabled.serve.max_sequences, enabled.serve.http_queue_depth), (16, Some(32)));
        assert_eq!(enabled.serve.prefix.host_cache_bytes, crate::shared::prefix::HostBudget::Bytes(0));
        assert_eq!(enabled.serve.prefill_chunk_s, Some(4.0));
    }

    #[test]
    fn flags_off_queue_depth_matches_legacy_for_every_sequence_count() {
        for sequences in [1, 4, 16, 32, 64] {
            let parsed = Cli::parse_from(["serve", "--snapshot", "/model", "--native-lib", "/lib",
                "--max-sequences", &sequences.to_string()]);
            assert_eq!(serving_queue_depth(parsed.serve.http_queue_depth), 16);
        }
        assert_eq!(serving_queue_depth(Some(32)), 32);
        assert_eq!(serving_queue_depth(Some(1)), 1);
    }

    #[test]
    fn flags_off_context_slots_match_legacy_explicit_policy() {
        for sequences in [4, 16, 32] {
            for rings in [4, 16, 32] {
                for explicit in [None, Some(1), Some(24), Some(64)] {
                    assert_eq!(serving_context_slots(sequences, rings, explicit, false),
                        explicit.unwrap_or(sequences).max(rings));
                }
            }
        }
        assert_eq!(serving_context_slots(32, 16, Some(1), true), 32);
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
    cuteafd_transport::route_log::step(|| vec![(p.ticket.id(), count as u32)]);
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
        cache.capture_session(p.ticket.session());
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
    cuteafd_transport::route_log::step(|| vec![(pair[0].ticket.id(), counts[0] as u32), (pair[1].ticket.id(), counts[1] as u32)]);
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
