//! OpenAI-compatible API over the Qwen 3.8 Flash Next engine: continuous
//! batching with one prefill per admitted request and one decode-shaped step
//! for every active sequence, verifying drafts: the native MTP layer's
//! (`--mtp N`, drafts per sequence planned by `mtp_policy`) or copy-window
//! drafts from the sequence's own history.
//!
//! Verify-by-replay: a step that verifies drafts runs speculatively (GDN and
//! PLE record each row's replay inputs and keep their state); the accepted
//! rows are then committed in one launch, the placement and n-gram history
//! rewound (K/V records past them are rewritten when those positions come
//! again). With MTP the kept rows' pre-mixer streams wait in the MTP stash
//! for the next cycle's draft step (see `speculate`).
//!
//! Prefix cache (`cuteafd_engine::prefix` over `super::prefix::Qwen4Prefix`):
//! admission restores the deepest retained snapshot whose tokens prefix the
//! prompt (shared units, the copied tail unit, the GDN/PLE state mark; the
//! n-gram history is recomputed from the tokens) and prefills only the rest;
//! a whole-prompt hit takes its first token from the retained logits. The
//! prompt is retained at prompt end (unless it was a whole hit), the
//! conversation at a normal finish (`Turn`; its kept rows are committed to
//! the state first), a prefill whose client left is parked at its last chunk;
//! a decode whose client left is not retained. A restored sequence's MTP
//! drafts once its own rows reach the stash. `prompt_cache_hit_tokens`
//! reports the restored rows; `/v1/stats` carries the cache's counters.
use super::engine::{history_of, Qwen4Engine, Qwen4Placement, DECODE_ROWS};
use super::mtp_policy;
use super::prefix::Qwen4Prefix;
use super::speculate::{self, DraftSeq, DraftTiming, MtpSeq, Verified};
use super::{open, Opened};
use crate::shared::prefix::CudaCopyEngine;
pub(crate) use crate::shared::prefix::{PrefixArgs, Toggle};
use cuteafd_engine::media::{EmbeddingCache, MediaAdmission, MediaPoll, MediaReady, MediaWaiter};
use crate::shared::probe;
use crate::shared::token_io::{RowResult, SelectBatch, SelectPlacement, TokenSelector};
use cuteafd_engine::prefix::{After, MarkArena, PointPlan, PointPolicy, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
use crate::shared::draft_policy::{Calibration, DraftHistory, Shape};
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

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: super::EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Sequences decoding at once (each holds two state slots).
    #[arg(long, default_value_t = 4)]
    pub max_sequences: usize,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    /// Decode one token per step (no copy-window drafts; ignored with --mtp).
    #[arg(long)]
    pub no_copy_drafts: bool,
    /// With --mtp: verify exactly --mtp drafts per sequence where room allows
    /// instead of the adaptive plan.
    #[arg(long)]
    pub mtp_fixed: bool,
    #[command(flatten)]
    pub decode_share: DecodeShareArgs,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub console: console::ConsoleArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
    /// Resolved global vision policy, assigned before dispatch.
    #[arg(skip = cuteafd_loader::plan::MediaMode::Auto)]
    pub vision: cuteafd_loader::plan::MediaMode,
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
    let encoding = Arc::new(QwenEncoding::from_snapshot(&snapshot)?);
    let eos = encoding.tokens().eos.clone();
    let mut profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&snapshot)).context("model id")?,
        ModelEncoding::Qwen(encoding),
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    engine_args.slots = engine_args.slots.max(args.max_sequences);
    let (worker_stats, max_sequences, decode_share) = (stats.clone(), args.max_sequences, args.decode_share);
    let drafts = if engine_args.mtp > 0 {
        Drafts::Mtp { depth: engine_args.mtp.min(DECODE_ROWS - 1), fixed: args.mtp_fixed }
    } else if args.no_copy_drafts {
        Drafts::None
    } else {
        Drafts::Copy
    };
    let prefix = args.prefix.clone();
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id, drafts, &eos)));
    let vision = args.vision;
    let remote = super::media::RemoteVision::from_args(&args)?;
    let media_cache_bytes = args.media_cache_bytes;
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_sequences, drafts, eos, decode_share, prefix, vision, media_cache_bytes, remote));
    let (max_context, media) = ready_rx.await.context("engine failed before it was ready")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    if let Some((preparer, health)) = media {
        profile = profile.with_loaded_vision(preparer);
        profile.vision_health = health;
    }
    cuteafd_bench::context::phase("engine loaded");
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, Duration::from_secs(25),
        hub.clone(), api.serve(profile.clone(), &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    // The ready ledger (placement gate): one report at readiness, before any request captures
    // lazy graphs or loads more code.
    crate::shared::memory_report::log("ready");
    tracing::info!(listen = %args.listen, model = %profile.id, "Qwen 3.8 Flash Next API is ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<std::net::SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
    }
    Ok(())
}

/// What the live console shows for Qwen 3.8 Flash Next.
fn console_layout(args: &ServeArgs, model: &str, drafts: Drafts, eos: &[u32]) -> console::Layout {
    use console::{Color::*, StepGroup};
    let mut layout = console::Layout::new("qwen4", model.into(), args.engine.snapshot.clone());
    let sparks = args.engine.peers.as_deref().map_or(0, |peers| peers.split(',').count());
    layout.hardware = console::hardware(1, sparks, args.engine.local_experts || args.engine.shared_only);
    layout.concurrency = args.max_sequences.min(DECODE_ROWS);
    layout.eos = eos.to_vec();
    layout.speculator = match drafts {
        Drafts::None => None,
        Drafts::Copy => Some(console::Speculator { name: "Copy window".into(), positions: COPY_DRAFT,
            policy: "adaptive length".into() }),
        Drafts::Mtp { depth, fixed } => Some(console::Speculator { name: "MTP".into(), positions: depth,
            policy: if fixed { format!("fixed {depth}") } else { format!("adaptive ≤ {depth}") } }),
    };
    layout.steps = vec![
        StepGroup::new("Decode step", "host clock", &[("round.cycle", "step", Target),
            ("draft", "draft plan + MTP / copy drafts", Accepted), ("verify", "verify pass + token selection", Target),
            ("emit", "accept + stream", Ink), ("commit", "commit kept rows + MTP stash", Accepted)]),
        StepGroup::new("Verify pass", "host clock, engine phases", &[
            ("gpu", "GPU until expert exchanges", Rtx), ("experts", "expert exchanges", Spark)]),
        StepGroup::layers(),
        StepGroup::admission(false),
    ];
    layout.layers = Some(console::Layers::host_clock());
    layout
}

/// Where a step's drafts come from.
#[derive(Debug, Clone, Copy)]
enum Drafts {
    None,
    Copy,
    /// Up to `depth` MTP drafts per sequence (all of them with `fixed`).
    Mtp { depth: usize, fixed: bool },
}

type VisionReady = Option<(Arc<cuteafd_api::openai::media::MediaPreparer>, Option<Arc<std::sync::atomic::AtomicBool>>)>;

#[allow(clippy::too_many_arguments)]
fn serve_loop(mut args: super::EngineArgs, mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<(usize, VisionReady)>>, stats: Arc<Mutex<serde_json::Value>>, max_sequences: usize,
    draft: Drafts, eos: Vec<u32>, decode_share: DecodeShareArgs, prefix: PrefixArgs,
    vision: cuteafd_loader::plan::MediaMode, media_cache_bytes: Option<u64>, remote: Option<super::media::RemoteVision>) -> Result<()> {
    args.planner_graph_modes = Some((max_sequences.min(DECODE_ROWS), !matches!(draft, Drafts::None)));
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
    let (encoder, bytes) = vision.map_or((super::media::Encoder::Off, 0), |vision| (vision.encoder, vision.cache_bytes));
    let health = encoder.health_handle();
    let mut media = MediaAdmission::new(EmbeddingCache::new(bytes), encoder, 16);
    if args.pool_tokens == 0 {
        let geometry = cuteafd_loader::serving_capacity::qwen_cache_geometry(&opened.cfg,
            opened.cfg.layers, args.mtp > 0)?;
        let mark = geometry.ranks[0].retained_mark_bytes as usize;
        let slots = MarkArena::slots_for(max_sequences, prefix.prefix_cache_entries, mark,
            prefix.prefix_cache_mark_mib << 20);
        args.planner_prefix_bytes = Some(slots as u64 * mark as u64);
    }
    let mut ready = Some(ready);
    let result = opened.with_engine(&args, |engine| {
        anyhow::ensure!(engine.weights.layers.len() == engine.cfg.layers, "serve-qwen4 needs every layer");
        anyhow::ensure!(engine.experts().is_some(),
            "serve-qwen4 needs --peers (or --local-experts) for the routed experts");
        let spark = matches!(engine.experts(), Some(super::engine::Experts::Spark { .. }));
        console::layer_classes(engine.weights.layers.iter().map(|_| console::layer_class(false, spark)).collect());
        if matches!(engine.experts(), Some(super::engine::Experts::SharedOnly)) {
            tracing::warn!("serve-qwen4 --shared-only: replies do not match the model (plumbing and cache gates only)");
        }
        engine.warm_decode_graphs(max_sequences.min(DECODE_ROWS), !matches!(draft, Drafts::None))?;
        if let Ok(path) = std::env::var("CUTEAFD_QWEN4_PADDING_TOKENS") {
            let bytes = std::fs::read(&path).with_context(|| format!("Qwen padding tokens {path}"))?;
            anyhow::ensure!(bytes.len() % 4 == 0, "Qwen padding token file is not U32-aligned");
            let tokens: Vec<_> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
            engine.check_decode_padding(&tokens)?;
        }
        // Startup captures must be visible in the first API statistics snapshot.
        {
            let mut stats = stats.lock().map_err(|_| anyhow::anyhow!("Qwen serving stats lock poisoned"))?;
            *stats = serde_json::json!({});
            probe::graph_capture_stats(&mut stats);
        }
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok((engine.max_context, preparer.clone().map(|p| (p, health.clone())))));
        }
        schedule(engine, &opened, &args.snapshot, &mut receive, &stats, max_sequences.min(DECODE_ROWS), draft, eos,
            decode_share, &prefix, args.token_io.token_select, &mut media, preparer.as_deref())
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| (args.max_context, None)).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

struct Active<'a> {
    job: NativeRequest,
    /// Prompt and generated tokens, for copy-window drafts.
    history: Vec<u32>,
    keyed_history: Vec<u32>,
    /// Current copy-draft length (halved after a fully rejected draft,
    /// doubled after a fully accepted one) and steps left before drafting
    /// resumes once it reached zero.
    draft_limit: usize,
    draft_pause: usize,
    constraint: Option<crate::shared::constraints::State<'a>>,
    placement: Qwen4Placement,
    /// MTP stash rows and recent draft outcomes; cycles in a row planned
    /// without drafts (a probe draft follows eight).
    mtp: MtpSeq,
    outcomes: DraftHistory,
    copy_outcomes: DraftHistory,
    idle: usize,
    /// Drafts proposed and accepted, verify cycles.
    proposed: usize,
    accepted: usize,
    cycles: usize,
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
    /// Admission order, for traces.
    id: u64,
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

    /// Streams `token` (special tokens stay text for the Qwen parser); returns
    /// true when the request is finished.
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

/// Remove only proposal suffixes, distributing trims across the longest sequences.
pub(crate) fn trim_copy_rows(sequences: &mut [Vec<u32>], limit: usize) {
    let mut rows: usize = sequences.iter().map(Vec::len).sum();
    while rows > limit {
        let Some((index, _)) = sequences.iter().enumerate().filter(|(_, rows)| rows.len() > 1)
            .max_by_key(|(_, rows)| rows.len()) else { break };
        sequences[index].pop();
        rows -= 1;
    }
}

/// An admitted prompt waiting for its remaining prefill chunks.
struct Prefill<'a> {
    job: NativeRequest,
    constraint: Option<crate::shared::constraints::State<'a>>,
    tokens: Vec<u32>,
    keys: cuteafd_engine::media::MediaKeys,
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
    placement: Qwen4Placement,
    capacity: usize,
    seq: MtpSeq,
    /// A whole-prompt prefix hit's retained logits (its first token is selected from them).
    logits: Option<Vec<f32>>,
    /// The first generated token, selected after the last chunk, and that
    /// row's logits when a prompt snapshot will keep them.
    first: Option<u32>,
    prompt_row: Option<Vec<f32>>,
    started: Instant,
    /// Seconds in this prompt's chunks, and their engine phases.
    busy: f64,
    phases: [f64; 2],
    id: u64,
    ticket: console::Ticket,
}

/// Per-cycle step costs for policy work (`CUTEAFD_SPECULATION_TRACE=path`, JSON lines).
struct Trace(std::io::BufWriter<std::fs::File>);

impl Trace {
    fn open() -> Result<Option<Self>> {
        let Some((var, path)) = crate::shared::draft_policy::speculation_trace_path("CUTEAFD_QWEN4_TRACE") else { return Ok(None) };
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            .with_context(|| format!("{var} {path}"))?;
        Ok(Some(Self(std::io::BufWriter::new(file))))
    }

    fn cycle(&mut self, line: serde_json::Value) {
        use std::io::Write;
        if let Err(error) = writeln!(self.0, "{line}").and_then(|_| self.0.flush()) {
            tracing::warn!("trace: {error:#}");
        }
    }
}

/// The prefix cache over `engine` (always present: with zero entries it is the page allocator),
/// its mark arena sized for `lanes` decoding sequences.
fn prefix_cache<'e, 'a>(engine: &'e Qwen4Engine<'a>, args: &PrefixArgs, lanes: usize)
    -> Result<(Qwen4Prefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    let entries = args.prefix_cache_entries;
    let budget = args.prefix_cache_mark_mib << 20;
    anyhow::ensure!(args.prefix_partial == Toggle::Off, "Qwen 3.8 Flash Next restores exact snapshots only (GDN state)");
    let family = Qwen4Prefix::new(engine, |mark| if entries == 0 { 0 } else { MarkArena::slots_for(lanes, entries, mark, budget) })?;
    let host = args.host_tier(engine.library, family.template(), family.layout(), engine.max_context)?;
    let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
    let layout = family.layout();
    let config = PrefixConfig { entries, mark_slots: family.slots(), keep_logits: true,
        min_tokens: args.prefix_cache_min_tokens };
    let cache = PrefixCache::new(layout, config, host)?;
    tracing::info!(entries, mark_slots = family.slots(), mark_bytes = family.mark_bytes(), page_bytes = layout.page_bytes,
        pages = layout.pages, page_rows = layout.page_rows, host_bytes, points = ?args.points(),
        "Qwen 3.8 Flash Next prefix cache");
    cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, layout.pages as u64,
        &"BF16 full-attention + GDN/PLE state".to_string(), host_bytes);
    Ok((family, cache))
}

/// Gives a finished or failed sequence's units and state slot back.
fn release(family: &Qwen4Prefix<'_, '_>, cache: &mut PrefixCache<CudaCopyEngine<'_>>, slots: &mut Vec<i32>,
    placement: &Qwen4Placement) {
    if let Err(error) = cache.release(family, &placement.units) {
        tracing::error!(%error, "releasing a Qwen 3.8 Flash Next sequence's units");
    }
    slots.push(placement.slot);
}

#[derive(Clone, Copy, Default, serde::Serialize)]
struct VerifyBucket {
    steps: u64,
    speculative_steps: u64,
    verify_ms_sum: f64,
    verify_ms_max: f64,
}

struct VerifyStats {
    real: [VerifyBucket; DECODE_ROWS + 1],
    bucket: [VerifyBucket; DECODE_ROWS + 1],
}

impl Default for VerifyStats {
    fn default() -> Self {
        Self { real: [VerifyBucket::default(); DECODE_ROWS + 1],
            bucket: [VerifyBucket::default(); DECODE_ROWS + 1] }
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

#[cfg(test)]
mod verify_tests {
    #[test]
    fn copy_trim_retains_mandatory_tokens_and_proposal_prefixes() {
        for sequences in 1..=16 {
            for extra in 0..=64-sequences {
                let original: Vec<Vec<u32>> = (0..sequences).map(|i| {
                    (0..=extra / sequences + usize::from(i < extra % sequences))
                        .map(|j| (i * 100 + j) as u32).collect()
                }).collect();
                let real = original.iter().map(Vec::len).sum::<usize>();
                let limit = super::super::engine::copy_row_limit(real, sequences);
                let mut trimmed = original.clone();
                super::trim_copy_rows(&mut trimmed, limit);
                assert_eq!(trimmed.iter().map(Vec::len).sum::<usize>(), limit);
                for (before, after) in original.iter().zip(&trimmed) {
                    assert!(!after.is_empty());
                    assert_eq!(after, &before[..after.len()]);
                }
            }
        }
    }

    #[test]
    fn histogram_counts_real_and_physical_rows_at_existing_timing_boundary() {
        let mut stats = super::VerifyStats::default();
        assert_eq!(stats.snapshot()["rows"], serde_json::json!({}));
        stats.record(17, 24, true, 12.5);
        stats.record(24, 24, true, 10.0);
        stats.record(16, 16, false, 3.0);
        let json = stats.snapshot();
        assert_eq!(json["rows"], serde_json::json!({"16": 1, "17": 1, "24": 1}));
        assert_eq!(json["by_bucket"]["24"]["steps"], 2);
        assert_eq!(json["by_bucket"]["24"]["speculative_steps"], 2);
        assert_eq!(json["by_bucket"]["24"]["verify_ms_sum"], 22.5);
        assert_eq!(json["by_bucket"]["24"]["verify_ms_max"], 12.5);
        assert_eq!(json["by_real_rows"]["16"]["speculative_steps"], 0);
    }
}

/// Serving statistics for `/v1/stats`.
fn publish(stats: &Mutex<serde_json::Value>, requests: u64, generated: u64, active: usize, prefilling: usize,
    cache: &PrefixCache<CudaCopyEngine<'_>>, media: &MediaAdmission<super::media::Prompt, super::media::Encoder>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>, verify: &VerifyStats) {
    if let Ok(mut stats) = stats.lock() {
        *stats = serde_json::json!({"requests": requests, "generated_tokens": generated, "active": active,
            "prefilling": prefilling, "prefix_cache": cache.stats(), "verify": verify.snapshot(),
            "media": media.stats(cache.stats().media_key_collisions, preparer.map_or(0, |p| p.memo_hits()))});
        probe::graph_capture_stats(&mut stats);
    }
}

/// Message starts of the Qwen chat template: a snapshot right before one is a message boundary.
pub(crate) const MESSAGE_STARTS: [&str; 1] = ["<|im_start|>"];

#[allow(clippy::too_many_arguments)]
fn schedule(engine: &Qwen4Engine<'_>, opened: &Opened, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    drafts: Drafts, eos: Vec<u32>, decode_share: DecodeShareArgs, prefix: &PrefixArgs, select: SelectPlacement,
    media: &mut MediaAdmission<super::media::Prompt, super::media::Encoder>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>)
    -> Result<()> {
    let (family, mut cache) = prefix_cache(engine, prefix, max_sequences)?;
    let markers = crate::shared::prefix::marker_ids(snapshot, &MESSAGE_STARTS)?;
    let mut free_slots: Vec<i32> = (0..engine.slots as i32).rev().collect();
    let mut grammars = crate::shared::constraints::Compiler::with_vocab(
        &opened.library, snapshot.join("tokenizer.json"), engine.cfg.vocab_size, eos);
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total, mut admissions) = (0u64, 0u64, 0u64);
    // Verify steps since the last completed request, host seconds in verifies and in MTP draft steps.
    let (mut steps, mut verify_s) = (0u64, 0f64);
    let mut timing = DraftTiming::default();
    let mut verify_stats = VerifyStats::default();
    let mut selector = TokenSelector::new(&opened.library, select, engine.cfg.vocab_size, DECODE_ROWS)?;
    let mtp = matches!(drafts, Drafts::Mtp { .. });
    let mut cost = mtp_policy::cycle_cost(DECODE_ROWS);
    let mut calibration = Calibration::default();
    let copy_policy = crate::shared::draft_policy::enabled("CUTEAFD_COPY_DRAFT_POLICY");
    tracing::info!(copy_policy, "shared Qwen copy policy experiment; existing MTP online calibration unchanged");
    let mut trace = Trace::open()?;
    let mut prefills = decode_share.queue::<Prefill<'_>>()?;
    let config = &opened.checkpoint.config;
    let mut kv_waiter = cuteafd_engine::prefix::DeferredAdmission::<MediaReady<super::media::Prompt>>::default();
    loop {
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("expert wire unavailable until restart: {reason}");
        }
        while active.len() + prefills.len() < max_sequences {
            let busy = !active.is_empty() || !prefills.is_empty();
            let ready = match kv_waiter.poll(cache.pool().admission_free(), cache.pool().release_epoch(), busy,
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
                        publish(stats, requests, generated_total, active.len(), prefills.len(), &cache, media, preparer, &verify_stats);
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
                        if !job.media.is_empty() && !media.encoder().available() {
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
            };
            if !ready.job().job.media.is_empty() && !media.encoder().available() {
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
                if let Err(error) = probe.spec.validate_cold_steps(ready.job().tokens.len(), engine.prefill_rows, DECODE_ROWS) {
                    probe.fail(format!("cold replay: {error:#}"));
                    reject(&ready, format!("cold replay: {error:#}"));
                    continue;
                }
            }
            let rope = match super::media::rope_positions(&ready.job().tokens, &ready.job().job, ready.media(), config) {
                Ok(rope) => rope,
                Err(error) => { reject(&ready, format!("rotary positions: {error:#}")); continue; }
            };
            let cold = ready.cold() || probe::cold(&ready.job().job.probe);
            let capacity = (ready.job().tokens.len() + ready.job().job.max_tokens).min(engine.max_context);
            let Some(slot) = free_slots.pop() else {
                reject(&ready, "state slots exhausted".into());
                continue;
            };
            cache.tick();
            let admit_started = Instant::now();
            // Lookup, fork of the retained units and restore of the state mark (byte-exact), with
            // room for the rows a verify may write past the last kept one.
            let start = history_of(&engine.cfg, &[]);
            let build = |units| Qwen4Placement::new(units, slot, start.clone());
            let extent = (capacity + DECODE_ROWS).min(engine.max_context);
            let admission = if cold { cache.admit_cold(&family, ready.job().tokens.len(), extent, build) }
                else { cache.admit_media(&family, ready.job().keys.tokens(), ready.job().keys.spans(), extent, true, build) };
            let admitted = match admission {
                Ok(admitted) => admitted,
                Err(error) => {
                    free_slots.push(slot);
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
                    release(&family, &mut cache, &mut free_slots, &admitted.placement);
                    let events = waiter.job.job.events.clone();
                    if let Err((_, error)) = media.enqueue(waiter) {
                        let _ = events.send(Err(super::media::failure(error)));
                    }
                    continue;
                }
            };
            let (super::media::Prompt { job, tokens, keys }, request_media) = ready.into_parts();
            admissions += 1;
            let mut placement = admitted.placement;
            // Native ids and this request's grids rebuild positions on EVERY restore.
            placement.rope = rope;
            placement.media = Some(request_media);
            // The PLE n-gram context at the restore point is a function of the token ids.
            placement.history = history_of(&engine.cfg, &tokens[..resume]);
            probe::admitted(&job.probe, "qwen4", &tokens, resume);
            let _ = job.events.send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: resume },
            }));
            if let Some(from) = probe::scoring(&job.probe) {
                // Teacher-forced scoring: every row's logits, no generation, nothing retained.
                let scored = probe::score(&opened.library, &job.probe, &tokens, from, engine.prefill_rows,
                    DECODE_ROWS, probe::verify_rows(&job.probe), engine.full_prefill_logits, &mut placement,
                    |placement, chunk, rows| Ok(engine.prefill_device(placement, chunk, None, None, rows)?
                        .map(probe::ScoreLogits::Device)),
                    |placement, chunk| engine.verify_device_ungraphed(&mut [(placement, chunk)], false)?
                        .context("scoring needs every layer"));
                match scored {
                    Ok(_) => { let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })); }
                    Err(error) => { let _ = job.events.send(Err(NativeFailure::Worker(format!("scoring: {error:#}")))); }
                }
                release(&family, &mut cache, &mut free_slots, &placement);
                continue;
            }
            let ticket = console::admit(tokens.len(), resume, job.max_tokens, constraint.is_some(), job.media.len(),
                admit_started, job.usage.clone());
            if let (Some(usage), Some(session)) = (&job.usage, admitted.source.as_ref().and_then(|s| s.session.as_ref())) {
                usage.session(session.clone(), "prefix");
            }
            if let Some(source) = admitted.source {
                tracing::info!(tokens = tokens.len(), resume, kind = ?source.kind, frontier = source.frontier,
                    host = source.host, "prefix cache hit");
            }
            // A cold probe keeps the same chunk plan (identical numerics); it only skips the captures.
            let plan = if cache.enabled() {
                cuteafd_engine::prefix::plan_media_points(resume, tokens.len(), engine.prefill_rows,
                    &crate::shared::prefix::boundaries(&tokens, &markers), family.capture_reach(),
                    prefix.prefix_cache_min_tokens, prefix.points(), keys.spans())
            } else {
                cuteafd_engine::prefix::plan_points(resume, tokens.len(), engine.prefill_rows, &[], 0, 0,
                    PointPolicy { gap: 0, boundaries: 0, per_request: 0 })
            };
            // A whole-prompt hit brings its first token's logits: nothing to prefill.
            let logits = admitted.after.and_then(|after| after.logits).map(|logits| logits.to_vec());
            let plan = if let Some(probe) = job.probe.as_ref().filter(|p| !p.spec.cold_steps.is_empty()) {
                PointPlan { chunks: probe.spec.cold_steps.iter().map(|step| step.end).collect(), points: Vec::new() }
            } else if logits.is_some() { PointPlan::default() } else { plan };
            // The first chunk's PLE rows page in while earlier work runs.
            if logits.is_none() {
                let end = plan.chunks.first().copied().unwrap_or(tokens.len());
                engine.prefetch_ple(&placement.history, &tokens[resume..end]);
            }
            prefills.push(Prefill { job, constraint, tokens, keys, done: resume, resume, plan, chunks: 0, cancelled: false,
                placement, capacity, seq: MtpSeq::default(), logits, first: None, prompt_row: None,
                started: Instant::now(), busy: 0.0, phases: [0.0; 2], id: admissions, ticket });
        }
        if prefills.due(!active.is_empty()) {
            let caching = cache.enabled();
            // One chunk of each waiting prompt (whole prompts with --decode-share 0).
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
                // The next chunk's PLE rows page in while this one runs.
                if end < p.tokens.len() {
                    let next = p.plan.chunks.get(p.chunks + 1).copied().unwrap_or(p.tokens.len());
                    engine.prefetch_ple(&history_of(&engine.cfg, &p.tokens[..end]), &p.tokens[end..next]);
                }
                let (result, phases) = isolated_phases(&engine.profile, || -> Result<()> {
                    let start = p.placement.len;
                    let decode = p.job.probe.as_ref().and_then(|probe| probe.spec.cold_steps.get(p.chunks))
                        .is_some_and(|step| step.decode);
                    let logits = if decode {
                        engine.verify_device_ungraphed(&mut [(&mut p.placement, chunk)], false)?
                    } else { engine.prefill_device(&mut p.placement, chunk, None, None, 1)? };
                    if end == p.tokens.len() {
                        // The first token, while this prompt's logits are the workspace's.
                        let mut logits = logits.context("prefill produced no logits")?;
                        // A decode-shaped cold replay returns all rows; the next
                        // token and prompt snapshot use only its last logit row.
                        if logits.rows > 1 {
                            logits.ptr = logits.ptr.cast::<u8>().wrapping_add((logits.rows - 1) * logits.stride * 4).cast();
                            logits.rows = 1;
                            logits.greedy = None;
                        }
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
                    if mtp && !probe::no_speculation(&p.job.probe) {
                        speculate::prefill_chunk(engine, &p.placement, start, chunk, p.tokens.get(end).copied(),
                            &mut p.seq)?;
                    }
                    Ok(())
                });
                p.done += chunk.len();
                add_phases(&mut p.phases, phases);
                p.busy += timer.elapsed().as_secs_f64();
                p.ticket.prefill(chunk.len(), p.chunks, p.plan.chunks.len(), timer);
                result?;
                // Intermediate snapshot points this chunk ends at (off unless configured).
                for &(_, point) in p.plan.points.iter().filter(|&&(chunk, _)| chunk == p.chunks && !probe::cold(&p.job.probe)) {
                    cache.capture_session(p.ticket.session());
                    if let Err(error) = cache.capture_media(&family, SnapshotKind::Prompt, &p.keys.tokens()[..point], p.keys.spans(), &p.placement,
                        After::default()) {
                        tracing::warn!("snapshot point {point} not retained: {error:#}");
                    }
                }
                p.chunks += 1;
                Ok(if p.done == p.tokens.len() { Chunk::Done } else { Chunk::More })
            });
            for (mut p, prefilled) in finished {
                let (placement, resume) = (p.placement.clone(), p.resume);
                if let Err(error) = &prefilled {
                    if p.cancelled {
                        p.ticket.cancel();
                        // The client left during the prefill: keep what it computed for a retry.
                        if placement.len > resume && !probe::cold(&p.job.probe) {
                            cache.capture_session(p.ticket.session());
                            if let Err(error) = cache.park_media(&family, &p.keys.tokens()[..placement.len], p.keys.spans(), &placement) {
                                tracing::warn!("parking a cancelled prefill: {error:#}");
                            }
                        }
                    } else {
                        tracing::warn!("prefill failed: {error:#}");
                        let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    }
                    release(&family, &mut cache, &mut free_slots, &placement);
                    continue;
                }
                // A whole-prompt hit selects from its retained logits; a prefill selected already.
                let first = match (p.first, p.logits.as_deref()) {
                    (Some(first), _) => Ok(first),
                    (None, Some(logits)) => {
                        if probe::wants_first(&p.job.probe) {
                            probe::host_row(&p.job.probe, p.tokens.len(), logits);
                        }
                        select_host(p.constraint.as_mut(), p.job.sampling, logits, p.placement.len as u64)
                    }
                    (None, None) => Err(anyhow::anyhow!("prefill produced no logits")),
                };
                let first = match first {
                    Ok(first) => first,
                    Err(error) => {
                        tracing::warn!("{error:#}");
                        release(&family, &mut cache, &mut free_slots, &placement);
                        continue;
                    }
                };
                let logits = p.prompt_row.take();
                if resume < p.tokens.len() {
                    let elapsed = p.started.elapsed().as_secs_f64();
                    tracing::info!(tokens = p.tokens.len(), cached = resume, elapsed_ms = (1e3 * elapsed) as u64,
                        busy_ms = (1e3 * p.busy) as u64, tok_s = (p.tokens.len() - resume) as f64 / p.busy,
                        gpu_wait_ms = (1e3 * p.phases[0]) as u64, experts_ms = (1e3 * p.phases[1]) as u64, "prefill");
                    engine.log_table_stats("prefill");
                }
                // The prompt snapshot, taken once the first token is out (it only enqueues copies).
                let prompt = (resume < p.tokens.len() && !probe::cold(&p.job.probe)).then(|| p.keys.tokens().to_vec());
                let spans = p.keys.spans().to_vec();
                let session = p.ticket.session();
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &Qwen4Placement| {
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
                        keyed_history: p.keys.tokens().to_vec(),
                        history: p.tokens,
                        draft_limit: COPY_DRAFT,
                        draft_pause: 0,
                        decoder: cuteafd_loader::streaming_token_decoder(snapshot, false)?,
                        job: p.job, constraint: p.constraint, placement: p.placement, mtp: p.seq,
                        outcomes: DraftHistory::default(), copy_outcomes: DraftHistory::default(), idle: 0, proposed: 0, accepted: 0, cycles: 0, turn: None,
                        capacity: p.capacity, next: 0, generated: 0, buffered: 0, started: Instant::now(), id: p.id,
                        ticket: p.ticket,
                    };
                    request.next = first;
                    request.mtp.close(request.next);
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
                                release(&family, &mut cache, &mut free_slots, &request.placement)
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!("admission failed: {error:#}");
                        let _ = job_events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        retain_prompt(&mut cache, &placement);
                        release(&family, &mut cache, &mut free_slots, &placement);
                    }
                }
            }
            prefills.settle(!active.is_empty());
        }
        if active.is_empty() {
            if !media.is_empty() { std::thread::sleep(Duration::from_millis(1)); }
            continue;
        }
        let cycle = Instant::now();
        let mut tally = console::Step::begin(0);
        let engine_before = tally.live().then(|| *engine.profile.borrow());
        let mut rates: Vec<Vec<f64>> = Vec::new();
        let mut neural_confidence: Vec<Vec<f64>> = vec![Vec::new(); active.len()];
        let (steps_before, draft_s_before) = (timing.steps, timing.seconds);
        // Each sequence verifies its next token plus its drafts within the decode programs' rows.
        let room = (DECODE_ROWS / active.len()).max(1) - 1;
        let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else {
            room.min(a.job.max_tokens - a.generated - 1).min(a.capacity - a.placement.len - 1) }).collect();
        let mut proposals: Vec<Vec<u32>> = match drafts {
            Drafts::None => vec![Vec::new(); active.len()],
            Drafts::Copy => active.iter_mut().zip(&limits).map(|(a, &limit)| {
                if a.draft_pause > 0 {
                    a.draft_pause -= 1;
                    if a.draft_pause == 0 {
                        a.draft_limit = 1;
                    }
                }
                // `emit` already appended `next` to the history.
                copy_drafts(&a.history, limit.min(a.draft_limit))
            }).collect(),
            Drafts::Mtp { depth, fixed } => {
                let limits: Vec<usize> = limits.iter().map(|&l| l.min(depth)).collect();
                let histories: Vec<&DraftHistory> = active.iter().map(|a| &a.outcomes).collect();
                let confidence;
                (rates, confidence) = mtp_policy::acceptance(&histories, &limits, &calibration);
                neural_confidence = confidence.clone();
                let mut depths = mtp_policy::plan(&confidence, fixed.then_some(depth), &cost);
                for ((a, d), &limit) in active.iter_mut().zip(depths.iter_mut()).zip(&limits) {
                    // A sequence planned without drafts for a while probes one.
                    a.idle = if *d == 0 { a.idle + 1 } else { 0 };
                    if a.idle > 8 && limit > 0 {
                        *d = 1;
                        a.idle = 0;
                    }
                    // A whole-prompt prefix hit has no MTP rows yet: it drafts after its first step.
                    if a.mtp.pending.is_empty() {
                        *d = 0;
                    }
                }
                let pending: usize = active.iter().map(|a| a.mtp.pending.len()).sum();
                let rows: usize = depths.iter().map(|d| d + 1).sum();
                if depths.iter().any(|&d| d > 0) || pending + rows > DECODE_ROWS / 2 {
                    let steps = timing.steps;
                    let timer = Instant::now();
                    let mut seqs: Vec<DraftSeq<'_>> = active.iter_mut().zip(&depths).map(|(a, &depth)| DraftSeq {
                        placement: &a.placement, seq: &mut a.mtp, depth }).collect();
                    let proposals = speculate::draft(engine, &mut seqs, &mut timing)?;
                    if depths.iter().any(|&d| d > 0) {
                        cost.observe_chain(active.len(), timing.steps - steps, 1e3 * timer.elapsed().as_secs_f64());
                    }
                    proposals
                } else {
                    vec![Vec::new(); active.len()]
                }
            }
        };
        let mut used_copy = vec![matches!(drafts, Drafts::Copy); active.len()];
        if copy_policy && !matches!(drafts, Drafts::None) {
            let copies: Vec<_> = if matches!(drafts, Drafts::Copy) { std::mem::take(&mut proposals) }
                else { active.iter().zip(&limits).map(|(a, &limit)| if a.job.sampling.is_greedy() {
                    copy_drafts(&a.history, limit.min(COPY_DRAFT))
                } else { Vec::new() }).collect() };
            if proposals.is_empty() { proposals = vec![Vec::new(); active.len()]; }
            let copy_rates: Vec<_> = active.iter().zip(&copies).map(|(a, copy)| a.copy_outcomes.conditional(copy.len())).collect();
            let inputs: Vec<_> = active.iter().enumerate().map(|(i, a)| crate::shared::draft_policy::CopyInput {
                key: (a.placement.len, i as u64), neural: &proposals[i], confidence: &neural_confidence[i],
                copy: &copies[i], copy_confidence: &copy_rates[i],
            }).collect();
            let (lengths, used) = crate::shared::draft_policy::compete_copies(&inputs,
                timing.steps > steps_before, timing.steps - steps_before, &cost);
            used_copy = used;
            for i in 0..proposals.len() {
                if used_copy[i] { proposals[i] = copies[i][..lengths[i]].to_vec(); }
            }
        }
        let mut sequences: Vec<Vec<u32>> = active.iter().zip(&proposals).map(|(a, drafted)| {
            let mut rows: Vec<u32> = std::iter::once(a.next).chain(drafted.iter().copied()).collect();
            // Drafts the grammar rejects could never be kept: verify none of them.
            if let Some(state) = a.constraint.as_ref() {
                state.truncate_proposal(&mut rows)?;
            }
            Ok(rows)
        }).collect::<Result<_>>()?;
        let diagnostic = active.iter().any(|a| a.job.probe.is_some());
        if matches!(drafts, Drafts::Copy) {
            let rows = sequences.iter().map(Vec::len).sum();
            let limit = engine.copy_verify_row_limit(rows, sequences.len(), diagnostic);
            trim_copy_rows(&mut sequences, limit);
        }
        let mut poisoned: Vec<Option<String>> = vec![None; sequences.len()];
        let spec = sequences.iter().any(|rows| rows.len() > 1);
        let draft_us = console::us(cycle);
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
        let histories: Vec<_> = active.iter().map(|a| a.placement.history.clone()).collect();
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        let mut rows: Vec<(&mut Qwen4Placement, &[u32])> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.as_slice())).collect();
        steps += 1;
        let timer = Instant::now();
        let step = if diagnostic {
            engine.verify_device_ungraphed(&mut rows, spec)
        } else {
            engine.verify_device(&mut rows, spec)
        }.and_then(|logits| logits.context("decode needs every layer"))
            .and_then(|logits| {
                // Each row draws at the position after it, masked along its sequence's drafts.
                let mut batch = SelectBatch::default();
                for (i, ((a, rows), &start)) in active.iter().zip(&sequences).zip(&starts).enumerate() {
                    // A grammar failure fails that sequence alone, after the step.
                    poisoned[i] = batch.push_sequence_isolated(a.job.sampling, a.constraint.as_ref(), rows, start as u64 + 1);
                }
                Ok((selector.select(&logits, &batch)?, logits))
            });
        let elapsed = timer.elapsed().as_secs_f64();
        verify_stats.record(tokens.len(), engine.verify_bucket_rows(tokens.len(), spec, diagnostic), spec, 1e3 * elapsed);
        verify_s += elapsed;
        let shape = Shape::plain(tokens.len(), sequences.len());
        let predicted_ms = cost.verify_ms(shape);
        cost.observe_verify(shape, 1e3 * elapsed);
        let (selected, logits) = match step {
            Ok(step) => step,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(&family, &mut cache, &mut free_slots, &request.placement);
                }
                continue;
            }
        };
        let caching = cache.enabled();
        let mut offset = 0;
        let mut kept = Vec::with_capacity(active.len());
        let before: Vec<usize> = active.iter().map(|a| a.history.len()).collect();
        let emit_timer = Instant::now();
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).enumerate()
            .map(|(index, ((request, rows), &start))| {
            if let Some(error) = poisoned[index].take() {
                let _ = request.job.events.send(Err(NativeFailure::Worker(error)));
                offset += rows.len();
                return true;
            }
            let mut finished = false;
            let mut last = None;
            for j in 0..rows.len() {
                // Rows 0..=j are committed; the token row j produces is next.
                request.placement.len = start + j + 1;
                probe::decode_row(&opened.library, &request.job.probe, &logits, offset + j, request.generated,
                    request.history.len());
                match take(request.constraint.as_mut(), &selected[offset + j]).and_then(|t| {
                    // The token's PLE rows page in during emission, the commit and (MTP)
                    // drafting, before the step that feeds it gathers them.
                    engine.prefetch_ple(&history_of(&engine.cfg, &request.history), &[t]);
                    Ok((t, request.emit(t)?))
                }) {
                    Ok((token, done)) => {
                        last = Some((j + 1, token));
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
            offset += rows.len();
            // Adapt the draft length to how much of it the model reproduced.
            let drafted = rows.len() - 1;
            let accepted = (request.placement.len - start).saturating_sub(1);
            request.cycles += 1;
            request.proposed += drafted;
            request.accepted += accepted;
            if used_copy[index] {
                request.copy_outcomes.observe(drafted.min(accepted + usize::from(!finished)), accepted);
            }
            if matches!(drafts, Drafts::Mtp { .. }) && !used_copy[index] {
                let observed = if copy_policy { drafted.min(accepted + usize::from(!finished)) } else { drafted };
                request.outcomes.observe(observed, accepted);
                if let Some(rates) = rates.get(index) {
                    calibration.observe_outcome(rates, observed, accepted);
                }
            } else if drafted > 0 && accepted == 0 {
                request.draft_limit /= 2;
                if request.draft_limit == 0 {
                    request.draft_pause = 8;
                }
            } else if drafted > 0 && accepted == drafted {
                request.draft_limit = (request.draft_limit * 2).clamp(1, COPY_DRAFT);
            }
            // A turn snapshot captures the state at the kept length: commit it too.
            kept.push(if !finished || request.turn.is_some() { last } else { None });
            finished
        }).collect();
        let (emit_us, commit_timer) = (console::us(emit_timer), Instant::now());
        // Commit the kept rows (speculative steps), rewind, and stash them for the MTP.
        let mut first_row = 0;
        let mut verified: Vec<Verified<'_>> = Vec::with_capacity(active.len());
        for ((((request, rows), (&start, history)), kept), &finished) in active.iter_mut().zip(&sequences)
            .zip(starts.iter().zip(histories)).zip(&kept).zip(&finished) {
            verified.push(Verified { placement: &mut request.placement, seq: &mut request.mtp, start, history,
                rows, first_row, kept: *kept, finished });
            first_row += rows.len();
        }
        speculate::accept(engine, &mut verified, spec, mtp)?;
        drop(verified);
        let commit_us = console::us(commit_timer);
        for (i, request) in active.iter().enumerate() {
            tally.member(&request.ticket, &proposals[i], sequences[i].len() - 1, &request.history[before[i]..],
                request.constraint.is_some(), finished[i]);
        }
        tally.end(|| {
            let engine_now = *engine.profile.borrow();
            let gpu = |i: usize| engine_before.map_or(f64::NAN, |before| 1e6 * (engine_now[i] - before[i]));
            vec![("draft", draft_us), ("verify", 1e6 * elapsed), ("emit", emit_us), ("commit", commit_us),
                ("gpu", gpu(0)), ("experts", gpu(1))]
        });
        if let Some(trace) = trace.as_mut() {
            trace.cycle(serde_json::json!({"rows": tokens.len(), "seqs": sequences.len(),
                "ids": active.iter().map(|a| a.id).collect::<Vec<_>>(),
                "depths": sequences.iter().map(|s| s.len() - 1).collect::<Vec<_>>(),
                "kept": kept.iter().map(|k| k.map_or(0, |(n, _)| n)).collect::<Vec<_>>(),
                "verify_ms": 1e3 * elapsed, "draft_steps": timing.steps - steps_before,
                "draft_ms": 1e3 * (timing.seconds - draft_s_before), "cycle_ms": 1e3 * cycle.elapsed().as_secs_f64(),
                "predicted_ms": predicted_ms, "fit": cost.fitted(active.len()), "rates": rates,
                "calibration": calibration.fitted()}));
        }
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
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps, cycles = request.cycles, proposed = request.proposed,
                accepted = request.accepted, tokens_per_cycle = request.generated as f64 / request.cycles.max(1) as f64,
                verify_s, draft_s = timing.seconds, draft_steps = timing.steps, gpu_wait_s = phases[0],
                experts_s = phases[1], "request complete");
            engine.log_table_stats("decode");
            (steps, verify_s, timing) = (0, 0.0, DraftTiming::default());
            if let Some(row) = &request.turn {
                // The conversation so far: every committed row (the last token is not in it).
                let rows = &request.keyed_history[..request.placement.len];
                let spans = request.placement.media.as_ref().map_or(&[][..], |media| media.spans());
                cache.capture_session(request.ticket.session());
                if let Err(error) = cache.capture_media(&family, SnapshotKind::Turn, rows, spans, &request.placement,
                    After::from_logits(row, true)) {
                    tracing::warn!("turn snapshot not retained: {error:#}");
                }
            }
            release(&family, &mut cache, &mut free_slots, &request.placement);
        }
        cache.tick();
        publish(stats, requests, generated_total, active.len(), prefills.len(), &cache, media, preparer, &verify_stats);
        console::gauges(|| console::Gauges::prefix_cache(&cache, active.len(), prefills.len(), receive.len() + kv_waiter.len()));
        prefills.stepped(cycle.elapsed().as_secs_f64());
    }
}
