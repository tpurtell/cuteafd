//! Continuous Qwen request scheduling over family-owned execution and state.
use anyhow::Result;
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::media::{EncodeJob, EncoderClient, MediaKeys, RequestMedia};
use cuteafd_ffi::NativeLibrary;
use crate::shared::{draft::routes::RoundRoutes, draft_policy::DraftHistory,
    prefix::{CudaCopyEngine, PrefixArgs}, token_io::DeviceLogits};
use std::{cell::RefCell, path::Path};

#[derive(Debug, Clone, Copy)]
pub(crate) enum Drafts {
    None,
    Neural { depth: usize, fixed: bool },
}

pub(crate) struct Geometry {
    pub family: &'static str,
    pub vocab_size: usize,
    pub max_context: usize,
    pub slots: usize,
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub full_prefill_logits: bool,
}

pub(crate) struct Prompt {
    pub job: NativeRequest,
    pub tokens: Vec<u32>,
    pub keys: MediaKeys,
}

pub(crate) trait Encoder: EncoderClient {
    fn available(&self) -> bool;
}

pub(crate) trait Placement: Clone {
    fn len(&self) -> usize;
    fn set_len(&mut self, len: usize);
    fn units(&self) -> &[u32];
    fn slot(&self) -> i32;
    fn spans(&self) -> &[cuteafd_core::MediaSpan];
}

pub(crate) trait DraftState: Default {
    fn pending_rows(&self) -> usize;
    fn close(&mut self, token: u32);
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DraftTiming {
    pub steps: usize,
    pub seconds: f64,
}

pub(crate) struct DraftSeq<'p, P, D> {
    pub placement: &'p P,
    pub seq: &'p mut D,
    pub depth: usize,
}

pub(crate) struct Verified<'p, P, D, H> {
    pub placement: &'p mut P,
    pub seq: &'p mut D,
    pub start: usize,
    pub history: H,
    pub rows: &'p [u32],
    pub first_row: usize,
    pub kept: Option<(usize, u32)>,
    pub finished: bool,
}

/// The scheduler owns admission, batching, token selection and publication.
/// Family adapters retain all model state, prefix byte layouts and draft math.
pub(crate) trait Backend<'a> {
    type Placement: Placement;
    type Draft: DraftState;
    type History;
    type Coordinates;
    type Encoder: Encoder;
    type Prefix: PrefixFamily<Placement = Self::Placement>;

    fn geometry(&self) -> Geometry;
    fn message_starts(&self) -> &[&str] { &MESSAGE_STARTS }
    fn library(&self) -> &'a NativeLibrary;
    fn profile(&self) -> &RefCell<[f64; 2]>;
    fn prefix_cache(&self, args: &PrefixArgs, lanes: usize)
        -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)>;
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(Prompt, RequestMedia, Vec<EncodeJob>)>;
    fn probe_features(&self, prompt: &Prompt, media: &mut RequestMedia,
        cache: &mut cuteafd_engine::media::EmbeddingCache, snapshot: &Path) -> Result<()>;
    fn coordinates(&self, tokens: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<Self::Coordinates>;
    fn placement(&self, units: Vec<u32>, slot: i32) -> Self::Placement;
    fn bind_request(&self, placement: &mut Self::Placement, coordinates: Self::Coordinates,
        media: RequestMedia, tokens: &[u32], resume: usize);
    fn history(&self, placement: &Self::Placement) -> Self::History;
    fn prefetch(&self, tokens: &[u32], next: &[u32]);
    fn require_live(&self) -> Result<()>;
    fn log_table_stats(&self, phase: &str);
    // Family write validation uses the actual cache pool after admission/COW.
    fn prefill_device(&self, placement: &mut Self::Placement, tokens: &[u32], logit_rows: usize,
        pool: &cuteafd_engine::prefix::RefPagePool)
        -> Result<Option<DeviceLogits>>;
    fn verify_device(&self, sequences: &mut [(&mut Self::Placement, &[u32])], speculative: bool,
        pool: &cuteafd_engine::prefix::RefPagePool)
        -> Result<Option<DeviceLogits>>;
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Self::Placement, &[u32])], speculative: bool,
        pool: &cuteafd_engine::prefix::RefPagePool)
        -> Result<Option<DeviceLogits>>;
    fn verify_bucket_rows(&self, rows: usize, speculative: bool, diagnostic: bool) -> usize;
    fn verify_attention_trace(&self, contexts: &[Vec<usize>], rows: usize) -> serde_json::Value;
    fn draft_geometry(&self, snapshot: &Path, width: usize) -> Result<cuteafd_core::PolicyGeometry>;
    fn arm_draft_probe(&self) -> Result<()>;
    fn prefill_draft(&self, placement: &Self::Placement, start: usize, chunk: &[u32], next: Option<u32>,
        draft: &mut Self::Draft) -> Result<()>;
    fn draft(&self, sequences: &mut [DraftSeq<'_, Self::Placement, Self::Draft>], timing: &mut DraftTiming)
        -> Result<Vec<Vec<u32>>>;
    fn accept(&self, verified: &mut [Verified<'_, Self::Placement, Self::Draft, Self::History>],
        speculative: bool, draft: bool) -> Result<()>;
    fn probe_begin(&self, rows: usize);
    fn probe_end(&self);
    fn probe_finish<T>(&self, observe: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T>;
}

use cuteafd_engine::media::{MediaAdmission, MediaPoll, MediaReady, MediaWaiter};
use crate::shared::probe;
use crate::shared::token_io::{RowResult, SelectBatch, SelectPlacement, TokenSelector};
use cuteafd_engine::prefix::{After, PointPlan, PointPolicy, PrefixCache, PrefixFamily, SnapshotKind};
use crate::shared::draft::{resource, clock::RoundClock, evidence::{Censor, DraftSource}};
use crate::shared::prefill_share::{add_phases, isolated_phases, Chunk, DecodeShareArgs};
use anyhow::Context;
use cuteafd_api::openai::{
    InferenceChunk, InferenceFinishReason, NativeFailure,
    PromptUsage,
};
use crate::shared::console;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;


struct Active<'a, P: Placement, D> {
    job: NativeRequest,
    /// Prompt and generated tokens, for copy-window drafts.
    history: Vec<u32>,
    keyed_history: Vec<u32>,
    constraint: Option<crate::shared::constraints::State<'a>>,
    placement: P,
    /// Family draft continuation and recent outcomes.
    draft: D,
    outcomes: DraftHistory,
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

impl<P: Placement, D> Active<'_, P, D> {
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
        } else if self.generated >= self.job.max_tokens || self.placement.len() + 1 >= self.capacity {
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

/// Diagnostic row trimming preserves every anchor and each proposal's prefix.
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
struct Prefill<'a, P, D> {
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
    placement: P,
    capacity: usize,
    seq: D,
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
struct Trace(crate::shared::speculation_trace::Trace);

impl Trace {
    fn open() -> Result<Option<Self>> {
        Ok(crate::shared::speculation_trace::Trace::open()?.map(Self))
    }

    fn cycle(&mut self, line: serde_json::Value) {
        if let Err(error) = self.0.write(&line) {
            tracing::warn!("trace: {error:#}");
        }
    }
}

/// Gives a finished or failed sequence's units and state slot back.
fn release<P: Placement, F: PrefixFamily<Placement = P>>(family: &F, cache: &mut PrefixCache<CudaCopyEngine<'_>>, slots: &mut Vec<i32>,
    placement: &P, family_name: &str) {
    if let Err(error) = cache.release(family, placement.units()) {
        tracing::error!(%error, family = family_name, "releasing sequence units");
    }
    slots.push(placement.slot());
}

#[derive(Clone, Copy, Default, serde::Serialize)]
struct VerifyBucket {
    steps: u64,
    speculative_steps: u64,
    verify_ms_sum: f64,
    verify_ms_max: f64,
}

struct VerifyStats {
    real: Vec<VerifyBucket>,
    bucket: Vec<VerifyBucket>,
}

impl VerifyStats {
    fn new(decode_rows: usize) -> Self {
        Self { real: vec![VerifyBucket::default(); decode_rows + 1],
            bucket: vec![VerifyBucket::default(); decode_rows + 1] }
    }

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
                let limit = sequences + (real - sequences) / 2;
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
    fn histogram_uses_target_decode_capacity() {
        for capacity in [1, 16, 64, 128] {
            let mut stats = super::VerifyStats::new(capacity);
            stats.record(capacity, capacity, false, 1.0);
            assert_eq!(stats.snapshot()["rows"][capacity.to_string()], serde_json::json!(1));
        }
    }

    #[test]
    fn histogram_counts_real_and_physical_rows_at_existing_timing_boundary() {
        let mut stats = super::VerifyStats::new(64);
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
fn publish<E: Encoder>(stats: &Mutex<serde_json::Value>, requests: u64, generated: u64, active: usize, prefilling: usize,
    cache: &PrefixCache<CudaCopyEngine<'_>>, media: &MediaAdmission<Prompt, E>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>, verify: &VerifyStats) {
    if let Ok(mut stats) = stats.lock() {
        *stats = serde_json::json!({ "draft_policy": resource::snapshot(),"requests": requests, "generated_tokens": generated, "active": active,
            "prefilling": prefilling, "prefix_cache": cache.stats(), "verify": verify.snapshot(),
            "media": media.stats(cache.stats().media_key_collisions, preparer.map_or(0, |p| p.memo_hits()))});
        probe::graph_capture_stats(&mut stats);
    }
}

/// Message starts of the Qwen chat template: a snapshot right before one is a message boundary.
pub(crate) const MESSAGE_STARTS: [&str; 1] = ["<|im_start|>"];

fn initialize<T>(init: impl FnOnce() -> Result<T>, ready: impl FnOnce() -> Result<()>) -> Result<T> {
    let state = init()?;
    ready()?;
    Ok(state)
}

#[cfg(test)]
mod readiness_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn scheduler_prefix_cache_is_initialized_before_readiness() {
        let source = include_str!("scheduler.rs");
        let schedule = source.split_once("\npub(crate) fn schedule").unwrap().1;
        assert!(schedule.find("engine.prefix_cache(").unwrap() < schedule.find("}, initialized)?").unwrap());
    }

    #[test]
    fn qwen_rewritten_assistant_history_keeps_an_exact_prefill_boundary() -> Result<()> {
        use clap::Parser;
        use cuteafd_api::openai::chat::qwen4::{ChatTemplate, QwenPromptOptions, template_context};
        use cuteafd_api::openai::chat::qwen4::prompt::QwenToolChoice;
        use serde_json::json;
        #[derive(Parser)]
        struct Args { #[command(flatten)] prefix: PrefixArgs }
        let args = Args::parse_from(["serve", "--prefix-point-boundaries", "1"]);
        // Dense Qwen's checkpoint always drops historical thinking; Flash Next's
        // otherwise shared fixture also offers a preserve_thinking override.
        let fixture = include_str!(
            "../../../../cuteafd-api/src/openai/chat/qwen4/fixtures/qwen38_flash_next.jinja");
        let historical_thinking = "preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index";
        assert!(fixture.contains(historical_thinking));
        let template = ChatTemplate::new(&fixture.replace(historical_thinking,
            "loop.index0 > ns.last_query_index"))?;
        let options = QwenPromptOptions { thinking: false, tool_names: vec![],
            tool_choice: QwenToolChoice::Auto, response_format: None };
        let mut body = json!({"messages":[{"role":"system", "content":"notes ".repeat(30)},
            {"role":"user", "content":"first question"}]});
        let first = template.render(&template_context(&body, &options).map_err(anyhow::Error::msg)?)?;
        body["messages"].as_array_mut().unwrap().extend([
            json!({"role":"assistant", "content":"answer"}),
            json!({"role":"user", "content":"next question"})]);
        let next = template.render(&template_context(&body, &options).map_err(anyhow::Error::msg)?)?;
        assert!(!next.starts_with(&first), "old thinking suffix is rewritten");
        // Byte ids stand in for text tokens; the marker positions are the rendered
        // template's real message boundaries, independent of tokenizer merges.
        let tokens = first.bytes().map(u32::from).collect::<Vec<_>>();
        let markers = first.match_indices("<|im_start|>").map(|(i, _)| i).collect::<Vec<_>>();
        let boundary = *markers.last().unwrap();
        let plan = cuteafd_engine::prefix::plan_media_points(0, tokens.len(), 4096, &markers,
            0, args.prefix.prefix_cache_min_tokens, args.prefix.points(), &[]);
        assert_eq!(plan.points, vec![(0, boundary)]);
        assert_eq!(plan.chunks, vec![boundary, tokens.len()]);
        assert!(next.starts_with(&first[..boundary]));
        // The live scheduler must use these points and retain all three frontiers.
        let source = include_str!("scheduler.rs").split_once("\npub(crate) fn schedule").unwrap().1;
        for fragment in ["prefix.points()", "&p.keys.tokens()[..point]", "retain_prompt(", "SnapshotKind::Turn"] {
            assert!(source.contains(fragment), "scheduler lost {fragment}");
        }
        Ok(())
    }

    #[test]
    fn scheduler_init_failure_never_publishes_ready() {
        let published = Cell::new(false);
        let result = initialize(|| -> Result<()> { anyhow::bail!("scheduler init failed") }, || {
            published.set(true);
            Ok(())
        });
        assert!(result.unwrap_err().to_string().contains("scheduler init failed"));
        assert!(!published.get());
    }

    #[test]
    fn scheduler_ready_follows_successful_initialization() -> Result<()> {
        let initialized = Cell::new(false);
        let result = initialize(|| { initialized.set(true); Ok(17) }, || {
            assert!(initialized.get());
            Ok(())
        })?;
        assert_eq!(result, 17);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn schedule<'a, B: Backend<'a>>(engine: &B, snapshot: &std::path::Path,
    receive: &mut mpsc::Receiver<NativeRequest>, stats: &Mutex<serde_json::Value>, max_sequences: usize,
    drafts: Drafts, eos: Vec<u32>, decode_share: DecodeShareArgs, prefix: &PrefixArgs, select: SelectPlacement,
    media: &mut MediaAdmission<Prompt, B::Encoder>,
    preparer: Option<&cuteafd_api::openai::media::MediaPreparer>,
    initialized: impl FnOnce() -> Result<()>)
    -> Result<()> {
    let geometry = engine.geometry();
    anyhow::ensure!(geometry.decode_rows > 0 && max_sequences <= geometry.decode_rows
        && max_sequences <= geometry.slots, "scheduler lanes exceed target decode/state capacity");
    let width = match drafts { Drafts::Neural { depth, .. } => depth, Drafts::None => 0 };
    // Publish readiness only after all fallible scheduler-owned startup work.
    let (family, mut cache, markers, mut grammars, tokenizer, mut selector, mut shared, mut trace, mut prefills) = initialize(|| {
        let (family, cache) = engine.prefix_cache(prefix, max_sequences)?;
        let markers = crate::shared::prefix::marker_ids(snapshot, engine.message_starts())?;
        let grammars = crate::shared::constraints::Compiler::with_vocab(
            engine.library(), snapshot.join("tokenizer.json"), geometry.vocab_size, eos);
        let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(snapshot)?;
        let selector = TokenSelector::new(engine.library(), select, geometry.vocab_size, geometry.decode_rows)?;
        let shared = if width > 0 {
            let geometry = engine.draft_geometry(snapshot, width)?;
            engine.arm_draft_probe()?;
            let mut policy = cuteafd_core::DraftPolicy::seeded(geometry, cuteafd_core::PolicySeed::weak(width))?;
            crate::shared::draft::apply_length_rule(&mut policy);
            Some(policy)
        } else { None };
        let trace = Trace::open()?;
        let prefills = decode_share.queue::<Prefill<'_, B::Placement, B::Draft>>()?;
        Ok((family, cache, markers, grammars, tokenizer, selector, shared, trace, prefills))
    }, initialized)?;
    let mut free_slots: Vec<i32> = (0..geometry.slots as i32).rev().collect();
    let mut active: Vec<Active<'_, B::Placement, B::Draft>> = Vec::new();
    let (mut requests, mut generated_total, mut admissions) = (0u64, 0u64, 0u64);
    // Verify steps since the last completed request, host seconds in verifies and in family draft steps.
    let (mut steps, mut verify_s) = (0u64, 0f64);
    let mut timing = DraftTiming::default();
    let mut verify_stats = VerifyStats::new(geometry.decode_rows);
    let neural = matches!(drafts, Drafts::Neural { .. });
    let mut kv_waiter = cuteafd_engine::prefix::DeferredAdmission::<MediaReady<Prompt>>::default();
    loop {
        engine.require_live()?;
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
                        let _ = prompt.job.events.send(Err(media_failure(error)));
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
                        if let Err(error) = probe::validate_scoring(&job.probe, geometry.full_prefill_logits) {
                            let _ = job.events.send(Err(NativeFailure::BadRequest(format!("scoring: {error:#}"))));
                            continue;
                        }
                        if !job.media.is_empty() && !media.encoder().available() {
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
                        let (prompt, mut request_media, jobs) = match engine.prepare(job, tokens) {
                            Ok(prepared) => prepared,
                            Err(error) => { let _ = events.send(Err(NativeFailure::BadRequest(format!("{error:#}")))); continue; }
                        };
                        if prompt.tokens.is_empty() || prompt.tokens.len() >= geometry.max_context {
                            let _ = events.send(Err(NativeFailure::BadRequest(format!("prompt of {} tokens is outside 1..{}",
                                prompt.tokens.len(), geometry.max_context))));
                            continue;
                        }
                        if let Err(error) = engine.probe_features(&prompt, &mut request_media, &mut media.cache, snapshot) {
                            if let Some(probe) = &prompt.job.probe { probe.fail(format!("reference features: {error:#}")); }
                            let _ = events.send(Err(NativeFailure::BadRequest(format!("reference features: {error:#}"))));
                            drop(request_media); media.cache.prune_reservations();
                            continue;
                        }
                        let resume = if probe::cold(&prompt.job.probe) { 0 }
                            else { cache.peek_media(prompt.keys.tokens(), prompt.keys.spans(), true) };
                        let waiter = MediaWaiter::new(prompt, request_media, jobs, resume)?;
                        if let Err((_, error)) = media.enqueue(waiter) {
                            let _ = events.send(Err(media_failure(error)));
                        }
                        continue;
                    }
                },
            };
            if !ready.job().job.media.is_empty() && !media.encoder().available() {
                let _ = ready.job().job.events.send(Err(NativeFailure::Unavailable("vision encoder unavailable".into())));
                continue;
            }
            let reject = |ready: &MediaReady<Prompt>, message: String| {
                let _ = ready.job().job.events.send(Err(NativeFailure::BadRequest(message)));
            };
            let constraint = match ready.job().job.constraint.as_ref().map(|spec| grammars.matcher(spec)).transpose() {
                Ok(constraint) => constraint,
                Err(error) => { reject(&ready, format!("{error:#}")); continue; }
            };
            if let Some(probe) = &ready.job().job.probe {
                if let Err(error) = probe.spec.validate_cold_steps(ready.job().tokens.len(), geometry.prefill_rows, geometry.decode_rows) {
                    probe.fail(format!("cold replay: {error:#}"));
                    reject(&ready, format!("cold replay: {error:#}"));
                    continue;
                }
            }
            let rope = match engine.coordinates(&ready.job().tokens, &ready.job().job, ready.media()) {
                Ok(rope) => rope,
                Err(error) => { reject(&ready, format!("rotary positions: {error:#}")); continue; }
            };
            let cold = ready.cold() || probe::cold(&ready.job().job.probe);
            let capacity = (ready.job().tokens.len() + ready.job().job.max_tokens).min(geometry.max_context);
            let Some(slot) = free_slots.pop() else {
                reject(&ready, "state slots exhausted".into());
                continue;
            };
            cache.tick();
            let admit_started = Instant::now();
            // Lookup, fork of the retained units and restore of the state mark (byte-exact), with
            // room for the rows a verify may write past the last kept one.
            let build = |units| engine.placement(units, slot);
            let extent = (capacity + geometry.decode_rows).min(geometry.max_context);
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
                    release(&family, &mut cache, &mut free_slots, &admitted.placement, geometry.family);
                    let events = waiter.job.job.events.clone();
                    if let Err((_, error)) = media.enqueue(waiter) {
                        let _ = events.send(Err(media_failure(error)));
                    }
                    continue;
                }
            };
            let (Prompt { job, tokens, keys }, request_media) = ready.into_parts();
            admissions += 1;
            let mut placement = admitted.placement;
            // Native ids and this request's grids rebuild positions on EVERY restore.
            engine.bind_request(&mut placement, rope, request_media, &tokens, resume);
            probe::admitted(&job.probe, geometry.family, &tokens, resume);
            let _ = job.events.send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: resume },
            }));
            if let Some(from) = probe::scoring(&job.probe) {
                // Teacher-forced scoring: every row's logits, no generation, nothing retained.
                let scored = probe::score(engine.library(), &job.probe, &tokens, from, geometry.prefill_rows,
                    geometry.decode_rows, probe::verify_rows(&job.probe), geometry.full_prefill_logits, &mut placement,
                    |placement, chunk, rows| Ok(engine.prefill_device(placement, chunk, rows, cache.pool())?
                        .map(probe::ScoreLogits::Device)),
                    |placement, chunk| engine.verify_device_ungraphed(&mut [(placement, chunk)], false, cache.pool())?
                        .context("scoring needs every layer"));
                match scored {
                    Ok(_) => { let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })); }
                    Err(error) => { let _ = job.events.send(Err(NativeFailure::Worker(format!("scoring: {error:#}")))); }
                }
                release(&family, &mut cache, &mut free_slots, &placement, geometry.family);
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
                cuteafd_engine::prefix::plan_media_points(resume, tokens.len(), geometry.prefill_rows,
                    &crate::shared::prefix::boundaries(&tokens, &markers), family.capture_reach(),
                    prefix.prefix_cache_min_tokens, prefix.points(), keys.spans())
            } else {
                cuteafd_engine::prefix::plan_points(resume, tokens.len(), geometry.prefill_rows, &[], 0, 0,
                    PointPolicy { gap: 0, boundaries: 0, per_request: 0 })
            };
            // A whole-prompt hit brings its first token's logits: nothing to prefill.
            let logits = admitted.after.and_then(|after| after.logits).map(|logits| logits.to_vec());
            let plan = if let Some(probe) = job.probe.as_ref().filter(|p| !p.spec.cold_steps.is_empty()) {
                PointPlan { chunks: probe.spec.cold_steps.iter().map(|step| step.end).collect(), points: Vec::new() }
            } else if logits.is_some() { PointPlan::default() } else { plan };
            // Family table reads for the first chunk overlap earlier work.
            if logits.is_none() {
                let end = plan.chunks.first().copied().unwrap_or(tokens.len());
                engine.prefetch(&tokens[..resume], &tokens[resume..end]);
            }
            prefills.push(Prefill { job, constraint, tokens, keys, done: resume, resume, plan, chunks: 0, cancelled: false,
                placement, capacity, seq: B::Draft::default(), logits, first: None, prompt_row: None,
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
                // Family table reads for the next chunk overlap this one.
                if end < p.tokens.len() {
                    let next = p.plan.chunks.get(p.chunks + 1).copied().unwrap_or(p.tokens.len());
                    engine.prefetch(&p.tokens[..end], &p.tokens[end..next]);
                }
                let (result, phases) = isolated_phases(engine.profile(), || -> Result<()> {
                    let start = p.placement.len();
                    let decode = p.job.probe.as_ref().and_then(|probe| probe.spec.cold_steps.get(p.chunks))
                        .is_some_and(|step| step.decode);
                    let logits = if decode {
                        engine.verify_device_ungraphed(&mut [(&mut p.placement, chunk)], false, cache.pool())?
                    } else { engine.prefill_device(&mut p.placement, chunk, 1, cache.pool())? };
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
                            p.prompt_row = Some(logits.row_host(engine.library(), 0)?);
                        }
                        if probe::wants_first(&p.job.probe) {
                            probe::device_rows(engine.library(), &p.job.probe, &logits, 0, 1, p.tokens.len())?;
                        }
                        let mut batch = SelectBatch::default();
                        batch.push_next(p.job.sampling, p.constraint.as_mut(), p.placement.len() as u64)?;
                        let selected = selector.select(&logits, &batch)?;
                        p.first = Some(take(p.constraint.as_mut(), &selected[0])?);
                    }
                    if neural && !probe::no_speculation(&p.job.probe) {
                        engine.prefill_draft(&p.placement, start, chunk, p.tokens.get(end).copied(),
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
                        if placement.len() > resume && !probe::cold(&p.job.probe) {
                            cache.capture_session(p.ticket.session());
                            if let Err(error) = cache.park_media(&family, &p.keys.tokens()[..placement.len()], p.keys.spans(), &placement) {
                                tracing::warn!("parking a cancelled prefill: {error:#}");
                            }
                        }
                    } else {
                        tracing::warn!("prefill failed: {error:#}");
                        let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    }
                    release(&family, &mut cache, &mut free_slots, &placement, geometry.family);
                    continue;
                }
                // A whole-prompt hit selects from its retained logits; a prefill selected already.
                let first = match (p.first, p.logits.as_deref()) {
                    (Some(first), _) => Ok(first),
                    (None, Some(logits)) => {
                        if probe::wants_first(&p.job.probe) {
                            probe::host_row(&p.job.probe, p.tokens.len(), logits);
                        }
                        select_host(p.constraint.as_mut(), p.job.sampling, logits, p.placement.len() as u64)
                    }
                    (None, None) => Err(anyhow::anyhow!("prefill produced no logits")),
                };
                let first = match first {
                    Ok(first) => first,
                    Err(error) => {
                        tracing::warn!("{error:#}");
                        release(&family, &mut cache, &mut free_slots, &placement, geometry.family);
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
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, placement: &B::Placement| {
                    if let (Some(prompt), Some(logits)) = (&prompt, &logits) {
                        cache.capture_session(session.clone());
                        if let Err(error) = cache.capture_media(&family, SnapshotKind::Prompt, prompt, &spans, placement,
                            After::from_logits(logits, true)) {
                            tracing::warn!("prompt snapshot not retained: {error:#}");
                        }
                    }
                };
                let job_events = p.job.events.clone();
                let admitted = (|| -> Result<Active<'_, B::Placement, B::Draft>> {
                    let mut request = Active {
                        keyed_history: p.keys.tokens().to_vec(),
                        history: p.tokens,
                        decoder: cuteafd_loader::streaming_token_decoder(snapshot, false)?,
                        job: p.job, constraint: p.constraint, placement: p.placement, draft: p.seq,
                        outcomes: DraftHistory::default(), proposed: 0, accepted: 0, cycles: 0, turn: None,
                        capacity: p.capacity, next: 0, generated: 0, buffered: 0, started: Instant::now(), id: p.id,
                        ticket: p.ticket,
                    };
                    request.next = first;
                    request.draft.close(request.next);
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
                                release(&family, &mut cache, &mut free_slots, &request.placement, geometry.family)
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!("admission failed: {error:#}");
                        let _ = job_events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        retain_prompt(&mut cache, &placement);
                        release(&family, &mut cache, &mut free_slots, &placement, geometry.family);
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
        let engine_before = tally.live().then(|| *engine.profile().borrow());
        let mut neural_confidence: Vec<Vec<f64>> = vec![Vec::new(); active.len()];
        let mut round_clock = RoundClock::at(cycle);
        let mut drafted_width = 0;
        let mut drafted_requests = 0;
        let mut predicted = None;
        let (steps_before, draft_s_before) = (timing.steps, timing.seconds);
        // Each sequence verifies its next token plus its drafts within the decode programs' rows.
        let room = geometry.decode_rows.saturating_sub(active.len());
        let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else {
            room.min(a.job.max_tokens - a.generated - 1).min(a.capacity - a.placement.len() - 1) }).collect();
        let mut proposals: Vec<Vec<u32>> = match drafts {
            Drafts::None => vec![Vec::new(); active.len()],
            Drafts::Neural { depth, fixed } => {
                let limits: Vec<usize> = limits.iter().map(|&l| l.min(depth)).collect();
                let histories: Vec<&DraftHistory> = active.iter().map(|a| &a.outcomes).collect();
                neural_confidence = DraftHistory::acceptance(&histories, &limits);
                for (a, prior) in active.iter().zip(&mut neural_confidence) {
                    if a.draft.pending_rows() == 0 { prior.clear(); }
                }
                let shared = shared.as_mut().expect("neural drafts have a shared policy");
                let candidates: Vec<_> = active.iter().zip(&neural_confidence).zip(&limits).map(|((a, prior), &limit)|
                    resource::Candidate { id: a.ticket.id(), prior, limit, selector: false, cold: false }).collect();
                let selection = resource::select_drafted(shared, &candidates, depth, geometry.decode_rows,
                    neural_confidence.iter().filter(|p| !p.is_empty()).count());
                predicted = selection.predicted;
                let mut depths = if fixed { neural_confidence.iter().map(Vec::len).collect() } else { selection.lengths };
                for (a, d) in active.iter().zip(depths.iter_mut()) {
                    // A whole-prompt prefix hit has no draft continuation yet: it drafts after its first step.
                    if a.draft.pending_rows() == 0 {
                        *d = 0;
                    }
                }
                let pending: usize = active.iter().map(|a| a.draft.pending_rows()).sum();
                if resource::trim_budget(&mut depths, geometry.decode_rows) { predicted = None; }
                let rows: usize = depths.iter().map(|d| d + 1).sum();
                if depths.iter().any(|&d| d > 0) || pending + rows > geometry.decode_rows / 2 {
                    let mut seqs: Vec<DraftSeq<'_, _, _>> = active.iter_mut().zip(&depths).map(|(a, &depth)| DraftSeq {
                        placement: &a.placement, seq: &mut a.draft, depth }).collect();
                    drafted_requests = depths.iter().filter(|&&n| n > 0).count();
                    drafted_width = depths.iter().copied().max().unwrap_or(0);
                    let proposals = round_clock.draft(|| engine.draft(&mut seqs, &mut timing))?;
                    proposals
                } else {
                    vec![Vec::new(); active.len()]
                }
            }
        };
        let trace_neural = trace.as_ref().map(|_| proposals.clone());
        let trace_copy = trace.as_ref().map(|_| active.iter().zip(&proposals).map(|(a, p)|
            crate::shared::draft::copy::trace_window(&a.history, p.len() + crate::shared::draft::copy::COPY_EXTENSION))
            .collect::<Vec<_>>());
        let mut neural_lengths: Vec<_> = proposals.iter().map(Vec::len).collect();
        let copies: Vec<_> = active.iter().zip(&proposals).zip(&limits).zip(&mut neural_confidence)
            .map(|(((a, proposal), &limit), prior)| {
                prior.truncate(proposal.len());
                shared.as_ref().filter(|_| !matches!(drafts, Drafts::Neural { fixed: true, .. }))
                    .and_then(|shared| resource::copy_continuation(shared, active.len(), a.ticket.id(), &a.history,
                        proposal, prior, limit, a.job.sampling.is_greedy()))
            }).collect();
        if let Some(shared) = shared.as_mut().filter(|_| drafted_width > 0 && !matches!(drafts, Drafts::Neural { fixed: true, .. })) {
            let candidates: Vec<_> = active.iter().zip(&neural_confidence).zip(&limits).map(|((a, prior), &limit)|
                resource::Candidate { id: a.ticket.id(), prior, limit, selector: false, cold: false }).collect();
            let selection = resource::select_drafted(shared, &candidates, drafted_width, geometry.decode_rows, drafted_requests);
            predicted = selection.predicted;
            for (((proposal, copy), neural), length) in proposals.iter_mut().zip(copies)
                .zip(&mut neural_lengths).zip(selection.lengths) {
                if let Some(copy) = copy {
                    *neural = copy.neural;
                    *proposal = copy.tokens;
                }
                proposal.truncate(length);
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
        let mut poisoned: Vec<Option<String>> = vec![None; sequences.len()];
        let spec = sequences.iter().any(|rows| rows.len() > 1);
        let draft_us = console::us(cycle);
        let starts: Vec<usize> = active.iter().map(|a| a.placement.len()).collect();
        let histories: Vec<_> = active.iter().map(|a| engine.history(&a.placement)).collect();
        let tokens: Vec<u32> = sequences.iter().flatten().copied().collect();
        let mut rows: Vec<(&mut B::Placement, &[u32])> = active.iter_mut().zip(&sequences)
            .map(|(a, s)| (&mut a.placement, s.as_slice())).collect();
        steps += 1;
        let timer = Instant::now();
        engine.probe_begin(tokens.len());
        let step = if diagnostic {
            engine.verify_device_ungraphed(&mut rows, spec, cache.pool())
        } else {
            engine.verify_device(&mut rows, spec, cache.pool())
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
        engine.probe_end();
        let elapsed = timer.elapsed().as_secs_f64();
        verify_stats.record(tokens.len(), engine.verify_bucket_rows(tokens.len(), spec, diagnostic), spec, 1e3 * elapsed);
        verify_s += elapsed;
        let (selected, logits) = match step {
            Ok(step) => step,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    if let Some(shared) = shared.as_mut() { shared.release(request.ticket.id()); }
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(&family, &mut cache, &mut free_slots, &request.placement, geometry.family);
                }
                continue;
            }
        };
        let caching = cache.enabled();
        let mut offset = 0;
        let mut kept = Vec::with_capacity(active.len());
        let before: Vec<usize> = active.iter().map(|a| a.history.len()).collect();
        let emit_timer = Instant::now();
        let mut outcomes = vec![(0usize, 1usize, Some(Censor::Cancelled)); active.len()];
        let finished: Vec<bool> = active.iter_mut().zip(&sequences).zip(&starts).enumerate()
            .map(|(index, ((request, rows), &start))| {
            if let Some(error) = poisoned[index].take() {
                let _ = request.job.events.send(Err(NativeFailure::Worker(error)));
                outcomes[index] = (rows.len(), 1, Some(Censor::Cancelled));
                kept.push(None);
                offset += rows.len();
                return true;
            }
            let mut finished = false;
            let mut last = None;
            let mut cancelled = false;
            for j in 0..rows.len() {
                // Rows 0..=j are committed; the token row j produces is next.
                request.placement.set_len(start + j + 1);
                probe::decode_row(engine.library(), &request.job.probe, &logits, offset + j, request.generated,
                    request.history.len());
                match take(request.constraint.as_mut(), &selected[offset + j]).and_then(|t| {
                    // Family table reads overlap emission, commit and drafting.
                    engine.prefetch(&request.history, &[t]);
                    Ok((t, request.emit(t)?))
                }) {
                    Ok((token, done)) => {
                        last = Some((j + 1, token));
                        finished = done;
                        if done && caching && !probe::cold(&request.job.probe) {
                            // A normal finish (the client took the last chunk): the row that
                            // produced the last token follows the turn snapshot.
                            request.turn = logits.row_host(engine.library(), offset + j).ok();
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
            offset += rows.len();
            // Adapt the draft length to how much of it the model reproduced.
            let drafted = rows.len() - 1;
            let accepted = (request.placement.len() - start).saturating_sub(1);
            request.cycles += 1;
            request.proposed += drafted;
            request.accepted += accepted;
            let censor = if cancelled { Some(Censor::Cancelled) } else {
                resource::finish_censor(finished, accepted + 1, rows, last.map(|(_, token)| token), &request.job.stop_token_ids) };
            outcomes[index] = (rows.len(), accepted + 1, censor);
            let neural = drafted.min(neural_lengths[index]);
            request.outcomes.observe(neural.min(accepted + usize::from(censor.is_none())), accepted.min(neural));
            // A turn snapshot captures the state at the kept length: commit it too.
            kept.push(if !finished || request.turn.is_some() { last } else { None });
            finished
        }).collect();
        let (emit_us, commit_timer) = (console::us(emit_timer), Instant::now());
        // Commit exact accepted state and update the family draft continuation.
        let mut first_row = 0;
        let mut verified: Vec<Verified<'_, _, _, _>> = Vec::with_capacity(active.len());
        for ((((request, rows), (&start, history)), kept), &finished) in active.iter_mut().zip(&sequences)
            .zip(starts.iter().zip(histories)).zip(&kept).zip(&finished) {
            verified.push(Verified { placement: &mut request.placement, seq: &mut request.draft, start, history,
                rows, first_row, kept: *kept, finished });
            first_row += rows.len();
        }
        engine.accept(&mut verified, spec, neural)?;
        drop(verified);
        let commit_us = console::us(commit_timer);
        if let Some(shared) = shared.as_mut() {
            let requests: Vec<_> = active.iter().enumerate().map(|(i, a)| {
                let (rows, accepted, censor) = outcomes[i];
                resource::Verified { id: a.ticket.id(), rows, accepted,
                    source: if rows > 1 { DraftSource::Neural } else { DraftSource::Undrafted },
                    prior: Some(neural_confidence[i].as_slice()), features: None, censor }
            }).collect();
            let times = round_clock.observe();
            if let Some(Err(error)) = engine.probe_finish(|routes, layer_us| resource::observe_drafted(shared,
                &requests, routes, layer_us, times, drafted_width, predicted, Some(drafted_requests))) {
                tracing::debug!(%error, "Qwen draft round not observed");
            }
            resource::publish(shared, shared.stats().rounds, 0, None);
        }
        for (i, request) in active.iter().enumerate() {
            tally.member(&request.ticket, &proposals[i], sequences[i].len() - 1, &request.history[before[i]..],
                request.constraint.is_some(), finished[i]);
        }
        tally.end(|| {
            let engine_now = *engine.profile().borrow();
            let gpu = |i: usize| engine_before.map_or(f64::NAN, |before| 1e6 * (engine_now[i] - before[i]));
            vec![("draft", draft_us), ("verify", 1e6 * elapsed), ("emit", emit_us), ("commit", commit_us),
                ("gpu", gpu(0)), ("experts", gpu(1))]
        });
        if let Some(trace) = trace.as_mut() {
            let contexts: Vec<Vec<usize>> = starts.iter().zip(&sequences)
                .map(|(&start, rows)| (1..=rows.len()).map(|offset| start + offset).collect()).collect();
            trace.cycle(serde_json::json!({"rows": tokens.len(), "seqs": sequences.len(),
                "attention": engine.verify_attention_trace(&contexts,
                    engine.verify_bucket_rows(tokens.len(), spec, diagnostic)),
                "contexts": contexts,
                "ids": active.iter().map(|a| a.id).collect::<Vec<_>>(),
                "depths": sequences.iter().map(|s| s.len() - 1).collect::<Vec<_>>(),
                "kept": kept.iter().map(|k| k.map_or(0, |(n, _)| n)).collect::<Vec<_>>(),
                "verify_ms": 1e3 * elapsed, "draft_steps": timing.steps - steps_before,
                "draft_ms": 1e3 * (timing.seconds - draft_s_before), "cycle_ms": 1e3 * cycle.elapsed().as_secs_f64(),
                "neural_tokens": trace_neural,
                "copy_windows": trace_copy.as_ref().map(|windows| windows.iter().map(|(start, suffix, tokens)|
                    serde_json::json!({"start": start, "matched_suffix": suffix, "tokens": tokens})).collect::<Vec<_>>()),
                "committed_tokens": active.iter().zip(&before).map(|(a, &n)| &a.history[n..]).collect::<Vec<_>>(),
                "predicted_us": predicted, "rates": neural_confidence,
                "draft_policy": resource::snapshot()}));
        }
        for index in (0..active.len()).rev() {
            if !finished[index] {
                continue;
            }
            let mut request = active.remove(index);
            if let Some(shared) = shared.as_mut() { shared.release(request.ticket.id()); }
            request.ticket.done(request.generated);
            requests += 1;
            generated_total += request.generated as u64;
            let seconds = request.started.elapsed().as_secs_f64();
            let phases = std::mem::take(&mut *engine.profile().borrow_mut());
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                active = active.len(), steps, cycles = request.cycles, proposed = request.proposed,
                accepted = request.accepted, tokens_per_cycle = request.generated as f64 / request.cycles.max(1) as f64,
                verify_s, draft_s = timing.seconds, draft_steps = timing.steps, gpu_wait_s = phases[0],
                experts_s = phases[1], "request complete");
            engine.log_table_stats("decode");
            (steps, verify_s, timing) = (0, 0.0, DraftTiming::default());
            if let Some(row) = &request.turn {
                // The conversation so far: every committed row (the last token is not in it).
                let rows = &request.keyed_history[..request.placement.len()];
                let spans = request.placement.spans();
                cache.capture_session(request.ticket.session());
                if let Err(error) = cache.capture_media(&family, SnapshotKind::Turn, rows, spans, &request.placement,
                    After::from_logits(row, true)) {
                    tracing::warn!("turn snapshot not retained: {error:#}");
                }
            }
            release(&family, &mut cache, &mut free_slots, &request.placement, geometry.family);
        }
        cache.tick();
        publish(stats, requests, generated_total, active.len(), prefills.len(), &cache, media, preparer, &verify_stats);
        console::gauges(|| console::Gauges::prefix_cache(&cache, active.len(), prefills.len(), receive.len() + kv_waiter.len()));
        prefills.stepped(cycle.elapsed().as_secs_f64());
    }
}

fn media_failure(error: cuteafd_engine::media::MediaError) -> cuteafd_api::openai::NativeFailure {
    match error {
        cuteafd_engine::media::MediaError::Encoder(_) => cuteafd_api::openai::NativeFailure::Unavailable("vision encoder unavailable".into()),
        error @ (cuteafd_engine::media::MediaError::CacheFull { .. } | cuteafd_engine::media::MediaError::QueueFull) =>
            cuteafd_api::openai::NativeFailure::Unavailable(error.to_string()),
        error => cuteafd_api::openai::NativeFailure::BadRequest(error.to_string()),
    }
}


#[cfg(test)]
mod media_failure_tests {
    #[test]
    fn permanent_image_admission_and_transient_pressure_have_distinct_errors() {
        use cuteafd_engine::media::MediaError;
        let error = MediaError::ImageTooLarge { needed: 5242880, capacity: 2621440 };
        assert!(matches!(super::media_failure(error), cuteafd_api::openai::NativeFailure::BadRequest(message)
            if message == "image needs 5242880 bytes > media cache capacity 2621440"));
        let error = MediaError::CacheFull { needed: 2621440, free: 0, capacity: 5242880 };
        assert!(matches!(super::media_failure(error), cuteafd_api::openai::NativeFailure::Unavailable(message)
            if message.contains("2621440") && message.contains("5242880")));
        assert!(matches!(super::media_failure(MediaError::QueueFull), cuteafd_api::openai::NativeFailure::Unavailable(_)));
    }
    #[test]
    fn encoder_failure_does_not_expose_internal_error() {
        assert!(matches!(super::media_failure(cuteafd_engine::media::MediaError::Encoder("socket died".into())),
            cuteafd_api::openai::NativeFailure::Unavailable(message) if message == "vision encoder unavailable"));
    }
}
