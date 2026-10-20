//! OpenAI-compatible serving for DeepSeek V4: continuous batching over the
//! prefill/decode engine, with resource-priced dSpark speculation.
//!
//! Prefix cache (`cuteafd_engine::prefix` over `super::prefix::Dsv4Prefix`):
//! admission restores the deepest retained snapshot whose tokens prefix the
//! prompt (shared units, the copied tail unit, the window/compressor mark with
//! the dSpark stages' rings) and prefills only the rest; a whole-prompt hit
//! takes its first token from the retained logits. The prompt is retained at
//! prompt end (unless it was a whole hit), the conversation at a normal finish
//! (`Turn`), a prefill whose client left is parked at its last chunk; a decode
//! whose client left is not retained. `prompt_cache_hit_tokens` reports the
//! restored rows; `/v1/stats` carries the cache's counters.
use super::pool::Placement;
use super::prefix::Dsv4Prefix;
use super::{with_engine, EngineArgs};
use crate::shared::prefix::CudaCopyEngine;
use crate::shared::prefix::{PrefixArgs, Toggle};
use crate::shared::probe;
use crate::shared::console;
use crate::shared::draft::{binding::{self, DraftPass, RoundRecord, RoundRequest}, clock::RoundClock,
    evidence::{Censor, DraftSource, Evidence}, resource, runtime::SharedPolicy};
use crate::shared::draft_policy::DraftHistory;
use crate::shared::token_io::{DeviceLogits, RowResult, SelectBatch, TokenSelector};
use crate::shared::prefill_share::{Chunk, DecodeShareArgs};
use cuteafd_engine::prefix::{After, MarkArena, PointPlan, PointPolicy, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{
    InferenceChunk, InferenceFinishReason, ModelEncoding, ModelProfile, NativeFailure, NativeLimits,
    NativeRequest, PromptUsage,
};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    /// Longest sequence; 0 selects checkpoint full, bounded by compiled support and the admitted pool.
    #[arg(long, default_value_t = 0)]
    pub max_context: u32,
    #[arg(long, default_value_t = 4096)]
    pub max_output: u32,
    /// Public model id; defaults to the snapshot's Hugging Face id.
    #[arg(long)]
    pub model_id: Option<String>,
    #[command(flatten)]
    pub decode_share: DecodeShareArgs,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub console: console::ConsoleArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
}

/// "…/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/<rev>" -> "deepseek-ai/DeepSeek-V4-Flash-0731".
fn model_id(snapshot: &Path) -> Option<String> {
    snapshot.ancestors().find_map(|dir| {
        let name = dir.file_name()?.to_str()?.strip_prefix("models--")?;
        let (org, model) = name.split_once("--")?;
        Some(format!("{org}/{model}"))
    })
}

fn eos_token(snapshot: &Path) -> Result<u32> {
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(snapshot.join("generation_config.json"))?)?;
    Ok(config["eos_token_id"].as_u64().context("generation_config.json eos_token_id")? as u32)
}

pub(crate) async fn run_serve(args: ServeArgs) -> Result<()> {
    let api = args.api.load()?;
    let profile = ModelProfile::new(
        args.model_id.clone().or_else(|| model_id(&args.engine.snapshot)).context("model id")?,
        ModelEncoding::DeepseekV4,
    );
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let mut engine_args = args.engine.clone();
    engine_args.max_context = args.max_context as usize;
    let worker_stats = stats.clone();
    let (max_context, decode_share, prefix) =
        (args.max_context as usize, args.decode_share, args.prefix.clone());
    let hub = console::hub(args.console.console_text, || Ok(console_layout(&args, &profile.id)));
    let worker = tokio::task::spawn_blocking(move ||
        serve_loop(engine_args, receive, ready_tx, worker_stats, max_context, decode_share, prefix));
    let max_context = ready_rx.await.context("engine failed before it was ready")??;
    let limits = NativeLimits::new(u32::try_from(max_context)?, args.max_output)?;
    cuteafd_bench::context::phase("engine loaded");
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, Duration::from_secs(25),
        hub.clone(), api.serve(profile.clone(), &args.engine.snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    // The ready ledger (placement gate): one report at readiness, before any request captures
    // lazy graphs or loads more code.
    crate::shared::memory_report::log("ready");
    tracing::info!(listen = %args.listen, model = %profile.id, "DeepSeek V4 API is ready");
    tokio::select! {
        served = axum::serve(listener, api.app(router, hub)
            .into_make_service_with_connect_info::<std::net::SocketAddr>()) => served?,
        _ = crate::shared::api::watch_scheduler(worker) => unreachable!(),
    }
    Ok(())
}

/// What the live console shows for DeepSeek V4 Flash / Pro.
fn console_layout(args: &ServeArgs, model: &str) -> console::Layout {
    use console::{Color::*, StepGroup};
    let mut layout = console::Layout::new("deepseek_v4", model.into(), args.engine.snapshot.clone());
    let sparks = args.engine.peers.split(',').filter(|peer| !peer.is_empty()).count();
    layout.hardware = console::hardware(1 + usize::from(args.engine.split_device.is_some()), sparks,
        args.engine.fixed_onboard());
    layout.split = args.engine.split_device.map(|_| "head split".into());
    layout.concurrency = args.engine.max_sequences;
    layout.eos = eos_token(&args.engine.snapshot).ok().into_iter().collect();
    if args.engine.dspark {
        let block = std::fs::read_to_string(args.engine.snapshot.join("config.json")).ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .and_then(|config| config["dspark_block_size"].as_u64()).unwrap_or(8) as usize;
        layout.speculator = Some(console::Speculator { name: "dSpark".into(), positions: block.clamp(1, 16),
            policy: "shared resource-priced drafts and agreeing copies".into() });
    }
    layout.steps = vec![
        StepGroup::new("Decode step", "host clock", &[("round.cycle", "step", Target),
            ("draft", "dSpark draft", Accepted), ("verify", "verify pass + token selection", Target),
            ("emit", "accept + stream", Ink)]),
        StepGroup::new("Verify pass", "host clock, engine phases", &[
            ("router_sync", "GPU through router + quantizer", Rtx), ("routing", "route packing", Ink),
            ("experts", "Spark expert exchanges", Spark), ("head", "head + logits", Target)]),
        StepGroup::layers(),
        StepGroup::admission(false),
    ];
    layout.layers = Some(console::Layers::host_clock());
    layout
}

/// What a decode step did, for the console: each sequence's draft and
/// verified draft rows, and the draft and verify host time.
#[derive(Default)]
struct StepShape {
    drafts: Vec<Vec<u32>>,
    verified: Vec<usize>,
    draft_us: f64,
    verify_us: f64,
}

#[allow(clippy::too_many_arguments)]
fn serve_loop(
    args: EngineArgs,
    mut receive: mpsc::Receiver<NativeRequest>,
    ready: tokio::sync::oneshot::Sender<Result<usize>>,
    stats: Arc<Mutex<serde_json::Value>>,
    max_context: usize,
    decode_share: DecodeShareArgs,
    prefix: PrefixArgs,
) -> Result<()> {
    let loaded = match super::load(&args) {
        Ok(loaded) => loaded,
        Err(error) => {
            let _ = ready.send(Err(anyhow::anyhow!("{error:#}")));
            return Ok(());
        }
    };
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(&args.snapshot)?;
    let eos = eos_token(&args.snapshot)?;
    let mut ready = Some(ready);
    // The mark arena is allocated before local experts fill the device.
    let arena = |engine: &super::engine::Engine<'_>| -> Result<usize> {
        if args.dspark && !engine.skip_routed && engine.weights.dspark.is_some() { engine.arm_draft_probe()?; }
        let mark = Dsv4Prefix::mark_bytes_of(engine)?;
        Ok(mark_slots(engine, &prefix, mark) * mark)
    };
    let result = with_engine(&loaded, &args, Some(&prefix), arena, |engine, transports, runtime| {
        if engine.skip_routed {
            tracing::warn!("serve-dsv4 --skip-routed-experts: replies do not match the model (plumbing and cache gates only)");
        }
        if max_context > engine.max_context {
            tracing::warn!(requested = max_context, supported = engine.max_context,
                "--max-context exceeds the exported programs; requests are limited to the programs' context");
        }
        let shared = super::draft_binding::new(engine, &loaded, transports)?;
        let local = engine.local_layers();
        console::layer_classes((0..engine.weights.layers.len()).map(|l| console::layer_class(false, l >= local)).collect());
        let mut selector = TokenSelector::new(&loaded.library, args.token_io.token_select, engine.cfg.vocab_size,
            engine.decode_rows)?;
        if let Some(ready) = ready.take() {
            let _ = ready.send(Ok(engine.max_context));
        }
        schedule(engine, &loaded, &tokenizer, eos, &mut receive, transports, runtime, &stats,
            decode_share, &prefix, &mut selector, shared)
    });
    if let Some(ready) = ready.take() {
        let _ = ready.send(result.as_ref().map(|_| args.max_context).map_err(|e| anyhow::anyhow!("{e:#}")));
    }
    result
}

/// An admitted prompt waiting for its remaining prefill chunks (from the
/// prefix cache's restore point; the last one returns the logits).
struct Prefill<'a> {
    job: NativeRequest,
    constraint: Option<crate::shared::constraints::State<'a>>,
    tokens: Vec<u32>,
    /// Rows restored from the prefix cache.
    resume: usize,
    /// Chunk ends (equal chunks of at most two prefill lanes) and intermediate
    /// snapshot points (`cuteafd_engine::prefix::plan_points`).
    plan: PointPlan,
    /// Chunks prefilled so far.
    chunks: usize,
    /// The client left mid-prefill: its prefilled rows are parked as a prompt snapshot.
    cancelled: bool,
    placement: Placement,
    capacity: usize,
    /// A whole-prompt prefix hit's retained logits (its first token is selected from them).
    logits: Option<Vec<f32>>,
    /// The first generated token, selected after the last chunk, and that
    /// row's logits when a prompt snapshot will keep them.
    first: Option<u32>,
    prompt_row: Option<Vec<f32>>,
    started: Instant,
    /// Seconds in this prompt's chunks.
    busy: f64,
    ticket: console::Ticket,
}

/// One admitted request: its placement, stream state and next input token.
struct Active<'a> {
    job: NativeRequest,
    /// Grammar for structured output and tool calls.
    constraint: Option<crate::shared::constraints::State<'a>>,
    placement: Placement,
    capacity: usize,
    next: u32,
    /// Prompt and emitted tokens: the rows of the placement, then `next`.
    history: Vec<u32>,
    /// The logit row that produced the last token, once the request finished
    /// normally (EOS or max_tokens with the client still there): what follows
    /// its `Turn` snapshot.
    turn: Option<Vec<f32>>,
    decoder: cuteafd_loader::StreamingTokenDecoder,
    generated: usize,
    draft_calls: usize,
    drafts: DraftHistory,
    buffered: usize,
    started: Instant,
    ticket: console::Ticket,
}

impl Active<'_> {
    fn send(&self, chunk: InferenceChunk) -> Result<()> {
        self.job.events.send(Ok(chunk)).map_err(|_| anyhow::anyhow!("client went away"))
    }

    /// Streams `token`; returns true when the request is finished.
    fn emit(&mut self, token: u32, eos: u32) -> Result<bool> {
        probe::token(&self.job.probe, token);
        self.history.push(token);
        self.generated += 1;
        self.buffered += 1;
        // A grammar that accepted its stop token has ended the request.
        let stop = token == eos || self.constraint.as_ref().is_some_and(|state| state.terminated());
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

/// Emits `token` for `request`; on a normal finish with `caching`, keeps the
/// logit row that produced it for the turn snapshot. Returns whether the
/// request finished (normally, or because the client left).
fn finish_row(request: &mut Active<'_>, token: Result<u32>, eos: u32, logits: &DeviceLogits, row: usize,
    caching: bool, library: &cuteafd_ffi::NativeLibrary) -> bool {
    finish_row_status(request, token, eos, logits, row, caching, library).0
}

/// Preserve failures during decoding/streaming so they cannot train a rejection.
fn finish_row_status(request: &mut Active<'_>, token: Result<u32>, eos: u32, logits: &DeviceLogits, row: usize,
    caching: bool, library: &cuteafd_ffi::NativeLibrary) -> (bool, bool) {
    probe::decode_row(library, &request.job.probe, logits, row, request.generated, request.history.len());
    match token.and_then(|token| request.emit(token, eos)) {
        Ok(false) => (false, false),
        Ok(true) => {
            if caching && !probe::cold(&request.job.probe) {
                request.turn = logits.row_host(library, row).ok();
            }
            (true, false)
        }
        Err(error) => {
            // The request fails alone, with its cause on the stream.
            let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
            (true, true)
        }
    }
}

fn round_censor(done: bool, failed: bool, committed: usize, rows: &[u32], token: Option<u32>, eos: u32) -> Option<Censor> {
    if failed {
        // A selected mismatch was judged even if streaming it failed.
        if token.is_some() && rows.get(committed).is_some_and(|next| Some(*next) != token) { None }
        else { Some(Censor::Cancelled) }
    } else { resource::finish_censor(done, committed, rows, token, &[eos]) }
}

/// Each row draws at the position after it, masked along its sequence's
/// drafts. A sequence whose grammar fails gets its error (and unmasked rows):
/// it fails alone, never the batch.
fn select_rows(selector: &mut TokenSelector<'_>, logits: &crate::shared::token_io::DeviceLogits, active: &[Active<'_>],
    sequences: &[Vec<u32>], starts: &[usize]) -> Result<(Vec<RowResult>, Vec<Option<String>>)> {
    let mut batch = SelectBatch::default();
    let poisoned = active.iter().zip(sequences).zip(starts).map(|((a, rows), &start)|
        batch.push_sequence_isolated(a.job.sampling, a.constraint.as_ref(), rows, start as u64 + 1)).collect();
    Ok((selector.select(logits, &batch)?, poisoned))
}

fn action_rooms(limits: &[usize], width: usize, rows: usize, start: usize) -> Vec<usize> {
    let mut rooms = vec![0; limits.len()];
    if width == 0 || limits.is_empty() { return rooms; }
    let mut remaining = rows / width;
    // Runtime workspace can fit fewer native than narrow participants. Rotate
    // the subset so workspace exclusion cannot starve an active request.
    for offset in 0..limits.len() {
        let index = (start % limits.len() + offset) % limits.len();
        if limits[index] == 0 || remaining == 0 { continue; }
        rooms[index] = limits[index].min(width);
        remaining -= 1;
    }
    rooms
}

fn cold_action(widths: &[usize], rooms: &[Vec<usize>], preferred: usize) -> usize {
    let fits = |index: usize| rooms[index].iter().any(|&room| room > 0);
    widths.iter().position(|&width| width == preferred).filter(|&index| fits(index))
        .or_else(|| (0..rooms.len()).find(|&index| fits(index))).unwrap_or(0)
}

/// One global verifier budget, including requests that did not draft.
#[allow(clippy::too_many_arguments)]
fn speculative_step(
    engine: &super::engine::Engine<'_>,
    active: &mut [Active<'_>],
    shared: &mut SharedPolicy,
    observed: &mut u64,
    skipped: &mut u64,
    eos: u32,
    transports: &mut [crate::shared::spark_intake::SparkLink<'_>],
    runtime: &tokio::runtime::Runtime,
    selector: &mut TokenSelector<'_>,
    caching: bool,
    shape: &mut StepShape,
    cycle: Instant,
) -> Result<Vec<bool>> {
    let mut clock = RoundClock::at(cycle);
    let limits: Vec<usize> = active.iter().map(|a| if probe::no_speculation(&a.job.probe) { 0 } else {
        engine.decode_rows.saturating_sub(active.len())
            .min(a.job.max_tokens.saturating_sub(a.generated + 1))
            .min(a.capacity.saturating_sub(a.placement.len + 2))
    }).collect();
    let requests: Vec<_> = active.iter().zip(&limits).map(|(request, &room)| (request.ticket.id(), room)).collect();
    let widths: Vec<_> = std::iter::once(0).chain(shared.policy.widths().iter().copied()).collect();
    let rooms: Vec<_> = widths.iter().map(|&width| action_rooms(&limits, width, engine.decode_rows,
        observed.wrapping_add(*skipped) as usize)).collect();
    let actions: Vec<_> = widths.iter().zip(&rooms).map(|(&width, rooms)| cuteafd_core::DraftAction {
        width, draft_us: shared.policy.draft_cost_drafted(false, active.len(),
            rooms.iter().filter(|&&room| room > 0).count(), width),
    }).collect();
    let masks: Vec<_> = rooms.iter().map(Vec::as_slice).collect();
    let priced = shared.policy.choose_action_with_rooms(false, &requests, &actions, &masks, engine.decode_rows)?;
    let chosen = priced.unwrap_or_else(|| {
        // Collect installed width fits before pricing can choose skip, but do
        // not discard an eligible narrow pass when the preferred width cannot fit.
        cold_action(&widths, &rooms, shared.policy.choose_width(false, &requests))
    });
    binding::trace_action(&shared.policy, false, &requests, &actions, Some(&masks), engine.decode_rows,
        priced, chosen);
    let block = actions[chosen].width;
    let noise = engine.cfg.dspark_noise_token_id as u32;
    let draft_indices: Vec<usize> = rooms[chosen].iter().enumerate().filter_map(|(i, &room)|
        (room > 0).then_some(i)).collect();
    let mut drafts = vec![Vec::new(); active.len()];
    if !draft_indices.is_empty() {
        let inputs: Vec<u32> = draft_indices.iter().flat_map(|&i|
            std::iter::once(active[i].next).chain(std::iter::repeat_n(noise, block - 1))).collect();
        let requests: Vec<super::engine::DraftRequest<'_>> = draft_indices.iter().map(|&i|
            super::engine::DraftRequest { placement: &active[i].placement, token: active[i].next }).collect();
        let proposed = clock.draft(|| engine.draft_width(&requests, &inputs, block))?;
        shape.draft_us = clock.observe().draft_us.unwrap_or(0) as f64;
        ensure!(proposed.len() == draft_indices.len(), "dSpark proposal count differs from participating requests");
        for (&i, proposal) in draft_indices.iter().zip(proposed) {
            ensure!(proposal.len() == block, "dSpark proposal width differs from installed block");
            active[i].draft_calls += 1;
            drafts[i] = proposal;
        }
    }
    let width = if draft_indices.is_empty() { 0 } else { block };
    let mut priors: Vec<Vec<f64>> = active.iter().enumerate().map(|(i, a)| {
        let rates = a.drafts.pooled(drafts[i].len(), active.iter().enumerate()
            .filter_map(|(j, other)| (i != j).then_some(&other.drafts)));
        shared.prior(&a.drafts, &resource::DraftEvidence::History(&rates), 0)
    }).collect();
    let copies: Vec<_> = active.iter().zip(&drafts).zip(&limits).zip(&mut priors)
        .map(|(((a, draft), &limit), prior)| resource::copy_continuation(&shared.policy, active.len(), a.ticket.id(),
            &a.history, draft, prior, limit, a.job.sampling.is_greedy())).collect();
    let candidates: Vec<_> = active.iter().zip(&priors).zip(&limits).map(|((a, prior), &limit)|
        resource::Candidate { id: a.ticket.id(), prior, limit, selector: false, cold: a.drafts.cold() }).collect();
    let selection = resource::select_drafted(&mut shared.policy, &candidates, width, engine.decode_rows,
        draft_indices.len());
    let sequences: Vec<Vec<u32>> = active.iter().enumerate().map(|(i, a)| {
        let proposal = copies[i].as_ref().map_or(drafts[i].as_slice(), |copy| copy.tokens.as_slice());
        let mut rows: Vec<u32> = std::iter::once(a.next)
            .chain(proposal.iter().copied().take(selection.lengths[i])).collect();
        if let Some(state) = a.constraint.as_ref() { state.truncate_proposal(&mut rows)?; }
        ensure!(!rows.is_empty(), "grammar removed V4 verifier anchor");
        Ok(rows)
    }).collect::<Result<_>>()?;
    let total_rows = sequences.iter().map(Vec::len).sum();
    ensure!(total_rows <= engine.decode_rows, "shared V4 verifier budget exceeded");
    // Prediction follows executed rows, not the pre-grammar selection.
    let ids: Vec<_> = active.iter().map(|a| a.ticket.id()).collect();
    let lengths: Vec<_> = sequences.iter().map(|rows| rows.len() - 1).collect();
    let predicted = shared.policy.predict_drafted(false, &ids, &lengths, width, draft_indices.len());
    let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
    let mut rows: Vec<(&mut Placement, &[u32])> = active.iter_mut().zip(&sequences)
        .map(|(a, tokens)| (&mut a.placement, tokens.as_slice())).collect();
    let timer = Instant::now();
    engine.probe_begin(total_rows);
    let verified = engine.verify_device(&mut rows, transports, runtime).and_then(|logits| {
        let (selected, poisoned) = select_rows(selector, &logits, active, &sequences, &starts)?;
        engine.check_device()?;
        Ok((logits, selected, poisoned))
    });
    engine.probe_end(verified.is_ok());
    let (logits, selected, mut poisoned) = verified?;
    shape.verify_us = console::us(timer);
    shape.verified = lengths;
    let mut offset = 0;
    let mut outcomes = Vec::with_capacity(active.len());
    let finished: Vec<_> = active.iter_mut().zip(&sequences).zip(starts).enumerate().map(|(i, ((request, rows), start))| {
        if let Some(error) = poisoned[i].take() {
            request.placement.len = start + 1;
            outcomes.push((1, Some(Censor::Cancelled)));
            let done = finish_row(request, Err(anyhow::anyhow!(error)), eos, &logits, offset, caching, engine.library);
            offset += rows.len();
            return done;
        }
        let (mut done, mut censor) = (false, None);
        for j in 0..rows.len() {
            request.placement.len = start + j + 1;
            let token = take(request.constraint.as_mut(), &selected[offset + j]);
            let kept = token.as_ref().ok().copied();
            let failed;
            (done, failed) = finish_row_status(request, token, eos, &logits, offset + j, caching, engine.library);
            censor = round_censor(done, failed, j + 1, rows, kept, eos);
            if done || rows.get(j + 1) != kept.as_ref() { break; }
        }
        outcomes.push((request.placement.len - start, censor));
        offset += rows.len();
        done
    }).collect();
    let times = clock.observe();
    let round: Vec<_> = active.iter().enumerate().map(|(i, request)| {
        let (accepted, censor) = outcomes[i];
        let neural = copies[i].as_ref().map_or(drafts[i].len(), |copy| copy.neural);
        let copied = sequences[i].len() - 1 > neural;
        RoundRequest { id: request.ticket.id(), rows: sequences[i].len(), accepted,
            source: if copied { DraftSource::Copy } else if drafts[i].is_empty() { DraftSource::Undrafted } else { DraftSource::Neural },
            evidence: (!drafts[i].is_empty()).then_some(Evidence::History(&priors[i][..neural.min(priors[i].len())])), censor }
    }).collect();
    let result = engine.probe_finish(|routes, layer_us| binding::observe(&mut shared.policy, &RoundRecord {
        shared: false, requests: &round, routes, layer_us, times,
        draft: (width > 0).then_some(DraftPass { width, requests: Some(draft_indices.len()) }), predicted,
    }));
    match result {
        Some(Ok(())) => *observed += 1,
        Some(Err(error)) => { *skipped += 1; tracing::debug!(%error, "V4 draft policy round not observed"); }
        None => *skipped += 1,
    }
    for (i, request) in active.iter_mut().enumerate() {
        let (accepted, censor) = outcomes[i];
        let neural = copies[i].as_ref().map_or(drafts[i].len(), |copy| copy.neural).min(sequences[i].len() - 1);
        let accepted = accepted.saturating_sub(1).min(neural);
        request.drafts.observe(neural.min(accepted + usize::from(censor.is_none())), accepted);
    }
    resource::publish(&shared.policy, *observed, *skipped, None);
    shape.drafts = sequences.iter().map(|rows| rows[1..].to_vec()).collect();
    Ok(finished)
}

/// Host staging slots, or legacy retention when this layout cannot use the host tier.
fn mark_slots(engine: &super::engine::Engine<'_>, args: &PrefixArgs, mark: usize) -> usize {
    let mut rule = args.mark_rule(engine.shape.sequences);
    rule.host = cuteafd_loader::placement::families::deepseek_v4::host_marks_enabled(
        engine.ranks(), engine.context_kv(), rule.host);
    rule.slots(mark)
}

/// The prefix cache over `engine` (always present: with zero entries it is the unit allocator).
fn prefix_cache<'e, 'a>(engine: &'e super::engine::Engine<'a>, args: &PrefixArgs)
    -> Result<(Dsv4Prefix<'e, 'a>, PrefixCache<CudaCopyEngine<'a>>)> {
    anyhow::ensure!(args.prefix_partial == Toggle::Off,
        "DeepSeek V4 restores exact snapshots only (window and compressor state)");
    let args = &args.with_scoring(if engine.full_prefill_logits { engine.prefill_capacity() } else { 64 },
        engine.prefill_rows, engine.cfg.vocab_size)?;
    let entries = args.prefix_cache_entries;
    let family = Dsv4Prefix::new(engine, |mark| mark_slots(engine, args, mark))?;
    // The pinned host tier copies through one GPU's copy engine; a heads split keeps its
    // (replicated) state on both GPUs, so it keeps device-resident snapshots only. Under context
    // each unit's C4 pages come from their owner and the replicated bytes from GPU0 (both GPUs'
    // pools and mark arenas registered, so every copy runs on its owner's copy streams).
    let host = if !cuteafd_loader::placement::families::deepseek_v4::host_marks_enabled(
        engine.ranks(), engine.context_kv(), true) {
        if args.host_cache_bytes.enabled() && args.prefix_cache_entries > 0 {
            tracing::warn!("DeepSeek V4 head split: the prefix cache's host tier is off (device-resident snapshots only)");
        }
        None
    } else if engine.ranks() > 1 {
        // SAFETY of the registration: the engine (and its pools) outlives the cache, which
        // `schedule` drops first.
        args.host_config(family.layout(), engine.max_context, engine.shape.sequences)?.map(|config|
            crate::shared::prefix::CudaCopyEngine::registered(engine.library, &family.buffers())
                .map(|copy| (config, copy))).transpose()?
    } else {
        args.host_tier(engine.library, family.template(), family.layout(), engine.max_context, engine.shape.sequences)?
    };
    let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
    let layout = family.layout();
    let config = PrefixConfig { entries, mark_slots: family.slots(), keep_logits: true,
        min_tokens: args.prefix_cache_min_tokens };
    let cache = PrefixCache::new(layout, config, host)?;
    tracing::info!(entries, mark_slots = family.slots(), mark_bytes = family.mark_bytes(), page_bytes = layout.page_bytes,
        pages = layout.pages, page_rows = layout.page_rows, host_bytes, points = ?args.points(),
        "DeepSeek V4 prefix cache");
    cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64, layout.pages as u64,
        &"compressed C4/C128 + index".to_string(), host_bytes);
    Ok((family, cache))
}

/// Gives a finished or failed sequence's units and state slot back.
fn release(family: &Dsv4Prefix<'_, '_>, cache: &mut PrefixCache<CudaCopyEngine<'_>>, states: &mut Vec<usize>,
    placement: &Placement) {
    if let Err(error) = cache.release(family, &placement.units) {
        tracing::error!(%error, "releasing a DeepSeek V4 sequence's units");
    }
    states.push(placement.state);
}

/// Serving statistics for `/v1/stats`.
fn publish(stats: &Mutex<serde_json::Value>, requests: u64, generated: u64, active: usize, prefilling: usize,
    cache: &PrefixCache<CudaCopyEngine<'_>>) {
    if let Ok(mut stats) = stats.lock() {
        *stats = serde_json::json!({"requests": requests, "generated_tokens": generated, "active": active,
            "prefilling": prefilling, "prefix_cache": cache.stats(), "draft_policy": resource::snapshot()});
    }
}

/// Message starts of the DeepSeek V4 chat template: a snapshot right before one is a message boundary.
pub(crate) const MESSAGE_STARTS: [&str; 2] = ["<｜User｜>", "<｜Assistant｜>"];

/// Continuous batching: prefill each new request (one sequence per prefill),
/// then advance every active sequence by one token in a single decode step.
#[allow(clippy::too_many_arguments)]
fn schedule(
    engine: &super::engine::Engine<'_>,
    loaded: &super::Loaded,
    tokenizer: &cuteafd_loader::LoadedTokenizer,
    eos: u32,
    receive: &mut mpsc::Receiver<NativeRequest>,
    transports: &mut [crate::shared::spark_intake::SparkLink<'_>],
    runtime: &tokio::runtime::Runtime,
    stats: &Mutex<serde_json::Value>,
    decode_share: DecodeShareArgs,
    prefix: &PrefixArgs,
    selector: &mut TokenSelector<'_>,
    mut shared: Option<SharedPolicy>,
) -> Result<()> {
    let (mut observed, mut skipped) = (0u64, 0u64);
    let (family, mut cache) = prefix_cache(engine, prefix)?;
    let markers = crate::shared::prefix::marker_ids(&loaded.snapshot, &MESSAGE_STARTS)?;
    let mut states: Vec<usize> = (0..engine.shape.sequences).rev().collect();
    let mut grammars = crate::shared::constraints::Compiler::new(
        &loaded.library, loaded.snapshot.join("tokenizer.json"), engine.cfg.vocab_size);
    let mut active: Vec<Active<'_>> = Vec::new();
    let (mut requests, mut generated_total) = (0u64, 0u64);
    let chunk_limit = engine.prefill_capacity().min(engine.max_context);
    let mut prefills = decode_share.queue::<Prefill<'_>>()?;
    let mut kv_waiter = cuteafd_engine::prefix::DeferredAdmission::<NativeRequest>::default();
    loop {
        engine.require_live()?;
        if let Some(reason) = cuteafd_transport::health::failure_reason() {
            anyhow::bail!("expert wire unavailable until restart: {reason}");
        }
        // Admit while sequence slots and decode rows remain.
        while !states.is_empty() && active.len() + prefills.len() < engine.decode_rows {
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
            if !job.images.is_empty() {
                reject(&job, "this checkpoint takes no images".into());
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
            let state = states.pop().expect("checked");
            cache.tick();
            let admit_started = Instant::now();
            // Lookup, fork of the retained units and restore of the mark (byte-exact).
            let lookup: &[u32] = if cold { &[] } else { &tokens };
            let admitted = match cache.admit(&family, lookup, capacity, true, |units| Placement::new(state, units)) {
                Ok(admitted) => admitted,
                Err(error) => {
                    states.push(state);
                    // Running requests keep their pages pinned. Delay a request
                    // that fits alone instead of rejecting transient KV pressure.
                    match kv_waiter.defer(job, &error, busy, cache.pool().release_epoch()) {
                        Ok(()) => break,
                        Err(job) => reject(&job, format!("{error:#}")),
                    }
                    continue;
                }
            };
            let resume = admitted.resume;
            job.prefix_cache_source.set(resume, admitted.source.as_ref().is_some_and(|source| source.host));
            probe::admitted(&job.probe, "deepseek_v4", &tokens, resume);
            let _ = job.events.send(Ok(InferenceChunk::Ready {
                system_fingerprint: None,
                prompt_usage: PromptUsage { prompt_tokens: tokens.len(), prompt_cache_hit_tokens: resume },
            }));
            if let Some(from) = probe::scoring(&job.probe) {
                // Teacher-forced scoring: every row's logits, no generation, nothing retained.
                let mut placement = admitted.placement;
                let scored = probe::score(&loaded.library, &job.probe, &tokens, from, chunk_limit, engine.decode_rows, probe::verify_rows(&job.probe), engine.full_prefill_logits,
                    &mut (&mut placement, &mut *transports),
                    |(placement, transports), chunk, rows| {
                        if rows > 1 {
                            Ok(Some(probe::ScoreLogits::Host {
                                values: engine.prefill(placement, chunk, transports, runtime, rows, None)?,
                                vocab: engine.cfg.vocab_size }))
                        } else {
                            Ok(engine.prefill_device(placement, chunk, transports, runtime, rows)?
                                .map(probe::ScoreLogits::Device))
                        }
                    },
                    |(placement, transports), chunk| engine.verify_device(&mut [(&mut **placement, chunk)], transports,
                        runtime));
                match scored {
                    Ok(_) => { let _ = job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })); }
                    Err(error) => { let _ = job.events.send(Err(NativeFailure::Worker(format!("scoring: {error:#}")))); }
                }
                release(&family, &mut cache, &mut states, &placement);
                continue;
            }
            let ticket = console::admit(tokens.len(), resume, job.max_tokens, constraint.is_some(), job.images.len(),
                admit_started, job.usage.clone());
            if let (Some(usage), Some(session)) = (&job.usage, admitted.source.as_ref().and_then(|s| s.session.as_ref())) {
                usage.session(session.clone(), "prefix");
            }
            if let Some(source) = admitted.source {
                tracing::info!(tokens = tokens.len(), resume, kind = ?source.kind, frontier = source.frontier,
                    host = source.host, "prefix cache hit");
            }
            // A whole-prompt hit brings its first token's logits: nothing to prefill.
            let logits = admitted.after.and_then(|after| after.logits).map(|logits| logits.to_vec());
            let plan = if resume == tokens.len() {
                PointPlan::default()
            } else {
                // Equal chunks, so no lane runs a tiny tail.
                let remaining = tokens.len() - resume;
                let limit = remaining.div_ceil(remaining.div_ceil(chunk_limit));
                // A cold probe keeps the same chunk plan (identical numerics); it only skips the captures.
                if cache.enabled() {
                    cuteafd_engine::prefix::plan_points(resume, tokens.len(), limit,
                        &crate::shared::prefix::boundaries(&tokens, &markers), family.capture_reach(),
                        prefix.prefix_cache_min_tokens, prefix.points())
                } else {
                    cuteafd_engine::prefix::plan_points(resume, tokens.len(), limit, &[], 0, 0,
                        PointPolicy { gap: 0, boundaries: 0, per_request: 0 })
                }
            };
            prefills.push(Prefill { job, constraint, tokens, resume, plan, chunks: 0, cancelled: false,
                placement: admitted.placement, capacity, logits, first: None, prompt_row: None,
                started: Instant::now(), busy: 0.0, ticket });
        }
        if prefills.due(!active.is_empty()) {
            let caching = cache.enabled();
            // One chunk of each waiting prompt (whole prompts with --decode-share 0).
            let finished = prefills.round(|p| {
                if p.placement.len == p.tokens.len() {
                    return Ok(Chunk::Done);
                }
                if p.job.events.is_closed() {
                    p.cancelled = true;
                    anyhow::bail!("client went away");
                }
                let timer = Instant::now();
                let end = p.plan.chunks.get(p.chunks).copied().unwrap_or(p.tokens.len());
                let last = end == p.tokens.len();
                let chunk = &p.tokens[p.placement.len..end];
                let rows = chunk.len();
                let logits = engine.prefill_device(&mut p.placement, chunk, transports, runtime, usize::from(last))?;
                if last {
                    // The first token, while this prompt's logits are the workspace's.
                    let logits = logits.context("prefill produced no logits")?;
                    if caching && !probe::cold(&p.job.probe) {
                        p.prompt_row = Some(logits.row_host(&loaded.library, 0)?);
                    }
                    if probe::wants_first(&p.job.probe) {
                        probe::device_rows(&loaded.library, &p.job.probe, &logits, 0, 1, p.tokens.len())?;
                    }
                    let mut batch = SelectBatch::default();
                    batch.push_next(p.job.sampling, p.constraint.as_mut(), p.placement.len as u64)?;
                    let selected = selector.select(&logits, &batch)?;
                    p.first = Some(take(p.constraint.as_mut(), &selected[0])?);
                }
                p.busy += timer.elapsed().as_secs_f64();
                p.ticket.prefill(rows, p.chunks, p.plan.chunks.len(), timer);
                // Intermediate snapshot points this chunk ends at (off unless configured).
                for &(_, point) in p.plan.points.iter().filter(|&&(chunk, _)| chunk == p.chunks && !probe::cold(&p.job.probe)) {
                    cache.capture_session(p.ticket.session());
                    if let Err(error) = cache.capture(&family, SnapshotKind::Prompt, &p.tokens[..point], &p.placement,
                        After::default()) {
                        tracing::warn!("snapshot point {point} not retained: {error:#}");
                    }
                }
                p.chunks += 1;
                Ok(if p.placement.len == p.tokens.len() { Chunk::Done } else { Chunk::More })
            });
            for (mut p, prefilled) in finished {
                let (placement, resume) = (p.placement.clone(), p.resume);
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
                    release(&family, &mut cache, &mut states, &placement);
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
                        let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        release(&family, &mut cache, &mut states, &placement);
                        continue;
                    }
                };
                if resume < p.tokens.len() {
                    tracing::debug!(tokens = p.tokens.len(), cached = resume,
                        elapsed_ms = p.started.elapsed().as_millis() as u64, busy_ms = (1e3 * p.busy) as u64, "prefill");
                }
                // The prompt snapshot, taken once the first token is out (it only enqueues copies).
                let prompt_row = p.prompt_row.take();
                let session = p.ticket.session();
                let retain_prompt = |cache: &mut PrefixCache<CudaCopyEngine<'_>>, tokens: &[u32], placement: &Placement| {
                    if let Some(row) = &prompt_row {
                        cache.capture_session(session.clone());
                        if let Err(error) = cache.capture(&family, SnapshotKind::Prompt, tokens, placement,
                            After::from_logits(row, true)) {
                            tracing::warn!("prompt snapshot not retained: {error:#}");
                        }
                    }
                };
                let decoder = match cuteafd_loader::streaming_token_decoder(&loaded.snapshot, false) {
                    Ok(decoder) => decoder,
                    Err(error) => {
                        tracing::warn!("admission failed: {error:#}");
                        let _ = p.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        retain_prompt(&mut cache, &p.tokens, &placement);
                        release(&family, &mut cache, &mut states, &placement);
                        continue;
                    }
                };
                let mut request = Active {
                    decoder, job: p.job, constraint: p.constraint, placement: p.placement, capacity: p.capacity,
                    next: first, history: p.tokens, turn: None, generated: 0, draft_calls: 0, buffered: 0, started: Instant::now(),
                    ticket: p.ticket, drafts: DraftHistory::default(),
                };
                let emitted = request.emit(first, eos);
                request.ticket.first(first);
                retain_prompt(&mut cache, &request.history[..request.placement.len], &request.placement);
                match &emitted {
                    Ok(false) => active.push(request),
                    // Finished at its first token: its turn is its prompt snapshot.
                    Ok(true) | Err(_) => {
                        if let Err(error) = &emitted {
                            let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                        }
                        request.ticket.done(request.generated);
                        release(&family, &mut cache, &mut states, &request.placement)
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
        let engine_before = tally.live().then(|| engine.profile.borrow().seconds);
        let before: Vec<usize> = active.iter().map(|a| a.history.len()).collect();
        let mut shape = StepShape::default();
        let caching = cache.enabled();
        // Undrafted rounds use the same route and resource observations.
        let step = if let Some(shared) = shared.as_mut() {
            speculative_step(engine, &mut active, shared, &mut observed, &mut skipped, eos,
                transports, runtime, selector, caching, &mut shape, cycle)
        } else {
            let sequences: Vec<Vec<u32>> = active.iter().map(|a| vec![a.next]).collect();
            let starts: Vec<usize> = active.iter().map(|a| a.placement.len).collect();
            let mut rows: Vec<(&mut Placement, u32)> = active.iter_mut().map(|a| (&mut a.placement, a.next)).collect();
            let timer = Instant::now();
            engine.decode_device(&mut rows, transports.first_mut(), runtime).and_then(|logits| {
                let (selected, mut poisoned) = select_rows(selector, &logits, &active, &sequences, &starts)?;
                engine.check_device()?;
                shape.verify_us = console::us(timer);
                Ok(active.iter_mut().zip(&selected).enumerate().map(|(row, (request, selected))| {
                    let token = match poisoned[row].take() {
                        Some(error) => Err(anyhow::anyhow!(error)),
                        None => take(request.constraint.as_mut(), selected),
                    };
                    finish_row(request, token, eos, &logits, row, caching, engine.library)
                }).collect::<Vec<bool>>())
            })
        };
        let finished = match step {
            Ok(finished) => finished,
            Err(error) => {
                tracing::warn!("decode step failed: {error:#}");
                for request in active.drain(..) {
                    if let Some(shared) = shared.as_mut() { shared.policy.release(request.ticket.id()); }
                    let _ = request.job.events.send(Err(NativeFailure::Worker(format!("{error:#}"))));
                    release(&family, &mut cache, &mut states, &request.placement);
                }
                continue;
            }
        };
        for (i, request) in active.iter().enumerate() {
            let proposal = shape.drafts.get(i).map_or(&[][..], Vec::as_slice);
            tally.member(&request.ticket, proposal, shape.verified.get(i).copied().unwrap_or(0),
                &request.history[before[i]..], request.constraint.is_some(), finished[i]);
        }
        tally.end(|| {
            let engine_now = engine.profile.borrow().seconds;
            let phase = |i: usize| engine_before.map_or(f64::NAN, |before| 1e6 * (engine_now[i] - before[i]));
            let emit = 1e6 * cycle.elapsed().as_secs_f64() - shape.draft_us - shape.verify_us;
            vec![("draft", shape.draft_us), ("verify", shape.verify_us), ("emit", emit),
                ("router_sync", phase(0)), ("routing", phase(1)), ("experts", phase(2)), ("head", phase(3))]
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
            tracing::info!(tokens = request.generated, seconds, tok_s = request.generated as f64 / seconds,
                draft_calls = request.draft_calls,
                active = active.len(), "request complete");
            if let Some(row) = &request.turn {
                // The conversation so far: every committed row (the last token is not in it).
                let rows = &request.history[..request.placement.len];
                cache.capture_session(request.ticket.session());
                if let Err(error) = cache.capture(&family, SnapshotKind::Turn, rows, &request.placement,
                    After::from_logits(row, true)) {
                    tracing::warn!("turn snapshot not retained: {error:#}");
                }
            }
            release(&family, &mut cache, &mut states, &request.placement);
        }
        cache.tick();
        publish(stats, requests, generated_total, active.len(), prefills.len(), &cache);
        console::gauges(|| console::Gauges::prefix_cache(&cache, active.len(), prefills.len(), receive.len() + kv_waiter.len()));
        prefills.stepped(cycle.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_room_masks_use_runtime_width_capacity_and_skip_ineligible_requests() {
        let limits = [5; 16];
        assert_eq!(action_rooms(&limits, 0, 64, 0), [0; 16]);
        assert_eq!(action_rooms(&limits, 3, 64, 0), [3; 16]);
        assert_eq!(action_rooms(&limits, 5, 64, 0), [vec![5; 12], vec![0; 4]].concat());
        assert_eq!(action_rooms(&[0, 1, 0, 5, 2], 3, 6, 0), [0, 1, 0, 3, 0]);
        assert_eq!(action_rooms(&limits, 5, 4, 0), [0; 16]);
        let rotated = action_rooms(&limits, 5, 64, 4);
        assert!(rotated[..4].iter().all(|&room| room == 0));
        assert!(rotated[4..].iter().all(|&room| room == 5));
        for width in [1, 3, 5] {
            let rooms = action_rooms(&limits, width, 64, 0);
            assert!(rooms.iter().filter(|&&room| room > 0).count() * width <= 64);
        }
    }

    #[test]
    fn cold_action_keeps_eligible_pass_when_preferred_width_cannot_fit() {
        let widths = [0, 3, 5];
        let limits = [5, 5];
        let rooms: Vec<_> = widths.iter().map(|&width| action_rooms(&limits, width, 4, 0)).collect();
        assert_eq!(cold_action(&widths, &rooms, 5), 1);
        assert_eq!(cold_action(&widths, &rooms, 3), 1);
        let rooms: Vec<_> = widths.iter().map(|&width| action_rooms(&limits, width, 10, 0)).collect();
        assert_eq!(cold_action(&widths, &rooms, 5), 2);
        let rooms: Vec<_> = widths.iter().map(|&width| action_rooms(&[0, 0], width, 10, 0)).collect();
        assert_eq!(cold_action(&widths, &rooms, 5), 0);
    }

    #[test]
    fn terminal_and_stream_failures_keep_only_judged_rejections() {
        let rows = [10, 11, 12, 13];
        assert_eq!(round_censor(false, false, 2, &rows, Some(99), 99), None);
        assert_eq!(round_censor(true, false, 2, &rows, Some(99), 99), None);
        assert_eq!(round_censor(true, false, 2, &rows, Some(12), 12), Some(Censor::Eos));
        assert_eq!(round_censor(true, false, 2, &rows, Some(12), 99), Some(Censor::OutputLimit));
        assert_eq!(round_censor(true, true, 2, &rows, Some(12), 99), Some(Censor::Cancelled));
        assert_eq!(round_censor(true, true, 2, &rows, None, 99), Some(Censor::Cancelled));
        assert_eq!(round_censor(true, true, 2, &rows, Some(99), 99), None);
        assert_eq!(round_censor(true, true, 4, &rows, Some(99), 99), Some(Censor::Cancelled));
    }

    #[test]
    fn model_id_from_snapshot_path() {
        let path = std::path::Path::new("/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/abc");
        assert_eq!(super::model_id(path).as_deref(), Some("deepseek-ai/DeepSeek-V4-Flash-0731"));
    }
}
