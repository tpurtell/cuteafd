//! Static prefill operations for the shared admission/continuation workflow.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::{CacheLease, CacheStage};
use crate::families::deepseek_v41::v41_block::EncoderSuffix;
use crate::shared::memory::device::DeviceOwner;
use crate::families::deepseek_v41::v41_requests::RequestBatch;
use crate::families::deepseek_v41::v41_target_pass::{DistributedTargetPass, VerificationTarget};
use speculative::DraftChain;
use std::cell::RefCell;
use crate::shared::prefill_share::{ChunkBudget, PrefillDriver, PrefillUnit};
use std::ops::Range;
use std::time::Duration;

#[cfg(test)]
pub(crate) fn exercise_prefill<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(
    lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime, requests: &mut Requests<'a>,
    prefill: &mut PrefillSlots<'a, P>, lease: CacheLease, tokens: &[u32], chunk_rows: usize, waves: usize,
    draft: Option<&mut DraftRuntime<'_, 'a, C>>,
) -> Result<u32> {
    let (events, _receive) = mpsc::unbounded_channel();
    let job = NativeRequest { prefix_cache_source: Default::default(), literal_spans: Vec::new(), prompt: String::new(), constraint: None, images: Vec::new(), media: Vec::new(), audio: Vec::new(), max_tokens: 16, sampling: Default::default(), stop_token_ids: Vec::new(), events, usage: None, probe: None };
    super::prefill(lib, runtime, requests, prefill, None, lease, tokens,
        super::PrefillShape { rows: chunk_rows, waves }, &job, draft, &mut || Ok(()))?.select(None)
}

/// [`exercise_prefill`] whose client disconnects at the first unit's begin
/// (its prefill hold drops the receiver): the units call fails mid-unit with
/// chained work queued. Returns that error; the slots must come back at rest,
/// windows at rows from 0 and slot 0 holding its link (C4, D1).
#[cfg(test)]
pub(crate) fn exercise_cancelled_prefill<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(
    lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime, requests: &mut Requests<'a>,
    prefill: &mut PrefillSlots<'a, P>, lease: CacheLease, tokens: &[u32], chunk_rows: usize, waves: usize,
) -> anyhow::Error {
    let (events, receive) = mpsc::unbounded_channel();
    let job = NativeRequest { prefix_cache_source: Default::default(), literal_spans: Vec::new(), prompt: String::new(), constraint: None, images: Vec::new(), media: Vec::new(), audio: Vec::new(), max_tokens: 16, sampling: Default::default(), stop_token_ids: Vec::new(), events, usage: None, probe: None };
    let mut receive = Some(receive);
    match super::prefill::<P, C>(lib, runtime, requests, prefill, None, lease, tokens,
        super::PrefillShape { rows: chunk_rows, waves }, &job, None,
        &mut || { receive.take(); Ok(()) }) {
        Ok(_) => anyhow::anyhow!("cancelled prefill unexpectedly finished"),
        Err(error) => error,
    }
}

/// Only request-owned state survives a unit; both execution lanes are drained.
pub(crate) struct V41Prefill<S> {
    state: PrefillState<S>,
}

enum PrefillState<S> {
    Start,
    Encoder { suffix: S, next: usize, replay_end: usize, streamed: bool },
    Replay { suffix: S },
    Continuation { next: usize },
    Done,
}

impl<S> V41Prefill<S> {
    pub(crate) fn new() -> Self { Self { state: PrefillState::Start } }
}

#[derive(Debug)]
pub(crate) enum V41Unit {
    ReplayChunk(Range<usize>),
    Single(Range<usize>),
    Stream(Vec<Range<usize>>),
    DecoderReplay,
    Continuation(Range<usize>),
}

impl V41Unit {
    fn kind(&self) -> usize {
        match self {
            Self::ReplayChunk(_) => 0,
            Self::Single(_) => 1,
            Self::Stream(_) => 2,
            Self::DecoderReplay => 3,
            Self::Continuation(_) => 4,
        }
    }
}

/// Critical-path microseconds per row, retained across requests by the scheduler.
#[derive(Default)]
pub(crate) struct V41PrefillTimes {
    us_per_row: [Option<f64>; 5],
}

/// One stream call's units: the shared cut for page-aligned unit widths;
/// test geometries below a page (2-RTX fixture: 3 and 16 rows) keep plain
/// `chunk_rows` units, as `V41PrefillTimes::encoder_unit` streams them.
/// The prefill units the encoder runs a prompt in from `start`, as the
/// stream path cuts them: an encoder-prefix replay (`start..replay_end`, at
/// most one 128-token window) first, then calls of `waves` units of
/// `chunk_rows` from its end, each call cut by [`stream_units`]; units run
/// `waves` deep in pipeline order.
pub(super) fn prompt_units(start: usize, replay_end: usize, tokens: usize, chunk_rows: usize, waves: usize)
    -> Result<cuteafd_loader::TableUnitPlan> {
    let call_rows = (waves * chunk_rows).max(1);
    let (mut ends, mut calls) = (Vec::new(), Vec::new());
    let mut at = replay_end.clamp(start, tokens);
    if at > start { calls.push(0); ends.push(at as u64); }
    while at < tokens {
        let rows = (tokens - at).min(call_rows);
        calls.push(ends.len());
        ends.extend(stream_units(rows, chunk_rows.max(1), waves.max(1))?.iter().map(|unit| (at + unit.end) as u64));
        at += rows;
    }
    Ok(cuteafd_loader::TableUnitPlan { ends, depth: waves.max(1), calls })
}

fn stream_units(rows: usize, chunk_rows: usize, waves: usize) -> Result<Vec<std::ops::Range<usize>>> {
    if chunk_rows % 64 != 0 {
        ensure!(rows > 0 && chunk_rows > 0, "empty prefill stream units");
        return Ok((0..rows).step_by(chunk_rows).map(|first| first..rows.min(first + chunk_rows)).collect());
    }
    crate::shared::prefill_pipeline::prefill_units(rows, chunk_rows, waves,
        cuteafd_core::coordinator_programs::V41_LIVE_MIN_ROWS as usize)
}

impl V41PrefillTimes {
    fn encoder_unit(&self, tokens: usize, chunk_rows: usize, waves: usize, next: usize,
        replay_end: usize, streamed: bool, budget: ChunkBudget) -> V41Unit {
        if next < replay_end {
            return V41Unit::ReplayChunk(next..(next + chunk_rows).min(replay_end));
        }
        if next == tokens { return V41Unit::DecoderReplay; }
        let end = (next + chunk_rows).min(tokens);
        if !streamed { return V41Unit::Single(next..end); }
        let whole = budget.target.is_none() && budget.row_cap == usize::MAX;
        if whole && waves > 1 && chunk_rows % 64 == 0 {
            // The shared cut: full units of the configured (captured) width and one tail.
            if let Ok(units) = stream_units(tokens - next, chunk_rows, waves) {
                return V41Unit::Stream(units.into_iter().map(|unit| next + unit.start..next + unit.end).collect());
            }
        }
        let mut chunks = vec![next..end];
        let mut next = end;
        while next < tokens && (whole || chunks.len() < 2) {
            let end = (next + chunk_rows).min(tokens);
            let rows = end - chunks[0].start;
            if !whole && (rows > budget.row_cap || budget.target.is_some_and(|target|
                self.estimate(2, rows) > target)) { break; }
            chunks.push(next..end);
            next = end;
        }
        V41Unit::Stream(chunks)
    }

    fn estimate(&self, kind: usize, rows: usize) -> Duration {
        // Stream samples use wall time across both lanes, not their summed execution times.
        Duration::from_secs_f64(self.us_per_row[kind].unwrap_or(100.0) * rows as f64 / 1_000_000.0)
    }

    fn observe(&mut self, kind: usize, rows: usize, took: Duration) {
        if rows == 0 { return; }
        let sample = took.as_secs_f64() * 1_000_000.0 / rows as f64;
        let previous = self.us_per_row[kind];
        self.us_per_row[kind] = Some(previous.map_or(sample, |old| old * 0.8 + sample * 0.2));
    }
}

/// Per-call borrows; the cursor and timing estimates outlive this step.
pub(crate) struct V41PrefillStep<'s, 'w, 'a, P: PrefillTarget<'a>, C: DraftChain<'a>> {
    pub(crate) lib: &'a NativeLibrary,
    pub(crate) runtime: &'s tokio::runtime::Runtime,
    pub(crate) requests: &'s mut Requests<'a>,
    pub(crate) lease: CacheLease,
    pub(crate) tokens: &'s [u32],
    pub(crate) chunk_rows: usize,
    /// Units in flight on the shared executor.
    pub(crate) waves: usize,
    /// The prefill slots: slot 0 (with the prefill link) runs non-unit work.
    pub(crate) prefill: &'s mut PrefillSlots<'a, P>,
    pub(crate) job: &'s NativeRequest,
    pub(crate) draft: Option<&'s mut DraftRuntime<'w, 'a, C>>,
    pub(crate) hold: &'s mut dyn FnMut() -> Result<()>,
    pub(crate) times: &'s mut V41PrefillTimes,
}

impl<'a, P: PrefillTarget<'a>, C: DraftChain<'a>> V41PrefillStep<'_, '_, 'a, P, C> {
    /// Whether `rows` uncached prompt rows run as units on the shared
    /// executor: more than one unit.
    fn streams(&self, rows: usize) -> bool { rows.div_ceil(self.chunk_rows.max(1)) > 1 }
    fn encoder_unit(&self, next: usize, replay_end: usize, streamed: bool,
        budget: ChunkBudget) -> V41Unit {
        self.times.encoder_unit(self.tokens.len(), self.chunk_rows, self.waves, next, replay_end, streamed, budget)
    }

    fn initialize(&mut self, cursor: &mut V41Prefill<P::Suffix>) -> Result<()> {
        if !matches!(cursor.state, PrefillState::Start) { return Ok(()); }
        // The whole uncached prompt is known: read its Engram rows into the row
        // caches ahead of the gathers. A failure only leaves rows to load on demand.
        let (tokens, chunk_rows, waves) = (self.tokens.len(), self.chunk_rows, self.waves);
        let replay_end = self.requests.cache().committed_end(self.lease)? as usize;
        if let Err(error) = self.requests.start_lookahead(self.lease, self.tokens,
            |start| prompt_units(start, replay_end, tokens, chunk_rows, waves)) {
            tracing::warn!(%error, "Engram look-ahead not started; rows load on demand");
        }
        self.requests.lookahead_running(self.lease);
        let end = self.tokens.len() as u64;
        let cached = self.requests.cache().committed_end(self.lease)? as usize;
        let stage = self.requests.cache().stage(self.lease)?;
        if cached > 0 && stage == CacheStage::Full {
            cursor.state = PrefillState::Continuation { next: cached };
            return Ok(());
        }
        let suffix = self.prefill.slot0().0.new_suffix(self.lib, end)?;
        let next = if stage == CacheStage::EncoderReplay {
            let start = self.requests.cache().history_end(self.lease)? as usize;
            ensure!(cached - start <= 128, "encoder prefix replay exceeds one window");
            start
        } else {
            if stage == CacheStage::Full { self.requests.begin_encoder(self.lease, end)?; }
            cached
        };
        cursor.state = PrefillState::Encoder { suffix, next, replay_end: cached,
            streamed: self.streams(self.tokens.len().saturating_sub(cached)) };
        Ok(())
    }
}

impl<'a, P: PrefillTarget<'a>, C: DraftChain<'a>> PrefillDriver for V41PrefillStep<'_, '_, 'a, P, C> {
    type Cursor = V41Prefill<P::Suffix>;
    type Unit = V41Unit;
    type Output = RetainedScores;

    fn plan(&self, cursor: &Self::Cursor, budget: ChunkBudget) -> Result<PrefillUnit<V41Unit>> {
        ensure!(self.chunk_rows > 0, "prefill chunk rows must be positive");
        let continuation = |next: usize| {
            let whole = budget.target.is_none() && budget.row_cap == usize::MAX;
            V41Unit::Continuation(next..if whole { self.tokens.len() }
                else { (next + self.chunk_rows).min(self.tokens.len()) })
        };
        let work = match &cursor.state {
            PrefillState::Start => {
                let cached = self.requests.cache().committed_end(self.lease)? as usize;
                let stage = self.requests.cache().stage(self.lease)?;
                if cached > 0 && stage == CacheStage::Full { continuation(cached) }
                else {
                    let next = if stage == CacheStage::EncoderReplay {
                        self.requests.cache().history_end(self.lease)? as usize
                    } else { cached };
                    self.encoder_unit(next, cached, self.streams(self.tokens.len().saturating_sub(cached)), budget)
                }
            }
            PrefillState::Encoder { next, replay_end, streamed, .. } =>
                self.encoder_unit(*next, *replay_end, *streamed, budget),
            PrefillState::Replay { .. } => V41Unit::DecoderReplay,
            PrefillState::Continuation { next } => continuation(*next),
            PrefillState::Done => anyhow::bail!("prefill already finished"),
        };
        let rows = match &work {
            V41Unit::ReplayChunk(range) | V41Unit::Single(range) | V41Unit::Continuation(range) => range.len(),
            V41Unit::Stream(chunks) => chunks.iter().map(|range| range.len()).sum(),
            V41Unit::DecoderReplay => self.tokens.len().min(128),
        };
        let finalizes = match &work {
            V41Unit::DecoderReplay => true,
            V41Unit::Continuation(range) => range.end == self.tokens.len(),
            _ => false,
        };
        let estimate = self.times.estimate(work.kind(), rows);
        Ok(PrefillUnit { work, rows, estimate, finalizes })
    }

    fn execute(&mut self, cursor: &mut Self::Cursor, unit: &PrefillUnit<V41Unit>) -> Result<Option<RetainedScores>> {
        use crate::families::deepseek_v41::v41_backbone_cache::CacheWork;
        self.initialize(cursor)?;
        match (&unit.work, &mut cursor.state) {
            (V41Unit::ReplayChunk(range) | V41Unit::Single(range), PrefillState::Encoder { suffix, next, .. }) => {
                let chunk = &self.tokens[range.clone()];
                ensure!(!self.job.events.is_closed(), "client disconnected");
                prefill_hold(self.hold);
                let mut batch = self.requests.prepare(&[RequestTokens { lease: self.lease, tokens: chunk,
                    image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
                let started = Instant::now();
                let (pass, transport) = self.prefill.slot0();
                let result = (|| -> Result<()> {
                    // SAFETY: the live request owns the batch/suffix until execution and commit drain.
                    self.runtime.block_on(unsafe { pass.encoder_part(self.requests, &mut batch, transport, suffix) })?;
                    ensure!(!self.job.events.is_closed(), "client disconnected");
                    self.runtime.block_on(pass.commit_prefill::<C>(self.requests, &mut batch, None, chunk.len() as u32))
                })();
                if result.is_err() { pass.discard(&mut batch)?; }
                result?;
                if matches!(unit.work, V41Unit::Single(_)) {
                    console::totals::prefill(chunk.len());
                    console::Prefill::done(console::PrefillKind::Single, 0, 0, 1, chunk.len(), started);
                    tracing::debug!(target: "cuteafd::timing", rows=chunk.len(), total_us=started.elapsed().as_micros() as u64, "target encoder step");
                }
                *next = range.end;
            }
            (V41Unit::Stream(ranges), PrefillState::Encoder { suffix, next, .. }) => {
                let span = ranges.first().context("empty encoder units")?.start..ranges.last().context("empty encoder units")?.end;
                let started = Instant::now();
                let hold = RefCell::new(&mut *self.hold);
                let before_unit = || -> Result<()> { prefill_hold(&mut *hold.borrow_mut()); Ok(()) };
                let keep_running = || !self.job.events.is_closed();
                // Slot 0 lends the prefill link to the span's calls; every
                // slot's waves go out on it. Nothing fallible runs between the
                // take and the loop, and every error inside the loop lands in
                // `result`, so the link returns on every exit (D1).
                let mut link = P::take_link(&mut self.prefill.slots[0].1)?;
                let mut result = Ok(());
                let mut at = span.start;
                let (waves, call_rows) = (self.waves, self.waves * self.chunk_rows);
                // One call per window cycle: `waves` units of `chunk_rows` rows.
                while at < span.end && result.is_ok() {
                    let rows = (span.end - at).min(call_rows);
                    result = (|| -> Result<()> {
                        ensure!(waves <= self.prefill.slots.len(), "{waves} prefill waves exceed the {} slots",
                            self.prefill.slots.len());
                        let unit_rows = stream_units(rows, self.chunk_rows, waves)?;
                        let chunks: Vec<_> = unit_rows.iter().map(|range| &self.tokens[at + range.start..at + range.end]).collect();
                        // The next call's first units, cut as that call will cut them.
                        let ahead: Vec<&[u32]> = lookahead(at + rows..span.end, self.chunk_rows, waves)?
                            .into_iter().map(|range| &self.tokens[range]).collect();
                        let mut slots: Vec<(&mut P, &mut P::Transport)> = self.prefill.slots.iter_mut()
                            .map(|(pass, transport)| (pass, transport)).collect();
                        slots.truncate(waves);
                        // SAFETY: every slot's pass and the link retain the request
                        // and suffix through drained completion or cancellation.
                        self.runtime.block_on(unsafe { P::encoder_units(slots, &mut link, self.requests,
                            self.lease, &chunks, suffix, &ahead, &keep_running, &before_unit) })?;
                        tracing::debug!(target: "cuteafd::timing", rows, units=chunks.len(), waves,
                            total_us=started.elapsed().as_micros() as u64, "target encoder units");
                        Ok(())
                    })();
                    at += rows;
                }
                P::restore_link(&mut self.prefill.slots[0].1, link)?;
                result?;
                *next = span.end;
            }
            (V41Unit::DecoderReplay, PrefillState::Replay { suffix } | PrefillState::Encoder { suffix, .. }) => {
                ensure!(!self.job.events.is_closed(), "client disconnected");
                let start = self.requests.begin_decoder_replay(self.lease)?;
                let rows = (self.tokens.len() as u64 - start) as u32;
                let mut batch = self.requests.prepare_replay(&[CacheWork { lease: self.lease, tokens: rows, kind: ExpertV2SourceKind::Prefill }])?;
                let started = Instant::now();
                let (pass, transport) = self.prefill.slot0();
                let result = (|| -> Result<RetainedScores> {
                    // SAFETY: all encoder rows committed and the retained suffix matches this replay.
                    let bytes = self.runtime.block_on(unsafe { pass.prefill_logits(self.lib, self.requests,
                        &mut batch, transport, &[rows as usize - 1], Some(suffix)) })?;
                    let scores = RetainedScores::new(scores::VOCAB, bytes)?;
                    ensure!(!self.job.events.is_closed(), "client disconnected");
                    self.runtime.block_on(pass.commit_prefill(self.requests, &mut batch, self.draft.as_deref_mut(), rows))?;
                    Ok(scores)
                })();
                if result.is_err() { pass.discard(&mut batch)?; }
                console::Prefill::done(console::PrefillKind::Replay, 0, 0, 1, rows as usize, started);
                tracing::debug!(target: "cuteafd::timing", rows, total_us=started.elapsed().as_micros() as u64, "target decoder replay");
                let scores = result?;
                cursor.state = PrefillState::Done;
                return Ok(Some(scores));
            }
            (V41Unit::Continuation(range), PrefillState::Continuation { next }) => {
                let (pass, transport) = self.prefill.slot0();
                let scores = super::prefill_continuation(self.lib, self.runtime, pass, self.requests,
                    transport, self.lease, &self.tokens[range.clone()], self.chunk_rows, self.job,
                    self.draft.as_deref_mut(), self.hold)?;
                *next = range.end;
                if range.end == self.tokens.len() {
                    cursor.state = PrefillState::Done;
                    return Ok(Some(scores));
                }
            }
            _ => anyhow::bail!("prefill unit does not match cursor"),
        }
        if matches!(cursor.state, PrefillState::Encoder { next, .. } if next == self.tokens.len()) {
            if let PrefillState::Encoder { suffix, .. } = std::mem::replace(&mut cursor.state, PrefillState::Done) {
                cursor.state = PrefillState::Replay { suffix };
            }
        }
        Ok(None)
    }

    fn observe(&mut self, unit: &PrefillUnit<V41Unit>, took: Duration) {
        self.times.observe(unit.work.kind(), unit.rows, took);
    }

    fn abort_and_drain(&mut self, _cursor: &mut Self::Cursor) -> Result<()> {
        // Batch failures discard locally; the stream and commit helpers drain their own lanes.
        Ok(())
    }
}

/// The prefill slots (PLAN "Prefill contract"), apart from the decode lanes:
/// `slots` passes on one shared storage set per GPU (row planes at the
/// reserve, scratch at the unit cap), each slot's FFN owner beside it. Slot
/// 0's FFN owner holds the one prefill link, whose return slots hold the
/// in-flight units' waves; slot 0 also runs every non-unit prefill step
/// (replay, continuation, scoring, cold steps) at rows from 0.
pub(crate) struct PrefillSlots<'a, P: PrefillTarget<'a>> {
    pub slots: Vec<(P, P::Transport)>,
    pub packed_suffix: P::Suffix,
}
impl<'a, P: PrefillTarget<'a>> PrefillSlots<'a, P> {
    pub(crate) fn slot0(&mut self) -> (&mut P, &mut P::Transport) {
        let (pass, transport) = &mut self.slots[0];
        (pass, transport)
    }
}

pub(crate) trait PrefillTarget<'a>: VerificationTarget<'a> + Sized {
    type Suffix;
    /// The one prefill transport (PLAN "Prefill contract"): its return slots
    /// hold the in-flight units' waves. Slot 0's FFN owner holds it.
    type PrefillLink;
    fn take_link(transport: &mut Self::Transport) -> Result<Self::PrefillLink>;
    fn restore_link(transport: &mut Self::Transport, link: Self::PrefillLink) -> Result<()>;
    fn new_packed_storage(&self, lib: &'a NativeLibrary, unit: usize) -> Result<Self::Suffix>;
    fn new_packed_suffix(storage: &Self::Suffix,
        segments: &[crate::shared::prefill_packing::Segment]) -> Result<Self::Suffix>;
    /// # Safety
    /// Member leases, passes and packed suffix remain exclusive until all waves drain.
    async unsafe fn packed_encoder_unit(slots: Vec<(&mut Self, &mut Self::Transport)>,
        prefill: &mut Self::PrefillLink, requests: &mut Requests<'a>,
        members: &[(CacheLease, &[u32])], suffix: &mut Self::Suffix) -> Result<()>;
    async fn commit_packed_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: &[u32]) -> Result<()>;
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix>;
    /// # Safety
    /// Batch and suffix belong to this request and layout; all owners remain
    /// live through completion/cancellation, with no conflicting access.
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()>;
    /// Encoder units of one prompt on the shared executor, one unit per slot
    /// in flight: slot k is `slots[k]`, a target pass with its reduce/FFN
    /// owner; every unit's wave goes out on the one `prefill` link.
    /// # Safety
    /// The passes have independent workspaces and the same layout/model; the
    /// chunks and suffix follow the encoder's request/cache contract.
    /// `before_unit` runs synchronously before each unit (host-cache store
    /// pacing, packet HC-9).
    async unsafe fn encoder_units(slots: Vec<(&mut Self, &mut Self::Transport)>, prefill: &mut Self::PrefillLink,
        requests: &mut Requests<'a>, lease: CacheLease, chunks: &[&[u32]], suffix: &mut Self::Suffix,
        ahead: &[&[u32]], keep_running: &dyn Fn() -> bool, before_unit: &dyn Fn() -> Result<()>) -> Result<()>;
    /// # Safety
    /// Batch inputs and optional completed encoder suffix match this target.
    async unsafe fn prefill_logits(&mut self, lib: &'a NativeLibrary, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, selected: &[usize],
        suffix: Option<&Self::Suffix>) -> Result<Vec<u8>>;
    async fn commit_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: u32) -> Result<()>;
}
impl<'a> PrefillTarget<'a> for TargetPass<'_, 'a> {
    type Suffix = EncoderSuffix<'a>;
    type PrefillLink = crate::families::deepseek_v41::v41_experts::coordinator::V41Link<'a>;
    fn take_link(transport: &mut Self::Transport) -> Result<Self::PrefillLink> { transport.take_link() }
    fn restore_link(transport: &mut Self::Transport, link: Self::PrefillLink) -> Result<()> {
        transport.restore_link(link)
    }
    fn new_packed_storage(&self, lib: &'a NativeLibrary, unit: usize) -> Result<Self::Suffix> {
        EncoderSuffix::packed_storage(lib, unit)
    }
    fn new_packed_suffix(storage: &Self::Suffix,
        segments: &[crate::shared::prefill_packing::Segment]) -> Result<Self::Suffix> {
        EncoderSuffix::packed(storage, segments)
    }
    async unsafe fn packed_encoder_unit(slots: Vec<(&mut Self, &mut Self::Transport)>,
        prefill: &mut Self::PrefillLink, requests: &mut Requests<'a>,
        members: &[(CacheLease, &[u32])], suffix: &mut Self::Suffix) -> Result<()> {
        // SAFETY: the caller retains all packed owners until the shared executor drains.
        unsafe { Self::execute_packed_encoder_unit(slots, prefill, requests, members, suffix).await }
    }
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix> {
        EncoderSuffix::new(lib, end, EncoderSuffix::device_bytes(end)?)
    }
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()> {
        self.set_attention_decode(false);
        if batch.cache()?.stage() == CacheStage::EncoderReplay {
            unsafe { self.execute_encoder_replay(requests, batch, transport, 0, suffix).await }
        } else { unsafe { self.execute_encoder(requests, batch, transport, 0, suffix).await } }
    }
    async unsafe fn encoder_units(slots: Vec<(&mut Self, &mut Self::Transport)>, prefill: &mut Self::PrefillLink,
        requests: &mut Requests<'a>, lease: CacheLease, chunks: &[&[u32]], suffix: &mut Self::Suffix,
        ahead: &[&[u32]], keep_running: &dyn Fn() -> bool, before_unit: &dyn Fn() -> Result<()>) -> Result<()> {
        unsafe { Self::execute_encoder_units(slots, prefill, requests, lease, chunks, suffix, ahead, keep_running, before_unit).await }
    }
    async unsafe fn prefill_logits(&mut self, lib: &'a NativeLibrary, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, selected: &[usize],
        suffix: Option<&Self::Suffix>) -> Result<Vec<u8>> {
        self.set_attention_decode(false);
        let logits = if let Some(suffix) = suffix {
            let encoder = suffix.output()?;
            unsafe { self.execute_replay(requests, batch, transport, 0, selected, &encoder).await? }
        } else { unsafe { self.execute(requests, batch, transport, 0, selected).await? } };
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        Ok(bytes)
    }
    async fn commit_packed_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: &[u32]) -> Result<()> {
        if let Some(draft) = draft { draft.commit_batch(self, requests, batch, accepted) }
        else { self.commit(requests, batch, accepted) }
    }
    async fn commit_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: u32) -> Result<()> {
        if let Some(draft) = draft { draft.commit(self, requests, batch, accepted) }
        else { self.commit(requests, batch, &[accepted]) }
    }
}
impl<'a> PrefillTarget<'a> for DistributedTargetPass<'_, 'a> {
    type Suffix = DeviceOwner<'a, EncoderSuffix<'a>>;
    type PrefillLink = DeviceOwner<'a, crate::families::deepseek_v41::v41_experts::coordinator::V41Link<'a>>;
    fn take_link(transport: &mut Self::Transport) -> Result<Self::PrefillLink> {
        let device = transport.device;
        device.own(|| transport.get_mut().take_link())
    }
    fn restore_link(transport: &mut Self::Transport, link: Self::PrefillLink) -> Result<()> {
        let device = transport.device;
        ensure!(link.device.id == device.id, "prefill link device differs from slot 0's");
        device.run(|| transport.get_mut().restore_link(link.into_inner()))
    }
    fn new_packed_storage(&self, lib: &'a NativeLibrary, unit: usize) -> Result<Self::Suffix> {
        self.encoder_device()?.own(|| EncoderSuffix::packed_storage(lib, unit))
    }
    fn new_packed_suffix(storage: &Self::Suffix,
        segments: &[crate::shared::prefill_packing::Segment]) -> Result<Self::Suffix> {
        storage.device.own(|| EncoderSuffix::packed(storage.get(), segments))
    }
    async unsafe fn packed_encoder_unit(slots: Vec<(&mut Self, &mut Self::Transport)>,
        prefill: &mut Self::PrefillLink, requests: &mut Requests<'a>,
        members: &[(CacheLease, &[u32])], suffix: &mut Self::Suffix) -> Result<()> {
        // SAFETY: the caller retains all packed owners until the shared executor drains.
        unsafe { Self::execute_packed_encoder_unit(slots, prefill, requests, members, suffix).await }
    }
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix> {
        self.encoder_device()?.own(|| EncoderSuffix::new(lib, end, EncoderSuffix::device_bytes(end)?))
    }
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()> {
        self.set_attention_decode(false);
        unsafe { self.execute(&RefCell::new(requests), batch, transport, 0, &[], Some(suffix), None, false).await }
    }
    async unsafe fn encoder_units(slots: Vec<(&mut Self, &mut Self::Transport)>, prefill: &mut Self::PrefillLink,
        requests: &mut Requests<'a>, lease: CacheLease, chunks: &[&[u32]], suffix: &mut Self::Suffix,
        ahead: &[&[u32]], keep_running: &dyn Fn() -> bool, before_unit: &dyn Fn() -> Result<()>) -> Result<()> {
        unsafe { Self::execute_encoder_units(slots, prefill, requests, lease, chunks, suffix, ahead, keep_running, before_unit).await }
    }
    async unsafe fn prefill_logits(&mut self, _lib: &'a NativeLibrary, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, selected: &[usize],
        suffix: Option<&Self::Suffix>) -> Result<Vec<u8>> {
        self.set_attention_decode(false);
        let encoder = suffix.map(|suffix| suffix.output()).transpose()?;
        unsafe { self.execute(&RefCell::new(requests), batch, transport, 0, selected, None, encoder.as_ref(), false).await?; }
        self.download_logits(batch, &(0..selected.len()).collect::<Vec<_>>()).await
    }
    async fn commit_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: u32) -> Result<()> {
        self.commit_packed_prefill(requests, batch, draft, &[accepted]).await
    }
    async fn commit_packed_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: &[u32]) -> Result<()> {
        struct Commit<'s, 'w, 'a, P: VerificationTarget<'a>, C: DraftChain<'a>> {
            pass: &'s mut P, requests: &'s mut Requests<'a>, batch: &'s mut RequestBatch,
            draft: Option<&'s mut DraftRuntime<'w, 'a, C>>, armed: bool,
        }
        impl<'a, P: VerificationTarget<'a>, C: DraftChain<'a>> Drop for Commit<'_, '_, 'a, P, C> {
            fn drop(&mut self) {
                if self.armed {
                    if let Err(error) = self.pass.abort_cache_commit(self.requests) { tracing::error!(%error, "aborting prefill target commit"); }
                    if let Some(draft) = &mut self.draft {
                        if let Err(error) = draft.abort_queued_commit(0, self.requests, self.batch) { tracing::error!(%error, "aborting prefill draft commit"); }
                    }
                }
            }
        }
        let mut commit = Commit { pass: self, requests, batch, draft, armed: true };
        if let Some(draft) = &mut commit.draft {
            draft.begin_queued_commit(0, commit.pass, commit.requests, commit.batch, accepted)?;
        }
        commit.pass.enqueue_cache_commit(commit.requests, commit.batch, accepted)?;
        loop {
            let target_ready = commit.pass.poll_cache_commit()?;
            let draft_ready = commit.draft.as_ref().map(|draft| draft.poll_queued_commit(0)).transpose()?.unwrap_or(true);
            if target_ready && draft_ready { break; }
            tokio::task::yield_now().await;
        }
        if let Some(draft) = &mut commit.draft {
            draft.finish_queued_commit(0, commit.pass, commit.requests, commit.batch, accepted)?;
        } else { commit.pass.commit(commit.requests, commit.batch, accepted)?; }
        commit.armed = false;
        Ok(())
    }
}

#[cfg(test)]
mod prefill_tests {
    use super::*;

    fn stream(unit: V41Unit) -> Vec<Range<usize>> {
        match unit { V41Unit::Stream(chunks) => chunks, _ => panic!("expected encoder stream") }
    }

    #[test]
    fn whole_preserves_all_chunk_boundaries_in_one_stream() {
        let times = V41PrefillTimes::default();
        assert_eq!(stream(times.encoder_unit(1050, 256, 1, 0, 0, true, ChunkBudget::WHOLE)),
            [0..256, 256..512, 512..768, 768..1024, 1024..1050]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 1, 300, 300, true, ChunkBudget::WHOLE)),
            [300..556, 556..812, 812..1050]);
        assert!(matches!(times.encoder_unit(1050, 256, 1, 950, 950, false, ChunkBudget::WHOLE), V41Unit::Single(range) if range == (950..1050)));
    }

    #[test]
    fn whole_matches_original_chunk_iterator_at_legal_capacities() {
        let times = V41PrefillTimes::default();
        for chunk_rows in [80usize, 256, 1024, 4096] {
            for cached in [0, 2, 128, 300] {
                for remaining in [0, 1, chunk_rows - 1, chunk_rows, chunk_rows + 1, 2 * chunk_rows + 17] {
                    let tokens = cached + remaining;
                    let streamed = remaining.div_ceil(chunk_rows) > 1;
                    let unit = times.encoder_unit(tokens, chunk_rows, 1, cached, cached, streamed, ChunkBudget::WHOLE);
                    let expected: Vec<_> = (cached..tokens).step_by(chunk_rows)
                        .map(|start| start..(start + chunk_rows).min(tokens)).collect();
                    match unit {
                        V41Unit::DecoderReplay => assert!(expected.is_empty()),
                        V41Unit::Single(range) => assert_eq!(expected, [range]),
                        V41Unit::Stream(chunks) => assert_eq!(expected, chunks),
                        _ => panic!("unexpected uncached encoder unit"),
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_stream_keeps_at_least_one_and_at_most_two_ordered_chunks() {
        let mut times = V41PrefillTimes::default();
        times.observe(2, 256, Duration::from_micros(256));
        let budget = ChunkBudget { target: Some(Duration::from_micros(511)), row_cap: usize::MAX };
        assert_eq!(stream(times.encoder_unit(1050, 256, 1, 0, 0, true, budget)), [0..256]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 1, 0, 0, true,
            ChunkBudget { target: Some(Duration::from_micros(512)), ..budget })), [0..256, 256..512]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 1, 0, 0, true,
            ChunkBudget { target: Some(Duration::ZERO), row_cap: 0 })), [0..256]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 1, 0, 0, true,
            ChunkBudget { target: Some(Duration::from_secs(1)), row_cap: 511 })), [0..256]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 1, 0, 0, true,
            ChunkBudget { target: Some(Duration::from_secs(1)), row_cap: 512 })), [0..256, 256..512]);
        // A streamed prompt's final singleton must not switch to encoder_part.
        assert_eq!(stream(times.encoder_unit(1050, 256, 1, 1024, 0, true, budget)), [1024..1050]);
    }

    #[test]
    fn whole_streams_cut_fixed_captured_units_and_one_tail() {
        let times = V41PrefillTimes::default();
        // Every unit but the tail is exactly the registered (captured) width.
        assert_eq!(stream(times.encoder_unit(8192, 2752, 3, 0, 0, true, ChunkBudget::WHOLE)),
            [0..2752, 2752..5504, 5504..8192]);
        assert_eq!(stream(times.encoder_unit(8192, 1024, 3, 0, 0, true, ChunkBudget::WHOLE)).len(), 8);
        let units = stream(times.encoder_unit(65536, 2752, 3, 0, 0, true, ChunkBudget::WHOLE));
        assert!(units[..units.len() - 1].iter().all(|unit| unit.len() == 2752));
        assert_eq!((units[0].start, units.last().unwrap().end), (0, 65536));
        assert_eq!(stream(times.encoder_unit(8492, 2752, 3, 300, 300, true, ChunkBudget::WHOLE)),
            [300..3052, 3052..5804, 5804..8492]);
        let budget = ChunkBudget { target: Some(Duration::ZERO), row_cap: 0 };
        assert_eq!(stream(times.encoder_unit(8192, 2752, 3, 0, 0, true, budget)), [0..2752]);
    }

    #[test]
    fn history_replay_stops_at_cached_frontier_before_the_encoder_stream() {
        let times = V41PrefillTimes::default();
        assert!(matches!(times.encoder_unit(1024, 80, 1, 128, 256, true, ChunkBudget::WHOLE), V41Unit::ReplayChunk(range) if range == (128..208)));
        assert!(matches!(times.encoder_unit(1024, 80, 1, 208, 256, true, ChunkBudget::WHOLE), V41Unit::ReplayChunk(range) if range == (208..256)));
        assert!(matches!(times.encoder_unit(1024, 80, 1, 1024, 256, true, ChunkBudget::WHOLE), V41Unit::DecoderReplay));
    }

    #[test]
    fn timing_ewma_is_per_kind_and_uses_elapsed_critical_path() {
        let mut times = V41PrefillTimes::default();
        times.observe(2, 512, Duration::from_micros(512));
        assert_eq!(times.estimate(2, 256), Duration::from_micros(256));
        times.observe(2, 512, Duration::from_micros(1024));
        assert_eq!(times.estimate(2, 100), Duration::from_micros(120));
        times.observe(2, 0, Duration::from_secs(1));
        assert_eq!(times.estimate(2, 100), Duration::from_micros(120));
        assert_eq!(times.estimate(3, 100), Duration::from_millis(10));
    }

    /// The shared queue changes only *when* other sequences step, never the
    /// rows a prompt's chunks are cut at: a long prompt's call plan is the same
    /// cut the whole-prompt stream makes (design §5.3).
    #[test]
    fn long_prompt_chunks_are_cut_at_the_same_rows_as_the_whole_prompt_stream() {
        let plan = prompt_units(0, 0, 10_000, 2752, 3).unwrap();
        assert_eq!(plan.ends, vec![2752, 5504, 8256, 10_000]);
        assert_eq!(plan.calls, vec![0, 3]);
        assert_eq!(plan.depth, 3);
        // A restored frontier keeps the same cut from its end.
        let plan = prompt_units(300, 300, 10_000, 2752, 3).unwrap();
        assert_eq!(plan.ends, vec![3052, 5804, 8556, 10_000]);
        assert_eq!(plan.calls, vec![0, 3]);
    }
}

#[cfg(test)]
pub(crate) use super::scheduler::exercise_distributed_decode;

/// The next call's units over the rest of `span`, cut exactly as that call
/// will cut them ([`stream_units`] of up to `waves` units of `chunk_rows`): the
/// rows whose Engram gathers the current call's last units prefetch.
fn lookahead(span: Range<usize>, chunk_rows: usize, waves: usize) -> Result<Vec<Range<usize>>> {
    if span.is_empty() { return Ok(Vec::new()); }
    let rows = span.len().min(waves.max(1) * chunk_rows.max(1));
    Ok(stream_units(rows, chunk_rows.max(1), waves.max(1))?.into_iter()
        .map(|unit| span.start + unit.start..span.start + unit.end).collect())
}

#[cfg(test)]
mod stream_tests {
    /// A call's last units prefetch the next call's first units (#762 window
    /// cycles; prof39: each cycle's first Engram layer-1 gathers waited
    /// 40-150 ms cold, ~2.3 s of a 64K prefill on max).
    #[test]
    fn engram_look_ahead_units_follow_the_encoder_cut() {
        let plan = super::prompt_units(100, 100, 100 + 8256 + 4000, 2752, 3).unwrap();
        assert_eq!(plan.depth, 3);
        assert_eq!(plan.ends[..3], [100 + 2752, 100 + 5504, 100 + 8256]);
        assert_eq!(*plan.ends.last().unwrap(), 100 + 8256 + 4000);
        assert!(plan.ends.windows(2).all(|w| w[0] < w[1]));
        // An encoder-prefix replay runs first; the stream cut starts at its end.
        let replay = super::prompt_units(400 - 128, 400, 400 + 8256, 2752, 3).unwrap();
        assert_eq!(replay.ends[..4], [400, 400 + 2752, 400 + 5504, 400 + 8256]);
        // Calls: the replay window, then each stream call's units.
        assert_eq!(replay.calls, [0, 1]);
        assert_eq!(plan.calls, [0, 3]);
    }

    #[test]
    fn calls_look_ahead_at_the_next_calls_first_units() {
        assert_eq!(super::lookahead(8256..20000, 2752, 3).unwrap(), [8256..11008, 11008..13760, 13760..16512]);
        assert!(super::lookahead(9000..9000, 2752, 3).unwrap().is_empty());
        // A partial tail is cut as the tail call cuts it, not at the first unit's width.
        for rows in [2400, 6000] {
            let tail = super::lookahead(8256..8256 + rows, 2752, 3).unwrap();
            assert_eq!(tail, super::stream_units(rows, 2752, 3).unwrap().into_iter()
                .map(|unit| 8256 + unit.start..8256 + unit.end).collect::<Vec<_>>());
            assert_eq!(tail.last().unwrap().end, 8256 + rows);
        }
    }
}

#[cfg(test)]
mod unit_exit_tests {
    /// Final-review regressions, at the source of each exit path:
    /// D1 the lent prefill link returns on every exit (nothing fallible
    /// between its take and the loop whose errors land in `result`);
    /// C4 every units call resets its slots' windows in `Drop`;
    /// C2 the layer-20 boundary releases the lane's guarded scratch.
    #[test]
    fn units_exits_return_the_link_reset_windows_and_release_boundary_scratch() {
        let step = include_str!("prefill_target.rs");
        let take = step.find("let mut link = P::take_link(").unwrap();
        let take = take + step[take..].find(";\n").unwrap() + 1;
        let restore = step[take..].find("P::restore_link(&mut self.prefill.slots[0].1, link)").unwrap() + take;
        let between = &step[take..restore];
        let loop_start = between.find("while at < span.end && result.is_ok() {").unwrap();
        assert!(!between[..loop_start].contains('?'), "a fallible step between taking and lending the link (D1)");
        for (name, source) in [("1 RTX", include_str!("../v41_target_pass/encoder_units.rs")),
            ("2 RTX", include_str!("../v41_target_pass/distributed/encoder_units.rs"))] {
            let drop_impl = &source[source.find("impl Drop for UnitHooks").unwrap()..];
            let drop_body = &drop_impl[..drop_impl.find("\n}\n").unwrap()];
            assert!(drop_body.contains("select_window(") && drop_body.contains("RowWindow::NONE"),
                "{name}: UnitHooks::drop must reset every slot's window (C4)");
        }
        let one = include_str!("../v41_target_pass/encoder_units.rs");
        let end = &one[one.find("async fn end_unit_inner").unwrap()..];
        let end = &end[..end.find("\n    }\n").unwrap()];
        assert!(end.find("publish_encoder_boundary").unwrap() < end.find("lane.release_scratch()").unwrap(),
            "1 RTX: the boundary must release the lane's scratch after its publication (C2)");
        let two = include_str!("../v41_target_pass/distributed.rs");
        let finish = &two[two.find("async unsafe fn finish_encoder_stage").unwrap()..];
        let finish = &finish[..finish.find("\n    }\n").unwrap()];
        assert!(finish.find("finish_decoder_source").unwrap() < finish.find("lane.release_scratch()").unwrap(),
            "2 RTX: the boundary must release the lane's scratch after its production (C2)");
    }

    /// Review 1.1 (v41-final-deltas): remote layers always download their
    /// routes to the router ring, and a router refuses a window change while
    /// its ring is undrained. So every pass and unit settles its rings: 2 RTX
    /// units drain at commit as 1 RTX units do, both distributed pass exits
    /// drain a success and discard a failure (no success-only guard), the
    /// units' Drop discards before it resets the windows, and the router's
    /// discard lets its ring copies finish before the ring is reused.
    #[test]
    fn route_rings_settle_on_every_unit_and_pass_exit() {
        fn body<'s>(source: &'s str, start: &str) -> &'s str {
            let at = &source[source.find(start).unwrap_or_else(|| panic!("{start}"))..];
            &at[..at.find("\n    }\n").unwrap()]
        }
        let one = include_str!("../v41_target_pass/encoder_units.rs");
        let two = include_str!("../v41_target_pass/distributed/encoder_units.rs");
        let commit = body(one, "async fn commit_unit(");
        assert!(commit.find("drain_route_ring()").unwrap() < commit.find("commit_unit_inner").unwrap(), "1 RTX: {commit}");
        let commit = body(two, "async fn commit_unit(");
        assert!(commit.find("drain_route_rings()").is_some_and(|at| at < commit.find("commit_unit_inner").unwrap()),
            "2 RTX units must drain their route rings before committing: {commit}");
        let drop_impl = &two[two.find("impl Drop for UnitHooks").unwrap()..];
        let drop_body = &drop_impl[..drop_impl.find("\n}\n").unwrap()];
        assert!(drop_body.find("discard_route_rings()").is_some_and(|at| at < drop_body.find("RowWindow::NONE").unwrap()),
            "2 RTX: a failed or cancelled unit's captures are dropped before the window reset: {drop_body}");
        let pass = include_str!("../v41_target_pass/distributed.rs");
        let execute = body(pass, "pub async unsafe fn execute(");
        assert_eq!(execute.matches("self.settle_route_rings(result)").count(), 2, "{execute}");
        assert!(!execute.contains("if result.is_ok() { self.drain_route_rings()"), "{execute}");
        let settle = body(pass, "fn settle_route_rings(");
        assert!(settle.contains("self.discard_route_rings()?"), "{settle}");
        let router = include_str!("../v41_backbone_router.rs");
        let discard = body(router, "pub fn discard_ring(");
        assert!(discard.find("self.synchronize()?").unwrap() < discard.find("self.ring_pending.clear()").unwrap(), "{discard}");
    }
}
