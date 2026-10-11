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
    lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime, pass: &mut P, other: &mut P,
    requests: &mut Requests<'a>, transports: [&mut P::Transport; 2], lease: CacheLease,
    tokens: &[u32], chunk_rows: usize, draft: Option<&mut DraftRuntime<'_, 'a, C>>,
) -> Result<u32> {
    let (events, _receive) = mpsc::unbounded_channel();
    let job = NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), media: Vec::new(), audio: Vec::new(), max_tokens: 16, sampling: Default::default(), stop_token_ids: Vec::new(), events, usage: None, probe: None };
    let [first_transport, second_transport] = transports;
    super::prefill(lib, runtime, pass, other, requests, first_transport, second_transport,
        lease, tokens, chunk_rows, &job, draft, &mut || Ok(()))?.select(None)
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

impl V41PrefillTimes {
    fn encoder_unit(&self, tokens: usize, chunk_rows: usize, next: usize,
        replay_end: usize, streamed: bool, budget: ChunkBudget) -> V41Unit {
        if next < replay_end {
            return V41Unit::ReplayChunk(next..(next + chunk_rows).min(replay_end));
        }
        if next == tokens { return V41Unit::DecoderReplay; }
        let end = (next + chunk_rows).min(tokens);
        if !streamed { return V41Unit::Single(next..end); }
        let mut chunks = vec![next..end];
        let whole = budget.target.is_none() && budget.row_cap == usize::MAX;
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
    pub(crate) pass: &'s mut P,
    pub(crate) other: &'s mut P,
    pub(crate) requests: &'s mut Requests<'a>,
    pub(crate) transport: &'s mut P::Transport,
    pub(crate) other_transport: &'s mut P::Transport,
    pub(crate) lease: CacheLease,
    pub(crate) tokens: &'s [u32],
    pub(crate) chunk_rows: usize,
    pub(crate) job: &'s NativeRequest,
    pub(crate) draft: Option<&'s mut DraftRuntime<'w, 'a, C>>,
    pub(crate) hold: &'s mut dyn FnMut() -> Result<()>,
    pub(crate) times: &'s mut V41PrefillTimes,
}

impl<'a, P: PrefillTarget<'a>, C: DraftChain<'a>> V41PrefillStep<'_, '_, 'a, P, C> {
    fn encoder_unit(&self, next: usize, replay_end: usize, streamed: bool,
        budget: ChunkBudget) -> V41Unit {
        self.times.encoder_unit(self.tokens.len(), self.chunk_rows, next, replay_end, streamed, budget)
    }

    fn initialize(&mut self, cursor: &mut V41Prefill<P::Suffix>) -> Result<()> {
        if !matches!(cursor.state, PrefillState::Start) { return Ok(()); }
        let end = self.tokens.len() as u64;
        let cached = self.requests.cache().committed_end(self.lease)? as usize;
        let stage = self.requests.cache().stage(self.lease)?;
        if cached > 0 && stage == CacheStage::Full {
            cursor.state = PrefillState::Continuation { next: cached };
            return Ok(());
        }
        let suffix = self.pass.new_suffix(self.lib, end)?;
        let next = if stage == CacheStage::EncoderReplay {
            let start = self.requests.cache().history_end(self.lease)? as usize;
            ensure!(cached - start <= 128, "encoder prefix replay exceeds one window");
            start
        } else {
            if stage == CacheStage::Full { self.requests.begin_encoder(self.lease, end)?; }
            cached
        };
        cursor.state = PrefillState::Encoder { suffix, next, replay_end: cached,
            streamed: self.tokens.len().saturating_sub(cached).div_ceil(self.chunk_rows) > 1 };
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
                    self.encoder_unit(next, cached,
                        self.tokens.len().saturating_sub(cached).div_ceil(self.chunk_rows) > 1, budget)
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
                let result = (|| -> Result<()> {
                    // SAFETY: the live request owns the batch/suffix until execution and commit drain.
                    self.runtime.block_on(unsafe { self.pass.encoder_part(self.requests, &mut batch, self.transport, suffix) })?;
                    ensure!(!self.job.events.is_closed(), "client disconnected");
                    self.runtime.block_on(self.pass.commit_prefill::<C>(self.requests, &mut batch, None, chunk.len() as u32))
                })();
                if result.is_err() { self.pass.discard(&mut batch)?; }
                result?;
                if matches!(unit.work, V41Unit::Single(_)) {
                    console::totals::prefill(chunk.len());
                    console::Prefill::done(console::PrefillKind::Single, 0, 0, 1, chunk.len(), started);
                    tracing::debug!(target: "cuteafd::timing", rows=chunk.len(), total_us=started.elapsed().as_micros() as u64, "target encoder step");
                }
                *next = range.end;
            }
            (V41Unit::Stream(ranges), PrefillState::Encoder { suffix, next, replay_end, .. }) => {
                let chunks: Vec<_> = ranges.iter().map(|range| &self.tokens[range.clone()]).collect();
                let started = Instant::now();
                // Both lanes call this synchronous hook without nesting.
                let hold = RefCell::new(&mut *self.hold);
                let before_chunk = || -> Result<()> { prefill_hold(&mut *hold.borrow_mut()); Ok(()) };
                // Preserve alternating lane ownership after a bounded singleton stream.
                let odd = (*next - *replay_end) / self.chunk_rows % 2 != 0;
                let (first, second, a, b) = if odd {
                    (&mut *self.other, &mut *self.pass, &mut *self.other_transport, &mut *self.transport)
                } else {
                    (&mut *self.pass, &mut *self.other, &mut *self.transport, &mut *self.other_transport)
                };
                // SAFETY: independent lanes retain the same request/suffix through drained completion.
                self.runtime.block_on(unsafe { first.execute_encoder_stream_held(second, self.requests,
                    self.lease, &chunks, [a, b], suffix,
                    &|| !self.job.events.is_closed(), &before_chunk) })?;
                tracing::debug!(target: "cuteafd::timing", rows=self.tokens.len(),
                    total_us=started.elapsed().as_micros() as u64, "target encoder stream");
                *next = ranges.last().context("empty encoder stream")?.end;
            }
            (V41Unit::DecoderReplay, PrefillState::Replay { suffix } | PrefillState::Encoder { suffix, .. }) => {
                ensure!(!self.job.events.is_closed(), "client disconnected");
                let start = self.requests.begin_decoder_replay(self.lease)?;
                let rows = (self.tokens.len() as u64 - start) as u32;
                let mut batch = self.requests.prepare_replay(&[CacheWork { lease: self.lease, tokens: rows, kind: ExpertV2SourceKind::Prefill }])?;
                let started = Instant::now();
                let result = (|| -> Result<RetainedScores> {
                    // SAFETY: all encoder rows committed and the retained suffix matches this replay.
                    let bytes = self.runtime.block_on(unsafe { self.pass.prefill_logits(self.lib, self.requests,
                        &mut batch, self.transport, &[rows as usize - 1], Some(suffix)) })?;
                    let scores = RetainedScores::new(scores::VOCAB, bytes)?;
                    ensure!(!self.job.events.is_closed(), "client disconnected");
                    self.runtime.block_on(self.pass.commit_prefill(self.requests, &mut batch, self.draft.as_deref_mut(), rows))?;
                    Ok(scores)
                })();
                if result.is_err() { self.pass.discard(&mut batch)?; }
                console::Prefill::done(console::PrefillKind::Replay, 0, 0, 1, rows as usize, started);
                tracing::debug!(target: "cuteafd::timing", rows, total_us=started.elapsed().as_micros() as u64, "target decoder replay");
                let scores = result?;
                cursor.state = PrefillState::Done;
                return Ok(Some(scores));
            }
            (V41Unit::Continuation(range), PrefillState::Continuation { next }) => {
                let scores = super::prefill_continuation(self.lib, self.runtime, self.pass, self.requests,
                    self.transport, self.lease, &self.tokens[range.clone()], self.chunk_rows, self.job,
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

pub(crate) trait PrefillTarget<'a>: VerificationTarget<'a> + Sized {
    type Suffix;
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix>;
    /// # Safety
    /// Batch and suffix belong to this request and layout; all owners remain
    /// live through completion/cancellation, with no conflicting access.
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()>;
    /// # Safety
    /// The passes have independent workspaces and the same layout/model. Input
    /// chunks and suffix follow the encoder stream's request/cache contract.
    async unsafe fn encoder_stream(&mut self, other: &mut Self, requests: &mut Requests<'a>, lease: CacheLease,
        chunks: &[&[u32]], transports: [&mut Self::Transport; 2], suffix: &mut Self::Suffix,
        keep_running: &dyn Fn() -> bool) -> Result<()>;
    /// # Safety
    /// Same contract as `encoder_stream`, plus: `before_chunk` runs synchronously
    /// before each dispatched chunk (host-cache store pacing, packet HC-9).
    async unsafe fn execute_encoder_stream_held(&mut self, other: &mut Self, requests: &mut Requests<'a>,
        lease: CacheLease, chunks: &[&[u32]], transports: [&mut Self::Transport; 2],
        suffix: &mut Self::Suffix, keep_running: &dyn Fn() -> bool,
        before_chunk: &dyn Fn() -> Result<()>) -> Result<()> {
        // Default for passes without a held dispatch (dual-RTX lane): the hold is
        // load-bearing on the single-GPU 5090 path only; log once so degradation
        // is observable rather than silent (rc2 lesson).
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!("held encoder stream unavailable on this pass; HC-9 per-chunk hold inactive");
        }
        self.encoder_stream(other, requests, lease, chunks, transports, suffix, keep_running).await
    }
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
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix> {
        EncoderSuffix::new(lib, end, EncoderSuffix::device_bytes(end)?)
    }
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()> {
        if batch.cache()?.stage() == CacheStage::EncoderReplay {
            unsafe { self.execute_encoder_replay(requests, batch, transport, 0, suffix).await }
        } else { unsafe { self.execute_encoder(requests, batch, transport, 0, suffix).await } }
    }
    async unsafe fn encoder_stream(&mut self, other: &mut Self, requests: &mut Requests<'a>, lease: CacheLease,
        chunks: &[&[u32]], transports: [&mut Self::Transport; 2], suffix: &mut Self::Suffix,
        keep_running: &dyn Fn() -> bool) -> Result<()> {
        unsafe { self.execute_encoder_stream(other, requests, lease, chunks, transports, suffix, keep_running).await }
    }
    async unsafe fn execute_encoder_stream_held(&mut self, other: &mut Self, requests: &mut Requests<'a>,
        lease: CacheLease, chunks: &[&[u32]], transports: [&mut Self::Transport; 2],
        suffix: &mut Self::Suffix, keep_running: &dyn Fn() -> bool,
        before_chunk: &dyn Fn() -> Result<()>) -> Result<()> {
        unsafe { self.execute_encoder_stream_held(other, requests, lease, chunks, transports, suffix,
            keep_running, before_chunk).await }
    }
    async unsafe fn prefill_logits(&mut self, lib: &'a NativeLibrary, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, selected: &[usize],
        suffix: Option<&Self::Suffix>) -> Result<Vec<u8>> {
        let logits = if let Some(suffix) = suffix {
            let encoder = suffix.output()?;
            unsafe { self.execute_replay(requests, batch, transport, 0, selected, &encoder).await? }
        } else { unsafe { self.execute(requests, batch, transport, 0, selected).await? } };
        let mut bytes = vec![0; logits.logits.bytes];
        lib.copy_d2h(&mut bytes, logits.logits)?;
        Ok(bytes)
    }
    async fn commit_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: u32) -> Result<()> {
        if let Some(draft) = draft { draft.commit(self, requests, batch, accepted) }
        else { self.commit(requests, batch, &[accepted]) }
    }
}
impl<'a> PrefillTarget<'a> for DistributedTargetPass<'_, 'a> {
    type Suffix = DeviceOwner<'a, EncoderSuffix<'a>>;
    fn new_suffix(&self, lib: &'a NativeLibrary, end: u64) -> Result<Self::Suffix> {
        self.encoder_device()?.own(|| EncoderSuffix::new(lib, end, EncoderSuffix::device_bytes(end)?))
    }
    async unsafe fn encoder_part(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        transport: &mut Self::Transport, suffix: &mut Self::Suffix) -> Result<()> {
        unsafe { self.execute(&RefCell::new(requests), batch, transport, 0, &[], Some(suffix), None, false).await }
    }
    async unsafe fn encoder_stream(&mut self, other: &mut Self, requests: &mut Requests<'a>, lease: CacheLease,
        chunks: &[&[u32]], transports: [&mut Self::Transport; 2], suffix: &mut Self::Suffix,
        keep_running: &dyn Fn() -> bool) -> Result<()> {
        unsafe { self.execute_encoder_stream(other, requests, lease, chunks, transports, suffix, keep_running).await }
    }
    async unsafe fn prefill_logits(&mut self, _lib: &'a NativeLibrary, requests: &mut Requests<'a>,
        batch: &mut RequestBatch, transport: &mut Self::Transport, selected: &[usize],
        suffix: Option<&Self::Suffix>) -> Result<Vec<u8>> {
        let encoder = suffix.map(|suffix| suffix.output()).transpose()?;
        unsafe { self.execute(&RefCell::new(requests), batch, transport, 0, selected, None, encoder.as_ref(), false).await?; }
        self.download_logits(batch, &(0..selected.len()).collect::<Vec<_>>()).await
    }
    async fn commit_prefill<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, accepted: u32) -> Result<()> {
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
            draft.begin_queued_commit(0, commit.pass, commit.requests, commit.batch, &[accepted])?;
        }
        commit.pass.enqueue_cache_commit(commit.requests, commit.batch, &[accepted])?;
        loop {
            let target_ready = commit.pass.poll_cache_commit()?;
            let draft_ready = commit.draft.as_ref().map(|draft| draft.poll_queued_commit(0)).transpose()?.unwrap_or(true);
            if target_ready && draft_ready { break; }
            tokio::task::yield_now().await;
        }
        if let Some(draft) = &mut commit.draft {
            draft.finish_queued_commit(0, commit.pass, commit.requests, commit.batch, &[accepted])?;
        } else { commit.pass.commit(commit.requests, commit.batch, &[accepted])?; }
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
        assert_eq!(stream(times.encoder_unit(1050, 256, 0, 0, true, ChunkBudget::WHOLE)),
            [0..256, 256..512, 512..768, 768..1024, 1024..1050]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 300, 300, true, ChunkBudget::WHOLE)),
            [300..556, 556..812, 812..1050]);
        assert!(matches!(times.encoder_unit(1050, 256, 950, 950, false, ChunkBudget::WHOLE), V41Unit::Single(range) if range == (950..1050)));
    }

    #[test]
    fn whole_matches_original_chunk_iterator_at_legal_capacities() {
        let times = V41PrefillTimes::default();
        for chunk_rows in [80usize, 256, 1024, 4096] {
            for cached in [0, 2, 128, 300] {
                for remaining in [0, 1, chunk_rows - 1, chunk_rows, chunk_rows + 1, 2 * chunk_rows + 17] {
                    let tokens = cached + remaining;
                    let streamed = remaining.div_ceil(chunk_rows) > 1;
                    let unit = times.encoder_unit(tokens, chunk_rows, cached, cached, streamed, ChunkBudget::WHOLE);
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
        assert_eq!(stream(times.encoder_unit(1050, 256, 0, 0, true, budget)), [0..256]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 0, 0, true,
            ChunkBudget { target: Some(Duration::from_micros(512)), ..budget })), [0..256, 256..512]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 0, 0, true,
            ChunkBudget { target: Some(Duration::ZERO), row_cap: 0 })), [0..256]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 0, 0, true,
            ChunkBudget { target: Some(Duration::from_secs(1)), row_cap: 511 })), [0..256]);
        assert_eq!(stream(times.encoder_unit(1050, 256, 0, 0, true,
            ChunkBudget { target: Some(Duration::from_secs(1)), row_cap: 512 })), [0..256, 256..512]);
        // A streamed prompt's final singleton must not switch to encoder_part.
        assert_eq!(stream(times.encoder_unit(1050, 256, 1024, 0, true, budget)), [1024..1050]);
    }

    #[test]
    fn history_replay_stops_at_cached_frontier_before_the_encoder_stream() {
        let times = V41PrefillTimes::default();
        assert!(matches!(times.encoder_unit(1024, 80, 128, 256, true, ChunkBudget::WHOLE), V41Unit::ReplayChunk(range) if range == (128..208)));
        assert!(matches!(times.encoder_unit(1024, 80, 208, 256, true, ChunkBudget::WHOLE), V41Unit::ReplayChunk(range) if range == (208..256)));
        assert!(matches!(times.encoder_unit(1024, 80, 1024, 256, true, ChunkBudget::WHOLE), V41Unit::DecoderReplay));
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
}

#[cfg(test)]
pub(crate) use super::scheduler::exercise_distributed_decode;
