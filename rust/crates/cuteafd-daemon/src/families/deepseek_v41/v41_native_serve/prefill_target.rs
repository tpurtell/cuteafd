//! Static prefill operations for the shared admission/continuation workflow.
use super::*;
use crate::families::deepseek_v41::v41_backbone_cache::{CacheLease, CacheStage};
use crate::families::deepseek_v41::v41_block::EncoderSuffix;
use crate::shared::memory::device::DeviceOwner;
use crate::families::deepseek_v41::v41_requests::RequestBatch;
use crate::families::deepseek_v41::v41_target_pass::{DistributedTargetPass, VerificationTarget};
use speculative::DraftChain;
use std::cell::RefCell;

#[cfg(test)]
pub(crate) fn exercise_prefill<'a, P: PrefillTarget<'a>, C: DraftChain<'a>>(
    lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime, pass: &mut P, other: &mut P,
    requests: &mut Requests<'a>, transports: [&mut P::Transport; 2], lease: CacheLease,
    tokens: &[u32], chunk_rows: usize, draft: Option<&mut DraftRuntime<'_, 'a, C>>,
) -> Result<u32> {
    let (events, _receive) = mpsc::unbounded_channel();
    let job = NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), media: Vec::new(), audio: Vec::new(), max_tokens: 16, sampling: Default::default(), stop_token_ids: Vec::new(), events, probe: None };
    let [first_transport, second_transport] = transports;
    super::prefill(lib, runtime, pass, other, requests, first_transport, second_transport,
        lease, tokens, chunk_rows, &job, draft, &mut || Ok(()))?.select(None)
}

/// Only request-owned storage survives a yield; every batch has committed and
/// both execution/transport lanes have drained before a decode wave can start.
pub(super) enum PrefillProgress<S> {
    Start,
    Encoder { suffix: S, next: usize, replay_end: usize, streamed: bool, index: usize },
    Replay { suffix: S },
    Continuation { next: usize },
    Done,
}

impl<S> Default for PrefillProgress<S> {
    fn default() -> Self { Self::Start }
}

impl<S> PrefillProgress<S> {
    pub(super) fn wave_shape(&self, tokens: usize, chunk_rows: usize) -> (&'static str, usize) {
        match self {
            Self::Start => ("start", tokens.min(chunk_rows)),
            Self::Encoder { next, replay_end, .. } => {
                let end = if next < replay_end { *replay_end } else { tokens };
                ("encoder", end.saturating_sub(*next).min(chunk_rows))
            }
            Self::Replay { .. } => ("replay", tokens.min(128)),
            Self::Continuation { next } => ("continuation", tokens.saturating_sub(*next).min(chunk_rows)),
            Self::Done => ("done", 0),
        }
    }

    pub(super) fn step<'a, P: PrefillTarget<'a, Suffix = S>, C: DraftChain<'a>>(&mut self,
        lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime, pass: &mut P, other: &mut P,
        requests: &mut Requests<'a>, transports: [&mut P::Transport; 2], lease: CacheLease,
        tokens: &[u32], chunk_rows: usize, job: &NativeRequest,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, hold: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<TokenScores>> {
        use crate::families::deepseek_v41::v41_backbone_cache::CacheWork;
        ensure!(!job.events.is_closed(), "client disconnected");
        let [transport, other_transport] = transports;
        if matches!(self, Self::Start) {
            let cached = requests.cache().committed_end(lease)? as usize;
            let stage = requests.cache().stage(lease)?;
            if cached > 0 && stage == CacheStage::Full {
                *self = Self::Continuation { next: cached };
            } else {
                let suffix = pass.new_suffix(lib, tokens.len() as u64)?;
                let next = if stage == CacheStage::EncoderReplay {
                    let start = requests.cache().history_end(lease)? as usize;
                    ensure!(cached - start <= 128, "encoder prefix replay exceeds one window");
                    start
                } else {
                    if stage == CacheStage::Full { requests.begin_encoder(lease, tokens.len() as u64)?; }
                    cached
                };
                *self = Self::Encoder { suffix, next, replay_end: cached,
                    streamed: tokens.len().saturating_sub(cached).div_ceil(chunk_rows) > 1, index: 0 };
            }
        }
        prefill_hold(hold);
        let (stage, rows) = self.wave_shape(tokens.len(), chunk_rows);
        super::prefill_wave_started(Some(requests.cache().request_id(lease)?), rows, stage);
        match self {
            Self::Encoder { suffix, next, replay_end, streamed, index } => {
                let replay = *next < *replay_end;
                let end = (*next + chunk_rows).min(if replay { *replay_end } else { tokens.len() });
                let chunk = &tokens[*next..end];
                let started = Instant::now();
                if !replay && *streamed {
                    // Keep the original chunk shapes and alternating lane ownership.
                    // A singleton stream still publishes every compressed/index boundary.
                    let (first, second, a, b) = if *index % 2 == 0 {
                        (pass, other, transport, other_transport)
                    } else { (other, pass, other_transport, transport) };
                    // SAFETY: this wave owns both lanes and retains its suffix until drained.
                    runtime.block_on(unsafe { first.execute_encoder_stream_held(second, requests,
                        lease, &[chunk], [a, b], suffix, &|| !job.events.is_closed(), &|| Ok(())) })?;
                    *index += 1;
                } else {
                    let mut batch = requests.prepare(&[RequestTokens { lease, tokens: chunk,
                        image_mask: None, kind: ExpertV2SourceKind::Prefill }])?;
                    let result = (|| -> Result<()> {
                        // SAFETY: the batch and suffix belong to this live request; completion precedes yielding.
                        runtime.block_on(unsafe { pass.encoder_part(requests, &mut batch, transport, suffix) })?;
                        ensure!(!job.events.is_closed(), "client disconnected");
                        runtime.block_on(pass.commit_prefill::<C>(requests, &mut batch, None, chunk.len() as u32))
                    })();
                    if result.is_err() { pass.discard(&mut batch)?; }
                    result?;
                    if !replay {
                        console::totals::prefill(chunk.len());
                        console::Prefill::done(console::PrefillKind::Single, 0, 0, 1, chunk.len(), started);
                    }
                }
                *next = end;
                if end == tokens.len() {
                    let state = std::mem::replace(self, Self::Done);
                    if let Self::Encoder { suffix, .. } = state { *self = Self::Replay { suffix }; }
                }
                Ok(None)
            }
            Self::Replay { suffix } => {
                let start = requests.begin_decoder_replay(lease)?;
                let rows = (tokens.len() as u64 - start) as u32;
                let mut batch = requests.prepare_replay(&[CacheWork { lease, tokens: rows, kind: ExpertV2SourceKind::Prefill }])?;
                let started = Instant::now();
                let result = (|| -> Result<TokenScores> {
                    // SAFETY: all encoder rows committed and the retained suffix matches this replay.
                    let bytes = runtime.block_on(unsafe { pass.prefill_logits(lib, requests, &mut batch,
                        transport, &[rows as usize - 1], Some(suffix)) })?;
                    let scores = TokenScores::new(bytes)?;
                    ensure!(!job.events.is_closed(), "client disconnected");
                    runtime.block_on(pass.commit_prefill(requests, &mut batch, draft, rows))?;
                    Ok(scores)
                })();
                if result.is_err() { pass.discard(&mut batch)?; }
                console::Prefill::done(console::PrefillKind::Replay, 0, 0, 1, rows as usize, started);
                let scores = result?;
                *self = Self::Done;
                Ok(Some(scores))
            }
            Self::Continuation { next } => {
                let end = (*next + chunk_rows).min(tokens.len());
                let scores = super::prefill_continuation(lib, runtime, pass, requests, transport,
                    lease, &tokens[*next..end], chunk_rows, job, draft, &mut || Ok(()))?;
                *next = end;
                if end == tokens.len() { *self = Self::Done; Ok(Some(scores)) } else { Ok(None) }
            }
            Self::Start | Self::Done => anyhow::bail!("invalid prefill progress"),
        }
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
pub(crate) use super::scheduler::exercise_distributed_decode;
