//! Token-budget admission is checked at a completed stack boundary.
use super::*;
use cuteafd_core::{ImageKey, MediaKey, MediaSpan};
use cuteafd_engine::media::{EmbeddingCache, EncoderClient, MediaAdmission, MediaError, MediaPoll, MediaWaiter,
    RequestMedia};

pub(super) struct Prepared {
    pub job: NativeRequest,
    pub prompt: Vec<u32>,
    pub images: Vec<cuteafd_loader::V41ImageSpan>,
}
impl Prepared {
    pub fn new(
        mut job: NativeRequest,
        snapshot: &std::path::Path,
        limits: cuteafd_api::openai::NativeLimits,
    ) -> Result<Self> {
        let prompt = crate::shared::probe::prompt_ids(&job.probe, || {
            Ok(cuteafd_loader::encode_tokenizer_text(snapshot, &job.prompt, false)?.token_ids)
        })?;
        let (prompt, images) = if job.images.is_empty() {
            (prompt, Vec::new())
        } else {
            let expanded = cuteafd_loader::V41VisionPrompt::expand(
                &prompt,
                std::mem::take(&mut job.images),
                limits.context() as usize,
            )?;
            (expanded.tokens, expanded.images)
        };
        job.max_tokens = limits.output_for_prompt(prompt.len(), job.max_tokens)?;
        Ok(Self {
            job,
            prompt,
            images,
        })
    }
}

/// An admitted request (lease, restored prefix, draft slot) waiting for the
/// encodes its restored frontier still needs. Decode keeps running meanwhile.
pub(super) struct Leased {
    pub prepared: Prepared,
    pub id: u64,
    pub lease: CacheLease,
    pub slot: usize,
    pub image_keys: ImageKeys,
    pub hit: Option<(usize, Option<RetainedScores>)>,
    pub restore: (Instant, Instant),
    pub started: Instant,
}
impl Leased {
    pub fn into_admitted(self) -> (Prepared, u64, CacheLease, usize, ImageKeys,
        Option<(usize, Option<RetainedScores>)>, (Instant, Instant)) {
        (self.prepared, self.id, self.lease, self.slot, self.image_keys, self.hit, self.restore)
    }
    /// Release the lease and draft slot without answering the client.
    pub fn release<'a, C: DraftChain<'a>>(&self, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<()> {
        let target = requests.release_if_present(self.lease);
        let speculative = draft.map(|draft| draft.release(self.id)).transpose();
        target.and(speculative.map(|_| ()))
    }
    /// Answer the client with `error` (unavailable unless typed) and release.
    pub fn fail<'a, C: DraftChain<'a>>(self, error: anyhow::Error, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<()> {
        let failure = error.downcast_ref::<cuteafd_api::openai::NativeFailure>().cloned()
            .unwrap_or_else(|| cuteafd_api::openai::NativeFailure::Unavailable(format!("{error:#}")));
        let _ = self.prepared.job.events.send(Err(failure));
        self.release(requests, draft)
    }
}

/// The engine's media admission over V4.1's encoder, keyed by request slot: a
/// leased request stays in this table (its lease must be released whatever the
/// engine does with the waiter), the engine waiter carries only the slot.
pub(super) struct Media<C: EncoderClient> {
    admission: MediaAdmission<usize, C>,
    leased: Vec<Option<Leased>>,
    /// Image requests over the waiter limit, prepared host-side, no lease.
    pub backlog: std::collections::VecDeque<Prepared>,
    limit: usize,
}

/// Feature bytes per image row: 5120 BF16 values.
const IMAGE_ROW_BYTES: usize = 5120 * 2;

impl<C: EncoderClient> Media<C> {
    /// V4.1 keeps up to `CUTEAFD_V41_IMAGE_ADMISSIONS` (1..4, default 2, never
    /// every slot) leased requests waiting on encodes; the host cache holds
    /// features across requests that share an image.
    pub fn new(encoder: C, slots: usize) -> Result<Self> {
        let limit = image_admission_limit(slots)?;
        // Room for every waiter's largest prompt pinned at once, else a share of host RAM.
        let pinned = limit * cuteafd_loader::V41_MAX_IMAGES * 1024 * IMAGE_ROW_BYTES;
        let budget = EmbeddingCache::default_budget(crate::shared::prefix::budget::host_total()? as usize).max(pinned);
        let mut admission = MediaAdmission::new(EmbeddingCache::new(budget), encoder, limit);
        admission.set_encode_limits(cuteafd_loader::V41_MAX_IMAGES, usize::MAX);
        Ok(Self { admission, leased: (0..slots).map(|_| None).collect(), backlog: Default::default(), limit })
    }
    /// Leased requests waiting on encodes.
    pub fn len(&self) -> usize { self.leased.iter().flatten().count() }
    pub fn is_empty(&self) -> bool { self.len() == 0 && self.backlog.is_empty() }
    pub fn full(&self) -> bool { self.len() >= self.limit }
    pub fn holds(&self, slot: usize) -> bool { self.leased.get(slot).is_some_and(Option::is_some) }
    pub fn leased(&self) -> impl Iterator<Item = &Leased> { self.leased.iter().flatten() }
    /// Park a request at the encoder: `resume` is the frontier its restored
    /// prefix reached (V4.1 restores before encoding, so the engine's resume is
    /// exact and never reconciles).
    pub fn enqueue(&mut self, leased: Leased, resume: usize) -> std::result::Result<(), (Leased, anyhow::Error)> {
        let slot = leased.slot;
        let waiter = (|| {
            let spans = &leased.prepared.images;
            let media: Vec<MediaSpan> = spans.iter().map(|span| MediaSpan { start: span.start,
                len: span.image.grid().tokens(), key: MediaKey::Image(ImageKey(*span.image.identity())) }).collect();
            let jobs = spans.iter().map(|span|
                crate::families::deepseek_v41::v41_vision_encoder::Encoder::image_job(&span.image)).collect();
            let request = RequestMedia::new(media, 5120, leased.prepared.prompt.len())?;
            MediaWaiter::new(slot, request, jobs, resume)
        })();
        let waiter = match waiter { Ok(waiter) => waiter, Err(error) => return Err((leased, error.into())) };
        if let Err((_, error)) = self.admission.enqueue(waiter) { return Err((leased, media_failure(error))); }
        self.leased[slot] = Some(leased);
        Ok(())
    }
    /// Retire waiters whose client left, then collect the requests whose
    /// features are all installed. A failed encode answers its client.
    pub fn poll<'a, D: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>,
        mut draft: Option<&mut DraftRuntime<'_, 'a, D>>) -> Result<Vec<Leased>> {
        for slot in 0..self.leased.len() {
            if self.leased[slot].as_ref().is_some_and(|leased| leased.prepared.job.events.is_closed()) {
                self.leased[slot].take().unwrap().release(requests, draft.as_deref_mut())?;
            }
        }
        let mut ready = Vec::new();
        loop {
            let leased = &self.leased;
            match self.admission.poll(|&slot| leased[slot].is_none()) {
                MediaPoll::Ready(waiter) => {
                    let (slot, media) = waiter.into_parts();
                    let request = self.leased[slot].take().context("media waiter without a request")?;
                    match install(requests, request.lease, &request.prepared, &media) {
                        Ok(()) => ready.push(request),
                        Err(error) => request.fail(error, requests, draft.as_deref_mut())?,
                    }
                }
                MediaPoll::Failed(slot, error) => {
                    if let Some(request) = self.leased[slot].take() {
                        request.fail(media_failure(error), requests, draft.as_deref_mut())?;
                    }
                }
                MediaPoll::Pending | MediaPoll::Empty => break,
            }
        }
        Ok(ready)
    }
    /// Answer every waiter and release its owners (the scheduler stopped).
    pub fn stop<'a, D: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>,
        mut draft: Option<&mut DraftRuntime<'_, 'a, D>>) {
        let unavailable = || cuteafd_api::openai::NativeFailure::Unavailable("vision admission stopped".into());
        for request in self.leased.iter_mut().filter_map(Option::take) {
            if let Err(error) = request.fail(anyhow::anyhow!(unavailable()), requests, draft.as_deref_mut()) {
                tracing::warn!(%error, "releasing stopped image owners");
            }
        }
        for prepared in self.backlog.drain(..) { let _ = prepared.job.events.send(Err(unavailable())); }
        let _ = self.admission.poll(|_| true);
    }
    pub fn stats(&self) -> cuteafd_engine::media::MediaStats { self.admission.stats(0, 0) }
}

/// Leased requests that may wait on encodes at once: `CUTEAFD_V41_IMAGE_ADMISSIONS`
/// (1..4, default 2), always leaving a slot for text.
fn image_admission_limit(concurrency: usize) -> Result<usize> {
    let limit = std::env::var("CUTEAFD_V41_IMAGE_ADMISSIONS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("image admission limit")?
        .unwrap_or(2);
    bounded_image_admissions(concurrency, limit)
}

fn bounded_image_admissions(concurrency: usize, limit: usize) -> Result<usize> {
    ensure!((1..=4).contains(&limit), "CUTEAFD_V41_IMAGE_ADMISSIONS must be 1..4");
    Ok(limit.min(concurrency.saturating_sub(1).max(1)))
}

/// Copy each span's encoded rows into the request's V4.1 image table.
fn install(requests: &mut Requests<'_>, lease: CacheLease, prepared: &Prepared, media: &RequestMedia) -> Result<()> {
    let needed: std::collections::BTreeSet<usize> = (0..prepared.images.len())
        .filter(|&index| media.span_features(index).is_some()).collect();
    for index in needed {
        if requests.images(lease)?.has_features(index) { continue; }
        let rows = media.span_features(index).context("encoded span missing")?;
        ensure!(rows.len() == prepared.images[index].image.grid().tokens() * IMAGE_ROW_BYTES,
            cuteafd_api::openai::NativeFailure::Unavailable("invalid V4.1 feature reply".into()));
        requests.install_image_features(lease, index, rows.to_vec())?;
    }
    Ok(())
}

/// The client-facing failure for a media error: unavailable for encoder
/// faults, timeouts and saturation, a bad request for malformed input.
pub(super) fn media_failure(error: MediaError) -> anyhow::Error {
    use cuteafd_api::openai::NativeFailure;
    anyhow::anyhow!(match error {
        MediaError::Encoder(_) | MediaError::QueueFull | MediaError::CacheFull { .. } =>
            NativeFailure::Unavailable(format!("vision encoding failed: {error}")),
        error => NativeFailure::BadRequest(error.to_string()),
    })
}

/// Stop independent lanes for admission only when it can make progress. In
/// particular, a full KV pool plus a nonempty HTTP queue must not repeatedly
/// drain both lanes before they have executed another token.
#[derive(Clone, Copy, Default)]
pub(super) struct Wake<'p> {
    pub media_pending: bool,
    pub media_slots: usize,
    /// The deferred request and the release epoch it waits past.
    pub retry: Option<(&'p NativeRequest, u64)>,
    /// Shared prefill: lanes stop at a completed round once this decode debt is paid.
    pub prefill_deadline: Option<Instant>,
}
impl Wake<'_> {
    pub fn prefill_due(self, completed_rounds: u64) -> bool {
        completed_rounds > 0 && self.prefill_deadline.is_some_and(|deadline| Instant::now() >= deadline)
    }
    pub fn poll_media(self, completed_rounds: u64) -> bool {
        self.media_pending && completed_rounds > 0
    }
    /// `epoch` is the current release epoch: a deferred request retries once
    /// it moved (a request retired) or its client left.
    pub fn ready(self, active: usize, slots: usize, queued: bool, epoch: u64) -> bool {
        match self.retry {
            Some((job, _)) if job.events.is_closed() => true,
            Some((_, blocked)) => active + self.media_slots < slots && epoch != blocked,
            None => active + self.media_slots < slots && queued,
        }
    }
}

pub(super) fn remaining_budget(
    tokens: usize,
    remaining_output: usize,
    committed: u64,
) -> Result<u32> {
    let end = tokens
        .checked_add(remaining_output)
        .context("request token budget overflow")?;
    let append = (end as u64)
        .checked_sub(committed)
        .context("cache exceeds request token budget")?;
    Ok(append
        .try_into()
        .context("request token budget exceeds u32")?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_wake_waits_for_committed_round_and_text_fast_loop_is_unchanged() {
        let media = Wake { media_pending:true, ..Default::default() };
        assert!(!media.poll_media(0));
        assert!(media.poll_media(1));
        let full = Wake { media_pending:true, media_slots:2, ..Default::default() };
        assert!(!full.ready(14, 16, true, 0));
        assert!(full.ready(13, 16, true, 0));
        assert!(!Wake::default().ready(16, 16, true, 0));
        assert!(!Wake::default().poll_media(1));
        assert!(!Wake::default().poll_media(u64::MAX));
    }
    #[test]
    fn parked_image_limit_preserves_a_text_slot() -> Result<()> {
        assert_eq!(bounded_image_admissions(16, 2)?, 2);
        assert_eq!(bounded_image_admissions(2, 4)?, 1);
        assert_eq!(bounded_image_admissions(1, 2)?, 1);
        assert!(bounded_image_admissions(16, 0).is_err());
        assert!(bounded_image_admissions(16, 5).is_err());
        Ok(())
    }
    fn request() -> (NativeRequest, mpsc::UnboundedReceiver<std::result::Result<InferenceChunk, cuteafd_api::openai::NativeFailure>>) {
        let (events, output) = mpsc::unbounded_channel();
        (NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), media: Vec::new(),
            audio: Vec::new(), max_tokens: 1, sampling: Default::default(), stop_token_ids: Vec::new(), events,
            usage: None, probe: None }, output)
    }
    #[test]
    fn blocked_admission_does_not_join_lanes_until_a_request_retires() {
        let (job, _output) = request();
        let blocked = Wake { retry: Some((&job, 7)), ..Wake::default() };
        // No retirement since the request was deferred: lanes keep decoding.
        assert!(!blocked.ready(2, 16, true, 7));
        assert!(!blocked.ready(2, 16, false, 7));
        // A request retired: stop for admission even with an empty channel.
        assert!(blocked.ready(1, 16, false, 8));
        assert!(!Wake::default().ready(16, 16, true, 0));
        assert!(!Wake::default().ready(2, 16, false, 0));
        assert!(Wake::default().ready(2, 16, true, 0));
    }
    #[test]
    fn cancelled_pending_request_wakes_admission_without_waiting_for_retirement() {
        let (job, output) = request();
        let wake = Wake { retry: Some((&job, 3)), ..Wake::default() };
        assert!(!wake.ready(2, 16, true, 3));
        drop(output);
        assert!(wake.ready(2, 16, true, 3));
    }
    #[test]
    fn budget_includes_uncommitted_anchor_and_entire_output_allowance() -> Result<()> {
        assert_eq!(remaining_budget(100, 20, 0)?, 120);
        assert_eq!(remaining_budget(101, 19, 100)?, 20);
        assert_eq!(remaining_budget(108, 12, 107)?, 13);
        assert_eq!(remaining_budget(120, 0, 120)?, 0);
        assert!(remaining_budget(120, 0, 121).is_err());
        assert!(remaining_budget(usize::MAX, 1, 0).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_output_budget_detects_pressure_before_prefill() -> Result<()> {
        use crate::families::deepseek_v41::v41_backbone_cache::BackboneCache;
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let mut cache =
            BackboneCache::new(&lib, 2, [2; 4], BackboneCache::device_bytes(2, [2; 4])?)?;
        let first = cache.begin_request(0, 1)?;
        let second = cache.begin_request(1, 2)?;
        // Both 256-token prompts fit, but their 256-token output allowances do
        // not. The old prompt-only admission would discover this during decode.
        cache.check_append_capacity(&[(first, 256), (second, 256)])?;
        let budget = remaining_budget(256, 256, 0)?;
        let error = cache
            .check_append_capacity(&[(first, budget), (second, budget)])
            .unwrap_err();
        assert!(error
            .downcast_ref::<crate::families::deepseek_v41::v41_compressor::SourcePoolExhausted>()
            .is_some());
        // A failed check must return its temporary reservations. Either request
        // can run alone, and the other can be retried after its peer retires.
        cache.check_append_capacity(&[(first, budget)])?;
        cache.release(&[first])?;
        cache.check_append_capacity(&[(second, budget)])?;
        cache.release(&[second])?;
        Ok(())
    }
}
