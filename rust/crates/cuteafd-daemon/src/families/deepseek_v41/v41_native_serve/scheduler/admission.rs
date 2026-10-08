//! Token-budget admission is checked at a completed stack boundary.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

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

pub(super) struct ImageAdmission {
    pub prepared: Prepared,
    pub id: u64,
    pub lease: CacheLease,
    pub image_keys: ImageKeys,
    pub hit: Option<(usize, Option<TokenScores>)>,
    pub restore: (Instant, Instant),
    pub needed: Vec<usize>,
    pub next: usize,
    pub ticket: Option<(cuteafd_engine::media::EncoderTicket, Instant)>,
    pub started: Instant,
}
impl ImageAdmission {
    pub fn ready(&self) -> bool {
        self.next == self.needed.len() && self.ticket.is_none()
    }
    pub fn cancel(&mut self, vision: &mut impl cuteafd_engine::media::EncoderClient) {
        if let Some((ticket, _)) = self.ticket.take() {
            vision.cancel(ticket);
        }
    }
    pub fn poll(
        &mut self,
        vision: &mut crate::families::deepseek_v41::v41_vision_encoder::Encoder,
        requests: &mut Requests<'_>,
    ) -> Result<()> {
        if self.ready() { return Ok(()); }
        let index = self.needed[self.next];
        let image = &self.prepared.images[index].image;
        if poll_image_ticket(&mut self.ticket, vision,
            cuteafd_core::MediaKey::Image(cuteafd_core::ImageKey(*image.identity())),
            image.grid().tokens() * 5120 * 2,
            || crate::families::deepseek_v41::v41_vision_encoder::Encoder::image_job(image),
            |output| requests.install_image_features(self.lease, index, output.features.to_vec()))? {
            self.next += 1;
        }

        Ok(())
    }
}

/// Poll without waiting; a saturated owner retries at the next committed decode boundary.
fn poll_image_ticket(
    ticket: &mut Option<(cuteafd_engine::media::EncoderTicket, Instant)>,
    client: &mut impl cuteafd_engine::media::EncoderClient,
    key: cuteafd_core::MediaKey,
    bytes: usize,
    job: impl FnOnce() -> cuteafd_engine::media::EncodeJob,
    install: impl FnOnce(cuteafd_engine::media::EncodeOutput) -> Result<()>,
) -> Result<bool> {
    use cuteafd_engine::media::MediaError;
    let unavailable = |message: String| anyhow::anyhow!(cuteafd_api::openai::NativeFailure::Unavailable(message));
    if let Some((id, started)) = *ticket {
        if started.elapsed() >= Duration::from_secs(60) {
            return Err(unavailable("vision encoding timed out".into()));
        }
        let Some(output) = client.poll(id) else { return Ok(false) };
        *ticket = None;
        let output = output.map_err(|error| unavailable(error.to_string()))?;
        if output.key != key || output.features.len() != bytes {
            return Err(unavailable("invalid V4.1 feature reply".into()));
        }
        let roundtrip_ms = started.elapsed().as_secs_f64()*1000.0;
        let finished_unix_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .context("image encoder telemetry clock")?.as_secs_f64()*1000.0;
        tracing::info!(tokens=bytes / (5120 * 2), owner_ms=output.elapsed_ms, roundtrip_ms,
            started_unix_ms=finished_unix_ms-roundtrip_ms, finished_unix_ms,
            "V4.1 asynchronous image encoder roundtrip");
        install(output)?;
        return Ok(true);
    }
    match client.submit(job()) {
        Ok(id) => *ticket = Some((id, Instant::now())),
        Err(MediaError::QueueFull) => (),
        Err(error) => return Err(unavailable(error.to_string())),
    }
    Ok(false)
}

pub(super) fn image_admission_limit(concurrency: usize) -> Result<usize> {
    let limit = std::env::var("CUTEAFD_V41_IMAGE_ADMISSIONS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()
        .context("image admission limit")?
        .unwrap_or(2);
    bounded_image_admissions(concurrency, limit)
}

fn bounded_image_admissions(concurrency: usize, limit: usize) -> Result<usize> {
    ensure!(
        (1..=4).contains(&limit),
        "CUTEAFD_V41_IMAGE_ADMISSIONS must be 1..4"
    );
    Ok(limit.min(concurrency.saturating_sub(1).max(1)))
}

/// A blocked request owns host input only; it holds no GPU lease while waiting.
pub(super) struct Pending {
    pub prepared: Prepared,
    pub active_when_blocked: usize,
}

/// Stop independent lanes for admission only when it can make progress. In
/// particular, a full KV pool plus a nonempty HTTP queue must not repeatedly
/// drain both lanes before they have executed another token.
#[derive(Clone, Copy, Default)]
pub(super) struct Wake<'p> {
    pub media_pending: bool,
    pub media_slots: usize,
    pub host_pending: bool,
    pub blocked_at: Option<usize>,
    pub pending: Option<&'p NativeRequest>,
}
impl Wake<'_> {
    pub fn poll_media(self, completed_rounds: u64) -> bool {
        self.media_pending && completed_rounds > 0
    }
    pub fn ready(self, active: usize, slots: usize, queued: bool) -> bool {
        if self.pending.is_some_and(|p| p.events.is_closed()) {
            return true;
        }
        active + self.media_slots < slots
            && match self.blocked_at {
                Some(previous) => active < previous,
                None => queued || self.host_pending,
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

/// Largest output allowance up to `maximum` that `fits`; `None` when not even one token does.
/// A request with active peers waits instead; call only for an otherwise idle pool.
pub(super) fn fit_output(
    maximum: usize,
    mut fits: impl FnMut(usize) -> Result<bool>,
) -> Result<Option<usize>> {
    if !fits(1)? {
        return Ok(None);
    }
    let (mut low, mut high) = (1, maximum);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(mid)? {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(Some(low))
}

static SHRUNK: AtomicU64 = AtomicU64::new(0);
static WITHHELD: AtomicU64 = AtomicU64::new(0);

/// An idle request whose prompt plus `requested` output tokens does not fit the KV pool gets
/// the longest output allowance that `fits` the `room`, and `take` reserves it there. The
/// caller first frees what only a cached prompt costs (`PrefixCache::release_copies`), so a
/// cached prompt gets what the same request gets cold. Every shrink is counted for `stats`.
/// `None`, with nothing taken, when not even one output token fits.
pub(super) fn shrink<R>(
    room: &mut R,
    requested: usize,
    mut fits: impl FnMut(&R, usize) -> Result<bool>,
    take: impl FnOnce(&mut R, usize) -> Result<()>,
) -> Result<Option<usize>> {
    let Some(granted) = fit_output(requested, |output| fits(room, output))? else {
        return Ok(None);
    };
    take(room, granted)?;
    if granted < requested {
        SHRUNK.fetch_add(1, Relaxed);
        WITHHELD.fetch_add((requested - granted) as u64, Relaxed);
    }
    Ok(Some(granted))
}

/// Lifetime admission counters for the per-second `stats` payload: requests whose output
/// allowance was shrunk to fit the KV pool, and the output tokens withheld from them.
pub(super) fn stats() -> serde_json::Value {
    serde_json::json!({
        "output_shrinks": SHRUNK.load(Relaxed),
        "output_tokens_withheld": WITHHELD.load(Relaxed),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_engine::media::{EncodeJob, EncodeOutput, EncoderClient, EncoderTicket, MediaError};

    #[derive(Default)]
    struct MockEncoder {
        full: bool,
        polls: usize,
        submitted: usize,
        cancelled: Vec<EncoderTicket>,
        result: Option<std::result::Result<EncodeOutput, MediaError>>,
    }
    impl EncoderClient for MockEncoder {
        fn submit(&mut self, _: EncodeJob) -> std::result::Result<EncoderTicket, MediaError> {
            if self.full { return Err(MediaError::QueueFull); }
            self.submitted += 1;
            Ok(EncoderTicket(self.submitted as u64))
        }
        fn poll(&mut self, _: EncoderTicket) -> Option<std::result::Result<EncodeOutput, MediaError>> {
            self.polls += 1;
            self.result.take()
        }
        fn cancel(&mut self, ticket: EncoderTicket) { self.cancelled.push(ticket); }
    }
    fn test_key() -> cuteafd_core::MediaKey {
        cuteafd_core::MediaKey::Image(cuteafd_core::ImageKey([7; 32]))
    }
    fn test_job() -> EncodeJob {
        EncodeJob::image(cuteafd_core::ImageKey([7; 32]), [1, 3, 3], vec![0; 9*588*2].into(), 4, 5120)
    }
    #[test]
    fn image_ticket_retries_full_queue_and_installs_only_a_complete_reply() -> Result<()> {
        let mut client = MockEncoder { full:true, ..Default::default() };
        let mut ticket = None;
        let mut installed = 0;
        assert!(!poll_image_ticket(&mut ticket, &mut client, test_key(), 8, test_job, |_| { installed+=1; Ok(()) })?);
        assert!(ticket.is_none());
        client.full = false;
        assert!(!poll_image_ticket(&mut ticket, &mut client, test_key(), 8, test_job, |_| { installed+=1; Ok(()) })?);
        assert!(ticket.is_some());
        assert!(!poll_image_ticket(&mut ticket, &mut client, test_key(), 8, test_job, |_| { installed+=1; Ok(()) })?);
        assert_eq!((client.submitted, installed), (1, 0));
        client.result = Some(Ok(EncodeOutput { key:test_key(), features:vec![0;8].into(), elapsed_ms:1.0 }));
        assert!(poll_image_ticket(&mut ticket, &mut client, test_key(), 8, test_job, |_| { installed+=1; Ok(()) })?);
        assert!(ticket.is_none());
        assert_eq!(installed, 1);
        Ok(())
    }
    #[test]
    fn image_ticket_fault_and_malformed_reply_are_typed_unavailable() {
        for result in [Err(MediaError::Encoder("lost".into())),
            Ok(EncodeOutput { key:test_key(), features:vec![0;7].into(), elapsed_ms:1.0 })] {
            let mut client = MockEncoder { result:Some(result), ..Default::default() };
            let mut ticket = Some((EncoderTicket(1), Instant::now()));
            let error = poll_image_ticket(&mut ticket, &mut client, test_key(), 8, test_job, |_| panic!("invalid features installed")).unwrap_err();
            assert!(matches!(error.downcast_ref::<cuteafd_api::openai::NativeFailure>(), Some(cuteafd_api::openai::NativeFailure::Unavailable(_))));
            assert!(ticket.is_none());
        }
    }
    #[test]
    fn image_ticket_timeout_retains_ticket_for_owner_cancellation() {
        let mut client = MockEncoder::default();
        let mut ticket = Some((EncoderTicket(9), Instant::now()-Duration::from_secs(61)));
        assert!(poll_image_ticket(&mut ticket, &mut client, test_key(), 8, test_job, |_| panic!("timed out features installed")).is_err());
        client.cancel(ticket.take().unwrap().0);
        assert_eq!(client.cancelled, vec![EncoderTicket(9)]);
        assert_eq!(client.polls, 0);
    }
    #[test]
    fn media_wake_waits_for_committed_round_and_text_fast_loop_is_unchanged() {
        let media = Wake { media_pending:true, ..Default::default() };
        assert!(!media.poll_media(0));
        assert!(media.poll_media(1));
        let full = Wake { media_pending:true, media_slots:2, ..Default::default() };
        assert!(!full.ready(14, 16, true));
        assert!(full.ready(13, 16, true));
        let backlog = Wake { host_pending:true, ..Default::default() };
        assert!(backlog.ready(1, 16, false));
        assert!(!backlog.ready(16, 16, false));
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
    #[test]
    fn output_reservation_is_the_longest_that_fits() -> Result<()> {
        for maximum in [1, 2, 1000] {
            for available in [0, 1, 2, 17, 1000] {
                assert_eq!(
                    fit_output(maximum, |output| Ok(output <= available))?,
                    (available > 0).then_some(maximum.min(available))
                );
            }
        }
        Ok(())
    }
    /// The idle shrink over a pool modelled in output tokens: `free` is the room the request has
    /// cold; a retained source sharing its tail (`copy`, one page of rows) costs that much more
    /// until `release_copies` drops it, as `PrefixCache::release_copies` does.
    struct Room {
        free: usize,
        copy: usize,
        taken: Option<usize>,
    }
    impl Room {
        fn release_copies(&mut self) {
            self.copy = 0;
        }
        fn fits(&self, output: usize) -> Result<bool> {
            Ok(output + self.copy <= self.free)
        }
        fn take(&mut self, output: usize) -> Result<()> {
            anyhow::ensure!(self.fits(output)?, "{output} does not fit");
            self.taken = Some(output);
            Ok(())
        }
    }
    #[test]
    fn shrink_grants_a_cached_prompt_what_the_same_request_gets_cold_and_counts_it() -> Result<()> {
        let before = (SHRUNK.load(Relaxed), WITHHELD.load(Relaxed));
        let mut cold = Room {
            free: 700,
            copy: 0,
            taken: None,
        };
        let mut cached = Room {
            free: 700,
            copy: 512,
            taken: None,
        };
        // Kept, the reused source would cost the cached request 512 tokens of output.
        assert_eq!(fit_output(1000, |output| cached.fits(output))?, Some(188));
        cached.release_copies();
        for room in [&mut cold, &mut cached] {
            let granted = shrink(room, 1000, Room::fits, Room::take)?;
            assert_eq!((granted, room.taken), (Some(700), Some(700)));
        }
        // A request that fits whole is granted whole and is not counted as a shrink.
        let mut whole = Room {
            free: 700,
            copy: 0,
            taken: None,
        };
        assert_eq!(shrink(&mut whole, 600, Room::fits, Room::take)?, Some(600));
        // Not even one token: refused, nothing taken.
        let mut full = Room {
            free: 0,
            copy: 0,
            taken: None,
        };
        assert_eq!(shrink(&mut full, 1000, Room::fits, Room::take)?, None);
        assert_eq!(full.taken, None);
        // A failed take is the request's error and is not counted.
        let mut failing = Room {
            free: 50,
            copy: 0,
            taken: None,
        };
        assert!(shrink(&mut failing, 1000, Room::fits, |_, _| anyhow::bail!(
            "take failed"
        ))
        .is_err());
        // Two shrinks of 300 withheld tokens each. No other test shrinks.
        assert_eq!(
            (
                SHRUNK.load(Relaxed) - before.0,
                WITHHELD.load(Relaxed) - before.1
            ),
            (2, 600)
        );
        assert_eq!(stats()["output_shrinks"], SHRUNK.load(Relaxed));
        Ok(())
    }
    #[test]
    fn blocked_admission_does_not_join_lanes_until_a_request_retires() {
        let blocked = Wake {
            blocked_at: Some(2),
            ..Wake::default()
        };
        assert!(!blocked.ready(2, 16, true));
        assert!(!blocked.ready(2, 16, false));
        assert!(blocked.ready(1, 16, true));
        assert!(blocked.ready(1, 16, false)); // The pending request is outside the channel.
        assert!(blocked.ready(0, 16, false));
        assert!(!Wake::default().ready(16, 16, true));
        assert!(!Wake::default().ready(2, 16, false));
        assert!(Wake::default().ready(2, 16, true));
    }
    #[test]
    fn cancelled_pending_request_wakes_admission_without_waiting_for_retirement() {
        let (events, output) = mpsc::unbounded_channel();
        let job = NativeRequest {
            prompt: String::new(),
            constraint: None,
            images: Vec::new(),
            media: Vec::new(),
            audio: Vec::new(),
            max_tokens: 1,
            sampling: Default::default(),
            stop_token_ids: Vec::new(),
            events,
            probe: None,
        };
        let wake = Wake {
            blocked_at: Some(2),
            pending: Some(&job),
            ..Wake::default()
        };
        assert!(!wake.ready(2, 16, true));
        drop(output);
        assert!(wake.ready(2, 16, true));
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
