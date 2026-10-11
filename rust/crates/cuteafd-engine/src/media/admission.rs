use super::{
    round_frontier, stats::Latencies, EmbeddingCache, EmbeddingLease, EncodeJob, EncoderClient,
    EncoderTicket, MediaKey, MediaError, MediaStats, RequestMedia,
};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant};

/// A waiter contains host descriptors only, never a placement, pages, or a state-slot lease.
#[derive(Debug)]
pub struct MediaWaiter<T> {
    pub job: T,
    pub media: RequestMedia,
    prepared: HashMap<MediaKey, EncodeJob>,
    resume: usize,
    retries: u8,
    cold: bool,
    submit_waited: bool,
    pins: Vec<EmbeddingLease>,
    error: Option<MediaError>,
}
impl<T> MediaWaiter<T> {
    pub fn new(
        job: T,
        media: RequestMedia,
        prepared: Vec<EncodeJob>,
        peek_resume: usize,
    ) -> Result<Self, MediaError> {
        if peek_resume > media.prompt_len() {
            return Err(MediaError::Spans);
        }
        let mut inputs = HashMap::new();
        for image in prepared {
            image.validate()?;
            if inputs.get(&image.key).is_some_and(|old| old != &image) {
                return Err(MediaError::Spans);
            }
            inputs.entry(image.key).or_insert(image);
        }
        for span in media.spans() {
            let image = inputs
                .get(&span.key)
                .ok_or(MediaError::NotReady(span.key))?;
            if image.tokens != span.len || image.feature_bytes()? != span.len * media.row_bytes() {
                return Err(MediaError::Features);
            }
        }
        let resume = round_frontier(peek_resume, media.spans());
        Ok(Self {
            job,
            media,
            prepared: inputs,
            resume,
            retries: 0,
            cold: false,
            submit_waited: false,
            pins: Vec::new(),
            error: None,
        })
    }
}
/// The scheduler may attempt normal prefix admission only after this is returned.
#[derive(Debug)]
pub struct MediaReady<T> {
    waiter: MediaWaiter<T>,
}
impl<T> MediaReady<T> {
    pub fn job(&self) -> &T {
        &self.waiter.job
    }
    pub fn media(&self) -> &RequestMedia {
        &self.waiter.media
    }
    pub fn resume(&self) -> usize {
        self.waiter.resume
    }
    pub fn cold(&self) -> bool {
        self.waiter.cold
    }
    pub fn retries(&self) -> u8 {
        self.waiter.retries
    }
    pub fn into_parts(self) -> (T, RequestMedia) {
        (self.waiter.job, self.waiter.media)
    }

    /// Admission rechecks the prefix after encoding. If it lost rows, release the attempted
    /// placement/KV/state slot BEFORE re-enqueueing the returned waiter. At most two such
    /// retries; the third materializes the whole prompt and requires cache-bypassing admission.
    pub fn reconcile(mut self, actual_resume: usize) -> Result<Self, MediaWaiter<T>> {
        let actual = round_frontier(
            actual_resume.min(self.waiter.media.prompt_len()),
            self.waiter.media.spans(),
        );
        if self
            .waiter
            .media
            .ready(actual, self.waiter.media.prompt_len())
        {
            self.waiter.resume = actual;
            return Ok(self);
        }
        self.waiter.retries = self.waiter.retries.saturating_add(1);
        self.waiter.cold |= self.waiter.retries > 2;
        self.waiter.resume = if self.waiter.cold { 0 } else { actual };
        Err(self.waiter)
    }
}

#[derive(Debug)]
pub enum MediaPoll<T> {
    Empty,
    Pending,
    Ready(MediaReady<T>),
    Failed(T, MediaError),
}
struct Flight {
    ticket: Option<EncoderTicket>,
    queued_at: Instant,
    submitted_at: Option<Instant>,
    _pin: EmbeddingLease,
}

/// A bounded media_pending queue and one in-flight encode per modality-tagged MediaKey, shared across requests.
/// poll is nonblocking and ready requests may pass a slower encoder job (decode stays runnable).
/// Encoder queue saturation keeps reservations pinned and retries submission in pending order.
pub struct MediaAdmission<T, C: EncoderClient> {
    pub cache: EmbeddingCache,
    encoder: C,
    pending: VecDeque<MediaWaiter<T>>,
    flights: HashMap<MediaKey, Flight>,
    capacity: usize,
    max_images: usize,
    max_tokens: usize,
    encodes: u64,
    submit_timeout: Duration,
    encode_timeout: Duration,
    encode_timeouts: u64,
    submit_waits: u64,
    submit_timeouts: u64,
    skipped: u64,
    latencies: Latencies,
}
impl<T, C: EncoderClient> MediaAdmission<T, C> {
    pub fn new(cache: EmbeddingCache, encoder: C, capacity: usize) -> Self {
        Self {
            cache,
            encoder,
            pending: VecDeque::new(),
            flights: HashMap::new(),
            capacity,
            max_images: 16,
            max_tokens: 32768,
            encodes: 0,
            submit_timeout: Duration::from_secs(60),
            encode_timeout: Duration::from_secs(60),
            encode_timeouts: 0,
            submit_waits: 0,
            submit_timeouts: 0,
            skipped: 0,
            latencies: Latencies::default(),
        }
    }
    pub fn set_encode_timeout(&mut self, timeout: Duration) {
        self.encode_timeout = timeout;
    }
    pub fn set_submit_timeout(&mut self, timeout: Duration) {
        self.submit_timeout = timeout;
    }
    pub fn set_encode_limits(&mut self, images: usize, tokens: usize) {
        self.max_images = images;
        self.max_tokens = tokens;
    }
    pub fn encoder(&self) -> &C {
        &self.encoder
    }
    pub fn encoder_mut(&mut self) -> &mut C {
        &mut self.encoder
    }
    pub fn len(&self) -> usize {
        self.pending.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
    pub fn enqueue(
        &mut self,
        mut waiter: MediaWaiter<T>,
    ) -> Result<(), (MediaWaiter<T>, MediaError)> {
        if self.pending.len() >= self.capacity {
            return Err((waiter, MediaError::QueueFull));
        }
        match self.prepare(&mut waiter) {
            Ok(()) => {
                if waiter.retries == 0 {
                    self.skipped += waiter
                        .media
                        .spans()
                        .iter()
                        .filter(|s| s.checked_end().unwrap() <= waiter.resume)
                        .count() as u64;
                }
                self.pending.push_back(waiter);
                Ok(())
            }
            Err(error) => {
                waiter.pins.clear();
                self.cancel_unused();
                self.cache.prune_reservations();
                Err((waiter, error))
            }
        }
    }
    fn prepare(&mut self, waiter: &mut MediaWaiter<T>) -> Result<(), MediaError> {
        let keys: BTreeSet<_> = waiter.media.spans().iter().map(|s| s.key).collect();
        for key in keys {
            // Touch even images fully restored by the prefix. Pin while this request exists.
            if !waiter.media.has_features(key) {
                if let Some(lease) = self.cache.get(key) {
                    waiter.media.attach(lease)?;
                }
            }
        }
        let missing: BTreeSet<_> = waiter
            .media
            .needed(waiter.resume, waiter.media.prompt_len())
            .filter(|s| !waiter.media.has_features(s.key))
            .map(|s| s.key)
            .collect();
        let new: Vec<_> = missing
            .iter()
            .filter(|key| !self.flights.contains_key(key))
            .copied()
            .collect();
        let tokens = new
            .iter()
            .try_fold(0usize, |n, key| n.checked_add(waiter.prepared[key].tokens))
            .ok_or(MediaError::Features)?;
        if new.len() > self.max_images || tokens > self.max_tokens {
            return Err(MediaError::EncodeLimit {
                images: new.len(),
                tokens,
                max_images: self.max_images,
                max_tokens: self.max_tokens,
            });
        }
        // Admit every output reservation before submitting any allocation-producing work.
        for key in &missing {
            let bytes = waiter.prepared[key].feature_bytes()?;
            waiter.pins.push(self.cache.reserve(*key, bytes)?);
        }
        let queued = self.flights.values().any(|flight| flight.ticket.is_none());
        for &key in &new {
            let pin = self
                .cache
                .reserve(key, waiter.prepared[&key].feature_bytes()?)?;
            self.flights.insert(key, Flight { ticket: None, queued_at: Instant::now(), submitted_at: None, _pin: pin });
        }
        if queued {
            self.mark_submit_wait(waiter);
            return Ok(());
        }
        for key in new {
            match self.encoder.submit(waiter.prepared[&key].clone()) {
                Ok(ticket) => {
                    let flight = self.flights.get_mut(&key).unwrap();
                    flight.ticket = Some(ticket);
                    flight.submitted_at = Some(Instant::now());
                    self.encodes += 1;
                }
                Err(MediaError::QueueFull) => {
                    self.mark_submit_wait(waiter);
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    fn mark_submit_wait(&mut self, waiter: &mut MediaWaiter<T>) {
        let blocked = waiter.media.needed(waiter.resume, waiter.media.prompt_len())
            .any(|span| self.flights.get(&span.key).is_some_and(|flight| flight.ticket.is_none()));
        if blocked && !waiter.submit_waited {
            waiter.submit_waited = true;
            self.submit_waits += 1;
        }
    }
    fn expire_submissions(&mut self) {
        for waiter in &mut self.pending {
            if waiter.error.is_some() { continue; }
            let unsubmitted: Vec<_> = waiter.media.needed(waiter.resume, waiter.media.prompt_len())
                .filter_map(|span| self.flights.get(&span.key))
                .filter(|flight| flight.ticket.is_none()).collect();
            if unsubmitted.is_empty() { continue; }
            if !waiter.submit_waited {
                waiter.submit_waited = true;
                self.submit_waits += 1;
            }
            if unsubmitted.iter().any(|flight| flight.queued_at.elapsed() >= self.submit_timeout) {
                waiter.error = Some(MediaError::QueueFull);
                self.submit_timeouts += 1;
            }
        }
    }
    fn submit_pending(&mut self) {
        for waiter in &mut self.pending {
            if waiter.error.is_some() { continue; }
            let keys: BTreeSet<_> = waiter.media
                .needed(waiter.resume, waiter.media.prompt_len()).map(|s| s.key).collect();
            for key in keys {
                let Some(flight) = self.flights.get_mut(&key).filter(|f| f.ticket.is_none()) else { continue };
                match self.encoder.submit(waiter.prepared[&key].clone()) {
                    Ok(ticket) => {
                        flight.ticket = Some(ticket);
                        flight.submitted_at = Some(Instant::now());
                        self.encodes += 1;
                    }
                    Err(MediaError::QueueFull) => return,
                    Err(error) => { waiter.error = Some(error); break; }
                }
            }
        }
    }
    /// Call once per serve-loop step. Cancelled requests do not hold up later ones.
    pub fn poll(&mut self, mut cancelled: impl FnMut(&T) -> bool) -> MediaPoll<T> {
        self.pending.retain(|w| !cancelled(&w.job));
        self.cancel_unused();
        let keys: Vec<_> = self.flights.keys().copied().collect();
        for key in keys {
            let Some(ticket) = self.flights[&key].ticket else { continue };
            let result = match self.encoder.poll(ticket) {
                Some(result) => result,
                None if self.flights[&key].submitted_at
                    .is_some_and(|submitted| submitted.elapsed() >= self.encode_timeout) => {
                    self.encoder.cancel(ticket);
                    self.encode_timeouts += 1;
                    Err(MediaError::Encoder("encode timed out".into()))
                }
                None => continue,
            };
            self.flights.remove(&key);
            let result = result.and_then(|output| {
                if output.key != key {
                    return Err(MediaError::Features);
                }
                let lease = self.cache.complete(key, output.features)?;
                self.latencies.record(output.elapsed_ms);
                Ok(lease)
            });
            for waiter in &mut self.pending {
                if waiter.error.is_some() { continue; }
                if !waiter
                    .media
                    .needed(waiter.resume, waiter.media.prompt_len())
                    .any(|s| s.key == key)
                {
                    continue;
                }
                match &result {
                    Ok(lease) => {
                        if let Err(error) = waiter.media.attach(lease.clone()) {
                            waiter.error = Some(error);
                        }
                    }
                    Err(error) => waiter.error = Some(error.clone()),
                }
            }
        }
        // Completions free encoder queue slots before retrying unsubmitted keys in FIFO order.
        self.submit_pending();
        self.expire_submissions();
        let ready = self
            .pending
            .iter()
            .position(|w| w.error.is_some() || w.media.ready(w.resume, w.media.prompt_len()));
        let Some(index) = ready else {
            self.cache.prune_reservations();
            return if self.pending.is_empty() {
                MediaPoll::Empty
            } else {
                MediaPoll::Pending
            };
        };
        let mut waiter = self.pending.remove(index).unwrap();
        waiter.pins.clear(); // ready features now hold their own leases
        let result = if let Some(error) = waiter.error.take() {
            MediaPoll::Failed(waiter.job, error)
        } else {
            MediaPoll::Ready(MediaReady { waiter })
        };
        self.cancel_unused();
        self.cache.prune_reservations();
        result
    }
    fn cancel_unused(&mut self) {
        let unused: Vec<_> = self
            .flights
            .keys()
            .filter(|&&key| {
                !self.pending.iter().any(|w| {
                    w.error.is_none()
                        && !w.media.has_features(key)
                        && w.media
                            .needed(w.resume, w.media.prompt_len())
                            .any(|s| s.key == key)
                })
            })
            .copied()
            .collect();
        for key in unused {
            if let Some(flight) = self.flights.remove(&key) {
                if let Some(ticket) = flight.ticket { self.encoder.cancel(ticket); }
            }
        }
    }
    pub fn stats(&self, media_key_collisions: u64, memo_hits: u64) -> MediaStats {
        let (encode_ms_p50, encode_ms_p99) = self.latencies.percentiles();
        MediaStats {
            encodes: self.encodes,
            submit_waits: self.submit_waits,
            submit_timeouts: self.submit_timeouts,
            encode_timeouts: self.encode_timeouts,
            encode_ms_p50,
            encode_ms_p99,
            cache_hits: self.cache.hits(),
            cache_bytes: self.cache.bytes(),
            memo_hits,
            prefix_skipped_images: self.skipped,
            media_key_collisions,
            pending: self.pending.len(),
        }
    }
}
impl<T, C: EncoderClient> Drop for MediaAdmission<T, C> {
    fn drop(&mut self) {
        for flight in self.flights.values() {
            if let Some(ticket) = flight.ticket { self.encoder.cancel(ticket); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::{FakeEncoder, ImageKey, MediaSpan};
    use std::sync::Arc;
    fn waiter(id: usize, keys: &[u8], resume: usize) -> MediaWaiter<usize> {
        let spans = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| MediaSpan {
                start: i * 2,
                len: 2,
                key: ImageKey([k; 32]).into(),
            })
            .collect();
        let media = RequestMedia::new(spans, 2, keys.len() * 2).unwrap();
        let jobs = keys
            .iter()
            .map(|&k| EncodeJob::image(ImageKey([k; 32]), [1, 2, 2], Arc::from([k; 12]), 2, 2))
            .collect();
        MediaWaiter::new(id, media, jobs, resume).unwrap()
    }
    fn ready<T>(poll: MediaPoll<T>) -> MediaReady<T> {
        match poll {
            MediaPoll::Ready(ready) => ready,
            _ => panic!("not ready"),
        }
    }
    struct SaturatedEncoder {
        inner: FakeEncoder,
        full_submits: usize,
        fail_submit: bool,
        order: Vec<MediaKey>,
        cancellations: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl SaturatedEncoder {
        fn new(full_submits: usize) -> Self {
            Self { inner: FakeEncoder::default(), full_submits, fail_submit: false,
                order: Vec::new(), cancellations: Arc::default() }
        }
    }
    impl EncoderClient for SaturatedEncoder {
        fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket, MediaError> {
            if self.full_submits > 0 {
                self.full_submits -= 1;
                return Err(MediaError::QueueFull);
            }
            if self.fail_submit { return Err(MediaError::Encoder("submit failed".into())); }
            if self.inner.pending() >= 1 { return Err(MediaError::QueueFull); }
            self.order.push(job.key);
            self.inner.submit(job)
        }
        fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<super::super::EncodeOutput, MediaError>> {
            self.inner.poll(ticket)
        }
        fn cancel(&mut self, ticket: EncoderTicket) {
            self.cancellations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.cancel(ticket);
        }
    }

    #[test]
    fn queue_full_retries_four_images_in_pending_order_without_failing() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(40), SaturatedEncoder::new(3), 3);
        queue.enqueue(waiter(1, &[1, 2, 3, 4], 0)).unwrap();
        queue.enqueue(waiter(2, &[5], 0)).unwrap();
        queue.enqueue(waiter(3, &[2], 0)).unwrap();
        assert_eq!(queue.cache.bytes(), 40, "unsubmitted outputs stay reserved");
        assert_eq!(queue.encoder().inner.submitted, 0);
        let mut completed = Vec::new();
        for _ in 0..16 {
            match queue.poll(|_| false) {
                MediaPoll::Ready(ready) => {
                    assert!(ready.media().ready(0, ready.media().prompt_len()));
                    completed.push(*ready.job());
                }
                MediaPoll::Failed(_, error) => panic!("saturation must wait: {error}"),
                MediaPoll::Empty => break,
                MediaPoll::Pending => (),
            }
        }
        completed.sort_unstable();
        assert_eq!(completed, [1, 2, 3]);
        assert_eq!(queue.encoder().order, (1..=5).map(|key| ImageKey([key; 32]).into()).collect::<Vec<_>>());
        assert_eq!(queue.stats(0, 0).encodes, 5);
        assert_eq!(queue.stats(0, 0).submit_waits, 3);
        assert_eq!(queue.stats(0, 0).submit_timeouts, 0);
    }

    #[test]
    fn borrowed_encoder_client_delegates_submission_polling_and_cancellation() {
        let mut encoder = FakeEncoder::default();
        {
            let mut queue = MediaAdmission::new(EmbeddingCache::new(8), &mut encoder, 1);
            queue.enqueue(waiter(1, &[1], 0)).unwrap();
            drop(ready(queue.poll(|_| false)));
            queue.enqueue(waiter(2, &[2], 0)).unwrap();
        }
        assert_eq!((encoder.submitted, encoder.cancelled, encoder.pending()), (2, 1, 0));
    }

    #[test]
    fn aged_completed_encode_is_accepted_before_timeout() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(8), FakeEncoder::default(), 1);
        queue.enqueue(waiter(1, &[1], 0)).unwrap();
        for flight in queue.flights.values_mut() {
            flight.submitted_at = Some(Instant::now() - Duration::from_secs(61));
        }
        assert_eq!(*ready(queue.poll(|_| false)).job(), 1);
        assert_eq!(queue.encoder().cancelled, 0);
        assert_eq!(queue.stats(0, 0).encode_timeouts, 0);
    }

    #[test]
    fn encode_timeout_cancels_shared_ticket_and_preserves_first_error() {
        let mut encoder = FakeEncoder::default();
        encoder.delay_polls = usize::MAX;
        let mut queue = MediaAdmission::new(EmbeddingCache::new(8), encoder, 3);
        queue.set_encode_timeout(Duration::from_secs(60));
        for id in 1..=3 { queue.enqueue(waiter(id, &[1], 0)).unwrap(); }
        queue.pending[0].error = Some(MediaError::Encoder("first error".into()));
        for flight in queue.flights.values_mut() {
            flight.submitted_at = Some(Instant::now() - Duration::from_secs(61));
        }
        for (id, message) in [(1, "first error"), (2, "encode timed out"), (3, "encode timed out")] {
            match queue.poll(|_| false) {
                MediaPoll::Failed(job, MediaError::Encoder(error)) => assert_eq!((job, error.as_str()), (id, message)),
                other => panic!("expected encoder timeout, got {other:?}"),
            }
        }
        assert_eq!(queue.encoder().cancelled, 1);
        assert_eq!(queue.stats(0, 0).encode_timeouts, 1);
        assert_eq!(queue.cache.bytes(), 0);
        assert!(matches!(queue.poll(|_| false), MediaPoll::Empty));
        assert_eq!(queue.encoder().cancelled, 1);
    }

    #[test]
    fn delayed_poll_retries_submission_before_expiring_the_waiter() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(8), SaturatedEncoder::new(1), 1);
        queue.enqueue(waiter(1, &[1], 0)).unwrap();
        for flight in queue.flights.values_mut() {
            flight.queued_at = Instant::now() - Duration::from_secs(61);
        }
        assert!(matches!(queue.poll(|_| false), MediaPoll::Pending));
        assert_eq!(queue.encoder().inner.submitted, 1);
        assert_eq!(queue.stats(0, 0).submit_timeouts, 0);
        assert_eq!(*ready(queue.poll(|_| false)).job(), 1);
        assert_eq!(queue.stats(0, 0).submit_waits, 1);
    }

    #[test]
    fn unsubmitted_waiter_times_out_once_and_cancellation_wins() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(32), SaturatedEncoder::new(usize::MAX), 2);
        queue.set_submit_timeout(Duration::from_secs(60));
        queue.enqueue(waiter(1, &[1, 2, 3, 4], 0)).unwrap();
        assert!(matches!(queue.poll(|_| false), MediaPoll::Pending));
        for flight in queue.flights.values_mut() {
            flight.queued_at = Instant::now() - Duration::from_secs(61);
        }
        assert!(matches!(queue.poll(|_| false), MediaPoll::Failed(1, MediaError::QueueFull)));
        assert_eq!(queue.cache.bytes(), 0);
        assert_eq!(queue.stats(0, 0).submit_waits, 1);
        assert_eq!(queue.stats(0, 0).submit_timeouts, 1);
        assert!(matches!(queue.poll(|_| false), MediaPoll::Empty));
        assert_eq!(queue.stats(0, 0).submit_timeouts, 1);
        queue.enqueue(waiter(2, &[1], 0)).unwrap();
        queue.set_submit_timeout(Duration::ZERO);
        assert!(matches!(queue.poll(|_| true), MediaPoll::Empty));
        assert_eq!(queue.stats(0, 0).submit_timeouts, 1);
        assert_eq!(queue.cache.bytes(), 0);
    }

    #[test]
    fn unsubmitted_keys_cancel_and_drop_without_encoder_cancellation() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(32), SaturatedEncoder::new(1), 1);
        queue.enqueue(waiter(1, &[1, 2, 3, 4], 0)).unwrap();
        assert!(matches!(queue.poll(|_| true), MediaPoll::Empty));
        assert_eq!(queue.cache.bytes(), 0);
        assert_eq!(queue.encoder().inner.submitted, 0);
        let cancellations = Arc::clone(&queue.encoder().cancellations);
        queue.encoder_mut().full_submits = 1;
        queue.enqueue(waiter(2, &[1, 2, 3, 4], 0)).unwrap();
        drop(queue);
        assert_eq!(cancellations.load(std::sync::atomic::Ordering::Relaxed), 0);
        let mut queue = MediaAdmission::new(EmbeddingCache::new(32), SaturatedEncoder::new(0), 1);
        queue.enqueue(waiter(3, &[1, 2, 3, 4], 0)).unwrap();
        let cancellations = Arc::clone(&queue.encoder().cancellations);
        drop(queue);
        assert_eq!(cancellations.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn retry_submit_errors_other_than_queue_full_fail_and_release_reservations() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(32), SaturatedEncoder::new(1), 1);
        queue.enqueue(waiter(1, &[1, 2, 3, 4], 0)).unwrap();
        queue.encoder_mut().fail_submit = true;
        assert!(matches!(queue.poll(|_| false), MediaPoll::Failed(1, MediaError::Encoder(_))));
        assert_eq!(queue.cache.bytes(), 0);
        assert_eq!(queue.encoder().inner.submitted, 0);
        assert!(queue.flights.is_empty());
        assert!(matches!(queue.enqueue(waiter(2, &[1], 0)), Err((_, MediaError::Encoder(_)))));
        assert_eq!(queue.cache.bytes(), 0);
    }

    #[test]
    fn audio_cache_dedupes_content_without_aliasing_images_and_reencodes_exactly() {
        use crate::media::{AudioKey, MediaKeys, verify_media};
        let audio_key = AudioKey([1; 32]);
        let span = MediaSpan { start: 0, len: 2, key: audio_key.into() };
        let pcm: Arc<[f32]> = vec![0.0; 4800].into();
        let make = |id, resume| MediaWaiter::new(id, RequestMedia::new(vec![span], 2, 2).unwrap(),
            vec![EncodeJob::audio(audio_key, pcm.clone(), 2, 2)], resume).unwrap();
        let mut queue = MediaAdmission::new(EmbeddingCache::new(8), FakeEncoder::default(), 3);
        queue.enqueue(make(1, 0)).unwrap();
        queue.enqueue(make(2, 0)).unwrap();
        let first = ready(queue.poll(|_| false));
        let second = ready(queue.poll(|_| false));
        assert_eq!(queue.encoder().submitted, 1);
        assert!(queue.cache.contains(audio_key));
        assert!(!queue.cache.contains(ImageKey(audio_key.0)));
        let mut a = crate::media::MediaChunk::default(); first.media().write_chunk(0, 2, &mut a).unwrap();
        let native = [3, 3];
        let image = MediaSpan { key: ImageKey(audio_key.0).into(), ..span };
        assert_ne!(MediaKeys::new(&native, 10, &[span]).unwrap().tokens(), MediaKeys::new(&native, 10, &[image]).unwrap().tokens());
        assert!(!verify_media(2, &[span], &[image]));
        drop(first); drop(second);
        let pin = queue.cache.reserve(ImageKey([2; 32]), 8).unwrap();
        assert!(!queue.cache.contains(audio_key)); drop(pin); queue.cache.prune_reservations();
        queue.enqueue(make(3, 2)).unwrap();
        let restored = ready(queue.poll(|_| false));
        assert_eq!(queue.encoder().submitted, 1, "prefix restore needs no embeddings");
        let retry = restored.reconcile(0).unwrap_err(); queue.enqueue(retry).unwrap();
        let reencoded = ready(queue.poll(|_| false));
        let mut b = crate::media::MediaChunk::default(); reencoded.media().write_chunk(0, 2, &mut b).unwrap();
        assert_eq!(a.features, b.features);
        assert_eq!(queue.encoder().submitted, 2);
    }

    #[test]
    fn dedupe_cancel_and_no_head_of_line_blocking() {
        let mut encoder = FakeEncoder::default();
        encoder.delay_polls = 1;
        let mut queue = MediaAdmission::new(EmbeddingCache::new(64), encoder, 4);
        queue.enqueue(waiter(1, &[1], 0)).unwrap();
        queue.enqueue(waiter(2, &[1], 0)).unwrap();
        queue.enqueue(waiter(3, &[], 0)).unwrap();
        assert_eq!(queue.encoder().submitted, 1);
        assert_eq!(*ready(queue.poll(|id| *id == 1)).job(), 3);
        assert_eq!(*ready(queue.poll(|_| false)).job(), 2);
        assert_eq!(queue.encoder().cancelled, 0);
        assert_eq!(queue.stats(0, 2).memo_hits, 2);
        queue.enqueue(waiter(4, &[2], 0)).unwrap();
        assert!(matches!(queue.poll(|_| true), MediaPoll::Empty));
        assert_eq!(queue.encoder().cancelled, 1);
        assert_eq!(queue.cache.bytes(), 8);
    }
    #[test]
    fn repeated_images_share_one_encode_and_hold_their_pin_until_request_release() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(8), FakeEncoder::default(), 2);
        queue.enqueue(waiter(1, &[1, 1], 0)).unwrap();
        let request = ready(queue.poll(|_| false));
        assert_eq!(queue.encoder().submitted, 1);
        let mut chunk = crate::media::MediaChunk::default();
        request.media().write_chunk(0, 4, &mut chunk).unwrap();
        assert_eq!(chunk.indices, [0, 1, 2, 3]);
        assert_eq!(chunk.features[..8], chunk.features[8..]);
        assert!(queue.cache.reserve(ImageKey([2; 32]), 8).is_err());
        drop(request);
        assert!(queue.cache.reserve(ImageKey([2; 32]), 8).is_ok());
    }

    #[test]
    fn prefix_skip_touches_cache_and_only_new_images_encode() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(64), FakeEncoder::default(), 4);
        queue.enqueue(waiter(1, &[1], 0)).unwrap();
        drop(ready(queue.poll(|_| false)));
        queue.enqueue(waiter(2, &[1, 2], 2)).unwrap();
        let r = ready(queue.poll(|_| false));
        assert!(r.media().has_features(ImageKey([1; 32])));
        assert_eq!(queue.encoder().submitted, 2);
        assert_eq!(queue.stats(0, 0).prefix_skipped_images, 1);
        assert_eq!(queue.stats(0, 0).cache_hits, 1);
        assert_eq!(queue.stats(0, 0).encode_ms_p50, 1.0);
    }
    #[test]
    fn peek_eviction_races_retry_twice_then_force_cold() {
        let mut queue = MediaAdmission::new(EmbeddingCache::new(64), FakeEncoder::default(), 4);
        queue.enqueue(waiter(1, &[1, 2, 3, 4], 8)).unwrap();
        for actual in [6, 4, 2] {
            let r = ready(queue.poll(|_| false));
            let retry = r.reconcile(actual).unwrap_err();
            queue.enqueue(retry).unwrap();
        }
        let r = ready(queue.poll(|_| false));
        assert!(r.cold());
        assert_eq!(r.resume(), 0);
        assert_eq!(r.retries(), 3);
        assert!(r.media().ready(0, 8));
        assert_eq!(queue.encoder().submitted, 4);
        assert!(r.reconcile(0).is_ok());
    }
    #[test]
    fn encoder_failure_releases_pins_and_limits_admit_before_submit() {
        let mut encoder = FakeEncoder::default();
        encoder.fail_next = true;
        let mut queue = MediaAdmission::new(EmbeddingCache::new(8), encoder, 4);
        queue.enqueue(waiter(1, &[1], 0)).unwrap();
        assert!(matches!(
            queue.poll(|_| false),
            MediaPoll::Failed(1, MediaError::Encoder(_))
        ));
        assert_eq!(queue.cache.bytes(), 0);
        assert!(queue.enqueue(waiter(2, &[2, 3], 0)).is_err());
        assert_eq!(queue.encoder().submitted, 1);
        assert_eq!(queue.cache.bytes(), 0);
        queue.set_encode_limits(0, 0);
        assert!(matches!(
            queue.enqueue(waiter(3, &[3], 0)),
            Err((_, MediaError::EncodeLimit { .. }))
        ));
    }
}
