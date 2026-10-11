use super::{AudioKey, ImageKey, MediaError, MediaKey};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Debug, PartialEq)]
pub enum EncodeInput {
    Image { grid: [u32; 3], rgb8: Arc<[u8]> },
    /// One complete clip of canonical finite mono FP32 PCM at 24 kHz.
    Audio { pcm: Arc<[f32]> },
}
#[derive(Clone, Debug, PartialEq)]
pub struct EncodeJob {
    pub key: MediaKey,
    pub input: EncodeInput,
    pub tokens: usize,
    pub hidden_width: usize,
}
impl EncodeJob {
    pub fn image_input(&self) -> Result<([u32; 3], &Arc<[u8]>), MediaError> {
        match (&self.key, &self.input) {
            (MediaKey::Image(_), EncodeInput::Image { grid, rgb8 }) => Ok((*grid, rgb8)),
            _ => Err(MediaError::Features),
        }
    }
    pub fn image(key: ImageKey, grid: [u32; 3], rgb8: Arc<[u8]>, tokens: usize, hidden_width: usize) -> Self {
        Self { key: key.into(), input: EncodeInput::Image { grid, rgb8 }, tokens, hidden_width }
    }
    pub fn audio(key: AudioKey, pcm: Arc<[f32]>, tokens: usize, hidden_width: usize) -> Self {
        Self { key: key.into(), input: EncodeInput::Audio { pcm }, tokens, hidden_width }
    }
    pub fn feature_bytes(&self) -> Result<usize, MediaError> {
        self.tokens.checked_mul(self.hidden_width).and_then(|n| n.checked_mul(2))
            .filter(|&n| n > 0).ok_or(MediaError::Features)
    }
    pub fn validate(&self) -> Result<(), MediaError> {
        self.feature_bytes()?;
        match (&self.key, &self.input) {
            (MediaKey::Image(_), EncodeInput::Image { grid, rgb8 }) if !grid.contains(&0) && !rgb8.is_empty() => Ok(()),
            (MediaKey::Audio(_), EncodeInput::Audio { pcm }) => {
                if !(481..=7_200_000).contains(&pcm.len()) || pcm.iter().any(|x| !x.is_finite()) {
                    return Err(MediaError::Features);
                }
                let frames = pcm.len() / 240 + 1;
                let codes = (frames / 6000) * 1500 + (frames % 6000).div_ceil(2).div_ceil(2);
                if self.tokens != codes.div_ceil(4) { return Err(MediaError::Features); }
                Ok(())
            }
            _ => Err(MediaError::Features),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EncoderTicket(pub u64);
#[derive(Clone, Debug)]
pub struct EncodeOutput {
    pub key: MediaKey,
    pub features: Arc<[u8]>,
    pub elapsed_ms: f64,
}

/// Nonblocking scheduler interface. An owner thread (local CUDA or remote TCP) owns all
/// execution and scratch. Cancellation must retain job storage until queued work drains.
pub trait EncoderClient {
    fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket, MediaError>;
    fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<EncodeOutput, MediaError>>;
    fn cancel(&mut self, ticket: EncoderTicket);
}

impl<C: EncoderClient + ?Sized> EncoderClient for &mut C {
    fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket, MediaError> {
        (**self).submit(job)
    }
    fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<EncodeOutput, MediaError>> {
        (**self).poll(ticket)
    }
    fn cancel(&mut self, ticket: EncoderTicket) {
        (**self).cancel(ticket);
    }
}

struct Pending {
    job: EncodeJob,
    polls: usize,
    fail: bool,
}
/// CPU fixture encoder: deterministic per-key bytes, manual poll delays, failure injection.
#[derive(Default)]
pub struct FakeEncoder {
    jobs: HashMap<EncoderTicket, Pending>,
    next: u64,
    pub delay_polls: usize,
    pub fail_next: bool,
    pub submitted: usize,
    pub cancelled: usize,
}
impl FakeEncoder {
    pub fn pending(&self) -> usize { self.jobs.len() }
    pub fn features(job: &EncodeJob) -> Result<Arc<[u8]>, MediaError> {
        let bytes = job.feature_bytes()?;
        let domain = if job.key.is_audio() { 0xa7 } else { 0 };
        Ok((0..bytes).map(|i| job.key.bytes()[i % 32].wrapping_add((i / 32) as u8).wrapping_add(domain))
            .collect::<Vec<_>>().into())
    }
}
impl EncoderClient for FakeEncoder {
    fn submit(&mut self, job: EncodeJob) -> Result<EncoderTicket, MediaError> {
        job.validate()?;
        let ticket = EncoderTicket(self.next);
        self.next += 1;
        self.jobs.insert(ticket, Pending { job, polls: self.delay_polls, fail: std::mem::take(&mut self.fail_next) });
        self.submitted += 1;
        Ok(ticket)
    }
    fn poll(&mut self, ticket: EncoderTicket) -> Option<Result<EncodeOutput, MediaError>> {
        let job = self.jobs.get_mut(&ticket)?;
        if job.polls > 0 { job.polls -= 1; return None; }
        let job = self.jobs.remove(&ticket).unwrap();
        Some(if job.fail { Err(MediaError::Encoder("fake failure".into())) } else {
            Self::features(&job.job).map(|features| EncodeOutput { key: job.job.key, features, elapsed_ms: 1.0 })
        })
    }
    fn cancel(&mut self, ticket: EncoderTicket) {
        if self.jobs.remove(&ticket).is_some() { self.cancelled += 1; }
    }
}
