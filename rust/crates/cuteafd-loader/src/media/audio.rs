//! Bounded, CPU-only MiMo audio preparation. No tokenizer decoder or CUDA is involved.
use super::{EncoderId, PreprocessId};
use cuteafd_core::{AudioKey, media::AudioTokenGeometry};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{io::Cursor, sync::Arc};

pub const SAMPLE_RATE: u32 = 24_000;
pub const MAX_CLIP_SAMPLES: usize = 300 * SAMPLE_RATE as usize;
pub const MAX_REQUEST_SAMPLES: usize = 600 * SAMPLE_RATE as usize;
pub const MAX_CLIPS: usize = 4;
const FFT: usize = 960;
const HOP: usize = 240;
const SEGMENT: usize = 6000;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("invalid audio: {0}")]
    Invalid(String),
    #[error("audio decoding failed: {0}")]
    Decode(String),
    #[error("audio clip too short: {0} samples, need > 480")]
    TooShort(usize),
    #[error("audio clip exceeds 300 seconds")]
    TooLong,
}
type Result<T> = std::result::Result<T, AudioError>;
fn invalid(message: impl Into<String>) -> AudioError {
    AudioError::Invalid(message.into())
}
fn decode_error(error: impl std::fmt::Display) -> AudioError {
    AudioError::Decode(error.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
}
impl std::str::FromStr for AudioFormat {
    type Err = AudioError;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "wav" => Ok(Self::Wav),
            "mp3" => Ok(Self::Mp3),
            "flac" => Ok(Self::Flac),
            _ => Err(invalid("format must be wav, mp3 or flac")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AudioDecodeLimits {
    pub encoded_bytes: usize,
    pub source_pcm_bytes: usize,
}
impl Default for AudioDecodeLimits {
    fn default() -> Self {
        Self {
            encoded_bytes: 128 << 20,
            source_pcm_bytes: 256 << 20,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioGeometry {
    #[serde(default, skip_serializing_if = "AudioTokenGeometry::is_mimo")]
    pub policy: AudioTokenGeometry,
    pub mel_frames: usize,
    pub segments: Vec<usize>,
    pub codes: usize,
    pub tokens: usize,
}
impl AudioGeometry {
    pub fn for_samples(samples: usize) -> Result<Self> {
        Self::for_policy(samples, AudioTokenGeometry::MimoRvq)
    }
    pub fn for_policy(samples: usize, policy: AudioTokenGeometry) -> Result<Self> {
        if samples > MAX_CLIP_SAMPLES { return Err(AudioError::TooLong); }
        if let AudioTokenGeometry::PcmPatches { .. } = policy {
            let tokens = policy.tokens(samples).ok_or_else(|| invalid("empty PCM or zero patch stride"))?;
            return Ok(Self { policy, mel_frames: 0, segments: Vec::new(), codes: 0, tokens });
        }
        if samples <= FFT / 2 {
            return Err(AudioError::TooShort(samples));
        }
        if samples > MAX_CLIP_SAMPLES {
            return Err(AudioError::TooLong);
        }
        let mel_frames = samples / HOP + 1;
        let mut segments = vec![SEGMENT; mel_frames / SEGMENT];
        if mel_frames % SEGMENT != 0 {
            segments.push(mel_frames % SEGMENT);
        }
        let codes = segments
            .iter()
            .map(|n| n.div_ceil(2).div_ceil(2))
            .sum::<usize>();
        Ok(Self {
            policy,
            mel_frames,
            segments,
            codes,
            tokens: codes.div_ceil(4),
        })
    }
}

#[derive(Debug, Clone)]
pub struct PreparedAudio {
    pub key: AudioKey,
    /// Canonical identity bytes are these finite samples serialized as f32 LE.
    pub pcm: Arc<[f32]>,
    pub geometry: AudioGeometry,
}

/// Identity of the pinned mel, tokenizer geometry, decode and resampling policy.
/// A native numerical change also bumps EncoderId's numerical version.
pub fn preprocess_id() -> PreprocessId {
    let mut hash = Sha256::new();
    hash.update(
        concat!(
            "cuteafd-audio-preprocess-v1;",
            "upstream=b62b59922979bf9f389b373169298a251587653f;",
            "pcm=f32le,24000,mono,finite;channels=resample_then_mean;",
            "resample=sinc_interp_hann,width6,rolloff0.99,zero_pad,ceil_length;",
            "decode=hound3.5.1,claxon0.4.3,nanomp3_0.2.0_gapless;",
            "stft=960,960,240,periodic_hann,reflect480,center,unnormalized,onesided,magnitude;",
            "mel=htk,128,0,12000,no_norm,ln_clamp1e-7;",
            "tokenizer=segment6000,conv3pad1,conv3stride2pad1,conv2stride2;",
            "batch=single_complete_clip,official_internal_padding,no_cross_clip;",
            "rvq=20,f32,first_tie;group4,repeat_last;mel_device=cuda_cufft;cpu_decode_numerics=2"
        )
        .as_bytes(),
    );
    PreprocessId(hash.finalize().into())
}

pub fn prepare_pcm(pcm: Vec<f32>, encoder: EncoderId) -> Result<PreparedAudio> {
    prepare_pcm_with_geometry(pcm, encoder, AudioTokenGeometry::MimoRvq)
}

/// The PCM stays canonical FP32; a native tower owns its specified BF16 boundary.
pub fn prepare_pcm_with_geometry(pcm: Vec<f32>, encoder: EncoderId, policy: AudioTokenGeometry) -> Result<PreparedAudio> {
    let geometry = AudioGeometry::for_policy(pcm.len(), policy)?;
    if pcm.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite PCM"));
    }
    let mut hash = Sha256::new();
    hash.update(b"cuteafd-audio-v1");
    hash.update(encoder.0);
    match policy {
        AudioTokenGeometry::MimoRvq => hash.update(preprocess_id().0),
        AudioTokenGeometry::PcmPatches { samples_per_token } => {
            hash.update(concat!("cuteafd-pcm-patch-preprocess-v1;pcm=f32le,24000,mono,finite;",
                "channels=resample_then_mean;resample=sinc_interp_hann,width6,rolloff0.99,zero_pad,ceil_length;",
                "decode=hound3.5.1,claxon0.4.3,nanomp3_0.2.0_gapless;cpu_decode_numerics=2;",
                "features=ceil_valid_samples/stride;right_pad_final_only").as_bytes());
            hash.update((samples_per_token as u64).to_le_bytes());
        }
    }
    hash.update((pcm.len() as u64).to_le_bytes());
    for sample in &pcm {
        hash.update(sample.to_le_bytes());
    }
    Ok(PreparedAudio {
        key: AudioKey(hash.finalize().into()),
        pcm: pcm.into(),
        geometry,
    })
}

pub fn prepare(
    encoded: &[u8],
    format: AudioFormat,
    encoder: EncoderId,
    limits: AudioDecodeLimits,
) -> Result<PreparedAudio> {
    prepare_pcm(decode(encoded, format, limits)?, encoder)
}

#[derive(Debug)]
struct Pcm {
    rate: u32,
    channels: usize,
    samples: Vec<f32>,
    max_samples: usize,
}
impl Pcm {
    fn new(
        rate: u32,
        channels: usize,
        declared_frames: Option<u64>,
        limits: AudioDecodeLimits,
    ) -> Result<Self> {
        if !(8_000..=192_000).contains(&rate) || !(1..=8).contains(&channels) {
            return Err(invalid(
                "sample rate must be 8000..192000 Hz and channels 1..8",
            ));
        }
        let duration_frames = u64::from(rate) * 300;
        if declared_frames.is_some_and(|n| n > duration_frames) {
            return Err(AudioError::TooLong);
        }
        let max_samples = (duration_frames as usize * channels)
            .min(limits.source_pcm_bytes / std::mem::size_of::<f32>());
        if declared_frames.is_some_and(|n| {
            n.checked_mul(channels as u64)
                .is_none_or(|samples| samples > max_samples as u64)
        }) {
            return Err(invalid("decoded source PCM exceeds byte limit"));
        }
        Ok(Self {
            rate,
            channels,
            samples: Vec::new(),
            max_samples,
        })
    }
    fn push(&mut self, value: f32) -> Result<()> {
        if !value.is_finite() {
            return Err(invalid("nonfinite PCM"));
        }
        if self.samples.len() == self.max_samples {
            return Err(invalid(
                "audio exceeds duration or decoded source PCM byte limit",
            ));
        }
        if self.samples.len() == self.samples.capacity() {
            let count = (16_384).min(self.max_samples - self.samples.len());
            self.samples
                .try_reserve_exact(count)
                .map_err(decode_error)?;
        }
        self.samples.push(value);
        Ok(())
    }
    fn finish(self) -> Result<Vec<f32>> {
        if self.samples.len() % self.channels != 0 {
            return Err(invalid("incomplete interleaved PCM frame"));
        }
        resample_pcm(&self.samples, self.rate, self.channels)
    }
}

/// The encoded extent and decoded accumulator are checked before allocation.
pub fn decode(encoded: &[u8], format: AudioFormat, limits: AudioDecodeLimits) -> Result<Vec<f32>> {
    decode_with_geometry(encoded, format, limits, AudioTokenGeometry::MimoRvq)
}

pub fn decode_with_geometry(encoded: &[u8], format: AudioFormat, limits: AudioDecodeLimits, policy: AudioTokenGeometry) -> Result<Vec<f32>> {
    let pcm = decode_pcm(encoded, format, limits)?;
    AudioGeometry::for_policy(pcm.len(), policy)?;
    Ok(pcm)
}

fn decode_pcm(encoded: &[u8], format: AudioFormat, limits: AudioDecodeLimits) -> Result<Vec<f32>> {
    if encoded.is_empty() || encoded.len() > limits.encoded_bytes {
        return Err(invalid("encoded audio is empty or exceeds byte limit"));
    }
    let pcm = match format {
        AudioFormat::Wav => {
            let mut wav = hound::WavReader::new(Cursor::new(encoded)).map_err(decode_error)?;
            let spec = wav.spec();
            let mut pcm = Pcm::new(
                spec.sample_rate,
                spec.channels as usize,
                Some(wav.duration() as u64),
                limits,
            )?;
            match spec.sample_format {
                hound::SampleFormat::Float => {
                    for sample in wav.samples::<f32>() {
                        pcm.push(sample.map_err(decode_error)?)?;
                    }
                }
                hound::SampleFormat::Int => {
                    if !(1..=32).contains(&spec.bits_per_sample) {
                        return Err(invalid("unsupported WAV PCM bit depth"));
                    }
                    let scale = 2f32.powi(-(i32::from(spec.bits_per_sample) - 1));
                    for sample in wav.samples::<i32>() {
                        pcm.push(sample.map_err(decode_error)? as f32 * scale)?;
                    }
                }
            }
            pcm
        }
        AudioFormat::Flac => {
            // Parse only STREAMINFO; skip bounded metadata extents without allocating
            // application/comment/picture records that the input encoder never uses.
            let (info, offset) = flac_streaminfo(encoded)?;
            let mut cursor = Cursor::new(encoded);
            cursor.set_position(offset as u64);
            let mut pcm = Pcm::new(
                info.sample_rate,
                info.channels as usize,
                info.samples,
                limits,
            )?;
            if !(1..=32).contains(&info.bits_per_sample) {
                return Err(invalid("unsupported FLAC PCM bit depth"));
            }
            let scale = 2f32.powi(-(info.bits_per_sample as i32 - 1));
            let mut buffer = Vec::new();
            let mut frames = 0u64;
            loop {
                let offset = cursor.position() as usize;
                if offset == encoded.len() {
                    break;
                }
                validate_flac_frame(&encoded[offset..], &info)?;
                let block = claxon::frame::FrameReader::new(&mut cursor)
                    .read_next_or_eof(buffer)
                    .map_err(decode_error)?
                    .ok_or_else(|| invalid("truncated FLAC frame"))?;
                // Fixed-block streams number their final short block using its
                // shorter duration, so only variable-block sample numbers are exact.
                if encoded[offset + 1] & 1 != 0 && block.time() != frames {
                    return Err(invalid("FLAC sample sequence is discontinuous"));
                }
                frames += u64::from(block.duration());
                if block.channels() as usize != pcm.channels {
                    return Err(invalid("FLAC channel count changed"));
                }
                for frame in 0..block.duration() {
                    for channel in 0..block.channels() {
                        pcm.push(block.sample(channel, frame) as f32 * scale)?;
                    }
                }
                buffer = block.into_buffer();
            }
            if info.samples.is_some_and(|declared| declared != frames) {
                return Err(invalid("FLAC decoded length differs from STREAMINFO"));
            }
            pcm
        }
        AudioFormat::Mp3 => {
            // Safe Rust decoder, gapless trimming, bounded frame buffer; no C bindings.
            let mut mp3 = nanomp3::SliceReader::<f32>::with_options(
                encoded,
                nanomp3::Options::default().skip_scan(true),
            );
            let channels = mp3
                .channels()
                .ok_or_else(|| invalid("MP3 has no audio frames"))?
                .num();
            let mut pcm = Pcm::new(
                mp3.sample_rate(),
                channels as usize,
                mp3.total_samples(),
                limits,
            )?;
            while let Some(frame) = mp3.read_frame().map_err(decode_error)? {
                for &sample in frame {
                    pcm.push(sample)?;
                }
            }
            pcm
        }
    };
    // No compressed bytes or decoder state survive request preparation.
    pcm.finish()
}

fn flac_streaminfo(encoded: &[u8]) -> Result<(claxon::metadata::StreamInfo, usize)> {
    if encoded.get(..4) != Some(b"fLaC") {
        return Err(invalid("missing FLAC stream marker"));
    }
    let mut offset = 4usize;
    let mut info = None;
    loop {
        let header = encoded
            .get(offset..offset + 4)
            .ok_or_else(|| invalid("truncated FLAC metadata header"))?;
        let kind = header[0] & 127;
        let last = header[0] & 128 != 0;
        let length = u32::from_be_bytes([0, header[1], header[2], header[3]]);
        let start = offset + 4;
        let end = start + length as usize;
        let body = encoded
            .get(start..end)
            .ok_or_else(|| invalid("truncated FLAC metadata"))?;
        if info.is_none() {
            if kind != 0 {
                return Err(invalid("FLAC STREAMINFO must be first"));
            }
            info = match claxon::metadata::read_metadata_block(&mut Cursor::new(body), 0, length)
                .map_err(decode_error)?
            {
                claxon::metadata::MetadataBlock::StreamInfo(info) => Some(info),
                _ => return Err(invalid("missing FLAC STREAMINFO")),
            };
        } else if kind == 0 || kind > 6 {
            return Err(invalid("duplicate or unsupported FLAC metadata block"));
        }
        offset = end;
        if last {
            return Ok((info.unwrap(), offset));
        }
    }
}

/// Claxon checks CRCs/bitstreams but does not expose rate or bit-depth changes.
fn validate_flac_frame(frame: &[u8], info: &claxon::metadata::StreamInfo) -> Result<()> {
    let header = frame
        .get(..5)
        .ok_or_else(|| invalid("truncated FLAC frame header"))?;
    if header[0] != 255 || header[1] & 254 != 248 {
        return Err(invalid("invalid FLAC frame sync"));
    }
    let assignment = header[3] >> 4;
    let channels = match assignment {
        0..=7 => u32::from(assignment) + 1,
        8..=10 => 2,
        _ => return Err(invalid("invalid FLAC channel assignment")),
    };
    let bits = match (header[3] >> 1) & 7 {
        0 => info.bits_per_sample,
        1 => 8,
        2 => 12,
        4 => 16,
        5 => 20,
        6 => 24,
        _ => return Err(invalid("unsupported FLAC frame bit depth")),
    };
    if channels != info.channels || bits != info.bits_per_sample {
        return Err(invalid("FLAC channel count or bit depth changed"));
    }
    let extra = match header[4].leading_ones() {
        0 => 0,
        2..=7 => header[4].leading_ones() as usize - 1,
        _ => return Err(invalid("invalid FLAC frame number")),
    };
    let mut offset = 5 + extra;
    offset += match header[2] >> 4 {
        6 => 1,
        7 => 2,
        _ => 0,
    };
    let rate = match header[2] & 15 {
        0 => info.sample_rate,
        1 => 88200,
        2 => 176400,
        3 => 192000,
        4 => 8000,
        5 => 16000,
        6 => 22050,
        7 => 24000,
        8 => 32000,
        9 => 44100,
        10 => 48000,
        11 => 96000,
        12 => {
            u32::from(
                *frame
                    .get(offset)
                    .ok_or_else(|| invalid("truncated FLAC sample rate"))?,
            ) * 1000
        }
        13 | 14 => {
            let bytes = frame
                .get(offset..offset + 2)
                .ok_or_else(|| invalid("truncated FLAC sample rate"))?;
            u32::from(u16::from_be_bytes([bytes[0], bytes[1]]))
                * if header[2] & 15 == 14 { 10 } else { 1 }
        }
        _ => return Err(invalid("invalid FLAC sample rate")),
    };
    if rate != info.sample_rate {
        return Err(invalid("FLAC sample rate changed"));
    }
    Ok(())
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Torchaudio's default Hann-windowed sinc geometry, resampling each channel
/// before averaging. Sparse phases avoid unbounded coprime-rate filter tables.
pub fn resample_mono(interleaved: &[f32], rate: u32, channels: usize) -> Result<Vec<f32>> {
    let pcm = resample_pcm(interleaved, rate, channels)?;
    AudioGeometry::for_samples(pcm.len())?;
    Ok(pcm)
}

fn resample_pcm(interleaved: &[f32], rate: u32, channels: usize) -> Result<Vec<f32>> {
    if !(8_000..=192_000).contains(&rate)
        || !(1..=8).contains(&channels)
        || interleaved.len() % channels != 0
        || interleaved.iter().any(|x| !x.is_finite())
    {
        return Err(invalid(
            "invalid interleaved PCM geometry or nonfinite samples",
        ));
    }
    let frames = interleaved.len() / channels;
    if frames > rate as usize * 300 {
        return Err(AudioError::TooLong);
    }
    let count = (frames * SAMPLE_RATE as usize).div_ceil(rate as usize);
    if count == 0 { return Err(invalid("empty PCM")); }
    if count > MAX_CLIP_SAMPLES { return Err(AudioError::TooLong); }
    let mut output = vec![0f32; count];
    if rate == SAMPLE_RATE {
        for (out, frame) in output.iter_mut().zip(interleaved.chunks_exact(channels)) {
            *out = frame.iter().sum::<f32>() / channels as f32;
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite mono PCM"));
        }
        return Ok(output);
    }
    let common = gcd(rate, SAMPLE_RATE);
    let orig = rate / common;
    let new = SAMPLE_RATE / common;
    let base = f64::from(orig.min(new)) * 0.99;
    let width = (6.0 * f64::from(orig) / base).ceil() as isize;
    let phases = (0..new)
        .map(|phase| {
            // The official phase arange is float32; its sample-index term is float64.
            let offset = (-(phase as f32) / new as f32) as f64;
            let center = -offset * f64::from(orig);
            let left = (center - width as f64).floor() as isize;
            let right = (center + width as f64).ceil() as isize;
            (left..=right)
                .map(|index| {
                    let t = ((offset + index as f64 / f64::from(orig)) * base).clamp(-6.0, 6.0);
                    let window = (t * std::f64::consts::PI / 6.0 / 2.0).cos().powi(2);
                    let angle = t * std::f64::consts::PI;
                    let sinc = if angle == 0.0 {
                        1.0
                    } else {
                        angle.sin() / angle
                    };
                    (index, (sinc * (window * (base / f64::from(orig)))) as f32)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // The pinned CPU oracle uses oneDNN for batches and MKL SGEMM for wide
    // phase banks; both accumulate fused. Its small mono THNN path is unfused.
    let fused = channels > 1 || new > 3;
    for (j, out) in output.iter_mut().enumerate() {
        let shift = (j / new as usize * orig as usize) as isize;
        for channel in 0..channels {
            let mut sum = 0f32;
            for &(index, weight) in &phases[j % new as usize] {
                let frame = shift + index;
                if frame >= 0 && (frame as usize) < frames {
                    let sample = interleaved[frame as usize * channels + channel];
                    sum = if fused {
                        sample.mul_add(weight, sum)
                    } else {
                        sum + sample * weight
                    };
                }
            }
            *out += sum;
        }
        *out /= channels as f32;
    }
    if output.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite resampled PCM"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(rate: u32, channels: u16, values: &[f32]) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut out,
            hound::WavSpec {
                channels,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for &value in values {
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        out.into_inner()
    }

    #[test]
    fn exact_geometry_and_limits() {
        let max = AudioGeometry::for_samples(MAX_CLIP_SAMPLES).unwrap();
        assert_eq!((max.mel_frames, max.codes, max.tokens), (30001, 7501, 1876));
        assert_eq!(max.segments, [6000, 6000, 6000, 6000, 6000, 1]);
        for n in [0, 1, 479, 480] {
            assert_eq!(
                AudioGeometry::for_samples(n).unwrap_err().to_string(),
                format!("audio clip too short: {n} samples, need > 480")
            );
        }
        assert!(matches!(
            AudioGeometry::for_samples(MAX_CLIP_SAMPLES + 1),
            Err(AudioError::TooLong)
        ));
        for (n, frames, codes, tokens) in [
            (481, 3, 1, 1),
            (961, 5, 2, 1),
            (24000, 101, 26, 7),
            (1439760, 6000, 1500, 375),
            (1440000, 6001, 1501, 376),
        ] {
            let g = AudioGeometry::for_samples(n).unwrap();
            assert_eq!((g.mel_frames, g.codes, g.tokens), (frames, codes, tokens));
        }
    }

    #[test]
    fn wav_float_mono_and_stereo_mean() {
        let values = vec![0.25; 1000];
        assert_eq!(
            decode(
                &wav(24000, 1, &values),
                AudioFormat::Wav,
                AudioDecodeLimits::default()
            )
            .unwrap(),
            values
        );
        let stereo = [0.25f32, -0.5].repeat(1000);
        assert_eq!(
            decode(
                &wav(24000, 2, &stereo),
                AudioFormat::Wav,
                AudioDecodeLimits::default()
            )
            .unwrap(),
            vec![-0.125; 1000]
        );
    }

    #[test]
    fn malformed_nonfinite_truncated_and_bounded_input() {
        for format in [AudioFormat::Wav, AudioFormat::Mp3, AudioFormat::Flac] {
            assert!(decode(b"not audio", format, AudioDecodeLimits::default()).is_err());
            assert!(decode(&[], format, AudioDecodeLimits::default()).is_err());
        }
        let encoded = wav(24000, 1, &vec![0.0; 1000]);
        let limits = AudioDecodeLimits {
            encoded_bytes: encoded.len() - 1,
            ..AudioDecodeLimits::default()
        };
        assert!(decode(&encoded, AudioFormat::Wav, limits).is_err());
        let limits = AudioDecodeLimits {
            source_pcm_bytes: 3999,
            ..AudioDecodeLimits::default()
        };
        assert!(decode(&encoded, AudioFormat::Wav, limits).is_err());
        assert!(decode(
            &encoded[..encoded.len() - 1],
            AudioFormat::Wav,
            AudioDecodeLimits::default()
        )
        .is_err());
        assert!(decode(
            &wav(24000, 1, &vec![f32::NAN; 1000]),
            AudioFormat::Wav,
            AudioDecodeLimits::default()
        )
        .is_err());
        assert!(resample_mono(&[0.0; 1000], 0, 1).is_err());
        assert!(resample_mono(&[0.0; 1001], 24000, 2).is_err());
    }

    #[test]
    fn canonical_key_and_encoder_are_part_of_identity() {
        let a = prepare_pcm(vec![0.25; 1000], EncoderId([1; 32])).unwrap();
        let b = prepare_pcm(vec![0.25; 1000], EncoderId([1; 32])).unwrap();
        let c = prepare_pcm(vec![0.25; 1000], EncoderId([2; 32])).unwrap();
        let d = prepare_pcm(vec![0.5; 1000], EncoderId([1; 32])).unwrap();
        assert_eq!(a.key, b.key);
        assert_ne!(a.key, c.key);
        assert_ne!(a.key, d.key);
        assert!(prepare_pcm(vec![f32::INFINITY; 1000], EncoderId([1; 32])).is_err());
    }

    #[test]
    fn resampling_ceil_length_and_channel_order() {
        for rate in [8000, 16000, 22050, 44100, 48000, 96000, 192000] {
            let frames = rate as usize / 10 + 1;
            let pcm = [0.25f32, -0.25].repeat(frames);
            let mono = resample_mono(&pcm, rate, 2).unwrap();
            assert_eq!(mono.len(), (frames * 24000).div_ceil(rate as usize));
            assert!(mono.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn malformed_rate_channels_and_finite_input_overflow_are_rejected() {
        for (rate, channels) in [(7999, 1), (192001, 1), (24000, 0), (24000, 9)] {
            assert!(Pcm::new(rate, channels, None, AudioDecodeLimits::default()).is_err());
        }
        assert!(Pcm::new(24000, 8, Some(u64::MAX), AudioDecodeLimits::default()).is_err());
        assert!(resample_mono(&vec![f32::MAX; 1000], 24000, 2).is_err());
    }

    #[test]
    fn clip_duration_limit_checks_declared_and_actual_frames() {
        let limits = AudioDecodeLimits::default();
        assert!(matches!(
            Pcm::new(24000, 1, Some(7_200_001), limits),
            Err(AudioError::TooLong)
        ));
        let mut pcm = Pcm::new(
            24000,
            1,
            None,
            AudioDecodeLimits {
                source_pcm_bytes: 481 * 4,
                ..limits
            },
        )
        .unwrap();
        for _ in 0..481 {
            pcm.push(0.0).unwrap();
        }
        assert!(pcm.push(0.0).is_err());
        assert_eq!(pcm.finish().unwrap().len(), 481);
        assert!(resample_mono(&vec![0.0; 7_200_001], 24000, 1).is_err());
    }

    #[test]
    fn flac_metadata_and_frame_geometry_fail_closed() {
        let mut stream = b"fLaC".to_vec();
        stream.extend([128, 0, 0, 34]);
        let mut body = [0u8; 34];
        body[..4].copy_from_slice(&[0, 16, 0, 16]);
        let packed = (24000u64 << 44) | (15u64 << 36) | 1000;
        body[10..18].copy_from_slice(&packed.to_be_bytes());
        stream.extend(body);
        let (info, offset) = flac_streaminfo(&stream).unwrap();
        assert_eq!(offset, 42);
        assert_eq!(info.samples, Some(1000));
        let frame = [255, 248, 135, 8, 0];
        validate_flac_frame(&frame, &info).unwrap();
        for header in [
            [255, 248, 138, 8, 0],
            [255, 248, 135, 24, 0],
            [255, 248, 135, 12, 0],
            [255, 248, 143, 8, 0],
            [255, 248, 135, 8, 128],
        ] {
            assert!(validate_flac_frame(&header, &info).is_err());
        }
        assert!(flac_streaminfo(&stream[..41]).is_err());
        stream[4] = 0;
        assert!(flac_streaminfo(&stream).is_err());
        stream.extend([128, 0, 0, 34]);
        stream.extend(body);
        assert!(flac_streaminfo(&stream).is_err());
    }

    #[test]
    fn wav_integer_bit_depth_normalization_and_repeat_decode() {
        for bits in [8, 16, 24, 32] {
            let mut out = Cursor::new(Vec::new());
            let mut writer = hound::WavWriter::new(
                &mut out,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 24000,
                    bits_per_sample: bits,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            let peak = 1i32 << (bits - 2);
            for _ in 0..250 {
                for value in [-peak, 0, peak, 0] {
                    writer.write_sample(value).unwrap();
                }
            }
            writer.finalize().unwrap();
            let encoded = out.into_inner();
            let a = decode(&encoded, AudioFormat::Wav, AudioDecodeLimits::default()).unwrap();
            assert_eq!(a, [-0.5, 0.0, 0.5, 0.0].repeat(250));
            assert_eq!(
                a,
                decode(&encoded, AudioFormat::Wav, AudioDecodeLimits::default()).unwrap()
            );
        }
    }
}

#[cfg(test)]
mod patch_geometry_tests {
    use super::*;
    #[test]
    fn short_pcm_patch_clips_and_identity_are_geometry_specific() {
        let policy = AudioTokenGeometry::PcmPatches { samples_per_token: 1920 };
        for samples in [1, 480, 481, 1920, 1921, 24000] {
            let clip = prepare_pcm_with_geometry(vec![0.25; samples], EncoderId([1; 32]), policy).unwrap();
            assert_eq!(clip.geometry.tokens, samples.div_ceil(1920));
            assert_eq!(clip.geometry.policy, policy);
            assert!(clip.geometry.segments.is_empty());
        }
        let pcm = vec![0.25; 24000];
        let old = prepare_pcm(pcm.clone(), EncoderId([1; 32])).unwrap();
        let explicit = prepare_pcm_with_geometry(pcm.clone(), EncoderId([1; 32]), AudioTokenGeometry::MimoRvq).unwrap();
        assert_eq!(old.key, explicit.key);
        let json = serde_json::to_string(&old.geometry).unwrap();
        assert_eq!(json, r#"{"mel_frames":101,"segments":[101],"codes":26,"tokens":7}"#);
        let patch = prepare_pcm_with_geometry(pcm, EncoderId([1; 32]), policy).unwrap();
        assert_ne!(old.key, patch.key);
        assert!(prepare_pcm_with_geometry(vec![], EncoderId([1;32]), policy).is_err());
        let mut data = Cursor::new(Vec::new());
        { let mut w = hound::WavWriter::new(&mut data, hound::WavSpec { channels:1, sample_rate:24000,
              bits_per_sample:16, sample_format:hound::SampleFormat::Int }).unwrap();
          for _ in 0..16 { w.write_sample(123i16).unwrap(); } w.finalize().unwrap(); }
        assert_eq!(decode_with_geometry(data.get_ref(), AudioFormat::Wav, Default::default(), policy).unwrap().len(), 16);
        assert!(decode(data.get_ref(), AudioFormat::Wav, Default::default()).is_err());
    }
}
