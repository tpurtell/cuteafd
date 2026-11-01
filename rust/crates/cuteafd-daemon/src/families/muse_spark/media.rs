//! Muse media binding. Towers supply prepared inputs; this layer owns final LM
//! offsets, typed cache identities and BF16 replacement after text embed_norm.
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_core::{media::AudioTokenGeometry, MediaKey, MediaSpan};
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TowerKind { Image, Audio }
impl TowerKind {
    fn matches(self, key: MediaKey) -> bool { key.is_audio() == (self == Self::Audio) }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Markers {
    pub kind: TowerKind,
    pub placeholder: u32,
    pub start: u32,
    pub end: u32,
}

/// Marker ids missing from config.json come from the checkpoint's tokenizer,
/// not engine-owned constants. Audio and vision share one expansion algorithm.
pub(crate) struct Binding { markers: Vec<Markers>, video: Option<u32> }
impl Binding {
    pub fn from_snapshot(snapshot: &Path, vocabulary: usize) -> Result<Self> {
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let tokenizer: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("tokenizer_config.json"))?)?;
        let marker = |text: &str| -> Result<u32> {
            let mut found = tokenizer["added_tokens_decoder"].as_object().context("Muse tokenizer special tokens")?
                .iter().filter(|(_, value)| value["content"].as_str() == Some(text));
            let (id, _) = found.next().with_context(|| format!("Muse tokenizer missing {text}"))?;
            ensure!(found.next().is_none(), "duplicate Muse marker {text}");
            let id: u32 = id.parse()?;
            ensure!((id as usize) < vocabulary, "Muse marker outside vocabulary");
            Ok(id)
        };
        let mut markers = Vec::new();
        for (kind, tower, pad, start, end) in [
            (TowerKind::Image, "vision_config", "<|patch|>", "<|image_start|>", "<|image_end|>"),
            (TowerKind::Audio, "audio_config", "<|audio|>", "<|soft_audio_start|>", "<|soft_audio_end|>"),
        ] {
            if config.get(tower).is_none_or(serde_json::Value::is_null) { continue; }
            let placeholder = marker(pad)?;
            let field = if kind == TowerKind::Image { "image_token_id" } else { "audio_token_id" };
            ensure!(config[field].as_u64() == Some(u64::from(placeholder)), "Muse {field} differs from tokenizer");
            markers.push(Markers { kind, placeholder, start: marker(start)?, end: marker(end)? });
        }
        let mut ids = Vec::new();
        for m in &markers { ids.extend([m.placeholder, m.start, m.end]); }
        ids.sort_unstable();
        ensure!(ids.windows(2).all(|p| p[0] != p[1]), "overlapping Muse media markers");
        let video = config.get("video_token_id").map(|id| -> Result<_> {
            let value = u32::try_from(id.as_u64().context("Muse video token")?)?;
            ensure!(value == marker("<|video|>")?, "Muse video token differs from tokenizer");
            Ok(value)
        }).transpose()?;
        Ok(Self { markers, video })
    }

    fn expand(&self, tokens: &[u32], inputs: &[(TowerKind, usize, MediaKey)],
        vocabulary: usize, context: usize) -> Result<(Vec<u32>, Vec<MediaSpan>)> {
        ensure!(vocabulary > 0 && vocabulary < (1 << 31) && tokens.iter().all(|&id| (id as usize) < vocabulary),
            "Muse token outside vocabulary");
        ensure!(self.video.is_none_or(|id| !tokens.contains(&id)), "Muse video input is not supported");
        let mut cursors = vec![0; self.markers.len()];
        let mut lengths = tokens.len();
        for m in &self.markers {
            let rows = inputs.iter().filter(|(kind, _, _)| *kind == m.kind).collect::<Vec<_>>();
            ensure!(tokens.iter().filter(|&&id| id == m.placeholder).count() == rows.len(),
                "Muse {:?} placeholder count differs from inputs", m.kind);
            for (_, count, key) in rows {
                ensure!(*count > 0 && m.kind.matches(*key), "Muse media count/kind");
                // A template placeholder becomes start + N feature rows + end.
                lengths = lengths.checked_add(*count).and_then(|n| n.checked_add(1)).context("Muse media extent overflow")?;
            }
        }
        ensure!(inputs.iter().all(|(kind, _, _)| self.markers.iter().any(|m| m.kind == *kind)),
            "Muse checkpoint has no requested media tower");
        ensure!(lengths <= context, "expanded Muse media prompt exceeds context limit");
        let mut output = Vec::with_capacity(lengths);
        let mut spans = Vec::with_capacity(inputs.len());
        for &token in tokens {
            if let Some((index, m)) = self.markers.iter().enumerate().find(|(_, m)| m.placeholder == token) {
                let (_, count, key) = inputs.iter().filter(|(kind, _, _)| *kind == m.kind)
                    .nth(cursors[index]).context("Muse media cursor")?;
                cursors[index] += 1;
                output.push(m.start);
                spans.push(MediaSpan { start: output.len(), len: *count, key: *key });
                output.extend(std::iter::repeat_n(token, *count));
                output.push(m.end);
            } else {
                ensure!(!self.markers.iter().any(|m| token == m.start || token == m.end),
                    "unexpected Muse media boundary in unexpanded prompt");
                output.push(token);
            }
        }
        Ok((output, spans))
    }

    /// LM positions remain ordinary one-dimensional text positions. Check every
    /// final span against its own tower markers, including mixed-modality turns.
    pub fn coordinates(&self, tokens: &[u32], media: &RequestMedia) -> Result<()> {
        ensure!(media.prompt_len() == tokens.len(), "Muse media prompt length differs");
        for span in media.spans() {
            let m = self.markers.iter().find(|m| m.kind.matches(span.key)).context("Muse span without tower")?;
            let end = span.checked_end().context("Muse span extent")?;
            ensure!(span.start > 0 && end < tokens.len() && tokens[span.start - 1] == m.start
                && tokens[end] == m.end && tokens[span.start..end].iter().all(|&id| id == m.placeholder),
                "Muse media markers or feature rows differ");
        }
        for m in &self.markers {
            ensure!(tokens.iter().filter(|&&id| id == m.placeholder).count() == media.spans().iter()
                .filter(|s| m.kind.matches(s.key)).map(|s| s.len).sum::<usize>(), "unbound Muse media rows");
        }
        Ok(())
    }

    pub fn prepare(&self, job: NativeRequest, tokens: Vec<u32>, vocabulary: usize, hidden: usize, context: usize)
        -> Result<(crate::shared::qwen::scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        let policy = AudioTokenGeometry::PcmPatches { samples_per_token: 1920 };
        let mut inputs = Vec::new();
        let mut jobs = Vec::new();
        for image in &job.media {
            ensure!(image.grid.t == 1 && image.grid.h > 0 && image.grid.w > 0 && image.grid.h % 2 == 0
                && image.grid.w % 2 == 0 && image.tokens == (image.grid.h as usize * image.grid.w as usize) / 4,
                "Muse image grid/token count differs; video is unsupported");
            inputs.push((TowerKind::Image, image.tokens, image.key.into()));
            jobs.push(EncodeJob::image(image.key, [1, image.grid.h, image.grid.w], image.rgb8.clone(), image.tokens, hidden));
        }
        for clip in &job.audio {
            ensure!(clip.geometry.policy == policy && policy.tokens(clip.pcm.len()) == Some(clip.geometry.tokens),
                "Muse audio needs canonical 24 kHz PCM with ceil(samples/1920) rows");
            inputs.push((TowerKind::Audio, clip.geometry.tokens, clip.key.into()));
            jobs.push(EncodeJob::audio_with_geometry(clip.key, clip.pcm.clone(), clip.geometry.tokens, hidden, policy));
        }
        for encode in &jobs { encode.validate()?; }
        // Expanded diagnostic prompts must be validated against prepared media,
        // never expanded a second time. Mixed image/audio probes use final offsets.
        let (tokens, spans) = if let Some(probe) = job.probe.as_ref().filter(|p| p.spec.prompt_ids.is_some()) {
            ensure!(tokens.len() <= context && tokens.iter().all(|&id| (id as usize) < vocabulary), "Muse probe context/vocabulary");
            ensure!(self.video.is_none_or(|id| !tokens.contains(&id)), "Muse video input is not supported");
            ensure!(probe.spec.media.len() == job.media.len() && probe.spec.audio.len() == job.audio.len(), "Muse probe media count");
            let mut spans = Vec::new();
            for (span, image) in probe.spec.media.iter().zip(&job.media) {
                ensure!(span.len == image.tokens && span.grid == [image.grid.t, image.grid.h, image.grid.w]
                    && span.key == key_hex(image.key.into()), "Muse image probe identity differs");
                spans.push(MediaSpan { start: span.start, len: span.len, key: image.key.into() });
            }
            for (span, clip) in probe.spec.audio.iter().zip(&job.audio) {
                ensure!(span.len == clip.geometry.tokens && span.samples == clip.pcm.len() && span.key == key_hex(clip.key.into())
                    && span.pcm_sha256 == pcm_hash(&clip.pcm), "Muse audio probe identity differs");
                spans.push(MediaSpan { start: span.start, len: span.len, key: clip.key.into() });
            }
            spans.sort_by_key(|span| span.start);
            (tokens, spans)
        } else { self.expand(&tokens, &inputs, vocabulary, context)? };
        let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
        self.coordinates(&tokens, &media)?;
        if let Some(probe) = &job.probe {
            probe.media(spans.iter().filter(|s| !s.key.is_audio()).zip(&job.media).map(|(span, image)|
                cuteafd_api::openai::probe::ProbeMedia { start: span.start, len: span.len, kind: "image".into(),
                    key: key_hex(span.key), grid: [image.grid.t, image.grid.h, image.grid.w], fixture: None, image_url: None }).collect());
            probe.audio(spans.iter().filter(|s| s.key.is_audio()).zip(&job.audio).map(|(span, clip)|
                cuteafd_api::openai::probe::ProbeAudio { start: span.start, len: span.len, key: key_hex(span.key),
                    samples: clip.pcm.len(), pcm_sha256: pcm_hash(&clip.pcm) }).collect());
        }
        let keys = MediaKeys::new(&tokens, u32::try_from(vocabulary)?, &spans)?;
        Ok((crate::shared::qwen::scheduler::Prompt { job, tokens, keys }, media, jobs))
    }
}

fn key_hex(key: MediaKey) -> String { key.bytes().iter().map(|byte| format!("{byte:02x}")).collect() }
fn pcm_hash(pcm: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for value in pcm { hash.update(value.to_le_bytes()); }
    format!("{:x}", hash.finalize())
}

pub(super) fn inventory(snapshot: &Path, vision: cuteafd_loader::plan::MediaMode,
    audio: cuteafd_loader::plan::MediaMode, replicas: usize)
    -> Result<cuteafd_loader::placement::families::muse_spark::MediaOptions> {
    // Coordinator role filtering deliberately excludes towers; read their headers separately.
    let checkpoint = cuteafd_loader::plan::Checkpoint::open_for_role(snapshot, "Muse bundled media", |name| {
        vision != cuteafd_loader::plan::MediaMode::Off && ["model.vision_tower.", "model.vision_adapter.",
            "model.vision_projection."].iter().any(|prefix| name.starts_with(prefix))
    })?;
    cuteafd_loader::placement::families::muse_spark::MediaOptions::from_checkpoint(&checkpoint, vision, audio, replicas)
}

pub(super) struct Ready {
    pub encoder: crate::shared::qwen::Encoder,
    pub image: Option<std::sync::Arc<cuteafd_api::openai::media::MediaPreparer>>,
    pub audio: Option<std::sync::Arc<cuteafd_api::openai::media::audio::AudioPreparer>>,
    pub cache_bytes: usize,
    pub prefix: crate::shared::prefix::PrefixArgs,
}

pub(super) struct ApiReady {
    pub context: usize,
    pub image: Option<std::sync::Arc<cuteafd_api::openai::media::MediaPreparer>>,
    pub audio: Option<std::sync::Arc<cuteafd_api::openai::media::audio::AudioPreparer>>,
    pub image_health: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub audio_health: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl ApiReady {
    pub fn install(self, mut profile: cuteafd_api::openai::ModelProfile) -> Result<cuteafd_api::openai::ModelProfile> {
        if let Some(image) = self.image {
            ensure!(self.image_health.is_some(), "Muse image owner has no health handle");
            profile = profile.with_loaded_vision(image);
            profile.vision_health = self.image_health;
        }
        if let Some(audio) = self.audio {
            profile = profile.with_loaded_audio(audio, self.audio_health.context("Muse audio owner has no health handle")?);
        }
        Ok(profile)
    }
}

impl Ready {
    pub fn load(args: &super::serve::ServeArgs, library: &cuteafd_ffi::NativeLibrary,
        admitted: &cuteafd_loader::placement::families::muse_spark::MediaAdmission) -> Result<Self> {
        use crate::shared::{qwen::Encoder, vision::{self, EncoderService, local::LocalEncoder, remote::*}};
        use cuteafd_loader::{media::{ImageFamily, ProcessorConfig}, plan::encoder::EncoderKind};
        use std::{sync::Arc, time::Duration};
        args.validate()?;
        let image_on = admitted.vision.admitted_bytes() > 0;
        let audio_on = admitted.audio.admitted_bytes() > 0;
        ensure!(image_on || args.vision_peers.is_empty(), "disabled vision cannot have encoder peers");
        ensure!(audio_on || args.audio_peers.is_empty(), "disabled audio cannot have encoder peers");
        if matches!(admitted.audio.kind, EncoderKind::Spark { .. }) {
            ensure!(args.audio_peers.len() == admitted.audio.spark_ranks().len(),
                "Muse audio replica peers differ from admission");
        }
        let (prefix, cache_bytes) = if image_on || audio_on { args.prefix.with_media_headroom(args.media_cache_bytes)? }
            else { (args.prefix.clone(), 0) };
        if !image_on && !audio_on {
            ensure!(args.vision_peers.is_empty() && args.audio_peers.is_empty(), "disabled media cannot have encoder peers");
            return Ok(Self { encoder: Encoder::Off, image: None, audio: None, cache_bytes, prefix });
        }
        let processor = if image_on { ProcessorConfig::from_snapshot(&args.snapshot, ImageFamily::Muse)? }
            // The audio-only adapter never reads image normalization or patch fields.
            else { ProcessorConfig::for_family(ImageFamily::Muse) };
        let slots = Arc::new(tokio::sync::Semaphore::new(4));
        let revision = args.snapshot.file_name().and_then(|name| name.to_str()).context("Muse snapshot revision")?;
        let physical = |gpu: usize| -> Result<i32> { args.devices().get(gpu).copied().context("Muse encoder GPU outside coordinator ranks") };
        let mut image_encoder = Encoder::Off;
        let mut local_image = None;
        let mut image = None;
        let image_gpu = match admitted.vision.kind { EncoderKind::Rtx { gpu } => Some(physical(gpu)?), _ => None };
        if image_on {
            let spec = vision::TowerSpec::muse(&args.snapshot, 4096)?;
            let bytes = cuteafd_ffi::vision::NativeVision::required(&args.native_lib, &spec.native)?.total_bytes();
            ensure!(bytes == admitted.vision.admitted_bytes(), "Muse vision native ledger differs from admission");
            let (id, remote) = match admitted.vision.kind {
                EncoderKind::Rtx { .. } => {
                    ensure!(args.vision_peers.is_empty(), "RTX vision cannot use remote peers");
                    let info = library.cuda_device_info(image_gpu.context("vision device")?)?;
                    let sm = u32::try_from(info.compute_capability_major * 10 + info.compute_capability_minor)?;
                    ensure!([120, 121].contains(&sm), "Muse vision requires SM120/SM121");
                    (spec.encoder_id(revision, sm), None)
                }
                EncoderKind::Spark { .. } => {
                    ensure!(args.vision_peers.len() == admitted.vision.spark_ranks().len(), "Muse vision replica peers differ from admission");
                    let rev = args.encoder_revision.as_deref().context("Spark vision needs encoder revision")?;
                    let id = spec.encoder_id(rev, 121);
                    let plan_hash = vision::worker::parse_plan_hash(args.encoder_plan_hash.as_deref().context("Spark vision needs encoder plan hash")?)?;
                    let expected = EncoderHandshake { encoder_id: id, max_patches: 16384,
                        output_width: 8192, patch_size: processor.patch, merge_size: processor.merge, plan_hash };
                    (id, Some(RemoteEncoder::connect(args.vision_peers.clone(), expected, Duration::from_secs(60))?))
                }
                _ => anyhow::bail!("Muse vision placement has no runnable owner"),
            };
            let mut preparer = cuteafd_api::openai::media::MediaPreparer::for_loaded_encoder(processor.clone(), id, 4)?;
            preparer.slots = slots.clone();
            image = Some(Arc::new(preparer));
            if let Some(remote) = remote { image_encoder = Encoder::Remote(remote); }
            else { local_image = Some(spec); }
        }
        let audio_gpu = match admitted.audio.kind { EncoderKind::Rtx { gpu } => Some(physical(gpu)?), _ => None };
        let (audio_revision, audio_hash) = if matches!(admitted.audio.kind, EncoderKind::Spark { .. }) {
            (args.audio_encoder_revision.as_deref().context("Spark audio needs encoder revision")?,
                vision::worker::parse_plan_hash(args.audio_encoder_plan_hash.as_deref()
                    .context("Spark audio needs encoder plan hash")?)?)
        } else { (revision, [0; 32]) };
        let mut ready_audio = super::audio::ReadyAudio::load(&args.snapshot, &admitted.audio, &args.native_lib,
            audio_gpu.unwrap_or(args.device), audio_revision, audio_hash, args.audio_peers.clone(), slots)?;
        let audio = ready_audio.as_ref().map(|audio| audio.preparer.clone());
        let mut audio_encoder = ready_audio.as_mut().and_then(|audio| audio.remote.take()).map(Encoder::Remote).unwrap_or(Encoder::Off);
        let mut local_audio = ready_audio.and_then(|audio| audio.local);
        let shared_local = local_image.is_some() && local_audio.is_some() && image_gpu == audio_gpu;
        if let Some(spec) = local_image.take() {
            let service = EncoderService::start_with_audio(spec, args.native_lib.clone(), image_gpu.context("vision GPU")?,
                admitted.vision.admitted_bytes(), if shared_local { local_audio.take() } else { None })?;
            image_encoder = Encoder::Local(LocalEncoder::new(service, &processor, 8192, 4096));
        }
        if let Some(audio) = local_audio {
            let service = EncoderService::start_audio(args.native_lib.clone(), audio_gpu.context("audio GPU")?, audio)?;
            audio_encoder = Encoder::Local(LocalEncoder::new(service, &processor, 8192, 4096));
        }
        let encoder = if shared_local { image_encoder } else { Encoder::split(image_encoder, audio_encoder) };
        Ok(Self { encoder, image, audio, cache_bytes, prefix })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn binding() -> Binding {
        Binding { markers: vec![Markers { kind: TowerKind::Image, placeholder: 7, start: 8, end: 9 },
            Markers { kind: TowerKind::Audio, placeholder: 10, start: 11, end: 12 }], video: Some(13) }
    }
    #[test]
    fn served_media_inventory_reads_headers_excluded_from_coordinator_role() -> Result<()> {
        use cuteafd_loader::{plan::{Checkpoint, MediaMode, testing}, placement::families::muse_spark::MediaOptions};
        let dir = tempfile::tempdir()?;
        testing::write_muse_media_snapshot(dir.path(), &(0..62).collect::<Vec<_>>());
        let coordinator = Checkpoint::coordinator(dir.path(), false, false)?;
        assert!(!coordinator.tensors.iter().any(|tensor| tensor.meta.name.starts_with("model.vision_tower.")));
        let full = Checkpoint::open(dir.path())?;
        let expected = MediaOptions::from_checkpoint(&full, MediaMode::Auto, MediaMode::Auto, 2)?;
        let actual = inventory(dir.path(), MediaMode::Auto, MediaMode::Auto, 2)?;
        assert_eq!(actual.vision_bytes, expected.vision_bytes);
        assert_eq!(actual.audio_bytes, expected.audio_bytes);
        assert_eq!(actual.replicas, 2);
        let disabled = inventory(dir.path(), MediaMode::Off, MediaMode::Off, 1)?;
        assert_eq!((disabled.vision_bytes, disabled.audio_bytes), ((0, 0), (0, 0)));
        Ok(())
    }

    fn profile() -> Result<cuteafd_api::openai::ModelProfile> {
        use cuteafd_api::openai::{chat::muse::{MuseEncoding, MuseTokenIds}, ModelEncoding, ModelProfile};
        let encoding = MuseEncoding::new("{{ bos_token }}", MuseTokenIds {
            bos: 1, eos: vec![2], start: 3, message: 4, end_of_message: 5,
        }, "<bos>".into())?;
        Ok(ModelProfile::new("muse-test", ModelEncoding::Muse(std::sync::Arc::new(encoding))))
    }

    #[test]
    fn capabilities_are_installed_only_for_loaded_owners_with_live_health() -> Result<()> {
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
        use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig};
        for (vision, audio) in [(false, false), (true, false), (false, true), (true, true)] {
            let image_health = Arc::new(AtomicBool::new(true));
            let audio_health = Arc::new(AtomicBool::new(true));
            let ready = ApiReady {
                context: 1024,
                image: vision.then(|| cuteafd_api::openai::media::MediaPreparer::for_loaded_encoder(
                    ProcessorConfig::for_family(ImageFamily::Muse), EncoderId([1; 32]), 4).map(Arc::new)).transpose()?,
                audio: audio.then(|| Arc::new(cuteafd_api::openai::media::audio::AudioPreparer::with_geometry(
                    EncoderId([2; 32]), Arc::new(tokio::sync::Semaphore::new(4)),
                    AudioTokenGeometry::PcmPatches { samples_per_token: 1920 }))),
                image_health: vision.then(|| image_health.clone()),
                audio_health: audio.then(|| audio_health.clone()),
            };
            let profile = ready.install(profile()?)?;
            assert_eq!((profile.capabilities.vision, profile.capabilities.audio), (vision, audio));
            assert_eq!((profile.media_preparer.is_some(), profile.audio_preparer.is_some()), (vision, audio));
            image_health.store(false, Ordering::Release);
            audio_health.store(false, Ordering::Release);
            if vision { assert!(!profile.vision_health.as_ref().unwrap().load(Ordering::Acquire)); }
            if audio { assert!(!profile.audio_health.as_ref().unwrap().load(Ordering::Acquire)); }
        }
        Ok(())
    }

    #[test]
    fn readiness_rejects_a_preparer_without_its_owner_health() -> Result<()> {
        use std::sync::Arc;
        use cuteafd_loader::media::{EncoderId, ImageFamily, ProcessorConfig};
        let image = Arc::new(cuteafd_api::openai::media::MediaPreparer::for_loaded_encoder(
            ProcessorConfig::for_family(ImageFamily::Muse), EncoderId([1; 32]), 4)?);
        assert!(ApiReady { context: 1, image: Some(image), audio: None,
            image_health: None, audio_health: None }.install(profile()?).is_err());
        let audio = Arc::new(cuteafd_api::openai::media::audio::AudioPreparer::with_geometry(
            EncoderId([2; 32]), Arc::new(tokio::sync::Semaphore::new(4)),
            AudioTokenGeometry::PcmPatches { samples_per_token: 1920 }));
        assert!(ApiReady { context: 1, image: None, audio: Some(audio),
            image_health: None, audio_health: None }.install(profile()?).is_err());
        Ok(())
    }

    #[test]
    fn interleaved_towers_expand_at_final_offsets_with_typed_keys() -> Result<()> {
        let image = cuteafd_core::ImageKey([1; 32]).into();
        let audio = cuteafd_core::AudioKey([1; 32]).into();
        let (tokens, spans) = binding().expand(&[1, 10, 2, 7, 3, 10], &[
            (TowerKind::Image, 4, image), (TowerKind::Audio, 2, audio), (TowerKind::Audio, 1, audio)], 32, 32)?;
        assert_eq!(tokens, [1, 11, 10, 10, 12, 2, 8, 7, 7, 7, 7, 9, 3, 11, 10, 12]);
        assert_eq!(spans.iter().map(|s| (s.start, s.len, s.key.is_audio())).collect::<Vec<_>>(),
            [(2, 2, true), (7, 4, false), (14, 1, true)]);
        // Repeated keys must have one stable feature extent.
        assert!(RequestMedia::new(spans.clone(), 8, tokens.len()).is_err());
        let spans = vec![spans[0], spans[1], MediaSpan { key: cuteafd_core::AudioKey([2; 32]).into(), ..spans[2] }];
        binding().coordinates(&tokens, &RequestMedia::new(spans, 8, tokens.len())?)?;
        Ok(())
    }
    #[test]
    fn every_admitted_image_extent_has_exact_feature_and_marker_spans() -> Result<()> {
        let binding = binding();
        let key = cuteafd_core::ImageKey([3; 32]).into();
        for count in 1..=4096 {
            let (tokens, spans) = binding.expand(&[1, 7, 2], &[(TowerKind::Image, count, key)], 32, 4100)?;
            assert_eq!(tokens.len(), count + 4);
            assert_eq!(spans, [MediaSpan { start: 2, len: count, key }]);
            assert_eq!(tokens[1], 8);
            assert!(tokens[2..2 + count].iter().all(|&id| id == 7));
            assert_eq!(&tokens[2 + count..], &[9, 2]);
            binding.coordinates(&tokens, &RequestMedia::new(spans, 8, tokens.len())?)?;
        }
        Ok(())
    }
    #[test]
    fn malformed_media_is_rejected_before_allocation() {
        let input = [(TowerKind::Image, 4, cuteafd_core::ImageKey([0; 32]).into())];
        for tokens in [&[13][..], &[7, 7], &[8, 7, 9], &[32], &[]] {
            assert!(binding().expand(tokens, &input, 32, 32).is_err());
        }
        assert!(binding().expand(&[7], &input, 32, 5).is_err());
        assert!(binding().expand(&[7], &[(TowerKind::Image, usize::MAX, input[0].2)], 32, usize::MAX).is_err());
        assert!(binding().expand(&[7], &[(TowerKind::Image, 0, input[0].2)], 32, 32).is_err());
    }
}
