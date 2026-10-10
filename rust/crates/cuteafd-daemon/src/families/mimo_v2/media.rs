//! MiMo's image request state and local/remote encoder readiness.
use anyhow::{Context, Result};
use cuteafd_api::openai::{media::MediaPreparer, NativeRequest};
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use cuteafd_loader::{media::{ImageFamily, ProcessorConfig, SpanExpander}, plan::MediaMode};
use std::sync::Arc;

fn vision_config(mode: MediaMode, snapshot: &std::path::Path) -> Result<Option<serde_json::Value>> {
    if mode == MediaMode::Off { return Ok(None); }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
    Ok(config.get("vision_config").is_some().then_some(config))
}

pub(super) struct RemoteVision {
    addresses: Vec<std::net::SocketAddr>,
    plan_hash: [u8; 32],
    revision: String,
}
impl RemoteVision {
    pub fn from_args(args: &super::serve::ServeArgs) -> Result<Option<Self>> {
        let Some(peers) = &args.vision_peers else {
            anyhow::ensure!(args.encoder_plan_hash.is_none() && args.encoder_revision.is_none(),
                "encoder plan hash/revision require --vision-peers");
            return Ok(None);
        };
        anyhow::ensure!(matches!(args.vision, MediaMode::Auto | MediaMode::Spark(_)),
            "--vision-peers requires Spark/auto vision placement");
        let addresses = peers.split(',').map(|peer| peer.trim().parse()
            .context("vision peer must be an IP:port")).collect::<Result<Vec<_>>>()?;
        anyhow::ensure!((1..=6).contains(&addresses.len()), "vision needs 1..6 replicas");
        let plan_hash = crate::shared::vision::worker::parse_plan_hash(args.encoder_plan_hash.as_deref()
            .context("--vision-peers requires --encoder-plan-hash")?)?;
        let revision = args.encoder_revision.clone().context("--vision-peers requires --encoder-revision")?;
        anyhow::ensure!(!revision.is_empty(), "encoder revision must not be empty");
        Ok(Some(Self { addresses, plan_hash, revision }))
    }
}

pub(super) struct RemoteAudio {
    addresses: Vec<std::net::SocketAddr>,
    plan_hash: [u8; 32],
    revision: String,
    backend: String,
}
impl RemoteAudio {
    pub fn from_args(args: &super::serve::ServeArgs) -> Result<Option<Self>> {
        let Some(peers) = &args.audio_peers else {
            anyhow::ensure!(args.audio_encoder_plan_hash.is_none() && args.audio_encoder_revision.is_none()
                && args.audio_encoder_backend.is_none(), "audio encoder identity requires --audio-peers");
            return Ok(None);
        };
        anyhow::ensure!(matches!(args.audio, MediaMode::Auto | MediaMode::Spark(_)),
            "--audio-peers requires Spark/auto audio placement");
        let addresses = peers.split(',').map(|peer| peer.trim().parse()
            .context("audio peer must be an IP:port")).collect::<Result<Vec<_>>>()?;
        anyhow::ensure!((1..=6).contains(&addresses.len()), "audio needs 1..6 replicas");
        let plan_hash = crate::shared::vision::worker::parse_plan_hash(args.audio_encoder_plan_hash.as_deref()
            .context("--audio-peers requires --audio-encoder-plan-hash")?)?;
        let revision = args.audio_encoder_revision.clone().context("--audio-peers requires --audio-encoder-revision")?;
        let backend = args.audio_encoder_backend.clone().context("--audio-peers requires --audio-encoder-backend")?;
        let export = backend.split_once("/cute_aot_sm121/export").map(|(_, hash)| hash);
        anyhow::ensure!(!revision.is_empty() && backend.starts_with("mimo_audio_fp32_v1/cuda")
            && export.is_some_and(|hash| hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()))
            && backend.len() <= 256 && !backend.chars().any(char::is_control), "invalid SM121 audio backend/revision");
        Ok(Some(Self { addresses, plan_hash, revision, backend }))
    }
}

pub(super) struct ReadyVision {
    pub encoder: Encoder,
    pub preparer: Option<Arc<MediaPreparer>>,
    pub audio_preparer: Option<Arc<cuteafd_api::openai::media::audio::AudioPreparer>>,
    pub cache_bytes: usize,
}
impl ReadyVision {
    pub fn load(args: &super::EngineArgs, library: &cuteafd_ffi::NativeLibrary, mode: MediaMode,
        audio_mode: MediaMode, prefix: &super::serve::PrefixArgs, cache_bytes: Option<u64>,
        remote: Option<RemoteVision>, remote_audio: Option<RemoteAudio>) -> Result<(Option<Self>, super::serve::PrefixArgs)> {
        let config = vision_config(mode, &args.snapshot)?;
        let audio_on = audio_mode != MediaMode::Off;
        if config.is_none() && !audio_on { return Ok((None, prefix.clone())); }
        anyhow::ensure!(!matches!(mode, MediaMode::Spark(_)) || remote.is_some(), "Spark vision requires --vision-peers");
        anyhow::ensure!(!matches!(audio_mode, MediaMode::Spark(_)) || remote_audio.is_some(), "Spark audio requires --audio-peers");
        let processor = ProcessorConfig::from_snapshot(&args.snapshot, ImageFamily::Mimo)?;
        let model_config: serde_json::Value = serde_json::from_slice(&std::fs::read(args.snapshot.join("config.json"))?)?;
        let width = model_config["hidden_size"].as_u64().context("MiMo hidden_size")? as usize;
        let mut spec = config.as_ref().map(|_| crate::shared::vision::TowerSpec::mimo(&args.snapshot, 4096)).transpose()?;
        let mut audio_spec = audio_on.then(|| crate::shared::vision::audio::AudioTowerSpec::from_snapshot(&args.snapshot,
            cuteafd_loader::media::audio::MAX_CLIP_SAMPLES)).transpose()?;
        if let Some(spec) = &audio_spec {
            anyhow::ensure!(spec.plan().output_width() == width, "MiMo audio tower/LM output width mismatch");
        }
        let (prefix, cache_bytes) = prefix.with_media_headroom(cache_bytes)?;
        let mut image_encoder = Encoder::Off;
        let mut audio_encoder = Encoder::Off;
        let mut preparer = None;
        let mut audio_preparer = None;
        if let Some(remote) = remote {
            let spec = spec.take().context("remote vision requires checkpoint vision tower")?;
            let id = spec.encoder_id(&remote.revision, 121);
            preparer = Some(Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), id, 4)?));
            anyhow::ensure!(preparer.as_ref().unwrap().config().max_image_tokens <= 4096, "MiMo tower capacity is 4096 tokens per image");
            let expected = crate::shared::vision::remote::EncoderHandshake {
                encoder_id: id, max_patches: 4096 * processor.merge.pow(2), output_width: spec.native.output_width,
                patch_size: processor.patch, merge_size: processor.merge, plan_hash: remote.plan_hash,
            };
            anyhow::ensure!(expected.output_width as usize == width, "MiMo tower/LM output width mismatch");
            image_encoder = Encoder::Remote(crate::shared::vision::remote::RemoteEncoder::connect(remote.addresses, expected,
                std::time::Duration::from_secs(60))?);
        }
        // Both decoders share the request-level CPU admission semaphore even when
        // their GPU owners live on different hosts.
        let slots = preparer.as_ref().map(|p| p.slots.clone()).unwrap_or_else(|| Arc::new(tokio::sync::Semaphore::new(4)));
        if let Some(remote) = remote_audio {
            let spec = audio_spec.take().context("remote audio requires enabled checkpoint audio tower")?;
            let id = spec.plan().encoder_id(&remote.revision, 121, &remote.backend);
            let expected = crate::shared::vision::remote::AudioHandshake {
                encoder_id: id, plan_hash: remote.plan_hash, max_samples: spec.native().max_samples,
                output_width: spec.native().output_width,
            };
            audio_encoder = Encoder::Remote(crate::shared::vision::remote::RemoteEncoder::connect_audio(remote.addresses, expected,
                std::time::Duration::from_secs(60))?);
            audio_preparer = Some(Arc::new(cuteafd_api::openai::media::audio::AudioPreparer::new(id, slots.clone())));
        }
        let has_local_image = spec.is_some();
        let has_local_audio = audio_spec.is_some();
        let local = if has_local_image || has_local_audio {
            let local_mode = if has_local_image { mode } else { audio_mode };
            let gpu = match local_mode {
                MediaMode::Rtx(gpu) => gpu.map(i32::try_from).transpose()?.unwrap_or(args.device),
                MediaMode::Auto => {
                    anyhow::ensure!(args.peers.is_none() || args.local_experts,
                        "auto media with Spark experts requires planner-resolved encoder peers");
                    args.device
                },
                _ => anyhow::bail!("local media requires RTX placement"),
            };
            if has_local_image && has_local_audio {
                if let MediaMode::Rtx(Some(audio_gpu)) = audio_mode {
                    anyhow::ensure!(audio_gpu == gpu as usize, "local image/audio owner requires the same device");
                }
            }
            let info = library.cuda_device_info(gpu)?;
            let sm = u32::try_from(info.compute_capability_major * 10 + info.compute_capability_minor)?;
            anyhow::ensure!(!has_local_audio || sm == 120, "local MiMo audio requires SM120");
            let revision = args.snapshot.file_name().and_then(|v| v.to_str()).context("snapshot revision")?;
            if let Some(spec) = &spec {
                let mut image_preparer = MediaPreparer::for_loaded_encoder(processor.clone(), spec.encoder_id(revision, sm), 4)?;
                image_preparer.slots = slots.clone();
                preparer = Some(Arc::new(image_preparer));
            }
            let vision_bytes = spec.as_ref().map(|spec| cuteafd_ffi::vision::NativeVision::required(&args.native_lib, &spec.native)
                .map(|ledger| ledger.total_bytes())).transpose()?.unwrap_or(0);
            let audio = audio_spec.map(|spec| -> Result<_> {
                let backend = cuteafd_ffi::audio::NativeAudio::backend(&args.native_lib)?;
                audio_preparer = Some(Arc::new(cuteafd_api::openai::media::audio::AudioPreparer::new(
                    spec.plan().encoder_id(revision, sm, &backend), slots.clone())));
                let bytes = cuteafd_ffi::audio::NativeAudio::required(&args.native_lib, spec.native())?.total_bytes()?;
                Ok(crate::shared::vision::audio::AudioOwnerConfig { spec, admitted_bytes: bytes })
            }).transpose()?;
            let audio_bytes = audio.as_ref().map_or(0, |config| config.admitted_bytes);
            library.cuda_set_device(gpu)?;
            let loaded = (|| -> Result<_> {
                let (free, total) = library.cuda_memory_info()?;
                cuteafd_core::serving_capacity::admit_device_reservations(95,
                    cuteafd_core::serving_capacity::DeviceMemory { device: gpu as u32,
                        total_bytes: total as u64, baseline_free_bytes: free as u64 },
                    &[cuteafd_core::serving_capacity::MemoryReservation { name: "vision.resident_weights_scratch".into(), bytes: vision_bytes },
                      cuteafd_core::serving_capacity::MemoryReservation { name: "audio.resident_weights_scratch".into(), bytes: audio_bytes }])?;
                let service = match spec {
                    Some(spec) => crate::shared::vision::EncoderService::start_with_audio(spec, args.native_lib.clone(), gpu, vision_bytes, audio)?,
                    None => crate::shared::vision::EncoderService::start_audio(args.native_lib.clone(), gpu, audio.context("audio owner config")?)?,
                };
                Ok(Encoder::Local(crate::shared::vision::local::LocalEncoder::new(service, &processor, width, 4096)))
            })();
            let restored = library.cuda_set_device(args.device);
            let encoder = loaded?;
            restored?;
            tracing::info!(gpu, vision_bytes, audio_bytes, sm, "MiMo resident media encoder ready");
            Some(encoder)
        } else { None };
        let encoder = match (has_local_image, has_local_audio, local) {
            (true, true, Some(local)) => local,
            (true, false, Some(local)) => Encoder::split(local, audio_encoder),
            (false, true, Some(local)) => Encoder::split(image_encoder, local),
            (_, _, None) => Encoder::split(image_encoder, audio_encoder),
            _ => unreachable!(),
        };
        Ok((Some(Self { encoder, preparer, audio_preparer, cache_bytes }), prefix))
    }
}

pub(super) enum Encoder {
    Local(crate::shared::vision::local::LocalEncoder),
    Remote(crate::shared::vision::remote::RemoteEncoder),
    Split { image: Box<Encoder>, audio: Box<Encoder> },
    #[cfg(test)]
    Fixture(cuteafd_engine::media::FakeEncoder, Arc<std::sync::atomic::AtomicBool>),
    Off,
}
impl Encoder {
    fn split(image: Self, audio: Self) -> Self {
        Self::Split { image: Box::new(image), audio: Box::new(audio) }
    }
    pub fn health_handle(&self, audio_job: bool) -> Option<Arc<std::sync::atomic::AtomicBool>> {
        match self {
            Self::Remote(client) => Some(client.health_handle()), Self::Local(client) => Some(client.health_handle()),
            Self::Split { image, audio } => if audio_job { audio } else { image }.health_handle(audio_job),
            #[cfg(test)]
            Self::Fixture(_, health) => Some(health.clone()),
            Self::Off => None,
        }
    }
    pub fn available_for(&self, audio_job: bool) -> bool {
        match self {
            Self::Remote(client) => client.healthy(), Self::Local(client) => client.healthy(),
            Self::Split { image, audio } => if audio_job { audio } else { image }.available_for(audio_job),
            #[cfg(test)]
            Self::Fixture(_, health) => health.load(std::sync::atomic::Ordering::Acquire),
            Self::Off => false,
        }
    }
}
impl cuteafd_engine::media::EncoderClient for Encoder {
    fn submit(&mut self, job: EncodeJob) -> std::result::Result<cuteafd_engine::media::EncoderTicket, cuteafd_engine::media::MediaError> {
        use cuteafd_engine::media::MediaError;
        match self {
            Self::Local(client) => client.submit(job), Self::Remote(client) => client.submit(job),
            Self::Split { image, audio } => {
                let is_audio = job.key.is_audio();
                let owner = if is_audio { audio } else { image };
                let ticket = owner.submit(job)?;
                // Each owner starts at ticket zero. Reserve the low bit for modality.
                match ticket.0.checked_mul(2).and_then(|id| id.checked_add(u64::from(is_audio))) {
                    Some(id) => Ok(cuteafd_engine::media::EncoderTicket(id)),
                    None => { owner.cancel(ticket); Err(MediaError::QueueFull) },
                }
            },
            #[cfg(test)]
            Self::Fixture(client, _) => client.submit(job),
            Self::Off => Err(MediaError::Encoder("encoder not loaded".into())),
        }
    }
    fn poll(&mut self, ticket: cuteafd_engine::media::EncoderTicket) -> Option<std::result::Result<cuteafd_engine::media::EncodeOutput, cuteafd_engine::media::MediaError>> {
        match self {
            Self::Local(client) => client.poll(ticket), Self::Remote(client) => client.poll(ticket),
            Self::Split { image, audio } => if ticket.0 & 1 == 1 { audio } else { image }
                .poll(cuteafd_engine::media::EncoderTicket(ticket.0 / 2)),
            #[cfg(test)]
            Self::Fixture(client, _) => client.poll(ticket),
            Self::Off => None,
        }
    }
    fn cancel(&mut self, ticket: cuteafd_engine::media::EncoderTicket) {
        match self {
            Self::Local(client) => client.cancel(ticket), Self::Remote(client) => client.cancel(ticket),
            Self::Split { image, audio } => if ticket.0 & 1 == 1 { audio } else { image }
                .cancel(cuteafd_engine::media::EncoderTicket(ticket.0 / 2)),
            #[cfg(test)]
            Self::Fixture(client, _) => client.cancel(ticket),
            Self::Off => (),
        }
    }
}
pub(super) fn failure(error: cuteafd_engine::media::MediaError) -> cuteafd_api::openai::NativeFailure {
    match error {
        cuteafd_engine::media::MediaError::CacheFull { .. } | cuteafd_engine::media::MediaError::QueueFull =>
            cuteafd_api::openai::NativeFailure::Unavailable(error.to_string()),
        cuteafd_engine::media::MediaError::Encoder(_) => cuteafd_api::openai::NativeFailure::Unavailable("vision encoder unavailable".into()),
        error => cuteafd_api::openai::NativeFailure::BadRequest(error.to_string()),
    }
}

pub(super) struct Prompt {
    pub job: NativeRequest,
    pub tokens: Vec<u32>,
    pub keys: MediaKeys,
}
pub(super) fn prepare(job: NativeRequest, tokens: Vec<u32>, config: &serde_json::Value,
    vocabulary: usize, hidden: usize, max_context: usize) -> Result<(Prompt, RequestMedia, Vec<EncodeJob>)> {
    anyhow::ensure!(job.audio.is_empty() || !job.probe.as_ref().is_some_and(|p| !p.spec.media.is_empty()),
        "audio cannot be combined with expanded image probes");
    let (tokens, spans) = if config.get("vision_config").is_some() || config.get("audio_token_id").is_some() {
        let expander = if config.get("vision_config").is_some() { SpanExpander::from_config(config, vocabulary as u32)? }
            else { SpanExpander::from_audio_config(config, vocabulary as u32)? };
        let images = job.media.iter().map(|image| image.as_ref().clone()).collect::<Vec<_>>();
        if job.probe.as_ref().is_some_and(|p| !p.spec.audio.is_empty()) {
            let probe = job.probe.as_ref().unwrap();
            probe.spec.validate_audio()?;
            anyhow::ensure!(images.is_empty() && probe.spec.audio.len() == job.audio.len() && tokens.len() <= max_context,
                "expanded audio probe count/context differs");
            anyhow::ensure!(tokens.iter().all(|&id| id < expander.vocabulary), "probe token outside vocabulary");
            let marker = |name| config.get(name).and_then(serde_json::Value::as_u64)
                .and_then(|id| u32::try_from(id).ok()).context("audio probe marker missing");
            let (start, pad, finish) = (marker("audio_start_token_id")?, marker("audio_token_id")?, marker("audio_end_token_id")?);
            let mut spans = Vec::with_capacity(job.audio.len());
            for (span, clip) in probe.spec.audio.iter().zip(&job.audio) {
                let end = span.start.checked_add(span.len).context("audio probe extent")?;
                anyhow::ensure!(span.len == clip.geometry.tokens && span.samples == clip.pcm.len()
                    && span.key == key_hex(clip.key) && span.pcm_sha256 == pcm_hash(&clip.pcm),
                    "probe prepared audio identity differs");
                anyhow::ensure!(span.start > 0 && end < tokens.len() && tokens[span.start - 1] == start
                    && tokens[end] == finish && tokens[span.start..end].iter().all(|&id| id == pad),
                    "probe audio rows/marker boundaries differ");
                spans.push(cuteafd_loader::media::MediaSpan { start: span.start, len: span.len, key: clip.key.into() });
            }
            anyhow::ensure!(tokens.iter().filter(|&&id| id == pad).count() == spans.iter().map(|s| s.len).sum::<usize>(),
                "unbound probe audio placeholders");
            (tokens, spans)
        } else if job.probe.as_ref().is_some_and(|p| p.spec.prompt_ids.is_some() && !images.is_empty()) {
            let probe = job.probe.as_ref().unwrap();
            probe.spec.validate_media()?;
            anyhow::ensure!(probe.spec.media.len() == images.len() && tokens.len() <= max_context,
                "expanded probe media count/context differs");
            anyhow::ensure!(tokens.iter().all(|&id| id < expander.vocabulary), "probe token outside vocabulary");
            let mut spans = Vec::with_capacity(images.len());
            for (span, image) in probe.spec.media.iter().zip(&images) {
                let end = span.start.checked_add(span.len).context("probe media extent")?;
                anyhow::ensure!(span.len == image.tokens && span.grid == [image.grid.t, image.grid.h, image.grid.w]
                    && span.key == key_hex(image.key), "probe prepared image identity differs");
                anyhow::ensure!(span.start > 0 && end < tokens.len() && tokens[span.start - 1] == expander.start
                    && tokens[end] == expander.end && tokens[span.start..end].iter().all(|&id| id == expander.placeholder),
                    "probe image rows/marker boundaries differ");
                spans.push(cuteafd_loader::media::MediaSpan { start: span.start, len: span.len, key: image.key.into() });
            }
            anyhow::ensure!(tokens.iter().filter(|&&id| id == expander.placeholder).count()
                == spans.iter().map(|span| span.len).sum::<usize>(), "unbound probe image placeholders");
            (tokens, spans)
        } else {
            let audio = job.audio.iter().map(|clip| clip.as_ref().clone()).collect::<Vec<_>>();
            let expanded = SpanExpander::expand_media(config, vocabulary as u32, &tokens, &images, &audio, max_context)?;
            (expanded.tokens, expanded.media)
        }
    } else {
        anyhow::ensure!(job.media.is_empty() && job.audio.is_empty(), "checkpoint has no media tower");
        (tokens, Vec::new())
    };
    if let Some(probe) = &job.probe {
        let image_spans = spans.iter().filter(|span| !span.key.is_audio()).collect::<Vec<_>>();
        anyhow::ensure!(image_spans.len() == job.media.len(), "probe image count differs");
        let echo = image_spans.into_iter().zip(&job.media).enumerate().map(|(i, (span, image))| {
            cuteafd_api::openai::probe::ProbeMedia { start: span.start, len: span.len, kind: "image".into(),
                key: key_hex(span.key), grid: [image.grid.t, image.grid.h, image.grid.w],
                fixture: probe.spec.media.get(i).and_then(|s| s.fixture.clone()), image_url: None }
        }).collect();
        probe.media(echo);
        let audio_spans = spans.iter().filter(|span| span.key.is_audio()).collect::<Vec<_>>();
        anyhow::ensure!(audio_spans.len() == job.audio.len(), "probe audio count differs");
        probe.audio(audio_spans.into_iter().zip(&job.audio).map(|(span, clip)| {
            cuteafd_api::openai::probe::ProbeAudio { start: span.start, len: span.len, key: key_hex(span.key),
                samples: clip.pcm.len(), pcm_sha256: pcm_hash(&clip.pcm) }
        }).collect());
    }
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let keys = MediaKeys::new(&tokens, vocabulary as u32, &spans)?;
    let jobs = job.media.iter().map(|image| EncodeJob::image(image.key, [image.grid.t, image.grid.h, image.grid.w], image.rgb8.clone(), image.tokens, hidden))
        .chain(job.audio.iter().map(|clip| EncodeJob::audio(clip.key, clip.pcm.clone(), clip.geometry.tokens, hidden))).collect();
    Ok((Prompt { job, tokens, keys }, media, jobs))
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureMetadata {
    schema: String,
    key: String,
    grid: [u32; 3],
    shape: [usize; 2],
    dtype: String,
    sha256: String,
    tower_dtype: String,
    fixture_sha256: String,
    snapshot_identity: serde_json::Value,
}

fn snapshot_identity(snapshot: &std::path::Path) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let mut identity = serde_json::json!({"snapshot_revision": snapshot.file_name().and_then(|s| s.to_str())
        .context("snapshot revision")?});
    for (key, file) in [("config", "config.json"), ("tokenizer", "tokenizer.json"),
        ("modeling", "modeling_mimo_v2.py"), ("preprocessor", "preprocessor_config.json")] {
        identity[format!("{key}_sha256")] = format!("{:x}", Sha256::digest(std::fs::read(snapshot.join(file))?)).into();
    }
    Ok(identity)
}

fn read_bounded(path: &std::path::Path, limit: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    anyhow::ensure!(file.metadata()?.len() <= limit as u64, "feature file exceeds bound");
    let mut bytes = Vec::new(); file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() <= limit, "feature file exceeds bound");
    Ok(bytes)
}

fn feature_payload(path: &std::path::Path, bytes: usize, sha256: &str, cached: Option<&Arc<[u8]>>) -> Result<Arc<[u8]>> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    if let Some(cached) = cached {
        // Revalidate disk bytes without allocating a second feature-sized payload on a warm hit.
        let mut file = std::fs::File::open(path)?;
        anyhow::ensure!(file.metadata()?.len() == bytes as u64, "reference feature payload length differs");
        let mut hash = Sha256::new(); let mut offset = 0usize; let mut buffer = [0; 8192];
        loop {
            let n = file.read(&mut buffer)?; if n == 0 { break; }
            anyhow::ensure!(offset.checked_add(n).is_some_and(|end| end <= bytes)
                && cached.get(offset..offset + n) == Some(&buffer[..n]), "reference feature payload differs");
            hash.update(&buffer[..n]); offset += n;
        }
        anyhow::ensure!(offset == bytes && format!("{:x}", hash.finalize()) == sha256, "reference feature payload hash differs");
        return Ok(cached.clone());
    }
    let payload = read_bounded(path, bytes)?;
    anyhow::ensure!(payload.len() == bytes && format!("{:x}", Sha256::digest(&payload)) == sha256,
        "reference feature payload length/hash differs");
    anyhow::ensure!(payload.chunks_exact(2).all(|b| (u16::from_le_bytes([b[0], b[1]]) & 0x7f80) != 0x7f80),
        "reference features contain nonfinite BF16 values");
    Ok(Arc::from(payload))
}

/// Strict paired-G4 hook. Only scoring probes bypass the encoder; ordinary
/// generation remains native. Separate cache identities cannot warm native images.
pub(super) fn probe_features(prompt: &Prompt, media: &mut RequestMedia,
    cache: &mut cuteafd_engine::media::EmbeddingCache, snapshot: &std::path::Path) -> Result<()> {
    let Some(probe) = prompt.job.probe.as_ref().filter(|p| p.spec.score_from.is_some() && (!p.spec.media.is_empty() || !p.spec.audio.is_empty())) else {
        return Ok(());
    };
    let audio = !probe.spec.audio.is_empty();
    let variable = if audio { "CUTEAFD_AUDIO_FEATURES_DIR" } else { "CUTEAFD_MEDIA_FEATURES_DIR" };
    let Some(root) = std::env::var_os(variable) else { return Ok(()); };
    anyhow::ensure!(probe.spec.cold && probe.spec.no_speculation, "feature probes require explicit cold/no_speculation");
    probe.spec.validate_media()?;
    let root = std::path::PathBuf::from(root);
    let identity = snapshot_identity(snapshot)?;
    if audio { apply_audio_probe_features(prompt, media, cache, &root, &identity) }
    else { apply_probe_features(prompt, media, cache, &root, &identity) }
}

fn apply_probe_features(prompt: &Prompt, media: &mut RequestMedia,
    cache: &mut cuteafd_engine::media::EmbeddingCache, root: &std::path::Path, identity: &serde_json::Value) -> Result<()> {
    use sha2::{Digest, Sha256};
    let probe = prompt.job.probe.as_ref().context("features require a probe")?;
    anyhow::ensure!(probe.spec.cold && probe.spec.no_speculation && probe.spec.score_from.is_some()
        && !probe.spec.media.is_empty(), "features require a cold, speculation-free media scoring probe");
    let root = root.canonicalize()?;
    let mut provenance = Vec::new();
    for span in &probe.spec.media {
        let fixture = span.fixture.as_ref().context("feature probes require fixture identity")?;
        anyhow::ensure!(cuteafd_api::openai::probe::sha256_hex(&span.key), "invalid feature key");
        let metadata_path = root.join(format!("{}.json", span.key)).canonicalize()?;
        let payload_path = root.join(format!("{}.bf16", span.key)).canonicalize()?;
        anyhow::ensure!(metadata_path.starts_with(&root) && payload_path.starts_with(&root), "feature path escapes root");
        let raw = read_bounded(&metadata_path, 64 << 10)?;
        let meta: FeatureMetadata = serde_json::from_slice(&raw)?;
        anyhow::ensure!(meta.schema == "cuteafd.media.features/1" && meta.key == span.key && meta.grid == span.grid
            && meta.shape == [span.len, media.row_bytes() / 2] && meta.dtype == "bf16-le"
            && matches!(meta.tower_dtype.as_str(), "bf16" | "fp32")
            && cuteafd_api::openai::probe::sha256_hex(&meta.sha256)
            && meta.fixture_sha256 == fixture.sha256 && meta.snapshot_identity == *identity,
            "reference feature metadata differs from prepared request/snapshot");
        let bytes = span.len.checked_mul(media.row_bytes()).context("feature byte extent")?;
        let mut hash = Sha256::new();
        hash.update(b"cuteafd.probe.feature_override/1\0"); hash.update(span.key.as_bytes()); hash.update(&raw);
        let override_key = cuteafd_loader::media::ImageKey(hash.finalize().into());
        let image = prompt.job.media.iter().find(|i| key_hex(i.key) == span.key).context("feature span has no prepared image")?;
        // Admission precedes payload allocation, including on a warm override-cache hit.
        let pin = cache.reserve(override_key, bytes)?;
        let payload = feature_payload(&payload_path, bytes, &meta.sha256, pin.features())?;
        let lease = cache.complete(override_key, payload)?;
        media.attach_probe_override(image.key, lease)?; drop(pin);
        provenance.push(serde_json::from_slice::<serde_json::Value>(&raw)?);
    }
    probe.provenance(serde_json::json!({"mode": "reference_features", "probe_only": true,
        "encoder_bypassed": true, "features": provenance}));
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AudioFeatureMetadata {
    schema: String,
    key: String,
    samples: usize,
    pcm_sha256: String,
    codes_sha256: String,
    shape: [usize; 2],
    dtype: String,
    sha256: String,
    tower_dtype: String,
    snapshot_identity: serde_json::Value,
}

fn apply_audio_probe_features(prompt: &Prompt, media: &mut RequestMedia,
    cache: &mut cuteafd_engine::media::EmbeddingCache, root: &std::path::Path, identity: &serde_json::Value) -> Result<()> {
    use sha2::{Digest, Sha256};
    let probe = prompt.job.probe.as_ref().context("audio features require a probe")?;
    anyhow::ensure!(probe.spec.cold && probe.spec.no_speculation && probe.spec.score_from.is_some()
        && !probe.spec.audio.is_empty() && probe.spec.media.is_empty(),
        "audio features require a cold, speculation-free audio scoring probe");
    probe.spec.validate_audio()?;
    let root = root.canonicalize()?;
    let mut provenance = Vec::new();
    for span in &probe.spec.audio {
        let metadata_path = root.join(format!("{}.json", span.key)).canonicalize()?;
        let payload_path = root.join(format!("{}.bf16", span.key)).canonicalize()?;
        let codes_path = root.join(format!("{}.codes.i64", span.key)).canonicalize()?;
        anyhow::ensure!([&metadata_path, &payload_path, &codes_path].iter().all(|p| p.starts_with(&root)),
            "audio feature path escapes root");
        let raw = read_bounded(&metadata_path, 64 << 10)?;
        let meta: AudioFeatureMetadata = serde_json::from_slice(&raw)?;
        anyhow::ensure!(meta.schema == "cuteafd.audio.features/1" && meta.key == span.key && meta.samples == span.samples
            && meta.pcm_sha256 == span.pcm_sha256 && meta.shape == [span.len, media.row_bytes() / 2]
            && meta.dtype == "bf16-le" && meta.tower_dtype == "fp32"
            && cuteafd_api::openai::probe::sha256_hex(&meta.sha256)
            && cuteafd_api::openai::probe::sha256_hex(&meta.codes_sha256) && meta.snapshot_identity == *identity,
            "audio reference feature metadata differs from prepared request/snapshot");
        let clip = prompt.job.audio.iter().find(|c| key_hex(c.key) == span.key).context("feature span has no prepared audio")?;
        anyhow::ensure!(span.samples == clip.pcm.len() && span.len == clip.geometry.tokens && pcm_hash(&clip.pcm) == span.pcm_sha256,
            "audio reference PCM identity differs");
        let codes = read_bounded(&codes_path, clip.geometry.codes.checked_mul(20 * 8).context("audio code extent")?)?;
        anyhow::ensure!(codes.len() == clip.geometry.codes * 20 * 8 && format!("{:x}", Sha256::digest(&codes)) == meta.codes_sha256
            && codes.chunks_exact(8).all(|b| (0..1024).contains(&i64::from_le_bytes(b.try_into().unwrap()))),
            "audio reference codes differ");
        let bytes = span.len.checked_mul(media.row_bytes()).context("audio feature extent")?;
        let mut hash = Sha256::new();
        hash.update(b"cuteafd.probe.audio_feature_override/1\0"); hash.update(span.key.as_bytes()); hash.update(&raw);
        let override_key = cuteafd_core::AudioKey(hash.finalize().into());
        let pin = cache.reserve(override_key, bytes)?;
        let payload = feature_payload(&payload_path, bytes, &meta.sha256, pin.features())?;
        let lease = cache.complete(override_key, payload)?;
        media.attach_probe_override(clip.key, lease)?; drop(pin);
        provenance.push(serde_json::from_slice::<serde_json::Value>(&raw)?);
    }
    probe.provenance(serde_json::json!({"mode": "audio_reference_features", "probe_only": true,
        "encoder_bypassed": true, "features": provenance}));
    Ok(())
}

fn pcm_hash(pcm: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for value in pcm { hash.update(value.to_le_bytes()); }
    format!("{:x}", hash.finalize())
}

fn key_hex(key: impl Into<cuteafd_core::MediaKey>) -> String {
    key.into().bytes().iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::TargetSamplingParams;
    use cuteafd_loader::media::{ImageGrid, ImageKey, PreparedImage};

    #[test]
    fn split_owner_tickets_cancel_poll_and_health_are_modality_scoped() {
        use cuteafd_engine::media::{EncoderClient, FakeEncoder};
        use std::sync::atomic::{AtomicBool, Ordering};
        let image_health = Arc::new(AtomicBool::new(true));
        let audio_health = Arc::new(AtomicBool::new(true));
        let mut encoder = Encoder::split(Encoder::Fixture(FakeEncoder::default(), image_health.clone()),
            Encoder::Fixture(FakeEncoder::default(), audio_health.clone()));
        let image = EncodeJob::image(ImageKey([7;32]), [1,4,4], Arc::from(vec![0;16*768]), 4, 4096);
        let clip = cuteafd_loader::media::audio::prepare_pcm(vec![0.;24000], cuteafd_loader::media::EncoderId([2;32])).unwrap();
        let audio = EncodeJob::audio(clip.key, clip.pcm, clip.geometry.tokens, 4096);
        let a = encoder.submit(audio.clone()).unwrap();
        let i = encoder.submit(image.clone()).unwrap();
        assert_ne!(a, i); assert_eq!((a.0, i.0), (1, 0));
        assert_eq!(encoder.poll(i).unwrap().unwrap().features, FakeEncoder::features(&image).unwrap());
        assert_eq!(encoder.poll(a).unwrap().unwrap().features, FakeEncoder::features(&audio).unwrap());
        let a = encoder.submit(audio).unwrap(); let i = encoder.submit(image).unwrap();
        encoder.cancel(a); assert!(encoder.poll(a).is_none());
        assert!(encoder.poll(i).unwrap().is_ok());
        audio_health.store(false, Ordering::Release);
        assert!(encoder.available_for(false)); assert!(!encoder.available_for(true));
        assert!(Arc::ptr_eq(&encoder.health_handle(false).unwrap(), &image_health));
        assert!(Arc::ptr_eq(&encoder.health_handle(true).unwrap(), &audio_health));
    }
    #[test]
    fn remote_audio_requires_complete_sm121_identity_and_enabled_placement() {
        use clap::Parser;
        let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-mimo", "--snapshot", "/not-read", "--native-lib", "/not-read.so"]).unwrap();
        let crate::cli::Commands::ServeMimo(mut args) = cli.command else { panic!("MiMo command") };
        args.audio = MediaMode::Spark(Some(0));
        assert!(RemoteAudio::from_args(&args).unwrap().is_none());
        args.audio_peers = Some("127.0.0.1:9300".into());
        args.audio_encoder_plan_hash = Some("ab".repeat(32));
        args.audio_encoder_revision = Some("revision".into());
        assert!(RemoteAudio::from_args(&args).is_err());
        args.audio_encoder_backend = Some(format!("mimo_audio_fp32_v1/cuda13000/cufft12000/cublas13.0.0/cute_aot_sm121/export{}", "cd".repeat(32)));
        assert_eq!(RemoteAudio::from_args(&args).unwrap().unwrap().plan_hash, [0xab;32]);
        for mode in [MediaMode::Off, MediaMode::Rtx(None)] {
            args.audio = mode; assert!(RemoteAudio::from_args(&args).is_err());
        }
        args.audio = MediaMode::Auto;
        args.audio_encoder_backend = Some("cute_aot_sm120/exportbad".into());
        assert!(RemoteAudio::from_args(&args).is_err());
    }
    #[test]
    fn remote_options_require_complete_identity_and_valid_placement() {
        use clap::Parser;
        let make = |options: Vec<String>| {
            let mut argv = vec!["cuteafd".into(), "serve-mimo".into(), "--snapshot".into(), "/not-read".into(),
                "--native-lib".into(), "/not-read.so".into()];
            argv.extend(options);
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Commands::ServeMimo(mut args) = cli.command else { panic!("MiMo command") };
            args.vision = MediaMode::Spark(Some(0));
            args
        };
        assert!(RemoteVision::from_args(&make(vec![])).unwrap().is_none());
        let complete = vec!["--vision-peers".into(), "127.0.0.1:1234,127.0.0.2:1234".into(),
            "--encoder-plan-hash".into(), "ab".repeat(32), "--encoder-revision".into(), "revision".into()];
        let args = make(complete.clone());
        let remote = RemoteVision::from_args(&args).unwrap().unwrap();
        assert_eq!(remote.addresses.len(), 2); assert_eq!(remote.plan_hash, [0xab;32]);
        for bad in ["", "hostname:1234", "127.0.0.1", "127.0.0.1:1234,"] {
            let mut args = make(complete.clone()); args.vision_peers = Some(bad.into());
            assert!(RemoteVision::from_args(&args).is_err());
        }
        let mut args = make(complete.clone()); args.encoder_plan_hash = None;
        assert!(RemoteVision::from_args(&args).is_err());
        let mut args = make(complete.clone()); args.encoder_revision = None;
        assert!(RemoteVision::from_args(&args).is_err());
        let mut args = make(complete); args.vision = MediaMode::Rtx(None);
        assert!(RemoteVision::from_args(&args).is_err());
        assert!(matches!(failure(cuteafd_engine::media::MediaError::Encoder("socket died".into())),
            cuteafd_api::openai::NativeFailure::Unavailable(message) if message == "vision encoder unavailable"));
    }
    fn request(images: Vec<Arc<PreparedImage>>) -> NativeRequest {
        NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), media: images, audio: Vec::new(),
            max_tokens: 8, sampling: TargetSamplingParams::default(), stop_token_ids: Vec::new(),
            events: tokio::sync::mpsc::unbounded_channel().0, usage: None, probe: None }
    }
    fn config() -> serde_json::Value {
        serde_json::json!({"vision_config": {}, "image_token_id": 5,
            "vision_start_token_id": 4, "vision_end_token_id": 6})
    }
    fn image() -> Arc<PreparedImage> {
        Arc::new(PreparedImage { key: ImageKey([7; 32]), grid: ImageGrid { t: 1, h: 4, w: 4 },
            rgb8: Arc::from(vec![0; 4 * 4 * 768]), tokens: 4 })
    }
    #[test]
    fn mixed_image_audio_jobs_and_span_identities_remain_in_prompt_order() {
        let mut cfg = config();
        cfg["audio_start_token_id"] = 7.into(); cfg["audio_token_id"] = 8.into(); cfg["audio_end_token_id"] = 9.into();
        let clip = Arc::new(cuteafd_loader::media::audio::prepare_pcm(vec![0.;24000],
            cuteafd_loader::media::EncoderId([2;32])).unwrap());
        let mut job = request(vec![image()]); job.audio = vec![clip.clone()];
        let (prompt, media, jobs) = prepare(job, vec![1,7,8,9,4,5,6,2], &cfg, 32, 4096, 100).unwrap();
        assert_eq!(prompt.tokens, [1,7,8,8,8,8,8,8,8,9,4,5,5,5,5,6,2]);
        assert_eq!(media.spans().iter().map(|span| (span.start, span.len)).collect::<Vec<_>>(), [(2,7),(11,4)]);
        assert_eq!(media.spans()[0].key, clip.key.into());
        assert_eq!(media.spans()[1].key, image().key.into());
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().any(|job| job.key == clip.key.into() && job.tokens == 7));
        assert!(jobs.iter().any(|job| job.key == image().key.into() && job.tokens == 4));
    }
    fn audio_feature_request() -> (Prompt, RequestMedia, serde_json::Value) {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec, ProbeAudio};
        let clip = Arc::new(cuteafd_loader::media::audio::prepare_pcm(vec![0.; 24000],
            cuteafd_loader::media::EncoderId([2;32])).unwrap());
        let tokens = vec![1,7,8,8,8,8,8,8,8,9,2];
        let spec = ProbeSpec { prompt_ids: Some(tokens.clone()), cold: true, no_speculation: true, score_from: Some(10),
            audio: vec![ProbeAudio { start: 2, len: 7, key: key_hex(clip.key), samples: 24000, pcm_sha256: pcm_hash(&clip.pcm) }],
            ..Default::default() };
        let mut job = request(Vec::new()); job.audio = vec![clip]; job.probe = Some(Probe::new(spec));
        let cfg = serde_json::json!({"audio_start_token_id":7,"audio_token_id":8,"audio_end_token_id":9});
        let (prompt, media, _) = prepare(job, tokens, &cfg, 32, 2, 32).unwrap();
        (prompt, media, serde_json::json!({"snapshot_revision":"test"}))
    }
    fn audio_feature_files(root: &std::path::Path, prompt: &Prompt, identity: &serde_json::Value, payload: &[u8]) -> serde_json::Value {
        use sha2::{Digest, Sha256};
        let span = &prompt.job.probe.as_ref().unwrap().spec.audio[0];
        let codes = vec![0u8; 26 * 20 * 8];
        let meta = serde_json::json!({"schema":"cuteafd.audio.features/1", "key":span.key,"samples":span.samples,
            "pcm_sha256":span.pcm_sha256,"codes_sha256":format!("{:x}",Sha256::digest(&codes)),"shape":[7,2],
            "dtype":"bf16-le","sha256":format!("{:x}",Sha256::digest(payload)),"tower_dtype":"fp32","snapshot_identity":identity});
        for (suffix, bytes) in [("json", serde_json::to_vec(&meta).unwrap()), ("bf16", payload.to_vec()), ("codes.i64", codes)] {
            std::fs::write(root.join(format!("{}.{suffix}", span.key)), bytes).unwrap();
        }
        meta
    }
    #[test]
    fn audio_probe_binds_expanded_rows_pcm_markers_and_echo() {
        use cuteafd_api::openai::probe::Probe;
        let (prompt, _, _) = audio_feature_request();
        assert_eq!(prompt.job.probe.as_ref().unwrap().record().audio, prompt.job.probe.as_ref().unwrap().spec.audio);
        for mutation in ["key", "pcm", "samples", "length", "start_marker", "end_marker", "pad", "extra_pad", "vocabulary", "sources"] {
            let (mut prompt, _, _) = audio_feature_request();
            let mut spec = prompt.job.probe.as_ref().unwrap().spec.clone();
            match mutation {
                "key" => spec.audio[0].key = "ab".repeat(32), "pcm" => spec.audio[0].pcm_sha256 = "ab".repeat(32),
                "samples" => spec.audio[0].samples = 24001, "length" => spec.audio[0].len = 6,
                "start_marker" => prompt.tokens[1] = 1, "end_marker" => prompt.tokens[9] = 1,
                "pad" => prompt.tokens[2] = 1, "extra_pad" => prompt.tokens[0] = 8,
                "vocabulary" => prompt.tokens[0] = 32, _ => prompt.job.audio.clear(),
            }
            spec.prompt_ids = Some(prompt.tokens.clone()); prompt.job.probe = Some(Probe::new(spec));
            let cfg = serde_json::json!({"audio_start_token_id":7,"audio_token_id":8,"audio_end_token_id":9});
            assert!(prepare(prompt.job, prompt.tokens, &cfg, 32, 2, 32).is_err(), "{mutation}");
        }
    }
    #[test]
    fn audio_override_is_budgeted_isolated_and_bypasses_native_owner() {
        use cuteafd_engine::media::{EmbeddingCache, MediaAdmission, MediaWaiter, MediaPoll};
        let root = tempfile::tempdir().unwrap(); let (prompt, mut media, identity) = audio_feature_request();
        audio_feature_files(root.path(), &prompt, &identity, &[0;28]);
        let key = prompt.job.audio[0].key;
        let mut cache = EmbeddingCache::new(28);
        apply_audio_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).unwrap();
        assert!(media.ready(0, prompt.tokens.len())); assert!(!cache.contains(key)); assert_eq!(cache.bytes(), 28);
        assert_eq!(media.spans()[0].key, key.into());
        let clip = &prompt.job.audio[0];
        let jobs = vec![EncodeJob::audio(clip.key, clip.pcm.clone(), 7, 2)];
        let waiter = MediaWaiter::new(prompt, media, jobs, 0).unwrap();
        let mut admission = MediaAdmission::new(cache, Encoder::Off, 1);
        assert!(admission.enqueue(waiter).is_ok()); assert!(matches!(admission.poll(|_| false), MediaPoll::Ready(_)));
        assert!(!admission.cache.contains(key)); assert_eq!(admission.stats(0,0).encodes,0);
    }
    #[test]
    fn audio_override_rejects_bad_provenance_payloads_and_non_scoring_requests() {
        use cuteafd_api::openai::probe::Probe;
        for field in ["schema","key","samples","pcm_sha256","codes_sha256","shape","dtype","sha256","tower_dtype","snapshot_identity","unknown"] {
            let root = tempfile::tempdir().unwrap(); let (prompt, mut media, identity) = audio_feature_request();
            let mut meta = audio_feature_files(root.path(), &prompt, &identity, &[0;28]); meta[field] = "wrong".into();
            let key = &prompt.job.probe.as_ref().unwrap().spec.audio[0].key;
            std::fs::write(root.path().join(format!("{key}.json")), serde_json::to_vec(&meta).unwrap()).unwrap();
            let mut cache = cuteafd_engine::media::EmbeddingCache::new(28);
            assert!(apply_audio_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err(),"{field}");
            assert_eq!(cache.bytes(),0);
        }
        for field in ["cold","speculation","score"] {
            let root = tempfile::tempdir().unwrap(); let (mut prompt, mut media, identity) = audio_feature_request();
            audio_feature_files(root.path(), &prompt, &identity, &[0;28]);
            let mut spec = prompt.job.probe.as_ref().unwrap().spec.clone();
            match field { "cold" => spec.cold = false, "speculation" => spec.no_speculation = false, _ => spec.score_from = None }
            prompt.job.probe = Some(Probe::new(spec)); let mut cache = cuteafd_engine::media::EmbeddingCache::new(28);
            assert!(apply_audio_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err());
        }
        for mode in ["missing", "escape", "codes", "payload"] {
            let root = tempfile::tempdir().unwrap(); let outside = tempfile::tempdir().unwrap();
            let (prompt, mut media, identity) = audio_feature_request();
            audio_feature_files(root.path(), &prompt, &identity, &[0;28]);
            let key = &prompt.job.probe.as_ref().unwrap().spec.audio[0].key;
            match mode {
                "missing" => std::fs::remove_file(root.path().join(format!("{key}.codes.i64"))).unwrap(),
                "codes" => std::fs::write(root.path().join(format!("{key}.codes.i64")), [1;8]).unwrap(),
                "payload" => std::fs::write(root.path().join(format!("{key}.bf16")), [1;28]).unwrap(),
                _ => {
                    let path = root.path().join(format!("{key}.bf16")); std::fs::remove_file(&path).unwrap();
                    std::fs::write(outside.path().join("rows"), [0;28]).unwrap();
                    #[cfg(unix)] std::os::unix::fs::symlink(outside.path().join("rows"), path).unwrap();
                }
            }
            let mut cache = cuteafd_engine::media::EmbeddingCache::new(28);
            assert!(apply_audio_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err(),"{mode}");
            drop(media); cache.prune_reservations(); assert_eq!(cache.bytes(),0);
        }
        for (payload,budget) in [(vec![0;27],28),(vec![0;29],28),([0x80,0x7f].repeat(14),28),(vec![0;28],27)] {
            let root = tempfile::tempdir().unwrap(); let (prompt, mut media, identity) = audio_feature_request();
            audio_feature_files(root.path(), &prompt, &identity, &payload);
            let mut cache = cuteafd_engine::media::EmbeddingCache::new(budget);
            assert!(apply_audio_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err());
            drop(media); cache.prune_reservations(); assert_eq!(cache.bytes(),0);
        }
    }
    fn feature_request() -> (Prompt, RequestMedia, serde_json::Value) {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec, ProbeMedia, ProbeFixture, ProbeImageUrl};
        let tokens = vec![1, 4, 5, 5, 5, 5, 6, 2];
        let spec = ProbeSpec { prompt_ids: Some(tokens.clone()), cold: true, no_speculation: true, score_from: Some(7),
            media: vec![ProbeMedia { start: 2, len: 4, kind: "image".into(), key: "07".repeat(32), grid: [1, 4, 4],
                fixture: Some(ProbeFixture { path: "chart.png".into(), sha256: "ab".repeat(32) }),
                image_url: Some(ProbeImageUrl { url: "fixture".into(), detail: None }) }], ..Default::default() };
        let mut job = request(vec![image()]); job.probe = Some(Probe::new(spec));
        let (prompt, media, _) = prepare(job, tokens, &config(), 32, 2, 16).unwrap();
        (prompt, media, serde_json::json!({"snapshot_revision": "test"}))
    }
    fn feature_files(root: &std::path::Path, identity: &serde_json::Value, payload: &[u8]) -> serde_json::Value {
        use sha2::{Digest, Sha256};
        let meta = serde_json::json!({"schema":"cuteafd.media.features/1", "key":"07".repeat(32), "grid":[1,4,4],
            "shape":[4,2], "dtype":"bf16-le", "sha256":format!("{:x}", Sha256::digest(payload)),
            "tower_dtype":"bf16", "fixture_sha256":"ab".repeat(32), "snapshot_identity":identity});
        std::fs::write(root.join(format!("{}.json", "07".repeat(32))), serde_json::to_vec(&meta).unwrap()).unwrap();
        std::fs::write(root.join(format!("{}.bf16", "07".repeat(32))), payload).unwrap();
        meta
    }
    #[test]
    fn feature_override_is_cold_budgeted_and_never_warms_native_keys_or_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let (prompt, mut media, identity) = feature_request();
        let payload = [0u8; 16]; feature_files(root.path(), &identity, &payload);
        let mut cache = cuteafd_engine::media::EmbeddingCache::new(16);
        apply_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).unwrap();
        assert!(media.ready(0, prompt.tokens.len()));
        assert_eq!(media.spans()[0].key, image().key.into());
        assert!(!cache.contains(image().key), "reference override must never enter the plain-image cache");
        assert_eq!((cache.bytes(), cache.len()), (16, 1));
        assert!(crate::shared::probe::cold(&prompt.job.probe), "scheduler must bypass prefix restore and captures");
        assert!(crate::shared::probe::no_speculation(&prompt.job.probe));
        let provenance = prompt.job.probe.as_ref().unwrap().record().provenance.unwrap();
        assert_eq!(provenance["encoder_bypassed"], true);
        let mut chunk = cuteafd_engine::media::MediaChunk::default();
        media.write_chunk(2, 6, &mut chunk).unwrap();
        assert_eq!(chunk.features, payload);
        drop(media);
        cache.reserve(ImageKey([8; 32]), 16).unwrap();
        assert!(!cache.contains(image().key));
    }
    #[test]
    fn override_admission_bypasses_even_an_unavailable_native_encoder() {
        use cuteafd_engine::media::{EmbeddingCache, MediaAdmission, MediaWaiter, MediaPoll};
        let root = tempfile::tempdir().unwrap(); let (prompt, mut request, identity) = feature_request();
        feature_files(root.path(), &identity, &[0; 16]);
        let mut cache = EmbeddingCache::new(16);
        apply_probe_features(&prompt, &mut request, &mut cache, root.path(), &identity).unwrap();
        let input = image();
        let jobs = vec![EncodeJob::image(input.key, [1,4,4], input.rgb8.clone(), 4, 2)];
        let waiter = MediaWaiter::new(prompt, request, jobs, 0).unwrap();
        let mut admission = MediaAdmission::new(cache, Encoder::Off, 1);
        assert!(admission.enqueue(waiter).is_ok());
        assert!(matches!(admission.poll(|_| false), MediaPoll::Ready(_)));
        assert!(!admission.cache.contains(input.key));
        assert_eq!(admission.stats(0, 0).encodes, 0);
    }
    #[test]
    fn override_requires_scoring_cold_no_speculation_and_safe_complete_files() {
        use cuteafd_api::openai::probe::Probe;
        for field in ["cold", "speculation", "scoring", "fixture"] {
            let root = tempfile::tempdir().unwrap(); let (mut prompt, mut media, identity) = feature_request();
            feature_files(root.path(), &identity, &[0; 16]);
            let mut spec = prompt.job.probe.as_ref().unwrap().spec.clone();
            match field { "cold" => spec.cold = false, "speculation" => spec.no_speculation = false,
                "scoring" => spec.score_from = None, _ => spec.media[0].fixture = None }
            prompt.job.probe = Some(Probe::new(spec));
            let mut cache = cuteafd_engine::media::EmbeddingCache::new(16);
            assert!(apply_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err());
            assert_eq!(cache.bytes(), 0);
        }
        for mode in ["missing", "json", "oversized", "escape"] {
            let root = tempfile::tempdir().unwrap(); let outside = tempfile::tempdir().unwrap();
            let (prompt, mut media, identity) = feature_request();
            let path = root.path().join(format!("{}.json", "07".repeat(32)));
            match mode {
                "missing" => (), "json" => { feature_files(root.path(), &identity, &[0; 16]); std::fs::write(&path, "{").unwrap(); }
                "oversized" => { feature_files(root.path(), &identity, &[0; 16]); std::fs::write(&path, vec![b' '; (64 << 10) + 1]).unwrap(); }
                _ => { feature_files(outside.path(), &identity, &[0; 16]);
                    #[cfg(unix)] {
                        std::os::unix::fs::symlink(outside.path().join(path.file_name().unwrap()), &path).unwrap();
                        std::os::unix::fs::symlink(outside.path().join(format!("{}.bf16", "07".repeat(32))),
                            root.path().join(format!("{}.bf16", "07".repeat(32)))).unwrap();
                    }
                }
            }
            let mut cache = cuteafd_engine::media::EmbeddingCache::new(16);
            assert!(apply_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err());
            assert_eq!(cache.bytes(), 0);
        }
    }
    #[test]
    fn feature_metadata_payload_and_admission_fail_closed() {
        for field in ["schema", "key", "grid", "shape", "dtype", "tower_dtype", "fixture_sha256", "snapshot_identity", "unknown"] {
            let root = tempfile::tempdir().unwrap(); let (prompt, mut media, identity) = feature_request();
            let mut meta = feature_files(root.path(), &identity, &[0; 16]); meta[field] = serde_json::json!("wrong");
            std::fs::write(root.path().join(format!("{}.json", "07".repeat(32))), serde_json::to_vec(&meta).unwrap()).unwrap();
            let mut cache = cuteafd_engine::media::EmbeddingCache::new(16);
            assert!(apply_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err(), "{field}");
            assert_eq!(cache.bytes(), 0); assert!(!media.ready(0, 8));
        }
        for (payload, budget) in [(vec![0; 15], 16), (vec![0; 17], 16), ([0x80, 0x7f].repeat(8), 16), (vec![0; 16], 15)] {
            let root = tempfile::tempdir().unwrap(); let (prompt, mut media, identity) = feature_request();
            feature_files(root.path(), &identity, &payload);
            let mut cache = cuteafd_engine::media::EmbeddingCache::new(budget);
            assert!(apply_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err());
            drop(media); cache.prune_reservations(); assert_eq!(cache.bytes(), 0);
        }
        let root = tempfile::tempdir().unwrap(); let (prompt, mut media, identity) = feature_request();
        feature_files(root.path(), &identity, &[0; 16]);
        std::fs::write(root.path().join(format!("{}.bf16", "07".repeat(32))), [1; 16]).unwrap();
        let mut cache = cuteafd_engine::media::EmbeddingCache::new(16);
        assert!(apply_probe_features(&prompt, &mut media, &mut cache, root.path(), &identity).is_err());
        drop(media); cache.prune_reservations(); assert_eq!(cache.bytes(), 0);
    }
    #[test]
    fn warm_override_revalidates_without_duplicate_feature_allocation() {
        use sha2::{Digest, Sha256};
        let root = tempfile::tempdir().unwrap(); let path = root.path().join("features");
        let cached: Arc<[u8]> = Arc::from([0; 16]); std::fs::write(&path, &cached).unwrap();
        let hash = format!("{:x}", Sha256::digest(cached.as_ref()));
        let warm = feature_payload(&path, 16, &hash, Some(&cached)).unwrap();
        assert!(Arc::ptr_eq(&warm, &cached));
        std::fs::write(&path, [1; 16]).unwrap();
        assert!(feature_payload(&path, 16, &hash, Some(&cached)).is_err());
        std::fs::write(&path, [0; 15]).unwrap();
        assert!(feature_payload(&path, 16, &hash, Some(&cached)).is_err());
    }
    #[test]
    fn snapshot_identity_binds_actual_snapshot_files() {
        use sha2::{Digest, Sha256};
        let root = tempfile::tempdir().unwrap();
        for file in ["config.json", "tokenizer.json", "modeling_mimo_v2.py", "preprocessor_config.json"] {
            std::fs::write(root.path().join(file), file).unwrap();
        }
        let before = snapshot_identity(root.path()).unwrap();
        assert_eq!(before["config_sha256"], format!("{:x}", Sha256::digest(b"config.json")));
        std::fs::write(root.path().join("modeling_mimo_v2.py"), "changed").unwrap();
        assert_ne!(snapshot_identity(root.path()).unwrap(), before);
        std::fs::remove_file(root.path().join("tokenizer.json")).unwrap();
        assert!(snapshot_identity(root.path()).is_err());
    }
    #[test]
    fn ordinary_and_generation_requests_never_consult_feature_files() {
        let (mut prompt, mut media, _) = feature_request(); prompt.job.probe = None;
        let mut cache = cuteafd_engine::media::EmbeddingCache::new(16);
        probe_features(&prompt, &mut media, &mut cache, std::path::Path::new("/does/not/exist")).unwrap();
        let (mut prompt, mut media, _) = feature_request();
        let mut spec = prompt.job.probe.as_ref().unwrap().spec.clone(); spec.score_from = None;
        prompt.job.probe = Some(cuteafd_api::openai::probe::Probe::new(spec));
        probe_features(&prompt, &mut media, &mut cache, std::path::Path::new("/does/not/exist")).unwrap();
        assert_eq!(cache.bytes(), 0);
    }
    #[test]
    fn off_vision_never_opens_config_or_tower_payloads() {
        assert!(vision_config(MediaMode::Off, std::path::Path::new("/does/not/exist")).unwrap().is_none());
        assert!(vision_config(MediaMode::Rtx(None), std::path::Path::new("/does/not/exist")).is_err());
    }
    #[test]
    fn expanded_image_ids_and_prefix_hints_are_separate() {
        let (prompt, media, jobs) = prepare(request(vec![image()]), vec![1, 4, 5, 6, 2],
            &config(), 32, 2, 16).unwrap();
        assert_eq!(prompt.tokens, [1, 4, 5, 5, 5, 5, 6, 2]);
        assert_eq!(&prompt.keys.tokens()[..2], &[1, 4]);
        assert!(prompt.keys.tokens()[2..6].iter().all(|t| t & (1 << 31) != 0));
        assert_eq!(media.spans()[0].start, 2);
        assert_eq!(media.spans()[0].len, 4);
        assert!(!media.ready(0, 8));
        assert!(media.ready(6, 8));
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].image_input().unwrap().0, [1, 4, 4]);
        assert_eq!(jobs[0].feature_bytes().unwrap(), 16);
    }
    #[test]
    fn generation_probe_echoes_real_expanded_spans_without_sources() {
        let mut job = request(vec![image()]);
        let probe = cuteafd_api::openai::probe::Probe::new(Default::default());
        job.probe = Some(probe.clone());
        prepare(job, vec![1, 4, 5, 6, 2], &config(), 32, 2, 16).unwrap();
        let record = probe.record();
        assert_eq!((record.media[0].start, record.media[0].len, record.media[0].grid), (2, 4, [1, 4, 4]));
        assert_eq!(record.media[0].key, "07".repeat(32));
        assert!(record.media[0].image_url.is_none());
    }
    #[test]
    fn expanded_probe_ids_are_verified_not_expanded_twice() {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec, ProbeMedia, ProbeImageUrl, ProbeFixture};
        let tokens = vec![1, 4, 5, 5, 5, 5, 6, 2];
        let span = ProbeMedia { start: 2, len: 4, kind: "image".into(), key: "07".repeat(32), grid: [1, 4, 4],
            fixture: Some(ProbeFixture { path: "chart.png".into(), sha256: "ab".repeat(32) }),
            image_url: Some(ProbeImageUrl { url: "data:image/png;base64,fixture".into(), detail: None }) };
        let run = |span: ProbeMedia, tokens: Vec<u32>| {
            let probe = Probe::new(ProbeSpec { prompt_ids: Some(tokens.clone()), media: vec![span], ..Default::default() });
            let mut job = request(vec![image()]); job.probe = Some(probe.clone());
            prepare(job, tokens, &config(), 32, 2, 16).map(|prepared| (prepared, probe.record()))
        };
        let ((prompt, _, _), record) = run(span.clone(), tokens.clone()).unwrap();
        assert_eq!(prompt.tokens, tokens);
        assert_eq!(record.media[0].fixture, span.fixture);
        assert!(record.media[0].image_url.is_none());
        let mut wrong = span.clone(); wrong.key = "08".repeat(32); assert!(run(wrong, tokens.clone()).is_err());
        let mut wrong = span.clone(); wrong.grid = [1, 2, 8]; assert!(run(wrong, tokens.clone()).is_err());
        let mut wrong = tokens.clone(); wrong[3] = 1; assert!(run(span.clone(), wrong).is_err());
        let mut wrong = tokens; wrong[1] = 1; assert!(run(span, wrong).is_err());
        let mut job = request(vec![image()]);
        job.probe = Some(Probe::new(ProbeSpec { prompt_ids: Some(vec![4, 5, 6]), ..Default::default() }));
        assert!(prepare(job, vec![4, 5, 6], &config(), 32, 2, 16).is_err());
    }
    #[test]
    fn text_path_keeps_native_ids_and_rejects_missing_image_rows() {
        let tokens = vec![1, 2, 3];
        let (prompt, media, jobs) = prepare(request(Vec::new()), tokens.clone(), &config(), 32, 2, 16).unwrap();
        assert_eq!(prompt.tokens, tokens);
        assert_eq!(prompt.keys.tokens(), tokens);
        assert!(media.spans().is_empty() && jobs.is_empty());
        assert!(prepare(request(vec![image()]), vec![1, 2], &config(), 32, 2, 16).is_err());
        assert!(prepare(request(vec![image()]), vec![4, 5, 6], &config(), 32, 2, 4).is_err());
        assert!(prepare(request(vec![image()]), vec![4, 5, 6], &serde_json::json!({}), 32, 2, 16).is_err());
    }
}
