//! GLM Flash's image request state and local/remote encoder readiness.
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

pub(super) struct ReadyVision {
    pub encoder: Encoder,
    pub preparer: Arc<MediaPreparer>,
    pub cache_bytes: usize,
}
impl ReadyVision {
    pub fn load(args: &super::EngineArgs, library: &cuteafd_ffi::NativeLibrary, mode: MediaMode, prefix: &crate::shared::prefix::PrefixArgs,
        cache_bytes: Option<u64>, remote: Option<RemoteVision>) -> Result<(Option<Self>, crate::shared::prefix::PrefixArgs)> {
        let Some(config) = vision_config(mode, &args.snapshot)? else { return Ok((None, prefix.clone())); };
        let processor = ProcessorConfig::from_snapshot(&args.snapshot, ImageFamily::GlmFlash)?;
        // Header-only on the coordinator: remote tower payload stays on its Spark.
        let spec = crate::shared::vision::TowerSpec::glm_flash(&args.snapshot, 4096)?;
        let width = config["text_config"]["hidden_size"].as_u64().context("GLM Flash hidden_size")? as usize;
        let (prefix, cache_bytes) = prefix.with_media_headroom(cache_bytes)?;
        if let Some(remote) = remote {
            anyhow::ensure!(matches!(mode, MediaMode::Auto | MediaMode::Spark(_)),
                "--vision-peers requires Spark/auto vision placement");
            let id = spec.encoder_id(&remote.revision, 121);
            let preparer = Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), id, 4)?);
            anyhow::ensure!(preparer.config().max_image_tokens <= 4096, "GLM Flash tower capacity is 4096 tokens per image");
            let expected = crate::shared::vision::remote::EncoderHandshake {
                encoder_id: id, max_patches: 4096 * processor.merge.pow(2), output_width: spec.native.output_width,
                patch_size: processor.patch, merge_size: processor.merge, plan_hash: remote.plan_hash,
            };
            anyhow::ensure!(expected.output_width as usize == width, "GLM Flash tower/LM output width mismatch");
            let encoder = crate::shared::vision::remote::RemoteEncoder::connect(remote.addresses, expected,
                std::time::Duration::from_secs(60))?;
            tracing::info!(cache_bytes, "GLM Flash remote vision encoder ready");
            return Ok((Some(Self { encoder: Encoder::Remote(encoder), preparer, cache_bytes }), prefix));
        }
        let gpu = match mode {
            MediaMode::Rtx(gpu) => gpu.map(|gpu| i32::try_from(gpu)).transpose()?.unwrap_or(args.device),
            MediaMode::Auto => {
                anyhow::ensure!(args.peers.is_none() || args.local_experts,
                    "auto vision with Spark experts requires planner-resolved --vision-peers");
                args.device
            },
            MediaMode::Spark(_) => anyhow::bail!("Spark vision placement requires --vision-peers, --encoder-plan-hash and --encoder-revision"),
            MediaMode::Off => unreachable!(),
        };
        let info = library.cuda_device_info(gpu)?;
        let sm = u32::try_from(info.compute_capability_major * 10 + info.compute_capability_minor)?;
        let revision = args.snapshot.file_name().and_then(|v| v.to_str()).context("snapshot revision")?;
        let preparer = Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), spec.encoder_id(revision, sm), 4)?);
        anyhow::ensure!(preparer.config().max_image_tokens <= 4096, "GLM Flash tower capacity is 4096 tokens per image");
        let ledger = cuteafd_ffi::vision::NativeVision::required(&args.native_lib, &spec.native)?;
        library.cuda_set_device(gpu)?;
        let admitted = ledger.total_bytes();
        let loaded = (|| -> Result<_> {
        let (free, total) = library.cuda_memory_info()?;
        cuteafd_core::serving_capacity::admit_device_reservations(95,
            cuteafd_core::serving_capacity::DeviceMemory { device: gpu as u32,
                total_bytes: total as u64, baseline_free_bytes: free as u64 },
            &[cuteafd_core::serving_capacity::MemoryReservation { name: "vision.resident_weights_scratch".into(), bytes: admitted }])?;
        // Start before LM preflight: its live baseline already includes the admitted tower.
        Ok(crate::shared::vision::EncoderService::start(spec, args.native_lib.clone(), gpu, admitted)?)
        })();
        let restored = library.cuda_set_device(args.device);
        let service = loaded?;
        restored?;
        tracing::info!(gpu, admitted_bytes = admitted, cache_bytes, sm, "GLM Flash resident vision encoder ready");
        Ok((Some(Self { encoder: Encoder::Local(crate::shared::vision::local::LocalEncoder::new(service, &processor, width, 4096)),
            preparer, cache_bytes }), prefix))
    }
}

pub(super) enum Encoder {
    Local(crate::shared::vision::local::LocalEncoder),
    Remote(crate::shared::vision::remote::RemoteEncoder),
    Off,
}
impl Encoder {
    pub fn health_handle(&self) -> Option<Arc<std::sync::atomic::AtomicBool>> {
        match self { Self::Remote(client) => Some(client.health_handle()), _ => None }
    }
    pub fn available(&self) -> bool {
        match self { Self::Remote(client) => client.healthy(), Self::Local(_) => true, Self::Off => false }
    }
}
impl cuteafd_engine::media::EncoderClient for Encoder {
    fn submit(&mut self, job: EncodeJob) -> std::result::Result<cuteafd_engine::media::EncoderTicket, cuteafd_engine::media::MediaError> {
        use cuteafd_engine::media::MediaError;
        match self { Self::Local(client) => client.submit(job), Self::Remote(client) => client.submit(job),
            Self::Off => Err(MediaError::Encoder("encoder not loaded".into())) }
    }
    fn poll(&mut self, ticket: cuteafd_engine::media::EncoderTicket) -> Option<std::result::Result<cuteafd_engine::media::EncodeOutput, cuteafd_engine::media::MediaError>> {
        match self { Self::Local(client) => client.poll(ticket), Self::Remote(client) => client.poll(ticket), Self::Off => None }
    }
    fn cancel(&mut self, ticket: cuteafd_engine::media::EncoderTicket) {
        match self { Self::Local(client) => client.cancel(ticket), Self::Remote(client) => client.cancel(ticket), Self::Off => () }
    }
}
pub(super) fn failure(error: cuteafd_engine::media::MediaError) -> cuteafd_api::openai::NativeFailure {
    match error {
        cuteafd_engine::media::MediaError::Encoder(_) => cuteafd_api::openai::NativeFailure::Unavailable("vision encoder unavailable".into()),
        error @ (cuteafd_engine::media::MediaError::QueueFull | cuteafd_engine::media::MediaError::CacheFull { .. }) =>
            cuteafd_api::openai::NativeFailure::Unavailable(error.to_string()),
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
    let (tokens, spans) = if config.get("vision_config").is_some() {
        let expander = SpanExpander::from_config(config, vocabulary as u32)?;
        let images = job.media.iter().map(|image| image.as_ref().clone()).collect::<Vec<_>>();
        if job.probe.as_ref().is_some_and(|p| p.spec.prompt_ids.is_some() && !images.is_empty()) {
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
                spans.push(cuteafd_loader::media::MediaSpan { start: span.start, len: span.len, key: image.key });
            }
            anyhow::ensure!(tokens.iter().filter(|&&id| id == expander.placeholder).count()
                == spans.iter().map(|span| span.len).sum::<usize>(), "unbound probe image placeholders");
            (tokens, spans)
        } else {
            let expanded = expander.expand(&tokens, &images, max_context)?;
            (expanded.tokens, expanded.media)
        }
    } else {
        anyhow::ensure!(job.media.is_empty(), "checkpoint has no vision tower");
        (tokens, Vec::new())
    };
    if let Some(probe) = &job.probe {
        anyhow::ensure!(spans.len() == job.media.len(), "probe image count differs");
        let echo = spans.iter().zip(&job.media).enumerate().map(|(i, (span, image))| {
            cuteafd_api::openai::probe::ProbeMedia { start: span.start, len: span.len, kind: "image".into(),
                key: key_hex(span.key), grid: [image.grid.t, image.grid.h, image.grid.w],
                fixture: probe.spec.media.get(i).and_then(|s| s.fixture.clone()), image_url: None }
        }).collect();
        probe.media(echo);
    }
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let keys = MediaKeys::new(&tokens, vocabulary as u32, &spans)?;
    let jobs = job.media.iter().map(|image| EncodeJob { key: image.key,
        grid: [image.grid.t, image.grid.h, image.grid.w], rgb8: image.rgb8.clone(),
        tokens: image.tokens, hidden_width: hidden }).collect();
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
        ("preprocessor", "processor_config.json"), ("index", "model.safetensors.index.json")] {
        identity[format!("{key}_sha256")] = format!("{:x}", Sha256::digest(std::fs::read(snapshot.join(file))?)).into();
    }
    // Compile the actual locked transformer module into the diagnostic identity;
    // serving images do not need an invented modeling file in the HF snapshot.
    identity["modeling_sha256"] = format!("{:x}", Sha256::digest(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../../third_party/transformers/src/transformers/models/glm5_next/modeling_glm5_next.py")))).into();
    identity["image_processing_sha256"] = format!("{:x}", Sha256::digest(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"), "/../../../third_party/transformers/src/transformers/models/glm5_next/image_processing_pil_glm5_next.py")))).into();
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

pub(super) fn reference_probe(probe: &crate::shared::probe::ProbeRef) -> bool {
    std::env::var_os("CUTEAFD_MEDIA_FEATURES_DIR").is_some() && probe.as_ref().is_some_and(|p|
        p.spec.cold && p.spec.no_speculation && p.spec.score_from.is_some() && !p.spec.media.is_empty())
}

/// Strict paired-G4 hook. Only scoring probes bypass the encoder; ordinary
/// generation remains native. Separate cache identities cannot warm native images.
pub(super) fn probe_features(prompt: &Prompt, media: &mut RequestMedia,
    cache: &mut cuteafd_engine::media::EmbeddingCache, snapshot: &std::path::Path) -> Result<()> {
    let Some(probe) = prompt.job.probe.as_ref().filter(|p| p.spec.score_from.is_some() && !p.spec.media.is_empty()) else {
        return Ok(());
    };
    let Some(root) = std::env::var_os("CUTEAFD_MEDIA_FEATURES_DIR") else { return Ok(()); };
    anyhow::ensure!(probe.spec.cold && probe.spec.no_speculation, "feature probes require explicit cold/no_speculation");
    probe.spec.validate_media()?;
    apply_probe_features(prompt, media, cache, &std::path::PathBuf::from(root), &snapshot_identity(snapshot)?)
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
        "encoder_bypassed": true, "modeling_source_revision": "62d7ebd7de4938e072b7aaeb881593b79dc56835", "features": provenance}));
    Ok(())
}

pub(super) fn scoring_rows(probe: &crate::shared::probe::ProbeRef, capacity: usize) -> Result<usize> {
    let probe = probe.as_ref().context("scoring probe required")?;
    anyhow::ensure!(probe.spec.score_path.as_deref().is_none_or(|p| p == "decode"),
        "GLM Flash serving admits decode scoring only; prefill needs an AllRows diagnostic engine");
    let rows = probe.spec.verify_rows.unwrap_or(capacity);
    anyhow::ensure!(rows > 0 && rows <= capacity, "verify_rows outside admitted decode capacity");
    probe.selected_score_path("decode");
    Ok(rows)
}

fn key_hex(key: cuteafd_loader::media::ImageKey) -> String {
    key.0.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::TargetSamplingParams;
    use cuteafd_loader::media::{ImageGrid, ImageKey, PreparedImage};

    #[test]
    fn remote_options_require_complete_identity_and_valid_placement() {
        use clap::Parser;
        let make = |options: Vec<String>| {
            let mut argv = vec!["cuteafd".into(), "serve-glmf".into(), "--snapshot".into(), "/not-read".into(),
                "--native-lib".into(), "/not-read.so".into()];
            argv.extend(options);
            let cli = crate::cli::Cli::try_parse_from(argv).unwrap();
            let crate::cli::Commands::ServeGlmf(mut args) = cli.command else { panic!("GLM Flash command") };
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
    #[test]
    fn pending_backpressure_is_retryable_but_invalid_features_are_not() {
        use cuteafd_api::openai::NativeFailure;
        use cuteafd_engine::media::MediaError;
        assert!(matches!(failure(MediaError::QueueFull), NativeFailure::Unavailable(_)));
        assert!(matches!(failure(MediaError::Features), NativeFailure::BadRequest(_)));
    }
    fn request(images: Vec<Arc<PreparedImage>>) -> NativeRequest {
        NativeRequest { prompt: String::new(), constraint: None, images: Vec::new(), media: images,
            max_tokens: 8, sampling: TargetSamplingParams::default(), stop_token_ids: Vec::new(),
            events: tokio::sync::mpsc::unbounded_channel().0, probe: None }
    }
    fn config() -> serde_json::Value {
        serde_json::json!({"vision_config": {}, "image_token_id": 5,
            "image_start_token_id": 4, "image_end_token_id": 6})
    }
    fn image() -> Arc<PreparedImage> {
        Arc::new(PreparedImage { key: ImageKey([7; 32]), grid: ImageGrid { t: 1, h: 4, w: 4 },
            rgb8: Arc::from(vec![0; 4 * 4 * 768]), tokens: 4 })
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
        assert_eq!(media.spans()[0].key, image().key);
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
        let jobs = vec![EncodeJob { key: input.key, grid: [1,4,4], rgb8: input.rgb8.clone(), tokens: 4, hidden_width: 2 }];
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
        for file in ["config.json", "tokenizer.json", "processor_config.json", "model.safetensors.index.json"] {
            std::fs::write(root.path().join(file), file).unwrap();
        }
        let before = snapshot_identity(root.path()).unwrap();
        assert_eq!(before["config_sha256"], format!("{:x}", Sha256::digest(b"config.json")));
        std::fs::write(root.path().join("processor_config.json"), "changed").unwrap();
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
    fn scoring_shape_is_honored_or_rejected_not_silently_ignored() {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec};
        let p = Some(Probe::new(ProbeSpec { verify_rows: Some(1), score_path: Some("decode".into()), ..Default::default() }));
        assert_eq!(scoring_rows(&p, 4).unwrap(), 1);
        assert_eq!(p.unwrap().record().score_path.as_deref(), Some("decode"));
        for (rows, path) in [(0, "decode"), (5, "decode"), (1, "prefill"), (1, "typo")] {
            let p = Some(Probe::new(ProbeSpec { verify_rows: Some(rows), score_path: Some(path.into()), ..Default::default() }));
            assert!(scoring_rows(&p, 4).is_err());
        }
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
        assert_eq!(jobs[0].grid, [1, 4, 4]);
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
