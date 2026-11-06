//! GLM Flash's image request state and local/remote encoder readiness.
use anyhow::{Context, Result};
use cuteafd_api::openai::{media::MediaPreparer, NativeRequest};
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use cuteafd_loader::{media::{ImageFamily, ProcessorConfig, SpanExpander}, plan::MediaMode};
use std::sync::Arc;

pub(super) fn vision_config(mode: MediaMode, snapshot: &std::path::Path) -> Result<Option<serde_json::Value>> {
    if mode == MediaMode::Off { return Ok(None); }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
    Ok(config.get("vision_config").is_some().then_some(config))
}

pub(crate) fn local_vision_gpu(mode: MediaMode, device: i32, split_device: Option<i32>) -> Result<i32> {
    // Explicit CLI indices name CUDA devices; only defaults use logical ranks.
    if let MediaMode::Rtx(Some(gpu)) = mode { return Ok(i32::try_from(gpu)?); }
    let rank = cuteafd_loader::plan::encoder::rtx_encoder_gpu(mode, false, 1 + usize::from(split_device.is_some()));
    Ok(if rank == 0 { device } else { split_device.context("vision peer GPU")? })
}

pub(crate) struct RemoteVision {
    addresses: Vec<std::net::SocketAddr>,
    plan_hash: [u8; 32],
    revision: String,
}
impl RemoteVision {
    pub(crate) fn revision(&self) -> &str { &self.revision }
    pub(crate) fn plan_hash(&self) -> [u8; 32] { self.plan_hash }
    /// Connects every replica with the coordinator's expected handshake.
    pub(crate) fn connect(self, expected: crate::shared::vision::remote::EncoderHandshake) -> Result<Encoder> {
        Ok(Encoder::Remote(crate::shared::vision::remote::RemoteEncoder::connect(self.addresses, expected,
            std::time::Duration::from_secs(60))?))
    }
    pub fn from_args(args: &super::serve::ServeArgs) -> Result<Option<Self>> {
        Self::from_parts(args.vision, args.vision_peers.as_deref(), args.encoder_plan_hash.as_deref(), args.encoder_revision.as_deref())
    }
    /// The planner-resolved Spark encoder endpoints from their CLI values (any serve family).
    pub(crate) fn from_parts(vision: MediaMode, peers: Option<&str>, plan_hash: Option<&str>, revision: Option<&str>)
        -> Result<Option<Self>> {
        let Some(peers) = peers else {
            anyhow::ensure!(plan_hash.is_none() && revision.is_none(),
                "encoder plan hash/revision require --vision-peers");
            return Ok(None);
        };
        anyhow::ensure!(matches!(vision, MediaMode::Auto | MediaMode::Spark(_)),
            "--vision-peers requires Spark/auto vision placement");
        let addresses = peers.split(',').map(|peer| peer.trim().parse()
            .context("vision peer must be an IP:port")).collect::<Result<Vec<_>>>()?;
        anyhow::ensure!((1..=8).contains(&addresses.len()), "vision needs 1..8 replicas");
        let plan_hash = crate::shared::vision::worker::parse_plan_hash(plan_hash
            .context("--vision-peers requires --encoder-plan-hash")?)?;
        let revision = revision.context("--vision-peers requires --encoder-revision")?.to_owned();
        anyhow::ensure!(!revision.is_empty(), "encoder revision must not be empty");
        Ok(Some(Self { addresses, plan_hash, revision }))
    }
}

pub(super) struct ReadyVision {
    pub push: crate::shared::vision::push::Contribution,
    pub encoder: Encoder,
    pub preparer: Arc<MediaPreparer>,
    pub cache_bytes: usize,
}
impl ReadyVision {
    pub fn plan(args: &super::EngineArgs, mode: MediaMode, remote: Option<&RemoteVision>)
        -> Result<Option<cuteafd_loader::plan::encoder::EncoderPlacement>> {
        use cuteafd_loader::plan::encoder::{EncoderKind, EncoderPlacement};
        if vision_config(mode, &args.snapshot)?.is_none() { return Ok(None); }
        let spec = crate::shared::vision::TowerSpec::glm_flash(&args.snapshot, 4096)?;
        let ledger = cuteafd_ffi::vision::NativeVision::required(&args.native_lib, &spec.native)?;
        let (kind, replicas) = if let Some(remote) = remote {
            let peers = args.peers.as_deref().context("remote Flash encoder needs active Spark workers")?
                .split(',').map(|peer| peer.parse::<std::net::SocketAddr>()).collect::<std::result::Result<Vec<_>, _>>()?;
            let ranks = remote.addresses.iter().map(|address| peers.iter().position(|peer| peer.ip() == address.ip())
                .context("remote Flash encoder is not an admitted expert worker"))
                .collect::<Result<Vec<_>>>()?;
            (EncoderKind::Spark { rank: ranks[0] }, ranks[1..].to_vec())
        } else {
            anyhow::ensure!(!matches!(mode, MediaMode::Spark(_)), "Spark vision requires planner-resolved peers");
            let device = local_vision_gpu(mode, args.device, args.split_device)?;
            let rank = std::iter::once(args.device).chain(args.split_device).position(|gpu| gpu == device)
                .context("Flash vision device is outside the target topology")?;
            (EncoderKind::Rtx { gpu: rank }, vec![])
        };
        Ok(Some(EncoderPlacement { kind, weights: ledger.weights,
            scratch: ledger.scratch.checked_add(ledger.blas_workspace).context("Flash vision ledger overflow")?,
            replicas, reason: "launcher-resolved bundled encoder".into(), shortfall: 0 }))
    }

    pub fn load(args: &super::EngineArgs, library: &cuteafd_ffi::NativeLibrary, mode: MediaMode,
        cache_bytes: usize, remote: Option<RemoteVision>) -> Result<Option<Self>> {
        let Some(config) = vision_config(mode, &args.snapshot)? else { return Ok(None); };
        let admitted = args.serving_encoder.as_ref().context("Flash vision lacks admitted physical owners")?;
        let processor = ProcessorConfig::from_snapshot(&args.snapshot, ImageFamily::GlmFlash)?;
        // Header-only on the coordinator: remote tower payload stays on its Spark.
        let spec = crate::shared::vision::TowerSpec::glm_flash(&args.snapshot, 4096)?;
        let width = config["text_config"]["hidden_size"].as_u64().context("GLM Flash hidden_size")? as usize;
        if let Some(remote) = remote {
            anyhow::ensure!(matches!(mode, MediaMode::Auto | MediaMode::Spark(_)),
                "--vision-peers requires Spark/auto vision placement");
            let id = spec.encoder_id(&remote.revision, 121);
            let preparer = Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), id, 4)?);
            anyhow::ensure!(preparer.config().max_image_tokens <= 4096, "GLM Flash tower capacity is 4096 tokens per image");
            let expected = crate::shared::vision::remote::EncoderHandshake {
                encoder_id: id, max_patches: 4096 * processor.merge.pow(2), output_width: spec.native.output_width,
                patch_size: processor.patch, merge_size: processor.merge, row_markers: 0, plan_hash: remote.plan_hash,
            };
            anyhow::ensure!(expected.output_width as usize == width, "GLM Flash tower/LM output width mismatch");
            let encoder = crate::shared::vision::remote::RemoteEncoder::connect(remote.addresses, expected,
                std::time::Duration::from_secs(60))?;
            tracing::info!(cache_bytes, "GLM Flash remote vision encoder ready");
            anyhow::ensure!(matches!(admitted.kind, cuteafd_loader::plan::encoder::EncoderKind::Spark { .. }),
                "remote Flash vision differs from admitted home");
            return Ok(Some(Self { push: Default::default(), encoder: Encoder::Remote(encoder), preparer, cache_bytes }));
        }
        let gpu = match mode {
            MediaMode::Rtx(_) => local_vision_gpu(mode, args.device, args.split_device)?,
            MediaMode::Auto => {
                anyhow::ensure!(args.peers.is_none() || args.local_experts,
                    "auto vision with Spark experts requires planner-resolved --vision-peers");
                local_vision_gpu(mode, args.device, args.split_device)?
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
        let rank = std::iter::once(args.device).chain(args.split_device).position(|device| device == gpu)
            .context("Flash vision GPU absent from target topology")?;
        anyhow::ensure!(admitted.kind == cuteafd_loader::plan::encoder::EncoderKind::Rtx { gpu: rank }
            && ledger.weights == admitted.weights && ledger.total_bytes() == admitted.admitted_bytes(),
            "Flash vision allocation differs from admitted physical owners");
        library.cuda_set_device(gpu)?;
        let admitted = admitted.admitted_bytes();
        let loaded = crate::shared::vision::EncoderService::start_coordinator(&args.snapshot,
            Some((spec, admitted)), args.native_lib.clone(), gpu, None);
        let restored = library.cuda_set_device(args.device);
        let (service, push) = loaded?;
        restored?;
        tracing::info!(gpu, admitted_bytes = admitted, cache_bytes, sm, "GLM Flash resident vision encoder allocated");
        Ok(Some(Self { push, encoder: Encoder::Local(crate::shared::vision::local::LocalEncoder::new(service, &processor, width, 4096)),
            preparer, cache_bytes }))
    }
}

pub(crate) enum Encoder {
    Local(crate::shared::vision::local::LocalEncoder),
    Remote(crate::shared::vision::remote::RemoteEncoder),
    Off,
}
impl Encoder {
    pub fn health_handle(&self) -> Option<Arc<std::sync::atomic::AtomicBool>> {
        match self { Self::Remote(client) => Some(client.health_handle()), Self::Local(client) => Some(client.health_handle()), Self::Off => None }
    }
    pub fn available(&self) -> bool {
        match self { Self::Remote(client) => client.healthy(), Self::Local(client) => client.healthy(), Self::Off => false }
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
pub(crate) fn failure(error: cuteafd_engine::media::MediaError) -> cuteafd_api::openai::NativeFailure {
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
            let spans = crate::shared::probe::image_spans(&probe.spec, &tokens, &images,
                crate::shared::probe::ImageMarkers::Bounded(expander), max_context)?;
            (tokens, spans)
        } else {
            let expanded = expander.expand(&tokens, &images, max_context)?;
            (expanded.tokens, expanded.media)
        }
    } else {
        anyhow::ensure!(job.media.is_empty(), "checkpoint has no vision tower");
        (tokens, Vec::new())
    };
    crate::shared::probe::image_echo(&job, &spans)?;
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let keys = MediaKeys::new(&tokens, vocabulary as u32, &spans)?;
    let jobs = job.media.iter().map(|image| EncodeJob::image(image.key, [image.grid.t, image.grid.h, image.grid.w], image.rgb8.clone(), image.tokens, hidden)).collect();
    Ok((Prompt { job, tokens, keys }, media, jobs))
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

pub(super) use crate::shared::probe::features::reference_probe;
#[cfg(test)]
use crate::shared::probe::features::feature_payload;

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
    crate::shared::probe::features::apply_probe_features(&prompt.job, media, cache, root, identity,
        "62d7ebd7de4938e072b7aaeb881593b79dc56835")
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::TargetSamplingParams;
    use cuteafd_loader::media::{ImageGrid, ImageKey, PreparedImage};

    #[test]
    fn local_encoder_defaults_match_planner_and_map_logical_ranks() {
        for mode in [MediaMode::Auto, MediaMode::Rtx(None)] {
            for (device, peer) in [(0, None), (3, None), (0, Some(1)), (1, Some(0)), (3, Some(5))] {
                let rank = cuteafd_loader::plan::encoder::rtx_encoder_gpu(mode, false, 1 + usize::from(peer.is_some()));
                assert_eq!(local_vision_gpu(mode, device, peer).unwrap(),
                    if rank == 0 { device } else { peer.unwrap() });
            }
        }
        assert_eq!(local_vision_gpu(MediaMode::Rtx(Some(0)), 1, Some(0)).unwrap(), 0);
        assert_eq!(local_vision_gpu(MediaMode::Rtx(Some(3)), 0, None).unwrap(), 3);
    }

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
        NativeRequest { prefix_cache_source: Default::default(), finish_detail: Default::default(), literal_spans: Vec::new(), prompt: String::new(), constraint: None, images: Vec::new(), media: images, audio: Vec::new(),
            max_tokens: 8, sampling: TargetSamplingParams::default(), stop_token_ids: Vec::new(),
            events: tokio::sync::mpsc::unbounded_channel().0, usage: None, probe: None }
    }
    fn config() -> serde_json::Value {
        serde_json::json!({"vision_config": {}, "image_token_id": 5,
            "image_start_token_id": 4, "image_end_token_id": 6})
    }
    fn image() -> Arc<PreparedImage> {
        Arc::new(PreparedImage { key: ImageKey([7; 32]), encoder: cuteafd_loader::media::EncoderId([0; 32]), processor: cuteafd_loader::media::PreprocessId([0; 32]), grid: ImageGrid { t: 1, h: 4, w: 4 },
            rgb8: Arc::from(vec![0; 4 * 4 * 14 * 14 * 3]), tokens: 4 })
    }
    fn feature_request() -> (Prompt, RequestMedia, serde_json::Value) {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec, ProbeMedia, ProbeFixture, ProbeImageUrl};
        let tokens = vec![1, 4, 5, 5, 5, 5, 6, 2];
        let spec = ProbeSpec { prompt_ids: Some(tokens.clone()), cold: true, no_speculation: true, score_from: Some(7),
            media: vec![ProbeMedia { start: 2, len: 4, kind: "image".into(), key: "07".repeat(32), grid: [1, 4, 4],
                identity: None, input_key: None, fixture: Some(ProbeFixture { path: "chart.png".into(), sha256: "ab".repeat(32) }),
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
    fn auto_without_a_tower_keeps_text_template_selection() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("config.json"), b"{}").unwrap();
        assert!(vision_config(MediaMode::Auto, temp.path()).unwrap().is_none());
        std::fs::write(temp.path().join("config.json"), br#"{"vision_config":{"depth":24}}"#).unwrap();
        assert!(vision_config(MediaMode::Auto, temp.path()).unwrap().is_some());
        assert!(vision_config(MediaMode::Off, temp.path()).unwrap().is_none());
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
            identity: None, input_key: None, fixture: Some(ProbeFixture { path: "chart.png".into(), sha256: "ab".repeat(32) }),
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
