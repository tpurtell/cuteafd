//! Mistral Large 4's image requests on the GLM engine: the bundled Pixtral tower
//! (resident on an RTX), the checkpoint's own Pixtral processor and its
//! `[IMG]` / `[IMG_BREAK]` / `[IMG_END]` span layout.
use anyhow::{Context, Result};
use cuteafd_api::openai::{media::MediaPreparer, NativeRequest};
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use cuteafd_loader::{media::{ImageFamily, MistralSpanExpander, ProcessorConfig}, plan::MediaMode};
use std::sync::Arc;

pub(crate) use crate::families::glm5_flash::media::{failure, Encoder};

/// Whether this checkpoint serves images: Mistral Large 4 with its tower, vision not off.
pub(crate) fn vision_config(mode: MediaMode, snapshot: &std::path::Path) -> Result<Option<serde_json::Value>> {
    if mode == MediaMode::Off { return Ok(None); }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
    Ok((config["model_type"] == "mistral_large4" && config.get("vision_config").is_some()).then_some(config))
}

pub(crate) struct ReadyVision {
    pub encoder: Encoder,
    pub preparer: Arc<MediaPreparer>,
    pub cache_bytes: usize,
    pub expander: MistralSpanExpander,
}

impl ReadyVision {
    /// The resident tower: on a Spark (planner-resolved `--vision-peers`, the remote
    /// encoder with Mistral's span rows in its handshake) or on this coordinator.
    pub fn load(args: &super::EngineArgs, library: &cuteafd_ffi::NativeLibrary, mode: MediaMode,
        prefix: &crate::shared::prefix::PrefixArgs, cache_bytes: Option<u64>,
        remote: Option<crate::families::glm5_flash::media::RemoteVision>)
        -> Result<(Option<Self>, crate::shared::prefix::PrefixArgs)> {
        let Some(config) = vision_config(mode, &args.snapshot)? else { return Ok((None, prefix.clone())); };
        let processor = ProcessorConfig::from_snapshot(&args.snapshot, ImageFamily::Mistral)?;
        let markers = crate::shared::vision::mistral::Markers::from_snapshot(&args.snapshot)
            .map_err(anyhow::Error::msg)?;
        let vocabulary = u32::try_from(config["text_config"]["vocab_size"].as_u64().context("vocab_size")?)?;
        let expander = MistralSpanExpander { image: markers.image, row_break: markers.row_break, end: markers.end,
            merge: processor.merge, vocabulary };
        let width = crate::shared::vision::mistral::LM_HIDDEN;
        let (prefix, cache_bytes) = prefix.with_media_headroom(cache_bytes)?;
        if let Some(remote) = remote {
            anyhow::ensure!(matches!(mode, MediaMode::Auto | MediaMode::Spark(_)),
                "--vision-peers requires Spark/auto vision placement");
            // Header-only here: the tower payload stays on its Spark.
            let spec = crate::shared::vision::TowerSpec::mistral_large4(&args.snapshot,
                crate::shared::vision::mistral::merged_capacity(processor.max_image_tokens))?;
            let id = spec.encoder_id(remote.revision(), 121);
            let preparer = Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), id, 4)?);
            let expected = crate::shared::vision::remote::EncoderHandshake {
                encoder_id: id, max_patches: spec.native.max_tokens * processor.merge.pow(2),
                output_width: spec.native.output_width, patch_size: processor.patch, merge_size: processor.merge,
                row_markers: spec.row_markers(), plan_hash: remote.plan_hash(),
            };
            anyhow::ensure!(expected.output_width as usize == width, "Pixtral tower/LM output width mismatch");
            let encoder = remote.connect(expected)?;
            tracing::info!(cache_bytes, "Mistral Large 4 remote vision encoder ready");
            return Ok((Some(Self { encoder, preparer, cache_bytes, expander }), prefix));
        }
        let gpu = match mode {
            MediaMode::Rtx(_) => crate::families::glm5_flash::media::local_vision_gpu(mode, args.device, args.split_device)?,
            MediaMode::Auto => {
                anyhow::ensure!(args.peers.is_none(),
                    "auto vision with Spark experts requires planner-resolved --vision-peers");
                crate::families::glm5_flash::media::local_vision_gpu(mode, args.device, args.split_device)?
            }
            MediaMode::Spark(_) => anyhow::bail!("Spark vision placement requires --vision-peers, --encoder-plan-hash and --encoder-revision"),
            MediaMode::Off => unreachable!(),
        };
        let spec = crate::shared::vision::TowerSpec::mistral_large4(&args.snapshot,
            crate::shared::vision::mistral::merged_capacity(processor.max_image_tokens))?;
        let info = library.cuda_device_info(gpu)?;
        let sm = u32::try_from(info.compute_capability_major * 10 + info.compute_capability_minor)?;
        let revision = args.snapshot.file_name().and_then(|v| v.to_str()).context("snapshot revision")?;
        let preparer = Arc::new(MediaPreparer::for_loaded_encoder(processor.clone(), spec.encoder_id(revision, sm), 4)?);
        let max_tokens = preparer.config().max_image_tokens;
        let ledger = cuteafd_ffi::vision::NativeVision::required(&args.native_lib, &spec.native)?;
        library.cuda_set_device(gpu)?;
        let admitted = ledger.total_bytes();
        let loaded = (|| -> Result<_> {
            let (free, total) = library.cuda_memory_info()?;
            cuteafd_core::serving_capacity::admit_device_reservations(95,
                cuteafd_core::serving_capacity::DeviceMemory { device: gpu as u32,
                    total_bytes: total as u64, baseline_free_bytes: free as u64 },
                &[cuteafd_core::serving_capacity::MemoryReservation { name: "vision.resident_weights_scratch".into(),
                    bytes: admitted }])?;
            // Start before the LM loads: its live baseline already includes the admitted tower.
            Ok(crate::shared::vision::EncoderService::start(spec, args.native_lib.clone(), gpu, admitted)?)
        })();
        let restored = library.cuda_set_device(args.device);
        let service = loaded?;
        restored?;
        tracing::info!(gpu, admitted_bytes = admitted, cache_bytes, sm, max_tokens, "Mistral Large 4 resident vision encoder ready");
        Ok((Some(Self { encoder: Encoder::Local(crate::shared::vision::local::LocalEncoder::new(service, &processor,
            width, max_tokens)), preparer, cache_bytes, expander }), prefix))
    }
}

/// A prompt waiting for its images' features.
pub(crate) struct Prompt {
    pub job: NativeRequest,
    pub tokens: Vec<u32>,
    pub keys: MediaKeys,
}

/// Expands each template `[IMG]` into its span, keys the prompt for the prefix
/// cache, and lists the encodes. Text-only requests pass through unchanged.
pub(crate) fn prepare(job: NativeRequest, tokens: Vec<u32>, expander: Option<&MistralSpanExpander>,
    vocabulary: usize, hidden: usize, max_context: usize) -> Result<(Prompt, RequestMedia, Vec<EncodeJob>)> {
    let (tokens, spans) = match expander {
        Some(expander) => {
            let images = job.media.iter().map(|image| image.as_ref().clone()).collect::<Vec<_>>();
            if let Some(probe) = job.probe.as_ref().filter(|p| p.spec.prompt_ids.is_some() && !images.is_empty()) {
                let spans = crate::shared::probe::image_spans(&probe.spec, &tokens, &images,
                    crate::shared::probe::ImageMarkers::Rows(crate::shared::probe::RowMarkers {
                        image: expander.image, row_break: expander.row_break, end: expander.end,
                        merge: expander.merge, vocabulary: expander.vocabulary,
                    }), max_context)?;
                (tokens, spans)
            } else {
                let expanded = expander.expand(&tokens, &images, max_context)?;
                (expanded.tokens, expanded.media)
            }
        }
        None => {
            anyhow::ensure!(job.media.is_empty(), "this deployment serves no images (vision off or no tower)");
            (tokens, Vec::new())
        }
    };
    crate::shared::probe::image_echo(&job, &spans)?;
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let keys = MediaKeys::new(&tokens, vocabulary as u32, &spans)?;
    let jobs = job.media.iter().map(|image| EncodeJob::image(image.key, [image.grid.t, image.grid.h, image.grid.w],
        image.rgb8.clone(), image.tokens, hidden)).collect();
    Ok((Prompt { job, tokens, keys }, media, jobs))
}

/// The official wheel is hash-checked by the Python reference, never loaded by serving.
fn snapshot_identity(snapshot: &std::path::Path) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let mut identity = serde_json::json!({
        "snapshot_revision": snapshot.file_name().and_then(|s| s.to_str()).context("snapshot revision")?,
        "transformers": "5.22.1",
        "modeling_sha256": "a83fc96564410fcfea5cedb3310b5a28f3c7967caddd3ee955d9131f5b974e84",
        "vision_modeling_sha256": "503e23525215ae1e59e32e1132159502b431003218874d185da9ee6842775099",
        "image_processing_sha256": "d0271e55872a2524538bc7bef05faf9fa7c395da9bb411d10c97c0d5c123bac4",
    });
    for (key, file) in [("config", "config.json"), ("tokenizer", "tokenizer.json"),
        ("preprocessor", "processor_config.json"), ("index", "consolidated.safetensors.index.json")] {
        identity[format!("{key}_sha256")] = format!("{:x}", Sha256::digest(std::fs::read(snapshot.join(file))?)).into();
    }
    Ok(identity)
}

pub(crate) fn probe_features(prompt: &Prompt, media: &mut RequestMedia,
    cache: &mut cuteafd_engine::media::EmbeddingCache, snapshot: &std::path::Path,
    expander: Option<&MistralSpanExpander>) -> Result<()> {
    let Some(probe) = prompt.job.probe.as_ref().filter(|p| p.spec.score_from.is_some() && !p.spec.media.is_empty()) else {
        return Ok(());
    };
    let Some(root) = std::env::var_os("CUTEAFD_MEDIA_FEATURES_DIR") else { return Ok(()); };
    let expander = expander.context("reference image features require the bundled vision tower")?;
    anyhow::ensure!(probe.spec.cold && probe.spec.no_speculation, "feature probes require explicit cold/no_speculation");
    probe.spec.validate_media_geometry(cuteafd_core::media::ImageTokenGeometry {
        merge: expander.merge, row_markers: 1 })?;
    crate::shared::probe::features::apply_probe_features(&prompt.job, media, cache,
        &std::path::PathBuf::from(root), &snapshot_identity(snapshot)?,
        "446a570e62b04a89ea001604ada7366baac2ea67")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_marker_features_bind_native_spans_and_chunk_boundaries() {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec, ProbeMedia, ProbeFixture, ProbeImageUrl};
        use cuteafd_loader::media::{ImageGrid, ImageKey, PreparedImage};
        use sha2::{Digest, Sha256};
        let tokens = vec![1, 5, 5, 6, 5, 5, 7, 2];
        let image = Arc::new(PreparedImage { key: ImageKey([7; 32]), encoder: cuteafd_loader::media::EncoderId([0; 32]), processor: cuteafd_loader::media::PreprocessId([0; 32]), grid: ImageGrid { t: 1, h: 4, w: 4 },
            rgb8: Arc::from(vec![0; 4 * 4 * 14 * 14 * 3]), tokens: 6 });
        let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true, score_from: Some(7),
            prompt_ids: Some(tokens.clone()), media: vec![ProbeMedia { start: 1, len: 6, kind: "image".into(),
                key: "07".repeat(32), grid: [1,4,4], identity: None, input_key: None, fixture: Some(ProbeFixture { path: "chart.png".into(),
                    sha256: "ab".repeat(32) }), image_url: Some(ProbeImageUrl { url: "fixture".into(), detail: None }) }],
            ..Default::default() });
        let job = NativeRequest { prefix_cache_source: Default::default(), finish_detail: Default::default(),
            literal_spans: Vec::new(),
            prompt: String::new(), constraint: None, images: Vec::new(), media: vec![image.clone()], audio: Vec::new(),
            max_tokens: 8, sampling: cuteafd_core::TargetSamplingParams::default(), stop_token_ids: Vec::new(),
            events: tokio::sync::mpsc::unbounded_channel().0, usage: None, probe: Some(probe.clone()) };
        let expander = MistralSpanExpander { image: 5, row_break: 6, end: 7, merge: 2, vocabulary: 32 };
        let (prompt, mut media, jobs) = prepare(job, tokens.clone(), Some(&expander), 32, 2, 16).unwrap();
        assert_eq!(prompt.tokens, tokens);
        assert_eq!(jobs[0].feature_bytes().unwrap(), 24);
        let root = tempfile::tempdir().unwrap();
        for file in ["config.json", "tokenizer.json", "processor_config.json", "consolidated.safetensors.index.json"] {
            std::fs::write(root.path().join(file), file).unwrap();
        }
        let identity = snapshot_identity(root.path()).unwrap();
        let payload: Vec<u8> = (0..12u16).flat_map(u16::to_le_bytes).collect();
        let metadata = serde_json::json!({"schema":"cuteafd.media.features/1", "key":"07".repeat(32),
            "grid":[1,4,4], "shape":[6,2], "dtype":"bf16-le", "tower_dtype":"bf16",
            "sha256":format!("{:x}", Sha256::digest(&payload)), "fixture_sha256":"ab".repeat(32),
            "snapshot_identity":identity});
        std::fs::write(root.path().join(format!("{}.json", "07".repeat(32))), serde_json::to_vec(&metadata).unwrap()).unwrap();
        std::fs::write(root.path().join(format!("{}.bf16", "07".repeat(32))), &payload).unwrap();
        let mut cache = cuteafd_engine::media::EmbeddingCache::new(24);
        crate::shared::probe::features::apply_probe_features(&prompt.job, &mut media, &mut cache,
            root.path(), &identity, "446a570e62b04a89ea001604ada7366baac2ea67").unwrap();
        assert!(!cache.contains(image.key));
        let mut chunk = cuteafd_engine::media::MediaChunk::default();
        media.write_chunk(0, 4, &mut chunk).unwrap();
        assert_eq!(chunk.indices, [1, 2, 3]);
        assert_eq!(chunk.features, payload[..12]);
        media.write_chunk(4, 8, &mut chunk).unwrap();
        assert_eq!(chunk.indices, [0, 1, 2]);
        assert_eq!(chunk.features, payload[12..]);
        media.write_chunk(8, 10, &mut chunk).unwrap();
        assert!(chunk.indices.is_empty());
        assert_eq!(probe.record().provenance.unwrap()["modeling_source_revision"],
            "446a570e62b04a89ea001604ada7366baac2ea67");
        std::fs::write(root.path().join("processor_config.json"), "changed").unwrap();
        assert!(crate::shared::probe::features::apply_probe_features(&prompt.job, &mut media, &mut cache,
            root.path(), &snapshot_identity(root.path()).unwrap(), "446a570e").is_err());
    }

    #[test]
    fn text_prompts_key_identically_to_their_tokens() {
        let tokens = vec![1u32, 5, 9, 2];
        let keys = cuteafd_engine::media::MediaKeys::new(&tokens, 100, &[]).unwrap();
        assert_eq!(keys.tokens(), tokens.as_slice());
        assert!(keys.spans().is_empty());
    }
}
