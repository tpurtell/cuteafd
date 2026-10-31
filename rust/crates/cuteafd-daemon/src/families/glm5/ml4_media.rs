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
    /// The resident tower on its RTX (`--vision rtx`, or auto without Spark experts).
    /// A Spark-hosted encoder (planner-resolved `--vision-peers`) is not wired for
    /// Mistral Large 4 yet: its span rows need their own remote handshake geometry.
    pub fn load(args: &super::EngineArgs, library: &cuteafd_ffi::NativeLibrary, mode: MediaMode,
        prefix: &crate::shared::prefix::PrefixArgs, cache_bytes: Option<u64>)
        -> Result<(Option<Self>, crate::shared::prefix::PrefixArgs)> {
        // The planner's rule: automatic vision with Spark experts serves text only.
        let resolved = cuteafd_loader::families::mistral_large4::vision::auto_vision(mode, args.peers.is_some());
        if resolved != mode {
            tracing::warn!("Mistral Large 4 VISION=auto with Spark experts serves text only: a Spark-hosted \
                Pixtral encoder is not wired yet; set VISION=rtx to serve images from the coordinator");
        }
        let mode = resolved;
        let Some(config) = vision_config(mode, &args.snapshot)? else { return Ok((None, prefix.clone())); };
        let gpu = match mode {
            MediaMode::Rtx(_) | MediaMode::Auto =>
                crate::families::glm5_flash::media::local_vision_gpu(mode, args.device, args.split_device)?,
            MediaMode::Spark(_) => anyhow::bail!("Mistral Large 4 Spark-hosted vision is not wired yet; set VISION=rtx"),
            MediaMode::Off => unreachable!(),
        };
        let processor = ProcessorConfig::from_snapshot(&args.snapshot, ImageFamily::Mistral)?;
        let markers = crate::shared::vision::mistral::Markers::from_snapshot(&args.snapshot)
            .map_err(anyhow::Error::msg)?;
        let vocabulary = u32::try_from(config["text_config"]["vocab_size"].as_u64().context("vocab_size")?)?;
        let expander = MistralSpanExpander { image: markers.image, row_break: markers.row_break, end: markers.end,
            merge: processor.merge, vocabulary };
        let spec = crate::shared::vision::TowerSpec::mistral_large4(&args.snapshot,
            crate::shared::vision::mistral::merged_capacity(processor.max_image_tokens))?;
        let width = crate::shared::vision::mistral::LM_HIDDEN;
        let (prefix, cache_bytes) = prefix.with_media_headroom(cache_bytes)?;
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
            anyhow::ensure!(job.probe.as_ref().is_none_or(|p| p.spec.prompt_ids.is_none() || job.media.is_empty()),
                "Mistral Large 4 probes with explicit prompt ids do not carry images");
            let images = job.media.iter().map(|image| image.as_ref().clone()).collect::<Vec<_>>();
            let expanded = expander.expand(&tokens, &images, max_context)?;
            (expanded.tokens, expanded.media)
        }
        None => {
            anyhow::ensure!(job.media.is_empty(), "this deployment serves no images (vision off or no tower)");
            (tokens, Vec::new())
        }
    };
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let keys = MediaKeys::new(&tokens, vocabulary as u32, &spans)?;
    let jobs = job.media.iter().map(|image| EncodeJob::image(image.key, [image.grid.t, image.grid.h, image.grid.w],
        image.rgb8.clone(), image.tokens, hidden)).collect();
    Ok((Prompt { job, tokens, keys }, media, jobs))
}

#[cfg(test)]
mod tests {
    #[test]
    fn text_prompts_key_identically_to_their_tokens() {
        let tokens = vec![1u32, 5, 9, 2];
        let keys = cuteafd_engine::media::MediaKeys::new(&tokens, 100, &[]).unwrap();
        assert_eq!(keys.tokens(), tokens.as_slice());
        assert!(keys.spans().is_empty());
    }
}
