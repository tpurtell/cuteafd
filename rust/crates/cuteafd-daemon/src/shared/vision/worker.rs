//! expertd tower startup: reserve the resident ledger before expert allocation.
use super::{normalization_lut, remote::{EncoderHandshake, EncoderServer}, EncoderService, TowerSpec};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::vision::NativeVision;
use cuteafd_loader::media::ProcessorConfig;
use std::{path::{Path, PathBuf}, time::Duration};

#[derive(Clone, Debug)]
pub struct EncoderWorkerConfig {
    pub listen: String,
    pub plan_hash: [u8; 32],
    pub revision: String,
    pub max_tokens: usize,
}
pub fn parse_plan_hash(value: &str) -> Result<[u8; 32]> {
    ensure!(value.len() == 64 && value.is_ascii(), "encoder plan hash must be 64 hexadecimal digits");
    let mut hash = [0; 32];
    for (i, byte) in hash.iter_mut().enumerate() { *byte = u8::from_str_radix(&value[i*2..i*2+2], 16).context("encoder plan hash")?; }
    Ok(hash)
}
pub(crate) fn architecture(library: &Path) -> Result<u32> {
    // SAFETY: this is the user-selected, architecture-matching native library.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load(library) }?;
    let info = library.cuda_device_info(0)?;
    Ok((info.compute_capability_major * 10 + info.compute_capability_minor) as u32)
}
/// Returns a server only after the tower owner has reported ready. The caller
/// subtracts this exact ledger before any expert weight/workspace admission.
pub fn start(config: &EncoderWorkerConfig, snapshot: &Path, library: PathBuf, budget: u64) -> Result<(EncoderServer, u64)> {
    let spec = TowerSpec::from_snapshot(snapshot, config.max_tokens)?;
    let sm = architecture(&library)?;
    let id = spec.encoder_id(&config.revision, sm);
    let ledger = NativeVision::required(&library, &spec.native)?;
    ensure!(ledger.total_bytes() <= budget, "vision tower admission needs {} bytes, budget {budget}", ledger.total_bytes());
    let processor = ProcessorConfig::from_snapshot(snapshot, spec.image_family())?;
    let handshake = EncoderHandshake { encoder_id: id, max_patches: spec.native.max_tokens * processor.merge.pow(2),
        output_width: spec.native.output_width, patch_size: processor.patch, merge_size: processor.merge, plan_hash: config.plan_hash };
    let service = EncoderService::start(spec, library, 0, ledger.total_bytes())?;
    let server = EncoderServer::start(config.listen.parse().context("encoder listen address")?, handshake, service, normalization_lut(&processor), Duration::from_secs(60))
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    tracing::info!(address = %server.address, sm, admitted_bytes = ledger.total_bytes(), "vision encoder ready");
    Ok((server, ledger.total_bytes()))
}

#[derive(Clone, Debug)]
pub struct AudioWorkerConfig {
    pub listen: String,
    pub plan_hash: [u8; 32],
    pub revision: String,
}

/// Admit both towers before either owner allocates; ports remain independent.
pub fn start_encoders(vision: Option<&EncoderWorkerConfig>, audio: Option<&AudioWorkerConfig>,
    snapshot: &Path, library: PathBuf, budget: u64) -> Result<(Option<EncoderServer>, Option<EncoderServer>, u64)> {
    if vision.is_none() && audio.is_none() { return Ok((None, None, 0)); }
    let config: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
    if config["model_type"].as_str() == Some("deepseek_v41") {
        ensure!(audio.is_none(), "V4.1 audio encoder is unsupported");
        return match vision {
            Some(config) => crate::families::deepseek_v41::v41_vision_encoder::start_worker(config, snapshot, library, budget)
                .map(|(server, bytes)| (Some(server), None, bytes)),
            None => Ok((None, None, 0)),
        };
    }
    if let (Some(vision), Some(audio)) = (vision, audio) {
        ensure!(vision.listen != audio.listen, "image/audio encoder listen addresses must differ");
    }
    let vision_bytes = vision.map(|config| -> Result<_> {
        let spec = TowerSpec::from_snapshot(snapshot, config.max_tokens)?;
        Ok(NativeVision::required(&library, &spec.native)?.total_bytes())
    }).transpose()?.unwrap_or(0);
    let audio_bytes = audio.map(|_| -> Result<_> {
        let spec = super::audio::BundledAudioSpec::from_snapshot(snapshot, cuteafd_loader::media::audio::MAX_CLIP_SAMPLES)?;
        let bytes = spec.required_bytes(&library)?;
        Ok(bytes)
    }).transpose()?.unwrap_or(0);
    let reserved = media_reservation(vision_bytes, audio_bytes, budget)?;
    ensure!(reserved <= budget, "media towers need {reserved} bytes including guard, budget {budget}");
    {
        // SAFETY: this is the selected architecture-matching native library.
        let native = unsafe { cuteafd_ffi::NativeLibrary::load(&library) }?;
        native.cuda_set_device(0)?;
        let (free, _) = native.cuda_memory_info()?;
        require_allocatable(reserved, free as u64)?;
    }
    if let (Some(image_config), Some(audio_config)) = (vision, audio) {
        let sm = architecture(&library)?;
        ensure!(sm == 121, "Spark audio worker requires SM121, got SM{sm}");
        let image_spec = TowerSpec::from_snapshot(snapshot, image_config.max_tokens)?;
        let processor = ProcessorConfig::from_snapshot(snapshot, image_spec.image_family())?;
        let image_handshake = EncoderHandshake {
            encoder_id: image_spec.encoder_id(&image_config.revision, sm),
            max_patches: image_spec.native.max_tokens * processor.merge.pow(2),
            output_width: image_spec.native.output_width, patch_size: processor.patch,
            merge_size: processor.merge, plan_hash: image_config.plan_hash,
        };
        let audio_spec = super::audio::BundledAudioSpec::from_snapshot(snapshot, cuteafd_loader::media::audio::MAX_CLIP_SAMPLES)?;
        let backend = audio_spec.backend(&library)?;
        let audio_handshake = audio_spec.handshake(&audio_config.revision, sm, &backend, audio_config.plan_hash);
        let service = std::sync::Arc::new(EncoderService::start_with_audio(image_spec, library, 0, vision_bytes,
            Some(audio_spec.owner(audio_bytes)))?);
        let image = EncoderServer::start_shared(image_config.listen.parse().context("encoder listen address")?,
            image_handshake, service.clone(), normalization_lut(&processor), Duration::from_secs(60))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let audio = EncoderServer::start_audio_shared(audio_config.listen.parse().context("audio encoder listen address")?,
            audio_handshake, service, Duration::from_secs(60)).map_err(|e| anyhow::anyhow!("{e}"))?;
        tracing::info!(address = %audio.address, sm, admitted_bytes = audio_bytes,
            reserved_bytes = reserved, backend = %backend, "audio encoder ready");
        return Ok((Some(image), Some(audio), reserved));
    }
    let image = vision.map(|config| start(config, snapshot, library.clone(), budget).map(|(server, _)| server)).transpose()?;
    let audio = audio.map(|config| start_audio(config, snapshot, library.clone(), budget).map(|(server, _)| server)).transpose()?;
    Ok((image, audio, reserved))
}

pub(crate) fn check_cuda_room(library: &Path, device: i32, required: u64) -> Result<()> {
    // SAFETY: selected architecture-matching native library, queried before tower allocation.
    let native = unsafe { cuteafd_ffi::NativeLibrary::load(library) }?;
    native.cuda_set_device(device)?;
    let (free, _) = native.cuda_memory_info()?;
    require_allocatable(required, free as u64)
}

pub(crate) fn require_allocatable(required: u64, free: u64) -> Result<()> {
    ensure!(required <= free,
        "media tower allocation needs {required} bytes but only {free} allocatable bytes remain; \
        GB10 uses MemAvailable including reclaimable file cache, discrete RTX uses cudaMemGetInfo; \
        free memory or choose another encoder owner");
    Ok(())
}

const MEDIA_GUARD_BYTES: u64 = 1 << 30;

fn media_reservation(vision: u64, audio: u64, budget: u64) -> Result<u64> {
    let admitted = vision.checked_add(audio).context("media reservation overflow")?;
    if admitted == 0 { return Ok(0); }
    // One per worker: spare room protects expert admission beside either tower.
    let reserved = admitted.checked_add(MEDIA_GUARD_BYTES).context("media reservation overflow")?;
    ensure!(reserved <= budget,
        "media towers need {admitted} bytes plus 1 GiB guard, budget {budget}");
    Ok(reserved)
}

/// Returns weights/scratch plus the Spark guard to charge before experts load.
pub fn start_audio(config: &AudioWorkerConfig, snapshot: &Path, library: PathBuf, budget: u64) -> Result<(EncoderServer, u64)> {
    let spec = super::audio::BundledAudioSpec::from_snapshot(snapshot, cuteafd_loader::media::audio::MAX_CLIP_SAMPLES)?;
    let sm = architecture(&library)?;
    ensure!(sm == 121, "Spark audio worker requires SM121, got SM{sm}");
    let admitted = spec.required_bytes(&library)?;
    let reserved = media_reservation(0, admitted, budget)?;
    let backend = spec.backend(&library)?;
    let handshake = spec.handshake(&config.revision, sm, &backend, config.plan_hash);
    let service = EncoderService::start_audio(library, 0, spec.owner(admitted))?;
    let server = EncoderServer::start_audio(config.listen.parse().context("audio encoder listen address")?,
        handshake, service, Duration::from_secs(60)).map_err(|e| anyhow::anyhow!("{e}"))?;
    tracing::info!(address = %server.address, sm, admitted_bytes = admitted, reserved_bytes = reserved,
        backend = %backend, "audio encoder ready");
    Ok((server, reserved))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn media_reservation_keeps_one_guard_outside_expert_budget() {
        assert_eq!(media_reservation(7, 0, MEDIA_GUARD_BYTES + 7).unwrap(), MEDIA_GUARD_BYTES + 7);
        assert_eq!(media_reservation(3, 4, MEDIA_GUARD_BYTES + 7).unwrap(), MEDIA_GUARD_BYTES + 7);
        assert_eq!(media_reservation(0, 0, 0).unwrap(), 0);
        assert_eq!(media_reservation(0, 7, MEDIA_GUARD_BYTES + 7).unwrap(), MEDIA_GUARD_BYTES + 7);
        assert!(media_reservation(0, 7, MEDIA_GUARD_BYTES + 6).is_err());
        assert!(media_reservation(0, u64::MAX, u64::MAX).is_err());
    }
    #[test]
    fn tower_preflight_requires_allocatable_memory() {
        assert!(require_allocatable(3 << 30, 3 << 30).is_ok());
        assert!(require_allocatable(0, 0).is_ok());
        let error = require_allocatable(3 << 30, 2 << 30).unwrap_err().to_string();
        for text in ["cudaMemGetInfo", "allocatable", "MemAvailable", "file cache", "another encoder owner"] {
            assert!(error.contains(text), "{error}");
        }
    }

    #[test]
    fn hash_parser_fails_closed() {
        assert_eq!(parse_plan_hash(&"ab".repeat(32)).unwrap(), [0xab; 32]);
        for bad in ["f".repeat(63), "z".repeat(64), "é".repeat(32)] { assert!(parse_plan_hash(&bad).is_err()); }
    }
}
