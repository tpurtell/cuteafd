//! Qualification-only audio runtime loaded on the existing media owner thread.
use super::{Result, VisionError};
use cuteafd_core::DType;
use cuteafd_ffi::audio::{
    audio_geometry, AudioCodecBlock, AudioLedger, AudioPatchBlock, AudioSpec, NativeAudio,
    AUDIO_ABI, AUDIO_FP32_NUMERICS,
};
use cuteafd_loader::media::audio_tower::{AudioStorage, AudioTowerPlan};
use std::{path::Path, sync::Arc};

/// Immutable validated weight plan; no checkpoint payload is read here.
#[derive(Clone, Debug)]
pub struct AudioTowerSpec {
    plan: AudioTowerPlan,
    native: AudioSpec,
}
impl AudioTowerSpec {
    pub fn from_snapshot(snapshot: &Path, max_samples: usize) -> Result<Self> {
        let plan = AudioTowerPlan::from_snapshot(snapshot, AudioStorage::Fp32)?;
        Self::from_plan(plan, max_samples)
    }
    pub fn from_plan(plan: AudioTowerPlan, max_samples: usize) -> Result<Self> {
        audio_geometry(max_samples)?;
        if plan.storage() != AudioStorage::Fp32 {
            return Err(VisionError::Unsupported(
                "audio native qualification requires FP32 resident weights".into(),
            ));
        }
        let offset = |name: &str| -> Result<u64> {
            let read = plan.reads().get(name).ok_or_else(|| {
                VisionError::Unsupported(format!("missing native audio tensor {name}"))
            })?;
            if read.resident_dtype != DType::F32 || read.destination % 256 != 0 {
                return Err(VisionError::Unsupported(format!(
                    "invalid native audio tensor {name}"
                )));
            }
            Ok(read.destination)
        };
        let mut native = AudioSpec {
            abi_version: AUDIO_ABI,
            numerics: AUDIO_FP32_NUMERICS,
            max_samples: max_samples as u32,
            output_width: plan.output_width() as u32,
            weight_bytes: plan.weight_bytes(),
            conv1: offset("encoder.conv1.weight")?,
            conv1b: offset("encoder.conv1.bias")?,
            conv2: offset("encoder.conv2.weight")?,
            conv2b: offset("encoder.conv2.bias")?,
            downsample: offset("encoder.down_sample_layer.0.weight")?,
            norm: offset("encoder.layer_norm.weight")?,
            normb: offset("encoder.layer_norm.bias")?,
            downnorm: offset("encoder.down_sample_norm.weight")?,
            downnormb: offset("encoder.down_sample_norm.bias")?,
            patch_norm: offset("audio_encoder.input_local_transformer.norm.weight")?,
            projection1: offset("audio_encoder.projection.mlp.0.weight")?,
            projection2: offset("audio_encoder.projection.mlp.2.weight")?,
            ..Default::default()
        };
        for (i, block) in native.codec.iter_mut().enumerate() {
            let field = |suffix: &str| offset(&format!("encoder.layers.{i}.{suffix}"));
            *block = AudioCodecBlock {
                q: field("self_attn.q_proj.weight")?,
                qb: field("self_attn.q_proj.bias")?,
                k: field("self_attn.k_proj.weight")?,
                v: field("self_attn.v_proj.weight")?,
                vb: field("self_attn.v_proj.bias")?,
                o: field("self_attn.out_proj.weight")?,
                ob: field("self_attn.out_proj.bias")?,
                norm1: field("self_attn_layer_norm.weight")?,
                norm1b: field("self_attn_layer_norm.bias")?,
                norm2: field("final_layer_norm.weight")?,
                norm2b: field("final_layer_norm.bias")?,
                fc1: field("fc1.weight")?,
                fc1b: field("fc1.bias")?,
                fc2: field("fc2.weight")?,
                fc2b: field("fc2.bias")?,
            };
        }
        for i in 0..20 {
            native.codebooks[i] =
                offset(&format!("encoder.quantizer.vq.layers.{i}._codebook.embed"))?;
            native.speech[i] = offset(&format!("speech_embeddings.{i}.weight"))?;
        }
        for (i, block) in native.patch.iter_mut().enumerate() {
            let field = |suffix: &str| {
                offset(&format!(
                    "audio_encoder.input_local_transformer.layers.{i}.{suffix}"
                ))
            };
            *block = AudioPatchBlock {
                norm1: field("input_layernorm.weight")?,
                norm2: field("post_attention_layernorm.weight")?,
                q: field("self_attn.q_proj.weight")?,
                qb: field("self_attn.q_proj.bias")?,
                k: field("self_attn.k_proj.weight")?,
                kb: field("self_attn.k_proj.bias")?,
                v: field("self_attn.v_proj.weight")?,
                vb: field("self_attn.v_proj.bias")?,
                o: field("self_attn.o_proj.weight")?,
                gate: field("mlp.gate_proj.weight")?,
                up: field("mlp.up_proj.weight")?,
                down: field("mlp.down_proj.weight")?,
            };
        }
        Ok(Self { plan, native })
    }
    pub fn native(&self) -> &AudioSpec {
        &self.native
    }
    pub fn plan(&self) -> &AudioTowerPlan {
        &self.plan
    }
}

/// Shared placement/worker description; numerics stay native to the checkpoint.
pub enum BundledAudioSpec {
    Mimo(AudioTowerSpec),
    Pcm(super::pcm_audio::PcmTowerSpec),
}
impl BundledAudioSpec {
    pub fn from_snapshot(snapshot: &Path, max_samples: usize) -> Result<Self> {
        let config: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        if config["model_type"] == "muse_spark" {
            Ok(Self::Pcm(super::pcm_audio::PcmTowerSpec::from_snapshot(snapshot, max_samples)?))
        } else { Ok(Self::Mimo(AudioTowerSpec::from_snapshot(snapshot, max_samples)?)) }
    }
    pub fn required_bytes(&self, library: &Path) -> Result<u64> {
        match self { Self::Mimo(spec) => Ok(NativeAudio::required(library, spec.native())?.total_bytes()?),
            Self::Pcm(spec) => spec.required_bytes() }
    }
    pub fn geometry(&self) -> cuteafd_core::media::AudioTokenGeometry {
        match self { Self::Mimo(_) => cuteafd_core::media::AudioTokenGeometry::MimoRvq,
            Self::Pcm(_) => cuteafd_core::media::AudioTokenGeometry::PcmPatches { samples_per_token: 1920 } }
    }
    pub fn backend(&self, library: &Path) -> Result<String> {
        match self { Self::Mimo(_) => Ok(NativeAudio::backend(library)?), Self::Pcm(_) => Ok("pcm-bf16-aot-v1".into()) }
    }
    pub fn handshake(&self, revision: &str, sm: u32, backend: &str, plan_hash: [u8; 32]) -> super::remote::AudioHandshake {
        let (encoder_id, max_samples, output_width) = match self {
            Self::Mimo(spec) => (spec.plan().encoder_id(revision, sm, backend), spec.native().max_samples, spec.native().output_width),
            Self::Pcm(spec) => (spec.encoder_id(revision, sm), spec.native_spec().max_samples, spec.native_spec().output_width),
        };
        super::remote::AudioHandshake { encoder_id, max_samples, output_width, plan_hash, geometry: self.geometry() }
    }
    pub fn owner(self, admitted_bytes: u64) -> AudioOwnerConfig {
        match self { Self::Mimo(spec) => AudioOwnerConfig::Mimo { spec, admitted_bytes },
            Self::Pcm(spec) => AudioOwnerConfig::Pcm { spec, admitted_bytes } }
    }
}

/// The native export owns its transform tables; no serving sidecar is required.
pub enum AudioOwnerConfig {
    Mimo { spec: AudioTowerSpec, admitted_bytes: u64 },
    Pcm { spec: super::pcm_audio::PcmTowerSpec, admitted_bytes: u64 },
}
impl AudioOwnerConfig {
    pub fn admitted_bytes(&self) -> u64 {
        match self { Self::Mimo { admitted_bytes, .. } | Self::Pcm { admitted_bytes, .. } => *admitted_bytes }
    }
    pub fn geometry(&self) -> cuteafd_core::media::AudioTokenGeometry {
        match self { Self::Mimo { .. } => cuteafd_core::media::AudioTokenGeometry::MimoRvq,
            Self::Pcm { .. } => cuteafd_core::media::AudioTokenGeometry::PcmPatches { samples_per_token: 1920 } }
    }
    pub fn required_bytes(&self, library: &Path) -> Result<u64> {
        match self { Self::Mimo { spec, .. } => Ok(NativeAudio::required(library, spec.native())?.total_bytes()?),
            Self::Pcm { spec, .. } => spec.required_bytes() }
    }
}
pub struct AudioEncodeJob {
    pub pcm: Arc<[f32]>,
    pub output: Vec<u16>,
    pub fp32_scratch: Vec<f32>,
}
impl AudioEncodeJob {
    fn validate(&self, spec: &AudioSpec) -> Result<()> {
        let (_, tokens) = audio_geometry(self.pcm.len())?;
        if self.pcm.len() > spec.max_samples as usize
            || self.pcm.iter().any(|value| !value.is_finite())
            || self.output.len() != tokens * spec.output_width as usize
            || self.fp32_scratch.len() != self.output.len()
        {
            return Err(
                cuteafd_ffi::audio::AudioError::InvalidInput("audio job extent or PCM").into(),
            );
        }
        Ok(())
    }
}

pub(super) enum AudioRuntime {
    Mimo { native: NativeAudio, spec: AudioSpec },
    Pcm(super::pcm_audio::PcmTower),
}
impl AudioRuntime {
    pub(super) fn load(config: AudioOwnerConfig, library: &Path, device: i32) -> Result<Self> {
        let (tower, admitted_bytes) = match config {
            AudioOwnerConfig::Pcm { spec, admitted_bytes } => return Ok(Self::Pcm(spec.load(library, device, admitted_bytes)?)),
            AudioOwnerConfig::Mimo { spec, admitted_bytes } => (spec, admitted_bytes),
        };
        let spec = *tower.native();
        let required = NativeAudio::required(library, &spec)?.total_bytes()?;
        if required > admitted_bytes {
            return Err(VisionError::Unsupported(format!(
                "audio weight/scratch admission needs {required}, got {admitted_bytes}"
            )));
        }
        let weights = tower.plan.load_weights(admitted_bytes)?;
        let native = NativeAudio::load_embedded(library, &spec, device, admitted_bytes, &weights)?;
        Ok(Self::Mimo { native, spec })
    }
    pub(super) fn ledger(&self) -> Result<AudioLedger> {
        match self {
            Self::Mimo { native, .. } => Ok(native.ledger()?),
            Self::Pcm(tower) => {
                let ledger = tower.ledger()?;
                Ok(AudioLedger { weights: ledger.weights, scratch: ledger.scratch,
                    device_allocations: ledger.device_allocations, encodes: ledger.encodes, ..Default::default() })
            }
        }
    }
    pub(super) fn encode(&mut self, mut job: AudioEncodeJob) -> Result<Vec<u16>> {
        let (native, spec) = match self {
            Self::Pcm(tower) => {
                let tokens = job.pcm.len().div_ceil(1920);
                if job.output.len() != tokens * 8192 || !job.fp32_scratch.is_empty() {
                    return Err(cuteafd_ffi::audio::AudioError::InvalidInput("PCM output extent").into());
                }
                return tower.encode(&job.pcm);
            }
            Self::Mimo { native, spec } => (native, spec),
        };
        job.validate(spec)?;
        native.encode_into(&job.pcm, &mut job.fp32_scratch, None)?;
        for (output, value) in job.output.iter_mut().zip(job.fp32_scratch) {
            if !value.is_finite() {
                return Err(cuteafd_ffi::audio::AudioError::Native(-2001).into());
            }
            *output = bf16_bits(value);
        }
        Ok(job.output)
    }
}
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bf16_rows_round_to_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits(-0.0), 0x8000);
        assert_eq!(bf16_bits(f32::from_bits(0x3f808000)), 0x3f80);
        assert_eq!(bf16_bits(f32::from_bits(0x3f818000)), 0x3f82);
    }
    #[test]
    #[ignore = "requires admitted SM120/SM121 native audio library and official snapshot"]
    fn resident_audio_only_owner_matches_qualified_rows_and_drains() {
        use super::super::{EncodeJob, EncoderService};
        use sha2::{Digest, Sha256};
        use std::{fs, path::PathBuf, time::Duration};
        let snapshot = PathBuf::from(std::env::var("CUTEAFD_AUDIO_SNAPSHOT").unwrap());
        let library = PathBuf::from(std::env::var("CUTEAFD_AUDIO_LIBRARY").unwrap());
        let fixtures = PathBuf::from(std::env::var("CUTEAFD_AUDIO_OWNER_FIXTURES").unwrap());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(fixtures.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(
            NativeAudio::backend(&library).unwrap(),
            manifest["native_backend"].as_str().unwrap()
        );
        let capacity = manifest["max_samples"].as_u64().unwrap() as usize;
        let spec = AudioTowerSpec::from_snapshot(&snapshot, capacity).unwrap();
        let ledger = NativeAudio::required(&library, spec.native()).unwrap();
        let config = || AudioOwnerConfig::Mimo {
            spec: spec.clone(),
            admitted_bytes: ledger.total_bytes().unwrap(),
        };
        let denied = AudioOwnerConfig::Mimo {
            spec: spec.clone(),
            admitted_bytes: ledger.total_bytes().unwrap() - 1,
        };
        assert!(EncoderService::start_audio(library.clone(), 0, denied).is_err());
        let service = EncoderService::start_audio(library, 0, config()).unwrap();
        assert!(service.healthy());
        assert_eq!(service.ledger.device_allocations, 0);
        assert_eq!(service.audio_ledger.unwrap().device_allocations, 2);
        let cases = manifest["cases"].as_array().unwrap();
        let job = |case: &serde_json::Value| {
            let raw = fs::read(fixtures.join(case["pcm_file"].as_str().unwrap())).unwrap();
            let pcm = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let (_, tokens) = audio_geometry(pcm.len()).unwrap();
            AudioEncodeJob {
                pcm: pcm.into(),
                output: vec![0; tokens * spec.native.output_width as usize],
                fp32_scratch: vec![0.0; tokens * spec.native.output_width as usize],
            }
        };
        let collect = |ticket: super::super::EncoderTicket| {
            ticket
                .result
                .recv_timeout(Duration::from_secs(120))
                .unwrap()
        };
        // Invalid work does not poison the shared owner or make vision resident.
        let invalid = service
            .submit(EncodeJob {
                rgb: Arc::from([]),
                grid: [0, 0],
                lut: Arc::new([0.0; 768]),
                output: vec![],
            })
            .unwrap();
        assert!(matches!(collect(invalid), Err(VisionError::Unsupported(_))));
        let mut invalid_pcm = job(&cases[0]);
        invalid_pcm.pcm = vec![f32::NAN; 481].into();
        assert!(collect(service.submit_audio(invalid_pcm).unwrap()).is_err());
        assert!(service.healthy());
        for case in cases {
            let ticket = service.submit_audio(job(case)).unwrap();
            let output = collect(ticket).unwrap();
            let bytes = output
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                case["bf16_sha256"].as_str().unwrap()
            );
            assert!(service.healthy());
        }
        // Ticket cancellation may race completion, but must not damage reuse.
        let cancelled = service.submit_audio(job(&cases[0])).unwrap();
        cancelled.cancel();
        let _ = collect(cancelled);
        let again = collect(service.submit_audio(job(&cases[0])).unwrap()).unwrap();
        let bytes = again
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            cases[0]["bf16_sha256"].as_str().unwrap()
        );
        let queued = service.submit_audio(job(&cases[0])).unwrap();
        drop(service); // closes admission and drains before weight/workspace destruction.
        assert!(collect(queued).is_ok());
    }
    #[test]
    fn job_extents_and_nonfinite_pcm_fail_before_native() {
        let spec = AudioSpec {
            max_samples: 72000,
            output_width: 4096,
            ..Default::default()
        };
        let mut job = AudioEncodeJob {
            pcm: vec![0.0; 481].into(),
            output: vec![0; 4096],
            fp32_scratch: vec![0.0; 4096],
        };
        assert!(job.validate(&spec).is_ok());
        job.output.pop();
        assert!(job.validate(&spec).is_err());
        job.output.push(0);
        job.pcm = vec![f32::NAN; 481].into();
        assert!(job.validate(&spec).is_err());
    }
}
