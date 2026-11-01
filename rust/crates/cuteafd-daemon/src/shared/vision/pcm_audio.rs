//! Shared encoder-thread owner for the checkpoint's causal BF16 PCM pyramid.
use super::{Result, VisionError};
use cuteafd_ffi::pcm_audio::{NativePcmAudio, PcmBlock, PcmLedger, PcmSpec, PcmStage};
use cuteafd_loader::media::pcm_audio::{scratch_bytes, PcmAudioPlan};
use std::path::Path;
#[derive(Clone, Debug)]
pub struct PcmTowerSpec {
    plan: PcmAudioPlan,
    native: PcmSpec,
}
impl PcmTowerSpec {
    pub fn from_snapshot(snapshot: &Path, max_samples: usize) -> Result<Self> {
        let plan = PcmAudioPlan::from_snapshot(snapshot)?;
        scratch_bytes(max_samples)?;
        let offset = |key: &str| -> Result<u64> {
            plan.reads()
                .get(key)
                .map(|r| r.destination)
                .ok_or_else(|| VisionError::Unsupported(format!("missing PCM tensor {key}")))
        };
        let mut native = PcmSpec {
            abi_version: 1,
            max_samples: max_samples as u32,
            output_width: 8192,
            weight_bytes: plan.weight_bytes(),
            norm: offset("norm.weight")?,
            projection: offset("proj.weight")?,
            bias: offset("proj.bias")?,
            post_norm: offset("post_proj_norm.weight")?,
            ..Default::default()
        };
        for (i, stage) in native.stages.iter_mut().enumerate() {
            let field = |suffix: &str| offset(&format!("layers.{i}.{suffix}"));
            *stage = PcmStage {
                projection: field("patch_embedder.proj.weight")?,
                bias: field("patch_embedder.proj.bias")?,
                norm: field("patch_embedder.norm.weight")?,
                ..Default::default()
            };
            for (j, block) in stage.blocks.iter_mut().enumerate() {
                let f = |suffix: &str| field(&format!("conv_blocks.{j}.{suffix}"));
                *block = PcmBlock {
                    conv: f("conv.weight")?,
                    conv_bias: f("conv.bias")?,
                    norm: f("norm.weight")?,
                    fc1: f("mlp.fc1.weight")?,
                    fc1_bias: f("mlp.fc1.bias")?,
                    fc2: f("mlp.fc2.weight")?,
                    fc2_bias: f("mlp.fc2.bias")?,
                };
            }
        }
        Ok(Self { plan, native })
    }
    pub fn required_bytes(&self) -> Result<u64> {
        Ok(self.plan.required_bytes(self.native.max_samples as usize)?)
    }
    pub fn native_spec(&self) -> &PcmSpec {
        &self.native
    }
    pub fn encoder_id(&self, revision: &str, sm: u32) -> cuteafd_loader::media::EncoderId {
        self.plan.encoder_id(revision, sm)
    }
    /// Load on the shared media owner's CUDA thread after planner admission.
    pub fn load(&self, library: &Path, device: i32, admitted: u64) -> Result<PcmTower> {
        let required = self.required_bytes()?;
        if admitted < required {
            return Err(VisionError::Unsupported(format!(
                "PCM admission {admitted} < {required}"
            )));
        }
        let weights = self.plan.read_weights(self.plan.weight_bytes())?;
        let native = NativePcmAudio::load(library, &self.native, device, admitted, &weights)?;
        let ledger = native.ledger()?;
        if ledger.weights != self.plan.weight_bytes()
            || ledger.scratch != scratch_bytes(self.native.max_samples as usize)?
            || ledger.device_allocations != 2
        {
            return Err(VisionError::Unsupported(
                "PCM ledger differs from admitted plan".into(),
            ));
        }
        Ok(PcmTower { native })
    }
}
pub struct PcmTower {
    native: NativePcmAudio,
}
impl PcmTower {
    pub fn encode(&mut self, pcm: &[f32]) -> Result<Vec<u16>> {
        Ok(self.native.encode(pcm)?)
    }
    pub fn ledger(&self) -> Result<PcmLedger> {
        Ok(self.native.ledger()?)
    }
}
