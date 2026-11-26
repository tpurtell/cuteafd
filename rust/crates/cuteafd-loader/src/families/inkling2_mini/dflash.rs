//! External standard GQA DFlash2 contract for Inkling Mini's visit-indexed target.
use super::Inkling2MiniConfig;
use crate::{families::{glm5::{draft_physical::{GlmDflashConfig, GlmDflashPhysicalLayout, gqa_checkpoint_arena_requests, RING},
    draft_representation::{GlmDraftCapacity, GlmDraftRepresentation}}, qwen4::Qwen4KvCache}, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Result};
use std::path::{Path, PathBuf};

pub const MODEL: &str = "incoai/Inkling-2-Mini-DFlash2";
pub const REVISION: &str = "4f9088ea66c21f2b99ed657d1941c306b03e84d2";
pub const ARENA: &str = "inkling2_mini.dflash.weights";

#[derive(Debug, Clone)]
pub struct Layout {
    pub snapshot: PathBuf,
    pub config: GlmDflashConfig,
    pub capacity: GlmDraftCapacity,
    pub physical: GlmDflashPhysicalLayout,
    pub arena: WeightArenaLayout,
    pub target: Vec<(String, usize)>,
    pub mark_bytes: usize,
    pub kv_cache: Qwen4KvCache,
}

pub fn config(snapshot: &Path, target: &Inkling2MiniConfig) -> Result<GlmDflashConfig> {
    let value = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let cfg = GlmDflashConfig::from_value(&value)?;
    ensure!(value["architectures"][0] == "DFlash2DraftModel" &&
        value["num_target_layers"].as_u64() == Some(target.total_visits() as u64) &&
        cfg.hidden == target.text.hidden && cfg.vocab == target.text.vocab_size &&
        cfg.mask_token < target.text.unpadded_vocab_size as u32 && cfg.block == 8 &&
        cfg.taps == [5,23,41,target.total_visits()] &&
        cfg.post_selector_softcap == Some(target.text.final_softcap as f32),
        "Inkling 2 Mini DFlash2 checkpoint geometry, visit taps or corrected-score cap differs from target");
    Ok(cfg)
}

pub fn mark_bytes(cfg: &GlmDflashConfig, kv: Qwen4KvCache) -> usize {
    (16 + cfg.layers * 2 * RING * kv.plane_record_bytes(cfg.kv_heads, cfg.head_dim)).div_ceil(256) * 256
}

impl Layout {
    pub fn new(snapshot: &Path, target: &Inkling2MiniConfig, slots: usize, prefill: usize,
        waves: usize, sms: usize, kv_cache: Qwen4KvCache, manifest: Option<&serde_json::Value>) -> Result<Self> {
        let config = config(snapshot, target)?;
        let capacity = GlmDraftCapacity::new(slots, slots.min(8), config.block)?;
        let representation = GlmDraftRepresentation::Bf16Only;
        let physical = GlmDflashPhysicalLayout::read(snapshot, representation, capacity, sms, 2, kv_cache)?;
        let arena = WeightArenaLayout::new(ARENA, gqa_checkpoint_arena_requests(snapshot, representation, 0, 1)?)?;
        if let Some(manifest) = manifest {
            for (k,n) in config.projection_shapes() {
                cuteafd_ffi::draft_projection::validate_package_geometry(manifest, k, n, RING)?;
            }
        }
        let (target, _) = crate::families::inkling2::draft::target_owners(config.hidden, config.taps.len(),
            slots, prefill, waves, super::execution::DECODE_ROWS, 1)?;
        let mark_bytes = mark_bytes(&config, kv_cache);
        Ok(Self { snapshot: snapshot.into(), config, capacity, physical, arena, target, mark_bytes, kv_cache })
    }
}
