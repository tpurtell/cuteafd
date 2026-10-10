//! CPU-only external drafter geometry shared by admission and allocation.
use anyhow::{ensure, Context, Result};
use std::path::Path;
use super::draft_representation::{MimoDraftCapacity, MimoDraftGeometry, MimoDraftRepresentation, MimoDraftRuntimeLayout, MimoDraftWeightLayout};
pub const RING: usize = 1024;
pub const TAP_ROWS: usize = RING;
const FP8_ROWS: usize = 128;

#[derive(Debug, Clone, serde::Deserialize)]
struct RawConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default)]
    v_head_dim: Option<usize>,
    #[serde(default = "one")]
    partial_rotary_factor: f64,
    rope_theta: f64,
    rms_norm_eps: f64,
    sliding_window: usize,
    vocab_size: usize,
    #[serde(default)]
    is_causal: bool,
    block_size: usize,
    dflash_config: RawDflash,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RawDflash {
    target_layer_ids: Vec<usize>,
    mask_token_id: u32,
    #[serde(default)]
    attention_value_scale: Option<f64>,
    #[serde(default)]
    attention_sink_bias: bool,
}

/// The drafter geometry the kernels are written for, read from `dflash/config.json`.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub theta: f32,
    pub eps: f32,
    pub block: usize,
    pub mask_token: u32,
    pub taps: Vec<usize>,
    pub vocab: usize,
    pub window: usize,
    pub v_scale: f32,
    pub sinks: bool,
}

impl DflashConfig {
    pub fn read(dir: &Path) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .context("parsing dflash/config.json")?;
        let d = &raw.dflash_config;
        let rope_dim = (raw.head_dim as f64 * raw.partial_rotary_factor) as usize;
        ensure!(raw.head_dim == 128 && raw.v_head_dim.unwrap_or(128) == 128 && matches!(rope_dim, 64 | 128)
            && raw.num_key_value_heads > 0 && raw.num_attention_heads % raw.num_key_value_heads == 0 && !raw.is_causal,
            "MiMo DFlash kernels take 128-wide heads, RoPE on 64 or 128 dims and non-causal blocks");
        ensure!(raw.sliding_window == RING, "DFlash window {} (the ring holds {RING})", raw.sliding_window);
        Ok(Self {
            hidden: raw.hidden_size,
            intermediate: raw.intermediate_size,
            layers: raw.num_hidden_layers,
            heads: raw.num_attention_heads,
            kv_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rope_dim,
            theta: raw.rope_theta as f32,
            eps: raw.rms_norm_eps as f32,
            block: raw.block_size,
            mask_token: d.mask_token_id,
            taps: d.target_layer_ids.clone(),
            vocab: raw.vocab_size,
            window: raw.sliding_window,
            v_scale: d.attention_value_scale.unwrap_or(1.0) as f32,
            sinks: d.attention_sink_bias,
        })
    }

    pub fn drafts(&self) -> usize {
        self.block - 1
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub fn qkv_width(&self) -> usize {
        (self.heads + 2 * self.kv_heads) * self.head_dim
    }

    fn weight_geometry(&self) -> MimoDraftGeometry {
        MimoDraftGeometry { hidden: self.hidden as u64,
            intermediate: self.intermediate as u64, layers: self.layers as u64, heads: self.heads as u64,
            kv_heads: self.kv_heads as u64, head_dim: self.head_dim as u64, taps: self.taps.len() as u64,
            vocab: self.vocab as u64, sinks: self.sinks }
    }

    pub fn weight_layout(&self, mode: MimoDraftRepresentation) -> Result<MimoDraftWeightLayout> {
        Ok(MimoDraftWeightLayout::new(self.weight_geometry(), mode)?)
    }

    /// The admission consumer and allocator use the same selected mode/rows.
    pub fn runtime_layout(&self, mode: MimoDraftRepresentation, capacity: MimoDraftCapacity)
        -> Result<MimoDraftRuntimeLayout> {
        Ok(MimoDraftRuntimeLayout::new(self.weight_geometry(), mode, capacity, TAP_ROWS, FP8_ROWS)?)
    }

    /// One additional activation bank; weights and context-update scratch are shared.
    pub fn prefill_lane_tap_bytes(&self) -> Result<usize> {
        TAP_ROWS.checked_mul(self.taps.len()).and_then(|n| n.checked_mul(self.hidden))
            .and_then(|n| n.checked_mul(2)).map(|n| n.max(256))
            .context("paired prefill tap bank size overflow")
    }
}
