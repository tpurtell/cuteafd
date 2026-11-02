//! Immutable GLM/GLM Flash DFlash storage and its device-loading contract.
//! The target owns its one vocabulary head (BF16, or GLM 5.3 Flash's FP8-only
//! head with --fp8-head); the drafter borrows it and never copies it.
use thiserror::Error;

/// Serving and standalone admission use the same drafter batch default.
pub const DEFAULT_DRAFT_SEQUENCES: usize = 16;

/// Default scheduler concurrency, shared with standalone admission.
pub const DEFAULT_MAX_SEQUENCES: usize = 16;

pub fn serving_draft_context_slots(sequences: usize, concurrency: usize, explicit: Option<usize>) -> usize {
    explicit.unwrap_or_else(|| 20.max(sequences).max(concurrency.saturating_mul(5).div_ceil(4)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmDraftRepresentation {
    Bf16Only,
    Fp8Only,
}

impl GlmDraftRepresentation {
    /// Current DFlash readers support BF16 checkpoint weights. Quantization
    /// is an explicit convenience choice, independent of activation formats.
    pub fn from_fp8_option(fp8: Option<bool>) -> Self {
        // Drafter precision cannot change committed tokens; FP8 drafts measured faster
        // (GLM 5.3, 1 RTX + 4 Sparks: C1 code 41.5 -> 43.9, C4 63.9 -> 71.1 tok/s).
        if fp8 == Some(false) { Self::Bf16Only } else { Self::Fp8Only }
    }

    pub fn name(self) -> &'static str {
        match self { Self::Bf16Only => "BF16", Self::Fp8Only => "FP8" }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmDraftCapacity {
    pub context_slots: usize,
    pub max_batch_sequences: usize,
    pub block_rows: usize,
}

impl GlmDraftCapacity {
    pub fn new(context_slots: usize, max_batch_sequences: usize, block: usize)
        -> Result<Self, GlmDraftStorageError> {
        if context_slots == 0 || max_batch_sequences == 0 || max_batch_sequences > context_slots || block < 2 {
            return Err(GlmDraftStorageError::Unsupported("draft batch must fit nonzero context slots and block >= 2"));
        }
        // Native ring destinations use signed32 row indices (-1 means skip).
        if mul(context_slots as u64, 2048)? > i32::MAX as u64 + 1 {
            return Err(GlmDraftStorageError::Unsupported("context slots exceed native ring row indices"));
        }
        let block_rows = max_batch_sequences.checked_mul(block).ok_or(GlmDraftStorageError::Overflow)?;
        Ok(Self { context_slots, max_batch_sequences, block_rows })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GlmDraftGeometry {
    pub hidden: u64,
    pub intermediate: u64,
    pub layers: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub taps: u64,
    pub vocab: u64,
    pub conv_group: u64,
    pub selector_rank: u64,
}

/// One owned GEMM `[n,k]`; the target vocabulary head is absent by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmDraftLinearShape {
    pub k: u64,
    pub n: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlmDraftWeightLayout {
    pub bf16_values: u64,
    pub fp8_values: u64,
    pub fp8_scales: u64,
    /// Norms and two-tap convolution base kernels, preserving BF16 values.
    pub bf16_auxiliary: u64,
    /// Distinct learned selector codebooks; they are not GEMM copies.
    pub bf16_codebooks: u64,
    /// One BF16 source matrix while packing. Drain before releasing it.
    pub max_load_staging: u64,
}

impl GlmDraftWeightLayout {
    pub fn resident_bytes(&self) -> Result<u64, GlmDraftStorageError> {
        sum([self.bf16_values, self.fp8_values, self.fp8_scales, self.bf16_auxiliary, self.bf16_codebooks])
    }

    pub fn loading_peak_bytes(&self) -> Result<u64, GlmDraftStorageError> {
        add(self.resident_bytes()?, self.max_load_staging)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmDraftFp8ScratchLayout {
    pub rows: usize,
    pub shapes: Vec<GlmDraftLinearShape>,
}

/// Admission and loading consume this same checked weight/scratch contract.
/// Ring, attention and activation arenas remain separately sized from capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmDraftRuntimeLayout {
    pub capacity: GlmDraftCapacity,
    pub weights: GlmDraftWeightLayout,
    pub fp8_scratch: Option<GlmDraftFp8ScratchLayout>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GlmDraftStorageError {
    #[error("unsupported GLM DFlash storage: {0}")]
    Unsupported(&'static str),
    #[error("GLM DFlash storage byte count overflow")]
    Overflow,
}

fn mul(a: u64, b: u64) -> Result<u64, GlmDraftStorageError> {
    a.checked_mul(b).ok_or(GlmDraftStorageError::Overflow)
}
fn add(a: u64, b: u64) -> Result<u64, GlmDraftStorageError> {
    a.checked_add(b).ok_or(GlmDraftStorageError::Overflow)
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64, GlmDraftStorageError> {
    values.into_iter().try_fold(0, add)
}

impl GlmDraftRuntimeLayout {
    pub fn new(g: GlmDraftGeometry, mode: GlmDraftRepresentation, capacity: GlmDraftCapacity,
        tap_rows: usize) -> Result<Self, GlmDraftStorageError> {
        if [g.hidden, g.intermediate, g.layers, g.heads, g.kv_heads, g.head_dim, g.taps, g.vocab,
            g.conv_group, g.selector_rank].contains(&0) || g.heads % g.kv_heads != 0
            || g.hidden % g.conv_group != 0 || tap_rows == 0 {
            return Err(GlmDraftStorageError::Unsupported("invalid drafter geometry"));
        }
        if !matches!(g.head_dim, 128 | 64) || g.selector_rank != 256 {
            return Err(GlmDraftStorageError::Unsupported("kernels require head_dim 128 or 64 and selector rank 256"));
        }
        let h = g.hidden;
        let attention = mul(g.heads, g.head_dim)?;
        let kv = mul(g.kv_heads, g.head_dim)?;
        let conv = mul(4, h)? / g.conv_group;
        let fc = GlmDraftLinearShape { k: mul(g.taps, h)?, n: h };
        let projection = GlmDraftLinearShape { k: h, n: g.selector_rank };
        let layer_shapes = [
            GlmDraftLinearShape { k: h, n: conv },
            GlmDraftLinearShape { k: h, n: add(attention, mul(2, kv)?)? },
            GlmDraftLinearShape { k: attention, n: h },
            GlmDraftLinearShape { k: h, n: conv },
            GlmDraftLinearShape { k: h, n: mul(2, g.intermediate)? },
            GlmDraftLinearShape { k: g.intermediate, n: h },
        ];
        let values = sum([mul(fc.k, fc.n)?, mul(projection.k, projection.n)?,
            mul(g.layers, sum(layer_shapes.iter().map(|s| mul(s.k, s.n)).collect::<Result<Vec<_>, _>>()?)?)?])?;
        // Two layer norms, two [2,2,hidden] base kernels and Q/K head norms.
        let bf16_auxiliary = mul(2, sum([mul(2, h)?,
            mul(g.layers, add(mul(10, h)?, mul(2, g.head_dim)?)?)?])?)?;
        let bf16_codebooks = mul(4, mul(g.vocab, g.selector_rank)?)?;
        let uses_fp8 = mode == GlmDraftRepresentation::Fp8Only;
        let mut shapes = vec![fc, projection];
        shapes.extend(layer_shapes);
        // Context updates consume only the K|V row slice of packed QKV.
        shapes.push(GlmDraftLinearShape { k: h, n: mul(2, kv)? });
        if uses_fp8 && shapes.iter().any(|s| s.n % 16 != 0 || s.k % 128 != 0) {
            return Err(GlmDraftStorageError::Unsupported("FP8 GEMMs require N multiples of 16 and K multiples of 128"));
        }
        let max_load_staging = if uses_fp8 {
            mul(2, shapes.iter().map(|s| mul(s.k, s.n)).collect::<Result<Vec<_>, _>>()?
                .into_iter().max().unwrap_or(0))?
        } else { 0 };
        let weights = GlmDraftWeightLayout {
            bf16_values: if uses_fp8 { 0 } else { mul(2, values)? },
            fp8_values: if uses_fp8 { values } else { 0 },
            fp8_scales: if uses_fp8 { mul(values / 128, 4)? } else { 0 },
            bf16_auxiliary, bf16_codebooks, max_load_staging,
        };
        Ok(Self { capacity, weights, fp8_scratch: uses_fp8.then_some(GlmDraftFp8ScratchLayout {
            rows: tap_rows.max(capacity.block_rows), shapes,
        }) })
    }
}

/// Native fp8_gemv.cu scratch contract; SM count affects the split partial arena.
pub fn draft_fp8_scratch_bytes(rows: u64, k: u64, n: u64, sms: u64, mode: u8) -> u64 {
    let align = |bytes: u64| bytes.div_ceil(256) * 256;
    if n < 16 || k < 128 || sms == 0 || mode > 2 { return 0; }
    let tiles = n / 16;
    let blocks = k / 128;
    let wanted = (sms * 16).div_ceil(tiles).clamp(1, blocks);
    let splits = blocks.div_ceil(blocks.div_ceil(wanted));
    let capacity = if mode == 0 { 64 } else { 128 };
    align(2 * capacity * 4) + align(k.div_ceil(256) * capacity * 4) + align(k * capacity * 2)
        + if mode == 2 { align(k / 128 * capacity * 4) } else { 0 }
        + if splits > 1 { align(splits * rows.min(capacity) * n * 4) } else { 0 }
}

/// SM-independent owned arenas and candidate-local linear/attention scratch at readiness.
/// The target head and its mask row are borrowed.
pub fn draft_workspace_bytes(g: GlmDraftGeometry, capacity: GlmDraftCapacity, block: u64,
    fp8: bool, sms: u64, mode: u8, kv_cache: crate::families::qwen4::Qwen4KvCache) -> Result<(u64, u64), GlmDraftStorageError> {
    // Validate before arithmetic, including hostile config.json values. Native kernels and
    // allocation indices are signed32; the additional bound keeps all arena products in u64.
    if [g.hidden, g.intermediate, g.layers, g.heads, g.kv_heads, g.head_dim, g.taps, g.vocab,
        g.conv_group, g.selector_rank, block].iter().any(|&n| n == 0 || n > 1_000_000)
        || g.layers > 1024 || g.heads > 4096 || g.kv_heads > 4096 || g.taps > 1024
        || capacity.max_batch_sequences > 32 || block > 128 || mode > 2 || sms == 0 || sms > 4096 {
        return Err(GlmDraftStorageError::Unsupported("draft geometry exceeds native arena limits"));
    }
    GlmDraftRuntimeLayout::new(g, if fp8 { GlmDraftRepresentation::Fp8Only } else { GlmDraftRepresentation::Bf16Only }, capacity, 2048)?;
    let allocation = |n: u64| n.max(256);
    let sequences = capacity.max_batch_sequences as u64;
    let rows = sequences * block;
    let drafted = sequences * (block - 1);
    let kv = g.kv_heads * g.head_dim;
    let attention = g.heads * g.head_dim;
    let conv = mul(4, g.hidden)? / g.conv_group;
    let owned = add(mul(mul(mul(2,g.layers)?,capacity.context_slots as u64)?,mul(2048, kv_cache.plane_record_bytes(g.kv_heads as usize, g.head_dim as usize) as u64)?)?,
        mul(2048,sum([mul(mul(g.taps,g.hidden)?,2)?,mul(g.hidden,4)?,mul(kv,4)?,12])?)?)?;
    let workspace = [4 << 20, rows * g.hidden * 2, rows * g.hidden * 2, rows * g.hidden * 2,
        rows * conv * 2, rows * (attention + 2 * kv) * 2, rows * attention * 2,
        rows * kv * 2, rows * kv * 2, rows * attention * 2, rows * g.hidden * 2,
        rows * 2 * g.intermediate * 2, rows * g.intermediate * 2, rows * g.vocab * 4,
        drafted * 16 * 4, drafted * 16 * 4, rows * g.selector_rank * 2, sequences * 4,
        rows * 4, drafted * 4, drafted * 16, rows * 8, 3 * sequences * 4,
        drafted * 64 * 16 * 8].into_iter().map(allocation).sum::<u64>();
    let scratch = if fp8 {
        let layout = GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Fp8Only, capacity, 2048)?;
        layout.fp8_scratch.unwrap().shapes.iter().map(|s|
            draft_fp8_scratch_bytes(2048.max(rows), s.k, s.n, sms, mode)).max().unwrap_or(256)
    } else { 0 };
    let attention_scratch = crate::serving_capacity::draft_attention_workspace_bytes(sequences,
        g.kv_heads, 2, 2048 + block, g.head_dim, sms).map_err(|_| GlmDraftStorageError::Overflow)?;
    // Both SM-sized arenas follow the selected GPU, even for BF16 projections.
    Ok((add(owned, workspace)?, add(scratch, allocation(attention_scratch))?))
}

pub fn is_dspark(config: &serde_json::Value) -> bool {
    config["speculators_model_type"] == "dspark" || config["architectures"].as_array()
        .is_some_and(|a| a.iter().any(|v| v == "DSparkDraftModel"))
}

/// Serving owns one draft ring per scheduler slot unless explicitly overridden.
pub fn flash_serving_draft_limits(max_sequences: usize, sequences: usize, slots: Option<usize>)
    -> (Option<usize>, usize) {
    match slots {
        Some(slots) => (Some(slots), sequences),
        None => { let slots = max_sequences.max(1); (Some(slots), sequences.min(slots)) }
    }
}

/// Flash planning, admission and loading share one checkpoint-derived head row capacity.
pub fn flash_draft_capacity(config: &serde_json::Value, context_slots: Option<usize>, sequences: usize)
    -> Result<GlmDraftCapacity, GlmDraftStorageError> {
    let slots = context_slots.unwrap_or(20.max(sequences));
    let (block, batch) = if is_dspark(config) {
        let cfg = crate::formats::block_dspark::BlockDsparkConfig::from_json(config)
            .map_err(|_| GlmDraftStorageError::Unsupported("invalid dSpark checkpoint"))?;
        (cfg.block, sequences.min(slots).min(32))
    } else {
        let (_, block) = draft_geometry(config)?;
        (usize::try_from(block).map_err(|_| GlmDraftStorageError::Overflow)?, sequences.min(slots))
    };
    GlmDraftCapacity::new(slots, batch, block)
}

pub fn draft_geometry(config: &serde_json::Value) -> Result<(GlmDraftGeometry, u64), GlmDraftStorageError> {
    let int = |value: &serde_json::Value, name: &str| value[name].as_u64()
        .ok_or(GlmDraftStorageError::Unsupported("missing draft geometry"));
    let d = &config["dflash_config"];
    Ok((GlmDraftGeometry {
        hidden: int(config, "hidden_size")?, intermediate: int(config, "intermediate_size")?,
        layers: int(config, "num_hidden_layers")?, heads: int(config, "num_attention_heads")?,
        kv_heads: int(config, "num_key_value_heads")?, head_dim: int(config, "head_dim")?,
        taps: d["target_layer_ids"].as_array().ok_or(GlmDraftStorageError::Unsupported("missing draft taps"))?.len() as u64,
        vocab: int(config, "vocab_size")?, conv_group: int(d, "conv_group_size")?,
        selector_rank: int(d, "selector_rank")?,
    }, int(d, "block_size")?))
}

pub fn draft_resident_bytes(config: &serde_json::Value, slots: usize, sequences: usize, sms: u64, kv_cache: crate::families::qwen4::Qwen4KvCache)
    -> Result<(u64, u64), GlmDraftStorageError> {
    draft_resident_bytes_with_mode(config, slots, sequences, sms, GlmDraftRepresentation::Fp8Only, 2, kv_cache)
}

pub fn draft_resident_bytes_with_mode(config: &serde_json::Value, slots: usize, sequences: usize, sms: u64,
    representation: GlmDraftRepresentation, mode: u8, kv_cache: crate::families::qwen4::Qwen4KvCache) -> Result<(u64, u64), GlmDraftStorageError> {
    if mode > 2 || sms == 0 || sms > 4096 { return Err(GlmDraftStorageError::Unsupported("invalid draft scratch mode or SM count")); }
    if is_dspark(config) {
        return dspark_resident_bytes(config, slots, sequences.min(32), sms, representation, mode, kv_cache);
    }
    let (g, block) = draft_geometry(config)?;
    let capacity = GlmDraftCapacity::new(slots, sequences, usize::try_from(block).map_err(|_| GlmDraftStorageError::Overflow)?)?;
    let layout = GlmDraftRuntimeLayout::new(g, representation, capacity, 2048)?;
    let (workspace, scratch) = draft_workspace_bytes(g, capacity, block,
        representation == GlmDraftRepresentation::Fp8Only, sms, mode, kv_cache)?;
    Ok((add(layout.weights.resident_bytes()?, workspace)?, scratch))
}

fn dspark_resident_bytes(config: &serde_json::Value, slots: usize, sequences: usize, sms: u64,
    representation: GlmDraftRepresentation, mode: u8, kv_cache: crate::families::qwen4::Qwen4KvCache) -> Result<(u64, u64), GlmDraftStorageError> {
    block_dspark_resident_bytes(config, slots, sequences, sms, representation, mode, 2048, kv_cache)
}

/// Shared dSpark inventory for an explicitly admitted context capacity. The
/// target embedding/head remain borrowed, never charged as owned draft weights.
#[allow(clippy::too_many_arguments)]
pub fn block_dspark_resident_bytes(config: &serde_json::Value, slots: usize, sequences: usize, sms: u64,
    representation: GlmDraftRepresentation, mode: u8, capacity: usize, kv_cache: crate::families::qwen4::Qwen4KvCache)
    -> Result<(u64, u64), GlmDraftStorageError> {
    if mode > 2 || sms == 0 || sms > 4096 {
        return Err(GlmDraftStorageError::Unsupported("invalid draft scratch mode or SM count"));
    }
    let cfg = crate::formats::block_dspark::BlockDsparkConfig::from_json(config)
        .map_err(|_| GlmDraftStorageError::Unsupported("invalid dSpark checkpoint"))?;
    if capacity == 0 || cfg.window.is_some_and(|window| capacity != window)
        || capacity.checked_add(cfg.block).is_none_or(|keys| keys > i32::MAX as usize)
        || slots.checked_mul(capacity).is_none_or(|rows| rows > i32::MAX as usize) {
        return Err(GlmDraftStorageError::Unsupported("invalid dSpark context admission"));
    }
    let (h, inter, layers, heads, kvh, dim, vocab) = (cfg.hidden as u64, cfg.intermediate as u64,
        cfg.layers as u64, cfg.heads as u64, cfg.kv_heads as u64, cfg.head_dim as u64, cfg.vocab as u64);
    let (taps, block, capacity) = (cfg.taps.len() as u64, cfg.block as u64, capacity as u64);
    if [h,inter,layers,heads,kvh,vocab,block,taps].iter().any(|&n| n > 1_000_000)
        || layers > 1024 || heads > 4096 || kvh > 4096 || taps > 1024 || block > 128
        || !matches!(dim, 64 | 128) || heads % kvh != 0 || taps == 0
        || cfg.rank != 256 || !crate::formats::block_dspark::supports_sequences(sequences) || sequences > slots {
        return Err(GlmDraftStorageError::Unsupported("invalid dSpark geometry or capacity"));
    }
    let q = mul(heads, dim)?;
    let kv = mul(kvh, dim)?;
    let shapes = [(mul(taps,h)?, h), (h, add(q,mul(2,kv)?)?), (q,h), (h,mul(2,inter)?), (inter,h)];
    let values = add(mul(shapes[0].0, shapes[0].1)?, mul(layers,
        sum(shapes[1..].iter().map(|&(k,n)| mul(k,n)).collect::<Result<Vec<_>,_>>()?)?)?)?;
    let fp8 = representation == GlmDraftRepresentation::Fp8Only;
    if fp8 && shapes.iter().any(|&(k,n)| k % 128 != 0 || n % 16 != 0) {
        return Err(GlmDraftStorageError::Unsupported("dSpark FP8 matrix alignment"));
    }
    let weights = sum([mul(values, if fp8 { 1 } else { 2 })?, if fp8 { mul(values / 128,4)? } else { 0 },
        mul(2,add(mul(2,h)?,mul(layers,add(mul(2,h)?,mul(2,dim)?)?)?)?)?,
        mul(4,mul(vocab,256)?)?, mul(vocab,4)?, mul(add(h,256)?,2)?.max(256),256, if dim == 128 { dim * 2 } else { 0 }])?;
    let rows = mul(sequences as u64,block)?;
    let owned = sum([mul(mul(mul(2,layers)?,slots as u64)?,mul(capacity, kv_cache.plane_record_bytes(kvh as usize, dim as usize) as u64)?)?,
        mul(2048,sum([mul(mul(taps,h)?,2)?,mul(h,4)?,mul(kv,4)?,12])?)?])?;
    let terms = [(rows,h,2),(rows,h,2),(rows,add(q,mul(2,kv)?)?,2),(rows,q,2),(rows,kv,2),
        (rows,kv,2),(rows,q,2),(rows,h,2),(rows,mul(2,inter)?,2),(rows,inter,2),(rows,vocab,4),
        (sequences as u64,1,4),(rows,1,4),(rows,1,4),(rows,1,4),(rows,1,8),(sequences as u64,3,4)];
    let mut workspace = 4 << 20;
    for (a,b,c) in terms { workspace = add(workspace,mul(mul(a,b)?,c)?.max(256))?; }
    let attention_scratch = crate::serving_capacity::draft_attention_workspace_bytes(
        sequences as u64, kvh, (heads / kvh * block).div_ceil(32), add(capacity,block)?, dim, sms)
        .map_err(|_| GlmDraftStorageError::Overflow)?.max(256);
    workspace = add(workspace,add(mul(sequences as u64,296*8)?,mul(rows,4)?)?.max(256))?;
    let mut scratch_shapes = shapes.to_vec(); scratch_shapes.push((h,mul(2,kv)?));
    let scratch = if fp8 { scratch_shapes.iter().map(|&(k,n)|
        draft_fp8_scratch_bytes(rows.max(2048),k,n,sms,mode)).max().unwrap_or(256) } else { 0 };
    Ok((sum([weights,owned,workspace])?,add(scratch,attention_scratch)?))
}

/// Shared-pool inventory: replace the legacy slot-local full-context planes
/// with the target-token-sized cache and its immutable page-table allocation.
#[allow(clippy::too_many_arguments)]
pub fn block_dspark_paged_resident_bytes(config: &serde_json::Value, slots: usize, sequences: usize,
    sms: u64, representation: GlmDraftRepresentation, mode: u8, context: usize,
    target_tokens: usize, kv: crate::families::qwen4::Qwen4KvCache)
    -> Result<(u64, u64), GlmDraftStorageError> {
    let cfg = crate::formats::block_dspark::BlockDsparkConfig::from_json(config)
        .map_err(|_| GlmDraftStorageError::Unsupported("invalid dSpark checkpoint"))?;
    let layout = cfg.paged_cache_layout(slots, context, target_tokens, kv)
        .map_err(|_| GlmDraftStorageError::Unsupported("invalid dSpark paged admission"))?;
    let (owned, scratch) = block_dspark_resident_bytes(config, slots, sequences, sms,
        representation, mode, context, kv)?;
    if cfg.window.is_some() { return Ok((owned, scratch)); }
    let old = mul(mul(mul(slots as u64, context as u64)?, cfg.layers as u64)?,
        mul(2, kv.plane_record_bytes(cfg.kv_heads, cfg.head_dim) as u64)?)?;
    let tables = layout.page_table_bytes.max(256);
    Ok((add(owned.checked_sub(old).ok_or(GlmDraftStorageError::Overflow)?,
        add(layout.full_bytes, tables)?)?, scratch))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(flash: bool) -> GlmDraftGeometry {
        GlmDraftGeometry {
            hidden: if flash { 4096 } else { 6144 }, intermediate: 12288,
            layers: if flash { 5 } else { 6 }, heads: if flash { 32 } else { 64 },
            kv_heads: 8, head_dim: 128, taps: if flash { 5 } else { 6 }, vocab: 154880,
            conv_group: 16, selector_rank: 256,
        }
    }

    /// RedHatAI/GLM-5.3-Flash-speculator.dspark-preview, revision 1972f1f0.
    fn flash_dspark() -> serde_json::Value {
        serde_json::json!({"architectures":["DSparkDraftModel"],"speculators_model_type":"dspark",
            "target_hidden_size":null,"aux_hidden_state_layer_ids":[20,28,32,36,40,44],"block_size":8,
            "confidence_head_with_markov":true,"draft_vocab_size":154880,"enable_confidence_head":true,
            "markov_head_type":"vanilla","markov_rank":256,"mask_token_id":154856,"sample_from_anchor":true,
            "sliding_window_non_causal":false,"transformer_layer_config":{"attention_bias":false,"head_dim":64,
                "hidden_act":"silu","hidden_size":4096,"intermediate_size":12288,
                "layer_types":vec!["sliding_attention";5],"max_position_embeddings":1048576,"model_type":"qwen3",
                "num_attention_heads":64,"num_hidden_layers":5,"num_key_value_heads":64,"rms_norm_eps":1e-5,
                "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},"sliding_window":2048,
                "use_sliding_window":true,"vocab_size":154880}})
    }

    #[test]
    fn flash_head_capacity_uses_real_dspark_config_without_dflash_geometry() {
        let config = flash_dspark();
        assert!(config.get("dflash_config").is_none());
        assert!(draft_geometry(&config).is_err());
        for (slots, sequences, expected) in [(None,8,(20,8,64)), (None,16,(20,16,128)),
            (Some(4),16,(4,4,32)), (None,64,(64,32,256))] {
            let cap = flash_draft_capacity(&config,slots,sequences).unwrap();
            assert_eq!((cap.context_slots,cap.max_batch_sequences,cap.block_rows),expected);
            let (owned,scratch) = draft_resident_bytes(&config,cap.context_slots,cap.max_batch_sequences,
                188,crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
            assert!(owned > 0 && scratch > 0);
        }
        let mut specforge = config["transformer_layer_config"].clone();
        for field in ["architectures","block_size","markov_rank","markov_head_type",
            "enable_confidence_head","confidence_head_with_markov"] {
            specforge[field] = config[field].clone();
        }
        specforge["dflash_config"] = serde_json::json!({"projector_type":"dspark",
            "mask_token_id":154856,"target_layer_ids":[19,27,31,35,39,43]});
        assert!(is_dspark(&specforge));
        assert_eq!(flash_draft_capacity(&specforge,None,8).unwrap().block_rows,64);
        assert!(draft_resident_bytes(&specforge,20,8,188,crate::families::qwen4::Qwen4KvCache::Bf16).is_ok());
        assert!(flash_draft_capacity(&config,Some(0),8).is_err());
        assert!(flash_draft_capacity(&config,None,0).is_err());
    }

    #[test]
    fn flash_dflash_head_capacity_is_independent_of_target_concurrency() {
        let config = serde_json::json!({"hidden_size":4096,"intermediate_size":12288,"num_hidden_layers":5,
            "num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"vocab_size":154880,
            "dflash_config":{"target_layer_ids":[0,8,16,24,32],"conv_group_size":16,"selector_rank":256,"block_size":16}});
        assert_eq!(flash_draft_capacity(&config,None,8).unwrap(),GlmDraftCapacity::new(20,8,16).unwrap());
        assert_eq!(flash_draft_capacity(&config,None,16).unwrap(),GlmDraftCapacity::new(20,16,16).unwrap());
        assert_eq!(flash_draft_capacity(&config,Some(4),16).unwrap(),GlmDraftCapacity::new(4,4,16).unwrap());
        for (target,drafts,slots,expected) in [(16,8,None,(16,8,128)),(8,16,None,(8,8,128)),
            (16,8,Some(4),(4,4,64)),(8,16,Some(20),(20,16,256))] {
            let (slots,batch) = flash_serving_draft_limits(target,drafts,slots);
            let cap = flash_draft_capacity(&config,slots,batch).unwrap();
            assert_eq!((cap.context_slots,cap.max_batch_sequences,cap.block_rows),expected);
        }
    }

    #[test]
    fn full_context_dspark_inventory_scales_without_sliding_truncation() {
        let config = serde_json::json!({"architectures":["DSparkDraftModel"],
            "hidden_size":5120,"intermediate_size":10240,"num_hidden_layers":5,
            "num_attention_heads":40,"num_key_value_heads":8,"head_dim":128,
            "rms_norm_eps":1e-6,"rope_parameters":{"rope_type":"yarn","rope_theta":10000000.0,
                "factor":32.0,"original_max_position_embeddings":8192,"beta_fast":32.0,"beta_slow":1.0},
            "vocab_size":248320,"hidden_act":"silu","layer_types":vec!["full_attention";5],
            "sliding_window":null,"block_size":7,"markov_rank":256,"markov_head_type":"vanilla",
            "enable_confidence_head":true,"confidence_head_with_markov":true,"num_target_layers":64,
            "dflash_config":{"projector_type":"dspark","mask_token_id":248077,
                "target_layer_ids":[4,16,28,40,52]}});
        use crate::families::qwen4::Qwen4KvCache;
        for (format, plane) in [(Qwen4KvCache::Bf16, 8 * 128 * 2), (Qwen4KvCache::Fp8, 1056)] {
            let bytes = |capacity| block_dspark_resident_bytes(&config, 1, 1, 188,
                GlmDraftRepresentation::Bf16Only, 0, capacity, format).unwrap();
            let (small, _) = bytes(2048);
            let (large, _) = bytes(8192);
            assert_eq!(large - small, (8192 - 2048) * 5 * 2 * plane);
            assert!(block_dspark_resident_bytes(&config, 1, 1, 188,
                GlmDraftRepresentation::Bf16Only, 0, 0, format).is_err());
            assert!(block_dspark_resident_bytes(&config, usize::MAX, 1, 188,
                GlmDraftRepresentation::Bf16Only, 0, 8192, format).is_err());
        }
    }

    #[test]
    fn flash_draft_ready_inventory_matches_rc3_without_borrowed_head() {
        let g = geometry(true);
        let capacity = GlmDraftCapacity::new(8,8,8).unwrap();
        let layout = GlmDraftRuntimeLayout::new(g,GlmDraftRepresentation::Fp8Only,capacity,2048).unwrap();
        let (arenas,scratch) = draft_workspace_bytes(g,capacity,8,true,188,2, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
        let old_attention = 8 * 8 * 17 * 64 * 130 * 4;
        let new_attention = crate::serving_capacity::draft_attention_workspace_bytes(8, 8, 2,
            2056, 128, 188).unwrap();
        assert_eq!(layout.weights.resident_bytes().unwrap() + arenas,
            1_835_700_096 - old_attention);
        // The shared fc arena adds 128 rows * (20480 / 256) K tiles * 4 B.
        assert_eq!(scratch,30_491_648 + 128 * 80 * 4 + new_attention);
        let mut invalid = g; invalid.conv_group = 0;
        assert!(draft_workspace_bytes(invalid,capacity,8,true,188,2, crate::families::qwen4::Qwen4KvCache::Bf16).is_err());
    }

    #[test]
    fn compact_rings_change_only_owned_bytes_for_both_glm_geometries() {
        use crate::families::qwen4::Qwen4KvCache;
        for flash in [false, true] {
            let g = geometry(flash);
            let capacity = GlmDraftCapacity::new(20, 16, 8).unwrap();
            for weights in [false, true] {
                for sms in [170, 188] {
                    let (bf16, bf16_scratch) = draft_workspace_bytes(g, capacity, 8, weights,
                        sms, 2, Qwen4KvCache::Bf16).unwrap();
                    let (fp8, fp8_scratch) = draft_workspace_bytes(g, capacity, 8, weights,
                        sms, 2, Qwen4KvCache::Fp8).unwrap();
                    assert_eq!(bf16 - fp8, 2 * g.layers * 20 * 2048 * (2048 - 1056));
                    assert_eq!(bf16_scratch, fp8_scratch);
                }
            }
        }
    }

    #[test]
    fn checkpoint_default_preserves_bf16_and_quantization_is_explicit() {
        assert_eq!(GlmDraftRepresentation::from_fp8_option(None), GlmDraftRepresentation::Fp8Only);
        assert_eq!(GlmDraftRepresentation::from_fp8_option(Some(false)), GlmDraftRepresentation::Bf16Only);
        assert_eq!(GlmDraftRepresentation::from_fp8_option(Some(true)), GlmDraftRepresentation::Fp8Only);
    }

    #[test]
    fn actual_glm_and_flash_weights_have_one_representation_and_no_head_copy() {
        let cap = GlmDraftCapacity::new(20, 16, 8).unwrap();
        for (flash, bf16, packed, scale, staging) in [
            (false, 4_759_486_464, 2_379_743_232, 74_366_976, 452_984_832),
            (true, 2_183_135_232, 1_091_567_616, 34_111_488, 201_326_592),
        ] {
            let a = GlmDraftRuntimeLayout::new(geometry(flash), GlmDraftRepresentation::Bf16Only, cap, 2048).unwrap();
            let b = GlmDraftRuntimeLayout::new(geometry(flash), GlmDraftRepresentation::Fp8Only, cap, 2048).unwrap();
            assert_eq!((a.weights.bf16_values, a.weights.fp8_values, a.weights.fp8_scales), (bf16, 0, 0));
            assert_eq!((b.weights.bf16_values, b.weights.fp8_values, b.weights.fp8_scales), (0, packed, scale));
            assert_eq!(a.weights.max_load_staging, 0);
            assert_eq!(b.weights.max_load_staging, staging);
            assert_eq!(a.weights.bf16_codebooks, 158_597_120);
            assert_eq!(a.weights.bf16_codebooks, b.weights.bf16_codebooks);
            assert!(a.fp8_scratch.is_none());
            assert!(b.fp8_scratch.as_ref().unwrap().shapes.iter().all(|s| s.n != 154880));
            assert_eq!(b.weights.loading_peak_bytes().unwrap(), b.weights.resident_bytes().unwrap() + staging);
        }
    }

    #[test]
    fn context_slots_and_draft_batch_are_independent_and_wide_updates_are_covered() {
        let cap = GlmDraftCapacity::new(20, 16, 8).unwrap();
        assert_eq!((cap.context_slots, cap.max_batch_sequences, cap.block_rows), (20, 16, 128));
        let l = GlmDraftRuntimeLayout::new(geometry(true), GlmDraftRepresentation::Fp8Only, cap, 2048).unwrap();
        let scratch = l.fp8_scratch.unwrap();
        assert_eq!(scratch.rows, 2048);
        assert!(scratch.shapes.contains(&GlmDraftLinearShape { k: 4096, n: 2048 }));
        let cap = GlmDraftCapacity::new(300, 300, 8).unwrap();
        let l = GlmDraftRuntimeLayout::new(geometry(true), GlmDraftRepresentation::Fp8Only, cap, 2048).unwrap();
        assert_eq!(l.fp8_scratch.unwrap().rows, 2400);
    }

    #[test]
    fn invalid_capacity_and_overflow_are_named_results() {
        assert!(GlmDraftCapacity::new(0, 16, 8).is_err());
        assert!(GlmDraftCapacity::new(16, 20, 8).is_err());
        assert!(GlmDraftCapacity::new(20, 16, 1).is_err());
        assert!(GlmDraftCapacity::new(1_048_577, 16, 8).is_err());
        let mut g = geometry(false);
        g.layers = u64::MAX;
        assert_eq!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only,
            GlmDraftCapacity::new(20, 16, 8).unwrap(), 2048).unwrap_err(), GlmDraftStorageError::Overflow);
    }

    #[test]
    fn fp8_scratch_admits_split_row_maxima_before_packed_activations() {
        // n=65536 removes projection split-K, isolating the scale/packing arenas.
        let bytes = draft_fp8_scratch_bytes(8, 4096, 65536, 188, 0);
        assert_eq!(bytes, 512 + 16 * 64 * 4 + 4096 * 64 * 2);
        let wide = draft_fp8_scratch_bytes(128, 4096, 65536, 188, 1);
        assert_eq!(wide, 1024 + 16 * 128 * 4 + 4096 * 128 * 2);
        assert_eq!(draft_fp8_scratch_bytes(128, 4096, 65536, 188, 2),
            wide + 32 * 128 * 4);
    }

    #[test]
    fn unsupported_fp8_shapes_do_not_change_bf16_storage() {
        let mut g = geometry(true);
        g.hidden = 4112; // BF16 cuBLAS can read this; FP8 K must align to 128.
        let cap = GlmDraftCapacity::new(20, 16, 8).unwrap();
        assert!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only, cap, 2048).is_ok());
        assert!(matches!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Fp8Only, cap, 2048),
            Err(GlmDraftStorageError::Unsupported(_))));
    }
}
