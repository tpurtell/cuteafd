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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    geometry: GlmDraftGeometry,
    tap_rows: usize,
    weight_owners: Vec<crate::formats::dflash::MlaDraftOwner>,
    packing_sources: Vec<crate::formats::dflash::MlaDraftOwner>,
    pinned_owners: Vec<crate::formats::dflash::MlaDraftOwner>,
    sync_upload_bytes: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GlmDraftStorageError {
    #[error(transparent)]
    Topk(#[from] crate::formats::dflash::DflashTopkVocabError),
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
        crate::formats::dflash::validate_topk_vocab(g.vocab)?;
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
        // Bound untrusted layer counts before expanding the named inventory.
        mul(g.layers, sum(layer_shapes.iter().map(|s| mul(s.k, s.n))
            .collect::<Result<Vec<_>, _>>()?)?)?;
        if g.layers > 1024 { return Err(GlmDraftStorageError::Unsupported("too many draft layers")); }
        let uses_fp8 = mode == GlmDraftRepresentation::Fp8Only;
        let mut shapes = vec![fc, projection];
        shapes.extend(layer_shapes);
        // Context updates consume only the K|V row slice of packed QKV.
        shapes.push(GlmDraftLinearShape { k: h, n: mul(2, kv)? });
        if uses_fp8 && shapes.iter().any(|s| s.n % 16 != 0 || s.k % 128 != 0) {
            return Err(GlmDraftStorageError::Unsupported("FP8 GEMMs require N multiples of 16 and K multiples of 128"));
        }
        use crate::formats::dflash::{MlaDraftDomain, MlaDraftDtype, MlaDraftLifetime, MlaDraftOwner};
        let owner = |id: String, dtype, dims: Vec<u64>, minimum| {
            let shape = dims.into_iter().map(usize::try_from).collect::<Result<Vec<_>, _>>()
                .map_err(|_| GlmDraftStorageError::Overflow)?;
            MlaDraftOwner::new(id, MlaDraftDomain::Device, dtype, shape, minimum)
                .map_err(|_| GlmDraftStorageError::Overflow)
        };
        let mut weight_owners = Vec::new();
        let mut packing_sources = Vec::new();
        let mut weights = GlmDraftWeightLayout { bf16_values: 0, fp8_values: 0, fp8_scales: 0,
            bf16_auxiliary: 0, bf16_codebooks: 0, max_load_staging: 0 };
        let mut add_weight = |id: String, dims: Vec<u64>, matrix: bool| -> Result<(), GlmDraftStorageError> {
            let source = owner(format!("{id}.source"), MlaDraftDtype::Bf16, dims.clone(), 256)?;
            if matrix && uses_fp8 {
                let values = source.bytes / 2;
                let packed = owner(format!("{id}.values"), MlaDraftDtype::Bytes, vec![values as u64], 0)?;
                let scales = owner(format!("{id}.scales"), MlaDraftDtype::F32, vec![dims[0], dims[1] / 128], 0)?;
                weights.fp8_values = add(weights.fp8_values, packed.bytes as u64)?;
                weights.fp8_scales = add(weights.fp8_scales, scales.bytes as u64)?;
                weights.max_load_staging = weights.max_load_staging.max(source.bytes as u64);
                weight_owners.extend([packed, scales]);
                packing_sources.push(MlaDraftOwner { dtype: MlaDraftDtype::Bytes,
                    shape: vec![source.bytes], ..source });
            } else {
                let total = if matrix { &mut weights.bf16_values }
                    else if id == "weight.predecessor" || id == "weight.successor" { &mut weights.bf16_codebooks }
                    else { &mut weights.bf16_auxiliary };
                *total = add(*total, source.bytes as u64)?;
                weight_owners.push(MlaDraftOwner { id, ..source });
            }
            Ok(())
        };
        // Exact fresh-per-matrix allocation order, including combined QKV and gate/up.
        for layer in 0..g.layers {
            let p = format!("weight.layer.{layer}");
            for leaf in ["input_norm", "post_norm"] {
                add_weight(format!("{p}.{leaf}"), vec![h], false)?;
            }
            for leaf in ["attn", "mlp"] {
                add_weight(format!("{p}.{leaf}_conv"), vec![conv, h], true)?;
                add_weight(format!("{p}.{leaf}_base"), vec![2, 2, h], false)?;
            }
            add_weight(format!("{p}.attention.qkv"), vec![add(attention, mul(2, kv)?)?, h], true)?;
            for leaf in ["q_norm", "k_norm"] {
                add_weight(format!("{p}.attention.{leaf}"), vec![g.head_dim], false)?;
            }
            add_weight(format!("{p}.attention.o"), vec![h, attention], true)?;
            add_weight(format!("{p}.gate_up"), vec![mul(2, g.intermediate)?, h], true)?;
            add_weight(format!("{p}.down"), vec![h, g.intermediate], true)?;
        }
        add_weight("weight.fc".into(), vec![h, mul(g.taps, h)?], true)?;
        for leaf in ["hidden_norm", "norm"] { add_weight(format!("weight.{leaf}"), vec![h], false)?; }
        add_weight("weight.projection".into(), vec![g.selector_rank, h], true)?;
        for leaf in ["predecessor", "successor"] {
            add_weight(format!("weight.{leaf}"), vec![g.vocab, g.selector_rank], false)?;
        }
        let sequences = capacity.max_batch_sequences as u64;
        let rows = capacity.block_rows as u64;
        if sequences == 0 || rows % sequences != 0 || rows / sequences < 2
            || capacity.context_slots < capacity.max_batch_sequences {
            return Err(GlmDraftStorageError::Unsupported("invalid drafter batch capacity"));
        }
        let host = |id: &str, bytes: u64| -> Result<MlaDraftOwner, GlmDraftStorageError> {
            MlaDraftOwner::new(id, MlaDraftDomain::HostPinned, MlaDraftDtype::Bytes,
                vec![usize::try_from(bytes).map_err(|_| GlmDraftStorageError::Overflow)?], 0)
                .map_err(|_| GlmDraftStorageError::Overflow)
        };
        let mut pinned_owners = vec![host("landing", mul(rows - sequences, 20)?)?];
        for (id, bytes, diagnostic) in [
            ("context_positions", mul(tap_rows as u64, 8)?, false),
            ("context_slots", mul(tap_rows as u64, 4)?, false),
            ("positions", mul(rows, 8)?, false), ("anchors", mul(sequences, 4)?, false),
            ("ids", mul(rows, 4)?, false), ("tables", mul(sequences, 12)?, false),
            ("taps", mul(mul(tap_rows as u64, g.taps)?, mul(h, 2)?)?, true),
            ("h", mul(rows, mul(h, 2)?)?, true),
        ] {
            let mut owner = host(&format!("{id}.upload"), bytes.max(256))?;
            owner.lifetime = MlaDraftLifetime::Library;
            owner.staging_for = Some(id.into());
            owner.on_use = diagnostic;
            pinned_owners.push(owner);
        }
        // Concatenated QKV and gate/up sources upload one checkpoint tensor at a time.
        let sync_values = [mul(fc.k, fc.n)?, mul(projection.k, projection.n)?,
            mul(conv, h)?, mul(attention, h)?, mul(kv, h)?, mul(g.intermediate, h)?,
            mul(g.vocab, g.selector_rank)?, mul(4, h)?, h, g.head_dim]
            .into_iter().max().unwrap_or(0);
        let sync_upload_bytes = usize::try_from(mul(sync_values, 2)?)
            .map_err(|_| GlmDraftStorageError::Overflow)?;
        Ok(Self { capacity, weights, fp8_scratch: uses_fp8.then_some(GlmDraftFp8ScratchLayout {
            rows: tap_rows.max(capacity.block_rows), shapes,
        }), geometry: g, tap_rows, weight_owners, packing_sources, pinned_owners, sync_upload_bytes })
    }

    pub fn weight_owners(&self) -> &[crate::formats::dflash::MlaDraftOwner] { &self.weight_owners }
    pub fn packing_sources(&self) -> &[crate::formats::dflash::MlaDraftOwner] { &self.packing_sources }
    pub fn pinned_owners(&self) -> &[crate::formats::dflash::MlaDraftOwner] { &self.pinned_owners }
    pub fn sync_upload_bytes(&self) -> usize { self.sync_upload_bytes }

    /// Slot-local rings and execution arenas. Native size queries are checked by
    /// the executor against these descriptors before an allocation is published.
    pub fn device_owners(&self, sms: u64, mode: u8, kv_cache: crate::families::qwen4::Qwen4KvCache)
        -> Result<Vec<crate::formats::dflash::MlaDraftOwner>, GlmDraftStorageError> {
        use crate::formats::dflash::{MlaDraftDomain, MlaDraftDtype::{Bf16, Bytes, F32, I32, I64, U32}, MlaDraftOwner};
        let g = self.geometry;
        let capacity = self.capacity;
        let block = capacity.block_rows / capacity.max_batch_sequences;
        if [g.hidden, g.intermediate, g.layers, g.heads, g.kv_heads, g.head_dim, g.taps, g.vocab,
            g.conv_group, g.selector_rank, block as u64].iter().any(|&n| n == 0 || n > 1_000_000)
            || g.layers > 1024 || g.heads > 4096 || g.kv_heads > 4096 || g.taps > 1024
            || capacity.max_batch_sequences > 32 || block > 128 || mode > 2 || sms == 0 || sms > 4096 {
            return Err(GlmDraftStorageError::Unsupported("draft geometry exceeds native arena limits"));
        }
        let (seq, rows, tap_rows) = (capacity.max_batch_sequences as u64, capacity.block_rows as u64, self.tap_rows as u64);
        let drafted = rows - seq;
        let kv = mul(g.kv_heads, g.head_dim)?;
        let attention = mul(g.heads, g.head_dim)?;
        let conv = mul(4, g.hidden)? / g.conv_group;
        let mut owners = self.weight_owners.clone();
        let mut add_owner = |id: String, dtype, dims: Vec<u64>| {
            let shape = dims.into_iter().map(usize::try_from).collect::<Result<Vec<_>, _>>()
                .map_err(|_| GlmDraftStorageError::Overflow)?;
            owners.push(MlaDraftOwner::new(id, MlaDraftDomain::Device, dtype, shape, 256)
                .map_err(|_| GlmDraftStorageError::Overflow)?);
            Ok::<_, GlmDraftStorageError>(())
        };
        let ring = mul(mul(capacity.context_slots as u64, 2048)?,
            kv_cache.plane_record_bytes(g.kv_heads as usize, g.head_dim as usize) as u64)?;
        for layer in 0..g.layers {
            for leaf in ["k_ring", "v_ring"] { add_owner(format!("layer.{layer}.{leaf}"), Bytes, vec![ring])?; }
        }
        for (id, dtype, dims) in [
            ("taps", Bf16, vec![tap_rows, g.taps, g.hidden]),
            ("fused", Bf16, vec![tap_rows, g.hidden]), ("fused_norm", Bf16, vec![tap_rows, g.hidden]),
            ("context_positions", I64, vec![tap_rows]), ("context_slots", I32, vec![tap_rows]),
            ("context_kv", Bf16, vec![tap_rows, 2 * kv]), ("head_workspace", Bytes, vec![4 << 20]),
            ("h", Bf16, vec![rows, g.hidden]), ("n", Bf16, vec![rows, g.hidden]),
            ("conv", Bf16, vec![rows, g.hidden]), ("dynamic", Bf16, vec![rows, conv]),
            ("qkv", Bf16, vec![rows, attention + 2 * kv]), ("q", Bf16, vec![rows, attention]),
            ("k", Bf16, vec![rows, kv]), ("v", Bf16, vec![rows, kv]), ("attn", Bf16, vec![rows, attention]),
            ("delta", Bf16, vec![rows, g.hidden]), ("gate_up", Bf16, vec![rows, 2 * g.intermediate]),
            ("act", Bf16, vec![rows, g.intermediate]), ("logits", F32, vec![rows, g.vocab]),
            ("unary", F32, vec![drafted, 16]), ("candidates", U32, vec![drafted, 16]),
            ("projected", Bf16, vec![rows, g.selector_rank]), ("anchors", U32, vec![seq]),
            ("ids", U32, vec![rows]), ("tokens", U32, vec![drafted]),
            ("features", Bytes, vec![drafted, 16]), ("positions", I64, vec![rows]),
            ("tables", I32, vec![3, seq]), ("topk_workspace", Bytes, vec![mul(drafted, 64 * 16 * 8)?]),
        ] { add_owner(id.into(), dtype, dims)?; }
        let attention_scratch = crate::serving_capacity::draft_attention_workspace_bytes(seq,
            g.kv_heads, 2, 2048 + block as u64, g.head_dim, sms).map_err(|_| GlmDraftStorageError::Overflow)?;
        add_owner("attention_scratch".into(), Bytes, vec![attention_scratch])?;
        if let Some(scratch) = &self.fp8_scratch {
            let bytes = scratch.shapes.iter().map(|s|
                draft_fp8_scratch_bytes(scratch.rows as u64, s.k, s.n, sms, mode)).max().unwrap_or(0).max(256);
            add_owner("fp8_scratch".into(), Bytes, vec![bytes])?;
        }
        owners.iter().try_fold(0u64, |total, owner| add(total, owner.bytes as u64))?;
        Ok(owners)
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
    let layout = GlmDraftRuntimeLayout::new(g,
        if fp8 { GlmDraftRepresentation::Fp8Only } else { GlmDraftRepresentation::Bf16Only }, capacity, 2048)?;
    if block != (capacity.block_rows / capacity.max_batch_sequences) as u64 {
        return Err(GlmDraftStorageError::Unsupported("block disagrees with batch capacity"));
    }
    let (mut owned, mut scratch) = (0, 0);
    for owner in layout.device_owners(sms, mode, kv_cache)? {
        if owner.id.starts_with("weight.") { continue; }
        let total = if matches!(owner.id.as_str(), "fp8_scratch" | "attention_scratch") {
            &mut scratch
        } else { &mut owned };
        *total = add(*total, owner.bytes as u64)?;
    }
    Ok((owned, scratch))
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
    crate::formats::dflash::validate_topk_vocab(int(config, "vocab_size")?)?;
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
    let tokens = slots.checked_mul(2048).ok_or(GlmDraftStorageError::Overflow)?;
    block_dspark_paged_resident_bytes(config, slots, sequences, sms, representation, mode, 2048, tokens, kv_cache)
}

/// Compatibility-free projection of the shared named recipe into the aggregate
/// consumed by unported family admission. Borrowed target tables are excluded.
#[allow(clippy::too_many_arguments)]
pub fn block_dspark_paged_resident_bytes(config: &serde_json::Value, slots: usize, sequences: usize,
    sms: u64, representation: GlmDraftRepresentation, mode: u8, context: usize,
    target_tokens: usize, kv: crate::families::qwen4::Qwen4KvCache)
    -> Result<(u64, u64), GlmDraftStorageError> {
    let cfg = crate::formats::block_dspark::BlockDsparkConfig::from_json(config)
        .map_err(|_| GlmDraftStorageError::Unsupported("invalid dSpark checkpoint"))?;
    let layout = cfg.runtime_layout(slots, sequences, context, target_tokens,
        usize::try_from(sms).map_err(|_| GlmDraftStorageError::Overflow)?, representation, mode, kv)
        .map_err(|_| GlmDraftStorageError::Unsupported("invalid dSpark runtime admission"))?;
    let scratch = layout.scratch_bytes().map_err(|_| GlmDraftStorageError::Overflow)?;
    let total = layout.device_bytes().map_err(|_| GlmDraftStorageError::Overflow)?;
    Ok((total.checked_sub(scratch).ok_or(GlmDraftStorageError::Overflow)?, scratch))
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

    #[test]
    fn topk_vocab_refuses_native_abi_before_weight_inventory() {
        let capacity = GlmDraftCapacity::new(20, 16, 8).unwrap();
        for vocab in [15, i32::MAX as u64 + 1] {
            let mut g = geometry(false);
            g.vocab = vocab;
            assert!(matches!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only,
                capacity, 2048), Err(GlmDraftStorageError::Topk(_))));
        }
        let mut g = geometry(false);
        g.vocab = 202048;
        assert!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only, capacity, 2048).is_ok());
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
            let bytes = |capacity| block_dspark_paged_resident_bytes(&config, 1, 1, 188,
                GlmDraftRepresentation::Bf16Only, 0, capacity, capacity, format).unwrap();
            let (small, _) = bytes(2048);
            let (large, _) = bytes(8192);
            let page_tables = |capacity: u64| (capacity.div_ceil(64) * 4).max(256);
            assert_eq!(large - small, (8192 - 2048) * 5 * 2 * plane
                + page_tables(8192) - page_tables(2048));
            assert!(block_dspark_paged_resident_bytes(&config, 1, 1, 188,
                GlmDraftRepresentation::Bf16Only, 0, 0, 64, format).is_err());
            assert!(block_dspark_paged_resident_bytes(&config, usize::MAX, 1, 188,
                GlmDraftRepresentation::Bf16Only, 0, 8192, 8192, format).is_err());
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
    fn gqa_named_attention_scratch_tracks_supported_head_dimensions() {
        use crate::families::qwen4::Qwen4KvCache;
        let capacity = GlmDraftCapacity::new(20, 16, 8).unwrap();
        for head_dim in [64, 128] {
            let mut g = geometry(true);
            g.head_dim = head_dim;
            for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                let layout = GlmDraftRuntimeLayout::new(g, representation, capacity, 2048).unwrap();
                for sms in [170, 188] {
                    let owners = layout.device_owners(sms, 2, Qwen4KvCache::Bf16).unwrap();
                    let attention = owners.iter().find(|owner| owner.id == "attention_scratch").unwrap();
                    let expected = crate::serving_capacity::draft_attention_workspace_bytes(16,
                        g.kv_heads, 2, 2056, head_dim, sms).unwrap().max(256);
                    assert_eq!(attention.bytes as u64, expected);
                    let (_, scratch) = draft_workspace_bytes(g, capacity, 8,
                        representation == GlmDraftRepresentation::Fp8Only, sms, 2, Qwen4KvCache::Bf16).unwrap();
                    assert_eq!(scratch, owners.iter().filter(|owner|
                        matches!(owner.id.as_str(), "attention_scratch" | "fp8_scratch"))
                        .map(|owner| owner.bytes as u64).sum::<u64>());
                }
            }
        }
        let mut g = geometry(true);
        g.head_dim = 96;
        assert!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only, capacity, 2048).is_err());
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
    fn gqa_pinned_owners_are_host_only_and_diagnostics_are_not_serving_reservations() {
        use crate::formats::dflash::{MlaDraftDomain, MlaDraftLifetime};
        for flash in [false, true] {
            for mode in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                let g = geometry(flash);
                let capacity = GlmDraftCapacity::new(20, 16, 8).unwrap();
                let layout = GlmDraftRuntimeLayout::new(g, mode, capacity, 2048).unwrap();
                assert!(layout.pinned_owners().iter().all(|o| o.domain == MlaDraftDomain::HostPinned));
                let serving: Vec<_> = layout.pinned_owners().iter().filter(|o| !o.on_use).collect();
                assert_eq!(serving.iter().map(|o| o.id.as_str()).collect::<Vec<_>>(), ["landing",
                    "context_positions.upload", "context_slots.upload", "positions.upload",
                    "anchors.upload", "ids.upload", "tables.upload"]);
                assert_eq!(serving.iter().map(|o| o.bytes).sum::<usize>(),
                    16 * 7 * 20 + 2048 * 8 + 2048 * 4 + 128 * 8 + 256 + 128 * 4 + 256);
                assert_eq!(serving[0].lifetime, MlaDraftLifetime::Runtime);
                assert!(serving[1..].iter().all(|o| o.lifetime == MlaDraftLifetime::Library && o.staging_for.is_some()));
                assert_eq!(layout.pinned_owners().iter().filter(|o| o.on_use).map(|o| o.id.as_str())
                    .collect::<Vec<_>>(), ["taps.upload", "h.upload"]);
                // Gate/up uploads its halves, not its combined source extent.
                assert_eq!(layout.sync_upload_bytes() as u64, (g.taps * g.hidden * g.hidden)
                    .max(g.vocab * g.selector_rank).max(g.heads * g.head_dim * g.hidden)
                    .max(g.hidden * g.intermediate) * 2);
            }
        }
    }

    #[test]
    fn gqa_named_device_owners_project_exact_aggregates_and_fresh_packing_order() {
        use crate::formats::dflash::MlaDraftDomain;
        use crate::families::qwen4::Qwen4KvCache;
        for flash in [false, true] {
            for mode in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                let g = geometry(flash);
                let capacity = GlmDraftCapacity::new(20, 16, 8).unwrap();
                let layout = GlmDraftRuntimeLayout::new(g, mode, capacity, 2048).unwrap();
                let sources = layout.packing_sources();
                if mode == GlmDraftRepresentation::Fp8Only {
                    assert_eq!(sources.len() as u64, g.layers * 6 + 2);
                    assert_eq!(sources.iter().take(6).map(|s| s.id.as_str()).collect::<Vec<_>>(), [
                        "weight.layer.0.attn_conv.source", "weight.layer.0.mlp_conv.source",
                        "weight.layer.0.attention.qkv.source", "weight.layer.0.attention.o.source",
                        "weight.layer.0.gate_up.source", "weight.layer.0.down.source"]);
                    assert_eq!(sources[sources.len() - 2].id, "weight.fc.source");
                    assert_eq!(sources.last().unwrap().id, "weight.projection.source");
                    assert_eq!(layout.weights.max_load_staging, sources.iter().map(|s| s.bytes as u64).max().unwrap());
                    assert!(sources.iter().all(|s| s.dtype == crate::formats::dflash::MlaDraftDtype::Bytes));
                } else { assert!(sources.is_empty()); }
                for sms in [170, 188] {
                    for arithmetic in 0..=2 {
                        for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                            let owners = layout.device_owners(sms, arithmetic, kv).unwrap();
                            assert!(owners.iter().all(|o| o.domain == MlaDraftDomain::Device && o.bytes >= 256));
                            assert_eq!(owners.iter().map(|o| &o.id).collect::<std::collections::BTreeSet<_>>().len(), owners.len());
                            assert!(owners.iter().all(|o| !o.id.contains("embedding") && !o.id.contains("vocabulary")));
                            let (workspace, scratch) = draft_workspace_bytes(g, capacity, 8,
                                mode == GlmDraftRepresentation::Fp8Only, sms, arithmetic, kv).unwrap();
                            assert_eq!(owners.iter().map(|o| o.bytes as u64).sum::<u64>(),
                                layout.weights.resident_bytes().unwrap() + workspace + scratch);
                            assert_eq!(owners.iter().find(|o| o.id == "context_slots").unwrap().shape, [2048]);
                            assert_eq!(owners.iter().find(|o| o.id == "context_slots").unwrap().bytes, 2048 * 4);
                        }
                    }
                }
            }
        }
        let g = geometry(true);
        let capacity = GlmDraftCapacity::new(20, 16, 8).unwrap();
        assert!(draft_workspace_bytes(g, capacity, 7, false, 188, 0, Qwen4KvCache::Bf16).is_err());
        let mut huge = g; huge.layers = 1025;
        assert!(GlmDraftRuntimeLayout::new(huge, GlmDraftRepresentation::Bf16Only, capacity, 2048).is_err());
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
