//! Immutable GLM/GLM Flash DFlash storage and its device-loading contract.
//! The target owns its one vocabulary head (BF16, or GLM 5.3 Flash's FP8-only
//! head with --fp8-head); the drafter borrows it and never copies it.
use thiserror::Error;

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
        if g.head_dim != 128 || g.selector_rank != 256 {
            return Err(GlmDraftStorageError::Unsupported("kernels require head_dim 128 and selector rank 256"));
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
    align(2 * capacity * 4) + align(k * capacity * 2)
        + if mode == 2 { align(k / 128 * capacity * 4) } else { 0 }
        + if splits > 1 { align(splits * rows.min(capacity) * n * 4) } else { 0 }
}

/// Owned draft arenas at readiness. The target head and its mask row are borrowed.
pub fn draft_workspace_bytes(g: GlmDraftGeometry, capacity: GlmDraftCapacity, block: u64,
    fp8: bool, sms: u64, mode: u8) -> Result<(u64, u64), GlmDraftStorageError> {
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
    let owned = add(mul(mul(mul(mul(4,g.layers)?,capacity.context_slots as u64)?,2048)?,kv)?,
        mul(2048,sum([mul(mul(g.taps,g.hidden)?,2)?,mul(g.hidden,4)?,mul(kv,4)?,12])?)?)?;
    let workspace = [4 << 20, rows * g.hidden * 2, rows * g.hidden * 2, rows * g.hidden * 2,
        rows * conv * 2, rows * (attention + 2 * kv) * 2, rows * attention * 2,
        rows * kv * 2, rows * kv * 2, rows * attention * 2, rows * g.hidden * 2,
        rows * 2 * g.intermediate * 2, rows * g.intermediate * 2, rows * g.vocab * 4,
        drafted * 16 * 4, drafted * 16 * 4, rows * g.selector_rank * 2, sequences * 4,
        rows * 4, drafted * 4, drafted * 16, rows * 8, 3 * sequences * 4,
        sequences * g.kv_heads * (2048 + block).div_ceil(128) * 64 * 130 * 4,
        drafted * 64 * 16 * 8].into_iter().map(allocation).sum::<u64>();
    let scratch = if fp8 {
        let layout = GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Fp8Only, capacity, 2048)?;
        layout.fp8_scratch.unwrap().shapes.iter().map(|s|
            draft_fp8_scratch_bytes(2048.max(rows), s.k, s.n, sms, mode)).max().unwrap_or(256)
    } else { 0 };
    // fp8_linear scratch has its own workspace ledger scope, unlike draft activations.
    Ok((add(owned, workspace)?, scratch))
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

pub fn draft_resident_bytes(config: &serde_json::Value, slots: usize, sequences: usize, sms: u64)
    -> Result<(u64, u64), GlmDraftStorageError> {
    draft_resident_bytes_with_mode(config, slots, sequences, sms, GlmDraftRepresentation::Fp8Only, 2)
}

pub fn draft_resident_bytes_with_mode(config: &serde_json::Value, slots: usize, sequences: usize, sms: u64,
    representation: GlmDraftRepresentation, mode: u8) -> Result<(u64, u64), GlmDraftStorageError> {
    if mode > 2 || sms == 0 || sms > 4096 { return Err(GlmDraftStorageError::Unsupported("invalid draft scratch mode or SM count")); }
    if config["speculators_model_type"] == "dspark" {
        return dspark_resident_bytes(config, slots, sequences.min(32), sms, representation, mode);
    }
    let (g, block) = draft_geometry(config)?;
    let capacity = GlmDraftCapacity::new(slots, sequences, usize::try_from(block).map_err(|_| GlmDraftStorageError::Overflow)?)?;
    let layout = GlmDraftRuntimeLayout::new(g, representation, capacity, 2048)?;
    let (workspace, scratch) = draft_workspace_bytes(g, capacity, block,
        representation == GlmDraftRepresentation::Fp8Only, sms, mode)?;
    Ok((add(layout.weights.resident_bytes()?, workspace)?, scratch))
}

fn dspark_resident_bytes(config: &serde_json::Value, slots: usize, sequences: usize, sms: u64,
    representation: GlmDraftRepresentation, mode: u8) -> Result<(u64, u64), GlmDraftStorageError> {
    let t = &config["transformer_layer_config"];
    let field = |name: &str| t[name].as_u64().filter(|&v| v > 0)
        .ok_or(GlmDraftStorageError::Unsupported("missing dSpark geometry"));
    let (h, inter, layers, heads, kvh, dim, vocab) = (field("hidden_size")?, field("intermediate_size")?,
        field("num_hidden_layers")?, field("num_attention_heads")?, field("num_key_value_heads")?,
        field("head_dim")?, field("vocab_size")?);
    let taps = config["aux_hidden_state_layer_ids"].as_array()
        .ok_or(GlmDraftStorageError::Unsupported("missing dSpark taps"))?.len() as u64;
    let block = config["block_size"].as_u64().filter(|&v| v > 0)
        .ok_or(GlmDraftStorageError::Unsupported("missing dSpark block"))?;
    if [h,inter,layers,heads,kvh,vocab,block,taps].iter().any(|&n| n > 1_000_000)
        || layers > 1024 || heads > 4096 || kvh > 4096 || taps > 1024 || block > 128
        || dim != 64 || heads % kvh != 0 || heads / kvh * block > 32 || taps == 0
        || config["markov_rank"] != 256 || sequences == 0 || sequences > slots {
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
        mul(4,mul(vocab,256)?)?, mul(vocab,4)?, mul(add(h,256)?,2)?.max(256),256])?;
    let rows = mul(sequences as u64,block)?;
    let owned = sum([mul(mul(mul(mul(4,layers)?,slots as u64)?,2048)?,kv)?,
        mul(2048,sum([mul(mul(taps,h)?,2)?,mul(h,4)?,mul(kv,4)?,12])?)?])?;
    let terms = [(rows,h,2),(rows,h,2),(rows,add(q,mul(2,kv)?)?,2),(rows,q,2),(rows,kv,2),
        (rows,kv,2),(rows,q,2),(rows,h,2),(rows,mul(2,inter)?,2),(rows,inter,2),(rows,vocab,4),
        (sequences as u64,1,4),(rows,1,4),(rows,1,4),(rows,1,4),(rows,1,8),(sequences as u64,3,4)];
    let mut workspace = 4 << 20;
    for (a,b,c) in terms { workspace = add(workspace,mul(mul(a,b)?,c)?.max(256))?; }
    workspace = add(workspace,mul(mul(mul(sequences as u64,kvh)?,add(2048,block)?.div_ceil(128))?,32*66*4)?.max(256))?;
    workspace = add(workspace,add(mul(sequences as u64,296*8)?,mul(rows,4)?)?.max(256))?;
    let mut scratch_shapes = shapes.to_vec(); scratch_shapes.push((h,mul(2,kv)?));
    let scratch = if fp8 { scratch_shapes.iter().map(|&(k,n)|
        draft_fp8_scratch_bytes(rows.max(2048),k,n,sms,mode)).max().unwrap_or(256) } else { 0 };
    Ok((sum([weights,owned,workspace])?,scratch))
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
    fn flash_draft_ready_inventory_matches_rc3_without_borrowed_head() {
        let g = geometry(true);
        let capacity = GlmDraftCapacity::new(8,8,8).unwrap();
        let layout = GlmDraftRuntimeLayout::new(g,GlmDraftRepresentation::Fp8Only,capacity,2048).unwrap();
        let (arenas,scratch) = draft_workspace_bytes(g,capacity,8,true,188,2).unwrap();
        assert_eq!(layout.weights.resident_bytes().unwrap() + arenas, 1_835_700_096);
        assert_eq!(scratch,30_491_648);
        let mut invalid = g; invalid.conv_group = 0;
        assert!(draft_workspace_bytes(invalid,capacity,8,true,188,2).is_err());
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
    fn unsupported_fp8_shapes_do_not_change_bf16_storage() {
        let mut g = geometry(true);
        g.hidden = 4112; // BF16 cuBLAS can read this; FP8 K must align to 128.
        let cap = GlmDraftCapacity::new(20, 16, 8).unwrap();
        assert!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Bf16Only, cap, 2048).is_ok());
        assert!(matches!(GlmDraftRuntimeLayout::new(g, GlmDraftRepresentation::Fp8Only, cap, 2048),
            Err(GlmDraftStorageError::Unsupported(_))));
    }
}
