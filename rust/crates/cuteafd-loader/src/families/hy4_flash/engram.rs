//! Hy4-flash Engram n-gram hashing, FP8-only mapped-table reader
//! (`engram-fp8-layer-*-part-*.safetensors`, `fp8_e4m3_f32scale_row_interleaved_v1`,
//! 100-byte rows), shared `TokenTable` row cache, and multi-layer look-ahead over
//! `formats::mapped_table::lookahead`.
//!
//! Matches `python/reference/families/hy4_flash/model.py` (`engram_rows`) and
//! `quantization/hy4_flash/engram.py` (`decode_rows` / `_codes_and_scale`).
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Component, Path};
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;

use crate::formats::mapped_table::lookahead::{Lookahead, LookaheadHandle, LookaheadTable, RowSource, UnitPlan};
use crate::formats::mapped_table::row_cache::RowCache;
use crate::formats::mapped_table::token_table::TokenTable;
use crate::formats::mapped_table::ReadPriority;
use crate::plan::checkpoint::mc::{EngramFp8Layer, ENGRAM_FP8_LAYOUT};
use crate::plan::checkpoint::Checkpoint;
use crate::{MappedTable, RowFormat, TablePart};

use super::Hy4FlashConfig;

/// Elements per Engram embedding row (`engram_n_embed_per_ngram / engram_n_head_per_ngram`).
pub const ENGRAM_EMBED_WIDTH: usize = 96;
/// Engram heads per token (`(engram_max_ngram_size - 1) * engram_n_head_per_ngram`).
pub const ENGRAM_HEADS: usize = 16;
/// Bytes per row in the FP8 shards (`96` E4M3 codes + little-endian `F32` scale).
pub const ENGRAM_FP8_ROW_BYTES: usize = 100;
/// Decoded BF16 payload bytes per token across all 16 heads (`16 * 96 * 2`).
pub const ENGRAM_DECODED_TOKEN_BYTES: usize = ENGRAM_HEADS * ENGRAM_EMBED_WIDTH * 2;

/// Host-memory sizing for Hy4-flash's 4 FP8 Engram tables, bounded row caches,
/// and pinned prefill/decode staging pools. Shared by planner admission and runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hy4EngramShape {
    pub unit_rows: usize,
    pub waves: usize,
    pub decode_rows: usize,
    pub layers: usize,
}

impl Hy4EngramShape {
    pub fn new(unit_rows: usize, waves: usize, decode_rows: usize, layers: usize) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&unit_rows)
                && (1..=4).contains(&waves)
                && (1..=256).contains(&decode_rows)
                && (1..=8).contains(&layers),
            "invalid Hy4 Engram shape: unit_rows={unit_rows} waves={waves} decode_rows={decode_rows} layers={layers}"
        );
        Ok(Self { unit_rows, waves, decode_rows, layers })
    }

    /// Per-layer `(prefill, decode)` `RowCache` slot counts matching the shared
    /// `lookahead::window_units(waves) + 1` prefill window and two decode rounds.
    pub fn cache_shares(self) -> [usize; 2] {
        TokenTable::shares(self.unit_rows, self.waves, self.decode_rows, ENGRAM_HEADS)
    }

    /// Total host bytes of all per-layer `RowCache`s (100-byte FP8 rows + index).
    pub fn cache_bytes(self) -> usize {
        self.layers * TokenTable::cache_bytes(ENGRAM_FP8_ROW_BYTES, self.cache_shares())
    }

    /// Pinned host staging slots for prefill gathers (one per Engram layer per wave + 1 in-flight copy).
    pub fn prefill_slots(self) -> usize {
        self.layers * (self.waves + 1)
    }

    /// Pinned host staging slots for decode/verify gathers (two rounds × all Engram layers).
    pub fn decode_slots(self) -> usize {
        self.layers * 2
    }

    /// Bytes of one prefill staging slot (decoded BF16 rows + per-row mask bytes).
    pub fn prefill_slot_bytes(self) -> usize {
        self.unit_rows * (ENGRAM_DECODED_TOKEN_BYTES + 1)
    }

    /// Bytes of one decode staging slot (decoded BF16 rows + per-row mask bytes).
    pub fn decode_slot_bytes(self) -> usize {
        self.decode_rows * (ENGRAM_DECODED_TOKEN_BYTES + 1)
    }

    /// Total pinned host bytes for prefill and decode Engram staging pools.
    pub fn staging_bytes(self) -> usize {
        self.prefill_slots() * self.prefill_slot_bytes() + self.decode_slots() * self.decode_slot_bytes()
    }
}

/// Advance a rolling 2-token Engram tail by `accepted` tokens (clamped to `tokens.len()`).
/// Used for accepted-prefix history transactions so rejected or cancelled rows never commit.
#[inline]
pub fn advance_tail(tail: [u32; 2], tokens: &[u32], accepted: usize) -> [u32; 2] {
    let take = accepted.min(tokens.len());
    match take {
        0 => tail,
        1 => [tail[1], tokens[0]],
        n => [tokens[n - 2], tokens[n - 1]],
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust NumPy SeedSequence + PCG64 + Lemire bounded uint64
// ---------------------------------------------------------------------------

const SEED_SEQ_INIT_A: u32 = 0x43b0_d7e5;
const SEED_SEQ_MULT_A: u32 = 0x931e_8875;
const SEED_SEQ_INIT_B: u32 = 0x8b51_f9dd;
const SEED_SEQ_MULT_B: u32 = 0x58f3_8ded;
const SEED_SEQ_MIX_L: u32 = 0xca01_f9dd;
const SEED_SEQ_MIX_R: u32 = 0x4973_f715;
const PCG_MULT_128: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

fn seed_sequence_u64_words(seed: u64) -> [u64; 4] {
    let mut entropy = Vec::with_capacity(2);
    if seed == 0 {
        entropy.push(0u32);
    } else {
        let mut rem = seed;
        while rem > 0 {
            entropy.push(rem as u32);
            rem >>= 32;
        }
    }
    let mut hash_const = SEED_SEQ_INIT_A;
    let mut hashmix = |mut v: u32| -> u32 {
        v ^= hash_const;
        hash_const = hash_const.wrapping_mul(SEED_SEQ_MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let r = SEED_SEQ_MIX_L.wrapping_mul(x).wrapping_sub(SEED_SEQ_MIX_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; 4];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for i_src in 0..4 {
        for i_dst in 0..4 {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src]));
            }
        }
    }
    let mut out_const = SEED_SEQ_INIT_B;
    let mut words32 = [0u32; 8];
    for (i, slot) in words32.iter_mut().enumerate() {
        let mut d = pool[i % 4] ^ out_const;
        out_const = out_const.wrapping_mul(SEED_SEQ_MULT_B);
        d = d.wrapping_mul(out_const);
        *slot = d ^ (d >> 16);
    }
    [
        u64::from(words32[0]) | (u64::from(words32[1]) << 32),
        u64::from(words32[2]) | (u64::from(words32[3]) << 32),
        u64::from(words32[4]) | (u64::from(words32[5]) << 32),
        u64::from(words32[6]) | (u64::from(words32[7]) << 32),
    ]
}

struct NumpyPcg64 {
    state: u128,
    inc: u128,
}

impl NumpyPcg64 {
    fn from_seed(seed: u64) -> Self {
        let words = seed_sequence_u64_words(seed);
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let inc = (initseq << 1) | 1;
        let mut rng = Self { state: 0, inc };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT_128).wrapping_add(self.inc);
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.step();
        let x = ((self.state >> 64) ^ self.state) as u64;
        let rot = (self.state >> 122) as u32;
        x.rotate_right(rot)
    }

    /// Exact match to NumPy's `bounded_lemire_uint64` (`rng.integers(0, bound, dtype=np.int64)`).
    fn next_bounded_u64(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let rng = bound - 1;
        let range = u128::from(rng) + 1;
        let mut m = u128::from(self.next_u64()) * range;
        let mut left = m as u64;
        if left < bound {
            let threshold = bound.wrapping_neg() % bound;
            while left < threshold {
                m = u128::from(self.next_u64()) * range;
                left = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

/// Deterministic Hy4-flash per-layer n-gram hasher matching `model.py:engram_rows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hy4EngramHasher {
    pub layer: usize,
    pub lookup: Vec<i32>,
    pub factors: Vec<i64>,
    pub sizes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub heads_per_ngram: usize,
    pub pad_id: u32,
    pub masked_tokens: [u32; 3],
}

/// Streaming n-gram token history (`max_ngram_size - 1` previous tokens, padded with `engram_pad_id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hy4EngramHistory(pub Vec<u32>);

impl Hy4EngramHasher {
    pub fn new(cfg: &Hy4FlashConfig, layer: usize, lookup: Vec<i32>) -> Result<Self> {
        ensure!(lookup.len() == cfg.engram_lookup_table_size,
            "Hy4 Engram layer {layer} lookup_table length {} != {}", lookup.len(), cfg.engram_lookup_table_size);
        let max_val = lookup.iter().copied().max().context("empty Hy4 Engram lookup_table")?;
        ensure!(max_val >= 0 && lookup.iter().all(|&v| v >= 0),
            "Hy4 Engram layer {layer} lookup_table contains negative compressed ids");
        ensure!((cfg.engram_pad_id as usize) < lookup.len(),
            "Hy4 Engram pad_id {} outside lookup_table {}", cfg.engram_pad_id, lookup.len());
        let sizes = cfg.engram_vocab_sizes.get(&layer)
            .with_context(|| format!("Hy4-flash layer {layer} has no Engram config"))?.clone();
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut total = 0u64;
        for &size in &sizes {
            offsets.push(total);
            total = total.checked_add(size).context("Hy4 Engram vocab size sum overflow")?;
        }
        let bound = (i64::MAX / (i64::from(max_val) + 1) / 2).max(1) as u64;
        let seed = cfg.engram_seed.checked_add(10_007u64 * (layer as u64 + 1))
            .context("Hy4 Engram seed overflow")?;
        let mut rng = NumpyPcg64::from_seed(seed);
        let factors = (0..cfg.engram_max_ngram_size)
            .map(|_| (rng.next_bounded_u64(bound) as i64) * 2 + 1)
            .collect();
        Ok(Self {
            layer,
            lookup,
            factors,
            sizes,
            offsets,
            heads_per_ngram: cfg.engram_n_head_per_ngram,
            pad_id: cfg.engram_pad_id,
            masked_tokens: cfg.masked_engram_tokens(),
        })
    }

    pub fn ngram_size(&self) -> usize { self.factors.len() }
    pub fn heads(&self) -> usize { self.sizes.len() }
    pub fn total_rows(&self) -> u64 { self.sizes.iter().sum() }

    pub fn start(&self) -> Hy4EngramHistory {
        Hy4EngramHistory(vec![self.pad_id; self.ngram_size() - 1])
    }

    #[inline]
    pub fn is_masked(&self, token: u32) -> bool {
        self.masked_tokens.contains(&token)
    }

    #[inline]
    fn compress(&self, token: u32) -> i64 {
        let idx = if (token as usize) < self.lookup.len() { token as usize } else { self.pad_id as usize };
        i64::from(self.lookup[idx])
    }

    /// Hash `tokens` starting from `history`, appending `tokens.len() * self.heads()`
    /// global table row IDs to `out` and updating `history` for streaming continuity.
    pub fn hash(&self, history: &mut Hy4EngramHistory, tokens: &[u32], out: &mut Vec<u64>) -> Result<()> {
        let ctx = self.ngram_size() - 1;
        ensure!(history.0.len() == ctx, "Hy4 Engram history must hold {ctx} tokens");
        if tokens.is_empty() { return Ok(()); }
        let mut window = Vec::with_capacity(ctx + tokens.len());
        window.extend_from_slice(&history.0);
        window.extend_from_slice(tokens);
        out.reserve(tokens.len() * self.heads());
        for pos in ctx..window.len() {
            let mut acc = self.compress(window[pos]) * self.factors[0];
            for order in 1..self.ngram_size() {
                acc ^= self.compress(window[pos - order]) * self.factors[order];
                let base = (order - 1) * self.heads_per_ngram;
                for h in base..base + self.heads_per_ngram {
                    let row = acc.rem_euclid(self.sizes[h] as i64) as u64 + self.offsets[h];
                    out.push(row);
                }
            }
        }
        history.0.copy_from_slice(&window[window.len() - ctx..]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bitwise-exact OCP E4M3FN -> F32 / BF16 decoder matching quantization/hy4_flash/engram.py
// ---------------------------------------------------------------------------

#[inline]
pub fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

/// Exact OCP FP8 E4M3FN (`float8_e4m3fn`) byte to F32.
#[inline]
pub fn e4m3fn_to_f32(code: u8) -> f32 {
    let sign = u32::from(code & 0x80) << 24;
    let exp = (code >> 3) & 0x0f;
    let mant = code & 0x07;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let val = (f32::from(mant)) * f32::from_bits((127 - 9) << 23); // mant * 2^-9
        return f32::from_bits(val.to_bits() | sign);
    }
    if exp == 15 && mant == 7 {
        return f32::from_bits(sign | 0x7fc0_0000);
    }
    let f_exp = u32::from(exp) + (127 - 7);
    let f_mant = u32::from(mant) << 20;
    f32::from_bits(sign | (f_exp << 23) | f_mant)
}

/// Decode one 100-byte `fp8_e4m3_f32scale_row_interleaved_v1` row into 96 BF16 elements:
/// `BF16(F32(code_e4m3) * scale_f32)`. Bitwise identical to `quantization/hy4_flash/engram.py:decode_rows`.
#[inline]
pub fn decode_fp8_row(packed: &[u8; ENGRAM_FP8_ROW_BYTES], out_bf16: &mut [u16; ENGRAM_EMBED_WIDTH]) {
    let scale = f32::from_le_bytes([packed[96], packed[97], packed[98], packed[99]]);
    for j in 0..ENGRAM_EMBED_WIDTH {
        out_bf16[j] = f32_to_bf16_rne(e4m3fn_to_f32(packed[j]) * scale);
    }
}

/// Decode a contiguous slice of 100-byte FP8 Engram rows into little-endian BF16 bytes (`rows * 192`).
pub fn decode_fp8_rows_to_bf16_bytes(raw_fp8: &[u8], out_bf16_bytes: &mut [u8]) -> Result<()> {
    ensure!(
        raw_fp8.len() % ENGRAM_FP8_ROW_BYTES == 0,
        "FP8 Engram payload {} not a multiple of {ENGRAM_FP8_ROW_BYTES}",
        raw_fp8.len()
    );
    let count = raw_fp8.len() / ENGRAM_FP8_ROW_BYTES;
    let expected_out = count * ENGRAM_EMBED_WIDTH * 2;
    ensure!(
        out_bf16_bytes.len() == expected_out,
        "FP8 Engram BF16 destination {} != expected {expected_out}",
        out_bf16_bytes.len()
    );
    let mut row_bf16 = [0u16; ENGRAM_EMBED_WIDTH];
    for (i, chunk) in raw_fp8.chunks_exact(ENGRAM_FP8_ROW_BYTES).enumerate() {
        let src: &[u8; ENGRAM_FP8_ROW_BYTES] = chunk.try_into().unwrap();
        decode_fp8_row(src, &mut row_bf16);
        let dst = &mut out_bf16_bytes[i * ENGRAM_EMBED_WIDTH * 2..(i + 1) * ENGRAM_EMBED_WIDTH * 2];
        for (j, &val) in row_bf16.iter().enumerate() {
            let le = val.to_le_bytes();
            dst[j * 2] = le[0];
            dst[j * 2 + 1] = le[1];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FP8-only mapped-table reader & multi-layer look-ahead over shared TokenTable
// ---------------------------------------------------------------------------

/// One Hy4-flash FP8 Engram embedding table backed by the shared [`TokenTable`]
/// (`MappedTable` + bounded [`RowCache`] + shared read reactor).
pub struct Hy4EngramTable {
    layer: usize,
    table: Arc<TokenTable>,
}

fn read_fp8_part_header(path: &Path, expected_tensor: &str, expected_row_start: u64, expected_rows: u64) -> Result<u64> {
    let mut file = File::open(path).with_context(|| format!("opening FP8 Engram part {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    ensure!((2..=64 * 1024 * 1024).contains(&header_len),
        "invalid safetensors header length {header_len} in {}", path.display());
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .with_context(|| format!("decoding safetensors header in {}", path.display()))?;
    let meta = header.get("__metadata__").and_then(serde_json::Value::as_object)
        .with_context(|| format!("FP8 Engram part {} lacks __metadata__", path.display()))?;
    ensure!(meta.get("engram_layout").and_then(serde_json::Value::as_str) == Some(ENGRAM_FP8_LAYOUT),
        "FP8 Engram part {} has wrong engram_layout", path.display());
    ensure!(meta.get("row_start").and_then(serde_json::Value::as_str) == Some(expected_row_start.to_string().as_str()),
        "FP8 Engram part {} row_start mismatch (expected {expected_row_start})", path.display());
    let entry = header.get(expected_tensor).and_then(serde_json::Value::as_object)
        .with_context(|| format!("FP8 Engram part {} lacks tensor {expected_tensor}", path.display()))?;
    ensure!(entry.get("dtype").and_then(serde_json::Value::as_str) == Some("U8"),
        "FP8 Engram part {} tensor {expected_tensor} must be U8", path.display());
    let shape = entry.get("shape").and_then(serde_json::Value::as_array)
        .context("FP8 Engram part missing shape")?;
    ensure!(shape.len() == 2
        && shape[0].as_u64() == Some(expected_rows)
        && shape[1].as_u64() == Some(ENGRAM_FP8_ROW_BYTES as u64),
        "FP8 Engram part {} tensor {expected_tensor} shape mismatch (expected [{expected_rows}, {ENGRAM_FP8_ROW_BYTES}])",
        path.display());
    let offsets = entry.get("data_offsets").and_then(serde_json::Value::as_array)
        .context("FP8 Engram part missing data_offsets")?;
    let start = offsets.first().and_then(serde_json::Value::as_u64).context("bad data_offsets[0]")?;
    let end = offsets.get(1).and_then(serde_json::Value::as_u64).context("bad data_offsets[1]")?;
    let expected_bytes = expected_rows.checked_mul(ENGRAM_FP8_ROW_BYTES as u64)
        .context("FP8 Engram payload overflow")?;
    ensure!(end.checked_sub(start) == Some(expected_bytes),
        "FP8 Engram part {} payload extent mismatch", path.display());
    let data_offset = 8u64.checked_add(header_len).and_then(|v| v.checked_add(start))
        .context("FP8 Engram data offset overflow")?;
    ensure!(data_offset.checked_add(expected_bytes).is_some_and(|e| e <= file_len),
        "FP8 Engram part {} is truncated", path.display());
    Ok(data_offset)
}

impl Hy4EngramTable {
    pub fn layer(&self) -> usize { self.layer }
    pub fn rows(&self) -> u64 { self.table.table().rows() }
    pub fn row_bytes(&self) -> usize { self.table.row_bytes() }
    pub fn mapped(&self) -> &MappedTable { self.table.table() }
    pub fn token_table(&self) -> &Arc<TokenTable> { &self.table }
    pub fn cache(&self) -> &Arc<RowCache> { self.table.cache() }

    /// Open a Hy4-flash FP8 Engram layer from `checkpoint` using `shares` for its `RowCache`.
    /// Refuses checkpoints without `engram_fp8` (`engram-fp8-layer-*-part-*.safetensors`) by name.
    ///
    /// # Safety
    /// Checkpoint shard files must remain immutable while this table is mapped.
    pub unsafe fn open(checkpoint: &Checkpoint, cfg: &Hy4FlashConfig, layer: usize, shares: [usize; 2]) -> Result<Self> {
        ensure!(
            cfg.engram_head_width() == ENGRAM_EMBED_WIDTH,
            "Hy4-flash Engram head width {} differs from expected {ENGRAM_EMBED_WIDTH}",
            cfg.engram_head_width()
        );
        let expected_rows = cfg.engram_rows(layer)
            .with_context(|| format!("Hy4-flash layer {layer} is not an Engram layer"))?;
        let engram_fp8 = checkpoint
            .view
            .as_ref()
            .and_then(|v| v.engram_fp8())
            .with_context(|| {
                format!(
                    "Hy4-flash requires FP8 Engram shards (engram-fp8-layer-{layer:02}-part-*.safetensors) declared in cuteafd-mc.json"
                )
            })?;
        let entry = engram_fp8.layer(layer).with_context(|| {
            format!(
                "cuteafd-mc.json engram_fp8 lacks layer {layer} (expected engram-fp8-layer-{layer:02}-part-*.safetensors)"
            )
        })?;
        // SAFETY: forwarded from caller.
        unsafe { Self::open_fp8_parts(&checkpoint.snapshot, layer, entry, expected_rows, shares) }
    }

    /// Open strictly the `engram_fp8` strided parts for `layer` backed by [`TokenTable`].
    ///
    /// # Safety
    /// Referenced shard files must remain immutable while mapped.
    pub unsafe fn open_fp8_parts(
        snapshot: &Path,
        layer: usize,
        entry: &EngramFp8Layer,
        expected_rows: u64,
        shares: [usize; 2],
    ) -> Result<Self> {
        let total = entry.total_rows()?;
        ensure!(total == expected_rows,
            "Hy4 Engram FP8 layer {layer} has {total} rows, config expects {expected_rows}");
        let mut parts = Vec::with_capacity(entry.parts.len());
        let mut row_start = 0u64;
        for part in &entry.parts {
            ensure!(!part.file.is_empty()
                && Path::new(&part.file).components().all(|c| matches!(c, Component::Normal(_))),
                "invalid FP8 Engram part shard path {}", part.file);
            let path = snapshot.join(&part.file);
            let offset = read_fp8_part_header(&path, &part.tensor, row_start, part.rows)?;
            parts.push(TablePart { path, offset, rows: part.rows });
            row_start = row_start.checked_add(part.rows).context("FP8 Engram row_start overflow")?;
        }
        // SAFETY: forwarded from caller.
        let mapped = unsafe { MappedTable::open(&parts, RowFormat::of(DType::U8, ENGRAM_FP8_ROW_BYTES)?)? };
        mapped.prepare_reads();
        mapped.name_stats(format!("hy4_engram.{layer}.fp8_strided"));
        let table = TokenTable::new(mapped, shares, ENGRAM_HEADS)?;
        Ok(Self { layer, table })
    }

    /// Gather raw 100-byte FP8 rows through the shared [`RowCache`] (joining in-flight
    /// look-ahead reads and falling back synchronously on cache misses).
    pub fn gather_raw(&self, rows: &[u64], priority: ReadPriority, first_layer: bool) -> Result<Vec<u8>> {
        let mut out = vec![0u8; rows.len() * ENGRAM_FP8_ROW_BYTES];
        if rows.is_empty() {
            return Ok(out);
        }
        let prio = if first_layer { priority.first() } else { priority };
        self.table.gather_at_priority(rows, &mut out, prio)?;
        Ok(out)
    }

    /// Gather `rows` through [`RowCache`] and decode into little-endian BF16 bytes (`rows.len() * 192`).
    pub fn gather_decoded_bf16_into(
        &self,
        rows: &[u64],
        priority: ReadPriority,
        first_layer: bool,
        out_bf16_bytes: &mut [u8],
    ) -> Result<()> {
        let raw = self.gather_raw(rows, priority, first_layer)?;
        decode_fp8_rows_to_bf16_bytes(&raw, out_bf16_bytes)
    }

    /// Gather `rows` through [`RowCache`] and decode into `Vec<u16>` (`rows.len() * 96` BF16 values).
    pub fn gather_decoded_bf16(&self, rows: &[u64], priority: ReadPriority, first_layer: bool) -> Result<Vec<u16>> {
        let raw = self.gather_raw(rows, priority, first_layer)?;
        let mut out = vec![0u16; rows.len() * ENGRAM_EMBED_WIDTH];
        for (i, chunk) in raw.chunks_exact(ENGRAM_FP8_ROW_BYTES).enumerate() {
            let src: &[u8; ENGRAM_FP8_ROW_BYTES] = chunk.try_into().unwrap();
            let dst: &mut [u16; ENGRAM_EMBED_WIDTH] =
                (&mut out[i * ENGRAM_EMBED_WIDTH..(i + 1) * ENGRAM_EMBED_WIDTH]).try_into().unwrap();
            decode_fp8_row(src, dst);
        }
        Ok(out)
    }
}

/// Multi-layer Hy4-flash prompt row source over a forked 2-token history per layer.
/// Never mutates any committed request history; dropping the handle cancels cleanly.
struct Hy4EngramRowSource {
    hashers: Vec<Arc<Hy4EngramHasher>>,
    histories: Vec<Hy4EngramHistory>,
    tokens: Vec<u32>,
}

impl RowSource for Hy4EngramRowSource {
    fn tokens(&self) -> usize { self.tokens.len() }
    fn rows(&mut self, tokens: Range<usize>) -> Result<Vec<Vec<u64>>> {
        let slice = &self.tokens[tokens];
        let mut per_layer = Vec::with_capacity(self.hashers.len());
        for (hasher, history) in self.hashers.iter().zip(&mut self.histories) {
            let mut rows = Vec::with_capacity(slice.len() * hasher.heads());
            hasher.hash(history, slice, &mut rows)?;
            per_layer.push(rows);
        }
        Ok(per_layer)
    }
}

/// Earliest-possible multi-layer Engram look-ahead over all Hy4-flash Engram layers
/// (`[1, 11, 21, 31]`), built directly on [`lookahead::Lookahead`].
pub struct Hy4EngramLookahead {
    planner: Lookahead,
    hashers: Vec<Arc<Hy4EngramHasher>>,
}

impl Hy4EngramLookahead {
    pub fn new(tables: Vec<Arc<TokenTable>>, hashers: Vec<Arc<Hy4EngramHasher>>) -> Result<Self> {
        ensure!(!tables.is_empty() && tables.len() == hashers.len(), "Hy4 Engram look-ahead table/hasher count mismatch");
        let lookahead_tables = tables
            .into_iter()
            .map(|table| table as Arc<dyn LookaheadTable>)
            .collect();
        let planner = Lookahead::new("hy4_engram", lookahead_tables, 1)?;
        Ok(Self { planner, hashers })
    }

    /// The look-ahead over arbitrary tables: tests wrap the real token tables to
    /// decide when each read lands.
    #[cfg(test)]
    fn over(tables: Vec<Arc<dyn LookaheadTable>>, hashers: Vec<Arc<Hy4EngramHasher>>) -> Result<Self> {
        ensure!(!tables.is_empty() && tables.len() == hashers.len(), "Hy4 Engram look-ahead table/hasher count mismatch");
        Ok(Self { planner: Lookahead::new("hy4_engram", tables, 1)?, hashers })
    }

    pub fn remote(&self) -> bool { self.planner.remote() }

    /// Start reading all Engram layers' rows for `tokens` (starting at absolute position
    /// `start` after a forked `initial_tail`) into each layer's `RowCache` in N-layer deadline order.
    pub fn start(
        &self,
        tag: u64,
        start: u64,
        initial_tail: [u32; 2],
        tokens: Vec<u32>,
        units: UnitPlan,
    ) -> LookaheadHandle {
        let histories = (0..self.hashers.len())
            .map(|_| Hy4EngramHistory(initial_tail.to_vec()))
            .collect();
        let source = Hy4EngramRowSource {
            hashers: self.hashers.clone(),
            histories,
            tokens,
        };
        self.planner.start(tag, start, Box::new(source), units)
    }

    /// Hash candidate decode/verify tokens from each sequence's forked rolling tail and
    /// submit all Engram layers' rows to the shared look-ahead's decode share. Never blocks.
    pub fn decode(&self, members: &[([u32; 2], &[u32])]) -> Result<()> {
        let mut per_layer: Vec<Vec<u64>> = (0..self.hashers.len()).map(|_| Vec::new()).collect();
        for &(tail, tokens) in members {
            if tokens.is_empty() {
                continue;
            }
            for (hasher, out) in self.hashers.iter().zip(&mut per_layer) {
                let mut hist = Hy4EngramHistory(tail.to_vec());
                hasher.hash(&mut hist, tokens, out)?;
            }
        }
        if per_layer.iter().any(|rows| !rows.is_empty()) {
            self.planner.decode(0, per_layer);
        }
        Ok(())
    }
}

/// Read the small per-layer `lookup_table` (I32 `[120818]`) directly from its shard.
pub fn read_lookup_table(checkpoint: &Checkpoint, layer: usize, expected_len: usize) -> Result<Vec<i32>> {
    let name = format!("model.layers.{layer}.engram_layer.lookup_table");
    let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .with_context(|| format!("missing Hy4 Engram tensor {name}"))?;
    ensure!(tensor.meta.dtype == DType::I32 && tensor.meta.shape == [expected_len]
        && tensor.meta.byte_length == expected_len as u64 * 4,
        "invalid Hy4 Engram lookup_table header for {name}: {:?} {:?}", tensor.meta.dtype, tensor.meta.shape);
    let mut file = File::open(checkpoint.snapshot.join(&tensor.shard))?;
    file.seek(SeekFrom::Start(tensor.meta.byte_offset))?;
    let mut bytes = vec![0u8; expected_len * 4];
    file.read_exact(&mut bytes)?;
    Ok(bytes.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Total on-disk FP8 Engram table bytes across all declared layers.
/// Refuses checkpoints without `engram_fp8` (`engram-fp8-layer-*-part-*.safetensors`) by name,
/// and cross-checks `cuteafd-mc.json`'s `engram_fp8.total_bytes()` (fail-closed).
pub fn mapped_engram_bytes(checkpoint: &Checkpoint, cfg: &Hy4FlashConfig) -> Result<(u64, BTreeMap<usize, u64>)> {
    let engram_fp8 = checkpoint
        .view
        .as_ref()
        .and_then(|v| v.engram_fp8())
        .context("Hy4-flash requires FP8 Engram shards (engram-fp8-layer-*-part-*.safetensors) declared in cuteafd-mc.json")?;
    let mut per_layer = BTreeMap::new();
    let mut total = 0u64;
    for &layer in &cfg.engram_layers {
        let rows = cfg.engram_rows(layer).context("missing engram rows")?;
        let entry = engram_fp8.layer(layer).with_context(|| {
            format!("cuteafd-mc.json engram_fp8 lacks layer {layer} (expected engram-fp8-layer-{layer:02}-part-*.safetensors)")
        })?;
        ensure!(
            entry.total_rows()? == rows,
            "cuteafd-mc.json engram_fp8 layer {layer} rows differ from config ({rows})"
        );
        let bytes = rows.checked_mul(ENGRAM_FP8_ROW_BYTES as u64).context("engram bytes overflow")?;
        per_layer.insert(layer, bytes);
        total = total.checked_add(bytes).context("total engram bytes overflow")?;
    }
    let declared = engram_fp8.total_bytes().context("invalid cuteafd-mc.json engram_fp8 byte total")?;
    ensure!(
        declared == total,
        "cuteafd-mc.json engram_fp8 total bytes ({declared}) differs from config rows * {ENGRAM_FP8_ROW_BYTES} ({total})"
    );
    Ok((total, per_layer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mapped_table::row_cache::{Lookup, RowClass};
    use crate::formats::mapped_table::lookahead::RowLanding;
    use crate::formats::mapped_table::TableStats;
    use std::os::unix::fs::FileExt;

    #[test]
    fn numpy_pcg64_factors_match_reference_exactly() {
        let bound = (i64::MAX / 92_620 / 2) as u64;
        assert_eq!(bound, 49_791_470_723_681);
        for (layer, expected) in [
            (1usize, [81_853_575_799_317i64, 39_690_085_502_883, 24_229_915_018_165]),
            (11, [19_457_948_045_719, 44_729_556_510_419, 84_779_352_922_315]),
            (21, [72_974_230_057_307, 21_530_602_121_221, 55_998_057_243_097]),
            (31, [29_128_509_562_787, 16_745_155_128_523, 22_474_588_302_733]),
        ] {
            let mut rng = NumpyPcg64::from_seed(10_007u64 * (layer as u64 + 1));
            let got = [
                (rng.next_bounded_u64(bound) as i64) * 2 + 1,
                (rng.next_bounded_u64(bound) as i64) * 2 + 1,
                (rng.next_bounded_u64(bound) as i64) * 2 + 1,
            ];
            assert_eq!(got, expected, "layer {layer}");
        }
    }

    #[test]
    fn decode_fp8_row_matches_exact_e4m3_scale_product() {
        let scale = 0.25f32;
        let mut packed = [0u8; ENGRAM_FP8_ROW_BYTES];
        for j in 0..ENGRAM_EMBED_WIDTH {
            // Cover positive/negative zeros, subnormals, normals, and max finite 448 (0x7e).
            let code = match j % 8 {
                0 => 0x00,
                1 => 0x80,
                2 => 0x01, // smallest subnormal 2^-9
                3 => 0x38, // 1.0
                4 => 0xb8, // -1.0
                5 => 0x44, // 3.0
                6 => 0x7e, // 448.0
                _ => 0xfe, // -448.0
            };
            packed[j] = code;
        }
        packed[96..100].copy_from_slice(&scale.to_le_bytes());
        let mut decoded = [0u16; ENGRAM_EMBED_WIDTH];
        decode_fp8_row(&packed, &mut decoded);
        assert_eq!(decoded[0], 0x0000);
        assert_eq!(decoded[1], 0x8000);
        assert_eq!(decoded[3], f32_to_bf16_rne(0.25));
        assert_eq!(decoded[4], f32_to_bf16_rne(-0.25));
        assert_eq!(decoded[6], f32_to_bf16_rne(112.0));
        assert_eq!(decoded[7], f32_to_bf16_rne(-112.0));

        let mut bytes = vec![0u8; ENGRAM_EMBED_WIDTH * 2];
        decode_fp8_rows_to_bf16_bytes(&packed, &mut bytes).unwrap();
        for (j, &u) in decoded.iter().enumerate() {
            assert_eq!(&bytes[j * 2..j * 2 + 2], &u.to_le_bytes());
        }
    }

    #[test]
    fn advance_tail_commits_only_accepted_prefix() {
        let start = [10, 20];
        assert_eq!(advance_tail(start, &[30, 40, 50], 0), [10, 20]);
        assert_eq!(advance_tail(start, &[30, 40, 50], 1), [20, 30]);
        assert_eq!(advance_tail(start, &[30, 40, 50], 2), [30, 40]);
        assert_eq!(advance_tail(start, &[30, 40, 50], 3), [40, 50]);
        assert_eq!(advance_tail(start, &[30], 5), [20, 30]);
    }

    /// Rows per head in the toy tables: the 128-token prompt reaches only part of
    /// each head's range, so decode candidates can be chosen off the prompt's rows.
    const TOY_HEAD_ROWS: u64 = 256;

    fn toy_hasher(layer: usize) -> Arc<Hy4EngramHasher> {
        let sizes = vec![TOY_HEAD_ROWS; ENGRAM_HEADS];
        let offsets = (0..ENGRAM_HEADS).map(|h| h as u64 * TOY_HEAD_ROWS).collect();
        Arc::new(Hy4EngramHasher {
            layer,
            lookup: (0..256i32).collect(),
            factors: vec![13 + layer as i64 * 2, 29 + layer as i64 * 2, 53 + layer as i64 * 2],
            sizes,
            offsets,
            heads_per_ngram: 8,
            pad_id: 0,
            masked_tokens: [0, 1, 2],
        })
    }

    fn toy_table(layer: usize, rows: u64) -> Result<(tempfile::NamedTempFile, Arc<TokenTable>)> {
        let file = tempfile::NamedTempFile::new()?;
        file.as_file().set_len(rows * ENGRAM_FP8_ROW_BYTES as u64)?;
        for row in 0..rows {
            let mut buf = [0u8; ENGRAM_FP8_ROW_BYTES];
            for j in 0..ENGRAM_EMBED_WIDTH {
                buf[j] = ((layer * 31 + row as usize * 7 + j) % 0x7e) as u8;
            }
            buf[96..100].copy_from_slice(&1.0f32.to_le_bytes());
            file.as_file().write_all_at(&buf, row * ENGRAM_FP8_ROW_BYTES as u64)?;
        }
        // SAFETY: file stays owned for the test's lifetime.
        let mapped = unsafe {
            MappedTable::single(
                file.path(),
                0,
                rows,
                RowFormat::of(DType::U8, ENGRAM_FP8_ROW_BYTES)?,
            )?
        };
        mapped.prepare_reads();
        mapped.name_stats(format!("hy4_test_engram.{layer}.fp8_strided"));
        let shares = TokenTable::shares(128, 1, 16, ENGRAM_HEADS);
        Ok((file, TokenTable::new(mapped, shares, ENGRAM_HEADS)?))
    }

    /// The test's clock for look-ahead reads: one shared record of every read the
    /// planner starts, in order, and one gate per table. A read lands (its poll
    /// returns true) only after its table's gate opens, so the test decides the
    /// landing order instead of the scheduler and the disk.
    #[derive(Default)]
    struct ReadLog {
        started: std::sync::Mutex<Vec<(usize, ReadPriority, Vec<u64>)>>,
        gates: Vec<std::sync::atomic::AtomicBool>,
    }

    impl ReadLog {
        fn new(tables: usize) -> Arc<Self> {
            Arc::new(Self { started: Default::default(), gates: (0..tables).map(|_| Default::default()).collect() })
        }
        fn open(&self, table: usize) { self.gates[table].store(true, std::sync::atomic::Ordering::Release); }
        fn reads(&self) -> Vec<(usize, ReadPriority, Vec<u64>)> { self.started.lock().unwrap().clone() }
        /// Block until the planner has started `count` reads; it cannot start more
        /// than the deadline order allows, so this is the order, not a timeout.
        fn await_reads(&self, count: usize) -> Vec<(usize, ReadPriority, Vec<u64>)> {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            loop {
                let reads = self.reads();
                if reads.len() >= count { return reads; }
                assert!(std::time::Instant::now() < deadline, "the planner started {} of {count} reads: {reads:?}", reads.len());
                std::thread::yield_now();
            }
        }
    }

    struct GatedTable { table: Arc<TokenTable>, index: usize, log: Arc<ReadLog> }
    struct GatedLanding { inner: Box<dyn RowLanding>, index: usize, log: Arc<ReadLog> }

    impl LookaheadTable for GatedTable {
        fn cache(&self) -> &Arc<RowCache> { self.table.cache() }
        fn start_read(&self, rows: &[u64], priority: ReadPriority) -> Result<Box<dyn RowLanding>> {
            self.log.started.lock().unwrap().push((self.index, priority, rows.to_vec()));
            Ok(Box::new(GatedLanding { inner: self.table.start_read(rows, priority)?, index: self.index, log: Arc::clone(&self.log) }))
        }
        fn stats(&self) -> &TableStats { LookaheadTable::stats(self.table.as_ref()) }
        fn rows_per_token(&self) -> usize { LookaheadTable::rows_per_token(self.table.as_ref()) }
        fn placement(&self) -> crate::formats::mapped_table::placement::Placement { self.table.placement() }
    }

    impl RowLanding for GatedLanding {
        fn poll(&mut self) -> Result<bool> {
            // The real read keeps landing underneath; only its publication waits for the gate.
            let landed = self.inner.poll()?;
            Ok(landed && self.log.gates[self.index].load(std::sync::atomic::Ordering::Acquire))
        }
        fn row(&self, index: usize) -> [&[u8]; 2] { self.inner.row(index) }
    }

    fn distinct_rows(hasher: &Hy4EngramHasher, tail: [u32; 2], tokens: &[u32]) -> Result<Vec<u64>> {
        let mut rows = Vec::new();
        hasher.hash(&mut Hy4EngramHistory(tail.to_vec()), tokens, &mut rows)?;
        rows.sort_unstable();
        rows.dedup();
        Ok(rows)
    }

    /// Deterministic (#2723): every read lands only when the test opens its table,
    /// so the asserted order is the planner's deadline order, never wall-clock
    /// scheduling. Table k + 1 of the prompt starts only after table k landed; once
    /// all four landed, every layer's demand gather hits with no fallback read; a
    /// decode round reads every layer's candidate rows at decode priority (layer 0
    /// first), and the verify gathers then hit them.
    #[test]
    fn four_layer_lookahead_fills_caches_in_deadline_order_and_decode_prefetch_hits() -> Result<()> {
        let layers = [1usize, 11, 21, 31];
        let hashers: Vec<Arc<Hy4EngramHasher>> = layers.iter().map(|&l| toy_hasher(l)).collect();
        let mut files = Vec::new();
        let mut tables = Vec::new();
        for &l in &layers {
            let (f, t) = toy_table(l, ENGRAM_HEADS as u64 * TOY_HEAD_ROWS)?;
            files.push(f);
            tables.push(t);
        }
        let log = ReadLog::new(layers.len());
        let gated = tables.iter().enumerate().map(|(index, table)|
            Arc::new(GatedTable { table: Arc::clone(table), index, log: Arc::clone(&log) }) as Arc<dyn LookaheadTable>).collect();
        let ahead = Hy4EngramLookahead::over(gated, hashers.clone())?;
        let tokens: Vec<u32> = (3..131).collect();
        let prompt: Vec<Vec<u64>> = hashers.iter().map(|h| distinct_rows(h, [0, 0], &tokens)).collect::<Result<_>>()?;
        // One piece covers the 128-token unit, so each table reads its rows in one batch.
        assert!(prompt.iter().all(|rows| !rows.is_empty() && rows.len() <= 2048));
        let handle = ahead.start(1, 0, [0, 0], tokens.clone(), UnitPlan::new(vec![128], 1));
        let rows_read = |reads: &[(usize, ReadPriority, Vec<u64>)], table: usize| {
            let mut rows: Vec<u64> = reads.iter().filter(|read| read.0 == table).flat_map(|read| read.2.iter().copied()).collect();
            rows.sort_unstable();
            rows
        };

        // Deadline order: with nothing landed only table 0 may read; table k + 1
        // starts only after table k landed. Each table's reads (one per piece, rows
        // another piece already holds are not read twice) cover exactly its prompt rows.
        for table in 0..layers.len() {
            let wait = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while rows_read(&log.reads(), table) != prompt[table] {
                assert!(std::time::Instant::now() < wait, "table {table} read {:?}", rows_read(&log.reads(), table));
                std::thread::yield_now();
            }
            // Shut gates hold the planner here: give it time to misbehave.
            std::thread::sleep(std::time::Duration::from_millis(20));
            let reads = log.reads();
            assert!(reads.iter().all(|read| read.0 <= table), "a later table read before table {table} landed: {reads:?}");
            assert!(reads.windows(2).all(|pair| pair[0].0 <= pair[1].0), "reads out of table order: {reads:?}");
            assert!(reads.iter().all(|read| read.1 == ReadPriority::Prefill), "{reads:?}");
            assert_eq!(rows_read(&reads, table), prompt[table], "table {table} read its rows twice");
            log.open(table);
        }
        // Every prompt row is published once the gates are open.
        let wait = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while tables.iter().zip(&prompt).any(|(t, rows)| t.table().stats().snapshot().lookahead_rows < rows.len() as u64) {
            assert!(std::time::Instant::now() < wait, "opened look-ahead reads did not publish");
            std::thread::yield_now();
        }

        // Every layer's demand gather over the prefetched prompt hits RowCache with 0 fallback misses.
        for (idx, (hasher, table)) in hashers.iter().zip(&tables).enumerate() {
            let mut hist = Hy4EngramHistory(vec![0, 0]);
            let mut row_ids = Vec::new();
            hasher.hash(&mut hist, &tokens, &mut row_ids)?;
            let before = table.table().stats().snapshot();
            let mut out = vec![0u8; row_ids.len() * ENGRAM_FP8_ROW_BYTES];
            let prio = if idx == 0 { ReadPriority::PrefillFirst } else { ReadPriority::Prefill };
            table.gather_at_priority(&row_ids, &mut out, prio)?;
            let delta = table.table().stats().snapshot().since(&before);
            assert_eq!((delta.stall_prefill_fallback_rows, delta.stall_prefill_pending_rows), (0, 0),
                "layer {} did not find its prefetched rows", layers[idx]);
            assert_eq!(delta.cache_hit_rows, prompt[idx].len() as u64, "layer {}", layers[idx]);
        }
        assert!(!handle.commit(128));
        drop(handle);

        // A decode candidate with rows the prompt never read in every layer; the
        // prompt's rows stay cached, so a decode round reads only the new ones.
        let committed_tail = [129u32, 130];
        let fresh = |hasher: &Hy4EngramHasher, prompt: &[u64], token: u32| -> Vec<u64> {
            distinct_rows(hasher, committed_tail, &[token]).unwrap().into_iter().filter(|row| prompt.binary_search(row).is_err()).collect()
        };
        let cand = (131..256u32).find(|&token| hashers.iter().zip(&prompt).all(|(h, rows)| !fresh(h, rows, token).is_empty()))
            .expect("a decode candidate with new rows in every layer");
        let decode: Vec<Vec<u64>> = hashers.iter().map(|h| distinct_rows(h, committed_tail, &[cand])).collect::<Result<_>>()?;
        let prefill_reads = log.reads().len();
        ahead.decode(&[(committed_tail, &[cand])])?;
        // One decode round starts every layer's read at once, layer 0 first, at decode priority.
        let round = log.await_reads(prefill_reads + layers.len())[prefill_reads..].to_vec();
        assert_eq!(round.len(), layers.len(), "{round:?}");
        for (table, read) in round.iter().enumerate() {
            assert_eq!((read.0, read.1), (table, if table == 0 { ReadPriority::DecodeFirst } else { ReadPriority::Decode }));
            assert_eq!(read.2, fresh(&hashers[table], &prompt[table], cand), "layer {} decode rows", layers[table]);
        }
        // The planner holds every decode row's fill (or it is cached): the verify gathers find them landed.
        let pins: Vec<_> = tables.iter().zip(&decode).map(|(table, rows)| table.cache().acquire(rows, RowClass::Decode)).collect();
        for (table, (pins, lookups)) in pins.iter().enumerate() {
            for lookup in lookups {
                match *lookup {
                    Lookup::Ready(_) => {}
                    Lookup::Wait(slot) => assert!(pins.wait(slot), "layer {}: a decode prefetch read failed", layers[table]),
                    other => panic!("layer {}: a decode row nobody prefetched: {other:?}", layers[table]),
                }
            }
        }
        drop(pins);
        for (idx, (table, rows)) in tables.iter().zip(&decode).enumerate() {
            let before = table.table().stats().snapshot();
            let mut out = vec![0u8; rows.len() * ENGRAM_FP8_ROW_BYTES];
            let prio = if idx == 0 { ReadPriority::DecodeFirst } else { ReadPriority::Decode };
            table.gather_at_priority(rows, &mut out, prio)?;
            let delta = table.table().stats().snapshot().since(&before);
            assert_eq!((delta.stall_decode_fallback_rows, delta.cache_hit_rows), (0, rows.len() as u64), "layer {}", layers[idx]);
        }
        drop(ahead);
        drop(files);
        Ok(())
    }
}
