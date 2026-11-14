//! Kimi NoPE MLA, residual-mix and Situ programs, resolved before capture.
//! The engine owns weights, caches, arenas, outputs and scratch through completion.
use crate::programs::{Program, ProgramSpec, Programs, Scalar};
use crate::CuteafdDeviceBuffer as Buffer;
use std::ffi::c_void;
use cuteafd_core::MlaCacheFormat;
use thiserror::Error;

pub const MLA_RECORD_BYTES: usize = 1152;
pub const MLA_PAGE_ROWS: usize = 64;
const PRODUCER: &[&str] = &["x", "kv_slots", "w_qkv_a", "q_a_norm", "kv_a_norm", "w_q_b", "w_uk", "kv_cache", "query", "q_resid", "scratch"];
const ATTENTION: &[&str] = &["query", "kv_cache", "page_table", "cache_seqlens", "cu_seqlens_q", "attn", "lse", "scratch"];
const OUTPUT: &[&str] = &["attn", "x", "w_uv", "w_g", "w_o", "out", "scratch"];

#[derive(Debug, Error)]
pub enum KimiMlaError {
    #[error("Kimi MLA invalid {0}")]
    Invalid(&'static str),
    #[error("Kimi MLA program: {0}")]
    Program(#[from] anyhow::Error),
}
type Result<T> = std::result::Result<T, KimiMlaError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KimiMlaMode { Decode, Verify, Extend }
impl KimiMlaMode {
    fn name(self) -> &'static str {
        match self { Self::Decode => "decode", Self::Verify => "verify", Self::Extend => "extend" }
    }
}

/// BF16 weights, packed once from the checkpoint. UK is [heads,512,128],
/// UV is [heads,128,512]; q_a and kv_a are concatenated by output rows.
#[derive(Clone, Copy, Debug)]
pub struct KimiMlaWeights {
    pub qkv_a: Buffer, pub q_a_norm: Buffer, pub kv_a_norm: Buffer,
    pub q_b: Buffer, pub uk: Buffer, pub uv: Buffer, pub gate: Buffer, pub output: Buffer,
}

/// All extents are bytes. Buffers are dense and on one CUDA device. The page
/// table and slots must address allocated pages; slots may be -1 to skip a
/// cache write. Each request has at least one query and sufficient KV history.
#[derive(Clone, Copy, Debug)]
pub struct KimiMlaBatch {
    pub rows: usize, pub requests: usize, pub page_table_width: usize,
    pub cache_pages: usize, pub active_splits: usize,
    pub x: Buffer, pub slots: Buffer, pub cache: Buffer,
    pub page_table: Buffer, pub lengths: Buffer, pub query_offsets: Buffer,
    pub query: Buffer, pub q_resid: Buffer, pub attention: Buffer, pub lse: Buffer,
    pub output: Buffer, pub producer_scratch: Buffer, pub attention_scratch: Buffer,
    pub output_scratch: Buffer,
}

/// Resolved producer -> full causal dense MLA -> UV/sigmoid gate/o_proj.
/// With 48 heads output is one TP2 partial; the caller performs the TP meet.
pub struct KimiMla<'a> {
    producer: Program<'a>, attention: Program<'a>, output: Program<'a>,
    heads: usize, rows: usize, context: usize, splits: usize,
    scratch: [usize; 3],
    cache_format: MlaCacheFormat,
}

impl<'a> KimiMla<'a> {
    pub fn load(programs: &'a Programs<'a>, heads: usize, rows: usize, mode: KimiMlaMode, cache_format: MlaCacheFormat) -> Result<Self> {
        if !matches!(heads, 48 | 96) || !(1..=4096).contains(&rows) {
            return Err(KimiMlaError::Invalid("head/row geometry"));
        }
        let suffix = if heads == 48 { "" } else { "_h96" };
        let cache = cache_format.program_suffix();
        let producer = programs.program(&format!("kimik3_mla_producer{suffix}{cache}_m{rows}"), PRODUCER)?;
        let attention = programs.program(&format!("kimik3_dense_mla_{}{suffix}{cache}_m{rows}", mode.name()), ATTENTION)?;
        let output = programs.program(&format!("kimik3_mla_output{suffix}_m{rows}"), OUTPUT)?;
        let (context, splits, scratch) = validate_manifest(
            [producer.spec(), attention.spec(), output.spec()], heads, rows, mode, cache_format)?;
        Ok(Self { producer, attention, output, heads, rows, context, splits, scratch, cache_format })
    }

    pub fn scratch_bytes(&self) -> [usize; 3] { self.scratch }
    pub fn max_context(&self) -> usize { self.context }
    pub fn splits(&self) -> usize { self.splits }

    /// Shared dense attention after an independently projected producer. The
    /// projected route retains the same cache/query layout and attention math.
    pub fn attention(&self,b:KimiMlaBatch,stream:*mut c_void)->Result<()> {
        validate_projected_attention(b,self.heads,self.rows,self.context,self.splits,self.scratch[1],
            self.cache_format.record_bytes())?;
        // SAFETY: metadata, device and extents checked; the caller publishes
        // query/cache/indices first and retains storage through stream drain.
        unsafe {self.attention.launch(&[b.query.ptr,b.cache.ptr,b.page_table.ptr,b.lengths.ptr,
            b.query_offsets.ptr,b.attention.ptr,b.lse.ptr,b.attention_scratch.ptr],
            &[Scalar::I32(b.rows as i32),Scalar::I32(b.requests as i32),Scalar::I64(b.page_table_width as i64),
                Scalar::I32(b.active_splits as i32)],stream)?;}
        Ok(())
    }

    /// Rank-local projection and owner-only cache writes. Slots are local to
    /// this rank's compact cache; -1 suppresses writes owned by the peer.
    pub fn produce(&self, weights: KimiMlaWeights, b: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        validate_producer(weights, b, self.heads, self.rows, self.scratch[0], self.cache_format.record_bytes())?;
        // SAFETY: phase-local extents, devices and writer aliases are checked;
        // caller retains the compact cache, fixed query and exact stream.
        unsafe { self.producer.launch(&[b.x.ptr, b.slots.ptr, weights.qkv_a.ptr, weights.q_a_norm.ptr,
            weights.kv_a_norm.ptr, weights.q_b.ptr, weights.uk.ptr, b.cache.ptr, b.query.ptr,
            b.q_resid.ptr, b.producer_scratch.ptr], &[Scalar::I32(b.rows as i32)], stream)?; }
        Ok(())
    }

    /// UV/gate/output after the fixed-order two-owner LSE combination. This
    /// phase does not read the cache, query or local partial scratch.
    pub fn output(&self, weights: KimiMlaWeights, b: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        validate_output(weights, b, self.heads, self.rows, self.scratch[2])?;
        // SAFETY: phase-local operands are validated; the combine precedes this
        // launch on the same retained compute stream.
        unsafe { self.output.launch(&[b.attention.ptr, b.x.ptr, weights.uv.ptr, weights.gate.ptr,
            weights.output.ptr, b.output.ptr, b.output_scratch.ptr], &[Scalar::I32(b.rows as i32)], stream)?; }
        Ok(())
    }

    /// Enqueue without allocation of device storage, synchronization or lazy
    /// kernel loading. The owning engine serializes accesses and retains all
    /// buffers through completion and every replay of a captured graph.
    pub fn launch(&self, weights: KimiMlaWeights, b: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        if b.rows == 0 || b.rows > self.rows || b.requests == 0 || b.requests > b.rows
            || b.page_table_width == 0 || b.page_table_width > self.context.div_ceil(MLA_PAGE_ROWS)
            || b.cache_pages == 0 || b.cache_pages > i32::MAX as usize
            || b.active_splits != self.splits {
            return Err(KimiMlaError::Invalid("batch/context/split geometry"));
        }
        let r = b.rows;
        let h = self.heads;
        let device = b.x.device_id;
        let requirements = [
            (b.x, r * 7168 * 2), (b.slots, r * 8),
            (b.cache, b.cache_pages * MLA_PAGE_ROWS * self.cache_format.record_bytes()),
            (b.page_table, b.requests * b.page_table_width * 4),
            (b.lengths, b.requests * 4), (b.query_offsets, (b.requests + 1) * 4),
            (b.query, r * h * 576 * 2), (b.q_resid, r * 1536 * 2),
            (b.attention, r * h * 512 * 2), (b.lse, r * h * 4), (b.output, r * 7168 * 2),
            (b.producer_scratch, self.scratch[0]), (b.attention_scratch, self.scratch[1]),
            (b.output_scratch, self.scratch[2]),
            (weights.qkv_a, 2112 * 7168 * 2), (weights.q_a_norm, 1536 * 2),
            (weights.kv_a_norm, 512 * 2), (weights.q_b, h * 192 * 1536 * 2),
            (weights.uk, h * 512 * 128 * 2), (weights.uv, h * 128 * 512 * 2),
            (weights.gate, h * 128 * 7168 * 2), (weights.output, 7168 * h * 128 * 2),
        ];
        validate_buffers(&requirements, device)?;
        // SAFETY: extents, alignment and devices are checked; the engine owns
        // the live buffers and stream, ordered metadata and graph lifetimes.
        unsafe {
            self.producer.launch(&[b.x.ptr, b.slots.ptr, weights.qkv_a.ptr, weights.q_a_norm.ptr,
                weights.kv_a_norm.ptr, weights.q_b.ptr, weights.uk.ptr, b.cache.ptr, b.query.ptr,
                b.q_resid.ptr, b.producer_scratch.ptr], &[Scalar::I32(r as i32)], stream)?;
            self.attention.launch(&[b.query.ptr, b.cache.ptr, b.page_table.ptr, b.lengths.ptr,
                b.query_offsets.ptr, b.attention.ptr, b.lse.ptr, b.attention_scratch.ptr],
                &[Scalar::I32(r as i32), Scalar::I32(b.requests as i32), Scalar::I64(b.page_table_width as i64),
                  Scalar::I32(b.active_splits as i32)], stream)?;
            self.output.launch(&[b.attention.ptr, b.x.ptr, weights.uv.ptr, weights.gate.ptr,
                weights.output.ptr, b.output.ptr, b.output_scratch.ptr], &[Scalar::I32(r as i32)], stream)?;
        }
        Ok(())
    }
}

/// K3.1's native BF16 MLA projection seams, without any dense-attention
/// program. The shared lane inserts index production and selection between them.
pub struct KimiMlaProjection<'a> {
    producer: Program<'a>, output: Program<'a>,
    heads: usize, rows: usize, scratch: [usize; 2], cache_format: MlaCacheFormat,
}

impl<'a> KimiMlaProjection<'a> {
    pub fn load_k31(programs: &'a Programs<'a>, heads: usize, rows: usize,
        cache_format: MlaCacheFormat) -> Result<Self> {
        let suffix = if heads == 48 { "" } else { "_h96" };
        let cache = cache_format.program_suffix();
        let producer = programs.program(&format!("kimik31_mla_producer{suffix}{cache}_m{rows}"), PRODUCER)?;
        let output = programs.program(&format!("kimik31_mla_output{suffix}_m{rows}"), OUTPUT)?;
        let scratch = validate_k31_projection_manifest([producer.spec(), output.spec()], heads, rows, cache_format)?;
        Ok(Self { producer, output, heads, rows, scratch, cache_format })
    }

    pub fn scratch_bytes(&self) -> [usize; 2] { self.scratch }

    pub fn produce(&self, weights: KimiMlaWeights, b: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        validate_producer(weights, b, self.heads, self.rows, self.scratch[0], self.cache_format.record_bytes())?;
        // SAFETY: validated native operands and disjoint writes; caller retains
        // the stream and all backing arenas through completion and graph replay.
        unsafe { self.producer.launch(&[b.x.ptr, b.slots.ptr, weights.qkv_a.ptr, weights.q_a_norm.ptr,
            weights.kv_a_norm.ptr, weights.q_b.ptr, weights.uk.ptr, b.cache.ptr, b.query.ptr,
            b.q_resid.ptr, b.producer_scratch.ptr], &[Scalar::I32(b.rows as i32)], stream)?; }
        Ok(())
    }

    pub fn output(&self, weights: KimiMlaWeights, b: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        validate_output(weights, b, self.heads, self.rows, self.scratch[1])?;
        // SAFETY: validated output phase; selected attention or peer combination
        // precedes this launch on the retained stream, with stable buffer owners.
        unsafe { self.output.launch(&[b.attention.ptr, b.x.ptr, weights.uv.ptr, weights.gate.ptr,
            weights.output.ptr, b.output.ptr, b.output_scratch.ptr], &[Scalar::I32(b.rows as i32)], stream)?; }
        Ok(())
    }
}

fn validate_k31_projection_manifest(specs: [&ProgramSpec; 2], heads: usize, rows: usize,
    format: MlaCacheFormat) -> Result<[usize; 2]> {
    if !matches!(heads, 48 | 96) || !(1..=4096).contains(&rows) {
        return Err(KimiMlaError::Invalid("K3.1 projection head/row geometry"));
    }
    let align = |n: usize| n.div_ceil(1024) * 1024;
    let scratch = [align(rows * 2112 * 2) + align(rows * heads * 192 * 2),
        2 * align(rows * heads * 128 * 2)];
    for (i, spec) in specs.iter().enumerate() {
        let g = &spec.geometry;
        let pointers = if i == 0 { PRODUCER } else { OUTPUT };
        if spec.capacity_rows as usize != rows || spec.scalar_kinds != b"i"
            || spec.pointers.iter().map(String::as_str).ne(pointers.iter().copied())
            || g["norm_eps"].as_f64() != Some(1e-5)
            || (i == 0 && g["kv_cache"].as_str() != Some(format.label()))
            || [("heads", heads), ("hidden", 7168), ("q_lora_rank", 1536), ("kv_lora_rank", 512),
                ("qk_nope_dim", 128), ("qk_rope_dim", 64), ("v_head_dim", 128), ("page_rows", 64)]
                .iter().any(|&(key, value)| g[key].as_u64() != Some(value as u64))
            || spec.scratch.len() != 1 || spec.scratch.get("scratch").copied() != Some(scratch[i] as u64) {
            return Err(KimiMlaError::Invalid("K3.1 projection manifest geometry/ABI/scratch"));
        }
    }
    Ok(scratch)
}

const INDEX_PRODUCER: &[&str] = &["x", "q_resid", "positions", "index_slots", "cos_sin",
    "w_iq", "w_ik", "k_norm_w", "k_norm_b", "index_cache", "q_fp8", "head_weights", "scratch"];

/// Native BF16 index projections. Key/head rows share one packed arena view;
/// neither projection participates in the selected dense-core conversion.
#[derive(Clone, Copy, Debug)]
pub struct KimiIndexWeights {
    pub query: Buffer, pub key_heads: Buffer, pub norm: Buffer, pub bias: Buffer,
}

#[derive(Clone, Copy, Debug)]
pub struct KimiIndexBatch {
    pub rows: usize, pub cache_pages: usize,
    pub x: Buffer, pub q_resid: Buffer, pub positions: Buffer, pub slots: Buffer,
    /// NoPE ignores this retained placeholder, matching the shared producer ABI.
    pub cos_sin: Buffer,
    pub cache: Buffer, pub query_fp8: Buffer, pub head_weights: Buffer, pub scratch: Buffer,
}

pub struct KimiIndexProducer<'a> {
    program: Program<'a>, rows: usize, scratch: usize,
}
impl<'a> KimiIndexProducer<'a> {
    pub fn load(programs: &'a Programs<'a>, rows: usize) -> Result<Self> {
        let name = format!("kimik31_index_producer_m{rows}");
        let scratch = validate_index_manifest(programs.spec(&name)?, rows)?;
        Ok(Self { program: programs.program(&name, INDEX_PRODUCER)?, rows, scratch })
    }

    pub fn scratch_bytes(&self) -> usize { self.scratch }

    /// The caller orders normalized q-LoRA and hidden inputs before this stage,
    /// and retains native weights/cache/outputs through stream and graph use.
    pub fn launch(&self, weights: KimiIndexWeights, b: KimiIndexBatch, stream: *mut c_void) -> Result<()> {
        validate_index_batch(weights, b, self.rows, self.scratch)?;
        // SAFETY: extents, devices and write aliases checked. The owning lane
        // supplies valid physical slots and retains the stream and storage.
        unsafe { self.program.launch(&[b.x.ptr, b.q_resid.ptr, b.positions.ptr, b.slots.ptr,
            b.cos_sin.ptr, weights.query.ptr, weights.key_heads.ptr, weights.norm.ptr,
            weights.bias.ptr, b.cache.ptr, b.query_fp8.ptr, b.head_weights.ptr, b.scratch.ptr],
            &[Scalar::I32(b.rows as i32)], stream)?; }
        Ok(())
    }
}

fn validate_index_manifest(spec: &ProgramSpec, rows: usize) -> Result<usize> {
    let g = &spec.geometry;
    if !(1..=4096).contains(&rows) || spec.capacity_rows as usize != rows
        || spec.scalar_kinds != b"i" || spec.pointers.iter().map(String::as_str).ne(INDEX_PRODUCER.iter().copied())
        || [("hidden", 7168), ("q_lora_rank", 1536), ("index_heads", 32),
            ("index_head_dim", 128), ("index_topk", 2048), ("page_rows", 64), ("max_rows", rows)]
            .iter().any(|&(key, value)| g[key].as_u64() != Some(value as u64))
        || g["index_norm_eps"].as_f64() != Some(1e-5) || g["rope"].as_bool() != Some(false)
        || g["scale_format"].as_str() != Some("ue8m0") || g["scale_storage"].as_str() != Some("float32")
        || g["query_amax_floor"].as_f64() != Some(1e-10) || g["key_amax_floor"].as_f64() != Some(1e-4) {
        return Err(KimiMlaError::Invalid("K3.1 index manifest arithmetic/geometry/ABI"));
    }
    let align = |n: usize| n.div_ceil(1024) * 1024;
    let scratch = align(rows * 4096 * 2) + align(rows * 160 * 2);
    if spec.scratch.len() != 1 || spec.scratch.get("scratch").copied() != Some(scratch as u64) {
        return Err(KimiMlaError::Invalid("K3.1 index manifest scratch"));
    }
    Ok(scratch)
}

fn validate_index_batch(w: KimiIndexWeights, b: KimiIndexBatch, rows: usize, scratch: usize) -> Result<()> {
    if b.rows == 0 || b.rows > rows || b.cache_pages == 0 || b.cache_pages > i32::MAX as usize {
        return Err(KimiMlaError::Invalid("K3.1 index batch/cache geometry"));
    }
    let cache = b.cache_pages.checked_mul(8448).ok_or(KimiMlaError::Invalid("index cache overflow"))?;
    validate_phase(&[(b.x, b.rows * 7168 * 2), (b.q_resid, b.rows * 1536 * 2),
        (b.positions, b.rows * 8), (b.slots, b.rows * 8), (b.cos_sin, 64 * 4),
        (w.query, 4096 * 1536 * 2), (w.key_heads, 160 * 7168 * 2),
        (w.norm, 128 * 2), (w.bias, 128 * 2), (b.cache, cache),
        (b.query_fp8, b.rows * 4096), (b.head_weights, b.rows * 32 * 4),
        (b.scratch, scratch)], 9..13, b.x.device_id)
}

const INDEX_TOPK: &[&str] = &["q_fp8", "weights", "index_k_cache", "page_table", "cache_lengths", "output_indices", "scratch"];
const SCORED_TOPK: &[&str] = &["q", "weights", "cache", "table", "lengths", "indices", "scores", "scratch"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KimiIndexMode { Decode, Prefill }
impl KimiIndexMode {
    fn name(self) -> &'static str { match self { Self::Decode => "decode", Self::Prefill => "prefill" } }
}

#[derive(Clone, Copy, Debug)]
pub struct KimiIndexSelection {
    pub rows: usize, pub cache_pages: usize, pub table_width: usize, pub table_stride: usize,
    /// Interleaved half tables use (2, rank); (1, 0) scores the whole cache.
    pub page_stride: usize, pub page_offset: usize,
    pub query_fp8: Buffer, pub head_weights: Buffer, pub cache: Buffer, pub table: Buffer,
    /// Per-query causal frontier, including the query's current token.
    pub lengths: Buffer,
    pub indices: Buffer, pub scores: Buffer, pub scratch: Buffer,
}

/// Shared DSA selector. Context scores each rank's parity-owned half through
/// its compact table and the shared candidate merge, returning global logical
/// candidates; (1, 0) keeps the ordinary whole-cache behavior byte for byte.
pub struct KimiIndexTopK<'a> {
    program: Program<'a>, rows: usize, pages: usize, mode: KimiIndexMode,
    scored: bool, runtime_sms: Option<i32>, scratch: usize,
}
impl<'a> KimiIndexTopK<'a> {
    pub fn load(programs: &'a Programs<'a>, rows: usize, pages: usize, mode: KimiIndexMode,
        context: bool) -> Result<Self> {
        let stem = if context { "scored_index_topk" } else { "index_topk" };
        let name = format!("kimik31_{stem}_{}_m{rows}", mode.name());
        let (scratch, runtime) = validate_topk_manifest(programs.spec(&name)?, rows, pages, mode, context)?;
        let runtime_sms = runtime.then(|| programs.library().sm_count()).transpose()?
            .map(|n| i32::try_from(n).map_err(|_| KimiMlaError::Invalid("index selector SM count"))).transpose()?;
        Ok(Self { program: programs.program(&name, if context { SCORED_TOPK } else { INDEX_TOPK })?,
            rows, pages, mode, scored: context, runtime_sms, scratch })
    }
    pub fn scratch_bytes(&self) -> usize { self.scratch }

    /// Table entries and causal frontiers are device metadata owned by the lane.
    /// Scratch is zero-initialized once by its admitted owner, before capture.
    pub fn launch(&self, b: KimiIndexSelection, stream: *mut c_void) -> Result<()> {
        validate_topk_batch(b, self.rows, self.pages, self.mode, self.scored, self.scratch)?;
        if self.scored {
            // SAFETY: writer aliases and extents checked; (2, rank) selects one
            // parity half through the compact table, (1, 0) the whole cache.
            unsafe { self.program.launch(&[b.query_fp8.ptr, b.head_weights.ptr, b.cache.ptr, b.table.ptr,
                b.lengths.ptr, b.indices.ptr, b.scores.ptr, b.scratch.ptr],
                &[Scalar::I32(b.rows as i32), Scalar::I32(b.table_width as i32),
                  Scalar::I32(b.table_stride as i32), Scalar::I32(b.page_stride as i32),
                  Scalar::I32(b.page_offset as i32)], stream)?; }
        } else {
            let scalars = [Scalar::I32(b.rows as i32), Scalar::I32(b.table_width as i32),
                Scalar::I32(b.table_stride as i32), Scalar::I32(self.runtime_sms.unwrap_or(0))];
            // SAFETY: physical index/MLA tables agree for ordinary placement;
            // outputs/scratch are disjoint and retained through the queued work.
            unsafe { self.program.launch(&[b.query_fp8.ptr, b.head_weights.ptr, b.cache.ptr, b.table.ptr,
                b.lengths.ptr, b.indices.ptr, b.scratch.ptr],
                &scalars[..if self.runtime_sms.is_some() { 4 } else { 3 }], stream)?; }
        }
        Ok(())
    }
}

fn validate_topk_manifest(spec: &ProgramSpec, rows: usize, pages: usize, mode: KimiIndexMode,
    scored: bool) -> Result<(usize, bool)> {
    let g = &spec.geometry;
    let runtime = !scored && g["runtime_sm_count"].as_bool() == Some(true);
    let pointers = if scored { SCORED_TOPK } else { INDEX_TOPK };
    let scalars = if scored { &b"iiiii"[..] } else if runtime { &b"iiii"[..] } else { &b"iii"[..] };
    if !(1..=4096).contains(&rows) || !(1..=16384).contains(&pages)
        || spec.capacity_rows as usize != rows || spec.scalar_kinds != scalars
        || spec.pointers.iter().map(String::as_str).ne(pointers.iter().copied())
        || g["topk"].as_u64() != Some(2048) || g["heads"].as_u64() != Some(32)
        || g["max_rows"].as_u64() != Some(rows as u64)
        || g["max_pages"].as_u64() != Some(pages as u64) || g["mode"].as_str() != Some(mode.name()) {
        return Err(KimiMlaError::Invalid("K3.1 selector manifest geometry/ABI"));
    }
    let scratch = spec.scratch.get("scratch").and_then(|n| usize::try_from(*n).ok())
        .filter(|n| *n > 0).ok_or(KimiMlaError::Invalid("K3.1 selector scratch"))?;
    if spec.scratch.len() != 1 { return Err(KimiMlaError::Invalid("K3.1 selector scratch inventory")); }
    Ok((scratch, runtime))
}

fn validate_topk_batch(b: KimiIndexSelection, rows: usize, pages: usize, mode: KimiIndexMode,
    scored: bool, scratch: usize) -> Result<()> {
    if b.rows == 0 || b.rows > rows || b.cache_pages == 0 || b.cache_pages > i32::MAX as usize / 64
        || b.table_width == 0 || b.table_width > pages || b.table_stride < b.table_width
        || b.table_stride > i32::MAX as usize
        || !matches!(b.page_stride, 1 | 2) || b.page_offset >= b.page_stride
        || (b.page_stride == 1 && b.page_offset != 0) {
        return Err(KimiMlaError::Invalid("K3.1 selector batch/table geometry"));
    }
    let table = if mode == KimiIndexMode::Prefill { b.table_width } else {
        (b.rows - 1).checked_mul(b.table_stride).and_then(|n| n.checked_add(b.table_width))
            .ok_or(KimiMlaError::Invalid("K3.1 selector table overflow"))?
    };
    let operands = [(b.query_fp8, b.rows * 4096), (b.head_weights, b.rows * 32 * 4),
        (b.cache, b.cache_pages * 8448), (b.table, table * 4), (b.lengths, b.rows * 4),
        (b.indices, b.rows * 2048 * 4), (b.scratch, scratch), (b.scores, b.rows * 2048 * 4)];
    validate_phase(&operands[..if scored { 8 } else { 7 }], 5..if scored { 8 } else { 7 }, b.query_fp8.device_id)
}

const CANDIDATE_MERGE: &[&str] = &["scores", "indices", "out_scores", "out_indices", "table", "local_slots", "local_lengths"];

#[derive(Clone, Copy, Debug)]
pub struct KimiCandidateBatch {
    pub rows: usize, pub table_stride: usize, pub rank: usize,
    /// Two score-ordered runs per row, each 2048 entries. K3.1 supplies global
    /// selection as the first run and (-inf,-1) as the second, not local top-k.
    pub scores: Buffer, pub indices: Buffer,
    pub output_scores: Buffer, pub output_indices: Buffer,
    /// Independent compact MLA table, not the replicated index-cache table.
    pub table: Buffer, pub local_slots: Buffer, pub local_lengths: Buffer,
}

pub struct KimiCandidateMerge<'a> { program: Program<'a> }
impl<'a> KimiCandidateMerge<'a> {
    pub fn load(programs: &'a Programs<'a>) -> Result<Self> {
        let name = "kimik31_dsa_candidate_merge";
        let spec = programs.spec(name)?;
        if spec.scalar_kinds != b"iii" || !spec.scratch.is_empty()
            || spec.pointers.iter().map(String::as_str).ne(CANDIDATE_MERGE.iter().copied())
            || spec.geometry["topk"].as_u64() != Some(2048)
            || spec.geometry["unit_rows"].as_u64() != Some(64) {
            return Err(KimiMlaError::Invalid("K3.1 candidate merge ABI"));
        }
        Ok(Self { program: programs.program(name, CANDIDATE_MERGE)? })
    }
    pub fn launch(&self, b: KimiCandidateBatch, stream: *mut c_void) -> Result<()> {
        validate_candidate_batch(b)?;
        // SAFETY: full global candidates are already selected, local mapping
        // extents/aliases checked; the lane retains both runs and compact table.
        unsafe { self.program.launch(&[b.scores.ptr, b.indices.ptr, b.output_scores.ptr,
            b.output_indices.ptr, b.table.ptr, b.local_slots.ptr, b.local_lengths.ptr],
            &[Scalar::I32(b.rows as i32), Scalar::I32(b.table_stride as i32), Scalar::I32(b.rank as i32)], stream)?; }
        Ok(())
    }
}
fn validate_candidate_batch(b: KimiCandidateBatch) -> Result<()> {
    if !(1..=4096).contains(&b.rows) || !(1..=8192).contains(&b.table_stride) || b.rank > 1 {
        return Err(KimiMlaError::Invalid("K3.1 candidate merge geometry"));
    }
    let operands = [(b.scores,b.rows*4096*4), (b.indices,b.rows*4096*4),
        (b.table,b.rows*b.table_stride*4), (b.output_scores,b.rows*2048*4),
        (b.output_indices,b.rows*2048*4), (b.local_slots,b.rows*2048*4), (b.local_lengths,b.rows*4)];
    validate_phase(&operands, 3..7, b.scores.device_id)
}

const SPARSE_ATTENTION: &[&str] = ATTENTION;

#[derive(Clone, Copy, Debug)]
pub struct KimiSparseBatch {
    pub rows: usize, pub cache_tokens: usize,
    pub query: Buffer, pub cache: Buffer, pub indices: Buffer, pub lengths: Buffer,
    pub query_offsets: Buffer, pub output: Buffer, pub lse: Buffer, pub scratch: Buffer,
}

/// Selected physical-token gathers over native Kimi records. Ordinary attention
/// emits natural-log LSE; Context partials emit base2 for the shared peer combine.
pub struct KimiSparseMla<'a> {
    program: Program<'a>, heads: usize, rows: usize, partial: bool,
    scratch: usize, format: MlaCacheFormat,
}
impl<'a> KimiSparseMla<'a> {
    pub fn load(programs: &'a Programs<'a>, heads: usize, rows: usize, format: MlaCacheFormat,
        context_head_begin: Option<usize>) -> Result<Self> {
        let suffix = format.program_suffix();
        let name = if let Some(begin) = context_head_begin {
            format!("kimik31_sparse_mla_partial{suffix}_h{begin}_m{rows}")
        } else {
            let h = if heads == 48 { "" } else { "_h96" };
            format!("kimik31_sparse_mla{h}{suffix}_m{rows}")
        };
        let scratch = validate_sparse_manifest(programs.spec(&name)?, heads, rows, format, context_head_begin)?;
        Ok(Self { program: programs.program(&name, SPARSE_ATTENTION)?, heads, rows,
            partial: context_head_begin.is_some(), scratch, format })
    }
    pub fn scratch_bytes(&self) -> usize { self.scratch }

    /// The native sparse kernel derives partial/LSE offsets from live rows,
    /// unlike the index selector whose prepared offsets are capacity-baked.
    pub fn scratch_bytes_for(rows: usize, heads: usize, partial: bool) -> Result<usize> {
        if !(1..=4096).contains(&rows) || !matches!(heads, 48 | 96) || (partial && heads != 96) {
            return Err(KimiMlaError::Invalid("selected MLA scratch rows/heads"));
        }
        let n = rows * if partial { 48 } else { heads } * 16;
        Ok((n * 512 * 2).next_multiple_of(1024)
            + if partial { n * 4 } else { (n * 4).next_multiple_of(1024) })
    }

    /// Indices are preselected, causal, valid local slots; each query is one
    /// request. Empty local lengths are permitted only for a Context partial.
    pub fn launch(&self, b: KimiSparseBatch, stream: *mut c_void) -> Result<()> {
        let scratch = Self::scratch_bytes_for(b.rows, self.heads, self.partial)?;
        validate_sparse_batch(b, self.heads, self.rows, self.partial, scratch, self.format)?;
        let partial_scalars = [Scalar::I32(b.rows as i32), Scalar::I64(2048), Scalar::I32(16)];
        let full_scalars = [Scalar::I32(b.rows as i32), Scalar::I32(b.rows as i32), Scalar::I64(2048), Scalar::I32(16)];
        let scalars = if self.partial { &partial_scalars[..] } else { &full_scalars[..] };
        // SAFETY: exact token addressing, extents, device and writer aliases
        // checked; caller owns causal table contents and queued graph lifetimes.
        unsafe { self.program.launch(&[b.query.ptr, b.cache.ptr, b.indices.ptr, b.lengths.ptr,
            b.query_offsets.ptr, b.output.ptr, b.lse.ptr, b.scratch.ptr], scalars, stream)?; }
        Ok(())
    }
}

fn validate_sparse_batch(b: KimiSparseBatch, heads: usize, rows: usize, partial: bool,
    scratch: usize, format: MlaCacheFormat) -> Result<()> {
    if b.rows == 0 || b.rows > rows || b.cache_tokens == 0 || b.cache_tokens > i32::MAX as usize {
        return Err(KimiMlaError::Invalid("selected MLA batch/cache geometry"));
    }
    let cache = b.cache_tokens.checked_mul(format.record_bytes())
        .ok_or(KimiMlaError::Invalid("selected MLA cache overflow"))?;
    let output_heads = if partial { 48 } else { heads };
    validate_phase(&[(b.query, b.rows * heads * 576 * 2), (b.cache, cache),
        (b.indices, b.rows * 2048 * 4), (b.lengths, b.rows * 4),
        (b.query_offsets, (b.rows + 1) * 4), (b.output, b.rows * output_heads * 512 * 2),
        (b.lse, b.rows * output_heads * 4), (b.scratch, scratch)], 5..8, b.query.device_id)
}

fn validate_sparse_manifest(spec: &ProgramSpec, heads: usize, rows: usize, format: MlaCacheFormat,
    begin: Option<usize>) -> Result<usize> {
    let g = &spec.geometry;
    let partial = begin.is_some();
    if !matches!(heads, 48 | 96) || !(1..=4096).contains(&rows)
        || begin.is_some_and(|b| heads != 96 || !matches!(b, 0 | 48))
        || spec.capacity_rows as usize != rows
        || spec.pointers.iter().map(String::as_str).ne(SPARSE_ATTENTION.iter().copied())
        || spec.scalar_kinds.as_slice() != if partial { &b"ili"[..] } else { &b"iili"[..] }
        || [("heads", heads), ("hidden", 7168), ("q_lora_rank", 1536), ("kv_lora_rank", 512),
            ("qk_nope_dim", 128), ("qk_rope_dim", 64), ("v_head_dim", 128), ("page_rows", 64),
            ("page_size", 1), ("max_rows", rows), ("topk", 2048), ("num_splits", 16)]
            .iter().any(|&(key, n)| g[key].as_u64() != Some(n as u64))
        || g["norm_eps"].as_f64() != Some(1e-5) || g["kv_cache"].as_str() != Some(format.label())
        || (if partial { g["lse_base"].as_u64() != Some(2) } else { g["lse_base"].as_str() != Some("e") })
        || begin.is_some_and(|b| g["head_begin"].as_u64() != Some(b as u64) || g["head_count"].as_u64() != Some(48)) {
        return Err(KimiMlaError::Invalid("selected MLA manifest geometry/LSE units/ABI"));
    }
    let align = |n: usize| n.div_ceil(1024) * 1024;
    let n = rows * if partial { 48 } else { heads } * 16;
    let scratch = align(n * 512 * 2) + if partial { n * 4 } else { align(n * 4) };
    if spec.scratch.len() != 1 || spec.scratch.get("scratch").copied() != Some(scratch as u64) {
        return Err(KimiMlaError::Invalid("selected MLA manifest scratch"));
    }
    Ok(scratch)
}

fn validate_phase(operands: &[(Buffer, usize)], writers: std::ops::Range<usize>, device: i32) -> Result<()> {
    validate_buffers(operands, device)?;
    for &(buffer, bytes) in operands {
        if buffer.flags != 0 || (buffer.ptr as usize).checked_add(bytes).is_none() {
            return Err(KimiMlaError::Invalid("phase flags/address overflow"));
        }
    }
    for writer in writers { for other in 0..operands.len() {
        if writer != other { reject_overlap(operands[writer], operands[other])
            .map_err(|_| KimiMlaError::Invalid("phase operands overlap"))?; }
    }}
    Ok(())
}

fn validate_producer(w: KimiMlaWeights, b: KimiMlaBatch, heads: usize, rows: usize,
    scratch: usize, record_bytes: usize) -> Result<()> {
    if b.rows == 0 || b.rows > rows || b.cache_pages == 0 || b.cache_pages > i32::MAX as usize {
        return Err(KimiMlaError::Invalid("producer batch/cache geometry"));
    }
    let cache = b.cache_pages.checked_mul(MLA_PAGE_ROWS * record_bytes)
        .ok_or(KimiMlaError::Invalid("producer cache overflow"))?;
    let r = b.rows;
    validate_phase(&[(b.x, r * 7168 * 2), (b.slots, r * 8),
        (w.qkv_a, 2112 * 7168 * 2), (w.q_a_norm, 1536 * 2), (w.kv_a_norm, 512 * 2),
        (w.q_b, heads * 192 * 1536 * 2), (w.uk, heads * 512 * 128 * 2),
        (b.cache, cache), (b.query, r * heads * 576 * 2), (b.q_resid, r * 1536 * 2),
        (b.producer_scratch, scratch)], 7..11, b.x.device_id)
}

fn validate_output(w: KimiMlaWeights, b: KimiMlaBatch, heads: usize, rows: usize, scratch: usize) -> Result<()> {
    if b.rows == 0 || b.rows > rows { return Err(KimiMlaError::Invalid("output batch geometry")); }
    let r = b.rows;
    validate_phase(&[(b.attention, r * heads * 512 * 2), (b.x, r * 7168 * 2),
        (w.uv, heads * 128 * 512 * 2), (w.gate, heads * 128 * 7168 * 2),
        (w.output, 7168 * heads * 128 * 2), (b.output, r * 7168 * 2),
        (b.output_scratch, scratch)], 5..7, b.x.device_id)
}

fn validate_projected_attention(b:KimiMlaBatch,heads:usize,rows:usize,context:usize,splits:usize,scratch:usize,
    record_bytes:usize)->Result<()> {
    if b.rows==0 || b.rows>rows || b.requests==0 || b.requests>b.rows
        || b.page_table_width==0 || b.page_table_width>context.div_ceil(MLA_PAGE_ROWS)
        || b.cache_pages==0 || b.cache_pages>i32::MAX as usize || b.active_splits!=splits {
        return Err(KimiMlaError::Invalid("projected attention batch/context/split geometry"));
    }
    let cache=b.cache_pages.checked_mul(MLA_PAGE_ROWS*record_bytes)
        .ok_or(KimiMlaError::Invalid("projected attention cache overflow"))?;
    let operands=[(b.query,b.rows*heads*576*2),(b.cache,cache),
        (b.page_table,b.requests*b.page_table_width*4),(b.lengths,b.requests*4),
        (b.query_offsets,(b.requests+1)*4),(b.attention,b.rows*heads*512*2),
        (b.lse,b.rows*heads*4),(b.attention_scratch,scratch)];
    validate_buffers(&operands,b.query.device_id)?;
    for (buffer,bytes) in operands {
        if buffer.flags!=0 || (buffer.ptr as usize).checked_add(bytes).is_none() {
            return Err(KimiMlaError::Invalid("attention flags/address overflow"));
        }
    }
    for writer in 5..8 {for other in 0..8 {
        if writer!=other && operands[writer].1>0 && operands[other].1>0 {
            let (a,an)=operands[writer];let (b,bn)=operands[other];
            let start=a.ptr as usize;let end=start.checked_add(an).ok_or(KimiMlaError::Invalid("attention address overflow"))?;
            let other_start=b.ptr as usize;let other_end=other_start.checked_add(bn).ok_or(KimiMlaError::Invalid("attention address overflow"))?;
            if start<other_end && other_start<end {return Err(KimiMlaError::Invalid("attention operands overlap"));}
        }
    }}
    Ok(())
}

/// A resolved owned-page partial; h96 queries produce compact h48 outputs.
/// The caller supplies one already-causal local length/table per query row.
/// LSE stays base2 through the partial and shared two-owner combine.
pub struct KimiMlaPartial<'a> {
    program: Program<'a>,
    rows: usize,
    context: usize,
    scratch: usize,
    record_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct KimiMlaPartialBatch {
    pub rows: usize,
    /// Fixed device scalar, updated outside capture; contents must be in 2..=16.
    pub active_splits: Buffer,
    pub cache_pages: usize,
    pub page_table_width: usize,
    pub query: Buffer,
    pub cache: Buffer,
    pub page_table: Buffer,
    pub lengths: Buffer,
    /// Identity cumulative query offsets [0, 1, ..., rows].
    pub query_offsets: Buffer,
    pub attention: Buffer,
    pub lse: Buffer,
    pub scratch: Buffer,
}

impl<'a> KimiMlaPartial<'a> {
    pub fn load(programs: &'a Programs<'a>, name: &str, head_begin: usize, rows: usize,
        format: MlaCacheFormat) -> Result<Self> {
        let program = programs.program(name, &["query", "kv_cache", "page_table", "cache_seqlens",
            "cu_seqlens_q", "attn", "lse", "scratch", "active_splits"])?;
        let (context, scratch) = validate_partial_manifest(program.spec(), head_begin, rows, format)?;
        Ok(Self { program, rows, context, scratch, record_bytes: format.record_bytes() })
    }

    pub fn scratch_bytes(&self) -> usize { self.scratch }
    pub fn max_context(&self) -> usize { self.context }

    pub fn launch(&self, b: KimiMlaPartialBatch, stream: *mut c_void) -> Result<()> {
        if b.rows == 0 || b.rows > self.rows || b.page_table_width == 0
            || b.page_table_width > self.context.div_ceil(MLA_PAGE_ROWS)
            || b.cache_pages == 0 || b.cache_pages > i32::MAX as usize {
            return Err(KimiMlaError::Invalid("context partial batch geometry"));
        }
        let cache = b.cache_pages.checked_mul(MLA_PAGE_ROWS * self.record_bytes)
            .ok_or(KimiMlaError::Invalid("context partial cache overflow"))?;
        let operands = [(b.query, b.rows * 96 * 576 * 2), (b.cache, cache),
            (b.page_table, b.rows * b.page_table_width * 4), (b.lengths, b.rows * 4),
            (b.query_offsets, (b.rows + 1) * 4), (b.attention, b.rows * 48 * 512 * 2),
            (b.lse, b.rows * 48 * 4), (b.scratch, self.scratch), (b.active_splits, 4)];
        validate_buffers(&operands, b.query.device_id)?;
        for (buffer, bytes) in operands {
            if buffer.flags != 0 || (buffer.ptr as usize).checked_add(bytes).is_none() {
                return Err(KimiMlaError::Invalid("context partial flags/address overflow"));
            }
        }
        if !(b.active_splits.ptr as usize).is_multiple_of(4) {
            return Err(KimiMlaError::Invalid("context partial split scalar alignment"));
        }
        for writer in 5..8 { for other in 0..9 {
            if writer != other {
                reject_overlap(operands[writer], operands[other])
                    .map_err(|_| KimiMlaError::Invalid("context partial operands overlap"))?;
            }
        }}
        // SAFETY: startup validates the base2 h96->h48 ABI; live extents,
        // devices and write aliases are checked. The caller owns the stream,
        // causal table contents and buffers through queued work/graph replay.
        unsafe { self.program.launch(&[b.query.ptr, b.cache.ptr, b.page_table.ptr, b.lengths.ptr,
            b.query_offsets.ptr, b.attention.ptr, b.lse.ptr, b.scratch.ptr, b.active_splits.ptr],
            &[Scalar::I32(b.rows as i32), Scalar::I64(b.page_table_width as i64)], stream)?; }
        Ok(())
    }
}

fn validate_partial_manifest(spec: &ProgramSpec, begin: usize, rows: usize, format: MlaCacheFormat)
    -> Result<(usize, usize)> {
    let g = &spec.geometry;
    if !matches!(begin, 0 | 48) || !(1..=4096).contains(&rows)
        || !matches!(g["heads_per_tile"].as_u64(), Some(8 | 16))
        || spec.capacity_rows as usize != rows || spec.scalar_kinds != b"il"
        || g["kv_cache"].as_str() != Some(format.label())
        || g["head_begin"].as_u64() != Some(begin as u64)
        || g["norm_eps"].as_f64() != Some(1e-6)
        || [("heads", 96), ("head_count", 48), ("hidden", 7168), ("q_lora_rank", 1536),
            ("kv_lora_rank", 512), ("qk_nope_dim", 128), ("qk_rope_dim", 64),
            ("v_head_dim", 128), ("page_rows", 64), ("lse_base", 2), ("max_rows", rows)]
            .iter().any(|&(key, n)| g[key].as_u64() != Some(n as u64)) {
        return Err(KimiMlaError::Invalid("context partial manifest geometry/scalars"));
    }
    let context = g["max_context"].as_u64().filter(|&n| (1..=1048576).contains(&n))
        .ok_or(KimiMlaError::Invalid("context partial manifest context"))? as usize;
    let splits = g["num_splits"].as_u64().filter(|&n| n == 16)
        .ok_or(KimiMlaError::Invalid("context partial manifest splits"))? as usize;
    let n = rows * 48 * splits;
    let scratch = (n * 512 * 2).div_ceil(1024) * 1024 + n * 4;
    if g["runtime_splits"].as_bool() != Some(true)
        || g["device_splits"].as_bool() != Some(true) || g["fixed_split_grid"] != 16
        || spec.scratch.get("scratch").copied() != Some(scratch as u64) {
        return Err(KimiMlaError::Invalid("context partial manifest traversal/scratch"));
    }
    Ok((context, scratch))
}

fn validate_manifest(specs: [&ProgramSpec; 3], heads: usize, rows: usize, mode: KimiMlaMode, cache_format: MlaCacheFormat)
    -> Result<(usize, usize, [usize; 3])> {
    for (i, spec) in specs.iter().enumerate() {
        let geometry = &spec.geometry;
        if (i != 2 && geometry["kv_cache"].as_str() != Some(cache_format.label()))
            || spec.capacity_rows as usize != rows || geometry["heads"].as_u64() != Some(heads as u64)
            || [("hidden", 7168), ("q_lora_rank", 1536), ("kv_lora_rank", 512),
                ("qk_nope_dim", 128), ("qk_rope_dim", 64), ("v_head_dim", 128), ("page_rows", 64)]
                .iter().any(|&(key, n)| geometry[key].as_u64() != Some(n))
            || geometry["norm_eps"].as_f64() != Some(1e-6)
            || spec.scalar_kinds.as_slice() != if i == 1 { &b"iili"[..] } else { &b"i"[..] } {
            return Err(KimiMlaError::Invalid("manifest geometry/capacity/scalars"));
        }
    }
    let attention = &specs[1].geometry;
    let context = attention["max_context"].as_u64().filter(|&n| n > 0 && n <= 1048576)
        .ok_or(KimiMlaError::Invalid("manifest context"))? as usize;
    let splits = if mode == KimiMlaMode::Extend { 1 } else { context.div_ceil(64).min(16) };
    let chunks = context.div_ceil(64).div_ceil(splits);
    if attention["mode"].as_str() != Some(mode.name()) || attention["num_splits"].as_u64() != Some(splits as u64)
        || attention["chunks_per_split"].as_u64() != Some(chunks as u64) {
        return Err(KimiMlaError::Invalid("manifest traversal"));
    }
    let align = |n: usize| n.div_ceil(1024) * 1024;
    let expected = [align(rows * 2112 * 2) + align(rows * heads * 192 * 2),
        if splits > 1 { align(rows * heads * splits * 512 * 2) + align(rows * heads * splits * 4) } else { 0 },
        2 * align(rows * heads * 128 * 2)];
    for (spec, bytes) in specs.iter().zip(expected) {
        if spec.scratch.get("scratch").copied() != Some(bytes as u64) {
            return Err(KimiMlaError::Invalid("manifest scratch"));
        }
    }
    Ok((context, splits, expected))
}

fn validate_buffers(requirements: &[(Buffer, usize)], device: i32) -> Result<()> {
    if device < 0 { return Err(KimiMlaError::Invalid("device")); }
    for (buffer, bytes) in requirements {
        if buffer.ptr.is_null() || buffer.bytes < *bytes || buffer.device_id != device
            || buffer.ptr as usize % 16 != 0 {
            return Err(KimiMlaError::Invalid("buffer extent/alignment/device"));
        }
    }
    Ok(())
}

/// The prefix update is stored in BF16 before candidate scores are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KimiPrefixMode { Keep, Add, Replace }
impl KimiPrefixMode {
    fn name(self) -> &'static str {
        match self { Self::Keep => "keep", Self::Add => "add", Self::Replace => "replace" }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KimiResidualSite {
    pub candidates: usize,
    pub prefix: KimiPrefixMode,
    pub append: bool,
}
impl KimiResidualSite {
    fn valid(self) -> bool {
        if self.candidates == 1 {
            return self.prefix == KimiPrefixMode::Keep && self.append;
        }
        (2..=9).contains(&self.candidates)
            && (!self.append || (self.candidates <= 8 && self.prefix == KimiPrefixMode::Add))
    }
    fn name(self) -> String {
        format!("kimik3_attn_res_c{}_{}{}", self.candidates, self.prefix.name(),
            if self.append { "_append" } else { "" })
    }
}

#[derive(Debug, Error)]
pub enum KimiW8Error {
    #[error("Kimi residual/Situ invalid {0}")]
    Invalid(&'static str),
    #[error("Kimi residual/Situ program: {0}")]
    Program(#[from] anyhow::Error),
}
type W8Result<T> = std::result::Result<T, KimiW8Error>;

/// Arena is [9, arena_rows, 7168] BF16, including immutable blocks 0..7
/// and the mutable prefix at slot 8. Padding rows retain their contents.
#[derive(Clone, Copy, Debug)]
pub struct KimiResidualBatch {
    pub rows: usize,
    pub arena_rows: usize,
    pub arena: Buffer,
    pub delta: Buffer,
    /// F32 checkpoint norm.float() * projection.float(), prepared once.
    pub score_weight: Buffer,
    pub norm_weight: Buffer,
    pub output: Buffer,
}

/// Resolved before capture; no allocation, lazy loading or synchronization
/// occurs on launch. The caller retains buffers through queued work/replays.
pub struct KimiResidual<'a> {
    program: Program<'a>,
    rows: usize,
}
impl<'a> KimiResidual<'a> {
    pub fn load(programs: &'a Programs<'a>, site: KimiResidualSite) -> W8Result<Self> {
        if !site.valid() { return Err(KimiW8Error::Invalid("residual site")); }
        let program = programs.program(&site.name(),
            &["arena", "delta", "score_weight", "norm_weight", "out"])?;
        validate_residual_manifest(program.spec(), site)?;
        let rows = program.spec().capacity_rows as usize;
        Ok(Self { program, rows })
    }

    pub fn arena_bytes(arena_rows: usize) -> W8Result<usize> {
        if arena_rows == 0 || arena_rows > i32::MAX as usize {
            return Err(KimiW8Error::Invalid("arena rows"));
        }
        arena_rows.checked_mul(9 * 7168 * 2).ok_or(KimiW8Error::Invalid("arena extent overflow"))
    }

    pub fn launch(&self, b: KimiResidualBatch, stream: *mut c_void) -> W8Result<()> {
        if b.rows == 0 || b.rows > self.rows || b.rows > b.arena_rows {
            return Err(KimiW8Error::Invalid("residual rows/capacity"));
        }
        let arena_bytes = Self::arena_bytes(b.arena_rows)?;
        let row_bytes = b.rows.checked_mul(7168 * 2).ok_or(KimiW8Error::Invalid("row extent overflow"))?;
        let buffers = [(b.arena, arena_bytes), (b.delta, row_bytes),
            (b.score_weight, 7168 * 4), (b.norm_weight, 7168 * 2), (b.output, row_bytes)];
        validate_w8_buffers(&buffers, b.arena.device_id)?;
        // Every mutable range is disjoint from all other operands, including
        // reads. Arena slots themselves are intentionally one inout operand.
        for writer in [0, 4] {
            for other in 0..buffers.len() {
                if writer != other { reject_overlap(buffers[writer], buffers[other])?; }
            }
        }
        // SAFETY: checked geometry, integer bounds, extents, alignment, device
        // and write aliases; the engine owns buffer/stream/graph lifetimes.
        unsafe {
            self.program.launch(&[b.arena.ptr, b.delta.ptr, b.score_weight.ptr,
                b.norm_weight.ptr, b.output.ptr], &[Scalar::I32(b.rows as i32),
                Scalar::I32(b.arena_rows as i32), Scalar::F32(1e-5)], stream)?;
        }
        Ok(())
    }
}

pub struct KimiSitu<'a> {
    program: Program<'a>,
    width: usize,
    rows: usize,
}
impl<'a> KimiSitu<'a> {
    pub fn load(programs: &'a Programs<'a>, intermediate: usize) -> W8Result<Self> {
        if !matches!(intermediate, 3072 | 6144 | 16896 | 33792) {
            return Err(KimiW8Error::Invalid("Situ width"));
        }
        let program = programs.program(&format!("kimik3_situ_i{intermediate}"), &["gate_up", "out"])?;
        validate_w8_manifest(program.spec(), b"i")?;
        if program.spec().geometry["intermediate"].as_u64() != Some(intermediate as u64) {
            return Err(KimiW8Error::Invalid("Situ manifest width"));
        }
        let rows = program.spec().capacity_rows as usize;
        Ok(Self { program, width: intermediate, rows })
    }

    /// Gate-first input [rows,2*I], output [rows,I], both dense BF16.
    pub fn launch(&self, rows: usize, gate_up: Buffer, output: Buffer, stream: *mut c_void) -> W8Result<()> {
        if rows == 0 || rows > self.rows { return Err(KimiW8Error::Invalid("Situ rows/capacity")); }
        let bytes = rows.checked_mul(self.width * 2).ok_or(KimiW8Error::Invalid("Situ extent overflow"))?;
        let input_bytes = bytes.checked_mul(2).ok_or(KimiW8Error::Invalid("Situ extent overflow"))?;
        validate_w8_buffers(&[(gate_up, input_bytes), (output, bytes)], gate_up.device_id)?;
        reject_overlap((gate_up, input_bytes), (output, bytes))?;
        // SAFETY: checked rows, extents, alignment, device and aliases; the
        // engine retains buffers and the stream through completion/replays.
        unsafe { self.program.launch(&[gate_up.ptr, output.ptr], &[Scalar::I32(rows as i32)], stream)?; }
        Ok(())
    }
}

/// Gate/up accumulation policy. Compensation is an explicit comparison opt-in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KimiFfnMode { #[default] Direct, Compensated }
impl KimiFfnMode {
    pub fn name(self) -> &'static str {
        match self { Self::Direct => "direct", Self::Compensated => "compensated" }
    }

    pub fn gate_up_stem(self, intermediate: usize) -> String {
        let suffix = if self == Self::Compensated { "_comp" } else { "" };
        format!("kimik3_ffn_gate_up_i{intermediate}{suffix}")
    }

    fn validate(self, spec: &ProgramSpec) -> W8Result<()> {
        let compensated = spec.geometry.get("compensated");
        let matches = match self {
            Self::Direct => compensated.is_none() || compensated.and_then(|v| v.as_bool()) == Some(false),
            Self::Compensated => compensated.and_then(|v| v.as_bool()) == Some(true),
        };
        if !matches { return Err(KimiW8Error::Invalid("FFN gate/up compensation manifest")); }
        Ok(())
    }
}

/// Dense first-layer or shared-expert MLP. The TP2 caller packs intermediate
/// slices gate-first and owns the partial-output meet after this chain.
pub struct KimiFfn<'a> {
    gate_up: Program<'a>,
    situ: KimiSitu<'a>,
    down: Program<'a>,
    rows: usize,
    width: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct KimiFfnBatch {
    pub rows: usize,
    pub input: Buffer,
    pub gate_up_weight: Buffer,
    pub down_weight: Buffer,
    pub gate_up: Buffer,
    pub activated: Buffer,
    pub output: Buffer,
}
impl<'a> KimiFfn<'a> {
    pub fn load(programs: &'a Programs<'a>, intermediate: usize, mode: KimiFfnMode) -> W8Result<Self> {
        let situ = KimiSitu::load(programs, intermediate)?;
        let gate_up = programs.program(&mode.gate_up_stem(intermediate), &["x", "w", "out"])?;
        let down = programs.program(&format!("kimik3_ffn_down_i{intermediate}"), &["x", "w", "out"])?;
        validate_projection_manifest(gate_up.spec(), 7168, 2 * intermediate)?;
        mode.validate(gate_up.spec())?;
        validate_projection_manifest(down.spec(), intermediate, 7168)?;
        KimiFfnMode::Direct.validate(down.spec())?;
        let rows = situ.rows.min(gate_up.spec().capacity_rows as usize).min(down.spec().capacity_rows as usize);
        Ok(Self { gate_up, situ, down, rows, width: intermediate })
    }

    /// Standalone BF16 projection with this FFN's gate/up geometry. It also
    /// provides a same-weight reduction-order control for projected seams.
    pub fn project_gate_up(&self,rows:usize,input:Buffer,weight:Buffer,output:Buffer,
        stream:*mut c_void)->W8Result<()> {
        validate_projection_buffers(rows,self.rows,7168,2*self.width,input,weight,output)?;
        // SAFETY: resolved program, checked geometry/devices/disjoint output;
        // the caller retains every operand until the stream drains.
        unsafe {self.gate_up.launch(&[input.ptr,weight.ptr,output.ptr],&[Scalar::I32(rows as i32)],stream)?;}
        Ok(())
    }

    /// Standalone down projection, retaining the existing BF16 reduction.
    pub fn project_down(&self,rows:usize,input:Buffer,weight:Buffer,output:Buffer,
        stream:*mut c_void)->W8Result<()> {
        validate_projection_buffers(rows,self.rows,self.width,7168,input,weight,output)?;
        // SAFETY: resolved program with validated dense disjoint operands;
        // the caller retains all storage through completion.
        unsafe {self.down.launch(&[input.ptr,weight.ptr,output.ptr],&[Scalar::I32(rows as i32)],stream)?;}
        Ok(())
    }

    pub fn launch(&self, b: KimiFfnBatch, stream: *mut c_void) -> W8Result<()> {
        if b.rows == 0 || b.rows > self.rows { return Err(KimiW8Error::Invalid("FFN rows/capacity")); }
        let row_bytes = b.rows.checked_mul(7168 * 2).ok_or(KimiW8Error::Invalid("FFN extent overflow"))?;
        let intermediate_bytes = b.rows.checked_mul(self.width * 2).ok_or(KimiW8Error::Invalid("FFN extent overflow"))?;
        let buffers = [(b.input, row_bytes), (b.gate_up_weight, 2 * self.width * 7168 * 2),
            (b.down_weight, 7168 * self.width * 2), (b.gate_up, 2 * intermediate_bytes),
            (b.activated, intermediate_bytes), (b.output, row_bytes)];
        validate_w8_buffers(&buffers, b.input.device_id)?;
        for writer in 3..6 {
            for other in 0..buffers.len() {
                if writer != other { reject_overlap(buffers[writer], buffers[other])?; }
            }
        }
        // SAFETY: all three programs are resolved, operands are validated and
        // disjoint, and the caller retains storage/stream through graph replay.
        unsafe {
            self.gate_up.launch(&[b.input.ptr, b.gate_up_weight.ptr, b.gate_up.ptr],
                &[Scalar::I32(b.rows as i32)], stream)?;
        }
        self.situ.launch(b.rows, b.gate_up, b.activated, stream)?;
        // SAFETY: validated down-projection extents/device/aliases above; the
        // preceding stages on the same stream establish the input dependency.
        unsafe {
            self.down.launch(&[b.activated.ptr, b.down_weight.ptr, b.output.ptr],
                &[Scalar::I32(b.rows as i32)], stream)?;
        }
        Ok(())
    }
}

/// Native BF16 KDA operands, packed by the shared KDA checkpoint reader.
#[derive(Clone, Copy, Debug)]
pub struct KimiKdaWeights {
    pub input: Buffer, pub forget: Buffer, pub convolution: Buffer,
    pub a_log: Buffer, pub dt_bias: Buffer, pub norm: Buffer, pub output: Buffer,
}

/// Each slot occurs in one contiguous row stretch; `first` names that stretch's
/// first row. Wide prefill is one sequence. Slot values and metadata contents
/// must address live storage; the caller retains all buffers through replay.
#[derive(Clone, Copy, Debug)]
pub struct KimiKdaBatch {
    pub rows: usize, pub sequences: usize, pub state_slots: usize, pub speculative: bool,
    pub input: Buffer, pub state: Buffer, pub conv_state: Buffer,
    pub slots: Buffer, pub first: Buffer, pub normalized: Buffer, pub output: Buffer,
    pub replay: Option<Buffer>, pub scratch: Buffer,
}

/// Physical KDA state/replay ownership, never two logical shards on one device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KimiKdaGeometry { heads: usize }
impl KimiKdaGeometry {
    pub fn new(heads: usize) -> W8Result<Self> {
        if !matches!(heads, 48 | 96) { return Err(KimiW8Error::Invalid("KDA local heads")); }
        Ok(Self { heads })
    }
    pub fn heads(self) -> usize { self.heads }
    pub fn width(self) -> usize { self.heads * 128 }
    pub fn aux_width(self) -> usize { 128 + self.heads }
    pub fn input_width(self) -> usize { 4 * self.width() + self.aux_width() }
    pub fn state_bytes(self, slots: usize) -> W8Result<usize> {
        kda_slot_bytes(slots, self.heads * 128 * 128 * 4)
    }
    pub fn conv_state_bytes(self, slots: usize) -> W8Result<usize> {
        kda_slot_bytes(slots, 3 * 3 * self.width() * 2)
    }
    pub fn replay_bytes(self) -> usize { 64 * self.heads * (3 * 128 * 4 + 4 + 3 * 128 * 2) }
}

/// Shared KDA recurrence -> head-local gated norm -> BF16 o_proj partial.
/// The TP2 meet belongs to the owning engine, never inside this binding.
pub struct KimiKda<'a> {
    program: Program<'a>, output: Program<'a>, rows: usize, decode: bool, scratch: usize,
    geometry: KimiKdaGeometry,
}
impl<'a> KimiKda<'a> {
    pub fn load(programs: &'a Programs<'a>, rows: usize, decode: bool) -> W8Result<Self> {
        Self::load_heads(programs, rows, decode, 48)
    }

    pub fn load_heads(programs: &'a Programs<'a>, rows: usize, decode: bool, heads: usize) -> W8Result<Self> {
        Self::load_variant(programs, rows, decode, heads, false)
    }

    pub fn load_k31(programs: &'a Programs<'a>, rows: usize, decode: bool, heads: usize) -> W8Result<Self> {
        Self::load_variant(programs, rows, decode, heads, true)
    }

    fn load_variant(programs: &'a Programs<'a>, rows: usize, decode: bool, heads: usize, k31: bool) -> W8Result<Self> {
        let geometry = KimiKdaGeometry::new(heads)?;
        let pointers = ["x", "w_in", "w_fg", "conv_w", "a_log", "dt_bias", "o_norm", "w_o",
            "conv_state", "state", "slots", "seq_first", "out", "replay", "scratch"];
        let prefill = [&pointers[..13], &pointers[14..]].concat();
        let name = if heads == 48 { format!("{}_kda_m{rows}", if k31 { "kimik31" } else { "kimik3" }) }
            else { format!("kda_fused_h96_m{rows}") };
        let program = programs.program(&name, if decode { &pointers } else { &prefill })?;
        let scratch = validate_kda_manifest_heads(program.spec(), rows, decode, geometry)?;
        let output_name = if heads == 48 { "kimik3_ffn_down_i6144" } else { "kda_output_h96" };
        let output = programs.program(output_name, &["x", "w", "out"])?;
        validate_projection_manifest(output.spec(), geometry.width(), 7168)?;
        if output.spec().capacity_rows < rows as u32 {
            return Err(KimiW8Error::Invalid("KDA output capacity"));
        }
        Ok(Self { program, output, rows, decode, scratch, geometry })
    }

    pub fn scratch_bytes(&self) -> usize { self.scratch }
    pub fn state_bytes(slots: usize) -> W8Result<usize> { kda_slot_bytes(slots, 48 * 128 * 128 * 4) }
    pub fn conv_state_bytes(slots: usize) -> W8Result<usize> { kda_slot_bytes(slots, 3 * 3 * 6144 * 2) }
    pub const fn replay_bytes() -> usize { 64 * 48 * (3 * 128 * 4 + 4 + 3 * 128 * 2) }

    /// No allocation, synchronization or lazy loading on the launch path.
    pub fn launch(&self, weights: KimiKdaWeights, b: KimiKdaBatch,
        stream: *mut c_void) -> W8Result<()> {
        validate_kda_buffers_heads(weights, b, self.rows, self.decode, self.scratch, self.geometry)?;
        let mut pointers = [b.input.ptr, weights.input.ptr, weights.forget.ptr,
            weights.convolution.ptr, weights.a_log.ptr, weights.dt_bias.ptr, weights.norm.ptr,
            weights.output.ptr, b.conv_state.ptr, b.state.ptr, b.slots.ptr, b.first.ptr,
            b.normalized.ptr, std::ptr::null_mut(), b.scratch.ptr];
        if let Some(replay) = b.replay { pointers[13] = replay.ptr; }
        if !self.decode { pointers[13] = b.scratch.ptr; }
        let scalars = [Scalar::I32(b.rows as i32), Scalar::I32(i32::from(b.speculative))];
        // SAFETY: all operands, writes and capacities are checked before either
        // launch; caller owns slot metadata, ordered state and graph lifetimes.
        unsafe {
            self.program.launch(if self.decode { &pointers } else { &pointers[..14] },
                if self.decode { &scalars } else { &scalars[..1] }, stream)?;
            self.output.launch(&[b.normalized.ptr, weights.output.ptr, b.output.ptr],
                &[Scalar::I32(b.rows as i32)], stream)?;
        }
        Ok(())
    }
}

/// Commit accepted speculative rows into all contiguous KDA layer pools.
/// Tables are `[slot, first_row, accepted_rows]` per sequence; the caller
/// guarantees in-range, non-overlapping sequences and a matching replay step.
pub struct KimiKdaCommit<'a> { program: Program<'a>, geometry: KimiKdaGeometry }
impl<'a> KimiKdaCommit<'a> {
    pub fn load(programs: &'a Programs<'a>) -> W8Result<Self> {
        Self::load_heads(programs, 48)
    }

    pub fn load_heads(programs: &'a Programs<'a>, heads: usize) -> W8Result<Self> {
        Self::load_variant(programs, heads, false)
    }

    pub fn load_k31(programs: &'a Programs<'a>, heads: usize) -> W8Result<Self> {
        Self::load_variant(programs, heads, true)
    }

    fn load_variant(programs: &'a Programs<'a>, heads: usize, k31: bool) -> W8Result<Self> {
        let geometry = KimiKdaGeometry::new(heads)?;
        let name = if heads == 48 {
            if k31 { "kimik31_kda_commit" } else { "kimik3_kda_commit" }
        } else { "kda_commit_h96" };
        let program = programs.program(name, &["state", "conv_state", "replay", "tables"])?;
        validate_kda_commit_manifest(program.spec(), geometry)?;
        Ok(Self { program, geometry })
    }

    pub fn launch(&self, sequences: usize, layers: usize, slots: usize,
        state: Buffer, conv_state: Buffer, replay: Buffer, tables: Buffer,
        stream: *mut c_void) -> W8Result<()> {
        let requirements = kda_commit_requirements(self.geometry, sequences, layers, slots, state, conv_state, replay, tables)?;
        validate_w8_buffers(&requirements, state.device_id)?;
        reject_writers(&requirements, &[0, 1])?;
        // SAFETY: checked pool extents and disjoint writes; the caller retains
        // replay, tables, state and stream through completion/graph replay.
        unsafe { self.program.launch(&[state.ptr, conv_state.ptr, replay.ptr, tables.ptr],
            &[Scalar::I32(sequences as i32), Scalar::I32(layers as i32), Scalar::I32(slots as i32)], stream)?; }
        Ok(())
    }
}

fn kda_slot_bytes(slots: usize, per_slot: usize) -> W8Result<usize> {
    if slots == 0 || slots > i32::MAX as usize { return Err(KimiW8Error::Invalid("KDA state slots")); }
    slots.checked_mul(per_slot).ok_or(KimiW8Error::Invalid("KDA extent overflow"))
}

#[cfg(test)]
fn validate_kda_manifest(spec: &ProgramSpec, rows: usize, decode: bool) -> W8Result<usize> {
    validate_kda_manifest_heads(spec, rows, decode, KimiKdaGeometry::new(48)?)
}

fn validate_kda_manifest_heads(spec: &ProgramSpec, rows: usize, decode: bool, geometry: KimiKdaGeometry) -> W8Result<usize> {
    let g = &spec.geometry;
    if rows != if decode { 64 } else { 4096 } || spec.capacity_rows as usize != rows
        || spec.scalar_kinds != if decode { &b"ii"[..] } else { &b"i"[..] }
        || [("hidden", 7168), ("heads", geometry.heads()), ("head_dim", 128), ("max_rows", rows), ("in_width", geometry.input_width())]
            .iter().any(|&(key, n)| g[key].as_u64() != Some(n as u64))
        || g["eps"].as_f64() != Some(1e-5) || g["lower_bound"].as_f64() != Some(-5.0)
        || g["output_kind"].as_str() != Some("norm") || g["output_gate"].as_str() != Some("full_rank")
        || g["o_norm_dtype"].as_str() != Some("float32") || g["beta_sigmoid_dtype"].as_str() != Some("float32")
        || g["use_gate_in_kernel"].as_bool() != Some(true) || g["safe_gate"].as_bool() != Some(true)
        || !g["state_dtype"].is_null() || !g["state_rounding"].is_null()
        || (decode && g["bf16_decode"].as_bool() != Some(true))
        || (!decode && !g["bf16_decode"].is_null()) || !g["replay_rows"].is_null()
        || (decode && !g["chunked_min_rows"].is_null())
        || (!decode && g["chunked_min_rows"].as_u64() != Some(64)) {
        return Err(KimiW8Error::Invalid("KDA manifest geometry/scalars"));
    }
    let align = |n: usize| n.div_ceil(1024) * 1024;
    let width = geometry.width();
    let base = align(rows * geometry.input_width() * 2) + align(rows * 2 * width * 2)
        + align(rows * 3 * width * 2) + 2 * align(rows * width * 2);
    let scratch = spec.scratch.get("scratch").copied()
        .and_then(|n| usize::try_from(n).ok()).ok_or(KimiW8Error::Invalid("KDA scratch manifest"))?;
    if spec.scratch.len() != 1 || scratch < base || (decode && scratch != base)
        || (!decode && scratch == base) || scratch % 1024 != 0 {
        return Err(KimiW8Error::Invalid("KDA scratch manifest"));
    }
    Ok(scratch)
}

fn validate_kda_commit_manifest(spec: &ProgramSpec, geometry: KimiKdaGeometry) -> W8Result<()> {
    let g = &spec.geometry;
    if spec.scalar_kinds != b"iii" || !spec.scratch.is_empty()
        || [("heads", geometry.heads()), ("channels", 3 * geometry.width()), ("replay_rows", 64), ("replay_bytes", geometry.replay_bytes())]
            .iter().any(|&(key, n)| g[key].as_u64() != Some(n as u64))
        || !g["state_dtype"].is_null() || !g["state_rounding"].is_null() {
        return Err(KimiW8Error::Invalid("KDA commit manifest"));
    }
    Ok(())
}

#[cfg(test)]
fn validate_kda_buffers(w: KimiKdaWeights, b: KimiKdaBatch, capacity: usize,
    decode: bool, scratch: usize) -> W8Result<()> {
    validate_kda_buffers_heads(w, b, capacity, decode, scratch, KimiKdaGeometry::new(48)?)
}

fn validate_kda_buffers_heads(w: KimiKdaWeights, b: KimiKdaBatch, capacity: usize,
    decode: bool, scratch: usize, geometry: KimiKdaGeometry) -> W8Result<()> {
    if b.rows == 0 || b.rows > capacity || b.rows > i32::MAX as usize
        || b.sequences == 0 || b.sequences > b.rows || b.sequences > b.state_slots
        || (!decode && (b.speculative || (b.rows > 64 && b.sequences != 1)))
        || b.replay.is_some() != decode {
        return Err(KimiW8Error::Invalid("KDA batch/replay geometry"));
    }
    let width = geometry.width();
    let buffers = [(b.input, b.rows * 7168 * 2), (w.input, geometry.input_width() * 7168 * 2),
        (w.forget, width * 128 * 2), (w.convolution, 3 * width * 4 * 4), (w.a_log, geometry.heads() * 4),
        (w.dt_bias, width * 4), (w.norm, 128 * 4), (w.output, 7168 * width * 2),
        (b.conv_state, geometry.conv_state_bytes(b.state_slots)?), (b.state, geometry.state_bytes(b.state_slots)?),
        (b.slots, b.rows * 4), (b.first, b.rows * 4), (b.normalized, b.rows * width * 2),
        (b.output, b.rows * 7168 * 2), (b.scratch, scratch),
        (b.replay.unwrap_or(b.scratch), geometry.replay_bytes())];
    let active = if decode { &buffers[..] } else { &buffers[..15] };
    validate_w8_buffers(active, b.input.device_id)?;
    reject_writers(active, if decode { &[8, 9, 12, 13, 14, 15] } else { &[8, 9, 12, 13, 14] })
}

fn reject_writers(buffers: &[(Buffer, usize)], writers: &[usize]) -> W8Result<()> {
    for &writer in writers {
        for other in 0..buffers.len() {
            if writer != other { reject_overlap(buffers[writer], buffers[other])?; }
        }
    }
    Ok(())
}

fn kda_commit_requirements(geometry: KimiKdaGeometry, sequences: usize, layers: usize, slots: usize,
    state: Buffer, conv_state: Buffer, replay: Buffer, tables: Buffer) -> W8Result<[(Buffer, usize); 4]> {
    if sequences == 0 || sequences > slots || sequences > i32::MAX as usize || !(1..=69).contains(&layers) {
        return Err(KimiW8Error::Invalid("KDA commit dimensions"));
    }
    let layer_bytes = |bytes: usize| bytes.checked_mul(layers).ok_or(KimiW8Error::Invalid("KDA extent overflow"));
    Ok([(state, layer_bytes(geometry.state_bytes(slots)?)?),
        (conv_state, layer_bytes(geometry.conv_state_bytes(slots)?)?),
        (replay, layer_bytes(geometry.replay_bytes())?), (tables, sequences * 3 * 4)])
}

/// Native RMSNorm: F32 normalization -> BF16 -> BF16 weight product -> BF16.
/// Unlike a one-round affine norm, this preserves the checkpoint's intermediate.
pub struct KimiRmsNorm<'a> {
    program: Program<'a>,
    hidden: usize,
    rows: usize,
}
impl<'a> KimiRmsNorm<'a> {
    pub fn load(programs: &'a Programs<'a>, hidden: usize) -> W8Result<Self> {
        let program = programs.program(&format!("kimik3_rmsnorm_h{hidden}"), &["x", "weight", "out"])?;
        validate_norm_manifest(program.spec(), hidden)?;
        let rows = program.spec().capacity_rows as usize;
        Ok(Self { program, hidden, rows })
    }

    /// Storage must remain live and unchanged until all enqueued work/replays finish.
    pub fn launch(&self, rows: usize, input: Buffer, weight: Buffer, output: Buffer,
        stream: *mut c_void) -> W8Result<()> {
        validate_norm_buffers(rows, self.rows, self.hidden, input, weight, output)?;
        // SAFETY: resolved program, checked dense extents/device/disjoint output,
        // and the caller owns storage and stream through completion and replay.
        unsafe { self.program.launch(&[input.ptr, weight.ptr, output.ptr],
            &[Scalar::I32(rows as i32), Scalar::F32(1e-5)], stream)?; }
        Ok(())
    }
}

/// Replicated latent down projection followed, after the complete expert meet,
/// by native RMSNorm and the latent up projection. No TP meet belongs inside.
pub struct KimiLatent<'a> {
    down: Program<'a>,
    norm: KimiRmsNorm<'a>,
    up: Program<'a>,
    rows: usize,
}
impl<'a> KimiLatent<'a> {
    pub fn load(programs: &'a Programs<'a>) -> W8Result<Self> {
        let down = programs.program("kimik3_latent_down", &["x", "w", "out"])?;
        let up = programs.program("kimik3_latent_up", &["x", "w", "out"])?;
        validate_projection_manifest(down.spec(), 7168, 3584)?;
        validate_projection_manifest(up.spec(), 3584, 7168)?;
        let norm = KimiRmsNorm::load(programs, 3584)?;
        let rows = norm.rows.min(down.spec().capacity_rows as usize).min(up.spec().capacity_rows as usize);
        Ok(Self { down, norm, up, rows })
    }

    pub fn down(&self, rows: usize, input: Buffer, weight: Buffer, output: Buffer,
        stream: *mut c_void) -> W8Result<()> {
        validate_projection_buffers(rows, self.rows, 7168, 3584, input, weight, output)?;
        // SAFETY: validated projection storage; the caller owns operands/stream
        // through completion, including every captured graph replay.
        unsafe { self.down.launch(&[input.ptr, weight.ptr, output.ptr],
            &[Scalar::I32(rows as i32)], stream)?; }
        Ok(())
    }

    /// `combined` is the complete BF16 weighted sum, never a Spark partial.
    /// All operands are validated before either stage is enqueued.
    pub fn up(&self, rows: usize, combined: Buffer, norm_weight: Buffer,
        normalized: Buffer, up_weight: Buffer, output: Buffer,
        stream: *mut c_void) -> W8Result<()> {
        validate_norm_buffers(rows, self.rows, 3584, combined, norm_weight, normalized)?;
        validate_projection_buffers(rows, self.rows, 3584, 7168, normalized, up_weight, output)?;
        let buffers = [(combined, rows * 3584 * 2), (norm_weight, 3584 * 2),
            (normalized, rows * 3584 * 2), (up_weight, 7168 * 3584 * 2), (output, rows * 7168 * 2)];
        for writer in [2, 4] {
            for other in 0..buffers.len() {
                if writer != other { reject_overlap(buffers[writer], buffers[other])?; }
            }
        }
        self.norm.launch(rows, combined, norm_weight, normalized, stream)?;
        // SAFETY: validated dense projection storage and ordered dependency on
        // the native normalization launch on the same caller-owned stream.
        unsafe { self.up.launch(&[normalized.ptr, up_weight.ptr, output.ptr],
            &[Scalar::I32(rows as i32)], stream)?; }
        Ok(())
    }
}

/// Device-resident U32 ids and F32 weights, both `[rows, 16]`.
/// Local experts consume these directly; Spark staging belongs to the shared exchange.
#[derive(Clone, Copy, Debug)]
pub struct KimiRoutes { pub ids: Buffer, pub weights: Buffer }

/// Native BF16 router projection followed by E896/K16 sigmoid selection.
/// The checkpoint correction bias selects experts, but does not weight their outputs.
pub struct KimiRouter<'a> {
    scores: Program<'a>, library: &'a crate::NativeLibrary, rows: usize,
}
impl<'a> KimiRouter<'a> {
    pub fn load(programs: &'a Programs<'a>) -> W8Result<Self> {
        let scores = programs.program("kimik3_router_scores", &["x", "w", "logits"])?;
        validate_router_manifest(scores.spec())?;
        let rows = scores.spec().capacity_rows as usize;
        Ok(Self { scores, library: programs.library(), rows })
    }

    pub fn launch(&self, rows: usize, input: Buffer, weight: Buffer, bias: Buffer,
        logits: Buffer, routes: KimiRoutes, stream: *mut c_void) -> W8Result<()> {
        validate_router_buffers(rows, self.rows, input, weight, bias, logits, routes)?;
        // SAFETY: dense operands and disjoint device outputs are checked before
        // either launch; the caller retains every buffer through completion/replay.
        unsafe {
            self.scores.launch(&[input.ptr, weight.ptr, logits.ptr], &[Scalar::I32(rows as i32)], stream)?;
            self.library.router_select(logits.ptr, bias.ptr, std::ptr::null(), std::ptr::null(),
                routes.ids.ptr, routes.weights.ptr, rows, 896, 16, 1.0, true, stream)?;
        }
        Ok(())
    }
}

/// Shared FP8-K32 expert wire quantizer, specialized at the latent width 3584.
pub struct KimiExpertInput<'a> { program: Program<'a>, rows: usize }
impl<'a> KimiExpertInput<'a> {
    pub const ROW_BYTES: usize = 3584 + 3584 / 32;

    pub fn load(programs: &'a Programs<'a>) -> W8Result<Self> {
        let program = programs.program("kimik3_expert_input_quant",
            &["source_ptr", "values_ptr", "scale_rows_ptr", "scale_mma_ptr"])?;
        validate_expert_input_manifest(program.spec())?;
        let rows = program.spec().capacity_rows as usize;
        Ok(Self { program, rows })
    }

    /// `grid` comes from the shared device-sized launch policy, cached at startup.
    pub fn launch(&self, rows: usize, grid: usize, input: Buffer, wire: Buffer,
        stream: *mut c_void) -> W8Result<()> {
        validate_expert_input_buffers(rows, self.rows, grid, input, wire)?;
        // SAFETY: wire contains the interleaved values/scales at stride ROW_BYTES;
        // scale_mma is unused in wire mode. Caller owns storage through replay.
        unsafe {
            self.program.launch(&[input.ptr, wire.ptr, wire.ptr.cast::<u8>().add(3584).cast(), wire.ptr],
                &[Scalar::I32(rows as i32), Scalar::I32(grid as i32)], stream)?;
        }
        Ok(())
    }
}

fn validate_router_manifest(spec: &ProgramSpec) -> W8Result<()> {
    validate_w8_manifest(spec, b"i")?;
    if [("hidden", 7168), ("experts", 896), ("top_k", 16)].iter()
        .any(|&(key, n)| spec.geometry[key].as_u64() != Some(n)) {
        return Err(KimiW8Error::Invalid("router manifest geometry"));
    }
    Ok(())
}

fn validate_router_buffers(rows: usize, capacity: usize, input: Buffer, weight: Buffer,
    bias: Buffer, logits: Buffer, routes: KimiRoutes) -> W8Result<()> {
    if rows == 0 || rows > capacity || rows > i32::MAX as usize {
        return Err(KimiW8Error::Invalid("router rows/capacity"));
    }
    let buffers = [(input, rows * 7168 * 2), (weight, 896 * 7168 * 2), (bias, 896 * 4),
        (logits, rows * 896 * 4), (routes.ids, rows * 16 * 4), (routes.weights, rows * 16 * 4)];
    validate_w8_buffers(&buffers, input.device_id)?;
    reject_writers(&buffers, &[3, 4, 5])
}

fn validate_expert_input_manifest(spec: &ProgramSpec) -> W8Result<()> {
    validate_w8_manifest(spec, b"ii")?;
    if [("hidden", 3584), ("row_bytes", KimiExpertInput::ROW_BYTES), ("expected_m", 80)].iter()
        .any(|&(key, n)| spec.geometry[key].as_u64() != Some(n as u64))
        || spec.geometry["amax_floor"].as_f64() != Some(1e-4) {
        return Err(KimiW8Error::Invalid("expert wire manifest geometry"));
    }
    Ok(())
}

fn validate_expert_input_buffers(rows: usize, capacity: usize, grid: usize,
    input: Buffer, wire: Buffer) -> W8Result<()> {
    if rows == 0 || rows > capacity || rows > i32::MAX as usize || grid == 0 || grid > i32::MAX as usize {
        return Err(KimiW8Error::Invalid("expert wire rows/grid/capacity"));
    }
    let buffers = [(input, rows * 3584 * 2), (wire, rows * KimiExpertInput::ROW_BYTES)];
    validate_w8_buffers(&buffers, input.device_id)?;
    reject_writers(&buffers, &[1])
}

fn validate_norm_manifest(spec: &ProgramSpec, hidden: usize) -> W8Result<()> {
    validate_w8_manifest(spec, b"if")?;
    if !matches!(hidden, 3584 | 7168) || spec.geometry["hidden"].as_u64() != Some(hidden as u64) {
        return Err(KimiW8Error::Invalid("native RMSNorm manifest geometry"));
    }
    Ok(())
}

fn validate_norm_buffers(rows: usize, capacity: usize, hidden: usize,
    input: Buffer, weight: Buffer, output: Buffer) -> W8Result<()> {
    if rows == 0 || rows > capacity || rows > i32::MAX as usize {
        return Err(KimiW8Error::Invalid("RMSNorm rows/capacity"));
    }
    let bytes = rows.checked_mul(hidden).and_then(|n| n.checked_mul(2))
        .ok_or(KimiW8Error::Invalid("RMSNorm extent overflow"))?;
    let weight_bytes = hidden.checked_mul(2).ok_or(KimiW8Error::Invalid("RMSNorm extent overflow"))?;
    validate_w8_buffers(&[(input, bytes), (weight, weight_bytes), (output, bytes)], input.device_id)?;
    reject_overlap((output, bytes), (input, bytes))?;
    reject_overlap((output, bytes), (weight, weight_bytes))
}

fn validate_projection_buffers(rows: usize, capacity: usize, width: usize, outputs: usize,
    input: Buffer, weight: Buffer, output: Buffer) -> W8Result<()> {
    if rows == 0 || rows > capacity || rows > i32::MAX as usize {
        return Err(KimiW8Error::Invalid("projection rows/capacity"));
    }
    let bytes = |a: usize, b: usize| a.checked_mul(b).and_then(|n| n.checked_mul(2))
        .ok_or(KimiW8Error::Invalid("projection extent overflow"));
    let (x, w, y) = (bytes(rows, width)?, bytes(width, outputs)?, bytes(rows, outputs)?);
    validate_w8_buffers(&[(input, x), (weight, w), (output, y)], input.device_id)?;
    reject_overlap((output, y), (input, x))?;
    reject_overlap((output, y), (weight, w))
}

fn validate_projection_manifest(spec: &ProgramSpec, width: usize, outputs: usize) -> W8Result<()> {
    validate_w8_manifest(spec, b"i")?;
    let g = &spec.geometry;
    if g["width"].as_u64() != Some(width as u64) || g["outputs"].as_u64() != Some(outputs as u64)
        || g["output_dtype"].as_str() != Some("torch.bfloat16") || g["skinny_rows"].as_u64() != Some(0) {
        return Err(KimiW8Error::Invalid("BF16 projection manifest geometry"));
    }
    Ok(())
}

fn validate_w8_manifest(spec: &ProgramSpec, scalars: &[u8]) -> W8Result<()> {
    if spec.capacity_rows == 0 || spec.capacity_rows > i32::MAX as u32
        || spec.scalar_kinds != scalars || !spec.scratch.is_empty() {
        return Err(KimiW8Error::Invalid("manifest capacity/scalars/scratch"));
    }
    Ok(())
}
fn validate_residual_manifest(spec: &ProgramSpec, site: KimiResidualSite) -> W8Result<()> {
    validate_w8_manifest(spec, b"iif")?;
    let g = &spec.geometry;
    if !site.valid() || g["hidden"].as_u64() != Some(7168)
        || g["candidates"].as_u64() != Some(site.candidates as u64)
        || g["prefix_mode"].as_str() != Some(site.prefix.name())
        || g["append"].as_bool() != Some(site.append)
        || g["has_mix"].as_bool() != Some(site.candidates != 1) {
        return Err(KimiW8Error::Invalid("residual manifest geometry"));
    }
    Ok(())
}
fn validate_w8_buffers(buffers: &[(Buffer, usize)], device: i32) -> W8Result<()> {
    if device < 0 { return Err(KimiW8Error::Invalid("device")); }
    for &(buffer, bytes) in buffers {
        if buffer.ptr.is_null() || buffer.bytes < bytes || buffer.device_id != device
            || buffer.ptr as usize % 16 != 0 || (buffer.ptr as usize).checked_add(bytes).is_none() {
            return Err(KimiW8Error::Invalid("buffer extent/alignment/device"));
        }
    }
    Ok(())
}
fn reject_overlap(a: (Buffer, usize), b: (Buffer, usize)) -> W8Result<()> {
    let start_a = a.0.ptr as usize;
    let start_b = b.0.ptr as usize;
    let end_a = start_a.checked_add(a.1).ok_or(KimiW8Error::Invalid("address overflow"))?;
    let end_b = start_b.checked_add(b.1).ok_or(KimiW8Error::Invalid("address overflow"))?;
    if start_a < end_b && start_b < end_a { return Err(KimiW8Error::Invalid("overlapping operands")); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn specs(heads: usize, rows: usize, mode: KimiMlaMode) -> [ProgramSpec; 3] {
        let context: usize = 1048576;
        let splits = if mode == KimiMlaMode::Extend { 1 } else { 16 };
        let align = |n: usize| n.div_ceil(1024) * 1024;
        let scratch = [align(rows * 2112 * 2) + align(rows * heads * 192 * 2),
            if splits > 1 { align(rows * heads * splits * 512 * 2) + align(rows * heads * splits * 4) } else { 0 },
            2 * align(rows * heads * 128 * 2)];
        std::array::from_fn(|i| ProgramSpec {
            name: format!("test{i}"), pointers: Vec::new(),
            scalar_kinds: if i == 1 { b"iili".to_vec() } else { b"i".to_vec() },
            scratch: [("scratch".to_string(), scratch[i] as u64)].into_iter().collect(),
            capacity_rows: rows as u32,
            geometry: serde_json::json!({"heads": heads, "hidden": 7168, "q_lora_rank": 1536,
                "kv_lora_rank": 512, "qk_nope_dim": 128, "qk_rope_dim": 64, "v_head_dim": 128,
                "page_rows": 64, "norm_eps": 1e-6, "kv_cache": "bf16", "max_context": context,
                "num_splits": splits, "chunks_per_split": context.div_ceil(64).div_ceil(splits), "mode": mode.name()}),
        })
    }

    #[test]
    fn k31_native_mla_projection_requires_variant_epsilon_and_exact_seam_abi() {
        for heads in [48, 96] { for rows in [64, 4096] {
            for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
                let entries = specs(heads, rows, KimiMlaMode::Decode);
                let mut seams = [entries[0].clone(), entries[2].clone()];
                for (i, spec) in seams.iter_mut().enumerate() {
                    spec.geometry["norm_eps"] = 1e-5.into();
                    spec.pointers = (if i == 0 { PRODUCER } else { OUTPUT })
                        .iter().map(|name| (*name).into()).collect();
                }
                seams[0].geometry["kv_cache"] = format.label().into();
                let validate = |s: &[ProgramSpec; 2]| validate_k31_projection_manifest(s.each_ref(), heads, rows, format);
                assert_eq!(validate(&seams).unwrap(), [entries[0].scratch["scratch"] as usize,
                    entries[2].scratch["scratch"] as usize]);
                for i in 0..2 {
                    for (key, value) in [("norm_eps", serde_json::json!(1e-6)),
                        ("heads", serde_json::json!(32)), ("page_rows", serde_json::json!(1))] {
                        let mut wrong = seams.clone(); wrong[i].geometry[key] = value;
                        assert!(validate(&wrong).is_err(), "{i}/{key}");
                    }
                    let mut wrong = seams.clone(); wrong[i].pointers.swap(0, 1);
                    assert!(validate(&wrong).is_err());
                    let mut wrong = seams.clone(); wrong[i].scratch.insert("unadmitted".into(), 16);
                    assert!(validate(&wrong).is_err());
                    let mut wrong = seams.clone(); wrong[i].capacity_rows += 1;
                    assert!(validate(&wrong).is_err());
                    let mut wrong = seams.clone(); wrong[i].scalar_kinds = b"ii".to_vec();
                    assert!(validate(&wrong).is_err());
                }
                seams[0].geometry["kv_cache"] = "wrong".into();
                assert!(validate(&seams).is_err());
            }
        }}
    }

    #[test]
    fn k31_index_selection_checks_causal_metadata_extents_and_writer_aliases() {
        let buffer = |index| Buffer { ptr: (index * 0x10000000usize) as *mut c_void,
            bytes: 0x10000000, device_id: 0, flags: 0 };
        let b = KimiIndexSelection { rows: 17, cache_pages: 128, table_width: 64, table_stride: 80,
            page_stride: 1, page_offset: 0,
            query_fp8: buffer(1), head_weights: buffer(2), cache: buffer(3), table: buffer(4),
            lengths: buffer(5), indices: buffer(6), scores: buffer(7), scratch: buffer(8) };
        for mode in [KimiIndexMode::Decode, KimiIndexMode::Prefill] {
            for scored in [false, true] {
                let check = |b| validate_topk_batch(b, 64, 16384, mode, scored, 4096);
                check(b).unwrap();
                let table = if mode == KimiIndexMode::Decode { (b.rows - 1) * b.table_stride + b.table_width }
                    else { b.table_width };
                for bad in [KimiIndexSelection { rows: 0, ..b }, KimiIndexSelection { rows: 65, ..b },
                    KimiIndexSelection { table_width: 16385, ..b }, KimiIndexSelection { table_stride: 63, ..b },
                    KimiIndexSelection { cache_pages: usize::MAX, ..b },
                    KimiIndexSelection { page_stride: 3, ..b }, KimiIndexSelection { page_stride: 2,
                        page_offset: 2, ..b }, KimiIndexSelection { page_stride: 1, page_offset: 1, ..b },
                    KimiIndexSelection { table: Buffer { bytes: table * 4 - 1, ..b.table }, ..b },
                    KimiIndexSelection { cache: Buffer { bytes: b.cache_pages * 8448 - 1, ..b.cache }, ..b },
                    KimiIndexSelection { lengths: Buffer { bytes: b.rows * 4 - 1, ..b.lengths }, ..b }] {
                    assert!(check(bad).is_err());
                }
                for reader in [b.query_fp8, b.head_weights, b.cache, b.table, b.lengths] {
                    assert!(check(KimiIndexSelection { indices: reader, ..b }).is_err());
                    assert!(check(KimiIndexSelection { scratch: reader, ..b }).is_err());
                }
                if scored {
                    assert!(check(KimiIndexSelection { scores: b.indices, ..b }).is_err());
                    assert!(check(KimiIndexSelection { scores: b.cache, ..b }).is_err());
                } else {
                    check(KimiIndexSelection { scores: Buffer { ptr: std::ptr::null_mut(), bytes: 0,
                        device_id: 0, flags: 0 }, ..b }).unwrap();
                }
            }
        }
    }

    #[test]
    fn k31_selector_manifest_pins_head_count_and_dynamic_sm_abi() {
        for mode in [KimiIndexMode::Decode, KimiIndexMode::Prefill] {
            for (scored, runtime) in [(false,false), (false,true), (true,false)] {
                let pointers = if scored { SCORED_TOPK } else { INDEX_TOPK };
                let spec = ProgramSpec { name: "topk".into(), capacity_rows: 64,
                    pointers: pointers.iter().map(|s| (*s).into()).collect(),
                    scalar_kinds: if scored { b"iiiii".to_vec() } else if runtime { b"iiii".to_vec() } else { b"iii".to_vec() },
                    scratch: [("scratch".into(),4096)].into_iter().collect(),
                    geometry: serde_json::json!({"topk":2048,"heads":32,"max_rows":64,
                        "max_pages":16384,"mode":mode.name(),"runtime_sm_count":runtime}) };
                assert_eq!(validate_topk_manifest(&spec,64,16384,mode,scored).unwrap(),(4096,runtime));
                for (key,value) in [("topk",1024),("heads",64)] {
                    let mut wrong=spec.clone(); wrong.geometry[key]=serde_json::json!(value);
                    assert!(validate_topk_manifest(&wrong,64,16384,mode,scored).is_err());
                }
                let mut wrong=spec.clone(); wrong.scalar_kinds.push(b'i');
                assert!(validate_topk_manifest(&wrong,64,16384,mode,scored).is_err());
                let mut wrong=spec.clone(); wrong.scratch.clear();
                assert!(validate_topk_manifest(&wrong,64,16384,mode,scored).is_err());
            }
        }
    }

    #[test]
    fn k31_candidate_merge_retains_full_global_runs_and_disjoint_local_outputs() {
        let buffer = |index| Buffer { ptr: (index * 0x10000000usize) as *mut c_void,
            bytes: 0x10000000, device_id: 0, flags: 0 };
        let b = KimiCandidateBatch { rows: 17, table_stride: 8192, rank: 0,
            scores: buffer(1), indices: buffer(2), output_scores: buffer(3), output_indices: buffer(4),
            table: buffer(5), local_slots: buffer(6), local_lengths: buffer(7) };
        validate_candidate_batch(b).unwrap();
        validate_candidate_batch(KimiCandidateBatch { rank: 1, ..b }).unwrap();
        for bad in [KimiCandidateBatch { rank: 2, ..b }, KimiCandidateBatch { rows: 0, ..b },
            KimiCandidateBatch { table_stride: 8193, ..b },
            KimiCandidateBatch { scores: Buffer { bytes: b.rows*2048*4, ..b.scores }, ..b },
            KimiCandidateBatch { indices: Buffer { bytes: b.rows*2048*4, ..b.indices }, ..b },
            KimiCandidateBatch { local_lengths: Buffer { bytes: b.rows*4-1, ..b.local_lengths }, ..b }] {
            assert!(validate_candidate_batch(bad).is_err());
        }
        for reader in [b.scores,b.indices,b.table] {
            assert!(validate_candidate_batch(KimiCandidateBatch { local_slots: reader, ..b }).is_err());
            assert!(validate_candidate_batch(KimiCandidateBatch { output_scores: reader, ..b }).is_err());
        }
        assert!(validate_candidate_batch(KimiCandidateBatch { output_indices: b.local_slots, ..b }).is_err());
    }

    #[test]
    fn k31_selected_mla_manifest_separates_final_and_peer_lse_units() {
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            for (heads, begin) in [(48, None), (96, None), (96, Some(0)), (96, Some(48))] {
                let rows = 64;
                let n = rows * if begin.is_some() { 48 } else { heads } * 16;
                let scratch = n * (512 * 2 + 4);
                let mut geometry = specs(heads, rows, KimiMlaMode::Decode)[1].geometry.clone();
                geometry["norm_eps"] = serde_json::json!(1e-5);
                geometry["kv_cache"] = serde_json::json!(format.label());
                geometry["max_rows"] = serde_json::json!(rows);
                geometry["topk"] = serde_json::json!(2048);
                geometry["page_size"] = serde_json::json!(1);
                geometry["lse_base"] = if begin.is_some() { serde_json::json!(2) } else { serde_json::json!("e") };
                if let Some(begin) = begin {
                    geometry["head_begin"] = serde_json::json!(begin);
                    geometry["head_count"] = serde_json::json!(48);
                }
                let spec = ProgramSpec { name: "selected".into(), geometry,
                    pointers: SPARSE_ATTENTION.iter().map(|name| (*name).into()).collect(),
                    scalar_kinds: if begin.is_some() { b"ili".to_vec() } else { b"iili".to_vec() },
                    capacity_rows: rows as u32,
                    scratch: [("scratch".into(), scratch as u64)].into_iter().collect() };
                assert_eq!(validate_sparse_manifest(&spec, heads, rows, format, begin).unwrap(), scratch);
                for (key, value) in [("norm_eps", serde_json::json!(1e-6)),
                    ("page_size", serde_json::json!(64)), ("topk", serde_json::json!(1024)),
                    ("lse_base", serde_json::json!(if begin.is_some() { "e" } else { "2" }))] {
                    let mut wrong = spec.clone(); wrong.geometry[key] = value;
                    assert!(validate_sparse_manifest(&wrong, heads, rows, format, begin).is_err(), "{key}");
                }
                let mut wrong = spec.clone(); wrong.scratch.insert("scratch".into(), 1);
                assert!(validate_sparse_manifest(&wrong, heads, rows, format, begin).is_err());
                if begin.is_some() {
                    assert!(validate_sparse_manifest(&spec, 48, rows, format, begin).is_err());
                    assert!(validate_sparse_manifest(&spec, heads, rows, format, Some(24)).is_err());
                }
            }
        }
    }

    #[test]
    fn k31_selected_mla_accepts_sealed_export_manifest() {
        // Unmodified entries from the f44d9776 SM120 export, not synthesized
        // from the consumer's pointer-name constants.
        let manifest: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/kimi_k31_sparse_mla_sm120.json")).unwrap();
        assert_eq!(manifest["sparkinfer_revision"], "f44d97769fd540fafd7ed341ae6966554fa80ea7");
        let entries = manifest["programs"].as_array().unwrap();
        assert_eq!(entries.len(), 16);
        for entry in entries {
            let pointers = entry["pointers"].as_array().unwrap().iter()
                .map(|p| p["name"].as_str().unwrap().to_owned()).collect();
            let scalar_kinds = entry["scalars"].as_array().unwrap().iter().map(|scalar| {
                match scalar["type"].as_str().unwrap() {
                    "int32" => b'i', "int64" => b'l', other => panic!("unexpected scalar {other}"),
                }
            }).collect();
            let spec = ProgramSpec {
                name: entry["name"].as_str().unwrap().to_owned(), pointers, scalar_kinds,
                scratch: entry["scratch_bytes_at_capacity"].as_object().unwrap().iter()
                    .map(|(name, bytes)| (name.clone(), bytes.as_u64().unwrap())).collect(),
                capacity_rows: entry["capacity_rows"].as_u64().unwrap().try_into().unwrap(),
                geometry: entry["geometry"].clone(),
            };
            let heads = spec.geometry["heads"].as_u64().unwrap() as usize;
            let rows = spec.capacity_rows as usize;
            let begin = spec.geometry.get("head_begin").map(|value| value.as_u64().unwrap() as usize);
            let format = match spec.geometry["kv_cache"].as_str().unwrap() {
                "bf16" => MlaCacheFormat::Bf16, "fp8" => MlaCacheFormat::Fp8,
                other => panic!("unexpected cache format {other}"),
            };
            validate_sparse_manifest(&spec, heads, rows, format, begin).unwrap();
            for (index, alias) in [(2, "indices"), (3, "lengths"), (4, "query_offsets"), (5, "out")] {
                let mut wrong = spec.clone();
                wrong.pointers[index] = alias.into();
                assert!(validate_sparse_manifest(&wrong, heads, rows, format, begin).is_err(),
                    "{} accepted stale alias {alias}", spec.name);
            }
        }
    }

    #[test]
    fn k31_sparse_scratch_uses_live_rows_and_rejects_larger_head_or_row_launches() {
        let buffer = |index| Buffer { ptr: (index * 0x100000000usize) as *mut c_void,
            bytes: 0x100000000, device_id: 0, flags: 0 };
        for rows in [1, 17, 64, 2752, 4096] {
            for heads in [48, 96] {
                let scratch = KimiSparseMla::scratch_bytes_for(rows, heads, false).unwrap();
                let b = KimiSparseBatch { rows, cache_tokens: 128, query: buffer(1), cache: buffer(2),
                    indices: buffer(3), lengths: buffer(4), query_offsets: buffer(5), output: buffer(6),
                    lse: buffer(7), scratch: Buffer { bytes: scratch, ..buffer(8) } };
                validate_sparse_batch(b, heads, 4096, false, scratch, MlaCacheFormat::Fp8).unwrap();
                let short = KimiSparseBatch { scratch: Buffer { bytes: scratch - 1, ..b.scratch }, ..b };
                assert!(validate_sparse_batch(short, heads, 4096, false, scratch, MlaCacheFormat::Fp8).is_err());
                if heads == 48 {
                    let wider = KimiSparseMla::scratch_bytes_for(rows, 96, false).unwrap();
                    assert!(validate_sparse_batch(b, 96, 4096, false, wider, MlaCacheFormat::Fp8).is_err());
                }
                if rows < 4096 {
                    let next = KimiSparseBatch { rows: rows + 1, ..b };
                    let needed = KimiSparseMla::scratch_bytes_for(rows + 1, heads, false).unwrap();
                    assert!(validate_sparse_batch(next, heads, 4096, false, needed, MlaCacheFormat::Fp8).is_err());
                }
            }
        }
        assert!(KimiSparseMla::scratch_bytes_for(4097, 96, false).is_err());
        assert!(KimiSparseMla::scratch_bytes_for(64, 47, false).is_err());
    }

    #[test]
    fn k31_selected_mla_batch_checks_gather_extents_and_phase_aliases() {
        let buffer = |index| Buffer { ptr: (index * 0x10000000usize) as *mut c_void,
            bytes: 0x10000000, device_id: 0, flags: 0 };
        let b = KimiSparseBatch { rows: 17, cache_tokens: 128, query: buffer(1), cache: buffer(2),
            indices: buffer(3), lengths: buffer(4), query_offsets: buffer(5),
            output: buffer(6), lse: buffer(7), scratch: buffer(8) };
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            for (heads, partial) in [(48, false), (96, false), (96, true)] {
                let check = |b| validate_sparse_batch(b, heads, 64, partial, 4096, format);
                check(b).unwrap();
                for bad in [KimiSparseBatch { rows: 0, ..b }, KimiSparseBatch { rows: 65, ..b },
                    KimiSparseBatch { cache_tokens: 0, ..b }, KimiSparseBatch { cache_tokens: usize::MAX, ..b },
                    KimiSparseBatch { cache: Buffer { bytes: b.cache_tokens * format.record_bytes() - 1, ..b.cache }, ..b },
                    KimiSparseBatch { indices: Buffer { bytes: b.rows * 2048 * 4 - 1, ..b.indices }, ..b },
                    KimiSparseBatch { query_offsets: Buffer { bytes: (b.rows + 1) * 4 - 1, ..b.query_offsets }, ..b },
                    KimiSparseBatch { query: Buffer { bytes: b.rows * heads * 576 * 2 - 1, ..b.query }, ..b }] {
                    assert!(check(bad).is_err());
                }
                for reader in [b.query, b.cache, b.indices, b.lengths, b.query_offsets] {
                    assert!(check(KimiSparseBatch { output: Buffer { ptr: reader.ptr, ..b.output }, ..b }).is_err());
                    assert!(check(KimiSparseBatch { scratch: Buffer { ptr: reader.ptr, ..b.scratch }, ..b }).is_err());
                }
                assert!(check(KimiSparseBatch { lse: b.output, ..b }).is_err());
                for bad in [Buffer { flags: 1, ..b.output }, Buffer { bytes: 1, ..b.output },
                    Buffer { device_id: 1, ..b.output },
                    Buffer { ptr: (usize::MAX - 15) as *mut c_void, ..b.output }] {
                    assert!(check(KimiSparseBatch { output: bad, ..b }).is_err());
                }
                if partial {
                    check(KimiSparseBatch { output: Buffer { bytes: b.rows * 48 * 512 * 2, ..b.output },
                        lse: Buffer { bytes: b.rows * 48 * 4, ..b.lse }, ..b }).unwrap();
                }
            }
        }
    }

    #[test]
    fn k31_index_manifest_requires_exact_native_qdq_and_cache_layout() {
        let rows = 64;
        let scratch = rows * (4096 + 160) * 2;
        let spec = ProgramSpec { name: "kimik31_index_producer_m64".into(),
            pointers: INDEX_PRODUCER.iter().map(|name| (*name).into()).collect(),
            scalar_kinds: b"i".to_vec(), capacity_rows: rows as u32,
            scratch: [("scratch".into(), scratch as u64)].into_iter().collect(),
            geometry: serde_json::json!({"hidden":7168,"q_lora_rank":1536,"index_heads":32,
                "index_head_dim":128,"index_topk":2048,"page_rows":64,"max_rows":64,
                "index_norm_eps":1e-5,"rope":false,"scale_format":"ue8m0","scale_storage":"float32",
                "query_amax_floor":1e-10,"key_amax_floor":1e-4}) };
        assert_eq!(validate_index_manifest(&spec, rows).unwrap(), scratch);
        for (key, value) in [("index_norm_eps", serde_json::json!(1e-6)),
            ("rope", serde_json::json!(true)), ("scale_format", serde_json::json!("fp32")),
            ("scale_storage", serde_json::json!("uint8")), ("query_amax_floor", serde_json::json!(0)),
            ("key_amax_floor", serde_json::json!(1e-10)), ("index_topk", serde_json::json!(1024))] {
            let mut wrong = spec.clone();
            wrong.geometry[key] = value;
            assert!(validate_index_manifest(&wrong, rows).is_err(), "{key}");
        }
        let mut wrong = spec.clone(); wrong.scratch.insert("scratch".into(), 1);
        assert!(validate_index_manifest(&wrong, rows).is_err());
        let mut wrong = spec.clone(); wrong.pointers.swap(1, 2);
        assert!(validate_index_manifest(&wrong, rows).is_err());
        assert!(validate_index_manifest(&spec, rows + 1).is_err());
    }

    #[test]
    fn k31_index_batch_rejects_writer_aliases_and_undercharged_cache() {
        let buffer = |index| Buffer { ptr: (index * 0x10000000usize) as *mut c_void,
            bytes: 0x10000000, device_id: 0, flags: 0 };
        let weights = KimiIndexWeights { query: buffer(1), key_heads: buffer(2), norm: buffer(3), bias: buffer(4) };
        let b = KimiIndexBatch { rows: 17, cache_pages: 3, x: buffer(5), q_resid: buffer(6),
            positions: buffer(7), slots: buffer(8), cos_sin: buffer(9), cache: buffer(10),
            query_fp8: buffer(11), head_weights: buffer(12), scratch: buffer(13) };
        let check = |b| validate_index_batch(weights, b, 64, 4096);
        check(b).unwrap();
        assert!(check(KimiIndexBatch { rows: 0, ..b }).is_err());
        assert!(check(KimiIndexBatch { rows: 65, ..b }).is_err());
        assert!(check(KimiIndexBatch { cache_pages: usize::MAX, ..b }).is_err());
        assert!(check(KimiIndexBatch { cache: Buffer { bytes: 3 * 8448 - 1, ..b.cache }, ..b }).is_err());
        for reader in [b.x, b.q_resid, b.positions, b.slots, b.cos_sin,
            weights.query, weights.key_heads, weights.norm, weights.bias] {
            assert!(check(KimiIndexBatch { cache: Buffer { ptr: reader.ptr, ..b.cache }, ..b }).is_err());
            assert!(check(KimiIndexBatch { query_fp8: Buffer { ptr: reader.ptr, ..b.query_fp8 }, ..b }).is_err());
        }
        assert!(check(KimiIndexBatch { head_weights: b.query_fp8, ..b }).is_err());
        assert!(check(KimiIndexBatch { scratch: b.cache, ..b }).is_err());
    }

    #[test]
    fn context_producer_and_output_validate_only_their_phase_operands() {
        let buffer = |index| Buffer { ptr: (index * 0x10000000usize) as *mut c_void,
            bytes: 0x10000000, device_id: 0, flags: 0 };
        let absent = Buffer { ptr: std::ptr::null_mut(), bytes: 0, device_id: 0, flags: 0 };
        let b = KimiMlaBatch { rows: 8, requests: 0, page_table_width: 0, cache_pages: 2, active_splits: 0,
            x: buffer(1), slots: buffer(2), cache: buffer(3), page_table: absent, lengths: absent,
            query_offsets: absent, query: buffer(4), q_resid: buffer(5), attention: buffer(6), lse: absent,
            output: buffer(7), producer_scratch: buffer(8), attention_scratch: absent, output_scratch: buffer(9) };
        let w = KimiMlaWeights { qkv_a: buffer(10), q_a_norm: buffer(11), kv_a_norm: buffer(12),
            q_b: buffer(13), uk: buffer(14), uv: buffer(15), gate: buffer(16), output: buffer(17) };
        let producer = |b| validate_producer(w, b, 48, 64, 4096, MLA_RECORD_BYTES);
        let output = |b| validate_output(w, b, 48, 64, 4096);
        producer(b).unwrap();
        output(b).unwrap();
        // No phase demands the exchange's h96 assembled query, full history,
        // attention tables, or another phase's scratch before enqueueing.
        producer(KimiMlaBatch { attention: absent, output: absent, output_scratch: absent, ..b }).unwrap();
        output(KimiMlaBatch { slots: absent, cache: absent, query: absent, q_resid: absent,
            producer_scratch: absent, cache_pages: 0, ..b }).unwrap();
        for bad in [KimiMlaBatch { rows: 0, ..b }, KimiMlaBatch { rows: 65, ..b }] {
            assert!(producer(bad).is_err()); assert!(output(bad).is_err());
        }
        assert!(producer(KimiMlaBatch { cache_pages: usize::MAX, ..b }).is_err());
        for reader in [b.x, b.slots, w.qkv_a, w.q_a_norm, w.kv_a_norm, w.q_b, w.uk] {
            assert!(producer(KimiMlaBatch { query: Buffer { ptr: reader.ptr, ..b.query }, ..b }).is_err());
        }
        for reader in [b.attention, b.x, w.uv, w.gate, w.output] {
            assert!(output(KimiMlaBatch { output: Buffer { ptr: reader.ptr, ..b.output }, ..b }).is_err());
        }
        for bad in [Buffer { flags: 1, ..b.query }, Buffer { bytes: 1, ..b.query },
            Buffer { device_id: 1, ..b.query }, Buffer { ptr: (usize::MAX - 15) as *mut c_void, ..b.query }] {
            assert!(producer(KimiMlaBatch { query: bad, ..b }).is_err());
            assert!(output(KimiMlaBatch { output: bad, ..b }).is_err());
        }
    }

    #[test]
    fn projected_attention_checks_geometry_extents_and_writer_aliases() {
        let buffer=|index|Buffer{ptr:(index*0x10000000usize)as *mut c_void,
            bytes:0x10000000,device_id:0,flags:0};
        let b=KimiMlaBatch{rows:8,requests:1,page_table_width:1,cache_pages:1,active_splits:16,
            x:buffer(9),slots:buffer(10),cache:buffer(2),page_table:buffer(3),lengths:buffer(4),
            query_offsets:buffer(5),query:buffer(1),q_resid:buffer(11),attention:buffer(6),lse:buffer(7),
            output:buffer(12),producer_scratch:buffer(13),attention_scratch:buffer(8),output_scratch:buffer(14)};
        let validate=|batch|validate_projected_attention(batch,96,64,1048576,16,4096,MLA_RECORD_BYTES);
        validate(b).unwrap();
        for bad in [KimiMlaBatch{rows:0,..b},KimiMlaBatch{rows:65,..b},
            KimiMlaBatch{requests:9,..b},KimiMlaBatch{page_table_width:16385,..b},
            KimiMlaBatch{cache_pages:usize::MAX,..b},KimiMlaBatch{active_splits:1,..b}] {
            assert!(validate(bad).is_err());
        }
        let readers=[b.query,b.cache,b.page_table,b.lengths,b.query_offsets];
        for reader in readers {
            assert!(validate(KimiMlaBatch{attention:Buffer{ptr:reader.ptr,..b.attention},..b}).is_err());
            assert!(validate(KimiMlaBatch{lse:Buffer{ptr:reader.ptr,..b.lse},..b}).is_err());
            assert!(validate(KimiMlaBatch{attention_scratch:Buffer{ptr:reader.ptr,..b.attention_scratch},..b}).is_err());
        }
        for query in [Buffer{bytes:1,..b.query},Buffer{device_id:1,..b.query},Buffer{flags:1,..b.query},
            Buffer{ptr:std::ptr::null_mut(),..b.query},
            Buffer{ptr:(usize::MAX-15)as *mut c_void,..b.query}] {
            assert!(validate(KimiMlaBatch{query,..b}).is_err());
        }
        assert!(validate(KimiMlaBatch{lse:Buffer{ptr:b.attention.ptr,..b.lse},..b}).is_err());
    }

    #[test]
    fn context_partial_manifest_rejects_lse_base_and_owned_traversal_drift() {
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            for begin in [0, 48] {
                let mut spec = specs(96, 64, KimiMlaMode::Decode)[1].clone();
                spec.scalar_kinds = b"il".to_vec();
                spec.geometry["runtime_splits"] = true.into();
                spec.geometry["device_splits"] = true.into();
                spec.geometry["fixed_split_grid"] = 16.into();
                spec.geometry["kv_cache"] = format.label().into();
                spec.geometry["head_begin"] = begin.into();
                spec.geometry["head_count"] = 48.into();
                spec.geometry["heads_per_tile"] = 16.into();
                spec.geometry["max_rows"] = 64.into();
                spec.geometry["lse_base"] = 2.into();
                spec.scratch.insert("scratch".into(), (64 * 48 * 16 * (512 * 2 + 4)) as u64);
                assert!(validate_partial_manifest(&spec, begin, 64, format).is_ok());
                let mut tile8 = spec.clone();
                tile8.geometry["heads_per_tile"] = 8.into();
                assert!(validate_partial_manifest(&tile8, begin, 64, format).is_ok());
                for (key, bad) in [("lse_base", 0), ("heads", 48), ("head_count", 96),
                    ("num_splits", 1), ("runtime_splits", 0), ("device_splits", 0),
                    ("fixed_split_grid", 8), ("max_context", 0), ("heads_per_tile", 4)] {
                    let mut wrong = spec.clone();
                    wrong.geometry[key] = bad.into();
                    assert!(validate_partial_manifest(&wrong, begin, 64, format).is_err(), "{key}");
                }
                let mut wrong = spec.clone();
                wrong.scratch.clear();
                assert!(validate_partial_manifest(&wrong, begin, 64, format).is_err());
                wrong = spec.clone();
                wrong.scalar_kinds = b"iili".to_vec();
                assert!(validate_partial_manifest(&wrong, begin, 64, format).is_err());
                assert!(validate_partial_manifest(&spec, 1, 64, format).is_err());
            }
        }
    }

    #[test]
    fn manifest_rejects_stale_geometry_traversal_and_scratch_before_launch() {
        for heads in [48, 96] {
            for mode in [KimiMlaMode::Decode, KimiMlaMode::Verify, KimiMlaMode::Extend] {
                let entries = specs(heads, 64, mode);
                assert!(validate_manifest(entries.each_ref(), heads, 64, mode, MlaCacheFormat::Bf16).is_ok());
                for index in 0..3 {
                    for (key, bad) in [("heads", serde_json::json!(32)), ("hidden", serde_json::json!(4096)),
                        ("page_rows", serde_json::json!(128)), ("norm_eps", serde_json::json!(1e-5))] {
                        let mut wrong = entries.clone(); wrong[index].geometry[key] = bad;
                        assert!(validate_manifest(wrong.each_ref(), heads, 64, mode, MlaCacheFormat::Bf16).is_err());
                    }
                    let mut wrong = entries.clone(); wrong[index].scratch.clear();
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode, MlaCacheFormat::Bf16).is_err());
                    let mut wrong = entries.clone(); wrong[index].scalar_kinds = b"iii".to_vec();
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode, MlaCacheFormat::Bf16).is_err());
                    let mut wrong = entries.clone(); wrong[index].capacity_rows = 4096;
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode, MlaCacheFormat::Bf16).is_err());
                }
                for (key, bad) in [("max_context", serde_json::json!(0)), ("num_splits", serde_json::json!(0)),
                    ("chunks_per_split", serde_json::json!(1)), ("mode", serde_json::json!("wrong"))] {
                    let mut wrong = entries.clone(); wrong[1].geometry[key] = bad;
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode, MlaCacheFormat::Bf16).is_err());
                }
            }
        }
    }

    #[test]
    fn compact_manifest_rejects_mixed_writer_reader_formats() {
        for heads in [48, 96] {
            for mode in [KimiMlaMode::Decode, KimiMlaMode::Verify, KimiMlaMode::Extend] {
                let mut compact = specs(heads, 64, mode);
                for spec in &mut compact[..2] { spec.geometry["kv_cache"] = "fp8".into(); }
                assert!(validate_manifest(compact.each_ref(), heads, 64, mode, MlaCacheFormat::Fp8).is_ok());
                assert!(validate_manifest(compact.each_ref(), heads, 64, mode, MlaCacheFormat::Bf16).is_err());
                for index in 0..2 {
                    let mut mixed = compact.clone();
                    mixed[index].geometry["kv_cache"] = "bf16".into();
                    assert!(validate_manifest(mixed.each_ref(), heads, 64, mode, MlaCacheFormat::Fp8).is_err());
                }
            }
        }
    }

    fn residual_spec(site: KimiResidualSite) -> ProgramSpec {
        ProgramSpec { name: site.name(), pointers: vec![], scalar_kinds: b"iif".to_vec(),
            scratch: Default::default(), capacity_rows: 4096,
            geometry: serde_json::json!({"hidden": 7168, "candidates": site.candidates,
                "prefix_mode": site.prefix.name(), "append": site.append,
                "has_mix": site.candidates != 1}) }
    }

    #[test]
    fn residual_sites_and_manifests_match_only_exported_transitions() {
        let mut count = 0;
        for candidates in 0..=10 {
            for prefix in [KimiPrefixMode::Keep, KimiPrefixMode::Add, KimiPrefixMode::Replace] {
                for append in [false, true] {
                    let site = KimiResidualSite { candidates, prefix, append };
                    let spec = residual_spec(site);
                    assert_eq!(validate_residual_manifest(&spec, site).is_ok(), site.valid());
                    if !site.valid() { continue; }
                    count += 1;
                    for (key, value) in [("hidden", serde_json::json!(3584)),
                        ("candidates", serde_json::json!(0)), ("prefix_mode", serde_json::json!("unknown")),
                        ("append", serde_json::json!(!append)), ("has_mix", serde_json::json!(candidates == 1))] {
                        let mut wrong = spec.clone(); wrong.geometry[key] = value;
                        assert!(validate_residual_manifest(&wrong, site).is_err());
                    }
                    let mut wrong = spec.clone(); wrong.scalar_kinds = b"iii".to_vec();
                    assert!(validate_residual_manifest(&wrong, site).is_err());
                    let mut wrong = spec.clone(); wrong.capacity_rows = 0;
                    assert!(validate_residual_manifest(&wrong, site).is_err());
                    let mut wrong = spec.clone(); wrong.scratch.insert("scratch".into(), 1);
                    assert!(validate_residual_manifest(&wrong, site).is_err());
                }
            }
        }
        assert_eq!(count, 32);
    }

    #[test]
    fn router_and_expert_wire_reject_stale_manifests_before_launch() {
        let site = KimiResidualSite { candidates: 1, prefix: KimiPrefixMode::Keep, append: true };
        let mut router = residual_spec(site);
        router.scalar_kinds = b"i".to_vec();
        router.geometry = serde_json::json!({"hidden": 7168, "experts": 896, "top_k": 16});
        validate_router_manifest(&router).unwrap();
        for key in ["hidden", "experts", "top_k"] {
            let mut bad = router.clone(); bad.geometry[key] = serde_json::json!(1);
            assert!(validate_router_manifest(&bad).is_err());
        }
        let mut wire = residual_spec(site);
        wire.scalar_kinds = b"ii".to_vec();
        wire.geometry = serde_json::json!({"hidden": 3584, "row_bytes": 3696, "expected_m": 80, "amax_floor": 1e-4});
        validate_expert_input_manifest(&wire).unwrap();
        for key in ["hidden", "row_bytes", "expected_m", "amax_floor"] {
            let mut bad = wire.clone(); bad.geometry[key] = serde_json::json!(1);
            assert!(validate_expert_input_manifest(&bad).is_err());
        }
        for (spec, validate) in [(router, validate_router_manifest as fn(&ProgramSpec) -> W8Result<()>),
            (wire, validate_expert_input_manifest)] {
            let mut bad = spec.clone(); bad.scalar_kinds.clear(); assert!(validate(&bad).is_err());
            let mut bad = spec.clone(); bad.capacity_rows = 0; assert!(validate(&bad).is_err());
            let mut bad = spec.clone(); bad.scratch.insert("scratch".into(), 1); assert!(validate(&bad).is_err());
        }
    }

    #[test]
    fn device_routes_and_wire_require_disjoint_admitted_storage() {
        let buffer = |index, bytes| Buffer { ptr: (0x10000000usize * index) as *mut c_void,
            bytes, device_id: 0, flags: 0 };
        let rows = 64;
        let input = buffer(1, rows * 7168 * 2);
        let weight = buffer(2, 896 * 7168 * 2);
        let bias = buffer(3, 896 * 4);
        let logits = buffer(4, rows * 896 * 4);
        let routes = KimiRoutes { ids: buffer(5, rows * 16 * 4), weights: buffer(6, rows * 16 * 4) };
        validate_router_buffers(rows, rows, input, weight, bias, logits, routes).unwrap();
        for writer in 3..6 {
            for other in 0..6 {
                if writer == other { continue; }
                let mut buffers = [input, weight, bias, logits, routes.ids, routes.weights];
                buffers[writer].ptr = buffers[other].ptr;
                assert!(validate_router_buffers(rows, rows, buffers[0], buffers[1], buffers[2], buffers[3],
                    KimiRoutes { ids: buffers[4], weights: buffers[5] }).is_err());
            }
        }
        for bad in [Buffer { bytes: 1, ..routes.ids }, Buffer { device_id: 1, ..routes.ids },
            Buffer { ptr: std::ptr::null_mut(), ..routes.ids },
            Buffer { ptr: (routes.ids.ptr as usize + 1) as *mut c_void, ..routes.ids }] {
            assert!(validate_router_buffers(rows, rows, input, weight, bias, logits,
                KimiRoutes { ids: bad, ..routes }).is_err());
        }
        for n in [0, rows + 1, usize::MAX] {
            assert!(validate_router_buffers(n, rows, input, weight, bias, logits, routes).is_err());
        }
        let latent = buffer(1, rows * 3584 * 2);
        let wire = buffer(2, rows * KimiExpertInput::ROW_BYTES);
        validate_expert_input_buffers(rows, rows, 170, latent, wire).unwrap();
        for grid in [0, usize::MAX] {
            assert!(validate_expert_input_buffers(rows, rows, grid, latent, wire).is_err());
        }
        for n in [0, rows + 1, usize::MAX] {
            assert!(validate_expert_input_buffers(n, rows, 1, latent, wire).is_err());
        }
        for bad in [Buffer { bytes: wire.bytes - 1, ..wire }, Buffer { device_id: 1, ..wire },
            Buffer { ptr: latent.ptr, ..wire }, Buffer { ptr: std::ptr::null_mut(), ..wire }] {
            assert!(validate_expert_input_buffers(rows, rows, 1, latent, bad).is_err());
        }
    }

    #[test]
    fn ffn_selection_defaults_direct_and_rejects_mislabeled_numerics() {
        assert_eq!(KimiFfnMode::default(), KimiFfnMode::Direct);
        let site = KimiResidualSite { candidates: 1, prefix: KimiPrefixMode::Keep, append: true };
        for intermediate in [3072, 6144, 16896, 33792] {
            for (mode, suffix) in [(KimiFfnMode::Direct, ""), (KimiFfnMode::Compensated, "_comp")] {
                assert_eq!(mode.gate_up_stem(intermediate), format!("kimik3_ffn_gate_up_i{intermediate}{suffix}"));
                let mut spec = residual_spec(site);
                spec.geometry = serde_json::json!({});
                if mode == KimiFfnMode::Compensated { spec.geometry["compensated"] = true.into(); }
                mode.validate(&spec).unwrap();
                spec.geometry["compensated"] = (mode == KimiFfnMode::Direct).into();
                assert!(mode.validate(&spec).is_err());
                spec.geometry["compensated"] = "true".into();
                assert!(mode.validate(&spec).is_err());
            }
        }
    }

    #[test]
    fn ffn_projection_manifests_preserve_bf16_materialization() {
        for inter in [3072, 6144, 16896, 33792] {
            for (width, outputs) in [(7168, 2 * inter), (inter, 7168)] {
                let site = KimiResidualSite { candidates: 1, prefix: KimiPrefixMode::Keep, append: true };
                let mut spec = residual_spec(site);
                spec.scalar_kinds = b"i".to_vec();
                spec.geometry = serde_json::json!({"width": width, "outputs": outputs,
                    "output_dtype": "torch.bfloat16", "skinny_rows": 0});
                validate_projection_manifest(&spec, width, outputs).unwrap();
                for (key, value) in [("width", serde_json::json!(1)), ("outputs", serde_json::json!(1)),
                    ("output_dtype", serde_json::json!("torch.float32")), ("skinny_rows", serde_json::json!(16))] {
                    let mut wrong = spec.clone(); wrong.geometry[key] = value;
                    assert!(validate_projection_manifest(&wrong, width, outputs).is_err());
                }
            }
        }
    }

    #[test]
    fn latent_native_norm_contract_and_projection_buffer_admission() {
        for hidden in [3584, 7168] {
            let site = KimiResidualSite { candidates: 1, prefix: KimiPrefixMode::Keep, append: true };
            let mut spec = residual_spec(site);
            spec.scalar_kinds = b"if".to_vec();
            spec.geometry = serde_json::json!({"hidden": hidden});
            validate_norm_manifest(&spec, hidden).unwrap();
            let mut bad = spec.clone(); bad.scalar_kinds = b"i".to_vec();
            assert!(validate_norm_manifest(&bad, hidden).is_err());
            let mut bad = spec.clone(); bad.geometry["hidden"] = serde_json::json!(1);
            assert!(validate_norm_manifest(&bad, hidden).is_err());
            assert!(validate_norm_manifest(&spec, 512).is_err());
            let buffer = |ptr, bytes| Buffer { ptr: ptr as *mut c_void, bytes, device_id: 0, flags: 0 };
            let x = buffer(0x1000usize, hidden * 2);
            let w = buffer(0x100000usize, hidden * 2);
            let y = buffer(0x200000usize, hidden * 2);
            validate_norm_buffers(1, 4096, hidden, x, w, y).unwrap();
            assert!(validate_norm_buffers(0, 4096, hidden, x, w, y).is_err());
            assert!(validate_norm_buffers(2, 1, hidden, x, w, y).is_err());
            assert!(validate_norm_buffers(2, 4096, hidden, x, w, y).is_err());
            assert!(validate_norm_buffers(1, 4096, hidden, x, w, x).is_err());
            assert!(validate_norm_buffers(1, 4096, hidden, x, w, w).is_err());
            assert!(validate_norm_buffers(1, 4096, hidden, x, w,
                Buffer { device_id: 1, ..y }).is_err());
        }
        for (width, outputs) in [(7168, 3584), (3584, 7168)] {
            let buffer = |ptr, bytes| Buffer { ptr: ptr as *mut c_void, bytes, device_id: 0, flags: 0 };
            let x = buffer(0x1000usize, width * 2);
            let w = buffer(0x100000usize, width * outputs * 2);
            let y = buffer(0x10000000usize, outputs * 2);
            validate_projection_buffers(1, 4096, width, outputs, x, w, y).unwrap();
            assert!(validate_projection_buffers(0, 4096, width, outputs, x, w, y).is_err());
            assert!(validate_projection_buffers(1, 4096, width, outputs, x, w,
                Buffer { ptr: w.ptr, ..y }).is_err());
            assert!(validate_projection_buffers(1, 4096, width, outputs, x,
                Buffer { bytes: w.bytes - 1, ..w }, y).is_err());
        }
    }

    #[test]
    fn w8_extents_aliases_and_graph_stable_arena_stride() {
        assert_eq!(KimiResidual::arena_bytes(4099).unwrap(), 9 * 4099 * 7168 * 2);
        assert!(KimiResidual::arena_bytes(0).is_err());
        assert!(KimiResidual::arena_bytes(i32::MAX as usize + 1).is_err());
        let a = Buffer { ptr: 0x1000usize as *mut c_void, bytes: 1024, device_id: 0, flags: 0 };
        let b = Buffer { ptr: 0x1400usize as *mut c_void, ..a };
        validate_w8_buffers(&[(a, 1024), (b, 1024)], 0).unwrap();
        reject_overlap((a, 1024), (b, 1024)).unwrap();
        assert!(reject_overlap((a, 1025), (b, 1024)).is_err());
        assert!(reject_overlap((a, 16), (a, 16)).is_err());
        assert!(validate_w8_buffers(&[(a, 1025)], 0).is_err());
        assert!(validate_w8_buffers(&[(a, 16)], 1).is_err());
        assert!(validate_w8_buffers(&[(Buffer { ptr: 0x1001usize as *mut c_void, ..a }, 16)], 0).is_err());
        let wrapping = Buffer { ptr: (usize::MAX - 15) as *mut c_void, ..a };
        assert!(validate_w8_buffers(&[(wrapping, 16)], 0).is_err());
        assert!(reject_overlap((wrapping, 16), (a, 16)).is_err());
    }

    fn kda_spec(decode: bool) -> ProgramSpec {
        let rows = if decode { 64 } else { 4096 };
        let align = |n: usize| n.div_ceil(1024) * 1024;
        let scratch = align(rows * 24752 * 2) + align(rows * 2 * 6144 * 2)
            + align(rows * 3 * 6144 * 2) + 2 * align(rows * 6144 * 2);
        ProgramSpec { name: "test_kda".into(), pointers: vec![],
            scalar_kinds: if decode { b"ii".to_vec() } else { b"i".to_vec() },
            scratch: [("scratch".into(), (scratch + if decode { 0 } else { 1024 }) as u64)].into_iter().collect(),
            capacity_rows: rows as u32,
            geometry: serde_json::json!({"hidden": 7168, "heads": 48, "head_dim": 128,
                "max_rows": rows, "in_width": 24752, "lower_bound": -5.0, "eps": 1e-5,
                "output_kind": "norm", "output_gate": "full_rank", "o_norm_dtype": "float32",
                "beta_sigmoid_dtype": "float32", "use_gate_in_kernel": true, "safe_gate": true,
                "bf16_decode": if decode { Some(true) } else { None },
                "chunked_min_rows": if decode { None } else { Some(64) }}) }
    }

    #[test]
    fn kda_manifests_reject_substitute_numerics_and_wrong_replay_geometry() {
        assert_eq!(KimiKda::state_bytes(1).unwrap(), 3 * 1024 * 1024);
        assert_eq!(KimiKda::conv_state_bytes(1).unwrap(), 110592);
        assert_eq!(KimiKda::replay_bytes(), 7090176);
        assert!(KimiKda::state_bytes(0).is_err());
        assert!(KimiKda::state_bytes(usize::MAX).is_err());
        for decode in [false, true] {
            let spec = kda_spec(decode);
            let rows = spec.capacity_rows as usize;
            validate_kda_manifest(&spec, rows, decode).unwrap();
            for (key, value) in [("heads", serde_json::json!(96)), ("in_width", serde_json::json!(24704)),
                ("output_kind", serde_json::json!("projection")), ("output_gate", serde_json::json!("low_rank")),
                ("o_norm_dtype", serde_json::json!("bfloat16")), ("beta_sigmoid_dtype", serde_json::json!("bfloat16")),
                ("state_dtype", serde_json::json!("bfloat16")), ("replay_rows", serde_json::json!(128)),
                ("safe_gate", serde_json::json!(false)), ("eps", serde_json::json!(1e-6))] {
                let mut wrong = spec.clone(); wrong.geometry[key] = value;
                assert!(validate_kda_manifest(&wrong, rows, decode).is_err(), "{key}");
            }
            let mut wrong = spec.clone(); wrong.scalar_kinds = b"iii".to_vec();
            assert!(validate_kda_manifest(&wrong, rows, decode).is_err());
            let mut wrong = spec.clone(); wrong.scratch.insert("scratch".into(), 0);
            assert!(validate_kda_manifest(&wrong, rows, decode).is_err());
            assert!(validate_kda_manifest(&spec, rows / 2, decode).is_err());
        }
        let spec = ProgramSpec { name: "commit".into(), pointers: vec![], scalar_kinds: b"iii".to_vec(),
            scratch: Default::default(), capacity_rows: 64,
            geometry: serde_json::json!({"heads": 48, "channels": 18432, "replay_rows": 64,
                "replay_bytes": KimiKda::replay_bytes()}) };
        validate_kda_commit_manifest(&spec, KimiKdaGeometry::new(48).unwrap()).unwrap();
        let mut wrong = spec.clone(); wrong.geometry["replay_bytes"] = serde_json::json!(1);
        assert!(validate_kda_commit_manifest(&wrong, KimiKdaGeometry::new(48).unwrap()).is_err());
    }

    #[test]
    fn whole_fused_kda_validates_native_full_head_operands() {
        let geometry = KimiKdaGeometry::new(96).unwrap();
        for decode in [false, true] {
            let mut spec = kda_spec(decode);
            let rows = spec.capacity_rows as usize;
            spec.geometry["heads"] = serde_json::json!(96);
            spec.geometry["in_width"] = serde_json::json!(geometry.input_width());
            let align = |n: usize| n.div_ceil(1024) * 1024;
            let width = geometry.width();
            let scratch = align(rows * geometry.input_width() * 2) + align(rows * 2 * width * 2)
                + align(rows * 3 * width * 2) + 2 * align(rows * width * 2)
                + if decode { 0 } else { 1024 };
            spec.scratch.insert("scratch".into(), scratch as u64);
            validate_kda_manifest_heads(&spec, rows, decode, geometry).unwrap();
            assert!(validate_kda_manifest(&spec, rows, decode).is_err());
            let (weights, batch) = kda_buffers(decode);
            validate_kda_buffers_heads(weights, batch, rows, decode, 1024, geometry).unwrap();
            let half_input = Buffer { bytes: 24752 * 7168 * 2, ..weights.input };
            assert!(validate_kda_buffers_heads(KimiKdaWeights { input: half_input, ..weights },
                batch, rows, decode, 1024, geometry).is_err());
            let half_state = Buffer { bytes: KimiKda::state_bytes(batch.state_slots).unwrap(), ..batch.state };
            assert!(validate_kda_buffers_heads(weights, KimiKdaBatch { state: half_state, ..batch },
                rows, decode, 1024, geometry).is_err());
        }
    }

    #[test]
    fn whole_kda_state_and_commit_use_all_96_heads() {
        let half = KimiKdaGeometry::new(48).unwrap();
        let whole = KimiKdaGeometry::new(96).unwrap();
        assert_eq!(whole.state_bytes(4).unwrap(), 2 * half.state_bytes(4).unwrap());
        assert_eq!(whole.conv_state_bytes(4).unwrap(), 2 * half.conv_state_bytes(4).unwrap());
        assert_eq!(whole.replay_bytes(), 2 * half.replay_bytes());
        assert_eq!((whole.width(), whole.aux_width(), whole.input_width()), (12288, 224, 49376));
        assert!(KimiKdaGeometry::new(24).is_err());
        assert!(whole.state_bytes(0).is_err());
        assert!(whole.conv_state_bytes(usize::MAX).is_err());
        let mut spec = ProgramSpec { name: "commit".into(), pointers: vec![], scalar_kinds: b"iii".to_vec(),
            scratch: Default::default(), capacity_rows: 64,
            geometry: serde_json::json!({"heads":96,"channels":36864,"replay_rows":64,
                "replay_bytes":whole.replay_bytes()}) };
        validate_kda_commit_manifest(&spec, whole).unwrap();
        assert!(validate_kda_commit_manifest(&spec, half).is_err());
        let (_, b) = kda_buffers(true);
        let requirements = kda_commit_requirements(whole, 2, 69, 4,
            b.state, b.conv_state, b.scratch, b.first).unwrap();
        assert_eq!(requirements[0].1, 69 * whole.state_bytes(4).unwrap());
        assert_eq!(requirements[1].1, 69 * whole.conv_state_bytes(4).unwrap());
        assert_eq!(requirements[2].1, 69 * whole.replay_bytes());
        spec.geometry["replay_bytes"] = serde_json::json!(half.replay_bytes());
        assert!(validate_kda_commit_manifest(&spec, whole).is_err());
    }

    fn kda_buffers(decode: bool) -> (KimiKdaWeights, KimiKdaBatch) {
        let buffer = |index: usize| Buffer { ptr: (index * 0x100000000usize) as *mut c_void,
            bytes: 0x40000000, device_id: 0, flags: 0 };
        (KimiKdaWeights { input: buffer(1), forget: buffer(2), convolution: buffer(3),
            a_log: buffer(4), dt_bias: buffer(5), norm: buffer(6), output: buffer(7) },
            KimiKdaBatch { rows: 16, sequences: 2, state_slots: 4, speculative: decode,
                input: buffer(8), state: buffer(9), conv_state: buffer(10), slots: buffer(11),
                first: buffer(12), normalized: buffer(13), output: buffer(14),
                replay: decode.then(|| buffer(15)), scratch: buffer(16) })
    }

    #[test]
    fn kda_admission_checks_every_writer_and_prefill_sequence_limit() {
        for decode in [false, true] {
            let (w, b) = kda_buffers(decode);
            let capacity = if decode { 64 } else { 4096 };
            validate_kda_buffers(w, b, capacity, decode, 1024).unwrap();
            assert!(validate_kda_buffers(w, KimiKdaBatch { rows: 0, ..b }, capacity, decode, 1024).is_err());
            assert!(validate_kda_buffers(w, KimiKdaBatch { state_slots: 0, ..b }, capacity, decode, 1024).is_err());
            assert!(validate_kda_buffers(w, KimiKdaBatch { output: w.input, ..b }, capacity, decode, 1024).is_err());
            assert!(validate_kda_buffers(w, KimiKdaBatch { normalized: b.input, ..b }, capacity, decode, 1024).is_err());
            assert!(validate_kda_buffers(w, KimiKdaBatch { scratch: b.state, ..b }, capacity, decode, 1024).is_err());
            assert!(validate_kda_buffers(w, KimiKdaBatch { conv_state: b.state, ..b }, capacity, decode, 1024).is_err());
            assert!(validate_kda_buffers(w, KimiKdaBatch { replay: if decode { None } else { Some(b.scratch) }, ..b },
                capacity, decode, 1024).is_err());
            assert!(validate_kda_buffers(KimiKdaWeights { norm: Buffer { bytes: 511, ..w.norm }, ..w },
                b, capacity, decode, 1024).is_err());
            assert!(validate_kda_buffers(w, KimiKdaBatch { input: Buffer { device_id: 1, ..b.input }, ..b },
                capacity, decode, 1024).is_err());
        }
        let (w, b) = kda_buffers(false);
        assert!(validate_kda_buffers(w, KimiKdaBatch { rows: 65, ..b }, 4096, false, 1024).is_err());
        assert!(validate_kda_buffers(w, KimiKdaBatch { speculative: true, ..b }, 4096, false, 1024).is_err());
        validate_kda_buffers(w, KimiKdaBatch { rows: 4096, sequences: 1, ..b }, 4096, false, 1024).unwrap();
        let requirements = kda_commit_requirements(KimiKdaGeometry::new(48).unwrap(), 2, 69, 4, b.state, b.conv_state, b.scratch, b.first).unwrap();
        assert_eq!(requirements[0].1, 69 * 4 * 3 * 1024 * 1024);
        assert!(kda_commit_requirements(KimiKdaGeometry::new(48).unwrap(), 2, 70, 4, b.state, b.conv_state, b.scratch, b.first).is_err());
        assert!(kda_commit_requirements(KimiKdaGeometry::new(48).unwrap(), 5, 69, 4, b.state, b.conv_state, b.scratch, b.first).is_err());
    }

    #[test]
    fn native_bf16_record_geometry_and_buffer_validation() {
        assert_eq!(MLA_RECORD_BYTES, 576 * 2);
        assert_eq!(MLA_RECORD_BYTES * MLA_PAGE_ROWS, 73728);
        let b = Buffer { ptr: 16usize as *mut c_void, bytes: 73728, device_id: 0, flags: 0 };
        validate_buffers(&[(b, 73728)], 0).unwrap();
        assert!(validate_buffers(&[(b, 73729)], 0).is_err());
        assert!(validate_buffers(&[(b, 73728)], 1).is_err());
        assert!(validate_buffers(&[(Buffer { ptr: std::ptr::null_mut(), ..b }, 0)], 0).is_err());
        assert!(validate_buffers(&[(Buffer { ptr: 17usize as *mut c_void, ..b }, 1)], 0).is_err());
    }
}
