//! Kimi NoPE MLA, residual-mix and Situ programs, resolved before capture.
//! The engine owns weights, caches, arenas, outputs and scratch through completion.
use crate::programs::{Program, ProgramSpec, Programs, Scalar};
use crate::CuteafdDeviceBuffer as Buffer;
use std::ffi::c_void;
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
}

impl<'a> KimiMla<'a> {
    pub fn load(programs: &'a Programs<'a>, heads: usize, rows: usize, mode: KimiMlaMode) -> Result<Self> {
        if !matches!(heads, 48 | 96) || !(1..=4096).contains(&rows) {
            return Err(KimiMlaError::Invalid("head/row geometry"));
        }
        let suffix = if heads == 48 { "" } else { "_h96" };
        let producer = programs.program(&format!("kimik3_mla_producer{suffix}_m{rows}"), PRODUCER)?;
        let attention = programs.program(&format!("kimik3_dense_mla_{}{suffix}_m{rows}", mode.name()), ATTENTION)?;
        let output = programs.program(&format!("kimik3_mla_output{suffix}_m{rows}"), OUTPUT)?;
        let (context, splits, scratch) = validate_manifest(
            [producer.spec(), attention.spec(), output.spec()], heads, rows, mode)?;
        Ok(Self { producer, attention, output, heads, rows, context, splits, scratch })
    }

    pub fn scratch_bytes(&self) -> [usize; 3] { self.scratch }
    pub fn max_context(&self) -> usize { self.context }
    pub fn splits(&self) -> usize { self.splits }

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
            (b.cache, b.cache_pages * MLA_PAGE_ROWS * MLA_RECORD_BYTES),
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

fn validate_manifest(specs: [&ProgramSpec; 3], heads: usize, rows: usize, mode: KimiMlaMode)
    -> Result<(usize, usize, [usize; 3])> {
    for (i, spec) in specs.iter().enumerate() {
        let geometry = &spec.geometry;
        if spec.capacity_rows as usize != rows || geometry["heads"].as_u64() != Some(heads as u64)
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

/// Shared KDA recurrence -> head-local gated norm -> BF16 o_proj partial.
/// The TP2 meet belongs to the owning engine, never inside this binding.
pub struct KimiKda<'a> {
    program: Program<'a>, output: Program<'a>, rows: usize, decode: bool, scratch: usize,
}
impl<'a> KimiKda<'a> {
    pub fn load(programs: &'a Programs<'a>, rows: usize, decode: bool) -> W8Result<Self> {
        let pointers = ["x", "w_in", "w_fg", "conv_w", "a_log", "dt_bias", "o_norm", "w_o",
            "conv_state", "state", "slots", "seq_first", "out", "replay", "scratch"];
        let prefill = [&pointers[..13], &pointers[14..]].concat();
        let program = programs.program(&format!("kimik3_kda_m{rows}"),
            if decode { &pointers } else { &prefill })?;
        let scratch = validate_kda_manifest(program.spec(), rows, decode)?;
        let output = programs.program("kimik3_ffn_down_i6144", &["x", "w", "out"])?;
        validate_projection_manifest(output.spec(), 6144, 7168)?;
        if output.spec().capacity_rows < rows as u32 {
            return Err(KimiW8Error::Invalid("KDA output capacity"));
        }
        Ok(Self { program, output, rows, decode, scratch })
    }

    pub fn scratch_bytes(&self) -> usize { self.scratch }
    pub fn state_bytes(slots: usize) -> W8Result<usize> { kda_slot_bytes(slots, 48 * 128 * 128 * 4) }
    pub fn conv_state_bytes(slots: usize) -> W8Result<usize> { kda_slot_bytes(slots, 3 * 3 * 6144 * 2) }
    pub const fn replay_bytes() -> usize { 64 * 48 * (3 * 128 * 4 + 4 + 3 * 128 * 2) }

    /// No allocation, synchronization or lazy loading on the launch path.
    pub fn launch(&self, weights: KimiKdaWeights, b: KimiKdaBatch,
        stream: *mut c_void) -> W8Result<()> {
        validate_kda_buffers(weights, b, self.rows, self.decode, self.scratch)?;
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
pub struct KimiKdaCommit<'a> { program: Program<'a> }
impl<'a> KimiKdaCommit<'a> {
    pub fn load(programs: &'a Programs<'a>) -> W8Result<Self> {
        let program = programs.program("kimik3_kda_commit", &["state", "conv_state", "replay", "tables"])?;
        validate_kda_commit_manifest(program.spec())?;
        Ok(Self { program })
    }

    pub fn launch(&self, sequences: usize, layers: usize, slots: usize,
        state: Buffer, conv_state: Buffer, replay: Buffer, tables: Buffer,
        stream: *mut c_void) -> W8Result<()> {
        let requirements = kda_commit_requirements(sequences, layers, slots, state, conv_state, replay, tables)?;
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

fn validate_kda_manifest(spec: &ProgramSpec, rows: usize, decode: bool) -> W8Result<usize> {
    let g = &spec.geometry;
    if rows != if decode { 64 } else { 4096 } || spec.capacity_rows as usize != rows
        || spec.scalar_kinds != if decode { &b"ii"[..] } else { &b"i"[..] }
        || [("hidden", 7168), ("heads", 48), ("head_dim", 128), ("max_rows", rows), ("in_width", 24752)]
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
    let base = align(rows * 24752 * 2) + align(rows * 2 * 6144 * 2)
        + align(rows * 3 * 6144 * 2) + 2 * align(rows * 6144 * 2);
    let scratch = spec.scratch.get("scratch").copied()
        .and_then(|n| usize::try_from(n).ok()).ok_or(KimiW8Error::Invalid("KDA scratch manifest"))?;
    if spec.scratch.len() != 1 || scratch < base || (decode && scratch != base)
        || (!decode && scratch == base) || scratch % 1024 != 0 {
        return Err(KimiW8Error::Invalid("KDA scratch manifest"));
    }
    Ok(scratch)
}

fn validate_kda_commit_manifest(spec: &ProgramSpec) -> W8Result<()> {
    let g = &spec.geometry;
    if spec.scalar_kinds != b"iii" || !spec.scratch.is_empty()
        || [("heads", 48), ("channels", 3 * 6144), ("replay_rows", 64), ("replay_bytes", KimiKda::replay_bytes())]
            .iter().any(|&(key, n)| g[key].as_u64() != Some(n as u64))
        || !g["state_dtype"].is_null() || !g["state_rounding"].is_null() {
        return Err(KimiW8Error::Invalid("KDA commit manifest"));
    }
    Ok(())
}

fn validate_kda_buffers(w: KimiKdaWeights, b: KimiKdaBatch, capacity: usize,
    decode: bool, scratch: usize) -> W8Result<()> {
    if b.rows == 0 || b.rows > capacity || b.rows > i32::MAX as usize
        || b.sequences == 0 || b.sequences > b.rows || b.sequences > b.state_slots
        || (!decode && (b.speculative || (b.rows > 64 && b.sequences != 1)))
        || b.replay.is_some() != decode {
        return Err(KimiW8Error::Invalid("KDA batch/replay geometry"));
    }
    let buffers = [(b.input, b.rows * 7168 * 2), (w.input, 24752 * 7168 * 2),
        (w.forget, 6144 * 128 * 2), (w.convolution, 3 * 6144 * 4 * 4), (w.a_log, 48 * 4),
        (w.dt_bias, 6144 * 4), (w.norm, 128 * 4), (w.output, 7168 * 6144 * 2),
        (b.conv_state, KimiKda::conv_state_bytes(b.state_slots)?), (b.state, KimiKda::state_bytes(b.state_slots)?),
        (b.slots, b.rows * 4), (b.first, b.rows * 4), (b.normalized, b.rows * 6144 * 2),
        (b.output, b.rows * 7168 * 2), (b.scratch, scratch),
        (b.replay.unwrap_or(b.scratch), KimiKda::replay_bytes())];
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

fn kda_commit_requirements(sequences: usize, layers: usize, slots: usize,
    state: Buffer, conv_state: Buffer, replay: Buffer, tables: Buffer) -> W8Result<[(Buffer, usize); 4]> {
    if sequences == 0 || sequences > slots || sequences > i32::MAX as usize || !(1..=69).contains(&layers) {
        return Err(KimiW8Error::Invalid("KDA commit dimensions"));
    }
    let layer_bytes = |bytes: usize| bytes.checked_mul(layers).ok_or(KimiW8Error::Invalid("KDA extent overflow"));
    Ok([(state, layer_bytes(KimiKda::state_bytes(slots)?)?),
        (conv_state, layer_bytes(KimiKda::conv_state_bytes(slots)?)?),
        (replay, layer_bytes(KimiKda::replay_bytes())?), (tables, sequences * 3 * 4)])
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
                "page_rows": 64, "norm_eps": 1e-6, "max_context": context,
                "num_splits": splits, "chunks_per_split": context.div_ceil(64).div_ceil(splits), "mode": mode.name()}),
        })
    }

    #[test]
    fn manifest_rejects_stale_geometry_traversal_and_scratch_before_launch() {
        for heads in [48, 96] {
            for mode in [KimiMlaMode::Decode, KimiMlaMode::Verify, KimiMlaMode::Extend] {
                let entries = specs(heads, 64, mode);
                assert!(validate_manifest(entries.each_ref(), heads, 64, mode).is_ok());
                for index in 0..3 {
                    for (key, bad) in [("heads", serde_json::json!(32)), ("hidden", serde_json::json!(4096)),
                        ("page_rows", serde_json::json!(128)), ("norm_eps", serde_json::json!(1e-5))] {
                        let mut wrong = entries.clone(); wrong[index].geometry[key] = bad;
                        assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                    }
                    let mut wrong = entries.clone(); wrong[index].scratch.clear();
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                    let mut wrong = entries.clone(); wrong[index].scalar_kinds = b"iii".to_vec();
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                    let mut wrong = entries.clone(); wrong[index].capacity_rows = 4096;
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                }
                for (key, bad) in [("max_context", serde_json::json!(0)), ("num_splits", serde_json::json!(0)),
                    ("chunks_per_split", serde_json::json!(1)), ("mode", serde_json::json!("wrong"))] {
                    let mut wrong = entries.clone(); wrong[1].geometry[key] = bad;
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
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
        validate_kda_commit_manifest(&spec).unwrap();
        let mut wrong = spec.clone(); wrong.geometry["replay_bytes"] = serde_json::json!(1);
        assert!(validate_kda_commit_manifest(&wrong).is_err());
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
        let requirements = kda_commit_requirements(2, 69, 4, b.state, b.conv_state, b.scratch, b.first).unwrap();
        assert_eq!(requirements[0].1, 69 * 4 * 3 * 1024 * 1024);
        assert!(kda_commit_requirements(2, 70, 4, b.state, b.conv_state, b.scratch, b.first).is_err());
        assert!(kda_commit_requirements(5, 69, 4, b.state, b.conv_state, b.scratch, b.first).is_err());
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
