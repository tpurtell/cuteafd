//! Checkpoint-native rotary MLA block drafting, independent of the target family.
//! Owners retain all storage and order cache/tap access on the supplied stream.
use crate::programs::{Program, ProgramSpec, Programs, Scalar};
use crate::CuteafdDeviceBuffer as Buffer;
use cuteafd_core::coordinator_programs::MlaDraftProgramLayout as Layout;
pub use cuteafd_core::coordinator_programs::MlaDraftKind;
use std::ffi::c_void;
use cuteafd_core::MlaCacheFormat;
use thiserror::Error;

const ROWS: usize = Layout::ROWS;
const HEADS: usize = Layout::HEADS;
const PAGE: usize = Layout::PAGE;
const PRODUCER: &[&str] = &["x", "kv_slots", "positions", "inv_freq", "w_qkv_a", "q_a_norm",
    "kv_a_norm", "w_q_b", "w_uk", "kv_cache", "query", "q_resid", "scratch"];
const ATTENTION: &[&str] = &["query", "kv_cache", "page_table", "cache_seqlens", "cu_seqlens_q",
    "attn", "lse", "scratch"];
const OUTPUT: &[&str] = &["attn", "w_uv", "w_o", "out", "scratch"];
/// Mistral Large 4's mtp_dspark producer (no q-LoRA; RoPE through a unit query scale).
const ML4_PRODUCER: &[&str] = &["x", "kv_slots", "positions", "cos_sin", "unit_s", "unit_b", "w_q", "w_kv_a",
    "kv_a_norm", "w_uk", "kv_cache", "query", "scratch"];

#[derive(Debug, Error)]
pub enum MlaDraftError {
    #[error("MLA draft invalid {0}")]
    Invalid(&'static str),
    #[error("MLA draft program: {0}")]
    Program(#[from] anyhow::Error),
}
type Result<T> = std::result::Result<T, MlaDraftError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaDraftWindow { Full, Sliding4096, BlockAnchored2048 }
impl MlaDraftWindow {
    fn suffix(self) -> &'static str {
        match self { Self::Full => "full", Self::Sliding4096 => "w4096", Self::BlockAnchored2048 => "w2048b" }
    }
    fn rows(self) -> Option<u64> {
        match self { Self::Full => None, Self::Sliding4096 => Some(4096), Self::BlockAnchored2048 => Some(2048) }
    }
}

/// BF16 low-rank operands. Unlike the target, draft output has no gate.
/// DFlash2 fills every field; mtp_dspark has no q-LoRA: `qkv_a` holds `wq`,
/// `q_b` holds `wkv_a_with_mqa`, `q_a_norm` the unit query-scale vectors
/// ([64] zeros then [64] ones, BF16).
#[derive(Clone, Copy, Debug)]
pub struct MlaDraftWeights {
    pub qkv_a: Buffer, pub q_a_norm: Buffer, pub kv_a_norm: Buffer,
    pub q_b: Buffer, pub uk: Buffer, pub uv: Buffer, pub output: Buffer,
}

/// Absolute I64 positions drive RoPE; I64 slots independently address physical
/// cache rows (-1 skips a write). I32 virtual page tables may repeat a physical
/// page at their ends. The owner validates metadata contents before staging.
#[derive(Clone, Copy, Debug)]
pub struct MlaDraftBatch {
    pub rows: usize, pub requests: usize, pub page_table_width: usize,
    pub cache_pages: usize, pub active_splits: usize,
    pub rotary_factor: f32, pub attention_scale: f32,
    pub x: Buffer, pub slots: Buffer, pub positions: Buffer, pub inv_freq: Buffer,
    pub cache: Buffer, pub page_table: Buffer, pub lengths: Buffer, pub query_offsets: Buffer,
    pub query: Buffer, pub q_resid: Buffer, pub attention: Buffer, pub lse: Buffer,
    pub output: Buffer, pub producer_scratch: Buffer, pub attention_scratch: Buffer,
    pub output_scratch: Buffer,
}

/// Native noncausal block attention with checkpoint-supplied YaRN scales.
/// Loading resolves every kernel before admission/capture, never on a request.
pub struct MlaDraft<'a> {
    producer: Program<'a>, attention: Program<'a>, output: Program<'a>,
    layout: Layout,
    cache_format: MlaCacheFormat,
    kind: MlaDraftKind,
}

impl<'a> MlaDraft<'a> {
    pub fn load(programs: &'a Programs<'a>, program_namespace: &str, window: MlaDraftWindow, cache_format: MlaCacheFormat) -> Result<Self> {
        let cache = cache_format.program_suffix();
        let producer = programs.program(&format!("{program_namespace}_mla_draft_producer{cache}_m64"), PRODUCER)?;
        let attention = programs.program(&format!("{program_namespace}_mla_draft_attention_{}{cache}_m64", window.suffix()), ATTENTION)?;
        let output = programs.program(&format!("{program_namespace}_mla_draft_output_m64"), OUTPUT)?;
        let layout = validate_manifest([producer.spec(), attention.spec(), output.spec()], window, cache_format)?;
        Ok(Self { producer, attention, output, layout, cache_format, kind: MlaDraftKind::Dflash2 })
    }

    /// Mistral Large 4's bundled mtp_dspark (`ml4_draft_*`): FP8 656-byte records,
    /// every block row windowed to the 2048 records before its block.
    pub fn load_ml4(programs: &'a Programs<'a>) -> Result<Self> {
        let producer = programs.program("ml4_draft_producer_m64", ML4_PRODUCER)?;
        let attention = programs.program("ml4_draft_attention_m64", ATTENTION)?;
        let output = programs.program("ml4_draft_output_m64", OUTPUT)?;
        let layout = validate_ml4_manifest([producer.spec(), attention.spec(), output.spec()])?;
        Ok(Self { producer, attention, output, layout, cache_format: MlaCacheFormat::Fp8, kind: MlaDraftKind::Ml4Dspark })
    }

    pub fn kind(&self) -> MlaDraftKind { self.kind }

    pub fn cache_format(&self) -> MlaCacheFormat { self.cache_format }

    pub fn scratch_bytes(&self) -> [usize; 3] { self.layout.scratch }
    pub fn max_context(&self) -> usize { self.layout.context }
    pub fn splits(&self) -> usize { self.layout.splits }

    /// Preflight an entire multi-layer/chunk operation before its first cache write.
    pub fn check_produce(&self, w: MlaDraftWeights, b: MlaDraftBatch) -> Result<()> {
        validate_producer_kind(self.kind, w, b, self.layout, self.cache_format)
    }

    pub fn check_launch(&self, w: MlaDraftWeights, b: MlaDraftBatch) -> Result<()> {
        validate_batch_kind(self.kind, w, b, self.layout, self.cache_format)
    }

    /// Context-only projection uses the same producer, without attention or
    /// output. The owner chunks taps to at most 64 rows and joins their producer.
    pub fn produce(&self, w: MlaDraftWeights, b: MlaDraftBatch, stream: *mut c_void) -> Result<()> {
        validate_producer_kind(self.kind, w, b, self.layout, self.cache_format)?;
        match self.kind {
            // SAFETY: host extents/devices/scalars are validated. The owner retains
            // storage and stream, and stages checked positions and physical slots.
            MlaDraftKind::Dflash2 => unsafe {
                self.producer.launch(&[b.x.ptr, b.slots.ptr, b.positions.ptr, b.inv_freq.ptr,
                    w.qkv_a.ptr, w.q_a_norm.ptr, w.kv_a_norm.ptr, w.q_b.ptr, w.uk.ptr,
                    b.cache.ptr, b.query.ptr, b.q_resid.ptr, b.producer_scratch.ptr],
                    &[Scalar::F32(b.rotary_factor), Scalar::I32(b.rows as i32)], stream)?;
            },
            // SAFETY: as above; `inv_freq` holds the FP32 cos|sin table [positions, 64]
            // and `q_a_norm` the unit scale vectors (zeros, then ones).
            MlaDraftKind::Ml4Dspark => unsafe {
                let unit_b = w.q_a_norm.ptr.cast::<u8>().add(HEADS * 2).cast();
                self.producer.launch(&[b.x.ptr, b.slots.ptr, b.positions.ptr, b.inv_freq.ptr,
                    w.q_a_norm.ptr, unit_b, w.qkv_a.ptr, w.q_b.ptr, w.kv_a_norm.ptr, w.uk.ptr,
                    b.cache.ptr, b.query.ptr, b.producer_scratch.ptr],
                    &[Scalar::I32(b.rows as i32)], stream)?;
            },
        }
        Ok(())
    }

    /// Validate the entire operation before the producer can mutate cache.
    /// Scratch may overlap across these strictly ordered phases, not live data.
    pub fn launch(&self, w: MlaDraftWeights, b: MlaDraftBatch, stream: *mut c_void) -> Result<()> {
        validate_batch_kind(self.kind, w, b, self.layout, self.cache_format)?;
        self.produce(w, b, stream)?;
        // SAFETY: the full batch was validated before any launch. Metadata,
        // buffers and stream remain owned and ordered through completion.
        unsafe {
            self.attention.launch(&[b.query.ptr, b.cache.ptr, b.page_table.ptr, b.lengths.ptr,
                b.query_offsets.ptr, b.attention.ptr, b.lse.ptr, b.attention_scratch.ptr],
                &[Scalar::I32(b.rows as i32), Scalar::I32(b.requests as i32),
                  Scalar::I64(b.page_table_width as i64), Scalar::I32(b.active_splits as i32),
                  Scalar::F32(b.attention_scale)], stream)?;
            self.output.launch(&[b.attention.ptr, w.uv.ptr, w.output.ptr, b.output.ptr,
                b.output_scratch.ptr], &[Scalar::I32(b.rows as i32)], stream)?;
        }
        Ok(())
    }
}

fn validate_manifest(specs: [&ProgramSpec; 3], window: MlaDraftWindow, cache_format: MlaCacheFormat) -> Result<Layout> {
    for (i, spec) in specs.iter().enumerate() {
        let g = &spec.geometry;
        let pointers = [PRODUCER, ATTENTION, OUTPUT][i];
        let scalars: &[u8] = [b"fi".as_slice(), b"iilif".as_slice(), b"i".as_slice()][i];
        if (i != 2 && g["kv_cache"].as_str() != Some(cache_format.label()))
            || spec.capacity_rows as usize != ROWS || spec.scalar_kinds != scalars
            || !spec.pointers.iter().map(String::as_str).eq(pointers.iter().copied())
            || [("heads", 64), ("hidden", 7168), ("q_lora_rank", 1536), ("kv_lora_rank", 512),
                ("qk_nope_dim", 128), ("qk_rope_dim", 64), ("v_head_dim", 128), ("page_rows", 64)]
                .iter().any(|&(key, n)| g[key].as_u64() != Some(n))
            || g["name"].as_str() != Some("mla_block_draft") || g["norm_eps"].as_f64() != Some(1e-5)
            || (i != 1 && !g["window_size"].is_null()) {
            return Err(MlaDraftError::Invalid("manifest geometry/pointers/scalars"));
        }
    }
    let p = &specs[0].geometry;
    let a = &specs[1].geometry;
    if p["rotary_style"].as_str() != Some("interleaved") || p["rotary_dim"].as_u64() != Some(64)
        || specs[2].geometry["output_gate"].as_bool() != Some(false)
        || a["noncausal_block"].as_bool() != Some(true)
        || match window.rows() { Some(n) => a["window_size"].as_u64() != Some(n),
            None => !a["window_size"].is_null() } {
        return Err(MlaDraftError::Invalid("manifest rotary/window/gate contract"));
    }
    let context = a["max_context"].as_u64().filter(|&n| n > 0 && n <= 1048576)
        .ok_or(MlaDraftError::Invalid("manifest context"))? as usize;
    let layout = Layout::new(context).map_err(|_| MlaDraftError::Invalid("manifest context"))?;
    if [("num_splits", layout.splits), ("chunks_per_split", layout.chunks_per_split),
        ("partial_output_bytes", layout.partial_output_bytes), ("partial_lse_bytes", layout.partial_lse_bytes),
        ("scratch_bytes", layout.scratch[1])]
        .iter().any(|&(key, n)| a[key].as_u64() != Some(n as u64))
        || specs.iter().zip(layout.scratch).any(|(s, bytes)| s.scratch.len() != 1
            || s.scratch.get("scratch").copied() != Some(bytes as u64)) {
        return Err(MlaDraftError::Invalid("manifest traversal/scratch"));
    }
    Ok(layout)
}

/// The ml4_draft_* manifest: Mistral Large 4's draft geometry (hidden 6144, no
/// q-LoRA, nope 64, v 128, FP8 656-byte records, block-anchored window 2048).
fn validate_ml4_manifest(specs: [&ProgramSpec; 3]) -> Result<Layout> {
    let scalars: [&[u8]; 3] = [b"i", b"iilif", b"i"];
    for (i, spec) in specs.iter().enumerate() {
        let g = &spec.geometry;
        let pointers = [ML4_PRODUCER, ATTENTION, OUTPUT][i];
        if spec.capacity_rows as usize != ROWS || spec.scalar_kinds != scalars[i]
            || !spec.pointers.iter().map(String::as_str).eq(pointers.iter().copied())
            || [("heads", 64), ("hidden", 6144), ("q_lora_rank", 0), ("kv_lora_rank", 512),
                ("qk_nope_dim", 64), ("qk_rope_dim", 64), ("v_head_dim", 128), ("page_rows", 64),
                ("window_size", 2048)].iter().any(|&(key, n)| g[key].as_u64() != Some(n))
            || g["name"].as_str() != Some("ml4_draft") || g["norm_eps"].as_f64() != Some(1e-5)
            || g["kv_cache"].as_str() != Some("fp8") {
            return Err(MlaDraftError::Invalid("ML4 draft manifest geometry/pointers/scalars"));
        }
    }
    let a = &specs[1].geometry;
    if a["noncausal_block"].as_bool() != Some(true) || a["block_window"].as_bool() != Some(true)
        || specs[2].geometry["output_gate"].as_bool() != Some(false) {
        return Err(MlaDraftError::Invalid("ML4 draft manifest window/gate contract"));
    }
    let context = a["max_context"].as_u64().filter(|&n| n > 0 && n <= 1048576)
        .ok_or(MlaDraftError::Invalid("manifest context"))? as usize;
    let layout = Layout::for_kind(MlaDraftKind::Ml4Dspark, context)
        .map_err(|_| MlaDraftError::Invalid("manifest context"))?;
    if [("num_splits", layout.splits), ("chunks_per_split", layout.chunks_per_split),
        ("partial_output_bytes", layout.partial_output_bytes), ("partial_lse_bytes", layout.partial_lse_bytes),
        ("scratch_bytes", layout.scratch[1])]
        .iter().any(|&(key, n)| a[key].as_u64() != Some(n as u64))
        || specs.iter().zip(layout.scratch).any(|(s, bytes)| s.scratch.len() != 1
            || s.scratch.get("scratch").copied() != Some(bytes as u64)) {
        return Err(MlaDraftError::Invalid("ML4 draft manifest traversal/scratch"));
    }
    Ok(layout)
}

fn ml4_producer_requirements(w: MlaDraftWeights, b: MlaDraftBatch, l: Layout) -> Result<Vec<(Buffer, usize)>> {
    if !(1..=ROWS).contains(&b.rows) || b.cache_pages == 0 || b.cache_pages > i32::MAX as usize {
        return Err(MlaDraftError::Invalid("producer geometry"));
    }
    let cache = b.cache_pages.checked_mul(PAGE * MlaCacheFormat::Fp8.record_bytes())
        .ok_or(MlaDraftError::Invalid("cache extent overflow"))?;
    let hidden = MlaDraftKind::Ml4Dspark.hidden();
    // The cos|sin table covers every position the owner stages (checked by the owner);
    // here at least one row.
    Ok(vec![(b.x, b.rows * hidden * 2), (b.slots, b.rows * 8), (b.positions, b.rows * 8),
        (b.inv_freq, 64 * 4), (w.q_a_norm, 2 * HEADS * 2), (w.qkv_a, HEADS * 128 * hidden * 2),
        (w.q_b, 576 * hidden * 2), (w.kv_a_norm, 512 * 2), (w.uk, HEADS * 512 * 64 * 2),
        (b.cache, cache), (b.query, b.rows * HEADS * 576 * 2), (b.producer_scratch, l.scratch[0])])
}

fn validate_producer_kind(kind: MlaDraftKind, w: MlaDraftWeights, b: MlaDraftBatch, l: Layout,
    cache_format: MlaCacheFormat) -> Result<()> {
    match kind {
        MlaDraftKind::Dflash2 => validate_producer(w, b, l, cache_format),
        MlaDraftKind::Ml4Dspark => validate_buffers(&ml4_producer_requirements(w, b, l)?, b.x.device_id),
    }
}

fn validate_batch_kind(kind: MlaDraftKind, w: MlaDraftWeights, b: MlaDraftBatch, l: Layout,
    cache_format: MlaCacheFormat) -> Result<()> {
    if kind == MlaDraftKind::Dflash2 { return validate_batch(w, b, l, cache_format); }
    if b.requests == 0 || b.requests > b.rows || b.page_table_width == 0
        || b.page_table_width > l.context.div_ceil(PAGE) + 1
        || b.active_splits != l.splits || !b.attention_scale.is_finite() || b.attention_scale <= 0.0 {
        return Err(MlaDraftError::Invalid("batch/context/split/attention scale"));
    }
    let hidden = MlaDraftKind::Ml4Dspark.hidden();
    let mut requirements = ml4_producer_requirements(w, b, l)?;
    let scratch = requirements.pop().ok_or(MlaDraftError::Invalid("producer scratch"))?;
    requirements.extend([(b.page_table, b.requests * b.page_table_width * 4),
        (b.lengths, b.requests * 4), (b.query_offsets, (b.requests + 1) * 4),
        (b.attention, b.rows * HEADS * 512 * 2), (b.lse, b.rows * HEADS * 4),
        (b.output, b.rows * hidden * 2), (w.uv, HEADS * 128 * 512 * 2),
        (w.output, hidden * HEADS * 128 * 2)]);
    validate_buffers(&requirements, b.x.device_id)?;
    for plane in [scratch, (b.attention_scratch, l.scratch[1]), (b.output_scratch, l.scratch[2])] {
        let mut phase = requirements.clone();
        phase.push(plane);
        validate_buffers(&phase, b.x.device_id)?;
    }
    Ok(())
}

fn producer_requirements(w: MlaDraftWeights, b: MlaDraftBatch, l: Layout, cache_format: MlaCacheFormat) -> Result<Vec<(Buffer, usize)>> {
    if !(1..=ROWS).contains(&b.rows) || b.cache_pages == 0 || b.cache_pages > i32::MAX as usize
        || !b.rotary_factor.is_finite() || b.rotary_factor <= 0.0 {
        return Err(MlaDraftError::Invalid("producer geometry/rotary scale"));
    }
    let cache = b.cache_pages.checked_mul(PAGE * cache_format.record_bytes())
        .ok_or(MlaDraftError::Invalid("cache extent overflow"))?;
    Ok(vec![(b.x, b.rows * 7168 * 2), (b.slots, b.rows * 8), (b.positions, b.rows * 8),
        (b.inv_freq, 32 * 4), (w.qkv_a, 2112 * 7168 * 2), (w.q_a_norm, 1536 * 2),
        (w.kv_a_norm, 512 * 2), (w.q_b, HEADS * 192 * 1536 * 2), (w.uk, HEADS * 512 * 128 * 2),
        (b.cache, cache), (b.query, b.rows * HEADS * 576 * 2), (b.q_resid, b.rows * 1536 * 2),
        (b.producer_scratch, l.scratch[0])])
}

fn validate_producer(w: MlaDraftWeights, b: MlaDraftBatch, l: Layout, cache_format: MlaCacheFormat) -> Result<()> {
    validate_buffers(&producer_requirements(w, b, l, cache_format)?, b.x.device_id)
}

fn validate_batch(w: MlaDraftWeights, b: MlaDraftBatch, l: Layout, cache_format: MlaCacheFormat) -> Result<()> {
    if b.requests == 0 || b.requests > b.rows || b.page_table_width == 0
        || b.page_table_width > l.context.div_ceil(PAGE) + 1
        || b.active_splits != l.splits || !b.attention_scale.is_finite() || b.attention_scale <= 0.0 {
        return Err(MlaDraftError::Invalid("batch/context/split/attention scale"));
    }
    let mut requirements = producer_requirements(w, b, l, cache_format)?;
    // Scratch planes may share a phase-disjoint allocation; remove them while
    // checking the cross-phase live set, then check each plane against that set.
    let scratch = requirements.pop().ok_or(MlaDraftError::Invalid("producer scratch"))?;
    requirements.extend([(b.page_table, b.requests * b.page_table_width * 4),
        (b.lengths, b.requests * 4), (b.query_offsets, (b.requests + 1) * 4),
        (b.attention, b.rows * HEADS * 512 * 2), (b.lse, b.rows * HEADS * 4),
        (b.output, b.rows * 7168 * 2), (w.uv, HEADS * 128 * 512 * 2),
        (w.output, 7168 * HEADS * 128 * 2)]);
    validate_buffers(&requirements, b.x.device_id)?;
    for plane in [scratch, (b.attention_scratch, l.scratch[1]), (b.output_scratch, l.scratch[2])] {
        let mut phase = requirements.clone();
        phase.push(plane);
        validate_buffers(&phase, b.x.device_id)?;
    }
    Ok(())
}

fn validate_buffers(requirements: &[(Buffer, usize)], device: i32) -> Result<()> {
    if device < 0 { return Err(MlaDraftError::Invalid("device")); }
    for (i, (buffer, bytes)) in requirements.iter().enumerate() {
        if buffer.ptr.is_null() || buffer.bytes < *bytes || buffer.device_id != device
            || buffer.ptr as usize % 16 != 0 {
            return Err(MlaDraftError::Invalid("buffer extent/alignment/device"));
        }
        let start = buffer.ptr as usize;
        let end = start.checked_add(*bytes).ok_or(MlaDraftError::Invalid("buffer address overflow"))?;
        for (other, len) in &requirements[..i] {
            let other_start = other.ptr as usize;
            let other_end = other_start.checked_add(*len).ok_or(MlaDraftError::Invalid("buffer address overflow"))?;
            if *bytes > 0 && *len > 0 && start < other_end && other_start < end {
                return Err(MlaDraftError::Invalid("overlapping live buffers"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs(window: MlaDraftWindow, context: usize) -> [ProgramSpec; 3] {
        let splits = context.div_ceil(PAGE).min(16);
        let align = |n: usize| n.div_ceil(1024) * 1024;
        let partial = if splits > 1 { align(ROWS * HEADS * splits * 512 * 2) } else { 0 };
        let lse = if splits > 1 { align(ROWS * HEADS * splits * 4) } else { 0 };
        let scratch = [align(ROWS * 2112 * 2) + align(ROWS * HEADS * 192 * 2),
            partial + lse, align(ROWS * HEADS * 128 * 2)];
        std::array::from_fn(|i| {
            let mut geometry = serde_json::json!({"name": "mla_block_draft", "heads": 64, "hidden": 7168,
                "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_dim": 128, "qk_rope_dim": 64,
                "v_head_dim": 128, "page_rows": 64, "norm_eps": 1e-5, "window_size": null, "kv_cache": "bf16"});
            match i {
                0 => { geometry["rotary_style"] = "interleaved".into(); geometry["rotary_dim"] = 64.into(); }
                1 => {
                    for (key, value) in [("max_context", context), ("num_splits", splits),
                        ("chunks_per_split", context.div_ceil(PAGE).div_ceil(splits)),
                        ("partial_output_bytes", partial), ("partial_lse_bytes", lse), ("scratch_bytes", scratch[1])] {
                        geometry[key] = value.into();
                    }
                    geometry["noncausal_block"] = true.into();
                    geometry["window_size"] = window.rows().into();
                }
                _ => geometry["output_gate"] = false.into(),
            }
            ProgramSpec { name: format!("test{i}"), capacity_rows: ROWS as u32, geometry,
                pointers: [PRODUCER, ATTENTION, OUTPUT][i].iter().map(|s| s.to_string()).collect(),
                scalar_kinds: [b"fi".as_slice(), b"iilif".as_slice(), b"i".as_slice()][i].to_vec(),
                scratch: [("scratch".into(), scratch[i] as u64)].into_iter().collect() }
        })
    }

    #[test]
    fn compact_draft_manifest_rejects_mixed_records() {
        for window in [MlaDraftWindow::Full, MlaDraftWindow::Sliding4096] {
            let mut compact = specs(window, 524288);
            for spec in &mut compact[..2] { spec.geometry["kv_cache"] = "fp8".into(); }
            assert!(validate_manifest(compact.each_ref(), window, MlaCacheFormat::Fp8).is_ok());
            for index in 0..2 {
                let mut wrong = compact.clone();
                wrong[index].geometry["kv_cache"] = "bf16".into();
                assert!(validate_manifest(wrong.each_ref(), window, MlaCacheFormat::Fp8).is_err());
            }
        }
    }

    #[test]
    fn native_draft_manifest_rejects_target_geometry_and_stale_abi() {
        for window in [MlaDraftWindow::Full, MlaDraftWindow::Sliding4096] {
            for context in [64, 262144, 1048576] {
                let good = specs(window, context);
                let layout = validate_manifest(good.each_ref(), window, MlaCacheFormat::Bf16).unwrap();
                assert_eq!(layout.context, context);
                assert_eq!(layout.scratch[2], 1048576, "ungated output has one value plane");
                for i in 0..3 {
                    for (key, bad) in [("heads", serde_json::json!(96)), ("hidden", 3584.into()),
                        ("norm_eps", serde_json::json!(1e-6)), ("page_rows", 128.into())] {
                        let mut wrong = good.clone(); wrong[i].geometry[key] = bad;
                        assert!(validate_manifest(wrong.each_ref(), window, MlaCacheFormat::Bf16).is_err());
                    }
                    let mut wrong = good.clone(); wrong[i].pointers.swap(0, 1);
                    assert!(validate_manifest(wrong.each_ref(), window, MlaCacheFormat::Bf16).is_err());
                    let mut wrong = good.clone(); wrong[i].scalar_kinds.push(b'i');
                    assert!(validate_manifest(wrong.each_ref(), window, MlaCacheFormat::Bf16).is_err());
                    let mut wrong = good.clone(); wrong[i].capacity_rows = 4096;
                    assert!(validate_manifest(wrong.each_ref(), window, MlaCacheFormat::Bf16).is_err());
                    let mut wrong = good.clone(); *wrong[i].scratch.get_mut("scratch").unwrap() += 1024;
                    assert!(validate_manifest(wrong.each_ref(), window, MlaCacheFormat::Bf16).is_err());
                }
                for (i, key, bad) in [(0, "rotary_style", "neox".into()), (0, "rotary_dim", 32.into()),
                    (1, "noncausal_block", false.into()), (1, "max_context", 0.into()),
                    (1, "num_splits", 0.into()), (1, "chunks_per_split", 0.into()),
                    (1, "window_size", 2048.into()), (2, "output_gate", true.into())] {
                    let mut wrong = good.clone(); wrong[i].geometry[key] = bad;
                    assert!(validate_manifest(wrong.each_ref(), window, MlaCacheFormat::Bf16).is_err());
                }
            }
        }
    }

    fn ml4_specs(context: usize) -> [ProgramSpec; 3] {
        let layout = Layout::for_kind(MlaDraftKind::Ml4Dspark, context).unwrap();
        std::array::from_fn(|i| {
            let mut geometry = serde_json::json!({"name": "ml4_draft", "heads": 64, "hidden": 6144,
                "q_lora_rank": 0, "kv_lora_rank": 512, "qk_nope_dim": 64, "qk_rope_dim": 64,
                "v_head_dim": 128, "page_rows": 64, "norm_eps": 1e-5, "window_size": 2048, "kv_cache": "fp8"});
            match i {
                0 => geometry["max_rows"] = 64.into(),
                1 => {
                    for (key, value) in [("max_context", context), ("num_splits", layout.splits),
                        ("chunks_per_split", layout.chunks_per_split),
                        ("partial_output_bytes", layout.partial_output_bytes),
                        ("partial_lse_bytes", layout.partial_lse_bytes), ("scratch_bytes", layout.scratch[1])] {
                        geometry[key] = value.into();
                    }
                    geometry["noncausal_block"] = true.into();
                    geometry["block_window"] = true.into();
                }
                _ => geometry["output_gate"] = false.into(),
            }
            ProgramSpec { name: format!("ml4{i}"), capacity_rows: ROWS as u32, geometry,
                pointers: [ML4_PRODUCER, ATTENTION, OUTPUT][i].iter().map(|s| s.to_string()).collect(),
                scalar_kinds: [b"i".as_slice(), b"iilif".as_slice(), b"i".as_slice()][i].to_vec(),
                scratch: [("scratch".into(), layout.scratch[i] as u64)].into_iter().collect() }
        })
    }

    #[test]
    fn ml4_draft_manifest_is_its_own_contract_and_dflash2_is_unchanged() {
        let good = ml4_specs(1048576);
        let layout = validate_ml4_manifest(good.each_ref()).unwrap();
        // mtp_dspark producer: [64, 64, 128] q then [64, 576] kv_a; the other planes are DFlash2's.
        assert_eq!(layout.scratch[0], 64 * 64 * 128 * 2 + 73728);
        assert_eq!(layout.scratch[1..], Layout::new(1048576).unwrap().scratch[1..]);
        assert_eq!(Layout::new(1048576).unwrap(), Layout::for_kind(MlaDraftKind::Dflash2, 1048576).unwrap());
        for (i, key, bad) in [(0, "hidden", serde_json::json!(7168)), (1, "block_window", false.into()),
            (1, "window_size", 4096.into()), (0, "q_lora_rank", 1536.into()), (2, "kv_cache", "bf16".into()),
            (1, "noncausal_block", false.into()), (2, "output_gate", true.into())] {
            let mut wrong = good.clone(); wrong[i].geometry[key] = bad;
            assert!(validate_ml4_manifest(wrong.each_ref()).is_err(), "{key}");
        }
        let mut wrong = good.clone(); wrong[0].pointers.swap(4, 5);
        assert!(validate_ml4_manifest(wrong.each_ref()).is_err());
        // Neither validator admits the other's programs.
        assert!(validate_manifest(good.each_ref(), MlaDraftWindow::BlockAnchored2048, MlaCacheFormat::Fp8).is_err());
        let kimi = specs(MlaDraftWindow::Sliding4096, 1048576);
        assert!(validate_ml4_manifest(kimi.each_ref()).is_err());
    }

    fn fixture() -> (MlaDraftWeights, MlaDraftBatch, Layout) {
        let l = validate_manifest(specs(MlaDraftWindow::Sliding4096, 1048576).each_ref(), MlaDraftWindow::Sliding4096, MlaCacheFormat::Bf16).unwrap();
        let mut next = 0x100000usize;
        let mut alloc = || { next += 0x10000000; Buffer { ptr: next as *mut c_void, bytes: 0x10000000, device_id: 1, flags: 0 } };
        let w = MlaDraftWeights { qkv_a: alloc(), q_a_norm: alloc(), kv_a_norm: alloc(), q_b: alloc(),
            uk: alloc(), uv: alloc(), output: alloc() };
        let b = MlaDraftBatch { rows: 64, requests: 8, page_table_width: 66, cache_pages: 520,
            active_splits: l.splits, rotary_factor: 1.0, attention_scale: 0.2,
            x: alloc(), slots: alloc(), positions: alloc(), inv_freq: alloc(), cache: alloc(),
            page_table: alloc(), lengths: alloc(), query_offsets: alloc(), query: alloc(), q_resid: alloc(),
            attention: alloc(), lse: alloc(), output: alloc(), producer_scratch: alloc(),
            attention_scratch: alloc(), output_scratch: alloc() };
        (w, b, l)
    }

    #[test]
    fn batch_checks_all_extents_devices_aliases_and_runtime_scales_before_mutation() {
        let (w, good, l) = fixture();
        validate_batch(w, good, l, MlaCacheFormat::Bf16).unwrap();
        let mut shared = good; shared.attention_scratch = shared.producer_scratch; shared.output_scratch = shared.producer_scratch;
        validate_batch(w, shared, l, MlaCacheFormat::Bf16).unwrap();
        let fields: [fn(&mut MlaDraftBatch) -> &mut Buffer; 16] = [|b| &mut b.x, |b| &mut b.slots,
            |b| &mut b.positions, |b| &mut b.inv_freq, |b| &mut b.cache, |b| &mut b.page_table,
            |b| &mut b.lengths, |b| &mut b.query_offsets, |b| &mut b.query, |b| &mut b.q_resid,
            |b| &mut b.attention, |b| &mut b.lse, |b| &mut b.output, |b| &mut b.producer_scratch,
            |b| &mut b.attention_scratch, |b| &mut b.output_scratch];
        for field in fields {
            for defect in 0..4 {
                let mut bad = good;
                match defect { 0 => field(&mut bad).bytes = 0, 1 => field(&mut bad).device_id = 0,
                    2 => field(&mut bad).ptr = 1usize as *mut c_void,
                    _ => field(&mut bad).ptr = std::ptr::null_mut() }
                assert!(validate_batch(w, bad, l, MlaCacheFormat::Bf16).is_err());
            }
        }
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut bad = good; bad.rotary_factor = scale;
            assert!(validate_batch(w, bad, l, MlaCacheFormat::Bf16).is_err());
            let mut bad = good; bad.attention_scale = scale;
            assert!(validate_batch(w, bad, l, MlaCacheFormat::Bf16).is_err());
        }
        for (rows, requests, pages, width, splits) in [(0, 1, 1, 1, 16), (65, 1, 1, 1, 16),
            (8, 9, 1, 1, 16), (8, 0, 1, 1, 16), (8, 1, 0, 1, 16),
            (8, 1, usize::MAX, 1, 16), (8, 1, 1, usize::MAX, 16), (8, 1, 1, 66, 1)] {
            let mut bad = good; bad.rows = rows; bad.requests = requests; bad.cache_pages = pages;
            bad.page_table_width = width; bad.active_splits = splits;
            assert!(validate_batch(w, bad, l, MlaCacheFormat::Bf16).is_err());
        }
        let mut bad = good; bad.query = bad.cache;
        assert!(validate_batch(w, bad, l, MlaCacheFormat::Bf16).is_err());
        let mut bad = good; bad.attention_scratch = bad.query;
        assert!(validate_batch(w, bad, l, MlaCacheFormat::Bf16).is_err());
        let mut bad = good; bad.positions.ptr = (usize::MAX - 15) as *mut c_void;
        assert!(validate_batch(w, bad, l, MlaCacheFormat::Bf16).is_err());
        let mut bad_w = w; bad_w.q_b.bytes = 0;
        assert!(validate_batch(bad_w, good, l, MlaCacheFormat::Bf16).is_err());
        // Context-only projection does not require attention/output operands.
        let mut context = good; context.page_table.ptr = std::ptr::null_mut();
        validate_producer(w, context, l, MlaCacheFormat::Bf16).unwrap();
    }
}
