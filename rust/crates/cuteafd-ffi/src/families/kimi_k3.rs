//! Kimi NoPE MLA programs, resolved once before graph capture. The engine owns
//! weights, page tables, cache, outputs and scratch through stream completion.
use crate::programs::{Program, Programs, Scalar};
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
        if !matches!(heads, 48 | 96) || rows == 0 || rows > i32::MAX as usize {
            return Err(KimiMlaError::Invalid("head/row geometry"));
        }
        let suffix = if heads == 48 { "" } else { "_h96" };
        let producer = programs.program(&format!("kimik3_mla_producer{suffix}_m{rows}"), PRODUCER)?;
        let attention = programs.program(&format!("kimik3_dense_mla_{}{suffix}_m{rows}", mode.name()), ATTENTION)?;
        let output = programs.program(&format!("kimik3_mla_output{suffix}_m{rows}"), OUTPUT)?;
        for p in [&producer, &attention, &output] {
            if p.spec().capacity_rows as usize != rows || p.spec().geometry["heads"].as_u64() != Some(heads as u64) {
                return Err(KimiMlaError::Invalid("manifest geometry/capacity"));
            }
        }
        let context = attention.spec().geometry["max_context"].as_u64()
            .filter(|&n| n > 0 && n <= 1048576).ok_or(KimiMlaError::Invalid("manifest context"))? as usize;
        let splits = attention.spec().geometry["num_splits"].as_u64()
            .filter(|&n| n > 0 && n <= 16).ok_or(KimiMlaError::Invalid("manifest split count"))? as usize;
        let scratch = [&producer, &attention, &output].map(|p| p.spec().scratch.get("scratch")
            .copied().and_then(|n| usize::try_from(n).ok()).ok_or(KimiMlaError::Invalid("manifest scratch")));
        let scratch = [scratch[0].as_ref().map_err(|_| KimiMlaError::Invalid("producer scratch"))?,
                       scratch[1].as_ref().map_err(|_| KimiMlaError::Invalid("attention scratch"))?,
                       scratch[2].as_ref().map_err(|_| KimiMlaError::Invalid("output scratch"))?].map(|n| *n);
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

#[cfg(test)]
mod tests {
    use super::*;
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
