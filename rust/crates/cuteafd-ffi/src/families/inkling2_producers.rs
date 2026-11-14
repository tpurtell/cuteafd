//! Family arithmetic producers; storage, streams and scheduling stay shared.
use crate::{programs::{Program, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure, Result};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision { Bf16, F32 }
impl Precision {
    fn suffix(self) -> &'static str { match self { Self::Bf16 => "bf16", Self::F32 => "f32" } }
    pub fn bytes(self) -> usize { match self { Self::Bf16 => 2, Self::F32 => 4 } }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Convolution { width: usize, dynamic: bool, precision: Precision },
    Clip { relative: bool, key: bool, global: bool, precision: Precision },
    Residual { queries: usize, addend: bool },
    Router,
    Gate { situ: bool, precision: Precision },
    Norm { value: bool, precision: Precision },
    KvPack { precision: Precision },
}
impl Op {
    pub fn name(self) -> Result<String> {
        Ok(format!("inkling2_{}", match self {
            Self::Convolution { width, dynamic, precision } => {
                ensure!(matches!((width, dynamic), (4608, false) | (1024, false) | (1024, true)),
                    "Inkling convolution geometry unsupported");
                format!("conv_{}_h{width}_{}", if dynamic { "dynamic" } else { "static" }, precision.suffix())
            }
            Self::Clip { relative, key, global, precision } => {
                ensure!(!(relative && key) && !(key && global), "Inkling clip site unsupported");
                format!("clip_{}_{}_{}", if global { "global" } else { "local" },
                    if key { "k" } else if relative { "r" } else { "q" }, precision.suffix())
            }
            Self::Residual { queries, addend } => {
                ensure!(matches!(queries, 1 | 3), "Inkling residual query count unsupported");
                format!("residual_q{queries}_{}", if addend { "add" } else { "keep" })
            }
            Self::Router => "router_e384_k8".into(),
            Self::Gate { situ, precision } => format!("{}_h{}_{}", if situ { "situ" } else { "gate" },
                if situ { 2304 } else { 6144 }, precision.suffix()),
            Self::Norm { value, precision } => format!("norm_{}_{}", if value { "v_h8_d128" } else { "h4608" }, precision.suffix()),
            Self::KvPack { precision } => format!("kv_pack_{}", precision.suffix()),
        }))
    }
    fn pointers(self) -> &'static [&'static str] {
        match self {
            Self::Convolution { .. } => &["x", "weight", "dynamic", "history", "first", "slots", "out"],
            Self::Clip { .. } => &["x", "scale", "positions", "out"],
            Self::Residual { .. } => &["arena", "addend", "query", "out"],
            Self::Router => &["logits", "bias", "ids", "weights"],
            Self::Gate { .. } => &["x", "gate", "gamma", "out"],
            Self::Norm { .. } => &["x", "weight", "out"],
            Self::KvPack { .. } => &["k", "v", "slots", "cache"],
        }
    }
}

pub struct Producer<'a> { program: Program<'a>, op: Op, rows: usize }
impl<'a> Producer<'a> {
    pub fn load(programs: &'a Programs<'a>, op: Op) -> Result<Self> {
        let program = programs.program(&op.name()?, op.pointers())?;
        let spec = program.spec();
        let kinds: &[u8] = if matches!(op, Op::Residual { .. }) { b"iii" } else { b"i" };
        ensure!(spec.capacity_rows > 0 && spec.capacity_rows <= 4096 && spec.scalar_kinds == kinds
            && spec.scratch.is_empty(), "Inkling producer export capacity/scalars/scratch differs");
        let g = &spec.geometry;
        let equal = |name: &str, expected: serde_json::Value| -> Result<()> {
            validate_geometry_value(&g[name], &expected, name)
        };
        match op {
            Op::Convolution { width, dynamic, .. } => {
                equal("width", width.into())?; equal("taps", 4.into())?;
                equal("heads", if dynamic { 8.into() } else { 1.into() })?;
                equal("dynamic", dynamic.into())?; equal("history", "newest_first".into())?;
            }
            Op::Clip { relative, key, global, .. } => {
                equal("heads", if key { 8.into() } else { 48.into() })?;
                equal("dim", if relative { 16.into() } else { 128.into() })?;
                equal("scaled", (!key).into())?;
                let threshold = if relative { 8. } else { 128f64.sqrt()*2. };
                equal("threshold", threshold.into())?;
                equal("log_floor", if global { 128000.into() } else { 0.into() })?;
                equal("log_alpha", if global { 0.1.into() } else { 0.0.into() })?;
                equal("eps", 1e-6.into())?;
            }
            Op::Residual { queries, addend } => {
                equal("hidden", 4608.into())?; equal("queries", queries.into())?;
                equal("max_candidates", 9.into())?; equal("normalize_values", true.into())?;
                equal("addend", addend.into())?; equal("eps", 1e-6.into())?;
            }
            Op::Router => {
                equal("experts", 384.into())?; equal("topk", 8.into())?; equal("route_scale", 8.0.into())?;
            }
            Op::Gate { situ, .. } => {
                equal("width", if situ { 2304.into() } else { 6144.into() })?;
                equal("situ", situ.into())?; equal("gate_cap", 4.0.into())?; equal("up_cap", 25.0.into())?;
            }
            Op::Norm { value, precision } => {
                equal("width", if value { 128.into() } else { 4608.into() })?;
                equal("heads", if value { 8.into() } else { 1.into() })?;
                equal("weighted", (!value).into())?; equal("eps", 1e-6.into())?;
                equal("output_dtype", match precision { Precision::Bf16 => "torch.bfloat16", Precision::F32 => "torch.float32" }.into())?;
            }
            Op::KvPack { precision } => {
                equal("heads", 8.into())?; equal("dim", 128.into())?;
                equal("input_dtype", match precision { Precision::Bf16 => "torch.bfloat16", Precision::F32 => "torch.float32" }.into())?;
            }
        }
        let rows = spec.capacity_rows as usize;
        Ok(Self { program, op, rows })
    }
    /// # Safety
    /// Buffers in the named ABI order have the declared typed extents on the
    /// stream's device. Writers do not alias readers. Stateful operations have
    /// valid sequence/slot metadata. Owners and stream survive queued work.
    /// Optional unread pointers may alias a valid supplied dummy allocation.
    pub unsafe fn launch(&self, buffers: &[Buffer], rows: usize, residual: Option<(usize, usize)>,
        stream: *mut c_void) -> Result<()> {
        ensure!(rows > 0 && rows <= self.rows && buffers.len() == self.op.pointers().len()
            && buffers.iter().all(|b| !b.ptr.is_null() && b.device_id == buffers[0].device_id && b.flags == 0),
            "Inkling producer live rows/device/pointer count differs");
        let mut scalars = vec![Scalar::I32(rows as i32)];
        match (self.op, residual) {
            (Op::Residual { .. }, Some((arena_rows, candidates))) => {
                ensure!(arena_rows >= rows && arena_rows <= i32::MAX as usize && (1..=9).contains(&candidates),
                    "Inkling residual arena/candidates differ");
                scalars.extend([Scalar::I32(arena_rows as i32), Scalar::I32(candidates as i32)]);
            }
            (Op::Residual { .. }, None) | (_, Some(_)) => anyhow::bail!("Inkling producer residual scalar ABI differs"),
            (_, None) => {}
        }
        let pointers = buffers.iter().map(|buffer| buffer.ptr).collect::<Vec<_>>();
        // SAFETY: caller supplies typed extents/metadata/lifetimes. ABI and live
        // counts are checked above; program load completed before this call.
        unsafe { self.program.launch(&pointers, &scalars, stream)?; }
        Ok(())
    }
}

fn validate_geometry_value(actual: &serde_json::Value, expected: &serde_json::Value, name: &str) -> Result<()> {
    // The manifest serializes Python F64 constants, but every kernel scalar is
    // F32. serde_json without float_roundtrip can move the last F64 bit; require
    // identical effective F32 bits, not an unrelated decimal tolerance.
    let same = if expected.is_f64() {
        actual.as_f64().zip(expected.as_f64()).is_some_and(|(a, b)|
            a.is_finite() && b.is_finite() && (a as f32).to_bits() == (b as f32).to_bits())
    } else { actual == expected };
    ensure!(same, "Inkling producer geometry {name} differs: actual={actual}, expected={expected}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serialized_clip_constants_require_identical_kernel_f32_bits() {
        let parsed: serde_json::Value = serde_json::from_str("22.627416997969522").unwrap();
        validate_geometry_value(&parsed, &serde_json::json!(128f64.sqrt()*2.), "threshold").unwrap();
        assert!(validate_geometry_value(&serde_json::json!(22.63), &parsed, "threshold").is_err());
        assert!(validate_geometry_value(&serde_json::json!("22.627416997969522"), &parsed, "threshold").is_err());
        assert!(validate_geometry_value(&serde_json::json!(48.0), &serde_json::json!(48), "heads").is_err());
    }
}
