//! Inkling 2 Mini TP1 arithmetic producers and looped cross-pass controller kernels.
use crate::{inkling2::producers::Precision, programs::{Program, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure, Result};
use std::ffi::c_void;

pub use crate::inkling2::producers::Precision as MiniPrecision;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Convolution { width: usize, dynamic: bool, precision: Precision },
    Clip { relative: bool, key: bool, global: bool, precision: Precision },
    Residual { queries: usize, max_candidates: usize, addend: bool },
    LoopPairNorm { slots: usize, broadcast_dest: bool },
    LoopBlendNorm { slots: usize, broadcast_dest: bool, broadcast_weight: bool },
    AttnInProjLocal,
    AttnInProjGlobal,
    RouterProj,
    Router,
    Gate { precision: Precision },
    Norm { value: bool, precision: Precision },
    KvPack { precision: Precision, fp8: bool },
}

fn precision_suffix(p: Precision) -> &'static str {
    match p { Precision::Bf16 => "bf16", Precision::F32 => "f32" }
}

impl Op {
    pub fn name(self) -> Result<String> {
        Ok(format!("inkling2_mini_{}", match self {
            Self::Convolution { width, dynamic, precision } => {
                ensure!(matches!((width, dynamic), (2304, false) | (1024, false) | (1024, true)),
                    "Inkling 2 Mini convolution geometry unsupported");
                format!("conv_{}_h{width}_{}_v2", if dynamic { "dynamic" } else { "static" }, precision_suffix(precision))
            }
            Self::Clip { relative, key, global, precision } => {
                ensure!(!(relative && key) && !(key && global), "Inkling 2 Mini clip site unsupported");
                format!("clip_{}_{}_{}", if global { "global" } else { "local" },
                    if key { "k" } else if relative { "r" } else { "q" }, precision_suffix(precision))
            }
            Self::Residual { queries, max_candidates, addend } => {
                ensure!(matches!((queries, max_candidates), (1, 5) | (3, 5) | (4, 2)),
                    "Inkling 2 Mini residual geometry unsupported");
                format!("residual_q{queries}_c{max_candidates}_{}", if addend { "add" } else { "keep" })
            }
            Self::LoopPairNorm { slots, broadcast_dest } => {
                ensure!(matches!((slots, broadcast_dest), (1, false) | (3, true) | (3, false)),
                    "Inkling 2 Mini loop pair_norm geometry unsupported");
                format!("loop_pair_norm_s{slots}{}", if broadcast_dest { "_bcast" } else { "" })
            }
            Self::LoopBlendNorm { slots, broadcast_dest, broadcast_weight } => {
                ensure!(matches!((slots, broadcast_dest, broadcast_weight),
                    (1, false, false) | (3, true, true) | (3, false, false)),
                    "Inkling 2 Mini loop blend_norm geometry unsupported");
                format!("loop_blend_norm_s{slots}{}", if broadcast_dest { "_bcast" } else { "" })
            }
            Self::AttnInProjLocal => "attn_in_proj_local".into(),
            Self::AttnInProjGlobal => "attn_in_proj_global".into(),
            Self::RouterProj => "router_proj".into(),
            Self::Router => "router_e128_k8".into(),
            Self::Gate { precision } => format!("gate_h3072_{}", precision_suffix(precision)),
            Self::Norm { value, precision } => if value {
                format!("norm_v_h8_d128_{}", precision_suffix(precision))
            } else {
                format!("norm_h2304_{}", precision_suffix(precision))
            },
            Self::KvPack { precision, fp8 } => format!("kv_pack_{}_{}_v2", if fp8 { "fp8" } else { "bf16" }, precision_suffix(precision)),
        }))
    }

    fn pointers(self) -> &'static [&'static str] {
        match self {
            Self::Convolution { .. } => &["x", "weight", "dynamic", "history", "first", "slots", "out"],
            Self::Clip { .. } => &["x", "scale", "positions", "out"],
            Self::Residual { .. } => &["arena", "addend", "query", "out"],
            Self::LoopPairNorm { .. } => &["dest", "msg", "out"],
            Self::LoopBlendNorm { .. } => &["dest", "msg", "forget_gain", "weight", "out"],
            Self::AttnInProjLocal => &[
                "xq", "xk", "xv",
                "wq", "wr", "wgate", "wsink", "wk", "wv",
                "out_q", "out_r", "out_gate", "out_sink", "out_k", "out_v",
            ],
            Self::AttnInProjGlobal => &[
                "xq", "xk", "xv",
                "wq", "wr", "wgate", "wsink", "wk", "wv", "wk_dyn", "wv_dyn",
                "out_q", "out_r", "out_gate", "out_sink", "out_k", "out_v", "out_k_dyn", "out_v_dyn",
            ],
            Self::RouterProj => &["x", "w", "out"],
            Self::Router => &["logits", "bias", "ids", "weights"],
            Self::Gate { .. } => &["x", "gate", "gamma", "out"],
            Self::Norm { .. } => &["x", "weight", "out"],
            Self::KvPack { .. } => &["k", "v", "slots", "cache"],
        }
    }
}

pub struct Producer<'a> {
    program: Program<'a>,
    op: Op,
    rows: usize,
}

impl<'a> Producer<'a> {
    pub fn load(programs: &'a Programs<'a>, op: Op) -> Result<Self> {
        let program = programs.program(&op.name()?, op.pointers())?;
        let spec = program.spec();
        let kinds: &[u8] = if matches!(op, Op::Residual { .. }) { b"iii" }
            else if matches!(op, Op::Convolution { .. }) { b"il" }
            else { b"i" };
        ensure!(spec.capacity_rows > 0 && spec.capacity_rows <= 4096 && spec.scalar_kinds == kinds
            && spec.scratch.is_empty(), "Inkling 2 Mini producer export capacity/scalars/scratch differs");
        let g = &spec.geometry;
        let equal = |name: &str, expected: serde_json::Value| -> Result<()> {
            ensure!(g[name] == expected,
                "Inkling 2 Mini producer geometry {name} differs: actual={}, expected={expected}", g[name]);
            Ok(())
        };
        match op {
            Op::Convolution { width, dynamic, .. } => {
                equal("cache_abi", 2.into())?;
                equal("width", width.into())?; equal("taps", 4.into())?;
                equal("heads", if dynamic { (width / 128).into() } else { 1.into() })?;
                equal("dynamic", dynamic.into())?; equal("history", "newest_first".into())?;
            }
            Op::Clip { relative, key, global, .. } => {
                equal("heads", (if key { 8 } else { 24 }).into())?;
                equal("dim", if relative { 16.into() } else { 128.into() })?;
                equal("scaled", (!key).into())?;
                let threshold = if relative { 8. } else { 128f64.sqrt() * 2. };
                equal("threshold", threshold.into())?;
                equal("log_floor", if global { 8192.into() } else { 0.into() })?;
                equal("log_alpha", if global { 0.1.into() } else { 0.0.into() })?;
                equal("eps", 1e-6.into())?;
            }
            Op::Residual { queries, max_candidates, addend } => {
                equal("hidden", 2304.into())?; equal("queries", queries.into())?;
                equal("max_candidates", max_candidates.into())?;
                equal("normalize_values", true.into())?;
                equal("addend", addend.into())?; equal("eps", 1e-6.into())?;
            }
            Op::LoopPairNorm { slots, broadcast_dest } => {
                equal("hidden", 2304.into())?; equal("slots", slots.into())?;
                equal("broadcast_dest", broadcast_dest.into())?; equal("eps", 1e-6.into())?;
            }
            Op::LoopBlendNorm { slots, broadcast_dest, broadcast_weight } => {
                equal("hidden", 2304.into())?; equal("slots", slots.into())?;
                equal("broadcast_dest", broadcast_dest.into())?;
                equal("broadcast_weight", broadcast_weight.into())?;
                equal("eps", 1e-6.into())?;
            }
            Op::AttnInProjLocal => {
                equal("width", 2304.into())?;
                equal("small_row_max", 8.into())?;
                equal("global_dynamic", false.into())?;
                equal("outputs", serde_json::json!([3072, 384, 3072, 24, 1024, 1024]))?;
                equal("reduction", "ordered_k16".into())?;
            }
            Op::AttnInProjGlobal => {
                equal("width", 2304.into())?;
                equal("small_row_max", 8.into())?;
                equal("global_dynamic", true.into())?;
                equal("outputs", serde_json::json!([3072, 384, 3072, 24, 1024, 1024, 32, 32]))?;
                equal("reduction", "ordered_k16".into())?;
            }
            Op::RouterProj => {
                equal("width", 2304.into())?;
                equal("outputs", 128.into())?;
                equal("output_dtype", "torch.float32".into())?;
                equal("small_row_max", 8.into())?;
                equal("reduction", "block_k288_le8_ordered_k16_gt8".into())?;
            }
            Op::Router => {
                equal("experts", 128.into())?; equal("topk", 8.into())?; equal("route_scale", 8.0.into())?;
            }
            Op::Gate { .. } => {
                equal("width", 3072.into())?; equal("situ", false.into())?;
            }
            Op::Norm { value, precision } => {
                equal("width", if value { 128.into() } else { 2304.into() })?;
                equal("heads", if value { 8.into() } else { 1.into() })?;
                equal("weighted", (!value).into())?; equal("eps", 1e-6.into())?;
                equal("output_dtype", match precision {
                    Precision::Bf16 => "torch.bfloat16", Precision::F32 => "torch.float32"
                }.into())?;
            }
            Op::KvPack { precision, fp8 } => {
                equal("cache_abi", 2.into())?;
                equal("kv_format", if fp8 { "fp8" } else { "bf16" }.into())?;
                equal("record_bytes", (if fp8 { 2112 } else { 4096 }).into())?;
                equal("scale_offset", (if fp8 { 2048 } else { 0 }).into())?;
                equal("heads", 8.into())?; equal("dim", 128.into())?;
                equal("input_dtype", match precision {
                    Precision::Bf16 => "torch.bfloat16", Precision::F32 => "torch.float32"
                }.into())?;
            }
        }
        let rows = spec.capacity_rows as usize;
        Ok(Self { program, op, rows })
    }

    /// # Safety
    /// Buffers in the named ABI order have the declared typed extents on the
    /// stream's device. Writers do not alias readers. Owners and stream survive queued work.
    pub unsafe fn launch(&self, buffers: &[Buffer], rows: usize, residual: Option<(usize, usize)>,
        stream: *mut c_void) -> Result<()> {
        ensure!(rows > 0 && rows <= self.rows && buffers.len() == self.op.pointers().len()
            && buffers.iter().all(|b| !b.ptr.is_null() && b.device_id == buffers[0].device_id && b.flags == 0),
            "Inkling 2 Mini producer live rows/device/pointer count differs");
        let required: Vec<usize> = match self.op {
            Op::AttnInProjLocal => vec![
                rows * 2304 * 2, rows * 2304 * 2, rows * 2304 * 2,
                3072 * 2304 * 2, 384 * 2304 * 2, 3072 * 2304 * 2, 24 * 2304 * 2, 1024 * 2304 * 2, 1024 * 2304 * 2,
                rows * 3072 * 2, rows * 384 * 2, rows * 3072 * 2, rows * 24 * 4, rows * 1024 * 2, rows * 1024 * 2,
            ],
            Op::AttnInProjGlobal => vec![
                rows * 2304 * 2, rows * 2304 * 2, rows * 2304 * 2,
                3072 * 2304 * 2, 384 * 2304 * 2, 3072 * 2304 * 2, 24 * 2304 * 2, 1024 * 2304 * 2, 1024 * 2304 * 2,
                32 * 2304 * 2, 32 * 2304 * 2,
                rows * 3072 * 2, rows * 384 * 2, rows * 3072 * 2, rows * 24 * 4, rows * 1024 * 2, rows * 1024 * 2,
                rows * 32 * 4, rows * 32 * 4,
            ],
            Op::RouterProj => vec![rows * 2304 * 2, 128 * 2304 * 2, rows * 128 * 4],
            _ => vec![],
        };
        ensure!(buffers.iter().zip(required).all(|(b, size)| b.bytes >= size && b.ptr as usize % 16 == 0),
            "Inkling 2 Mini projection extent/alignment differs");
        let mut scalars = vec![Scalar::I32(rows as i32)];
        match (self.op, residual) {
            (Op::Residual { max_candidates, .. }, Some((arena_rows, candidates))) => {
                ensure!(arena_rows >= rows && arena_rows <= i32::MAX as usize
                    && (1..=max_candidates).contains(&candidates),
                    "Inkling 2 Mini residual arena/candidates differ");
                scalars.extend([Scalar::I32(arena_rows as i32), Scalar::I32(candidates as i32)]);
            }
            (Op::Convolution { width, .. }, Some((stride, slots))) => {
                ensure!(stride >= 3 * width && stride <= i64::MAX as usize && slots > 0
                    && buffers[3].bytes >= (slots - 1).checked_mul(stride)
                        .and_then(|n| n.checked_add(3 * width))
                        .and_then(|n| n.checked_mul(4)).unwrap_or(usize::MAX),
                    "Inkling 2 Mini convolution history stride/extent differs");
                scalars.push(Scalar::I64(stride as i64));
            }
            (Op::Convolution { width, precision, .. }, None) => {
                ensure!(buffers[3].bytes >= 3 * width * precision.bytes(), "Inkling 2 Mini history too short");
                scalars.push(Scalar::I64((3 * width) as i64));
            }
            (Op::Residual { .. }, None) | (_, Some(_)) => {
                anyhow::bail!("Inkling 2 Mini producer residual scalar ABI differs");
            }
            (_, None) => {}
        }
        let pointers = buffers.iter().map(|b| b.ptr).collect::<Vec<_>>();
        // SAFETY: caller supplies typed extents/metadata/lifetimes; ABI and counts checked above.
        unsafe { self.program.launch(&pointers, &scalars, stream)?; }
        Ok(())
    }
}
