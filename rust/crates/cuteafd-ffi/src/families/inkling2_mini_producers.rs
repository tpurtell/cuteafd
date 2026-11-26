//! Inkling 2 Mini TP1 arithmetic producers, looped cross-pass controllers, and fused per-visit glue.
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
    Router,
    Gate { precision: Precision },
    Norm { value: bool, precision: Precision },
    KvPack { precision: Precision, fp8: bool },
    // Dedicated Mini fused per-visit glue programs:
    FastNorm,
    ResidualNorm { queries: usize },
    LoopBlendNormFast { slots: usize, broadcast_dest: bool, broadcast_weight: bool },
    ResidualPairNorm,
    AttnCrossPairNorm { broadcast_dest: bool },
    AttnPrep { local: bool, fp8: bool, replay: bool },
    PostAttnConv { starts_block: bool, replay: bool },
    PostMlpConvResidual { replay: bool },
    FastRouter,
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
            Self::Router => "router_e128_k8".into(),
            Self::Gate { precision } => format!("gate_h3072_{}", precision_suffix(precision)),
            Self::Norm { value, precision } => if value {
                format!("norm_v_h8_d128_{}", precision_suffix(precision))
            } else {
                format!("norm_h2304_{}", precision_suffix(precision))
            },
            Self::KvPack { precision, fp8 } => format!("kv_pack_{}_{}_v2", if fp8 { "fp8" } else { "bf16" }, precision_suffix(precision)),
            Self::FastNorm => "fast_norm_h2304_bf16".into(),
            Self::ResidualNorm { queries } => {
                ensure!(matches!(queries, 1 | 3), "Inkling 2 Mini residual_norm queries unsupported");
                format!("residual_norm_q{queries}_c5")
            }
            Self::LoopBlendNormFast { slots, broadcast_dest, broadcast_weight } => {
                ensure!(matches!((slots, broadcast_dest, broadcast_weight),
                    (1, false, false) | (3, true, true) | (3, false, false)),
                    "Inkling 2 Mini fast loop blend_norm geometry unsupported");
                format!("loop_blend_norm_fast_s{slots}{}", if broadcast_dest { "_bcast" } else { "" })
            }
            Self::ResidualPairNorm => "residual_pair_norm_s1".into(),
            Self::AttnCrossPairNorm { broadcast_dest } => {
                format!("attn_cross_pair_norm{}", if broadcast_dest { "_bcast" } else { "" })
            }
            Self::AttnPrep { local, fp8, replay } => {
                ensure!(!(local && fp8), "Inkling 2 Mini local attn_prep does not use fp8");
                format!("attn_prep_{}_{}_{}",
                    if local { "local" } else { "global" },
                    if fp8 { "fp8" } else { "bf16" },
                    if replay { "replay" } else { "strided" })
            }
            Self::PostAttnConv { starts_block, replay } => {
                format!("post_attn_conv_{}_{}",
                    if starts_block { "start" } else { "cont" },
                    if replay { "replay" } else { "strided" })
            }
            Self::PostMlpConvResidual { replay } => {
                format!("post_mlp_conv_residual_{}", if replay { "replay" } else { "strided" })
            }
            Self::FastRouter => "fast_router_e128_k8".into(),
        }))
    }

    fn pointers(self) -> &'static [&'static str] {
        match self {
            Self::Convolution { .. } => &["x", "weight", "dynamic", "history", "first", "slots", "out"],
            Self::Clip { .. } => &["x", "scale", "positions", "out"],
            Self::Residual { .. } => &["arena", "addend", "query", "out"],
            Self::LoopPairNorm { .. } => &["dest", "msg", "out"],
            Self::LoopBlendNorm { .. } | Self::LoopBlendNormFast { .. } => {
                &["dest", "msg", "forget_gain", "weight", "out"]
            }
            Self::Router | Self::FastRouter => &["logits", "bias", "ids", "weights"],
            Self::Gate { .. } => &["x", "gate", "gamma", "out"],
            Self::Norm { .. } | Self::FastNorm => &["x", "weight", "out"],
            Self::KvPack { .. } => &["k", "v", "slots", "cache"],
            Self::ResidualNorm { .. } => &["arena", "query", "weight", "out"],
            Self::ResidualPairNorm => &["arena", "query", "msg", "mixed_out", "paired_out"],
            Self::AttnCrossPairNorm { .. } => {
                &["arena", "attn_query", "prev_arena", "cross_query", "dest_out", "msg_out", "paired_out"]
            }
            Self::AttnPrep { .. } => &[
                "q", "q_scale", "r", "r_scale", "positions", "q_out", "rel_input_out",
                "k_bf16", "k_weight", "k_dynamic", "k_history",
                "v_bf16", "v_weight", "v_dynamic", "v_history",
                "first", "conv_slots", "cache_slots", "cache", "k_replay", "v_replay",
            ],
            Self::PostAttnConv { .. } => &[
                "x_bf16", "weight", "history", "first", "slots", "partial", "arena", "out_base", "replay",
            ],
            Self::PostMlpConvResidual { .. } => &[
                "x_bf16", "weight", "history", "first", "slots", "base", "partial", "arena", "replay",
            ],
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
        let kinds: &[u8] = match op {
            Op::Residual { .. }
            | Op::ResidualNorm { .. }
            | Op::ResidualPairNorm
            | Op::AttnCrossPairNorm { .. } => b"iii",
            Op::Convolution { .. } => b"il",
            Op::AttnPrep { .. } => b"illl",
            Op::PostAttnConv { .. } | Op::PostMlpConvResidual { .. } => b"iiilll",
            _ => b"i",
        };
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
            Op::LoopBlendNorm { slots, broadcast_dest, broadcast_weight }
            | Op::LoopBlendNormFast { slots, broadcast_dest, broadcast_weight } => {
                equal("hidden", 2304.into())?; equal("slots", slots.into())?;
                equal("broadcast_dest", broadcast_dest.into())?;
                equal("broadcast_weight", broadcast_weight.into())?;
                equal("eps", 1e-6.into())?;
            }
            Op::Router | Op::FastRouter => {
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
            Op::FastNorm => {
                equal("width", 2304.into())?; equal("eps", 1e-6.into())?;
                equal("output_dtype", "torch.bfloat16".into())?;
            }
            Op::ResidualNorm { queries } => {
                equal("hidden", 2304.into())?; equal("queries", queries.into())?;
                equal("max_candidates", 5.into())?; equal("eps", 1e-6.into())?;
                equal("output_dtype", "torch.bfloat16".into())?;
            }
            Op::ResidualPairNorm => {
                equal("hidden", 2304.into())?; equal("slots", 1.into())?;
                equal("max_candidates", 5.into())?; equal("eps", 1e-6.into())?;
            }
            Op::AttnCrossPairNorm { broadcast_dest } => {
                equal("hidden", 2304.into())?; equal("broadcast_dest", broadcast_dest.into())?;
                equal("max_candidates", 5.into())?; equal("eps", 1e-6.into())?;
            }
            Op::AttnPrep { local, fp8, replay } => {
                equal("local", local.into())?; equal("fp8", fp8.into())?; equal("replay", replay.into())?;
                equal("heads", 24.into())?; equal("kv_heads", 8.into())?; equal("head_dim", 128.into())?;
                equal("rel_dim", 16.into())?; equal("rel_padded", 64.into())?; equal("taps", 4.into())?;
            }
            Op::PostAttnConv { starts_block, replay } => {
                equal("hidden", 2304.into())?; equal("taps", 4.into())?;
                equal("starts_block", starts_block.into())?; equal("replay", replay.into())?;
            }
            Op::PostMlpConvResidual { replay } => {
                equal("hidden", 2304.into())?; equal("taps", 4.into())?; equal("replay", replay.into())?;
            }
        }
        let rows = spec.capacity_rows as usize;
        Ok(Self { program, op, rows })
    }

    fn check_common(&self, buffers: &[Buffer], rows: usize) -> Result<()> {
        ensure!(rows > 0 && rows <= self.rows && buffers.len() == self.op.pointers().len()
            && buffers.iter().all(|b| !b.ptr.is_null() && b.device_id == buffers[0].device_id && b.flags == 0),
            "Inkling 2 Mini producer live rows/device/pointer count differs");
        Ok(())
    }

    /// # Safety
    /// Buffers in the named ABI order have the declared typed extents on the
    /// stream's device. Writers do not alias readers. Owners and stream survive queued work.
    pub unsafe fn launch(&self, buffers: &[Buffer], rows: usize, residual: Option<(usize, usize)>,
        stream: *mut c_void) -> Result<()> {
        self.check_common(buffers, rows)?;
        let mut scalars = vec![Scalar::I32(rows as i32)];
        match (self.op, residual) {
            (
                Op::Residual { .. }
                | Op::ResidualNorm { .. }
                | Op::ResidualPairNorm
                | Op::AttnCrossPairNorm { .. },
                Some((arena_rows, candidates)),
            ) => {
                let max_c = match self.op {
                    Op::Residual { max_candidates, .. } => max_candidates,
                    _ => 5,
                };
                ensure!(arena_rows >= rows && arena_rows <= i32::MAX as usize
                    && (1..=max_c).contains(&candidates),
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
            (Op::AttnPrep { .. } | Op::PostAttnConv { .. } | Op::PostMlpConvResidual { .. }, _) => {
                anyhow::bail!("use dedicated launch helper for fused Mini conv producers");
            }
            (
                Op::Residual { .. }
                | Op::ResidualNorm { .. }
                | Op::ResidualPairNorm
                | Op::AttnCrossPairNorm { .. },
                None,
            ) | (_, Some(_)) => {
                anyhow::bail!("Inkling 2 Mini producer residual scalar ABI differs");
            }
            (_, None) => {}
        }
        let pointers = buffers.iter().map(|b| b.ptr).collect::<Vec<_>>();
        // SAFETY: caller supplies typed extents/metadata/lifetimes; ABI and counts checked above.
        unsafe { self.program.launch(&pointers, &scalars, stream)?; }
        Ok(())
    }

    /// # Safety
    /// Buffers match `Op::AttnPrep` ABI extents and do not alias across readers/writers.
    pub unsafe fn launch_attn_prep(
        &self,
        buffers: &[Buffer],
        rows: usize,
        slot_stride: usize,
        lag_stride: usize,
        replay_stride: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let Op::AttnPrep { replay, .. } = self.op else {
            anyhow::bail!("launch_attn_prep requires Op::AttnPrep");
        };
        self.check_common(buffers, rows)?;
        ensure!(
            lag_stride >= 1024 && slot_stride >= 3 * lag_stride && (!replay || replay_stride >= 1024),
            "Inkling 2 Mini attn_prep strides differ"
        );
        let pointers = buffers.iter().map(|b| b.ptr).collect::<Vec<_>>();
        let scalars = [
            Scalar::I32(rows as i32),
            Scalar::I64(slot_stride as i64),
            Scalar::I64(lag_stride as i64),
            Scalar::I64(replay_stride as i64),
        ];
        // SAFETY: validated extents/strides/lifetimes on retained stream.
        unsafe { self.program.launch(&pointers, &scalars, stream)?; }
        Ok(())
    }

    /// # Safety
    /// Buffers match `Op::PostAttnConv` or `Op::PostMlpConvResidual` ABI extents and do not alias.
    pub unsafe fn launch_post_conv(
        &self,
        buffers: &[Buffer],
        rows: usize,
        arena_rows: usize,
        mlp_cand_idx: usize,
        slot_stride: usize,
        lag_stride: usize,
        replay_stride: usize,
        stream: *mut c_void,
    ) -> Result<()> {
        let (starts_block, replay) = match self.op {
            Op::PostAttnConv { starts_block, replay } => (starts_block, replay),
            Op::PostMlpConvResidual { replay } => (false, replay),
            _ => anyhow::bail!("launch_post_conv requires Op::PostAttnConv or Op::PostMlpConvResidual"),
        };
        self.check_common(buffers, rows)?;
        ensure!(
            arena_rows >= rows
                && arena_rows <= i32::MAX as usize
                && mlp_cand_idx < 5
                && (!starts_block || mlp_cand_idx >= 1)
                && lag_stride >= 2304
                && slot_stride >= 3 * lag_stride
                && (!replay || replay_stride >= 2304),
            "Inkling 2 Mini post_conv arena/strides differ"
        );
        let pointers = buffers.iter().map(|b| b.ptr).collect::<Vec<_>>();
        let scalars = [
            Scalar::I32(rows as i32),
            Scalar::I32(arena_rows as i32),
            Scalar::I32(mlp_cand_idx as i32),
            Scalar::I64(slot_stride as i64),
            Scalar::I64(lag_stride as i64),
            Scalar::I64(replay_stride as i64),
        ];
        // SAFETY: validated extents/strides/lifetimes on retained stream.
        unsafe { self.program.launch(&pointers, &scalars, stream)?; }
        Ok(())
    }
}
