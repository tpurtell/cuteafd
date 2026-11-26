//! How every family's target logits run through a BF16 vocabulary head.
//!
//! One route for all families (GLM 5.3, GLM 5.3 Flash, MiMo, Qwen, DeepSeek V4 and V4.1): a BF16
//! head's logits run as a BF16 tensor-core GEMM with FP32 accumulation and FP32 logits at every
//! row count ([`TargetHead::Tensor`], the default). Their reference implementations compute the
//! head as BF16 products with FP32 sums. On five real heads (RTX PRO 6000,
//! final-norm-shaped rows), the fixed route is byte-identical for batches of
//! 1..64 versus serial rows, with FP64-reference mean KL at most 1.46e-11 nats.
//! It is faster than pedantic at all tested shapes (1, 4, 8, 16 and 64 rows);
//! `native/tests/target_head_selftest.py` also checks capacity and graph stability.
//!
//! [`TargetHead::Exact`] is a diagnostic: the pedantic FP32 cuBLAS GEMM (FP32 products and sums on
//! CUDA cores) at every row count. FP8 heads run their own programs and ignore the setting.
//! Drafters keep their own route (`shared::draft`); only target logits come through here.
//!
//! Selected by `CUTEAFD_TARGET_HEAD` (`tensor` or `exact`; launcher key `TARGET_HEAD`), read once
//! per process before the first launch: decode graphs capture the head's launch.
//! The tensor route pins one cuBLASLt algorithm/tile with split-K disabled at load;
//! decode and verify must not choose different reductions based on row count.

use anyhow::{bail, ensure, Context, Result};
use cuteafd_ffi::{programs::{Programs, VocabularyHead}, CuteafdDeviceBuffer as Buffer};
use std::ffi::c_void;

/// How target logits run through a BF16 head.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TargetHead {
    /// BF16 tensor cores, FP32 accumulation and logits, one read of the head, at every row count.
    #[default]
    Tensor,
    /// Diagnostic: the pedantic FP32 cuBLAS GEMM on CUDA cores at every row count.
    Exact,
}

impl TargetHead {
    /// `CUTEAFD_TARGET_HEAD` (unset: [`TargetHead::Tensor`]).
    pub fn parse(value: Option<&str>) -> Result<Self> {
        Ok(match value.map(str::trim) {
            None | Some("" | "tensor") => Self::Tensor,
            Some("exact") => Self::Exact,
            Some(other) => bail!("CUTEAFD_TARGET_HEAD must be tensor or exact, not {other:?}"),
        })
    }

    /// The process's setting, read once. An invalid value is reported and keeps the default.
    pub fn configured() -> Self {
        static MODE: std::sync::OnceLock<TargetHead> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| {
            let value = std::env::var("CUTEAFD_TARGET_HEAD").ok();
            let mode = Self::parse(value.as_deref()).unwrap_or_else(|error| {
                tracing::warn!(%error, "keeping the tensor-core target head");
                Self::Tensor
            });
            tracing::info!(target_head = ?mode, "BF16 target vocabulary heads");
            mode
        })
    }

    pub fn blas_head(self, bf16: bool) -> cuteafd_loader::placement::planner::serving::BlasHead {
        use cuteafd_loader::placement::planner::serving::BlasHead;
        if !bf16 { BlasHead::Native } else {
            match self { Self::Tensor => BlasHead::TensorLt, Self::Exact => BlasHead::Gemm }
        }
    }

    pub fn native_mode(self) -> cuteafd_ffi::programs::HeadMode {
        match self {
            Self::Tensor => cuteafd_ffi::programs::HeadMode::TensorLt,
            Self::Exact => cuteafd_ffi::programs::HeadMode::Gemm,
        }
    }

    /// Whether this setting takes tensor cores (every row count, or none).
    pub fn tensor_cores(self) -> bool {
        self == Self::Tensor
    }
}

/// Target logits `[rows, vocab]` (FP32) of BF16 rows `x [rows, width]` through the BF16 head
/// `weight [vocab, width]` under `mode`.
///
/// # Safety
/// `x`, `weight` and `logits` are live device buffers of these shapes on `stream`'s device until
/// it drains, and `head` was created for this width, vocabulary and at least `rows` rows.
pub(crate) unsafe fn launch(mode: TargetHead, head: &VocabularyHead<'_>, x: *const c_void,
    weight: *const c_void, logits: *mut f32, rows: usize, stream: *mut c_void) -> Result<()> {
    let rows = u32::try_from(rows).map_err(|_| anyhow::anyhow!("target head rows exceed u32"))?;
    // SAFETY: the caller's contract.
    unsafe {
        if mode.tensor_cores() {
            head.launch_tensor_invariant(x.cast(), weight.cast(), logits, rows, stream)
        } else {
            head.launch(x.cast(), weight.cast(), logits, rows, stream)
        }
    }
}

/// Typed head row views, shared by target and drafter calls at the stored
/// vocabulary width. Validate the complete batch before launching any chunk.
pub(crate) fn head_chunks(rows: usize, input: Buffer, output: Buffer, k: usize, n: usize,
    mut launch: impl FnMut(usize, Buffer, Buffer) -> Result<()>) -> Result<()> {
    let extent = |width: usize, element: usize| -> Result<usize> {
        rows.checked_mul(width).and_then(|v| v.checked_mul(element)).filter(|&v| v > 0)
            .context("head typed extent overflow or empty")
    };
    let (input_bytes, output_bytes) = (extent(k, 2)?, extent(n, 4)?);
    ensure!(input.bytes >= input_bytes && output.bytes >= output_bytes,
        "head input/output exceeds admitted extent");
    let start = input.ptr as usize;
    let end = start.checked_add(input_bytes).context("head input address overflow")?;
    let out = output.ptr as usize;
    let out_end = out.checked_add(output_bytes).context("head output address overflow")?;
    ensure!(start != 0 && out != 0 && (end <= out || out_end <= start), "head input/output aliases or is null");
    for first in (0..rows).step_by(64) {
        let count = (rows - first).min(64);
        let view = |buffer: Buffer, stride: usize| Buffer {
            ptr: buffer.ptr.cast::<u8>().wrapping_add(first * stride).cast(), bytes: count * stride, ..buffer
        };
        launch(count, view(input, k * 2), view(output, n * 4))?;
    }
    Ok(())
}

/// Allocation-free block-FP8 vocabulary projection. Families retain the same
/// admitted weight arena and serialized scratch; crop/softcap remain outside.
pub(crate) struct BlockFp8Head<'a> {
    registry: cuteafd_ffi::native_attention::Fp8Projections<'a>,
    shape: [usize; 2],
}

impl<'a> BlockFp8Head<'a> {
    pub(crate) fn load(programs: &'a Programs<'a>, shape: [usize; 2]) -> Result<Self> {
        let [n, k] = shape;
        Ok(Self { registry: cuteafd_ffi::native_attention::Fp8Projections::load(programs,
            &[(k,n)], 64, cuteafd_ffi::draft_projection::ProjectionOutput::F32)?, shape })
    }

    pub(crate) fn scratch_bytes(&self) -> Result<usize> {
        Ok(self.registry.scratch_bytes(self.shape[1], self.shape[0])?)
    }

    pub(crate) fn launch(&self, rows: usize, input: Buffer,
        weights: &crate::shared::native_attention::NativeAttentionWeights<'_>,
        output: Buffer, scratch: Buffer, stream: *mut c_void) -> Result<()> {
        use cuteafd_loader::formats::trellis_core::CoreSplit;
        ensure!(weights.layout.shape == self.shape && weights.layout.fp8
            && weights.layout.ranks == 1 && weights.layout.rank == 0
            && weights.layout.split == CoreSplit::Whole,
            "block FP8 head differs from bound whole-owner shape");
        let scales = weights.scales.as_ref().context("block FP8 head scales absent")?.buffer;
        self.launch_buffers(rows,input,weights.values.buffer,scales,output,scratch,stream)
    }

    /// Final arena values/scales are borrowed without manufacturing new owners.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_buffers(&self, rows:usize,input:Buffer,values:Buffer,scales:Buffer,
        output:Buffer,scratch:Buffer,stream:*mut c_void) -> Result<()> {
        ensure!([input.device_id, values.device_id, scales.device_id, scratch.device_id]
            .into_iter().all(|device| device == output.device_id)
            && scratch.bytes >= self.scratch_bytes()?, "block FP8 head device/scratch differs from admission");
        let [n, k] = self.shape;
        ensure!(values.bytes >= n*k && scales.bytes >= n/128*(k/128)*4,
            "block FP8 head final buffers differ from admitted shape");
        head_chunks(rows, input, output, k, n, |count, x, y| {
            // SAFETY: typed disjoint row views, admitted weights and serialized
            // scratch remain live on this device until the caller drains stream.
            unsafe { self.registry.launch(x.ptr, values.ptr, scales.ptr,
                y, scratch, count, k, n, stream)?; }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires sealed SM120 native FP8 head exports"]
    fn shared_block_fp8_head_matches_muse_legacy_launch_bytes() -> Result<()> {
        use crate::shared::{memory::{DeviceAllocation, device::cache_test_device, TEST},
            native_attention::NativeAttentionWeights};
        use cuteafd_loader::formats::{native_attention::NativeLayout, trellis_core::CoreSplit};
        // SAFETY: this explicit GPU test uses the slot's sealed native ABI.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = cache_test_device(&library)?;
        let programs = library.programs()?.with_manifest(std::path::Path::new(
            &std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        let (n, k) = (202112usize, 8192usize);
        let rows = 129usize;
        let shared = BlockFp8Head::load(&programs, [n,k])?;
        let legacy = cuteafd_ffi::native_attention::Fp8Projections::load(&programs, &[(k,n)], 64,
            cuteafd_ffi::draft_projection::ProjectionOutput::F32)?;
        let weights = NativeAttentionWeights {
            layout: NativeLayout { stem:"lm_head".into(),shape:[n,k],fp8:true,rank:0,ranks:1,split:CoreSplit::Whole },
            values: DeviceAllocation::new(&library,n*k,TEST)?,
            scales: Some(DeviceAllocation::new(&library,n/128*(k/128)*4,TEST)?),
        };
        let input = DeviceAllocation::new(&library,rows*k*2,TEST)?;
        let output = DeviceAllocation::new(&library,rows*n*4,TEST)?;
        let expected = DeviceAllocation::new(&library,rows*n*4,TEST)?;
        let scratch = DeviceAllocation::new(&library,shared.scratch_bytes()?,TEST)?;
        let weight_bytes: Vec<u8> = (0..n*k).map(|i| [0x38u8,0xb8,0x30,0xb0][i%4]).collect();
        let input_bytes: Vec<u8> = (0..rows*k).flat_map(|i| {
            let value = ((i*17%97) as f32 - 48.) / 32.;
            ((value.to_bits()>>16) as u16).to_le_bytes()
        }).collect();
        let scale_bytes: Vec<u8> = (0..n/128*(k/128)).flat_map(|i| (0.01f32+(i%7) as f32*0.001).to_le_bytes()).collect();
        library.copy_h2d(weights.values.buffer,&weight_bytes)?;
        library.copy_h2d(weights.scales.as_ref().unwrap().buffer,&scale_bytes)?;
        library.copy_h2d(input.buffer,&input_bytes)?;
        for rows in [1,16,63,64,65,80,128,129] {
            // This is Muse's pre-extraction invocation: same m64 registry and
            // whole stored-width chunks, before family-specific crop/softcap.
            head_chunks(rows,input.buffer,expected.buffer,k,n,|count,x,y| {
                // SAFETY: test owns disjoint buffers and drains before reading.
                unsafe { legacy.launch(x.ptr,weights.values.buffer.ptr,
                    weights.scales.as_ref().unwrap().buffer.ptr,y,scratch.buffer,count,k,n,std::ptr::null_mut())?; }
                Ok(())
            })?;
            shared.launch(rows,input.buffer,&weights,output.buffer,scratch.buffer,std::ptr::null_mut())?;
            // SAFETY: both launches use the current device's default stream.
            unsafe { library.cuda_stream_synchronize(std::ptr::null_mut())?; }
            let mut actual = vec![0u8;rows*n*4];
            let mut reference = vec![0u8;rows*n*4];
            library.copy_d2h_quiescent(&mut actual,output.buffer)?;
            library.copy_d2h_quiescent(&mut reference,expected.buffer)?;
            assert_eq!(actual,reference,"Muse shared head changed bytes at {rows} rows");
        }
        assert_eq!(device.id,output.buffer.device_id);
        Ok(())
    }

    /// Unset and `tensor` take tensor cores; `exact` never does; anything else is refused.
    #[test]
    fn the_target_head_defaults_to_tensor_cores_and_keeps_exact_as_a_diagnostic() {
        assert_eq!(TargetHead::default(), TargetHead::Tensor);
        for value in [None, Some(""), Some("tensor"), Some(" tensor ")] {
            assert_eq!(TargetHead::parse(value).unwrap(), TargetHead::Tensor, "{value:?}");
        }
        assert_eq!(TargetHead::parse(Some("exact")).unwrap(), TargetHead::Exact);
        for value in ["fp8", "pedantic", "Tensor"] {
            assert!(TargetHead::parse(Some(value)).is_err(), "{value}");
        }
        assert!(TargetHead::Tensor.tensor_cores());
        assert!(!TargetHead::Exact.tensor_cores());
    }

}
