//! Shared combined-reply codec; the worker's existing BF16 output is its
//! source, and the transport retains the packed destination through its fence.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::{BoundProgram, Scalar}, CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::{expert::BackboneRequest, ExpertV2Dtype};
use std::ffi::c_void;

pub(crate) struct ReplyCodec<'a> { program: BoundProgram<'a> }
impl<'a> ReplyCodec<'a> {
    pub(crate) fn load(library: &'a NativeLibrary) -> Result<Self> {
        let programs = library.programs()?;
        Ok(Self { program: programs.bound("expert_reply_quantize", &["source", "out", "mode"])? })
    }
    /// # Safety
    /// Source is BF16 `[rows, hidden]`; output is an exclusive chunk of the request's wire
    /// plane. Both remain live until this owner stream passes its unit fence.
    pub(crate) unsafe fn quantize(&self, request: &BackboneRequest<'_>, source: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer, rows: usize, stream: *mut c_void) -> Result<()> {
        let format = match request.output_dtype() {
            ExpertV2Dtype::Bf16 => return Ok(()),
            ExpertV2Dtype::Fp8E4m3Ue8m0K32 => 1,
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => 2,
            _ => anyhow::bail!("unsupported combined expert reply dtype"),
        };
        let hidden = cuteafd_core::expert_geometry().hidden;
        ensure!(rows > 0 && rows <= request.rows() as usize, "expert reply codec chunk rows exceed request");
        let source_bytes = rows.checked_mul(hidden as usize).and_then(|n| n.checked_mul(2))
            .context("expert reply source extent overflow")?;
        let output_bytes = rows.checked_mul(request.output_row_bytes()? as usize)
            .context("expert reply output extent overflow")?;
        ensure!(source.bytes >= source_bytes && output.bytes >= output_bytes, "expert reply codec extent mismatch");
        // SAFETY: caller's source/destination contract; constant format means mode is not read.
        unsafe { self.program.launch(&[source.ptr, output.ptr, std::ptr::null_mut()],
            &[Scalar::I32(i32::try_from(rows)?), Scalar::I32(i32::try_from(hidden)?),
                Scalar::I32(format)], stream) }
    }
}
