//! A target-owned vocabulary weight borrowed by target, MTP and DFlash.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, programs::{Programs, Scalar, VocabularyHead}};
use std::ffi::c_void;

pub(crate) const FP8_HEAD_ROWS: usize = 16;

fn fp8_head_spans(rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows).step_by(FP8_HEAD_ROWS).map(move |first| (first, FP8_HEAD_ROWS.min(rows -first)))
}

/// Exactly one resident representation. FP8 uses the target's row-major AOT
/// layout; drafters never create another fragment-packed copy of this head.
pub(crate) enum MimoHead<'a> {
    Bf16(DeviceAllocation<'a>),
    Fp8 { values: DeviceAllocation<'a>, scales: DeviceAllocation<'a> },
}

impl MimoHead<'_> {
    pub fn bytes(&self) -> usize { self.allocations().iter().map(|a| a.buffer.bytes).sum() }
    pub fn allocations(&self) -> Vec<&DeviceAllocation<'_>> {
        match self {
            Self::Bf16(w) => vec![w],
            Self::Fp8 { values, scales } => vec![values, scales],
        }
    }
    pub fn bf16_ptr(&self) -> Result<*const c_void> {
        match self {
            Self::Bf16(w) => Ok(w.buffer.ptr.cast_const()),
            Self::Fp8 { .. } => anyhow::bail!("MiMo BF16-head diagnostic requires --fp8-head false and a separately loaded BF16 head; the FP8-only model owns no BF16 fallback"),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BorrowedHead<'h, 'a> {
    pub weight: &'h MimoHead<'a>,
    pub programs: &'h Programs<'a>,
    pub family: &'static str,
    pub hidden: usize,
    pub vocab: usize,
}

impl BorrowedHead<'_, '_> {
    pub fn launch(self, bf16: Option<&VocabularyHead<'_>>, x: CuteafdDeviceBuffer,
        logits: CuteafdDeviceBuffer, rows: usize, stream: *mut c_void) -> Result<()> {
        ensure!(rows > 0 && rows.checked_mul(self.hidden).and_then(|n| n.checked_mul(2)).is_some_and(|n| n <= x.bytes)
            && rows.checked_mul(self.vocab).and_then(|n| n.checked_mul(4)).is_some_and(|n| n <= logits.bytes),
            "MiMo borrowed head input/output exceeds its admitted rows");
        match self.weight {
            MimoHead::Bf16(w) => {
                // SAFETY: checked input/output extents and the model-owned BF16
                // matrix remain live until the owner's stream drain.
                unsafe { crate::shared::target_head::launch(crate::shared::target_head::TargetHead::configured(),
                    bf16.context("BF16 LM head workspace")?, x.ptr.cast_const(), w.buffer.ptr.cast_const(),
                    logits.ptr.cast(), rows, stream) }
            }
            MimoHead::Fp8 { values, scales } => {
                let program = self.programs.program(&format!("{}_head_fp8", self.family),
                    &["x", "w_fp8", "scale", "logits"])?;
                ensure!(program.spec().capacity_rows as usize >= FP8_HEAD_ROWS,
                    "MiMo FP8-only head needs the complete16-row target head export");
                for (first, n) in fp8_head_spans(rows) {
                    let x = x.ptr.cast::<u8>().wrapping_add(first * self.hidden *2).cast();
                    let out = logits.ptr.cast::<u8>().wrapping_add(first * self.vocab *4).cast();
                    // SAFETY: checked row regions of the live input/logits and
                    // borrowed target weight, ordered on the same owned stream.
                    unsafe { program.launch(&[x, values.buffer.ptr, scales.buffer.ptr, out],
                        &[Scalar::I32(n as i32)], stream)?; }
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_target_and_drafter_tail_is_covered_without_a_fallback() {
        for rows in 1..=4096 {
            let mut next = 0;
            for (first, n) in fp8_head_spans(rows) {
                assert_eq!(first, next);
                assert!((1..=16).contains(&n));
                next += n;
            }
            assert_eq!(next, rows);
        }
    }
}
