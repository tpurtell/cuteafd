//! E4M3 copies of BF16 GEMM weights for skinny steps (drafters, draft heads):
//! packed at load by `fp8_gemv.cu` (FP32 scales per output row and 128-wide
//! K block, `amax / 448` or with `pow2` the smallest power of two >= it) and
//! applied by its W8A16 tensor-core GEMV (BF16 activations, FP32 sums).
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

#[cfg(all(test, target_os = "linux"))]
#[path = "fp8_linear_lifetime_tests.rs"]
mod lifetime_tests;

/// How FP8 copies of BF16 weights pick a block's scale (`--fp8-scales`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Fp8Scales {
    /// amax / 448.
    Amax,
    /// The smallest power of two >= amax / 448 (after Hugh Madden's
    /// glm53f-afd pow2_scale, MIT e56ce35): values with at most E4M3's 3
    /// mantissa bits in the block's range quantize exactly.
    Pow2,
    /// Per block, whichever of the two leaves the smaller squared error.
    Best,
}

impl Fp8Scales {
    /// The native kernels' rule code.
    pub fn code(self) -> i32 {
        match self {
            Fp8Scales::Amax => 0,
            Fp8Scales::Pow2 => 1,
            Fp8Scales::Best => 2,
        }
    }
}

/// How [`Fp8Weight::apply_rows`] runs (`cuteafd_fp8_linear`'s modes; GLM 5.3 Flash's
/// `--draft-linear`). Ordered by the scratch each needs: a scratch admitted for one mode serves
/// the modes before it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
pub(crate) enum Fp8Rows {
    /// W8A16 in passes of 64 rows ([`Fp8Weight::apply`]).
    #[default]
    W8a16,
    /// The same bits in passes of 128 rows: one pass over the weight up to 128 rows, and one
    /// launch chain where passes of 64 take two.
    Wide,
    /// W8A16 up to 8 rows (one draft block: the same bits); past them E4M3 activations per row
    /// and 128-wide K block (amax / 448) on FP8 tensor cores, in passes of 128 rows.
    W8a8,
}

impl Fp8Rows {
    /// The native mode code.
    pub fn code(self) -> i32 {
        match self {
            Fp8Rows::W8a16 => 0,
            Fp8Rows::Wide => 1,
            Fp8Rows::W8a8 => 2,
        }
    }
}

/// An E4M3 copy of a `[n, k]` BF16 weight in the GEMV's fragment order.
pub(crate) struct Fp8Weight<'a> {
    packed: DeviceAllocation<'a>,
    scale: DeviceAllocation<'a>,
    pub n: usize,
    pub k: usize,
}

impl<'a> Fp8Weight<'a> {
    /// Packs the live BF16 weight `w` [n, k] on `stream` (the caller
    /// synchronizes before the source is freed, including on errors, and
    /// retains its source owner if completion cannot be proved).
    pub fn pack(library: &'a NativeLibrary, w: *const c_void, n: usize, k: usize, scales: Fp8Scales,
        stream: *mut c_void) -> Result<Self> {
        ensure!(n % 16 == 0 && k % 128 == 0, "FP8 copy of [{n}, {k}]: needs n % 16 == 0 and k % 128 == 0");
        let packed = DeviceAllocation::new(library, n * k)?;
        let scale = DeviceAllocation::new(library, n * k / 128 * 4)?;
        // SAFETY: `w` is a live [n, k] BF16 weight; the new buffers hold the packed copy.
        let launched = unsafe { library.fp8_w8a16_pack(w, packed.buffer.ptr, scale.buffer.ptr, n, k, scales.code(), stream) };
        if let Err(error) = launched {
            // SAFETY: source, destination and scale remain live here. A native
            // launch error may follow queued work; drain before their owners
            // leave this scope, including the new destination allocations.
            let drained = unsafe { library.cuda_stream_synchronize(stream) };
            return match drained {
                Ok(()) => Err(error),
                Err(drain) => {
                    // Completion is unknown. Retain these allocations until
                    // process teardown rather than running cuda_free on Drop.
                    // The borrowed library may otherwise unload as the loader
                    // unwinds; the caller must also retain the source owner.
                    library.quarantine_module_after_failed_drain();
                    std::mem::forget(packed);
                    std::mem::forget(scale);
                    Err(error.context(format!("FP8 destinations and native module quarantined after failed drain: {drain}")))
                }
            };
        }
        Ok(Self { packed, scale, n, k })
    }

    /// Device bytes of the copy and its scales.
    pub fn bytes(&self) -> usize {
        self.n * self.k + self.n * self.k / 128 * 4
    }

    /// A failed stream drain cannot prove that queued reads/writes retired.
    /// Keep the owning allocations and their native module live until process
    /// teardown. The caller must also retain any outstanding source/scratch.
    pub fn quarantine(self) {
        self.packed.library.quarantine_module_after_failed_drain();
        std::mem::forget(self);
    }

    /// `out` [rows, n'] = `x` [rows, k] @ rows `first..first + n'` of the
    /// weight (`first` a multiple of 16), BF16 out or FP32 with `out_f32`.
    ///
    /// # Safety
    /// `x` and `out` are live device buffers of those shapes; `scratch` was
    /// sized for at least `rows` rows of this shape.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn apply(&self, library: &NativeLibrary, x: *const c_void, out: *mut c_void, out_f32: bool, rows: usize,
        first: usize, n: usize, scratch: &DeviceAllocation<'_>, stream: *mut c_void) -> Result<()> {
        ensure!(first % 16 == 0 && first + n <= self.n, "FP8 rows {first}..{} of {}", first + n, self.n);
        // SAFETY: the offsets stay inside the packed copy (16-row tiles are contiguous) and its scales.
        let (packed, scale) = unsafe {
            (self.packed.buffer.ptr.cast::<u8>().add(first * self.k).cast::<c_void>(),
                self.scale.buffer.ptr.cast::<u8>().add(first * self.k / 128 * 4).cast::<c_void>())
        };
        // SAFETY: the caller's contract.
        unsafe { library.fp8_w8a16_linear(x, packed, scale, out, out_f32, rows, self.k, n, scratch.buffer.ptr,
            scratch.buffer.bytes, stream) }
    }

    /// [`Self::apply`] in `mode` ([`Fp8Rows::W8a16`] is [`Self::apply`]).
    ///
    /// # Safety
    /// As [`Self::apply`], with `scratch` sized by [`scratch_rows`] for `mode` (or a later one).
    // The GLM drafters call it; test crates that include this file alone do not.
    #[cfg_attr(test, allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn apply_rows(&self, library: &NativeLibrary, x: *const c_void, out: *mut c_void, out_f32: bool,
        rows: usize, first: usize, n: usize, scratch: &DeviceAllocation<'_>, stream: *mut c_void, mode: Fp8Rows)
        -> Result<()> {
        if mode == Fp8Rows::W8a16 {
            // SAFETY: the caller's contract.
            return unsafe { self.apply(library, x, out, out_f32, rows, first, n, scratch, stream) };
        }
        ensure!(first % 16 == 0 && first + n <= self.n, "FP8 rows {first}..{} of {}", first + n, self.n);
        // SAFETY: the offsets stay inside the packed copy (16-row tiles are contiguous) and its scales.
        let (packed, scale) = unsafe {
            (self.packed.buffer.ptr.cast::<u8>().add(first * self.k).cast::<c_void>(),
                self.scale.buffer.ptr.cast::<u8>().add(first * self.k / 128 * 4).cast::<c_void>())
        };
        // SAFETY: the caller's contract.
        unsafe { library.fp8_linear(x, packed, scale, out, out_f32, rows, self.k, n, mode.code(), scratch.buffer.ptr,
            scratch.buffer.bytes, stream) }
    }
}

/// GEMV scratch for up to `rows` rows of every `(k, n)` shape.
pub(crate) fn scratch<'a>(library: &'a NativeLibrary, rows: usize, shapes: &[(usize, usize)])
    -> Result<DeviceAllocation<'a>> {
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace/fp8-linear");
    let bytes = shapes.iter().map(|&(k, n)| library.fp8_w8a16_workspace(rows, k, n))
        .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    DeviceAllocation::new(library, bytes.max(256))
}

/// [`scratch`] for `mode` ([`Fp8Rows::W8a16`] is [`scratch`]).
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn scratch_rows<'a>(library: &'a NativeLibrary, rows: usize, shapes: &[(usize, usize)], mode: Fp8Rows)
    -> Result<DeviceAllocation<'a>> {
    if mode == Fp8Rows::W8a16 {
        return scratch(library, rows, shapes);
    }
    let _memory_scope = cuteafd_ffi::memory_ledger::scope("workspace/fp8-linear");
    let bytes = shapes.iter().map(|&(k, n)| library.fp8_linear_workspace(rows, k, n, mode.code()))
        .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
    DeviceAllocation::new(library, bytes.max(256))
}

#[cfg(test)]
mod rows_tests {
    use super::*;

    /// The native codes, and the order a scratch admitted for one mode serves the earlier ones
    /// in (passes of 64 rows, then 128, then 128 with the W8A8 scales).
    #[test]
    fn modes_map_to_native_codes_in_scratch_order() {
        assert_eq!([Fp8Rows::W8a16, Fp8Rows::Wide, Fp8Rows::W8a8].map(Fp8Rows::code), [0, 1, 2]);
        assert!(Fp8Rows::W8a16 < Fp8Rows::Wide && Fp8Rows::Wide < Fp8Rows::W8a8);
        assert_eq!(Fp8Rows::default(), Fp8Rows::W8a16);
    }
}
