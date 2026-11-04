//! Coordinator dense load records, shared by every family: class-2 operands
//! (checkpoint rectangles landing unchanged), class-3 steps (rectangles staged
//! in pinned memory, at most [`STEP_BYTES`], then GPU jobs) and finishes
//! (device-to-device transforms of landed operands). Loaders build them in
//! mount order; [`DenseLoad`] fills each through the mount at once, or records
//! it (under [`recording`] or a push sink) for the coordinator's push ranks.
use crate::shared::memory::{device::Device, load_timing, HostAllocation};
use crate::shared::fp8_load_quant::{LoadKernels, Target};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::{ffi::c_void, ops::Range, os::unix::fs::FileExt, path::PathBuf};

/// The most bytes one step stages (one or more whole scale units).
pub(crate) use cuteafd_loader::shared::dense_read::STEP_BYTES;
/// Preferred transform window; indivisible scale units still fit STEP_BYTES.
pub(crate) const TRANSFORM_BYTES: usize = cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES;
use cuteafd_loader::shared::dense_read::{ReadBatch, PINNED_FLOOR_BYTES};

/// Rows `rows` of a checkpoint tensor (rows of `row_bytes` at `base` in `path`),
/// bytes `cols` of each, packed row by row.
#[derive(Clone, Debug)]
pub(crate) struct Rect {
    pub name: String,
    pub path: PathBuf,
    pub base: u64,
    pub row_bytes: usize,
    pub rows: Range<usize>,
    pub cols: Range<usize>,
}

impl Rect {
    /// Header-only source selection; never opens the tensor payload.
    pub fn tensor(checkpoint: &cuteafd_loader::plan::Checkpoint, name: &str,
        rows: Option<Range<usize>>, cols: Option<Range<usize>>) -> Result<Self> {
        let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("checkpoint lacks {name}"))?;
        let tensor = &checkpoint.tensors[at];
        ensure!(checkpoint.weight_map.get(name) == Some(&tensor.shard),
            "tensor header differs from checkpoint index: {name}");
        let total = if tensor.meta.shape.len() >= 2 { tensor.meta.shape[0] } else { 1 };
        let bytes = usize::try_from(tensor.meta.byte_length)?;
        ensure!(total > 0 && bytes % total == 0, "{name}: invalid tensor row extent");
        let row_bytes = bytes / total;
        let (rows, cols) = (rows.unwrap_or(0..total), cols.unwrap_or(0..row_bytes));
        ensure!(!rows.is_empty() && !cols.is_empty() && rows.end <= total && cols.end <= row_bytes,
            "{name}: rectangle outside the tensor");
        Ok(Self { name: name.into(), path: checkpoint.snapshot.join(&tensor.shard),
            base: tensor.meta.byte_offset, row_bytes, rows, cols })
    }

    pub fn width(&self) -> usize { self.cols.len() }
    pub fn bytes(&self) -> usize { self.rows.len() * self.cols.len() }

    /// Reads this rectangle into `out` (exactly [`Self::bytes`] long).
    pub fn read(&self, out: &mut [u8]) -> Result<()> { self.read_cached(out, &mut None) }

    /// [`Self::read`], keeping the whole rows of a partial-width read in
    /// `cache` so the next rank's column window of the same rows reads nothing.
    pub fn read_cached(&self, out: &mut [u8], cache: &mut Option<(Rect, Vec<u8>)>) -> Result<()> {
        ensure!(out.len() == self.bytes() && !self.rows.is_empty() && !self.cols.is_empty()
            && self.cols.end <= self.row_bytes, "{}: rectangle outside its tensor", self.name);
        let file = std::fs::File::open(&self.path).with_context(|| format!("opening {}", self.name))?;
        let start = self.base + (self.rows.start * self.row_bytes) as u64;
        if self.width() == self.row_bytes {
            return load_timing::read(out.len(), || file.read_exact_at(out, start))
                .with_context(|| format!("reading {}", self.name));
        }
        let whole = Rect { cols: 0..self.row_bytes, ..self.clone() };
        let hit = cache.as_ref().is_some_and(|(rows, _)| rows.path == whole.path && rows.base == whole.base
            && rows.row_bytes == whole.row_bytes && rows.rows == whole.rows);
        if !hit {
            // Release the previous window before allocating its replacement.
            *cache = None;
            let batch = ReadBatch::new(self.rows.len(), self.row_bytes, self.width(), usize::MAX)?;
            let mut source = vec![0u8; batch.cached_bytes];
            load_timing::read(source.len(), || file.read_exact_at(&mut source, start))
                .with_context(|| format!("reading {}", self.name))?;
            *cache = Some((whole, source));
        }
        let (_, source) = cache.as_ref().context("row cache")?;
        for (row, chunk) in source.chunks_exact(self.row_bytes).zip(out.chunks_exact_mut(self.width())) {
            chunk.copy_from_slice(&row[self.cols.clone()]);
        }
        Ok(())
    }

    /// Rows `rows` (relative to this rectangle) of it.
    pub fn sub(&self, rows: Range<usize>) -> Self {
        Self { rows: self.rows.start + rows.start..self.rows.start + rows.end, ..self.clone() }
    }
}

/// A class-2 operand: rectangles landing unchanged at byte offsets of `buffer`
/// (a descriptor of an allocation the loader owns).
#[derive(Clone)]
pub(crate) struct Operand {
    pub device: i32,
    pub buffer: CuteafdDeviceBuffer,
    pub lands: Vec<(Rect, usize)>,
    pub bytes: usize,
}

/// One GPU transform of a step's staged bytes (byte offsets are into the staging).
#[derive(Clone)]
pub(crate) enum Job {
    /// Staged bytes `at..at + bytes` copied unchanged to `dst`.
    Upload { at: usize, bytes: usize, dst: CuteafdDeviceBuffer },
    /// Raw UE8M0 block grid expanded by the family's existing AOT program.
    BlockFp8Scales { family: &'static str, n_blocks: usize, k_blocks: usize, dst: CuteafdDeviceBuffer },
    /// Staged I64 IDs narrowed on GPU, with checked range status.
    NarrowI64I32 { dst: CuteafdDeviceBuffer },
    /// BF16 gate parameters converted by the checkpoint's host coefficient
    /// contract (no device transcendental rounding changes).
    ResidualGate { temperature: f32, alpha: CuteafdDeviceBuffer, beta: CuteafdDeviceBuffer },
    /// BF16 `at..at + bytes` widened to F32 into `dst`.
    Widen { at: usize, bytes: usize, dst: CuteafdDeviceBuffer },
    /// F32 `at..at + bytes` narrowed to BF16 (round to nearest even, NaN
    /// quieted: `OutputPrecision::Bf16::round`) into `dst`.
    NarrowF32Bf16 { at: usize, bytes: usize, dst: CuteafdDeviceBuffer },
    /// Staged BF16 rows `rows` (of `pitch` elements), columns `col0..col0 + cols`,
    /// quantized to FP8 into `target`.
    Quant { rows: Range<usize>, pitch: usize, col0: usize, cols: usize, layout: crate::shared::fp8_load_quant::Layout,
        rule: crate::shared::fp8_linear::Fp8Scales, target: Target },
    /// The staged `kv_b` rows of `heads` heads absorbed into `uk`/`uv`.
    Absorb { heads: usize, nope: usize, v: usize, lat: usize, uk: CuteafdDeviceBuffer, uv: CuteafdDeviceBuffer },
    /// The staged raw F32 `A_log` keeps `heads` active heads: its padding is +0
    /// and its live values finite.
    ALog { heads: usize },
    /// Staged BF16 rows `[n, k]` (n % 16 == 0) packed into W8A16 MMA fragment
    /// order under `rule` (the vocabulary head): `packed` (n * k bytes) and
    /// `scale` ([n, k / 128] F32), both offsets into the whole weight. Each
    /// 16-row tile is independent, so row windows reproduce the whole pack.
    W8a16Pack { n: usize, k: usize, rule: crate::shared::fp8_linear::Fp8Scales,
        packed: CuteafdDeviceBuffer, scale: CuteafdDeviceBuffer },
    /// Tile-aligned source rows of one native gate/up/down weight projection.
    NativeWindow { intermediate: u32, projection: u32, first: u32, rows: u32, dst: CuteafdDeviceBuffer },
    /// The staged FP8 `kv_b` rows of `heads` heads (E4M3 at `at..`, the first
    /// at checkpoint row `row0`) and the whole tensor's FP32 128x128 grid (at
    /// `grid_at..`, `grid_bytes`) split into the GLM FP8 programs' `uk`/`uk_scale`
    /// and `uv`/`uv_scale`: bytes unchanged, one grid entry per 64-wide tile
    /// (the mount loader's host rule).
    AbsorbFp8 { at: usize, grid_at: usize, grid_bytes: usize, row0: usize, heads: usize, nope: usize, v: usize,
        lat: usize, uk: CuteafdDeviceBuffer, uk_scale: CuteafdDeviceBuffer, uv: CuteafdDeviceBuffer,
        uv_scale: CuteafdDeviceBuffer },
    /// Staged BF16 rows `[rows, cols]` quantized to E4M3 with per-row x 128-K
    /// FP32 scales by the row kernel (`cuteafd_fp8_row_quant_rule`, `rule`)
    /// into `q` [rows, cols] and `scale` [rows, cols / 128].
    RowQuant { rows: usize, cols: usize, rule: crate::shared::fp8_linear::Fp8Scales, q: CuteafdDeviceBuffer,
        scale: CuteafdDeviceBuffer, block: bool },
    /// Staged BF16 `norm` (`at..at + bytes`) and `proj` (`with..with + bytes`)
    /// promoted to F32 and multiplied once per element into `dst` (the Kimi
    /// residual score weight).
    Product { at: usize, with: usize, bytes: usize, dst: CuteafdDeviceBuffer },
    /// Staged EXL3 trellis rows (`rows` K16 rows from local row `k0`, `tiles`
    /// 16x16 tiles of `tile_bytes` each, checkpoint order) reordered into the
    /// native shard `dst` of `k16` K16 rows; `input` selects the input-side
    /// layout. Every tile moves whole: no bit of the payload changes.
    TrellisTiles { rows: usize, k0: usize, tiles: usize, tile_bytes: usize, k16: usize, input: bool,
        dst: CuteafdDeviceBuffer },
    /// One raw UE8M0 rectangle, checked and coded directly from pinned staging.
    /// `rect` retains the checkpoint tensor and original row for refusals.
    ScaleCodes { rect: usize, at: usize, stored_rows: usize, start_row: usize, dst: CuteafdDeviceBuffer },
}

/// A device-to-device transform of landed class-2 bytes, run once every
/// operand of the rank has landed (push: at Done; mount: after its fill).
#[derive(Clone)]
pub(crate) enum Finish {
    /// V4.1 UE8M0 scales `[n/32, k/32]` packed for the `(k, n)` FP8 matrix kernel.
    V41Scales { capacity: u32, k: u32, n: u32, source: CuteafdDeviceBuffer, packed: CuteafdDeviceBuffer },
    /// One plane of a native expert packed from its landed raw image (the
    /// scale planes: their raw bytes sit apart from the weights in the file).
    NativePlane { intermediate: u32, plane: u32, source: CuteafdDeviceBuffer, dst: CuteafdDeviceBuffer },
    /// Native conservative row-norm bounds of a landed BF16 Markov projection.
    RowNorms { source: CuteafdDeviceBuffer, norms: CuteafdDeviceBuffer, rows: usize, cols: usize },
    /// Per-row scales gathered from a landed `grid` [*, grid_cols] (columns
    /// `col0..col0 + kb` of row `index[r]`, zero for `u32::MAX`) into
    /// `row_major` [rows, kb] and/or `k_major` [kb, rows].
    ScaleGather { grid: CuteafdDeviceBuffer, index: std::sync::Arc<Vec<u32>>, kb: usize, grid_cols: usize,
        col0: usize, row_major: Option<CuteafdDeviceBuffer>, k_major: Option<CuteafdDeviceBuffer> },
}

impl Finish {
    pub fn device(&self) -> i32 {
        match self { Finish::V41Scales { packed, .. } | Finish::NativePlane { dst: packed, .. } => packed.device_id,
            Finish::ScaleGather { grid, .. } => grid.device_id,
            Finish::RowNorms { norms, .. } => norms.device_id }
    }

    /// [`Self::run`], then drained; a failed drain quarantines the module so
    /// its operands (and any index scratch) outlive the unproved work.
    pub fn run_drained(&self, library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
        let ran = self.run(library, stream);
        // SAFETY: `stream` is this GPU's load stream, live for the call.
        if let Err(drain) = unsafe { library.cuda_stream_synchronize(stream) } {
            library.quarantine_module_after_failed_drain();
            let error = ran.err().unwrap_or_else(|| anyhow::anyhow!("load finish completion failed"));
            return Err(error.context(format!("load owners and native module quarantined after a failed drain: {drain}")));
        }
        ran
    }

    /// On the current device (`device`), on `stream` (the caller drains).
    pub fn run(&self, library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
        match *self {
            Finish::V41Scales { capacity, k, n, source, packed } => {
                ensure!(source.device_id == packed.device_id, "scale pack across GPUs");
                let kernel = library.v41_fp8_matrix_kernel(capacity, k, n)?;
                // SAFETY: both buffers are live operand allocations on the current
                // device; the kernel checks their extents, the caller drains.
                unsafe { kernel.pack_scales(source, packed, stream) }
            }
            Finish::NativePlane { intermediate, plane, source, dst } => {
                ensure!(source.device_id == dst.device_id, "plane pack across GPUs");
                // SAFETY: both buffers are live operand allocations on the current
                // device; the native entry point checks extents, the caller drains.
                unsafe { library.v41_pack_plane_async(source, dst, intermediate, plane, stream) }
            }
            Finish::RowNorms { source, norms, rows, cols } => {
                validate_row_norms(source, norms, rows, cols)?;
                // SAFETY: checked BF16 source and FP32 output extents share the
                // current GPU; owners survive the caller's completion drain.
                unsafe {
                    match cols {
                        256 => library.glmf_dspark_markov_norms(source.ptr, norms.ptr, rows, cols, stream),
                        512 => library.ml4_dspark_markov_norms(source.ptr, norms.ptr, rows, cols, stream),
                        _ => unreachable!("checked native row width"),
                    }
                }
            }
            Finish::ScaleGather { grid, ref index, kb, grid_cols, col0, row_major, k_major } => {
                let rows = index.len();
                ensure!(col0 + kb <= grid_cols && rows > 0
                    && row_major.is_none_or(|b| b.bytes >= rows * kb * 4 && b.device_id == grid.device_id)
                    && k_major.is_none_or(|b| b.bytes >= rows * kb * 4 && b.device_id == grid.device_id)
                    && index.iter().all(|&r| r == u32::MAX || (r as usize + 1) * grid_cols * 4 <= grid.bytes),
                    "scale gather outside its operands");
                let staged = crate::shared::memory::DeviceAllocation::new(library, rows * 4)?;
                let bytes: Vec<u8> = index.iter().flat_map(|r| r.to_le_bytes()).collect();
                library.copy_h2d(staged.buffer, &bytes)?;
                // SAFETY: every buffer is live on the current device and sized as
                // checked above; drained before the index allocation drops.
                unsafe {
                    library.load_scale_gather(grid.ptr, staged.buffer.ptr, row_major.map_or(std::ptr::null_mut(), |b| b.ptr),
                        k_major.map_or(std::ptr::null_mut(), |b| b.ptr), rows, kb, grid_cols, col0, stream)?;
                    library.cuda_stream_synchronize(stream)
                }
            }
        }
    }
}

fn validate_row_norms(source: CuteafdDeviceBuffer, norms: CuteafdDeviceBuffer, rows: usize, cols: usize) -> Result<()> {
    let source_bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(2)).context("row norms source overflow")?;
    let output_bytes = rows.checked_mul(4).context("row norms output overflow")?;
    let source_end = (source.ptr as usize).checked_add(source_bytes).context("row norms source pointer overflow")?;
    let output_end = (norms.ptr as usize).checked_add(output_bytes).context("row norms output pointer overflow")?;
    ensure!(rows > 0 && matches!(cols, 256 | 512) && source_bytes <= source.bytes && output_bytes <= norms.bytes
        && source.device_id == norms.device_id && !source.ptr.is_null() && !norms.ptr.is_null()
        && ((norms.ptr as usize) >= source_end || (source.ptr as usize) >= output_end),
        "row norms outside their operands, overlapping, or unsupported width");
    Ok(())
}

#[cfg(test)]
mod finish_tests {
    use super::*;

    #[test]
    fn row_norm_finish_records_destination_device_and_checks_native_extents() -> Result<()> {
        let mut source_bytes = vec![0u16; 2 * 512];
        let mut output = vec![0f32; 2];
        let source = CuteafdDeviceBuffer { ptr: source_bytes.as_mut_ptr().cast(), bytes: source_bytes.len() * 2, device_id: 3, ..Default::default() };
        let norms = CuteafdDeviceBuffer { ptr: output.as_mut_ptr().cast(), bytes: output.len() * 4, device_id: 3, ..Default::default() };
        for cols in [256, 512] {
            validate_row_norms(source, norms, 2, cols)?;
            let (_, records) = recording(|| {
                assert!(recorded(|r| r.finishes.push(Finish::RowNorms { source, norms, rows: 2, cols })));
                Ok(())
            })?;
            assert_eq!(records.summary(), (0, 0, 0, 0, 0, 1));
            assert_eq!(records.finishes[0].device(), 3);
            assert!(matches!(records.finishes[0], Finish::RowNorms { rows: 2, cols: c, .. } if c == cols));
            assert_eq!(output, [0., 0.]);
        }
        for (source, norms, rows, cols) in [
            (source, norms, 0, 256), (source, norms, 2, 128), (source, norms, 3, 512),
            (source, CuteafdDeviceBuffer { bytes: 4, ..norms }, 2, 256),
            (source, CuteafdDeviceBuffer { device_id: 2, ..norms }, 2, 256),
            (source, CuteafdDeviceBuffer { ptr: std::ptr::null_mut(), ..norms }, 2, 256),
            (source, CuteafdDeviceBuffer { ptr: source.ptr, ..norms }, 2, 256),
            (source, norms, usize::MAX, 512),
        ] {
            assert!(validate_row_norms(source, norms, rows, cols).is_err());
        }
        Ok(())
    }
}

/// A class-3 transform: `rects` staged back to back, then `jobs` on `device`
/// (descriptors of allocations the loader owns).
#[derive(Clone)]
pub(crate) struct Step {
    pub device: i32,
    pub rects: Vec<Rect>,
    pub jobs: Vec<Job>,
}

impl Step {
    pub fn bytes(&self) -> usize { self.rects.iter().map(Rect::bytes).sum() }

    /// Runs the jobs on the current device (`device`) from `staged`; jobs that
    /// gather (expert and W8A16 packs) first copy the step into `scratch`, a
    /// [`STEP_BYTES`] device buffer allocated on first use.
    pub fn run<'l>(&self, kernels: &mut LoadKernels<'_>, library: &'l NativeLibrary, staged: &[u8], stream: *mut c_void,
        scratch: &mut Option<crate::shared::memory::DeviceAllocation<'l>>) -> Result<()> {
        let ran = self.run_jobs(kernels, library, staged, stream, scratch);
        if let Err(error) = ran {
            // A failed job may leave queued work reading the staging or scratch:
            // drain it, or quarantine so every owner outlives that work.
            // SAFETY: `stream` is this GPU's load stream, live for the call.
            if let Err(drain) = unsafe { library.cuda_stream_synchronize(stream) } {
                library.quarantine_module_after_failed_drain();
                return Err(error.context(format!("load owners and native module quarantined after a failed drain: {drain}")));
            }
            return Err(error);
        }
        Ok(())
    }

    fn run_jobs<'l>(&self, kernels: &mut LoadKernels<'_>, library: &'l NativeLibrary, staged: &[u8], stream: *mut c_void,
        scratch: &mut Option<crate::shared::memory::DeviceAllocation<'l>>) -> Result<()> {
        ensure!(staged.len() >= self.bytes(), "step staging is short");
        let mut on_device = None;
        let mut device_copy = |scratch: &mut Option<crate::shared::memory::DeviceAllocation<'l>>| -> Result<*const u8> {
            if let Some(ptr) = on_device { return Ok(ptr); }
            if scratch.is_none() {
                *scratch = Some(crate::shared::memory::DeviceAllocation::named(library, STEP_BYTES,
                    "staging/coordinator-dense-device")?);
            }
            let buffer = scratch.as_ref().context("dense device scratch")?.buffer;
            load_timing::upload(self.bytes(), || library.copy_h2d(buffer, &staged[..self.bytes()]))?;
            on_device = Some(buffer.ptr.cast::<u8>().cast_const());
            Ok(buffer.ptr.cast::<u8>().cast_const())
        };
        for job in &self.jobs {
            match *job {
                Job::ResidualGate { temperature, alpha, beta } => {
                    let (a, b) = residual_gate_bytes(&staged[..self.bytes()], temperature)?;
                    ensure!(a.len() == alpha.bytes && b.len() == beta.bytes,
                        "residual gate differs from admitted operands");
                    load_timing::upload(a.len() + b.len(), || {
                        library.copy_h2d(alpha, &a)?;
                        library.copy_h2d(beta, &b)
                    })?;
                }
                Job::Upload { at, bytes, dst } => {
                    ensure!(at + bytes <= staged.len() && bytes <= dst.bytes, "upload outside its step");
                    load_timing::upload(bytes, || library.copy_h2d(CuteafdDeviceBuffer { bytes, ..dst },
                        &staged[at..at + bytes]))?;
                }
                Job::BlockFp8Scales { family, n_blocks, k_blocks, dst } => {
                    let bytes = n_blocks.checked_mul(k_blocks).context("block scale grid overflow")?;
                    ensure!(bytes == self.bytes() && bytes.checked_mul(512).is_some_and(|size| size <= dst.bytes),
                        "block scale expansion outside its operands");
                    let source = device_copy(scratch)?;
                    let programs = library.programs()?;
                    let program = programs.program(&format!("{family}_block_fp8_scale_prep"), &["scale", "scale_mma"])?;
                    // SAFETY: the bounded source and expanded destination are live
                    // on this device until the completion drain below.
                    unsafe {
                        program.launch(&[source.cast_mut().cast(), dst.ptr],
                            &[cuteafd_ffi::programs::Scalar::I32(i32::try_from(n_blocks)?),
                              cuteafd_ffi::programs::Scalar::I32(i32::try_from(k_blocks)?)], stream)?;
                        library.cuda_stream_synchronize(stream)?;
                    }
                }
                Job::NarrowI64I32 { dst } => {
                    let bytes = self.bytes();
                    ensure!(bytes > 0 && bytes % 8 == 0 && dst.bytes >= bytes / 2,
                        "I64 narrowing outside its operands");
                    let status = HostAllocation::named(library, 4, "staging/coordinator-dense-status")?;
                    let alias = library.cuda_host_buffer_device_alias(status.buffer)?;
                    // SAFETY: pinned staging, aligned status and I32 destination
                    // remain live through the drain before status is inspected.
                    unsafe {
                        library.load_narrow_i64_i32(staged.as_ptr().cast(), dst.ptr, bytes / 8, alias.ptr, stream)?;
                        if let Err(error) = library.cuda_stream_synchronize(stream) {
                            // The status is job-local: quarantine before it drops,
                            // rather than waiting for Step::run's outer drain.
                            library.quarantine_module_after_failed_drain();
                            return Err(error);
                        }
                    }
                    ensure!(status.bytes() == [0; 4], "checkpoint I64 router ID exceeds I32 range");
                }
                Job::NarrowF32Bf16 { at, bytes, dst } => {
                    ensure!(at + bytes <= staged.len() && bytes % 4 == 0 && dst.bytes >= bytes / 2,
                        "narrowing outside its step");
                    // SAFETY: `staged` is pinned, device-mapped memory the caller keeps
                    // alive until the drain below; `dst` holds bytes / 2.
                    load_timing::quantize(bytes, || unsafe {
                        library.load_narrow_f32_bf16(staged[at..].as_ptr().cast(), dst.ptr, bytes / 4, stream)?;
                        library.cuda_stream_synchronize(stream)
                    })?;
                }
                Job::Widen { at, bytes, dst } => {
                    ensure!(at + bytes <= staged.len(), "widening outside its step");
                    load_timing::quantize(bytes, || kernels.widen_from(&staged[at..at + bytes], dst, stream))?;
                }
                Job::Quant { ref rows, pitch, col0, cols, layout, rule, target } => {
                    load_timing::quantize(rows.len() * cols * 2, || kernels.quantize_from(staged, rows.clone(),
                        pitch, col0, cols, layout, rule, target, stream))?;
                }
                Job::Absorb { heads, nope, v, lat, uk, uv } => {
                    load_timing::quantize(heads * (nope + v) * lat * 2, || kernels.absorb_from(staged, heads, 0,
                        heads, nope, v, lat, uk, uv, stream))?;
                }
                Job::AbsorbFp8 { at, grid_at, grid_bytes, row0, heads, nope, v, lat, uk, uk_scale, uv, uv_scale } => {
                    let rows = heads * (nope + v);
                    ensure!(heads > 0 && nope % 64 == 0 && lat % 128 == 0 && at + rows * lat <= staged.len()
                        && grid_at + grid_bytes <= staged.len()
                        && grid_bytes >= (row0 + rows).div_ceil(128) * (lat / 128) * 4
                        && uk.bytes >= heads * lat * nope && uk_scale.bytes >= heads * lat * (nope / 64) * 4
                        && uv.bytes >= heads * v * lat && uv_scale.bytes >= heads * v * (lat / 64) * 4,
                        "FP8 kv_b split outside its step or operands");
                    // SAFETY: `staged` is pinned, device-mapped memory the caller keeps
                    // alive until the drain below; the four operands hold `heads` heads.
                    load_timing::quantize(rows * lat, || unsafe {
                        library.load_kvb_split_fp8(staged[at..].as_ptr().cast(), staged[grid_at..].as_ptr().cast(),
                            uk.ptr, uk_scale.ptr, uv.ptr, uv_scale.ptr, row0, heads, nope, v, lat, stream)?;
                        library.cuda_stream_synchronize(stream)
                    })?;
                }
                Job::ALog { heads } => {
                    cuteafd_loader::kda::narrow_a_log(&staged[..self.bytes()], heads)?;
                }
                Job::W8a16Pack { n, k, rule, packed, scale } => {
                    ensure!(n * k * 2 <= self.bytes() && packed.bytes >= n * k && scale.bytes >= n * k / 128 * 4,
                        "W8A16 pack outside its step or operands");
                    let source = device_copy(scratch)?;
                    // SAFETY: the step's bytes are in the live device scratch; both
                    // outputs are live operands; the stream drains before return.
                    load_timing::quantize(n * k * 2, || unsafe {
                        library.fp8_w8a16_pack(source.cast(), packed.ptr, scale.ptr, n, k, rule.code(), stream)?;
                        library.cuda_stream_synchronize(stream)
                    })?;
                }
                Job::RowQuant { rows, cols, rule, q, scale, block } => {
                    // Per-row x 128-K scales [rows, cols/128], or with `block` 128x128
                    // block scales [ceil(rows/128), ceil(cols/128)] (`cuteafd_fp8_block_quant_rule`).
                    let scale_bytes = if block { rows.div_ceil(128) * cols.div_ceil(128) * 4 } else { rows * cols / 128 * 4 };
                    ensure!(rows * cols * 2 <= self.bytes() && (block || cols % 128 == 0) && q.bytes >= rows * cols
                        && scale.bytes >= scale_bytes, "row quantization outside its step or operands");
                    let source = device_copy(scratch)?;
                    // SAFETY: the staged rows are in the live device scratch; both outputs
                    // are live operands; the stream drains before return.
                    load_timing::quantize(rows * cols * 2, || unsafe {
                        library.fp8_quant_rule(source.cast(), q.ptr, scale.ptr, rows, cols, !block, rule.code(), stream)?;
                        library.cuda_stream_synchronize(stream)
                    })?;
                }
                Job::ScaleCodes { rect, at, stored_rows, start_row, dst } => {
                    let source = self.rects.get(rect).context("scale-code source rectangle absent")?;
                    let bytes = staged.get(at..at.checked_add(source.bytes()).context("scale-code source overflow")?)
                        .context("scale-code source outside staging")?;
                    let target = crate::shared::fp8_load_quant::ScaleCodeTarget {
                        source: cuteafd_loader::formats::scale_codes::Rows { tensor: &source.name, bytes,
                            rows: source.rows.len(), pitch: source.width(), columns: source.width(),
                            first_row: source.rows.start },
                        destination: dst, stored_rows, start_row,
                    };
                    load_timing::quantize(source.bytes(), || kernels.scale_codes_from(&[target], stream))?;
                }
                Job::Product { at, with, bytes, dst } => {
                    ensure!(at + bytes <= staged.len() && with + bytes <= staged.len() && bytes % 2 == 0
                        && dst.bytes >= bytes * 2, "product outside its step or operand");
                    // SAFETY: both staged ranges lie in pinned, device-mapped memory the
                    // caller keeps alive until the drain below; `dst` holds bytes * 2.
                    load_timing::quantize(bytes * 2, || unsafe {
                        library.load_bf16_product_f32(staged[at..].as_ptr().cast(), staged[with..].as_ptr().cast(),
                            dst.ptr, bytes / 2, stream)?;
                        library.cuda_stream_synchronize(stream)
                    })?;
                }
                Job::TrellisTiles { rows, k0, tiles, tile_bytes, k16, input, dst } => {
                    let bytes = rows * tiles * tile_bytes;
                    ensure!(bytes <= self.bytes() && k0 + rows <= k16 && dst.bytes >= k16 * tiles * tile_bytes,
                        "trellis tiles outside their step or shard");
                    let source = device_copy(scratch)?;
                    // SAFETY: the staged rows are in the live device scratch and `dst`
                    // holds the whole shard (checked above); drained before return.
                    load_timing::quantize(bytes, || unsafe {
                        library.load_trellis_tiles(source.cast(), dst.ptr, rows, k0, tiles, tile_bytes,
                            i32::from(input), k16, stream)?;
                        library.cuda_stream_synchronize(stream)
                    })?;
                }
                Job::NativeWindow { intermediate, projection, first, rows, dst } => {
                    let base = device_copy(scratch)?;
                    let source = CuteafdDeviceBuffer { ptr: base.cast_mut().cast(), bytes: self.bytes(), ..dst };
                    // SAFETY: source is the live step scratch, dst the owned plane;
                    // native checks tile geometry and extents before launch.
                    load_timing::quantize(self.bytes(), || unsafe {
                        library.v41_pack_window_async(source, dst, intermediate, projection, first, rows, stream)?;
                        library.cuda_stream_synchronize(stream)
                    })?;
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn residual_gate_bytes(bytes: &[u8], temperature: f32) -> Result<(Vec<u8>, Vec<u8>)> {
    ensure!(!bytes.is_empty() && bytes.len() % 2 == 0 && temperature.is_finite() && temperature > 0.0,
        "invalid residual gate parameters");
    let gate = bytes.chunks_exact(2).map(|b|
        f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect::<Vec<_>>();
    let coefficients = cuteafd_loader::families::muse_spark::reference::residual_gate_coefficients(&gate, temperature);
    Ok((coefficients.iter().flat_map(|(a, _)| a.to_le_bytes()).collect(),
        coefficients.iter().flat_map(|(_, b)| b.to_le_bytes()).collect()))
}

/// `rects` (rows of equal width, back to back) as steps of at most
/// [`STEP_BYTES`] in whole `unit`-row groups: each step's rectangles and the
/// row range of the whole it covers.
///
/// A step also ends at every rectangle boundary that falls on a whole unit,
/// so a step reads one tensor's contiguous rows: delivered in file order, a
/// step that mixed tensors (q rows, then k rows far later) would hold every
/// later step open until its last tensor arrived, and staging would grow
/// with the tensor count rather than the source count.
pub(crate) fn windows(rects: &[Rect], unit: usize) -> Result<Vec<(Vec<Rect>, Range<usize>)>> {
    let total: usize = rects.iter().map(|r| r.rows.len()).sum();
    ensure!(unit > 0 && total % unit == 0, "staged rows are not whole {unit}-row units");
    windows_with_tail(rects, unit)
}

/// [`windows`] whose last unit may be partial (block quantization of a matrix
/// whose rows are not a multiple of the block): every window but the last
/// covers whole units.
pub(crate) fn windows_with_tail(rects: &[Rect], unit: usize) -> Result<Vec<(Vec<Rect>, Range<usize>)>> {
    let width = rects.first().context("no rows to stage")?.width();
    ensure!(rects.iter().all(|r| r.width() == width), "staged rows differ in width");
    let total: usize = rects.iter().map(|r| r.rows.len()).sum();
    ensure!(unit > 0 && total > 0, "no staged rows");
    let per = (TRANSFORM_BYTES / width / unit).max(1) * unit;
    // Unit-aligned rectangle boundaries (row offsets of the whole).
    let mut cuts = Vec::new();
    let mut at = 0;
    for rect in rects {
        at += rect.rows.len();
        if at % unit == 0 && at < total { cuts.push(at); }
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < total {
        let next_cut = cuts[cuts.partition_point(|&c| c <= start)..].first().copied().unwrap_or(total);
        let end = (start + per).min(total).min(next_cut);
        let (mut parts, mut at) = (Vec::new(), 0);
        for rect in rects {
            let n = rect.rows.len();
            let (lo, hi) = (start.max(at), end.min(at + n));
            if lo < hi { parts.push(rect.sub(lo - at..hi - at)); }
            at += n;
        }
        out.push((parts, start..end));
        start = end;
    }
    Ok(out)
}

/// Push-mode loader output: class-2 operands and class-3 steps, in loader order.
#[derive(Default)]
pub(crate) struct DenseSink {
    operands: std::cell::RefCell<Vec<Operand>>,
    steps: std::cell::RefCell<Vec<Step>>,
    finishes: std::cell::RefCell<Vec<Finish>>,
    peer_copies: std::cell::RefCell<Vec<PeerCopy>>,
}

impl DenseSink {
    pub fn operand(&self, operand: Operand) -> Result<()> {
        ensure!(operand.bytes <= operand.buffer.bytes
            && operand.lands.iter().all(|(r, at)| at + r.bytes() <= operand.bytes), "pushed landing outside its operand");
        self.operands.borrow_mut().push(operand);
        Ok(())
    }
    pub fn step(&self, step: Step) -> Result<()> {
        ensure!(step.bytes() > 0 && step.bytes() <= STEP_BYTES,
            "a step stages {} bytes, beyond the {STEP_BYTES}-byte staging buffer", step.bytes());
        self.steps.borrow_mut().push(step);
        Ok(())
    }
    pub fn finish(&self, finish: Finish) { self.finishes.borrow_mut().push(finish); }
    /// The recorded operands and steps (leaves the sink empty).
    pub fn take(&self) -> (Vec<Operand>, Vec<Step>) {
        assert!(self.finishes.borrow().is_empty() && self.peer_copies.borrow().is_empty(),
            "dense load has deferred work; consume take_all");
        (self.operands.take(), self.steps.take())
    }
    /// Every record, finishes included (leaves the sink empty).
    pub fn take_all(&self) -> Records {
        Records { operands: self.operands.take(), steps: self.steps.take(), finishes: self.finishes.take(),
            peer_copies: self.peer_copies.take() }
    }
}

thread_local! {
    /// Records of the load running on this thread under [`recording`].
    static RECORDING: std::cell::RefCell<Option<Records>> = const { std::cell::RefCell::new(None) };
}

/// Runs `load` with every mount-mode [`DenseLoad`] on this thread recording
/// instead of filling (allocations happen exactly as through the mount), and
/// returns its value with the records in loader order.
pub(crate) fn recording<T>(load: impl FnOnce() -> Result<T>) -> Result<(T, Records)> {
    RECORDING.with(|r| -> Result<()> {
        ensure!(r.borrow().is_none(), "nested dense recording");
        *r.borrow_mut() = Some(Records::default());
        Ok(())
    })?;
    let value = load();
    let records = RECORDING.with(|r| r.borrow_mut().take()).unwrap_or_default();
    Ok((value?, records))
}

/// A [`recording`] over straight-line loader code: records from [`Self::start`]
/// until [`Self::finish`] (dropping it unfinished discards the records).
pub(crate) struct Recorder(());

impl Recorder {
    /// `None` when `enabled` is false (every load fills through the mount).
    pub fn start(enabled: bool) -> Result<Option<Self>> {
        if !enabled { return Ok(None); }
        RECORDING.with(|r| -> Result<()> {
            ensure!(r.borrow().is_none(), "nested dense recording");
            *r.borrow_mut() = Some(Records::default());
            Ok(())
        })?;
        Ok(Some(Self(())))
    }
    pub fn finish(self) -> Records {
        let records = RECORDING.with(|r| r.borrow_mut().take()).unwrap_or_default();
        std::mem::forget(self);
        records
    }
}

impl Drop for Recorder {
    fn drop(&mut self) { RECORDING.with(|r| r.borrow_mut().take()); }
}

fn recorded(record: impl FnOnce(&mut Records)) -> bool {
    RECORDING.with(|r| match r.borrow_mut().as_mut() { Some(records) => { record(records); true } None => false })
}

/// Whether a [`recording`] is active on this thread.
pub(crate) fn is_recording() -> bool { RECORDING.with(|r| r.borrow().is_some()) }

/// A V4.1 scale pack of landed scales: recorded as a [`Finish`] under
/// [`recording`], else launched now on `stream` (the caller drains).
///
/// # Safety
/// As [`cuteafd_ffi::V41Fp8Kernel::pack_scales`].
pub(crate) unsafe fn v41_pack_scales(kernel: &cuteafd_ffi::V41Fp8Kernel<'_>, source: CuteafdDeviceBuffer,
    packed: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
    let info = kernel.info();
    if recorded(|r| r.finishes.push(Finish::V41Scales { capacity: info.capacity_rows, k: info.input_dim,
        n: info.output_dim, source, packed })) { return Ok(()); }
    // SAFETY: the caller's contract.
    unsafe { kernel.pack_scales(source, packed, stream) }
}

/// A copy whose source may belong to another push rank. Addresses, not owners,
/// cross the planning thread; the loader retains both allocations until commit.
#[derive(Clone, Debug)]
pub(crate) struct PeerCopy {
    source: usize,
    destination: usize,
    bytes: usize,
    pub source_device: i32,
    pub destination_device: i32,
    source_flags: u64,
    destination_flags: u64,
}

impl PeerCopy {
    pub fn new(source: CuteafdDeviceBuffer, destination: CuteafdDeviceBuffer, bytes: usize) -> Result<Self> {
        ensure!(bytes > 0 && bytes <= source.bytes && bytes <= destination.bytes
            && !source.ptr.is_null() && !destination.ptr.is_null(), "peer load copy exceeds its owners");
        Ok(Self { source: source.ptr as usize, destination: destination.ptr as usize, bytes,
            source_device: source.device_id, destination_device: destination.device_id,
            source_flags: source.flags, destination_flags: destination.flags })
    }

    fn run_drained(&self, library: &NativeLibrary) -> Result<()> {
        Device { library, id: self.destination_device }.run(|| {
            let source = CuteafdDeviceBuffer { ptr: self.source as *mut c_void, bytes: self.bytes,
                device_id: self.source_device, flags: self.source_flags };
            let destination = CuteafdDeviceBuffer { ptr: self.destination as *mut c_void, bytes: self.bytes,
                device_id: self.destination_device, flags: self.destination_flags };
            // SAFETY: the loader retains both admitted regions through group commit
            // and this drain; no consumer reads either unpublished allocation yet.
            let copied = unsafe { library.copy_d2d_async(destination, source, self.bytes, std::ptr::null_mut()) };
            // SAFETY: the destination's default stream issued this copy.
            if let Err(drain) = unsafe { library.cuda_stream_synchronize(std::ptr::null_mut()) } {
                library.quarantine_module_after_failed_drain();
                return Err(copied.err().unwrap_or_else(|| anyhow::anyhow!("peer load copy completion failed"))
                    .context(format!("load owners quarantined after peer-copy drain failed: {drain}")));
            }
            copied
        })
    }
}

/// Runs only after all source ranks have committed, or after a complete mount fill.
pub(crate) fn run_peer_copies(library: &NativeLibrary, copies: &[PeerCopy]) -> Result<()> {
    for copy in copies { copy.run_drained(library)?; }
    Ok(())
}

/// Preserves failed/uncommitted delivery without touching either copy owner.
pub(crate) fn committed_peer_copies<T>(library: &NativeLibrary, delivered: Result<Option<T>>,
    copies: &[PeerCopy]) -> Result<Option<T>> {
    let delivered = delivered?;
    if delivered.is_some() { run_peer_copies(library, copies)?; }
    Ok(delivered)
}

/// One load's recorded operands, staged transforms, rank finishes and group copies.
#[derive(Clone, Default)]
pub(crate) struct Records {
    pub operands: Vec<Operand>,
    pub steps: Vec<Step>,
    pub finishes: Vec<Finish>,
    pub peer_copies: Vec<PeerCopy>,
}

impl Records {
    /// `other`'s records after these (loader order).
    pub fn extend(&mut self, other: Records) {
        self.operands.extend(other.operands);
        self.steps.extend(other.steps);
        self.finishes.extend(other.finishes);
        self.peer_copies.extend(other.peer_copies);
    }

    /// Runs every finish on its GPU (after all operands landed), drained.
    pub fn run_finishes(&self, library: &NativeLibrary, stream: impl Fn(i32) -> *mut c_void) -> Result<()> {
        for finish in &self.finishes {
            let device = Device { library, id: finish.device() };
            device.run(|| finish.run_drained(library, stream(finish.device())))?;
        }
        Ok(())
    }

    /// Bytes and counts per class: (class-2 bytes, operands, class-3 bytes, steps, largest step, finishes).
    pub fn summary(&self) -> (usize, usize, usize, usize, usize, usize) {
        (self.operands.iter().map(|o| o.bytes).sum(), self.operands.len(),
            self.steps.iter().map(Step::bytes).sum(), self.steps.len(),
            self.steps.iter().map(Step::bytes).max().unwrap_or(0), self.finishes.len())
    }
}

/// How one load fills its dense records: through the mount as each is made
/// (bounded pinned staging), or recorded for the coordinator's push ranks
/// (the same records then fill through the mount if the group falls back).
pub(crate) struct DenseLoad<'a> {
    library: &'a NativeLibrary,
    mount: std::cell::RefCell<Option<Mount<'a>>>,
    pub sink: Option<DenseSink>,
}

impl<'a> DenseLoad<'a> {
    pub fn mount(library: &'a NativeLibrary) -> Self { Self { library, mount: Default::default(), sink: None } }
    pub fn push(library: &'a NativeLibrary) -> Self {
        Self { library, mount: Default::default(), sink: Some(Default::default()) }
    }
    pub fn pushing(&self) -> bool { self.sink.is_some() }

    fn runner(&self) -> Result<std::cell::RefMut<'_, Mount<'a>>> {
        let mut mount = self.mount.borrow_mut();
        if mount.is_none() { *mount = Some(Mount::new(self.library)?); }
        Ok(std::cell::RefMut::map(mount, |m| m.as_mut().expect("mount runner")))
    }

    /// Records `operand`, or fills it on its GPU now.
    pub fn operand(&self, operand: Operand) -> Result<()> {
        if let Some(sink) = &self.sink { return sink.operand(operand); }
        ensure!(operand.bytes <= operand.buffer.bytes
            && operand.lands.iter().all(|(r, at)| at + r.bytes() <= operand.bytes), "landing outside its operand");
        let mut pending = Some(operand);
        if recorded(|r| r.operands.push(pending.take().expect("one operand"))) { return Ok(()); }
        let operand = pending.take().context("operand")?;
        let device = Device { library: self.library, id: operand.device };
        device.run(|| self.runner()?.operand(&operand))
    }

    /// Records `step`, or stages and runs it on its GPU now (`stream`: a
    /// stream of that GPU; null is the legacy default stream).
    pub fn step(&self, step: Step, stream: *mut c_void) -> Result<()> {
        if let Some(sink) = &self.sink { return sink.step(step); }
        ensure!(step.bytes() > 0 && step.bytes() <= STEP_BYTES,
            "a step stages {} bytes, beyond the {STEP_BYTES}-byte staging buffer", step.bytes());
        let mut pending = Some(step);
        if recorded(|r| r.steps.push(pending.take().expect("one step"))) { return Ok(()); }
        let step = pending.take().context("step")?;
        let device = Device { library: self.library, id: step.device };
        device.run(|| self.runner()?.step(&step, stream))
    }

    /// Records `finish`, or runs it now (its operands were filled before it).
    pub fn finish(&self, finish: Finish, stream: *mut c_void) -> Result<()> {
        if let Some(sink) = &self.sink { sink.finish(finish); return Ok(()); }
        if recorded(|r| r.finishes.push(finish.clone())) { return Ok(()); }
        let device = Device { library: self.library, id: finish.device() };
        device.run(|| finish.run_drained(self.library, stream))
    }

    /// Mount copies drain immediately; recorded copies wait for whole-group commit.
    pub fn peer_copy(&self, source: CuteafdDeviceBuffer, destination: CuteafdDeviceBuffer, bytes: usize) -> Result<()> {
        let copy = PeerCopy::new(source, destination, bytes)?;
        if let Some(sink) = &self.sink { sink.peer_copies.borrow_mut().push(copy); return Ok(()); }
        if recorded(|r| r.peer_copies.push(copy.clone())) { return Ok(()); }
        copy.run_drained(self.library)
    }

    /// Releases the mount runner's pinned staging (after the load).
    pub fn release(&self) { drop(self.mount.borrow_mut().take()); }
}

/// Mount-path runner: reads into operands, or into one reused pinned buffer
/// for a step's jobs.
pub(crate) struct Mount<'a> {
    library: &'a NativeLibrary,
    /// The FP8/widen/absorb load kernels, bound on the first step that needs them.
    kernels: Option<LoadKernels<'a>>,
    staging: Option<HostAllocation<'a>>,
    /// The whole rows of the last partial-width read (column shares of one tensor).
    rows: Option<(Rect, Vec<u8>)>,
    /// Per-GPU device scratch of gathering jobs.
    scratch: std::collections::HashMap<i32, Option<crate::shared::memory::DeviceAllocation<'a>>>,
}

impl<'a> Mount<'a> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self { library, kernels: None, staging: None, rows: None, scratch: Default::default() })
    }

    /// Fills `operand` on the current device, rectangle by rectangle through
    /// pinned staging of at most [`STEP_BYTES`] (row windows of larger ones).
    pub fn operand(&mut self, operand: &Operand) -> Result<()> {
        for (rect, at) in &operand.lands {
            let per = ReadBatch::new(rect.rows.len(), rect.row_bytes, rect.width(), STEP_BYTES)?.rows;
            let mut first = 0;
            while first < rect.rows.len() {
                let part = rect.sub(first..(first + per).min(rect.rows.len()));
                let bytes = part.bytes();
                self.grow(bytes)?;
                let staging = self.staging.as_mut().context("mount staging")?;
                part.read_cached(&mut staging.bytes_mut()[..bytes], &mut self.rows)?;
                let offset = at + first * rect.width();
                // SAFETY: `offset + bytes` lies inside the operand (checked when recorded).
                let destination = CuteafdDeviceBuffer { ptr: unsafe { operand.buffer.ptr.cast::<u8>().add(offset) }.cast(),
                    bytes, ..operand.buffer };
                load_timing::upload(bytes, || self.library.copy_h2d(destination, &staging.bytes()[..bytes]))?;
                first += part.rows.len();
            }
        }
        Ok(())
    }

    fn grow(&mut self, bytes: usize) -> Result<()> {
        if self.staging.as_ref().is_none_or(|s| s.buffer.bytes < bytes) {
            drop(self.staging.take());
            self.staging = Some(HostAllocation::named(self.library, bytes.max(PINNED_FLOOR_BYTES), "staging/coordinator-dense")?);
        }
        Ok(())
    }

    /// Stages and runs `step` on the current device (`stream`).
    pub fn step(&mut self, step: &Step, stream: *mut c_void) -> Result<()> {
        let bytes = step.bytes();
        ensure!(bytes > 0 && bytes <= STEP_BYTES, "a step stages {bytes} bytes, beyond the {STEP_BYTES}-byte bound");
        self.grow(bytes)?;
        let staging = self.staging.as_mut().context("mount staging")?;
        let mut at = 0;
        for rect in &step.rects {
            rect.read_cached(&mut staging.bytes_mut()[at..at + rect.bytes()], &mut self.rows)?;
            at += rect.bytes();
        }
        if self.kernels.is_none() { self.kernels = Some(LoadKernels::new(self.library)?); }
        let kernels = self.kernels.as_mut().context("load kernels")?;
        let scratch = self.scratch.entry(step.device).or_default();
        step.run(kernels, self.library, &staging.bytes()[..bytes], stream, scratch)
    }
}

/// Fills recorded operands and runs recorded steps through the mount (the
/// whole push group fell back), each on its GPU; `stream` names a GPU's stream.
pub(crate) fn fill_by_mount(library: &NativeLibrary, operands: &[Operand], steps: &[Step],
    stream: impl Fn(i32) -> *mut c_void) -> Result<()> {
    let mut mount = Mount::new(library)?;
    for operand in operands {
        Device { library, id: operand.device }.run(|| mount.operand(operand))?;
    }
    for step in steps {
        Device { library, id: step.device }.run(|| mount.step(step, stream(step.device)))?;
    }
    Ok(())
}

/// [`fill_by_mount`] for a whole record set, then its finishes.
pub(crate) fn fill_records_by_mount(library: &NativeLibrary, records: &Records,
    stream: impl Fn(i32) -> *mut c_void) -> Result<()> {
    fill_by_mount(library, &records.operands, &records.steps, &stream)?;
    records.run_finishes(library, stream)?;
    run_peer_copies(library, &records.peer_copies)
}


#[cfg(test)]
mod window_tests {
    use super::*;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn peer_load_copies_wait_for_group_success_and_mount_fill() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        for id in 0..2 { fixture.configure_memory(&library, id, 1 << 30)?; }
        fixture.immediate_copies(&library)?;
        let source = Device { library: &library, id: 1 }.run(||
            crate::shared::memory::DeviceAllocation::new(&library, 256))?;
        let destination = Device { library: &library, id: 0 }.run(||
            crate::shared::memory::DeviceAllocation::new(&library, 256))?;
        let bytes: Vec<_> = (0..256).map(|i| (i as u8).wrapping_mul(29)).collect();
        let untouched = vec![0xa5; 256];
        library.copy_h2d(source.buffer, &bytes)?;
        library.copy_h2d(destination.buffer, &untouched)?;
        let (_, records) = recording(|| DenseLoad::mount(&library).peer_copy(source.buffer, destination.buffer, 256))?;
        assert_eq!(records.peer_copies.len(), 1);
        let read = || -> Result<Vec<u8>> {
            let mut bytes = vec![0; 256];
            library.copy_d2h(&mut bytes, destination.buffer)?;
            Ok(bytes)
        };
        assert_eq!(read()?, untouched);
        assert!(committed_peer_copies::<()>(&library, Ok(None), &records.peer_copies)?.is_none());
        assert!(committed_peer_copies::<()>(&library, Err(anyhow::anyhow!("failed group")), &records.peer_copies).is_err());
        assert_eq!(read()?, untouched, "uncommitted or failed groups never copy");
        committed_peer_copies(&library, Ok(Some(())), &records.peer_copies)?;
        assert_eq!(read()?, bytes);
        library.copy_h2d(destination.buffer, &untouched)?;
        fill_records_by_mount(&library, &records, |_| std::ptr::null_mut())?;
        assert_eq!(read()?, bytes, "mount fill runs deferred copies after all operands");
        library.copy_h2d(destination.buffer, &untouched)?;
        DenseLoad::mount(&library).peer_copy(source.buffer, destination.buffer, 256)?;
        assert_eq!(read()?, bytes, "ordinary mount copy is immediate and drained");
        Ok(())
    }

    fn rect(name: &str, base: u64, rows: usize, width: usize) -> Rect {
        Rect { name: name.into(), path: "shard".into(), base, row_bytes: width, rows: 0..rows, cols: 0..width }
    }

    /// Steps open at the first delivered byte and close at the last; one source
    /// streams the file in offset order: the peak number open at once.
    fn peak_open(steps: &[Vec<Rect>]) -> usize {
        let mut events = Vec::new();
        for (index, rects) in steps.iter().enumerate() {
            for r in rects {
                let start = r.base + (r.rows.start * r.row_bytes) as u64;
                events.push((start, index, r.bytes() as u64));
            }
        }
        events.sort_unstable();
        let mut left: Vec<u64> = steps.iter().map(|rs| rs.iter().map(|r| r.bytes() as u64).sum()).collect();
        let (mut open, mut peak) = (std::collections::BTreeSet::new(), 0);
        for (_, index, bytes) in events {
            open.insert(index);
            peak = peak.max(open.len());
            left[index] -= bytes;
            if left[index] == 0 { open.remove(&index); }
        }
        peak
    }

    /// The NVFP4/KDA shape: one operand row-concatenating three tensors that
    /// sit far apart in one shard, each larger than a step. Windows used to mix
    /// q rows with k and v rows, holding every q step open until v arrived.
    #[test]
    fn steps_never_mix_tensors_so_one_file_stream_holds_one_open_step() -> Result<()> {
        let width = 8192 * 2;
        let rows = 4096; // 64 MiB a tensor, two steps each
        let q = rect("q", 0, rows, width);
        let k = rect("k", 1 << 30, rows, width);
        let v = rect("v", 2 << 30, rows, width);
        for unit in [1, 128] {
            let steps: Vec<Vec<Rect>> = windows(&[q.clone(), k.clone(), v.clone()], unit)?
                .into_iter().map(|(parts, _)| parts).collect();
            assert!(steps.iter().all(|parts| parts.len() == 1), "a step reads one tensor");
            assert!(steps.iter().all(|parts| parts.iter().map(Rect::bytes).sum::<usize>() <= STEP_BYTES));
            assert_eq!(peak_open(&steps), 1, "unit {unit}");
        }
        // Rows still cover the whole in order, in whole units.
        let covered: Vec<_> = windows(&[q, k, v], 128)?.into_iter().map(|(_, r)| r).collect();
        assert!(covered.windows(2).all(|w| w[0].end == w[1].start) && covered[0].start == 0
            && covered.last().unwrap().end == 3 * rows && covered.iter().all(|r| r.len() % 128 == 0));
        Ok(())
    }

    /// A unit straddling two tensors stays in one step (its rows quantize together).
    #[test]
    fn a_unit_across_tensors_stays_whole() -> Result<()> {
        let (a, b) = (rect("a", 0, 100, 256), rect("b", 1 << 20, 156, 256));
        let out = windows(&[a, b], 128)?;
        assert!(out.iter().all(|(_, rows)| rows.len() % 128 == 0));
        assert_eq!(out.iter().map(|(_, r)| r.len()).sum::<usize>(), 256);
        Ok(())
    }
}

#[cfg(test)]
mod absorb_fp8_tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn checked_i64_ids_replay_exactly_and_reject_overflow() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let path = dir.path().join("router.bin");
        for values in [vec![i64::from(i32::MIN), -1, 0, i64::from(i32::MAX)],
            vec![i64::from(i32::MAX) + 1], vec![i64::from(i32::MIN) - 1]] {
            let valid = values.iter().all(|&value| i32::try_from(value).is_ok());
            std::fs::write(&path, values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>())?;
            let output = crate::shared::memory::DeviceAllocation::new(&library, values.len() * 4)?;
            let rect = Rect { name: "router".into(), path: path.clone(), base: 0,
                row_bytes: 8, rows: 0..values.len(), cols: 0..8 };
            let step = Step { device: 0, rects: vec![rect], jobs: vec![Job::NarrowI64I32 { dst: output.buffer }] };
            let (_, records) = recording(|| DenseLoad::mount(&library).step(step, std::ptr::null_mut()))?;
            assert_eq!(records.steps.len(), 1);
            let layout = super::super::RankLayout::new(records.operands, records.steps, 0)?;
            assert_eq!(layout.host_staging.table_bytes, 4);
            let replay = super::super::testing::replay(device, layout, dir.path(), 1, 0);
            if valid {
                replay?;
                let mut actual = vec![0; values.len() * 4];
                library.copy_d2h(&mut actual, output.buffer)?;
                let expected = values.iter().flat_map(|&value| i32::try_from(value).unwrap().to_le_bytes()).collect::<Vec<_>>();
                assert_eq!(actual, expected);
            } else {
                assert!(format!("{:#}", replay.unwrap_err()).contains("I64 router ID exceeds I32 range"));
                assert!(!library.is_quarantined_after_failed_drain(), "a drained range error is recoverable");
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn i64_status_survives_a_first_drain_failure() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        let mut source = HostAllocation::new(&library, 8)?;
        source.bytes_mut().copy_from_slice(&1i64.to_le_bytes());
        let output = crate::shared::memory::DeviceAllocation::new(&library, 4)?;
        let step = Step { device: 0, rects: vec![Rect { name: "router".into(),
            path: fixture.directory().join("unused"), base: 0, row_bytes: 8, rows: 0..1, cols: 0..8 }],
            jobs: vec![Job::NarrowI64I32 { dst: output.buffer }] };
        let mut kernels = LoadKernels::new(&library)?;
        fixture.configure_pack(&library, 0, 1)?;
        fixture.configure_drain_after(&library, 0)?;
        let before = fixture.events()?.len();
        let result = step.run(&mut kernels, &library, source.bytes(), std::ptr::null_mut(), &mut None);
        assert!(result.is_err());
        assert!(library.is_quarantined_after_failed_drain());
        drop(output);
        drop(source);
        let events = fixture.events()?;
        let events = &events[before..];
        assert!(events.contains('A'), "the job allocated its mapped status: {events}");
        assert_eq!(events.matches('S').count(), 2, "first drain and outer error drain: {events}");
        assert!(!events.contains('F'), "neither status nor pinned source can be freed: {events}");
        Ok(())
    }

    /// GLM 5.3 head geometry (448 rows per head spans 3.5 grid blocks), the
    /// second of two ranks, staged through the mount: every byte and scale
    /// matches the loader's host rule (`weights::GlmLoader::kv_b`).
    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn fp8_kv_b_step_matches_the_host_split() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        let (heads, nope, v, lat) = (8usize, 192usize, 256usize, 512usize);
        let rows = heads * (nope + v);
        let kb = lat / 128;
        let grid_rows = rows.div_ceil(128);
        let kv_b: Vec<u8> = (0..rows * lat).map(|i| (i as u32).wrapping_mul(2654435761).rotate_right(13) as u8).collect();
        let grid: Vec<f32> = (0..grid_rows * kb).map(|i| 1.0 + i as f32 * 0.25).collect();
        let dir = tempfile::tempdir_in(fixture.directory())?;
        let path = dir.path().join("kv_b.bin");
        let grid_bytes: Vec<u8> = grid.iter().flat_map(|g| g.to_le_bytes()).collect();
        std::fs::write(&path, [kv_b.as_slice(), grid_bytes.as_slice()].concat())?;
        let (share, rank) = (heads / 2, 1usize);
        let first = rank * share * (nope + v);
        let alloc = |bytes: usize| crate::shared::memory::DeviceAllocation::new(&library, bytes);
        let (uk, uk_s) = (alloc(share * lat * nope)?, alloc(share * lat * (nope / 64) * 4)?);
        let (uv, uv_s) = (alloc(share * v * lat)?, alloc(share * v * (lat / 64) * 4)?);
        let values = Rect { name: "kv_b".into(), path: path.clone(), base: 0, row_bytes: lat,
            rows: first..first + share * (nope + v), cols: 0..lat };
        let scales = Rect { name: "kv_b.grid".into(), path, base: (rows * lat) as u64, row_bytes: kb * 4,
            rows: 0..grid_rows, cols: 0..kb * 4 };
        let step = Step { device: 0, jobs: vec![Job::AbsorbFp8 { at: 0, grid_at: values.bytes(),
            grid_bytes: scales.bytes(), row0: first, heads: share, nope, v, lat, uk: uk.buffer,
            uk_scale: uk_s.buffer, uv: uv.buffer, uv_scale: uv_s.buffer }], rects: vec![values, scales] };
        fill_by_mount(&library, &[], &[step], |_| std::ptr::null_mut())?;
        let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut out = vec![0u8; buffer.bytes];
            library.copy_d2h(&mut out, buffer)?;
            Ok(out)
        };
        let floats = |bytes: Vec<u8>| -> Vec<f32> {
            bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
        };
        let (k, ks, w, ws) = (read(uk.buffer)?, floats(read(uk_s.buffer)?), read(uv.buffer)?, floats(read(uv_s.buffer)?));
        let scale = |row: usize, col: usize| grid[(row / 128) * kb + col / 128];
        for h in 0..share {
            let base = (rank * share + h) * (nope + v);
            for c in 0..lat {
                for d in 0..nope { assert_eq!(k[(h * lat + c) * nope + d], kv_b[(base + d) * lat + c], "uk h{h} c{c} d{d}"); }
                for t in 0..nope / 64 { assert_eq!(ks[(h * lat + c) * (nope / 64) + t], scale(base + t * 64, c)); }
            }
            for r in 0..v {
                assert_eq!(&w[(h * v + r) * lat..][..lat], &kv_b[(base + nope + r) * lat..][..lat], "uv h{h} r{r}");
                for t in 0..lat / 64 { assert_eq!(ws[(h * v + r) * (lat / 64) + t], scale(base + nope + r, t * 64)); }
            }
        }
        Ok(())
    }
}
