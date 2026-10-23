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
pub(crate) const STEP_BYTES: usize = 32 << 20;

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
            let mut source = vec![0u8; self.rows.len() * self.row_bytes];
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
    /// One plane (`plane`: 0 gate/up weights, 2 down weights) of a native
    /// MXFP4 expert packed from its fused raw image at staged `at..at + bytes`
    /// (plane 0: up rows then gate rows, each padded) into `dst`.
    NativePlane { intermediate: u32, plane: u32, at: usize, bytes: usize, dst: CuteafdDeviceBuffer },
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
    /// Per-row scales gathered from a landed `grid` [*, grid_cols] (columns
    /// `col0..col0 + kb` of row `index[r]`, zero for `u32::MAX`) into
    /// `row_major` [rows, kb] and/or `k_major` [kb, rows].
    ScaleGather { grid: CuteafdDeviceBuffer, index: std::sync::Arc<Vec<u32>>, kb: usize, grid_cols: usize,
        col0: usize, row_major: Option<CuteafdDeviceBuffer>, k_major: Option<CuteafdDeviceBuffer> },
}

impl Finish {
    pub fn device(&self) -> i32 {
        match self { Finish::V41Scales { packed, .. } | Finish::NativePlane { dst: packed, .. } => packed.device_id,
            Finish::ScaleGather { grid, .. } => grid.device_id }
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
                Job::Upload { at, bytes, dst } => {
                    ensure!(at + bytes <= staged.len() && bytes <= dst.bytes, "upload outside its step");
                    load_timing::upload(bytes, || library.copy_h2d(CuteafdDeviceBuffer { bytes, ..dst },
                        &staged[at..at + bytes]))?;
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
                Job::NativePlane { intermediate, plane, at, bytes, dst } => {
                    ensure!(at + bytes <= self.bytes() && at % 4 == 0, "native plane outside its step");
                    let base = device_copy(scratch)?;
                    // SAFETY: the plane's source lies in the live device scratch (checked
                    // above), `dst` is a live operand slice; drained before reuse.
                    let source = CuteafdDeviceBuffer { ptr: unsafe { base.add(at) }.cast_mut().cast(), bytes, ..dst };
                    load_timing::quantize(bytes, || unsafe {
                        library.v41_pack_plane_async(source, dst, intermediate, plane, stream)?;
                        library.cuda_stream_synchronize(stream)
                    })?;
                }
            }
        }
        Ok(())
    }
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
    let per = (STEP_BYTES / width / unit).max(1) * unit;
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
    pub fn take(&self) -> (Vec<Operand>, Vec<Step>) { (self.operands.take(), self.steps.take()) }
    /// Every record, finishes included (leaves the sink empty).
    pub fn take_all(&self) -> Records {
        Records { operands: self.operands.take(), steps: self.steps.take(), finishes: self.finishes.take() }
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

/// One load's recorded class-2 operands, class-3 steps and finishes.
#[derive(Clone, Default)]
pub(crate) struct Records {
    pub operands: Vec<Operand>,
    pub steps: Vec<Step>,
    pub finishes: Vec<Finish>,
}

impl Records {
    /// `other`'s records after these (loader order).
    pub fn extend(&mut self, other: Records) {
        self.operands.extend(other.operands);
        self.steps.extend(other.steps);
        self.finishes.extend(other.finishes);
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
            let per = (STEP_BYTES / rect.width()).max(1);
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
            self.staging = Some(HostAllocation::named(self.library, bytes.max(1 << 20), "staging/coordinator-dense")?);
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
    records.run_finishes(library, stream)
}


#[cfg(test)]
mod window_tests {
    use super::*;

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
