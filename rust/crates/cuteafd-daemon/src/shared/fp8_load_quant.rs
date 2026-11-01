//! Coordinator load-time BF16 -> FP8 on the GPU: the source tensor's bytes sit
//! in pinned host memory the kernel reads directly (no device staging), and
//! each scale unit writes its final E4M3 rows and scales into the operand.
//! Bit-for-bit the host rule (`glm5_flash::fp8::quantize`); see
//! native/shared/cuda/fp8_load_quant.h.
use crate::shared::{fp8_linear::Fp8Scales, memory::HostAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct Segment { offset: u64, ptr: *const u8, bytes: u64 }

#[repr(C)]
#[derive(Clone, Copy)]
struct Job {
    segments: *const Segment,
    count: i32,
    rows: i32,
    cols: i32,
    layout: i32,
    rule: i32,
    pad: i32,
    col0: i64,
    pitch: i64,
    q: *mut u8,
    q_stride: i64,
    scales: *mut f32,
    scale_stride: i64,
}

/// FP8 scale layout: 128x128 blocks, per row and 128-wide K block, or per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    Block,
    Row128,
    Channel,
}

fn layout_code(layout: Layout) -> i32 {
    match layout { Layout::Block => 0, Layout::Row128 => 1, Layout::Channel => 2 }
}

/// Where a quantized window lands: E4M3 rows (row stride `q_stride` bytes) and
/// its scales, row-major (`kmajor` false: `[units, kb]`) or K-block major
/// (`[kb, scale_rows]`, unit `u` at column `scale_row0 + u`).
#[derive(Clone, Copy)]
pub(crate) struct Target {
    pub q: CuteafdDeviceBuffer,
    pub q_stride: usize,
    pub scales: CuteafdDeviceBuffer,
    pub kmajor: Option<(usize, usize)>,
}

/// A scale source already read into bounded pinned staging, and its admitted
/// final coded plane. Gate/up sources have distinct tensor names and row bases.
pub(crate) struct ScaleCodeTarget<'a> {
    pub source: cuteafd_loader::formats::scale_codes::Rows<'a>,
    pub destination: CuteafdDeviceBuffer,
    pub stored_rows: usize,
    pub start_row: usize,
}

/// One load's pinned source (mount reads) and the load kernels, reused across tensors.
pub(crate) struct GpuQuantizer<'a> {
    source: Option<HostAllocation<'a>>,
    kernels: LoadKernels<'a>,
}

/// The GPU load transforms, each from caller-owned pinned, device-visible
/// source bytes (a mount read, or a push step's staging), with their job tables.
pub(crate) struct LoadKernels<'a> {
    library: &'a NativeLibrary,
    tables: Option<HostAllocation<'a>>,
}

impl<'a> GpuQuantizer<'a> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        Ok(Self { source: None, kernels: LoadKernels::new(library)? })
    }

    /// The load kernels alone (for sources the caller stages).
    pub fn kernels(&mut self) -> &mut LoadKernels<'a> { &mut self.kernels }

    /// The pinned source buffer, at least `bytes` long, for the caller to fill
    /// with one tensor's BF16 rows before `quantize`.
    pub fn source_mut(&mut self, bytes: usize) -> Result<&mut [u8]> {
        LoadKernels::grow(self.kernels.library, &mut self.source, bytes, "staging/fp8-load-quant")?;
        Ok(&mut self.source.as_mut().context("pinned source")?.bytes_mut()[..bytes])
    }

    fn staged(&mut self) -> Result<(&[u8], &mut LoadKernels<'a>)> {
        Ok((self.source.as_ref().context("no staged source")?.bytes(), &mut self.kernels))
    }

    /// Quantizes rows `rows` of the staged source (rows of `pitch` elements),
    /// columns `col0..col0 + cols`, into `target` on the current device, and
    /// drains `stream` before returning (the tables and source are reused).
    #[allow(clippy::too_many_arguments)]
    pub fn quantize(&mut self, rows: std::ops::Range<usize>, pitch: usize, col0: usize, cols: usize,
        layout: Layout, rule: Fp8Scales, target: Target, stream: *mut c_void) -> Result<()> {
        let (source, kernels) = self.staged()?;
        kernels.quantize_from(source, rows, pitch, col0, cols, layout, rule, target, stream)
    }

    /// The first `elements` BF16 values of the staged source widened to F32
    /// into `dst` on the current device (exact), drained before return.
    pub fn widen(&mut self, elements: usize, dst: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        let (source, kernels) = self.staged()?;
        ensure!(elements * 2 <= source.len(), "BF16 widening exceeds the staged source");
        kernels.widen_from(&source[..elements * 2], dst, stream)
    }

    /// The staged MLA `kv_b` absorbed per head (see [`LoadKernels::absorb_from`]).
    #[allow(clippy::too_many_arguments)]
    pub fn absorb(&mut self, heads_total: usize, h0: usize, heads: usize, nope: usize, v: usize, lat: usize,
        uk: CuteafdDeviceBuffer, uv: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        let (source, kernels) = self.staged()?;
        kernels.absorb_from(source, heads_total, h0, heads, nope, v, lat, uk, uv, stream)
    }
}

impl<'a> LoadKernels<'a> {
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        let (job, segment, _) = library.fp8_load_quant_abi()?;
        ensure!(job as usize == std::mem::size_of::<Job>() && segment as usize == std::mem::size_of::<Segment>(),
            "FP8 load quantization ABI differs from this daemon");
        Ok(Self { library, tables: None })
    }

    fn grow(library: &'a NativeLibrary, slot: &mut Option<HostAllocation<'a>>, bytes: usize, scope: &'static str)
        -> Result<()> {
        if slot.as_ref().is_none_or(|a| a.buffer.bytes < bytes) {
            drop(slot.take());
            *slot = Some(HostAllocation::named(library, bytes.max(1 << 20), scope)?);
        }
        Ok(())
    }

    /// Quantizes rows `rows` of `source` (pinned, device-visible BF16 rows of
    /// `pitch` elements the caller keeps alive), columns `col0..col0 + cols`,
    /// into `target` on the current device, drained before return.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_from(&mut self, source: &[u8], rows: std::ops::Range<usize>, pitch: usize, col0: usize,
        cols: usize, layout: Layout, rule: Fp8Scales, target: Target, stream: *mut c_void) -> Result<()> {
        let unit = if layout == Layout::Block { 128 } else { 1 };
        ensure!(cols > 0 && cols % 128 == 0 && col0 + cols <= pitch && rows.len() % unit == 0 && rows.start % unit == 0,
            "FP8 load quantization window is not whole scale units");
        ensure!(layout != Layout::Channel || (col0 == 0 && cols == pitch), "channel scales span whole rows");
        ensure!(rows.end * pitch * 2 <= source.len(), "window exceeds the staged source");
        let (units, kb) = (rows.len() / unit, cols / 128);
        ensure!(target.q.bytes >= rows.len() * target.q_stride && target.q_stride >= cols,
            "quantized rows exceed their operand");
        let scale_need = match target.kmajor { Some((scale_rows, row0)) => (kb - 1) * scale_rows + row0 + units, None => units * kb };
        ensure!(target.scales.bytes >= scale_need * 4, "quantized scales exceed their operand");
        let tables = std::mem::size_of::<Segment>() * units + std::mem::size_of::<Job>() * units;
        Self::grow(self.library, &mut self.tables, tables, "staging/fp8-load-quant")?;
        let base = source.as_ptr();
        let table = self.tables.as_mut().context("pinned tables")?;
        let (segments, jobs) = table.bytes_mut().split_at_mut(std::mem::size_of::<Segment>() * units);
        let segments = segments.as_mut_ptr().cast::<Segment>();
        let jobs_ptr = jobs.as_mut_ptr().cast::<Job>();
        for u in 0..units {
            let first = rows.start + u * unit;
            // SAFETY: both tables were sized for `units` entries above and are
            // pinned, device-visible memory owned by this quantizer.
            unsafe {
                let segment = segments.add(u);
                // SAFETY (pointer math): `first * pitch * 2` is inside the staged source (checked above).
                segment.write(Segment { offset: 0, ptr: base.add(first * pitch * 2), bytes: (unit * pitch * 2) as u64 });
                let (scales, scale_stride) = match target.kmajor {
                    Some((scale_rows, row0)) => (target.scales.ptr.cast::<f32>().add(row0 + u), scale_rows as i64),
                    None => (target.scales.ptr.cast::<f32>().add(u * kb), 1),
                };
                jobs_ptr.add(u).write(Job { segments: segment, count: 1, rows: unit as i32, cols: cols as i32,
                    layout: layout_code(layout), rule: rule.code(), pad: 0, col0: col0 as i64, pitch: pitch as i64,
                    q: target.q.ptr.cast::<u8>().add(u * unit * target.q_stride), q_stride: target.q_stride as i64,
                    scales, scale_stride });
            }
        }
        // SAFETY: the jobs name the pinned source, these pinned tables and the
        // target operands, all live until the stream drains below.
        unsafe {
            self.library.fp8_load_quant(jobs_ptr.cast(), units, stream)?;
            self.library.cuda_stream_synchronize(stream)
        }
    }

    /// GPU span-check and nibble pack from bounded mapped-host source bytes.
    /// The stream always drains before the job table/source can be reused.
    pub fn scale_codes_from(&mut self, targets: &[ScaleCodeTarget<'_>], stream: *mut c_void) -> Result<()> {
        use cuteafd_ffi::V41ScaleCodeJob;
        use cuteafd_loader::formats::scale_codes::{plane_bytes, ScaleCodeError};
        ensure!(!targets.is_empty(), "no scale-code source rectangles");
        let bytes = 8usize.checked_add(targets.len().checked_mul(std::mem::size_of::<V41ScaleCodeJob>())
            .context("scale-code job table overflow")?).context("scale-code job table overflow")?;
        Self::grow(self.library, &mut self.tables, bytes, "staging/scale-code-jobs")?;
        let table = self.tables.as_mut().context("scale-code job table")?;
        table.bytes_mut()[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        let jobs = table.bytes_mut()[8..].as_mut_ptr().cast::<V41ScaleCodeJob>();
        for (index, target) in targets.iter().enumerate() {
            let source = target.source;
            ensure!(source.rows > 0 && source.columns > 0 && source.pitch >= source.columns
                && target.stored_rows % 32 == 0
                && target.start_row.checked_add(source.rows).is_some_and(|n| n <= target.stored_rows)
                && (source.rows - 1).checked_mul(source.pitch).and_then(|n| n.checked_add(source.columns))
                    .is_some_and(|n| n <= source.bytes.len())
                && plane_bytes(target.stored_rows, source.columns) == Some(target.destination.bytes),
                "{}: invalid scale-code job geometry", source.tensor);
            // SAFETY: mapped pinned job storage was sized for all targets above;
            // the source and admitted output remain live through the drain below.
            unsafe { jobs.add(index).write(V41ScaleCodeJob {
                source: source.bytes.as_ptr(), destination: target.destination.ptr.cast(),
                source_bytes: source.bytes.len() as u64, destination_bytes: target.destination.bytes as u64,
                pitch: source.pitch as u64, rows: u32::try_from(source.rows)?, columns: u32::try_from(source.columns)?,
                stored_rows: u32::try_from(target.stored_rows)?, start_row: u32::try_from(target.start_row)?,
            }); }
        }
        // SAFETY: the tables and all source/output buffers are live and mapped;
        // draining even an enqueue failure prevents staging reuse racing a GPU.
        let submitted = unsafe { self.library.v41_pack_scale_codes_async(jobs, targets.len(),
            table.buffer.ptr.cast::<u64>(), stream) };
        let drained = unsafe { self.library.cuda_stream_synchronize(stream) };
        if drained.is_err() { self.library.quarantine_module_after_failed_drain(); }
        drained?;
        submitted?;
        let error = u64::from_le_bytes(table.bytes()[..8].try_into().unwrap());
        if error != u64::MAX {
            let target = targets.get((error >> 32) as usize).context("invalid scale-code refusal job")?;
            let row = error as u32 as usize;
            ensure!(row < target.source.rows, "invalid scale-code refusal row");
            let values = &target.source.bytes[row * target.source.pitch..row * target.source.pitch + target.source.columns];
            let span = values.iter().max().unwrap() - values.iter().min().unwrap();
            return Err(ScaleCodeError::Span { tensor: target.source.tensor.into(),
                row: target.source.first_row + row, span }.into());
        }
        Ok(())
    }

    /// BF16 `source` (pinned, device-visible) widened to F32 into `dst`.
    pub fn widen_from(&mut self, source: &[u8], dst: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        let elements = source.len() / 2;
        ensure!(elements > 0 && source.len() % 2 == 0 && elements * 4 <= dst.bytes,
            "BF16 widening exceeds its source or destination");
        // SAFETY: the pinned source holds `elements` BF16 values and `dst` room
        // for as many F32 (checked above); the stream drains before return.
        unsafe {
            self.library.load_widen_bf16_f32(source.as_ptr().cast(), dst.ptr, elements, stream)?;
            self.library.cuda_stream_synchronize(stream)
        }
    }

    /// MLA `kv_b` [heads_total * (nope + v), lat] (BF16) in `source` (pinned,
    /// device-visible) absorbed for heads `h0..h0 + heads` into `uk`
    /// [heads][lat][nope] and `uv` [heads][v][lat] on the current device,
    /// drained before return.
    #[allow(clippy::too_many_arguments)]
    pub fn absorb_from(&mut self, source: &[u8], heads_total: usize, h0: usize, heads: usize, nope: usize, v: usize,
        lat: usize, uk: CuteafdDeviceBuffer, uv: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        ensure!(heads > 0 && h0 + heads <= heads_total && heads_total * (nope + v) * lat * 2 <= source.len()
            && uk.bytes >= heads * lat * nope * 2 && uv.bytes >= heads * v * lat * 2,
            "kv_b absorption exceeds its source or operands");
        // SAFETY: the pinned source holds the whole kv_b and both operands hold
        // `heads` heads (checked above); the stream drains before return.
        unsafe {
            self.library.load_kvb_absorb(source.as_ptr().cast(), uk.ptr, uv.ptr, h0, heads, nope, v, lat, stream)?;
            self.library.cuda_stream_synchronize(stream)
        }
    }
}
