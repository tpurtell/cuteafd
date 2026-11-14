//! Header-only drafter sources and bounded dense load records, shared by block drafters.
use crate::shared::{experts::rtx::dense::{DenseLoad, Finish, Job, Operand, Rect, Step, STEP_BYTES},
    fp8_linear::{Fp8Scales, Fp8Weight}, memory::{DeviceAllocation, device::{Allocation, Device}}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{read_safetensors_metadata, SafetensorsTensorMetadata};
use std::{borrow::Cow, collections::HashMap, ffi::c_void, path::{Path, PathBuf}};
use crate::shared::memory::Tag;

pub(crate) use cuteafd_loader::families::glm5::draft_physical::{
    ShardAxis, gqa_weight_recipe, gqa_checkpoint_arena_requests as gqa_arena_requests,
    mla_weight_recipe, mla_checkpoint_arena_requests as mla_arena_requests, absorb_geometry,
};

fn shard(mut rect: Rect, axis: ShardAxis, rank: usize, ranks: usize) -> Result<Rect> {
    ensure!(matches!(ranks, 1 | 2) && rank < ranks, "drafter shard rank {rank} of {ranks}");
    let range = match axis { ShardAxis::Rows => &mut rect.rows, ShardAxis::Columns => &mut rect.cols };
    let units = match axis { ShardAxis::Rows => 1, ShardAxis::Columns => 2 };
    ensure!(range.len() % (ranks * units) == 0, "{}: drafter shard cuts a BF16 element", rect.name);
    let local = range.len() / ranks;
    let start = range.start + rank * local;
    *range = start..start + local;
    Ok(rect)
}

fn prefetched_rect<'b>(bytes: &'b [u8], base: usize, rect: &Rect) -> Result<Cow<'b, [u8]>> {
    let start = base.checked_add(rect.rows.start.checked_mul(rect.row_bytes)
        .context("drafter prefetched row overflow")?).context("drafter prefetched extent overflow")?;
    let end = base.checked_add(rect.rows.end.checked_mul(rect.row_bytes)
        .context("drafter prefetched row overflow")?).context("drafter prefetched extent overflow")?;
    ensure!(!rect.rows.is_empty() && !rect.cols.is_empty() && rect.cols.end <= rect.row_bytes,
        "{}: invalid prefetched rectangle", rect.name);
    let rows = bytes.get(start..end).context("drafter tensor lies past prefetched source")?;
    if rect.width() == rect.row_bytes { return Ok(Cow::Borrowed(rows)); }
    let mut out = Vec::with_capacity(rect.bytes());
    for row in rows.chunks_exact(rect.row_bytes) { out.extend_from_slice(&row[rect.cols.clone()]); }
    Ok(Cow::Owned(out))
}

pub(crate) struct Source {
    path: PathBuf,
    tensors: HashMap<String, SafetensorsTensorMetadata>,
    file_bytes: u64,
}

impl Source {
    pub fn open(snapshot: &Path) -> Result<Self> {
        Self::open_file(&snapshot.join("model.safetensors"))
    }

    pub fn open_file(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let tensors = read_safetensors_metadata(&path)?.into_iter().map(|t| (t.name.clone(), t)).collect();
        let file_bytes = std::fs::metadata(&path)?.len();
        Ok(Self { path, tensors, file_bytes })
    }

    fn parts(&self, parts: &[(&str, usize)], cols: usize) -> Result<Vec<Rect>> {
        parts.iter().map(|(name, rows)| self.rect(name, &[*rows, cols])).collect()
    }

    pub fn rect(&self, name: &str, shape: &[usize]) -> Result<Rect> {
        let tensor = self.tensors.get(name).with_context(|| format!("drafter checkpoint has no {name}"))?;
        ensure!(tensor.dtype == DType::Bf16, "{name}: drafter weights must be BF16, found {:?}", tensor.dtype);
        let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
            .with_context(|| format!("{name}: BF16 tensor byte count overflow"))?;
        ensure!(tensor.shape == shape && tensor.byte_length == bytes as u64,
            "{name}: shape {:?}/{} bytes, expected BF16 {shape:?}/{bytes} bytes", tensor.shape, tensor.byte_length);
        ensure!(tensor.byte_offset.checked_add(tensor.byte_length).is_some_and(|end| end <= self.file_bytes),
            "{name} lies past the drafter checkpoint file");
        let rows = if shape.len() >= 2 { shape[0] } else { 1 };
        ensure!(rows > 0 && bytes > 0 && bytes % rows == 0, "{name}: invalid tensor row extent");
        Ok(Rect { name: name.into(), path: self.path.clone(), base: tensor.byte_offset,
            row_bytes: bytes / rows, rows: 0..rows, cols: 0..bytes / rows })
    }
}

/// One resident representation; FP8 conversion stages bounded source windows,
/// never a second, full-sized BF16 device owner.
pub(crate) enum Weight<'a> {
    Bf16(DeviceAllocation<'a>),
    Fp8(Fp8Weight<'a>),
}

impl Weight<'_> {
    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        match self {
            Self::Bf16(owner) => owner.inventory(inventory, name),
            Self::Fp8(owner) => owner.inventory(inventory, name),
        }
    }
}

/// One projection dispatch shared by block backbones; storage stays with its owner.
pub(crate) trait Projection {
    /// # Safety
    /// The device spans hold BF16 [rows,k] input and [rows,n] output, do not
    /// overlap, and survive all queued work on `stream`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn project(&self, x: *const c_void, first: usize, out: *mut c_void,
        rows: usize, k: usize, n: usize, stream: *mut c_void) -> Result<()>;
}

pub(crate) struct LegacyProjection<'w, 'a> {
    pub weight: &'w Weight<'a>,
    pub library: &'a NativeLibrary,
    pub programs: &'w cuteafd_ffi::draft_projection::DraftProjections<'a>,
    pub scratch: Option<&'w DeviceAllocation<'a>>,
    pub mode: crate::shared::fp8_linear::Fp8Rows,
}

impl Projection for LegacyProjection<'_, '_> {
    unsafe fn project(&self, x: *const c_void, first: usize, out: *mut c_void,
        rows: usize, k: usize, n: usize, stream: *mut c_void) -> Result<()> {
        match self.weight {
            Weight::Bf16(weight) => {
                // SAFETY: the caller retains input/output and the selected weight rows.
                let source = unsafe { weight.buffer.ptr.cast::<u8>().add(first * k * 2) }.cast();
                // SAFETY: the selected matrix and spans satisfy the caller's contract.
                unsafe { Ok(self.programs.launch(x, source, out, rows, k, n, stream)?) }
            }
            Weight::Fp8(weight) => {
                let scratch = self.scratch.context("FP8 dSpark scratch was not admitted")?;
                // SAFETY: mode and scratch capacity were checked during admission.
                unsafe { weight.apply_rows(self.library, x, out, false, rows, first, n,
                    scratch, stream, self.mode) }
            }
        }
    }
}

pub(crate) type WeightAllocator<'a, 'b> = dyn FnMut(i32, &str, usize) -> Result<Allocation<'a>> + 'b;

pub(crate) fn standalone<'a>(library: &'a NativeLibrary, device: i32, _: &str, bytes: usize) -> Result<Allocation<'a>> {
    Allocation::new(Device { library, id: device }, bytes, Tag::drafter_weights("shared/drafter"))
}

pub(crate) fn checked_allocation<'a>(library: &'a NativeLibrary, name: &str, bytes: usize,
    allocate: &mut WeightAllocator<'a, '_>) -> Result<DeviceAllocation<'a>> {
    let device = library.cuda_get_device()?;
    let owner = allocate(device, name, bytes)?;
    ensure!(library.cuda_get_device()? == device && std::ptr::eq(owner.device.library, library)
        && owner.device.id == device && owner.buffer.device_id == device && owner.buffer.bytes == bytes
        && !owner.buffer.ptr.is_null(), "drafter allocation differs from {device}:{name}/{bytes}");
    Ok(DeviceAllocation::from_allocation(owner))
}

pub(crate) struct Load<'a, 'b> {
    library: &'a NativeLibrary,
    source: Source,
    dense: DenseLoad<'a>,
    prefetched: Option<Prefetched>,
    allocate: Option<std::cell::RefCell<&'b mut WeightAllocator<'a, 'b>>>,
}

enum Prefetched {
    File(std::rc::Rc<Vec<u8>>),
    Tensors(HashMap<String, Vec<u8>>),
}

impl<'a, 'b> Load<'a, 'b> {
    pub(crate) fn with_allocator(mut self, allocate: &'b mut WeightAllocator<'a, 'b>) -> Self {
        self.allocate = Some(std::cell::RefCell::new(allocate));
        self
    }

    pub(crate) fn allocation(&self, name: &str, bytes: usize) -> Result<DeviceAllocation<'a>> {
        if let Some(allocate) = &self.allocate {
            checked_allocation(self.library, name, bytes, &mut **allocate.borrow_mut())
        } else {
            DeviceAllocation::new(self.library, bytes, Tag::drafter_weights("shared/drafter"))
        }
    }
    pub fn open(library: &'a NativeLibrary, snapshot: &Path) -> Result<Self> {
        Self::open_file(library, &snapshot.join("model.safetensors"))
    }

    pub fn open_file(library: &'a NativeLibrary, path: &Path) -> Result<Self> {
        Ok(Self { library, source: Source::open_file(path)?, dense: DenseLoad::mount(library), prefetched: None, allocate: None })
    }

    /// Preserve the existing overlapped mount read when no Recorder is active.
    pub fn with_prefetched(library: &'a NativeLibrary, snapshot: &Path, bytes: Vec<u8>) -> Result<Self> {
        Self::with_prefetched_file(library, &snapshot.join("model.safetensors"), bytes)
    }

    pub fn with_prefetched_file(library: &'a NativeLibrary, path: &Path, bytes: Vec<u8>) -> Result<Self> {
        Self::with_shared_prefetched_file(library, path, std::rc::Rc::new(bytes))
    }

    pub fn with_shared_prefetched_file(library: &'a NativeLibrary, path: &Path, bytes: std::rc::Rc<Vec<u8>>) -> Result<Self> {
        let mut load = Self { library, source: Source::open_file(path)?, dense: DenseLoad::mount(library), prefetched: None, allocate: None };
        if !crate::shared::experts::rtx::dense::is_recording() { load.prefetched = Some(Prefetched::File(bytes)); }
        Ok(load)
    }

    pub fn with_tensors(library: &'a NativeLibrary, snapshot: &Path, tensors: HashMap<String, Vec<u8>>) -> Result<Self> {
        let mut load = Self::open(library, snapshot)?;
        if !crate::shared::experts::rtx::dense::is_recording() { load.prefetched = Some(Prefetched::Tensors(tensors)); }
        Ok(load)
    }

    pub fn row_norms(&self, source: CuteafdDeviceBuffer, norms: CuteafdDeviceBuffer, rows: usize,
        cols: usize, stream: *mut c_void) -> Result<()> {
        self.dense.finish(Finish::RowNorms { source, norms, rows, cols }, stream)
    }

    pub fn validate(&self, shapes: impl IntoIterator<Item = (String, Vec<usize>)>) -> Result<()> {
        for (name, shape) in shapes { self.source.rect(&name, &shape)?; }
        Ok(())
    }

    pub fn tensor(&self, name: &str, shape: &[usize]) -> Result<DeviceAllocation<'a>> {
        self.operand(vec![self.source.rect(name, shape)?])
    }

    pub(crate) fn tensor_parts(&self, parts: &[(&str, Vec<usize>)]) -> Result<DeviceAllocation<'a>> {
        self.operand(parts.iter().map(|(name, shape)| self.source.rect(name, shape)).collect::<Result<_>>()?)
    }

    pub fn tensor_shard(&self, name: &str, shape: &[usize], axis: ShardAxis,
        rank: usize, ranks: usize) -> Result<DeviceAllocation<'a>> {
        self.operand(vec![shard(self.source.rect(name, shape)?, axis, rank, ranks)?])
    }

    fn operand(&self, rects: Vec<Rect>) -> Result<DeviceAllocation<'a>> {
        let name = rects.iter().map(|r| r.name.as_str()).collect::<Vec<_>>().join("+");
        let bytes = rects.iter().try_fold(0usize, |n, r| n.checked_add(r.bytes())).context("drafter operand overflow")?;
        self.land_operand(rects, self.allocation(&name, bytes.max(256))?)
    }

    fn land_operand(&self, rects: Vec<Rect>, allocation: DeviceAllocation<'a>) -> Result<DeviceAllocation<'a>> {
        let bytes = rects.iter().try_fold(0usize, |n, r| n.checked_add(r.bytes())).context("drafter operand overflow")?;
        let mut at = 0;
        let lands: Vec<_> = rects.into_iter().map(|rect| { let offset = at; at += rect.bytes(); (rect, offset) }).collect();
        if let Some(prefetched) = &self.prefetched {
            for (rect, offset) in &lands {
                let data = match prefetched {
                    Prefetched::File(bytes) => prefetched_rect(bytes, usize::try_from(rect.base)?, rect)?,
                    Prefetched::Tensors(tensors) => {
                        let bytes = tensors.get(&rect.name).with_context(|| format!("{} was not prefetched", rect.name))?;
                        let tensor = self.source.tensors.get(&rect.name).context("drafter source metadata disappeared")?;
                        ensure!(bytes.len() as u64 == tensor.byte_length,
                            "{}: prefetched tensor extent differs from its header", rect.name);
                        prefetched_rect(bytes, 0, rect)?
                    }
                };
                self.library.copy_h2d(view(allocation.buffer, *offset, data.len())?, &data)?;
            }
        } else {
            self.dense.operand(Operand { device: allocation.buffer.device_id, buffer: allocation.buffer, lands, bytes })?;
        }
        Ok(allocation)
    }

    pub fn absorb(&self, name: &str, heads: usize, nope: usize, value: usize, latent: usize,
        stream: *mut c_void) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let (rows, _, uk_head, uv_head) = absorb_geometry(heads, nope, value, latent)?;
        let source = self.source.rect(name, &[heads * rows, latent])?;
        let uk = self.allocation(&format!("{name}.uk"), heads * uk_head)?;
        let uv = self.allocation(&format!("{name}.uv"), heads * uv_head)?;
        absorb_into(&self.dense, source, heads, nope, value, latent, uk.buffer, uv.buffer, stream)?;
        Ok((uk, uv))
    }

    pub fn concat(&self, parts: &[(&str, usize)], cols: usize) -> Result<DeviceAllocation<'a>> {
        self.operand(self.parts(parts, cols)?)
    }

    fn parts(&self, parts: &[(&str, usize)], cols: usize) -> Result<Vec<Rect>> {
        parts.iter().map(|(name, rows)| self.source.rect(name, &[*rows, cols])).collect()
    }

    pub fn matrix(&self, parts: &[(&str, usize)], cols: usize, fp8: bool, scales: Fp8Scales,
        stream: *mut c_void) -> Result<Weight<'a>> {
        self.matrix_rects(self.parts(parts, cols)?, fp8, scales, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matrix_shard(&self, parts: &[(&str, usize)], cols: usize, axis: ShardAxis,
        rank: usize, ranks: usize, fp8: bool, scales: Fp8Scales, stream: *mut c_void) -> Result<Weight<'a>> {
        let rects = self.parts(parts, cols)?.into_iter().map(|rect| shard(rect, axis, rank, ranks))
            .collect::<Result<Vec<_>>>()?;
        self.matrix_rects(rects, fp8, scales, stream)
    }

    fn matrix_rects(&self, rects: Vec<Rect>, fp8: bool, scales: Fp8Scales,
        stream: *mut c_void) -> Result<Weight<'a>> {
        if !fp8 { return self.operand(rects).map(Weight::Bf16); }
        let cols = rects.first().context("drafter matrix has no source parts")?.width() / 2;
        ensure!(cols > 0 && rects.iter().all(|rect| rect.width() == cols * 2), "drafter matrix source widths differ");
        let rows = rects.iter().try_fold(0usize, |n, rect| n.checked_add(rect.rows.len())).context("drafter matrix overflow")?;
        let name = rects.iter().map(|rect| rect.name.as_str()).collect::<Vec<_>>().join("+");
        let mut plane = |name: &str, bytes| -> Result<Allocation<'a>> {
            if let Some(allocator) = &self.allocate {
                let device = self.library.cuda_get_device()?;
                (**allocator.borrow_mut())(device, name, bytes)
            } else {
                standalone(self.library, self.library.cuda_get_device()?, name, bytes)
            }
        };
        if self.prefetched.is_some() {
            let bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(2)).context("drafter source extent overflow")?;
            // Conversion sources retire after packing and never belong to the immutable arena.
            let source = self.land_operand(rects, DeviceAllocation::new(self.library, bytes.max(256), Tag::staging("shared/drafter-source"))?)?;
            let packed = Fp8Weight::pack_with(self.library, source.buffer.ptr, rows, cols, scales, stream, &name, &mut plane);
            // SAFETY: source and pack destinations remain owned until this load stream drains.
            if let Err(error) = unsafe { self.library.cuda_stream_synchronize(stream) } {
                self.library.quarantine_module_after_failed_drain();
                std::mem::forget(source);
                if let Ok(packed) = packed { packed.quarantine(); }
                return Err(error.context("drafter pack owners quarantined after failed drain"));
            }
            return packed.map(Weight::Fp8);
        }
        let packed = Fp8Weight::allocate_with(self.library, rows, cols, &name, &mut plane)?;
        let (values, scale) = packed.buffers();
        for step in pack_steps(rects, cols, scales, values, scale)? { self.dense.step(step, stream)?; }
        Ok(Weight::Fp8(packed))
    }
}

/// Source-backed destinations from the admitted MLA recipe; synthetic scales
/// are initialized separately and Uv shares its Uk source staging.
pub(crate) fn mla_weights_into(
    dense: &DenseLoad<'_>,
    weights: &[cuteafd_loader::formats::dflash::MlaDraftWeight],
    geometry: &cuteafd_loader::formats::mla::MlaGeometry,
    destination: impl Fn(&str) -> Result<CuteafdDeviceBuffer>,
    source: &dyn Fn(&str, &[usize]) -> Result<Rect>,
    stream: *mut c_void,
) -> Result<()> {
    use cuteafd_loader::formats::dflash::MlaDraftPacking;
    for weight in weights {
        if matches!(weight.packing, MlaDraftPacking::UnitScale | MlaDraftPacking::Uv) { continue; }
        let buffer = destination(&weight.owner)?;
        let rects = weight.sources.iter().map(|(name, shape)| source(name, shape)).collect::<Result<Vec<_>>>()?;
        if weight.packing == MlaDraftPacking::Uk {
            let uv_id = weight.owner.strip_suffix(".uk").context("draft Uk owner suffix")?.to_owned() + ".uv";
            absorb_into(dense, rects.into_iter().next().context("draft Uk source missing")?,
                geometry.heads, geometry.nope, geometry.value, geometry.kv_rank,
                buffer, destination(&uv_id)?, stream)?;
        } else {
            let mut bytes = 0usize;
            let mut lands = Vec::with_capacity(rects.len());
            for rect in rects {
                let at = bytes;
                bytes = bytes.checked_add(rect.bytes()).context("draft weight extent overflow")?;
                lands.push((rect, at));
            }
            ensure!(buffer.bytes == bytes.max(256), "draft weight owner extent differs from packing");
            dense.operand(Operand { device: buffer.device_id, buffer, lands, bytes })?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn absorb_into(dense: &DenseLoad<'_>, source: Rect, heads: usize, nope: usize, value: usize, latent: usize,
    uk: CuteafdDeviceBuffer, uv: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
    let (rows, head_bytes, uk_head, uv_head) = absorb_geometry(heads, nope, value, latent)?;
    ensure!(source.rows.len() == heads * rows && source.width() == latent * 2
        && uk.device_id == uv.device_id && !uk.ptr.is_null() && !uv.ptr.is_null()
        && uk.bytes >= heads * uk_head && uv.bytes >= heads * uv_head,
        "draft absorption geometry differs from source or destination");
    let per = STEP_BYTES / head_bytes;
    for first in (0..heads).step_by(per) {
        let count = per.min(heads - first);
        dense.step(Step { device: uk.device_id, rects: vec![source.sub(first * rows..(first + count) * rows)],
            jobs: vec![Job::Absorb { heads: count, nope, v: value, lat: latent,
                uk: view(uk, first * uk_head, count * uk_head)?,
                uv: view(uv, first * uv_head, count * uv_head)? }] }, stream)?;
    }
    Ok(())
}


fn view(buffer: CuteafdDeviceBuffer, at: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(at.checked_add(bytes).is_some_and(|end| end <= buffer.bytes), "drafter pack destination out of bounds");
    // SAFETY: the checked byte range stays in the loader-owned allocation.
    Ok(CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(at) }.cast(), bytes, ..buffer })
}

fn pack_steps(rects: Vec<Rect>, cols: usize, rule: Fp8Scales, packed: CuteafdDeviceBuffer,
    scale: CuteafdDeviceBuffer) -> Result<Vec<Step>> {
    let mut first = 0usize;
    let mut steps = Vec::new();
    ensure!(cols > 0 && cols % 128 == 0, "drafter FP8 input width must be a multiple of 128");
    let row_bytes = cols.checked_mul(2).context("drafter FP8 row extent overflow")?;
    let scale_cols = cols / 128 * 4;
    let window = (STEP_BYTES / row_bytes / 16) * 16;
    ensure!(window > 0, "drafter FP8 row tile exceeds bounded staging");
    for rect in rects {
        ensure!(rect.rows.len() % 16 == 0 && rect.width() == row_bytes, "drafter FP8 source must hold whole 16-row tiles");
        for start in (0..rect.rows.len()).step_by(window) {
            let rows = window.min(rect.rows.len() - start);
            let values_at = first.checked_mul(cols).context("drafter FP8 value offset overflow")?;
            let scales_at = first.checked_mul(scale_cols).context("drafter FP8 scale offset overflow")?;
            let job = Job::W8a16Pack { n: rows, k: cols, rule,
                packed: view(packed, values_at, rows * cols)?,
                scale: view(scale, scales_at, rows * scale_cols)? };
            steps.push(Step { device: packed.device_id, rects: vec![rect.sub(start..start + rows)], jobs: vec![job] });
            first = first.checked_add(rows).context("drafter FP8 row count overflow")?;
        }
    }
    ensure!(first.checked_mul(cols) == Some(packed.bytes) && first.checked_mul(scale_cols) == Some(scale.bytes),
        "drafter FP8 packing footprint differs from its resident allocation");
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::experts::rtx::dense::{self, RankLayout};
    use cuteafd_loader::plan::testing::{t, write_safetensors};
    use std::os::unix::fs::FileExt;

    #[test]
    fn draft_shards_select_exact_rows_and_strided_bf16_columns() -> Result<()> {
        let rect = Rect { name: "weight".into(), path: PathBuf::new(), base: 7,
            row_bytes: 8, rows: 0..4, cols: 0..8 };
        let data: Vec<u8> = (0..32).collect();
        let mut file = vec![255; 7];
        file.extend_from_slice(&data);
        for rank in 0..2 {
            let rows = shard(rect.clone(), ShardAxis::Rows, rank, 2)?;
            assert_eq!(prefetched_rect(&file, 7, &rows)?.as_ref(), &data[rank * 16..(rank + 1) * 16]);
            let columns = shard(rect.clone(), ShardAxis::Columns, rank, 2)?;
            let expected: Vec<u8> = data.chunks_exact(8).flat_map(|row| row[rank * 4..(rank + 1) * 4].iter().copied()).collect();
            assert_eq!(prefetched_rect(&file, 7, &columns)?.as_ref(), expected);
            assert_eq!(prefetched_rect(&data, 0, &columns)?.as_ref(), expected);
        }
        assert!(shard(rect.clone(), ShardAxis::Rows, 2, 2).is_err());
        assert!(shard(rect.clone(), ShardAxis::Rows, 0, 0).is_err());
        assert!(shard(Rect { cols: 0..6, ..rect.clone() }, ShardAxis::Columns, 0, 2).is_err());
        assert!(prefetched_rect(&data[..31], 0, &rect).is_err());
        Ok(())
    }

    fn config() -> super::super::dflash::DflashConfig {
        super::super::dflash::DflashConfig { hidden: 128, intermediate: 128, layers: 2,
            heads: 1, kv_heads: 1, head_dim: 128, eps: 1e-6, theta: 10000., block: 8,
            group: 16, mask_token: 0, rank: 256, taps: vec![0], vocab: 16,
            window: 2048, row_window: true, sample_from_anchor: false }
    }

    fn write_drafter(root: &Path) -> Result<()> {
        std::fs::create_dir_all(root)?;
        let tensors = config().tensor_shapes().into_iter().map(|(name, shape)| t(name, "BF16", &shape)).collect::<Vec<_>>();
        let path = root.join("model.safetensors");
        write_safetensors(&path, &tensors);
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        for tensor in read_safetensors_metadata(&path)? {
            let seed = tensor.name.bytes().fold(0xcbf29ce484222325u64,
                |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)).to_le_bytes();
            let bytes = (0..tensor.byte_length).map(|i| (i as u8).wrapping_mul(29) ^ seed[i as usize % seed.len()]).collect::<Vec<_>>();
            file.write_all_at(&bytes, tensor.byte_offset)?;
        }
        Ok(())
    }

    #[test]
    fn gqa_final_arena_planes_match_standalone_mount_and_record_bytes() -> Result<()> {
        use crate::shared::memory::{weight_arena::{DeviceWeightArena, weight_allocation}, receipts::Inventory};
        use cuteafd_loader::{serving_capacity::WeightArenaLayout,
            families::glm5::draft_representation::GlmDraftRepresentation,
            placement::planner::{CoordinatorDevices, Lifetime, Phase}};
        use cuteafd_core::memory_layout::Category;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        let device = Device { library: &library, id: 1 };
        let root = tempfile::tempdir_in(fixture.directory())?;
        let snapshot = root.path().join("draft");
        std::fs::create_dir_all(&snapshot)?;
        let cfg = super::super::dflash::DflashConfig { hidden: 256, intermediate: 256,
            heads: 2, kv_heads: 2, ..config() };
        std::fs::write(snapshot.join("config.json"), serde_json::to_vec(&serde_json::json!({
            "hidden_size":cfg.hidden, "intermediate_size":cfg.intermediate,
            "num_hidden_layers":cfg.layers, "num_attention_heads":cfg.heads,
            "num_key_value_heads":cfg.kv_heads, "head_dim":cfg.head_dim,
            "rms_norm_eps":cfg.eps, "rope_theta":cfg.theta, "sliding_window":cfg.window,
            "vocab_size":cfg.vocab, "dflash_config":{"block_size":cfg.block,
                "conv_group_size":cfg.group, "conv_kernel_size":2, "mask_token_id":0,
                "selector_rank":cfg.rank, "selector_top_k":16, "target_layer_ids":cfg.taps}
        }))?)?;
        let path = snapshot.join("model.safetensors");
        write_safetensors(&path, &cfg.tensor_shapes().iter().map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>());
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        for tensor in read_safetensors_metadata(&path)? {
            let bytes = (0..tensor.byte_length / 2).flat_map(|i| {
                let value = (i % 127) as f32 * 0.015625 - 0.75;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            }).collect::<Vec<_>>();
            file.write_all_at(&bytes, tensor.byte_offset)?;
        }
        for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
            for ranks in [1, 2] { for rank in 0..ranks {
                let expected = cuteafd_loader::families::glm5::draft_physical::gqa_checkpoint_arena_requests(
                    &snapshot, representation, rank, ranks)?;
                assert_eq!(expected, cuteafd_loader::families::glm5::draft_physical::gqa_arena_requests(
                    &cfg, representation, rank, ranks)?);
                #[allow(clippy::too_many_arguments)]
                fn run<'a>(device: Device<'a>, snapshot: &Path, cfg: &super::super::dflash::DflashConfig,
                    rank: usize, ranks: usize, representation: GlmDraftRepresentation,
                    expected: &[(String, usize)], arena: Option<&DeviceWeightArena<'a>>)
                    -> Result<std::collections::BTreeMap<String, Weight<'a>>> {
                    let library = device.library;
                    let path = snapshot.join("model.safetensors");
                    device.run(|| {
                    let requested = std::cell::RefCell::new(Vec::new());
                    let mut end = 0usize;
                    let mut base = None;
                    let mut allocate = |id, name: &str, bytes| {
                        assert_eq!(id, device.id);
                        requested.borrow_mut().push((name.to_owned(), bytes));
                        let owner = weight_allocation(device, arena, name, bytes, crate::shared::memory::TEST_ARENA)?;
                        if let Some(arena) = arena {
                            let offset = end.div_ceil(256) * 256;
                            let root = *base.get_or_insert(owner.buffer.ptr as usize);
                            assert_eq!(owner.buffer.ptr as usize - root, offset);
                            let view = arena.layout().view(name).expect("GQA planned view");
                            assert_eq!((view.offset(), view.bytes()), (offset, bytes));
                            end = offset + bytes;
                        }
                        Ok(owner)
                    };
                    let bytes = if dense::is_recording() { Vec::new() } else { std::fs::read(&path)? };
                    let load = Load::with_prefetched(&library, &snapshot, bytes)?.with_allocator(&mut allocate);
                    let owners = std::cell::RefCell::new(std::collections::BTreeMap::new());
                    gqa_weight_recipe(&cfg, rank, ranks,
                        |name, shape| { owners.borrow_mut().insert(name.to_owned(), Weight::Bf16(load.tensor(name, shape)?)); Ok(()) },
                        |parts, cols, axis| {
                            let value = match axis {
                                Some(axis) => load.matrix_shard(parts, cols, axis, rank, ranks,
                                    representation == GlmDraftRepresentation::Fp8Only, Fp8Scales::Best, std::ptr::null_mut())?,
                                None => load.matrix(parts, cols,
                                    representation == GlmDraftRepresentation::Fp8Only, Fp8Scales::Best, std::ptr::null_mut())?,
                            };
                            owners.borrow_mut().insert(parts.iter().map(|(name, _)| *name).collect::<Vec<_>>().join("+"), value);
                            Ok(())
                        })?;
                    drop(load);
                    drop(allocate);
                    assert_eq!(requested.into_inner(), expected);
                    if let Some(arena) = arena {
                        assert_eq!(end, arena.layout().bytes());
                        assert_eq!(expected.len(), arena.layout().views().len());
                    }
                    Ok(owners.into_inner())
                    })
                }
                let baseline = run(device, &snapshot, &cfg, rank, ranks, representation, &expected, None)?;
                let digest = |owners: &std::collections::BTreeMap<String, Weight<'_>>| device.run(|| {
                    let mut bytes = std::collections::BTreeMap::new();
                    for (name, owner) in owners {
                        let buffers = match owner {
                            Weight::Bf16(owner) => vec![(name.clone(), owner.buffer)],
                            Weight::Fp8(owner) => { let (payload, scales) = owner.buffers();
                                vec![(format!("{name}.payload"), payload), (format!("{name}.scales"), scales)] },
                        };
                        for (name, buffer) in buffers {
                            let mut data = vec![0; buffer.bytes];
                            library.copy_d2h(&mut data, buffer)?;
                            assert!(data.iter().any(|&value| value != 0), "unfilled {name}");
                            bytes.insert(name, data);
                        }
                    }
                    Ok(bytes)
                });
                let baseline_bytes = digest(&baseline)?;
                for recorded in [false, true] {
                    let arena = DeviceWeightArena::new(device, WeightArenaLayout::new("test/draft", expected.clone())?, crate::shared::memory::TEST_ARENA)?;
                    let owners = if recorded {
                        let (owners, records) = dense::recording(|| run(device, &snapshot, &cfg, rank, ranks, representation, &expected, Some(&arena)))?;
                        assert!(!records.operands.is_empty());
                        // The nested DSO fixture path exceeds Unix socket path limits.
                        let sockets = tempfile::tempdir()?;
                        crate::shared::experts::rtx::push::tests::receive_pushed(device, None, root.path(), sockets.path(),
                            &records, None, 6, 1)?;
                        owners
                    } else { run(device, &snapshot, &cfg, rank, ranks, representation, &expected, Some(&arena))? };
                    drop(arena);
                    assert_eq!(digest(&owners)?, baseline_bytes, "{representation:?} TP{ranks} rank{rank} record={recorded}");
                    let mut inventory = Inventory::new(CoordinatorDevices::new(&[1])?, "draft",
                        Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "draft owners");
                    for (name, owner) in &owners { owner.allocation_receipts(&mut inventory, name)?; }
                    assert_eq!(inventory.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).count(), 1);
                    assert_eq!(inventory.receipts.iter().filter(|receipt| receipt.allocation.view.is_some()).count(), expected.len());
                }
            }}
        }
        Ok(())
    }

    #[test]
    fn drafter_and_target_repositories_share_one_namespace_plan() -> Result<()> {
        let root = tempfile::tempdir()?;
        let target = root.path().join("hf-home/hub/models--target/snapshots/a");
        let draft = root.path().join("hf-home/hub/models--draft/snapshots/b");
        write_drafter(&target)?;
        write_drafter(&draft)?;
        let mut operands = Vec::new();
        for snapshot in [&target, &draft] {
            let source = Source::open(snapshot)?;
            for (name, shape) in config().tensor_shapes() {
                let rect = source.rect(&name, &shape)?;
                let bytes = rect.bytes();
                operands.push(Operand { device: 1, bytes, lands: vec![(rect, 0)],
                    buffer: CuteafdDeviceBuffer { bytes, device_id: 1, ..Default::default() } });
            }
        }
        let layout = RankLayout::new(operands, Vec::new(), 1)?;
        let plan = layout.checked(root.path(), 8, 7)?;
        assert_eq!(plan.plan.files.len(), 2);
        assert!(plan.plan.files.iter().any(|file| file.contains("models--target/")));
        assert!(plan.plan.files.iter().any(|file| file.contains("models--draft/")));
        assert!(plan.plan.entries.iter().all(|entry| entry.rank == 7));
        let bytes = config().tensor_shapes().iter().map(|(_, s)| s.iter().product::<usize>() * 2).sum::<usize>();
        assert_eq!(plan.totals.rank_bytes[7], (2 * bytes) as u64);
        assert!(layout.checked(&target, 8, 7).is_err(), "snapshot root must not masquerade as namespace root");
        Ok(())
    }

    #[test]
    fn kimi_mla_drafter_joins_target_namespace_at_one_and_two_coordinator_ranks() -> Result<()> {
        use cuteafd_loader::formats::{dflash::MlaDflashConfig, mla::MlaGeometry,
            block_dspark::DsparkRotaryPlan};
        let cfg = MlaDflashConfig { mla: MlaGeometry { hidden: 128, heads: 2,
            q_rank: 128, kv_rank: 128, nope: 64, rope: 64, value: 64 },
            intermediate: 128, eps: 1e-5, block: 8, group: 16, selector_rank: 256,
            selector_top_k: 16, taps: vec![0, 1], vocab: 32, mask_token: 0,
            max_context: 8192, windows: vec![Some(4096), Some(4096), Some(4096), Some(4096), None],
            rotary: DsparkRotaryPlan { inv_freq: vec![1.; 32], attention_factor: 1., softmax_factor: 1. },
            kind: cuteafd_core::coordinator_programs::MlaDraftKind::Dflash2, markov_rank: 0 };
        let root = tempfile::tempdir()?;
        let target = root.path().join("hf-home/hub/models--target/snapshots/a");
        let draft = root.path().join("hf-home/hub/models--lightseekorg--kimi-k3-dflash2/snapshots/b");
        write_drafter(&target)?;
        std::fs::create_dir_all(&draft)?;
        let shapes = cfg.tensor_shapes();
        let tensors = shapes.iter().map(|(name, shape)| t(name, "BF16", shape)).collect::<Vec<_>>();
        write_safetensors(&draft.join("model.safetensors"), &tensors);
        let source = Source::open(&draft)?;
        source.rect("fc.weight", &[128, 256])?;
        let expected = shapes.iter().map(|(_, shape)| shape.iter().product::<usize>() * 2).sum::<usize>();
        for ranks in [1, 2] {
            let device = ranks as i32 - 1;
            let mut operands = shapes.iter().map(|(name, shape)| -> Result<_> {
                let rect = source.rect(name, shape)?;
                Ok(Operand { device, bytes: rect.bytes(), lands: vec![(rect.clone(), 0)],
                    buffer: CuteafdDeviceBuffer { bytes: rect.bytes(), device_id: device, ..Default::default() } })
            }).collect::<Result<Vec<_>>>()?;
            let target_source = Source::open(&target)?;
            let rect = target_source.rect("norm.weight", &[128])?;
            let target_bytes = rect.bytes();
            operands.push(Operand { device, bytes: target_bytes, lands: vec![(rect, 0)],
                buffer: CuteafdDeviceBuffer { bytes: target_bytes, device_id: device, ..Default::default() } });
            let layout = RankLayout::new(operands, Vec::new(), device)?;
            let plan = layout.checked(root.path(), 6 + ranks, 6 + ranks - 1)?;
            assert_eq!(plan.plan.files.len(), 2);
            assert!(plan.plan.files.iter().any(|file| file.contains("models--lightseekorg--kimi-k3-dflash2/")));
            assert!(plan.plan.files.iter().any(|file| file.contains("models--target/")));
            assert!(plan.plan.entries.iter().all(|entry| entry.rank as usize == 6 + ranks - 1));
            assert_eq!(plan.totals.rank_bytes[6 + ranks - 1], (expected + target_bytes) as u64);
            assert!(layout.checked(&target, 6 + ranks, 6 + ranks - 1).is_err());
        }
        Ok(())
    }

    #[test]
    fn fp8_concat_steps_cover_disjoint_tiles_without_bf16_residency() -> Result<()> {
        let cols = 128;
        let per = STEP_BYTES / (cols * 2);
        let rect = |name: &str, rows| Rect { name: name.into(), path: "draft".into(), base: 0,
            row_bytes: cols * 2, rows: 0..rows, cols: 0..cols * 2 };
        let rows = per + 32;
        let mut values = vec![0u8; 2 * rows * cols];
        let mut scales = vec![0u8; 2 * rows * cols / 128 * 4];
        let buffer = |data: &mut [u8]| CuteafdDeviceBuffer { ptr: data.as_mut_ptr().cast(),
            bytes: data.len(), device_id: 1, ..Default::default() };
        let steps = pack_steps(vec![rect("gate", rows), rect("up", rows)], cols, Fp8Scales::Amax,
            buffer(&mut values), buffer(&mut scales))?;
        assert_eq!(steps.len(), 4);
        assert!(steps.iter().all(|step| step.device == 1 && step.bytes() <= STEP_BYTES));
        let mut packed_bytes = 0;
        for step in steps {
            let Job::W8a16Pack { n, k, packed, scale, .. } = step.jobs[0] else { panic!("pack job") };
            assert_eq!(n % 16, 0);
            assert_eq!(packed.ptr as usize - values.as_ptr() as usize, packed_bytes);
            assert_eq!(scale.ptr as usize - scales.as_ptr() as usize, packed_bytes / 128 * 4);
            packed_bytes += n * k;
        }
        assert_eq!(packed_bytes, 2 * rows * cols);
        Ok(())
    }

    #[test]
    fn fp8_sharded_steps_preserve_source_windows_and_local_destinations() -> Result<()> {
        let cols = 128;
        let rows = STEP_BYTES / (cols * 2) + 32;
        let source = Rect { name: "down".into(), path: "draft".into(), base: 31,
            row_bytes: cols * 4, rows: 16..16 + rows * 2, cols: 0..cols * 4 };
        let selected = shard(shard(source.clone(), ShardAxis::Rows, 1, 2)?, ShardAxis::Columns, 1, 2)?;
        let mut values = vec![0u8; rows * cols];
        let mut scales = vec![0u8; rows * cols / 128 * 4];
        let buffer = |data: &mut [u8]| CuteafdDeviceBuffer { ptr: data.as_mut_ptr().cast(),
            bytes: data.len(), device_id: 3, ..Default::default() };
        let steps = pack_steps(vec![selected.clone()], cols, Fp8Scales::Amax,
            buffer(&mut values), buffer(&mut scales))?;
        assert_eq!(steps.len(), 2);
        let mut completed = 0;
        for step in steps {
            assert_eq!(step.device, 3);
            assert!(step.bytes() <= STEP_BYTES);
            assert_eq!(step.rects.len(), 1);
            let rect = &step.rects[0];
            let Job::W8a16Pack { n, k, packed, scale, .. } = step.jobs[0] else { panic!("pack job") };
            assert_eq!((rect.base, rect.row_bytes, rect.cols.clone()),
                (source.base, source.row_bytes, cols * 2..cols * 4));
            assert_eq!(rect.rows, selected.rows.start + completed..selected.rows.start + completed + n);
            assert_eq!((n % 16, k), (0, cols));
            assert_eq!(packed.ptr as usize - values.as_ptr() as usize, completed * cols);
            assert_eq!(scale.ptr as usize - scales.as_ptr() as usize, completed * cols / 128 * 4);
            assert_eq!((packed.bytes, scale.bytes), (n * cols, n * cols / 128 * 4));
            completed += n;
        }
        assert_eq!(completed, rows);
        Ok(())
    }

    #[test]
    fn absorption_geometry_rejects_empty_overflow_and_oversized_heads() {
        for (heads, nope, value, latent) in [(0, 192, 256, 512), (1, 0, 256, 512),
            (1, 192, 0, 512), (1, 192, 256, 0), (1, usize::MAX, 1, 512),
            (1, 192, 256, usize::MAX), (usize::MAX, 192, 256, 512), (1, 192, 256, STEP_BYTES)] {
            assert!(absorb_geometry(heads, nope, value, latent).is_err());
        }
        assert_eq!(absorb_geometry(8, 192, 256, 512).unwrap(), (448, 458752, 196608, 262144));
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn mla_absorption_and_deferred_norms_match_mount_after_push_commit() -> Result<()> {
        use crate::shared::memory::device::Device;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let root = tempfile::tempdir_in(fixture.directory())?;
        let path = root.path().join("draft.bin");
        let (heads, nope, value, latent) = (80usize, 192usize, 256usize, 512usize);
        let (rows, _, uk_head, uv_head) = absorb_geometry(heads, nope, value, latent)?;
        let source = (0..heads * rows * latent).flat_map(|i|
            ((i as u16).wrapping_mul(29) ^ ((i >> 16) as u16)).to_le_bytes()).collect::<Vec<_>>();
        std::fs::write(&path, &source)?;
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        let mut at = source.len();
        let mut norms = Vec::new();
        for cols in [256usize, 512] {
            let bytes = (0..3 * cols).flat_map(|i| {
                let row = i / cols;
                let value = if row == 0 { 0.0f32 } else if row == 1 { 1.0 } else { ((i % 7) as f32 - 3.0) * 0.25 };
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            }).collect::<Vec<_>>();
            file.write_all_at(&bytes, at as u64)?;
            norms.push(Rect { name: format!("markov-{cols}"), path: path.clone(), base: at as u64,
                row_bytes: cols * 2, rows: 0..3, cols: 0..cols * 2 });
            at += bytes.len();
        }
        let device = Device { library: &library, id: 1 };
        let load = || device.run(|| {
            let dense = DenseLoad::mount(&library);
            let uk = DeviceAllocation::new(&library, heads * uk_head, crate::shared::memory::TEST)?;
            let uv = DeviceAllocation::new(&library, heads * uv_head, crate::shared::memory::TEST)?;
            absorb_into(&dense, Rect { name: "kv_b".into(), path: path.clone(), base: 0,
                row_bytes: latent * 2, rows: 0..heads * rows, cols: 0..latent * 2 },
                heads, nope, value, latent, uk.buffer, uv.buffer, std::ptr::null_mut())?;
            let mut owners = vec![uk, uv];
            for rect in &norms {
                let values = DeviceAllocation::new(&library, rect.bytes(), crate::shared::memory::TEST)?;
                let output = DeviceAllocation::new(&library, 3 * 4, crate::shared::memory::TEST)?;
                // Poison output to prove recording does not run a Finish early.
                library.copy_h2d(output.buffer, &[0xff; 12])?;
                dense.operand(Operand { device: 1, bytes: rect.bytes(), buffer: values.buffer,
                    lands: vec![(rect.clone(), 0)] })?;
                dense.finish(Finish::RowNorms { source: values.buffer, norms: output.buffer,
                    rows: 3, cols: rect.width() / 2 }, std::ptr::null_mut())?;
                owners.extend([values, output]);
            }
            Ok(owners)
        });
        let mounted = load()?;
        let (pushed, records) = dense::recording(load)?;
        assert_eq!((records.operands.len(), records.steps.len(), records.finishes.len()), (2, 2, 2));
        assert!(records.steps.iter().all(|step| step.bytes() <= STEP_BYTES));
        let read = |owner: &DeviceAllocation<'_>| device.run(|| {
            let mut bytes = vec![0; owner.buffer.bytes];
            library.copy_d2h(&mut bytes, owner.buffer)?;
            Ok(bytes)
        });
        for index in [3, 5] { assert_eq!(read(&pushed[index])?, [0xff; 12]); }
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        crate::shared::experts::rtx::push::tests::receive_pushed(device, None, root.path(), sockets.path(),
            &records, None, 6, 1)?;
        for (index, (expected, actual)) in mounted.iter().zip(&pushed).enumerate() {
            assert_eq!(read(expected)?, read(actual)?, "owner {index}");
        }
        let (uk, uv) = (read(&pushed[0])?, read(&pushed[1])?);
        for h in 0..heads {
            for c in 0..latent {
                for r in 0..nope {
                    let src = ((h * rows + r) * latent + c) * 2;
                    let dst = ((h * latent + c) * nope + r) * 2;
                    assert_eq!(&uk[dst..dst + 2], &source[src..src + 2]);
                }
            }
            let start = (h * rows + nope) * latent * 2;
            assert_eq!(&uv[h * uv_head..(h + 1) * uv_head], &source[start..start + uv_head]);
        }
        for (index, cols) in [(3, 256usize), (5, 512)] {
            let bytes = read(&pushed[index])?;
            let floats = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            let expected = if cols == 256 { [0.0f32, 16.0] } else { [1e-6f32, (cols as f32).sqrt().mul_add(1.0001, 1e-6)] };
            assert_eq!(floats[0].to_bits(), expected[0].to_bits());
            assert_eq!(floats[1].to_bits(), expected[1].to_bits());
            assert!(floats[2].is_finite() && floats[2] > 0.0);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn ml4_recipe_joins_target_rank_and_matches_mount_after_commit() -> Result<()> {
        use crate::shared::memory::device::Device;
        use cuteafd_loader::{formats::{dflash::MlaDflashConfig, mla::MlaGeometry,
            block_dspark::DsparkRotaryPlan}, families::glm5::draft_representation::{GlmDraftCapacity,
            GlmDraftRepresentation::Bf16Only}};
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let root = tempfile::tempdir_in(fixture.directory())?;
        let snapshot = root.path().join("models--ml4/snapshots/test");
        std::fs::create_dir_all(&snapshot)?;
        // Scaled source dimensions retain the native packing/rank/window recipe;
        // no target or drafter execution programs are needed for load replay.
        let cfg = MlaDflashConfig { mla: MlaGeometry { hidden: 128, heads: 2, q_rank: 0,
            kv_rank: 128, nope: 64, rope: 64, value: 128 }, intermediate: 128, eps: 1e-5,
            block: 3, group: 0, selector_rank: 512, selector_top_k: 0, taps: vec![0, 1],
            vocab: 16, mask_token: 0, max_context: 4096, windows: vec![Some(2048); 2],
            rotary: DsparkRotaryPlan { inv_freq: vec![1.; 32], attention_factor: 1., softmax_factor: 1. },
            kind: cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark, markov_rank: 512 };
        let shapes = cfg.tensor_shapes();
        let tensors = shapes.iter().map(|(name, shape)| t(name.clone(), "BF16", shape))
            .chain(std::iter::once(t("model.norm.weight", "BF16", &[128]))).collect::<Vec<_>>();
        let path = snapshot.join("model.safetensors");
        write_safetensors(&path, &tensors);
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        for tensor in read_safetensors_metadata(&path)? {
            let bytes = (0..tensor.byte_length / 2).flat_map(|i|
                (((i % 7) as f32 * 0.25).to_bits().wrapping_shr(16) as u16).to_le_bytes()).collect::<Vec<_>>();
            file.write_all_at(&bytes, tensor.byte_offset)?;
        }
        let layout = cfg.runtime_layout(Bf16Only, GlmDraftCapacity::new(1, 1, 3)?,
            4096, 4096, 1, 0, cuteafd_core::MlaCacheFormat::Fp8)?;
        let source = Source::open(&snapshot)?;
        let device = Device { library: &library, id: 1 };
        let load = || device.run(|| {
            let dense = DenseLoad::mount(&library);
            let mut owners = std::collections::BTreeMap::new();
            for weight in layout.weight_destinations() {
                let owner = DeviceAllocation::new(&library, (weight.shape.iter().product::<usize>() * 2).max(256), crate::shared::memory::TEST)?;
                if weight.packing == cuteafd_loader::formats::dflash::MlaDraftPacking::UnitScale {
                    library.copy_h2d(owner.buffer, &weight.pack(&mut |_, _| unreachable!())?)?;
                }
                owners.insert(weight.owner.clone(), owner);
            }
            mla_weights_into(&dense, layout.weight_destinations(), &cfg.mla,
                |name| Ok(owners.get(name).context("fixture destination")?.buffer),
                &|name, shape| source.rect(name, shape), std::ptr::null_mut())?;
            let norms = DeviceAllocation::new(&library, cfg.vocab * 4, crate::shared::memory::TEST)?;
            dense.finish(Finish::RowNorms { source: owners["weight.autoregressive_head.output.weight"].buffer,
                norms: norms.buffer, rows: cfg.vocab, cols: cfg.markov_rank }, std::ptr::null_mut())?;
            owners.insert("markov_norms".into(), norms);
            let target = DeviceAllocation::new(&library, 256, crate::shared::memory::TEST)?;
            dense.operand(Operand { device: 1, buffer: target.buffer, bytes: 256,
                lands: vec![(source.rect("model.norm.weight", &[128])?, 0)] })?;
            owners.insert("target.norm".into(), target);
            Ok(owners)
        });
        let mounted = load()?;
        let (pushed, records) = dense::recording(load)?;
        let rank = RankLayout::new(records.operands.clone(), records.steps.clone(), 1)?.with_finishes(&records.finishes);
        let plan = rank.checked(root.path(), 8, 7)?;
        let bytes = shapes.iter().map(|(_, shape)| shape.iter().product::<usize>() * 2).sum::<usize>();
        assert_eq!(plan.totals.rank_bytes[7], (bytes + 256) as u64);
        assert_eq!(plan.plan.files.len(), 1);
        assert!(plan.plan.entries.iter().all(|entry| entry.rank == 7));
        assert_eq!(records.finishes.len(), 1);
        let names = records.operands.iter().flat_map(|o| o.lands.iter().map(|(r, _)| &r.name))
            .chain(records.steps.iter().flat_map(|s| s.rects.iter().map(|r| &r.name)))
            .collect::<std::collections::BTreeSet<_>>();
        for (name, _) in &shapes { assert!(names.contains(name), "missing draft source {name}"); }
        assert!(names.contains(&"model.norm.weight".to_owned()));
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        crate::shared::experts::rtx::push::tests::receive_pushed(device, None, root.path(), sockets.path(),
            &records, None, 6, 1)?;
        for (name, expected) in &mounted {
            let actual = &pushed[name];
            let read = |owner: &DeviceAllocation<'_>| device.run(|| {
                let mut bytes = vec![0; owner.buffer.bytes];
                library.copy_d2h(&mut bytes, owner.buffer)?;
                Ok(bytes)
            });
            // Sub-256-byte allocation padding is not checkpoint state.
            let bytes = layout.weight_destinations().iter().find(|w| w.owner == *name)
                .map_or(expected.buffer.bytes, |w| w.shape.iter().product::<usize>() * 2);
            assert_eq!(&read(expected)?[..bytes], &read(actual)?[..bytes], "{name}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn muse_drafter_all_tensor_bytes_match_mount_after_push_commit() -> Result<()> {
        use crate::shared::memory::device::Device;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let root = tempfile::tempdir_in(fixture.directory())?;
        let snapshot = root.path().join("models--muse-draft/snapshots/test");
        write_drafter(&snapshot)?;
        let device = Device { library: &library, id: 1 };
        let shapes = config().tensor_shapes();
        // Fuse the production GQA destinations: Q/K/V and gate/up share owners;
        // every other source keeps its native shape, including the 3-D bases.
        let mut groups = shapes.iter().map(|(name, shape)| vec![(name.clone(), shape.clone())]).collect::<Vec<_>>();
        for layer in 0..config().layers {
            for suffixes in [vec!["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
                vec!["mlp.gate_proj.weight", "mlp.up_proj.weight"]] {
                let mut fused = Vec::new();
                for suffix in suffixes {
                    let name = format!("layers.{layer}.{suffix}");
                    let at = groups.iter().position(|group| group[0].0 == name).context("fused source")?;
                    fused.extend(groups.remove(at));
                }
                groups.push(fused);
            }
        }
        let load = || device.run(|| {
            // Mount keeps its existing full-file prefetch; recording must ignore it.
            let bytes = if dense::is_recording() { Vec::new() } else { std::fs::read(snapshot.join("model.safetensors"))? };
            let load = Load::with_prefetched(&library, &snapshot, bytes)?;
            load.validate(shapes.clone())?;
            groups.iter().map(|group| {
                if group.len() == 1 { return load.tensor(&group[0].0, &group[0].1); }
                let cols = group[0].1[1];
                let parts = group.iter().map(|(name, shape)| (name.as_str(), shape[0])).collect::<Vec<_>>();
                let Weight::Bf16(owner) = load.matrix(&parts, cols, false, Fp8Scales::Amax, std::ptr::null_mut())?
                    else { unreachable!("BF16 fixture") };
                Ok(owner)
            }).collect::<Result<Vec<_>>>()
        });
        let mounted = load()?;
        let (pushed, records) = dense::recording(load)?;
        assert_eq!(records.operands.len(), shapes.len() - 3 * config().layers);
        assert_eq!(records.operands.iter().map(|operand| operand.lands.len()).sum::<usize>(), shapes.len());
        let layout = RankLayout::new(records.operands.clone(), Vec::new(), 1)?;
        let plan = layout.checked(root.path(), 8, 7)?;
        let source_bytes = shapes.iter().map(|(_, shape)| shape.iter().product::<usize>() * 2).sum::<usize>();
        assert_eq!(plan.totals.rank_bytes[7], source_bytes as u64);
        assert_eq!(plan.plan.files.len(), 1);
        assert!(plan.plan.files[0].contains("models--muse-draft/"));
        assert!(records.steps.is_empty() && records.finishes.is_empty());
        assert!(records.operands.iter().all(|operand| operand.device == 1));
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        // The same receive_rank completion/commit protocol as serving, no GPU.
        crate::shared::experts::rtx::push::tests::receive_pushed(device, None, root.path(), sockets.path(),
            &records, None, 6, 1)?;
        for (((group, expected), actual), operand) in groups.iter().zip(&mounted).zip(&pushed).zip(&records.operands) {
            let bytes = operand.bytes;
            let read = |owner: &DeviceAllocation<'_>| device.run(|| {
                let mut out = vec![0u8; bytes];
                library.copy_d2h(&mut out, owner.buffer)?;
                Ok(out)
            });
            let expected_bytes = read(expected)?;
            assert_eq!(expected_bytes, read(actual)?, "{group:?}");
            let source = Source::open(&snapshot)?;
            let native = group.iter().map(|(name, shape)| {
                let rect = source.rect(name, shape)?;
                let mut bytes = vec![0; rect.bytes()];
                rect.read(&mut bytes)?;
                Ok(bytes)
            }).collect::<Result<Vec<_>>>()?.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(expected_bytes, native, "mount concatenation {group:?}");
        }
        Ok(())
    }
}
