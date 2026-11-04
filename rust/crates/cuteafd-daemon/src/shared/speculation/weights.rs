//! Header-only drafter sources and bounded dense load records, shared by block drafters.
use crate::shared::{experts::rtx::dense::{DenseLoad, Finish, Job, Operand, Rect, Step, STEP_BYTES},
    fp8_linear::{Fp8Scales, Fp8Weight}, memory::DeviceAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{read_safetensors_metadata, SafetensorsTensorMetadata};
use std::{collections::HashMap, ffi::c_void, path::{Path, PathBuf}};

pub(crate) struct Source {
    path: PathBuf,
    tensors: HashMap<String, SafetensorsTensorMetadata>,
    file_bytes: u64,
}

impl Source {
    pub fn open(snapshot: &Path) -> Result<Self> {
        let path = snapshot.join("model.safetensors");
        let tensors = read_safetensors_metadata(&path)?.into_iter().map(|t| (t.name.clone(), t)).collect();
        let file_bytes = std::fs::metadata(&path)?.len();
        Ok(Self { path, tensors, file_bytes })
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

pub(crate) struct Load<'a> {
    library: &'a NativeLibrary,
    source: Source,
    dense: DenseLoad<'a>,
    prefetched: Option<Prefetched>,
}

enum Prefetched {
    File(Vec<u8>),
    Tensors(HashMap<String, Vec<u8>>),
}

impl<'a> Load<'a> {
    pub fn open(library: &'a NativeLibrary, snapshot: &Path) -> Result<Self> {
        Ok(Self { library, source: Source::open(snapshot)?, dense: DenseLoad::mount(library), prefetched: None })
    }

    /// Preserve the existing overlapped mount read when no Recorder is active.
    pub fn with_prefetched(library: &'a NativeLibrary, snapshot: &Path, bytes: Vec<u8>) -> Result<Self> {
        let mut load = Self::open(library, snapshot)?;
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

    fn operand(&self, rects: Vec<Rect>) -> Result<DeviceAllocation<'a>> {
        let bytes = rects.iter().try_fold(0usize, |n, r| n.checked_add(r.bytes())).context("drafter operand overflow")?;
        let allocation = DeviceAllocation::new(self.library, bytes.max(256))?;
        let mut at = 0;
        let lands: Vec<_> = rects.into_iter().map(|rect| { let offset = at; at += rect.bytes(); (rect, offset) }).collect();
        if let Some(prefetched) = &self.prefetched {
            for (rect, offset) in &lands {
                let data = match prefetched {
                    Prefetched::File(bytes) => {
                        let start = usize::try_from(rect.base)?;
                        let end = start.checked_add(rect.bytes()).context("drafter prefetched extent overflow")?;
                        bytes.get(start..end).context("drafter tensor lies past prefetched file")?
                    }
                    Prefetched::Tensors(tensors) => {
                        let bytes = tensors.get(&rect.name).with_context(|| format!("{} was not prefetched", rect.name))?;
                        ensure!(bytes.len() == rect.bytes(), "{}: prefetched tensor extent differs from its header", rect.name);
                        bytes.as_slice()
                    }
                };
                self.library.copy_h2d(view(allocation.buffer, *offset, data.len())?, data)?;
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
        let uk = DeviceAllocation::new(self.library, heads * uk_head)?;
        let uv = DeviceAllocation::new(self.library, heads * uv_head)?;
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
        if !fp8 { return self.concat(parts, cols).map(Weight::Bf16); }
        let rects = self.parts(parts, cols)?;
        let rows = parts.iter().try_fold(0usize, |n, (_, rows)| n.checked_add(*rows)).context("drafter matrix overflow")?;
        if self.prefetched.is_some() {
            let source = self.operand(rects)?;
            let packed = Fp8Weight::pack(self.library, source.buffer.ptr, rows, cols, scales, stream);
            // SAFETY: source and pack destinations remain owned until this load stream drains.
            if let Err(error) = unsafe { self.library.cuda_stream_synchronize(stream) } {
                self.library.quarantine_module_after_failed_drain();
                std::mem::forget(source);
                if let Ok(packed) = packed { packed.quarantine(); }
                return Err(error.context("drafter pack owners quarantined after failed drain"));
            }
            return packed.map(Weight::Fp8);
        }
        let packed = Fp8Weight::allocate(self.library, rows, cols)?;
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

fn absorb_geometry(heads: usize, nope: usize, value: usize, latent: usize) -> Result<(usize, usize, usize, usize)> {
    ensure!(heads > 0 && nope > 0 && value > 0 && latent > 0, "empty draft absorption geometry");
    let rows = nope.checked_add(value).context("draft absorption head rows overflow")?;
    let width = latent.checked_mul(2).context("draft absorption row bytes overflow")?;
    let head = |rows: usize| rows.checked_mul(width).context("draft absorption head bytes overflow");
    let (source, uk, uv) = (head(rows)?, head(nope)?, head(value)?);
    for bytes in [rows, source, uk, uv] {
        heads.checked_mul(bytes).context("draft absorption total extent overflow")?;
    }
    ensure!(source <= STEP_BYTES, "draft absorption head exceeds staging bound");
    Ok((rows, source, uk, uv))
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
            let uk = DeviceAllocation::new(&library, heads * uk_head)?;
            let uv = DeviceAllocation::new(&library, heads * uv_head)?;
            absorb_into(&dense, Rect { name: "kv_b".into(), path: path.clone(), base: 0,
                row_bytes: latent * 2, rows: 0..heads * rows, cols: 0..latent * 2 },
                heads, nope, value, latent, uk.buffer, uv.buffer, std::ptr::null_mut())?;
            let mut owners = vec![uk, uv];
            for rect in &norms {
                let values = DeviceAllocation::new(&library, rect.bytes())?;
                let output = DeviceAllocation::new(&library, 3 * 4)?;
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
                let owner = DeviceAllocation::new(&library, (weight.shape.iter().product::<usize>() * 2).max(256))?;
                if weight.packing == cuteafd_loader::formats::dflash::MlaDraftPacking::UnitScale {
                    library.copy_h2d(owner.buffer, &weight.pack(&mut |_, _| unreachable!())?)?;
                }
                owners.insert(weight.owner.clone(), owner);
            }
            mla_weights_into(&dense, layout.weight_destinations(), &cfg.mla,
                |name| Ok(owners.get(name).context("fixture destination")?.buffer),
                &|name, shape| source.rect(name, shape), std::ptr::null_mut())?;
            let norms = DeviceAllocation::new(&library, cfg.vocab * 4)?;
            dense.finish(Finish::RowNorms { source: owners["weight.autoregressive_head.output.weight"].buffer,
                norms: norms.buffer, rows: cfg.vocab, cols: cfg.markov_rank }, std::ptr::null_mut())?;
            owners.insert("markov_norms".into(), norms);
            let target = DeviceAllocation::new(&library, 256)?;
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
