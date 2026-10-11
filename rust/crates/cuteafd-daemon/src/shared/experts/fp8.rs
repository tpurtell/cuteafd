//! Exact FP8 routed experts (the `fp8` family): the checkpoint's E4M3 expert
//! weights with FP32 128x128 block scales (or MXFP4, MiMo V2.6 Pro:
//! `fp8-mimop` packages; or ModelOpt NVFP4 run W4A16, `fp8-<family>-nvfp4`
//! packages), resident per TP slice and run by
//! an `fp8-<family>` package (`python/tools/aot/package_fp8_moe_aot.py`,
//! `native/shared/include/cuteafd_fp8_moe.h`). Nothing is re-quantized. Output is
//! the BF16 `[rows, H]` route sum of the slice: the Spark rank partial of the
//! compact BF16 response, or the whole layer at TP1 on the coordinator.
pub(crate) mod worker;

use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::fp8_moe::{Fp8MoeInfo, Fp8MoeModule, Fp8MoePrefill, Fp8MoeWeights, FP8_MOE_POINTERS};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection, Slicing};
use std::ffi::c_void;
use std::path::{Path, PathBuf};

/// Parallel readers per layer load.
const READERS: usize = 16;

/// Admit the whole resident expert allocation before allocating any weights or
/// scratch. Callers keep worker/step and prefix-cache reservations out of `budget`.
fn resident_admission(layer_bytes: usize, layers: usize, primary_scratch: usize, bf16_scratch: Option<usize>,
    budget: usize) -> Result<(usize, usize)> {
    let scratch_bytes = primary_scratch.max(bf16_scratch.unwrap_or(0)).max(256);
    let resident = layer_bytes.checked_mul(layers).context("FP8 resident expert size overflow")?;
    let required = resident.checked_add(scratch_bytes).context("FP8 expert allocation size overflow")?;
    ensure!(required <= budget,
        "FP8 experts need {:.3} GiB resident + {:.3} GiB scratch ({} bytes); budget {:.3} GiB ({} bytes)",
        resident as f64 / (1u64 << 30) as f64, scratch_bytes as f64 / (1u64 << 30) as f64, required,
        budget as f64 / (1u64 << 30) as f64, budget);
    Ok((resident, scratch_bytes))
}

/// Both programs share the same resident slice and scratch arena. Input dtype
/// and compiled capacities may differ, but the weight layout and arithmetic may not.
fn validate_bf16_package(main: &Fp8MoeInfo, bf16: &Fp8MoeInfo, capacity: usize) -> Result<usize> {
    ensure!(!bf16.wire_input && bf16.weights == main.weights && bf16.hidden == main.hidden
        && bf16.experts == main.experts && bf16.topk == main.topk && bf16.intermediate == main.intermediate
        && bf16.tp == main.tp && bf16.slice == main.slice && bf16.swiglu_limit == main.swiglu_limit,
        "{bf16:?} is not the BF16-input form of the primary FP8 package {main:?}");
    bf16.capacity_for(capacity)
        .with_context(|| format!("BF16-input FP8 package has no capacity for {capacity} rows"))
}

/// Environment switch for how FP8 expert packages run prefill row counts:
/// `auto` (default: FP8 wire rows W8A8, block-scaled E4M3 x E4M3 gate/up; BF16
/// rows W8A16, exact), `w8a16` (the former programs, weights widened to
/// `bf16(w * s)`) or `w8a8` (BF16 rows quantized to wire rows too).
pub(crate) const PREFILL_ENV: &str = "CUTEAFD_FP8_EXPERT_PREFILL";

/// The prefill form `PREFILL_ENV` asks for.
pub(crate) fn prefill_mode() -> Result<Fp8MoePrefill> {
    match std::env::var(PREFILL_ENV) {
        Ok(value) if !value.is_empty() => value.parse().with_context(|| format!("{PREFILL_ENV}={value}")),
        _ => Ok(Fp8MoePrefill::default()),
    }
}

/// Loads an FP8 expert package and selects its prefill form.
///
/// # Safety
/// As `Fp8MoeModule::load`.
unsafe fn load_module(directory: &Path) -> Result<Fp8MoeModule> {
    let mut module = Fp8MoeModule::load(directory)?;
    let prefill = prefill_mode()?;
    module.set_prefill(prefill).with_context(|| format!("{} ({PREFILL_ENV})", directory.display()))?;
    if prefill != Fp8MoePrefill::default() {
        tracing::info!(package = %directory.display(), ?prefill, "FP8 expert prefill form");
    }
    Ok(module)
}

fn package_family(family: Option<&'static str>, format: ExpertFormat) -> &'static str {
    match (family, format) {
        (Some("mimo"), ExpertFormat::Mxfp4) => "mimof",
        (family, _) => family.unwrap_or("unknown"),
    }
}

/// `<libdir>/fp8/fp8-<family>[-nvfp4|-nvfp4a4]/tp<world>`: the package layout
/// serving TP degree `tp` of the process expert geometry in `format` (NVFP4
/// releases share their geometry with the FP8 ones and get packages of their
/// own). The W4A4 package (`-nvfp4a4`: large-row steps quantize activations
/// with the checkpoint's input_scale, as ModelOpt calibrated them) is the
/// default when built; `CUTEAFD_NVFP4_ACTIVATIONS=a16` keeps W4A16 (GLM 5.3
/// Flash: KL vs golden 0.0589 W4A16, 0.0791 W4A4; 8K prefill 1.33x).
pub(crate) fn package_directory(native_lib: &Path, tp: usize, format: ExpertFormat) -> PathBuf {
    let family = package_family(cuteafd_core::expert_geometry().family(), format);
    let root = native_lib.parent().unwrap_or(Path::new(".")).join("fp8");
    let layout = format!("tp{tp}");
    if format == ExpertFormat::Nvfp4 && nvfp4_activations() == Nvfp4Activations::A4 {
        let a4 = root.join(format!("fp8-{family}-nvfp4a4")).join(&layout);
        if a4.is_dir() {
            return a4;
        }
    }
    root.join(format!("fp8-{family}{}", format.package_suffix())).join(layout)
}

/// `<libdir>/fp8/fp8-<geometry>-nvfp4[a4]/tp1`: a coordinator NVFP4 dense-MLP
/// package (one always-selected expert), W4A4 when built unless
/// `CUTEAFD_NVFP4_ACTIVATIONS=a16`.
pub(crate) fn dense_package_directory(native_lib: &Path, geometry: &str) -> PathBuf {
    cuteafd_loader::placement::inventory::dense_package_directory(
        native_lib.parent().unwrap_or(Path::new(".")), geometry,
        nvfp4_activations() == Nvfp4Activations::A4)
}

/// How NVFP4 experts treat activations (`CUTEAFD_NVFP4_ACTIVATIONS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Nvfp4Activations {
    /// W4A4 large-row steps where a `-nvfp4a4` package is built (default).
    A4,
    /// W4A16 at every row count.
    A16,
}

pub(crate) fn nvfp4_activations() -> Nvfp4Activations {
    match std::env::var("CUTEAFD_NVFP4_ACTIVATIONS").as_deref() {
        Ok("a16") => Nvfp4Activations::A16,
        _ => Nvfp4Activations::A4,
    }
}

/// Exact slices: a package that also holds `tp<n>-w<width>` layouts (ranks
/// owning whole 128-row blocks, each stored at its own width) serves rank
/// `rank` from its width's layout when every rank's width is packaged (all
/// ranks decide alike). Otherwise the padded `tp<n>` layout.
/// CUTEAFD_FP8_EXACT_SLICES=0 keeps the padded layout.
pub(crate) fn exact_layout(directory: &Path, tensors: &Fp8ExpertTensors, tp: usize, rank: usize)
    -> (PathBuf, Slicing) {
    let padded = (directory.to_path_buf(), Slicing::Padded);
    if std::env::var("CUTEAFD_FP8_EXACT_SLICES").is_ok_and(|v| v == "0") || tp < 2 {
        return padded;
    }
    let (Some(package), Some(layout)) = (directory.parent(), directory.file_name().and_then(|n| n.to_str())) else {
        return padded;
    };
    let exact = Slicing::Blocks(128);
    let dir_of = |r: usize| tensors.rank_width(tp, r, exact).ok().map(|w| package.join(format!("{layout}-w{w}")));
    match (0..tp).map(dir_of).collect::<Option<Vec<_>>>() {
        Some(dirs) if dirs.iter().all(|d| d.is_dir()) => (dirs[rank].clone(), exact),
        _ => padded,
    }
}

/// The BF16-input sibling of an FP8 package directory:
/// `.../fp8-<family>/tp<n>` -> `.../fp8-<family>-bf16/tp<n>`.
pub(crate) fn bf16_sibling(directory: &Path) -> Option<PathBuf> {
    let layout = directory.file_name()?;
    let package = directory.parent()?;
    let name = package.file_name()?.to_str()?;
    Some(package.with_file_name(format!("{name}-bf16")).join(layout))
}

/// One layer's resident slice: per projection the weights of every expert
/// (`[E, rows, cols]`) and their scale grids (NVFP4: then the experts'
/// FP32 alphas).
pub(crate) struct Fp8Layer<'a> {
    pub layer: usize,
    regions: Vec<DeviceAllocation<'a>>,
}

impl<'a> Fp8Layer<'a> {
    /// Device bytes of one layer's slice.
    pub fn bytes(tensors: &Fp8ExpertTensors, tp: usize) -> Result<usize> {
        Self::bytes_for(tensors, tp, 0, Slicing::Padded)
    }

    /// Device bytes of rank `rank`'s slice of one layer under `slicing`.
    pub fn bytes_for(tensors: &Fp8ExpertTensors, tp: usize, rank: usize, slicing: Slicing) -> Result<usize> {
        let experts = tensors.shape().experts;
        Fp8Projection::ALL.iter().try_fold(0usize, |total, &p| {
            let (w, _) = tensors.slice_bytes_with(p, tp, rank, slicing)?;
            Ok(total + experts * w + tensors.scale_region_bytes_with(p, tp, rank, slicing)?)
        })
    }

    pub fn load(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, layer: usize, tp: usize, rank: usize)
        -> Result<Self> {
        Self::load_with(library, tensors, layer, tp, rank, Slicing::Padded)
    }

    pub fn load_with(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, layer: usize, tp: usize, rank: usize,
        slicing: Slicing) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        ensure!(tensors.has_layer(layer), "layer {layer} has no routed FP8 experts");
        let experts = tensors.shape().experts;
        let mut regions = Vec::with_capacity(6);
        // NVFP4 gate and up input scales, checked once both are read.
        let mut gate_inputs = Vec::new();
        for projection in Fp8Projection::ALL {
            let (w_bytes, s_bytes) = tensors.slice_bytes_with(projection, tp, rank, slicing)?;
            let mut weights = vec![0u8; experts * w_bytes];
            let mut scales = vec![0u8; tensors.scale_region_bytes_with(projection, tp, rank, slicing)?];
            // NVFP4: the experts' FP32 alphas, then their input scales, follow the scale grids.
            let (grids, scalars) = scales.split_at_mut(experts * s_bytes);
            let nvfp4 = !scalars.is_empty();
            let (alphas, inputs) = scalars.split_at_mut(scalars.len() / 2);
            let mut alpha_slots: Vec<(&mut [u8], &mut [u8])> =
                alphas.chunks_exact_mut(4).zip(inputs.chunks_exact_mut(4)).collect();
            alpha_slots.resize_with(experts, Default::default);
            let mut jobs: Vec<(usize, &mut [u8], &mut [u8], (&mut [u8], &mut [u8]))> = weights.chunks_exact_mut(w_bytes)
                .zip(grids.chunks_exact_mut(s_bytes)).zip(alpha_slots).enumerate()
                .map(|(e, ((w, s), a))| (e, w, s, a)).collect();
            let per = jobs.len().div_ceil(READERS);
            std::thread::scope(|scope| -> Result<()> {
                let handles: Vec<_> = jobs.chunks_mut(per).map(|chunk| scope.spawn(move || -> Result<()> {
                    let mut staging = Vec::new();
                    for (expert, w, s, (alpha, input)) in chunk.iter_mut() {
                        tensors.read_slice_with(layer, *expert, projection, tp, rank, slicing, w, s, &mut staging)?;
                        if nvfp4 {
                            alpha.copy_from_slice(&tensors.read_alpha(layer, *expert, projection)?.to_le_bytes());
                            input.copy_from_slice(&tensors.read_input_scale(layer, *expert, projection)?.to_le_bytes());
                        }
                    }
                    Ok(())
                })).collect();
                for handle in handles {
                    handle.join().map_err(|_| anyhow::anyhow!("FP8 expert reader panicked"))??;
                }
                Ok(())
            })?;
            if nvfp4 {
                // The W4A4 FC1 kernel quantizes with the gate projection's
                // input_scale and dequantizes the up half with the up
                // projection's; refuse a load whose two differ. Every NVFP4
                // expert-package load (routed, Spark, windowed) passes here.
                let inputs = &scales[scales.len() - experts * 4..];
                match projection {
                    Fp8Projection::Gate => gate_inputs = inputs.to_vec(),
                    Fp8Projection::Up => Fp8ExpertTensors::check_input_scales(layer, &gate_inputs, inputs)?,
                    Fp8Projection::Down => {}
                }
            }
            for bytes in [&weights, &scales] {
                let region = DeviceAllocation::new(library, bytes.len())?;
                library.copy_h2d(region.buffer, bytes)?;
                regions.push(region);
            }
        }
        // Regions in [w1, s1, w3, s3, w2, s2] order (gate, up, down).
        Ok(Self { layer, regions })
    }

    fn pointers(&self) -> [*mut c_void; 6] {
        std::array::from_fn(|i| self.regions[i].buffer.ptr)
    }
}

/// Resident FP8 layers of one TP slice and the package that runs them.
pub(crate) struct Fp8Experts<'a> {
    // Drop order: the module goes last; callers drain their streams first.
    pub layers: Vec<Fp8Layer<'a>>,
    scratch: DeviceAllocation<'a>,
    /// A second package over the same weights taking BF16 rows (a Spark
    /// rank whose coordinator sends unquantized expert input); it shares
    /// `scratch`, sized for both.
    pub bf16_module: Option<Fp8MoeModule>,
    pub module: Fp8MoeModule,
    pub tp: usize,
    pub rank: usize,
}

impl<'a> Fp8Experts<'a> {
    /// Loads the package at `directory` and layers `layers` of the slice
    /// `rank` of `tp`, with scratch for `capacity` rows.
    pub fn load(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: std::ops::Range<usize>, tp: usize, rank: usize, capacity: usize, budget: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        Self::load_with_bf16(library, tensors, directory, None, layers, tp, rank, capacity, budget)
    }

    /// Loads an optional BF16-input sibling over the same weights. Validate
    /// both packages and admit their maximum scratch before allocating weights.
    pub fn load_with_bf16(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        bf16_directory: Option<&Path>, layers: std::ops::Range<usize>, tp: usize, rank: usize, capacity: usize,
        budget: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        let mut experts = Self::prepare(library, tensors, directory, bf16_directory, layers.clone(), tp,
            rank, capacity, budget)?;
        let slicing = if directory.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("-w")) {
            Slicing::Blocks(128)
        } else { Slicing::Padded };
        experts.layers = layers.map(|layer| {
            let started = std::time::Instant::now();
            let loaded = Fp8Layer::load_with(library, tensors, layer, tp, rank, slicing)?;
            tracing::info!(layer, tp, rank, elapsed_ms = started.elapsed().as_millis() as u64,
                "FP8 expert layer resident");
            Ok(loaded)
        }).collect::<Result<Vec<_>>>()?;
        Ok(experts)
    }

    fn prepare(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        bf16_directory: Option<&Path>, layers: std::ops::Range<usize>, tp: usize, rank: usize, capacity: usize,
        budget: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        // `tp<n>-w<width>` layouts store each rank's own whole 128-row blocks (see `exact_layout`).
        let slicing = match directory.file_name().and_then(|n| n.to_str()).and_then(|n| n.split_once("-w")) {
            Some(_) => Slicing::Blocks(128),
            None => Slicing::Padded,
        };
        for layer in layers.clone() {
            tensors.validate_layer(layer)?;
        }
        // SAFETY: a trusted package for the current device; the owner drains
        // its streams before dropping.
        let module = unsafe { load_module(directory) }
            .with_context(|| format!("FP8 expert package {} (build it with package_fp8_moe_aot.py)",
                directory.display()))?;
        let info = module.info().clone();
        let shape = tensors.shape();
        let weights = match tensors.format() {
            ExpertFormat::Fp8Block128 => Fp8MoeWeights::Fp8,
            ExpertFormat::Mxfp4 => Fp8MoeWeights::Mxfp4,
            ExpertFormat::Nvfp4 => Fp8MoeWeights::Nvfp4 { w4a4: matches!(info.weights, Fp8MoeWeights::Nvfp4 { w4a4: true }) },
        };
        ensure!(info.hidden == shape.hidden && info.experts == shape.experts && info.topk == shape.topk
            && info.intermediate == shape.intermediate && info.tp == tp && info.weights == weights
            && info.slice == tensors.rank_width(tp, rank, slicing)?,
            "FP8 package {} ({info:?}) does not serve this checkpoint at TP{tp}", directory.display());
        let top = info.capacity_for(capacity)
            .with_context(|| format!("FP8 package has no capacity for {capacity} rows"))?;
        let primary_scratch = module.scratch_bytes(top)?;
        let bf16_module = bf16_directory.map(|directory| {
            // SAFETY: a trusted package for the current device; the owner
            // drains its streams before dropping either module.
            let bf16 = unsafe { load_module(directory) }
                .with_context(|| format!("BF16-input FP8 expert package {}", directory.display()))?;
            let top = validate_bf16_package(&info, bf16.info(), capacity)
                .with_context(|| format!("incompatible BF16-input FP8 expert package {}", directory.display()))?;
            let scratch = bf16.scratch_bytes(top)?;
            Ok::<_, anyhow::Error>((bf16, scratch))
        }).transpose()?;
        let (resident_bytes, scratch_bytes) = resident_admission(Fp8Layer::bytes_for(tensors, tp, rank, slicing)?,
            layers.len(), primary_scratch,
            bf16_module.as_ref().map(|(_, scratch)| *scratch), budget)?;
        tracing::info!(resident_bytes, scratch_bytes, budget, bf16_package = ?bf16_directory,
            "FP8 expert allocation admitted");
        let scratch = DeviceAllocation::new(library, scratch_bytes)?;
        Ok(Self { layers: Vec::with_capacity(layers.len()), scratch,
            bf16_module: bf16_module.map(|(module, _)| module), module, tp, rank })
    }

    /// Bounded P4 double banks: each checkpoint projection is read once, then
    /// both GPUs take their own row/column window from the same pinned bytes.
    pub fn load_pair(devices: [crate::shared::memory::device::Device<'a>; 2], tensors: &Fp8ExpertTensors,
        directories: [&Path; 2], slicing: [Slicing; 2], layers: std::ops::Range<usize>, capacity: usize,
        budgets: [usize; 2]) -> Result<[crate::shared::memory::device::DeviceOwner<'a, Self>; 2]> {
        use super::paired_load::{self, Fences};
        use crate::shared::memory::device::Device;
        use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer};
        use cuteafd_loader::formats::fp8_experts::Fp8MatrixSlice;
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        let started = std::time::Instant::now();
        ensure!(slicing[0] == slicing[1], "FP8 paired ranks disagree on slicing");
        let mut owners = [
            devices[0].own(|| Self::prepare(devices[0].library, tensors, directories[0], None,
                layers.clone(), 2, 0, capacity, budgets[0]))?,
            devices[1].own(|| Self::prepare(devices[1].library, tensors, directories[1], None,
                layers.clone(), 2, 1, capacity, budgets[1]))?,
        ];
        for layer in layers.clone() {
            for rank in 0..2 {
                devices[rank].run(|| {
                    let mut regions = Vec::with_capacity(6);
                    for projection in Fp8Projection::ALL {
                        let (weights, _) = tensors.slice_bytes_with(projection, 2, rank, slicing[rank])?;
                        for bytes in [weights * tensors.shape().experts,
                            tensors.scale_region_bytes_with(projection, 2, rank, slicing[rank])?] {
                            regions.push(DeviceAllocation::new(devices[rank].library, bytes)?);
                        }
                    }
                    owners[rank].layers.push(Fp8Layer { layer, regions });
                    Ok(())
                })?;
            }
        }
        let sizes = Fp8Projection::ALL.map(|p| tensors.projection_bytes(p));
        let mut offsets = [0; 3];
        let mut full_bytes = 0;
        let mut zero_bytes = 0;
        for (slot, projection) in Fp8Projection::ALL.into_iter().enumerate() {
            offsets[slot] = full_bytes;
            full_bytes += sizes[slot].0 + sizes[slot].1;
            for rank in 0..2 {
                let bytes = tensors.slice_bytes_with(projection, 2, rank, slicing[rank])?;
                zero_bytes = zero_bytes.max(bytes.0).max(bytes.1);
            }
        }
        let bank_bytes = full_bytes.checked_add(zero_bytes).context("FP8 paired bank overflow")?;
        let pinned_bytes = bank_bytes.checked_mul(READERS * 2).context("FP8 paired staging overflow")?;
        let mut hosts = paired_load::banks(devices[0], READERS, bank_bytes, pinned_bytes)?;
        let mut fences = Fences::new(devices)?;
        // Last owner drops first and drains both streams on every error path,
        // before pinned banks, device weights or package modules can be freed.
        let streams = paired_load::streams(devices)?;
        let allocation_seconds = started.elapsed().as_secs_f64();

        fn upload(device: Device<'_>, host: CuteafdHostBuffer, source_offset: usize,
            mut destination: CuteafdDeviceBuffer, destination_offset: usize, window: Fp8MatrixSlice,
            zero_offset: usize, stream: *mut c_void) -> Result<()> {
            ensure!(destination_offset.checked_add(window.destination_bytes).is_some_and(|n| n <= destination.bytes),
                "FP8 paired upload exceeds resident region");
            ensure!(source_offset.checked_add(window.source_offset).is_some_and(|n| n < host.bytes),
                "FP8 paired source exceeds pinned bank");
            device.run(|| {
                // SAFETY: windows are checked against the retained resident region;
                // pinned banks are fenced before reuse and streams drain on unwind.
                unsafe {
                    destination.ptr = destination.ptr.cast::<u8>().add(destination_offset).cast();
                    destination.bytes = window.destination_bytes;
                    let copied = window.width * window.rows;
                    if copied < window.destination_bytes {
                        let mut zeros = host;
                        zeros.ptr = zeros.ptr.cast::<u8>().add(zero_offset).cast();
                        zeros.bytes -= zero_offset;
                        device.library.copy_host_buffer_h2d_async(destination, zeros,
                            window.destination_bytes, stream)?;
                    }
                    let mut source = host;
                    let offset = source_offset + window.source_offset;
                    source.ptr = source.ptr.cast::<u8>().add(offset).cast();
                    source.bytes -= offset;
                    device.library.copy_host_buffer_h2d_2d_async(destination, window.destination_pitch, source,
                        window.source_pitch, window.width, window.rows, stream)
                }
            })
        }

        let mut group = 0;
        for (index, layer) in layers.enumerate() {
            let layer_started = std::time::Instant::now();
            let mut storage_bytes_read = 0;
            let mut read_seconds = 0.;
            let mut bank_wait_seconds = 0.;
            let mut upload_submit_seconds = 0.;
            for first in (0..tensors.shape().experts).step_by(READERS) {
                let count = READERS.min(tensors.shape().experts - first);
                let bank = group % 2;
                let wait = std::time::Instant::now();
                fences.reuse(bank)?;
                bank_wait_seconds += wait.elapsed().as_secs_f64();
                paired_load::trace_overlap(&streams, "read_start", layer, first)?;
                let read_started = std::time::Instant::now();
                let mut jobs = Vec::with_capacity(count * 3);
                for (lane, host) in hosts[bank][..count].iter_mut().enumerate() {
                    let mut bytes = &mut host.bytes_mut()[..full_bytes];
                    for (slot, projection) in Fp8Projection::ALL.into_iter().enumerate() {
                        let (source, rest) = bytes.split_at_mut(sizes[slot].0 + sizes[slot].1);
                        let (weight, scale) = source.split_at_mut(sizes[slot].0);
                        jobs.push((first + lane, projection, weight, scale));
                        bytes = rest;
                    }
                }
                jobs.sort_unstable_by_key(|(_, _, w, s)| w.len() + s.len());
                storage_bytes_read += paired_load::read_jobs(jobs, READERS * 2,
                    |(expert, projection, weight, scale)| tensors.read_projection_once(layer, expert, projection,
                        weight, scale))?;
                if tensors.format() == ExpertFormat::Nvfp4 {
                    for (lane, host) in hosts[bank][..count].iter().enumerate() {
                        let bytes = host.bytes();
                        let gate = offsets[0] + sizes[0].0 + sizes[0].1 - 4;
                        let up = offsets[1] + sizes[1].0 + sizes[1].1 - 4;
                        Fp8ExpertTensors::check_input_scales(layer, &bytes[gate..gate + 4], &bytes[up..up + 4])
                            .with_context(|| format!("shared projection expert {}", first + lane))?;
                    }
                }
                read_seconds += read_started.elapsed().as_secs_f64();
                let upload_started = std::time::Instant::now();
                for rank in 0..2 {
                    for (lane, host) in hosts[bank][..count].iter().enumerate() {
                        let expert = first + lane;
                        for (slot, projection) in Fp8Projection::ALL.into_iter().enumerate() {
                            let windows = tensors.projection_slices(projection, 2, rank, slicing[rank])?;
                            let regions = &owners[rank].layers[index].regions;
                            upload(devices[rank], host.buffer, offsets[slot], regions[2 * slot].buffer,
                                expert * windows[0].destination_bytes, windows[0], full_bytes, streams[rank].raw)?;
                            upload(devices[rank], host.buffer, offsets[slot] + sizes[slot].0, regions[2 * slot + 1].buffer,
                                expert * windows[1].destination_bytes, windows[1], full_bytes, streams[rank].raw)?;
                            if tensors.format() == ExpertFormat::Nvfp4 {
                                let grid_bytes = windows[1].destination_bytes * tensors.shape().experts;
                                for scalar in 0..2 {
                                    let source = offsets[slot] + sizes[slot].0 + sizes[slot].1 - 8 + scalar * 4;
                                    upload(devices[rank], host.buffer, source, regions[2 * slot + 1].buffer,
                                        grid_bytes + scalar * tensors.shape().experts * 4 + expert * 4,
                                        Fp8MatrixSlice { source_offset: 0, source_pitch: 4, destination_pitch: 4,
                                            width: 4, rows: 1, destination_bytes: 4 }, full_bytes, streams[rank].raw)?;
                                }
                            }
                        }
                    }
                }
                upload_submit_seconds += upload_started.elapsed().as_secs_f64();
                fences.record(bank, &streams)?;
                paired_load::trace_overlap(&streams, "both_submitted", layer, first)?;
                group += 1;
            }
            tracing::info!(layer, storage_bytes_read, read_seconds, bank_wait_seconds, upload_submit_seconds,
                elapsed_seconds = layer_started.elapsed().as_secs_f64(), "FP8 TP2 shared-read layer timeline");
        }
        paired_load::drain(&streams)?;
        tracing::info!(allocation_seconds, pinned_host_bytes = pinned_bytes, read_scratch_bytes = 0,
            elapsed_seconds = started.elapsed().as_secs_f64(), "FP8 TP2 shared-read load complete");
        Ok(owners)
    }

    pub fn index_of(&self, layer: usize) -> Result<usize> {
        self.layers.iter().position(|l| l.layer == layer)
            .with_context(|| format!("FP8 expert layer {layer} is not resident"))
    }

    /// Exact device allocation sizes, without querying CUDA after admission.
    pub fn resident_bytes(&self) -> usize {
        self.layers.iter().flat_map(|layer| &layer.regions).map(|region| region.buffer.bytes).sum()
    }

    pub fn scratch_bytes(&self) -> usize {
        self.scratch.buffer.bytes
    }

    /// Whether the package takes FP8 K32 wire rows (else BF16 rows).
    pub fn wire_input(&self) -> bool {
        self.module.info().wire_input
    }

    /// Routed experts of resident layer `index` for `rows` input rows (wire
    /// or BF16, per `wire_input`) into `out` (BF16 `[rows, H]`).
    ///
    /// # Safety
    /// `wire`, `ids` (I32 `[rows, k]`), `weights` (F32 `[rows, k]`) and `out`
    /// are live device buffers of those extents; the stream is drained before
    /// any of them, or this object, is released.
    pub unsafe fn run(&self, index: usize, rows: usize, wire: *mut c_void, ids: *mut c_void, weights: *mut c_void,
        out: *mut c_void, stream: *mut c_void) -> Result<()> {
        self.run_with(&self.module, index, rows, wire, ids, weights, out, stream)
    }

    /// `run` over BF16 input rows through the BF16-input package.
    ///
    /// # Safety
    /// As `run`, with `rows` BF16 `[rows, H]` input rows.
    pub unsafe fn run_bf16(&self, index: usize, rows: usize, input: *mut c_void, ids: *mut c_void,
        weights: *mut c_void, out: *mut c_void, stream: *mut c_void) -> Result<()> {
        let module = self.bf16_module.as_ref().context("no BF16-input FP8 expert package is loaded")?;
        self.run_with(module, index, rows, input, ids, weights, out, stream)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn run_with(&self, module: &Fp8MoeModule, index: usize, rows: usize, input: *mut c_void,
        ids: *mut c_void, weights: *mut c_void, out: *mut c_void, stream: *mut c_void) -> Result<()> {
        let layer = self.layers.get(index).context("FP8 expert layer index out of range")?;
        let [w1, s1, w3, s3, w2, s2] = layer.pointers();
        let pointers: [*mut c_void; FP8_MOE_POINTERS] =
            [input, ids, weights, w1, s1, w3, s3, w2, s2, out, self.scratch.buffer.ptr];
        module.launch(&pointers, rows, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::{resident_admission, validate_bf16_package};
    use clap::Parser;
    use cuteafd_ffi::fp8_moe::{Fp8MoeInfo, Fp8MoeWeights};

    #[test]
    fn package_family_includes_format_without_changing_existing_families() {
        use cuteafd_loader::formats::fp8_experts::ExpertFormat;
        assert_eq!(super::package_family(Some("mimo"), ExpertFormat::Mxfp4), "mimof");
        for family in ["mimo", "mimop", "glm", "glmf", "qwen4", "dsv4f"] {
            assert_eq!(super::package_family(Some(family), ExpertFormat::Fp8Block128), family);
            assert_eq!(super::package_family(Some(family), ExpertFormat::Nvfp4), family);
            if family != "mimo" { assert_eq!(super::package_family(Some(family), ExpertFormat::Mxfp4), family); }
        }
    }

    #[derive(Parser)]
    struct QwenArgs {
        #[command(flatten)]
        engine: crate::families::qwen4::EngineArgs,
    }

    #[derive(Parser)]
    struct GlmfArgs {
        #[command(flatten)]
        engine: crate::families::glm5_flash::EngineArgs,
    }

    #[test]
    fn local_expert_paging_requires_an_explicit_diagnostic_window() {
        let args = ["test", "--snapshot", "/snapshot", "--native-lib", "/native.so", "--local-experts"];
        let qwen = QwenArgs::try_parse_from(args).unwrap();
        let glmf = GlmfArgs::try_parse_from(args).unwrap();
        assert_eq!(qwen.engine.expert_window, None);
        assert_eq!(glmf.engine.expert_window, None);
        assert_eq!(qwen.engine.expert_reserve_gib, 12);
        assert_eq!(glmf.engine.expert_reserve_gib, 12);
        let paged: Vec<_> = args.into_iter().chain(["--expert-window", "16"]).collect();
        assert_eq!(QwenArgs::try_parse_from(&paged).unwrap().engine.expert_window, Some(16));
        assert_eq!(GlmfArgs::try_parse_from(&paged).unwrap().engine.expert_window, Some(16));
        let no_local = ["test", "--snapshot", "/snapshot", "--native-lib", "/native.so", "--expert-window", "1"];
        assert!(QwenArgs::try_parse_from(no_local).is_err());
        assert!(GlmfArgs::try_parse_from(no_local).is_err());
    }

    #[test]
    #[ignore = "requires two CUDA GPUs, CUTEAFD_NATIVE_LIB, CUTEAFD_FP8_TP2_SNAPSHOT and CUTEAFD_FP8_TP2_PACKAGE"]
    fn paired_uploads_are_byte_exact_against_checkpoint_rank_slices() -> anyhow::Result<()> {
        use super::{exact_layout, ExpertFormat, Fp8Experts, Fp8ExpertTensors, Fp8Layer, Fp8Projection, NativeLibrary};
        use anyhow::ensure;
        use crate::shared::memory::device::Device;
        use cuteafd_ffi::fp8_moe::Fp8MoeMetadata;
        use cuteafd_loader::RoutedExpertShape;
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_FP8_TP2_SNAPSHOT")?);
        let package = std::path::PathBuf::from(std::env::var("CUTEAFD_FP8_TP2_PACKAGE")?);
        let layer = std::env::var("CUTEAFD_FP8_TP2_LAYER").unwrap_or_else(|_| "3".into()).parse::<usize>()?;
        // SAFETY: qualification paths name this task's trusted native/package artifacts.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        // SAFETY: the same trusted package supplies only static geometry/scratch ABI here.
        let metadata = unsafe { Fp8MoeMetadata::read(&package) }?;
        let info = &metadata.info;
        let tensors = Fp8ExpertTensors::read(&snapshot, RoutedExpertShape { layers: layer + 1,
            first_layer: layer, experts: info.experts, topk: info.topk, hidden: info.hidden,
            intermediate: info.intermediate, draft_stages: 0, draft_experts: 0 })?;
        let layouts = [exact_layout(&package, &tensors, 2, 0), exact_layout(&package, &tensors, 2, 1)];
        let devices = [Device { library: &library, id: 0 }, Device { library: &library, id: 1 }];
        let budgets = [0, 1].map(|rank| Fp8Layer::bytes_for(&tensors, 2, rank, layouts[rank].1)
            .and_then(|resident| Ok(resident + metadata.scratch_for(1)?.max(256))));
        let [left, right] = budgets;
        let owners = Fp8Experts::load_pair(devices, &tensors, [&layouts[0].0, &layouts[1].0],
            layouts.each_ref().map(|layout| layout.1), layer..layer + 1, 1, [left?, right?])?;
        for rank in 0..2 {
            for (slot, projection) in Fp8Projection::ALL.into_iter().enumerate() {
                let sizes = tensors.slice_bytes_with(projection, 2, rank, layouts[rank].1)?;
                let mut weights = vec![0; sizes.0 * info.experts];
                let mut scales = vec![0; tensors.scale_region_bytes_with(projection, 2, rank, layouts[rank].1)?];
                for expert in 0..info.experts {
                    tensors.read_slice_with(layer, expert, projection, 2, rank, layouts[rank].1,
                        &mut weights[expert * sizes.0..][..sizes.0], &mut scales[expert * sizes.1..][..sizes.1],
                        &mut Vec::new())?;
                    if tensors.format() == ExpertFormat::Nvfp4 {
                        let scalar = sizes.1 * info.experts;
                        scales[scalar + expert * 4..][..4]
                            .copy_from_slice(&tensors.read_alpha(layer, expert, projection)?.to_le_bytes());
                        scales[scalar + info.experts * 4 + expert * 4..][..4]
                            .copy_from_slice(&tensors.read_input_scale(layer, expert, projection)?.to_le_bytes());
                    }
                }
                for (region, expected) in [(2 * slot, weights), (2 * slot + 1, scales)] {
                    let mut actual = vec![0; expected.len()];
                    devices[rank].run(|| library.copy_d2h(&mut actual, owners[rank].layers[0].regions[region].buffer))?;
                    ensure!(actual == expected, "TP2 paired upload differs: rank {rank} layer {layer} {projection:?} region {region}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn resident_admission_includes_every_layer_and_scratch() {
        let (layer, scratch) = (1_415_589_888, 64 << 20);
        let required = layer * 48 + scratch;
        assert_eq!(resident_admission(layer, 48, scratch, None, required).unwrap(), (layer * 48, scratch));
        assert!(resident_admission(layer, 48, scratch, None, required - 1).is_err());
        // An explicit paging window admits only the empty initial window's scratch.
        assert_eq!(resident_admission(layer, 0, scratch, None, scratch).unwrap(), (0, scratch));
        assert!(resident_admission(layer, 0, scratch, None, scratch - 1).is_err());
    }

    #[test]
    fn resident_admission_sizes_one_arena_for_both_packages() {
        let (layer, primary, bf16) = (1024, 512, 2048);
        let required = layer * 48 + bf16;
        assert_eq!(resident_admission(layer, 48, primary, Some(bf16), required).unwrap(), (layer * 48, bf16));
        assert!(resident_admission(layer, 48, primary, Some(bf16), required - 1).is_err());
        // Sharing requires the maximum, rather than two scratch allocations.
        assert_eq!(resident_admission(layer, 48, bf16, Some(primary), required).unwrap(), (layer * 48, bf16));
        assert_eq!(resident_admission(0, 0, 0, Some(0), 256).unwrap(), (0, 256));
        assert!(resident_admission(0, 0, 0, None, 255).is_err());
    }

    #[test]
    fn resident_admission_rejects_overflow() {
        assert!(resident_admission(usize::MAX, 2, 0, None, usize::MAX).is_err());
        assert!(resident_admission(usize::MAX, 1, 1, None, usize::MAX).is_err());
        assert!(resident_admission(1, 1, 0, Some(usize::MAX), usize::MAX).is_err());
    }

    #[test]
    fn bf16_sibling_must_share_weight_layout_arithmetic_and_capacity() {
        let primary = Fp8MoeInfo { hidden: 6144, slice: 512, experts: 256, topk: 8, intermediate: 2048, tp: 4,
            wire_input: true, swiglu_limit: 10.0, capacities: vec![64, 4096], weights: Fp8MoeWeights::Fp8 };
        let mut bf16 = primary.clone();
        bf16.wire_input = false;
        // A sibling may compile different capacities; its own scratch uses its own top.
        bf16.capacities = vec![128, 8192];
        assert_eq!(validate_bf16_package(&primary, &bf16, 64).unwrap(), 128);
        assert_eq!(validate_bf16_package(&primary, &bf16, 4096).unwrap(), 8192);
        assert!(validate_bf16_package(&primary, &bf16, 8193).is_err());
        for variant in 0..5 {
            let mut incompatible = bf16.clone();
            match variant {
                0 => incompatible.wire_input = true,
                1 => incompatible.slice = 640,
                2 => incompatible.swiglu_limit = f32::INFINITY,
                3 => incompatible.tp = 2,
                _ => incompatible.weights = Fp8MoeWeights::Mxfp4,
            }
            assert!(validate_bf16_package(&primary, &incompatible, 64).is_err(), "{incompatible:?}");
        }
    }
}
