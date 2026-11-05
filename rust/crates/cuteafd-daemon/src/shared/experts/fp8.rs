//! Exact FP8 routed experts (the `fp8` family): the checkpoint's E4M3 expert
//! weights with FP32 128x128 block scales (or MXFP4, MiMo V2.6 Pro:
//! `fp8-mimop` packages; or ModelOpt NVFP4 run W4A4, `fp8-<family>-nvfp4a4`
//! packages), resident per TP slice and run by
//! an `fp8-<family>` package (`python/tools/aot/package_fp8_moe_aot.py`,
//! `native/shared/include/cuteafd_fp8_moe.h`). Nothing is re-quantized. Output is
//! the BF16 `[rows, H]` route sum of the slice: the Spark rank partial of the
//! compact BF16 response, or the whole layer at TP1 on the coordinator.
pub(crate) mod worker;
#[cfg(test)]
#[path = "fp8/checkpoint_tests.rs"]
mod checkpoint_tests;

use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::fp8_moe::{Fp8MoeActivation, Fp8MoeProjection, Fp8MoeMetadata, Fp8MoeInfo, Fp8MoeModule, Fp8MoePrefill, Fp8MoeWeights, FP8_MOE_POINTERS};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection, Slicing};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use crate::shared::memory::weight_arena::ReadyArena;

pub(crate) fn validate_reply_formats(format: ExpertFormat,
    replies: cuteafd_core::spark_reply::SparkReplyFormats) -> Result<()> {
    ensure!(format != ExpertFormat::Nvfp4 || !replies.can_compress(),
        "NVFP4-weight Spark packages support only BF16 replies; compressed prefill/decode replies are unsupported; select bf16 for both reply formats");
    Ok(())
}

#[test]
fn nvfp4_weights_reject_compressed_reply_selection_at_startup() {
    use cuteafd_core::{spark_reply::SparkReplyFormats, SparkReplyFormat::*};
    use cuteafd_loader::formats::fp8_experts::ExpertFormat;
    for prefill in [None, Some(Bf16), Some(Fp8), Some(Nvfp4)] {
        for decode in [None, Some(Bf16), Some(Fp8), Some(Nvfp4)] {
            let replies = SparkReplyFormats { prefill, decode };
            let result = validate_reply_formats(ExpertFormat::Nvfp4, replies);
            assert_eq!(result.is_err(), replies.can_compress());
            if let Err(error) = result { assert!(error.to_string().contains("only BF16 replies")); }
            for format in [ExpertFormat::Fp8Block128, ExpertFormat::Mxfp4, ExpertFormat::Bf16] {
                validate_reply_formats(format, replies).unwrap();
            }
        }
    }
}

pub(super) fn validate_checkpoint_contract(info: &Fp8MoeInfo, tensors: &Fp8ExpertTensors, layers: &[usize],
    tp: usize, rank: usize, slicing: Slicing) -> Result<()> {
    let shape = tensors.shape();
    ensure!(info.hidden == shape.hidden && info.experts == shape.experts && info.topk == shape.topk
        && info.intermediate == shape.intermediate && info.tp == tp,
        "expert package does not serve checkpoint geometry at TP{tp}: {info:?}");
    for &layer in layers {
        let view = tensors.for_layer_metadata(layer)?;
        let weights = match view.format() {
            ExpertFormat::Bf16 => Fp8MoeWeights::Bf16,
            ExpertFormat::Fp8Block128 => Fp8MoeWeights::Fp8,
            ExpertFormat::Mxfp4 => Fp8MoeWeights::Mxfp4,
            ExpertFormat::Nvfp4 => Fp8MoeWeights::Nvfp4,
        };
        ensure!(info.weights == weights,
            "expert layer {layer}: checkpoint dtype/layout {:?} has no matching RTX package (selected {:?})",
            view.format(), info.weights);
        ensure!(view.rank_width(tp, rank, slicing)? == info.slice,
            "expert layer {layer}: package slice {} differs from checkpoint TP{tp} rank {rank}", info.slice);
        ensure!(!matches!(weights, Fp8MoeWeights::Bf16 | Fp8MoeWeights::Nvfp4) || !info.wire_input,
            "expert layer {layer}: native BF16/NVFP4 weights require BF16 input");
        view.validate_layer(layer)?;
        let activation = tensors.activation(layer).unwrap_or(shape.geometry()?.activation());
        let (mode, limit) = match activation {
            cuteafd_core::FfnActivation::SiluBf16Product { limit } =>
                (Fp8MoeActivation::SiluBf16Product, limit.unwrap_or(0.0)),
            cuteafd_core::FfnActivation::SiluPostClampFp32Product { limit } =>
                (Fp8MoeActivation::SiluPostClampFp32Product, limit),
            cuteafd_core::FfnActivation::Situ { .. } =>
                anyhow::bail!("expert layer {layer}: native FP8 packages do not export the checkpoint's Situ activation"),
        };
        info.validate_activation(mode, limit, Fp8MoeProjection::Bf16)
            .with_context(|| format!("expert layer {layer}"))?;
    }
    Ok(())
}

pub(super) fn package_metadata(directory: &Path, tensors: &Fp8ExpertTensors, layers: &[usize],
    tp: usize, rank: usize, slicing: Slicing) -> Result<Fp8MoeMetadata> {
    // SAFETY: only trusted generated native packages are accepted by the service.
    let metadata = unsafe { Fp8MoeMetadata::read(directory) }
        .with_context(|| format!("expert package {}", directory.display()))?;
    validate_checkpoint_contract(&metadata.info, tensors, layers, tp, rank, slicing)?;

    Ok(metadata)
}

pub(super) fn validate_reply_exports(directory: &Path, format: ExpertFormat) -> Result<()> {
    if cuteafd_core::spark_reply::requested().can_compress() {
        validate_reply_formats(format, cuteafd_core::spark_reply::requested())?;
        let root = directory.parent().context("expert package has no root")?;
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let layout = directory.file_name().and_then(|name| name.to_str()).context("expert layout name")?;
        let available = manifest["layouts"][layout]["reply_producers"].as_array().is_some_and(|names| !names.is_empty());
        cuteafd_ffi::programs::require_reply_export(
            &format!("{}/libcuteafd_fp8moe.so:core-only producers", directory.display()), available)?;
    }
    Ok(())
}

fn traffic_contract(contract: cuteafd_ffi::fp8_moe::Fp8MoeTrafficContract)
    -> cuteafd_transport::worker_selection::ExpertTrafficContract {
    use cuteafd_ffi::fp8_moe::Fp8MoeTrafficContract as Ffi;
    use cuteafd_transport::worker_selection::ExpertTrafficContract as Wire;
    match contract { Ffi::ExactGrouping => Wire::ExactGrouping,
        Ffi::ConservativeRouteUpperBound => Wire::ConservativeRouteUpperBound }
}

fn installed_variants(module: &Fp8MoeModule)
    -> Result<Vec<cuteafd_transport::worker_selection::InstalledExpertVariant>> {
    module.info().capacities.iter().map(|&capacity| {
        let (_, contract) = module.weight_read_group_for_capacity(capacity, 0)?;
        cuteafd_transport::worker_selection::InstalledExpertVariant::from_piecewise_queries(
            capacity, traffic_contract(contract), |rows| {
                let (group, current) = module.weight_read_group_for_capacity(capacity, rows)?;
                ensure!(current == contract, "FP8 expert traffic contract changes within a capacity");
                Ok(group)
            })?.with_arithmetic_routes(|rows| module.arithmetic_route_for_capacity(capacity, rows))
    }).collect()
}

/// Parallel readers per layer load.
const READERS: usize = cuteafd_loader::formats::fp8_experts::FP8_LOAD_READERS;

fn validate_layer_ids(layers: &[usize]) -> Result<()> {
    ensure!(layers.windows(2).all(|pair| pair[0] < pair[1]),
        "FP8 expert layer IDs must be sorted and unique");
    Ok(())
}

/// Resident admission contains only weights. Execution owners admit scratch separately.
fn resident_admission(layer_bytes: usize, layers: usize, budget: usize) -> Result<usize> {
    let resident = layer_bytes.checked_mul(layers).context("FP8 resident expert size overflow")?;
    ensure!(resident <= budget, "FP8 experts need {resident} resident bytes; budget {budget}");
    Ok(resident)
}

/// Both programs share the same resident slice and scratch arena. Input dtype
/// and compiled capacities may differ, but the weight layout and arithmetic may not.
pub(super) fn validate_bf16_package(main: &Fp8MoeInfo, bf16: &Fp8MoeInfo, capacity: usize) -> Result<usize> {
    ensure!(!bf16.wire_input && bf16.weights == main.weights && bf16.hidden == main.hidden
        && bf16.experts == main.experts && bf16.topk == main.topk && bf16.intermediate == main.intermediate
        && bf16.tp == main.tp && bf16.slice == main.slice && bf16.swiglu_limit == main.swiglu_limit
        && bf16.activation == main.activation && bf16.projection == main.projection,
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

pub(super) fn package_family(family: Option<&'static str>, format: ExpertFormat) -> &'static str {
    match (family, format) {
        (Some("mimo"), ExpertFormat::Mxfp4) => "mimof",
        (family, _) => family.unwrap_or("unknown"),
    }
}

/// `<libdir>/fp8/fp8-<family>[-nvfp4a4]/tp<world>`: the package layout
/// serving TP degree `tp` of the process expert geometry in `format` (NVFP4
/// releases share their geometry with the FP8 ones and get packages of their
/// own). NVFP4 always uses W4A4, quantizing original BF16 activations with
/// each expert's checkpoint input_scale at every row count.
pub(crate) fn package_directory(native_lib: &Path, tp: usize, format: ExpertFormat) -> PathBuf {
    package_directory_for(native_lib, tp, format, cuteafd_core::expert_geometry().family())
}

fn package_directory_for(native_lib: &Path, tp: usize, format: ExpertFormat,
    family: Option<&'static str>) -> PathBuf {
    let family = package_family(family, format);
    let root = native_lib.parent().unwrap_or(Path::new(".")).join("fp8");
    let layout = format!("tp{tp}");
    root.join(format.package_name(family)).join(layout)
}

/// `<libdir>/fp8/fp8-<geometry>-nvfp4a4/tp1`: mandatory coordinator W4A4
/// dense-MLP package (one always-selected expert).
pub(crate) fn dense_package_directory(native_lib: &Path, geometry: &str) -> PathBuf {
    cuteafd_loader::placement::inventory::dense_package_directory(
        native_lib.parent().unwrap_or(Path::new(".")), geometry)
}

/// Exact slices: a package that also holds `tp<n>-w<width>` layouts (ranks
/// owning whole 128-row blocks, each stored at its own width) serves rank
/// `rank` from its width's layout when every rank's width is packaged (all
/// ranks decide alike). Otherwise the padded `tp<n>` layout.
/// CUTEAFD_FP8_EXACT_SLICES=0 keeps the padded layout.
pub(crate) fn exact_layout(directory: &Path, tensors: &Fp8ExpertTensors, tp: usize, rank: usize)
    -> (PathBuf, Slicing) {
    cuteafd_loader::formats::fp8_experts::exact_layout_directory(directory, tensors, tp, rank)
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
    pub(crate) slice: cuteafd_core::SparkSlice,
    regions: Vec<Option<Fp8Region<'a>>>,
}

struct Fp8Region<'a> {
    buffer: cuteafd_ffi::CuteafdDeviceBuffer,
    _storage: Fp8Storage<'a>,
}
enum Fp8Storage<'a> {
    Owned { _allocation: DeviceAllocation<'a> },
    Shared { _arena: Rc<ReadyArena<'a>> },
}
impl<'a> From<DeviceAllocation<'a>> for Fp8Region<'a> {
    fn from(allocation: DeviceAllocation<'a>) -> Self {
        Self { buffer: allocation.buffer, _storage: Fp8Storage::Owned { _allocation: allocation } }
    }
}

impl<'a> Fp8Layer<'a> {
    pub(crate) fn allocation_receipts(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>,
        name: &str) -> Result<()> {
        for (index, region) in self.regions.iter().enumerate() {
            if let Some(region) = region {
                let owner = match &region._storage {
                    Fp8Storage::Owned { _allocation } => _allocation.buffer,
                    Fp8Storage::Shared { _arena } => _arena.allocation_buffer(),
                };
                inventory.backed_device(&format!("{name}.plane.{index}"), region.buffer, owner)?;
            }
        }
        Ok(())
    }

    pub(crate) fn from_push(owner: Rc<ReadyArena<'a>>, region: usize,
        plan: &crate::shared::memory::push_fp8::LayerPlan, base: usize) -> Result<Self> {
        let view = owner.region(region)?;
        let regions = plan.planes.iter().map(|plane| {
            if plane.is_empty() { return Ok(None); }
            let offset = plane.start.checked_sub(base).context("native push plane precedes layer")?;
            let buffer = view.subregion(offset, plane.len())?.buffer();
            Ok(Some(Fp8Region { buffer, _storage: Fp8Storage::Shared { _arena: owner.clone() } }))
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self { layer: plan.layer, slice: plan.slice.clone(), regions })
    }

    fn loaded_slice(tensors: &Fp8ExpertTensors, layer: usize, tp: usize, rank: usize,
        slicing: Slicing) -> Result<cuteafd_core::SparkSlice> {
        // The unit-map seal includes the absolute layer, even for contiguous TP2.
        crate::shared::memory::push_fp8::slice(tensors, layer, tp, rank, slicing)
    }

    fn validate_ownership(&self, tensors: &Fp8ExpertTensors, layer: usize, tp: usize,
        rank: usize, slicing: Slicing) -> Result<()> {
        let expected = Self::loaded_slice(tensors, layer, tp, rank, slicing)?;
        ensure!(self.layer == layer && self.slice == expected,
            "pushed FP8-family layer {layer} ownership differs from admission: tp={tp} rank={rank} slicing={slicing:?} pushed={:?} expected={expected:?}",
            self.slice);
        Ok(())
    }

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
            let load = tensors.projection_load(layer, projection, tp, rank, slicing)?;
            let mut weights = vec![0u8; load.weights];
            let mut scales = vec![0u8; load.scales];
            // NVFP4: the experts' FP32 alphas, then their input scales, follow the scale grids.
            let (grids, scalars) = scales.split_at_mut(experts * s_bytes);
            let nvfp4 = !scalars.is_empty();
            let (alphas, inputs) = scalars.split_at_mut(scalars.len() / 2);
            let mut alpha_slots: Vec<(&mut [u8], &mut [u8])> =
                alphas.chunks_exact_mut(4).zip(inputs.chunks_exact_mut(4)).collect();
            alpha_slots.resize_with(experts, Default::default);
            let grids: Vec<&mut [u8]> = if s_bytes == 0 {
                (0..experts).map(|_| &mut [][..]).collect()
            } else { grids.chunks_exact_mut(s_bytes).collect() };
            let mut jobs: Vec<(usize, &mut [u8], &mut [u8], (&mut [u8], &mut [u8]))> = weights.chunks_exact_mut(w_bytes)
                .zip(grids).zip(alpha_slots).enumerate()
                .map(|(e, ((w, s), a))| (e, w, s, a)).collect();
            let per = jobs.len().div_ceil(READERS);
            std::thread::scope(|scope| -> Result<()> {
                let handles: Vec<_> = jobs.chunks_mut(per).map(|chunk| scope.spawn(move || -> Result<()> {
                    let mut staging = Vec::with_capacity(load.reader_staging);
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
                let region = if bytes.is_empty() { None } else {
                    let region = DeviceAllocation::new(library, bytes.len())?;
                    library.copy_h2d(region.buffer, bytes)?;
                    Some(region.into())
                };
                regions.push(region);
            }
        }
        // Regions in [w1, s1, w3, s3, w2, s2] order (gate, up, down).
        Ok(Self { layer, slice: Self::loaded_slice(tensors, layer, tp, rank, slicing)?, regions })
    }

    fn pointers(&self) -> [*mut c_void; 6] {
        std::array::from_fn(|i| self.regions[i].as_ref().map_or(std::ptr::null_mut(), |region| region.buffer.ptr))
    }
}

/// Code-only owners; no weights, scratch, streams or graph storage.
pub(crate) struct Fp8Modules {
    directory: PathBuf,
    bf16_directory: Option<PathBuf>,
    module: Fp8MoeModule,
    bf16: Option<Fp8MoeModule>,
}
impl Fp8Modules {
    pub(crate) fn segment_pinned_bytes(&self, extent: cuteafd_core::OwnerExtent) -> Result<usize> {
        Ok(if segment_rows(&self.module, extent).is_some() { extent.staging_rows as usize * 16 } else { 0 })
    }

    pub(crate) fn owner_workspace_bytes(&self, extent: cuteafd_core::OwnerExtent) -> Result<(usize, usize)> {
        let (mut scratch, mut io) = self.workspace_bytes(extent.capacity as usize)?;
        if let Some(rows) = segment_rows(&self.module, extent) {
            scratch = scratch.max(self.module.segment_scratch_bytes(rows)?);
            io += worker::Fp8Worker::segment_io_bytes(&self.module, extent, self.bf16.is_some())?;
        }
        Ok((scratch, io))
    }

    pub(crate) fn workspace_bytes(&self, capacity: usize) -> Result<(usize, usize)> {
        use cuteafd_ffi::fp8_moe::{fp8_slot_io_bytes, fp8_slot_scratch_bytes};
        let mut scratch = fp8_slot_scratch_bytes(self.module.scratch_layout(), capacity)?;
        if let Some(module) = &self.bf16 {
            scratch = scratch.max(fp8_slot_scratch_bytes(module.scratch_layout(), capacity)?);
        }
        let info = self.module.info();
        Ok((scratch, fp8_slot_io_bytes(info.hidden, info.topk, info.wire_input,
            self.bf16.is_some(), capacity)?))
    }

    pub(crate) fn load(directory: &Path, bf16_directory: Option<&Path>) -> Result<Self> {
        // SAFETY: selected trusted AOT packages for the current owner device.
        let module = unsafe { load_module(directory) }?;
        // SAFETY: the sibling has the same trusted-package/current-device contract.
        let bf16 = bf16_directory.map(|path| unsafe { load_module(path) }).transpose()?;
        Ok(Self { directory: directory.into(), bf16_directory: bf16_directory.map(Path::to_owned),
            module, bf16 })
    }
}

/// Shared resident layers and code of one TP slice; no execution scratch.
pub(crate) struct Fp8Weights<'a> {
    // Drop order: the module goes last; callers drain their streams first.
    pub layers: Vec<Fp8Layer<'a>>,
    /// A second package over the same weights taking BF16 rows (a Spark
    /// rank whose coordinator sends unquantized expert input).
    pub bf16_module: Option<Fp8MoeModule>,
    pub module: Fp8MoeModule,
    pub tp: usize,
    pub rank: usize,
    weight_read_bytes: usize,
}

impl<'a> Fp8Weights<'a> {
    pub(crate) fn packed_row_cap(&self, bf16: bool, rows: usize) -> Result<usize> {
        let module = if bf16 && self.wire_input() {
            self.bf16_module.as_ref().context("no BF16-input FP8 expert package is loaded")?
        } else { &self.module };
        Ok(installed_variants(module)?.iter().filter_map(|v| v.packed_row_cap()).min()
            .unwrap_or(rows).min(rows))
    }

    /// Loads resident weights and code; the execution owner allocates scratch.
    pub fn load(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: &[usize], tp: usize, rank: usize, capacity: usize, budget: usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        Self::load_with_bf16(library, tensors, directory, None, layers, tp, rank, capacity, budget)
    }

    /// Loads an optional BF16-input sibling over the same weights. Validate
    /// both packages before allocating weights; executions admit their scratch.
    pub fn load_with_bf16(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        bf16_directory: Option<&Path>, layers: &[usize], tp: usize, rank: usize, capacity: usize,
        budget: usize) -> Result<Self> {
        Self::load_with_modules(library, tensors, directory, bf16_directory, layers, tp, rank,
            capacity, budget, None)
    }

    pub(crate) fn load_with_modules(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors,
        directory: &Path, bf16_directory: Option<&Path>, layers: &[usize], tp: usize, rank: usize,
        capacity: usize, budget: usize, modules: Option<Fp8Modules>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        let mut experts = Self::prepare_with_modules(library, tensors, directory, bf16_directory,
            layers, tp, rank, capacity, budget, modules)?;
        let slicing = tensors.slicing_for_directory(directory, tp);
        experts.load_mount(library, tensors, layers, slicing)?;
        Ok(experts)
    }

    /// Publishes committed layers using the same module, slicing and byte admission as mount loading.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_pushed(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors,
        directory: &Path, layers: &[usize], tp: usize, rank: usize, capacity: usize,
        budget: usize, pushed: Vec<Fp8Layer<'a>>) -> Result<Self> {
        let mut weights = Self::prepare_with_modules(library, tensors, directory, None, layers,
            tp, rank, capacity, budget, None)?;
        let slicing = tensors.slicing_for_directory(directory, tp);
        ensure!(pushed.len() == layers.len(), "pushed FP8-family inventory length differs from admission");
        for (weight, &layer) in pushed.iter().zip(layers) {
            weight.validate_ownership(tensors, layer, tp, rank, slicing)?;
        }
        weights.layers = pushed;
        let expected = Fp8Layer::bytes_for(tensors, tp, rank, slicing)?.checked_mul(layers.len())
            .context("pushed FP8-family resident size overflow")?;
        ensure!(weights.resident_bytes() == expected, "pushed FP8-family arena differs from admitted bytes");
        Ok(weights)
    }

    pub(crate) fn load_mount(&mut self, library: &'a NativeLibrary, tensors: &Fp8ExpertTensors,
        layers: &[usize], slicing: Slicing) -> Result<()> {
        ensure!(self.layers.is_empty(), "native expert weights already published");
        let (tp, rank) = (self.tp, self.rank);
        self.layers = super::load_in_rank_order(layers, rank, |_, layer| {
            let started = std::time::Instant::now();
            let loaded = Fp8Layer::load_with(library, &tensors.for_layer(layer)?, layer, tp, rank, slicing)?;
            tracing::info!(layer, tp, rank, elapsed_ms = started.elapsed().as_millis() as u64,
                "FP8 expert layer resident");
            Ok(loaded)
        })?;
        Ok(())
    }

    fn prepare(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        bf16_directory: Option<&Path>, layers: &[usize], tp: usize, rank: usize, capacity: usize,
        budget: usize) -> Result<Self> {
        Self::prepare_with_modules(library, tensors, directory, bf16_directory, layers, tp, rank,
            capacity, budget, None)
    }

    pub(crate) fn prepare_with_modules(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        bf16_directory: Option<&Path>, layers: &[usize], tp: usize, rank: usize, capacity: usize,
        budget: usize, modules: Option<Fp8Modules>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        validate_layer_ids(layers)?;
        // `tp<n>-w<width>` layouts store each rank's own whole 128-row blocks (see `exact_layout`).
        let slicing = tensors.slicing_for_directory(directory, tp);
        for &layer in layers {
            tensors.validate_layer(layer)?;
        }
        package_metadata(directory, tensors, layers, tp, rank, slicing)?;
        if let Some(directory) = bf16_directory {
            package_metadata(directory, tensors, layers, tp, rank, slicing)?;
        }
        let modules = match modules {
            Some(modules) => modules,
            None => Fp8Modules::load(directory, bf16_directory)?,
        };
        ensure!(modules.directory == directory && modules.bf16_directory.as_deref() == bf16_directory,
            "FP8 warmed package path mismatch");
        let Fp8Modules { module, bf16, .. } = modules;
        let info = module.info().clone();
        let shape = tensors.shape();
        validate_checkpoint_contract(&info, tensors, layers, tp, rank, slicing)?;
        let top = info.capacity_for(capacity)
            .with_context(|| format!("FP8 package has no capacity for {capacity} rows"))?;
        module.scratch_bytes(top)?;
        let bf16_module = bf16.map(|bf16| {
            let top = validate_bf16_package(&info, bf16.info(), capacity)
                .with_context(|| format!("incompatible BF16-input FP8 expert package {}", directory.display()))?;
            bf16.scratch_bytes(top)?;
            Ok::<_, anyhow::Error>(bf16)
        }).transpose()?;
        let layer_bytes = Fp8Layer::bytes_for(tensors, tp, rank, slicing)?;
        let weight_read_bytes = tensors.weight_read_bytes_with(tp, rank, slicing)?;
        ensure!(weight_read_bytes > 0 && weight_read_bytes.checked_mul(shape.experts) == Some(layer_bytes),
            "FP8 layer read extent does not divide into experts");
        let resident_bytes = resident_admission(layer_bytes, layers.len(), budget)?;
        tracing::info!(resident_bytes, budget, bf16_package = ?bf16_directory,
            "FP8 resident expert allocation admitted");
        Ok(Self { layers: Vec::with_capacity(layers.len()),
            bf16_module, module, tp, rank, weight_read_bytes })
    }

    /// Bounded P4 double banks: each checkpoint projection is read once, then
    /// both GPUs take their own row/column window from the same pinned bytes.
    pub fn load_pair(devices: [crate::shared::memory::device::Device<'a>; 2], tensors: &Fp8ExpertTensors,
        directories: [&Path; 2], slicing: [Slicing; 2], layers: &[usize], capacity: usize,
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
                layers, 2, 0, capacity, budgets[0]))?,
            devices[1].own(|| Self::prepare(devices[1].library, tensors, directories[1], None,
                layers, 2, 1, capacity, budgets[1]))?,
        ];
        for &layer in layers {
            for rank in 0..2 {
                devices[rank].run(|| {
                    let mut regions = Vec::with_capacity(6);
                    for projection in Fp8Projection::ALL {
                        let (weights, _) = tensors.slice_bytes_with(projection, 2, rank, slicing[rank])?;
                        for bytes in [weights * tensors.shape().experts,
                            tensors.scale_region_bytes_with(projection, 2, rank, slicing[rank])?] {
                            regions.push(if bytes == 0 { None } else { Some(DeviceAllocation::new(devices[rank].library, bytes)?.into()) });
                        }
                    }
                    owners[rank].layers.push(Fp8Layer { layer,
                        slice: Fp8Layer::loaded_slice(tensors, layer, 2, rank, slicing[rank])?, regions });
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
        for (index, &layer) in layers.iter().enumerate() {
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
                            upload(devices[rank], host.buffer, offsets[slot], regions[2 * slot].as_ref().unwrap().buffer,
                                expert * windows[0].destination_bytes, windows[0], full_bytes, streams[rank].raw)?;
                            if let Some(region) = &regions[2 * slot + 1] {
                                upload(devices[rank], host.buffer, offsets[slot] + sizes[slot].0, region.buffer,
                                    expert * windows[1].destination_bytes, windows[1], full_bytes, streams[rank].raw)?;
                            }
                            if tensors.format() == ExpertFormat::Nvfp4 {
                                let grid_bytes = windows[1].destination_bytes * tensors.shape().experts;
                                for scalar in 0..2 {
                                    let source = offsets[slot] + sizes[slot].0 + sizes[slot].1 - 8 + scalar * 4;
                                    upload(devices[rank], host.buffer, source, regions[2 * slot + 1].as_ref().unwrap().buffer,
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
        self.layers.iter().flat_map(|layer| layer.regions.iter().flatten()).map(|region| region.buffer.bytes).sum()
    }

    pub(crate) fn owner_scratch_bytes(&self, extent: cuteafd_core::OwnerExtent) -> Result<usize> {
        let mut scratch = self.slot_scratch_bytes(extent.capacity as usize)?;
        if let Some(rows) = segment_rows(&self.module, extent) {
            scratch = scratch.max(self.module.segment_scratch_bytes(rows)?);
        }
        Ok(scratch)
    }

    pub fn slot_scratch_bytes(&self, capacity: usize) -> Result<usize> {
        use cuteafd_ffi::fp8_moe::fp8_slot_scratch_bytes;
        let primary = fp8_slot_scratch_bytes(self.module.scratch_layout(), capacity)?;
        Ok(primary.max(self.bf16_module.as_ref().map(|module|
            fp8_slot_scratch_bytes(module.scratch_layout(), capacity)).transpose()?.unwrap_or(0)))
    }

    /// Whether the package takes FP8 K32 wire rows (else BF16 rows).
    pub fn wire_input(&self) -> bool {
        self.module.info().wire_input
    }

    pub fn weight_read_group(&self, rows: usize, bf16: bool)
        -> Result<(usize, cuteafd_transport::worker_selection::ExpertTrafficContract)> {
        let module = if bf16 && self.wire_input() {
            self.bf16_module.as_ref().context("no BF16-input FP8 expert package is loaded")?
        } else { &self.module };
        let (group, contract) = module.weight_read_group(rows)?;
        Ok((group, traffic_contract(contract)))
    }

    /// Stored checkpoint read extent, excluding scratch and allocator alignment.
    pub fn weight_read_bytes(&self) -> usize { self.weight_read_bytes }

    pub fn installed_grouping(&self, loaded: cuteafd_transport::worker_selection::WorkerSelection,
        rank: usize, capacity: usize) -> Result<cuteafd_transport::worker_selection::InstalledExpertGrouping> {
        use cuteafd_transport::worker_selection::{ExpertGroupingSelector, InstalledExpertGrouping};
        let report = InstalledExpertGrouping::new(loaded, rank, capacity, ExpertGroupingSelector::Fp8Nearest,
            installed_variants(&self.module)?)?.with_weight_read_bytes(self.weight_read_bytes)?;
        match &self.bf16_module {
            Some(module) => report.with_bf16_variants(installed_variants(module)?),
            None => Ok(report),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn run_with(&self, module: &Fp8MoeModule, scratch: cuteafd_ffi::CuteafdDeviceBuffer, index: usize,
        rows: usize, input: *mut c_void, ids: *mut c_void, weights: *mut c_void, out: *mut c_void,
        stream: *mut c_void) -> Result<()> {
        let layer = self.layers.get(index).context("FP8 expert layer index out of range")?;
        let [w1, s1, w3, s3, w2, s2] = layer.pointers();
        let pointers: [*mut c_void; FP8_MOE_POINTERS] =
            [input, ids, weights, w1, s1, w3, s3, w2, s2, out, scratch.ptr];
        // SAFETY: the execution owner retains private scratch and the caller retains I/O/stream.
        unsafe { module.launch(&pointers, rows, stream) }
    }
}

fn segment_rows(module: &Fp8MoeModule, extent: cuteafd_core::OwnerExtent) -> Option<u32> {
    let minimum = module.segment_minimum_rows()?;
    (extent.staging_rows >= minimum).then(|| extent.segment_rows()).flatten()
}

/// One serialized execution over resident weights. Local/paged owners keep
/// weights directly; Spark slots share immutable weights and own distinct scratch.
pub(crate) struct Fp8Experts<'a, W = Fp8Weights<'a>> {
    weights: W,
    scratch: Rc<DeviceAllocation<'a>>,
}

impl<'a, W: std::borrow::Borrow<Fp8Weights<'a>>> Fp8Experts<'a, W> {
    pub(crate) fn from_shared(library: &'a NativeLibrary, weights: W, capacity: usize) -> Result<Self> {
        let scratch = Rc::new(Self::allocate_scratch(library, weights.borrow().slot_scratch_bytes(capacity)?)?);
        Self::from_scratch(weights, capacity, scratch)
    }

    pub(crate) fn from_scratch(weights: W, capacity: usize, scratch: Rc<DeviceAllocation<'a>>) -> Result<Self> {
        ensure!(scratch.buffer.bytes >= weights.borrow().slot_scratch_bytes(capacity)?,
            "shared native scratch does not cover this expert package");
        Ok(Self { weights, scratch })
    }
}

impl<'a, W> Fp8Experts<'a, W> {
    fn allocate_scratch(library: &'a NativeLibrary, bytes: usize) -> Result<DeviceAllocation<'a>> {
        DeviceAllocation::named(library, bytes, "experts/workspace/native_scratch")
    }
    pub fn scratch_bytes(&self) -> usize { self.scratch.buffer.bytes }
}

impl<'a> Fp8Experts<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_pushed(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: &[usize], tp: usize, rank: usize, capacity: usize, budget: usize,
        pushed: Vec<Fp8Layer<'a>>) -> Result<Self> {
        let weights = Fp8Weights::from_pushed(library, tensors, directory, layers, tp, rank,
            capacity, budget, pushed)?;
        let scratch = weights.slot_scratch_bytes(capacity)?;
        ensure!(weights.resident_bytes().checked_add(scratch).is_some_and(|bytes| bytes <= budget),
            "pushed FP8 resident weights plus execution scratch exceed budget");
        Self::from_shared(library, weights, capacity)
    }

    pub fn load(library: &'a NativeLibrary, tensors: &Fp8ExpertTensors, directory: &Path,
        layers: &[usize], tp: usize, rank: usize, capacity: usize, budget: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("experts/weights");
        let mut weights = Fp8Weights::prepare(library, tensors, directory, None, layers,
            tp, rank, capacity, budget)?;
        let bytes = weights.slot_scratch_bytes(capacity)?;
        let resident = Fp8Layer::bytes_for(tensors, tp, rank, tensors.slicing_for_directory(directory, tp))?
            .checked_mul(layers.len()).context("FP8 resident size overflow")?;
        ensure!(resident.checked_add(bytes).is_some_and(|total| total <= budget),
            "FP8 resident weights plus execution scratch exceed budget");
        let scratch = Self::allocate_scratch(library, bytes)?;
        weights.load_mount(library, tensors, layers, tensors.slicing_for_directory(directory, tp))?;
        Ok(Self { weights, scratch: Rc::new(scratch) })
    }
}

impl<'a, W: std::borrow::Borrow<Fp8Weights<'a>>> std::ops::Deref for Fp8Experts<'a, W> {
    type Target = Fp8Weights<'a>;
    fn deref(&self) -> &Self::Target { self.weights.borrow() }
}
impl<'a> std::ops::DerefMut for Fp8Experts<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.weights }
}

impl<'a, W: std::borrow::Borrow<Fp8Weights<'a>>> Fp8Experts<'a, W> {
    /// Routed experts of resident layer `index` for `rows` input rows (wire
    /// or BF16, per `wire_input`) into `out` (BF16 `[rows, H]`).
    ///
    /// # Safety
    /// `wire`, `ids` (I32 `[rows, k]`), `weights` (F32 `[rows, k]`) and `out`
    /// are live device buffers of those extents; the stream is drained before
    /// any of them, or this object, is released.
    pub unsafe fn run(&self, index: usize, rows: usize, wire: *mut c_void, ids: *mut c_void, weights: *mut c_void,
        out: *mut c_void, stream: *mut c_void) -> Result<()> {
        self.weights.borrow().run_with(&self.module, self.scratch.buffer, index, rows, wire, ids, weights, out, stream)
    }

    /// # Safety
    /// All row addresses, route planes, resident weights and scratch remain
    /// live on the owner stream until the grouped completion fences.
    pub(crate) unsafe fn run_segments(&self, index: usize, rows: usize, pointers: [*mut c_void; 4],
        stream: *mut c_void) -> Result<()> {
        ensure!(self.scratch.buffer.bytes >= self.module.segment_scratch_bytes(rows.try_into()?)?,
            "segmented launch exceeds admitted scratch");
        let [input, ids, weights, output] = pointers;
        let [w1, s1, w3, s3, w2, s2] = self.layers.get(index).context("segmented layer outside weights")?.pointers();
        // SAFETY: caller retains all operands and the owner retains scratch/weights.
        unsafe { self.module.launch_segments(&[input, ids, weights, w1, s1, w3, s3, w2, s2, output,
            self.scratch.buffer.ptr], rows, stream) }
    }

    /// Execute the same selected arithmetic without its BF16 token combine.
    /// # Safety
    /// All inputs and the exclusive output remain live through the epilogue fence.
    pub(crate) unsafe fn run_reply(&self, reply: &super::reply::Fp8Reply<'_>, index: usize,
        rows: usize, pointers: [cuteafd_ffi::CuteafdDeviceBuffer; 4], bf16: bool, segmented: bool,
        dtype: cuteafd_transport::ExpertV2Dtype, stream: *mut c_void) -> Result<()> {
        let module = if bf16 && self.wire_input() {
            self.bf16_module.as_ref().context("missing BF16-input reply package")?
        } else { &self.module };
        let sources = module.reply_sources(self.scratch.buffer, rows, segmented)?;
        let [input, ids, weights, output] = pointers;
        let [w1, s1, w3, s3, w2, s2] = self.layers.get(index).context("reply layer outside weights")?.pointers();
        // SAFETY: same retained scratch/weights as run; the consuming epilogue
        // immediately follows the core on this owner stream, before scratch reuse.
        unsafe {
            module.launch_core(&[input.ptr, ids.ptr, weights.ptr, w1, s1, w3, s3, w2, s2, output.ptr,
                self.scratch.buffer.ptr], rows, segmented, stream)?;
            reply.launch(dtype, sources, weights, output, rows, segmented, stream)
        }
    }

    /// `run` over BF16 input rows through the BF16-input package.
    ///
    /// # Safety
    /// As `run`, with `rows` BF16 `[rows, H]` input rows.
    pub unsafe fn run_bf16(&self, index: usize, rows: usize, input: *mut c_void, ids: *mut c_void,
        weights: *mut c_void, out: *mut c_void, stream: *mut c_void) -> Result<()> {
        let module = if !self.wire_input() { &self.module } else {
            self.bf16_module.as_ref().context("no BF16-input FP8 expert package is loaded")?
        };
        self.weights.borrow().run_with(module, self.scratch.buffer, index, rows, input, ids, weights, out, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::{resident_admission, validate_bf16_package};
    use clap::Parser;
    use cuteafd_ffi::fp8_moe::{Fp8MoeInfo, Fp8MoeWeights};

    #[test]
    fn native_multilayer_push_admission_uses_each_absolute_layer() -> anyhow::Result<()> {
        use super::Fp8Layer;
        use cuteafd_loader::formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Slicing};
        use cuteafd_loader::plan::testing::{fp8, mxfp4, nvfp4, t, write_snapshot};
        use crate::shared::memory::push_fp8::RankPlan;

        // MiMo Pro's real onboarded geometry, plus every native storage family.
        for format in [ExpertFormat::Mxfp4, ExpertFormat::Fp8Block128,
            ExpertFormat::Nvfp4, ExpertFormat::Bf16] {
            let root = tempfile::tempdir()?;
            let (experts, hidden, intermediate) = if format == ExpertFormat::Mxfp4 {
                (384, 6144, 2048)
            } else { (64, 256, 512) };
            let layers = [1, 2, 5];
            let mut headers = Vec::new();
            for layer in layers {
                for expert in 0..experts {
                    for (projection, n, k) in [("gate", intermediate, hidden),
                        ("up", intermediate, hidden), ("down", hidden, intermediate)] {
                        let name = format!("model.layers.{layer}.mlp.experts.{expert}.{projection}_proj");
                        headers.extend(match format {
                            ExpertFormat::Mxfp4 => mxfp4(&name, n, k),
                            ExpertFormat::Fp8Block128 => fp8(&name, n, k, None),
                            ExpertFormat::Nvfp4 => nvfp4(&name, n, k),
                            ExpertFormat::Bf16 => vec![t(format!("{name}.weight"), "BF16", &[n, k])],
                        });
                    }
                }
            }
            write_snapshot(root.path(), &serde_json::json!({}), &headers, None);
            let tensors = Fp8ExpertTensors::read(root.path(), cuteafd_loader::RoutedExpertShape {
                layers: 6, first_layer: 1, experts, topk: 8, hidden, intermediate,
                draft_stages: 0, draft_experts: 0 })?.for_coordinator().for_layer_metadata(layers[0])?;
            for tp in [1, 2] {
                for rank in 0..tp {
                    let slicing = tensors.slicing_for_directory(std::path::Path::new("tp2"), tp);
                    let selected = layers.map(|layer| (layer, slicing));
                    let plan = RankPlan::from_tensors(&tensors, &selected, tp, rank, usize::MAX)?;
                    for planned in &plan.layers {
                        // Only ownership metadata is needed: no CUDA or checkpoint payload reads.
                        let mut pushed = Fp8Layer { layer: planned.layer, slice: planned.slice.clone(), regions: vec![] };
                        let expected = Fp8Layer::loaded_slice(&tensors, planned.layer, tp, rank, slicing)?;
                        eprintln!("{format:?} tp={tp} rank={rank} slicing={slicing:?} pushed={:?} expected={expected:?}", pushed.slice);
                        pushed.validate_ownership(&tensors, planned.layer, tp, rank, slicing)?;
                        assert_eq!(planned.planes.iter().map(|plane| plane.len()).sum::<usize>(),
                            Fp8Layer::bytes_for(&tensors, tp, rank, slicing)?);
                        if planned.layer != layers[0] {
                            pushed.slice.unit_map = Some(tensors.unit_map(tp, rank, slicing)?.fingerprint());
                            let error = pushed.validate_ownership(&tensors, planned.layer, tp, rank, slicing).unwrap_err();
                            let message = error.to_string();
                            for field in ["pushed=", "expected=", "slicing=", "tp=", "rank="] {
                                assert!(message.contains(field), "{message}");
                            }
                        }
                    }
                    assert!(matches!(slicing, Slicing::Blocks(128)));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn native_bf16_load_contract_rejects_wrong_storage_rounding_limit_and_input() {
        use cuteafd_ffi::fp8_moe::{Fp8MoeActivation, Fp8MoeProjection};
        use cuteafd_loader::plan::testing::{t,write_snapshot};
        use cuteafd_loader::formats::fp8_experts::{Fp8ExpertTensors,Slicing};
        let dir = tempfile::tempdir().unwrap();
        let shape = cuteafd_loader::RoutedExpertShape { layers:1,first_layer:0,experts:2,topk:1,
            hidden:128,intermediate:1536,draft_stages:0,draft_experts:0 };
        let fixture = [("gate",1536,128),("up",1536,128),("down",128,1536)].map(|(p,n,k)|
            t(format!("model.layers.0.moe.{p}_proj.weight"),"BF16",&[2,n,k]));
        // A real declaration supplies the activation independently of this reduced test geometry.
        let source: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/plan/families/step5/fixtures/official-fp8.json"))).unwrap();
        write_snapshot(dir.path(),&source["config"],&fixture,None);
        let tensors = Fp8ExpertTensors::read(dir.path(),shape).unwrap();
        let info = Fp8MoeInfo {hidden:128,slice:256,experts:2,topk:1,intermediate:1536,tp:6,
            wire_input:false,swiglu_limit:7.0,activation:Fp8MoeActivation::SiluPostClampFp32Product,
            projection:Fp8MoeProjection::Bf16,capacities:vec![16],weights:Fp8MoeWeights::Bf16};
        super::validate_checkpoint_contract(&info,&tensors,&[0],6,0,Slicing::Padded).unwrap();
        for mutation in 0..4 {
            let mut wrong = info.clone();
            match mutation {
                0 => wrong.weights = Fp8MoeWeights::Fp8,
                1 => wrong.activation = Fp8MoeActivation::SiluBf16Product,
                2 => wrong.swiglu_limit = 10.0,
                _ => wrong.wire_input = true,
            }
            assert!(super::validate_checkpoint_contract(&wrong,&tensors,&[0],6,0,Slicing::Padded).is_err());
        }
    }

    #[test]
    fn native_bf16_package_locator_is_not_the_fp8_bf16_input_sibling() {
        use cuteafd_loader::formats::fp8_experts::ExpertFormat;
        let root = std::path::Path::new("/exports");
        assert_eq!(root.join(ExpertFormat::Bf16.package_name("step5")),
            std::path::Path::new("/exports/fp8-step5-bf16-weights"));
        assert_eq!(cuteafd_core::ExpertGeometry::STEP5.family(), Some("step5"));
        assert_eq!(super::bf16_sibling(std::path::Path::new("/exports/fp8-step5/tp6")).unwrap(),
            std::path::Path::new("/exports/fp8-step5-bf16/tp6"));
    }

    #[test]
    fn step5_nvfp4_uses_the_single_native_w4a4_package() {
        use std::path::Path;
        use cuteafd_loader::formats::fp8_experts::ExpertFormat;
        let library = Path::new("/exports/libcuteafd_native.so");
        for world in [4, 6] {
            assert_eq!(super::package_directory_for(library, world, ExpertFormat::Nvfp4, Some("step5")),
                Path::new("/exports/fp8/fp8-step5-nvfp4a4").join(format!("tp{world}")));
            for family in ["glm", "glmf", "qwen4"] {
                assert_eq!(super::package_directory_for(library, world, ExpertFormat::Nvfp4, Some(family)),
                    Path::new("/exports/fp8").join(format!("fp8-{family}-nvfp4a4/tp{world}")));
            }
        }
    }

    #[test]
    fn exact_layer_ids_are_ordered_unique_and_may_be_sparse_or_empty() {
        for layers in [&[][..], &[13][..], &[13, 15, 20][..]] {
            super::validate_layer_ids(layers).unwrap();
        }
        for layers in [&[13, 13][..], &[15, 13][..], &[13, 20, 15][..]] {
            assert!(super::validate_layer_ids(layers).is_err());
        }
    }

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
        use super::{exact_layout, ExpertFormat, Fp8Weights, Fp8ExpertTensors, Fp8Layer, Fp8Projection, NativeLibrary};
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
        let owners = Fp8Weights::load_pair(devices, &tensors, [&layouts[0].0, &layouts[1].0],
            layouts.each_ref().map(|layout| layout.1), &[layer], 1, [left?, right?])?;
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
                    devices[rank].run(|| library.copy_d2h(&mut actual, owners[rank].layers[0].regions[region].as_ref().unwrap().buffer))?;
                    ensure!(actual == expected, "TP2 paired upload differs: rank {rank} layer {layer} {projection:?} region {region}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn resident_admission_contains_only_weights() {
        let layer = 1_415_589_888;
        assert_eq!(resident_admission(layer, 48, layer * 48).unwrap(), layer * 48);
        assert!(resident_admission(layer, 48, layer * 48 - 1).is_err());
        assert_eq!(resident_admission(layer, 0, 0).unwrap(), 0);
        assert!(resident_admission(usize::MAX, 2, usize::MAX).is_err());
        assert_eq!(resident_admission(usize::MAX, 1, usize::MAX).unwrap(), usize::MAX);
    }

    #[test]
    fn bf16_sibling_must_share_weight_layout_arithmetic_and_capacity() {
        let primary = Fp8MoeInfo { hidden: 6144, slice: 512, experts: 256, topk: 8, intermediate: 2048, tp: 4,
            wire_input: true, swiglu_limit: 10.0, activation: cuteafd_ffi::fp8_moe::Fp8MoeActivation::SiluBf16Product, projection: cuteafd_ffi::fp8_moe::Fp8MoeProjection::Bf16, capacities: vec![64, 4096], weights: Fp8MoeWeights::Fp8 };
        let mut bf16 = primary.clone();
        bf16.wire_input = false;
        // A sibling may compile different capacities; its own scratch uses its own top.
        bf16.capacities = vec![128, 8192];
        assert_eq!(validate_bf16_package(&primary, &bf16, 64).unwrap(), 128);
        assert_eq!(validate_bf16_package(&primary, &bf16, 4096).unwrap(), 8192);
        assert!(validate_bf16_package(&primary, &bf16, 8193).is_err());
        for variant in 0..6 {
            let mut incompatible = bf16.clone();
            match variant {
                0 => incompatible.wire_input = true,
                1 => incompatible.slice = 640,
                2 => incompatible.swiglu_limit = f32::INFINITY,
                3 => incompatible.tp = 2,
                4 => incompatible.weights = Fp8MoeWeights::Mxfp4,
                _ => incompatible.activation = cuteafd_ffi::fp8_moe::Fp8MoeActivation::SiluPostClampFp32Product,
            }
            assert!(validate_bf16_package(&primary, &incompatible, 64).is_err(), "{incompatible:?}");
        }
    }
}
