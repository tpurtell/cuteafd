//! Bound EXL3 launch tables. Allocations and name resolution happen at setup;
//! each launch only substitutes live inputs/row bounds and enqueues GPU work.
use super::Exl3Weights;
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41Exl3Kernel, V41Exl3Layout, V41Exl3Routes};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ffi::c_void, path::{Path, PathBuf}, rc::Rc};

/// Capacity selects the compiled reduction geometry, never the live row count.
/// GLM Flash's m1/m80 use K64; m16 uses K128 and adds serial/verify drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Exl3RowPolicy {
    Nearest,
    GlmFlashK64,
}

impl Exl3RowPolicy {
    pub(crate) fn for_geometry(geometry: cuteafd_core::ExpertGeometry) -> Self {
        if geometry.same_shape(&cuteafd_core::ExpertGeometry::GLM5_FLASH) {
            Self::GlmFlashK64
        } else {
            Self::Nearest
        }
    }

    pub(crate) fn active() -> Self { Self::for_geometry(cuteafd_core::expert_geometry()) }

    pub(crate) fn required_capacity(self, live_rows: usize) -> usize {
        if self == Self::GlmFlashK64 && (2..=16).contains(&live_rows) { 80 } else { live_rows }
    }

    /// Used by artifact validation and workspace admission before weight reads,
    /// including a narrow declared maximum that otherwise would only load m16.
    pub(crate) fn capacities(self, max_live_rows: usize) -> Result<Vec<u32>> {
        const CAPACITIES: [u32; 6] = [1, 16, 80, 256, 1024, 4096];
        ensure!((1..=4096).contains(&max_live_rows), "EXL3 live capacity must be 1..4096");
        let required = max_live_rows.max(self.required_capacity(max_live_rows));
        let maximum = CAPACITIES.into_iter().find(|&c| c as usize >= required).unwrap();
        Ok(CAPACITIES.into_iter().filter(|&c| c <= maximum).collect())
    }
}

/// Spark decode schedule of the EXL3 exports (`--exl3-schedule`). `Gb10`
/// runs the `m<capacity>-gb10` siblings of the GLM 5.3 Flash TP4 decode
/// capacities (m1 for one row, m80 for 2-80): the same products and sums as
/// the default exports, so the same bits, with the weight words staged L2
/// evict-first and, at m80, 64x128 tiles at two CTAs per SM. Every other
/// capacity runs its default export.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Exl3Schedule {
    #[default]
    Default,
    Gb10,
}

impl Exl3Schedule {
    const GB10_CAPACITIES: [u32; 2] = [1, 80];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Gb10 => "gb10",
        }
    }

    fn scheduled(self, capacity: u32) -> bool {
        self == Self::Gb10 && Self::GB10_CAPACITIES.contains(&capacity)
    }

    /// The export directory a capacity runs under this schedule.
    pub(crate) fn directory(self, root: &Path, capacity: u32) -> PathBuf {
        if self.scheduled(capacity) {
            root.join(format!("m{capacity}-{}", self.name()))
        } else {
            root.join(format!("m{capacity}"))
        }
    }

    /// Before any weight read: the GB10 exports exist for GLM 5.3 Flash only.
    pub(crate) fn validate(self, row_policy: Exl3RowPolicy) -> Result<()> {
        ensure!(self == Self::Default || row_policy == Exl3RowPolicy::GlmFlashK64,
            "--exl3-schedule {} serves the GLM 5.3 Flash EXL3 experts only", self.name());
        Ok(())
    }

    /// A scheduled capacity's export must record its decode schedule and a
    /// default one none, so a mislabelled directory fails at start-up.
    pub(crate) fn check_export(self, root: &Path, capacity: u32) -> Result<()> {
        #[derive(Deserialize)]
        struct Record {
            #[serde(default)]
            decode_schedule: Option<String>,
        }
        let directory = self.directory(root, capacity);
        let meta: Record = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))
            .with_context(|| format!("EXL3 {} schedule export {}", self.name(), directory.display()))?)?;
        ensure!(meta.decode_schedule.is_some() == self.scheduled(capacity),
            "EXL3 export {} does not carry the {} decode schedule", directory.display(),
            if self.scheduled(capacity) { self.name() } else { "default" });
        Ok(())
    }
}

#[derive(Deserialize)]
struct Buffer {
    bytes: usize,
    dtype: String,
    allocation: String,
    zero_on_create: bool,
}
#[derive(Deserialize)]
struct Object {
    label: String,
    pointer_slots: Vec<String>,
    scalar_slots: Vec<String>,
}
#[derive(Deserialize)]
struct Asset {
    file: String,
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
struct RouteManifest {
    manifest: String,
    sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    schema: String,
    output_dtype: String,
    sparkinfer_revision: String,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    capacity: usize,
    top_k: usize,
    bits: Vec<usize>,
    /// `null` for an unclamped SwiGLU (GLM).
    swiglu_limit: Option<f32>,
    direct: bool,
    sms: usize,
    blocks_per_sm: usize,
    buffers: BTreeMap<String, Buffer>,
    objects: Vec<Object>,
    trellis_lut: Asset,
    route_preparation: Option<RouteManifest>,
    paired_boundary: Option<String>,
    descriptor_rows: Option<usize>,
    native_info_version: Option<u32>,
    /// `e4m3_k32`: the core reads E4M3 + UE8M0 K32 wire rows itself (no
    /// BF16 decode pass); absent or `bf16`: BF16 rows.
    #[serde(default)]
    input_format: Option<String>,
}
impl Manifest {
    fn wire_input(&self) -> Result<bool> {
        match self.input_format.as_deref() {
            None | Some("bf16") => Ok(false),
            Some("e4m3_k32") => Ok(true),
            Some(other) => anyhow::bail!("unknown EXL3 input format {other}"),
        }
    }
    fn native_layout(&self) -> Result<V41Exl3Layout> {
        match self.paired_boundary.as_deref() {
            None => {
                ensure!(self.descriptor_rows.is_none_or(|rows| rows == 3)
                    && self.native_info_version.is_none_or(|version| version == 2),
                    "disjoint EXL3 descriptor/version mismatch");
                Ok(V41Exl3Layout::Disjoint)
            }
            Some(boundary @ ("first" | "last")) => {
                ensure!(self.intermediate == 640 && self.bits.len() == 2 && self.top_k == 6
                    && self.descriptor_rows == Some(4) && self.native_info_version == Some(3),
                    "paired EXL3 manifest contract mismatch");
                Ok(if boundary == "first" { V41Exl3Layout::PairedFirst } else { V41Exl3Layout::PairedLast })
            }
            Some(_) => anyhow::bail!("unknown paired EXL3 boundary"),
        }
    }
    fn workspace_bytes(&self, format: Exl3InputFormat) -> Result<usize> {
        let mut bytes = self.trellis_lut.bytes.max(16);
        for (name, buffer) in &self.buffers {
            if name == &buffer.allocation {
                bytes = bytes
                    .checked_add(buffer.bytes.max(16))
                    .context("EXL3 workspace budget overflow")?;
            }
        }
        if format == Exl3InputFormat::Fp8K32 && !self.wire_input()? {
            bytes = bytes
                .checked_add(
                    self.capacity
                        .checked_mul(self.hidden * 2)
                        .context("EXL3 wire workspace overflow")?,
                )
                .context("EXL3 workspace budget overflow")?;
        }
        Ok(bytes)
    }
}

/// Mutually exclusive capacity specializations in one lane may reuse data
/// scratch. Persistent synchronization state remains private to each kernel.
pub(crate) struct Exl3Workspace<'a> {
    allocations: BTreeMap<String, DeviceAllocation<'a>>,
    dtypes: BTreeMap<String, String>,
}
impl<'a> Exl3Workspace<'a> {
    pub(crate) fn bytes(&self) -> usize {
        self.allocations.values().map(|allocation| allocation.buffer.bytes).sum()
    }
    fn layout(directories: &[PathBuf]) -> Result<BTreeMap<String, (usize, String)>> {
        ensure!(!directories.is_empty(), "empty EXL3 capacity workspace");
        let mut layout: BTreeMap<String, (usize, String)> = BTreeMap::new();
        for directory in directories {
            let meta: Manifest = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
            for (name, spec) in meta.buffers {
                if name != spec.allocation || spec.zero_on_create { continue; }
                let entry = layout.entry(name).or_insert((0, spec.dtype.clone()));
                ensure!(entry.1 == spec.dtype, "EXL3 shared scratch dtype mismatch");
                entry.0 = entry.0.max(spec.bytes.max(16));
            }
        }
        Ok(layout)
    }
    /// Total allocation payload for all specializations sharing one lane arena.
    pub(crate) fn plan(directories: &[PathBuf], format: Exl3InputFormat) -> Result<usize> {
        let manifests = directories.iter().map(|directory| -> Result<serde_json::Value> {
            Ok(serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?)
        }).collect::<Result<Vec<_>>>()?;
        Ok(usize::try_from(cuteafd_loader::serving_capacity::exl3_workspace_bytes(&manifests,
            format == Exl3InputFormat::Fp8K32)?)?)
    }

    pub(crate) fn new(library: &'a NativeLibrary, directories: &[PathBuf]) -> Result<Rc<Self>> {
        let mut allocations = BTreeMap::new();
        let mut dtypes = BTreeMap::new();
        for (name, (bytes, dtype)) in Self::layout(directories)? {
            allocations.insert(name.clone(), DeviceAllocation::new(library, bytes)?);
            dtypes.insert(name, dtype);
        }
        Ok(Rc::new(Self { allocations, dtypes }))
    }
}

struct Table {
    pointers: Vec<*mut c_void>,
    scalars: Vec<i32>,
    live: usize,
    inputs: Vec<(usize, usize)>,
}
impl Table {
    fn build(
        object: &Object,
        pointers: &BTreeMap<String, CuteafdDeviceBuffer>,
        values: &BTreeMap<String, i32>,
    ) -> Result<Self> {
        let mut inputs = Vec::new();
        let mut table = Vec::new();
        for (index, name) in object.pointer_slots.iter().enumerate() {
            let dynamic = match name.as_str() {
                "rotation_input_ptr" => Some(0),
                "raw_topk_ids" | "route_expert_ids_ptr" => Some(1),
                "topk_weights_ptr" => Some(2),
                _ => None,
            };
            if let Some(slot) = dynamic {
                inputs.push((index, slot));
                table.push(std::ptr::null_mut());
            } else {
                table.push(
                    pointers
                        .get(name)
                        .with_context(|| format!("missing EXL3 pointer {name}"))?
                        .ptr,
                );
            }
        }
        let scalars = object
            .scalar_slots
            .iter()
            .map(|name| {
                values
                    .get(name)
                    .copied()
                    .with_context(|| format!("missing EXL3 scalar {name}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let live = object
            .scalar_slots
            .iter()
            .position(|s| s == "active_m")
            .context("missing EXL3 live bound")?;
        Ok(Self {
            pointers: table,
            scalars,
            live,
            inputs,
        })
    }
    fn bind(&mut self, inputs: &[CuteafdDeviceBuffer; 3], rows: usize) {
        for &(index, slot) in &self.inputs {
            self.pointers[index] = inputs[slot].ptr;
        }
        self.scalars[self.live] = rows as i32;
    }
}

struct LayerBinding {
    core: Table,
    sum: Table,
    expert_map: CuteafdDeviceBuffer,
    ownership: Option<CuteafdDeviceBuffer>,
    output_slot: usize,
}

pub(crate) struct Exl3Execution<'a> {
    kernel: V41Exl3Kernel,
    routes: Option<V41Exl3Routes>,
    wire: Option<(cuteafd_ffi::V41Exl3Wire<'a>, DeviceAllocation<'a>)>,
    /// Inputs arrive as wire rows (decoded by `wire`, or read by the core).
    wire_rows: bool,
    _storage: Vec<DeviceAllocation<'a>>,
    _shared_workspace: Option<Rc<Exl3Workspace<'a>>>,
    // Keep every prebound pointer alive through the last graph replay.
    _weights: Rc<Vec<Exl3Weights<'a>>>,
    layers: Vec<LayerBinding>,
    route_pointers: [*mut c_void; 7],
    route_bytes: [u64; 7],
    output: CuteafdDeviceBuffer,
    output_element_bytes: usize,
    device: i32,
    hidden: usize,
    capacity: usize,
    topk: usize,
    library: &'a NativeLibrary,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Exl3InputFormat {
    Bf16,
    Fp8K32,
}

impl<'a> Exl3Execution<'a> {
    pub(crate) fn artifact_layout(directory: &Path) -> Result<V41Exl3Layout> {
        let meta: Manifest = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        meta.native_layout()
    }

    pub(crate) fn ownership_bytes(directory: &Path) -> Result<usize> {
        let meta: Manifest = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        Ok(if meta.native_layout()? == V41Exl3Layout::Disjoint { 0 } else { meta.experts * meta.bits.len() * 4 })
    }

    /// Device allocation payload per lane, including aliases and optional wire
    /// reconstruction. Weight storage and CUDA module/driver reserve are separate.
    pub(crate) fn plan(directory: &Path, format: Exl3InputFormat) -> Result<usize> {
        let meta: Manifest =
            serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        meta.workspace_bytes(format)
    }

    /// Trusted build artifacts only. Each owner is exclusive to one decode lane;
    /// callers drain work and destroy graphs before releasing it or its weights.
    pub(crate) unsafe fn new(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
    ) -> Result<Self> {
        Self::with_input_format(library, weights, directory, Exl3InputFormat::Bf16)
    }

    pub(crate) unsafe fn with_input_format(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        format: Exl3InputFormat,
    ) -> Result<Self> {
        Self::with_shared_workspace(library, weights, directory, format, None)
    }

    /// # Safety
    /// Executions sharing this workspace must never overlap, including graph
    /// replays. Use separate arenas for independently scheduled lanes/devices.
    pub(crate) unsafe fn with_shared_workspace(
        library: &'a NativeLibrary,
        weights: Rc<Vec<Exl3Weights<'a>>>,
        directory: &Path,
        format: Exl3InputFormat,
        shared: Option<Rc<Exl3Workspace<'a>>>,
    ) -> Result<Self> {
        let meta: Manifest =
            serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        let core_reads_wire = meta.wire_input()?;
        ensure!(!core_reads_wire || format == Exl3InputFormat::Fp8K32,
            "EXL3 wire-input export requires FP8 K32 expert inputs");
        let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../third_party/sparkinfer.lock.json"
        )))?;
        ensure!(
            meta.schema == "cuteafd.v41-exl3-aot.v1"
                && meta.sparkinfer_revision == lock["revision"].as_str().unwrap_or(""),
            "EXL3 export/source mismatch"
        );
        ensure!(
            !weights.is_empty(),
            "EXL3 execution requires resident layers"
        );
        let expected_layout = meta.native_layout()?;
        let device = library.cuda_get_device()?;
        for weight in weights.iter() {
            let resident_layout = match weight.layout.layout {
                cuteafd_loader::V41Exl3Partition::Disjoint => V41Exl3Layout::Disjoint,
                cuteafd_loader::V41Exl3Partition::PairedTp4 => {
                    ensure!(weight.layout.world == 4 && weight.layout.rank < 4,
                        "paired EXL3 resident rank/world mismatch");
                    if weight.layout.rank % 2 == 0 { V41Exl3Layout::PairedLast } else { V41Exl3Layout::PairedFirst }
                }
            };
            ensure!(resident_layout == expected_layout, "EXL3 manifest/resident layout mismatch");
            let descriptor_rows = if resident_layout == V41Exl3Layout::Disjoint { 3 } else { 4 };
            ensure!(weight.buffer("descriptor_map")?.bytes == descriptor_rows * meta.experts * meta.bits.len() * 4,
                "EXL3 resident descriptor extent mismatch");
            ensure!(
                weight.layout.world == weights[0].layout.world
                    && weight.layout.rank == weights[0].layout.rank
                    && weight.buffer("global_to_combined")?.device_id == device,
                "EXL3 resident layers must share one device and TP rank"
            );
            ensure!(
                meta.hidden == cuteafd_core::expert_geometry().hidden as usize
                    && meta.intermediate == weight.layout.intermediate
                    && meta.experts == weight.layout.experts
                    && meta.bits == weight.layout.tiers
                    && meta.swiglu_limit == cuteafd_core::expert_geometry().swiglu_limit(),
                "EXL3 export/residency geometry mismatch"
            );
            // V4.1 dSpark drafts route top-3; DeepSeek V4 stages and every
            // backbone follow the model.
            let expected_topk =
                if matches!(weight.layout.layer, cuteafd_loader::V41Exl3Layer::Dspark(_))
                    && cuteafd_core::expert_geometry().family() == Some("v41") {
                    3
                } else {
                    cuteafd_core::expert_geometry().topk as usize
                };
            ensure!(meta.top_k == expected_topk, "EXL3 expert top-k mismatch");
        }
        let kernel = V41Exl3Kernel::load_with_layout(directory.join("libcuteafd_exl3.so"), expected_layout)?;
        let info = kernel.info();
        ensure!(
            info.hidden == meta.hidden
                && info.intermediate == meta.intermediate
                && info.experts == meta.experts
                && info.capacity == meta.capacity
                && info.topk == meta.top_k
                && info.bits[..info.tier_count]
                    .iter()
                    .map(|b| *b as usize)
                    .eq(meta.bits.iter().copied()),
            "EXL3 binary/manifest mismatch"
        );
        let output_dtype = match info.output_element_bytes {
            2 => ("bf16", "torch.bfloat16"),
            4 => ("fp32", "torch.float32"),
            _ => anyhow::bail!("unsupported EXL3 output precision"),
        };
        let output_spec = meta.buffers.get("output").context("missing EXL3 output")?;
        ensure!(
            meta.output_dtype == output_dtype.0
                && output_spec.dtype == output_dtype.1
                && output_spec.bytes == meta.capacity * meta.hidden * info.output_element_bytes,
            "EXL3 output precision/size mismatch"
        );
        let mut storage = Vec::new();
        let mut pointers = BTreeMap::new();
        // Never resize to the current device's SM count: compiled grid barriers
        // retain export-time offsets, even when the bridge caps the launch grid.
        for (name, spec) in &meta.buffers {
            if name != &spec.allocation {
                continue;
            }
            if let Some(arena) = shared.as_ref().filter(|_| !spec.zero_on_create) {
                let buffer = arena.allocations.get(name).context("missing shared EXL3 scratch")?.buffer;
                ensure!(buffer.device_id == device && buffer.bytes >= spec.bytes
                    && arena.dtypes.get(name) == Some(&spec.dtype), "invalid shared EXL3 scratch");
                pointers.insert(name.clone(), buffer);
                continue;
            }
            let allocation = DeviceAllocation::new(library, spec.bytes.max(16))?;
            if spec.zero_on_create {
                library.copy_h2d(allocation.buffer, &vec![0; spec.bytes])?;
            }
            pointers.insert(name.clone(), allocation.buffer);
            storage.push(allocation);
        }
        for (name, spec) in &meta.buffers {
            let mut buffer = *pointers
                .get(&spec.allocation)
                .context("missing EXL3 workspace allocation")?;
            ensure!(
                spec.bytes <= buffer.bytes,
                "EXL3 workspace alias exceeds owner"
            );
            buffer.bytes = spec.bytes;
            pointers.insert(name.clone(), buffer);
        }
        let lut = std::fs::read(directory.join(&meta.trellis_lut.file))?;
        ensure!(
            lut.len() == meta.trellis_lut.bytes
                && format!("{:x}", Sha256::digest(&lut)) == meta.trellis_lut.sha256,
            "EXL3 LUT artifact mismatch"
        );
        let lut_buffer = DeviceAllocation::new(library, lut.len())?;
        library.copy_h2d(lut_buffer.buffer, &lut)?;
        pointers.insert("trellis_lut_ptr".into(), lut_buffer.buffer);
        storage.push(lut_buffer);
        let mut layers = Vec::with_capacity(weights.len());
        for weight in weights.iter() {
            let mut pointers = pointers.clone();
            for (target, source) in [
                ("descriptor_map_ptr", "descriptor_map"),
                ("global_to_combined_ptr", "global_to_combined"),
                ("expert_map_ptr", "global_to_combined"),
                ("intermediate_rotations_ptr", "intermediate_rotations"),
                ("gate_suh_ptr", "gate_suh"),
                ("up_suh_ptr", "up_suh"),
                ("svh_ptr", "down_svh"),
            ] {
                pointers.insert(target.into(), weight.buffer(source)?);
            }
            for (target, source) in [("fc2_ptr", "fc2"), ("output_ptr", "output")] {
                pointers.insert(
                    target.into(),
                    *pointers
                        .get(source)
                        .context("missing EXL3 output workspace")?,
                );
            }
            let mut values = BTreeMap::from([
                ("active_m".into(), 1),
                (
                    "grid_x".into(),
                    // The native bridge caps this request to the smaller of
                    // export/current SM counts; larger devices cannot expand it.
                    i32::try_from(meta.sms.checked_mul(meta.blocks_per_sm)
                        .context("EXL3 grid capacity overflow")?)?,
                ),
                ("route_num_experts".into(), meta.experts as i32),
                (
                    "weight_num_experts".into(),
                    (meta.experts * meta.bits.len()) as i32,
                ),
            ]);
            for (tier, counts) in weight.layout.projection_counts.iter().enumerate() {
                for (field, source) in [
                    ("w13", format!("tier{tier}_w13")),
                    ("w2", format!("tier{tier}_w2")),
                    ("w13_scales", "dummy_scales".into()),
                    ("w2_scales", "dummy_scales".into()),
                    ("w13_global", "unit_scales".into()),
                    ("w2_global", "unit_scales".into()),
                ] {
                    pointers.insert(format!("t{tier}_{field}_ptr"), weight.buffer(&source)?);
                }
                for (name, value) in [
                    ("num_experts", meta.experts),
                    ("fc2_experts", counts[2]),
                    ("gate_experts", counts[0]),
                    ("up_experts", counts[1]),
                ] {
                    values.insert(format!("tier{tier}_{name}"), value as i32);
                }
            }
            ensure!(
                meta.objects.len() == 2
                    && meta.objects[0].label == "v41_exl3_core"
                    && meta.objects[1].label == "v41_exl3_sum",
                "EXL3 object ordering mismatch"
            );
            let core = Table::build(&meta.objects[0], &pointers, &values)?;
            let sum = Table::build(&meta.objects[1], &pointers, &values)?;
            ensure!(
                core.pointers.len() == info.core_pointers
                    && core.scalars.len() == info.core_scalars
                    && sum.pointers.len() == info.sum_pointers
                    && sum.scalars.len() == info.sum_scalars,
                "EXL3 native argument lengths disagree"
            );
            let ownership = if expected_layout == V41Exl3Layout::Disjoint { None } else {
                let descriptor = weight.buffer("descriptor_map")?;
                let bytes = meta.experts * meta.bits.len() * 4;
                Some(CuteafdDeviceBuffer {
                    ptr: descriptor.ptr.cast::<u8>().add(3 * bytes).cast(),
                    bytes, ..descriptor
                })
            };
            layers.push(LayerBinding {
                ownership,
                core,
                sum,
                expert_map: weight.buffer("global_to_combined")?,
                output_slot: meta.objects[1]
                    .pointer_slots
                    .iter()
                    .position(|name| name == "output_ptr")
                    .context("missing EXL3 output pointer slot")?,
            });
        }
        let mut route_pointers = [std::ptr::null_mut(); 7];
        let mut route_bytes = [0; 7];
        let routes = if meta.direct {
            None
        } else {
            let route = meta
                .route_preparation
                .context("missing EXL3 packed route export")?;
            let path = directory.join(route.manifest);
            let bytes = std::fs::read(&path)?;
            ensure!(
                format!("{:x}", Sha256::digest(&bytes)) == route.sha256,
                "EXL3 route export hash mismatch"
            );
            for (index, name) in [
                "global_to_combined_ptr",
                "packed_route_indices",
                "block_expert_ids",
                "packed_route_count",
                "expert_offsets",
                "expert_counts",
            ]
            .iter()
            .enumerate()
            {
                let buffer = if index == 0 {
                    &layers[0].expert_map
                } else {
                    pointers
                        .get(*name)
                        .context("missing EXL3 packed route buffer")?
                };
                route_pointers[index + 1] = buffer.ptr;
                route_bytes[index + 1] = buffer.bytes as u64;
            }
            Some(V41Exl3Routes::load(
                path.parent().unwrap().join("libv41_exl3_routes.so"),
            )?)
        };
        let wire = if format == Exl3InputFormat::Fp8K32 && !core_reads_wire {
            // The router's hidden-wide wire row is replicated across TP ranks;
            // only intermediate expert weights are sliced. Local RTX TP1/TP2
            // therefore use the same decoder as Spark TP4.
            Some((
                library.v41_exl3_wire()?,
                DeviceAllocation::new(library, meta.capacity * meta.hidden * 2)?,
            ))
        } else {
            None
        };
        Ok(Self {
            kernel,
            routes,
            wire,
            wire_rows: format == Exl3InputFormat::Fp8K32,
            _storage: storage,
            _shared_workspace: shared,
            _weights: weights,
            layers,
            route_pointers,
            route_bytes,
            output_element_bytes: info.output_element_bytes,
            output: *pointers.get("output").context("missing EXL3 output")?,
            device,
            hidden: meta.hidden,
            capacity: meta.capacity,
            topk: meta.top_k,
            library,
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
    pub(crate) fn output_element_bytes(&self) -> usize {
        self.output_element_bytes
    }
    /// Whether this execution reads its FP8 wire rows only through its own
    /// decode pass (one read of each row), rather than in the core.
    pub(crate) fn decodes_wire_rows(&self) -> bool {
        self.wire.is_some()
    }

    /// Privately allocated payload; shared arenas are counted once by their planner.
    pub(crate) fn workspace_bytes(&self) -> usize {
        self._storage.iter().map(|b| b.buffer.bytes).sum::<usize>()
            + self.wire.as_ref().map_or(0, |(_, b)| b.buffer.bytes)
    }

    /// # Safety
    /// Same stream, graph and buffer contract as `launch_layer`.
    pub(crate) unsafe fn launch(
        &mut self,
        inputs: [CuteafdDeviceBuffer; 3],
        rows: usize,
        stream: *mut c_void,
    ) -> Result<CuteafdDeviceBuffer> {
        self.launch_layer(0, inputs, rows, stream)
    }

    /// # Safety
    /// Inputs are BF16[rows,H] or FP8 wire[rows,H+H/32] as selected at setup,
    /// int32[rows,topk], FP32[rows,topk], contiguous
    /// on this owner device and live through completion. No overlapping input /
    /// workspace storage or concurrent use of this lane, including graph replay.
    pub(crate) unsafe fn launch_layer(
        &mut self,
        layer: usize,
        inputs: [CuteafdDeviceBuffer; 3],
        rows: usize,
        stream: *mut c_void,
    ) -> Result<CuteafdDeviceBuffer> {
        self.launch_layer_into(layer, inputs, rows, stream, self.output)
    }

    /// # Safety
    /// Same contract as `launch_layer`; output is an exclusive, aligned GPU or
    /// mapped-host allocation on this device, live through execution/graph replay.
    pub(crate) unsafe fn launch_layer_into(
        &mut self,
        layer: usize,
        inputs: [CuteafdDeviceBuffer; 3],
        rows: usize,
        stream: *mut c_void,
        output: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        self.launch_layer_with_ownership(layer, inputs, rows, stream, output, None)
    }

    /// # Safety
    /// In addition to `launch_layer_into`, ownership is a device int32 row
    /// validated by the batch decoder, with zero inactive/padded entries. Its
    /// producer precedes this call on `stream`, and it remains live until copy
    /// completion. No other execution may use or mutate this layer's shared
    /// descriptor until the launch (or its captured graph replay) completes.
    pub(crate) unsafe fn launch_paired_layer_into(
        &mut self, layer: usize, inputs: [CuteafdDeviceBuffer; 3], rows: usize,
        stream: *mut c_void, output: Option<CuteafdDeviceBuffer>, ownership: CuteafdDeviceBuffer,
    ) -> Result<CuteafdDeviceBuffer> {
        self.launch_layer_with_ownership(layer, inputs, rows, stream, output.unwrap_or(self.output), Some(ownership))
    }

    unsafe fn launch_layer_with_ownership(
        &mut self,
        layer: usize,
        mut inputs: [CuteafdDeviceBuffer; 3],
        rows: usize,
        stream: *mut c_void,
        mut output: CuteafdDeviceBuffer,
        ownership: Option<CuteafdDeviceBuffer>,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "EXL3 live rows exceed capacity"
        );
        ensure!(
            self.library.cuda_get_device()? == self.device,
            "EXL3 execution on wrong device"
        );
        for (buffer, bytes) in inputs.iter().zip([
            rows * if self.wire_rows { self.hidden + self.hidden / 32 } else { self.hidden * 2 },
            rows * self.topk * 4,
            rows * self.topk * 4,
        ]) {
            ensure!(
                !buffer.ptr.is_null()
                    && buffer.ptr as usize % 16 == 0
                    && buffer.bytes >= bytes
                    && buffer.device_id == self.device,
                "EXL3 input buffer contract mismatch"
            );
        }
        ensure!(
            !output.ptr.is_null()
                && output.ptr as usize % 16 == 0
                && output.device_id == self.device
                && output.bytes >= rows * self.hidden * self.output_element_bytes,
            "EXL3 output buffer contract mismatch"
        );
        let binding = self.layers.get_mut(layer).context("EXL3 layer is not bound")?;
        ensure!(binding.ownership.is_some() == ownership.is_some(),
            "EXL3 launch ownership/layout mismatch");
        if let Some(source) = ownership {
            let destination = binding.ownership.context("missing EXL3 ownership binding")?;
            ensure!(!source.ptr.is_null() && source.ptr as usize % 4 == 0
                && source.device_id == self.device && source.bytes == destination.bytes,
                "EXL3 ownership device row mismatch");
            self.library.copy_d2d_async(destination, source, destination.bytes, stream)?;
        }
        if let Some((wire, decoded)) = &self.wire {
            wire.decode(inputs[0], decoded.buffer, rows, stream)?;
            inputs[0] = decoded.buffer;
        }
        if let Some(routes) = &self.routes {
            self.route_pointers[0] = inputs[1].ptr;
            self.route_pointers[1] = binding.expert_map.ptr;
            self.route_bytes[1] = binding.expert_map.bytes as u64;
            // Router views may expose only live IDs; metadata scratch retains
            // its compiled capacity while all input reads are live-row bounded.
            self.route_bytes[0] = inputs[1].bytes as u64;
            routes.launch(&self.route_pointers, &self.route_bytes, rows as i32, stream)?;
        }
        binding.core.bind(&inputs, rows);
        binding.sum.bind(&inputs, rows);
        binding.sum.pointers[binding.output_slot] = output.ptr;
        self.kernel
            .launch_core(&binding.core.pointers, &binding.core.scalars, stream)?;
        self.kernel
            .launch_sum(&binding.sum.pointers, &binding.sum.scalars, stream)?;
        output.bytes = rows * self.hidden * self.output_element_bytes;
        Ok(output)
    }
}

#[cfg(test)]
#[path = "execution/shared_tests.rs"]
mod shared_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_row_policy_admits_every_live_tail() -> Result<()> {
        use cuteafd_core::ExpertGeometry as G;
        for geometry in [G::DEEPSEEK_V41, G::DEEPSEEK_V4_FLASH, G::DEEPSEEK_V4_PRO,
            G::GLM5, G::MIMO_V2_FLASH, G::MIMO_V26_PRO, G::QWEN4] {
            assert_eq!(Exl3RowPolicy::for_geometry(geometry), Exl3RowPolicy::Nearest);
        }
        let glmf = Exl3RowPolicy::for_geometry(G::GLM5_FLASH);
        assert_eq!(glmf, Exl3RowPolicy::GlmFlashK64);
        for policy in [glmf, Exl3RowPolicy::Nearest] {
            assert!(policy.capacities(0).is_err());
            assert!(policy.capacities(4097).is_err());
            for maximum in [1, 2, 4, 9, 15, 16, 17, 64, 79, 80, 81, 255, 256,
                257, 1024, 1536, 2048, 4096] {
                let preloaded = policy.capacities(maximum)?;
                assert!(preloaded.windows(2).all(|pair| pair[0] < pair[1]));
                for live in 1..=maximum {
                    let selected = preloaded.iter().find(|&&capacity|
                        capacity as usize >= policy.required_capacity(live)).unwrap();
                    assert!(*selected as usize >= live, "execution must fit every admitted tail");
                    if policy == glmf && live <= 16 {
                        assert!([1, 80].contains(selected), "GLM Flash must retain K64 geometry");
                    }
                }
            }
        }
        assert_eq!(glmf.capacities(1)?, vec![1]);
        Ok(())
    }

    #[test]
    fn narrow_glmf_plan_includes_m80_scratch_and_requires_its_artifact() -> Result<()> {
        let root = tempfile::tempdir()?;
        for capacity in [1, 16, 80] {
            let directory = root.path().join(format!("m{capacity}"));
            std::fs::create_dir(&directory)?;
            let metadata = serde_json::json!({
                "schema":"cuteafd.v41-exl3-aot.v1", "output_dtype":"bf16",
                "sparkinfer_revision":"test", "hidden":16,"intermediate":16,
                "experts":8,"capacity":capacity,"top_k":2,"bits":[3],
                "direct":false,"sms":1,"blocks_per_sm":1,
                "buffers":{
                    "scratch":{"bytes":capacity*64,"dtype":"float32","allocation":"scratch","zero_on_create":false},
                    "epoch":{"bytes":16,"dtype":"int32","allocation":"epoch","zero_on_create":true}
                },
                "objects":[],"trellis_lut":{"file":"lut","bytes":16,"sha256":"test"}
            });
            std::fs::write(directory.join("v41_exl3.json"), serde_json::to_vec(&metadata)?)?;
        }
        let directories = |policy: Exl3RowPolicy, rows| -> Result<Vec<PathBuf>> {
            Ok(policy.capacities(rows)?.iter().map(|c| root.path().join(format!("m{c}"))).collect())
        };
        let ordinary = Exl3Workspace::plan(&directories(Exl3RowPolicy::Nearest, 16)?, Exl3InputFormat::Fp8K32)?;
        for maximum in [2, 4, 9, 16] {
            let planned = directories(Exl3RowPolicy::GlmFlashK64, maximum)?;
            // Admission includes the shared arena and private state even though
            // the transport/input limit stays at the real declared maximum.
            assert_eq!(Exl3Workspace::plan(&planned, Exl3InputFormat::Fp8K32)?, 8320);
            assert!(ordinary < 8320);
        }
        std::fs::remove_file(root.path().join("m80/v41_exl3.json"))?;
        assert!(Exl3Workspace::plan(&directories(Exl3RowPolicy::GlmFlashK64, 16)?, Exl3InputFormat::Fp8K32).is_err());
        assert!(Exl3Workspace::plan(&directories(Exl3RowPolicy::Nearest, 16)?, Exl3InputFormat::Fp8K32).is_ok());
        Ok(())
    }

    #[test]
    fn paired_manifest_requires_complete_explicit_contract() {
        let original = serde_json::json!({
            "schema":"cuteafd.v41-exl3-aot.v1", "output_dtype":"bf16", "sparkinfer_revision":"test",
            "hidden":5120,"intermediate":640,"experts":384,"capacity":80,"top_k":6,
            "bits":[3,4],"swiglu_limit":10.0,"direct":false,"sms":48,"blocks_per_sm":1,
            "buffers":{},"objects":[],"trellis_lut":{"file":"lut","bytes":16,"sha256":"test"}
        });
        let parse = |value: serde_json::Value| serde_json::from_value::<super::Manifest>(value).unwrap();
        assert_eq!(parse(original.clone()).native_layout().unwrap(), cuteafd_ffi::V41Exl3Layout::Disjoint);
        for (boundary, expected) in [("first", cuteafd_ffi::V41Exl3Layout::PairedFirst), ("last", cuteafd_ffi::V41Exl3Layout::PairedLast)] {
            let mut paired = original.clone();
            paired["paired_boundary"] = boundary.into();
            assert!(parse(paired.clone()).native_layout().is_err());
            paired["descriptor_rows"] = 4.into();
            paired["native_info_version"] = 3.into();
            assert_eq!(parse(paired.clone()).native_layout().unwrap(), expected);
            for (key, value) in [("descriptor_rows", serde_json::json!(3)), ("native_info_version", serde_json::json!(2)),
                ("intermediate", serde_json::json!(512)), ("top_k", serde_json::json!(3)), ("bits", serde_json::json!([2,3,4])),
                ("paired_boundary", serde_json::json!("unknown"))] {
                let mut invalid = paired.clone(); invalid[key] = value;
                assert!(parse(invalid).native_layout().is_err());
            }
        }
        let mut invalid = original; invalid["descriptor_rows"] = 4.into();
        assert!(parse(invalid).native_layout().is_err());
    }

    use crate::shared::experts::layer::ExpertLayer;
    use crate::shared::memory::LoadStream;

    struct Fixture<'a> {
        inputs: Vec<DeviceAllocation<'a>>,
        original_input: Vec<u8>,
        expected: Vec<u8>,
    }
    impl Fixture<'_> {
        fn inputs(&self) -> [CuteafdDeviceBuffer; 3] {
            std::array::from_fn(|i| self.inputs[i].buffer)
        }
        fn verify(&self, library: &NativeLibrary, output: CuteafdDeviceBuffer) -> Result<()> {
            let mut actual = vec![0; output.bytes];
            library.copy_d2h(&mut actual, output)?;
            ensure!(
                actual == self.expected[..actual.len()],
                "EXL3 layer output differs from B12x"
            );
            Ok(())
        }
    }
    struct Graph<'a> {
        library: &'a NativeLibrary,
        exec: *mut c_void,
        stream: *mut c_void,
    }
    impl Drop for Graph<'_> {
        fn drop(&mut self) {
            unsafe {
                let _ = self.library.cuda_stream_synchronize(self.stream);
                let _ = self.library.cuda_graph_exec_destroy(self.exec);
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA, CUTEAFD_NATIVE_LIB, CUTEAFD_EXL3_SNAPSHOT, CUTEAFD_EXL3_AOT and CUTEAFD_EXL3_FIXTURE; optional CUTEAFD_EXL3_SECOND_FIXTURE exercises layer 1"]
    fn native_resident_layer_matches_b12x_fixture() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_SNAPSHOT")?);
        let catalog = cuteafd_loader::read_official_v41_catalog(
            cuteafd_loader::OFFICIAL_V41_MODEL_ID,
            &snapshot,
        )?;
        let aot = std::path::PathBuf::from(std::env::var("CUTEAFD_EXL3_AOT")?);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(aot.join("v41_exl3.json"))?)?;
        let mut paths = vec![std::path::PathBuf::from(std::env::var(
            "CUTEAFD_EXL3_FIXTURE",
        )?)];
        if let Some(second) = std::env::var_os("CUTEAFD_EXL3_SECOND_FIXTURE") {
            paths.push(second.into());
        }
        let mut fixtures = Vec::new();
        let mut resident = Vec::new();
        let mut input_format = None;
        for (layer, path) in paths.iter().enumerate() {
            let info: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path.join("fixture.json"))?)?;
            for key in ["direct", "tile", "output_dtype"] {
                ensure!(
                    !info[key].is_null() && info[key] == manifest[key],
                    "EXL3 fixture policy mismatch: {key}"
                );
            }
            ensure!(
                info["slice_start"] == 1280
                    && info["width"] == 512
                    && info["topk"] == 6
                    && info["capacity"] == 16
                    && info["layer"] == format!("layers.{layer}")
                    && info["snapshot_revision"].as_str()
                        == snapshot.file_name().and_then(|s| s.to_str()),
                "unexpected EXL3 fixture geometry or snapshot"
            );
            let format = match info["input_format"].as_str() {
                Some("bf16") => Exl3InputFormat::Bf16,
                Some("fp8_k32") => Exl3InputFormat::Fp8K32,
                _ => anyhow::bail!("invalid fixture input format"),
            };
            ensure!(
                input_format.is_none_or(|old| old == format),
                "fixture input format mismatch"
            );
            input_format = Some(format);
            let layer = ExpertLayer::Backbone { layer, rank: 2 };
            let budget = Exl3Weights::plan(&catalog, layer)?;
            let (free, _) = library.cuda_memory_info()?;
            ensure!(
                free > budget.resident_bytes + 256 * 1024 * 1024,
                "insufficient GPU headroom"
            );
            resident.push(Exl3Weights::load(
                &library,
                &catalog,
                layer,
                budget.resident_bytes,
            )?);
            let mut fixture = Fixture {
                inputs: Vec::new(),
                original_input: Vec::new(),
                expected: Vec::new(),
            };
            for name in ["input", "ids", "weights", "expected"] {
                let bytes = std::fs::read(path.join(format!("{name}.bin")))?;
                ensure!(
                    Some(bytes.len() as u64) == info["artifacts"][name]["bytes"].as_u64()
                        && Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
                            == info["artifacts"][name]["sha256"].as_str(),
                    "corrupt EXL3 fixture {name}"
                );
                if name == "expected" {
                    fixture.expected = bytes;
                } else {
                    let allocation = DeviceAllocation::new(&library, bytes.len())?;
                    library.copy_h2d(allocation.buffer, &bytes)?;
                    fixture.inputs.push(allocation);
                    if name == "input" {
                        fixture.original_input = bytes;
                    }
                }
            }
            fixtures.push(fixture);
        }
        if fixtures.len() == 2 {
            ensure!(
                fixtures[0].expected != fixtures[1].expected,
                "layer fixtures must differ"
            );
        }
        let weights = Rc::new(resident);
        let mut execution = unsafe {
            Exl3Execution::with_input_format(
                &library,
                weights.clone(),
                &aot,
                input_format.unwrap(),
            )?
        };
        let workspace_bytes = execution.workspace_bytes();
        ensure!(
            workspace_bytes == Exl3Execution::plan(&aot, input_format.unwrap())?,
            "EXL3 workspace plan disagrees with allocations"
        );
        let mut second_lane = unsafe {
            Exl3Execution::with_input_format(&library, weights, &aot, input_format.unwrap())?
        };
        ensure!(
            second_lane.workspace_bytes() == workspace_bytes
                && second_lane.output.ptr != execution.output.ptr,
            "lanes must own separate equal-sized workspace"
        );
        let stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        let other_stream = LoadStream {
            library: &library,
            raw: library.cuda_stream_create()?,
        };
        // Different layer orders on two lanes; enqueue both before either host wait.
        for rows in [16, 3, 1, 16] {
            for layer in 0..fixtures.len() {
                let other = fixtures.len() - 1 - layer;
                let output = unsafe {
                    execution.launch_layer(layer, fixtures[layer].inputs(), rows, stream.raw)?
                };
                let other_output = unsafe {
                    second_lane.launch_layer(
                        other,
                        fixtures[other].inputs(),
                        rows,
                        other_stream.raw,
                    )?
                };
                unsafe {
                    library.cuda_stream_synchronize(stream.raw)?;
                    library.cuda_stream_synchronize(other_stream.raw)?;
                }
                fixtures[layer].verify(&library, output)?;
                fixtures[other].verify(&library, other_output)?;
            }
        }
        ensure!(
            unsafe { execution.launch_layer(fixtures.len(), fixtures[0].inputs(), 1, stream.raw) }
                .is_err(),
            "invalid layer must reject before launch"
        );
        let mut graphs = Vec::new();
        for (layer, fixture) in fixtures.iter().enumerate() {
            unsafe {
                library.cuda_graph_begin_capture(stream.raw)?;
            }
            let captured =
                unsafe { execution.launch_layer(layer, fixture.inputs(), 3, stream.raw) };
            let graph = Graph {
                library: &library,
                exec: unsafe { library.cuda_graph_end_capture(stream.raw)? },
                stream: stream.raw,
            };
            graphs.push((graph, captured?));
        }
        // Replay graphs after another layer has used the shared workspace.
        for layer in (0..fixtures.len()).chain((0..fixtures.len()).rev()) {
            let fixture = &fixtures[layer];
            let (graph, output) = &graphs[layer];
            library.copy_h2d(
                fixture.inputs[0].buffer,
                &vec![0; fixture.original_input.len()],
            )?;
            library.copy_h2d(*output, &vec![0xff; output.bytes])?;
            unsafe {
                library.cuda_graph_launch(graph.exec, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            let mut actual = vec![0; output.bytes];
            library.copy_d2h(&mut actual, *output)?;
            ensure!(
                actual.iter().all(|b| *b == 0),
                "EXL3 graph did not consume changed zero input"
            );
            library.copy_h2d(fixture.inputs[0].buffer, &fixture.original_input)?;
            library.copy_h2d(*output, &vec![0xff; output.bytes])?;
            unsafe {
                library.cuda_graph_launch(graph.exec, stream.raw)?;
                library.cuda_stream_synchronize(stream.raw)?;
            }
            fixture.verify(&library, *output)?;
        }
        println!("Rust EXL3: {} layers x 384 resident experts, TP4 rank 2; two independent lanes, {} workspace bytes each; alternating layer rows=16/3/1/16 and changed/restored-input graph replay bitwise equal to six-expert B12x references", fixtures.len(), workspace_bytes);
        Ok(())
    }
}
