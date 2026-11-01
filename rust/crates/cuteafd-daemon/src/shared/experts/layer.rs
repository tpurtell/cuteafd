//! Routed-expert layer identity and native resident weights; one GPU worker owns
//! each layer and its buffers. The layer selects the native kernel family (V4.1
//! MXFP4, NVFP4 W4A4) that the expert service, V4 local experts and V4.1 run.
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
pub(crate) mod nvfp4;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{NativeLibrary, V41ExpertKernel, V41_EXPERT_POINTER_COUNT};
use cuteafd_loader::{OfficialV41Catalog, V41ExpertSelection};
use std::ffi::c_void;

// Bound disk concurrency and pinned staging independently of model layer count.
pub(crate) const EXPERT_READ_LANES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertLayer {
    Backbone { layer: usize, rank: usize },
    BackboneFull { layer: usize },
    BackboneTp2 { layer: usize, rank: usize },
    /// Explicit replicated-group native Spark shard. `world` is this group's
    /// tensor-parallel degree (2 through 8) and `rank` is the shard index
    /// inside the group (`global_rank % world`). Native checkpoints support
    /// replicated groups; NVFP4 uses pure TP3/TP6 only. `world: 6`
    /// is the pure unreplicated `TP6EP1` layout: six disjoint intermediate
    /// slices of every expert, so every rank sees every route.
    BackboneReplicatedTp { layer: usize, rank: usize, world: usize },
    /// EXL3 compact TP shard: one rank of an implicit (no `--spark-tp/--spark-ep`
    /// keys), unreplicated intermediate split of the compressed checkpoint,
    /// carved on whole H128 blocks. `world` is 1 (Qwen whole experts), 3 or 6;
    /// the two-rank compact profile keeps the dedicated
    /// `BackboneTp2` layer so its published behavior is unchanged. This layer
    /// must never resolve to a native FP8/W4A4 kernel or native staging
    /// selection, which would substitute a different weight format.
    BackboneExl3Tp { layer: usize, rank: usize, world: usize },
    Dspark { stage: usize },
    DsparkTp2 { stage: usize, rank: usize },
}

/// `role()` reports a native expert role id. EXL3 shards publish none, so they
/// report a sentinel that is deliberately outside the native role range
/// (coordinator 0, Spark TP4 1, rtx_backbone 2, rtx_tp2 3, dspark_tp2 4,
/// spark_tp2 5, spark_tp3 6, spark_tp6 7). It exists only for startup logging
/// and as a fail-loud guard: it can never equal a native kernel's role.
const EXL3_SHARD_ROLE_SENTINEL: u32 = 100;

impl ExpertLayer {
    pub(crate) fn expert(self, expert: usize) -> V41ExpertSelection {
        match self {
            Self::Backbone { layer, rank } => V41ExpertSelection::Backbone {
                layer,
                rank,
                expert,
            },
            Self::BackboneFull { layer } => V41ExpertSelection::BackboneFull { layer, expert },
            Self::BackboneTp2 { layer, rank } => V41ExpertSelection::BackboneTp2 { layer, expert, rank },
            Self::BackboneReplicatedTp { layer, rank, world } =>
                V41ExpertSelection::BackboneTp { layer, expert, rank, world },
            Self::BackboneExl3Tp { .. } => unreachable!(
                "EXL3 Spark shards never stage native expert weights"
            ),
            Self::Dspark { stage } => V41ExpertSelection::Dspark { stage, expert },
            Self::DsparkTp2 { stage, rank } => V41ExpertSelection::DsparkTp2 { stage, expert, rank },
        }
    }
    /// Physical layer index for resident-weight rebinding.
    pub(crate) fn layer(self) -> usize {
        match self {
            Self::Backbone { layer, .. }
            | Self::BackboneFull { layer }
            | Self::BackboneTp2 { layer, .. }
            | Self::BackboneExl3Tp { layer, .. }
            | Self::BackboneReplicatedTp { layer, .. } => layer,
            Self::Dspark { .. } | Self::DsparkTp2 { .. } => usize::MAX,
        }
    }
    pub(crate) fn native_shard(self) -> Result<cuteafd_core::tp_units::LayerShard> {
        self.shard_for_geometry(cuteafd_core::expert_geometry())
    }

    fn shard_for_geometry(self, geometry: cuteafd_core::ExpertGeometry)
        -> Result<cuteafd_core::tp_units::LayerShard> {
        let (layer, world, rank) = match self {
            Self::Backbone { layer, rank } => (layer, 4, rank),
            Self::BackboneFull { layer } => (layer, 1, 0),
            Self::BackboneTp2 { layer, rank } => (layer, 2, rank),
            Self::BackboneReplicatedTp { layer, rank, world } => (layer, world, rank),
            _ => anyhow::bail!("not a native backbone shard: {self:?}"),
        };
        Ok(cuteafd_core::tp_units::LayerShard::new(geometry.intermediate as usize,
            32, layer, world, rank)?)
    }

    fn unequal_width(self, geometry: cuteafd_core::ExpertGeometry) -> Option<usize> {
        let Self::BackboneReplicatedTp { world, .. } = self else { return None; };
        if world == 0 || (geometry.intermediate as usize / 32) % world == 0 { return None; }
        self.shard_for_geometry(geometry).ok().map(|shard| shard.width)
    }

    pub(crate) fn role(self) -> u32 {
        self.role_for_geometry(cuteafd_core::expert_geometry())
    }

    fn role_for_geometry(self, geometry: cuteafd_core::ExpertGeometry) -> u32 {
        if let Some(role) = self.unequal_width(geometry)
            .and_then(cuteafd_core::tp_units::native_width_role) { return role; }
        match self {
            Self::Dspark { .. } => 0,
            Self::Backbone { .. } => 1,
            Self::BackboneFull { .. } => 2,
            Self::BackboneTp2 { .. } => 3,
            // Explicit replicated shards publish their own native roles; TP4 is
            // only reachable through the legacy `Backbone` layer.
            Self::BackboneReplicatedTp { world: 2, .. } => 5,
            Self::BackboneReplicatedTp { world: 3, .. } => 6,
            Self::BackboneReplicatedTp { world: 6, .. } => 7,
            Self::BackboneReplicatedTp { world: 8, .. } => 8,
            Self::BackboneReplicatedTp { .. } => 1,
            // Compressed shards have no native role; see the sentinel doc.
            Self::BackboneExl3Tp { world, .. } => EXL3_SHARD_ROLE_SENTINEL + world as u32,
            Self::DsparkTp2 { .. } => 4,
        }
    }
    fn info(self, library: &NativeLibrary, capacity: u32) -> Result<cuteafd_ffi::V41ExpertInfo> {
        // The chain below ends in a native catch-all, so the compressed shard
        // has to be refused before it: resolving it to a native expert family
        // would silently serve the wrong weight format.
        if let Self::BackboneExl3Tp { .. } = self {
            return Self::exl3_shard_refusal();
        }
        if let Some(width) = self.unequal_width(cuteafd_core::expert_geometry()) {
            return library.native_width_expert_info(capacity, width);
        }
        if matches!(self, Self::BackboneFull { .. }) {
            library.v41_local_expert_info(capacity)
        } else if matches!(self, Self::BackboneTp2 { .. }) {
            library.v41_tp2_expert_info(capacity)
        } else if matches!(self,Self::DsparkTp2 { .. }) {
            library.v41_dspark_tp2_expert_info(capacity)
        } else if let Self::BackboneReplicatedTp { world, .. } = self {
            match world {
                2 => library.v41_spark_tp2_expert_info(capacity),
                3 => library.v41_spark_tp3_expert_info(capacity),
                4 => library.v41_expert_info(capacity),
                6 => library.v41_spark_tp6_expert_info(capacity),
                8 => library.v41_spark_tp8_expert_info(capacity),
                other => anyhow::bail!("unsupported replicated Spark TP degree {other}"),
            }
        } else { library.v41_expert_info(capacity) }
    }
    fn kernel(self, library: &NativeLibrary, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        if let Self::BackboneExl3Tp { .. } = self {
            return Self::exl3_shard_refusal();
        }
        if let Some(width) = self.unequal_width(cuteafd_core::expert_geometry()) {
            return library.native_width_expert_kernel(capacity, width);
        }
        if matches!(self, Self::BackboneFull { .. }) {
            library.v41_local_expert_kernel(capacity)
        } else if matches!(self, Self::BackboneTp2 { .. }) {
            library.v41_tp2_expert_kernel(capacity)
        } else if matches!(self,Self::DsparkTp2 { .. }) {
            library.v41_dspark_tp2_expert_kernel(capacity)
        } else if let Self::BackboneReplicatedTp { world, .. } = self {
            match world {
                2 => library.v41_spark_tp2_expert_kernel(capacity),
                3 => library.v41_spark_tp3_expert_kernel(capacity),
                4 => library.v41_expert_kernel(capacity),
                6 => library.v41_spark_tp6_expert_kernel(capacity),
                8 => library.v41_spark_tp8_expert_kernel(capacity),
                other => anyhow::bail!("unsupported replicated Spark TP degree {other}"),
            }
        } else { library.v41_expert_kernel(capacity) }
    }

    /// Shared refusal for compressed shards reaching a native-only resolver.
    fn exl3_shard_refusal<T>() -> Result<T> {
        anyhow::bail!(
            "EXL3 Spark shards execute through the EXL3 worker package, \
             not the native expert kernels"
        )
    }

    /// W4A4 NVFP4 family selection. Only backbone experts are quantized; the
    /// MTP draft path stays on the native family.
    fn nvfp4_kernel(self, library: &NativeLibrary, capacity: u32) -> Result<V41ExpertKernel<'_>> {
        match self {
            Self::Backbone { .. } => library.v41_nvfp4_expert_kernel(capacity),
            Self::BackboneTp2 { .. } => library.v41_nvfp4_tp2_expert_kernel(capacity),
            Self::BackboneFull { .. } => library.v41_nvfp4_local_expert_kernel(capacity),
            Self::BackboneReplicatedTp { world: tp @ (3 | 6), .. } =>
                library.v41_nvfp4_spark_expert_kernel(capacity, tp),
            // Raw publications keep draft experts at source MXFP4 precision.
            other => other.kernel(library, capacity),
        }
    }

    fn nvfp4_info(self, library: &NativeLibrary, capacity: u32) -> Result<cuteafd_ffi::V41ExpertInfo> {
        match self {
            Self::Backbone { .. } => library.v41_nvfp4_expert_info(capacity),
            Self::BackboneTp2 { .. } => library.v41_nvfp4_tp2_expert_info(capacity),
            Self::BackboneFull { .. } => library.v41_nvfp4_local_expert_info(capacity),
            Self::BackboneReplicatedTp { world: tp @ (3 | 6), .. } =>
                library.v41_nvfp4_spark_expert_info(capacity, tp),
            other => other.info(library, capacity),
        }
    }

    /// Kernel for this layer under the checkpoint's expert format. Draft
    /// layers always resolve through the native family.
    pub(crate) fn select_kernel(
        self,
        library: &NativeLibrary,
        capacity: u32,
        nvfp4: bool,
    ) -> Result<V41ExpertKernel<'_>> {
        if nvfp4 {
            self.nvfp4_kernel(library, capacity)
        } else {
            self.kernel(library, capacity)
        }
    }

    pub(crate) fn select_info(
        self,
        library: &NativeLibrary,
        capacity: u32,
        nvfp4: bool,
    ) -> Result<cuteafd_ffi::V41ExpertInfo> {
        if nvfp4 {
            self.nvfp4_info(library, capacity)
        } else {
            self.info(library, capacity)
        }
    }

    /// True when this layer's routed experts use the W4A4 NVFP4 family.
    fn is_quantized_nvfp4(self, catalog: &OfficialV41Catalog) -> bool {
        self.uses_nvfp4(ExpertFormat::of(catalog))
    }

    fn uses_nvfp4(self, format: ExpertFormat) -> bool {
        format.is_nvfp4()
            && matches!(
                self,
                Self::Backbone { .. }
                    | Self::BackboneTp2 { .. }
                    | Self::BackboneFull { .. }
                    | Self::BackboneReplicatedTp { world: 3 | 6, .. }
            )
    }
}

/// Expert checkpoint format selected from the validated catalog. All three
/// families can coexist in one library; the daemon picks one per deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertFormat {
    /// Official checkpoint: MXFP4 routed experts with E8M0 K32 scales.
    Native,
    /// Staged or raw EXL3 trellis publications.
    Exl3,
    /// ModelOpt NVFP4 (W4A4) publications.
    Nvfp4,
}

impl ExpertFormat {
    pub(crate) fn of(catalog: &OfficialV41Catalog) -> Self {
        if catalog.exl3().is_some() {
            Self::Exl3
        } else if catalog.nvfp4().is_some() {
            Self::Nvfp4
        } else {
            Self::Native
        }
    }
    pub(crate) fn is_exl3(self) -> bool {
        matches!(self, Self::Exl3)
    }
    pub(crate) fn is_nvfp4(self) -> bool {
        matches!(self, Self::Nvfp4)
    }
}

#[cfg(test)]
mod format_dispatch_tests {
    use super::{ExpertFormat, ExpertLayer};

    #[test]
    fn unequal_native_roles_follow_rotated_adjacent_widths_only() {
        use cuteafd_core::{ExpertGeometry, tp_units::{LayerShard, native_width_role}};
        for geometry in [ExpertGeometry::DEEPSEEK_V41, ExpertGeometry::DEEPSEEK_V4_FLASH,
            ExpertGeometry::DEEPSEEK_V4_PRO] {
            for world in 2..=8 {
                let unequal = (geometry.intermediate as usize / 32) % world != 0;
                for rank in 0..world {
                    let mut roles = std::collections::BTreeSet::new();
                    for layer in 0..world * 2 {
                        let selected = ExpertLayer::BackboneReplicatedTp { layer, world, rank };
                        let shard = LayerShard::new(geometry.intermediate as usize, 32, layer, world, rank).unwrap();
                        assert_eq!(selected.shard_for_geometry(geometry).unwrap(), shard);
                        let role = selected.role_for_geometry(geometry);
                        assert_eq!(selected.unequal_width(geometry), unequal.then_some(shard.width));
                        if unequal { assert_eq!(role, native_width_role(shard.width).unwrap()); }
                        else { assert!(role <= 8); }
                        roles.insert(role);
                    }
                    assert_eq!(roles.len(), if unequal { 2 } else { 1 });
                }
            }
        }
    }

    #[test]
    fn every_backbone_variant_dispatches_by_checkpoint_format() {
        let cases = [
            (ExpertLayer::Backbone { layer: 2, rank: 0 }, true),
            (ExpertLayer::BackboneFull { layer: 2 }, true),
            (ExpertLayer::BackboneTp2 { layer: 2, rank: 0 }, true),
            (ExpertLayer::BackboneReplicatedTp { layer: 2, rank: 0, world: 3 }, true),
            (ExpertLayer::BackboneReplicatedTp { layer: 2, rank: 0, world: 6 }, true),
            (ExpertLayer::BackboneReplicatedTp { layer: 2, rank: 0, world: 2 }, false),
            (ExpertLayer::BackboneReplicatedTp { layer: 2, rank: 0, world: 4 }, false),
            (ExpertLayer::BackboneReplicatedTp { layer: 2, rank: 0, world: 5 }, false),
            (ExpertLayer::BackboneExl3Tp { layer: 2, rank: 0, world: 3 }, false),
            (ExpertLayer::BackboneExl3Tp { layer: 2, rank: 0, world: 6 }, false),
            (ExpertLayer::Dspark { stage: 0 }, false),
            (ExpertLayer::DsparkTp2 { stage: 0, rank: 0 }, false),
        ];
        for format in [ExpertFormat::Native, ExpertFormat::Exl3, ExpertFormat::Nvfp4] {
            for (layer, supports_nvfp4) in cases {
                assert_eq!(layer.uses_nvfp4(format), format == ExpertFormat::Nvfp4 && supports_nvfp4,
                    "{layer:?} with {format:?}");
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertLoadBudget {
    pub resident_bytes: usize,
    pub device_staging_bytes: usize,
    pub pinned_host_bytes: usize,
    pub read_scratch_bytes: usize,
}
impl ExpertLoadBudget {
    pub fn peak_device_bytes(self) -> Result<usize> {
        self.resident_bytes
            .checked_add(self.device_staging_bytes)
            .context("expert load budget overflow")
    }
}

/// A resident plane owns either its mount allocation or a completed rank arena.
pub(crate) struct ResidentPlane<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) buffer: cuteafd_ffi::CuteafdDeviceBuffer,
    _owner: PlaneOwner<'a>,
}
enum PlaneOwner<'a> {
    Mount { _allocation: DeviceAllocation<'a> },
    Push { _rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>> },
}
impl<'a> From<DeviceAllocation<'a>> for ResidentPlane<'a> {
    fn from(allocation: DeviceAllocation<'a>) -> Self {
        Self { library: allocation.library, buffer: allocation.buffer,
            _owner: PlaneOwner::Mount { _allocation: allocation } }
    }
}
impl<'a> ResidentPlane<'a> {
    fn pushed(library: &'a NativeLibrary,
        rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>>,
        region: usize, offset: usize, bytes: usize) -> Result<Self> {
        let buffer = rank.region(region)?.subregion(offset, bytes)?.buffer();
        Ok(Self { library, buffer, _owner: PlaneOwner::Push { _rank: rank } })
    }
}

/// Resident packed weights borrow the native library; no logical layer copy remains.
/// Captured graphs must be destroyed before this owner is dropped.
pub(crate) struct ExpertWeights<'a> {
    pub(crate) buffers: [ResidentPlane<'a>; 4],
    pub(crate) layer: ExpertLayer,
    experts: usize,
    pub(crate) budget: ExpertLoadBudget,
    /// W4A4 resident storage. `None` for the native and EXL3 families.
    nvfp4: Option<nvfp4::Nvfp4Side<'a>>,
    format: ExpertFormat,
    /// A recorded load's raw scale images, until their Done finishes ran.
    raw_scales: RawScales<'a>,
}

/// A recorded load's raw scale images (one device allocation), freed once its
/// Done finishes ran, or with the layer. Covariant (no lifetime in a cell).
pub(crate) struct RawScales<'a> {
    library: Option<&'a NativeLibrary>,
    buffer: std::cell::Cell<Option<cuteafd_ffi::CuteafdDeviceBuffer>>,
}
impl<'a> RawScales<'a> {
    fn none() -> Self { Self { library: None, buffer: std::cell::Cell::new(None) } }
    fn owned(allocation: DeviceAllocation<'a>) -> Self {
        let allocation = std::mem::ManuallyDrop::new(allocation);
        Self { library: Some(allocation.library), buffer: std::cell::Cell::new(Some(allocation.buffer)) }
    }
    /// Frees the images (their packs ran).
    pub(crate) fn release(&self) {
        if let (Some(library), Some(buffer)) = (self.library, self.buffer.take()) {
            drop(DeviceAllocation { library, buffer });
        }
    }
}
impl Drop for RawScales<'_> {
    fn drop(&mut self) { self.release(); }
}
impl<'a> ExpertWeights<'a> {
    /// Frees raw scale images kept for recorded plane packs (after they ran).
    pub(crate) fn release_raw_scales(&self) { self.raw_scales.release(); }

    fn layout(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<(ExpertLoadBudget, [usize; 4], u32, usize)> {
        if layer.is_quantized_nvfp4(catalog) {
            return Self::nvfp4_layout(library, catalog, layer);
        }
        let first = catalog.expert_staging(layer.expert(0))?;
        let info = layer.info(library, 16)?;
        ensure!(
            info.role == layer.role(),
            "native expert role does not match layer placement"
        );
        ensure!(
            info.logical_intermediate as usize == first.intermediate_size(),
            "native expert intermediate mismatch"
        );
        let experts = info.experts as usize;
        let packer = library.v41_expert_packer(info.logical_intermediate)?;
        let coded = catalog.mxfp4_scale_codes();
        let strides = if coded {
            cuteafd_loader::formats::scale_codes::expert_planes(info.hidden_size as usize,
                info.logical_intermediate as usize, info.kernel_intermediate as usize)
                .context("invalid coded expert geometry")?.map(|n| n as u64)
        } else { packer.packed_bytes() };
        let mut sizes = [0usize; 4];
        for (size, stride) in sizes.iter_mut().zip(strides) {
            *size = usize::try_from(stride)?.checked_mul(experts)
                .context("resident expert allocation overflow")?;
        }
        let resident_bytes = usize::try_from(if coded {
            cuteafd_loader::formats::scale_codes::expert_bytes(experts as u64,
                info.hidden_size as u64, info.logical_intermediate as u64)
                .context("coded expert residency overflow")?
        } else {
            cuteafd_loader::placement::families::deepseek_v41::resident_bytes(
                info.hidden_size as u64, info.kernel_intermediate as u64, experts as u64, false)?
        })?;
        ensure!(sizes.iter().sum::<usize>() == resident_bytes,
            "native packer planes disagree with shared resident demand");
        let budget = ExpertLoadBudget {
            resident_bytes,
            device_staging_bytes: first.staging_bytes(),
            pinned_host_bytes: first
                .staging_bytes()
                .checked_mul(EXPERT_READ_LANES)
                .and_then(|n| n.checked_add(if coded { 1 << 20 } else { 0 }))
                .context("expert pinned staging overflow")?,
            read_scratch_bytes: first
                .minimum_read_scratch_bytes()
                .checked_mul(64 * EXPERT_READ_LANES)
                .context("expert read scratch overflow")?
                .max(cuteafd_loader::EXPERT_READ_WINDOW_BYTES * EXPERT_READ_LANES),
        };
        Ok((budget, sizes, info.logical_intermediate, experts))
    }
    pub fn plan(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<ExpertLoadBudget> {
        Ok(Self::layout(library, catalog, layer)?.0)
    }

    /// W4A4 layout: the four per-expert planes replace the packed W4A8 slabs.
    fn nvfp4_layout(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
    ) -> Result<(ExpertLoadBudget, [usize; 4], u32, usize)> {
        let (intermediate, experts) =
            nvfp4::Nvfp4Side::rank_planes(layer, catalog)?;
        let budget = nvfp4::Nvfp4Side::plan(library, catalog, layer)?;
        let sizes = nvfp4::plane_sizes(nvfp4::Nvfp4Side::kernel_intermediate(library, catalog, layer)?);
        Ok((budget, sizes, intermediate as u32, experts))
    }

    /// True when this layer's experts are W4A4 ModelOpt NVFP4.
    pub(crate) fn is_nvfp4(&self) -> bool {
        self.nvfp4.is_some()
    }

    pub(crate) fn format(&self) -> ExpertFormat {
        self.format
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<Self> {
        if layer.is_quantized_nvfp4(catalog) {
            return Self::load_nvfp4(library, catalog, layer, available_device_bytes);
        }
        let (budget, sizes, intermediate, experts) = Self::layout(library, catalog, layer)?;
        let packer = library.v41_expert_packer(intermediate)?;
        ensure!(budget.peak_device_bytes()? <= available_device_bytes,
            "expert layer needs {} device bytes including staging, budget is {available_device_bytes}", budget.peak_device_bytes()?);
        // Fail role/device checks and allocation admission before opening payloads.
        // The NVFP4 family loads its own variants; both report the same roles.
        let format = ExpertFormat::of(catalog);
        let _kernel = if format.is_nvfp4() {
            layer.nvfp4_kernel(library, 16)?
        } else {
            layer.kernel(library, 16)?
        };
        let load_started = std::time::Instant::now();
        let mut owned = Vec::with_capacity(4);
        for size in sizes {
            owned.push(DeviceAllocation::new(library, size)?);
        }
        let buffers: [DeviceAllocation<'a>; 4] =
            owned.try_into().ok().expect("four packed buffers");
        if crate::shared::experts::rtx::dense::is_recording() {
            // Recorded for the coordinator push group: one staged step per expert,
            // packed on the GPU into its slice of the four planes.
            let raw_scales = Self::record_steps(library, catalog, layer, intermediate, experts, &sizes, &buffers)?;
            return Ok(Self { buffers: buffers.map(Into::into), layer, experts, budget, nvfp4: None,
                format: ExpertFormat::of(catalog), raw_scales });
        }
        let mut scale_kernels = if catalog.mxfp4_scale_codes() {
            Some(crate::shared::fp8_load_quant::LoadKernels::new(library)?)
        } else { None };
        if scale_kernels.is_some() {
            // SAFETY: these unpublished planes are retained through this drain.
            let queued = unsafe { (|| {
                for i in [1, 3] { library.zero_pitched_async(buffers[i].buffer, sizes[i], sizes[i], 1,
                    std::ptr::null_mut())?; }
                Ok::<_, anyhow::Error>(())
            })() };
            let drained = unsafe { library.cuda_stream_synchronize(std::ptr::null_mut()) };
            if drained.is_err() { library.quarantine_module_after_failed_drain(); }
            drained?;
            queued?;
        }
        let device_staging = DeviceAllocation::new(library, budget.device_staging_bytes)?;
        let mut hosts = (0..EXPERT_READ_LANES)
            .map(|_| HostAllocation::new(library, budget.device_staging_bytes))
            .collect::<Result<Vec<_>>>()?;
        let mut read_scratch = (0..EXPERT_READ_LANES)
            .map(|_| vec![0; budget.read_scratch_bytes / EXPERT_READ_LANES])
            .collect::<Vec<_>>();
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let allocation_seconds = load_started.elapsed().as_secs_f64();
        let mut planning_seconds = 0.;
        let mut read_seconds = 0.;
        let mut upload_pack_seconds = 0.;
        let mut storage_bytes_read = 0;
        // WILLNEED can perform blocking I/O, serializing all six tensor extents
        // per expert. Let the bounded parallel readers issue the reads instead.
        // CPU readers borrow disjoint pinned byte slices; all CUDA calls remain
        // on this owning thread, after the scoped readers have joined.
        for first in (0..experts).step_by(EXPERT_READ_LANES) {
            let end = (first + EXPERT_READ_LANES).min(experts);
            let started = std::time::Instant::now();
            let plans = (first..end)
                .map(|expert| catalog.expert_staging(layer.expert(expert)))
                .collect::<Result<Vec<_>>>()?;
            planning_seconds += started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            std::thread::scope(|scope| -> Result<()> {
                let mut readers = Vec::with_capacity(plans.len());
                for ((plan, host), scratch) in plans.iter().zip(&mut hosts).zip(&mut read_scratch) {
                    let bytes = host.bytes_mut();
                    readers.push(scope.spawn(move || OfficialV41Catalog::count_storage_reads(|| plan.read_into(bytes, scratch))));
                }
                for reader in readers {
                    storage_bytes_read += reader
                        .join()
                        .map_err(|_| anyhow::anyhow!("expert read thread panicked"))??.1;
                }
                Ok(())
            })?;
            read_seconds += started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            for (offset, (plan, host)) in plans.iter().zip(&hosts).enumerate() {
                let expert = first + offset;
                unsafe {
                    library.copy_host_buffer_h2d_async(
                        device_staging.buffer,
                        host.buffer,
                        plan.staging_bytes(),
                        stream.raw,
                    )?;
                    let sources = std::array::from_fn(|i| {
                        device_staging
                            .buffer
                            .ptr
                            .cast::<u8>()
                            .add(plan.tensor_ranges()[i].start)
                            .cast_const()
                    });
                    let destinations = std::array::from_fn(|i| {
                        buffers[i]
                            .buffer
                            .ptr
                            .cast::<u8>()
                            .add(expert * (sizes[i] / experts))
                    });
                    if scale_kernels.is_some() {
                        library.v41_pack_weights_async([sources[0], sources[1], sources[2]],
                            [destinations[0], destinations[2]], intermediate, stream.raw)?;
                    } else { packer.pack(sources, destinations, stream.raw)?; }
                    // The next copy reuses device staging on this same stream,
                    // after this pack. Distinct pinned inputs stay alive for the
                    // entire group. LoadStream::drop drains on partial failure.
                }
                if let Some(kernels) = &mut scale_kernels {
                    use crate::shared::fp8_load_quant::ScaleCodeTarget;
                    use cuteafd_loader::formats::scale_codes::Rows;
                    let hidden = catalog.routed_experts().hidden;
                    let logical = intermediate as usize;
                    let stored = cuteafd_core::mxfp4_layouts::stored_width(hidden, logical)
                        .context("invalid coded expert resident width")?;
                    let first_row = if let Some(shard) = plan.geometry() {
                        shard.first_intermediate()
                    } else { match plan.selection() {
                        V41ExpertSelection::Backbone { rank, .. }
                        | V41ExpertSelection::BackboneTp2 { rank, .. }
                        | V41ExpertSelection::BackboneTp { rank, .. }
                        | V41ExpertSelection::DsparkTp2 { rank, .. } => rank * logical,
                        _ => 0,
                    }};
                    let targets = [3usize, 4, 5].map(|slot| {
                        let down = slot == 5;
                        let plane = if down { 3 } else { 1 };
                        let stride = sizes[plane] / experts;
                        // SAFETY: this expert's slice lies in its admitted plane.
                        let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                            ptr: unsafe { buffers[plane].buffer.ptr.cast::<u8>().add(expert * stride) }.cast(),
                            bytes: stride, ..buffers[plane].buffer };
                        ScaleCodeTarget { source: Rows { tensor: &plan.tensor_names()[slot],
                            bytes: &host.bytes()[plan.tensor_ranges()[slot].clone()],
                            rows: if down { hidden } else { logical },
                            pitch: if down { logical / 32 } else { hidden / 32 },
                            columns: if down { logical / 32 } else { hidden / 32 },
                            first_row: if down { 0 } else { first_row } },
                            destination, stored_rows: if down { hidden } else { 2 * stored },
                            start_row: if slot == 3 { stored } else { 0 } }
                    });
                    kernels.scale_codes_from(&targets, stream.raw)?;
                }
            }
            // Only CPU reuse of pinned staging requires host completion.
            unsafe { library.cuda_stream_synchronize(stream.raw)?; }
            upload_pack_seconds += started.elapsed().as_secs_f64();
        }
        tracing::info!(?layer, experts, storage_bytes_read, allocation_seconds, planning_seconds, read_seconds,
            upload_pack_seconds, elapsed_seconds = load_started.elapsed().as_secs_f64(),
            staging_bytes_per_expert = budget.device_staging_bytes,
            "native expert load timeline");
        Ok(Self {
            buffers: buffers.map(Into::into),
            layer,
            experts,
            budget,
            nvfp4: None,
            format: ExpertFormat::of(catalog), raw_scales: RawScales::none() })
    }

    /// A native layer's records for the coordinator push group, bounded per
    /// source in file order: each expert's weights ([w3, w1, w2], one
    /// contiguous file region) are one staged step packed into planes 0 and 2;
    /// all scales (stored apart, at the shard's front) land class 2 in a
    /// transient device image `raw` and finishes pack planes 1 and 3 at Done.
    fn record_steps(library: &'a NativeLibrary, catalog: &OfficialV41Catalog, layer: ExpertLayer, intermediate: u32,
        experts: usize, sizes: &[usize; 4], buffers: &[DeviceAllocation<'a>; 4]) -> Result<RawScales<'a>> {
        use crate::shared::experts::rtx::dense::{DenseLoad, Finish, Job, Operand, Rect, Step, STEP_BYTES};
        let dense = DenseLoad::mount(library);
        let device = library.cuda_get_device()?;
        let whole = |name: &str| -> Result<Rect> {
            let tensor = catalog.tensor(name)?;
            let bytes = usize::try_from(tensor.metadata.byte_length)?;
            Ok(Rect { name: name.into(), path: catalog.snapshot().join(&tensor.shard), base: tensor.metadata.byte_offset,
                row_bytes: bytes, rows: 0..1, cols: 0..bytes })
        };
        let plane = |i: usize, expert: usize| {
            let stride = sizes[i] / experts;
            // SAFETY (offset): expert `expert`'s slice lies inside plane `i`.
            cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe { buffers[i].buffer.ptr.cast::<u8>().add(expert * stride) }.cast(),
                bytes: stride, ..buffers[i].buffer }
        };
        // Raw scale images, per expert: [w3 scale; w1 scale] then the w2 scale.
        let first = catalog.expert_staging(layer.expert(0))?;
        let names = first.tensor_names();
        let (gate_scale, down_scale) = (2 * catalog.tensor(&names[3])?.metadata.byte_length as usize,
            catalog.tensor(&names[5])?.metadata.byte_length as usize);
        let per = (gate_scale + down_scale).next_multiple_of(256);
        let raw = if catalog.mxfp4_scale_codes() {
            // Generated planes must be initialized before independently ordered
            // scale tensors arrive; a later up/gate job must never erase a peer.
            // SAFETY: unpublished buffers remain owned through the drain.
            let queued = unsafe { (|| {
                for i in [1, 3] { library.zero_pitched_async(buffers[i].buffer, sizes[i], sizes[i], 1,
                    std::ptr::null_mut())?; }
                Ok::<_, anyhow::Error>(())
            })() };
            let drained = unsafe { library.cuda_stream_synchronize(std::ptr::null_mut()) };
            if drained.is_err() { library.quarantine_module_after_failed_drain(); }
            drained?;
            queued?;
            None
        } else { Some(DeviceAllocation::new(library, per * experts)?) };
        for expert in 0..experts {
            let plan = catalog.expert_staging(layer.expert(expert))?;
            ensure!(plan.tp_world().is_none() && !matches!(layer.expert(expert),
                V41ExpertSelection::BackboneTp2 { .. } | V41ExpertSelection::DsparkTp2 { .. }),
                "recorded native experts are whole tensors");
            let n = plan.tensor_names();
            // Weights: one contiguous step (bounded per open source).
            let rects = vec![whole(&n[1])?, whole(&n[0])?, whole(&n[2])?];
            let gate_bytes = rects[0].bytes() + rects[1].bytes();
            let bytes = gate_bytes + rects[2].bytes();
            ensure!(bytes <= STEP_BYTES, "native expert weights exceed one staged step");
            dense.step(Step { device, rects, jobs: vec![
                Job::NativePlane { intermediate, plane: 0, at: 0, bytes: gate_bytes, dst: plane(0, expert) },
                Job::NativePlane { intermediate, plane: 2, at: gate_bytes, bytes: bytes - gate_bytes, dst: plane(2, expert) },
            ] }, std::ptr::null_mut())?;
            if catalog.mxfp4_scale_codes() {
                let hidden = catalog.routed_experts().hidden;
                let stored = cuteafd_core::mxfp4_layouts::stored_width(hidden, intermediate as usize)
                    .context("invalid coded expert resident width")?;
                for (slot, start_row) in [(4, 0), (3, stored), (5, 0)] {
                    let tensor = catalog.tensor(&n[slot])?;
                    let rows = tensor.metadata.shape[0];
                    let columns = tensor.metadata.shape[1];
                    let rect = Rect { name: n[slot].clone(), path: catalog.snapshot().join(&tensor.shard),
                        base: tensor.metadata.byte_offset, row_bytes: columns, rows: 0..rows, cols: 0..columns };
                    dense.step(Step { device, rects: vec![rect], jobs: vec![Job::ScaleCodes {
                        rect: 0, at: 0, stored_rows: if slot == 5 { hidden } else { 2 * stored },
                        start_row, dst: plane(if slot == 5 { 3 } else { 1 }, expert),
                    }] }, std::ptr::null_mut())?;
                }
                continue;
            }
            // Other native packages retain their raw layout and post-Done pack.
            let raw = raw.as_ref().context("raw scale image absent")?;
            let (up, gate, down) = (whole(&n[4])?, whole(&n[3])?, whole(&n[5])?);
            ensure!(up.bytes() + gate.bytes() == gate_scale && down.bytes() == down_scale, "native scale extents differ");
            // SAFETY (offset): expert `expert`'s image lies inside `raw`.
            let image = cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe { raw.buffer.ptr.cast::<u8>().add(expert * per) }.cast(),
                bytes: per, ..raw.buffer };
            dense.operand(Operand { device, buffer: image, bytes: gate_scale + down_scale,
                lands: vec![(up.clone(), 0), (gate, up.bytes()), (down, gate_scale)] })?;
            // SAFETY (offsets): both halves lie inside this expert's image.
            let at = |offset: usize, bytes: usize| cuteafd_ffi::CuteafdDeviceBuffer {
                ptr: unsafe { image.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..image };
            dense.finish(Finish::NativePlane { intermediate, plane: 1, source: at(0, gate_scale), dst: plane(1, expert) },
                std::ptr::null_mut())?;
            dense.finish(Finish::NativePlane { intermediate, plane: 3, source: at(gate_scale, down_scale), dst: plane(3, expert) },
                std::ptr::null_mut())?;
        }
        Ok(raw.map(RawScales::owned).unwrap_or_else(RawScales::none))
    }

    /// Read each native projection once for both GPUs. Eight readers in each
    /// of two full-width banks fit the original pair's 16 half-width slots.
    pub(crate) fn load_tp2_pair(
        devices: [crate::shared::memory::device::Device<'a>; 2],
        catalog: &OfficialV41Catalog,
        layers: &[usize],
        available: [usize; 2],
    ) -> Result<[crate::shared::memory::device::DeviceOwner<'a, Vec<Self>>; 2]> {
        use super::paired_load::{self, Projection, Fences};
        let started = std::time::Instant::now();
        let lanes = EXPERT_READ_LANES / 2;
        let mut weights = [devices[0].own(|| Ok(Vec::new()))?, devices[1].own(|| Ok(Vec::new()))?];
        let mut staging_bytes = [0; 2];
        let mut pinned_admitted = 0;
        let mut full_bytes = 0;
        let mut remaining = available;
        for &layer in layers {
            let full = catalog.expert_staging(V41ExpertSelection::BackboneFull { layer, expert: 0 })?;
            full_bytes = full_bytes.max(full.staging_bytes());
            let mut pinned = 0;
            for rank in 0..2 {
                let selection = ExpertLayer::BackboneTp2 { layer, rank };
                let (budget, sizes, _, experts) = Self::layout(devices[rank].library, catalog, selection)?;
                ensure!(budget.peak_device_bytes()? <= remaining[rank], "paired native admission exceeded");
                staging_bytes[rank] = staging_bytes[rank].max(budget.device_staging_bytes);
                pinned += budget.pinned_host_bytes;
                remaining[rank] -= budget.resident_bytes;
                devices[rank].run(|| {
                    let mut owned = Vec::with_capacity(4);
                    for bytes in sizes { owned.push(DeviceAllocation::new(devices[rank].library, bytes)?); }
                    weights[rank].push(Self { buffers: <[DeviceAllocation<'a>; 4]>::try_from(owned).ok().expect("four native slabs").map(Into::into),
                        layer: selection, experts, budget, nvfp4: None, format: ExpertFormat::Native, raw_scales: RawScales::none() });
                    Ok(())
                })?;
            }
            pinned_admitted = pinned_admitted.max(pinned);
        }
        let coded = catalog.mxfp4_scale_codes();
        let mut scale_kernels = devices.map(|device| device.own(|| {
            if coded { Ok(Some(crate::shared::fp8_load_quant::LoadKernels::new(device.library)?)) }
            else { Ok(None) }
        })).into_iter().collect::<Result<Vec<_>>>()?;
        if coded {
            for rank in 0..2 {
                devices[rank].run(|| {
                    // SAFETY: all planes are unpublished and retained through the drain.
                    let queued = unsafe { (|| {
                        for weight in weights[rank].iter() {
                            for plane in [1, 3] {
                                let dst = weight.buffers[plane].buffer;
                                devices[rank].library.zero_pitched_async(dst, dst.bytes, dst.bytes, 1,
                                    std::ptr::null_mut())?;
                            }
                        }
                        Ok::<_, anyhow::Error>(())
                    })() };
                    let drained = unsafe { devices[rank].library.cuda_stream_synchronize(std::ptr::null_mut()) };
                    if drained.is_err() { devices[rank].library.quarantine_module_after_failed_drain(); }
                    drained?;
                    queued
                })?;
            }
        }
        let staging = [devices[0].own(|| DeviceAllocation::new(devices[0].library, staging_bytes[0]))?,
            devices[1].own(|| DeviceAllocation::new(devices[1].library, staging_bytes[1]))?];
        let table_bytes = if coded { 2 << 20 } else { 0 };
        let bank_admitted = pinned_admitted.checked_sub(table_bytes).context("paired scale tables under-admitted")?;
        let mut hosts = paired_load::banks(devices[0], lanes, full_bytes, bank_admitted)?;
        let mut fences = Fences::new(devices)?;
        // Last owners constructed: drain both streams before any bank/slab release.
        let streams = paired_load::streams(devices)?;
        let allocation_seconds = started.elapsed().as_secs_f64();
        let mut group = 0;
        for (index, &layer) in layers.iter().enumerate() {
            let layer_started = std::time::Instant::now();
            let experts = weights[0][index].experts;
            let mut storage_bytes_read = 0;
            let mut read_seconds = 0.;
            let mut bank_wait_seconds = 0.;
            let mut upload_submit_seconds = 0.;
            for first in (0..experts).step_by(lanes) {
                let bank = group % 2;
                let wait_started = std::time::Instant::now();
                fences.reuse(bank)?;
                bank_wait_seconds += wait_started.elapsed().as_secs_f64();
                let count = lanes.min(experts - first);
                let full_plans = (first..first + count).map(|expert|
                    catalog.expert_staging(V41ExpertSelection::BackboneFull { layer, expert }))
                    .collect::<Result<Vec<_>>>()?;
                let rank_plans = (0..2).map(|rank| (first..first + count).map(|expert|
                    catalog.expert_staging(V41ExpertSelection::BackboneTp2 { layer, expert, rank }))
                    .collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
                let plans = full_plans.iter().map(|plan| plan.tensor_names().iter().enumerate().map(|(slot, name)| {
                    Projection::new(catalog, name.clone(), plan.tensor_ranges()[slot].start,
                        [catalog.backbone_tp2_slice(name, 0)?, catalog.backbone_tp2_slice(name, 1)?])
                }).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
                paired_load::trace_overlap(&streams, "read_start", layer, first)?;
                let read_started = std::time::Instant::now();
                storage_bytes_read += paired_load::read_group(catalog, &plans, &mut hosts[bank])?;
                read_seconds += read_started.elapsed().as_secs_f64();
                let upload_started = std::time::Instant::now();
                for rank in 0..2 {
                    devices[rank].run(|| {
                        let weight = &weights[rank][index];
                        let packer = devices[rank].library.v41_expert_packer(rank_plans[rank][0].intermediate_size() as u32)?;
                        for (lane, jobs) in plans.iter().enumerate() {
                            let ranges = rank_plans[rank][lane].tensor_ranges();
                            for (slot, job) in jobs.iter().enumerate() {
                                if coded && slot >= 3 { continue; }
                                let mut destination = staging[rank].buffer;
                                // SAFETY: rank staging and tensor ranges were admitted before allocation.
                                unsafe { destination.ptr = destination.ptr.cast::<u8>().add(ranges[slot].start).cast(); }
                                destination.bytes = ranges[slot].len();
                                job.upload(devices[rank], rank, hosts[bank][lane].buffer, destination, &streams[rank])?;
                            }
                            // SAFETY: all sources/destinations belong to this rank; its
                            // serialized stream packs before the next staging overwrite.
                            unsafe {
                                let sources = std::array::from_fn(|slot| staging[rank].buffer.ptr.cast::<u8>()
                                    .add(ranges[slot].start).cast_const());
                                let destinations = std::array::from_fn(|slot| weight.buffers[slot].buffer.ptr.cast::<u8>()
                                    .add((first + lane) * (weight.buffers[slot].buffer.bytes / experts)));
                                if coded {
                                    devices[rank].library.v41_pack_weights_async([sources[0], sources[1], sources[2]],
                                        [destinations[0], destinations[2]], rank_plans[rank][lane].intermediate_size() as u32,
                                        streams[rank].raw)?;
                                } else { packer.pack(sources, destinations, streams[rank].raw)?; }
                            }
                            if let Some(kernels) = &mut *scale_kernels[rank] {
                                use crate::shared::fp8_load_quant::ScaleCodeTarget;
                                use cuteafd_loader::formats::scale_codes::Rows;
                                let hidden = catalog.routed_experts().hidden;
                                let logical = rank_plans[rank][lane].intermediate_size();
                                let stored = cuteafd_core::mxfp4_layouts::stored_width(hidden, logical)
                        .context("invalid coded expert resident width")?;
                                let first_row = rank_plans[rank][lane].geometry()
                                    .map_or(rank * logical, |shard| shard.first_intermediate());
                                let targets = [3usize, 4, 5].map(|slot| {
                                    let job = &jobs[slot];
                                    let slice = job.slices[rank];
                                    let down = slot == 5;
                                    let plane = if down { 3 } else { 1 };
                                    let stride = weight.buffers[plane].buffer.bytes / experts;
                                    // SAFETY: this expert's view lies inside its admitted coded plane.
                                    let destination = cuteafd_ffi::CuteafdDeviceBuffer {
                                        ptr: unsafe { weight.buffers[plane].buffer.ptr.cast::<u8>()
                                            .add((first + lane) * stride) }.cast(),
                                        bytes: stride, ..weight.buffers[plane].buffer };
                                    let start = job.offset + slice.column_start_bytes;
                                    ScaleCodeTarget { source: Rows { tensor: &job.name,
                                        bytes: &hosts[bank][lane].bytes()[start..job.offset + job.bytes],
                                        rows: if down { hidden } else { logical },
                                        pitch: if down { slice.source_row_bytes } else { hidden / 32 },
                                        columns: if down { logical / 32 } else { hidden / 32 },
                                        first_row: if down { 0 } else { first_row } },
                                        destination, stored_rows: if down { hidden } else { 2 * stored },
                                        start_row: if slot == 3 { stored } else { 0 } }
                                });
                                kernels.scale_codes_from(&targets, streams[rank].raw)?;
                            }
                        }
                        Ok(())
                    })?;
                }
                upload_submit_seconds += upload_started.elapsed().as_secs_f64();
                fences.record(bank, &streams)?;
                paired_load::trace_overlap(&streams, "both_submitted", layer, first)?;
                group += 1;
            }
            tracing::info!(layer, storage_bytes_read, read_seconds, bank_wait_seconds, upload_submit_seconds,
                elapsed_seconds = layer_started.elapsed().as_secs_f64(),
                "native TP2 shared-read layer timeline");
        }
        paired_load::drain(&streams)?;
        tracing::info!(allocation_seconds, elapsed_seconds = started.elapsed().as_secs_f64(),
            pinned_host_bytes = full_bytes * lanes * 2, pinned_admitted,
            read_scratch_bytes = 0, "native TP2 shared-read load complete");
        Ok(weights)
    }

    /// Load the W4A4 planes for one layer and adopt its buffers.
    fn load_nvfp4(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: ExpertLayer,
        available_device_bytes: usize,
    ) -> Result<Self> {
        let (buffers, side) = nvfp4::Nvfp4Side::load(library, catalog, layer, available_device_bytes)?;
        let budget = nvfp4::Nvfp4Side::plan(library, catalog, layer)?;
        let (_, experts) = nvfp4::Nvfp4Side::rank_planes(layer, catalog)?;
        Ok(Self {
            buffers,
            layer,
            experts,
            budget,
            nvfp4: Some(side),
            format: ExpertFormat::Nvfp4, raw_scales: RawScales::none() })
    }
    pub(crate) fn from_push(library: &'a NativeLibrary,
        rank: std::rc::Rc<crate::shared::memory::weight_arena::ReadyArena<'a>>,
        region: usize, layer: ExpertLayer, layout: &cuteafd_loader::push_load::native::NativePlanes,
        mut budget: ExpertLoadBudget) -> Result<Self> {
        ensure!(rank.region(region)?.buffer().bytes == layout.bytes && layout.bytes == budget.resident_bytes,
            "native pushed layer differs from admission");
        let mut planes = Vec::new();
        for (offset, bytes) in layout.planes {
            planes.push(ResidentPlane::pushed(library, rank.clone(), region, offset, bytes)?);
        }
        let nvfp4 = if layout.nvfp4 {
            let vector = |index: usize| {
                let (offset, bytes) = layout.vectors[index];
                ResidentPlane::pushed(library, rank.clone(), region, offset, bytes)
            };
            Some(nvfp4::Nvfp4Side { alphas: vector(0)?, down_alphas: vector(1)?,
                input_scales: vector(2)?, down_input_scales: vector(3)? })
        } else { None };
        budget.device_staging_bytes = 0;
        budget.pinned_host_bytes = 0;
        budget.read_scratch_bytes = 0;
        Ok(Self { buffers: planes.try_into().ok().expect("four native planes"), layer,
            experts: layout.experts, budget, nvfp4,
            format: if layout.nvfp4 { ExpertFormat::Nvfp4 } else { ExpertFormat::Native }, raw_scales: RawScales::none() })
    }

    /// Intermediate values per expert held by this resident layer.
    ///
    /// Dimensional check: the W2 carrier for one expert is `hidden * intermediate
    /// / 2` bytes (two packed values per byte), and the allocation holds one such
    /// carrier per expert, so `w2_bytes * 2 / experts / hidden` recovers the
    /// intermediate extent. Dividing by `experts` alone would be dimensionally
    /// wrong by a factor of `hidden`.
    pub fn intermediate(&self) -> usize {
        let w2_bytes = self.buffers[2].buffer.bytes;
        let per_expert = w2_bytes / self.experts.max(1);
        per_expert * 2 / cuteafd_core::expert_geometry().hidden as usize
    }

    pub(crate) fn resident_bytes(&self) -> Result<usize> {
        let mut bytes = self.buffers.iter().try_fold(0usize, |sum, owner|
            sum.checked_add(owner.buffer.bytes).context("native resident bytes overflow"))?;
        if let Some(side) = &self.nvfp4 {
            for owner in [&side.alphas, &side.down_alphas, &side.input_scales, &side.down_input_scales] {
                bytes = bytes.checked_add(owner.buffer.bytes).context("NVFP4 resident bytes overflow")?;
            }
        }
        Ok(bytes)
    }

    pub fn budget(&self) -> ExpertLoadBudget {
        self.budget
    }

    /// Bind prepared weight carriers after scratch binding and before argument validation.
    /// Returned raw slots borrow this owner and must not outlive it, including graph replay.
    pub fn bind(
        &self,
        kernel: &V41ExpertKernel<'_>,
        slots: &mut [*mut c_void; V41_EXPERT_POINTER_COUNT],
    ) -> Result<()> {
        ensure!(
            kernel.info().role == self.layer.role()
                && kernel.info().experts as usize == self.experts,
            "expert weights do not match kernel role"
        );
        ensure!(
            self.is_nvfp4() == (kernel.output_kind() == cuteafd_ffi::V41ExpertOutputKind::Bf16Routes),
            "expert weights and kernel quantization families differ: layer={:?} weights_nvfp4={} \
             kernel_role={} kernel_output_kind={:?}",
            self.layer,
            self.is_nvfp4(),
            kernel.info().role,
            kernel.output_kind()
        );
        ensure!(
            !slots[34].is_null() && !slots[37].is_null(),
            "bind initialized scratch before weights"
        );
        if let Some(side) = &self.nvfp4 {
            // W4A4: fused FC1 payload and scale plane, FC2 payload and scale
            // plane, then the resident per-expert alphas and activation scales.
            let (w13, s13, w2, s2) = (
                self.buffers[0].buffer.ptr,
                self.buffers[1].buffer.ptr,
                self.buffers[2].buffer.ptr,
                self.buffers[3].buffer.ptr,
            );
            // 26..33 are W4A8-only planes the generated bridge never forwards,
            // but the engine refuses to launch with a null slot, so each one
            // aliases a live plane. 28/29 hold the residual planes.
            for (slot, pointer) in [
                (22, w13),
                (23, s13),
                (24, w2),
                (25, s2),
                (26, s13),
                (27, s2),
                (28, w13),
                (29, w2),
                (30, w13),
                (31, s13),
                (32, w2),
                (33, s2),
                (37, side.input_scales.buffer.ptr),
                (38, side.alphas.buffer.ptr),
                (39, side.down_alphas.buffer.ptr),
                (40, side.down_input_scales.buffer.ptr),
            ] {
                slots[slot] = pointer;
            }
            return Ok(());
        }
        let [w13, s13, w2, s2] = std::array::from_fn(|i| self.buffers[i].buffer.ptr);
        for (slot, pointer) in [
            (22, w13),
            (23, s13),
            (24, w2),
            (25, s2),
            (26, s13),
            (27, s2),
            (28, slots[34]),
            (29, slots[34]),
            (30, w13),
            (31, s13),
            (32, w2),
            (33, s2),
            (38, slots[37]),
            (39, slots[37]),
        ] {
            slots[slot] = pointer;
        }
        Ok(())
    }
}
