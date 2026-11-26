//! Thin Hy4-flash binding over `families::hy4_common::native`.
use crate::families::hy4_common::native as shared;
use crate::shared::memory::device::Device;
use crate::shared::memory::weight_arena::DeviceWeightArena;
use anyhow::Result;
use cuteafd_ffi::programs::Programs;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::hy4_flash::weight_layout::{fp8_role, CoordinatorWeights};
use cuteafd_loader::formats::trellis_core::{CoreProjection, ProjectionScratchLayout};
use cuteafd_loader::plan::checkpoint::Checkpoint;

pub(crate) use shared::{Hy4NativeCore, Hy4NativeScratch};

pub(crate) fn load_native_core<'a>(
    device: Device<'a>,
    programs: &'a Programs<'a>,
    checkpoint: &Checkpoint,
    layout: &CoordinatorWeights,
    rows: usize,
    arena: &DeviceWeightArena<'a>,
) -> Result<Hy4NativeCore<'a>> {
    Hy4NativeCore::load_with(
        device,
        programs,
        checkpoint,
        &layout.core,
        &layout.fp8,
        fp8_role,
        shared::HY4_FLASH_NATIVE_TAGS,
        rows,
        arena,
    )
}

pub(crate) fn allocate_native_scratch<'a>(
    device: Device<'a>,
    core_layout: Option<&ProjectionScratchLayout>,
    head_layout: Option<&ProjectionScratchLayout>,
) -> Result<Hy4NativeScratch<'a>> {
    Hy4NativeScratch::allocate_with(device, shared::HY4_FLASH_NATIVE_TAGS, core_layout, head_layout)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_kv_b_trellis_absorbed(
    device: Device<'_>,
    programs: &Programs<'_>,
    checkpoint: &Checkpoint,
    proj: &CoreProjection,
    rank: usize,
    gpus: usize,
    local_heads: usize,
    nope: usize,
    v_dim: usize,
    kv_lora: usize,
    destination: Buffer,
) -> Result<()> {
    shared::dequant_kv_b_trellis_absorbed_with(
        device,
        programs,
        checkpoint,
        proj,
        rank,
        gpus,
        local_heads,
        nope,
        v_dim,
        kv_lora,
        shared::HY4_FLASH_NATIVE_TAGS,
        destination,
    )
}
