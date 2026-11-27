//! Native compact-core projections shared by `hy4` and `hy4_flash`
//! (`k8-compact`, `k6-compact`, `fp8-compact`).
//!
//! Each replaces one BF16 `(x, w, out)` projection launch with the shared
//! `dense_trellis` MUL1 (K6/K8) or block-128 `Fp8Projections` program, backed
//! directly by views inside the rank's packed `DeviceWeightArena`.
use crate::shared::dense_trellis::{
    codebook, CoreWeightLayout, CoreWeights, Projection, ProjectionLayout, ProjectionScratch,
};
use crate::shared::experts::rtx::dense::DenseLoad;
use crate::shared::fp8_linear::Fp8Scales;
use crate::shared::memory::device::{Allocation, Device, Stream};
use crate::shared::memory::receipts::Inventory;
use crate::shared::memory::weight_arena::{weight_allocation, DeviceWeightArena};
use crate::shared::memory::Tag;
use crate::shared::native_attention::{attention_records, AttentionRecord};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_ffi::dense_trellis::TrellisDType;
use cuteafd_ffi::draft_projection::ProjectionOutput;
use cuteafd_ffi::native_attention::Fp8Projections;
use cuteafd_ffi::programs::Programs;
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, NativeLibrary};
use cuteafd_loader::families::hy4_common::HEAD_STEM;
use cuteafd_loader::formats::dense_trellis::TrellisSide;
use cuteafd_loader::formats::native_attention::{Fp8Role, NativeLayout};
use cuteafd_loader::formats::trellis_core::{
    CoreProjection, CoreSplit, CoreWeightLayout as SharedCoreWeightLayout, ProjectionScratchLayout,
};
use cuteafd_loader::plan::checkpoint::Checkpoint;
use std::collections::BTreeMap;
use std::ffi::c_void;

/// Family-specific allocation/inventory tags for shared native compact-core owners.
#[derive(Clone, Copy)]
pub(crate) struct Hy4NativeTags {
    pub(crate) family: &'static str,
    pub(crate) target_arena: &'static str,
    pub(crate) core_scratch: &'static str,
    pub(crate) head_scratch: &'static str,
    pub(crate) transient_kv_b: &'static str,
    pub(crate) transient_eye: &'static str,
    pub(crate) transient_kv_b_out: &'static str,
}

pub(crate) const HY4_FLASH_NATIVE_TAGS: Hy4NativeTags = Hy4NativeTags {
    family: "hy4_flash",
    target_arena: "hy4_flash/target",
    core_scratch: "hy4_flash/native-core-scratch",
    head_scratch: "hy4_flash/native-head-scratch",
    transient_kv_b: "hy4_flash/transient-kv-b",
    transient_eye: "hy4_flash/transient-eye",
    transient_kv_b_out: "hy4_flash/transient-kv-b-out",
};

pub(crate) const HY4_NATIVE_TAGS: Hy4NativeTags = Hy4NativeTags {
    family: "hy4",
    target_arena: "hy4/target",
    core_scratch: "hy4/native-core-scratch",
    head_scratch: "hy4/native-head-scratch",
    transient_kv_b: "hy4/transient-kv-b",
    transient_eye: "hy4/transient-eye",
    transient_kv_b_out: "hy4/transient-kv-b-out",
};

/// One resident compact-core projection (EXL3 K6/K8 MUL1 or block-128 FP8) and its resolved program.
pub(crate) enum Hy4NativeLinear<'a> {
    Trellis {
        weights: CoreWeights<'a>,
        projection: Projection<'a>,
    },
    Fp8 {
        payload: Allocation<'a>,
        scales: Allocation<'a>,
        layout: NativeLayout,
        projection: Fp8Projections<'a>,
        library: &'a NativeLibrary,
    },
}

impl<'a> Hy4NativeLinear<'a> {
    pub(crate) fn shape(&self) -> [usize; 2] {
        match self {
            Self::Trellis { weights, .. } => weights.shape,
            Self::Fp8 { layout, .. } => layout.shape,
        }
    }

    pub(crate) fn side(&self) -> TrellisSide {
        match self {
            Self::Trellis { weights, .. } => weights.side,
            Self::Fp8 { layout, .. } => {
                if layout.ranks > 1 && layout.split == CoreSplit::Input {
                    TrellisSide::Input
                } else {
                    TrellisSide::Output
                }
            }
        }
    }

    fn fp8_launch(
        &self,
        rows: usize,
        input: Buffer,
        output: Buffer,
        scratch: &ProjectionScratch<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        let Self::Fp8 {
            payload,
            scales,
            layout,
            projection,
            ..
        } = self
        else {
            bail!("not a Hy4 native FP8 linear");
        };
        let [n, k] = layout.shape;
        let fp8_ws = *scratch
            .buffers
            .get("fp8")
            .context("Hy4 FP8 native scratch missing 'fp8' plane")?;
        // SAFETY: caller retains admitted input/output buffers, arena weights, and lane scratch through drain.
        unsafe {
            projection.launch(input.ptr, payload.buffer.ptr, scales.buffer.ptr, output, fp8_ws, rows, k, n, stream)?;
        }
        Ok(())
    }

    /// Launch a whole or output-split (`side == Output`) native projection (BF16 or F32 for `lm_head`).
    pub(crate) fn launch(
        &self,
        rows: usize,
        input: Buffer,
        output: Buffer,
        scratch: &ProjectionScratch<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            self.side() == TrellisSide::Output,
            "K-split Hy4 native projection requires partial + TP2 reduction + egress"
        );
        match self {
            Self::Trellis { weights, projection } => projection.local(rows, input, weights, scratch, output, stream),
            Self::Fp8 { .. } => self.fp8_launch(rows, input, output, scratch, stream),
        }
    }

    /// Launch the F32 partial core of a K-split TP2 projection before peer reduction.
    pub(crate) fn partial(
        &self,
        rows: usize,
        input: Buffer,
        output_f32: Buffer,
        scratch: &ProjectionScratch<'_>,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            self.side() == TrellisSide::Input && output_f32.bytes >= rows * self.shape()[0] * 4,
            "Hy4 K-split native partial geometry mismatch"
        );
        match self {
            Self::Trellis { weights, projection } => {
                projection.core_into(rows, input, weights, scratch, output_f32, stream)
            }
            Self::Fp8 { .. } => self.fp8_launch(rows, input, output_f32, scratch, stream),
        }
    }

    /// Apply `svh` + narrow F32 -> BF16 (Trellis) or narrow F32 -> BF16 (FP8) after TP2 F32 sum.
    pub(crate) fn egress(
        &self,
        rows: usize,
        sum_f32: Buffer,
        output_bf16: Buffer,
        stream: *mut c_void,
    ) -> Result<()> {
        match self {
            Self::Trellis { weights, projection } => {
                projection.program.output(rows, sum_f32, weights.svh.buffer, output_bf16, stream)?;
                Ok(())
            }
            Self::Fp8 { library, .. } => {
                // SAFETY: caller orders TP2 reduction before narrowing disjoint admitted buffers.
                unsafe { library.cuda_f32_to_bf16_async(sum_f32, output_bf16, rows * self.shape()[0], stream) }
            }
        }
    }

    fn scratch_layout(&self, rows: usize) -> Result<ProjectionScratchLayout> {
        match self {
            Self::Trellis { weights, projection } => ProjectionScratchLayout::new_with_fused(
                weights.shape[1], weights.shape[0], rows, projection.program.has_fused(),
            ),
            Self::Fp8 { .. } => Ok(ProjectionScratchLayout {
                rows,
                extents: BTreeMap::from([("fp8", 256)]),
            }),
        }
    }
}

/// All compact-core projections owned by one coordinator rank, keyed by checkpoint stem (no `.weight`).
#[derive(Default)]
pub(crate) struct Hy4NativeCore<'a> {
    linears: BTreeMap<String, Hy4NativeLinear<'a>>,
}

impl<'a> Hy4NativeCore<'a> {
    pub(crate) fn get(&self, stem: &str) -> Option<&Hy4NativeLinear<'a>> {
        self.linears.get(stem)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.linears.is_empty()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_with(
        device: Device<'a>,
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        core_map: &BTreeMap<String, SharedCoreWeightLayout>,
        fp8_map: &BTreeMap<String, NativeLayout>,
        fp8_role: fn(&str) -> Fp8Role,
        tags: Hy4NativeTags,
        rows: usize,
        arena: &DeviceWeightArena<'a>,
    ) -> Result<Self> {
        let mut linears = BTreeMap::new();
        let dense = DenseLoad::mount(device.library);

        for (stem, core) in core_map {
            let core_layout = CoreWeightLayout::from_shared(core.clone());
            let weights = device.run(|| {
                core_layout.load_recorded_with(device, checkpoint, &dense, std::ptr::null_mut(), |name, bytes| {
                    weight_allocation(device, Some(arena), name, bytes, ArenaTag::weights(tags.target_arena))
                })
            })?;
            let side = if core_layout.side == TrellisSide::Input { 2 } else { 1 };
            let output = if stem == HEAD_STEM {
                TrellisDType::F32
            } else {
                TrellisDType::Bf16
            };
            let proj_rows = if stem == HEAD_STEM { rows.min(64) } else { rows };
            let projection = device.run(|| {
                Projection::load_geometry_bits(
                    programs,
                    core_layout.shape[1],
                    core_layout.shape[0],
                    side,
                    core_layout.bits,
                    codebook(&weights),
                    proj_rows,
                    output,
                )
            })?;
            linears.insert(stem.clone(), Hy4NativeLinear::Trellis { weights, projection });
        }

        for (stem, native) in fp8_map {
            let [stored_n, stored_k] = native.shape;
            let output = if stem == HEAD_STEM || (native.ranks > 1 && native.split == CoreSplit::Input) {
                ProjectionOutput::F32
            } else {
                ProjectionOutput::Bf16
            };
            let proj_rows = if stem == HEAD_STEM { rows.min(64) } else { rows };
            let projection = device.run(|| {
                Ok(Fp8Projections::load(programs, &[(stored_k, stored_n)], proj_rows, output)?)
            })?;
            ensure!(
                projection.scratch_bytes(stored_k, stored_n)? <= 256,
                "{} FP8 scratch for {stem} exceeds 256 B admission",
                tags.family
            );
            let payload_bytes = stored_n * stored_k;
            let scales_bytes = ((stored_n / 128) * (stored_k / 128) * 4).max(256);
            let payload = arena.view(&format!("{stem}.payload"), payload_bytes)?;
            let scales = arena.view(&format!("{stem}.scales"), scales_bytes)?;
            let selected = checkpoint.projection(stem)?;
            let records = attention_records(
                &checkpoint.snapshot,
                &selected,
                checkpoint.projection_view(stem),
                native,
                fp8_role(&format!("{stem}.weight")),
                Fp8Scales::Amax,
                payload.buffer,
                Some(scales.buffer),
            )?;
            let fp8_dense = DenseLoad::mount(device.library);
            fp8_dense.admit_mount(cuteafd_loader::shared::dense_read::MountLoad::from_headers(
                selected.tensors.iter().map(|t| &t.meta),
                native.ranks,
            )?)?;
            for record in records {
                device.run(|| match record {
                    AttentionRecord::Operand(op) => fp8_dense.operand(op),
                    AttentionRecord::Step(step) => fp8_dense.step(step, std::ptr::null_mut()),
                })?;
            }
            linears.insert(
                stem.clone(),
                Hy4NativeLinear::Fp8 {
                    payload,
                    scales,
                    layout: native.clone(),
                    projection,
                    library: device.library,
                },
            );
        }

        Ok(Self { linears })
    }

    pub(crate) fn scratch_layout(&self, rows: usize) -> Result<Option<ProjectionScratchLayout>> {
        let mut merged: Option<ProjectionScratchLayout> = None;
        for (stem, linear) in &self.linears {
            if stem == HEAD_STEM {
                continue;
            }
            let next = linear.scratch_layout(rows)?;
            match &mut merged {
                Some(acc) => acc.merge(&next)?,
                None => merged = Some(next),
            }
        }
        Ok(merged)
    }

    pub(crate) fn head_scratch_layout(&self, head_rows: usize) -> Result<Option<ProjectionScratchLayout>> {
        self.linears
            .get(HEAD_STEM)
            .map(|linear| linear.scratch_layout(head_rows))
            .transpose()
    }

    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>, rank: usize) -> Result<()> {
        for (stem, linear) in &self.linears {
            match linear {
                Hy4NativeLinear::Trellis { weights, .. } => {
                    weights.payload.inventory(inventory, &format!("rank{rank}.{stem}.trellis"))?;
                    weights.suh.inventory(inventory, &format!("rank{rank}.{stem}.suh"))?;
                    weights.svh.inventory(inventory, &format!("rank{rank}.{stem}.svh"))?;
                }
                Hy4NativeLinear::Fp8 { payload, scales, .. } => {
                    payload.inventory(inventory, &format!("rank{rank}.{stem}.payload"))?;
                    scales.inventory(inventory, &format!("rank{rank}.{stem}.scales"))?;
                }
            }
        }
        Ok(())
    }
}

/// Per-rank serialized scratch shared by all compact-core native projections (and tiled `lm_head`).
pub(crate) struct Hy4NativeScratch<'a> {
    pub(crate) tags: Hy4NativeTags,
    pub(crate) scratch: Option<ProjectionScratch<'a>>,
    pub(crate) head_scratch: Option<ProjectionScratch<'a>>,
}

impl<'a> Hy4NativeScratch<'a> {
    pub(crate) fn allocate_with(
        device: Device<'a>,
        tags: Hy4NativeTags,
        core_layout: Option<&ProjectionScratchLayout>,
        head_layout: Option<&ProjectionScratchLayout>,
    ) -> Result<Self> {
        let alloc = |layout: &ProjectionScratchLayout, tag: &'static str| -> Result<ProjectionScratch<'a>> {
            let owned = ProjectionLayout {
                extents: layout.extents.clone(),
                rows: layout.rows,
            };
            owned.allocate_packed(device, Tag::workspace(tag))
        };
        Ok(Self {
            tags,
            scratch: core_layout.map(|l| alloc(l, tags.core_scratch)).transpose()?,
            head_scratch: head_layout.map(|l| alloc(l, tags.head_scratch)).transpose()?,
        })
    }

    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>, gpu: usize) -> Result<()> {
        let family = self.tags.family;
        if let Some(scratch) = &self.scratch {
            scratch.allocation_receipts(inventory, &format!("{family}.target.{gpu}.native_scratch"))?;
        }
        if let Some(head_scratch) = &self.head_scratch {
            head_scratch.allocation_receipts(inventory, &format!("{family}.target.{gpu}.native_head_scratch"))?;
        }
        Ok(())
    }
}

/// Dequantize an EXL3 `kv_b_proj` (`[local_heads * (nope + v_dim), kv_lora]`) via transient GPU
/// storage into the final absorbed BF16 `[w_uk, w_uv]` views of the target arena.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_kv_b_trellis_absorbed_with(
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
    tags: Hy4NativeTags,
    destination: Buffer,
) -> Result<()> {
    let core_layout = CoreWeightLayout::new(proj, TrellisSide::Output, rank, gpus)?;
    let [local_n, local_k] = core_layout.shape;
    ensure!(
        local_k == kv_lora && local_n == local_heads * (nope + v_dim),
        "{}: EXL3 kv_b_proj shard [{local_n}, {local_k}] != expected [{}, {kv_lora}]",
        proj.stem,
        local_heads * (nope + v_dim)
    );
    let dense = DenseLoad::mount(device.library);
    let weights = device.run(|| {
        core_layout.load_recorded_with(device, checkpoint, &dense, std::ptr::null_mut(), |name, bytes| {
            weight_allocation(device, None, name, bytes, ArenaTag::weights(tags.transient_kv_b))
        })
    })?;
    let projection = device.run(|| {
        Projection::load_geometry_bits(
            programs,
            local_k,
            local_n,
            1,
            core_layout.bits,
            codebook(&weights),
            local_k,
            TrellisDType::Bf16,
        )
    })?;
    let scratch_layout = ProjectionScratchLayout::new(local_k, local_n, local_k)?;
    let scratch = Hy4NativeScratch::allocate_with(device, tags, Some(&scratch_layout), None)?;

    // Upload [kv_lora, kv_lora] BF16 identity matrix I; local(I) produces Y = I @ W^T = W^T [kv_lora, local_n].
    let one_bf16: u16 = 0x3F80;
    let mut eye_bytes = vec![0u8; local_k * local_k * 2];
    for i in 0..local_k {
        let off = (i * local_k + i) * 2;
        eye_bytes[off..off + 2].copy_from_slice(&one_bf16.to_le_bytes());
    }
    let eye = Allocation::new(device, eye_bytes.len(), Tag::workspace(tags.transient_eye))?;
    let out = Allocation::new(device, local_k * local_n * 2, Tag::workspace(tags.transient_kv_b_out))?;
    let stream = Stream::new(device)?;
    let uk_bytes = local_heads * kv_lora * nope * 2;
    ensure!(
        destination.device_id == device.id && destination.bytes == local_k * local_n * 2,
        "{}: absorbed KV-B arena extent mismatch",
        proj.stem
    );
    let submitted = device.run(|| {
        device.library.copy_h2d(eye.buffer, &eye_bytes)?;
        projection.local(local_k, eye.buffer, &weights, scratch.scratch.as_ref().unwrap(), out.buffer, stream.raw)?;
        // SAFETY: W^T is produced on this stream; the disjoint UK/UV views
        // exhaust the validated destination and the stream drains before return.
        unsafe {
            device.library.load_kvb_absorb_transposed(
                out.buffer.ptr,
                destination.ptr,
                destination.ptr.wrapping_byte_add(uk_bytes),
                local_heads,
                nope,
                v_dim,
                kv_lora,
                stream.raw,
            )
        }
    });
    stream.drain_or_abort("Hy4 KV-B dequant/absorb");
    submitted
}
