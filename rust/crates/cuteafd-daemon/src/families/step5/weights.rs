//! Header-admitted target weights. No routed-expert or MTP residency is implied.
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::memory::weight_arena::{DeviceWeightArena, weight_allocation};
use crate::shared::peer_split::{slice_2d, Axis};
#[cfg(test)]
use crate::shared::qwen::weights::materialize_bf16;
use crate::shared::token_io::{EmbedPlacement, EmbedSource, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::plan::families::step5::coordinator::{
    RankWeightLayout, WeightOperand, WeightSlice,
};
use std::collections::BTreeMap;
use cuteafd_core::memory_layout::ArenaTag;

const EMBEDDING: &str = "model.embed_tokens.weight";

/// One rank's target arena, planned and served by the same loader function.
pub(super) fn target_arena(layout: &RankWeightLayout, rope: Option<&cuteafd_loader::plan::rope::Tables>,
    device: i32, count: usize, embedding: bool) -> Result<cuteafd_loader::serving_capacity::WeightArenaLayout> {
    layout.target_arena(rope, device, count, embedding)
}

pub(super) use cuteafd_loader::plan::families::step5::coordinator::ROPE_PREFIX;

/// The host packing the dense records replaced (the unit tests' oracle).
#[cfg(test)]
fn pack(
    operand: &WeightOperand,
    layout: &RankWeightLayout,
    mut read: impl FnMut(&str) -> Result<(Vec<u8>, DType, Vec<usize>)>,
) -> Result<Vec<u8>> {
    let mut packed = Vec::new();
    for source in &operand.sources {
        let (bytes, dtype, shape) = read(&source.meta.name)?;
        ensure!(
            shape == source.meta.shape
                && dtype == source.meta.dtype
                && bytes.len() as u64 == source.meta.byte_length,
            "Step-5 source changed after admission"
        );
        let bytes = if operand.dtype == DType::Bf16 {
            materialize_bf16(&bytes, dtype)?
        } else {
            ensure!(
                dtype == operand.dtype,
                "Step-5 native parameter precision changed"
            );
            bytes
        };
        if operand.slice == WeightSlice::Replicated {
            packed.extend_from_slice(&bytes);
        } else {
            ensure!(
                operand.sources.len() == 1 && shape.len() == 2,
                "Step-5 split source layout"
            );
            let elem = if operand.dtype == DType::Bf16 { 2 } else { 4 };
            let axis = if operand.slice == WeightSlice::Rows {
                Axis::Rows
            } else {
                Axis::Cols
            };
            packed.extend_from_slice(&slice_2d(
                &bytes,
                shape[0],
                shape[1],
                elem,
                axis,
                layout.rank,
                layout.ranks,
            ));
        }
    }
    let elem = if operand.dtype == DType::Bf16 { 2 } else { 4 };
    let expected = operand
        .shape
        .iter()
        .try_fold(elem, |n: usize, dim| n.checked_mul(*dim))
        .context("Step-5 packed size overflow")?;
    ensure!(
        packed.len() == expected && packed.len().max(256) as u64 == operand.allocation_bytes,
        "Step-5 packed storage disagrees with header admission"
    );
    Ok(packed)
}

pub(super) fn validate_live_weights(device: i32, weights: &[Allocation<'_>]) -> Result<()> {
    validate_live_buffers(device, &weights.iter().map(|w| w.buffer).collect::<Vec<_>>())
}

pub(super) fn validate_live_buffers(device: i32, weights: &[CuteafdDeviceBuffer]) -> Result<()> {
    ensure!(!weights.is_empty() && weights.iter().all(|buffer| {
        !buffer.ptr.is_null() && buffer.device_id == device && buffer.bytes >= 256
    }), "Step-5 live native weight allocation inventory");
    Ok(())
}

pub(super) fn load_operand<'a>(
    device: Device<'a>,
    checkpoint: &'a Checkpoint,
    layout: &RankWeightLayout,
    operand: &WeightOperand,
    arena: Option<&DeviceWeightArena<'a>>,
) -> Result<Allocation<'a>> {
    let _scope = cuteafd_ffi::memory_ledger::scope("step5_target_weights");
    cuteafd_ffi::memory_ledger::tensor(&operand.name);
    let _format = cuteafd_ffi::memory_ledger::format(if operand.dtype == DType::Bf16 {
        "bf16"
    } else if operand.dtype == DType::F8E4M3 { "fp8-block128" } else {
        "f32"
    });
    let allocation = weight_allocation(device, arena, &operand.name, usize::try_from(operand.allocation_bytes)?, ArenaTag::weights("step5/target"))?;
    if operand.dtype == DType::F8E4M3 {
        use crate::shared::{native_attention::{attention_records, AttentionRecord}, experts::rtx::dense::DenseLoad};
        let native = operand.native_layout(layout.rank, layout.ranks)?;
        ensure!(native.bytes()? == operand.allocation_bytes, "Step-5 FP8 owner extent");
        let selected = checkpoint.projection(&native.stem)?;
        let (values, scales) = fp8_buffers(allocation.buffer, &operand.shape)?;
        let records = attention_records(&checkpoint.snapshot, &selected, checkpoint.projection_view(&native.stem), &native, cuteafd_loader::formats::native_attention::Fp8Role::exact("projection", true),
            crate::shared::fp8_linear::Fp8Scales::Amax, values, Some(scales))?;
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(cuteafd_loader::shared::dense_read::MountLoad::from_headers(
            selected.tensors.iter().map(|tensor| &tensor.meta), layout.ranks)?)?;
        for record in records {
            device.run(|| match record {
                AttentionRecord::Operand(record) => dense.operand(record),
                AttentionRecord::Step(record) => dense.step(record, std::ptr::null_mut()),
            })?;
        }
    } else {
        load_records(device, checkpoint, layout, operand, allocation.buffer)?;
    }
    Ok(allocation)
}

/// One admitted allocation holds final E4M3 values followed by aligned F32 block scales.
pub(super) fn fp8_buffers(buffer: CuteafdDeviceBuffer, shape: &[usize])
    -> Result<(CuteafdDeviceBuffer, CuteafdDeviceBuffer)> {
    ensure!(shape.len() == 2 && shape.iter().all(|n| *n > 0 && n % 128 == 0), "Step-5 FP8 matrix geometry");
    let values = shape[0].checked_mul(shape[1]).context("Step-5 FP8 values overflow")?.max(256);
    let scales = (shape[0] / 128).checked_mul(shape[1] / 128).and_then(|n| n.checked_mul(4))
        .context("Step-5 FP8 scales overflow")?.max(256);
    ensure!(buffer.bytes == values.checked_add(scales).context("Step-5 FP8 owner overflow")?,
        "Step-5 FP8 actual owner extent");
    Ok((CuteafdDeviceBuffer { bytes: values, ..buffer }, CuteafdDeviceBuffer {
        ptr: buffer.ptr.wrapping_byte_add(values), bytes: scales, ..buffer }))
}

/// `operand`'s bytes as shared dense records into `buffer` (allocated as the
/// mount allocates it): unchanged checkpoint rectangles (whole tensors, a
/// rank's rows, a rank's column window, in source order), or for an F32
/// source of a BF16 operand one staged step narrowed on the GPU. Filled
/// through the mount at once, or recorded for the coordinator push group.
fn load_records(device: Device<'_>, checkpoint: &Checkpoint, layout: &RankWeightLayout, operand: &WeightOperand,
    buffer: CuteafdDeviceBuffer) -> Result<()> {
    use crate::shared::experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, STEP_BYTES};
    let dense = DenseLoad::mount(device.library);
    let elem = if operand.dtype == DType::Bf16 { 2 } else { 4 };
    let mut lands = Vec::new();
    let mut at = 0usize;
    for source in &operand.sources {
        let meta = &source.meta;
        ensure!(meta.dtype == operand.dtype || (operand.dtype == DType::Bf16 && meta.dtype == DType::F32),
            "Step-5 native parameter precision changed");
        let total = if meta.shape.len() >= 2 { meta.shape[0] } else { 1 };
        let bytes = usize::try_from(meta.byte_length)?;
        ensure!(total > 0 && bytes % total == 0, "{}: rows do not divide the tensor", meta.name);
        let whole = Rect { name: meta.name.clone(), path: checkpoint.snapshot.join(&source.shard), base: meta.byte_offset,
            row_bytes: bytes / total, rows: 0..total, cols: 0..bytes / total };
        let rect = match operand.slice {
            WeightSlice::Replicated => whole,
            WeightSlice::Rows | WeightSlice::Cols => {
                ensure!(operand.sources.len() == 1 && meta.shape.len() == 2, "Step-5 split source layout");
                let (rank, ranks) = (layout.rank, layout.ranks);
                // Narrowing is elementwise: a source element's width selects the window.
                let elem = if meta.dtype == DType::F32 { 4 } else { elem };
                if operand.slice == WeightSlice::Rows {
                    ensure!(total % ranks == 0, "{}: rows do not split over {ranks} GPUs", meta.name);
                    whole.sub(rank * total / ranks..(rank + 1) * total / ranks)
                } else {
                    let width = meta.shape[1] / ranks * elem;
                    ensure!(meta.shape[1] % ranks == 0, "{}: columns do not split over {ranks} GPUs", meta.name);
                    Rect { cols: rank * width..(rank + 1) * width, ..whole }
                }
            }
        };
        if meta.dtype == operand.dtype {
            let n = rect.bytes();
            lands.push((rect, at));
            at += n;
        } else {
            // Elementwise narrowing permits small windows even for a vector
            // represented as one checkpoint row.
            let limit = crate::shared::experts::rtx::dense::TRANSFORM_BYTES;
            for row in 0..rect.rows.len() {
                for first in (0..rect.width()).step_by(limit) {
                    let n = limit.min(rect.width() - first);
                    ensure!(n % 4 == 0 && n <= STEP_BYTES, "{}: invalid F32 narrowing window", meta.name);
                    let piece = Rect { cols: rect.cols.start + first..rect.cols.start + first + n,
                        ..rect.sub(row..row + 1) };
                    // SAFETY: every narrowed window lies inside the operand.
                    let dst = CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(at) }.cast(),
                        bytes: n / 2, ..buffer };
                    device.run(|| dense.step(Step { device: device.id, rects: vec![piece],
                        jobs: vec![Job::NarrowF32Bf16 { at: 0, bytes: n, dst }] }, std::ptr::null_mut()))?;
                    at += n / 2;
                }
            }
        }
    }
    let expected = operand.shape.iter().try_fold(elem, |n: usize, dim| n.checked_mul(*dim))
        .context("Step-5 packed size overflow")?;
    ensure!(at == expected && expected.max(256) as u64 == operand.allocation_bytes && buffer.bytes >= expected,
        "Step-5 packed storage disagrees with header admission");
    if !lands.is_empty() {
        device.run(|| dense.operand(Operand { device: device.id, buffer, lands, bytes: at }))?;
    }
    Ok(())
}

/// The host packing of `operand` (the records' reference).
#[cfg(test)]
pub(super) fn pack_reference(operand: &WeightOperand, layout: &RankWeightLayout,
    read: impl FnMut(&str) -> Result<(Vec<u8>, DType, Vec<usize>)>) -> Result<Vec<u8>> {
    pack(operand, layout, read)
}

pub(super) struct RankWeights<'a> {
    pub embedding: TokenEmbedding<'a>,
    operands: BTreeMap<String, Allocation<'a>>,
    pub layout: RankWeightLayout,
}

impl<'a> RankWeights<'a> {
    pub fn load(
        device: Device<'a>,
        checkpoint: &'a Checkpoint,
        rank: usize,
        ranks: usize,
        available: u64,
        reservations: &[u64],
    ) -> Result<Self> {
        // No CUDA calls, including device selection, precede these checks.
        let layout = RankWeightLayout::from_checkpoint(checkpoint, rank, ranks)?;
        let packed = target_arena(&layout, None, device.id, 92, true)?;
        layout.check_budget(available, &[reservations, &[packed.padding_bytes() as u64]].concat())?;
        let source = layout
            .operands
            .iter()
            .find(|t| t.name == EMBEDDING)
            .context("Step-5 embedding operand")?
            .sources
            .first()
            .context("Step-5 embedding source")?;
        let embedding = EmbedSource::new(&checkpoint.snapshot, &source.shard, &source.meta, 4096)?;
        let arena = DeviceWeightArena::new(device, packed, ArenaTag::weights("step5/target"))?;
        device.run(|| {
            let (embedding, operands) =
                TokenEmbedding::load_with(device.library, embedding, EmbedPlacement::Gpu, || {
                    let mut operands = BTreeMap::new();
                    for operand in &layout.operands {
                        if operand.name == EMBEDDING {
                            continue;
                        }
                        operands.insert(
                            operand.name.clone(),
                            load_operand(device, checkpoint, &layout, operand, Some(&arena))?,
                        );
                    }
                    Ok(operands)
                }, |name, bytes| weight_allocation(device, Some(&arena), name, bytes, ArenaTag::weights("step5/target")))?;
            Ok(Self {
                embedding,
                operands,
                layout,
            })
        })
    }

    pub fn get(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        Ok(self
            .operands
            .get(name)
            .with_context(|| format!("Step-5 operand {name}"))?
            .buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::{plan::checkpoint::CheckpointTensor, SafetensorsTensorMetadata};

    fn source(name: &str, dtype: DType, shape: Vec<usize>) -> CheckpointTensor {
        let size = shape.iter().product::<usize>() * if dtype == DType::Bf16 { 2 } else { 4 };
        CheckpointTensor {
            shard: "unused".into(),
            meta: SafetensorsTensorMetadata {
                name: name.into(),
                dtype,
                shape,
                byte_offset: 0,
                byte_length: size as u64,
            },
        }
    }

    fn layout(rank: usize) -> RankWeightLayout {
        RankWeightLayout {
            rank,
            ranks: 2,
            operands: vec![],
            core: Default::default(),
            resident_bytes: 0,
        }
    }

    #[test]
    fn fp8_final_owner_has_disjoint_exact_values_and_scales() -> Result<()> {
        for shape in [[4096, 12288], [4096, 6144], [128, 128]] {
            let values = shape[0] * shape[1];
            let scales = (shape[0] / 128 * (shape[1] / 128) * 4).max(256);
            let owner = CuteafdDeviceBuffer { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr().cast(),
                bytes: values + scales, device_id: 1, ..Default::default() };
            let (q, grid) = fp8_buffers(owner, &shape)?;
            assert_eq!((q.ptr, q.bytes, q.device_id), (owner.ptr, values, 1));
            assert_eq!((grid.ptr, grid.bytes, grid.device_id), (owner.ptr.wrapping_byte_add(values), scales, 1));
            for bytes in [owner.bytes - 1, owner.bytes + 1] {
                assert!(fp8_buffers(CuteafdDeviceBuffer { bytes, ..owner }, &shape).is_err());
            }
        }
        assert!(fp8_buffers(CuteafdDeviceBuffer::default(), &[4096, 6143]).is_err());
        Ok(())
    }

    #[test]
    fn fp8_conversion_records_keep_selected_tp_and_mtp_source_rectangles() -> Result<()> {
        use crate::shared::{experts::rtx::dense::Job, native_attention::{attention_records, AttentionRecord}};
        use cuteafd_loader::plan::checkpoint::{CoreView, ProjectionStorage, SelectedProjection};
        for (layer, ranks, rank) in [(0, 1, 0), (0, 2, 0), (0, 2, 1), (92, 1, 0)] {
            let name = format!("model.layers.{layer}.self_attn.o_proj.weight");
            let src = source(&name, DType::Bf16, vec![4096, 12288]);
            let selected = SelectedProjection { storage: ProjectionStorage::Native {
                dtype: DType::Bf16, shape: src.meta.shape.clone() },
                source_bytes: src.meta.byte_length, tensors: vec![src.clone()] };
            let operand = WeightOperand { name, sources: vec![src], dtype: DType::F8E4M3,
                shape: vec![4096, 12288 / ranks],
                slice: if ranks == 1 { WeightSlice::Replicated } else { WeightSlice::Cols }, allocation_bytes: 0 };
            let layout = operand.native_layout(rank, ranks)?;
            let owner = CuteafdDeviceBuffer { bytes: layout.bytes()? as usize, ..Default::default() };
            let (q, scales) = fp8_buffers(owner, &operand.shape)?;
            let records = attention_records(std::path::Path::new("/selected"), &selected,
                CoreView::Fp8Compact, &layout, cuteafd_loader::formats::native_attention::Fp8Role::exact("projection", true), crate::shared::fp8_linear::Fp8Scales::Amax, q, Some(scales))?;
            let mut rows = 0;
            for record in records {
                let AttentionRecord::Step(step) = record else { panic!("BF16 conversion must be staged, not resident"); };
                assert_eq!(step.rects.len(), 1);
                let rect = &step.rects[0];
                assert_eq!(rect.name, operand.name);
                assert_eq!(rect.row_bytes, 12288 * 2);
                assert_eq!(rect.cols, rank * 12288 / ranks * 2..(rank + 1) * 12288 / ranks * 2);
                assert_eq!(rect.rows.start, rows);
                assert_eq!(rect.rows.len() % 128, 0);
                let Job::Quant { rows: window, pitch, cols, target, .. } = &step.jobs[0]
                    else { panic!("FP8-compact must quantize once"); };
                assert_eq!(*window, 0..rect.rows.len());
                assert_eq!((*pitch, *cols, target.q_stride), (12288 / ranks, 12288 / ranks, 12288 / ranks));
                assert_eq!(target.q.bytes, rect.rows.len() * 12288 / ranks);
                assert_eq!(target.scales.bytes, rect.rows.len() / 128 * (12288 / ranks / 128) * 4);
                rows += rect.rows.len();
            }
            assert_eq!(rows, 4096);
        }
        Ok(())
    }

    #[test]
    fn packing_matches_reference_indexer_boundary_and_attention_partition() -> Result<()> {
        let values = [
            1.00390625f32,
            1.01171875,
            -1.00390625,
            -1.01171875,
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            2.0,
        ];
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = source("index_w", DType::F32, vec![4, 2]);
        for rank in 0..2 {
            for slice in [WeightSlice::Rows, WeightSlice::Cols] {
                let shape = if slice == WeightSlice::Rows {
                    vec![2, 2]
                } else {
                    vec![4, 1]
                };
                let operand = WeightOperand {
                    name: "index_w".into(),
                    sources: vec![tensor.clone()],
                    dtype: DType::Bf16,
                    shape,
                    slice,
                    allocation_bytes: 256,
                };
                let actual = pack(&operand, &layout(rank), |_| {
                    Ok((bytes.clone(), DType::F32, vec![4, 2]))
                })?;
                let selected: Vec<_> = if slice == WeightSlice::Rows {
                    values[rank * 4..rank * 4 + 4].to_vec()
                } else {
                    (0..4).map(|row| values[row * 2 + rank]).collect()
                };
                let expected: Vec<_> = selected
                    .iter()
                    .flat_map(|v| {
                        let rounded =
                            cuteafd_core::dense_attention::OutputPrecision::Bf16.round(*v);
                        ((rounded.to_bits() >> 16) as u16).to_le_bytes()
                    })
                    .collect();
                assert_eq!(actual, expected);
            }
        }
        let mut operand = WeightOperand {
            name: "norm".into(),
            sources: vec![tensor],
            dtype: DType::F32,
            shape: vec![4, 2],
            slice: WeightSlice::Replicated,
            allocation_bytes: 256,
        };
        assert_eq!(
            pack(&operand, &layout(0), |_| Ok((
                bytes.clone(),
                DType::F32,
                vec![4, 2]
            )))?,
            bytes
        );
        assert!(pack(&operand, &layout(0), |_| Ok((
            vec![0; 4],
            DType::F32,
            vec![4, 2]
        )))
        .is_err());
        operand.allocation_bytes = 257;
        assert!(pack(&operand, &layout(0), |_| Ok((
            bytes.clone(),
            DType::F32,
            vec![4, 2]
        )))
        .is_err());
        Ok(())
    }

    #[test]
    fn replicated_router_keeps_native_bf16_bytes_on_both_ranks() -> Result<()> {
        let name = "model.layers.3.moe.gate.weight";
        let shape = vec![352, 4096];
        let bytes: Vec<u8> = (0..352 * 4096)
            .flat_map(|i| (0x3f00u16 + (i % 128) as u16).to_le_bytes())
            .collect();
        let operand = WeightOperand {
            name: name.into(),
            sources: vec![source(name, DType::Bf16, shape.clone())],
            dtype: DType::Bf16,
            shape: shape.clone(),
            slice: WeightSlice::Replicated,
            allocation_bytes: bytes.len() as u64,
        };
        for rank in 0..2 {
            let actual = pack(&operand, &layout(rank), |_| {
                Ok((bytes.clone(), DType::Bf16, shape.clone()))
            })?;
            assert_eq!(actual, bytes);
        }
        Ok(())
    }

    #[test]
    fn packed_ffn_keeps_gate_then_up_without_a_second_device_copy() -> Result<()> {
        let operand = WeightOperand {
            name: "gate_up".into(),
            sources: vec![
                source("gate", DType::Bf16, vec![2, 2]),
                source("up", DType::Bf16, vec![2, 2]),
            ],
            dtype: DType::Bf16,
            shape: vec![4, 2],
            slice: WeightSlice::Replicated,
            allocation_bytes: 256,
        };
        let actual = pack(&operand, &layout(0), |name| {
            Ok((
                vec![if name == "gate" { 0x19 } else { 0x29 }; 8],
                DType::Bf16,
                vec![2, 2],
            ))
        })?;
        assert_eq!(actual, [vec![0x19; 8], vec![0x29; 8]].concat());
        Ok(())
    }
}
