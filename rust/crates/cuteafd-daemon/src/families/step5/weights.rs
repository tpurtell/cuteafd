//! Header-admitted target weights. No routed-expert or MTP residency is implied.
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::peer_split::{slice_2d, Axis};
use crate::shared::qwen::weights::{materialize_bf16, WeightReader};
use crate::shared::token_io::{EmbedPlacement, EmbedSource, TokenEmbedding};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::plan::families::step5::coordinator::{
    RankWeightLayout, WeightOperand, WeightSlice,
};
use std::collections::BTreeMap;

const EMBEDDING: &str = "model.embed_tokens.weight";

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
    ensure!(!weights.is_empty() && weights.iter().all(|weight| {
        let buffer = weight.buffer;
        !buffer.ptr.is_null() && buffer.device_id == device && buffer.bytes >= 256
    }), "Step-5 live native weight allocation inventory");
    Ok(())
}

pub(super) fn load_operand<'a>(
    device: Device<'a>,
    checkpoint: &'a Checkpoint,
    layout: &RankWeightLayout,
    operand: &WeightOperand,
) -> Result<Allocation<'a>> {
    let _scope = cuteafd_ffi::memory_ledger::scope("step5_target_weights");
    cuteafd_ffi::memory_ledger::tensor(&operand.name);
    let _format = cuteafd_ffi::memory_ledger::format(if operand.dtype == DType::Bf16 {
        "bf16"
    } else {
        "f32"
    });
    let reader = WeightReader {
        library: device.library,
        checkpoint,
    };
    let bytes = pack(operand, layout, |name| reader.raw(name))?;
    device.run(|| reader.upload(&bytes))
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
        layout.check_budget(available, reservations)?;
        let source = layout
            .operands
            .iter()
            .find(|t| t.name == EMBEDDING)
            .context("Step-5 embedding operand")?
            .sources
            .first()
            .context("Step-5 embedding source")?;
        let embedding = EmbedSource::new(&checkpoint.snapshot, &source.shard, &source.meta, 4096)?;
        device.run(|| {
            let (embedding, operands) =
                TokenEmbedding::load(device.library, embedding, EmbedPlacement::Gpu, || {
                    let mut operands = BTreeMap::new();
                    for operand in &layout.operands {
                        if operand.name == EMBEDDING {
                            continue;
                        }
                        operands.insert(
                            operand.name.clone(),
                            load_operand(device, checkpoint, &layout, operand)?,
                        );
                    }
                    Ok(operands)
                })?;
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
            resident_bytes: 0,
        }
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
