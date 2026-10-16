//! Shared header validation for checkpoint-native dense coordinator weights.
use crate::plan::checkpoint::{Checkpoint, CheckpointTensor};
use crate::serving_capacity::CacheGeometryError;
use cuteafd_core::DType;

pub(crate) fn tensor<'a>(
    checkpoint: &'a Checkpoint,
    name: &str,
    shape: &[usize],
    dtypes: &[DType],
) -> Result<&'a CheckpointTensor, CacheGeometryError> {
    let at = checkpoint
        .tensors
        .binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| CacheGeometryError::ResidentTensor {
            name: name.into(),
            what: "missing from checkpoint headers".into(),
        })?;
    let t = &checkpoint.tensors[at];
    if t.meta.shape != shape || !dtypes.contains(&t.meta.dtype) {
        return Err(CacheGeometryError::ResidentTensor {
            name: name.into(),
            what: format!(
                "expected {dtypes:?} {shape:?}, found {:?} {:?}",
                t.meta.dtype, t.meta.shape
            ),
        });
    }
    let element_bytes = match t.meta.dtype {
        DType::Bf16 => 2,
        DType::F32 => 4,
        DType::F8E4M3 => 1,
        _ => {
            return Err(CacheGeometryError::ResidentTensor {
                name: name.into(),
                what: "resident size requires BF16, FP32 or E4M3".into(),
            })
        }
    };
    let expected = shape
        .iter()
        .try_fold(element_bytes, |bytes: u64, &dim| {
            bytes.checked_mul(dim as u64)
        })
        .ok_or(CacheGeometryError::Overflow("native source tensor extent"))?;
    if t.meta.byte_length != expected {
        return Err(CacheGeometryError::ResidentTensor {
            name: name.into(),
            what: format!("expected {expected} bytes, found {}", t.meta.byte_length),
        });
    }
    Ok(t)
}

/// Exact source views and conservative replicated resident storage. TP slicing,
/// kernel packing and runtime reservations are separate contracts; this does
/// not authorize serving or infer missing workspace/graph costs as zero.
#[derive(Debug, Clone)]
pub struct NativeDenseLayout {
    pub tensors: Vec<CheckpointTensor>,
    pub resident_bytes: u64,
}

impl NativeDenseLayout {
    pub fn from_headers(
        checkpoint: &Checkpoint,
        declarations: impl IntoIterator<Item = (String, Vec<usize>, DType)>,
    ) -> Result<Self, CacheGeometryError> {
        let mut tensors = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        let mut resident_bytes = 0u64;
        for (name, shape, dtype) in declarations {
            if !names.insert(name.clone()) || shape.is_empty() || shape.contains(&0) {
                return Err(CacheGeometryError::ResidentTensor {
                    name,
                    what: "duplicate or empty native declaration".into(),
                });
            }
            let source = tensor(checkpoint, &name, &shape, &[dtype])?;
            source
                .meta
                .byte_offset
                .checked_add(source.meta.byte_length)
                .ok_or(CacheGeometryError::Overflow("native source offset"))?;
            resident_bytes = resident_bytes
                .checked_add(source.meta.byte_length.max(256))
                .ok_or(CacheGeometryError::Overflow("native resident weights"))?;
            tensors.push(source.clone());
        }
        if tensors.is_empty() {
            return Err(CacheGeometryError::ResidentTensor {
                name: "native dense layout".into(),
                what: "empty declaration".into(),
            });
        }
        Ok(Self {
            tensors,
            resident_bytes,
        })
    }

    /// Weight-only fit with explicit reservations supplied by the caller.
    /// Serial host source staging is not a second device weight copy.
    pub fn check_weight_budget(
        &self,
        ceiling: u64,
        reservations: &[u64],
    ) -> Result<u64, CacheGeometryError> {
        let required = reservations
            .iter()
            .try_fold(self.resident_bytes, |sum, bytes| sum.checked_add(*bytes))
            .ok_or(CacheGeometryError::Overflow(
                "native weights and reservations",
            ))?;
        if required > ceiling {
            return Err(CacheGeometryError::ResidentTensor {
                name: "native dense layout".into(),
                what: format!(
                    "weights plus supplied reservations need {required} bytes, ceiling {ceiling}"
                ),
            });
        }
        Ok(required)
    }
}
