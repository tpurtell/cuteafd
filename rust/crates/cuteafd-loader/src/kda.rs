//! Checkpoint-name-independent KDA operand packing, shared by hybrid families.
use cuteafd_core::DType;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KdaOutputGate { LowRank, FullRank }

#[derive(Clone, Debug)]
pub struct KdaGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub output_gate: KdaOutputGate,
    pub o_norm_dtype: DType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KdaRole { Q, K, V, Fa, Fb, Ga, Gb, G, Beta, ConvQ, ConvK, ConvV, ALog, DtBias, ONorm, O }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KdaOperand { WIn, WFg, ConvW, ALog, DtBias, ONorm, WO }

impl KdaOperand {
    pub fn name(self) -> &'static str {
        match self {
            Self::WIn => "w_in", Self::WFg => "w_fg", Self::ConvW => "conv_w", Self::ALog => "a_log",
            Self::DtBias => "dt_bias", Self::ONorm => "o_norm", Self::WO => "w_o",
        }
    }
}

#[derive(Clone, Debug)]
pub struct KdaTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum KdaError {
    #[error("invalid KDA geometry: {0}")]
    Geometry(String),
    #[error("KDA {role:?}: {detail}")]
    Tensor { role: KdaRole, detail: String },
    #[error("KDA tensor source: {0}")]
    Source(String),
}

fn tensor_error(role: KdaRole, detail: impl Into<String>) -> KdaError {
    KdaError::Tensor { role, detail: detail.into() }
}

/// Retain the active heads only after checking every padded F32 word is +0.
/// In particular, -0, a NaN and a finite nonzero tail are all rejected.
pub fn narrow_a_log(raw: &[u8], heads: usize) -> Result<Vec<f32>, KdaError> {
    if heads == 0 || raw.len() % 4 != 0 || raw.len() / 4 < heads {
        return Err(tensor_error(KdaRole::ALog, "expected F32 bytes with at least one entry per active head"));
    }
    let words: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(index) = words[heads..].iter().position(|word| *word != 0) {
        return Err(tensor_error(KdaRole::ALog, format!("padding entry {} is {:#010x}, not +0.0", heads + index, words[heads + index])));
    }
    let live: Vec<f32> = words[..heads].iter().map(|word| f32::from_bits(*word)).collect();
    if live.iter().any(|value| !value.is_finite()) {
        return Err(tensor_error(KdaRole::ALog, "live heads must be finite"));
    }
    Ok(live)
}

#[derive(Clone, Copy)]
enum Split { Rows, Cols, Replicate }

impl KdaGeometry {
    pub fn validate(&self, ranks: usize) -> Result<(), KdaError> {
        if self.hidden == 0 || self.hidden % 128 != 0 || self.head_dim != 128 || self.conv_kernel != 4
            || ranks == 0 || self.heads == 0 || self.heads % ranks != 0 || (self.heads / ranks) % 8 != 0
            || !matches!(self.o_norm_dtype, DType::Bf16 | DType::F32)
            || self.heads.checked_mul(self.head_dim).and_then(|d| d.checked_mul(self.hidden)).is_none()
        {
            return Err(KdaError::Geometry(format!("{self:?}, ranks={ranks}")));
        }
        Ok(())
    }

    pub fn roles(&self, operand: KdaOperand) -> Vec<KdaRole> {
        use KdaOperand as Op;
        use KdaRole as R;
        let full = self.output_gate == KdaOutputGate::FullRank;
        match operand {
            Op::WIn if full => vec![R::Q, R::K, R::V, R::Fa, R::G, R::Beta],
            Op::WIn => vec![R::Q, R::K, R::V, R::Fa, R::Ga, R::Beta],
            Op::WFg if full => vec![R::Fb],
            Op::WFg => vec![R::Fb, R::Gb],
            Op::ConvW => vec![R::ConvQ, R::ConvK, R::ConvV],
            Op::ALog => vec![R::ALog], Op::DtBias => vec![R::DtBias], Op::ONorm => vec![R::ONorm],
            Op::WO => vec![R::O],
        }
    }

    /// Read each tensor role once, preserving BF16/F32 bytes and head order.
    /// Only the operand being packed is materialized, not a second whole layer.
    /// Input projections split rows, o_proj splits columns, low-rank inputs
    /// and the per-head norm replicate. Conv and dt_bias become F32; o_norm
    /// retains the geometry's native dtype. A_log arrives padded and raw.
    pub fn pack(&self, operand: KdaOperand, ranks: usize,
        mut read: impl FnMut(KdaRole) -> Result<KdaTensor, KdaError>) -> Result<Vec<KdaTensor>, KdaError>
    {
        self.validate(ranks)?;
        use KdaOperand as Op;
        use KdaRole as R;
        let full = self.output_gate == KdaOutputGate::FullRank;
        let roles = self.roles(operand);
        let d = self.heads * self.head_dim;
        let local_d = d / ranks;
        let dtype = match operand {
            Op::ConvW | Op::ALog | Op::DtBias => DType::F32,
            Op::ONorm => self.o_norm_dtype.clone(),
            _ => DType::Bf16,
        };
        let mut out = vec![KdaTensor { dtype: dtype.clone(), shape: Vec::new(), bytes: Vec::new() }; ranks];
        for role in roles {
            let mut tensor = read(role)?;
            let element: usize = match tensor.dtype { DType::Bf16 => 2, DType::F32 => 4,
                _ => return Err(tensor_error(role, format!("unsupported dtype {:?}", tensor.dtype))) };
            let expected = match role {
                R::Q | R::K | R::V | R::G => vec![d, self.hidden],
                R::Fa | R::Ga => vec![self.head_dim, self.hidden],
                R::Fb | R::Gb => vec![d, self.head_dim],
                R::Beta => vec![self.heads, self.hidden],
                R::ConvQ | R::ConvK | R::ConvV => {
                    if tensor.shape == [d, 1, self.conv_kernel] { tensor.shape = vec![d, self.conv_kernel]; }
                    vec![d, self.conv_kernel]
                },
                R::ALog => tensor.shape.clone(),
                R::DtBias => vec![d], R::ONorm => vec![self.head_dim], R::O => vec![self.hidden, d],
            };
            let size = tensor.shape.iter().try_fold(element, |n, dim| n.checked_mul(*dim));
            if tensor.shape != expected || size != Some(tensor.bytes.len()) {
                return Err(tensor_error(role, format!("expected {expected:?}, found {:?} ({} bytes)", tensor.shape, tensor.bytes.len())));
            }
            if role == R::ALog {
                if tensor.dtype != DType::F32 || tensor.shape.len() != 1 {
                    return Err(tensor_error(role, "A_log must be a raw F32 vector"));
                }
                tensor.bytes = narrow_a_log(&tensor.bytes, self.heads)?.into_iter().flat_map(f32::to_le_bytes).collect();
                tensor.shape = vec![self.heads];
            }
            if tensor.dtype != dtype {
                if dtype == DType::F32 && tensor.dtype == DType::Bf16 && operand != Op::ONorm {
                    tensor.bytes = tensor.bytes.chunks_exact(2).flat_map(|b|
                        (u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes()).collect();
                } else {
                    return Err(tensor_error(role, format!("expected {dtype:?}, found {:?}", tensor.dtype)));
                }
            }
            let element = if dtype == DType::F32 { 4 } else { 2 };
            let split = match role { R::Fa | R::Ga | R::ONorm => Split::Replicate, R::O => Split::Cols, _ => Split::Rows };
            for (rank, dest) in out.iter_mut().enumerate() {
                match split {
                    Split::Replicate => dest.bytes.extend_from_slice(&tensor.bytes),
                    Split::Rows => {
                        let share = tensor.bytes.len() / ranks;
                        dest.bytes.extend_from_slice(&tensor.bytes[rank * share..(rank + 1) * share]);
                    },
                    Split::Cols => {
                        let row_bytes = d * element;
                        let share = row_bytes / ranks;
                        for row in tensor.bytes.chunks_exact(row_bytes) {
                            dest.bytes.extend_from_slice(&row[rank * share..(rank + 1) * share]);
                        }
                    },
                }
            }
        }
        let shape = match operand {
            Op::WIn => vec![3 * local_d + self.head_dim + if full { local_d } else { self.head_dim } + self.heads / ranks, self.hidden],
            Op::WFg => vec![if full { 1 } else { 2 }, local_d, self.head_dim],
            Op::ConvW => vec![3 * local_d, self.conv_kernel],
            Op::ALog => vec![self.heads / ranks], Op::DtBias => vec![local_d],
            Op::ONorm => vec![self.head_dim], Op::WO => vec![self.hidden, local_d],
        };
        for tensor in &mut out { tensor.shape = shape.clone(); }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(gate: KdaOutputGate) -> KdaGeometry {
        KdaGeometry { hidden: 128, heads: 16, head_dim: 128, conv_kernel: 4, output_gate: gate, o_norm_dtype: DType::F32 }
    }

    #[test]
    fn padding_checked_before_head_split() {
        let mut values: Vec<f32> = (0..96).map(|n| n as f32 / 10.).chain([0.; 32]).collect();
        let raw = |v: &[f32]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let g = KdaGeometry { hidden: 7168, heads: 96, ..geometry(KdaOutputGate::FullRank) };
        let packed = g.pack(KdaOperand::ALog, 2, |_| Ok(KdaTensor { dtype: DType::F32, shape: vec![128], bytes: raw(&values) })).unwrap();
        assert_eq!(packed[1].bytes, raw(&values[48..96]));
        for invalid in [-0., 1., f32::NAN, f32::INFINITY] {
            values[127] = invalid;
            assert!(narrow_a_log(&raw(&values), 96).is_err());
        }
        assert!(narrow_a_log(&raw(&values[..95]), 96).is_err());
    }

    #[test]
    fn low_rank_layout_matches_legacy_concatenate_then_slice() {
        let g = geometry(KdaOutputGate::LowRank);
        let d = g.heads * g.head_dim;
        let mut legacy = vec![Vec::new(); 2];
        let packed = g.pack(KdaOperand::WIn, 2, |role| {
            let rows = match role { KdaRole::Fa | KdaRole::Ga => 128, KdaRole::Beta => 16, _ => d };
            let bytes: Vec<u8> = (0..rows * 128).flat_map(|n| ((n + role as usize * 17) as u16).to_le_bytes()).collect();
            for (rank, dest) in legacy.iter_mut().enumerate() {
                if matches!(role, KdaRole::Fa | KdaRole::Ga) { dest.extend_from_slice(&bytes); }
                else { dest.extend_from_slice(&bytes[rank * bytes.len() / 2..(rank + 1) * bytes.len() / 2]); }
            }
            Ok(KdaTensor { dtype: DType::Bf16, shape: vec![rows, 128], bytes })
        }).unwrap();
        assert_eq!(packed[0].bytes, legacy[0]);
        assert_eq!(packed[1].bytes, legacy[1]);
    }

    #[test]
    fn norm_preserves_f32_bits_and_replicates() {
        let g = geometry(KdaOutputGate::FullRank);
        let bytes: Vec<u8> = (0..128).flat_map(|i| (1.0 + i as f32 / 12345.).to_le_bytes()).collect();
        let packed = g.pack(KdaOperand::ONorm, 2, |_| Ok(KdaTensor { dtype: DType::F32, shape: vec![128], bytes: bytes.clone() })).unwrap();
        assert!(packed.iter().all(|t| t.dtype == DType::F32 && t.bytes == bytes));
        assert!(g.pack(KdaOperand::ONorm, 2, |_| Ok(KdaTensor { dtype: DType::Bf16, shape: vec![128], bytes: vec![0; 256] })).is_err());
    }
}
