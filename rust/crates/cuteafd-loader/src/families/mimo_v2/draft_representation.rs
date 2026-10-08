//! Immutable MiMo DFlash storage and its device-loading contract.
//! The target owns the selected vocabulary-head format; drafters borrow it.
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoDraftRepresentation {
    Bf16Only,
    Fp8Only,

}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimoDraftCapacity {
    pub context_slots: usize,
    pub max_batch_sequences: usize,
    pub block_rows: usize,
}

impl MimoDraftCapacity {
    pub fn new(
        context_slots: usize,
        max_batch_sequences: usize,
        block: usize,
    ) -> Result<Self, MimoDraftStorageError> {
        if context_slots == 0
            || max_batch_sequences == 0
            || max_batch_sequences > context_slots
            || block < 2
        {
            return Err(MimoDraftStorageError::Unsupported(
                "draft batch must fit nonzero context slots and block >= 2",
            ));
        }
        // Native ring destinations use signed32 row indices (-1 means skip).
        if mul(context_slots as u64, 1024)? > i32::MAX as u64 + 1 {
            return Err(MimoDraftStorageError::Unsupported(
                "context slots exceed native ring row indices",
            ));
        }
        Ok(Self {
            context_slots,
            max_batch_sequences,
            block_rows: mul(max_batch_sequences as u64, block as u64)?
                .try_into()
                .map_err(|_| MimoDraftStorageError::Overflow)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MimoDraftGeometry {
    pub hidden: u64,
    pub intermediate: u64,
    pub layers: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub taps: u64,
    pub vocab: u64,
    pub sinks: bool,
}

/// One byte-exact retained DFlash context mark (BF16 K/V and valid floor).
pub fn mimo_draft_mark_bytes(layers: u64, kv_width: u64) -> Result<u64, MimoDraftStorageError> {
    mul(mul(mul(layers, 1024)?, kv_width)?, 4)?.checked_add(8)
        .ok_or(MimoDraftStorageError::Overflow)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimoDraftWeightLayout {
    /// Owned BF16 GEMM values, excluding the borrowed target head.
    pub bf16_values: u64,
    pub fp8_values: u64,
    pub fp8_scales: u64,
    /// Norms, sinks and trained mask row, always BF16.
    pub small_bf16: u64,
    /// Always zero: both immutable modes borrow the target head.
    pub head_fp8_values: u64,
    pub head_fp8_scales: u64,
    /// One BF16 source matrix while packing FP8-only storage. Packing is
    /// drained before this allocation is released or the next matrix loads.
    pub max_load_staging: u64,
}

/// Native FP8 scratch query for a selected operand shape, `[rows,k] @ [n,k]`.
/// Shape metadata owns no weights and does not allocate or load a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MimoDraftLinearShape {
    pub k: u64,
    pub n: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoDraftFp8ScratchLayout {
    pub rows: usize,
    pub shapes: Vec<MimoDraftLinearShape>,
}

/// One immutable drafter mode's weight/loading and FP8 scratch contracts.
/// Ring/activation/attention arenas still use `capacity` and native queries;
/// BF16-only modes never query or allocate FP8 scratch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoDraftRuntimeLayout {
    pub capacity: MimoDraftCapacity,
    pub weights: MimoDraftWeightLayout,
    pub fp8_scratch: Option<MimoDraftFp8ScratchLayout>,
}

impl MimoDraftRuntimeLayout {
    pub fn new(
        geometry: MimoDraftGeometry,
        mode: MimoDraftRepresentation,
        capacity: MimoDraftCapacity,
        tap_rows: usize,
        _legacy_fp8_rows: usize,
    ) -> Result<Self, MimoDraftStorageError> {
        if tap_rows == 0 {
            return Err(MimoDraftStorageError::Unsupported("drafter scratch needs positive row capacity"));
        }
        let weights = MimoDraftWeightLayout::new(geometry, mode)?;
        let h = geometry.hidden;
        let attention = mul(geometry.heads, geometry.head_dim)?;
        let kv = mul(geometry.kv_heads, geometry.head_dim)?;
        let taps = mul(geometry.taps, h)?;
        let qkv = add(attention, mul(2, kv)?)?;
        let shapes = vec![
            MimoDraftLinearShape { k: taps, n: h },
            MimoDraftLinearShape { k: h, n: qkv },
            MimoDraftLinearShape { k: h, n: mul(2, kv)? },
            MimoDraftLinearShape { k: attention, n: h },
            MimoDraftLinearShape { k: h, n: mul(2, geometry.intermediate)? },
            MimoDraftLinearShape { k: geometry.intermediate, n: h },
        ];
        let fp8_scratch = match mode {
            MimoDraftRepresentation::Bf16Only => None,
            MimoDraftRepresentation::Fp8Only => Some(MimoDraftFp8ScratchLayout {
                rows: tap_rows.max(capacity.block_rows), shapes,
            }),
        };
        Ok(Self { capacity, weights, fp8_scratch })
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MimoDraftStorageError {
    #[error("unsupported MiMo DFlash storage: {0}")]
    Unsupported(&'static str),
    #[error("MiMo DFlash storage byte count overflow")]
    Overflow,
}

fn mul(a: u64, b: u64) -> Result<u64, MimoDraftStorageError> {
    a.checked_mul(b).ok_or(MimoDraftStorageError::Overflow)
}
fn add(a: u64, b: u64) -> Result<u64, MimoDraftStorageError> {
    a.checked_add(b).ok_or(MimoDraftStorageError::Overflow)
}
fn sum(values: impl IntoIterator<Item = u64>) -> Result<u64, MimoDraftStorageError> {
    values.into_iter().try_fold(0, add)
}

impl MimoDraftWeightLayout {
    pub fn new(
        g: MimoDraftGeometry,
        mode: MimoDraftRepresentation,
    ) -> Result<Self, MimoDraftStorageError> {
        if [
            g.hidden,
            g.intermediate,
            g.layers,
            g.heads,
            g.kv_heads,
            g.head_dim,
            g.taps,
            g.vocab,
        ]
        .contains(&0)
            || g.heads % g.kv_heads != 0
        {
            return Err(MimoDraftStorageError::Unsupported(
                "invalid drafter geometry",
            ));
        }
        let attention = mul(g.heads, g.head_dim)?;
        let kv = mul(g.kv_heads, g.head_dim)?;
        let layer_shapes = [
            (add(attention, mul(2, kv)?)?, g.hidden),
            (g.hidden, attention),
            (mul(2, g.intermediate)?, g.hidden),
            (g.hidden, g.intermediate),
        ];
        let fc = (g.hidden, mul(g.taps, g.hidden)?);
        let uses_fp8 = mode != MimoDraftRepresentation::Bf16Only;
        let matrix =
            |(n, k): (u64, u64), packed: bool| -> Result<(u64, u64, u64), MimoDraftStorageError> {
                if packed && (n % 16 != 0 || k % 128 != 0) {
                    return Err(MimoDraftStorageError::Unsupported(
                        "packed FP8 matrices require N % 16 == 0 and K % 128 == 0",
                    ));
                }
                let values = mul(n, k)?;
                Ok((
                    mul(values, 2)?.max(256),
                    values.max(256),
                    mul(values / 128, 4)?.max(256),
                ))
            };
        let layer = layer_shapes.into_iter().map(|shape| matrix(shape, uses_fp8))
            .collect::<Result<Vec<_>, _>>()?;
        let fc_bytes = matrix(fc, uses_fp8)?;
        let own = |index: usize| -> Result<u64, MimoDraftStorageError> {
            let get = |m: &(u64, u64, u64)| match index {
                0 => m.0,
                1 => m.1,
                _ => m.2,
            };
            add(get(&fc_bytes), mul(g.layers, sum(layer.iter().map(get))?)?)
        };
        let small_per_layer = sum([
            mul(2, mul(g.hidden, 2)?.max(256))?,
            mul(2, mul(g.head_dim, 2)?.max(256))?,
            if g.sinks {
                mul(g.heads, 2)?.max(256)
            } else {
                0
            },
        ])?;
        let small_bf16 = add(
            mul(g.layers, small_per_layer)?,
            mul(3, mul(g.hidden, 2)?.max(256))?,
        )?;
        Ok(Self {
            bf16_values: match mode {
                MimoDraftRepresentation::Fp8Only => 0,
                _ => own(0)?,
            },
            fp8_values: match mode {
                MimoDraftRepresentation::Bf16Only => 0,
                _ => own(1)?,
            },
            fp8_scales: match mode {
                MimoDraftRepresentation::Bf16Only => 0,
                _ => own(2)?,
            },
            small_bf16,
            head_fp8_values: 0,
            head_fp8_scales: 0,
            max_load_staging: match mode {
                MimoDraftRepresentation::Fp8Only => layer
                    .iter()
                    .map(|m| m.0)
                    .chain([fc_bytes.0])
                    .max()
                    .unwrap_or(0),
                _ => 0,
            },
        })
    }

    pub fn resident_bytes(&self) -> Result<u64, MimoDraftStorageError> {
        sum([
            self.bf16_values,
            self.fp8_values,
            self.fp8_scales,
            self.small_bf16,
            self.head_fp8_values,
            self.head_fp8_scales,
        ])
    }

    pub fn loading_peak_bytes(&self) -> Result<u64, MimoDraftStorageError> {
        add(self.resident_bytes()?, self.max_load_staging)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_draft_context_mark_counts_both_bf16_planes_and_floor() {
        assert_eq!(mimo_draft_mark_bytes(5, 8 * 128).unwrap(), 20 * 1024 * 1024 + 8);
        assert!(mimo_draft_mark_bytes(u64::MAX, 1024).is_err());
    }

    fn pro() -> MimoDraftGeometry {
        MimoDraftGeometry {
            hidden: 6144,
            intermediate: 16384,
            layers: 5,
            heads: 128,
            kv_heads: 8,
            head_dim: 128,
            taps: 5,
            vocab: 152576,
            sinks: true,
        }
    }

    #[test]
    fn selected_runtime_layout_covers_wide_updates_and_independent_slots() {
        let capacity = MimoDraftCapacity::new(20, 16, 8).unwrap();
        for mode in [MimoDraftRepresentation::Fp8Only] {
            let layout = MimoDraftRuntimeLayout::new(pro(), mode, capacity, 1024, 128).unwrap();
            let scratch = layout.fp8_scratch.unwrap();
            assert_eq!(scratch.rows, 1024);
            assert_eq!(scratch.shapes.len(), 6);
            assert!(scratch.shapes.contains(&MimoDraftLinearShape { k: 30_720, n: 6144 }));
            assert!(scratch.shapes.contains(&MimoDraftLinearShape { k: 6144, n: 2048 }));
            assert!(!scratch.shapes.iter().any(|shape| shape.n == 152_576));
            assert_eq!(layout.capacity.context_slots, 20);
            assert_eq!(layout.capacity.block_rows, 128);
            assert_eq!(layout.weights.max_load_staging, 402_653_184);
        }
        let bf16 = MimoDraftRuntimeLayout::new(pro(), MimoDraftRepresentation::Bf16Only, capacity, 1024, 0).unwrap();
        assert!(bf16.fp8_scratch.is_none());
        assert_eq!(bf16.weights.max_load_staging, 0);

    }

    #[test]
    fn runtime_scratch_admits_larger_supported_draft_batches() {
        let capacity = MimoDraftCapacity::new(200, 160, 8).unwrap();
        let layout = MimoDraftRuntimeLayout::new(pro(), MimoDraftRepresentation::Fp8Only, capacity, 1024, 128).unwrap();
        assert_eq!(layout.fp8_scratch.unwrap().rows, 1280);
        assert!(MimoDraftRuntimeLayout::new(pro(), MimoDraftRepresentation::Fp8Only, capacity, 0, 128).is_err());
    }
    #[test]
    fn pro_single_copy_counts_and_one_matrix_loading_peak() {
        let fp8 = MimoDraftWeightLayout::new(pro(), MimoDraftRepresentation::Fp8Only).unwrap();
        let bf16 = MimoDraftWeightLayout::new(pro(), MimoDraftRepresentation::Bf16Only).unwrap();
        assert_eq!(bf16.bf16_values, 5_536_481_280);
        assert_eq!(fp8.fp8_values, 2_768_240_640);
        assert_eq!(fp8.fp8_scales, 86_507_520);
        assert_eq!(fp8.max_load_staging, 402_653_184);
        assert_eq!(
            fp8.bf16_values + fp8.head_fp8_values + fp8.head_fp8_scales,
            0
        );
        assert_eq!(
            bf16.fp8_values + bf16.fp8_scales + bf16.head_fp8_values + bf16.max_load_staging,
            0
        );
        assert_eq!(
            fp8.loading_peak_bytes().unwrap(),
            fp8.resident_bytes().unwrap() + 402_653_184
        );
    }
    #[test]
    fn twenty_context_slots_and_sixteen_block8_members_are_independent() {
        assert_eq!(
            MimoDraftCapacity::new(20, 16, 8).unwrap(),
            MimoDraftCapacity {
                context_slots: 20,
                max_batch_sequences: 16,
                block_rows: 128
            }
        );
        assert!(MimoDraftCapacity::new(16, 20, 8).is_err());
        assert!(MimoDraftCapacity::new(20, 0, 8).is_err());
        assert!(MimoDraftCapacity::new(20, 16, 1).is_err());
        assert!(MimoDraftCapacity::new(2_097_153, 16, 8).is_err());
    }
    #[test]
    fn reject_overflow_and_fp8_alignment_but_preserve_bf16_geometry() {
        let mut g = pro();
        g.hidden = 6145;
        assert!(MimoDraftWeightLayout::new(g, MimoDraftRepresentation::Fp8Only).is_err());
        assert!(MimoDraftWeightLayout::new(g, MimoDraftRepresentation::Bf16Only).is_ok());
        g.hidden = u64::MAX;
        assert_eq!(
            MimoDraftWeightLayout::new(g, MimoDraftRepresentation::Bf16Only),
            Err(MimoDraftStorageError::Overflow)
        );
    }
}
