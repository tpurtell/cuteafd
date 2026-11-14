//! Native attention geometry shared by checkpoint inventories and runtime owners.
use super::trellis_core::CoreSplit;
use anyhow::{Context, Result};

/// A family explicitly permits output padding for this role; no shape infers it.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Role {
    pub name: &'static str,
    pub supported: bool,
    pub pad_to: Option<usize>,
}
impl Fp8Role {
    pub const fn exact(name: &'static str, supported: bool) -> Self {
        Self { name, supported, pad_to: None }
    }
    pub const fn padded(name: &'static str, supported: bool, alignment: usize) -> Self {
        Self { name, supported, pad_to: Some(alignment) }
    }
    pub fn stored_shape(self, [n, k]: [usize; 2]) -> Result<[usize; 2]> {
        anyhow::ensure!(self.supported && n > 0 && k > 0 && k % 128 == 0,
            "role {} requires supported positive block-128 geometry [{n}, {k}]", self.name);
        let stored = match self.pad_to {
            None => { anyhow::ensure!(n % 128 == 0, "role {} requires whole output blocks; declare pad_to explicitly", self.name); n },
            Some(alignment) => {
                anyhow::ensure!(alignment > 0 && alignment % 128 == 0, "role {} pad_to must be a multiple of 128", self.name);
                n.checked_add(alignment - 1).context("FP8 role output padding overflow")? / alignment * alignment
            }
        };
        Ok([stored, k])
    }
}

/// Unrotated selected-view attention with the same logical TP axes as K6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProjection { pub shape: [usize; 2], pub fp8: bool, pub split: CoreSplit }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLayout {
    pub stem: String, pub shape: [usize; 2], pub fp8: bool,
    pub rank: usize, pub ranks: usize, pub split: CoreSplit,
}
impl NativeLayout {
    pub fn allocation_extents(&self) -> Result<Vec<u64>> {
        let [n, k] = self.shape;
        let values = n.checked_mul(k).and_then(|n| n.checked_mul(if self.fp8 { 1 } else { 2 }))
            .context("native attention projection extent overflow")?.max(256);
        let scales = if self.fp8 { n.div_ceil(128).checked_mul(k.div_ceil(128)).and_then(|n| n.checked_mul(4))
            .context("native attention scale extent overflow")?.max(256) } else { 0 };
        Ok(if self.fp8 { vec![values as u64, scales as u64] } else { vec![values as u64] })
    }

    pub fn bytes(&self) -> Result<u64> {
        self.allocation_extents()?.into_iter().try_fold(0u64, |sum, bytes|
            sum.checked_add(bytes).context("native attention projection byte overflow"))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{tensor}: native FP8 attention requires one declared F32 block-128 scale tensor; {reason}")]
pub struct InvalidFp8Scale { pub tensor: String, pub reason: &'static str }

/// Resolve only tensors in the selected index, preserving each publication's name.
pub fn fp8_scale<'a>(selected: &'a crate::plan::checkpoint::SelectedProjection, stem: &str)
    -> Result<&'a crate::plan::checkpoint::CheckpointTensor> {
    let invalid = |reason| InvalidFp8Scale { tensor: format!("{stem}.weight"), reason };
    let crate::plan::checkpoint::ProjectionStorage::Native { dtype: cuteafd_core::DType::F8E4M3, shape }
        = &selected.storage else { return Err(invalid("source is not native E4M3").into()); };
    let mut scales = selected.tensors.iter().filter(|tensor| tensor.meta.name.strip_prefix(stem)
        .is_some_and(|suffix| matches!(suffix, ".weight_scale_inv" | ".qscale_weight")));
    let scale = scales.next().ok_or_else(|| invalid("scale is missing"))?;
    if scales.next().is_some() { return Err(invalid("scale is ambiguous").into()); }
    if shape.len() != 2 || scale.meta.dtype != cuteafd_core::DType::F32
        || scale.meta.shape != [shape[0].div_ceil(128), shape[1].div_ceil(128)]
        || scale.meta.byte_length != scale.meta.shape.iter().product::<usize>() as u64 * 4 {
        return Err(invalid("scale dtype or geometry differs from the selected weight").into());
    }
    Ok(scale)
}

/// FP8-compact is a conversion of native BF16, never a requantization of native FP8.
#[derive(Debug, thiserror::Error)]
#[error("{tensor}: native core operand is already FP8; CORE_VIEW=fp8-compact applies only to BF16 operands (found {found})")]
pub struct UnsupportedFp8Conversion { pub tensor: String, pub found: String }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{plan::checkpoint::{CheckpointTensor, ProjectionStorage, SelectedProjection}, SafetensorsTensorMetadata};
    use cuteafd_core::DType;

    #[test]
    fn physical_native_extents_keep_values_and_fp8_scales_separate() -> Result<()> {
        for (shape, fp8, expected) in [([256, 512], false, vec![262144]),
            ([256, 512], true, vec![131072, 256]), ([1, 1], true, vec![256, 256])] {
            let layout = NativeLayout { stem: "projection".into(), shape, fp8, rank: 0, ranks: 1, split: CoreSplit::Whole };
            assert_eq!(layout.allocation_extents()?, expected);
            assert_eq!(layout.bytes()?, expected.iter().sum::<u64>());
        }
        let overflow = NativeLayout { stem: "overflow".into(), shape: [usize::MAX, 2], fp8: false,
            rank: 0, ranks: 1, split: CoreSplit::Whole };
        assert!(overflow.allocation_extents().is_err());
        Ok(())
    }

    #[test]
    fn selected_fp8_scales_preserve_both_names_and_refuse_missing_or_ambiguous() -> Result<()> {
        for suffix in ["weight_scale_inv", "qscale_weight"] {
            let stem = "layers.0.attention.wq";
            let scale = CheckpointTensor { shard: "native.safetensors".into(), meta: SafetensorsTensorMetadata {
                name: format!("{stem}.{suffix}"), dtype: DType::F32, shape: vec![2, 4], byte_offset: 512, byte_length: 32 } };
            let mut selected = SelectedProjection { storage: ProjectionStorage::Native { dtype: DType::F8E4M3,
                shape: vec![256, 512] }, tensors: vec![scale.clone()], source_bytes: 32 };
            assert_eq!(fp8_scale(&selected, stem)?.meta.name, scale.meta.name);
            selected.tensors[0].meta.dtype = DType::Bf16;
            assert!(fp8_scale(&selected, stem).unwrap_err().is::<InvalidFp8Scale>());
            selected.tensors.clear();
            assert!(fp8_scale(&selected, stem).unwrap_err().is::<InvalidFp8Scale>());
            selected.tensors.push(scale.clone());
            let mut alias = scale;
            alias.meta.name = format!("{stem}.{}", if suffix == "qscale_weight" { "weight_scale_inv" } else { "qscale_weight" });
            selected.tensors.push(alias);
            assert!(fp8_scale(&selected, stem).unwrap_err().is::<InvalidFp8Scale>());
        }
        Ok(())
    }
}

#[cfg(test)]
mod padding_tests {
    use super::*;
    #[test]
    fn padding_is_declared_and_checked_without_changing_logical_source() -> Result<()> {
        let exact = Fp8Role::exact("projection", true);
        assert_eq!(exact.stored_shape([512, 7168])?, [512, 7168]);
        assert!(exact.stored_shape([576, 7168]).is_err());
        let padded = Fp8Role::padded("projection", true, 128);
        assert_eq!(padded.stored_shape([576, 7168])?, [640, 7168]);
        assert_eq!(padded.stored_shape([202048, 8192])?, [202112, 8192]);
        assert!(padded.stored_shape([576, 7169]).is_err());
        assert!(padded.stored_shape([usize::MAX, 8192]).is_err());
        assert!(Fp8Role::padded("projection", true, 64).stored_shape([576,7168]).is_err());
        assert!(Fp8Role::padded("projection", false, 128).stored_shape([576,7168]).is_err());
        Ok(())
    }
}
