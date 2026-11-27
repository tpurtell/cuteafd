//! Private native vocabulary-head owners, separate from reusable logit planes.
use super::trellis_core::{CoreWeightLayout, ProjectionScratchLayout};
use anyhow::Result;

/// Match the native projection's compiled unit, not the retained logit-row count.
/// This scratch stays with the head owner and cannot alias the target layer arena.
pub fn native_private_scratch(shape: [usize; 2], unit_rows: usize) -> Result<ProjectionScratchLayout> {
    let [outputs, hidden] = shape;
    ProjectionScratchLayout::new(hidden, outputs, unit_rows)
}

/// As [`native_private_scratch`], charging the fused K8 route extent only when
/// `weight` actually runs the fused K8 decode program.
pub fn native_private_scratch_for_weight(weight: &CoreWeightLayout, unit_rows: usize) -> Result<ProjectionScratchLayout> {
    ProjectionScratchLayout::for_weight(weight, unit_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_head_private_scratch_uses_compiled_unit() -> Result<()> {
        let layout = native_private_scratch([128896, 4096], 256)?;
        assert_eq!(layout.rows, 256);
        let owners = layout.storage()?.owners;
        assert_eq!(owners["rotated"], 256 * 4096 * 2);
        assert_eq!(owners["accumulator"], 256 * 128896 * 4);
        assert_eq!(owners["route arena"], 129 * 4 * 128896 * 4);
        assert_eq!(layout.bytes()?, 400128072);
        assert!(native_private_scratch([128896, 4096], 64)?.bytes()? < layout.bytes()?);
        Ok(())
    }
}
