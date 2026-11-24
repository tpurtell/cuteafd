//! GLM 5.3 / Mistral Large 4 exact-row rank-local decode graph geometry and W1 admission.
//!
//! Every decode/verify graph segment is rank-local (`Scope::Local`, `spans_devices = false`)
//! and keyed by exact live rows `1..=64` (no row bucketing, no rank-local startup prewarm).
//! Reserves are sized through the shared `graph_reserve::resolve_working_set` rule from
//! measured C1 `(w1, step_growth, capture_growth)` inputs.
use crate::families::glm5::GlmVariant;
use crate::placement::graph_reserve::WorkingSet;

/// Which checkpoint family's working-set reserve a glm5-engine layout uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmPrewarmFamily {
    /// GLM 5.x (78 layers).
    Glm5,
    /// Mistral Large 4 (88 layers).
    MistralLarge4,
}

impl GlmPrewarmFamily {
    pub fn of(variant: &GlmVariant) -> Self {
        if variant.is_glm() { Self::Glm5 } else { Self::MistralLarge4 }
    }
    pub fn name(self) -> &'static str {
        match self { Self::Glm5 => "glm5", Self::MistralLarge4 => "mistral_large4" }
    }
}

/// C1 exact-row working-set inputs for GLM 5.3 / Mistral Large 4.
///
/// Provisional inputs before the hardware W1 census on GLM 5.3 MC K8 min/max
/// and ML4 max; replaced with measured `(w1, step_growth, capture_growth)` before
/// final cards. Soft and hard caps are derived by `graph_reserve::resolve_working_set`.
pub fn glm_working_set(family: GlmPrewarmFamily, ranks: usize, rank: usize) -> WorkingSet {
    let idx = rank.min(1);
    let (w1_mib, step_growth, capture_growth_mib) = match (family, ranks) {
        // Provisional before W1 census (GLM 5.3: 78 layers; ML4: 88 layers).
        (GlmPrewarmFamily::Glm5, 0 | 1) => (160, 14 << 20, 4),
        (GlmPrewarmFamily::Glm5, _) => [(256, 18 << 20, 4), (256, 18 << 20, 4)][idx],
        (GlmPrewarmFamily::MistralLarge4, 0 | 1) => (256, 18 << 20, 4),
        (GlmPrewarmFamily::MistralLarge4, _) => [(320, 20 << 20, 4), (320, 20 << 20, 4)][idx],
    };
    WorkingSet { w1: w1_mib << 20, step_growth, capture_growth: capture_growth_mib << 20 }
}

/// Initial per-executable byte estimate before the device controller observes live captures.
pub fn glm_bootstrap_bytes(family: GlmPrewarmFamily, ranks: usize, rank: usize) -> u64 {
    let idx = rank.min(1);
    match (family, ranks) {
        (GlmPrewarmFamily::Glm5, 0 | 1) => 176_000,
        (GlmPrewarmFamily::Glm5, _) => [160_000, 156_000][idx],
        (GlmPrewarmFamily::MistralLarge4, 0 | 1) => 188_000,
        (GlmPrewarmFamily::MistralLarge4, _) => [172_000, 168_000][idx],
    }
}

/// What precedes a captured decode segment's residual norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GlmGraphPrevious {
    First,
    Delta,
    Planes(usize),
}

/// Stable launch geometry for a rank-local decode/verify graph segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub struct GlmGraphKey {
    pub layer: usize,
    pub sub: u8,
    pub rows: usize,
    pub table_width: usize,
    pub table_stride: usize,
    pub previous: GlmGraphPrevious,
    pub head: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct GlmGraphLayer {
    pub dense: bool,
    /// Whole or paired coordinator-local FFN; both produce a reduced delta.
    pub local: bool,
    pub attention_owner: Option<u8>,
}

/// Persistent graph-key phase for `ATTENTION_PLACEMENT=layers` segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlmOwnedPhase {
    Attention = 0,
    FfnFront = 1,
    Tp2Expert = 2,
    Head = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GlmGraphShape {
    pub rows: usize,
    pub table_width: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn working_set_inputs_are_chunk_aligned_and_positive() {
        const CHUNK: u64 = 2 << 20;
        for family in [GlmPrewarmFamily::Glm5, GlmPrewarmFamily::MistralLarge4] {
            for ranks in [1, 2] {
                for rank in 0..ranks {
                    let measured = glm_working_set(family, ranks, rank);
                    assert!(measured.w1 > 0 && measured.w1 % CHUNK == 0, "{family:?} ranks={ranks} rank={rank}");
                    assert!(measured.step_growth > 0 && measured.capture_growth > 0);
                    assert!(glm_bootstrap_bytes(family, ranks, rank) > 0);
                }
            }
        }
    }
}
