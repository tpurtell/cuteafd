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

/// Measured short + 32K C1 exact-row working-set inputs for GLM 5.3 / Mistral Large 4.
///
/// Shipping-config C1 census (donor-off, no-prewarm, normalized non-indexer keys,
/// `MIN_PAGE_STRIDE=64`, head `1c58d8fce0`, `GRAPH_TRACE` at 1536 MiB ceiling):
/// - GLM 5.3 1x RTX min (`k8-compact`, `heads`, DFlash2 whole): W1 174 MiB,
///   step 24 MiB, capture 4 MiB -> soft 270 / hard 278 MiB.
/// - GLM 5.3 2x RTX max (`k8-compact`, `context`, DFlash2 remote): W1 350/354 MiB,
///   step 38/40 MiB, capture 4/4 MiB -> soft 446/450 / hard 454/458 MiB.
/// - Mistral Large 4 2x RTX max (`original`, `context`, dSpark remote): W1 292/290 MiB,
///   step 34/36 MiB, capture 4/4 MiB -> soft 388/386 / hard 396/394 MiB.
/// - Mistral Large 4 1x RTX (non-shipping; scaled from GLM 5.3 1x by 88/78 layers):
///   W1 196 MiB, step 28 MiB, capture 4 MiB -> soft 292 / hard 300 MiB.
///
/// Soft and hard caps are derived by `graph_reserve::resolve_working_set`.
pub fn glm_working_set(family: GlmPrewarmFamily, ranks: usize, rank: usize) -> WorkingSet {
    let idx = rank.min(1);
    let (w1_mib, step_growth, capture_growth_mib) = match (family, ranks) {
        (GlmPrewarmFamily::Glm5, 0 | 1) => (174, 24 << 20, 4),
        (GlmPrewarmFamily::Glm5, _) => [(350, 38 << 20, 4), (354, 40 << 20, 4)][idx],
        (GlmPrewarmFamily::MistralLarge4, 0 | 1) => (196, 28 << 20, 4),
        (GlmPrewarmFamily::MistralLarge4, _) => [(292, 34 << 20, 4), (290, 36 << 20, 4)][idx],
    };
    WorkingSet { w1: w1_mib << 20, step_growth, capture_growth: capture_growth_mib << 20 }
}

/// Initial per-executable byte estimate before the device controller observes live captures
/// (seeded from measured C1 census per-graph averages).
pub fn glm_bootstrap_bytes(family: GlmPrewarmFamily, ranks: usize, rank: usize) -> u64 {
    let idx = rank.min(1);
    match (family, ranks) {
        (GlmPrewarmFamily::Glm5, 0 | 1) => 193_000,
        (GlmPrewarmFamily::Glm5, _) => [136_000, 138_000][idx],
        (GlmPrewarmFamily::MistralLarge4, 0 | 1) => 196_000,
        (GlmPrewarmFamily::MistralLarge4, _) => [127_000, 126_000][idx],
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

impl GlmGraphKey {
    /// Zero out page-table dimensions on segments that never read `table_width`
    /// or `table_stride` (shared-indexer attention, combine/o_proj, FFN/MoE, head)
    /// using the shared [`super::decode_graph_key::GraphGeometry`] helper so short
    /// and long context steps share those rank-local graphs.
    pub fn normalized(mut self, reads_page_table: bool) -> Self {
        let geometry = super::decode_graph_key::GraphGeometry::paged(
            self.table_width,
            self.table_stride,
            reads_page_table,
        );
        self.table_width = geometry.pool_width;
        self.table_stride = geometry.page_stride;
        self
    }
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

    #[test]
    fn working_set_inputs_derive_expected_soft_and_hard_caps() {
        use crate::placement::graph_reserve::resolve_working_set;
        for (family, ranks, rank, soft_mib, hard_mib) in [
            (GlmPrewarmFamily::Glm5, 1, 0, 270, 278),
            (GlmPrewarmFamily::Glm5, 2, 0, 446, 454),
            (GlmPrewarmFamily::Glm5, 2, 1, 450, 458),
            (GlmPrewarmFamily::MistralLarge4, 1, 0, 292, 300),
            (GlmPrewarmFamily::MistralLarge4, 2, 0, 388, 396),
            (GlmPrewarmFamily::MistralLarge4, 2, 1, 386, 394),
        ] {
            let measured = glm_working_set(family, ranks, rank);
            let policy = resolve_working_set(None, None, None, rank, measured).unwrap();
            assert_eq!((policy.base_bytes, policy.bytes), (soft_mib << 20, hard_mib << 20),
                "{family:?} ranks={ranks} rank={rank}");
            assert_eq!(policy.growth_headroom, Some(8 << 20));
        }
    }

    #[test]
    fn graph_key_normalization_zeros_table_dims_only_when_unused() {
        let base = GlmGraphKey {
            layer: 4,
            sub: 3,
            rows: 5,
            table_width: 512,
            table_stride: 512,
            previous: GlmGraphPrevious::First,
            head: false,
        };
        let normalized = base.normalized(false);
        assert_eq!((normalized.table_width, normalized.table_stride), (0, 0));
        assert_eq!(base.normalized(true), base);
        let long = GlmGraphKey { table_width: 1024, table_stride: 1024, ..base };
        assert_eq!(base.normalized(false), long.normalized(false));
        assert_ne!(base.normalized(true), long.normalized(true));
    }
}
