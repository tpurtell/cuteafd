//! Routed-expert geometry shared by the coordinator, the transport and the
//! Spark expert service. One process serves one model, so the geometry is a
//! process-wide value fixed at startup; it defaults to DeepSeek V4.1 so every
//! V4.1 path (and its tests) behaves exactly as before.
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertGeometry {
    /// Model hidden size: the width of every routed row and partial.
    pub hidden: u32,
    /// Routed experts per layer.
    pub experts: u32,
    /// Experts selected per token.
    pub topk: u32,
    /// Full (un-sliced) expert intermediate size.
    pub intermediate: u32,
    /// Backbone layers with routed experts.
    pub layers: u32,
}

impl ExpertGeometry {
    pub const DEEPSEEK_V41: Self = Self { hidden: 5120, experts: 384, topk: 6, intermediate: 2304, layers: 40 };
    pub const DEEPSEEK_V4_FLASH: Self = Self { hidden: 4096, experts: 256, topk: 6, intermediate: 2048, layers: 43 };
    pub const DEEPSEEK_V4_PRO: Self = Self { hidden: 7168, experts: 384, topk: 6, intermediate: 3072, layers: 61 };
    /// GLM 5.3 (glm_moe_dsa): routed layers 3..78 (layers is the id bound).
    pub const GLM5: Self = Self { hidden: 6144, experts: 256, topk: 8, intermediate: 2048, layers: 78 };
    /// MiMo V2 Flash (mimo_v2_flash): routed layers 1..48, FP8 experts.
    pub const MIMO_V2_FLASH: Self = Self { hidden: 4096, experts: 256, topk: 8, intermediate: 2048, layers: 48 };
    /// MiMo V2.6 Pro (mimo_v2): routed layers 1..70, MXFP4 experts.
    pub const MIMO_V26_PRO: Self = Self { hidden: 6144, experts: 384, topk: 8, intermediate: 2048, layers: 70 };
    /// GLM 5.3 Flash (glm5_next): routed layers 3..45, SwiGLU clamped at 10.
    pub const GLM5_FLASH: Self = Self { hidden: 4096, experts: 288, topk: 8, intermediate: 2048, layers: 45 };
    /// Qwen 3.8 Flash Next (qwen4_exp): 48 routed layers, softmax top-10, SiLU unclamped.
    pub const QWEN4: Self = Self { hidden: 2560, experts: 512, topk: 10, intermediate: 640, layers: 48 };

    /// Step-5 routed target layers 3..91 (layer 91 is dense).
    pub const STEP5: Self = Self { hidden: 4096, experts: 352, topk: 8, intermediate: 1536, layers: 91 };

    /// BF16 bytes of one hidden-width row (a routed input or a rank partial).
    pub const fn row_bytes(&self) -> u32 {
        self.hidden * 2
    }

    /// FP8 K32 request and row-indexed compact BF16 response frame upper bounds.
    pub fn compact_wire_bytes(&self, capacity: u32, bf16: bool) -> Option<(usize, usize)> {
        let rows = capacity as usize;
        let hidden = self.hidden as usize;
        let request = rows.checked_mul(40 + self.topk as usize * 12 + if bf16 { hidden * 2 } else { hidden + hidden / 32 })?.checked_add(96)?;
        let response = rows.checked_mul(4 + hidden * 2)?.checked_add(96)?;
        Some((request, response))
    }

    /// Persistent request/response mapped rings, shared by admission and planning.
    pub fn compact_ring_bytes(&self, capacity: u32, max_frame: usize, depth: usize,
        slot: usize, alignment: usize, endpoints: usize, bf16: bool) -> Option<usize> {
        if !(1..=8).contains(&depth) || slot == 0 || alignment == 0 || endpoints == 0 { return None; }
        let (request, response) = self.compact_wire_bytes(capacity, bf16)?;
        if request > max_frame || response > max_frame { return None; }
        let align = |bytes: usize| bytes.max(slot).min(max_frame).checked_add(alignment - 1)
            .map(|bytes| bytes / alignment * alignment);
        align(request)?.checked_add(align(response)?)?.checked_mul(depth)?.checked_mul(endpoints)
    }

    /// Per-rank intermediate slice for tensor parallelism of degree `tp`.
    pub fn slice(&self, tp: u32) -> Option<u32> {
        (tp > 0 && self.intermediate % tp == 0).then(|| self.intermediate / tp)
    }

    /// Native expert kernel family: the symbol prefix (`cuteafd_{family}_*`)
    /// and exporter `--geometry` name of the AOT kernels built for this shape.
    /// The layer-id bound is not part of the kernel shape: an FP8 GLM catalog
    /// also serves its MTP layer (id 78, bound 79) with the `glm` kernels.
    pub fn family(&self) -> Option<&'static str> {
        [
            (Self::DEEPSEEK_V41, "v41"),
            (Self::DEEPSEEK_V4_FLASH, "dsv4f"),
            (Self::DEEPSEEK_V4_PRO, "dsv4p"),
            (Self::GLM5, "glm"),
            (Self::MIMO_V2_FLASH, "mimo"),
            (Self::MIMO_V26_PRO, "mimop"),
            (Self::GLM5_FLASH, "glmf"),
            (Self::QWEN4, "qwen4"),
            (Self::STEP5, "step5"),
        ]
        .into_iter()
        .find_map(|(shape, family)| self.same_shape(&shape).then_some(family))
    }

    /// Routed-expert SwiGLU clamp: DeepSeek and GLM 5.3 Flash clamp gate/up at
    /// 10; Step-5 clamps post-SiLU at 7 (see `activation`). GLM 5.x, MiMo
    /// and Qwen 3.8 Flash Next are unclamped (`None`).
    pub fn swiglu_limit(&self) -> Option<f32> {
        if self.same_shape(&Self::STEP5) { return Some(7.0); }
        (!self.same_shape(&Self::GLM5) && !self.same_shape(&Self::MIMO_V2_FLASH)
            && !self.same_shape(&Self::MIMO_V26_PRO) && !self.same_shape(&Self::QWEN4)).then_some(10.0)
    }

    /// Expert arithmetic; existing families retain their legacy rounding exactly.
    pub fn activation(&self) -> crate::FfnActivation {
        if self.same_shape(&Self::STEP5) {
            crate::FfnActivation::SiluPostClampFp32Product { limit: 7.0 }
        } else {
            crate::FfnActivation::SiluBf16Product { limit: self.swiglu_limit() }
        }
    }

    /// Equal kernel shape (hidden, experts, top-k, intermediate), any layer bound.
    pub fn same_shape(&self, other: &Self) -> bool {
        Self { layers: other.layers, ..*self } == *other
    }

    /// A short stable key for artifact and symbol names.
    pub fn key(&self) -> String {
        format!("h{}e{}k{}i{}l{}", self.hidden, self.experts, self.topk, self.intermediate, self.layers)
    }
}

/// Capacity selects the compiled reduction geometry, never the live row count.
/// GLM Flash's m1/m80 use K64; m16 uses K128 and adds serial/verify drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3RowPolicy {
    Nearest,
    GlmFlashK64,
}

impl Exl3RowPolicy {
    pub fn for_geometry(geometry: ExpertGeometry) -> Self {
        if geometry.same_shape(&ExpertGeometry::GLM5_FLASH) { Self::GlmFlashK64 } else { Self::Nearest }
    }

    pub fn active() -> Self { Self::for_geometry(expert_geometry()) }

    pub fn required_capacity(self, live_rows: usize) -> usize {
        if self == Self::GlmFlashK64 && (2..=16).contains(&live_rows) { 80 } else { live_rows }
    }

    /// Shared by loader admission, artifact preload and runtime dispatch.
    pub fn capacities(self, max_live_rows: usize) -> Result<Vec<u32>, crate::serving_capacity::CapacityError> {
        const CAPACITIES: [u32; 6] = [1, 16, 80, 256, 1024, 4096];
        if !(1..=4096).contains(&max_live_rows) {
            return Err(crate::serving_capacity::CapacityError::Invalid("EXL3 live capacity must be 1..4096"));
        }
        // Keep a canonical-reduction decode export available even at narrow admission.
        let required = max_live_rows.max(if max_live_rows > 1 { 80 } else { 1 });
        let maximum = CAPACITIES.into_iter().find(|&c| c as usize >= required).unwrap();
        Ok(CAPACITIES.into_iter().filter(|&c| c <= maximum).collect())
    }
}

static GEOMETRY: OnceLock<ExpertGeometry> = OnceLock::new();

/// Fixes the process geometry. Allowed once, before first use; setting the
/// value it already has is a no-op.
pub fn set_expert_geometry(geometry: ExpertGeometry) -> Result<(), ExpertGeometry> {
    match GEOMETRY.set(geometry) {
        Ok(()) => Ok(()),
        Err(_) if *GEOMETRY.get().unwrap() == geometry => Ok(()),
        Err(_) => Err(*GEOMETRY.get().unwrap()),
    }
}

/// The process geometry (DeepSeek V4.1 unless set at startup).
pub fn expert_geometry() -> ExpertGeometry {
    *GEOMETRY.get_or_init(|| ExpertGeometry::DEEPSEEK_V41)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exl3_row_policy_preserves_glmf_k64_at_narrow_admission() {
        let policy = Exl3RowPolicy::for_geometry(ExpertGeometry::GLM5_FLASH);
        assert_eq!(policy, Exl3RowPolicy::GlmFlashK64);
        assert!(policy.capacities(0).is_err());
        assert!(policy.capacities(4097).is_err());
        assert_eq!(policy.capacities(1).unwrap(), [1]);
        for rows in 2..=16 {
            assert_eq!(policy.required_capacity(rows), 80);
            assert_eq!(policy.capacities(rows).unwrap(), [1, 16, 80]);
            assert_eq!(Exl3RowPolicy::Nearest.capacities(rows).unwrap(), [1, 16, 80]);
        }
        assert_eq!(policy.capacities(4096).unwrap(), [1, 16, 80, 256, 1024, 4096]);
    }

    #[test]
    fn v41_defaults_and_slices() {
        let g = ExpertGeometry::DEEPSEEK_V41;
        assert_eq!(g.row_bytes(), 10240);
        assert_eq!(g.slice(4), Some(576));
        assert_eq!(ExpertGeometry::DEEPSEEK_V4_FLASH.slice(4), Some(512));
        assert_eq!(ExpertGeometry::DEEPSEEK_V4_PRO.slice(6), Some(512));
        assert_eq!(g.slice(5), None);
        assert_eq!(g.key(), "h5120e384k6i2304l40");
        assert_eq!(g.family(), Some("v41"));
        assert_eq!(ExpertGeometry::DEEPSEEK_V4_FLASH.family(), Some("dsv4f"));
        assert_eq!(g.swiglu_limit(), Some(10.0));
        assert_eq!(ExpertGeometry::GLM5.swiglu_limit(), None);
        assert_eq!(ExpertGeometry::MIMO_V2_FLASH.family(), Some("mimo"));
        assert_eq!(ExpertGeometry::MIMO_V2_FLASH.swiglu_limit(), None);
        assert_eq!(ExpertGeometry::MIMO_V2_FLASH.slice(4), Some(512));
        let glm_with_mtp = ExpertGeometry { layers: 79, ..ExpertGeometry::GLM5 };
        assert_eq!(glm_with_mtp.family(), Some("glm"));
        assert_eq!(glm_with_mtp.swiglu_limit(), None);
        assert_eq!(ExpertGeometry::QWEN4.family(), Some("qwen4"));
        assert_eq!(ExpertGeometry::QWEN4.swiglu_limit(), None);
        assert_eq!(ExpertGeometry::QWEN4.slice(4), Some(160));
    }
}
