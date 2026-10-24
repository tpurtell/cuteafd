//! Routed-expert geometry shared by the coordinator, the transport and the
//! Spark expert service. One process serves one model, so the geometry is a
//! process-wide value fixed at startup; it defaults to DeepSeek V4.1 so every
//! V4.1 path (and its tests) behaves exactly as before.
use std::sync::OnceLock;
use serde::{Deserialize, Serialize};

/// Native EXLR8 workers reuse one device arena across serial request chunks.
pub const EXLR8_EXECUTION_ROWS: usize = 64;

/// One persistent coordinator-to-Spark connection, resolved before allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkEndpoint {
    pub capacity: u32,
    pub ingress: SparkIngress,
    /// Overrides only this connection's depth; other endpoints keep the model default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ring_depth: Option<usize>,
}

/// Most execution slots one Spark rank runs (a link admits at most four waves).
pub const MAX_SPARK_EXECUTION_SLOTS: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparkIngress {
    Fp8K32,
    Bf16,
    /// Decode rows are unquantized; prefill uses FP8 K32 on the same QP.
    Bf16Decode { rows: u32 },
}

/// Header-derived resident slice. EXL3 counts retain gate/up/down tier geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkSlice {
    /// Canonical source-unit map seal for unit-sliced expert backends.
    pub unit_map: Option<String>,
    pub layer: u32,
    pub first: u32,
    pub width: u32,
    /// Physical kernel width, including checkpoint-format block padding.
    pub stored_width: u32,
    pub tiers: Vec<u32>,
    pub projection_counts: Vec<[u32; 3]>,
}

/// Shared runtime sample, measured on the device that owns the allocations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSample {
    pub device: i32,
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// Device use outside this process's tracked allocations, including modules.
    pub context_bytes: u64,
    /// Program module growth measured between the two runtime samples.
    pub module_bytes: u64,
    pub sms: u32,
    pub arch: String,
    pub driver: String,
}

/// A retained allocation delta from the process ledger. Aliases are not owners.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkAllocationTerm {
    pub space: String,
    pub scope: String,
    pub tensor: Option<String>,
    pub format: Option<String>,
    pub bytes: u64,
    pub allocations: u64,
    pub method: String,
}

#[path = "spark_census.rs"]
mod spark_census;
pub use spark_census::{spark_census_deltas, CensusDeviceKind, CensusDeviceWitness, SparkCensusDeltas, SparkCensusError};

/// Raw counters for one visible memory-cgroup ancestor; not process attribution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CgroupMemorySample {
    pub path: String,
    pub version: u8,
    pub current_bytes: Option<u64>,
    pub peak_bytes: Option<u64>,
    /// Missing file and an explicitly unlimited ceiling are distinct.
    pub max: Option<CgroupMemoryLimit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgroupMemoryLimit { Bytes(u64), Unlimited }

/// Same-process evidence: integrated CUDA uses system/cgroup phase deltas;
/// process maps attribute host-only memory, never driver-owned CUDA UMA.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkProcessSample {
    pub phase: String,
    pub pid: u32,
    pub monotonic_ns: u64,
    pub rss_bytes: u64,
    /// Process lifetime peak RSS from /proc/self/status, absent without an mm.
    pub peak_rss_bytes: Option<u64>,
    /// Current and lifetime peak counters, including visible ancestor limits.
    pub cgroup_memory: Vec<CgroupMemorySample>,
    pub pss_bytes: u64,
    pub private_bytes: u64,
    pub tracked_device_bytes: u64,
    pub tracked_pinned_bytes: u64,
    pub raw_cuda_free_bytes: Option<u64>,
    pub device_kind: Option<CensusDeviceKind>,
    pub device_witness: Option<CensusDeviceWitness>,
    pub cuda_mapping_pss_bytes: u64,
    pub anonymous_pss_bytes: u64,
    pub file_pss_bytes: u64,
    pub system_available_bytes: u64,
    pub system_total_bytes: u64,
    pub system_headroom_bytes: u64,
    pub allocatable_bytes: Option<u64>,
    pub method: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkRuntimeDetail {
    /// Checkpoint-derived, layer-sequential load-only owners, absent for backends
    /// without a qualified load plan. Never added to the serving workspace.
    pub load_transient: Vec<SparkAllocationTerm>,
    pub allocations: Vec<SparkAllocationTerm>,
    pub process_samples: Vec<SparkProcessSample>,
    /// PostLoad's GB10 free-memory residual includes unrelated system/cache
    /// state and must not become an offline context/modules calibration.
    pub runtime_sample_method: String,
    pub context_modules_qualified: bool,
}

/// Measurements made by the Spark owner, not coordinator-architecture estimates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkRuntimeInventory {
    pub detail: SparkRuntimeDetail,
    pub sample: RuntimeSample,
    /// All retained native/AOT device workspace, including worker intake.
    pub workspace_bytes: u64,
    /// Retained native/AOT pinned scratch, excluding endpoint rings and exchange.
    pub pinned_workspace_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAdmissionResponse {
    pub rank: u32,
    pub resident_bytes: u64,
    pub slices: Vec<SparkSlice>,
    pub runtime: SparkRuntimeInventory,
    pub mapped_ring_bytes: u64,
    pub host_exchange_bytes: u64,
    pub row_indices_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkRings {
    pub max_frame_bytes: usize,
    pub depth: usize,
    pub slot_bytes: usize,
    pub alignment: usize,
}

impl SparkEndpoint {
    /// The planner and both transport peers use the same admitted ring depth.
    pub fn resolved_rings(self, rings: SparkRings) -> Option<SparkRings> {
        let depth = self.ring_depth.unwrap_or(rings.depth);
        (1..=8).contains(&depth).then_some(SparkRings { depth, ..rings })
    }

    /// Retained response metadata per QP: largest header plus indexed rows.
    pub fn response_prefix_bytes(self) -> Option<usize> {
        if !(1..=4096).contains(&self.capacity) { return None; }
        (self.capacity as usize).checked_mul(4)?.checked_add(128)
    }

    pub fn wire_bytes(self, geometry: ExpertGeometry) -> Option<(usize, usize)> {
        if !(1..=4096).contains(&self.capacity) { return None; }
        let (request, response) = geometry.compact_wire_bytes(self.capacity,
            self.ingress == SparkIngress::Bf16)?;
        let request = match self.ingress {
            SparkIngress::Bf16Decode { rows } => {
                if rows == 0 || rows > self.capacity { return None; }
                request.max(geometry.compact_wire_bytes(rows, true)?.0)
            }
            _ => request,
        };
        Some((request, response))
    }

    pub fn ring_bytes(self, geometry: ExpertGeometry, rings: SparkRings) -> Option<usize> {
        let (request, response) = self.wire_bytes(geometry)?;
        geometry.ring_bytes_for_wire(request, response, self.resolved_rings(rings)?, 1)
    }
}

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
    /// Kimi K3 latent MoE: experts see the 3584-wide `routed_expert_down_proj`
    /// output (not the 7168 residual); routed layers 1..93, sigmoid top-16, Situ.
    pub const KIMI_K3: Self = Self { hidden: 3584, experts: 896, topk: 16, intermediate: 3072, layers: 93 };

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
        let (request, response) = self.compact_wire_bytes(capacity, bf16)?;
        self.ring_bytes_for_wire(request, response, SparkRings {
            max_frame_bytes: max_frame, depth, slot_bytes: slot, alignment }, endpoints)
    }

    fn ring_bytes_for_wire(&self, request: usize, response: usize, rings: SparkRings,
        endpoints: usize) -> Option<usize> {
        let SparkRings { max_frame_bytes, depth, slot_bytes, alignment } = rings;
        if !(1..=8).contains(&depth) || slot_bytes == 0 || alignment == 0 || endpoints == 0 { return None; }
        if request > max_frame_bytes || response > max_frame_bytes { return None; }
        let align = |bytes: usize| bytes.max(slot_bytes).min(max_frame_bytes).checked_add(alignment - 1)
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
            (Self::KIMI_K3, "kimik3"),
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
            && !self.same_shape(&Self::MIMO_V26_PRO) && !self.same_shape(&Self::QWEN4)
            && !self.same_shape(&Self::KIMI_K3)).then_some(10.0)
    }

    /// Expert arithmetic; existing families retain their legacy rounding exactly.
    pub fn activation(&self) -> crate::FfnActivation {
        if self.same_shape(&Self::KIMI_K3) {
            crate::FfnActivation::Situ { beta: 4.0, linear_beta: Some(25.0) }
        } else if self.same_shape(&Self::STEP5) {
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
/// GLM Flash and Kimi use K64 at m1/m80; m16 uses K128 and causes drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exl3RowPolicy {
    Nearest,
    GlmFlashK64,
}

impl Exl3RowPolicy {
    pub fn for_geometry(geometry: ExpertGeometry) -> Self {
        if geometry.same_shape(&ExpertGeometry::GLM5_FLASH)
            || geometry.same_shape(&ExpertGeometry::KIMI_K3)
        { Self::GlmFlashK64 } else { Self::Nearest }
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
        Ok(CAPACITIES.into_iter().filter(|&c| c <= maximum
            && (self != Self::GlmFlashK64 || c != 16)).collect())
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
    fn endpoint_ring_depth_resolves_default_and_rejects_invalid_depths() {
        let endpoint = SparkEndpoint { capacity: 4096, ingress: SparkIngress::Fp8K32, ring_depth: None };
        let geometry = ExpertGeometry::DEEPSEEK_V41;
        let rings = SparkRings { max_frame_bytes: 64 << 20, depth: 8,
            slot_bytes: 8 << 20, alignment: 4096 };
        assert_eq!(endpoint.resolved_rings(rings), Some(rings));
        let default_bytes = endpoint.ring_bytes(geometry, rings).unwrap();
        for depth in 1..=8 {
            let endpoint = SparkEndpoint { ring_depth: Some(depth), ..endpoint };
            assert_eq!(endpoint.resolved_rings(rings), Some(SparkRings { depth, ..rings }));
            assert_eq!(endpoint.ring_bytes(geometry, rings), Some(default_bytes / 8 * depth));
        }
        for depth in [0, 9, usize::MAX] {
            assert!(endpoint.resolved_rings(SparkRings { depth, ..rings }).is_none());
            let endpoint = SparkEndpoint { ring_depth: Some(depth), ..endpoint };
            assert!(endpoint.resolved_rings(rings).is_none());
            assert!(endpoint.ring_bytes(geometry, rings).is_none());
        }
        let serialized = serde_json::to_value(endpoint).unwrap();
        assert!(serialized.get("ring_depth").is_none());
        assert_eq!(serde_json::from_value::<SparkEndpoint>(serialized).unwrap(), endpoint);
        let endpoint = SparkEndpoint { ring_depth: Some(2), ..endpoint };
        let serialized = serde_json::to_value(endpoint).unwrap();
        assert_eq!(serialized["ring_depth"], 2);
        assert_eq!(serde_json::from_value::<SparkEndpoint>(serialized).unwrap(), endpoint);
    }

    #[test]
    fn endpoint_response_prefix_is_bounded_by_admitted_rows() {
        for capacity in [0, 1, 640, 4096, 4097, u32::MAX] {
            let endpoint = SparkEndpoint { ring_depth: None, capacity, ingress: SparkIngress::Bf16 };
            assert_eq!(endpoint.response_prefix_bytes(), if (1..=4096).contains(&capacity) {
                Some(128 + capacity as usize * 4)
            } else { None });
        }
    }

    #[test]
    fn exl3_row_policy_preserves_glmf_k64_at_narrow_admission() {
        let policy = Exl3RowPolicy::for_geometry(ExpertGeometry::GLM5_FLASH);
        assert_eq!(policy, Exl3RowPolicy::GlmFlashK64);
        assert!(policy.capacities(0).is_err());
        assert!(policy.capacities(4097).is_err());
        assert_eq!(policy.capacities(1).unwrap(), [1]);
        for rows in 2..=16 {
            assert_eq!(policy.required_capacity(rows), 80);
            assert_eq!(policy.capacities(rows).unwrap(), [1, 80]);
            assert_eq!(Exl3RowPolicy::Nearest.capacities(rows).unwrap(), [1, 16, 80]);
        }
        assert_eq!(policy.capacities(4096).unwrap(), [1, 80, 256, 1024, 4096]);
        assert_eq!(Exl3RowPolicy::for_geometry(ExpertGeometry::KIMI_K3), policy);
        let old = [1, 16, 80, 256, 1024, 4096];
        let new = policy.capacities(4096).unwrap();
        for rows in 1..=4096 {
            let required = policy.required_capacity(rows);
            let before = old.iter().find(|&&c| c >= required).unwrap();
            let after = new.iter().find(|&&c| c as usize >= required).unwrap();
            assert_eq!(*before, *after as usize, "GLM Flash selection changed at {rows}");
            assert_ne!(*after, 16);
        }
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
        let kimi = ExpertGeometry::KIMI_K3;
        assert_eq!(kimi.family(), Some("kimik3"));
        assert_eq!(kimi.swiglu_limit(), None);
        assert_eq!(kimi.activation(), crate::FfnActivation::Situ { beta: 4.0, linear_beta: Some(25.0) });
        assert_eq!(kimi.slice(6), Some(512));
        // Latent rows: BF16 ingress is sized from 3584, not the 7168 residual.
        assert_eq!(kimi.row_bytes(), 7168);
        assert_eq!(kimi.compact_wire_bytes(1, true), Some((40 + 16 * 12 + 3584 * 2 + 96, 4 + 3584 * 2 + 96)));
        assert_eq!(ExpertGeometry::QWEN4.slice(4), Some(160));
    }
}
