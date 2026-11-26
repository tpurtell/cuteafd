//! GLM launch-count model, deliberately not wired to admission until the memory
//! campaign lands. This first calibration covers EXL3 K4/Core K6 HQA, remote
//! experts, single RTX. Do not silently apply it to local/TP2/head variants.
use crate::placement::graph_nodes::{GraphNodeError, GraphNodeRate, GraphRateProvenance};
use super::glm_graphs::{GlmGraphKey, GlmGraphPrevious};

const PROVENANCE: GraphRateProvenance = GraphRateProvenance {
    engine_sha: "73cecead4692efc6b78f832414252b9439149baa",
    sparkinfer_sha: "851c42833da26711c68868014cad2fc043916416",
    cuda_driver_version: Some(13020), nvml_driver_version: Some("595.91.07"),
    evidence: "#1615; graph-param-model/analysis/family-node-graph-coefficients.json; binary f1ad35334731bf087782bdda3d2ad0fb2f1a7b5ca745811625aa26cc8ea29d8b",
    limitation: "SM120 remote HQA only; 1M calibration, independent 128K validation; local/TP2 unmeasured",
};

// Ordered 256-capture block fit: a=2675.552041747176, c=103661.85617785223.
// Whole-byte coefficients preserve the 1M calibration; independent 128K is +1 chunk.
pub const GLM_SM120_NODE_RATES: [GraphNodeRate; 1] = [
    GraphNodeRate { family: "glm5", architecture: "sm_120", bytes_per_node: 2676,
        bytes_per_executable: 103662, provenance: PROVENANCE },
];

// Same block fit on Flash: a=1777.5189825935627, c=125262.64261471576.
// Full-set prediction is one chunk above observation; no independent validation.
pub const GLMF_SM120_NODE_RATES: [GraphNodeRate; 1] = [
    GraphNodeRate { family: "glm5_flash", architecture: "sm_120", bytes_per_node: 1778,
        bytes_per_executable: 125263, provenance: GraphRateProvenance {
            engine_sha: "342a40506a0967670e150bb913370553b336c9b5",
            sparkinfer_sha: "851c42833da26711c68868014cad2fc043916416",
            cuda_driver_version: None, nvml_driver_version: None,
            evidence: "#1605; mem-campaign-glm5_flash/1rtx/graph-census-captures-raptor.jsonl; ordered 256-capture block fit",
            limitation: "single-rank remote HQA; inferred key join; driver unrecorded; full-set residual +1 chunk",
        } },
];

/// Launch metadata, not a per-layer capture table. The caller must establish
/// this measured route (including projection/head format) before using it.
#[derive(Debug, Clone, Copy)]
pub struct GlmRemoteHqaNodeClass {
    pub dense: bool,
    pub paged_dsa: bool,
    pub speculator_tap: bool,
}

pub fn glm_remote_hqa_nodes(key: &GlmGraphKey, class: GlmRemoteHqaNodeClass)
    -> Result<u64, GraphNodeError> {
    if key.rows == 0 || key.rows > 64 || key.table_width == 0 || key.table_stride != key.table_width {
        return Err(GraphNodeError::Nodes("GLM remote HQA rows/width/stride outside measured launch route".into()));
    }
    if key.head {
        return Ok(7);
    }
    let previous = match (class.dense, key.previous) {
        (true, GlmGraphPrevious::First) => 1i64,
        (true, GlmGraphPrevious::Delta) => 0,
        (false, GlmGraphPrevious::Delta) => -1,
        (false, GlmGraphPrevious::Planes(4)) => 0,
        _ => return Err(GraphNodeError::Nodes("GLM residual route needs a node-count derivation (local/TP2 is not the archived remote route)".into())),
    };
    if class.dense && !class.paged_dsa {
        return Err(GraphNodeError::Nodes("GLM dense non-DSA route is not calibrated".into()));
    }
    let base = if class.dense { 35 } else if class.paged_dsa { 28 } else { 24 };
    let ordinary = base + previous + i64::from(class.speculator_tap) + 2 * i64::from(key.rows > 16);
    let mut nodes = ordinary as u64;
    if class.paged_dsa {
        let repeats = u64::try_from(key.table_width.div_ceil(512))
            .ok().and_then(|v| v.checked_mul(2)).ok_or(GraphNodeError::Overflow("DSA launches"))?;
        nodes = nodes.checked_add(repeats).ok_or(GraphNodeError::Overflow("GLM nodes"))?;
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_hqa_node_counts_and_unknown_residual_route() {
        let key = GlmGraphKey { layer: 0, sub: 0, rows: 1, table_width: 16, table_stride: 16,
            previous: GlmGraphPrevious::First, head: false };
        assert_eq!(glm_remote_hqa_nodes(&key, GlmRemoteHqaNodeClass { dense: true,
            paged_dsa: true, speculator_tap: false }).unwrap(), 38);
        let head = GlmGraphKey { layer: 78, sub: 0, rows: 1, table_width: 16, table_stride: 16,
            previous: GlmGraphPrevious::Planes(4), head: true };
        assert_eq!(glm_remote_hqa_nodes(&head, GlmRemoteHqaNodeClass { dense: false,
            paged_dsa: false, speculator_tap: false }).unwrap(), 7);
        let unknown = GlmGraphKey { layer: 4, sub: 0, rows: 1, table_width: 16, table_stride: 16,
            previous: GlmGraphPrevious::Planes(0), head: false };
        assert!(glm_remote_hqa_nodes(&unknown, GlmRemoteHqaNodeClass { dense: false,
            paged_dsa: false, speculator_tap: false }).is_err());
    }
}
