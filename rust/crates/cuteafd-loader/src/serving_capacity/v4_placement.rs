//! V4 expert residency and local-expert workspace without CUDA, the inputs
//! `placement::families::deepseek_v4` charges (the solver is `placement::solve`).
use super::CacheGeometryError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V4ExpertCost {
    pub resident_bytes: u64,
    pub staging_bytes: u64,
}

/// Exact whole-layer residency without a GPU/native library. The native
/// MXFP4 packer's four planes pad intermediate rows to 128; EXL3 includes
/// aligned rotations and all resident buffers, not merely trellis storage.
pub fn deepseek_v4_expert_cost(catalog: &crate::OfficialV41Catalog, layer: usize,
    draft: bool) -> anyhow::Result<V4ExpertCost> {
    if let Some(exl3) = catalog.exl3() {
        let selection = if draft { crate::V41Exl3Layer::Dspark(layer) } else { crate::V41Exl3Layer::Backbone(layer) };
        let residency = exl3.residency(selection, 1, 0)?;
        return Ok(V4ExpertCost { resident_bytes: residency.device_arena_layout()?.1 as u64, staging_bytes: 0 });
    }
    let shape = catalog.routed_experts();
    let staging = catalog.expert_staging(crate::V41ExpertSelection::BackboneFull {
        layer: if draft { shape.layers + layer } else { layer }, expert: 0 })?;
    let h = shape.hidden as u64;
    let padded = (shape.intermediate as u64).div_ceil(128) * 128;
    Ok(V4ExpertCost { resident_bytes: (padded * h + padded * h / 16 + h * padded / 2 + h * padded / 32)
        * shape.experts as u64, staging_bytes: staging.staging_bytes() as u64 })
}

/// Standard rtx_backbone export scratch: 16-byte-aligned native slots,
/// ordered route output below 256 rows, atomic token output above it.
/// Mirrors export_b12x_slices_aot.py and expert_families.cmake without CUDA.
pub fn deepseek_v4_native_workspace(hidden: u64, intermediate: u64, experts: u64,
    topk: u64, max_rows: u64) -> Result<u64, CacheGeometryError> {
    use super::{product, sum};
    let maximum = max_rows.max(1);
    let capacities = [1_u64, 16, 80, 256, 1024, 4096];
    if maximum > 4096 {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v4", what: "local expert capacity exceeds 4096 rows" });
    }
    let mut scratch = 0;
    for capacity in capacities.into_iter().filter(|&n| n <= maximum)
        .chain(capacities.into_iter().find(|&n| n >= maximum)) {
        let routes = product("V4 expert routes", &[capacity, topk])?;
        let atomic = capacity >= 256;
        let planes = intermediate.div_ceil(if capacity == 1 { 64 } else { 128 });
        let partial = if atomic { 4 } else { product("V4 ordered partials", &[4, planes, routes, hidden])? };
        let output = product("V4 routed output", &[4, if atomic { capacity } else { routes }, hidden])?;
        let slots = [4, product("V4 packed routes", &[4, experts, routes])?, experts * 4, experts * 8,
            routes * 19 * 4, routes * 4, routes * 4, partial, output, experts * 4, experts * 4];
        let mut bytes = 0_u64;
        for slot in slots {
            bytes = sum("V4 aligned expert scratch", &[bytes.div_ceil(16) * 16, slot])?;
        }
        scratch = scratch.max(bytes);
    }
    sum("V4 native expert workspace", &[scratch, product("V4 local output/routes", &[maximum, hidden * 2 + topk * 8])?])
}
