//! Pool-first whole-layer placement shared by V4 planning and runtime admission.
use super::CacheGeometryError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V4ExpertCost {
    pub resident_bytes: u64,
    pub staging_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V4LocalRange {
    pub first: usize,
    pub layers: usize,
    pub peak_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V4Placement {
    pub pool_tokens: u64,
    pub ranks: Vec<V4LocalRange>,
}

/// `available` excludes every fixed reservation, but not the pool or experts.
/// Reserve the desired pool on ALL ranks first. Whole layers then form one
/// contiguous prefix, GPU0 followed by GPU1; dSpark remains on GPU0.
pub fn deepseek_v4_placement(
    available: &[u64], total: &[u64], pool_unit_bytes: &[u64], unit_rows: u64,
    requested_pool: Option<u64>, target_pool: Option<u64>, first_layer: usize,
    layers: &[V4ExpertCost], draft: &[V4ExpertCost], workspace: u64,
    requested_end: Option<usize>,
) -> Result<V4Placement, CacheGeometryError> {
    use CacheGeometryError::Overflow;
    if available.is_empty() || available.len() > 2 || available.len() != total.len()
        || available.len() != pool_unit_bytes.len() || unit_rows == 0 || pool_unit_bytes.contains(&0) {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v4", what: "invalid placement inventory" });
    }
    let draft_peak = |mut resident: u64| -> Result<u64, CacheGeometryError> {
        let mut peak = resident;
        for weight in draft {
            peak = peak.max(resident.checked_add(weight.resident_bytes).and_then(|n| n.checked_add(weight.staging_bytes))
                .ok_or(Overflow("V4 draft peak"))?);
            resident = resident.checked_add(weight.resident_bytes).ok_or(Overflow("V4 draft resident"))?;
        }
        Ok(peak)
    };
    let mandatory = if draft.is_empty() { 0 } else { draft_peak(workspace)? };
    let feasible_units = available.iter().zip(pool_unit_bytes).enumerate()
        .map(|(rank, (&bytes, &unit))| bytes.saturating_sub(if rank == 0 { mandatory } else { 0 }) / unit)
        .min().unwrap_or(0);
    let target = target_pool.unwrap_or_else(|| if total.iter().any(|&n| n <= 32 << 30) { 1 << 20 } else { 2 << 20 });
    let wanted = requested_pool.filter(|&n| n > 0).unwrap_or(target).div_ceil(unit_rows);
    if requested_pool.is_some_and(|n| n > 0) && wanted > feasible_units {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v4", what: "explicit KV pool does not fit before local experts" });
    }
    let units = wanted.min(feasible_units);
    if units == 0 {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v4", what: "KV pool and mandatory dSpark experts do not fit" });
    }
    let limit = requested_end.unwrap_or(usize::MAX).saturating_sub(first_layer).min(layers.len());
    let mut next = 0;
    let mut ranks = Vec::new();
    for (rank, (&bytes, &unit)) in available.iter().zip(pool_unit_bytes).enumerate() {
        let budget = bytes - units * unit;
        let first = first_layer + next;
        let mut resident = workspace;
        let mut peak = workspace;
        if rank == 0 {
            peak = draft_peak(workspace)?;
            resident = draft.iter().try_fold(resident, |n, w| n.checked_add(w.resident_bytes)
                .ok_or(Overflow("V4 draft resident")))?;
        }
        while next < limit {
            let w = layers[next];
            let candidate = resident.checked_add(w.resident_bytes).and_then(|n| n.checked_add(w.staging_bytes))
                .ok_or(Overflow("V4 expert peak"))?.max(peak);
            if candidate > budget { break; }
            resident = resident.checked_add(w.resident_bytes).ok_or(Overflow("V4 expert resident"))?;
            peak = candidate;
            next += 1;
        }
        let count = first_layer + next - first;
        ranks.push(V4LocalRange { first, layers: count,
            peak_bytes: if count > 0 || (rank == 0 && !draft.is_empty()) { peak } else { 0 } });
    }
    if requested_end.is_some() && next != limit {
        return Err(CacheGeometryError::Unsupported { family: "deepseek_v4", what: "explicit local expert layers do not fit after reserving KV on every GPU" });
    }
    Ok(V4Placement { pool_tokens: units.checked_mul(unit_rows).ok_or(Overflow("V4 pool tokens"))?, ranks })
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pool_precedes_contiguous_layers_and_explicit_admission_fails() {
        let weights = vec![V4ExpertCost { resident_bytes: 1 << 30, staging_bytes: 1 << 28 }; 60];
        let solve = |end| deepseek_v4_placement(&[12 << 30, 20 << 30], &[96 << 30; 2],
            &[4096; 2], 256, None, None, 0, &weights, &[], 1 << 28, end);
        let plan = solve(None).unwrap();
        assert_eq!(plan.pool_tokens, 2 << 20);
        assert_eq!((plan.ranks[0].first, plan.ranks[0].layers), (0, 11));
        assert_eq!((plan.ranks[1].first, plan.ranks[1].layers), (11, 19));
        assert!(solve(Some(31)).is_err());
        assert_eq!(solve(Some(0)).unwrap().pool_tokens, 2 << 20);
        let small = deepseek_v4_placement(&[12 << 30], &[32 << 30], &[4096], 256,
            None, None, 0, &weights, &[], 1 << 28, None).unwrap();
        assert_eq!(small.pool_tokens, 1 << 20);
    }
}
