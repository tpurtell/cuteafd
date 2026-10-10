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

/// Exact half-width backbone residency, including native staging, per rank.
pub fn deepseek_v4_tp2_expert_cost(catalog: &crate::OfficialV41Catalog, layer: usize,
    rank: usize) -> anyhow::Result<V4ExpertCost> {
    anyhow::ensure!(rank < 2, "TP2 expert rank must be 0 or 1");
    if let Some(exl3) = catalog.exl3() {
        let residency = exl3.residency(crate::V41Exl3Layer::Backbone(layer), 2, rank)?;
        return Ok(V4ExpertCost { resident_bytes: residency.device_arena_layout()?.1 as u64, staging_bytes: 0 });
    }
    let shape = catalog.routed_experts();
    let staging = catalog.expert_staging(crate::V41ExpertSelection::BackboneTp2 { layer, expert: 0, rank })?;
    let h = shape.hidden as u64;
    let padded = (shape.intermediate as u64 / 2).div_ceil(128) * 128;
    Ok(V4ExpertCost { resident_bytes: (padded * h + padded * h / 16 + h * padded / 2 + h * padded / 32)
        * shape.experts as u64, staging_bytes: staging.staging_bytes() as u64 })
}

/// `rtx_tp2` uses the same slots/atomic threshold as `rtx_backbone`, at I/2,
/// but its executor output is FP32 and has no TP1 output/route tail.
pub fn deepseek_v4_tp2_native_workspace(hidden: u64, intermediate: u64, experts: u64,
    topk: u64, max_rows: u64) -> Result<u64, CacheGeometryError> {
    use super::{product, sum};
    let rows = max_rows.max(1);
    let native = deepseek_v4_native_workspace(hidden, intermediate / 2, experts, topk, rows)?;
    let tail = product("V4 TP1 output tail", &[rows, hidden * 2 + topk * 8])?;
    sum("V4 TP2 native workspace", &[native - tail, product("V4 TP2 output", &[rows, hidden, 4])?])
}

/// Read the TP2 package arenas once per compiled capacity. Unlike the legacy
/// TP1 helper, an exact maximum (e.g. 4096) is not counted twice.
pub fn deepseek_v4_tp2_workspace(catalog: &crate::OfficialV41Catalog,
    manifest: Option<&std::path::Path>, max_rows: u64) -> anyhow::Result<u64> {
    let shape = catalog.routed_experts();
    let Some(exl3) = catalog.exl3() else {
        return Ok(deepseek_v4_tp2_native_workspace(shape.hidden as u64, shape.intermediate as u64,
            shape.experts as u64, shape.topk as u64, max_rows)?);
    };
    let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let family = if shape.hidden == 4096 { "dsv4f" } else { "dsv4p" };
    let parent = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"))
        .parent().ok_or_else(|| anyhow::anyhow!("program manifest has no parent"))?;
    let stem = format!("exl3-{family}-k{tiers}");
    let root = [parent.join("exl3").join(&stem), parent.join("../lib/exl3").join(&stem)]
        .into_iter().find(|p| p.join("rtx-tp2").is_dir())
        .ok_or_else(|| anyhow::anyhow!("missing {stem}/rtx-tp2 workspace manifests"))?;
    let capacities = [1u64, 16, 80, 256, 1024, 4096];
    let compiled = capacities.into_iter().find(|&c| c >= max_rows.max(1))
        .ok_or_else(|| anyhow::anyhow!("TP2 expert capacity exceeds 4096 rows"))?;
    let manifests = capacities.into_iter().filter(|&c| c <= compiled).map(|c| {
        let path = root.join(format!("rtx-tp2/m{c}/v41_exl3.json"));
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
            && value["intermediate"].as_u64() == Some(shape.intermediate as u64 / 2)
            && value["experts"].as_u64() == Some(shape.experts as u64), "TP2 EXL3 workspace geometry mismatch");
        Ok(value)
    }).collect::<anyhow::Result<Vec<_>>>()?;
    Ok(super::exl3_workspace_bytes(&manifests, true)? + max_rows.max(1) * shape.hidden as u64 * 4)
}

#[cfg(test)]
mod tp2_tests {
    use super::*;
    #[test]
    fn native_tp2_workspace_matches_export_slots() {
        // Export widths are 64 at m1, 128 otherwise; ordered output below
        // m256 dominates the native scratch at m80 for both V4 geometries.
        for (h, i, e, k, expected) in [(4096, 2048, 256, 6, 161453088), (7168, 3072, 384, 6, 297139088)] {
            assert_eq!(deepseek_v4_tp2_native_workspace(h, i, e, k, 4096).unwrap(), expected);
        }
        assert!(deepseek_v4_tp2_native_workspace(4096, 2048, 256, 6, 4097).is_err());
    }
    #[test]
    fn real_tp2_costs_match_catalog_reference() {
        let hub = std::path::Path::new("/mnt/sparknest/hf-home/hub");
        for (model, family, resident, staging) in [
            ("deepseek-ai--DeepSeek-V4-Flash-0731", "flash", 1711276032u64, 6684672u64),
            ("wrldsuksgo2mars--DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1", "pro", 3212836864, 0)] {
            let Some(snapshot) = std::fs::read_dir(hub.join(format!("models--{model}/snapshots"))).ok()
                .and_then(|entries| entries.filter_map(|e| e.ok()).map(|e| e.path())
                    .find(|p| p.join("model.safetensors.index.json").is_file())) else { continue };
            let catalog = crate::read_expert_catalog(&snapshot).unwrap();
            let manifest = std::env::var_os("HOME").map(std::path::PathBuf::from).unwrap()
                .join(format!(".cache/cuteafd/builds/v4-placement/plans-input/{family}-2-1048576/PROGRAMS.json"));
            for rank in 0..2 {
                let cost = deepseek_v4_tp2_expert_cost(&catalog, catalog.routed_experts().first_layer, rank).unwrap();
                assert_eq!((cost.resident_bytes, cost.staging_bytes), (resident, staging));
                let workspace = deepseek_v4_tp2_workspace(&catalog, Some(&manifest), 4096).unwrap();
                eprintln!("TP2 {family} rank{rank}: resident={} staging={} workspace={workspace}", cost.resident_bytes, cost.staging_bytes);
            }
        }
    }

}
