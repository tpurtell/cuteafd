//! RTX expert onboarding for EXL3 routed experts, shared by every family whose
//! experts run the shared mixed-trellis packages: exact header-only residency
//! per routed layer and the compiled package's workspace arena.
use super::{inventory, Bytes2, ExpertCost};
use anyhow::{ensure, Context, Result};

/// Whole and TP2-half residency of each routed layer from the EXL3 headers.
pub fn expert_costs(catalog: &crate::OfficialV41Catalog, tp2: bool) -> Result<Vec<ExpertCost>> {
    let exl3 = catalog.exl3().context("EXL3 onboarding needs EXL3 expert headers")?;
    let shape = catalog.routed_experts();
    let cost = |layer, tp, rank| -> Result<Bytes2> {
        Ok(Bytes2 { resident: exl3.residency(crate::V41Exl3Layer::Backbone(layer), tp, rank)?
            .device_arena_layout()?.1 as u64, staging: 0 })
    };
    (shape.first_layer..shape.layers).map(|layer| Ok(ExpertCost { whole: cost(layer, 1, 0)?,
        half: if tp2 { [cost(layer, 2, 0)?, cost(layer, 2, 1)?] } else { [Bytes2::default(); 2] },
        tp2, spark_ok: true })).collect()
}

/// The installed `exl3-<family>-k<tiers>/rtx-tp<tp>` arena at `rows`, plus the
/// routed output (F32 partials for TP2, BF16 for one GPU). Missing packages
/// fail closed rather than becoming zero scratch.
pub fn rtx_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize) -> Result<u64> {
    ensure!([1, 2].contains(&tp), "EXL3 RTX experts need TP1 or TP2");
    ensure!((1..=4096).contains(&rows), "EXL3 RTX expert capacity must be 1..=4096 rows");
    let exl3 = catalog.exl3().context("EXL3 onboarding needs EXL3 expert headers")?;
    let shape = catalog.routed_experts();
    let geometry = shape.geometry()?;
    let family = geometry.family().context("EXL3 routed geometry has no expert package family")?;
    let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let stem = format!("exl3-{family}-k{tiers}");
    let lib = inventory::image_lib(manifest);
    let parent = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json")).parent()
        .context("program manifest directory")?;
    let root = [parent.join("exl3").join(&stem), lib.join("exl3").join(&stem)]
        .into_iter().find(|p| p.join(format!("rtx-tp{tp}")).is_dir())
        .with_context(|| format!("missing {stem}/rtx-tp{tp} package"))?;
    let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(geometry).capacities(usize::try_from(rows)?)?;
    let manifests = capacities.into_iter().map(|capacity| -> Result<serde_json::Value> {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join(format!("rtx-tp{tp}/m{capacity}/v41_exl3.json")))?)?;
        ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
            && value["intermediate"].as_u64() == Some((shape.intermediate / tp) as u64)
            && value["experts"].as_u64() == Some(shape.experts as u64), "{stem} package geometry mismatch");
        Ok(value)
    }).collect::<Result<Vec<_>>>()?;
    Ok(crate::serving_capacity::exl3_workspace_bytes(&manifests, true)?
        + rows.max(1) * shape.hidden as u64 * if tp == 2 { 4 } else { 2 })
}
