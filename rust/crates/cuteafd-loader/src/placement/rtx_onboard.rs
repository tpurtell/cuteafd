//! RTX expert onboarding shared by every family whose routed experts run the
//! shared RTX expert packages (EXL3 mixed trellis, or fp8moe FP8 / NVFP4 /
//! BF16): exact header-only residency per routed layer, whole (TP1) and as
//! TP2 halves, and the installed package's workspace arena. The package family
//! comes from the routed geometry, so a family adds no onboarding code here.
use super::{inventory, Bytes2, ExpertCost};
use crate::serving_capacity::Exl3Epilogue;
use anyhow::{ensure, Context, Result};

/// Whole and (when `tp2`) TP2-half residency of each routed layer, from the
/// checkpoint's own expert headers.
pub fn expert_costs(catalog: &crate::OfficialV41Catalog, tp2: bool) -> Result<Vec<ExpertCost>> {
    if catalog.exl3().is_none() { return fp8_expert_costs(catalog, tp2); }
    let exl3 = catalog.exl3().context("EXL3 onboarding needs EXL3 expert headers")?;
    let shape = catalog.routed_experts();
    let cost = |layer, tp, rank| -> Result<Bytes2> {
        Ok(Bytes2 { resident: exl3.resident_arena_bytes(crate::V41Exl3Layer::Backbone(layer), tp, rank, false)?, staging: 0 })
    };
    (shape.first_layer..shape.layers).map(|layer| Ok(ExpertCost { whole: cost(layer, 1, 0)?,
        half: if tp2 { [cost(layer, 2, 0)?, cost(layer, 2, 1)?] } else { [Bytes2::default(); 2] },
        tp2, whole_supported: true, spark_ok: true })).collect()
}

/// fp8moe (FP8 block-128, NVFP4, MXFP4, BF16) residency over whole H128 blocks.
fn fp8_expert_costs(catalog: &crate::OfficialV41Catalog, tp2: bool) -> Result<Vec<ExpertCost>> {
    use crate::formats::fp8_experts::{Fp8Projection, Slicing};
    let fp8 = catalog.fp8().context("RTX onboarding needs EXL3 or fp8moe expert headers")?;
    let shape = catalog.routed_experts();
    let coordinator = fp8.for_coordinator();
    (shape.first_layer..shape.layers).map(|layer| {
        let view = coordinator.for_layer_metadata(layer)?;
        let supported = view.format() == fp8.format();
        let cost = |tp, rank| -> Result<Bytes2> {
            let resident = Fp8Projection::ALL.iter().try_fold(0u64, |sum, &p| -> Result<u64> {
                let (w, _) = view.slice_bytes_with(p, tp, rank, Slicing::Blocks(128))?;
                Ok(sum + (shape.experts * w).max(256) as u64
                    + view.scale_region_bytes_with(p, tp, rank, Slicing::Blocks(128))?.max(256) as u64)
            })?;
            Ok(Bytes2 { resident, staging: 0 })
        };
        // One RTX executor owns one homogeneous package; other native formats stay on Sparks.
        Ok(ExpertCost { whole: cost(1, 0)?,
            half: if tp2 { [cost(2, 0)?, cost(2, 1)?] } else { [Bytes2::default(); 2] },
            tp2: tp2 && supported, whole_supported: supported, spark_ok: true })
    }).collect()
}

/// The installed RTX expert package's arena at `rows` for TP1 or TP2, plus
/// the routed output. `family` names the family's fp8moe package
/// (`fp8-<family>[-format]`); EXL3 packages follow the routed geometry.
/// Missing packages fail closed rather than becoming zero scratch.
pub fn expert_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize, family: &str) -> Result<u64> {
    ensure!([1, 2].contains(&tp), "RTX experts need TP1 or TP2");
    ensure!((1..=4096).contains(&rows), "RTX expert capacity must be 1..=4096 rows");
    expert_workspace_requests(catalog, manifest, rows, tp, family)?.into_iter()
        .try_fold(0u64, |sum, bytes| sum.checked_add(bytes)).context("RTX workspace sum overflow")
}

/// Physical CUDA requests, excluding expert weights and family exchange planes.
pub fn expert_workspace_requests(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize, family: &str) -> Result<Vec<u64>> {
    ensure!([1, 2].contains(&tp) && (1..=4096).contains(&rows), "RTX workspace geometry");
    if catalog.exl3().is_some() { exl3_workspace(catalog, manifest, rows, tp) }
    else { fp8_workspace(catalog, manifest, rows, tp, family) }
}

/// `fp8-<family>[-format]/tp<tp>` scratch (and a BF16-input sibling's, when
/// installed) plus the TP2 BF16 partial plane.
fn fp8_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize, family: &str) -> Result<Vec<u64>> {
    fp8_workspace_requests(catalog.fp8().context("fp8moe expert catalog missing")?, manifest, rows, tp, family)
}

/// One homogeneous native catalog view, including mixed-format checkpoints.
/// Returns the same requests as `expert_workspace_requests`, without reopening
/// the checkpoint or replacing its selected view.
pub fn fp8_workspace_requests(fp8: &crate::formats::fp8_experts::Fp8ExpertTensors,
    manifest: Option<&std::path::Path>, rows: u64, tp: usize, family: &str) -> Result<Vec<u64>> {
    ensure!([1, 2].contains(&tp) && (1..=4096).contains(&rows), "RTX workspace geometry");
    let shape = fp8.shape();
    let name = fp8.format().package_name(family);
    // Beside the program manifest (a WIP slot's share), else the image lib.
    let parent = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json")).parent()
        .context("program manifest directory")?;
    let lib = [parent.to_path_buf(), inventory::image_lib(manifest)].into_iter()
        .find(|lib| lib.join("fp8").join(&name).join("manifest.json").is_file())
        .with_context(|| format!("missing {name} package"))?;
    if fp8.format() == crate::formats::fp8_experts::ExpertFormat::Nvfp4 {
        ensure!(lib.join("fp8").join(&name).join(format!("tp{tp}")).is_dir(), "missing {name}/tp{tp} package");
    }
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(lib.join("fp8").join(&name).join("manifest.json"))?)?;
    let scratch = inventory::fp8moe_scratch_bytes(&value, &format!("tp{tp}"), rows)?
        .with_context(|| format!("missing {name} tp{tp} scratch for {rows} rows"))?;
    let bf16 = lib.join("fp8").join(format!("{name}-bf16"));
    let sibling = if bf16.join(format!("tp{tp}")).is_dir() {
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(bf16.join("manifest.json"))?)?;
        inventory::fp8moe_scratch_bytes(&manifest, &format!("tp{tp}"), rows)?
            .with_context(|| format!("missing BF16-input {name} scratch"))?
    } else { 0 };
    let mut requests = vec![scratch.max(sibling)];
    if tp == 2 { requests.push(rows * shape.hidden as u64 * 2); }
    Ok(requests)
}

/// The installed `exl3-<family>-k<tiers>/rtx-tp<tp>` arena at `rows`, plus the
/// routed output (F32 for both whole and TP2 execution).
fn exl3_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize) -> Result<Vec<u64>> {
    let exl3 = catalog.exl3().context("EXL3 onboarding needs EXL3 expert headers")?;
    let shape = catalog.routed_experts();
    let geometry = shape.geometry()?;
    let family = geometry.family().context("EXL3 routed geometry has no expert package family")?;
    let stem = exl3.package_name(family);
    let lib = inventory::image_lib(manifest);
    let parent = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json")).parent()
        .context("program manifest directory")?;
    let layout = format!("rtx-tp{tp}");
    let exl3_roots = [parent.join("exl3"), lib.join("exl3")];
    let root = exl3_roots.iter().map(|r| r.join(&stem))
        .find(|p| p.join(&layout).is_dir())
        .ok_or_else(|| anyhow::anyhow!(
            "{}",
            exl3.codebook().missing_package_detail(family, exl3.decoder_tiers(), &layout, &exl3_roots)
        ))?;
    let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(geometry).capacities(usize::try_from(rows)?)?;
    let manifests = capacities.into_iter().map(|capacity| -> Result<serde_json::Value> {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
            root.join(format!("rtx-tp{tp}/m{capacity}/v41_exl3.json")))?)?;
        exl3.codebook().check_aot_manifest(&value)?;
        ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
            && value["intermediate"].as_u64() == Some((shape.intermediate / tp) as u64)
            && value["experts"].as_u64() == Some(shape.experts as u64), "{stem} package geometry mismatch");
        Ok(value)
    }).collect::<Result<Vec<_>>>()?;
    let mut requests = crate::serving_capacity::Exl3WorkspaceLayout::from_manifests(
        &manifests, true, Exl3Epilogue::Sum)?.owners.into_iter().map(|owner| owner.bytes).collect::<Vec<_>>();
    requests.push(crate::serving_capacity::exl3_rtx_output_bytes(rows, shape.hidden as u64)?);
    Ok(requests)
}
