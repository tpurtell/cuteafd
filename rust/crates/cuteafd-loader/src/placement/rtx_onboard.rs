//! RTX expert onboarding shared by every family whose routed experts run the
//! shared RTX expert packages (EXL3 mixed trellis, or fp8moe FP8 / NVFP4 /
//! BF16): exact header-only residency per routed layer, whole (TP1) and as
//! TP2 halves, and the installed package's workspace arena. The package family
//! comes from the routed geometry, so a family adds no onboarding code here.
use super::{inventory, Bytes2, ExpertCost};
use anyhow::{ensure, Context, Result};

/// Physical route/diagnostic storage of the shared paired executor. Backend
/// weights/workspaces and native code/events compose separately.
#[derive(Debug, Clone)]
pub struct PairedRouteBuffers {
    pub rows: u64,
    pub topk: u64,
    pub lanes: u64,
    pub input_row_bytes: u64,
    pub layers: Vec<usize>,
    pub peer_diagnostics: bool,
    pub route_checks: bool,
}

impl PairedRouteBuffers {
    pub fn allocations(&self, prefix: &str, lifetime: super::planner::Lifetime)
        -> Result<Vec<super::planner::Allocation>, super::planner::Refusal> {
        use cuteafd_core::memory_layout::Category;
        use super::planner::{Allocation, Location, Refusal};
        let invalid = || Refusal::InvalidFact("paired expert route geometry".into());
        if prefix.is_empty() || self.rows == 0 || self.topk == 0 || self.lanes == 0
            || self.layers.is_empty() || self.layers.windows(2).any(|ids| ids[0] >= ids[1]) {
            return Err(invalid());
        }
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(Refusal::Overflow("paired expert routes"));
        let add = |a: u64, b: u64| a.checked_add(b).ok_or(Refusal::Overflow("paired expert routes"));
        let aligned = |bytes: u64| add(bytes, 15).map(|n| n / 16 * 16);
        let section = mul(mul(self.rows, self.topk)?, 4)?;
        let packed = aligned(add(aligned(section)?, section)?)?;
        let receive = add(packed, mul(self.rows, self.input_row_bytes)?)?;
        if receive % 16 != 0 { return Err(invalid()); }
        let slots = mul(self.lanes, 2)?;
        let flags = add(slots, 1)?;
        let control = mul(flags, 16)?.max(256);
        let progress = mul(flags, std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>() as u64)?.max(256);
        let checks = mul(mul(self.rows, self.topk)?, 8)?;
        let rank_count = add(add(slots, 2 + u64::from(self.peer_diagnostics))?,
            if self.route_checks { mul(self.lanes, self.layers.len() as u64)? } else { 0 })?;
        let count = add(mul(rank_count, 2)?, u64::from(self.peer_diagnostics))?;
        let mut out = Vec::new();
        out.try_reserve_exact(usize::try_from(count).map_err(|_| invalid())?)
            .map_err(|_| invalid())?;
        for rank in 0..2 {
            let location = Location::coordinator(rank);
            let name = format!("{prefix}.rank.{rank}");
            for slot in 0..slots {
                out.push(Allocation::buffer(format!("{name}.routes.peer.recv.{slot}"), location,
                    receive.max(256), lifetime, Category::Transport, "paired expert routes"));
            }
            for (field, bytes) in [("control", control), ("abort", 256)] {
                out.push(Allocation::buffer(format!("{name}.routes.peer.{field}"), location,
                    bytes, lifetime, Category::Transport, "paired expert routes"));
            }
            if self.peer_diagnostics {
                out.push(Allocation::buffer(format!("{name}.routes.peer.progress"), location,
                    progress, lifetime, Category::Transport, "paired expert routes"));
            }
            if self.route_checks {
                for slot in 0..self.lanes {
                    for layer in &self.layers {
                        out.push(Allocation::buffer(format!("{name}.checks.slot.{slot}.layer.{layer}"),
                            Location::PINNED, checks, lifetime, Category::Staging, "paired expert diagnostics"));
                    }
                }
            }
        }
        if self.peer_diagnostics {
            let bytes = mul(flags, std::mem::size_of::<[[u64; 2]; 2]>() as u64)?;
            out.push(Allocation::buffer(format!("{prefix}.routes.peer.host_ordinals"), Location::HOST,
                bytes, lifetime, Category::Transport, "paired expert diagnostics"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod paired_owner_tests {
    use super::*;
    use super::super::planner::{Lifetime, Location};

    #[test]
    fn route_storage_prices_floors_wire_progress_and_each_retained_check() -> Result<()> {
        for rows in [1u64, 16, 64, 128] {
            for lanes in [1u64, 2, 3] {
                for input_row_bytes in [0, 32, 6336] {
                    for peer_diagnostics in [false, true] {
                        for route_checks in [false, true] {
                            let geometry = PairedRouteBuffers { rows, topk: 8, lanes, input_row_bytes,
                                layers: vec![3, 7], peer_diagnostics, route_checks };
                            let allocations = geometry.allocations("test", Lifetime::RESIDENT)?;
                            let receive = (rows * 64 + rows * input_row_bytes).max(256);
                            let flags = 2 * lanes + 1;
                            let rank_bytes = 2 * lanes * receive + (flags * 16).max(256) + 256
                                + if peer_diagnostics { (flags * 48).max(256) } else { 0 };
                            for rank in 0..2 {
                                assert_eq!(allocations.iter().filter(|a| a.location == Location::coordinator(rank))
                                    .map(|a| a.bytes).sum::<u64>(), rank_bytes);
                            }
                            assert_eq!(allocations.iter().filter(|a| a.location == Location::PINNED)
                                .map(|a| a.bytes).sum::<u64>(), if route_checks { 2 * lanes * 2 * rows * 8 * 8 } else { 0 });
                            assert_eq!(allocations.iter().filter(|a| a.location == Location::HOST)
                                .map(|a| a.bytes).sum::<u64>(), if peer_diagnostics { flags * 32 } else { 0 });
                            assert_eq!(allocations.iter().map(|a| &a.id).collect::<std::collections::BTreeSet<_>>().len(), allocations.len());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn route_storage_refuses_invalid_and_overflowing_geometry() {
        let good = PairedRouteBuffers { rows: 64, topk: 8, lanes: 2, input_row_bytes: 0,
            layers: vec![3, 7], peer_diagnostics: false, route_checks: false };
        assert!(good.allocations("", Lifetime::RESIDENT).is_err());
        for case in 0..8 {
            let mut bad = good.clone();
            match case {
                0 => bad.rows = 0,
                1 => bad.topk = 0,
                2 => bad.lanes = 0,
                3 => bad.layers.clear(),
                4 => bad.layers = vec![7, 3],
                5 => bad.rows = u64::MAX,
                6 => bad.lanes = u64::MAX,
                7 => bad.input_row_bytes = u64::MAX,
                _ => unreachable!(),
            }
            assert!(bad.allocations("test", Lifetime::RESIDENT).is_err(), "case {case}");
        }
    }
}

/// Whole and (when `tp2`) TP2-half residency of each routed layer, from the
/// checkpoint's own expert headers.
pub fn expert_costs(catalog: &crate::OfficialV41Catalog, tp2: bool) -> Result<Vec<ExpertCost>> {
    if catalog.exl3().is_none() { return fp8_expert_costs(catalog, tp2); }
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

/// fp8moe (FP8 block-128, NVFP4, MXFP4, BF16) residency over whole H128 blocks.
fn fp8_expert_costs(catalog: &crate::OfficialV41Catalog, tp2: bool) -> Result<Vec<ExpertCost>> {
    use crate::formats::fp8_experts::{Fp8Projection, Slicing};
    let fp8 = catalog.fp8().context("RTX onboarding needs EXL3 or fp8moe expert headers")?;
    let shape = catalog.routed_experts();
    let cost = |tp, rank| -> Result<Bytes2> {
        let resident = Fp8Projection::ALL.iter().try_fold(0u64, |sum, &p| -> Result<u64> {
            let (w, _) = fp8.slice_bytes_with(p, tp, rank, Slicing::Blocks(128))?;
            Ok(sum + (shape.experts * w).max(256) as u64
                + fp8.scale_region_bytes_with(p, tp, rank, Slicing::Blocks(128))?.max(256) as u64)
        })?;
        Ok(Bytes2 { resident, staging: 0 })
    };
    (shape.first_layer..shape.layers).map(|_| Ok(ExpertCost { whole: cost(1, 0)?,
        half: if tp2 { [cost(2, 0)?, cost(2, 1)?] } else { [Bytes2::default(); 2] },
        tp2, spark_ok: true })).collect()
}

/// The installed RTX expert package's arena at `rows` for TP1 or TP2, plus
/// the routed output. `family` names the family's fp8moe package
/// (`fp8-<family>[-format]`); EXL3 packages follow the routed geometry.
/// Missing packages fail closed rather than becoming zero scratch.
pub fn expert_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize, family: &str) -> Result<u64> {
    ensure!([1, 2].contains(&tp), "RTX experts need TP1 or TP2");
    ensure!((1..=4096).contains(&rows), "RTX expert capacity must be 1..=4096 rows");
    if catalog.exl3().is_some() { exl3_workspace(catalog, manifest, rows, tp) }
    else { fp8_workspace(catalog, manifest, rows, tp, family) }
}

/// `fp8-<family>[-format]/tp<tp>` scratch (and a BF16-input sibling's, when
/// installed) plus the TP2 BF16 partial plane.
fn fp8_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize, family: &str) -> Result<u64> {
    let shape = catalog.routed_experts();
    let fp8 = catalog.fp8().context("fp8moe expert catalog missing")?;
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
    let scratch = inventory::fp8moe_scratch_bytes(&value, &format!("tp{tp}"), rows)
        .with_context(|| format!("missing {name} tp{tp} scratch for {rows} rows"))?;
    let bf16 = lib.join("fp8").join(format!("{name}-bf16"));
    let sibling = if bf16.join(format!("tp{tp}")).is_dir() {
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(bf16.join("manifest.json"))?)?;
        inventory::fp8moe_scratch_bytes(&manifest, &format!("tp{tp}"), rows)
            .with_context(|| format!("missing BF16-input {name} scratch"))?
    } else { 0 };
    Ok(scratch.max(sibling) + if tp == 2 { rows * shape.hidden as u64 * 2 } else { 0 })
}

/// The installed `exl3-<family>-k<tiers>/rtx-tp<tp>` arena at `rows`, plus the
/// routed output (F32 partials for TP2, BF16 for one GPU).
fn exl3_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize) -> Result<u64> {
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
