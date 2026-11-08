//! Physical local expert owners. Pair payload rows and routers are target-owned.
use super::{GlmLocalFormat, LocalInventory};
use crate::{families::glm5::GlmDsaConfig, OfficialV41Catalog, V41Exl3Layer};
use crate::placement::planner::{Allocation, Lifetime, Location, View};
use crate::plan::exlr8::{self, residency::{ExpertSet, MetadataInput}, ExecutionVariant, PrefillActivations};
use crate::serving_capacity::{expert_weight_arena_id, PackedWeightRegions, WeightArenaLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Category;
use std::{collections::BTreeMap, path::Path};

/// Supplied by the same mount/push decision that selects the serving loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalStorage { Mount, Push }

impl LocalStorage {
    pub fn for_snapshot(snapshot: &Path) -> Result<Self> {
        match crate::push_load::namespace(snapshot) {
            Ok(_) => Ok(Self::Push),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound
                && error.raw_os_error().is_none() && snapshot.exists() => Ok(Self::Mount),
            Err(error) => Err(error).context("discovering checkpoint storage namespace"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocalPhysicalInput {
    pub layers: Vec<usize>,
    pub max_rows: usize,
    pub lanes: usize,
    pub storage: LocalStorage,
    pub route_checks: bool,
    pub peer_diagnostics: bool,
    pub activations: PrefillActivations,
}

#[derive(Debug, Clone, Default)]
pub struct LocalPhysicalInventory {
    /// Physical backings only. Sum this list, never view extents.
    pub owners: Vec<Allocation>,
    /// Positive view extents are reconciliation evidence, not physical charges.
    pub views: Vec<Allocation>,
}

struct LayerStorage {
    bytes: usize,
    /// Receipt suffix, offset within the layer, bytes.
    views: Vec<(String, usize, usize)>,
}

/// Parsed headers and installed execution manifests reused for finite prefix
/// choices. `read` validates the largest candidate set once; `inventory` performs
/// only region packing and itemization, without reopening checkpoint headers.
pub struct LocalPhysicalSource {
    weights: [BTreeMap<usize, LayerStorage>; 2],
    backend: LocalPhysicalInventory,
    rows: usize,
    activations: PrefillActivations,
    topk: usize,
    world: usize,
    format: GlmLocalFormat,
}

impl LocalPhysicalSource {
    pub fn world(&self) -> usize { self.world }

    #[cfg(test)]
    pub(super) fn test_source(world: usize, layers: std::ops::Range<usize>, rows: usize) -> Self {
        Self { weights: std::array::from_fn(|rank| if rank < world {
            layers.clone().map(|layer| (layer, LayerStorage { bytes: 4096,
                views: vec![(String::new(), 0, 4096)] })).collect()
        } else { BTreeMap::new() }), backend: LocalPhysicalInventory::default(),
            rows, activations: PrefillActivations::A16, topk: 8, world, format: GlmLocalFormat::Exlr8 }
    }

    pub fn read(cfg: &GlmDsaConfig, catalog: &OfficialV41Catalog, local: &LocalInventory,
        input: &LocalPhysicalInput, exlr8: Option<&MetadataInput>) -> Result<Self> {
        validate_input(cfg, input)?;
        ensure!((1..=2).contains(&local.world)
            && (local.world == 2 || matches!(local.format, GlmLocalFormat::Exlr8 | GlmLocalFormat::Exl3)),
            "GLM local physical source has no executor for this format/world");
        let mut source = Self { weights: Default::default(), backend: Default::default(),
            rows: input.max_rows, activations: input.activations, topk: cfg.topk, world: local.world,
            format: local.format };
        if input.layers.is_empty() { return Ok(source); }
        let programs = if local.format == GlmLocalFormat::Exlr8 {
            Some(serde_json::from_slice::<serde_json::Value>(&std::fs::read(&local.package)?)?)
        } else { None };
        for rank in 0..source.world {
            let prefix = format!("glm.expert.routed.rank.{rank}");
            let location = Location::coordinator(rank as u8);
            let mut unit_limit = None;
            for &layer in &input.layers {
                let weight = match local.format {
                    GlmLocalFormat::Exl3 => {
                        let plan = catalog.exl3().context("EXL3 local headers missing")?
                            .residency(V41Exl3Layer::Backbone(layer), local.world, rank)?;
                        let (offsets, bytes) = plan.device_arena_layout()?;
                        LayerStorage { bytes,
                            views: plan.buffers.iter().zip(offsets).enumerate().map(|(index, (spec, offset))|
                                (index.to_string(), offset, spec.bytes.max(16))).collect() }
                    }
                    GlmLocalFormat::Exlr8 => {
                        let (bytes, units) = exlr8.context("EXLR8 selected metadata missing")?
                            .admission_geometry(layer, local.world, rank, ExpertSet::Routed)?;
                        ensure!(unit_limit.is_none_or(|previous| previous == units),
                            "EXLR8 local layers have inconsistent unit limits");
                        unit_limit = Some(units);
                        LayerStorage { bytes,
                            views: vec![(String::new(), 0, bytes)] }
                    }
                    GlmLocalFormat::Nvfp4 { w4a4: true } => {
                        use crate::formats::fp8_experts::Fp8Projection;
                        let tensors = catalog.fp8().context("NVFP4 local headers missing")?.for_coordinator()
                            .for_layer_metadata(layer)?;
                        let (_, slicing) = crate::formats::fp8_experts::exact_layout_directory(&local.package, &tensors, 2, rank);
                        tensors.validate_layer(layer)?;
                        let mut planes = Vec::new();
                        for projection in Fp8Projection::ALL {
                            planes.push(tensors.slice_bytes_with(projection, 2, rank, slicing)?.0
                                .checked_mul(cfg.experts).context("NVFP4 weights overflow")?);
                            planes.push(tensors.scale_region_bytes_with(projection, 2, rank, slicing)?);
                        }
                        let packed = PackedWeightRegions::new(planes.iter().copied())?;
                        LayerStorage { bytes: packed.bytes,
                            views: packed.regions.into_iter().enumerate().filter(|(_, region)| !region.is_empty())
                                .map(|(plane, region)| (format!("plane.{plane}"), region.start, region.len())).collect() }
                    }
                    GlmLocalFormat::Nvfp4 { w4a4: false } => anyhow::bail!("GLM NVFP4 requires W4A4"),
                };
                source.weights[rank].insert(layer, weight);
            }
            match local.format {
                GlmLocalFormat::Exl3 => source.backend.exl3_workspace(&local.package, &prefix, location, input.max_rows)?,
                GlmLocalFormat::Exlr8 => {
                    for (index, (variant, rows)) in [(ExecutionVariant::DecodeCompact, 64),
                        (ExecutionVariant::DecodeGrouped, 64), (ExecutionVariant::Prefill, input.max_rows)]
                        .into_iter().enumerate() {
                        if index == 2 && input.max_rows == 64 { continue; }
                        let layout = exlr8::execution_workspace_from_manifest(programs.as_ref().unwrap(),
                            cuteafd_ffi::exlr8::KernelFamily::Glm, rows, variant,
                            unit_limit.context("EXLR8 local unit limit missing")?, input.activations)?;
                        for allocation in layout.allocations(&format!("{prefix}.execution.{index}"), location, Lifetime::TARGET)? {
                            source.backend.record(allocation);
                        }
                    }
                    source.backend.add(format!("{prefix}.output"), location,
                        product(input.max_rows, cfg.hidden, 2)?, Category::Experts)?;
                }
                GlmLocalFormat::Nvfp4 { .. } => {
                    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(
                        local.package.parent().context("NVFP4 package parent missing")?.join("manifest.json"))?)?;
                    let tensors = catalog.fp8().context("NVFP4 local headers missing")?.for_coordinator();
                    let (directory, _) = crate::formats::fp8_experts::exact_layout_directory(&local.package, &tensors, 2, rank);
                    let layout = directory.file_name().and_then(|name| name.to_str()).context("NVFP4 layout name missing")?;
                    let scratch = crate::placement::inventory::fp8moe_scratch_bytes(&manifest, layout, input.max_rows as u64)
                        .context("NVFP4 scratch extent missing")?;
                    let (scratch, output, _) = crate::serving_capacity::fp8moe_tp2_workspace(
                        usize::try_from(scratch)?, cfg.hidden, input.max_rows)?;
                    source.backend.add(format!("{prefix}.scratch"), location, scratch, Category::Experts)?;
                    source.backend.add(format!("{prefix}.output"), location, output, Category::Experts)?;
                }
            }
        }
        Ok(source)
    }

    pub fn inventory(&self, input: &LocalPhysicalInput) -> Result<LocalPhysicalInventory> {
        ensure!(input.max_rows == self.rows && input.activations == self.activations
            && (1..=4).contains(&input.lanes) && input.layers.windows(2).all(|p| p[0] < p[1]),
            "GLM local physical source differs from requested geometry");
        if input.layers.is_empty() { return Ok(LocalPhysicalInventory::default()); }
        let mut out = LocalPhysicalInventory::default();
        for rank in 0..self.world {
            let prefix = format!("glm.expert.routed.rank.{rank}");
            let location = Location::coordinator(rank as u8);
            let layers = input.layers.iter().map(|layer| self.weights[rank].get(layer)
                .with_context(|| format!("GLM local physical layer {layer} not parsed"))).collect::<Result<Vec<_>>>()?;
            let packed = PackedWeightRegions::new(layers.iter().map(|layer| layer.bytes))?;
            let arena = if input.storage == LocalStorage::Mount && self.format != GlmLocalFormat::Exlr8 {
                let mut regions = Vec::new();
                for (&layer, weight) in input.layers.iter().zip(&layers) {
                    match self.format {
                        GlmLocalFormat::Exl3 => regions.push((exl3_region(layer, self.world, rank), weight.bytes)),
                        GlmLocalFormat::Nvfp4 { .. } => {
                            for (suffix, _, bytes) in &weight.views {
                                regions.push((format!("{layer}.{}", suffix.strip_prefix("plane.")
                                    .context("NVFP4 plane missing")?), *bytes));
                            }
                        }
                        GlmLocalFormat::Exlr8 => unreachable!(),
                    }
                }
                let format = if self.format == GlmLocalFormat::Exl3 { "exl3" } else { "fp8" };
                let layout = WeightArenaLayout::new(expert_weight_arena_id(format, rank as i32, &regions)?, regions)?;
                out.record(Allocation::buffer(layout.id(), location, u64::try_from(layout.bytes())?,
                    Lifetime::RESIDENT, Category::Weights, "weight arena"));
                Some(layout)
            } else { None };
            let mut rank_backing: Option<String> = None;
            for ((&layer, weight), region) in input.layers.iter().zip(layers).zip(packed.regions) {
                let name = format!("{prefix}.weight.{layer}");
                match input.storage {
                    LocalStorage::Mount => {
                        if let Some(arena) = &arena {
                            let layer_offset = if self.format == GlmLocalFormat::Exl3 {
                                let view = arena.view(&exl3_region(layer, self.world, rank)).context("EXL3 layer view missing")?;
                                out.add_view(name.clone(), location, view.bytes(), arena.id().into(), view.offset())?;
                                view.offset()
                            } else { 0 };
                            for (suffix, offset, bytes) in &weight.views {
                                let offset = if self.format == GlmLocalFormat::Exl3 {
                                    layer_offset.checked_add(*offset).context("EXL3 view offset overflow")?
                                } else {
                                    arena.view(&format!("{layer}.{}", suffix.strip_prefix("plane.").context("NVFP4 plane missing")?))
                                        .context("NVFP4 plane view missing")?.offset()
                                };
                                out.add_view(suffix_name(&name, suffix), location, *bytes, arena.id().into(), offset)?;
                            }
                        } else {
                            let backing = format!("{name}.backing");
                            out.add(backing.clone(), location, weight.bytes, Category::Experts)?;
                            for (suffix, offset, bytes) in &weight.views {
                                out.add_view(suffix_name(&name, suffix), location, *bytes, backing.clone(), *offset)?;
                            }
                        }
                    }
                    LocalStorage::Push => {
                        let backing = if let Some(backing) = &rank_backing { backing.clone() } else {
                            let id = format!("{}.backing", suffix_name(&name, &weight.views[0].0));
                            out.add(id.clone(), location, packed.bytes, Category::Experts)?;
                            rank_backing = Some(id.clone());
                            id
                        };
                        for (suffix, offset, bytes) in &weight.views {
                            out.add_view(suffix_name(&name, suffix), location, *bytes, backing.clone(),
                                region.start.checked_add(*offset).context("local weight view offset overflow")?)?;
                        }
                    }
                }
            }
        }
        // Weight backings and aliases already exist when KV is first allocated;
        // executor, pair and check storage binds afterward.
        for weight in out.owners.iter_mut().chain(&mut out.views) {
            weight.lifetime = Lifetime::RESIDENT;
        }
        out.owners.extend(self.backend.owners.iter().cloned());
        out.views.extend(self.backend.views.iter().cloned());
        if self.world == 1 { return Ok(out); }
        let routes = PackedRoutes::new(input.max_rows, self.topk);
        let peer = crate::serving_capacity::PeerStorageLayout::new(2 * input.lanes, routes.bytes, input.peer_diagnostics)?;
        for rank in 0..2 {
            let location = Location::coordinator(rank);
            for slot in 0..2 * input.lanes {
                out.add(format!("glm.expert.routed.routes.rank.{rank}.peer.recv.{slot}"), location, peer.receive_bytes, Category::Experts)?;
            }
            for (name, bytes) in [("control", peer.control_bytes), ("abort", 256), ("progress", peer.progress_bytes)] {
                if bytes > 0 { out.add(format!("glm.expert.routed.routes.rank.{rank}.peer.{name}"), location, bytes, Category::Experts)?; }
            }
        }
        if input.route_checks {
            let bytes = route_check_bytes(input.max_rows, self.topk)?;
            for slot in 0..input.lanes { for layer in &input.layers { for rank in 0..2 {
                out.add(format!("glm.expert.routed.checks.{slot}.{layer}.{rank}"), Location::PINNED, bytes, Category::Experts)?;
            } } }
        }
        Ok(out)
    }
}

impl LocalPhysicalInventory {
    pub fn new(cfg: &GlmDsaConfig, catalog: &OfficialV41Catalog, local: &LocalInventory,
        input: &LocalPhysicalInput, exlr8: Option<&MetadataInput>) -> Result<Self> {
        LocalPhysicalSource::read(cfg, catalog, local, input, exlr8)?.inventory(input)
    }

    /// Real per-rank retained loader peak, excluding pair routes and checks.
    /// Both loaders have zero additional device staging.
    pub fn backend_bytes(&self, rank: usize) -> Result<u64> {
        ensure!(rank < 2, "invalid local backend rank {rank}");
        self.owners.iter().filter(|owner| owner.view.is_none()
            && owner.location == Location::coordinator(rank as u8)
            && !owner.id.starts_with("glm.expert.routed.routes."))
            .try_fold(0u64, |sum, owner| sum.checked_add(owner.bytes).context("local backend bytes overflow"))
    }

    fn record(&mut self, allocation: Allocation) {
        if allocation.view.is_some() { self.views.push(allocation); } else { self.owners.push(allocation); }
    }

    fn add(&mut self, id: String, location: Location, bytes: usize, category: Category) -> Result<()> {
        ensure!(bytes > 0 && !self.owners.iter().any(|owner| owner.id == id), "empty or duplicate local owner {id}");
        self.owners.push(Allocation::buffer(id, location, u64::try_from(bytes)?, Lifetime::TARGET, category, "resident expert owners"));
        Ok(())
    }

    fn add_view(&mut self, id: String, location: Location, bytes: usize, backing: String, offset: usize) -> Result<()> {
        let mut allocation = Allocation::buffer(id, location, u64::try_from(bytes)?, Lifetime::TARGET, Category::Experts, "resident expert owners");
        allocation.view = Some(View { backing, offset: u64::try_from(offset)? });
        self.views.push(allocation);
        Ok(())
    }

    fn exl3_workspace(&mut self, package: &Path, prefix: &str, location: Location, rows: usize) -> Result<()> {
        let capacities = cuteafd_core::Exl3RowPolicy::Nearest.capacities(rows)?;
        let manifests = capacities.iter().map(|capacity| -> Result<serde_json::Value> {
            Ok(serde_json::from_slice(&std::fs::read(package.join(format!("m{capacity}/v41_exl3.json")))?)?)
        }).collect::<Result<Vec<_>>>()?;
        let layout = crate::serving_capacity::Exl3WorkspaceLayout::from_manifests(&manifests, true,
            crate::serving_capacity::Exl3Epilogue::Sum)?;
        let mut shared = BTreeMap::new();
        for owner in layout.owners.iter().filter(|owner| owner.name.starts_with("shared.")) {
            let id = format!("{prefix}.execution.0.{}.backing", owner.name);
            self.add(id.clone(), location, usize::try_from(owner.bytes)?, Category::Experts)?;
            shared.insert(owner.name.clone(), (id, usize::try_from(owner.bytes)?));
        }
        for (index, manifest) in manifests.iter().enumerate() {
            let name = format!("{prefix}.execution.{index}");
            let capacity = manifest["capacity"].as_u64().context("EXL3 capacity missing")?;
            let capacity_prefix = format!("capacity.{capacity}.");
            let mut storage = 0;
            for owner in layout.owners.iter().filter(|owner| owner.name.starts_with(&capacity_prefix)) {
                let suffix = owner.name.strip_prefix(&capacity_prefix).unwrap();
                if suffix == "wire" {
                    self.add(format!("{name}.wire"), location, usize::try_from(owner.bytes)?, Category::Experts)?;
                } else {
                    self.add(format!("{name}.storage.{storage}"), location, usize::try_from(owner.bytes)?, Category::Experts)?;
                    storage += 1;
                }
            }
            for (key, (backing, bytes)) in &shared {
                self.add_view(format!("{name}.{key}"), location, *bytes, backing.clone(), 0)?;
            }
        }
        let hidden = manifests[0]["hidden"].as_u64().context("EXL3 hidden missing")? as usize;
        self.add(format!("{prefix}.output"), location, product(rows, hidden, 4)?, Category::Experts)
    }
}

fn validate_input(cfg: &GlmDsaConfig, input: &LocalPhysicalInput) -> Result<()> {
    ensure!((64..=4096).contains(&input.max_rows) && (1..=4).contains(&input.lanes)
        && input.layers.windows(2).all(|p| p[0] < p[1])
        && input.layers.iter().all(|layer| (cfg.first_moe_layer..cfg.layers).contains(layer)),
        "invalid GLM local physical layer/row/slot selection");
    Ok(())
}

fn suffix_name(name: &str, suffix: &str) -> String {
    if suffix.is_empty() { name.into() } else { format!("{name}.{suffix}") }
}

fn exl3_region(layer: usize, world: usize, rank: usize) -> String {
    if world == 1 { format!("BackboneFull {{ layer: {layer} }}.0") }
    else { format!("BackboneTp2 {{ layer: {layer}, rank: {rank} }}.0") }
}

fn product(a: usize, b: usize, c: usize) -> Result<usize> {
    a.checked_mul(b).and_then(|n| n.checked_mul(c)).context("GLM local physical extent overflow")
}

pub fn route_check_bytes(rows: usize, topk: usize) -> Result<usize> {
    ensure!(rows > 0 && topk > 0, "empty route check shape");
    product(rows, topk, 8)
}

/// Shared packed-route ABI used by the actual pair allocation and its planner.
#[derive(Debug, Clone, Copy)]
pub struct PackedRoutes { pub section: usize, pub weights: usize, pub bytes: usize }
impl PackedRoutes {
    pub fn new(rows: usize, topk: usize) -> Self {
        let section = rows * topk * 4;
        let weights = section.next_multiple_of(16);
        Self { section, weights, bytes: (weights + section).next_multiple_of(16) }
    }
    pub fn with_input_bytes(rows: usize, topk: usize, input_row_bytes: usize) -> usize {
        Self::new(rows, topk).bytes + rows * input_row_bytes
    }
}
