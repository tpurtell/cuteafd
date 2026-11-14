//! Finite routed-prefix candidates shared by Qwen planning and loading.
use super::*;
use crate::placement::planner::{Choice, Requires};
use crate::serving_capacity::WeightArenaLayout;

#[derive(Debug, Clone)]
pub struct ExpertCandidate {
    pub layers: Vec<usize>,
    pub arenas: Vec<(usize, WeightArenaLayout)>,
    pub workspace: Vec<Allocation>,
    /// Loader fit witness, including shared mutable owners charged elsewhere.
    pub backend_bytes: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct QwenExperts {
    pub layers: Vec<usize>,
    pub destinations: Vec<u8>,
    pub candidates: Vec<ExpertCandidate>,
}

impl QwenExperts {
    pub fn read(catalog: &crate::OfficialV41Catalog, native_lib: &std::path::Path,
        layers: usize, gpus: usize, rows: u64) -> anyhow::Result<Self> {
        Self::read_with_packages(catalog, native_lib, layers, gpus, rows, true)
    }

    /// Header-only inventories remain useful before exports exist. They do not
    /// claim to have priced an executable backend; runtime always uses `read`.
    pub fn headers(catalog: &crate::OfficialV41Catalog, layers: usize, gpus: usize) -> anyhow::Result<Self> {
        Self::read_with_packages(catalog, std::path::Path::new("/uninstalled/lib.so"), layers, gpus, 64, false)
    }

    fn read_with_packages(catalog: &crate::OfficialV41Catalog, native_lib: &std::path::Path,
        layers: usize, gpus: usize, rows: u64, installed: bool) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        use crate::formats::fp8_experts::Slicing;
        use crate::serving_capacity::{expert_weight_arena_id, fp8_expert_regions};
        ensure!((1..=2).contains(&gpus) && layers > 0 && rows > 0,
            "invalid Qwen resident expert geometry");
        let root = native_lib.parent().unwrap_or(std::path::Path::new("."));
        let shape = catalog.routed_experts();
        ensure!(shape.first_layer == 0 && layers <= shape.layers, "Qwen routed prefix outside checkpoint");
        let packages = if let Some(exl3) = catalog.exl3() {
            let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
            let root = root.join("exl3").join(format!("exl3-qwen4-k{tiers}"));
            (0..gpus).map(|rank| root.join(if gpus == 1 { "rtx-tp1".into() }
                else { format!("rtx-tp2-rank{rank}") })).collect::<Vec<_>>()
        } else if gpus == 2 { dual_packages(catalog, native_lib)?.to_vec() }
        else {
            let tensors = catalog.fp8().context("Qwen expert headers unavailable")?;
            vec![root.join("fp8").join(tensors.format().package_name("qwen4")).join("tp1")]
        };
        let mut workspace = Vec::new();
        let mut regions = Vec::new();
        for rank in 0..gpus {
            let location = Location::coordinator(rank as u8);
            let prefix = format!("qwen.experts.rank{rank}");
            let mut allocations = Vec::new();
            let layer_regions = if let Some(exl3) = catalog.exl3() {
                let capacities = if installed {
                    cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::QWEN4)
                        .capacities(usize::try_from(rows)?)?
                } else { Vec::new() };
                let manifests = capacities.iter().map(|capacity| -> anyhow::Result<serde_json::Value> {
                    let path = packages[rank].join(format!("m{capacity}/v41_exl3.json"));
                    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)
                        .with_context(|| format!("read Qwen expert package {}", path.display()))?)?;
                    let width = exl3.residency(crate::V41Exl3Layer::Backbone(0), gpus, rank)?.intermediate;
                    ensure!(manifest["hidden"].as_u64() == Some(shape.hidden as u64)
                        && manifest["intermediate"].as_u64() == Some(width as u64)
                        && manifest["experts"].as_u64() == Some(shape.experts as u64)
                        && manifest["top_k"].as_u64() == Some(shape.topk as u64)
                        && manifest["capacity"].as_u64() == Some(*capacity as u64)
                        && manifest["output_dtype"].as_str() == Some("fp32"),
                        "Qwen expert package geometry mismatch: {}", path.display());
                    Ok(manifest)
                }).collect::<anyhow::Result<Vec<_>>>()?;
                if installed {
                    allocations.extend(crate::serving_capacity::Exl3WorkspaceLayout::from_manifests(
                        &manifests, true, Exl3Epilogue::Sum)?.allocations(&prefix, location, PlanLifetime::RESIDENT));
                    allocations.push(Allocation::buffer(format!("{prefix}.output"), location,
                        rows * shape.hidden as u64 * 4, PlanLifetime::RESIDENT, Category::Experts, "expert output"));
                }
                (0..layers).map(|layer| {
                    let name = if gpus == 1 { format!("BackboneFull {{ layer: {layer} }}.0") }
                        else { format!("BackboneTp2 {{ layer: {layer}, rank: {rank} }}.0") };
                    Ok((name, exl3.residency(crate::V41Exl3Layer::Backbone(layer), gpus, rank)?
                        .device_arena_layout()?.1))
                }).collect::<anyhow::Result<Vec<_>>>()?
            } else {
                let tensors = catalog.fp8().context("Qwen native experts unavailable")?.for_coordinator();
                if installed {
                let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(
                    packages[rank].parent().context("Qwen package parent")?.join("manifest.json"))?)?;
                let layout = packages[rank].file_name().and_then(|name| name.to_str()).context("Qwen package layout")?;
                let scratch = crate::placement::inventory::fp8moe_scratch_bytes(&manifest, layout, rows)?
                    .context("Qwen native expert capacity unavailable")?.max(256);
                for (name, bytes) in [("scratch", scratch), ("output", rows * shape.hidden as u64 * 2)] {
                    allocations.push(Allocation::buffer(format!("{prefix}.{name}"), location, bytes,
                        PlanLifetime::RESIDENT, Category::Experts, format!("expert {name}")));
                }
                }
                fp8_expert_regions(&tensors, &(0..layers).collect::<Vec<_>>(), gpus, rank,
                    if gpus == 1 { Slicing::Padded } else { Slicing::Blocks(128) })?
            };
            regions.push(layer_regions);
            workspace.extend(allocations);
        }
        if gpus == 2 {
            let demands = crate::serving_capacity::OwnerTp2Rows::new(shape.hidden as u64,
                shape.topk as u64, rows, catalog.exl3().is_some())?.demands()?;
            // The receive demand represents two independent driver requests.
            for demand in demands {
                let copies = if demand.group == "expert receive" { 2 } else { 1 };
                for index in 0..copies {
                    workspace.push(Allocation::buffer(format!("qwen.experts.{}.{}.{index}", demand.gpu, demand.group),
                        Location::coordinator(demand.gpu), demand.bytes / copies,
                        PlanLifetime::RESIDENT, demand.category, demand.group.clone()));
                }
            }
        }
        let mut candidates = Vec::new();
        for count in 0..=layers {
            let selected = (0..count).collect::<Vec<_>>();
            let arenas = if count == 0 { Vec::new() } else {
                (0..gpus).map(|rank| {
                    let selected_regions = if catalog.exl3().is_some() { regions[rank][..count].to_vec() }
                        else { fp8_expert_regions(&catalog.fp8().unwrap().for_coordinator(), &selected,
                            gpus, rank, if gpus == 1 { Slicing::Padded } else { Slicing::Blocks(128) })? };
                    Ok((rank, WeightArenaLayout::new(expert_weight_arena_id(
                        if catalog.exl3().is_some() { "exl3" } else { "fp8" }, rank as i32, &selected_regions)?, selected_regions)?))
                }).collect::<anyhow::Result<Vec<_>>>()?
            };
            let workspace = if count == 0 {
                if gpus == 2 { Self::terminal_owners() } else { Vec::new() }
            } else { workspace.clone() };
            let backend_bytes = (0..gpus).map(|rank| {
                arenas.iter().filter(|(owner, _)| *owner == rank).map(|(_, arena)| arena.bytes() as u64).sum::<u64>()
                    + workspace.iter().filter(|allocation| allocation.view.is_none()
                        && allocation.location == Location::coordinator(rank as u8)
                        && allocation.id.starts_with("qwen.experts.rank")).map(|allocation| allocation.bytes).sum::<u64>()
            }).collect();
            candidates.push(ExpertCandidate { layers: selected, arenas, workspace, backend_bytes });
        }
        Ok(Self { layers: (0..layers).collect(), destinations: (0..gpus as u8).collect(), candidates })
    }

    /// Share the mutable TP1 scratch/output with the independently backed MTP
    /// weight set. Each binding keeps only its private counters/LUT/wire owners.
    pub fn serial_draft(&mut self, catalog: &crate::OfficialV41Catalog)
        -> anyhow::Result<(WeightArenaLayout, Vec<Demand>)> {
        use anyhow::{ensure, Context};
        ensure!(self.destinations == [0], "Qwen serial draft requires one whole owner");
        let exl3 = catalog.exl3().context("Qwen serial draft requires EXL3")?;
        let template = self.candidates.last().context("Qwen expert candidates missing")?.workspace.clone();
        let mut fixed = Vec::new();
        for allocation in template.iter().filter(|allocation| allocation.view.is_none()) {
            let shared = allocation.id.contains(".shared.") || allocation.id.ends_with(".output");
            fixed.push(Demand::new(0, allocation.category,
                if shared { format!("serial EXL3 {}", allocation.id) }
                else { format!("MTP EXL3 {}", allocation.id) }, allocation.bytes, Basis::Exact));
        }
        for candidate in &mut self.candidates {
            candidate.workspace.retain(|allocation| !allocation.id.contains(".shared.")
                && !allocation.id.ends_with(".output"));
        }
        let bytes = exl3.residency(crate::V41Exl3Layer::Dspark(0), 1, 0)?.device_arena_layout()?.1;
        let regions = vec![("Dspark { stage: 0 }.0".to_string(), bytes)];
        let arena = WeightArenaLayout::new(crate::serving_capacity::expert_weight_arena_id("exl3", 0, &regions)?, regions)?;
        Ok((arena, fixed))
    }

    /// Minimal existing PeerExchange: one 256-byte receive/control/abort request
    /// per rank, retained as the residual hop terminal owner without RTX experts.
    pub fn terminal_owners() -> Vec<Allocation> {
        (0..2).flat_map(|rank| ["receive", "control", "abort"].into_iter()
            .chain(crate::serving_capacity::peer_wait_diagnostics().then_some("progress"))
            .map(move |name| Allocation::buffer(format!("qwen.terminal.{rank}.{name}"), Location::coordinator(rank),
                256, PlanLifetime::RESIDENT, Category::Transport, "Qwen terminal exchange"))).collect()
    }

    pub fn pin(onboard: Onboard) -> Pin<u64> {
        match onboard { Onboard::Auto => Pin::Auto, Onboard::Layers(count) => Pin::Exact(count as u64) }
    }

    pub fn extend(&self, variables: &mut Vec<Variable>,
        components: &mut Vec<Component>, maximum: u64) -> Result<(), Refusal> {
        if self.layers.is_empty() || self.candidates.is_empty()
            || self.candidates.iter().any(|candidate| candidate.layers.len() > self.layers.len()
                || candidate.layers != self.layers[..candidate.layers.len()])
            || self.candidates.windows(2).any(|pair| pair[0].layers.len() >= pair[1].layers.len()) {
            return Err(Refusal::InvalidFact("invalid Qwen routed prefix candidates".into()));
        }
        let variable = variables.len();
        variables.push(Variable::Onboard { layers: self.layers.clone(), destination: self.destinations.clone(),
            counts: self.candidates.iter().map(|candidate| candidate.layers.len()).collect() });
        let first_component = components.len();
        for (option, candidate) in self.candidates.iter().enumerate() {
            for (rank, layout) in &candidate.arenas {
                for mut component in weight_arena_components(layout, Location::coordinator(*rank as u8),
                    PlanLifetime::RESIDENT, maximum, cuteafd_core::memory_layout::MemoryKind::Experts)? {
                    component.activation = Activation::choice(variable, option);
                    components.push(component);
                }
            }
            for allocation in &candidate.workspace {
                let mut component = Component::allocation(allocation.clone(),
                    ByteCurve::fixed(allocation.bytes, maximum), false, maximum);
                component.activation = Activation::choice(variable, option);
                components.push(component);
            }
        }
        let mut owners = components.split_off(first_component);
        crate::placement::planner::fixed_request_tails(&mut owners, maximum, &[])?;
        components.extend(owners);
        Ok(())
    }

    pub fn requirements(&self, sparks: usize, variable: usize) -> Vec<Requires> {
        self.candidates.iter().enumerate().filter(|(_, candidate)|
            sparks == 0 && candidate.layers.len() != self.layers.len()).map(|(option, _)|
                Requires { when: Choice { variable, option }, any_of: vec![] }).collect()
    }

    pub fn remap_devices(&mut self, devices: &[i32]) -> anyhow::Result<()> {
        use anyhow::ensure;
        ensure!(devices.len() == self.destinations.len(), "Qwen expert device map differs from owners");
        for candidate in &mut self.candidates {
            for (rank, arena) in &mut candidate.arenas {
                let regions = arena.views().iter().map(|(name, view)| (name.clone(), view.bytes())).collect::<Vec<_>>();
                // Preserve the declared order, not the BTreeMap's lexical order.
                let mut regions = regions;
                regions.sort_by_key(|(name, _)| arena.view(name).unwrap().offset());
                let format = if regions[0].0.starts_with("Backbone") { "exl3" } else { "fp8" };
                *arena = WeightArenaLayout::new(crate::serving_capacity::expert_weight_arena_id(
                    format, devices[*rank], &regions)?, regions)?;
            }
        }
        Ok(())
    }

    pub fn selected(&self, layers: &[usize]) -> Result<&ExpertCandidate, Refusal> {
        self.candidates.iter().find(|candidate| candidate.layers == layers)
            .ok_or_else(|| Refusal::InvalidFact("Qwen admitted an undeclared routed prefix".into()))
    }
}
