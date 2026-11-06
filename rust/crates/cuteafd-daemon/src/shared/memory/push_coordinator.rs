//! Coordinator (RTX) ranks of a push group: the two coordinator GPUs join the
//! Spark group as ranks `sparks` and `sparks + 1`, each with its own ring and
//! admitted device arena. Plans reuse the shared builder; delivery reuses the
//! shared consumer and post-Done initialization. Nothing publishes before the
//! whole group commits; failures drain every unpublished owner before startup fails.
use super::{weight_arena::ArenaLayout, weight_validation::ValidationLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{push_load::builder::{Builder, CheckedPlan, Destination}, OfficialV41Catalog, V41Exl3Layer,
    V41Exl3Residency};

/// One coordinator GPU's admitted expert layers, laid out as today's mount
/// arenas, one region per layer: EXL3 residencies of TP2 halves or of whole
/// (world 1) layers and draft stages, native EXLR8 world-1/2 unit arenas, or native
/// MXFP4 TP2 planes (the Spark native images at world 2, packed after Done).
pub(crate) struct CoordinatorExpertPlan {
    pub residencies: Vec<V41Exl3Residency>,
    pub exlr8: Vec<cuteafd_loader::plan::exlr8::residency::Residency>,
    /// Native planes per layer (empty for EXL3).
    pub native: Vec<cuteafd_loader::push_load::native::NativePlanes>,
    pub native_layers: Vec<crate::shared::experts::layer::ExpertLayer>,
    pub fp8: Option<super::push_fp8::RankPlan>,
    pub layers: Vec<usize>,
    pub gpu: usize,
    pub world: usize,
    pub layout: ArenaLayout,
    pub validation: ValidationLayout,
    pub scales: super::push_native::ScalePlan,
}

impl CoordinatorExpertPlan {
    pub fn new(catalog: &OfficialV41Catalog, layers: &[usize], gpu: usize, admitted: usize) -> Result<Self> {
        ensure!(gpu < 2, "coordinator TP2 has two GPUs");
        if catalog.exl3().is_none() { return Self::native(catalog, layers, 2, gpu, admitted); }
        let manifest = catalog.exl3().context("coordinator TP2 push requires EXL3 checkpoint headers")?;
        let residencies = layers.iter().map(|&layer| manifest.residency(V41Exl3Layer::Backbone(layer), 2, gpu))
            .collect::<Result<Vec<_>>>()?;
        Self::exl3(residencies, layers.to_vec(), gpu, admitted)
    }

    pub fn exlr8(catalog: &OfficialV41Catalog, manifest: &str, layers: &[usize], world: usize, gpu: usize,
        admitted: usize) -> Result<Self> {
        ensure!((1..=2).contains(&world) && gpu < world && !layers.is_empty()
            && layers.windows(2).all(|ids| ids[0] < ids[1]), "EXLR8 RTX needs ordered layers on one or two GPUs");
        let plan = super::push_exlr8::RankPlan::new(catalog.snapshot(), manifest, layers, world, gpu, admitted)?;
        let validation = ValidationLayout::exl3(&[], plan.layout.bytes() as u64, usize::MAX)?;
        Ok(Self { residencies: Vec::new(), exlr8: plan.residencies, native: Vec::new(), native_layers: Vec::new(), fp8: None,
            layers: layers.to_vec(), gpu, world, layout: plan.layout, validation, scales: Default::default() })
    }

    /// Whole (world 1) EXL3 residencies on one coordinator GPU, backbone
    /// layers and draft stages, as the caller planned them (the local mount
    /// loader's own residencies).
    pub fn whole(residencies: Vec<V41Exl3Residency>, admitted: usize) -> Result<Self> {
        ensure!(residencies.iter().all(|r| r.world == 1 && r.rank == 0),
            "a whole-layer coordinator plan takes world-1 residencies");
        let layers = residencies.iter().filter_map(|r| match r.layer { V41Exl3Layer::Backbone(layer) => Some(layer),
            _ => None }).collect();
        Self::exl3(residencies, layers, 0, admitted)
    }

    pub fn whole_native(catalog: &OfficialV41Catalog, layers: &[usize], admitted: usize) -> Result<Self> {
        Self::native(catalog, layers, 1, 0, admitted)
    }

    fn exl3(residencies: Vec<V41Exl3Residency>, layers: Vec<usize>, gpu: usize, admitted: usize) -> Result<Self> {
        ensure!(residencies.iter().all(|r| r.unit_map.is_none()), "coordinator EXL3 keeps the legacy residency");
        let sizes = residencies.iter().map(|r| r.device_arena_layout().map(|(_, bytes)| bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let layout = ArenaLayout::new(sizes, admitted)?;
        let validation = ValidationLayout::exl3(&residencies, layout.bytes() as u64, usize::MAX)?;
        let layout = layout.with_validation(validation.bytes(), validation.bytes())?;
        let world = residencies.first().map_or(1, |r| r.world);
        Ok(Self { residencies, exlr8: Vec::new(), native: Vec::new(), native_layers: Vec::new(), fp8: None, layers, gpu, world,
            layout, validation, scales: Default::default() })
    }

    /// Native MXFP4 halves: each layer's four planes for intermediate half
    /// `gpu` (W1/W3 rows, W2 columns), exactly the mount's TP2 residency.
    fn native(catalog: &OfficialV41Catalog, layers: &[usize], world: usize, gpu: usize, admitted: usize) -> Result<Self> {
        ensure!((1..=2).contains(&world) && gpu < world && !layers.is_empty()
            && layers.windows(2).all(|ids| ids[0] < ids[1]), "coordinator native layers must be ordered on one or two GPUs");
        if let Some(tensors) = catalog.fp8() {
            let tensors = tensors.for_coordinator();
            let selected = layers.iter().map(|&layer| (layer, tensors.exact_slicing(world))).collect::<Vec<_>>();
            let plan = super::push_fp8::RankPlan::from_tensors(&tensors, &selected, world, gpu, admitted)?;
            let validation = ValidationLayout::exl3(&[], plan.layout.bytes() as u64, usize::MAX)?;
            return Ok(Self { residencies: Vec::new(), exlr8: Vec::new(), native: Vec::new(), native_layers: Vec::new(),
                layers: layers.to_vec(), gpu, world, layout: plan.layout.clone(), validation,
                scales: Default::default(), fp8: Some(plan) });
        }
        use crate::shared::experts::layer::ExpertLayer;
        let selections = layers.iter().map(|&layer| if world == 1 { ExpertLayer::BackboneFull { layer } }
            else { ExpertLayer::BackboneTp2 { layer, rank: gpu } }).collect::<Vec<_>>();
        let mut plan = Self::native_selected(catalog, &selections, gpu, admitted)?;
        plan.layers = layers.to_vec();
        plan.world = world;
        Ok(plan)
    }

    pub fn native_selected(catalog: &OfficialV41Catalog,
        selections: &[crate::shared::experts::layer::ExpertLayer], gpu: usize, admitted: usize) -> Result<Self> {
        use crate::shared::experts::layer::ExpertLayer;
        ensure!(!selections.is_empty() && catalog.exl3().is_none() && catalog.fp8().is_none()
            && catalog.nvfp4().is_none(), "coordinator native push takes MXFP4 selections");
        let mut seen = Vec::new();
        let native = selections.iter().map(|&selection| {
            ensure!(!seen.contains(&selection), "duplicate coordinator native selection");
            seen.push(selection);
            let experts = match selection {
                ExpertLayer::BackboneFull { .. } => catalog.routed_experts().experts,
                ExpertLayer::BackboneTp2 { rank, .. } | ExpertLayer::DsparkTp2 { rank, .. } => {
                    ensure!(rank == gpu && gpu < 2, "native TP2 selection differs from GPU");
                    if matches!(selection, ExpertLayer::DsparkTp2 { .. }) { catalog.routed_experts().draft_experts }
                    else { catalog.routed_experts().experts }
                }
                _ => anyhow::bail!("unsupported coordinator native selection {selection:?}"),
            };
            let staging = catalog.expert_staging(selection.expert(0))?;
            let width = staging.intermediate_size();
            cuteafd_loader::push_load::native::NativePlanes::with_scale_codes(catalog.routed_experts().hidden,
                width, width.div_ceil(128) * 128, experts, false, catalog.mxfp4_scale_codes())
        }).collect::<Result<Vec<_>>>()?;
        let layout = ArenaLayout::new(native.iter().map(|p| p.bytes), admitted)?;
        let validation = ValidationLayout::exl3(&[], layout.bytes() as u64, usize::MAX)?;
        let mut generated = Vec::new();
        let mut scale_layers = Vec::new();
        for (region, (&selection, planes)) in selections.iter().zip(&native).enumerate() {
            let base = layout.region(region)?.start;
            generated.extend(planes.generated_ranges().into_iter().map(|r| base + r.start..base + r.end));
            scale_layers.push((selection, planes.clone(), base));
        }
        let layout = layout.with_generated(generated)?;
        let scales = super::push_native::ScalePlan::selected(catalog, scale_layers)?;
        Ok(Self { residencies: Vec::new(), exlr8: Vec::new(), native, native_layers: selections.to_vec(),
            fp8: None, layers: Vec::new(), gpu, world: 1, layout, validation, scales })
    }

    /// EXLR8 keeps its source/index metadata through serving and constructs one
    /// layer's relocated descriptors at a time after Done. Mount upload banks
    /// are not used by push; its receive rings are admitted separately below.
    pub fn exlr8_host_bytes(&self) -> Result<usize> {
        let mut retained = 0usize;
        let mut scratch = 0usize;
        for residency in &self.exlr8 {
            let host = residency.host_bytes();
            retained = retained.checked_add(host).context("EXLR8 retained host overflow")?;
            scratch = scratch.max(residency.descriptor_host_bytes()?);
        }
        retained.checked_add(scratch.max(self.fp8.as_ref().map_or(0, |p| p.host_scratch)))
            .context("coordinator host admission overflow")
    }

    /// Source and index metadata carried into the published EXLR8 owners.
    pub fn retained_host_bytes(&self) -> Result<usize> {
        self.exlr8.iter().try_fold(0usize, |bytes, residency|
            bytes.checked_add(residency.host_bytes()).context("EXLR8 retained host overflow"))
    }

    pub fn transient_host_bytes(&self) -> Result<usize> {
        self.host_bytes()?.checked_sub(self.retained_host_bytes()?)
            .context("coordinator retained host exceeds total")
    }

    /// Expert-owned host storage, independent of the family DenseArena and ring.
    pub fn host_bytes(&self) -> Result<usize> {
        let scales = self.scales.host_peak()?;
        self.exlr8_host_bytes()?.checked_add(self.validation.bytes())
            .and_then(|bytes| bytes.checked_add(scales))
            .context("coordinator expert host admission overflow")
    }

    /// All ranks and both DenseArenas per rank coexist with their receive rings.
    /// Dense staging is supplied only once its records are complete, before offers.
    pub fn admit_host(plans: &[Option<&Self>], dense: &[crate::shared::experts::rtx::dense::HostStaging],
        slots: usize, slot_bytes: usize, available: u64) -> Result<usize> {
        ensure!(dense.is_empty() || dense.len() == plans.len(), "dense host owners differ from coordinator ranks");
        let rings = slots.checked_mul(slot_bytes).and_then(|bytes| bytes.checked_mul(plans.len()))
            .context("coordinator push ring host overflow")?;
        let peak = plans.iter().try_fold(rings, |bytes, plan| {
            let extra = plan.map_or(Ok(0), Self::host_bytes)?;
            bytes.checked_add(extra).context("coordinator push host admission overflow")
        })?;
        let peak = dense.iter().try_fold(peak, |bytes, staging|
            bytes.checked_add(staging.bytes()?).context("coordinator dense host admission overflow"))?;
        ensure!(u64::try_from(peak)? <= available,
            "coordinator push experts, dense staging and receive rings need {peak} host bytes, available {available}");
        Ok(peak)
    }

    /// Device scratch the post-Done native plane transform needs (0 for EXL3).
    pub fn scratch_bytes(&self) -> usize { self.native.iter().map(|p| p.scratch_bytes()).max().unwrap_or(0) }
    pub fn logical_bytes(&self) -> u64 { self.scales.logical_bytes(&self.layout) }

    /// Entries for group rank `rank` (the coordinator GPU's rank in the group).
    pub fn append(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, rank: u32) -> Result<()> {
        self.append_at(builder, catalog, rank, 0)
    }

    /// As `append`, with this plan's logical space starting at `base` of the
    /// rank's (the coordinator's dense operands come first).
    pub fn append_at(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, rank: u32, base: u64) -> Result<()> {
        if let Some(plan) = &self.fp8 {
            return plan.append_tensors(builder, &catalog.fp8().context("FP8 expert headers absent")?.for_coordinator(),
                rank, self.world, self.gpu, base);
        }
        if !self.exlr8.is_empty() {
            for (index, residency) in self.exlr8.iter().enumerate() {
                let range = self.layout.region(index)?;
                builder.exlr8(catalog.snapshot(), residency, Destination {
                    rank, offset: base + range.start as u64, bytes: range.len() as u64 })?;
            }
            return Ok(());
        }
        if !self.native.is_empty() {
            for (index, (planes, &selection)) in self.native.iter().zip(&self.native_layers).enumerate() {
                let range = self.layout.region(index)?;
                let destination = Destination { rank, offset: base + range.start as u64, bytes: range.len() as u64 };
                for expert in 0..planes.experts {
                    let world = if matches!(selection, crate::shared::experts::layer::ExpertLayer::BackboneFull { .. }) { 1 } else { 2 };
                    let checkpoint_rank = if world == 1 { 0 } else { self.gpu };
                    let names = catalog.expert_staging(selection.expert(expert))?.tensor_names().to_vec();
                    builder.v41_resident(catalog, &names, planes, expert, world, checkpoint_rank as u32, destination, None)?;
                }
            }
            self.scales.append(builder, rank, base + self.scales.base(&self.layout))?;
            return Ok(());
        }
        for (index, residency) in self.residencies.iter().enumerate() {
            let range = self.layout.region(index)?;
            let (offsets, _) = residency.device_arena_layout()?;
            builder.exl3_residency(catalog, residency, &offsets, Destination {
                rank, offset: base + range.start as u64, bytes: range.len() as u64 })?;
        }
        let validation = ValidationLayout::exl3(&self.residencies, base + self.layout.bytes() as u64, usize::MAX)?;
        ensure!(validation.bytes() == self.validation.bytes(), "shifted validation differs");
        validation.append(builder, catalog, rank)
    }

    /// This rank's checked entries alone, against a group of `world` ranks.
    pub fn checked(&self, catalog: &OfficialV41Catalog, root: &std::path::Path, world: usize, rank: usize)
        -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        *arenas.get_mut(rank).context("coordinator rank outside group")? = self.logical_bytes();
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, catalog, u32::try_from(rank)?)?;
        Ok(builder.finish()?)
    }
}

