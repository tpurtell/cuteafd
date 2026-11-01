//! Native MXFP4/NVFP4 rank images, sealed against worker-reported admission.
#[cfg(test)]
#[path = "push_native/tests.rs"]
mod tests;
use super::{weight_arena::ArenaLayout, push_consumer::PushDestination};
use anyhow::{ensure, Context, Result};
use cuteafd_core::SparkSlice;
use cuteafd_loader::{OfficialV41Catalog, V41ExpertSelection,
    push_load::{builder::{Builder, CheckedPlan, Destination}, native::NativePlanes, PushClient, PushGroup}};
use cuteafd_transport::worker_selection::push::{Offer, Placement};
use sha2::{Digest, Sha256};
use std::{ops::Range, path::Path};

/// Device-free description of bounded scale sources and their generated planes.
#[derive(Default)]
pub(crate) struct ScalePlan {
    steps: Vec<(u64, ScaleStep)>,
    bytes: u64,
}
struct ScaleStep {
    rects: Vec<crate::shared::experts::rtx::dense::Rect>,
    outputs: Vec<ScaleOutput>,
}
struct ScaleOutput {
    rect: usize,
    at: usize,
    stored_rows: usize,
    start_row: usize,
    offset: usize,
    bytes: usize,
}

impl ScalePlan {
    pub fn new(catalog: &OfficialV41Catalog,
        layers: impl IntoIterator<Item = (usize, NativePlanes, usize)>, world: usize, rank: usize) -> Result<Self> {
        use crate::shared::experts::rtx::dense::{Rect, STEP_BYTES};
        let mut plan = Self::default();
        for (layer, planes, base) in layers {
            if !planes.scale_codes { continue; }
            let geometry = catalog.routed_experts();
            let shard = cuteafd_core::tp_units::LayerShard::new(geometry.intermediate, 32, layer, world, rank)?;
            ensure!(shard.width == planes.intermediate, "native scale shard differs from resident planes");
            for expert in 0..planes.experts {
                let (names, _) = names(catalog, layer, expert, world, rank)?;
                // A step must complete within one source tensor: projections
                // need not be adjacent in safetensors file order.
                for slot in [4, 3, 5] {
                    let down = slot == 5;
                    let tensor = catalog.tensor(&names[slot])?;
                    let row_bytes = tensor.metadata.shape[1];
                    let rows = if down { 0..planes.hidden }
                        else { shard.source_range() };
                    let cols = if down { shard.first / 32..(shard.first + shard.width) / 32 }
                        else { 0..row_bytes };
                    let rect = Rect { name: names[slot].clone(), path: catalog.snapshot().join(&tensor.shard),
                        base: tensor.metadata.byte_offset, row_bytes, rows, cols };
                    let (offset, bytes) = planes.expert_plane(if down { 3 } else { 1 }, expert)?;
                    let output = ScaleOutput { rect: 0, at: 0,
                        stored_rows: if down { planes.hidden } else { 2 * planes.kernel_intermediate },
                        start_row: if slot == 3 { planes.kernel_intermediate } else { 0 }, offset: base + offset, bytes };
                    ensure!(rect.bytes() <= STEP_BYTES, "native expert scales exceed bounded staging");
                    let start = plan.bytes.next_multiple_of(256);
                    plan.bytes = start.checked_add(rect.bytes() as u64 + 1).context("scale-code staged space overflow")?;
                    plan.steps.push((start, ScaleStep { rects: vec![rect], outputs: vec![output] }));
                }
            }
        }
        Ok(plan)
    }
    pub fn base(&self, layout: &ArenaLayout) -> u64 { (layout.logical_bytes() as u64 + 1).next_multiple_of(256) }
    pub fn logical_bytes(&self, layout: &ArenaLayout) -> u64 {
        if self.steps.is_empty() { layout.logical_bytes() as u64 } else { self.base(layout) + self.bytes }
    }
    pub fn host_peak(&self) -> usize {
        if self.steps.is_empty() { 0 } else { crate::shared::experts::rtx::dense::MAX_OPEN_STEPS
            * crate::shared::experts::rtx::dense::STEP_BYTES + (1 << 20) }
    }
    pub fn append(&self, builder: &mut Builder, rank: u32, base: u64) -> Result<()> {
        for (start, step) in &self.steps {
            let mut at = base + start;
            for rect in &step.rects {
                crate::shared::experts::rtx::dense::append(rect, builder, rank, at)?;
                at += rect.bytes() as u64;
            }
        }
        Ok(())
    }
    pub fn destination<'a>(&self, device: super::device::Device<'a>, layout: ArenaLayout,
        expected: super::weight_arena::ExpectedCoverage) -> Result<StagedArena<'a>> {
        use crate::shared::experts::rtx::dense::{DenseArena, Job, RankLayout, Step};
        let base = self.base(&layout);
        let (resident_expected, staged_expected) = expected.split_at(base)?;
        let resident = super::weight_arena::LoadingArena::new(device, layout, resident_expected)?;
        let staged = if self.steps.is_empty() { None } else {
            let mut steps = Vec::new();
            for (_, template) in &self.steps {
                let jobs = template.outputs.iter().map(|output| Ok(Job::ScaleCodes {
                    rect: output.rect, at: output.at, stored_rows: output.stored_rows, start_row: output.start_row,
                    dst: resident.destination(output.offset, output.bytes)?,
                })).collect::<Result<Vec<_>>>()?;
                steps.push(Step { device: device.id, rects: template.rects.clone(), jobs });
            }
            let bound = RankLayout::new(Vec::new(), steps, device.id)?;
            ensure!(bound.host_bytes == self.bytes && bound.base == 0, "bound scale plan changed its footprint");
            Some(DenseArena::new(device, bound, staged_expected)?)
        };
        Ok(StagedArena { resident, staged, base, bytes: self.bytes })
    }
}

pub(crate) struct StagedArena<'a> {
    resident: super::weight_arena::LoadingArena<'a>,
    staged: Option<crate::shared::experts::rtx::dense::DenseArena<'a>>,
    base: u64,
    bytes: u64,
}
impl<'a> super::push_consumer::PushDestination for StagedArena<'a> {
    type Delivered = super::weight_arena::DeliveredArena<'a>;
    fn belongs_to(&self, device: super::device::Device<'_>) -> bool { self.resident.belongs_to(device) }
    fn logical_bytes(&self) -> usize { if self.staged.is_some() { (self.base + self.bytes) as usize }
        else { self.resident.logical_bytes() } }
    fn device_bytes(&self) -> usize { self.resident.device_bytes() }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        if offset < self.base { self.resident.check_fragment(offset, bytes) }
        else { self.staged.as_ref().context("scale tail absent")?.check_fragment(offset - self.base, bytes) }
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        self.resident.destination(offset, bytes)
    }
    fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        if offset < self.base { self.resident.copy_validation(offset, bytes) }
        else { self.staged.as_mut().context("scale tail absent")?.copy_validation(offset - self.base, bytes) }
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> { self.resident.copied(offset, bytes) }
    fn done(self) -> Result<Self::Delivered> {
        if let Some(staged) = self.staged { staged.done()?; }
        self.resident.done()
    }
}

pub(crate) struct LayerPlan {
    pub slice: SparkSlice,
    pub planes: NativePlanes,
    pub scalars: Range<usize>,
}
pub(crate) struct RankPlan {
    pub layers: Vec<LayerPlan>,
    pub layout: ArenaLayout,
    pub placement: Vec<Placement>,
    pub scratch: usize,
    pub host_scratch: usize,
    pub scales: ScalePlan,
}

fn names(catalog: &OfficialV41Catalog, layer: usize, expert: usize, world: usize, rank: usize)
    -> Result<(Vec<String>, usize)> {
    let selection = if world == 1 { V41ExpertSelection::BackboneFull { layer, expert } }
        else { V41ExpertSelection::BackboneTp { layer, expert, world, rank } };
    if catalog.nvfp4().is_some() {
        let staging = catalog.nvfp4_expert_staging(selection)?;
        Ok((staging.tensor_names().to_vec(), staging.intermediate_size()))
    } else {
        let staging = catalog.expert_staging(selection)?;
        Ok((staging.tensor_names().to_vec(), staging.intermediate_size()))
    }
}

impl RankPlan {
    pub fn new(catalog: &OfficialV41Catalog, slices: &[SparkSlice], world: usize, rank: usize,
        admitted: usize) -> Result<Self> {
        ensure!(world > 0 && rank < world && !slices.is_empty()
            && slices.windows(2).all(|s| s[0].layer < s[1].layer), "invalid native push rank layers");
        ensure!(catalog.exl3().is_none() && catalog.fp8().is_none(), "native push cannot convert checkpoint format");
        let geometry = catalog.routed_experts();
        let mut layers = Vec::new();
        let mut placement = Vec::new();
        let mut scalar_bytes = 0usize;
        for slice in slices {
            ensure!(slice.unit_map.is_none() && slice.tiers.is_empty() && slice.projection_counts.is_empty(),
                "native push slice has compressed metadata");
            let planes = NativePlanes::with_scale_codes(geometry.hidden as usize, slice.width as usize,
                slice.stored_width as usize, geometry.experts as usize, catalog.nvfp4().is_some(),
                catalog.nvfp4().is_none() && catalog.mxfp4_scale_codes())?;
            let shard = cuteafd_core::tp_units::LayerShard::new(geometry.intermediate, 32,
                slice.layer as usize, world, rank)?;
            ensure!(slice.first as usize == shard.first && planes.intermediate == shard.width,
                "native push slice differs from layer ownership");
            let mut hash = Sha256::new();
            hash.update(serde_json::to_vec(slice)?);
            hash.update(format!("{planes:?}"));
            for expert in 0..planes.experts {
                let (names, width) = names(catalog, slice.layer as usize, expert, world, rank)?;
                ensure!(width == planes.intermediate, "native push header width differs from admission");
                for name in names {
                    let source = Builder::v41_source(catalog, &name)?;
                    hash.update(serde_json::to_vec(&(name, source.path.strip_prefix(catalog.snapshot())?,
                        source.offset, source.bytes))?);
                }
            }
            let end = scalar_bytes.checked_add(planes.scalar_bytes()).context("native scalar tail overflow")?;
            layers.push(LayerPlan { slice: slice.clone(), planes, scalars: scalar_bytes..end });
            scalar_bytes = end;
            placement.push(Placement { slice: slice.clone(), fingerprint: format!("{:x}", hash.finalize()) });
        }
        let layout = ArenaLayout::new(layers.iter().map(|l| l.planes.bytes), admitted)?
            .with_validation(scalar_bytes, scalar_bytes)?;
        ensure!(layout.bytes() == admitted, "native push planes differ from mount admission");
        let scratch = layers.iter().map(|l| l.planes.scratch_bytes()).max().unwrap_or(0);
        // Scalar validation and launch-vector construction coexist with the delivered tail.
        let vectors = layers.iter().filter(|l| l.planes.nvfp4)
            .map(|l| l.planes.experts.checked_mul(128).context("native scalar scratch overflow"))
            .collect::<Result<Vec<_>>>()?.into_iter().max().unwrap_or(0);
        let mut generated = Vec::new();
        let mut scale_layers = Vec::new();
        for (region, layer) in layers.iter().enumerate() {
            let base = layout.region(region)?.start;
            generated.extend(layer.planes.generated_ranges().into_iter().map(|r| base + r.start..base + r.end));
            scale_layers.push((layer.slice.layer as usize, layer.planes.clone(), base));
        }
        let layout = layout.with_generated(generated)?;
        let scales = ScalePlan::new(catalog, scale_layers, world, rank)?;
        let host_scratch = scalar_bytes.checked_add(vectors).and_then(|n| n.checked_add(scales.host_peak()))
            .context("native host scratch overflow")?;
        Ok(Self { layers, layout, placement, scratch, host_scratch, scales })
    }

    pub fn append(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, world: usize, rank: usize) -> Result<()> {
        for (region, layer) in self.layers.iter().enumerate() {
            let range = self.layout.region(region)?;
            let destination = Destination { rank: u32::try_from(rank)?, offset: range.start as u64, bytes: range.len() as u64 };
            let scalars = layer.planes.nvfp4.then_some(Destination { rank: destination.rank,
                offset: (self.layout.bytes() + layer.scalars.start) as u64, bytes: layer.scalars.len() as u64 });
            for expert in 0..layer.planes.experts {
                let (names, _) = names(catalog, layer.slice.layer as usize, expert, world, rank)?;
                builder.v41_resident(catalog, &names, &layer.planes, expert, world as u32, rank as u32, destination, scalars)?;
            }
        }
        self.scales.append(builder, u32::try_from(rank)?, self.scales.base(&self.layout))?;
        Ok(())
    }

    pub fn logical_bytes(&self) -> u64 { self.scales.logical_bytes(&self.layout) }

    pub fn checked(&self, catalog: &OfficialV41Catalog, root: &Path, world: usize, rank: usize) -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        arenas[rank] = self.logical_bytes();
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, catalog, world, rank)?;
        Ok(builder.finish()?)
    }
}

fn checked_plan(root: &Path, catalog: &OfficialV41Catalog,
    assignments: &[cuteafd_loader::placement::SparkAssignment], offers: &[Offer],
    coordinator: Option<&super::push_exl3::CoordinatorRanks>) -> Result<CheckedPlan> {
    let sparks = assignments.len();
    ensure!(sparks > 0 && offers.len() == sparks + coordinator.map_or(0, |c| c.ranks),
        "native push offer world differs from admission");
    let mut plans = Vec::new();
    for (rank, (offer, assignment)) in offers[..sparks].iter().zip(assignments).enumerate() {
        let slices = offer.placement.iter().map(|p| p.slice.clone()).collect::<Vec<_>>();
        let plan = RankPlan::new(catalog, &slices, sparks, rank, usize::try_from(assignment.resident_bytes)?)?;
        super::push_exl3::validate_offer(offer, rank, assignment, &plan.placement, plan.logical_bytes())?;
        plans.push(plan);
    }
    let mut builder = Builder::new(root, offers.iter().map(|o| o.arena_bytes).collect());
    for (rank, plan) in plans.iter().enumerate() { plan.append(&mut builder, catalog, sparks, rank)?; }
    if let Some(coordinator) = coordinator { (coordinator.append)(&mut builder, Some(catalog), &offers[sparks..])?; }
    Ok(builder.finish()?)
}

pub(crate) fn prepare(snapshot: &Path, catalog: &OfficialV41Catalog,
    assignments: &[cuteafd_loader::placement::SparkAssignment], offers: &[Offer],
    coordinator: Option<&super::push_exl3::CoordinatorRanks>) -> Result<PushGroup> {
    let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
    let checked = checked_plan(&root, catalog, assignments, offers, coordinator)?;
    ensure!(checked.plan.files.iter().all(|file| cuteafd_loader::push_load::is_push_shard(&root.join(file))),
        "native push source is not a SparkNest shard");
    let ranks = offers.iter().map(|o| Ok(cuteafd_loader::push_load::proto::RankSpec {
        rank: u32::try_from(o.rank)?, host: o.host.clone(), arena_bytes: o.arena_bytes,
    })).collect::<Result<Vec<_>>>()?;
    let group = PushClient::default().create(ranks, checked.plan.files.clone())?;
    checked.upload(&group)?;
    Ok(group)
}
