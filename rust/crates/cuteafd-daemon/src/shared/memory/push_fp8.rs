//! Planar native expert images, preserving the mount loader's projection order.
#[cfg(test)]
#[path = "push_fp8/tests.rs"]
pub(crate) mod tests;
use super::{push_initialize, weight_arena::ArenaLayout, weight_views::RegionLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{DType, SparkSlice};
use cuteafd_loader::{
    formats::fp8_experts::{ExpertFormat, Fp8ExpertTensors, Fp8Projection, Slicing},
    push_load::{
        builder::{Builder, CheckedPlan, Destination},
        PushClient, PushGroup,
    },
    OfficialV41Catalog,
};
use cuteafd_transport::worker_selection::push::{Offer, Placement};
use sha2::{Digest, Sha256};
use std::{ops::Range, path::Path};

pub(crate) struct LayerPlan {
    pub layer: usize,
    pub slicing: Slicing,
    pub slice: SparkSlice,
    pub planes: [Range<usize>; 6],
    pub initialize: push_initialize::Layer<'static>,
}
pub(crate) struct RankPlan {
    pub layout: ArenaLayout,
    pub layers: Vec<LayerPlan>,
    pub placement: Vec<Placement>,
    pub host_scratch: usize,
}

pub(crate) fn slice(
    tensors: &Fp8ExpertTensors,
    layer: usize,
    world: usize,
    rank: usize,
    slicing: Slicing,
) -> Result<SparkSlice> {
    let tensors = tensors.for_layer_metadata(layer)?;
    let (first, width) = tensors.rank_range_with(world, rank, slicing)?;
    Ok(SparkSlice {
        unit_map: Some(tensors.unit_map(world, rank, slicing)?.fingerprint()),
        layer: u32::try_from(layer)?,
        first: u32::try_from(first)?,
        width: u32::try_from(width)?,
        stored_width: u32::try_from(tensors.rank_width(world, rank, slicing)?)?,
        tiers: vec![],
        projection_counts: vec![],
    })
}

impl RankPlan {
    pub fn new(
        catalog: &OfficialV41Catalog,
        layers: &[(usize, Slicing)],
        world: usize,
        rank: usize,
        admitted: usize,
    ) -> Result<Self> {
        Self::from_tensors(catalog.fp8().context("push requires native expert headers")?,
            layers, world, rank, admitted)
    }

    pub(crate) fn from_tensors(tensors: &Fp8ExpertTensors, layers: &[(usize, Slicing)],
        world: usize, rank: usize, admitted: usize) -> Result<Self> {
        ensure!(
            !layers.is_empty() && layers.windows(2).all(|v| v[0].0 < v[1].0),
            "push native layers must be sorted and unique"
        );
        let mut sizes = Vec::new();
        let mut plans = Vec::new();
        for &(layer, slicing) in layers {
            let tensors = tensors.for_layer_metadata(layer)?;
            tensors.validate_layer(layer)?;
            let planes = RegionLayout::fp8(&tensors, world, rank, slicing)?;
            // Do not silently widen the worker-reported checkpoint admission.
            ensure!(
                planes.padding_bytes() == 0,
                "native push plane alignment exceeds mount admission"
            );
            let offsets = planes.addresses(256)?;
            let mut ranges = Vec::new();
            for (index, projection) in Fp8Projection::ALL.into_iter().enumerate() {
                let (w, _) = tensors.slice_bytes_with(projection, world, rank, slicing)?;
                for (plane, bytes) in [
                    (
                        2 * index,
                        w.checked_mul(tensors.shape().experts)
                            .context("native push weight extent overflow")?,
                    ),
                    (
                        2 * index + 1,
                        tensors.scale_region_bytes_with(projection, world, rank, slicing)?,
                    ),
                ] {
                    let start = if bytes == 0 {
                        0
                    } else {
                        usize::try_from(offsets[plane] - 256)?
                    };
                    ranges.push(start..start + bytes);
                }
            }
            sizes.push(planes.bytes());
            plans.push(LayerPlan {
                layer,
                slicing,
                slice: slice(&tensors, layer, world, rank, slicing)?,
                planes: ranges.try_into().unwrap(),
                initialize: push_initialize::Layer::Fp8 {
                    widen: vec![],
                    scalars: vec![],
                    inputs: None,
                    expand: vec![],
                },
            });
        }
        let layout = ArenaLayout::new(sizes, admitted)?;
        let mut placement = Vec::new();
        for (region, plan) in plans.iter_mut().enumerate() {
            let base = layout.region(region)?.start;
            for range in &mut plan.planes {
                range.start += base;
                range.end += base;
            }
            let tensors = tensors.for_layer_metadata(plan.layer)?;
            plan.initialize = initialize(&tensors, plan, world, rank)?;
            let mut hash = Sha256::new();
            hash.update(serde_json::to_vec(&plan.slice)?);
            // Equivalent padded/exact recipes must agree when they produce the
            // same admitted map, source windows and destination planes.
            hash.update(format!("{:?}", plan.planes));
            // Seal all source headers, including replicated scalars, not only
            // the public rank width. No source payload is read here.
            for projection in Fp8Projection::ALL {
                hash.update(format!(
                    "{:?}",
                    tensors.projection_slices(projection, world, rank, plan.slicing)?
                ));
                for expert in 0..tensors.shape().experts {
                    for name in source_names(&tensors, plan.layer, expert, projection) {
                        hash.update(format!("{:?}", tensors.push_source(&name)?));
                    }
                }
            }
            placement.push(Placement {
                slice: plan.slice.clone(),
                fingerprint: format!("{:x}", hash.finalize()),
            });
        }
        // Post-Done checks reuse one scalar vector or one bounded widening
        // chunk at a time. Include both simultaneous gate/up check vectors.
        let host_scratch = plans
            .iter()
            .map(|plan| match &plan.initialize {
                push_initialize::Layer::Fp8 {
                    widen,
                    scalars,
                    inputs,
                    expand,
                } => {
                    let widening = widen
                        .iter()
                        .map(|range| {
                            range.len().min(1024 * 1024) + range.len().min(1024 * 1024) / 2
                        })
                        .max()
                        .unwrap_or(0);
                    let scalar = scalars.iter().map(|range| range.len()).max().unwrap_or(0);
                    let inputs = inputs
                        .as_ref()
                        .map_or(0, |(gate, up, _)| gate.len() + up.len());
                    let expanding = expand
                        .iter()
                        .map(|(_, _, width, pitch)| {
                            let rows = (1024 * 1024 / pitch).max(1);
                            rows * (width + pitch)
                        })
                        .max()
                        .unwrap_or(0);
                    widening.max(scalar).max(inputs).max(expanding)
                }
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        Ok(Self {
            layout,
            layers: plans,
            placement,
            host_scratch,
        })
    }

    pub fn append(
        &self,
        builder: &mut Builder,
        catalog: &OfficialV41Catalog,
        rank: u32,
        world: usize,
    ) -> Result<()> {
        self.append_tensors(builder, catalog.fp8().context("native expert headers absent")?,
            rank, world, rank as usize, 0)
    }

    /// The destination group rank is independent of the checkpoint TP shard.
    pub(crate) fn append_tensors(&self, builder: &mut Builder, tensors: &Fp8ExpertTensors,
        rank: u32, world: usize, shard: usize, base: u64) -> Result<()> {
        let destination = |rank, plane: &Range<usize>, offset, bytes| -> Result<Destination> {
            let mut destination = destination(rank, plane, offset, bytes)?;
            destination.offset = destination.offset.checked_add(base).context("FP8 destination overflow")?;
            Ok(destination)
        };
        for layer in &self.layers {
            let tensors = tensors.for_layer_metadata(layer.layer)?;
            let experts = tensors.shape().experts;
            for (index, projection) in Fp8Projection::ALL.into_iter().enumerate() {
                let [weight, scale] =
                    tensors.projection_slices(projection, world, shard, layer.slicing)?;
                let (w_bytes, s_bytes) =
                    tensors.slice_bytes_with(projection, world, shard, layer.slicing)?;
                for expert in 0..experts {
                    let names = source_names(&tensors, layer.layer, expert, projection);
                    let (source, _) = tensors.push_source(&names[0])?;
                    builder.fp8_matrix(
                        &source,
                        weight,
                        destination(rank, &layer.planes[2 * index], expert * w_bytes, w_bytes)?,
                    )?;
                    if tensors.format() == ExpertFormat::Bf16 {
                        continue;
                    }
                    let (source, dtype) = tensors.push_source(&names[1])?;
                    let mut image = scale;
                    // A grid whose rows pad to a wider pitch lands packed (so its
                    // runs merge into whole slots) and spreads after Done; a BF16
                    // source lands in the first half of its packed FP32 extent.
                    let packed = packed_rows(&scale);
                    let (at, extent) = if packed {
                        image.destination_pitch = image.width;
                        image.destination_bytes = image.rows * image.width;
                        (expert * image.rows * image.width, image.rows * image.width)
                    } else { (expert * s_bytes, s_bytes) };
                    if dtype == DType::Bf16 {
                        for value in [
                            &mut image.source_offset,
                            &mut image.source_pitch,
                            &mut image.destination_pitch,
                            &mut image.width,
                            &mut image.destination_bytes,
                        ] {
                            ensure!(*value % 2 == 0, "unaligned BF16 native scale view");
                            *value /= 2;
                        }
                    }
                    builder.fp8_matrix(
                        &source,
                        image,
                        destination(rank, &layer.planes[2 * index + 1], at, extent)?,
                    )?;
                    if tensors.format() == ExpertFormat::Nvfp4 {
                        for (name, offset) in names[2..].iter().zip([
                            experts * s_bytes + expert * 4,
                            experts * s_bytes + experts * 4 + expert * 4,
                        ]) {
                            let (source, dtype) = tensors.push_source(name)?;
                            ensure!(
                                dtype == DType::F32 && source.bytes == 4,
                                "NVFP4 scalar must be F32"
                            );
                            builder.contiguous(
                                &source,
                                0,
                                4,
                                destination(rank, &layer.planes[2 * index + 1], offset, 4)?,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub fn checked(
        &self,
        catalog: &OfficialV41Catalog,
        root: &Path,
        world: usize,
        rank: usize,
    ) -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        arenas[rank] = self.layout.bytes() as u64;
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, catalog, u32::try_from(rank)?, world)?;
        Ok(builder.finish()?)
    }
}

fn destination(
    rank: u32,
    plane: &Range<usize>,
    offset: usize,
    bytes: usize,
) -> Result<Destination> {
    ensure!(
        offset
            .checked_add(bytes)
            .is_some_and(|end| end <= plane.len()),
        "native push exceeds plane"
    );
    Ok(Destination {
        rank,
        offset: (plane.start + offset) as u64,
        bytes: bytes as u64,
    })
}
fn source_names(
    tensors: &Fp8ExpertTensors,
    layer: usize,
    expert: usize,
    projection: Fp8Projection,
) -> Vec<String> {
    let name = tensors.name(layer, expert, projection);
    let mut names = vec![name.clone()];
    if tensors.format() != ExpertFormat::Bf16 {
        names.push(format!(
            "{name}{}",
            if tensors.format().packed_fp4() {
                "_scale"
            } else {
                "_scale_inv"
            }
        ));
    }
    if tensors.format() == ExpertFormat::Nvfp4 {
        names.push(format!("{name}_scale_2"));
        names.push(format!(
            "{}.input_scale",
            name.strip_suffix(".weight").unwrap()
        ));
    }
    names
}
/// A scale grid whose execution rows pad to a wider pitch (MXFP4 down
/// projections pad K/32 scale bytes to 4): pushed packed, spread after Done.
fn packed_rows(image: &cuteafd_loader::formats::fp8_experts::Fp8MatrixSlice) -> bool {
    image.width > 0 && image.width < image.destination_pitch
}

fn initialize(
    tensors: &Fp8ExpertTensors,
    layer: &LayerPlan,
    world: usize,
    rank: usize,
) -> Result<push_initialize::Layer<'static>> {
    let mut widen = Vec::new();
    let mut scalars = Vec::new();
    let mut inputs = Vec::new();
    let mut expand = Vec::new();
    let experts = tensors.shape().experts;
    for (index, projection) in Fp8Projection::ALL.into_iter().enumerate() {
        if tensors.format() == ExpertFormat::Bf16 {
            continue;
        }
        let (_, grid) = tensors.slice_bytes_with(projection, world, rank, layer.slicing)?;
        let plane = &layer.planes[2 * index + 1];
        let [_, image] = tensors.projection_slices(projection, world, rank, layer.slicing)?;
        let packed = packed_rows(&image);
        if packed {
            ensure!(image.rows * image.destination_pitch == grid, "padded scale grid differs from its slice bytes");
            expand.push((plane.start..plane.start + experts * grid, experts * image.rows, image.width,
                image.destination_pitch));
        }
        if tensors.format() == ExpertFormat::Fp8Block128 {
            for expert in 0..experts {
                let name = tensors.name(layer.layer, expert, projection);
                if tensors.push_source(&format!("{name}_scale_inv"))?.1 == DType::Bf16 {
                    // A packed grid widens as one packed FP32 extent before it spreads.
                    widen.push(if packed { plane.start + expert * image.rows * image.width
                        ..plane.start + (expert + 1) * image.rows * image.width }
                        else { plane.start + expert * grid..plane.start + (expert + 1) * grid });
                }
            }
        }
        if tensors.format() == ExpertFormat::Nvfp4 {
            let start = plane.start + experts * grid;
            scalars.push(start..plane.end);
            inputs.push(start + experts * 4..plane.end);
        }
    }
    let inputs = (!inputs.is_empty()).then(|| (inputs[0].clone(), inputs[1].clone(), layer.layer));
    Ok(push_initialize::Layer::Fp8 {
        widen,
        scalars,
        inputs,
        expand,
    })
}

fn checked_plan(
    root: &Path,
    catalog: &OfficialV41Catalog,
    assignments: &[cuteafd_loader::placement::SparkAssignment],
    offers: &[Offer],
    coordinator: Option<&super::push_exl3::CoordinatorRanks>,
) -> Result<CheckedPlan> {
    let sparks = assignments.len();
    ensure!(
        offers.len() == sparks + coordinator.map_or(0, |c| c.ranks),
        "native push world differs from admission"
    );
    let tensors = catalog.fp8().context("native expert headers absent")?;
    let mut plans = Vec::new();
    for (rank, (offer, assignment)) in offers[..sparks].iter().zip(assignments).enumerate() {
        let layers = assignment
            .slices
            .iter()
            .map(|expected| {
                let layer = expected.layer as usize;
                let tensors = tensors.for_layer_metadata(layer)?;
                let slicing = [Slicing::Padded, tensors.exact_slicing(sparks)]
                    .into_iter()
                    .find(|&slicing| {
                        slice(&tensors, layer, sparks, rank, slicing)
                            .is_ok_and(|slice| &slice == expected)
                    })
                    .context("native push worker slice differs from admission")?;
                Ok((layer, slicing))
            })
            .collect::<Result<Vec<_>>>()?;
        let plan = RankPlan::new(
            catalog,
            &layers,
            sparks,
            rank,
            usize::try_from(assignment.resident_bytes)?,
        )?;
        super::push_exl3::validate_offer(
            offer,
            rank,
            assignment,
            &plan.placement,
            plan.layout.bytes() as u64,
        )?;
        plans.push(plan);
    }
    let mut builder = Builder::new(
        root,
        offers.iter().map(|offer| offer.arena_bytes).collect(),
    );
    for (rank, plan) in plans.iter().enumerate() {
        plan.append(&mut builder, catalog, u32::try_from(rank)?, sparks)?;
    }
    if let Some(coordinator) = coordinator { (coordinator.append)(&mut builder, Some(catalog), &offers[sparks..])?; }
    Ok(builder.finish()?)
}

pub(crate) fn prepare(
    snapshot: &Path,
    catalog: &OfficialV41Catalog,
    assignments: &[cuteafd_loader::placement::SparkAssignment],
    offers: &[Offer],
    coordinator: Option<&super::push_exl3::CoordinatorRanks>,
) -> Result<PushGroup> {
    let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
    let checked = checked_plan(&root, catalog, assignments, offers, coordinator)?;
    ensure!(
        checked
            .plan
            .files
            .iter()
            .all(|file| cuteafd_loader::push_load::is_push_shard(&root.join(file))),
        "native push source is not a SparkNest shard"
    );
    let ranks = offers
        .iter()
        .map(|offer| {
            Ok(cuteafd_loader::push_load::proto::RankSpec {
                rank: u32::try_from(offer.rank)?,
                host: offer.host.clone(),
                arena_bytes: offer.arena_bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let group = PushClient::default().create(ranks, checked.plan.files.clone())?;
    checked.upload(&group)?;
    Ok(group)
}
