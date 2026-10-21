//! Header-derived EXL3 rank plans shared by startup admission and consumers.
#[cfg(test)]
#[path = "push_exl3/tests.rs"]
mod tests;
use super::{weight_arena::ArenaLayout, weight_validation::ValidationLayout};
use anyhow::{ensure, Context, Result};
use cuteafd_core::SparkSlice;
use cuteafd_loader::{push_load::{builder::{Builder, CheckedPlan, Destination}, PushClient, PushGroup},
    OfficialV41Catalog, V41Exl3Layer, V41Exl3Residency};
use cuteafd_transport::worker_selection::push::{Offer, Placement};
use sha2::{Digest, Sha256};
use std::path::Path;

pub(crate) struct RankPlan {
    pub residencies: Vec<V41Exl3Residency>,
    pub layout: ArenaLayout,
    pub validation: ValidationLayout,
    pub placement: Vec<Placement>,
}

impl RankPlan {
    pub fn new(catalog: &OfficialV41Catalog, layers: &[usize], world: usize, rank: usize,
        admitted: usize) -> Result<Self> {
        let manifest = catalog.exl3().context("push requires EXL3 checkpoint headers")?;
        let residencies = layers.iter().map(|&layer|
            manifest.balanced_residency(V41Exl3Layer::Backbone(layer), world, rank))
            .collect::<Result<Vec<_>>>()?;
        let sizes = residencies.iter().map(|r| r.device_arena_layout().map(|(_, bytes)| bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let layout = ArenaLayout::new(sizes, admitted)?;
        let validation = ValidationLayout::exl3(&residencies, layout.bytes() as u64, usize::MAX)?;
        let layout = layout.with_validation(validation.bytes(), validation.bytes())?;
        let placement = residencies.iter().map(|r| {
            let V41Exl3Layer::Backbone(layer) = r.layer else { anyhow::bail!("draft in Spark rank plan") };
            let slice = SparkSlice {
                unit_map: r.unit_map.as_ref().map(|map| map.fingerprint()),
                layer: u32::try_from(layer)?, first: u32::try_from(r.intermediate_start)?,
                width: u32::try_from(r.intermediate)?, stored_width: u32::try_from(r.intermediate)?,
                tiers: r.tiers.iter().copied().map(u32::try_from).collect::<Result<_, _>>()?,
                projection_counts: r.projection_counts.iter().map(|v| Ok([
                    u32::try_from(v[0])?, u32::try_from(v[1])?, u32::try_from(v[2])?]))
                    .collect::<Result<_, std::num::TryFromIntError>>()?,
            };
            // Include the source-unit seal: the public residency JSON omits it.
            let fingerprint = format!("{:x}", Sha256::digest(serde_json::to_vec(&(r, &slice))?));
            Ok(Placement { slice, fingerprint })
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self { residencies, layout, validation, placement })
    }

    pub fn append(&self, builder: &mut Builder, catalog: &OfficialV41Catalog, rank: u32) -> Result<()> {
        for (index, residency) in self.residencies.iter().enumerate() {
            let range = self.layout.region(index)?;
            let (offsets, _) = residency.device_arena_layout()?;
            builder.exl3_residency(catalog, residency, &offsets, Destination {
                rank, offset: range.start as u64, bytes: range.len() as u64 })?;
        }
        self.validation.append(builder, catalog, rank)?;
        Ok(())
    }

    pub fn checked(&self, catalog: &OfficialV41Catalog, root: &Path, world: usize, rank: usize) -> Result<CheckedPlan> {
        let mut arenas = vec![0; world];
        arenas[rank] = self.layout.logical_bytes() as u64;
        let mut builder = Builder::new(root, arenas);
        self.append(&mut builder, catalog, u32::try_from(rank)?)?;
        Ok(builder.finish()?)
    }
}

fn rank_plans(catalog: &OfficialV41Catalog, layers: &[usize],
    assignments: &[cuteafd_loader::placement::SparkAssignment], offers: &[Offer]) -> Result<Vec<RankPlan>> {
    ensure!(offers.len() == assignments.len(), "push offer world differs from admission");
    let mut plans = Vec::with_capacity(offers.len());
    // Bound CPU/memory concurrency and join in rank order so plan IDs stay deterministic.
    for start in (0..offers.len()).step_by(8) {
        std::thread::scope(|scope| -> Result<()> {
            let jobs: Vec<_> = (start..offers.len().min(start + 8)).map(|rank| scope.spawn(move || {
                let assignment = &assignments[rank];
                let plan = RankPlan::new(catalog, layers, offers.len(), rank,
                    usize::try_from(assignment.resident_bytes)?)?;
                validate_offer(&offers[rank], rank, assignment, &plan.placement,
                    plan.layout.logical_bytes() as u64)?;
                Ok::<_, anyhow::Error>(plan)
            })).collect();
            for job in jobs {
                plans.push(job.join().map_err(|_| anyhow::anyhow!("push rank planner panicked"))??);
            }
            Ok(())
        })?;
    }
    Ok(plans)
}

/// Offers past the Spark assignments are coordinator ranks; `coordinator`
/// appends their entries (it receives the group's root and those offers).
pub(crate) fn prepare(snapshot: &Path, catalog: &OfficialV41Catalog, layers: &[usize],
    assignments: &[cuteafd_loader::placement::SparkAssignment], offers: &[Offer],
    coordinator: Option<&CoordinatorRanks>) -> Result<PushGroup> {
    let started = std::time::Instant::now();
    let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
    let sparks = assignments.len();
    ensure!(offers.len() == sparks + coordinator.map_or(0, |c| c.ranks), "push offer world differs from admission");
    let plans = rank_plans(catalog, layers, assignments, &offers[..sparks])?;
    let planned = started.elapsed();
    // Only validated, worker-reported placements become source/destination plans.
    let mut builder = Builder::new(&root, offers.iter().map(|o| o.arena_bytes).collect());
    for (rank, plan) in plans.iter().enumerate() { plan.append(&mut builder, catalog, u32::try_from(rank)?)?; }
    if let Some(coordinator) = coordinator { (coordinator.append)(&mut builder, catalog, &offers[sparks..])?; }
    let appended = started.elapsed();
    let checked = builder.finish()?;
    let checked_at = started.elapsed();
    ensure!(checked.plan.files.iter().all(|file| cuteafd_loader::push_load::is_push_shard(&root.join(file))),
        "push source is not a SparkNest shard");
    let ranks = offers.iter().map(|offer| Ok(cuteafd_loader::push_load::proto::RankSpec {
        rank: u32::try_from(offer.rank)?, host: offer.host.clone(), arena_bytes: offer.arena_bytes,
    })).collect::<Result<Vec<_>>>()?;
    let source_checked = started.elapsed();
    let group = PushClient::default().create(ranks, checked.plan.files.clone())?;
    let created = started.elapsed();
    checked.upload(&group)?;
    tracing::info!(ranks = offers.len(), rank_plan_ms = planned.as_millis(),
        append_ms = (appended - planned).as_millis(), check_ms = (checked_at - appended).as_millis(),
        source_check_ms = (source_checked - checked_at).as_millis(),
        create_ms = (created - source_checked).as_millis(), upload_ms = (started.elapsed() - created).as_millis(),
        "Spark EXL3 push group prepared");
    Ok(group)
}

/// In-process coordinator-GPU ranks of a Spark push group.
pub(crate) struct CoordinatorRanks {
    pub ranks: usize,
    pub append: Box<dyn Fn(&mut Builder, &OfficialV41Catalog, &[Offer]) -> Result<()> + Send>,
}

pub(crate) fn validate_offer(offer: &Offer, rank: usize,
    assignment: &cuteafd_loader::placement::SparkAssignment, expected: &[Placement], logical: u64) -> Result<()> {
    ensure!(offer.rank == rank && assignment.rank as usize == rank && !offer.host.is_empty()
        && offer.arena_bytes == logical && offer.placement == expected
        && offer.placement.iter().map(|p| &p.slice).eq(assignment.slices.iter()),
        "worker-reported push placement differs from admitted balanced residency");
    Ok(())
}
