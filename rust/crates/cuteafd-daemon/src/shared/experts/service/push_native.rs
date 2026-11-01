//! Optional native Spark load; execution views are constructed only after Done.
use super::{ExpertLayer, ExpertWeights, NativeExpertServiceConfig};
use crate::shared::{experts::layer::ExpertLoadBudget, memory::{device::Device,
    push_consumer, push_initialize, push_native::RankPlan,
    weight_arena::ExpectedCoverage}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{NativeLibrary, push_ring::{RegisteredRing, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES}};
use cuteafd_loader::{OfficialV41Catalog, push_load::{JoinArgs, PushClient}};
use cuteafd_transport::worker_selection::push::{self, Offer};
use std::{net::TcpStream, os::fd::AsRawFd, rc::Rc};

pub(super) fn load<'a>(library: &'a NativeLibrary, catalog: &OfficialV41Catalog,
    config: &NativeExpertServiceConfig, resident: usize, stream: &mut TcpStream)
    -> Result<Option<Vec<ExpertWeights<'a>>>> {
    let planned = (|| -> Result<_> {
        let (root, host) = cuteafd_loader::push_load::namespace(&config.snapshot)?;
        let selections = config.resident_layers(catalog.routed_experts().layers)?.into_iter()
            .map(|layer| config.selection(layer)).collect::<Result<Vec<_>>>()?;
        let mut slices = Vec::new();
        let mut budgets = Vec::new();
        for &selection in &selections {
            let info = selection.select_info(library, config.compiled_capacity()?, catalog.nvfp4().is_some())?;
            let budget = ExpertWeights::plan(library, catalog, selection)?;
            slices.push(cuteafd_core::SparkSlice { unit_map: None, layer: selection.layer() as u32,
                first: u32::try_from(selection.native_shard()?.first)?, width: info.logical_intermediate,
                stored_width: info.kernel_intermediate, tiers: vec![], projection_counts: vec![] });
            budgets.push(budget);
        }
        let plan = RankPlan::new(catalog, &slices, config.world, config.rank, resident)?;
        let checked = plan.checked(catalog, &root, config.world, config.rank)?;
        ensure!(checked.plan.files.iter().all(|file| cuteafd_loader::push_load::is_push_shard(&root.join(file))),
            "native push source is not a SparkNest shard");
        let headroom = super::spark_runtime_headroom()?;
        let peak = admit_peak(plan.layout.bytes(), plan.scratch, plan.host_scratch, headroom,
            library.cuda_allocatable_memory_info()?.0)?;
        ensure!(plan.layout.bytes().checked_add(plan.scratch).is_some_and(|n| n <= config.device_budget),
            "native push transform exceeds device budget");
        if config.declared_world.ownership_topology().is_some() {
            ensure!(peak <= config.device_budget, "native push unified peak exceeds configured admission");
        }
        let offer = Offer { rank: config.rank, host, arena_bytes: plan.logical_bytes(),
            placement: plan.placement.clone() };
        tracing::info!(rank = config.rank, resident_bytes = resident, transform_bytes = plan.scratch,
            host_scratch_bytes = plan.host_scratch, load_peak_bytes = peak, "Spark native planes push admitted");
        Ok((plan, checked, offer, selections, budgets))
    })();
    if let Err(error) = &planned { tracing::warn!(%error, "Spark native planes push unavailable; offering mount"); }
    let credentials = push::offer_with(stream, planned.as_ref().map(|(_, _, offer, _, _)| offer.clone()).map_err(|error| format!("{error:#}")))?;
    let Some((group, token)) = credentials else { return Ok(None); };
    let (plan, checked, _, selections, budgets) = planned?;
    receive(library, plan, &checked, &selections, &budgets, config.rank, &PushClient::default(), stream, group, token)
}

fn admit_peak(resident: usize, scratch: usize, host: usize, headroom: usize, available: usize) -> Result<usize> {
    let peak = super::push::admit_peak(resident, host, host, headroom, available)?
        .checked_add(scratch).context("native transform peak overflow")?;
    ensure!(peak <= available, "native push transform exceeds available memory");
    Ok(peak)
}

fn receive<'a>(library: &'a NativeLibrary, plan: RankPlan,
    checked: &cuteafd_loader::push_load::builder::CheckedPlan,
    selections: &[ExpertLayer], budgets: &[ExpertLoadBudget], rank_id: usize,
    client: &PushClient, stream: &mut TcpStream, group: u64, token: u64)
    -> Result<Option<Vec<ExpertWeights<'a>>>> {
    let device = Device { library, id: library.cuda_get_device()? };
    tracing::info!(group, rank = rank_id, "Spark native planes push rank joining load group");
    let joined = (|| -> Result<_> {
        let expected = ExpectedCoverage::from_plan(checked, u32::try_from(rank_id)?)?;
        let arena = plan.scales.destination(device, plan.layout.clone(), expected)?;
        let ring = RegisteredRing::new(library, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES)?;
        let rank = client.join(JoinArgs { group, token, rank: u32::try_from(rank_id)?,
            ring_fd: ring.fd().as_raw_fd(), slots: u32::try_from(ring.slots())?,
            slot_bytes: u32::try_from(ring.slot_bytes())? })?;
        Ok((arena, ring, rank))
    })();
    push::joined_with(stream, joined.as_ref().map(|_| ()).map_err(|error| format!("{error:#}")))?;
    let candidate = joined.and_then(|(arena, ring, rank)| {
        let delivered = push_consumer::consume(device, arena, ring, rank, DEFAULT_SLOTS)?;
        let initializers = plan.layers.iter().enumerate().map(|(region, layer)|
            push_initialize::Layer::Native { region, layout: &layer.planes, scalars: layer.scalars.clone() })
            .collect::<Vec<_>>();
        let ready = Rc::new(push_initialize::initialize(delivered, &initializers, None, plan.scratch)?);
        ensure!(selections.len() == plan.layers.len() && budgets.len() == selections.len(),
            "native push publication layer count differs");
        plan.layers.iter().zip(selections).zip(budgets).enumerate()
            .map(|(region, ((layer, &selection), &budget))|
                ExpertWeights::from_push(library, ready.clone(), region, selection, &layer.planes, budget))
            .collect::<Result<Vec<_>>>()
    });
    if let Err(error) = &candidate {
        tracing::warn!(group, rank = rank_id, %error, "Spark native planes push candidate rejected");
    }
    if push::completed_with(stream, candidate.as_ref().map(|_| ()).map_err(|error| format!("{error:#}")))? {
        let weights = candidate?;
        tracing::info!(group, rank = rank_id, layers = weights.len(),
            "Spark native planes push committed; publishing ready rank arena");
        return Ok(Some(weights));
    }
    drop(candidate);
    ensure!(!library.is_quarantined_after_failed_drain(), "failed push drain prevents safe mount retry");
    push::drained(stream).context("native planes push fallback drain barrier")?;
    Ok(None)
}

#[cfg(test)]
#[path = "push_native/tests.rs"]
mod tests;
