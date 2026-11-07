//! Namespace-required EXLR8 transaction; no execution view exists before Done.
#[cfg(test)]
#[path = "push_exlr8/tests.rs"]
mod tests;
use super::NativeExpertServiceConfig;
use crate::shared::{experts::{exlr8::Exlr8Weights, layer::ExpertLoadBudget}, memory::{
    device::Device, push_consumer, push_exlr8::RankPlan, push_initialize,
    weight_arena::{ExpectedCoverage, LoadingArena},
}};
use anyhow::{ensure, Result};
use cuteafd_ffi::{push_ring::{RegisteredRing, DEFAULT_SLOT_BYTES, DEFAULT_SLOTS}, NativeLibrary};
use cuteafd_loader::{plan::exlr8::residency::Residency, push_load::{JoinArgs, PushClient}};
use cuteafd_transport::worker_selection::push::{self, Offer};
use std::{net::TcpStream, os::fd::AsRawFd, rc::Rc};

pub(super) fn load<'a>(library: &'a NativeLibrary, config: &NativeExpertServiceConfig,
    residencies: &[Residency], budgets: &[ExpertLoadBudget], resident: usize,
    workspace: usize, retained_host: usize, descriptors: usize, stream: &mut TcpStream,
) -> Result<Option<Vec<Exlr8Weights<'a>>>> {
    let Some(((plan, checked), (group, token))) = super::push::plan_offer(&config.snapshot, stream, |root, host| {
        let plan = RankPlan::from_residencies(&config.snapshot, residencies.to_vec(), resident)?;
        let checked = plan.checked(&config.snapshot, &root, config.world, config.rank)?;
        ensure!(checked.plan.files.iter().all(|file|
            cuteafd_loader::push_load::is_push_shard(&root.join(file))), "EXLR8 push source is not a SparkNest shard");
        // Original plans stay alive until the group commits or drains.
        let headroom = super::spark_runtime_headroom(config, cuteafd_core::expert_geometry(), plan.layout.bytes())?;
        let extra = workspace.checked_add(retained_host.checked_mul(2).ok_or_else(||
            anyhow::anyhow!("EXLR8 retained host overflow"))?)
            .and_then(|n| n.checked_add(descriptors))
            .and_then(|n| n.checked_add(headroom))
            .ok_or_else(|| anyhow::anyhow!("EXLR8 push headroom overflow"))?;
        let peak = super::push::admit_peak(plan.layout.bytes(), 0, 0, extra,
            library.cuda_allocatable_memory_info()?.0)?;
        tracing::info!(rank = config.rank, resident_bytes = plan.layout.bytes(),
            load_peak_bytes = peak, "Spark EXLR8 push admitted");
        let offer = Offer { rank: config.rank, host: host.into(), arena_bytes: plan.layout.logical_bytes() as u64,
            placement: plan.placement.clone() };
        Ok(((plan, checked), offer))
    })? else { return Ok(None); };
    super::push::required_receive(receive(library, plan, &checked, budgets, config.rank,
        &PushClient::default(), stream, group, token))
}

fn receive<'a>(library: &'a NativeLibrary, plan: RankPlan,
    checked: &cuteafd_loader::push_load::builder::CheckedPlan, budgets: &[ExpertLoadBudget],
    rank_id: usize, client: &PushClient, stream: &mut TcpStream, group: u64, token: u64,
) -> Result<Option<Vec<Exlr8Weights<'a>>>> {
    let device = Device { library, id: library.cuda_get_device()? };
    tracing::info!(group, rank = rank_id, "Spark EXLR8 push rank joining load group");
    let joined = (|| -> Result<_> {
        let expected = ExpectedCoverage::from_plan(checked, u32::try_from(rank_id)?)?;
        let arena = LoadingArena::new(device, plan.layout.clone(), expected)?;
        let ring = RegisteredRing::new(library, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES)?;
        let rank = client.join(JoinArgs { group, token, rank: u32::try_from(rank_id)?,
            ring_fd: ring.fd().as_raw_fd(), slots: u32::try_from(ring.slots())?,
            slot_bytes: u32::try_from(ring.slot_bytes())? })?;
        Ok((arena, ring, rank))
    })();
    push::joined_with(stream, joined.as_ref().map(|_| ()).map_err(|error| format!("{error:#}")))?;
    let candidate = joined.and_then(|(arena, ring, rank)| {
        let delivered = push_consumer::consume(device, arena, ring, rank, DEFAULT_SLOTS)?;
        let initializers = plan.residencies.iter().enumerate().map(|(region, residency)|
            push_initialize::Layer::Exlr8 { region, residency }).collect::<Vec<_>>();
        let ready = Rc::new(push_initialize::initialize(delivered, &initializers, None, 0)?);
        plan.residencies.into_iter().zip(budgets).enumerate().map(|(region, (layout, &budget))|
            Exlr8Weights::from_push(ready.clone(), region, layout, budget)).collect::<Result<Vec<_>>>()
    });
    if let Err(error) = &candidate { tracing::warn!(group, rank = rank_id, %error, "Spark EXLR8 push candidate rejected"); }
    if push::completed_with(stream, candidate.as_ref().map(|_| ()).map_err(|error| format!("{error:#}")))? {
        let weights = candidate?;
        tracing::info!(group, rank = rank_id, layers = weights.len(),
            "Spark EXLR8 push committed; publishing ready rank arena");
        return Ok(Some(weights));
    }
    drop(candidate);
    ensure!(!library.is_quarantined_after_failed_drain(), "failed push drain prevents safe startup cleanup");
    push::drained(stream)?;
    Ok(None)
}
