//! Coordinator (RTX) ranks of a push group, receive side: each coordinator GPU
//! receives its TP2 expert halves on its own thread and ring, under the same
//! control protocol as a Spark rank. Publication follows the whole-group commit.
#[cfg(test)]
mod tests;
use crate::shared::memory::push_coordinator::Tp2ExpertPlan;
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{push_load::builder::CheckedPlan, OfficialV41Catalog, V41Exl3Layer};
/// One coordinator GPU's half of a push group, received on its own thread with
/// its own ring, under the same control protocol as a Spark rank (a loopback
/// stream to the group driver). `Ok(None)` means the whole group fell back:
/// every arena reference is dropped before the drained acknowledgment.
#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_tp2<'a>(
    device: crate::shared::memory::device::Device<'a>, catalog: &OfficialV41Catalog, plan: Tp2ExpertPlan,
    checked: &CheckedPlan, rank: usize, client: &cuteafd_loader::push_load::PushClient,
    stream: &mut std::net::TcpStream, credentials: (u64, u64), slots: usize, slot_bytes: usize,
) -> Result<Option<Vec<crate::shared::experts::exl3::Exl3Weights<'a>>>> {
    use crate::shared::memory::{push_consumer, push_initialize, weight_arena::{ExpectedCoverage, LoadingArena}};
    use cuteafd_ffi::push_ring::RegisteredRing;
    use cuteafd_transport::worker_selection::push;
    use std::os::fd::AsRawFd;
    let library = device.library;
    let (group, token) = credentials;
    tracing::info!(group, rank, device = device.id, "coordinator push rank joining load group");
    let joined = device.run(|| -> Result<_> {
        let expected = ExpectedCoverage::from_plan(checked, u32::try_from(rank)?)?;
        let arena = LoadingArena::new(device, plan.layout.clone(), expected)?;
        let ring = RegisteredRing::new(library, slots, slot_bytes)?;
        let joined = client.join(cuteafd_loader::push_load::JoinArgs { group, token,
            rank: u32::try_from(rank)?, ring_fd: ring.fd().as_raw_fd(),
            slots: u32::try_from(ring.slots())?, slot_bytes: u32::try_from(ring.slot_bytes())? })?;
        Ok((arena, ring, joined))
    });
    if let Err(error) = &joined { tracing::info!(group, rank, %error, "coordinator push rank join failed"); }
    push::joined(stream, joined.is_ok()).context("coordinator push joined frame")?;
    let candidate = joined.and_then(|(arena, ring, joined)| {
        let delivered = push_consumer::consume(device, arena, ring, joined, slots)?;
        let initialize = plan.residencies.iter().enumerate().map(|(region, residency)|
            push_initialize::Layer::Exl3 { region, residency }).collect::<Vec<_>>();
        let ready = std::rc::Rc::new(push_initialize::initialize(delivered, &initialize, Some(&plan.validation), 0)?);
        plan.residencies.into_iter().enumerate().map(|(region, layout)| {
            let budget = crate::shared::experts::exl3::Exl3Weights::plan(catalog,
                crate::shared::experts::layer::ExpertLayer::BackboneTp2 {
                    layer: match layout.layer { V41Exl3Layer::Backbone(l) => l, _ => anyhow::bail!("draft layer in coordinator TP2") },
                    rank: layout.rank })?;
            crate::shared::experts::exl3::Exl3Weights::from_push(ready.clone(), region, layout, budget)
        }).collect::<Result<Vec<_>>>()
    });
    if let Err(error) = &candidate { tracing::info!(group, rank, %error, "coordinator push candidate rejected"); }
    if push::completed(stream, candidate.is_ok()).context("coordinator push completion")? {
        let weights = candidate?;
        tracing::info!(group, rank, layers = weights.len(), "coordinator TP2 push committed; publishing ready rank arena");
        return Ok(Some(weights));
    }
    drop(candidate);
    ensure!(!library.is_quarantined_after_failed_drain(), "failed push drain prevents safe mount retry");
    push::drained(stream).context("coordinator push drained barrier")?;
    tracing::info!(group, rank, "whole push group discarded; coordinator loads through mount");
    Ok(None)
}
