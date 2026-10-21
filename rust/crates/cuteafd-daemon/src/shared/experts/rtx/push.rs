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

/// The coordinator side of one Spark push group for the two coordinator GPUs'
/// admitted TP2 expert halves: per-GPU plans, the in-process loopback streams,
/// and the plan entries the group driver appends after the Spark ranks.
pub(crate) struct CoordinatorTp2Push {
    plans: [Tp2ExpertPlan; 2],
    streams: [std::net::TcpStream; 2],
    host: String,
    snapshot: std::path::PathBuf,
}

/// Group rank of coordinator GPU `gpu` after `sparks` Spark ranks.
fn group_rank(sparks: usize, gpu: usize) -> usize { sparks + gpu }

impl CoordinatorTp2Push {
    /// Metadata only: plans and loopback pairs, before any allocation.
    /// `budgets` are the admitted per-GPU TP2 arenas (resident bytes only).
    pub(crate) fn prepare(snapshot: &std::path::Path, catalog: &OfficialV41Catalog, layers: &[usize],
        budgets: [usize; 2]) -> Result<(Self, Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks)> {
        let (_, host) = cuteafd_loader::push_load::namespace(snapshot)?;
        let plans = [Tp2ExpertPlan::new(catalog, layers, 0, budgets[0])?, Tp2ExpertPlan::new(catalog, layers, 1, budgets[1])?];
        let mut driver = Vec::new();
        let mut ranks = Vec::new();
        for _ in 0..2 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let rank = std::net::TcpStream::connect(listener.local_addr()?)?;
            let (accepted, _) = listener.accept()?;
            driver.push(accepted);
            ranks.push(rank);
        }
        let [first, second]: [std::net::TcpStream; 2] = ranks.try_into().map_err(|_| anyhow::anyhow!("two loopback ranks"))?;
        // The driver re-derives each GPU's entries from headers alone, so the
        // closure owns no device state.
        let (append_layers, arenas) = (layers.to_vec(), budgets);
        let append = Box::new(move |builder: &mut cuteafd_loader::push_load::builder::Builder,
            catalog: &OfficialV41Catalog, offers: &[cuteafd_transport::worker_selection::push::Offer]| -> Result<()> {
            ensure!(offers.len() == 2, "coordinator push needs two GPU offers");
            for (gpu, offer) in offers.iter().enumerate() {
                let plan = Tp2ExpertPlan::new(catalog, &append_layers, gpu, arenas[gpu])?;
                ensure!(offer.placement.is_empty() && offer.arena_bytes == plan.layout.logical_bytes() as u64,
                    "coordinator push offer differs from its admitted TP2 arena");
                plan.append(builder, catalog, u32::try_from(offer.rank)?)?;
            }
            Ok(())
        });
        let ranks = crate::shared::memory::push_exl3::CoordinatorRanks { ranks: 2, append };
        Ok((Self { plans, streams: [first, second], host, snapshot: snapshot.to_path_buf() }, driver, ranks))
    }

    /// Receives both GPUs (each on its own thread) while `beside` runs on this
    /// thread. `Ok(None)`: the whole group fell back; nothing was published.
    pub(crate) fn receive<'a, U>(self, devices: [crate::shared::memory::device::Device<'a>; 2],
        catalog: &OfficialV41Catalog, sparks: usize, beside: impl FnOnce() -> U)
        -> (Result<Option<[Vec<crate::shared::experts::exl3::Exl3Weights<'a>>; 2]>>, U) {
        use cuteafd_ffi::push_ring::{DEFAULT_SLOTS, DEFAULT_SLOT_BYTES};
        let Self { plans, streams, host, snapshot } = self;
        let world = sparks + 2;
        let slots: Vec<std::sync::Mutex<Option<(Tp2ExpertPlan, std::net::TcpStream)>>> = plans.into_iter().zip(streams)
            .map(|job| std::sync::Mutex::new(Some(job))).collect();
        let client = cuteafd_loader::push_load::PushClient::default();
        let (snapshot, host) = (&snapshot, &host);
        // SAFETY: each GPU result is a freshly committed, drained rank arena and
        // its views, unaliased (every Rc clone stays inside that GPU's Vec); a
        // fallback returns None after dropping and draining its arena.
        let (pair, value) = unsafe { cuteafd_ffi::synchronized_load::load_pair_beside(|gpu| {
            let (plan, mut stream) = slots[gpu].lock().map_err(|_| anyhow::anyhow!("coordinator push job poisoned"))?
                .take().context("coordinator push job taken twice")?;
            let rank = group_rank(sparks, gpu);
            let planned = (|| -> Result<_> {
                let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
                let checked = plan.checked(catalog, &root, world, rank)?;
                ensure!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.join(f))),
                    "coordinator push source is not a SparkNest shard");
                Ok(checked)
            })();
            if let Err(error) = &planned { tracing::info!(gpu, %error, "coordinator push unavailable; offering mount"); }
            let offer = planned.as_ref().ok().map(|_| cuteafd_transport::worker_selection::push::Offer {
                rank, host: host.clone(), arena_bytes: plan.layout.logical_bytes() as u64, placement: Vec::new() });
            let credentials = cuteafd_transport::worker_selection::push::offer(&mut stream, offer)?;
            let Some(credentials) = credentials else { return Ok(None) };
            receive_tp2(devices[gpu], catalog, plan, &planned?, rank, &client, &mut stream, credentials,
                DEFAULT_SLOTS, DEFAULT_SLOT_BYTES)
        }, beside) };
        let pushed = pair.and_then(|[left, right]| match (left, right) {
            (Some(left), Some(right)) => Ok(Some([left, right])),
            (None, None) => Ok(None),
            _ => anyhow::bail!("coordinator push GPUs disagree on the group commit"),
        });
        (pushed, value)
    }
}
