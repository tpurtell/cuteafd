//! Coordinator (RTX) ranks of a push group, receive side: each coordinator GPU
//! receives its TP2 expert halves (and, where the family lands them, its dense
//! operands) on its own thread and ring, under the same control protocol as a
//! Spark rank. Publication follows the whole-group commit.
#[cfg(test)]
mod tests;
use crate::shared::memory::{push_consumer::PushDestination, push_coordinator::Tp2ExpertPlan,
    weight_arena::{DeliveredArena, LoadingArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{push_load::builder::CheckedPlan, OfficialV41Catalog, V41Exl3Layer};

/// A coordinator rank's logical space: its dense operands (a family-provided
/// destination) in `[0, base)`, then the TP2 expert arena (device bytes, then
/// its host validation tail) from `base`. Nothing lands in the gap between, so
/// no fragment spans the two.
pub(crate) struct Combined<'a, D> {
    dense: D,
    base: u64,
    tp2: LoadingArena<'a>,
}

impl<'a, D: PushDestination<Delivered = ()>> Combined<'a, D> {
    pub fn new(dense: D, base: u64, tp2: LoadingArena<'a>) -> Result<Self> {
        ensure!(dense.logical_bytes() as u64 <= base && base % 256 == 0, "dense operands overlap the TP2 arena");
        Ok(Self { dense, base, tp2 })
    }
    fn tp2_offset(&self, offset: u64) -> Option<u64> { offset.checked_sub(self.base) }
}

impl<'a, D: PushDestination<Delivered = ()>> PushDestination for Combined<'a, D> {
    type Delivered = DeliveredArena<'a>;
    fn belongs_to(&self, device: crate::shared::memory::device::Device<'_>) -> bool {
        self.dense.belongs_to(device) && PushDestination::belongs_to(&self.tp2, device)
    }
    fn logical_bytes(&self) -> usize { self.base as usize + PushDestination::logical_bytes(&self.tp2) }
    fn device_bytes(&self) -> usize { self.base as usize + PushDestination::device_bytes(&self.tp2) }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        match self.tp2_offset(offset) {
            Some(at) => PushDestination::check_fragment(&self.tp2, at, bytes),
            None => self.dense.check_fragment(offset, bytes),
        }
    }
    fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        match self.tp2_offset(offset as u64) {
            Some(at) => PushDestination::destination(&self.tp2, at as usize, bytes),
            None => self.dense.destination(offset, bytes),
        }
    }
    fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        let at = self.tp2_offset(offset).context("host tail fragment in the dense operands")?;
        PushDestination::copy_validation(&mut self.tp2, at, bytes)
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        match self.tp2_offset(offset) {
            Some(at) => PushDestination::copied(&mut self.tp2, at, bytes),
            None => self.dense.copied(offset, bytes),
        }
    }
    fn done(self) -> Result<DeliveredArena<'a>> {
        self.dense.done()?;
        PushDestination::done(self.tp2)
    }
}

/// A family's dense operands for one GPU: their destination, and where the
/// TP2 arena starts in the rank's logical space.
pub(crate) struct DenseRank<D> {
    pub destination: D,
    pub base: u64,
}

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
    receive_rank::<NoDense>(device, catalog, plan, checked, None, rank, client, stream, credentials, slots, slot_bytes)
}

/// No dense operands on this rank (families whose dense weights load by mount).
pub(crate) struct NoDense;
impl PushDestination for NoDense {
    type Delivered = ();
    fn belongs_to(&self, _: crate::shared::memory::device::Device<'_>) -> bool { true }
    fn logical_bytes(&self) -> usize { 0 }
    fn device_bytes(&self) -> usize { 0 }
    fn check_fragment(&self, _: u64, _: u64) -> Result<()> { anyhow::bail!("no dense operands on this rank") }
    fn destination(&self, _: usize, _: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        anyhow::bail!("no dense operands on this rank")
    }
    fn copy_validation(&mut self, _: u64, _: &[u8]) -> Result<()> { anyhow::bail!("no dense operands on this rank") }
    fn copied(&mut self, _: u64, _: u64) -> Result<()> { anyhow::bail!("no dense operands on this rank") }
    fn done(self) -> Result<()> { Ok(()) }
}

/// `checked_tp2` covers the TP2 entries alone, from offset 0 (the arena's own
/// footprint); `dense` lands in `[0, dense.base)` of the same rank.
#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_rank<'a, D: PushDestination<Delivered = ()>>(
    device: crate::shared::memory::device::Device<'a>, catalog: &OfficialV41Catalog, plan: Tp2ExpertPlan,
    checked_tp2: &CheckedPlan, dense: Option<DenseRank<D>>, rank: usize,
    client: &cuteafd_loader::push_load::PushClient, stream: &mut std::net::TcpStream, credentials: (u64, u64),
    slots: usize, slot_bytes: usize,
) -> Result<Option<Vec<crate::shared::experts::exl3::Exl3Weights<'a>>>> {
    use crate::shared::memory::{push_consumer, push_initialize, weight_arena::ExpectedCoverage};
    use cuteafd_ffi::push_ring::RegisteredRing;
    use cuteafd_transport::worker_selection::push;
    use std::os::fd::AsRawFd;
    let library = device.library;
    let (group, token) = credentials;
    tracing::info!(group, rank, device = device.id, dense = dense.is_some(), "coordinator push rank joining load group");
    let joined = device.run(|| -> Result<_> {
        let expected = ExpectedCoverage::from_plan(checked_tp2, u32::try_from(rank)?)?;
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
        let delivered = match dense {
            Some(DenseRank { destination, base }) =>
                push_consumer::consume(device, Combined::new(destination, base, arena)?, ring, joined, slots)?,
            None => push_consumer::consume(device, arena, ring, joined, slots)?,
        };
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
        tracing::info!(group, rank, layers = weights.len(), "coordinator push committed; publishing ready rank arena");
        return Ok(Some(weights));
    }
    drop(candidate);
    ensure!(!library.is_quarantined_after_failed_drain(), "failed push drain prevents safe mount retry");
    push::drained(stream).context("coordinator push drained barrier")?;
    tracing::info!(group, rank, "whole push group discarded; coordinator loads through mount");
    Ok(None)
}

/// A family's dense plan for one GPU: its entries (offsets from 0 of the
/// rank's logical space) and where the TP2 arena starts after them.
pub(crate) struct DensePlan {
    pub base: u64,
    pub append: Box<dyn Fn(&mut cuteafd_loader::push_load::builder::Builder, &OfficialV41Catalog, u32) -> Result<()> + Send>,
}

type DenseSlot = std::sync::Arc<std::sync::Mutex<Option<[DensePlan; 2]>>>;

/// The coordinator side of one Spark push group for the two coordinator GPUs:
/// their admitted TP2 expert halves and, when the family lands them, their
/// dense operands. Per-GPU plans, the in-process loopback streams, and the
/// entries the group driver appends after the Spark ranks.
pub(crate) struct CoordinatorTp2Push {
    plans: [Tp2ExpertPlan; 2],
    streams: [std::net::TcpStream; 2],
    host: String,
    snapshot: std::path::PathBuf,
    dense: DenseSlot,
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
        let dense: DenseSlot = Default::default();
        // TP2 entries are re-derived from headers; dense entries (set before the
        // offers are sent) carry no device state either.
        let (append_layers, arenas, append_dense) = (layers.to_vec(), budgets, dense.clone());
        let append = Box::new(move |builder: &mut cuteafd_loader::push_load::builder::Builder,
            catalog: &OfficialV41Catalog, offers: &[cuteafd_transport::worker_selection::push::Offer]| -> Result<()> {
            ensure!(offers.len() == 2, "coordinator push needs two GPU offers");
            let dense = append_dense.lock().map_err(|_| anyhow::anyhow!("coordinator dense plan poisoned"))?;
            for (gpu, offer) in offers.iter().enumerate() {
                let plan = Tp2ExpertPlan::new(catalog, &append_layers, gpu, arenas[gpu])?;
                let rank = u32::try_from(offer.rank)?;
                let base = match dense.as_ref() {
                    Some(dense) => { (dense[gpu].append)(builder, catalog, rank)?; dense[gpu].base }
                    None => 0,
                };
                ensure!(offer.placement.is_empty() && offer.arena_bytes == base + plan.layout.logical_bytes() as u64,
                    "coordinator push offer differs from its admitted arenas");
                plan.append_at(builder, catalog, rank, base)?;
            }
            Ok(())
        });
        let ranks = crate::shared::memory::push_exl3::CoordinatorRanks { ranks: 2, append };
        Ok((Self { plans, streams: [first, second], host, snapshot: snapshot.to_path_buf(), dense }, driver, ranks))
    }

    /// Receives both GPUs' TP2 halves (each on its own thread) while `beside`
    /// runs on this thread. `Ok(None)`: the whole group fell back.
    pub(crate) fn receive<'a, U>(self, devices: [crate::shared::memory::device::Device<'a>; 2],
        catalog: &OfficialV41Catalog, sparks: usize, beside: impl FnOnce() -> U)
        -> (Result<Option<[Vec<crate::shared::experts::exl3::Exl3Weights<'a>>; 2]>>, U) {
        self.receive_with::<NoDense, U>(devices, catalog, sparks, None, beside)
    }

    /// As `receive`, with each GPU's dense operands landing in the same rank:
    /// `dense` is each GPU's plan (published to the group driver before the
    /// offers) and destination (unaliased descriptors of operand allocations the
    /// caller keeps alive and unread until this returns and the group commits).
    /// A dense planning error makes every coordinator GPU offer mount, so the
    /// whole group (Sparks and TP2 halves too) loads through the mount.
    pub(crate) fn receive_with<'a, D: PushDestination<Delivered = ()>, U>(self,
        devices: [crate::shared::memory::device::Device<'a>; 2], catalog: &OfficialV41Catalog, sparks: usize,
        dense: Option<Result<([DensePlan; 2], [D; 2])>>, beside: impl FnOnce() -> U)
        -> (Result<Option<[Vec<crate::shared::experts::exl3::Exl3Weights<'a>>; 2]>>, U) {
        use cuteafd_ffi::push_ring::{DEFAULT_SLOTS, DEFAULT_SLOT_BYTES};
        let Self { plans, streams, host, snapshot, dense: slot } = self;
        let world = sparks + 2;
        let declined = match &dense {
            Some(Err(error)) => {
                tracing::info!(%error, "coordinator dense push planning failed; whole group uses mount");
                true
            }
            _ => false,
        };
        let (bases, destinations) = match dense.and_then(Result::ok) {
            Some((plans, destinations)) => {
                let bases = [plans[0].base, plans[1].base];
                match slot.lock() {
                    Ok(mut slot) => *slot = Some(plans),
                    Err(_) => return (Err(anyhow::anyhow!("coordinator dense plan poisoned")), beside()),
                }
                let [left, right] = destinations;
                (bases, [Some(left), Some(right)])
            }
            None => ([0, 0], [None, None]),
        };
        let [first, second] = plans;
        let [first_stream, second_stream] = streams;
        let [first_dense, second_dense] = destinations;
        let inputs = [(first, first_stream, first_dense), (second, second_stream, second_dense)];
        let client = cuteafd_loader::push_load::PushClient::default();
        let (snapshot, host) = (&snapshot, &host);
        // SAFETY: each GPU's input (its plan, stream and dense destination) is used
        // by that GPU's thread only; the dense descriptors refer to allocations the
        // caller keeps alive and unread until this returns. Each GPU result is a
        // freshly committed, drained rank arena and its views, unaliased (every Rc
        // clone stays inside that GPU's Vec); a fallback returns None after
        // dropping and draining its arena.
        let (pair, value) = unsafe { cuteafd_ffi::synchronized_load::load_pair_with(inputs,
            |gpu, (plan, mut stream, dense): (Tp2ExpertPlan, std::net::TcpStream, Option<D>)| {
            let rank = group_rank(sparks, gpu);
            let base = bases[gpu];
            let planned = (|| -> Result<_> {
                ensure!(!declined, "coordinator dense operands were not planned");
                let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
                let checked = plan.checked(catalog, &root, world, rank)?;
                ensure!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.join(f))),
                    "coordinator push source is not a SparkNest shard");
                Ok(checked)
            })();
            if let Err(error) = &planned { tracing::info!(gpu, %error, "coordinator push unavailable; offering mount"); }
            let offer = planned.as_ref().ok().map(|_| cuteafd_transport::worker_selection::push::Offer {
                rank, host: host.clone(), arena_bytes: base + plan.layout.logical_bytes() as u64, placement: Vec::new() });
            let credentials = cuteafd_transport::worker_selection::push::offer(&mut stream, offer)?;
            let Some(credentials) = credentials else { return Ok(None) };
            receive_rank(devices[gpu], catalog, plan, &planned?,
                dense.map(|destination| DenseRank { destination, base }), rank, &client, &mut stream, credentials,
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
