//! Coordinator (RTX) ranks of a push group, receive side: each coordinator GPU
//! receives its TP2 expert halves (and, where the family lands them, its dense
//! operands) on its own thread and ring, under the same control protocol as a
//! Spark rank. Publication follows the whole-group commit.
#[cfg(test)]
pub(crate) mod tests;
mod policy;
pub(crate) use policy::{PushPolicy, PushStartupError};
use crate::shared::memory::{push_consumer::PushDestination, push_coordinator::CoordinatorExpertPlan};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{push_load::builder::CheckedPlan, OfficialV41Catalog, V41Exl3Layer};

/// A coordinator rank's logical space: its dense operands (a family-provided
/// destination) in `[0, base)`, then the TP2 expert arena (device bytes, then
/// its host validation tail) from `base`, then the dense destination's staged
/// bytes (its host part) from `host_base`. Gaps separate the three, so no
/// fragment spans two.
pub(crate) struct Combined<D, T> {
    dense: D,
    base: u64,
    host_base: u64,
    tp2: T,
}

impl<D: PushDestination<Delivered = ()>, T: PushDestination> Combined<D, T> {
    pub fn new(dense: D, base: u64, tp2: T) -> Result<Self> {
        ensure!(dense.device_bytes() as u64 <= base && base % 256 == 0, "dense operands overlap the TP2 arena");
        let (host_base, _) = rank_bytes(base, PushDestination::logical_bytes(&tp2) as u64, 1);
        Ok(Self { dense, base, host_base, tp2 })
    }
    fn dense_host(&self) -> u64 { (self.dense.logical_bytes() - self.dense.device_bytes()) as u64 }
    /// The dense destination's own offset of a staged fragment at `offset`.
    fn staged(&self, offset: u64) -> Option<u64> {
        (self.dense_host() > 0 && offset >= self.host_base).then(|| offset - self.host_base + self.dense.device_bytes() as u64)
    }
    fn tp2_offset(&self, offset: u64) -> Option<u64> { offset.checked_sub(self.base) }
}

impl<D: PushDestination<Delivered = ()>, T: PushDestination> PushDestination for Combined<D, T> {
    type Delivered = T::Delivered;
    fn belongs_to(&self, device: crate::shared::memory::device::Device<'_>) -> bool {
        self.dense.belongs_to(device) && PushDestination::belongs_to(&self.tp2, device)
    }
    fn logical_bytes(&self) -> usize {
        match self.dense_host() { 0 => self.base as usize + PushDestination::logical_bytes(&self.tp2),
            host => (self.host_base + host) as usize }
    }
    fn device_bytes(&self) -> usize { self.base as usize + PushDestination::device_bytes(&self.tp2) }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> {
        if let Some(at) = self.staged(offset) { return self.dense.check_fragment(at, bytes); }
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
        if let Some(at) = self.staged(offset) { return self.dense.copy_validation(at, bytes); }
        let at = self.tp2_offset(offset).context("host tail fragment in the dense operands")?;
        PushDestination::copy_validation(&mut self.tp2, at, bytes)
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> {
        match self.tp2_offset(offset) {
            Some(at) => PushDestination::copied(&mut self.tp2, at, bytes),
            None => self.dense.copied(offset, bytes),
        }
    }
    fn done(self) -> Result<T::Delivered> {
        self.dense.done()?;
        PushDestination::done(self.tp2)
    }
}

/// One coordinator GPU's published TP2 halves: EXL3 or native layers (empty
/// when the rank carried dense operands alone).
pub(crate) enum PushedExperts<'a> {
    Exl3(Vec<crate::shared::experts::exl3::Exl3Weights<'a>>),
    Exlr8(Vec<crate::shared::experts::exlr8::Exlr8Weights<'a>>),
    Native(Vec<crate::shared::experts::layer::ExpertWeights<'a>>),
    Fp8(Vec<crate::shared::experts::fp8::Fp8Layer<'a>>),
}

impl<'a> Default for PushedExperts<'a> {
    fn default() -> Self { Self::Exl3(Vec::new()) }
}

impl<'a> PushedExperts<'a> {
    pub fn len(&self) -> usize { match self { Self::Exl3(w) => w.len(), Self::Native(w) => w.len(), Self::Exlr8(w) => w.len(), Self::Fp8(w) => w.len() } }
    pub fn exl3(self) -> Result<Vec<crate::shared::experts::exl3::Exl3Weights<'a>>> {
        match self { Self::Exl3(w) => Ok(w), _ => anyhow::bail!("non-EXL3 TP2 halves where EXL3 expected") }
    }
    pub fn exlr8(self) -> Result<Vec<crate::shared::experts::exlr8::Exlr8Weights<'a>>> {
        match self { Self::Exlr8(w) => Ok(w), _ => anyhow::bail!("non-EXLR8 TP2 halves where EXLR8 expected") }
    }
    pub fn fp8(self) -> Result<Vec<crate::shared::experts::fp8::Fp8Layer<'a>>> {
        match self { Self::Fp8(w) => Ok(w), _ => anyhow::bail!("non-FP8-family payload where FP8-family expected") }
    }
    pub fn native(self) -> Result<Vec<crate::shared::experts::layer::ExpertWeights<'a>>> {
        match self { Self::Native(w) => Ok(w), Self::Exl3(w) if w.is_empty() => Ok(Vec::new()),
            _ => anyhow::bail!("non-native TP2 halves where native expected") }
    }

    pub fn whole_native(self, expected: &[(crate::shared::experts::layer::ExpertLayer, usize)],
        admitted: usize) -> Result<Vec<crate::shared::experts::layer::ExpertWeights<'a>>> {
        let weights = self.native()?;
        let actual = weights.iter().map(|w| (w.layer, w.budget.resident_bytes)).collect::<Vec<_>>();
        validate_whole_inventory(&actual, expected, admitted)?;
        Ok(weights)
    }
}

fn validate_whole_inventory(actual: &[(crate::shared::experts::layer::ExpertLayer, usize)],
    expected: &[(crate::shared::experts::layer::ExpertLayer, usize)], admitted: usize) -> Result<()> {
    ensure!(actual == expected, "committed whole expert inventory differs from selected layers or resident bytes");
    let bytes = actual.iter().try_fold(0usize, |total, (_, bytes)| total.checked_add(*bytes))
        .context("committed whole expert bytes overflow")?;
    ensure!(bytes <= admitted, "committed whole experts exceed admitted resident bytes");
    Ok(())
}

/// A family's dense operands for one GPU: their destination, and where the
/// TP2 arena starts in the rank's logical space.
pub(crate) struct DenseRank<D> {
    pub destination: D,
    pub base: u64,
}

/// One coordinator GPU's TP2 halves alone (the TP2-only group replay).
#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_tp2<'a>(
    device: crate::shared::memory::device::Device<'a>, catalog: &OfficialV41Catalog, plan: CoordinatorExpertPlan,
    checked: &CheckedPlan, rank: usize, client: &cuteafd_loader::push_load::PushClient,
    stream: &mut std::net::TcpStream, credentials: (u64, u64), slots: usize, slot_bytes: usize,
) -> Result<Option<PushedExperts<'a>>> {
    receive_rank::<NoDense>(device, Some((catalog, plan, checked)), None, rank, client, stream, credentials,
        slots, slot_bytes)
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

/// One coordinator GPU's rank of a push group, received on its own thread and
/// ring under the same control protocol as a Spark rank (a loopback stream to
/// the group driver): its TP2 halves (`tp2`: their entries alone from offset
/// 0, the arena's own footprint) and/or its dense operands (`dense`, in
/// `[0, dense.base)` of the rank). `Ok(None)`: the whole group fell back; every
/// arena reference is dropped before the drained acknowledgment. With no TP2,
/// a committed rank returns no expert weights.
#[allow(clippy::too_many_arguments)]
pub(crate) fn receive_rank<'a, D: PushDestination<Delivered = ()>>(
    device: crate::shared::memory::device::Device<'a>,
    tp2: Option<(&OfficialV41Catalog, CoordinatorExpertPlan, &CheckedPlan)>, dense: Option<DenseRank<D>>, rank: usize,
    client: &cuteafd_loader::push_load::PushClient, stream: &mut std::net::TcpStream, credentials: (u64, u64),
    slots: usize, slot_bytes: usize,
) -> Result<Option<PushedExperts<'a>>> {
    use crate::shared::memory::{push_consumer, push_initialize, weight_arena::ExpectedCoverage};
    use cuteafd_ffi::push_ring::RegisteredRing;
    use cuteafd_transport::worker_selection::push;
    use std::os::fd::AsRawFd;
    ensure!(tp2.is_some() || dense.is_some(), "coordinator rank receives nothing");
    let library = device.library;
    let (group, token) = credentials;
    tracing::info!(group, rank, device = device.id, tp2 = tp2.is_some(), dense = dense.is_some(),
        "coordinator push rank joining load group");
    let joined = device.run(|| -> Result<_> {
        let arena = tp2.as_ref().map(|(_, plan, checked)| -> Result<_> {
            let expected = ExpectedCoverage::from_plan(checked, u32::try_from(rank)?)?;
            plan.scales.destination(device, plan.layout.clone(), expected)
        }).transpose()?;
        let ring = RegisteredRing::new(library, slots, slot_bytes)?;
        let joined = client.join(cuteafd_loader::push_load::JoinArgs { group, token,
            rank: u32::try_from(rank)?, ring_fd: ring.fd().as_raw_fd(),
            slots: u32::try_from(ring.slots())?, slot_bytes: u32::try_from(ring.slot_bytes())? })?;
        Ok((arena, ring, joined))
    });
    if let Err(error) = &joined { tracing::info!(group, rank, %error, "coordinator push rank join failed"); }
    push::joined_with(stream, joined.as_ref().map(|_| ()).map_err(|error| format!("{error:#}"))).context("coordinator push joined frame")?;
    let candidate = joined.and_then(|(arena, ring, joined)| {
        let delivered = match (arena, dense) {
            (Some(arena), Some(DenseRank { destination, base })) =>
                Some(push_consumer::consume(device, Combined::new(destination, base, arena)?, ring, joined, slots)?),
            (Some(arena), None) => Some(push_consumer::consume(device, arena, ring, joined, slots)?),
            (None, Some(DenseRank { destination, .. })) => {
                push_consumer::consume(device, destination, ring, joined, slots)?;
                None
            }
            (None, None) => unreachable!("checked above"),
        };
        let Some(((catalog, plan, _), delivered)) = tp2.zip(delivered) else { return Ok(PushedExperts::default()) };
        if let Some(fp8) = &plan.fp8 {
            let initialize = fp8.layers.iter().map(|layer| match &layer.initialize {
                push_initialize::Layer::Fp8 { widen, scalars, inputs, expand } => push_initialize::Layer::Fp8 {
                    widen: widen.clone(), scalars: scalars.clone(), inputs: inputs.clone(), expand: expand.clone(),
                },
                _ => unreachable!("FP8 planner only emits FP8 initializers"),
            }).collect::<Vec<_>>();
            let ready = std::rc::Rc::new(push_initialize::initialize(delivered, &initialize, None, 0)?);
            return fp8.layers.iter().enumerate().map(|(region, layer)| {
                crate::shared::experts::fp8::Fp8Layer::from_push(ready.clone(), region, layer,
                    fp8.layout.region(region)?.start)
            }).collect::<Result<Vec<_>>>().map(PushedExperts::Fp8);
        }
        if !plan.exlr8.is_empty() {
            let initialize = plan.exlr8.iter().enumerate().map(|(region, residency)|
                push_initialize::Layer::Exlr8 { region, residency }).collect::<Vec<_>>();
            let ready = std::rc::Rc::new(push_initialize::initialize(delivered, &initialize, None, 0)?);
            return plan.exlr8.into_iter().enumerate().map(|(region, layout)| {
                let budget = crate::shared::experts::exlr8::Exlr8Weights::plan(&layout)?;
                crate::shared::experts::exlr8::Exlr8Weights::from_push(ready.clone(), region, layout, budget)
            }).collect::<Result<Vec<_>>>().map(PushedExperts::Exlr8);
        }
        if !plan.native.is_empty() {
            // Native halves: the Spark native post-Done plane transform, then views.
            let initialize = plan.native.iter().enumerate().map(|(region, layout)|
                push_initialize::Layer::Native { region, layout, scalars: 0..0 }).collect::<Vec<_>>();
            let ready = std::rc::Rc::new(push_initialize::initialize(delivered, &initialize, None, plan.scratch_bytes())?);
            return plan.native.iter().zip(&plan.layers).enumerate().map(|(region, (layout, &layer))| {
                let selection = if plan.world == 1 {
                    crate::shared::experts::layer::ExpertLayer::BackboneFull { layer }
                } else { crate::shared::experts::layer::ExpertLayer::BackboneTp2 { layer, rank: plan.gpu } };
                let budget = crate::shared::experts::layer::ExpertWeights::plan(library, catalog, selection)?;
                crate::shared::experts::layer::ExpertWeights::from_push(library, ready.clone(), region, selection,
                    layout, budget)
            }).collect::<Result<Vec<_>>>().map(PushedExperts::Native);
        }
        let initialize = plan.residencies.iter().enumerate().map(|(region, residency)|
            push_initialize::Layer::Exl3 { region, residency }).collect::<Vec<_>>();
        let ready = std::rc::Rc::new(push_initialize::initialize(delivered, &initialize, Some(&plan.validation), 0)?);
        plan.residencies.into_iter().enumerate().map(|(region, layout)| {
            use crate::shared::experts::layer::ExpertLayer;
            // The mount loader's selection of this residency.
            let selection = match (layout.layer, layout.world) {
                (V41Exl3Layer::Backbone(layer), 2) => ExpertLayer::BackboneTp2 { layer, rank: layout.rank },
                (V41Exl3Layer::Backbone(layer), 1) => ExpertLayer::BackboneFull { layer },
                (V41Exl3Layer::Dspark(stage), 1) => ExpertLayer::Dspark { stage },
                (layer, world) => anyhow::bail!("coordinator EXL3 residency {layer:?} at world {world}"),
            };
            let budget = crate::shared::experts::exl3::Exl3Weights::plan(catalog, selection)?;
            crate::shared::experts::exl3::Exl3Weights::from_push(ready.clone(), region, layout, budget)
        }).collect::<Result<Vec<_>>>().map(PushedExperts::Exl3)
    });
    if let Err(error) = &candidate { tracing::warn!(group, rank, %error, "coordinator push candidate rejected"); }
    if push::completed_with(stream, candidate.as_ref().map(|_| ()).map_err(|error| format!("{error:#}"))).context("coordinator push completion")? {
        let weights = candidate?;
        tracing::info!(group, rank, layers = weights.len(), "coordinator push committed; publishing ready rank arena");
        return Ok(Some(weights));
    }
    drop(candidate);
    ensure!(!library.is_quarantined_after_failed_drain(), "failed push drain prevents safe startup cleanup");
    push::drained(stream).context("coordinator push drained barrier")?;
    tracing::warn!(group, rank, "whole push group discarded before coordinator startup failure");
    Ok(None)
}

/// A family's dense plan for one GPU: device entries from 0 of the rank's
/// logical space up to `base` (where the TP2 arena starts), and `host_bytes`
/// of staged entries placed at the `host_base` the driver passes (after the
/// TP2 arena's device and host bytes).
pub(crate) struct DensePlan {
    pub base: u64,
    pub host_bytes: u64,
    pub host_staging: super::dense::HostStaging,
    pub peer_copies: Vec<super::dense::PeerCopy>,
    pub append: Box<dyn Fn(&mut cuteafd_loader::push_load::builder::Builder, u32, u64) -> Result<()> + Send>,
}

/// A rank's logical bytes: dense device `[0, base)`, the TP2 arena, then
/// (after a gap) the dense staged steps: (where those start, the total).
fn rank_bytes(base: u64, tp2: u64, host: u64) -> (u64, u64) {
    let host_base = if tp2 == 0 { base } else { (base + tp2 + 1).next_multiple_of(256) };
    (host_base, if host == 0 { base + tp2 } else { host_base + host })
}

type DenseSlot = std::sync::Arc<std::sync::Mutex<Option<Vec<DensePlan>>>>;

/// Whole experts on one placement-selected coordinator rank. The checkpoint
/// shard remains world 1/rank 0, independently of the owning coordinator rank.
#[derive(Clone, Debug)]
pub(crate) struct WholeOwner {
    pub rank: usize,
    pub selections: Vec<crate::shared::experts::layer::ExpertLayer>,
    pub budget: usize,
}

/// Expert owners are independent of the dense ranks: every dense GPU joins,
/// including ranks with no whole expert layers assigned by placement.
#[derive(Clone, Debug)]
pub(crate) enum CoordinatorExperts {
    Tp2 { layers: Vec<usize>, budgets: Vec<usize> },
    Exlr8 { manifest: String, layers: Vec<usize>, budgets: Vec<usize> },
    Whole { owners: Vec<WholeOwner> },
}

impl CoordinatorExperts {
    fn validate_ranks(&self, gpus: usize) -> Result<()> {
        match self {
            Self::Tp2 { budgets, .. } => ensure!(gpus == 2 && budgets.len() == gpus,
                "coordinator TP2 budgets require two ranks"),
            Self::Exlr8 { budgets, .. } => ensure!(budgets.len() == gpus,
                "coordinator EXLR8 budgets differ from dense ranks"),
            Self::Whole { owners } => {
                ensure!(!owners.is_empty(), "whole coordinator push needs expert owners");
                let mut ranks = std::collections::BTreeSet::new();
                for owner in owners {
                    ensure!(owner.rank < gpus && ranks.insert(owner.rank) && !owner.selections.is_empty(),
                        "whole expert owner is empty, duplicated or outside coordinator ranks");
                }
            }
        }
        Ok(())
    }

    /// Dense-only ranks have no expert plan. A whole owner's checkpoint shard
    /// identity never depends on its position in the coordinator group.
    fn plan(&self, catalog: &OfficialV41Catalog, gpu: usize) -> Result<Option<CoordinatorExpertPlan>> {
        let plan = match self {
            Self::Exlr8 { manifest, layers, budgets } => CoordinatorExpertPlan::exlr8(catalog, manifest, layers, budgets.len(), gpu,
                *budgets.get(gpu).context("coordinator EXLR8 TP2 budget per GPU")?)?,
            Self::Tp2 { layers, budgets } => CoordinatorExpertPlan::new(catalog, layers, gpu,
                *budgets.get(gpu).context("coordinator TP2 budget per GPU")?)?,
            Self::Whole { owners } => {
                let Some(WholeOwner { selections, budget, .. }) = owners.iter().find(|owner| owner.rank == gpu)
                    else { return Ok(None) };
                use crate::shared::experts::layer::ExpertLayer;
                ensure!(selections.iter().all(|s| matches!(s, ExpertLayer::BackboneFull { .. } | ExpertLayer::Dspark { .. })),
                    "a whole-layer coordinator plan takes full backbone layers and draft stages");
                if catalog.exl3().is_none() {
                    let layers = selections.iter().map(|s| match s {
                        ExpertLayer::BackboneFull { layer } => Ok(*layer),
                        _ => anyhow::bail!("native whole coordinator push requires backbone layers"),
                    }).collect::<Result<Vec<_>>>()?;
                    CoordinatorExpertPlan::whole_native(catalog, &layers, *budget)?
                } else {
                    let residencies = selections.iter().map(|&s| crate::shared::experts::exl3::planned_residency(catalog, s))
                        .collect::<Result<Vec<_>>>()?;
                    CoordinatorExpertPlan::whole(residencies, *budget)?
                }
            }
        };
        Ok(Some(plan))
    }
}

/// The coordinator side of one Spark push group: one rank per coordinator GPU
/// (one or two), each carrying its admitted TP2 expert halves and/or the
/// family's dense operands. Per-GPU TP2 plans, the in-process loopback
/// streams, and the entries the group driver appends after the Spark ranks.
pub(crate) struct CoordinatorPush {
    gpus: Vec<(Option<CoordinatorExpertPlan>, std::net::TcpStream)>,
    host: String,
    snapshot: std::path::PathBuf,
    dense: DenseSlot,
    /// The group driver when no Spark admission drives the group (coordinator ranks alone).
    driver: Option<AloneDriver>,
    /// The SparkNest push endpoint the ranks join (and an alone driver creates on).
    socket: Option<String>,
}

/// The push group driver of coordinator ranks alone, on its own thread; joined
/// after every rank has committed or drained.
struct AloneDriver {
    cancel: Vec<std::net::TcpStream>,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
}
impl AloneDriver {
    fn join(mut self) -> Result<()> {
        self.thread.take().map_or(Ok(()), |thread| thread.join()
            .map_err(|_| anyhow::anyhow!("coordinator push driver panicked"))?)
    }
}
impl Drop for AloneDriver {
    fn drop(&mut self) {
        for stream in &self.cancel { let _ = stream.shutdown(std::net::Shutdown::Both); }
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

/// The push client on `socket`, or the configured endpoint.
fn client(socket: Option<String>) -> Result<cuteafd_loader::push_load::PushClient> {
    Ok(match socket {
        Some(socket) => cuteafd_loader::push_load::PushClient::at(socket, cuteafd_loader::push_load::PUSH_IDLE_TIMEOUT)?,
        None => cuteafd_loader::push_load::PushClient::default(),
    })
}

/// Group rank of coordinator GPU `gpu` after `sparks` Spark ranks.
fn group_rank(sparks: usize, gpu: usize) -> usize { sparks + gpu }

impl CoordinatorPush {
    /// Metadata only, before any allocation: `gpus` coordinator ranks, with each
    /// GPU's admitted TP2 halves when `tp2` is (layers, per-GPU resident budgets).
    pub(crate) fn prepare(snapshot: &std::path::Path, catalog: Option<&OfficialV41Catalog>, gpus: usize,
        experts: Option<CoordinatorExperts>)
        -> Result<(Self, Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks)> {
        ensure!((1..=2).contains(&gpus), "one or two coordinator GPUs");
        let (_, host) = cuteafd_loader::push_load::namespace(snapshot)?;
        let plans = match &experts {
            Some(experts) => {
                experts.validate_ranks(gpus)?;
                let catalog = catalog.context("coordinator expert push needs the expert catalog")?;
                (0..gpus).map(|gpu| experts.plan(catalog, gpu)).collect::<Result<Vec<_>>>()?
            }
            None => (0..gpus).map(|_| None).collect(),
        };
        let (_, available) = cuteafd_loader::formats::mapped_preload::host_memory()?;
        let host_peak = CoordinatorExpertPlan::admit_host(&plans.iter().map(Option::as_ref).collect::<Vec<_>>(), &[], cuteafd_ffi::push_ring::DEFAULT_SLOTS,
            cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES, available)?;
        tracing::info!(gpus, host_peak_bytes = host_peak, "coordinator push host memory admitted");
        let (mut driver, mut ranks) = (Vec::new(), Vec::new());
        for _ in 0..gpus {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let rank = std::net::TcpStream::connect(listener.local_addr()?)?;
            let (accepted, _) = listener.accept()?;
            driver.push(accepted);
            ranks.push(rank);
        }
        let dense: DenseSlot = Default::default();
        // TP2 entries are re-derived from headers; dense entries (set before the
        // offers are sent) carry no device state either.
        let append_experts = experts;
        let append_dense = dense.clone();
        let append = Box::new(move |builder: &mut cuteafd_loader::push_load::builder::Builder,
            catalog: Option<&OfficialV41Catalog>, offers: &[cuteafd_transport::worker_selection::push::Offer]| -> Result<()> {
            ensure!(offers.len() == gpus, "coordinator push offer count differs");
            let dense = append_dense.lock().map_err(|_| anyhow::anyhow!("coordinator dense plan poisoned"))?;
            for (gpu, offer) in offers.iter().enumerate() {
                let rank = u32::try_from(offer.rank)?;
                let (base, host) = dense.as_ref().map_or((0, 0), |dense| (dense[gpu].base, dense[gpu].host_bytes));
                let tp2 = append_experts.as_ref().map(|experts| experts.plan(
                    catalog.context("coordinator expert push needs the expert catalog")?, gpu))
                    .transpose()?.flatten();
                let (host_base, total) = rank_bytes(base, tp2.as_ref().map_or(0, |p| p.logical_bytes()), host);
                if let Some(dense) = dense.as_ref() { (dense[gpu].append)(builder, rank, host_base)?; }
                if let (Some(plan), Some(catalog)) = (&tp2, catalog) { plan.append_at(builder, catalog, rank, base)?; }
                ensure!(offer.placement.is_empty() && offer.arena_bytes == total,
                    "coordinator push offer differs from its admitted arenas");
            }
            Ok(())
        });
        let ranks_spec = crate::shared::memory::push_exl3::CoordinatorRanks { ranks: gpus, append };
        Ok((Self { gpus: plans.into_iter().zip(ranks).collect(), host, snapshot: snapshot.to_path_buf(), dense,
            driver: None, socket: None }, driver, ranks_spec))
    }

    /// [`Self::prepare`] for a launch without Spark peers: the coordinator GPUs
    /// form the push group alone, driven by a thread of this owner (the same
    /// whole-group commit or drained failure). Every family's coordinator dense
    /// records push this way on Spark-free layouts.
    /// With `experts`, the GPUs' admitted expert layers join the group too.
    pub(crate) fn prepare_alone(snapshot: &std::path::Path, catalog: Option<&OfficialV41Catalog>, gpus: usize,
        experts: Option<CoordinatorExperts>) -> Result<Self> {
        Self::prepare_alone_at(snapshot, catalog, gpus, experts, None)
    }

    /// [`Self::prepare_alone`] on push endpoint `socket` (None: the configured one).
    pub(crate) fn prepare_alone_at(snapshot: &std::path::Path, catalog: Option<&OfficialV41Catalog>, gpus: usize,
        experts: Option<CoordinatorExperts>, socket: Option<String>) -> Result<Self> {
        let has_experts = experts.is_some();
        let (mut push, streams, ranks) = Self::prepare(snapshot, catalog, gpus, experts)?;
        push.socket = socket.clone();
        let cancel = streams.iter().map(std::net::TcpStream::try_clone).collect::<std::io::Result<Vec<_>>>()?;
        let snapshot = snapshot.to_path_buf();
        let thread = std::thread::Builder::new().name("coordinator-push".into()).spawn(move || {
            let mut streams = streams;
            cuteafd_transport::worker_selection::push::coordinate(&mut streams, |offers| {
                // As the Spark driver: the expert catalog is reopened on this thread.
                let catalog = has_experts.then(|| cuteafd_loader::read_expert_catalog(&snapshot)).transpose()?;
                crate::shared::memory::push_exl3::prepare_coordinator(&snapshot, catalog.as_ref(), offers, &ranks,
                    &client(socket.clone())?)
            })
        })?;
        push.driver = Some(AloneDriver { cancel, thread: Some(thread) });
        Ok(push)
    }

    /// Receives every coordinator GPU (two: each on its own thread) while
    /// `beside` runs on this thread. `dense`: each GPU's plan (published to the
    /// group driver before the offers) and destination (unaliased descriptors of
    /// operand allocations the caller keeps alive and unread until this returns
    /// and the group commits). Planning/admission failures decline all coordinator
    /// ranks and fail startup only after the peer handshake completes.
    #[allow(clippy::type_complexity)]
    pub(crate) fn receive_with<'a, D: PushDestination<Delivered = ()>, U>(self,
        devices: Vec<crate::shared::memory::device::Device<'a>>, catalog: Option<&OfficialV41Catalog>, sparks: usize,
        dense: Option<Result<(Vec<DensePlan>, Vec<D>)>>, beside: impl FnOnce() -> U)
        -> (Result<Option<Vec<PushedExperts<'a>>>>, U) {
        use cuteafd_ffi::push_ring::{DEFAULT_SLOTS, DEFAULT_SLOT_BYTES};
        let Self { gpus, host, snapshot, dense: slot, driver, socket } = self;
        if devices.len() != gpus.len() || dense.as_ref().is_some_and(|d| d.as_ref()
            .is_ok_and(|(p, d)| p.len() != gpus.len() || d.len() != gpus.len())) {
            return (Err(anyhow::anyhow!("coordinator push GPUs differ from their plans")), beside());
        }
        let peer_copies = dense.as_ref().and_then(|d| d.as_ref().ok()).map(|(plans, _)|
            plans.iter().flat_map(|plan| plan.peer_copies.iter().cloned()).collect::<Vec<_>>()).unwrap_or_default();
        let world = sparks + gpus.len();
        // Destinations have no pinned pool/table allocations yet. Admit all independent
        // owners before offers; a refusal still sends declines to release Spark peers.
        let admission = (|| -> Result<()> {
            if let Some(Err(error)) = &dense { anyhow::bail!("dense planning: {error:#}"); }
            let staging = dense.as_ref().and_then(|d| d.as_ref().ok())
                .map(|(plans, _)| plans.iter().map(|p| p.host_staging).collect::<Vec<_>>()).unwrap_or_default();
            let plans = gpus.iter().map(|(plan, _)| plan.as_ref()).collect::<Vec<_>>();
            let (_, available) = cuteafd_loader::formats::mapped_preload::host_memory()?;
            let peak = CoordinatorExpertPlan::admit_host(&plans, &staging, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES, available)?;
            tracing::info!(host_peak_bytes = peak, "coordinator push complete host peak admitted before offers");
            Ok(())
        })();
        let declined = admission.is_err();
        let (bases, destinations): (Vec<(u64, u64)>, Vec<Option<D>>) = match dense.and_then(Result::ok) {
            Some((plans, destinations)) => {
                let bases = plans.iter().map(|p| (p.base, p.host_bytes)).collect();
                match slot.lock() {
                    Ok(mut slot) => *slot = Some(plans),
                    Err(_) => return (Err(anyhow::anyhow!("coordinator dense plan poisoned")), beside()),
                }
                (bases, destinations.into_iter().map(Some).collect())
            }
            None => (vec![(0, 0); gpus.len()], (0..gpus.len()).map(|_| None).collect()),
        };
        let client = match client(socket) { Ok(client) => client, Err(error) => return (Err(error), beside()) };
        let (snapshot, namespace_host) = (&snapshot, &host);
        let receive = |gpu: usize, (tp2, mut stream, dense): (Option<CoordinatorExpertPlan>, std::net::TcpStream, Option<D>)| {
            let rank = group_rank(sparks, gpu);
            let (base, host) = bases[gpu];
            let planned = (|| -> Result<_> {
                ensure!(!declined, "coordinator dense operands were not planned");
                let Some(plan) = &tp2 else { return Ok(None) };
                let catalog = catalog.context("coordinator TP2 push needs the expert catalog")?;
                let (root, _) = cuteafd_loader::push_load::namespace(snapshot)?;
                let checked = plan.checked(catalog, &root, world, rank)?;
                ensure!(checked.plan.files.iter().all(|f| cuteafd_loader::push_load::is_push_shard(&root.join(f))),
                    "coordinator push source is not a SparkNest shard");
                Ok(Some(checked))
            })();
            if let Err(error) = &planned { tracing::warn!(gpu, %error, "coordinator push planning failed; declining group"); }
            let tp2_bytes = tp2.as_ref().map_or(0, |p| p.logical_bytes());
            let (host_base, arena_bytes) = rank_bytes(base, tp2_bytes, host);
            let offer = planned.as_ref().map(|_| cuteafd_transport::worker_selection::push::Offer {
                rank, host: namespace_host.clone(), arena_bytes, placement: Vec::new() })
                .map_err(|error| format!("{error:#}"));
            let credentials = planned_offer(&mut stream, planned, offer)?;
            let Some((checked, credentials)) = credentials else { return Ok(None) };
            let tp2 = match (tp2, checked.as_ref()) {
                (Some(plan), Some(checked)) => Some((catalog.context("expert catalog")?, plan, checked)),
                _ => None,
            };
            receive_rank(devices[gpu], tp2, dense.map(|destination| DenseRank { destination, base }), rank,
                &client, &mut stream, credentials, DEFAULT_SLOTS, DEFAULT_SLOT_BYTES)
        };
        let inputs: Vec<_> = gpus.into_iter().zip(destinations).map(|((tp2, stream), dense)| (tp2, stream, dense)).collect();
        let (results, value) = if inputs.len() == 2 {
            let [first, second]: [_; 2] = match inputs.try_into() { Ok(pair) => pair, Err(_) => unreachable!() };
            // SAFETY: each GPU's input (its plan, stream and dense destination) is used by
            // that GPU's thread only; the dense descriptors refer to allocations the caller
            // keeps alive and unread until this returns. Each GPU result is a freshly
            // committed, drained rank arena and its views, unaliased (every Rc clone stays
            // inside that GPU's Vec); a fallback returns None after dropping its arena.
            let (pair, value) = unsafe { cuteafd_ffi::synchronized_load::load_pair_with([first, second], receive, beside) };
            (pair.map(Vec::from), value)
        } else {
            let mut inputs = inputs;
            let result = inputs.pop().context("one coordinator GPU").and_then(|input| receive(0, input)).map(|r| vec![r]);
            (result, beside())
        };
        // Every rank has committed or drained: the driver has finished its protocol.
        let driven = driver.map_or(Ok(()), AloneDriver::join);
        let pushed = results.and_then(|results| {
            driven?;
            let committed = results.iter().filter(|r| r.is_some()).count();
            match committed {
                0 => Ok(None),
                n if n == results.len() => Ok(Some(results.into_iter().map(|r| r.unwrap_or_default()).collect())),
                _ => anyhow::bail!("coordinator push GPUs disagree on the group commit"),
            }
        });
        let pushed = match admission {
            Err(error) => Err(PushStartupError::Planning(error).into()),
            Ok(()) => PushPolicy::Required.committed(pushed).map_err(Into::into),
        };
        let pushed = devices.first().context("peer load copies need a coordinator device")
            .and_then(|device| super::dense::committed_peer_copies(device.library, pushed, &peer_copies));
        (pushed, value)
    }
}

/// Planning failures must participate in Begin/decline before being returned.
fn planned_offer<T>(stream: &mut std::net::TcpStream, planned: Result<T>,
    offer: std::result::Result<cuteafd_transport::worker_selection::push::Offer, String>)
    -> Result<Option<(T, (u64, u64))>> {
    let credentials = cuteafd_transport::worker_selection::push::offer_with(stream, offer)?;
    let planned = planned.map_err(PushStartupError::Planning)?;
    Ok(credentials.map(|credentials| (planned, credentials)))
}
