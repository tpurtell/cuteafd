//! One admitted SparkNest group for native dense weights and all expert sets.
//! Loaded targets remain private until every coordinator and Spark rank commits.
use crate::shared::{
    experts::rtx::{dense, push::{CoordinatorExperts, CoordinatorPush, DensePlan, PushedExperts, PushPolicy}},
    memory::{device::Device, push_exl3::CoordinatorRanks},
};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{plan::checkpoint::CheckpointView, push_load::PushHostEnvelope, OfficialV41Catalog};

pub(crate) type Membership = (Vec<std::net::TcpStream>, CoordinatorRanks);

pub(crate) struct Prepared<'a, T> {
    push: CoordinatorPush,
    owners: T,
    devices: Vec<Device<'a>>,
    sparks: usize,
    dense: (Vec<DensePlan>, Vec<dense::DenseArena<'a>>),
}

/// The caller has solved the complete host/device recipe before invoking this.
/// Each expert set is an independently retired onboarding owner. Spark streams
/// join this same group through `Admission::start_with`, never a second load.
/// Mount snapshots use the ordinary family loader instead of this entry point.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare<'a, T>(view: &CheckpointView, catalog: &OfficialV41Catalog,
    devices: &[Device<'a>], sparks: usize, expert_sets: Vec<CoordinatorExperts>,
    envelope: PushHostEnvelope, load: impl FnOnce() -> Result<T>)
    -> Result<(Prepared<'a, T>, Option<Membership>)> {
    ensure!(PushPolicy::for_snapshot(view.snapshot())? == PushPolicy::Required,
        "MiniMax grouped push needs a SparkNest namespace");
    ensure!(matches!(devices.len(), 1 | 2) && devices.iter().enumerate().all(|(rank, device)|
        std::ptr::eq(device.library, devices[0].library)
            && devices[..rank].iter().all(|other| other.id != device.id)),
        "MiniMax push needs distinct coordinator devices in one library");
    let (owners, records) = dense::recording(load)?;
    let dense = dense::plan_records(view.snapshot(), records, devices, sparks)?;
    let experts = (!expert_sets.is_empty()).then_some(CoordinatorExperts::Sets(expert_sets));
    let (push, membership) = if sparks == 0 {
        (CoordinatorPush::prepare_alone_view_admitted(view, Some(catalog), devices.len(), experts, envelope)?, None)
    } else {
        let (push, streams, ranks) = CoordinatorPush::prepare_view_admitted(view, Some(catalog),
            devices.len(), experts, envelope)?;
        (push, Some((streams, ranks)))
    };
    Ok((Prepared { push, owners, devices: devices.to_vec(), sparks, dense }, membership))
}

impl<'a, T> Prepared<'a, T> {
    /// Call while awaiting the Spark admission transaction. No initialized owner
    /// escapes a declined, failed, or partially committed group.
    pub(crate) fn receive(self, catalog: &OfficialV41Catalog) -> Result<(T, Vec<Vec<PushedExperts<'a>>>)> {
        let Self { push, owners, devices, sparks, dense } = self;
        let (received, ()) = push.receive_sets_with(devices, Some(catalog), sparks, Some(Ok(dense)), || ());
        let experts = PushPolicy::Required.committed(received)?
            .context("MiniMax group did not publish coordinator ranks")?;
        Ok((owners, experts))
    }
}

/// A published independently retired expert set, in logical coordinator order.
/// Backing identity comes from admission, never from the network's rank numbers.
pub(crate) struct ServingExpertSet<'a> {
    pub backings: Vec<cuteafd_loader::placement::families::minimax_m3::ExpertBacking>,
    pub ranks: Vec<PushedExperts<'a>>,
}

pub(crate) struct ServingWeights<'a> {
    pub targets: Vec<super::TargetWeights<'a>>,
    pub embedding: crate::shared::token_io::TokenEmbedding<'a>,
    pub expert_sets: Vec<ServingExpertSet<'a>>,
}

/// Both coordinator delivery and Spark validation must succeed before publication.
/// Dropping this transaction drops all unpublished target and expert backings.
pub(crate) struct PendingServing<'a> {
    prepared: Prepared<'a, (Vec<super::TargetWeights<'a>>, crate::shared::token_io::TokenEmbedding<'a>)>,
    expected: ServingContract,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ServingRefusal {
    #[error("MiniMax serving loading requires the admitted SparkNest push route")]
    PushRequired,
    #[error("MiniMax serving weights differ from admission: {0}")]
    Contract(String),
}

struct ServingContract {
    targets: Vec<cuteafd_loader::families::minimax_m3::weight_layout::CoordinatorWeights>,
    backings: Vec<cuteafd_loader::placement::families::minimax_m3::ExpertBacking>,
    facts: cuteafd_loader::placement::planner::Facts,
    devices: Vec<i32>,
    embedding_bytes: u64,
}

/// The startup edge owns Spark selection/handshake policy. It passes returned
/// membership to SparkAdmission::start_with, then consumes the pending barrier
/// in finish. This function accepts only the recipe's selected host envelope.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_serving<'a>(view: &CheckpointView, catalog: &OfficialV41Catalog,
    checkpoint: &cuteafd_loader::plan::Checkpoint, devices: &[Device<'a>],
    inputs: &cuteafd_loader::placement::families::minimax_m3::MiniMaxInputs,
    admitted: &cuteafd_loader::placement::families::minimax_m3::AdmittedMiniMax)
    -> Result<(PendingServing<'a>, Option<Membership>)> {
    use crate::shared::token_io::EmbedPlacement;
    use cuteafd_loader::{families::minimax_m3::weight_layout::{CoordinatorWeights, HeadsWeights, SplitFfnWeights},
        placement::{families::minimax_m3::ExpertStorage, planner::EmbeddingHome}};
    if inputs.split() && inputs.split_wire.is_none() {
        return Err(cuteafd_loader::placement::families::minimax_m3::ServingRefusal::SplitWireNotQualified.into());
    }
    if PushPolicy::for_snapshot(view.snapshot())? != PushPolicy::Required {
        return Err(ServingRefusal::PushRequired.into());
    }
    ensure!(checkpoint.snapshot == view.snapshot() && catalog.snapshot() == view.snapshot()
        && checkpoint.view.as_ref().is_none_or(|selected| selected.fingerprint() == view.fingerprint()),
        ServingRefusal::Contract("checkpoint, catalog and push view differ".into()));
    let envelope = admitted.push_host_envelope.clone().ok_or(ServingRefusal::PushRequired)?;
    let expected = ServingContract::new(checkpoint, devices, inputs, admitted)?;
    ensure!(expected.backings.iter().all(|backing| backing.storage == ExpertStorage::Push),
        ServingRefusal::Contract("selected expert backing is mount-only".into()));
    let heads = if inputs.heads() { Some(HeadsWeights::plan(checkpoint, &inputs.cfg, EmbeddingHome::Device(0))?) }
        else { None };
    let split = if inputs.split() { Some(SplitFfnWeights::plan(checkpoint, &inputs.cfg,
        &inputs.owners.iter().map(|&gpu| usize::from(gpu)).collect::<Vec<_>>(), EmbeddingHome::Device(0))?) }
        else { None };
    let planned: Vec<CoordinatorWeights> = match (&heads, &split) {
        (Some(heads), _) => heads.iter().map(|rank| rank.target.clone()).collect(),
        (_, Some(split)) => split.iter().map(|rank| rank.target.clone()).collect(),
        _ => CoordinatorWeights::plan(checkpoint, &inputs.cfg, inputs.gpus,
            &inputs.owners.iter().map(|&rank| rank as usize).collect::<Vec<_>>())?,
    };
    ensure!(planned == inputs.targets, ServingRefusal::Contract("checkpoint target layout changed".into()));
    let expert_sets = serving_expert_plan(&expected.backings, devices.len())?.into_iter().collect();
    let (prepared, membership) = prepare(view, catalog, devices, inputs.spark_ranks, expert_sets, envelope, || {
        let (target, embedding) = match (&heads, &split) {
            (Some(heads), _) => super::TargetWeights::load_heads_with_embedding(devices[0], checkpoint,
                &heads[0], EmbedPlacement::Gpu)?,
            (_, Some(split)) => super::TargetWeights::load_split_ffn_with_embedding(devices[0], checkpoint,
                &split[0], EmbedPlacement::Gpu)?,
            _ => super::TargetWeights::load_with_embedding(devices[0], checkpoint,
                &inputs.targets[0], EmbedPlacement::Gpu)?,
        };
        let mut targets = vec![target];
        for rank in 1..devices.len() {
            targets.push(match (&heads, &split) {
                (Some(heads), _) => super::TargetWeights::load_heads(devices[rank], checkpoint, &heads[rank])?,
                (_, Some(split)) => super::TargetWeights::load_split_ffn(devices[rank], checkpoint, &split[rank])?,
                _ => super::TargetWeights::load(devices[rank], checkpoint, &inputs.targets[rank])?,
            });
        }
        Ok((targets, embedding))
    })?;
    Ok((PendingServing { prepared, expected }, membership))
}

fn serving_expert_plan(backings: &[cuteafd_loader::placement::families::minimax_m3::ExpertBacking],
    gpus: usize) -> Result<Option<CoordinatorExperts>> {
    use crate::shared::experts::{layer::ExpertLayer, rtx::push::WholeOwner};
    if backings.is_empty() { return Ok(None); }
    ensure!(backings.len() == gpus && matches!(gpus, 1 | 2),
        ServingRefusal::Contract("expert rank count changed".into()));
    let layers = backings[0].layers.iter().map(|layer| layer.layer).collect::<Vec<_>>();
    ensure!(backings.iter().enumerate().all(|(gpu, backing)| backing.gpu == gpu
        && backing.layers.iter().map(|layer| layer.layer).eq(layers.iter().copied())),
        ServingRefusal::Contract("expert rank/layer identity changed".into()));
    let budgets = backings.iter().map(|backing| backing.arena.bytes()).collect::<Vec<_>>();
    Ok(Some(if gpus == 2 { CoordinatorExperts::Tp2 { layers, budgets } }
        else { CoordinatorExperts::Whole { owners: vec![WholeOwner { rank: 0,
            selections: layers.into_iter().map(|layer| ExpertLayer::BackboneFull { layer }).collect(),
            budget: budgets[0] }] } }))
}

impl<'a> PendingServing<'a> {
    pub(crate) fn finish(self, catalog: &OfficialV41Catalog,
        pending: Option<crate::shared::placement::PendingSparkAdmission>) -> Result<ServingWeights<'a>> {
        let Self { prepared, expected } = self;
        ensure!(pending.is_some() == (prepared.sparks > 0),
            ServingRefusal::Contract("Spark validation barrier missing or unexpected".into()));
        crate::shared::placement::load_while_admitting(pending, || {
            let ((targets, embedding), ranks) = prepared.receive(catalog)?;
            let ranks = collect_serving_sets(ranks, &expected.backings)?;
            let weights = ServingWeights { targets, embedding, expert_sets: ranks };
            expected.reconcile(&weights)?;
            Ok(weights)
        })
    }
}

fn collect_serving_sets<'a>(ranks: Vec<Vec<PushedExperts<'a>>>,
    backings: &[cuteafd_loader::placement::families::minimax_m3::ExpertBacking])
    -> Result<Vec<ServingExpertSet<'a>>> {
    if backings.is_empty() {
        ensure!(ranks.iter().all(Vec::is_empty), ServingRefusal::Contract("unexpected coordinator experts".into()));
        return Ok(Vec::new());
    }
    ensure!(ranks.len() == backings.len() && ranks.iter().all(|sets| sets.len() == 1),
        ServingRefusal::Contract("committed expert set/rank count changed".into()));
    let ranks = ranks.into_iter().zip(backings).map(|(mut sets, backing)| {
        let set = sets.pop().context("missing committed expert set")?;
        let PushedExperts::Fp8(layers) = &set else {
            return Err(ServingRefusal::Contract("committed experts are not NVFP4-family planes".into()).into());
        };
        ensure!(layers.iter().map(|layer| layer.layer).eq(backing.layers.iter().map(|layer| layer.layer)),
            ServingRefusal::Contract("committed expert layer order changed".into()));
        Ok(set)
    }).collect::<Result<Vec<_>>>()?;
    Ok(vec![ServingExpertSet { backings: backings.to_vec(), ranks }])
}

impl ServingContract {
    fn new(checkpoint: &cuteafd_loader::plan::Checkpoint, devices: &[Device<'_>],
        inputs: &cuteafd_loader::placement::families::minimax_m3::MiniMaxInputs,
        admitted: &cuteafd_loader::placement::families::minimax_m3::AdmittedMiniMax) -> Result<Self> {
        use cuteafd_loader::{families::minimax_m3::MiniMaxM3Config,
            placement::planner::CoordinatorDevices};
        ensure!(matches!(devices.len(), 1 | 2) && devices.len() == inputs.gpus
            && inputs.targets.len() == devices.len() && admitted.onboard <= inputs.routed.len()
            && MiniMaxM3Config::from_hf(&checkpoint.config)? == inputs.cfg,
            ServingRefusal::Contract("configuration/rank/onboarding selection changed".into()));
        let physical = devices.iter().map(|device| device.id).collect::<Vec<_>>();
        CoordinatorDevices::new(&physical)?;
        ensure!(devices.iter().all(|device| std::ptr::eq(device.library, devices[0].library)),
            ServingRefusal::Contract("coordinator native libraries differ".into()));
        ensure!(inputs.expert_backings.get(admitted.onboard) == Some(&admitted.expert_backings),
            ServingRefusal::Contract("selected expert backing differs from candidate".into()));
        for (gpu, backing) in admitted.expert_backings.iter().enumerate() {
            ensure!(backing.gpu == gpu && physical.get(gpu) == Some(&backing.physical_device)
                && backing.count == admitted.onboard
                && backing.layers.iter().map(|layer| layer.layer).eq(inputs.routed[..admitted.onboard].iter().copied()),
                ServingRefusal::Contract("selected expert physical device or prefix changed".into()));
        }
        ensure!((admitted.onboard == 0 && admitted.expert_backings.is_empty())
            || admitted.expert_backings.len() == inputs.gpus,
            ServingRefusal::Contract("onboarding count has incomplete physical backings".into()));
        let contract = Self { targets: inputs.targets.clone(), backings: admitted.expert_backings.clone(),
            facts: admitted.solved.facts.clone(), devices: physical,
            embedding_bytes: (inputs.cfg.vocab_size as u64).checked_mul(inputs.cfg.hidden as u64)
                .and_then(|bytes| bytes.checked_mul(2)).context("MiniMax embedding extent overflow")? };
        contract.validate_admitted_roots()?;
        Ok(contract)
    }

    fn layouts(&self) -> Vec<(usize, &cuteafd_loader::serving_capacity::WeightArenaLayout,
        cuteafd_core::memory_layout::Category)> {
        use cuteafd_core::memory_layout::Category;
        self.targets.iter().map(|target| (target.rank, &target.arena, Category::Weights))
            .chain(self.backings.iter().map(|backing| (backing.gpu, &backing.arena, Category::Experts))).collect()
    }

    fn validate_admitted_roots(&self) -> Result<()> {
        use cuteafd_loader::placement::planner::{GranularityBacking, Lifetime, Location};
        let mut ids = std::collections::BTreeSet::new();
        for (gpu, layout, category) in self.layouts() {
            ensure!(gpu < self.devices.len() && ids.insert(layout.id()),
                ServingRefusal::Contract("duplicate or out-of-range physical backing".into()));
            let roots = self.facts.allocations.iter().filter(|owner| owner.id == layout.id()).collect::<Vec<_>>();
            ensure!(roots.len() == 1 && roots[0].view.is_none()
                && roots[0].bytes == layout.bytes() as u64 && roots[0].location == Location::coordinator(gpu as u8)
                && roots[0].lifetime == Lifetime::RESIDENT && roots[0].category == category,
                ServingRefusal::Contract(format!("physical backing {} is not admitted exactly", layout.id())));
            let tails = self.facts.allocation_granularity.iter().filter(|tail|
                tail.backings.iter().any(|backing| backing.backing == layout.id())).collect::<Vec<_>>();
            ensure!(tails.len() == 1 && tails[0].id == format!("{}.tail", layout.id())
                && tails[0].location == roots[0].location && tails[0].lifetime == roots[0].lifetime
                && tails[0].backings == [GranularityBacking { backing: layout.id().into(),
                    requests: vec![layout.bytes() as u64] }],
                ServingRefusal::Contract(format!("physical backing {} needs exactly one allocation tail", layout.id())));
        }
        Ok(())
    }

    fn reconcile(&self, weights: &ServingWeights<'_>) -> Result<()> {
        use crate::shared::memory::receipts::{Census, Inventory};
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Lifetime, Location};
        use cuteafd_core::memory_layout::Category;
        ensure!(weights.targets.len() == self.targets.len(),
            ServingRefusal::Contract("loaded target rank count changed".into()));
        let mut inventory = Inventory::new(CoordinatorDevices::new(&self.devices)?, "minimax", Lifetime::RESIDENT,
            Category::Weights, "checkpoint-native target");
        for (gpu, (target, layout)) in weights.targets.iter().zip(&self.targets).enumerate() {
            ensure!(target.rank == gpu && target.layers == layout.layers,
                ServingRefusal::Contract("loaded target rank/layers changed".into()));
            target.inventory(&mut inventory, &format!("target.{gpu}"))?;
        }
        weights.embedding.validate_residency(self.devices[0], self.embedding_bytes)?;
        weights.embedding.inventory(&mut inventory, "embedding", Lifetime::RESIDENT)?;
        for set in &weights.expert_sets {
            ensure!(set.backings == self.backings && set.ranks.len() == set.backings.len(),
                ServingRefusal::Contract("published expert set identity changed".into()));
            for (backing, rank) in set.backings.iter().zip(&set.ranks) {
                let prefix = backing.inventory_prefix.strip_prefix("minimax.")
                    .context("MiniMax expert inventory prefix")?;
                let PushedExperts::Fp8(layers) = rank else { anyhow::bail!("MiniMax expected FP8-family planes") };
                for layer in layers {
                    layer.allocation_receipts(&mut inventory, &format!("{prefix}.weight.{}", layer.layer))?;
                }
            }
        }
        let mut census = Census::default();
        census.owners(inventory.receipts);
        census.normalize(self.facts.phase_order)?;
        let roots = census.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()).collect::<Vec<_>>();
        let layouts = self.layouts();
        ensure!(roots.len() == layouts.len(), ServingRefusal::Contract("retained physical backing count changed".into()));
        for (gpu, layout, category) in layouts {
            let root = roots.iter().find(|receipt| receipt.allocation.id == layout.id())
                .with_context(|| format!("missing retained MiniMax arena {}", layout.id()))?;
            ensure!(root.allocation.location == Location::coordinator(gpu as u8)
                && root.allocation.bytes == layout.bytes() as u64 && root.allocation.category == category
                && root.allocation.lifetime == Lifetime::RESIDENT,
                ServingRefusal::Contract(format!("retained arena {} differs from admission", layout.id())));
            let views = census.receipts.iter().filter(|receipt|
                receipt.allocation.view.as_ref().is_some_and(|view| view.backing == layout.id())).collect::<Vec<_>>();
            ensure!(views.len() == layout.views().len(),
                ServingRefusal::Contract(format!("retained view count differs for {}", layout.id())));
            for (name, planned) in layout.views() {
                let alias = if category == Category::Weights {
                    if name == super::EMBEDDING { "minimax.embedding.table".into() }
                    else { format!("minimax.target.{gpu}.{name}") }
                } else {
                    let (layer, plane) = name.split_once('.').context("MiniMax expert plane identity")?;
                    let backing = self.backings.iter().find(|backing| backing.gpu == gpu).context("MiniMax expert rank")?;
                    format!("{}.weight.{layer}.plane.{plane}", backing.inventory_prefix)
                };
                let actual = views.iter().find(|receipt| receipt.allocation.id == alias)
                    .with_context(|| format!("missing MiniMax native alias {alias}"))?;
                ensure!(actual.allocation.bytes == planned.bytes() as u64
                    && actual.allocation.view.as_ref().is_some_and(|view| view.offset == planned.offset() as u64),
                    ServingRefusal::Contract(format!("native alias {alias} differs from admission")));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod serving_tests {
    use super::*;
    use crate::shared::{experts::rtx::push::tests::receive_pushed_sets,
        memory::push_coordinator::CoordinatorExpertPlan, token_io::EmbedPlacement};
    use cuteafd_core::memory_layout::MemoryKind;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::{families::minimax_m3::weight_layout::CoordinatorWeights,
        placement::{families::minimax_m3::ExpertBacking,
            planner::{weight_arena_components, ComponentKind, Facts, Lifetime, Location, PhaseOrder}},
        serving_capacity::WeightArenaLayout};

    fn facts(targets: &[CoordinatorWeights], experts: &[ExpertBacking]) -> Result<Facts> {
        let mut facts = Facts { phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            resident_weights: vec![], allocations: vec![], deferred: vec![], sequential: vec![],
            reservations: vec![], allowances: vec![], driver_page_tables: vec![], allocation_granularity: vec![] };
        for (gpu, layout, kind) in targets.iter().map(|target| (target.rank, &target.arena, MemoryKind::Weights))
            .chain(experts.iter().map(|backing| (backing.gpu, &backing.arena, MemoryKind::Experts))) {
            for component in weight_arena_components(layout, Location::coordinator(gpu as u8),
                Lifetime::RESIDENT, 1, kind)? {
                match component.kind {
                    ComponentKind::Allocation { template, .. } => facts.allocations.push(template),
                    ComponentKind::AllocationGranularity(owner) => facts.allocation_granularity.push(owner),
                    ComponentKind::ResidentWeight(owner) => facts.resident_weights.push(owner),
                    _ => anyhow::bail!("unexpected test weight component"),
                }
            }
        }
        Ok(facts)
    }

    fn target_snapshot(root: &std::path::Path, gpus: usize) -> Result<(cuteafd_loader::plan::Checkpoint, Vec<CoordinatorWeights>)> {
        let tensors = vec![(super::super::EMBEDDING.into(), "BF16", vec![4, 128]),
            ("norm.weight".into(), "BF16", vec![128])];
        cuteafd_loader::plan::testing::write_snapshot(root, &serde_json::json!({"model_type":"fixture"}), &tensors, None);
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(root)?;
        let layouts = (0..gpus).map(|rank| Ok(CoordinatorWeights { rank, layers: vec![rank], projections: vec![],
            arena: WeightArenaLayout::new(format!("minimax/target/{rank}"),
                tensors.iter().filter(|(name, _, _)| rank == 0 || name != super::super::EMBEDDING)
                    .map(|(name, _, shape)| (name.clone(), shape.iter().product::<usize>() * 2)))? }))
            .collect::<Result<Vec<_>>>()?;
        Ok((checkpoint, layouts))
    }

    #[test]
    fn serving_native_targets_and_expert_sets_reconcile_exactly_and_retire() -> Result<()> {
        use cuteafd_loader::formats::fp8_experts::ExpertFormat;
        for gpus in [1, 2] {
            let fixture = Fixture::build_for_tests()?;
            let library = fixture.load()?;
            fixture.immediate_copies(&library)?;
            fixture.strict_device_free(&library, true)?;
            for device in 0..gpus { fixture.configure_memory(&library, device as i32, 1 << 30)?; }
            let root = tempfile::tempdir_in(fixture.directory())?;
            let target_root = root.path().join("targets");
            std::fs::create_dir(&target_root)?;
            let (checkpoint, layouts) = target_snapshot(&target_root, gpus)?;
            let expert_root = root.path().join("experts");
            std::fs::create_dir(&expert_root)?;
            let catalog = crate::shared::memory::push_fp8::tests::fixture(&expert_root, ExpertFormat::Nvfp4, false)?;
            let devices = (0..gpus).map(|gpu| Device { library: &library, id: (gpus - gpu - 1) as i32 }).collect::<Vec<_>>();
            let backings = devices.iter().enumerate().map(|(gpu, device)|
                ExpertBacking::plan(catalog.fp8().unwrap(), &[1], gpus, gpu, device.id)).collect::<Result<Vec<_>>>()?;
            let expected = ServingContract { facts: facts(&layouts, &backings)?, targets: layouts.clone(),
                backings: backings.clone(), devices: devices.iter().map(|device| device.id).collect(), embedding_bytes: 1024 };
            expected.validate_admitted_roots()?;
            let ((targets, embedding), records) = dense::recording(|| {
                let (first, embedding) = super::super::TargetWeights::load_with_embedding(devices[0], &checkpoint,
                    &layouts[0], EmbedPlacement::Gpu)?;
                let mut targets = vec![first];
                for gpu in 1..gpus { targets.push(super::super::TargetWeights::load(devices[gpu], &checkpoint, &layouts[gpu])?); }
                Ok((targets, embedding))
            })?;
            let ranks = devices.iter().enumerate().map(|(gpu, &device)| {
                let plan = if gpus == 1 { CoordinatorExpertPlan::whole_native(&catalog, &[1], backings[gpu].arena.bytes())? }
                    else { CoordinatorExpertPlan::new(&catalog, &[1], gpu, backings[gpu].arena.bytes())? };
                receive_pushed_sets(device, Some(&catalog), root.path(), root.path(), &records, vec![plan], 4, gpu)
            }).collect::<Result<Vec<_>>>()?;
            let weights = ServingWeights { targets, embedding, expert_sets: collect_serving_sets(ranks, &backings)? };
            expected.reconcile(&weights)?;
            assert_eq!(weights.expert_sets.len(), 1);
            assert_eq!(weights.expert_sets[0].backings, backings);
            let mut bad = ServingContract { facts: expected.facts.clone(), targets: layouts.clone(),
                backings: backings.clone(), devices: expected.devices.clone(), embedding_bytes: 1024 };
            bad.facts.allocation_granularity.pop();
            assert!(bad.validate_admitted_roots().is_err());
            assert!(collect_serving_sets((0..gpus).map(|_| vec![PushedExperts::default()]).collect(), &backings).is_err());
            // A post-delivery receipt refusal drops the unpublished bundle,
            // including all separately committed native expert arenas.
            bad.targets[0].arena = WeightArenaLayout::new("unselected", [("wrong".into(), 256)])?;
            let failed = (|| -> Result<ServingWeights<'_>> {
                bad.reconcile(&weights)?;
                Ok(weights)
            })();
            assert!(failed.is_err());
            drop(records);
            assert_eq!(fixture.device_free_mismatch(&library)?.0, 0);
            let events = fixture.events()?;
            assert_eq!(events.matches('D').count(), events.matches('d').count(), "all target/expert backings retired");
        }
        Ok(())
    }

    #[test]
    fn serving_set_refusals_do_not_publish_unselected_or_wrong_format_owners() -> Result<()> {
        assert!(collect_serving_sets(vec![vec![PushedExperts::default()]], &[]).is_err());
        assert!(matches!(serving_expert_plan(&[], 1)?, None));
        let error: anyhow::Error = ServingRefusal::PushRequired.into();
        assert!(matches!(error.downcast_ref::<ServingRefusal>(), Some(ServingRefusal::PushRequired)));
        Ok(())
    }
}
