//! GLM physical component composition and the common plan/serve assembler.
//! Loading provenance never changes an owner's charge or its activation.
use super::{PAGE_ROWS, weights::WeightInventory};
use super::{buffers::{FLOOR, TargetBuffers}, exchanges::ExchangeBuffers, hooks::HookBuffers};
use crate::families::glm5::GlmDsaConfig;
use crate::placement::inventory::{DeclaredGraphSet, FigureAttribution, FigureIdentity, MemoryFigure};
use crate::placement::{AttentionPlacement, GraphRank, GraphSet, LayerMode, planner::*};
use cuteafd_core::memory_layout::Basis;
use crate::serving_capacity::glm_graphs::{GlmGraphKey, GlmGraphLayer, glm_declared_graphs, glm_graph_shapes};
use cuteafd_core::spark_reply::SparkReplyFormats;
use std::collections::{BTreeMap, BTreeSet};

/// A condition names a domain value, never a solver variable index. Conditions
/// in one case are conjunctive; membership in a count interval is disjunctive.
#[derive(Debug, Clone)]
pub enum When {
    Prefill(PrefillShape),
    Lanes(u64),
    Embedding(EmbeddingHome),
    /// The single drafter copy's coordinator GPU.
    DraftHome(u8),
    Onboard { first: usize, last: usize },
    LocalLayer(usize),
    RemoteLayer(usize),
    SparkRanks(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct Case<T> {
    pub when: Vec<When>,
    pub value: T,
}

/// One canonical target-only prefix. TP2 names both destinations but counts a
/// routed layer once. A fallback rank domain is absent only for all-local work.
#[derive(Debug, Clone)]
pub struct ExpertProfile {
    pub layers: Vec<usize>,
    pub destination: Vec<u8>,
    pub counts: Vec<usize>,
    pub rank_sets: Vec<Vec<u8>>,
}

/// Physical subrecipes prepared from headers and selected program tables, not
/// pre-subtracted free-memory totals. This builder does not select a pool or
/// solve once per count. Additional owners use the same named finite domains.
#[derive(Debug, Clone)]
pub struct GlmRecipe<'a> {
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub cfg: &'a GlmDsaConfig,
    pub context: u64,
    pub page_tokens: u64,
    pub maximum_pages: u64,
    /// The executor's existing attention ownership, unchanged by this port.
    pub attention: Vec<LayerMode>,
    pub gpus: usize,
    /// Retained links, even when one assignment onboards every target layer.
    pub spark_links: bool,
    pub prefill: Vec<PrefillShape>,
    pub lanes: Vec<u64>,
    pub embedding: Vec<EmbeddingHome>,
    /// Ordered coordinator homes of the single drafter copy; empty without a
    /// drafter. GPU1 is offered only when its remote bridge owners are charged.
    pub draft_homes: Vec<u8>,
    pub encoders: Vec<crate::plan::encoder::EncoderPlacement>,
    pub media_cache_bytes: u64,
    pub blas: Vec<serving::BlasOwners>,
    pub experts: Option<ExpertProfile>,
    pub targets: Vec<Case<TargetBuffers>>,
    pub hooks: Vec<Case<HookBuffers>>,
    pub exchanges: Vec<Case<ExchangeBuffers>>,
    pub weights: Vec<Case<WeightInventory>>,
    pub graphs: Vec<Case<StartupGraphs>>,
    /// Process-wide ordinary RSS; subsumes heap subsets, never pinned storage.
    pub host: Option<HostFigures>,
    /// Expert, drafter, serving, code and temporary physical components.
    /// Their constructors remain responsible for independent actual receipts.
    pub owners: Vec<Case<Vec<Component>>>,
}

impl GlmRecipe<'_> {
    /// The search prices each pool cudaMalloc at its rounded driver-block
    /// extent. Split that back into the exact request and one named
    /// granularity tail per rank, conserving every selected phase cost.
    pub fn refine_pool_receipts(&self, solved: &mut ResolvedPlan) -> Result<(), Refusal> {
        if solved.pool_tokens == 0 || solved.pool_tokens % self.page_tokens != 0 {
            return Err(invalid("selected pool is not a multiple of the prepared page quantum"));
        }
        let searched = evaluate(&solved.facts, &solved.budgets)?;
        if searched.len() != solved.phases.len() || !searched.iter().zip(&solved.phases).all(|(exact, ledger)|
            exact == ledger)
        {
            return Err(invalid("selected ledger differs from the searched phases"));
        }
        let mut tails = Vec::new();
        for device in 0..self.gpus as u8 {
            let target = self.targets.iter().map(|case| &case.value)
                .find(|target| target.device == device)
                .ok_or_else(|| invalid("prepared target buffers are missing for a selected device"))?;
            if target.page_tokens != self.page_tokens {
                return Err(invalid("prepared page quantum differs from the recipe"));
            }
            let pages = solved.pool_tokens / target.page_tokens;
            let (requests, floors) = target.pool_requests(self.cfg, pages)
                .map_err(|error| invalid(&error.to_string()))?;
            for id in floors {
                let root = solved.facts.allocations.iter().find(|owner| owner.id == id)
                    .ok_or_else(|| invalid(&format!("selected pool owner {id} is missing")))?;
                // The pool's own small-driver-block component priced these.
                if root.view.is_some() || root.lifetime != Lifetime::TARGET || root.bytes != FLOOR {
                    return Err(invalid(&format!("selected latent slot {id} changed shape")));
                }
            }
            let mut backings = Vec::new();
            let mut small = Vec::new();
            for (id, bytes) in requests {
                let root = solved.facts.allocations.iter_mut().find(|owner| owner.id == id)
                    .ok_or_else(|| invalid(&format!("selected pool owner {id} is missing")))?;
                if root.view.is_some() || root.lifetime != Lifetime::TARGET {
                    return Err(invalid(&format!("selected pool owner {id} changed shape")));
                }
                let searched_bytes = if independent_request(bytes) {
                    bytes.checked_add(allocation_granularity_forecast(&[bytes])?)
                        .ok_or(Refusal::Overflow("GLM pool cudaMalloc backing"))?
                } else {
                    bytes
                };
                if root.bytes != searched_bytes {
                    return Err(invalid(&format!("selected pool owner {id} searched cost changed")));
                }
                if !independent_request(bytes) { small.push(bytes); }
                root.bytes = bytes;
                root.shape = vec![bytes];
                backings.push(GranularityBacking { backing: id, requests: vec![bytes] });
            }
            if backings.is_empty() { continue; }
            // Only requests below the shared-block boundary were priced exactly;
            // the block they share adds the rest of the refinement.
            let delta = allocation_granularity_forecast(&small)?;
            tails.push((Location::coordinator(device), delta));
            solved.facts.allocation_granularity.push(AllocationGranularity {
                id: format!("glm.target.gpu.{device}.kv.tail"),
                location: Location::coordinator(device),
                lifetime: Lifetime::TARGET,
                backings,
                evidence: GranularityEvidence::Forecast,
            });
        }
        let phases = evaluate(&solved.facts, &solved.budgets)?;
        if phases.len() != searched.len() {
            return Err(invalid("pool receipt refinement changed the evaluated phase rows"));
        }
        for (exact, before) in phases.iter().zip(&searched) {
            if exact.budget != before.budget || exact.phase != before.phase || exact.available != before.available {
                return Err(invalid("pool receipt refinement changed the evaluated phase rows"));
            }
            let budget = solved.budgets.iter().find(|budget| budget.id == exact.budget)
                .ok_or_else(|| invalid("evaluated phase names an unknown budget"))?;
            let extra = tails.iter()
                .filter(|(location, delta)| *delta > 0 && budget.locations.contains(location)
                    && Lifetime::TARGET.contains(exact.phase, solved.facts.phase_order))
                .try_fold(0u64, |sum, (_, delta)| sum.checked_add(*delta)
                    .ok_or(Refusal::Overflow("GLM pool granularity")))
                ?;
            if exact.owned.checked_add(exact.opaque)
                != before.owned.checked_add(before.opaque).and_then(|total| total.checked_add(extra))
            {
                return Err(invalid("pool receipt refinement changed physical phase costs"));
            }
        }
        solved.phases = phases;
        Ok(())
    }

    /// Mechanical executor output. Backend bytes come from the selected local
    /// physical inventory, not the report's route/payload/check/view totals.
    pub fn execution(&self, solved: &ResolvedPlan, local: &super::local_physical::LocalPhysicalInventory)
        -> Result<(crate::placement::Placement, crate::placement::SparkWorkingSet), Refusal> {
        use crate::placement::{AttentionClass, ExpertHome, ExpertRange, FfnMode, HopSpec,
            HopStorage, LayerAssignment, Placement, PoolLimits, SparkWorkingSet, TerminalHome,
            Tp2Range, plan_hops, residual_homes};
        if !(1..=2).contains(&self.gpus) || solved.decisions.attention != self.attention
            || self.attention.is_empty() || self.attention.len() > self.cfg.layers
            || solved.decisions.onboard_layers.len() != self.gpus
            || !self.prefill.contains(&solved.decisions.prefill)
            || !self.lanes.contains(&solved.decisions.lanes)
            || solved.decisions.embedding.is_none_or(|home| !self.embedding.contains(&home))
            || solved.decisions.draft_home.is_some() == self.draft_homes.is_empty()
            || solved.decisions.draft_home.is_some_and(|home| !self.draft_homes.contains(&home))
        {
            return Err(invalid("resolved execution differs from prepared geometry"));
        }
        let backend_bytes = [
            local.backend_bytes(0).map_err(|error| invalid(&error.to_string()))?,
            local.backend_bytes(1).map_err(|error| invalid(&error.to_string()))?,
        ];
        let modes = &solved.decisions.attention;
        let attention = if modes.iter().all(|mode| *mode == LayerMode::ContextSplit) {
            AttentionPlacement::Context
        } else if modes.iter().all(|mode| *mode == LayerMode::HeadSplit)
            || self.gpus == 1 && modes.iter().all(|mode|
                *mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner })
        {
            AttentionPlacement::Heads
        } else if modes.iter().all(|mode| matches!(mode, LayerMode::Whole { .. })) {
            AttentionPlacement::Layers
        } else {
            return Err(invalid("executor cannot report mixed GLM attention modes"));
        };
        let mut local = Vec::new();
        let mut remote = Vec::new();
        let mut local_by_rank = vec![Vec::new(); self.gpus];
        let mut ranks = None;
        let mut assignments = Vec::with_capacity(modes.len());
        for (layer, &mode) in modes.iter().enumerate() {
            let experts = match solved.decisions.expert_homes.get(&layer) {
                None if self.experts.is_none() || layer < self.cfg.first_moe_layer => ExpertHome::Dense,
                None => return Err(invalid("resolved routed layer has no expert home")),
                Some(_) if self.experts.is_none() || layer < self.cfg.first_moe_layer =>
                    return Err(invalid("resolved expert home names a dense layer")),
                Some(ExpertDestination::Coordinator { gpus }) => {
                    if self.experts.as_ref().is_none_or(|profile| *gpus != profile.destination) {
                        return Err(invalid("resolved local home differs from prepared destination"));
                    }
                    let home = match gpus.as_slice() {
                        [0, 1] if self.gpus == 2 => ExpertHome::RtxTp2,
                        [gpu] if usize::from(*gpu) < self.gpus => ExpertHome::RtxWhole { gpu: *gpu },
                        _ => return Err(invalid("executor cannot bind resolved local expert home")),
                    };
                    for &gpu in gpus { local_by_rank[usize::from(gpu)].push(layer); }
                    local.push(layer);
                    home
                }
                Some(ExpertDestination::Sparks { ranks: selected }) => {
                    if selected.is_empty() || selected.len() > 8
                        || selected.iter().enumerate().any(|(rank, &value)| usize::from(value) != rank)
                    {
                        return Err(invalid("executor requires a Spark rank prefix starting at logical rank 0; non-prefix rank sets cannot be renumbered"));
                    }
                    if self.experts.as_ref().is_none_or(|profile| !profile.rank_sets.contains(selected))
                        || ranks.as_ref().is_some_and(|ranks| ranks != selected)
                    {
                        return Err(invalid("resolved Spark layers do not share a prepared rank set"));
                    }
                    ranks = Some(selected.clone());
                    remote.push(layer);
                    ExpertHome::Spark
                }
            };
            assignments.push(LayerAssignment { mode, experts });
        }
        if solved.decisions.expert_homes.keys().any(|&layer| layer >= modes.len())
            || local_by_rank != solved.decisions.onboard_layers
            || local.iter().copied().ne(self.cfg.first_moe_layer..self.cfg.first_moe_layer + local.len())
            || self.experts.as_ref().is_some_and(|profile| !profile.counts.contains(&local.len()))
            || backend_bytes.iter().enumerate().any(|(rank, &bytes)|
                (rank >= self.gpus || local_by_rank[rank].is_empty()) && bytes != 0)
        {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let whole = local.iter().any(|&layer| matches!(assignments[layer].experts, ExpertHome::RtxWhole { .. }));
        let tp2 = if !local.is_empty() && !whole {
            Some(Tp2Range { first: local[0], layers: local.len(), peak_bytes: backend_bytes })
        } else { None };
        let ranges = local_by_rank.iter().enumerate().map(|(rank, layers)| {
            if tp2.is_some() || layers.is_empty() { ExpertRange { first: 0, layers: 0, peak_bytes: 0 } }
            else { ExpertRange { first: layers[0], layers: layers.len(), peak_bytes: backend_bytes[rank] } }
        }).collect();
        let hop_spec = HopSpec { storage: HopStorage::ReceiveSlots,
            row_bytes: (self.cfg.hidden as u64).checked_mul(2).ok_or(Refusal::Overflow("GLM hop rows"))?,
            rows: solved.decisions.prefill.rows.max(super::DECODE_ROWS),
            lanes: solved.decisions.prefill.waves, entry_gpu: 0, head_gpu: 0,
            terminal: if self.gpus == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } };
        let residual = residual_homes(modes, &hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let hops = plan_hops(modes, &hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let context = super::context_buffers(self.cfg, self.context, solved.decisions.prefill.rows,
            solved.decisions.prefill.waves, super::GlmContext::default());
        let mut peer_row_bytes = 0u64;
        for (layer, mode) in modes.iter().enumerate() {
            if *mode != LayerMode::ContextSplit { continue; }
            peer_row_bytes = peer_row_bytes.checked_add(context.query_row_bytes)
                .and_then(|bytes| bytes.checked_add(context.partial_row_bytes))
                .and_then(|bytes| bytes.checked_add(if self.cfg.indexers[layer] == crate::families::glm5::GlmIndexer::Full {
                    context.candidate_row_bytes
                } else { 0 })).ok_or(Refusal::Overflow("GLM context peer rows"))?;
        }
        let placement = Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: attention, attention_by_kind: vec![(AttentionClass::Dsa, attention)],
            peer_row_bytes, onboard_layers: local.len(), max_pool_tokens_no_onboard: None,
            layers: assignments, movables: solved.decisions.draft_home
                .map(|gpu| (crate::placement::MovableId::Drafter, gpu)).into_iter().collect(),
            expert_ranges: ranges, residual, hops, tp2,
            items: (0..self.gpus).map(|rank| solved.items(Location::coordinator(rank as u8)))
                .collect::<Result<_, _>>()? };
        super::super::GLM5.check(&placement).map_err(|error| invalid(&error.to_string()))?;
        let working = SparkWorkingSet { rtx_layers: local, spark_layers: remote,
            spark_ranks: ranks.map_or(0, |ranks| ranks.len()), prefill_lanes: solved.decisions.prefill.waves };
        Ok((placement, working))
    }
}

/// Native readers share one Mount bank across dense ranks. Shared expert loads
/// serialize by layer/rank, but each invocation owns all 32 async reader banks.
#[derive(Debug, Clone, Default)]
pub struct NativeLoad {
    pub gpus: usize,
    pub sync_upload_bytes: u64,
    pub coarse_overcharge_bytes: u64,
    groups: Vec<Sequential>,
}

impl NativeLoad {
    pub fn validate_vocabulary(&self, bytes: u64) -> anyhow::Result<()> {
        let admitted = self.groups.iter().flat_map(|group| &group.members)
            .find(|owner| owner.id == "glm.load.native.vocabulary").map_or(0, |owner| owner.bytes);
        anyhow::ensure!(bytes == admitted, "native vocabulary staging differs from admission: {bytes} != {admitted}");
        Ok(())
    }

    pub fn validate_shared(&self, plans: &[crate::plan::exlr8::residency::Residency]) -> anyhow::Result<()> {
        for plan in plans {
            anyhow::ensure!(plan.world == self.gpus, "shared loader rank extent differs from admission");
            let prefix = format!("glm.load.shared.rank.{}.layer.{}.", plan.rank, plan.layer);
            let owners = self.groups.iter().flat_map(|group| &group.members)
                .filter(|owner| owner.id.starts_with(&prefix)).collect::<Vec<_>>();
            anyhow::ensure!(owners.len() == crate::plan::exlr8::residency::Residency::LOAD_BANKS
                * crate::plan::exlr8::residency::Residency::LOAD_LANES
                && owners.iter().all(|owner| owner.bytes == plan.staging_bytes.max(16) as u64),
                "shared loader banks differ from admission");
        }
        Ok(())
    }

    pub fn new(dense: &[&crate::plan::exlr8::dense::DensePlan],
        vocabulary: &crate::plan::exlr8::dense::DensePlan,
        shared: &[Vec<crate::plan::exlr8::residency::Residency>],
        storage: super::local_physical::LocalStorage) -> anyhow::Result<Self> {
        use crate::plan::exlr8::{dense::DensePlan, residency::{ExpertSet, Residency}};
        use cuteafd_core::memory_layout::Category;
        use anyhow::{ensure, Context};
        let gpus = dense.len();
        let (mount, _) = DensePlan::mount_staging_bytes(dense)?;
        ensure!(vocabulary.world == gpus && vocabulary.rank == 0 && shared.len() == gpus,
            "native load rank identity");
        let mut out = Self { gpus, ..Self::default() };
        let lifetime = Lifetime { first: Phase::Load, last: Phase::Load };
        let owner = |id: String, bytes: usize| Allocation::buffer(id, Location::PINNED,
            bytes as u64, lifetime, Category::Staging, "GLM native load pinned banks");
        let mut native = Sequential { id: "glm.load.native.readers".into(), members: vec![] };
        let vocabulary = vocabulary.unit_staging_bytes()?;
        if vocabulary > 0 {
            native.members.push(owner("glm.load.native.vocabulary".into(), vocabulary));
        }
        if storage == super::local_physical::LocalStorage::Mount && mount > 0 {
            native.members.push(owner("glm.load.native.mount".into(), mount));
        }
        let native_peak = native.peak();
        if !native.members.is_empty() { out.groups.push(native); }
        let mut lanes = (0..Residency::LOAD_BANKS).flat_map(|bank|
            (0..Residency::LOAD_LANES).map(move |lane| Sequential {
                id: format!("glm.load.shared.bank.{bank}.lane.{lane}"), members: vec![],
            })).collect::<Vec<_>>();
        for (rank, plans) in shared.iter().enumerate() {
            let mut layers = BTreeSet::new();
            for plan in plans {
                ensure!(plan.world == gpus && plan.rank == rank && plan.set == ExpertSet::Shared
                    && layers.insert(plan.layer), "native shared load identity");
                out.sync_upload_bytes = out.sync_upload_bytes.max(plan.sync_upload_bytes()? as u64);
                let per_lane = plan.staging_bytes.max(16);
                ensure!(plan.mount_pinned_bytes()? == per_lane.checked_mul(lanes.len())
                    .context("native shared reader banks overflow")?, "native shared bank geometry");
                for (index, group) in lanes.iter_mut().enumerate() {
                    let bank = index / Residency::LOAD_LANES;
                    let lane = index % Residency::LOAD_LANES;
                    group.members.push(owner(format!("glm.load.shared.rank.{rank}.layer.{}.bank.{bank}.lane.{lane}",
                        plan.layer), per_lane));
                }
            }
        }
        let shared_peak = lanes.iter().try_fold(0u64, |bytes, group|
            bytes.checked_add(group.peak()).context("native shared load peak overflow"))?;
        // Coarse Load charges native_peak + shared_peak instead of their true
        // serialized max. The exact overcharge in bytes is retained and reported;
        // every bank identity remains available for independent reconciliation.
        out.coarse_overcharge_bytes = native_peak.checked_add(shared_peak)
            .context("native load peak overflow")? - native_peak.max(shared_peak);
        out.groups.extend(lanes.into_iter().filter(|group| !group.members.is_empty()));
        Ok(out)
    }

    fn components(&self, maximum_pages: u64) -> Vec<Component> {
        self.groups.iter().cloned().map(|group| Component {
            activation: Activation::always(), bytes: ByteCurve::fixed(group.peak(), maximum_pages),
            kind: ComponentKind::Sequential(group),
        }).collect()
    }
}

/// Selected dense and expert push owners, charged before the complete recipe solve.
pub fn push_host_envelope(checkpoint: &crate::plan::Checkpoint, cfg: &GlmDsaConfig,
    layers: usize, ranks: usize, catalog: Option<&crate::OfficialV41Catalog>,
    metadata: Option<&crate::plan::exlr8::residency::MetadataInput>, largest_native: u64)
    -> anyhow::Result<Option<crate::push_load::PushHostEnvelope>> {
    use anyhow::{ensure, Context};
    if super::local_physical::LocalStorage::for_snapshot(&checkpoint.snapshot)?
        == super::local_physical::LocalStorage::Mount { return Ok(None); }
    ensure!((1..=2).contains(&ranks) && layers <= cfg.layers, "GLM push identity");
    let largest = checkpoint.tensors.iter().filter(|t| !t.meta.name.contains(".experts.")
        && !t.meta.name.starts_with("experts.")).map(|t| t.meta.byte_length).max().unwrap_or(0).max(largest_native);
    let slots = cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64;
    let slot = cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64;
    let dense = crate::push_load::admitted_push_envelope(largest, slots as usize, slot)?;
    let mut expert = vec![0u64; ranks];
    let mut retained = vec![0u64; ranks];
    for rank in 0..ranks {
        for layer in cfg.first_moe_layer..layers {
            if let Some(metadata) = metadata {
                let plan = metadata.plan(layer, ranks, rank, crate::plan::exlr8::residency::ExpertSet::Routed)?;
                expert[rank] = expert[rank].max(plan.descriptor_host_bytes()? as u64);
                retained[rank] = retained[rank].checked_add(plan.host_bytes() as u64)
                    .context("GLM retained push metadata overflow")?;
            } else {
                let catalog = catalog.context("GLM push experts require a catalog or selected EXLR8 metadata")?;
                if let Some(manifest) = catalog.exl3() {
                    let plan = manifest.residency(crate::V41Exl3Layer::Backbone(layer), ranks, rank)?;
                    expert[rank] = expert[rank].checked_add(plan.loads.iter()
                        .filter(|load| load.tensor.ends_with(".trellis")).count() as u64 * 4)
                        .context("GLM push validation overflow")?;
                } else {
                    expert[rank] = catalog.tensors().iter().filter(|t| t.metadata.name.ends_with("input_scale")
                        || t.metadata.name.ends_with("weight_scale_2"))
                        .map(|t| t.metadata.byte_length).max().unwrap_or(0).checked_mul(2)
                        .context("GLM native push scratch overflow")?.max(2 << 20);
                }
            }
        }
    }
    Ok(Some(crate::push_load::PushHostEnvelope::with_retained(expert, vec![dense; ranks], retained, slots, slot)?))
}

/// Target owners assembled from the same physical plans consumed by the loaders.
/// These facts contain no sampled-free subtraction or admission decision.
#[derive(Debug, Clone)]
pub struct TargetFacts {
    pub weights: Vec<(EmbeddingHome, WeightInventory)>,
    pub hooks: Vec<HookBuffers>,
    pub embedding_read: Option<crate::shared::dense_read::ReadBatch>,
    pub native_load: Option<NativeLoad>,
}

impl TargetFacts {
    /// Exact target allocation bytes charged by the selected inventory, excluding host embedding.
    pub fn device_weight_bytes(&self, home: EmbeddingHome) -> anyhow::Result<Vec<u64>> {
        use anyhow::Context;
        let (_, inventory) = self.weights.iter().find(|(candidate, _)| *candidate == home)
            .context("target embedding inventory absent")?;
        let mut bytes = vec![0u64; inventory.ranks];
        for owner in &inventory.owners {
            if let Device::Coordinator(rank) = owner.allocation.location.device {
                if owner.allocation.location == Location::coordinator(rank) {
                    bytes[rank as usize] = bytes[rank as usize].checked_add(owner.allocation.bytes)
                        .context("target resident byte overflow")?;
                }
            }
        }
        Ok(bytes)
    }

    pub fn checkpoint(checkpoint: &crate::plan::Checkpoint, layers: usize, gpus: usize,
        attention_owners: Option<&[u8]>, native_bf16: bool, embedding: &[EmbeddingHome],
        core: Option<&crate::families::glm5::core::CoreCharges>) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        ensure!((1..=2).contains(&gpus) && !embedding.is_empty()
            && embedding.iter().enumerate().all(|(i, home)| !embedding[..i].contains(home))
            && core.is_none_or(|core| core.ranks.len() == gpus), "GLM target fact identity");
        let cfg = GlmDsaConfig::from_hf(&checkpoint.config)?;
        let weights = embedding.iter().map(|&home| Ok((home, WeightInventory::from_checkpoint(
            checkpoint, gpus, layers, attention_owners, native_bf16, false, home)?)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let hooks = (0..gpus).map(|rank| HookBuffers { device: rank as u8,
            maximum_pages: 1, native: None, core: core.map(|core| core.ranks[rank].clone()),
            shared_workspace: None, full_prefill_logits: false }).collect();
        let row = cfg.hidden.checked_mul(2).context("GLM embedding row overflow")?;
        let embedding_read = Some(crate::shared::dense_read::ReadBatch::new(
            cfg.vocab_size, row, row, crate::shared::dense_read::STEP_BYTES)?);
        Ok(Self { weights, hooks, embedding_read, native_load: None })
    }

    pub fn native(selection: &crate::plan::exlr8::dense::DenseSelection, layers: usize,
        dense: &[crate::plan::exlr8::dense::DensePlan],
        vocabulary: &[crate::plan::exlr8::dense::DensePlan],
        shared: &[Vec<crate::plan::exlr8::residency::Residency>],
        scratch: &[[crate::families::glm5::native::ScratchLayout; 2]],
        workspace: &crate::plan::exlr8::ExecutionWorkspace,
        storage: super::local_physical::LocalStorage) -> anyhow::Result<Self> {
        use anyhow::ensure;
        let gpus = dense.len();
        ensure!((1..=2).contains(&gpus) && vocabulary.len() == gpus
            && shared.len() == gpus && scratch.len() == gpus, "GLM native target fact identity");
        let mut weights = WeightInventory::from_native(selection, layers, dense)?;
        weights.extend_native(selection, layers, &vocabulary.iter().collect::<Vec<_>>())?;
        weights.extend_shared(shared)?;
        let load = NativeLoad::new(&dense.iter().collect::<Vec<_>>(), &vocabulary[0], shared, storage)?;
        let hooks = scratch.iter().enumerate().map(|(rank, layouts)| HookBuffers {
            device: rank as u8, maximum_pages: 1, native: Some(layouts.clone()), core: None,
            shared_workspace: Some(workspace.clone()), full_prefill_logits: false,
        }).collect();
        Ok(Self { weights: vec![(EmbeddingHome::Device(0), weights)], hooks,
            embedding_read: None, native_load: Some(load) })
    }
}

/// Itemized drafter owners. The library-wide synchronous upload bank is handed
/// back to the target assembler so all its users combine by max, not addition.
#[derive(Debug, Clone, Default)]
pub struct DraftFacts {
    owners: Vec<Allocation>,
    weights: Vec<ResidentWeight>,
    arena: Option<crate::serving_capacity::WeightArenaLayout>,
    arena_weights: u64,
    loading: Vec<Sequential>,
    prefetch: Option<DflashPrefetch>,
    /// Target-side staging and exchange controls of the shared device-driven
    /// bridge that runs this drafter on GPU1 while GPU0 keeps the embedding and
    /// head. Charged only when the solver selects the remote home.
    bridge: Vec<crate::placement::Demand>,
    pub sync_upload_bytes: u64,
}

impl DraftFacts {
    pub fn gqa(snapshot: &std::path::Path,
        representation: crate::families::glm5::draft_representation::GlmDraftRepresentation,
        capacity: crate::families::glm5::draft_representation::GlmDraftCapacity,
        sm_count: usize, fp8_mode: u8, kv_cache: crate::families::qwen4::Qwen4KvCache)
        -> anyhow::Result<Self> {
        let layout = crate::families::glm5::draft_physical::GlmDflashPhysicalLayout::read(
            snapshot, representation, capacity, sm_count, fp8_mode, kv_cache)?;
        let mut facts = Self::from_gqa(&layout)?;
        facts.pack_arena(crate::families::glm5::draft_physical::gqa_checkpoint_arena_requests(
            snapshot, representation, 0, 1)?)?;
        facts.prefetch = Some(DflashPrefetch::from_snapshot(snapshot)?);
        let cfg = crate::families::glm5::draft_physical::GlmDflashConfig::read(snapshot)?;
        facts.bridge = Self::bridge_demands(cfg.hidden, cfg.vocab, cfg.taps.len(), capacity.block_rows)?;
        Ok(facts)
    }

    fn from_gqa(layout: &crate::families::glm5::draft_physical::GlmDflashPhysicalLayout)
        -> anyhow::Result<Self> {
        use anyhow::Context;
        use cuteafd_core::memory_layout::Category;
        let mut facts = Self { owners: layout.allocations("glm.draft", Location::coordinator(0)),
            ..Self::default() };
        let mut source = Sequential { id: "glm.draft.load.sources".into(), members: vec![] };
        for owner in layout.weight_owners() {
            let mut weight = ResidentWeight::from_allocation(owner, crate::placement::ResidentWeights(owner.bytes));
            weight.id = format!("glm.draft.{}", owner.id);
            facts.weights.push(weight);
            facts.sync_upload_bytes = facts.sync_upload_bytes.max(owner.bytes);
            if let Some(matrix) = owner.id.strip_suffix(".values") {
                let bytes = owner.bytes.checked_mul(2).context("GLM draft BF16 source overflow")?.max(256);
                facts.sync_upload_bytes = facts.sync_upload_bytes.max(bytes);
                source.members.push(Allocation::buffer(format!("glm.draft.load.{matrix}.source"),
                    Location::coordinator(0), bytes,
                    Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad }, Category::Staging,
                    "prefetched GQA draft FP8 conversion source"));
            }
        }
        if !source.members.is_empty() { facts.loading.push(source); }
        Ok(facts)
    }

    /// ML4 owns a different native representation, not the standard GQA layout.
    /// Lazy diagnostic inputs are absent from normal device-input serving.
    pub fn ml4(layout: &crate::formats::dflash::MlaDraftRuntimeLayout) -> anyhow::Result<Self> {
        use crate::formats::dflash::{MlaDraftDomain, MlaDraftDtype, MlaDraftPacking};
        use anyhow::{ensure, Context};
        use cuteafd_core::memory_layout::Category;
        ensure!(layout.kind() == cuteafd_core::coordinator_programs::MlaDraftKind::Ml4Dspark,
            "GLM ML4 facts require the bundled mtp_dspark layout");
        let mut facts = Self::default();
        for owner in layout.owners().iter().filter(|owner| !owner.on_use) {
            if owner.shared_key.as_deref() == Some("library/sync-h2d") {
                facts.sync_upload_bytes = facts.sync_upload_bytes.max(owner.bytes as u64);
                continue;
            }
            let mut allocation = Allocation::buffer(format!("glm.draft.{}", owner.id),
                match owner.domain { MlaDraftDomain::Device => Location::coordinator(0),
                    MlaDraftDomain::HostPinned => Location::PINNED }, owner.bytes as u64,
                Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, Category::Drafter, "ML4 draft owners");
            allocation.alignment = owner.alignment as u64;
            allocation.dtype = match owner.dtype { MlaDraftDtype::Bytes => "bytes", MlaDraftDtype::Bf16 => "bf16",
                MlaDraftDtype::F32 => "f32", MlaDraftDtype::I32 => "i32", MlaDraftDtype::U32 => "u32",
                MlaDraftDtype::I64 => "i64" }.into();
            allocation.shape = owner.shape.iter().map(|&n| n as u64).collect();
            if owner.resident_weights.0 > 0 {
                facts.weights.push(ResidentWeight::from_allocation(&allocation, owner.resident_weights));
            }
            facts.owners.push(allocation);
        }
        // One DenseLoad::Mount is reused for all bundled weight destinations.
        let mut pinned = 0;
        for weight in layout.weight_destinations() {
            if matches!(weight.packing, MlaDraftPacking::UnitScale | MlaDraftPacking::Uv) { continue; }
            if weight.packing == MlaDraftPacking::Uk {
                let heads = weight.shape[0];
                let shape = &weight.sources.first().context("ML4 Uk source")?.1;
                ensure!(heads > 0 && shape.len() == 2 && shape[0] % heads == 0, "ML4 absorption source geometry");
                let head = (shape[0] / heads).checked_mul(shape[1]).and_then(|n| n.checked_mul(2))
                    .context("ML4 absorption head overflow")?;
                let per = crate::shared::dense_read::STEP_BYTES / head;
                ensure!(per > 0, "ML4 absorption head exceeds Mount bank");
                pinned = pinned.max((per.min(heads) * head).max(crate::shared::dense_read::PINNED_FLOOR_BYTES));
            } else {
                for (_, shape) in &weight.sources {
                    let (rows, columns) = if shape.len() >= 2 { (shape[0], &shape[1..]) } else { (1, &shape[..]) };
                    let row = columns.iter().try_fold(2usize, |bytes, &n| bytes.checked_mul(n))
                        .context("ML4 draft read row overflow")?;
                    pinned = pinned.max(crate::shared::dense_read::ReadBatch::new(rows, row, row,
                        crate::shared::dense_read::STEP_BYTES)?.pinned_bytes());
                }
            }
        }
        if pinned > 0 {
            facts.owners.push(Allocation::buffer("glm.draft.load.mount", Location::PINNED, pinned as u64,
                Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad }, Category::Staging,
                "ML4 bundled draft Mount bank"));
        }
        facts.bridge = Self::bridge_demands(layout.target_hidden(), layout.target_vocab(),
            layout.target_taps().len(), layout.capacity().block_rows)?;
        facts.pack_arena(layout.immutable_arena_requests())?;
        Ok(facts)
    }

    fn pack_arena(&mut self, requests: Vec<(String, usize)>) -> anyhow::Result<()> {
        use anyhow::{ensure, Context};
        ensure!(self.arena.is_none(), "GLM draft arena supplied twice");
        let arena = crate::serving_capacity::WeightArenaLayout::new("glm/draft/0", requests)?;
        let ids = self.weights.iter().map(|weight| weight.id.clone()).collect::<BTreeSet<_>>();
        self.arena_weights = self.weights.iter().try_fold(0u64, |sum, weight|
            sum.checked_add(weight.weights.0).context("GLM draft logical weight overflow"))?;
        let removed = self.owners.iter().filter(|owner| ids.contains(&owner.id)
            || owner.id == "glm.draft.markov_norms").try_fold(0u64, |sum, owner|
                sum.checked_add(owner.bytes).context("GLM draft immutable owner overflow"))?;
        ensure!(removed == arena.payload_bytes() as u64,
            "GLM draft immutable owners differ from shared arena requests: {removed}/{}", arena.payload_bytes());
        self.owners.retain(|owner| !ids.contains(&owner.id) && owner.id != "glm.draft.markov_norms");
        self.weights.clear();
        self.arena = Some(arena);
        Ok(())
    }

    /// The shared remote DFlash bridge with the head on GPU0: tap, input,
    /// hidden and logit staging plus head scratch on GPU0, controls on both.
    fn bridge_demands(hidden: usize, vocab: usize, taps: usize, rows: usize)
        -> anyhow::Result<Vec<crate::placement::Demand>> {
        Ok(crate::placement::draft::remote_dflash_bridge_demands(hidden as u64, vocab as u64,
            taps as u64, rows as u64, 0)?.demands)
    }

    pub fn is_empty(&self) -> bool {
        self.owners.is_empty() && self.weights.is_empty() && self.arena.is_none()
            && self.loading.is_empty() && self.prefetch.is_none()
    }

    /// Ordered homes: GPU1 needs a second coordinator and the bridge inventory.
    pub fn homes(&self, gpus: usize) -> Vec<u8> {
        if self.is_empty() { vec![] } else if gpus == 2 && !self.bridge.is_empty() { vec![0, 1] } else { vec![0] }
    }

    /// Prefix window marks follow the drafter they snapshot.
    pub fn with_window_marks(mut self, bytes: u64) -> Self {
        if bytes > 0 {
            self.owners.push(Allocation::buffer("glm.draft.window_marks", Location::coordinator(0), bytes,
                Lifetime::TARGET, cuteafd_core::memory_layout::Category::Prefix, "GLM draft window marks"));
        }
        self
    }

    /// Owners in their GPU0-home form.
    pub fn owners(&self) -> &[Allocation] { &self.owners }

    /// The bridge's admitted owners for the runtime's exact allocation check.
    pub fn bridge(&self) -> &[crate::placement::Demand] { &self.bridge }

    /// Owners on `home`: device owners move with the drafter; host and pinned
    /// owners stay. A GPU1 home adds the bridge owners named by their own GPU.
    pub fn components(&self, home: u8, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero draft fact pool extent")); }
        if home > 1 || home == 1 && self.bridge.is_empty() { return Err(invalid("draft home has no executor")); }
        let moved = |mut location: Location| {
            if location == Location::coordinator(0) { location = Location::coordinator(home); }
            location
        };
        let mut out = physical_components(self.owners.iter().cloned().map(|mut owner| {
            owner.location = moved(owner.location); owner }), maximum_pages);
        if let Some(arena) = &self.arena {
            let id = format!("glm/draft/{home}");
            let location = Location::coordinator(home);
            // Push planning retains the drafter while target shared experts load.
            // One contract covers push and mount, so include that Load overlap.
            let lifetime = Lifetime::RESIDENT;
            let mut components = weight_arena_extent_components(&id, arena.bytes() as u64,
                location, lifetime, maximum_pages, cuteafd_core::memory_layout::MemoryKind::DrafterWeights)?;
            components.retain(|component| !matches!(component.kind, ComponentKind::ResidentWeight(_)));
            let root = Allocation::buffer(&id, location, arena.bytes() as u64,
                lifetime, cuteafd_core::memory_layout::Category::Drafter, "weight arena");
            components.push(Component::resident_weight(ResidentWeight::from_allocation(&root,
                crate::placement::ResidentWeights(self.arena_weights)),
                ByteCurve::fixed(self.arena_weights, maximum_pages)));
            out.extend(components);
        }
        out.extend(self.weights.iter().cloned().map(|mut weight| {
            weight.location = moved(weight.location);
            let bytes = ByteCurve::fixed(weight.weights.0, maximum_pages);
            Component::resident_weight(weight, bytes)
        }));
        out.extend(self.loading.iter().cloned().map(|mut group| {
            for member in &mut group.members { member.location = moved(member.location); }
            Component { activation: Activation::always(),
                bytes: ByteCurve::fixed(group.peak(), maximum_pages), kind: ComponentKind::Sequential(group) }
        }));
        if let Some(prefetch) = &self.prefetch { out.extend(prefetch.components(maximum_pages)?); }
        if home == 1 {
            out.extend(physical_components(self.bridge.iter().enumerate().map(|(index, demand)| {
                Allocation::buffer(format!("glm.draft.bridge.{index}.rank.{}", demand.gpu),
                    Location::coordinator(demand.gpu), demand.bytes,
                    Lifetime { first: Phase::DraftLoad, last: Phase::Growth }, demand.category, demand.group.clone())
            }), maximum_pages));
        }
        Ok(out)
    }
}

/// Immutable physical facts shared by offline planning and serving admission.
/// Topology, programs and the host availability sample are resolved before this
/// assembler runs; none of them may be changed by a failed allocation.
pub struct ProductionInput<'a> {
    pub expert_usage: bool,
    pub cfg: &'a GlmDsaConfig,
    pub layers: usize,
    pub context: u64,
    pub maximum_pool_tokens: u64,
    pub attention: AttentionPlacement,
    pub gpus: usize,
    pub prefill: PrefillShape,
    pub blas_head: serving::BlasHead,
    pub blas: Vec<serving::BlasOwners>,
    pub target: TargetFacts,
    pub draft: DraftFacts,
    pub scratch: [super::GlmScratch; 2],
    pub context_scratch: Option<super::GlmContext>,
    pub exchange_element_bytes: u64,
    pub peer_diagnostics: bool,
    pub local: Option<(&'a super::local_physical::LocalPhysicalSource,
        super::local_physical::LocalPhysicalInput)>,
    pub spark_ranks: usize,
    /// The prefix cache's pinned host tier is admitted for this layout, so the
    /// host copy engine's streams are part of the per-context stream inventory.
    pub prefix_host: bool,
    pub decode_link: Option<crate::placement::PreparedSparkLink>,
    pub prefill_link: Option<crate::placement::PreparedSparkLink>,
    pub skip_routed_experts: bool,
    pub operation: Operation,
    /// Largest synchronous upload from additional serving/media owners. Target
    /// and draft users join the same library-wide bank by max, not addition.
    pub sync_upload_bytes: u64,
    pub owners: Vec<Component>,
    pub budgets: Vec<Budget>,
}

/// Prepared physical alternatives, retained until the one shared solve selects
/// them. Enumerating finite owner geometries is not a family-private fit loop.
pub struct PreparedGlm<'a> {
    pub recipe: GlmRecipe<'a>,
    pub budgets: Vec<Budget>,
    local: BTreeMap<usize, super::local_physical::LocalPhysicalInventory>,
    decode_link: Option<crate::placement::PreparedSparkLink>,
    prefill_link: Option<crate::placement::PreparedSparkLink>,
    pub runtime: DeviceFigures,
    pub load_overcharge_bytes: u64,
}

pub struct AdmittedGlm {
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub placement: crate::placement::Placement,
    pub spark: crate::placement::SparkWorkingSet,
    pub solved: ResolvedPlan,
    pub local: super::local_physical::LocalPhysicalInventory,
    pub decode_link: Option<crate::placement::PreparedSparkLink>,
    pub prefill_link: Option<crate::placement::PreparedSparkLink>,
    pub graphs: StartupGraphs,
    pub runtime: DeviceFigures,
    pub host: HostFigures,
    pub load_overcharge_bytes: u64,
}

impl AdmittedGlm {
    pub fn load_overlap_note(&self) -> Option<String> {
        (self.load_overcharge_bytes > 0).then(|| format!("estimated native Load overlap: {} bytes conservatively added by charging native reader peak + shared reader peak instead of their serialized maximum; exact overcharge = min(native peak, shared peak)", self.load_overcharge_bytes))
    }

    /// Classify only selected figures after admission; never change charged bytes.
    pub fn census_recommendation(&self, identity: FigureIdentity<'_>) -> Result<Option<String>, Refusal> {
        use crate::placement::inventory::{FigureUse, census_recommendation};
        let layout = format!("{}; attention/{:?}; pool/{}; onboard/{:?}; sparks/{}; prefill/{:?}; embedding/{:?}",
            identity.layout, self.placement.attention_placement, self.solved.pool_tokens,
            self.solved.decisions.onboard_layers, self.spark.spark_ranks,
            self.solved.decisions.prefill, self.solved.decisions.embedding);
        let graphs = self.graphs.declared().iter().enumerate().map(|(rank, set)|
            set.fingerprint(rank as u8).map_err(Refusal::InvalidFact)).collect::<Result<Vec<_>, _>>()?;
        let host_graphs = graphs.join(",");
        let identity = FigureIdentity { layout: &layout, ..identity };
        let mut uses = Vec::new();
        for (rank, graph) in graphs.iter().enumerate() {
            let current = FigureIdentity { declared_graphs: graph, ..identity };
            uses.push(FigureUse { name: "graph instantiate", figure: self.graphs.figures()[rank], current });
            uses.push(FigureUse { name: "context/modules", figure: self.runtime.context()[rank], current });
            for (index, row) in self.runtime.stages().iter().enumerate().filter(|(_, row)| row.rank == rank) {
                uses.push(FigureUse { name: row.label, figure: self.runtime.stage_figure(index), current });
            }
        }
        let current = FigureIdentity { declared_graphs: &host_graphs, ..identity };
        uses.extend([FigureUse { name: "host load", figure: self.host.load(), current },
            FigureUse { name: "host steady", figure: self.host.steady(), current }]);
        Ok(census_recommendation(uses))
    }
}

/// Physical capacity and operator ceiling are independent inputs. Runtime
/// preweight samples are evidence, not another subtraction of these owners.
pub fn production_budgets(cards: &[(u64, u64)], headroom: u64, host: (u64, u64),
    host_headroom: u64, spark_bytes: &[u64]) -> Result<Vec<Budget>, Refusal> {
    if !(1..=2).contains(&cards.len()) || host.1 > host.0 || host.0 == 0 {
        return Err(invalid("invalid production physical capacities"));
    }
    let forecast = || Baseline::Forecast { occupied: 0,
        source: "physical capacity; all GLM owners and maintained runtime figures charged in recipe".into(),
        assumed: false };
    let mut budgets = cards.iter().enumerate().map(|(rank, &(physical, logical))| Budget {
        id: format!("rtx{rank}"), locations: vec![Location::coordinator(rank as u8)],
        physical_bytes: physical, logical_bytes: logical, baseline: forecast(), headroom,
    }).collect::<Vec<_>>();
    budgets.push(Budget { id: "coordinator host RAM".into(), locations: vec![Location::HOST, Location::PINNED],
        physical_bytes: host.0, logical_bytes: host.0,
        baseline: Baseline::Sampled { free: host.1, phase: Phase::Load,
            source: "host MemAvailable normalized to external occupancy before whole-process recipe".into() }, headroom: host_headroom });
    budgets.extend(spark_bytes.iter().enumerate().map(|(rank, &bytes)| Budget {
        id: format!("spark{rank}"), locations: vec![Location { device: Device::Spark(rank as u8), domain: Domain::Managed }],
        physical_bytes: bytes, logical_bytes: bytes, baseline: forecast(), headroom: 0,
    }));
    Ok(budgets)
}

pub fn production_links(cfg: &GlmDsaConfig, ranks: usize, prefill: PrefillShape,
    ingress: cuteafd_core::SparkIngress, rings: cuteafd_core::SparkRings,
    replies: SparkReplyFormats) -> Result<(Option<crate::placement::PreparedSparkLink>,
        Option<crate::placement::PreparedSparkLink>), Refusal> {
    use crate::placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates};
    if ranks == 0 { return Ok((None, None)); }
    let geometry = cuteafd_core::ExpertGeometry { hidden: cfg.hidden as u32, experts: cfg.experts as u32,
        topk: cfg.topk as u32, intermediate: cfg.moe_intermediate as u32, layers: cfg.layers as u32 };
    let endpoint = |name, shape: PrefillShape, prefill, reply| PreparedSparkLink::endpoint(name, 0,
        SparkEndpointOwners { ranks, geometry, endpoint: cuteafd_core::SparkEndpoint {
            capacity: u32::try_from(shape.rows).map_err(|_| Refusal::Overflow("GLM transport rows"))?,
            ingress, ring_depth: None, partial_reply_bytes: None }, rings },
        shape.waves as usize, prefill, true, SparkIntakeCandidates::Probed, reply, Lifetime::TARGET);
    Ok((Some(endpoint("glm.decode", PrefillShape { rows: super::DECODE_ROWS, waves: 1 }, false, replies.decode())?),
        Some(endpoint("glm.prefill", prefill, true, replies.prefill(geometry.partial_replies()))?)))
}

pub fn production_request(context: impl Into<crate::placement::planner::Context>, pool: Option<u64>, onboard: crate::placement::Onboard,
    prefill: PrefillShape, operation: Operation, resident_weight_limit: crate::placement::ResidentWeights,
    spark_reply_formats: SparkReplyFormats) -> Request {
    Request { context: context.into(), pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard { crate::placement::Onboard::Auto => Pin::Auto,
            crate::placement::Onboard::Layers(n) => Pin::Exact(n as u64) },
        prefill: Pin::Exact(prefill), operation, resident_weight_limit: Some(resident_weight_limit), spark_reply_formats }
}

fn physical_components(owners: impl IntoIterator<Item = Allocation>, maximum: u64)
    -> Vec<Component> {
    owners.into_iter().filter(|owner| owner.bytes > 0).map(|owner| {
        let bytes = ByteCurve::fixed(owner.bytes, maximum);
        Component::allocation(owner, bytes, false, maximum)
    }).collect()
}

fn production_counts(routed: usize, local: bool, waves: u64, skip: bool) -> Vec<usize> {
    if skip || routed == 0 || !local { return vec![0]; }
    (0..=routed).filter(|&count| count < routed || waves == 1).collect()
}

impl<'a> ProductionInput<'a> {
    pub fn prepare(self) -> Result<PreparedGlm<'a>, Refusal> {
        use super::local_physical::LocalPhysicalInventory;
        use crate::placement::{FfnMode, SparkIntakeMode, SparkIntakeOwners};
        if self.maximum_pool_tokens < self.context {
            return Err(Refusal::PoolBelowContext { tokens: self.maximum_pool_tokens, context: self.context });
        }
        if !(1..=2).contains(&self.gpus) || self.layers == 0 || self.layers > self.cfg.layers
            || self.context == 0
            || self.spark_ranks > 8 || self.target.weights.is_empty()
            || self.target.weights.iter().any(|(_, weights)| weights.ranks != self.gpus)
            || self.target.hooks.len() != self.gpus
            || self.target.hooks.iter().enumerate().any(|(rank, hook)| usize::from(hook.device) != rank)
            || self.local.as_ref().is_some_and(|(source, _)| source.world() != self.gpus)
            || self.target.native_load.as_ref().is_some_and(|load| load.gpus != self.gpus)
            || (self.spark_ranks == 0) != self.decode_link.is_none()
            || (self.spark_ranks == 0) != self.prefill_link.is_none()
            || self.skip_routed_experts && (self.local.is_some() || self.spark_ranks > 0)
            || self.spark_ranks == 0 && self.prefill.waves != 1
        {
            return Err(invalid("incomplete production geometry"));
        }
        let TargetFacts { weights, hooks, embedding_read, native_load } = self.target;
        let partitioned = self.attention == AttentionPlacement::Context;
        if self.attention != AttentionPlacement::Heads && self.gpus != 2
            || partitioned != self.context_scratch.is_some()
            || ![2, 4].contains(&self.exchange_element_bytes)
        {
            return Err(invalid("production attention or exchange geometry"));
        }
        let page_tokens = if partitioned { 2 * PAGE_ROWS } else { PAGE_ROWS };
        let maximum_pages = self.maximum_pool_tokens.div_ceil(page_tokens);
        let owners = (self.attention == AttentionPlacement::Layers)
            .then(|| super::attention_owners(self.cfg, self.layers))
            .transpose().map_err(|error| invalid(&error.to_string()))?;
        let attention = (0..self.layers).map(|layer| match self.attention {
            AttentionPlacement::Context => LayerMode::ContextSplit,
            AttentionPlacement::Heads if self.gpus == 2 => LayerMode::HeadSplit,
            AttentionPlacement::Heads => LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
            AttentionPlacement::Layers => LayerMode::Whole { gpu: owners.as_ref().unwrap()[layer],
                ffn: FfnMode::Split },
        }).collect::<Vec<_>>();
        let routed = self.layers.saturating_sub(self.cfg.first_moe_layer);
        // The current executor accepts full onboarding only in a one-wave
        // deployment; multiwave Spark layouts do not change shape after solve.
        let counts = production_counts(routed, self.local.is_some(), self.prefill.waves,
            self.skip_routed_experts);
        let experts = (!self.skip_routed_experts && routed > 0).then(|| ExpertProfile {
            layers: (self.cfg.first_moe_layer..self.layers).collect(),
            destination: (0..self.gpus as u8).collect(), counts: if self.spark_ranks == 0 {
                vec![routed]
            } else { counts.clone() },
            rank_sets: if self.spark_ranks == 0 { vec![] }
                else { vec![(0..self.spark_ranks as u8).collect()] },
        });
        if experts.as_ref().is_some_and(|profile| profile.rank_sets.is_empty()) && self.local.is_none() {
            return Err(invalid("no executable local package or Spark destination"));
        }
        let counts = experts.as_ref().map_or(vec![0], |profile| profile.counts.clone());
        // Source-reference bytes are fixed before admission. Classification
        // against the canonical resolved layout happens after the single solve.
        let reference = if self.gpus == 1 { MIN_LAYOUT } else { MAX_LAYOUT };
        let runtime = DeviceFigures::new(self.gpus, self.spark_ranks > 0, self.prefix_host)?;
        let host = HostFigures::for_layout(self.gpus, reference)?;
        let expert_usage = (self.expert_usage && !self.skip_routed_experts && routed > 0)
            .then(|| cuteafd_core::expert_usage::Geometry { layers: self.layers,
                experts: self.cfg.experts, topk: self.cfg.topk,
                routed_layers: (self.cfg.first_moe_layer..self.layers).collect() });
        let mut recipe = GlmRecipe { expert_usage, cfg: self.cfg, context: self.context, page_tokens,
            maximum_pages, attention, gpus: self.gpus, spark_links: self.spark_ranks > 0,
            prefill: vec![self.prefill], lanes: vec![1],
            embedding: weights.iter().map(|(home, _)| *home).collect(), draft_homes: self.draft.homes(self.gpus),
            encoders: vec![], media_cache_bytes: 0, blas: self.blas, experts,
            targets: vec![], hooks: vec![], exchanges: vec![], weights: vec![], graphs: vec![],
            host: Some(host), owners: vec![], };
        if self.blas_head.bf16() {
            let target = serving::BlasOwners { name: "GLM target vocabulary".into(),
                process: "coordinator".into(), thread: "glm/blas/serve".into(),
                location: Location::coordinator(0), gemm: self.blas_head.gemm(), lt: self.blas_head.lt(),
                lifetime: Lifetime::TARGET, activation: Activation::always() };
            recipe.blas.push(target.clone());
            if !self.draft.is_empty() {
                // A local draft or remote BF16 bridge owns its GEMM head on GPU0.
                recipe.blas.push(serving::BlasOwners { name: "GLM draft vocabulary bridge".into(),
                    gemm: true, lt: false, lifetime: Lifetime::RESIDENT, ..target });
            }
        }
        recipe.owners.push(Case { when: vec![], value: self.owners });
        if let Some(geometry) = &recipe.expert_usage {
            let mut owners = crate::placement::expert_usage::components(geometry, &[0], maximum_pages)?;
            owners.extend(super::usage_metadata_components(self.prefill.waves, maximum_pages)?);
            recipe.owners.push(Case { when: vec![], value: owners });
        }
        recipe.owners.push(Case { when: vec![], value: runtime.components(maximum_pages)? });
        for home in recipe.draft_homes.clone() {
            recipe.owners.push(Case { when: vec![When::DraftHome(home)],
                value: self.draft.components(home, maximum_pages)? });
        }
        if let Some(load) = &native_load {
            recipe.owners.push(Case { when: vec![], value: load.components(maximum_pages) });
        }
        for (home, weights) in &weights {
            recipe.weights.push(Case { when: vec![When::Embedding(*home)], value: weights.clone() });
            let filler = embedding_read.filter(|_| *home == EmbeddingHome::Device(0));
            let sync = self.sync_upload_bytes.max(self.draft.sync_upload_bytes)
                .max(native_load.as_ref().map_or(0, |load| load.sync_upload_bytes))
                .max(weights.core_loading.sync_upload_bytes)
                .max(weights.kv_b_loading.sync_upload_bytes)
                .max(weights.row_loading.sync_upload_bytes)
                .max(filler.map_or(0, |batch| batch.packed_bytes as u64));
            let lifetime = Lifetime { first: Phase::Load, last: Phase::PostTargetKv };
            let owners = [
                ("glm.load.sync_h2d", sync),
                ("glm.load.embedding.read", filler.map_or(0, |batch| batch.pinned_bytes() as u64)),
            ].into_iter().filter(|(_, bytes)| *bytes > 0).map(|(id, bytes)|
                Allocation::buffer(id, Location::PINNED, bytes, lifetime,
                    cuteafd_core::memory_layout::Category::Staging, "GLM load pinned banks"));
            recipe.owners.push(Case { when: vec![When::Embedding(*home)],
                value: physical_components(owners, maximum_pages) });
        }
        for mut hook in hooks {
            hook.maximum_pages = maximum_pages;
            hook.full_prefill_logits = self.operation == Operation::Scoring;
            recipe.hooks.push(Case { when: vec![], value: hook });
        }
        if self.gpus == 2 {
            let core_rows = recipe.hooks.iter().any(|hook| hook.value.core.is_some());
            let partial = recipe.hooks.iter().filter_map(|hook| hook.value.core.as_ref())
                .map(|core| core.attention_partial_bytes as u64).max().unwrap_or(0);
            let width = self.cfg.hidden as u64 * (self.exchange_element_bytes + if core_rows { partial } else { 0 });
            let receive = self.prefill.rows.max(super::DECODE_ROWS).checked_mul(width)
                .ok_or(Refusal::Overflow("GLM peer receive slots"))?;
            recipe.exchanges.push(Case { when: vec![When::Prefill(self.prefill)], value: ExchangeBuffers {
                physical_devices: [0, 1], maximum_pages,
                receive_bytes: vec![receive; (4 * self.prefill.waves) as usize],
                terminal_abort: true, diagnostics: self.peer_diagnostics,
                context: self.context_scratch.map(|scratch| super::exchanges::ContextOwners {
                    extent: self.context, prefill: self.prefill, scratch }),
            } });
        }
        let mut local = BTreeMap::new();
        for &count in &counts {
            let when = if recipe.experts.is_some() {
                vec![When::Onboard { first: count, last: count }]
            } else { vec![] };
            let inventory = match &self.local {
                Some((source, input)) => {
                    let mut input = input.clone();
                    input.layers = (self.cfg.first_moe_layer..self.cfg.first_moe_layer + count).collect();
                    if input.max_rows as u64 != self.prefill.rows.max(super::DECODE_ROWS)
                        || input.lanes as u64 != self.prefill.waves {
                        return Err(invalid("local owners differ from selected prefill geometry"));
                    }
                    source.inventory(&input).map_err(|error| invalid(&error.to_string()))?
                }
                None => LocalPhysicalInventory::default(),
            };
            let mut components = physical_components(inventory.owners.iter().chain(&inventory.views).cloned(), maximum_pages);
            // Checkpoint resource facts follow weight backings, never backend,
            // route or workspace bytes. Push views retain the weight identity.
            for owner in inventory.owners.iter().filter(|owner| owner.id.contains(".weight.")) {
                components.push(Component::resident_weight(ResidentWeight::from_allocation(owner,
                    crate::placement::ResidentWeights(owner.bytes)), ByteCurve::fixed(owner.bytes, maximum_pages)));
            }
            recipe.owners.push(Case { when: when.clone(), value: components });
            if count > 0 && self.gpus == 2 {
                for (home, weights) in &weights {
                    let mut replicated = weights.clone();
                    let start = replicated.owners.len();
                    replicated.add_peer_routers(&(self.cfg.first_moe_layer..self.cfg.first_moe_layer + count)
                        .collect::<Vec<_>>()).map_err(|error| invalid(&error.to_string()))?;
                    replicated.owners.drain(..start);
                    replicated.core_loading = Default::default();
                    replicated.kv_b_loading = Default::default();
                    replicated.row_loading = Default::default();
                    replicated.attention_mount = Default::default();
                    let mut condition = when.clone();
                    condition.push(When::Embedding(*home));
                    recipe.owners.push(Case { when: condition,
                        value: WeightInventory::standalone_components(&replicated.owners, maximum_pages) });
                }
            }
            for device in 0..self.gpus as u8 {
                recipe.targets.push(Case { when: {
                    let mut condition = when.clone(); condition.push(When::Prefill(self.prefill)); condition
                }, value: TargetBuffers { device, lead: device == 0, page_tokens, maximum_pages,
                    context: self.context, partitioned,
                    owned_layers: recipe.attention.iter().map(|mode| match mode {
                        LayerMode::Whole { gpu, .. } => *gpu == device, _ => true,
                    }).collect(), split_heads: self.gpus == 2 && owners.is_none(),
                    router: count > 0 && device == 1,
                    native_head: recipe.hooks[device as usize].value.core.as_ref()
                        .is_some_and(|core| core.target_head.is_some()),
                    decode: Some(self.scratch[0]), prefill: Some((self.prefill, self.scratch[1])),
                    full_prefill_logits: self.operation == Operation::Scoring,
                    payload_row_bytes: if count > 0 && self.gpus == 2 {
                        self.cfg.hidden as u64 * self.exchange_element_bytes
                    } else { 0 },
                    core_rows: recipe.hooks[device as usize].value.core.is_some(),
                    attention_partial_bytes: recipe.hooks[device as usize].value.core.as_ref()
                        .map_or(2, |core| core.attention_partial_bytes as u64),
                } });
            }
            let layers = (0..self.layers).map(|layer| GlmGraphLayer {
                dense: layer < self.cfg.first_moe_layer,
                local: (self.cfg.first_moe_layer..self.cfg.first_moe_layer + count).contains(&layer),
                attention_owner: owners.as_ref().map(|owners| owners[layer]),
            }).collect::<Vec<_>>();
            let return_ranks = if self.skip_routed_experts { super::SKIP_RANKS as usize }
                else if count < routed { self.spark_ranks } else { 0 };
            recipe.graphs.push(Case { when, value: StartupGraphs::new(self.context,
                self.attention, self.gpus, &layers, return_ranks)? });
            local.insert(count, inventory);
        }
        if self.skip_routed_experts {
            let owners = SparkIntakeOwners { gpu: 0, ranks: super::SKIP_RANKS,
                rows: self.prefill.rows.max(super::DECODE_ROWS), row_bytes: self.cfg.hidden as u64 * 2,
                waves: 1, mode: SparkIntakeMode::Host }.allocations("glm.diagnostic", Lifetime::TARGET)?;
            recipe.owners.push(Case { when: vec![], value: physical_components(owners, maximum_pages) });
        }
        for link in [&self.decode_link, &self.prefill_link].into_iter().flatten() {
            if link.geometry().intake.ranks != self.spark_ranks {
                return Err(invalid("prepared link rank count differs from deployment"));
            }
            if let Some(last) = counts.iter().copied().filter(|&count| count < routed).max() {
                recipe.owners.push(Case { when: vec![When::Onboard { first: 0, last }],
                    value: link.components(maximum_pages)? });
            }
        }
        Ok(PreparedGlm { recipe, budgets: self.budgets, local, runtime,
            load_overcharge_bytes: native_load.as_ref().map_or(0, |load| load.coarse_overcharge_bytes),
            decode_link: self.decode_link, prefill_link: self.prefill_link })
    }
}

impl PreparedGlm<'_> {
    pub fn admit(mut self, request: Request) -> Result<AdmittedGlm, Refusal> {
        let mut solved = crate::placement::planner::solve(request, self.budgets, &self.recipe)?;
        self.recipe.refine_pool_receipts(&mut solved)?;
        let count = solved.decisions.onboard_layers.iter().flatten().copied()
            .collect::<BTreeSet<_>>().len();
        let local = self.local.remove(&count).ok_or_else(|| invalid("selected local inventory was not prepared"))?;
        let (placement, spark) = self.recipe.execution(&solved, &local)?;
        let graph = self.recipe.graphs.into_iter().find(|case| case.when.iter().all(|condition|
            matches!(condition, When::Onboard { first, last } if *first <= count && count <= *last)))
            .ok_or_else(|| invalid("selected startup graph set was not prepared"))?.value;
        let remote = !spark.spark_layers.is_empty();
        Ok(AdmittedGlm { expert_usage: self.recipe.expert_usage.clone(), placement, spark, solved, local,
            decode_link: self.decode_link.filter(|_| remote),
            prefill_link: self.prefill_link.filter(|_| remote),
            graphs: graph, runtime: self.runtime, host: self.recipe.host.unwrap(),
            load_overcharge_bytes: self.load_overcharge_bytes })
    }
}

/// The current DFlash prefetch reads one entire file while target weights load.
/// Its ordinary Vec payload is an assumption, not an allocator/RSS receipt.
#[derive(Debug, Clone)]
pub struct DflashPrefetch {
    file_bytes: u64,
}

impl DflashPrefetch {
    pub fn from_snapshot(snapshot: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::{Context, ensure};
        let path = snapshot.join("model.safetensors");
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("DFlash prefetch file {}", path.display()))?;
        ensure!(
            metadata.is_file() && metadata.len() > 0,
            "DFlash prefetch needs a nonempty regular file: {}",
            path.display()
        );
        Ok(Self {
            file_bytes: metadata.len(),
        })
    }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 {
            return Err(invalid("zero DFlash prefetch pool extent"));
        }
        Ok(vec![Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(self.file_bytes, maximum_pages),
            kind: ComponentKind::Allowance(Allowance {
                location: Location::HOST,
                bytes: self.file_bytes,
                lifetime: Lifetime { first: Phase::Load, last: Phase::DraftLoad },
                source: "DFlash whole-file prefetch Vec payload: model.safetensors metadata.len(); overlaps target read/pack heap in Load and is released when drafter load returns; ordinary Rust heap, unmeasured assumption; excludes allocator/tensor-map metadata overhead".into(),
                assumed: true,
            }),
        }])
    }
}

const CENSUS_PROGRAMS: &str = "15a9aea77733f9be49e7871642e92fd65a64a46d4c7eb1711026307c90c391cf";
const CENSUS_NATIVE: &str = "f5b8d1e1371ae31c97164f08a6da5de293241ef0a4c1c89beeeba7c52a479513";
const CENSUS_TOOLCHAIN: &str = "rustc 1.98.1 (48a229cea 2026-09-01); binary: rustc; commit-hash: 48a229ceaefd4985c50990b14116b6d856af0985; commit-date: 2026-09-01; host: x86_64-unknown-linux-gnu; release: 1.98.1; LLVM version: 22.1.8";
const MIN_LAYOUT: &str = "glm53-exl3-k4/47af23347db743b4666d952e2eb48f2b01c3fede; dflash/425aa615ce320caac34400208b30808c8f14f76c/fp8; heads/1; sparks/4; onboard/0; context/1048576; compiled/1048576; pool/1257856; C16; prefill/2752x3; lanes/1; embedding/device; selector/device; replies/bf16,bf16; l2/auto; host-prefix/34359738368; instrumentation/census";
const MAX_LAYOUT: &str = "glm53-exl3-k4/47af23347db743b4666d952e2eb48f2b01c3fede; dflash/425aa615ce320caac34400208b30808c8f14f76c/fp8; context/2; sparks/6; onboard/3..9/tp2; context/1048576; compiled/1048576; pool/2097152; C16; prefill/2752x3; lanes/1; embedding/device; selector/device; replies/bf16,bf16; l2/auto; host-prefix/34359738368; instrumentation/census";

/// The HQA min layout declares the same startup key set as its EXL3 sibling
/// (`MIN_GRAPHS`), so the row is keyed by that fingerprint; this string records
/// the checkpoint and attention placement that actually produced the bytes.
const HQA_MIN_LAYOUT: &str = "glm53-hqa-k4-core-k6/354226e187c9f6c318a633aeee3a1bb327ae11f7/heads/1; sparks/4; onboard/0; context/1048576; compiled/1048576; C16; prefill/2752x3; lanes/1; instrumentation/census";
const MIN_GRAPHS: &str = "2bae8417cb9862ba77083453922d96de79b52db8de16a9ddebbd821d476afaf6";
const MAX_GRAPHS: [&str; 2] = [
    "b3acfa6e5d560b9f0440af62250a9f3b32424f9186deeb769f1a1ed0d5311543",
    "7f4662ddadd8137dc854f960c69c41469d82662e9e51073d7747ff4b31a61ccf",
];

const fn measured(bytes: u64, graphs: &'static str, layout: &'static str,
    commit: &'static str, evidence: &'static str) -> MemoryFigure {
    MemoryFigure { bytes, attribution: FigureAttribution::Measured, evidence,
        producing_commit: commit, identity: FigureIdentity { declared_graphs: graphs,
            programs_sha256: CENSUS_PROGRAMS, native_sha256: CENSUS_NATIVE,
            toolchain: CENSUS_TOOLCHAIN, gpu_class: "RTX PRO 6000 Blackwell/sm_120/188",
            driver: "13.2", layout } }
}

/// Measured parameters, #1242 min census (2026-11-04), retained under
/// builds/shared-planner-glm-census-min-4a56925e7/glm53-exl3-min-census-r1.
/// Context is normalized at preweight (context + streams + selected modules).
pub const GLM_MIN_CONTEXT: MemoryFigure = measured(598_999_040, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519", "#1242 glm53-exl3-min-census-r1/preweight");

/// Measured parameters, #1242 max census (2026-11-04), retained under
/// builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1.
/// Preserve logical-rank asymmetry; rank 0 alone captures the terminal head.
pub const GLM_MAX_CONTEXT: [MemoryFigure; 2] = [
    measured(594_804_736, MAX_GRAPHS[0], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/preweight/rank0"),
    measured(603_193_344, MAX_GRAPHS[1], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/preweight/rank1"),
];

/// One measured startup-graph residency: the summed
/// `CUTEAFD_GRAPH_CENSUS {"event":"capture"}` `instantiate_bytes` records of one
/// census device, over exactly the declared key set that produced `fingerprint`.
/// Graph instantiate storage is 2 MiB-quantized driver memory whose size follows
/// the captured kernels, not the key count: one fingerprint measured on two
/// builds differs (16,827 keys needed 3.73 GB at 9567fdf6 and 3.04 GB at
/// 5d180b5c8), so a row is never scaled to another key set or another build.
/// A declared set with no row is served on the best measured estimate for its
/// rank (#1605); a census replaces the estimate with the real row.
/// `commit` is the cuteafd SHA that produced the census and `sparkinfer` the
/// SparkInfer revision that build pinned; the bytes follow the fork, so a row
/// measured under another pin is reported loudly at plan time.
///
/// Measure a new row with the family census card, then sum that device's
/// `profile/graph-census-captures-*.jsonl` `instantiate_bytes` values:
///
/// ```text
/// scripts/bench/wip-cards.py --build --cards glm53-exl3-min \
///     --set glm53-exl3-min:PROFILE=census --set glm53-exl3-min:MEMORY_PADDING=2GiB \
///     --probe glm53-exl3-min:memory,console --panels none
/// ```
#[derive(Debug, Clone, Copy)]
struct MeasuredGraphSet {
    keys: usize,
    /// Logical rank whose census device produced these bytes.
    rank: u8,
    fingerprint: &'static str,
    bytes: u64,
    layout: &'static str,
    census: &'static str,
    commit: &'static str,
    sparkinfer: &'static str,
}

/// GLM 5.3 K6-HQA (#1242 census, revision 47af23347db743b4666d952e2eb48f2b01c3fede),
/// GLM 5.3 K6 at cuteafd 29212c52, and Mistral Large 4 at cuteafd 3baa1a4a and
/// 29212c52. Each byte figure is one device's capture sum; count and fingerprint
/// identify the set. Every row records both the cuteafd commit and the
/// SparkInfer revision of the build that measured it.
const MEASURED_GLM_GRAPHS: [MeasuredGraphSet; 8] = [
    MeasuredGraphSet { rank: 0, keys: 16_827, fingerprint: MIN_GRAPHS, bytes: 2_984_247_296, layout: HQA_MIN_LAYOUT,
        census: "glm53-hqa-min-glm5-r1 device 0: 16827 capture instantiate records; free 8245936128->5261688832",
        commit: "a5aa27a3f8968b6c25821f6baf6a10559b4ef5a6",
        sparkinfer: "851c42833da26711c68868014cad2fc043916416" },
    MeasuredGraphSet { rank: 0, keys: 16_827, fingerprint: MAX_GRAPHS[0], bytes: 3_042_967_552, layout: MAX_LAYOUT,
        census: "glm53-exl3-max-census-r1 device 0: 16827 capture instantiate records",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226",
        sparkinfer: "78cfee4e04e8d8055853945711e5f28ea27216e0" },
    MeasuredGraphSet { rank: 1, keys: 16_614, fingerprint: MAX_GRAPHS[1], bytes: 2_971_664_384, layout: MAX_LAYOUT,
        census: "glm53-exl3-max-census-r1 device 1: 16614 capture instantiate records",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226",
        sparkinfer: "78cfee4e04e8d8055853945711e5f28ea27216e0" },
    MeasuredGraphSet { rank: 0, keys: 16_827, fingerprint: "743e61e45b39ba6bdb04043401a1b5f9330863bda205bbb35615ff9512aea6dc",
        bytes: 3_732_930_560, layout: "glm53-exl3-k4/core-k6; attention/Context; sparks/6",
        census: "typed-tags-phase-c glm53-exl3-max-fixed-r1 device 0: 16827 capture instantiate records",
        commit: "29212c52f262afcda61c940f6c007e837d231057",
        sparkinfer: "9567fdf609b135968d6fb5ed647a6e0d1bcd8b9f" },
    MeasuredGraphSet { rank: 1, keys: 16_614, fingerprint: "b6f823f3b9defca4fcd752fab3567a19d8056921c5277355e8a5be925bae1d54",
        bytes: 3_986_685_952, layout: "glm53-exl3-k4/core-k6; attention/Context; sparks/6",
        census: "typed-tags-phase-c glm53-exl3-max-fixed-r1 device 1: 16614 capture instantiate records",
        commit: "29212c52f262afcda61c940f6c007e837d231057",
        sparkinfer: "9567fdf609b135968d6fb5ed647a6e0d1bcd8b9f" },
    MeasuredGraphSet { rank: 0, keys: 15_549, fingerprint: "5a2e6d744e3c72d4b192e3e8732ac7681eb2f4f9d44acfa8216420f9df693f70",
        bytes: 2_740_977_664, layout: "ml4-exl3; attention/Heads; 1 rank",
        census: "ml4 1x census memory-census-20261108T172510Z-raptor-coordinator-57-0",
        commit: "3baa1a4ab72f8e378c07abd0d22ac473db53a840",
        sparkinfer: "851c42833da26711c68868014cad2fc043916416" },
    MeasuredGraphSet { rank: 0, keys: 15_549, fingerprint: "bc9d9f1d011860a37be849433d80b8b4c728751d7a5bbd8e483d4e147d80c031",
        bytes: 3_445_620_736, layout: "ml4-exl3; attention/Context; 2 ranks",
        census: "typed-tags-phase-c ml4-exl3-max-fixed-r1 device 0: 15549 capture instantiate records",
        commit: "29212c52f262afcda61c940f6c007e837d231057",
        sparkinfer: "9567fdf609b135968d6fb5ed647a6e0d1bcd8b9f" },
    MeasuredGraphSet { rank: 1, keys: 15_336, fingerprint: "4ca4be3bc0b20179dc457a8eff3ec0d9425bb6a4ef1a6851955a9f6a9d4073aa",
        bytes: 3_720_347_648, layout: "ml4-exl3; attention/Context; 2 ranks",
        census: "typed-tags-phase-c ml4-exl3-max-fixed-r1 device 1: 15336 capture instantiate records",
        commit: "29212c52f262afcda61c940f6c007e837d231057",
        sparkinfer: "9567fdf609b135968d6fb5ed647a6e0d1bcd8b9f" },
];

/// The SparkInfer revision this build pins. Graph instantiate storage is the
/// fork's kernels, so a row measured under another pin is stale evidence.
fn pinned_sparkinfer_revision() -> &'static str {
    static REVISION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    REVISION.get_or_init(|| {
        let lock: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"), "/../../../third_party/sparkinfer.lock.json")))
            .expect("embedded SparkInfer lock");
        lock["revision"].as_str().expect("SparkInfer lock revision").to_owned()
    })
}

fn measured_graph_row(fingerprint: &str) -> Option<&'static MeasuredGraphSet> {
    MEASURED_GLM_GRAPHS.iter().find(|row| row.fingerprint == fingerprint)
}

/// One resolved graph row: the charge, its attribution and the warning the
/// caller reports (a stale pin, or an unmeasured fingerprint served from an
/// estimate). Admission never refuses for a missing row (#1605).
struct GraphRow {
    bytes: u64,
    basis: Basis,
    figure: MemoryFigure,
    warning: Option<String>,
}

/// The measured row for one declared set. A fingerprint with no row is served
/// with the family's best measured estimate for that rank (#1605), never
/// refused: the default-context row upper-bounds a lower context because
/// context can only go down from the checkpoint maximum, and a layout with no
/// comparable row charges the largest measured row.
fn graph_row(fingerprint: &str, keys: usize, rank: usize) -> GraphRow {
    if let Some(row) = measured_graph_row(fingerprint).filter(|row| row.keys == keys) {
        let pinned = pinned_sparkinfer_revision();
        let warning = (row.sparkinfer != pinned).then(|| format!(
            "GLM startup graph row {fingerprint} ({n} keys, rank-set {}) was measured at cuteafd {} \
             with SparkInfer {} but this build pins {pinned}; instantiate storage follows the fork, \
             so re-run this layout's memory census and update MEASURED_GLM_GRAPHS",
            row.layout, row.commit, row.sparkinfer, n = row.keys));
        return GraphRow { bytes: row.bytes, basis: Basis::Calibrated, warning,
            figure: measured(row.bytes, row.fingerprint, row.layout, row.commit, row.census) };
    }
    {
        // Unit fixtures build plans on synthetic geometries no census covers.
        // Production startup never enables this; see `allow_synthetic_graphs`.
        if SYNTHETIC_GRAPHS.with(std::cell::Cell::get) {
            // Size a synthetic geometry the way a census of it would: the
            // largest measured per-key rate above (3,720,347,648 B over 15,336
            // keys, the ML4 two-rank census), rounded up to 256 KiB per key. A
            // flat 2 MiB per declared graph overstated a 15.5k-key ML4 plan by
            // 26.6 GiB and turned its witness plan into a capacity refusal.
            let bytes = keys as u64 * (256 << 10);
            return GraphRow { bytes, basis: Basis::Estimated, warning: None,
                figure: MemoryFigure { bytes,
                    attribution: FigureAttribution::Estimated { derivation: "256 KiB per declared graph (largest measured per-key rate)" },
                    evidence: "synthetic unit fixture", producing_commit: "test",
                    identity: FigureIdentity { declared_graphs: "", programs_sha256: CENSUS_PROGRAMS,
                        native_sha256: CENSUS_NATIVE, toolchain: CENSUS_TOOLCHAIN,
                        gpu_class: "synthetic/sm_120", driver: "test", layout: "synthetic unit fixture" } } };
        }
        // A census run is how an unmeasured fingerprint gets its row: it needs
        // *some* charge to be admitted, and the live capture replaces it through
        // `reconcile_startup_graphs` before Ready. Charge the largest measured
        // row for the same declared count (or 2 MiB per executable when there is
        // none), never a proportional scale, and say so loudly.
        //
        // The pool is sized from this placeholder, so a live capture above it is
        // resident over the admitted ledger. Reconcile reports that budget gap
        // and serving continues (#1605): the KV pool is allocated before capture
        // (`GlmEngine::new`) and the captured graphs bake in its pointers, so it
        // can be neither shrunk in place nor reallocated. The census row itself
        // is still the measurement; the row's bytes do not depend on headroom.
        if std::env::var("CUTEAFD_GRAPH_CENSUS").is_ok_and(|value| value == "1") {
            let bytes = MEASURED_GLM_GRAPHS.iter().filter(|row| row.keys == keys)
                .map(|row| row.bytes).max()
                .unwrap_or_else(|| keys as u64 * (2 << 20));
            tracing::warn!(fingerprint, keys, bytes,
                "GLM startup graph fingerprint has no measured row; this census run charges the largest \
                 measured row for {keys} keys until the live capture reconciles it");
            return GraphRow { bytes, basis: Basis::Estimated, warning: None,
                figure: MemoryFigure { attribution: FigureAttribution::Estimated {
                        derivation: "unmeasured declared fingerprint; census run placeholder, reconciled from the live capture" },
                    ..measured(bytes, "", "unmeasured fingerprint; census run", "census",
                        "census run placeholder; reconciled from the live capture") } };
        }
    }
    // #1605: an unmeasured fingerprint is served, never refused. Charge the
    // best measured row for this logical rank, identically-keyed first: the
    // default-context census row is an upper bound for a lower context because
    // context cannot exceed the checkpoint maximum. A layout the family has no
    // comparable row for (e.g. ML4's layer placement) charges the largest
    // measured row. A runtime OOM on an unusual layout is the user's to fix
    // through GPU_MEMORY_CAP or GPU_RESERVE.
    let same_rank = || MEASURED_GLM_GRAPHS.iter().filter(|row| row.rank == rank as u8);
    let source = same_rank().filter(|row| row.keys == keys).max_by_key(|row| row.bytes)
        .or_else(|| same_rank().max_by_key(|row| row.bytes))
        .or_else(|| MEASURED_GLM_GRAPHS.iter().max_by_key(|row| row.bytes))
        .expect("measured graph rows are non-empty");
    GraphRow { bytes: source.bytes, basis: Basis::Estimated,
        warning: Some(format!(
            "GLM startup graph fingerprint {fingerprint} ({keys} keys, rank {rank}) has no measured row; \
             serving with the measured rank-{} row {} ({} bytes, {}) as an estimated upper bound; \
             run a memory census of this layout to measure it",
            source.rank, source.fingerprint, source.bytes, source.layout)),
        figure: MemoryFigure { bytes: source.bytes,
            attribution: FigureAttribution::Estimated { derivation: "unmeasured declared fingerprint; charged the measured row of the same rank as an upper bound (#1605)" },
            evidence: source.census, producing_commit: source.commit,
            identity: FigureIdentity { declared_graphs: source.fingerprint,
                programs_sha256: CENSUS_PROGRAMS, native_sha256: CENSUS_NATIVE,
                toolchain: CENSUS_TOOLCHAIN, gpu_class: "RTX PRO 6000 Blackwell/sm_120/188",
                driver: "13.2", layout: source.layout } } }
}

thread_local! {
    /// Unit fixtures plan synthetic geometries that no census covers. On by
    /// default in this crate's tests; another crate's fixture must opt in, and
    /// production startup never does, so an unmeasured fingerprint is charged a
    /// synthetic size. Production charges the best measured estimate instead.
    static SYNTHETIC_GRAPHS: std::cell::Cell<bool> = const { std::cell::Cell::new(cfg!(test)) };
}

/// Test fixtures in another crate whose planner geometry no census covers.
/// Never call this from serving code.
#[doc(hidden)]
pub fn allow_synthetic_graphs() { SYNTHETIC_GRAPHS.with(|flag| flag.set(true)); }

#[cfg(test)]
pub(crate) fn disallow_synthetic_graphs() { SYNTHETIC_GRAPHS.with(|flag| flag.set(false)); }

const MIN_CENSUS: &str = "builds/shared-planner-glm-census-min-4a56925e7/glm53-exl3-min-census-r1";
const MAX_CENSUS: &str = "builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1";

/// One device residency that no ledger allocation owns: the `cudaMemGetInfo`
/// used-minus-tracked delta across the named census stage transition, where the
/// named code path allocates it. `label` names the allocator, not a residual.
#[derive(Debug, Clone, Copy)]
struct MeasuredDeviceStage {
    ranks: usize,
    rank: usize,
    label: &'static str,
    first: Phase,
    last: Phase,
    bytes: u64,
    stage: &'static str,
    census: &'static str,
    evidence: &'static str,
    commit: &'static str,
}

/// Each row is charged exactly once, in the phase window where the memory is
/// resident. Stage deltas partition Ready's untracked bytes beside the preweight
/// context and the graph owner; no row overlaps another.
const MEASURED_DEVICE_STAGES: [MeasuredDeviceStage; 12] = [
    MeasuredDeviceStage { ranks: 1, rank: 0, label: "target weight load device staging", first: Phase::Load,
        last: Phase::DraftLoad, bytes: 143_391_104,
        stage: "programs-before-weights -> weights-before-draft-pool", census: MIN_CENSUS,
        evidence: "target weight load device staging: programs-before-weights -> weights-before-draft-pool; builds/shared-planner-glm-census-min-4a56925e7/glm53-exl3-min-census-r1",
        commit: "4a56925e7986a57e3194c0cca6df256838386519" },
    MeasuredDeviceStage { ranks: 1, rank: 0, label: "drafter load device staging", first: Phase::Load,
        last: Phase::Warmup, bytes: 123_296_064,
        stage: "weights-before-draft-pool -> draft-before-kv", census: MIN_CENSUS,
        evidence: "drafter load device staging: weights-before-draft-pool -> draft-before-kv; builds/shared-planner-glm-census-min-4a56925e7/glm53-exl3-min-census-r1",
        commit: "4a56925e7986a57e3194c0cca6df256838386519" },
    MeasuredDeviceStage { ranks: 1, rank: 0, label: "transport device staging", first: Phase::Load,
        last: Phase::Warmup, bytes: 90_144_768,
        stage: "pool-before-transport -> transport-before-graphs", census: MIN_CENSUS,
        evidence: "transport device staging: pool-before-transport -> transport-before-graphs; builds/shared-planner-glm-census-min-4a56925e7/glm53-exl3-min-census-r1",
        commit: "4a56925e7986a57e3194c0cca6df256838386519" },
    MeasuredDeviceStage { ranks: 1, rank: 0, label: "startup prewarm module first touch", first: Phase::Warmup,
        last: Phase::Growth, bytes: 72_615_168,
        stage: "transport-before-graphs -> graphs-before-serving, minus that transition's measured instantiate sum",
        census: MIN_CENSUS,
        evidence: "startup prewarm module first touch: transport-before-graphs -> graphs-before-serving, minus that transition's measured instantiate sum; builds/shared-planner-glm-census-min-4a56925e7/glm53-exl3-min-census-r1",
        commit: "4a56925e7986a57e3194c0cca6df256838386519" },
    MeasuredDeviceStage { ranks: 1, rank: 0, label: "first ready-step CUDA module touch", first: Phase::Ready,
        last: Phase::Growth, bytes: 69_045_248,
        stage: "graphs-before-serving -> ready", census: MIN_CENSUS,
        evidence: "first ready-step CUDA module touch: graphs-before-serving -> ready; builds/shared-planner-glm-census-min-4a56925e7/glm53-exl3-min-census-r1",
        commit: "4a56925e7986a57e3194c0cca6df256838386519" },
    MeasuredDeviceStage { ranks: 2, rank: 0, label: "target weight load device staging", first: Phase::Load,
        last: Phase::DraftLoad, bytes: 269_002_112,
        stage: "programs-before-weights -> weights-before-draft-pool", census: MAX_CENSUS,
        evidence: "target weight load device staging: programs-before-weights -> weights-before-draft-pool; builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226" },
    MeasuredDeviceStage { ranks: 2, rank: 0, label: "drafter load device staging", first: Phase::Load,
        last: Phase::Warmup, bytes: 77_117_248,
        stage: "weights-before-draft-pool -> draft-before-kv", census: MAX_CENSUS,
        evidence: "drafter load device staging: weights-before-draft-pool -> draft-before-kv; builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226" },
    MeasuredDeviceStage { ranks: 2, rank: 0, label: "transport device staging", first: Phase::Load,
        last: Phase::Warmup, bytes: 169_868_800,
        stage: "pool-before-transport -> transport-before-graphs", census: MAX_CENSUS,
        evidence: "transport device staging: pool-before-transport -> transport-before-graphs; builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226" },
    MeasuredDeviceStage { ranks: 2, rank: 0, label: "first ready-step CUDA module touch", first: Phase::Ready,
        last: Phase::Growth, bytes: 69_045_248,
        stage: "graphs-before-serving -> ready", census: MAX_CENSUS,
        evidence: "first ready-step CUDA module touch: graphs-before-serving -> ready; builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226" },
    MeasuredDeviceStage { ranks: 2, rank: 1, label: "target weight load device staging", first: Phase::Load,
        last: Phase::DraftLoad, bytes: 140_074_880,
        stage: "programs-before-weights -> weights-before-draft-pool", census: MAX_CENSUS,
        evidence: "target weight load device staging: programs-before-weights -> weights-before-draft-pool; builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226" },
    MeasuredDeviceStage { ranks: 2, rank: 1, label: "transport device staging", first: Phase::Load,
        last: Phase::Warmup, bytes: 27_965_952,
        stage: "pool-before-transport -> transport-before-graphs", census: MAX_CENSUS,
        evidence: "transport device staging: pool-before-transport -> transport-before-graphs; builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226" },
    MeasuredDeviceStage { ranks: 2, rank: 1, label: "first ready-step CUDA module touch", first: Phase::Ready,
        last: Phase::Growth, bytes: 71_303_168,
        stage: "graphs-before-serving -> ready", census: MAX_CENSUS,
        evidence: "first ready-step CUDA module touch: graphs-before-serving -> ready; builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1",
        commit: "5d180b5c8d76636d5937336638dfd71c15a8d226" },
];

/// Peak simultaneously-live CUDA streams in one coordinator GPU's context.
///
/// #1561 measured on raptor SM120 (driver/runtime 13020, LAZY) that retained
/// stream counts 2, 6, 10… grow device residency by 2 MiB, that destroying a
/// stream retains its slot, and that a later creation reuses it. So this counts
/// streams that are ever simultaneously live, never cumulative creations.
///
/// The inventory is the engine's compute stream (rank 1's is the head split's
/// peer stream), the prefix cache's one copy stream per rank (`shared/prefix`),
/// the host-prefix copy engine's copy and probe streams when that tier is
/// admitted (`shared/prefix/cuda_copy.rs`), the decode and prefill Spark intake
/// copy streams on the rank that owns the intake planes (`shared/spark_intake`),
/// and the shared delivery copies every coordinator load stages through
/// (`push_load::COPY_STREAMS`).
pub fn startup_peak_live_streams(rank: usize, spark_links: bool, host_prefix: bool) -> u64 {
    1 // the engine's compute stream; rank 1's is the head split's peer stream
        + 1 // the prefix cache's snapshot/restore copy stream, one per rank
        + 2 * u64::from(host_prefix) // host-prefix copy engine: copy + probe
        + 2 * u64::from(spark_links && rank == 0) // decode + prefill Spark intake copies
        + crate::push_load::COPY_STREAMS as u64
}

/// Every device residency a census found outside the ledger's named owners,
/// beside the preweight context figure. One named allocator per row, charged
/// once in the phase window where it is resident: no envelope, no margin, no
/// max-of-samples padding and no unmeasured "unattributed" rest.
#[derive(Debug, Clone)]
pub struct DeviceFigures {
    context: Vec<MemoryFigure>,
    stages: Vec<&'static MeasuredDeviceStage>,
    /// Stream-slot slabs per rank (#1561), charged as their own owner and
    /// removed from the context row that absorbed them, with the peak live
    /// stream count that produced each.
    slabs: Vec<(u64, u64)>,
}

impl DeviceFigures {
    pub fn new(gpus: usize, spark_links: bool, host_prefix: bool) -> Result<Self, Refusal> {
        if !(1..=2).contains(&gpus) {
            return Err(invalid("invalid device figure rank count"));
        }
        let context = if gpus == 1 { vec![GLM_MIN_CONTEXT] } else { GLM_MAX_CONTEXT.to_vec() };
        let slabs = (0..gpus).map(|rank| {
            let peak = startup_peak_live_streams(rank, spark_links, host_prefix);
            crate::placement::planner::cuda_stream_slab_bytes(peak).map(|bytes| (peak, bytes))
        }).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { context, stages: MEASURED_DEVICE_STAGES.iter().filter(|row| row.ranks == gpus).collect(),
            slabs })
    }

    pub fn context(&self) -> &[MemoryFigure] { &self.context }
    pub fn stages(&self) -> &[&'static MeasuredDeviceStage] { &self.stages }
    /// Slab bytes per rank.
    pub fn slabs(&self) -> Vec<u64> { self.slabs.iter().map(|&(_, bytes)| bytes).collect() }

    /// The row's receipt for the census recommendation; charged bytes live in
    /// [`Self::components`] only.
    pub fn stage_figure(&self, index: usize) -> MemoryFigure {
        let row = self.stages[index];
        measured(row.bytes, "", MIN_LAYOUT, row.commit, row.evidence)
    }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero device figure pool extent")); }
        // The preweight context figure normalizes context + streams + selected
        // modules, so the slab owner takes its bytes from that row rather than
        // charging the same driver residency twice.
        let mut components = self.context.iter().zip(&self.slabs).enumerate().map(|(rank, (figure, &(_, slab)))| {
            let bytes = figure.bytes.checked_sub(slab)
                .ok_or_else(|| invalid("GLM stream slabs exceed the measured context figure"))?;
            Ok(Component {
                activation: Activation::always(),
                bytes: ByteCurve::fixed(bytes, maximum_pages),
                kind: ComponentKind::Allowance(Allowance { location: Location::coordinator(rank as u8),
                    bytes, lifetime: Lifetime::RESIDENT,
                    source: format!("GLM context and selected modules: {:?}; {}; producer {}; less the \
                        {slab} B stream-slot slabs charged by glm/driver-stream-slabs/gpu{rank}",
                        figure.attribution, figure.evidence, figure.producing_commit),
                    assumed: matches!(figure.attribution, FigureAttribution::Estimated { .. }),
                }),
            })
        }).collect::<Result<Vec<_>, Refusal>>()?;
        for (rank, &(peak, slab)) in self.slabs.iter().enumerate() {
            components.push(Component { activation: Activation::always(),
                bytes: ByteCurve::fixed(slab, maximum_pages),
                kind: ComponentKind::Allowance(Allowance { location: Location::coordinator(rank as u8),
                    bytes: slab, lifetime: Lifetime::RESIDENT, assumed: false,
                    source: format!("glm/driver-stream-slabs/gpu{rank} ({peak} peak live streams, \
                        SM120 driver/runtime13020 LAZY, #1561 2026-11-08)") }) });
        }
        components.extend(self.stages.iter().map(|row| Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(row.bytes, maximum_pages),
            kind: ComponentKind::Allowance(Allowance { location: Location::coordinator(row.rank as u8),
                bytes: row.bytes, lifetime: Lifetime { first: row.first, last: row.last },
                source: format!("GLM {}: measured census stage delta {}; {}; producer {}",
                    row.label, row.stage, row.census, row.commit),
                assumed: false,
            }),
        }));
        Ok(components)
    }
}

/// Process-wide ordinary RSS from #1242, not one charge per coordinator GPU.
/// Warm-up samples are included in the load peak. Max uses its larger Ready
/// sample conservatively before Ready as well; these are sampled envelopes,
/// not independently qualified bounds on unobserved intra-phase peaks.
pub const GLM_MIN_HOST_LOAD: MemoryFigure = measured(5_759_569_920, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519",
    "#1242 glm53-exl3-min-census-r1/glm/weights-before-draft-pool/ordinary_rss_bytes");
pub const GLM_MIN_HOST_STEADY: MemoryFigure = MemoryFigure {
    attribution: FigureAttribution::Estimated {
        derivation: "Ready smaps_rss_bytes 40403427328 - file_mapping_resident_bytes 36902883328 = 3500544000 B; ordinary_rss_bytes missing and complete=false, not a raw Ready ordinary-RSS receipt",
    },
    ..measured(3_500_544_000, MIN_GRAPHS, MIN_LAYOUT,
        "4a56925e7986a57e3194c0cca6df256838386519",
        "#1242 glm53-exl3-min-census-r1/glm/ready/incomplete-host-reconstruction")
};
const MAX_HOST_GRAPHS: &str = "b3acfa6e5d560b9f0440af62250a9f3b32424f9186deeb769f1a1ed0d5311543,7f4662ddadd8137dc854f960c69c41469d82662e9e51073d7747ff4b31a61ccf";
pub const GLM_MAX_HOST_LOAD: MemoryFigure = measured(8_439_906_304, MAX_HOST_GRAPHS, MAX_LAYOUT,
    "5d180b5c8d76636d5937336638dfd71c15a8d226",
    "#1242 glm53-exl3-max-census-r1/glm/ready/ordinary_rss_bytes; maximum including startup warm-up");
pub const GLM_MAX_HOST_STEADY: MemoryFigure = measured(8_439_906_304, MAX_HOST_GRAPHS, MAX_LAYOUT,
    "5d180b5c8d76636d5937336638dfd71c15a8d226",
    "#1242 glm53-exl3-max-census-r1/glm/ready/ordinary_rss_bytes");

#[derive(Debug, Clone)]
pub struct HostFigures {
    gpus: usize,
    load: MemoryFigure,
    steady: MemoryFigure,
}

impl HostFigures {
    /// Other one-/two-coordinator layouts use the matching reference unchanged.
    /// Spark count does not scale process RSS; uncovered layouts stay estimated.
    pub fn for_layout(gpus: usize, layout: &str) -> Result<Self, Refusal> {
        if !(1..=2).contains(&gpus) || layout.trim().is_empty() {
            return Err(invalid("invalid host figure layout or rank count"));
        }
        let (load, steady, derivation) = if gpus == 1 {
            (GLM_MIN_HOST_LOAD, GLM_MIN_HOST_STEADY,
                "unchanged one-coordinator host envelope from source layout glm53-exl3-min-census-r1 (heads/1, sparks/4, onboard/0), #1242 retained coordinator log; no Spark-count scaling; min Ready remains an incomplete smaps-minus-file estimate")
        } else {
            (GLM_MAX_HOST_LOAD, GLM_MAX_HOST_STEADY,
                "unchanged two-coordinator host envelope from source layout glm53-exl3-max-census-r1 (context/2, sparks/6, onboard/3..9/tp2), #1242 retained coordinator log; no Spark-count scaling")
        };
        let derive = |figure: MemoryFigure| {
            if layout == figure.identity.layout { Ok(figure) }
            else { figure.scaled(1, 1, derivation).map_err(invalid) }
        };
        Ok(Self { gpus, load: derive(load)?, steady: derive(steady)? })
    }

    pub fn load(&self) -> MemoryFigure { self.load }
    pub fn steady(&self) -> MemoryFigure { self.steady }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero host figure pool extent")); }
        Ok([("load", self.load, Lifetime { first: Phase::Load, last: Phase::Warmup }),
            ("steady", self.steady, Lifetime { first: Phase::Ready, last: Phase::Growth })]
            .into_iter().map(|(name, figure, lifetime)| Component {
                activation: Activation::always(),
                bytes: ByteCurve::fixed(figure.bytes, maximum_pages),
                kind: ComponentKind::Allowance(Allowance {
                    location: Location::HOST, bytes: figure.bytes, lifetime,
                    source: format!("GLM process ordinary RSS {name}: {:?}; {}; source layout {}; producer {}",
                        figure.attribution, figure.evidence, figure.identity.layout, figure.producing_commit),
                    assumed: matches!(figure.attribution, FigureAttribution::Estimated { .. }),
                }),
            }).collect())
    }
}

/// Whole ordinary RSS already includes these physical heap subsets. Resource
/// facts remain independent, and pinned memory never belongs to this envelope.
/// The text-only census excludes the media cache's future embedding capacity.
fn ordinary_heap_subset(component: &Component) -> bool {
    match &component.kind {
        ComponentKind::Allocation { template, .. } => template.location == Location::HOST
            && !matches!(template.id.as_str(), "glm.media.cache" | "glm.scoring.host_logits"),
        ComponentKind::Sequential(group) => !group.members.is_empty()
            && group.members.iter().all(|owner| owner.location == Location::HOST),
        ComponentKind::Allowance(allowance) => allowance.location == Location::HOST,
        ComponentKind::Reservation(_) | ComponentKind::ResidentWeight(_)
            | ComponentKind::DriverPageTables { .. } | ComponentKind::AllocationGranularity(_) => false,
    }
}

/// Independently declared full-context keys plus one measured instantiate row
/// per rank. The row is keyed by that rank's declared fingerprint; an
/// undeclared fingerprint charges the best measured estimate for that rank,
/// warns once and serves (#1605), never refuses. No scaling, margin or invented
/// post-Ready reserve.
#[derive(Debug, Clone)]
pub struct StartupGraphs {
    context: u64,
    shapes: u64,
    attention: AttentionPlacement,
    layers: Vec<GlmGraphLayer>,
    declared: Vec<DeclaredGraphSet<GlmGraphKey>>,
    fingerprints: Vec<String>,
    ranks: Vec<GraphRank>,
    figures: Vec<MemoryFigure>,
}

impl StartupGraphs {
    pub fn new(context: u64, attention: AttentionPlacement, gpus: usize,
        layers: &[GlmGraphLayer], return_ranks: usize) -> Result<Self, Refusal> {
        if context == 0 { return Err(invalid("zero graph context")); }
        let extent = usize::try_from(context).map_err(|_| Refusal::Overflow("GLM graph context"))?;
        let shapes = glm_graph_shapes(extent, extent.div_ceil(PAGE_ROWS as usize));
        let declared = glm_declared_graphs(&shapes, attention, gpus, layers, return_ranks)
            .map_err(Refusal::InvalidFact)?;
        let (mut fingerprints, mut ranks, mut figures) = (Vec::new(), Vec::new(), Vec::new());
        let mut estimated = Vec::new();
        for (rank, keys) in declared.iter().enumerate() {
            let fingerprint = keys.fingerprint(rank as u8).map_err(Refusal::InvalidFact)?;
            let row = graph_row(&fingerprint, keys.len(), rank);
            match (row.basis, row.warning) {
                // One warning for the whole plan, not one per rank (#1605).
                (Basis::Estimated, Some(warning)) => estimated.push(warning),
                (_, Some(warning)) => tracing::warn!("{warning}"),
                _ => {}
            }
            figures.push(row.figure);
            ranks.push(GraphRank { executables: keys.len() as u64, bytes: row.bytes, margin: 0, basis: row.basis });
            fingerprints.push(fingerprint);
        }
        if !estimated.is_empty() {
            warn_once(&fingerprints.join(","), || format!(
                "GLM startup graphs are admitted on estimated rows (#1605); no census covers this declared set: {}",
                estimated.join("; ")));
        }
        Ok(Self { context, shapes: shapes.len() as u64, attention, layers: layers.to_vec(),
            declared, fingerprints, ranks, figures })
    }

    pub fn declared(&self) -> &[DeclaredGraphSet<GlmGraphKey>] { &self.declared }
    pub fn fingerprints(&self) -> &[String] { &self.fingerprints }
    pub fn ranks(&self) -> &[GraphRank] { &self.ranks }
    pub fn figures(&self) -> &[MemoryFigure] { &self.figures }

    /// The same inventory the daemon reconciles its live capture against.
    pub fn set(&self) -> GraphSet {
        GraphSet { ranks: self.ranks.clone(), lifetime: crate::placement::Lifetime::Startup, shapes: self.shapes }
    }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero graph pool extent")); }
        self.ranks.iter().enumerate().map(|(rank, graph)|
            crate::placement::planner::serving::startup_graph_component(
                &format!("glm.graphs.{rank}"), Location::coordinator(rank as u8), *graph, maximum_pages)
        ).collect()
    }

    /// The measured rows beside the live capture, so a build whose captured
    /// executable residency moved away from its row is reported, not absorbed.
    pub fn capture_note(&self, measured: &GraphSet) -> Option<String> {
        let drift: Vec<_> = self.ranks.iter().zip(&measured.ranks).enumerate()
            .filter_map(|(rank, (row, live))| (row.bytes != live.bytes)
                .then(|| format!("rank {rank}: row {} -> captured {}", row.bytes, live.bytes)))
            .collect();
        (!drift.is_empty()).then(|| format!(
            "GLM startup graph capture differs from the measured row ({}); update MEASURED_GLM_GRAPHS for this build",
            drift.join(", ")))
    }
}
/// One warning per declared set per process: a recipe prepares a case per
/// onboarding count, and an estimate is a report, not a fault (#1605).
fn warn_once(key: &str, message: impl FnOnce() -> String) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<BTreeSet<String>>> = std::sync::OnceLock::new();
    let mut seen = SEEN.get_or_init(Default::default).lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if seen.insert(key.to_owned()) { tracing::warn!("{}", message()); }
}

struct Domains<'a> {
    prefill: (usize, &'a [PrefillShape]),
    lanes: (usize, &'a [u64]),
    embedding: (usize, &'a [EmbeddingHome]),
    draft_home: Option<(usize, &'a [u8])>,
    onboard: Option<(usize, &'a ExpertProfile)>,
    remote: Option<usize>,
}

fn invalid(detail: &str) -> Refusal {
    Refusal::InvalidFact(format!("GLM component contract: {detail}"))
}

impl Domains<'_> {
    fn activation(&self, conditions: &[When]) -> Result<Activation, Refusal> {
        let mut domains = BTreeMap::<usize, BTreeSet<usize>>::new();
        for condition in conditions {
            let (variable, options): (_, Vec<_>) = match condition {
                When::Prefill(shape) => (
                    self.prefill.0,
                    self.prefill
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == shape).then_some(i))
                        .collect(),
                ),
                When::Lanes(lanes) => (
                    self.lanes.0,
                    self.lanes
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == lanes).then_some(i))
                        .collect(),
                ),
                When::Embedding(home) => (
                    self.embedding.0,
                    self.embedding
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == home).then_some(i))
                        .collect(),
                ),
                When::DraftHome(home) => {
                    let (variable, homes) = self.draft_home.ok_or_else(|| invalid("draft home without a drafter"))?;
                    (variable, homes.iter().enumerate().filter_map(|(i, value)| (value == home).then_some(i)).collect())
                }
                When::Onboard { first, last } => {
                    let (variable, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("count without routed experts"))?;
                    if first > last || *last > profile.layers.len() {
                        return Err(invalid("invalid onboarding interval"));
                    }
                    (
                        variable,
                        profile
                            .counts
                            .iter()
                            .enumerate()
                            .filter_map(|(i, count)| (first <= count && count <= last).then_some(i))
                            .collect(),
                    )
                }
                When::LocalLayer(layer) | When::RemoteLayer(layer) => {
                    let (variable, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("layer without routed experts"))?;
                    let position = profile
                        .layers
                        .iter()
                        .position(|value| value == layer)
                        .ok_or_else(|| invalid("condition names a non-target routed layer"))?;
                    let local = matches!(condition, When::LocalLayer(_));
                    (
                        variable,
                        profile
                            .counts
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &count)| ((position < count) == local).then_some(i))
                            .collect(),
                    )
                }
                When::SparkRanks(ranks) => {
                    let (_, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("ranks without routed experts"))?;
                    (
                        self.remote
                            .ok_or_else(|| invalid("ranks without remote homes"))?,
                        profile
                            .rank_sets
                            .iter()
                            .enumerate()
                            .filter_map(|(i, value)| (value == ranks).then_some(i))
                            .collect(),
                    )
                }
            };
            if options.is_empty() {
                return Err(invalid("condition has no domain value"));
            }
            let options: BTreeSet<_> = options.into_iter().collect();
            if let Some(existing) = domains.get_mut(&variable) {
                existing.retain(|option| options.contains(option));
                if existing.is_empty() {
                    return Err(invalid("contradictory component conditions"));
                }
            } else {
                domains.insert(variable, options);
            }
        }
        let mut clauses = vec![vec![]];
        for (variable, options) in domains {
            clauses = clauses
                .into_iter()
                .flat_map(|clause| {
                    options.iter().map(move |&option| {
                        let mut clause = clause.clone();
                        clause.push(Choice { variable, option });
                        clause
                    })
                })
                .collect();
        }
        Ok(Activation { clauses })
    }

    fn append(
        &self,
        contract: &mut Contract,
        conditions: &[When],
        components: Vec<Component>,
    ) -> Result<(), Refusal> {
        let activation = self.activation(conditions)?;
        for mut component in components {
            // Subrecipes cannot smuggle private variable indexes into the family contract.
            if component.activation != Activation::always() {
                return Err(invalid("subrecipe supplied a private activation"));
            }
            component.activation = activation.clone();
            contract.components.push(component);
        }
        Ok(())
    }
}

impl Recipe for GlmRecipe<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if !(1..=2).contains(&self.gpus)
            || self.attention.is_empty()
            || self.attention.len() > self.cfg.layers
            || self.context == 0
            || request.context.maximum == 0
            || request.context.maximum != self.context
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || self.page_tokens == 0
            || self.page_tokens % PAGE_ROWS != 0
            || self.prefill.is_empty()
            || self.lanes.is_empty()
            || self.embedding.is_empty()
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        self.page_tokens
            .checked_mul(self.maximum_pages)
            .ok_or(Refusal::Overflow("GLM pool extent"))?;
        for (i, shape) in self.prefill.iter().enumerate() {
            if !(64..=4096).contains(&shape.rows)
                || shape.rows % PAGE_ROWS != 0
                || !(1..=4).contains(&shape.waves)
                || self.prefill[..i].contains(shape)
            {
                return Err(invalid("invalid or duplicate prefill shape"));
            }
        }
        if self
            .lanes
            .iter()
            .enumerate()
            .any(|(i, lane)| *lane == 0 || self.lanes[..i].contains(lane))
            || self.embedding.iter().enumerate().any(|(i, home)| {
                self.embedding[..i].contains(home)
                    || matches!(home, EmbeddingHome::Device(gpu) if usize::from(*gpu) >= self.gpus)
            })
        {
            return Err(invalid("invalid or duplicate lane/embedding value"));
        }
        let partitioned = self
            .attention
            .iter()
            .any(|mode| *mode == LayerMode::ContextSplit);
        if self.attention.iter().any(|mode| match mode {
            LayerMode::HeadSplit | LayerMode::ContextSplit => self.gpus != 2,
            LayerMode::Whole { gpu, .. } => usize::from(*gpu) >= self.gpus,
        }) || partitioned && self.page_tokens % (2 * PAGE_ROWS) != 0
        {
            return Err(invalid("attention ownership or context page quantum"));
        }
        let mut contract = Contract { floor_fallback: None,
            resident_weight_limit: request.resident_weight_limit,
            phase_order: crate::placement::planner::PhaseOrder::DRAFT_THEN_TARGET,
            spark_reply_formats: self.spark_links.then(|| SparkReplyFormats {
                prefill: Some(
                    request
                        .spark_reply_formats
                        .prefill(cuteafd_core::ExpertGeometry::GLM5.partial_replies()),
                ),
                decode: Some(request.spark_reply_formats.decode()),
            }),
            page_rows: self.page_tokens,
            pool_page_multiple: 1,
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: self
                .attention
                .iter()
                .enumerate()
                .map(|(layer, &mode)| Variable::Attention {
                    layer,
                    values: vec![mode],
                })
                .collect(),
            requires: vec![],
            components: vec![],
        };
        let prefill = contract.variables.len();
        contract
            .variables
            .push(Variable::Prefill(self.prefill.clone()));
        let lanes = contract.variables.len();
        contract.variables.push(Variable::Lanes(self.lanes.clone()));
        let embedding = contract.variables.len();
        contract
            .variables
            .push(Variable::Embedding(self.embedding.clone()));
        if self.draft_homes.iter().enumerate().any(|(i, &gpu)|
            usize::from(gpu) >= self.gpus || self.draft_homes[..i].contains(&gpu)) {
            return Err(invalid("invalid or duplicate draft home"));
        }
        let draft_home = (!self.draft_homes.is_empty()).then(|| {
            contract.variables.push(Variable::DraftHome(self.draft_homes.clone()));
            (contract.variables.len() - 1, self.draft_homes.as_slice())
        });
        let onboard = self
            .experts
            .as_ref()
            .map(|profile| -> Result<_, Refusal> {
                if profile.layers.is_empty()
                    || profile.counts.is_empty()
                    || profile.layers
                        != (self.cfg.first_moe_layer..self.attention.len()).collect::<Vec<_>>()
                    || profile.layers.iter().collect::<BTreeSet<_>>().len() != profile.layers.len()
                    || profile.destination.is_empty()
                    || profile
                        .destination
                        .iter()
                        .any(|&gpu| usize::from(gpu) >= self.gpus)
                    || profile
                        .destination
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || profile
                        .counts
                        .iter()
                        .any(|&count| count > profile.layers.len())
                    || profile.counts.windows(2).any(|pair| pair[0] >= pair[1])
                    || profile.rank_sets.iter().enumerate().any(|(i, ranks)| {
                        ranks.is_empty()
                            || ranks.windows(2).any(|pair| pair[0] >= pair[1])
                            || profile.rank_sets[..i].contains(ranks)
                    })
                    || profile.rank_sets.is_empty() && profile.counts != [profile.layers.len()]
                {
                    return Err(invalid("invalid canonical expert profile"));
                }
                let variable = contract.variables.len();
                contract.variables.push(Variable::Onboard {
                    layers: profile.layers.clone(),
                    destination: profile.destination.clone(),
                    counts: profile.counts.clone(),
                });
                Ok((variable, profile))
            })
            .transpose()?;
        let remote = onboard.and_then(|(_, profile)| {
            (!profile.rank_sets.is_empty()).then(|| {
                let variable = contract.variables.len();
                contract.variables.push(Variable::ExpertHome {
                    layers: profile.layers.clone(),
                    rank_sets: profile.rank_sets.clone(),
                });
                variable
            })
        });
        let mut blas = self.blas.clone();
        if !self.encoders.is_empty() {
            use crate::plan::encoder::EncoderKind;
            let variable = contract.variables.len();
            contract.variables.push(Variable::Encoder((0..self.encoders.len()).collect()));
            for (option, encoder) in self.encoders.iter().enumerate() {
                let when = Choice { variable, option };
                let locations = match &encoder.kind {
                    EncoderKind::Off => vec![],
                    EncoderKind::Rtx { gpu } if *gpu < self.gpus => vec![Location::coordinator(*gpu as u8)],
                    EncoderKind::Spark { .. } => {
                        encoder.spark_ranks().into_iter().map(|rank| Location {
                            device: Device::Spark(rank as u8), domain: Domain::Managed }).collect()
                    }
                    _ => return Err(invalid("encoder device absent from executable topology")),
                };
                let mut components = Vec::new();
                for (copy, location) in locations.iter().copied().enumerate() {
                    blas.push(serving::BlasOwners { name: format!("GLM vision {option}/{copy}"),
                        process: match location.device { Device::Spark(rank) => format!("glm-media-worker-{rank}"),
                            _ => "coordinator".into() }, thread: "glm/blas/media".into(), location,
                        gemm: true, lt: false, lifetime: Lifetime::RESIDENT,
                        activation: Activation::choice(variable, option) });
                    for (part, bytes, category) in [("weights", encoder.weights, cuteafd_core::memory_layout::Category::Weights),
                        ("scratch", encoder.scratch, cuteafd_core::memory_layout::Category::Workspace)] {
                        if bytes == 0 { continue; }
                        let owner = Allocation::buffer(format!("glm.encoder.{option}.{copy}.{part}"), location,
                            bytes, Lifetime::RESIDENT, category, format!("bundled encoder {part}"));
                        if part == "weights" {
                            components.push(Component::resident_weight(ResidentWeight::from_allocation(
                                &owner, crate::placement::ResidentWeights(bytes)), ByteCurve::fixed(bytes, self.maximum_pages)));
                        }
                        components.push(Component::allocation(owner, ByteCurve::fixed(bytes, self.maximum_pages), false, self.maximum_pages));
                    }
                }
                if !locations.is_empty() && self.media_cache_bytes > 0 {
                    let owner = Allocation::buffer("glm.media.cache", Location::HOST, self.media_cache_bytes,
                        Lifetime::TARGET, cuteafd_core::memory_layout::Category::Embedding, "media embedding cache");
                    components.push(Component::allocation(owner, ByteCurve::fixed(self.media_cache_bytes, self.maximum_pages), false, self.maximum_pages));
                }
                for mut component in components {
                    component.activation = Activation::choice(variable, option);
                    contract.components.push(component);
                }
            }
        }
        for arch in ["sm_120", "sm_121"] {
            let owners = blas.iter().filter(|owner| matches!(owner.location.device, Device::Coordinator(_))
                == (arch == "sm_120")).cloned().collect::<Vec<_>>();
            contract.components.extend(serving::blas_components(arch, &owners,
                self.maximum_pages, contract.phase_order)?);
        }
        if request.operation == Operation::Scoring {
            for (option, shape) in self.prefill.iter().enumerate() {
                // Scoring's serial-lane calls download one unit directly; the
                // normal multi-wave prefill gather is not used by this path.
                let bytes = cuteafd_core::memory_layout::host_logits_bytes(shape.rows, shape.rows, self.cfg.vocab_size as u64)
                    .ok_or(Refusal::Overflow("scoring host logits"))?;
                let owner = Allocation::buffer("glm.scoring.host_logits", Location::HOST, bytes,
                    Lifetime::TARGET, cuteafd_core::memory_layout::Category::Workspace, "scoring host logits");
                let mut component = Component::allocation(owner, ByteCurve::fixed(bytes, self.maximum_pages), false, self.maximum_pages);
                component.activation = Activation::choice(prefill, option);
                contract.components.push(component);
            }
        }
        let domains = Domains {
            prefill: (prefill, &self.prefill),
            lanes: (lanes, &self.lanes),
            embedding: (embedding, &self.embedding),
            draft_home,
            onboard,
            remote,
        };
        for &shape in &self.prefill {
            let owner = super::buffers::sync_h2d_successor(shape)?;
            let bytes = ByteCurve::fixed(owner.bytes, self.maximum_pages);
            domains.append(
                &mut contract,
                &[When::Prefill(shape)],
                vec![Component::allocation(
                    owner,
                    bytes,
                    false,
                    self.maximum_pages,
                )],
            )?;
        }
        for case in &self.targets {
            let target = &case.value;
            let expected: Vec<_> = self
                .attention
                .iter()
                .map(|mode| match mode {
                    LayerMode::Whole { gpu, .. } => *gpu == target.device,
                    _ => true,
                })
                .collect();
            let whole = self
                .attention
                .iter()
                .any(|mode| matches!(mode, LayerMode::Whole { .. }));
            if usize::from(target.device) >= self.gpus
                || target.lead != (target.device == 0)
                || target.page_tokens != self.page_tokens
                || target.maximum_pages != self.maximum_pages
                || target.context != self.context
                || target.partitioned != partitioned
                || target.owned_layers != expected
                || target.split_heads != (self.gpus == 2 && !whole)
                || target.full_prefill_logits != (request.operation == Operation::Scoring)
                || target.prefill.is_some_and(|(shape, _)| {
                    !case
                        .when
                        .iter()
                        .any(|when| matches!(when, When::Prefill(value) if *value == shape))
                })
            {
                return Err(invalid(
                    "target geometry differs from declared execution domains",
                ));
            }
            domains.append(&mut contract, &case.when, target.components(self.cfg)?)?;
        }
        for case in &self.hooks {
            if usize::from(case.value.device) >= self.gpus
                || case.value.maximum_pages != self.maximum_pages
            {
                return Err(invalid("hook geometry differs from contract"));
            }
            domains.append(&mut contract, &case.when, case.value.components()?)?;
        }
        for case in &self.exchanges {
            if self.gpus != 2 || case.value.maximum_pages != self.maximum_pages
                || case.value.context.is_some() != partitioned
                || case.value.context.is_some_and(|context| context.extent != self.context
                    || !case.when.iter().any(|when| matches!(when, When::Prefill(value) if *value == context.prefill))) {
                return Err(invalid("exchange geometry differs from contract"));
            }
            domains.append(&mut contract, &case.when, case.value.components(self.cfg)?)?;
        }
        for case in &self.weights {
            if case.value.ranks != self.gpus {
                return Err(invalid("weight rank count differs from contract"));
            }
            let mut components = case.value.components(self.maximum_pages)?;
            components.extend(case.value.loading_components(self.maximum_pages)?);
            domains.append(&mut contract, &case.when, components)?;
        }
        for case in &self.graphs {
            let graph = &case.value;
            if graph.declared.len() != self.gpus
                || graph.context != self.context
                || graph.layers.len() != self.attention.len()
                || self.attention.iter().zip(&graph.layers).enumerate().any(|(index, (mode, layer))| {
                    layer.dense != (index < self.cfg.first_moe_layer) || match (graph.attention, mode) {
                        (AttentionPlacement::Heads, LayerMode::HeadSplit) => layer.attention_owner.is_some(),
                        (AttentionPlacement::Heads, LayerMode::Whole { gpu: 0, .. }) if self.gpus == 1 => layer.attention_owner.is_some(),
                        (AttentionPlacement::Context, LayerMode::ContextSplit) => layer.attention_owner.is_some(),
                        (AttentionPlacement::Layers, LayerMode::Whole { gpu, .. }) => layer.attention_owner != Some(*gpu),
                        _ => true,
                    }
                })
            {
                return Err(invalid("startup graph geometry differs from contract"));
            }
            let activation = domains.activation(&case.when)?;
            for clause in &activation.clauses {
                let counts: Vec<_> = if let Some((variable, profile)) = onboard {
                    profile
                        .counts
                        .iter()
                        .enumerate()
                        .filter(|(option, _)| {
                            clause.iter().all(|choice| {
                                choice.variable != variable || choice.option == *option
                            })
                        })
                        .map(|(_, &count)| count)
                        .collect()
                } else {
                    vec![0]
                };
                for count in counts {
                    let mut local = vec![false; self.attention.len()];
                    if let Some((_, profile)) = onboard {
                        for &layer in &profile.layers[..count] {
                            local[layer] = true;
                        }
                    }
                    if !graph.layers.iter().map(|layer| layer.local).eq(local) {
                        return Err(invalid("startup graph local declaration differs from activation"));
                    }
                }
            }
            domains.append(
                &mut contract,
                &case.when,
                graph.components(self.maximum_pages)?,
            )?;
        }
        for case in &self.owners {
            domains.append(&mut contract, &case.when, case.value.clone())?;
        }
        if let Some(host) = &self.host {
            if host.gpus != self.gpus {
                return Err(invalid("host figure rank count differs from contract"));
            }
            contract.components.retain(|component| !ordinary_heap_subset(component));
            contract.components.extend(host.components(self.maximum_pages)?);
        }
        Ok(contract)
    }
}

#[cfg(test)]
mod tests;
