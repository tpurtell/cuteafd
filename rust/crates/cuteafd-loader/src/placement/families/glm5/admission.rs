//! GLM physical component composition and the common plan/serve assembler.
//! Loading provenance never changes an owner's charge or its activation.
use super::{PAGE_ROWS, weights::WeightInventory};
use super::{buffers::TargetBuffers, exchanges::ExchangeBuffers, hooks::HookBuffers};
use crate::families::glm5::GlmDsaConfig;
use crate::placement::inventory::{DeclaredGraphSet, FigureAttribution, FigureIdentity, MemoryFigure};
use crate::placement::{AttentionPlacement, LayerMode, planner::*};
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
            shared_workspace: None }).collect();
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
            shared_workspace: Some(workspace.clone()),
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
        Ok(facts)
    }

    /// The shared remote DFlash bridge with the head on GPU0: tap, input,
    /// hidden and logit staging plus head scratch on GPU0, controls on both.
    fn bridge_demands(hidden: usize, vocab: usize, taps: usize, rows: usize)
        -> anyhow::Result<Vec<crate::placement::Demand>> {
        Ok(crate::placement::draft::remote_dflash_bridge_demands(hidden as u64, vocab as u64,
            taps as u64, rows as u64, 0)?.demands)
    }

    pub fn is_empty(&self) -> bool {
        self.owners.is_empty() && self.weights.is_empty() && self.loading.is_empty() && self.prefetch.is_none()
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
    pub target: TargetFacts,
    pub draft: DraftFacts,
    pub scratch: [super::GlmScratch; 2],
    pub context_scratch: Option<super::GlmContext>,
    pub exchange_element_bytes: u64,
    pub peer_diagnostics: bool,
    pub local: Option<(&'a super::local_physical::LocalPhysicalSource,
        super::local_physical::LocalPhysicalInput)>,
    pub spark_ranks: usize,
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
    pub runtime: RuntimeFigures,
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
    pub runtime: RuntimeFigures,
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
            for (name, figure) in [("graph instantiate", self.graphs.figures()[rank]),
                ("context/modules", self.runtime.context()[rank]), ("load residual", self.runtime.load()[rank]),
                ("runtime residual", self.runtime.unattributed()[rank])] {
                uses.push(FigureUse { name, figure, current });
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

pub fn production_request(context: u64, pool: Option<u64>, onboard: crate::placement::Onboard,
    prefill: PrefillShape, operation: Operation, resident_weight_limit: crate::placement::ResidentWeights,
    spark_reply_formats: SparkReplyFormats) -> Request {
    Request { context, pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
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
        let runtime = RuntimeFigures::new(self.gpus, reference)?;
        let host = HostFigures::for_layout(self.gpus, reference)?;
        let expert_usage = (self.expert_usage && !self.skip_routed_experts && routed > 0)
            .then(|| cuteafd_core::expert_usage::Geometry { layers: self.layers,
                experts: self.cfg.experts, topk: self.cfg.topk,
                routed_layers: (self.cfg.first_moe_layer..self.layers).collect() });
        let mut recipe = GlmRecipe { expert_usage, cfg: self.cfg, context: self.context, page_tokens,
            maximum_pages, attention, gpus: self.gpus, spark_links: self.spark_ranks > 0,
            prefill: vec![self.prefill], lanes: vec![1],
            embedding: weights.iter().map(|(home, _)| *home).collect(), draft_homes: self.draft.homes(self.gpus),
            encoders: vec![], media_cache_bytes: 0, experts,
            targets: vec![], hooks: vec![], exchanges: vec![], weights: vec![], graphs: vec![],
            host: Some(host), owners: vec![], };
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
                    recipe.weights.push(Case { when: condition, value: replicated });
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
        let solved = crate::placement::planner::solve(request, self.budgets, &self.recipe)?;
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
/// Graph bytes are the signed immediate instantiate sum, not whole warm growth.
/// These are not independently qualified bounds, receipts or post-Ready Growth.
pub const GLM_MIN_CONTEXT: MemoryFigure = measured(598_999_040, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519", "#1242 glm53-exl3-min-census-r1/preweight");
pub const GLM_MIN_GRAPH: MemoryFigure = measured(2_560_622_592, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519", "#1242 glm53-exl3-min-census-r1/instantiate");

/// Measured parameters, #1242 max census (2026-11-04), retained under
/// builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1.
/// Preserve logical-rank asymmetry; rank 0 alone captures the terminal head.
pub const GLM_MAX_CONTEXT: [MemoryFigure; 2] = [
    measured(594_804_736, MAX_GRAPHS[0], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/preweight/rank0"),
    measured(603_193_344, MAX_GRAPHS[1], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/preweight/rank1"),
];
pub const GLM_MAX_GRAPH: [MemoryFigure; 2] = [
    measured(3_042_967_552, MAX_GRAPHS[0], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/instantiate/rank0"),
    measured(2_971_664_384, MAX_GRAPHS[1], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/instantiate/rank1"),
];

/// Measured #1242 peak over recorded startup phases after subtracting the
/// normalized preweight context, exact native backings and signed instantiate
/// bytes. Ready was the largest recorded remainder on all three ranks. These
/// are observed peaks, not independently qualified bounds or attributed owners.
pub const GLM_MIN_RUNTIME_UNATTRIBUTED: MemoryFigure = measured(621_275_328, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519", "#1242 glm53-exl3-min-census-r1/runtime-unattributed-peak");
pub const GLM_MAX_RUNTIME_UNATTRIBUTED: [MemoryFigure; 2] = [
    measured(912_069_460, MAX_GRAPHS[0], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/runtime-unattributed-peak/rank0"),
    measured(581_053_972, MAX_GRAPHS[1], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/runtime-unattributed-peak/rank1"),
];

/// Recorded Load..Warmup residual peaks, including graphs-before-serving after
/// subtracting signed instantiate bytes. Unknown FP8 conversion/upload device
/// temporaries are covered opaquely, not enumerated owners; samples do not bound
/// unobserved intra-operation peaks. Other layouts require a census.
pub const GLM_MIN_LOAD_UNATTRIBUTED: MemoryFigure = measured(552_230_080, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519", "#1242 glm53-exl3-min-census-r1/load-through-warmup-unattributed-peak");
pub const GLM_MAX_LOAD_UNATTRIBUTED: [MemoryFigure; 2] = [
    measured(843_024_212, MAX_GRAPHS[0], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/load-through-warmup-unattributed-peak/rank0"),
    measured(509_750_804, MAX_GRAPHS[1], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/load-through-warmup-unattributed-peak/rank1"),
];

/// Opaque non-graph runtime parameters, charged independently of physical
/// weights, payloads, pinned storage and graph instantiate bytes.
#[derive(Debug, Clone)]
pub struct RuntimeFigures {
    context: Vec<MemoryFigure>,
    load: Vec<MemoryFigure>,
    unattributed: Vec<MemoryFigure>,
}

impl RuntimeFigures {
    /// `layout` is the complete selected layout identity, not a graph count or
    /// placement name. Uncovered layouts keep the measured bytes unchanged and
    /// explicitly record the one-to-one derivation; staleness never adds bytes.
    pub fn new(gpus: usize, layout: &str) -> Result<Self, Refusal> {
        if !(1..=2).contains(&gpus) || layout.trim().is_empty() {
            return Err(invalid("invalid runtime figure layout or rank count"));
        }
        let context = if gpus == 1 { vec![GLM_MIN_CONTEXT] } else { GLM_MAX_CONTEXT.to_vec() };
        let unattributed = if gpus == 1 { vec![GLM_MIN_RUNTIME_UNATTRIBUTED] }
            else { GLM_MAX_RUNTIME_UNATTRIBUTED.to_vec() };
        let derive = |figures: Vec<MemoryFigure>, derivation| -> Result<_, Refusal> {
            figures.into_iter().map(|figure| {
                if layout == figure.identity.layout { Ok(figure) }
                else { figure.scaled(1, 1, derivation).map_err(invalid) }
            }).collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            context: derive(context,
                "unchanged measured min/max preweight context; different layout, no finer context measurement")?,
            load: derive(if gpus == 1 { vec![GLM_MIN_LOAD_UNATTRIBUTED] }
                else { GLM_MAX_LOAD_UNATTRIBUTED.to_vec() },
                "unchanged measured min/max Load..Warmup residual peak; unmodelled FP8 conversion/upload device temporaries, no finer load census")?,
            unattributed: derive(unattributed,
                "unchanged measured min/max recorded runtime-unattributed peak; different layout, no finer runtime measurement")?,
        })
    }

    pub fn context(&self) -> &[MemoryFigure] { &self.context }
    pub fn load(&self) -> &[MemoryFigure] { &self.load }
    pub fn unattributed(&self) -> &[MemoryFigure] { &self.unattributed }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero runtime figure pool extent")); }
        Ok([("context and selected modules", &self.context, Lifetime::RESIDENT),
            ("load unattributed (FP8 conversion/upload device temporaries; sampled, not fully modelled)",
                &self.load, Lifetime { first: Phase::Load, last: Phase::Warmup }),
            ("runtime unattributed", &self.unattributed,
                Lifetime { first: Phase::Ready, last: Phase::Growth })]
            .into_iter().flat_map(|(name, figures, lifetime)| {
                figures.iter().enumerate().map(move |(rank, figure)| Component {
                    activation: Activation::always(),
                    bytes: ByteCurve::fixed(figure.bytes, maximum_pages),
                    kind: ComponentKind::Allowance(Allowance {
                        location: Location::coordinator(rank as u8), bytes: figure.bytes,
                        lifetime,
                        source: format!("GLM {name}: {:?}; {}; producer {}",
                            figure.attribution, figure.evidence, figure.producing_commit),
                        assumed: matches!(figure.attribution, FigureAttribution::Estimated { .. }),
                    }),
                })
            }).collect())
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
            | ComponentKind::DriverPageTables { .. } => false,
    }
}

/// Independently declared full-context keys plus maintained whole-set figures.
/// No per-executable calibration, margin or invented post-Ready reserve.
#[derive(Debug, Clone)]
pub struct StartupGraphs {
    context: u64,
    attention: AttentionPlacement,
    layers: Vec<GlmGraphLayer>,
    declared: Vec<DeclaredGraphSet<GlmGraphKey>>,
    figures: Vec<MemoryFigure>,
}

impl StartupGraphs {
    pub fn new(context: u64, attention: AttentionPlacement, gpus: usize,
        layers: &[GlmGraphLayer], return_ranks: usize) -> Result<Self, Refusal> {
        if context == 0 { return Err(invalid("zero graph context")); }
        let extent = usize::try_from(context).map_err(|_| Refusal::Overflow("GLM graph context"))?;
        let declared = glm_declared_graphs(&glm_graph_shapes(extent, extent.div_ceil(PAGE_ROWS as usize)),
            attention, gpus, layers, return_ranks).map_err(Refusal::InvalidFact)?;
        let mut figures = Vec::with_capacity(gpus);
        for (rank, keys) in declared.iter().enumerate() {
            let source = if gpus == 1 { GLM_MIN_GRAPH } else { GLM_MAX_GRAPH[rank] };
            let identity = keys.fingerprint(rank as u8).map_err(Refusal::InvalidFact)?;
            figures.push(if identity == source.identity.declared_graphs { source } else {
                // Layers splits a layer into phase executables, not multiple whole
                // layers. Retain their real declarations but scale the whole-set
                // census by layer/shape equivalents, not phase-segment count.
                let equivalents = if attention == AttentionPlacement::Layers {
                    layers.len() * glm_graph_shapes(extent, extent.div_ceil(PAGE_ROWS as usize)).len()
                } else { keys.len() };
                source.scaled(equivalents as u64, if rank == 0 { 16_827 } else { 16_614 },
                    "whole measured min/max instantiate bytes * whole-layer shape equivalents / measured declared keys; phase segmentation is not per-executable calibration")
                    .map_err(|error| invalid(error))?
            });
        }
        Ok(Self { context, attention, layers: layers.to_vec(), declared, figures })
    }

    pub fn declared(&self) -> &[DeclaredGraphSet<GlmGraphKey>] { &self.declared }
    pub fn figures(&self) -> &[MemoryFigure] { &self.figures }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero graph pool extent")); }
        Ok(self.figures.iter().enumerate().map(|(rank, figure)| Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(figure.bytes, maximum_pages),
            kind: ComponentKind::Allowance(Allowance { location: Location::coordinator(rank as u8),
                bytes: figure.bytes, lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth },
                source: format!("GLM startup graph instantiate: {:?}; {}; exact {} declared keys at context {}; producer {}",
                    figure.attribution, figure.evidence, self.declared[rank].len(), self.context, figure.producing_commit),
                assumed: matches!(figure.attribution, FigureAttribution::Estimated { .. }),
            }),
        }).collect())
    }
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
            || request.context == 0
            || request.context != self.context
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
