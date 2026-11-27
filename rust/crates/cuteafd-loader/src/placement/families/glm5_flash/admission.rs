//! Finite GLM Flash physical alternatives; the shared planner owns selection.
use super::*;
use crate::placement::planner as p;

/// The coordinator context's peak live streams (#1511: SM120 retains 2 MiB per
/// four live streams; destroyed streams reuse the slots). One serving lane
/// (`glm5_flash/mod.rs` serving stream), the four delivery lanes
/// (`push_load::COPY_STREAMS` in `shared/memory/push_consumer.rs`), the packed
/// head's pack lane (`glm5_flash/head.rs`), the embedding verify lane
/// (`shared/token_io.rs`) and the expert-usage lane
/// (`cuteafd-ffi/src/shared/expert_usage.rs`) overlap during the weight load;
/// a split adds the peer lane (`glm5_flash/mod.rs` peer stream). The Spark
/// intake and prefix-cache lanes are created after the delivery lanes retire,
/// so they refill the same slots and never raise the peak.
pub fn startup_peak_live_streams(gpus: usize) -> u64 {
    8 + u64::from(gpus > 1)
}

/// Caller-resolved process owners. Preparing a recipe never samples memory.
#[derive(Debug, Clone)]
pub struct ProcessOwners {
    pub host: (u64, u64),
    pub host_headroom: u64,
    pub allocations: Vec<p::Allocation>,
    pub links: Vec<PreparedSparkLink>,
    /// Physical media ranks remain available when their expert package is absent.
    pub remote_experts: bool,
    /// EXLR8 weights plus all execution owners per onboarding count and rank.
    pub spark_candidates: Vec<Vec<u64>>,
    pub spark_rank_caps: Vec<u64>,
    /// Additional admitted media hosts, independent of expert peers.
    pub physical_sparks: usize,
    pub native_load: Option<NativeLoad>,
    pub mount_load: Option<crate::shared::dense_read::MountLoad>,
    pub encoders: Vec<crate::plan::encoder::EncoderPlacement>,
    pub media_cache_bytes: u64,
    pub alternate_resident: Option<Vec<GlmfResidentRank>>,
    /// Exact target layouts by embedding-home option, then coordinator rank.
    pub target_arenas: Vec<Vec<crate::serving_capacity::WeightArenaLayout>>,
    pub alternate_target_arenas: Vec<Vec<crate::serving_capacity::WeightArenaLayout>>,
    /// Native shared experts retain their independent format-owned backings.
    pub shared_arenas: Vec<crate::serving_capacity::WeightArenaLayout>,
    /// Drafter backings indexed by coordinator home (sharded only for TP2).
    pub draft_arenas: Vec<crate::serving_capacity::WeightArenaLayout>,
    pub push_required: bool,
    pub retained_expert_metadata: Vec<u64>,
    /// The drafter's standalone push group, after the target group has committed.
    pub draft_push: Option<DraftPush>,
    /// Fixed rank-local GQA owners, not a movable single-home drafter.
    pub draft_tp2: Vec<crate::families::glm5::draft_physical::GlmDflashPhysicalLayout>,
}

/// Owners of the drafter's standalone coordinator push group (draft loading only):
/// its pinned receive ring and staged FP8 pack windows, and the drafter GPU's
/// one-step device scratch the W8A16 packs read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftPush {
    ranks: usize,
    pub device_scratch: u64,
}

impl DraftPush {
    /// Header-only, or None outside a push namespace. Every staged step is one
    /// contiguous row window of one file, none overlapping, so the runtime
    /// `HostStaging` bound is min(total, ring windows + one frontier step).
    pub fn for_snapshot(snapshot: &std::path::Path,
        representation: crate::families::glm5::draft_representation::GlmDraftRepresentation)
        -> anyhow::Result<Option<Self>> {
        match crate::push_load::namespace(snapshot) {
            Ok(_) => {},
            Err(error) if error.kind() == std::io::ErrorKind::NotFound
                && error.raw_os_error().is_none() && snapshot.exists() => return Ok(None),
            Err(error) => return Err(error.into()),
        }
        let step = crate::shared::dense_read::STEP_BYTES as u64;
        let fp8 = representation == crate::families::glm5::draft_representation::GlmDraftRepresentation::Fp8Only;
        Ok(Some(Self { ranks: 1, device_scratch: if fp8 { step } else { 0 } }))
    }

    pub fn for_tp(mut self, ranks: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(matches!(ranks, 1 | 2), "DFlash push rank count");
        self.ranks = ranks;
        Ok(self)
    }

    /// Recipe owners on draft home `gpu`, alive only while the drafter loads.
    pub fn allocations(&self, gpu: u8) -> Result<Vec<p::Allocation>, p::Refusal> {
        let lifetime = p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::DraftLoad };
        let mut owners = Vec::new();
        if self.device_scratch > 0 {
            for rank in 0..self.ranks {
                let rank = gpu + rank as u8;
                owners.push(p::Allocation::buffer(format!("glmf.draft.push.scratch.{rank}"), p::Location::coordinator(rank),
                    self.device_scratch, lifetime, Category::Staging, "draft push step scratch"));
            }
        }
        Ok(owners)
    }
}

/// The same native reader extents are checked by the admitted executor.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeLoad {
    pub vocabulary: u64,
    pub mount_pinned: u64,
    pub mount_rows: u64,
    pub shared_pinned: u64,
    pub shared_scratch: u64,
    pub shared_retained: Vec<u64>,
    pub sync_upload: u64,
}

impl NativeLoad {
    pub fn from_plans(dense: &[crate::plan::exlr8::dense::DensePlan],
        vocabulary: &crate::plan::exlr8::dense::DensePlan,
        shared: &[Vec<crate::plan::exlr8::residency::Residency>]) -> anyhow::Result<Self> {
        use anyhow::Context;
        let (mount, rows) = crate::plan::exlr8::dense::DensePlan::mount_staging_bytes(
            &dense.iter().collect::<Vec<_>>())?;
        anyhow::ensure!(vocabulary.world == dense.len() && vocabulary.rank == 0 && shared.len() == dense.len(),
            "Flash native load rank identity");
        anyhow::ensure!(dense.iter().enumerate().all(|(rank, plan)| plan.rank == rank && plan.world == dense.len()),
            "Flash body reader identity");
        let mut out = Self { vocabulary: vocabulary.unit_staging_bytes()? as u64,
            mount_pinned: mount as u64, mount_rows: rows as u64,
            shared_retained: vec![0; dense.len()], ..Default::default() };
        for (rank, layers) in shared.iter().enumerate() {
            let mut seen = std::collections::BTreeSet::new();
            for layer in layers {
                anyhow::ensure!(layer.rank == rank && layer.world == dense.len()
                    && layer.set == crate::plan::exlr8::residency::ExpertSet::Shared
                    && seen.insert(layer.layer), "Flash shared reader identity");
                out.shared_pinned = out.shared_pinned.max(layer.mount_pinned_bytes()? as u64);
                out.shared_scratch = out.shared_scratch.max(layer.descriptor_host_bytes()? as u64);
                out.shared_retained[rank] = out.shared_retained[rank].checked_add(layer.host_bytes() as u64)
                    .context("Flash retained shared metadata overflow")?;
                out.sync_upload = out.sync_upload.max(layer.sync_upload_bytes()? as u64);
            }
        }
        out.mount_pinned.checked_add(out.shared_pinned).context("Flash reader extent overflow")?;
        Ok(out)
    }

    pub fn validate_vocabulary(&self, bytes: u64) -> anyhow::Result<()> {
        anyhow::ensure!(bytes == self.vocabulary, "Flash native vocabulary staging differs from admission: {bytes} != {}", self.vocabulary);
        Ok(())
    }

    pub fn validate_shared(&self, plans: &[crate::plan::exlr8::residency::Residency]) -> anyhow::Result<()> {
        use anyhow::{Context, ensure};
        let first = plans.first().context("Flash shared reader inventory is empty")?;
        let retained = self.shared_retained.get(first.rank).context("Flash shared reader rank")?;
        let mut actual = 0u64;
        for (index, plan) in plans.iter().enumerate() {
            ensure!(plan.world == self.shared_retained.len() && plan.rank == first.rank
                && plan.layer == first.layer + index
                && plan.set == crate::plan::exlr8::residency::ExpertSet::Shared,
                "Flash shared reader identity differs from admission");
            ensure!(plan.mount_pinned_bytes()? as u64 <= self.shared_pinned
                && plan.descriptor_host_bytes()? as u64 <= self.shared_scratch
                && plan.sync_upload_bytes()? as u64 <= self.sync_upload,
                "Flash shared reader scratch exceeds admission");
            actual = actual.checked_add(plan.host_bytes() as u64).context("Flash shared metadata overflow")?;
        }
        ensure!(actual <= *retained, "Flash shared metadata exceeds admission");
        Ok(())
    }

    fn allocations(&self, push: bool) -> Vec<p::Allocation> {
        let load = p::Lifetime { first: p::Phase::Load, last: p::Phase::Load };
        // Recording allocates no Mount/shared reader banks; vocabulary loading
        // still owns its staging. Descriptor setup runs on both storage routes.
        let readers = self.vocabulary.max(if push { 0 } else { self.mount_pinned + self.shared_pinned });
        let mut owners = [("readers", p::Location::PINNED, readers, load),
            ("rows", p::Location::HOST, if push { 0 } else { self.mount_rows }, load),
            ("shared.scratch", p::Location::HOST, self.shared_scratch, load),
            ("sync", p::Location::PINNED, self.sync_upload, p::Lifetime::RESIDENT)]
            .into_iter().filter(|(_, _, bytes, _)| *bytes > 0).map(|(name, location, bytes, lifetime)|
                p::Allocation::buffer(format!("glmf.load.{name}"), location, bytes, lifetime,
                    Category::Staging, "native reader storage")).collect::<Vec<_>>();
        owners.extend(self.shared_retained.iter().enumerate().filter(|(_, bytes)| **bytes > 0)
            .map(|(rank, &bytes)| p::Allocation::buffer(format!("glmf.load.shared.retained.{rank}"),
                p::Location::HOST, bytes, p::Lifetime::RESIDENT, Category::Weights, "shared expert metadata")));
        owners
    }
}

impl ProcessOwners {
    pub fn new(host: (u64, u64), host_headroom: u64) -> Self {
        Self { host, host_headroom, allocations: vec![], links: vec![], remote_experts: true, spark_candidates: vec![], spark_rank_caps: Vec::new(), physical_sparks: 0, native_load: None, mount_load: None,
            encoders: vec![], media_cache_bytes: 0, alternate_resident: None,
            target_arenas: vec![], alternate_target_arenas: vec![], shared_arenas: vec![], draft_arenas: vec![], push_required: false, retained_expert_metadata: vec![],
            draft_push: None, draft_tp2: vec![] }
    }

    pub fn with_spark_capacities(mut self, bytes: Vec<u64>) -> Self {
        self.spark_rank_caps = bytes;
        self
    }

    pub fn draft(&mut self, snapshot: &std::path::Path,
        representation: crate::families::glm5::draft_representation::GlmDraftRepresentation,
        gpus: usize, tp2: bool) -> anyhow::Result<()> {
        use crate::families::glm5::{draft_physical::{gqa_arena_requests, GlmDflashConfig},
            draft_representation::is_dspark};
        let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let block = is_dspark(&config);
        anyhow::ensure!(!tp2 || gpus == 2 && !block, "TP2 requires a two-rank GQA drafter");
        self.draft_arenas = (0..gpus).map(|home| {
            let requests = if block {
                crate::formats::block_dspark::block_dspark_arena_requests(snapshot, representation)?
            } else {
                gqa_arena_requests(&GlmDflashConfig::from_value(&config)?, representation,
                    if tp2 { home } else { 0 }, if tp2 { gpus } else { 1 })?
            };
            crate::serving_capacity::WeightArenaLayout::new(format!("glm5_flash/draft/{home}"), requests)
        }).collect::<anyhow::Result<_>>()?;
        Ok(())
    }

    /// Prepare the exact target backings before admission, including the
    /// alternate whole-MLA profile when the solver may choose its first owner.
    pub fn target(&mut self, checkpoint: &crate::plan::Checkpoint, i: &GlmfInputs<'_>,
        native: Option<(&crate::plan::exlr8::native::DenseFacts, &crate::plan::exlr8::residency::MetadataInput)>) -> anyhow::Result<()> {
        let layouts = |input: &GlmfInputs<'_>| crate::families::glm5_flash::weight_layout::target_options(
            checkpoint, input.cfg, &modes(input), input.gpus.len(), input.representation,
            input.resident.iter().map(|rank| rank.embedding).sum(), native.map(|(facts, _)| facts));
        self.target_arenas = layouts(i)?;
        if let Some(resident) = &self.alternate_resident {
            let mut alternate = i.clone();
            alternate.attention.first_owner ^= 1;
            alternate.resident = resident.clone();
            self.alternate_target_arenas = layouts(&alternate)?;
        }
        self.shared_arenas = if let Some((_, metadata)) = native {
            (0..i.gpus.len()).map(|rank| {
                let plans = (0..i.layers).filter(|&layer| !i.cfg.dense[layer]).map(|layer|
                    metadata.plan(layer, i.gpus.len(), rank, crate::plan::exlr8::residency::ExpertSet::Shared))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                crate::plan::exlr8::residency::shared_arena(&plans)
            }).collect::<anyhow::Result<Vec<_>>>()?
        } else { vec![] };
        Ok(())
    }

    pub fn serving(&mut self, i: &GlmfInputs<'_>, quota: u64, chunk: u64, host_embedding: u64,
        media: u64) -> Result<(), p::Refusal> {
        let lifetime = p::serving::ServingOwners::LIFETIME;
        self.allocations.extend(p::serving::HostPrefixOwners { quota_bytes: quota, chunk_bytes: chunk }
            .allocations("glmf.host.prefix", lifetime)?);
        let logits = cuteafd_core::memory_layout::host_logits_bytes(
            64,
            i.prefill_rows, i.cfg.vocab_size as u64).ok_or(p::Refusal::Overflow("Flash host logits"))?;
        for (name, bytes, category, location) in [("embedding", host_embedding, Category::Embedding, p::Location::PINNED),
            ("media", media, Category::Prefix, p::Location::HOST), ("logits", logits, Category::Workspace, p::Location::PINNED)] {
            if bytes > 0 { self.allocations.push(p::Allocation::buffer(format!("glmf.host.{name}"),
                location, bytes, if name == "embedding" { p::Lifetime::RESIDENT } else { lifetime },
                category, format!("host {name}"))); }
        }
        self.allocations.extend(p::serving::SelectorOwners { location: p::Location::coordinator(0),
            capacity: i.decode_rows as usize, vocab: i.cfg.vocab_size, device_sampling: true }
            .allocations("glmf.selector", lifetime)?);
        Ok(())
    }

    pub fn spark_link(&mut self, i: &GlmfInputs<'_>, bf16: bool) -> Result<(), p::Refusal> {
        if i.spark_ranks == 0 { return Ok(()); }
        let geometry = cuteafd_core::ExpertGeometry { hidden: i.cfg.hidden as u32, experts: i.cfg.experts as u32,
            topk: i.cfg.topk as u32, intermediate: i.cfg.moe_intermediate as u32, layers: i.cfg.layers as u32,
            activation: cuteafd_core::ExpertActivation::SiluClamp10 };
        self.links.push(PreparedSparkLink::endpoint("glmf.spark", 0, SparkEndpointOwners {
            ranks: i.spark_ranks, geometry, endpoint: cuteafd_core::SparkEndpoint {
                capacity: u32::try_from(i.prefill_rows.max(i.decode_rows)).map_err(|_| p::Refusal::Overflow("Flash wire rows"))?,
                ingress: if bf16 { cuteafd_core::SparkIngress::Bf16 } else { cuteafd_core::SparkIngress::Fp8K32 },
                ring_depth: None, partial_reply_bytes: None }, rings: cuteafd_core::SparkRings {
                    max_frame_bytes: 64 << 20, depth: 8, slot_bytes: 8 << 20, alignment: 4096 } },
            i.prefill_lanes as usize, true, true, SparkIntakeCandidates::Probed,
            i.spark_reply_format, p::Lifetime::TARGET)?);
        Ok(())
    }
}

pub fn host_geometry(i: &GlmfInputs<'_>) -> Result<cuteafd_core::prefix::HostCacheGeometry, p::Refusal> {
    let geometry = facts(i).map_err(invalid)?.geometry;
    let ranks = if matches!(i.attention.mode, AttentionPlacement::Heads | AttentionPlacement::Context) {
        &geometry.ranks[..1]
    } else { &geometry.ranks[..] };
    Ok(cuteafd_core::prefix::HostCacheGeometry { unit_rows: 256,
        page_bytes: ranks.iter().map(|rank| rank.persistent_unit_bytes
            * if i.attention.mode == AttentionPlacement::Context { 2 } else { 1 }
            + rank.glmf_pool_mark_backing_unit_bytes(i.pool_marks)).sum(),
        mark_bytes: geometry.ranks.iter().map(|rank| rank.retained_mark_bytes).sum(), draft_bytes: 0 })
}

/// One reused legacy dense reader; routed experts have independent owners.
pub fn mount_load(checkpoint: &crate::plan::Checkpoint, ranks: usize)
    -> anyhow::Result<crate::shared::dense_read::MountLoad> {
    anyhow::ensure!((1..=2).contains(&ranks), "Flash mount rank count");
    crate::shared::dense_read::MountLoad::from_headers(checkpoint.tensors.iter()
        .filter(|tensor| !tensor.meta.name.contains(".experts."))
        .map(|tensor| &tensor.meta), ranks)
}

/// Persistent EXLR8 source/index metadata, separate from temporary push staging.
pub fn retained_expert_metadata(cfg: &GlmNextConfig, ranks: usize,
    metadata: &crate::plan::exlr8::residency::MetadataInput) -> anyhow::Result<Vec<u64>> {
    use anyhow::Context;
    (0..ranks).map(|rank| (0..cfg.layers).filter(|&layer| !cfg.dense[layer]).try_fold(0u64, |sum, layer| {
        let plan = metadata.plan(layer, ranks, rank, crate::plan::exlr8::residency::ExpertSet::Routed)?;
        sum.checked_add(plan.host_bytes() as u64).context("Flash retained expert metadata overflow")
    })).collect()
}

pub struct GlmfRecipe {
    contract: p::Contract,
}

impl p::Recipe for GlmfRecipe {
    fn prepare(&self, request: &p::Request) -> Result<p::Contract, p::Refusal> {
        let mut contract = self.contract.clone();
        contract.resident_weight_limit = request.resident_weight_limit;
        Ok(contract)
    }
}

pub struct PreparedGlmf<'a> {
    pub pool: PoolPolicy,
    started: std::time::Instant,
    input: &'a GlmfInputs<'a>,
    pub recipe: GlmfRecipe,
    pub budgets: Vec<p::Budget>,
    pub request: p::Request,
    counts: Vec<usize>,
    arena: Vec<Vec<u64>>,
    graphs: GlmfGraphs,
    native_load: Option<NativeLoad>,
    mount_load: Option<crate::shared::dense_read::MountLoad>,
    encoders: Vec<crate::plan::encoder::EncoderPlacement>,
}

pub struct AdmittedGlmf {
    pub placement: Placement,
    pub graphs: GlmfGraphs,
    pub working: GlmfWorkingSet,
    pub solved: p::ResolvedPlan,
    pub native_load: Option<NativeLoad>,
    pub mount_load: Option<crate::shared::dense_read::MountLoad>,
    pub encoder: Option<crate::plan::encoder::EncoderPlacement>,
}

fn invalid(error: impl std::fmt::Display) -> p::Refusal {
    p::Refusal::InvalidFact(error.to_string())
}

/// The exact-row working-set reserve admitted once per GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmfGraphs {
    pub policies: Vec<crate::placement::graph_reserve::Resolved>,
    pub executables: Vec<u64>,
    pub bootstraps: Vec<u64>,
}

impl GlmfGraphs {
    pub fn charges(&self) -> Vec<u64> {
        self.policies.iter().map(|policy| policy.bytes).collect()
    }
}

fn graph_reserve(i: &GlmfInputs<'_>, split_draft: bool) -> Result<GlmfGraphs, p::Refusal> {
    let gpus = i.gpus.len();
    let policies: Vec<_> = (0..gpus).map(|gpu| {
        crate::placement::graph_reserve::current_working_set(gpu, glmf_graphs::working_set(gpus, split_draft, gpu))
            .map_err(invalid)
    }).collect::<Result<_, _>>()?;
    let bootstraps: Vec<_> = (0..gpus).map(|gpu| glmf_graphs::bootstrap_bytes(gpus, i.spark_ranks > 0, gpu)).collect();
    let executables = policies.iter().zip(&bootstraps).map(|(policy, &boot)| policy.base_bytes.div_ceil(boot)).collect();
    Ok(GlmfGraphs { policies, executables, bootstraps })
}

/// One fixed owner per GPU with a nonzero reserve. `GPU_GRAPH_RESERVE=0` charges
/// nothing and registers no owner (as GLM 5.3 does; the contract drops zero-byte
/// allowances anyway): decode then runs eagerly, the graph cache issuing no
/// capture ticket against a zero reserve.
fn graph_components(graphs: &GlmfGraphs, maximum: u64) -> Result<Vec<p::Component>, p::Refusal> {
    graphs.policies.iter().enumerate().filter(|(_, resolved)| resolved.bytes != 0).map(|(gpu, resolved)| {
        let id = format!("glmf.graphs.{gpu}");
        let mut component = p::serving::startup_graph_component(&id, p::Location::coordinator(gpu as u8),
            crate::placement::GraphRank::estimated(graphs.executables[gpu], resolved.bytes), maximum)?;
        if let p::ComponentKind::Allowance(owner) = &mut component.kind {
            owner.source = crate::placement::graph_reserve::source(&id);
            // The configured budget, not a census estimate awaiting capture.
            owner.assumed = false;
        }
        Ok(component)
    }).collect()
}

/// What the exact 12-bit head (`--fp8-head packed`) owns beside its resident bytes, which
/// `i.resident` already holds.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PackedHeadOwners {
    /// The landed BF16 head it packs from on GPU 0, released once the pack verifies.
    pub source: Option<p::Allocation>,
    /// The drafter's coarse-plane GEMV scratch, from the drafter's load on.
    pub draft_scratch: u64,
}

/// [`PackedHeadOwners`] of `i` on an `sms`-SM coordinator (none for the other head forms).
/// Target logits run the GEMV in each lead step workspace's head workspace
/// ([`cache::GLMF_HEAD_WORKSPACE`], which only the BF16 head's cuBLAS route uses), so the
/// GEMV's scratch must fit it; the packed head serves one coordinator GPU.
pub(crate) fn packed_head_owners(i: &GlmfInputs<'_>, sms: usize) -> Result<PackedHeadOwners, p::Refusal> {
    use crate::families::glm5_flash::resident::GlmfHeadFormat;
    let head = i.representation.head;
    if head != GlmfHeadFormat::Packed { return Ok(PackedHeadOwners::default()); }
    if i.gpus.len() != 1 {
        return Err(invalid("the exact 12-bit head (--fp8-head packed) serves one coordinator GPU; \
            a head split keeps the BF16 or FP8 head"));
    }
    let (vocab, hidden, sms) = (i.cfg.vocab_size as u64, i.cfg.hidden as u64, sms as u64);
    let target = head.scratch_bytes(i.prefill_rows.max(i.decode_rows), vocab, hidden, sms);
    if target > cache::GLMF_HEAD_WORKSPACE {
        return Err(invalid(format!("the exact 12-bit head's GEMV needs {target} bytes of scratch on {sms} SMs, \
            more than the step workspaces' {}-byte head workspace", cache::GLMF_HEAD_WORKSPACE)));
    }
    let source = p::Allocation::buffer("glmf.load.head.bf16", p::Location::coordinator(0),
        head.load_source_bytes(vocab, hidden), p::Lifetime { first: p::Phase::Load, last: p::Phase::Load },
        Category::Staging, "BF16 head packed at load");
    // The drafter's rows run in GEMV passes: one pass's scratch serves every row count.
    let draft_scratch = if i.drafter_bytes > 0 {
        head.scratch_bytes(cuteafd_core::packed_head::PASS_ROWS, vocab, hidden, sms)
    } else { 0 };
    Ok(PackedHeadOwners { source: Some(source), draft_scratch })
}

fn component(id: impl Into<String>, gpu: u8, category: Category, group: &str,
    bytes: p::ByteCurve, lifetime: p::Lifetime, maximum: u64) -> Result<p::Component, p::Refusal> {
    let allocation = p::Allocation::buffer(id, p::Location::coordinator(gpu), bytes.at(1)?,
        lifetime, category, group);
    let mut owner = p::Component::allocation(allocation, bytes.clone(), false, maximum);
    owner.byte_extent(bytes)?;
    Ok(owner)
}

fn add_demand(contract: &mut p::Contract, demand: &Demand, requests: &[u64], id: String,
    activation: p::Activation) -> Result<(), p::Refusal> {
    add_demand_allocated_at(contract, demand, requests, id, activation, None)
}

/// `allocated_at` names the phases the engine really allocates the owner in, when the
/// owner's category alone would name a later one (the DFlash linear/attention scratch is
/// a workspace allocation made while the drafter loads).
fn add_demand_allocated_at(contract: &mut p::Contract, demand: &Demand, requests: &[u64], id: String,
    activation: p::Activation, allocated_at: Option<p::Lifetime>) -> Result<(), p::Refusal> {
    let lifetime = if let Some(lifetime) = allocated_at { lifetime }
    else if demand.lifetime == cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve {
        p::Lifetime { first: p::Phase::Growth, last: p::Phase::Growth }
    } else if matches!(demand.category, Category::Weights | Category::Embedding | Category::Runtime) {
        p::Lifetime::RESIDENT
    } else if demand.category == Category::Drafter {
        // Drafter owners are allocated while the drafter loads, not with the target KV.
        p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::Growth }
    } else { p::Lifetime::TARGET };
    let maximum = contract.maximum_pages;
    // A supplied request list must sit inside the owner it describes: a view (the K6
    // receive suffix inside the peer exchange slot) or an unlisted small field can
    // make it smaller, never larger.
    if !requests.is_empty() {
        let total = requests.iter().try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
            .ok_or(p::Refusal::Overflow("GLM Flash owner request list"))?;
        if total > demand.bytes {
            return Err(p::Refusal::InvalidFact(format!(
                "GLM Flash owner {id}: its request list is {total} bytes, more than its {}-byte extent",
                demand.bytes)));
        }
    }
    let mut owner = component(id.clone(), demand.gpu, demand.category, &demand.group,
        p::ByteCurve::fixed(demand.bytes, maximum), lifetime, maximum)?;
    owner.activation = activation.clone();
    contract.components.push(owner);
    // One named SM120 backing per physical request of this owner. Every caller
    // that knows the engine's real request list passes it; an owner that is one
    // request of its whole extent needs none.
    if let Some(mut tail) = p::exact_owner_tail(format!("{id}.tail"), id.clone(),
        p::Location::coordinator(demand.gpu), lifetime, requests.to_vec(), maximum)? {
        tail.activation = activation.clone();
        contract.components.push(tail);
    }
    if demand.resident_weights.0 > 0 {
        let mut weight = p::Component::resident_weight(p::ResidentWeight {
            id, location: p::Location::coordinator(demand.gpu), weights: demand.resident_weights,
        }, p::ByteCurve::fixed(demand.resident_weights.0, maximum));
        weight.activation = activation;
        contract.components.push(weight);
    }
    Ok(())
}

fn add_arena(contract: &mut p::Contract, layout: &crate::serving_capacity::WeightArenaLayout,
    gpu: usize, lifetime: p::Lifetime, activation: &p::Activation, kind: cuteafd_core::memory_layout::MemoryKind) -> Result<(), p::Refusal> {
    for mut owner in p::weight_arena_components(layout, p::Location::coordinator(gpu as u8),
        lifetime, contract.maximum_pages, kind)? {
        owner.activation = activation.clone();
        contract.components.push(owner);
    }
    Ok(())
}

fn modes(i: &GlmfInputs<'_>) -> Vec<LayerMode> {
    let mut mla_owner = i.attention.first_owner;
    (0..i.layers).map(|layer| {
        if i.gpus.len() == 1 { return LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }; }
        if i.cfg.attention[layer] != GlmNextAttention::Mla { return LayerMode::HeadSplit; }
        match i.attention.mode {
            AttentionPlacement::Heads => LayerMode::HeadSplit,
            AttentionPlacement::Context => LayerMode::ContextSplit,
            AttentionPlacement::Layers => {
                let owner = mla_owner;
                mla_owner ^= 1;
                LayerMode::Whole { gpu: owner, ffn: FfnMode::Split }
            }
        }
    }).collect()
}

pub fn prepare<'a>(i: &'a GlmfInputs<'a>, sms: usize, rtx_experts: bool, mut process: ProcessOwners)
    -> Result<PreparedGlmf<'a>, p::Refusal> {
    let started = std::time::Instant::now();
    let alternate = process.alternate_resident.take();
    let alternate_arenas = std::mem::take(&mut process.alternate_target_arenas);
    let mut prepared = prepare_fixed(i, sms, rtx_experts, process.clone())?;
    if let Some(resident) = alternate {
        if i.gpus.len() != 2 || i.attention.mode != AttentionPlacement::Layers || resident.len() != 2 {
            return Err(invalid("Flash alternate attention owner inventory"));
        }
        let mut input = i.clone();
        input.attention.first_owner ^= 1;
        input.resident = resident;
        if !process.target_arenas.is_empty() && alternate_arenas.is_empty() {
            return Err(invalid("Flash alternate attention arena inventory"));
        }
        process.target_arenas = alternate_arenas;
        let alternate = prepare_fixed(&input, sms, rtx_experts, process)?;
        let first_mla = i.cfg.attention[..i.layers].iter().position(|kind| *kind == GlmNextAttention::Mla)
            .ok_or_else(|| invalid("Flash alternating ownership lacks MLA layer"))?;
        let selector = first_mla;
        for (left, right) in prepared.recipe.contract.variables.iter_mut().zip(&alternate.recipe.contract.variables) {
            if let (p::Variable::Attention { values, .. }, p::Variable::Attention { values: other, .. }) = (left, right) {
                if *values != *other { values.push(other[0]); }
            }
        }
        for layer in 0..i.layers {
            if layer == selector || i.cfg.attention[layer] != GlmNextAttention::Mla { continue; }
            for option in 0..2 {
                prepared.recipe.contract.requires.push(p::Requires { when: p::Choice { variable: selector, option },
                    any_of: vec![p::Choice { variable: layer, option }] });
            }
        }
        let mut components = Vec::new();
        for (option, source) in [prepared.recipe.contract.components.clone(), alternate.recipe.contract.components]
            .into_iter().enumerate() {
            for mut component in source {
                for clause in &mut component.activation.clauses { clause.push(p::Choice { variable: selector, option }); }
                components.push(component);
            }
        }
        prepared.recipe.contract.components = components;
    }
    prepared.started = started;
    Ok(prepared)
}

fn prepare_fixed<'a>(i: &'a GlmfInputs<'a>, sms: usize, rtx_experts: bool, process: ProcessOwners)
    -> Result<PreparedGlmf<'a>, p::Refusal> {
    let facts = facts(i).map_err(invalid)?;
    let gpus = i.gpus.len();
    let physical_sparks = i.spark_ranks.max(process.physical_sparks);
    if physical_sparks > 8 { return Err(invalid("Flash encoder Spark rank outside transport inventory")); }
    if i.max_context == 0 || i.spark_ranks > 8
        || i.gpus.iter().any(|(total, baseline)| *total == 0
            || matches!(baseline, Baseline::Measured { free_bytes } if free_bytes > total))
        || i.drafter_bridge.as_ref().is_some_and(|bridge| gpus != 2 || i.drafter_bytes == 0
            || bridge.placement_gpu != 1 || bridge.demands.iter().any(|d| usize::from(d.gpu) >= gpus))
        || gpus == 2 && i.decode_rows != cache::GLMF_DECODE_ROWS
        || i.attention.mode == AttentionPlacement::Context && i.pool_marks
        || i.drafter_bytes == 0 && i.drafter_scratch.iter().any(|&bytes| bytes > 0)
 {
        return Err(invalid("GLM Flash executor geometry"));
    }
    if i.shared_replay && (gpus != 1 || i.spark_ranks == 0 || !process.remote_experts) {
        return Err(invalid("--replay-records shared needs one coordinator GPU with Spark experts"));
    }
    let head = packed_head_owners(i, sms)?;
    for profile in [&i.workspace, &i.local_workspace] {
        if profile.iter().any(|row| usize::from(row.demand.gpu) >= gpus)
            || (0..gpus).any(|rank| !profile.iter().any(|row|
                usize::from(row.demand.gpu) == rank && row.component == GlmfComponent::Always)) {
            return Err(invalid("GLM Flash step inventory"));
        }
    }
    let capacities = i.gpus.iter().map(|gpu| gpu.0).collect::<Vec<_>>();
    let mut pool_facts = crate::placement::PoolFacts::new(i.context, i.requested_pool, 256);
    pool_facts.memory = facts.geometry.ranks.iter().zip(&capacities).map(|(rank, &available_bytes)|
        crate::placement::PoolMemory { available_bytes, bytes_per_unit: rank.persistent_unit_bytes }).collect();
    pool_facts.target = i.target_pool_tokens;
    let pool = crate::placement::prepare_pool(pool_facts)
        .map_err(|error| p::Refusal::InvalidFact(error.to_string()))?;
    let maximum = pool.maximum_tokens().div_ceil(256);
    let attention = modes(i);
    let routed: Vec<_> = (0..i.layers).filter(|&layer| !i.cfg.dense[layer]).collect();
    let rtx_experts = rtx_experts && (gpus == 1 || i.experts.iter().all(|cost| cost.tp2));
    for &scratch in &i.drafter_scratch {
        i.drafter_bytes.checked_add(scratch).ok_or(p::Refusal::Overflow("Flash draft inventory"))?;
    }
    let counts = if i.spark_ranks == 0 { vec![routed.len()] }
        else if let Some(n) = i.onboard.layers(routed.len()) { vec![n] }
        else if rtx_experts { (0..=routed.len()).collect() } else { vec![0] };
    let counts = counts.into_iter().filter(|&count| process.remote_experts || count == routed.len()).collect::<Vec<_>>();
    if i.shared_replay && counts.iter().all(|&count| count == routed.len()) {
        return Err(invalid("--replay-records shared needs one coordinator GPU with Spark experts"));
    }
    let counts = counts.into_iter().filter(|&count| !i.shared_replay || count < routed.len()).collect::<Vec<_>>();
    if counts.is_empty() || counts.iter().any(|&n| n > routed.len()) || !rtx_experts && counts.iter().any(|&n| n > 0) {
        return Err(p::Refusal::Unsupported);
    }
    let mut contract = p::Contract::fixed(256, maximum, gpus, p::Facts { allocation_granularity: vec![], driver_page_tables: vec![],
        phase_order: p::PhaseOrder::TARGET_THEN_DRAFT, allocations: vec![], deferred: vec![],
        sequential: vec![], reservations: vec![], allowances: vec![], resident_weights: vec![],
    });
    // Target lanes and local BF16 draft projections borrow the same serving
    // thread/device pool. Native vocabulary callbacks create neither backend.
    let mut blas = Vec::new();
    if i.blas_head.bf16() {
        blas.push(p::serving::BlasOwners {
            name: "GLM Flash target projections".into(), process: "coordinator".into(),
            thread: "glmf/blas/serve".into(), location: p::Location::coordinator(0),
            gemm: i.blas_head.gemm(), lt: i.blas_head.lt(),
            lifetime: p::Lifetime { first: if i.shared_replay { p::Phase::PostTargetKv } else { p::Phase::Warmup },
                last: p::Phase::Growth }, activation: p::Activation::always(),
        });
    }
    contract.pool_page_multiple = if i.attention.mode == AttentionPlacement::Context { 2 } else { 1 };
    contract.spark_reply_formats = (i.spark_ranks > 0).then_some(cuteafd_core::spark_reply::SparkReplyFormats {
        prefill: Some(i.spark_reply_format), decode: Some(cuteafd_core::SparkReplyFormat::Bf16),
    });
    contract.variables.extend(attention.iter().enumerate().map(|(layer, &mode)|
        p::Variable::Attention { layer, values: vec![mode] }));
    let prefill = p::PrefillShape { rows: i.prefill_rows, waves: i.prefill_lanes };
    contract.variables.push(p::Variable::Prefill(vec![prefill]));
    contract.variables.push(p::Variable::Lanes(vec![1]));
    let embedding = i.resident.iter().map(|rank| rank.embedding).sum::<u64>();
    let embedding_variable = contract.variables.len();
    contract.variables.push(p::Variable::Embedding(if embedding == 0 { vec![p::EmbeddingHome::HostPinned] }
        else { (0..gpus).map(|gpu| p::EmbeddingHome::Device(gpu as u8)).collect() }));
    if !routed.is_empty() && i.spark_ranks > 0 {
        contract.variables.push(p::Variable::ExpertHome { layers: routed.clone(),
            rank_sets: vec![(0..i.spark_ranks as u8).collect()] });
    }
    let onboard_variable = contract.variables.len();
    if !routed.is_empty() {
        contract.variables.push(p::Variable::Onboard { layers: routed.clone(),
            destination: (0..gpus as u8).collect(), counts: counts.clone() });
    }
    if !process.encoders.is_empty() {
        use crate::plan::encoder::EncoderKind;
        let variable = contract.variables.len();
        contract.variables.push(p::Variable::Encoder(process.encoders.iter().map(crate::plan::encoder::EncoderPlacement::vision_on).collect()));
        for (option, encoder) in process.encoders.iter().enumerate() {
            let locations = match &encoder.kind {
                EncoderKind::Off => vec![],
                EncoderKind::Rtx { gpu } if *gpu < gpus => vec![p::Location::coordinator(*gpu as u8)],
                EncoderKind::Spark { .. } => {
                    if encoder.spark_ranks().iter().any(|&rank| rank >= physical_sparks) {
                        return Err(invalid("Flash encoder Spark rank absent"));
                    }
                    encoder.spark_ranks().into_iter().map(|rank| p::Location {
                        device: p::Device::Spark(rank as u8), domain: p::Domain::Managed }).collect()
                }
                _ => return Err(invalid("Flash encoder owner absent from executable topology")),
            };
            let activation = p::Activation::choice(variable, option);
            for (copy, location) in locations.iter().copied().enumerate() {
                let process = match location.device {
                    p::Device::Coordinator(_) => "coordinator".into(),
                    p::Device::Spark(rank) => format!("glmf-media-worker-{rank}"),
                    _ => unreachable!("encoder locations are CUDA devices"),
                };
                blas.push(p::serving::BlasOwners { name: format!("GLM Flash vision {option}.{copy}"), process,
                    thread: "glmf/blas/media".into(), location, gemm: true, lt: false, lifetime: p::Lifetime::RESIDENT,
                    activation: activation.clone() });
                for (part, bytes, category) in [("weights", encoder.weights, Category::Weights),
                    ("scratch", encoder.scratch, Category::Workspace)] {
                    let owner = p::Allocation::buffer(format!("glmf.encoder.{option}.{copy}.{part}"),
                        location, bytes, p::Lifetime::RESIDENT, category, format!("vision {part}"));
                    let mut component = p::Component::allocation(owner.clone(), p::ByteCurve::fixed(bytes, maximum), false, maximum);
                    component.activation = activation.clone();
                    contract.components.push(component);
                    if part == "weights" {
                        let mut component = p::Component::resident_weight(p::ResidentWeight::from_allocation(
                            &owner, ResidentWeights(bytes)), p::ByteCurve::fixed(bytes, maximum));
                        component.activation = activation.clone();
                        contract.components.push(component);
                    }
                }
            }
            if !locations.is_empty() && process.media_cache_bytes > 0 {
                let owner = p::Allocation::buffer("glmf.media.cache", p::Location::HOST,
                    process.media_cache_bytes, p::serving::ServingOwners::LIFETIME,
                    Category::Prefix, "host media embedding cache");
                let mut component = p::Component::allocation(owner,
                    p::ByteCurve::fixed(process.media_cache_bytes, maximum), false, maximum);
                component.activation = activation;
                contract.components.push(component);
            }
        }
    }
    let mut budgets = Vec::new();
    for (gpu, &(total, baseline)) in i.gpus.iter().enumerate() {
        let occupied = match baseline {
            Baseline::Planned { context_bytes, loaded_bytes } => context_bytes.checked_add(loaded_bytes)
                .ok_or(p::Refusal::Overflow("GLM Flash runtime baseline"))?,
            Baseline::Measured { free_bytes } => total.saturating_sub(free_bytes),
        };
        budgets.push(p::Budget { id: format!("rtx{gpu}"), locations: vec![p::Location::coordinator(gpu as u8)],
            physical_bytes: total, logical_bytes: total, headroom: i.headroom_bytes,
            baseline: p::Baseline::Forecast { occupied: 0,
                source: "GLM Flash physical capacity; runtime owner charged in recipe".into(), assumed: false } });
        if occupied > 0 {
            if matches!(baseline, Baseline::Measured { .. }) {
                contract.components.push(p::Component { activation: p::Activation::always(),
                    bytes: p::ByteCurve::fixed(occupied, maximum),
                    kind: p::ComponentKind::Allowance(p::Allowance {
                        location: p::Location::coordinator(gpu as u8), bytes: occupied,
                        lifetime: p::Lifetime::RESIDENT, assumed: false,
                        source: crate::placement::sampled_baseline::source("GLM Flash context and selected modules"),
                    }) });
            } else {
                add_demand(&mut contract, &Demand::new(gpu as u8, Category::Runtime, "context+modules", occupied,
                    Basis::Calibrated), &[], format!("glmf.runtime.{gpu}"), p::Activation::always())?;
            }
        }
    }
    // HOST and PINNED share one physical capacity; external occupancy is sampled
    // once before the recipe's process-owned bytes are admitted.
    let host = process.host;
    if host.0 == 0 || host.1 > host.0 { return Err(invalid("invalid Flash host capacity")); }
    budgets.push(p::Budget { id: "coordinator host RAM".into(), locations: vec![p::Location::HOST, p::Location::PINNED],
        physical_bytes: host.0, logical_bytes: host.0, headroom: process.host_headroom,
        baseline: p::Baseline::Sampled { free: host.1, phase: p::Phase::Load,
            source: "MemAvailable plus process anonymous RSS".into() } });
    for rank in 0..physical_sparks {
        budgets.push(p::Budget { id: format!("spark{rank}"), locations: vec![p::Location {
            device: p::Device::Spark(rank as u8), domain: p::Domain::Managed }],
            physical_bytes: process.spark_rank_caps.get(rank).copied().filter(|bytes| *bytes > 0)
                .ok_or_else(|| invalid(&format!("missing measured capacity for physical Spark rank {rank}")))?,
            logical_bytes: process.spark_rank_caps.get(rank).copied().filter(|bytes| *bytes > 0)
                .ok_or_else(|| invalid(&format!("missing measured capacity for physical Spark rank {rank}")))?, headroom: 0,
            baseline: p::Baseline::Forecast { occupied: 0,
                source: "Spark worker physical capacity; worker owners supplied by caller".into(), assumed: true } });
    }
    for owner in process.allocations.iter().cloned().chain(process.native_load.iter()
        .flat_map(|load| load.allocations(process.push_required)))
        .chain(process.mount_load.iter().flat_map(|load| load.allocations("glmf.load.mount")))
        .chain(head.source.clone()) {
        let bytes = p::ByteCurve::fixed(owner.bytes, maximum);
        contract.components.push(p::Component::allocation(owner, bytes, false, maximum));
    }
    if !process.draft_tp2.is_empty() {
        if gpus != 2 || process.draft_tp2.len() != 2 || i.drafter_bytes != 0 || i.drafter_bridge.is_some() {
            return Err(invalid("TP2 DFlash requires two fixed rank owners, not a movable"));
        }
        if process.draft_arenas.len() != gpus { return Err(invalid("TP2 draft arena inventory")); }
        for (rank, layout) in process.draft_tp2.iter().enumerate() {
            let prefix = if rank == 0 { "glmf.draft" } else { "glmf.draft.peer" };
            let location = p::Location::coordinator(rank as u8);
            if i.blas_head.bf16() && layout.owners().iter().any(|owner| owner.id == "workspace.head") {
                blas.push(p::serving::BlasOwners { name: "GLM Flash TP2 DFlash head".into(), process: "coordinator".into(),
                    thread: "glmf/blas/serve".into(), location, gemm: true, lt: false,
                    lifetime: p::Lifetime { first: p::Phase::Warmup, last: p::Phase::Growth },
                    activation: p::Activation::always() });
            }
            let immutable: std::collections::BTreeSet<_> = layout.weight_owners()
                .map(|owner| format!("{prefix}.{}", owner.id)).collect();
            for owner in layout.allocations(prefix, location).into_iter()
                .filter(|owner| !immutable.contains(&owner.id)) {
                // Each retained device workspace is a standalone CUDA allocation.
                if owner.location == location {
                    if let Some(tail) = p::exact_owner_tail(format!("{}.tail", owner.id), owner.id.clone(),
                        location, owner.lifetime, vec![owner.bytes], maximum)? {
                        contract.components.push(tail);
                    }
                }
                let bytes = p::ByteCurve::fixed(owner.bytes, maximum);
                contract.components.push(p::Component::allocation(owner, bytes, false, maximum));
            }
            add_arena(&mut contract, &process.draft_arenas[rank], rank,
                p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::Growth }, &p::Activation::always(), cuteafd_core::memory_layout::MemoryKind::DrafterWeights)?;
        }
        if let Some(draft) = &process.draft_push {
            for owner in draft.allocations(0)? {
                let bytes = p::ByteCurve::fixed(owner.bytes, maximum);
                contract.components.push(p::Component::allocation(owner, bytes, false, maximum));
            }
        }
    }
    for (rank, &bytes) in process.retained_expert_metadata.iter().enumerate().filter(|(_, bytes)| **bytes > 0) {
        let owner = p::Allocation::buffer(format!("glmf.expert-metadata.{rank}"), p::Location::HOST,
            bytes, p::Lifetime::RESIDENT, Category::Staging, "coordinator EXLR8 expert metadata");
        contract.components.push(p::Component::allocation(owner, p::ByteCurve::fixed(bytes, maximum), false, maximum));
    }
    let spark_activation = p::Activation { clauses: counts.iter().enumerate()
        .filter(|(_, n)| **n < routed.len()).map(|(option, _)|
            vec![p::Choice { variable: onboard_variable, option }]).collect() };
    for link in &process.links {
        for mut owner in link.components(maximum)? {
            owner.activation = spark_activation.clone();
            contract.components.push(owner);
        }
    }
    let target_arenas = if process.target_arenas.is_empty() {
        // Byte-only unit fixtures model final arena requests, never a second
        // production payload route. Real callers must supply header recipes.
        #[cfg(test)]
        {
            let homes = if embedding == 0 { 1 } else { gpus };
            (0..homes).map(|home| (0..gpus).map(|rank| {
                let mut requests = vec![("weights".into(), i.resident[rank].weights as usize)];
                if embedding > 0 && home == rank { requests.push(("embedding.table".into(), embedding as usize)); }
                crate::serving_capacity::WeightArenaLayout::new(format!("glm5_flash/target/{rank}"), requests).map_err(invalid)
            }).collect::<Result<Vec<_>, _>>()).collect::<Result<Vec<_>, _>>()?
        }
        #[cfg(not(test))]
        { return Err(invalid("Flash target arena recipe missing")); }
    } else { process.target_arenas.clone() };
    let homes = if embedding == 0 { 1 } else { gpus };
    if target_arenas.len() != homes || target_arenas.iter().any(|ranks| ranks.len() != gpus) {
        return Err(invalid("Flash target arena candidate inventory"));
    }
    for (home, layouts) in target_arenas.iter().enumerate() {
        let activation = p::Activation::choice(embedding_variable, home);
        for (rank, layout) in layouts.iter().enumerate() {
            add_arena(&mut contract, layout, rank, p::Lifetime::RESIDENT, &activation, cuteafd_core::memory_layout::MemoryKind::Weights)?;
            if layout.views().keys().any(|name| name.ends_with(".trellis")) {
                let mut owner = crate::shared::dense_read::trellis_device_staging(rank as u8, maximum);
                owner.activation = activation.clone();
                contract.components.push(owner);
            }
            for (name, view) in layout.views().iter().filter(|(name, _)|
                name.as_str() == "embedding.table" && embedding > 0 && process.shared_arenas.is_empty()
                    || name.starts_with("embedding.") && name.as_str() != "embedding.table") {
                let mut alias = p::Allocation::buffer(format!("glmf.embedding.{rank}.{name}"),
                    p::Location::coordinator(rank as u8), view.bytes() as u64,
                    p::Lifetime::RESIDENT, Category::Embedding, "embedding");
                alias.view = Some(p::View { backing: layout.id().into(), offset: view.offset() as u64 });
                let mut owner = p::Component::allocation(alias, p::ByteCurve::fixed(view.bytes() as u64, maximum), false, maximum);
                owner.activation = activation.clone();
                contract.components.push(owner);
            }
        }
    }
    for (rank, layout) in process.shared_arenas.iter().enumerate() {
        add_arena(&mut contract, layout, rank, p::Lifetime::RESIDENT, &p::Activation::always(),
            cuteafd_core::memory_layout::MemoryKind::Weights)?;
    }
    for (index, row) in facts.fixed.iter().enumerate() {
        if row.demand.group == "pending code" {
            if row.demand.bytes > 0 {
                contract.components.push(p::Component { activation: p::Activation::always(),
                    bytes: p::ByteCurve::fixed(row.demand.bytes, maximum),
                    kind: p::ComponentKind::Allowance(p::Allowance {
                        location: p::Location::coordinator(row.demand.gpu), bytes: row.demand.bytes,
                        lifetime: p::Lifetime::RESIDENT, assumed: true,
                        source: "selected-program loaded code estimate (unqualified module residency; #1463)".into(),
                    }) });
            }
            continue;
        }
        if matches!(row.demand.group.as_str(), "embedding" | "target weights" | "graphs" | "graph growth")
            || row.demand.group == "Spark intake" && !process.links.is_empty()
            || row.demand.group == "wide decode selector" { continue; }
        let activation = match row.component {
            GlmfComponent::Always => p::Activation::always(),
            GlmfComponent::RtxExperts | GlmfComponent::SparkExperts => p::Activation { clauses: counts.iter()
                .enumerate().filter(|(_, n)| if row.component == GlmfComponent::RtxExperts { **n > 0 }
                    else { **n < routed.len() }).map(|(option, _)| vec![p::Choice { variable: onboard_variable, option }]).collect() },
        };
        add_demand(&mut contract, &row.demand, &row.requests, format!("glmf.fixed.{index}"), activation)?;
    }
    for gpu in 0..gpus {
        let geometry = &facts.geometry.ranks[gpu];
        // One device request per pool piece and resident MLA layer, each rounded to its own
        // driver granule: charge those backings instead of the flat payload.
        let layers = facts.mla_layers[gpu];
        let mut unit: u64 = 0;
        for (name, per_layer) in cache::glm_flash_mla_unit_pieces(i.index, i.attention.mode) {
            let piece = per_layer.checked_mul(layers).ok_or(p::Refusal::Overflow("GLM Flash KV piece"))?;
            unit = unit.checked_add(piece).ok_or(p::Refusal::Overflow("GLM Flash KV unit"))?;
            if piece == 0 { continue; }
            contract.components.push(component(format!("glmf.pool.{gpu}.{name}"), gpu as u8,
                Category::Kv, "KV pool", p::ByteCurve::paged(piece, piece,
                    p::SM120_ALLOCATION_GRANULARITY_BYTES, layers, maximum), p::Lifetime::TARGET, maximum)?);
        }
        if unit != geometry.persistent_unit_bytes {
            return Err(p::Refusal::InvalidFact("GLM Flash KV unit pieces do not sum".into()));
        }
        // The shared unit-index map is one small allocation (4 B per unit) and the
        // KDA-only pool's mark backing reserves its whole state per unit.
        let metadata = geometry.pool_metadata_unit_bytes.checked_add(
            geometry.glmf_pool_mark_backing_unit_bytes(i.pool_marks))
            .ok_or(p::Refusal::Overflow("GLM Flash KV unit"))?;
        if metadata > 0 {
            contract.components.push(component(format!("glmf.pool.{gpu}.metadata"), gpu as u8,
                Category::Kv, "KV pool", p::ByteCurve::linear(metadata, metadata, maximum),
                p::Lifetime::TARGET, maximum)?);
        }
    }
    if i.attention.mode == AttentionPlacement::Context {
        for (index, demand) in context_buffers(i.cfg, i.max_context, i.prefill_rows, i.prefill_lanes,
            i.attention).demands().map_err(invalid)?.iter().enumerate() {
            add_demand(&mut contract, demand, &[], format!("glmf.context.{index}"), p::Activation::always())?;
        }
    }
    let hop = hop_spec(i);
    let hops = plan_hops(&attention, &hop).map_err(invalid)?;
    for (gpu, bytes) in hop_buffer_bytes(&hops, &hop, gpus).ok_or(p::Refusal::Overflow("GLM Flash hops"))?.into_iter().enumerate() {
        add_demand(&mut contract, &Demand::new(gpu as u8, Category::Transport, "residual hops", bytes,
            Basis::Formula), &[], format!("glmf.hops.{gpu}"), p::Activation::always())?;
    }
    if gpus == 2 && i.attention.mode == AttentionPlacement::Layers {
        let layout = crate::placement::HopLinkLayout::new(&hops, &hop)
            .ok_or(p::Refusal::Overflow("GLM Flash hop control"))?;
        contract.components.extend(layout.control_components("glmf.hops", p::Lifetime::RESIDENT, maximum));
    }
    if i.drafter_bytes > 0 {
        let nonweights = i.drafter_bytes.checked_sub(i.drafter_weights.0)
            .ok_or_else(|| invalid("draft weights exceed draft owned storage"))?;
        let draft_arenas = if process.draft_arenas.is_empty() {
            #[cfg(test)]
            { (0..gpus).map(|rank| crate::serving_capacity::WeightArenaLayout::new(
                format!("glm5_flash/draft/{rank}"), [("fixture.weights".into(), i.drafter_weights.0 as usize)])
                .map_err(invalid)).collect::<Result<Vec<_>, _>>()? }
            #[cfg(not(test))]
            { return Err(invalid("Flash draft arena recipe missing")); }
        } else { process.draft_arenas.clone() };
        if draft_arenas.len() != gpus { return Err(invalid("Flash draft arena candidate inventory")); }
        let homes = if i.drafter_bridge.is_some() { vec![0, 1] } else { vec![0] };
        let variable = contract.variables.len();
        contract.variables.push(p::Variable::DraftHome(homes.clone()));
        let (draft_variable, draft_homes) = (variable, homes.clone());
        for (option, gpu) in homes.into_iter().enumerate() {
            let activation = p::Activation::choice(variable, option);
            if gpu == 0 && i.blas_head.bf16() {
                blas.push(p::serving::BlasOwners { name: "GLM Flash drafter vocabulary head".into(), process: "coordinator".into(),
                    thread: "glmf/blas/serve".into(), location: p::Location::coordinator(gpu), gemm: true, lt: false,
                    lifetime: p::Lifetime { first: p::Phase::Warmup, last: p::Phase::Growth },
                    activation: activation.clone() });
            }
            add_arena(&mut contract, &draft_arenas[usize::from(gpu)], usize::from(gpu),
                p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::Growth }, &activation, cuteafd_core::memory_layout::MemoryKind::DrafterWeights)?;
            // One device request per drafter owner: the rings, taps, context update and
            // workspace buffers, then the linear and attention scratch allocations.
            let requests = facts.drafter_requests.as_ref();
            let none: &[u64] = &[];
            add_demand(&mut contract, &Demand::new(gpu, Category::Drafter, "DFlash", nonweights,
                Basis::Exact), requests.map_or(none, |rows| rows.resident(usize::from(gpu))),
                format!("glmf.draft.{gpu}"), activation.clone())?;
            add_demand_allocated_at(&mut contract, &Demand::new(gpu, Category::Workspace, "DFlash linear/attention scratch", i.drafter_scratch[usize::from(gpu)],
                Basis::Formula), requests.map_or(none, |rows| rows.scratch(usize::from(gpu))),
                format!("glmf.draft.scratch.{gpu}"), activation.clone(),
                Some(p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::Growth }))?;
            if head.draft_scratch > 0 {
                add_demand(&mut contract, &Demand::new(gpu, Category::Workspace, "packed head draft scratch",
                    head.draft_scratch, Basis::Formula), &[], format!("glmf.draft.head.{gpu}"), activation.clone())?;
            }
            if gpu != 0 {
                if i.blas_head.bf16() {
                    blas.push(p::serving::BlasOwners { name: "GLM Flash remote drafter bridge head".into(), process: "coordinator".into(),
                        thread: "glmf/blas/serve".into(), location: p::Location::coordinator(0), gemm: true, lt: false,
                        lifetime: p::Lifetime { first: p::Phase::Warmup, last: p::Phase::Growth },
                        activation: activation.clone() });
                }
                for (index, demand) in i.drafter_bridge.as_ref().into_iter().flat_map(|bridge| &bridge.demands).enumerate() {
                    add_demand(&mut contract, demand, &[], format!("glmf.draft.bridge.{index}"), activation.clone())?;
                }
            }
        }
        for owner in &mut contract.components {
            if let p::ComponentKind::Allocation { template, .. } = &mut owner.kind {
                if template.id.starts_with("glmf.draft.") {
                    template.lifetime = p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::Growth };
                }
            }
        }
        // After the resident lifetime above: these owners die when the drafter group commits.
        if let Some(draft) = &process.draft_push {
            for (option, gpu) in draft_homes.into_iter().enumerate() {
                for owner in draft.allocations(gpu)? {
                    let bytes = p::ByteCurve::fixed(owner.bytes, maximum);
                    let mut owner = p::Component::allocation(owner, bytes, false, maximum);
                    owner.activation = p::Activation::choice(draft_variable, option);
                    contract.components.push(owner);
                }
            }
        }
    }
    let mut arena = Vec::new();
    for (option, &count) in counts.iter().enumerate() {
        let activation = if routed.is_empty() { p::Activation::always() }
            else { p::Activation::choice(onboard_variable, option) };
        if let Some(ranks) = process.spark_candidates.get(count) {
            for (rank, &bytes) in ranks.iter().enumerate().filter(|(_, bytes)| **bytes > 0) {
                let owner = p::Allocation::buffer(format!("exlr8.spark{rank}.owners.count{count}"),
                    p::Location { device: p::Device::Spark(rank as u8), domain: p::Domain::Managed },
                    bytes, p::Lifetime::RESIDENT, Category::Workspace, "EXLR8 Spark weights + execution owners");
                let mut owner = p::Component::allocation(owner, p::ByteCurve::fixed(bytes, maximum), false, maximum);
                owner.activation = activation.clone();
                contract.components.push(owner);
            }
        }
        let working = GlmfWorkingSet::candidate(i, count).map_err(invalid)?;
        let workspace = if working.spark_ranks == 0 { &i.local_workspace } else { &i.workspace };
        for (index, row) in working.demands(workspace).enumerate() {
            add_demand(&mut contract, &row.demand, &row.requests, format!("glmf.steps.{option}.{index}"), activation.clone())?;
        }
        let mut peak = vec![0; gpus];
        for gpu in 0..gpus {
            let mut resident = if count == 0 { 0 } else if gpus == 2 { i.tp2_workspace[gpu] } else { i.expert_workspace };
            let mut weights = 0u64;
            let mut staging = 0;
            peak[gpu] = resident;
            for cost in i.experts.iter().take(count) {
                let part = if gpus == 2 { cost.half[gpu] } else { cost.whole };
                resident = resident.checked_add(part.resident).ok_or(p::Refusal::Overflow("GLM Flash expert arena"))?;
                weights = weights.checked_add(part.resident).ok_or(p::Refusal::Overflow("GLM Flash expert weights"))?;
                staging = staging.max(part.staging);
                peak[gpu] = peak[gpu].max(resident.checked_add(if gpus == 2 { staging } else { part.staging })
                    .ok_or(p::Refusal::Overflow("GLM Flash expert load peak"))?);
            }
            add_demand(&mut contract, &Demand::new(gpu as u8, Category::Experts, "RTX expert arena", peak[gpu], Basis::Exact)
                .with_resident_weights(ResidentWeights(weights)), &[], format!("glmf.experts.{option}.{gpu}"), activation.clone())?;
        }
        arena.push(peak);
    }
    // The graph reserve is one fixed per-device owner, charged once here (before
    // the pool is solved), and independent of the onboarding choice.
    let graphs = graph_reserve(i, !process.draft_tp2.is_empty())?;
    contract.components.extend(graph_components(&graphs, maximum)?);
    let (rtx_blas, spark_blas): (Vec<_>, Vec<_>) = blas.into_iter()
        .partition(|owner| matches!(owner.location.device, p::Device::Coordinator(_)));
    contract.components.extend(p::serving::blas_components("sm_120", &rtx_blas, maximum, contract.phase_order)?);
    contract.components.extend(p::serving::blas_components("sm_121", &spark_blas, maximum, contract.phase_order)?);
    contract.add_driver_page_tables();
    // Driver stream slots are slabs, not cudaMalloc requests: no allocation-tail
    // charge, and the slab survives the streams that first filled it.
    let peak = startup_peak_live_streams(gpus);
    let slabs = p::cuda_stream_slab_bytes(peak).map_err(invalid)?;
    for rank in 0..gpus {
        contract.components.push(p::Component { activation: p::Activation::always(),
            bytes: p::ByteCurve::fixed(slabs, maximum),
            kind: p::ComponentKind::Allowance(p::Allowance {
                location: p::Location::coordinator(rank as u8), bytes: slabs,
                lifetime: p::Lifetime::RESIDENT, assumed: false,
                source: format!("glmf/driver-stream-slabs/gpu{rank} ({peak} peak live streams, \
                    SM120 driver/runtime13020 LAZY, #1511 2026-11-08)") }) });
    }
    let request = p::Request { context: i.context,
        pool: PoolPolicy::normalize_request(i.requested_pool).map_or(p::Pin::Auto, p::Pin::Exact),
        onboard: i.onboard.layers(routed.len()).map_or(p::Pin::Auto, |n| p::Pin::Exact(n as u64)),
        prefill: p::Pin::Exact(prefill), operation: p::Operation::Serving,
        resident_weight_limit: i.resident_weight_limit,
        spark_reply_formats: contract.spark_reply_formats.unwrap_or_default() };
    Ok(PreparedGlmf { pool, started: std::time::Instant::now(), input: i, recipe: GlmfRecipe { contract }, budgets, request, counts, arena, graphs,
        native_load: process.native_load, mount_load: process.mount_load, encoders: process.encoders })
}

fn hop_spec(i: &GlmfInputs<'_>) -> HopSpec {
    HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: 4 * i.cfg.hidden as u64 * 2,
        rows: i.prefill_rows.max(i.decode_rows), lanes: i.prefill_lanes, entry_gpu: 0, head_gpu: 0,
        terminal: if i.gpus.len() == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } }
}

impl PreparedGlmf<'_> {
    pub fn admit(self) -> Result<AdmittedGlmf, p::Refusal> {
        let solved = p::solve_prepared(self.request, self.budgets, &self.recipe, &self.pool)?;
        let i = self.input;
        let count = solved.decisions.onboard_layers.iter().flatten().copied()
            .collect::<std::collections::BTreeSet<_>>().len();
        let option = self.counts.iter().position(|&n| n == count).ok_or_else(|| invalid("unprepared onboarding count"))?;
        let working = GlmfWorkingSet::candidate(i, count).map_err(invalid)?;
        // The working-set reserve does not depend on the solved KV pool.
        let graphs = self.graphs;
        for (gpu, policy) in graphs.policies.iter().enumerate() {
            let source = crate::placement::graph_reserve::source(&format!("glmf.graphs.{gpu}"));
            let owner = solved.facts.allowances.iter().find(|owner| owner.source == source);
            match (policy.bytes, owner) {
                (0, None) => {}
                (0, Some(_)) => return Err(invalid("a zero graph reserve has an admitted owner")),
                (_, None) => return Err(invalid("missing admitted graph reserve owner")),
                (bytes, Some(owner)) if bytes != owner.bytes =>
                    return Err(invalid("solved graph reserve differs from the charge")),
                _ => {}
            }
        }
        let modes = &solved.decisions.attention;
        let layers = modes.iter().enumerate().map(|(layer, &mode)| LayerAssignment { mode,
            experts: if i.cfg.dense[layer] { ExpertHome::Dense }
                else if working.rtx_layers.contains(&layer) { if i.gpus.len() == 2 { ExpertHome::RtxTp2 } else { ExpertHome::RtxWhole { gpu: 0 } } }
                else { ExpertHome::Spark } }).collect();
        let hop = hop_spec(i);
        let mut placement = Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: i.attention.mode,
            attention_by_kind: vec![(AttentionClass::Dsa, i.attention.mode), (AttentionClass::Kda, AttentionPlacement::Heads)],
            peer_row_bytes: 0, onboard_layers: count, max_pool_tokens_no_onboard: None, layers,
            movables: solved.decisions.draft_home.into_iter().map(|gpu| (MovableId::Drafter, gpu)).collect(),
            expert_ranges: (0..i.gpus.len()).map(|gpu| ExpertRange { first: working.rtx_layers.first().copied().unwrap_or(0),
                layers: if i.gpus.len() == 1 { count } else { 0 }, peak_bytes: if i.gpus.len() == 1 { self.arena[option][gpu] } else { 0 } }).collect(),
            residual: residual_homes(modes, &hop).map_err(invalid)?, hops: plan_hops(modes, &hop).map_err(invalid)?,
            tp2: (i.gpus.len() == 2 && count > 0).then(|| Tp2Range {
                first: working.rtx_layers[0], layers: count, peak_bytes: [self.arena[option][0], self.arena[option][1]] }),
            items: (0..i.gpus.len()).map(|gpu| solved.items(p::Location::coordinator(gpu as u8))).collect::<Result<_, _>>()? };
        super::super::GLM5_FLASH.check(&placement).map_err(invalid)?;
        for (gpu, items) in placement.items.iter_mut().enumerate() {
            if solved.facts.allocations.iter().any(|owner| owner.location == p::Location::coordinator(gpu as u8)
                && owner.category == Category::Embedding && owner.group == "embedding" && owner.view.is_some()) {
                items.push(cuteafd_core::memory_layout::Item::new(Category::Embedding, "embedding",
                    "alias of target weight arena; charged in weight arena", 0, Basis::Exact));
            }
            items.extend(solved.facts.allocations.iter().filter(|owner|
                owner.location == p::Location::coordinator(gpu as u8)
                    && owner.view.is_none() && owner.lifetime.first == p::Phase::Growth)
                .map(|owner| cuteafd_core::memory_layout::Item::new(owner.category, &owner.group, "",
                    owner.bytes, Basis::Exact).with_lifetime(cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve)));
        }
        let encoder = solved.decisions.encoder.map(|index| self.encoders[index].clone());
        tracing::debug!(target: "cuteafd::planner", complete_solve_us = self.started.elapsed().as_micros() as u64,
            graph_reserve = ?graphs.charges(),
            "Flash recipe preparation, search and exact reconciliation completed");
        Ok(AdmittedGlmf { placement, graphs, working, solved,
            native_load: self.native_load, mount_load: self.mount_load, encoder })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn measured_process(host: (u64, u64), headroom: u64) -> ProcessOwners {
        let mut process = ProcessOwners::new(host, headroom);
        process.spark_rank_caps = crate::serving_capacity::spark_memory::serving_bytes();
        process
    }
    use crate::plan::exlr8::residency::{ExpertSet, Residency};

    #[test]
    fn trellis_staging_is_target_only_and_rank_local() -> anyhow::Result<()> {
        use p::Recipe;
        use crate::serving_capacity::WeightArenaLayout;
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4))?;
        let input = crate::placement::families::glm5_flash::tests::inputs(&cfg, 2, 96 << 30);
        for ranks in [&[][..], &[0][..], &[0, 1][..]] {
            let mut process = measured_process((512 << 30, 512 << 30), 8 << 30);
            let embedding = input.resident.iter().map(|rank| rank.embedding).sum::<u64>() as usize;
            let homes = if embedding == 0 { 1 } else { 2 };
            for home in 0..homes {
                let mut layouts = Vec::new();
                for rank in 0..2 {
                    let mut tensors = vec![("weights".into(), input.resident[rank].weights as usize)];
                    if ranks.contains(&(rank as u8)) {
                        tensors.extend([("q.trellis".into(), 1 << 20), ("o.trellis".into(), 1 << 20)]);
                    }
                    if embedding > 0 && home == rank { tensors.push(("embedding.table".into(), embedding)); }
                    layouts.push(WeightArenaLayout::new(format!("target.{rank}"), tensors)?);
                }
                process.target_arenas.push(layouts);
            }
            let prepared = prepare(&input, 188, true, process)?;
            let contract = prepared.recipe.prepare(&prepared.request)?;
            let facts = contract.facts(&vec![0; contract.variables.len()], 1)?;
            crate::shared::dense_read::assert_trellis_device_staging(&facts, ranks);
        }
        Ok(())
    }

    #[test]
    fn sampled_preweight_is_one_protected_capacity_neutral_owner_per_rank() {
        use cuteafd_core::memory_policy::MemoryPolicy;
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for gpus in [1, 2] {
            let mut input = crate::placement::families::glm5_flash::tests::inputs(&cfg, gpus,
                crate::placement::inventory::PRO_TOTAL_BYTES);
            let samples = [682_885_120, 691_273_728];
            for (rank, (total, baseline)) in input.gpus.iter_mut().enumerate() {
                *baseline = Baseline::Measured { free_bytes: *total - samples[rank] };
            }
            for reserve in [None, Some(0), Some(200 << 20)] {
                MemoryPolicy { gpu_reserve: reserve, ..Default::default() }.scoped(|| {
                    let mut reference = prepare(&input, 188, true, measured_process((512 << 30, 512 << 30), 8 << 30)).unwrap();
                    reference.recipe.contract.components.retain(|component| !matches!(&component.kind,
                        p::ComponentKind::Allowance(owner) if crate::placement::sampled_baseline::is_source(&owner.source)));
                    for rank in 0..gpus {
                        reference.budgets[rank].baseline = p::Baseline::Sampled {
                            free: input.gpus[rank].0 - samples[rank], phase: p::Phase::Load, source: "reference".into() };
                    }
                    let before = reference.admit().unwrap().solved;
                    let plan = prepare(&input, 188, true, measured_process((512 << 30, 512 << 30), 8 << 30))
                        .unwrap().admit().unwrap().solved;
                    assert_eq!(plan.pool_tokens, before.pool_tokens);
                    assert_eq!(plan.decisions, before.decisions);
                    for rank in 0..gpus {
                        let owners = plan.facts.allowances.iter().filter(|owner|
                            owner.location == p::Location::coordinator(rank as u8)
                                && crate::placement::sampled_baseline::is_source(&owner.source)).collect::<Vec<_>>();
                        assert_eq!(owners.len(), 1);
                        assert_eq!(owners[0].bytes, samples[rank]);
                        for phase in [p::Phase::Load, p::Phase::Ready, p::Phase::Growth] {
                            let now = plan.phases.iter().find(|row| row.budget == format!("rtx{rank}") && row.phase == phase).unwrap();
                            let old = before.phases.iter().find(|row| row.budget == format!("rtx{rank}") && row.phase == phase).unwrap();
                            assert_eq!(now.available - now.owned - now.opaque, old.available - old.owned - old.opaque);
                        }
                    }
                });
            }
        }
    }

    /// SM120 retains a 2 MiB driver slab per four peak live streams (#1511);
    /// the charge is one named resident allowance per coordinator GPU.
    #[test]
    fn driver_stream_slabs_are_named_resident_allowances_per_gpu() {
        assert_eq!((startup_peak_live_streams(1), startup_peak_live_streams(2)), (8, 9));
        assert_eq!(p::cuda_stream_slab_bytes(8).unwrap(), 4 << 20);
        assert_eq!(p::cuda_stream_slab_bytes(9).unwrap(), 4 << 20);
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for ranks in [1, 2] {
            let mut input = crate::placement::families::glm5_flash::tests::inputs(&cfg, ranks,
                crate::placement::inventory::PRO_TOTAL_BYTES);
            input.drafter_bytes = 0;
            input.drafter_weights = ResidentWeights(0);
            input.requested_pool = Some(262_144);
            let solved = prepare(&input, 188, true, measured_process((512 << 30, 512 << 30), 8 << 30))
                .unwrap().admit().unwrap().solved;
            let slabs = p::cuda_stream_slab_bytes(startup_peak_live_streams(ranks)).unwrap();
            for rank in 0..ranks {
                let owners = solved.facts.allowances.iter().filter(|owner|
                    owner.location == p::Location::coordinator(rank as u8)
                        && owner.source.starts_with(&format!("glmf/driver-stream-slabs/gpu{rank} "))
                        && owner.lifetime.contains(p::Phase::Ready, solved.facts.phase_order)).collect::<Vec<_>>();
                assert_eq!(owners.len(), 1, "{ranks} GPUs, rank {rank}");
                assert_eq!(owners[0].bytes, slabs);
                assert!(!owners[0].assumed);
            }
        }
    }

    #[test]
    fn selected_arenas_replace_payload_and_share_activation_with_their_single_tail() -> anyhow::Result<()> {
        use crate::plan::testing::{glm5_flash_config, glm5_flash_tensors, write_snapshot};
        use crate::serving_capacity::WeightArenaLayout;
        let root = tempfile::tempdir()?;
        let config = glm5_flash_config(4);
        write_snapshot(root.path(), &config, &glm5_flash_tensors(&config), None);
        let checkpoint = crate::plan::Checkpoint::open(root.path())?;
        let cfg = GlmNextConfig::from_hf(&config)?;
        for ranks in [1, 2] {
            let mut input = super::super::tests::inputs(&cfg, ranks, 96 << 30);
            input.requested_pool = Some(262_144);
            if ranks == 2 { input.attention.mode = AttentionPlacement::Layers; }
            input.resident = crate::families::glm5_flash::resident::resident_weights_placed(
                &checkpoint, &cfg, cfg.layers, ranks, input.representation,
                input.attention.mode, 0).map_err(anyhow::Error::msg)?;
            let mut process = measured_process((512 << 30, 512 << 30), 8 << 30);
            if ranks == 2 {
                process.alternate_resident = Some(crate::families::glm5_flash::resident::resident_weights_placed(
                    &checkpoint, &cfg, cfg.layers, ranks, input.representation,
                    input.attention.mode, 1).map_err(anyhow::Error::msg)?);
            }
            process.target(&checkpoint, &input, None)?;
            process.draft_arenas = (0..ranks).map(|rank| WeightArenaLayout::new(
                format!("glm5_flash/draft/{rank}"), [("a".into(), (3 << 20) + 1), ("b".into(), 17)]))
                .collect::<anyhow::Result<_>>()?;
            let targets = process.target_arenas.clone();
            let alternate_targets = process.alternate_target_arenas.clone();
            let drafts = process.draft_arenas.clone();
            let prepared = prepare(&input, 188, true, process)?;
            let components = &prepared.recipe.contract.components;
            for (index, owner) in components.iter().enumerate() {
                let p::ComponentKind::Allocation { template, .. } = &owner.kind else { continue; };
                if template.group != "weight arena" { continue; }
                let resident = components.iter().find(|other| other.activation == owner.activation
                    && matches!(&other.kind, p::ComponentKind::ResidentWeight(weight) if weight.id == template.id)).unwrap();
                let tail = components.iter().find(|other| other.activation == owner.activation
                    && matches!(&other.kind, p::ComponentKind::AllocationGranularity(tail) if tail.id == format!("{}.tail", template.id))).unwrap();
                assert!(matches!(&resident.kind, p::ComponentKind::ResidentWeight(weight) if weight.weights.0 == template.bytes));
                let p::ComponentKind::AllocationGranularity(tail) = &tail.kind else { unreachable!() };
                assert_eq!(tail.requests(), [template.bytes]);
                assert_eq!(tail.lifetime, template.lifetime);
                assert!(!components[..index].iter().any(|other| other.activation == owner.activation
                    && matches!(&other.kind, p::ComponentKind::Allocation { template: prior, .. } if prior.id == template.id)));
            }
            let admitted = prepared.admit()?;
            let home = super::super::embedding_home(&admitted.placement) as usize;
            let targets = if ranks == 2 && super::super::placed_attention(input.attention, &admitted.placement).first_owner == 1 {
                &alternate_targets
            } else { &targets };
            for (rank, layout) in targets[home].iter().enumerate() {
                let facts = &admitted.solved.facts;
                let root = facts.allocations.iter().find(|owner| owner.id == layout.id()).unwrap();
                assert_eq!((root.bytes, root.location, root.lifetime),
                    (layout.bytes() as u64, p::Location::coordinator(rank as u8), p::Lifetime::RESIDENT));
                assert_eq!(facts.allocation_granularity.iter().filter(|tail| tail.backings.iter()
                    .any(|backing| backing.backing == root.id)).count(), 1);
                assert!(!facts.allocations.iter().any(|owner| owner.group == "target weights"));
                let aliases = facts.allocations.iter().filter(|owner| owner.category == Category::Embedding
                    && owner.location == root.location).collect::<Vec<_>>();
                assert_eq!(aliases.len(), usize::from(rank == home));
                for alias in aliases {
                    let view = alias.view.as_ref().unwrap();
                    let expected = layout.view("embedding.table").unwrap();
                    assert_eq!((&view.backing, view.offset, alias.bytes),
                        (&root.id, expected.offset() as u64, expected.bytes() as u64));
                }
            }
            let draft = admitted.solved.facts.allocations.iter().find(|owner| owner.id == drafts[0].id()).unwrap();
            assert_eq!(draft.bytes, drafts[0].bytes() as u64);
            assert_eq!(draft.lifetime, p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::Growth });
            let owned = admitted.solved.facts.allocations.iter().find(|owner| owner.id == "glmf.draft.0").unwrap();
            assert_eq!(owned.bytes, input.drafter_bytes - input.drafter_weights.0);
            assert_eq!(admitted.solved.facts.phase_order, p::PhaseOrder::TARGET_THEN_DRAFT);
        }
        Ok(())
    }

    #[test]
    fn tp2_draft_arenas_replace_only_immutable_physical_owners() -> anyhow::Result<()> {
        use crate::{families::glm5::{draft_physical::GlmDflashConfig,
            draft_representation::{GlmDraftCapacity, GlmDraftRepresentation}},
            families::qwen4::Qwen4KvCache, plan::testing::{t, write_safetensors}};
        let snapshot = tempfile::tempdir()?;
        let config = serde_json::json!({"hidden_size":4096, "intermediate_size":12288, "num_hidden_layers":2,
            "num_attention_heads":32, "num_key_value_heads":8, "head_dim":128, "vocab_size":154880,
            "rms_norm_eps":1e-5, "sliding_window":2048, "rope_theta":10000,
            "dflash_config":{"target_layer_ids":[0,8], "conv_group_size":16, "conv_kernel_size":2,
                "mask_token_id":0, "selector_rank":256, "selector_top_k":16, "block_size":8}});
        std::fs::write(snapshot.path().join("config.json"), config.to_string())?;
        let draft = GlmDflashConfig::from_value(&config)?;
        write_safetensors(&snapshot.path().join("model.safetensors"), &draft.tensor_shapes().into_iter()
            .map(|(name, shape)| t(name, "BF16", &shape)).collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4))?;
        for representation in [GlmDraftRepresentation::Fp8Only, GlmDraftRepresentation::Bf16Only] {
            let mut input = super::super::tests::inputs(&cfg, 2, 96 << 30);
            input.drafter_bytes = 0;
            input.drafter_weights = ResidentWeights(0);
            input.requested_pool = Some(262_144);
            let mut process = measured_process((512 << 30, 512 << 30), 8 << 30);
            process.draft_tp2 = super::super::drafter_tp2_inventory(snapshot.path(),
                GlmDraftCapacity::new(8, 8, draft.block)?, &[188, 188], representation, 2, Qwen4KvCache::Bf16)?;
            process.draft(snapshot.path(), representation, 2, true)?;
            let layouts = process.draft_tp2.clone();
            let arenas = process.draft_arenas.clone();
            let admitted = prepare(&input, 188, true, process)?.admit()?;
            assert_eq!(admitted.graphs.charges(), [198 << 20; 2]);
            for policy in &admitted.graphs.policies {
                assert_eq!(policy.base_bytes, 190 << 20);
                assert_eq!(policy.growth_headroom, Some(8 << 20));
            }
            let facts = admitted.solved.facts;
            for rank in 0..2 {
                let prefix = if rank == 0 { "glmf.draft" } else { "glmf.draft.peer" };
                let location = p::Location::coordinator(rank as u8);
                let immutable = layouts[rank].weight_owners().map(|owner| format!("{prefix}.{}", owner.id))
                    .collect::<std::collections::BTreeSet<_>>();
                let expected = layouts[rank].allocations(prefix, location).into_iter()
                    .filter(|owner| !immutable.contains(&owner.id)).collect::<Vec<_>>();
                for owner in expected {
                    assert!(facts.allocations.contains(&owner), "changed nonweight owner {}", owner.id);
                    let tails = facts.allocation_granularity.iter().filter(|tail|
                        tail.backings.iter().any(|backing| backing.backing == owner.id)).collect::<Vec<_>>();
                    let slack = if owner.location == location {
                        p::sm120_request_backing(owner.bytes)? - owner.bytes
                    } else { 0 };
                    assert_eq!(tails.len(), usize::from(slack != 0), "{}", owner.id);
                    for tail in tails {
                        assert_eq!((tail.location, tail.lifetime, tail.bytes()?),
                            (location, owner.lifetime, slack), "{}", owner.id);
                    }
                }
                assert!(!facts.allocations.iter().any(|owner| immutable.contains(&owner.id)));
                assert!(!facts.resident_weights.iter().any(|owner| immutable.contains(&owner.id)));
                let root = facts.allocations.iter().find(|owner| owner.id == arenas[rank].id()).unwrap();
                assert_eq!((root.bytes, root.location, root.lifetime), (arenas[rank].bytes() as u64, location,
                    p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::Growth }));
                assert_eq!(facts.resident_weights.iter().find(|owner| owner.id == root.id).unwrap().weights.0, root.bytes);
                let tails = facts.allocation_granularity.iter().filter(|tail| tail.backings.iter()
                    .any(|backing| backing.backing == root.id)).collect::<Vec<_>>();
                assert_eq!(tails.len(), 1);
                assert_eq!(tails[0].requests(), [root.bytes]);
                assert_eq!(tails[0].lifetime, root.lifetime);
            }
            assert_eq!(facts.phase_order, p::PhaseOrder::TARGET_THEN_DRAFT);
        }
        Ok(())
    }

    #[test]
    fn retained_heads_and_portable_prefix_follow_selected_physical_owners() {
        use crate::plan::encoder::{EncoderKind, EncoderPlacement};
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for (draft, waves, head) in [
            (false, 1, p::serving::BlasHead::TensorLt), (true, 1, p::serving::BlasHead::TensorLt),
            (true, 3, p::serving::BlasHead::TensorLt), (true, 3, p::serving::BlasHead::Native),
            (false, 1, p::serving::BlasHead::Gemm), (true, 3, p::serving::BlasHead::Gemm),
        ] {
            let mut input = crate::placement::families::glm5_flash::tests::inputs(&cfg, 2,
                crate::placement::inventory::PRO_TOTAL_BYTES);
            input.prefill_lanes = waves;
            input.blas_head = head;
            input.requested_pool = Some(262_144);
            if !draft { input.drafter_bytes = 0; input.drafter_weights = ResidentWeights(0); }
            let mut process = measured_process((512 << 30, 512 << 30), 8 << 30);
            process.encoders = vec![EncoderPlacement { kind: EncoderKind::Rtx { gpu: 1 },
                weights: 1024, scratch: 4096, replicas: vec![], reason: "test".into(), shortfall: 0 }];
            process.serving(&input, 32 << 30, 256 << 20, 0, 0).unwrap();
            let solved = prepare(&input, 188, true, process).unwrap().admit().unwrap().solved;
            let handles = |rank, phase| solved.facts.allowances.iter().filter(|owner|
                owner.location == p::Location::coordinator(rank) && owner.source.starts_with("cuBLAS handles:")
                    && owner.lifetime.contains(phase, solved.facts.phase_order)).map(|owner| owner.bytes).sum::<u64>();
            assert_eq!(handles(0, p::Phase::Ready), u64::from(head.gemm() || (draft && head.bf16())) * (64 << 20));
            assert_eq!(handles(1, p::Phase::Load), 64 << 20);
            assert_eq!(handles(1, p::Phase::Ready), 64 << 20);
            for rank in [0, 1] {
                let init = solved.facts.allowances.iter().filter(|owner|
                    owner.location == p::Location::coordinator(rank) && owner.source.starts_with("BLAS device initialization")
                        && owner.lifetime.contains(p::Phase::Ready, solved.facts.phase_order))
                    .map(|owner| owner.bytes).sum::<u64>();
                assert_eq!(init, if rank == 1 || head.bf16() { 2 << 20 } else { 0 });
                let prefix = solved.facts.driver_page_tables.iter().filter(|owner|
                    owner.location == p::Location::coordinator(rank) && owner.backing.starts_with("glmf.host.prefix."))
                    .collect::<Vec<_>>();
                assert_eq!(prefix.len(), 128);
                assert_eq!(prefix.iter().map(|owner| owner.bytes()).sum::<u64>(), 64 << 20);
                assert!(prefix.iter().all(|owner| owner.lifetime == p::serving::ServingOwners::LIFETIME));
            }
        }
    }

    #[test]
    fn remote_vision_prices_each_sm121_worker_pool() {
        use crate::plan::encoder::{EncoderKind, EncoderPlacement};
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = crate::placement::families::glm5_flash::tests::inputs(&cfg, 2,
            crate::placement::inventory::PRO_TOTAL_BYTES);
        input.blas_head = p::serving::BlasHead::Native;
        input.drafter_bytes = 0;
        input.drafter_weights = ResidentWeights(0);
        input.requested_pool = Some(262_144);
        let mut process = measured_process((512 << 30, 512 << 30), 8 << 30);
        input.spark_ranks = 1;
        process.physical_sparks = 3;
        process.spark_rank_caps = vec![118 << 30, 117 << 30, 116 << 30];
        process.encoders = vec![EncoderPlacement { kind: EncoderKind::Spark { rank: 2 },
            weights: 1024, scratch: 4096, replicas: vec![], reason: "test".into(), shortfall: 0 }];
        let solved = prepare(&input, 188, true, process).unwrap().admit().unwrap().solved;
        let media = solved.budgets.iter().find(|budget| budget.id == "spark2").unwrap();
        assert_eq!(media.physical_bytes, 116 << 30);
        assert_eq!(media.logical_bytes, 116 << 30);
        for rank in [2] {
            let location = p::Location { device: p::Device::Spark(rank), domain: p::Domain::Managed };
            let owners = solved.facts.allowances.iter().filter(|owner| owner.location == location
                && owner.lifetime.contains(p::Phase::Ready, solved.facts.phase_order)).collect::<Vec<_>>();
            assert_eq!(owners.iter().filter(|owner| owner.source.starts_with("cuBLAS handles:"))
                .map(|owner| owner.bytes).sum::<u64>(), 67_508_224);
            assert_eq!(owners.iter().filter(|owner| owner.source.starts_with("cuBLAS first use"))
                .map(|owner| owner.bytes).sum::<u64>(), 158_674_944);
        }
    }

    #[test]
    fn native_shared_receipt_checks_rank_scratch_and_retained_metadata() {
        let mut plan = Residency { layer: 3, world: 2, rank: 1, hidden: 128,
            experts: 1, units: 2, set: ExpertSet::Shared, set_id: "identity".into(),
            manifest: "k4".into(), loads: vec![], signs: [vec![1; 128], vec![1; 128]],
            signs_offsets: [0, 128], pointers_offset: 256, bits_offset: 512,
            compact_offsets: [0; 2], resident_bytes: 4096, payload_bytes: 0, staging_bytes: 1024 };
        let admitted = NativeLoad { vocabulary: 256, mount_pinned: 4096, mount_rows: 128,
            shared_pinned: plan.mount_pinned_bytes().unwrap() as u64,
            shared_scratch: plan.descriptor_host_bytes().unwrap() as u64,
            shared_retained: vec![0, plan.host_bytes() as u64],
            sync_upload: plan.sync_upload_bytes().unwrap() as u64 };
        admitted.validate_shared(std::slice::from_ref(&plan)).unwrap();
        plan.staging_bytes += 1;
        assert!(admitted.validate_shared(std::slice::from_ref(&plan)).is_err());
        plan.staging_bytes -= 1;
        plan.signs[0].push(2);
        assert!(admitted.validate_shared(std::slice::from_ref(&plan)).is_err());
        plan.signs[0] = vec![1; 128];
        plan.rank = 0;
        assert!(admitted.validate_shared(std::slice::from_ref(&plan)).is_err());
        plan.rank = 1;
        plan.set = ExpertSet::Routed;
        assert!(admitted.validate_shared(std::slice::from_ref(&plan)).is_err());
        assert!(admitted.validate_shared(&[]).is_err());
    }

    #[test]
    fn draft_push_owners_charge_draft_loading_only_on_the_selected_home() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let input = crate::placement::families::glm5_flash::tests::inputs(&cfg, 1, 96 << 30);
        let draft = DraftPush { ranks: 1, device_scratch: 32 << 20 };
        let solve = |push: Option<DraftPush>| {
            let mut process = measured_process((512 << 30, 512 << 30), 8 << 30);
            process.draft_push = push;
            prepare(&input, 188, true, process).unwrap().admit().unwrap().solved
        };
        let (mount, pushed) = (solve(None), solve(Some(draft.clone())));
        let owners = pushed.facts.allocations.iter().filter(|a| a.id.starts_with("glmf.draft.push.")).collect::<Vec<_>>();
        assert!(mount.facts.allocations.iter().all(|a| !a.id.starts_with("glmf.draft.push.")));
        assert_eq!(owners.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
            ["glmf.draft.push.scratch.0"]);
        let draft_load = p::Lifetime { first: p::Phase::DraftLoad, last: p::Phase::DraftLoad };
        assert!(owners.iter().all(|a| a.lifetime == draft_load));
        assert!(owners.iter().all(|a| a.location == p::Location::coordinator(0)));
        let phase = |plan: &p::ResolvedPlan, budget: &str, phase| plan.phases.iter()
            .find(|row| row.budget == budget && row.phase == phase).map(|row| row.owned).unwrap();
        assert_eq!(phase(&pushed, "coordinator host RAM", p::Phase::DraftLoad),
            phase(&mount, "coordinator host RAM", p::Phase::DraftLoad));
        for at in [p::Phase::Load, p::Phase::Ready] {
            assert_eq!(phase(&pushed, "coordinator host RAM", at), phase(&mount, "coordinator host RAM", at));
        }
        // A BF16 drafter stages nothing; FP8 packs read one bounded device window.
        let bf16 = DraftPush { ranks: 1,
            device_scratch: 0 };
        assert!(bf16.allocations(0).unwrap().iter().all(|a| !a.id.contains("scratch") && !a.id.ends_with(".dense")));
    }

    #[test]
    fn native_reader_owners_follow_storage_route_and_physical_lifetimes() {
        let receipt = NativeLoad { vocabulary: 256, mount_pinned: 1024, mount_rows: 128,
            shared_pinned: 512, shared_scratch: 64, shared_retained: vec![32, 48], sync_upload: 16 };
        for push in [false, true] {
            let owners = receipt.allocations(push);
            let readers = owners.iter().find(|owner| owner.id == "glmf.load.readers").unwrap();
            assert_eq!(readers.bytes, if push { 256 } else { 1536 });
            assert_eq!(readers.location, p::Location::PINNED);
            assert_eq!(readers.lifetime, p::Lifetime { first: p::Phase::Load, last: p::Phase::Load });
            assert_eq!(owners.iter().any(|owner| owner.id == "glmf.load.rows"), !push);
            let retained = owners.iter().filter(|owner| owner.id.contains("shared.retained")).collect::<Vec<_>>();
            assert_eq!(retained.iter().map(|owner| owner.bytes).sum::<u64>(), 80);
            assert!(retained.iter().all(|owner| owner.location == p::Location::HOST
                && owner.lifetime == p::Lifetime::RESIDENT));
            let scratch = owners.iter().find(|owner| owner.id == "glmf.load.shared.scratch").unwrap();
            assert_eq!(scratch.lifetime.last, p::Phase::Load);
        }
    }

    #[test]
    fn shared_replay_requires_admitted_spark_experts() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let routed = cfg.dense[..cfg.layers].iter().filter(|&&d| !d).count();
        let mut input = crate::placement::families::glm5_flash::tests::inputs(&cfg, 1, 96 << 30);
        input.shared_replay = true;
        input.spark_ranks = 2;

        // Admitted Spark experts succeed.
        let process = measured_process((512 << 30, 512 << 30), 8 << 30);
        assert!(prepare(&input, 188, true, process).is_ok());

        // Media-only Spark ranks without remote experts refuse shared replay.
        let mut media_only = measured_process((512 << 30, 512 << 30), 8 << 30);
        media_only.remote_experts = false;
        let err = prepare(&input, 188, true, media_only).err().unwrap().to_string();
        assert!(err.contains("--replay-records shared needs one coordinator GPU with Spark experts"), "{err}");

        // Onboarding all routed layers to RTX leaves admitted expert world 0 and refuses shared replay.
        input.onboard = Onboard::Layers(routed);
        let process = measured_process((512 << 30, 512 << 30), 8 << 30);
        let err = prepare(&input, 188, true, process).err().unwrap().to_string();
        assert!(err.contains("--replay-records shared needs one coordinator GPU with Spark experts"), "{err}");
    }
}
