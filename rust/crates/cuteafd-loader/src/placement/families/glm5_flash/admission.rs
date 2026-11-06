//! Finite GLM Flash physical alternatives; the shared planner owns selection.
use super::*;
use crate::placement::planner as p;

/// Caller-resolved process owners. Preparing a recipe never samples memory.
#[derive(Debug, Clone)]
pub struct ProcessOwners {
    pub host: (u64, u64),
    pub host_headroom: u64,
    pub allocations: Vec<p::Allocation>,
    pub links: Vec<PreparedSparkLink>,
    pub native_load: Option<NativeLoad>,
    pub encoders: Vec<crate::plan::encoder::EncoderPlacement>,
    pub media_cache_bytes: u64,
    pub alternate_resident: Option<Vec<GlmfResidentRank>>,
    pub push_host: Option<crate::push_load::PushHostEnvelope>,
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
        Self { host, host_headroom, allocations: vec![], links: vec![], native_load: None,
            encoders: vec![], media_cache_bytes: 0, alternate_resident: None, push_host: None }
    }

    pub fn serving(&mut self, i: &GlmfInputs<'_>, quota: u64, chunk: u64, host_embedding: u64,
        media: u64) -> Result<(), p::Refusal> {
        let lifetime = p::serving::ServingOwners::LIFETIME;
        self.allocations.extend(p::serving::HostPrefixOwners { quota_bytes: quota, chunk_bytes: chunk }
            .allocations("glmf.host.prefix", lifetime)?);
        let logits = cuteafd_core::memory_layout::host_logits_bytes(
            if i.full_prefill_logits > 0 { i.prefill_rows * i.prefill_lanes } else { 64 },
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
            topk: i.cfg.topk as u32, intermediate: i.cfg.moe_intermediate as u32, layers: i.cfg.layers as u32 };
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

/// Header-only estimate shared by offline planning and live startup. Exact
/// selected DenseArena records validate against this receipt before offers;
/// they never resample capacity or run a second solve.
pub fn push_host_envelope(checkpoint: &crate::plan::Checkpoint, cfg: &GlmNextConfig,
    ranks: usize, catalog: Option<&crate::OfficialV41Catalog>,
    native: Option<(&crate::plan::exlr8::native::DenseFacts,
        &crate::plan::exlr8::residency::MetadataInput)>) -> anyhow::Result<Option<crate::push_load::PushHostEnvelope>> {
    use anyhow::{Context, ensure};
    match crate::push_load::namespace(&checkpoint.snapshot) {
        Ok(_) => {},
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
            && error.raw_os_error().is_none() && checkpoint.snapshot.exists() => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    ensure!((1..=2).contains(&ranks), "Flash push rank count");
    let slots = cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64;
    let slot = cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64;
    let step = crate::shared::dense_read::STEP_BYTES as u64;
    let mut largest = checkpoint.tensors.iter().filter(|tensor|
        !tensor.meta.name.contains(".experts.")).map(|tensor| tensor.meta.byte_length).max().unwrap_or(0);
    if let Some((facts, _)) = native {
        largest = largest.max(facts.records.iter().flat_map(|record| &record.components)
            .map(|component| component.source.bytes as u64).max().unwrap_or(0));
    }
    // A transform is split into <=32 MiB windows. Two uses of a source
    // (ordinary and absorbed operands) can straddle every outstanding slot;
    // include both boundary windows and the worst residual/table workspace.
    // This is deliberately an assumed bound until the chosen records exist.
    let largest = largest.min(step);
    let dense = largest.checked_mul(2 * slots + 1).and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(slots * slot))
        .and_then(|n| n.checked_add(10 * largest)).context("Flash dense push estimate overflow")?;
    let mut expert = vec![0u64; ranks];
    let mut retained = vec![0u64; ranks];
    for rank in 0..ranks {
        if let Some((_, metadata)) = native {
            for layer in (0..cfg.layers).filter(|&layer| !cfg.dense[layer]) {
                let plan = metadata.plan(layer, ranks, rank, crate::plan::exlr8::residency::ExpertSet::Routed)?;
                expert[rank] = expert[rank].max(plan.descriptor_host_bytes()? as u64);
                retained[rank] = retained[rank].checked_add(plan.host_bytes() as u64)
                    .context("Flash retained expert metadata overflow")?;
            }
        } else if let Some(manifest) = catalog.and_then(|catalog| catalog.exl3()) {
            for layer in (0..cfg.layers).filter(|&layer| !cfg.dense[layer]) {
                let plan = manifest.residency(crate::V41Exl3Layer::Backbone(layer), ranks, rank)?;
                let bytes = plan.loads.iter().filter(|load| load.tensor.ends_with(".trellis")).count() as u64 * 4;
                expert[rank] = expert[rank].checked_add(bytes).context("Flash validation image overflow")?;
            }
        } else {
            // Native FP8 post-Done checks use bounded widening chunks and the
            // checkpoint's scalar/input vectors, not the routed weight image.
            expert[rank] = catalog.into_iter().flat_map(|catalog| catalog.tensors()).filter(|tensor|
                tensor.metadata.name.ends_with("input_scale") || tensor.metadata.name.ends_with("weight_scale_2"))
                .map(|tensor| tensor.metadata.byte_length).max().unwrap_or(0).checked_mul(2)
                .context("Flash FP8 input scratch overflow")?.max(2 << 20);
        }
    }
    Ok(Some(crate::push_load::PushHostEnvelope::with_retained(expert, vec![dense; ranks], retained, slots, slot)?))
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
    started: std::time::Instant,
    input: &'a GlmfInputs<'a>,
    pub recipe: GlmfRecipe,
    pub budgets: Vec<p::Budget>,
    pub request: p::Request,
    counts: Vec<usize>,
    arena: Vec<Vec<u64>>,
    sms: usize,
    native_load: Option<NativeLoad>,
    push_host: Option<crate::push_load::PushHostEnvelope>,
    encoders: Vec<crate::plan::encoder::EncoderPlacement>,
}

pub struct AdmittedGlmf {
    pub placement: Placement,
    pub graphs: GraphSet,
    pub working: GlmfWorkingSet,
    pub solved: p::ResolvedPlan,
    /// Unused estimate-minus-selected bytes; selection is never rerun to reclaim them.
    pub graph_slack: Vec<u64>,
    pub native_load: Option<NativeLoad>,
    pub push_host: Option<crate::push_load::PushHostEnvelope>,
    pub encoder: Option<crate::plan::encoder::EncoderPlacement>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TopologyRefusal {
    #[error("all routed layers fit on the RTX GPUs; set SPARK_COUNT=0")]
    IdleSparks,
}

impl AdmittedGlmf {
    pub fn validate_spark_topology(&self, requested: usize) -> Result<(), TopologyRefusal> {
        if requested > 0 && self.working.spark_ranks == 0 {
            return Err(TopologyRefusal::IdleSparks);
        }
        Ok(())
    }
}

fn invalid(error: impl std::fmt::Display) -> p::Refusal {
    p::Refusal::InvalidFact(error.to_string())
}

fn component(id: impl Into<String>, gpu: u8, category: Category, group: &str,
    bytes: p::ByteCurve, lifetime: p::Lifetime, maximum: u64) -> Result<p::Component, p::Refusal> {
    let allocation = p::Allocation::buffer(id, p::Location::coordinator(gpu), bytes.at(1)?,
        lifetime, category, group);
    let mut owner = p::Component::allocation(allocation, bytes.clone(), false, maximum);
    owner.byte_extent(bytes)?;
    Ok(owner)
}

fn add_demand(contract: &mut p::Contract, demand: &Demand, id: String,
    activation: p::Activation) -> Result<(), p::Refusal> {
    let lifetime = if demand.lifetime == cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve {
        p::Lifetime { first: p::Phase::Growth, last: p::Phase::Growth }
    } else if matches!(demand.category, Category::Weights | Category::Embedding | Category::Runtime) {
        p::Lifetime::RESIDENT
    } else { p::Lifetime::TARGET };
    let maximum = contract.maximum_pages;
    let mut owner = component(id.clone(), demand.gpu, demand.category, &demand.group,
        p::ByteCurve::fixed(demand.bytes, maximum), lifetime, maximum)?;
    owner.activation = activation.clone();
    contract.components.push(owner);
    if demand.resident_weights.0 > 0 {
        let mut weight = p::Component::resident_weight(p::ResidentWeight {
            id, location: p::Location::coordinator(demand.gpu), weights: demand.resident_weights,
        }, p::ByteCurve::fixed(demand.resident_weights.0, maximum));
        weight.activation = activation;
        contract.components.push(weight);
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
    let mut prepared = prepare_fixed(i, sms, rtx_experts, process.clone())?;
    if let Some(resident) = alternate {
        if i.gpus.len() != 2 || i.attention.mode != AttentionPlacement::Layers || resident.len() != 2 {
            return Err(invalid("Flash alternate attention owner inventory"));
        }
        let mut input = i.clone();
        input.attention.first_owner ^= 1;
        input.resident = resident;
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
    if i.graphs.ranks.iter().any(|rank| rank.margin > rank.bytes) {
        return Err(invalid("Flash graph margin exceeds physical extent"));
    }
    let facts = facts(i).map_err(invalid)?;
    let gpus = i.gpus.len();
    if i.max_context == 0 || i.spark_ranks > 8
        || i.graphs.ranks.iter().any(|rank| rank.margin > rank.bytes)
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
    if i.shared_replay && (gpus != 1 || i.spark_ranks == 0) {
        return Err(invalid("--replay-records shared needs one coordinator GPU with Spark experts"));
    }
    for profile in [&i.workspace, &i.local_workspace] {
        if profile.iter().any(|row| usize::from(row.demand.gpu) >= gpus)
            || (0..gpus).any(|rank| !profile.iter().any(|row|
                usize::from(row.demand.gpu) == rank && row.component == GlmfComponent::Always)) {
            return Err(invalid("GLM Flash step inventory"));
        }
    }
    let pool = PoolPolicy::resolve(&i.gpus.iter().map(|g| g.0).collect::<Vec<_>>(),
        i.max_context, PoolPolicy::normalize_request(i.requested_pool), 256, i.spark_ranks == 0);
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
    if counts.iter().any(|&n| n > routed.len()) || !rtx_experts && counts.iter().any(|&n| n > 0) {
        return Err(p::Refusal::Unsupported);
    }
    let mut contract = p::Contract::fixed(256, maximum, gpus, p::Facts {
        phase_order: p::PhaseOrder::TARGET_THEN_DRAFT, allocations: vec![], deferred: vec![],
        sequential: vec![], reservations: vec![], allowances: vec![], resident_weights: vec![],
    });
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
        contract.variables.push(p::Variable::Encoder((0..process.encoders.len()).collect()));
        for (option, encoder) in process.encoders.iter().enumerate() {
            let when = p::Choice { variable, option };
            let locations = match &encoder.kind {
                EncoderKind::Off => vec![],
                EncoderKind::Rtx { gpu } if *gpu < gpus => vec![p::Location::coordinator(*gpu as u8)],
                EncoderKind::Spark { .. } if !routed.is_empty() && i.spark_ranks > 0 => {
                    if encoder.spark_ranks().iter().any(|&rank| rank >= i.spark_ranks) {
                        return Err(invalid("Flash encoder Spark rank absent"));
                    }
                    contract.requires.push(p::Requires { when, any_of: counts.iter().enumerate()
                        .filter(|(_, count)| **count < routed.len())
                        .map(|(option, _)| p::Choice { variable: onboard_variable, option }).collect() });
                    encoder.spark_ranks().into_iter().map(|rank| p::Location {
                        device: p::Device::Spark(rank as u8), domain: p::Domain::Managed }).collect()
                }
                _ => return Err(invalid("Flash encoder owner absent from executable topology")),
            };
            let activation = p::Activation::choice(variable, option);
            for (copy, location) in locations.iter().copied().enumerate() {
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
        add_demand(&mut contract, &Demand::new(gpu as u8, Category::Runtime, "context+modules", occupied,
            Basis::Calibrated), format!("glmf.runtime.{gpu}"), p::Activation::always())?;
    }
    // HOST and PINNED share one physical capacity; external occupancy is sampled
    // once before the recipe's process-owned bytes are admitted.
    let host = process.host;
    if host.0 == 0 || host.1 > host.0 { return Err(invalid("invalid Flash host capacity")); }
    budgets.push(p::Budget { id: "coordinator host RAM".into(), locations: vec![p::Location::HOST, p::Location::PINNED],
        physical_bytes: host.0, logical_bytes: host.0, headroom: process.host_headroom,
        baseline: p::Baseline::Sampled { free: host.1, phase: p::Phase::Load,
            source: "MemAvailable plus process anonymous RSS".into() } });
    for rank in 0..i.spark_ranks {
        budgets.push(p::Budget { id: format!("spark{rank}"), locations: vec![p::Location {
            device: p::Device::Spark(rank as u8), domain: p::Domain::Managed }],
            physical_bytes: 121 << 30, logical_bytes: 121 << 30, headroom: 0,
            baseline: p::Baseline::Forecast { occupied: 0,
                source: "Spark worker physical capacity; worker owners supplied by caller".into(), assumed: true } });
    }
    for owner in process.allocations.iter().cloned().chain(process.native_load.iter()
        .flat_map(|load| load.allocations(process.push_host.is_some()))) {
        let bytes = p::ByteCurve::fixed(owner.bytes, maximum);
        contract.components.push(p::Component::allocation(owner, bytes, false, maximum));
    }
    if let Some(envelope) = &process.push_host {
        for owner in envelope.allocations("glmf.push")? {
            let curve = p::ByteCurve::fixed(owner.bytes, maximum);
            if owner.id.ends_with(".dense") {
                contract.components.push(p::Component { activation: p::Activation::always(), bytes: curve,
                    kind: p::ComponentKind::Allowance(p::Allowance { location: owner.location,
                        bytes: owner.bytes, lifetime: owner.lifetime,
                        source: format!("{}: header-derived dense staging upper estimate", owner.id), assumed: true }) });
            } else {
                contract.components.push(p::Component::allocation(owner, curve, false, maximum));
            }
        }
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
    for (index, row) in facts.fixed.iter().enumerate() {
        if row.demand.group == "embedding" || matches!(row.demand.group.as_str(), "graphs" | "graph growth")
            || row.demand.group == "Spark intake" && !process.links.is_empty()
            || row.demand.group == "wide decode selector" { continue; }
        let activation = match row.component {
            GlmfComponent::Always => p::Activation::always(),
            GlmfComponent::RtxExperts | GlmfComponent::SparkExperts => p::Activation { clauses: counts.iter()
                .enumerate().filter(|(_, n)| if row.component == GlmfComponent::RtxExperts { **n > 0 }
                    else { **n < routed.len() }).map(|(option, _)| vec![p::Choice { variable: onboard_variable, option }]).collect() },
        };
        add_demand(&mut contract, &row.demand, format!("glmf.fixed.{index}"), activation)?;
    }
    for gpu in 0..gpus {
        if embedding > 0 {
            add_demand(&mut contract, &Demand::new(gpu as u8, Category::Embedding, "embedding", embedding, Basis::Exact)
                .with_resident_weights(ResidentWeights(embedding)), format!("glmf.embedding.{gpu}"),
                p::Activation::choice(embedding_variable, gpu))?;
        }
        let geometry = &facts.geometry.ranks[gpu];
        let bytes = geometry.persistent_unit_bytes.checked_add(geometry.pool_metadata_unit_bytes)
            .and_then(|bytes| bytes.checked_add(geometry.glmf_pool_mark_backing_unit_bytes(i.pool_marks)))
            .ok_or(p::Refusal::Overflow("GLM Flash KV unit"))?;
        contract.components.push(component(format!("glmf.pool.{gpu}"), gpu as u8, Category::Kv, "KV pool",
            p::ByteCurve::linear(bytes, bytes, maximum), p::Lifetime::TARGET, maximum)?);
    }
    if i.attention.mode == AttentionPlacement::Context {
        for (index, demand) in context_buffers(i.cfg, i.max_context, i.prefill_rows, i.prefill_lanes,
            i.attention).demands().map_err(invalid)?.iter().enumerate() {
            add_demand(&mut contract, demand, format!("glmf.context.{index}"), p::Activation::always())?;
        }
    }
    let hop = hop_spec(i);
    let hops = plan_hops(&attention, &hop).map_err(invalid)?;
    for (gpu, bytes) in hop_buffer_bytes(&hops, &hop, gpus).ok_or(p::Refusal::Overflow("GLM Flash hops"))?.into_iter().enumerate() {
        add_demand(&mut contract, &Demand::new(gpu as u8, Category::Transport, "residual hops", bytes,
            Basis::Formula), format!("glmf.hops.{gpu}"), p::Activation::always())?;
    }
    if i.drafter_bytes > 0 {
        let homes = if i.drafter_bridge.is_some() { vec![0, 1] } else { vec![0] };
        let variable = contract.variables.len();
        contract.variables.push(p::Variable::DraftHome(homes.clone()));
        for (option, gpu) in homes.into_iter().enumerate() {
            let activation = p::Activation::choice(variable, option);
            add_demand(&mut contract, &Demand::new(gpu, Category::Drafter, "DFlash", i.drafter_bytes,
                Basis::Exact).with_resident_weights(i.drafter_weights), format!("glmf.draft.{gpu}"), activation.clone())?;
            add_demand(&mut contract, &Demand::new(gpu, Category::Workspace, "DFlash linear/attention scratch", i.drafter_scratch[usize::from(gpu)],
                Basis::Formula), format!("glmf.draft.scratch.{gpu}"), activation.clone())?;
            if gpu != 0 {
                for (index, demand) in i.drafter_bridge.as_ref().into_iter().flat_map(|bridge| &bridge.demands).enumerate() {
                    add_demand(&mut contract, demand, format!("glmf.draft.bridge.{index}"), activation.clone())?;
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
    }
    let mut arena = Vec::new();
    for (option, &count) in counts.iter().enumerate() {
        let activation = if routed.is_empty() { p::Activation::always() }
            else { p::Activation::choice(onboard_variable, option) };
        let working = GlmfWorkingSet::candidate(i, count).map_err(invalid)?;
        let workspace = if working.spark_ranks == 0 { &i.local_workspace } else { &i.workspace };
        for (index, row) in working.demands(workspace).enumerate() {
            add_demand(&mut contract, row, format!("glmf.steps.{option}.{index}"), activation.clone())?;
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
                .with_resident_weights(ResidentWeights(weights)), format!("glmf.experts.{option}.{gpu}"), activation.clone())?;
        }
        arena.push(peak);
    }
    let mut graph = if i.graphs.lifetime == Lifetime::Startup {
        startup_graph_bound(i.cfg, i.layers, i.max_context as usize, (maximum * 256) as usize,
            i.sequences as usize, i.speculation, gpus, i.decode_rows as usize, sms)
    } else { i.graphs.clone() };
    if i.local_tp2_exl3 {
        // Both runtime routes remain below this per-rank maximum.
        let mut local = graph.clone();
        calibrate_local_tp2_graphs(&mut local);
        for (bound, calibrated) in graph.ranks.iter_mut().zip(local.ranks) {
            let ready = (bound.bytes - bound.margin).max(calibrated.bytes - calibrated.margin);
            bound.margin = bound.margin.max(calibrated.margin);
            bound.bytes = ready.checked_add(bound.margin)
                .ok_or(p::Refusal::Overflow("Flash graph bound"))?;
        }
    }
    for gpu in 0..gpus {
        let startup = graph.lifetime == Lifetime::Startup;
        let measured = if startup { graph.bytes(gpu) - graph.ranks[gpu].margin } else { 0 };
        for (suffix, group, bytes, first) in [
            ("ready", "graphs", measured, p::Phase::Warmup),
            ("growth", "graph growth", graph.bytes(gpu) - measured, p::Phase::Growth),
        ] {
            contract.components.push(component(format!("glmf.graphs.{gpu}.{suffix}"), gpu as u8,
                Category::Runtime, group, p::ByteCurve::fixed(bytes, maximum),
                p::Lifetime { first, last: p::Phase::Growth }, maximum)?);
        }
    }
    let required_context = if i.automatic_context {
        if i.spark_ranks == 0 { crate::placement::pool::AGENTIC_FLOOR_TOKENS } else { 256 }
    } else { i.max_context };
    let request = p::Request { context: required_context,
        pool: PoolPolicy::normalize_request(i.requested_pool).map_or(p::Pin::Auto, p::Pin::Exact),
        onboard: i.onboard.layers(routed.len()).map_or(p::Pin::Auto, |n| p::Pin::Exact(n as u64)),
        prefill: p::Pin::Exact(prefill), operation: p::Operation::Serving,
        resident_weight_limit: i.resident_weight_limit,
        spark_reply_formats: contract.spark_reply_formats.unwrap_or_default() };
    Ok(PreparedGlmf { started: std::time::Instant::now(), input: i, recipe: GlmfRecipe { contract }, budgets, request, counts, arena, sms,
        native_load: process.native_load, push_host: process.push_host, encoders: process.encoders })
}

fn hop_spec(i: &GlmfInputs<'_>) -> HopSpec {
    HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: 4 * i.cfg.hidden as u64 * 2,
        rows: i.prefill_rows.max(i.decode_rows), lanes: i.prefill_lanes, entry_gpu: 0, head_gpu: 0,
        terminal: if i.gpus.len() == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } }
}

impl PreparedGlmf<'_> {
    pub fn admit(self) -> Result<AdmittedGlmf, p::Refusal> {
        let mut solved = p::solve(self.request, self.budgets, &self.recipe)?;
        let i = self.input;
        let count = solved.decisions.onboard_layers.iter().flatten().copied()
            .collect::<std::collections::BTreeSet<_>>().len();
        let option = self.counts.iter().position(|&n| n == count).ok_or_else(|| invalid("unprepared onboarding count"))?;
        let working = GlmfWorkingSet::candidate(i, count).map_err(invalid)?;
        let mut graphs = if i.graphs.lifetime == Lifetime::Startup {
            startup_graphs(i.cfg, i.layers, glmf_graphs::admitted_graph_context(i.max_context as usize,
                solved.pool_tokens as usize, i.automatic_context), solved.pool_tokens as usize,
                i.sequences as usize, i.speculation, i.gpus.len(), i.decode_rows as usize, self.sms)
        } else { i.graphs.clone() };
        if i.local_tp2_exl3 && working.spark_ranks == 0 { calibrate_local_tp2_graphs(&mut graphs); }
        let mut graph_slack = vec![0; i.gpus.len()];
        for gpu in 0..i.gpus.len() {
            let ready = if graphs.lifetime == Lifetime::Startup { graphs.bytes(gpu) - graphs.ranks[gpu].margin } else { 0 };
            for (suffix, bytes) in [("ready", ready), ("growth", graphs.bytes(gpu) - ready)] {
                let owner = solved.facts.allocations.iter_mut().find(|owner| owner.id == format!("glmf.graphs.{gpu}.{suffix}"));
                let Some(owner) = owner else {
                    if bytes == 0 { continue; }
                    return Err(invalid("missing admitted graph owner"));
                };
                if bytes > owner.bytes { return Err(invalid("exact graphs exceed prepared estimate")); }
                graph_slack[gpu] += owner.bytes - bytes;
                owner.bytes = bytes;
                owner.shape = vec![bytes];
            }
        }
        // Exact evaluation only decreases the bound. Reconcile the selected
        // facts, never run a second admission or alter the selected geometry.
        solved.phases = p::evaluate(&solved.facts, &solved.budgets)?;
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
            items.extend(solved.facts.allocations.iter().filter(|owner|
                owner.location == p::Location::coordinator(gpu as u8)
                    && owner.view.is_none() && owner.lifetime.first == p::Phase::Growth)
                .map(|owner| cuteafd_core::memory_layout::Item::new(owner.category, &owner.group, "",
                    owner.bytes, Basis::Exact).with_lifetime(cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve)));
        }
        let encoder = solved.decisions.encoder.map(|index| self.encoders[index].clone());
        tracing::debug!(target: "cuteafd::planner", complete_solve_us = self.started.elapsed().as_micros() as u64,
            graph_slack = ?graph_slack, "Flash recipe preparation, search and exact reconciliation completed");
        Ok(AdmittedGlmf { placement, graphs, working, solved, graph_slack, native_load: self.native_load, push_host: self.push_host, encoder })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::exlr8::residency::{ExpertSet, Residency};

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
}
