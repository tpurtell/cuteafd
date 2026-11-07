//! Kimi K3 physical owners for the shared planner. Callers resolve checkpoint,
//! executor and retention facts once; `planner::solve` is the only admission,
//! and both `cuteafd plan --layout` and live serve admission consume the same
//! facts. Nothing here samples free memory or runs a second fit loop.
use super::super::planner::*;
use super::super::{
    hop_buffer_bytes, plan_hops, residual_homes, AttentionClass, AttentionPlacement, Bytes2,
    Demand, ExpertHome, ExpertRange, HopSpec, LayerAssignment, LayerMode, ModeBytes, MovableId,
    Placement, PoolLimits, ResidentWeights, Tp2Range,
};
use crate::formats::dflash::MlaDraftRuntimeLayout;
use cuteafd_core::memory_layout::Category;
use cuteafd_core::spark_reply::SparkReplyFormats;

/// One shared decode/prefill endpoint per lane, using the same finite probe
/// candidates as execution. Its GPU intake is separately owned by the lane.
pub fn production_links(gpu: u8, ranks: usize, lanes: u64, prefill: PrefillShape,
    rings: cuteafd_core::SparkRings, replies: SparkReplyFormats)
    -> Result<Vec<crate::placement::PreparedSparkLink>, Refusal> {
    use crate::placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates};
    if ranks == 0 { return Ok(Vec::new()); }
    (0..lanes).map(|lane| PreparedSparkLink::endpoint(&format!("kimi.lane.{lane}"), gpu,
        SparkEndpointOwners { ranks, geometry: cuteafd_core::ExpertGeometry::KIMI_K3,
            endpoint: cuteafd_core::SparkEndpoint {
                capacity: u32::try_from(prefill.rows).map_err(|_| Refusal::Overflow("Kimi transport rows"))?,
                ingress: cuteafd_core::SparkIngress::Fp8K32, ring_depth: None, partial_reply_bytes: None }, rings },
        usize::try_from(prefill.waves).map_err(|_| Refusal::Overflow("Kimi transport waves"))?,
        true, true, SparkIntakeCandidates::Probed, replies.prefill(false), Lifetime::TARGET)).collect()
}

/// Complete #1329 K6 max push bound was 9,606,907,904 B across both
/// ranks; the split is unknown. Reserve 10 GiB independently on each rank
/// until the shared push log supplies exact per-rank dense bounds.
pub const PUSH_DENSE_BYTES_PER_RANK: u64 = 10 << 30;

pub fn push_host_envelope(cfg: &crate::families::kimi_k3::config::KimiK3Config, gpus: usize)
    -> Result<crate::push_load::PushHostEnvelope, Refusal> {
    // EXL3 validation has one four-byte MCG scalar per expert projection.
    // Any selected local prefix is bounded by the complete routed inventory.
    let validation = (cfg.dense.iter().filter(|dense| !**dense).count() as u64)
        .checked_mul(cfg.experts as u64).and_then(|bytes| bytes.checked_mul(3 * 4))
        .ok_or(Refusal::Overflow("Kimi push EXL3 validation"))?;
    crate::push_load::PushHostEnvelope::new(vec![validation; gpus],
        vec![PUSH_DENSE_BYTES_PER_RANK; gpus], 128, 1 << 20)
}

pub fn selected_push_host_envelope(cfg: &crate::families::kimi_k3::config::KimiK3Config,
    core: &crate::families::kimi_k3::core::CoreInventory, gpus: usize)
    -> Result<crate::push_load::PushHostEnvelope, Refusal> {
    let mut envelope = push_host_envelope(cfg, gpus)?;
    if let Some(largest) = core.largest_selected_dense_bytes {
        let dense = crate::push_load::admitted_push_envelope(largest, envelope.slots as usize,
            envelope.slot_bytes)?;
        envelope.dense_bytes.fill(dense);
    }
    Ok(envelope)
}

pub fn native_load_owners(core: &crate::families::kimi_k3::core::CoreInventory,
    gpus: usize, maximum: u64) -> anyhow::Result<Vec<Component>> {
    Ok(core.native_mount_load(gpus)?.allocations("kimi.native_attention.load").into_iter()
        .map(|owner| Component::allocation(owner.clone(), ByteCurve::fixed(owner.bytes, maximum),
            false, maximum)).collect())
}

pub fn push_host_owners(envelope: &crate::push_load::PushHostEnvelope, maximum: u64)
    -> Result<Vec<Component>, Refusal> {
    Ok(envelope.allocations("kimi.push")?.into_iter().map(|owner| {
        if owner.id.ends_with(".dense") {
            Component { activation: Activation::always(), bytes: ByteCurve::fixed(owner.bytes, maximum),
                kind: ComponentKind::Allowance(Allowance { location: owner.location, bytes: owner.bytes,
                    lifetime: owner.lifetime, assumed: true,
                    source: if envelope.dense_bytes.iter().all(|&bytes| bytes == PUSH_DENSE_BYTES_PER_RANK) {
                        format!("{} estimated: #1329 K6 max smoke 2026-11-06 complete push 9,606,907,904 B; per-rank split unknown", owner.id)
                    } else {
                        format!("{} estimated: selected dense headers and bounded shared push windows; actual records validate before load", owner.id)
                    } }) }
        } else {
            Component::allocation(owner.clone(), ByteCurve::fixed(owner.bytes, maximum), false, maximum)
        }
    }).collect())
}

/// Local-only routed execution still needs startup/lifetime hardware qualification.
pub const LOCAL_ONLY_REFUSAL: Refusal = Refusal::MissingExecutor {
    family: "Kimi K3", executor: "coordinator-only routed experts", issue: 1395,
};

/// Native page quantum of the retained MLA records.
pub const PAGE_ROWS: u64 = 64;

fn invalid(detail: &str) -> Refusal {
    Refusal::InvalidFact(format!("Kimi component contract: {detail}"))
}

/// Attention weight bytes one rank retains for a layer's resolved mode.
fn attention_bytes(weights: &ModeBytes, mode: LayerMode, rank: usize) -> u64 {
    match mode {
        LayerMode::HeadSplit | LayerMode::ContextSplit => weights.split[rank],
        LayerMode::Whole { gpu, .. } if usize::from(gpu) == rank => weights.whole,
        LayerMode::Whole { .. } => 0,
    }
}

/// One rank-fixed owner. `weights` is the retained-weight part of the same
/// exact allocation, never a second charge.
#[derive(Debug, Clone)]
pub struct FixedOwner {
    pub rank: u8,
    pub category: Category,
    pub group: String,
    pub bytes: u64,
    pub weights: u64,
    pub lifetime: Lifetime,
}

impl FixedOwner {
    pub fn ready(demand: &Demand) -> Self {
        Self {
            rank: demand.gpu,
            category: demand.category,
            group: demand.group.clone(),
            bytes: demand.bytes,
            weights: demand.resident_weights.0,
            lifetime: Lifetime::RESIDENT,
        }
    }
}

/// One GPU's local expert arena at one admitted onboard count. The peak
/// includes the transient staging the loader needs above the resident halves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arena {
    pub count: usize,
    pub peak: Vec<u64>,
    pub resident: Vec<u64>,
    pub weights: Vec<u64>,
}

/// One Spark rank's retained work at one admitted onboard count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkOwner {
    pub count: usize,
    pub rank: usize,
    pub experts: u64,
    pub overhead: u64,
    pub runtime: u64,
}

/// One drafter owner whose extent grows with the retained pool.
#[derive(Debug, Clone)]
pub struct DraftOwner {
    pub gpu: u8,
    pub category: Category,
    pub group: String,
    pub bytes_at_first: u64,
    pub bytes_per_page: u64,
    pub weights_at_first: u64,
    pub weights_per_page: u64,
}

/// Drafter owners plus the transient staging that overlaps them while they
/// load. The owners themselves are the resident part of the load peak, so the
/// transient is only the difference above them.
#[derive(Debug, Clone)]
pub struct Draft {
    pub context_split: bool,
    pub owners: Vec<DraftOwner>,
    pub staging_at_first: u64,
    pub staging_per_page: u64,
}

/// The declared local expert prefix domain. `destination` names the
/// coordinator GPUs that hold a layer; TP2 names both and counts it once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertProfile {
    pub layers: Vec<usize>,
    pub destination: Vec<u8>,
    pub counts: Vec<usize>,
    pub rank_sets: Vec<Vec<u8>>,
}

/// Retained routed-expert halves per routed layer for TP2.
pub fn tp2_arenas(counts: &[usize], workspace: [u64; 2], halves: &[[Bytes2; 2]])
    -> Result<Vec<Arena>, Refusal> {
    counts.iter().map(|&count| {
        if count > halves.len() {
            return Err(invalid("arena count exceeds the routed inventory"));
        }
        let mut arena = Arena { count, peak: vec![0; 2], resident: vec![0; 2], weights: vec![0; 2] };
        for gpu in 0..2 {
            let resident = halves[..count].iter().try_fold(workspace[gpu], |sum, pair|
                sum.checked_add(pair[gpu].resident).ok_or(Refusal::Overflow("Kimi TP2 expert arena")))?;
            let staging = halves[..count].iter().map(|pair| pair[gpu].staging).max().unwrap_or(0);
            arena.resident[gpu] = resident;
            arena.peak[gpu] = resident.checked_add(staging)
                .ok_or(Refusal::Overflow("Kimi TP2 expert staging"))?;
            arena.weights[gpu] = resident.checked_sub(workspace[gpu])
                .ok_or(Refusal::Overflow("Kimi TP2 resident weights"))?;
        }
        Ok(arena)
    }).collect()
}

/// Whole-layer arenas: parts load in order into the same workspace.
pub fn whole_arenas(counts: &[usize], workspace: u64, parts: &[Bytes2])
    -> Result<Vec<Arena>, Refusal> {
    counts.iter().map(|&count| {
        if count > parts.len() {
            return Err(invalid("arena count exceeds the routed inventory"));
        }
        let (mut resident, mut peak) = (workspace, workspace);
        for part in &parts[..count] {
            peak = peak.max(resident
                .checked_add(part.resident)
                .and_then(|n| n.checked_add(part.staging))
                .ok_or(Refusal::Overflow("Kimi expert arena"))?);
            resident = resident.checked_add(part.resident)
                .ok_or(Refusal::Overflow("Kimi expert arena"))?;
        }
        Ok(Arena { count, peak: vec![peak], resident: vec![resident], weights: vec![resident - workspace] })
    }).collect()
}

/// Derive the drafter's pool-dependent owners from the admitted layout. The
/// native recipe is linear in whole pages; a non-linear recipe is refused
/// instead of sampled.
pub fn draft_facts(layout: &MlaDraftRuntimeLayout, page_rows: u64, maximum_pages: u64, gpu: u8)
    -> anyhow::Result<Draft> {
    use anyhow::Context;
    let multiple = if layout.context_split() && page_rows == 64 { 2 } else { 1 };
    anyhow::ensure!(maximum_pages >= multiple, "Kimi draft pool cannot hold one physical pair");
    let at = |pages: u64| -> anyhow::Result<(Vec<(u8, String, Category, u64, u64)>, u64)> {
        let tokens = page_rows.checked_mul(pages).context("Kimi draft pool overflow")?;
        let layout = layout.with_pool_tokens(usize::try_from(tokens).context("Kimi draft pool extent")?)?;
        let (demand, peak) = layout.demands(gpu)?;
        Ok((demand.owned.into_iter().map(|d| (d.gpu, d.group, d.category, d.bytes, d.resident_weights.0)).collect(),
            peak.bytes))
    };
    let last_page = maximum_pages / multiple * multiple;
    let (one, peak_one) = at(multiple)?;
    let (last, peak_last) = at(last_page)?;
    let step = |first: u64, end: u64| -> anyhow::Result<u64> {
        if last_page == multiple { return Ok(0); }
        let total = end.checked_sub(first).context("Kimi draft owner shrinks with the pool")?;
        anyhow::ensure!(total % (last_page - multiple) == 0, "Kimi draft owner is not linear in legal pool pages");
        Ok(total / (last_page - multiple))
    };
    if one.len() != last.len() || one.iter().zip(&last).any(|(a, b)| a.0 != b.0 || a.1 != b.1 || a.2 != b.2) {
        return Err(anyhow::anyhow!("Kimi draft owner set depends on the pool extent"));
    }
    // The load peak is the owners plus transient packing; charging the peak on
    // top of the owners would reserve the resident part twice.
    let resident = |owners: &[(u8, String, Category, u64, u64)]| -> anyhow::Result<u64> {
        owners.iter().try_fold(0u64, |sum, owner| sum.checked_add(owner.3)
            .context("Kimi draft owner total overflow"))
    };
    let staging_one = peak_one.checked_sub(resident(&one)?)
        .context("Kimi draft load peak excludes its own owners")?;
    let staging_last = peak_last.checked_sub(resident(&last)?)
        .context("Kimi draft load peak excludes its own owners")?;
    let owners = one.into_iter().zip(last).map(|((gpu, group, category, bytes, weights), (_, _, _, end, end_weights))| {
        let per_page = step(bytes, end)?;
        let weights_per_page = step(weights, end_weights)?;
        Ok(DraftOwner { gpu, category, group,
            bytes_at_first: bytes.checked_sub(per_page * (multiple - 1)).context("Kimi draft owner intercept")?,
            bytes_per_page: per_page,
            weights_at_first: weights.checked_sub(weights_per_page * (multiple - 1)).context("Kimi draft weights intercept")?,
            weights_per_page })
    }).collect::<anyhow::Result<Vec<_>>>()?;
    let staging_per_page = step(staging_one, staging_last)?;
    Ok(Draft { context_split: layout.context_split(), owners,
        staging_at_first: staging_one.checked_sub(staging_per_page * (multiple - 1)).context("Kimi draft staging intercept")?,
        staging_per_page })
}

/// Kimi's physical owners and finite domains. Every field is a resolved fact:
/// attention modes, retention, expert inventory and the transient load peaks.
#[derive(Debug, Clone)]
pub struct KimiRecipe {
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub context: u64,
    pub gpus: usize,
    /// Executor attention placement this recipe was prepared for.
    pub placement: AttentionPlacement,
    /// One resolved mode per layer, in order.
    pub attention: Vec<LayerMode>,
    pub prefill: PrefillShape,
    pub lanes: u64,
    pub spark_links: bool,
    pub hop_spec: HopSpec,
    /// Context exchange bytes per row, each way, under context placement.
    pub peer_row_bytes: u64,
    /// Biggest pool the build can index, in pages.
    pub maximum_pages: u64,
    /// Bytes of retained MLA records one page costs each rank.
    pub records_per_page: Vec<u64>,
    /// Attention weight bytes per rank for each layer.
    pub layer_attention: Vec<ModeBytes>,
    pub first_moe: usize,
    /// Attention classes the checkpoint runs, in layer order.
    pub kinds: Vec<AttentionClass>,
    pub experts: Option<ExpertProfile>,
    pub arenas: Vec<Arena>,
    pub spark: Vec<SparkOwner>,
    pub links: Vec<crate::placement::PreparedSparkLink>,
    pub draft: Option<Draft>,
    pub fixed: Vec<FixedOwner>,
    /// Additional physical owners (context, host retention and loaded code).
    pub owners: Vec<Component>,
}

#[derive(Debug, Clone)]
pub struct AdmittedKimi {
    pub placement: Placement,
    pub solved: ResolvedPlan,
    pub onboard: usize,
}

/// Exact checkpoint geometry; only canonical routed-layer producers own counters.
pub fn usage_geometry(cfg: &crate::families::kimi_k3::config::KimiK3Config, enabled: bool)
    -> Option<cuteafd_core::expert_usage::Geometry> {
    enabled.then(|| cuteafd_core::expert_usage::Geometry { layers: cfg.layers, experts: cfg.experts,
        topk: cfg.topk, routed_layers: cfg.dense.iter().enumerate()
            .filter_map(|(layer, dense)| (!dense).then_some(layer)).collect() })
}

pub fn usage_devices(geometry: &cuteafd_core::expert_usage::Geometry, ranks: usize) -> Vec<u8> {
    (0..ranks.min(geometry.routed_layers.len())).map(|rank| rank as u8).collect()
}

impl KimiRecipe {
    /// Context placement stores half of every page pair on each rank.
    pub fn page_rows(&self) -> u64 {
        if self.attention.iter().any(|mode| *mode == LayerMode::ContextSplit) {
            PAGE_ROWS * 2
        } else {
            PAGE_ROWS
        }
    }

    pub fn admit(&self, request: Request, budgets: Vec<Budget>) -> Result<AdmittedKimi, Refusal> {
        if self.spark_links && self.experts.as_ref().is_some_and(|profile|
            request.onboard == Pin::Exact(profile.layers.len() as u64)) {
            return Err(LOCAL_ONLY_REFUSAL);
        }
        let solved = solve(request, budgets, self)?;
        let placement = self.execution(&solved)?;
        if self.spark_links && self.experts.is_some()
            && !placement.layers.iter().any(|layer| layer.experts == ExpertHome::Spark) {
            return Err(LOCAL_ONLY_REFUSAL);
        }
        let onboard = solved.decisions.onboard_layers.iter().map(Vec::len).max().unwrap_or(0);
        Ok(AdmittedKimi { placement, solved, onboard })
    }

    /// Mechanical executor output. Every byte an executor check names comes
    /// from the admitted owners; nothing is re-sampled here.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Placement, Refusal> {
        let modes = &solved.decisions.attention;
        if !self.attention_profiles().contains(modes)
            || solved.decisions.prefill != self.prefill
            || solved.decisions.lanes != self.lanes
            || solved.decisions.onboard_layers.len() != self.gpus
        {
            return Err(invalid("resolved execution differs from the prepared facts"));
        }
        let count = solved.decisions.onboard_layers.first().map(Vec::len).unwrap_or(0);
        if solved.decisions.onboard_layers.iter().any(|layers| layers.len() != count) {
            return Err(invalid("resolved onboarding is not one count on every rank"));
        }
        let mut local_by_rank = vec![Vec::new(); self.gpus];
        let mut assignments = Vec::with_capacity(modes.len());
        for (layer, &mode) in modes.iter().enumerate() {
            let experts = match solved.decisions.expert_homes.get(&layer) {
                None if self.experts.is_none() || layer < self.first_moe => ExpertHome::Dense,
                None => return Err(invalid("resolved routed layer has no expert home")),
                Some(_) if self.experts.is_none() || layer < self.first_moe =>
                    return Err(invalid("resolved expert home names a dense layer")),
                Some(ExpertDestination::Coordinator { gpus }) => {
                    if self.experts.as_ref().is_none_or(|profile| *gpus != profile.destination) {
                        return Err(invalid("resolved local home differs from prepared destination"));
                    }
                    for &gpu in gpus {
                        local_by_rank[usize::from(gpu)].push(layer);
                    }
                    match gpus.as_slice() {
                        [0, 1] if self.gpus == 2 => ExpertHome::RtxTp2,
                        [gpu] if usize::from(*gpu) < self.gpus => ExpertHome::RtxWhole { gpu: *gpu },
                        _ => return Err(invalid("executor cannot bind the resolved local expert home")),
                    }
                }
                Some(ExpertDestination::Sparks { ranks }) => {
                    if ranks.is_empty() || ranks.len() > 8
                        || ranks.iter().enumerate().any(|(rank, &value)| usize::from(value) != rank)
                    {
                        return Err(invalid("executor requires a Spark rank prefix starting at logical rank 0"));
                    }
                    if self.experts.as_ref().is_none_or(|profile| !profile.rank_sets.contains(ranks)) {
                        return Err(invalid("resolved Spark layers are not a prepared rank set"));
                    }
                    ExpertHome::Spark
                }
            };
            assignments.push(LayerAssignment { mode, experts });
        }
        if solved.decisions.expert_homes.keys().any(|&layer| layer >= modes.len())
            || local_by_rank != solved.decisions.onboard_layers
            || self.experts.as_ref().is_some_and(|profile| !profile.counts.contains(&count))
            || self.arenas.iter().all(|arena| arena.count != count)
        {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let arena = self.arenas.iter().find(|arena| arena.count == count).expect("checked above");
        if arena.peak.len() != self.gpus || arena.resident.len() != self.gpus || arena.weights.len() != self.gpus {
            return Err(invalid("arena rank geometry differs from the coordinator count"));
        }
        let whole = assignments.iter().any(|a| matches!(a.experts, ExpertHome::RtxWhole { .. }));
        let tp2 = (self.gpus == 2 && count > 0 && !whole).then(|| Tp2Range {
            first: self.first_moe,
            layers: count,
            peak_bytes: [arena.peak[0], arena.peak[1]],
        });
        let expert_ranges = (0..self.gpus).map(|rank| {
            if tp2.is_some() || count == 0 || self.gpus != 1 {
                ExpertRange { first: 0, layers: 0, peak_bytes: 0 }
            } else {
                ExpertRange { first: self.first_moe, layers: count, peak_bytes: arena.peak[rank] }
            }
        }).collect();
        let residual = residual_homes(modes, &self.hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let hops = plan_hops(modes, &self.hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let peer_row_bytes = if modes.iter().any(|mode| *mode == LayerMode::ContextSplit) {
            self.peer_row_bytes
        } else {
            0
        };
        let attention_by_kind = self.kind_order().into_iter()
            .map(|kind| (kind, self.placement)).collect();
        Ok(Placement {
            pool_tokens: solved.pool_tokens,
            pool_limits: PoolLimits(vec![]),
            attention_placement: self.placement,
            attention_by_kind,
            peer_row_bytes,
            onboard_layers: count,
            max_pool_tokens_no_onboard: None,
            layers: assignments,
            movables: solved.decisions.draft_home.map(|gpu| (MovableId::Drafter, gpu)).into_iter().collect(),
            expert_ranges,
            residual,
            hops,
            tp2,
            items: (0..self.gpus)
                .map(|rank| solved.items(Location::coordinator(rank as u8)))
                .collect::<Result<_, _>>()?,
        })
    }

    /// Equal count and its nearest two boundaries on either side. The shared
    /// solver prices these correlated choices against every other owner.
    pub fn attention_profiles(&self) -> Vec<Vec<LayerMode>> {
        if self.gpus != 2 || self.placement != AttentionPlacement::Layers {
            return vec![self.attention.clone()];
        }
        let count = self.attention.iter().filter(|mode| matches!(mode, LayerMode::Whole { .. })).count();
        let middle = count.div_ceil(2);
        (middle.saturating_sub(2)..=(middle + 2).min(count)).map(|boundary| {
            let mut index = 0;
            self.attention.iter().map(|&mode| {
                if matches!(mode, LayerMode::Whole { .. }) {
                    let mode = LayerMode::Whole { gpu: u8::from(index >= boundary),
                        ffn: crate::placement::FfnMode::Split };
                    index += 1;
                    mode
                } else { mode }
            }).collect()
        }).collect()
    }

    /// Only preparatory geometry callers use this representative assignment;
    /// execution always consumes the selected profile from the solved decisions.
    pub fn resolved_attention(&self) -> Vec<LayerMode> {
        let profiles = self.attention_profiles();
        let middle = self.attention.iter().filter(|mode| matches!(mode, LayerMode::Whole { .. })).count().div_ceil(2);
        profiles[if self.gpus == 2 && self.placement == AttentionPlacement::Layers { middle.min(2) } else { 0 }].clone()
    }

    /// Graph bytes follow the same finite boundary as weights and KV. Callers
    /// build the executable keys only after admission selects that boundary.
    pub fn balance_graphs(&mut self, cfg: &crate::families::kimi_k3::config::KimiK3Config,
        slots: usize, verify: usize) -> anyhow::Result<()> {
        if self.gpus != 2 || self.placement != AttentionPlacement::Layers { return Ok(()); }
        use crate::families::kimi_k3::allocation;
        self.fixed.retain(|owner| owner.group != allocation::GRAPH_RESERVE);
        for (option, modes) in self.attention_profiles().iter().enumerate() {
            let graphs = allocation::graphs_for_ranks(cfg, self.lanes as usize, self.prefill.rows as usize,
                slots, verify, self.gpus, |rank, layer|
                    !matches!(modes[layer], LayerMode::Whole { gpu, .. } if usize::from(gpu) != rank))?;
            for rank in 0..self.gpus {
                let demand = allocation::graph_demand(&graphs, rank);
                let owner = Allocation::buffer(format!("graphs.{option}.{rank}"), Location::coordinator(rank as u8),
                    demand.bytes, Lifetime::RESIDENT, demand.category, demand.group);
                let mut component = Component::allocation(owner, ByteCurve::fixed(demand.bytes, self.maximum_pages),
                    false, self.maximum_pages);
                component.activation = Activation::choice(0, option);
                self.owners.push(component);
            }
        }
        Ok(())
    }

    fn kind_order(&self) -> Vec<AttentionClass> {
        self.kinds.clone()
    }
}

impl Recipe for KimiRecipe {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if !(1..=2).contains(&self.gpus)
            || self.context == 0
            || request.context != self.context
            || self.attention.is_empty()
            || self.layer_attention.len() != self.attention.len()
            || self.records_per_page.len() != self.gpus
            || self.kinds.is_empty()
            || self.prefill.rows == 0
            || self.prefill.waves == 0
            || self.lanes == 0
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        self.page_rows()
            .checked_mul(self.maximum_pages)
            .ok_or(Refusal::Overflow("Kimi pool extent"))?;
        if self.attention.iter().any(|mode| match mode {
            LayerMode::HeadSplit | LayerMode::ContextSplit => self.gpus != 2,
            LayerMode::Whole { gpu, .. } => usize::from(*gpu) >= self.gpus,
        }) || self.fixed.iter().any(|owner| usize::from(owner.rank) >= self.gpus)
            || self.draft.as_ref().is_some_and(|draft|
                (draft.context_split && self.gpus != 2)
                    || draft.owners.iter().any(|owner| usize::from(owner.gpu) >= self.gpus))
        {
            return Err(invalid("attention ownership or fixed owner rank"));
        }
        let mut contract = Contract { floor_fallback: None,
            resident_weight_limit: request.resident_weight_limit,
            phase_order: PhaseOrder::DRAFT_THEN_TARGET,
            spark_reply_formats: self.spark_links.then(|| SparkReplyFormats {
                prefill: Some(request.spark_reply_formats.prefill(
                    cuteafd_core::ExpertGeometry::KIMI_K3.partial_replies())),
                decode: Some(request.spark_reply_formats.decode()),
            }),
            page_rows: self.page_rows(),
            pool_page_multiple: if self.draft.as_ref().is_some_and(|draft| draft.context_split)
                && self.page_rows() == PAGE_ROWS { 2 } else { 1 },
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: if self.gpus == 2 && self.placement == AttentionPlacement::Layers {
                vec![Variable::AttentionProfiles { layers: (0..self.attention.len()).collect(),
                    values: self.attention_profiles() }]
            } else {
                self.attention.iter().enumerate().map(|(layer, &mode)|
                    Variable::Attention { layer, values: vec![mode] }).collect()
            },
            requires: vec![],
            components: vec![],
        };
        let profiles = self.attention_profiles();
        let modes = self.resolved_attention();
        let profiled = self.gpus == 2 && self.placement == AttentionPlacement::Layers;
        let whole = modes.iter().enumerate().filter_map(|(layer, mode)|
            (profiled && matches!(mode, LayerMode::Whole { .. })).then_some(layer)).collect::<Vec<_>>();
        contract.components.extend(self.owners.clone());
        if let Some(geometry) = &self.expert_usage {
            contract.components.extend(crate::placement::expert_usage::components(
                geometry, &usage_devices(geometry, self.gpus), self.maximum_pages)?);
        }
        // Intake GPU planes are already fixed owners. Only the transport's
        // future host/pinned storage joins the shared physical RAM budget here.
        for link in &self.links {
            contract.components.extend(link.components(self.maximum_pages)?.into_iter().filter(|component|
                matches!(&component.kind, ComponentKind::Allowance(owner)
                    if matches!(owner.location, Location::HOST | Location::PINNED))));
        }
        contract.variables.push(Variable::Prefill(vec![self.prefill]));
        contract.variables.push(Variable::Lanes(vec![self.lanes]));
        let onboard = self.experts.as_ref().map(|profile| -> Result<(usize, &ExpertProfile), Refusal> {
            if profile.layers.is_empty()
                || profile.layers != (self.first_moe..self.attention.len()).collect::<Vec<_>>()
                || profile.destination.is_empty()
                || profile.destination.iter().any(|&gpu| usize::from(gpu) >= self.gpus)
                || profile.destination.windows(2).any(|pair| pair[0] >= pair[1])
                || profile.counts.is_empty()
                || profile.counts.iter().any(|&count| count > profile.layers.len())
                || profile.counts.windows(2).any(|pair| pair[0] >= pair[1])
                || profile.rank_sets.iter().enumerate().any(|(index, ranks)| {
                    ranks.is_empty() || ranks.len() > 8
                        || ranks.iter().enumerate().any(|(rank, &value)| usize::from(value) != rank)
                        || profile.rank_sets[..index].contains(ranks)
                })
            {
                return Err(invalid("invalid expert profile"));
            }
            let variable = contract.variables.len();
            contract.variables.push(Variable::Onboard {
                layers: profile.layers.clone(),
                destination: profile.destination.clone(),
                counts: profile.counts.clone(),
            });
            Ok((variable, profile))
        }).transpose()?;
        if let Some((variable, profile)) = onboard {
            // Arenas and rank sets are prepared from exported executors. Exclude
            // unavailable shapes before optimizing, not after choosing a winner.
            for (option, &count) in profile.counts.iter().enumerate() {
                let local = count == 0 || self.arenas.iter().any(|arena| arena.count == count);
                let remote = count == profile.layers.len() || !profile.rank_sets.is_empty();
                let executor = !self.spark_links || count < profile.layers.len();
                if !local || !remote || !executor {
                    contract.requires.push(Requires { when: Choice { variable, option }, any_of: vec![] });
                }
            }
            if !profile.rank_sets.is_empty() {
                contract.variables.push(Variable::ExpertHome {
                    layers: profile.layers.clone(),
                    rank_sets: profile.rank_sets.clone(),
                });
            }
        }
        let option = |count: usize| -> Result<Choice, Refusal> {
            let (variable, profile) = onboard.ok_or_else(|| invalid("arena without an expert profile"))?;
            let option = profile.counts.iter().position(|&value| value == count)
                .ok_or_else(|| invalid("arena count is not in the declared profile"))?;
            Ok(Choice { variable, option })
        };
        for (rank, &bytes) in self.records_per_page.iter().enumerate() {
            if bytes == 0 || !whole.is_empty() { continue; }
            let owner = Allocation::buffer(format!("records.{rank}"),
                Location::coordinator(rank as u8), self.page_rows(), Lifetime::RESIDENT,
                Category::Kv, "records");
            contract.components.push(Component::allocation(owner,
                ByteCurve::linear(bytes, bytes, self.maximum_pages), false, self.maximum_pages));
        }
        for fixed in &self.fixed {
            if fixed.bytes == 0 { continue; }
            let owner = Allocation::buffer(format!("{}.{}", fixed.group, fixed.rank),
                Location::coordinator(fixed.rank),
                fixed.bytes, fixed.lifetime, fixed.category, fixed.group.clone());
            if fixed.weights > 0 {
                if fixed.weights > fixed.bytes {
                    return Err(invalid("retained weights exceed their allocation"));
                }
                contract.components.push(Component::resident_weight(
                    ResidentWeight::from_allocation(&owner, ResidentWeights(fixed.weights)),
                    ByteCurve::fixed(fixed.weights, self.maximum_pages)));
            }
            contract.components.push(Component::allocation(owner,
                ByteCurve::fixed(fixed.bytes, self.maximum_pages), false, self.maximum_pages));
        }
        for layer in 0..modes.len() {
            let domain = if profiled { profiles.iter().map(|modes| modes[layer]).collect::<Vec<_>>() }
                else { vec![modes[layer]] };
            for (option, mode) in domain.into_iter().enumerate() {
                let activation = Activation::choice(if profiled { 0 } else { layer }, option);
                for rank in 0..self.gpus {
                    let weights = attention_bytes(&self.layer_attention[layer], mode, rank);
                    if weights == 0 { continue; }
                    let owner = Allocation::buffer(format!("layers.{layer}.{option}.{rank}"),
                        Location::coordinator(rank as u8), weights, Lifetime::RESIDENT, Category::Weights, "layers");
                    let mut weight = Component::resident_weight(
                        ResidentWeight::from_allocation(&owner, ResidentWeights(weights)),
                        ByteCurve::fixed(weights, self.maximum_pages));
                    weight.activation = activation.clone();
                    contract.components.push(weight);
                    let mut allocation = Component::allocation(owner,
                        ByteCurve::fixed(weights, self.maximum_pages), false, self.maximum_pages);
                    allocation.activation = activation.clone();
                    contract.components.push(allocation);
                }
                if whole.contains(&layer) {
                    let LayerMode::Whole { gpu, .. } = mode else { unreachable!() };
                    let bytes = self.records_per_page[0];
                    let owner = Allocation::buffer(format!("records.{layer}.{option}"),
                        Location::coordinator(gpu), bytes, Lifetime::RESIDENT, Category::Kv, "records");
                    let mut allocation = Component::allocation(owner,
                        ByteCurve::linear(bytes, bytes, self.maximum_pages), false, self.maximum_pages);
                    allocation.activation = activation;
                    contract.components.push(allocation);
                }
            }
        }
        if self.hop_spec.row_bytes > 0 {
            for (option, modes) in profiles.iter().enumerate() {
                let hops = plan_hops(modes, &self.hop_spec).map_err(|error| invalid(&error.to_string()))?;
                let buffers = hop_buffer_bytes(&hops, &self.hop_spec, self.gpus)
                    .ok_or(Refusal::Overflow("Kimi residual hop buffers"))?;
                for (rank, &bytes) in buffers.iter().enumerate() {
                    if bytes == 0 { continue; }
                    let owner = Allocation::buffer(format!("residual hops.{option}.{rank}"),
                        Location::coordinator(rank as u8), bytes, Lifetime::RESIDENT,
                        Category::Transport, "residual hops");
                    let mut component = Component::allocation(owner,
                        ByteCurve::fixed(bytes, self.maximum_pages), false, self.maximum_pages);
                    if profiled { component.activation = Activation::choice(0, option); }
                    contract.components.push(component);
                }
            }
        }
        for arena in &self.arenas {
            if self.experts.is_none() && arena.count == 0 { continue; }
            let choice = option(arena.count)?;
            if arena.peak.len() != self.gpus || arena.resident.len() != self.gpus || arena.weights.len() != self.gpus {
                return Err(invalid("arena rank geometry differs from the coordinator count"));
            }
            let group = match arena.count {
                0 => continue,
                count if self.gpus == 2 => format!("TP2 expert layer halves {first}..{end}",
                    first = self.first_moe, end = self.first_moe + count),
                count => format!("resident routed layers {:?}",
                    &self.experts.as_ref().map(|profile| profile.layers[..count].to_vec()).unwrap_or_default()),
            };
            for rank in 0..self.gpus {
                let peak = arena.peak[rank];
                if peak == 0 { continue; }
                let owner = Allocation::buffer(format!("experts.{}.{rank}", arena.count),
                    Location::coordinator(rank as u8), arena.resident[rank], Lifetime::RESIDENT,
                    Category::Experts, group.clone());
                if arena.weights[rank] > 0 {
                    if arena.weights[rank] > peak {
                        return Err(invalid("resident experts exceed their arena peak"));
                    }
                    let mut weights = Component::resident_weight(
                        ResidentWeight::from_allocation(&owner, ResidentWeights(arena.weights[rank])),
                        ByteCurve::fixed(arena.weights[rank], self.maximum_pages));
                    weights.activation = Activation::choice(choice.variable, choice.option);
                    contract.components.push(weights);
                }
                let mut component = Component::allocation(owner,
                    ByteCurve::fixed(arena.resident[rank], self.maximum_pages), false, self.maximum_pages);
                component.activation = Activation::choice(choice.variable, choice.option);
                contract.components.push(component);
                let staging = peak.checked_sub(arena.resident[rank]).ok_or_else(|| invalid("expert peak below resident"))?;
                if staging > 0 {
                    let owner = Allocation::buffer(format!("experts.staging.{}.{rank}", arena.count),
                        Location::coordinator(rank as u8), staging,
                        Lifetime { first: Phase::Load, last: Phase::Load }, Category::Experts, "expert load staging");
                    let mut component = Component::allocation(owner, ByteCurve::fixed(staging, self.maximum_pages), false, self.maximum_pages);
                    component.activation = Activation::choice(choice.variable, choice.option);
                    contract.components.push(component);
                }
            }
        }
        for spark in &self.spark {
            let activation = if self.experts.is_none() && spark.count == 0 { Activation::always() }
                else { let choice = option(spark.count)?; Activation::choice(choice.variable, choice.option) };
            for (owner_kind, group, bytes, category) in [
                ("experts", "routed experts (balanced H128, replicated 3584 rotations)", spark.experts, Category::Experts),
                ("overhead", "Spark rank I/O and execution owners", spark.overhead, Category::Workspace),
                ("startup", cuteafd_core::memory_layout::spark_runtime_reserve_group(), spark.runtime, Category::Runtime),
                ("arena-growth", cuteafd_core::memory_layout::SPARK_ARENA_DRIVER_OVERHEAD_GROUP,
                    cuteafd_core::memory_layout::spark_arena_driver_overhead_bytes(spark.experts), Category::Workspace),
            ] {
                if bytes == 0 { continue; }
                let owner = Allocation::buffer(format!("spark.{}.{owner_kind}.{}", spark.rank, spark.count),
                    Location { device: Device::Spark(spark.rank as u8), domain: Domain::Managed },
                    bytes, Lifetime::RESIDENT, category, group);
                let mut component = Component::allocation(owner,
                    ByteCurve::fixed(bytes, self.maximum_pages), false, self.maximum_pages);
                component.activation = activation.clone();
                contract.components.push(component);
            }
        }
        if let Some(draft) = &self.draft {
            let rank = (self.gpus - 1) as u8;
            contract.variables.push(Variable::DraftHome(vec![rank]));
            let lifetime = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            for (index, owner) in draft.owners.iter().enumerate() {
                let allocation = Allocation::buffer(format!("draft.{index}"),
                    Location::coordinator(owner.gpu), owner.bytes_at_first, lifetime,
                    owner.category, owner.group.clone());
                if owner.weights_at_first > 0 || owner.weights_per_page > 0 {
                    contract.components.push(Component::resident_weight(
                        ResidentWeight::from_allocation(&allocation, ResidentWeights(owner.weights_at_first)),
                        ByteCurve::linear(owner.weights_at_first, owner.weights_per_page, self.maximum_pages)));
                }
                contract.components.push(Component::allocation(allocation,
                    ByteCurve::linear(owner.bytes_at_first, owner.bytes_per_page, self.maximum_pages),
                    false, self.maximum_pages));
            }
            // The packing peak overlaps the target weights and the drafter's
            // own resident owners; only the staging above them is transient.
            if draft.staging_at_first > 0 || draft.staging_per_page > 0 {
                let peak = Allocation::buffer("draft.load_peak", Location::coordinator(rank),
                    draft.staging_at_first, Lifetime { first: Phase::DraftLoad, last: Phase::DraftLoad },
                    Category::Drafter, "MLA DFlash load staging above its owners");
                contract.components.push(Component::allocation(peak,
                    ByteCurve::linear(draft.staging_at_first, draft.staging_per_page, self.maximum_pages),
                    false, self.maximum_pages));
            }
        }
        Ok(contract)
    }
}

/// Physical capacities and operator ceilings are independent inputs. `occupied`
/// is what the process already holds at `Load` (the CUDA context), not a second
/// subtraction of owners the recipe charges.
pub fn budgets(cards: &[(u64, u64)], headroom: u64, occupied: &[u64],
    sparks: &[u64]) -> Result<Vec<Budget>, Refusal> {
    if !(1..=2).contains(&cards.len()) || occupied.len() != cards.len() {
        return Err(invalid("invalid physical capacities"));
    }
    let forecast = |bytes: u64| Baseline::Forecast {
        occupied: bytes,
        source: "measured CUDA context; every Kimi owner charged in the recipe".into(),
        assumed: true,
    };
    let mut budgets = cards.iter().enumerate().map(|(rank, &(physical, logical))| Budget {
        id: format!("rtx{rank}"),
        locations: vec![Location::coordinator(rank as u8)],
        physical_bytes: physical,
        logical_bytes: logical,
        baseline: forecast(occupied[rank]),
        headroom,
    }).collect::<Vec<_>>();
    budgets.extend(sparks.iter().enumerate().map(|(rank, &bytes)| Budget {
        id: format!("spark{rank}"),
        locations: vec![Location { device: Device::Spark(rank as u8), domain: Domain::Managed }],
        physical_bytes: bytes,
        logical_bytes: bytes,
        baseline: forecast(0),
        headroom: 0,
    }));
    Ok(budgets)
}

/// HOST and PINNED are one physical capacity, even when retention uses ordinary RAM.
pub fn host_budget(total: u64, available: u64) -> Budget {
    Budget { id: "host".into(), locations: vec![Location::HOST, Location::PINNED],
        physical_bytes: total, logical_bytes: total,
        baseline: Baseline::Forecast { occupied: total.saturating_sub(available),
            source: "host MemAvailable before retained owners".into(), assumed: true },
        headroom: (8 << 30).max(total / 10) }
}

/// Estimated retained upload storage: #1329 K6 max smoke observed 39,656,288 B
/// in staging/sync-h2d after loading. This is independent of the link receipts.
pub const HOST_UPLOAD_BYTES: u64 = 64 << 20;

pub fn host_upload_owner(maximum: u64) -> Component {
    Component { activation: Activation::always(), bytes: ByteCurve::fixed(HOST_UPLOAD_BYTES, maximum),
        kind: ComponentKind::Allowance(Allowance { location: Location::PINNED,
            bytes: HOST_UPLOAD_BYTES, lifetime: Lifetime::RESIDENT,
            source: "Kimi reusable H2D staging (estimated): #1329 K6 max smoke 2026-11-06 measured 39,656,288 B".into(),
            assumed: true }) }
}

pub fn host_scoring_bytes(decode_rows: usize, prefill_rows: usize, vocab: usize) -> Result<u64, Refusal> {
    cuteafd_core::memory_layout::host_logits_bytes(decode_rows as u64, prefill_rows as u64, vocab as u64)
        .ok_or(Refusal::Overflow("Kimi host scoring logits"))
}

pub fn host_scoring_owner(bytes: u64, maximum: u64) -> Component {
    Component::allocation(
        Allocation::buffer("kimi.host.scoring", Location::HOST, bytes,
            Lifetime { first: Phase::Ready, last: Phase::Growth }, Category::Staging, "Kimi host scoring logits"),
        ByteCurve::fixed(bytes, maximum), true, maximum,
    )
}

pub fn host_prefix_owner(bytes: u64, maximum: u64) -> Component {
    Component::allocation(Allocation::buffer("kimi.host.prefix", Location::HOST, bytes,
        Lifetime::RESIDENT, Category::Prefix, "Kimi retained host prefix quota"),
        ByteCurve::fixed(bytes, maximum), true, maximum)
}

/// Zero and omission normalize identically; an explicit pin stays exact.
pub fn request(context: u64, pool: Option<u64>, onboard: crate::placement::Onboard,
    prefill: PrefillShape, operation: Operation, weight_limit: ResidentWeights,
    spark_reply_formats: SparkReplyFormats) -> Request {
    Request {
        context,
        pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard {
            crate::placement::Onboard::Auto => Pin::Auto,
            crate::placement::Onboard::Layers(count) => Pin::Exact(count as u64),
        },
        prefill: Pin::Exact(prefill),
        operation,
        resident_weight_limit: Some(weight_limit),
        spark_reply_formats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::FfnMode;

    #[test]
    fn expert_usage_recipe_charges_exact_canonical_device_and_host_owners() {
        use cuteafd_core::expert_usage::{device_owner, Geometry};
        for ranks in [1, 2] {
            let mut recipe = recipe();
            recipe.gpus = ranks;
            recipe.records_per_page = vec![64; ranks];
            for arena in &mut recipe.arenas {
                arena.peak.resize(ranks, 0);
                arena.resident.resize(ranks, 0);
                arena.weights.resize(ranks, 0);
            }
            let off = recipe.prepare(&single()).unwrap();
            assert!(!off.components.iter().any(|component| matches!(&component.kind, ComponentKind::Allocation { template, .. } if template.id.starts_with("expert-usage."))));
            let geometry = Geometry { layers: 5, experts: 896, topk: 16, routed_layers: vec![1, 2, 3, 4] };
            // The ownership split is by routed-layer ordinal, not global layer parity.
            assert_eq!(usage_devices(&geometry, ranks), (0..ranks as u8).collect::<Vec<_>>());
            recipe.expert_usage = Some(geometry.clone());
            let on = recipe.prepare(&single()).unwrap();
            let owners = on.components.iter().filter(|component| matches!(&component.kind, ComponentKind::Allocation { template, .. } if template.id.starts_with("expert-usage.")))
                .collect::<Vec<_>>();
            assert_eq!(owners.len(), ranks * 3 + 1);
            for rank in 0..ranks {
                for suffix in ["counters", "snapshot", "pinned"] {
                    let component = owners.iter().find(|owner| matches!(&owner.kind, ComponentKind::Allocation { template, .. } if template.id == device_owner(rank as u8, suffix))).unwrap();
                    assert_eq!(component.bytes.at(1).unwrap(), geometry.table_bytes().unwrap() as u64);
                    assert_eq!(component.bytes.at(recipe.maximum_pages).unwrap(), geometry.table_bytes().unwrap() as u64);
                }
            }
            let history = owners.iter().find(|owner| matches!(&owner.kind, ComponentKind::Allocation { template, .. } if template.id == "expert-usage.history")).unwrap();
            assert_eq!(history.bytes.at(1).unwrap(), geometry.host_bytes(ranks).unwrap() as u64);
        }
    }

    fn single() -> Request {
        request_with(Some(0))
    }

    fn request_with(pool: Option<u64>) -> Request {
        crate::placement::families::kimi::request(256, pool, crate::placement::Onboard::Auto,
            PrefillShape { rows: 64, waves: 1 },
            crate::placement::planner::Operation::Serving, ResidentWeights(1 << 30),
            SparkReplyFormats::default())
    }

    fn recipe() -> KimiRecipe {
        KimiRecipe {
            expert_usage: None,
            context: 256,
            gpus: 1,
            placement: AttentionPlacement::Heads,
            attention: vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }; 2],
            prefill: PrefillShape { rows: 64, waves: 1 },
            lanes: 1,
            spark_links: false,
            hop_spec: HopSpec { storage: crate::placement::HopStorage::ReceiveSlots, row_bytes: 0,
                rows: 0, lanes: 0, entry_gpu: 0, head_gpu: 0,
                terminal: crate::placement::TerminalHome::AttentionOwner },
            peer_row_bytes: 0,
            maximum_pages: 4,
            records_per_page: vec![64],
            layer_attention: vec![ModeBytes { whole: 100, split: [0; 2] }; 2],
            first_moe: 1,
            kinds: vec![AttentionClass::Kda],
            experts: Some(ExpertProfile { layers: vec![1], destination: vec![0],
                counts: vec![0, 1], rank_sets: vec![] }),
            arenas: vec![
                Arena { count: 0, peak: vec![0], resident: vec![0], weights: vec![0] },
                Arena { count: 1, peak: vec![1000], resident: vec![1000], weights: vec![900] },
            ],
            spark: vec![],
            links: vec![],
            draft: None,
            owners: vec![],
            fixed: vec![FixedOwner { rank: 0, category: Category::Prefix, group: "Kimi exact KDA marks".into(),
                bytes: 32, weights: 0, lifetime: Lifetime::RESIDENT }],
        }
    }

    #[test]
    fn push_host_load_peak_is_admitted_without_reducing_ready_retention() {
        let cfg = crate::families::kimi_k3::config::KimiK3Config::from_hf(
            &crate::plan::testing::kimi_k3_config()).unwrap();
        let envelope = push_host_envelope(&cfg, 1).unwrap();
        assert_eq!(envelope.dense_bytes, vec![10 << 30]);
        assert_eq!(envelope.expert_bytes[0], cfg.dense.iter().filter(|dense| !**dense).count() as u64
            * cfg.experts as u64 * 12);
        let mut recipe = recipe();
        let owners = push_host_owners(&envelope, recipe.maximum_pages).unwrap();
        assert!(owners.iter().any(|component| matches!(&component.kind,
            ComponentKind::Allowance(owner) if owner.assumed && owner.lifetime.last == Phase::Load)));
        recipe.owners.extend(owners);
        recipe.owners.push(host_prefix_owner(1024, recipe.maximum_pages));
        let mut budgets = budgets(&[(1 << 30, 1 << 30)], 0, &[0], &[]).unwrap();
        let mut host = host_budget(32 << 30, 32 << 30);
        host.headroom = 0;
        host.logical_bytes = envelope.bytes().unwrap() + 1024;
        budgets.push(host);
        let admitted = recipe.admit(single(), budgets.clone()).unwrap();
        assert_eq!(admitted.solved.remaining("host", Phase::Load), Some(0));
        assert_eq!(admitted.solved.remaining("host", Phase::Ready), Some(envelope.bytes().unwrap()));
        budgets.last_mut().unwrap().logical_bytes -= 1;
        let refusal = recipe.admit(single(), budgets).unwrap_err();
        let limiting = match &refusal {
            Refusal::FullContextShortfall { limiting, .. } => limiting.as_ref(),
            other => other,
        };
        assert!(matches!(limiting, Refusal::Capacity { phase: Phase::Load, .. }), "{refusal:?}");
        let mut actual = envelope.clone();
        actual.dense_bytes[0] += 1;
        assert!(matches!(envelope.validate_usage(&actual), Err(Refusal::Capacity { .. })));
    }

    #[test]
    fn transport_host_receipts_share_capacity_without_duplicate_device_intake() {
        let mut recipe = recipe();
        let rings = cuteafd_core::SparkRings { max_frame_bytes: 64 << 20, depth: 8,
            slot_bytes: 8 << 20, alignment: 4096 };
        let links = production_links(0, 6, 1, PrefillShape { rows: 2752, waves: 3 },
            rings, SparkReplyFormats::default()).unwrap();
        let host_bytes = links[0].envelope().iter().filter(|(location, _)|
            matches!(**location, Location::HOST | Location::PINNED)).map(|(_, bytes)| *bytes).sum::<u64>();
        assert!(host_bytes >= 1_466_499_072, "host envelope: {host_bytes}");
        let before = recipe.prepare(&single()).unwrap();
        recipe.links = links;
        let contract = recipe.prepare(&single()).unwrap();
        let device_allowances = |contract: &Contract| contract.components.iter().filter(|component|
            matches!(&component.kind, ComponentKind::Allowance(owner)
                if owner.location == Location::coordinator(0))).count();
        assert_eq!(device_allowances(&contract), device_allowances(&before));
        let mut budgets = budgets(&[(1 << 30, 1 << 30)], 0, &[0], &[]).unwrap();
        let mut host = host_budget(4 << 30, 4 << 30);
        host.headroom = 0;
        host.logical_bytes = host_bytes;
        budgets.push(host);
        let admitted = recipe.admit(single(), budgets.clone()).unwrap();
        assert_eq!(admitted.solved.remaining("host", Phase::Ready), Some(0));
        budgets.last_mut().unwrap().logical_bytes -= 1;
        assert!(recipe.admit(single(), budgets).is_err());
        assert!(production_links(0, 0, 1, recipe.prefill, rings, SparkReplyFormats::default()).unwrap().is_empty());
    }

    #[test]
    fn host_retention_and_scoring_share_one_physical_budget() {
        let mut recipe = recipe();
        let scoring = host_scoring_bytes(64, 1024, 100).unwrap();
        assert_eq!(scoring, 64 * 100 * 4);
        recipe.owners.push(host_prefix_owner(1024, recipe.maximum_pages));
        recipe.owners.push(host_scoring_owner(scoring, recipe.maximum_pages));
        let mut budgets = budgets(&[(1 << 30, 1 << 30)], 0, &[0], &[]).unwrap();
        let mut host = host_budget(1 << 30, 1 << 30);
        host.headroom = 0;
        host.logical_bytes = 1024 + scoring;
        budgets.push(host);
        let admitted = recipe.admit(single(), budgets.clone()).unwrap();
        assert_eq!(admitted.solved.remaining("host", Phase::Ready), Some(0));
        assert_eq!(admitted.solved.remaining("host", Phase::Load), Some(scoring));
        budgets.last_mut().unwrap().logical_bytes -= 1;
        assert!(recipe.admit(single(), budgets).is_err());
    }

    #[test]
    fn zero_pool_pin_normalizes_like_omission() {
        assert_eq!(request_with(Some(0)).pool, request_with(None).pool);
        assert_ne!(request_with(Some(256)).pool, request_with(None).pool);
    }

    #[test]
    fn context_draft_preserves_physical_homes_and_requires_whole_page_pairs() {
        let mut recipe = recipe();
        recipe.gpus = 2;
        recipe.hop_spec.terminal = crate::placement::TerminalHome::Replicated;
        recipe.records_per_page = vec![64; 2];
        recipe.attention = vec![LayerMode::HeadSplit; 2];
        recipe.layer_attention = vec![ModeBytes { whole: 100, split: [50; 2] }; 2];
        recipe.experts = None;
        recipe.arenas = vec![Arena { count: 0, peak: vec![0; 2], resident: vec![0; 2], weights: vec![0; 2] }];
        recipe.draft = Some(Draft { context_split: true, staging_at_first: 100,
            staging_per_page: 0, owners: (0..2).map(|gpu| DraftOwner { gpu,
                category: Category::Drafter, group: "physical draft cache".into(),
                bytes_at_first: 32, bytes_per_page: 32, weights_at_first: 0,
                weights_per_page: 0 }).collect() });
        let budget = budgets(&[(1 << 30, 1 << 30); 2], 0, &[0; 2], &[]).unwrap();
        let contract = recipe.prepare(&single()).unwrap();
        assert_eq!(contract.pool_page_multiple, 2);
        let admitted = recipe.admit(single(), budget.clone()).unwrap();
        for rank in 0..2 {
            let owners: Vec<_> = admitted.placement.items[rank].iter()
                .filter(|owner| owner.group == "physical draft cache").collect();
            assert_eq!(owners.len(), 1);
            assert_eq!(owners[0].bytes, 128);
        }
        let request = |pool| Request { context: 128, ..request_with(Some(pool)) };
        recipe.context = 128;
        assert!(recipe.admit(request(192), budget.clone()).is_err());
        assert_eq!(recipe.admit(request(128), budget).unwrap().solved.pool_tokens, 128);
    }

    #[test]
    fn pool_domain_covers_every_page_and_only_the_selected_onboard_count_charges() {
        let recipe = recipe();
        let contract = recipe.prepare(&single()).unwrap();
        // Two attention layers, prefill, lanes, and the onboard count profile.
        assert_eq!(contract.variables.len(), 5);
        for pages in 1..=recipe.maximum_pages {
            let idle = contract.facts(&[0, 0, 0, 0, 0], pages).unwrap();
            let records = idle.allocations.iter().find(|owner| owner.group == "records").unwrap();
            assert_eq!(records.bytes, pages * 64);
            let weights: u64 = idle.resident_weights.iter().map(|owner| owner.weights.0).sum();
            assert_eq!(weights, 200);
            assert!(!idle.allocations.iter().any(|owner| owner.category == Category::Experts));
            let onboarded = contract.facts(&[0, 0, 0, 0, 1], pages).unwrap();
            let arena: Vec<_> = onboarded.allocations.iter()
                .filter(|owner| owner.category == Category::Experts).collect();
            assert_eq!(arena.len(), 1);
            assert_eq!(arena[0].bytes, 1000);
            assert_eq!(onboarded.resident_weights.iter().map(|owner| owner.weights.0).sum::<u64>(), 1100);
        }
    }

    #[test]
    fn admission_resolves_the_declared_prefix_and_refuses_a_pin_above_it() {
        let recipe = recipe();
        let budgets = budgets(&[(1 << 30, 1 << 30)], 0, &[0], &[]).unwrap();
        let admitted = recipe.admit(single(), budgets.clone()).unwrap();
        assert_eq!(admitted.solved.pool_tokens, 256);
        assert_eq!(admitted.placement.attention_placement, AttentionPlacement::Heads);
        assert_eq!(admitted.placement.layers.len(), 2);
        assert_eq!(admitted.onboard,
            admitted.solved.decisions.onboard_layers.iter().map(Vec::len).max().unwrap_or(0));
        let pinned = crate::placement::families::kimi::request(256, Some(320), crate::placement::Onboard::Auto,
            PrefillShape { rows: 64, waves: 1 }, crate::placement::planner::Operation::Serving,
            ResidentWeights(1 << 30), SparkReplyFormats::default());
        assert!(recipe.admit(pinned, budgets).is_err());
    }

    #[test]
    fn linked_executor_refuses_all_local_but_pure_solver_still_admits() {
        let mut recipe = recipe();
        let budgets = budgets(&[(1 << 30, 1 << 30)], 0, &[0], &[]).unwrap();
        let admitted = recipe.admit(single(), budgets.clone()).unwrap();
        assert!(!admitted.placement.layers.iter().any(|layer| layer.experts == ExpertHome::Spark));
        recipe.spark_links = true;
        let mut request = single();
        request.onboard = Pin::Exact(1);
        assert!(matches!(recipe.admit(request, budgets), Err(Refusal::MissingExecutor { issue: 1395, .. })));
    }

    #[test]
    fn linked_search_selects_the_only_feasible_partial_executor() {
        let mut recipe = recipe();
        recipe.spark_links = true;
        recipe.attention.push(recipe.attention[0]);
        recipe.layer_attention.push(recipe.layer_attention[0].clone());
        recipe.experts = Some(ExpertProfile { layers: vec![1, 2], destination: vec![0],
            counts: vec![0, 1, 2], rank_sets: vec![vec![0]] });
        recipe.arenas.push(Arena { count: 2, peak: vec![2000], resident: vec![2000], weights: vec![1800] });
        recipe.spark = vec![
            SparkOwner { count: 0, rank: 0, experts: 200, overhead: 0, runtime: 0 },
            SparkOwner { count: 1, rank: 0, experts: 100, overhead: 0, runtime: 0 },
        ];
        let budgets = budgets(&[(1 << 30, 1 << 30)], 0, &[0], &[150]).unwrap();
        let admitted = recipe.admit(single(), budgets).unwrap();
        assert_eq!(admitted.onboard, 1);
        assert_eq!(admitted.placement.layers[1].experts, ExpertHome::RtxWhole { gpu: 0 });
        assert_eq!(admitted.placement.layers[2].experts, ExpertHome::Spark);
    }

    #[test]
    fn shipped_max_layers_gap_is_no_worse_than_best_single_layer_move() {
        let mut recipe = recipe();
        recipe.gpus = 2;
        recipe.hop_spec.terminal = crate::placement::TerminalHome::Replicated;
        recipe.placement = AttentionPlacement::Layers;
        recipe.context = 2_097_152;
        recipe.maximum_pages = recipe.context / PAGE_ROWS;
        // v5 readiness replay, K2 experts / K6 core, FP8 KV, 2M pool.
        let per_layer = 1_566_104_832;
        recipe.attention = vec![LayerMode::Whole { gpu: 0, ffn: crate::placement::FfnMode::Split }; 24];
        recipe.layer_attention = vec![ModeBytes { whole: per_layer - 64 * recipe.maximum_pages,
            split: [0; 2] }; 24];
        recipe.records_per_page = vec![64; 2];
        recipe.experts = None;
        recipe.arenas = vec![Arena { count: 0, peak: vec![0; 2], resident: vec![0; 2], weights: vec![0; 2] }];
        recipe.fixed = [96_382_880_182, 100_432_126_894].into_iter().enumerate().map(|(rank, bytes)|
            FixedOwner { rank: rank as u8, category: Category::Runtime, group: "replayed max owners".into(),
                bytes: bytes - 12 * per_layer, weights: 0, lifetime: Lifetime::RESIDENT }).collect();
        let request = request(recipe.context, Some(recipe.context), crate::placement::Onboard::Auto,
            recipe.prefill, Operation::Serving, ResidentWeights(u64::MAX), SparkReplyFormats::default());
        let started = std::time::Instant::now();
        let admitted = recipe.admit(request, budgets(&[(102 << 30, 102 << 30); 2], 2 << 30, &[0; 2], &[]).unwrap()).unwrap();
        assert!(started.elapsed() < std::time::Duration::from_millis(100));
        let totals = admitted.placement.items.iter().map(|items| items.iter().map(|i| i.bytes).sum::<u64>()).collect::<Vec<_>>();
        let best_move_gap = (96_382_880_182u64 + per_layer).abs_diff(100_432_126_894 - per_layer);
        assert!(totals[0].abs_diff(totals[1]) <= best_move_gap, "{totals:?}");
        assert_eq!(admitted.placement.layers.iter().filter(|layer|
            matches!(layer.mode, LayerMode::Whole { gpu: 0, .. })).count(), 13);
        assert_eq!(admitted.placement.layers.iter().map(|layer| layer.mode).collect::<Vec<_>>(),
            admitted.solved.decisions.attention);
    }

    #[test]
    fn a_missing_physical_budget_is_refused_before_any_fit() {
        let recipe = recipe();
        assert!(recipe.admit(single(), Vec::new()).is_err());
        assert!(recipe.prepare(&Request { context: 512, ..single() }).is_err());
    }
}
