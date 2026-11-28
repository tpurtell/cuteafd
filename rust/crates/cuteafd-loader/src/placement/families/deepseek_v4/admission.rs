//! V4's physical component contract: the cache geometry, workspace formulas,
//! exchange slots and expert arenas measured by `fixed_demands`, composed once
//! into a [`Contract`] the shared solver prices. Plan and serve build the same
//! [`V4Inputs`], so an admitted layout is identical on both sides.
//!
//! The experts' ordered arenas are the one structural fact this file adds: a
//! whole-layer prefix (or a TP2 half prefix) loads after dSpark's stages into
//! the same arena, so its peak is the running resident total plus the largest
//! staging step, never the sum of every step.
use super::{context_buffers, fixed_demands, steps_requests, V4Inputs};
use crate::placement::planner::*;
use crate::placement::{
    hop_buffer_bytes, plan_hops, residual_homes, AttentionClass, AttentionPlacement, ExpertHome,
    ExpertRange, FfnMode, GraphRank, HopSpec, HopStorage, LayerAssignment, LayerMode, MovableId,
    Onboard, Placement, PoolLimits, ResidentWeights, TerminalHome, Tp2Range,
};
use crate::serving_capacity::V4ExpertCost;
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_core::spark_reply::SparkReplyFormats;
use std::collections::BTreeSet;

/// V4's compressed-record geometry: one pool unit is 256 tokens of every
/// layer's records. A context split prices pairs so both head halves own whole
/// units, exactly as the legacy even-unit rule did.
const V4_PAGE_ROWS: u64 = 256;
/// Symbolic physical extent of a Spark rank. V4's planner names the remote
/// expert destination and never charges Spark memory; Spark admission owns
/// that budget, so the rank exists here only to satisfy the location rules.
const SPARK_UNPRICED_BYTES: u64 = 1 << 50;
/// Steady owners exist after the target KV is admitted; sequential dense
/// loading is Load-only and is judged alone, exactly as the legacy solver did.
const STEADY: Lifetime = Lifetime::TARGET;
const LOAD_ONLY: Lifetime = Lifetime {
    first: Phase::Load,
    last: Phase::Load,
};

fn invalid<E: std::fmt::Display>(error: E) -> Refusal {
    Refusal::InvalidFact(error.to_string())
}

/// The exporter's program family: `dsv4f` (Flash, hidden 4096) or `dsv4p` (Pro).
fn program_family(inputs: &V4Inputs) -> &'static str {
    if inputs.cfg.dim == 4096 { "dsv4f" } else { "dsv4p" }
}

/// The attention placement V4 resolves. `None` is auto: two GPUs take V4's
/// recorded context default (a build without the C4 programs refuses there, it
/// never silently replicates KV), one GPU runs head-split.
fn attention(inputs: &V4Inputs) -> AttentionPlacement {
    inputs.attention.unwrap_or(if inputs.gpus.len() == 2 {
        AttentionPlacement::Context
    } else {
        AttentionPlacement::Heads
    })
}


/// Running arena peak: the workspace, then each part's resident bytes plus its
/// transient staging. Equal to the legacy `Arena` peak, and zero when nothing
/// ever opens the arena.
fn arena_peak(
    workspace: u64,
    parts: impl IntoIterator<Item = (u64, u64)>,
) -> Result<u64, Refusal> {
    let mut parts = parts.into_iter().peekable();
    if parts.peek().is_none() {
        return Ok(0);
    }
    let mut resident = workspace;
    let mut peak = workspace;
    for (resident_bytes, staging_bytes) in parts {
        let loading = resident
            .checked_add(resident_bytes)
            .and_then(|n| n.checked_add(staging_bytes))
            .ok_or(Refusal::Overflow("V4 expert arena peak"))?;
        peak = peak.max(loading);
        resident = resident
            .checked_add(resident_bytes)
            .ok_or(Refusal::Overflow("V4 expert arena"))?;
    }
    Ok(peak)
}

fn sum(bytes: impl IntoIterator<Item = u64>) -> Result<u64, Refusal> {
    bytes.into_iter().try_fold(0u64, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or(Refusal::Overflow("V4 resident weights"))
    })
}

/// Everything V4 resolves once: the immutable modes, the pool geometry the
/// request must match, and the expert count profile.
#[derive(Debug)]
pub struct PreparedV4<'a> {
    pool: crate::placement::PoolPolicy,
    inputs: &'a V4Inputs,
    expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    budgets: Vec<Budget>,
    reserves: Vec<u64>,
    modes: Vec<LayerMode>,
    attention: AttentionPlacement,
    prefill: PrefillShape,
    /// The pool floor: the compiled context, or the explicit pool an automatic
    /// checkpoint is clamped to.
    context: u64,
    page_rows: u64,
    maximum_pages: u64,
    /// Routed layers in the contract's onboarding order.
    routed: Vec<usize>,
    /// Strictly increasing RTX layer counts. A Spark-free layout holds every
    /// routed layer locally and reports exactly that count.
    counts: Vec<usize>,
    tp2: bool,
    /// The retained host prefix, resolved before admission.
    prefix: super::HostPrefix,
    /// Per-rank graph reserve (`GPU_GRAPH_RESERVE` resolved) and the estimate a
    /// device uses until its first measured capture.
    graph_reserves: Vec<u64>,
    graph_bootstraps: Vec<u64>,
    prepare_time: std::time::Duration,
}

#[derive(Debug)]
pub struct AdmittedV4 {
    pub solved: ResolvedPlan,
    pub placement: Placement,
    pub reserves: Vec<u64>,
    /// The graph arena per rank, charged once before KV sizing.
    pub graph_reserves: Vec<u64>,
    pub graph_bootstraps: Vec<u64>,
    /// Prepare plus solve/execution work, excluding checkpoint inventory and
    /// any caller delay between preparing and admitting the request.
    pub solve_us: u64,
}

fn modes(inputs: &V4Inputs) -> Result<(Vec<LayerMode>, AttentionPlacement), Refusal> {
    let gpus = inputs.gpus.len();
    if !(1..=2).contains(&gpus) {
        return Err(invalid("V4 admits one or two coordinator GPUs"));
    }
    if inputs.cfg.n_layers == 0 || inputs.cfg.dim == 0 {
        return Err(invalid("zero V4 layers or hidden size"));
    }
    let attention = attention(inputs);
    if attention == AttentionPlacement::Layers {
        return Err(Refusal::Unsupported);
    }
    if attention == AttentionPlacement::Context {
        if gpus != 2 {
            return Err(invalid("V4 context placement needs two coordinator GPUs"));
        }
        if inputs.context.is_none() {
            let family = program_family(inputs);
            let rows = inputs.decode_rows;
            let mut programs = [
                "dsa_candidate_merge",
                "lse_combine2",
                "paged_staging_gather_c4",
                "paged_staging_gather_index",
            ]
            .iter()
            .map(|suffix| format!("{family}_{suffix}"))
            .collect::<Vec<_>>();
            programs.push(format!("{family}_scored_index_topk_decode_m{rows}"));
            for begin in [0, inputs.cfg.n_heads / 2] {
                programs.push(format!("{family}_sparse_mla_partial_h{begin}_s*_m{rows}"));
            }
            return Err(invalid(format!(
                "V4 context placement needs this build's C4 programs: {}",
                programs.join(", ")
            )));
        }
    }
    if inputs.weights.target.len() != gpus {
        return Err(invalid("V4 target weight arena per GPU"));
    }
    if inputs.cfg.compress_ratios.len() != inputs.cfg.n_layers {
        return Err(invalid("V4 compress ratio per layer"));
    }
    if inputs.experts.len() > inputs.cfg.n_layers.saturating_sub(inputs.first_routed) {
        return Err(invalid("V4 expert costs exceed the model's routed layers"));
    }
    let split = attention == AttentionPlacement::Context;
    Ok((
        (0..inputs.cfg.n_layers)
            .map(|layer| match (gpus, split && inputs.cfg.compress_ratios[layer] == 4) {
                (2, true) => LayerMode::ContextSplit,
                (2, false) => LayerMode::HeadSplit,
                (_, _) => LayerMode::Whole {
                    gpu: 0,
                    ffn: FfnMode::Owner,
                },
            })
            .collect(),
        attention,
    ))
}

/// Physical capacities, baselines and reserves as the shared solver reads them.
fn budgets(inputs: &V4Inputs, reserves: &[u64]) -> Result<Vec<Budget>, Refusal> {
    if reserves.len() != inputs.gpus.len() {
        return Err(invalid("V4 reserve per GPU"));
    }
    let mut budgets = inputs
        .gpus
        .iter()
        .zip(reserves)
        .enumerate()
        .map(|(rank, (&(logical, baseline), &headroom))| {
            let physical = inputs.physical_bytes.get(rank).copied().unwrap_or(logical).max(logical);
            // `fixed_demands` charges the baseline's occupancy as the named
            // `context+modules` Runtime owner (the planned allowance offline,
            // the pre-weight sample in serve), so the cap starts whole here and
            // the ledger names the occupancy on every coordinator GPU (#1614).
            let baseline = match baseline {
                crate::placement::Baseline::Planned { .. } => Baseline::Forecast {
                    occupied: 0,
                    source: "full cap; planned context and modules charged as named owners".into(),
                    assumed: false,
                },
                crate::placement::Baseline::Measured { free_bytes } => Baseline::Forecast {
                    occupied: 0,
                    source: format!("full cap; pre-weight free sample {free_bytes} charged as context+modules"),
                    assumed: false,
                },
            };
            Ok(Budget {
                id: format!("rtx{rank}"),
                locations: vec![Location::coordinator(rank as u8)],
                physical_bytes: physical,
                logical_bytes: logical,
                baseline,
                headroom,
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    if inputs.spark_ranks > 0 {
        if inputs.spark_ranks > 8 {
            return Err(invalid("V4 admits at most eight Spark ranks"));
        }
        budgets.extend((0..inputs.spark_ranks as u8).map(|rank| Budget {
            id: format!("spark{rank}"),
            locations: vec![Location {
                device: Device::Spark(rank),
                domain: Domain::Managed,
            }],
            physical_bytes: SPARK_UNPRICED_BYTES,
            logical_bytes: SPARK_UNPRICED_BYTES,
            baseline: Baseline::Forecast {
                occupied: 0,
                source: "Spark claim memory is admitted by the workers; V4's planner names the \
                         remote expert destination without pricing it"
                    .into(),
                assumed: true,
            },
            headroom: 0,
        }));
    }
    if let Some(host) = &inputs.host {
        budgets.push(super::host::budget(host)?);
    }
    Ok(budgets)
}

/// The smallest pool the request admits.
fn context_floor(inputs: &V4Inputs) -> u64 {
    inputs.max_context
}

/// Prepare V4's immutable contract terms from one admission input.
pub fn prepare(inputs: &V4Inputs) -> Result<PreparedV4<'_>, Refusal> {
    let started = std::time::Instant::now();
    let (modes, attention) = modes(inputs)?;
    let (_, reserves) = fixed_demands(inputs).map_err(invalid)?;
    let budgets = budgets(inputs, &reserves)?;
    if inputs.max_context == 0
        || inputs.prefill_rows == 0
        || inputs.prefill_waves == 0
        || inputs.decode_rows == 0
    {
        return Err(invalid("zero V4 context or prefill geometry"));
    }
    let routed =
        (inputs.first_routed..inputs.first_routed + inputs.experts.len()).collect::<Vec<_>>();
    let tp2 = inputs.gpus.len() == 2 && (!inputs.experts_half.is_empty() || !inputs.draft_half.is_empty());
    if !inputs.draft_half.is_empty() && (!tp2 || inputs.draft_half.len() != inputs.draft.len()) {
        return Err(invalid("V4 dSpark TP2 stage halves require both GPUs and every stage"));
    }
    if tp2 && inputs.experts_half.len() != routed.len() {
        return Err(invalid("V4 TP2 expert halves per routed layer"));
    }
    if inputs.gpus.len() == 2 && !tp2 && !routed.is_empty() && inputs.spark_ranks == 0 {
        return Err(invalid(
            "V4 has no local expert arena for these experts and no Spark destination",
        ));
    }
    // A whole-layer arena exists on one GPU; on two, only distinct TP2 half
    // arenas do. Without one, every routed layer stays remote.
    let local = inputs.gpus.len() == 1 || tp2;
    let counts = if routed.is_empty() {
        Vec::new()
    } else if !local {
        vec![0]
    } else if inputs.spark_ranks == 0 {
        vec![routed.len()]
    } else {
        // V4 admits every local prefix, including all routed layers.
        (0..=routed.len()).collect()
    };
    // The graph arena is charged once, before KV sizing; the cache captures
    // lazily inside it and runs eagerly when it is full.
    let sparks = inputs.spark_ranks > 0;
    let graph_reserves = crate::serving_capacity::v4_graphs::v4_reserves(
        inputs.gpus.len(), sparks, inputs.decode_graphs)
        .map_err(invalid)?;
    let graph_bootstraps = crate::serving_capacity::v4_graphs::v4_bootstraps(inputs.gpus.len(), sparks)
        .map_err(invalid)?;
    let context = context_floor(inputs);
    let page_rows = if attention == AttentionPlacement::Context {
        2 * V4_PAGE_ROWS
    } else {
        V4_PAGE_ROWS
    };
    let cards = inputs.gpus.iter().map(|&(bytes, _)| bytes).collect::<Vec<_>>();
    let mut facts = crate::placement::PoolFacts::new(inputs.serving_context,
        inputs.requested_pool, page_rows);
    // Records one pool unit costs: a context split hands a rank half of every
    // C4 page while C128 pages replicate, as V4KvLayout prices. A whole-layer
    // placement may give every cached layer to either owner, so its summed
    // slope bounds every legal cut.
    let context_split = attention == AttentionPlacement::Context;
    let pages_per_unit = page_rows / V4_PAGE_ROWS;
    let mut unit_bytes = 0u64;
    for &ratio in &inputs.cfg.compress_ratios {
        let bytes = crate::serving_capacity::deepseek_v4_layer_unit_bytes(ratio)
            .checked_mul(pages_per_unit).ok_or(Refusal::Overflow("V4 pool unit"))?;
        let owned = if context_split && ratio == 4 { bytes / 2 } else { bytes };
        unit_bytes = unit_bytes.checked_add(owned).ok_or(Refusal::Overflow("V4 pool unit"))?;
    }
    facts.memory = if attention == AttentionPlacement::Layers {
        vec![crate::placement::PoolMemory {
            available_bytes: cards.iter().sum(), bytes_per_unit: unit_bytes }]
    } else {
        cards.iter().map(|&available_bytes| crate::placement::PoolMemory {
            available_bytes, bytes_per_unit: unit_bytes }).collect()
    };
    facts.encoding_limit = Some(super::POOL_CEILING_TOKENS);
    facts.target = inputs.target_pool;
    let pool = crate::placement::prepare_pool(facts).map_err(|error| Refusal::InvalidFact(error.to_string()))?;
    let maximum_tokens = pool.maximum_tokens().max(context);
    let maximum_pages = maximum_tokens.div_ceil(page_rows).max(1);
    let prefix = match &inputs.host {
        Some(host) => super::host::prefix_owners(inputs, host, attention == AttentionPlacement::Context)?,
        None => super::HostPrefix { owners: crate::placement::planner::serving::HostPrefixOwners {
            quota_bytes: 0, chunk_bytes: 0 }, geometry: None },
    };
    let expert_usage = inputs.expert_usage.then(|| cuteafd_core::expert_usage::Geometry {
        layers: inputs.cfg.n_layers,
        experts: inputs.cfg.n_routed_experts,
        topk: inputs.cfg.n_activated_experts,
        routed_layers: (inputs.first_routed..inputs.cfg.n_layers).collect(),
    });
    if expert_usage.as_ref().is_some_and(|geometry| !geometry.validate()) {
        return Err(invalid("invalid V4 expert usage geometry"));
    }
    Ok(PreparedV4 {
        pool,
        inputs,
        expert_usage,
        budgets,
        reserves,
        modes,
        attention,
        prefill: PrefillShape {
            rows: inputs.prefill_rows,
            waves: inputs.prefill_waves,
        },
        context,
        page_rows,
        maximum_pages,
        routed,
        counts,
        tp2,
        prefix,
        graph_reserves,
        graph_bootstraps,
        prepare_time: started.elapsed(),
    })
}

/// The request `plan --layout` and serve-dsv4 both admit.
pub fn production_request(inputs: &V4Inputs) -> Request {
    Request {
        context: inputs.serving_context,
        pool: inputs
            .requested_pool
            .filter(|&tokens| tokens > 0)
            .map_or(Pin::Auto, Pin::Exact),
        onboard: match inputs.onboard {
            Onboard::Auto => Pin::Auto,
            Onboard::Layers(count) => Pin::Exact(count as u64),
        },
        prefill: Pin::Exact(PrefillShape {
            rows: inputs.prefill_rows,
            waves: inputs.prefill_waves,
        }),
        operation: Operation::Serving,
        resident_weight_limit: inputs.resident_weight_limit,
        // V4's facts already price the resolved reply width; the request only
        // carries an operator selection for cross-checking.
        spark_reply_formats: SparkReplyFormats::default(),
    }
}

/// Collects components at one page domain so every curve covers
/// `maximum_pages` and every resident weight names its owner.
struct Owners {
    maximum: u64,
    components: Vec<Component>,
}

impl Owners {
    fn new(maximum: u64) -> Self {
        Self {
            maximum,
            components: Vec::new(),
        }
    }

    /// A steady owner with a pool-independent extent.
    fn owner(
        &mut self,
        id: String,
        gpu: u8,
        bytes: u64,
        category: Category,
        group: &str,
        activation: Activation,
    ) {
        if bytes == 0 {
            return;
        }
        let mut component = Component::allocation(
            Allocation::buffer(id, Location::coordinator(gpu), bytes, STEADY, category, group),
            ByteCurve::fixed(bytes, self.maximum),
            false,
            self.maximum,
        );
        component.activation = activation;
        self.components.push(component);
    }

    /// Weight residency of an owner that already exists (same id and group).
    fn weights(&mut self, id: String, gpu: u8, bytes: u64, activation: Activation) {
        if bytes == 0 {
            return;
        }
        let mut component = Component::resident_weight(
            ResidentWeight {
                id,
                location: Location::coordinator(gpu),
                weights: ResidentWeights(bytes),
            },
            ByteCurve::fixed(bytes, self.maximum),
        );
        component.activation = activation;
        self.components.push(component);
    }

    /// A pool-independent owner whose exact cudaMalloc requests are known
    /// here. Its single 2 MiB backing tail is charged as a separate named
    /// owner; requests that are already aligned add nothing.
    fn physical(
        &mut self,
        id: String,
        gpu: u8,
        requests: Vec<u64>,
        category: Category,
        group: &str,
        activation: Activation,
    ) -> Result<(), Refusal> {
        let requests: Vec<u64> = requests.into_iter().filter(|&bytes| bytes > 0).collect();
        let bytes = requests.iter().try_fold(0u64, |sum, &n| sum.checked_add(n)
            .ok_or(Refusal::Overflow("V4 physical owner")))?;
        if bytes == 0 {
            return Ok(());
        }
        let location = Location::coordinator(gpu);
        let mut component = Component::allocation(
            Allocation::buffer(id.clone(), location, bytes, STEADY, category, group),
            ByteCurve::fixed(bytes, self.maximum), false, self.maximum);
        component.activation = activation;
        self.components.push(component);
        // Requests at or below 1 MiB share driver blocks; their rounding is a
        // pool-level fact, not this owner's. One request above 1 MiB already
        // forces the whole extent into its own cudaMalloc, so a mixed list
        // prices its backing exactly like an all-large one.
        if requests.iter().any(|&bytes| bytes > (1 << 20)) {
            let granularity = AllocationGranularity { id: format!("{id}.tail"), location,
                lifetime: STEADY, evidence: GranularityEvidence::Forecast,
                backings: vec![GranularityBacking { backing: id, requests }] };
            if granularity.bytes()? > 0 {
                self.components.push(allocation_granularity_component(granularity, self.maximum)?);
            }
        }
        Ok(())
    }

    /// One exact dense-loading destination. The legacy solver judged each
    /// rank's loading extent against physical memory independently of the
    /// steady ledger; a Load-only sequential group keeps that independence.
    fn loading(&mut self, gpu: u8, bytes: u64, group: &str) {
        if bytes == 0 {
            return;
        }
        let id = format!("v4.load.{gpu}");
        let member = Allocation::buffer(
            format!("{id}.dense"),
            Location::coordinator(gpu),
            bytes,
            LOAD_ONLY,
            Category::Weights,
            group,
        );
        let sequential = Sequential {
            id: format!("{id}.group"),
            members: vec![member],
        };
        self.components.push(Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(sequential.peak(), self.maximum),
            kind: ComponentKind::Sequential(sequential),
        });
    }
}

impl<'a> PreparedV4<'a> {
    pub fn budgets(&self) -> &[Budget] {
        &self.budgets
    }

    pub fn reserves(&self) -> &[u64] {
        &self.reserves
    }

    pub fn modes(&self) -> &[LayerMode] {
        &self.modes
    }

    pub fn attention(&self) -> AttentionPlacement {
        self.attention
    }

    pub fn context(&self) -> u64 {
        self.context
    }

    /// The exact geometry priced by Recipe and passed unchanged to the engine.
    pub fn expert_usage(&self) -> Option<&cuteafd_core::expert_usage::Geometry> {
        self.expert_usage.as_ref()
    }

    /// The retained host prefix quota this contract declares.
    pub fn prefix(&self) -> super::HostPrefix {
        self.prefix
    }

    /// Whole dSpark stages open GPU0's TP1 arena unless their halves share
    /// the TP2 arena. In whole mode they precede the backbone layers.
    fn dspark_peak(&self) -> Result<u64, Refusal> {
        if !self.inputs.draft_half.is_empty() { return Ok(0); }
        arena_peak(
            self.inputs.expert_workspace,
            self.inputs
                .draft
                .iter()
                .map(|cost| (cost.resident_bytes, cost.staging_bytes)),
        )
    }

    fn dspark_resident(&self) -> Result<u64, Refusal> {
        if !self.inputs.draft_half.is_empty() { return Ok(0); }
        sum(self.inputs.draft.iter().map(|cost| cost.resident_bytes))
    }

    /// Restrict a remote-executor geometry to candidates that actually use it.
    pub fn require_remote_experts(mut self) -> Result<Self, Refusal> {
        self.counts.retain(|&count| count < self.routed.len());
        if self.counts.is_empty() { return Err(Refusal::PinConflict); }
        Ok(self)
    }

    /// Whole-layer arena peak on GPU0 for `count` onboarded layers, including
    /// dSpark's stages ahead of them. Zero on every other GPU.
    fn whole_peak(&self, count: usize) -> Result<u64, Refusal> {
        if !self.inputs.draft_half.is_empty() { return Ok(0); }
        arena_peak(
            self.inputs.expert_workspace,
            self.inputs
                .draft
                .iter()
                .map(|cost| (cost.resident_bytes, cost.staging_bytes))
                .chain(
                    self.inputs.experts[..count]
                        .iter()
                        .map(|cost: &V4ExpertCost| (cost.resident_bytes, cost.staging_bytes)),
                ),
        )
    }

    fn whole_expert_resident(&self, count: usize) -> Result<u64, Refusal> {
        sum(self.inputs.experts[..count]
            .iter()
            .map(|cost| cost.resident_bytes))
    }

    /// TP2 half-arena peak for `count` layers on `gpu`. The staged order is
    /// per step, so the peak is the resident total plus the largest staging.
    fn tp2_peak(&self, gpu: usize, count: usize) -> Result<u64, Refusal> {
        if count == 0 && self.inputs.draft_half.is_empty() {
            return Ok(0);
        }
        // The arena's ordered-exchange payload is allocated only while it holds
        // layers, beside the package's capacity workspace and one layer's
        // staging (all layers stage one at a time).
        let payload = crate::serving_capacity::deepseek_v4_tp2_payload_bytes(
            self.inputs.cfg.dim as u64,
            self.inputs.prefill_rows,
            self.inputs.decode_rows,
            self.inputs.prefill_waves,
        )
        .map_err(|_| Refusal::Overflow("V4 TP2 payload"))?;
        let halves = self.inputs.experts_half[..count].iter().chain(&self.inputs.draft_half);
        let resident = sum(halves.clone().map(|half| half[gpu].resident_bytes))?;
        let staging = halves
            .map(|half| half[gpu].staging_bytes)
            .max()
            .unwrap_or(0);
        self.inputs.tp2_workspace[gpu]
            .checked_add(payload)
            .and_then(|n| n.checked_add(resident))
            .and_then(|n| n.checked_add(staging))
            .ok_or(Refusal::Overflow("V4 TP2 arena peak"))
    }

    fn tp2_resident(&self, gpu: usize, count: usize) -> Result<u64, Refusal> {
        sum(self.inputs.experts_half[..count].iter().chain(&self.inputs.draft_half)
            .map(|half| half[gpu].resident_bytes))
    }

    /// Mechanical executor output for one solved assignment.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Placement, Refusal> {
        let decisions = &solved.decisions;
        if decisions.attention != self.modes {
            return Err(invalid(
                "V4 attention decisions differ from the prepared modes",
            ));
        }
        let onboarded: BTreeSet<usize> = decisions
            .onboard_layers
            .iter()
            .flatten()
            .copied()
            .collect();
        let count = onboarded.len();
        let expected: Vec<usize> = self.routed[..count].to_vec();
        if onboarded.iter().copied().collect::<Vec<_>>() != expected {
            return Err(invalid("V4 onboarding is not the declared ordered prefix"));
        }
        let routed: BTreeSet<usize> = self.routed.iter().copied().collect();
        if decisions
            .onboard_layers
            .iter()
            .enumerate()
            .any(|(gpu, layers)| {
                let owned = if self.tp2 || gpu == 0 { count } else { 0 };
                layers.len() != owned
            })
        {
            return Err(invalid("V4 onboarding layers per GPU"));
        }
        let mut homes = vec![ExpertHome::Dense; self.inputs.cfg.n_layers];
        for layer in &routed {
            if onboarded.contains(layer) {
                let home = match decisions.expert_homes.get(layer) {
                    Some(ExpertDestination::Coordinator { gpus })
                        if self.tp2 && gpus == &[0, 1] =>
                    {
                        ExpertHome::RtxTp2
                    }
                    Some(ExpertDestination::Coordinator { gpus }) if !self.tp2 && gpus == &[0] => {
                        ExpertHome::RtxWhole { gpu: 0 }
                    }
                    _ => return Err(invalid("V4 onboarded layer has no coordinator home")),
                };
                homes[*layer] = home;
            } else {
                match decisions.expert_homes.get(layer) {
                    Some(ExpertDestination::Sparks { .. }) | None => {
                        homes[*layer] = ExpertHome::Spark
                    }
                    Some(ExpertDestination::Coordinator { .. }) => {
                        return Err(invalid("V4 remote layer claims a coordinator home"))
                    }
                }
            }
        }
        if decisions
            .expert_homes
            .keys()
            .any(|layer| !routed.contains(layer))
        {
            return Err(invalid("V4 expert home on a dense layer"));
        }
        let gpus = self.inputs.gpus.len();
        let mut expert_ranges = vec![
            ExpertRange {
                first: self.inputs.first_routed,
                layers: 0,
                peak_bytes: 0,
            };
            gpus
        ];
        // Explicit-off stages retain their TP1 arena beside the backbone halves.
        // Split stages instead share the TP2 arena, even at zero onboarding.
        let tp2 = if self.tp2 {
            expert_ranges[0].peak_bytes = self.dspark_peak()?;
            (count > 0 || !self.inputs.draft_half.is_empty())
                .then(|| {
                    Ok::<_, Refusal>(Tp2Range {
                        first: self.inputs.first_routed,
                        layers: count,
                        peak_bytes: [self.tp2_peak(0, count)?, self.tp2_peak(1, count)?],
                    })
                })
                .transpose()?
        } else {
            if gpus > 0 {
                expert_ranges[0] = ExpertRange {
                    first: self.inputs.first_routed,
                    layers: count,
                    peak_bytes: self.whole_peak(count)?,
                };
            }
            None
        };
        let hop_spec = self.hop_spec();
        let hops = plan_hops(&self.modes, &hop_spec).map_err(invalid)?;
        let residual = residual_homes(&self.modes, &hop_spec).map_err(invalid)?;
        let peer_row_bytes = if self.attention == AttentionPlacement::Context {
            let buffers = self.context_buffers();
            self.modes
                .iter()
                .zip(&self.inputs.cfg.compress_ratios)
                .filter(|(mode, _)| **mode == LayerMode::ContextSplit)
                .try_fold(0u64, |total, (_, &ratio)| {
                    buffers
                        .query_row_bytes
                        .checked_add(buffers.partial_row_bytes)
                        .and_then(|n| {
                            n.checked_add(if ratio == 4 {
                                buffers.candidate_row_bytes
                            } else {
                                0
                            })
                        })
                        .and_then(|n| n.checked_add(total))
                        .ok_or(Refusal::Overflow("V4 context peer rows"))
                })?
        } else {
            0
        };
        let items = (0..gpus)
            .map(|gpu| solved.items(Location::coordinator(gpu as u8)))
            .collect::<Result<Vec<_>, _>>()?;
        let placement = Placement {
            pool_tokens: solved.pool_tokens,
            pool_limits: PoolLimits(Vec::new()),
            attention_placement: self.attention,
            attention_by_kind: vec![(AttentionClass::Csa, self.attention)],
            peer_row_bytes,
            onboard_layers: count,
            max_pool_tokens_no_onboard: None,
            layers: self
                .modes
                .iter()
                .zip(homes)
                .map(|(&mode, experts)| LayerAssignment { mode, experts })
                .collect(),
            movables: if self.inputs.draft.is_empty() {
                Vec::new()
            } else {
                vec![(MovableId::DsparkExperts, 0)]
            },
            expert_ranges,
            residual,
            hops,
            tp2,
            items,
        };
        super::super::DEEPSEEK_V4
            .check(&placement)
            .map_err(invalid)?;
        Ok(placement)
    }

    fn hop_spec(&self) -> HopSpec {
        HopSpec {
            storage: HopStorage::ReceiveSlots,
            row_bytes: 4 * self.inputs.cfg.dim as u64 * 2,
            rows: self.inputs.prefill_rows.max(self.inputs.decode_rows),
            lanes: self.inputs.prefill_waves,
            entry_gpu: 0,
            head_gpu: 0,
            terminal: TerminalHome::Head,
        }
    }

    /// Context buffers are sized by the compiled extent, never by a smaller
    /// pool: the operator's pool only clamps the served context.
    fn context_buffers(&self) -> crate::placement::ContextBuffers {
        context_buffers(
            &self.inputs.cfg,
            self.inputs.max_context,
            self.inputs.prefill_rows,
            self.inputs.decode_rows,
            self.inputs.prefill_waves,
            self.inputs.context.unwrap_or_default(),
        )
    }

    pub fn admit(self, request: Request) -> Result<AdmittedV4, Refusal> {
        let started = std::time::Instant::now();
        if let Pin::Exact(count) = request.onboard {
            if count as usize > self.routed.len() {
                return Err(invalid(format!(
                    "--rtx-expert-layers {count} exceeds the {} routed layers",
                    self.routed.len()
                )));
            }
        }
        let solved = solve_prepared(request, self.budgets.clone(), &self, &self.pool)?;
        let placement = self.execution(&solved)?;
        let reserves = self.reserves.clone();
        let graph_reserves = self.graph_reserves.clone();
        let graph_bootstraps = self.graph_bootstraps.clone();
        let solve_us = self.prepare_time.saturating_add(started.elapsed()).as_micros()
            .try_into().unwrap_or(u64::MAX);
        Ok(AdmittedV4 { solved, placement, reserves, graph_reserves, graph_bootstraps, solve_us })
    }
}

impl Recipe for PreparedV4<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        let inputs = self.inputs;
        if request.context.maximum != self.context {
            return Err(invalid(format!(
                "V4 request context {} differs from prepared {}",
                request.context.maximum, self.context
            )));
        }
        if let Pin::Exact(shape) = request.prefill {
            if shape != self.prefill {
                return Err(Refusal::PinConflict);
            }
        }
        let mut owners = Owners::new(self.maximum_pages);
        if let Some(geometry) = self.expert_usage() {
            owners.components.extend(crate::placement::expert_usage::components(geometry, &[0], self.maximum_pages)?);
        }
        let (fixed, _) = fixed_demands(inputs).map_err(invalid)?;
        for (index, demand) in fixed.iter().enumerate() {
            let id = format!("v4.fixed.{index}");
            let activation = Activation::always();
            if demand.group == "context+modules"
                && matches!(inputs.gpus[demand.gpu as usize].1, crate::placement::Baseline::Measured { .. }) {
                owners.components.push(Component { activation,
                    bytes: ByteCurve::fixed(demand.bytes, self.maximum_pages),
                    kind: ComponentKind::Allowance(crate::placement::planner::Allowance {
                        location: Location::coordinator(demand.gpu), bytes: demand.bytes,
                        lifetime: Lifetime::RESIDENT, assumed: false,
                        source: crate::placement::sampled_baseline::source("V4 context and selected modules"),
                    }) });
                continue;
            }
            // The daemon's own builders report their exact cudaMalloc request
            // lists for the step workspaces, Spark intake planes and route
            // probe; the prefix mark arena is one request. Both thereby price
            // their shared 2 MiB backing instead of only their payload. Any
            // other demand keeps its logical extent (and is either a single
            // aligned request or already priced from its own geometry).
            let requests = match demand.group.as_str() {
                "steps" => steps_requests(inputs, demand.gpu as usize).map_err(invalid)?,
                "marks" => Some(vec![demand.bytes]),
                _ => None,
            };
            match requests {
                Some(requests) => owners.physical(id.clone(), demand.gpu, requests,
                    demand.category, &demand.group, activation.clone())?,
                None => owners.owner(id.clone(), demand.gpu, demand.bytes,
                    demand.category, &demand.group, activation.clone()),
            }
            owners.weights(id, demand.gpu, demand.resident_weights.0, activation);
        }
        // Owners a caller adds after the family's own facts (an RTX media
        // tower), priced exactly like a fixed demand.
        for (index, demand) in inputs.extra.iter().enumerate() {
            let id = format!("v4.extra.{index}");
            let activation = Activation::always();
            owners.owner(
                id.clone(),
                demand.gpu,
                demand.bytes,
                demand.category,
                &demand.group,
                activation.clone(),
            );
            owners.weights(id, demand.gpu, demand.resident_weights.0, activation);
        }
        // One physical request per GPU: views never add payload or rounding owners.
        let gpus = inputs.gpus.len();
        let context = self.attention == AttentionPlacement::Context;
        let sequences = inputs.sequences.max(1);
        let ring_pages = cuteafd_core::prefix::swa_ring(inputs.cfg.window_size as u64,
            inputs.prefill_rows, V4_PAGE_ROWS).ok_or(Refusal::Overflow("V4 window ring"))?.rows / V4_PAGE_ROWS;
        for gpu in 0..gpus {
            let waves = inputs.workspace.as_ref().and_then(|ranks| ranks.get(gpu))
                .map_or(inputs.prefill_waves, |rank| rank.prefill_waves);
            let extent = |pages| -> Result<u64, Refusal> {
                let units = super::host::pool_units(pages, self.page_rows, sequences, context)?;
                let layout = crate::serving_capacity::V4KvLayout::new(&inputs.cfg,
                    sequences.try_into().map_err(invalid)?, ring_pages.try_into().map_err(invalid)?,
                    units.try_into().map_err(invalid)?, context,
                    if gpu == 0 { inputs.cache_stages } else { 0 },
                    inputs.max_context.try_into().map_err(invalid)?, waves.try_into().map_err(invalid)?,
                    inputs.decode_rows.try_into().map_err(invalid)?).map_err(invalid)?;
                Ok(layout.bytes() as u64)
            };
            // Only the final decode table's 256-byte minimum can be nonlinear.
            // Once past it, every earlier view's slope is 64-byte aligned.
            let linear = (1..=64).find(|&pages| super::host::pool_units(pages, self.page_rows,
                sequences, context).is_ok_and(|units| units.saturating_mul(inputs.decode_rows) >= 64))
                .unwrap_or(64).min(self.maximum_pages);
            let mut pieces = Vec::new();
            for pages in 1..linear {
                pieces.push(BytePiece { first: pages, last: pages, bytes_at_first: extent(pages)?,
                    bytes_per_page: 0, granularity: SM120_ALLOCATION_GRANULARITY_BYTES, backings: 1 });
            }
            let first = extent(linear)?;
            let slope = if linear < self.maximum_pages { extent(linear + 1)?.checked_sub(first)
                .ok_or(Refusal::Overflow("V4 KV arena slope"))? } else { 0 };
            pieces.push(BytePiece { first: linear, last: self.maximum_pages, bytes_at_first: first,
                bytes_per_page: slope, granularity: SM120_ALLOCATION_GRANULARITY_BYTES, backings: 1 });
            let curve = ByteCurve(pieces);
            let mut component = Component::allocation(Allocation::buffer(format!("v4.kv.arena.{gpu}"),
                Location::coordinator(gpu as u8), curve.at(1)?, STEADY, Category::Kv, "KV arena"),
                curve.clone(), false, self.maximum_pages);
            component.byte_extent(curve)?;
            owners.components.push(component);
        }
        // Retained target and dSpark weights: each arena's exact extent B,
        // charged once with its single SM120 allocation tail. They exist from
        // Load (allocated before KV) until Growth, like the runtime receipts.
        for (gpu, arena) in inputs.weights.target.iter().enumerate() {
            owners.components.extend(weight_arena_components(arena, Location::coordinator(gpu as u8),
                Lifetime::RESIDENT, self.maximum_pages, cuteafd_core::memory_layout::MemoryKind::Weights)?);
        }
        if let Some(arena) = &inputs.weights.draft {
            owners.components.extend(weight_arena_components(arena, Location::coordinator(0),
                Lifetime::RESIDENT, self.maximum_pages, cuteafd_core::memory_layout::MemoryKind::DrafterWeights)?);
        }
        // The lead GPU's one borrowed vocabulary GEMM handle and the shared
        // cuBLASLt initialization: 64 MiB + 2 MiB, created on the first target
        // or draft vocabulary head. Measured once on the sealed V4 census
        // (2026-11-08, SM120/driver 13020 LAZY); the head split's peer runs no
        // vocabulary head, so no second owner exists (#1579).
        let vocabulary = serving::BlasOwners {
            name: "shared/blas/vocabulary".into(),
            process: "coordinator".into(),
            thread: "v4/blas/serve".into(),
            location: Location::coordinator(0),
            gemm: false,
            lt: true,
            lifetime: Lifetime::RESIDENT,
            activation: Activation::always(),
        };
        owners.components.extend(serving::blas_components("sm_120", &[vocabulary],
            self.maximum_pages, PhaseOrder::TARGET_THEN_DRAFT)?);
        // The driver's retained stream slabs (#1511): one 2 MiB band per four
        // live streams, which stays on the device after the streams that grew
        // it retire. V4 holds its engine stream and the prefix cache's copy
        // stream on every rank, one intake copy stream per Spark link, and --
        // when the retained host tier is live, exactly when `prefix_owners`
        // admits a quota -- the host copy engine's store/restore pair with an
        // event probe on each device. This is not a cudaMalloc request, so it
        // receives no allocation-tail charge.
        let host_tier = self.prefix.owners.quota_bytes > 0;
        let peak = super::startup_peak_live_streams(host_tier, inputs.spark_ranks > 0);
        let slab = cuda_stream_slab_bytes(peak).map_err(invalid)?;
        for gpu in 0..gpus {
            owners.components.push(Component {
                activation: Activation::always(),
                bytes: ByteCurve::fixed(slab, self.maximum_pages),
                kind: ComponentKind::Allowance(Allowance {
                    location: Location::coordinator(gpu as u8),
                    bytes: slab,
                    lifetime: Lifetime::RESIDENT,
                    assumed: false,
                    source: format!("v4/driver-stream-slabs/gpu{gpu} ({peak} peak live streams, SM120 driver/runtime13020 LAZY, #1511 2026-11-08)"),
                }),
            });
        }
        // The graph arena, one owner per coordinator rank: executable residency
        // plus the disposable graph pool. It is charged once here, before the
        // pool sizes KV, and the bounded cache captures inside it. The source
        // survives GPU_RESERVE (its bytes are admitted, not a policy margin).
        for (rank, &reserve) in self.graph_reserves.iter().enumerate() {
            let id = format!("v4.graphs.{rank}");
            let mut component = crate::placement::planner::serving::startup_graph_component(
                &id, Location::coordinator(rank as u8),
                GraphRank { executables: 0, bytes: reserve, margin: 0, basis: Basis::Exact, node_charge: None },
                self.maximum_pages)?;
            if let ComponentKind::Allowance(owner) = &mut component.kind {
                owner.source = crate::placement::graph_reserve::source(&id);
                // This owner is the configured budget, not a per-capture estimate.
                owner.assumed = false;
            }
            owners.components.push(component);
        }
        // Context staging, exchange and workspace exist only under a split.
        if self.attention == AttentionPlacement::Context {
            for demand in self.context_buffers().demands().map_err(invalid)? {
                owners.owner(
                    format!("v4.context.{}.{}", demand.gpu, demand.group),
                    demand.gpu,
                    demand.bytes,
                    demand.category,
                    &demand.group,
                    Activation::always(),
                );
            }
        }
        // The residual hop is the embedding broadcast into GPU1.
        let hop_spec = self.hop_spec();
        let hops = plan_hops(&self.modes, &hop_spec).map_err(invalid)?;
        let hop_bytes = hop_buffer_bytes(&hops, &hop_spec, gpus)
            .ok_or_else(|| invalid("V4 residual hop buffers"))?;
        for (gpu, bytes) in hop_bytes.into_iter().enumerate() {
            owners.owner(
                format!("v4.hops.{gpu}"),
                gpu as u8,
                bytes,
                Category::Transport,
                "residual hops",
                Activation::always(),
            );
        }
        // Dense loading stays a Load-only peak per rank.
        for demand in inputs.weights.startup.device_phases(gpus).map_err(invalid)? {
            owners.loading(demand.gpu, demand.bytes, &demand.group);
        }
        // Mandatory dSpark residency exists without onboarded layers; the
        // backbone adds only its marginal arena cost per onboard count.
        let dspark = self.dspark_peak()?;
        owners.owner(
            "v4.tp1.0".into(),
            0,
            dspark,
            Category::Experts,
            "local",
            Activation::always(),
        );
        owners.weights(
            "v4.tp1.0".into(),
            0,
            self.dspark_resident()?,
            Activation::always(),
        );

        for gpu in 0..2 {
            if !inputs.draft_half.is_empty() {
                owners.owner(format!("v4.tp2.draft.{gpu}"), gpu as u8, self.tp2_peak(gpu, 0)?,
                    Category::Experts, "tp2", Activation::always());
                owners.weights(format!("v4.tp2.draft.{gpu}"), gpu as u8, self.tp2_resident(gpu, 0)?, Activation::always());
            }
        }
        let mut variables = self
            .modes
            .iter()
            .enumerate()
            .map(|(layer, &mode)| Variable::Attention {
                layer,
                values: vec![mode],
            })
            .collect::<Vec<_>>();
        variables.push(Variable::Prefill(vec![self.prefill]));
        variables.push(Variable::Lanes(vec![inputs.prefill_waves]));
        let onboard_variable = if self.routed.is_empty() {
            None
        } else {
            let variable = variables.len();
            variables.push(Variable::Onboard {
                layers: self.routed.clone(),
                destination: if self.tp2 { vec![0, 1] } else { vec![0] },
                counts: self.counts.clone(),
            });
            Some(variable)
        };
        if !self.routed.is_empty() && inputs.spark_ranks > 0 {
            variables.push(Variable::ExpertHome {
                layers: self.routed.clone(),
                rank_sets: vec![(0..inputs.spark_ranks as u8).collect()],
            });
        }
        if let Some(variable) = onboard_variable {
            for (option, &count) in self.counts.iter().enumerate() {
                let activation = Activation::choice(variable, option);
                if self.tp2 {
                    for gpu in 0..2 {
                        owners.owner(
                            format!("v4.tp2.{gpu}"),
                            gpu as u8,
                            self.tp2_peak(gpu, count)?.saturating_sub(self.tp2_peak(gpu, 0)?),
                            Category::Experts,
                            "tp2",
                            activation.clone(),
                        );
                        owners.weights(
                            format!("v4.tp2.{gpu}"),
                            gpu as u8,
                            self.tp2_resident(gpu, count)?.saturating_sub(self.tp2_resident(gpu, 0)?),
                            activation.clone(),
                        );
                    }
                } else {
                    // The arena's marginal cost over dSpark's stages: the
                    // ledger charges `v4.tp1.0` unconditionally.
                    let peak = self.whole_peak(count)?.saturating_sub(dspark);
                    owners.owner(
                        "v4.local.0".into(),
                        0,
                        peak,
                        Category::Experts,
                        "local",
                        activation.clone(),
                    );
                    owners.weights(
                        "v4.local.0".into(),
                        0,
                        self.whole_expert_resident(count)?,
                        activation,
                    );
                }
            }
        }
        if let Some(host) = &inputs.host {
            owners.components.extend(super::host::components(inputs, host, self.prefix.owners, self.page_rows,
                self.attention == AttentionPlacement::Context, self.maximum_pages)?);
        }
        Ok(Contract {
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: None,
            resident_weight_limit: request.resident_weight_limit,
            page_rows: self.page_rows,
            pool_page_multiple: 1,
            maximum_pages: self.maximum_pages,
            gpus,
            variables,
            requires: Vec::new(),
            components: owners.components,
        })
    }
}
