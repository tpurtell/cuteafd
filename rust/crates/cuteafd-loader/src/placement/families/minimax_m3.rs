//! MiniMax M3 physical owners for the shared contract planner (#1856 H1).
//!
//! A family supplies facts, never a private solver. Both edges, `cuteafd plan
//! --layout` and serving startup, prepare one [`MiniMaxInputs`] and submit it
//! to [`planner::solve`](crate::placement::planner::solve). They differ only
//! in the coordinator [`Baseline`] (planned context forecast vs one measured
//! sample). Every owner is named once:
//!
//! - target arena per coordinator rank (globals and each owned layer's
//!   attention, indexer, router and dense/shared FFN), one backing plus one tail;
//! - RTX expert arena per onboarding count (whole on one RTX, TP2 halves on a
//!   pair) with the installed fp8moe package's executor workspace;
//! - Spark experts per rank per onboarding count, with their I/O, typed
//!   execution owners, runtime residual and SM121 arena page tables;
//! - MSA KV + index pool pages (whole-layer owners: never replicated);
//! - per-sequence page tables, decode lane and per-slot prefill unit owners,
//!   TP2 expert exchange and the boundary hop on a pair,
//!   MSA/producer/dense scratch, head, selector, rotary table;
//! - one canonical graph reserve per device, loaded code, stream slabs, BLAS;
//! - host: prefix quota, upload staging, mount source, scoring and transport;
//! - optional: bundled dSpark drafter arena (own arena, `DraftHome` domain)
//!   and the bundled vision tower (own arena, `Encoder` domain).
//!
//! No figure here is a margin: unmeasured owners say so in their source.
use super::super::planner::*;
use super::super::{
    plan_hops, residual_homes, AttentionClass, AttentionPlacement, ExecutorModes, ExpertHome, ExpertRange,
    FfnMode, HopSpec, HopStorage, LayerAssignment, LayerMode, MovableId, Placement, PoolLimits,
    ResidentWeights, TerminalHome, Tp2Range,
};
use crate::families::minimax_m3::{execution::*, weight_layout::CoordinatorWeights, MiniMaxM3Config};
use crate::serving_capacity::WeightArenaLayout;
use cuteafd_core::memory_layout::{Category, MemoryKind};
use cuteafd_core::spark_reply::SparkReplyFormats;

/// Native MSA page quantum (64-token pages; the selector's blocks are 128).
pub const PAGE_ROWS: u64 = 64;
/// Prefill transport endpoints: one shared link (LaneSupport::One).
pub const LANES: u64 = 1;
/// Decode/verify rows per step, the Spark decode owner's admitted extent.
pub const DECODE_ROWS: u32 = 64;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };

/// What the MiniMax executor runs. One RTX owns every layer. A pair owns
/// contiguous whole layers (`layers`, the qualified Qwen4 dual pattern):
/// attention, indexer, router and dense/shared FFN on the owner, one boundary
/// hop at the cut, routed experts as TP2 halves on both GPUs. A whole owner
/// never replicates KV or index pages; 4 KV heads make a head split uneven
/// against the 4 independent index heads, and no context-split MSA program
/// exists. It runs only the existing whole-width MXFP8 exports.
pub const MINIMAX_M3: ExecutorModes = ExecutorModes { family: "minimax_m3",
    modes: &[WHOLE, LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }], hops: true,
    attention: AttentionPlacement::Layers };

fn invalid(detail: impl std::fmt::Display) -> Refusal {
    Refusal::InvalidFact(format!("MiniMax component contract: {detail}"))
}

fn refusal(error: anyhow::Error) -> Refusal { invalid(format!("{error:#}")) }

/// Per-rank resident bytes of one routed layer on the coordinator: whole or a
/// TP2 half. Header-exact from the checkpoint's own compressed NVFP4 tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayerExperts {
    pub whole: u64,
    pub half: [u64; 2],
}

/// The installed coordinator fp8moe package (`fp8-minimax-nvfp4a4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPackage {
    /// Executor workspace per GPU at the admitted unit (TP1 or TP2 layout).
    Installed { workspace: [u64; 2] },
    /// No package beside the manifest: onboarding counts above zero are refused.
    Missing(String),
}

/// One Spark rank's charges at one onboarding count (layers `count..` remote).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkCharge {
    pub experts: u64,
    /// Endpoint I/O plus typed execution owners (prefill + decode lane).
    pub owners: u64,
    pub runtime: u64,
}

/// Bundled dSpark drafter: its own arena on the head GPU plus its scratch.
#[derive(Debug, Clone)]
pub struct Drafter {
    pub arena: WeightArenaLayout,
    /// Exact scratch owners (name, bytes); state that follows the pool is
    /// charged per page.
    pub scratch: Vec<(String, u64)>,
    pub state_per_page: u64,
}

/// Bundled CLIP 3D-RoPE tower and projector: own arena plus scratch.
#[derive(Debug, Clone)]
pub struct Vision {
    pub arena: WeightArenaLayout,
    pub scratch: u64,
    /// Host embedding cache quota for prepared images.
    pub media_cache: u64,
}

/// Everything both edges know before one solve.
#[derive(Debug, Clone)]
pub struct MiniMaxInputs {
    pub cfg: MiniMaxM3Config,
    pub gpus: usize,
    pub spark_ranks: usize,
    pub context: Context,
    /// Largest pool the build indexes (KV max), in tokens.
    pub maximum_pool_tokens: u64,
    pub prefill: PrefillShape,
    /// Boot concurrency C: per-sequence page tables, lane members, selector.
    pub sequences: u64,
    /// Per-rank target arenas (one per coordinator GPU).
    pub targets: Vec<CoordinatorWeights>,
    /// Owner GPU of every layer's attention (`0` everywhere on one RTX).
    pub owners: Vec<u8>,
    pub manifest: std::sync::Arc<serde_json::Value>,
    /// Routed layer order (the onboarding prefix is `routed[..count]`).
    pub routed: Vec<usize>,
    pub experts: Vec<LayerExperts>,
    pub local: LocalPackage,
    /// Remote charges per onboarding count, per rank (`spark[count][rank]`).
    pub spark: Vec<Vec<SparkCharge>>,
    pub links: Vec<crate::placement::PreparedSparkLink>,
    pub spark_reply_formats: SparkReplyFormats,
    pub drafter: Option<Drafter>,
    pub vision: Option<Vision>,
    /// Coordinator budgets (one per GPU) plus one Spark budget per rank and the host.
    pub budgets: Vec<Budget>,
    /// Per-rank planned code allowance (zero on the measured edge).
    pub code: Vec<u64>,
    /// Per-rank graph reserve (the canonical `graph reserve` owner).
    pub graph_reserve: Vec<u64>,
    pub head: serving::BlasHead,
    pub host_prefix: serving::HostPrefixOwners,
    pub device_sampling: bool,
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub resident_weight_limit: Option<ResidentWeights>,
    /// Mount source per rank (Load only), header-bounded.
    pub mount: Vec<crate::shared::dense_read::MountLoad>,
}

/// The admitted answer. Plan and runtime read the same fields.
#[derive(Debug, Clone)]
pub struct AdmittedMiniMax {
    pub placement: Placement,
    pub solved: ResolvedPlan,
    pub onboard: usize,
    pub draft_home: Option<u8>,
    pub encoder: Option<usize>,
}

/// The one request both edges submit.
pub fn request(context: Context, pool: Option<u64>, onboard: crate::placement::Onboard, prefill: PrefillShape,
    operation: Operation, resident_weight_limit: Option<ResidentWeights>, replies: SparkReplyFormats) -> Request {
    Request { context,
        pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard { crate::placement::Onboard::Auto => Pin::Auto,
            crate::placement::Onboard::Layers(count) => Pin::Exact(count as u64) },
        prefill: Pin::Exact(prefill), operation, resident_weight_limit, spark_reply_formats: replies }
}

/// One Spark endpoint at the admitted unit: native BF16 ingress (NVFP4 W4A4
/// experts quantize their own activations from the calibrated input scale).
pub fn spark_link(gpu: u8, ranks: usize, prefill: PrefillShape, replies: SparkReplyFormats)
    -> Result<Vec<crate::placement::PreparedSparkLink>, Refusal> {
    use crate::placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates};
    if ranks == 0 { return Ok(Vec::new()); }
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    Ok(vec![PreparedSparkLink::endpoint("minimax.spark", gpu, SparkEndpointOwners { ranks,
        geometry: cuteafd_core::ExpertGeometry::MINIMAX_M3,
        endpoint: cuteafd_core::SparkEndpoint { ring_depth: None,
            capacity: u32::try_from(prefill.rows).map_err(|_| Refusal::Overflow("MiniMax transport rows"))?,
            ingress: cuteafd_core::SparkIngress::Bf16, partial_reply_bytes: None },
        rings: cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
            slot_bytes: rings.slot_bytes, alignment: rings.alignment } },
        usize::try_from(prefill.waves).map_err(|_| Refusal::Overflow("MiniMax transport waves"))?,
        true, true, SparkIntakeCandidates::Probed, replies.prefill(false), Lifetime::TARGET)?])
}

/// Typed Spark owners at the admitted unit (one prefill + one decode lane).
pub fn spark_executions(prefill: PrefillShape) -> anyhow::Result<cuteafd_core::SparkExecutions> {
    crate::plan::spark_executions::admit_family("minimax_m3",
        &crate::plan::spark_executions::SparkBackend::Fp8.exports(), None, prefill.rows, prefill.waves)
}

/// Balanced whole-layer attention owners for a pair: a contiguous cut that
/// gives GPU1 the head, so the embedding GPU gets the extra layer.
pub fn layer_owners(layers: usize, gpus: usize) -> Vec<u8> {
    if gpus == 1 { return vec![0; layers]; }
    let cut = layers.div_ceil(2);
    (0..layers).map(|layer| u8::from(layer >= cut)).collect()
}

/// One allocation; zero bytes charge nothing.
fn buffer(id: impl Into<String>, location: Location, bytes: u64, lifetime: Lifetime, category: Category,
    group: &str, maximum: u64) -> Option<Component> {
    (bytes > 0).then(|| Component::allocation(Allocation::buffer(id, location, bytes, lifetime, category, group),
        ByteCurve::fixed(bytes, maximum), false, maximum))
}

fn allowance(location: Location, bytes: u64, lifetime: Lifetime, source: String, assumed: bool, maximum: u64)
    -> Option<Component> {
    (bytes > 0).then(|| Component { activation: Activation::always(), bytes: ByteCurve::fixed(bytes, maximum),
        kind: ComponentKind::Allowance(Allowance { location, bytes, lifetime, source, assumed }) })
}

fn with(mut components: Vec<Component>, activation: &Activation) -> Vec<Component> {
    for component in &mut components { component.activation = activation.clone(); }
    components
}

impl MiniMaxInputs {
    pub fn maximum_pages(&self) -> u64 { self.maximum_pool_tokens.div_ceil(PAGE_ROWS) }

    fn modes(&self) -> Vec<LayerMode> {
        self.owners.iter().map(|&gpu| LayerMode::Whole { gpu, ffn: FfnMode::Owner }).collect()
    }

    fn head_gpu(&self) -> u8 { (self.gpus - 1) as u8 }

    fn hop_spec(&self) -> HopSpec {
        HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: self.cfg.hidden as u64 * 2,
            rows: self.prefill.rows.max(u64::from(DECODE_ROWS)), lanes: LANES, entry_gpu: 0,
            head_gpu: self.head_gpu(),
            terminal: TerminalHome::AttentionOwner }
    }

    /// The available onboarding counts: an ordered prefix of the routed layers.
    pub fn counts(&self) -> Vec<usize> { (0..=self.routed.len()).collect() }

    pub fn admit(&self, request: Request) -> Result<AdmittedMiniMax, Refusal> {
        let solved = solve(request, self.budgets.clone(), self)?;
        let placement = self.execution(&solved)?;
        let onboard = placement.onboard_layers;
        Ok(AdmittedMiniMax { placement, onboard, draft_home: solved.decisions.draft_home,
            encoder: solved.decisions.encoder, solved })
    }

    /// Mechanical executor output from the solved decisions; never a re-solve.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Placement, Refusal> {
        let modes = self.modes();
        if solved.decisions.attention != modes || solved.decisions.prefill != self.prefill
            || solved.decisions.lanes != LANES {
            return Err(invalid("resolved execution differs from the prepared facts"));
        }
        let local = solved.decisions.onboard_layers.iter().flatten().copied()
            .collect::<std::collections::BTreeSet<_>>().into_iter().collect::<Vec<_>>();
        let count = local.len();
        if local != self.routed[..count] {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let pair = self.gpus == 2;
        let peaks = self.arena_bytes(count)?;
        let layers = modes.iter().enumerate().map(|(layer, &mode)| LayerAssignment { mode,
            experts: if !self.cfg.moe[layer] { ExpertHome::Dense }
                else if local.contains(&layer) { if pair { ExpertHome::RtxTp2 } else { ExpertHome::RtxWhole { gpu: 0 } } }
                else { ExpertHome::Spark } }).collect();
        let tp2 = (pair && count > 0).then(|| Tp2Range { first: local[0], layers: count, peak_bytes: peaks });
        let expert_ranges = (0..self.gpus).map(|gpu| if pair || count == 0 {
            ExpertRange { first: 0, layers: 0, peak_bytes: 0 }
        } else { ExpertRange { first: local[0], layers: count, peak_bytes: peaks[gpu] } }).collect();
        let spec = self.hop_spec();
        let residual = residual_homes(&modes, &spec).map_err(|error| invalid(error))?;
        let hops = plan_hops(&modes, &spec).map_err(|error| invalid(error))?;
        let placement = Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: if pair { AttentionPlacement::Layers } else { AttentionPlacement::Heads },
            attention_by_kind: vec![(AttentionClass::Gqa,
                if pair { AttentionPlacement::Layers } else { AttentionPlacement::Heads })],
            peer_row_bytes: 0, onboard_layers: count, max_pool_tokens_no_onboard: None, layers,
            movables: solved.decisions.draft_home.map(|gpu| (MovableId::Drafter, gpu)).into_iter().collect(),
            expert_ranges, residual, hops, tp2,
            items: (0..self.gpus).map(|gpu| solved.items(Location::coordinator(gpu as u8)))
                .collect::<Result<_, _>>()? };
        MINIMAX_M3.check(&placement).map_err(|error| invalid(error))?;
        Ok(placement)
    }

    /// Per-GPU expert arena backing at `count` onboarded layers, including
    /// the executor workspace (charged only when a layer is onboarded).
    fn arena_bytes(&self, count: usize) -> Result<[u64; 2], Refusal> {
        let mut out = [0u64; 2];
        if count == 0 { return Ok(out); }
        let LocalPackage::Installed { workspace } = &self.local else {
            return Err(invalid("onboarded layers need the installed fp8-minimax-nvfp4a4 package"));
        };
        for gpu in 0..self.gpus {
            let weights = self.experts[..count].iter().try_fold(0u64, |sum, layer| sum.checked_add(
                if self.gpus == 2 { layer.half[gpu] } else { layer.whole })).ok_or(Refusal::Overflow("MiniMax expert arena"))?;
            out[gpu] = weights.checked_add(workspace[gpu]).ok_or(Refusal::Overflow("MiniMax expert arena"))?;
        }
        Ok(out)
    }

    /// Fixed owners of one coordinator rank that do not depend on any domain.
    fn rank_owners(&self, gpu: usize, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let location = Location::coordinator(gpu as u8);
        let cfg = &self.cfg;
        let rows = u32::try_from(self.prefill.rows).map_err(|_| Refusal::Overflow("MiniMax unit rows"))?;
        let context = u32::try_from(self.context.maximum).map_err(|_| Refusal::Overflow("MiniMax context"))?;
        let sequences = u32::try_from(self.sequences).map_err(|_| Refusal::Overflow("MiniMax sequences"))?;
        let mut out = Vec::new();
        let target = self.targets.get(gpu).ok_or_else(|| invalid("missing target arena"))?;
        out.extend(target.weight_components(maximum)?);
        // Whole-owner MSA layers: KV and index pages on the owner only. Each
        // layer plane is one physical backing that scales with the pool.
        for (layer, &owner) in self.owners.iter().enumerate() {
            if usize::from(owner) != gpu { continue; }
            for (name, stride) in [("keys", 32768u64), ("values", 32768), ("index", 8448)] {
                out.push(Component::allocation(Allocation::buffer(format!("minimax.kv.{layer}.{name}"), location,
                    stride, Lifetime::TARGET, Category::Kv, "MSA paged KV and index records"),
                    ByteCurve::linear_rounded(stride, stride, maximum, SM120_ALLOCATION_GRANULARITY_BYTES),
                    false, maximum));
            }
        }
        // Page tables and lengths for every admitted sequence, at the compiled context.
        let cache = MsaCache::new(1, 1, sequences, context)?;
        for (name, bytes) in cache.buffers()?.into_iter().filter(|(name, _)| !name.starts_with("layer.")) {
            out.extend(buffer(format!("minimax.cache.{gpu}.{name}"), location, bytes, Lifetime::TARGET,
                Category::Kv, "MSA page tables", maximum));
        }
        out.extend(MsaRope::new(cfg, context)?.components(&format!("minimax.rope.{gpu}"), gpu as u8, maximum)?);
        let (msa, _) = MsaPrograms::resolve(cfg, &self.manifest, DECODE_ROWS, rows, context)?;
        let (producers, _) = ProducerPrograms::resolve(&self.manifest, rows)?;
        ForwardPrograms::resolve(&self.manifest, rows)?;
        target.resolve_dense_programs(&self.manifest, rows)?;
        // One decode lane on every rank (its layers' work), then one owner
        // set per in-flight prefill slot (PrefillHooks unit slots).
        let decode = format!("minimax.decode.{gpu}");
        out.extend(UnitLayout::new(cfg, DECODE_ROWS)?.components(&decode, gpu as u8, Lifetime::TARGET, maximum)?);
        out.extend(UnitMetadata::new(DECODE_ROWS)?.components(&format!("{decode}.metadata"), gpu as u8, maximum)?);
        out.extend(msa.workspace_components(&format!("{decode}.attention"), gpu as u8, false, Lifetime::TARGET, maximum)?);
        out.extend(producers.workspace_components(&format!("{decode}.attention"), gpu as u8, Lifetime::TARGET, maximum)?);
        out.extend(target.dense_input(DECODE_ROWS).map_err(refusal)?.components(&format!("{decode}.dense"), gpu as u8, Lifetime::TARGET, maximum)?);
        out.extend(target.dense_workspace(DECODE_ROWS).map_err(refusal)?.components(&format!("{decode}.dense"), gpu as u8, Lifetime::TARGET, maximum)?);
        for slot in 0..self.prefill.waves {
            let prefix = format!("minimax.unit.{gpu}.{slot}");
            out.extend(UnitLayout::new(cfg, rows)?.components(&prefix, gpu as u8, Lifetime::TARGET, maximum)?);
            out.extend(UnitMetadata::new(rows)?.components(&format!("{prefix}.metadata"), gpu as u8, maximum)?);
            out.extend(target.dense_input(rows).map_err(refusal)?.components(&format!("{prefix}.dense"), gpu as u8, Lifetime::TARGET, maximum)?);
            out.extend(target.dense_workspace(rows).map_err(refusal)?.components(&format!("{prefix}.dense"), gpu as u8, Lifetime::TARGET, maximum)?);
            // MSA prefill scratch and producers are per slot: units overlap in flight.
            out.extend(msa.workspace_components(&format!("{prefix}.attention"), gpu as u8, true, Lifetime::TARGET, maximum)?);
            out.extend(producers.workspace_components(&format!("{prefix}.attention"), gpu as u8, Lifetime::TARGET, maximum)?);
        }
        if self.gpus == 2 {
            // TP2 expert exchange (shared `QwenTp2Rows` ABI: BF16 rows, two
            // abortable slots, reduction, peer control and abort word), one
            // unit wide; decode rows fit within it.
            let rows = crate::placement::families::qwen4::QwenTp2Rows::new(cfg.hidden as u64, cfg.topk as u64,
                self.prefill.rows.max(u64::from(DECODE_ROWS)), false).map_err(refusal)?;
            for demand in rows.demands().map_err(refusal)?.into_iter().filter(|d| usize::from(d.gpu) == gpu) {
                out.extend(buffer(format!("minimax.tp2.{gpu}.{}", demand.group.replace(' ', "_")), location,
                    demand.bytes, Lifetime::TARGET, demand.category, &demand.group, maximum));
            }
        }
        out.extend(allowance(location, self.graph_reserve[gpu], Lifetime { first: Phase::Warmup, last: Phase::Growth },
            crate::placement::graph_reserve::source(&format!("minimax_m3.graphs.{gpu}")), true, maximum));
        out.extend(allowance(location, self.code[gpu], Lifetime::RESIDENT,
            "MiniMax selected-program loaded code (estimated module residency)".into(), true, maximum));
        let peak = 1 + crate::push_load::COPY_STREAMS as u64 + 1;
        out.extend(allowance(location, cuda_stream_slab_bytes(peak)?, Lifetime::RESIDENT,
            format!("minimax/driver-stream-slabs/gpu{gpu} ({peak} peak live streams: model + push copies + rotary)"),
            false, maximum));
        for owner in self.mount.get(gpu).copied().unwrap_or_default().allocations(&format!("minimax.mount.{gpu}")) {
            let bytes = ByteCurve::fixed(owner.bytes, maximum);
            out.push(Component::allocation(owner, bytes, false, maximum));
        }
        Ok(out)
    }
}

impl Recipe for MiniMaxInputs {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        let cfg = &self.cfg;
        let maximum = self.maximum_pages();
        if !(1..=2).contains(&self.gpus) || self.spark_ranks > 8 || maximum == 0 || maximum == u64::MAX
            || request.context.maximum != self.context.maximum || self.targets.len() != self.gpus
            || self.owners.len() != cfg.layers || self.graph_reserve.len() != self.gpus
            || self.code.len() != self.gpus || self.experts.len() != self.routed.len()
            || self.spark.len() != self.routed.len() + 1 || self.prefill.rows == 0 || self.prefill.waves == 0
            || self.sequences == 0 {
            return Err(invalid("invalid extent or empty domain"));
        }
        if self.owners.iter().any(|&gpu| usize::from(gpu) >= self.gpus) {
            return Err(invalid("attention owner outside the coordinator ranks"));
        }
        let pair = self.gpus == 2;
        let head = self.head_gpu();
        let mut components = Vec::new();
        for gpu in 0..self.gpus { components.extend(self.rank_owners(gpu, maximum)?); }
        // Head-rank owners: vocabulary head workspace and the shared selector.
        components.extend(HeadLayout::new(cfg, DECODE_ROWS)?.components("minimax.head", head, maximum)?);
        components.extend(serving::ServingOwners {
            selector: serving::SelectorOwners { location: Location::coordinator(head),
                capacity: usize::try_from(self.sequences).map_err(|_| Refusal::Overflow("MiniMax selector"))?
                    .max(DECODE_ROWS as usize), vocab: cfg.vocab_size, device_sampling: self.device_sampling },
            prefix: self.host_prefix }.components(maximum)?);
        if let Some(geometry) = &self.expert_usage {
            let devices = (0..self.gpus.min(geometry.routed_layers.len())).map(|gpu| gpu as u8).collect::<Vec<_>>();
            components.extend(crate::placement::expert_usage::components(geometry, &devices, maximum)?);
        }
        // Transport: coordinator intake planes are fixed owners on the link's
        // GPU; host/pinned candidates join the host budget as one envelope.
        let mut contract_variables = Vec::new();
        let modes = self.modes();
        for (layer, &mode) in modes.iter().enumerate() {
            contract_variables.push(Variable::Attention { layer, values: vec![mode] });
        }
        contract_variables.push(Variable::Prefill(vec![self.prefill]));
        contract_variables.push(Variable::Lanes(vec![LANES]));
        // Residual hop receive slots implied by the fixed owners.
        let spec = self.hop_spec();
        let hops = plan_hops(&modes, &spec).map_err(|error| invalid(error))?;
        let hop_bytes = crate::placement::hop_buffer_bytes(&hops, &spec, self.gpus)
            .ok_or(Refusal::Overflow("MiniMax residual hop buffers"))?;
        for (gpu, &bytes) in hop_bytes.iter().enumerate() {
            components.extend(buffer(format!("minimax.hops.{gpu}"), Location::coordinator(gpu as u8), bytes,
                Lifetime::TARGET, Category::Transport, "residual hops", maximum));
        }
        // Onboarding: one ordered count profile; destination both GPUs (TP2) or GPU0.
        let counts = self.counts();
        let onboard = contract_variables.len();
        contract_variables.push(Variable::Onboard { layers: self.routed.clone(),
            destination: if pair { vec![0, 1] } else { vec![0] }, counts: counts.clone() });
        if self.spark_ranks > 0 {
            contract_variables.push(Variable::ExpertHome { layers: self.routed.clone(),
                rank_sets: vec![(0..self.spark_ranks as u8).collect()] });
        }
        let mut requires = Vec::new();
        for (option, &count) in counts.iter().enumerate() {
            let activation = Activation::choice(onboard, option);
            let local = count == 0 || matches!(self.local, LocalPackage::Installed { .. });
            let remote = count == self.routed.len() || self.spark_ranks > 0;
            // A linked executor needs at least one remote layer; all-local is a separate qualification.
            let linked = self.spark_ranks == 0 || count < self.routed.len();
            if !local || !remote || !linked {
                requires.push(Requires { when: Choice { variable: onboard, option }, any_of: vec![] });
                continue;
            }
            let arenas = self.arena_bytes(count)?;
            for gpu in 0..self.gpus {
                if arenas[gpu] == 0 { continue; }
                let weights = arenas[gpu] - match &self.local {
                    LocalPackage::Installed { workspace } => workspace[gpu], LocalPackage::Missing(_) => 0 };
                let group = if pair { format!("TP2 expert layer halves 0..{count}") }
                    else { format!("resident routed layers 0..{count}") };
                let id = format!("minimax.experts.{gpu}.{count}");
                let location = Location::coordinator(gpu as u8);
                let owner = Allocation::buffer(&id, location, arenas[gpu], Lifetime::RESIDENT, Category::Experts, group);
                let mut resident = Component::resident_weight(
                    ResidentWeight::from_allocation(&owner, ResidentWeights(weights)), ByteCurve::fixed(weights, maximum));
                resident.activation = activation.clone();
                components.push(resident);
                let mut arena = Component::allocation(owner, ByteCurve::fixed(arenas[gpu], maximum), false, maximum);
                arena.activation = activation.clone();
                components.push(arena);
            }
            if count < self.routed.len() {
                for (rank, charge) in self.spark[count].iter().enumerate() {
                    let location = Location { device: Device::Spark(rank as u8), domain: Domain::Managed };
                    for (kind, group, bytes, category) in [
                        ("experts", "routed experts (native NVFP4 intermediate slices)", charge.experts, Category::Experts),
                        ("owners", "Spark rank I/O and execution owners", charge.owners, Category::Workspace),
                        ("startup", cuteafd_core::memory_layout::spark_runtime_reserve_group(), charge.runtime, Category::Runtime),
                    ] {
                        components.extend(buffer(format!("minimax.spark.{rank}.{kind}.{count}"), location, bytes,
                            Lifetime::RESIDENT, category, group, maximum).map(|mut c| { c.activation = activation.clone(); c }));
                    }
                    if charge.experts > 0 {
                        components.push(Component { activation: activation.clone(),
                            bytes: ByteCurve::fixed(charge.experts, maximum),
                            kind: ComponentKind::DriverPageTables { model: DriverPageTableModel::Sm121Arena, location,
                                lifetime: Lifetime::RESIDENT, backing: format!("minimax.spark.{rank}.experts.{count}") } });
                    }
                }
                for link in &self.links {
                    components.extend(with(link.components(maximum)?, &activation));
                }
            }
        }
        // Optional bundled drafter: own arena on the head GPU (DraftHome domain).
        if let Some(draft) = &self.drafter {
            contract_variables.push(Variable::DraftHome(vec![head]));
            let location = Location::coordinator(head);
            let lifetime = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            components.extend(weight_arena_components(&draft.arena, location, lifetime, maximum, MemoryKind::DrafterWeights)?);
            for (name, bytes) in &draft.scratch {
                components.extend(buffer(format!("minimax.draft.{name}"), location, *bytes, lifetime,
                    Category::Drafter, "bundled dSpark scratch", maximum));
            }
            if draft.state_per_page > 0 {
                components.push(Component::allocation(Allocation::buffer("minimax.draft.state", location,
                    draft.state_per_page, lifetime, Category::Drafter, "bundled dSpark per-page state"),
                    ByteCurve::linear(draft.state_per_page, draft.state_per_page, maximum), false, maximum));
            }
        }
        // Optional bundled vision tower: own arena; `Encoder` domain [on, off].
        if let Some(vision) = &self.vision {
            let variable = contract_variables.len();
            contract_variables.push(Variable::Encoder(vec![0, 1]));
            let on = Activation::choice(variable, 0);
            let location = Location::coordinator(head);
            components.extend(with(weight_arena_components(&vision.arena, location, Lifetime::RESIDENT, maximum,
                MemoryKind::Weights)?, &on));
            components.extend(buffer("minimax.vision.scratch", location, vision.scratch, Lifetime::TARGET,
                Category::Workspace, "vision tower scratch", maximum).map(|mut c| { c.activation = on.clone(); c }));
            components.extend(buffer("minimax.host.media", Location::HOST, vision.media_cache,
                Lifetime { first: Phase::Warmup, last: Phase::Growth }, Category::Embedding, "media/embedding-cache",
                maximum).map(|mut c| { c.activation = on.clone(); c }));
        }
        // BLAS: the BF16 vocabulary head borrows one serving-thread pool.
        if self.head.bf16() {
            components.extend(serving::blas_components("sm_120", &[serving::BlasOwners { name: "MiniMax target head".into(),
                process: "coordinator".into(), thread: "minimax/blas/serve".into(), location: Location::coordinator(head),
                gemm: self.head.gemm(), lt: self.head.lt(), lifetime: Lifetime::TARGET, activation: Activation::always() }],
                maximum, PhaseOrder::TARGET_THEN_DRAFT)?);
        }
        components.extend(allowance(Location::PINNED, 64 << 20, Lifetime::RESIDENT,
            "MiniMax reusable H2D staging (estimated: library-wide sync-h2d backing, cold golden reconciled)".into(),
            true, maximum));
        fixed_request_tails(&mut components, maximum, &[])?;
        let mut contract = Contract { floor_fallback: None, phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: (self.spark_ranks > 0).then(|| SparkReplyFormats {
                prefill: Some(request.spark_reply_formats.prefill(false)),
                decode: Some(request.spark_reply_formats.decode()) }),
            resident_weight_limit: request.resident_weight_limit, page_rows: PAGE_ROWS, pool_page_multiple: 1,
            maximum_pages: maximum, gpus: self.gpus, variables: contract_variables, requires, components };
        contract.add_driver_page_tables();
        Ok(contract)
    }
}

/// Coordinator budgets: physical card identity and an independent cap; the
/// named reserve stays free inside it. `context` is the planned CUDA context.
pub fn planned_budget(gpu: u8, physical: u64, logical: u64, context: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)], physical_bytes: physical,
        logical_bytes: logical, headroom: crate::placement::reserve::gpu("minimax_m3"),
        baseline: Baseline::Forecast { occupied: context, source: "CUDA context and modules on the coordinator GPU".into(),
            assumed: false } }
}

/// The measured edge: one CUDA sample before weights.
pub fn measured_budget(gpu: u8, physical: u64, logical: u64, free: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)], physical_bytes: physical,
        logical_bytes: logical, headroom: crate::placement::reserve::gpu("minimax_m3"),
        baseline: Baseline::Sampled { free, phase: Phase::Load, source: "MiniMax pre-weight CUDA sample".into() } }
}

pub fn spark_budget(rank: u8, bytes: u64) -> Budget {
    Budget { id: format!("spark{rank}"), locations: vec![Location { device: Device::Spark(rank), domain: Domain::Managed }],
        physical_bytes: bytes, logical_bytes: bytes, headroom: 0,
        baseline: Baseline::Forecast { occupied: 0, source: "Spark worker arena owned by the serving process".into(),
            assumed: false } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_runs_whole_owner_layers_only() {
        assert!(MINIMAX_M3.runs(WHOLE) && MINIMAX_M3.runs(LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }));
        assert!(!MINIMAX_M3.runs(LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }));
        assert!(!MINIMAX_M3.runs(LayerMode::HeadSplit));
        assert!(!MINIMAX_M3.runs(LayerMode::ContextSplit));
        assert_eq!(MINIMAX_M3.attention, AttentionPlacement::Layers);
        assert_eq!(layer_owners(60, 1), vec![0; 60]);
        let pair = layer_owners(60, 2);
        assert_eq!(pair.iter().filter(|&&gpu| gpu == 0).count(), 30);
        assert!(pair.windows(2).filter(|w| w[0] != w[1]).count() == 1 && pair[59] == 1);
    }

    #[test]
    fn spark_owners_cover_one_prefill_and_one_decode_lane() {
        let executions = spark_executions(PrefillShape { rows: 2752, waves: 3 }).unwrap();
        assert_eq!(executions.decode.len(), 1);
        assert_eq!(executions.prefill.unit_rows, 2752);
        assert_eq!(executions.prefill.buffered_units, 3);
    }
}
