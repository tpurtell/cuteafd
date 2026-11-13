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
//! - MSA KV + index pool pages (whole owners, or half-KV/replicated index for heads);
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
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };

/// Shared constructor/admission capacities. Native decode consumes one token
/// per request; multi-token verification uses packed prefill units instead.
/// Verify never launches more than one prefill unit's rows; the constructor
/// must partition, preserve causal sequence order and consume head tiles before
/// reusing a unit. Speculative batch totals are not dispatcher extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestCapacity {
    pub decode: UnitMetadata,
    pub prefill: UnitMetadata,
}
impl RequestCapacity {
    pub fn new(concurrency: u64, unit_rows: u64) -> Result<Self, Refusal> {
        cuteafd_core::concurrency::validate("minimax_m3", concurrency).map_err(invalid)?;
        let requests = u32::try_from(concurrency).map_err(|_| Refusal::Overflow("MiniMax requests"))?;
        let rows = u32::try_from(unit_rows).map_err(|_| Refusal::Overflow("MiniMax unit rows"))?;
        Ok(Self { decode: UnitMetadata::for_requests(requests, requests)?,
            prefill: UnitMetadata::for_requests(rows, requests.min(rows))? })
    }

    /// Native expert workspace and serial dispatch capacity for either mode.
    pub fn execution_rows(self) -> u32 { self.decode.rows.max(self.prefill.rows) }

    pub fn metadata(self, prefill: bool) -> UnitMetadata {
        if prefill { self.prefill } else { self.decode }
    }

    /// Retained ordered H2D payloads, one per mutable metadata destination.
    /// Decode batch IDs/offsets are startup iotas; descales are startup ones.
    pub fn metadata_uploads(self, prefill: bool) -> Vec<(&'static str, u64)> {
        self.metadata(prefill).buffers().into_iter().filter(|(name, _)| matches!(*name,
            "positions" | "slots" | "causal_lengths" | "cache_lengths")
            || (prefill && matches!(*name, "batch_ids" | "query_offsets"))).collect()
    }

    /// Globals only; the constructor derives the physical per-layer pool separately.
    pub fn cache(self, context: u32) -> Result<MsaCache, Refusal> {
        MsaCache::new(1, 1, self.decode.requests, context)
    }
}

/// Stable recipe/runtime owner identity. `id` is the full planner identity;
/// `inventory_name` omits the `minimax.` prefix supplied by runtime Inventory.
/// Experts/Spark identities are selection groups, not physical backing IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    Target { gpu: usize },
    Kv { layer: usize, plane: &'static str },
    HeadsKv { gpu: usize, layer: usize, plane: &'static str },
    Cache { gpu: usize },
    DecodeTable { gpu: usize },
    DecodeTokenIds { gpu: usize },
    Rope { gpu: usize },
    Decode { gpu: usize },
    Unit { gpu: usize, slot: u64 },
    Tp2 { gpu: usize },
    Hops { gpu: usize },
    Head,
    Experts { gpu: usize, count: usize },
    Spark { rank: usize, kind: &'static str, count: usize },
    Mount { gpu: usize },
    PrefixH2d { gpu: usize, slot: u64, plane: &'static str },
}

impl Owner {
    pub fn id(self) -> String {
        match self {
            Self::Target { gpu } => format!("minimax/target/{gpu}"),
            Self::Kv { layer, plane } => format!("minimax.kv.{layer}.{plane}"),
            Self::HeadsKv { gpu, layer, plane } => format!("minimax.kv.heads.{gpu}.{layer}.{plane}"),
            Self::Cache { gpu } => format!("minimax.cache.{gpu}"),
            Self::DecodeTable { gpu } => format!("minimax.cache.{gpu}.decode_table"),
            Self::DecodeTokenIds { gpu } => format!("minimax.decode.token_ids.{gpu}"),
            Self::Rope { gpu } => format!("minimax.rope.{gpu}"),
            Self::Decode { gpu } => format!("minimax.decode.{gpu}"),
            Self::Unit { gpu, slot } => format!("minimax.unit.{gpu}.{slot}"),
            Self::Tp2 { gpu } => format!("minimax.tp2.{gpu}"),
            Self::Hops { gpu } => format!("minimax.hops.{gpu}"),
            Self::Head => "minimax.head".into(),
            Self::Experts { gpu, count } => format!("minimax.experts.{gpu}.{count}"),
            Self::Spark { rank, kind, count } => format!("minimax.spark.{rank}.{kind}.{count}"),
            Self::Mount { gpu } => format!("minimax.mount.{gpu}"),
            Self::PrefixH2d { gpu, slot, plane } => format!("minimax.prefix.h2d.{gpu}.{slot}.{plane}"),
        }
    }
    /// Name passed to an Inventory constructed with the `minimax` prefix.
    /// Target arenas carry their own canonical backing ID rather than an alias.
    pub fn inventory_name(self) -> Option<String> {
        self.id().strip_prefix("minimax.").map(str::to_owned)
    }

}

/// Qualified whole-layer execution remains the automatic policy. Explicit TP2
/// heads has a separate planning descriptor and a typed serving refusal until
/// narrow-head native kernels qualify.
pub const MINIMAX_M3: ExecutorModes = ExecutorModes { family: "minimax_m3",
    modes: &[WHOLE, LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }], hops: true,
    attention: AttentionPlacement::Layers };

const HEADS_PLAN: ExecutorModes = ExecutorModes { family: "minimax_m3",
    modes: &[LayerMode::HeadSplit], hops: false, attention: AttentionPlacement::Heads };

/// Planning facts may describe unqualified geometry; serving must reject it.
#[derive(Debug, thiserror::Error)]
pub enum ServingRefusal {
    #[error("narrow-head MiniMax kernels not qualified")]
    NarrowHeadsNotQualified,
    #[error(transparent)]
    Plan(#[from] Refusal),
}

fn invalid(detail: impl std::fmt::Display) -> Refusal {
    Refusal::InvalidFact(format!("MiniMax component contract: {detail}"))
}

fn refusal(error: anyhow::Error) -> Refusal { invalid(format!("{error:#}")) }

/// Source route selected by the shared push policy. A failed push may fall
/// back to mount only with this same canonical final layout and identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertStorage { Push, Mount }

/// A layer's final planes are offsets in its rank's one physical backing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLayerBacking {
    pub layer: usize,
    pub region: std::ops::Range<usize>,
    pub slicing: crate::formats::fp8_experts::Slicing,
    pub planes: [std::ops::Range<usize>; 6],
}

/// Passive view of the shared loaders' canonical backing. Never injects an
/// identity or changes their packing. Both push and mount receipts use `arena`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBacking {
    pub gpu: usize,
    pub physical_device: i32,
    pub count: usize,
    pub storage: ExpertStorage,
    /// Full alias prefix; runtime strips only its Inventory's `minimax.` once.
    pub inventory_prefix: String,
    pub arena: WeightArenaLayout,
    pub layers: Vec<ExpertLayerBacking>,
    /// Host-only post-delivery validation, not part of the device backing.
    pub validation_bytes: usize,
}

/// Header/source inventory is linear in the routed layers, not in all their
/// onboarding prefixes. Prefix materialization below only packs cached regions.
#[derive(Debug)]
pub struct ExpertInventory {
    ranks: Vec<Vec<ExpertLayerInventory>>,
}

#[derive(Debug)]
struct ExpertLayerInventory {
    layer: usize,
    regions: Vec<(String, usize)>,
    has_sources: bool,
    push_sources: bool,
}

impl ExpertInventory {
    pub fn new(tensors: &crate::formats::fp8_experts::Fp8ExpertTensors, layers: &[usize],
        gpus: usize) -> anyhow::Result<Self> {
        use crate::formats::fp8_experts::{Fp8Projection, Slicing};
        anyhow::ensure!((1..=2).contains(&gpus) && !layers.is_empty()
            && layers.windows(2).all(|pair| pair[0] < pair[1]), "invalid MiniMax expert selection");
        let tensors = tensors.for_coordinator();
        let mut ranks = (0..gpus).map(|_| Vec::with_capacity(layers.len())).collect::<Vec<_>>();
        let mut shards = std::collections::BTreeMap::new();
        for &layer in layers {
            let view = tensors.for_layer_metadata(layer)?;
            let mut has_sources = false;
            let mut push_sources = true;
            for expert in 0..view.shape().experts {
                for projection in Fp8Projection::ALL {
                    for name in view.projection_source_names(layer, expert, projection) {
                        let path = view.push_source(&name)?.0.path;
                        has_sources = true;
                        push_sources &= *shards.entry(path.clone())
                            .or_insert_with(|| crate::push_load::is_push_shard(&path));
                    }
                }
            }
            for (gpu, rank) in ranks.iter_mut().enumerate() {
                rank.push(ExpertLayerInventory { layer, has_sources, push_sources,
                    regions: crate::serving_capacity::fp8_expert_regions(&tensors, &[layer],
                        gpus, gpu, Slicing::Blocks(128))? });
            }
        }
        Ok(Self { ranks })
    }

    /// Count zero has no backing; all other selections retain canonical shared
    /// arena IDs, plane order and alignment for the supplied physical devices.
    pub fn prefixes(&self, physical_devices: &[i32]) -> anyhow::Result<Vec<Vec<ExpertBacking>>> {
        anyhow::ensure!(physical_devices.len() == self.ranks.len()
            && physical_devices.iter().all(|&device| device >= 0)
            && physical_devices.iter().collect::<std::collections::BTreeSet<_>>().len() == self.ranks.len(),
            "invalid MiniMax expert physical devices");
        (0..=self.ranks[0].len()).map(|count| {
            if count == 0 { return Ok(Vec::new()); }
            physical_devices.iter().enumerate().map(|(gpu, &device)| self.backing(count, gpu, device))
                .collect()
        }).collect()
    }

    fn backing(&self, count: usize, gpu: usize, physical_device: i32) -> anyhow::Result<ExpertBacking> {
        use crate::formats::fp8_experts::Slicing;
        anyhow::ensure!(gpu < self.ranks.len() && physical_device >= 0
            && count > 0 && count <= self.ranks[gpu].len(), "invalid MiniMax expert prefix");
        let selected = &self.ranks[gpu][..count];
        let regions = selected.iter().flat_map(|layer| layer.regions.iter().cloned()).collect::<Vec<_>>();
        let arena = WeightArenaLayout::new(crate::serving_capacity::expert_weight_arena_id(
            "fp8", physical_device, &regions)?, regions)?;
        let mut views = Vec::with_capacity(count);
        let mut push_compatible = true;
        for cached in selected {
            let layer = cached.layer;
            let planes = std::array::from_fn(|plane| arena.view(&format!("{layer}.{plane}"))
                .map_or(0..0, |view| view.offset()..view.offset() + view.bytes()));
            let first = planes.iter().filter(|plane| !plane.is_empty()).map(|plane| plane.start).min()
                .ok_or_else(|| anyhow::anyhow!("empty MiniMax expert layer"))?;
            let end = planes.iter().map(|plane| plane.end).max().unwrap();
            // RankPlan rejects inter-plane padding; admission must choose mount.
            push_compatible &= cached.push_sources
                && end - first == planes.iter().map(|plane| plane.len()).sum::<usize>();
            views.push(ExpertLayerBacking { layer, region: first..end,
                slicing: Slicing::Blocks(128), planes });
        }
        let storage = if push_compatible && selected.iter().any(|layer| layer.has_sources) {
            ExpertStorage::Push
        } else { ExpertStorage::Mount };
        Ok(ExpertBacking { gpu, physical_device, count, storage,
            inventory_prefix: format!("minimax.experts.{gpu}.{count}"), arena,
            layers: views, validation_bytes: 0 })
    }
}

impl ExpertBacking {
    pub fn plan(tensors: &crate::formats::fp8_experts::Fp8ExpertTensors, layers: &[usize],
        gpus: usize, gpu: usize, physical_device: i32) -> anyhow::Result<Self> {
        ExpertInventory::new(tensors, layers, gpus)?.backing(layers.len(), gpu, physical_device)
    }
}

/// The installed coordinator fp8moe package (`fp8-minimax-nvfp4a4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPackage {
    /// Executor workspace per GPU at the admitted unit (TP1 or TP2 layout).
    Installed { scratch: u64, output: u64 },
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
    pub attention: AttentionPlacement,
    pub spark_ranks: usize,
    pub context: Context,
    /// Largest pool the build indexes (KV max), in tokens.
    pub maximum_pool_tokens: u64,
    pub prefill: PrefillShape,
    /// Boot concurrency C: per-sequence page tables, lane members, selector.
    pub sequences: u64,
    /// Per-rank target arenas (one per coordinator GPU).
    pub targets: Vec<CoordinatorWeights>,
    /// Whole-layer owner map; empty for TP2 heads (both ranks execute every layer).
    pub owners: Vec<u8>,
    pub manifest: std::sync::Arc<serde_json::Value>,
    /// Routed layer order (the onboarding prefix is `routed[..count]`).
    pub routed: Vec<usize>,
    /// One backing per GPU per ordered prefix; count zero owns none.
    pub expert_backings: Vec<Vec<ExpertBacking>>,
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
    /// Selected startup route for each onboarding count; None means mount.
    pub push_host_envelopes: Vec<Option<crate::push_load::PushHostEnvelope>>,
    /// Mount source per rank (Load only), header-bounded.
    pub mount: Vec<crate::shared::dense_read::MountLoad>,
}

/// The admitted answer. Plan and runtime read the same fields.
#[derive(Debug, Clone)]
pub struct AdmittedMiniMax {
    pub placement: Placement,
    pub solved: ResolvedPlan,
    pub onboard: usize,
    /// The exact selected rank backings; runtime never derives IDs or offsets.
    pub expert_backings: Vec<ExpertBacking>,
    /// Exact selected push receive owners; None means mount loading.
    pub push_host_envelope: Option<crate::push_load::PushHostEnvelope>,
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

fn metadata_upload_components(capacity: RequestCapacity, prefill: bool, prefix: &str, maximum: u64)
    -> Vec<Component> {
    capacity.metadata_uploads(prefill).into_iter().filter_map(|(name, bytes)|
        buffer(format!("{prefix}.metadata.{name}.upload"), Location::PINNED, bytes,
            Lifetime::TARGET, Category::Staging, "metadata ordered H2D staging", maximum)).collect()
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

    pub fn request_capacity(&self) -> Result<RequestCapacity, Refusal> {
        RequestCapacity::new(self.sequences, self.prefill.rows)
    }

    pub fn heads(&self) -> bool { self.gpus == 2 && self.attention == AttentionPlacement::Heads }

    fn modes(&self) -> Vec<LayerMode> {
        if self.heads() { vec![LayerMode::HeadSplit; self.cfg.layers] }
        else { self.owners.iter().map(|&gpu| LayerMode::Whole { gpu, ffn: FfnMode::Owner }).collect() }
    }

    fn validate_attention(&self) -> Result<(), Refusal> {
        let valid = match (self.gpus, self.attention) {
            (1, AttentionPlacement::Heads) | (2, AttentionPlacement::Layers) =>
                self.owners == layer_owners(self.cfg.layers, self.gpus),
            (2, AttentionPlacement::Heads) => self.owners.is_empty(),
            _ => false,
        };
        if !valid { return Err(invalid("attention must be contiguous whole-layer owners or explicit TP2 heads")); }
        Ok(())
    }

    /// This milestone has no embedding-home choice: the only embedding view
    /// aliases coordinator GPU 0's target arena, never host or another rank.
    fn validate_embedding(&self) -> Result<(), Refusal> {
        let bytes = self.cfg.vocab_size.checked_mul(self.cfg.hidden).and_then(|n| n.checked_mul(2))
            .ok_or(Refusal::Overflow("MiniMax embedding"))?;
        let name = "language_model.model.embed_tokens.weight";
        if self.targets.iter().enumerate().any(|(gpu, target)| target.rank != gpu
            || target.arena.view(name).map(|view| view.bytes()) != (gpu == 0).then_some(bytes)) {
            return Err(invalid("embedding home = coordinator GPU 0"));
        }
        Ok(())
    }

    pub fn admit_serving(&self, request: Request) -> Result<AdmittedMiniMax, ServingRefusal> {
        self.validate_attention()?;
        if self.heads() { return Err(ServingRefusal::NarrowHeadsNotQualified); }
        Ok(self.admit(request)?)
    }

    fn head_gpu(&self) -> u8 { (self.gpus - 1) as u8 }

    fn hop_spec(&self) -> HopSpec {
        HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: self.cfg.hidden as u64 * 4,
            rows: self.prefill.rows.max(self.sequences), lanes: LANES, entry_gpu: 0,
            head_gpu: self.head_gpu(),
            terminal: if self.heads() { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } }
    }

    /// The available onboarding counts: an ordered prefix of the routed layers.
    pub fn counts(&self) -> Vec<usize> { (0..=self.routed.len()).collect() }

    pub fn admit(&self, request: Request) -> Result<AdmittedMiniMax, Refusal> {
        let solved = solve(request, self.budgets.clone(), self)?;
        let placement = self.execution(&solved)?;
        let onboard = placement.onboard_layers;
        Ok(AdmittedMiniMax { placement, onboard, expert_backings: self.expert_backings[onboard].clone(),
            push_host_envelope: self.push_host_envelopes[onboard].clone(), draft_home: solved.decisions.draft_home,
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
            attention_placement: self.attention,
            attention_by_kind: vec![(AttentionClass::Gqa, self.attention)],
            peer_row_bytes: 0, onboard_layers: count, max_pool_tokens_no_onboard: None, layers,
            movables: solved.decisions.draft_home.map(|gpu| (MovableId::Drafter, gpu)).into_iter().collect(),
            expert_ranges, residual, hops, tp2,
            items: (0..self.gpus).map(|gpu| solved.items(Location::coordinator(gpu as u8)))
                .collect::<Result<_, _>>()? };
        // The heads descriptor validates planning geometry, not native qualification.
        let descriptor = if self.heads() { HEADS_PLAN } else { MINIMAX_M3 };
        descriptor.check(&placement).map_err(invalid)?;
        Ok(placement)
    }

    /// Executor peak is derived from the canonical backing plus the same
    /// separately owned scratch/output the runtime retains.
    fn arena_bytes(&self, count: usize) -> Result<[u64; 2], Refusal> {
        let mut out = [0u64; 2];
        if count == 0 { return Ok(out); }
        let LocalPackage::Installed { scratch, output } = self.local else {
            return Err(invalid("onboarded layers need the installed fp8-minimax-nvfp4a4 package"));
        };
        for backing in &self.expert_backings[count] {
            out[backing.gpu] = (backing.arena.bytes() as u64).checked_add(scratch)
                .and_then(|bytes| bytes.checked_add(output)).ok_or(Refusal::Overflow("MiniMax expert peak"))?;
        }
        Ok(out)
    }

    /// Pure execution inventory before allocation-tail and driver-page-table
    /// finalization. Target/head weights, expert choices and transport stay with
    /// their own constructors; `prepare` finalizes the combined set once.
    pub fn execution_components(&self) -> Result<Vec<Component>, Refusal> {
        let maximum = self.maximum_pages();
        if !(1..=2).contains(&self.gpus) || maximum == 0 || maximum == u64::MAX
            || self.targets.len() != self.gpus
            || self.graph_reserve.len() != self.gpus || self.code.len() != self.gpus {
            return Err(invalid("invalid execution inventory extent"));
        }
        self.validate_attention()?;
        self.validate_embedding()?;
        cuteafd_core::concurrency::validate("minimax_m3", self.sequences).map_err(invalid)?;
        let mut out = Vec::new();
        // Host decode IDs enter only the embedding rank; peers receive residuals.
        let token_ids = Owner::DecodeTokenIds { gpu: 0 }.id();
        let token_bytes = self.sequences.checked_mul(4).ok_or(Refusal::Overflow("MiniMax decode token IDs"))?;
        out.extend(buffer(token_ids.clone(), Location::coordinator(0), token_bytes, Lifetime::TARGET,
            Category::Workspace, "decode token IDs", maximum));
        out.extend(buffer(format!("{token_ids}.upload"), Location::PINNED, token_bytes, Lifetime::TARGET,
            Category::Staging, "decode token IDs ordered H2D staging", maximum));
        for gpu in 0..self.gpus { out.extend(self.rank_execution_components(gpu, maximum)?); }
        for owner in self.host_prefix.allocations("host.prefix", serving::ServingOwners::LIFETIME)? {
            let bytes = ByteCurve::fixed(owner.bytes, maximum);
            out.push(Component::allocation(owner, bytes, false, maximum));
        }
        if let Some(geometry) = &self.expert_usage {
            let devices = (0..self.gpus.min(geometry.routed_layers.len())).map(|gpu| gpu as u8).collect::<Vec<_>>();
            out.extend(crate::placement::expert_usage::components(geometry, &devices, maximum)?);
        }
        // The library releases this shared startup copy chunk before readiness.
        out.extend(buffer("staging/sync-h2d", Location::PINNED,
            cuteafd_ffi::SYNC_H2D_CHUNK_BYTES as u64,
            Lifetime { first: Phase::Load, last: Phase::Warmup }, Category::Staging,
            "library-wide synchronous H2D startup chunk", maximum));
        Ok(out)
    }

    fn rank_execution_components(&self, gpu: usize, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let location = Location::coordinator(gpu as u8);
        let cfg = &self.cfg;
        let capacity = self.request_capacity()?;
        let rows = capacity.prefill.rows;
        let decode_rows = capacity.decode.rows;
        let context = u32::try_from(self.context.maximum).map_err(|_| Refusal::Overflow("MiniMax context"))?;
        let mut out = Vec::new();
        let target = self.targets.get(gpu).ok_or_else(|| invalid("missing target arena"))?;
        // The native cache descriptor is the physical source of page strides.
        let cache = if self.heads() {
            MsaCache::for_heads(&HeadsLayout::new(cfg, gpu as u8, rows)?, 1,
                capacity.decode.requests, context)?
        } else { MsaCache::new(1, 1, capacity.decode.requests, context)? };
        let buffers = cache.buffers()?;
        for layer in 0..cfg.layers {
            if !self.heads() && usize::from(self.owners[layer]) != gpu { continue; }
            for plane in ["keys", "values", "index"] {
                let stride = buffers.iter().find(|(name, _)| name == &format!("layer.0.{plane}"))
                    .ok_or_else(|| invalid("native cache plane missing"))?.1;
                let owner = if self.heads() { Owner::HeadsKv { gpu, layer, plane } }
                    else { Owner::Kv { layer, plane } };
                out.push(Component::allocation(Allocation::buffer(owner.id(), location,
                    stride, Lifetime::TARGET, Category::Kv, "MSA paged KV and index records"),
                    ByteCurve::linear_rounded(stride, stride, maximum, SM120_ALLOCATION_GRANULARITY_BYTES),
                    false, maximum));
            }
        }
        // Page tables and lengths for every admitted sequence, at the compiled context.
        for (name, bytes) in cache.buffers()?.into_iter().filter(|(name, _)| !name.starts_with("layer.")) {
            out.extend(buffer(format!("{}.{name}", Owner::Cache { gpu }.id()), location, bytes, Lifetime::TARGET,
                Category::Kv, "MSA page tables", maximum));
        }
        // Compact decode tables have their own retained full-table upload,
        // separate from prefix restores into individual page-table slots.
        out.extend(buffer(format!("{}.upload", Owner::DecodeTable { gpu }.id()), Location::PINNED,
            u64::from(cache.requests) * u64::from(cache.table_width) * 4, Lifetime::TARGET,
            Category::Staging, "decode page table ordered H2D staging", maximum));
        // Ordered H2D staging is keyed by destination and copy stream, not a
        // ring: each retained request table and length has its own pinned owner.
        // Shared pinned accounting charges exact payload; no device rounding.
        for slot in 0..self.sequences {
            for (plane, bytes) in [("table", u64::from(cache.table_width) * 4), ("length", 4)] {
                out.extend(buffer(Owner::PrefixH2d { gpu, slot, plane }.id(), Location::PINNED, bytes,
                    Lifetime::TARGET, Category::Staging, "prefix ordered H2D staging", maximum));
            }
        }
        out.extend(MsaRope::new(cfg, context)?.components(&Owner::Rope { gpu }.id(), gpu as u8, maximum)?);
        let (msa, _) = MsaPrograms::resolve(cfg, &self.manifest, decode_rows, rows, context)?;
        let (producers, _) = ProducerPrograms::resolve(&self.manifest, capacity.execution_rows())?;
        ForwardPrograms::resolve(&self.manifest, capacity.execution_rows())?;
        target.resolve_dense_programs(&self.manifest, capacity.execution_rows())?;
        // One decode lane on every rank (its layers' work), then one owner
        // set per in-flight prefill slot (PrefillHooks unit slots).
        let decode = Owner::Decode { gpu }.id();
        if self.heads() {
            out.extend(HeadsLayout::new(cfg, gpu as u8, decode_rows)?.components(&decode, Lifetime::TARGET, maximum)?);
        } else {
            out.extend(UnitLayout::new(cfg, decode_rows)?.components(&decode, gpu as u8, Lifetime::TARGET, maximum)?);
        }
        out.extend(capacity.decode.components(&format!("{decode}.metadata"), gpu as u8, maximum)?);
        out.extend(metadata_upload_components(capacity, false, &decode, maximum));
        out.extend(msa.workspace_components(&format!("{decode}.attention"), gpu as u8, false, Lifetime::TARGET, maximum)?);
        out.extend(producers.workspace_components(&format!("{decode}.attention"), gpu as u8, Lifetime::TARGET, maximum)?);
        out.extend(target.dense_input(decode_rows).map_err(refusal)?.components(&format!("{decode}.dense"), gpu as u8, Lifetime::TARGET, maximum)?);
        out.extend(target.dense_workspace(decode_rows).map_err(refusal)?.components(&format!("{decode}.dense"), gpu as u8, Lifetime::TARGET, maximum)?);
        for slot in 0..self.prefill.waves {
            let prefix = Owner::Unit { gpu, slot }.id();
            let unit = UnitLayout::new(cfg, rows)?;
            if self.heads() {
                out.extend(HeadsLayout::new(cfg, gpu as u8, rows)?.components(&prefix, Lifetime::TARGET, maximum)?);
            } else { out.extend(unit.components(&prefix, gpu as u8, Lifetime::TARGET, maximum)?); }
            if gpu == 0 {
                // Host tokens enter only the entry rank's prefill units. The
                // stable U32 upload view shares route_ids' address, not its
                // larger top-k extent; device-ID decode/verify uploads nothing.
                out.extend(buffer(format!("{prefix}.route_ids.upload"), Location::PINNED,
                    u64::from(unit.rows) * 4, Lifetime::TARGET, Category::Staging,
                    "entry prefill ordered token IDs", maximum));
            }
            out.extend(capacity.prefill.components(&format!("{prefix}.metadata"), gpu as u8, maximum)?);
            out.extend(metadata_upload_components(capacity, true, &prefix, maximum));
            out.extend(target.dense_input(rows).map_err(refusal)?.components(&format!("{prefix}.dense"), gpu as u8, Lifetime::TARGET, maximum)?);
            out.extend(target.dense_workspace(rows).map_err(refusal)?.components(&format!("{prefix}.dense"), gpu as u8, Lifetime::TARGET, maximum)?);
            // MSA prefill scratch and producers are per slot: units overlap in flight.
            out.extend(msa.workspace_components(&format!("{prefix}.attention"), gpu as u8, true, Lifetime::TARGET, maximum)?);
            out.extend(producers.workspace_components(&format!("{prefix}.attention"), gpu as u8, Lifetime::TARGET, maximum)?);
        }
        if self.heads() {
            let mut slot_rows = vec![decode_rows];
            slot_rows.extend((0..self.prefill.waves).map(|_| rows));
            out.extend(HeadsLayout::peer_components(gpu as u8, &slot_rows, false,
                &format!("minimax.heads.peer.{gpu}"), maximum)?);
        }
        out.extend(allowance(location, self.graph_reserve[gpu], Lifetime { first: Phase::Warmup, last: Phase::Growth },
            crate::placement::graph_reserve::source(&format!("minimax_m3.graphs.{gpu}")), true, maximum));
        out.extend(allowance(location, self.code[gpu], Lifetime::RESIDENT,
            "MiniMax selected-program loaded code (estimated module residency)".into(), true, maximum));
        let peak = 1 + crate::push_load::COPY_STREAMS as u64 + 1;
        out.extend(allowance(location, cuda_stream_slab_bytes(peak)?, Lifetime::RESIDENT,
            format!("minimax/driver-stream-slabs/gpu{gpu} ({peak} peak live streams: model + push copies + rotary)"),
            false, maximum));
        Ok(out)
    }
}

impl Recipe for MiniMaxInputs {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        let cfg = &self.cfg;
        let maximum = self.maximum_pages();
        if !(1..=2).contains(&self.gpus) || self.spark_ranks > 8 || maximum == 0 || maximum == u64::MAX
            || request.context.maximum != self.context.maximum || self.targets.len() != self.gpus
            || self.graph_reserve.len() != self.gpus
            || self.code.len() != self.gpus || self.expert_backings.len() != self.routed.len() + 1
            || self.spark.len() != self.routed.len() + 1
            || self.push_host_envelopes.len() != self.routed.len() + 1
            || self.mount.len() != self.gpus || self.prefill.rows == 0 || self.prefill.waves == 0
            || self.sequences == 0 {
            return Err(invalid("invalid extent or empty domain"));
        }
        cuteafd_core::concurrency::validate("minimax_m3", self.sequences).map_err(invalid)?;
        self.validate_attention()?;
        let routed = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
        if self.routed != routed || self.spark.iter().any(|ranks| ranks.len() != self.spark_ranks) {
            return Err(invalid("routed layer order or Spark rank inventory differs from the executor"));
        }
        let push = self.push_host_envelopes[0].is_some();
        for envelope in &self.push_host_envelopes {
            if envelope.is_some() != push {
                return Err(invalid("startup route differs between onboarding candidates"));
            }
            if let Some(envelope) = envelope {
                envelope.bytes()?;
                if envelope.expert_bytes.len() != self.gpus {
                    return Err(invalid("push host envelope rank inventory differs from coordinator"));
                }
            }
        }
        for (count, backings) in self.expert_backings.iter().enumerate() {
            if backings.len() != if count == 0 { 0 } else { self.gpus } {
                return Err(invalid("expert backing rank inventory differs from onboarding count"));
            }
            for (gpu, backing) in backings.iter().enumerate() {
                if backing.gpu != gpu || backing.count != count
                    || backing.layers.iter().map(|layer| layer.layer).collect::<Vec<_>>() != self.routed[..count] {
                    return Err(invalid("expert backing selection differs from routed prefix"));
                }
            }
        }
        if let Pin::Exact(count) = request.onboard {
            if count > self.routed.len() as u64 {
                return Err(invalid(format!("requested {count} onboarded layers, but only {} routed layers exist", self.routed.len())));
            }
            if count > 0 {
                if let LocalPackage::Missing(reason) = &self.local {
                    return Err(invalid(format!("missing native fp8-minimax-nvfp4a4 coordinator exports: {reason}")));
                }
            }
        }
        let capacity = self.request_capacity()?;
        let decode_rows = capacity.decode.rows;
        let pair = self.gpus == 2;
        let head = self.head_gpu();
        let mut components = self.execution_components()?;
        for gpu in 0..self.gpus {
            components.extend(self.targets[gpu].weight_components(maximum)?);
            if !push {
                for owner in self.mount[gpu].allocations(&Owner::Mount { gpu }.id()) {
                    let bytes = ByteCurve::fixed(owner.bytes, maximum);
                    components.push(Component::allocation(owner, bytes, false, maximum));
                }
            }
        }
        // Head-rank owners: vocabulary head workspace and the shared selector.
        components.extend(HeadLayout::new(cfg, decode_rows)?.components(&Owner::Head.id(), head, maximum)?);
        components.extend(serving::ServingOwners {
            selector: serving::SelectorOwners { location: Location::coordinator(head),
                capacity: decode_rows as usize, vocab: cfg.vocab_size, device_sampling: self.device_sampling },
            prefix: serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 } }.components(maximum)?);
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
            components.extend(buffer(Owner::Hops { gpu }.id(), Location::coordinator(gpu as u8), bytes,
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
            let push_compatible = !push || self.expert_backings[count].iter()
                .all(|backing| backing.storage == ExpertStorage::Push);
            if !local || !remote || !linked || !push_compatible {
                requires.push(Requires { when: Choice { variable: onboard, option }, any_of: vec![] });
                continue;
            }
            if let Some(envelope) = &self.push_host_envelopes[count] {
                for owner in envelope.allocations("minimax.push")? {
                    let bytes = ByteCurve::fixed(owner.bytes, maximum);
                    let component = Component::allocation(owner, bytes, false, maximum);
                    components.extend(with(vec![component], &activation));
                }
            }
            if count > 0 {
                if pair {
                    // No local halves means no OwnerTp2 exchange. Boundary
                    // hops retain their own storage independently of onboarding.
                    let rows = crate::serving_capacity::OwnerTp2Rows::new(cfg.hidden as u64,
                        cfg.topk as u64, u64::from(capacity.execution_rows()), false).map_err(refusal)?;
                    for demand in rows.demands().map_err(refusal)? {
                        components.extend(with(buffer(format!("{}.{}", Owner::Tp2 { gpu: demand.gpu as usize }.id(),
                            demand.group.replace(' ', "_")), Location::coordinator(demand.gpu), demand.bytes,
                            Lifetime::TARGET, demand.category, &demand.group, maximum).into_iter().collect(), &activation));
                    }
                }
                let LocalPackage::Installed { scratch, output } = self.local else { unreachable!() };
                for backing in &self.expert_backings[count] {
                    let gpu = backing.gpu;
                    let location = Location::coordinator(gpu as u8);
                    components.extend(with(weight_arena_components(&backing.arena, location, Lifetime::RESIDENT,
                        maximum, MemoryKind::Experts)?, &activation));
                    for (name, bytes) in [("scratch", scratch), ("output", output)] {
                        components.extend(with(buffer(format!("minimax.expert_workspace.{gpu}.{count}.{name}"),
                            location, bytes, Lifetime::RESIDENT, Category::Workspace,
                            "native expert workspace", maximum).into_iter().collect(), &activation));
                    }
                }
            }
            if count < self.routed.len() {
                for (rank, charge) in self.spark[count].iter().enumerate() {
                    let location = Location { device: Device::Spark(rank as u8), domain: Domain::Managed };
                    for (kind, group, bytes, category) in [
                        ("experts", "routed experts (native NVFP4 intermediate slices)", charge.experts, Category::Experts),
                        ("owners", "Spark rank I/O and execution owners", charge.owners, Category::Workspace),
                        ("startup", cuteafd_core::memory_layout::spark_runtime_reserve_group(), charge.runtime, Category::Runtime),
                    ] {
                        components.extend(buffer(Owner::Spark { rank, kind, count }.id(), location, bytes,
                            Lifetime::RESIDENT, category, group, maximum).map(|mut c| { c.activation = activation.clone(); c }));
                    }
                    if charge.experts > 0 {
                        components.push(Component { activation: activation.clone(),
                            bytes: ByteCurve::fixed(charge.experts, maximum),
                            kind: ComponentKind::DriverPageTables { model: DriverPageTableModel::Sm121Arena, location,
                                lifetime: Lifetime::RESIDENT, backing: Owner::Spark { rank, kind: "experts", count }.id() } });
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
