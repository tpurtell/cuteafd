//! Inkling 2 Small physical owners for the shared contract planner (#1930).
//!
//! A family supplies facts, never a private solver. Both edges, `cuteafd plan
//! --layout` and serving startup, prepare one [`InklingInputs`] and submit it
//! to [`planner::solve`](crate::placement::planner::solve). They differ only
//! in the coordinator [`Baseline`] (planned context forecast vs one measured
//! sample). Every owner is named once:
//!
//! - target arena per coordinator rank (globals and each owned layer's
//!   attention, convolutions, norms, router and residual queries), one backing
//!   plus one tail;
//! - RTX expert arena per onboarding count (whole on one RTX, TP2 halves on a
//!   pair) with the installed package's executor workspace;
//! - Spark experts per rank per onboarding count, with their I/O, typed
//!   execution owners, runtime residual and SM121 arena page tables;
//! - KV pool pages for the eight global layers only, charged by the layer's
//!   attention mode (heads: half the KV heads per rank; context: page parity,
//!   plus the shared context exchange; layers: the whole owner); the forty
//!   sliding-window layers keep per-sequence append rings, and every layer
//!   keeps its short-conv lag rows, both in the prefix-mark arena that scales
//!   with concurrency, not context, and split or replicated the same way;
//! - per-sequence page tables, decode lane and per-slot prefill unit owners
//!   (projections, relative logits, attention, residual stack, router, expert
//!   activation staging), TP2 expert exchange and the boundary hop on a pair,
//!   head rows, selector;
//! - one canonical graph reserve per device, loaded code, stream slabs, BLAS;
//! - host: prefix quota, upload staging, mount source, transport;
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
use crate::families::inkling2::{experts::LayerExperts, weight_layout::CoordinatorWeights, Inkling2Config};
use crate::serving_capacity::WeightArenaLayout;
use cuteafd_core::memory_layout::{Category, MemoryKind};
use cuteafd_core::spark_reply::SparkReplyFormats;

/// KV page quantum of the global layers (64-token pages).
pub const PAGE_ROWS: u64 = 64;
/// Prefill transport endpoints: one shared link (LaneSupport::One).
pub const LANES: u64 = 1;
/// Decode/verify rows per step, the Spark decode owner's admitted extent.
pub const DECODE_ROWS: u32 = 64;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };

/// What the Inkling 2 executor runs, on the shared attention-placement
/// mechanism (`ATTENTION_PLACEMENT`): every layer is GQA 48/8, so all three
/// placements partition it.
/// - `heads` (auto on a pair): 24 query and 4 KV heads per GPU; KV pages,
///   SWA rings and conv lag rows split by head with no replication; the
///   relative logits, sinks and output gate are per-head and split with them;
///   the residual is replicated (one all-reduce per layer, no boundary hop).
/// - `context`: head-split projections with page-parity KV on the eight
///   global layers (the forty window layers keep head-split rings), plus the
///   shared context exchange per lane.
/// - `layers`: contiguous whole owners (24/24, one boundary hop), routed
///   experts as TP2 halves; nothing replicated.
/// Plan admits all three; serve runs the ones with a qualified executor
/// (stage 4) and names the rest as refusals.
pub const INKLING2: ExecutorModes = ExecutorModes { family: "inkling2",
    modes: &[LayerMode::HeadSplit, LayerMode::ContextSplit, WHOLE, LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }],
    hops: true, attention: AttentionPlacement::Heads };

/// Per-layer modes for a resolved placement: one GPU runs whole layers;
/// `heads` splits every layer; `context` page-splits the global layers and
/// head-splits the window layers (a bounded ring gains nothing from parity);
/// `layers` cuts contiguous whole owners in the middle, head on GPU1.
pub fn layer_modes(cfg: &Inkling2Config, gpus: usize, placement: AttentionPlacement) -> Vec<LayerMode> {
    if gpus == 1 { return vec![WHOLE; cfg.layers]; }
    let cut = cfg.layers.div_ceil(2);
    (0..cfg.layers).map(|layer| match placement {
        AttentionPlacement::Heads => LayerMode::HeadSplit,
        AttentionPlacement::Context => if cfg.local[layer] { LayerMode::HeadSplit } else { LayerMode::ContextSplit },
        AttentionPlacement::Layers => LayerMode::Whole { gpu: u8::from(layer >= cut), ffn: FfnMode::Owner },
    }).collect()
}

fn invalid(detail: impl std::fmt::Display) -> Refusal {
    Refusal::InvalidFact(format!("Inkling 2 component contract: {detail}"))
}

fn refusal(error: anyhow::Error) -> Refusal { invalid(format!("{error:#}")) }

/// The installed coordinator expert package for the checkpoint's storage.
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

/// Bundled PE-Lang ViT and merge projector: own arena plus scratch.
#[derive(Debug, Clone)]
pub struct Vision {
    pub arena: WeightArenaLayout,
    pub scratch: u64,
    /// Host embedding cache quota for prepared images.
    pub media_cache: u64,
}

/// Everything both edges know before one solve.
#[derive(Debug, Clone)]
pub struct InklingInputs {
    pub cfg: Inkling2Config,
    pub gpus: usize,
    pub spark_ranks: usize,
    pub context: Context,
    /// Largest pool the build indexes (KV max), in tokens.
    pub maximum_pool_tokens: u64,
    pub prefill: PrefillShape,
    /// Boot concurrency C: per-sequence rings, conv lags, page tables, selector.
    pub sequences: u64,
    /// Per-rank target arenas (one per coordinator GPU).
    pub targets: Vec<CoordinatorWeights>,
    /// The resolved attention placement and every layer's mode under it.
    pub placement: AttentionPlacement,
    pub modes: Vec<LayerMode>,
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
pub struct AdmittedInkling {
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

/// The Spark request wire for a layer's expert storage: MXFP8 experts
/// consume the checkpoint's own group-32 E4M3 + UE8M0 activations (the K32
/// frame the FP8 families already send), NVFP4 and EXL3 experts consume BF16
/// rows and quantize or dequantize on the Spark.
pub fn ingress(storage: Option<crate::plan::families::inkling2::ExpertStorage>) -> cuteafd_core::SparkIngress {
    match storage {
        Some(crate::plan::families::inkling2::ExpertStorage::Mxfp8) => cuteafd_core::SparkIngress::Fp8K32,
        _ => cuteafd_core::SparkIngress::Bf16,
    }
}

/// One Spark endpoint at the admitted unit. A mixed checkpoint (native NVFP4
/// layers 0-39, MXFP8 40-47) uses the per-layer native ingress.
pub fn spark_link(gpu: u8, ranks: usize, prefill: PrefillShape, replies: SparkReplyFormats,
    ingress: cuteafd_core::SparkIngress) -> Result<Vec<crate::placement::PreparedSparkLink>, Refusal> {
    use crate::placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates};
    if ranks == 0 { return Ok(Vec::new()); }
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    Ok(vec![PreparedSparkLink::endpoint("inkling2.spark", gpu, SparkEndpointOwners { ranks,
        geometry: cuteafd_core::ExpertGeometry::INKLING2,
        endpoint: cuteafd_core::SparkEndpoint { ring_depth: None,
            capacity: u32::try_from(prefill.rows).map_err(|_| Refusal::Overflow("Inkling 2 transport rows"))?,
            ingress, partial_reply_bytes: None },
        rings: cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
            slot_bytes: rings.slot_bytes, alignment: rings.alignment } },
        usize::try_from(prefill.waves).map_err(|_| Refusal::Overflow("Inkling 2 transport waves"))?,
        true, true, SparkIntakeCandidates::Probed, replies.prefill(false), Lifetime::TARGET)?])
}

/// Typed Spark owners at the admitted unit (one prefill + one decode lane).
pub fn spark_executions(prefill: PrefillShape, exl3: bool) -> anyhow::Result<cuteafd_core::SparkExecutions> {
    let backend = if exl3 {
        crate::plan::spark_executions::SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::for_geometry(
            cuteafd_core::ExpertGeometry::INKLING2))
    } else { crate::plan::spark_executions::SparkBackend::Fp8 };
    crate::plan::spark_executions::admit_family("inkling2", &backend.exports(), None, prefill.rows, prefill.waves)
}

/// Per-sequence sliding-window ring and conv-lag bytes of one layer, the
/// prefix-mark region a whole owner keeps beside its global-layer pages.
pub struct SequenceState {
    /// `window + 64` rows of K and V per local layer, page aligned (`AppendRing`).
    pub ring_bytes: u64,
    /// `(kernel - 1)` lag rows of the K, V, attention-output and MLP-output
    /// convolution inputs; dynamic global taps are recomputed per token.
    pub lag_bytes: u64,
    /// The K and V part of `lag_bytes` (split by head; the rest replicates).
    pub kv_lag_bytes: u64,
}

impl SequenceState {
    pub fn new(cfg: &Inkling2Config, local: bool) -> Result<Self, Refusal> {
        let kv_row = (cfg.kv_heads * cfg.head_dim * 2) as u64;
        let ring = if local {
            let ring = cuteafd_core::prefix::swa_ring(cfg.window as u64, u64::from(DECODE_ROWS), PAGE_ROWS)
                .ok_or_else(|| invalid("sliding-window ring geometry"))?;
            ring.rows.checked_mul(2 * kv_row).ok_or(Refusal::Overflow("Inkling 2 SWA ring"))?
        } else { 0 };
        let kv_lags = (cfg.sconv_kernel as u64 - 1) * 2 * kv_row;
        let lags = kv_lags + (cfg.sconv_kernel as u64 - 1) * 2 * cfg.hidden as u64 * 2;
        Ok(Self { ring_bytes: ring, lag_bytes: lags, kv_lag_bytes: kv_lags })
    }
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

impl InklingInputs {
    pub fn maximum_pages(&self) -> u64 { self.maximum_pool_tokens.div_ceil(self.page_rows()) }

    fn modes(&self) -> Vec<LayerMode> { self.modes.clone() }

    fn head_gpu(&self) -> u8 { (self.gpus - 1) as u8 }

    /// Context pages are parity-interleaved over the pair: twice the rows per page.
    pub fn page_rows(&self) -> u64 {
        if self.modes.iter().any(|mode| *mode == LayerMode::ContextSplit) { 2 * PAGE_ROWS } else { PAGE_ROWS }
    }

    /// Shared context exchange geometry for the global layers under `context`:
    /// one 64-token K+V page per staging unit, BF16 query rows and this
    /// rank's 24-head partials as the dimension-generic exchange writes them
    /// (#1955: `heads x (D*2 + 4)`, BF16 partial plus F32 LSE), two parity
    /// slots per lane.
    fn context_buffers(&self) -> crate::placement::ContextBuffers {
        let cfg = &self.cfg;
        let kv_row = (cfg.kv_heads * cfg.head_dim * 2) as u64;
        let heads = (cfg.heads / 2) as u64;
        crate::placement::ContextBuffers {
            staging_unit_bytes: PAGE_ROWS * 2 * kv_row, staging_unit_rows: PAGE_ROWS,
            query_row_bytes: heads * cfg.head_dim as u64 * 2,
            partial_row_bytes: heads * (cfg.head_dim as u64 * 2 + 4),
            candidate_row_bytes: 0, compiled_extent: self.prefill.rows.max(u64::from(DECODE_ROWS)),
            decode_rows: u64::from(DECODE_ROWS), lanes: LANES, workspace_bytes: 0 }
    }

    fn hop_spec(&self) -> HopSpec {
        // The hop carries the partial residual block and the completed blocks
        // the next owner mixes: at most (blocks + 1) hidden rows per token.
        let rows = self.cfg.residual_blocks() as u64 + 1;
        // Whole owners hand the residual to the last owner; split layers keep
        // a replicated residual on both ranks (no hop).
        let terminal = if self.modes.iter().all(|mode| matches!(mode, LayerMode::Whole { .. })) {
            TerminalHome::AttentionOwner } else { TerminalHome::Replicated };
        HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: rows * self.cfg.hidden as u64 * 2,
            rows: self.prefill.rows.max(u64::from(DECODE_ROWS)), lanes: LANES, entry_gpu: 0,
            head_gpu: self.head_gpu(), terminal }
    }

    /// The available onboarding counts: an ordered prefix of the routed layers.
    pub fn counts(&self) -> Vec<usize> { (0..=self.routed.len()).collect() }

    pub fn admit(&self, request: Request) -> Result<AdmittedInkling, Refusal> {
        let solved = solve(request, self.budgets.clone(), self)?;
        let placement = self.execution(&solved)?;
        let onboard = placement.onboard_layers;
        Ok(AdmittedInkling { placement, onboard, draft_home: solved.decisions.draft_home,
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
            experts: if local.contains(&layer) { if pair { ExpertHome::RtxTp2 } else { ExpertHome::RtxWhole { gpu: 0 } } }
                else { ExpertHome::Spark } }).collect();
        let tp2 = (pair && count > 0).then(|| Tp2Range { first: local[0], layers: count, peak_bytes: peaks });
        let expert_ranges = (0..self.gpus).map(|gpu| if pair || count == 0 {
            ExpertRange { first: 0, layers: 0, peak_bytes: 0 }
        } else { ExpertRange { first: local[0], layers: count, peak_bytes: peaks[gpu] } }).collect();
        let spec = self.hop_spec();
        let residual = residual_homes(&modes, &spec).map_err(|error| invalid(error))?;
        let hops = plan_hops(&modes, &spec).map_err(|error| invalid(error))?;
        let attention = if pair { self.placement } else { AttentionPlacement::Heads };
        let context = modes.iter().any(|mode| *mode == LayerMode::ContextSplit);
        let buffers = self.context_buffers();
        let placement = Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: attention,
            // Window layers never page-split: under `context` they stay head-split.
            attention_by_kind: vec![(AttentionClass::Swa, if context { AttentionPlacement::Heads } else { attention }),
                (AttentionClass::Gqa, attention)],
            peer_row_bytes: if context { (buffers.query_row_bytes + buffers.partial_row_bytes)
                * modes.iter().filter(|m| **m == LayerMode::ContextSplit).count() as u64 } else { 0 },
            onboard_layers: count, max_pool_tokens_no_onboard: None, layers,
            movables: solved.decisions.draft_home.map(|gpu| (MovableId::Drafter, gpu)).into_iter().collect(),
            expert_ranges, residual, hops, tp2,
            items: (0..self.gpus).map(|gpu| solved.items(Location::coordinator(gpu as u8)))
                .collect::<Result<_, _>>()? };
        INKLING2.check(&placement).map_err(|error| invalid(error))?;
        Ok(placement)
    }

    /// Per-GPU expert arena backing at `count` onboarded layers, including
    /// the executor workspace (charged only when a layer is onboarded).
    fn arena_bytes(&self, count: usize) -> Result<[u64; 2], Refusal> {
        let mut out = [0u64; 2];
        if count == 0 { return Ok(out); }
        let LocalPackage::Installed { workspace } = &self.local else {
            return Err(invalid("onboarded layers need an installed Inkling 2 RTX expert package"));
        };
        for gpu in 0..self.gpus {
            let weights = self.experts[..count].iter().try_fold(0u64, |sum, layer| sum.checked_add(
                if self.gpus == 2 { layer.half[gpu] } else { layer.whole })).ok_or(Refusal::Overflow("Inkling 2 expert arena"))?;
            out[gpu] = weights.checked_add(workspace[gpu]).ok_or(Refusal::Overflow("Inkling 2 expert arena"))?;
        }
        Ok(out)
    }

    /// Owners of one in-flight row set (a decode step or one prefill unit
    /// slot): projections, relative logits, attention output, the residual
    /// block stack, router scores and expert activation staging. Named per
    /// plane so the ledger reconciles them one by one.
    fn unit_owners(&self, prefix: &str, gpu: usize, rows: u32, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let cfg = &self.cfg;
        let location = Location::coordinator(gpu as u8);
        let r = u64::from(rows);
        let (h, d, heads, kv) = (cfg.hidden as u64, cfg.head_dim as u64, cfg.heads as u64, cfg.kv_heads as u64);
        let planes: [(&str, u64); 14] = [
            ("x", r * h * 2),                                   // mixed layer input (BF16)
            ("q", r * heads * d * 2),
            ("r", r * heads * cfg.d_rel as u64 * 2),
            ("gate", r * heads * d * 2),
            ("sinks", r * heads * 4),                           // F32 per (token, head)
            ("k", r * kv * d * 2),
            ("v", r * kv * d * 2),
            ("rel_logits", r * heads * cfg.window.max(cfg.rel_extent) as u64 * 2), // the largest plane
            ("attention", r * heads * d * 2),
            ("conv_out", r * h * 2),                            // attn_sconv / mlp_sconv output
            ("blocks", r * (cfg.residual_blocks() as u64 + 1) * h * 2), // completed blocks + partial
            ("router", r * cfg.experts as u64 * 4),             // F32 scores
            ("routes", r * cfg.topk as u64 * 8),                // ids + weights
            ("expert_in", r * (h + h / 32)),                    // MXFP8 K32 request staging
        ];
        let mut out = Vec::new();
        for (name, bytes) in planes {
            out.extend(buffer(format!("{prefix}.{name}"), location, bytes, Lifetime::TARGET, Category::Workspace,
                "Inkling 2 row-unit planes (geometry formula)", maximum));
        }
        let metadata: [(&str, u64); 4] = [("positions", r * 8), ("slots", r * 8), ("batch_ids", r * 4), ("causal_lengths", r * 4)];
        for (name, bytes) in metadata {
            out.extend(buffer(format!("{prefix}.metadata.{name}"), location, bytes, Lifetime::TARGET, Category::Workspace,
                "Inkling 2 unit metadata", maximum));
        }
        Ok(out)
    }

    /// Fixed owners of one coordinator rank that do not depend on any domain.
    fn rank_owners(&self, gpu: usize, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let location = Location::coordinator(gpu as u8);
        let cfg = &self.cfg;
        let rows = u32::try_from(self.prefill.rows).map_err(|_| Refusal::Overflow("Inkling 2 unit rows"))?;
        let context = u32::try_from(self.context.maximum).map_err(|_| Refusal::Overflow("Inkling 2 context"))?;
        let sequences = self.sequences;
        let mut out = Vec::new();
        let target = self.targets.get(gpu).ok_or_else(|| invalid("missing target arena"))?;
        out.extend(target.weight_components(maximum)?);
        // Whole-owner global layers: K and V pages on the owner only, one
        // physical backing per plane that scales with the pool. Local layers
        // keep no pool pages: their window lives in the per-sequence ring.
        let kv_row = (cfg.kv_heads * cfg.head_dim * 2) as u64;
        let page_rows = self.page_rows();
        let mut marks = 0u64;
        for (layer, &mode) in self.modes.iter().enumerate() {
            // This rank's share of the layer's KV heads: all of a whole layer it
            // owns, half of a split layer, none of a layer another GPU owns.
            let share = match mode {
                LayerMode::Whole { gpu: owner, .. } if usize::from(owner) == gpu => Some((1u64, 1u64)),
                LayerMode::Whole { .. } => None,
                LayerMode::HeadSplit | LayerMode::ContextSplit => Some((1, 2)),
            };
            let Some((num, den)) = share else { continue };
            if !cfg.local[layer] {
                // One page of records on this rank: heads split the row; context
                // parity interleaves whole pages (page_rows doubles, half land here).
                let stride = page_rows * kv_row * num / den;
                for name in ["keys", "values"] {
                    out.push(Component::allocation(Allocation::buffer(format!("inkling2.kv.{layer}.{gpu}.{name}"), location,
                        stride, Lifetime::TARGET, Category::Kv, "global-layer paged KV records"),
                        ByteCurve::linear_rounded(stride, stride, maximum, SM120_ALLOCATION_GRANULARITY_BYTES),
                        false, maximum));
                }
            }
            let state = SequenceState::new(cfg, cfg.local[layer])?;
            // Rings split by KV head; the hidden-stream conv lags are replicated
            // under a split (the residual is replicated), the K/V lags split.
            let ring = state.ring_bytes * num / den;
            let lags = if den == 1 { state.lag_bytes } else { state.kv_lag_bytes / 2 + (state.lag_bytes - state.kv_lag_bytes) };
            marks = marks.checked_add((ring + lags).checked_mul(sequences)
                .ok_or(Refusal::Overflow("Inkling 2 sequence state"))?).ok_or(Refusal::Overflow("Inkling 2 sequence state"))?;
        }
        if self.modes.iter().any(|mode| *mode == LayerMode::ContextSplit) {
            for demand in self.context_buffers().demands().map_err(|error| invalid(error))?
                .into_iter().filter(|d| usize::from(d.gpu) == gpu) {
                out.extend(buffer(format!("inkling2.context.{gpu}.{}", demand.group.replace(' ', "_")), location,
                    demand.bytes, Lifetime::TARGET, demand.category, &demand.group, maximum));
            }
        }
        // Sliding-window rings and short-conv lag rows for every admitted
        // sequence: the prefix-mark arena a snapshot copies (byte-identical
        // restores), sized by concurrency, not context.
        out.extend(buffer(format!("inkling2.marks.{gpu}"), location, marks, Lifetime::TARGET, Category::Prefix,
            "sliding-window rings and conv lag rows per sequence (geometry formula)", maximum));
        // Page tables for every admitted sequence at the compiled context (global layers only).
        let pages = u64::from(context).div_ceil(page_rows);
        out.extend(buffer(format!("inkling2.cache.{gpu}.page_table"), location, sequences * pages * 4 + sequences * 8,
            Lifetime::TARGET, Category::Kv, "global-layer page tables and lengths", maximum));
        // One decode lane on every rank (its layers' work), then one owner set
        // per in-flight prefill slot (PrefillHooks unit slots).
        out.extend(self.unit_owners(&format!("inkling2.decode.{gpu}"), gpu, DECODE_ROWS, maximum)?);
        for slot in 0..self.prefill.waves {
            out.extend(self.unit_owners(&format!("inkling2.unit.{gpu}.{slot}"), gpu, rows, maximum)?);
        }
        if self.gpus == 2 {
            // TP2 expert exchange (shared `QwenTp2Rows` ABI: BF16 rows, two
            // abortable slots, reduction, peer control and abort word), one
            // unit wide; decode rows fit within it.
            let rows = crate::placement::families::qwen4::QwenTp2Rows::new(cfg.hidden as u64, cfg.topk as u64,
                self.prefill.rows.max(u64::from(DECODE_ROWS)), false).map_err(refusal)?;
            for demand in rows.demands().map_err(refusal)?.into_iter().filter(|d| usize::from(d.gpu) == gpu) {
                out.extend(buffer(format!("inkling2.tp2.{gpu}.{}", demand.group.replace(' ', "_")), location,
                    demand.bytes, Lifetime::TARGET, demand.category, &demand.group, maximum));
            }
        }
        out.extend(allowance(location, self.graph_reserve[gpu], Lifetime { first: Phase::Warmup, last: Phase::Growth },
            crate::placement::graph_reserve::source(&format!("inkling2.graphs.{gpu}")), true, maximum));
        out.extend(allowance(location, self.code[gpu], Lifetime::RESIDENT,
            "Inkling 2 selected-program loaded code (estimated module residency)".into(), true, maximum));
        let peak = 1 + crate::push_load::COPY_STREAMS as u64 + 1;
        out.extend(allowance(location, cuda_stream_slab_bytes(peak)?, Lifetime::RESIDENT,
            format!("inkling2/driver-stream-slabs/gpu{gpu} ({peak} peak live streams: model + push copies + conv)"),
            false, maximum));
        for owner in self.mount.get(gpu).copied().unwrap_or_default().allocations(&format!("inkling2.mount.{gpu}")) {
            let bytes = ByteCurve::fixed(owner.bytes, maximum);
            out.push(Component::allocation(owner, bytes, false, maximum));
        }
        Ok(out)
    }

    /// Head-rank owners: 64-row F32 logits over the stored vocabulary (trim
    /// and softcap run in place), tokens and the normalized input rows.
    fn head_owners(&self, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let head = self.head_gpu();
        let cfg = &self.cfg;
        let rows = u64::from(DECODE_ROWS);
        let buffers: [(&str, u64); 3] = [("tokens", rows * 4), ("normalized", rows * cfg.hidden as u64 * 2),
            ("logits", rows * cfg.vocab_size as u64 * 4)];
        let mut out = Vec::new();
        for (name, bytes) in buffers {
            out.extend(buffer(format!("inkling2.head.{name}"), Location::coordinator(head), bytes, Lifetime::TARGET,
                Category::Workspace, "Inkling 2 bounded vocabulary head rows", maximum));
        }
        Ok(out)
    }
}

impl Recipe for InklingInputs {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        let cfg = &self.cfg;
        let maximum = self.maximum_pages();
        if !(1..=2).contains(&self.gpus) || self.spark_ranks > 8 || maximum == 0 || maximum == u64::MAX
            || request.context.maximum != self.context.maximum || self.targets.len() != self.gpus
            || self.modes.len() != cfg.layers || self.graph_reserve.len() != self.gpus
            || self.code.len() != self.gpus || self.experts.len() != self.routed.len()
            || self.spark.len() != self.routed.len() + 1 || self.prefill.rows == 0 || self.prefill.waves == 0
            || self.sequences == 0 {
            return Err(invalid("invalid extent or empty domain"));
        }
        if self.modes.iter().any(|mode| match mode {
            LayerMode::Whole { gpu, .. } => usize::from(*gpu) >= self.gpus,
            LayerMode::HeadSplit | LayerMode::ContextSplit => self.gpus != 2 }) {
            return Err(invalid("attention mode outside the coordinator ranks"));
        }
        if self.modes != layer_modes(cfg, self.gpus, self.placement) {
            return Err(invalid("layer modes differ from the resolved attention placement"));
        }
        let pair = self.gpus == 2;
        let head = self.head_gpu();
        let mut components = Vec::new();
        for gpu in 0..self.gpus { components.extend(self.rank_owners(gpu, maximum)?); }
        components.extend(self.head_owners(maximum)?);
        components.extend(serving::ServingOwners {
            selector: serving::SelectorOwners { location: Location::coordinator(head),
                capacity: usize::try_from(self.sequences).map_err(|_| Refusal::Overflow("Inkling 2 selector"))?
                    .max(DECODE_ROWS as usize), vocab: cfg.unpadded_vocab_size, device_sampling: self.device_sampling },
            prefix: self.host_prefix }.components(maximum)?);
        if let Some(geometry) = &self.expert_usage {
            let devices = (0..self.gpus.min(geometry.routed_layers.len())).map(|gpu| gpu as u8).collect::<Vec<_>>();
            components.extend(crate::placement::expert_usage::components(geometry, &devices, maximum)?);
        }
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
            .ok_or(Refusal::Overflow("Inkling 2 residual hop buffers"))?;
        for (gpu, &bytes) in hop_bytes.iter().enumerate() {
            components.extend(buffer(format!("inkling2.hops.{gpu}"), Location::coordinator(gpu as u8), bytes,
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
                let id = format!("inkling2.experts.{gpu}.{count}");
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
                        ("experts", "routed experts (header-exact intermediate slices)", charge.experts, Category::Experts),
                        ("owners", "Spark rank I/O and execution owners", charge.owners, Category::Workspace),
                        ("startup", cuteafd_core::memory_layout::spark_runtime_reserve_group(), charge.runtime, Category::Runtime),
                    ] {
                        components.extend(buffer(format!("inkling2.spark.{rank}.{kind}.{count}"), location, bytes,
                            Lifetime::RESIDENT, category, group, maximum).map(|mut c| { c.activation = activation.clone(); c }));
                    }
                    if charge.experts > 0 {
                        components.push(Component { activation: activation.clone(),
                            bytes: ByteCurve::fixed(charge.experts, maximum),
                            kind: ComponentKind::DriverPageTables { model: DriverPageTableModel::Sm121Arena, location,
                                lifetime: Lifetime::RESIDENT, backing: format!("inkling2.spark.{rank}.experts.{count}") } });
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
                components.extend(buffer(format!("inkling2.draft.{name}"), location, *bytes, lifetime,
                    Category::Drafter, "bundled dSpark scratch", maximum));
            }
            if draft.state_per_page > 0 {
                components.push(Component::allocation(Allocation::buffer("inkling2.draft.state", location,
                    draft.state_per_page, lifetime, Category::Drafter, "bundled dSpark per-page state"),
                    ByteCurve::linear(draft.state_per_page, draft.state_per_page, maximum), false, maximum));
            }
        }
        // Optional bundled vision tower: own arena; `Encoder` domain [on, off].
        if let Some(vision) = &self.vision {
            let variable = contract_variables.len();
            contract_variables.push(Variable::Encoder(vec![0, 1]));
            let on = Activation::choice(variable, 0);
            let location = Location::coordinator(0);
            components.extend(with(weight_arena_components(&vision.arena, location, Lifetime::RESIDENT, maximum,
                MemoryKind::Weights)?, &on));
            components.extend(buffer("inkling2.vision.scratch", location, vision.scratch, Lifetime::TARGET,
                Category::Workspace, "vision tower scratch", maximum).map(|mut c| { c.activation = on.clone(); c }));
            components.extend(buffer("inkling2.host.media", Location::HOST, vision.media_cache,
                Lifetime { first: Phase::Warmup, last: Phase::Growth }, Category::Embedding, "media/embedding-cache",
                maximum).map(|mut c| { c.activation = on.clone(); c }));
        }
        // BLAS: the BF16 vocabulary head borrows one serving-thread pool.
        if self.head.bf16() {
            components.extend(serving::blas_components("sm_120", &[serving::BlasOwners { name: "Inkling 2 target head".into(),
                process: "coordinator".into(), thread: "inkling2/blas/serve".into(), location: Location::coordinator(head),
                gemm: self.head.gemm(), lt: self.head.lt(), lifetime: Lifetime::TARGET, activation: Activation::always() }],
                maximum, PhaseOrder::TARGET_THEN_DRAFT)?);
        }
        components.extend(allowance(Location::PINNED, 64 << 20, Lifetime::RESIDENT,
            "Inkling 2 reusable H2D staging (estimated: library-wide sync-h2d backing, cold golden reconciles)".into(),
            true, maximum));
        fixed_request_tails(&mut components, maximum, &[])?;
        let mut contract = Contract { floor_fallback: None, phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: (self.spark_ranks > 0).then(|| SparkReplyFormats {
                prefill: Some(request.spark_reply_formats.prefill(false)),
                decode: Some(request.spark_reply_formats.decode()) }),
            resident_weight_limit: request.resident_weight_limit, page_rows: self.page_rows(), pool_page_multiple: 1,
            maximum_pages: maximum, gpus: self.gpus, variables: contract_variables, requires, components };
        // `page_rows` is the parity-doubled page under context; one backing per plane either way.
        contract.add_driver_page_tables();
        Ok(contract)
    }
}

/// Coordinator budgets: physical card identity and an independent cap; the
/// named reserve stays free inside it. `context` is the planned CUDA context.
pub fn planned_budget(gpu: u8, physical: u64, logical: u64, context: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)], physical_bytes: physical,
        logical_bytes: logical, headroom: crate::placement::reserve::gpu("inkling2"),
        baseline: Baseline::Forecast { occupied: context, source: "CUDA context and modules on the coordinator GPU".into(),
            assumed: false } }
}

/// The measured edge: one CUDA sample before weights.
pub fn measured_budget(gpu: u8, physical: u64, logical: u64, free: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)], physical_bytes: physical,
        logical_bytes: logical, headroom: crate::placement::reserve::gpu("inkling2"),
        baseline: Baseline::Sampled { free, phase: Phase::Load, source: "Inkling 2 pre-weight CUDA sample".into() } }
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
    fn executor_admits_every_shared_attention_placement() {
        assert!(INKLING2.runs(WHOLE) && INKLING2.runs(LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }));
        assert!(INKLING2.runs(LayerMode::HeadSplit) && INKLING2.runs(LayerMode::ContextSplit));
        assert!(!INKLING2.runs(LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }));
        assert_eq!(INKLING2.attention_default(), AttentionPlacement::Heads);
        for placement in [AttentionPlacement::Heads, AttentionPlacement::Context, AttentionPlacement::Layers] {
            assert_eq!(INKLING2.check_attention(Some(placement), 2, true).unwrap(), placement);
        }
        assert_eq!(INKLING2.check_attention(None, 1, false).unwrap(), AttentionPlacement::Heads);
        let cfg = Inkling2Config::from_hf(&crate::families::inkling2::config::tests::release_config()).unwrap();
        assert_eq!(layer_modes(&cfg, 1, AttentionPlacement::Context), vec![WHOLE; 48]);
        assert!(layer_modes(&cfg, 2, AttentionPlacement::Heads).iter().all(|m| *m == LayerMode::HeadSplit));
        let context = layer_modes(&cfg, 2, AttentionPlacement::Context);
        assert_eq!(context.iter().filter(|m| **m == LayerMode::ContextSplit).count(), 8);
        assert!(cfg.global_layers().iter().all(|&layer| context[layer] == LayerMode::ContextSplit));
        let layers = layer_modes(&cfg, 2, AttentionPlacement::Layers);
        assert_eq!(layers.iter().filter(|m| matches!(m, LayerMode::Whole { gpu: 0, .. })).count(), 24);
        assert!(matches!(layers[47], LayerMode::Whole { gpu: 1, .. }));
    }

    #[test]
    fn spark_owners_cover_one_prefill_and_one_decode_lane() {
        for exl3 in [false, true] {
            let executions = spark_executions(PrefillShape { rows: 2752, waves: 3 }, exl3).unwrap();
            assert_eq!(executions.decode.len(), 1);
            assert_eq!(executions.prefill.unit_rows, 2752);
            assert_eq!(executions.prefill.buffered_units, 3);
        }
    }

    #[test]
    fn sequence_state_keeps_the_window_plus_one_step_and_three_lag_rows() {
        let cfg = Inkling2Config::from_hf(&crate::families::inkling2::config::tests::release_config()).unwrap();
        let local = SequenceState::new(&cfg, true).unwrap();
        // 1024 + 64 rows of K and V at 8 x 128 BF16 each.
        assert_eq!(local.ring_bytes, 1088 * 2 * 2048);
        // 3 lag rows of K, V (1024 each) and the attention/MLP hidden streams (4608 each), BF16.
        assert_eq!(local.lag_bytes, 3 * (2 * 2048 + 2 * 9216));
        let global = SequenceState::new(&cfg, false).unwrap();
        assert_eq!(global.ring_bytes, 0);
        assert_eq!(global.lag_bytes, local.lag_bytes);
        assert_eq!(local.kv_lag_bytes, 3 * 2 * 2048);
    }
}
