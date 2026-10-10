//! One admission solver for every family (PLAN "v3 placement: design",
//! section 1). CPU-only: the planner (`plan::layout`) and each family's
//! runtime admission build a [`PlacementRequest`] from the same family
//! inputs and call [`solve`]; they differ only in the GPU [`Baseline`]
//! (planned charges vs one measured sample).
//!
//! Order (TJ): reserve the KV pool first, then graphs and workspaces (fixed
//! demands, charged before the pool), then RTX expert layers. Mandatory
//! movables (drafter stage experts) are charged before the pool too, so an
//! automatic pool can never crowd them out.
//!
//! Placement PR 1 added `HeadSplit` and `Whole` layer modes and contiguous
//! whole-layer expert ranges (GPU0, then GPU1). PR 3 adds per-layer
//! ownership: the residual's home at every layer boundary ([`ResidualHome`]),
//! the [`Hop`]s the modes imply (their receive buffers charged as fixed
//! demands) and the executor's mode set ([`ExecutorModes`]), refused at plan
//! time when a layer would need a mode the family cannot run.
//! Peer-accessible two-GPU builds use a contiguous TP2 expert prefix; TP1
//! arena-sharing movables retain a separate arena and workspace.
use cuteafd_core::memory_layout::{Basis, Category, Item};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod families;
mod pool;
mod residual;
mod solve;
#[cfg(test)]
mod tests;

pub use pool::PoolPolicy;
pub use residual::{hop_buffer_bytes, plan_hops, Hop, HopKind, HopPoint, HopSpec, ResidualHome, Transition, HOP_SLOTS};
pub use solve::solve;

/// What a request asks of the hardware, built by a family from its own
/// geometry; see `families::<family>`.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementRequest {
    pub inventory: Inventory,
    pub pool: PoolPolicy,
    /// One per backbone layer, in order.
    pub layers: Vec<LayerDemand>,
    /// Per-unit pool bytes each KV-owning GPU charges beside its layers'
    /// records (unit metadata, lane page tables), indexed by GPU.
    pub pool_overhead: Vec<u64>,
    /// Non-layer items charged before the pool, per GPU.
    pub fixed: Vec<Demand>,
    /// Mandatory items the solver places on an allowed GPU (drafter, encoders).
    pub movables: Vec<Movable>,
    /// Workspace of a GPU's local expert arena, charged once on every GPU
    /// that holds a routed layer or an arena-sharing movable.
    pub expert_workspace: u64,
    /// Separate TP2 backbone workspace per rank; excludes TP1 movables.
    pub tp2_workspace: [u64; 2],
    /// Which fixes the plan: the pool (`Auto`) or the RTX expert layers.
    pub onboard: Onboard,
    /// GPUs (from GPU0) whose executors can hold whole routed layers.
    pub expert_gpus: usize,
    pub policy: LayerPolicy,
    /// What one residual hop moves, for the hops the chosen modes imply.
    pub hops: HopSpec,
    /// The modes and hops the family's executor runs; anything else is
    /// refused at plan time.
    pub executor: ExecutorModes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inventory {
    pub gpus: Vec<GpuBudget>,
    /// Spark expert ranks; 0 makes every routed layer mandatory on RTX.
    pub spark_ranks: usize,
    pub peer_access: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBudget {
    pub capacity_bytes: u64,
    /// Kept free for runtime growth and allocator slack; never allocated.
    pub headroom_bytes: u64,
    pub baseline: Baseline,
}

/// Where a GPU's free bytes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Baseline {
    /// Nothing allocated yet: the context/modules and every weight the
    /// planner already laid out (`loaded_bytes`) are charged here.
    Planned { context_bytes: u64, loaded_bytes: u64 },
    /// One CUDA sample after the context, modules and whatever the family
    /// loaded before admission; every demand in the request is still future.
    Measured { free_bytes: u64 },
}

impl GpuBudget {
    /// Bytes the request's demands may use.
    pub fn available(&self) -> u64 {
        match self.baseline {
            Baseline::Planned { context_bytes, loaded_bytes } => self.capacity_bytes
                .saturating_sub(self.headroom_bytes).saturating_sub(context_bytes).saturating_sub(loaded_bytes),
            Baseline::Measured { free_bytes } => free_bytes.saturating_sub(self.headroom_bytes),
        }
    }
}

/// Attention class of a layer: what the policy and the memory lever key on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionClass {
    /// DeepSeek V4/V4.1 compressed sparse attention (latent replicated under a split).
    Csa,
    Mla,
    Dsa,
    Gqa,
    Swa,
    Kda,
    Gdn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerDemand {
    pub kind: AttentionClass,
    /// Coordinator weights of the layer that the solver charges per mode
    /// (0 when the family loads them before admission, inside the baseline).
    pub weights: ModeBytes,
    /// Pool bytes per unit: whole on the owner, or per rank under a split
    /// (halves for partitioned heads, full copies for replicated latents).
    pub kv_unit: ModeBytes,
    /// Routed experts; `None` for a dense layer.
    pub experts: Option<ExpertCost>,
    /// The modes this build can execute for the layer, in preference order.
    pub modes: Vec<LayerMode>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModeBytes {
    pub whole: u64,
    pub split: [u64; 2],
}

impl ModeBytes {
    pub fn replicated(bytes: u64) -> Self { Self { whole: bytes, split: [bytes, bytes] } }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bytes2 {
    pub resident: u64,
    /// Transient load peak beside the resident bytes.
    pub staging: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertCost {
    pub whole: Bytes2,
    /// Each rank's half under TP2 (rank 0, rank 1); equal for even splits.
    pub half: [Bytes2; 2],
    /// Half-width TP2 layers exist for this build (placement PR 4): on two
    /// RTX with peer access the layer's RTX home is `RtxTp2`, never `RtxWhole`.
    pub tp2: bool,
    pub spark_ok: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum LayerMode {
    /// Attention heads split over both GPUs; the residual is replicated.
    HeadSplit,
    /// One GPU owns attention, KV and the layer's work.
    Whole { gpu: u8, ffn: FfnMode },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnMode {
    /// TP2 halves plus the fused all-reduce (PR 4).
    Split,
    /// The owner reduces (V4.1).
    Owner,
}

/// What a family's executor can run. The solver refuses a placement whose
/// layers would need another mode, and engines check the placement they are
/// handed against the same set before allocating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorModes {
    pub family: &'static str,
    /// Every mode the executor runs, on any layer.
    pub modes: &'static [LayerMode],
    /// Whether it runs hop points beyond the entry broadcast (ownership
    /// boundaries, split-FFN broadcasts, the exit hop to the head's GPU).
    pub hops: bool,
}

impl ExecutorModes {
    pub fn runs(&self, mode: LayerMode) -> bool {
        self.modes.contains(&mode)
    }

    /// Refuses `placement` unless every layer's mode and every hop is one
    /// this executor runs.
    pub fn check(&self, placement: &Placement) -> Result<(), PlacementError> {
        if let Some((layer, assignment)) = placement.layers.iter().enumerate().find(|(_, a)| !self.runs(a.mode)) {
            return Err(PlacementError::UnsupportedMode { family: self.family, layer, mode: assignment.mode });
        }
        match placement.hops.iter().find(|h| h.at != HopPoint::Entry) {
            Some(hop) if !self.hops => Err(PlacementError::UnsupportedHop { family: self.family, hop: *hop }),
            _ => Ok(()),
        }
    }
}

/// Mode preference by attention class; classes absent use `default`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LayerPolicy {
    pub default: Vec<LayerMode>,
    pub by_kind: Vec<(AttentionClass, Vec<LayerMode>)>,
}

impl LayerPolicy {
    pub fn preference(&self, kind: AttentionClass) -> &[LayerMode] {
        self.by_kind.iter().find(|(k, _)| *k == kind).map_or(&self.default, |(_, modes)| modes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Demand {
    pub gpu: u8,
    pub category: Category,
    pub group: String,
    pub bytes: u64,
    pub basis: Basis,
}

impl Demand {
    pub fn new(gpu: u8, category: Category, group: impl Into<String>, bytes: u64, basis: Basis) -> Self {
        Self { gpu, category, group: group.into(), bytes, basis }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovableId {
    /// dSpark stage experts (V4); loaded into the GPU's local expert arena.
    DsparkExperts,
    Drafter,
    Vision,
    Audio,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movable {
    pub id: MovableId,
    /// Loaded in order; each part's staging is transient.
    pub parts: Vec<Bytes2>,
    /// Allowed GPUs; the solver takes the one with the most free bytes.
    pub allowed: Vec<u8>,
    /// Shares the GPU's local expert arena (workspace and peak accounting).
    pub expert_arena: bool,
}

/// How many routed-expert layers are RTX-resident (`RTX_EXPERT_LAYERS`,
/// `--rtx-expert-layers`: `auto`, `N`, `N%`, `all`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "onboard", content = "value")]
pub enum Onboard {
    /// Reserve the pool target first, then place expert layers in what is
    /// left (Spark-free: every layer, the pool takes the rest).
    #[default]
    Auto,
    /// Exactly this many routed layers on RTX; the pool fills every
    /// remaining byte (clamped to the policy's ceiling) and is refused below
    /// its floor. TP2 halves count as one layer.
    Layers(usize),
    /// This fraction of the routed layers, rounded to the nearest layer.
    Fraction(f64),
    /// Experts first (v2's V4 policy): the most routed layers that still
    /// leave a pool of `pool_floor` tokens; the pool then takes what is left
    /// up to the target. `RTX_EXPERT_LAYERS=max`.
    ExpertsFirst { pool_floor: u64 },
}

/// v2's V4 expert-first reserve: experts fill the GPUs above a 262K pool.
pub const EXPERTS_FIRST_POOL_FLOOR: u64 = 262_144;

impl Onboard {
    /// Routed layers this onboard fixes out of `routed`, or `None` for `Auto`.
    pub fn layers(self, routed: usize) -> Option<usize> {
        match self {
            Self::Auto | Self::ExpertsFirst { .. } => None,
            Self::Layers(n) => Some(n.min(routed)),
            Self::Fraction(f) => Some(((routed as f64 * f.clamp(0.0, 1.0)) + 0.5).floor() as usize),
        }
    }
}

impl std::fmt::Display for Onboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Layers(n) => write!(f, "{n}"),
            Self::Fraction(x) => write!(f, "{}%", x * 100.0),
            Self::ExpertsFirst { .. } => f.write_str("max"),
        }
    }
}

impl std::str::FromStr for Onboard {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let invalid = || format!("RTX expert layers must be auto, max, all, N or N% (got {text:?})");
        match text {
            "auto" | "" => Ok(Self::Auto),
            "all" => Ok(Self::Fraction(1.0)),
            "max" => Ok(Self::ExpertsFirst { pool_floor: EXPERTS_FIRST_POOL_FLOOR }),
            _ => match text.strip_suffix('%') {
                Some(percent) => percent.parse::<f64>().ok().filter(|p| (0.0..=100.0).contains(p))
                    .map(|p| Self::Fraction(p / 100.0)).ok_or_else(invalid),
                None => text.parse().map(Self::Layers).map_err(|_| invalid()),
            },
        }
    }
}

/// What the solver decided. Equal requests give equal placements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub pool_tokens: u64,
    /// The onboard this placement resolved: RTX-resident routed layers.
    pub onboard_layers: usize,
    pub layers: Vec<LayerAssignment>,
    pub movables: Vec<(MovableId, u8)>,
    /// Contiguous whole-layer RTX expert range per GPU (`layers == 0`: none).
    /// Under TP2 every GPU's range is empty and [`Placement::tp2`] holds the layers.
    pub expert_ranges: Vec<ExpertRange>,
    /// The residual's home at each layer boundary: before layer `i` at `i`,
    /// after the last layer at `layers.len()`.
    pub residual: Vec<ResidualHome>,
    /// Every residual move the layer modes imply, in execution order.
    pub hops: Vec<Hop>,
    /// TP2 RTX expert halves (placement PR 4): one contiguous range of routed
    /// layers whose halves live on both GPUs, with each GPU's arena peak.
    pub tp2: Option<Tp2Range>,
    /// Every item the solver charged, per GPU (fixed demands, KV records,
    /// expert arenas); the baseline's loaded bytes are not repeated here.
    pub items: Vec<Vec<Item>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerAssignment {
    pub mode: LayerMode,
    pub experts: ExpertHome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "home")]
pub enum ExpertHome {
    Dense,
    /// Half-width layers on both GPUs (placement PR 4).
    RtxTp2,
    RtxWhole { gpu: u8 },
    Spark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertRange {
    pub first: usize,
    pub layers: usize,
    /// Arena peak: workspace, arena movables and every layer, with the
    /// largest transient load staging.
    pub peak_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tp2Range {
    pub first: usize,
    pub layers: usize,
    /// Each GPU's TP2 arena peak: workspace and half layers with transient
    /// load staging. Excludes the separate TP1 movable arena.
    pub peak_bytes: [u64; 2],
}

impl Placement {
    /// The `cuteafd plan --layout` line runtimes log once at admission.
    pub fn summary(&self) -> String {
        if let Some(t) = self.tp2 {
            return format!("pool {} tokens; onboard {} RTX expert layers: rtx0/rtx1: {} TP2 expert layer halves ({}..{}); arenas [{}, {}] B; TP1 dSpark {} B",
                self.pool_tokens, self.onboard_layers, t.layers, t.first, t.first + t.layers,
                t.peak_bytes[0], t.peak_bytes[1], self.expert_ranges[0].peak_bytes);
        }
        let ranges = self.expert_ranges.iter().enumerate()
            .map(|(gpu, r)| format!("rtx{gpu} {}..{} ({} B)", r.first, r.first + r.layers, r.peak_bytes))
            .collect::<Vec<_>>().join(", ");
        let hops = self.hops.iter().filter(|h| h.charged()).count();
        let hops = if hops == 0 { String::new() } else { format!("; {hops} residual hops") };
        format!("pool {} tokens; onboard {} RTX expert layers: {ranges}{hops}", self.pool_tokens, self.onboard_layers)
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PlacementError {
    #[error("placement inventory is invalid: {0}")]
    Inventory(&'static str),
    #[error("explicit KV pool of {requested} tokens does not fit before local experts (fits {fit})")]
    PoolDoesNotFit { requested: u64, fit: u64 },
    #[error("KV pool and mandatory {what} do not fit on rtx{gpu}")]
    Mandatory { gpu: u8, what: String },
    #[error("{requested} RTX expert layers do not fit beside the fixed demands (at most {placed})")]
    ExpertLayers { requested: usize, placed: usize },
    #[error("KV pool of {pool} tokens is below the {floor}-token floor after {layers} RTX expert layers (short {short} tokens)")]
    BelowFloor { pool: u64, floor: u64, layers: usize, short: u64 },
    #[error("Spark-free layout needs every routed layer on RTX: {placed} of {layers} fit")]
    SparkFree { layers: usize, placed: usize },
    #[error("layer {layer} has no mode this build executes (allowed {allowed:?})")]
    NoMode { layer: usize, allowed: Vec<LayerMode> },
    #[error("{family} cannot run layer {layer} as {mode:?}")]
    UnsupportedMode { family: &'static str, layer: usize, mode: LayerMode },
    #[error("{family} cannot run the residual hop {hop:?}")]
    UnsupportedHop { family: &'static str, hop: Hop },
    #[error("placement arithmetic overflows: {0}")]
    Overflow(&'static str),
}

impl From<PlacementError> for crate::serving_capacity::CacheGeometryError {
    fn from(error: PlacementError) -> Self {
        Self::ResidentTensor { name: "placement".into(), what: error.to_string() }
    }
}
