//! Family-independent description of a model, derived from its configuration.
use serde::Serialize;

/// How one decoder layer attends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AttentionKind {
    /// DeepSeek V4/V4.1 compressed MLA: one KV head, `ratio` 0 = window only.
    CompressedMla { ratio: usize, indexer: bool },
    /// MLA with a DeepSeek Sparse Attention indexer (GLM 5.x).
    MlaDsa { indexer: bool },
    /// Grouped-query attention over the full context.
    Gqa { heads: usize, kv_heads: usize, head_dim: usize },
    /// Grouped-query sliding-window attention.
    SlidingGqa { heads: usize, kv_heads: usize, head_dim: usize, window: usize, sinks: bool },
    /// Full causal MLA without an indexer (Kimi K3: no RoPE, sigmoid output gate).
    Mla { rope: bool, output_gate: bool },
    /// Kimi Delta Attention (linear recurrent).
    Kda,
    /// Gated DeltaNet (linear recurrent).
    GatedDeltaNet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FfnKind {
    Dense { intermediate: usize },
    Moe,
}

/// Rotary embedding of one layer's attention: the rotated leading dims of
/// each query/key head and the base.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RopeSpec {
    pub dims: usize,
    pub theta: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerSpec {
    pub attention: AttentionKind,
    pub ffn: FfnKind,
    /// `None` for layers without RoPE (linear attention, no-RoPE MLA) or
    /// families whose spec does not carry it.
    pub rope: Option<RopeSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MoeSpec {
    pub experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
    pub shared_experts: usize,
    pub shared_intermediate: usize,
    /// "softmax", "sigmoid", "sqrtsoftplus", ...
    pub scoring: String,
    pub routed_scaling: Option<f64>,
    /// Expert-choice groups (DeepSeek `n_group` / `topk_group`), when used.
    pub groups: Option<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SpeculatorSpec {
    /// Native multi-token prediction layers shipped in the checkpoint.
    NativeMtp { layers: usize },
    /// DeepSeek V4.x dSpark drafter stages shipped in the checkpoint.
    Dspark { stages: usize, experts: usize, top_k: usize, target_layers: Vec<usize> },
}

/// Resolved ownership of draft tensors in the target checkpoint. External
/// dSpark (GLM Flash) is deliberately distinct from bundled dSpark (V4.x).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftOwnership {
    None,
    NativeMtp,
    BundledDspark,
}

impl DraftOwnership {
    pub fn selected(family: &str, speculator: &str) -> Self {
        match (family, speculator) {
            ("mimo_v2" | "qwen4", "mtp") => Self::NativeMtp,
            ("deepseek_v4" | "deepseek_v41", "dspark") => Self::BundledDspark,
            _ => Self::None,
        }
    }

    pub fn uses_checkpoint(self) -> bool { self != Self::None }

    pub fn includes(self, component: super::Component) -> bool {
        !matches!(component, super::Component::Speculator | super::Component::SpeculatorExpert)
            || self.uses_checkpoint()
    }
}

/// A large lookup table that stays in host memory (engram, n-gram memory, PLE).
#[derive(Debug, Clone, Serialize)]
pub struct MappedTableSpec {
    pub name: String,
    pub layers: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSpec {
    pub family: &'static str,
    pub architecture: String,
    pub hidden: usize,
    pub vocab: usize,
    pub layers: Vec<LayerSpec>,
    pub moe: Option<MoeSpec>,
    pub speculator: Option<SpeculatorSpec>,
    pub tables: Vec<MappedTableSpec>,
    pub vision: bool,
    /// Free-form facts worth printing (compression schedule, hyper-connection width, ...).
    pub notes: Vec<String>,
}

impl ModelSpec {
    pub fn moe_layers(&self) -> usize {
        self.layers.iter().filter(|layer| layer.ffn == FfnKind::Moe).count()
    }
}

/// What a tensor is for. Families map names onto these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Embedding,
    LmHead,
    Norm,
    Attention,
    Indexer,
    Compressor,
    HyperConnection,
    /// Attention-residual block mixing (Kimi K3), not a hyper-connection.
    ResidualMix,
    Router,
    SharedExpert,
    RoutedExpert,
    /// Latent-MoE projections around the routed experts (down, norm, up).
    LatentProjection,
    DenseFfn,
    MappedTable,
    /// On-GPU projections that consume a mapped table (engram, PLE).
    TableProjection,
    Speculator,
    SpeculatorExpert,
    Vision,
    Audio,
    Other,
}

impl Component {
    pub fn label(self) -> &'static str {
        match self {
            Component::Embedding => "embedding",
            Component::LmHead => "lm_head",
            Component::Norm => "norm",
            Component::Attention => "attention",
            Component::Indexer => "indexer",
            Component::Compressor => "compressor",
            Component::HyperConnection => "hyper_connection",
            Component::ResidualMix => "residual_mix",
            Component::LatentProjection => "latent_projection",
            Component::Router => "router",
            Component::SharedExpert => "shared_expert",
            Component::RoutedExpert => "routed_expert",
            Component::DenseFfn => "dense_ffn",
            Component::MappedTable => "mapped_table",
            Component::TableProjection => "table_projection",
            Component::Speculator => "speculator",
            Component::SpeculatorExpert => "speculator_expert",
            Component::Vision => "vision",
            Component::Audio => "audio",
            Component::Other => "other",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorRole {
    pub component: Component,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
}

impl TensorRole {
    pub fn new(component: Component) -> Self {
        Self { component, layer: None, expert: None }
    }
    pub fn layer(component: Component, layer: usize) -> Self {
        Self { component, layer: Some(layer), expert: None }
    }
    pub fn expert(component: Component, layer: usize, expert: usize) -> Self {
        Self { component, layer: Some(layer), expert: Some(expert) }
    }
}
