//! Family registry: detection, the canonical model a family parses from a
//! checkpoint's configuration, and tensor classification.
use super::checkpoint::Checkpoint;
use super::format::QuantOperand;
use super::spec::{Component, ModelSpec, TensorRole};
use crate::serving_capacity::{CacheGeometryError, CacheOptions, FamilyCacheGeometry};

/// Where a family's implementation stands in this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    /// Serves end to end.
    Serving,
    /// Recognized and planned; the execution path is not written yet.
    Planned,
}

/// Guidance a code agent needs to make one requirement work.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hint {
    pub what: String,
    pub how: String,
}

/// A configuration the family's runtime would refuse (the runtime parser's
/// own message, so the planner and the engine say the same thing).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ConfigError(pub String);

impl ConfigError {
    pub fn from_anyhow(error: anyhow::Error) -> Self {
        Self(format!("{error:#}"))
    }
}

/// How this build places a checkpoint's routed experts, by storage format:
/// the expert package family and the layouts it is built in.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ExpertContract {
    /// The package (`mimo:fp8`, `glm:exl3-k45`, `dsv4f:mxfp4`, ...).
    pub package: String,
    /// Intermediate rows per slice block: ranks own whole blocks (128 for
    /// FP8 and EXL3, 32 for MXFP4), each rank's slice stored padded to 128.
    pub block: usize,
    /// Spark worlds with a package layout (the transport runs 2, 3, 4 or 6).
    pub spark_worlds: Vec<usize>,
    /// Local-only placement (every routed layer on the coordinator GPU):
    /// the coordinator package and serve option, or why there is none.
    pub local: Result<String, String>,
}

/// A family's validated view of one checkpoint, parsed once: the spec, the
/// tensor contract and the expert placement all derive from it.
pub trait FamilyModel: Send + Sync {
    fn spec(&self) -> &ModelSpec;

    /// Placement accounts for executable runtime owners, not only checkpoint inventory.
    fn native_serving_admission(&self) -> bool { false }

    /// Resolve automatic vision to the mode this executor can actually admit.
    fn vision_mode(&self, mode: super::MediaMode) -> super::MediaMode { mode }

    /// The family's memory layout: its own adapter over the shared solver,
    /// solving the same request its runtime admits. `None` means the family
    /// has no adapter: `plan` refuses its layout.
    fn placement(&self, _input: &super::PlanInput, _report: &mut super::PlanReport,
        _options: &super::layout::LayoutOptions, _requested: super::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        None
    }

    /// Physical cache records and state from the same parsed configuration as
    /// the runtime. Weight/workspace admission remains a separate requirement.
    fn cache_geometry(&self, _options: CacheOptions) -> Result<Option<FamilyCacheGeometry>, CacheGeometryError> {
        Ok(None)
    }

    /// Routed-expert weight bytes one Spark rank may hold when the operator
    /// sets no `--spark-budget-gib`. Families whose release needs more state
    /// the measured reason with their override.
    fn spark_budget_bytes(&self) -> u64 { super::DEFAULT_SPARK_BUDGET_BYTES }

    /// Per-layer routed/dense or shared FFN arithmetic, when declared by a family.
    fn activation(&self, _layer: usize, _shared: bool) -> Option<cuteafd_core::FfnActivation> { None }

    /// The tensor contract: whether this build's loaders take `operand` (the
    /// logical weight `stem`) for `role`, and the precise reason when not.
    /// A contract may refine the operand (resolve a segmented scale grid).
    fn accepts(&self, _role: &TensorRole, _stem: &str, _operand: &mut QuantOperand) -> Result<(), String> {
        Err("the family has no execution path in this build".into())
    }

    /// Placement contract for routed experts stored as `operand`; `None`
    /// when no expert package serves that format.
    fn experts(&self, _operand: &QuantOperand) -> Option<ExpertContract> {
        None
    }
}

/// A [`FamilyModel`] that only describes (families without a runtime).
pub struct Described(pub ModelSpec);

impl FamilyModel for Described {
    fn spec(&self) -> &ModelSpec {
        &self.0
    }
}

pub trait Family: Sync {
    fn id(&self) -> &'static str;
    fn detect(&self, checkpoint: &Checkpoint) -> bool;
    fn runtime(&self) -> RuntimeStatus;
    /// Parses the checkpoint's configuration into the family's canonical model
    /// (the same parser its runtime uses), or the reason the runtime refuses it.
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError>;
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole>;
    /// Implementation notes for components this build cannot execute yet.
    fn component_hint(&self, component: Component) -> Option<Hint>;

    /// `component_hint` knowing the checkpoint's spec and the component's
    /// detected formats (labels); families whose checkpoints differ in
    /// storage (MiMo V2 Flash FP8 vs V2.6 Pro MXFP4) override it.
    fn component_hint_for(&self, _spec: &ModelSpec, component: Component, _formats: &[String]) -> Option<Hint> {
        self.component_hint(component)
    }

    /// Components the engine can serve without (a speculator it does not run yet).
    fn optional(&self, _component: Component) -> bool {
        false
    }

    /// Whether the Spark expert service (and the coordinator's local expert
    /// path) opens this family's routed experts through
    /// `read_expert_catalog`: the plan then runs it too, so the staging
    /// contract (storage maps, every projection's header) is checked as
    /// the service checks it.
    fn expert_catalog(&self) -> bool {
        false
    }
}

static REGISTRY: [&dyn Family; 11] = [
    &super::families::deepseek::DEEPSEEK_V41,
    &super::families::deepseek::DEEPSEEK_V4,
    &super::families::glm::GLM5,
    &super::families::glm::GLM5_FLASH,
    &super::families::mimo::MIMO_V2,
    &super::families::qwen::QWEN4,
    &super::families::qwen35::QWEN35,
    &super::families::step5::STEP5,
    &super::families::kimi::KIMI_K3,
    &super::families::dflash::DFLASH2,
    &super::families::dspark::DSPARK,
];

pub fn registry() -> &'static [&'static dyn Family] {
    &REGISTRY
}

pub fn detect(checkpoint: &Checkpoint) -> Option<&'static dyn Family> {
    registry().iter().copied().find(|family| family.detect(checkpoint))
}
