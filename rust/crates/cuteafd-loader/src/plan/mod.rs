//! `cuteafd plan`: what a checkpoint is, where each part would live, and what
//! this build can or cannot run - with hints a code agent can act on.
pub mod checkpoint;
pub mod decode_lanes;
pub mod deployment;
pub mod experts;
pub mod exlr8;
pub mod encoder;
pub mod families;
pub mod family;
pub mod files;
pub mod format;
pub mod launch;
pub mod layout;
pub mod names;
pub mod rope;
pub mod spec;
pub mod spark_range;
#[doc(hidden)]
pub mod testing;

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::OnceLock;

pub use checkpoint::Checkpoint;
pub use family::{ConfigError, ExpertContract, Family, FamilyModel, Hint, RuntimeStatus};
pub use format::{Encoding, Malformed, QuantOperand, RowTiling, ScaleEncoding};
pub use spec::{AttentionKind, Component, FfnKind, ModelSpec, TensorRole};

/// Requested policy; never implies that an encoder has actually been loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaMode { Auto, Off, Rtx(Option<usize>), Spark(Option<usize>) }
impl std::str::FromStr for MediaMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto), "off" => Ok(Self::Off),
            "rtx" => Ok(Self::Rtx(None)), "spark" => Ok(Self::Spark(None)),
            _ => {
                let (kind, rank) = value.split_once(':').ok_or("media mode must be auto, off, rtx[:gpu] or spark[:rank]")?;
                let rank = rank.parse::<usize>().map_err(|_| "encoder device must be an unsigned integer")?;
                match kind { "rtx" => Ok(Self::Rtx(Some(rank))), "spark" => Ok(Self::Spark(Some(rank))),
                    _ => Err("media mode must be auto, off, rtx[:gpu] or spark[:rank]".into()) }
            }
        }
    }
}

/// Auto enables only the complete, qualified MiMo V2.6 Flash/Pro tower.
/// Explicit placement still reports missing or unsupported checkpoint tensors.
pub fn resolve_audio(mode: MediaMode, snapshot: &Path) -> Result<MediaMode, PlanError> {
    if mode == MediaMode::Off { return Ok(mode); }
    match crate::media::audio_tower::AudioTowerPlan::from_snapshot(snapshot,
        crate::media::audio_tower::AudioStorage::Fp32) {
        Ok(_) => Ok(mode),
        Err(_) if mode == MediaMode::Auto => Ok(MediaMode::Off),
        Err(error) => Err(PlanError::InvalidOption { option: "audio placement", reason: error.to_string() }),
    }
}

const GIB: f64 = (1u64 << 30) as f64;
/// Rejected tensors kept per component (the count covers the rest).
const REJECTIONS_KEPT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    Rtx,
    SparkSliced,
    HostMapped,
}

impl Owner {
    fn label(self, placement: ExpertPlacement) -> String {
        match (self, placement) {
            (Owner::Rtx, _) => "rtx".into(),
            (Owner::SparkSliced, ExpertPlacement::Sparks { ranks }) => format!("spark x{ranks}"),
            (Owner::SparkSliced, ExpertPlacement::Local) => "rtx (local)".into(),
            (Owner::HostMapped, _) => "host-mapped".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Executes in this build.
    Ready,
    /// The family runs, but this format for this component does not.
    MissingKernel,
    /// The family's execution path is not written yet.
    Planned,
    /// Optional for serving and not executed by this build (e.g. a speculator).
    Unused,
    /// Explicitly disabled by the deployment.
    Disabled,
}

/// A tensor (group) the family's loaders would not take, and why.
#[derive(Debug, Clone, Serialize)]
pub struct Rejection {
    pub tensor: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComponentPlan {
    pub component: Component,
    pub owner: Owner,
    pub tensors: usize,
    pub bytes: u64,
    /// Detected storage formats with the number of logical weights in each.
    pub formats: BTreeMap<String, usize>,
    pub status: Status,
    /// Logical weights the contract rejects (malformed or not executed).
    pub rejected: usize,
    /// The first of them, with reasons.
    pub rejections: Vec<Rejection>,
}

/// Where the routed experts live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ExpertPlacement {
    /// Every routed-expert layer resident on the coordinator GPU.
    Local,
    /// Tensor-parallel slices over `ranks` Spark ranks.
    Sparks { ranks: usize },
}

impl ExpertPlacement {
    /// The `--spark-ranks` spelling: 0 is the local-only placement.
    pub fn from_spark_ranks(ranks: usize) -> Self {
        if ranks == 0 { Self::Local } else { Self::Sparks { ranks } }
    }

    pub fn spark_ranks(self) -> usize {
        match self {
            Self::Local => 0,
            Self::Sparks { ranks } => ranks,
        }
    }
}

/// Planning options that cannot describe a deployment.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error(transparent)]
    SparkRange(#[from] spark_range::RangeRefusal),
    #[error("invalid {option}: {reason}")]
    InvalidOption { option: &'static str, reason: String },
    #[error("unsupported {option}={value} for {family}: {reason}")]
    UnsupportedOption { option: &'static str, value: u8, family: String, reason: &'static str },
    #[error(transparent)]
    Checkpoint(#[from] anyhow::Error),
}

/// Resolve the decode-lane count before layout or CUDA allocation
/// ([`decode_lanes::resolve`]).
pub fn resolve_decode_lanes(family: &str, requested: Option<u8>, concurrency: u64) -> Result<u8, PlanError> {
    decode_lanes::resolve(family, requested, concurrency)
}

#[derive(Debug, Clone, Serialize)]
pub struct RoleReadiness {
    pub role: String,
    pub required_shards: Vec<String>,
    pub unavailable: Vec<Rejection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanReport {
    pub mapped_placement: crate::formats::mapped_preload::MappedPlacement,
    /// Coordinator-host admission including estimated owner bounds, separate from mapped-scale preload.
    pub host_budget_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_ledger: Option<cuteafd_core::memory_layout::HostLayout>,
    pub attention_placement: crate::placement::AttentionPlacement,
    /// The routed-expert weight budget per Spark rank this plan used
    /// (explicit, or the family's default).
    pub spark_budget_bytes: u64,
    /// What the caller asked for (None: auto); families whose solver resolves
    /// auto itself (GLM 5.3) take this rather than the resolved default.
    #[serde(skip)]
    pub requested_attention: Option<crate::placement::AttentionPlacement>,
    pub vision: MediaMode,
    pub audio: MediaMode,
    pub disabled_media_bytes: u64,
    pub encoder: Option<encoder::EncoderPlacement>,
    pub audio_encoder: Option<encoder::EncoderPlacement>,
    pub encoder_plan_hash: String,
    /// Qualified family ceiling in merged image rows, even when vision is off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_image_tokens: Option<usize>,
    pub snapshot: String,
    pub family: Option<String>,
    pub architectures: Vec<String>,
    /// The checkpoint's ModelOpt quantization metadata (hf_quant_config.json /
    /// config.json), summarized; every weight was checked against it.
    pub quantization: Option<String>,
    /// Checkpoint declarations, before optional measured precision optimizations.
    pub module_numerics: BTreeMap<String, checkpoint::ModuleNumerics>,
    /// Effective launcher defaults, not claims about overrides on a running server.
    pub precision_optimizations: Vec<String>,
    /// Named deviations that still need a native arithmetic kernel.
    pub numerics_deviations: Vec<String>,
    pub runtime: Option<RuntimeStatus>,
    /// Successful native runtime-owner admission, never inferred from inventory layouts.
    #[serde(skip)]
    pub(crate) native_serving_admitted: bool,
    /// Why the family's runtime refuses this configuration, when it does.
    pub config_error: Option<String>,
    pub spec: Option<ModelSpec>,
    /// Canonical target cache storage, separate from the weight-only verdict.
    /// Complete serving admission also needs actual loaded representations,
    /// modules/workspaces, prefix marks and optional drafter reservations.
    pub cache_requirements: Option<crate::serving_capacity::CacheRequirements>,
    pub components: Vec<ComponentPlan>,
    pub unclassified: Vec<String>,
    pub missing_shards: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub role_readiness: Vec<RoleReadiness>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unneeded_missing_shards: Vec<String>,
    pub bytes_by_owner: BTreeMap<String, u64>,
    pub placement: ExpertPlacement,
    /// Spark ranks of the placement (0: local-only).
    pub spark_ranks: usize,
    /// A single diagnostic when an unspecified Spark count falls back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spark_count_warning: Option<String>,
    /// Policy refusal, distinct from absent executable packages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spark_range_refusal: Option<spark_range::RangeRefusal>,
    /// Resolved retained prefill span, not the row count of an individual unit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_span_rows: Option<u64>,
    /// Family-resolved rows per unit; never inferred from the retained span.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_unit_rows: Option<u64>,
    /// Family-resolved units in flight on each transport.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_waves: Option<u64>,
    /// Resolved decode views, independent of prefill units in flight.
    pub decode_lanes: u8,
    /// Solver-selected transport count, independent of units in flight.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_transports: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_selection: Option<String>,
    /// The package and layouts that serve this checkpoint's routed experts.
    pub experts: Option<ExpertContract>,
    /// Whether this build has a layout for the requested placement.
    pub placement_supported: bool,
    /// Fewest Spark ranks with a package layout whose budget holds every
    /// routed expert, if any does.
    pub min_spark_ranks: Option<usize>,
    /// Share of the routed-expert bytes the widest Spark rank holds: ranks
    /// own whole blocks of the intermediate (TP6 of 2048: 3 of 16).
    pub spark_rank_share: f64,
    pub fits: bool,
    /// Family and routed-expert layers for the launch scripts.
    pub launch: Option<launch::LaunchDescription>,
    /// Where the expert service read the routed EXL3 storage layout from.
    pub expert_storage: Option<crate::formats::exl3_storage::Exl3StorageSource>,
    pub hints: Vec<Hint>,
    /// Per-device memory layout, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_layout: Option<cuteafd_core::memory_layout::MemoryLayout>,
    /// Offline solver result; runtime admission still samples live resources.
    #[serde(skip)]
    pub solver_placement: Option<crate::placement::Placement>,
}

impl PlanReport {
    pub fn executable(&self) -> bool { self.launchable(false) }

    pub fn wip_launchable(&self) -> bool { self.launchable(true) }

    fn launchable(&self, wip: bool) -> bool {
        let planned = wip && self.runtime == Some(RuntimeStatus::Planned) && self.native_serving_admitted;
        (self.runtime != Some(RuntimeStatus::Planned) || planned)
            && self.family.is_some()
            && self.config_error.is_none()
            && self.missing_shards.is_empty()
            && self.role_readiness.iter().all(|role| role.unavailable.is_empty())
            && self.unclassified.is_empty()
            && self.components.iter().all(|c| matches!(c.status, Status::Ready | Status::Unused | Status::Disabled)
                || (planned && c.status == Status::Planned && c.rejected == 0))
            && self.placement_supported
            && self.fits
    }
}

#[derive(Debug, Clone)]
pub struct PlanOptions {
    /// None preserves each family's qualified decode-view default.
    pub decode_lanes: Option<u8>,
    pub mapped_scale_preload_bytes: u64,
    /// Available host budget override for offline planning; live cgroup-aware sample otherwise.
    pub mapped_scale_host_budget_bytes: Option<u64>,
    /// Resolved target-checkpoint draft ownership; absent keeps family defaults.
    pub draft_ownership: Option<spec::DraftOwnership>,
    pub attention_placement: Option<crate::placement::AttentionPlacement>,
    pub vision: MediaMode,
    pub audio: MediaMode,
    pub placement: ExpertPlacement,
    /// Routed-expert bytes one Spark rank may hold (weights only); `None`
    /// takes the family's default (`FamilyModel::spark_budget_bytes`).
    pub spark_budget_bytes: Option<u64>,
    /// Weight bytes the coordinator GPU may hold (its own tensors, plus every
    /// routed expert in the local-only placement).
    pub coordinator_budget_bytes: u64,
    /// Also lay out every device's memory (`plan --layout`).
    pub layout: Option<layout::LayoutOptions>,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            decode_lanes: None,
            mapped_scale_preload_bytes: crate::formats::mapped_preload::DEFAULT_CAP_BYTES,
            mapped_scale_host_budget_bytes: None,
            draft_ownership: None,
            attention_placement: None,
            vision: MediaMode::Auto,
            audio: MediaMode::Auto,
            placement: ExpertPlacement::Sparks { ranks: 6 },
            spark_budget_bytes: None,
            coordinator_budget_bytes: 80 << 30,
            layout: None,
        }
    }
}

impl PlanOptions {
    /// An explicit Spark ceiling applies to the complete layout as well as
    /// weights. Unset keeps the separate historical device/family defaults.
    pub fn apply_spark_budget(&mut self) {
        if let (Some(bytes), Some(layout)) = (self.spark_budget_bytes, &mut self.layout) {
            layout.spark_bytes = bytes;
            layout.spark_allocation_budget_bytes = Some(bytes);
        }
    }

    /// Rejects counts outside the shared world ceiling and empty budgets.
    /// Zero is the local-only placement.
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.decode_lanes.is_some_and(|lanes| !(1..=2).contains(&lanes)) {
            return Err(PlanError::InvalidOption { option: "--decode-lanes", reason: "must be 1 or 2".into() });
        }
        if let ExpertPlacement::Sparks { ranks } = self.placement {
            if !(1..=cuteafd_core::MAX_SPARK_WORLD).contains(&ranks) {
                return Err(PlanError::InvalidOption {
                    option: "spark ranks",
                    reason: format!("{ranks}: the expert transport runs 1 through 8 Spark ranks \
                        (0 places every routed expert on the coordinator)"),
                });
            }
        }
        for (option, bytes) in [("Spark budget", self.spark_budget_bytes.unwrap_or(DEFAULT_SPARK_BUDGET_BYTES)), ("coordinator budget", self.coordinator_budget_bytes)] {
            if bytes == 0 {
                return Err(PlanError::InvalidOption { option, reason: "must be positive".into() });
            }
        }
        Ok(())
    }
}

/// A scale preload cap accepts zero to disable it, unlike device budgets.
pub fn mapped_scale_preload_bytes(gib: f64) -> Result<u64, PlanError> {
    if gib == 0.0 { Ok(0) } else { budget_bytes("--mapped-scale-preload-gib", gib) }
}

/// The shared routed-expert weight budget per Spark rank, for families
/// without their own (`FamilyModel::spark_budget_bytes`).
pub const DEFAULT_SPARK_BUDGET_BYTES: u64 = 100 << 30;

impl PlanOptions {
    /// Routed-expert bytes per Spark rank: explicit, else the shared default
    /// (`plan` resolves an unset budget to the family's before using it).
    pub fn spark_budget(&self) -> u64 {
        self.spark_budget_bytes.unwrap_or(DEFAULT_SPARK_BUDGET_BYTES)
    }
}

/// `gib` as bytes: a finite, positive number of GiB that fits in u64.
pub fn budget_bytes(option: &'static str, gib: f64) -> Result<u64, PlanError> {
    let bytes = gib * GIB;
    if !gib.is_finite() || gib <= 0.0 || bytes < 1.0 || bytes >= u64::MAX as f64 {
        return Err(PlanError::InvalidOption { option, reason: format!("{gib} GiB is not a finite positive size") });
    }
    Ok(bytes as u64)
}

fn owner_for(component: Component) -> Owner {
    match component {
        Component::RoutedExpert => Owner::SparkSliced,
        Component::MappedTable => Owner::HostMapped,
        _ => Owner::Rtx,
    }
}

fn precision_optimizations(family: &str, options: &PlanOptions) -> Vec<String> {
    let mut active = Vec::new();
    match family {
        "glm5" => {
            if std::env::var("CUTEAFD_GLM_BF16").as_deref() != Ok("native") {
                active.push("attention, indexer and shared experts: load-time FP8 block weights (existing default; CUTEAFD_GLM_BF16=native disables)".into());
            }
            if !options.layout.as_ref().is_some_and(|layout| layout.glm5_drafter_disabled || layout.glm5_draft_bf16) {
                active.push("external drafter: FP8 weights (existing launcher default)".into());
            }
        }
        "glm5_flash" => {
            let repr = options.layout.as_ref().map(|layout| layout.glmf_representation).unwrap_or_default();
            active.push("MLA and shared experts: FP8 block weights; W8A8 prefill (existing launcher default)".into());
            if repr.kda_fp8 { active.push("KDA projections: row-K128 FP8 weights (existing launcher default)".into()); }
            if repr.fp8_head { active.push("head: row-K128 FP8 weights (existing launcher default)".into()); }
            if !options.layout.as_ref().is_some_and(|layout| layout.glmf_drafter_disabled) {
                active.push("external drafter: FP8 weights and W8A8 draft linear (existing launcher default)".into());
            }
        }
        "qwen4" => active.push("target/MTP shared head: FP8 weights (existing launcher default; attention projections remain checkpoint precision)".into()),
        _ => {}
    }
    active
}

fn numerics_deviations(family: &str, modules: &BTreeMap<String, checkpoint::ModuleNumerics>) -> Vec<String> {
    if family == "qwen4" && modules.iter().any(|(name, numerics)|
        name.starts_with("mtp.layers.") && name.contains(".mlp.experts.")
            && *numerics == checkpoint::ModuleNumerics::W8A8) {
        vec!["Qwen MTP experts: deviation from declared W8A8, kernel pending; existing FP8 expert path retained (both-projection dynamic K128 W8A8 is a named follow-up)".into()]
    } else { Vec::new() }
}

/// A checkpoint opened once, with the routed-expert catalog parsed from it:
/// what a plan is built from. Reading every shard header and building the
/// expert catalog dominate planning and neither changes between layouts, so
/// a caller that evaluates many plans over one snapshot opens it once and
/// asks `PlanInput` for each report (`plan_spark_count` walks one per
/// candidate Spark count; `cuteafd plan --layout` adds more).
pub struct PlanInput {
    checkpoint: Checkpoint,
    /// The routed-expert catalog, read on first use.
    catalog: OnceLock<anyhow::Result<crate::OfficialV41Catalog>>,
}

impl PlanInput {
    pub fn open(snapshot: &Path) -> anyhow::Result<Self> {
        Ok(Self::from_checkpoint(Checkpoint::open(snapshot)?))
    }

    /// Any role's checkpoint (`Checkpoint::open`, `::inventory`, `::coordinator`).
    pub fn from_checkpoint(checkpoint: Checkpoint) -> Self {
        Self { checkpoint, catalog: OnceLock::new() }
    }

    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }

    /// The routed-expert catalog of this checkpoint, read on first call and
    /// kept for the lifetime of the input. Families that stage no routed
    /// experts never ask for it, so no read happens for them.
    pub fn expert_catalog(&self) -> anyhow::Result<&crate::OfficialV41Catalog> {
        match self.catalog.get_or_init(|| crate::read_expert_catalog(&self.checkpoint.snapshot)) {
            Ok(catalog) => Ok(catalog),
            Err(error) => Err(anyhow::anyhow!("{error:#}")),
        }
    }

    /// The plan for `options` over this checkpoint.
    pub fn plan(&self, options: &PlanOptions) -> Result<PlanReport, PlanError> {
        plan_of(self, options)
    }
}

pub fn plan(snapshot: &Path, options: &PlanOptions) -> Result<PlanReport, PlanError> {
    PlanInput::open(snapshot)
        .map_err(|error| error.context(format!("reading checkpoint at {}", snapshot.display())))?
        .plan(options)
}

fn family_layout(model: &dyn FamilyModel, input: &PlanInput, report: &mut PlanReport,
    options: &mut layout::LayoutOptions, requested: ExpertPlacement)
    -> anyhow::Result<cuteafd_core::memory_layout::MemoryLayout> {
    match model.placement(input, report, options, requested) {
        Some(Ok(memory)) => {
            report.native_serving_admitted = model.native_serving_admission();
            Ok(memory)
        },
        Some(Err(error)) => {
            report.fits = false;
            report.placement_supported = false;
            report.hints.push(Hint { what: format!("native placement refused: {error:#}"),
                how: "Supply the native exports and a shape supported by the family's executor.".into() });
            Ok(cuteafd_core::memory_layout::MemoryLayout { devices: vec![], pool_tokens: 0,
                waste: vec![], notes: vec![format!("native placement refused: {error:#}")] })
        }
        None => {
            let family = report.family.as_deref().unwrap_or("unknown");
            report.fits = false;
            report.placement_supported = false;
            let drafter = matches!(family, "dflash2" | "dspark");
            let what = if drafter { format!("{family}: a drafter checkpoint has no memory layout of its own") }
                else { format!("{family}: no placement adapter in this build") };
            let how = if drafter {
                "Plan the target model with this snapshot as its drafter (--drafter-snapshot): the target's \
                    placement charges the drafter beside its own weights and KV pool.".to_string()
            } else {
                format!("Implement FamilyModel::placement for {family} (plan/families/), solving the same \
                    PlacementRequest its runtime admits, before claiming a memory layout.")
            };
            report.hints.push(Hint { what: what.clone(), how });
            Ok(cuteafd_core::memory_layout::MemoryLayout { devices: vec![], pool_tokens: 0, waste: vec![], notes: vec![what] })
        }
    }
}

fn plan_of(input: &PlanInput, options: &PlanOptions) -> Result<PlanReport, PlanError> {
    let mut report = plan_unrestricted(input, options)?;
    match spark_range::admit(input, options, &report) {
        Ok(()) => {},
        Err(PlanError::SparkRange(refusal)) => {
            report.placement_supported = false;
            report.hints.push(Hint { what: refusal.to_string(),
                how: "Choose a Spark count within the checkpoint's supported range.".into() });
            report.spark_range_refusal = Some(refusal);
        },
        Err(error) => return Err(error),
    }
    Ok(report)
}

pub(crate) fn empty_report(checkpoint: &Checkpoint, options: &PlanOptions) -> anyhow::Result<PlanReport> {
    Ok(PlanReport {
        mapped_placement: Default::default(),
        host_budget_bytes: 0,
        host_ledger: None,
        attention_placement: options.attention_placement.unwrap_or_default(),
        spark_budget_bytes: options.spark_budget(),
        requested_attention: options.attention_placement,
        vision: options.vision,
        audio: resolve_audio(options.audio, &checkpoint.snapshot)?,
        disabled_media_bytes: 0,
        encoder: None,
        audio_encoder: None,
        encoder_plan_hash: String::new(),
        max_image_tokens: None,
        snapshot: checkpoint.snapshot.display().to_string(),
        family: None,
        architectures: checkpoint.architectures(),
        quantization: None,
        module_numerics: checkpoint.declared_module_numerics()?,
        precision_optimizations: Vec::new(),
        numerics_deviations: Vec::new(),
        runtime: None,
        native_serving_admitted: false,
        config_error: None,
        spec: None,
        cache_requirements: None,
        components: Vec::new(),
        unclassified: Vec::new(),
        missing_shards: checkpoint.missing_shards.clone(),
        role_readiness: Vec::new(),
        unneeded_missing_shards: Vec::new(),
        bytes_by_owner: BTreeMap::new(),
        placement: options.placement,
        spark_ranks: options.placement.spark_ranks(),
        spark_count_warning: None,
        spark_range_refusal: None,
        prefill_span_rows: None,
        prefill_unit_rows: None,
        prefill_waves: None,
        decode_lanes: 1,
        prefill_transports: None,
        prefill_selection: None,
        experts: None,
        placement_supported: true,
        min_spark_ranks: None,
        spark_rank_share: 0.0,
        fits: true,
        launch: launch::describe(&checkpoint.config).ok(),
        expert_storage: None,
        hints: Vec::new(),
        memory_layout: None,
        solver_placement: None,
    })
}

fn plan_unrestricted(input: &PlanInput, options: &PlanOptions) -> Result<PlanReport, PlanError> {
    options.validate()?;
    let checkpoint = input.checkpoint();
    let mut report = empty_report(checkpoint, options)?;
    // ModelOpt exports describe what they quantized; each weight's tensors
    // must agree with that description.
    let modelopt = match crate::formats::modelopt::ModelOpt::read(&checkpoint.snapshot, &checkpoint.config) {
        Ok(modelopt) => modelopt,
        Err(error) => {
            report.hints.push(Hint {
                what: format!("unreadable quantization metadata: {error}"),
                how: "ModelOpt checkpoints are read from their own hf_quant_config.json and config.json \
                      quantization_config (cuteafd-loader/src/formats/modelopt.rs); extend the reader for a new \
                      algorithm or spelling.".into(),
            });
            report.config_error = Some(error.to_string());
            None
        }
    };
    report.quantization = modelopt.as_ref().map(|m| m.summary());
    let Some(family) = family::detect(&checkpoint) else {
        // No tensor contract or residency model was evaluated. An empty
        // inventory is not evidence that an unknown architecture fits.
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint {
            what: format!(
                "no family recognizes architectures {:?} (model_type {:?})",
                report.architectures,
                checkpoint.model_type()
            ),
            how: "Add a family under rust/crates/cuteafd-loader/src/plan/families/ implementing \
                  Family (detect, open, classify, component_hint) and register it in \
                  plan/family.rs::registry. Start from the closest existing family."
                .into(),
        });
        return Ok(report);
    };
    report.family = Some(family.id().into());
    report.decode_lanes = resolve_decode_lanes(family.id(), options.decode_lanes,
        options.layout.as_ref().map_or(2, |layout| layout.concurrency))?;
    report.precision_optimizations = precision_optimizations(family.id(), options);
    report.numerics_deviations = numerics_deviations(family.id(), &report.module_numerics);
    if let Some(executor) = crate::placement::families::executor(family.id()) {
        let gpus = options.layout.as_ref().map_or(1, |layout| layout.rtx_bytes.len());
        let peer = options.layout.as_ref().is_some_and(|layout| layout.head_split)
            && (family.id() != "qwen4" || options.placement == ExpertPlacement::Local);
        match executor.check_attention(options.attention_placement, gpus, peer) {
            Ok(mode) => report.attention_placement = mode,
            Err(error) => {
                report.fits = false;
                report.placement_supported = false;
                report.config_error = Some(error.to_string());
                report.hints.push(Hint { what: error.to_string(),
                    how: format!("Implement the {} attention executor before enabling this placement; use --attention-placement heads today.", family.id()) });
                return Ok(report);
            }
        }
    }
    if family.id() == "qwen4" && checkpoint.config.get("vision_config").is_some() {
        report.max_image_tokens = Some(crate::media::QWEN_MAX_IMAGE_TOKENS);
    }
    report.runtime = Some(family.runtime());
    let model = match family.open(&checkpoint).and_then(|mut model| {
        model.configure(options)?;
        Ok(model)
    }) {
        Ok(model) => model,
        Err(error) => {
            report.hints.push(Hint {
                what: format!("{} configuration is not servable: {error}", family.id()),
                how: format!("The {} runtime parses config.json with the same reader \
                    (cuteafd-loader/src/families/{}/config.rs); extend it and the engine together.",
                    family.id(), family.id()),
            });
            report.config_error.get_or_insert(error.0);
            return Ok(report);
        }
    };
    report.vision = model.vision_mode(options.vision);
    // An unset Spark budget is the family's own (Kimi K3 admits more than the
    // shared default); an explicit --spark-budget-gib always wins.
    let mut resolved = options.clone();
    if let Some(layout) = &mut resolved.layout { layout.decode_lanes = report.decode_lanes; }
    resolved.apply_spark_budget();
    resolved.spark_budget_bytes = Some(options.spark_budget_bytes.unwrap_or_else(|| model.spark_budget_bytes()));
    let options = &resolved;
    report.spark_budget_bytes = options.spark_budget();
    let spec = model.spec();
    report.spec = Some(spec.clone());
    let ownership = options.draft_ownership.unwrap_or_else(|| {
        let native = options.layout.as_ref().is_none_or(|layout| layout.native_mtp_layers > 0);
        spec::DraftOwnership::selected(family.id(), if !native { "off" }
            else if family.id() == "mimo_v2" && checkpoint.snapshot.join("dflash").is_dir() { "dflash2" }
            else if matches!(family.id(), "deepseek_v4" | "deepseek_v41") { "dspark" }
            else { "mtp" })
    });
    if !checkpoint.missing_shards.is_empty() {
        let mut requirements: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        let ranks = options.placement.spark_ranks();
        let mut read_roles = vec![files::ReadRole::Coordinator {
            local_experts: ranks == 0, speculator: ownership.uses_checkpoint() }];
        read_roles.extend((0..ranks).map(|rank| files::ReadRole::Spark { rank, world: ranks }));
        if report.vision != MediaMode::Off { read_roles.push(files::ReadRole::Vision); }
        if report.audio != MediaMode::Off { read_roles.push(files::ReadRole::Audio); }
        for role in read_roles {
            let names = files::required_tensors(&checkpoint, role)?;
            requirements.insert(role.label(), names.into_iter().filter_map(|name| {
                checkpoint.weight_map.get(&name).cloned().map(|shard| (name, shard))
            }).collect());
        }
        let mut needed = BTreeSet::new();
        for (role, tensors) in requirements {
            let shards: BTreeSet<_> = tensors.values().cloned().collect();
            needed.extend(shards.iter().cloned());
            let unavailable = tensors.iter().filter(|(name, _)|
                checkpoint.tensors.binary_search_by(|tensor| tensor.meta.name.as_str().cmp(name.as_str())).is_err())
                .map(|(tensor, shard)| Rejection { tensor: tensor.clone(), reason:
                    format!("role {role} lacks header for shard {shard} at snapshot {}", checkpoint.snapshot.display()) }).collect();
            report.role_readiness.push(RoleReadiness { role, required_shards: shards.into_iter().collect(), unavailable });
        }
        report.unneeded_missing_shards = checkpoint.missing_shards.iter()
            .filter(|shard| !needed.contains(*shard)).cloned().collect();
        report.missing_shards.retain(|shard| needed.contains(shard));
    }
    match crate::serving_capacity::cache_requirements(model.as_ref(), &checkpoint.config) {
        Ok(requirements) => report.cache_requirements = requirements,
        Err(error) => report.hints.push(Hint {
            what: format!("cache capacity cannot be described: {error}"),
            how: "Use a positive checkpoint max_position_embeddings and an implemented family cache geometry; \
                serving admission must resolve actual GPU and workspace reservations before allocating.".into(),
        }),
    }

    // Classify every tensor, then group each component's tensors by stem: one
    // operand per logical weight, checked against the family's contract.
    let mut by_component: BTreeMap<Component, Vec<checkpoint::CheckpointTensor>> = BTreeMap::new();
    let mut roles: BTreeMap<String, TensorRole> = BTreeMap::new();
    for tensor in &checkpoint.tensors {
        match family.classify(spec, &tensor.meta.name) {
            Some(role) => {
                roles.insert(format::split_stem(&tensor.meta.name).0.to_owned(), role.clone());
                by_component.entry(role.component).or_default().push(tensor.clone());
            }
            None => report.unclassified.push(tensor.meta.name.clone()),
        }
    }
    // Count logical random-read tensors, not storage shards or global scalars.
    let mapped = by_component.get(&Component::MappedTable).map(Vec::as_slice).unwrap_or(&[]);
    let mut names = BTreeSet::new();
    let mut scales = 0;
    let mut scale_bytes = 0;
    for tensor in mapped {
        let meta = &tensor.meta;
        if meta.shape.len() != 2 || meta.shape[0] <= 1 { continue }
        let (stem, suffix) = format::split_stem(&meta.name);
        if !matches!(suffix, "weight" | "scale" | "weight_scale" | "weight_scale_inv") { continue }
        // Qwen splits one row space into shard_N tensors.
        let logical = stem.rsplit_once(".shard_").map_or(stem, |(prefix, _)| prefix);
        if names.insert((logical.to_owned(), suffix.to_owned())) && matches!(suffix, "scale" | "weight_scale" | "weight_scale_inv") {
            scales += 1;
        }
        if matches!(suffix, "scale" | "weight_scale" | "weight_scale_inv") { scale_bytes += meta.byte_length; }
    }
    let host_budget = if scales == 0 { 0 } else {
        options.mapped_scale_host_budget_bytes.unwrap_or_else(||
            crate::formats::mapped_preload::host_budget_bytes().unwrap_or(0))
    };
    report.mapped_placement = crate::formats::mapped_preload::MappedPlacement::resolve(
        names.len(), scales, scale_bytes, options.mapped_scale_preload_bytes, host_budget);
    if report.mapped_placement.preloaded_bytes > 0 {
        report.bytes_by_owner.insert("host-preloaded".into(), report.mapped_placement.preloaded_bytes);
    }
    // The expert staging's own verdict on the routed experts.
    let catalog = (family.expert_catalog() && by_component.contains_key(&Component::RoutedExpert))
        .then(|| input.expert_catalog());
    report.expert_storage = catalog.as_ref().and_then(|catalog| catalog.as_ref().ok()?.exl3().map(|m| m.storage));
    let catalog_error = catalog.and_then(Result::err).map(|error| format!("{error:#}"));
    let mut hinted = BTreeSet::new();
    // Routed-expert operands by label, for the placement contract.
    let mut routed_operands: BTreeMap<String, QuantOperand> = BTreeMap::new();
    for (component, tensors) in &by_component {
        let groups = format::group_by_stem(tensors);
        let mut formats: BTreeMap<String, usize> = BTreeMap::new();
        let mut rejected = 0usize;
        let mut rejections = Vec::new();
        for (stem, members) in &groups {
            let role = roles.get(stem).cloned().unwrap_or_else(|| TensorRole::new(*component));
            let verdict = match format::detect(members) {
                Ok(mut operand) => {
                    let declared = modelopt.as_ref().map_or(Ok(()), |m| m.check(stem, &operand));
                    let accepted = declared.and_then(|()| model.accepts(&role, stem, &mut operand));
                    *formats.entry(operand.label()).or_default() += 1;
                    if accepted.is_ok() && *component == Component::RoutedExpert {
                        routed_operands.entry(operand.label()).or_insert(operand);
                    }
                    accepted.map_err(|reason| Rejection { tensor: stem.clone(), reason })
                }
                Err(malformed) => {
                    *formats.entry("malformed".into()).or_default() += 1;
                    Err(Rejection { tensor: malformed.tensor, reason: malformed.reason })
                }
            };
            if let Err(rejection) = verdict {
                rejected += 1;
                if rejections.len() < REJECTIONS_KEPT {
                    rejections.push(rejection);
                }
            }
        }
        if let (Component::RoutedExpert, Some(error)) = (component, &catalog_error) {
            rejected += 1;
            rejections.insert(0, Rejection { tensor: "routed experts (read_expert_catalog)".into(), reason: error.clone() });
            rejections.truncate(REJECTIONS_KEPT);
        }
        let disabled = (*component == Component::Vision && report.vision == MediaMode::Off)
            || (*component == Component::Audio && report.audio == MediaMode::Off)
            || !ownership.includes(*component);
        let status = if disabled { Status::Disabled } else { match family.runtime() {
            RuntimeStatus::Planned => Status::Planned,
            RuntimeStatus::Serving if rejected == 0 => Status::Ready,
            RuntimeStatus::Serving if family.optional(*component) => Status::Unused,
            RuntimeStatus::Serving => Status::MissingKernel,
        }};
        if !matches!(status, Status::Ready | Status::Unused | Status::Disabled) && hinted.insert(*component) {
            if let Some(first) = rejections.first() {
                report.hints.push(Hint {
                    what: format!("{}: {rejected} of {} weights not executable, e.g. {}: {}", component.label(),
                        groups.len(), first.tensor, first.reason),
                    how: "The family's tensor contract (plan/families/) mirrors its loaders; a new format needs \
                        the loader staging and kernels first, then the contract."
                        .into(),
                });
            }
            let labels: Vec<String> = formats.keys().cloned().collect();
            if let Some(hint) = family.component_hint_for(spec, *component, &labels) {
                if !report.hints.iter().any(|h| h.what == hint.what) {
                    report.hints.push(hint);
                }
            }
        }
        let source_bytes: u64 = tensors.iter().map(|t| t.meta.byte_length).sum();
        if disabled && matches!(component, Component::Vision | Component::Audio) {
            report.disabled_media_bytes += source_bytes;
        }
        let bytes = if disabled { 0 } else { source_bytes };
        let owner = owner_for(*component);
        report.components.push(ComponentPlan {
            component: *component,
            owner,
            tensors: tensors.len(),
            bytes,
            formats,
            status,
            rejected,
            rejections,
        });
    }
    place(&mut report, options, spec, model.as_ref(), &routed_operands);
    if let Some(layout_options) = &options.layout {
        let mut layout_options = layout_options.clone();
        if !ownership.uses_checkpoint() { layout_options.native_mtp_layers = 0; }
        layout_options.weight_budget_bytes = options.coordinator_budget_bytes;
        let memory = family_layout(model.as_ref(), input, &mut report, &mut layout_options, options.placement)?;
        report.prefill_unit_rows = layout_options.prefill_unit_rows;
        report.prefill_waves = Some(layout_options.prefill_waves);
        if layout_options.host_embedding {
            for note in memory.notes.iter().filter(|n| n.starts_with("host embedding ineligible:")) {
                report.fits = false;
                report.hints.push(Hint { what: note.clone(), how: "Use --embedding-placement gpu; host placement requires a distinct untied LM head.".into() });
            }
        }
        for device in &memory.devices {
            let required = device.used_bytes();
            if required > device.capacity_bytes {
                report.fits = false;
                report.hints.push(Hint {
                    what: format!("{} full memory layout needs {required} bytes, budget {} bytes, shortfall {} bytes",
                        device.name(), device.capacity_bytes, required - device.capacity_bytes),
                    how: "Weights, KV, workspaces, graphs and drafts must fit together; reduce the pool/placement \
                        or raise the device budget.".into(),
                });
            }
        }
        let refused = !report.placement_supported && memory.devices.is_empty();
        report.memory_layout = Some(memory);
        if !refused { layout::charge_host(&mut report, &layout_options, checkpoint, false)?; }
    }
    if !report.unclassified.is_empty() {
        report.hints.push(Hint {
            what: format!("{} tensors match no {} rule", report.unclassified.len(), family.id()),
            how: format!(
                "Extend {}::classify in cuteafd-loader/src/plan/families/ (first: {}).",
                family.id(),
                report.unclassified[0]
            ),
        });
    }
    if !report.missing_shards.is_empty() {
        report.hints.push(Hint {
            what: format!("{} roles lack required checkpoint headers", report.role_readiness.iter().filter(|role| !role.unavailable.is_empty()).count()),
            how: report.role_readiness.iter().filter_map(|role| role.unavailable.first())
                .map(|missing| format!("{}: {}", missing.tensor, missing.reason)).collect::<Vec<_>>().join("; "),
        });
    }
    if options.layout.is_none() {
        // A tensor-only inventory does not request an external speculation checkpoint.
        let mut inventory = layout::LayoutOptions { rtx_bytes: vec![options.coordinator_budget_bytes], spark_bytes: options.spark_budget(),
            spark_allocation_budget_bytes: Some(options.spark_budget()), glm5_drafter_disabled: true,
            glmf_drafter_disabled: true, native_mtp_layers: if ownership.uses_checkpoint() { 3 } else { 0 },
            weight_budget_bytes: options.coordinator_budget_bytes, ..Default::default() };
        let _ = family_layout(model.as_ref(), input, &mut report, &mut inventory, options.placement)?;
        report.prefill_unit_rows = inventory.prefill_unit_rows;
        report.prefill_waves = Some(inventory.prefill_waves);
    }
    if let Some(encoder) = &report.encoder {
        let owner = match &encoder.kind {
            encoder::EncoderKind::Rtx { gpu } => Some(format!("rtx{gpu} vision")),
            encoder::EncoderKind::Spark { rank } => Some(format!("spark{rank} vision")),
            encoder::EncoderKind::SparkIdle { host } => Some(format!("{host} vision")),
            encoder::EncoderKind::Off => None,
        };
        if let Some(owner) = owner { report.bytes_by_owner.insert(owner, encoder.admitted_bytes()); }
    }
    if let Some(encoder) = &report.audio_encoder {
        let owner = match &encoder.kind {
            encoder::EncoderKind::Rtx { gpu } => Some(format!("rtx{gpu} audio")),
            encoder::EncoderKind::Spark { rank } => Some(format!("spark{rank} audio")),
            encoder::EncoderKind::SparkIdle { host } => Some(format!("{host} audio")),
            encoder::EncoderKind::Off => None,
        };
        if let Some(owner) = owner { report.bytes_by_owner.insert(owner, encoder.admitted_bytes()); }
    }
    use sha2::{Digest, Sha256};
    let tower_headers: BTreeMap<_, _> = checkpoint.tensors.iter()
        .filter(|t| family.classify(spec, &t.meta.name).is_some_and(|r| r.component == Component::Vision
            || (report.audio != MediaMode::Off && r.component == Component::Audio)))
        .map(|t| (&t.meta.name, format!("{:?}:{:?}:{}:{}", t.meta.dtype, t.meta.shape, t.meta.byte_offset, t.meta.byte_length))).collect();
    // Preserve the vision-only peer contract exactly while audio is disabled.
    let contract = if report.audio == MediaMode::Off {
        serde_json::to_vec(&(&checkpoint.config, tower_headers, &report.family, report.placement,
            report.vision, report.audio, &report.encoder))
    } else {
        serde_json::to_vec(&(&checkpoint.config, tower_headers, &report.family, report.placement,
            report.vision, report.audio, &report.encoder, &report.audio_encoder))
    }.expect("plan serializes");
    let mut hash = Sha256::new();
    hash.update(contract);
    if let Some(cap) = report.max_image_tokens { hash.update((cap as u64).to_le_bytes()); }
    if report.audio != MediaMode::Off {
        if let Ok(plan) = crate::media::audio_tower::AudioTowerPlan::from_snapshot(&checkpoint.snapshot,
            crate::media::audio_tower::AudioStorage::Fp32) {
            for (name, read) in plan.reads() {
                hash.update(name.as_bytes());
                hash.update(format!("{:?}:{:?}:{}:{}", read.metadata.dtype, read.metadata.shape,
                    read.metadata.byte_offset, read.metadata.byte_length).as_bytes());
            }
        }
    }
    report.encoder_plan_hash = format!("{:x}", hash.finalize());
    if family.runtime() == RuntimeStatus::Planned && !report.native_serving_admitted {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint {
            what: "planned family has no executable serving admission in this build".into(),
            how: "The component bytes are a checkpoint inventory, not a serving-fit result; implement and gate the runtime first.".into(),
        });
    }
    Ok(report)
}

/// Places the routed experts (Spark slices or the coordinator) and checks
/// memory: the widest Spark rank against the Spark budget, the coordinator's
/// own tensors (plus every expert, local-only) against its budget.
fn place(report: &mut PlanReport, options: &PlanOptions, spec: &ModelSpec, model: &dyn FamilyModel,
    routed_operands: &BTreeMap<String, QuantOperand>) {
    if spec.moe.is_none() {
        report.placement = ExpertPlacement::Local;
        report.spark_ranks = 0;
        report.min_spark_ranks = Some(0);
    }
    let bytes_of = |owner: Owner| -> u64 {
        report.components.iter().filter(|c| c.owner == owner && !matches!(c.component, Component::Vision | Component::Audio)).map(|c| c.bytes).sum()
    };
    let (routed, rtx, mapped) = (bytes_of(Owner::SparkSliced), bytes_of(Owner::Rtx),
        bytes_of(Owner::HostMapped).saturating_sub(report.mapped_placement.preloaded_bytes));
    for (owner, bytes) in [(Owner::Rtx, rtx), (Owner::SparkSliced, routed), (Owner::HostMapped, mapped)] {
        if bytes > 0 {
            *report.bytes_by_owner.entry(owner.label(options.placement)).or_default() += bytes;
        }
    }
    // One contract for every routed format present (EXL3 K3 and K4 tiers
    // share theirs); mixed packages keep the layouts they all have.
    let contracts: Vec<ExpertContract> = routed_operands.values().filter_map(|op| model.experts(op)).collect();
    let contract = contracts.first().cloned().map(|mut first| {
        for other in &contracts[1..] {
            first.spark_worlds.retain(|w| other.spark_worlds.contains(w));
            if first.block != other.block {
                first.spark_worlds.clear();
            }
            if other.local.is_err() {
                first.local = other.local.clone();
            }
            if !first.package.split(" + ").any(|p| p == other.package) {
                first.package = format!("{} + {}", first.package, other.package);
            }
        }
        first
    });
    let intermediate = spec.moe.as_ref().map(|moe| moe.intermediate);
    let gib = |bytes: f64| bytes / GIB;
    let coordinator = options.coordinator_budget_bytes as f64;
    if routed > 0 {
        let share = |ranks: usize| -> Option<f64> {
            let contract = contract.as_ref()?;
            let i = intermediate?;
            // New generic counts require nonempty H128 ownership; a 16-row
            // NVFP4 fallback remains valid only for previously calibrated layouts.
            if matches!(ranks, 1 | 5) && (i % 128 != 0 || i / 128 < ranks) {
                return None;
            }
            experts::stored_slice(i, contract.block, ranks).map(|slice| slice as f64 / i as f64)
        };
        let fits_on = |ranks: usize| share(ranks).is_some_and(|s| routed as f64 * s <= options.spark_budget() as f64);
        report.min_spark_ranks = contract.as_ref().and_then(|c| c.spark_worlds.iter().copied().find(|&r| fits_on(r)));
        let advice = match report.min_spark_ranks {
            Some(ranks) => format!("Use {ranks} Spark ranks (--spark-ranks {ranks})"),
            None => "No Spark layout of this package holds them at this budget".into(),
        };
        match (options.placement, &contract) {
            (_, None) => {} // the routed-expert component already reports its missing kernel
            (ExpertPlacement::Sparks { ranks }, Some(contract)) => {
                report.spark_rank_share = share(ranks).unwrap_or(0.0);
                if share(ranks).is_none() {
                    report.placement_supported = false;
                    report.fits = false;
                    report.hints.push(Hint {
                        what: format!("geometry-invalid: {} intermediate {} cannot give each of {ranks} Spark ranks a nonempty whole-{} row slice",
                            contract.package, intermediate.unwrap_or(0), if matches!(ranks, 1 | 5) { 128 } else { contract.block }),
                        how: format!("{advice}; adding a package cannot repair empty or unaligned expert slices."),
                    });
                } else if !contract.spark_worlds.contains(&ranks) {
                    report.placement_supported = false;
                    report.hints.push(Hint {
                        what: format!("no {} layout for {ranks} Spark ranks (this build packages {:?}{})",
                            contract.package, contract.spark_worlds,
                            intermediate.map_or(String::new(), |i| format!(", intermediate {i} in {}-row blocks", contract.block))),
                        how: match share(ranks) {
                            Some(_) => format!("{advice}, or build a tp{ranks} layout (python/tools/aot/\
                                package_fp8_moe_aot.py --layouts / package_exl3_aot.py)."),
                            None => format!("{advice}: {ranks} ranks cannot each own whole {}-row blocks.",
                                contract.block),
                        },
                    });
                } else if !fits_on(ranks) {
                    report.fits = false;
                    report.hints.push(Hint {
                        what: format!("routed experts need {:.1} GiB on the widest of {ranks} Spark ranks, over the \
                            {:.0} GiB budget", gib(routed as f64 * report.spark_rank_share),
                            gib(options.spark_budget() as f64)),
                        how: format!("{advice}, keep bottom layers' experts resident on the RTX cards, or quantize \
                            the experts further (EXL3 K2-K3)."),
                    });
                }
            }
            (ExpertPlacement::Local, Some(contract)) => match &contract.local {
                Err(why) => {
                    report.placement_supported = false;
                    report.hints.push(Hint {
                        what: format!("unsupported: no local expert package for {}: {why}", spec.family),
                        how: format!("{advice}."),
                    });
                }
                // GLM Flash's selected solver ledger checks the full resident layout,
                // including TP2 halves; source bytes on one GPU are not its admission.
                Ok(_) if spec.family == "glm5_flash" && options.layout.is_some() => {}
                Ok(_) => {
                    let need = (rtx + routed) as f64;
                    if need > coordinator {
                        report.fits = false;
                        report.hints.push(Hint {
                            what: format!("local-only placement needs {:.1} GiB on the coordinator GPU ({:.1} GiB \
                                routed experts + {:.1} GiB other weights), {:.1} GiB over the {:.0} GiB budget",
                                gib(need), gib(routed as f64), gib(rtx as f64), gib(need - coordinator), gib(coordinator)),
                            how: format!("{advice}."),
                        });
                    }
                }
            },
        }
    }
    // The weight budget is per coordinator GPU; a head-split layout spreads
    // the backbone over each GPU it names.
    let gpus = options.layout.as_ref().filter(|l| l.head_split).map_or(1, |l| l.rtx_bytes.len().max(1));
    if options.placement != ExpertPlacement::Local && rtx as f64 > coordinator * gpus as f64 {
        report.fits = false;
        report.hints.push(Hint {
            what: format!("coordinator weights need {:.1} GiB, over the {:.0} GiB coordinator budget{}",
                gib(rtx as f64), gib(coordinator * gpus as f64), if gpus > 1 { format!(" ({gpus} GPUs)") } else { String::new() }),
            how: "Split the backbone over two coordinator GPUs or move components to Sparks.".into(),
        });
    }
    report.experts = contract;
}

/// Plan an exact count, or the widest feasible count in the available inventory.
/// Count policy never depends on card size; each candidate uses the same admission.
pub fn plan_spark_count(snapshot: &Path, options: &PlanOptions, available: usize,
    explicit: Option<usize>) -> Result<PlanReport, PlanError> {
    let input = PlanInput::open(snapshot)
        .map_err(|error| error.context(format!("reading checkpoint at {}", snapshot.display())))?;
    plan_spark_count_of(&input, options, available, explicit)
}

/// Count candidates differ only in placement, so every one shares the
/// checkpoint and catalog `input` already read.
fn plan_spark_count_of(input: &PlanInput, options: &PlanOptions, available: usize,
    explicit: Option<usize>) -> Result<PlanReport, PlanError> {
    if available > 8 {
        return Err(PlanError::InvalidOption { option: "available Sparks", reason: "inventory exceeds transport limit 8".into() });
    }
    let count = explicit.unwrap_or(available);
    let candidate = |ranks| -> Result<PlanReport, PlanError> {
        let mut candidate = PlanOptions {
            placement: ExpertPlacement::from_spark_ranks(ranks), ..options.clone()
        };
        let mut report = input.plan(&candidate)?;
        if report.family.as_deref() != Some("deepseek_v41") || candidate.layout.is_none() {
            return Ok(report);
        }
        let rows = candidate.layout.as_ref().unwrap().prefill_rows;
        report.prefill_span_rows = Some(if rows == 0 { 2048 } else { rows });
        if rows != 0 {
            if !report.executable() {
                report.hints.push(Hint {
                    what: format!("explicit V4.1 prefill capacity {rows} refused by checkpoint or memory admission"),
                    how: "Keep the requested shape with more memory, or omit prefill rows for automatic admission.".into(),
                });
            }
            return Ok(report);
        }
        // Count candidates each solve their largest executable default shape.
        let reason = report.hints.iter().map(|hint| hint.what.as_str()).collect::<Vec<_>>().join("; ");
        if !report.executable() {
            for rows in [1024, 256] {
                let layout = candidate.layout.as_mut().unwrap();
                layout.prefill_rows = rows;
                layout.spark_capacity_rows = rows;
                let mut smaller = input.plan(&candidate)?;
                smaller.prefill_span_rows = Some(rows);
                if smaller.executable() {
                    smaller.prefill_selection = Some(format!(
                        "V4.1 automatic prefill selected {rows} rows (2048 refused: {reason})"));
                    return Ok(smaller);
                }
            }
        } else {
            report.prefill_selection = Some("V4.1 automatic prefill selected 2048 rows".into());
        }
        Ok(report)
    };
    let mut requested = candidate(count)?;
    if count > available {
        requested.placement_supported = false;
        requested.hints.push(Hint {
            what: format!("requested {count} Sparks but only {available} are available"),
            how: "Choose a count within the available inventory.".into(),
        });
    }
    if requested.wip_launchable() { return Ok(requested); }
    let reason = requested.hints.iter().map(|hint| hint.what.as_str()).collect::<Vec<_>>().join("; ");
    let reason = if reason.is_empty() { requested.config_error.clone().unwrap_or_else(||
        "checkpoint tensors or required package artifacts unavailable".into()) } else { reason };
    let mut working = Vec::new();
    let mut best = None;
    for ranks in (0..=available).rev().filter(|&ranks| ranks != count) {
        let report = candidate(ranks)?;
        if report.wip_launchable() {
            working.push(ranks);
            if best.is_none() { best = Some(report); }
        }
    }
    if explicit.is_none() {
        if let Some(mut report) = best {
            report.spark_count_warning = Some(format!(
                "requested all {available} available Sparks: {reason}; using {} Sparks", report.spark_ranks));
            return Ok(report);
        }
    }
    working.sort_unstable();
    requested.hints.push(Hint {
        what: format!("Spark count {count} refused: {reason}; counts that work: {working:?}"),
        how: "Choose a working SPARK_COUNT/SPARK_HOSTS or change the reported admission constraint.".into(),
    });
    Ok(requested)
}

/// Unspecified placement uses all available Sparks before considering fallback.
pub fn plan_preferred(snapshot: &Path, options: &PlanOptions) -> Result<PlanReport, PlanError> {
    plan_spark_count(snapshot, options, options.placement.spark_ranks(), None)
}

/// Human-readable report.
pub fn render(report: &PlanReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "decode     {} lane(s)", report.decode_lanes);
    if report.mapped_placement.mapped + report.mapped_placement.preloaded > 0 {
        let p = &report.mapped_placement;
        let _ = writeln!(out, "{}; scales {} bytes / cap {} bytes; host preload {} bytes / budget {} bytes ({})",
            p.label(), p.scale_bytes, p.cap_bytes, p.preloaded_bytes, p.host_budget_bytes, p.reason);
    }
    if let Some(host) = &report.host_ledger {
        let _ = writeln!(out, "host       cache {} bytes; consumers {} bytes; reserve {} bytes; headroom {} bytes; shortfall {} bytes",
            host.host_budget_bytes, host.deferred_bytes, host.reserve_bytes, host.headroom_bytes, host.shortfall_bytes);
        for item in &host.items {
            let _ = writeln!(out, "host item  {}: {} bytes ({:?})", item.group, item.bytes, item.basis);
        }
    }
    if let Some(warning) = &report.spark_count_warning { let _ = writeln!(out, "warning: {warning}"); }
    if let Some(selection) = &report.prefill_selection { let _ = writeln!(out, "{selection}"); }
    if report.disabled_media_bytes > 0 { out.push_str(&format!("media disabled: {} bytes saved\n", report.disabled_media_bytes)); }
    if let Some(cap) = report.max_image_tokens {
        let _ = writeln!(out, "image cap  {cap} merged tokens per image (detail=low 256); BF16 residual");
    }
    if let Some(encoder) = &report.encoder {
        let arch = if matches!(encoder.kind, encoder::EncoderKind::Rtx { .. }) { "vision.sm120" } else { "vision.sm121" };
        let attention = match report.family.as_deref() {
            Some("qwen4") => "Qwen full-image head72 attention",
            Some("glm5_flash") => "GLM Flash full-image head64 attention",
            _ => "MiMo key-0 attention",
        };
        let _ = writeln!(out, "vision     {:?}: {}; weights {} scratch {} bytes; {} ({attention}); shortfall {} bytes", encoder.kind, encoder.reason, encoder.weights, encoder.scratch, arch, encoder.shortfall);
        let _ = writeln!(out, "media hash {}", report.encoder_plan_hash);
    }
    if let Some(encoder) = &report.audio_encoder {
        let arch = if matches!(encoder.kind, encoder::EncoderKind::Rtx { .. }) { "audio.sm120" } else { "audio.sm121" };
        let _ = writeln!(out, "audio      {:?}: {}; weights {} scratch {} bytes; {}; shortfall {} bytes", encoder.kind,
            encoder.reason, encoder.weights, encoder.scratch, arch, encoder.shortfall);
    }
    let _ = writeln!(out, "snapshot   {}", report.snapshot);
    let _ = writeln!(out, "arch       {}", report.architectures.join(", "));
    if let Some(quantization) = &report.quantization {
        let _ = writeln!(out, "quant      {quantization}");
    }
    for (module, numerics) in &report.module_numerics {
        let _ = writeln!(out, "numerics   {module}: {numerics:?} (checkpoint declaration)");
    }
    for optimization in &report.precision_optimizations {
        let _ = writeln!(out, "precision  {optimization}");
    }
    for deviation in &report.numerics_deviations {
        let _ = writeln!(out, "deviation  {deviation}");
    }
    match (&report.family, &report.spec) {
        (Some(family), Some(spec)) => {
            let runtime = match report.runtime {
                Some(RuntimeStatus::Serving) => "serving",
                _ => "planned (execution path not written yet)",
            };
            let _ = writeln!(out, "family     {family}: {runtime}");
            let moe_layers = spec.moe_layers();
            let _ = write!(
                out,
                "spec       hidden {}, vocab {}, {} layers ({} MoE)",
                spec.hidden,
                spec.vocab,
                spec.layers.len(),
                moe_layers
            );
            if let Some(moe) = &spec.moe {
                let _ = write!(
                    out,
                    ", {} experts top-{} (intermediate {}, {} shared, {})",
                    moe.experts, moe.top_k, moe.intermediate, moe.shared_experts, moe.scoring
                );
            }
            let _ = writeln!(out);
            let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
            for layer in &spec.layers {
                let label = match &layer.attention {
                    AttentionKind::CompressedMla { ratio, indexer } => {
                        format!("compressed-mla r{ratio}{}", if *indexer { "+idx" } else { "" })
                    }
                    AttentionKind::MlaDsa { indexer } => format!("mla-dsa{}", if *indexer { "+idx" } else { "(shared idx)" }),
                    AttentionKind::Gqa { heads, kv_heads, head_dim } => format!("gqa {heads}/{kv_heads}x{head_dim}"),
                    AttentionKind::SlidingGqa { window, sinks, .. } => {
                        format!("swa-gqa w{window}{}", if *sinks { "+sink" } else { "" })
                    }
                    AttentionKind::Kda => "kda".into(),
                    AttentionKind::Mla { rope, output_gate } => format!("mla{}{}",
                        if *rope { "" } else { " nope" }, if *output_gate { "+gate" } else { "" }),
                    AttentionKind::GatedDeltaNet => "gated-deltanet".into(),
                };
                *kinds.entry(label).or_default() += 1;
            }
            let _ = writeln!(
                out,
                "attention  {}",
                kinds.iter().map(|(k, n)| format!("{k} x{n}")).collect::<Vec<_>>().join(", ")
            );
            if let Some(speculator) = &spec.speculator {
                let _ = writeln!(out, "speculator {}", serde_json::to_string(speculator).unwrap_or_default());
            }
            for table in &spec.tables {
                let _ = writeln!(out, "table      {} on layers {:?}", table.name, table.layers);
            }
            for note in &spec.notes {
                let _ = writeln!(out, "note       {note}");
            }
        }
        (Some(family), None) => {
            let _ = writeln!(out, "family     {family}: configuration refused ({})",
                report.config_error.as_deref().unwrap_or("no spec"));
        }
        _ => {
            let _ = writeln!(out, "family     not recognized");
        }
    }
    if !report.components.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "{:<18} {:>8} {:>9}  {:<13} {:<9} formats", "component", "tensors", "GiB", "placement", "status");
        for c in &report.components {
            let formats = c.formats.iter().map(|(f, n)| format!("{f} x{n}")).collect::<Vec<_>>().join(", ");
            let status = match c.status {
                Status::Ready => "ready",
                Status::MissingKernel => "MISSING",
                Status::Planned => "planned",
                Status::Unused => "unused",
                Status::Disabled => "disabled",
            };
            let _ = writeln!(
                out,
                "{:<18} {:>8} {:>9.2}  {:<13} {:<9} {}",
                c.component.label(),
                c.tensors,
                c.bytes as f64 / GIB,
                c.owner.label(report.placement),
                status,
                formats
            );
        }
        if let Some(storage) = report.expert_storage {
            let _ = writeln!(out, "exl3 map   {}", storage.describe());
        }
        let rejected: Vec<&ComponentPlan> =
            report.components.iter().filter(|c| c.status == Status::MissingKernel && c.rejected > 0).collect();
        if !rejected.is_empty() {
            let _ = writeln!(out, "\nrejected weights:");
            for c in rejected {
                for r in c.rejections.iter().take(3) {
                    let _ = writeln!(out, "  {:<16} {}: {}", c.component.label(), r.tensor, r.reason);
                }
                if c.rejected > 3 {
                    let _ = writeln!(out, "  {:<16} ... {} more", c.component.label(), c.rejected - 3);
                }
            }
        }
        let _ = writeln!(out);
        for (owner, bytes) in &report.bytes_by_owner {
            let per = if owner.starts_with("spark") {
                format!(" ({:.1} GiB on the widest rank)", *bytes as f64 * report.spark_rank_share / GIB)
            } else {
                String::new()
            };
            let _ = writeln!(out, "total {:<12} {:>9.2} GiB{per}", owner, *bytes as f64 / GIB);
        }
    }
    if let Some(cache) = &report.cache_requirements {
        let context = cache.checkpoint_max_context_tokens.map_or("missing".to_string(), |v| v.to_string());
        let capability = if cache.compiled_index_extent_required {
            "serving manifest must provide the compiled index extent"
        } else { "dynamic context extent" };
        let _ = writeln!(out, "context    checkpoint maximum {context}; {capability}");
        let floor = cache.requested_kv_floor_tokens.map_or("unresolved".to_string(), |v| v.to_string());
        let _ = writeln!(out, "KV target  {floor} tokens (common default pool); C{} with {} state/ring slots",
            cache.concurrency, cache.state_slots);
        for layout in &cache.target_only_layouts {
            let placement = match layout.placement {
                crate::serving_capacity::KvPlacement::SingleDevice => "one owner",
                crate::serving_capacity::KvPlacement::Replicated => "replicated KV",
                crate::serving_capacity::KvPlacement::PartitionedHeads => "partitioned KV heads",
                crate::serving_capacity::KvPlacement::PartitionedLayers => "partitioned KV layers/sources",
                crate::serving_capacity::KvPlacement::PartitionedContext => "partitioned KV context",
            };
            for (rank, cost) in layout.ranks.iter().enumerate() {
                let bytes_per_token = cost.persistent_unit_bytes as f64 / layout.logical_unit_rows as f64;
                let state_gib = cost.active_state_per_sequence_bytes as f64 * f64::from(cache.state_slots) / GIB;
                let pool = cache.requested_kv_floor_tokens.map_or("unresolved".to_string(), |tokens| {
                    let units = tokens.div_ceil(layout.logical_unit_rows);
                    format!("{:.2} GiB", units as f64 * cost.persistent_unit_bytes as f64 / GIB)
                });
                let _ = writeln!(out, "  {} RTX rank {rank}: {placement}, {bytes_per_token:.0} KV B/token, \
                    target {pool}, active state {state_gib:.2} GiB, mark {:.2} MiB/slot",
                    layout.ranks.len(), cost.retained_mark_bytes as f64 / (1u64 << 20) as f64);
            }
        }
        for unavailable in &cache.unavailable_layouts {
            let _ = writeln!(out, "  {} RTX cache layout: {}", unavailable.coordinator_ranks, unavailable.reason);
        }
        let _ = writeln!(out, "  storage costs only: runtime admission must also reserve actual weights, modules, \
            all workspace shapes, prefix marks, transport and optional drafters against each GPU's live budget");
    }
    if !report.unclassified.is_empty() {
        let _ = writeln!(out, "\nunclassified tensors: {}", report.unclassified.len());
        for name in report.unclassified.iter().take(10) {
            let _ = writeln!(out, "  {name}");
        }
    }
    for role in &report.role_readiness {
        let _ = writeln!(out, "role {}: {} required shards, {} unavailable tensor headers", role.role,
            role.required_shards.len(), role.unavailable.len());
        for missing in role.unavailable.iter().take(3) {
            let _ = writeln!(out, "  {}: {}", missing.tensor, missing.reason);
        }
    }
    if !report.unneeded_missing_shards.is_empty() {
        let _ = writeln!(out, "absent unneeded shards (info): {:?}", report.unneeded_missing_shards);
    }
    if !report.missing_shards.is_empty() {
        let _ = writeln!(out, "\nmissing shards: {}", report.missing_shards.len());
    }
    if let Some(experts) = &report.experts {
        let local = match &experts.local {
            Ok(package) => package.clone(),
            Err(_) => "none".into(),
        };
        let _ = writeln!(out, "experts    {}: Spark worlds {:?}; local {local}", experts.package, experts.spark_worlds);
    }
    if report.spec.is_some() && report.bytes_by_owner.keys().any(|o| o != "rtx") {
        let min = report.min_spark_ranks.map_or("no Spark layout fits".into(), |r| format!(">= {r} Spark ranks"));
        let at = match report.placement {
            ExpertPlacement::Local => "local-only".into(),
            ExpertPlacement::Sparks { ranks } => format!("{ranks} ranks"),
        };
        let verdict = match (report.placement_supported, report.fits) {
            (false, _) => "UNSUPPORTED",
            (true, true) => "fits",
            (true, false) => "DOES NOT FIT",
        };
        let _ = writeln!(out, "capacity   routed experts need {min}; at {at}: {verdict} (Spark budget {:.1} GiB/rank)",
            report.spark_budget_bytes as f64 / GIB);
    }
    let _ = writeln!(out, "\nverdict    {}", if report.executable() { "READY to serve" } else { "NOT servable by this build" });
    if !report.hints.is_empty() {
        let _ = writeln!(out, "\n--- hints for a code agent ---");
        for hint in &report.hints {
            let _ = writeln!(out, "* {}\n  {}", hint.what, hint.how);
        }
    }
    out
}

#[cfg(test)]
mod tests;
