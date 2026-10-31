//! Target-only shared admission; offline and live differ only in inventory samples.
use super::{native::RankAdmission, ownership::ExecutionOwnership, Config};
use crate::{
    placement::*, plan::checkpoint::Checkpoint, serving_capacity::step5::Step5CacheLayout,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarkRule {
    pub lanes: usize,
    pub entries: usize,
    pub budget_bytes: u64,
    pub host: bool,
}
impl MarkRule {
    pub fn slots(self, bytes: usize) -> usize {
        cuteafd_core::prefix::mark_slots_with_host(
            self.lanes as u64,
            self.entries as u64,
            bytes as u64,
            self.budget_bytes,
            self.host,
        ) as usize
    }
}

/// Checked cache-only reservations. Weights, code, workspace and transport must
/// be subtracted before supplying the available bytes to `CacheRank::new`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reservation {
    pub active: usize,
    pub marks: usize,
    pub pages: usize,
    pub total: usize,
}

impl Reservation {
    pub fn new(
        layout: &Step5CacheLayout,
        rank: usize,
        sequences: u64,
        slots: u64,
        pages: u64,
        available: usize,
    ) -> Result<Self> {
        ensure!(sequences > 0 && pages > 0, "empty Step-5 cache capacity");
        let state = layout.ranks.get(rank).context("Step-5 cache rank")?;
        let geometry = layout
            .geometry
            .ranks
            .get(rank)
            .context("Step-5 cache geometry rank")?;
        let declared = state
            .regions
            .iter()
            .try_fold(0u64, |sum, region| {
                sum.checked_add(region.page_bytes.checked_mul(2)?)
            })
            .context("Step-5 CSA page geometry overflow")?;
        ensure!(
            geometry.persistent_unit_bytes == declared,
            "Step-5 inconsistent CSA page geometry"
        );
        let bytes = |unit: u64, count: u64| -> Result<usize> {
            Ok(usize::try_from(
                unit.checked_mul(count)
                    .context("Step-5 cache allocation overflow")?,
            )?)
        };
        let active = bytes(state.bytes, sequences)?;
        let marks = bytes(state.bytes, slots)?;
        let pages = bytes(geometry.persistent_unit_bytes, pages)?;
        let total = active
            .checked_add(marks)
            .and_then(|n| n.checked_add(pages))
            .context("Step-5 cache total overflow")?;
        ensure!(
            total <= available,
            "Step-5 cache needs {total} bytes, admitted only {available}"
        );
        Ok(Self {
            active,
            marks,
            pages,
            total,
        })
    }
}

pub fn input_bytes(rows: usize) -> Result<u64> {
    super::native::validate_span_rows(rows)?;
    Ok((rows * (4 + 4096 * 6)) as u64)
}

fn retained_wave_rows(rows: usize, sequences: usize) -> Result<usize> {
    ensure!(
        rows > 0 && sequences > 0,
        "Step-5 staging requires rows and sequences"
    );
    rows.checked_add(
        sequences
            .min(rows)
            .checked_mul(7)
            .context("Step-5 retained pending rows overflow")?,
    )
    .context("Step-5 retained wave rows overflow")
}
pub fn staging_bytes(layout: &Step5CacheLayout, rank: usize, sequences: usize) -> Result<usize> {
    let compact = layout.target_pack_planes(rank)?.into_iter().try_fold(0usize,
        |sum, bytes| sum.checked_add(bytes)).context("Step-5 compact staging overflow")?;
    let layers = layout
        .ranks
        .get(rank)
        .context("Step-5 staging rank")?
        .regions
        .len();
    let step = usize::try_from(layout.step_rows)?;
    let rows = retained_wave_rows(step, sequences)?;
    let regions = (step + 7) / 8;
    let metadata = regions
        .checked_mul(4)
        .context("Step-5 region metadata overflow")?
        .max(256);
    let scratch = regions
        .checked_mul(256 * 4)
        .context("Step-5 region scratch overflow")?
        .max(256);
    let table = usize::try_from(layout.context.div_ceil(64))?
        .checked_mul(4)
        .context("Step-5 attention table overflow")?
        .max(256);
    rows.checked_mul(256 * 2 * 2)
        .and_then(|n| n.checked_mul(layers))
        .and_then(|n| {
            n.checked_add(if layers == 0 {
                0
            } else {
                metadata.checked_mul(2)?
            })
        })
        .and_then(|n| n.checked_add(if layers == 0 { 0 } else { scratch }))
        .and_then(|n| n.checked_add(step.checked_mul(8)?.max(256)))
        .and_then(|n| n.checked_add(step.checked_mul(4)?.max(256)))
        .and_then(|n| n.checked_add(table.checked_mul(sequences.checked_add(1)?)?))
        .and_then(|n| n.checked_add(compact))
        .context("Step-5 staging bytes overflow")
}

const FLOOR: u64 = 262_144;
const EXTENT: u64 = 1_048_576;
const WHOLE0: LayerMode = LayerMode::Whole {
    gpu: 0,
    ffn: FfnMode::Owner,
};
const WHOLE1: LayerMode = LayerMode::Whole {
    gpu: 1,
    ffn: FfnMode::Owner,
};
const EXECUTOR: ExecutorModes = crate::placement::families::STEP5;

/// All demands are future to the pre-weight inventory. Startup phases are
/// absolute additional peaks above that same sample, not steady-state deltas.
pub struct RuntimeInputs {
    pub kv_format: crate::families::qwen4::Qwen4KvCache,
    pub inventory: Inventory,
    /// Span rows: every per-call plane's rows (`waves` units).
    pub rows: usize,
    /// Prefill slots on the one Spark transport (PLAN "Prefill
    /// contract"): hop lanes, return slots and Spark owner units.
    pub waves: usize,
    /// The admitted unit (#762); 0 takes `rows / waves` (a static span).
    pub unit: usize,
    /// All-row prefill scoring was admitted at launch; otherwise the head holds
    /// only the decode/verify width (prefill needs its last row).
    pub full_prefill_logits: bool,
    pub sequences: usize,
    pub context: u64,
    pub requested_pool: Option<u64>,
    pub marks: MarkRule,
    pub peer_diagnostics: bool,
    pub pending_code: Vec<u64>,
    pub startup: Vec<Demand>,
    /// RTX expert onboarding (every routed format; pool-first `Auto` by
    /// default); `None` only when onboarding is explicitly off.
    pub onboard: Option<OnboardInputs>,
}

/// The Step TP2 exchange a routed format runs: EXL3 halves take FP8 K32 wire
/// rows and exchange F32 partials (one rounding, as a Spark reply); fp8moe
/// halves take BF16 rows and exchange BF16 partials (as GLM Flash FP8 ships:
/// each half rounds to BF16 before the rank-order sum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tp2Exchange { Exl3, Fp8 }

impl Tp2Exchange {
    pub fn of(catalog: &crate::OfficialV41Catalog) -> Self {
        if catalog.exl3().is_some() { Self::Exl3 } else { Self::Fp8 }
    }
    /// Bytes of one exchanged partial element.
    pub fn partial_bytes(self) -> usize { match self { Self::Exl3 => 4, Self::Fp8 => 2 } }
    /// Input bytes per row a whole layer's owner broadcasts.
    pub fn input_row_bytes(self, hidden: usize) -> usize {
        match self { Self::Exl3 => hidden + hidden / 32, Self::Fp8 => hidden * 2 }
    }
}

/// One rank's Step TP2 onboarding exchange planes beside the package arena:
/// per prefill lane a routed partial, the peer's partial as received (one
/// plane for head-split layers, one for a whole layer's owner, so the two
/// exchanges never share storage) and the route broadcast scratch (packed
/// top-8 routes plus the input rows). The daemon allocates exactly these.
pub fn tp2_exchange_bytes(unit_rows: usize, lanes: usize, exchange: Tp2Exchange) -> Result<u64> {
    const H: usize = 4096;
    ensure!((1..=4096).contains(&unit_rows) && (1..=4).contains(&lanes), "Step-5 TP2 exchange geometry");
    let plane = |bytes: usize| bytes.max(256) as u64;
    let section = unit_rows * 8 * 4;
    let routes = (section.next_multiple_of(16) + section).next_multiple_of(16);
    let per_lane = 3 * plane(unit_rows * H * exchange.partial_bytes())
        + plane(routes + unit_rows * exchange.input_row_bytes(H));
    (lanes as u64).checked_mul(per_lane).context("Step-5 TP2 exchange overflow")
}

/// Why NVFP4 Step plans onboard nothing (and refuse an explicit layer count).
pub const NVFP4_ONBOARDING_DEPRECATED: &str = "Step-5 NVFP4 is deprecated (its cards retire this release): \
    RTX expert onboarding is not built for it; its routed experts stay on the Sparks";
/// FP8 Step is qualified only on two RTX (2 RTX + 6 Sparks); there its
/// onboarded layers split TP2. No whole-layer fp8moe executor exists.
pub const FP8_ONBOARDING_TWO_RTX: &str = "Step-5 FP8 onboards TP2 on two RTX only (its qualified card): \
    its routed experts stay on the Sparks";

/// The named reason a routed format does not onboard here, or None.
pub fn onboarding_unavailable(catalog: &crate::OfficialV41Catalog, gpus: usize) -> Option<&'static str> {
    use crate::formats::fp8_experts::ExpertFormat;
    match catalog.fp8().map(|fp8| fp8.format()) {
        Some(ExpertFormat::Nvfp4) => Some(NVFP4_ONBOARDING_DEPRECATED),
        Some(_) if gpus != 2 => Some(FP8_ONBOARDING_TWO_RTX),
        _ => None,
    }
}

/// Header-exact costs of the routed layers in order (whole, and TP2 halves on
/// two GPUs) and the installed package arenas; the shared solver chooses how
/// many layers the RTX keeps (pool first).
#[derive(Debug, Clone)]
pub struct OnboardInputs {
    pub costs: Vec<ExpertCost>,
    /// One GPU: the whole-layer (TP1) arena.
    pub workspace: u64,
    /// Two GPUs: each rank's TP2 arena.
    pub tp2_workspace: [u64; 2],
    pub mode: Onboard,
}

impl OnboardInputs {
    /// The shared RTX onboarding inputs for any routed format: whole layers on
    /// one GPU, TP2 halves over two, at the admitted unit rows.
    pub fn new(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, unit_rows: usize,
        gpus: usize, lanes: usize, mode: Onboard) -> Result<Option<Self>> {
        use crate::placement::rtx_onboard::{expert_costs, expert_workspace};
        ensure!(matches!(gpus, 1 | 2), "Step-5 RTX onboarding needs one or two GPUs");
        if mode == Onboard::Layers(0) { return Ok(None); }
        if let Some(reason) = onboarding_unavailable(catalog, gpus) {
            ensure!(mode == Onboard::Auto, "{reason}");
            return Ok(None);
        }
        let rows = unit_rows as u64;
        let (workspace, tp2_workspace) = if gpus == 2 {
            let half = expert_workspace(catalog, manifest, rows, 2, "step5")?
                .checked_add(tp2_exchange_bytes(unit_rows, lanes, Tp2Exchange::of(catalog))?)
                .context("Step-5 TP2 workspace overflow")?;
            (0, [half; 2])
        } else { (expert_workspace(catalog, manifest, rows, 1, "step5")?, [0; 2]) };
        Ok(Some(Self { costs: expert_costs(catalog, gpus == 2)?, workspace, tp2_workspace, mode }))
    }
}

pub struct Admission {
    pub placement: Placement,
    pub ownership: ExecutionOwnership,
    pub cache: Step5CacheLayout,
    pub marks: usize,
    pub reservations: Vec<Reservation>,
    /// Exact fixed demands; residual receive slots are charged by solve itself.
    pub fixed: Vec<Demand>,
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values
        .into_iter()
        .try_fold(0u64, |n, v| n.checked_add(v))
        .context("Step-5 admission sum overflow")
}
fn scaled(bytes: u64, count: usize) -> Result<u64> {
    bytes
        .checked_mul(u64::try_from(count)?)
        .context("Step-5 admission product overflow")
}

impl RuntimeInputs {
    pub fn logit_rows(&self) -> usize {
        super::native::head_rows(self.rows, self.full_prefill_logits)
    }

    /// Rows one hook runs: attention, FFN scratch, peer partials and hops.
    pub fn unit_rows(&self) -> usize { if self.unit > 0 { self.unit } else { self.rows / self.waves.max(1) } }

    pub fn validate(&self, cfg: &Config) -> Result<()> {
        let ranks = self.inventory.gpus.len();
        ensure!(
            matches!(ranks, 1 | 2) && (ranks == 1 || self.inventory.peer_access),
            "Step-5 requires one GPU or two genuine peer-accessible GPUs"
        );
        ensure!(
            self.inventory.spark_ranks > 0 && self.inventory.spark_ranks <= 8,
            "Step-5 target serving requires native Spark expert ranks"
        );
        super::native::validate_span_rows(self.rows)?;
        ensure!(
            (1..=4096).contains(&self.unit_rows())
                && (1..=4).contains(&self.waves)
                && self.rows % self.waves == 0
                && (self.waves == 1 || self.unit_rows() % 64 == 0)
                && (1..=64).contains(&self.sequences)
                && self.sequences <= self.unit_rows(),
            "Step-5 span rows, prefill waves or sequence slots"
        );
        use crate::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        prefill_shape(self.inventory.spark_ranks, crate::plan::Family::prefill_limits(&super::STEP5), PrefillOverrides {
            unit_rows: Some(self.unit_rows() as u64), waves: Some(self.waves as u64),
            span_rows: Some(self.rows as u64), reserve_rows: None,
        })?;
        ensure!(
            cfg.max_context == EXTENT && (FLOOR..=EXTENT).contains(&self.context),
            "Step-5 requires 1M compiled extent and at least 256K request context"
        );
        ensure!(
            self.requested_pool
                .filter(|&n| n > 0)
                .is_none_or(|n| n >= FLOOR && n % 64 == 0),
            "Step-5 explicit pool must be page-aligned and at least 256K tokens"
        );
        ensure!(
            self.pending_code.len() == ranks && self.marks.lanes == self.sequences,
            "Step-5 code or prefix sequence inventory differs"
        );
        ensure!(
            self.startup.iter().all(|d| usize::from(d.gpu) < ranks),
            "Step-5 startup phase names an absent GPU"
        );
        Ok(())
    }

    fn hops(&self) -> HopSpec {
        HopSpec {
            storage: HopStorage::ReceiveSlots,
            row_bytes: 4096 * 4,
            rows: self.unit_rows() as u64,
            lanes: self.waves as u64,
            entry_gpu: 0,
            head_gpu: 0,
            terminal: TerminalHome::AttentionOwner,
        }
    }

    /// Diagnostic alternating ownership; production admission scores complete ranges.
    #[cfg(test)]
    pub fn ownership(&self, cfg: &Config) -> Result<ExecutionOwnership> {
        let owners = attention::alternating_owners(
            cfg.full[..cfg.layers].iter().map(|&full| (full, None)),
            0,
        )?;
        self.ownership_for(cfg, &owners)
    }

    fn ownership_for(&self, cfg: &Config, owners: &[u8]) -> Result<ExecutionOwnership> {
        let ranks = self.inventory.gpus.len();
        ensure!(owners.len() == cfg.layers, "Step-5 CSA owner extent");
        let modes = (0..cfg.layers)
            .map(|layer| {
                if ranks == 1 {
                    WHOLE0
                } else if cfg.full[layer] {
                    if owners[layer] == 0 {
                        WHOLE0
                    } else {
                        WHOLE1
                    }
                } else {
                    LayerMode::HeadSplit
                }
            })
            .collect::<Vec<_>>();
        let spec = self.hops();
        let placement = Placement {
            pool_tokens: 0,
            pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Heads,
            attention_by_kind: vec![],
            peer_row_bytes: 0,
            onboard_layers: 0,
            max_pool_tokens_no_onboard: 0,
            layers: modes
                .iter()
                .enumerate()
                .map(|(layer, &mode)| LayerAssignment {
                    mode,
                    experts: if cfg.moe[layer] {
                        ExpertHome::Spark
                    } else {
                        ExpertHome::Dense
                    },
                })
                .collect(),
            movables: vec![],
            expert_ranges: vec![],
            residual: residual_homes(&modes, &spec)?,
            hops: plan_hops(&modes, &spec)?,
            tp2: None,
            items: vec![],
        };
        ExecutionOwnership::new(cfg, ranks, LayerExecutionPlan::new(&placement, spec)?)
    }
}

pub fn admit(checkpoint: &Checkpoint, inputs: &RuntimeInputs) -> Result<Admission> {
    let cfg = Config::parse(checkpoint)?;
    admit_with(&cfg, inputs, |ownership| {
        (0..ownership.ranks())
            .map(|rank| super::native::rank_admission(checkpoint, ownership, rank, inputs.unit_rows(),
                inputs.rows, inputs.logit_rows()))
            .collect()
    })
}

/// Offline forecast uses the same allocation recipe as opt-in native MTP.
/// Numerical and live-state qualification are separate from capacity admission.
pub fn forecast_mtp(checkpoint: &Checkpoint, inputs: &RuntimeInputs, depth: usize) -> Result<Admission> {
    admit_mtp(checkpoint, inputs, depth)
}

/// Includes native stage weights, private workspace, independent histories and
/// the actual host-aware staging mark count before any live allocation.
pub fn admit_mtp(checkpoint: &Checkpoint, inputs: &RuntimeInputs, depth: usize) -> Result<Admission> {
    if depth == 0 { return admit(checkpoint, inputs); }
    let cfg = Config::parse(checkpoint)?;
    ensure!((1..=cfg.mtp).contains(&depth) && inputs.unit_rows() > depth,
        "Step-5 predictor depth must fit checkpoint and row capacity");
    // Native K4 predictor owners keep their own projection scratch beside their weights.
    let weights = super::coordinator::RankWeightLayout::for_mtp(checkpoint, depth)?.resident_bytes
        .checked_add(super::native::mtp_native_scratch_bytes(checkpoint, depth, inputs.rows)?)
        .context("Step-5 predictor native forecast overflow")?;
    let workspace = super::native::mtp_forecast_workspace(inputs.unit_rows(), depth, inputs.kv_format)?;
    admit_depth_with(&cfg, inputs, depth, |ownership| {
        (0..ownership.ranks()).map(|rank| {
            let mut native = super::native::rank_admission(checkpoint, ownership, rank, inputs.unit_rows(),
                inputs.rows, inputs.logit_rows())?;
            if rank == usize::from(ownership.execution().spec().head_gpu) {
                native.immutable = native.immutable.checked_add(weights)
                    .context("Step-5 predictor weight forecast overflow")?;
                native.workspace = native.workspace.max(workspace);
            }
            Ok(native)
        }).collect()
    })
}

pub fn admit_with(
    cfg: &Config,
    inputs: &RuntimeInputs,
    native: impl FnMut(&ExecutionOwnership) -> Result<Vec<RankAdmission>>,
) -> Result<Admission> {
    admit_depth_with(cfg, inputs, 0, native)
}

/// The caller includes every predictor's private immutable/workspace owner.
fn admit_depth_with(
    cfg: &Config,
    inputs: &RuntimeInputs,
    depth: usize,
    mut native: impl FnMut(&ExecutionOwnership) -> Result<Vec<RankAdmission>>,
) -> Result<Admission> {
    ensure!(depth <= cfg.mtp, "Step-5 forecast predictor depth exceeds checkpoint");
    inputs.validate(cfg)?;
    let ranks = inputs.inventory.gpus.len();
    let groups = cfg.full[..cfg.layers].iter().map(|&full| (ranks == 2 && full, None))
        .collect::<Vec<_>>();
    let policy = if ranks == 2 { attention::LayerOwnership::Ranges }
        else { attention::LayerOwnership::Alternating };
    attention::solve_owners(&groups, policy, 0, Onboard::Auto, |owners| -> Result<_> {
        let ownership = inputs.ownership_for(cfg, owners)?;
        let native = native(&ownership)?;
        let admitted = admit_frozen(cfg, inputs, depth, &ownership, &native)?;
        Ok((admitted.placement.clone(), admitted))
    }).map(|(_, admitted)| admitted)
}

fn admit_frozen(cfg: &Config, inputs: &RuntimeInputs, depth: usize,
    ownership: &ExecutionOwnership, native: &[RankAdmission]) -> Result<Admission> {
    let ranks = ownership.ranks();
    ensure!(
        native.len() == ranks,
        "Step-5 native admission rank inventory"
    );
    let mut pool = PoolPolicy::resolve(
        &inputs
            .inventory
            .gpus
            .iter()
            .map(|g| g.capacity_bytes)
            .collect::<Vec<_>>(),
        inputs.context,
        inputs.requested_pool,
        64,
        false,
    );
    // Request context may descend with an automatic aggregate pool, but never
    // below the agentic floor. It is not the compiled kernel extent.
    pool.floor = pool.requested.unwrap_or(FLOOR);
    let initial = pool.requested.unwrap_or(pool.target);
    settle_pool(
        (initial, inputs.context.min(initial)),
        |tokens, &context| {
            let cache =
                Step5CacheLayout::for_execution(cfg, &ownership, context, depth, inputs.unit_rows() as u64,
                    inputs.rows as u64, inputs.kv_format)?;
            let mark_bytes = usize::try_from(checked_sum(cache.ranks.iter().map(|r| r.bytes))?)?;
            let marks = inputs.marks.slots(mark_bytes);
            let mut fixed = Vec::new();
            let mut reservations = Vec::new();
            for rank in 0..ranks {
                let reservation = Reservation::new(
                    &cache,
                    rank,
                    inputs.sequences as u64,
                    marks as u64,
                    1,
                    usize::MAX,
                )?;
                let mut add = |category, group, bytes| {
                    fixed.push(Demand::new(
                        rank as u8,
                        category,
                        group,
                        bytes,
                        Basis::Exact,
                    ))
                };
                add(
                    Category::Weights,
                    "Step-5 native target and head",
                    native[rank].immutable,
                );
                add(
                    Category::Workspace,
                    "Step-5 native workspace",
                    native[rank].workspace,
                );
                add(
                    Category::Kv,
                    "Step-5 active sequence state",
                    reservation.active as u64,
                );
                add(
                    Category::Prefix,
                    "Step-5 exact retained marks",
                    reservation.marks as u64,
                );
                add(
                    Category::Workspace,
                    "Step-5 private transaction staging",
                    staging_bytes(&cache, rank, inputs.sequences)? as u64,
                );
                add(
                    Category::Runtime,
                    "Step-5 pending code",
                    inputs.pending_code[rank],
                );
                if rank == 0 {
                    add(
                        Category::Embedding,
                        "Step-5 GPU embedding",
                        scaled(4096 * 2, cfg.vocab)?,
                    );
                    add(
                        Category::Workspace,
                        "Step-5 input",
                        input_bytes(inputs.rows)?,
                    );
                    let capacity = inputs.rows.min(64);
                    add(
                        Category::Workspace,
                        "Step-5 selector and device sampler",
                        checked_sum([
                            scaled(12, capacity)?,
                            cuteafd_core::target_sampler_device_bytes(capacity, cfg.vocab) as u64,
                        ])?,
                    );
                }
                if rank == usize::from(ownership.execution().spec().entry_gpu) {
                    add(
                        Category::Transport,
                        "Step-5 Spark intake planes",
                        // One reserve range per rank every admitted depth is cut from (#762).
                        crate::serving_capacity::prefill_return_bytes(1, inputs.rows as u64,
                            inputs.inventory.spark_ranks as u64, 4096)
                            .max(256 * inputs.inventory.spark_ranks as u64),
                    );
                    add(Category::Workspace, "Step-5 routed reply", (inputs.rows as u64 * 4096 * 2).max(256));
                }
                if ranks == 2 {
                    let progress = if inputs.peer_diagnostics {
                        (9 * 48).max(256) as u64
                    } else {
                        0
                    };
                    // Eight peer receive slots, peer control, abort word and one
                    // private hop control plane. solve adds hop receives separately.
                    add(
                        Category::Transport,
                        "Step-5 private peer and hop control",
                        checked_sum([
                                8 * (inputs.unit_rows() as u64 * 4096 * 4).max(256),
                                256,
                                256,
                                256,
                                progress,
                            ])?,
                    );
                }
                reservations.push(reservation);
            }
            let layers = ownership
                .execution()
                .modes()
                .iter()
                .enumerate()
                .map(|(layer, &mode)| LayerDemand {
                    kind: if cfg.full[layer] {
                        AttentionClass::Csa
                    } else {
                        AttentionClass::Swa
                    },
                    weights: ModeBytes::default(),
                    kv_unit: KvDemand::default(),
                    fixed_bytes: ModeBytes::default(),
                    extra: vec![],
                    context_indexer: false,
                    colocate: None,
                    experts: cfg.moe[layer].then(|| inputs.onboard.as_ref()
                        .and_then(|onboard| onboard.costs.get(layer - 3).copied())
                        .unwrap_or(ExpertCost {
                            whole: Bytes2::default(),
                            half: [Bytes2::default(); 2],
                            tp2: false,
                            spark_ok: true,
                        })),
                    modes: vec![mode],
                })
                .collect();
            let mut trial_pool = pool;
            if trial_pool.requested.is_none() {
                trial_pool.target = tokens;
            }
            let request = PlacementRequest {
                inventory: inputs.inventory.clone(),
                attention_placement: None,
                context_buffers: ContextBuffers::default(),
                layers_first_gpu: 0,
                pool: trial_pool,
                layers,
                pool_overhead: cache
                    .geometry
                    .ranks
                    .iter()
                    .map(|r| r.persistent_unit_bytes)
                    .collect(),
                pool_bounded: vec![],
                fixed: fixed.clone(),
                startup: inputs.startup.clone(),
                movables: vec![],
                expert_workspace: inputs.onboard.as_ref().map_or(0, |o| o.workspace),
                tp2_workspace: inputs.onboard.as_ref().map_or([0; 2], |o| o.tp2_workspace),
                onboard: inputs.onboard.as_ref().map_or(Onboard::Auto, |o| o.mode),
                // One GPU holds whole layers; two split every onboarded layer
                // TP2 (the solver's TP2 prefix takes its cap from GPU 0's).
                expert_gpus: usize::from(inputs.onboard.is_some()),
                minimum_expert_layers: 0,
                policy: LayerPolicy {
                    attention: if ranks == 2 {
                        vec![
                            (AttentionClass::Csa, AttentionPlacement::Layers),
                            (AttentionClass::Swa, AttentionPlacement::Heads),
                        ]
                    } else {
                        vec![]
                    },
                    ..Default::default()
                },
                hops: inputs.hops(),
                executor: EXECUTOR,
            };
            let retention_guidance = |error: anyhow::Error| {
                if !inputs.marks.host && marks > 0 {
                    anyhow::anyhow!("{error}; Step-5 device-only prefix retention with HOST_CACHE_BYTES=0 does not fit at request context {context}; enable --host-cache-bytes auto (HOST_CACHE_BYTES=auto), or lower --max-context/--context-tokens (MAX_CONTEXT_TOKENS), for example to 262144")
                } else {
                    error
                }
            };
            let placement = solve(&request).map_err(|error| {
                if matches!(&error, PlacementError::Mandatory { .. }
                    | PlacementError::PoolDoesNotFit { .. } | PlacementError::BelowFloor { .. }) {
                    retention_guidance(error.into())
                } else {
                    error.into()
                }
            })?;
            if placement.pool_tokens < FLOOR {
                return Err(retention_guidance(anyhow::anyhow!(
                    "Step-5 aggregate pool is below the 256K agentic floor"
                )));
            }
            EXECUTOR.check(&placement)?;
            ensure!(
                placement.layers.iter().map(|l| l.mode).eq(ownership
                    .execution()
                    .modes()
                    .iter()
                    .copied()),
                "Step-5 solver ownership differs from native allocation owners"
            );
            let checked_ownership = ExecutionOwnership::new(
                cfg,
                ranks,
                LayerExecutionPlan::new(&placement, inputs.hops())?,
            )?;
            let onboarded = placement.layers.iter().enumerate()
                .filter(|(_, l)| matches!(l.experts, ExpertHome::RtxWhole { .. } | ExpertHome::RtxTp2))
                .map(|(layer, l)| {
                    ensure!(l.experts == if ranks == 2 { ExpertHome::RtxTp2 }
                        else { ExpertHome::RtxWhole { gpu: inputs.hops().entry_gpu } },
                        "Step-5 onboarded layer {layer}: whole on one GPU, TP2 over two");
                    Ok(layer)
                }).collect::<Result<Vec<_>>>()?;
            let checked_ownership = checked_ownership.with_onboarded(onboarded)?;
            let next_context = context.min(placement.pool_tokens);
            let pages = placement.pool_tokens / 64;
            ensure!(
                pages <= u32::MAX as u64,
                "Step-5 pool page identifiers overflow"
            );
            for rank in 0..ranks {
                reservations[rank] = Reservation::new(
                    &cache,
                    rank,
                    inputs.sequences as u64,
                    marks as u64,
                    pages,
                    usize::MAX,
                )?;
            }
            Ok((
                placement.pool_tokens,
                next_context,
                Admission {
                    placement,
                    ownership: checked_ownership,
                    cache,
                    marks,
                    reservations,
                    fixed,
                },
            ))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            norm_eps: 1e-5,
            hidden: 4096,
            vocab: 128896,
            layers: 92,
            mtp: 3,
            dense: 13824,
            experts: 352,
            intermediate: 1536,
            shared: 1536,
            topk: 8,
            heads: 64,
            groups: 4,
            dim: 192,
            window: 512,
            max_context: EXTENT,
            full: (0..95).map(|l| l < 92 && l % 4 == 3).collect(),
            moe: (0..95).map(|l| (3..91).contains(&l)).collect(),
            theta: vec![],
            rotary: vec![],
            rope_scaling: None,
            scale_full: false,
            scale_sliding: false,
            activation: vec![],
            shared_activation: vec![],
        }
    }
    fn inputs(ranks: usize) -> RuntimeInputs {
        RuntimeInputs {
                kv_format: crate::families::qwen4::Qwen4KvCache::Bf16,
            full_prefill_logits: false, waves: 1, unit: 0,
            inventory: Inventory {
                gpus: vec![
                    GpuBudget {
                        capacity_bytes: 96 << 30,
                        headroom_bytes: 1 << 30,
                        baseline: Baseline::Measured {
                            free_bytes: 96 << 30
                        }
                    };
                    ranks
                ],
                spark_ranks: 4,
                peer_access: ranks == 2,
            },
            rows: 64,
            sequences: 16,
            context: FLOOR,
            requested_pool: Some(FLOOR),
            marks: MarkRule {
                lanes: 16,
                entries: 16,
                budget_bytes: 0,
                host: false,
            },
            peer_diagnostics: false,
            pending_code: vec![1 << 20; ranks],
            startup: vec![], onboard: None,
        }
    }
    fn run(inputs: &RuntimeInputs) -> Result<Admission> {
        admit_with(&config(), inputs, |o| {
            Ok(vec![
                RankAdmission {
                    immutable: 1 << 30,
                    workspace: 1 << 20
                };
                o.ranks()
            ])
        })
    }

    #[test]
    fn admission_binds_real_one_and_two_rank_owners_and_exact_pages() -> Result<()> {
        for ranks in [1, 2] {
            let inputs = inputs(ranks);
            let admission = run(&inputs)?;
            assert_eq!(admission.ownership.ranks(), ranks);
            assert_eq!(admission.ownership.execution().spec().lanes, 1);
            assert_eq!(admission.cache.context, FLOOR);
            assert_eq!(admission.placement.pool_tokens, FLOOR);
            assert!(admission
                .cache
                .ranks
                .iter()
                .all(|r| r.mtp_lengths.is_empty()));
            for rank in 0..ranks {
                assert_eq!(
                    admission.reservations[rank].pages as u64,
                    admission.cache.geometry.ranks[rank].persistent_unit_bytes * (FLOOR / 64)
                );
                let fixed = checked_sum(
                    admission
                        .fixed
                        .iter()
                        .filter(|d| d.gpu == rank as u8)
                        .map(|d| d.bytes),
                )?;
                let hops = hop_buffer_bytes(
                    admission.ownership.execution().hops(),
                    &inputs.hops(),
                    ranks,
                )
                .context("Step-5 test hop size overflow")?;
                let charged = checked_sum(admission.placement.items[rank].iter().map(|i| i.bytes))?;
                assert_eq!(
                    charged,
                    fixed + hops[rank] + admission.reservations[rank].pages as u64
                );
            }
            assert_eq!(admission.ownership.execution().mode(3)?, WHOLE0);
            if ranks == 2 {
                assert_eq!(
                    admission.ownership.execution().mode(0)?,
                    LayerMode::HeadSplit
                );
                let owners = (0..92).filter(|&layer| config().full[layer])
                    .map(|layer| admission.ownership.execution().mode(layer))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                assert!(owners.contains(&WHOLE0) && owners.contains(&WHOLE1));
                assert!(owners.windows(2).all(|pair| pair != [WHOLE1, WHOLE0]));
            }
        }
        Ok(())
    }

    #[test]
    fn owner_search_scores_predictor_complete_inventory_and_keeps_winning_layout() -> Result<()> {
        let cfg = config();
        for format in [crate::families::qwen4::Qwen4KvCache::Bf16,
            crate::families::qwen4::Qwen4KvCache::Fp8] {
            let mut inputs = inputs(2);
            inputs.kv_format = format;
            inputs.requested_pool = None;
            inputs.marks.host = true;
            let native = |ownership: &ExecutionOwnership, predictor: bool| -> Result<Vec<RankAdmission>> {
                Ok((0..2).map(|rank| RankAdmission {
                    // Cross a whole-CSA boundary in both cache formats, not just consume its slack.
                    immutable: (1 << 30) + if rank == 0 && predictor { 32 << 30 } else { 0 },
                    workspace: if rank == 0 && predictor { 512 << 20 } else { 1 << 20 },
                }).take(ownership.ranks()).collect())
            };
            let base = admit_depth_with(&cfg, &inputs, 0, |o| native(o, false))?;
            let without_weights = admit_depth_with(&cfg, &inputs, 3, |o| native(o, false))?;
            let candidate = admit_depth_with(&cfg, &inputs, 3, |o| native(o, true))?;
            let old_pressure = admit_depth_with(&cfg, &inputs, 3, |o| {
                let mut inventory = native(o, true)?;
                inventory[0].immutable -= 20 << 30;
                Ok(inventory)
            })?;
            let count = |a: &Admission| a.placement.layers.iter().enumerate().filter(|(layer, item)|
                cfg.full[*layer] && item.mode == WHOLE0).count();
            assert!(count(&candidate) < count(&base), "{format:?}: predictor pressure must move CSA off the head GPU: base={} candidate={}, scores {:?} -> {:?}",
                count(&base), count(&candidate), attention::owner_score(Onboard::Auto, &base.placement),
                attention::owner_score(Onboard::Auto, &candidate.placement));
            // At 12 GiB the neighboring 10/13 cut is still slightly worse:
            // BF16 38,284 vs 38,338 pages; FP8 75,409 vs 75,631 pages.
            assert_eq!(count(&old_pressure), 11);
            assert!(attention::owner_score(Onboard::Auto, &old_pressure.placement)
                < attention::owner_score(Onboard::Auto, &without_weights.placement));
            eprintln!("{format:?}: omitted weights cut={}, 12 GiB cut={}, physical scores {:?} -> {:?}",
                count(&without_weights), count(&old_pressure),
                attention::owner_score(Onboard::Auto, &without_weights.placement),
                attention::owner_score(Onboard::Auto, &old_pressure.placement));
            assert_eq!(count(&without_weights), 11);
            assert_eq!((count(&base), count(&candidate)), (11, 9),
                "{format:?}: complete predictor costs must select the expected CSA cut");
            let alternating = inputs.ownership(&cfg)?;
            let old = admit_frozen(&cfg, &inputs, 3, &alternating, &native(&alternating, true)?)?;
            eprintln!("{format:?}: CSA owners base={}/{}, MTP={}/{}, pool alternating={} solved={}, scores {:?} -> {:?}",
                count(&base), 23 - count(&base), count(&candidate), 23 - count(&candidate),
                old.placement.pool_tokens, candidate.placement.pool_tokens,
                attention::owner_score(Onboard::Auto, &old.placement),
                attention::owner_score(Onboard::Auto, &candidate.placement));
            assert!(attention::owner_score(Onboard::Auto, &candidate.placement)
                > attention::owner_score(Onboard::Auto, &old.placement));
            assert_eq!(candidate.marks, 2);
            assert_eq!(candidate.cache.ranks[0].mtp_lengths.len(), 3);
            assert!(candidate.cache.ranks[1].mtp_lengths.is_empty());
            let expected = Step5CacheLayout::for_execution(&cfg, &candidate.ownership,
                candidate.cache.context, 3, inputs.unit_rows() as u64, inputs.rows as u64, format)?;
            assert_eq!(candidate.cache, expected);
            for rank in 0..2 {
                let reservation = Reservation::new(&expected, rank, inputs.sequences as u64,
                    candidate.marks as u64, candidate.placement.pool_tokens / 64, usize::MAX)?;
                assert_eq!(candidate.reservations[rank].total, reservation.total);
            }
        }
        Ok(())
    }

    #[test]
    fn host_retention_reserves_only_shared_staging_marks() -> Result<()> {
        for ranks in [1, 2] {
            let mut inputs = inputs(ranks);
            inputs.context = EXTENT;
            inputs.requested_pool = None;
            inputs.marks.host = true;
            for entries in [0, 20, 200] {
                inputs.marks.entries = entries;
                for budget in [0, 2 << 30, 32 << 30] {
                    inputs.marks.budget_bytes = budget;
                    let admitted = run(&inputs)?;
                    let slots = if entries == 0 { 0 } else { 2 };
                    assert_eq!(admitted.marks, slots);
                    for rank in 0..ranks {
                        assert_eq!(admitted.reservations[rank].marks as u64,
                            admitted.cache.ranks[rank].bytes * slots as u64);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn device_only_retention_fit_refusal_names_host_and_context_remedies() -> Result<()> {
        let mut inputs = inputs(1);
        inputs.context = EXTENT;
        inputs.requested_pool = None;
        inputs.inventory.gpus[0].baseline = Baseline::Measured { free_bytes: 50 << 30 };
        let refusal = run(&inputs).err().expect("device marks exceed the available budget").to_string();
        for message in ["device-only prefix retention with HOST_CACHE_BYTES=0",
            "request context 1048576", "--host-cache-bytes auto", "MAX_CONTEXT_TOKENS", "262144"] {
            assert!(refusal.contains(message), "{refusal}");
        }
        inputs.marks.host = true;
        run(&inputs)?;
        inputs.marks.host = false;
        inputs.context = FLOOR;
        run(&inputs)?;
        Ok(())
    }

    #[test]
    fn unit_spans_keep_unit_sized_hooks_and_exact_input_charges() -> Result<()> {
        let cfg = config();
        for (unit, waves) in [(4096, 2), (2752, 3), (2048, 4)] {
            let mut input = inputs(2);
            input.rows = unit * waves;
            input.waves = waves;
            input.validate(&cfg)?;
            assert_eq!(input.unit_rows(), unit);
            assert_eq!(input_bytes(input.rows)?, (unit * waves * (4 + 4096 * 6)) as u64);
            assert_eq!(input.logit_rows(), 64);
            input.full_prefill_logits = true;
            assert_eq!(input.logit_rows(), input.rows);
            assert!(super::super::native::mtp_forecast_workspace(unit, 3, input.kv_format).is_ok());
            assert!(super::super::native::mtp_forecast_workspace(input.rows, 3, input.kv_format).is_err());
        }
        for (rows, waves) in [(8192, 1), (8257, 3), (16384, 4), (5000, 5)] {
            let mut input = inputs(2);
            input.rows = rows;
            input.waves = waves;
            assert!(input.validate(&cfg).is_err(), "{rows} rows / {waves} waves");
        }
        let limit = crate::serving_capacity::prefill_waves::PrefillLimits::default().max_span_rows as usize;
        assert!(input_bytes(limit).is_ok());
        for rows in [0, limit + 1, usize::MAX] { assert!(input_bytes(rows).is_err()); }
        Ok(())
    }

    #[test]
    fn admission_checks_capacity_before_any_native_allocation() -> Result<()> {
        let mut inputs = inputs(1);
        let admission = run(&inputs)?;
        let charged = checked_sum(admission.placement.items[0].iter().map(|i| i.bytes))?;
        inputs.inventory.gpus[0].baseline = Baseline::Measured {
            free_bytes: charged + (1 << 30),
        };
        run(&inputs)?;
        inputs.inventory.gpus[0].baseline = Baseline::Measured {
            free_bytes: charged + (1 << 30) - 1,
        };
        assert!(run(&inputs).is_err());
        assert!(scaled(u64::MAX, 2).is_err());
        assert!(checked_sum([u64::MAX, 1]).is_err());
        Ok(())
    }

    #[test]
    fn admission_refuses_fake_peer_and_tiny_context() {
        let mut inputs = inputs(2);
        inputs.inventory.peer_access = false;
        assert!(run(&inputs).is_err());
        inputs.inventory.peer_access = true;
        inputs.context = FLOOR - 1;
        assert!(run(&inputs).is_err());
    }

    #[test]
    fn admission_reserves_actual_marks_diagnostics_and_independent_startup_peaks() -> Result<()> {
        let mut inputs = inputs(2);
        let base = run(&inputs)?;
        inputs.marks.budget_bytes = 32 << 30;
        inputs.marks.entries = 24;
        inputs.peer_diagnostics = true;
        let retained = run(&inputs)?;
        let mark_bytes = checked_sum(retained.cache.ranks.iter().map(|r| r.bytes))?;
        assert_eq!(
            retained.marks,
            inputs.marks.slots(usize::try_from(mark_bytes)?)
        );
        assert!(retained.marks > base.marks);
        for rank in 0..2 {
            assert_eq!(
                retained.reservations[rank].marks as u64,
                retained.cache.ranks[rank].bytes * retained.marks as u64
            );
            let transport = |a: &Admission| {
                checked_sum(
                    a.fixed
                        .iter()
                        .filter(|d| {
                            d.gpu == rank as u8 && d.group == "Step-5 private peer and hop control"
                        })
                        .map(|d| d.bytes),
                )
            };
            assert_eq!(
                transport(&retained)? - transport(&base)?,
                9 * 48 as u64
            );
        }
        inputs.startup.push(Demand::new(
            0,
            Category::Workspace,
            "load phase",
            96 << 30,
            Basis::Exact,
        ));
        run(&inputs)?;
        inputs.startup[0].bytes += 1;
        assert!(run(&inputs).is_err());
        inputs.startup.clear();
        assert!(admit_with(&config(), &inputs, |_| Ok(vec![])).is_err());
        Ok(())
    }

    #[test]
    fn automatic_pool_descends_without_confusing_context_and_compiled_extent() -> Result<()> {
        let mut inputs = inputs(1);
        inputs.requested_pool = None;
        inputs.context = EXTENT;
        inputs.marks.entries = 0;
        inputs.sequences = 1;
        inputs.marks.lanes = 1;
        inputs.inventory.gpus[0].baseline = Baseline::Measured {
            free_bytes: 40 << 30,
        };
        let admission = run(&inputs)?;
        assert!((FLOOR..EXTENT).contains(&admission.placement.pool_tokens));
        assert_eq!(admission.cache.context, admission.placement.pool_tokens);
        assert_eq!(config().max_context, EXTENT);
        let initial =
            Step5CacheLayout::for_execution(&config(), &admission.ownership, EXTENT, 0, 64, 64, crate::families::qwen4::Qwen4KvCache::Bf16)?;
        assert!(initial.ranks[0].bytes > admission.cache.ranks[0].bytes);
        Ok(())
    }
}
