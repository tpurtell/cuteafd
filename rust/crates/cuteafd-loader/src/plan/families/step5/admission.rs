//! Target-only shared admission; offline and live differ only in inventory samples.
use super::{native::RankAdmission, ownership::ExecutionOwnership, Config};
use crate::{
    placement::{planner::{solve, Refusal}, *},
    plan::checkpoint::Checkpoint,
    serving_capacity::step5::Step5CacheLayout,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};

/// Cache maintenance and transaction lane survive loading. Delivery adds one
/// dense transform stream and the shared copy-stream set on each device.
/// Rotary streams retire before delivery; MTP uses the transaction lane, and
/// unused diagnostic streams remain lazy.
pub fn startup_peak_live_streams() -> u64 {
    2 + 1 + crate::push_load::COPY_STREAMS as u64
}

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
    Ok(staging_layout(layout, rank, sequences)?.bytes)
}

/// Transaction-local producer planes, including temporary compact K/V output.
/// These are copied into independently owned cache storage before reuse.
/// Order: metadata, optional compressor, raw region pairs, optional compact pair.
pub fn staging_layout(layout: &Step5CacheLayout, rank: usize, sequences: usize)
    -> Result<crate::serving_capacity::PackedWeightRegions> {
    let compact = layout.target_pack_planes(rank)?;
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
    let mut sizes = vec![step.checked_mul(8).context("Step-5 positions overflow")?.max(256),
        step.checked_mul(4).context("Step-5 requests overflow")?.max(256), table,
        table.checked_mul(sequences).context("Step-5 page tables overflow")?,
        step.checked_mul(4).context("Step-5 graph requests overflow")?.max(256),
        table.checked_mul(sequences).context("Step-5 graph ring tables overflow")?,
        table.checked_mul(sequences).context("Step-5 graph page tables overflow")?];
    if layers > 0 { sizes.extend([metadata, metadata, scratch]); }
    let raw = rows.checked_mul(256 * 2).context("Step-5 raw staging overflow")?;
    for _ in 0..layers { sizes.extend([raw, raw]); }
    if compact[0] > 0 { sizes.extend(compact); }
    crate::serving_capacity::PackedWeightRegions::new(sizes)
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
#[derive(Clone)]
pub struct RuntimeInputs {
    pub graph_layout: Option<crate::serving_capacity::step5_graphs::Step5GraphLayout>,
    pub decode_graphs: bool,
    pub host: HostInputs,
    pub bf16_target_head: crate::placement::planner::serving::BlasHead,
    pub resident_weight_limit: Option<ResidentWeights>,
    pub spark_reply_format: cuteafd_core::SparkReplyFormat,
    pub kv_format: crate::families::qwen4::Qwen4KvCache,
    pub inventory: Inventory,
    /// Span rows: every per-call plane's rows (`waves` units).
    pub rows: usize,
    /// Prefill slots on the one Spark transport (PLAN "Prefill
    /// contract"): hop lanes, return slots and Spark owner units.
    pub waves: usize,
    /// The admitted unit (#762); 0 takes `rows / waves` (a static span).
    pub unit: usize,
    pub sequences: usize,
    pub context: u64,
    pub serving_context: crate::placement::planner::Context,
    pub requested_pool: Option<u64>,
    pub marks: MarkRule,
    pub peer_diagnostics: bool,
    pub pending_code: Vec<u64>,
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

/// Host inventory is sampled once, before the single admission.
#[derive(Debug, Clone)]
pub struct HostInputs {
    pub memory: (u64, u64),
    pub cache_bytes: Option<u64>,
    pub headroom: u64,
    pub retained_sync_h2d_bytes: u64,
    pub attention_mount_load: crate::shared::dense_read::MountLoad,
    pub push: Option<crate::push_load::PushHostEnvelope>,
}
impl Default for HostInputs {
    fn default() -> Self {
        Self { memory: (128 << 30, 128 << 30), cache_bytes: Some(0), headroom: 8 << 30,
            retained_sync_h2d_bytes: 0, attention_mount_load: Default::default(), push: None }
    }
}

impl HostInputs {
    pub fn from_contract(contract: &super::coordinator::CoordinatorContract, cfg: &Config,
        gpus: usize, memory: (u64, u64), cache_bytes: Option<u64>, headroom: u64) -> Result<Self> {
        Ok(Self { memory, cache_bytes, headroom,
            retained_sync_h2d_bytes: contract.retained_sync_h2d_bytes()?,
            attention_mount_load: contract.attention_mount_load(gpus)?,
            push: Some(push_host_envelope(contract, cfg, gpus)?),
        })
    }
}

/// Header-derived upper bound for dense transforms and expert validation.
/// Estimated staging until a Step-5 census supplies a tighter per-rank bound.
pub fn push_host_envelope(contract: &super::coordinator::CoordinatorContract, cfg: &Config,
    gpus: usize) -> Result<crate::push_load::PushHostEnvelope> {
    let (target, draft) = contract.core.bytes(cfg.layers);
    let dense = checked_sum([contract.weights.resident_bytes, target, draft])?
        .checked_mul(2).and_then(|n| n.checked_add(64 << 20))
        .context("Step-5 dense push host bound overflow")?;
    let validation = (cfg.layers as u64).checked_mul(cfg.experts as u64)
        .and_then(|n| n.checked_mul(3 * 4)).context("Step-5 expert validation overflow")?;
    Ok(crate::push_load::PushHostEnvelope::new(vec![validation.max(4 << 20); gpus],
        vec![dense; gpus], 128, 1 << 20)?)
}

/// One rank's Step TP2 onboarding exchange planes beside the package arena:
/// per prefill lane a routed partial, the peer's partial as received (one
/// plane for head-split layers, one for a whole layer's owner, so the two
/// exchanges never share storage) and the route broadcast scratch (packed
/// top-8 routes plus the input rows). The daemon allocates exactly these.
pub fn tp2_exchange_bytes(unit_rows: usize, lanes: usize, exchange: Tp2Exchange) -> Result<u64> {
    checked_sum(tp2_exchange_requests(unit_rows, lanes, exchange)?)
}

pub fn tp2_exchange_requests(unit_rows: usize, lanes: usize, exchange: Tp2Exchange) -> Result<Vec<u64>> {
    const H: usize = 4096;
    ensure!((1..=4096).contains(&unit_rows) && (1..=4).contains(&lanes), "Step-5 TP2 exchange geometry");
    let section = unit_rows * 8 * 4;
    let routes = (section.next_multiple_of(16) + section).next_multiple_of(16);
    let scratch = routes + unit_rows * exchange.input_row_bytes(H);
    let mut requests = vec![(lanes * unit_rows * H * exchange.partial_bytes()).max(256) as u64; 3];
    requests.push((lanes * scratch).max(256) as u64);
    // Pair::new_with_wire retains two route/input slots per lane and controls.
    requests.extend(vec![scratch.max(256) as u64; 2 * lanes]);
    let storage = crate::serving_capacity::PeerStorageLayout::new(2 * lanes, scratch, false)?;
    requests.extend([storage.control_bytes as u64, 256]);
    Ok(requests)
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
    pub workspace_requests: Vec<Vec<u64>>,
    pub mode: Onboard,
}

impl OnboardInputs {
    /// The shared RTX onboarding inputs for any routed format: whole layers on
    /// one GPU, TP2 halves over two, at the admitted unit rows.
    pub fn new(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, unit_rows: usize,
        gpus: usize, lanes: usize, mode: Onboard) -> Result<Option<Self>> {
        use crate::placement::rtx_onboard::{expert_costs, expert_workspace_requests};
        ensure!(matches!(gpus, 1 | 2), "Step-5 RTX onboarding needs one or two GPUs");
        if mode == Onboard::Layers(0) { return Ok(None); }
        if let Some(reason) = onboarding_unavailable(catalog, gpus) {
            ensure!(mode == Onboard::Auto, "{reason}");
            return Ok(None);
        }
        let rows = unit_rows as u64;
        let mut requests = expert_workspace_requests(catalog, manifest, rows, gpus, "step5")?;
        if gpus == 2 { requests.extend(tp2_exchange_requests(unit_rows, lanes, Tp2Exchange::of(catalog))?); }
        let bytes = checked_sum(requests.iter().copied())?;
        let (workspace, tp2_workspace) = if gpus == 2 { (0, [bytes; 2]) } else { (bytes, [0; 2]) };
        Ok(Some(Self { costs: expert_costs(catalog, gpus == 2)?, workspace, tp2_workspace,
            workspace_requests: vec![requests; gpus], mode }))
    }
}

pub use crate::placement::families::step5::Admission;

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
        super::native::head_rows(self.rows)
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
            self.inventory.spark_ranks <= 8,
            "Step-5 Spark inventory exceeds eight ranks"
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
        // An explicit pool below the request context is refused, never clamped.
        PoolPolicy::resolve(
            &self.inventory.gpus.iter().map(|g| g.capacity_bytes).collect::<Vec<_>>(),
            self.context,
            self.requested_pool,
            64).with_context(self.serving_context)
        .validate_explicit()?;
        ensure!(
            self.pending_code.len() == ranks && self.marks.lanes == self.sequences,
            "Step-5 code or prefix sequence inventory differs"
        );
        Ok(())
    }

    pub fn hops(&self) -> HopSpec {
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
        ExecutionOwnership::for_modes(cfg, ranks, &modes, self.hops())
    }
}

pub fn admit(checkpoint: &Checkpoint, inputs: &RuntimeInputs) -> Result<Admission> {
    let cfg = Config::parse(checkpoint)?;
    let contract = super::coordinator::CoordinatorContract::from_checkpoint(checkpoint)?;
    let mut selected_inputs = inputs.clone();
    selected_inputs.host.attention_mount_load = contract.attention_mount_load(inputs.inventory.gpus.len())?;
    let inputs = &selected_inputs;
    admit_with(&cfg, inputs, |ownership| {
        (0..ownership.ranks())
            .map(|rank| super::native::rank_admission_from_contract(&contract, &cfg, ownership, rank, inputs.unit_rows(),
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
    let contract = super::coordinator::CoordinatorContract::from_checkpoint(checkpoint)?;
    admit_mtp_with_contract(checkpoint, inputs, depth, &contract)
}

/// Reuses launch-scoped immutable headers; every candidate keeps its own rank layout.
pub fn admit_mtp_with_contract(checkpoint: &Checkpoint, inputs: &RuntimeInputs, depth: usize,
    contract: &super::coordinator::CoordinatorContract) -> Result<Admission> {
    let cfg = Config::parse(checkpoint)?;
    let mut selected_inputs = inputs.clone();
    selected_inputs.host.attention_mount_load = contract.attention_mount_load(inputs.inventory.gpus.len())?;
    let inputs = &selected_inputs;
    if depth == 0 {
        return admit_with(&cfg, inputs, |ownership| (0..ownership.ranks()).map(|rank|
            super::native::rank_admission_from_contract(contract, &cfg, ownership, rank,
                inputs.unit_rows(), inputs.rows, inputs.logit_rows())).collect());
    }
    ensure!((1..=cfg.mtp).contains(&depth) && inputs.unit_rows() > depth,
        "Step-5 predictor depth must fit checkpoint and row capacity");
    let predictor = super::coordinator::RankWeightLayout::mtp_from_contract(checkpoint, contract, depth)?;
    let weights = predictor.resident_bytes;
    let private_requests = super::native::mtp_native_scratch_requests(checkpoint, contract, depth, inputs.unit_rows())?;
    let immutable = checked_sum(private_requests.iter().copied())?;
    let workspace = super::native::mtp_forecast_workspace(inputs.unit_rows(), depth, inputs.kv_format)?;
    admit_depth_with(&cfg, inputs, depth, |ownership| {
        (0..ownership.ranks()).map(|rank| {
            let mut native = super::native::rank_admission_from_contract(&contract, &cfg, ownership, rank, inputs.unit_rows(),
                inputs.rows, inputs.logit_rows())?;
            if rank == usize::from(ownership.execution().spec().head_gpu) {
                native.resident_weights = native.resident_weights.checked_add(ResidentWeights(weights))?;
                native.weight_arenas.push((predictor.mtp_arena(rank as i32)?, ResidentWeights(weights), cuteafd_core::memory_layout::MemoryKind::DrafterWeights));
                native.immutable = native.immutable.checked_add(immutable)
                    .context("Step-5 predictor weight forecast overflow")?;
                native.drafter_requests.extend(private_requests.iter().copied());
                native.vocabulary_heads.extend((0..depth).filter(|stage| !predictor.core.contains_key(
                    &format!("model.layers.{}.transformer.shared_head.output.weight", cfg.layers + stage)))
                    .map(|stage| format!("mtp-{stage}")));
                native.workspace = super::native::with_arena_phase(native.workspace, native.projection_bytes, workspace)?;
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
    native: impl FnMut(&ExecutionOwnership) -> Result<Vec<RankAdmission>>,
) -> Result<Admission> {
    use crate::placement::families::step5 as family;
    ensure!(depth <= cfg.mtp, "Step-5 forecast predictor depth exceeds checkpoint");
    inputs.validate(cfg)?;
    let recipe = family::Step5Recipe::new(cfg, inputs, depth, native)?;
    let budgets = family::budgets(inputs).map_err(anyhow::Error::from)?;
    let solved = solve(recipe.request(), budgets, &recipe)
        .map_err(|error| {
            let clamped = matches!(error, Refusal::Capacity { .. }
                | Refusal::FullContextShortfall { .. });
            let error = anyhow::Error::from(error);
            if !inputs.marks.host && recipe.marks > 0 && clamped {
                anyhow::anyhow!("{error}; Step-5 device-only prefix retention with HOST_CACHE_BYTES=0 does not fit at request context {}; enable --host-cache-bytes auto (HOST_CACHE_BYTES=auto), or lower --max-context-tokens (MAX_CONTEXT_TOKENS), for example to 262144", inputs.context)
            } else {
                error
            }
        })?;
    recipe.execution(&solved)
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
        RuntimeInputs { graph_layout: None, decode_graphs: false, bf16_target_head: crate::placement::planner::serving::BlasHead::TensorLt,
            host: Default::default(), resident_weight_limit: None, spark_reply_format: cuteafd_core::SparkReplyFormat::Bf16,
                kv_format: crate::families::qwen4::Qwen4KvCache::Bf16,
            waves: 1, unit: 0,
            inventory: Inventory {
                gpus: vec![
                    GpuBudget {
                        capacity_bytes: 96 << 30,
                        reserve_bytes: 1 << 30,
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
            context: FLOOR, serving_context: crate::placement::planner::Context::pinned(FLOOR),
            requested_pool: Some(FLOOR),
            marks: MarkRule {
                lanes: 16,
                entries: 16,
                budget_bytes: 0,
                host: false,
            },
            peer_diagnostics: false,
            pending_code: vec![1 << 20; ranks],
            onboard: None,
        }
    }
    fn run(inputs: &RuntimeInputs) -> Result<Admission> {
        admit_with(&config(), inputs, |o| {
            Ok(vec![
                RankAdmission {
                    resident_weights: ResidentWeights(0),
                    weight_arenas: vec![],
                    immutable: 1 << 30,
                    private_requests: vec![1 << 30],
                    drafter_requests: vec![],
                    vocabulary_heads: vec![],
                    projection_bytes: 0,
                    workspace: 1 << 20
                };
                o.ranks()
            ])
        })
    }

    #[test]
    fn sampled_preweight_occupancy_is_capacity_neutral_and_named_once_per_rank() -> Result<()> {
        use crate::placement::{families::step5 as family, planner::*};
        struct Prepared(Contract);
        impl Recipe for Prepared {
            fn prepare(&self, _: &Request) -> std::result::Result<Contract, Refusal> { Ok(self.0.clone()) }
        }
        for reserve in [None, Some(0), Some(200 << 20)] {
            cuteafd_core::memory_policy::MemoryPolicy { gpu_reserve: reserve, ..Default::default() }.scoped(|| -> Result<()> {
        for occupied in [[0, 0], [682_885_120, 711_196_672]] {
            let mut input = inputs(2);
            for gpu in &mut input.inventory.gpus {
                gpu.reserve_bytes = crate::placement::reserve::gpu("step5");
            }
            for (gpu, bytes) in input.inventory.gpus.iter_mut().zip(occupied) {
                gpu.baseline = crate::placement::Baseline::Measured { free_bytes: gpu.capacity_bytes - bytes };
            }
            let cfg = config();
            let recipe = family::Step5Recipe::new(&cfg, &input, 0, |ownership| Ok(vec![RankAdmission {
                resident_weights: ResidentWeights(0), weight_arenas: vec![], immutable: 1 << 30,
                private_requests: vec![1 << 30], drafter_requests: vec![], vocabulary_heads: vec![],
                projection_bytes: 0, workspace: 1 << 20 }; ownership.ranks()]))?;
            let request = recipe.request();
            let mut contract = recipe.prepare(&request)?;
            contract.components.retain(|component| !matches!(&component.kind,
                ComponentKind::Allowance(owner) if crate::placement::sampled_baseline::is_source(&owner.source)));
            let mut old_budgets = family::budgets(&input)?;
            for (budget, bytes) in old_budgets.iter_mut().take(2).zip(occupied) {
                budget.baseline = planner::Baseline::Sampled { free: budget.physical_bytes - bytes,
                    phase: Phase::Load, source: "pre-weight CUDA free memory".into() };
            }
            let before = solve(request, old_budgets, &Prepared(contract))?;
            let now = run(&input)?.solved;
            assert_eq!(now.pool_tokens, before.pool_tokens);
            assert_eq!(now.decisions, before.decisions);
            assert_eq!(&now.budgets[2..], &before.budgets[2..], "Spark/host budgets unchanged");
            for (rank, bytes) in occupied.into_iter().enumerate() {
                assert!(matches!(now.budgets[rank].baseline, planner::Baseline::Forecast { occupied: 0, .. }));
                for phase in [Phase::Load, Phase::Ready, Phase::Growth] {
                    let items = now.items_at(Location::coordinator(rank as u8), phase)?;
                    let owners = items.iter().filter(|item| crate::placement::sampled_baseline::is_source(&item.group)).collect::<Vec<_>>();
                    assert_eq!(owners.len(), usize::from(bytes > 0));
                    assert_eq!(owners.iter().map(|owner| owner.bytes).sum::<u64>(), bytes);
                    let row = |plan: &ResolvedPlan| plan.phases.iter().find(|row|
                        row.budget == format!("rtx{rank}") && row.phase == phase).unwrap().clone();
                    let a = row(&now); let b = row(&before);
                    assert_eq!(a.available - a.owned - a.opaque, b.available - b.owned - b.opaque);
                    let breakdown = crate::placement::reserve::breakdown(format!("rtx{rank}"),
                        cuteafd_core::memory_policy::DeviceKind::Gpu, now.budgets[rank].physical_bytes,
                        now.budgets[rank].headroom, &items);
                    assert_eq!(breakdown.reserve.iter().filter(|owner|
                        owner.name == "sampled baseline").map(|owner| owner.bytes).sum::<u64>(), bytes);
                }
            }
        }
                Ok(())
            })?;
        }
        let mut offline = inputs(2);
        for gpu in &mut offline.inventory.gpus {
            gpu.baseline = crate::placement::Baseline::Planned { context_bytes: 600 << 20, loaded_bytes: 40 << 20 };
        }
        let planned = run(&offline)?.solved;
        assert!(!planned.facts.allowances.iter().any(|owner|
            crate::placement::sampled_baseline::is_source(&owner.source)));
        assert!(matches!(planned.budgets[0].baseline, planner::Baseline::Forecast { occupied, .. }
            if occupied == 640 << 20));
        let mut invalid = inputs(2);
        invalid.inventory.gpus[1].baseline = crate::placement::Baseline::Measured {
            free_bytes: invalid.inventory.gpus[1].capacity_bytes + 1 };
        assert!(run(&invalid).is_err());
        Ok(())
    }

    #[test]
    fn graph_reserve_zero_is_eager_equivalent_and_explicit_charge_is_once_per_rank() -> Result<()> {
        const CHILD: &str = "CUTEAFD_TEST_STEP5_GRAPH_RESERVE";
        let Ok(value) = std::env::var(CHILD) else {
            // Isolate reserve policy from the parent environment without mutating
            // process-wide variables while other Rust tests are running.
            let name = format!("{}::graph_reserve_zero_is_eager_equivalent_and_explicit_charge_is_once_per_rank",
                module_path!().split_once("::").unwrap().1);
            for value in ["untraced", "gpu0=64MiB,gpu1=96MiB"] {
                let mut child = std::process::Command::new(std::env::current_exe()?);
                child.args(["--exact", &name, "--nocapture"]).env(CHILD, value)
                    .env_remove("CUTEAFD_GPU_GRAPH_RESERVE")
                    .env_remove("CUTEAFD_GPU_GRAPH_RESERVE_EXTRA")
                    .env_remove("CUTEAFD_GPU_GRAPH_RESERVE_EXTRA_CAP");
                if value != "untraced" { child.env("CUTEAFD_GPU_GRAPH_RESERVE", value); }
                ensure!(child.status()?.success(), "Step-5 graph reserve child failed: {value}");
            }
            return Ok(());
        };
        for ranks in [1, 2] {
            let mut policy = inputs(ranks);
            let eager = run(&policy)?;
            policy.decode_graphs = true;
            let graphs = run(&policy)?;
            // An asymmetric nonzero reserve can legitimately change the selected
            // CSA boundary; only zero reserve promises eager-equivalent placement.
            if value == "untraced" {
                assert_eq!(graphs.placement.pool_tokens, eager.placement.pool_tokens);
                assert_eq!(graphs.placement.layers, eager.placement.layers);
            }
            for rank in 0..ranks {
                let expected = if value == "untraced" { 0 } else { [64 << 20, 96 << 20][rank] };
                let owners = graphs.solved.facts.allowances.iter().filter(|owner|
                    owner.location == crate::placement::planner::Location::coordinator(rank as u8)
                        && crate::placement::graph_reserve::is_source(&owner.source)).collect::<Vec<_>>();
                // The shared physical ledger omits zero-byte allowances.
                assert_eq!(owners.len(), usize::from(expected != 0));
                if let Some(owner) = owners.first() {
                    assert_eq!(owner.bytes, expected);
                    assert_eq!(owner.source, crate::placement::graph_reserve::source(&format!("step5.graphs.{rank}")));
                    assert!(!owner.assumed);
                }
                for phase in [crate::placement::planner::Phase::Ready, crate::placement::planner::Phase::Growth] {
                    let total = |admission: &Admission| admission.solved.facts.allowances.iter()
                        .filter(|owner| owner.location == crate::placement::planner::Location::coordinator(rank as u8)
                            && owner.lifetime.contains(phase, admission.solved.facts.phase_order))
                        .map(|owner| owner.bytes).sum::<u64>();
                    assert_eq!(total(&graphs), total(&eager) + expected);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn executor_metadata_matches_installed_csa_owners_and_swa_heads() -> Result<()> {
        assert_eq!(EXECUTOR.attention_default(), AttentionPlacement::Layers);
        let admitted = run(&inputs(2))?;
        assert_eq!(admitted.placement.attention_placement, AttentionPlacement::Layers);
        assert!(admitted.placement.attention_by_kind.contains(&(AttentionClass::Csa, AttentionPlacement::Layers)));
        assert!(admitted.placement.attention_by_kind.contains(&(AttentionClass::Swa, AttentionPlacement::Heads)));
        Ok(())
    }

    #[test]
    fn reused_header_contract_preserves_every_candidate_and_resolved_plan() -> Result<()> {
        use super::super::{coordinator::{CoordinatorContract, RankWeightLayout}, native};
        let fixture = super::super::tests::exl3_snapshot();
        let checkpoint = Checkpoint::open(fixture.path())?;
        let cfg = Config::parse(&checkpoint)?;
        let contract = CoordinatorContract::from_checkpoint(&checkpoint)?;
        let mut inputs = inputs(2);
        inputs.requested_pool = None;
        inputs.context = EXTENT;
        inputs.serving_context = crate::placement::planner::Context::pinned(EXTENT);
        inputs.inventory.spark_ranks = 6;
        inputs.kv_format = crate::families::qwen4::Qwen4KvCache::Fp8;
        let groups = cfg.full[..cfg.layers].iter().map(|&full| (full, None)).collect::<Vec<_>>();
        let mut recipes = Vec::new();
        for boundary in attention::owner_boundaries(&groups, false) {
            let owners = (0..cfg.layers).map(|layer| u8::from(layer >= boundary)).collect::<Vec<_>>();
            let ownership = inputs.ownership_for(&cfg, &owners)?;
            let mut legacy = Vec::new();
            let mut reused = Vec::new();
            for rank in 0..2 {
                // Reproduce the old independent header builds, not the optimized wrapper.
                let old_contract = CoordinatorContract::from_checkpoint(&checkpoint)?;
                let old_layout = RankWeightLayout::for_execution(&checkpoint, rank, &ownership)?;
                legacy.push(native::rank_admission_for_layout(&old_contract, &cfg, &old_layout,
                    &ownership, rank, inputs.unit_rows(), inputs.rows, inputs.logit_rows())?);
                reused.push(native::rank_admission_from_contract(&contract, &cfg, &ownership, rank,
                    inputs.unit_rows(), inputs.rows, inputs.logit_rows())?);
            }
            assert_eq!(legacy, reused, "CSA boundary {boundary}");
            recipes.push((ownership.execution().modes().to_vec(), legacy, reused));
        }
        assert_eq!(recipes.len(), 24);
        // Replay attested old/new bytes so neither resolved-plan comparison repeats header I/O.
        let resolve = |old: bool| -> Result<Admission> {
            let started = std::time::Instant::now();
            let result = admit_with(&cfg, &inputs, |ownership| {
                let (_, legacy, reused) = recipes.iter().find(|(modes, _, _)|
                    modes == ownership.execution().modes()).context("missing CSA candidate")?;
                Ok(if old { legacy.clone() } else { reused.clone() })
            })?;
            assert!(started.elapsed() < std::time::Duration::from_secs(30));
            Ok(result)
        };
        let old = resolve(true)?;
        let new = resolve(false)?;
        assert_eq!(serde_json::to_vec(&old.placement)?, serde_json::to_vec(&new.placement)?);
        assert_eq!(old.ownership.execution().modes(), new.ownership.execution().modes());
        assert_eq!(old.cache, new.cache);
        assert_eq!(old.marks, new.marks);
        assert_eq!(old.reservations, new.reservations);
        assert_eq!(old.fixed, new.fixed);
        Ok(())
    }

    #[test]
    fn combined_reply_formats_reuse_bf16_capacity_on_the_entry_gpu() -> Result<()> {
        for ranks in [1, 2] {
            for format in [cuteafd_core::SparkReplyFormat::Bf16, cuteafd_core::SparkReplyFormat::Fp8,
                    cuteafd_core::SparkReplyFormat::Nvfp4] {
                let mut input = inputs(ranks);
                input.spark_reply_format = format;
                input.rows = 3 * 2752;
                input.waves = 3;
                input.unit = 2752;
                let admission = run(&input)?;
                let intake = admission.fixed.iter().filter(|d| d.group == "Step-5 Spark intake planes")
                    .collect::<Vec<_>>();
                assert_eq!(intake.len(), 1);
                assert_eq!(intake[0].gpu, admission.ownership.execution().spec().entry_gpu);
                assert_eq!(intake[0].bytes, 4
                    * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(input.rows as u64 * 4096 * 2)?);
            }
        }
        Ok(())
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
                let tails = admission.solved.facts.allocation_granularity.iter()
                    .filter(|owner| owner.location == crate::placement::planner::Location::coordinator(rank as u8))
                    .try_fold(0u64, |total, owner| -> Result<u64> {
                        total.checked_add(owner.bytes()?).context("Step-5 test physical tails overflow")
                    })?;
                let allowances = admission.solved.facts.allowances.iter()
                    .filter(|owner| owner.location == crate::placement::planner::Location::coordinator(rank as u8))
                    .collect::<Vec<_>>();
                assert_eq!(allowances.len(), 1);
                assert!(allowances[0].source.starts_with("step5/driver-stream-slabs/"));
                assert_eq!(allowances[0].bytes, 4 << 20);
                assert!(!allowances[0].assumed);
                let payload = admission.reservations[rank].pages as u64;
                let mappings = admission.solved.facts.driver_page_tables.iter()
                    .filter(|owner| owner.location == crate::placement::planner::Location::coordinator(rank as u8)
                        && owner.lifetime.contains(crate::placement::planner::Phase::Ready, admission.solved.facts.phase_order))
                    .map(|owner| owner.bytes()).sum::<u64>();
                assert!(mappings > 0);
                assert_eq!(charged, fixed + hops[rank] + payload + tails + allowances[0].bytes + mappings);
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
    fn predictor_owners_on_the_head_gpu_move_csa_off_it() -> Result<()> {
        let cfg = config();
        for format in [crate::families::qwen4::Qwen4KvCache::Bf16,
            crate::families::qwen4::Qwen4KvCache::Fp8] {
            let mut inputs = inputs(2);
            inputs.kv_format = format;
            inputs.requested_pool = None;
            inputs.marks.host = true;
            let native = |predictor: bool| -> Vec<RankAdmission> {
                (0..2).map(|rank| RankAdmission {
                    resident_weights: ResidentWeights(0),
                    weight_arenas: vec![],
                    immutable: (1 << 30) + if rank == 0 && predictor { 32 << 30 } else { 0 },
                    private_requests: vec![(1 << 30) + if rank == 0 && predictor { 32 << 30 } else { 0 }],
                    drafter_requests: vec![],
                    vocabulary_heads: vec![],
                    projection_bytes: 0,
                    workspace: if rank == 0 && predictor { 512 << 20 } else { 1 << 20 },
                }).collect()
            };
            let base = admit_depth_with(&cfg, &inputs, 0, |_| Ok(native(false)))?;
            let candidate = admit_depth_with(&cfg, &inputs, 3, |_| Ok(native(true)))?;
            let count = |a: &Admission| a.placement.layers.iter().enumerate().filter(|(layer, item)|
                cfg.full[*layer] && item.mode == WHOLE0).count();
            // The 32 GiB predictor on rank 0 outweighs one CSA layer's pool
            // records, so the fixed boundary keeps fewer CSA layers there.
            assert!(count(&candidate) < count(&base),
                "{format:?}: predictor pressure must move CSA off the head GPU: {} vs {}",
                count(&base), count(&candidate));
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
            inputs.serving_context = crate::placement::planner::Context::pinned(EXTENT);
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
        inputs.serving_context = crate::placement::planner::Context::pinned(EXTENT);
        inputs.requested_pool = None;
        inputs.inventory.gpus[0].baseline = Baseline::Measured { free_bytes: 50 << 30 };
        inputs.marks.budget_bytes = 32 << 30;
        let refusal = run(&inputs).err().expect("device marks exceed the available budget").to_string();
        for message in ["device-only prefix retention with HOST_CACHE_BYTES=0",
            "request context 1048576", "--host-cache-bytes auto", "MAX_CONTEXT_TOKENS", "262144"] {
            assert!(refusal.contains(message), "{refusal}");
        }
        inputs.marks.host = true;
        // Host retention moves the marks off the device, but the full 1M
        // context still does not fit in 50 GiB: the old planner silently
        // clamped the pool below the request; the contract refuses instead.
        assert!(run(&inputs).is_err());
        inputs.inventory.gpus[0].baseline = Baseline::Measured { free_bytes: 96 << 30 };
        run(&inputs)?;
        inputs.marks.host = false;
        inputs.context = FLOOR;
        inputs.serving_context = crate::placement::planner::Context::pinned(FLOOR);
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
        let single = run(&inputs(1))?;
        assert!(!single.fixed.iter().any(|d| d.group == "residual hop control"));
        let mut inputs = inputs(2);
        let base = run(&inputs)?;
        for rank in 0..2 {
            let controls = base.fixed.iter().filter(|d| d.gpu == rank && d.group == "residual hop control")
                .collect::<Vec<_>>();
            assert_eq!(controls.len(), 1);
            assert_eq!(controls[0].bytes, 256);
        }
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
                            d.gpu == rank as u8 && d.group == "Step-5 private peer control"
                        })
                        .map(|d| d.bytes),
                )
            };
            assert_eq!(
                transport(&retained)? - transport(&base)?,
                9 * 48 as u64
            );
        }
        assert!(admit_with(&config(), &inputs, |_| Ok(vec![])).is_err());
        Ok(())
    }

    #[test]
    fn automatic_context_uses_admitted_pool_but_explicit_pin_refuses() -> Result<()> {
        let mut inputs = inputs(1);
        inputs.requested_pool = None;
        inputs.context = EXTENT;
        inputs.marks.entries = 0;
        inputs.sequences = 1;
        inputs.marks.lanes = 1;
        inputs.inventory.gpus[0].baseline = Baseline::Measured {
            free_bytes: 40 << 30,
        };
        inputs.serving_context = crate::placement::planner::Context::automatic(EXTENT);
        let automatic = run(&inputs)?;
        assert!((FLOOR..EXTENT).contains(&automatic.placement.pool_tokens));
        assert_eq!(automatic.solved.served_context(), automatic.placement.pool_tokens);
        assert_eq!(automatic.cache.context, EXTENT, "physical extent stays unchanged");
        inputs.serving_context = crate::placement::planner::Context::pinned(EXTENT);
        let error = run(&inputs).expect_err("40 GiB cannot hold a pinned 1M context");
        assert!(format!("{error:#}").contains("cannot fit"), "{error:#}");
        // The same device admits the 256K floor, at the request context.
        inputs.context = FLOOR;
        inputs.serving_context = crate::placement::planner::Context::pinned(FLOOR);
        let admission = run(&inputs)?;
        assert!((FLOOR..=EXTENT).contains(&admission.placement.pool_tokens));
        assert_eq!(admission.cache.context, FLOOR);
        Ok(())
    }
    #[test]
    fn host_quota_and_push_share_one_capacity_before_allocation() -> Result<()> {
        let mut inputs = inputs(1);
        inputs.marks.host = true;
        inputs.host.cache_bytes = Some(64 << 30);
        inputs.host.memory = (1 << 30, 1 << 30);
        let error = run(&inputs).expect_err("host quota must be admitted");
        assert!(error.downcast_ref::<crate::placement::planner::Refusal>().is_some(), "{error:#}");
        assert!(format!("{error:#}").contains("host"), "{error:#}");
        inputs.host.memory = (128 << 30, 128 << 30);
        inputs.host.push = Some(crate::push_load::PushHostEnvelope::new(vec![1024], vec![2048], 128, 1 << 20)?);
        let admitted = run(&inputs)?;
        assert_eq!(admitted.host_cache_bytes, 64 << 30);
        let load = admitted.solved.phases.iter().find(|row| row.budget == "host"
            && row.phase == crate::placement::planner::Phase::Load).unwrap();
        assert!(load.owned >= inputs.host.push.as_ref().unwrap().bytes()?);
        Ok(())
    }

    #[test]
    fn portable_host_prefix_maps_once_per_context_only_while_retained() -> Result<()> {
        use crate::placement::planner::{driver_page_table_bytes, Location, Phase};
        for ranks in [1, 2] {
            let mut input = inputs(ranks);
            input.marks.host = true;
            input.host.cache_bytes = Some(32_941_696_000);
            input.host.memory = (128 << 30, 128 << 30);
            let admitted = run(&input)?;
            let facts = &admitted.solved.facts;
            let prefix = facts.deferred.iter().find(|owner| owner.id == "step5.host.prefix").unwrap();
            assert_eq!((prefix.location, prefix.bytes), (Location::PINNED, 32_941_696_000));
            let mappings = facts.driver_page_tables.iter().filter(|owner|
                owner.backing == prefix.id).collect::<Vec<_>>();
            assert_eq!(mappings.len(), ranks);
            for rank in 0..ranks {
                let mapping = mappings.iter().find(|owner|
                    owner.location == Location::coordinator(rank as u8)).unwrap();
                assert_eq!(mapping.mapped_bytes, prefix.bytes);
                assert_eq!(mapping.bytes(), driver_page_table_bytes(prefix.bytes));
                assert_eq!(mapping.bytes(), 64_339_250);
                assert!(!mapping.lifetime.contains(Phase::Load, facts.phase_order));
                assert!(mapping.lifetime.contains(Phase::Ready, facts.phase_order));
                assert!(mapping.lifetime.contains(Phase::Growth, facts.phase_order));
            }
            input.host.cache_bytes = Some(0);
            let disabled = run(&input)?;
            assert!(!disabled.solved.facts.driver_page_tables.iter().any(|owner|
                owner.backing == "step5.host.prefix"));
            assert!(!disabled.solved.facts.deferred.iter().any(|owner|
                owner.id == "step5.host.prefix"));
        }
        Ok(())
    }

    #[test]
    fn synchronous_h2d_staging_remains_owned_through_ready_and_growth() -> Result<()> {
        let mut inputs = inputs(1);
        inputs.host.retained_sync_h2d_bytes = 395_968_512;
        let admitted = run(&inputs)?;
        let owner = admitted.solved.facts.allocations.iter()
            .find(|owner| owner.id == "step5.host.sync_h2d").unwrap();
        assert_eq!(owner.bytes, 395_968_512);
        assert_eq!(owner.location, crate::placement::planner::Location::PINNED);
        assert_eq!(owner.lifetime, crate::placement::planner::Lifetime::RESIDENT);
        Ok(())
    }

    #[test]
    fn bounded_host_logits_do_not_grow_with_prefill_span() -> Result<()> {
        for rows in [64, 256, 4096] {
            let mut inputs = inputs(1);
            inputs.rows = rows;
            inputs.unit = rows;
            let admitted = run(&inputs)?;
            let logits = admitted.solved.facts.deferred.iter()
                .find(|owner| owner.id == "step5.host.logits").unwrap();
            assert_eq!(logits.bytes, 64 * config().vocab as u64 * 4);
        }
        Ok(())
    }

    #[test]
    fn bf16_predictors_share_one_named_gemm_pool_and_native_heads_have_none() -> Result<()> {
        use crate::placement::planner::{Location, Phase, serving::BlasHead};
        for native in [false, true] {
            let fixture = if native { super::super::tests::exl3_snapshot() }
                else { super::super::tests::snapshot("bf16") };
            let checkpoint = Checkpoint::open(fixture.path())?;
            let contract = super::super::coordinator::CoordinatorContract::from_checkpoint(&checkpoint)?;
            for ranks in [1, 2] {
                for mode in [BlasHead::TensorLt, BlasHead::Gemm] {
                    let mut input = inputs(ranks);
                    input.bf16_target_head = mode;
                    for depth in [0, 1, 3] {
                        let admitted = admit_mtp_with_contract(&checkpoint, &input, depth, &contract)?;
                        let facts = &admitted.solved.facts;
                        for &phase in facts.phase_order.phases() {
                            let handles = facts.allowances.iter().filter(|owner|
                                owner.source.starts_with("cuBLAS handles: step5/blas/serve")
                                    && owner.lifetime.contains(phase, facts.phase_order)).collect::<Vec<_>>();
                            let expected = usize::from(!native && (depth > 0 || mode.gemm()));
                            assert_eq!(handles.len(), expected, "one GEMM per host thread/device, not predictor");
                            assert_eq!(handles.iter().map(|owner| owner.bytes).sum::<u64>(),
                                expected as u64 * (64 << 20));
                            assert!(handles.iter().all(|owner| owner.location == Location::coordinator(0)));
                        }
                        let init = facts.allowances.iter().filter(|owner| owner.source.starts_with("BLAS device initialization")
                            && owner.lifetime.contains(Phase::Growth, facts.phase_order)).collect::<Vec<_>>();
                        assert_eq!(init.len(), usize::from(!native));
                        assert_eq!(init.iter().map(|owner| owner.bytes).sum::<u64>(), u64::from(!native) * (2 << 20));
                        assert!(init.iter().all(|owner| owner.location == Location::coordinator(0)));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn runtime_request_tails_reference_exact_payload_owners_once() -> Result<()> {
        use crate::placement::planner::{allocation_granularity_forecast, Location};
        let admitted = run(&inputs(2))?;
        let facts = &admitted.solved.facts;
        let mut seen = std::collections::BTreeSet::new();
        for owner in &facts.allocation_granularity {
            let requests = owner.requests();
            assert_eq!(owner.bytes()?, allocation_granularity_forecast(&requests)?);
            for backing in &owner.backings {
                assert!(seen.insert(&backing.backing), "duplicate physical request charge");
                let payload = facts.allocations.iter().chain(&facts.deferred)
                    .find(|allocation| allocation.id == backing.backing).unwrap();
                assert_eq!(payload.location, owner.location);
                assert_eq!(payload.bytes, backing.requests.iter().sum::<u64>());
                assert!(payload.view.is_none());
                if payload.group == "Step-5 private transaction staging" {
                    assert_eq!(payload.category, Category::Staging);
                    assert_eq!(backing.requests, vec![payload.bytes]);
                }
            }
        }
        assert_eq!(facts.allocation_granularity.len(), 4);
        for rank in 0..2 {
            assert!(facts.allocation_granularity.iter().any(|owner| owner.location == Location::coordinator(rank)));
        }
        Ok(())
    }

    #[test]
    fn explicit_unaligned_pool_keeps_its_pin_and_charges_physical_records() -> Result<()> {
        let mut input = inputs(1);
        input.context = EXTENT;
        input.serving_context = crate::placement::planner::Context::resolve(EXTENT, Some(FLOOR));
        input.requested_pool = Some(FLOOR + 64);
        let admitted = run(&input)?;
        assert_eq!(admitted.placement.pool_tokens, FLOOR + 64);
        let payload = admitted.reservations[0].pages as u64;
        let page = crate::placement::planner::SM120_ALLOCATION_GRANULARITY_BYTES;
        let owner = admitted.solved.facts.allocations.iter()
            .find(|owner| owner.id == "step5.records.0").unwrap();
        assert_eq!(owner.bytes, payload);
        assert_eq!(owner.shape, vec![payload]);
        let tail = admitted.solved.facts.allocation_granularity.iter()
            .find(|tail| tail.id == "step5.records.0.tail").unwrap();
        assert_eq!(tail.backings[0].requests, vec![payload]);
        assert_eq!(payload + tail.bytes()?, payload.div_ceil(page) * page);
        Ok(())
    }

    #[test]
    fn automatic_pool_keeps_every_affordable_page_without_alignment_loss() -> Result<()> {
        let mut input = inputs(1);
        input.context = EXTENT;
        input.serving_context = crate::placement::planner::Context::automatic(EXTENT);
        input.requested_pool = None;
        input.kv_format = crate::families::qwen4::Qwen4KvCache::Fp8;
        let probe = run(&input)?;
        let location = crate::placement::planner::Location::coordinator(0);
        let phase = probe.solved.phases.iter().find(|phase| phase.phase ==
            crate::placement::planner::Phase::Growth && phase.budget == probe.solved.budgets[0].id).unwrap();
        let records = probe.solved.facts.allocations.iter().find(|owner|
            owner.id == "step5.records.0" && owner.location == location).unwrap();
        let tail = probe.solved.facts.allocation_granularity.iter()
            .find(|owner| owner.id == "step5.records.0.tail").unwrap().bytes()?;
        let fixed = phase.owned + phase.opaque - records.bytes - tail;
        let unit = probe.cache.geometry.ranks[0].persistent_unit_bytes;
        let quantum = crate::placement::planner::SM120_ALLOCATION_GRANULARITY_BYTES;
        let wanted = 22_096;
        let physical = (unit * wanted).div_ceil(quantum) * quantum;
        let budget = fixed + physical;
        input.inventory.gpus[0].reserve_bytes = 0;
        input.inventory.gpus[0].baseline = Baseline::Measured { free_bytes: budget };
        let admitted = run(&input)?;
        assert_eq!(admitted.placement.pool_tokens / 64, physical / unit);
        assert!(admitted.placement.pool_tokens >= wanted * 64);
        assert_ne!(admitted.placement.pool_tokens / 64 % 1024, 0);
        Ok(())
    }

    #[test]
    fn full_onboard_option_charges_its_expert_arena() -> Result<()> {
        let mut inputs = inputs(2);
        let routed = config().moe[..92].iter().filter(|&&moe| moe).count();
        inputs.onboard = Some(OnboardInputs {
            costs: vec![ExpertCost { whole: Bytes2 { resident: 1024, staging: 0 },
                half: [Bytes2 { resident: 512, staging: 0 }; 2], tp2: true,
                whole_supported: true, spark_ok: true }; routed],
            workspace: 12345, tp2_workspace: [23456, 34567],
            workspace_requests: vec![vec![23456], vec![34567]], mode: Onboard::Layers(routed),
        });
        let admitted = run(&inputs)?;
        assert_eq!(admitted.placement.onboard_layers, routed);
        for rank in 0..2 {
            let arena = admitted.placement.items[rank].iter()
                .filter(|item| item.group == "Step-5 RTX expert arena").map(|item| item.bytes).sum::<u64>();
            assert_eq!(arena, inputs.onboard.as_ref().unwrap().tp2_workspace[rank]);
            assert_eq!(admitted.placement.tp2.as_ref().unwrap().peak_bytes[rank], arena + routed as u64 * 512);
        }
        Ok(())
    }

}
