//! Muse Spark physical owners for the shared contract planner.
//!
//! A family supplies facts, never a private solver: this module names every
//! owner Muse charges (layer weights, KV, rings, marks, workspaces, transport,
//! expert arenas, bundled towers, drafter), where it lives and for which
//! lifetime. `cuteafd plan --layout` and serving startup build the same recipe
//! and run the same solve, so the two edges differ only in the coordinator
//! baseline (a planned forecast versus one measured sample) and in whether the
//! image's code is already loaded.
use super::{combine_tables_bytes, executor, intake, mark_bytes, ring_rows, rope_tables, spark_endpoint,
    spark_reserve_bytes, step_tables, tp2_exchange_bytes, tp2_partial_bytes, LocalPackage, MediaAdmission,
    MediaOptions, MuseDrafter, LANES, MUSE_SPARK, PAGE_ROWS};
use crate::families::muse_spark::{core::CoreInventory, weight_layout::{GlobalLayout, LayerLayout},
    MuseAttention, MuseSparkConfig};
use crate::placement::planner::*;
use crate::placement::{plan_hops, residual_homes, AttentionClass, AttentionPlacement, Demand, ExpertHome,
    ExpertRange, HopSpec, HopStorage, LayerAssignment, LayerMode, Onboard, Placement, PoolLimits, PoolPolicy,
    ResidentWeights, SparkBudgetInventory, SparkWorkingSet, TerminalHome, Tp2Range};
use crate::plan::encoder::EncoderPlacement;
use crate::plan::MediaMode;
use crate::serving_capacity::prefill_waves::PrefillPlan;
use cuteafd_core::memory_layout::{Basis, Category, Item};
use cuteafd_core::spark_reply::SparkReplyFormats;
use anyhow::Context as _;

/// A condition names a domain value, never a solver variable index: one
/// onboarding count or one joint bundled-tower option, or nothing at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum When {
    Onboard(usize),
    Encoder(usize),
}

#[derive(Debug, Clone)]
pub struct Case<T> {
    pub when: Vec<When>,
    pub value: T,
}

fn invalid(detail: &str) -> Refusal {
    Refusal::InvalidFact(format!("Muse Spark component contract: {detail}"))
}

fn anyhow_refusal(error: anyhow::Error) -> Refusal { invalid(&format!("{error:#}")) }

fn dense_push_staging(core: &CoreInventory, gpus: usize) -> Result<(u64, String), Refusal> {
    if core.native.is_empty() {
        // Preserve the parent K6 for_steps bounds, including their attribution.
        let bytes = if gpus == 2 { 2 << 30 } else { 1 << 30 };
        return Ok((bytes, format!("Muse dense push staging: estimated {} GiB/rank from 1191-muse-min-pair / muse-max-push-incast8 for_steps bounds (#1274)", bytes >> 30)));
    }
    let largest = core.projections.values().map(|projection| projection.source_bytes).max().unwrap_or(0);
    // Native FP8 conversion starts from the whole selected BF16 source, before TP slicing.
    let largest = core.native.values().try_fold(largest, |largest, projection| {
        let [n, k] = projection.shape;
        let bytes = u64::try_from(n).ok().and_then(|n| u64::try_from(k).ok().and_then(|k| n.checked_mul(k)))
            .and_then(|elements| elements.checked_mul(2)).ok_or(Refusal::Overflow("Muse native push source"))?;
        Ok::<_, Refusal>(largest.max(bytes))
    })?;
    let bytes = crate::push_load::admitted_push_envelope(largest,
        cuteafd_ffi::push_ring::DEFAULT_SLOTS, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64)?;
    Ok((bytes, "Muse native dense push staging: shared windowed receive and transform envelope from selected source headers".into()))
}

/// The largest coordinator KV pool this deployment admits, from the one shared
/// pool policy. Both edges call this before either solves.
pub fn maximum_pool_tokens(capacities: &[u64], context: u64, requested: Option<u64>) -> u64 {
    PoolPolicy::resolve(capacities, context, requested, PAGE_ROWS).maximum_tokens()
}

/// One coordinator GPU's planned budget: the CUDA context is the forecast
/// charge; everything else the family states as an owner.
pub fn planned_budget(gpu: u8, physical_bytes: u64, capacity_bytes: u64, headroom_bytes: u64, context_bytes: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)],
        physical_bytes, logical_bytes: capacity_bytes, headroom: headroom_bytes,
        baseline: Baseline::Forecast { occupied: context_bytes,
            source: "CUDA context and modules on the coordinator GPU".into(), assumed: false } }
}

/// A measured sample keeps physical occupancy independent of the logical cap.
pub fn measured_budget(gpu: u8, physical_bytes: u64, capacity_bytes: u64, headroom_bytes: u64, free_bytes: u64) -> Budget {
    Budget { id: format!("rtx{gpu}"), locations: vec![Location::coordinator(gpu)],
        physical_bytes, logical_bytes: capacity_bytes, headroom: headroom_bytes,
        baseline: Baseline::Sampled { free: free_bytes, phase: Phase::Load,
            source: "coordinator free memory at admission".into() } }
}

/// One Spark worker's managed arena: the operator budget is both ceilings, and
/// its I/O and runtime reserve are owners inside it (as the runtime charges them).
pub fn spark_budget(rank: u8, bytes: u64) -> Budget {
    Budget { id: format!("spark{rank}"),
        locations: vec![Location { device: Device::Spark(rank), domain: Domain::Managed }],
        physical_bytes: bytes, logical_bytes: bytes, headroom: 0,
        baseline: Baseline::Forecast { occupied: 0,
            source: "Spark worker arena owned by the serving process".into(), assumed: false } }
}

/// The one request both edges submit; no edge re-derives a pin.
pub fn production_request(context: impl Into<crate::placement::planner::Context>, pool: Option<u64>, onboard: Onboard, prefill: PrefillShape,
    operation: Operation, resident_weight_limit: Option<ResidentWeights>,
    spark_reply_formats: cuteafd_core::spark_reply::SparkReplyFormats) -> Request {
    Request { context: context.into(),
        pool: PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard { Onboard::Auto => Pin::Auto, Onboard::Layers(n) => Pin::Exact(n as u64) },
        prefill: Pin::Exact(prefill), operation, resident_weight_limit, spark_reply_formats }
}

/// One host sample and the operator's retention settings. HOST and PINNED
/// consume the same physical budget; quotas are resolved before the solve.
#[derive(Debug, Clone, Copy)]
pub struct HostInput {
    pub total: u64,
    pub available: u64,
    pub reserve: u64,
    pub prefix_bytes: Option<u64>,
    pub prefix_entries: u64,
    pub media_bytes: Option<u64>,
    pub device_sampling: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct HostAdmission {
    pub prefix_bytes: u64,
    pub prefix_chunk: u64,
    pub media_bytes: u64,
}

/// Conservative estimate rounded above the complete 1,233,498,112-byte ordinary
/// RSS census; not an exact residual calibration of allocator/metadata state.
pub const HOST_RUNTIME: crate::placement::inventory::MemoryFigure = crate::placement::inventory::MemoryFigure {
    bytes: 1536 << 20,
    attribution: crate::placement::inventory::FigureAttribution::Estimated {
        derivation: "1.5 GiB ordinary state; 2026-11-06 Muse min scheduler-ready RSS 1233498112 B rounded up (#1326)" },
    evidence: "planner-muse-close-build media smoke; complete ordinary RSS census (#1326)",
    producing_commit: "c6d6fd47efa3daeac460772673f9826e20a32944",
    identity: crate::placement::inventory::FigureIdentity { declared_graphs: "Muse serving",
        programs_sha256: "unmeasured", native_sha256: "unmeasured", toolchain: "unmeasured",
        gpu_class: "sm120", driver: "unmeasured", layout: "Muse host ordinary residual" },
};

/// Streams that can be live at once on one coordinator GPU: the model stream
/// (`serve.rs` `Stream::new` per device), the push loader's copy streams
/// (`push_load::COPY_STREAMS`, used while the dense load is recorded), and the
/// transient rotary-table builder (`shared/rope.rs` `RotaryTables::load`).
/// Destroyed streams reuse driver slots, so only the peak counts (#1561).
pub fn startup_peak_live_streams() -> u64 { 1 + crate::push_load::COPY_STREAMS as u64 + 1 }

impl HostInput {
    pub fn budget(self) -> Budget {
        Budget { id: "host".into(), locations: vec![Location::HOST, Location::PINNED],
            physical_bytes: self.total, logical_bytes: self.total, headroom: self.reserve.max(self.total / 10),
            baseline: Baseline::Sampled { free: self.available, phase: Phase::Load,
                source: "coordinator MemAvailable/cgroup working set before admission".into() } }
    }

    pub fn quotas(self, cfg: &MuseSparkConfig, context: u64, slots: u64) -> Result<HostAdmission, Refusal> {
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: PAGE_ROWS,
            page_bytes: cfg.full_layers() as u64 * cfg.kv_record_bytes() as u64 * PAGE_ROWS,
            mark_bytes: mark_bytes(cfg, 1), draft_bytes: 0 };
        let prefix_bytes = if self.prefix_entries == 0 { 0 } else {
            geometry.resolve(self.prefix_bytes, context, slots, self.total, geometry.chunk_bytes())
                .ok_or_else(|| invalid("host prefix quota geometry overflow"))? };
        if self.prefix_entries > 0 && self.prefix_bytes != Some(0) && prefix_bytes == 0 {
            return Err(invalid("host prefix quota resolves to zero; disable explicitly or provide a positive quota"));
        }
        Ok(HostAdmission { prefix_bytes, prefix_chunk: geometry.chunk_bytes().min(prefix_bytes),
            media_bytes: self.media_bytes.unwrap_or((8 << 30).min(self.total / 20)) })
    }
}

/// Everything Muse knows before a solve: geometry, budgets and header-exact
/// cost tables, identical on the planned and measured edges.
pub struct ProductionInput<'a> {
    pub cfg: &'a MuseSparkConfig,
    pub core: &'a CoreInventory,
    /// Preweight resolution retained unchanged through admission and owner checks.
    pub projections: Vec<crate::families::muse_spark::weight_layout::RankProjections>,
    pub catalog: &'a crate::OfficialV41Catalog,
    /// One coordinator budget per RTX plus one managed budget per Spark rank.
    pub budgets: Vec<Budget>,
    pub host: HostInput,
    pub prefill: PrefillPlan,
    pub full_prefill_logits: bool,
    /// Active sequence slots (each owns its sliding rings).
    pub slots: u64,
    /// Device mark slots from the shared prefix rule (the host tier holds retention).
    pub mark_slots: u64,
    /// The compiled context floor; `maximum_pool_tokens` is the extent.
    pub context: u64,
    pub maximum_pool_tokens: u64,
    pub partial_bf16_max_rows: usize,
    pub partial_reply_formats: cuteafd_core::spark_reply::SparkReplyFormats,
    /// The coordinator EXL3 package for onboarded layers.
    pub local: LocalPackage,
    /// The DFlash2 drafter, charged on the head GPU.
    pub drafter: Option<MuseDrafter>,
    /// Runtime-only code the image will load (the planner's full allowance,
    /// startup's not-yet-loaded remainder).
    pub fixed: Vec<Demand>,
    /// Bundled-tower headroom facts.
    pub media: MediaOptions,
}

/// The prepared recipe plus everything the caller reports from it.
pub struct PreparedMuse<'a> {
    pub recipe: MuseRecipe<'a>,
    budgets: Vec<Budget>,
    prefill_logits_bytes: usize,
    local: LocalPackage,
    drafter: Option<MuseDrafter>,
    layouts: Vec<Vec<LayerLayout>>,
    global: GlobalLayout,
    projections: Vec<crate::families::muse_spark::weight_layout::RankProjections>,
    spark_budget: SparkBudgetInventory,
    host: HostAdmission,
    push: Vec<crate::push_load::PushHostEnvelope>,
}

/// The one admitted answer. Both edges read the same fields.
pub struct AdmittedMuse {
    pub placement: Placement,
    pub spark: SparkWorkingSet,
    pub solved: ResolvedPlan,
    pub media: MediaAdmission,
    pub local: LocalPackage,
    pub drafter: Option<MuseDrafter>,
    /// The coordinator GPU the solve placed the drafter on (the target head).
    pub draft_home: Option<u8>,
    pub prefill_logits_bytes: usize,
    /// Per layer, per coordinator rank: the operands the device packs.
    pub layouts: Vec<Vec<LayerLayout>>,
    pub global: GlobalLayout,
    /// Per rank: the K6 programs it binds and their shared scratch.
    pub projections: Vec<crate::families::muse_spark::weight_layout::RankProjections>,
    /// Header-exact per-rank Spark arena slices for the selected onboarding.
    pub spark_budget: SparkBudgetInventory,
    pub solve_us: u64,
    pub host: HostAdmission,
    pub push: crate::push_load::PushHostEnvelope,
}

impl AdmittedMuse {
    /// Bytes of one rank's layer weights across all layers.
    pub fn layer_bytes(&self, gpu: usize) -> anyhow::Result<u64> {
        self.layouts.iter().try_fold(0u64, |sum, ranks| {
            let bytes = ranks.get(gpu).context("Muse layout rank")?.bytes()? as u64;
            sum.checked_add(bytes).context("Muse layer bytes overflow")
        })
    }
}

/// The family's immutable cost terms. Every component byte figure is fixed;
/// the pool charge is the paged full-layer KV slope the contract prices per page.
pub struct MuseRecipe<'a> {
    pub cfg: &'a MuseSparkConfig,
    pub context: u64,
    pub gpus: usize,
    pub maximum_pages: u64,
    pub prefill: PrefillShape,
    pub attention: Vec<LayerMode>,
    pub spark_ranks: usize,
    /// Joint bundled-tower options, in preference order (`Off` last for `Auto`).
    pub encoders: Vec<(EncoderPlacement, EncoderPlacement)>,
    pub modes: (MediaMode, MediaMode),
    pub cases: Vec<Case<Vec<Component>>>,
    /// Expert arena peak per onboarding count, per GPU.
    pub peaks: Vec<[u64; 2]>,
    pub local: LocalPackage,
    /// The DFlash2 drafter whose head-adjacent owner the contract declares.
    pub drafter: Option<MuseDrafter>,
}

/// One allocation, dropped when it charges nothing.
fn buffer(id: String, location: Location, bytes: u64, lifetime: Lifetime, category: Category, group: &str,
    maximum: u64) -> Option<Component> {
    (bytes > 0).then(|| Component::allocation(
        Allocation::buffer(id, location, bytes, lifetime, category, group),
        ByteCurve::fixed(bytes, maximum), false, maximum))
}

/// Retained weights: their allocation plus the residency fact, whether or not
/// the allocation is still in the ledger.
fn weights(id: String, location: Location, bytes: u64, resident: u64, category: Category, group: &str,
    maximum: u64) -> Result<Vec<Component>, Refusal> {
    if resident > bytes { return Err(invalid("resident weights exceed their backing allocation")); }
    let mut out = buffer(id.clone(), location, bytes, Lifetime::RESIDENT, category, group, maximum)
        .into_iter().collect::<Vec<_>>();
    if resident > 0 {
        out.push(Component::resident_weight(ResidentWeight { id, location, weights: ResidentWeights(resident) },
            ByteCurve::fixed(resident, maximum)));
    }
    Ok(out)
}

/// The joint preference order of the two bundled towers: every Spark replica
/// set (the requested lead first), then the RTX owner, then — for automatic
/// media only — disabled last. No free-memory screen: the one solve decides
/// capacity, so a plan and a measured startup build the same order.
fn tower_options(media: &MediaOptions, gpus: usize, sparks: usize)
    -> Result<Vec<(EncoderPlacement, EncoderPlacement)>, Refusal> {
    use crate::plan::encoder::{candidates, encoder_placement, EncoderGpuBudget, EncoderHardware,
        EncoderRequest, EncoderSparkBudget};
    let hardware = EncoderHardware { v41: false,
        gpus: (0..gpus).map(|_| EncoderGpuBudget { free_bytes: u64::MAX, kv_target_bytes: 0 }).collect(),
        sparks: (0..sparks).map(|rank| EncoderSparkBudget { rank, host: format!("spark{rank}"), idle: false,
            expert_bytes: 0, free_bytes: u64::MAX }).collect() };
    let tower = |name: &str, mode: MediaMode, bytes: (u64, u64), copies: usize| -> Result<Vec<EncoderPlacement>, Refusal> {
        let mut list = candidates(&EncoderRequest { mode, weights: bytes.0, scratch: bytes.1, copies }, &hardware);
        if mode == MediaMode::Auto && bytes.0 > 0 {
            // Auto always closes its domain with Off; an enabled tower the
            // solver rejects is then reported by the shared capacity
            // diagnosis, while a selected Off shows as disabled media bytes.
            list.push(encoder_placement(MediaMode::Off, &hardware, 0, 0, 1));
        }
        if list.is_empty() {
            return Err(invalid(&format!("requested {name} tower ({mode:?}) has no executable device")));
        }
        Ok(list)
    };
    let vision = tower("vision", media.vision, media.vision_bytes, media.replicas)?;
    let audio = tower("audio", media.audio, media.audio_bytes, 1)?;
    Ok(vision.iter().flat_map(|vision| audio.iter().map(move |audio| (vision.clone(), audio.clone())))
        .collect())
}

impl<'a> ProductionInput<'a> {
    pub fn prepare(self) -> Result<PreparedMuse<'a>, Refusal> {
        let ProductionInput { cfg, core, projections, catalog, mut budgets, host, prefill, full_prefill_logits, slots, mark_slots,
            context, maximum_pool_tokens, partial_bf16_max_rows, partial_reply_formats, local, drafter, fixed,
            media } = self;
        let coordinator = budgets.iter().filter(|budget| budget.locations.iter()
            .any(|location| matches!(location.device, Device::Coordinator(_)))).count();
        let spark_ranks = budgets.iter().filter(|budget| budget.locations.iter()
            .any(|location| matches!(location.device, Device::Spark(_)))).count();
        if !(1..=2).contains(&coordinator) || spark_ranks > 8 {
            return Err(invalid("Muse Spark runs on one or two RTX with at most eight Spark ranks"));
        }
        if maximum_pool_tokens < context {
            return Err(Refusal::PoolBelowContext { tokens: maximum_pool_tokens, context });
        }
        if slots == 0 || context == 0 || context > cfg.max_position as u64 {
            return Err(invalid(&format!("slots {slots} / context {context} outside 1..={}", cfg.max_position)));
        }
        if !catalog.per_expert_output_norm() {
            return Err(invalid("Spark experts must carry their post-expert norm"));
        }
        let host_quotas = host.quotas(cfg, context, slots)?;
        budgets.push(host.budget());
        let gpus = coordinator;
        let pair = gpus == 2;
        let (_, modes) = executor(gpus).map_err(anyhow_refusal)?;
        let attention = vec![modes[0]; cfg.layers];
        let layouts = (0..cfg.layers).map(|layer| LayerLayout::ranks(cfg, layer, gpus, core))
            .collect::<anyhow::Result<Vec<_>>>().map_err(anyhow_refusal)?;
        let global = GlobalLayout::new(cfg, core).map_err(anyhow_refusal)?;
        let maximum_pages = maximum_pool_tokens.div_ceil(PAGE_ROWS);
        let slots_usize = usize::try_from(slots).map_err(|_| Refusal::Overflow("Muse slots"))?;
        let unit_rows = usize::try_from(prefill.unit_rows).map_err(|_| Refusal::Overflow("Muse prefill rows"))?;
        let waves = usize::try_from(prefill.waves).map_err(|_| Refusal::Overflow("Muse prefill waves"))?;
        let record = cfg.kv_record_bytes() as u64;
        let rings = cfg.sliding_layers() as u64 * ring_rows(cfg).map_err(anyhow_refusal)? * record;
        let marks = mark_bytes(cfg, 1);
        let lane = crate::families::muse_spark::lane::LanePlanes::with_waves(cfg, gpus, unit_rows, waves)
            .and_then(|planes| planes.bytes()).map_err(anyhow_refusal)? as u64;
        let tables = step_tables(unit_rows, slots_usize, context, waves).map_err(anyhow_refusal)?.bytes()
            as u64;
        if projections.len() != gpus { return Err(invalid("projection rank count differs from budgets")); }
        for (gpu, admitted) in projections.iter().enumerate() {
            let actual = layouts.iter().flat_map(|ranks| ranks[gpu].native.values().cloned()
                .map(crate::families::muse_spark::weight_layout::NativeRequest::new)).collect::<Vec<_>>();
            admitted.validate_native_owners(&actual, unit_rows)?;
        }
        let exchange = if pair { Some(tp2_exchange_bytes(cfg, unit_rows).map_err(anyhow_refusal)? as u64) }
            else { None };
        let combine = combine_tables_bytes(unit_rows) as u64 * prefill.waves;
        let rope = rope_tables(cfg, context).map_err(anyhow_refusal)?.allocation_bytes;
        let (stored, trimmed) = crate::families::muse_spark::lane::head_planes(
            global.head.as_ref().map_or(cfg.vocab_size, |head| head.shape[0]), cfg.vocab_size)
            .map_err(anyhow_refusal)?;
        let prefill_logits_bytes = if full_prefill_logits {
            crate::families::muse_spark::lane::prefill_logits_bytes(
                usize::try_from(prefill.span_rows).map_err(|_| Refusal::Overflow("Muse span rows"))?,
                cfg.vocab_size).map_err(anyhow_refusal)?
        } else { 0 };
        // Validate the same widest verify step the runtime seals.
        let cutoff = crate::families::muse_spark::replies::Cutoff::with_formats(partial_bf16_max_rows,
            crate::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
            partial_reply_formats).map_err(anyhow_refusal)?;
        let (_, planes) = intake(cfg, prefill, cutoff, slots_usize, spark_ranks).map_err(anyhow_refusal)?;
        let head_gpu = u8::try_from(gpus - 1).map_err(|_| Refusal::Overflow("Muse GPUs"))?;

        // Fixed owners: the list the request builder charged, now with real
        // lifetimes and one uniform charge path.
        let mut components = crate::placement::planner::serving::ServingOwners {
            selector: crate::placement::planner::serving::SelectorOwners {
                location: Location::coordinator(head_gpu),
                capacity: crate::families::muse_spark::lane::HEAD_ROWS, vocab: cfg.vocab_size,
                device_sampling: host.device_sampling },
            prefix: crate::placement::planner::serving::HostPrefixOwners {
                quota_bytes: host_quotas.prefix_bytes, chunk_bytes: host_quotas.prefix_chunk },
        }.components(maximum_pages)?;
        let transport = super::host_transport(cfg, prefill, cutoff, slots_usize, spark_ranks)
            .map_err(anyhow_refusal)?;
        for (id, group, bytes) in [("rings", "transport/rdma-rings", transport.rings),
            ("egress", "transport/egress", transport.egress),
            ("intake", "spark-intake", transport.intake)] {
            components.extend(buffer(format!("muse.host.{id}"), Location::PINNED, bytes,
                Lifetime::RESIDENT, Category::Staging, group, maximum_pages));
        }
        if full_prefill_logits {
            let bytes = cuteafd_core::memory_layout::host_logits_bytes(prefill.span_rows,
                prefill.unit_rows, cfg.vocab_size as u64).ok_or(Refusal::Overflow("Muse scoring logits"))?;
            components.extend(buffer("muse.host.scoring".into(), Location::HOST, bytes,
                Lifetime { first: Phase::Warmup, last: Phase::Growth }, Category::Workspace,
                "host scoring logits", maximum_pages));
        }
        components.push(Component { activation: Activation::always(),
            bytes: ByteCurve::fixed(HOST_RUNTIME.bytes, maximum_pages),
            kind: ComponentKind::Allowance(Allowance { location: Location::HOST, bytes: HOST_RUNTIME.bytes,
                lifetime: Lifetime::RESIDENT, source: HOST_RUNTIME.evidence.into(), assumed: true }) });
        for gpu in 0..gpus {
            let location = Location::coordinator(gpu as u8);
            // Driver stream slabs survive delivery-stream retirement and are
            // driver-allocated, not a cudaMalloc request, so they receive no
            // allocation-tail charge of their own.
            let peak = startup_peak_live_streams();
            let slab = cuda_stream_slab_bytes(peak)?;
            components.push(Component { activation: Activation::always(),
                bytes: ByteCurve::fixed(slab, maximum_pages),
                kind: ComponentKind::Allowance(Allowance { location, bytes: slab,
                    lifetime: Lifetime::RESIDENT, assumed: false,
                    source: format!("muse/driver-stream-slabs/gpu{gpu} ({peak} peak live streams: \
                        model + {} push copies + transient rotary, SM120 driver/runtime13020 LAZY, \
                        #1511 2026-11-08)", crate::push_load::COPY_STREAMS) }) });
            let rank_layers = layouts.iter().map(|ranks| &ranks[gpu]).collect::<Vec<_>>();
            let arena = crate::families::muse_spark::weight_layout::coordinator_arena(
                gpu, gpus, &global, &rank_layers).map_err(anyhow_refusal)?;
            components.extend(weight_arena_components(&arena, location, Lifetime::RESIDENT, maximum_pages, cuteafd_core::memory_layout::MemoryKind::Weights)?);
            let mount = crate::families::muse_spark::weight_layout::native_mount_load(
                layouts.iter().flat_map(|ranks| ranks[gpu].native.values())).map_err(anyhow_refusal)?;
            for owner in mount.allocations(&format!("muse.native.load.{gpu}")) {
                let bytes = ByteCurve::fixed(owner.bytes, maximum_pages);
                components.push(Component::allocation(owner, bytes, false, maximum_pages));
            }
            components.extend(buffer(format!("muse.rings.{gpu}"), location, rings * slots / gpus as u64,
                Lifetime::RESIDENT, Category::Kv,
                &format!("sliding rings, {slots} sequences x {} layers", cfg.sliding_layers()), maximum_pages));
            if mark_slots > 0 {
                components.extend(buffer(format!("muse.marks.{gpu}"), location, marks * mark_slots / gpus as u64,
                    Lifetime::RESIDENT, Category::Prefix,
                    &format!("Muse exact sliding marks, {mark_slots} slots"), maximum_pages));
            }
            components.extend(buffer(format!("muse.lane_planes.{gpu}"), location, lane, Lifetime::RESIDENT,
                Category::Workspace, "Muse lane planes", maximum_pages));
            components.extend(buffer(format!("muse.step_tables.{gpu}"), location, tables, Lifetime::RESIDENT,
                Category::Workspace, "Muse step metadata", maximum_pages));
            if let Some(scratch) = &projections[gpu].scratch {
                let bytes = scratch.bytes().map_err(anyhow_refusal)? as u64;
                components.extend(buffer(format!("muse.projections.{gpu}"), location, bytes,
                    Lifetime::RESIDENT, Category::Workspace, "Muse K6 projection scratch", maximum_pages));
            }
            let native = &projections[gpu].native_programs;
            if native.fp8_scratch_bytes > 0 {
                let group = if native.resolution == crate::placement::planner::native_attention::NativeResolution::Assumed {
                    "Muse native FP8 projection scratch (assumed; program manifest absent)"
                } else { "Muse native FP8 projection scratch" };
                let owner = Allocation::buffer(format!("muse.native.scratch.{gpu}"), location,
                    native.fp8_scratch_bytes, Lifetime::RESIDENT, Category::Workspace, group);
                components.push(Component::allocation(owner, ByteCurve::fixed(native.fp8_scratch_bytes, maximum_pages),
                    false, maximum_pages));
            }
            if let Some(exchange) = exchange {
                components.extend(buffer(format!("muse.tp2_exchange.{gpu}"), location, exchange,
                    Lifetime::RESIDENT, Category::Transport, "Muse TP2 exchange slots", maximum_pages));
            }
            components.extend(buffer(format!("muse.rope.{gpu}"), location, rope, Lifetime::RESIDENT,
                Category::Workspace, "Muse RoPE tables", maximum_pages));
            // Full layers page BF16 K+V records: the pool's only slope, split
            // across a head-split pair like every other per-layer owner.
            let slope = record.checked_mul(PAGE_ROWS).ok_or(Refusal::Overflow("Muse full-layer KV"))?
                / gpus as u64;
            for layer in 0..cfg.layers {
                if cfg.attention[layer] != MuseAttention::Full { continue; }
                components.push(Component::allocation(
                    Allocation::buffer(format!("muse.kv.{gpu}.{layer}"), location, slope, Lifetime::TARGET,
                        Category::Kv, "full-layer KV (paged K+V records)"),
                    ByteCurve::linear(slope, slope, maximum_pages), false, maximum_pages));
            }
        }
        components.extend(buffer("muse.combine".into(), Location::coordinator(0), combine, Lifetime::RESIDENT,
            Category::Workspace, "Muse Spark combine tables", maximum_pages));
        components.extend(buffer("muse.head_logits".into(), Location::coordinator(head_gpu),
            (stored + trimmed) as u64, Lifetime::RESIDENT, Category::Workspace, "Muse head logits",
            maximum_pages));
        components.extend(buffer("muse.prefill_logits".into(), Location::coordinator(head_gpu),
            prefill_logits_bytes as u64, Lifetime::RESIDENT, Category::Workspace, "Muse prefill logits",
            maximum_pages));
        components.extend(buffer("muse.partial_planes".into(), Location::coordinator(0), planes,
            Lifetime::RESIDENT, Category::Transport, "Muse Spark partial reply planes", maximum_pages));
        // The drafter is not a fixed owner: it borrows the target head (its
        // weights, embedding and taps), so its home is a declared domain.
        for (index, demand) in fixed.iter().enumerate() {
            if demand.bytes == 0 { continue; }
            components.extend(weights(format!("muse.fixed.{index}"), Location::coordinator(demand.gpu),
                demand.bytes, demand.resident_weights.0, demand.category, &demand.group, maximum_pages)?);
        }
        let mut cases = vec![Case { when: vec![], value: components }];

        // Expert arenas per onboarding count. A pair splits every routed layer
        // across both GPUs (TP2 halves plus the partial combine); one RTX holds
        // the whole layer. The executor workspace is charged exactly when a
        // layer is, and the Spark side keeps the intermediate slices of every
        // layer not onboarded.
        let manifest = catalog.exl3().ok_or_else(|| invalid("routed experts need EXL3 headers"))?;
        let tp2_partial = tp2_partial_bytes(cfg, unit_rows) as u64;
        let base = if pair { [tp2_partial + local.workspace(); 2] }
            else { [tp2_partial / 2 + local.workspace(), 0] };
        let arena = |layer: usize, world: usize, rank: usize| -> Result<u64, Refusal> {
            let residency = manifest.residency(crate::V41Exl3Layer::Backbone(layer), world, rank)
                .map_err(anyhow_refusal)?;
            let (_, bytes) = residency.device_arena_layout()
                .map_err(|error| invalid(&format!("Muse expert arena: {error}")))?;
            Ok(bytes as u64)
        };
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: (0..cfg.layers).collect(),
            spark_ranks, prefill_lanes: LANES };
        let (endpoint, spark_rings) = spark_endpoint(prefill, cutoff, slots_usize).map_err(anyhow_refusal)?;
        let spark_bytes = budgets.iter().find(|budget| budget.locations.iter()
            .any(|location| matches!(location.device, Device::Spark(0))))
            .map_or(0, |budget| budget.physical_bytes);
        let spark_budget = if spark_ranks == 0 {
            SparkBudgetInventory { layers: working.spark_layers.clone(), resident: vec![], overhead: vec![], budget: 0 }
        } else {
            SparkBudgetInventory::exl3(catalog, &working, &[endpoint], spark_rings, spark_bytes,
                spark_reserve_bytes(spark_ranks, prefill.waves, prefill.unit_rows).map_err(anyhow_refusal)?)
                .map_err(anyhow_refusal)?
        };
        let mut peaks = vec![[0u64; 2]; cfg.layers + 1];
        let mut expert_cases = Vec::with_capacity(cfg.layers + 1);
        // Count zero is a case too: with every routed layer on the Sparks, the
        // slice ledger is the whole Spark charge and must bind their budgets.
        let mut push = Vec::with_capacity(cfg.layers + 1);
        let (dense_bound, dense_source) = dense_push_staging(core, gpus)?;
        let mut validation = vec![0u64; gpus];
        let manifest = catalog.exl3().ok_or_else(|| invalid("Muse push requires EXL3 headers"))?;
        for count in 0..=cfg.layers {
            if count > 0 {
                for (gpu, bytes) in validation.iter_mut().enumerate() {
                    let residency = manifest.residency(crate::V41Exl3Layer::Backbone(count - 1), gpus, gpu)
                        .map_err(anyhow_refusal)?;
                    *bytes = bytes.checked_add(residency.loads.iter()
                        .filter(|load| load.tensor.ends_with(".trellis")).count() as u64 * 4)
                        .ok_or(Refusal::Overflow("Muse push validation"))?;
                }
            }
            let mut sum = [0u64; 2];
            for layer in 0..count {
                for gpu in 0..gpus {
                    let bytes = if pair { arena(layer, 2, gpu)? } else { arena(layer, 1, 0)? };
                    sum[gpu] = sum[gpu].checked_add(bytes).ok_or(Refusal::Overflow("Muse expert arena"))?;
                }
            }
            let mut components = Vec::new();
            for gpu in 0..gpus {
                // The executor workspace opens with the first onboarded layer:
                // a zero-layer case carries no coordinator expert demand.
                let bytes = if count == 0 { 0 }
                    else { base[gpu].checked_add(sum[gpu]).ok_or(Refusal::Overflow("Muse expert arena"))? };
                peaks[count][gpu] = bytes;
                let group = if pair { format!("TP2 expert layer halves 0..{count}") }
                    else { format!("resident routed layers 0..{count}") };
                components.extend(weights(format!("muse.experts.{gpu}.{count}"),
                    Location::coordinator(gpu as u8), bytes, sum[gpu], Category::Experts, &group,
                    maximum_pages)?);
            }
            for rank in 0..spark_ranks {
                let location = Location { device: Device::Spark(rank as u8), domain: Domain::Managed };
                if let Some(costs) = spark_budget.resident.get(rank).and_then(|costs| costs.get(count..)) {
                    components.extend(buffer(format!("muse.spark.experts.{rank}.{count}"), location,
                        costs.iter().sum::<u64>(), Lifetime::RESIDENT, Category::Experts,
                        "routed experts (intermediate slices)", maximum_pages));
                }
                if count < cfg.layers {
                    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0)
                        .ok_or(Refusal::Overflow("Muse Spark reserve"))?;
                    let owners = spark_budget.overhead.get(rank).copied().unwrap_or(0)
                        .checked_sub(runtime).ok_or(Refusal::Overflow("Muse Spark owner decomposition"))?;
                    components.extend(buffer(format!("muse.spark.io.{rank}.{count}"), location,
                        owners, Lifetime::RESIDENT, Category::Workspace, "Spark I/O and execution owners", maximum_pages));
                    components.extend(buffer(format!("muse.spark.startup.{rank}.{count}"), location,
                        runtime, Lifetime::RESIDENT, Category::Runtime,
                        cuteafd_core::memory_layout::spark_runtime_reserve_group(), maximum_pages));
                    let resident = spark_budget.resident[rank][count..].iter().sum();
                    if resident != 0 {
                        components.push(Component {
                            activation: Activation::always(),
                            bytes: ByteCurve::fixed(resident, maximum_pages),
                            kind: ComponentKind::DriverPageTables {
                                model: DriverPageTableModel::Sm121Arena, location,
                                lifetime: Lifetime::RESIDENT,
                                backing: format!("muse.spark.experts.{rank}.{count}"),
                            },
                        });
                    }
                }
            }
            let envelope = crate::push_load::PushHostEnvelope::new(validation.clone(), vec![dense_bound; gpus],
                cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64,
                cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64)?;
            for owner in envelope.allocations("muse.push")? {
                if owner.id.ends_with(".dense") {
                    components.push(Component { activation: Activation::always(),
                        bytes: ByteCurve::fixed(owner.bytes, maximum_pages),
                        kind: ComponentKind::Allowance(Allowance { location: owner.location, bytes: owner.bytes,
                            lifetime: owner.lifetime, source: dense_source.clone(),
                            assumed: true }) });
                } else {
                    let bytes = ByteCurve::fixed(owner.bytes, maximum_pages);
                    components.push(Component::allocation(owner, bytes, false, maximum_pages));
                }
            }
            push.push(envelope);
            expert_cases.push(Case { when: vec![When::Onboard(count)], value: components });
        }

        // Bundled towers: one joint option per (vision, audio) candidate pair.
        let encoders = tower_options(&media, gpus, spark_ranks)?;
        let mut encoder_cases = Vec::with_capacity(encoders.len());
        for (option, (vision, audio)) in encoders.iter().enumerate() {
            let mut components = Vec::new();
            if vision.admitted_bytes() > 0 || audio.admitted_bytes() > 0 {
                components.extend(buffer(format!("muse.host.media.{option}"), Location::HOST,
                    host_quotas.media_bytes, Lifetime { first: Phase::Warmup, last: Phase::Growth },
                    Category::Embedding, "media/embedding-cache", maximum_pages));
            }
            for (name, placement) in [("vision", vision), ("audio", audio)] {
                let locations: Vec<Location> = match placement.kind {
                    crate::plan::encoder::EncoderKind::Off => Vec::new(),
                    crate::plan::encoder::EncoderKind::Rtx { gpu } => vec![Location::coordinator(gpu as u8)],
                    _ => placement.spark_ranks().into_iter()
                        .map(|rank| Location { device: Device::Spark(rank as u8), domain: Domain::Managed })
                        .collect(),
                };
                for (copy, location) in locations.into_iter().enumerate() {
                    // Residency is a coordinator ceiling; Spark towers are
                    // charged only inside their worker's managed arena.
                    let resident = if matches!(location.device, Device::Coordinator(_)) {
                        placement.weights
                    } else { 0 };
                    components.extend(weights(format!("muse.{name}.weights.{option}.{copy}"), location,
                        placement.weights, resident, Category::Weights, &format!("{name} tower"),
                        maximum_pages)?);
                    components.extend(buffer(format!("muse.{name}.scratch.{option}.{copy}"), location,
                        placement.scratch, Lifetime::RESIDENT, Category::Workspace,
                        &format!("{name} scratch"), maximum_pages));
                }
            }
            encoder_cases.push(Case { when: vec![When::Encoder(option)], value: components });
        }
        cases.extend(expert_cases);
        cases.extend(encoder_cases);
        let recipe = MuseRecipe { cfg, context, gpus, maximum_pages,
            prefill: PrefillShape { rows: prefill.unit_rows, waves: prefill.waves }, attention,
            spark_ranks, encoders, modes: (media.vision, media.audio), cases, peaks, local: local.clone(),
            drafter: drafter.clone() };
        Ok(PreparedMuse { recipe, budgets, prefill_logits_bytes, local, drafter, layouts, global, projections,
            spark_budget, host: host_quotas, push })
    }
}

struct Domains {
    onboard: usize,
    encoders: usize,
}

impl Domains {
    /// The activation of one component condition: every option satisfying all
    /// of its conditions, as a cartesian product of single-choice clauses.
    fn activation(&self, when: &[When], onboard: usize, encoder: usize) -> Result<Activation, Refusal> {
        let mut domains = std::collections::BTreeMap::<usize, std::collections::BTreeSet<usize>>::new();
        for condition in when {
            let (variable, options): (usize, Vec<usize>) = match condition {
                When::Onboard(count) => {
                    if *count > self.onboard {
                        return Err(invalid("onboarding condition above the routed layer count"));
                    }
                    (onboard, vec![*count])
                }
                When::Encoder(option) => {
                    if *option >= self.encoders { return Err(invalid("encoder condition outside the options")); }
                    (encoder, vec![*option])
                }
            };
            match domains.get_mut(&variable) {
                Some(existing) => {
                    existing.retain(|option| options.contains(option));
                    if existing.is_empty() { return Err(invalid("contradictory component conditions")); }
                }
                None => { domains.insert(variable, options.into_iter().collect()); }
            }
        }
        let mut clauses = vec![vec![]];
        for (variable, options) in domains {
            clauses = clauses.into_iter().flat_map(|clause| {
                options.iter().map(move |&option| {
                    let mut clause = clause.clone();
                    clause.push(Choice { variable, option });
                    clause
                })
            }).collect();
        }
        Ok(Activation { clauses })
    }
}

impl MuseRecipe<'_> {
    /// Place the selected assignment: layer modes, expert homes, residual hops
    /// and the Spark working set. Plan and runtime read this verbatim.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<(Placement, SparkWorkingSet), Refusal> {
        if solved.decisions.attention != self.attention {
            return Err(invalid("resolved layer modes differ from the contract"));
        }
        let local: Vec<usize> = solved.decisions.onboard_layers.iter().flatten().copied()
            .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let count = local.len();
        if count > self.cfg.layers || local.iter().copied().ne(0..count) {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let peak = *self.peaks.get(count).ok_or_else(|| invalid("resolved count outside the peaks"))?;
        let pair = self.gpus == 2;
        let assignments = (0..self.cfg.layers).map(|layer| LayerAssignment {
            mode: self.attention[layer],
            experts: if layer >= count { ExpertHome::Spark }
                else if pair { ExpertHome::RtxTp2 } else { ExpertHome::RtxWhole { gpu: 0 } },
        }).collect();
        let tp2 = (!local.is_empty() && pair)
            .then(|| Tp2Range { first: local[0], layers: count, peak_bytes: peak });
        let ranges = (0..self.gpus).map(|gpu| {
            if tp2.is_some() || local.is_empty() { ExpertRange { first: 0, layers: 0, peak_bytes: 0 } }
            else { ExpertRange { first: local[0], layers: count, peak_bytes: peak[gpu] } }
        }).collect();
        let hop_spec = HopSpec { storage: HopStorage::ReceiveSlots,
            row_bytes: (self.cfg.hidden as u64).checked_mul(2).ok_or(Refusal::Overflow("Muse hop rows"))?,
            rows: 0, lanes: 0, entry_gpu: 0, head_gpu: u8::try_from(self.gpus - 1).map_err(|_| Refusal::Overflow("Muse GPUs"))?,
            terminal: if pair { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } };
        let residual = residual_homes(&self.attention, &hop_spec)
            .map_err(|error| invalid(&error.to_string()))?;
        let hops = plan_hops(&self.attention, &hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let home = solved.decisions.draft_home;
        if home.is_some() != self.drafter.is_some()
            || home.is_some_and(|gpu| usize::from(gpu) + 1 != self.gpus) {
            return Err(invalid("resolved draft home differs from the declared head"));
        }
        let placement = Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Heads,
            attention_by_kind: vec![(AttentionClass::Gqa, AttentionPlacement::Heads),
                (AttentionClass::Swa, AttentionPlacement::Heads)],
            peer_row_bytes: 0, onboard_layers: count, max_pool_tokens_no_onboard: None,
            layers: assignments, movables: home.into_iter().map(|gpu| (crate::placement::MovableId::Drafter, gpu)).collect(), expert_ranges: ranges, residual, hops, tp2,
            items: (0..self.gpus).map(|gpu| solved.items(Location::coordinator(gpu as u8)))
                .collect::<Result<_, _>>()? };
        MUSE_SPARK.check(&placement).map_err(|error| invalid(&error.to_string()))?;
        self.local.check(&placement).map_err(anyhow_refusal)?;
        let working = SparkWorkingSet { rtx_layers: local,
            spark_layers: (count..self.cfg.layers).collect(),
            spark_ranks: if count == self.cfg.layers { 0 } else { self.spark_ranks },
            prefill_lanes: LANES };
        Ok((placement, working))
    }

    /// The selected towers. An automatic tower the shared solution disabled
    /// carries the solution's own diagnostic, in the shared wording.
    fn media(&self, solved: &ResolvedPlan) -> Result<MediaAdmission, Refusal> {
        let option = solved.decisions.encoder.ok_or_else(|| invalid("missing admitted encoder"))?;
        let (vision, audio) = self.encoders.get(option)
            .ok_or_else(|| invalid("admitted encoder outside the prepared options"))?.clone();
        let mut spark_items = vec![Vec::new(); self.spark_ranks];
        for (name, placement) in [("vision", &vision), ("audio", &audio)] {
            let items = [Item::new(Category::Weights, format!("{name} tower"), "BF16 native",
                placement.weights, Basis::Exact),
                Item::new(Category::Workspace, format!("{name} scratch"), "resident", placement.scratch,
                    Basis::Exact)];
            for rank in placement.spark_ranks() {
                let rank_items = spark_items.get_mut(rank)
                    .ok_or_else(|| invalid("tower rank outside the Spark worker set"))?;
                rank_items.extend(items.iter().cloned());
            }
        }
        let disabled = |placement: &mut EncoderPlacement, mode: MediaMode| {
            use crate::plan::encoder::EncoderKind;
            if mode != MediaMode::Auto || placement.kind != EncoderKind::Off || placement.weights > 0 {
                return;
            }
            placement.reason = format!("auto native tower disabled by shared admission: {}",
                solved.encoder_diagnostics.iter().map(|(reason, _)| reason.as_str())
                    .collect::<Vec<_>>().join("; "));
            placement.shortfall = solved.encoder_diagnostics.iter().map(|(_, bytes)| *bytes)
                .filter(|bytes| *bytes > 0).min().unwrap_or(0);
        };
        let mut vision = vision;
        let mut audio = audio;
        disabled(&mut vision, self.modes.0);
        disabled(&mut audio, self.modes.1);
        Ok(MediaAdmission { vision, audio, spark_items })
    }
}

impl Recipe for MuseRecipe<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if let (LocalPackage::Missing(package), Pin::Exact(n)) = (&self.local, request.onboard) {
            if n > 0 {
                return Err(invalid(&format!("{}; an explicit {n} RTX expert layers was requested",
                    LocalPackage::refusal(package))));
            }
        }
        if !(1..=2).contains(&self.gpus)
            || self.cfg.layers == 0
            || self.context == 0
            || request.context.maximum != self.context
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || self.prefill.rows == 0
            || self.prefill.waves == 0
            || self.attention.len() != self.cfg.layers
            || self.spark_ranks > 8
            || self.encoders.is_empty()
            || self.peaks.len() != self.cfg.layers + 1
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        if self.attention.iter().any(|mode| match mode {
            LayerMode::HeadSplit => self.gpus != 2,
            LayerMode::Whole { gpu, .. } => usize::from(*gpu) >= self.gpus,
            _ => true,
        }) {
            return Err(invalid("layer mode outside the Muse executor"));
        }
        let mut contract = Contract {
            floor_fallback: None,
            resident_weight_limit: request.resident_weight_limit,
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: Some(SparkReplyFormats {
                prefill: Some(request.spark_reply_formats.prefill(
                    cuteafd_core::ExpertGeometry::MUSE_SPARK.partial_replies())),
                decode: Some(request.spark_reply_formats.decode()),
            }),
            page_rows: PAGE_ROWS,
            pool_page_multiple: 1,
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: self.attention.iter().enumerate()
                .map(|(layer, &mode)| Variable::Attention { layer, values: vec![mode] })
                .collect(),
            requires: vec![],
            components: vec![],
        };
        contract.variables.push(Variable::Prefill(vec![self.prefill]));
        contract.variables.push(Variable::Lanes(vec![LANES]));
        let onboard = contract.variables.len();
        contract.variables.push(Variable::Onboard {
            layers: (0..self.cfg.layers).collect(),
            destination: (0..self.gpus as u8).collect(),
            counts: (0..=self.cfg.layers).collect(),
        });
        if self.spark_ranks == 0 {
            for option in 0..self.cfg.layers {
                contract.requires.push(Requires { when: Choice { variable: onboard, option }, any_of: vec![] });
            }
        }
        let encoder = contract.variables.len();
        contract.variables.push(Variable::Encoder((0..self.encoders.len()).collect()));
        // The DFlash2 drafter reads the target head, its embedding and its
        // taps, so the head GPU is its only home; declaring the domain keeps
        // the owner charged and named by the shared solve.
        if let Some(drafter) = &self.drafter {
            let home = u8::try_from(self.gpus - 1).map_err(|_| Refusal::Overflow("Muse GPUs"))?;
            let variable = contract.variables.len();
            contract.variables.push(Variable::DraftHome(vec![home]));
            let activation = Activation::choice(variable, 0);
            let lifetime = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            let ranks = if drafter.tp2.is_empty() { 1 } else { 2 };
            if drafter.arena_requests.len() != ranks {
                return Err(invalid("Muse drafter arena recipes differ from its rank count"));
            }
            for rank in 0..ranks {
                let gpu = if ranks == 1 { home } else { (1 - rank) as u8 };
                let arena = drafter.arena(rank, gpu).map_err(anyhow_refusal)?;
                for mut component in weight_arena_components(&arena, Location::coordinator(gpu),
                    lifetime, self.maximum_pages, cuteafd_core::memory_layout::MemoryKind::DrafterWeights)? {
                    component.activation = activation.clone();
                    contract.components.push(component);
                }
            }
            if !drafter.tp2.is_empty() {
                if self.gpus != 2 || drafter.tp2.len() != 2 {
                    return Err(invalid("Muse TP2 drafter needs two rank-local layouts"));
                }
                for (rank, layout) in drafter.tp2.iter().enumerate() {
                    let location = Location::coordinator((1 - rank) as u8);
                    let prefix = format!("muse.drafter.rank{rank}");
                    let weights = layout.weight_owners().map(|owner| format!("{prefix}.{}", owner.id))
                        .collect::<std::collections::BTreeSet<_>>();
                    for owner in layout.allocations(&prefix, location).into_iter()
                        .filter(|owner| !weights.contains(&owner.id)) {
                        let bytes = ByteCurve::fixed(owner.bytes, self.maximum_pages);
                        let mut component = Component::allocation(owner, bytes, false, self.maximum_pages);
                        component.activation = activation.clone();
                        contract.components.push(component);
                    }
                }
            }
            let owned = if drafter.tp2.is_empty() {
                drafter.owned.checked_sub(drafter.weights.0).and_then(|bytes| bytes.checked_add(drafter.scratch))
                    .ok_or(Refusal::Overflow("Muse drafter nonweight storage"))?
            } else { 0 };
            let bytes = [owned, drafter.head_scratch, drafter.runtime].into_iter()
                .try_fold(0u64, |sum, bytes| sum.checked_add(bytes)).ok_or(Refusal::Overflow("Muse drafter"))?;
            if let Some(mut component) = buffer("muse.drafter".into(), Location::coordinator(home), bytes,
                Lifetime::RESIDENT, Category::Drafter, "Drafter", self.maximum_pages) {
                component.activation = activation;
                contract.components.push(component);
            }
        }
        // A Spark tower keeps a retained expert worker: a full onboarding has
        // no Spark layer left to run it.
        for (option, (vision, audio)) in self.encoders.iter().enumerate() {
            if vision.spark_ranks().is_empty() && audio.spark_ranks().is_empty() { continue; }
            contract.requires.push(Requires { when: Choice { variable: encoder, option },
                any_of: (0..self.cfg.layers).map(|count| Choice { variable: onboard, option: count }).collect() });
        }
        let mut blas = Vec::new();
        for (option, (vision, audio)) in self.encoders.iter().enumerate() {
            for (name, placement) in [("vision", vision), ("audio", audio)] {
                let locations = match placement.kind {
                    crate::plan::encoder::EncoderKind::Off => vec![],
                    crate::plan::encoder::EncoderKind::Rtx { gpu } => vec![Location::coordinator(gpu as u8)],
                    _ => placement.spark_ranks().into_iter().map(|rank| Location {
                        device: Device::Spark(rank as u8), domain: Domain::Managed }).collect(),
                };
                for location in locations {
                    blas.push(serving::BlasOwners { name: format!("Muse {name} {option}/{:?}", location.device),
                        process: match location.device { Device::Spark(rank) => format!("muse-media-worker-{rank}"),
                            _ => "coordinator".into() }, thread: "muse/blas/media".into(), location,
                        gemm: true, lt: false, lifetime: Lifetime::RESIDENT,
                        activation: Activation::choice(encoder, option) });
                }
            }
        }
        for arch in ["sm_120", "sm_121"] {
            let owners = blas.iter().filter(|owner| matches!(owner.location.device, Device::Coordinator(_))
                == (arch == "sm_120")).cloned().collect::<Vec<_>>();
            contract.components.extend(serving::blas_components(arch, &owners,
                self.maximum_pages, contract.phase_order)?);
        }
        let domains = Domains { onboard: self.cfg.layers, encoders: self.encoders.len() };
        for case in &self.cases {
            let activation = domains.activation(&case.when, onboard, encoder)?;
            for mut component in case.value.iter().cloned() {
                if component.activation != Activation::always() {
                    return Err(invalid("subrecipe supplied a private activation"));
                }
                component.activation = activation.clone();
                contract.components.push(component);
            }
        }
        Ok(contract)
    }
}

/// The linked expert path remains required until local-only startup is qualified.
pub const LOCAL_ONLY_REFUSAL: Refusal = Refusal::MissingExecutor {
    family: "Muse Spark", executor: "coordinator-only routed experts", issue: 1395,
};

impl PreparedMuse<'_> {
    /// The single admission solve. Planner and serving call this one verb.
    pub fn admit(self, request: Request) -> Result<AdmittedMuse, Refusal> {
        let PreparedMuse { recipe, budgets, prefill_logits_bytes, local, drafter, layouts, global, projections,
            spark_budget, mut host, push } = self;
        let started = std::time::Instant::now();
        let solved = solve(request, budgets, &recipe)?;
        let solve_us = started.elapsed().as_micros().try_into().unwrap_or(u64::MAX);
        let (placement, spark) = recipe.execution(&solved)?;
        if spark.spark_layers.is_empty() { return Err(LOCAL_ONLY_REFUSAL); }
        let push = push.get(placement.onboard_layers).ok_or_else(|| invalid("push onboarding extent"))?.clone();
        let media = recipe.media(&solved)?;
        if media.vision.admitted_bytes() == 0 && media.audio.admitted_bytes() == 0 { host.media_bytes = 0; }
        Ok(AdmittedMuse { placement, spark, draft_home: solved.decisions.draft_home, solved, media, local,
            drafter, prefill_logits_bytes, layouts, global, projections, spark_budget, solve_us, host, push })
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    #[test]
    fn blas_joint_towers_share_only_their_selected_process_thread() {
        use crate::plan::encoder::EncoderKind;
        let cfg = MuseSparkConfig::from_hf(&crate::plan::testing::muse_spark_config()).unwrap();
        let tower = |kind| EncoderPlacement { kind, weights: 0, scratch: 0, replicas: vec![],
            reason: "test".into(), shortfall: 0 };
        let prefill = PrefillShape { rows: 64, waves: 1 };
        let recipe = MuseRecipe { cfg: &cfg, context: 64, gpus: 2, maximum_pages: 1, prefill,
            attention: vec![LayerMode::HeadSplit; cfg.layers], spark_ranks: 2,
            encoders: vec![(tower(EncoderKind::Off), tower(EncoderKind::Off)),
                (tower(EncoderKind::Rtx { gpu: 0 }), tower(EncoderKind::Rtx { gpu: 0 })),
                (tower(EncoderKind::Rtx { gpu: 0 }), tower(EncoderKind::Rtx { gpu: 1 })),
                (tower(EncoderKind::Spark { rank: 1 }), tower(EncoderKind::Spark { rank: 1 }))],
            modes: (MediaMode::Auto, MediaMode::Auto), cases: vec![], peaks: vec![[0; 2]; cfg.layers + 1],
            local: LocalPackage::Unknown, drafter: None };
        let request = production_request(64, None, Onboard::Auto, prefill, Operation::Serving, None,
            SparkReplyFormats::default());
        let contract = recipe.prepare(&request).unwrap();
        let encoder = contract.variables.iter().position(|value| matches!(value, Variable::Encoder(_))).unwrap();
        for option in 0..4 {
            let mut choices = vec![0; contract.variables.len()];
            choices[encoder] = option;
            let facts = contract.facts(&choices, 1).unwrap();
            let handles = |location| facts.allowances.iter().filter(|owner|
                owner.location == location && owner.source.starts_with("cuBLAS handles:")
                    && owner.lifetime.contains(Phase::Ready, contract.phase_order))
                .map(|owner| owner.bytes).sum::<u64>();
            assert_eq!(handles(Location::coordinator(0)), u64::from(option == 1 || option == 2) * (64 << 20));
            assert_eq!(handles(Location::coordinator(1)), u64::from(option == 2) * (64 << 20));
            assert_eq!(handles(Location { device: Device::Spark(1), domain: Domain::Managed }),
                u64::from(option == 3) * 67_508_224);
        }
    }

    #[test]
    fn drafter_charge_table_matches_both_bf16_recipes() {
        use crate::families::glm5::{draft_physical::{GlmDflashConfig, GlmDflashPhysicalLayout,
            gqa_arena_requests}, draft_representation::GlmDraftCapacity};
        let target = MuseSparkConfig::from_hf(&crate::plan::testing::muse_spark_config()).unwrap();
        let configs = [
            (include_str!("../../../../../cuteafd-daemon/src/families/muse_spark/fixtures/drafters/nvidia.json"),
                8_768_324_864u64, [4_993_452_544u64, 4_782_361_088]),
            (include_str!("../../../../../cuteafd-daemon/src/families/muse_spark/fixtures/drafters/radixark.json"),
                9_774_959_104, [5_496_769_024, 5_285_677_568]),
        ];
        for (raw, old_single, old_pair) in configs {
            let config = serde_json::from_str(raw).unwrap();
            let cfg = GlmDflashConfig::from_value(&config).unwrap();
            let draft = super::super::drafter_inventory(&target, &config, 16, 16, 202_112, 188,
                crate::families::qwen4::Qwen4KvCache::Fp8).unwrap();
            assert_eq!(draft.weights.0, old_single);
            let arena = draft.arena(0, 0).unwrap();
            let floor_delta = if cfg.head_dim == 64 { 1_280 } else { 0 };
            assert_eq!(arena.bytes() as u64, old_single + floor_delta);
            assert_eq!((arena.bytes() as u64).div_ceil(2 << 20) * (2 << 20)
                - arena.bytes() as u64, 1_963_520);
            let capacity = GlmDraftCapacity::new(16, 16, cfg.block).unwrap();
            for (rank, old) in old_pair.into_iter().enumerate() {
                let physical = GlmDflashPhysicalLayout::new(&cfg.for_tp(2).unwrap(),
                    draft.representation, capacity, 188, 2,
                    crate::families::qwen4::Qwen4KvCache::Fp8).unwrap();
                let payload = physical.weight_owners().filter(|owner| rank == 0 || !owner.id.starts_with("head."))
                    .map(|owner| owner.bytes).sum::<u64>();
                assert_eq!(payload, old);
                let arena = crate::serving_capacity::WeightArenaLayout::new("draft", gqa_arena_requests(
                    &cfg, draft.representation, rank, 2).unwrap()).unwrap();
                assert_eq!(arena.bytes() as u64, old);
                let tail = (arena.bytes() as u64).div_ceil(2 << 20) * (2 << 20) - arena.bytes() as u64;
                assert_eq!(tail, if rank == 0 { 1_963_520 } else { 1_242_624 });
            }
        }
    }

    #[test]
    fn drafter_arena_contract_replaces_only_weights_and_preserves_chronology() {
        use crate::families::glm5::{draft_physical::{GlmDflashConfig, GlmDflashPhysicalLayout,
            gqa_arena_requests}, draft_representation::GlmDraftCapacity};
        let cfg = MuseSparkConfig::from_hf(&crate::plan::testing::muse_spark_config()).unwrap();
        let config = serde_json::from_str(include_str!(
            "../../../../../cuteafd-daemon/src/families/muse_spark/fixtures/drafters/nvidia.json")).unwrap();
        let draft_cfg = GlmDflashConfig::from_value(&config).unwrap();
        let base = super::super::drafter_inventory(&cfg, &config, 16, 16, 202_112, 188,
            crate::families::qwen4::Qwen4KvCache::Fp8).unwrap();
        for ranks in [1, 2] {
            let mut draft = base.clone();
            if ranks == 2 {
                let capacity = GlmDraftCapacity::new(16, 16, draft.block).unwrap();
                for rank in 0..2 {
                    let physical = GlmDflashPhysicalLayout::new(&draft_cfg.for_tp(2).unwrap(),
                        draft.representation, capacity, 188, 2,
                        crate::families::qwen4::Qwen4KvCache::Fp8).unwrap();
                    // Synthetic rank owners exercise the immutable/nonweight filter;
                    // the installed fixture covers read_tp's real peer exclusions.
                    draft.tp2.push(physical);
                    if rank == 0 { draft.arena_requests.clear(); }
                    draft.arena_requests.push(gqa_arena_requests(&draft_cfg, draft.representation, rank, 2).unwrap());
                }
            }
            let off = EncoderPlacement { kind: crate::plan::encoder::EncoderKind::Off,
                weights: 0, scratch: 0, replicas: vec![], reason: String::new(), shortfall: 0 };
            let prefill = PrefillShape { rows: 64, waves: 1 };
            let recipe = MuseRecipe { cfg: &cfg, context: 64, gpus: 2, maximum_pages: 1, prefill,
                attention: vec![LayerMode::HeadSplit; cfg.layers], spark_ranks: 6,
                encoders: vec![(off.clone(), off)], modes: (MediaMode::Off, MediaMode::Off),
                cases: vec![], peaks: vec![[0; 2]; cfg.layers + 1], local: LocalPackage::Unknown,
                drafter: Some(draft.clone()) };
            let request = production_request(64, None, Onboard::Auto, prefill, Operation::Serving, None,
                SparkReplyFormats::default());
            let contract = recipe.prepare(&request).unwrap();
            assert_eq!(contract.phase_order, PhaseOrder::TARGET_THEN_DRAFT);
            let variable = contract.variables.iter().position(|value| matches!(value, Variable::DraftHome(_))).unwrap();
            for component in &contract.components {
                assert_eq!(component.activation, Activation::choice(variable, 0));
            }
            let selected = vec![0; contract.variables.len()];
            let facts = contract.facts(&selected, 1).unwrap();
            assert_eq!(facts.resident_weights.len(), ranks);
            assert_eq!(facts.allocation_granularity.len(), ranks);
            for rank in 0..ranks {
                let gpu = if ranks == 1 { 1 } else { (1 - rank) as u8 };
                let arena = draft.arena(rank, gpu).unwrap();
                let root = facts.allocations.iter().find(|owner| owner.id == arena.id()).unwrap();
                assert_eq!((root.location, root.bytes, root.lifetime), (Location::coordinator(gpu),
                    arena.bytes() as u64, Lifetime { first: Phase::DraftLoad, last: Phase::Growth }));
                let tail = facts.allocation_granularity.iter().find(|owner| owner.id == format!("{}.tail", root.id)).unwrap();
                assert_eq!(tail.lifetime, root.lifetime);
                assert_eq!(tail.backings, [GranularityBacking { backing: root.id.clone(), requests: vec![root.bytes] }]);
                if ranks == 2 {
                    let layout = &draft.tp2[rank];
                    let weights = layout.weight_owners().map(|owner| owner.id.as_str()).collect::<std::collections::BTreeSet<_>>();
                    for owner in layout.owners() {
                        let id = format!("muse.drafter.rank{rank}.{}", owner.id);
                        assert_eq!(facts.allocations.iter().any(|allocation| allocation.id == id), !weights.contains(owner.id.as_str()));
                    }
                }
            }
            let nonweights = facts.allocations.iter().find(|owner| owner.id == "muse.drafter").unwrap();
            assert_eq!(nonweights.bytes, draft.head_scratch + draft.runtime + if ranks == 1 {
                draft.owned - draft.weights.0 + draft.scratch
            } else { 0 });
            assert_eq!(nonweights.lifetime, Lifetime::RESIDENT);
        }
    }

    #[test]
    fn capped_sample_keeps_physical_occupied_bytes() {
        let budget = measured_budget(0, 96 << 30, 80 << 30, 2 << 30, 92 << 30);
        assert_eq!(budget.available().unwrap(), 74 << 30);
        let planned = planned_budget(0, 96 << 30, 80 << 30, 2 << 30, 4 << 30);
        assert_eq!(planned.available().unwrap(), budget.available().unwrap());
        assert!(measured_budget(0, 96 << 30, 2 << 30, 0, 92 << 30).available().is_err());
    }

    #[test]
    fn host_prefix_quota_matches_runtime_chunk_geometry() {
        let cfg = MuseSparkConfig::from_hf(&crate::plan::testing::muse_spark_config()).unwrap();
        let mut input = HostInput { total: 256 << 30, available: 256 << 30, reserve: 8 << 30,
            prefix_bytes: None, prefix_entries: 20, media_bytes: Some(123), device_sampling: true };
        let quota = input.quotas(&cfg, 1 << 20, 16).unwrap();
        let chunk = (256u64 << 20).max(mark_bytes(&cfg, 1));
        assert_eq!(quota.prefix_bytes, (32u64 << 30) / chunk * chunk);
        assert_eq!(quota.prefix_chunk, chunk);
        assert_eq!(quota.media_bytes, 123);
        input.prefix_bytes = Some(1234);
        assert_eq!(input.quotas(&cfg, 1 << 20, 16).unwrap().prefix_chunk, 1234);
        input.prefix_entries = 0;
        assert_eq!(input.quotas(&cfg, 1 << 20, 16).unwrap().prefix_bytes, 0);
        input.prefix_entries = 20;
        input.prefix_bytes = None;
        input.total = 1;
        assert!(input.quotas(&cfg, 1 << 20, 16).is_err());
    }

    #[test]
    fn native_push_staging_is_load_only_host_demand_and_compact_keeps_its_bound() {
        use crate::formats::{native_attention::NativeProjection, trellis_core::CoreSplit};
        let mut core = CoreInventory::default();
        for gpus in [1, 2] {
            let (bytes, source) = dense_push_staging(&core, gpus).unwrap();
            assert_eq!(bytes, if gpus == 2 { 2 << 30 } else { 1 << 30 });
            assert_eq!(source, format!("Muse dense push staging: estimated {} GiB/rank from 1191-muse-min-pair / muse-max-push-incast8 for_steps bounds (#1274)", bytes >> 30));
        }
        for fp8 in [false, true] {
            core.native.insert("o_proj".into(), NativeProjection { shape: [8192, 8192], fp8, split: CoreSplit::Input });
            for gpus in [1, 2] {
                let (bytes, _) = dense_push_staging(&core, gpus).unwrap();
                assert_eq!(bytes, 17_716_740_096);
                let envelope = crate::push_load::PushHostEnvelope::new(vec![0; gpus], vec![bytes; gpus],
                    cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64,
                    cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64).unwrap();
                let host = HostInput { total: 256 << 30, available: 256 << 30, reserve: 0,
                    prefix_bytes: Some(0), prefix_entries: 0, media_bytes: Some(0), device_sampling: true }.budget();
                for owner in envelope.allocations("muse.push").unwrap() {
                    assert_eq!(owner.location, Location::PINNED);
                    assert_eq!(owner.lifetime, Lifetime { first: Phase::Load, last: Phase::Load });
                    assert!(host.locations.contains(&owner.location));
                    for gpu in 0..gpus {
                        assert!(!planned_budget(gpu as u8, 96 << 30, 80 << 30, 0, 0).locations.contains(&owner.location));
                    }
                }
            }
        }
        core.native.get_mut("o_proj").unwrap().shape = [usize::MAX; 2];
        assert!(matches!(dense_push_staging(&core, 2), Err(Refusal::Overflow(_))));
    }

    #[test]
    fn host_and_pinned_share_one_sample_and_headroom() {
        let host = HostInput { total: 128 << 30, available: 100 << 30, reserve: 8 << 30,
            prefix_bytes: Some(0), prefix_entries: 20, media_bytes: Some(0), device_sampling: true };
        let budget = host.budget();
        assert_eq!(budget.locations, vec![Location::HOST, Location::PINNED]);
        assert_eq!(budget.available().unwrap(), (100 << 30) - (128 << 30) / 10);
    }
}
