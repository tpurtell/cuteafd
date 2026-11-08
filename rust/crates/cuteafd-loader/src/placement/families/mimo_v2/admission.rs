//! MiMo supplies physical facts; the shared planner admits one typed request.
//!
//! Every owner the engine allocates is declared here with the phase it exists
//! in: steady reservations, KV records and page tables, the local expert arena
//! and its retained weights, and the TP2 route exchange. Plan and serve build
//! the same `MimoRecipe`; only the sampled baseline differs. Execution later
//! consumes the admitted owners and never re-samples memory or re-solves.
use crate::families::mimo_v2::{MimoAttention, MimoKvFormats, MimoV2Config, MimoExpertInput, MimoPrefillOutput};
use crate::families::mimo_v2::capacity::MimoCapacityProfiles;
use crate::placement::planner::{
    Allocation, Baseline, Budget, ByteCurve, Component, Contract, EmbeddingHome, ExpertDestination,
    Lifetime, Location, Operation, Phase, PhaseOrder, Pin, PrefillShape, Recipe, Refusal, Request,
    ResolvedPlan, SM120_ALLOCATION_GRANULARITY_BYTES, Variable,
};
use crate::placement::{
    plan_hops, residual_homes, AttentionClass, AttentionPlacement, ExpertCost, ExpertHome,
    ExpertRange, FfnMode, HopSpec, HopStorage, LayerAssignment, LayerMode, Placement,
    PlacementError, PoolLimits, ResidentWeights, SparkWorkingSet, TerminalHome, Tp2Range,
};
use crate::serving_capacity::{mimo_cache_geometry, CacheGeometryError};
use cuteafd_core::memory_layout::Category;
use cuteafd_core::serving_capacity::{
    CapacityProfile, MemoryReservation, ResolvedCapacity, ResolvedDeviceCapacity,
};

/// Decode rows a hop or route exchange carries at most.
pub const DECODE_ROWS: u64 = 64;

/// Measured post-module baseline for MiMo V2.6 Flash min on SM120, observer
/// subtracted: 1,179,189,248 - 576,716,800 = 602,472,448 B, rounded up to MiB.
/// Source: 389a8037e, drop-occupancy-ceiling-mimo-smoke, 2026-11-06.
/// Other shapes retain their own forecast; this is not a cross-family margin.
pub const FLASH_MIN_BASELINE_BYTES: u64 = 575 << 20;

/// Peak CUDA streams live in one coordinator context after Ready. Enumerated
/// from their creators: the engine's serving stream
/// (`families/mimo_v2/mod.rs`), the prefix `PrefixCopies` rank stream
/// (`shared/prefix/copy_stream.rs`), the prefix cache's Store and Restore
/// `StreamState`s, each with its nonblocking probe stream
/// (`shared/prefix/cuda_copy.rs`, 4 per registered device), and the Spark
/// intake copy stream (`shared/spark_intake.rs`). Load, push and warm streams
/// have retired before Ready. A TP2 peer exchange adds one stream
/// (`families/mimo_v2/mod.rs`), still inside the same four-stream slab band.
pub fn startup_peak_live_streams() -> u64 {
    1 + 1 + 4 + 1
}

fn invalid(detail: impl std::fmt::Display) -> Refusal {
    Refusal::InvalidFact(format!("MiMo component contract: {detail}"))
}

/// The largest pool tokens this build may index: the automatic target or the
/// explicit request, rounded up to the profile's pool quantum.
pub fn maximum_pool_tokens(cards: &[u64], context: u64, requested: Option<u64>, page_rows: u64) -> u64 {
    let page = page_rows.max(1);
    crate::placement::PoolPolicy::resolve(cards, context, requested, page)
        .maximum_tokens()
        .div_ceil(page)
        .saturating_mul(page)
}

/// Physical budgets for the coordinator GPUs plus one inert per-Spark budget,
/// so every declared expert home has a physical owner.
pub fn production_budgets(cards: &[(u64, u64)], headrooms: &[u64], baselines: Vec<Baseline>,
    host: (u64, u64), spark_bytes: u64, spark_ranks: usize) -> Result<Vec<Budget>, Refusal> {
    if !(1..=2).contains(&cards.len())
        || headrooms.len() != cards.len()
        || baselines.len() != cards.len()
        || host.1 > host.0
        || host.0 == 0
        || spark_ranks > 8
    {
        return Err(invalid("invalid production physical capacities"));
    }
    let mut budgets = cards.iter().zip(headrooms).zip(baselines).enumerate().map(
        |(rank, ((&(physical, logical), &headroom), baseline))| Budget {
            id: format!("rtx{rank}"),
            locations: vec![Location::coordinator(rank as u8)],
            physical_bytes: physical,
            logical_bytes: logical,
            baseline,
            headroom,
        },
    ).collect::<Vec<_>>();
    budgets.push(Budget {
        id: "coordinator host RAM".into(),
        locations: vec![Location::HOST, Location::PINNED],
        physical_bytes: host.0,
        logical_bytes: host.0,
        baseline: Baseline::Sampled { free: host.1, phase: Phase::Load,
            source: "host MemAvailable normalized to external occupancy before whole-process recipe".into() },
        headroom: 0,
    });
    budgets.extend((0..spark_ranks).map(|rank| Budget {
        id: format!("spark{rank}"),
        locations: vec![Location { device: crate::placement::planner::Device::Spark(rank as u8),
            domain: crate::placement::planner::Domain::Managed }],
        physical_bytes: spark_bytes,
        logical_bytes: spark_bytes,
        baseline: Baseline::Forecast { occupied: 0,
            source: "Spark worker budget; MiMo Spark owners are admitted by the Spark leg".into(),
            assumed: false },
        headroom: 0,
    }));
    Ok(budgets)
}

/// Rank reserve kept free: the memory policy's named reserve, and on the head
/// GPU at least the drained Spark intake startup probe (released before serving).
/// Release padding stays additive outside that maximum; `GPU_RESERVE` replaces both.
pub fn device_headroom(rank: usize, startup_probe_bytes: u64) -> u64 {
    let probe = if rank == 0 { crate::placement::reserve::gpu_component(startup_probe_bytes) } else { 0 };
    let padding = cuteafd_core::memory_policy::MemoryPolicy::current().release_padding();
    crate::placement::reserve::gpu("mimo_v2").saturating_sub(padding).max(probe).saturating_add(padding)
}

pub fn production_request(context: impl Into<crate::placement::planner::Context>, pool: Option<u64>, onboard: crate::placement::Onboard,
    prefill: PrefillShape, operation: Operation, resident_weight_limit: ResidentWeights,
    spark_reply_formats: cuteafd_core::spark_reply::SparkReplyFormats) -> Request {
    Request {
        context: context.into(),
        pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard {
            crate::placement::Onboard::Auto => Pin::Auto,
            crate::placement::Onboard::Layers(n) => Pin::Exact(n as u64),
        },
        prefill: Pin::Exact(prefill),
        operation,
        resident_weight_limit: Some(resident_weight_limit),
        spark_reply_formats,
    }
}

/// Future host allocations, resolved before the shared admission (not a second ledger).
#[derive(Clone, Copy)]
pub struct HostOwners {
    pub expert_input: MimoExpertInput,
    pub output: MimoPrefillOutput,
    pub serving: bool,
    pub scoring_rows: usize,
    pub diagnostic_rows: usize,
    pub prefix_bytes: u64,
    pub media_bytes: u64,
    pub headroom: u64,
}

impl HostOwners {
    pub(super) fn demands(&self, recipe: &MimoRecipe<'_>, pages: u64, count: usize) -> Result<Vec<(String, u64)>, Refusal> {
        let mut owners = crate::families::mimo_v2::admission::future_host_workspaces(recipe.cfg,
            recipe.prefill.rows as usize, recipe.lanes as usize, recipe.gpus,
            pages.checked_mul(recipe.page_rows).ok_or(Refusal::Overflow("MiMo host pool"))?,
            recipe.context as usize, if count < recipe.routed.len() { recipe.spark_ranks } else { 0 },
            self.expert_input, recipe.kv, self.output, self.serving, self.diagnostic_rows,
            if recipe.gpus == 2 { count } else { 0 }).map_err(invalid)?;
        if recipe.embedding == EmbeddingHome::HostPinned {
            owners.push(("mimo/embedding".into(), (recipe.cfg.vocab_size as u64)
                .checked_mul(recipe.cfg.hidden as u64).and_then(|n| n.checked_mul(2))
                .ok_or(Refusal::Overflow("MiMo host embedding"))?));
        }
        let scoring = cuteafd_core::memory_layout::host_logits_bytes(self.scoring_rows as u64,
            self.scoring_rows.max(64) as u64, recipe.cfg.vocab_size as u64)
            .ok_or(Refusal::Overflow("MiMo host scoring"))?;
        owners.extend([("mimo/scoring".into(), scoring), ("mimo/prefix".into(), self.prefix_bytes),
            ("mimo/media-cache".into(), self.media_bytes)]);
        Ok(owners)
    }

    pub(super) fn components(&self, recipe: &MimoRecipe<'_>, count: usize) -> Result<Vec<Component>, Refusal> {
        let maximum = recipe.maximum_pages;
        // Ordered host tables use max(256, pages * rows * 4). The sync buffer
        // uses max(64 MiB, largest table); these are their only slope changes.
        let table_slope = 4 * recipe.lanes.max(DECODE_ROWS);
        let mut starts = vec![1, 65, (64u64 << 20) / table_slope + 1];
        starts.retain(|&n| n <= maximum);
        starts.sort_unstable();
        starts.dedup();
        starts.push(maximum + 1);
        let first = self.demands(recipe, 1, count)?;
        let mut curves = vec![Vec::new(); first.len()];
        for interval in starts.windows(2) {
            let (start, last) = (interval[0], interval[1] - 1);
            let a = self.demands(recipe, start, count)?;
            let b = self.demands(recipe, last, count)?;
            for (index, ((_, a), (_, b))) in a.iter().zip(&b).enumerate() {
                curves[index].push(crate::placement::planner::BytePiece { granularity: 1, first: start, last,
                    bytes_at_first: *a, bytes_per_page: if last == start { 0 } else { b.saturating_sub(*a).div_ceil(last - start) }, backings: 1, });
            }
        }
        Ok(first.into_iter().zip(curves).map(|((name, bytes), pieces)| {
            let category = if name == "mimo/embedding" { Category::Embedding }
                else if name.starts_with("mimo/transport/") { Category::Transport }
                else if name == "mimo/prefix" { Category::Prefix } else { Category::Staging };
            let owner = Allocation::buffer(format!("host.{name}"), Location::PINNED, bytes,
                Lifetime::RESIDENT, category, name);
            Component::allocation(owner, ByteCurve(pieces), false, maximum)
        }).collect())
    }
}

/// Everything both entries know about MiMo's physical owners before admission.
pub struct ProductionInput<'a> {
    pub expert_usage: bool,
    pub cfg: &'a MimoV2Config,
    pub layers: usize,
    pub kv: MimoKvFormats,
    pub mtp_stages: usize,
    pub blas_head: crate::placement::planner::serving::BlasHead,
    pub draft_head: bool,
    pub context: u64,
    pub maximum_pool_tokens: u64,
    pub prefill: PrefillShape,
    pub lanes: u64,
    pub embedding: EmbeddingHome,
    pub gpus: usize,
    pub spark_ranks: usize,
    /// Backbone order (including dense layers), exact whole/TP2 package costs.
    pub experts: Vec<Option<ExpertCost>>,
    pub expert_workspace: u64,
    pub tp2_workspace: [u64; 2],
    pub spark: MimoCapacityProfiles,
    pub no_spark: MimoCapacityProfiles,
    pub post_kv_additional: Vec<Vec<MemoryReservation>>,
    pub draft_packing: Vec<Vec<MemoryReservation>>,
    pub host: HostOwners,
    pub spark_reply_formats: cuteafd_core::spark_reply::SparkReplyFormats,
    /// Retained checkpoint weights per coordinator, media towers included.
    pub resident_weights: Vec<ResidentWeights>,
    pub operation: Operation,
    pub budgets: Vec<Budget>,
}

/// Prepared facts retained until the one shared solve selects them.
pub struct PreparedMimo<'a> {
    pub recipe: MimoRecipe<'a>,
    pub budgets: Vec<Budget>,
}

pub struct AdmittedMimo {
    pub placement: Placement,
    pub spark: SparkWorkingSet,
    pub solved: ResolvedPlan,
    /// The steady profile whose owners the admitted onboarding executes.
    profile: CapacityProfile,
    arena: [u64; 2],
    route: [u64; 2],
}

impl AdmittedMimo {
    /// Admitted expert-arena peak per coordinator, route exchange included.
    pub fn arena(&self) -> [u64; 2] { self.arena }

    /// TP2 route exchange carved out of each coordinator's arena.
    pub fn route(&self) -> [u64; 2] { self.route }
}

/// MiMo's resolved physical facts, priced by the shared planner.
pub struct MimoRecipe<'a> {
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub cfg: &'a MimoV2Config,
    pub gpus: usize,
    pub layers: usize,
    /// Routed backbone layers in execution order.
    pub routed: Vec<usize>,
    /// Ordered local prefixes the executor can actually place.
    pub counts: Vec<usize>,
    pub experts: Vec<Option<ExpertCost>>,
    pub expert_workspace: u64,
    pub tp2_workspace: [u64; 2],
    /// Ready TP2 route exchange bytes per GPU (only with two GPUs).
    pub route: [u64; 2],
    pub spark_ranks: usize,
    pub context: u64,
    pub page_rows: u64,
    pub maximum_pages: u64,
    pub prefill: PrefillShape,
    pub lanes: u64,
    pub embedding: EmbeddingHome,
    pub attention: Vec<LayerMode>,
    pub rows: u64,
    pub spark_profile: MimoCapacityProfiles,
    pub no_spark_profile: MimoCapacityProfiles,
    pub post_kv_additional: Vec<Vec<MemoryReservation>>,
    pub draft_packing: Vec<Vec<MemoryReservation>>,
    pub host: HostOwners,
    pub kv: MimoKvFormats,
    pub mtp_stages: usize,
    pub blas_head: crate::placement::planner::serving::BlasHead,
    pub draft_head: bool,
    pub spark_reply_formats: cuteafd_core::spark_reply::SparkReplyFormats,
    pub resident_weights: Vec<ResidentWeights>,
}

impl<'a> ProductionInput<'a> {
    pub fn prepare(self) -> Result<PreparedMimo<'a>, Refusal> {
        let routed = self.experts.iter().enumerate().filter_map(|(layer, cost)| cost.map(|_| layer))
            .collect::<Vec<_>>();
        if !(1..=2).contains(&self.gpus)
            || self.layers == 0
            || self.layers > self.cfg.layers
            || self.experts.len() != self.layers
            || self.resident_weights.len() != self.gpus
            || self.spark.steady.devices.len() != self.gpus
            || self.no_spark.steady.devices.len() != self.gpus
            || self.spark.steady.pool_unit_rows == 0
            || self.spark.steady.pool_unit_rows != self.no_spark.steady.pool_unit_rows
            || self.context == 0
            || self.spark_ranks > 8
        {
            return Err(invalid("incomplete production geometry"));
        }
        let expert_usage = self.expert_usage.then(|| cuteafd_core::expert_usage::Geometry {
            layers: self.layers, experts: self.cfg.experts, topk: self.cfg.topk,
            routed_layers: self.cfg.dense.iter().take(self.layers).enumerate()
                .filter_map(|(layer, &dense)| (!dense).then_some(layer)).collect(),
        }).filter(|geometry| !geometry.routed_layers.is_empty());
        if expert_usage.as_ref().is_some_and(|geometry| !geometry.validate()) {
            return Err(invalid("invalid expert usage geometry"));
        }
        let page_rows = self.spark.steady.pool_unit_rows;
        let cache = mimo_cache_geometry(self.cfg, self.layers, self.gpus, self.kv, self.mtp_stages)
            .map_err(|error| invalid(error))?;
        if cache.logical_unit_rows != page_rows {
            return Err(invalid("pool quantum differs from the cache geometry"));
        }
        // Only prefixes the executor can place are candidate facts; the rest
        // stays a Spark suffix and is never a silent local home.
        let executable = routed.iter().take_while(|&&layer| {
            let cost = self.experts[layer].expect("routed cost");
            if self.gpus == 2 { cost.tp2 && self.tp2_workspace.iter().all(|&bytes| bytes > 0) }
            else { cost.whole_supported && self.expert_workspace > 0 }
        }).count();
        let counts = if routed.is_empty() {
            vec![0]
        } else if self.spark_ranks == 0 {
            if executable != routed.len() {
                return Err(Refusal::Unsupported);
            }
            vec![routed.len()]
        } else {
            (0..=executable).collect()
        };
        let route = if self.gpus == 2 && self.tp2_workspace.iter().any(|&bytes| bytes > 0) {
            let bytes = route_exchange_bytes(self.prefill.rows.max(DECODE_ROWS), self.cfg.topk as u64, self.lanes)
                .map_err(|error| invalid(error))?;
            [bytes; 2]
        } else { [0; 2] };
        let attention = (0..self.layers).map(|_| if self.gpus == 2 {
            LayerMode::HeadSplit
        } else {
            LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }
        }).collect();
        let maximum_pages = self.maximum_pool_tokens.div_ceil(page_rows);
        Ok(PreparedMimo {
            recipe: MimoRecipe {
                expert_usage, cfg: self.cfg, gpus: self.gpus, layers: self.layers, routed, counts,
                experts: self.experts, expert_workspace: self.expert_workspace,
                tp2_workspace: self.tp2_workspace, route, spark_ranks: self.spark_ranks,
                context: self.context, page_rows, maximum_pages, prefill: self.prefill,
                lanes: self.lanes, embedding: self.embedding, attention, rows: self.prefill.rows,
                spark_profile: self.spark, no_spark_profile: self.no_spark,
                resident_weights: self.resident_weights, post_kv_additional: self.post_kv_additional,
                draft_packing: self.draft_packing, host: self.host, kv: self.kv, mtp_stages: self.mtp_stages,
                spark_reply_formats: self.spark_reply_formats,
                blas_head: self.blas_head, draft_head: self.draft_head,
            },
            budgets: self.budgets.into_iter().map(|mut budget| {
                if budget.locations.contains(&Location::HOST) { budget.headroom = self.host.headroom; }
                budget
            }).collect(),
        })
    }
}

impl PreparedMimo<'_> {
    /// One typed admission over the prepared facts. The shared planner picks
    /// the pool and the onboarding prefix; nothing here re-solves.
    pub fn admit(&self, mut request: Request) -> Result<AdmittedMimo, Refusal> {
        // The operator's explicit pool keeps the old quantum rounding; an
        // unaligned request is not a reason to plan a different pool.
        if let Pin::Exact(tokens) = request.pool {
            let page = self.recipe.page_rows;
            if tokens % page != 0 {
                request.pool = Pin::Exact(tokens.div_ceil(page).checked_mul(page)
                    .ok_or(Refusal::Overflow("MiMo pool quantum"))?);
            }
        }
        let solved = crate::placement::planner::solve(request, self.budgets.clone(), &self.recipe)?;
        let (placement, spark) = self.recipe.execution(&solved)?;
        let count = spark.rtx_layers.len();
        let profile = self.recipe.profile(count).clone();
        let (arena, _) = self.recipe.arena(count)?;
        Ok(AdmittedMimo { placement, spark, solved, profile, arena, route: self.recipe.route })
    }
}

impl MimoRecipe<'_> {
    pub fn profile(&self, count: usize) -> &CapacityProfile {
        if count < self.routed.len() { &self.spark_profile.steady } else { &self.no_spark_profile.steady }
    }

    /// Arena peak and retained expert weights per GPU for one local prefix.
    fn arena(&self, count: usize) -> Result<([u64; 2], [u64; 2]), Refusal> {
        let mut peak = [0u64; 2];
        let mut weights = [0u64; 2];
        if count == 0 { return Ok((peak, weights)); }
        let prefix = self.routed.get(..count).ok_or_else(|| invalid("expert prefix past the routed layers"))?;
        if self.gpus == 2 {
            let mut resident = [0u64; 2];
            let mut staging = [0u64; 2];
            for &layer in prefix {
                let cost = self.experts[layer].ok_or_else(|| invalid("missing expert cost"))?;
                if !cost.tp2 { return Err(invalid("TP2 placement of a layer without TP2 halves")); }
                for gpu in 0..2 {
                    resident[gpu] = resident[gpu].checked_add(cost.half[gpu].resident)
                        .ok_or(Refusal::Overflow("MiMo expert arena"))?;
                    staging[gpu] = staging[gpu].max(cost.half[gpu].staging);
                }
            }
            for gpu in 0..2 {
                weights[gpu] = resident[gpu];
                peak[gpu] = self.tp2_workspace[gpu]
                    .checked_add(self.route[gpu])
                    .and_then(|n| n.checked_add(resident[gpu]))
                    .and_then(|n| n.checked_add(staging[gpu]))
                    .ok_or(Refusal::Overflow("MiMo TP2 arena peak"))?;
            }
        } else {
            let mut resident = self.expert_workspace;
            peak[0] = resident;
            for &layer in prefix {
                let cost = self.experts[layer].ok_or_else(|| invalid("missing expert cost"))?;
                if !cost.whole_supported { return Err(invalid("whole placement of an unsupported layer")); }
                peak[0] = peak[0].max(resident
                    .checked_add(cost.whole.resident)
                    .and_then(|n| n.checked_add(cost.whole.staging))
                    .ok_or(Refusal::Overflow("MiMo expert arena peak"))?);
                resident = resident.checked_add(cost.whole.resident)
                    .ok_or(Refusal::Overflow("MiMo expert arena"))?;
                weights[0] = weights[0].checked_add(cost.whole.resident)
                    .ok_or(Refusal::Overflow("MiMo expert weights"))?;
            }
        }
        Ok((peak, weights))
    }

    /// The runtime allocates the target record pool as one page buffer per
    /// full-attention layer, so each layer rounds its own SM120 backing. A
    /// layer's request is `persistent_unit_bytes / backings` at every page
    /// count, which keeps the shared pool curve's split exact.
    fn pool_backings(&self) -> Result<u64, Refusal> {
        let full = self.cfg.attention[..self.layers].iter()
            .filter(|attention| **attention == MimoAttention::Full).count() as u64;
        if full == 0 { return Err(invalid("MiMo pool has no full-attention layers")); }
        Ok(full)
    }

    /// Per-page steady bytes on `rank`, split into KV records and page tables.
    fn pool_pages(&self, profile: &CapacityProfile, rank: usize) -> Result<(u64, u64), Refusal> {
        let costs = profile.devices.get(rank).ok_or_else(|| invalid("profile rank"))?;
        let cache = mimo_cache_geometry(self.cfg, self.layers, self.gpus,
            self.kv, self.mtp_stages).map_err(|error| invalid(error))?;
        let records = cache.ranks[rank].persistent_unit_bytes
            .checked_add(cache.ranks[rank].pool_metadata_unit_bytes)
            .ok_or(Refusal::Overflow("MiMo pool records"))?;
        let tables = costs.pool_unit_bytes.checked_sub(records)
            .ok_or_else(|| invalid("profile pool unit bytes omit layer records"))?;
        Ok((records, tables))
    }

    /// Validate the admitted decisions against the prepared geometry and build
    /// the execution placement from the owners that were admitted.
    pub fn execution(&self, solved: &ResolvedPlan) -> Result<(Placement, SparkWorkingSet), Refusal> {
        let decisions = &solved.decisions;
        if decisions.attention != self.attention
            || decisions.prefill != self.prefill
            || decisions.lanes != self.lanes
            || decisions.embedding.is_some_and(|home| home != self.embedding)
        {
            return Err(invalid("admitted decisions differ from the prepared geometry"));
        }
        let local = self.routed.iter().copied().filter(|layer|
            matches!(decisions.expert_homes.get(layer), Some(ExpertDestination::Coordinator { .. })))
            .collect::<Vec<_>>();
        let remote = self.routed.iter().copied().filter(|layer|
            matches!(decisions.expert_homes.get(layer), Some(ExpertDestination::Sparks { .. })))
            .collect::<Vec<_>>();
        let count = local.len();
        if local.len() + remote.len() != self.routed.len()
            || !self.counts.contains(&count)
            || !local.iter().copied().eq(self.routed[..count].iter().copied())
            || !remote.iter().copied().eq(self.routed[count..].iter().copied())
        {
            return Err(invalid("admitted expert homes are not a declared local prefix"));
        }
        for (rank, layers) in decisions.onboard_layers.iter().enumerate() {
            let expected = if self.gpus == 2 || rank == 0 { &self.routed[..count] } else { &[][..] };
            if layers.as_slice() != expected {
                return Err(invalid("admitted onboarding differs from the expert homes"));
            }
        }
        let (arena, _) = self.arena(count)?;
        let first = self.routed.first().copied().unwrap_or(0);
        let mut expert_ranges = (0..self.gpus).map(|_| ExpertRange { first, layers: 0, peak_bytes: 0 })
            .collect::<Vec<_>>();
        let tp2 = if self.gpus == 2 && count > 0 {
            Some(Tp2Range { first, layers: count, peak_bytes: arena })
        } else {
            if self.gpus == 1 && count > 0 {
                expert_ranges[0] = ExpertRange { first, layers: count, peak_bytes: arena[0] };
            }
            None
        };
        let spec = HopSpec {
            storage: HopStorage::ReceiveSlots,
            row_bytes: self.cfg.hidden as u64 * 2,
            rows: self.rows.max(DECODE_ROWS),
            lanes: self.lanes,
            entry_gpu: 0,
            head_gpu: 0,
            terminal: if self.gpus == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner },
        };
        let residual = residual_homes(&self.attention, &spec).map_err(|error| invalid(error))?;
        let hops = plan_hops(&self.attention, &spec).map_err(|error| invalid(error))?;
        let local_set = local.iter().collect::<std::collections::BTreeSet<_>>();
        let layers = (0..self.layers).map(|layer| LayerAssignment {
            mode: self.attention[layer],
            experts: if !self.routed.contains(&layer) {
                ExpertHome::Dense
            } else if local_set.contains(&layer) {
                if self.gpus == 2 { ExpertHome::RtxTp2 } else { ExpertHome::RtxWhole { gpu: 0 } }
            } else {
                ExpertHome::Spark
            },
        }).collect::<Vec<_>>();
        let attention_by_kind = (0..self.layers)
            .map(|layer| (if self.cfg.attention[layer] == MimoAttention::Full {
                AttentionClass::Gqa
            } else {
                AttentionClass::Swa
            }, AttentionPlacement::Heads))
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_iter()
            .collect::<Vec<_>>();
        let items = (0..self.gpus).map(|rank| solved.items(Location::coordinator(rank as u8)))
            .collect::<Result<Vec<_>, _>>()?;
        let placement = Placement {
            pool_tokens: solved.pool_tokens,
            pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Heads,
            attention_by_kind,
            peer_row_bytes: 0,
            onboard_layers: count,
            max_pool_tokens_no_onboard: None,
            layers,
            movables: vec![],
            expert_ranges,
            residual,
            hops,
            tp2,
            items,
        };
        super::super::MIMO_V2.check(&placement).map_err(|error| invalid(error))?;
        Ok((placement, SparkWorkingSet {
            rtx_layers: local,
            spark_layers: remote,
            spark_ranks: if count < self.routed.len() { self.spark_ranks } else { 0 },
            prefill_lanes: self.lanes,
        }))
    }
}

impl Recipe for MimoRecipe<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if request.context.maximum == 0 { return Err(invalid("zero context")); }
        let maximum = self.maximum_pages;
        let mut contract = Contract { floor_fallback: None,
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: (self.spark_ranks > 0).then_some(cuteafd_core::spark_reply::SparkReplyFormats {
                prefill: Some(self.spark_reply_formats.prefill(false)), decode: Some(self.spark_reply_formats.decode()) }),
            resident_weight_limit: request.resident_weight_limit,
            page_rows: self.page_rows,
            pool_page_multiple: 1,
            maximum_pages: maximum,
            gpus: self.gpus,
            variables: vec![],
            requires: vec![],
            components: vec![],
        };
        if let Some(geometry) = &self.expert_usage {
            contract.components.extend(crate::placement::expert_usage::components(geometry, &[0], maximum)?);
        }
        for (layer, mode) in self.attention.iter().enumerate() {
            contract.variables.push(Variable::Attention { layer, values: vec![*mode] });
        }
        contract.variables.push(Variable::Prefill(vec![self.prefill]));
        contract.variables.push(Variable::Lanes(vec![self.lanes]));
        contract.variables.push(Variable::Embedding(vec![self.embedding]));
        let onboard = contract.variables.len();
        contract.variables.push(Variable::Onboard {
            layers: self.routed.clone(),
            destination: (0..self.gpus as u8).collect(),
            counts: self.counts.clone(),
        });
        if self.spark_ranks > 0 {
            contract.variables.push(Variable::ExpertHome {
                layers: self.routed.clone(),
                rank_sets: vec![(0..self.spark_ranks as u8).collect()],
            });
        }
        let mut blas = Vec::new();
        if self.blas_head.bf16() {
            blas.push(crate::placement::planner::serving::BlasOwners {
                name: "MiMo target and MTP vocabulary projections".into(), process: "coordinator".into(),
                thread: "mimo/blas/serve".into(), location: Location::coordinator(0),
                gemm: self.blas_head.gemm(), lt: self.blas_head.lt(),
                lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth },
                activation: crate::placement::planner::Activation::always(),
            });
            if self.draft_head {
                blas.push(crate::placement::planner::serving::BlasOwners {
                    name: "MiMo DFlash borrowed BF16 vocabulary".into(), process: "coordinator".into(),
                    thread: "mimo/blas/serve".into(), location: Location::coordinator(0), gemm: true, lt: false,
                    lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth },
                    activation: crate::placement::planner::Activation::always(),
                });
            }
        }
        contract.components.extend(crate::placement::planner::serving::blas_components(
            "sm_120", &blas, maximum, contract.phase_order)?);
        // Retained driver stream slabs are a named owner, not a cudaMalloc
        // request, so they receive no allocation-tail charge. Pinned host
        // backings' driver page tables are added by the shared solve
        // (`Contract::add_driver_page_tables`) and charged at runtime through
        // `resolved_capacity`.
        let stream_slabs = crate::placement::planner::cuda_stream_slab_bytes(startup_peak_live_streams())?;
        let streams = startup_peak_live_streams();
        for rank in 0..self.gpus {
            contract.components.push(Component {
                activation: crate::placement::planner::Activation::always(),
                bytes: ByteCurve::fixed(stream_slabs, maximum),
                kind: crate::placement::planner::ComponentKind::Allowance(crate::placement::planner::Allowance {
                    location: Location::coordinator(rank as u8), bytes: stream_slabs,
                    lifetime: Lifetime::RESIDENT, assumed: false,
                    source: format!("mimo/driver-stream-slabs/gpu{rank} ({streams} peak live streams, SM120 driver/runtime13020 LAZY, #1511 2026-11-08)"),
                }),
            });
        }
        // Immutable arenas replace every per-tensor payload and claim. Their
        // separately retired lifetimes are independent of expert onboarding.
        if self.spark_profile.weight_arenas != self.no_spark_profile.weight_arenas {
            return Err(invalid("weight arenas differ with expert placement"));
        }
        let mut arena_names = std::collections::BTreeSet::new();
        for (rank, arenas) in self.spark_profile.weight_arenas.iter().enumerate() {
            for arena in arenas {
                arena_names.insert(arena.layout.id().to_owned());
                arena_names.insert(format!("{}.tail", arena.layout.id()));
                contract.components.extend(crate::placement::planner::weight_arena_components(
                    &arena.layout, Location::coordinator(rank as u8), arena.lifetime, maximum, arena.kind)?);
            }
        }
        // Media towers already in the baseline and separately managed expert
        // windows retain their old claims, excluding the arena claims above.
        for (rank, &weights) in self.resident_weights.iter().enumerate() {
            let arena_bytes = self.spark_profile.weight_arenas.get(rank).into_iter().flatten()
                .try_fold(0u64, |sum, arena| sum.checked_add(arena.layout.bytes() as u64))
                .ok_or(Refusal::Overflow("MiMo arena weight claims"))?;
            let weights = ResidentWeights(weights.0.checked_sub(arena_bytes)
                .ok_or_else(|| invalid("arena weight claims exceed retained weights"))?);
            if weights.0 == 0 { continue; }
            let location = Location::coordinator(rank as u8);
            contract.components.push(Component::resident_weight(
                crate::placement::planner::ResidentWeight {
                    id: format!("rtx{rank}.weights"),
                    location,
                    weights,
                },
                ByteCurve::fixed(weights.0, maximum),
            ));
        }
        for (option, &count) in self.counts.iter().enumerate() {
            let activation = crate::placement::planner::Activation::choice(onboard, option);
            let profiles = if count < self.routed.len() { &self.spark_profile } else { &self.no_spark_profile };
            let profile = &profiles.steady;
            for mut component in self.host.components(self, count)? {
                component.activation = activation.clone();
                contract.components.push(component);
            }
            for (phase, costs) in [(Phase::Load, &profiles.loading),
                (Phase::PostTargetKv, &profiles.post_target_kv), (Phase::DraftLoad, &profiles.post_target_kv)] {
                for (rank, costs) in costs.iter().enumerate() {
                    let mut reservations = costs.reservations.clone();
                    if phase != Phase::Load {
                        reservations.extend(self.post_kv_additional.get(rank).into_iter().flatten().cloned());
                        if phase == Phase::DraftLoad { reservations.extend(self.draft_packing.get(rank).into_iter().flatten().cloned()); }
                    }
                    for reservation in reservations {
                        if arena_names.contains(&reservation.name) { continue; }
                        let owner = Allocation::buffer(format!("rtx{rank}.{phase:?}.{}", reservation.name),
                            Location::coordinator(rank as u8), reservation.bytes, Lifetime { first: phase, last: phase },
                            super::reservation_category(&reservation.name), super::reservation_group(&reservation.name));
                        let mut component = Component::allocation(owner, ByteCurve::fixed(reservation.bytes, maximum), false, maximum);
                        component.activation = activation.clone();
                        contract.components.push(component);
                    }
                    if costs.pool_unit_bytes > 0 {
                        let owner = Allocation::buffer(format!("rtx{rank}.{phase:?}.pool"), Location::coordinator(rank as u8),
                            costs.pool_unit_bytes, Lifetime { first: phase, last: phase }, Category::Kv, "records");
                        // The runtime allocates one page buffer per full-attention
                        // layer; each rounds its own SM120 backing (see pool_pages).
                        let mut component = Component::allocation(owner,
                            ByteCurve::paged(costs.pool_unit_bytes, costs.pool_unit_bytes,
                                SM120_ALLOCATION_GRANULARITY_BYTES, self.pool_backings()?, maximum),
                            false, maximum);
                        component.activation = activation.clone();
                        contract.components.push(component);
                    }
                }
            }
            let (arena, weights) = self.arena(count)?;
            let first = self.routed.first().copied().unwrap_or(0);
            for (rank, costs) in profile.devices.iter().enumerate() {
                let location = Location::coordinator(rank as u8);
                for reservation in &costs.reservations {
                    if arena_names.contains(&reservation.name) { continue; }
                    let owner = Allocation::buffer(format!("rtx{rank}.{}", reservation.name),
                        location, reservation.bytes, Lifetime { first: Phase::Warmup, last: Phase::Growth },
                        super::reservation_category(&reservation.name),
                        super::reservation_group(&reservation.name));
                    let mut component = Component::allocation(owner,
                        ByteCurve::fixed(reservation.bytes, maximum), false, maximum);
                    component.activation = activation.clone();
                    contract.components.push(component);
                }
                let (records, tables) = self.pool_pages(profile, rank)?;
                for (group, category, bytes, curve) in [
                    ("records", Category::Kv, records,
                        ByteCurve::paged(records, records, SM120_ALLOCATION_GRANULARITY_BYTES,
                            self.pool_backings()?, maximum)),
                    ("pool page tables", Category::Workspace, tables,
                        ByteCurve::linear(tables, tables, maximum)),
                ] {
                    if bytes == 0 { continue; }
                    let owner = Allocation::buffer(format!("rtx{rank}.pool.{group}"), location,
                        bytes, Lifetime { first: Phase::Warmup, last: Phase::Growth }, category, group);
                    let mut component = Component::allocation(owner, curve, false, maximum);
                    component.activation = activation.clone();
                    contract.components.push(component);
                }
                if arena[rank] > 0 {
                    let group = if self.gpus == 2 {
                        format!("TP2 expert layer halves {first}..{}", first + count)
                    } else {
                        format!("resident routed layers {:?}", &self.routed[..count])
                    };
                    let owner = Allocation::buffer(format!("rtx{rank}.experts.arena"), location,
                        arena[rank].saturating_sub(self.route[rank]), Lifetime { first: Phase::Warmup, last: Phase::Growth },
                        Category::Experts, group);
                    let mut component = Component::allocation(owner,
                        ByteCurve::fixed(arena[rank].saturating_sub(self.route[rank]), maximum),
                        false, maximum);
                    component.activation = activation.clone();
                    contract.components.push(component);
                    if self.route[rank] > 0 {
                        let owner = Allocation::buffer(format!("rtx{rank}.transport.tp2_routes"),
                            location, self.route[rank], Lifetime { first: Phase::Warmup, last: Phase::Growth }, Category::Transport,
                            "TP2 canonical routes");
                        let mut component = Component::allocation(owner,
                            ByteCurve::fixed(self.route[rank], maximum), false, maximum);
                        component.activation = activation.clone();
                        contract.components.push(component);
                    }
                    if weights[rank] > 0 {
                        contract.components.push(Component::resident_weight(
                            crate::placement::planner::ResidentWeight {
                                id: format!("rtx{rank}.experts.arena"),
                                location,
                                weights: ResidentWeights(weights[rank]),
                            },
                            ByteCurve::fixed(weights[rank], maximum),
                        ));
                        // The arena resident weight shares its owner and the
                        // activation of the arena it backs.
                        contract.components.last_mut().expect("weight component").activation =
                            activation.clone();
                    }
                }
            }
        }
        Ok(contract)
    }
}

/// Preserve the capacity interface consumed by context/prefix startup, without
/// a second pool solve. Load and post-KV packing are phases of the same admission.
pub fn resolved_capacity(admitted: &AdmittedMimo, concurrency: u32, rings: u32)
    -> Result<ResolvedCapacity, CacheGeometryError> {
    let profile = &admitted.profile;
    let placement = &admitted.placement;
    if concurrency == 0 || rings < concurrency || profile.pool_unit_rows == 0 {
        return Err(PlacementError::Inventory("MiMo concurrency, rings or pool quantum").into());
    }
    let units = placement.pool_tokens / profile.pool_unit_rows;
    let mut feasible_units = u64::MAX;
    let mut devices = Vec::new();
    for (rank, costs) in profile.devices.iter().enumerate() {
        let budget = admitted.solved.budgets.iter().find(|budget| budget.id == format!("rtx{rank}"))
            .ok_or(PlacementError::Inventory("MiMo rank budget"))?;
        let available = budget.available().map_err(|_| PlacementError::Inventory("MiMo rank budget"))?;
        let mut reservations = costs.reservations.clone();
        if admitted.arena[rank] > 0 {
            reservations.push(MemoryReservation {
                name: format!("experts.{}", if admitted.route[rank] > 0 { "TP2 expert layer halves" }
                    else { "resident routed layers" }),
                bytes: admitted.arena[rank].saturating_sub(admitted.route[rank]),
            });
        }
        if admitted.route[rank] > 0 {
            reservations.push(MemoryReservation { name: "transport.tp2_routes".into(),
                bytes: admitted.route[rank] });
        }
        reservations.extend(admitted.solved.facts.allowances.iter().filter(|owner|
            owner.location == Location::coordinator(rank as u8)
                && owner.lifetime.contains(Phase::Ready, admitted.solved.facts.phase_order))
            .map(|owner| MemoryReservation { name: owner.source.clone(), bytes: owner.bytes }));
        // Portable CUDA host backings (the retained prefix chunks and every
        // other PINNED owner) register driver page tables in each coordinator
        // context. The solve charged them; this list must too, or the pool
        // sized around them reads back as an unexplained residual.
        reservations.extend(admitted.solved.facts.driver_page_tables.iter().filter(|mapping|
            mapping.location == Location::coordinator(rank as u8)
                && mapping.lifetime.contains(Phase::Ready, admitted.solved.facts.phase_order))
            .map(|mapping| MemoryReservation { name: mapping.source(), bytes: mapping.bytes() }));
        let reserved = reservations.iter().try_fold(0u64, |sum, reservation|
            sum.checked_add(reservation.bytes))
            .ok_or(PlacementError::Overflow("MiMo reservations"))?;
        let pool_bytes = units.checked_mul(costs.pool_unit_bytes)
            .ok_or(PlacementError::Overflow("MiMo pool bytes"))?;
        let used = reserved.checked_add(pool_bytes)
            .ok_or(PlacementError::Overflow("MiMo engine bytes"))?;
        let unused = available.checked_sub(used).ok_or(PlacementError::Mandatory {
            gpu: rank as u8, what: "MiMo resolved capacity".into() })?;
        if costs.pool_unit_bytes > 0 {
            feasible_units = feasible_units.min((available - reserved) / costs.pool_unit_bytes);
        }
        let non_engine_bytes = match &budget.baseline {
            Baseline::Sampled { free, .. } => budget.physical_bytes.saturating_sub(*free),
            Baseline::Forecast { occupied, .. } => *occupied,
        };
        devices.push(ResolvedDeviceCapacity {
            device: costs.device,
            total_bytes: budget.physical_bytes,
            non_engine_bytes,
            engine_budget_bytes: available,
            reservations,
            reserved_bytes: reserved,
            pool_bytes,
            unused_budget_bytes: unused,
        });
    }
    let feasible = feasible_units.checked_mul(profile.pool_unit_rows)
        .ok_or(PlacementError::Overflow("MiMo feasible pool"))?;
    let floor = match admitted.solved.request.pool {
        Pin::Exact(tokens) => tokens,
        Pin::Auto => {
            let small = admitted.solved.budgets.iter().any(|budget|
                budget.physical_bytes <= 34 << 30
                    && budget.locations.iter().any(|location|
                        matches!(location.device, crate::placement::planner::Device::Coordinator(_))));
            (if small { 1u64 << 20 } else { 2u64 << 20 })
                .max(admitted.solved.request.context.maximum)
                .div_ceil(profile.pool_unit_rows)
                .saturating_mul(profile.pool_unit_rows)
        }
    };
    Ok(ResolvedCapacity {
        concurrency,
        state_slots: rings,
        checkpoint_max_context_tokens: profile.context.checkpoint_max_tokens,
        compiled_index_max_context_tokens: profile.context.compiled_index_max_tokens,
        effective_max_context_tokens: admitted.solved.request.context.served(admitted.solved.pool_tokens),
        requested_kv_floor_tokens: floor,
        feasible_gpu_kv_tokens: feasible,
        allocated_gpu_kv_tokens: placement.pool_tokens,
        requested_floor_fits_hardware: feasible >= floor,
        requested_floor_allocated: placement.pool_tokens >= floor,
        requested_floor_shortfall_tokens: floor.saturating_sub(placement.pool_tokens),
        active_max_context_sequences: (units / admitted.solved.request.context.served(admitted.solved.pool_tokens).div_ceil(profile.pool_unit_rows))
            .min(concurrency as u64) as u32,
        host_prefix_bytes: profile.host_prefix_bytes,
        explicit_pool_override: matches!(admitted.solved.request.pool, Pin::Exact(_)),
        devices,
    })
}

/// Ready route exchange includes receive slots, control and abort words on each rank.
pub fn route_exchange_bytes(rows: u64, topk: u64, lanes: u64) -> anyhow::Result<u64> {
    let slot = |rows: u64| -> anyhow::Result<u64> {
        rows.checked_mul(topk).and_then(|n| n.checked_mul(4)).map(|n| n.div_ceil(16))
            .and_then(|n| n.checked_mul(32)).map(|n| n.max(256))
            .ok_or_else(|| anyhow::anyhow!("MiMo route slot overflow"))
    };
    let prefill_slots = lanes.checked_mul(2).ok_or_else(|| anyhow::anyhow!("MiMo route slots overflow"))?;
    let slots = prefill_slots.checked_add(2).ok_or_else(|| anyhow::anyhow!("MiMo route slots overflow"))?;
    let control = slots.checked_add(1).and_then(|words| words.checked_mul(16))
        .ok_or_else(|| anyhow::anyhow!("MiMo route control overflow"))?;
    let decode = slot(64)?.checked_mul(2).ok_or_else(|| anyhow::anyhow!("MiMo decode routes overflow"))?;
    prefill_slots.checked_mul(slot(rows)?).and_then(|bytes| bytes.checked_add(decode))
        .and_then(|bytes| bytes.checked_add(control.max(256))).and_then(|bytes| bytes.checked_add(256))
        .ok_or_else(|| anyhow::anyhow!("MiMo route exchange bytes overflow"))
}

#[cfg(test)]
mod release_padding_tests {
    use super::*;
    use cuteafd_core::memory_policy::{MemoryPolicy, DEFAULT_MEMORY_PADDING};

    #[test]
    fn padding_is_not_swallowed_by_the_drained_intake_probe() {
        let probe = 64 << 20;
        assert_eq!(device_headroom(0, probe), DEFAULT_MEMORY_PADDING + probe);
        assert_eq!(device_headroom(1, probe), DEFAULT_MEMORY_PADDING);
        MemoryPolicy { memory_padding: Some(0), ..Default::default() }.scoped(|| {
            assert_eq!(device_headroom(0, probe), probe);
            assert_eq!(device_headroom(1, probe), 0);
        });
        for reserve in [0, 123] {
            MemoryPolicy { gpu_reserve: Some(reserve), ..Default::default() }.scoped(|| {
                assert_eq!(device_headroom(0, probe), reserve);
                assert_eq!(device_headroom(1, probe), reserve);
            });
        }
    }
}
