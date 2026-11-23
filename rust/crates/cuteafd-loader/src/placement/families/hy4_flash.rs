//! Hy4-flash physical owners for the shared contract planner (#2478).
//!
//! One `Recipe` over `placement::planner::solve` shared by `cuteafd plan --layout`
//! and runtime admission:
//! - packed target arena per coordinator GPU (globals + 42 layers, head-split or whole),
//! - optional RTX routed-expert onboarding arena per prefix count (whole on 1 GPU,
//!   TP2 halves on 2 GPUs; 41 MoE layers × 512 × 3 EXL3 K4 MUL1),
//! - Spark experts + owners + runtime residual + SM121 arena page tables when remote,
//! - 12 DSA layers' paged KV + compressed-indexer group cache (linear curve in pages),
//! - 30 SWA-512 layers' fixed per-sequence circular rings + exact ~16.9 MiB prefix marks,
//! - decode + prefill unit workspaces, FP32 vocabulary head rows, canonical `graph reserve`,
//!   loaded code, stream slabs, BLAS, and host mapped-table staging/prefix quota.
use cuteafd_core::memory_layout::{Category, MemoryKind};
use cuteafd_core::spark_reply::SparkReplyFormats;

use super::super::planner::*;
use super::super::{
    plan_hops, residual_homes, AttentionClass, AttentionPlacement, ExecutorModes, ExpertHome,
    ExpertRange, FfnMode, HopSpec, HopStorage, LayerAssignment, LayerMode, Placement, PoolLimits,
    ResidentWeights, TerminalHome, Tp2Range,
};
use crate::families::hy4_flash::execution::{CacheLayout, UnitLayout, PAGE_ROWS};
use crate::families::hy4_flash::weight_layout::CoordinatorWeights;
use crate::families::hy4_flash::{Hy4FlashConfig, Hy4KvCache};
use crate::serving_capacity::WeightArenaLayout;

pub const LANES: u64 = 1;
pub const DECODE_ROWS: u32 = 64;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };

/// Supported coordinator attention placements for Hy4-flash:
/// - `context` (default on 2 GPUs): head-split SWA + context-split DSA (page-parity KV across GPUs, zero KV replication).
/// - `layers`: contiguous whole-layer halves (21/21, one boundary hop, zero KV replication).
/// - `heads`: explicit opt-in head-split across both GPUs (replicates DSA KV).
pub const HY4_FLASH: ExecutorModes = ExecutorModes {
    family: "hy4_flash",
    modes: &[
        LayerMode::HeadSplit,
        LayerMode::ContextSplit,
        WHOLE,
        LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner },
    ],
    hops: true,
    attention: AttentionPlacement::Context,
};

pub fn layer_modes(cfg: &Hy4FlashConfig, gpus: usize, placement: AttentionPlacement) -> Vec<LayerMode> {
    if gpus == 1 {
        return vec![WHOLE; cfg.layers];
    }
    let cut = cfg.layers.div_ceil(2);
    (0..cfg.layers)
        .map(|layer| match placement {
            AttentionPlacement::Heads => LayerMode::HeadSplit,
            AttentionPlacement::Context => {
                if cfg.sparse_layer[layer] { LayerMode::ContextSplit } else { LayerMode::HeadSplit }
            }
            AttentionPlacement::Layers => LayerMode::Whole { gpu: u8::from(layer >= cut), ffn: FfnMode::Owner },
        })
        .collect()
}

fn invalid(detail: impl std::fmt::Display) -> Refusal {
    Refusal::InvalidFact(format!("Hy4-flash component contract: {detail}"))
}

fn refusal(error: anyhow::Error) -> Refusal {
    invalid(format!("{error:#}"))
}

fn buffer(
    id: impl Into<String>,
    location: Location,
    bytes: u64,
    lifetime: Lifetime,
    category: Category,
    group: &str,
    maximum: u64,
) -> Option<Component> {
    (bytes > 0).then(|| {
        Component::allocation(
            Allocation::buffer(id, location, bytes, lifetime, category, group),
            ByteCurve::fixed(bytes, maximum),
            false,
            maximum,
        )
    })
}

fn allowance(
    location: Location,
    bytes: u64,
    lifetime: Lifetime,
    source: String,
    assumed: bool,
    maximum: u64,
) -> Option<Component> {
    (bytes > 0).then(|| Component {
        activation: Activation::always(),
        bytes: ByteCurve::fixed(bytes, maximum),
        kind: ComponentKind::Allowance(Allowance { location, bytes, lifetime, source, assumed }),
    })
}

fn with(mut components: Vec<Component>, activation: &Activation) -> Vec<Component> {
    for component in &mut components {
        component.activation = activation.clone();
    }
    components
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalBacking {
    pub gpu: usize,
    pub layers: Vec<usize>,
    pub arena: WeightArenaLayout,
    pub workspace: Vec<(String, u64)>,
    pub tp2_rows: Option<crate::serving_capacity::OwnerTp2Rows>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPackage {
    Installed { backings: Vec<Vec<LocalBacking>> },
    Missing(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparkCharge {
    pub experts: u64,
    pub owners: u64,
    pub runtime: u64,
}

#[derive(Debug, Clone)]
pub struct Hy4Inputs {
    pub cfg: Hy4FlashConfig,
    pub kv_format: Hy4KvCache,
    pub gpus: usize,
    pub spark_ranks: usize,
    pub context: Context,
    pub pool: crate::placement::PoolPolicy,
    pub prefill: PrefillShape,
    pub decode_rows: u32,
    pub sequences: u64,
    pub prefix_mark_slots: usize,
    pub targets: Vec<CoordinatorWeights>,
    pub placement: AttentionPlacement,
    pub modes: Vec<LayerMode>,
    pub routed: Vec<usize>,
    pub local: LocalPackage,
    pub spark: Vec<Vec<SparkCharge>>,
    pub links: Vec<crate::placement::PreparedSparkLink>,
    pub budgets: Vec<Budget>,
    pub code: Vec<u64>,
    pub graph_reserve: Vec<u64>,
    pub host_prefix: serving::HostPrefixOwners,
    /// Host-mapped FP8 Engram preload policy (`MappedPlacement`, 61.47 GiB `engram_fp8`),
    /// charged strictly to raptor host RAM (never GPU memory).
    pub engram_mapped: crate::formats::mapped_preload::MappedPlacement,
    /// Measured per-layer `RowCache` host bytes (`Hy4EngramShape::cache_bytes()`).
    pub engram_row_cache_bytes: u64,
    /// Measured pinned host staging bytes for prefill & decode Engram gathers (`Hy4EngramShape::staging_bytes()`).
    pub engram_staging_bytes: u64,
    pub resident_weight_limit: Option<ResidentWeights>,
}

#[derive(Debug, Clone)]
pub struct AdmittedHy4 {
    pub placement: Placement,
    pub solved: ResolvedPlan,
    pub onboard: usize,
}

impl AdmittedHy4 {
    /// Projection of the admitted shared host budget, including host-mapped Engram tables.
    pub fn host_ledger(&self, quota: u64) -> Result<cuteafd_core::memory_layout::HostLayout, Refusal> {
        use cuteafd_core::memory_layout::{Basis, HostLayout, Item, ItemLifetime};
        let budget = self.solved.budgets.iter().find(|b| b.locations.contains(&Location::HOST))
            .ok_or_else(|| invalid("missing host budget"))?;
        let available = budget.available()?;
        let peak = self.solved.phases.iter().filter(|p| p.budget == budget.id)
            .map(|p| p.owned + p.opaque).max().unwrap_or(0);
        let mut items = self.solved.items(Location::HOST)?;
        items.extend(self.solved.items(Location::PINNED)?);
        items.retain(|item| item.group != "host prefix cache");
        let ready = items.iter().map(|i| i.bytes).sum::<u64>() + quota;
        if peak > ready {
            items.push(Item::new(Category::Staging, "load phase above retained host owners", "", peak - ready,
                Basis::Exact).with_lifetime(ItemLifetime::ReadyResident));
        }
        Ok(HostLayout {
            total_bytes: budget.physical_bytes,
            available_bytes: available,
            reserve_bytes: budget.headroom,
            host_budget_bytes: quota,
            resident_bytes: peak.saturating_sub(quota),
            deferred_bytes: 0,
            headroom_bytes: available.saturating_sub(peak),
            shortfall_bytes: peak.saturating_sub(available),
            items,
        })
    }
}

pub fn request(
    context: Context,
    pool: Option<u64>,
    onboard: crate::placement::Onboard,
    prefill: PrefillShape,
    operation: Operation,
    resident_weight_limit: Option<ResidentWeights>,
    replies: SparkReplyFormats,
) -> Request {
    Request {
        context,
        pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: match onboard {
            crate::placement::Onboard::Auto => Pin::Auto,
            crate::placement::Onboard::Layers(count) => Pin::Exact(count as u64),
        },
        prefill: Pin::Exact(prefill),
        operation,
        resident_weight_limit,
        spark_reply_formats: replies,
    }
}

pub fn spark_link(
    gpu: u8,
    ranks: usize,
    prefill: PrefillShape,
    replies: SparkReplyFormats,
) -> Result<Vec<crate::placement::PreparedSparkLink>, Refusal> {
    use crate::placement::{PreparedSparkLink, SparkEndpointOwners, SparkIntakeCandidates};
    if ranks == 0 {
        return Ok(Vec::new());
    }
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    Ok(vec![PreparedSparkLink::endpoint(
        "hy4_flash.spark",
        gpu,
        SparkEndpointOwners {
            ranks,
            geometry: cuteafd_core::ExpertGeometry::HY4_FLASH,
            endpoint: cuteafd_core::SparkEndpoint {
                ring_depth: None,
                capacity: u32::try_from(prefill.rows).map_err(|_| Refusal::Overflow("Hy4 transport rows"))?,
                ingress: cuteafd_core::SparkIngress::Fp8K32,
                partial_reply_bytes: None,
            },
            rings: cuteafd_core::SparkRings {
                max_frame_bytes: rings.max_frame_bytes,
                depth: rings.depth,
                slot_bytes: rings.slot_bytes,
                alignment: rings.alignment,
            },
        },
        usize::try_from(prefill.waves).map_err(|_| Refusal::Overflow("Hy4 transport waves"))?,
        true,
        true,
        SparkIntakeCandidates::Probed,
        replies.prefill(false),
        Lifetime::TARGET,
    )?])
}

impl Hy4Inputs {
    pub fn page_rows(&self) -> u64 {
        if self.modes.iter().any(|m| *m == LayerMode::ContextSplit) {
            2 * PAGE_ROWS as u64
        } else {
            PAGE_ROWS as u64
        }
    }

    pub fn maximum_pages(&self) -> u64 {
        self.pool.maximum_tokens().div_ceil(self.page_rows())
    }

    fn head_gpu(&self) -> u8 {
        if self.modes.iter().all(|m| matches!(m, LayerMode::HeadSplit | LayerMode::ContextSplit)) {
            0
        } else {
            (self.gpus - 1) as u8
        }
    }

    fn hop_spec(&self) -> HopSpec {
        let terminal = if self.modes.iter().all(|m| matches!(m, LayerMode::Whole { .. })) {
            TerminalHome::AttentionOwner
        } else {
            TerminalHome::Head
        };
        HopSpec {
            storage: HopStorage::ReceiveSlots,
            row_bytes: (self.cfg.hc_mult * self.cfg.hidden * 2) as u64,
            rows: self.prefill.rows.max(u64::from(self.decode_rows)),
            lanes: LANES,
            entry_gpu: 0,
            head_gpu: self.head_gpu(),
            terminal,
        }
    }

    pub fn counts(&self) -> Vec<usize> {
        (0..=self.routed.len()).collect()
    }

    pub fn admit(&self, request: Request) -> Result<AdmittedHy4, Refusal> {
        let solved = solve_prepared(request, self.budgets.clone(), self, &self.pool)?;
        let placement = self.execution(&solved)?;
        let onboard = placement.onboard_layers;
        Ok(AdmittedHy4 { placement, solved, onboard })
    }

    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Placement, Refusal> {
        let modes = self.modes.clone();
        if solved.decisions.attention != modes
            || solved.decisions.prefill != self.prefill
            || solved.decisions.lanes != LANES
        {
            return Err(invalid("resolved execution differs from prepared facts"));
        }
        let local: Vec<usize> = solved
            .decisions
            .onboard_layers
            .iter()
            .flatten()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let count = local.len();
        if local != self.routed[..count] {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let pair = self.gpus == 2;
        let peaks = self.arena_bytes(count)?;
        let layers = modes
            .iter()
            .enumerate()
            .map(|(layer, &mode)| LayerAssignment {
                mode,
                experts: if self.cfg.dense_layer[layer] {
                    ExpertHome::Dense
                } else if local.contains(&layer) {
                    if pair { ExpertHome::RtxTp2 } else { ExpertHome::RtxWhole { gpu: 0 } }
                } else {
                    ExpertHome::Spark
                },
            })
            .collect();
        let tp2 = (pair && count > 0).then(|| Tp2Range { first: local[0], layers: count, peak_bytes: peaks });
        let expert_ranges = (0..self.gpus)
            .map(|gpu| {
                if pair || count == 0 {
                    ExpertRange { first: 0, layers: 0, peak_bytes: 0 }
                } else {
                    ExpertRange { first: local[0], layers: count, peak_bytes: peaks[gpu] }
                }
            })
            .collect();
        let spec = self.hop_spec();
        let residual = residual_homes(&modes, &spec).map_err(invalid)?;
        let hops = plan_hops(&modes, &spec).map_err(invalid)?;
        let attention = if pair { self.placement } else { AttentionPlacement::Heads };
        let context = modes.iter().any(|m| *m == LayerMode::ContextSplit);
        let placement = Placement {
            pool_tokens: solved.pool_tokens,
            pool_limits: PoolLimits(vec![]),
            attention_placement: attention,
            attention_by_kind: vec![
                (AttentionClass::Swa, if context { AttentionPlacement::Heads } else { attention }),
                (AttentionClass::Mla, attention),
            ],
            peer_row_bytes: 0,
            onboard_layers: count,
            max_pool_tokens_no_onboard: None,
            layers,
            movables: Vec::new(),
            expert_ranges,
            residual,
            hops,
            tp2,
            items: (0..self.gpus)
                .map(|gpu| solved.items(Location::coordinator(gpu as u8)))
                .collect::<Result<_, _>>()?,
        };
        HY4_FLASH.check(&placement).map_err(invalid)?;
        Ok(placement)
    }

    fn arena_bytes(&self, count: usize) -> Result<[u64; 2], Refusal> {
        let mut out = [0u64; 2];
        if count == 0 {
            return Ok(out);
        }
        let LocalPackage::Installed { backings } = &self.local else {
            return Err(invalid("onboarded layers require installed Hy4-flash RTX expert backings"));
        };
        for backing in backings.get(count).ok_or_else(|| invalid("missing onboarding prefix"))? {
            let ws = backing.workspace.iter().try_fold(0u64, |sum, (_, b)| sum.checked_add(*b))
                .ok_or(Refusal::Overflow("Hy4 expert workspace"))?;
            out[backing.gpu] = out[backing.gpu]
                .checked_add(backing.arena.bytes() as u64)
                .and_then(|v| v.checked_add(ws))
                .ok_or(Refusal::Overflow("Hy4 local expert bytes"))?;
        }
        Ok(out)
    }

    fn rank_owners(&self, gpu: usize, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let location = Location::coordinator(gpu as u8);
        let cfg = &self.cfg;
        let rows = u32::try_from(self.prefill.rows).map_err(|_| Refusal::Overflow("Hy4 prefill rows"))?;
        let mut out = Vec::new();

        let target = self.targets.get(gpu).ok_or_else(|| invalid("missing target arena"))?;
        out.extend(target.weight_components(maximum)?);
        if !target.core.is_empty() {
            out.push(crate::shared::dense_read::trellis_device_staging(gpu as u8, maximum));
            if let Some(scratch) = target.trellis_scratch(rows as usize, self.decode_rows as usize).map_err(refusal)? {
                out.extend(buffer(
                    format!("hy4_flash.target.{gpu}.trellis"),
                    location,
                    scratch.packed_bytes().map_err(refusal)? as u64,
                    Lifetime::TARGET,
                    Category::Workspace,
                    "serialized target trellis projection",
                    maximum,
                ));
            }
        }

        // Owned DSA and SWA layers on this GPU.
        let mut owned_dsa = 0usize;
        let mut owned_swa = 0usize;
        for (layer, &mode) in self.modes.iter().enumerate() {
            let owns = match mode {
                LayerMode::Whole { gpu: owner, .. } => usize::from(owner) == gpu,
                LayerMode::HeadSplit | LayerMode::ContextSplit => true,
            };
            if !owns {
                continue;
            }
            if cfg.sparse_layer[layer] {
                owned_dsa += 1;
            } else {
                owned_swa += 1;
            }
        }

        let cache = CacheLayout::new(cfg, self.kv_format, self.sequences as usize, owned_dsa, owned_swa)
            .map_err(refusal)?;
        let context_split = self.modes.iter().any(|m| *m == LayerMode::ContextSplit);
        // Under ContextSplit, page_rows doubles (128) and each GPU stores half the pages (1 page = 64 rows/GPU).
        for (layer, &mode) in self.modes.iter().enumerate() {
            if !cfg.sparse_layer[layer] {
                continue;
            }
            let owns = match mode {
                LayerMode::Whole { gpu: owner, .. } => usize::from(owner) == gpu,
                LayerMode::HeadSplit | LayerMode::ContextSplit => true,
            };
            if !owns {
                continue;
            }
            let kv_stride = cache.dsa_kv_page_bytes as u64;
            let idx_stride = cache.dsa_index_page_bytes as u64;
            let _ = context_split;
            out.push(Component::allocation(
                Allocation::buffer(
                    format!("hy4_flash.kv.{layer}.{gpu}.records"),
                    location,
                    kv_stride,
                    Lifetime::TARGET,
                    Category::Kv,
                    "DSA MLA paged KV records",
                ),
                ByteCurve::linear_rounded(kv_stride, kv_stride, maximum, SM120_ALLOCATION_GRANULARITY_BYTES),
                false,
                maximum,
            ));
            out.push(Component::allocation(
                Allocation::buffer(
                    format!("hy4_flash.index.{layer}.{gpu}.groups"),
                    location,
                    idx_stride,
                    Lifetime::TARGET,
                    Category::Kv,
                    "DSA compressed-indexer group cache",
                ),
                ByteCurve::linear_rounded(idx_stride, idx_stride, maximum, SM120_ALLOCATION_GRANULARITY_BYTES),
                false,
                maximum,
            ));
        }

        // Fixed per-sequence SWA-512 rings + pending indexer tokens + exact retained prefix marks.
        out.extend(buffer(
            format!("hy4_flash.cache.{gpu}.swa_rings"),
            location,
            cache.active_swa_ring_bytes as u64,
            Lifetime::TARGET,
            Category::Kv,
            "active SWA-512 circular rings",
            maximum,
        ));
        out.extend(buffer(
            format!("hy4_flash.cache.{gpu}.indexer_pending"),
            location,
            cache.active_indexer_pending_bytes as u64,
            Lifetime::TARGET,
            Category::Kv,
            "pending incomplete compressed-indexer token keys",
            maximum,
        ));
        let mark_bytes = (cache.mark_bytes_per_slot as u64)
            .checked_mul(self.prefix_mark_slots as u64)
            .ok_or(Refusal::Overflow("Hy4 prefix marks"))?;
        out.extend(buffer(
            format!("hy4_flash.cache.{gpu}.marks"),
            location,
            mark_bytes,
            Lifetime::TARGET,
            Category::Prefix,
            "exact retained SWA-512 and indexer prefix marks",
            maximum,
        ));

        // Decode + prefill unit workspaces.
        let head_parts = if self.modes.iter().all(|m| matches!(m, LayerMode::HeadSplit | LayerMode::ContextSplit)) {
            2
        } else {
            1
        };
        out.extend(buffer(
            format!("hy4_flash.decode.{gpu}.mla_split_partials"),
            location,
            crate::families::hy4_flash::execution::mla_split_scratch_bytes(self.decode_rows as usize, head_parts) as u64,
            Lifetime::TARGET,
            Category::Workspace,
            "Hy4-flash MLA split-K decode/verify partials",
            maximum,
        ));
        let decode_unit = UnitLayout::new(cfg, self.kv_format, self.decode_rows as usize, head_parts).map_err(refusal)?;
        for (name, bytes) in decode_unit.buffers {
            out.extend(buffer(
                format!("hy4_flash.decode.{gpu}.{name}"),
                location,
                bytes as u64,
                Lifetime::TARGET,
                Category::Workspace,
                "Hy4-flash decode unit workspace",
                maximum,
            ));
        }
        let prefill_unit = UnitLayout::new(cfg, self.kv_format, rows as usize, head_parts).map_err(refusal)?;
        for slot in 0..self.prefill.waves {
            for (name, &bytes) in &prefill_unit.buffers {
                out.extend(buffer(
                    format!("hy4_flash.unit.{gpu}.{slot}.{name}"),
                    location,
                    bytes as u64,
                    Lifetime::TARGET,
                    Category::Workspace,
                    "Hy4-flash prefill unit workspace",
                    maximum,
                ));
            }
        }

        // Canonical graph reserve, loaded code, and driver stream slabs.
        out.extend(allowance(
            location,
            self.graph_reserve[gpu],
            Lifetime { first: Phase::Warmup, last: Phase::Growth },
            crate::placement::graph_reserve::source(&format!("hy4_flash.graphs.{gpu}")),
            true,
            maximum,
        ));
        out.extend(allowance(
            location,
            self.code[gpu],
            Lifetime::RESIDENT,
            "Hy4-flash loaded program code (estimated module residency)".into(),
            true,
            maximum,
        ));
        let peak_streams = 1 + crate::push_load::COPY_STREAMS as u64 + 1;
        out.extend(allowance(
            location,
            cuda_stream_slab_bytes(peak_streams)?,
            Lifetime::RESIDENT,
            format!("hy4_flash/driver-stream-slabs/gpu{gpu} ({peak_streams} peak live streams)"),
            false,
            maximum,
        ));
        Ok(out)
    }

    fn head_owners(&self, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let head = self.head_gpu();
        let rows = u64::from(self.decode_rows);
        let mut out = Vec::new();
        for (name, bytes) in [
            ("tokens", rows * 4),
            ("normalized", rows * self.cfg.hidden as u64 * 2),
            ("logits_f32", rows * self.cfg.vocab_size as u64 * 4),
            ("workspace", cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE as u64),
        ] {
            out.extend(buffer(
                format!("hy4_flash.head.{name}"),
                Location::coordinator(head),
                bytes,
                Lifetime::TARGET,
                Category::Workspace,
                "Hy4-flash FP32 vocabulary head rows",
                maximum,
            ));
        }
        Ok(out)
    }
}

impl Recipe for Hy4Inputs {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        let maximum = self.maximum_pages();
        if !(1..=2).contains(&self.gpus)
            || self.spark_ranks > 8
            || maximum == 0
            || maximum == u64::MAX
            || request.context.maximum != self.context.maximum
            || self.targets.len() != self.gpus
            || self.modes.len() != self.cfg.layers
            || self.graph_reserve.len() != self.gpus
            || self.code.len() != self.gpus
            || self.spark.len() != self.routed.len() + 1
            || self.prefill.rows == 0
            || self.prefill.waves == 0
            || !(1..=DECODE_ROWS).contains(&self.decode_rows)
            || self.sequences > u64::from(self.decode_rows)
            || self.sequences == 0
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        let pair = self.gpus == 2;
        let head = self.head_gpu();
        let mut components = Vec::new();
        for gpu in 0..self.gpus {
            components.extend(self.rank_owners(gpu, maximum)?);
        }
        components.extend(self.head_owners(maximum)?);
        components.extend(
            serving::ServingOwners {
                selector: serving::SelectorOwners {
                    location: Location::coordinator(head),
                    capacity: usize::try_from(self.sequences)
                        .map_err(|_| Refusal::Overflow("Hy4 selector"))?
                        .max(self.decode_rows as usize),
                    vocab: self.cfg.vocab_size,
                    device_sampling: true,
                },
                prefix: self.host_prefix,
            }
            .components(maximum)?,
        );
        // Real measured host Engram owners (never GPU memory):
        // 1. Preloaded tables via MappedPlacement (`mapped-tables/preloaded (coordinator local)`)
        // 2. Shared table-read reactor workspace (`table_reactor_workspace_bytes`) for any mapped tables
        // 3. Per-layer `RowCache` (`Hy4EngramShape::cache_bytes`)
        // 4. Pinned prefill/decode staging pools (`Hy4EngramShape::staging_bytes`)
        components.extend(buffer(
            "hy4_flash.engram.preloaded",
            Location::HOST,
            self.engram_mapped.preloaded_bytes,
            Lifetime::RESIDENT,
            Category::Tables,
            "mapped-tables/preloaded (coordinator local)",
            maximum,
        ));
        components.extend(buffer(
            "hy4_flash.engram.reactor",
            Location::HOST,
            crate::formats::mapped_table::table_reactor_workspace_bytes(self.engram_mapped.mapped) as u64,
            Lifetime::RESIDENT,
            Category::Workspace,
            "workspace/table-read-reactor",
            maximum,
        ));
        components.extend(buffer(
            "hy4_flash.engram.row_cache",
            Location::HOST,
            self.engram_row_cache_bytes,
            Lifetime::RESIDENT,
            Category::Workspace,
            "hy4_flash/engram-row-cache",
            maximum,
        ));
        components.extend(buffer(
            "hy4_flash.engram.staging",
            Location::PINNED,
            self.engram_staging_bytes,
            Lifetime::RESIDENT,
            Category::Staging,
            "hy4_flash/engram-staging",
            maximum,
        ));

        let mut contract_variables = Vec::new();
        for (layer, &mode) in self.modes.iter().enumerate() {
            contract_variables.push(Variable::Attention { layer, values: vec![mode] });
        }
        contract_variables.push(Variable::Prefill(vec![self.prefill]));
        contract_variables.push(Variable::Lanes(vec![LANES]));

        let spec = self.hop_spec();
        let hops = plan_hops(&self.modes, &spec).map_err(invalid)?;
        let hop_bytes = crate::placement::hop_buffer_bytes(&hops, &spec, self.gpus)
            .ok_or(Refusal::Overflow("Hy4 residual hop buffers"))?;
        for (gpu, &bytes) in hop_bytes.iter().enumerate() {
            components.extend(buffer(
                format!("hy4_flash.hops.{gpu}"),
                Location::coordinator(gpu as u8),
                bytes,
                Lifetime::TARGET,
                Category::Transport,
                "iHC residual hops",
                maximum,
            ));
        }

        let counts = self.counts();
        let onboard = contract_variables.len();
        contract_variables.push(Variable::Onboard {
            layers: self.routed.clone(),
            destination: if pair { vec![0, 1] } else { vec![0] },
            counts: counts.clone(),
        });
        if self.spark_ranks > 0 {
            contract_variables.push(Variable::ExpertHome {
                layers: self.routed.clone(),
                rank_sets: vec![(0..self.spark_ranks as u8).collect()],
            });
        }

        let mut requires = Vec::new();
        for (option, &count) in counts.iter().enumerate() {
            let activation = Activation::choice(onboard, option);
            let local = count == 0 || matches!(self.local, LocalPackage::Installed { .. });
            let remote = count == self.routed.len() || self.spark_ranks > 0;
            let linked = self.spark_ranks == 0 || count < self.routed.len();
            if !local || !remote || !linked {
                requires.push(Requires { when: Choice { variable: onboard, option }, any_of: vec![] });
                continue;
            }
            if count > 0 {
                let LocalPackage::Installed { backings } = &self.local else { unreachable!() };
                for (group, backing) in backings
                    .get(count)
                    .ok_or_else(|| invalid("missing onboarding prefix"))?
                    .iter()
                    .enumerate()
                {
                    let location = Location::coordinator(backing.gpu as u8);
                    components.extend(with(
                        weight_arena_components(
                            &backing.arena,
                            location,
                            Lifetime::RESIDENT,
                            maximum,
                            MemoryKind::Experts,
                        )?,
                        &activation,
                    ));
                    if let Some(rows) = backing.tp2_rows {
                        let peer = crate::serving_capacity::PeerStorageLayout::new(
                            2,
                            usize::try_from(rows.exchange_slot_bytes)
                                .map_err(|_| Refusal::Overflow("Hy4 expert peer slot"))?,
                            crate::serving_capacity::peer_wait_diagnostics(),
                        )
                        .map_err(refusal)?;
                        for (name, bytes) in [
                            ("send", rows.bytes),
                            ("reduced", rows.reduction_bytes),
                            ("peer.recv.0", peer.receive_bytes as u64),
                            ("peer.recv.1", peer.receive_bytes as u64),
                            ("peer.control", peer.control_bytes as u64),
                            ("peer.progress", peer.progress_bytes as u64),
                            ("peer.abort", 256),
                        ] {
                            components.extend(with(
                                buffer(
                                    format!("hy4_flash.tp2.{group}.{name}"),
                                    location,
                                    bytes,
                                    Lifetime::TARGET,
                                    Category::Transport,
                                    "shared resident expert exchange",
                                    maximum,
                                )
                                .into_iter()
                                .collect(),
                                &activation,
                            ));
                        }
                    }
                    for (name, bytes) in &backing.workspace {
                        components.extend(with(
                            buffer(
                                format!("hy4_flash.expert_workspace.{}.{}.{}.{}", backing.gpu, count, group, name),
                                location,
                                *bytes,
                                Lifetime::RESIDENT,
                                Category::Workspace,
                                "Hy4-flash RTX expert workspace",
                                maximum,
                            )
                            .into_iter()
                            .collect(),
                            &activation,
                        ));
                    }
                }
            }
            if count < self.routed.len() {
                for (rank, charge) in self.spark[count].iter().enumerate() {
                    let location = Location { device: Device::Spark(rank as u8), domain: Domain::Managed };
                    for (kind, group, bytes, category) in [
                        ("experts", "routed experts (header-exact intermediate slices)", charge.experts, Category::Experts),
                        ("owners", "Spark rank I/O and execution owners", charge.owners, Category::Workspace),
                        ("startup", cuteafd_core::memory_layout::spark_runtime_reserve_group(), charge.runtime, Category::Runtime),
                    ] {
                        components.extend(
                            buffer(
                                format!("hy4_flash.spark.{rank}.{kind}.{count}"),
                                location,
                                bytes,
                                Lifetime::RESIDENT,
                                category,
                                group,
                                maximum,
                            )
                            .map(|mut c| {
                                c.activation = activation.clone();
                                c
                            }),
                        );
                    }
                    if charge.experts > 0 {
                        components.push(Component {
                            activation: activation.clone(),
                            bytes: ByteCurve::fixed(charge.experts, maximum),
                            kind: ComponentKind::DriverPageTables {
                                model: DriverPageTableModel::Sm121Arena,
                                location,
                                lifetime: Lifetime::RESIDENT,
                                backing: format!("hy4_flash.spark.{rank}.experts.{count}"),
                            },
                        });
                    }
                }
                for link in &self.links {
                    components.extend(with(link.components(maximum)?, &activation));
                }
            }
        }

        fixed_request_tails(&mut components, maximum, &[])?;
        let mut contract = Contract {
            floor_fallback: None,
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: (self.spark_ranks > 0).then(|| SparkReplyFormats {
                prefill: Some(request.spark_reply_formats.prefill(false)),
                decode: Some(request.spark_reply_formats.decode()),
            }),
            resident_weight_limit: request.resident_weight_limit,
            page_rows: self.page_rows(),
            pool_page_multiple: 1,
            maximum_pages: maximum,
            gpus: self.gpus,
            variables: contract_variables,
            requires,
            components,
        };
        contract.add_driver_page_tables();
        Ok(contract)
    }
}

pub fn planned_budget(gpu: u8, physical: u64, logical: u64, context: u64) -> Budget {
    Budget {
        id: format!("rtx{gpu}"),
        locations: vec![Location::coordinator(gpu)],
        physical_bytes: physical,
        logical_bytes: logical,
        headroom: crate::placement::reserve::gpu("hy4_flash"),
        baseline: Baseline::Forecast {
            occupied: context,
            source: "CUDA context and modules on the coordinator GPU".into(),
            assumed: false,
        },
    }
}

pub fn spark_budget(rank: u8, bytes: u64) -> Budget {
    Budget {
        id: format!("spark{rank}"),
        locations: vec![Location { device: Device::Spark(rank), domain: Domain::Managed }],
        physical_bytes: bytes,
        logical_bytes: bytes,
        headroom: 0,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "Spark worker arena owned by the serving process".into(),
            assumed: false,
        },
    }
}
