//! Inkling 2 Mini (`inkling2_mini`) physical owners for the shared contract planner (TP1, 0 Sparks).
//!
//! Both `cuteafd plan --layout` and serving startup prepare one [`MiniInputs`] and
//! submit it to [`planner::solve_prepared`]. Every owner is named once:
//! - target packed arena on GPU0 (globals, 24 layers, 144 controllers, cross-pass queries, padded rel proj);
//! - routed expert arenas on GPU0: homogeneous NVFP4 (layers 0..20) and MXFP8 (layers 20..24),
//!   each with its installed package workspace;
//! - 8 global paged KV planes (`slots [5, 11, 17, 23, 29, 35, 41, 47]`) + 40 SWA-1024 active rings
//!   + 48-slot contiguous FP32 short-conv bank + exact retained prefix marks;
//! - decode (64 rows) and per-wave prefill `MiniUnitLayout` scratch owners + pinned uploads;
//! - vocabulary head, device selector, BLAS, `#2697` working-set graph reserve, loaded code, stream slabs;
//! - optional bundled 3-layer recurrent dSpark drafter (own arena) and 23-layer PE-Lang vision tower (own arena).
use super::super::planner::*;
use super::super::{
    plan_hops, residual_homes, AttentionClass, AttentionPlacement, ExecutorModes, ExpertHome, ExpertRange,
    FfnMode, HopSpec, HopStorage, LayerAssignment, LayerMode, MovableId, Placement, PoolLimits,
    ResidentWeights, TerminalHome,
};
use crate::families::inkling2::execution::SequenceLayout;
use crate::families::inkling2_mini::{
    execution::MiniUnitLayout,
    weight_layout::{MiniLayerExperts, MiniTargetWeights},
    Inkling2MiniConfig, InklingKvCache,
};
use crate::serving_capacity::WeightArenaLayout;
use cuteafd_core::memory_layout::{Category, MemoryKind};

pub const PAGE_ROWS: u64 = 64;
pub const LANES: u64 = 1;
/// Default decode/verify rows per step (`DECODE_ROWS=auto`).
pub const DECODE_ROWS: u32 = 64;
/// Maximum decode/verify rows per step supported by exported Inkling 2 Mini decode programs.
pub const MAX_DECODE_ROWS: u32 = 128;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };

/// TP1-only executor modes for Inkling 2 Mini.
pub const INKLING2_MINI: ExecutorModes = ExecutorModes {
    family: "inkling2_mini",
    modes: &[WHOLE],
    hops: false,
    attention: AttentionPlacement::Heads,
};

fn invalid(detail: impl std::fmt::Display) -> Refusal {
    Refusal::InvalidFact(format!("Inkling 2 Mini component contract: {detail}"))
}

fn refusal(error: anyhow::Error) -> Refusal {
    invalid(format!("{error:#}"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalBacking {
    pub gpu: usize,
    pub layers: Vec<usize>,
    pub arena: WeightArenaLayout,
    pub workspace: Vec<(String, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPackage {
    Installed { backings: Vec<LocalBacking> },
    Missing(String),
}

#[derive(Debug, Clone)]
pub struct Drafter {
    pub layout: crate::families::inkling2::draft::DraftLayout,
}

pub type Vision = crate::placement::still_encoder::StillEncoder;

/// Complete owner facts shared by `cuteafd plan --layout` and serving startup.
#[derive(Debug, Clone)]
pub struct MiniInputs {
    pub cfg: Inkling2MiniConfig,
    pub kv_format: InklingKvCache,
    pub context: Context,
    pub pool: crate::placement::PoolPolicy,
    pub prefill: PrefillShape,
    /// Admitted per-lane decode/verify row budget (`DECODE_ROWS`, 1..=128).
    pub decode_rows: u32,
    pub sequences: u64,
    pub prefix_entries: u64,
    pub prefix_mark_budget: u64,
    pub prefix_mark_slots: usize,
    pub target: MiniTargetWeights,
    pub routed: Vec<usize>,
    pub experts: Vec<MiniLayerExperts>,
    pub local: LocalPackage,
    pub drafter: Option<Drafter>,
    pub dflash: Option<crate::families::inkling2_mini::dflash::Layout>,
    pub vision: Option<Vision>,
    pub budgets: Vec<Budget>,
    pub code: u64,
    pub graph_reserve: u64,
    pub head: serving::BlasHead,
    pub host_prefix: serving::HostPrefixOwners,
    pub device_sampling: bool,
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub resident_weight_limit: Option<ResidentWeights>,
    pub mount: crate::shared::dense_read::MountLoad,
    pub push_required: bool,
}

#[derive(Debug, Clone)]
pub struct AdmittedMini {
    pub placement: Placement,
    pub solved: ResolvedPlan,
    pub draft_home: Option<u8>,
    pub encoder: Option<usize>,
}

pub fn request(
    context: Context,
    pool: Option<u64>,
    routed_layers: usize,
    prefill: PrefillShape,
    operation: Operation,
    resident_weight_limit: Option<ResidentWeights>,
) -> Request {
    Request {
        context,
        pool: crate::placement::PoolPolicy::normalize_request(pool).map_or(Pin::Auto, Pin::Exact),
        onboard: Pin::Exact(routed_layers as u64),
        prefill: Pin::Exact(prefill),
        operation,
        resident_weight_limit,
        spark_reply_formats: Default::default(),
    }
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
        kind: ComponentKind::Allowance(Allowance {
            location,
            bytes,
            lifetime,
            source,
            assumed,
        }),
    })
}

impl MiniInputs {
    pub fn maximum_pages(&self) -> u64 {
        self.pool.maximum_tokens().div_ceil(PAGE_ROWS)
    }

    pub fn modes(&self) -> Vec<LayerMode> {
        vec![WHOLE; self.cfg.text.layers]
    }

    fn hop_spec(&self) -> HopSpec {
        HopSpec {
            storage: HopStorage::ReceiveSlots,
            row_bytes: (self.cfg.text.residual_blocks() as u64 + 1) * self.cfg.text.hidden as u64 * 4,
            rows: self.prefill.rows.max(u64::from(self.decode_rows)),
            lanes: LANES,
            entry_gpu: 0,
            head_gpu: 0,
            terminal: TerminalHome::Head,
        }
    }

    pub fn admit(&self, req: Request) -> Result<AdmittedMini, Refusal> {
        let solved = solve_prepared(req, self.budgets.clone(), self, &self.pool)?;
        let placement = self.execution(&solved)?;
        Ok(AdmittedMini {
            placement,
            draft_home: solved.decisions.draft_home,
            encoder: solved.decisions.encoder,
            solved,
        })
    }

    pub fn execution(&self, solved: &ResolvedPlan) -> Result<Placement, Refusal> {
        let modes = self.modes();
        if solved.decisions.attention != modes
            || solved.decisions.prefill != self.prefill
            || solved.decisions.lanes != LANES
        {
            return Err(invalid("resolved execution differs from prepared facts"));
        }
        let local = solved
            .decisions
            .onboard_layers
            .iter()
            .flatten()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if local != self.routed {
            return Err(invalid("Inkling 2 Mini requires all 24 routed layers resident on GPU0"));
        }
        let peak = self.arena_bytes()?;
        let layers = modes
            .iter()
            .map(|&mode| LayerAssignment {
                mode,
                experts: ExpertHome::RtxWhole { gpu: 0 },
            })
            .collect();
        let spec = self.hop_spec();
        let residual = residual_homes(&modes, &spec).map_err(invalid)?;
        let hops = plan_hops(&modes, &spec).map_err(invalid)?;
        let placement = Placement {
            pool_tokens: solved.pool_tokens,
            pool_limits: PoolLimits(vec![]),
            attention_placement: AttentionPlacement::Heads,
            attention_by_kind: vec![
                (AttentionClass::Swa, AttentionPlacement::Heads),
                (AttentionClass::Gqa, AttentionPlacement::Heads),
            ],
            peer_row_bytes: 0,
            onboard_layers: self.routed.len(),
            max_pool_tokens_no_onboard: None,
            layers,
            movables: solved
                .decisions
                .draft_home
                .map(|gpu| (MovableId::Drafter, gpu))
                .into_iter()
                .collect(),
            expert_ranges: vec![ExpertRange {
                first: 0,
                layers: self.routed.len(),
                peak_bytes: peak,
            }],
            residual,
            hops,
            tp2: None,
            items: vec![solved.items(Location::coordinator(0))?],
        };
        INKLING2_MINI.check(&placement).map_err(invalid)?;
        Ok(placement)
    }

    fn arena_bytes(&self) -> Result<u64, Refusal> {
        let LocalPackage::Installed { backings } = &self.local else {
            return Err(invalid("Inkling 2 Mini requires installed NVFP4 and MXFP8 RTX expert packages"));
        };
        let mut total = 0u64;
        for backing in backings {
            let workspace = backing
                .workspace
                .iter()
                .try_fold(0u64, |sum, (_, bytes)| sum.checked_add(*bytes))
                .ok_or(Refusal::Overflow("Inkling 2 Mini expert workspace"))?;
            total = total
                .checked_add(backing.arena.bytes() as u64)
                .and_then(|bytes| bytes.checked_add(workspace))
                .ok_or(Refusal::Overflow("Inkling 2 Mini local backend"))?;
        }
        Ok(total)
    }

    fn unit_owners(&self, prefix: &str, rows: u32, decode: bool, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let location = Location::coordinator(0);
        let context = usize::try_from(self.context.maximum).map_err(|_| Refusal::Overflow("Mini context"))?;
        let layout = if decode {
            MiniUnitLayout::serving_decode_rows(&self.cfg, rows as usize, context, self.kv_format, self.sequences as usize)
        } else {
            MiniUnitLayout::serving(&self.cfg, rows as usize, context, self.kv_format, self.sequences as usize)
        }
        .map_err(refusal)?;
        let mut out = Vec::new();
        let uploads = layout
            .upload_shape(
                &self.cfg,
                usize::try_from(self.sequences)
                    .map_err(|_| Refusal::Overflow("Mini upload members"))?
                    .min(rows as usize),
            )
            .map_err(refusal)?;
        out.extend(buffer(
            format!("{prefix}.uploads"),
            Location::PINNED,
            uploads.capacity_bytes().map_err(refusal)? as u64,
            Lifetime::TARGET,
            Category::Staging,
            "shared completion-held metadata and token sources",
            maximum,
        ));
        for (name, bytes) in layout.buffers {
            out.extend(buffer(
                format!("{prefix}.{name}"),
                location,
                bytes as u64,
                Lifetime::TARGET,
                Category::Workspace,
                "Inkling 2 Mini native forward UnitLayout owner",
                maximum,
            ));
        }
        Ok(out)
    }

    fn rank_owners(&self, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let location = Location::coordinator(0);
        let cfg = &self.cfg;
        let t = &cfg.text;
        let rows = u32::try_from(self.prefill.rows).map_err(|_| Refusal::Overflow("Mini unit rows"))?;
        let sequences = self.sequences;
        let mut out = Vec::new();
        out.extend(self.target.weight_components(maximum)?);

        // 48 distinct KV + short-conv cache slots: 8 global paged planes + 40 SWA-1024 rings + 48x3 conv lags.
        let record_bytes = self.kv_format.record_bytes(t.kv_heads, t.head_dim) as u64;
        let stride = PAGE_ROWS * record_bytes;
        let mut rings = 0u64;
        let mut conv = 0u64;
        for slot in 0..cfg.cache_slots() {
            let local = cfg.cache_slot_is_local(slot);
            if !local {
                out.push(Component::allocation(
                    Allocation::buffer(
                        format!("inkling2_mini.kv.{slot}.0.records"),
                        location,
                        stride,
                        Lifetime::TARGET,
                        Category::Kv,
                        "global-slot native KV records and scales",
                    ),
                    ByteCurve::linear_rounded(stride, stride, maximum, SM120_ALLOCATION_GRANULARITY_BYTES),
                    false,
                    maximum,
                ));
            }
            let layout = SequenceLayout::new(t, local, 1).map_err(refusal)?;
            let ring = layout.fields["ring"].len() as u64;
            let lags = (layout.bytes - ring as usize) as u64;
            rings = rings
                .checked_add(ring.checked_mul(sequences).ok_or(Refusal::Overflow("Mini ring state"))?)
                .ok_or(Refusal::Overflow("Mini ring state"))?;
            conv = conv
                .checked_add(lags.checked_mul(sequences).ok_or(Refusal::Overflow("Mini conv state"))?)
                .ok_or(Refusal::Overflow("Mini conv state"))?;
        }
        out.extend(buffer(
            "inkling2_mini.cache.0.active",
            location,
            rings,
            Lifetime::TARGET,
            Category::Kv,
            "active 40-slot SWA-1024 rings",
            maximum,
        ));
        out.extend(buffer(
            "inkling2_mini.cache.0.conv",
            location,
            conv,
            Lifetime::TARGET,
            Category::Kv,
            "contiguous 48-slot FP32 convolution lag backing",
            maximum,
        ));
        let per_sequence = rings
            .checked_add(conv)
            .ok_or(Refusal::Overflow("Mini complete sequence state"))?
            / sequences;
        let retained = per_sequence
            .checked_mul(self.prefix_mark_slots as u64)
            .ok_or(Refusal::Overflow("Mini prefix marks"))?;
        out.extend(buffer(
            "inkling2_mini.cache.0.marks",
            location,
            retained,
            Lifetime::TARGET,
            Category::Prefix,
            "exact retained 48-slot SWA and FP32 convolution marks",
            maximum,
        ));

        out.extend(self.unit_owners("inkling2_mini.decode.0", self.decode_rows, true, maximum)?);
        for slot in 0..self.prefill.waves {
            out.extend(self.unit_owners(&format!("inkling2_mini.unit.0.{slot}"), rows, false, maximum)?);
        }

        out.extend(allowance(
            location,
            self.graph_reserve,
            Lifetime {
                first: Phase::Warmup,
                last: Phase::Growth,
            },
            crate::placement::graph_reserve::source("inkling2_mini.graphs.0"),
            true,
            maximum,
        ));
        out.extend(allowance(
            location,
            self.code,
            Lifetime::RESIDENT,
            "Inkling 2 Mini selected-program loaded code (estimated module residency)".into(),
            true,
            maximum,
        ));
        let peak = 1 + crate::push_load::COPY_STREAMS as u64 + 1;
        out.extend(allowance(
            location,
            cuda_stream_slab_bytes(peak)?,
            Lifetime::RESIDENT,
            format!("inkling2_mini/driver-stream-slabs/gpu0 ({peak} peak live streams: model + push copies + conv)"),
            false,
            maximum,
        ));
        if !self.push_required {
            for owner in self.mount.allocations("inkling2_mini.mount.0") {
                let bytes = ByteCurve::fixed(owner.bytes, maximum);
                out.push(Component::allocation(owner, bytes, false, maximum));
            }
        }
        Ok(out)
    }

    fn head_owners(&self, maximum: u64) -> Result<Vec<Component>, Refusal> {
        let t = &self.cfg.text;
        let rows = u64::from(self.decode_rows);
        let location = Location::coordinator(0);
        let mut out = Vec::new();
        for (name, bytes) in [
            ("tokens", rows * 4),
            ("normalized", rows * t.hidden as u64 * 2),
            ("logits", rows * t.unpadded_vocab_size as u64 * 4),
            ("workspace", cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE as u64),
        ] {
            out.extend(buffer(
                format!("inkling2_mini.head.{name}"),
                location,
                bytes,
                Lifetime::TARGET,
                Category::Workspace,
                "Inkling 2 Mini bounded vocabulary head rows",
                maximum,
            ));
        }
        Ok(out)
    }
}

impl Recipe for MiniInputs {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        let t = &self.cfg.text;
        let maximum = self.maximum_pages();
        if maximum == 0
            || maximum == u64::MAX
            || request.context.maximum != self.context.maximum
            || self.experts.len() != self.routed.len()
            || self.prefill.rows == 0
            || self.prefill.waves == 0
            || !(1..=MAX_DECODE_ROWS).contains(&self.decode_rows)
            || self.sequences == 0
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        let mut components = self.rank_owners(maximum)?;
        components.extend(self.head_owners(maximum)?);
        components.extend(
            serving::ServingOwners {
                selector: serving::SelectorOwners {
                    location: Location::coordinator(0),
                    capacity: usize::try_from(self.sequences)
                        .map_err(|_| Refusal::Overflow("Mini selector"))?
                        .max(self.decode_rows as usize),
                    vocab: t.unpadded_vocab_size,
                    device_sampling: self.device_sampling,
                },
                prefix: self.host_prefix,
            }
            .components(maximum)?,
        );
        if let Some(geometry) = &self.expert_usage {
            components.extend(crate::placement::expert_usage::components(geometry, &[0], maximum)?);
        }

        let mut contract_variables = Vec::new();
        for layer in 0..t.layers {
            contract_variables.push(Variable::Attention {
                layer,
                values: vec![WHOLE],
            });
        }
        contract_variables.push(Variable::Prefill(vec![self.prefill]));
        contract_variables.push(Variable::Lanes(vec![LANES]));

        let onboard = contract_variables.len();
        contract_variables.push(Variable::Onboard {
            layers: self.routed.clone(),
            destination: vec![0],
            counts: vec![self.routed.len()],
        });

        let mut requires = Vec::new();
        match &self.local {
            LocalPackage::Installed { backings } => {
                for (group, backing) in backings.iter().enumerate() {
                    let location = Location::coordinator(backing.gpu as u8);
                    components.extend(weight_arena_components(
                        &backing.arena,
                        location,
                        Lifetime::RESIDENT,
                        maximum,
                        MemoryKind::Experts,
                    )?);
                    for (name, bytes) in &backing.workspace {
                        components.extend(buffer(
                            format!("inkling2_mini.expert_workspace.{}.{group}.{name}", backing.gpu),
                            location,
                            *bytes,
                            Lifetime::RESIDENT,
                            Category::Workspace,
                            "installed homogeneous expert backend request",
                            maximum,
                        ));
                    }
                }
            }
            LocalPackage::Missing(_) => {
                requires.push(Requires {
                    when: Choice {
                        variable: onboard,
                        option: 0,
                    },
                    any_of: vec![],
                });
            }
        }

        if let Some(draft) = &self.drafter {
            contract_variables.push(Variable::DraftHome(vec![0]));
            let lifetime = Lifetime {
                first: Phase::DraftLoad,
                last: Phase::Growth,
            };
            if draft.layout.weights.len() != 1 || draft.layout.runtime.len() != 1 {
                return Err(invalid("Mini drafter must be single-rank TP1"));
            }
            let location = Location::coordinator(0);
            let weights = &draft.layout.weights[0];
            let work = &draft.layout.runtime[0];
            components.extend(weight_arena_components(
                &weights.arena,
                location,
                lifetime,
                maximum,
                MemoryKind::DrafterWeights,
            )?);
            for (name, bytes) in &work.buffers {
                let full = draft.layout.config.context.iter().enumerate().any(|(layer, policy)| {
                    matches!(policy, crate::formats::block_dspark::recurrent::ContextPolicy::Full)
                        && name.starts_with(&format!("cache.layer.{layer}."))
                });
                let id = format!("inkling2_mini.draft.0.{name}");
                if full {
                    let stride = PAGE_ROWS * work.cache.record_bytes as u64 / 2;
                    components.push(Component::allocation(
                        Allocation::buffer(id, location, stride, lifetime, Category::Drafter, "recurrent full-context KV plane"),
                        ByteCurve::linear(stride, stride, maximum),
                        false,
                        maximum,
                    ));
                } else {
                    components.extend(buffer(
                        id,
                        location,
                        *bytes as u64,
                        lifetime,
                        Category::Drafter,
                        "recurrent fixed owner",
                        maximum,
                    ));
                }
            }
            if self.prefix_mark_slots > 0 {
                let bytes = work
                    .window_mark_bytes
                    .checked_mul(self.prefix_mark_slots)
                    .ok_or(Refusal::Overflow("Mini draft marks"))? as u64;
                components.extend(buffer(
                    "inkling2_mini.draft.0.marks",
                    location,
                    bytes,
                    lifetime,
                    Category::Drafter,
                    "exact drafter window and last-feature marks",
                    maximum,
                ));
            }
            for (name, bytes) in &draft.layout.target {
                components.extend(buffer(
                    format!("inkling2_mini.draft.target.{name}"),
                    location,
                    *bytes as u64,
                    lifetime,
                    Category::Drafter,
                    "target tap and accepted-prefix FIR owner",
                    maximum,
                ));
            }
        }

        if let Some(draft) = &self.dflash {
            contract_variables.push(Variable::DraftHome(vec![0]));
            let lifetime = Lifetime { first: Phase::DraftLoad, last: Phase::Growth };
            let location = Location::coordinator(0);
            components.extend(weight_arena_components(&draft.arena, location, lifetime, maximum,
                MemoryKind::DrafterWeights)?);
            let weights = draft.physical.weight_owners().map(|a| &a.id).collect::<std::collections::BTreeSet<_>>();
            for owner in draft.physical.owners().iter().filter(|a| !weights.contains(&a.id)) {
                components.extend(buffer(format!("inkling2_mini.dflash.{}", owner.id), owner.location,
                    owner.bytes, lifetime, owner.category, "DFlash2 fixed owner", maximum));
            }
            components.extend(buffer("inkling2_mini.dflash.marks", location,
                (draft.mark_bytes * self.prefix_mark_slots) as u64, lifetime, Category::Drafter,
                "exact DFlash2 ring marks", maximum));
            for (name, bytes) in &draft.target {
                components.extend(buffer(format!("inkling2_mini.draft.target.{name}"), location,
                    *bytes as u64, lifetime, Category::Drafter, "target taps and FIR", maximum));
            }
        }

        if let Some(vision) = &self.vision {
            vision.extend(&mut contract_variables, &mut components, maximum)?;
        }

        if self.head.bf16() {
            components.extend(serving::blas_components(
                "sm_120",
                &[serving::BlasOwners {
                    name: "Inkling 2 Mini target head".into(),
                    process: "coordinator".into(),
                    thread: "inkling2_mini/blas/serve".into(),
                    location: Location::coordinator(0),
                    gemm: self.head.gemm(),
                    lt: self.head.lt(),
                    lifetime: Lifetime::TARGET,
                    activation: Activation::always(),
                }],
                maximum,
                PhaseOrder::TARGET_THEN_DRAFT,
            )?);
        }

        if !self.push_required {
            components.extend(allowance(
                Location::PINNED,
                64 << 20,
                Lifetime::RESIDENT,
                "Inkling 2 Mini reusable H2D staging (mount path)".into(),
                true,
                maximum,
            ));
        }

        fixed_request_tails(&mut components, maximum, &[])?;
        let mut contract = Contract {
            phase_order: PhaseOrder::TARGET_THEN_DRAFT,
            spark_reply_formats: None,
            resident_weight_limit: request.resident_weight_limit,
            page_rows: PAGE_ROWS,
            pool_page_multiple: 1,
            maximum_pages: maximum,
            gpus: 1,
            variables: contract_variables,
            requires,
            components,
        };
        contract.add_driver_page_tables();
        Ok(contract)
    }
}

pub fn planned_budget(physical: u64, logical: u64, context: u64) -> Budget {
    Budget {
        id: "rtx0".into(),
        locations: vec![Location::coordinator(0)],
        physical_bytes: physical,
        logical_bytes: logical,
        headroom: crate::placement::reserve::gpu("inkling2_mini"),
        baseline: Baseline::Forecast {
            occupied: context,
            source: "CUDA context and modules on the coordinator GPU".into(),
            assumed: false,
        },
    }
}

pub fn measured_budget(physical: u64, logical: u64, free: u64) -> Budget {
    Budget {
        id: "rtx0".into(),
        locations: vec![Location::coordinator(0)],
        physical_bytes: physical,
        logical_bytes: logical,
        headroom: crate::placement::reserve::gpu("inkling2_mini"),
        baseline: Baseline::Sampled {
            free,
            phase: Phase::Load,
            source: "Inkling 2 Mini pre-weight CUDA sample".into(),
        },
    }
}
