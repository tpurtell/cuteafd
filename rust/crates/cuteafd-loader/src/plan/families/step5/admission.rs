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
    ensure!((1..=4096).contains(&rows), "Step-5 input row capacity");
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
    let layers = layout
        .ranks
        .get(rank)
        .context("Step-5 staging rank")?
        .regions
        .len();
    let step = usize::try_from(layout.ring.step)?;
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
    pub inventory: Inventory,
    pub rows: usize,
    pub sequences: usize,
    pub lanes: usize,
    pub context: u64,
    pub requested_pool: Option<u64>,
    pub marks: MarkRule,
    pub peer_diagnostics: bool,
    pub pending_code: Vec<u64>,
    pub startup: Vec<Demand>,
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
        ensure!(
            (1..=4096).contains(&self.rows)
                && (1..=64).contains(&self.sequences)
                && self.sequences <= self.rows
                && matches!(self.lanes, 1 | 2),
            "Step-5 rows, sequence slots or lane capacity"
        );
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
            rows: self.rows as u64,
            lanes: self.lanes as u64,
            entry_gpu: 0,
            head_gpu: 0,
            terminal: TerminalHome::AttentionOwner,
        }
    }

    /// Shared alternating whole-layer ownership for CSA; SWA uses the native
    /// head split on TP2. The same modes bind weights, state and residual hops.
    pub fn ownership(&self, cfg: &Config) -> Result<ExecutionOwnership> {
        let ranks = self.inventory.gpus.len();
        let owners = attention::alternating_owners(
            cfg.full[..cfg.layers].iter().map(|&full| (full, None)),
            0,
        )?;
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
            .map(|rank| super::native::rank_admission(checkpoint, ownership, rank, inputs.rows))
            .collect()
    })
}

pub fn admit_with(
    cfg: &Config,
    inputs: &RuntimeInputs,
    native: impl FnOnce(&ExecutionOwnership) -> Result<Vec<RankAdmission>>,
) -> Result<Admission> {
    inputs.validate(cfg)?;
    let ownership = inputs.ownership(cfg)?;
    let ranks = ownership.ranks();
    let native = native(&ownership)?;
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
                Step5CacheLayout::for_execution(cfg, &ownership, context, 0, inputs.rows as u64)?;
            let mark_bytes = usize::try_from(checked_sum(cache.ranks.iter().map(|r| r.bytes))?)?;
            let marks = inputs.marks.slots(mark_bytes);
            let mut fixed = Vec::new();
            let mut reservations = Vec::new();
            let wave_ranks = ownership.routed_wave_ranks()?;
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
                    "Step-5 private native lane workspace",
                    scaled(native[rank].workspace, inputs.lanes)?,
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
                    scaled(
                        staging_bytes(&cache, rank, inputs.sequences)? as u64,
                        inputs.lanes,
                    )?,
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
                        "Step-5 private input lanes",
                        scaled(input_bytes(inputs.rows)?, inputs.lanes)?,
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
                if wave_ranks.contains(&rank) {
                    add(
                        Category::Transport,
                        "Step-5 Spark intake planes",
                        scaled(
                            scaled(
                                (inputs.rows as u64 * 4096 * 2).max(256),
                                inputs.inventory.spark_ranks,
                            )?,
                            inputs.lanes,
                        )?,
                    );
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
                        scaled(
                            checked_sum([
                                8 * (inputs.rows as u64 * 4096 * 4).max(256),
                                256,
                                256,
                                256,
                                progress,
                            ])?,
                            inputs.lanes,
                        )?,
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
                    experts: cfg.moe[layer].then_some(ExpertCost {
                        whole: Bytes2::default(),
                        half: [Bytes2::default(); 2],
                        tp2: false,
                        spark_ok: true,
                    }),
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
                expert_workspace: 0,
                tp2_workspace: [0; 2],
                onboard: Onboard::Auto,
                expert_gpus: 0,
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
            let placement = solve(&request)?;
            ensure!(
                placement.pool_tokens >= FLOOR,
                "Step-5 aggregate pool is below the 256K agentic floor"
            );
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
            lanes: 2,
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
            startup: vec![],
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
            assert_eq!(admission.ownership.execution().spec().lanes, 2);
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
                assert_eq!(admission.ownership.execution().mode(7)?, WHOLE1);
            }
        }
        Ok(())
    }

    #[test]
    fn host_retention_reserves_only_shared_staging_marks() -> Result<()> {
        for ranks in [1, 2] {
            let mut inputs = inputs(ranks);
            inputs.context = EXTENT;
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
                9 * 48 as u64 * inputs.lanes as u64
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
            Step5CacheLayout::for_execution(&config(), &admission.ownership, EXTENT, 0, 64)?;
        assert!(initial.ranks[0].bytes > admission.cache.ranks[0].bytes);
        Ok(())
    }
}
