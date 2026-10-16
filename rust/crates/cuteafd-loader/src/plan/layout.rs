//! Placement planner core: a per-device memory layout for a checkpoint on a
//! hardware inventory (1-2 coordinator GPUs, 0-8 Spark ranks).
//!
//! Weights come from the checkpoint plan (source bytes per component) mapped
//! to each family's resident representation and head-split ownership; KV from
//! the family cache geometry; workspaces, runtime overhead and Spark buffers
//! from per-family costs calibrated against the allocation ledger
//! (`scripts/bench/memory-audit.py`, PLAN.md Phase 6 audit). The KV pool takes
//! what the tightest KV-owning device has left, capped at the target.
use super::{Component, ExpertPlacement, Owner, PlanReport, Status};
use crate::serving_capacity::{CacheOptions, KvPlacement};
use cuteafd_core::memory_layout::{size_pool, Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout, Waste};

pub(crate) mod deepseek_v4;
pub(crate) mod glm5_flash;
pub(crate) mod v41;
pub(crate) mod mimo;
pub(crate) mod glm5;
pub(crate) mod qwen;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
// 2026-10-07 SM120/SM121 resident ledger at the fixed 4096-image-token capacity:
// native scratch_glm's aligned arenas, including its dedicated 4 MiB BLAS workspace.
const GLM_FLASH_ENCODER_SCRATCH_BYTES: u64 = 791_907_584;

/// The hardware and serving shape to lay out.
#[derive(Debug, Clone)]
pub struct LayoutOptions {
    /// Usable bytes of each coordinator GPU (1 or 2).
    pub rtx_bytes: Vec<u64>,
    /// Number of Spark vision tower replicas (default one).
    pub vision_replicas: usize,
    /// One mapped pinned token embedding instead of a device allocation.
    pub host_embedding: bool,
    /// Explicit device placement disables the memory-constrained auto profile.
    pub force_gpu_embedding: bool,
    /// Usable bytes of one Spark rank (unified memory).
    pub spark_bytes: u64,
    /// Runtime allocation ceiling, distinct from unified physical memory.
    pub spark_allocation_budget_bytes: Option<u64>,
    /// Two coordinator GPUs split attention heads (generic families) rather
    /// than V4.1's layer ranges.
    pub head_split: bool,
    /// Prefill rows per step (workspace shape; GLM 5.3 Flash: per lane).
    pub prefill_rows: u64,
    /// Admit the probe-only all-row vocabulary output on the lead GPU.
    pub full_prefill_logits: bool,
    /// Prefill lanes (GLM 5.3 Flash); 0 selects the family default.
    pub prefill_lanes: u64,
    /// Rows of a GLM 5.3 Flash decode or verify step (`--decode-rows`): 64, or 128 with the wide
    /// `_m128` programs (one GPU). The decode workspace, the token selector and the speculative
    /// replay records hold this many rows.
    pub glmf_decode_rows: u64,
    /// Loaded projection/head formats; defaults match the launcher.
    pub glmf_representation: crate::families::glm5_flash::resident::GlmfRepresentation,
    /// Recurrent state precision, including active slots, prefix marks and program scratch.
    pub glmf_kda_state: crate::serving_capacity::GlmfKdaState,
    /// Expanded KDA head-split prefill; the launcher enables it.
    pub glmf_prefill_expanded: bool,
    /// GLM Flash index storage; a head split keeps the token keys, as serving does.
    pub glmf_index: crate::serving_capacity::GlmfIndexCache,
    /// Decode graph budget (GLM 5.3 Flash `--graph-budget-mib`), in place of the graph allowance
    /// where the engine admits from measured memory (one GPU, Spark experts, an automatic pool),
    /// else in its place only when larger.
    pub graph_budget_bytes: Option<u64>,
    /// GLM 5.3 Flash's prefix marks in pool units (`--prefix-marks pool`): no mark arena, and
    /// `GLMF_POOL_MARK_RESERVED_UNITS` units allocated beside the pool and never handed out.
    pub glmf_pool_marks: bool,
    /// GLM 5.3 Flash's KDA replay records in the prefill lanes' scratch (`--replay-records shared`):
    /// out of the state, and the scratch holds at least them. Supported with one GPU and Spark
    /// experts, for either an automatic or an explicit pool.
    pub glmf_shared_replay: bool,
    /// Spark wave capacity in rows (`expertd --capacity`).
    pub spark_capacity_rows: u64,
    /// Explicit pool tokens; `None` or `Some(0)` sizes the pool from what is left.
    pub pool_tokens: Option<u64>,
    /// Upper bound for an automatically sized pool.
    pub target_pool_tokens: u64,
    /// An automatic pool under a fixed onboard stops at its target instead of
    /// growing to all free memory (it still shrinks to fit, down to its floor).
    pub pool_ceiling_at_target: bool,
    /// External drafter resident bytes on the lead GPU (DFlash), if any.
    pub drafter_bytes: u64,
    /// Explicitly disable GLM Flash external speculation and its arenas.
    pub glmf_drafter_disabled: bool,
    /// GLM Flash external draft checkpoint; absent resolves the launcher default from HF_HOME.
    pub glmf_drafter_snapshot: Option<std::path::PathBuf>,
    /// Dense NVFP4 package manifest; otherwise discovered beside PROGRAMS.json.
    pub glmf_dense_manifest: Option<std::path::PathBuf>,
    /// GLM 5.3's external drafter; absent resolves its launcher default.
    pub glm5_drafter_snapshot: Option<std::path::PathBuf>,
    pub glm5_drafter_disabled: bool,
    pub glm5_draft_bf16: bool,
    /// Keep this much of every GPU free for runtime growth.
    pub headroom_bytes: u64,
    /// Concurrent sequences (state slots = concurrency + 2).
    pub concurrency: u64,
    /// Exact engine slots when they differ from concurrency + 2.
    pub state_slots: Option<u64>,
    /// The lanes GLM 5.3 Flash's prefix mark arena counts (its 2C + 2 floor), when they differ
    /// from concurrency: serve-glmf counts at most `DECODE_ROWS` (64) decoding sequences.
    pub glmf_mark_lanes: Option<u64>,
    /// Prefix mark arena slots; absent selects the family policy.
    pub prefix_slots: Option<u64>,
    /// MiMo's and GLM 5.3 Flash's retained snapshots per bank and device mark budget, matching
    /// serving.
    pub mimo_prefix_entries: u64,
    pub mimo_prefix_mark_bytes: u64,
    /// Retain warm external MiMo drafter context with prefix marks (candidate opt-in).
    pub mimo_prefix_draft: bool,
    /// MiMo target rings (serving default: 16, at least concurrency).
    pub mimo_rings: u64,
    /// MiMo Spark activation format, matching serving's --expert-input.
    pub mimo_expert_input: crate::families::mimo_v2::MimoExpertInput,
    /// MiMo drafter batch limit (serving raises it to concurrency).
    pub draft_sequences: u64,
    /// Explicit MiMo DFlash context arena; does not widen its draft batch.
    pub draft_context_slots: Option<u64>,
    /// MiMo segmented decode graphs (runtime default: off).
    pub mimo_decode_graphs: bool,
    /// Physical SM count for scratch when simulating a smaller memory budget.
    pub physical_sms: Option<u32>,
    /// MiMo bundled drafter representation; launch cards select single-copy FP8.
    pub mimo_draft_bf16: bool,
    /// Native local expert package manifest, required to qualify its scratch.
    pub mimo_expert_manifest: Option<std::path::PathBuf>,
    /// Compiled context extent (RoPE and index workspaces).
    pub context_tokens: u64,
    /// Resident native drafter stages (zero disables optional native MTP).
    pub native_mtp_layers: usize,
    /// Qwen full-attention record format; index keys remain BF16.
    pub qwen_kv: crate::families::qwen4::Qwen4KvCache,
    /// Selected Qwen projection/head residency, matching serve-qwen4.
    pub qwen_representation: crate::families::qwen4::resident::Qwen4Representation,
    /// Qwen immutable attention scratch override.
    pub qwen_attention_plan_rows: Option<usize>,
    /// Whole routed backbone layers resident on the coordinator.
    pub local_expert_layers: Option<usize>,
    /// Families on the shared solver (V4): how many routed layers are
    /// RTX-resident (`--rtx-expert-layers auto|max|N|N%|all`); a fixed onboard
    /// makes the KV pool the output. `None` takes the family default (V4:
    /// `max`); `local_expert_layers` maps onto `Layers(n)` when unset.
    pub onboard: Option<crate::placement::Onboard>,
    /// V4: force FP32 FFN exchange payloads for prefill as well as TP2 decode.
    pub exchange_f32: bool,
    /// GLM Flash KDA partials and their peer payloads use FP32.
    pub glmf_kda_fp32_partials: bool,
    /// Matching image's PROGRAMS.json for exact V4 workspace geometry.
    pub workspace_manifest: Option<std::path::PathBuf>,
    /// Per-GPU budget for resident weights (`--coordinator-weight-budget-gib`).
    pub weight_budget_bytes: u64,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES],
            vision_replicas: 1,
            host_embedding: false,
            force_gpu_embedding: false,
            // 121.7 GiB GB10 minus the host OS and sparknestd measured idle (~13 GiB).
            spark_bytes: 108 * GIB,
            spark_allocation_budget_bytes: None,
            head_split: true,
            prefill_rows: 0,
            full_prefill_logits: false,
            prefill_lanes: 0,
            glmf_decode_rows: crate::serving_capacity::GLMF_DECODE_ROWS,
            glmf_representation: Default::default(),
            glmf_kda_state: Default::default(),
            glmf_prefill_expanded: true,
            glmf_index: crate::serving_capacity::GlmfIndexCache::Keys,
            graph_budget_bytes: None,
            glmf_pool_marks: false,
            glmf_shared_replay: false,
            spark_capacity_rows: 4096,
            pool_tokens: None,
            target_pool_tokens: cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS,
            pool_ceiling_at_target: false,
            drafter_bytes: 0,
            glmf_drafter_disabled: false,
            glmf_drafter_snapshot: None,
            glmf_dense_manifest: None,
            glm5_drafter_snapshot: None,
            glm5_drafter_disabled: false,
            glm5_draft_bf16: false,
            headroom_bytes: 2 * GIB,
            concurrency: 0,
            state_slots: None,
            glmf_mark_lanes: None,
            prefix_slots: None,
            mimo_prefix_entries: 20,
            mimo_prefix_mark_bytes: 2 * GIB,
            mimo_prefix_draft: false,
            mimo_rings: 16,
            mimo_expert_input: Default::default(),
            draft_sequences: 4,
            draft_context_slots: None,
            mimo_decode_graphs: false,
            physical_sms: None,
            mimo_draft_bf16: false,
            mimo_expert_manifest: None,
            context_tokens: 0,
            native_mtp_layers: 3,
            qwen_kv: Default::default(),
            qwen_representation: crate::families::qwen4::resident::Qwen4Representation {
                fp8_projections: false, fp8_head: true },
            qwen_attention_plan_rows: None,
            local_expert_layers: None,
            onboard: None,
            exchange_f32: false,
            glmf_kda_fp32_partials: false,
            workspace_manifest: None,
            weight_budget_bytes: u64::MAX,
        }
    }
}

/// Shared with the GLM runtime so admission and allocation use the same lanes.
pub fn glm_prefill_lanes(value: Option<&str>) -> usize {
    value.and_then(|v| v.parse().ok()).unwrap_or(3).clamp(1, 4)
}

/// Additional lead-GPU output bytes for admitted all-row fidelity probes.
/// GLM uses its configured lanes; GLM Flash's lanes (a serial prefill runs in
/// the first) share one set of temporaries and so one logits buffer.
/// V4 reuses its bounded head buffer; V4.1 already supports prefill scoring.
pub fn full_prefill_logits_bytes(family: &str, rows: u64, vocab: u64) -> u64 {
    let lanes = glm_prefill_lanes(std::env::var("CUTEAFD_GLM_PREFILL_LANES").ok().as_deref());
    full_prefill_logits_bytes_with_lanes(family, rows, vocab, lanes)
}

fn effective_glm_prefill_lanes(sparks: bool, value: Option<&str>) -> usize {
    if sparks { glm_prefill_lanes(value) } else { 1 }
}

pub fn full_prefill_logits_bytes_with_lanes(family: &str, rows: u64, vocab: u64, glm_lanes: usize) -> u64 {
    let (lanes, ordinary_rows) = match family {
        "mimo_v2" => (1, 1),
        "qwen4" => (1, 1),
        "glm5" => (glm_lanes.clamp(1, 4) as u64, rows.min(64)),
        "glm5_flash" => (1, rows.min(64)),
        _ => return 0,
    };
    let extra_rows = rows.saturating_sub(ordinary_rows);
    let logits = extra_rows.saturating_mul(lanes).saturating_mul(vocab).saturating_mul(4);
    // Qwen's output workspace also owns one argmax and selection pair per row.
    logits.saturating_add(if family == "qwen4" { extra_rows.saturating_mul(16) } else { 0 })
}

#[cfg(test)]
mod scoring_workspace_tests {
    use super::*;

    #[test]
    fn full_rows_are_opt_in_and_reserve_the_family_output_delta() {
        assert!(!LayoutOptions::default().full_prefill_logits);
        assert_eq!(full_prefill_logits_bytes("mimo_v2", 128, 1000), 127 * 1000 * 4);
        assert_eq!(full_prefill_logits_bytes("qwen4", 128, 1000), 127 * (1000 * 4 + 16));
        assert_eq!(glm_prefill_lanes(None), 3);
        assert_eq!(effective_glm_prefill_lanes(false, None), 1);
        assert_eq!(effective_glm_prefill_lanes(false, Some("4")), 1);
        assert_eq!(effective_glm_prefill_lanes(true, None), 3);
        for lanes in 1..=4 {
            assert_eq!(glm_prefill_lanes(Some(&lanes.to_string())), lanes);
            assert_eq!(full_prefill_logits_bytes_with_lanes("glm5", 128, 1000, lanes),
                lanes as u64 * 64 * 1000 * 4);
        }
        assert_eq!(glm_prefill_lanes(Some("0")), 1);
        assert_eq!(glm_prefill_lanes(Some("9")), 4);
        assert_eq!(glm_prefill_lanes(Some("invalid")), 3);
        assert_eq!(full_prefill_logits_bytes("glm5_flash", 128, 1000), 64 * 1000 * 4);
        for family in ["deepseek_v4", "deepseek_v41"] {
            assert_eq!(full_prefill_logits_bytes(family, 2048, 1000), 0);
        }
        for family in ["mimo_v2", "qwen4", "glm5", "glm5_flash"] {
            assert_eq!(full_prefill_logits_bytes(family, 1, 1000), 0);
            assert_eq!(full_prefill_logits_bytes(family, u64::MAX, u64::MAX), u64::MAX);
        }
    }
}

/// How a component's weights sit on two coordinator GPUs.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Share {
    /// Lead GPU only (embedding, head, router, speculators).
    Lead,
    /// A full copy on every GPU (norms, small operands).
    Replicated,
    /// Split by heads / intermediate rows; `replicated` of it is copied on both.
    Sharded { replicated: f64 },
}

/// Runtime allowances checked against allocation ledgers after 8K prefill,
/// C4 and C1. The layout marks only measured reference configurations calibrated.
#[derive(Debug, Clone, Copy)]
pub struct FamilyCosts {
    /// CUDA context, modules and cuBLAS at ready (untracked): one GPU, lead and peer of a head split.
    pub runtime_bytes: [u64; 3],
    /// Allowance for decode/verify graph executables captured as traffic arrives
    /// (keyed by exact row counts and table widths; grows after ready).
    pub graph_bytes: [u64; 3],
    /// Step workspaces incl. sampler and Spark intake at 4096 prefill rows: one GPU, lead, peer.
    pub workspace_bytes: [u64; 3],
    /// Head-split peer exchange slots on each GPU.
    pub exchange_bytes: u64,
    /// The family's default drafter (weights and its buffers) on the lead GPU.
    pub drafter_bytes: u64,
    /// Prefix-cache mark slots resident on the device (per GPU).
    pub mark_slots: u64,
    /// Native MTP layers stay resident (false: the default drafter replaces them).
    pub mtp_resident: bool,
    /// Fraction of attention weights replicated on both GPUs under a head split.
    pub attention_replicated: f64,
    /// Resident bytes per source byte of coordinator weights (load-time conversion).
    pub resident_factor: f64,
    /// Spark worker scratch, workspace and host exchange at 4096 rows.
    pub spark_workspace_bytes: u64,
    /// RDMA rings per Spark rank (every coordinator endpoint).
    pub spark_ring_bytes: u64,
    /// Non-reclaimable host memory charged beside CUDA allocations on GB10.
    pub spark_host_bytes: u64,
}


/// Compatibility query for the dedicated V4.1 worker. Generic planning uses
/// the selected working set's explicit SparkIoBytes instead of allowances.
pub fn v41_spark_ring_allowance(family: &str, capacity: u64, calibrated: u64, bf16: bool) -> u64 {
    if family != "deepseek_v41" { return calibrated; }
    let exact = u32::try_from(capacity).ok().and_then(|capacity|
        cuteafd_core::ExpertGeometry::DEEPSEEK_V41.compact_ring_bytes(
            capacity, 64 << 20, 8, 8 << 20, 4096, 2, bf16));
    exact.map_or(u64::MAX, |bytes| bytes as u64)
}


/// Ring endpoints a family's coordinator opens to each selected Spark rank,
/// given its prefill lanes (V4/V4.1: two fixed; GLM 5.3: one decode endpoint
/// plus one per lane when lanes run in parallel; GLM Flash/MiMo: one per lane;
/// Qwen: one).
pub fn spark_endpoints(family: &str, lanes: u64) -> usize {
    match family {
        "deepseek_v41" | "deepseek_v4" => 2,
        "glm5" => 1 + if lanes > 1 { lanes as usize } else { 0 },
        "glm5_flash" | "mimo_v2" => lanes as usize,
        "qwen4" => 1,
        _ => 0,
    }
}

/// The selected ring geometry for `endpoints` coordinator endpoints per rank.
pub fn spark_ring_geometry(endpoints: usize) -> crate::placement::SparkRingGeometry {
    crate::placement::SparkRingGeometry { max_frame_bytes: 64 << 20, depth: 8,
        slot_bytes: 8 << 20, alignment: 4096, endpoints }
}

/// Split the old combined wave allowance into its non-I/O residual and the
/// selected rank's explicit allocations. BF16 replaces input bytes; it is not
/// a second wire buffer or an RTX/world-size side subtraction.
pub(crate) fn charge_spark_io(device: &mut DeviceLayout, working: &crate::placement::SparkWorkingSet,
    geometry: cuteafd_core::ExpertGeometry, capacity: u32, bf16: bool,
    rings: crate::placement::SparkRingGeometry, wave_allowance: u64)
    -> Result<(), crate::placement::PlacementError> {
    let rank = device.index as usize;
    let io = working.rank_io(rank, geometry, capacity, bf16, rings)?;
    let legacy = working.rank_io(rank, geometry, capacity, false, rings)?;
    let legacy_io = legacy.intake.checked_add(legacy.host_exchange)
        .and_then(|bytes| bytes.checked_add(legacy.row_indices))
        .ok_or(crate::placement::PlacementError::Overflow("Spark legacy I/O"))?;
    device.items.retain(|item| !matches!(item.group.as_str(), "expert waves" | "expert intake"
        | "host expert exchange" | "expert row indices" | "rdma rings"));
    if working.selected_ranks().contains(&rank) {
        device.items.push(Item::new(Category::Workspace, "expert waves", "non-I/O residual",
            wave_allowance.saturating_sub(legacy_io), Basis::Calibrated));
        device.items.push(Item::new(Category::Workspace, "expert intake", if bf16 { "BF16" } else { "FP8 K32" },
            io.intake, Basis::Formula));
        device.items.push(Item::new(Category::Workspace, "host expert exchange", "", io.host_exchange, Basis::Formula));
        device.items.push(Item::new(Category::Workspace, "expert row indices", "", io.row_indices, Basis::Formula));
        device.items.push(Item::new(Category::Transport, "rdma rings", "", io.mapped_rings, Basis::Formula));
    }
    Ok(())
}

#[cfg(test)]
mod selected_spark_io_tests {
    use super::*;
    use cuteafd_core::ExpertGeometry;
    use crate::placement::SparkWorkingSet;

    #[test]
    fn selected_rank_rings_fix_legacy_undercharges_for_every_wire_format() {
        for (family, geometry, lanes, expected) in [
            ("deepseek_v4", ExpertGeometry::DEEPSEEK_V4_FLASH, 2, 821428224),
            ("deepseek_v4", ExpertGeometry::DEEPSEEK_V4_PRO, 2, 1431699456),
            ("glm5", ExpertGeometry::GLM5, 3, 2459697152),
            ("mimo_v2", ExpertGeometry::MIMO_V26_PRO, 3, 1844772864),
        ] {
            let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
                prefill_lanes: lanes, ..Default::default() };
            let rings = spark_ring_geometry(spark_endpoints(family, lanes));
            let io = working.rank_io(0, geometry, 4096, false, rings).unwrap();
            assert_eq!(io.mapped_rings, expected);
            assert!(io.mapped_rings > family_costs(family).spark_ring_bytes);
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0,
                capacity_bytes: 121 * GIB, items: vec![], kv_tokens: 0 };
            charge_spark_io(&mut device, &working, geometry, 4096, false, rings,
                family_costs(family).spark_workspace_bytes).unwrap();
            assert_eq!(device.items.iter().find(|item| item.group == "rdma rings").unwrap().bytes, expected);
        }
    }

    #[test]
    fn bf16_replaces_selected_rank_intake_and_empty_selection_has_no_io() {
        for (family, geometry, lanes) in [("deepseek_v41", ExpertGeometry::DEEPSEEK_V41, 1),
            ("glm5", ExpertGeometry::GLM5, 3), ("glm5_flash", ExpertGeometry::GLM5_FLASH, 2),
            ("qwen4", ExpertGeometry::QWEN4, 1)] {
            let mut working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 6,
                prefill_lanes: lanes, ..Default::default() };
            let rings = spark_ring_geometry(spark_endpoints(family, lanes));
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 5,
                capacity_bytes: 121 * GIB, items: vec![], kv_tokens: 0 };
            let wave = family_costs(family).spark_workspace_bytes;
            charge_spark_io(&mut device, &working, geometry, 4096, false, rings, wave).unwrap();
            let wire = device.items.iter().find(|item| item.group == "expert intake").unwrap().bytes;
            charge_spark_io(&mut device, &working, geometry, 4096, true, rings, wave).unwrap();
            let bf16 = device.items.iter().find(|item| item.group == "expert intake").unwrap().bytes;
            assert_eq!(bf16 - wire, 4096 * (geometry.hidden as u64 - geometry.hidden as u64 / 32));
            assert_eq!(device.items.iter().filter(|item| item.group == "rdma rings").count(), 1);
            working.spark_layers.clear();
            charge_spark_io(&mut device, &working, geometry, 4096, true, rings, wave).unwrap();
            assert!(device.items.is_empty());
        }
        let glm = crate::placement::families::glm5::prefill_lanes(4, glm_prefill_lanes(None) as u64);
        assert_eq!(spark_endpoints("glm5", glm), 4);
        assert_eq!(spark_endpoints("glm5", crate::placement::families::glm5::prefill_lanes(0, 3)), 1);
    }
}

const fn gib(hundredths: u64) -> u64 {
    hundredths * GIB / 100
}

pub fn family_costs(family: &str) -> FamilyCosts {
    let generic = FamilyCosts {
        runtime_bytes: [gib(90), gib(90), gib(85)],
        graph_bytes: [gib(150), gib(150), gib(150)],
        workspace_bytes: [gib(500), gib(450), gib(300)],
        exchange_bytes: gib(50),
        drafter_bytes: 0,
        mark_slots: 0,
        mtp_resident: true,
        attention_replicated: 0.0,
        resident_factor: 1.0,
        spark_workspace_bytes: gib(100),
        spark_ring_bytes: gib(150),
        spark_host_bytes: 13 * GIB,
    };
    match family {
        "deepseek_v4" => FamilyCosts {
            runtime_bytes: [gib(65), gib(60), gib(60)],
            graph_bytes: [gib(35), gib(40), gib(25)],
            workspace_bytes: [gib(480), gib(414), gib(355)],
            exchange_bytes: gib(31),
            mark_slots: 18,
            spark_workspace_bytes: gib(35),
            spark_ring_bytes: gib(52),
            ..generic
        },
        "deepseek_v41" => FamilyCosts {
            runtime_bytes: [gib(133), gib(100), gib(100)],
            graph_bytes: [gib(150), gib(100), gib(90)],
            workspace_bytes: [gib(1101), gib(948), gib(570)],
            mark_slots: 42,
            spark_workspace_bytes: gib(21),
            spark_ring_bytes: gib(48),
            spark_host_bytes: 9 * GIB,
            ..generic
        },
        "qwen4" => FamilyCosts {
            runtime_bytes: [0; 3],
            graph_bytes: [0; 3],
            workspace_bytes: [0; 3],
            mark_slots: crate::placement::families::qwen4::DEFAULT_MARK_SLOTS,
            spark_workspace_bytes: crate::placement::families::qwen4::SPARK_WORKSPACE_BYTES,
            spark_ring_bytes: crate::placement::families::qwen4::SPARK_RING_BYTES,
            ..generic
        },
        _ => generic,
    }
}

fn share_of(family: &str, component: Component) -> Share {
    match component {
        Component::Norm | Component::HyperConnection => Share::Replicated,
        Component::Attention | Component::DenseFfn | Component::SharedExpert | Component::Indexer
        | Component::Compressor => Share::Sharded { replicated: 0.0 },
        _ => Share::Lead,
    }
}

fn mimo_draft_prefix_bytes(config: &serde_json::Value, marks: u64, rings: u64) -> Result<(u64, u64), String> {
    let field = |name| config[name].as_u64().filter(|&n| n > 0)
        .ok_or_else(|| format!("missing or zero {name}"));
    let width = field("num_key_value_heads")?.checked_mul(field("head_dim")?)
        .ok_or("DFlash KV width overflow")?;
    let mark = crate::families::mimo_v2::draft_representation::mimo_draft_mark_bytes(field("num_hidden_layers")?, width)
        .map_err(|e| e.to_string())?;
    Ok((mark.checked_mul(marks).ok_or("DFlash marks overflow")?,
        rings.checked_mul(8).ok_or("DFlash floors overflow")?))
}

fn resolved_context(policy: ContextPolicy, requested: u64, checkpoint: u64, manifest: Option<&serde_json::Value>) -> u64 {
    if requested > 0 { return requested; }
    if policy == ContextPolicy::Checkpoint { return checkpoint; }
    manifest.and_then(|m| m["capacities"]["max_context"].as_u64()).filter(|&n| n > 0)
        .map_or(checkpoint, |limit| if checkpoint > 0 { limit.min(checkpoint) } else { limit })
}

#[cfg(test)]
mod draft_prefix_tests {
    use super::*;

    #[test]
    fn full_context_defaults_preserve_dynamic_families_and_bound_indexed_families() {
        let manifest = serde_json::json!({"capacities": {"max_context": 131072}});
        for policy in [ContextPolicy::Checkpoint, ContextPolicy::ManifestBounded] {
            let expected = if policy == ContextPolicy::Checkpoint { 1048576 } else { 131072 };
            assert_eq!(resolved_context(policy, 0, 1048576, Some(&manifest)), expected);
            assert_eq!(resolved_context(policy, 65536, 1048576, Some(&manifest)), 65536);
            assert_eq!(resolved_context(policy, 0, 1048576, None), 1048576);
        }
    }

    #[test]
    fn draft_prefix_reserves_marks_and_all_ring_floors() {
        let config = serde_json::json!({"num_hidden_layers": 5, "num_key_value_heads": 8, "head_dim": 128});
        assert_eq!(mimo_draft_prefix_bytes(&config, 42, 18).unwrap(), ((20 * MIB + 8) * 42, 144));
        assert!(mimo_draft_prefix_bytes(&config, u64::MAX, 18).is_err());
        assert!(mimo_draft_prefix_bytes(&config, 42, u64::MAX).is_err());
        assert!(mimo_draft_prefix_bytes(&serde_json::json!({}), 42, 18).is_err());
    }
}

/// Where a family's default serving context comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextPolicy {
    /// The checkpoint's own maximum (MiMo, V4.1: dynamic RoPE/cache extents).
    Checkpoint,
    /// The checkpoint's maximum, bounded by the program manifest's compiled extent.
    ManifestBounded,
}

/// How a family resolves the step shape from the options and its manifest.
#[derive(Debug, Clone, Copy)]
pub struct ShapePolicy {
    pub context: ContextPolicy,
    /// Prefill rows when the options leave them at 0: `None` takes the
    /// manifest's compiled capacity (else 4096).
    pub prefill_rows: Option<u64>,
    /// A drafter request keeps every discovered native stage (V4's dSpark)
    /// instead of `min(requested, discovered)`.
    pub all_native_stages: bool,
}

impl ShapePolicy {
    pub const fn manifest_bounded() -> Self {
        Self { context: ContextPolicy::ManifestBounded, prefill_rows: None, all_native_stages: false }
    }
}

/// The serving shape every family adapter starts from: the image's program
/// manifest, step rows, sequences and context, resolved once.
#[derive(Debug, Clone)]
pub struct Shape {
    /// The program manifest (`--workspace-manifest`, else the image's).
    pub manifest: Option<serde_json::Value>,
    pub prefill_rows: u64,
    pub decode_rows: u64,
    pub concurrency: u64,
    /// The checkpoint's own maximum context (0: undeclared).
    pub checkpoint_context: u64,
    pub context_tokens: u64,
    /// Native drafter stages kept resident.
    pub native_layers: usize,
}

impl Shape {
    pub fn resolve(report: &mut PlanReport, model: &dyn super::FamilyModel, options: &LayoutOptions,
        policy: ShapePolicy) -> Self {
        let discovered = match model.spec().speculator.as_ref() {
            Some(super::spec::SpeculatorSpec::Dspark { stages, .. }) => *stages,
            Some(super::spec::SpeculatorSpec::NativeMtp { layers }) => *layers,
            None => 0,
        };
        let native_layers = if policy.all_native_stages && options.native_mtp_layers > 0 { discovered }
            else { options.native_mtp_layers.min(discovered) };
        let manifest = options.workspace_manifest.as_deref().or_else(|| {
            let path = std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json");
            path.is_file().then_some(path)
        }).and_then(|path| std::fs::read(path).ok())
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
        let prefill_rows = if options.prefill_rows > 0 { options.prefill_rows }
            else if let Some(rows) = policy.prefill_rows { rows }
            else { manifest.as_ref().and_then(|m| m["capacities"]["prefill_rows"].as_u64()).unwrap_or(4096) };
        report.prefill_rows = Some(prefill_rows);
        let decode_rows = manifest.as_ref().and_then(|m| m["capacities"]["decode_rows"].as_u64()).unwrap_or(64);
        let concurrency = if options.concurrency > 0 { options.concurrency }
            else { cuteafd_core::serving_capacity::default_concurrency(options.rtx_bytes.first().copied().unwrap_or(0)) };
        let checkpoint_context = report.cache_requirements.as_ref()
            .and_then(|r| r.checkpoint_max_context_tokens).unwrap_or(0);
        let context_tokens = resolved_context(policy.context, options.context_tokens, checkpoint_context,
            manifest.as_ref());
        Self { manifest, prefill_rows, decode_rows, concurrency, checkpoint_context, context_tokens, native_layers }
    }

    /// Selects the manifest's card plans for the lead GPU's SM count; a
    /// manifest without one for this card leaves the placement unsupported.
    pub fn select_card_plans(&mut self, report: &mut PlanReport, options: &LayoutOptions, notes: &mut Vec<String>) {
        if let Some(manifest) = self.manifest.as_mut() {
            let sms = options.physical_sms.unwrap_or_else(|| crate::placement::ArchContext::for_device(
                "sm_120", options.rtx_bytes[0]).sms);
            if let Err(error) = cuteafd_core::coordinator_programs::select_card_plans(manifest, sms) {
                report.placement_supported = false;
                notes.push(error.to_string());
            }
        }
    }
}

/// A coordinator GPU's runtime before its first weight: the arch context plus
/// the code the family holds at ready where that ledger is measured
/// (`placement::inventory::LOADED_CODE` for `package`), else the context,
/// cuBLAS and the manifest's modules of `programs` (`ProgramSet`), plus the
/// peer context under a split.
pub fn runtime_item(manifest: Option<&serde_json::Value>, programs: &[String], package: &str, split: bool,
    rank: usize, total_bytes: u64) -> Item {
    match crate::placement::loaded_code(&programs[0], package, split, rank as u8) {
        Some(code) => Item::new(Category::Runtime, "context+modules", "",
            crate::placement::ArchContext::coordinator(total_bytes, None).context_bytes + code.bytes, Basis::Calibrated),
        None => Item::new(Category::Runtime, "context+modules", "", planned_context(manifest, programs, total_bytes)
            + if split { crate::placement::ArchContext::for_device("sm_120", total_bytes).peer_context_bytes } else { 0 },
            Basis::Formula),
    }
}

/// Program families (manifest name prefixes) a coordinator GPU loads: `base`
/// and, under a split, its `base2` peer programs.
pub fn program_families(base: &str, split: bool) -> Vec<String> {
    std::iter::once(base.to_string()).chain(split.then(|| format!("{base}2"))).collect()
}

/// Lays out `report` (a `plan` of the checkpoint) on the inventory, for
/// families without their own `FamilyModel::placement` adapter.
pub fn layout(report: &mut PlanReport, model: &dyn super::FamilyModel, input: &super::PlanInput,
    options: &LayoutOptions) -> MemoryLayout {
    let checkpoint = input.checkpoint();
    let family_name = report.family.clone().unwrap_or_else(|| "unknown".into());
    let family = family_name.as_str();
    let policy = ShapePolicy {
        context: if matches!(family, "mimo_v2" | "deepseek_v41") { ContextPolicy::Checkpoint }
            else { ContextPolicy::ManifestBounded },
        prefill_rows: (family == "deepseek_v41").then_some(2048),
        all_native_stages: false,
    };
    let mut shape = Shape::resolve(report, model, options, policy);
    let native_layers = shape.native_layers;
    // Cache and graph ownership follows the selected checkpoint drafter.
    let cache_native_layers = native_layers;
    let small_card = matches!(family, "deepseek_v41" | "mimo_v2")
        && options.rtx_bytes.first().is_some_and(|&bytes| bytes <= 32 * GIB);
    let (prefill_rows, decode_rows, concurrency, checkpoint_context, context_tokens) = (shape.prefill_rows,
        shape.decode_rows, shape.concurrency, shape.checkpoint_context, shape.context_tokens);
    let target_pool_tokens = options.target_pool_tokens.max(context_tokens);
    let towers = match family {
        "deepseek_v41" => v41_towers(report),
        _ => Towers::checkpoint(report),
    };
    let mut costs = family_costs(family);
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    let split = gpus == 2 && options.head_split;
    let active_gpus = if split { 2 } else { 1 };
    let automatic = options.pool_tokens == Some(0);
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: if matches!(family, "deepseek_v41" | "mimo_v2") && bytes <= 32 * GIB {
                ((bytes as u128 * 97 / 100) as u64).min(bytes.saturating_sub(
                    options.headroom_bytes.max(cuteafd_core::serving_capacity::SMALL_CARD_HEADROOM_BYTES)))
            } else { (if family == "deepseek_v41" && automatic { (bytes as u128 * 97 / 100) as u64 } else { bytes })
                .saturating_sub(if family == "deepseek_v41" {
                    crate::placement::families::deepseek_v41::runtime_reserve(bytes, active_gpus, automatic)
                } else { options.headroom_bytes }) },
            items: Vec::new(), kv_tokens: 0 })
        .collect();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {context_tokens} tokens (checkpoint {checkpoint_context})"),
        format!("attention placement {}", report.attention_placement)];
    shape.select_card_plans(report, options, &mut notes);
    let workspace_manifest = shape.manifest;
    let reference_gpu = gpus == 1 && (94 * GIB..=96 * GIB).contains(&options.rtx_bytes[0])
        && native_layers > 0 && options.local_expert_layers.is_none();
    let package = report.experts.as_ref().map(|e| e.package.as_str());
    let reference = reference_gpu && match family {
        "deepseek_v41" => matches!(report.placement, ExpertPlacement::Sparks { ranks: 4 })
            && model.spec().hidden == 5120 && model.spec().layers.len() == 40
            && package == Some("v41:mxfp4 (expertd-native)") && prefill_rows == 2048
            && concurrency == 16 && matches!(options.prefix_slots, None | Some(42))
            && matches!(context_tokens, 0 | 1_048_576),
        _ => false,
    };
    let qualified = reference;
    let allowance_basis = if qualified { Basis::Calibrated } else { Basis::Estimated };
    if !qualified {
        notes.push(format!("{family}: workspace/runtime/Spark allowances are unqualified; validate against the allocation ledger before using this layout for admission"));
    }
    if reference {
        notes.push("Runtime allowances calibrated at the natural-minimum reference layout after an 8K prefill and C4 decode; other layouts remain estimates".into());
    }

    let mapped: u64 = report.components.iter().filter(|c| c.component == Component::MappedTable).map(|c| c.bytes).sum();
    if mapped > 0 { notes.push(format!("host-mapped table backing: {:.2} GiB (device budget includes staging only)", mapped as f64 / GIB as f64)); }

    // Coordinator weights: a family's exact resident layout where it has one,
    // else checkpoint bytes per component under the family's conversions.
    let v41_weights = if family == "deepseek_v41" { v41::resident_weights(checkpoint, active_gpus, native_layers > 0) } else { None };
    if let Some(ranks) = &v41_weights {
        for (device, items) in devices.iter_mut().zip(ranks) {
            device.items.extend(items.iter().filter(|item|
                report.vision != super::MediaMode::Off || item.group != "vision").cloned());
        }
    }
    let covered = |c: Component| (v41_weights.is_some() && !matches!(c, Component::RoutedExpert | Component::MappedTable))
        || (!costs.mtp_resident && matches!(c, Component::Speculator | Component::SpeculatorExpert));
    for component in report.components.iter()
        .filter(|c| (c.owner == Owner::Rtx
            || (c.owner == Owner::SparkSliced && report.placement == ExpertPlacement::Local))
            && c.status != Status::Unused && c.status != Status::Disabled
            && !matches!(c.component, Component::Vision | Component::Audio) && !covered(c.component)) {
        let (resident, format) = ((component.bytes as f64 * costs.resident_factor) as u64,
            component.formats.keys().cloned().collect::<Vec<_>>().join("+"));
        let category = match component.component {
            Component::Embedding => Category::Embedding,
            Component::Speculator | Component::SpeculatorExpert => Category::Drafter,
            Component::TableProjection => Category::Tables,
            Component::RoutedExpert => Category::Experts,
            _ => Category::Weights,
        };
        let group = component.component.label();
        let share = if split { share_of(family, component.component) } else { Share::Lead };
        let attention_share = if component.component == Component::Attention { costs.attention_replicated } else { 0.0 };
        match share {
            Share::Lead => devices[0].items.push(Item::new(category, group, &format, resident, Basis::Exact)),
            Share::Replicated => {
                for device in devices.iter_mut() {
                    device.items.push(Item::new(category, group, &format, resident, Basis::Exact));
                }
                if gpus == 2 && resident >= 16 * MIB {
                    waste.push(Waste { device: "rtx1".into(), what: format!("{group} replicated on both GPUs"),
                        bytes: resident });
                }
            }
            Share::Sharded { replicated } => {
                let replicated = (resident as f64 * (replicated + attention_share)) as u64;
                let sharded = resident - replicated;
                for device in devices.iter_mut() {
                    device.items.push(Item::new(category, group, &format, sharded / gpus as u64 + replicated,
                        Basis::Exact));
                }
                if replicated >= 16 * MIB {
                    waste.push(Waste { device: "rtx1".into(), what: format!("{group}: operands replicated under \
                        the head split"), bytes: replicated });
                }
            }
        }
    }
    if options.host_embedding || (small_card && !options.force_gpu_embedding) {
        place_host_embedding(report, checkpoint, &mut devices, &mut waste, &mut notes);
    }
    let drafter = if options.drafter_bytes > 0 { options.drafter_bytes } else { costs.drafter_bytes };
    if drafter > 0 {
        devices[0].items.push(Item::new(Category::Drafter, "drafter", "", drafter, allowance_basis));
    }

    // Fixed runtime costs.
    let gpus_now = active_gpus;
    for (index, device) in devices.iter_mut().take(active_gpus).enumerate() {
        let role = if gpus_now == 1 { 0 } else if index == 0 { 1 } else { 2 };
        device.items.push(match program_base(family, model, checkpoint) {
            // Measured inventory: the arch's CUDA context and cuBLAS (`placement::ArchContext`), plus the
            // modules of the programs this family launches (`ProgramSet`), what the runtime's sample sees
            // before its first weight.
            // Where the family's ready ledger is measured (`placement::inventory::LOADED_CODE`), the
            // context plus every byte of code it holds at ready (lazily loaded functions, packages,
            // cuBLAS) instead: serve reserves what has not arrived by its admission sample.
            Some(base) => runtime_item(workspace_manifest.as_ref(),
                &program_families(base, split), local_package(report), split, index,
                options.rtx_bytes[index]),
            None => Item::new(Category::Runtime, "context+modules", "", costs.runtime_bytes[role], allowance_basis),
        });
        let graph_allowance = if family == "deepseek_v41" && options.rtx_bytes[index] <= 32 * GIB {
            // Match the qualified fixed-bank envelope reserved by measured_pool_memory.
            2 * GIB
        } else if family == "deepseek_v41" {
            crate::placement::families::deepseek_v41::graph_reserve(options.rtx_bytes[index], active_gpus)
        } else { costs.graph_bytes[role] };
        device.items.push(Item::new(Category::Runtime, "graph allowance", "", graph_allowance, allowance_basis));
        let workspace = costs.workspace_bytes[role] * prefill_rows.max(1) / if family == "deepseek_v41" { 2048 } else { 4096 };
        device.items.push(Item::new(Category::Workspace, "steps", "", workspace, allowance_basis));
        if family == "deepseek_v41" {
            device.items.push(Item::new(Category::Workspace, "verifier lane growth", "48 to 64 rows",
                crate::placement::families::deepseek_v41::verify_workspace_growth(gpus_now, index), Basis::Formula));
        }
        if options.full_prefill_logits && index == 0 {
            device.items.push(Item::new(Category::Workspace, "probe prefill logits", "",
                full_prefill_logits_bytes_with_lanes(family, prefill_rows, model.spec().vocab as u64,
                    effective_glm_prefill_lanes(matches!(report.placement, ExpertPlacement::Sparks { .. }),
                        std::env::var("CUTEAFD_GLM_PREFILL_LANES").ok().as_deref())), Basis::Formula));
        }
        if split {
            device.items.push(Item::new(Category::Transport, "peer exchange", "", costs.exchange_bytes, allowance_basis));
        }
    }

    // Resident V4.1 layers are taken out of Spark slices. Experts are charged
    // before the KV pool. Explicit placement is reproducible on any inventory.
    let routed_bytes: u64 = report.components.iter().filter(|c| c.component == Component::RoutedExpert)
        .map(|c| c.bytes).sum();
    let package = report.experts.as_ref().map_or("", |e| e.package.as_str());
    let source_layer_bytes = routed_bytes / model.spec().layers.len().max(1) as u64;
    let packed_layer = |ranks: usize| model.spec().moe.as_ref().and_then(|moe|
        v41::packed_expert_bytes(package, moe.experts as u64, model.spec().hidden as u64,
            (moe.intermediate / ranks) as u64));
    let layer_bytes = packed_layer(1).unwrap_or(source_layer_bytes);
    let mut local_layers = if family == "deepseek_v41" {
        options.local_expert_layers.unwrap_or(0).min(model.spec().layers.len())
    } else { 0 };
    let mut local_bytes = layer_bytes * local_layers as u64;

    let spark_geometry = model.spec().moe.as_ref().map(|moe| cuteafd_core::ExpertGeometry {
        hidden: model.spec().hidden as u32, experts: moe.experts as u32,
        topk: moe.top_k as u32, intermediate: moe.intermediate as u32,
        layers: model.spec().layers.len() as u32,
    });
    let spark_format = report.components.iter().find(|c| c.owner == Owner::SparkSliced)
        .map(|c| c.formats.keys().cloned().collect::<Vec<_>>().join("+")).unwrap_or_default();
    let spark_bf16 = checkpoint.spark_input_is_bf16();
    let glmf_lanes = if options.prefill_lanes > 0 { options.prefill_lanes }
        else { crate::serving_capacity::GLMF_DEFAULT_PREFILL_LANES };
    let spark_lanes = match family {
        "glm5_flash" => if std::env::var("CUTEAFD_GLMF_PREFILL_LANES").is_ok_and(|v| v == "1") { 1 } else { glmf_lanes },
        "mimo_v2" => crate::families::mimo_v2::MimoV2Config::from_hf(&checkpoint.config).ok()
            .and_then(|cfg| cfg.program_family().ok()).and_then(|id|
                crate::families::mimo_v2::admission::resolve_transport_lanes(true, id,
                    std::env::var("CUTEAFD_MIMO_PREFILL_LANES").ok().as_deref()).ok()).unwrap_or(0) as u64,
        _ => 1,
    };
    let mut spark_working = crate::placement::SparkWorkingSet {
        rtx_layers: (0..local_layers).collect(),
        spark_layers: if matches!(report.placement, ExpertPlacement::Sparks { .. }) {
            ((if family == "deepseek_v41" { local_layers } else { 0 })..model.spec().layers.len()).collect()
        } else { Vec::new() },
        spark_ranks: match report.placement { ExpertPlacement::Sparks { ranks } => ranks, _ => 0 },
        prefill_lanes: spark_lanes,
    };
    let mut spark_devices = Vec::new();
    // Spark ranks.
    if let ExpertPlacement::Sparks { ranks } = report.placement {
        let routed: u64 = report.components.iter().filter(|c| c.owner == Owner::SparkSliced && c.component != Component::SpeculatorExpert).map(|c| c.bytes).sum::<u64>().saturating_sub(if family == "deepseek_v41" { local_bytes } else { 0 });
        let stored = (routed as f64 * report.spark_rank_share) as u64;
        let even = routed / ranks.max(1) as u64;
        // EXL3 packages and FP8/MXFP4/NVFP4 packages with exact layouts store each
        // rank's own whole 128-row blocks; other packages (V4.1 native) pad every
        // rank to the widest slice.
        let package = report.experts.as_ref().map_or("", |e| e.package.as_str());
        let intermediate = model.spec().moe.as_ref().map_or(0, |m| m.intermediate);
        let recipe = (!package.starts_with("v41")).then(||
            super::experts::spark_recipe(package, intermediate, ranks).ok()).flatten();
        let exact = recipe.as_ref().is_some_and(|recipe|
            matches!(recipe.slicing, crate::formats::fp8_experts::Slicing::Blocks(_)));
        let rank_bytes = |rank: usize| -> u64 {
            if !exact {
                return stored;
            }
            let width = recipe.as_ref().expect("exact recipe").widths[rank];
            (routed as f64 * width as f64 / intermediate as f64) as u64
        };
        for rank in 0..ranks {
            let stored = model.spec().moe.as_ref().and_then(|moe|
                v41::packed_expert_bytes(package, moe.experts as u64, model.spec().hidden as u64,
                    (moe.intermediate / ranks) as u64))
                .map_or_else(|| rank_bytes(rank), |bytes| bytes * (model.spec().layers.len() - local_layers) as u64);
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: options.spark_bytes,
                items: Vec::new(), kv_tokens: 0 };
            if family == "deepseek_v41" {
                // Recover physical memory from the legacy usable inventory
                // (13 GiB excluded), then charge the family's observed host
                // footprint separately from expert allocations.
                device.capacity_bytes += 13 * GIB;
                device.items.push(Item::new(Category::Reserved, "host OS+sparknestd", "", costs.spark_host_bytes, allowance_basis));
            }
            device.items.push(Item::new(Category::Experts, "routed_expert", spark_format.clone(), stored, Basis::Exact));
            let io = spark_geometry.ok_or(crate::placement::PlacementError::Inventory("Spark experts need routed geometry"))
                .and_then(|geometry| u32::try_from(options.spark_capacity_rows)
                    .map_err(|_| crate::placement::PlacementError::Overflow("Spark capacity"))
                    .and_then(|capacity| charge_spark_io(&mut device, &spark_working, geometry, capacity,
                        spark_bf16, spark_ring_geometry(spark_endpoints(family, spark_working.prefill_lanes)),
                        costs.spark_workspace_bytes * options.spark_capacity_rows / 4096)));
            if let Err(error) = io {
                report.placement_supported = false;
                notes.push(format!("Spark rank {rank} I/O admission: {error}"));
            }
            device.items.push(Item::new(Category::Runtime, "context+modules", "", 512 * MIB, allowance_basis));
            spark_devices.push(device);
        }
        if !exact && report.spark_rank_share * ranks as f64 > 1.001 {
            waste.push(Waste { device: format!("spark x{ranks}"), what: format!("routed slices padded to the widest \
                128-row slice ({:.1}% of the even share) on every rank", 100.0 * (stored - even) as f64 / even as f64),
                bytes: (stored - even) * ranks as u64 });
        }
    }

    let mut pool_tokens = 0;
    // KV pool: per-device bytes per logical token from the family geometry.
    let geometry = { model.cache_geometry(CacheOptions { coordinator_ranks: active_gpus,
        native_mtp_layers: 0,
        prefill_rows,
        ..Default::default() }) };
    match geometry {
        Ok(Some(mut geometry)) => {
            for rank in &mut geometry.ranks {
                rank.pool_metadata_unit_bytes += 0;
            }
            // GLM 5.3 Flash takes no mark without entries, so pool marks then keep no unit back.
            let pool_marks = false;
            let marks = if pool_marks { 0 } else { options.prefix_slots.unwrap_or_else(|| {
                let bytes: u64 = geometry.ranks.iter().map(|r| r.retained_mark_bytes).sum();
                let _ = bytes;
                costs.mark_slots
            }) };
            let unit = geometry.logical_unit_rows.max(1);
            let per_token: Vec<u64> = (0..devices.len()).map(|d| geometry.ranks.get(d)
                .map_or(0, |r| (r.persistent_unit_bytes + r.pool_metadata_unit_bytes).div_ceil(unit))).collect();
            // Reserve fixed state and marks before sizing records. Otherwise
            // an automatic pool consumes the bytes those allocations need.
            for (device, rank) in devices.iter_mut().zip(&geometry.ranks) {
                let state_slots = options.state_slots.unwrap_or(if family == "deepseek_v41" { concurrency }
                    else { concurrency + 2 });
                if family != "deepseek_v41" { device.items.push(Item::new(Category::Kv, "state", "", rank.fixed_state_bytes
                    + rank.active_state_per_sequence_bytes * state_slots
                    + rank.speculative_replay_bytes
                    + rank.context_table_bytes_per_token * context_tokens, Basis::Formula)); }
                if family != "deepseek_v41" && marks > 0 && rank.retained_mark_bytes > 0 {
                    device.items.push(Item::new(Category::Prefix, "marks", "", rank.retained_mark_bytes * marks,
                        Basis::Formula));
                }
                // Pool marks: the reserved units beside the pool (never handed out, so outside its tokens).
                if pool_marks {
                    device.items.push(Item::new(Category::Prefix, "reserved units", "",
                        (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes)
                            * crate::serving_capacity::GLMF_POOL_MARK_RESERVED_UNITS, Basis::Formula));
                }
            }
            if family == "deepseek_v41" {
                let prefixes = v41::prefix_arena_bytes(marks, native_layers > 0, active_gpus);
                for (device, &bytes) in devices.iter_mut().zip(&prefixes) {
                    device.items.push(Item::new(Category::Prefix, "snapshot arenas", "", bytes, Basis::Formula));
                }
                if native_layers > 0 {
                    devices[active_gpus - 1].items.push(Item::new(Category::Drafter, "dSpark window state", "", v41::dspark_cache_bytes(concurrency, prefill_rows), Basis::Formula));
                }
            }
            {
                let target = options.pool_tokens.filter(|&tokens| tokens != 0)
                    .unwrap_or_else(|| if small_card { (1 << 20).max(context_tokens) } else { target_pool_tokens });
                let kv: Vec<u64> = per_token.iter().map(|&cost| cost.saturating_mul(target)).collect();
                resolve_encoder(report, &towers, &mut devices, &mut spark_devices, &kv, options, &mut notes);
            }
            let free: Vec<i64> = devices.iter().map(DeviceLayout::free_bytes).collect();
            if family == "deepseek_v41" {
                use crate::placement::*;
                let inventory = Inventory {
                    gpus: devices.iter().map(|d| GpuBudget {
                        capacity_bytes: d.capacity_bytes, headroom_bytes: 0,
                        baseline: Baseline::Measured { free_bytes: d.free_bytes().max(0) as u64 },
                    }).collect(), spark_ranks: report.spark_ranks, peer_access: active_gpus == 2,
                };
                let mut pool = PoolPolicy::resolve(&options.rtx_bytes, context_tokens,
                    options.pool_tokens, unit, report.spark_ranks == 0);
                if pool.requested.is_none() {
                    pool.floor = pool.floor.max(context_tokens.div_ceil(512) * 512 + concurrency * 512);
                    pool.target = pool.target.max(pool.floor);
                }
                let tails = concurrency + 2 * (marks.saturating_sub(2) / 2);
                pool.ceiling = ((131_072 - tails) * unit).min(pool.requested.unwrap_or(pool.target).div_ceil(unit) * unit);
                let exact = (|| -> anyhow::Result<_> {
                    let catalog = crate::read_expert_catalog(&checkpoint.snapshot)?;
                    let manifest = options.workspace_manifest.as_deref()
                        .unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
                    let sms = options.physical_sms.unwrap_or_else(||
                        ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms);
                    families::deepseek_v41::expert_inventory(&catalog, manifest, prefill_rows, active_gpus, sms)
                })();
                let placement = exact.and_then(|(experts, workspace)| {
                    let mut request = families::deepseek_v41::request(inventory, pool,
                        geometry.ranks.iter().map(|r| r.persistent_unit_bytes + r.pool_metadata_unit_bytes).collect(),
                        experts, workspace,
                        options.onboard.or(options.local_expert_layers.map(Onboard::Layers)).unwrap_or(Onboard::Auto));
                    request.minimum_expert_layers = usize::from(active_gpus == 2);
                    families::deepseek_v41::cache_demands(&mut request, &checkpoint.config,
                        concurrency, marks.saturating_sub(2) / 2, false)?;
                    let placement = solve(&request)?;
                    families::deepseek_v41::validate_pool_floor(&request.pool, &placement)?;
                    let local_bytes = request.layers.iter().zip(&placement.layers).map(|(demand, assigned)| {
                        demand.experts.map_or(0, |cost| match assigned.experts {
                            ExpertHome::RtxWhole { .. } => cost.whole.resident,
                            ExpertHome::RtxTp2 => cost.half.iter().map(|c| c.resident).sum(),
                            _ => 0,
                        })
                    }).sum::<u64>();
                    Ok((placement, local_bytes))
                });
                match placement {
                    Ok((placement, exact_local_bytes)) => {
                        report.solver_placement = Some(placement.clone());
                        notes.push(format!("placement: {}", placement.summary()));
                        pool_tokens = placement.pool_tokens;
                        local_layers = placement.onboard_layers;
                        local_bytes = exact_local_bytes;
                        for (device, items) in devices.iter_mut().zip(placement.items) {
                            device.items.extend(items);
                            device.kv_tokens = pool_tokens;
                        }
                    }
                    Err(error) => {
                        report.placement_supported = false;
                        report.hints.push(super::Hint {
                            what: format!("V4.1 shared admission: {error}"),
                            how: "Reduce the pool or local expert request, or increase the device budget.".into(),
                        });
                        notes.push(format!("V4.1 shared admission: {error}"));
                    }
                }
            } else {
                pool_tokens = options.pool_tokens.filter(|&tokens| tokens != 0)
                    .unwrap_or_else(|| size_pool(&free, &per_token, unit,
                        if small_card { (1 << 20).max(context_tokens) } else { target_pool_tokens }));
            }
            if pool_tokens < context_tokens {
                notes.push(format!("full-context admission shortfall: context {context_tokens} tokens, pool {pool_tokens} tokens, shortfall {} tokens", context_tokens - pool_tokens));
            }
            for (device, &cost) in devices.iter_mut().zip(&per_token) {
                if cost > 0 && family != "deepseek_v41" {
                    let units = pool_tokens.div_ceil(unit);
                    let rank = &geometry.ranks[device.index as usize];
                    device.items.push(Item::new(Category::Kv, "records", "",
                        (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes) * units, Basis::Formula));
                    device.kv_tokens = pool_tokens;
                }
                if device.free_bytes() < 0 {
                    notes.push(format!("{}: requested KV pool exceeds the device budget", device.name()));
                }
            }
            if geometry.placement == KvPlacement::Replicated && devices.len() == 2 {
                waste.push(Waste { device: "rtx1".into(), what: "KV records replicated on both GPUs (MLA latent)".into(),
                    bytes: per_token.get(1).copied().unwrap_or(0) * pool_tokens });
            }
        }
        Ok(None) => notes.push(format!("{family}: no cache geometry in the planner yet (engine sizes its own pool)")),
        Err(error) => {
            if family == "deepseek_v41" {
                report.placement_supported = false;
                report.hints.push(super::Hint {
                    what: format!("V4.1 cache geometry admission: {error}"),
                    how: "Use a supported CED source layout before allocating the pool.".into(),
                });
            }
            notes.push(format!("{family}: cache geometry: {error}"));
        },
    }

    if local_layers > 0 { notes.push(format!("{local_layers} routed backbone layers resident on RTX")); }
    // V4.1 auto-selects local experts after KV sizing. Preserve its original
    // Spark slice accounting even though generic encoder placement needs the
    // Spark inventory earlier.
    if family == "deepseek_v41" {
        let routed: u64 = report.components.iter().filter(|c| c.owner == Owner::SparkSliced && c.component != Component::SpeculatorExpert).map(|c| c.bytes).sum();
        let package = report.experts.as_ref().map_or("", |e| e.package.as_str());
        let packed_rank = model.spec().moe.as_ref().and_then(|moe|
            (!spark_devices.is_empty()).then(|| spark_devices.len()).and_then(|ranks|
                v41::packed_expert_bytes(package, moe.experts as u64, model.spec().hidden as u64,
                    (moe.intermediate / ranks) as u64)))
            .map(|bytes| bytes * (model.spec().layers.len() - local_layers) as u64);
        for spark in &mut spark_devices {
            if let Some(experts) = spark.items.iter_mut().find(|i| i.category == Category::Experts) {
                experts.bytes = packed_rank.unwrap_or_else(||
                    (routed.saturating_sub(local_bytes) as f64 * report.spark_rank_share) as u64);
            }
        }
    }

    if family == "deepseek_v41" {
        spark_working.rtx_layers = (0..local_layers).collect();
        spark_working.spark_layers = (local_layers..model.spec().layers.len()).collect();
    }
    // Reapply after family admission has selected its executable suffix and
    // lane count. Rank-local I/O never belongs to the coordinator KV budget.
    for device in &mut spark_devices {
        let io = spark_geometry.ok_or(crate::placement::PlacementError::Inventory("Spark experts need routed geometry"))
            .and_then(|geometry| u32::try_from(options.spark_capacity_rows)
                .map_err(|_| crate::placement::PlacementError::Overflow("Spark capacity"))
                .and_then(|capacity| charge_spark_io(device, &spark_working, geometry, capacity,
                    spark_bf16, spark_ring_geometry(spark_endpoints(family, spark_working.prefill_lanes)),
                    costs.spark_workspace_bytes * options.spark_capacity_rows / 4096)));
        if let Err(error) = io {
            report.placement_supported = false;
            notes.push(format!("{} I/O admission: {error}", device.name()));
        }
    }
    if report.encoder.is_none() {
        resolve_encoder(report, &towers, &mut devices, &mut spark_devices, &vec![0; gpus], options, &mut notes);
    }
    devices.extend(spark_devices);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// One pinned, mapped host copy of the untied token embedding in place of its
/// device allocations (`--embedding-placement host`), when the checkpoint has
/// exactly one embedding distinct from its LM head.
pub(crate) fn place_host_embedding(report: &PlanReport, checkpoint: &super::Checkpoint, devices: &mut [DeviceLayout],
    waste: &mut Vec<Waste>, notes: &mut Vec<String>) {
    let embedding = report.components.iter().find(|c| c.component == Component::Embedding);
    let names: Vec<_> = checkpoint.tensors.iter().filter(|t|
        t.meta.name == "embed.weight" || t.meta.name.ends_with("embed_tokens.weight")).collect();
    let Some(tensor) = names.first().filter(|_| names.len() == 1) else {
        notes.push("host embedding ineligible: no unique token embedding; GPU residency retained".into());
        return;
    };
    match checkpoint.require_untied_embedding(&tensor.meta.name) {
        Ok(()) => {
            let mut saved = 0;
            for device in devices.iter_mut() {
                device.items.retain(|item| {
                    if item.category == Category::Embedding { saved += item.bytes; false } else { true }
                });
            }
            waste.retain(|w| !w.what.starts_with("embedding"));
            notes.push(format!("embedding placement: host (single pinned mapped copy, {} bytes backing, {saved} device bytes freed)",
                embedding.map_or(tensor.meta.byte_length, |c| c.bytes)));
        }
        Err(error) => notes.push(format!("host embedding ineligible: {error:#}; GPU residency retained")),
    }
}

/// Exact resident weight groups `(group, format, bytes)` per coordinator rank,
/// each under the category `category` names for its group.
pub(crate) fn push_resident(devices: &mut [DeviceLayout], ranks: &[Vec<(String, String, u64)>],
    category: impl Fn(&str) -> Category) {
    for (device, rank) in devices.iter_mut().zip(ranks) {
        for (group, format, bytes) in rank {
            device.items.push(Item::new(category(group), group.clone(), format.clone(), *bytes, Basis::Exact));
        }
    }
}

/// The host-mapped table note (engram/PLE tables live on the host; the
/// device budget holds only their staging).
pub(crate) fn mapped_table_note(report: &PlanReport, notes: &mut Vec<String>) {
    let mapped: u64 = report.components.iter().filter(|c| c.component == Component::MappedTable).map(|c| c.bytes).sum();
    if mapped > 0 { notes.push(format!("host-mapped table backing: {:.2} GiB (device budget includes staging only)", mapped as f64 / GIB as f64)); }
}

/// What a family charges on each Spark rank beside its routed slices and the
/// selected rank I/O (`charge_spark_io`), until exact Spark admission (P11)
/// replaces these with the worker's own measurement.
#[derive(Debug, Clone, Copy)]
pub struct SparkCharges {
    /// Observed host footprint, charged against physical memory (the usable
    /// inventory's 13 GiB host exclusion is added back to capacity).
    pub host_bytes: Option<u64>,
    /// Non-I/O wave residual at 4096 capacity rows.
    pub wave_bytes: u64,
    /// Basis of the host and runtime charges.
    pub basis: Basis,
}

/// Each Spark rank's routed slice bytes from the checkpoint (`routed` source
/// bytes): exact 128-row block packages store their own widths, other
/// packages pad every rank to the widest slice (reported as waste).
pub(crate) fn spark_slices(report: &PlanReport, model: &dyn super::FamilyModel, ranks: usize, routed: u64)
    -> (Vec<u64>, Option<Waste>) {
    let stored = (routed as f64 * report.spark_rank_share) as u64;
    let even = routed / ranks.max(1) as u64;
    let package = report.experts.as_ref().map_or("", |e| e.package.as_str());
    let intermediate = model.spec().moe.as_ref().map_or(0, |m| m.intermediate);
    let recipe = (!package.starts_with("v41")).then(||
        super::experts::spark_recipe(package, intermediate, ranks).ok()).flatten();
    let exact = recipe.as_ref().filter(|recipe|
        matches!(recipe.slicing, crate::formats::fp8_experts::Slicing::Blocks(_)));
    let bytes = (0..ranks).map(|rank| exact.map_or(stored, |recipe|
        (routed as f64 * recipe.widths[rank] as f64 / intermediate as f64) as u64)).collect();
    let waste = (exact.is_none() && report.spark_rank_share * ranks as f64 > 1.001).then(|| Waste {
        device: format!("spark x{ranks}"), what: format!("routed slices padded to the widest \
            128-row slice ({:.1}% of the even share) on every rank", 100.0 * (stored - even) as f64 / even as f64),
        bytes: (stored - even) * ranks as u64 });
    (bytes, waste)
}

/// The routed experts' Spark geometry.
pub(crate) fn spark_geometry(model: &dyn super::FamilyModel) -> Option<cuteafd_core::ExpertGeometry> {
    model.spec().moe.as_ref().map(|moe| cuteafd_core::ExpertGeometry {
        hidden: model.spec().hidden as u32, experts: moe.experts as u32,
        topk: moe.top_k as u32, intermediate: moe.intermediate as u32,
        layers: model.spec().layers.len() as u32,
    })
}

/// The Spark ranks of `report.placement`: host footprint, routed slices and
/// worker runtime, then the selected rank I/O for `working`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spark_devices(report: &mut PlanReport, model: &dyn super::FamilyModel, checkpoint: &super::Checkpoint,
    options: &LayoutOptions, charges: SparkCharges, working: &crate::placement::SparkWorkingSet, endpoints: usize,
    waste: &mut Vec<Waste>, notes: &mut Vec<String>) -> Vec<DeviceLayout> {
    let ExpertPlacement::Sparks { ranks } = report.placement else { return Vec::new() };
    let routed: u64 = report.components.iter().filter(|c| c.owner == Owner::SparkSliced
        && c.component != Component::SpeculatorExpert).map(|c| c.bytes).sum();
    let format = report.components.iter().find(|c| c.owner == Owner::SparkSliced)
        .map(|c| c.formats.keys().cloned().collect::<Vec<_>>().join("+")).unwrap_or_default();
    let (slices, padded) = spark_slices(report, model, ranks, routed);
    let mut devices = slices.into_iter().enumerate().map(|(rank, stored)| {
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: options.spark_bytes,
            items: Vec::new(), kv_tokens: 0 };
        if let Some(host) = charges.host_bytes {
            device.capacity_bytes += 13 * GIB;
            device.items.push(Item::new(Category::Reserved, "host OS+sparknestd", "", host, charges.basis));
        }
        device.items.push(Item::new(Category::Experts, "routed_expert", format.clone(), stored, Basis::Exact));
        device
    }).collect::<Vec<_>>();
    for device in &mut devices {
        charge_spark_devices(report, model, checkpoint, options, charges, working, endpoints,
            std::slice::from_mut(device), notes, |device| format!("Spark rank {} I/O admission", device.index));
        device.items.push(Item::new(Category::Runtime, "context+modules", "", 512 * MIB, charges.basis));
    }
    waste.extend(padded);
    devices
}

/// (Re)charges each Spark rank's selected I/O for `working`, after family
/// admission has selected its Spark layers and lanes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn charge_spark_devices(report: &mut PlanReport, model: &dyn super::FamilyModel, checkpoint: &super::Checkpoint,
    options: &LayoutOptions, charges: SparkCharges, working: &crate::placement::SparkWorkingSet, endpoints: usize,
    devices: &mut [DeviceLayout], notes: &mut Vec<String>, what: impl Fn(&DeviceLayout) -> String) {
    for device in devices {
        let io = spark_geometry(model).ok_or(crate::placement::PlacementError::Inventory("Spark experts need routed geometry"))
            .and_then(|geometry| u32::try_from(options.spark_capacity_rows)
                .map_err(|_| crate::placement::PlacementError::Overflow("Spark capacity"))
                .and_then(|capacity| charge_spark_io(device, working, geometry, capacity,
                    checkpoint.spark_input_is_bf16(), spark_ring_geometry(endpoints),
                    charges.wave_bytes * options.spark_capacity_rows / 4096)));
        if let Err(error) = io {
            report.placement_supported = false;
            notes.push(format!("{}: {error}", what(device)));
        }
    }
}

/// The program family (manifest name prefix) a family's coordinator loads, or
/// `None` where it still loads every program in the image (its context stays the allowance).
fn program_base(family: &str, model: &dyn super::FamilyModel, checkpoint: &super::Checkpoint) -> Option<&'static str> {
    Some(match family {
        "mimo_v2" => crate::families::mimo_v2::MimoV2Config::from_hf(&checkpoint.config).ok()?.program_family().ok()?,
        "glm5_flash" => "glmf",
        _ => return None,
    })
}

/// The local expert package kind a coordinator GPU loads (`LoadedCode::experts`): `exl3`, `fp8`
/// (fp8moe/NVFP4 packages) or `none` (Spark experts).
pub(crate) fn local_package(report: &PlanReport) -> &'static str {
    let package = report.experts.as_ref().map(|e| e.package.as_str()).unwrap_or("");
    if report.placement != ExpertPlacement::Local { "none" }
    else if package.contains("exl3") { "exl3" } else { "fp8" }
}

/// The planner's runtime baseline on a GPU of `total_bytes`: arch context + the selected modules.
fn planned_context(manifest: Option<&serde_json::Value>, families: &[String], total_bytes: u64) -> u64 {
    let names = families.iter().map(String::as_str).collect::<Vec<_>>();
    let modules = manifest.map_or(32 * MIB, |m| crate::placement::ProgramSet::from_manifest(m, &names).module_bytes());
    let arch = crate::placement::ArchContext::coordinator(total_bytes, None);
    arch.context_bytes + arch.cublas_bytes + modules
}

/// GLM 5.3 Flash's step workspaces on one GPU from its program manifest: the bytes its engine
/// allocates for the decode workspace of `decode_rows` rows and `lanes` prefill lanes of `rows` rows,
/// whose scratch also holds `shared_records` bytes of replay records (`--replay-records shared`)
/// (`serving_capacity::glmf_*`, which the engine sizes its buffers from).
#[allow(clippy::too_many_arguments)]
fn glmf_step_workspace(manifest: &serde_json::Value, checkpoint: &super::Checkpoint, placement: &ExpertPlacement,
    lanes: u64, rows: u64, context: u64, decode_rows: u64, shared_records: u64, index_compact: bool, split: bool,
    repr: crate::families::glm5_flash::resident::GlmfRepresentation, expanded: bool, rtx_experts: bool, fp32_partials: bool,
    kda_state: crate::serving_capacity::GlmfKdaState) -> Option<Vec<crate::placement::families::glm5_flash::GlmfBufferDemand>> {
    use crate::serving_capacity::{glmf_manifest_scratch, glmf_step_scratch, glmf_table_pages,
        GlmfScratchOptions, GlmfStepShape};
    let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&checkpoint.config).ok()?;
    let lookup = glmf_manifest_scratch(manifest);
    // Requested precision must have its programs; never account for a silent BF16 fallback.
    let options = GlmfScratchOptions { index_compact, split, kda_fp32_partials: fp32_partials, kda_output_shard: split && repr.output_shard, kda_prefill_expanded: split && expanded,
        kda_w8: repr.kda_fp8, kda_state,
        ..Default::default() };
    let context = if context > 0 { context } else { manifest["capacities"]["max_context"].as_u64().unwrap_or(131_072) };
    let (table_pages, table_pool_pages) = glmf_table_pages(context);
    let spark = matches!(placement, ExpertPlacement::Sparks { .. });
    let shape = GlmfStepShape { lead: true, split, local_experts: !split && rtx_experts, tp2_experts: split && rtx_experts, spark, partial_bytes: if fp32_partials { 4 } else { 2 },
        output_shard: split && repr.output_shard, full_prefill_logits: false, table_pages, table_pool_pages };
    let decode = glmf_step_scratch(&lookup, &cfg, options, decode_rows, true).ok()?;
    let mut prefill = glmf_step_scratch(&lookup, &cfg, options, rows, false).ok()?;
    // Shared replay records live in the prefill scratch.
    prefill.programs = prefill.programs.max(shared_records);
    Some(crate::placement::families::glm5_flash::step_workspace_demands(&cfg,
        usize::try_from(lanes).ok()?, rows, decode_rows, shape, decode, prefill))
}

/// Local EXL3 arenas from the same capacity manifests used by the loader.
/// An exported PROGRAMS.json can have its expert JSON tree alongside it;
/// in an image the standard tree lives in ../lib/exl3 instead.
pub(crate) fn expert_workspace(report: &PlanReport, model: &dyn super::FamilyModel, input: &super::PlanInput,
    manifest: Option<&std::path::Path>, rows: u64) -> Option<u64> {
    if !report.experts.as_ref()?.package.contains("exl3") { return None; }
    // Display labels name the checkpoint tier; the decoder can include an
    // adjacent tier too (Pro K2 uses its k23 package). Read the catalog contract.
    let catalog = input.expert_catalog().ok()?;
    let tiers = catalog.exl3()?.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let family = match report.family.as_deref()? {
        "qwen4" => "qwen4",
        "deepseek_v4" if model.spec().hidden == 4096 => "dsv4f",
        "deepseek_v4" => "dsv4p",
        _ => return None,
    };
    let parent = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json")).parent()?;
    let stem = format!("exl3-{family}-k{tiers}");
    let root = [parent.join("exl3").join(&stem), parent.join("../lib/exl3").join(&stem)]
        .into_iter().find(|p| p.join("rtx-tp1/m4096/v41_exl3.json").is_file())?;
    let maximum = if family.starts_with("dsv4") { rows.max(64) } else { rows.max(1) };
    if maximum > 4096 { return None; }
    let manifests = crate::placement::inventory::exl3_capacities(maximum).into_iter()
        .map(|capacity| serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(root.join(format!("rtx-tp1/m{capacity}/v41_exl3.json"))).ok()?).ok())
        .collect::<Option<Vec<_>>>()?;
    let moe = model.spec().moe.as_ref()?;
    if manifests.iter().any(|m| m["hidden"].as_u64() != Some(model.spec().hidden as u64)
        || m["intermediate"].as_u64() != Some(moe.intermediate as u64)
        || m["experts"].as_u64() != Some(moe.experts as u64)) { return None; }
    crate::serving_capacity::exl3_workspace_bytes(&manifests, true).ok()?
        .checked_add(maximum * model.spec().hidden as u64 * 2)
}

// The host footprint belongs to physical capacity, not expertd's allocation
// budget. Neither the tower nor its admission guard may displace experts.
fn spark_encoder_headroom(device: &DeviceLayout, budget: Option<u64>) -> u64 {
    let physical = device.free_bytes().max(0) as u64;
    let allocated: u64 = device.items.iter().filter(|i| i.category != Category::Reserved)
        .map(|i| i.bytes).sum();
    budget.map_or(physical, |bytes| physical.min(bytes.saturating_sub(allocated)))
}

/// The bundled media towers a family runs, as the encoder placement charges them.
#[derive(Debug, Clone)]
pub struct Towers {
    /// The family name a refused tower's hint carries.
    pub name: &'static str,
    /// Vision tower resident weights and scratch, or why the checkpoint's tower
    /// cannot run (refused when vision is requested, else charged as nothing).
    pub vision: Result<(u64, u64), String>,
    /// A Spark-hosted tower also charges V4.1's measured SM121 CUDA overhead,
    /// and an automatic RTX tower prefers rtx0.
    pub v41: bool,
    /// The bundled audio tower (weights, scratch), or why none runs.
    pub audio: Result<(u64, u64), String>,
}

impl Towers {
    /// A tower charged at its checkpoint bytes with a 512 MiB scratch, no audio.
    pub fn checkpoint(report: &PlanReport) -> Self {
        Self { name: "", vision: Ok((vision_source_bytes(report), 512 * MIB)), v41: false, audio: no_audio() }
    }
}

/// The checkpoint bytes of the vision tower (0 when absent or disabled).
pub fn vision_source_bytes(report: &PlanReport) -> u64 {
    report.components.iter().find(|c| c.component == Component::Vision).map_or(0, |c| c.bytes)
}

pub fn no_audio() -> Result<(u64, u64), String> {
    Err("checkpoint family has no bundled audio runtime".into())
}

/// The bundled FP32 audio tower's weights and its scratch at the longest clip.
pub fn fp32_audio_tower(checkpoint: &super::Checkpoint) -> Result<(u64, u64), String> {
    let plan = crate::media::audio_tower::AudioTowerPlan::from_snapshot(&checkpoint.snapshot,
        crate::media::audio_tower::AudioStorage::Fp32).map_err(|e| e.to_string())?;
    plan.scratch_bytes(crate::media::audio::MAX_CLIP_SAMPLES)
        .map(|scratch| (plan.weight_bytes(), scratch)).map_err(|e| e.to_string())
}

/// MiMo's bundled towers: FP32 vectors in the measured 4096-token native
/// ledger (fixed 4096-row GEMM staging and the 4 MiB BLAS workspace), and its
/// FP32 audio tower.
pub fn mimo_towers(report: &PlanReport, checkpoint: &super::Checkpoint, hidden: u64) -> Towers {
    let source = vision_source_bytes(report);
    Towers { name: "MiMo", vision: Ok(if source > 0 {
        (1_458_170_944 + hidden.saturating_sub(4096) * 5120 * 2, mimo_scratch_bytes(hidden))
    } else { (0, 512 * MIB) }), v41: false, audio: fp32_audio_tower(checkpoint) }
}

/// Qwen's resident tower at its image-token capacity.
pub fn qwen_towers(report: &PlanReport) -> Towers {
    let source = vision_source_bytes(report);
    Towers { name: "Qwen", vision: Ok(if source > 0 {
        (898_680_904, qwen_scratch_bytes(crate::media::QWEN_MAX_IMAGE_TOKENS as u64))
    } else { (0, 512 * MIB) }), v41: false, audio: no_audio() }
}

/// GLM Flash's official BF16 tower (refused for any other geometry).
pub fn glm_flash_towers(report: &PlanReport, checkpoint: &super::Checkpoint) -> Towers {
    let vision = if vision_source_bytes(report) > 0 {
        super::families::glm::glm_flash_vision_resident_weights(checkpoint)
            .map(|weights| (weights, GLM_FLASH_ENCODER_SCRATCH_BYTES))
    } else { Ok((0, 512 * MIB)) };
    Towers { name: "GLM Flash", vision, v41: false, audio: no_audio() }
}

/// V4.1's tower: checkpoint weights, its exact 9216-patch scratch and the
/// SM121 CUDA overhead when Spark-hosted.
pub fn v41_towers(report: &PlanReport) -> Towers {
    let source = vision_source_bytes(report);
    Towers { name: "V4.1", vision: Ok((source, if source > 0 { v41::vision_scratch_bytes() } else { 512 * MIB })),
        v41: true, audio: no_audio() }
}

/// Places the media towers beside a layout's devices: `kv` is each GPU's KV
/// target the tower must leave room for.
#[allow(clippy::too_many_arguments)]
pub fn resolve_encoder(report: &mut PlanReport, towers: &Towers, rtx: &mut [DeviceLayout],
    sparks: &mut [DeviceLayout], kv: &[u64], options: &LayoutOptions, notes: &mut Vec<String>) {
    use super::encoder::*;
    let (weights, scratch) = match &towers.vision {
        Ok(bytes) => *bytes,
        Err(reason) if report.vision != super::MediaMode::Off => {
            report.placement_supported = false;
            report.hints.push(super::Hint {
                what: format!("{} vision tower unavailable: {reason}", towers.name),
                how: "Use the qualified official BF16 tower geometry or explicitly use --vision=off.".into(),
            });
            notes.push(format!("vision Off: {reason}; 0 bytes admitted"));
            report.encoder = Some(EncoderPlacement {
                kind: EncoderKind::Off, weights: 0, scratch: 0, replicas: Vec::new(), reason: reason.clone(), shortfall: 0,
            });
            return;
        }
        Err(_) => (0, 0),
    };
    let hardware = EncoderHardware { v41: towers.v41,
        gpus: rtx.iter().enumerate().map(|(i,d)| EncoderGpuBudget { free_bytes: d.free_bytes().max(0) as u64, kv_target_bytes: kv.get(i).copied().unwrap_or(0) }).collect(),
        sparks: sparks.iter().map(|d| EncoderSparkBudget { rank: d.index as usize, host: format!("spark{}",d.index), idle: false,
            expert_bytes: d.items.iter().filter(|i| i.category == Category::Experts).map(|i| i.bytes).sum(), free_bytes: spark_encoder_headroom(d, options.spark_allocation_budget_bytes) }).collect() };
    let mut placement = encoder_placement(report.vision, &hardware, weights, scratch, options.vision_replicas);
    // Launch starts a Spark worker only for an expert rank and attaches the
    // encoder to that worker. No expert package means no worker to host it.
    let vision_only_spark = matches!(&placement.kind, EncoderKind::Off)
        && matches!(report.vision, super::MediaMode::Spark(_))
        && sparks.is_empty()
        && placement.reason == "requested Spark encoder/replicas unavailable";
    if vision_only_spark {
        placement.reason = "vision-only Spark workers are not supported; set VISION=rtx or RTX_EXPERT_LAYERS".into();
        placement.shortfall = 0;
    }
    let add = |d: &mut DeviceLayout| {
        d.items.push(Item::new(Category::Weights, "vision tower", "BF16 + FP32 vectors", placement.weights, Basis::Formula));
        let spark_overhead = if hardware.v41 && d.kind == DeviceKind::Spark {
            super::encoder::V41_SPARK_CUDA_OVERHEAD_BYTES
        } else { 0 };
        d.items.push(Item::new(Category::Workspace, "vision scratch", "resident", placement.scratch.saturating_sub(spark_overhead), Basis::Formula));
        if spark_overhead > 0 {
            d.items.push(Item::new(Category::Runtime, "vision CUDA overhead", "SM121 measured delta + margin", spark_overhead, Basis::Formula));
        }
    };
    match placement.kind {
        EncoderKind::Rtx { gpu } => add(&mut rtx[gpu]),
        EncoderKind::Spark { rank } => {
            for d in sparks.iter_mut().filter(|d| d.index as usize == rank || placement.replicas.contains(&(d.index as usize))) { add(d); }
        }
        EncoderKind::Off if matches!(report.vision, super::MediaMode::Rtx(_) | super::MediaMode::Spark(_)) => {
            report.placement_supported = false;
            let (what, how) = if vision_only_spark {
                (placement.reason.clone(), "Place the tower on an RTX with VISION=rtx, or keep expert layers on Sparks with RTX_EXPERT_LAYERS.".into())
            } else {
                (format!("requested vision placement unavailable: {}", placement.reason),
                    format!("Select an available encoder device with tower/scratch/KV headroom (shortfall {} bytes), or explicitly use --vision=off.", placement.shortfall))
            };
            report.hints.push(super::Hint { what, how });
        }
        EncoderKind::Off => {
            if let Some(c) = report.components.iter_mut().find(|c| c.component == Component::Vision)
                .filter(|c| c.status != Status::Unused) {
                report.disabled_media_bytes += c.bytes;
                c.bytes = 0;
                c.status = Status::Disabled;
            }
        }
        EncoderKind::SparkIdle { .. } => {},
    }
    notes.push(format!("vision {:?}: {}; {} bytes admitted; shortfall {} bytes", placement.kind, placement.reason, placement.admitted_bytes(), placement.shortfall));
    report.encoder = Some(placement);
    if report.audio == super::MediaMode::Off { return; }
    let (weights, scratch) = match &towers.audio {
        Ok(bytes) => *bytes,
        Err(reason) => {
            report.placement_supported = false;
            report.hints.push(super::Hint { what: format!("audio tower unavailable: {reason}"),
                how: "Use --audio off or add the bundled tensors/exporter".into() });
            report.audio_encoder = Some(EncoderPlacement { kind: EncoderKind::Off, weights: 0, scratch: 0,
                replicas: vec![], shortfall: 0, reason: reason.clone() });
            return;
        }
    };
    // Vision was admitted first: audio sees its reservation and may choose another Spark.
    let hardware = EncoderHardware { v41: false,
        gpus: rtx.iter().enumerate().map(|(i,d)| EncoderGpuBudget { free_bytes: d.free_bytes().max(0) as u64,
            kv_target_bytes: kv.get(i).copied().unwrap_or(0) }).collect(),
        sparks: sparks.iter().map(|d| EncoderSparkBudget { rank: d.index as usize, host: format!("spark{}", d.index),
            idle: false, expert_bytes: d.items.iter().filter(|i| i.category == Category::Experts).map(|i| i.bytes).sum(),
            free_bytes: spark_encoder_headroom(d, options.spark_allocation_budget_bytes) }).collect() };
    let placement = encoder_placement(report.audio, &hardware, weights, scratch, 1);
    let add = |d: &mut DeviceLayout| {
        d.items.push(Item::new(Category::Weights, "audio tower", "FP32 native", placement.weights, Basis::Formula));
        d.items.push(Item::new(Category::Workspace, "audio scratch", "resident BLAS/FFT included", placement.scratch, Basis::Formula));
    };
    match placement.kind {
        EncoderKind::Rtx { gpu } => add(&mut rtx[gpu]),
        EncoderKind::Spark { rank } => { if let Some(d) = sparks.iter_mut().find(|d| d.index as usize == rank) { add(d); } },
        EncoderKind::Off if report.audio == super::MediaMode::Auto => {
            report.audio = super::MediaMode::Off;
            if let Some(c) = report.components.iter_mut().find(|c| c.component == Component::Audio) {
                report.disabled_media_bytes += c.bytes;
                c.bytes = 0;
                c.status = Status::Disabled;
            }
        }
        _ => {
            report.placement_supported = false;
            report.hints.push(super::Hint { what: format!("audio placement unavailable: {}", placement.reason),
                how: "Select an encoder with tower + scratch + guard headroom, or --audio off".into() });
        }
    }
    notes.push(format!("audio {:?}: {}; {} bytes admitted; shortfall {} bytes", placement.kind,
        placement.reason, placement.admitted_bytes(), placement.shortfall));
    report.audio_encoder = Some(placement);
}

pub(crate) fn mimo_scratch_bytes(width: u64) -> u64 {
    let n = 4096 * 4;
    [n*1280*4,n*1280*4,n*1280*2,n*1536*2,n*2048*2,n*512*2,n*512*2,n*2048*2,4096*width*2,
     4096*3072*4,4096*1280*4,4096*2*4608*4,4096*4608*2,4096*5120*4,4096*5120*2,4096*width*4,
     n*16*16*3,3*256*4,n*2*4,n*2*4,4096*4,4096*4,4*MIB].into_iter().map(|b| b.div_ceil(256)*256).sum()
}

/// Mirrors scratch_qwen, including fixed staging, LSE and the BLAS workspace.
pub(crate) fn qwen_scratch_bytes(tokens: u64) -> u64 {
    let n = tokens * 4;
    [n*1152*4, n*1152*4, n*1152*2, n*1536*2, n*1152*2, n*1152*2,
     n*1152*2, n*1152*2, tokens*2560*2, 4096*3456*4, 4096*1152*4,
     4096*4304*4, 4096*4304*2, 4096*4608*4, 4096*4608*2, 4096*2560*4,
     n*16*16*3, 3*256*4, n*2*4, n*4*4, n*4*4, 16384*16*4, 2*4, 4*MIB]
        .into_iter().map(|b| b.div_ceil(256)*256).sum()
}
