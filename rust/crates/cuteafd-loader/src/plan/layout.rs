//! Planner layout core: the options every family adapter reads and the
//! family-neutral pieces they share (step shape, runtime item, media towers,
//! Spark rank devices and their selected I/O). Each family lays itself out in
//! its own adapter under `layout/` (`FamilyModel::placement`), solving the
//! same placement request its runtime admits.
use super::{Component, ExpertPlacement, Owner, PlanReport, Status};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout, Waste};

pub(crate) mod deepseek_v4;
pub(crate) mod glm5_flash;
pub(crate) mod v41;
pub(crate) mod mimo;
pub mod glm5;
pub(crate) mod qwen;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
/// Historical offline GB10 device capacity estimate, not a measured admission.
/// Kept distinct from the family's weights-only default until load/steady census.
pub const DEFAULT_SPARK_DEVICE_BYTES: u64 = 108 * GIB;
// 2026-10-07 SM120/SM121 resident ledger at the fixed 4096-image-token capacity:
// native scratch_glm's aligned arenas, including its dedicated 4 MiB BLAS workspace.
const GLM_FLASH_ENCODER_SCRATCH_BYTES: u64 = 791_907_584;

/// The hardware and serving shape to lay out.
#[derive(Debug, Clone)]
pub struct LayoutOptions {
    /// None sizes full-context KV for concurrency + 4, capped at min(32 GiB, 25% host RAM).
    /// Explicit bytes bypass both; Some(0) disables the host tier.
    pub host_cache_bytes: Option<u64>,
    /// None selects min(8 GiB, 5% host total) when a supported media tower is loaded.
    pub media_cache_bytes: Option<u64>,
    /// Deterministic host/cgroup sample for offline inventory tests.
    pub host_memory: Option<(u64, u64)>,
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
    /// Family-resolved rows per wave, independent of the admitted span.
    pub prefill_unit_rows: Option<u64>,
    /// Requested rows per continuous prefill call (units-span families).
    pub prefill_span_rows: Option<u64>,
    /// Continuous row-unit waves on one transport.
    pub prefill_waves: u64,
    /// Family-resolved decode views; unrelated to prefill wave depth.
    pub decode_lanes: u8,
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
    pub dense_qwen_drafter_snapshot: Option<std::path::PathBuf>,
    pub dense_qwen_mtp: bool,
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
    pub glm5_attention_plan_rows: Option<usize>,
    /// Kimi MLA DFlash is opt-in and charged from its native checkpoint headers.
    pub kimi_drafter_snapshot: Option<std::path::PathBuf>,
    pub kimi_draft_fp8: bool,
    pub mla_kv: cuteafd_core::MlaCacheFormat,
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
    /// Sliding/native-MTP ring storage, independent of the full-attention pool.
    pub mimo_swa_kv: crate::families::mimo_v2::MimoKvCache,
    /// MiMo Spark activation format, matching serving's --expert-input.
    pub mimo_expert_input: crate::families::mimo_v2::MimoExpertInput,
    /// MiMo drafter batch limit (serving raises it to concurrency).
    pub draft_sequences: u64,
    /// Shared GQA drafter rings, independent of target KV and weight precision.
    pub draft_kv_cache: crate::families::qwen4::Qwen4KvCache,
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
    pub dense_qwen_kv: crate::families::qwen4::Qwen4KvCache,
    pub qwen_kv: crate::families::qwen4::Qwen4KvCache,
    pub step5_kv: crate::families::qwen4::Qwen4KvCache,
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
            host_cache_bytes: None,
            media_cache_bytes: None,
            host_memory: None,
            rtx_bytes: vec![crate::placement::inventory::PRO_TOTAL_BYTES],
            vision_replicas: 1,
            host_embedding: false,
            force_gpu_embedding: false,
            spark_bytes: DEFAULT_SPARK_DEVICE_BYTES,
            spark_allocation_budget_bytes: None,
            head_split: true,
            prefill_rows: 0,
            full_prefill_logits: false,
            prefill_lanes: 0,
            prefill_unit_rows: None,
            prefill_span_rows: None,
            prefill_waves: 1,
            decode_lanes: 1,
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
            dense_qwen_drafter_snapshot: None,
            dense_qwen_mtp: false,
            glmf_drafter_disabled: false,
            glmf_drafter_snapshot: None,
            glmf_dense_manifest: None,
            glm5_drafter_snapshot: None,
            glm5_drafter_disabled: false,
            glm5_draft_bf16: false,
            glm5_attention_plan_rows: None,
            kimi_drafter_snapshot: None,
            kimi_draft_fp8: false,
            mla_kv: Default::default(),
            headroom_bytes: 2 * GIB,
            concurrency: 0,
            state_slots: None,
            glmf_mark_lanes: None,
            prefix_slots: None,
            mimo_prefix_entries: 20,
            mimo_prefix_mark_bytes: 2 * GIB,
            mimo_prefix_draft: false,
            mimo_rings: 16,
            mimo_swa_kv: crate::families::mimo_v2::MimoKvFormats::default().sliding,
            mimo_expert_input: Default::default(),
            draft_sequences: 4,
            draft_kv_cache: crate::families::qwen4::Qwen4KvCache::Fp8,
            draft_context_slots: None,
            mimo_decode_graphs: false,
            physical_sms: None,
            mimo_draft_bf16: false,
            mimo_expert_manifest: None,
            context_tokens: 0,
            native_mtp_layers: 3,
            dense_qwen_kv: crate::families::qwen4::Qwen4KvCache::Fp8,
            qwen_kv: Default::default(),
            step5_kv: crate::families::qwen4::Qwen4KvCache::Fp8,
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

/// GLM prefill units in flight: the default (None) or a parsed value, clamped
/// to 1..=4. Default: 2 units of 4096 rows (PLAN "Prefill contract"); serve-glm
/// takes `--prefill-waves`, the planner `--prefill-lanes`.
pub const GLM_DEFAULT_PREFILL_SPAN_ROWS: u64 = 16384;

pub fn glm_prefill_lanes(value: Option<&str>) -> usize {
    value.and_then(|v| v.parse().ok()).unwrap_or(2).clamp(1, 4)
}

/// Additional lead-GPU output bytes for admitted all-row fidelity probes.
/// GLM uses its default waves (a planner `--prefill-lanes` selects others); GLM Flash's lanes (a serial prefill runs in
/// the first) share one set of temporaries and so one logits buffer.
/// V4 reuses its bounded head buffer; V4.1 already supports prefill scoring.
pub fn full_prefill_logits_bytes(family: &str, rows: u64, vocab: u64) -> u64 {
    let lanes = glm_prefill_lanes(None);
    full_prefill_logits_bytes_with_lanes(family, rows, vocab, lanes)
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
    fn unknown_package_runtime_item_uses_manifest_formula() {
        let manifest = serde_json::json!({"programs":[{"name":"qwen4_decode"}]});
        let item = runtime_item(Some(&manifest), &["qwen4".into()], "unmeasured", false, 0, 96 << 30);
        let arch = crate::placement::ArchContext::for_device("sm_120", 96 << 30);
        assert_eq!(item.basis, Basis::Formula);
        assert_eq!(item.bytes, arch.context_bytes + arch.cublas_bytes + 34 * MIB);
        assert!(item.bytes > arch.context_bytes);
    }

    #[test]
    fn full_rows_are_opt_in_and_reserve_the_family_output_delta() {
        assert!(!LayoutOptions::default().full_prefill_logits);
        assert_eq!(full_prefill_logits_bytes("mimo_v2", 128, 1000), 127 * 1000 * 4);
        assert_eq!(full_prefill_logits_bytes("qwen4", 128, 1000), 127 * (1000 * 4 + 16));
        assert_eq!(glm_prefill_lanes(None), 2);
        for lanes in 1..=4 {
            assert_eq!(glm_prefill_lanes(Some(&lanes.to_string())), lanes);
            assert_eq!(full_prefill_logits_bytes_with_lanes("glm5", 128, 1000, lanes),
                lanes as u64 * 64 * 1000 * 4);
        }
        assert_eq!(glm_prefill_lanes(Some("0")), 1);
        assert_eq!(glm_prefill_lanes(Some("9")), 4);
        assert_eq!(glm_prefill_lanes(Some("invalid")), 2);
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

/// Ring endpoints a family's coordinator opens to each selected Spark rank,
/// given its prefill lanes (V4/V4.1: two fixed; GLM 5.3: one decode endpoint
/// plus one per lane when lanes run in parallel; GLM Flash/MiMo: one per lane;
/// Qwen: one).
pub fn spark_endpoints(family: &str, lanes: u64) -> usize {
    match family {
        "deepseek_v41" | "deepseek_v4" => 2,
        // The decode link and one prefill link (its waves share that endpoint's rings).
        "glm5" => 1 + usize::from(lanes > 1),
        "glm5_flash" => 1,
        "mimo_v2" => lanes as usize,
        "qwen4" => 1,
        _ => 0,
    }
}

/// Execution slots each Spark rank runs for a family: the most requests the
/// coordinator keeps in flight to one rank. Prefill waves and decode lanes
/// never share a rank at once (each family's scheduler stops decode for
/// admission), so the bound is the larger of the two, not their sum.
pub fn spark_execution_slots(family: &str, prefill_waves: u64, decode_lanes: u8) -> u8 {
    let prefill = match family {
        // V4.1's two transports carry both prefill encoder streams at once.
        "deepseek_v41" => 2,
        "glm5" | "glm5_flash" | "mimo_v2" | "kimi_k3" => prefill_waves.max(1),
        _ => 1,
    };
    let decode = u64::from(decode_lanes.max(1));
    u8::try_from(prefill.max(decode).min(u64::from(cuteafd_core::expert_geometry::MAX_SPARK_EXECUTION_SLOTS)))
        .unwrap_or(cuteafd_core::expert_geometry::MAX_SPARK_EXECUTION_SLOTS)
}

/// Device bytes of one Spark execution slot (stream, scratch, inputs and
/// staging at the family's unit capacity), calibrated against the workers'
/// `experts/workspace*` ledger on the shipped cards (2026-10-24): MiMo Pro max
/// FP8 146 MiB, GLM 5.3 EXL3 max 1001 MiB, GLM Flash EXL3 569 MiB, V4.1 Flash
/// max 208 MiB, Kimi K3 EXL3-K2 1175 MiB. Scaled by unit rows over 4096.
pub fn spark_slot_bytes(family: &str, unit_rows: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    let at_4096 = match family {
        "mimo_v2" => 146 * MIB,
        "glm5" => 1001 * MIB,
        "glm5_flash" => 569 * MIB,
        "deepseek_v41" | "deepseek_v4" => 208 * MIB,
        "kimi_k3" => 1175 * MIB,
        _ => 0,
    };
    at_4096.saturating_mul(unit_rows.max(1)).div_ceil(4096)
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
        | "host expert exchange" | "expert row indices" | "endpoint response prefixes" | "rdma rings")
        && item.group != cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_GROUP);
    if working.selected_ranks().contains(&rank) {
        device.items.push(cuteafd_core::memory_layout::spark_runtime_residual());
        device.items.push(Item::new(Category::Workspace, "expert waves", "non-I/O residual",
            wave_allowance.saturating_sub(legacy_io), Basis::Calibrated));
        device.items.push(Item::new(Category::Workspace, "expert intake", if bf16 { "BF16" } else { "FP8 K32" },
            io.intake, Basis::Formula));
        device.items.push(Item::new(Category::Workspace, "host expert exchange", "", io.host_exchange, Basis::Formula));
        device.items.push(Item::new(Category::Workspace, "expert row indices", "", io.row_indices, Basis::Formula));
        device.items.push(Item::new(Category::Transport, "endpoint response prefixes", "", io.endpoint_staging, Basis::Formula));
        device.items.push(Item::new(Category::Transport, "rdma rings", "", io.mapped_rings, Basis::Formula));
    }
    Ok(())
}

/// Charges each selected Spark rank's execution slots beyond the first (the
/// first is inside the family's wave allowance): `(slots - 1)` slot workspaces.
pub(crate) fn charge_spark_slots(device: &mut DeviceLayout, working: &crate::placement::SparkWorkingSet,
    family: &str, slots: u8, unit_rows: u64) {
    device.items.retain(|item| item.group != "execution slots");
    if working.selected_ranks().contains(&(device.index as usize)) && slots > 1 {
        device.items.push(Item::new(Category::Workspace, "execution slots", format!("{} more", slots - 1),
            spark_slot_bytes(family, unit_rows) * u64::from(slots - 1), Basis::Calibrated));
    }
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
            // Decode link + one prefill link (formerly one endpoint per lane: 2459697152).
            ("glm5", ExpertGeometry::GLM5, 3, 1229848576),
            ("mimo_v2", ExpertGeometry::MIMO_V26_PRO, 3, 1844772864),
        ] {
            let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4,
                prefill_lanes: lanes, ..Default::default() };
            let rings = spark_ring_geometry(spark_endpoints(family, lanes));
            let io = working.rank_io(0, geometry, 4096, false, rings).unwrap();
            assert_eq!(io.mapped_rings, expected);
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0,
                capacity_bytes: 121 * GIB, items: vec![], kv_tokens: 0 };
            charge_spark_io(&mut device, &working, geometry, 4096, false, rings, GIB).unwrap();
            assert_eq!(device.items.iter().find(|item| item.group == "rdma rings").unwrap().bytes, expected);
            let residual = device.items.iter().find(|item|
                item.group == cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_GROUP).unwrap();
            assert_eq!(residual.bytes, 5 * (1 << 29));
            assert_eq!(residual.basis, Basis::Estimated);
            assert_eq!(residual.lifetime, cuteafd_core::memory_layout::ItemLifetime::ReadyResident);
            assert!(cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_SOURCE.contains("0a6fc7b1"));
        }
    }

    #[test]
    fn execution_slots_follow_admitted_waves_and_lanes_and_charge_their_unit_size() {
        assert_eq!(spark_execution_slots("glm5", 2, 1), 2);
        assert_eq!(spark_execution_slots("glm5", 4, 1), 4);
        assert_eq!(spark_execution_slots("mimo_v2", 3, 2), 3);
        assert_eq!(spark_execution_slots("mimo_v2", 1, 2), 2);
        assert_eq!(spark_execution_slots("deepseek_v41", 1, 2), 2);
        assert_eq!(spark_execution_slots("qwen4", 1, 1), 1);
        assert_eq!(spark_execution_slots("glm5", 9, 1), cuteafd_core::expert_geometry::MAX_SPARK_EXECUTION_SLOTS);
        // A smaller unit with more waves costs more, smaller slots (no hard-wired 4096 x 2).
        assert_eq!(spark_slot_bytes("glm5", 2048) * 2, spark_slot_bytes("glm5", 4096));
        let working = SparkWorkingSet { spark_layers: vec![3], spark_ranks: 4, prefill_lanes: 2, ..Default::default() };
        let charged = |slots, rows| {
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: 0, capacity_bytes: 121 * GIB,
                items: vec![], kv_tokens: 0 };
            charge_spark_slots(&mut device, &working, "glm5", slots, rows);
            device.items.iter().filter(|item| item.group == "execution slots").map(|item| item.bytes).sum::<u64>()
        };
        assert_eq!(charged(1, 4096), 0);
        assert_eq!(charged(2, 4096), spark_slot_bytes("glm5", 4096));
        assert_eq!(charged(4, 2048), 3 * spark_slot_bytes("glm5", 2048));
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
            let wave = GIB;
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
        assert_eq!(spark_endpoints("glm5", glm), 2);
        assert_eq!(spark_endpoints("glm5", crate::placement::families::glm5::prefill_lanes(0, 3)), 1);
    }
}

const fn gib(hundredths: u64) -> u64 {
    hundredths * GIB / 100
}

fn mimo_draft_prefix_bytes(config: &serde_json::Value, marks: u64, rings: u64, kv_cache: crate::families::qwen4::Qwen4KvCache) -> Result<(u64, u64), String> {
    let field = |name| config[name].as_u64().filter(|&n| n > 0)
        .ok_or_else(|| format!("missing or zero {name}"));
    let mark = crate::families::mimo_v2::draft_representation::mimo_draft_mark_bytes(field("num_hidden_layers")?, field("num_key_value_heads")?, field("head_dim")?, kv_cache)
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
    fn kimi_host_snapshots_use_serve_context_and_selected_draft_record() -> anyhow::Result<()> {
        use crate::plan::{self, testing, ExpertPlacement, PlanOptions};
        let target = tempfile::tempdir()?;
        testing::write_kimi_snapshot(target.path(), true);
        let checkpoint = super::super::Checkpoint::open(target.path())?;
        let draft = tempfile::tempdir()?;
        let config = serde_json::json!({"architectures":["DFlash2DraftModel"], "hidden_size":7168,
            "intermediate_size":14336, "num_hidden_layers":5, "num_attention_heads":64,
            "num_key_value_heads":64, "q_lora_rank":1536, "kv_lora_rank":512,
            "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128,
            "rms_norm_eps":1e-5, "vocab_size":163840, "draft_vocab_size":163840,
            "max_position_embeddings":1048576, "num_target_layers":93,
            "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","full_attention"], "sliding_window":4096,
            "use_sliding_window":true, "is_causal":false, "mla_use_output_gate":false,
            "attention_bias":false, "hidden_act":"silu", "fc_norm":false,
            "rope_parameters":{"rope_type":"yarn", "rope_theta":50000.0, "factor":32.0,
                "original_max_position_embeddings":32768, "beta_fast":32, "beta_slow":1,
                "mscale":1.0, "mscale_all_dim":1.0},
            "dflash_config":{"attention_mode":"mla", "block_size":8,
                "conv_kernel_size":2, "conv_group_size":16, "selector_rank":256,
                "selector_top_k":16, "target_layer_ids":[0,1], "mask_token_id":163592}});
        std::fs::write(draft.path().join("config.json"), config.to_string())?;
        for format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
            let mut options = LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; 2], concurrency: 2,
                mla_kv: format, prefix_slots: Some(2), host_cache_bytes: Some(0),
                onboard: Some(crate::placement::Onboard::Layers(5)), ..Default::default() };
            let mut report = plan::plan(target.path(), &PlanOptions {
                placement: ExpertPlacement::Sparks { ranks: 6 }, layout: Some(options.clone()),
                ..Default::default()
            })?;
            let (base, context, concurrency) = selected_host_geometry(&mut report, &options, &checkpoint)?;
            assert_eq!(context, crate::families::kimi_k3::DEFAULT_CONTEXT as u64);
            assert_eq!(concurrency, 2);
            assert_eq!(base.page_bytes, 24 * 64 * format.record_bytes() as u64);
            options.kimi_drafter_snapshot = Some(draft.path().to_path_buf());
            let (with_draft, _, _) = selected_host_geometry(&mut report, &options, &checkpoint)?;
            assert_eq!(with_draft.mark_bytes - base.mark_bytes, 4 * 4160 * format.record_bytes() as u64);
            assert_eq!(with_draft.page_bytes, base.page_bytes);
            for context in [262_144, 524_288, 1_048_576] {
                options.context_tokens = context;
                let (_, selected, _) = selected_host_geometry(&mut report, &options, &checkpoint)?;
                assert_eq!(selected, context);
            }
        }
        Ok(())
    }

    #[test]
    fn draft_prefix_reserves_marks_and_all_ring_floors() {
        let config = serde_json::json!({"num_hidden_layers": 5, "num_key_value_heads": 8, "head_dim": 128});
        assert_eq!(mimo_draft_prefix_bytes(&config, 42, 18, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap(), ((20 * MIB + 8) * 42, 144));
        assert_eq!(mimo_draft_prefix_bytes(&config, 42, 18, crate::families::qwen4::Qwen4KvCache::Fp8).unwrap(),
            ((5 * 1024 * 2 * 1056 + 8) * 42, 144));
        assert!(mimo_draft_prefix_bytes(&config, u64::MAX, 18, crate::families::qwen4::Qwen4KvCache::Bf16).is_err());
        assert!(mimo_draft_prefix_bytes(&config, 42, u64::MAX, crate::families::qwen4::Qwen4KvCache::Bf16).is_err());
        assert!(mimo_draft_prefix_bytes(&serde_json::json!({}), 42, 18, crate::families::qwen4::Qwen4KvCache::Bf16).is_err());
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
    pub fn resolve(report: &mut PlanReport, model: &dyn super::FamilyModel, options: &mut LayoutOptions,
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
        report.prefill_span_rows = Some(prefill_rows);
        // Legacy one-wave adapters use their resolved step as the unit.
        // Continuous adapters must supply their actual unit before host admission.
        if options.prefill_waves == 1 && options.prefill_unit_rows.is_none() {
            options.prefill_unit_rows = Some(prefill_rows);
        }
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
    let arch = crate::placement::ArchContext::coordinator(total_bytes, None);
    let families = programs.iter().map(String::as_str).collect::<Vec<_>>();
    let set = manifest.map(|m| crate::placement::ProgramSet::from_manifest(m, &families));
    let allowance = crate::placement::inventory::loaded_code_allowance(&programs[0], package,
        split, rank as u8, &arch, set.as_ref());
    Item::new(Category::Runtime, "context+modules", "", arch.context_bytes + allowance.bytes, allowance.basis)
}

/// Program families (manifest name prefixes) a coordinator GPU loads: `base`
/// and, under a split, its `base2` peer programs.
pub fn program_families(base: &str, split: bool) -> Vec<String> {
    std::iter::once(base.to_string()).chain(split.then(|| format!("{base}2"))).collect()
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
    let p = &report.mapped_placement;
    if p.mapped + p.preloaded > 0 {
        notes.push(format!("{}; {} host-resident scale bytes charged against {} available bytes (cap {} bytes)",
            p.label(), p.preloaded_bytes, p.host_budget_bytes, p.cap_bytes));
    }
    let mapped: u64 = report.components.iter().filter(|c| c.component == Component::MappedTable).map(|c| c.bytes).sum::<u64>()
        .saturating_sub(p.preloaded_bytes);
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
        let family = report.family.as_deref().unwrap_or_default();
        // The native EXLR8 Spark backend serves one execution slot.
        let slots = if crate::plan::exlr8::recognizes(&checkpoint.snapshot).unwrap_or(false) { 1 }
            else { spark_execution_slots(family, working.prefill_lanes, options.decode_lanes) };
        charge_spark_slots(device, working, family, slots, options.spark_capacity_rows);
    }
}

/// The local expert package kind a coordinator GPU loads (`LoadedCode::experts`): `exl3`, `fp8`
/// (fp8moe/NVFP4 packages) or `none` (Spark experts).
pub(crate) fn local_package(report: &PlanReport) -> &'static str {
    let package = report.experts.as_ref().map(|e| e.package.as_str()).unwrap_or("");
    if report.placement != ExpertPlacement::Local { "none" }
    else if package.contains("exl3") { "exl3" } else { "fp8" }
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

/// Resolve the serving representation, not the inventory's target-only defaults.
fn selected_host_geometry(report: &mut PlanReport, options: &LayoutOptions, checkpoint: &super::Checkpoint)
    -> anyhow::Result<(cuteafd_core::prefix::HostCacheGeometry, u64, u64)> {
    use anyhow::{Context, ensure};
    use crate::serving_capacity::{CacheOptions, KvPlacement};
    let family = super::family::detect(checkpoint).context("missing host cache family")?;
    let model = family.open(checkpoint)?;
    let id = family.id();
    let policy = ShapePolicy {
        context: if matches!(id, "mimo_v2" | "deepseek_v41") { ContextPolicy::Checkpoint }
            else { ContextPolicy::ManifestBounded },
        prefill_rows: (id == "deepseek_v41").then_some(2048),
        all_native_stages: id == "deepseek_v4",
    };
    let admitted_rows = report.prefill_span_rows;
    let shape = Shape::resolve(report, model.as_ref(), &mut options.clone(), policy);
    report.prefill_span_rows = admitted_rows;
    let ranks = if id == "deepseek_v41" || options.head_split { options.rtx_bytes.len().clamp(1, 2) } else { 1 };
    let selected = CacheOptions { coordinator_ranks: ranks,
        attention_placement: Some(report.attention_placement),
        native_mtp_layers: if matches!(id, "deepseek_v4" | "qwen4") { shape.native_layers } else { 0 },
        mimo_kv: crate::families::mimo_v2::MimoKvFormats { full: crate::families::mimo_v2::MimoKvCache::Int8, sliding: options.mimo_swa_kv },
        dense_qwen_kv: options.dense_qwen_kv, qwen_kv: options.qwen_kv, mla_kv: options.mla_kv,
        step5_kv: options.step5_kv, prefill_rows: admitted_rows.unwrap_or(shape.prefill_rows),
        glmf_index: options.glmf_index, kda_state_bytes: options.glmf_kda_state.bytes(),
        glmf_decode_rows: options.glmf_decode_rows, ..Default::default() };
    // Qwen's dual-owner layout partitions whole layers. Its host snapshot
    // concatenates those owners, so its aggregate is the single-owner geometry.
    let cache = if id == "glm5_flash" {
        use crate::placement::AttentionPlacement;
        let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&checkpoint.config)?;
        let attention = report.attention_placement;
        let index = match attention {
            AttentionPlacement::Context => crate::serving_capacity::GlmfIndexCache::Compact,
            AttentionPlacement::Heads if ranks == 2 => crate::serving_capacity::GlmfIndexCache::Keys,
            _ => options.glmf_index,
        };
        crate::serving_capacity::glm_flash_placed_cache_geometry_rows(&cfg, cfg.layers, ranks,
            index, options.glmf_kda_state.bytes(), options.glmf_decode_rows, attention, 0)?
    } else {
        model.cache_geometry(if id == "qwen4" { CacheOptions { coordinator_ranks: 1, ..selected } }
            else { selected })?.context("host cache layout unavailable")?
    };
    ensure!(!cache.ranks.is_empty(), "host cache has no owners");
    let pages = if cache.placement == KvPlacement::Replicated { &cache.ranks[..1] } else { &cache.ranks[..] };
    let page_bytes = pages.iter().try_fold(0u64, |n, r| n.checked_add(r.persistent_unit_bytes))
        .context("host cache pages overflow")?;
    // V4 stores the lead's complete mark once, then publishes its backbone prefix
    // to the peer. Other families retain distinct per-owner positional state.
    let marks = if id == "deepseek_v4" { &cache.ranks[..1] } else { &cache.ranks[..] };
    let mut mark_bytes = marks.iter().try_fold(0u64, |n, r| n.checked_add(r.retained_mark_bytes))
        .context("host cache marks overflow")?;
    if id == "deepseek_v41" && shape.native_layers > 0 {
        ensure!(shape.native_layers == 3, "V4.1 dSpark retains all three stages");
        mark_bytes = mark_bytes.checked_add(3 * 128 * 528).context("dSpark mark overflow")?;
    }
    let mut context = shape.context_tokens;
    if id == "kimi_k3" {
        if options.context_tokens == 0 {
            context = crate::families::kimi_k3::default_context(
                &crate::plan::families::kimi::core_inventory(checkpoint)?) as u64;
        }
        if let Some(snapshot) = &options.kimi_drafter_snapshot {
            let draft = crate::formats::dflash::MlaDflashConfig::read(snapshot)?;
            let windows = draft.paged_cache_layout(1, usize::try_from(context)?,
                usize::try_from(context)?, options.mla_kv)?.window_bytes;
            mark_bytes = mark_bytes.checked_add(windows).context("Kimi draft mark overflow")?;
        }
    }
    if id == "mimo_v2" && options.mimo_prefix_draft {
        let dir = checkpoint.snapshot.join("dflash");
        let draft = if dir.join("dflash_draft_model.safetensors").is_file() {
            let cfg = crate::families::mimo_v2::draft_config::DflashConfig::read(&dir)?;
            crate::families::mimo_v2::draft_representation::mimo_draft_mark_bytes(cfg.layers as u64, cfg.kv_heads as u64, cfg.head_dim as u64, options.draft_kv_cache)?
        } else if dir.join("config.json").is_file() {
            let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
            mimo_draft_prefix_bytes(&config, 1, options.mimo_rings.max(shape.concurrency), options.draft_kv_cache)
                .map_err(anyhow::Error::msg)?.0
        } else { 0 };
        mark_bytes = mark_bytes.checked_add(draft).context("MiMo draft mark overflow")?;
    }
    if id == "deepseek_v4" && options.context_tokens == 0 {
        if let Some(memory) = &report.memory_layout {
            if memory.pool_tokens < context { context = memory.pool_tokens.saturating_sub(256); }
        }
    }
    Ok((cuteafd_core::prefix::HostCacheGeometry { unit_rows: cache.logical_unit_rows,
        page_bytes, mark_bytes, draft_bytes: 0 }, context, shape.concurrency))
}

/// Offline host owners are future allocations, unlike the runtime's live pinned
/// ledger. The OS sample excludes external consumers, not this unstarted server.
pub(crate) fn charge_host(report: &mut PlanReport, options: &LayoutOptions, checkpoint: &super::Checkpoint, native: bool) -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress, SparkRings};
    use cuteafd_core::memory_layout::{HostLayout, ItemLifetime, host_logits_bytes};
    let Some(spec) = report.spec.as_ref() else { return Ok(()) };
    let family = report.family.as_deref().unwrap_or("");
    let (total, available) = match options.host_memory {
        Some(memory) => memory,
        None => {
            #[cfg(test)]
            panic!("planner tests must inject LayoutOptions.host_memory");
            #[cfg(not(test))]
            crate::formats::mapped_preload::host_memory()?
        }
    };
    // Native plans keep only coordinator devices in their memory layout.
    let ranks = report.spark_ranks;
    let rows = report.prefill_span_rows.unwrap_or(options.prefill_rows.max(4096));
    // Scoring retains the span; wire/router scratch serves one explicit unit.
    // The live return slots alone multiply by the number of waves.
    ensure!(ranks == 0 || options.prefill_unit_rows.is_some(),
        "{family} Spark host admission requires family-resolved prefill_unit_rows (span {rows}, waves {})",
        options.prefill_waves);
    let unit_rows = options.prefill_unit_rows.unwrap_or(rows);
    let intake_rows = if options.prefill_waves == 1 {
        ensure!(ranks == 0 || (1..=4096).contains(&unit_rows),
            "{family} Spark host admission needs prefill_unit_rows in 1..=4096, got {unit_rows}");
        unit_rows
    } else {
        crate::serving_capacity::PrefillShape { unit_rows, waves: options.prefill_waves }.validate()?.slot_rows()
    };
    let lanes = match report.prefill_transports {
        Some(transports) => transports,
        None => match family {
            "glm5" => if options.prefill_lanes > 0 { options.prefill_lanes } else {
                glm_prefill_lanes(None) as u64 },
            "glm5_flash" => if options.prefill_lanes > 0 { options.prefill_lanes }
                else { crate::serving_capacity::GLMF_DEFAULT_PREFILL_LANES },
            "mimo_v2" => {
                let cfg = crate::families::mimo_v2::MimoV2Config::from_hf(&checkpoint.config)?;
                let configured = if options.prefill_lanes > 0 { Some(options.prefill_lanes.to_string()) }
                    else { std::env::var("CUTEAFD_MIMO_PREFILL_LANES").ok() };
                crate::families::mimo_v2::admission::resolve_transport_lanes(ranks > 0,
                    cfg.program_family()?, configured.as_deref())? as u64
            }
            _ => 1,
        },
    };
    report.prefill_transports = Some(lanes);
    let hidden = spec.hidden as u64;
    let topk = spec.moe.as_ref().map_or(0, |m| m.top_k as u64);
    let mut items = Vec::new();
    let mut push = |category, group: &str, format: &str, bytes, basis| {
        items.push(Item::new(category, group, format, bytes, basis)
            .with_lifetime(ItemLifetime::PostReadyReserve));
    };
    if ranks > 0 {
        let geometry = ExpertGeometry { hidden: u32::try_from(hidden)?,
            experts: u32::try_from(spec.moe.as_ref().context("Spark layout without MoE geometry")?.experts)?,
            topk: u32::try_from(topk)?, intermediate: u32::try_from(spec.moe.as_ref().unwrap().intermediate)?,
            layers: u32::try_from(spec.layers.len())? };
        let endpoints = spark_endpoints(family, lanes);
        let bf16 = native || matches!(family, "glm5" | "qwen4");
        let ingress = if native { SparkIngress::Fp8K32 } else if bf16 { SparkIngress::Bf16 }
            else { SparkIngress::Fp8K32 };
        let endpoint = SparkEndpoint { ring_depth: None, capacity: u32::try_from(unit_rows)?, ingress };
        let (request, response) = endpoint.wire_bytes(geometry).context("host wire size overflow")?;
        let rings = SparkRings { max_frame_bytes: (64 << 20).max(request).max(response),
            depth: 8, slot_bytes: 8 << 20, alignment: 4096 };
        let bytes = (endpoint.ring_bytes(geometry, rings).context("host ring geometry overflow")? as u64)
            .checked_mul(endpoints as u64).and_then(|n| n.checked_mul(ranks as u64))
            .context("coordinator host rings overflow")?;
        push(Category::Transport, "transport/rdma-rings", "registered", bytes, Basis::Formula);
        let egress_request = geometry.compact_wire_bytes(u32::try_from(unit_rows)?,
            !native && checkpoint.spark_input_is_bf16()).context("host egress geometry overflow")?.0;
        let egress = (egress_request as u64).checked_mul(options.prefill_waves)
            .and_then(|n| n.checked_mul(endpoints as u64)).context("coordinator registered egress overflow")?;
        push(Category::Transport, "transport/egress", "registered request slots", egress, Basis::Formula);
        // Host-mode intake is the upper bound; auto/GPU landing can use less.
        let intake = intake_rows.checked_mul(hidden).and_then(|n| n.checked_mul(if native { 4 } else { 2 }))
            .and_then(|n| n.checked_mul(ranks as u64)).and_then(|n| n.checked_mul(endpoints as u64))
            .context("coordinator intake overflow")?;
        push(Category::Transport, "spark-intake", "host-mode upper bound", intake, Basis::Estimated);
    }
    let work_rows = unit_rows.checked_mul(lanes).and_then(|n| n.checked_add(64)).context("host workspace rows overflow")?;
    let router = work_rows.checked_mul(hidden.checked_mul(2).and_then(|n| n.checked_add(topk * 8))
        .context("host routing stride overflow")?).context("host routing workspace overflow")?;
    push(Category::Workspace, "workspace/router", "BF16 upper bound", router, Basis::Estimated);
    let mut staging = work_rows.checked_mul(hidden).and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(2 << 20)).context("host staging overflow")?;
    if family == "qwen4" {
        // A local owner has workspace tables even without a Spark transport.
        let lanes = lanes.max(1);
        let work_rows = unit_rows.checked_mul(lanes).and_then(|n| n.checked_add(64))
            .context("Qwen host workspace rows overflow")?;
        let cfg = crate::families::qwen4::Qwen4Config::from_hf(&checkpoint.config)?;
        let pool = report.memory_layout.as_ref().map_or(0, |m| m.pool_tokens);
        // Decode owns a table per row; prefill owns one. Charge the full
        // record and compressed-pool tables, even on partitioned owners.
        let tables = pool.div_ceil(64).checked_add(pool.div_ceil(64 * 4))
            .and_then(|n| n.checked_mul(64 + lanes)).and_then(|n| n.checked_mul(4))
            .context("Qwen host page tables overflow")?;
        let stride = (cfg.ple_rows() as u64).checked_mul(8)
            .and_then(|n| n.checked_add(64))
            .and_then(|n| hidden.checked_mul(10).and_then(|h| n.checked_add(h)))
            .context("Qwen host row metadata overflow")?;
        staging = work_rows.checked_mul(stride).and_then(|n| n.checked_add(tables))
            .and_then(|n| n.checked_add(256 * (lanes + 1)))
            .and_then(|n| n.checked_mul(options.rtx_bytes.len() as u64))
            .context("Qwen host staging overflow")?;
    }
    push(Category::Staging, "staging/tables-and-sync", "row metadata", staging, Basis::Estimated);
    if family == "qwen4" && ranks > 0 && options.prefill_waves > 1 {
        use crate::families::qwen4::{Qwen4Config, StepTableBytes};
        use crate::serving_capacity::qwen_graphs::{qwen_pool_pages, QWEN_UNIT_PAGES};
        let cfg = Qwen4Config::from_hf(&checkpoint.config)?;
        let pool = usize::try_from(report.memory_layout.as_ref().context("Qwen ordered tables need a KV layout")?.pool_tokens)?;
        let pages = qwen_pool_pages(pool).context("Qwen host page tables overflow")?.max(QWEN_UNIT_PAGES);
        let tables = StepTableBytes::new(usize::try_from(unit_rows)?, false, pages, pages / QWEN_UNIT_PAGES, cfg.ple_rows())?;
        // One unit workspace owns these destinations, shared by both waves.
        // Mapped PLE uploads gathered rows, not the separate PLE-ID table.
        let ple_ids = !cfg.ple_layers.first().is_some_and(|&layer| layer < spec.layers.len());
        push(Category::Staging, "staging/ordered-tables", "full destination extents",
            tables.ordered_bytes(ple_ids)?, Basis::Formula);
    }
    let score_lanes = if matches!(family, "glm5" | "glm5_flash") { lanes } else { 1 };
    let score_rows = if options.full_prefill_logits { rows.checked_mul(score_lanes).context("host scoring rows overflow")? }
        else if family == "deepseek_v41" { rows.max(256) } else { 64 };
    let logits = host_logits_bytes(score_rows, rows, spec.vocab as u64).context("host logits overflow")?;
    push(Category::Workspace, "scoring/full-logits", "FP32", logits, Basis::Formula);
    if family == "qwen4" {
        // Decode and prefill arenas coexist on each owner. A peer may need
        // fewer output rows; charge the all-owner upper bound.
        let pinned_rows = 64u64.checked_add(if options.full_prefill_logits { rows } else { 1 })
            .and_then(|n| n.checked_mul(options.rtx_bytes.len() as u64))
            .context("Qwen pinned logits rows overflow")?;
        let pinned = host_logits_bytes(pinned_rows, pinned_rows, spec.vocab as u64)
            .context("Qwen pinned logits overflow")?;
        push(Category::Workspace, "workspace/logits-host", "FP32 all-owner upper bound", pinned, Basis::Estimated);
    }
    if options.host_embedding || report.memory_layout.as_ref().is_some_and(|m|
        m.notes.iter().any(|n| n.starts_with("embedding placement: host"))) {
        let bytes = report.components.iter().filter(|c| c.component == Component::Embedding)
            .try_fold(0u64, |n, c| n.checked_add(c.bytes)).context("host embedding overflow")?;
        push(Category::Embedding, "embedding", "checkpoint", bytes, Basis::Exact);
    }
    push(Category::Tables, "mapped-scales/preloaded", "checkpoint", report.mapped_placement.preloaded_bytes, Basis::Exact);
    let media_cache_loaded = (matches!(family, "mimo_v2" | "qwen4" | "glm5_flash")
        && report.vision != super::MediaMode::Off)
        || (family == "mimo_v2" && report.audio != super::MediaMode::Off);
    if media_cache_loaded {
        let media = options.media_cache_bytes.unwrap_or((8u64 << 30).min(total / 20));
        push(Category::Embedding, "media/embedding-cache", "host quota", media, Basis::Formula);
    }
    let tier_disabled = options.rtx_bytes.len() > 1 && matches!(family, "glm5" | "deepseek_v4")
        && report.attention_placement == crate::placement::AttentionPlacement::Heads;
    let family = family.to_owned();
    let cache = if options.mimo_prefix_entries == 0 || tier_disabled || options.host_cache_bytes == Some(0) { 0 } else {
        let resolved = selected_host_geometry(report, options, checkpoint).and_then(|(geometry, context, concurrency)| {
            let chunk = geometry.chunk_bytes();
            let bytes = geometry.resolve(options.host_cache_bytes, context, concurrency, total, chunk)
                .with_context(|| format!("invalid host snapshot geometry {geometry:?}, context {context}, concurrency {concurrency}"))?;
            ensure!(bytes > 0, "host prefix quota resolves to zero; use a positive quota or explicitly disable host retention");
            let max = geometry.conversations(bytes, context, chunk.min(bytes));
            let short = geometry.conversations(bytes, 50_000, chunk.min(bytes));
            if let Some(memory) = &mut report.memory_layout {
                memory.notes.push(format!("host prefix: {} B per mark; {} conversations at max context, {} at 50K (two marks each)",
                    geometry.mark_bytes, max, short));
                if short < concurrency + 4 {
                    memory.notes.push(format!("WARNING: host prefix 50K capacity {short} is below concurrency+4 ({})", concurrency + 4));
                }
            }
            Ok::<_, anyhow::Error>(bytes)
        });
        match resolved {
            Ok(bytes) => bytes,
            Err(error) => {
                report.fits = false;
                report.placement_supported = false;
                report.hints.push(super::Hint {
                    what: format!("{family} host prefix cache unsupported (requested {}): {error:#}",
                        options.host_cache_bytes.map_or_else(|| "auto bytes".into(), |n| format!("{n} bytes"))),
                    how: "Add exact retained page and mark geometry and a host restore implementation for this family/shape, or disable the host tier with --host-cache-bytes 0.".into(),
                });
                options.host_cache_bytes.unwrap_or(0)
            }
        }
    };
    let host = HostLayout::new(total, available, (8 << 30).max(total / 10), cache, items)
        .context("coordinator host total overflow")?;
    ensure!(host.total_bytes > 0, "coordinator host memory is zero");
    report.host_budget_bytes = cache;
    if host.shortfall_bytes > 0 {
        report.fits = false;
        report.hints.push(super::Hint { what: format!("{family} coordinator host memory shortfall {} bytes: prefix cache {}, other consumers {}, reserve {}, available {}",
            host.shortfall_bytes, cache, host.deferred_bytes, host.reserve_bytes, available),
            how: "Reduce --host-cache-bytes or the scoring/transport shape, or free host memory.".into() });
    }
    report.host_ledger = Some(host);
    Ok(())
}
