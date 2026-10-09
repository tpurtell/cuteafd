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

mod deepseek_v4;
mod v41;

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
    /// Decode graph budget (GLM 5.3 Flash `--graph-budget-mib`), in place of the graph allowance
    /// where the engine admits from measured memory (one GPU, Spark experts, an automatic pool),
    /// else in its place only when larger.
    pub graph_budget_bytes: Option<u64>,
    /// GLM 5.3 Flash's prefix marks in pool units (`--prefix-marks pool`): no mark arena, and
    /// `GLMF_POOL_MARK_RESERVED_UNITS` units allocated beside the pool and never handed out.
    pub glmf_pool_marks: bool,
    /// GLM 5.3 Flash's KDA replay records in the prefill lanes' scratch (`--replay-records shared`):
    /// out of the state, and the scratch holds at least them. The engine takes them only where it
    /// sizes the pool from measured memory (one GPU, Spark experts, an automatic pool).
    pub glmf_shared_replay: bool,
    /// Spark wave capacity in rows (`expertd --capacity`).
    pub spark_capacity_rows: u64,
    /// Explicit pool tokens; `None` or `Some(0)` sizes the pool from what is left.
    pub pool_tokens: Option<u64>,
    /// Upper bound for an automatically sized pool.
    pub target_pool_tokens: u64,
    /// External drafter resident bytes on the lead GPU (DFlash), if any.
    pub drafter_bytes: u64,
    /// Keep this much of every GPU free for runtime growth.
    pub headroom_bytes: u64,
    /// Concurrent sequences (state slots = concurrency + 2).
    pub concurrency: u64,
    /// Exact engine slots when they differ from concurrency + 2.
    pub state_slots: Option<u64>,
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
    /// MiMo drafter batch limit (serving raises it to concurrency).
    pub draft_sequences: u64,
    /// Explicit MiMo DFlash context arena; does not widen its draft batch.
    pub draft_context_slots: Option<u64>,
    /// Compiled context extent (RoPE and index workspaces).
    pub context_tokens: u64,
    /// Resident native drafter stages (zero disables optional native MTP).
    pub native_mtp_layers: usize,
    /// Whole routed backbone layers resident on the coordinator.
    pub local_expert_layers: Option<usize>,
    /// Matching image's PROGRAMS.json for exact V4 workspace geometry.
    pub workspace_manifest: Option<std::path::PathBuf>,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            rtx_bytes: vec![95 * GIB + 512 * MIB],
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
            graph_budget_bytes: None,
            glmf_pool_marks: false,
            glmf_shared_replay: false,
            spark_capacity_rows: 4096,
            pool_tokens: None,
            target_pool_tokens: cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS,
            drafter_bytes: 0,
            headroom_bytes: 2 * GIB,
            concurrency: 0,
            state_slots: None,
            prefix_slots: None,
            mimo_prefix_entries: 20,
            mimo_prefix_mark_bytes: 2 * GIB,
            mimo_prefix_draft: false,
            mimo_rings: 16,
            draft_sequences: 4,
            draft_context_slots: None,
            context_tokens: 0,
            native_mtp_layers: 3,
            local_expert_layers: None,
            workspace_manifest: None,
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
    /// Host memory charged by CUDA on GB10 after checkpoint caches are dropped.
    pub spark_host_bytes: u64,
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
        // GLM 5.3 EXL3 K4 + DFlash2 (BF16, 4.58 GiB checkpoint + 1.3 GiB context/buffers).
        "glm5" => FamilyCosts {
            // Untracked 0.89 / 0.82 GiB at ready, 2.62 / 2.14 after one decode+prefill
            // bench and still rising (graphs per layer x exact rows x table width).
            runtime_bytes: [gib(74), gib(89), gib(82)],
            graph_bytes: [gib(300), gib(300), gib(300)],
            workspace_bytes: [gib(651), gib(559), gib(422)],
            exchange_bytes: gib(56),
            drafter_bytes: gib(588),
            mtp_resident: false,
            attention_replicated: 0.10,
            spark_workspace_bytes: gib(190),
            spark_ring_bytes: gib(231),
            ..generic
        },
        // MiMo V2.6 Pro + embedded DFlash (qualified FP8 bundle); MTP unused.
        "mimo_v2" => FamilyCosts {
            runtime_bytes: [gib(100), gib(111), gib(100)],
            graph_bytes: [gib(50), gib(50), gib(50)],
            workspace_bytes: [gib(290), gib(249), gib(120)],
            exchange_bytes: gib(38),
            drafter_bytes: gib(321),
            mark_slots: 42,
            mtp_resident: false,
            spark_workspace_bytes: gib(51),
            spark_ring_bytes: gib(117),
            ..generic
        },
        // GLM 5.3 Flash EXL3 + DFlash2; one GPU (no head split). Without a program manifest,
        // one GPU's step workspaces: decode and two lanes of 4,096 rows over shared temporaries
        // (`serving_capacity::glmf_step_workspaces` of the 5090 export, 2.68 GiB).
        "glm5_flash" => FamilyCosts {
            runtime_bytes: [gib(78), gib(78), gib(78)],
            graph_bytes: [gib(150), gib(150), gib(150)],
            workspace_bytes: [gib(268), gib(472), gib(472)],
            drafter_bytes: gib(324),
            mtp_resident: false,
            spark_workspace_bytes: gib(56),
            spark_ring_bytes: gib(78),
            ..generic
        },
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
            runtime_bytes: [gib(91), gib(91), 0],
            graph_bytes: [gib(50), gib(50), 0],
            workspace_bytes: [gib(120), gib(120), 0],
            mark_slots: 18,
            spark_workspace_bytes: gib(56),
            spark_ring_bytes: gib(78),
            ..generic
        },
        _ => generic,
    }
}

fn share_of(family: &str, component: Component) -> Share {
    match component {
        Component::Norm | Component::HyperConnection => Share::Replicated,
        Component::Indexer if family == "glm5" => Share::Replicated,
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

fn resolved_context(family: &str, requested: u64, checkpoint: u64, manifest: Option<&serde_json::Value>) -> u64 {
    if requested > 0 { return requested; }
    if matches!(family, "mimo_v2" | "deepseek_v41") { return checkpoint; }
    manifest.and_then(|m| m["capacities"]["max_context"].as_u64()).filter(|&n| n > 0)
        .map_or(checkpoint, |limit| if checkpoint > 0 { limit.min(checkpoint) } else { limit })
}

#[cfg(test)]
mod draft_prefix_tests {
    use super::*;

    #[test]
    fn full_context_defaults_preserve_dynamic_families_and_bound_indexed_families() {
        let manifest = serde_json::json!({"capacities": {"max_context": 131072}});
        for family in ["mimo_v2", "deepseek_v41", "glm5", "glm5_flash", "qwen4", "deepseek_v4"] {
            let expected = if matches!(family, "mimo_v2" | "deepseek_v41") { 1048576 } else { 131072 };
            assert_eq!(resolved_context(family, 0, 1048576, Some(&manifest)), expected);
            assert_eq!(resolved_context(family, 65536, 1048576, Some(&manifest)), 65536);
            assert_eq!(resolved_context(family, 0, 1048576, None), 1048576);
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

/// Lays out `report` (a `plan` of the checkpoint) on the inventory.
pub fn layout(report: &mut PlanReport, model: &dyn super::FamilyModel, checkpoint: &super::Checkpoint,
    options: &LayoutOptions) -> MemoryLayout {
    let family_name = report.family.clone().unwrap_or_else(|| "unknown".into());
    let family = family_name.as_str();
    let discovered_native_layers = match model.spec().speculator.as_ref() {
        Some(super::spec::SpeculatorSpec::Dspark { stages, .. }) => *stages,
        Some(super::spec::SpeculatorSpec::NativeMtp { layers }) => *layers,
        None => 0,
    };
    let native_layers = if family == "deepseek_v4" && options.native_mtp_layers > 0 {
        discovered_native_layers
    } else { options.native_mtp_layers.min(discovered_native_layers) };
    // V4 always loads all checkpoint stages and their caches; --dspark only
    // changes expert residency and whether the scheduler drafts with them.
    let cache_native_layers = if family == "deepseek_v4" { discovered_native_layers } else { native_layers };
    let workspace_manifest = options.workspace_manifest.as_deref().or_else(|| {
        let path = std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path)
    }).and_then(|path| std::fs::read(path).ok()).and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let small_card = matches!(family, "deepseek_v41" | "mimo_v2")
        && options.rtx_bytes.first().is_some_and(|&bytes| bytes <= 32 * GIB);
    let prefill_rows = if options.prefill_rows > 0 { options.prefill_rows }
        else if family == "deepseek_v41" { if small_card { 1024 } else { 2048 } }
        else { workspace_manifest.as_ref().and_then(|m| m["capacities"]["prefill_rows"].as_u64()).unwrap_or(4096) };
    let decode_rows = workspace_manifest.as_ref().and_then(|m| m["capacities"]["decode_rows"].as_u64()).unwrap_or(64);
    let concurrency = if options.concurrency > 0 { options.concurrency }
        else if family == "deepseek_v41" || (family == "mimo_v2" && small_card) { 16 }
        else { 8 };
    let checkpoint_context = report.cache_requirements.as_ref().and_then(|r| r.checkpoint_max_context_tokens).unwrap_or(0);
    let context_tokens = resolved_context(family, options.context_tokens, checkpoint_context, workspace_manifest.as_ref());
    let target_pool_tokens = options.target_pool_tokens.max(context_tokens);
    let conversions = load_conversions(family, checkpoint);
    let costs = family_costs(family);
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    // Qwen currently executes entirely on the first coordinator GPU.
    let split = gpus == 2 && options.head_split && family != "qwen4";
    let active_gpus = if family == "qwen4" { 1 } else if split { 2 } else { 1 };
    let automatic = options.pool_tokens.unwrap_or(0) == 0;
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: if matches!(family, "deepseek_v41" | "mimo_v2") && bytes <= 32 * GIB {
                ((bytes as u128 * 97 / 100) as u64).min(bytes.saturating_sub(
                    options.headroom_bytes.max(cuteafd_core::serving_capacity::SMALL_CARD_HEADROOM_BYTES)))
            } else { (if family == "deepseek_v41" && automatic { (bytes as u128 * 97 / 100) as u64 } else { bytes })
                .saturating_sub(if matches!(family, "deepseek_v4" | "deepseek_v41" | "qwen4") { options.headroom_bytes.max(3 * GIB) } else { options.headroom_bytes }) },
            items: Vec::new(), kv_tokens: 0 })
        .collect();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {context_tokens} tokens (checkpoint {checkpoint_context})")];
    let reference_gpu = gpus == 1 && (94 * GIB..=96 * GIB).contains(&options.rtx_bytes[0])
        && native_layers > 0 && options.local_expert_layers.is_none();
    let package = report.experts.as_ref().map(|e| e.package.as_str());
    let reference = reference_gpu && match family {
        "deepseek_v4" => matches!(report.placement, ExpertPlacement::Sparks { ranks: 2 })
            && package == Some("dsv4f:mxfp4 (expertd-native)")
            && model.spec().hidden == 4096 && model.spec().layers.len() == 43
            && workspace_manifest.is_some() && prefill_rows == 4096 && concurrency == 8
            && matches!(options.prefix_slots, None | Some(42)) && context_tokens == 32768,
        "deepseek_v41" => matches!(report.placement, ExpertPlacement::Sparks { ranks: 4 })
            && model.spec().hidden == 5120 && model.spec().layers.len() == 40
            && package == Some("v41:mxfp4 (expertd-native)") && prefill_rows == 2048
            && concurrency == 16 && matches!(options.prefix_slots, None | Some(42))
            && matches!(context_tokens, 0 | 1_048_576),
        "qwen4" => report.placement == ExpertPlacement::Local && package == Some("qwen4:exl3-k45")
            && model.spec().hidden == 2560 && model.spec().layers.len() == 48
            && prefill_rows == 4096 && concurrency == 8 && matches!(options.prefix_slots, None | Some(18))
            && matches!(context_tokens, 0 | 32768),
        _ => false,
    };
    let qualified = reference || matches!(family, "mimo_v2" | "glm5" | "glm5_flash");
    let allowance_basis = if qualified { Basis::Calibrated } else { Basis::Estimated };
    if !qualified {
        notes.push(format!("{family}: workspace/runtime/Spark allowances are unqualified; validate against the allocation ledger before using this layout for admission"));
    }
    if reference {
        notes.push("Runtime allowances calibrated at the natural-minimum reference layout after an 8K prefill and C4 decode; other layouts remain estimates".into());
    }
    if family == "qwen4" && gpus == 2 {
        notes.push("Qwen serves on rtx0; rtx1 is idle and contributes no KV capacity".into());
    }

    let mapped: u64 = report.components.iter().filter(|c| c.component == Component::MappedTable).map(|c| c.bytes).sum();
    if mapped > 0 { notes.push(format!("host-mapped table backing: {:.2} GiB (device budget includes staging only)", mapped as f64 / GIB as f64)); }

    // Coordinator weights: a family's exact resident layout where it has one,
    // else checkpoint bytes per component under the family's conversions.
    let v41_weights = if family == "deepseek_v41" { v41::resident_weights(checkpoint, active_gpus, native_layers > 0) } else { None };
    let qwen_exl3 = if family == "qwen4" {
        qwen_exl3_arenas(checkpoint, native_layers > 0)
    } else { None };
    if let Some(ranks) = &v41_weights {
        for (device, items) in devices.iter_mut().zip(ranks) {
            device.items.extend(items.iter().filter(|item|
                report.vision != super::MediaMode::Off || item.group != "vision").cloned());
        }
    }
    let exact = resident_layout(family, checkpoint, if split { 2 } else { 1 }, native_layers > 0);
    if let Some(ranks) = &exact {
        for (device, rank) in devices.iter_mut().zip(ranks) {
            for (group, format, bytes) in rank {
                let category = match group.as_str() {
                    "embedding" => Category::Embedding,
                    "speculator" | "speculator_expert" => Category::Drafter,
                    "table_projection" => Category::Tables,
                    _ => Category::Weights,
                };
                device.items.push(Item::new(category, group.clone(), format.clone(), *bytes, Basis::Exact));
            }
        }
    }
    let covered = |c: Component| (v41_weights.is_some() && !matches!(c, Component::RoutedExpert | Component::MappedTable))
        || (qwen_exl3.is_some() && matches!(c, Component::RoutedExpert | Component::SpeculatorExpert))
        || (family.starts_with("deepseek_v4") && exact.is_some() && c == Component::Speculator)
        || (family == "qwen4" && exact.is_some() && matches!(c, Component::Speculator | Component::TableProjection))
        || (exact.is_some() && !matches!(c, Component::Speculator | Component::SpeculatorExpert
        | Component::Vision | Component::TableProjection | Component::RoutedExpert)
            && !(family.starts_with("deepseek_v4") && matches!(c, Component::Speculator | Component::SpeculatorExpert)))
        || (!costs.mtp_resident && matches!(c, Component::Speculator | Component::SpeculatorExpert));
    for component in report.components.iter()
        .filter(|c| (c.owner == Owner::Rtx
            || (c.owner == Owner::SparkSliced && report.placement == ExpertPlacement::Local))
            && c.status != Status::Unused && c.status != Status::Disabled
            && !matches!(c.component, Component::Vision | Component::Audio) && !covered(c.component)) {
        let (resident, format) = match conversions.iter().find(|c| c.component == component.component) {
            Some(c) => (component.bytes.saturating_sub(c.saved_bytes), c.format.to_string()),
            None => ((component.bytes as f64 * costs.resident_factor) as u64,
                component.formats.keys().cloned().collect::<Vec<_>>().join("+")),
        };
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
        let embedding = report.components.iter().find(|c| c.component == Component::Embedding);
        let names: Vec<_> = checkpoint.tensors.iter().filter(|t|
            t.meta.name == "embed.weight" || t.meta.name.ends_with("embed_tokens.weight")).collect();
        if let Some(tensor) = names.first().filter(|_| names.len() == 1) {
            match checkpoint.require_untied_embedding(&tensor.meta.name) {
                Ok(()) => {
                    let mut saved = 0;
                    for device in &mut devices {
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
        } else {
            notes.push("host embedding ineligible: no unique token embedding; GPU residency retained".into());
        }
    }
    if let Some((backbone, draft)) = qwen_exl3 {
        if report.placement == ExpertPlacement::Local {
            devices[0].items.push(Item::new(Category::Experts, "routed expert arenas", "exl3", backbone, Basis::Formula));
        }
        if draft > 0 {
            devices[0].items.push(Item::new(Category::Experts, "native MTP expert arena", "exl3", draft, Basis::Formula));
        }
    } else if family == "qwen4" && native_layers > 0 {
        let native_experts: u64 = report.components.iter().filter(|c| c.component == Component::SpeculatorExpert && c.status == Status::Unused)
            .map(|c| c.bytes).sum();
        if native_experts > 0 {
            devices[0].items.push(Item::new(Category::Experts, "native MTP experts", "native", native_experts, Basis::Exact));
        }
    }
    let exl3_workspace = expert_workspace(report, model, checkpoint, options.workspace_manifest.as_deref(), prefill_rows);
    if exl3_workspace.is_none() && report.experts.as_ref().is_some_and(|e| e.package.contains("exl3"))
        && (family == "deepseek_v4" || (family == "qwen4"
            && (report.placement == ExpertPlacement::Local || native_layers > 0))) {
        notes.push("Local EXL3 workspace allowance is estimated without matching rtx-tp1/m*/v41_exl3.json capacity manifests; images bundle them, or export the exl3 tree alongside PROGRAMS.json".into());
    }
    if family == "qwen4" && report.placement != ExpertPlacement::Local && native_layers > 0 {
        notes.push("Native Qwen MTP experts stay on rtx0; Spark ranks serve backbone experts only".into());
    }
    if family == "qwen4" && (report.placement == ExpertPlacement::Local || native_layers > 0)
        && report.experts.as_ref().is_some_and(|e| e.package.contains("exl3")) {
        // The EXL3 window retains its shared capacity arenas in addition to
        // checkpoint trellis bytes (1.15 GiB in the reference allocation ledger).
        devices[0].items.push(Item::new(Category::Experts, "local EXL3 workspace", "",
            exl3_workspace.unwrap_or(gib(115) * prefill_rows / 4096),
            if exl3_workspace.is_some() { Basis::Formula } else { allowance_basis }));
    }
    // The drafter lives on the lead GPU (taps and head are there under a head split).
    let drafter = if options.drafter_bytes > 0 { options.drafter_bytes } else { costs.drafter_bytes };
    if drafter > 0 {
        devices[0].items.push(Item::new(Category::Drafter, "drafter", "", drafter, allowance_basis));
    }

    let v4_workspace = if family == "deepseek_v4" {
        (|| {
            let manifest = workspace_manifest.as_ref()?;
            let cfg = crate::families::deepseek_v4::DeepseekV4Config::read(&checkpoint.snapshot, cache_native_layers).ok()?;
            let id = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
            let scratch = crate::serving_capacity::deepseek_v4_workspace_scratch(manifest, id, prefill_rows, decode_rows).ok()?;
            crate::serving_capacity::deepseek_v4_workspace_geometry(&cfg, prefill_rows, decode_rows,
                crate::serving_capacity::compiled_c128_width(manifest, id).ok()?.checked_mul(128)?,
                active_gpus, scratch).ok()
        })()
    } else { None };
    if family == "deepseek_v4" && v4_workspace.is_none() {
        notes.push("V4 workspace allowance is estimated without a matching PROGRAMS.json; use --workspace-manifest for Flash/Pro allocation geometry".into());
    }

    // GLM 5.3 Flash on one GPU: the step workspaces its engine allocates, from the program manifest.
    let glmf_lanes = if options.prefill_lanes > 0 { options.prefill_lanes }
        else { crate::serving_capacity::GLMF_DEFAULT_PREFILL_LANES };
    // `--decode-rows 128` runs the wide `_m128` programs on one GPU; the build must export them.
    let glmf_decode_rows = if family == "glm5_flash" { options.glmf_decode_rows }
        else { crate::serving_capacity::GLMF_DECODE_ROWS };
    let glmf_wide = glmf_decode_rows > crate::serving_capacity::GLMF_DECODE_ROWS;
    if glmf_wide && split {
        report.placement_supported = false;
        notes.push("GLM 5.3 Flash's 128-row decode programs run on one GPU: a head split takes --decode-rows 64".into());
    }
    if glmf_wide && workspace_manifest.as_ref().is_some_and(|manifest|
        crate::serving_capacity::glmf_manifest_scratch(manifest)("glmf_mhc_post_pre_m128").is_none()) {
        report.placement_supported = false;
        notes.push("GLM 5.3 Flash --decode-rows 128 needs the 128-row decode programs, which this program manifest \
            lacks (build with CUTEAFD_GLMF_WIDE_DECODE_ROWS=128)".into());
    }
    // A GLM 5.3 Flash graph budget, kept as the engine's KV admission keeps it: from measured free
    // memory (one GPU, Spark experts, an automatic pool) the budget itself, from the planner's costs
    // (a head split, local experts, a fixed pool) the budget or the graph allowance, whichever is larger.
    let glmf_measured = !split && automatic && matches!(report.placement, ExpertPlacement::Sparks { .. });
    // Shared replay records (measured admission only): the KDA records of the decode rows, in the
    // prefill scratch.
    let glmf_shared_records = if family == "glm5_flash" && options.glmf_shared_replay {
        if glmf_measured {
            crate::families::glm5_flash::GlmNextConfig::from_hf(&checkpoint.config).ok()
                .and_then(|cfg| crate::serving_capacity::glm_flash_kda_replay_bytes_rows(&cfg, cfg.layers, 1,
                    glmf_decode_rows).ok())
                .unwrap_or(0)
        } else {
            notes.push("GLM 5.3 Flash --replay-records shared needs one GPU, Spark experts and an automatic pool \
                (the engine refuses it otherwise); planned with records of their own".into());
            0
        }
    } else { 0 };
    let glmf_steps = (family == "glm5_flash" && !split).then(|| workspace_manifest.as_ref()
        .and_then(|manifest| glmf_step_workspace(manifest, checkpoint, &report.placement, glmf_lanes, prefill_rows,
            context_tokens, glmf_decode_rows, glmf_shared_records))).flatten();

    // Fixed runtime costs.
    let gpus_now = active_gpus;
    for (index, device) in devices.iter_mut().take(active_gpus).enumerate() {
        let role = if gpus_now == 1 { 0 } else if index == 0 { 1 } else { 2 };
        device.items.push(Item::new(Category::Runtime, "context+modules", "", costs.runtime_bytes[role],
            allowance_basis));
        let graph_allowance = if family == "deepseek_v41" && options.rtx_bytes[index] <= 32 * GIB {
            // Match the qualified fixed-bank envelope reserved by measured_pool_memory.
            2 * GIB
        } else { costs.graph_bytes[role] };
        match options.graph_budget_bytes.filter(|&budget| family == "glm5_flash"
            && (glmf_measured || budget > graph_allowance)) {
            Some(budget) => device.items.push(Item::new(Category::Runtime, "graph budget", "", budget, Basis::Formula)),
            None => device.items.push(Item::new(Category::Runtime, "graph allowance", "", graph_allowance,
                allowance_basis)),
        }
        let workspace = match glmf_steps {
            Some(steps) if index == 0 => steps,
            // The allowance covers the default lanes' rows in flight; more rows in flight take more.
            None if family == "glm5_flash" && role == 0 =>
                costs.workspace_bytes[role] * (glmf_lanes * prefill_rows.max(1)).max(8192) / 8192,
            _ => v4_workspace.as_ref().and_then(|ranks| ranks.get(index)).map_or_else(
                || costs.workspace_bytes[role] * prefill_rows.max(1) / if family == "deepseek_v41" { 2048 } else { 4096 },
                |rank| rank.fixed_device_bytes),
        };
        // V4 keeps one 4096-row intake plane per Spark and prefill lane; GLM 5.3 Flash one plane
        // per Spark and lane of its widest step's rows (a lane's, or 128 decode rows past a
        // narrower lane). Decode reuses lane zero; every plane belongs to the lead GPU.
        let intake = match (family, report.placement) {
            ("deepseek_v4", ExpertPlacement::Sparks { ranks }) if index == 0 =>
                2 * ranks as u64 * 4096 * model.spec().hidden as u64 * 2,
            ("glm5_flash", ExpertPlacement::Sparks { ranks }) if index == 0 && !split =>
                crate::serving_capacity::glmf_spark_intake_bytes(glmf_lanes, ranks as u64,
                    crate::serving_capacity::glmf_expert_rows(prefill_rows, glmf_decode_rows), model.spec().hidden as u64),
            _ => 0,
        };
        let expert_exchange = if family == "deepseek_v4" && split {
            crate::serving_capacity::deepseek_v4_expert_exchange_bytes(model.spec().hidden as u64,
                model.spec().moe.as_ref().unwrap().top_k as u64, prefill_rows, decode_rows, index).unwrap_or(0)
        } else { 0 };
        let workspace = workspace + intake;
        if family == "deepseek_v4" {
            // --reserve-gib 10 covers the future workspace and graph budget;
            // admission keeps the unused remainder, with a 3 GiB floor.
            let headroom = (10 * GIB).saturating_sub(workspace + expert_exchange + costs.graph_bytes[role])
                .max(options.headroom_bytes).max(3 * GIB);
            device.capacity_bytes = options.rtx_bytes[index].saturating_sub(headroom);
        }
        let workspace_basis = if v4_workspace.is_some() || (glmf_steps.is_some() && index == 0) { Basis::Formula }
            else { allowance_basis };
        device.items.push(Item::new(Category::Workspace, "steps", "", workspace, workspace_basis));
        if glmf_wide && !split && index == 0 {
            // The token selector and GPU sampler of 128-row steps, beyond the 64-row ones the runtime
            // allowance holds.
            let vocab = model.spec().vocab as u64;
            device.items.push(Item::new(Category::Workspace, "wide decode selector", "",
                crate::serving_capacity::glmf_selector_bytes(glmf_decode_rows, vocab)
                    - crate::serving_capacity::glmf_selector_bytes(crate::serving_capacity::GLMF_DECODE_ROWS, vocab),
                Basis::Formula));
            if glmf_steps.is_none() {
                notes.push("GLM 5.3 Flash --decode-rows 128: without a program manifest the steps allowance \
                    assumes 64-row decode workspaces (pass --workspace-manifest)".into());
            }
        }
        if options.full_prefill_logits && index == 0 {
            device.items.push(Item::new(Category::Workspace, "probe prefill logits", "",
                full_prefill_logits_bytes_with_lanes(family, prefill_rows, model.spec().vocab as u64,
                    effective_glm_prefill_lanes(matches!(report.placement, ExpertPlacement::Sparks { .. }),
                        std::env::var("CUTEAFD_GLM_PREFILL_LANES").ok().as_deref())), Basis::Formula));
        }
        if split {
            let exact_peer = if family == "deepseek_v4" {
                crate::serving_capacity::deepseek_v4_peer_exchange_bytes(model.spec().hidden as u64, prefill_rows, decode_rows).ok()
            } else { None };
            device.items.push(Item::new(Category::Transport, "peer exchange", "", exact_peer.unwrap_or(costs.exchange_bytes),
                if exact_peer.is_some() { Basis::Formula } else { allowance_basis }));
        }
    }

    // Resident V4.1 layers are taken out of Spark slices; V4 retains the complete
    // target on Sparks even when RTX also holds a layer. Experts are charged
    // before the KV pool. Explicit placement is reproducible on any inventory.
    let routed_bytes: u64 = report.components.iter().filter(|c| c.component == Component::RoutedExpert)
        .map(|c| c.bytes).sum();
    let layer_bytes = routed_bytes / model.spec().layers.len().max(1) as u64;
    let mut local_layers = if family.starts_with("deepseek_v4") {
        options.local_expert_layers.unwrap_or(0).min(model.spec().layers.len())
    } else { 0 };
    let draft_experts: u64 = if family == "deepseek_v4" && native_layers > 0 {
        report.components.iter().filter(|c| c.component == Component::SpeculatorExpert).map(|c| c.bytes).sum()
    } else { 0 };
    if draft_experts > 0 && family != "deepseek_v4" {
        devices[0].items.push(Item::new(Category::Experts, "dSpark stage experts", "native", draft_experts, Basis::Exact));
    }
    let mut local_bytes = layer_bytes * local_layers as u64;
    if family != "deepseek_v4" && local_bytes > 0 && !matches!(report.placement, ExpertPlacement::Local) {
        if family == "deepseek_v41" && gpus == 2 {
            for device in devices.iter_mut().take(2) {
                device.items.push(Item::new(Category::Experts, "resident routed layers", "native-tp2", local_bytes / 2, Basis::Formula));
            }
        } else {
            devices[0].items.push(Item::new(Category::Experts, "resident routed layers", "native", local_bytes, Basis::Exact));
        }
    }

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
        let exact = !package.starts_with("v41") && intermediate % 128 == 0 && intermediate / 128 >= ranks;
        let rank_bytes = |rank: usize| -> u64 {
            if !exact {
                return stored;
            }
            let blocks = intermediate / 128;
            let own = blocks / ranks + usize::from(rank < blocks % ranks);
            (routed as f64 * (own * 128) as f64 / intermediate as f64) as u64
        };
        for rank in 0..ranks {
            let stored = rank_bytes(rank);
            let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: options.spark_bytes,
                items: Vec::new(), kv_tokens: 0 };
            if matches!(family, "deepseek_v4" | "deepseek_v41" | "qwen4") {
                // Recover physical memory from the legacy usable inventory
                // (13 GiB excluded), then charge the family's observed host
                // footprint separately from expert allocations.
                device.capacity_bytes += 13 * GIB;
                device.items.push(Item::new(Category::Reserved, "host OS+sparknestd", "", costs.spark_host_bytes, allowance_basis));
            }
            let format = report.components.iter().find(|c| c.owner == Owner::SparkSliced)
                .map(|c| c.formats.keys().cloned().collect::<Vec<_>>().join("+")).unwrap_or_default();
            device.items.push(Item::new(Category::Experts, "routed_expert", format, stored, Basis::Exact));
            let workspace = costs.spark_workspace_bytes * options.spark_capacity_rows / 4096;
            device.items.push(Item::new(Category::Workspace, "expert waves", "", workspace, allowance_basis));
            device.items.push(Item::new(Category::Transport, "rdma rings", "", costs.spark_ring_bytes, allowance_basis));
            device.items.push(Item::new(Category::Runtime, "context+modules", "", 512 * MIB, allowance_basis));
            spark_devices.push(device);
        }
        if !exact && report.spark_rank_share * ranks as f64 > 1.001 {
            waste.push(Waste { device: format!("spark x{ranks}"), what: format!("routed slices padded to the widest \
                128-row slice ({:.1}% of the even share) on every rank", 100.0 * (stored - even) as f64 / even as f64),
                bytes: (stored - even) * ranks as u64 });
        }
    }

    // KV pool: per-device bytes per logical token from the family geometry.
    let geometry = model.cache_geometry(CacheOptions { coordinator_ranks: active_gpus,
        native_mtp_layers: if family == "deepseek_v4" || family == "qwen4" { cache_native_layers } else { 0 },
        prefill_rows: prefill_rows, glmf_decode_rows, ..Default::default() });
    let mut pool_tokens = 0;
    match geometry {
        Ok(Some(mut geometry)) => {
            for rank in &mut geometry.ranks {
                rank.pool_metadata_unit_bytes += match family {
                    "deepseek_v4" => (2 * prefill_rows + decode_rows) * 4,
                    "qwen4" => (1 + 64) * 5 * 4,
                    _ => 0,
                };
            }
            let warm_draft = if family == "mimo_v2" && drafter > 0 && options.mimo_prefix_draft {
                let draft_config = checkpoint.snapshot.join("dflash/config.json");
                if draft_config.is_file() {
                    Some(std::fs::read(&draft_config).map_err(|e| e.to_string())
                        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| e.to_string()))
                        .and_then(|draft| mimo_draft_prefix_bytes(&draft, 1, options.mimo_rings.max(concurrency))))
                } else { None }
            } else { None };
            // GLM 5.3 Flash takes no mark without entries, so pool marks then keep no unit back.
            let pool_marks = family == "glm5_flash" && options.glmf_pool_marks && options.mimo_prefix_entries > 0;
            let marks = if pool_marks { 0 } else { options.prefix_slots.unwrap_or_else(|| {
                let bytes: u64 = geometry.ranks.iter().map(|r| r.retained_mark_bytes).sum();
                if family == "mimo_v2" {
                    let draft = warm_draft.as_ref().and_then(|r| r.as_ref().ok()).map_or(0, |r| r.0);
                    cuteafd_core::prefix::mark_slots_for(concurrency, options.mimo_prefix_entries,
                        bytes.saturating_add(draft), options.mimo_prefix_mark_bytes)
                } else if family == "glm5_flash" {
                    // serve-glmf's arena with the same knobs, over one whole mark of the layout
                    // planned here (token keys, which a head split keeps).
                    cuteafd_core::prefix::mark_slots_for(concurrency, options.mimo_prefix_entries, bytes,
                        options.mimo_prefix_mark_bytes)
                } else if matches!(family, "deepseek_v4" | "qwen4") {
                    // The arena the family's server allocates at the default knobs (`MarkArena::slots_for`).
                    42.min((2 * GIB) / bytes.max(1)).max(2 * concurrency + 2)
                } else { costs.mark_slots }
            }) };
            let unit = geometry.logical_unit_rows.max(1);
            let per_token: Vec<u64> = (0..devices.len()).map(|d| geometry.ranks.get(d)
                .map_or(0, |r| (r.persistent_unit_bytes + r.pool_metadata_unit_bytes).div_ceil(unit))).collect();
            // Reserve fixed state and marks before sizing records. Otherwise
            // an automatic pool consumes the bytes those allocations need.
            for (device, rank) in devices.iter_mut().zip(&geometry.ranks) {
                let state_slots = options.state_slots.unwrap_or(if matches!(family, "qwen4" | "deepseek_v4" | "deepseek_v41") { concurrency } else { concurrency + 2 });
                device.items.push(Item::new(Category::Kv, "state", "", rank.fixed_state_bytes
                    + (rank.active_state_per_sequence_bytes + if family == "deepseek_v4" { rank.pool_metadata_unit_bytes } else { 0 }) * state_slots
                    + rank.speculative_replay_bytes.saturating_sub(glmf_shared_records)
                    + rank.context_table_bytes_per_token * context_tokens, Basis::Formula));
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
            if let Some(reservation) = warm_draft {
                let rings = options.mimo_rings.max(concurrency);
                let context_slots = crate::families::mimo_v2::draft_representation::mimo_draft_context_slots(
                    options.draft_sequences.max(concurrency), rings, options.draft_context_slots);
                notes.push(format!("MiMo DFlash context slots {context_slots}; valid-floor transfer uses {rings} target rings"));
                match reservation.and_then(|(bytes, floors)| bytes.checked_mul(marks)
                    .map(|bytes| (bytes, floors)).ok_or_else(|| "DFlash prefix marks overflow".into())) {
                    Ok((bytes, floors)) => {
                        devices[0].items.push(Item::new(Category::Prefix, "DFlash context marks", "BF16", bytes, Basis::Formula));
                        devices[0].items.push(Item::new(Category::Prefix, "DFlash valid-floor transfer", "i64", floors, Basis::Formula));
                    }
                    Err(reason) => {
                        report.placement_supported = false;
                        notes.push(format!("DFlash prefix reservation unsupported: {reason}"));
                        report.hints.push(super::Hint { what: format!("invalid dflash/config.json: {reason}"),
                            how: "Provide the loaded DFlash geometry before admitting its prefix marks.".into() });
                    }
                }
            }
            if family == "deepseek_v41" {
                let prefixes = v41::prefix_arena_bytes(marks, native_layers > 0, active_gpus);
                let retained_turns = marks.saturating_sub(2) / 2;
                for (device, &bytes) in devices.iter_mut().zip(&prefixes) {
                    device.items.push(Item::new(Category::Prefix, "snapshot arenas", "", bytes, Basis::Formula));
                    device.items.push(Item::new(Category::Kv, "active and retained COW tails", "", geometry.ranks[device.index as usize].persistent_unit_bytes * (concurrency + 2 * retained_turns), Basis::Formula));
                }
                if native_layers > 0 {
                    devices[active_gpus - 1].items.push(Item::new(Category::Drafter, "dSpark window state", "", v41::dspark_cache_bytes(concurrency, prefill_rows), Basis::Formula));
                }
            }
            {
                let target = options.pool_tokens.filter(|&tokens| tokens != 0)
                    .unwrap_or_else(|| if small_card { (1 << 20).max(context_tokens) } else { target_pool_tokens });
                let kv: Vec<u64> = per_token.iter().map(|&cost| cost.saturating_mul(target)).collect();
                resolve_encoder(checkpoint, report, model, &mut devices, &mut spark_devices, &kv, options, &mut notes);
            }
            let v4_placement = if family == "deepseek_v4" {
                (|| -> anyhow::Result<_> {
                    let catalog = crate::read_expert_catalog(&checkpoint.snapshot)?;
                    let routed = catalog.routed_experts();
                    let layers = (routed.first_layer..routed.layers).map(|layer|
                        crate::serving_capacity::deepseek_v4_expert_cost(&catalog, layer, false))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    let draft = (0..native_layers).map(|stage|
                        crate::serving_capacity::deepseek_v4_expert_cost(&catalog, stage, true))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    if split {
                        for (rank, device) in devices.iter_mut().take(active_gpus).enumerate() {
                            device.items.push(Item::new(Category::Transport, "expert peer exchange", "",
                                crate::serving_capacity::deepseek_v4_expert_exchange_bytes(model.spec().hidden as u64,
                                    model.spec().moe.as_ref().unwrap().top_k as u64, prefill_rows, decode_rows, rank)?, Basis::Formula));
                        }
                    }
                    let available: Vec<_> = devices.iter().take(active_gpus).map(|d| d.free_bytes().max(0) as u64).collect();
                    let costs: Vec<_> = geometry.ranks.iter().map(|r|
                        r.persistent_unit_bytes + r.pool_metadata_unit_bytes).collect();
                    let placement = crate::serving_capacity::deepseek_v4_placement(&available,
                        &options.rtx_bytes[..active_gpus], &costs, unit, options.pool_tokens, None,
                        routed.first_layer, &layers, &draft,
                        if catalog.exl3().is_some() { exl3_workspace.unwrap_or(160 * MIB * prefill_rows / 4096) }
                        else { crate::serving_capacity::deepseek_v4_native_workspace(routed.hidden as u64,
                            routed.intermediate as u64, routed.experts as u64, routed.topk as u64,
                            prefill_rows.max(decode_rows))? }, options.local_expert_layers)?;
                    for (rank, range) in placement.ranks.iter().enumerate() {
                        devices[rank].items.push(Item::new(Category::Experts,
                            &format!("resident routed layers {}..{}", range.first, range.first + range.layers),
                            "local", range.peak_bytes, Basis::Formula));
                        notes.push(format!("rtx{rank}: {} local expert layers ({}..{})", range.layers,
                            range.first, range.first + range.layers));
                    }
                    local_layers = placement.ranks.iter().map(|r| r.layers).sum();
                    local_bytes = placement.ranks.iter().map(|r| r.peak_bytes).sum();
                    Ok(placement)
                })().map_err(|error| {
                    report.placement_supported = false;
                    notes.push(format!("V4 pool-first placement: {error}"));
                }).ok()
            } else { None };
            let free: Vec<i64> = devices.iter().map(DeviceLayout::free_bytes).collect();
            pool_tokens = v4_placement.as_ref().map(|p| p.pool_tokens)
                .or_else(|| options.pool_tokens.filter(|&tokens| tokens != 0))
                .unwrap_or_else(|| size_pool(&free, &per_token, unit,
                    if family == "deepseek_v41" { if small_card { 1 << 20 } else { v41::DEFAULT_POOL_TOKENS } }
                    else if small_card { (1 << 20).max(context_tokens) } else { target_pool_tokens }));
            if pool_tokens < context_tokens {
                notes.push(format!("full-context admission shortfall: context {context_tokens} tokens, pool {pool_tokens} tokens, shortfall {} tokens", context_tokens - pool_tokens));
            }
            for (device, &cost) in devices.iter_mut().zip(&per_token) {
                if cost > 0 {
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
        Err(error) => notes.push(format!("{family}: cache geometry: {error}")),
    }

    if family == "deepseek_v41" && options.local_expert_layers.is_none() && layer_bytes > 0 {
        let per_gpu = if active_gpus == 2 { layer_bytes / 2 } else { layer_bytes };
        let room = devices.iter().take(active_gpus).map(|d| d.free_bytes().max(0) as u64).min().unwrap_or(0);
        local_layers = (room.saturating_sub(512 * MIB) / per_gpu).min(model.spec().layers.len() as u64) as usize;
        local_bytes = layer_bytes * local_layers as u64;
        for device in devices.iter_mut().take(active_gpus) {
            device.items.push(Item::new(Category::Experts, "resident routed layers", "native", per_gpu * local_layers as u64 + 256 * MIB, Basis::Formula));
        }
    }
    if local_layers > 0 { notes.push(format!("{local_layers} routed backbone layers resident on RTX")); }
    // V4.1 auto-selects local experts after KV sizing. Preserve its original
    // Spark slice accounting even though generic encoder placement needs the
    // Spark inventory earlier.
    if family == "deepseek_v41" {
        let routed: u64 = report.components.iter().filter(|c| c.owner == Owner::SparkSliced && c.component != Component::SpeculatorExpert).map(|c| c.bytes).sum();
        for spark in &mut spark_devices {
            if let Some(experts) = spark.items.iter_mut().find(|i| i.category == Category::Experts) {
                experts.bytes = (routed.saturating_sub(local_bytes) as f64 * report.spark_rank_share) as u64;
            }
        }
    }

    if report.encoder.is_none() {
        resolve_encoder(checkpoint, report, model, &mut devices, &mut spark_devices, &vec![0; gpus], options, &mut notes);
    }
    devices.extend(spark_devices);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// GLM 5.3 Flash's step workspaces on one GPU from its program manifest: the bytes its engine
/// allocates for the decode workspace of `decode_rows` rows and `lanes` prefill lanes of `rows` rows,
/// whose scratch also holds `shared_records` bytes of replay records (`--replay-records shared`)
/// (`serving_capacity::glmf_*`, which the engine sizes its buffers from).
#[allow(clippy::too_many_arguments)]
fn glmf_step_workspace(manifest: &serde_json::Value, checkpoint: &super::Checkpoint, placement: &ExpertPlacement,
    lanes: u64, rows: u64, context: u64, decode_rows: u64, shared_records: u64) -> Option<u64> {
    use crate::serving_capacity::{glmf_manifest_scratch, glmf_step_scratch, glmf_step_workspaces, glmf_table_pages,
        GlmfScratchOptions, GlmfStepShape};
    let cfg = crate::families::glm5_flash::GlmNextConfig::from_hf(&checkpoint.config).ok()?;
    let lookup = glmf_manifest_scratch(manifest);
    // FP8 KDA projections run the w8 programs: charge their scratch too where the build has them.
    let options = GlmfScratchOptions { kda_w8: lookup("glmf_kda_w8_m64").is_some() && lookup("glmf_kda_w8_m4096").is_some(),
        ..Default::default() };
    let context = if context > 0 { context } else { manifest["capacities"]["max_context"].as_u64().unwrap_or(131_072) };
    let (table_pages, table_pool_pages) = glmf_table_pages(context);
    let spark = matches!(placement, ExpertPlacement::Sparks { .. });
    let shape = GlmfStepShape { lead: true, split: false, local_experts: !spark, spark, partial_bytes: 2,
        output_shard: false, full_prefill_logits: false, table_pages, table_pool_pages };
    let decode = glmf_step_scratch(&lookup, &cfg, options, decode_rows, true).ok()?;
    let mut prefill = glmf_step_scratch(&lookup, &cfg, options, rows, false).ok()?;
    // Shared replay records live in the prefill scratch.
    prefill.programs = prefill.programs.max(shared_records);
    // A lane needs a Spark transport of its own: local experts prefill in one.
    let lanes = if spark { lanes } else { 1 };
    Some(glmf_step_workspaces(&cfg, usize::try_from(lanes).ok()?, rows, decode_rows, &shape, decode, prefill)
        .device_bytes())
}

fn qwen_exl3_arenas(checkpoint: &super::Checkpoint, mtp: bool) -> Option<(u64, u64)> {
    use crate::V41Exl3Layer;
    let catalog = crate::read_expert_catalog(&checkpoint.snapshot).ok()?;
    let shape = catalog.routed_experts();
    let manifest = catalog.exl3()?;
    let bytes = |layer| -> Option<u64> {
        u64::try_from(manifest.residency(layer, 1, 0).ok()?.device_arena_layout().ok()?.1).ok()
    };
    let backbone = (shape.first_layer..shape.layers).try_fold(0u64,
        |total, layer| total.checked_add(bytes(V41Exl3Layer::Backbone(layer))?))?;
    let draft = (0..if mtp { shape.draft_stages.min(1) } else { 0 }).try_fold(0u64,
        |total, stage| total.checked_add(bytes(V41Exl3Layer::Dspark(stage))?))?;
    Some((backbone, draft))
}

/// Local EXL3 arenas from the same capacity manifests used by the loader.
/// An exported PROGRAMS.json can have its expert JSON tree alongside it;
/// in an image the standard tree lives in ../lib/exl3 instead.
fn expert_workspace(report: &PlanReport, model: &dyn super::FamilyModel, checkpoint: &super::Checkpoint,
    manifest: Option<&std::path::Path>, rows: u64) -> Option<u64> {
    if !report.experts.as_ref()?.package.contains("exl3") { return None; }
    // Display labels name the checkpoint tier; the decoder can include an
    // adjacent tier too (Pro K2 uses its k23 package). Read the catalog contract.
    let catalog = crate::read_expert_catalog(&checkpoint.snapshot).ok()?;
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
    const CAPACITIES: [u64; 6] = [1, 16, 80, 256, 1024, 4096];
    if maximum > 4096 { return None; }
    let manifests = CAPACITIES.into_iter().filter(|&n| n <= maximum)
        .chain(CAPACITIES.into_iter().find(|&n| n >= maximum))
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

/// A component the family loader converts at load: bytes saved against the
/// checkpoint's source storage, and the resident format.
struct Conversion {
    component: Component,
    saved_bytes: u64,
    format: &'static str,
}

fn load_conversions(family: &str, checkpoint: &super::Checkpoint) -> Vec<Conversion> {
    use crate::families::mimo_v2::projection::{MimoProjectionLayout, MimoProjectionRepresentation as R};
    use crate::families::mimo_v2::weight_policy::{default_policy, MimoDefaultPolicy};
    match family {
        // The measured MiMo default stores the head and every target o_proj
        // as FP8 (one copy), not the checkpoint's BF16.
        "mimo_v2" => {
            let Ok(cfg) = crate::families::mimo_v2::MimoV2Config::from_hf(&checkpoint.config) else { return Vec::new() };
            if default_policy(checkpoint, &cfg) != MimoDefaultPolicy::Fp8 {
                return Vec::new();
            }
            let bytes = |rows: usize, cols: usize, r: R, ranks: u64| MimoProjectionLayout::new(rows as u64, cols as u64, r, ranks)
                .ok().and_then(|l| l.resident_bytes().ok()).unwrap_or(0);
            let head = bytes(cfg.vocab_size, cfg.hidden, R::Bf16, 1).saturating_sub(bytes(cfg.vocab_size, cfg.hidden, R::Fp8, 1));
            let o = cfg.heads * cfg.v_head_dim;
            let output = (bytes(cfg.hidden, o, R::Bf16, 1).saturating_sub(bytes(cfg.hidden, o, R::Fp8, 1))) * cfg.layers as u64;
            vec![Conversion { component: Component::LmHead, saved_bytes: head, format: "fp8-block128" },
                Conversion { component: Component::Attention, saved_bytes: output, format: "fp8-block128" }]
        }
        // GLM 5.3 Flash serves MLA, dense and shared-expert projections as FP8
        // from the official FP8 release (--fp8-snapshot); a BF16 checkpoint's
        // copies of them are not loaded. KDA in/out projections and the head
        // stay the checkpoint's BF16 by default (FP8 copies are opt-in:
        // --kda-fp8 row128, --fp8-head), so only the MLA half converts.
        "glm5_flash" => {
            let bf16 = |c: &crate::plan::checkpoint::CheckpointTensor| c.meta.dtype == cuteafd_core::DType::Bf16;
            let bytes = |filter: &dyn Fn(&str) -> bool| -> u64 {
                checkpoint.tensors.iter().filter(|t| bf16(t) && filter(&t.meta.name)).map(|t| t.meta.byte_length).sum()
            };
            let mla_layers: std::collections::BTreeSet<String> = checkpoint.tensors.iter()
                .filter(|t| t.meta.name.ends_with("self_attn.q_a_proj.weight"))
                .map(|t| t.meta.name.trim_end_matches("q_a_proj.weight").to_string()).collect();
            let mla = bytes(&|n: &str| mla_layers.iter().any(|p| n.starts_with(p.as_str()))
                && ["q_a_proj.weight", "kv_a_proj_with_mqa.weight", "q_b_proj.weight", "o_proj.weight"].iter().any(|s| n.ends_with(s)));
            let shared = bytes(&|n: &str| n.contains("shared_experts.") && n.ends_with("_proj.weight"));
            let dense = bytes(&|n: &str| n.contains(".mlp.") && !n.contains("experts") && n.ends_with("_proj.weight")
                && !n.contains(".gate."));
            vec![Conversion { component: Component::Attention, saved_bytes: mla / 2, format: "bf16+fp8" },
                Conversion { component: Component::SharedExpert, saved_bytes: shared / 2, format: "fp8" },
                Conversion { component: Component::DenseFfn, saved_bytes: dense / 2, format: "fp8" }]
        }
        _ => Vec::new(),
    }
}

/// Exact per-rank resident weights `(group, format, bytes)` for families
/// whose loader publishes its resident layout (MiMo: the codex capacity
/// contract's `MimoResidentLayout`, with the default weight policy).
fn resident_layout(family: &str, checkpoint: &super::Checkpoint, ranks: usize, mtp: bool) -> Option<Vec<Vec<(String, String, u64)>>> {
    use crate::families::mimo_v2::projection::MimoProjectionRepresentation as R;
    use crate::families::mimo_v2::resident::{MimoResidentLayout, MimoResidentOptions};
    use crate::families::mimo_v2::weight_policy::{default_policy, MimoDefaultPolicy};
    if family == "qwen4" {
        use crate::families::qwen4::{Qwen4Config, resident::{checkpoint_resident_bytes, Qwen4Representation}};
        let cfg = Qwen4Config::from_hf(&checkpoint.config).ok()?;
        let mtp = mtp && cfg.mtp_layers > 0;
        let bytes = checkpoint_resident_bytes(checkpoint, &cfg, cfg.layers, mtp,
            Qwen4Representation { fp8_projections: false, fp8_head: true }).ok()?;
        return Some(vec![vec![
            ("target incl PLE projections".into(), "bf16+f32".into(), bytes.target_bytes),
            ("embedding".into(), "bf16".into(), bytes.embedding_bytes),
            ("lm_head".into(), "fp8-row128".into(), bytes.head_bytes),
            ("speculator".into(), "bf16+f32".into(), bytes.mtp_bytes),
        ]]);
    }
    if family == "deepseek_v4" {
        return deepseek_v4::resident_weights(checkpoint, ranks);
    }
    if family != "mimo_v2" {
        return None;
    }
    let cfg = crate::families::mimo_v2::MimoV2Config::from_hf(&checkpoint.config).ok()?;
    let qualified = default_policy(checkpoint, &cfg) == MimoDefaultPolicy::Fp8;
    let source = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .map(|t| if t.meta.dtype == cuteafd_core::DType::Bf16 && !qualified { R::Bf16 } else { R::Fp8 });
    let output_formats = (0..cfg.layers).filter_map(|layer| {
        let name = format!("model.layers.{layer}.self_attn.o_proj.weight");
        source(&name).map(|r| (name, r))
    }).collect();
    let options = MimoResidentOptions {
        layers: cfg.layers,
        coordinator_ranks: ranks,
        checkpoint_tp: crate::families::mimo_v2::qkv::checkpoint_tp(&checkpoint.snapshot).ok()?,
        native_mtp_layers: 0,
        gpu_embedding: true,
        head_format: source("lm_head.weight").unwrap_or(R::Bf16),
        output_formats,
    };
    let layout = MimoResidentLayout::new(checkpoint, &cfg, &options).ok()?;
    Some(layout.ranks.iter().map(|rank| {
        let mut groups: std::collections::BTreeMap<(String, String), u64> = std::collections::BTreeMap::new();
        for reservation in rank {
            let name = reservation.name.as_str();
            let group = if name.contains("embed") { "embedding" } else if name.starts_with("lm_head") { "lm_head" }
                else if name.contains("mlp.gate") { "router" } else if name.contains("norm") { "norm" }
                else if name.contains("gate_up") || name.contains(".down") || name.contains("mlp.") { "dense_ffn" }
                else { "attention" };
            let format = if name.ends_with(".fp8") { "fp8" } else if name.ends_with("scale") { "fp8-scale" }
                else if name.ends_with(".bf16") { "bf16" } else { "native" };
            *groups.entry((group.to_string(), format.to_string())).or_default() += reservation.bytes;
        }
        groups.into_iter().map(|((g, f), b)| (g, f, b)).collect()
    }).collect())
}

// The host footprint belongs to physical capacity, not expertd's allocation
// budget. Neither the tower nor its admission guard may displace experts.
fn spark_encoder_headroom(device: &DeviceLayout, budget: Option<u64>) -> u64 {
    let physical = device.free_bytes().max(0) as u64;
    let allocated: u64 = device.items.iter().filter(|i| i.category != Category::Reserved)
        .map(|i| i.bytes).sum();
    budget.map_or(physical, |bytes| physical.min(bytes.saturating_sub(allocated)))
}

fn resolve_encoder(checkpoint: &super::Checkpoint, report: &mut PlanReport, model: &dyn super::FamilyModel, rtx: &mut [DeviceLayout], sparks: &mut [DeviceLayout], kv: &[u64], options: &LayoutOptions, notes: &mut Vec<String>) {
    use super::encoder::*;
    let source = report.components.iter().find(|c| c.component == Component::Vision);
    let source_bytes = source.map_or(0, |c| c.bytes);
    // MiMo resident vectors are FP32. The measured 4096-token native ledger
    // includes the fixed 4096-row GEMM staging and 4 MiB BLAS workspace.
    let (weights, scratch) = if report.family.as_deref() == Some("deepseek_v41") && source_bytes > 0 {
        (source_bytes, v41::vision_scratch_bytes())
    } else if report.family.as_deref() == Some("mimo_v2") && source_bytes > 0 {
        let width = model.spec().hidden as u64;
        (1_458_170_944 + width.saturating_sub(4096) * 5120 * 2, mimo_scratch_bytes(width))
    } else if report.family.as_deref() == Some("glm5_flash") && source_bytes > 0 {
        match super::families::glm::glm_flash_vision_resident_weights(checkpoint) {
            Ok(weights) => (weights, GLM_FLASH_ENCODER_SCRATCH_BYTES),
            Err(reason) if report.vision != super::MediaMode::Off => {
                report.placement_supported = false;
                report.hints.push(super::Hint {
                    what: format!("GLM Flash vision tower unavailable: {reason}"),
                    how: "Use the qualified official BF16 tower geometry or explicitly use --vision=off.".into(),
                });
                notes.push(format!("vision Off: {reason}; 0 bytes admitted"));
                report.encoder = Some(EncoderPlacement {
                    kind: EncoderKind::Off, weights: 0, scratch: 0, replicas: Vec::new(), reason, shortfall: 0,
                });
                return;
            }
            Err(_) => (0, 0),
        }
    } else if report.family.as_deref() == Some("qwen4") && source_bytes > 0 {
        (898_680_904, qwen_scratch_bytes(crate::media::QWEN_MAX_IMAGE_TOKENS as u64))
    } else { (source_bytes, 512 * MIB) };
    let hardware = EncoderHardware { v41: report.family.as_deref() == Some("deepseek_v41"),
        gpus: rtx.iter().enumerate().map(|(i,d)| EncoderGpuBudget { free_bytes: d.free_bytes().max(0) as u64, kv_target_bytes: kv.get(i).copied().unwrap_or(0) }).collect(),
        sparks: sparks.iter().map(|d| EncoderSparkBudget { rank: d.index as usize, host: format!("spark{}",d.index), idle: false,
            expert_bytes: d.items.iter().filter(|i| i.category == Category::Experts).map(|i| i.bytes).sum(), free_bytes: spark_encoder_headroom(d, options.spark_allocation_budget_bytes) }).collect() };
    let placement = encoder_placement(report.vision, &hardware, weights, scratch, options.vision_replicas);
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
            report.hints.push(super::Hint {
                what: format!("requested vision placement unavailable: {}", placement.reason),
                how: format!("Select an available encoder device with tower/scratch/KV headroom (shortfall {} bytes), or explicitly use --vision=off.", placement.shortfall),
            });
        }
        EncoderKind::Off => {
            if let Some(c) = report.components.iter_mut().find(|c| c.component == Component::Vision) {
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
    let plan = if report.family.as_deref() == Some("mimo_v2") {
        crate::media::audio_tower::AudioTowerPlan::from_snapshot(&checkpoint.snapshot,
            crate::media::audio_tower::AudioStorage::Fp32).map_err(|e| e.to_string())
    } else { Err("checkpoint family has no bundled audio runtime".into()) };
    let audio = plan.and_then(|plan| plan.scratch_bytes(crate::media::audio::MAX_CLIP_SAMPLES)
        .map(|scratch| (plan.weight_bytes(), scratch)).map_err(|e| e.to_string()));
    let (weights, scratch) = match audio {
        Ok(bytes) => bytes,
        Err(reason) => {
            report.placement_supported = false;
            report.hints.push(super::Hint { what: format!("audio tower unavailable: {reason}"),
                how: "Use --audio off or add the bundled tensors/exporter".into() });
            report.audio_encoder = Some(EncoderPlacement { kind: EncoderKind::Off, weights: 0, scratch: 0,
                replicas: vec![], shortfall: 0, reason });
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

fn mimo_scratch_bytes(width: u64) -> u64 {
    let n = 4096 * 4;
    [n*1280*4,n*1280*4,n*1280*2,n*1536*2,n*2048*2,n*512*2,n*512*2,n*2048*2,4096*width*2,
     4096*3072*4,4096*1280*4,4096*2*4608*4,4096*4608*2,4096*5120*4,4096*5120*2,4096*width*4,
     n*16*16*3,3*256*4,n*2*4,n*2*4,4096*4,4096*4,4*MIB].into_iter().map(|b| b.div_ceil(256)*256).sum()
}

/// Mirrors scratch_qwen, including fixed staging, LSE and the BLAS workspace.
fn qwen_scratch_bytes(tokens: u64) -> u64 {
    let n = tokens * 4;
    [n*1152*4, n*1152*4, n*1152*2, n*1536*2, n*1152*2, n*1152*2,
     n*1152*2, n*1152*2, tokens*2560*2, 4096*3456*4, 4096*1152*4,
     4096*4304*4, 4096*4304*2, 4096*4608*4, 4096*4608*2, 4096*2560*4,
     n*16*16*3, 3*256*4, n*2*4, n*4*4, n*4*4, 16384*16*4, 2*4, 4*MIB]
        .into_iter().map(|b| b.div_ceil(256)*256).sum()
}
