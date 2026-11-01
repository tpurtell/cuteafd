//! Dense hybrid Qwen admission. Every byte is future to the supplied baseline;
//! the executor supplies native packed sizes and exported workspace costs.
use crate::families::{qwen::HybridAttention, qwen3_5::Qwen35Config};
use crate::placement::*;
use anyhow::{ensure, Result};
use cuteafd_core::DType;
use cuteafd_core::memory_layout::{Basis, Category};

pub const PAGE_ROWS: u64 = 64;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
/// Only the initial single-device dense executor; not a claim of TP2 support.
pub const SINGLE: ExecutorModes = ExecutorModes { family: "qwen3_5", modes: &[WHOLE],
    hops: false, attention: AttentionPlacement::Heads };

#[derive(Debug, Clone, Copy)]
pub struct LayerBytes {
    pub weights: u64,
    /// Active recurrence and convolution slots, excluding retained prefix marks.
    pub state: u64,
    pub replay: u64,
    pub prefix: u64,
}

/// Drafter-owned allocations, excluding the target embedding and vocabulary head.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrafterBytes {
    pub owned: u64,
    pub scratch: u64,
    /// Full-context draft K/V per target pool page, not per sequence/context.
    pub pool_unit: u64,
}

impl DrafterBytes {
    /// Native MTP owns one dense block, bounded hidden/logit planes and pooled KV sidecars.
    pub fn native_mtp(checkpoint: &crate::plan::checkpoint::Checkpoint, cfg: &Qwen35Config) -> anyhow::Result<Self> {
        use anyhow::Context;
        ensure!(cfg.mtp_layers == 1, "dense MTP requires one checkpoint-native layer");
        let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
            .with_context(|| format!("dense MTP tensor missing: {name}"));
        let fp8 = tensor("mtp.layers.0.mlp.gate_proj.weight")?.meta.dtype == DType::F8E4M3;
        let mut groups = vec![vec!["input_layernorm.weight"], vec!["post_attention_layernorm.weight"],
            vec!["mlp.down_proj.weight"], vec!["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            vec!["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
            vec!["self_attn.q_norm.weight"], vec!["self_attn.k_norm.weight"], vec!["self_attn.o_proj.weight"]];
        let mut owned = 0u64;
        for group in groups.drain(..) {
            let projection = group[0].starts_with("mlp.") || matches!(group[0], "self_attn.q_proj.weight" | "self_attn.o_proj.weight");
            let mut bytes = 0;
            let mut scales = 0;
            for name in group {
                let full = format!("mtp.layers.0.{name}");
                let value = &tensor(&full)?.meta;
                ensure!(value.dtype == if fp8 && projection { DType::F8E4M3 } else { DType::Bf16 },
                    "{full}: unsupported native MTP dtype");
                bytes += value.byte_length;
                if fp8 && projection {
                    let scale = &tensor(&format!("{full}_scale_inv"))?.meta;
                    ensure!(matches!(scale.dtype, DType::Bf16 | DType::F32), "{full}: unsupported scale dtype");
                    scales += scale.byte_length * if scale.dtype == DType::Bf16 { 2 } else { 1 };
                }
            }
            owned += bytes.max(256);
            if fp8 && projection { owned += scales.max(256); }
        }
        for (name, shape) in [("fc", vec![cfg.hidden, 2 * cfg.hidden]), ("norm", vec![cfg.hidden]),
            ("pre_fc_norm_hidden", vec![cfg.hidden]), ("pre_fc_norm_embedding", vec![cfg.hidden])] {
            let full = format!("mtp.{name}.weight");
            let value = &tensor(&full)?.meta;
            ensure!(value.dtype == DType::Bf16 && value.shape == shape, "{full}: native MTP shape/dtype differs");
            owned += value.byte_length.max(256);
        }
        owned += ((64 * 2 + 4096) * cfg.hidden * 2 + 64 * cfg.vocab * 4) as u64;
        // The independent greedy selector owns one ID/status/log-probability row.
        owned += 12;
        let record = crate::families::qwen4::Qwen4KvCache::Bf16.record_bytes(cfg.kv_heads, cfg.head_dim);
        Ok(Self { owned, scratch: 0, pool_unit: (64 * record + cfg.hidden * 2) as u64 })
    }

    pub fn block_dspark(config: &serde_json::Value, slots: usize, sms: u64, context: usize)
        -> anyhow::Result<Self> {
        use crate::families::glm5::draft_representation::{block_dspark_paged_resident_bytes, GlmDraftRepresentation};
        use crate::families::qwen4::Qwen4KvCache;
        let cfg = crate::formats::block_dspark::BlockDsparkConfig::from_json(config)?;
        ensure!(cfg.window.is_none(), "dense dSpark requires full-context cache");
        let layout = cfg.paged_cache_layout(slots, context, PAGE_ROWS as usize, Qwen4KvCache::Bf16)?;
        let (owned, scratch) = block_dspark_paged_resident_bytes(config, slots, slots, sms,
            GlmDraftRepresentation::Bf16Only, 0, context, PAGE_ROWS as usize, Qwen4KvCache::Bf16)?;
        Ok(Self { owned: owned.checked_sub(layout.full_bytes).ok_or_else(|| anyhow::anyhow!("draft inventory underflow"))?,
            scratch, pool_unit: layout.full_bytes })
    }

    fn movable(self) -> Result<Movable, PlacementError> {
        crate::placement::draft::DrafterDemand::local(MovableId::Drafter, vec![Bytes2 { resident: self.owned, staging: 0 },
            Bytes2 { resident: self.scratch, staging: 0 }], 0).movable(1)
    }
}

pub struct DenseInputs<'a> {
    pub cfg: &'a Qwen35Config,
    pub gpu: GpuBudget,
    pub layers: &'a [LayerBytes],
    /// Embedding, head, native norm, workspaces, graphs and pending loaded code.
    /// Callers retain the categories and names of their actual allocations.
    pub fixed: Vec<Demand>,
    pub drafter: Option<DrafterBytes>,
    pub requested_pool: Option<u64>,
    pub max_context: u64,
    /// Checkpoint/export-native K|V record width (BF16 initially, native FP8 later).
    pub record_bytes: u64,
    pub pool_overhead: u64,
    pub pool_bounded: Vec<BoundedDemand>,
}

pub fn request(mut inputs: DenseInputs<'_>) -> Result<PlacementRequest> {
    ensure!(!inputs.cfg.attention.is_empty() && inputs.layers.len() == inputs.cfg.attention.len(),
        "dense Qwen layer byte inventory differs from checkpoint");
    ensure!(inputs.record_bytes > 0 && inputs.max_context > 0, "invalid dense Qwen cache geometry");
    ensure!(inputs.fixed.iter().all(|item| item.gpu == 0)
        && inputs.pool_bounded.iter().all(|item| item.gpu == 0),
        "single-device dense Qwen demand names a peer");
    if let Some(draft) = inputs.drafter.filter(|draft| draft.pool_unit > 0) {
        inputs.pool_bounded.push(BoundedDemand { gpu: 0, category: Category::Drafter,
            group: "full-context dSpark KV".into(), bytes_per_unit: draft.pool_unit,
            offset_units: 0, cap_units: u64::MAX, basis: Basis::Exact });
    }
    let layers = inputs.cfg.attention.iter().zip(inputs.layers).map(|(&kind, bytes)| {
        ensure!(bytes.weights > 0, "dense Qwen layer has no native weight admission");
        let mut extra = Vec::new();
        let kv = match kind {
            HybridAttention::Gdn => {
                ensure!(bytes.state > 0 && bytes.replay > 0, "dense GDN lacks exact state/replay admission");
                for (category, group, value) in [(Category::Kv, "GDN state", bytes.state),
                    (Category::Workspace, "GDN replay", bytes.replay),
                    (Category::Prefix, "GDN exact marks", bytes.prefix)] {
                    extra.push(LayerExtraDemand::new(category, group, value, LayerResidency::Owner, Basis::Exact));
                }
                0
            }
            HybridAttention::Full => {
                ensure!(bytes.state == 0 && bytes.replay == 0 && bytes.prefix == 0,
                    "full GQA must not carry fabricated recurrent state");
                inputs.record_bytes.checked_mul(PAGE_ROWS).ok_or_else(|| anyhow::anyhow!("dense KV unit size overflow"))?
            }
        };
        Ok(LayerDemand { kind: if kind == HybridAttention::Gdn { AttentionClass::Gdn } else { AttentionClass::Gqa },
            weights: ModeBytes { whole: bytes.weights, ..Default::default() },
            kv_unit: KvDemand { unit_bytes_whole: kv, ..Default::default() },
            fixed_bytes: ModeBytes::default(), extra, context_indexer: false, colocate: None,
            experts: None, modes: vec![WHOLE] })
    }).collect::<Result<_>>()?;
    Ok(PlacementRequest {
        inventory: Inventory { gpus: vec![inputs.gpu], spark_ranks: 0, peer_access: false },
        attention_placement: None, context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0,
        pool: PoolPolicy::resolve(&[inputs.gpu.capacity_bytes], inputs.max_context,
            inputs.requested_pool, PAGE_ROWS, true),
        layers, pool_overhead: vec![inputs.pool_overhead], pool_bounded: inputs.pool_bounded,
        fixed: inputs.fixed, startup: Vec::new(), movables: inputs.drafter.map(DrafterBytes::movable).transpose()?.into_iter().collect(),
        expert_workspace: 0, tp2_workspace: [0; 2],
        onboard: Onboard::Auto, expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![WHOLE], ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: inputs.cfg.hidden as u64 * 2,
            rows: 0, lanes: 0, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner },
        executor: SINGLE,
    })
}

/// Native admission uses the same solver in offline planning and device startup.
pub fn placement(inputs: DenseInputs<'_>) -> Result<Placement> {
    let result = solve(&request(inputs)?)?;
    SINGLE.check(&result)?;
    Ok(result)
}

pub const PREFILL_ROWS: usize = 4096;
pub const VERIFY_ROWS: usize = 64;
pub const HEAD_WORKSPACE: usize = 4 << 20;

/// Exact allocation recipe, excluding checkpoint weights and the KV pool.
/// Dense execution currently launches directly; it captures no CUDA graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBuffers {
    pub planes: [usize; 13],
    pub logits: usize,
    pub head_rows: usize,
    pub ids: usize,
    pub head: usize,
    pub commit: usize,
    pub state_per_layer: usize,
    pub replay_per_layer: usize,
    pub verify: [usize; 3],
    pub prefix: usize,
    pub mark_bytes: usize,
    pub marks: usize,
}

impl NativeBuffers {
    pub fn new(cfg: &Qwen35Config, scratch: usize, slots: usize, marks: usize, full_prefill_logits: bool) -> Result<Self> {
        ensure!((1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots), "dense state storage capacity is 160 slots");
        // Bound all products below before computing the native allocation recipe.
        ensure!(cfg.hidden <= 16_777_216 && cfg.vocab <= 16_777_216
            && cfg.heads <= 1024 && cfg.head_dim <= 256 && cfg.attention.len() <= 4096,
            "dense allocation geometry exceeds supported bounds");
        let g = cfg.gdn;
        ensure!(g.key_heads > 0 && g.key_heads <= 1024 && g.value_heads > 0 && g.value_heads <= 1024
            && g.key_dim == 128 && g.value_dim == 128 && g.conv_kernel == 4
            && cfg.max_context <= 16_777_216, "unsupported dense native state geometry");
        let state_per_layer = g.state_bytes();
        let replay_row = g.key_heads * g.key_dim * 4 + g.value_heads * g.value_dim * 4
            + g.value_heads * 8 + g.conv_width() * 2;
        let replay_per_layer = (VERIFY_ROWS * replay_row).div_ceil(1024) * 1024;
        let gdn = cfg.attention.iter().filter(|&&kind| kind == HybridAttention::Gdn).count();
        let mark_bytes = gdn.checked_mul(state_per_layer).ok_or_else(|| anyhow::anyhow!("dense mark overflow"))?;
        let hidden = PREFILL_ROWS * cfg.hidden * 2;
        let query = PREFILL_ROWS * cfg.heads * cfg.head_dim * 2;
        let planes = [hidden, hidden, hidden, query, query, query, scratch,
            PREFILL_ROWS * 4, PREFILL_ROWS * 4, PREFILL_ROWS * 12, PREFILL_ROWS * 8,
            PREFILL_ROWS * 8, VERIFY_ROWS * cfg.max_context.div_ceil(64) * 4].map(|n| n.max(256));
        let head_rows = if full_prefill_logits { PREFILL_ROWS } else { VERIFY_ROWS };
        Ok(Self { planes, logits: head_rows * cfg.vocab * 4, head_rows, ids: PREFILL_ROWS * 4,
            head: HEAD_WORKSPACE, commit: 3 * VERIFY_ROWS * 4, state_per_layer, replay_per_layer,
            verify: [mark_bytes.checked_mul(slots).ok_or_else(|| anyhow::anyhow!("dense verify mark overflow"))?.max(256),
                VERIFY_ROWS * cfg.vocab * 4, cfg.attention.len() * VERIFY_ROWS * cfg.hidden * 2],
            prefix: marks.checked_mul(mark_bytes).ok_or_else(|| anyhow::anyhow!("dense prefix overflow"))?, mark_bytes, marks })
    }

    pub fn fixed(&self) -> Vec<Demand> {
        [(Category::Workspace, "native step planes and scratch", self.planes.iter().sum()),
            (Category::Workspace, "vocabulary head workspace", self.head),
            (Category::Workspace, "logits", self.logits), (Category::Workspace, "token ids", self.ids),
            (Category::Workspace, "GDN commit table", self.commit),
            (Category::Prefix, "exact retained prefix marks", self.prefix),
            (Category::Workspace, "block verify marks, logits and taps", self.verify.iter().sum()),
            (Category::Runtime, "head and runtime growth", 512 << 20)]
            .into_iter().map(|(category, group, bytes)| Demand::new(0, category, group, bytes as u64, Basis::Exact)).collect()
    }
}

/// Header-only native BF16/E4M3 allocations, including exact block scales and widened GDN operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeWeights {
    pub layers: Vec<u64>,
    pub norm: u64,
    pub head: u64,
    pub embedding: u64,
}

impl NativeWeights {
    pub fn native(checkpoint: &crate::plan::checkpoint::Checkpoint, cfg: &Qwen35Config,
        norm_dtype: cuteafd_core::DType) -> Result<Self> {
        use cuteafd_core::DType;
        ensure!(matches!(norm_dtype, DType::Bf16 | DType::F32), "invalid native GDN norm dtype");
        let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
            .ok_or_else(|| anyhow::anyhow!("checkpoint lacks {name}"));
        let bytes = |name: &str, widen: bool, norm: bool| -> Result<u64> {
            let t = tensor(name)?;
            let dtype = &t.meta.dtype;
            ensure!((widen && matches!(dtype, DType::Bf16 | DType::F32))
                || (!widen && *dtype == if norm { norm_dtype.clone() } else { DType::Bf16 }),
                "{name}: unsupported native dense BF16 operand {dtype:?}");
            Ok(if widen && *dtype == DType::Bf16 { t.meta.byte_length * 2 } else { t.meta.byte_length })
        };
        let fp8 = tensor("model.language_model.layers.0.mlp.gate_proj.weight")?.meta.dtype == DType::F8E4M3;
        let mut layers = Vec::new();
        for (index, &kind) in cfg.attention.iter().enumerate() {
            let p = format!("model.language_model.layers.{index}.");
            let one = |name: &str| bytes(&format!("{p}{name}.weight"), false, false).map(|n| n.max(256));
            let pack = |names: &[&str]| -> Result<u64> {
                names.iter().try_fold(0u64, |n, name| Ok(n + bytes(&format!("{p}{name}.weight"), false, false)?))
                    .map(|n| n.max(256))
            };
            let projection = |names: &[&str]| -> Result<u64> {
                if !fp8 { return pack(names); }
                let mut weights = 0u64;
                let mut scales = 0u64;
                let mut columns = None;
                for name in names {
                    let name = format!("{p}{name}.weight");
                    let w = tensor(&name)?;
                    let scale = tensor(&format!("{name}_scale_inv"))?;
                    ensure!(w.meta.dtype == DType::F8E4M3 && w.meta.shape.len() == 2
                        && w.meta.shape.iter().all(|n| *n > 0 && n % 128 == 0),
                        "{name}: expected whole 128x128 E4M3 blocks");
                    ensure!(columns.is_none_or(|n| n == w.meta.shape[1]), "{name}: packed columns differ");
                    columns = Some(w.meta.shape[1]);
                    ensure!(scale.meta.shape == w.meta.shape.iter().map(|n| n / 128).collect::<Vec<_>>()
                        && matches!(scale.meta.dtype, DType::Bf16 | DType::F32),
                        "{name}: invalid native FP8 block scale grid");
                    weights += w.meta.byte_length;
                    scales += if scale.meta.dtype == DType::Bf16 { scale.meta.byte_length * 2 } else { scale.meta.byte_length };
                }
                Ok(weights.max(256) + scales.max(256))
            };
            let mut total = one("input_layernorm")? + one("post_attention_layernorm")?
                + projection(&["mlp.down_proj"])? + projection(&["mlp.gate_proj", "mlp.up_proj"])?;
            total += if kind == HybridAttention::Gdn {
                (if fp8 { projection(&["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"])?
                        + pack(&["linear_attn.in_proj_b", "linear_attn.in_proj_a"])? }
                     else { pack(&["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"])? })
                    + projection(&["linear_attn.out_proj"])?
                    + bytes(&format!("{p}linear_attn.conv1d.weight"), true, false)?.max(256)
                    + bytes(&format!("{p}linear_attn.A_log"), true, false)?.max(256)
                    + bytes(&format!("{p}linear_attn.dt_bias"), true, false)?.max(256)
                    + bytes(&format!("{p}linear_attn.norm.weight"), false, true)?.max(256)
            } else {
                projection(&["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])?
                    + projection(&["self_attn.o_proj"])? + one("self_attn.q_norm")? + one("self_attn.k_norm")?
            };
            layers.push(total);
        }
        let embedding = bytes("model.language_model.embed_tokens.weight", false, false)?;
        Ok(Self { layers, norm: bytes("model.language_model.norm.weight", false, false)?.max(256),
            head: bytes(if cfg.tied_embeddings { "model.language_model.embed_tokens.weight" } else { "lm_head.weight" }, false, false)?.max(256), embedding })
    }

    pub fn demands(&self, cfg: &Qwen35Config, buffers: &NativeBuffers, slots: usize) -> (Vec<LayerBytes>, Vec<Demand>) {
        let layers = cfg.attention.iter().zip(&self.layers).map(|(&kind, &weights)| LayerBytes { weights,
            state: if kind == HybridAttention::Gdn { (buffers.state_per_layer * slots) as u64 } else { 0 },
            replay: if kind == HybridAttention::Gdn { buffers.replay_per_layer as u64 } else { 0 }, prefix: 0 }).collect();
        let mut fixed = buffers.fixed();
        fixed.extend([(Category::Embedding, "token embedding", self.embedding),
            (Category::Weights, "final norm", self.norm), (Category::Weights, "vocabulary head", self.head)]
            .into_iter().map(|(category, group, bytes)| Demand::new(0, category, group, bytes, Basis::Exact)));
        (layers, fixed)
    }
}

/// CPU-readable contract shared with runtime native ABI validation.
#[derive(Debug, Clone)]
pub struct NativeProgram {
    pub name: String,
    pub geometry: serde_json::Value,
    pub capacity_rows: u64,
    pub pointers: Vec<String>,
    pub scalar_kinds: Vec<u8>,
    pub scratch: std::collections::HashMap<String, u64>,
}

impl NativeProgram {
    pub fn from_manifest(entry: &serde_json::Value) -> Result<Self> {
        use anyhow::Context;
        let name = entry["name"].as_str().context("native program name missing")?.to_owned();
        let pointers = entry["pointers"].as_array().context("native pointers missing")?.iter()
            .map(|p| p["name"].as_str().map(str::to_owned).context("native pointer name missing"))
            .collect::<Result<_>>()?;
        let scalar_kinds = entry["scalars"].as_array().context("native scalars missing")?.iter()
            .map(|p| match p["type"].as_str() {
                Some("int32") => Ok(b'i'), Some("int64") => Ok(b'l'), Some("float32") => Ok(b'f'),
                _ => anyhow::bail!("{name}: unsupported native scalar"),
            }).collect::<Result<_>>()?;
        let scratch = entry["scratch_bytes_at_capacity"].as_object().context("native scratch inventory missing")?
            .iter().map(|(key, n)| Ok((key.clone(), n.as_u64().context("invalid native scratch size")?)))
            .collect::<Result<_>>()?;
        Ok(Self { name, geometry: entry["geometry"].clone(),
            capacity_rows: entry["capacity_rows"].as_u64().context("native capacity missing")?,
            pointers, scalar_kinds, scratch })
    }
}

fn integer(spec: &NativeProgram, field: &str, expected: usize) -> Result<()> {
    ensure!(spec.geometry[field].as_u64() == Some(expected as u64),
        "{}: geometry.{field} must be {expected}, found {}", spec.name, spec.geometry[field]);
    Ok(())
}

fn number(spec: &NativeProgram, field: &str, expected: f64) -> Result<()> {
    ensure!(spec.geometry[field].as_f64() == Some(expected),
        "{}: geometry.{field} must be {expected}, found {}", spec.name, spec.geometry[field]);
    Ok(())
}

pub fn norm_dtype(spec: &NativeProgram) -> Result<DType> {
    match spec.geometry["norm_dtype"].as_str() {
        Some("torch.float32") => Ok(DType::F32),
        Some("torch.bfloat16") => Ok(DType::Bf16),
        _ => anyhow::bail!("{}: missing or unsupported GDN norm dtype", spec.name),
    }
}

pub fn program_name(prefix: &str, op: &str, rows: usize, format: crate::families::qwen4::Qwen4KvCache) -> String {
    let suffix = if matches!(op, "attn_producer" | "full_gqa") && format == crate::families::qwen4::Qwen4KvCache::Fp8 { "_kv_fp8" } else { "" };
    if rows == 0 { format!("{prefix}_{op}{suffix}") } else { format!("{prefix}_{op}{suffix}_m{rows}") }
}

pub fn native_program_name(prefix: &str, op: &str, rows: usize,
    kv: crate::families::qwen4::Qwen4KvCache, fp8: bool) -> String {
    let op = if fp8 && matches!(op, "gdn" | "attn_producer" | "attn_o" | "ffn") {
        format!("{op}_fp8")
    } else { op.to_owned() };
    let suffix = if kv == crate::families::qwen4::Qwen4KvCache::Fp8
        && matches!(op.as_str(), "attn_producer_fp8" | "full_gqa") { "_kv_fp8" } else { "" };
    if fp8 { if rows == 0 { format!("{prefix}_{op}{suffix}") } else { format!("{prefix}_{op}{suffix}_m{rows}") } }
    else { program_name(prefix, &op, rows, kv) }
}

pub fn validate_projection_format(op: &str, fp8: bool, spec: &NativeProgram) -> Result<()> {
    if matches!(op, "gdn" | "attn_producer" | "attn_o" | "ffn") {
        ensure!((spec.geometry["fp8_weights"] == "only") == fp8,
            "{}: checkpoint projection format mismatch", spec.name);
        if fp8 {
            ensure!(spec.geometry["weight_block"] == serde_json::json!([128, 128]),
                "{}: invalid FP8 block geometry", spec.name);
        }
    }
    Ok(())
}

pub fn validate(cfg: &Qwen35Config, op: &str, rows: usize, format: crate::families::qwen4::Qwen4KvCache, spec: &NativeProgram) -> Result<()> {
    if rows > 0 {
        ensure!(spec.capacity_rows as usize >= rows, "{} capacity is smaller than {rows}", spec.name);
        integer(spec, "max_rows", rows)?;
    }
    if matches!(op, "norm" | "add" | "gdn" | "attn_producer" | "attn_o" | "ffn") {
        integer(spec, "hidden", cfg.hidden)?;
    }
    match op {
        "norm" => {
            number(spec, "eps", cfg.rms_norm_eps)?;
            ensure!(spec.geometry["zero_centered"] == true, "{}: dense norm must be zero-centered", spec.name);
        }
        "gdn" => {
            integer(spec, "key_heads", cfg.gdn.key_heads)?;
            integer(spec, "value_heads", cfg.gdn.value_heads)?;
            integer(spec, "head_dim", cfg.gdn.key_dim)?;
            integer(spec, "in_width", cfg.gdn.conv_width() + cfg.gdn.value_width() + 2 * cfg.gdn.value_heads)?;
            number(spec, "eps", cfg.rms_norm_eps)?;
            ensure!(spec.geometry["output_gate"] == "silu"
                && (spec.geometry["fp8_weights"] == false || spec.geometry["fp8_weights"] == "only"),
                "{}: dense GDN needs SiLU and native checkpoint projections", spec.name);
            if spec.geometry["fp8_weights"] == "only" {
                ensure!(spec.geometry["gate_weights"] == "bf16" && spec.geometry["gate_order"] == serde_json::json!(["b", "a"]),
                    "{}: native GDN gates require BF16 b/a order", spec.name);
            }
            norm_dtype(spec)?;
        }
        "gdn_commit" => {
            integer(spec, "key_heads", cfg.gdn.key_heads)?;
            integer(spec, "value_heads", cfg.gdn.value_heads)?;
            integer(spec, "channels", cfg.gdn.conv_width())?;
            integer(spec, "replay_rows", VERIFY_ROWS)?;
            integer(spec, "replay_bytes", NativeBuffers::new(cfg, 0, 1, 0, false)?.replay_per_layer)?;
        }
        "attn_producer" | "full_gqa" => {
            integer(spec, "heads", cfg.heads)?;
            integer(spec, "kv_heads", cfg.kv_heads)?;
            integer(spec, "head_dim", cfg.head_dim)?;
            integer(spec, "record_bytes", format.record_bytes(cfg.kv_heads, cfg.head_dim))?;
            ensure!(spec.geometry["kv_format"] == format.to_string(), "{}: dense KV format mismatch", spec.name);
            ensure!(spec.geometry["indexed"] == false, "{}: dense attention must not use QSA", spec.name);
            if op == "attn_producer" {
                integer(spec, "in_width", 2 * (cfg.heads + cfg.kv_heads) * cfg.head_dim)?;
                integer(spec, "rope_dim", cfg.rope_dim)?;
                number(spec, "rope_theta", cfg.rope_theta)?;
            } else {
                integer(spec, "page_rows", 64)?;
                integer(spec, "max_splits", 32)?;
                ensure!(spec.geometry["causal"] == true && spec.geometry["empty_rows_zero"] == true,
                    "{}: dense attention requires causal traversal and zero inactive rows", spec.name);
            }
        }
        "attn_o" => {
            integer(spec, "width", cfg.heads * cfg.head_dim)?;
            ensure!(spec.geometry["fp8_weights"] == false || spec.geometry["fp8_weights"] == "only", "{}: expected native checkpoint output projection", spec.name);
        }
        "ffn" => integer(spec, "intermediate", cfg.intermediate)?,
        "add" => (),
        _ => anyhow::bail!("unknown dense Qwen op {op}"),
    }
    Ok(())
}

pub fn validate_abi(op: &str, rows: usize, spec: &NativeProgram) -> Result<()> {
    let mut pointers: Vec<&str> = match op {
        "norm" => vec!["x", "weight", "out"],
        "add" => vec!["a", "b", "out"],
        "gdn_commit" => vec!["state", "conv_state", "replay", "tables"],
        "gdn" => vec!["x", "w_in", "conv_w", "a_log", "dt_bias", "norm_w", "w_out",
            "conv_state", "state", "slots", "seq_first", "out"],
        "attn_producer" => vec!["x", "w_in", "q_norm", "k_norm", "rope_positions", "kv_slots",
            "kv_cache", "query", "gate", "scratch"],
        "full_gqa" => vec!["query", "kv_cache", "positions", "page_table", "out", "scratch"],
        "attn_o" => vec!["attn", "gate", "w_o", "out", "scratch"],
        "ffn" => vec!["x", "w_gate_up", "w_down", "out", "scratch"],
        _ => anyhow::bail!("unknown dense Qwen ABI {op}"),
    };
    let fp8 = spec.geometry["fp8_weights"] == "only";
    if fp8 {
        pointers = match op {
            "gdn" => vec!["x", "w_qkvz_fp8", "w_qkvz_scale", "w_ba", "conv_w", "a_log", "dt_bias", "norm_w",
                "w_out_fp8", "w_out_scale", "conv_state", "state", "slots", "seq_first", "out"],
            "attn_producer" => vec!["x", "w_in_fp8", "w_in_scale", "q_norm", "k_norm", "rope_positions", "kv_slots",
                "kv_cache", "query", "gate", "scratch"],
            "attn_o" => vec!["attn", "gate", "w_o_fp8", "w_o_scale", "out", "scratch"],
            "ffn" => vec!["x", "w_gate_up_fp8", "w_gate_up_scale", "w_down_fp8", "w_down_scale", "out", "scratch"],
            _ => anyhow::bail!("unexpected FP8 projection ABI {op}"),
        };
    }
    if op == "gdn" {
        if rows == VERIFY_ROWS { pointers.push("replay"); }
        pointers.push("scratch");
    }
    ensure!(spec.pointers == pointers, "{}: native pointer order differs from dense {op} ABI", spec.name);
    let scalars = match op {
        "gdn_commit" | "full_gqa" => 3,
        "gdn" if rows == VERIFY_ROWS => 2,
        "attn_o" if fp8 && rows > VERIFY_ROWS => 2,
        _ => 1,
    };
    ensure!(spec.scalar_kinds == vec![b'i'; scalars], "{}: dense scalar ABI differs", spec.name);
    if pointers.contains(&"scratch") {
        ensure!(spec.scratch.get("scratch").is_some_and(|&n| n > 0 && n % 1024 == 0),
            "{}: missing or unaligned scratch admission", spec.name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen::{GdnGeometry, GdnOutputGate};

    fn cfg() -> Qwen35Config {
        Qwen35Config { hidden: 4096, intermediate: 12288, vocab: 248320,
            attention: vec![HybridAttention::Gdn, HybridAttention::Full], heads: 16, kv_heads: 4, head_dim: 256,
            gdn: GdnGeometry { key_heads: 16, value_heads: 32, key_dim: 128, value_dim: 128,
                conv_kernel: 4, output_gate: GdnOutputGate::Silu }, rope_dim: 64, rope_theta: 1e7,
            rms_norm_eps: 1e-6, max_context: 262144, mtp_layers: 0, tied_embeddings: false }
    }

    fn inputs<'a>(cfg: &'a Qwen35Config, layers: &'a [LayerBytes], scratch: u64) -> DenseInputs<'a> {
        DenseInputs { cfg, layers, gpu: GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 1 << 30,
            baseline: Baseline::Measured { free_bytes: 14 << 30 } },
            fixed: vec![Demand::new(0, Category::Workspace, "native AOT scratch", scratch, Basis::Exact)],
            drafter: None, requested_pool: None, max_context: 262144, record_bytes: 4096,
            pool_overhead: 4, pool_bounded: Vec::new() }
    }

    #[test]
    fn native_fp8_header_inventory_charges_scales_and_keeps_bf16_gates() -> Result<()> {
        use crate::plan::checkpoint::{Checkpoint, CheckpointTensor};
        let cfg = cfg();
        let mut checkpoint = Checkpoint { snapshot: Default::default(), config: serde_json::json!({}),
            quantize_config: None, weight_map: Default::default(), tensors: vec![], missing_shards: vec![], shard_bytes: 0 };
        let mut add = |name: String, dtype, shape: Vec<usize>, byte_length| checkpoint.tensors.push(CheckpointTensor {
            shard: "model.safetensors".into(), meta: crate::SafetensorsTensorMetadata {
                name, dtype, shape, byte_offset: 0, byte_length } });
        for name in ["embed_tokens.weight", "norm.weight"] {
            add(format!("model.language_model.{name}"), DType::Bf16, vec![128], 256);
        }
        add("lm_head.weight".into(), DType::Bf16, vec![128], 256);
        for (index, kind) in cfg.attention.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{index}.");
            let mut ordinary = vec!["input_layernorm.weight", "post_attention_layernorm.weight"];
            let mut projections = vec!["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"];
            if *kind == HybridAttention::Gdn {
                ordinary.extend(["linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
                    "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.norm.weight"]);
                projections.extend(["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]);
            } else {
                ordinary.extend(["self_attn.q_norm.weight", "self_attn.k_norm.weight"]);
                projections.extend(["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]);
            }
            for name in ordinary { add(format!("{prefix}{name}"), DType::Bf16, vec![128], 256); }
            for name in projections {
                add(format!("{prefix}{name}.weight"), DType::F8E4M3, vec![128, 128], 16384);
                add(format!("{prefix}{name}.weight_scale_inv"), DType::Bf16, vec![1, 1], 2);
            }
        }
        let inventory = NativeWeights::native(&checkpoint, &cfg, DType::Bf16)?;
        assert_eq!(inventory.layers, [6 * 16384 + 4 * 256 + 2 * 256 + 2 * 256 + 3 * 512 + 256,
            7 * 16384 + 4 * 256 + 4 * 256]);
        for tensor in &mut checkpoint.tensors {
            if tensor.meta.name.ends_with("weight_scale_inv") {
                tensor.meta.dtype = DType::F32;
                tensor.meta.byte_length = 4;
            }
        }
        assert_eq!(NativeWeights::native(&checkpoint, &cfg, DType::Bf16)?, inventory);
        let scale = checkpoint.tensors.iter().position(|tensor| tensor.meta.name.ends_with("weight_scale_inv")).unwrap();
        checkpoint.tensors[scale].meta.shape = vec![2, 1];
        assert!(NativeWeights::native(&checkpoint, &cfg, DType::Bf16).is_err());
        checkpoint.tensors[scale].meta.shape = vec![1, 1];
        let gate = checkpoint.tensors.iter().position(|tensor| tensor.meta.name.ends_with("in_proj_a.weight")).unwrap();
        checkpoint.tensors[gate].meta.dtype = DType::F8E4M3;
        assert!(NativeWeights::native(&checkpoint, &cfg, DType::Bf16).is_err());
        checkpoint.tensors[gate].meta.dtype = DType::Bf16;
        checkpoint.tensors.remove(scale);
        assert!(NativeWeights::native(&checkpoint, &cfg, DType::Bf16).is_err());
        Ok(())
    }

    #[test]
    fn native_fp8_names_and_block_contract_match_offline_and_runtime_admission() -> Result<()> {
        use crate::families::qwen4::Qwen4KvCache::{Bf16, Fp8};
        for rows in [64, 4096] {
            assert_eq!(native_program_name("qwen38", "attn_producer", rows, Fp8, true),
                format!("qwen38_attn_producer_fp8_kv_fp8_m{rows}"));
            assert_eq!(native_program_name("qwen38", "full_gqa", rows, Fp8, true),
                format!("qwen38_full_gqa_kv_fp8_m{rows}"));
            assert_eq!(native_program_name("qwen38", "gdn", rows, Bf16, true),
                format!("qwen38_gdn_fp8_m{rows}"));
            for op in ["gdn", "attn_producer", "attn_o", "ffn"] {
                let mut spec = NativeProgram { name: op.into(), geometry: serde_json::json!({
                    "fp8_weights":"only", "weight_block":[128,128]}), capacity_rows: rows as u64,
                    pointers: vec![], scalar_kinds: vec![], scratch: Default::default() };
                validate_projection_format(op, true, &spec)?;
                assert!(validate_projection_format(op, false, &spec).is_err());
                spec.geometry["weight_block"] = serde_json::json!([64, 128]);
                assert!(validate_projection_format(op, true, &spec).is_err());
                spec.geometry["fp8_weights"] = serde_json::json!(false);
                validate_projection_format(op, false, &spec)?;
                assert!(validate_projection_format(op, true, &spec).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn native_scratch_and_exact_marks_are_charged_before_dense_pool() -> Result<()> {
        let cfg = cfg();
        let layers = [LayerBytes { weights: 1 << 30, state: 1 << 20, replay: 1 << 20, prefix: 2 << 20 },
            LayerBytes { weights: 1 << 30, state: 0, replay: 0, prefix: 0 }];
        let req = request(inputs(&cfg, &layers, 2 << 30))?;
        assert!(req.layers.iter().all(|layer| layer.experts.is_none()));
        assert_eq!(req.layers[0].kv_unit.unit_bytes_whole, 0);
        assert_eq!(req.layers[1].kv_unit.unit_bytes_whole, 64 * 4096);
        let base = solve(&req)?;
        assert!(base.layers.iter().all(|layer| layer.experts == ExpertHome::Dense));
        assert_eq!(base.pool_tokens, 2 << 20);
        let constrained = solve(&request(inputs(&cfg, &layers, 5 << 30))?)?;
        assert!(constrained.pool_tokens < base.pool_tokens && constrained.pool_tokens >= 262144);
        assert!(constrained.items[0].iter().any(|item| item.group == "native AOT scratch" && item.bytes == 5 << 30));
        let mut explicit = inputs(&cfg, &layers, 5 << 30);
        explicit.requested_pool = Some(2 << 20);
        assert!(solve(&request(explicit)?).is_err());
        Ok(())
    }

    #[test]
    fn drafter_owned_and_scratch_are_charged_once_before_pool_admission() -> Result<()> {
        let cfg = cfg();
        let layers = [LayerBytes { weights: 1 << 30, state: 1 << 20, replay: 1 << 20, prefix: 0 },
            LayerBytes { weights: 1 << 30, state: 0, replay: 0, prefix: 0 }];
        let base = solve(&request(inputs(&cfg, &layers, 2 << 30))?)?;
        let mut with_draft = inputs(&cfg, &layers, 2 << 30);
        with_draft.drafter = Some(DrafterBytes { owned: 3 << 30, scratch: 1 << 30, ..Default::default() });
        let placement = solve(&request(with_draft)?)?;
        assert_eq!(placement.movables, [(MovableId::Drafter, 0)]);
        assert!(placement.pool_tokens < base.pool_tokens && placement.pool_tokens >= 262144);
        let charged: u64 = placement.items[0].iter().filter(|item| item.category == Category::Drafter)
            .map(|item| item.bytes).sum();
        assert_eq!(charged, 4 << 30);
        let mut explicit = inputs(&cfg, &layers, 2 << 30);
        explicit.drafter = Some(DrafterBytes { owned: 3 << 30, scratch: 1 << 30, ..Default::default() });
        explicit.requested_pool = Some(base.pool_tokens);
        assert!(solve(&request(explicit)?).is_err());
        let mut overflow = inputs(&cfg, &layers, 0);
        overflow.drafter = Some(DrafterBytes { owned: u64::MAX, scratch: 1, ..Default::default() });
        assert!(solve(&request(overflow)?).is_err());
        Ok(())
    }

    #[test]
    fn full_context_drafter_kv_tracks_aggregate_pool_pages() -> Result<()> {
        let cfg = cfg();
        let layers = [LayerBytes { weights: 1 << 30, state: 1 << 20, replay: 1 << 20, prefix: 0 },
            LayerBytes { weights: 1 << 30, state: 0, replay: 0, prefix: 0 }];
        for tokens in [262144, 524288] {
            let mut input = inputs(&cfg, &layers, 0);
            input.gpu.capacity_bytes = 32 << 30;
            input.gpu.baseline = Baseline::Measured { free_bytes: 32 << 30 };
            input.requested_pool = Some(tokens);
            input.drafter = Some(DrafterBytes { owned: 1 << 30, scratch: 1 << 20, pool_unit: 64 * 20480 });
            let req = request(input)?;
            assert_eq!(req.pool_bounded.len(), 1);
            assert_eq!(req.pool_bounded[0].bytes(tokens / PAGE_ROWS)?, tokens * 20480);
            let placement = solve(&req)?;
            assert_eq!(placement.pool_tokens, tokens);
            let charged: u64 = placement.items[0].iter().filter(|item| item.category == Category::Drafter)
                .map(|item| item.bytes).sum();
            assert_eq!(charged, (1 << 30) + (1 << 20) + tokens * 20480);
        }
        Ok(())
    }

    #[test]
    fn admission_rejects_missing_or_fabricated_layer_costs() {
        let cfg = cfg();
        assert!(request(inputs(&cfg, &[], 1)).is_err());
        let layers = [LayerBytes { weights: 1, state: 1, replay: 0, prefix: 0 }; 2];
        assert!(request(inputs(&cfg, &layers, 1)).is_err());
        let layers = [LayerBytes { weights: 1, state: 1, replay: 1, prefix: 0 }; 2];
        assert!(request(inputs(&cfg, &layers, 1)).is_err());
    }
}
