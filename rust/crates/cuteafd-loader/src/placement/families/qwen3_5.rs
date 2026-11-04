//! Dense hybrid Qwen admission. Every byte is future to the supplied baseline;
//! the executor supplies native packed sizes and exported workspace costs.
use crate::families::{qwen::HybridAttention, qwen3_5::Qwen35Config};
use crate::placement::*;
use anyhow::{ensure, Result};
use cuteafd_core::DType;
use cuteafd_core::memory_layout::{Basis, Category};

pub mod weights;
mod admission;
pub use admission::{DenseFacts, ServingFacts, ServingRequest, admit, forecast_budgets, sampled_budgets};

pub const PAGE_ROWS: u64 = 64;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
/// Only the initial single-device dense executor; not a claim of TP2 support.
pub const SINGLE: ExecutorModes = ExecutorModes { family: "qwen3_5", modes: &[WHOLE],
    hops: false, attention: AttentionPlacement::Heads };

pub const PREFILL_ROWS: usize = 4096;
pub const VERIFY_ROWS: usize = 64;
pub const HEAD_WORKSPACE: usize = 4 << 20;

/// Diagnostic inputs and greedy selection are independent usage restrictions;
/// normal serving owns its selector but never the diagnostic input plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeMtpUsage {
    pub selector: bool,
    pub diagnostic_input: bool,
}

impl NativeMtpUsage {
    pub const SERVING: Self = Self { selector: true, diagnostic_input: false };
    pub const DIAGNOSTIC: Self = Self { selector: false, diagnostic_input: true };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeMtpLayout {
    pub owners: Vec<crate::formats::dflash::MlaDraftOwner>,
    pub sync_upload_bytes: u64,
    pub loading_host_bytes: u64,
    record_bytes: usize,
    hidden_bytes: usize,
}

impl NativeMtpLayout {
    pub fn new(checkpoint: &crate::plan::checkpoint::Checkpoint, cfg: &Qwen35Config,
        pool_tokens: usize, usage: NativeMtpUsage) -> Result<Self> {
        use crate::formats::dflash::{MlaDraftDomain as Domain, MlaDraftDtype as Dtype,
            MlaDraftLifetime as Life, MlaDraftOwner};
        use anyhow::Context;
        ensure!(cfg.mtp_layers == 1, "dense MTP requires one checkpoint-native layer");
        ensure!(pool_tokens > 0 && pool_tokens % PAGE_ROWS as usize == 0, "native MTP pool must contain complete pages");
        let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
            .with_context(|| format!("dense MTP tensor missing: {name}"));
        let fp8 = tensor("mtp.layers.0.mlp.gate_proj.weight")?.meta.dtype == DType::F8E4M3;
        let inventory = weights::native_inventory(checkpoint, "mtp.layers.0", HybridAttention::Full, fp8, DType::Bf16)?;
        let mut loading_host_bytes = inventory.loading_host_bytes;
        let mut sync_upload_bytes = inventory.allocations.iter().map(|a| a.upload_bytes).max().unwrap_or(0);
        let mut owners = Vec::new();
        let mut add = |id: String, domain, dtype, shape: Vec<usize>, bytes, weight: bool| {
            owners.push(MlaDraftOwner { id, domain, dtype, shape, bytes,
                resident_weights: ResidentWeights(if weight { bytes as u64 } else { 0 }), alignment: 1,
                lifetime: Life::Runtime, shared_key: None, staging_for: None, on_use: false });
        };
        for operand in inventory.allocations {
            let bytes = usize::try_from(operand.bytes)?;
            add(format!("draft.mtp.layer.{}", operand.id), Domain::Device, Dtype::Bytes, vec![bytes], bytes, true);
        }
        for (id, name, shape) in [
            ("fc", "fc", vec![cfg.hidden, cfg.hidden.checked_mul(2).context("MTP FC shape overflow")?]),
            ("norm", "norm", vec![cfg.hidden]),
            ("norm_hidden", "pre_fc_norm_hidden", vec![cfg.hidden]),
            ("norm_embedding", "pre_fc_norm_embedding", vec![cfg.hidden]),
        ] {
            let full = format!("mtp.{name}.weight");
            let value = &tensor(&full)?.meta;
            ensure!(value.dtype == DType::Bf16 && value.shape == shape, "{full}: native MTP shape/dtype differs");
            sync_upload_bytes = sync_upload_bytes.max(value.byte_length);
            loading_host_bytes = loading_host_bytes.max(value.byte_length);
            let bytes = usize::try_from(value.byte_length.max(256))?;
            add(format!("draft.mtp.{id}"), Domain::Device, Dtype::Bytes, vec![bytes], bytes, true);
        }
        let record_bytes = crate::families::qwen4::Qwen4KvCache::Bf16.record_bytes(cfg.kv_heads, cfg.head_dim);
        let hidden_bytes = cfg.hidden.checked_mul(2).context("MTP hidden extent overflow")?;
        let kv_bytes = pool_tokens.checked_mul(record_bytes).context("MTP KV extent overflow")?;
        let sidecar_bytes = (pool_tokens / PAGE_ROWS as usize).checked_mul(hidden_bytes).context("MTP sidecar extent overflow")?;
        for (id, dtype, shape, width) in [
            ("concat", Dtype::Bf16, vec![VERIFY_ROWS, 2, cfg.hidden], 2usize),
            ("hidden", Dtype::Bf16, vec![PREFILL_ROWS, cfg.hidden], 2),
            ("logits", Dtype::F32, vec![VERIFY_ROWS, cfg.vocab], 4),
        ] {
            let bytes = shape.iter().try_fold(width, |bytes, &n| bytes.checked_mul(n)).context("MTP plane extent overflow")?;
            add(format!("draft.mtp.{id}"), Domain::Device, dtype, shape, bytes, false);
        }
        add("draft.mtp.kv".into(), Domain::Device, Dtype::Bytes, vec![kv_bytes], kv_bytes, false);
        add("draft.mtp.sidecars".into(), Domain::Device, Dtype::Bf16,
            vec![pool_tokens / PAGE_ROWS as usize, cfg.hidden], sidecar_bytes, false);
        if usage.selector {
            add("draft.mtp.selector.output".into(), Domain::Device, Dtype::Bytes, vec![12], 12, false);
            add("draft.mtp.selector.landing".into(), Domain::HostPinned, Dtype::Bytes, vec![12], 12, false);
        }
        if usage.diagnostic_input {
            let bytes = VERIFY_ROWS.checked_mul(hidden_bytes).context("MTP diagnostic extent overflow")?;
            add("draft.mtp.diagnostic.input".into(), Domain::Device, Dtype::Bf16, vec![VERIFY_ROWS, cfg.hidden], bytes, false);
        }
        Ok(Self { owners, sync_upload_bytes, loading_host_bytes, record_bytes, hidden_bytes })
    }

    pub fn with_pool_tokens(&self, pool_tokens: usize) -> Result<Self> {
        use anyhow::Context;
        ensure!(pool_tokens > 0 && pool_tokens % PAGE_ROWS as usize == 0, "native MTP pool must contain complete pages");
        let mut layout = self.clone();
        for owner in &mut layout.owners {
            match owner.id.as_str() {
                "draft.mtp.kv" => {
                    owner.bytes = pool_tokens.checked_mul(self.record_bytes).context("MTP KV extent overflow")?;
                    owner.shape = vec![owner.bytes];
                }
                "draft.mtp.sidecars" => {
                    let pages = pool_tokens / PAGE_ROWS as usize;
                    owner.bytes = pages.checked_mul(self.hidden_bytes).context("MTP sidecar extent overflow")?;
                    owner.shape[0] = pages;
                }
                _ => {}
            }
        }
        Ok(layout)
    }

    pub fn device_bytes(&self) -> Result<u64> {
        use crate::formats::dflash::MlaDraftDomain;
        self.owners.iter().filter(|o| o.domain == MlaDraftDomain::Device)
            .try_fold(0u64, |sum, o| sum.checked_add(o.bytes as u64).ok_or_else(|| anyhow::anyhow!("MTP owner total overflow")))
    }

    /// Body allocations checked by Mtp::load, excluding pooled state and the
    /// independent serving selector or diagnostic input allocation.
    pub fn body_bytes(&self) -> Result<u64> {
        self.owners.iter().filter(|o| !matches!(o.id.as_str(), "draft.mtp.kv" | "draft.mtp.sidecars"
            | "draft.mtp.selector.output" | "draft.mtp.selector.landing" | "draft.mtp.diagnostic.input"))
            .try_fold(0u64, |sum, o| sum.checked_add(o.bytes as u64).ok_or_else(|| anyhow::anyhow!("MTP body total overflow")))
    }

    pub fn pool_unit_bytes(&self) -> Result<u64> {
        (self.record_bytes as u64).checked_mul(PAGE_ROWS).and_then(|n| n.checked_add(self.hidden_bytes as u64))
            .ok_or_else(|| anyhow::anyhow!("MTP page extent overflow"))
    }
}

/// Native MTP and block dSpark both require named actual owners before Ready.
#[derive(Debug, Clone, Copy)]
pub enum DenseDraft<'a> {
    NativeMtp(&'a NativeMtpLayout),
    Block(&'a crate::formats::block_dspark::BlockDraftLayout),
}

impl DenseDraft<'_> {
    pub fn sync_upload_bytes(self) -> u64 {
        match self { Self::NativeMtp(layout) => layout.sync_upload_bytes,
            Self::Block(layout) => layout.sync_upload_bytes as u64 }
    }

    pub fn page_bytes(self) -> Result<u64> {
        match self { Self::NativeMtp(layout) => layout.pool_unit_bytes(), Self::Block(layout) => Ok(layout.page_bytes()?) }
    }
}

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
        ensure!((1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots), "dense state storage must be in 1..=128 slots");
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
            (Category::Workspace, "block verify marks, logits and taps", self.verify.iter().sum())]
            .into_iter().map(|(category, group, bytes)| Demand::new(0, category, group, bytes as u64, Basis::Exact)).collect()
    }
}

/// Header-only native BF16/E4M3 allocations, including exact block scales and widened GDN operands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeWeights {
    pub allocations: Vec<NativeOperand>,
    pub layers: Vec<u64>,
    pub norm: u64,
    pub head: u64,
    pub embedding: u64,
    /// Peak serialized tensor reads plus their packed/widened output; ordinary
    /// heap only. The embedding thread uses separately charged pinned staging.
    pub loading_host_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperand { pub id: String, pub bytes: u64, pub upload_bytes: u64 }

impl NativeWeights {
    pub fn native(checkpoint: &crate::plan::checkpoint::Checkpoint, cfg: &Qwen35Config,
        norm_dtype: cuteafd_core::DType) -> Result<Self> {
        use cuteafd_core::DType;
        ensure!(matches!(norm_dtype, DType::Bf16 | DType::F32), "invalid native GDN norm dtype");
        use anyhow::Context;
        use weights::native_inventory;
        let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
            .with_context(|| format!("checkpoint lacks {name}"));
        let fp8 = tensor("model.language_model.layers.0.mlp.gate_proj.weight")?.meta.dtype == DType::F8E4M3;
        let mut allocations = Vec::new();
        let mut layers = Vec::new();
        let mut loading_host_bytes = 0;
        for (index, &kind) in cfg.attention.iter().enumerate() {
            let inventory = native_inventory(checkpoint, &format!("model.language_model.layers.{index}"), kind, fp8, norm_dtype.clone())?;
            layers.push(inventory.bytes()?);
            loading_host_bytes = loading_host_bytes.max(inventory.loading_host_bytes);
            allocations.extend(inventory.allocations.into_iter().map(|mut operand| {
                operand.id = format!("layer.{index}.{}", operand.id);
                operand
            }));
        }
        let mut one = |id: &str, name: &str| -> Result<u64> {
            let value = &tensor(name)?.meta;
            ensure!(value.dtype == DType::Bf16, "{name}: expected native BF16");
            let bytes = value.byte_length.max(256);
            allocations.push(NativeOperand { id: id.into(), bytes, upload_bytes: value.byte_length });
            loading_host_bytes = loading_host_bytes.max(value.byte_length);
            Ok(bytes)
        };
        let norm = one("norm", "model.language_model.norm.weight")?;
        let head = one("head", if cfg.tied_embeddings { "model.language_model.embed_tokens.weight" } else { "lm_head.weight" })?;
        // The embedding uploader uses its own chunked staging, not sync-H2D.
        let embedding = tensor("model.language_model.embed_tokens.weight")?.meta.byte_length;
        ensure!(tensor("model.language_model.embed_tokens.weight")?.meta.dtype == DType::Bf16, "embedding is not native BF16");
        Ok(Self { layers, allocations, norm, head, embedding, loading_host_bytes })
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
        let f32_scales = NativeWeights::native(&checkpoint, &cfg, DType::Bf16)?;
        assert_eq!(f32_scales.allocations, inventory.allocations);
        assert_eq!(f32_scales.layers, inventory.layers);
        assert!(f32_scales.loading_host_bytes >= inventory.loading_host_bytes);
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
    fn native_mtp_named_owners_follow_packing_pages_and_diagnostic_usage() -> Result<()> {
        use crate::formats::dflash::MlaDraftDomain;
        use crate::plan::checkpoint::{Checkpoint, CheckpointTensor};
        let mut cfg = cfg();
        cfg.mtp_layers = 1;
        for fp8 in [false, true] {
            let mut checkpoint = Checkpoint { snapshot: Default::default(), config: serde_json::json!({}),
                quantize_config: None, weight_map: Default::default(), tensors: vec![], missing_shards: vec![], shard_bytes: 0 };
            let mut add = |name: String, dtype, shape: Vec<usize>| {
                let width = if dtype == DType::F8E4M3 { 1 } else { 2 };
                let byte_length = shape.iter().product::<usize>() as u64 * width;
                checkpoint.tensors.push(CheckpointTensor { shard: "model.safetensors".into(),
                    meta: crate::SafetensorsTensorMetadata { name, dtype, shape, byte_offset: 0, byte_length } });
            };
            for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                add(format!("mtp.layers.0.{leaf}"), DType::Bf16, vec![128]);
            }
            for leaf in ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] {
                add(format!("mtp.layers.0.{leaf}.weight"), if fp8 { DType::F8E4M3 } else { DType::Bf16 }, vec![128, 128]);
                if fp8 { add(format!("mtp.layers.0.{leaf}.weight_scale_inv"), DType::Bf16, vec![1, 1]); }
            }
            add("mtp.fc.weight".into(), DType::Bf16, vec![cfg.hidden, 2 * cfg.hidden]);
            for name in ["norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding"] {
                add(format!("mtp.{name}.weight"), DType::Bf16, vec![cfg.hidden]);
            }
            let serving = NativeMtpLayout::new(&checkpoint, &cfg, 64, NativeMtpUsage::SERVING)?;
            let bytes = |id: &str| serving.owners.iter().find(|o| o.id == id).unwrap().bytes as u64;
            assert_eq!(serving.owners.iter().filter(|o| o.id.starts_with("draft.mtp.layer.")).map(|o| o.bytes as u64).sum::<u64>(),
                7 * 16384 * if fp8 { 1 } else { 2 } + 4 * 256 + if fp8 { 4 * 256 } else { 0 });
            assert_eq!(bytes("draft.mtp.concat"), (64 * 2 * cfg.hidden * 2) as u64);
            assert_eq!(bytes("draft.mtp.hidden"), (4096 * cfg.hidden * 2) as u64);
            assert_eq!(bytes("draft.mtp.logits"), (64 * cfg.vocab * 4) as u64);
            assert_eq!(bytes("draft.mtp.kv"), (64 * 4096) as u64);
            assert_eq!(bytes("draft.mtp.sidecars"), (cfg.hidden * 2) as u64);
            assert_eq!(serving.sync_upload_bytes, (cfg.hidden * 2 * cfg.hidden * 2) as u64);
            assert!(serving.loading_host_bytes >= serving.sync_upload_bytes);
            assert!(serving.owners.iter().all(|o| !o.id.contains("diagnostic") && !o.id.contains("embedding.weight") && !o.id.contains("head.weight")));
            assert_eq!(serving.owners.iter().filter(|o| o.domain == MlaDraftDomain::HostPinned).map(|o| o.bytes).sum::<usize>(), 12);
            let larger = NativeMtpLayout::new(&checkpoint, &cfg, 128, NativeMtpUsage::SERVING)?;
            assert_eq!(larger.device_bytes()? - serving.device_bytes()?, serving.pool_unit_bytes()?);
            for (one, two) in serving.owners.iter().zip(&larger.owners) {
                if matches!(one.id.as_str(), "draft.mtp.kv" | "draft.mtp.sidecars") { assert_eq!(two.bytes, one.bytes * 2); }
                else { assert_eq!(one, two); }
            }
            let diagnostic = NativeMtpLayout::new(&checkpoint, &cfg, 64, NativeMtpUsage::DIAGNOSTIC)?;
            assert!(!diagnostic.owners.iter().any(|o| o.id.contains("selector")));
            assert_eq!(diagnostic.device_bytes()? + 12 - serving.device_bytes()?, (64 * cfg.hidden * 2) as u64);
            assert_eq!(diagnostic.body_bytes()?, serving.body_bytes()?);
            let combined = NativeMtpLayout::new(&checkpoint, &cfg, 64, NativeMtpUsage { selector: true, diagnostic_input: true })?;
            assert_eq!(combined.device_bytes()? - serving.device_bytes()?, (64 * cfg.hidden * 2) as u64);
            assert_eq!(serving.with_pool_tokens(128)?, larger);
            {
                use crate::placement::planner::*;
                let weights = NativeWeights { allocations: (0..cfg.attention.len()).map(|layer| NativeOperand {
                    id: format!("layer.{layer}.fixture"), bytes: 256, upload_bytes: 256 }).collect(),
                    layers: vec![256; cfg.attention.len()], norm: 0, head: 0, embedding: 256, loading_host_bytes: 256 };
                let buffers = NativeBuffers::new(&cfg, 256, 4, 0, false)?;
                let budgets = forecast_budgets(&ARCH_CONTEXTS[1], 96 << 30, 0, (128 << 30, 128 << 30))?;
                let recipe = DenseFacts { cfg: &cfg, weights: &weights, buffers: &buffers, slots: 4,
                    record_bytes: 4096, drafter: Some(DenseDraft::NativeMtp(&larger)), serving: None };
                let request = Request { context: cfg.max_context as u64, pool: Pin::Exact(cfg.max_context as u64),
                    onboard: Pin::Auto, prefill: Pin::Auto, spark_reply_format: None, operation: Operation::Serving , resident_weight_limit: None,};
                let contract = recipe.prepare(&request)?;
                let selected = vec![0; contract.variables.len()];
                for pages in [1, 2, 4096, contract.maximum_pages] {
                    let facts = contract.facts(&selected, pages)?;
                    let exact = serving.with_pool_tokens(usize::try_from(pages * PAGE_ROWS)?)?;
                    for owner in &exact.owners {
                        let actual = facts.allocations.iter().find(|a| a.id == owner.id).unwrap();
                        assert_eq!(actual.bytes, owner.bytes as u64);
                        assert_eq!(actual.shape, owner.shape.iter().map(|&n| n as u64).collect::<Vec<_>>());
                        assert_eq!(actual.lifetime.first, Phase::DraftLoad);
                    }
                    assert!(facts.deferred.is_empty());
                }
                let plan = admit(request, budgets, recipe)?;
                let upload = plan.facts.allocations.iter().find(|a| a.id == "upload.sync").unwrap();
                assert_eq!(upload.bytes, serving.sync_upload_bytes);
                assert!(plan.facts.allocations.iter().all(|a| !a.id.contains("diagnostic")));
                let mut receipts: Vec<_> = plan.facts.allocations.iter().filter(|a| a.view.is_none()).enumerate().map(|(i, allocation)| Receipt {
                    allocation: allocation.clone(), address: (i as u64 + 1) * (1 << 40) }).collect();
                for allocation in plan.facts.allocations.iter().filter(|a| a.view.is_some()) {
                    let view = allocation.view.as_ref().unwrap();
                    let backing = receipts.iter().find(|r| r.allocation.id == view.backing).unwrap();
                    receipts.push(Receipt { allocation: allocation.clone(), address: backing.address + view.offset });
                }
                receipts.retain(|r| r.allocation.lifetime.contains(Phase::Ready, plan.facts.phase_order));
                reconcile(&plan, Phase::Ready, &receipts)?;
                let sidecar = receipts.iter().position(|r| r.allocation.id == "draft.mtp.sidecars").unwrap();
                receipts[sidecar].allocation.shape[0] += 1;
                assert!(reconcile(&plan, Phase::Ready, &receipts).is_err());
                receipts[sidecar].allocation.shape[0] -= 1;
                let landing = receipts.iter().position(|r| r.allocation.id == "draft.mtp.selector.landing").unwrap();
                receipts.remove(landing);
                assert!(reconcile(&plan, Phase::Ready, &receipts).is_err());
            }
            assert!(NativeMtpLayout::new(&checkpoint, &cfg, 0, NativeMtpUsage::SERVING).is_err());
            assert!(NativeMtpLayout::new(&checkpoint, &cfg, 65, NativeMtpUsage::SERVING).is_err());
            if fp8 {
                let index = checkpoint.tensors.iter().position(|t| t.meta.name.ends_with("weight_scale_inv")).unwrap();
                checkpoint.tensors[index].meta.shape = vec![2, 1];
                assert!(NativeMtpLayout::new(&checkpoint, &cfg, 64, NativeMtpUsage::SERVING).is_err());
            }
        }
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

}
