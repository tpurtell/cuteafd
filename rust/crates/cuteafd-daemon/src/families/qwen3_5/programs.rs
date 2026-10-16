//! Fail-closed checkpoint/AOT agreement, before weight or workspace admission.
use super::weights::DenseFormat;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::programs::{ProgramSpec, Programs};
use cuteafd_loader::families::qwen3_5::Qwen35Config;

pub(crate) struct DensePrograms {
    pub(crate) prefix: &'static str,
    pub(crate) format: DenseFormat,
    pub(crate) norm_dtype: DType,
    pub(crate) scratch_bytes: usize,
}

fn integer(spec: &ProgramSpec, field: &str, expected: usize) -> Result<()> {
    ensure!(spec.geometry[field].as_u64() == Some(expected as u64),
        "{}: geometry.{field} must be {expected}, found {}", spec.name, spec.geometry[field]);
    Ok(())
}

fn number(spec: &ProgramSpec, field: &str, expected: f64) -> Result<()> {
    ensure!(spec.geometry[field].as_f64() == Some(expected),
        "{}: geometry.{field} must be {expected}, found {}", spec.name, spec.geometry[field]);
    Ok(())
}

fn norm_dtype(spec: &ProgramSpec) -> Result<DType> {
    match spec.geometry["norm_dtype"].as_str() {
        Some("torch.float32") => Ok(DType::F32),
        Some("torch.bfloat16") => Ok(DType::Bf16),
        _ => anyhow::bail!("{}: missing or unsupported GDN norm dtype", spec.name),
    }
}

fn validate(cfg: &Qwen35Config, op: &str, rows: usize, spec: &ProgramSpec) -> Result<()> {
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
            ensure!(spec.geometry["output_gate"] == "silu" &&
                (spec.geometry["fp8_weights"] == false || spec.geometry["fp8_weights"] == "only"),
                "{}: dense GDN needs SiLU and native checkpoint projections", spec.name);
            if spec.geometry["fp8_weights"] == "only" {
                ensure!(spec.geometry["gate_weights"] == "bf16" && spec.geometry["gate_order"] == serde_json::json!(["b", "a"]),
                    "{}: native FP8 GDN gates must remain BF16 in b/a order", spec.name);
            }
            norm_dtype(spec)?;
        }
        "gdn_commit" => {
            integer(spec, "key_heads", cfg.gdn.key_heads)?;
            integer(spec, "value_heads", cfg.gdn.value_heads)?;
            integer(spec, "channels", cfg.gdn.conv_width())?;
            integer(spec, "replay_rows", crate::shared::qwen::REPLAY_ROWS)?;
            integer(spec, "replay_bytes", crate::shared::qwen::GdnLayout::new(cfg.gdn)?.replay_layer_bytes)?;
        }
        "attn_producer" | "full_gqa" => {
            integer(spec, "heads", cfg.heads)?;
            integer(spec, "kv_heads", cfg.kv_heads)?;
            integer(spec, "head_dim", cfg.head_dim)?;
            integer(spec, "record_bytes", 2 * cfg.kv_heads * cfg.head_dim * 2)?;
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
            ensure!(spec.geometry["fp8_weights"] == false || spec.geometry["fp8_weights"] == "only",
                "{}: expected native checkpoint output projection", spec.name);
        }
        "ffn" => integer(spec, "intermediate", cfg.intermediate)?,
        "add" => (),
        _ => anyhow::bail!("unknown dense Qwen op {op}"),
    }
    Ok(())
}

fn validate_abi(op: &str, rows: usize, spec: &ProgramSpec) -> Result<()> {
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
            _ => anyhow::bail!("unexpected native FP8 ABI {op}"),
        };
    }
    if op == "gdn" {
        if rows == crate::shared::qwen::REPLAY_ROWS { pointers.push("replay"); }
        pointers.push("scratch");
    }
    ensure!(spec.pointers == pointers, "{}: native pointer order differs from dense {op} ABI", spec.name);
    let scalars = match op {
        "gdn_commit" | "full_gqa" => 3,
        "gdn" if rows == crate::shared::qwen::REPLAY_ROWS => 2,
        "attn_o" if fp8 && rows > crate::shared::qwen::REPLAY_ROWS => 2,
        _ => 1,
    };
    ensure!(spec.scalar_kinds == vec![b'i'; scalars], "{}: dense scalar ABI differs", spec.name);
    if pointers.contains(&"scratch") {
        ensure!(spec.scratch.get("scratch").is_some_and(|&n| n > 0 && n % 1024 == 0),
            "{}: missing or unaligned scratch admission", spec.name);
    }
    Ok(())
}

impl DensePrograms {
    pub(crate) fn new(cfg: &Qwen35Config, programs: &Programs<'_>, format: DenseFormat) -> Result<Self> {
        // These are actual exported geometries, not aliases for Flash Next.
        let prefix = match (cfg.hidden, cfg.intermediate, cfg.heads, cfg.gdn.value_heads) {
            (4096, 12288, 16, 32) => "qwen35",
            (5120, 17408, 24, 48) => "qwen38",
            _ => anyhow::bail!("dense Qwen H{}/I{}/heads{}/GDN{} has no native BF16 export",
                cfg.hidden, cfg.intermediate, cfg.heads, cfg.gdn.value_heads),
        };
        ensure!(cfg.gdn.key_dim == 128 && cfg.gdn.value_dim == 128 && cfg.gdn.conv_kernel == 4,
            "dense Qwen exports require GDN128 and conv4");
        let mut scratch_bytes = 0;
        let mut norm = None;
        for (op, rows) in [("norm", 0), ("add", 0), ("gdn_commit", 0)].into_iter().chain(
            [64, 4096].into_iter().flat_map(|rows|
                ["gdn", "attn_producer", "full_gqa", "attn_o", "ffn"].map(|op| (op, rows)))) {
            let quantized = format.fp8() && matches!(op, "gdn" | "attn_producer" | "attn_o" | "ffn");
            let suffix = if quantized { "_fp8" } else { "" };
            let name = if rows == 0 { format!("{prefix}_{op}") } else { format!("{prefix}_{op}{suffix}_m{rows}") };
            let spec = programs.spec(&name)?;
            if matches!(op, "gdn" | "attn_producer" | "attn_o" | "ffn") {
                ensure!((spec.geometry["fp8_weights"] == "only") == quantized,
                    "{name}: projection format disagrees with checkpoint");
                if quantized {
                    ensure!(spec.geometry["weight_block"] == serde_json::json!([128, 128]),
                        "{name}: expected checkpoint 128x128 FP8 block scales");
                }
            }
            validate(cfg, op, rows, spec)?;
            validate_abi(op, rows, spec)?;
            if op == "gdn" {
                let dtype = norm_dtype(spec)?;
                ensure!(norm.as_ref().is_none_or(|previous| *previous == dtype), "decode/prefill GDN norm dtype differs");
                norm = Some(dtype);
            }
            for &bytes in spec.scratch.values() {
                scratch_bytes = scratch_bytes.max(usize::try_from(bytes).context("dense Qwen scratch exceeds usize")?);
            }
        }
        Ok(Self { prefix, format, norm_dtype: norm.context("dense Qwen GDN norm ABI missing")?, scratch_bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::qwen::{GdnGeometry, GdnOutputGate, HybridAttention};
    use serde_json::json;

    fn cfg() -> Qwen35Config {
        Qwen35Config { hidden: 4096, intermediate: 12288, vocab: 248320, attention: vec![HybridAttention::Gdn],
            heads: 16, kv_heads: 4, head_dim: 256, gdn: GdnGeometry { key_heads: 16, value_heads: 32,
                key_dim: 128, value_dim: 128, conv_kernel: 4, output_gate: GdnOutputGate::Silu },
            rope_dim: 64, rope_theta: 1e7, rms_norm_eps: 1e-6, max_context: 262144, mtp_layers: 0, tied_embeddings: false }
    }

    #[test]
    fn native_pointer_scalar_and_scratch_contracts_fail_before_allocation() -> Result<()> {
        let mut spec = ProgramSpec { name: "full_gqa".into(),
            pointers: ["query", "kv_cache", "positions", "page_table", "out", "scratch"]
                .map(str::to_owned).to_vec(), scalar_kinds: vec![b'i'; 3],
            scratch: [("scratch".into(), 1024)].into(), capacity_rows: 64, geometry: json!({}) };
        validate_abi("full_gqa", 64, &spec)?;
        spec.pointers.swap(0, 1);
        assert!(validate_abi("full_gqa", 64, &spec).is_err());
        spec.pointers.swap(0, 1);
        spec.scalar_kinds[1] = b'l';
        assert!(validate_abi("full_gqa", 64, &spec).is_err());
        spec.scalar_kinds[1] = b'i';
        spec.scratch.insert("scratch".into(), 1023);
        assert!(validate_abi("full_gqa", 64, &spec).is_err());
        spec.scratch.clear();
        assert!(validate_abi("full_gqa", 64, &spec).is_err());
        Ok(())
    }

    #[test]
    fn gdn_replay_layout_and_checkpoint_gate_are_an_explicit_contract() -> Result<()> {
        let cfg = cfg();
        let mut spec = ProgramSpec { name: "gdn".into(), pointers: vec![], scalar_kinds: vec![],
            scratch: Default::default(), capacity_rows: 64, geometry: json!({"hidden":4096,"max_rows":64,
                "key_heads":16,"value_heads":32,"head_dim":128,"in_width":12352,"eps":1e-6,
                "output_gate":"silu","fp8_weights":false,"norm_dtype":"torch.float32"}) };
        validate(&cfg, "gdn", 64, &spec)?;
        assert_eq!(norm_dtype(&spec)?, DType::F32);
        for (key, wrong) in [("value_heads", json!(48)), ("output_gate", json!("sigmoid")),
            ("norm_dtype", json!("torch.float16")), ("fp8_weights", json!(true)), ("in_width", json!(16480))] {
            let saved = spec.geometry[key].clone();
            spec.geometry[key] = wrong;
            assert!(validate(&cfg, "gdn", 64, &spec).is_err(), "{key}");
            spec.geometry[key] = saved;
        }
        spec.geometry = json!({"key_heads":16,"value_heads":32,"channels":8192,"replay_rows":64,
            "replay_bytes":crate::shared::qwen::GdnLayout::new(cfg.gdn)?.replay_layer_bytes});
        validate(&cfg, "gdn_commit", 0, &spec)?;
        spec.geometry["replay_bytes"] = json!(0);
        assert!(validate(&cfg, "gdn_commit", 0, &spec).is_err());
        Ok(())
    }

    #[test]
    fn native_fp8_projection_abis_preserve_scales_bf16_gates_and_prefill_threshold() -> Result<()> {
        for rows in [64, 4096] {
            for (op, pointers) in [
                ("gdn", vec!["x", "w_qkvz_fp8", "w_qkvz_scale", "w_ba", "conv_w", "a_log", "dt_bias", "norm_w",
                    "w_out_fp8", "w_out_scale", "conv_state", "state", "slots", "seq_first", "out"]),
                ("attn_producer", vec!["x", "w_in_fp8", "w_in_scale", "q_norm", "k_norm", "rope_positions", "kv_slots",
                    "kv_cache", "query", "gate", "scratch"]),
                ("attn_o", vec!["attn", "gate", "w_o_fp8", "w_o_scale", "out", "scratch"]),
                ("ffn", vec!["x", "w_gate_up_fp8", "w_gate_up_scale", "w_down_fp8", "w_down_scale", "out", "scratch"]),
            ] {
                let mut pointers: Vec<String> = pointers.into_iter().map(str::to_owned).collect();
                if op == "gdn" {
                    if rows == 64 { pointers.push("replay".into()); }
                    pointers.push("scratch".into());
                }
                let scalars = if (op == "gdn" && rows == 64) || (op == "attn_o" && rows == 4096) { 2 } else { 1 };
                let mut spec = ProgramSpec { name: format!("{op}_fp8_m{rows}"), pointers,
                    scalar_kinds: vec![b'i'; scalars], scratch: [("scratch".into(), 1024)].into(),
                    capacity_rows: rows as u32, geometry: json!({"fp8_weights":"only"}) };
                validate_abi(op, rows, &spec)?;
                spec.pointers.swap(1, 2);
                assert!(validate_abi(op, rows, &spec).is_err(), "{op} rows={rows}: reordered native operands");
                spec.pointers.swap(1, 2);
                spec.scalar_kinds.push(b'i');
                assert!(validate_abi(op, rows, &spec).is_err(), "{op} rows={rows}: wrong threshold scalar");
            }
        }
        Ok(())
    }

    #[test]
    fn stale_attention_or_norm_geometry_is_rejected_before_loading() -> Result<()> {
        let cfg = cfg();
        let mut spec = ProgramSpec { name: "full_gqa".into(), pointers: vec![], scalar_kinds: vec![],
            scratch: Default::default(), capacity_rows: 64, geometry: json!({"max_rows":64,"heads":16,"kv_heads":4,
                "head_dim":256,"record_bytes":4096,"indexed":false,"page_rows":64,"max_splits":32,
                "causal":true,"empty_rows_zero":true}) };
        validate(&cfg, "full_gqa", 64, &spec)?;
        for (key, wrong) in [("heads", json!(24)), ("record_bytes", json!(2048)), ("indexed", json!(true)),
            ("empty_rows_zero", json!(false)), ("causal", json!(false)), ("max_rows", json!(32))] {
            let saved = spec.geometry[key].clone();
            spec.geometry[key] = wrong;
            assert!(validate(&cfg, "full_gqa", 64, &spec).is_err(), "{key}");
            spec.geometry[key] = saved;
        }
        spec.geometry = json!({"hidden":4096,"eps":1e-6,"zero_centered":true});
        validate(&cfg, "norm", 0, &spec)?;
        spec.geometry["zero_centered"] = json!(false);
        assert!(validate(&cfg, "norm", 0, &spec).is_err());
        assert!(norm_dtype(&spec).is_err());
        Ok(())
    }
}
