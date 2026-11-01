//! The Muse Spark coordinator programs the engine binds, by exported name and
//! checked pointer order (fork `b12x/integration/cuteafd/muse.py` and
//! `_muse_attention.py` at the pinned revision; exported by
//! `python/tools/aot/export_b12x_muse_aot.py`). `muse_*` serve 128 query heads
//! (one RTX), `muse2_*` 64 per GPU (head-split TP2); geometry-free programs
//! (norms, gate, router, softcap) exist once per prefix. K6 core projections
//! are bound by package geometry (`shared/dense_trellis.rs`), not here.
use anyhow::Result;
use cuteafd_ffi::programs::{Program, Programs};

/// The shared rotary-table builder the sliding layers' tables come from
/// (`rope_tables_bf16` at span 64; `shared/rope.rs`).
pub(crate) const ROPE_TABLES: &str = "rope_s64_plain";

/// (stem without prefix, pointer order). Head-dependent programs exist per prefix.
pub(crate) const POINTWISE: &[(&str, &[&str])] = &[
    ("embed_norm", &["x", "out"]),
    ("input_norm", &["x", "weight", "out"]),
    ("pre_ffn_norm", &["x", "weight", "out"]),
    ("pre_expert_norm", &["x", "weight", "out"]),
    ("post_attn_norm", &["x", "out"]),
    ("post_ffn_norm", &["x", "weight", "out"]),
    ("final_norm", &["x", "weight", "out"]),
    ("residual_gate", &["residual", "x", "alpha", "beta", "out"]),
    ("route_select", &["logits", "bias", "ids", "weights"]),
    ("softcap", &["logits", "out"]),
    ("router", &["x", "weight", "out"]),
    // Expert-width rows -> the shared FP8 K32 Spark wire (E 4096).
    ("expert_input_quant", &["source_ptr", "values_ptr", "scale_rows_ptr", "scale_mma_ptr"]),
    // Partials -> combined expert rows (rank sum, post-expert norm, weight, top-k sum), per reply format;
    // `rank_regions` is an i64 device table of each rank's region base (contract addendum 2026-11-01).
    ("tp_combine_bf16", &["rank_regions", "row_map", "weights", "norm", "out"]),
    ("tp_combine_nvfp4", &["rank_regions", "row_map", "weights", "norm", "out"]),
    ("tp_combine_fp8", &["rank_regions", "row_map", "weights", "norm", "out"]),
];

pub(crate) const PER_HEADS: &[(&str, &[&str])] = &[
    // q plus each row's K|V record at kv_dst + kv_slots[row] * R (contract addendum 2026-10-31).
    // cos/sin: the shared half-width table [P, 32] (contract addendum, rotary tables).
    ("qk_norm_rope_swa", &["q_raw", "k_raw", "v", "positions", "cos", "sin", "kv_slots", "q", "kv_dst"]),
    ("qk_norm_nope_full", &["q_raw", "k_raw", "v", "positions", "cos", "sin", "kv_slots", "q", "kv_dst"]),
    ("attention_output", &["attn", "gate", "out"]),
    // The shared BF16 projection exporter names its weight `w`.
    ("gate_proj", &["x", "w", "out"]),
    ("attention_full_prefill", &["q", "kv_cache", "positions", "page_table", "out", "scratch"]),
    ("attention_full_decode", &["q", "kv_cache", "positions", "page_table", "out", "scratch"]),
    ("attention_swa_prefill", &["q", "kv_step", "ring", "positions", "ring_slots", "seq_first", "out", "scratch"]),
    ("attention_swa_decode", &["q", "kv_step", "ring", "positions", "ring_slots", "seq_first", "out", "scratch"]),
    ("swa_commit", &["kv_step", "ring", "ring_slots", "seq_first"]),
];

/// The program prefix of a coordinator layout: whole heads on one RTX or a TP2 half.
pub(crate) fn prefix(ranks: usize) -> Result<&'static str> {
    match ranks {
        1 => Ok("muse"),
        2 => Ok("muse2"),
        other => anyhow::bail!("Muse Spark coordinators run one RTX or a head-split pair, not {other}"),
    }
}

/// Every program a coordinator rank binds, as (exported name, pointer order).
pub(crate) fn required(ranks: usize) -> Result<Vec<(String, &'static [&'static str])>> {
    let head = prefix(ranks)?;
    Ok(POINTWISE.iter().map(|(stem, pointers)| (format!("muse_{stem}"), *pointers))
        .chain(PER_HEADS.iter().map(|(stem, pointers)| (format!("{head}_{stem}"), *pointers)))
        .collect())
}

/// Resolve every required program on the current device: a missing program or
/// a drifted pointer order fails startup before any allocation.
pub(crate) fn bind<'a>(programs: &'a Programs<'a>, ranks: usize) -> Result<Vec<(String, Program<'a>)>> {
    required(ranks)?.into_iter().map(|(name, pointers)| {
        let program = programs.program(&name, pointers)?;
        Ok((name, program))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exported() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families/muse_spark/fixtures/exported_abi.json");
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    /// Every bound program's pointer order is the exported manifest's (the
    /// fixture comes from the pinned exporters' SM188 export).
    #[test]
    fn bound_pointers_match_the_exported_manifests() {
        let abi = exported();
        for ranks in [1, 2] {
            for (name, pointers) in required(ranks).unwrap() {
                let program = &abi["programs"][&name];
                assert!(program.is_object(), "{name} was not exported");
                let names: Vec<&str> = program["pointers"].as_array().unwrap().iter().map(|p| p.as_str().unwrap()).collect();
                assert_eq!(names, pointers, "{name} pointer order");
            }
        }
    }




    /// The binding table lists every program the pinned exporter builds, with no
    /// extras: parsed from `export_b12x_muse_aot.py` and the fork's NORMS table.
    #[test]
    fn binding_table_matches_the_exporter() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let exporter = std::fs::read_to_string(root.join("python/tools/aot/export_b12x_muse_aot.py")).unwrap();
        let muse = std::fs::read_to_string(root.join("third_party/sparkinfer/b12x/integration/cuteafd/muse.py")).unwrap();
        let norms: Vec<&str> = muse.split("NORMS = {").nth(1).unwrap().split('}').next().unwrap()
            .lines().filter_map(|line| line.trim().strip_prefix('"').and_then(|l| l.split('"').next())).collect();
        assert_eq!(norms.len(), 7, "{norms:?}");
        let mut table: Vec<&str> = POINTWISE.iter().map(|(stem, _)| *stem).collect();
        table.extend(PER_HEADS.iter().map(|(stem, _)| *stem));
        for stem in &table {
            let named = exporter.contains(&format!("{stem}=")) || exporter.contains(&format!("\"{stem}\""))
                || norms.contains(stem) || (stem.starts_with("attention_") && exporter.contains("attention_{kind}_{route}"))
                || (stem.starts_with("tp_combine_") && exporter.contains("tp_combine_{format}")
                    && exporter.contains(&format!("\"{}\"", stem.trim_start_matches("tp_combine_"))));
            assert!(named, "{stem} is not built by the exporter");
        }
        for norm in &norms { assert!(table.contains(norm), "exporter norm {norm} is not bound"); }
        assert_eq!(required(1).unwrap().len(), POINTWISE.len() + PER_HEADS.len());
        assert!(required(2).unwrap().iter().any(|(name, _)| name == "muse2_attention_swa_decode"));
        assert!(required(2).unwrap().iter().any(|(name, _)| name == "muse_route_select"));
        assert!(prefix(3).is_err());
    }
}
