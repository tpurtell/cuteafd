//! Validated AOT program table for Hy4-flash TP1 (`hy4f_*`) and TP2 (`hy4f2_*`).
use crate::shared::launch_grid::mla_wave_balanced_splits;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, Programs};
use cuteafd_loader::families::hy4_flash::Hy4KvCache;

#[derive(Clone, Copy)]
pub(super) struct MlaSplitPlan {
    pub dsa: i32,
    pub dsa_partial: i32,
    pub swa: i32,
}

impl MlaSplitPlan {
    pub(super) fn new(programs: &Hy4Programs<'_>, context: usize, rows: usize, sms: usize) -> Result<Self> {
        let choose = |program: &Program<'_>, is_swa: bool| -> Result<i32> {
            let spec = program.spec();
            let read = |name: &str| -> Result<usize> {
                let value = spec.geometry[name].as_u64()
                    .with_context(|| format!("{} lacks MLA geometry {name}", spec.name))?;
                Ok(usize::try_from(value)?)
            };
            let max_splits = read("num_splits")?;
            let heads = read("heads")?;
            let limit = if is_swa { 512 } else { 2048 };
            let expected_heads = if is_swa { 64 } else { 32 } / programs.tp;
            ensure!(rows <= spec.capacity_rows as usize && heads == expected_heads
                && (1..=16).contains(&max_splits) && read("max_context")? == limit,
                "{} has incompatible MLA split geometry for {rows} rows", spec.name);
            // Both decode exports use 64-record chunks and eight heads per CTA.
            // SWA's 512-token window can straddle an extra chunk.
            let chunks = if is_swa && context > limit {
                limit.div_ceil(64) + 1
            } else {
                context.min(limit).div_ceil(64)
            };
            // Plan for a single query using startup-stable context capacity.
            // Changing partitions with live width or a neighbor's history
            // would break exact verify and invalidate row-only graph keys.
            Ok(mla_wave_balanced_splits(chunks, heads / 8, sms, max_splits) as i32)
        };
        let dsa_partial = if let Some(ctx) = &programs.context {
            let spec = ctx.dsa_partial_h0.spec();
            let max_splits = usize::try_from(
                spec.geometry["num_splits"]
                    .as_u64()
                    .with_context(|| format!("{} lacks MLA geometry num_splits", spec.name))?,
            )?;
            let head_count = usize::try_from(
                spec.geometry["head_count"]
                    .as_u64()
                    .with_context(|| format!("{} lacks MLA geometry head_count", spec.name))?,
            )?;
            let heads_per_tile = usize::try_from(
                spec.geometry["heads_per_tile"]
                    .as_u64()
                    .unwrap_or(8),
            )?;
            ensure!(
                rows <= spec.capacity_rows as usize && head_count == 16 && (2..=16).contains(&max_splits),
                "{} has incompatible ContextSplit DSA partial geometry for {rows} rows",
                spec.name
            );
            // Each parity shard owns half of the 2048 selected tokens (up to 1024 -> 16 chunks of 64).
            // DenseMlaForwardKernel only writes base-2 partial_lse when active_splits >= 2.
            let chunks = context.min(2048).div_ceil(2).max(1).div_ceil(64);
            mla_wave_balanced_splits(chunks, head_count / heads_per_tile.max(1), sms, max_splits).max(2) as i32
        } else {
            2
        };
        Ok(Self {
            dsa: choose(&programs.mla_dsa_decode, false)?,
            dsa_partial,
            swa: choose(&programs.mla_swa_decode, true)?,
        })
    }
}

/// Optional 2-GPU ContextSplit DSA AOT programs (loaded when `ATTENTION_PLACEMENT=context`).
pub(crate) struct Hy4ContextPrograms<'a> {
    pub(crate) dsa_partial_h0: Program<'a>,
    pub(crate) dsa_partial_h16: Program<'a>,
    pub(crate) lse_combine2: Program<'a>,
    pub(crate) mla_gate_uv_dsa: Program<'a>,
    pub(crate) indexer_scored_topk: Program<'a>,
    pub(crate) indexer_merge_expand_owned: Program<'a>,
}

/// Resolved, ABI-validated Hy4-flash coordinator AOT programs for one GPU rank.
pub(crate) struct Hy4Programs<'a> {
    pub(crate) tp: usize,
    pub(crate) kv_format: Hy4KvCache,
    // iHC
    pub(crate) ihc_pre: Program<'a>,
    pub(crate) ihc_post: Program<'a>,
    pub(crate) ihc_post_pre: Program<'a>,
    pub(crate) hc_head: Program<'a>,
    // Engram
    pub(crate) engram_head_norm: Program<'a>,
    pub(crate) engram_gate_add: Program<'a>,
    // Absorbed MLA
    pub(crate) mla_rank_norm_pack: Program<'a>,
    pub(crate) mla_q_absorb_dsa: Program<'a>,
    pub(crate) mla_q_absorb_swa: Program<'a>,
    pub(crate) mla_sink_gate_uv_dsa: Program<'a>,
    pub(crate) mla_sink_gate_uv_swa: Program<'a>,
    pub(crate) mla_dsa_decode: Program<'a>,
    pub(crate) mla_swa_decode: Program<'a>,
    pub(crate) mla_swa_extend: Program<'a>,
    // 16-head compressed indexer
    pub(crate) indexer_query_rope: Program<'a>,
    pub(crate) indexer_pool_group: Program<'a>,
    pub(crate) indexer_score_groups: Program<'a>,
    pub(crate) indexer_topk_expand: Program<'a>,
    // Pointwise & routing
    pub(crate) rms_norm_3072: Program<'a>,
    pub(crate) rms_norm_128: Program<'a>,
    pub(crate) route_weights: Program<'a>,
    pub(crate) expert_input_quant: Program<'a>,
    pub(crate) post_silu_split_dense: Program<'a>,
    pub(crate) post_silu_split_shared: Program<'a>,
    // BF16 linear projections (original view)
    pub(crate) proj_q_a: Program<'a>,
    pub(crate) proj_kv_a: Program<'a>,
    pub(crate) proj_q_b_dsa: Program<'a>,
    pub(crate) proj_q_b_swa: Program<'a>,
    pub(crate) proj_linear_gate_dsa: Program<'a>,
    pub(crate) proj_linear_gate_swa: Program<'a>,
    pub(crate) proj_o_dsa: Program<'a>,
    pub(crate) proj_o_swa: Program<'a>,
    pub(crate) proj_indexer_wq_b: Program<'a>,
    pub(crate) proj_indexer_wk: Program<'a>,
    pub(crate) proj_indexer_weights: Program<'a>,
    pub(crate) proj_engram_value: Program<'a>,
    pub(crate) proj_engram_key: Program<'a>,
    pub(crate) proj_router: Program<'a>,
    pub(crate) proj_latent_down: Program<'a>,
    pub(crate) proj_latent_up: Program<'a>,
    pub(crate) proj_dense_gate_up: Program<'a>,
    pub(crate) proj_dense_down: Program<'a>,
    pub(crate) proj_shared_gate_up: Program<'a>,
    pub(crate) proj_shared_down: Program<'a>,
    // 2-GPU ContextSplit DSA programs (loaded when context_split == true)
    pub(crate) context: Option<Hy4ContextPrograms<'a>>,
}

impl<'a> Hy4Programs<'a> {
    pub(crate) fn load(
        programs: &'a Programs<'a>,
        tp: usize,
        kv_format: Hy4KvCache,
        context_split: bool,
    ) -> Result<Self> {
        ensure!(matches!(tp, 1 | 2), "Hy4-flash programs support TP1 or TP2, got {tp}");
        ensure!(!context_split || tp == 2, "Hy4-flash ContextSplit programs require TP2");
        let prefix = if tp == 1 { "hy4f" } else { "hy4f2" };
        let kv = match kv_format {
            Hy4KvCache::Bf16 => "bf16",
            Hy4KvCache::Fp8 => "fp8",
        };
        let load = |suffix: &str, ptrs: &[&str]| -> Result<Program<'a>> {
            let name = format!("{prefix}_{suffix}");
            programs.program(&name, ptrs)
        };
        let dsa_attn_name = format!("mla_dsa_decode_{kv}");
        let swa_dec_name = format!("mla_swa_decode_{kv}");
        let swa_ext_name = format!("mla_swa_extend_{kv}");
        let attn_ptrs = &[
            "query",
            "kv_cache",
            "page_table",
            "cache_seqlens",
            "cu_seqlens_q",
            "attn",
            "lse",
            "scratch",
        ];
        let norm_ptrs = &["x", "weight", "bias", "cos", "sin", "positions", "out"];
        let proj_ptrs = &["x", "w", "out"];

        Ok(Self {
            tp,
            kv_format,
            ihc_pre: load("ihc_pre", &["residual", "fn", "scale", "base", "norm", "post_out", "y_out"])?,
            ihc_post: load("ihc_post", &["delta", "residual", "prev_post", "out"])?,
            ihc_post_pre: load(
                "ihc_post_pre",
                &["delta", "residual", "prev_post", "fn", "scale", "base", "norm", "residual_out", "post_out", "y_out"],
            )?,
            hc_head: load("hc_head", &["residual", "fn", "scale", "base", "norm", "collapsed", "out"])?,
            engram_head_norm: load("engram_head_norm", &["raw", "weights", "out"])?,
            engram_gate_add: load(
                "engram_gate_add",
                &[
                    "residual",
                    "keys",
                    "key_bias",
                    "value",
                    "value_bias",
                    "norm_key",
                    "norm_query",
                    "pre_gamma",
                    "hc_scales",
                    "masked",
                ],
            )?,
            mla_rank_norm_pack: load(
                &format!("mla_rank_norm_pack_{kv}"),
                &["q_a", "kv_a", "q_norm", "kv_norm", "slots", "positions", "inv_freq", "q_resid", "kv_cache"],
            )?,
            mla_q_absorb_dsa: load("mla_q_absorb_dsa", &["q_raw", "w_uk", "positions", "inv_freq", "query"])?,
            mla_q_absorb_swa: load("mla_q_absorb_swa", &["q_raw", "w_uk", "positions", "inv_freq", "query"])?,
            mla_sink_gate_uv_dsa: load("mla_sink_gate_uv_dsa", &["attn", "w_uv", "gate", "lse", "sink", "out"])?,
            mla_sink_gate_uv_swa: load("mla_sink_gate_uv_swa", &["attn", "w_uv", "gate", "lse", "sink", "out"])?,
            mla_dsa_decode: load(&dsa_attn_name, attn_ptrs)?,
            mla_swa_decode: load(&swa_dec_name, attn_ptrs)?,
            mla_swa_extend: load(&swa_ext_name, attn_ptrs)?,
            indexer_query_rope: load(
                "indexer_query_rope",
                &["q_raw", "w_raw", "positions", "inv_freq", "q_out", "w_out"],
            )?,
            indexer_pool_group: load(
                "indexer_pool_group",
                &["tokens", "w_kv", "w_kg", "ape", "key_norm", "group_positions", "group_slots", "inv_freq", "group_cache"],
            )?,
            indexer_score_groups: load(
                "indexer_score_groups",
                &["q_rot", "weights", "group_cache", "page_table", "group_limits", "scores_out"],
            )?,
            indexer_topk_expand: load(
                "indexer_topk_expand",
                &["scores", "positions", "page_table", "slots_out", "lengths_out"],
            )?,
            rms_norm_3072: load("rms_norm_3072", norm_ptrs)?,
            rms_norm_128: load("rms_norm_128", norm_ptrs)?,
            route_weights: load("route_weights", &["logits", "bias", "ids", "out"])?,
            expert_input_quant: load(
                "expert_input_quant",
                &["source_ptr", "values_ptr", "scale_rows_ptr", "scale_mma_ptr"],
            )?,
            post_silu_split_dense: load("post_silu_split_dense", &["gate", "up", "out"])?,
            post_silu_split_shared: load("post_silu_split_shared", &["gate", "up", "out"])?,
            proj_q_a: load("q_a", proj_ptrs)?,
            proj_kv_a: load("kv_a", proj_ptrs)?,
            proj_q_b_dsa: load("q_b_dsa", proj_ptrs)?,
            proj_q_b_swa: load("q_b_swa", proj_ptrs)?,
            proj_linear_gate_dsa: load("linear_gate_dsa", proj_ptrs)?,
            proj_linear_gate_swa: load("linear_gate_swa", proj_ptrs)?,
            proj_o_dsa: load("o_proj_dsa", proj_ptrs)?,
            proj_o_swa: load("o_proj_swa", proj_ptrs)?,
            proj_indexer_wq_b: load("indexer_wq_b", proj_ptrs)?,
            proj_indexer_wk: load("indexer_wk", proj_ptrs)?,
            proj_indexer_weights: load("indexer_weights", proj_ptrs)?,
            proj_engram_value: load("engram_value", proj_ptrs)?,
            proj_engram_key: load("engram_key", proj_ptrs)?,
            proj_router: load("router", proj_ptrs)?,
            proj_latent_down: load("latent_down", proj_ptrs)?,
            proj_latent_up: load("latent_up", proj_ptrs)?,
            proj_dense_gate_up: load("dense_gate_up", proj_ptrs)?,
            proj_dense_down: load("dense_down", proj_ptrs)?,
            proj_shared_gate_up: load("shared_gate_up", proj_ptrs)?,
            proj_shared_down: load("shared_down", proj_ptrs)?,
            context: if context_split {
                let partial_ptrs = &[
                    "query",
                    "kv_cache",
                    "page_table",
                    "cache_seqlens",
                    "cu_seqlens_q",
                    "attn",
                    "lse",
                    "scratch",
                ];
                Some(Hy4ContextPrograms {
                    dsa_partial_h0: load(&format!("mla_dsa_partial_{kv}_h0"), partial_ptrs)?,
                    dsa_partial_h16: load(&format!("mla_dsa_partial_{kv}_h16"), partial_ptrs)?,
                    lse_combine2: load("lse_combine2", &["o0", "l0", "o1", "l1", "sink", "out"])?,
                    mla_gate_uv_dsa: load("mla_gate_uv_dsa", &["attn", "w_uv", "gate", "out"])?,
                    indexer_scored_topk: load(
                        "indexer_scored_topk",
                        &["scores", "positions", "local_limits", "out_scores", "out_indices"],
                    )?,
                    indexer_merge_expand_owned: load(
                        "indexer_merge_expand_owned",
                        &["scores", "indices", "positions", "page_table", "local_slots", "local_lengths", "draft_slots"],
                    )?,
                })
            } else {
                None
            },
        })
    }
}
