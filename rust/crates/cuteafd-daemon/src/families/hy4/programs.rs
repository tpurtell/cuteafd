//! Validated AOT program table for Hy4 TP1 (`hy4_*`) and TP2 (`hy42_*`).
use crate::shared::launch_grid::mla_wave_balanced_splits;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, Programs};
use cuteafd_loader::families::hy4::Hy4KvCache;

#[derive(Clone, Copy)]
pub(super) struct MlaSplitPlan {
    pub dsa: i32,
    pub dsa_partial: i32,
}

impl MlaSplitPlan {
    pub(super) fn new(
        programs: &Hy4Programs<'_>,
        context: usize,
        rows: usize,
        sms: usize,
    ) -> Result<Self> {
        let dsa = {
            let spec = programs.mla_dsa_decode.spec();
            let read = |name: &str| -> Result<usize> {
                let value = spec.geometry[name]
                    .as_u64()
                    .with_context(|| format!("{} lacks MLA geometry {name}", spec.name))?;
                Ok(usize::try_from(value)?)
            };
            let max_splits = read("num_splits")?;
            let heads = read("heads")?;
            let expected_heads = 64 / programs.tp;
            ensure!(
                rows <= spec.capacity_rows as usize
                    && heads == expected_heads
                    && (1..=16).contains(&max_splits)
                    && read("max_context")? == 2048,
                "{} has incompatible MLA split geometry for {rows} rows",
                spec.name
            );
            let chunks = context.min(2048).div_ceil(64);
            mla_wave_balanced_splits(chunks, heads / 8, sms, max_splits) as i32
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
                spec.geometry["heads_per_tile"].as_u64().unwrap_or(8),
            )?;
            ensure!(
                rows <= spec.capacity_rows as usize
                    && head_count == 32
                    && (2..=16).contains(&max_splits),
                "{} has incompatible ContextSplit DSA partial geometry for {rows} rows",
                spec.name
            );
            let chunks = context.min(2048).div_ceil(2).max(1).div_ceil(64);
            mla_wave_balanced_splits(chunks, head_count / heads_per_tile.max(1), sms, max_splits)
                .max(2) as i32
        } else {
            2
        };
        Ok(Self { dsa, dsa_partial })
    }
}

/// Optional 2-GPU ContextSplit DSA AOT programs (loaded when `ATTENTION_PLACEMENT=context`).
pub(crate) struct Hy4ContextPrograms<'a> {
    pub(crate) dsa_partial_h0: Program<'a>,
    pub(crate) dsa_partial_h32: Program<'a>,
    pub(crate) lse_combine2: Program<'a>,
    pub(crate) mla_gate_uv_dsa: Program<'a>,
    pub(crate) scored_index_topk_decode: Program<'a>,
    pub(crate) scored_index_topk_prefill: Program<'a>,
    pub(crate) dsa_candidate_merge: Program<'a>,
    pub(crate) paged_staging_gather_kv: Program<'a>,
    pub(crate) paged_staging_gather_index: Program<'a>,
}

/// Resolved, ABI-validated Hy4 coordinator AOT programs for one GPU rank.
pub(crate) struct Hy4Programs<'a> {
    pub(crate) tp: usize,
    pub(crate) kv_format: Hy4KvCache,
    // iHC (H=6144)
    pub(crate) ihc_pre: Program<'a>,
    pub(crate) ihc_post: Program<'a>,
    pub(crate) ihc_post_pre: Program<'a>,
    pub(crate) hc_head: Program<'a>,
    // Absorbed gated DSA MLA (64H TP1 / 32H TP2, q_lora=2048, qa_kva_eps=1e-6)
    pub(crate) mla_rank_norm_pack: Program<'a>,
    pub(crate) mla_q_absorb_dsa: Program<'a>,
    pub(crate) mla_sink_gate_uv_dsa: Program<'a>,
    pub(crate) mla_dsa_decode: Program<'a>,
    // 32-head per-token DSA indexer (21 full layers only)
    pub(crate) index_post: Program<'a>,
    pub(crate) index_topk_decode: Program<'a>,
    pub(crate) index_topk_prefill: Program<'a>,
    // Pointwise & routing
    pub(crate) rms_norm_6144: Program<'a>,
    pub(crate) route_weights: Program<'a>,
    pub(crate) expert_input_quant: Program<'a>,
    pub(crate) post_silu_split_dense: Program<'a>,
    pub(crate) post_silu_split_shared: Program<'a>,
    // BF16 linear projections (original view)
    pub(crate) proj_q_a: Program<'a>,
    pub(crate) proj_kv_a: Program<'a>,
    pub(crate) proj_q_b_dsa: Program<'a>,
    pub(crate) proj_linear_gate_dsa: Program<'a>,
    pub(crate) proj_o_dsa: Program<'a>,
    pub(crate) proj_indexer_wq_b: Program<'a>,
    pub(crate) proj_indexer_wk: Program<'a>,
    pub(crate) proj_indexer_weights: Program<'a>,
    pub(crate) proj_router: Program<'a>,
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
        ensure!(matches!(tp, 1 | 2), "Hy4 programs support TP1 or TP2, got {tp}");
        ensure!(!context_split || tp == 2, "Hy4 ContextSplit programs require TP2");
        let prefix = if tp == 1 { "hy4" } else { "hy42" };
        let kv = match kv_format {
            Hy4KvCache::Bf16 => "bf16",
            Hy4KvCache::Fp8 => "fp8",
        };
        let load = |suffix: &str, ptrs: &[&str]| -> Result<Program<'a>> {
            let name = format!("{prefix}_{suffix}");
            programs.program(&name, ptrs)
        };
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
        let topk_ptrs = &[
            "q_fp8",
            "weights",
            "index_k_cache",
            "page_table",
            "cache_lengths",
            "output_indices",
            "scratch",
        ];
        let norm_ptrs = &["x", "weight", "bias", "cos", "sin", "positions", "out"];
        let proj_ptrs = &["x", "w", "out"];
        let proj_router = load("router", &["x", "w", "logits"])?;
        crate::families::hy4_common::validate_router_spec(proj_router.spec(), 6144, 256)?;

        Ok(Self {
            tp,
            kv_format,
            ihc_pre: load("ihc_pre", &["residual", "fn", "scale", "base", "norm", "post_out", "y_out"])?,
            ihc_post: load("ihc_post", &["delta", "residual", "prev_post", "out"])?,
            ihc_post_pre: load(
                "ihc_post_pre",
                &[
                    "delta",
                    "residual",
                    "prev_post",
                    "fn",
                    "scale",
                    "base",
                    "norm",
                    "residual_out",
                    "post_out",
                    "y_out",
                ],
            )?,
            hc_head: load("hc_head", &["residual", "fn", "scale", "base", "norm", "collapsed", "out"])?,
            mla_rank_norm_pack: load(
                &format!("mla_rank_norm_pack_{kv}"),
                &["q_a", "kv_a", "q_norm", "kv_norm", "slots", "positions", "inv_freq", "q_resid", "kv_cache"],
            )?,
            mla_q_absorb_dsa: load("mla_q_absorb_dsa", &["q_raw", "w_uk", "positions", "inv_freq", "query"])?,
            mla_sink_gate_uv_dsa: load(
                "mla_sink_gate_uv_dsa",
                &["attn", "w_uv", "gate", "lse", "sink", "out"],
            )?,
            mla_dsa_decode: load(&format!("mla_dsa_decode_{kv}"), attn_ptrs)?,
            index_post: load(
                "index_post",
                &[
                    "iq",
                    "k_raw",
                    "w_raw",
                    "positions",
                    "slots",
                    "inv_freq",
                    "k_weight",
                    "k_bias",
                    "q_fp8",
                    "head_weights",
                    "cache",
                ],
            )?,
            index_topk_decode: load("index_topk_decode", topk_ptrs)?,
            index_topk_prefill: load("index_topk_prefill", topk_ptrs)?,
            rms_norm_6144: load("rms_norm_6144", norm_ptrs)?,
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
            proj_linear_gate_dsa: load("linear_gate_dsa", proj_ptrs)?,
            proj_o_dsa: load("o_proj_dsa", proj_ptrs)?,
            proj_indexer_wq_b: load("indexer_wq_b", proj_ptrs)?,
            proj_indexer_wk: load("indexer_wk", proj_ptrs)?,
            proj_indexer_weights: load("indexer_weights", proj_ptrs)?,
            proj_router,
            proj_dense_gate_up: load("dense_gate_up", proj_ptrs)?,
            proj_dense_down: load("dense_down", proj_ptrs)?,
            proj_shared_gate_up: load("shared_gate_up", proj_ptrs)?,
            proj_shared_down: load("shared_down", proj_ptrs)?,
            context: if context_split {
                let scored_ptrs = &["q", "weights", "cache", "table", "lengths", "indices", "scores", "scratch"];
                let gather_ptrs = &["pool", "table", "staging"];
                Some(Hy4ContextPrograms {
                    dsa_partial_h0: load(&format!("mla_dsa_partial_{kv}_h0"), attn_ptrs)?,
                    dsa_partial_h32: load(&format!("mla_dsa_partial_{kv}_h32"), attn_ptrs)?,
                    lse_combine2: load("lse_combine2", &["o0", "l0", "o1", "l1", "sink", "out"])?,
                    mla_gate_uv_dsa: load("mla_gate_uv_dsa", &["attn", "w_uv", "gate", "out"])?,
                    scored_index_topk_decode: load("scored_index_topk_decode", scored_ptrs)?,
                    scored_index_topk_prefill: load("scored_index_topk_prefill", scored_ptrs)?,
                    dsa_candidate_merge: load(
                        "dsa_candidate_merge",
                        &[
                            "scores",
                            "indices",
                            "out_scores",
                            "out_indices",
                            "table",
                            "local_slots",
                            "local_lengths",
                        ],
                    )?,
                    paged_staging_gather_kv: load(&format!("paged_staging_gather_kv_{kv}"), gather_ptrs)?,
                    paged_staging_gather_index: load("paged_staging_gather_index", gather_ptrs)?,
                })
            } else {
                None
            },
        })
    }
}
