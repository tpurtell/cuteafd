//! GLM 5.3 Flash (glm5_next) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod dspark;
pub(crate) mod draft_binding;
mod draft_probe;
pub(crate) mod engine;
pub(crate) mod fp8;
pub(crate) mod prefix;
pub(crate) mod serve;
mod media;
mod speculate;
mod expert_rows;
mod graphs;
mod lane_check;
mod packed_check;
pub(crate) mod packing;
mod header;
pub(crate) mod head;
mod precision;
pub(crate) mod verify;
pub(crate) mod weights;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::families::glm5_flash::GlmNextConfig;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

/// What every GLM 5.3 Flash command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    /// Checkpoint snapshot (coordinator weights, config, tokenizer).
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Second GPU of a two-GPU head split: each GPU runs half the KDA and MLA heads (its
    /// KDA state) and half the dense and shared-expert intermediate; mHC, the MLA latent
    /// records and the DSA indexer are replicated; the partial sums meet over peer memory.
    /// Router, routed experts, LM head and drafter stay on --device.
    #[arg(long)]
    pub split_device: Option<i32>,
    /// Run only the first N layers (layers 0-2 are the dense ones).
    #[arg(long)]
    pub layers: Option<usize>,
    /// Longest sequence; 0 selects checkpoint full, bounded by compiled support and the admitted pool.
    #[arg(long, default_value_t = 0)]
    pub max_context: usize,
    /// Tokens the MLA record pools hold across sequences.
    #[arg(long, default_value_t = 32_768)]
    pub pool_tokens: usize,
    /// Sequences with KDA state (136 MiB each; 68 MiB with --kda-state bf16).
    #[arg(long, default_value_t = 8)]
    pub slots: usize,
    /// The prefix mark arena the command allocates on every GPU (serve: its prefix cache;
    /// golden: the --resume-at check), counted on the layout the engine serves and reserved
    /// before an automatic pool is sized, by the planned admission and by the measured one alike.
    #[arg(skip)]
    pub mark_arena: prefix::ArenaMarks,
    /// Where prefix-cache snapshots keep their KDA state marks: `arena`, a device arena of
    /// 2C + 2 marks (147.6 MB each with FP32 state) beside the KV pool, or `pool`, units of
    /// the KV pool itself (49 per mark), taken at capture and evicted (to the host tier when
    /// it is on) like any snapshot's rows, with unit 0 reserved.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_PREFIX_MARKS", default_value = "arena")]
    pub prefix_marks: prefix::PrefixMarks,
    /// The DSA index cache: `keys` keeps every token's BF16 key | gate row beside its latent
    /// record (11,804 B per token over the 11 MLA layers); `compact` keeps only the pooled keys
    /// and each sequence's open pool (at most three rows), 6,172 B per token, with the same
    /// pooled keys bit for bit. A two-GPU head split keeps `keys`.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_INDEX_CACHE", default_value = "keys")]
    pub index_cache: engine::IndexCache,
    /// Rows of one prefill lane, and of a serial prefill chunk (the programs take up to 4096).
    #[arg(long, visible_alias = "prefill-lane-rows", default_value_t = 4096)]
    pub prefill_rows: usize,
    /// The most rows of one decode or verify step: 64 (the `_m64` programs), or 128, where a step of
    /// more than 64 rows runs the wide `_m128` programs (a build with CUTEAFD_GLMF_WIDE_DECODE_ROWS=128;
    /// one GPU) and fewer rows keep the `_m64` ones, their bits and speed. A verify step then schedules
    /// up to the GPU's whole sparse MLA waves (127 rows on an RTX 5090, 128 on 188 SMs): at 16 sequences
    /// each verifies 6 or 7 drafts instead of 3. The decode workspace, the token selector and the
    /// speculative replay records grow with it, and every expert resource holds at least its rows.
    #[arg(long, env = "CUTEAFD_GLMF_DECODE_ROWS", default_value_t = engine::DECODE_ROWS,
        value_parser = parse_decode_rows)]
    pub decode_rows: usize,
    /// Lanes a Spark prefill chunk runs in, each with its own Spark transport and exchange in
    /// flight while the other lanes' GPU layers run (1 to 4): a chunk of up to lanes x
    /// --prefill-rows rows. The lanes share one set of attention temporaries.
    #[arg(long, default_value_t = engine::DEFAULT_PREFILL_LANES,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..=engine::MAX_PREFILL_LANES as u64))]
    pub prefill_lanes: usize,
    /// GPU memory (GiB) an automatic pool leaves free for growth the start-up cannot measure
    /// (lazily loaded kernels, cuBLAS, allocator rounding), when every other allocation precedes
    /// the pool (one GPU, Spark experts): the pool takes the rest, less the graph reserve and the
    /// cache state. The planner's default; 1 suits a 32 GB card.
    #[arg(long, default_value_t = 2.0)]
    pub headroom_gib: f64,
    /// Device memory (MiB) the captured decode graphs may hold on each GPU: past it the least
    /// recently launched executables are destroyed between steps and recaptured when needed.
    /// Unset: unbounded. The KV admission keeps this much free for them on every GPU, and one
    /// planned before start-up (a head split, local experts, a fixed pool) at least the planner's
    /// 1.5 GiB graph allowance (unset: the startup set's reserve, else the allowance).
    #[arg(long)]
    pub graph_budget_mib: Option<u64>,
    /// Where the KDA speculative replay records live: `own`, their own allocation (321 MB with 64
    /// decode rows), or `shared`, the prefill lanes' scratch, which no decode step reads (782 MB with
    /// the 4,096-row KDA prefill programs). A record lives only from a speculative verify to its
    /// commit, with no prefill in between; a commit that would read records a prefill overwrote
    /// fails. One GPU whose pool is sized from measured memory (an automatic pool with Spark
    /// experts, or a coordinator GPU budget), where the step workspaces precede the KV pool.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_REPLAY_RECORDS", default_value = "own")]
    pub replay_records: engine::ReplayRecords,
    /// Spark ranks in TP order (HOST:PORT,...) serving the fp8 expert family.
    #[arg(long, conflicts_with = "local_experts")]
    pub peers: Option<String>,
    /// Run the routed experts on this GPU: EXL3 checkpoints through the
    /// coordinator `exl3-glmf-k<tiers>/rtx-tp1` package, FP8 ones through the
    /// TP1 `fp8-glmf` package. `--experts-snapshot` names another checkpoint
    /// for the experts (the official FP8 one with EXL3 coordinator weights).
    #[arg(long)]
    pub local_experts: bool,
    #[arg(long)]
    pub experts_snapshot: Option<PathBuf>,
    /// TP1 FP8 package directory (default `<libdir>/fp8/fp8-glmf/tp1`).
    #[arg(long)]
    pub fp8_package: Option<PathBuf>,
    /// Diagnostic paging: keep only N local expert layers resident. By default
    /// all routed experts stay on the GPU; checkpoints that do not fit need Sparks.
    #[arg(long, requires = "local_experts")]
    pub expert_window: Option<usize>,
    /// GPU memory (GiB) kept free of local experts for step workspaces and
    /// prefix-cache state marks. Attention and recurrent pools are already allocated.
    #[arg(long, default_value_t = 12)]
    pub expert_reserve_gib: usize,
    /// Kept for launch scripts: the MLA, dense and shared-expert projections
    /// are always FP8 (their only copies): the official FP8 release's E4M3
    /// blocks with --fp8-snapshot (or native FP8 in the primary checkpoint),
    /// else 128x128 blocks quantized from BF16 at load.
    #[arg(long)]
    pub fp8_decode: bool,
    /// The official FP8 checkpoint (zai-org/GLM-5.3-Flash): the MLA, dense and
    /// shared-expert weights (E4M3 with FP32 128x128 block scales).
    #[arg(long)]
    pub fp8_snapshot: Option<PathBuf>,
    /// KDA in/out projections: the checkpoint's BF16 (off), or E4M3 quantized
    /// per row x 128-K block (row128) or per row (channel) at load, then the
    /// only resident copy: decode rows up to 16 on the FP8 GEMV, wider verify
    /// steps and prefill W8A16 (W8A8 with --fp8-prefill kda-in/kda-o).
    /// Default off (the checkpoint's BF16, one copy). row128 is faster (1 RTX + 2
    /// Sparks, with the FP8 head and drafter: C4 code 113.8 -> 131.8 tok/s) but costs
    /// 3.4 points top-1 (89.1% -> 85.7%) and doubles verify-vs-decode rounding
    /// (6-row replay check: KL 0.0056 -> 0.0121 nat), enough to flip greedy
    /// output under speculation; opt-in.
    #[arg(long, value_enum, default_value = "off")]
    pub kda_fp8: fp8::KdaFp8,
    /// KDA recurrent state storage: `f32` (default), or `bf16`: computed in FP32 and rounded to
    /// nearest even after every decode, verify and commit row (after the row's read-out, so a
    /// verify committed at k rows stores the bits of k serial steps) and where each chunked-prefill
    /// window stores it; `bf16-tile` rounds the chunked prefill after every 16-row tile instead.
    /// Half the state and prefix-mark bytes (34 layers: 136 -> 68 MiB per sequence). Runs the
    /// BF16-projection KDA programs on one GPU (with --kda-fp8 off, no --split-device).
    #[arg(long, value_enum, default_value = "f32")]
    pub kda_state: engine::KdaState,
    /// Keep FP8 KDA output partials in FP32 until the two-GPU sum.
    #[arg(long, env = "CUTEAFD_GLMF_KDA_FP32_PARTIALS", default_value_t = false)]
    pub kda_fp32_partials: bool,
    /// Split KDA output token rows after sharing BF16 heads; round each output once.
    #[arg(long, env = "CUTEAFD_GLMF_KDA_OUTPUT_SHARD", default_value_t = false, conflicts_with = "kda_fp32_partials")]
    pub kda_output_shard: bool,
    /// Expand W8A16 prefill weights once in existing scratch.
    #[arg(long, env = "CUTEAFD_GLMF_KDA_PREFILL_EXPANDED", default_value_t = false)]
    pub kda_prefill_expanded: bool,
    /// Keep only an E4M3 LM head (per row x 128-K scales, quantized at load):
    /// every logits call (target, verify, prefill, DFlash drafts) runs the FP8
    /// head program in 16-row spans; no BF16 head stays resident.
    /// Default off (BF16 head, one copy); opt-in with the FP8 KDA projections above.
    #[arg(long, default_value_t = false, num_args = 0..=1, default_missing_value = "true",
        action = clap::ArgAction::Set)]
    pub fp8_head: bool,
    /// Numerics gate only: round the KDA projections through NVFP4 (group 16,
    /// E4M3 scales) at load and run them as BF16: `rtn` (amax/6) or `search`.
    #[arg(long, hide = true)]
    pub kda_nvfp4_gate: Option<String>,
    /// Keep every prefill row's logits (glmf-golden --nll; 2.5 GiB at 4096 rows).
    #[arg(long, hide = true)]
    pub full_prefill_logits: bool,
    /// Most EXL3 expert layers resident at once (about 3 GiB each; the free
    /// memory decides first).
    #[arg(long, default_value_t = 64)]
    pub exl3_window: usize,
    /// Prefill projections that run W8A8 block-FP8 GEMMs (E4M3 activations per
    /// row and 128-K block, FP32 scales): `mla` (q_a|kv_a, q_b, o_proj) and
    /// `ffn` (dense and shared-expert MLPs) over their FP8 weights, the default
    /// (without them those run W8A16), `kda-in` / `kda-o` (the KDA in-projection
    /// and o_proj over their per-row FP8 weights; needs --kda-fp8 row128 or
    /// channel), `all`, or `none` (every FP8 weight W8A16, BF16 KDA BF16).
    #[arg(long, value_enum, value_delimiter = ',', default_value = "mla,ffn")]
    pub fp8_prefill: Vec<Fp8PrefillGroup>,
    /// Profiling only: MoE layers run the router, the expert wire rows and the
    /// shared expert; the routed experts contribute nothing.
    #[arg(long, hide = true)]
    pub skip_experts: bool,
    /// Drafter snapshot: a dSpark checkpoint
    /// (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview) or DFlash2
    /// (incoai/GLM-5.3-Flash-DFlash2), told apart by its config; taps the mHC
    /// stream mean after its target layers and drafts on this GPU.
    #[arg(long)]
    pub draft: Option<PathBuf>,
    /// Maximum members of one draft batch, independent of context slots.
    #[arg(long, default_value_t = 16)]
    pub draft_sequences: usize,
    /// Context slots: one ring per sequence under serve-glmf (--max-sequences; the draft batch
    /// then takes at most that many), max(20, draft_sequences) for glmf-golden. The target head
    /// is shared.
    #[arg(long)]
    pub draft_context_slots: Option<usize>,
    /// Explicit calibration-free E4M3 quantization of own drafter GEMMs.
    /// Unset/true: E4M3 single copy (measured faster); false keeps checkpoint BF16.
    #[arg(long, action = clap::ArgAction::Set)]
    pub draft_fp8: Option<bool>,
    /// The drafter's vocabulary head over the target's BF16 head (DFlash2 and dSpark): `exact`
    /// (default) as the target's own head: the few-row FP32 kernel up to 24 rows (one read of the
    /// head per 8 rows), the pedantic FP32 cuBLAS GEMM on CUDA cores past them (128 rows at 16
    /// sequences: 4.5 ms on an RTX 5090). `tensor`: one draft block (8 rows) as `exact`, two and
    /// more as a BF16 tensor-core GEMM with FP32 accumulation, one read of the head. Drafts only:
    /// the target verifies every proposal through its own head, which this leaves as it is. The
    /// FP8 head (--fp8-head) runs its own program either way.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_DRAFT_HEAD", default_value = "exact")]
    pub draft_head: crate::families::glm5::DraftHead,
    /// The FP8 drafter's GEMMs (DFlash2 and dSpark): `w8a16` (default: BF16 activations, exact
    /// in f16, on the W8A16 GEMV in passes of 64 rows), `wide` (the same bits in passes of 128
    /// rows: one read of the weights at 16 sequences), or `w8a8`: one draft block (8 rows) as
    /// `w8a16`, more as E4M3 activations per row and 128-wide K block (amax / 448) on FP8 tensor
    /// cores, half the MMAs, in passes of 128 rows. Drafts only: the target verifies every
    /// proposal. The BF16 drafter (--draft-fp8 false) ignores it.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_DRAFT_LINEAR", default_value = "w8a16")]
    pub draft_linear: crate::shared::fp8_linear::Fp8Rows,
    /// Scale rule of the FP8 copies made from BF16 weights at load (KDA
    /// projections, LM head, drafter): amax / 448, the smallest power of two
    /// >= it (pow2), or per block whichever of the two leaves the smaller
    /// error (best). pow2: 84-86% of the KDA q/k/v/o weights have at most
    /// E4M3's 3 mantissa bits and quantize exactly (relative RMS 2.4e-2 ->
    /// 4.5e-4); the decode-path gate passed (NLL 1.1443 -> 1.1429, KL 0.0443
    /// -> 0.0386).
    #[arg(long, value_enum, default_value_t = crate::shared::fp8_linear::Fp8Scales::Pow2)]
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    #[command(flatten)]
    pub l2: crate::shared::l2_prefetch::L2PrefetchArgs,
    #[command(flatten)]
    pub token_io: crate::shared::token_io::TokenIoArgs,
    /// Serving-only policy used to reserve the complete graph set before KV allocation.
    #[arg(skip)]
    pub serving_graph_policy: Option<(usize, bool)>,
}

/// `--decode-rows`: the decode programs' 64 rows, or the wide programs' 128.
fn parse_decode_rows(value: &str) -> std::result::Result<usize, String> {
    match value.parse::<usize>() {
        Ok(rows) if rows == engine::DECODE_ROWS || rows == engine::WIDE_DECODE_ROWS => Ok(rows),
        _ => Err(format!("{} or {}", engine::DECODE_ROWS, engine::WIDE_DECODE_ROWS)),
    }
}

/// The wide `_m128` programs a launch with `--decode-rows 128` runs (none with 64): every decode
/// program its steps launch, at 128 rows, for its KDA state, KDA projections and index cache, and the
/// replay commit over 128-row records. Start-up names the first one the build lacks.
pub(crate) fn wide_decode_programs(args: &EngineArgs, cfg: &GlmNextConfig) -> Vec<String> {
    if args.decode_rows <= engine::DECODE_ROWS {
        return Vec::new();
    }
    let cap = format!("m{}", engine::WIDE_DECODE_ROWS);
    let compact = args.index_cache == engine::IndexCache::Compact;
    let mut names: Vec<String> = ["mhc_post_pre".to_string(), "mla_producer".into(), "o".into(),
        "sparse_mla_decode".into(), "index_producer".into(), "index_topk_decode".into(),
        format!("ffn_i{}", cfg.moe_intermediate), format!("ffn_i{}", cfg.dense_intermediate)]
        .into_iter().map(|name| format!("glmf_{name}_{cap}")).collect();
    names.push(format!("glmf_{}", args.kda_state.program(&cap)));
    if args.kda_fp8 != fp8::KdaFp8::Off {
        names.push(format!("glmf_kda_w8_{cap}"));
    }
    if compact {
        names.push(format!("glmf_index_producer_c_{cap}"));
    }
    let commit = if compact { args.kda_state.compact_commit_program() } else { args.kda_state.commit_program() };
    names.push(format!("glmf_{commit}_{cap}"));
    names
}

/// The token selector and GPU sampler of `decode_rows`-row steps beyond the 64-row ones (none at 64):
/// what `--decode-rows 128` adds where the serving loop makes the selector after the pool.
fn wide_selector_bytes(decode_rows: usize, vocab: usize) -> u64 {
    use cuteafd_loader::serving_capacity::glmf_selector_bytes;
    glmf_selector_bytes(decode_rows as u64, vocab as u64)
        .saturating_sub(glmf_selector_bytes(engine::DECODE_ROWS as u64, vocab as u64))
}

#[cfg(test)]
mod draft_cli_tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Parse {
        #[command(flatten)]
        engine: EngineArgs,
    }

    #[test]
    fn index_cache_defaults_to_keys_and_takes_compact() {
        let parse_cache = |extra: &[&str]| Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib",
            "/native"].into_iter().chain(extra.iter().copied())).map(|p| p.engine.index_cache);
        assert_eq!(parse_cache(&[]).unwrap(), engine::IndexCache::Keys);
        assert_eq!(parse_cache(&["--index-cache", "compact"]).unwrap(), engine::IndexCache::Compact);
        assert_eq!(parse_cache(&["--index-cache", "keys"]).unwrap(), engine::IndexCache::Keys);
        assert!(parse_cache(&["--index-cache", "tails"]).is_err());
    }

    /// Start-up loads GLM Flash's own programs, the head split's shares with a second GPU, the wide
    /// `_m128` decode programs only with `--decode-rows 128`, nothing of another family: the engine
    /// resolves programs only as `glmf_`/`glmf2_` (`run_on`) and the FP8 head's `glmf_head_fp8`.
    #[test]
    fn startup_loads_only_the_programs_a_glm_flash_engine_launches() {
        let names = ["glmf_kda_m64", "glmf_kda_s16_m4096", "glmf_kda_commit_c_s16", "glmf_index_producer_c_m64",
            "glmf_head_fp8", "glmf2_kda_m64", "glmf2_join_rows", "dsv4f_attention_m64", "dsv4p_compressor_m4096",
            "glm_mla_m64", "mimo_attention_m64", "qwen4_gdn_m64", "glmf_kda_m128", "glmf_kda_commit_c_s16_m128"];
        let kept = |split: bool, wide: bool| -> Vec<&str> {
            names.iter().copied().filter(|n| glmf_startup_program(n, split, wide)).collect()
        };
        assert_eq!(kept(false, false), ["glmf_kda_m64", "glmf_kda_s16_m4096", "glmf_kda_commit_c_s16",
            "glmf_index_producer_c_m64", "glmf_head_fp8"]);
        assert_eq!(kept(true, false), ["glmf_kda_m64", "glmf_kda_s16_m4096", "glmf_kda_commit_c_s16",
            "glmf_index_producer_c_m64", "glmf_head_fp8", "glmf2_kda_m64", "glmf2_join_rows"]);
        // `--decode-rows 128` (one GPU) adds the wide programs, which no step of up to 64 rows launches.
        assert_eq!(kept(false, true), ["glmf_kda_m64", "glmf_kda_s16_m4096", "glmf_kda_commit_c_s16",
            "glmf_index_producer_c_m64", "glmf_head_fp8", "glmf_kda_m128", "glmf_kda_commit_c_s16_m128"]);
        assert_eq!(format!("_m{}", engine::WIDE_DECODE_ROWS), "_m128");
        // Every program lookup the engine makes: `run_on`'s prefix and the FP8 head (re-check the
        // predicate if another appears).
        let engine = include_str!("engine.rs");
        let engine = &engine[..engine.find("\n#[cfg(test)]\n").unwrap()];
        assert_eq!(engine.matches("programs.program(").count(), 1);
        assert!(engine.contains("let name = format!(\"{}_{name}\", if split { \"glmf2\" } else { \"glmf\" });"));
        let head = include_str!("head.rs");
        assert_eq!((head.matches("programs.program(").count(), head.matches("programs.program(\"glmf_head_fp8\"").count()),
            (1, 1));
    }

    #[test]
    fn drafter_defaults_are_checkpoint_bf16_with_independent_capacities() {
        let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native"])
            .unwrap().engine;
        assert_eq!(parsed.draft_fp8, None);
        assert_eq!(parsed.draft_sequences, 16);
        assert_eq!(parsed.draft_context_slots.unwrap_or(20.max(parsed.draft_sequences)), 20);
    }

    #[test]
    fn explicit_fp8_option_and_context_batch_limits_are_forwarded() {
        for (value, expected) in [("true", true), ("false", false)] {
            let parsed = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--draft-fp8", value, "--draft-context-slots", "20", "--draft-sequences", "16"])
                .unwrap().engine;
            assert_eq!(parsed.draft_fp8, Some(expected));
            assert_eq!((parsed.draft_context_slots, parsed.draft_sequences), (Some(20), 16));
        }
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--draft-fp8", "auto"]).is_err());
    }
    fn parse(extra: &[&str]) -> EngineArgs {
        Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native"].into_iter()
            .chain(extra.iter().copied())).unwrap().engine
    }

    /// `--draft-linear`: w8a16 by default (today's W8A16 GEMV), wide or w8a8 on request.
    #[test]
    fn draft_linear_defaults_to_w8a16_and_takes_wide_and_w8a8() {
        use crate::shared::fp8_linear::Fp8Rows;
        assert_eq!(parse(&[]).draft_linear, Fp8Rows::W8a16);
        for (value, mode) in [("w8a16", Fp8Rows::W8a16), ("wide", Fp8Rows::Wide), ("w8a8", Fp8Rows::W8a8)] {
            assert_eq!(parse(&["--draft-linear", value]).draft_linear, mode);
        }
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--draft-linear", "w4a16"]).is_err());
    }

    /// `--draft-head`: exact by default (the target's own head route), tensor on request.
    #[test]
    fn draft_head_defaults_to_exact_and_takes_tensor() {
        use crate::families::glm5::DraftHead;
        assert_eq!(parse(&[]).draft_head, DraftHead::Exact);
        assert_eq!(parse(&["--draft-head", "exact"]).draft_head, DraftHead::Exact);
        assert_eq!(parse(&["--draft-head", "tensor"]).draft_head, DraftHead::Tensor);
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--draft-head", "fp8"]).is_err());
    }

    #[test]
    fn single_copy_fp8_options_are_accepted() {
        for extra in [&[][..], &["--kda-fp8", "row128"][..], &["--kda-fp8", "channel"][..], &["--fp8-head"][..],
            &["--kda-fp8", "row128", "--fp8-prefill", "kda-in,kda-o"][..], &["--fp8-prefill", "all"][..],
            &["--kda-fp8", "channel", "--fp8-prefill", "all", "--fp8-head"][..], &["--fp8-prefill", "none"][..]] {
            check_options(&parse(extra)).unwrap();
        }
    }

    #[test]
    fn kda_w8a8_prefill_needs_fp8_kda_weights() {
        let defaults = parse(&[]);
        assert_eq!((defaults.kda_fp8, defaults.fp8_head, defaults.draft_fp8), (fp8::KdaFp8::Off, false, None));
        check_options(&parse(&["--kda-fp8", "row128", "--fp8-prefill", "kda-in"])).unwrap();
        for extra in [&["--kda-fp8", "off", "--fp8-prefill", "kda-in"][..],
            &["--kda-fp8", "off", "--fp8-prefill", "mla,kda-o"][..], &["--kda-fp8", "off", "--fp8-prefill", "kda-o"][..]] {
            let error = check_options(&parse(extra)).unwrap_err().to_string();
            assert!(error.contains("--kda-fp8 row128 or channel"), "{error}");
        }
        assert!(check_options(&parse(&["--fp8-prefill", "none,mla"])).is_err());
        assert!(check_options(&parse(&["--kda-fp8", "row128", "--kda-nvfp4-gate", "rtn"])).is_err());
    }

    /// `--replay-records shared` keeps the records in one GPU's prefill scratch.
    #[test]
    fn replay_records_take_shared_on_one_gpu() {
        assert_eq!(parse(&[]).replay_records, engine::ReplayRecords::Own);
        assert_eq!(parse(&["--replay-records", "shared"]).replay_records, engine::ReplayRecords::Shared);
        check_options(&parse(&["--replay-records", "shared"])).unwrap();
        check_options(&parse(&["--replay-records", "own", "--split-device", "1"])).unwrap();
        let error = check_options(&parse(&["--replay-records", "shared", "--split-device", "1"])).unwrap_err()
            .to_string();
        assert!(error.contains("--replay-records shared") && error.contains("--split-device"), "{error}");
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--replay-records", "host"]).is_err());
    }

    #[test]
    fn bf16_kda_state_runs_the_bf16_projection_programs_on_one_gpu() {
        assert_eq!(parse(&[]).kda_state, engine::KdaState::F32);
        for (value, state) in [("f32", engine::KdaState::F32), ("bf16", engine::KdaState::Bf16),
            ("bf16-tile", engine::KdaState::Bf16Tile)] {
            let args = parse(&["--kda-state", value]);
            assert_eq!((args.kda_state, args.kda_state.name()), (state, value));
            check_options(&args).unwrap();
        }
        for extra in [&["--kda-state", "bf16", "--kda-fp8", "row128"][..],
            &["--kda-state", "bf16-tile", "--kda-fp8", "channel"][..], &["--kda-state", "bf16", "--split-device", "1"][..]] {
            let error = check_options(&parse(extra)).unwrap_err().to_string();
            assert!(error.contains("--kda-state bf16"), "{error}");
        }
        check_options(&parse(&["--kda-state", "f32", "--kda-fp8", "row128", "--split-device", "1"])).unwrap();
        // With the compact index cache too: its commit then rebuilds the index tails over the BF16 state.
        check_options(&parse(&["--kda-state", "bf16", "--index-cache", "compact"])).unwrap();
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--kda-state", "fp16"]).is_err());
    }

    /// `--decode-rows`: 64 by default (the `_m64` programs, nothing changes), 128 with the wide programs
    /// on one GPU; a head split is refused before any checkpoint or native work.
    #[test]
    fn decode_rows_default_to_64_and_take_128_on_one_gpu() {
        assert_eq!(parse(&[]).decode_rows, engine::DECODE_ROWS);
        let wide = parse(&["--decode-rows", "128"]);
        assert_eq!(wide.decode_rows, engine::WIDE_DECODE_ROWS);
        check_options(&wide).unwrap();
        check_options(&parse(&["--decode-rows", "128", "--index-cache", "compact", "--kda-state", "bf16"])).unwrap();
        for rows in ["0", "32", "65", "127", "256", "wide"] {
            assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--decode-rows", rows]).is_err(), "{rows}");
        }
        let error = check_options(&parse(&["--decode-rows", "128", "--split-device", "1"])).unwrap_err().to_string();
        assert!(error.contains("--decode-rows 64"), "{error}");
        check_options(&parse(&["--decode-rows", "64", "--split-device", "1"])).unwrap();
    }

    /// The wide programs a launch needs follow its KDA state, KDA projections and index cache; 64 rows
    /// need none.
    #[test]
    fn wide_launches_name_the_wide_programs_of_their_configuration() {
        let cfg = GlmNextConfig { vocab_size: 154880, hidden: 4096, layers: 45,
            attention: vec![cuteafd_loader::families::glm5_flash::GlmNextAttention::Kda; 45], dense: vec![false; 45],
            dense_intermediate: 12288, experts: 288, topk: 8, moe_intermediate: 2048, routed_scale: 2.5,
            swiglu_limit: 10.0, rms_norm_eps: 1e-5, hc_mult: 4, kda_heads: 64, kda_head_dim: 128, heads: 64,
            q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_dim: 256, v_head_dim: 256, index_topk: 2048,
            index_kpool: 4, eos: vec![] };
        assert!(wide_decode_programs(&parse(&[]), &cfg).is_empty());
        let base = ["glmf_mhc_post_pre_m128", "glmf_mla_producer_m128", "glmf_o_m128", "glmf_sparse_mla_decode_m128",
            "glmf_index_producer_m128", "glmf_index_topk_decode_m128", "glmf_ffn_i2048_m128", "glmf_ffn_i12288_m128"];
        let with = |extra: &[&str]| base.iter().chain(extra).map(|name| name.to_string()).collect::<Vec<_>>();
        assert_eq!(wide_decode_programs(&parse(&["--decode-rows", "128"]), &cfg),
            with(&["glmf_kda_m128", "glmf_kda_commit_m128"]));
        assert_eq!(wide_decode_programs(&parse(&["--decode-rows", "128", "--kda-fp8", "row128"]), &cfg),
            with(&["glmf_kda_m128", "glmf_kda_w8_m128", "glmf_kda_commit_m128"]));
        assert_eq!(wide_decode_programs(&parse(&["--decode-rows", "128", "--kda-state", "bf16", "--index-cache",
            "compact"]), &cfg), with(&["glmf_kda_s16_m128", "glmf_index_producer_c_m128", "glmf_kda_commit_c_s16_m128"]));
        // Every name is one of the exporter's 16 wide stems.
        let stems = ["index_producer", "index_producer_c", "index_topk_decode", "mhc_post_pre", "kda", "kda_w8",
            "kda_s16", "mla_producer", "o", "sparse_mla_decode", "ffn_i2048", "ffn_i12288", "kda_commit",
            "kda_commit_s16", "kda_commit_c", "kda_commit_c_s16"].map(|stem| format!("glmf_{stem}_m128"));
        for flags in [&["--kda-fp8", "row128", "--index-cache", "compact"][..], &["--kda-state", "bf16-tile"][..]] {
            let args = parse(&[&["--decode-rows", "128"][..], flags].concat());
            assert!(wide_decode_programs(&args, &cfg).iter().all(|name| stems.contains(name)), "{flags:?}");
        }
        // Start-up loads all 16 with 128 rows and none with 64 (`glmf_startup_program`).
        for name in &stems {
            assert!(glmf_startup_program(name, false, true) && !glmf_startup_program(name, false, false), "{name}");
        }
    }

    /// The selector and sampler `--decode-rows 128` adds where the serving loop makes them after the pool:
    /// the loader's formula (which the planner charges) is the daemon's allocation, 3,209,984 B over 64 rows.
    #[test]
    fn the_wide_selector_reserve_is_what_the_selector_allocates() {
        use crate::shared::sampler::TargetSamplingWave;
        for rows in [engine::DECODE_ROWS, engine::WIDE_DECODE_ROWS] {
            assert_eq!(cuteafd_loader::serving_capacity::glmf_selector_bytes(rows as u64, 154_880),
                (rows * 12 + TargetSamplingWave::device_bytes(rows.min(128), 154_880)) as u64);
        }
        assert_eq!(wide_selector_bytes(engine::DECODE_ROWS, 154_880), 0);
        assert_eq!(wide_selector_bytes(engine::WIDE_DECODE_ROWS, 154_880), 3_209_984);
    }

    /// `--prefill-rows 64 --decode-rows 128`: a 16-stream verify wave of long drafts fills the GPU's
    /// verify budget (127 rows on 170 SMs, 128 on 188) and runs every real row, more than the prefill
    /// lane's 64, through the experts; every expert resource holds the widest step's rows, and a
    /// planned admission charges the Spark intake planes' rows past the lane's.
    #[test]
    fn experts_hold_a_verify_wave_wider_than_the_prefill_lane() {
        for (flags, rows) in [(&[][..], 4096), (&["--decode-rows", "128"][..], 4096),
            (&["--prefill-rows", "64"][..], 64), (&["--prefill-rows", "64", "--decode-rows", "128"][..], 128),
            (&["--prefill-rows", "2048", "--decode-rows", "128"][..], 2048), (&["--prefill-rows", "32"][..], 64)] {
            assert_eq!(parse(flags).expert_rows(), rows, "{flags:?}");
        }
        let args = parse(&["--prefill-rows", "64", "--decode-rows", "128"]);
        check_options(&args).unwrap();
        for (sms, budget) in [(170, 127), (188, 128)] {
            let verify_rows = engine::verify_budget(args.decode_rows, sms);
            assert_eq!(verify_rows, budget);
            // The serving loop's limits at 16 sequences that can all draft past the even share.
            let sequences = 16;
            let room = (verify_rows / sequences).max(1) - 1;
            let mut limits = vec![room; sequences];
            engine::hand_out_remainder(&mut limits, room, verify_rows, |_| true);
            let real: usize = limits.iter().map(|limit| limit + 1).sum();
            assert_eq!(real, verify_rows);
            // Padded into its bucket, the step's real rows run the router, the expert wire and the
            // routed experts.
            let bucket = engine::DecodeBuckets::new(verify_rows).bucket(real, true);
            let mut experts = None;
            crate::shared::decode_graph::real_row_moe(real, bucket, |rows| { experts = Some(rows); Ok(()) },
                |_| Ok(())).unwrap();
            let rows = experts.unwrap();
            assert!(rows > args.prefill_rows && rows <= args.expert_rows(), "{rows} rows at {sms} SMs");
        }
        // Two lanes' transports, each with a plane per Spark (4 ranks) of 64 more rows of 4,096 BF16.
        assert_eq!(wide_intake_bytes(&args, 4, 4096), 2 * 4 * 64 * 4096 * 2);
        assert_eq!(wide_intake_bytes(&parse(&["--prefill-rows", "64", "--decode-rows", "128", "--prefill-lanes", "4"]),
            4, 4096), 4 * 4 * 64 * 4096 * 2);
        for flags in [&[][..], &["--decode-rows", "128"][..], &["--prefill-rows", "64"][..],
            &["--prefill-rows", "128", "--decode-rows", "128"][..]] {
            assert_eq!(wide_intake_bytes(&parse(flags), 4, 4096), 0, "{flags:?}");
        }
    }

    /// The dense NVFP4 package, local FP8 and EXL3 experts and every Spark transport are sized by
    /// `expert_rows`, never by the prefill lane's rows alone.
    #[test]
    fn every_expert_resource_takes_the_expert_rows() {
        let source = include_str!("mod.rs");
        // The needles are joined here, so that this test's own text does not match them.
        let body = |name: &str| {
            let start = source.find(&["    fn ", name, "<'s>(&'s self, args: &EngineArgs"].concat()).unwrap();
            let rest = &source[start..];
            &rest[..rest.find("\n    }\n").unwrap()]
        };
        let (dense, experts) = (body("load_dense"), body("experts"));
        let sized = ["args.expert", "_rows()"].concat();
        assert!(dense.contains(&["DenseNvfp4::load(&self.library, &directory, &self.cfg, ", &sized, ")?"].concat()));
        // FP8 experts, EXL3 experts and the Spark transports, each with the rows among its arguments.
        for site in ["Fp8Experts::load(", "max_rows: ", "SparkLink::new("] {
            let at = experts.find(site).unwrap_or_else(|| panic!("{site}"));
            let arguments: String = experts[at..].chars().take(200).collect();
            assert!(arguments.contains(&sized), "{site}");
        }
        assert_eq!(experts.matches(&sized).count(), 4);
        let lane = ["args.prefill", "_rows"].concat();
        assert!(!dense.contains(&lane) && !experts.contains(&lane));
    }

    #[test]
    fn precise_kda_partials_need_compatible_split_programs() {
        let defaults = parse(&[]);
        assert!(!defaults.kda_fp32_partials && !defaults.kda_output_shard && !defaults.kda_prefill_expanded);
        for option in ["--kda-fp32-partials", "--kda-output-shard", "--kda-prefill-expanded"] {
            assert!(check_options(&parse(&[option])).is_err());
            assert!(check_options(&parse(&["--kda-fp8", "row128", option])).is_err());
            check_options(&parse(&["--split-device", "1", "--kda-fp8", "row128", option])).unwrap();
        }
        for option in ["--kda-fp32-partials", "--kda-output-shard"] {
            for group in ["kda-o", "all"] {
                assert!(check_options(&parse(&["--split-device", "1", "--kda-fp8", "row128",
                    option, "--fp8-prefill", group])).is_err());
            }
        }
        check_options(&parse(&["--split-device", "1", "--kda-fp8", "row128", "--kda-fp32-partials",
            "--kda-prefill-expanded", "--fp8-prefill", "kda-in"])).unwrap();
        assert_eq!(engine::fp32_partial_reserve(2, 4096, 4096), 369_623_040);
        assert_eq!(engine::fp32_partial_reserve(2, 32, 4096), 6_291_456);
        assert_eq!(engine::output_shard_reserve(2, 4096, 4096), 134_217_728);
        assert_eq!(engine::output_shard_reserve(2, 32, 4096), 2_097_152);
        // Four lanes of half the rows hold the same partial rows in flight.
        assert_eq!(engine::output_shard_reserve(4, 2048, 4096), engine::output_shard_reserve(2, 4096, 4096));
    }

    #[test]
    fn measured_admission_keeps_the_planners_headroom_and_graph_allowance() {
        let defaults = parse(&[]);
        assert_eq!(defaults.headroom_bytes().unwrap(),
            cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes);
        assert_eq!(parse(&["--headroom-gib", "1"]).headroom_bytes().unwrap(), 1 << 30);
        assert!(parse(&["--headroom-gib=-1"]).headroom_bytes().is_err());
        assert_eq!(graph_reserve(&defaults), 1_610_612_736);
        let budgeted = parse(&["--graph-budget-mib", "512"]);
        assert_eq!((budgeted.graph_budget_mib, graph_reserve(&budgeted)), (Some(512), 512 << 20));
        // A graph budget captures lazily within it: never at startup, whatever the environment.
        assert!(!budgeted.startup_graphs());
        assert_eq!(defaults.startup_graphs(), engine::startup_graphs_enabled());
    }

    const SPARKS: [&str; 2] = ["--peers", "127.0.0.1:19441"];

    /// `flags` with `--graph-budget-mib budget` when there is one.
    fn with_budget(flags: &[&str], budget: Option<&str>) -> EngineArgs {
        let mut flags = flags.to_vec();
        if let Some(mib) = budget {
            flags.extend(["--graph-budget-mib", mib]);
        }
        parse(&flags)
    }

    /// A fixed pool is admitted from the planner's costs, which kept only the 1.5 GiB graph allowance
    /// on the GPU whatever the graph budget: at `--graph-budget-mib 4096`, 2,684,354,560 B short of what
    /// lazily captured graphs may fill. It now keeps the budget or the allowance, whichever is larger,
    /// with Spark or local experts, served or not (without a serving loop it was not admitted at all).
    #[test]
    fn a_fixed_pool_keeps_the_graph_budget_or_the_allowance_free() {
        let allowance = cuteafd_loader::plan::layout::family_costs("glm5_flash").graph_bytes;
        assert_eq!(allowance, [1_610_612_736; 3]);
        for (budget, excess) in [("4096", 2_684_354_560), ("512", 0), ("1536", 0), ("1537", 1 << 20)] {
            for experts in [&SPARKS[..], &["--local-experts"][..]] {
                let mut args = with_budget(&[experts, &["--pool-tokens", "131072"][..]].concat(), Some(budget));
                // The GPU's graph bytes: the planner's allowance (role 0) and the extra each GPU's reserve takes.
                let extra = planned_graph_extra(&args, None, allowance[0]);
                assert_eq!(extra, excess, "{budget} MiB");
                assert_eq!(allowance[0] + extra, (budget.parse::<u64>().unwrap() << 20).max(allowance[0]));
                for serving in [Some((16, true)), None] {
                    args.serving_graph_policy = serving;
                    assert_eq!(kv_admission(&args, false, false), KvAdmission::Planned, "{budget} MiB {serving:?}");
                }
            }
        }
        // Without a budget: served, planned; otherwise planned beside the startup set, else unchecked.
        let mut lazy = with_budget(&[&SPARKS[..], &["--pool-tokens", "131072"][..]].concat(), None);
        assert_eq!(kv_admission(&lazy, false, false),
            if lazy.startup_graphs() { KvAdmission::Planned } else { KvAdmission::Unchecked });
        lazy.serving_graph_policy = Some((16, true));
        assert_eq!(kv_admission(&lazy, false, false), KvAdmission::Planned);
    }

    /// A head split is admitted from the planner's costs, its pool automatic or fixed, and each rank's
    /// graph cache may fill the budget (`GlmfEngine::set_graph_budget`): both GPUs keep the budget or
    /// the allowance, whichever is larger.
    #[test]
    fn a_head_split_keeps_the_graph_budget_or_the_allowance_free_on_both_gpus() {
        let allowance = cuteafd_loader::plan::layout::family_costs("glm5_flash").graph_bytes;
        for (budget, excess) in [("4096", 2_684_354_560u64), ("512", 0), ("1536", 0)] {
            for pool in ["0", "131072"] {
                let args = with_budget(&[&SPARKS[..], &["--split-device", "1", "--pool-tokens", pool][..]].concat(),
                    Some(budget));
                // A coordinator GPU budget measures one GPU only.
                for gpu_budget in [false, true] {
                    assert_eq!(kv_admission(&args, gpu_budget, true), KvAdmission::Planned,
                        "{budget} MiB, pool {pool}");
                }
                // Both GPUs (the planner's roles 1 and 2) keep their allowance and the same extra.
                let extra = planned_graph_extra(&args, None, allowance[0]);
                assert_eq!(extra, excess, "{budget} MiB, pool {pool}");
                for role in [1, 2] {
                    assert_eq!(allowance[role] + extra, (budget.parse::<u64>().unwrap() << 20).max(allowance[role]));
                }
            }
        }
    }

    /// The default path is unchanged: one GPU with Spark experts and an automatic pool, served or not,
    /// or any one-GPU launch under a coordinator GPU budget, is admitted from measured free memory,
    /// which keeps the graph budget itself, below the allowance or above it (else the allowance).
    #[test]
    fn the_measured_admission_keeps_the_graph_budget_itself() {
        let allowance = cuteafd_loader::plan::layout::family_costs("glm5_flash").graph_bytes[0];
        for (budget, kept) in [(None, allowance), (Some("512"), 536_870_912), (Some("4096"), 4_294_967_296)] {
            let mut args = with_budget(&[&SPARKS[..], &["--pool-tokens", "0"][..]].concat(), budget);
            for serving in [Some((16, true)), None] {
                args.serving_graph_policy = serving;
                assert_eq!(kv_admission(&args, false, false), KvAdmission::Measured, "{budget:?}");
                assert_eq!(kv_admission(&args, true, false), KvAdmission::Measured, "{budget:?}");
            }
            assert_eq!(graph_reserve(&args), kept, "{budget:?}");
            // A fixed pool or local experts: measured under a GPU budget, planned without one.
            for flags in [&[&SPARKS[..], &["--pool-tokens", "131072"][..]].concat(), &vec!["--local-experts"]] {
                let mut other = with_budget(flags, budget);
                other.serving_graph_policy = Some((16, true));
                assert_eq!(kv_admission(&other, true, false), KvAdmission::Measured, "{budget:?} {flags:?}");
                assert_eq!(kv_admission(&other, false, false), KvAdmission::Planned, "{budget:?} {flags:?}");
            }
        }
        // The startup set's planned extra is unchanged: its largest rank above the allowance.
        let lazy = with_budget(&SPARKS, None);
        assert_eq!(planned_graph_extra(&lazy, Some(&[5_000_000_000, 4_000_000_000]), allowance),
            5_000_000_000 - allowance);
        assert_eq!(planned_graph_extra(&lazy, Some(&[1_000_000_000]), allowance), 0);
        assert_eq!(planned_graph_extra(&lazy, None, allowance), 0);
    }

    /// `measured_pool_tokens` modelled on a 5090 (33,711,521,792 B) at default flags with Spark experts,
    /// an automatic pool and 16 sequences, from what its launches measured once start-up had allocated
    /// everything but the pool: `free` bytes free (11,876,761,600 at 8,192 and at 131,072 tokens), and
    /// 2 GiB of headroom and 7,702,414,080 B of state and marks kept beside the graphs, at 11,804 B a
    /// token in 256-token units. The GPU's one-unit check, then the pool in what the reserve leaves
    /// (`admitted_pool_tokens`). Records each measurement's graph bytes in `calls`.
    fn measured_on_a_5090(free: u64, calls: &std::cell::RefCell<Vec<u64>>)
        -> impl FnMut(u64) -> anyhow::Result<usize> + '_ {
        use cuteafd_core::serving_capacity::{DeviceMemory, GpuMemoryBudget, DEFAULT_GPU_KV_TOKENS};
        const TOTAL: u64 = 33_711_521_792;
        move |graphs| {
            calls.borrow_mut().push(graphs);
            let reserve = (2u64 << 30) + graphs + 7_702_414_080;
            GpuMemoryBudget(TOTAL).admit(DeviceMemory { device: 0, total_bytes: TOTAL, baseline_free_bytes: free },
                reserve + 256 * 11_804)?;
            let room = free as i64 - reserve as i64;
            let tokens = cuteafd_core::memory_layout::size_pool(&[room], &[11_804], 256, DEFAULT_GPU_KV_TOKENS);
            anyhow::ensure!(tokens >= 256, crate::shared::memory_report::NoKvRoom { free_after_reserve: vec![room] });
            Ok(tokens as usize)
        }
    }

    /// The 5090's measured refusal (its one-unit check): `required` bytes against 33,711,521,792. At default
    /// flags, 131,072 tokens and 16 sequences the launch beside the startup set (28,060 graphs,
    /// 4,998,676,312 B) was refused needing 36,686,356,056 B, short 2,974,834,264 B.
    fn the_5090_refusal(required: u64, shortfall: u64) -> cuteafd_core::serving_capacity::CapacityError {
        cuteafd_core::serving_capacity::CapacityError::GpuBudgetExceeded { device: 0, required,
            budget: 33_711_521_792, shortfall }
    }

    /// The launch the measured admission refused falls back to lazily captured graphs, which keep the
    /// 1.5 GiB allowance: 3,388,063,576 B more room, 35,072 tokens. With less free memory, short either
    /// way, the startup refusal stands.
    #[test]
    fn a_measured_admission_falls_back_to_lazy_capture_on_a_real_shortfall() {
        use cuteafd_core::serving_capacity::CapacityError;
        let args = parse(&[&SPARKS[..], &["--pool-tokens", "0"][..]].concat());
        assert_eq!(kv_admission(&args, false, false), KvAdmission::Measured);
        let (allowance, set) = (graph_reserve(&args), 4_998_676_312);
        assert_eq!(allowance, 1_610_612_736);
        let calls = std::cell::RefCell::new(Vec::new());
        // The model refuses the startup set as the launch did.
        let refused = measured_on_a_5090(11_876_761_600, &calls)(set).unwrap_err();
        assert_eq!(refused.downcast_ref::<CapacityError>(), Some(&the_5090_refusal(36_686_356_056, 2_974_834_264)));
        assert_eq!(calls.take(), [set]);
        // Measured again keeping the allowance: lazy capture, 35,072 tokens.
        let admitted = measured_admission(&args, Some(set), measured_on_a_5090(11_876_761_600, &calls)).unwrap();
        assert_eq!(admitted, (35_072, false));
        assert_eq!(calls.take(), [set, allowance]);
        // 876,761,600 B less free: short by 463,532,288 B with the allowance too, so the startup refusal stands.
        let error = measured_admission(&args, Some(set), measured_on_a_5090(11_000_000_000, &calls)).unwrap_err();
        assert_eq!(error.downcast_ref::<CapacityError>(), Some(&the_5090_refusal(37_563_117_656, 3_851_595_864)));
        assert_eq!(calls.take(), [set, allowance]);
    }

    /// A startup set that fits is kept as before: measured once, captured at startup. At 8,192 tokens the
    /// launch's set (10,580 graphs, 1,926,552,328 B) left the 8,448 tokens it admitted. A set within the
    /// allowance (1,474,297,856 B, what that launch captured) is kept exactly, not as the allowance, and
    /// refused, has no room to gain from lazy capture: its refusal stands, measured once.
    #[test]
    fn a_measured_admission_keeps_a_startup_set_that_fits_unchanged() {
        let args = parse(&[&SPARKS[..], &["--pool-tokens", "0"][..]].concat());
        let calls = std::cell::RefCell::new(Vec::new());
        for (set, tokens) in [(1_926_552_328, 8_448), (1_474_297_856, 46_592)] {
            let admitted = measured_admission(&args, Some(set), measured_on_a_5090(11_876_761_600, &calls)).unwrap();
            assert_eq!(admitted, (tokens, true));
            assert_eq!(calls.take(), [set]);
        }
        let error = measured_admission(&args, Some(1_474_297_856), measured_on_a_5090(11_000_000_000, &calls))
            .unwrap_err();
        assert!(crate::shared::memory_report::kv_shortfall(&error), "{error:#}");
        assert_eq!(calls.take(), [1_474_297_856]);
    }

    /// A lazy retry that fails for another reason (its memory sample, an overflow, a CUDA query) returns
    /// its own error, whole, beneath the startup refusal; a first measurement that fails so is not retried.
    #[test]
    fn a_measured_retry_that_fails_for_another_reason_keeps_its_own_error() {
        use crate::shared::memory_report::kv_shortfall;
        use cuteafd_core::serving_capacity::CapacityError;
        let args = parse(&[&SPARKS[..], &["--pool-tokens", "0"][..]].concat());
        let (allowance, set) = (graph_reserve(&args), 4_998_676_312);
        let refusal = || anyhow::Error::from(the_5090_refusal(36_686_356_056, 2_974_834_264));
        let others: [fn() -> anyhow::Error; 3] = [
            || CapacityError::Invalid("invalid GPU budget or physical memory sample").into(),
            || anyhow::anyhow!("KV admission reserve overflows"),
            || anyhow::anyhow!("cuteafd_cuda_memory_info returned status 1: unspecified launch failure"),
        ];
        let calls = std::cell::RefCell::new(Vec::new());
        // Measurements answering `answers` in turn.
        let scripted = |mut answers: Vec<anyhow::Result<usize>>| { answers.reverse(); let calls = &calls;
            move |graphs: u64| { calls.borrow_mut().push(graphs); answers.pop().expect("at most two measurements") } };
        let chain = |error: &anyhow::Error| error.chain().map(|cause| cause.to_string()).collect::<Vec<_>>();
        for other in others {
            let error = measured_admission(&args, Some(set), scripted(vec![Err(refusal()), Err(other())])).unwrap_err();
            assert!(!kv_shortfall(&error), "{error:#}");
            assert_eq!(&chain(&error)[1..], chain(&other()).as_slice());
            assert!(error.to_string().contains(&format!("({:#})", refusal())), "{error:#}");
            assert_eq!(calls.take(), [set, allowance]);
            let error = measured_admission(&args, Some(set), scripted(vec![Err(other())])).unwrap_err();
            assert_eq!(chain(&error), chain(&other()));
            assert_eq!(calls.take(), [set]);
        }
    }

    /// Without a startup set (a graph budget turns startup capture off, as `CUTEAFD_GLMF_STARTUP_GRAPHS=0`
    /// does) the measured admission measures once, keeping the budget, else the allowance, and captures
    /// lazily, with nothing to fall back to: at 512 MiB the 126,208 tokens the budgeted launch admitted;
    /// at 4096 MiB its refusal stands.
    #[test]
    fn a_measured_admission_without_a_startup_set_measures_once() {
        let calls = std::cell::RefCell::new(Vec::new());
        for (budget, kept, admitted) in [(None, 1_610_612_736, Some(35_072)), (Some("512"), 536_870_912, Some(126_208)),
            (Some("4096"), 4_294_967_296, None)] {
            let args = with_budget(&[&SPARKS[..], &["--pool-tokens", "0"][..]].concat(), budget);
            assert!(budget.is_none() || !args.startup_graphs());
            let result = measured_admission(&args, None, measured_on_a_5090(11_876_761_600, &calls));
            assert_eq!(result.ok(), admitted.map(|tokens| (tokens, false)), "{budget:?}");
            assert_eq!(calls.take(), [kept], "{budget:?}");
        }
    }

    #[test]
    fn prefix_marks_default_to_the_arena_and_take_the_pool() {
        assert_eq!(parse(&[]).prefix_marks, prefix::PrefixMarks::Arena);
        assert_eq!(parse(&["--prefix-marks", "pool"]).prefix_marks, prefix::PrefixMarks::Pool);
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--prefix-marks", "host"]).is_err());
    }

    #[test]
    fn prefill_lanes_default_to_two_of_4096_rows_and_take_one_to_four() {
        let defaults = parse(&[]);
        assert_eq!((defaults.prefill_lanes, defaults.prefill_rows), (2, 4096));
        let four = parse(&["--prefill-lanes", "4", "--prefill-lane-rows", "2048"]);
        assert_eq!((four.prefill_lanes, four.prefill_rows), (4, 2048));
        assert_eq!(parse(&["--prefill-rows", "2048"]).prefill_rows, 2048);
        for lanes in ["0", "5"] {
            assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--prefill-lanes", lanes]).is_err());
        }
    }

    /// Only Spark experts have transports to warm at start-up: `experts` returns skipped and local
    /// experts before any transport exists, and `--local-experts` cannot be given with `--peers`.
    #[test]
    fn only_spark_experts_have_transports_to_warm() {
        assert!(parse(&["--local-experts"]).peers.is_none());
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--local-experts", "--peers", "127.0.0.1:19441"]).is_err());
        let spark = parse(&SPARKS);
        assert_eq!((spark.peers.as_deref(), spark.local_experts, spark.prefill_rows), (Some(SPARKS[1]), false, 4096));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Fp8PrefillGroup {
    Mla,
    Ffn,
    KdaIn,
    KdaO,
    /// Every group.
    All,
    /// No group: every prefill projection over FP8 weights runs W8A16.
    None,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, layerNN.bin and logits.bin from
    /// python/reference/families/glm5_flash/golden.py.
    #[arg(long)]
    pub golden: PathBuf,
    /// Prefill only the first N tokens, then feed the rest through decode
    /// steps of --step-rows rows (teacher-forced), comparing their rows.
    #[arg(long)]
    pub prefill: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub step_rows: usize,
    /// Decode steps run speculatively (KDA state untouched, replay rows
    /// recorded) and then commit every row: the verify-by-replay path.
    #[arg(long)]
    pub spec_steps: bool,
    /// Feed each layer the golden output of the previous one (prefill only),
    /// so every layer's cosine measures that layer alone.
    #[arg(long)]
    pub teacher_force: bool,
    /// Score every prefill row's logits against the golden (mean NLL, top-1).
    #[arg(long)]
    pub nll: bool,
    /// With --nll: also write tokens.bin and every prefill row's logits.bin (F32) to this
    /// directory, a golden for A/B runs between builds or numerics (with --skip-experts).
    #[arg(long, hide = true)]
    pub save_logits: Option<PathBuf>,
    /// Decode steps compare logits only (no per-layer downloads; graphs run).
    #[arg(long)]
    pub logits_only: bool,
    /// After the comparison, time this many greedy single-row decode steps
    /// (no layer downloads) from the prefilled sequence.
    #[arg(long, default_value_t = 0)]
    pub bench_decode: usize,
    /// Time this many more prefills of the golden prompt (up to --prefill-rows
    /// tokens) on fresh sequences, without layer downloads.
    #[arg(long, default_value_t = 0)]
    pub bench_prefill: usize,
    /// Prompt length of --bench-prefill (the golden tokens repeated), in
    /// chunks of the engine's prefill capacity; default the golden prompt up
    /// to one chunk.
    #[arg(long)]
    pub bench_prefill_tokens: Option<usize>,
    /// With --draft: run only the drafter on the golden taps (the layers'
    /// stream means) and compare with python/reference/families/glm5/dflash2/reference.py's
    /// output directory.
    #[arg(long)]
    pub draft_oracle: Option<PathBuf>,
    /// With --draft: replay the drafter alone on the golden taps, drafting
    /// after every token from this position on, BF16 and FP8 (acceptance
    /// against the text and the golden greedy picks, draft time).
    #[arg(long)]
    pub draft_replay: Option<usize>,
    /// With --draft: after the --prefill tokens, decode N greedy tokens one row
    /// per step, drafting before each, and report how many drafts the target
    /// reproduced (0: teacher-forced on tokens.bin, scoring against it).
    #[arg(long)]
    pub generate: Option<usize>,
    /// With --draft: the draft-kernel A/B. After --prefill tokens (default 1024), windows of N
    /// consecutive anchors draft in one step of N sequences (8N rows) under every --draft-head
    /// and --draft-linear setting the drafter admits (load it with --draft-linear w8a8 for all
    /// three linear modes), and each anchor alone, on the same teacher-forced contexts; prints the
    /// drafts kept as a prefix of the text, identical drafts and step times per setting. Anchors
    /// end at position 2048 (the ring's length).
    #[arg(long)]
    pub draft_modes: Option<usize>,
    /// Verify-by-replay check: after --prefill tokens, for every kept count k
    /// in 1..=N, require identical kept logits, KDA state, MLA rows and next
    /// decode when only the rejected suffix of the same N-row verify changes.
    /// Serial single-row differences are reported separately (kernel geometry
    /// may reorder floating point). Also check full commit against plain N-row
    /// verify, then time spec + commit from identical recurrent state.
    #[arg(long)]
    pub replay_check: Option<usize>,
    /// Require byte-exact real-row logits for speculative masked decode padding 3->4, 9->16, 17->32 and 33->64.
    #[arg(long)]
    pub padding_check: bool,
    /// Isolate one real routed-expert layer with --local-experts: compare a
    /// fixed first row across m1/m16/m80 packages and require that changing
    /// later inputs in the same row geometry cannot change it. No backbone
    /// weights or drafter load; this checks expert compute, not model KL.
    #[arg(long)]
    pub expert_row_check: Option<usize>,
    /// Dump fixed-token serial and --step-rows-wide layer outputs and router
    /// inputs to this directory, then report their full-vocabulary KL.
    #[arg(long)]
    pub geometry_trace: Option<PathBuf>,
    /// Time verify steps of 1..=N rows per sequence (C sequences, see
    /// --bench-sequences) after --prefill tokens: the step cost by rows.
    #[arg(long)]
    pub bench_verify: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub bench_sequences: usize,
    /// Prefix-cache restore check at token P: prefill the first P tokens (of --prefill, default
    /// all), capture their snapshot (shared units, the copied tail unit, the KDA state mark),
    /// restore it into a second sequence, continue both (prefill in --prefill-chunk rows, then
    /// --resume-decode greedy steps) and compare every layer's rows, the logits, the paged rows,
    /// the KDA state and the mark round trip byte for byte (a restore must be exact).
    #[arg(long)]
    pub resume_at: Option<usize>,
    /// Prefill chunk rows of --resume-at (default the engine's prefill rows).
    #[arg(long)]
    pub prefill_chunk: Option<usize>,
    #[arg(long, default_value_t = 4)]
    pub resume_decode: usize,
    /// With --resume-at: prefill the second sequence cold on its own pages instead of
    /// restoring it (the floor: what page placement alone changes).
    #[arg(long, hide = true)]
    pub resume_cold: bool,
    /// --resume-at attempts on fresh sequences (all must be byte-identical).
    #[arg(long, default_value_t = 1)]
    pub resume_repeat: usize,
    /// Diagnostic for --resume-at with --prefix-marks pool: fill the reserved unit 0's first MLA
    /// record with 0xFF in every MLA layer before the continuations and report the non-finite
    /// logits instead of failing on them (non-zero: a kernel reads that record for masked
    /// candidates).
    #[arg(long, hide = true, requires = "resume_at")]
    pub resume_poison_unit0: bool,
    /// Prefill lanes against serial passes: one pipelined chunk of the golden prompt (--prefill N
    /// truncates it) through the lanes, and through serial passes over the same cuts; every row's
    /// logits, the KDA state and every paged byte must be identical. Needs Spark --peers.
    #[arg(long)]
    pub lane_check: bool,
    /// Packed prefill against each sequence's own pass: N (2 to 10) sequences of mixed lengths cut
    /// from the golden prompt (the last resumed past the dense context) in one packed pass and
    /// each in its own pass; every sequence's last-row logits, KDA state and paged bytes must be
    /// identical. One GPU, every layer.
    #[arg(long)]
    pub packed_check: Option<usize>,
    /// Token I/O gate after the golden prompt (--prefill N truncates it), then
    /// stop: the resident embedding table against the shard, device against
    /// host greedy selection over this many decode steps, and device against
    /// host sampling.
    #[arg(long)]
    pub token_check: Option<usize>,
}

/// Option combinations rejected before any checkpoint or native work.
fn check_options(args: &EngineArgs) -> Result<()> {
    let precise = args.kda_fp32_partials || args.kda_output_shard;
    ensure!(!(precise || args.kda_prefill_expanded) ||
        (args.split_device.is_some() && args.kda_fp8 != fp8::KdaFp8::Off),
        "KDA partial/expanded options require --split-device and --kda-fp8 row128 or channel");
    ensure!(!precise || !args.fp8_prefill.iter().any(|g|
        matches!(g, Fp8PrefillGroup::KdaO | Fp8PrefillGroup::All)),
        "KDA precise partials require W8A16 output (--fp8-prefill kda-o/all is unsupported)");
    let kda_prefill = args.fp8_prefill.iter().any(|g| matches!(g, Fp8PrefillGroup::KdaIn | Fp8PrefillGroup::KdaO));
    ensure!(!kda_prefill || args.kda_fp8 != fp8::KdaFp8::Off,
        "--fp8-prefill kda-in/kda-o run over FP8 KDA weights; add --kda-fp8 row128 or channel");
    ensure!(!args.fp8_prefill.contains(&Fp8PrefillGroup::None) || args.fp8_prefill.len() == 1,
        "--fp8-prefill none takes no other group");
    ensure!(args.kda_nvfp4_gate.is_none() || args.kda_fp8 == fp8::KdaFp8::Off,
        "--kda-nvfp4-gate rounds the BF16 KDA projections; it takes --kda-fp8 off");
    ensure!(args.kda_state == engine::KdaState::F32 || (args.kda_fp8 == fp8::KdaFp8::Off && args.split_device.is_none()),
        "--kda-state bf16 runs the BF16-projection KDA programs on one GPU: it takes --kda-fp8 off and no \
        --split-device (the FP8-KDA and head-split programs keep an FP32 state)");
    ensure!(args.replay_records == engine::ReplayRecords::Own || args.split_device.is_none(),
        "--replay-records shared keeps the records in one GPU's prefill scratch: it takes no --split-device");
    ensure!(args.decode_rows == engine::DECODE_ROWS || args.split_device.is_none(),
        "--decode-rows {} runs the wide decode programs on one GPU: a head split (--split-device) takes \
        --decode-rows {}", args.decode_rows, engine::DECODE_ROWS);
    Ok(())
}

impl EngineArgs {
    /// Whether serving captures every decode graph at startup (`CUTEAFD_GLMF_STARTUP_GRAPHS`, on unless
    /// 0): a graph budget (`--graph-budget-mib`) bounds lazily captured graphs instead.
    pub(crate) fn startup_graphs(&self) -> bool {
        engine::startup_graphs_enabled() && self.graph_budget_mib.is_none()
    }

    /// `--headroom-gib` in bytes.
    pub(crate) fn headroom_bytes(&self) -> Result<u64> {
        ensure!(self.headroom_gib.is_finite() && self.headroom_gib >= 0.0, "--headroom-gib must be a size in GiB");
        Ok((self.headroom_gib * (1u64 << 30) as f64) as u64)
    }

    /// The rows every routed-expert resource holds (the dense NVFP4 package, local FP8 or EXL3
    /// experts, the Spark transports and their intake planes): the widest step, a prefill lane of
    /// `--prefill-rows` or a decode or verify step of `--decode-rows` (with `--prefill-rows 64
    /// --decode-rows 128`, a verify step's 127 rows).
    pub(crate) fn expert_rows(&self) -> usize {
        cuteafd_loader::serving_capacity::glmf_expert_rows(self.prefill_rows as u64, self.decode_rows as u64) as usize
    }
}

/// Device bytes of the Spark intake planes that experts sized for the decode rows hold beyond a
/// narrower prefill lane's (none when a lane is at least as wide as a decode step): what a planned
/// admission charges on the lead GPU, as it charges the wide selector, beyond the allowances it
/// keeps for `--prefill-rows` rows. `ranks` Spark ranks, one transport per prefill lane.
fn wide_intake_bytes(args: &EngineArgs, ranks: usize, hidden: usize) -> u64 {
    use cuteafd_loader::serving_capacity::glmf_spark_intake_bytes;
    let at = |rows: usize| glmf_spark_intake_bytes(args.prefill_lanes as u64, ranks as u64, rows as u64, hidden as u64);
    at(args.expert_rows()) - at(args.prefill_rows)
}

/// The step settings these arguments give the engine (its step plan's inputs besides layers and experts),
/// with the DSA index cache `with_engine` resolved (a head split keeps `keys`).
fn step_settings(args: &EngineArgs, index_cache: engine::IndexCache) -> engine::StepSettings {
    engine::StepSettings { kda_fp32_partials: args.kda_fp32_partials, kda_output_shard: args.kda_output_shard,
        kda_prefill_expanded: args.kda_prefill_expanded, full_prefill_logits: args.full_prefill_logits,
        max_context: args.max_context, index_cache, kda_state: args.kda_state, replay_records: args.replay_records,
        decode_rows: args.decode_rows }
}

/// Whether start-up loads program `name` for a GLM Flash engine: its own programs (`glmf_`), the
/// head split's `glmf2_` shares with a second GPU (rank 0 runs shares too), and the wide `_m128`
/// decode programs only with `--decode-rows 128` (`wide`: steps of up to 64 rows never launch them).
/// Every program a step can launch inside a decode graph capture is loaded; the engine launches no
/// other family's program.
pub(crate) fn glmf_startup_program(name: &str, split: bool, wide: bool) -> bool {
    (name.starts_with("glmf_") && (wide || !name.ends_with("_m128"))) || (split && name.starts_with("glmf2_"))
}

/// What a measured admission keeps free for decode graph executables: the graph budget, else the
/// planner's allowance.
fn graph_reserve(args: &EngineArgs) -> u64 {
    args.graph_budget_mib.map_or(cuteafd_loader::plan::layout::family_costs("glm5_flash").graph_bytes[0], |mib| mib << 20)
}

/// How a launch admits its KV pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvAdmission {
    /// From the free memory measured once everything else is allocated: one GPU with Spark experts
    /// and an automatic pool, or any one-GPU launch under a coordinator GPU budget. It keeps the
    /// startup set's reserve free for decode graphs, else `graph_reserve` (`measured_admission`).
    Measured,
    /// From the planner's per-GPU costs before the engine allocates: a head split, local experts
    /// without a GPU budget, a fixed pool. Every GPU keeps the planner's graph allowance and
    /// `planned_graph_extra` free.
    Planned,
    /// None: a fixed pool taken as given by a command without a serving loop, a GPU budget or
    /// all-row prefill logits, and with no decode graph memory to keep (neither the startup set nor
    /// a graph budget).
    Unchecked,
}

/// The KV admission of a launch with these arguments: `gpu_budget`, a coordinator GPU budget is set;
/// `split`, a head split's second GPU serves.
fn kv_admission(args: &EngineArgs, gpu_budget: bool, split: bool) -> KvAdmission {
    if (args.pool_tokens == 0 || gpu_budget) && !split && (!args.local_experts || gpu_budget) {
        KvAdmission::Measured
    } else if args.pool_tokens == 0 || args.full_prefill_logits || gpu_budget || args.startup_graphs()
        || args.graph_budget_mib.is_some() || args.serving_graph_policy.is_some() {
        KvAdmission::Planned
    } else {
        KvAdmission::Unchecked
    }
}

/// What a planned admission keeps free for decode graph executables on every GPU beyond the
/// planner's `allowance`, which `planned_pool_tokens_with_reserves` charges each GPU: the startup
/// set's largest rank when graphs are captured at startup, else the graph budget, which each rank's
/// cache may fill (`GlmfEngine::set_graph_budget`). So every GPU keeps max(allowance, startup set or
/// budget): a budget below the allowance keeps the allowance.
fn planned_graph_extra(args: &EngineArgs, startup_reserve: Option<&[u64]>, allowance: u64) -> u64 {
    let startup = startup_reserve.and_then(|reserve| reserve.iter().copied().max()).unwrap_or(0);
    startup.max(args.graph_budget_mib.map_or(0, |mib| mib << 20)).saturating_sub(allowance)
}

/// The measured admission beside the startup decode graphs (`startup`, the set's reserve on its one
/// GPU; None when graphs are captured lazily). `measure(graphs)` sizes the pool from the free memory
/// measured now, keeping `graphs` bytes free for decode graphs. It keeps the startup set's reserve
/// exactly, below the allowance or above it. When that leaves no room for a pool, it measures again
/// keeping `graph_reserve` (with a startup set, the planner's allowance), and decode graphs are
/// captured lazily: the planned admission's rule (`engine::admit_beside_decode_graphs`). Everything
/// else precedes the pool, so only the graph reserve differs between the two measurements. Returns the
/// pool and whether decode graphs are captured at startup.
fn measured_admission(args: &EngineArgs, startup: Option<u64>, mut measure: impl FnMut(u64) -> Result<usize>)
    -> Result<(usize, bool)> {
    let lazy = graph_reserve(args);
    engine::admit_beside_decode_graphs(startup.map(|reserve| engine::StartupGraphReserve { reserve, allowance: lazy }),
        |graphs| measure(graphs.unwrap_or(lazy)))
}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub checkpoint: Checkpoint,
    pub fp8_checkpoint: Option<Checkpoint>,
    pub cfg: GlmNextConfig,
    pub library: NativeLibrary,
    /// The FP8 expert catalog for --local-experts.
    pub experts: Option<cuteafd_loader::OfficialV41Catalog>,
}

impl Opened {
    fn fp8(&self) -> Option<&Fp8ExpertTensors> {
        self.experts.as_ref().and_then(|c| c.fp8())
    }
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    check_options(args)?;
    let checkpoint = Checkpoint::coordinator(&args.snapshot, args.local_experts, false)?;
    let cfg = GlmNextConfig::read(&args.snapshot)?;
    let fp8_checkpoint = args.fp8_snapshot.as_deref()
        .map(|snapshot| Checkpoint::coordinator(snapshot, false, false)).transpose()?;
    header::check_kda_inputs(&checkpoint, &cfg, args.layers.unwrap_or(cfg.layers))?;
    precision::check_projection_inputs(&checkpoint, fp8_checkpoint.as_ref(), &cfg,
        args.layers.unwrap_or(cfg.layers))?;
    if let Some(snapshot) = &args.draft {
        let head = checkpoint.tensors.iter().find(|t| t.meta.name == "lm_head.weight")
            .context("DFlash target has no lm_head.weight")?;
        // The drafter borrows the target's one head: BF16, or the FP8 head made from it.
        crate::families::glm5::dflash::check_target_head_source(&head.meta, cfg.hidden, cfg.vocab_size)?;
        if dspark::is_dspark(snapshot)? {
            dspark::check_checkpoint(snapshot, cfg.hidden, cfg.vocab_size, cfg.layers)?;
        } else {
            crate::families::glm5::dflash::check_checkpoint(snapshot, args.draft_fp8,
                args.draft_context_slots, args.draft_sequences)?;
        }
    }
    // The expert geometry is process-wide and must be fixed before the native
    // library loads (its expert helpers size rows from it).
    let geometry = cuteafd_core::ExpertGeometry::GLM5_FLASH;
    ensure!(geometry.hidden as usize == cfg.hidden && geometry.experts as usize == cfg.experts
        && geometry.topk as usize == cfg.topk && geometry.intermediate as usize == cfg.moe_intermediate,
        "checkpoint experts do not match the GLM 5.3 Flash geometry");
    cuteafd_core::set_expert_geometry(geometry).map_err(|g| anyhow::anyhow!("expert geometry already {g:?}"))?;
    let experts = if args.local_experts {
        let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
        let catalog = cuteafd_loader::read_expert_catalog(source)?;
        ensure!(catalog.fp8().is_some() || catalog.exl3().is_some(),
            "--local-experts runs FP8 or EXL3 experts; {} has neither", source.display());
        Some(catalog)
    } else {
        None
    };
    // SAFETY: the library is the cuteafd native shim built for this engine.
    let library = unsafe { NativeLibrary::load(&args.native_lib) }?;
    library.cuda_set_device(args.device)?;
    Ok(Opened { checkpoint, fp8_checkpoint, cfg, library, experts })
}

/// The DSA index cache an engine builds for `requested`: both GPUs of a head split run the indexer,
/// and their index tails are not built yet, so a head split keeps the token keys.
pub(crate) fn served_index_cache(requested: engine::IndexCache, head_split: bool) -> engine::IndexCache {
    if head_split { engine::IndexCache::Keys } else { requested }
}

impl Opened {
    /// Builds the engine and hands it to `body`. Either KV admission keeps the prefix marks of
    /// `args.mark_arena` the caller allocates once the engine exists free, counted on the layout
    /// the engine serves (`engine.mark_slots`).
    pub fn with_engine<T>(&self, args: &EngineArgs, body: impl FnOnce(&engine::GlmfEngine<'_>) -> Result<T>)
        -> Result<T> {
        // A head split whose GPUs lack peer access serves from --device alone, decided before any load or
        // admission. The arena is counted after both decisions (below), on the layout this engine serves:
        // each admitted GPU reserves its part of every mark (a lone GPU, the whole mark), as the prefix
        // cache then allocates.
        let mut resolved = args.clone();
        if args.split_device.is_some()
            && crate::shared::peer_split::probed_device(&self.library, args.device, args.split_device)?.is_none() {
            resolved.split_device = None;
            resolved.kda_fp32_partials = false;
            resolved.kda_output_shard = false;
            resolved.kda_prefill_expanded = false;
        }
        let args = &resolved;
        let automatic_context = args.max_context == 0;
        let mut context_args = args.clone();
        context_args.max_context = crate::shared::context::checkpoint_context(
            &args.snapshot, &args.manifest, "glm5_flash", args.max_context)?;
        let args = &context_args;
        let programs = self.library.programs()?.with_manifest(&args.manifest)?;
        programs.capacities().require_context("glm5_flash", args.max_context)?;
        // The single-copy FP8 consumers of the selected representations, before any weight loads.
        let mut needed = Vec::new();
        if args.kda_fp8 != fp8::KdaFp8::Off {
            needed.extend(["glmf_kda_w8_m64", "glmf_kda_w8_m4096"]);
        }
        if args.fp8_head {
            needed.push("glmf_head_fp8");
        }
        let compact = args.index_cache == engine::IndexCache::Compact;
        if compact {
            needed.extend(["glmf_index_producer_c_m64", "glmf_index_producer_c_m4096"]);
        }
        if args.kda_state != engine::KdaState::F32 || compact {
            // The state's KDA programs and the commit a speculative verify runs: the state's, or with
            // the compact index cache the one that also rebuilds the tails (`kda_commit_c[_s16]`).
            let kda: Vec<String> = if args.kda_state == engine::KdaState::F32 { Vec::new() }
                else { ["m64", "m4096"].iter().map(|cap| args.kda_state.program(cap)).collect() };
            let commit = if compact { args.kda_state.compact_commit_program() }
                else { args.kda_state.commit_program() };
            for name in kda.into_iter().chain([commit.to_string()]) {
                programs.spec(&format!("glmf_{name}")).with_context(|| format!("--kda-state {} with --index-cache {} \
                    needs program glmf_{name}; this native library predates it", args.kda_state.name(),
                    if compact { "compact" } else { "keys" }))?;
            }
        }
        if args.kda_fp32_partials {
            needed.extend(["glmf2_kda_w8_f32_m64", "glmf2_kda_w8_f32_m4096"]);
            needed.push("glmf2_add_fp32");
        }
        if args.kda_output_shard {
            ensure!(self.cfg.kda_heads * self.cfg.kda_head_dim == 2 * self.cfg.hidden && self.cfg.hidden % 2 == 0,
                "KDA output sharding needs KDA width=2*hidden and even hidden, got heads={} head_dim={} hidden={}",
                self.cfg.kda_heads, self.cfg.kda_head_dim, self.cfg.hidden);
            needed.extend(["glmf2_kda_w8_norm_m64", "glmf2_kda_w8_norm_m4096",
                "glmf2_kda_output_rows_m64", "glmf2_kda_output_rows_m4096",
                "glmf2_join_heads", "glmf2_join_rows"]);
        }
        if args.kda_prefill_expanded {
            needed.push(if args.kda_output_shard { "glmf2_kda_w8_norm_expanded_m4096" }
                else if args.kda_fp32_partials { "glmf2_kda_w8_f32_expanded_m4096" }
                else { "glmf2_kda_w8_expanded_m4096" });
            if args.kda_output_shard { needed.push("glmf2_kda_output_rows_expanded_m4096"); }
        }
        for name in needed {
            programs.spec(name).with_context(|| format!("--kda-fp8/--fp8-head/--index-cache compact need \
                program {name}; this native library predates it"))?;
        }
        // `--decode-rows 128`: every wide program this configuration's steps and commits launch.
        for name in wide_decode_programs(args, &self.cfg) {
            programs.spec(&name).with_context(|| format!("--decode-rows {} needs program {name}; this build \
                exports no 128-row decode programs (build with CUTEAFD_GLMF_WIDE_DECODE_ROWS=128)", args.decode_rows))?;
        }
        // GLM Flash's programs only (a release library carries every family's): the head split's
        // `glmf2_` share programs only with a second GPU, the wide `_m128` ones only with
        // `--decode-rows 128`.
        let (loaded, skipped) = programs.load_matching(|name| glmf_startup_program(name, args.split_device.is_some(),
            args.decode_rows > engine::DECODE_ROWS))?;
        tracing::info!(loaded, skipped, "GLM 5.3 Flash programs loaded on the coordinator GPU");
        let stream = self.library.cuda_stream_create()?;
        // The head split's second GPU and its stream; a head split needs its share's programs
        // (`glmf2`) in this build.
        let split_device = match args.split_device {
            Some(device) if programs.spec("glmf2_kda_m64").is_ok() => Some(device),
            Some(device) => {
                tracing::info!(device, "no head-split programs (glmf2) in this build; serving from --device alone");
                None
            }
            None => None,
        };
        ensure!(!(args.kda_fp32_partials || args.kda_output_shard || args.kda_prefill_expanded)
            || split_device.is_some(),
            "KDA partial/expanded options require native head-split programs (missing glmf2_kda_m64)");
        let index_cache = served_index_cache(args.index_cache, split_device.is_some());
        if let (Some(device), true) = (split_device, index_cache != args.index_cache) {
            tracing::warn!(device, "--index-cache compact is single-GPU for now; the head split keeps the token keys");
        }
        let peer_stream = match split_device {
            Some(device) => {
                ensure!(device != args.device, "--split-device must differ from --device");
                self.library.cuda_enable_peer(device)?;
                self.library.cuda_set_device(device)?;
                let stream = self.library.cuda_enable_peer(args.device).and_then(|()| self.library.cuda_stream_create());
                self.library.cuda_set_device(args.device)?;
                Some((device, stream?))
            }
            None => None,
        };
        let started = Instant::now();
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        // The prefix mark arena on the layout this engine serves, its index cache resolved above:
        // both admissions reserve these marks, and `prefix_cache` allocates as many.
        let mark_slots = args.mark_arena.slots_on(&self.cfg, layers, index_cache, args.kda_state)?;
        let loader = weights::GlmfLoader { library: &self.library, checkpoint: &self.checkpoint, stream,
            fp8_source: self.fp8_checkpoint.as_ref(), kda_fp8: args.kda_fp8, kda_output_shard: args.kda_output_shard,
            fp8_head: args.fp8_head, kda_nvfp4: args.kda_nvfp4_gate.as_deref().map(|mode| mode == "search"),
            fp8_scales: args.fp8_scales, device: args.device,
            peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
                .collect() };
        let source = self.embed_source()?;
        let (embedding, (model, mut shares)) = crate::shared::token_io::TokenEmbedding::load(&self.library, source,
            args.token_io.embed_placement, || {
                let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                loader.model(&self.cfg, layers)
            })?;
        let resident: usize = model.layers.iter().map(weights::GlmfLayer::bytes).sum();
        let peer_resident: usize = shares.iter().flatten().map(weights::GlmfLayer::bytes).sum();
        let single = model.check_single_residency(args.kda_fp8, args.fp8_head)?;
        let mib = |bytes: usize| bytes as f64 / (1u64 << 20) as f64;
        tracing::info!(layers, gib = resident as f64 / (1u64 << 30) as f64,
            split_gib = peer_resident as f64 / (1u64 << 30) as f64,
            fp8_source = self.fp8_checkpoint.is_some(), kda_fp8 = ?args.kda_fp8, fp8_prefill = ?args.fp8_prefill,
            kda_bf16_mib = mib(single.kda_bf16), kda_fp8_mib = mib(single.kda_fp8),
            head = model.head.name(), head_mib = mib(single.head_bf16 + single.head_fp8),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "GLM 5.3 Flash coordinator weights resident (one copy each)");
        // All-row prefill logits (`--full-prefill-logits`) admit as a GPU budget does: experts before the
        // pool, and a fixed pool checked against the exact workspaces. Only a GPU budget also widens the
        // eager admission below.
        let gpu_budget = cuteafd_ffi::coordinator_gpu_budget().is_some();
        let budgeted = args.full_prefill_logits || gpu_budget;
        // With a ceiling, establish local expert ownership before KV spends
        // the remaining budget. Lazy EXL3 owners reserve their loader peak.
        let mut future_expert_bytes = 0;
        let mut admitted_experts = if budgeted {
            ensure!(args.expert_window.is_none(),
                "coordinator GPU budget admission requires resident experts, not diagnostic --expert-window paging");
            let experts = self.experts(args)?;
            if let Some(engine::Experts::LocalExl3(local)) = &experts {
                ensure!(local.window >= layers,
                    "coordinator GPU budget admission requires all EXL3 layers resident; increase --exl3-window or use Sparks");
                let plan = crate::families::deepseek_v4::local::plan(&self.library, &local.native_lib,
                    local.catalog, 0, layers, local.max_rows, local.budget)?;
                let expected = layers.saturating_sub(local.catalog.routed_experts().first_layer);
                ensure!(plan.layers == expected, "coordinator GPU budget cannot fit the requested EXL3 expert window");
                future_expert_bytes = plan.peak_bytes as u64;
            }
            Some(experts)
        } else { None };
        let moe = (0..layers).any(|l| !self.cfg.dense[l]);
        // Pool marks keep `GLMF_POOL_MARK_RESERVED_UNITS` units beside the admitted pool, never
        // handed out (see `prefix`): the admission reserves their bytes like any fixed cost and the
        // engine allocates them past the pool's tokens, so the pool admitted is the pool usable.
        let reserved_units = if args.prefix_marks == prefix::PrefixMarks::Pool {
            cuteafd_loader::serving_capacity::GLMF_POOL_MARK_RESERVED_UNITS } else { 0 };
        let reserved_bytes = if reserved_units == 0 { 0 } else {
            let geometry = cuteafd_loader::serving_capacity::glm_flash_rank_cache_geometry(&self.cfg, layers, 1,
                index_cache.into(), args.kda_state.bytes() as u64)?;
            let rank = geometry.ranks.first().context("GLM 5.3 Flash cache geometry without a rank")?;
            reserved_units * (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes)
        };
        // Startup decode graphs (`CUTEAFD_GLMF_STARTUP_GRAPHS`, the default): every graph a serving
        // loop launches is captured before readiness, so the pool leaves their reserve free (per rank).
        // A graph budget (`--graph-budget-mib`) asks for lazily captured graphs within it instead.
        let startup_graphs = args.startup_graphs();
        if engine::startup_graphs_enabled() && !startup_graphs {
            tracing::info!(graph_budget_mib = ?args.graph_budget_mib,
                "GLM Flash decode graphs captured lazily within the graph budget, not at startup");
        }
        let startup_reserve = if startup_graphs {
            let (sequences, speculation) = args.serving_graph_policy.unwrap_or((16, true));
            // An automatic pool can only shrink this geometry, never exceed the 2M cap.
            let pool = if args.pool_tokens == 0 { cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS as usize }
                else { args.pool_tokens };
            // The engine's bucket sets: with the wide programs its verify budget, from this GPU's SMs.
            let buckets = engine::DecodeBuckets::new(engine::verify_budget(args.decode_rows, self.library.sm_count()?));
            Some(engine::serving_graph_reserve(args.max_context, pool, self.cfg.dense_context(),
                sequences, speculation, layers, peer_stream.is_some(), &buckets))
        } else { None };
        // Eager admission (an automatic or budgeted pool on one GPU, its experts on Sparks or
        // admitted under the budget): the drafter, the Spark transports and intake, the dense
        // package, the token selector and every step workspace exist before the pool is sized, so
        // the pool takes the free memory they leave, less the headroom, the graph reserve and the
        // cache state still to come; the MLA pools are allocated last (`GlmfEngine::new`). With
        // all-row prefill logits the lanes' temporaries already hold them.
        let admission = kv_admission(args, gpu_budget, peer_stream.is_some());
        let eager = admission == KvAdmission::Measured;
        ensure!(eager || args.replay_records == engine::ReplayRecords::Own, "--replay-records shared needs the step \
            workspaces before the KV pool: one GPU whose pool is sized from measured memory (an automatic pool with \
            Spark experts, or a coordinator GPU budget)");
        let mut early = None;
        let mut records = None;
        // Admit the package before KV sizing; measured free memory then excludes its buffers (the
        // eager admission loads it with the rest of start-up).
        let mut scoring_dense = if args.full_prefill_logits && !eager { self.load_dense(args, &model.layers)? }
            else { None };
        let (pool_tokens, startup_graphs) = if eager {
            let experts = if moe {
                match admitted_experts.take() { Some(experts) => experts, None => self.experts(args)? }
            } else { None };
            let drafter = self.load_drafter(args, stream, &embedding)?;
            if let Some(drafter) = &drafter {
                drafter.prepare_workspace()?;
            }
            let dense = self.load_dense(args, &model.layers)?;
            let mut selector = crate::shared::token_io::TokenSelector::new(&self.library, args.token_io.token_select,
                self.cfg.vocab_size, args.decode_rows)?;
            selector.reserve_sampler()?;
            let lanes = if engine::prefill_pipelines(layers, self.cfg.layers, experts.as_ref(), args.prefill_lanes) {
                args.prefill_lanes
            } else { 1 };
            let workspaces = engine::StepWorkspaces::allocate(&engine::StepPlan::new(&self.library, &programs, &self.cfg,
                &model.layers, experts.as_ref(), step_settings(args, index_cache)), args.prefill_rows, lanes)?;
            // The caches this engine builds: records per token, the sequences' state and marks, and
            // the speculative records follow the index cache, and a BF16 KDA state halves the state
            // per sequence and each mark the arena holds. The replay records and commit tables hold the
            // decode rows (`later` charges them: they are allocated with the pool).
            let geometry = cuteafd_loader::serving_capacity::glm_flash_rank_cache_geometry_rows(&self.cfg, layers, 1,
                index_cache.into(), args.kda_state.bytes() as u64, args.decode_rows as u64)?;
            let rank = geometry.ranks.first().context("GLM 5.3 Flash cache geometry without a rank")?;
            let unit = geometry.logical_unit_rows.max(1);
            // Shared replay records (of the decode rows) sit in the prefill scratch the workspaces already
            // hold.
            let shared_records = if args.replay_records == engine::ReplayRecords::Shared {
                cuteafd_loader::serving_capacity::glm_flash_kda_replay_bytes_rows(&self.cfg, layers, 1,
                    args.decode_rows as u64)?
            } else { 0 };
            let state = rank.fixed_state_bytes + rank.active_state_per_sequence_bytes * args.slots as u64
                + rank.speculative_replay_bytes - shared_records;
            // The prefix mark arena the caller allocates once the engine exists.
            let marks = mark_slots as u64 * rank.retained_mark_bytes;
            tracing::info!(device = args.device, state_bytes = state, mark_slots,
                mark_bytes = marks, "KV admission reserves recurrent state, replay records and prefix marks");
            // Graphs: the startup set's reserve, or the lazy graphs' budget (else the planner's allowance),
            // which they keep too when the startup set leaves no room for a pool (`measured_admission`).
            let (headroom, later) = (args.headroom_bytes()?, state + marks + future_expert_bytes + reserved_bytes);
            let startup = startup_reserve.as_ref().and_then(|reserve| reserve.first().copied());
            let admitted = measured_admission(args, startup, |graphs| {
                let reserve = crate::shared::memory_report::MeasuredReserve { headroom, graphs, later };
                crate::shared::memory_report::measured_pool_tokens(&self.library, args.device,
                    (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes).div_ceil(unit), unit, reserve,
                    (args.pool_tokens > 0).then_some(args.pool_tokens as u64))
            })?;
            records = if shared_records > 0 {
                tracing::info!(records_bytes = shared_records, later_bytes = later,
                    "GLM 5.3 Flash KDA replay records in the prefill lanes' scratch");
                Some(workspaces.prefill_scratch().context("shared replay records need the prefill lanes")?)
            } else { None };
            early = Some((experts, drafter, dense, selector, workspaces));
            admitted
        } else if admission == KvAdmission::Planned {
            // 0: automatic; fixed pools (budgeted, served, or beside decode graphs) retain their
            // requested size and refuse before allocation if the future storage would not fit.
            let precise_split = args.kda_fp32_partials || args.kda_output_shard;
            let devices: Vec<i32> = if precise_split {
                vec![args.device, args.split_device.context("precise head split")?]
            } else {
                std::iter::once(args.device)
                    .chain(peer_stream.map(|(d, _)| d)).collect()
            };
            // The head split's partials and the pool marks' reserved units (on every GPU).
            let extra = if args.kda_output_shard {
                engine::output_shard_reserve(args.prefill_lanes, args.prefill_rows, self.cfg.hidden)
            } else if args.full_prefill_logits {
                engine::partial_exchange_reserve(args.prefill_lanes, args.prefill_rows, self.cfg.hidden,
                    if args.kda_fp32_partials { 4 } else { 2 })
            } else {
                engine::partial_reserve(args.prefill_lanes, args.prefill_rows, self.cfg.hidden,
                    if args.kda_fp32_partials { 4 } else { 2 })
            } + reserved_bytes;
            // The step workspaces this engine makes before readiness (`workspace_reserve`, per rank: the
            // decode workspace, the prefill lanes over their shared temporaries, and the measured runtime
            // allowance per workspace), past the planner's workspace allowance; with all-row prefill
            // logits, their exact union replaces the allowance.
            let spark = args.peers.is_some();
            let plan = engine::StepPlan::new(&self.library, &programs, &self.cfg, &model.layers, None,
                step_settings(args, index_cache)).with_experts(!args.skip_experts && self.fp8().is_some(), spark);
            let lanes = engine::configured_prefill_lanes(spark, layers == self.cfg.layers, args.prefill_lanes);
            let workspace_reserve = engine::workspace_reserve(&plan, args.prefill_rows, lanes, peer_stream.is_some(),
                args.draft.is_some())?;
            let costs = cuteafd_loader::plan::layout::family_costs("glm5_flash");
            let workspace_allowance = costs.workspace_bytes[0] * args.prefill_rows.max(1) as u64 / 4096;
            let workspace_extra = workspace_reserve.iter().copied().max().unwrap_or(0).saturating_sub(workspace_allowance);
            tracing::info!(lanes, workspace_reserve_bytes = ?workspace_reserve, planner_allowance_bytes = workspace_allowance,
                extra_reserve_bytes = workspace_extra, "GLM Flash derived workspace reserve before KV admission");
            // The startup set or the graph budget above the planner's graph allowance, on every GPU.
            let allowance = costs.graph_bytes[0];
            if startup_graphs || args.graph_budget_mib.is_some() {
                tracing::info!(allowance_bytes = allowance, graph_budget_mib = ?args.graph_budget_mib,
                    extra_reserve_bytes = planned_graph_extra(args, startup_reserve.as_deref(), allowance),
                    "GLM Flash graph reserve above planner allowance");
            }
            // If the startup set leaves no room for a pool, decode graphs are captured lazily instead,
            // keeping the graph budget or the allowance (`admit_beside_decode_graphs`).
            let lazy_extra = planned_graph_extra(args, None, allowance);
            let startup = startup_reserve.as_deref().map(|reserve| engine::StartupGraphReserve {
                reserve: reserve.iter().copied().max().unwrap_or(0), allowance });
            // Every GPU keeps the prefix mark arena the caller allocates (its part of `mark_slots`
            // marks, whatever the KDA state) and the speculative replay records free, as the runtime
            // allocates them.
            // The token selector and GPU sampler of `--decode-rows 128` beyond the 64-row ones the
            // runtime allowance holds: the serving loop makes them after the pool, on the lead GPU.
            let selector = wide_selector_bytes(args.decode_rows, self.cfg.vocab_size);
            // The Spark intake planes of experts sized for decode steps wider than a prefill lane
            // (`--prefill-rows 64 --decode-rows 128`), past the lane's rows, also on the lead GPU.
            let intake = match args.peers.as_deref() {
                Some(peers) if moe && !args.skip_experts => wide_intake_bytes(args, peers.split(',').count(),
                    self.cfg.hidden),
                _ => 0,
            };
            engine::admit_beside_decode_graphs(startup, |graphs| {
                // The startup set's bytes above the allowance (none when captured lazily), or the budget's.
                let graph_extra = graphs.map_or(0, |bytes| bytes.saturating_sub(allowance)).max(lazy_extra);
                let reserves: Vec<_> = workspace_reserve.iter().enumerate().map(|(rank, &workspace)|
                    crate::shared::memory_report::RankReserve {
                        extra_bytes: extra + graph_extra + if args.full_prefill_logits { 0 } else { workspace_extra }
                            + if rank == 0 { selector + intake } else { 0 },
                        workspace_bytes: args.full_prefill_logits.then_some(workspace),
                    }).collect();
                crate::shared::memory_report::planned_pool_tokens_with_reserves(&self.library, &args.snapshot, &devices,
                    args.draft.as_deref(), args.prefill_rows, args.slots, mark_slots as u64,
                    (args.pool_tokens > 0).then_some(args.pool_tokens as u64), future_expert_bytes, &reserves,
                    index_cache.into(), args.kda_state.bytes() as u64, args.decode_rows as u64)
            })?
        } else {
            (args.pool_tokens, startup_graphs)
        };
        let max_context = crate::shared::context::pool_context("glm5_flash", args.max_context, automatic_context, pool_tokens, 256)?;
        let pages = (pool_tokens + reserved_units as usize * engine::UNIT_ROWS).div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::GlmfEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            max_context, args.prefill_rows, args.prefill_lanes, pages, args.slots, embedding, index_cache,
            args.kda_state, args.decode_rows, records)?;
        if !startup_graphs {
            engine.capture_graphs_lazily();
        }
        tracing::info!(index_cache = ?engine.index_cache, pool_tokens, reserved_units, "GLM 5.3 Flash DSA index cache");
        engine.kda_fp32_partials = args.kda_fp32_partials;
        engine.kda_output_shard = args.kda_output_shard;
        engine.kda_prefill_expanded = args.kda_prefill_expanded;
        engine.mark_slots = mark_slots;
        if let Some((device, peer_stream)) = peer_stream {
            engine.attach_peer(device, peer_stream, shares.pop().context("head-split shares")?)?;
            tracing::info!(device = args.device, split_device = device, "GLM 5.3 Flash head split over two GPUs");
        }
        engine.set_graph_budget(args.graph_budget_mib.map(|mib| mib << 20));
        engine.full_prefill_logits = args.full_prefill_logits;
        let group = |g: Fp8PrefillGroup| args.fp8_prefill.iter().any(|&x| x == g || x == Fp8PrefillGroup::All);
        // `all`: every group with FP8 weights (BF16 KDA has none to run W8A8 over).
        let kda = args.kda_fp8 != fp8::KdaFp8::Off;
        engine.fp8_prefill = engine::Fp8Prefill { mla: group(Fp8PrefillGroup::Mla), ffn: group(Fp8PrefillGroup::Ffn),
            kda_bits: i32::from(kda && group(Fp8PrefillGroup::KdaIn))
                | (i32::from(kda && group(Fp8PrefillGroup::KdaO)) << 1) };
        match early {
            Some((experts, drafter, dense, selector, workspaces)) => {
                engine.drafter = drafter;
                if let Some(dense) = dense {
                    engine.set_dense_nvfp4(dense);
                }
                if let Some(experts) = experts {
                    engine.set_experts(experts);
                }
                engine.install_workspaces(workspaces)?;
                engine.set_selector(selector);
            }
            None => {
                engine.drafter = self.load_drafter(args, stream, &engine.embedding)?;
                let dense = match scoring_dense.take() {
                    Some(dense) => Some(dense),
                    None => self.load_dense(args, &engine.weights.layers)?,
                };
                if let Some(dense) = dense {
                    engine.set_dense_nvfp4(dense);
                }
                if moe {
                    if let Some(experts) = match admitted_experts { Some(experts) => experts, None => self.experts(args)? } {
                        engine.set_experts(experts);
                    }
                }
            }
        }
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::GLM_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
            if engine.ranks() > 1 {
                engine.attach_peer_l2(budget)?;
            }
        }
        if args.serving_graph_policy.is_some() { engine.prepare_serving_workspaces()?; }
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        let result = body(&engine);
        drop(engine);
        // SAFETY: the engine that used the streams is gone.
        unsafe { self.library.cuda_stream_destroy(stream)? };
        if let Some((device, peer_stream)) = peer_stream {
            crate::shared::peer_split::on_device(&self.library, device, args.device,
                // SAFETY: as above.
                || unsafe { self.library.cuda_stream_destroy(peer_stream) })?;
        }
        result
    }

    /// The drafter `--draft` names, on `stream` (it reads only the mask token's embedding row).
    fn load_drafter<'s>(&'s self, args: &EngineArgs, stream: *mut std::ffi::c_void,
        embedding: &crate::shared::token_io::TokenEmbedding<'_>) -> Result<Option<dspark::Drafter<'s>>> {
        let Some(snapshot) = &args.draft else { return Ok(None) };
        let started = Instant::now();
        let representation = cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation
            ::from_fp8_option(args.draft_fp8);
        let drafter = dspark::Drafter::load(&self.library, snapshot, stream,
            args.draft_context_slots.unwrap_or(20.max(args.draft_sequences)), args.draft_sequences,
            embedding, self.cfg.hidden, self.cfg.vocab_size, self.cfg.layers, representation, args.fp8_scales,
            args.draft_linear)?;
        drafter.set_draft_head(args.draft_head);
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, draft_head = ?args.draft_head,
            draft_linear = ?args.draft_linear, "{} drafter resident", drafter.name());
        Ok(Some(drafter))
    }

    /// The one-expert NVFP4 package of ModelOpt NVFP4 dense MLPs, when `layers` have them.
    fn load_dense<'s>(&'s self, args: &EngineArgs, layers: &[weights::GlmfLayer<'_>])
        -> Result<Option<engine::DenseNvfp4<'s>>> {
        if !layers.iter().any(|layer| layer.has("nvfp4_w1")) {
            return Ok(None);
        }
        let directory = crate::shared::experts::fp8::dense_package_directory(&args.native_lib, "glmfdense");
        let dense = engine::DenseNvfp4::load(&self.library, &directory, &self.cfg, args.expert_rows())?;
        tracing::info!(package = %directory.display(), "NVFP4 dense MLPs on their own package");
        Ok(Some(dense))
    }

    fn experts<'s>(&'s self, args: &EngineArgs) -> Result<Option<engine::Experts<'s>>> {
        if args.skip_experts {
            return Ok(Some(engine::Experts::Skip));
        }
        if let Some(tensors) = self.fp8() {
            let directory = args.fp8_package.clone()
                .unwrap_or_else(|| crate::shared::experts::fp8::package_directory(&args.native_lib, 1, tensors.format()));
            let (free, _) = self.library.cuda_memory_info()?;
            ensure!(args.expert_window != Some(0), "--expert-window must be at least 1");
            let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
            let first = (0..layers).find(|&layer| !self.cfg.dense[layer]).unwrap_or(layers);
            let resident = if args.expert_window.is_some() { 0..0 } else { first..layers };
            let started = Instant::now();
            let experts = crate::shared::experts::fp8::Fp8Experts::load(&self.library, tensors, &directory, resident, 1, 0,
                args.expert_rows(), free.saturating_sub(args.expert_reserve_gib.saturating_mul(1 << 30)))
                .context("local routed experts must fit with step and prefix-cache reservations; use --peers for Sparks, \
                    or --expert-window N for diagnostic paging")?;
            let loads = experts.layers.len();
            tracing::info!(layers = loads, window = ?args.expert_window, elapsed_ms = started.elapsed().as_millis() as u64,
                "GLM 5.3 Flash routed experts resident on this GPU");
            return Ok(Some(engine::Experts::Local(engine::LocalExperts {
                library: &self.library, tensors, experts: std::cell::RefCell::new(experts),
                window: args.expert_window, loads: std::cell::RefCell::new(loads),
            })));
        }
        if let Some(catalog) = self.experts.as_ref().filter(|c| c.exl3().is_some()) {
            let (free, _) = self.library.cuda_memory_info()?;
            return Ok(Some(engine::Experts::LocalExl3(engine::LocalExl3 {
                library: &self.library, native_lib: args.native_lib.clone(), catalog,
                resident: std::cell::RefCell::new(None), window: args.exl3_window.max(1),
                layers: args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers), max_rows: args.expert_rows(),
                // Room for the step workspace (logits alone are 2.4 GiB at 4096 rows).
                budget: free.saturating_sub(12 << 30), loads: std::cell::RefCell::new(0),
            })));
        }
        let Some(peers) = args.peers.as_deref() else { return Ok(None) };
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let executors: Vec<u64> = (0..peers.len())
            .map(|rank| cuteafd_transport::expert::v41_spark_executor_id(peers.len(), rank))
            .collect::<Result<_>>()?;
        // One transport per prefill lane: each lane's wave stays in flight on its own QPs. Each takes
        // waves of the widest step (`expert_rows`): a verify step may be wider than a prefill lane.
        let mut transports = (0..args.prefill_lanes).map(|_| crate::shared::spark_intake::SparkLink::new(&self.library,
            &peers, &executors, u32::try_from(args.expert_rows())?, cuteafd_transport::TcpTransportConfig { timing: false,
                timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 }, self.cfg.hidden * 2))
            .collect::<Result<Vec<_>>>()?;
        crate::shared::memory_report::release_load_staging(&self.library);
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        // Connect every rank and register full-size buffers now, as GLM-5, MiMo and DeepSeek V4 do: a
        // rank's session fits the request that opened it, and a request that does not fit reconnects
        // the rank first (`LocalTp4Client::post`). A lane's rows follow the prompt, so each new longest
        // prompt would reconnect four ranks mid-prefill, and the first request would open transport 0.
        // Every transport a prefill runs on (transport 0 also carries decode, verify and serial chunks)
        // is warmed with `expert_rows` rows, the most any of its waves carries.
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let warmed = engine::configured_prefill_lanes(true, layers == self.cfg.layers, transports.len());
        let warm_rows = args.expert_rows();
        let warm = engine::spark_warmup_request(&self.cfg, warm_rows)?;
        let rings = || cuteafd_ffi::memory_ledger::snapshot().by_scope(cuteafd_ffi::memory_ledger::Space::Pinned, -1)
            .get("transport/rdma-rings").copied().unwrap_or(0);
        let (started, before) = (Instant::now(), rings());
        // Drained and destroyed on every way out, before the transports drop.
        let warm_stream = crate::shared::spark_intake::WarmStream::new(&self.library)?;
        for (index, transport) in transports[..warmed].iter_mut().enumerate() {
            runtime.block_on(async {
                let wave = transport.dispatch(&warm)?;
                transport.receive(wave, warm_rows, warm_stream.raw()).await
            }).with_context(|| format!("warming Spark expert transport {index} with {warm_rows} rows"))?;
        }
        warm_stream.finish()?;
        tracing::info!(transports = warmed, lanes = transports.len(), ranks = peers.len(), rows = warm_rows,
            ring_bytes = rings().saturating_sub(before), elapsed_ms = started.elapsed().as_millis() as u64,
            "Spark expert transports warm");
        Ok(Some(engine::Experts::Spark { transports: std::cell::RefCell::new(transports), runtime }))
    }
}

impl Opened {
    /// The checkpoint's `embed_tokens` (BF16 [vocab, hidden]).
    fn embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let name = format!("{}embed_tokens.weight", weights::PREFIX);
        let at = self.checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(&name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no {name}"))?;
        let tensor = &self.checkpoint.tensors[at];
        crate::shared::token_io::EmbedSource::new(&self.checkpoint.snapshot, &tensor.shard, &tensor.meta,
            self.cfg.hidden)
    }
}

fn bf16s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn similarity(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut diff) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE), diff.sqrt() / nb.sqrt().max(f64::MIN_POSITIVE))
}

pub(crate) async fn run_golden(mut args: GoldenArgs) -> Result<()> {
    args.engine.full_prefill_logits |= args.nll || args.resume_at.is_some() || args.lane_check;
    // --resume-at captures into two arena marks (`prefix::resume_check`); pool marks take units.
    args.engine.mark_arena = if args.resume_at.is_some() && args.engine.prefix_marks == prefix::PrefixMarks::Arena {
        prefix::ArenaMarks::Slots(2) } else { prefix::ArenaMarks::None };
    tokio::task::spawn_blocking(move || golden(args)).await?
}

fn golden(args: GoldenArgs) -> Result<()> {
    let opened = open(&args.engine)?;
    if let Some(rows) = args.expert_row_check {
        return expert_rows::check(&args.engine, &opened, rows);
    }
    opened.with_engine(&args.engine, |engine| golden_run(&args, &opened, engine))
}

/// Mean NLL of `logits` rows against the next tokens, and top-1 agreements.
/// Mean KL(golden || engine) over rows, in float64 (the golden's next-token
/// distribution against the engine's, as the published KL gates compute it).
pub(crate) fn mean_kl(logits: &[f32], golden: &[f32], first: usize, vocab: usize) -> f64 {
    let log_softmax = |l: &[f32]| -> Vec<f64> {
        let top = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = top + l.iter().map(|&x| (x as f64 - top).exp()).sum::<f64>().ln();
        l.iter().map(|&x| x as f64 - lse).collect()
    };
    let rows = logits.len() / vocab;
    let total: f64 = logits.chunks_exact(vocab).enumerate().map(|(r, ours)| {
        let (p, q) = (log_softmax(&golden[(first + r) * vocab..][..vocab]), log_softmax(ours));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum::<f64>()
    }).sum();
    total / rows.max(1) as f64
}

fn score(logits: &[f32], golden: &[f32], tokens: &[u32], first: usize, vocab: usize) -> (usize, usize, usize, f64, usize) {
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
    let (mut agree, mut next_ok, mut golden_next, mut nll, mut scored) = (0, 0, 0, 0f64, 0);
    for (r, ours) in logits.chunks_exact(vocab).enumerate() {
        let theirs = &golden[(first + r) * vocab..][..vocab];
        agree += usize::from(argmax(ours) == argmax(theirs));
        if let Some(&next) = tokens.get(first + r + 1) {
            next_ok += usize::from(argmax(ours) == next as usize);
            golden_next += usize::from(argmax(theirs) == next as usize);
            let top = ours.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = ours.iter().map(|&l| (l as f64 - top).exp()).sum();
            nll += top + sum.ln() - ours[next as usize] as f64;
            scored += 1;
        }
    }
    (agree, next_ok, golden_next, nll, scored)
}

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmfEngine<'_>) -> Result<()> {
    if args.padding_check {
        let bytes = std::fs::read(args.golden.join("tokens.bin"))?;
        anyhow::ensure!(bytes.len() % 4 == 0, "padding check token bytes must be u32-aligned");
        let tokens: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        return engine.check_decode_padding(&tokens);
    }
    if let Some(sequences) = args.draft_modes {
        return speculate::draft_modes(args, engine, sequences);
    }
    if let Some(dir) = &args.draft_oracle {
        return speculate::draft_oracle(args, opened, engine, dir);
    }
    if let Some(start) = args.draft_replay {
        return speculate::draft_replay(args, opened, engine, start);
    }
    if let Some(rows) = args.replay_check {
        return speculate::replay_check(args, engine, rows);
    }
    if let Some(dir) = &args.geometry_trace {
        return speculate::geometry_trace(args, engine, dir);
    }
    if let Some(rows) = args.bench_verify {
        return speculate::bench_verify(args, engine, rows);
    }
    if let Some(at) = args.resume_at {
        let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
            .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return prefix::resume_check(engine, &tokens, at, n,
            args.prefill_chunk.unwrap_or(engine.prefill_rows), args.resume_decode, args.resume_cold,
            args.resume_repeat, args.engine.prefix_marks, args.resume_poison_unit0);
    }
    if let Some(steps) = args.token_check {
        return token_check(args, opened, engine, steps);
    }
    if args.lane_check {
        let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
            .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return lane_check::lane_check(engine, &tokens[..n]);
    }
    if let Some(sequences) = args.packed_check {
        let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
            .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        return packed_check::packed_check(engine, &tokens, sequences);
    }
    if engine.drafter.is_some() {
        return speculate::draft_run(args, opened, engine);
    }
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    ensure!(!args.nll || engine.full_prefill_logits, "--nll needs full prefill logits");
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut placement = engine::Allocator::new(engine.pages, engine.slots).admit(tokens.len() + args.bench_decode)?;
    let row = cfg.hidden * 2;
    let stream_row = row * 4;
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    // Rows [first, first + n) of layer `layer`'s golden streams.
    let compare = |layer: usize, first: usize, streams: &[u8], worst: &mut Vec<f64>| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        if let Ok(golden) = std::fs::read(&path) {
            ensure!(golden.len() >= first * stream_row + streams.len(), "golden layer {layer} is short");
            let golden = &golden[first * stream_row..][..streams.len()];
            let (ours, theirs) = (bf16s(streams), bf16s(golden));
            let (cosine, rel) = similarity(&ours, &theirs);
            if worst.len() <= layer {
                worst.resize(layer + 1, 1.0);
            }
            worst[layer] = worst[layer].min(cosine);
            if first == 0 {
                let mut rows: Vec<f64> = ours.chunks_exact(row * 2).zip(theirs.chunks_exact(row * 2))
                    .map(|(a, b)| similarity(a, b).0).collect();
                rows.sort_by(f64::total_cmp);
                let bad = rows.iter().filter(|&&c| c < 0.999).count();
                println!("layer {layer:2} ({:?}): cosine {cosine:.6} rel_l2 {rel:.3e} | rows: median {:.6} p1 {:.6} \
                    worst {:.6}, {bad} of {} below 0.999", cfg.attention[layer], rows[rows.len() / 2],
                    rows[rows.len() / 100], rows[0], rows.len());
            }
        }
        Ok(())
    };
    let mut worst = Vec::new();
    let started = Instant::now();
    let forced = |layer: usize| -> Option<Vec<u8>> {
        std::fs::read(args.golden.join(format!("layer{layer:02}.bin"))).ok().map(|rows| rows[..prefill * stream_row].to_vec())
    };
    // Prefill in chunks of the engine's prefill rows (teacher forcing needs one chunk).
    ensure!(!args.teacher_force || prefill <= engine.prefill_rows, "teacher forcing takes one prefill chunk");
    let mut logits: Option<Vec<f32>> = None;
    let mut done = 0;
    // --logits-only prefills without layer downloads (Spark prefill then runs
    // pipelined in lanes, chunks up to the engine's prefill capacity).
    let chunk_rows = if args.logits_only && !args.teacher_force { engine.prefill_capacity() } else { engine.prefill_rows };
    while done < prefill {
        let n = chunk_rows.min(prefill - done);
        let first = done;
        let mut compare_layer = |layer, streams: &[u8]| compare(layer, first, streams, &mut worst);
        let on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
            if args.logits_only && !args.teacher_force { None } else { Some(&mut compare_layer) };
        let chunk = engine.prefill_forced(&mut placement, &tokens[done..done + n], on_layer,
            args.teacher_force.then_some(&forced as &dyn Fn(usize) -> Option<Vec<u8>>), args.nll)?;
        logits = match (logits, chunk) {
            (Some(mut all), Some(more)) if args.nll => {
                all.extend(more);
                Some(all)
            }
            (_, chunk) => chunk,
        };
        done += n;
    }
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut decode_worst = Vec::new();
    let mut decode_logits: Vec<f32> = Vec::new();
    let mut position = prefill;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        let mut compare_layer = |layer, streams: &[u8]| compare(layer, first, streams, &mut decode_worst);
        let on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>> =
            if args.logits_only { None } else { Some(&mut compare_layer) };
        let step = &tokens[position..position + n];
        let logits = if args.spec_steps {
            let logits = engine.verify_spec(&mut [(&mut placement, n)], step)?;
            engine.commit(&[(placement.slot, 0, n)])?;
            logits
        } else {
            engine.verify(&mut [(&mut placement, n)], step, on_layer)?
        };
        if let Some(logits) = logits {
            decode_logits.extend(logits);
        }
        position += n;
    }
    let golden_logits = || -> Result<Vec<f32>> {
        Ok(std::fs::read(args.golden.join("logits.bin"))?
            .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    };
    let vocab = cfg.vocab_size;
    if prefill < tokens.len() {
        println!("decode: {} rows in steps of {} in {:.2} s; worst row-block cosine per layer {:?}",
            tokens.len() - prefill, args.step_rows, started.elapsed().as_secs_f64(),
            decode_worst.iter().map(|c| format!("{c:.6}")).collect::<Vec<_>>());
        if !decode_logits.is_empty() {
            // Numerics A/B between engine configs (benchmarks): the decode rows' logits, F32.
            if let Ok(path) = std::env::var("CUTEAFD_DUMP_DECODE_LOGITS") {
                std::fs::write(&path, decode_logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
            let golden = golden_logits()?;
            let (agree, next_ok, golden_next, nll, scored) = score(&decode_logits, &golden, &tokens, prefill, vocab);
            let (_, _, _, golden_nll, _) =
                score(&golden[prefill * vocab..(prefill * vocab + decode_logits.len())], &golden, &tokens, prefill, vocab);
            let rows = decode_logits.len() / vocab;
            println!("decode logits: top-1 agreement {:.2}% over {rows} rows | next-token accuracy engine {:.1}% \
                golden {:.1}% | mean NLL engine {:.4} golden {:.4} | mean KL(golden||engine) {:.5}",
                100.0 * agree as f64 / rows as f64, 100.0 * next_ok as f64 / scored.max(1) as f64,
                100.0 * golden_next as f64 / scored.max(1) as f64, nll / scored.max(1) as f64,
                golden_nll / scored.max(1) as f64, mean_kl(&decode_logits, &golden, prefill, vocab));
        }
    }
    if args.bench_prefill > 0 {
        *engine.profile.borrow_mut() = [0.0; 3];
        engine.op_profile()?;
        let n = args.bench_prefill_tokens.unwrap_or(prefill.min(engine.prefill_capacity()));
        let long: Vec<u32> = tokens.iter().copied().cycle().take(n).collect();
        let mut allocator = engine::Allocator::new(engine.pages, engine.slots);
        let _held = allocator.admit(tokens.len() + args.bench_decode)?;
        let mut times = Vec::new();
        for _ in 0..args.bench_prefill {
            let mut fresh = allocator.admit(n)?;
            let started = Instant::now();
            for chunk in long.chunks(engine.prefill_capacity()) {
                engine.prefill(&mut fresh, chunk, None)?;
            }
            times.push(started.elapsed().as_secs_f64());
            allocator.release(fresh);
        }
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        let phases = std::mem::take(&mut *engine.profile.borrow_mut());
        println!("prefill bench phases per prefill: GPU until the expert exchange {:.1} ms, Spark exchange {:.1} ms",
            1e3 * phases[0] / times.len() as f64, 1e3 * phases[1] / times.len() as f64);
        println!("prefill bench: {n} tokens through {layers} layers, median {:.1} ms ({:.0} tok/s), min {:.1} ms",
            1e3 * median, n as f64 / median, 1e3 * times[0]);
        let ops = engine.op_profile()?;
        if !ops.is_empty() {
            let runs = times.len() as f64;
            let total: f64 = ops.iter().filter(|(k, _)| !k.starts_with("host")).map(|(_, v)| v.0).sum();
            println!("prefill ops per prefill (GPU ms between events, {total:.1} ms total per {runs} runs):");
            let mut rows: Vec<_> = ops.into_iter().collect();
            rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
            for (label, (ms, count)) in rows {
                println!("  {label:44} {:9.2} ms  {:6.1}%  x{}", ms / runs, 100.0 * ms / total, count as f64 / runs);
            }
        }
    }
    if args.bench_decode > 0 {
        *engine.profile.borrow_mut() = [0.0; 3];
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
        let mut token = tokens[placement.len.min(tokens.len() - 1)];
        let mut times = Vec::new();
        let mut produced = Vec::new();
        // FNV-1a over every step's logits bits (bit-identity checks between builds).
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..args.bench_decode {
            let started = Instant::now();
            let logits = engine.verify(&mut [(&mut placement, 1)], &[token], None)?;
            times.push(started.elapsed().as_secs_f64());
            if let Some(logits) = logits {
                for v in &logits {
                    digest = (digest ^ u64::from(v.to_bits())).wrapping_mul(0x0100_0000_01b3);
                }
                token = argmax(&logits);
                produced.push(token);
            }
        }
        times.sort_by(f64::total_cmp);
        let profile = engine.profile.borrow();
        let steps = times.len() as f64;
        println!("decode bench: {} steps through {layers} layers, median {:.2} ms (min {:.2}, max {:.2}); \
            per step: GPU until the expert exchanges {:.2} ms, Spark exchanges {:.2} ms, head {:.2} ms; \
            logits digest {digest:016x}; tokens {:?}", times.len(),
            1e3 * times[times.len() / 2], 1e3 * times[0], 1e3 * times[times.len() - 1], 1e3 * profile[0] / steps,
            1e3 * profile[1] / steps, 1e3 * profile[2] / steps, &produced[..produced.len().min(16)]);
    }
    let loads = match engine.experts() {
        Some(engine::Experts::Local(local)) => format!(", {} FP8 expert layer loads", local.loads.borrow()),
        Some(engine::Experts::LocalExl3(local)) => format!(", {} EXL3 expert layer loads", local.loads.borrow()),
        _ => String::new(),
    };
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s{loads}");
    if let (Some(dir), Some(logits), true) = (&args.save_logits, &logits, args.nll) {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("tokens.bin"), tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())?;
        std::fs::write(dir.join("logits.bin"), logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    if let (Some(logits), false) = (&logits, args.golden.join("logits.bin").exists()) {
        if args.nll {
            // No golden logits (a token file alone): the engine's NLL of the text and a digest
            // of every logit (bitwise A/B between builds and runs).
            use std::hash::{Hash, Hasher};
            let mut digest = std::collections::hash_map::DefaultHasher::new();
            logits.iter().for_each(|v| v.to_bits().hash(&mut digest));
            let (_, _, _, nll, scored) = score(logits, logits, &tokens, 0, vocab);
            println!("prefill logits: {prefill} rows | mean NLL engine {:.4} | logits digest {:016x}",
                nll / scored.max(1) as f64, digest.finish());
        }
        return Ok(());
    }
    if let Some(logits) = logits {
        let golden = golden_logits()?;
        if args.nll {
            let (agree, next_ok, golden_next, nll, scored) = score(&logits, &golden, &tokens, 0, vocab);
            let (_, _, _, golden_nll, _) = score(&golden[..prefill * vocab], &golden, &tokens, 0, vocab);
            println!("prefill logits: top-1 agreement {:.1}% over {prefill} rows | next-token accuracy engine {:.1}% \
                golden {:.1}% | mean NLL engine {:.4} golden {:.4}", 100.0 * agree as f64 / prefill as f64,
                100.0 * next_ok as f64 / scored.max(1) as f64, 100.0 * golden_next as f64 / scored.max(1) as f64,
                nll / scored.max(1) as f64, golden_nll / scored.max(1) as f64);
            println!("prefill logits: mean KL(golden||engine) {:.5}", mean_kl(&logits, &golden, 0, vocab));
        }
        let logits = &logits[logits.len() - vocab..];
        let last = &golden[(prefill - 1) * vocab..][..vocab];
        let (cosine, _) = similarity(logits, last);
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(logits), argmax(last));
    }
    Ok(())
}

/// `--token-check N`: prefill the golden prompt, then [`crate::shared::token_io::gate`]
/// over N greedy decode steps.
fn token_check(args: &GoldenArgs, opened: &Opened, engine: &engine::GlmfEngine<'_>, steps: usize) -> Result<()> {
    let mut tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(p) = args.prefill {
        tokens.truncate(p);
    }
    let mut allocator = engine::Allocator::new(engine.pages, engine.slots);
    let mut placement = allocator.admit(tokens.len() + steps + 1)?;
    let mut last = None;
    for chunk in tokens.chunks(engine.prefill_capacity()) {
        last = engine.prefill_device(&mut placement, chunk)?;
    }
    let last = last.ok_or_else(|| anyhow::anyhow!("--token-check needs every layer"))?.row_host(&opened.library, 0)?;
    let first = cuteafd_core::TargetSamplingParams::greedy().select_token(&last, None, 0)? as u32;
    let result = crate::shared::token_io::gate(&opened.library, &engine.embedding, first, steps, |token| {
        engine.verify_device(&mut [(&mut placement, 1)], &[token], false)?
            .ok_or_else(|| anyhow::anyhow!("decode needs every layer"))
    });
    allocator.release(placement);
    result
}
