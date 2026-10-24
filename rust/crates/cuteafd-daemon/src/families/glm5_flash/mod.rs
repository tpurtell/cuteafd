//! GLM 5.3 Flash (glm5_next) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod dspark;
mod remote_dflash;
pub(crate) use crate::shared::draft::resource as draft_binding;
use crate::shared::draft::probe as draft_probe;
pub(crate) mod engine;
pub(crate) mod fp8;
pub(crate) mod prefix;
pub(crate) mod serve;
mod media;
mod speculate;
mod expert_rows;
mod unit_check;
mod packed_check;
pub(crate) mod packing;
mod header;
pub(crate) mod head;
mod precision;
pub(crate) mod weights;

/// Whether this launch's coordinator operands can join a push group: every
/// representation the push path records (BF16 operands, GPU FP8 copies,
/// native E4M3 blocks, token-row output shards). The NVFP4 numerics gate
/// (a measurement mode) loads through the mount.
fn dense_pushable(args: &EngineArgs) -> bool {
    args.kda_nvfp4_gate.is_none()
}

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
    /// Shared GQA drafter-ring storage (independent of weight precision).
    #[arg(long, default_value = "fp8")]
    pub draft_kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache,
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
    /// Router replicas and admitted routed expert TP2 halves run on both GPUs;
    /// LM head and drafter stay on --device.
    #[arg(long)]
    pub split_device: Option<i32>,
    /// MLA attention over the two GPUs of a head split: auto (context where this build carries its
    /// programs), context (records and pool keys by 256-token unit on one GPU each, a q /
    /// candidate / partial exchange per MLA layer, the compact index cache), or heads (replicated
    /// latent and token keys, head-split attention), or explicit layers (complete MLA operands
    /// and caches on alternating owners, split FFNs). KDA layers are always head-split.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto")]
    pub attention_placement: String,
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
    #[arg(long, default_value_t = 4096)]
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
    /// Numerical decode/verify reduction rows, independent of execution capacity.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLAN_ROWS")]
    pub attention_plan_rows: Option<usize>,
    /// Maximum live prefill units on one Spark transport (1 to 4; mixed layer ownership caps
    /// this at 2). Units retain their own operands and share attention temporaries; long
    /// spans reuse slots continuously. --prefill-rows bounds each unit's rows.
    #[arg(long = "prefill-waves", default_value_t = engine::DEFAULT_PREFILL_WAVES,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..=engine::MAX_PREFILL_WAVES as u64))]
    pub prefill_waves: usize,
    /// Maximum rows of a continuous prefill call; units recycle the admitted slots.
    #[arg(long, default_value_t = 16384,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    pub prefill_span_rows: usize,
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
    /// fails. Supported on one GPU with Spark experts, with either an automatic or an explicit
    /// pool; the step workspaces precede the KV pool.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_REPLAY_RECORDS", default_value = "own")]
    pub replay_records: engine::ReplayRecords,
    /// Spark ranks in TP order (HOST:PORT,...) serving the fp8 expert family.
    #[arg(long)]
    pub peers: Option<String>,
    /// Run the routed experts on this GPU: EXL3 checkpoints through the
    /// coordinator `exl3-glmf-k<tiers>/rtx-tp1` package, FP8 ones through the
    /// TP1 `fp8-glmf` package. `--experts-snapshot` names another checkpoint
    /// for the experts (the official FP8 one with EXL3 coordinator weights).
    #[arg(long)]
    pub local_experts: bool,
    /// Routed layer onboarding: auto (pool first), max, N, N%, or all.
    #[arg(long, env = "RTX_EXPERT_LAYERS", value_parser = parse_onboard)]
    pub rtx_expert_layers: Option<cuteafd_loader::placement::Onboard>,
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
    /// BF16 or single-copy FP8 KDA projections on one GPU (no --split-device).
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
    /// The target's logits through the BF16 head (the shared route, `crate::shared::target_head`):
    /// `tensor`, a BF16 tensor-core GEMM with FP32 accumulation at every row count, or `exact`
    /// (diagnostic), the pedantic FP32 cuBLAS GEMM on CUDA cores. Unset: CUTEAFD_TARGET_HEAD
    /// (tensor by default). Unset with --fp8-head, the FP8 head's own program runs; an explicit
    /// `tensor` with --fp8-head is refused.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_TARGET_HEAD")]
    pub target_head: Option<head::TargetHead>,
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
    /// Diagnostic drafter head arithmetic for golden/replay and --draft-modes: `exact` uses
    /// the target's FP32 path; `tensor` uses the legacy tensor route past one draft block.
    /// Serving always uses the shared BF16 tensor head at every row count. The target's own
    /// arithmetic and the FP8 head (--fp8-head) remain unchanged.
    #[arg(long, value_enum, env = "CUTEAFD_GLMF_DRAFT_HEAD", default_value = "exact")]
    pub draft_head: crate::shared::speculation::dflash::DraftHead,
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
    names.push(format!("glmf_{}", if args.kda_fp8 == fp8::KdaFp8::Off { args.kda_state.program(&cap) }
        else { args.kda_state.w8_program(&cap) }));
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

/// One startup owner, before TP2 diagnostics and the serving/golden body.
fn with_startup_graphs<T>(tp2: bool, serving: Option<(usize, bool)>, golden: (usize, bool),
    warm: impl FnOnce(usize, bool) -> Result<()>, check_tp2: impl FnOnce() -> Result<()>,
    body: impl FnOnce() -> Result<T>) -> Result<T> {
    if let Some((sequences, speculation)) = serving.or(tp2.then_some(golden)) {
        warm(sequences, speculation)?;
    }
    if tp2 { check_tp2()?; }
    body()
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
    fn serving_startup_seals_once_for_tp2_and_spark_layouts() {
        use std::cell::{Cell, RefCell};
        use crate::shared::decode_graph::{GraphBank, GraphPolicy};
        use cuteafd_loader::serving_capacity::glmf_graphs::{startup_set, DecodeBuckets};

        for (name, tp2, spark_ranks) in [("tp2-zero-spark", true, 0),
            ("tp2-spark-suffix", true, 4), ("spark-only", false, 4)] {
            let startup = startup_set(262144, 262144, 2048, 8, true, 45, true, &DecodeBuckets::new(64));
            let banks = RefCell::new([GraphBank::with_policy(GraphPolicy::FixedStartup),
                GraphBank::with_policy(GraphPolicy::FixedStartup)]);
            let seals = Cell::new([0; 2]);
            let events = RefCell::new(Vec::new());
            // The production with_engine sequence, with CPU executable stand-ins.
            with_startup_graphs(tp2, Some((8, true)), (16, false), |sequences, speculation| {
                assert_eq!((sequences, speculation), (8, true));
                events.borrow_mut().push("warm");
                for (rank, bank) in banks.borrow_mut().iter_mut().enumerate() {
                    let keys: Vec<_> = startup.keys.iter().enumerate().flat_map(|(shape, _)|
                        (0..if rank == 0 { 46 } else { 45 }).map(move |segment| (shape, segment))).collect();
                    for key in &keys { bank.insert(*key, (), None); }
                    bank.seal_startup(&keys)?;
                    let mut counts = seals.get(); counts[rank] += 1; seals.set(counts);
                }
                Ok(())
            }, || {
                assert_eq!(seals.get(), [1, 1]);
                events.borrow_mut().push("tp2-check");
                Ok(())
            }, || {
                assert_eq!(seals.get(), [1, 1], "{name}, {spark_ranks} Sparks");
                events.borrow_mut().push("serve");
                Ok(())
            }).unwrap();
            assert_eq!(seals.get(), [1, 1]);
            assert_eq!(*events.borrow(), if tp2 { vec!["warm", "tp2-check", "serve"] }
                else { vec!["warm", "serve"] });
        }
        let serve = include_str!("serve.rs");
        assert!(!serve.contains("engine.warm_decode_graphs("), "serving must not own a second warm-up");
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

    #[test]
    fn coordinator_push_admits_primary_weights_and_token_row_sharding_but_not_the_nvfp4_gate() {
        assert!(dense_pushable(&parse(&[])));
        assert!(dense_pushable(&parse(&["--kda-fp8", "row128", "--kda-output-shard"])));
        assert!(!dense_pushable(&parse(&["--kda-fp8", "off", "--kda-nvfp4-gate", "rtn"])));
    }

    #[test]
    fn attention_plan_override_has_one_domain_for_narrow_and_wide_execution() {
        use cuteafd_core::coordinator_programs::AttentionReductionPlan;
        for execution in ["64", "128"] {
            let args = parse(&["--decode-rows", execution, "--attention-plan-rows", "8"]);
            assert_eq!(args.attention_plan_rows, Some(8));
            assert_eq!(AttentionReductionPlan::new(engine::DECODE_ROWS, Some(7), args.attention_plan_rows)
                .unwrap().rows(), 8);
            for rows in [0, 65, 128] {
                assert!(AttentionReductionPlan::new(engine::DECODE_ROWS, Some(7), Some(rows)).is_err());
            }
        }
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
        use crate::shared::speculation::dflash::DraftHead;
        assert_eq!(parse(&[]).draft_head, DraftHead::Exact);
        assert_eq!(parse(&["--draft-head", "exact"]).draft_head, DraftHead::Exact);
        assert_eq!(parse(&["--draft-head", "tensor"]).draft_head, DraftHead::Tensor);
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--draft-head", "fp8"]).is_err());
    }

    #[test]
    fn attention_selectors_fail_closed_before_checkpoint_or_cuda() {
        for mode in ["auto", "heads"] {
            for peer in [false, true] {
                let mut args = parse(&["--attention-placement", mode]);
                args.split_device = peer.then_some(1);
                check_options(&args).unwrap();
            }
        }
        // context and explicit owned layers need two GPUs; unknown modes are refused.
        let mut args = parse(&["--attention-placement", "context"]);
        args.split_device = Some(1);
        check_options(&args).unwrap();
        args.prefix_marks = prefix::PrefixMarks::Pool;
        assert!(check_options(&args).unwrap_err().to_string().contains("prefix marks"));
        let mut layers = parse(&["--attention-placement", "layers"]);
        layers.split_device = Some(1);
        check_options(&layers).unwrap();
        for (mode, peers) in [("context", &[false][..]), ("layers", &[false]), ("unknown", &[false, true])] {
            for &peer in peers {
                let mut args = parse(&["--attention-placement", mode]);
                args.split_device = peer.then_some(1);
                assert!(check_options(&args).unwrap_err().to_string().contains("attention placement"));
            }
        }
    }

    #[test]
    fn runtime_host_embedding_inventory_preserves_target_head_and_peer_weights() {
        use crate::shared::token_io::EmbedPlacement;
        use cuteafd_loader::families::glm5_flash::resident::GlmfRepresentation;
        use cuteafd_loader::plan::checkpoint::CheckpointTensor;
        let config = cuteafd_loader::plan::testing::glm5_flash_config(2);
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let mut offset = 0;
        let tensors = cuteafd_loader::plan::testing::glm5_flash_tensors(&config)
            .into_iter().map(|(name, dtype, shape)| {
                let (dtype, width) = match dtype {
                    "BF16" => (cuteafd_core::DType::Bf16, 2),
                    "F32" => (cuteafd_core::DType::F32, 4),
                    "F8_E4M3" => (cuteafd_core::DType::F8E4M3, 1),
                    other => panic!("unsupported fixture dtype {other}"),
                };
                let byte_length = shape.iter().product::<usize>() as u64 * width;
                let tensor = CheckpointTensor { shard: "fixture".into(),
                    meta: cuteafd_loader::SafetensorsTensorMetadata { name,
                        dtype, shape, byte_length, byte_offset: offset } };
                offset += byte_length;
                tensor
            }).collect();
        let mut checkpoint = Checkpoint { snapshot: "fixture".into(), config, quantize_config: None,
            weight_map: Default::default(), tensors, missing_shards: vec![], shard_bytes: 0 };
        for ranks in [1, 2] {
            for fp8_head in [false, true] {
                let representation = GlmfRepresentation { fp8_head, ..Default::default() };
                let gpu = runtime_resident_weights(&checkpoint, &cfg, cfg.layers, ranks, representation, EmbedPlacement::Gpu, Default::default()).unwrap();
                let host = runtime_resident_weights(&checkpoint, &cfg, cfg.layers, ranks, representation, EmbedPlacement::Host, Default::default()).unwrap();
                assert_eq!((gpu[0].embedding, host[0].embedding), (524_288, 0));
                assert!(gpu[0].weights > 0);
                assert_eq!(gpu.iter().map(|r| r.weights).collect::<Vec<_>>(), host.iter().map(|r| r.weights).collect::<Vec<_>>());
                if ranks == 2 { assert_eq!(host[1].embedding, 0); assert!(host[1].weights > 0); }
            }
        }
        for first_owner in [0, 1] {
            let attention = engine::GlmfAttention { mode: cuteafd_loader::placement::AttentionPlacement::Layers,
                first_owner, ..Default::default() };
            let actual = runtime_resident_weights(&checkpoint, &cfg, cfg.layers, 2,
                Default::default(), EmbedPlacement::Gpu, attention).unwrap();
            let expected = cuteafd_loader::families::glm5_flash::resident::resident_weights_placed(
                &checkpoint, &cfg, cfg.layers, 2, Default::default(), attention.mode, first_owner as usize).unwrap();
            assert_eq!(actual.iter().map(|r| (r.embedding, r.weights)).collect::<Vec<_>>(),
                expected.iter().map(|r| (r.embedding, r.weights)).collect::<Vec<_>>());
        }
        checkpoint.config["tie_word_embeddings"] = serde_json::json!(true);
        let error = runtime_resident_weights(&checkpoint, &cfg, 0, 2, Default::default(), EmbedPlacement::Host, Default::default()).unwrap_err();
        assert!(error.to_string().contains("tie_word_embeddings=true"));
        assert!(runtime_resident_weights(&checkpoint, &cfg, 0, 2, Default::default(), EmbedPlacement::Gpu, Default::default()).is_ok());
        checkpoint.config["tie_word_embeddings"] = serde_json::json!(false);
        checkpoint.tensors[2].meta.byte_offset = 0;
        let error = runtime_resident_weights(&checkpoint, &cfg, 0, 2, Default::default(), EmbedPlacement::Host, Default::default()).unwrap_err();
        assert!(error.to_string().contains("aliases its storage"));
    }

    #[test]
    fn served_index_cache_preserves_whole_owner_compact_state_without_changing_heads() {
        use cuteafd_loader::placement::AttentionPlacement::{Context, Heads, Layers};
        use engine::IndexCache::{Compact, Keys};
        for requested in [Compact, Keys] {
            assert_eq!(served_index_cache(requested, true, Heads), Keys);
            assert_eq!(served_index_cache(requested, true, Context), Compact);
            assert_eq!(served_index_cache(requested, true, Layers), requested);
            assert_eq!(served_index_cache(requested, false, Heads), requested);
        }
    }

    /// `--target-head`: unset follows the shared setting (tensor by default) and stays inert over
    /// the FP8 head; exact on request; an explicit tensor with the FP8 head is refused.
    #[test]
    fn target_head_defaults_to_the_shared_route_and_refuses_explicit_tensor_with_fp8() {
        use head::TargetHead::{Exact, Tensor};
        let unset = parse(&[]);
        assert_eq!(unset.target_head, None);
        assert_eq!(unset.resolved_target_head(), head::TargetHead::configured());
        check_options(&unset).unwrap();
        let fp8 = parse(&["--fp8-head"]);
        assert_eq!(fp8.target_head, None);
        check_options(&fp8).unwrap();
        assert_eq!(parse(&["--target-head", "exact"]).resolved_target_head(), Exact);
        let tensor = parse(&["--target-head", "tensor"]);
        assert_eq!(tensor.resolved_target_head(), Tensor);
        check_options(&tensor).unwrap();
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--target-head", "fp8"]).is_err());
        let error = check_options(&parse(&["--target-head", "tensor", "--fp8-head"])).unwrap_err().to_string();
        assert!(error.contains("--target-head tensor") && error.contains("--fp8-head"), "{error}");
        check_options(&parse(&["--target-head", "exact", "--fp8-head"])).unwrap();
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
    fn bf16_kda_state_runs_either_projection_storage_on_one_gpu() {
        assert_eq!(parse(&[]).kda_state, engine::KdaState::F32);
        for (value, state) in [("f32", engine::KdaState::F32), ("bf16", engine::KdaState::Bf16),
            ("bf16-tile", engine::KdaState::Bf16Tile)] {
            let args = parse(&["--kda-state", value]);
            assert_eq!((args.kda_state, args.kda_state.name()), (state, value));
            check_options(&args).unwrap();
        }
        for extra in [&["--kda-state", "bf16", "--kda-fp8", "row128"][..],
            &["--kda-state", "bf16-tile", "--kda-fp8", "channel"][..]] {
            check_options(&parse(extra)).unwrap();
        }
        for extra in [&["--kda-state", "bf16", "--split-device", "1"][..],
            &["--kda-state", "bf16", "--kda-fp8", "row128", "--split-device", "1"][..]] {
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
            with(&["glmf_kda_w8_m128", "glmf_kda_commit_m128"]));
        assert_eq!(wide_decode_programs(&parse(&["--decode-rows", "128", "--kda-state", "bf16", "--index-cache",
            "compact"]), &cfg), with(&["glmf_kda_s16_m128", "glmf_index_producer_c_m128", "glmf_kda_commit_c_s16_m128"]));
        // Every name is one of the exporter's 16 wide stems.
        let stems = ["index_producer", "index_producer_c", "index_topk_decode", "mhc_post_pre", "kda", "kda_w8",
            "kda_s16", "kda_w8_s16", "mla_producer", "o", "sparse_mla_decode", "ffn_i2048", "ffn_i12288", "kda_commit",
            "kda_commit_s16", "kda_commit_c", "kda_commit_c_s16"].map(|stem| format!("glmf_{stem}_m128"));
        for flags in [&["--kda-fp8", "row128", "--index-cache", "compact"][..], &["--kda-state", "bf16-tile"][..],
            &["--kda-state", "bf16", "--kda-fp8", "row128"][..]] {
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
        assert_eq!(wide_intake_bytes(&parse(&["--prefill-rows", "64", "--decode-rows", "128", "--prefill-waves", "4"]),
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
        let (dense, experts) = (body("load_dense"), body("experts_for_layers"));
        let sized = ["args.expert", "_rows()"].concat();
        assert!(dense.contains(&["DenseNvfp4::load(&self.library, &directory, &self.cfg, ", &sized, ")?"].concat()));
        // FP8 experts, EXL3 experts and the Spark transports, each with the rows among its arguments.
        for site in ["Fp8Experts::load(", "max_rows: ", "SparkLink::new_waves("] {
            let at = experts.find(site).unwrap_or_else(|| panic!("{site}"));
            let arguments: String = experts[at..].chars().take(200).collect();
            assert!(arguments.contains(&sized), "{site}");
        }
        assert_eq!(experts.matches(&sized).count(), 4);
        let lane = ["args.prefill", "_rows"].concat();
        assert!(!dense.contains(&lane));
        // Only the intake plane uses the unit width; all expert math holds expert_rows.
        assert_eq!(experts.matches(&lane).count(), 1);
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

    const SPARKS: [&str; 2] = ["--peers", "127.0.0.1:19441"];

    #[test]
    fn shared_admission_preserves_headroom_and_onboarding_options() {
        assert_eq!(parse(&[]).headroom_bytes().unwrap(),
            cuteafd_loader::plan::layout::LayoutOptions::default().headroom_bytes);
        assert_eq!(parse(&["--headroom-gib", "1"]).headroom_bytes().unwrap(), 1 << 30);
        assert!(parse(&["--headroom-gib=-1"]).headroom_bytes().is_err());
        for (option, expected) in [("auto", "auto"), ("all", "100%"), ("3", "3"), ("50%", "50%")] {
            assert_eq!(parse(&["--rtx-expert-layers", option]).rtx_expert_layers.unwrap().to_string(), expected);
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
        assert_eq!((defaults.prefill_waves, defaults.prefill_rows), (2, 4096));
        let four = parse(&["--prefill-waves", "4", "--prefill-rows", "2048"]);
        assert_eq!((four.prefill_waves, four.prefill_rows), (4, 2048));
        assert_eq!(parse(&["--prefill-rows", "2048"]).prefill_rows, 2048);
        for lanes in ["0", "5"] {
            assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                "--prefill-waves", lanes]).is_err());
        }
    }

    #[test]
    fn prefill_span_is_an_explicit_positive_engine_flag() {
        assert_eq!(parse(&[]).prefill_span_rows, 16384);
        assert_eq!(parse(&["--prefill-span-rows", "32768"]).prefill_span_rows, 32768);
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--prefill-span-rows", "0"]).is_err());
    }

    #[test]
    fn solved_spark_selection_excludes_dense_and_local_layers_and_keeps_absolute_ids() {
        use cuteafd_loader::placement::{ExpertHome, LayerAssignment, LayerMode};
        let mut cfg = GlmNextConfig::from_hf(&cuteafd_loader::plan::testing::glm5_flash_config(2)).unwrap();
        cfg.layers = 45;
        cfg.dense = (0..45).map(|layer| layer < 3).collect();
        let mut assignments = vec![LayerAssignment { mode: LayerMode::HeadSplit, experts: ExpertHome::Spark }; 45];
        for assigned in &mut assignments[3..24] { assigned.experts = ExpertHome::RtxTp2; }
        assert_eq!(admitted_spark_layers(&cfg, &assignments, false).unwrap(), (24..45).collect::<Vec<_>>());
        assert!(admitted_spark_layers(&cfg, &assignments, true).unwrap().is_empty());
        for assigned in &mut assignments[24..] { assigned.experts = ExpertHome::RtxWhole { gpu: 0 }; }
        assert!(admitted_spark_layers(&cfg, &assignments, false).unwrap().is_empty());
        assert!(admitted_spark_layers(&cfg, &assignments[..3], false).unwrap().is_empty());
        assignments[24].experts = ExpertHome::Spark;
        assignments[26].experts = ExpertHome::Spark;
        assert_eq!(admitted_spark_layers(&cfg, &assignments, false).unwrap(), vec![24, 26]);
        assert_eq!(admitted_expert_layers(&cfg, &assignments, ExpertHome::RtxTp2, false).unwrap(),
            (3..24).collect::<Vec<_>>());
        assert_eq!(admitted_expert_layers(&cfg, &assignments, ExpertHome::RtxWhole { gpu: 0 }, false).unwrap(),
            (25..45).filter(|&layer| layer != 26).collect::<Vec<_>>());
        for invalid in [&[24, 24][..], &[26, 24], &[2], &[45]] {
            assert!(validate_expert_layers(&cfg, 45, invalid).is_err());
        }
        assert!(validate_expert_layers(&cfg, 26, &[24, 26]).is_err());
        validate_expert_layers(&cfg, 45, &[24, 26]).unwrap();
        assignments.push(assignments[0]);
        assert!(admitted_spark_layers(&cfg, &assignments, false).is_err());
    }

    #[test]
    fn step_experts_ignore_empty_rtx_ranges() -> Result<()> {
        use cuteafd_loader::placement::{AttentionPlacement, ExpertHome, ExpertRange, FfnMode,
            LayerAssignment, LayerMode, Placement, PoolLimits};
        let mut cfg = GlmNextConfig::from_hf(&cuteafd_loader::plan::testing::glm5_flash_config(45))?;
        cfg.dense = (0..45).map(|layer| layer < 3).collect();
        // The small-budget NVFP4 placement retains one empty range per RTX GPU.
        let mut placement = Placement {
            pool_tokens: 1 << 20, pool_limits: PoolLimits(vec![]), attention_placement: AttentionPlacement::Heads,
            attention_by_kind: vec![], peer_row_bytes: 0, onboard_layers: 0,
            layers: (0..45).map(|layer| LayerAssignment {
                mode: LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
                experts: if cfg.dense[layer] { ExpertHome::Dense } else { ExpertHome::Spark },
            }).collect(),
            movables: vec![], expert_ranges: vec![ExpertRange { first: 3, layers: 0, peak_bytes: 0 }],
            residual: vec![], hops: vec![], tp2: None, items: vec![],
        };
        assert_eq!(admitted_step_experts(&cfg, &placement, false, true)?,
            engine::StepExperts { local_fp8: false, spark: true, tp2: false });
        // Local buffers follow actual layer homes, not the catalog's format alone.
        placement.layers[3].experts = ExpertHome::RtxWhole { gpu: 0 };
        placement.expert_ranges[0].layers = 1;
        assert_eq!(admitted_step_experts(&cfg, &placement, false, true)?,
            engine::StepExperts { local_fp8: true, spark: true, tp2: false });
        assert_eq!(admitted_step_experts(&cfg, &placement, false, false)?,
            engine::StepExperts { local_fp8: false, spark: true, tp2: false });
        placement.layers[3].experts = ExpertHome::RtxTp2;
        assert_eq!(admitted_step_experts(&cfg, &placement, false, true)?,
            engine::StepExperts { local_fp8: false, spark: true, tp2: true });
        assert_eq!(admitted_step_experts(&cfg, &placement, true, true)?,
            engine::StepExperts { local_fp8: false, spark: false, tp2: false });
        assert_eq!(admitted_step_experts(&cfg, &Placement { layers: placement.layers[..3].to_vec(), ..placement }, false, true)?,
            engine::StepExperts { local_fp8: false, spark: false, tp2: false });
        Ok(())
    }

    #[test]
    fn sparse_admission_partitions_backends_without_loading_gaps() -> Result<()> {
        use crate::shared::placement::layers::{BackendLayers, ExpertRouting};
        use cuteafd_loader::placement::{ExpertHome, LayerAssignment, LayerMode};
        let mut cfg = GlmNextConfig::from_hf(&cuteafd_loader::plan::testing::glm5_flash_config(18))?;
        cfg.dense = (0..18).map(|layer| layer < 3).collect();
        let mut assignments = vec![LayerAssignment { mode: LayerMode::HeadSplit, experts: ExpertHome::Spark }; 18];
        let local_home = ExpertHome::RtxWhole { gpu: 0 };
        for layer in [3, 7] { assignments[layer].experts = local_home; }
        for layer in [13, 15] { assignments[layer].experts = ExpertHome::RtxTp2; }
        let local = admitted_expert_layers(&cfg, &assignments, local_home, false)?;
        let tp2 = admitted_expert_layers(&cfg, &assignments, ExpertHome::RtxTp2, false)?;
        let spark = admitted_spark_layers(&cfg, &assignments, false)?;
        assert_eq!(local, vec![3, 7]);
        assert_eq!(tp2, vec![13, 15]);
        assert_eq!(spark, vec![4, 5, 6, 8, 9, 10, 11, 12, 14, 16, 17]);
        let bind = |tp2: &[usize]| ExpertRouting::bind(&assignments, &cfg.dense, &[
            BackendLayers { home: local_home, layers: &local },
            BackendLayers { home: ExpertHome::RtxTp2, layers: tp2 },
            BackendLayers { home: ExpertHome::Spark, layers: &spark },
        ]);
        let routes = bind(&tp2)?;
        assert_ne!(routes.route(13), routes.route(14));
        assert_eq!(routes.route(13), routes.route(15));
        assert!(bind(&[13]).is_err());
        assert!(bind(&[13, 14, 15]).is_err());
        for home in [local_home, ExpertHome::RtxTp2, ExpertHome::Spark] {
            assert!(admitted_expert_layers(&cfg, &assignments, home, true)?.is_empty());
        }
        Ok(())
    }

    /// Local onboarding and Spark transports coexist: the admitted layer home selects each.
    #[test]
    fn only_spark_experts_have_transports_to_warm() {
        assert!(parse(&["--local-experts"]).peers.is_none());
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--local-experts", "--peers", "127.0.0.1:19441"]).is_ok());
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
    /// Directory with tokens.bin, layerNN.bin and logits.bin from the family reference;
    /// token-only diagnostics also accept the published Flash fidelity dataset's quick windows.
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
    /// Time this many more prefills of the golden prompt on fresh sequences,
    /// without layer downloads. With --draft, run after draft/decode and retain
    /// the drafter and its exchange state between requests.
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
    /// Run --replay-check through serving's captured graphs and padded row buckets,
    /// N may include copy-extension rows beyond the neural draft window.
    #[arg(long, requires = "replay_check")]
    pub replay_production: bool,
    /// Replay a one-request speculation JSONL trace round by round against serial decode.
    #[arg(long, requires = "replay_production")]
    pub replay_trace: Option<PathBuf>,
    /// Private per-round logits/state/MLA dumps for same-width placement diagnostics.
    #[arg(long, requires = "replay_trace")]
    pub replay_dump: Option<PathBuf>,
    /// Continue private dump diagnostics across serial-geometry differences, never a correctness PASS.
    #[arg(long, requires = "replay_dump")]
    pub replay_dump_continue: bool,
    /// Diagnostic full-device drain before drafter launches and context updates.
    #[arg(long, requires = "replay_trace")]
    pub replay_draft_sync: bool,
    /// Require byte-exact real-row logits for speculative masked decode padding 3->4, 9->16, 17->32 and 33->64.
    #[arg(long)]
    pub padding_check: bool,
    /// Compare queued multi-chunk prefill and decode replay with a drained reference, byte-for-byte.
    #[arg(long)]
    pub upload_check: bool,
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
    /// Compare executor N=1 and N=2 at identical row cuts, including recycled slots:
    /// every row's logits, KDA state and paged bytes must be identical.
    #[arg(long)]
    pub unit_check: bool,
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

#[cfg(test)]
mod replay_cli_tests {
    use super::GoldenArgs;
    use clap::Parser;

    #[derive(Parser)]
    struct Parse {
        #[command(flatten)]
        args: GoldenArgs,
    }

    #[test]
    fn unit_identity_is_a_golden_option_not_an_engine_key() {
        #[derive(Parser)]
        struct EngineOnly {
            #[command(flatten)]
            args: super::EngineArgs,
        }
        let golden = ["test", "--snapshot", "/checkpoint", "--native-lib", "/native", "--golden", "/golden"];
        assert!(Parse::try_parse_from(golden.into_iter().chain(["--unit-check"])).unwrap().args.unit_check);
        assert!(Parse::try_parse_from(golden.into_iter().chain(["--lane-check"])).is_err());
        assert!(EngineOnly::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--unit-check"]).is_err());
    }

    #[test]
    fn production_replay_requires_a_replay_width() {
        let base = ["test", "--snapshot", "/checkpoint", "--native-lib", "/native", "--golden", "/golden"];
        assert!(Parse::try_parse_from(base.into_iter().chain(["--replay-production"])).is_err());
        let parsed = Parse::try_parse_from(base.into_iter()
            .chain(["--replay-production", "--replay-check", "3"])).unwrap();
        assert!(parsed.args.replay_production);
        assert_eq!(parsed.args.replay_check, Some(3));
        let eager = Parse::try_parse_from(base.into_iter().chain(["--replay-check", "9"])).unwrap();
        assert!(!eager.args.replay_production);
        assert_eq!(eager.args.replay_check, Some(9));
        assert!(Parse::try_parse_from(base.into_iter().chain(["--replay-check", "9", "--replay-trace", "/trace"])).is_err());
        let trace = Parse::try_parse_from(base.into_iter().chain(["--replay-check", "9", "--replay-production",
            "--replay-trace", "/trace"])).unwrap();
        assert_eq!(trace.args.replay_trace.as_deref(), Some(std::path::Path::new("/trace")));
        assert!(trace.args.replay_dump.is_none() && !trace.args.replay_draft_sync && !trace.args.replay_dump_continue);
        for option in [vec!["--replay-dump", "/dump"], vec!["--replay-draft-sync"], vec!["--replay-dump-continue"]] {
            assert!(Parse::try_parse_from(base.into_iter().chain(option)).is_err());
        }
        let diagnostic = Parse::try_parse_from(base.into_iter().chain(["--replay-check", "6",
            "--replay-production", "--replay-trace", "/trace", "--replay-dump", "/dump",
            "--replay-draft-sync"])).unwrap();
        assert!(diagnostic.args.replay_draft_sync);
        assert_eq!(diagnostic.args.replay_dump.as_deref(), Some(std::path::Path::new("/dump")));
        assert!(Parse::try_parse_from(base.into_iter().chain(["--replay-check", "6", "--replay-production",
            "--replay-trace", "/trace", "--replay-dump-continue"])).is_err());
        let continuation = Parse::try_parse_from(base.into_iter().chain(["--replay-check", "6",
            "--replay-production", "--replay-trace", "/trace", "--replay-dump", "/dump",
            "--replay-dump-continue"])).unwrap();
        assert!(continuation.args.replay_dump_continue);
    }
}

/// Option combinations rejected before any checkpoint or native work.
fn check_options(args: &EngineArgs) -> Result<()> {
    let requested = cuteafd_loader::placement::attention::parse(&args.attention_placement)
        .map_err(anyhow::Error::msg)?;
    let executor = cuteafd_loader::placement::families::executor("glm5_flash").context("GLM Flash executor")?;
    executor.check_attention(requested, 1 + usize::from(args.split_device.is_some()), args.split_device.is_some())?;
    ensure!(requested != Some(cuteafd_loader::placement::AttentionPlacement::Context)
        || args.prefix_marks == prefix::PrefixMarks::Arena,
        "--attention-placement context keeps prefix marks in the arena (--prefix-marks arena)");
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
    ensure!(args.kda_state == engine::KdaState::F32 || args.split_device.is_none(),
        "--kda-state bf16 takes no --split-device (head-split BF16 state is not yet qualified)");
    ensure!(args.replay_records == engine::ReplayRecords::Own || args.split_device.is_none(),
        "--replay-records shared keeps the records in one GPU's prefill scratch: it takes no --split-device");
    ensure!(args.decode_rows == engine::DECODE_ROWS || args.split_device.is_none(),
        "--decode-rows {} runs the wide decode programs on one GPU: a head split (--split-device) takes \
        --decode-rows {}", args.decode_rows, engine::DECODE_ROWS);
    ensure!(args.target_head != Some(head::TargetHead::Tensor) || !args.fp8_head,
        "--target-head tensor runs the BF16 head; the FP8 head (--fp8-head) has its own program");
    Ok(())
}

impl EngineArgs {
    /// The target head's mode: explicit `--target-head`, else the shared setting (tensor by
    /// default). The FP8 head's own program ignores it.
    pub fn resolved_target_head(&self) -> head::TargetHead {
        head::resolve(self.target_head)
    }

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
    use cuteafd_loader::serving_capacity::spark_intake_bytes;
    let at = |rows: usize| spark_intake_bytes(args.prefill_waves as u64, ranks as u64, rows as u64, hidden as u64);
    at(args.expert_rows()) - at(args.prefill_rows)
}

/// The step settings these arguments give the engine (its step plan's inputs besides layers and experts),
/// with the DSA index cache `with_engine` resolved (a head split keeps `keys`).
fn step_settings(args: &EngineArgs, index_cache: engine::IndexCache, spark_bf16: bool) -> engine::StepSettings {
    engine::StepSettings { spark_bf16, kda_fp32_partials: args.kda_fp32_partials, kda_output_shard: args.kda_output_shard,
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

fn parse_onboard(text: &str) -> std::result::Result<cuteafd_loader::placement::Onboard, String> {
    text.parse()
}

fn runtime_resident_weights(checkpoint: &Checkpoint, cfg: &GlmNextConfig, layers: usize, ranks: usize,
    representation: cuteafd_loader::families::glm5_flash::resident::GlmfRepresentation,
    embedding: crate::shared::token_io::EmbedPlacement, attention: engine::GlmfAttention)
    -> Result<Vec<cuteafd_loader::families::glm5_flash::resident::GlmfResidentRank>> {
    let mut resident = cuteafd_loader::families::glm5_flash::resident::resident_weights_placed(
        checkpoint, cfg, layers, ranks, representation, attention.mode, attention.first_owner as usize).map_err(anyhow::Error::msg)?;
    if embedding == crate::shared::token_io::EmbedPlacement::Host {
        checkpoint.require_untied_embedding("model.language_model.embed_tokens.weight")?;
        resident[0].embedding = 0;
    }
    Ok(resident)
}

fn validate_expert_layers(cfg: &GlmNextConfig, served: usize, layers: &[usize]) -> Result<()> {
    ensure!(served <= cfg.layers && cfg.dense.len() == cfg.layers
        && layers.windows(2).all(|ids| ids[0] < ids[1])
        && layers.iter().all(|&layer| layer < served && !cfg.dense[layer]),
        "GLM Flash expert inventory must contain ordered unique served routed layers: {layers:?}");
    Ok(())
}

fn admitted_expert_layers(cfg: &GlmNextConfig, assignments: &[cuteafd_loader::placement::LayerAssignment],
    home: cuteafd_loader::placement::ExpertHome, skip: bool) -> Result<Vec<usize>> {
    ensure!(assignments.len() <= cfg.layers && cfg.dense.len() == cfg.layers,
        "GLM Flash expert selection has invalid layer bounds");
    // Dense assignments may carry the solver's default Spark home but own no routed experts.
    let selected = assignments.iter().enumerate().filter_map(|(layer, assigned)|
        (!skip && !cfg.dense[layer] && assigned.experts == home).then_some(layer)).collect::<Vec<_>>();
    validate_expert_layers(cfg, assignments.len(), &selected)?;
    Ok(selected)
}

fn admitted_spark_layers(cfg: &GlmNextConfig, layers: &[cuteafd_loader::placement::LayerAssignment],
    skip: bool) -> Result<Vec<usize>> {
    admitted_expert_layers(cfg, layers, cuteafd_loader::placement::ExpertHome::Spark, skip)
}

fn admitted_step_experts(cfg: &GlmNextConfig, placement: &cuteafd_loader::placement::Placement,
    skip: bool, fp8: bool) -> Result<engine::StepExperts> {
    Ok(engine::StepExperts {
        local_fp8: fp8 && !admitted_expert_layers(cfg, &placement.layers,
            cuteafd_loader::placement::ExpertHome::RtxWhole { gpu: 0 }, skip)?.is_empty(),
        spark: !admitted_spark_layers(cfg, &placement.layers, skip)?.is_empty(),
        tp2: !admitted_expert_layers(cfg, &placement.layers,
            cuteafd_loader::placement::ExpertHome::RtxTp2, skip)?.is_empty(),
    })
}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub checkpoint: Checkpoint,
    pub cfg: GlmNextConfig,
    pub library: std::sync::Arc<NativeLibrary>,
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
    let checkpoint = Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = GlmNextConfig::read(&args.snapshot)?;
    header::check_kda_inputs(&checkpoint, &cfg, args.layers.unwrap_or(cfg.layers))?;
    precision::check_projection_inputs(&checkpoint, &cfg,
        args.layers.unwrap_or(cfg.layers))?;
    if let Some(snapshot) = &args.draft {
        let head = checkpoint.tensors.iter().find(|t| t.meta.name == "lm_head.weight")
            .context("DFlash target has no lm_head.weight")?;
        // The drafter borrows the target's one head: BF16, or the FP8 head made from it.
        crate::shared::speculation::dflash::check_target_head_source(&head.meta, cfg.hidden, cfg.vocab_size)?;
        if dspark::is_dspark(snapshot)? {
            dspark::check_checkpoint(snapshot, cfg.hidden, cfg.vocab_size, cfg.layers)?;
        } else {
            crate::shared::speculation::dflash::check_checkpoint(snapshot, args.draft_fp8,
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
    let experts = if args.layers.unwrap_or(cfg.layers) > cfg.dense.iter().take_while(|&&d| d).count() {
        let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
        let mut catalog = cuteafd_loader::read_expert_catalog(source)?;
        catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
        ensure!(catalog.fp8().is_some() || catalog.exl3().is_some(),
            "--local-experts runs FP8 or EXL3 experts; {} has neither", source.display());
        Some(catalog)
    } else {
        None
    };
    // SAFETY: the library is the cuteafd native shim built for this engine.
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&args.native_lib) }?);
    library.cuda_set_device(args.device)?;
    Ok(Opened { checkpoint, cfg, library, experts })
}

/// Heads on two GPUs needs token keys; context needs compact pools. Whole MLA
/// owners can serve either cache, with compact tails only on the actual owner.
pub(crate) fn served_index_cache(requested: engine::IndexCache, split: bool,
    attention: cuteafd_loader::placement::AttentionPlacement) -> engine::IndexCache {
    use cuteafd_loader::placement::AttentionPlacement;
    match attention {
        AttentionPlacement::Context => engine::IndexCache::Compact,
        AttentionPlacement::Heads if split => engine::IndexCache::Keys,
        _ => requested,
    }
}

/// The MLA placement an engine serves (`glm5_flash::resolve_attention` over the requested mode and
/// the build's context programs), with its programs' scratch.
fn resolve_attention(args: &EngineArgs, programs: &cuteafd_ffi::programs::Programs<'_>, split: bool)
    -> Result<engine::GlmfAttention> {
    let requested = cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?;
    let names: Vec<&str> = programs.names().collect();
    let available = cuteafd_loader::placement::families::glm5_flash::context_programs(names.iter().copied(),
        |name| programs.spec(name).ok().map(|p| p.scratch.get("scratch").copied().unwrap_or(0)));
    let attention = cuteafd_loader::placement::families::glm5_flash::resolve_attention(requested,
        1 + usize::from(split), available)?;
    // Auto with pool marks keeps heads (context keeps marks in the arena).
    if requested.is_none() && args.prefix_marks == prefix::PrefixMarks::Pool {
        return Ok(engine::GlmfAttention { mode: cuteafd_loader::placement::AttentionPlacement::Heads, ..attention });
    }
    Ok(attention)
}

impl Opened {
    #[allow(clippy::too_many_arguments)]
    fn admit(&self, args: &EngineArgs, programs: &cuteafd_ffi::programs::Programs<'_>, layers: usize,
        index: engine::IndexCache, marks: usize, peer: Option<i32>, automatic_context: bool,
        prepared_draft: Option<&crate::shared::draft::remote::Prepared<'_>>, attention: engine::GlmfAttention)
        -> Result<(cuteafd_loader::placement::Placement, cuteafd_loader::placement::GraphSet,
            cuteafd_loader::placement::families::glm5_flash::GlmfWorkingSet)> {
        use cuteafd_loader::placement::{self, families::glm5_flash as admission};
        use cuteafd_loader::families::glm5_flash::resident::GlmfRepresentation;
        use cuteafd_loader::serving_capacity::{glmf_step_scratch, glmf_table_pages,
            GlmfScratchOptions, GlmfStepShape};
        ensure!(args.expert_window.is_none(), "solver admission does not support diagnostic expert paging");
        let devices: Vec<_> = std::iter::once(args.device).chain(peer).collect();
        let split = devices.len() == 2;
        let inventory = crate::shared::inventory::RuntimeInventory::measure(&self.library, &devices, crate::shared::inventory::InventoryPhase::PreWeight, |_| {
            programs.load_matching(|name| glmf_startup_program(name, split, args.decode_rows > engine::DECODE_ROWS))?;
            Ok(())
        })?;
        let local_tp2_exl3 = split && args.peers.is_none() && !args.skip_experts
            && self.experts.as_ref().is_some_and(|c| c.exl3().is_some());
        let code_package = if local_tp2_exl3 { "exl3-tp2-local" } else { "*" };
        let pending_code = devices.iter().enumerate().map(|(rank, &device)|
            crate::shared::inventory::pending_code(&self.library, device, rank, split, "glmf", code_package,
                &placement::ProgramSet::from_names(&["glmf", "glmf2"], programs.names())))
            .collect::<Result<Vec<_>>>()?;
        let headers = Checkpoint::coordinator(&args.snapshot, false, false)?;
        let representation = GlmfRepresentation { kda_fp8: args.kda_fp8 != fp8::KdaFp8::Off,
            fp8_head: args.fp8_head, output_shard: args.kda_output_shard };
        let resident = runtime_resident_weights(&headers, &self.cfg, layers, devices.len(), representation,
            args.token_io.embed_placement, attention)?;
        let alternate_resident = (attention.mode == placement::AttentionPlacement::Layers).then(||
            runtime_resident_weights(&headers, &self.cfg, layers, devices.len(), representation,
                args.token_io.embed_placement, engine::GlmfAttention { first_owner: 1 - attention.first_owner, ..attention }))
            .transpose()?;
        let spark_ranks = args.peers.as_deref().map_or(0, |p| p.split(',').count());
        let lanes = engine::configured_prefill_waves(layers == self.cfg.layers, args.prefill_waves);
        let lanes = usize::try_from(admission::admitted_prefill_waves(attention.mode, lanes as u64))?;
        let lookup = |name: &str| programs.spec(name).ok().map(|p| p.scratch.get("scratch").copied().unwrap_or(0));
        let scratch = GlmfScratchOptions { split, kda_w8: representation.kda_fp8,
            kda_fp32_partials: args.kda_fp32_partials, kda_output_shard: args.kda_output_shard,
            kda_prefill_expanded: args.kda_prefill_expanded, index_compact: index == engine::IndexCache::Compact,
            kda_state: args.kda_state.into() };
        let decode = glmf_step_scratch(lookup, &self.cfg, scratch, args.decode_rows as u64, true)?;
        let mut prefill = glmf_step_scratch(lookup, &self.cfg, scratch, args.prefill_rows as u64, false)?;
        if args.replay_records == engine::ReplayRecords::Shared {
            prefill.programs = prefill.programs.max(cuteafd_loader::serving_capacity::glm_flash_kda_replay_bytes_rows(
                &self.cfg, layers, devices.len(), args.decode_rows as u64)?);
        }
        let (table_pages, table_pool_pages) = glmf_table_pages(args.max_context as u64);
        let shape = GlmfStepShape { lead: true, split, local_experts: !args.skip_experts && !split && self.fp8().is_some(),
            tp2_experts: !args.skip_experts && split, spark: spark_ranks > 0, partial_bytes: if args.kda_fp32_partials { 4 } else { 2 },
            output_shard: args.kda_output_shard, full_prefill_logits: args.full_prefill_logits, table_pages, table_pool_pages };
        let workspaces = |spark, lanes| admission::step_workspace_demands(&self.cfg, lanes,
            args.prefill_rows as u64, args.decode_rows as u64, GlmfStepShape { spark, ..shape }, decode, prefill);
        let workspace = workspaces(spark_ranks > 0, lanes);
        let local_workspace = workspaces(false, lanes);
        let mut experts = self.experts.as_ref().map(|c| admission::expert_costs(c, split)).transpose()?.unwrap_or_default();
        experts.truncate(self.cfg.dense[..layers].iter().filter(|&&d| !d).count());
        if args.skip_experts { for cost in &mut experts { cost.whole = Default::default(); cost.half = Default::default(); } }
        let mut tp2_workspace = [0; 2];
        let mut expert_workspace = 0;
        if let Some(catalog) = self.experts.as_ref().filter(|_| !args.skip_experts) {
            if split {
                if let Some(tensors) = catalog.fp8() {
                    let package = args.fp8_package.clone().unwrap_or_else(|| crate::shared::experts::fp8::package_directory(
                        &args.native_lib, 2, tensors.format()));
                    for rank in 0..2 { tp2_workspace[rank] = crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::plan(
                        tensors, &package, rank, args.expert_rows())?.workspace_bytes as u64; }
                } else {
                    let exl3 = catalog.exl3().context("EXL3 inventory")?;
                    let package = crate::shared::experts::exl3::aot_layout_directory(&args.native_lib, exl3.decoder_tiers(), "rtx-tp2");
                    tp2_workspace = [crate::shared::experts::rtx::exl3::Exl3Tp2::workspace_bytes_for(&package,
                        self.cfg.hidden, args.expert_rows())? as u64; 2];
                }
            } else { expert_workspace = admission::expert_workspace(catalog, Some(&args.manifest), args.expert_rows() as u64, 1)?; }
        }
        let sms = inventory.gpus[0].sms as usize;
        let draft_sms: Vec<_> = inventory.gpus.iter().map(|gpu| gpu.sms as u64).collect();
        let (drafter_bytes, drafter_scratch, drafter_bridge) = args.draft.as_ref().map(|path| -> Result<_> {
            let config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
            let slots = args.draft_context_slots.unwrap_or(20.max(args.draft_sequences));
            let batch = args.draft_sequences.min(slots);
            let (bytes, scratch) = admission::drafter_inventory(&config, slots, batch, &draft_sms,
                cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::from_fp8_option(args.draft_fp8),
                args.draft_linear.code() as u8, args.draft_kv_cache)?;
            let bridge = if prepared_draft.is_some() { admission::drafter_bridge_inventory(&config, slots, batch)? }
                else { None };
            Ok((bytes, scratch, bridge))
        }).transpose()?.unwrap_or_else(|| (0, vec![0; devices.len()], None));
        let (sequences, speculation) = args.serving_graph_policy.unwrap_or((16, args.draft.is_some()));
        let pool = if args.pool_tokens > 0 { args.pool_tokens } else { cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS as usize };
        let graphs = if args.startup_graphs() {
            admission::startup_graphs(&self.cfg, layers, args.max_context, pool,
                sequences, speculation, devices.len(), args.decode_rows, sms)
        } else { placement::GraphSet::budget(&vec![args.graph_budget_mib.map_or(1536 << 20, |mib| mib << 20); devices.len()]) };
        let onboard = args.rtx_expert_layers.unwrap_or_else(|| if args.local_experts {
            placement::Onboard::Layers(experts.len()) } else { admission::default_onboard() });
        let inputs = admission::GlmfInputs { automatic_context: automatic_context, attention, cfg: &self.cfg, layers, local_tp2_exl3,
            gpus: inventory.gpus.iter().map(|gpu| (gpu.total_bytes, placement::Baseline::Measured {
                free_bytes: gpu.free_bytes + prepared_draft.map_or(0, crate::shared::draft::remote::Prepared::sampled_bytes) })).collect(),
            pending_code, headroom_bytes: args.headroom_bytes()?, spark_ranks,
            prefill_lanes: lanes as u64, prefill_rows: args.prefill_rows as u64, decode_rows: args.decode_rows as u64,
            partial_bytes: if args.kda_fp32_partials { 4 } else { 2 }, max_context: args.max_context as u64,
            sequences: sequences.min(args.decode_rows) as u64, speculation, state_slots: args.slots as u64,
            mark_slots: marks as u64, pool_marks: args.prefix_marks == prefix::PrefixMarks::Pool, index: index.into(),
            kda_state_bytes: args.kda_state.bytes() as u64, shared_replay: args.replay_records == engine::ReplayRecords::Shared,
            representation, resident, workspace, local_workspace, graphs, experts, expert_workspace, tp2_workspace,
            drafter_bytes, drafter_scratch, drafter_bridge, requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64), onboard,
            full_prefill_logits: 0 };
        let target = placement::PoolPolicy::resolve(&inputs.gpus.iter().map(|g| g.0).collect::<Vec<_>>(),
            args.max_context as u64, inputs.requested_pool, 256, spark_ranks == 0).target;
        let (placement, graphs, working) = admission::solve_attention_working_set_with_graphs(&inputs,
            alternate_resident.as_deref(), target, automatic_context, sms, true)?;
        let attention = admission::placed_attention(attention, &placement);
        ensure!(placement.movables.iter().all(|(id, gpu)| *gpu == 0 || *id == placement::MovableId::Drafter),
            "GLM Flash placement has no matching movable executor");
        placement::draft::require_home(&placement, "GLM Flash",
            if prepared_draft.is_some() { &[0, 1] } else { &[0] })?;
        tracing::info!(placement = %placement.summary(), spark_ranks = working.spark_ranks,
            prefill_lanes = working.prefill_lanes, spark_layers = ?working.spark_layers,
            attention_first_owner = attention.first_owner, "GLM Flash admission before weights");
        let devices: Vec<_> = placement.items.iter().zip(&inventory.gpus).enumerate().map(|(rank, (items, gpu))| {
            let mut items = items.clone();
            items.push(cuteafd_core::memory_layout::Item::new(
                cuteafd_core::memory_layout::Category::Runtime, "sampled context+modules", "",
                gpu.context_bytes, cuteafd_core::memory_layout::Basis::Calibrated));
            serde_json::json!({"kind": "rtx", "index": rank, "device": gpu.device, "items": items})
        }).collect();
        tracing::info!(target: "cuteafd::placement", plan = %serde_json::json!({
            "memory_layout": {"devices": devices, "pool_tokens": placement.pool_tokens},
            "graph_shapes": graphs.shapes, "graphs": graphs.ranks.iter().map(|r| r.executables).collect::<Vec<_>>(),
            "speculation": speculation, "sequences": sequences,
            "attention_first_owner": attention.first_owner,
            "layer_modes": placement.layers.iter().map(|layer| layer.mode).collect::<Vec<_>>(),
        }), "GLM Flash startup plan");
        cuteafd_bench::context::set_resolved("rtx-expert-layers", &placement.onboard_layers.to_string());
        Ok((placement, graphs, working))
    }

    /// Builds the engine and hands it to `body`. Either KV admission keeps the prefix marks of
    /// `args.mark_arena` the caller allocates once the engine exists free, counted on the layout
    /// the engine serves (`engine.mark_slots`).
    pub fn with_engine<T>(&self, args: &EngineArgs, body: impl FnOnce(&engine::GlmfEngine<'_>) -> Result<T>)
        -> Result<T> {
        // A head split whose GPUs lack peer access serves from --device alone, decided before any load or
        // admission. The arena is counted after both decisions (below), on the layout this engine serves:
        // each admitted GPU reserves its part of every mark (a lone GPU, the whole mark), as the prefix
        // cache then allocates.
        let probe_devices: Vec<_> = std::iter::once(args.device).chain(args.split_device).collect();
        let probe = |stage| crate::shared::inventory::startup_probe(&self.library, &probe_devices, stage);
        probe("contexts-before-peer")?;
        let mut resolved = args.clone();
        if args.split_device.is_some()
            && crate::shared::peer_split::probed_device(&self.library, args.device, args.split_device)?.is_none() {
            resolved.split_device = None;
            resolved.kda_fp32_partials = false;
            resolved.kda_output_shard = false;
            resolved.kda_prefill_expanded = false;
        }
        probe("peer-enabled")?;
        let args = &resolved;
        let automatic_context = args.max_context == 0;
        let mut context_args = args.clone();
        context_args.max_context = crate::shared::context::checkpoint_context(
            &args.snapshot, &args.manifest, "glm5_flash", args.max_context)?;
        let args = &context_args;
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let active_drafts = args.draft.as_deref().map(|path| -> Result<usize> {
            let (hidden, vocab, taps, drafts) = if dspark::is_dspark(path)? {
                let cfg = dspark::DsparkConfig::read(path)?;
                (cfg.hidden, cfg.vocab, cfg.taps.clone(), cfg.drafts())
            } else {
                let cfg = crate::shared::speculation::dflash::DflashConfig::read(path)?;
                (cfg.hidden, cfg.vocab, cfg.taps.clone(), cfg.drafts())
            };
            ensure!(hidden == self.cfg.hidden && vocab == self.cfg.vocab_size && taps.iter().all(|&l| l < layers),
                "the provisional attention drafter does not fit this target");
            Ok(drafts)
        }).transpose()?;
        // m64 and m128 are execution capacities of one numerical domain.
        let attention_plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
            engine::DECODE_ROWS, active_drafts, args.attention_plan_rows)?;
        let programs = self.library.programs()?.with_manifest(&args.manifest)?.with_attention_plan(
            attention_plan, |name| glmf_startup_program(name, args.split_device.is_some(),
                args.decode_rows > engine::DECODE_ROWS))?;
        programs.capacities().require_context("glm5_flash", args.max_context)?;
        // The single-copy FP8 consumers of the selected representations, before any weight loads.
        let mut needed = Vec::new();
        if args.kda_fp8 != fp8::KdaFp8::Off && args.kda_state == engine::KdaState::F32 {
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
                else { ["m64", "m4096"].iter().map(|cap| if args.kda_fp8 == fp8::KdaFp8::Off { args.kda_state.program(cap) }
                    else { args.kda_state.w8_program(cap) }).collect() };
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
        probe("lead-programs")?;
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
        // The MLA placement: auto is context on two GPUs where this build carries its programs
        // (with the compact index cache), else heads (whose split keeps the token keys).
        let mut attention = resolve_attention(args, &programs, split_device.is_some())?;
        let context_manifest = if attention.mode == cuteafd_loader::placement::AttentionPlacement::Context {
            let fallback = args.manifest.with_file_name("DSV4_PROGRAMS.json");
            let path = if !args.manifest.exists()
                && args.manifest.file_name() == Some("PROGRAMS.json".as_ref()) && fallback.exists() {
                &fallback
            } else { &args.manifest };
            let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            let mut scratch = 0;
            for device in std::iter::once(args.device).chain(split_device) {
                let bytes = crate::shared::peer_split::on_device(&self.library, device, args.device, || {
                    let sms = self.library.sm_count()?;
                    let mut bytes = 0;
                    for begin in [0, self.cfg.heads / 2] {
                        let name = cuteafd_core::coordinator_programs::attention_partial_program(
                            &manifest, "glmf", begin, engine::DECODE_ROWS, sms, &attention_plan)?;
                        bytes = bytes.max(programs.spec(&name)?.scratch.get("scratch").copied().unwrap_or(0));
                    }
                    Ok(bytes)
                })?;
                scratch = scratch.max(bytes);
            }
            attention.partial_scratch = scratch;
            Some(manifest)
        } else { None };
        let context_mode = (attention.mode == cuteafd_loader::placement::AttentionPlacement::Context).then_some(attention);
        let index_cache = served_index_cache(args.index_cache, split_device.is_some(), attention.mode);
        if let (Some(device), true, None) = (split_device, index_cache != args.index_cache, context_mode) {
            tracing::warn!(device, "--index-cache compact is single-GPU under heads; the head split keeps the token keys");
        }
        cuteafd_bench::context::set_resolved("attention-placement", &attention.mode.to_string());
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
        // The prefix mark arena on the layout this engine serves, its index cache resolved above:
        // both admissions reserve these marks, and `prefix_cache` allocates as many.
        let mark_slots = args.mark_arena.slots_placed(&self.cfg, layers, index_cache, args.kda_state,
            context_mode.is_some())?;
        let loader = weights::GlmfLoader { library: &self.library, checkpoint: &self.checkpoint, stream,
            kda_fp8: args.kda_fp8, kda_output_shard: args.kda_output_shard,
            fp8_head: args.fp8_head, kda_nvfp4: args.kda_nvfp4_gate.as_deref().map(|mode| mode == "search"),
            fp8_scales: args.fp8_scales, device: args.device,
            peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
                .collect(), dense: std::rc::Rc::new(weights::DenseLoad::mount(&self.library)) };
        let mut prepared_draft = match (split_device, args.draft.as_ref()) {
            (Some(_), Some(path)) if cuteafd_loader::placement::families::glm5_flash::remote_dflash_requested()
                && !dspark::is_dspark(path)? => {
                let (device, peer_stream) = peer_stream.context("remote draft needs the peer target stream")?;
                Some(crate::shared::draft::remote::Prepared::new(&self.library,
                    crate::shared::peer_split::RankDevice { device: args.device, stream },
                    crate::shared::peer_split::RankDevice { device, stream: peer_stream },
                    0, remote_dflash::IDENTITY)?)
            },
            _ => None,
        };
        let (placement, admitted_graphs, working) = self.admit(args, &programs, layers, index_cache, mark_slots,
            split_device, automatic_context, prepared_draft.as_ref(), attention)?;
        cuteafd_loader::placement::families::GLM5_FLASH.check(&placement)?;
        let attention = cuteafd_loader::placement::families::glm5_flash::placed_attention(attention, &placement);
        ensure!(placement.attention_placement == attention.mode,
            "GLM Flash admission placed {} attention, the engine serves {}",
            placement.attention_placement, attention.mode);
        let draft_gpu = cuteafd_loader::placement::draft::home(&placement).unwrap_or(0);
        if draft_gpu == 0 { drop(prepared_draft.take()); }
        probe("admission-programs")?;
        let mut selected_args = args.clone();
        selected_args.prefill_waves = usize::try_from(working.prefill_lanes)?;
        let args = &selected_args;

        let spark_layers = admitted_spark_layers(&self.cfg, &placement.layers, args.skip_experts)?;
        if !args.skip_experts {
            ensure!(spark_layers == working.spark_layers,
                "GLM Flash runtime Spark selection disagrees with admitted working set");
            let rtx_layers = placement.layers.iter().enumerate().filter_map(|(layer, assigned)|
                (!self.cfg.dense[layer] && matches!(assigned.experts,
                    cuteafd_loader::placement::ExpertHome::RtxWhole { .. }
                    | cuteafd_loader::placement::ExpertHome::RtxTp2)).then_some(layer)).collect::<Vec<_>>();
            ensure!(rtx_layers == working.rtx_layers,
                "GLM Flash runtime RTX selection disagrees with admitted working set");
        }
        let tp2_layers = admitted_expert_layers(&self.cfg, &placement.layers,
            cuteafd_loader::placement::ExpertHome::RtxTp2, args.skip_experts)?;
        let startup_graphs = admitted_graphs.lifetime == cuteafd_loader::placement::Lifetime::Startup;
        ensure!(tp2_layers.is_empty() || startup_graphs, "GLM Flash TP2 experts require startup graphs; remove \
            --graph-budget-mib and set CUTEAFD_GLMF_STARTUP_GRAPHS=1 (post-ready captures are not admitted)");
        let tp2_devices = || -> Result<[crate::shared::memory::device::Device<'_>; 2]> {
            let peer = peer_stream.context("TP2 placement without peer device")?;
            Ok([crate::shared::memory::device::Device { library: &self.library, id: args.device },
                crate::shared::memory::device::Device { library: &self.library, id: peer.0 }])
        };
        let tp2_budgets = || -> Result<[usize; 2]> {
            let budgets = placement.tp2.context("TP2 arena admission")?.peak_bytes;
            Ok([usize::try_from(budgets[0])?, usize::try_from(budgets[1])?])
        };
        let tp2_package = |manifest: &cuteafd_loader::V41Exl3Manifest| crate::shared::experts::exl3::aot_layout_directory(
            &args.native_lib, manifest.decoder_tiers(), "rtx-tp2");
        let execution = cuteafd_loader::placement::LayerExecutionPlan::new(&placement,
            cuteafd_loader::placement::HopSpec {
                storage: cuteafd_loader::placement::HopStorage::ReceiveSlots,
                row_bytes: 4 * self.cfg.hidden as u64 * 2,
                rows: args.prefill_rows.max(args.decode_rows) as u64,
                lanes: working.prefill_lanes,
                entry_gpu: 0, head_gpu: 0,
                terminal: if peer_stream.is_some() { cuteafd_loader::placement::TerminalHome::Replicated }
                    else { cuteafd_loader::placement::TerminalHome::AttentionOwner },
            })?;
        // Each coordinator GPU is one push rank: its dense operands (class 2
        // landing, class 3 staged and transformed on the GPU) and, with TP2
        // layers, its EXL3 expert halves. Records are planned from headers
        // before membership: a planning error offers no coordinator ranks
        // (the coordinator loads by mount; Sparks still push). Without Spark
        // peers the coordinator GPUs form the push group alone.
        let gpus = 1 + usize::from(peer_stream.is_some());
        let devices: Vec<_> = std::iter::once(args.device).chain(peer_stream.map(|(d, _)| d))
            .map(|id| crate::shared::memory::device::Device { library: &self.library, id }).collect();
        let plan_dense = |sparks: usize| -> Result<_> {
            let load = std::rc::Rc::new(weights::DenseLoad::push(&self.library));
            let planning = weights::GlmfLoader { dense: load.clone(), ..loader.clone() };
            let model = {
                let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                planning.model(&self.cfg, &execution)?
            };
            let (operands, steps) = load.sink.as_ref().context("push sink")?.take();
            let dense = crate::shared::experts::rtx::dense::plan_ranks(&args.snapshot, operands.clone(),
                steps.clone(), &devices, sparks)?;
            Ok((model, operands, steps, dense))
        };
        // Each GPU's TP2 halves' resident budget (admitted minus workspace), when any.
        let tp2_budget_plan = || -> Result<Option<Vec<usize>>> {
            Ok(match self.experts.as_ref().and_then(|c| c.exl3()) {
                Some(manifest) if !tp2_layers.is_empty() => {
                    let workspace = crate::shared::experts::rtx::exl3::Exl3Tp2::workspace_bytes_for(
                        &tp2_package(manifest), self.cfg.hidden, args.expert_rows())?;
                    Some(tp2_budgets()?.map(|b| b.saturating_sub(workspace)).to_vec())
                }
                None if !tp2_layers.is_empty() => anyhow::bail!("coordinator TP2 push needs EXL3 experts"),
                _ => None,
            })
        };
        let mut coordinator_push = None;
        let admission = if args.peers.is_some() {
            let peers = args.peers.as_deref().context("worker placement handshake needs Spark peers")?
                .split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let geometry = if let Some(catalog) = &self.experts { catalog.routed_experts().geometry()? }
                else { cuteafd_core::ExpertGeometry { hidden: u32::try_from(self.cfg.hidden)?,
                    experts: u32::try_from(self.cfg.experts)?, topk: u32::try_from(self.cfg.topk)?,
                    intermediate: u32::try_from(self.cfg.moe_intermediate)?, layers: u32::try_from(self.cfg.layers)? } };
            let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
            let capacity = u32::try_from(args.expert_rows())?;
            let ingress = if self.checkpoint.spark_input_is_bf16() { cuteafd_core::SparkIngress::Bf16 }
                else { cuteafd_core::SparkIngress::Fp8K32 };
            let model = cuteafd_transport::worker_selection::ModelAdmission {
                capacity, compiled_capacity: capacity,
                endpoints: vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity, ingress }],
                rings: crate::shared::placement::spark_rings(64 << 20)?,
                // One execution slot per prefill wave the link keeps in flight.
                execution_slots: cuteafd_loader::plan::layout::spark_execution_slots("glm5_flash",
                    args.prefill_waves as u64, 1)
            };
            let admission = crate::shared::placement::SparkAdmission::plan(source, geometry, self.experts.as_ref(),
                &working, peers.len(), None, model)?;
            let ranks = if admission.coordinator_pushes() && dense_pushable(&args) {
                (|| -> Result<_> {
                    let tp2 = tp2_budget_plan()?;
                    let planned = plan_dense(peers.len())?;
                    let prepared = crate::shared::experts::rtx::push::CoordinatorPush::prepare(&args.snapshot,
                        self.experts.as_ref().filter(|_| tp2.is_some()), gpus,
                        tp2.map(|budgets| crate::shared::experts::rtx::push::CoordinatorExperts::Tp2 { layers: tp2_layers.clone(), budgets }))?;
                    Ok((prepared, planned))
                })().map_err(|error| tracing::warn!(%error, "coordinator push unavailable; mount")).ok()
            } else { None };
            let ranks = ranks.map(|((push, streams, ranks), planned)| {
                coordinator_push = Some((push, peers.len(), devices.clone(), planned));
                (streams, ranks)
            });
            Some(admission.start_with(&peers, ranks)?)
        } else {
            // No Spark peers: the coordinator GPUs push alone (dense operands and
            // their TP2 expert halves).
            if dense_pushable(&args) {
                coordinator_push = (|| -> Result<_> {
                    let tp2 = tp2_budget_plan()?;
                    let planned = plan_dense(0)?;
                    let push = crate::shared::experts::rtx::push::CoordinatorPush::prepare_alone(&args.snapshot,
                        self.experts.as_ref().filter(|_| tp2.is_some()), gpus,
                        tp2.map(|budgets| crate::shared::experts::rtx::push::CoordinatorExperts::Tp2 { layers: tp2_layers.clone(), budgets }))?;
                    Ok((push, 0, devices.clone(), planned))
                })().map_err(|error| tracing::warn!(%error, "coordinator push unavailable; mount")).ok();
            }
            None
        };
        // The admitted TP2 expert halves land inside the Spark admission window;
        // execution (modules, workspaces, routes) binds once the engine exists.
        let tp2_weights = |pushed| -> Result<crate::shared::experts::rtx::Tp2Weights<'_>> {
            let catalog = self.experts.as_ref().context("TP2 expert catalog")?;
            let (devices, budgets) = (tp2_devices()?, tp2_budgets()?);
            use crate::shared::experts::rtx::{exl3::Exl3Tp2, fp8moe::Fp8MoeTp2, RtxSelectionPolicy, Tp2Weights};
            Ok(if let Some(manifest) = catalog.exl3() {
                let package = tp2_package(manifest);
                let geometry = cuteafd_core::expert_geometry();
                Tp2Weights::Exl3(match pushed {
                    Some(pushed) => Exl3Tp2::pushed_pair_weights(devices, catalog, geometry, [&package; 2],
                        tp2_layers.clone(), args.expert_rows(), budgets, RtxSelectionPolicy::CanonicalDecode, pushed)?,
                    None => Exl3Tp2::load_pair_weights(devices, catalog, geometry, [&package; 2],
                        tp2_layers.clone(), args.expert_rows(), budgets, RtxSelectionPolicy::CanonicalDecode)?,
                })
            } else {
                let tensors = catalog.fp8().context("FP8 TP2 tensors")?;
                let package = args.fp8_package.clone().unwrap_or_else(||
                    crate::shared::experts::fp8::package_directory(&args.native_lib, 2, tensors.format()));
                Tp2Weights::Fp8(Fp8MoeTp2::load_pair(devices, tensors, &package, tp2_layers.clone(),
                    args.expert_rows(), budgets)?)
            })
        };
        let embed = self.embed_source()?;
        // Admission is complete; overlap the drafter read with pushed expert/dense operands.
        let draft_read = args.draft.as_deref().map(dspark::DraftRead::start).transpose()?;
        let (embedding, ((model, mut shares), tp2)) = crate::shared::placement::load_while_admitting(admission, || {
            let (pushed, loaded) = match coordinator_push.take() {
                Some((push, sparks, devices, (model, operands, steps, dense))) => {
                    // The planned dense operands (and the TP2 halves) arrive in one
                    // group while the embedding loads beside.
                    let catalog = self.experts.as_ref().filter(|_| !tp2_layers.is_empty());
                    // Receive (or fallback fill) through landing: the dense load's wall time.
                    let split = crate::shared::memory::load_timing::Split::start();
                    let (pushed, embedding) = push.receive_with(devices, catalog, sparks, Some(Ok(dense)), || {
                        crate::shared::token_io::TokenEmbedding::load(&self.library, embed, args.token_io.embed_placement,
                            || Ok(()))
                    });
                    let pushed = pushed?;
                    let (embedding, ()) = embedding?;
                    if pushed.is_none() {
                        let rank_stream = |device: i32| if device == args.device { stream }
                            else { peer_stream.map_or(stream, |(_, s)| s) };
                        crate::shared::experts::rtx::dense::fill_by_mount(&self.library, &operands, &steps, rank_stream)?;
                    }
                    split.log(if pushed.is_some() { "glm5_flash dense push" } else { "glm5_flash dense mount fill" });
                    tracing::info!(pushed = pushed.is_some(), operands = operands.len(), steps = steps.len(),
                        "GLM Flash coordinator dense operands landed");
                    let pushed = match pushed {
                        Some(ranks) if !tp2_layers.is_empty() => Some(<[_; 2]>::try_from(
                            ranks.into_iter().map(|r| r.exl3()).collect::<Result<Vec<_>>>()?)
                            .map_err(|_| anyhow::anyhow!("two coordinator GPUs"))?),
                        _ => None,
                    };
                    (pushed, (embedding, model))
                }
                None => (None, crate::shared::token_io::TokenEmbedding::load(&self.library, embed,
                    args.token_io.embed_placement, || {
                        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                        loader.model(&self.cfg, &execution)
                    })?),
            };
            let tp2 = if tp2_layers.is_empty() { None } else { Some(tp2_weights(pushed)?) };
            Ok((loaded.0, (loaded.1, tp2)))
        })?;
        probe("target-weights")?;
        let resident: usize = model.layers.iter().map(weights::GlmfLayer::bytes).sum();
        let peer_resident: usize = shares.iter().flatten().map(weights::GlmfLayer::bytes).sum();
        let single = model.check_single_residency(args.kda_fp8, args.fp8_head)?;
        // Header-only resident inventory uses the actual loader flags, not launcher defaults.
        if args.kda_nvfp4_gate.is_none() {
            let headers = cuteafd_loader::plan::Checkpoint::coordinator(&args.snapshot, false, false)?;
            let expected = cuteafd_loader::families::glm5_flash::resident::resident_weights_placed(
                &headers, &self.cfg, layers, 1 + usize::from(peer_stream.is_some()),
                cuteafd_loader::families::glm5_flash::resident::GlmfRepresentation {
                    kda_fp8: args.kda_fp8 != fp8::KdaFp8::Off, fp8_head: args.fp8_head,
                    output_shard: args.kda_output_shard,
                }, attention.mode, attention.first_owner as usize).map_err(anyhow::Error::msg)?;
            let actual = resident + model.norm.buffer.bytes + model.head.bytes();
            let peer_expected = expected.get(1).map_or(0, |r| r.weights);
            ensure!(expected[0].weights == actual as u64 && peer_expected == peer_resident as u64,
                "GLM Flash uploaded weights differ from admission: lead {} versus {} bytes; peer {} versus {} bytes",
                actual, expected[0].weights, peer_resident, peer_expected);
            tracing::info!(planned_lead_bytes = expected[0].weights, resident_lead_bytes = actual,
                planned_peer_bytes = peer_expected, resident_peer_bytes = peer_resident,
                "GLM Flash shared resident inventory versus uploaded operands");
        }

        let mib = |bytes: usize| bytes as f64 / (1u64 << 20) as f64;
        tracing::info!(layers, gib = resident as f64 / (1u64 << 30) as f64,
            split_gib = peer_resident as f64 / (1u64 << 30) as f64,
            kda_fp8 = ?args.kda_fp8, fp8_prefill = ?args.fp8_prefill,
            kda_bf16_mib = mib(single.kda_bf16), kda_fp8_mib = mib(single.kda_fp8),
            head = model.head.name(), head_mib = mib(single.head_bf16 + single.head_fp8),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "GLM 5.3 Flash coordinator weights validated after Spark admission (one copy each; elapsed includes wait)");
        let moe = (0..layers).any(|l| !self.cfg.dense[l]);
        let reserved_units = if args.prefix_marks == prefix::PrefixMarks::Pool {
            cuteafd_loader::serving_capacity::GLMF_POOL_MARK_RESERVED_UNITS } else { 0 };
        let pool_tokens = usize::try_from(placement.pool_tokens)?;
        if engine::startup_graphs_enabled() && !startup_graphs {
            tracing::info!(graph_budget_mib = ?args.graph_budget_mib,
                "GLM Flash decode graphs captured lazily within the graph budget, not at startup");
        }
        let max_context = crate::shared::context::pool_context("glm5_flash", args.max_context, automatic_context, pool_tokens, 256)?;
        let step_experts = admitted_step_experts(&self.cfg, &placement, args.skip_experts, self.fp8().is_some())?;
        let early_workspaces = if args.replay_records == engine::ReplayRecords::Shared {
            let mut settings = step_settings(args, index_cache, self.checkpoint.spark_input_is_bf16());
            settings.max_context = max_context;
            let plan = engine::StepPlan::new(&self.library, &programs, &self.cfg, &model.layers, step_experts, settings);
            let lanes = engine::configured_prefill_waves(layers == self.cfg.layers, args.prefill_waves);
            Some(engine::StepWorkspaces::allocate(&plan, args.prefill_rows, lanes)?)
        } else { None };
        let records = early_workspaces.as_ref().map(|w| w.prefill_scratch()
            .context("shared replay records need prefill lanes")).transpose()?;
        let pages = (pool_tokens + reserved_units as usize * engine::UNIT_ROWS).div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::GlmfEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            max_context, args.prefill_rows, args.prefill_waves, args.prefill_span_rows, pages, args.slots, embedding, index_cache,
            args.kda_state, args.decode_rows, records, context_mode, args.prefix_marks == prefix::PrefixMarks::Pool,
            &execution, step_experts)?;
        engine.attention_plan = attention_plan;
        engine.spark_bf16 = self.checkpoint.spark_input_is_bf16();

        if !startup_graphs {
            engine.capture_graphs_lazily();
        }
        tracing::info!(index_cache = ?engine.index_cache, pool_tokens, reserved_units, "GLM 5.3 Flash DSA index cache");
        engine.kda_fp32_partials = args.kda_fp32_partials;
        engine.kda_output_shard = args.kda_output_shard;
        engine.kda_prefill_expanded = args.kda_prefill_expanded;
        engine.mark_slots = mark_slots;
        engine.target_head = args.resolved_target_head();
        tracing::info!(target_head = ?engine.target_head, requested = ?args.target_head, fp8_head = args.fp8_head,
            "GLM 5.3 Flash target head");
        if let Some((device, peer_stream)) = peer_stream {
            engine.attach_peer(device, peer_stream, shares.pop().context("head-split shares")?, &execution)?;
            tracing::info!(device = args.device, split_device = device, attention = %attention.mode,
                "GLM 5.3 Flash head split over two GPUs");
        }
        engine.validate_attention_homes(&execution)?;
        if attention.mode == cuteafd_loader::placement::AttentionPlacement::Layers {
            engine.bind_layer_execution(execution)?;
        }
        if let Some(manifest) = &context_manifest {
            // The solver charged selected partial scratch and full-extent staging.
            engine.attach_context(args.max_context, manifest)?;
        }
        probe("peer-exchanges")?;
        engine.set_graph_budget(args.graph_budget_mib.map(|mib| mib << 20));
        engine.full_prefill_logits = args.full_prefill_logits;
        let group = |g: Fp8PrefillGroup| args.fp8_prefill.iter().any(|&x| x == g || x == Fp8PrefillGroup::All);
        // `all`: every group with FP8 weights (BF16 KDA has none to run W8A8 over).
        let kda = args.kda_fp8 != fp8::KdaFp8::Off;
        engine.fp8_prefill = engine::Fp8Prefill { mla: group(Fp8PrefillGroup::Mla), ffn: group(Fp8PrefillGroup::Ffn),
            kda_bits: i32::from(kda && group(Fp8PrefillGroup::KdaIn))
                | (i32::from(kda && group(Fp8PrefillGroup::KdaO)) << 1) };
        if let Some(tp2) = tp2 {
            let ranks = tp2.execution()?;
            probe("expert-packages")?;
            engine.install_tp2(ranks)?;
            probe("route-exchange")?;
        }
        // Spark links remain available for every layer whose admitted home is Spark.
        let spark = !spark_layers.is_empty();
        if moe {
            let local_layers = admitted_expert_layers(&self.cfg, &placement.layers,
                cuteafd_loader::placement::ExpertHome::RtxWhole { gpu: 0 }, args.skip_experts)?;
            if !local_layers.is_empty() {
                let budget = usize::try_from(placement.expert_ranges.first().context("local expert arena admission")?.peak_bytes)?;
                let mut expert_args = args.clone();
                expert_args.local_experts = true;
                expert_args.peers = None;
                let experts = self.experts_for_layers(&expert_args, &local_layers, Some(budget))?
                    .context("admitted local expert backend is missing")?;
                if let engine::Experts::LocalExl3(local) = &experts { local.ensure(local_layers[0], stream)?; }
                engine.install_local(local_layers, experts)?;
            }
            if spark || args.skip_experts {
                let mut expert_args = args.clone();
                expert_args.local_experts = false;
                if let Some(experts) = self.experts_for_layers(&expert_args, &spark_layers, None)? {
                    engine.set_experts(experts)?;
                }
            }
        }
        engine.bind_expert_homes(&placement, &spark_layers, args.skip_experts)?;
        engine.drafter = self.load_drafter(args, stream, &engine.embedding, prepared_draft.take(), draft_read)?;
        engine.attention_plan = attention_plan.validate_loaded(engine.drafter.as_ref().map(dspark::Drafter::drafts))?;
        tracing::info!(plan_rows = engine.attention_plan.rows(), active_drafts = ?engine.attention_plan.active_drafts(),
            "GLM Flash immutable attention reduction plan");
        if let Some(dspark::Drafter::RemoteDflash(draft)) = &engine.drafter {
            let (device, stream) = peer_stream.context("remote draft lost its target producer")?;
            draft.validate_target_peer(crate::shared::peer_split::RankDevice { device, stream })?;
        }
        if let Some(dense) = self.load_dense(args, &engine.weights.layers)? { engine.set_dense_nvfp4(dense); }
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::GLM_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
            if engine.ranks() > 1 {
                engine.attach_peer_l2(budget)?;
            }
        }
        if let Some(workspaces) = early_workspaces { engine.install_workspaces(workspaces)?; }
        if args.serving_graph_policy.is_some() { engine.prepare_serving_workspaces()?; }
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        probe("workspaces")?;
        engine.prime_tp2()?;
        if let Some(dspark::Drafter::RemoteDflash(draft)) = &engine.drafter {
            draft.prime(&engine.embedding, engine.draft_head())?;
        }
        probe("expert-prime")?;
        let result = with_startup_graphs(engine.has_tp2(), args.serving_graph_policy,
            (16, engine.drafter.is_some()), |sequences, speculation| {
                engine.warm_decode_graphs(sequences.min(args.decode_rows), speculation)?;
                probe("graph-capture")
            }, || {
                engine.check_admitted_graphs(&admitted_graphs)?;
                engine.check_tp2_routes()
            }, || body(&engine));
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
        embedding: &crate::shared::token_io::TokenEmbedding<'_>, prepared: Option<crate::shared::draft::remote::Prepared<'s>>,
        read: Option<dspark::DraftRead>)
        -> Result<Option<dspark::Drafter<'s>>> {
        let Some(snapshot) = &args.draft else { return Ok(None) };
        let read = read.context("the drafter read was not started with --draft")?;
        let started = Instant::now();
        let representation = cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation
            ::from_fp8_option(args.draft_fp8);
        let slots = args.draft_context_slots.unwrap_or(20.max(args.draft_sequences));
        let drafter = if let Some(prepared) = prepared {
            use crate::shared::speculation::dflash::{DflashConfig, DflashDrafter};
            let cfg = DflashConfig::read(snapshot)?;
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "DFlash2 does not fit this target");
            let mask = embedding.host_rows(&[cfg.mask_token])?;
            let file = read.join_dflash()?;
            let device = prepared.stream.device;
            let draft = device.own(|| DflashDrafter::load(&self.library, snapshot, file, prepared.stream.raw,
                slots, args.draft_sequences.min(slots), mask, true, representation, args.fp8_scales, args.draft_linear, args.draft_kv_cache))?;
            dspark::Drafter::RemoteDflash(remote_dflash::RemoteDflash::new(draft, prepared)?)
        } else {
            dspark::Drafter::load(&self.library, snapshot, stream, slots, args.draft_sequences.min(slots),
                embedding, self.cfg.hidden, self.cfg.vocab_size, self.cfg.layers, representation, args.fp8_scales,
                args.draft_linear, args.draft_kv_cache, read)?
        };
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

    fn experts_for_layers<'s>(&'s self, args: &EngineArgs, selected: &[usize], budget: Option<usize>) -> Result<Option<engine::Experts<'s>>> {
        if args.skip_experts {
            return Ok(Some(engine::Experts::Skip));
        }
        let served = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        validate_expert_layers(&self.cfg, served, selected)?;
        if selected.is_empty() { return Ok(None); }
        if let Some(tensors) = self.fp8().filter(|_| args.local_experts) {
            let directory = args.fp8_package.clone()
                .unwrap_or_else(|| crate::shared::experts::fp8::package_directory(&args.native_lib, 1, tensors.format()));
            let budget = match budget { Some(bytes) => bytes, None => self.library.cuda_memory_info()?.0
                .saturating_sub(args.expert_reserve_gib.saturating_mul(1 << 30)) };
            ensure!(args.expert_window != Some(0), "--expert-window must be at least 1");
            let started = Instant::now();
            let experts = crate::shared::experts::fp8::Fp8Experts::load(&self.library, tensors, &directory, selected, 1, 0,
                args.expert_rows(), budget)
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
        if let Some(catalog) = self.experts.as_ref().filter(|c| args.local_experts && c.exl3().is_some()) {
            let budget = match budget { Some(bytes) => bytes,
                None => self.library.cuda_memory_info()?.0.saturating_sub(12 << 30) };
            return Ok(Some(engine::Experts::LocalExl3(engine::LocalExl3 {
                library: &self.library, native_lib: args.native_lib.clone(), catalog,
                resident: std::cell::RefCell::new(None), window: selected.len(),
                layers: selected.to_vec(), max_rows: args.expert_rows(),
                // Room for the step workspace (logits alone are 2.4 GiB at 4096 rows).
                budget, loads: std::cell::RefCell::new(0),
            })));
        }
        let Some(peers) = args.peers.as_deref() else { return Ok(None) };
        let remote = selected;
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let spark_world = cuteafd_transport::expert::SparkWorld::canonical(peers.len())?;
        // One link retains independent intake planes for every outstanding unit wave.
        let mut transport = crate::shared::spark_intake::SparkLink::new_waves(&self.library,
            spark_world, &peers, u32::try_from(args.expert_rows())?, cuteafd_transport::TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(),
                timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 }, self.cfg.hidden * 2,
            args.prefill_waves, u32::try_from(args.prefill_rows)?)?;
        transport.enable_terminal_ownership(self.library.clone())?;
        crate::shared::memory_report::release_load_staging(&self.library);
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        let warm_rows = args.expert_rows();
        let warm = engine::spark_warmup_request(&self.cfg, warm_rows, remote[0], self.checkpoint.spark_input_is_bf16())?;
        // Publish grouping before the draft planner prices the installed workers.
        transport.bootstrap(&warm)?;
        let rings = || cuteafd_ffi::memory_ledger::snapshot().by_scope(cuteafd_ffi::memory_ledger::Space::Pinned, -1)
            .get("transport/rdma-rings").copied().unwrap_or(0);
        let (started, before) = (Instant::now(), rings());
        let warm_stream = crate::shared::spark_intake::WarmStream::new(self.library.as_ref())?;
        runtime.block_on(async {
            let wave = transport.dispatch(&warm)?;
            transport.receive(wave, warm_rows, warm_stream.raw()).await
        }).with_context(|| format!("warming Spark expert transport with {warm_rows} rows"))?;
        warm_stream.finish()?;
        tracing::info!(waves = args.prefill_waves, ranks = peers.len(), rows = warm_rows,
            ring_bytes = rings().saturating_sub(before), elapsed_ms = started.elapsed().as_millis() as u64,
            "Spark expert transport warm");
        Ok(Some(engine::Experts::Spark { transport: std::cell::RefCell::new(transport), runtime }))
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
    if args.replay_production {
        // Capture speculative buckets even when this fixed-token diagnostic has no drafter.
        args.engine.serving_graph_policy = Some((16, true));
    }
    args.engine.full_prefill_logits |= args.nll || args.resume_at.is_some() || args.unit_check;
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

/// Prints the per-program GPU times collected since the last read (`CUTEAFD_GLMF_PROFILE_OPS=1`; nothing
/// without it), per run of `runs`.
fn print_op_profile(engine: &engine::GlmfEngine<'_>, title: &str, runs: usize) -> Result<()> {
    let ops = engine.op_profile()?;
    if ops.is_empty() {
        return Ok(());
    }
    let runs = runs as f64;
    let total: f64 = ops.iter().filter(|(k, _)| !k.starts_with("host")).map(|(_, v)| v.0).sum();
    println!("{title} (GPU ms between events, {total:.1} ms total per {runs} runs):");
    let mut rows: Vec<_> = ops.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    for (label, (ms, count)) in rows {
        println!("  {label:44} {:9.2} ms  {:6.1}%  x{}", ms / runs, 100.0 * ms / total, count as f64 / runs);
    }
    Ok(())
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
    if args.upload_check {
        let tokens = unit_check::input_tokens(&args.golden, engine.cfg.vocab_size)?;
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return prefix::upload_check(engine, &tokens[..n], args.prefill_chunk.unwrap_or(256));
    }
    if let Some(at) = args.resume_at {
        let tokens = unit_check::input_tokens(&args.golden, engine.cfg.vocab_size)?;
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return prefix::resume_check(engine, &tokens, at, n,
            args.prefill_chunk.unwrap_or(engine.prefill_rows), args.resume_decode, args.resume_cold,
            args.resume_repeat, args.engine.prefix_marks, args.resume_poison_unit0);
    }
    if let Some(steps) = args.token_check {
        return token_check(args, opened, engine, steps);
    }
    if args.unit_check {
        let tokens = unit_check::input_tokens(&args.golden, engine.cfg.vocab_size)?;
        let n = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
        return unit_check::unit_check(engine, &tokens[..n]);
    }
    if let Some(sequences) = args.packed_check {
        let tokens = unit_check::input_tokens(&args.golden, engine.cfg.vocab_size)?;
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
    let mut placement = engine::Allocator::for_engine(engine).admit(tokens.len() + args.bench_decode)?;
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
    // through continuous units, chunks up to the engine's prefill capacity).
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
        let mut allocator = engine::Allocator::for_engine(engine);
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
        print_op_profile(engine, "prefill ops per prefill", times.len())?;
    }
    if args.bench_decode > 0 {
        *engine.profile.borrow_mut() = [0.0; 3];
        engine.op_profile()?;
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
        print_op_profile(engine, "decode ops per step", times.len())?;
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
    let mut allocator = engine::Allocator::for_engine(engine);
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
