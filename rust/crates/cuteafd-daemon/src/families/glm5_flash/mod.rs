//! GLM 5.3 Flash (glm5_next) on the generic engine: weights, the coordinator
//! programs' layer chain, and the golden comparison command.
pub(crate) mod dspark;
#[cfg(test)]
mod draft_push_tests;
pub(crate) use crate::shared::draft::resource as draft_binding;
use crate::shared::draft::probe as draft_probe;
pub(crate) mod engine;
pub(crate) mod fp8;
pub(crate) mod prefix;
pub(crate) mod serve;
pub(crate) mod media;
mod speculate;
mod expert_rows;
mod unit_check;
mod packed_check;
pub(crate) mod packing;
mod header;
pub(crate) mod head;
mod precision;
mod native;
pub(crate) mod weights;
use cuteafd_loader::families::glm5_flash::weight_layout;

/// Whether this launch's coordinator operands can join a push group: every
/// representation the push path records (BF16 operands, GPU FP8 copies,
/// native E4M3 blocks, token-row output shards). The NVFP4 numerics gate
/// (a measurement mode) loads through the mount.
fn dense_pushable(args: &EngineArgs) -> bool {
    args.kda_nvfp4_gate.is_none()
}

fn resolve_prefill(args: &mut EngineArgs, complete: bool) -> Result<()> {
    use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides, MAX_WAVES};
    let ranks = args.peers.as_deref().map_or(0, |peers| peers.split(',').count());
    cuteafd_loader::placement::attention::parse(&args.attention_placement)
        .map_err(anyhow::Error::msg)?;
    let shape = prefill_shape(if complete { ranks } else { 0 }, PrefillLimits {
        max_waves: if !complete || ranks == 0 { 1 } else { MAX_WAVES },
        ..cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::glm::GLM5_FLASH)
    }, PrefillOverrides {
        unit_rows: (args.prefill_rows > 0).then_some(args.prefill_rows as u64),
        waves: (args.prefill_waves > 0).then_some(args.prefill_waves as u64),
        span_rows: (args.prefill_span_rows > 0).then_some(args.prefill_span_rows as u64),
        reserve_rows: None,
    })?;
    args.prefill_rows = usize::try_from(shape.unit_rows)?;
    args.prefill_waves = usize::try_from(shape.waves)?;
    args.prefill_span_rows = usize::try_from(shape.span_rows)?;
    cuteafd_bench::context::ResolvedSettings::prefill(shape).publish();
    Ok(())
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
    #[command(flatten)]
    pub attention: crate::cli::AttentionArgs,
    /// Packed prefill arithmetic policy. Unset: unrestricted serving, exact --packed-check.
    #[arg(long, value_enum)]
    pub prefill_packing: Option<crate::shared::prefill_packing::PackingMode>,
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
    /// A manifest published inside the native EXLR8 checkpoint (k2, k3 or k4).
    #[arg(long)]
    pub exlr8_manifest: Option<String>,
    /// Native Flash preserves the checkpoint's A16 activation numerics.
    #[arg(long, default_value = "a16", value_parser = native::prefill_activations)]
    pub exlr8_prefill_activations: cuteafd_loader::plan::exlr8::PrefillActivations,
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
    /// Served context: auto fits the admitted pool; a positive count pins it.
    #[arg(long = "max-context-tokens", default_value = "auto", value_parser = crate::shared::context::parse_tokens::<usize>)]
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
    /// Rows per unit; 0 selects the shared topology policy (at most 4096).
    #[arg(long, default_value_t = 0)]
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
    /// Maximum live prefill units on one Spark transport (1 to 4).
    /// Units retain their own operands and share attention temporaries; long
    /// spans reuse slots continuously. --prefill-rows bounds each unit's rows.
    #[arg(long = "prefill-waves", default_value_t = 0,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(0..=engine::MAX_PREFILL_WAVES as u64))]
    pub prefill_waves: usize,
    /// Shared prefill span; 0 resolves to waves times per-unit rows.
    #[arg(long, default_value_t = 0)]
    pub prefill_span_rows: usize,
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
    /// Routed layer onboarding: auto (pool first), all (every routed-expert
    /// layer of the target model), or an exact N.
    #[arg(long, env = "RTX_EXPERT_LAYERS", value_parser = parse_onboard)]
    pub rtx_expert_layers: Option<cuteafd_loader::placement::OnboardSetting>,
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
    /// The one resident LM head target logits and the drafter share. `false` (also `bf16`,
    /// `off`): the checkpoint's BF16 head. `true` (a bare `--fp8-head`; also `fp8`,
    /// `on`): only an E4M3 copy (per row x 128-K scales, quantized at load); every logits call
    /// (target, verify, prefill, DFlash drafts) runs the FP8 head program in 16-row spans.
    /// `packed`: only the exact 12-bit copy of the BF16 values (`shared::packed_head`), packed
    /// and verified at load, then the BF16 released: target logits read it exactly on its
    /// tensor-core GEMV, the drafter through its coarse plane; one GPU (no --split-device).
    /// `auto` (default): packed on one GPU, FP8 on a head split; native heads stay native.
    #[arg(long = "fp8-head", default_value = "auto", num_args = 0..=1, default_missing_value = "true",
        action = clap::ArgAction::Set)]
    pub head_format: head::GlmfHeadFormat,
    /// The target's logits through the BF16 head (the shared route, `crate::shared::target_head`):
    /// `tensor`, a BF16 tensor-core GEMM with FP32 accumulation at every row count, or `exact`
    /// (diagnostic), the pedantic FP32 cuBLAS GEMM on CUDA cores. Unset: CUTEAFD_TARGET_HEAD
    /// (tensor by default). Unset with the FP8 or packed head, that head's own GEMV runs (an
    /// explicit `exact` is inert); an explicit `tensor` with either is refused.
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
    #[arg(long, default_value_t = cuteafd_loader::families::glm5::draft_representation::DEFAULT_DRAFT_SEQUENCES)]
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
    /// Split GQA drafter heads, KV and MLP over the target's peer GPUs.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub tp2_dflash: bool,
    #[arg(skip = true)]
    pub tp2_dflash_auto: bool,

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
    #[arg(skip)]
    pub serving_prefix: Option<crate::shared::prefix::PrefixArgs>,
    #[arg(skip)]
    pub serving_media_bytes: u64,
    #[arg(skip)]
    pub serving_encoder: Option<cuteafd_loader::plan::encoder::EncoderPlacement>,
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

/// Finish every loader before releasing its reusable staging; priming must not retain it.
fn finish_loading<S, T>(library: &NativeLibrary, state: &mut S,
    load: impl FnOnce(&mut S) -> Result<()>, prime: impl FnOnce(&mut S) -> Result<T>) -> Result<T> {
    load(state)?;
    crate::shared::memory_report::release_load_staging(library)?;
    prime(state)
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
mod admission_tests {
    /// Native module code is lazy (`CUDA_MODULE_LOADING` default): it lands on
    /// first launch or `cudaFuncGetAttributes`, which is after the pre-weight
    /// baseline unless it is forced inside the measured closure (#1554, #1577).
    #[test]
    fn startup_native_modules_load_inside_the_measured_baseline() {
        let source = include_str!("mod.rs");
        // The last occurrence is the real closure; the test modules above it
        // (which quote this text) are not part of the measured region.
        let measured = source.rsplit("RuntimeInventory::measure(&self.library, &devices")
            .next().unwrap();
        let measured = measured.split("Ok(())\n            })").next().unwrap();
        let native = measured.find("self.library.tp2_native_initialize()?").unwrap();
        let peer = measured.find("if let Some(other)").unwrap();
        assert!(native < peer, "native modules must load before peer initialization");
        assert_eq!(measured.matches("self.library.tp2_native_initialize()?").count(), 1);
    }
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

    fn staging_boundary(spark: bool, draft: bool) -> Result<()> {
        use crate::shared::memory::DeviceAllocation;
        use cuteafd_ffi::{memory_ledger::{self, Space}, native_library_lifetime_fixture::Fixture};

        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let tag = format!("flash-load-staging/{}", fixture.directory().display());
        let _owner = memory_ledger::owner(&tag);
        let staging_bytes = || memory_ledger::snapshot().by_owner(Space::Pinned, -1)
            .get(tag.as_str()).copied().unwrap_or(0);
        let mut target = DeviceAllocation::new(&library, 512, crate::shared::memory::TEST)?;
        library.copy_h2d(target.buffer, &[1; 64])?;
        assert_eq!(staging_bytes(), 64);
        assert_eq!(fixture.events()?, "DAH");

        // The real production boundary runs these allocation-only load/prime stand-ins.
        // This checks lifetime ordering, not Flash's numerical kernels or Spark transport.
        finish_loading(&library, &mut target, |target| {
            if spark {
                library.copy_h2d(target.buffer, &[2; 128])?;
                assert_eq!(staging_bytes(), 128);
            }
            if draft {
                library.copy_h2d(target.buffer, &[3; 256])?;
                assert_eq!(staging_bytes(), 256);
            }
            library.copy_h2d(target.buffer, &[4; 512])?;
            assert_eq!(staging_bytes(), 512);
            assert!(fixture.events()?.ends_with('H'), "last load must still own staging");
            Ok(())
        }, |_| {
            assert_eq!(staging_bytes(), 0, "first prime must not retain loader staging");
            assert!(fixture.events()?.ends_with("HF"), "release must follow the last upload");
            let prime = DeviceAllocation::new(&library, 16, crate::shared::memory::TEST)?;
            drop(prime);
            with_startup_graphs(false, Some((1, draft)), (1, false), |_, _| {
                assert_eq!(staging_bytes(), 0);
                let warm = DeviceAllocation::new(&library, 16, crate::shared::memory::TEST)?;
                drop(warm);
                Ok(())
            }, || unreachable!("no TP2 diagnostics in this fixture"), || {
                assert_eq!(staging_bytes(), 0, "Ready must not retain loader staging");
                Ok(())
            })
        })?;
        let events = fixture.events()?;
        let expected = format!("DAH{}{}FAHFDdDd", if spark { "FAH" } else { "" },
            if draft { "FAH" } else { "" });
        assert_eq!(events, expected, "actual DSO alloc/upload/free chronology");
        assert_eq!(events.matches('A').count(), events.matches('F').count());
        assert_eq!(library.release_sync_h2d_staging()?, 0);
        drop(target);
        assert!(memory_ledger::snapshot().rows.iter().all(|row|
            row.key.owner != Some(tag.as_str()) || row.allocations == 0));
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn local_load_staging_is_released_before_prime_and_ready() -> Result<()> {
        for draft in [false, true] { staging_boundary(false, draft)?; }
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn spark_then_drafter_staging_is_released_before_prime_and_ready() -> Result<()> {
        staging_boundary(true, true)
    }

    #[test]
    fn serving_startup_seals_once_for_tp2_and_spark_layouts() {
        use std::cell::{Cell, RefCell};
        use crate::shared::decode_graph::{GraphBank, GraphPolicy};
        use cuteafd_loader::serving_capacity::glmf_graphs::{serving_graph_shapes, DecodeBuckets, PAGE_ROWS};

        for (name, tp2, spark_ranks) in [("tp2-zero-spark", true, 0),
            ("tp2-spark-suffix", true, 4), ("spark-only", false, 4)] {
            // Only the shape list matters here; it needs no measured census row.
            let shapes = serving_graph_shapes(262144, 262144 / PAGE_ROWS, 2048, 8, true, &DecodeBuckets::new(64));
            let banks = RefCell::new([GraphBank::with_policy(GraphPolicy::FixedStartup),
                GraphBank::with_policy(GraphPolicy::FixedStartup)]);
            let seals = Cell::new([0; 2]);
            let events = RefCell::new(Vec::new());
            // The production with_engine sequence, with CPU executable stand-ins.
            with_startup_graphs(tp2, Some((8, true)), (16, false), |sequences, speculation| {
                assert_eq!((sequences, speculation), (8, true));
                events.borrow_mut().push("warm");
                for (rank, bank) in banks.borrow_mut().iter_mut().enumerate() {
                    let keys: Vec<_> = shapes.iter().enumerate().flat_map(|(shape, _)|
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
    fn native_push_records_shared_and_vocabulary_before_group_membership() {
        let source = include_str!("mod.rs");
        let plan = source.split("let plan_native =").last().unwrap()
            .split("let mut native_push =").next().unwrap();
        let body = plan.find("native.load(").unwrap();
        let shared = plan.find("native.load_shared(").unwrap();
        let destinations = plan.find("dense::plan_records(").unwrap();
        assert!(body < shared && shared < destinations);
        assert!(plan.contains("dense::recording("));
        let startup = source.split("let mut native_push =").last().unwrap()
            .split("// The admitted TP2 expert halves").next().unwrap();
        for arm in startup.split("let planned = plan_native(").skip(1) {
            assert!(arm.find("CoordinatorPush::prepare").is_some());
        }
        assert_eq!(startup.matches("let planned = plan_native(").count(), 2);
        let receive = source.split("let mut native_local =").last().unwrap()
            .split("embedding.validate_residency(").next().unwrap();
        assert!(receive.contains("Some((push, sparks, (owners, dense)))"));
        assert!(receive.contains("Some(Ok(dense))"));
        assert!(receive.contains("push_policy.committed(pushed)?"));
        assert!(!receive.contains("NoDense"));
        assert!(source.contains("native_shared.take().context(\"native shared owners missing after load\")?"));
    }

    #[test]
    fn resolved_flash_capacity_and_native_flags_are_runtime_facts() {
        let mut args = Parse::try_parse_from(["test", "--snapshot", "/s", "--native-lib", "/l"]).unwrap().engine;
        crate::commands::bench::assert_resolved_family_seam("glm5_flash",
            &["max-context-tokens", "pool-tokens", "rtx-expert-layers", "prefix-marks", "split-device", "attention-plan-rows"], || {
                crate::commands::bench::fixture_prefix().resolve_entries(16);
                serve::resolve_serving_capacity(&mut args, 16);
                native::normalize(&mut args);
                native::publish_resolved(&args);
                resolve_prefill(&mut args, true)?;
                served_index_cache(args.index_cache, false,
                    cuteafd_loader::placement::AttentionPlacement::Heads);
                Ok(())
            }).unwrap();
        assert_eq!(args.slots, 16);
        assert_eq!(served_index_cache(engine::IndexCache::Keys, true,
            cuteafd_loader::placement::AttentionPlacement::Context), engine::IndexCache::Compact);
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
        let mut args = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native"].into_iter()
            .chain(extra.iter().copied())).unwrap().engine;
        if args.attention_placement != "unknown" {
            resolve_prefill(&mut args, true).unwrap();
        }
        args
    }

    #[test]
    fn plan_and_serve_auto_share_checkpoint_topology_resolution() {
        use clap::Parser;
        use head::GlmfHeadFormat::{Bf16, Fp8, Packed};
        for (native, ranks, expected) in [(false, 1, Packed), (false, 2, Fp8), (true, 1, Bf16), (true, 2, Bf16)] {
            let cli = crate::cli::Cli::try_parse_from(["cuteafd", "plan", "/checkpoint", "--layout", "--rtx", &ranks.to_string()]).unwrap();
            let crate::cli::Commands::Plan(plan) = cli.command else { panic!("plan") };
            let mut serve = if ranks == 2 { parse(&["--split-device", "1"]) } else { parse(&[]) };
            native::normalize_head(&mut serve, native);
            assert_eq!(plan.head_format.resolve(native, ranks), expected);
            assert_eq!(serve.head_format, expected);
            let mut context = cuteafd_bench::context::ServerContext::default();
            context.require_resolved("fp8-head");
            for (name, value) in native::resolved_settings(&serve) { context.resolve(name, &value); }
            let settings = context.report_settings();
            let reported = settings.iter().find(|setting| setting.name == "fp8-head").unwrap();
            assert_eq!(reported.source, "resolved");
            assert_eq!(reported.value.as_deref(), Some(expected.setting()));
            assert_ne!(reported.value.as_deref(), Some("auto"));
        }
    }

    #[test]
    fn native_workspace_is_invariant_to_legacy_conversion_flags() -> Result<()> {
        use cuteafd_loader::serving_capacity::{glmf_core_step_scratch, GlmfScratchOptions,
            GlmfCoreScratch, GlmfStepShape};
        let cfg = GlmNextConfig::from_hf(&cuteafd_loader::plan::testing::glm5_flash_config(45))?;
        let mut ordinary = parse(&[]);
        let mut converted = ordinary.clone();
        converted.kda_fp8 = fp8::KdaFp8::Row128;
        converted.head_format = head::GlmfHeadFormat::Packed;
        converted.kda_fp32_partials = true;
        converted.kda_output_shard = true;
        converted.kda_prefill_expanded = true;
        converted.kda_nvfp4_gate = Some("search".into());
        converted.fp8_prefill = vec![Fp8PrefillGroup::All];
        native::normalize(&mut ordinary);
        native::normalize(&mut converted);
        assert_eq!(native::resolved_settings(&ordinary), native::resolved_settings(&converted));
        let settings: std::collections::BTreeMap<_, _> = native::resolved_settings(&converted).into_iter().collect();
        for key in ["kda-output-shard", "kda-prefill-expanded", "kda-fp32-partials", "fp8-head"] {
            assert_eq!(settings[key], "false");
        }
        assert_eq!(settings["kda-fp8"], "off");
        assert_eq!(settings["fp8-prefill"], "none");
        let demands = |args: &EngineArgs| -> Result<Vec<_>> {
            check_options(args)?;
            let options = GlmfScratchOptions { split: false, kda_w8: args.kda_fp8 != fp8::KdaFp8::Off,
                kda_fp32_partials: args.kda_fp32_partials, kda_output_shard: args.kda_output_shard,
                kda_prefill_expanded: args.kda_prefill_expanded, index_compact: false, kda_state: args.kda_state.into() };
            let mask = GlmfCoreScratch { kda: true, mla: true, ffn: true };
            let decode = glmf_core_step_scratch(|_| Some(4096), &cfg, options, 64, true, mask)?;
            let prefill = glmf_core_step_scratch(|_| Some(4096), &cfg, options, 4096, false, mask)?;
            Ok(cuteafd_loader::placement::families::glm5_flash::step_workspace_demands(&cfg, 3, 4096, 64,
                GlmfStepShape { lead: true, split: false, local_experts: false, tp2_experts: false,
                    spark: true, partial_bytes: if args.kda_fp32_partials { 4 } else { 2 },
                    output_shard: args.kda_output_shard, full_prefill_logits: false, table_pages: 512, table_pool_pages: 512 },
                decode, prefill).into_iter().map(|d| d.demand.bytes).collect())
        };
        assert_eq!(demands(&ordinary)?, demands(&converted)?);
        assert_eq!(step_settings(&ordinary, ordinary.index_cache).kda_fp32_partials, false);
        assert_eq!(converted.fp8_prefill, vec![Fp8PrefillGroup::None]);
        Ok(())
    }

    #[test]
    fn packed_mode_cli_preserves_default_and_accepts_both_values() {
        use crate::shared::prefill_packing::PackingMode;
        assert_eq!(parse(&[]).prefill_packing.unwrap_or_default(), PackingMode::Unrestricted);
        assert_eq!(parse(&["--prefill-packing", "exact"]).prefill_packing, Some(PackingMode::Exact));
        assert_eq!(parse(&["--prefill-packing", "unrestricted"]).prefill_packing, Some(PackingMode::Unrestricted));
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
        let mut checkpoint = Checkpoint { view: None, snapshot: "fixture".into(), config, quantize_config: None,
            weight_map: Default::default(), tensors, missing_shards: vec![], shard_bytes: 0 };
        for ranks in [1, 2] {
            for head in [head::GlmfHeadFormat::Bf16, head::GlmfHeadFormat::Fp8, head::GlmfHeadFormat::Packed]
                .into_iter().filter(|&head| ranks == 1 || head != head::GlmfHeadFormat::Packed) {
                let representation = GlmfRepresentation { head, ..Default::default() };
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
        let tensor = parse(&["--target-head", "tensor", "--fp8-head", "false"]);
        assert_eq!(tensor.resolved_target_head(), Tensor);
        check_options(&tensor).unwrap();
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--target-head", "fp8"]).is_err());
        let error = check_options(&parse(&["--target-head", "tensor", "--fp8-head"])).unwrap_err().to_string();
        assert!(error.contains("--target-head tensor") && error.contains("--fp8-head"), "{error}");
        check_options(&parse(&["--target-head", "exact", "--fp8-head"])).unwrap();
        // The packed head runs its own exact GEMV too: tensor is refused, exact inert.
        let error = check_options(&parse(&["--fp8-head", "packed", "--target-head", "tensor"])).unwrap_err().to_string();
        assert!(error.contains("--target-head tensor") && error.contains("--fp8-head packed"), "{error}");
        check_options(&parse(&["--fp8-head", "packed", "--target-head", "exact"])).unwrap();
    }

    #[test]
    fn native_k6_head_normalizes_before_conversion_only_refusals() {
        for extra in [
            &["--fp8-head", "packed", "--split-device", "1"][..],
            &["--fp8-head", "packed", "--target-head", "tensor"][..],
            &["--fp8-head", "true", "--target-head", "tensor"][..],
        ] {
            let mut args = parse(extra);
            native::normalize_head(&mut args, false);
            assert!(check_options(&args).is_err());
            native::normalize_head(&mut args, true);
            assert_eq!(args.head_format, head::GlmfHeadFormat::Bf16);
            check_options(&args).unwrap();
        }
    }

    /// `--fp8-head` keeps its boolean spellings (a bare flag is FP8) and adds `packed`, the exact
    /// 12-bit head, which serves one GPU: the head split is refused before any load.
    #[test]
    fn fp8_head_selects_bf16_fp8_or_the_packed_head_on_one_gpu() {
        use head::GlmfHeadFormat::{Bf16, Fp8, Packed};
        for (extra, format) in [(&[][..], head::GlmfHeadFormat::Auto), (&["--fp8-head"][..], Fp8), (&["--fp8-head", "true"][..], Fp8),
            (&["--fp8-head", "on"][..], Fp8), (&["--fp8-head", "false"][..], Bf16), (&["--fp8-head", "bf16"][..], Bf16),
            (&["--fp8-head", "packed"][..], Packed), (&["--fp8-head", "--kda-fp8", "row128"][..], Fp8)] {
            let args = parse(extra);
            assert_eq!(args.head_format, format, "{extra:?}");
            check_options(&args).unwrap();
            let settings: std::collections::BTreeMap<_, _> = native::resolved_settings(&args).into_iter().collect();
            assert_eq!(settings["fp8-head"], format.setting());
        }
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--fp8-head", "maybe"]).is_err());
        let error = check_options(&parse(&["--fp8-head", "packed", "--split-device", "1"])).unwrap_err().to_string();
        assert_eq!(error, "--fp8-head packed keeps one exact 12-bit head on one GPU; the two-GPU head split (--split-device) is not supported yet: use --fp8-head true or false");
        for format in ["true", "false"] {
            check_options(&parse(&["--fp8-head", format, "--split-device", "1"])).unwrap();
        }
        // A native checkpoint's own head ignores the setting.
        let mut native_args = parse(&["--fp8-head", "packed"]);
        native::normalize(&mut native_args);
        assert_eq!(native_args.head_format, Bf16);
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
        assert_eq!((defaults.kda_fp8, defaults.head_format, defaults.draft_fp8),
            (fp8::KdaFp8::Off, head::GlmfHeadFormat::Auto, None));
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
            (&["--prefill-rows", "2048", "--decode-rows", "128"][..], 2048), (&["--prefill-rows", "64"][..], 64)] {
            assert_eq!(parse(flags).expert_rows(), rows, "{flags:?}");
        }
        let args = parse(&["--peers", "127.0.0.1:19441", "--prefill-rows", "64", "--decode-rows", "128"]);
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
        // A whole reserve is one allocation per rank, even when a wider
        // verify step leaves its physical backing in the same allocation quantum.
        for waves in [3, 4] {
            let bytes = cuteafd_loader::serving_capacity::prefill_return_bytes(waves, 64, 4, 4096);
            assert_eq!(bytes, 4 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(
                waves * 64 * 4096 * 2).unwrap());
        }
    }

    #[test]
    fn expert_capacity_reserves_layer_free_sessions_and_charges_wider_intake() -> Result<()> {
        let mut cfg = GlmNextConfig::from_hf(&cuteafd_loader::plan::testing::glm5_flash_config(8))?;
        cfg.dense = (0..8).map(|layer| layer < 3).collect();
        cfg.hidden = 64;
        for (prefill, decode, capacity) in [(32, 64, 64), (64, 128, 128), (128, 128, 128), (256, 64, 256)] {
            let mut args = parse(&["--peers", "127.0.0.1:19441"]);
            args.prefill_rows = prefill;
            args.decode_rows = decode;
            args.prefill_waves = 3;
            assert_eq!(args.expert_rows(), capacity);
            for bf16 in [false, true] {
                let shape = engine::spark_session_shape(&cfg, args.expert_rows(), bf16)?;
                let stride = if bf16 { cfg.hidden * 2 } else { cfg.hidden + cfg.hidden / 32 };
                assert_eq!(shape.row_count as usize, capacity);
                assert_eq!(shape.route_count as usize, capacity * cfg.topk);
                assert_eq!(shape.hidden_payload_bytes as usize, capacity * stride);
            }
            for ranks in [1u64, 4] {
                use cuteafd_core::serving_capacity::spark_intake_allocation_bytes;
                use cuteafd_loader::serving_capacity::prefill_return_bytes;
                let waves = args.prefill_waves as u64;
                let hidden = cfg.hidden as u64;
                let charged = |rows| prefill_return_bytes(waves, rows, ranks, hidden);
                let allocation = |rows| spark_intake_allocation_bytes(waves * rows * hidden * 2);
                assert_eq!(charged(capacity as u64) - charged(prefill as u64),
                    ranks * (allocation(capacity as u64)? - allocation(prefill as u64)?));
            }
        }
        Ok(())
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
    fn shared_admission_rejects_family_headroom_and_preserves_onboarding_options() {
        for removed in [&["--headroom-gib", "1"][..], &["--graph-budget-mib", "512"]] {
            assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native"].into_iter()
                .chain(removed.iter().copied())).is_err(), "{removed:?}");
        }
        for (option, expected) in [("auto", "auto"), ("all", "all"), ("3", "3")] {
            assert_eq!(parse(&["--rtx-expert-layers", option]).rtx_expert_layers.unwrap().to_string(), expected);
        }
        for option in ["max", "50%", "x"] {
            assert!(option.parse::<cuteafd_loader::placement::OnboardSetting>().is_err(), "{option}");
        }
    }

    #[test]
    fn native_local_default_preserves_automatic_pool_policy_and_explicit_pins() {
        use cuteafd_loader::placement::Onboard;
        let local = parse(&["--local-experts", "--split-device", "1"]);
        assert_eq!(runtime_onboard(&local, true, 42, 42), Onboard::Auto);
        assert_eq!(runtime_onboard(&local, false, 42, 42), Onboard::Layers(42));
        for (option, expected) in [("auto", Onboard::Auto), ("all", Onboard::Layers(42)),
            ("3", Onboard::Layers(3))] {
            let args = parse(&["--local-experts", "--rtx-expert-layers", option]);
            assert_eq!(runtime_onboard(&args, true, 42, 42), expected);
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
    fn prefill_units_follow_shared_topology_and_preserve_explicit_pins() {
        for ranks in [0, 1, 2, 4, 5, 6, 8] {
            let peers = (0..ranks).map(|rank| format!("127.0.0.1:{}", 19441 + rank)).collect::<Vec<_>>().join(",");
            let args = if ranks == 0 { parse(&[]) } else { parse(&["--peers", &peers]) };
            let waves = cuteafd_loader::serving_capacity::prefill_waves::depth(ranks, Default::default());
            assert_eq!((args.prefill_waves as u64, args.prefill_rows as u64),
                (waves, cuteafd_loader::serving_capacity::prefill_waves::unit_rows(waves)));
        }
        let four = parse(&["--peers", SPARKS[1], "--prefill-waves", "4", "--prefill-rows", "2048"]);
        assert_eq!((four.prefill_waves, four.prefill_rows), (4, 2048));
        assert_eq!(parse(&["--prefill-rows", "2048"]).prefill_rows, 2048);
        for (peers, waves, rows) in [(None, 2, 4096), (Some(SPARKS[1]), 4, 4096)] {
            let mut args = parse(&[]);
            args.peers = peers.map(str::to_owned);
            args.prefill_waves = waves;
            args.prefill_rows = rows;
            assert!(resolve_prefill(&mut args, true).is_err());
        }
        assert!(Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--prefill-waves", "5"]).is_err());
    }

    #[test]
    fn layers_prefill_uses_shared_topology_and_pinned_waves() {
        use cuteafd_loader::serving_capacity::prefill_waves::{unit_rows, PrefillError, MAX_WAVES};
        for complete in [false, true] {
            for ranks in 0..=8 {
                let peers = (0..ranks).map(|rank| format!("worker{rank}:1")).collect::<Vec<_>>().join(",");
                for requested in std::iter::once(None).chain((1..=MAX_WAVES).map(Some)) {
                    let wave_flag = requested.map(|waves| waves.to_string());
                    let mut argv = vec!["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
                        "--attention-placement", "layers", "--split-device", "1"];
                    if ranks > 0 { argv.extend(["--peers", peers.as_str()]); }
                    if let Some(value) = &wave_flag { argv.extend(["--prefill-waves", value.as_str()]); }
                    let mut args = Parse::try_parse_from(argv).unwrap().engine;
                    let waves = requested.unwrap_or_else(|| cuteafd_loader::serving_capacity::prefill_waves::depth(if complete { ranks } else { 0 }, Default::default()));
                    let result = resolve_prefill(&mut args, complete);
                    if (!complete || ranks == 0) && waves > 1 {
                        assert!(matches!(result.unwrap_err().downcast_ref::<PrefillError>(),
                            Some(PrefillError::ExecutorCapacity { max_waves: 1, .. })));
                    } else {
                        result.unwrap();
                        let rows = unit_rows(waves);
                        assert_eq!((args.prefill_rows as u64, args.prefill_waves as u64, args.prefill_span_rows as u64),
                            (rows, waves, rows * waves), "complete={complete}, ranks={ranks}, requested={requested:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn prefill_span_is_exactly_the_shared_wave_extent() {
        assert_eq!(parse(&[]).prefill_span_rows, 4096);
        assert_eq!(parse(&["--prefill-span-rows", "4096"]).prefill_span_rows, 4096);
        let mut args = Parse::try_parse_from(["test", "--snapshot", "/checkpoint", "--native-lib", "/native",
            "--prefill-span-rows", "16384"]).unwrap().engine;
        assert!(resolve_prefill(&mut args, true).is_err());
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
    fn whole_push_inventory_follows_solved_owners_and_keeps_dense_only_ranks() -> Result<()> {
        use cuteafd_loader::placement::{ExpertHome, LayerAssignment, LayerMode};
        let mut cfg = GlmNextConfig::from_hf(&cuteafd_loader::plan::testing::glm5_flash_config(8))?;
        cfg.dense = (0..8).map(|layer| layer < 3).collect();
        for (gpus, owner) in [(1, 0), (2, 0), (2, 1)] {
            let mut assignments = vec![LayerAssignment { mode: LayerMode::HeadSplit, experts: ExpertHome::Spark }; 8];
            for layer in [1, 3, 6] { assignments[layer].experts = ExpertHome::RtxWhole { gpu: owner }; }
            let selected = admitted_whole_layers(&cfg, &assignments, gpus, false)?;
            assert_eq!(selected.len(), gpus);
            for (rank, layers) in selected.iter().enumerate() {
                assert_eq!(layers.as_slice(), if rank == usize::from(owner) { &[3, 6][..] } else { &[] });
            }
            assert!(admitted_whole_layers(&cfg, &assignments, gpus, true)?.iter().all(Vec::is_empty));
            assignments[6].experts = ExpertHome::RtxWhole { gpu: gpus as u8 };
            assert!(admitted_whole_layers(&cfg, &assignments, gpus, false).is_err());
        }
        Ok(())
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
            attention_by_kind: vec![], peer_row_bytes: 0, onboard_layers: 0, max_pool_tokens_no_onboard: None,
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
        assert_eq!((spark.peers.as_deref(), spark.local_experts, spark.prefill_rows), (Some(SPARKS[1]), false, 2752));
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
    /// Compare executor N=1 and the admitted depth at identical cuts, including recycled slots:
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
    ensure!(args.target_head != Some(head::TargetHead::Tensor) || args.head_format == head::GlmfHeadFormat::Bf16,
        "--target-head tensor runs the BF16 head; the {} head (--fp8-head {}) has its own GEMV",
        args.head_format.name(), args.head_format.setting());
    ensure!(args.head_format != head::GlmfHeadFormat::Packed || args.split_device.is_none(),
        "--fp8-head packed keeps one exact 12-bit head on one GPU; the two-GPU head split (--split-device) \
         is not supported yet: use --fp8-head true or false");
    Ok(())
}

impl EngineArgs {
    /// The target head's mode: explicit `--target-head`, else the shared setting (tensor by
    /// default). The FP8 head's own program ignores it.
    pub fn resolved_target_head(&self) -> head::TargetHead {
        head::resolve(self.target_head)
    }

    /// Whether serving captures every decode graph at startup (on unless `CUTEAFD_GLMF_GRAPHS=0`, which
    /// runs decode eagerly and charges no graph bytes).
    pub(crate) fn startup_graphs(&self) -> bool {
        engine::startup_graphs_enabled()
    }

    /// The rows every routed-expert resource holds (the dense NVFP4 package, local FP8 or EXL3
    /// experts, the Spark transports and their intake planes): the widest step, a prefill lane of
    /// `--prefill-rows` or a decode or verify step of `--decode-rows` (with `--prefill-rows 64
    /// --decode-rows 128`, a verify step's 127 rows).
    pub(crate) fn expert_rows(&self) -> usize {
        cuteafd_loader::serving_capacity::glmf_expert_rows(self.prefill_rows as u64, self.decode_rows as u64) as usize
    }
}

/// The step settings these arguments give the engine (its step plan's inputs besides layers and experts),
/// with the DSA index cache `with_engine` resolved (a head split keeps `keys`).
fn step_settings(args: &EngineArgs, index_cache: engine::IndexCache) -> engine::StepSettings {
    engine::StepSettings { target_head: args.resolved_target_head(), kda_fp32_partials: args.kda_fp32_partials, kda_output_shard: args.kda_output_shard,
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

fn parse_onboard(text: &str) -> std::result::Result<cuteafd_loader::placement::OnboardSetting, String> {
    text.parse()
}

fn runtime_onboard(args: &EngineArgs, native: bool, routed: usize, experts: usize)
    -> cuteafd_loader::placement::Onboard {
    args.rtx_expert_layers.map(|setting| setting.resolve(routed)).unwrap_or_else(|| {
        if args.local_experts && !native { cuteafd_loader::placement::Onboard::Layers(experts) }
        else { cuteafd_loader::placement::families::glm5_flash::default_onboard() }
    })
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

fn admitted_whole_layers(cfg: &GlmNextConfig, assignments: &[cuteafd_loader::placement::LayerAssignment],
    gpus: usize, skip: bool) -> Result<Vec<Vec<usize>>> {
    ensure!((1..=2).contains(&gpus), "GLM Flash coordinator rank count");
    ensure!(skip || assignments.iter().enumerate().all(|(layer, assigned)|
        cfg.dense.get(layer).copied().unwrap_or(false)
            || !matches!(assigned.experts, cuteafd_loader::placement::ExpertHome::RtxWhole { gpu } if usize::from(gpu) >= gpus)),
        "GLM Flash whole expert owner is outside the coordinator group");
    (0..gpus).map(|rank| admitted_expert_layers(cfg, assignments,
        cuteafd_loader::placement::ExpertHome::RtxWhole { gpu: rank as u8 }, skip)).collect()
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
    pub view: cuteafd_loader::plan::checkpoint::CheckpointView,
    pub expert_view: cuteafd_loader::plan::checkpoint::CheckpointView,
    pub checkpoint: Checkpoint,
    pub cfg: GlmNextConfig,
    pub library: std::sync::Arc<NativeLibrary>,
    /// The FP8 expert catalog for --local-experts.
    pub experts: Option<cuteafd_loader::OfficialV41Catalog>,
    pub native: bool,
}

impl Opened {
    fn fp8(&self) -> Option<&Fp8ExpertTensors> {
        self.experts.as_ref().and_then(|c| c.fp8())
    }
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    let view = args.attention.selected_view.clone().map(Ok)
        .unwrap_or_else(|| args.attention.select(&args.snapshot))?;
    let format = cuteafd_loader::plan::checkpoint::CheckpointFormat::detect_view(&view)?;
    let native = format == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8;
    let mut resolved = args.clone();
    if native {
        if args.head_format != head::GlmfHeadFormat::Bf16 {
            tracing::warn!(requested = args.head_format.setting(), "--fp8-head does not apply to a native checkpoint's own head");
        }
        native::normalize(&mut resolved);
        ensure!(args.exlr8_manifest.is_some(), "native Flash requires --exlr8-manifest k2, k3 or k4");
        ensure!(args.kda_state == engine::KdaState::F32, "native Flash requires F32 KDA state programs");
    } else { ensure!(args.exlr8_manifest.is_none(), "--exlr8-manifest requires a native EXLR8 checkpoint"); }
    let checkpoint = Checkpoint::coordinator_view(&view, false, false)?;
    let cfg = GlmNextConfig::from_hf(&checkpoint.config)?;
    let core = if native { Default::default() } else {
        cuteafd_loader::families::glm5_flash::core::CoreInventory::from_checkpoint(&cfg, &checkpoint)? };
    native::normalize_head(&mut resolved, core.get_weight("lm_head").is_some());
    native::publish_resolved(&resolved);
    let args = &resolved;
    check_options(args)?;
    cuteafd_bench::context::set_family(cfg.family_id());
    native::validate_headers(format, || {
        header::check_kda_inputs(&checkpoint, &cfg, args.layers.unwrap_or(cfg.layers))?;
        precision::check_projection_inputs(&checkpoint, &cfg, args.layers.unwrap_or(cfg.layers))?;
        Ok(())
    })?;
    if let Some(snapshot) = &args.draft {
        if !native && core.get_weight("lm_head").is_none() {
            let head = checkpoint.tensors.iter().find(|t| t.meta.name == "lm_head.weight")
                .context("DFlash target has no supported vocabulary head")?;
            crate::shared::speculation::dflash::check_target_head_source(&head.meta, cfg.hidden, cfg.vocab_size)?;
        }
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
    let expert_view = match args.experts_snapshot.as_deref() {
        Some(source) if source != args.snapshot => args.attention.select(source)?,
        _ => view.clone(),
    };
    let experts = if args.layers.unwrap_or(cfg.layers) > cfg.dense.iter().take_while(|&&d| d).count() {
        let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
        let mut catalog = cuteafd_loader::read_expert_catalog_view(&expert_view)?;
        catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::None)?;
        ensure!(catalog.fp8().is_some() || catalog.exl3().is_some()
            || catalog.format() == cuteafd_loader::plan::checkpoint::CheckpointFormat::Exlr8,
            "Flash experts need FP8, EXL3 or native EXLR8; {} has none", source.display());
        Some(catalog)
    } else {
        None
    };
    // SAFETY: the library is the cuteafd native shim built for this engine.
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&args.native_lib) }?);
    library.cuda_set_device(args.device)?;
    let mode = match view.mode() {
        cuteafd_loader::plan::checkpoint::AttentionMode::Hqa => "hqa",
        cuteafd_loader::plan::checkpoint::AttentionMode::Compact => "compact",
        cuteafd_loader::plan::checkpoint::AttentionMode::Fp8Compact => "fp8-compact",
    };
    cuteafd_bench::context::set_resolved("attention-mode", mode);
    cuteafd_bench::context::set_resolved("attention-bundle-fingerprint", view.fingerprint());
    let recipe = if view.mode() == cuteafd_loader::plan::checkpoint::AttentionMode::Fp8Compact {
        cuteafd_loader::plan::checkpoint::view::FP8_RECIPE
    } else { "none" };
    cuteafd_bench::context::set_resolved("attention-fp8-recipe", recipe);
    let active_bytes: u64 = checkpoint.tensors.iter().map(|tensor| tensor.meta.byte_length).sum();
    tracing::info!(attention_mode = mode, bundle = view.fingerprint(), fp8_recipe = recipe,
        active_dense_source_bytes = active_bytes, "selected immutable attention bundle");
    Ok(Opened { view, expert_view, checkpoint, cfg, library, experts, native })
}

/// Heads on two GPUs needs token keys; context needs compact pools. Whole MLA
/// owners can serve either cache, with compact tails only on the actual owner.
pub(crate) fn served_index_cache(requested: engine::IndexCache, split: bool,
    attention: cuteafd_loader::placement::AttentionPlacement) -> engine::IndexCache {
    use cuteafd_loader::placement::AttentionPlacement;
    let cache = match attention {
        AttentionPlacement::Context => engine::IndexCache::Compact,
        AttentionPlacement::Heads if split => engine::IndexCache::Keys,
        _ => requested,
    };
    cuteafd_bench::context::set_resolved("index-cache", match cache {
        engine::IndexCache::Keys => "keys", engine::IndexCache::Compact => "compact" });
    cache
}

/// The MLA placement an engine serves (`glm5_flash::resolve_attention` over the requested mode and
/// the build's context programs), with its programs' scratch.
fn resolve_attention(args: &EngineArgs, available: Option<engine::GlmfAttention>, split: bool)
    -> Result<engine::GlmfAttention> {
    let requested = cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?;
    let attention = cuteafd_loader::placement::families::glm5_flash::resolve_attention(requested,
        1 + usize::from(split), available)?;
    // Auto with pool marks keeps heads (context keeps marks in the arena).
    if requested.is_none() && args.prefix_marks == prefix::PrefixMarks::Pool {
        return Ok(engine::GlmfAttention { mode: cuteafd_loader::placement::AttentionPlacement::Heads, ..attention });
    }
    Ok(attention)
}

impl Opened {
    fn core_admission(&self, args: &EngineArgs, layers: usize, ranks: usize,
        attention: engine::GlmfAttention) -> Result<Option<cuteafd_loader::families::glm5_flash::core_admission::CoreCharges>> {
        use cuteafd_loader::families::glm5_flash::{core::CoreInventory, core_admission::CoreCharges, GlmNextAttention};
        if self.native { return Ok(None); }
        let headers = Checkpoint::coordinator_view(&self.view, false, false)?;
        let core = CoreInventory::from_checkpoint(&self.cfg, &headers)?;
        if core.projections.is_empty() { return Ok(None); }
        ensure!(args.kda_state == engine::KdaState::F32, "native K6 requires F32 KDA state programs");
        let owners = (attention.mode == cuteafd_loader::placement::AttentionPlacement::Layers).then(|| {
            let mut next = attention.first_owner as u8;
            self.cfg.attention[..layers].iter().map(|kind| {
                let owner = next;
                if *kind == GlmNextAttention::Mla { next ^= 1; }
                owner
            }).collect::<Vec<_>>()
        });
        let draft_rows = args.draft.as_ref().map(|path| -> Result<usize> {
            let config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
            Ok(cuteafd_loader::families::glm5::draft_representation::flash_draft_capacity(&config,
                args.draft_context_slots, args.draft_sequences)?.block_rows)
        }).transpose()?;
        let manifest = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
        Ok(Some(CoreCharges::from_package(&self.cfg, &core, layers, ranks, owners.as_deref(),
            args.prefill_rows, args.prefill_waves, args.decode_rows, draft_rows, args.kda_output_shard, &manifest)?))
    }

    fn admission_inventory(&self, args: &EngineArgs, programs: &cuteafd_ffi::programs::Programs<'_>,
        layers: usize, peer: Option<i32>, attention: engine::GlmfAttention,
        native_admission: Option<&native::Admission>) -> Result<crate::shared::inventory::RuntimeInventory> {
        let devices: Vec<_> = std::iter::once(args.device).chain(peer).collect();
        let split = devices.len() == 2;
        let core_charge = self.core_admission(args, layers, devices.len(), attention)?;
        let mut native_code = native_admission.map(|c| c.charges.programs.clone()).unwrap_or_default();
        if self.native {
            native_code.extend(cuteafd_loader::plan::exlr8::dense::programs::vocabulary(
                programs, self.cfg.hidden, self.cfg.vocab_size)?);
        }
        crate::shared::inventory::RuntimeInventory::measure(&self.library, &devices,
            crate::shared::inventory::InventoryPhase::PreWeight, |id| {
                programs.load_matching(|name| glmf_startup_program(name, split, args.decode_rows > engine::DECODE_ROWS)
                    || core_charge.as_ref().is_some_and(|c| c.programs.iter().any(|n| n == name))
                    || native_code.contains(name)
                    || (self.native && name.starts_with("exlr8_glmf_")))?;
                // Spark intake and the peer exchange load every registered native
                // module; load them before this device's sample so admission sees
                // their exact occupancy (#1554). Lazy module code would otherwise
                // land after the baseline, uncharged.
                self.library.tp2_native_initialize()?;
                if let Some(other) = devices.iter().copied().find(|&other| other != id) {
                    self.library.cuda_enable_peer(other)?;
                    self.library.peer_exchange_initialize()?;
                    self.library.peer_abort_initialize()?;
                }
                Ok(())
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_candidate(&self, args: &EngineArgs, programs: &cuteafd_ffi::programs::Programs<'_>, layers: usize,
        index: engine::IndexCache, marks: usize, peer: Option<i32>, context: cuteafd_loader::placement::planner::Context,
        remote_draft: bool, attention: engine::GlmfAttention,
        native_admission: Option<&native::Admission>,
        inventory: &crate::shared::inventory::RuntimeInventory)
        -> Result<(cuteafd_loader::placement::Placement, cuteafd_loader::placement::GraphSet,
            cuteafd_loader::placement::families::glm5_flash::GlmfWorkingSet,
            Option<cuteafd_hostcache::config::Config>,
            Option<cuteafd_loader::placement::families::glm5_flash::admission::NativeLoad>,
            Option<cuteafd_loader::push_load::PushHostEnvelope>,
            Option<cuteafd_loader::shared::dense_read::MountLoad>,
            cuteafd_loader::placement::planner::ResolvedPlan, Vec<String>)> {
        use cuteafd_loader::placement::{self, families::glm5_flash as admission};
        use cuteafd_loader::families::glm5_flash::resident::GlmfRepresentation;
        use cuteafd_loader::serving_capacity::{glmf_core_step_scratch, glmf_table_pages,
            GlmfScratchOptions, GlmfStepShape};
        ensure!(args.expert_window.is_none(), "solver admission does not support diagnostic expert paging");
        let devices: Vec<_> = std::iter::once(args.device).chain(peer).collect();
        let split = devices.len() == 2;
        let core_charge = self.core_admission(args, layers, devices.len(), attention)?;
        let mut native_code = native_admission.map(|c| c.charges.programs.clone()).unwrap_or_default();
        if self.native {
            native_code.extend(cuteafd_loader::plan::exlr8::dense::programs::vocabulary(
                programs, self.cfg.hidden, self.cfg.vocab_size)?);
        }
        let local_tp2_exl3 = split && args.peers.is_none() && !args.skip_experts
            && self.experts.as_ref().is_some_and(|c| c.exl3().is_some());
        let code_package = if self.native { "glmf:exlr8" } else if local_tp2_exl3 { "exl3-tp2-local" } else { "*" };
        let mut program_set = placement::ProgramSet::from_names(
            if self.native { &["glmf", "glmf2", "exlr8_glmf"][..] } else { &["glmf", "glmf2"][..] }, programs.names());
        program_set.names.extend(native_code);
        program_set.names.sort(); program_set.names.dedup();
        if let Some(core) = &core_charge {
            program_set.names.extend(core.programs.iter().cloned());
            program_set.names.sort(); program_set.names.dedup();
        }
        let pending_code = inventory.gpus.iter().enumerate().map(|(rank, sample)| {
            let arch = placement::ArchContext::for_device(&sample.arch, sample.total_bytes);
            let code = placement::inventory::loaded_code_allowance(
                if core_charge.is_some() { "glmf-k6" } else { "glmf" }, code_package,
                split, rank as u8, arch, Some(&program_set));
            code.pending(sample.context_bytes, arch.context_bytes)
        }).collect();
        let headers = Checkpoint::coordinator_view(&self.view, false, false)?;
        let representation = GlmfRepresentation { kda_fp8: args.kda_fp8 != fp8::KdaFp8::Off,
            head: args.head_format, output_shard: args.kda_output_shard };
        let resident = if let Some(native) = native_admission { native.resident.clone() } else {
            runtime_resident_weights(&headers, &self.cfg, layers, devices.len(), representation,
                args.token_io.embed_placement, attention)? };
        let spark_ranks = args.peers.as_deref().map_or(0, |p| p.split(',').count());
        let lanes = usize::try_from(admission::admitted_prefill_waves(attention.mode, args.prefill_waves as u64)?)?;
        let lookup = |name: &str| programs.spec(name).ok().map(|p| p.scratch.get("scratch").copied().unwrap_or(0));
        let scratch = GlmfScratchOptions { split, kda_w8: representation.kda_fp8,
            kda_fp32_partials: args.kda_fp32_partials, kda_output_shard: args.kda_output_shard,
            kda_prefill_expanded: args.kda_prefill_expanded, index_compact: index == engine::IndexCache::Compact,
            kda_state: args.kda_state.into() };
        let core = if self.native { Default::default() } else {
            cuteafd_loader::families::glm5_flash::core::CoreInventory::from_checkpoint(&self.cfg, &headers)? };
        let native = if self.native { cuteafd_loader::serving_capacity::GlmfCoreScratch { kda: true, mla: true, ffn: true } }
            else { core.scratch_mask(&self.cfg, layers) };
        let decode = glmf_core_step_scratch(lookup, &self.cfg, scratch, args.decode_rows as u64, true, native)?;
        let mut prefill = glmf_core_step_scratch(lookup, &self.cfg, scratch, args.prefill_rows as u64, false, native)?;
        if args.replay_records == engine::ReplayRecords::Shared {
            prefill.programs = prefill.programs.max(cuteafd_loader::serving_capacity::glm_flash_kda_replay_bytes_rows(
                &self.cfg, layers, devices.len(), args.decode_rows as u64)?);
        }
        let (table_pages, table_pool_pages) = glmf_table_pages(args.max_context as u64);
        let shape = GlmfStepShape { lead: true, split, local_experts: !args.skip_experts && !split && self.fp8().is_some(),
            tp2_experts: !args.skip_experts && split, spark: spark_ranks > 0, partial_bytes: if args.kda_fp32_partials { 4 } else { 2 },
            output_shard: args.kda_output_shard, full_prefill_logits: args.full_prefill_logits && !self.native && core.get_weight("lm_head").is_none(), table_pages, table_pool_pages };
        let workspaces = |spark, lanes| admission::step_workspace_demands(&self.cfg, lanes,
            args.prefill_rows as u64, args.decode_rows as u64, GlmfStepShape { spark, ..shape }, decode, prefill);
        let mut workspace = workspaces(spark_ranks > 0, lanes);
        let mut local_workspace = workspaces(false, lanes);
        if let Some(core) = &core_charge {
            for profile in [&mut workspace, &mut local_workspace] {
                for (rank, bytes) in core.bytes(self.cfg.hidden)?.into_iter().enumerate() {
                    profile.push(admission::GlmfBufferDemand::new(admission::GlmfComponent::Always, rank as u8,
                        cuteafd_core::memory_layout::Category::Workspace, "native K6 core", bytes,
                        cuteafd_core::memory_layout::Basis::Exact));
                }
            }
        }
        if let Some(native) = native_admission {
            for profile in [&mut workspace, &mut local_workspace] {
                for (rank, bytes) in native.charges.bytes()?.into_iter().enumerate() {
                    profile.push(admission::GlmfBufferDemand::new(admission::GlmfComponent::Always, rank as u8,
                        cuteafd_core::memory_layout::Category::Workspace, "native EXLR8 core/shared expert", bytes,
                        cuteafd_core::memory_layout::Basis::Exact));
                }
            }
        }
        let mut experts = if let Some(native) = native_admission {
            let routed = (0..layers).filter(|&layer| !self.cfg.dense[layer]).collect::<Vec<_>>();
            cuteafd_loader::plan::exlr8::routed_expert_costs(&native.metadata, &routed)?
        } else { self.experts.as_ref().map(|c| admission::expert_costs(c, split)).transpose()?.unwrap_or_default() };
        experts.truncate(self.cfg.dense[..layers].iter().filter(|&&d| !d).count());
        if args.skip_experts { for cost in &mut experts { cost.whole = Default::default(); cost.half = Default::default(); } }
        let mut tp2_workspace = [0; 2];
        let mut expert_workspace = 0;
        if let Some(native) = native_admission.filter(|_| !args.skip_experts) {
            let manifest = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
            let bytes = cuteafd_loader::plan::exlr8::native::flash_routed_workspace(&native.metadata,
                &self.cfg, &manifest, devices.len(), args.expert_rows() as u64)?;
            if split { tp2_workspace = [bytes; 2]; } else { expert_workspace = bytes; }
        } else if let Some(catalog) = self.experts.as_ref().filter(|_| !args.skip_experts) {
            if split {
                if let Some(tensors) = catalog.fp8() {
                    let package = args.fp8_package.clone().unwrap_or_else(|| crate::shared::experts::fp8::package_directory(
                        &args.native_lib, 2, tensors.format()));
                    for rank in 0..2 { tp2_workspace[rank] = crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::plan(
                        tensors, &package, &[tensors.shape().first_layer], rank, args.expert_rows())?.workspace_bytes as u64; }
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
        let mut draft_tp2 = Vec::new();
        let (drafter_bytes, drafter_weights, drafter_scratch, drafter_bridge) = args.draft.as_ref().map(|path| -> Result<_> {
            let config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
            let capacity = cuteafd_loader::families::glm5::draft_representation::flash_draft_capacity(&config,
                args.draft_context_slots, args.draft_sequences)?;
            let (slots, batch) = (capacity.context_slots, capacity.max_batch_sequences);
            if args.tp2_dflash {
                draft_tp2 = admission::drafter_tp2_inventory(path, capacity, &draft_sms,
                    cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::from_fp8_option(args.draft_fp8),
                    args.draft_linear.code() as u8, args.draft_kv_cache)?;
                return Ok((0, placement::ResidentWeights::default(), vec![0; devices.len()], None));
            }
            let (bytes, scratch) = admission::drafter_inventory(&config, slots, batch, &draft_sms,
                cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::from_fp8_option(args.draft_fp8),
                args.draft_linear.code() as u8, args.draft_kv_cache)?;
            let bridge = if remote_draft { admission::drafter_bridge_inventory(&config, slots, batch)? }
                else { None };
            Ok((bytes.memory.resident, bytes.resident_weights, scratch, bridge))
        }).transpose()?.unwrap_or_else(|| (0, placement::ResidentWeights::default(), vec![0; devices.len()], None));
        let (sequences, speculation) = args.serving_graph_policy.unwrap_or((16, args.draft.is_some()));
        let pool = if args.pool_tokens > 0 { args.pool_tokens } else { cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS as usize };
        let graphs = if args.startup_graphs() {
            admission::startup_graph_bound(&self.cfg, layers, args.max_context, pool,
                sequences, speculation, devices.len(), args.decode_rows, sms).map_err(anyhow::Error::msg)?
        } else {
            // `CUTEAFD_GLMF_GRAPHS=0` captures no graphs and allocates none.
            placement::GraphSet::budget(&vec![cuteafd_loader::placement::reserve::gpu_component(0); devices.len()])
        };
        let routed = self.cfg.dense.iter().filter(|dense| !**dense).count();
        let onboard = runtime_onboard(args, self.native, routed, experts.len());
        let inputs = admission::GlmfInputs {
            resident_weight_limit: Some(crate::shared::memory::budget::resident_weight_limit()), spark_reply_format: cuteafd_core::SparkReplyFormat::resolve(cuteafd_core::spark_reply::requested().prefill, false), context, attention, cfg: &self.cfg, layers, local_tp2_exl3: local_tp2_exl3 && core_charge.is_none(),
            gpus: inventory.gpus.iter().map(|gpu| (gpu.total_bytes, placement::Baseline::Measured {
                free_bytes: gpu.free_bytes })).collect(),
            pending_code, headroom_bytes: cuteafd_loader::placement::reserve::gpu("glm5_flash"), spark_ranks,
            prefill_lanes: lanes as u64, prefill_rows: args.prefill_rows as u64, decode_rows: args.decode_rows as u64,
            partial_bytes: if args.kda_fp32_partials { 4 } else { 2 }, max_context: args.max_context as u64,
            sequences: sequences.min(args.decode_rows) as u64, speculation, state_slots: args.slots as u64,
            mark_slots: marks as u64, pool_marks: args.prefix_marks == prefix::PrefixMarks::Pool, index: index.into(),
            kda_state_bytes: args.kda_state.bytes() as u64, shared_replay: args.replay_records == engine::ReplayRecords::Shared,
            blas_head: args.resolved_target_head().blas_head(!self.native && core.get_weight("lm_head").is_none()
                && representation.head == head::GlmfHeadFormat::Bf16),
            representation, resident, workspace, local_workspace, graphs, experts, expert_workspace, tp2_workspace,
            drafter_bytes, drafter_weights, drafter_scratch, drafter_bridge, requested_pool: (args.pool_tokens > 0).then_some(args.pool_tokens as u64), onboard,
            full_prefill_logits: if shape.full_prefill_logits {
                cuteafd_loader::plan::layout::full_prefill_logits_bytes("glm5_flash", args.prefill_rows as u64,
                    self.cfg.vocab_size as u64)
            } else { 0 } };
        let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
        let headroom = args.serving_prefix.as_ref().map_or(8 << 30, |prefix| prefix.host_cache_headroom_bytes).max(host.0 / 10);
        let mut process = admission::admission::ProcessOwners::new(host, headroom);
        if split && attention.mode == placement::AttentionPlacement::Layers && native_admission.is_none()
            && core_charge.is_none() {
            process.alternate_resident = Some(runtime_resident_weights(&headers, &self.cfg, layers, devices.len(),
                representation, args.token_io.embed_placement,
                engine::GlmfAttention { first_owner: attention.first_owner ^ 1, ..attention })?);
        }
        process.target(&headers, &inputs, native_admission.map(|native| (&native.facts, &native.metadata)))?;
        let geometry = admission::admission::host_geometry(&inputs)?;
        let host_config = args.serving_prefix.as_ref().map(|prefix| {
            use cuteafd_engine::prefix::{FamilyLayout, MarkStore, PageOwners};
            prefix.resolve_host_config(FamilyLayout { page_rows: 256, pages: 0,
                page_bytes: usize::try_from(geometry.page_bytes)?, mark_bytes: usize::try_from(geometry.mark_bytes)?,
                draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT, mark_store: MarkStore::Arena,
                page_owners: if attention.mode == placement::AttentionPlacement::Context { PageOwners::Parity } else { PageOwners::Uniform } },
                args.max_context, sequences)
        }).transpose()?.flatten();
        let host_embedding = if args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Host {
            headers.tensors.iter().find(|tensor| tensor.meta.name == "model.language_model.embed_tokens.weight")
                .context("host embedding tensor missing")?.meta.byte_length
        } else { 0 };
        process.encoders = args.serving_encoder.iter().cloned().collect();
        process.physical_sparks = process.encoders.iter().flat_map(|encoder| encoder.spark_ranks())
            .max().map_or(0, |rank| rank + 1);
        process.media_cache_bytes = args.serving_media_bytes;
        process.serving(&inputs, host_config.as_ref().map_or(0, |c| c.bytes),
            host_config.as_ref().map_or(0, |c| c.chunk_bytes), host_embedding, 0)?;
        process.spark_link(&inputs, self.checkpoint.spark_input_is_bf16())?;
        process.native_load = native_admission.map(|native| native.load_owners(&self.cfg)).transpose()?;
        process.push_host = admission::admission::push_host_envelope(&headers, &self.cfg, devices.len(),
            self.experts.as_ref(),
            native_admission.map(|native| (&native.facts, &native.metadata)))?;
        process.draft_tp2 = draft_tp2;
        if let Some(path) = &args.draft {
            process.draft(path,
                cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::from_fp8_option(args.draft_fp8),
                devices.len(), args.tp2_dflash)?;
        }
        process.draft_push = self.draft_push(args)?;
        if native_admission.is_none() && process.push_host.is_none() {
            process.mount_load = Some(admission::admission::mount_load(&headers, devices.len())?);
        }
        let admitted = admission::admission::prepare(&inputs, sms, true, process)?.admit()?;
        // Under TP2_DFLASH auto both drafter candidates are admitted; each logs its breakdown.
        tracing::info!(target: "cuteafd::memory_policy", tp2_dflash = args.tp2_dflash, "GLM Flash admission candidate");
        cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "glm5_flash");
        tracing::info!(graph_slack = ?admitted.graph_slack, phases = ?admitted.solved.phases,
            "GLM Flash admitted exact physical owners");
        let (placement, graphs, working) = (admitted.placement, admitted.graphs, admitted.working);
        ensure!(placement.movables.iter().all(|(id, gpu)| *gpu == 0 || *id == placement::MovableId::Drafter),
            "GLM Flash placement has no matching movable executor");
        placement::draft::require_home(&placement, "GLM Flash",
            if remote_draft { &[0, 1] } else { &[0] })?;
        Ok((placement, graphs, working, host_config, admitted.native_load, admitted.push_host,
            admitted.mount_load, admitted.solved, admitted.declared_graphs))
    }

    /// Builds the engine and hands it to `body`. Either KV admission keeps the prefix marks of
    /// `args.mark_arena` the caller allocates once the engine exists free, counted on the layout
    /// the engine serves (`engine.mark_slots`).
    pub fn with_engine<'s, T>(&'s self, args: &EngineArgs, body: impl FnOnce(&engine::GlmfEngine<'_>) -> Result<T>)
        -> Result<T> {
        self.with_engine_media(args, || Ok(Default::default()), body)
    }
    pub(super) fn with_engine_media<'s, T>(&'s self, args: &EngineArgs,
        prepare_media: impl FnOnce() -> Result<crate::shared::experts::rtx::dense::Records>,
        body: impl FnOnce(&engine::GlmfEngine<'_>) -> Result<T>) -> Result<T> {
        // A head split whose GPUs lack peer access serves from --device alone, decided before any load or
        // admission. The arena is counted after both decisions (below), on the layout this engine serves:
        // each admitted GPU reserves its part of every mark (a lone GPU, the whole mark), as the prefix
        // cache then allocates.
        let probe_devices: Vec<_> = std::iter::once(args.device).chain(args.split_device).collect();
        let probe = |stage| crate::shared::inventory::startup_probe(&self.library, &probe_devices, stage, Default::default());
        probe("contexts-before-peer")?;
        let mut resolved = args.clone();
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let core = if self.native { Default::default() } else {
            cuteafd_loader::families::glm5_flash::core::CoreInventory::from_checkpoint(&self.cfg, &self.checkpoint)? };
        if self.native { native::normalize(&mut resolved); }
        if core.scratch_mask(&self.cfg, layers).kda {
            // Native K6 reduces unrotated F32 cores, not FP8 KDA partials.
            resolved.kda_fp8 = fp8::KdaFp8::Off;
            resolved.kda_fp32_partials = false;
            resolved.kda_output_shard = false;
            resolved.kda_prefill_expanded = false;
        }
        native::normalize_head(&mut resolved, core.get_weight("lm_head").is_some());
        if args.split_device.is_some()
            && crate::shared::peer_split::probed_device(&self.library, args.device, args.split_device)?.is_none() {
            resolved.split_device = None;
            resolved.kda_fp32_partials = false;
            resolved.kda_output_shard = false;
            resolved.kda_prefill_expanded = false;
        }
        probe("peer-enabled")?;
        native::publish_resolved(&resolved);
        cuteafd_bench::context::set_resolved("split-device", &resolved.split_device.map_or_else(|| "none".into(), |device| device.to_string()));
        let args = &resolved;
        let requested_context = (args.max_context > 0).then_some(args.max_context as u64);
        let mut context_args = args.clone();
        resolve_prefill(&mut context_args, args.layers.is_none_or(|n| n == self.cfg.layers))?;
        context_args.max_context = crate::shared::context::checkpoint_context_config(
            &self.checkpoint.config, &args.manifest, "glm5_flash", args.max_context)?;
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
        if args.head_format == head::GlmfHeadFormat::Fp8 {
            needed.push("glmf_head_fp8");
        }
        if args.head_format == head::GlmfHeadFormat::Packed {
            // On the head's GPU: the native GEMV fixes its SM count at the first query.
            crate::shared::memory::device::Device { library: &self.library, id: args.device }.run(||
                self.library.packed_head_workspace(cuteafd_core::packed_head::PASS_ROWS as usize, self.cfg.hidden,
                    self.cfg.vocab_size)).context("--fp8-head packed needs the shared packed head kernels \
                (cuteafd_packed_head_*); this native library predates them")?;
        }
        let compact = args.index_cache == engine::IndexCache::Compact;
        if compact && !self.native {
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
        // Resolve card-specific recipes before choosing cache and operand owners.
        let requested = cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?;
        let available_manifest = if split_device.is_some() && !matches!(requested,
            Some(cuteafd_loader::placement::AttentionPlacement::Heads | cuteafd_loader::placement::AttentionPlacement::Layers))
            && !(requested.is_none() && args.prefix_marks == prefix::PrefixMarks::Pool) {
            let fallback = args.manifest.with_file_name("DSV4_PROGRAMS.json");
            let path = if !args.manifest.exists()
                && args.manifest.file_name() == Some("PROGRAMS.json".as_ref()) && fallback.exists() {
                &fallback
            } else { &args.manifest };
            Some(serde_json::from_slice::<serde_json::Value>(&std::fs::read(path)?)?)
        } else { None };
        let available = available_manifest.as_ref().map(|manifest| -> Result<_> {
            let sms = std::iter::once(args.device).chain(split_device).map(|device|
                crate::shared::peer_split::on_device(&self.library, device, || self.library.sm_count()))
                .collect::<Result<Vec<_>>>()?;
            cuteafd_loader::placement::families::glm5_flash::manifest_context_for_cards(
                manifest, self.cfg.heads, &sms, attention_plan)
        }).transpose()?.flatten();
        let attention = resolve_attention(args, available, split_device.is_some())?;
        let context_manifest = (attention.mode == cuteafd_loader::placement::AttentionPlacement::Context)
            .then_some(available_manifest).flatten();
        let context_mode = (attention.mode == cuteafd_loader::placement::AttentionPlacement::Context).then_some(attention);
        let index_cache = served_index_cache(args.index_cache, split_device.is_some(), attention.mode);
        if let (Some(device), true, None) = (split_device, index_cache != args.index_cache, context_mode) {
            tracing::warn!(device, "--index-cache compact is single-GPU under heads; the head split keeps the token keys");
        }
        cuteafd_bench::context::set_resolved("attention-placement", &attention.mode.to_string());
        cuteafd_bench::context::set_resolved("index-cache", match index_cache { engine::IndexCache::Compact => "compact", engine::IndexCache::Keys => "keys" });
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
        let loader = weights::GlmfLoader { arenas: Default::default(), library: &self.library, checkpoint: &self.checkpoint, stream,
            kda_fp8: args.kda_fp8, kda_output_shard: args.kda_output_shard,
            // A packed head lands as BF16 and packs once every operand has landed (below).
            fp8_head: args.head_format == head::GlmfHeadFormat::Fp8,
            packed_head: args.head_format == head::GlmfHeadFormat::Packed,
            kda_nvfp4: args.kda_nvfp4_gate.as_deref().map(|mode| mode == "search"),
            fp8_scales: args.fp8_scales, device: args.device,
            peers: peer_stream.iter().map(|&(device, stream)| crate::shared::peer_split::RankDevice { device, stream })
                .collect(), dense: std::rc::Rc::new(weights::DenseLoad::mount(&self.library)) };
        ensure!(!args.tp2_dflash || (split_device.is_some() && args.draft.as_ref()
            .is_some_and(|path| dspark::is_dspark(path).is_ok_and(|plain| !plain))),
            "TP2 DFlash requires two coordinator GPUs and a GQA DFlash2 checkpoint");
        // GPU1-whole has an unresolved cold-replay mismatch; keep it out of admission.
        let remote_draft = false;
        let native_admission = self.native.then(|| {
            let manifest = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
            native::Admission::new(args, &self.cfg, layers, 1 + usize::from(split_device.is_some()),
                attention, index_cache, &manifest)
        }).transpose()?;
        if native_admission.is_some() {
            use cuteafd_loader::plan::family::Family;
            cuteafd_loader::plan::families::glm::GLM5_FLASH.exlr8_execution()
                .check(1 + usize::from(split_device.is_some()), false)?;
        }
        let inventory = self.admission_inventory(args, &programs, layers, split_device, attention,
            native_admission.as_ref())?;
        let mut draft_args = args.clone();
        let admitted = if args.tp2_dflash_auto {
            draft_args.tp2_dflash = false;
            let whole = self.admit_candidate(&draft_args, &programs, layers, index_cache, mark_slots,
                split_device, cuteafd_loader::placement::planner::Context::resolve(args.max_context as u64, requested_context), false, attention, native_admission.as_ref(), &inventory);
            draft_args.tp2_dflash = true;
            let split = self.admit_candidate(&draft_args, &programs, layers, index_cache, mark_slots,
                split_device, cuteafd_loader::placement::planner::Context::resolve(args.max_context as u64, requested_context), false, attention, native_admission.as_ref(), &inventory);
            let whole_pool = whole.as_ref().ok().map(|a| a.0.pool_tokens);
            let split_pool = split.as_ref().ok().map(|a| a.0.pool_tokens);
            draft_args.tp2_dflash = cuteafd_loader::placement::families::glm5_flash::prefer_tp2_dflash(whole_pool, split_pool);
            tracing::info!(whole_pool, split_pool, split = draft_args.tp2_dflash, "TP2_DFLASH auto selection");
            if draft_args.tp2_dflash { split } else { whole }
        } else {
            self.admit_candidate(args, &programs, layers, index_cache, mark_slots,
                split_device, cuteafd_loader::placement::planner::Context::resolve(args.max_context as u64, requested_context), remote_draft, attention, native_admission.as_ref(), &inventory)
        }?;
        let args = &draft_args;
        let remote_draft = false;
        cuteafd_bench::context::set_resolved("tp2-dflash", if args.tp2_dflash { "on" } else { "off" });
        let (placement, admitted_graphs, working, admitted_host, admitted_native_load,
            admitted_push_host, admitted_mount_load, mut admitted_plan, declared_graphs) = admitted;
        if let Some(receipt) = admitted_mount_load { loader.dense.admit_mount(receipt)?; }
        let media = prepare_media()?;
        cuteafd_loader::placement::families::GLM5_FLASH.check(&placement)?;
        if native_admission.is_some() {
            use cuteafd_loader::plan::family::Family;
            cuteafd_loader::plan::families::glm::GLM5_FLASH.exlr8_execution()
                .check(1 + usize::from(split_device.is_some()), placement.onboard_layers > 0)?;
        }
        let attention = cuteafd_loader::placement::families::glm5_flash::placed_attention(attention, &placement);
        ensure!(placement.attention_placement == attention.mode,
            "GLM Flash admission placed {} attention, the engine serves {}",
            placement.attention_placement, attention.mode);
        tracing::info!(placement = %placement.summary(), spark_ranks = working.spark_ranks,
            prefill_lanes = working.prefill_lanes, spark_layers = ?working.spark_layers,
            attention_first_owner = attention.first_owner, "GLM Flash admission before weights");
        let devices: Vec<_> = placement.items.iter().zip(std::iter::once(args.device).chain(split_device))
            .enumerate().map(|(rank, (items, device))|
                serde_json::json!({"kind": "rtx", "index": rank, "device": device, "items": items})).collect();
        let (sequences, speculation) = args.serving_graph_policy.unwrap_or((16, args.draft.is_some()));
        tracing::info!(target: "cuteafd::placement", plan = %serde_json::json!({
            "memory_layout": {"devices": devices, "pool_tokens": placement.pool_tokens},
            "graph_shapes": admitted_graphs.shapes,
            "graphs": admitted_graphs.ranks.iter().map(|r| r.executables).collect::<Vec<_>>(),
            "speculation": speculation, "sequences": sequences,
            "attention_first_owner": attention.first_owner,
            "layer_modes": placement.layers.iter().map(|layer| layer.mode).collect::<Vec<_>>(),
        }), "GLM Flash startup plan");
        cuteafd_bench::context::set_rtx_expert_layers(placement.onboard_layers);
        let draft_gpu = cuteafd_loader::placement::draft::home(&placement).unwrap_or(0);
        let mut prepared_draft = if draft_gpu == 1 {
            ensure!(remote_draft, "remote drafter was not declared before admission");
            let path = args.draft.as_ref().context("admitted remote draft checkpoint")?;
            let (device, peer_stream) = peer_stream.context("remote draft needs the peer target stream")?;
            let config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
            let capacity = cuteafd_loader::families::glm5::draft_representation::flash_draft_capacity(&config,
                args.draft_context_slots, args.draft_sequences)?;
            let draft = crate::shared::speculation::dflash::DflashConfig::read(path)?;
            Some(crate::shared::draft::remote::Prepared::new(&self.library,
                crate::shared::peer_split::RankDevice { device: args.device, stream },
                crate::shared::peer_split::RankDevice { device, stream: peer_stream },
                0, "glmf-remote-dflash:taps,input,hidden,logits", crate::shared::draft::remote::decode_extent(
                    capacity.block_rows, draft.hidden, draft.vocab, 0)?)?)
        } else { None };
        probe("admission-programs")?;
        let mut selected_args = args.clone();
        selected_args.prefill_waves = usize::try_from(working.prefill_lanes)?;
        cuteafd_bench::context::ResolvedSettings::placement(&placement).publish();
        cuteafd_bench::context::set_resolved("prefill-waves", &selected_args.prefill_waves.to_string());
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
        ensure!(tp2_layers.is_empty() || startup_graphs, "GLM Flash TP2 experts require the startup \
            decode graphs; CUTEAFD_GLMF_GRAPHS=0 disables them (runtime captures are not admitted)");
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
        use crate::shared::experts::rtx::push::{CoordinatorExperts, CoordinatorPush, PushedExperts, PushPolicy, WholeOwner};
        let push_policy = PushPolicy::for_snapshot(&args.snapshot)?;
        // Native checkpoints describe their row-FP8 vocabulary in the native manifest.
        let embed = native::legacy_embedding_source(native_admission.is_some(), || self.embed_source())?;
        let embedding_rank = cuteafd_loader::placement::families::glm5_flash::embedding_home(&placement) as usize;
        let embedding_device = crate::shared::memory::device::Device { library: &self.library,
            id: if embedding_rank == 0 { args.device } else { split_device.context("embedding peer GPU")? } };
        // Every dense rank participates, including ranks without routed experts.
        // Namespace identity alone selects push; planning cannot change that route.
        let gpus = 1 + usize::from(peer_stream.is_some());
        let whole_layers = admitted_whole_layers(&self.cfg, &placement.layers, gpus, args.skip_experts)?;
        ensure!(whole_layers.iter().skip(1).all(Vec::is_empty),
            "GLM Flash executor only binds whole experts on coordinator rank zero");
        let devices: Vec<_> = std::iter::once(args.device).chain(peer_stream.map(|(d, _)| d))
            .map(|id| crate::shared::memory::device::Device { library: &self.library, id }).collect();
        let layouts = if let Some(native) = &native_admission {
            native.arena_layouts(&self.cfg, embedding_rank)?
        } else {
            let embedding = (args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu)
                .then(|| embed.as_ref().context("Flash embedding source").and_then(|source|
                    source.allocation_request("embedding.table", cuteafd_loader::placement::planner::EmbeddingHome::Device(embedding_rank as u8))))
                .transpose()?.map(|request| (embedding_rank, request.bytes as usize));
            weight_layout::target(&self.checkpoint, &self.cfg, execution.modes(), gpus, args.kda_fp8 != fp8::KdaFp8::Off,
                args.kda_output_shard, args.head_format, embedding)?
        };
        let arenas = layouts.into_iter().zip(&devices).map(|(layout, device)| {
            Ok((device.id, crate::shared::memory::weight_arena::DeviceWeightArena::new(*device, layout, cuteafd_core::memory_layout::ArenaTag::weights("glmf/target"))?))
        }).collect::<Result<std::collections::BTreeMap<_, _>>>()?;
        let loader = weights::GlmfLoader { arenas: std::rc::Rc::new(arenas), ..loader };
        let plan_dense = |sparks: usize| -> Result<_> {
            ensure!(dense_pushable(args), "GLM Flash NVFP4 measurement mode cannot record required push operands");
            let load = std::rc::Rc::new(weights::DenseLoad::push(&self.library));
            let planning = weights::GlmfLoader { dense: load.clone(), ..loader.clone() };
            let ((embedding, model), mut records) = crate::shared::experts::rtx::dense::recording(|| {
                embedding_device.run(|| crate::shared::token_io::TokenEmbedding::load_with(
                    &self.library, embed.as_ref().context("legacy Flash embedding source")?.clone(), args.token_io.embed_placement, || {
                        devices[0].run(|| planning.model(&self.cfg, &execution))
                    }, |name, bytes| planning.allocate(embedding_device.id, name, bytes)))
            })?;
            records.extend(load.sink.as_ref().context("push sink")?.take_all());
            records.extend(media.clone());
            let dense = crate::shared::experts::rtx::dense::plan_records(&args.snapshot, records, &devices, sparks)?;
            Ok((embedding, model, dense))
        };
        let expert_plan = || -> Result<Option<CoordinatorExperts>> {
            let catalog = self.experts.as_ref();
            if !tp2_layers.is_empty() {
                ensure!(placement.layers.iter().enumerate().all(|(layer, assigned)| self.cfg.dense[layer]
                    || !matches!(assigned.experts, cuteafd_loader::placement::ExpertHome::RtxWhole { .. })),
                    "GLM Flash combined whole and TP2 expert push requires a mixed coordinator plan");
                let catalog = catalog.context("TP2 expert catalog")?;
                let budgets = tp2_budgets()?;
                let budgets = (0..2).map(|rank| {
                    let workspace = if let Some(manifest) = catalog.exl3() {
                        crate::shared::experts::rtx::exl3::Exl3Tp2::workspace_bytes_for(
                            &tp2_package(manifest), self.cfg.hidden, args.expert_rows())?
                    } else {
                        let tensors = catalog.fp8().context("FP8 TP2 tensors")?;
                        let package = args.fp8_package.clone().unwrap_or_else(||
                            crate::shared::experts::fp8::package_directory(&args.native_lib, 2, tensors.format()));
                        crate::shared::experts::rtx::fp8moe::Fp8MoeTp2::plan(tensors, &package, &tp2_layers, rank,
                            args.expert_rows())?.workspace_bytes
                    };
                    budgets[rank].checked_sub(workspace).context("GLM Flash TP2 workspace exceeds admitted peak")
                }).collect::<Result<Vec<_>>>()?;
                return Ok(Some(CoordinatorExperts::Tp2 { layers: tp2_layers.clone(), budgets }));
            }
            let mut owners = Vec::new();
            for (rank, selected) in whole_layers.iter().enumerate() {
                if selected.is_empty() { continue; }
                let catalog = catalog.context("whole expert catalog")?;
                let peak = usize::try_from(placement.expert_ranges.get(rank).context("whole expert arena")?.peak_bytes)?;
                let selections = if catalog.exl3().is_some() {
                    let (selections, _) = crate::families::deepseek_v4::local::LocalExperts::exl3_selections(
                        &args.native_lib, catalog, 0, selected, args.expert_rows(), peak)?
                        .context("GLM Flash whole EXL3 package")?;
                    ensure!(selections.len() == selected.len(), "GLM Flash admitted whole EXL3 inventory does not fit");
                    selections
                } else {
                    selected.iter().map(|&layer| crate::shared::experts::layer::ExpertLayer::BackboneFull { layer }).collect()
                };
                let resident = if let Some(tensors) = catalog.fp8() {
                    let tensors = tensors.for_coordinator();
                    crate::shared::experts::fp8::Fp8Layer::bytes_for(&tensors, 1, 0, tensors.exact_slicing(1))?
                        .checked_mul(selected.len()).context("GLM Flash whole resident size overflow")?
                } else {
                    selections.iter().try_fold(0usize, |total, &selection| {
                        total.checked_add(crate::shared::experts::exl3::Exl3Weights::plan(catalog, selection)?.resident_bytes)
                            .context("GLM Flash whole resident size overflow")
                    })?
                };
                ensure!(resident <= peak, "GLM Flash whole expert residency exceeds admission");
                owners.push(WholeOwner { rank, selections, budget: resident });
            }
            Ok((!owners.is_empty()).then_some(CoordinatorExperts::Whole { owners }))
        };
        let local_layers = admitted_expert_layers(&self.cfg, &placement.layers,
            cuteafd_loader::placement::ExpertHome::RtxWhole { gpu: 0 }, args.skip_experts)?;
        let native_local_budget = || -> Result<usize> {
            usize::try_from(placement.expert_ranges.first().context("native local expert admission")?.peak_bytes)
                .context("native local expert budget exceeds address space")
        };
        let native_push_experts = || -> Result<_> {
            let selected = if tp2_layers.is_empty() { &local_layers } else { &tp2_layers };
            if selected.is_empty() { return Ok(None); }
            ensure!(tp2_layers.is_empty() || local_layers.is_empty(), "native Flash mixed whole/TP2 expert homes");
            let native = native_admission.as_ref().context("native expert admission")?;
            let manifest = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
            let world = if tp2_layers.is_empty() { 1 } else { 2 };
            let workspace = usize::try_from(cuteafd_loader::plan::exlr8::native::flash_routed_workspace(
                &native.metadata, &self.cfg, &manifest, world, args.expert_rows() as u64)?)?;
            let peaks = if world == 1 { vec![native_local_budget()?] } else { tp2_budgets()?.to_vec() };
            let budgets = peaks.into_iter().map(|peak| peak.checked_sub(workspace)
                .context("native expert workspace exceeds admission")).collect::<Result<Vec<_>>>()?;
            Ok(Some(CoordinatorExperts::Exlr8 {
                manifest: args.exlr8_manifest.clone().context("native Flash manifest")?,
                layers: selected.clone(), budgets }))
        };
        let plan_native = |sparks: usize| -> Result<_> {
            let native = native_admission.as_ref().context("native Flash admission")?;
            let (owners, mut records) = crate::shared::experts::rtx::dense::recording(|| {
                let loaded = native.load(&self.library, &programs, &loader, &self.cfg, &execution, embedding_device.id,
                    admitted_native_load.as_ref().context("native load envelope missing")?)?;
                let shared = native.load_shared(args, &self.cfg, &self.library, &programs,
                    admitted_native_load.as_ref().context("native load envelope missing")?)?;
                Ok((loaded, shared))
            })?;
            records.extend(media.clone());
            let dense = crate::shared::experts::rtx::dense::plan_records(&args.snapshot, records, &devices, sparks)?;
            Ok((owners, dense))
        };
        let mut native_push = None;
        let mut coordinator_push = None;
        let admission = if !working.spark_layers.is_empty() && args.peers.is_some() {
            let peers = args.peers.as_deref().context("worker placement handshake needs Spark peers")?
                .split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let geometry = if let Some(catalog) = &self.experts { catalog.routed_experts().geometry()? }
                else { cuteafd_core::ExpertGeometry { hidden: u32::try_from(self.cfg.hidden)?,
                    experts: u32::try_from(self.cfg.experts)?, topk: u32::try_from(self.cfg.topk)?,
                    intermediate: u32::try_from(self.cfg.moe_intermediate)?, layers: u32::try_from(self.cfg.layers)? } };
            let capacity = u32::try_from(args.expert_rows())?;
            let ingress = if self.checkpoint.spark_input_is_bf16() { cuteafd_core::SparkIngress::Bf16 }
                else { cuteafd_core::SparkIngress::Fp8K32 };
            let catalog = self.experts.as_ref().context("GLM Flash Spark admission requires routed expert headers")?;
            let executions = cuteafd_loader::plan::spark_executions::admit_family("glm5_flash",
                &cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(catalog)?.exports(),
                Some(u32::try_from(args.decode_rows)?), args.prefill_rows as u64, args.prefill_waves as u64)?;
            let model = cuteafd_transport::worker_selection::ModelAdmission::new(
                vec![cuteafd_core::SparkEndpoint { ring_depth: None, capacity, ingress, partial_reply_bytes: None }],
                crate::shared::placement::spark_rings(64 << 20)?, executions)?;
            let source_view = &self.expert_view;
            let admission = crate::shared::placement::SparkAdmission::plan_view(&source_view, geometry, self.experts.as_ref(),
                &working, peers.len(), model, args.exlr8_manifest.as_deref())?;
            let native_ranks = if native_admission.is_some() {
                push_policy.plan(|| {
                    ensure!(admission.coordinator_pushes(), "native Flash Spark admission did not select required push");
                    let planned = plan_native(peers.len())?;
                    let (push, streams, ranks) = CoordinatorPush::prepare_admitted(
                        &args.snapshot, self.experts.as_ref(), gpus, native_push_experts()?,
                        admitted_push_host.clone().context("Flash push host receipt")?)?;
                    native_push = Some((push, peers.len(), planned));
                    Ok((streams, ranks))
                })?
            } else { None };
            let ranks = if native_admission.is_none() {
                push_policy.plan(|| {
                    ensure!(admission.coordinator_pushes(), "GLM Flash Spark admission did not select required push");
                    let experts = expert_plan()?;
                    let planned = plan_dense(peers.len())?;
                    let prepared = CoordinatorPush::prepare_admitted(&args.snapshot, self.experts.as_ref(), gpus, experts,
                        admitted_push_host.clone().context("Flash push host receipt")?)?;
                    Ok((prepared, planned))
                })?
            } else { None };
            let ranks = ranks.map(|((push, streams, ranks), planned)| {
                coordinator_push = Some((push, peers.len(), devices.clone(), planned));
                (streams, ranks)
            });
            Some(admission.start_with(&peers, native_ranks.or(ranks))?)
        } else {
            if native_admission.is_some() {
                native_push = push_policy.plan(|| {
                    let planned = plan_native(0)?;
                    let push = CoordinatorPush::prepare_alone_view_admitted(
                        &self.view, self.experts.as_ref(), gpus, native_push_experts()?,
                        admitted_push_host.clone().context("Flash push host receipt")?)?;
                    Ok((push, 0, planned))
                })?;
            }
            if native_admission.is_none() {
                coordinator_push = push_policy.plan(|| {
                    let experts = expert_plan()?;
                    let planned = plan_dense(0)?;
                    let push = CoordinatorPush::prepare_alone_view_admitted(&self.view, self.experts.as_ref(), gpus, experts,
                        admitted_push_host.clone().context("Flash push host receipt")?)?;
                    Ok((push, 0, devices.clone(), planned))
                })?;
            }
            None
        };
        // The admitted TP2 expert halves land inside the Spark admission window;
        // execution (modules, workspaces, routes) binds once the engine exists.
        let tp2_weights = |pushed: Option<Vec<PushedExperts<'s>>>| -> Result<crate::shared::experts::rtx::Tp2Weights<'_>> {
            let catalog = self.experts.as_ref().context("TP2 expert catalog")?;
            let (devices, budgets) = (tp2_devices()?, tp2_budgets()?);
            use crate::shared::experts::rtx::{exl3::Exl3Tp2, fp8moe::Fp8MoeTp2, RtxSelectionPolicy, Tp2Weights};
            Ok(if native_admission.is_some() {
                let pushed = pushed.context("native TP2 expert destination missing")?;
                Tp2Weights::Exlr8(crate::shared::experts::rtx::exlr8::Exlr8RtxWeights::pushed_pair(
                    devices, &programs, tp2_layers.clone(), args.expert_rows(), budgets,
                    cuteafd_loader::plan::exlr8::PrefillActivations::A16,
                    pushed.into_iter().map(PushedExperts::exlr8).collect::<Result<Vec<_>>>()?.try_into()
                        .map_err(|_| anyhow::anyhow!("two committed EXLR8 coordinator ranks required"))?)?)
            } else if let Some(manifest) = catalog.exl3() {
                let package = tp2_package(manifest);
                let geometry = cuteafd_core::expert_geometry();
                Tp2Weights::Exl3(match pushed {
                    Some(pushed) => Exl3Tp2::pushed_pair_weights(devices, catalog, geometry, [&package; 2],
                        tp2_layers.clone(), args.expert_rows(), budgets, RtxSelectionPolicy::CanonicalDecode,
                        pushed.into_iter().map(PushedExperts::exl3).collect::<Result<Vec<_>>>()?.try_into()
                            .map_err(|_| anyhow::anyhow!("two committed EXL3 coordinator ranks required"))?)?,
                    None => Exl3Tp2::load_pair_weights(devices, catalog, geometry, [&package; 2],
                        tp2_layers.clone(), args.expert_rows(), budgets, RtxSelectionPolicy::CanonicalDecode)?,
                })
            } else {
                let tensors = catalog.fp8().context("FP8 TP2 tensors")?;
                let package = args.fp8_package.clone().unwrap_or_else(||
                    crate::shared::experts::fp8::package_directory(&args.native_lib, 2, tensors.format()));
                Tp2Weights::Fp8(match pushed {
                    Some(pushed) => Fp8MoeTp2::pushed_pair(devices, tensors, &package, tp2_layers.clone(),
                        args.expert_rows(), budgets, pushed.into_iter().map(PushedExperts::fp8)
                            .collect::<Result<Vec<_>>>()?.try_into()
                            .map_err(|_| anyhow::anyhow!("two committed FP8 coordinator ranks required"))?)?,
                    None => Fp8MoeTp2::load_pair(devices, tensors, &package, tp2_layers.clone(),
                        args.expert_rows(), budgets)?,
                })
            })
        };
        // Gather reads the single admitted table over peer access when necessary.
        // The drafter read overlaps the committed target group.
        let draft_read = args.draft.as_deref().map(dspark::DraftRead::start).transpose()?;
        let mut native_local = None;
        let mut native_shared = None;
        let mut pushed_whole = None;
        let (embedding, ((model, mut shares), tp2)) = if let Some(native) = &native_admission {
            crate::shared::placement::load_while_admitting(admission, || {
                let (mut pushed, (loaded, shared)) = match native_push.take() {
                    Some((push, sparks, (owners, dense))) => {
                        // Records borrow these native allocations; retain every owner until
                        // the single dense/shared/routed group has committed or drained.
                        let split = crate::shared::memory::load_timing::Split::start();
                        let (pushed, ()) = push.receive_with(devices.clone(), self.experts.as_ref(), sparks,
                            Some(Ok(dense)), || ());
                        let pushed = push_policy.committed(pushed)?;
                        split.log("glm5_flash native dense push");
                        (pushed, owners)
                    }
                    None => (None, (native.load(&self.library, &programs, &loader, &self.cfg, &execution, embedding_device.id,
                    admitted_native_load.as_ref().context("native load envelope missing")?)?,
                        native.load_shared(args, &self.cfg, &self.library, &programs,
                    admitted_native_load.as_ref().context("native load envelope missing")?)?)),
                };
                native_shared = Some(shared);
                if !local_layers.is_empty() {
                    let mut ranks = pushed.take().context("native RTX expert destination missing")?;
                    ensure!(ranks.len() == 1, "native whole expert push must publish one rank");
                    let weights = ranks.pop().context("native RTX expert rank")?.exlr8()?;
                    native_local = Some(crate::shared::experts::rtx::exlr8::Exlr8RtxWeights::pushed_rank(
                        devices[0], &programs, local_layers.clone(), args.expert_rows(), native_local_budget()?,
                        cuteafd_loader::plan::exlr8::PrefillActivations::A16, 1, 0, weights)?);
                }
                let tp2 = if tp2_layers.is_empty() { None } else { Some(tp2_weights(pushed)?) };
                Ok((loaded.0, (loaded.1, tp2)))
            })?
        } else {
            crate::shared::placement::load_while_admitting(admission, || {
            let (pushed, loaded) = match coordinator_push.take() {
                Some((push, sparks, devices, (embedding, model, dense))) => {
                    let split = crate::shared::memory::load_timing::Split::start();
                    let (pushed, beside) = push.receive_with(devices, self.experts.as_ref(), sparks, Some(Ok(dense)), || Ok::<(), anyhow::Error>(()));
                    let pushed = push_policy.committed(pushed)?;
                    beside?;
                    split.log("glm5_flash dense push");
                    (pushed, (embedding, model))
                }
                None => (None, embedding_device.run(|| crate::shared::token_io::TokenEmbedding::load_with(&self.library,
                    embed.as_ref().context("legacy Flash embedding source")?.clone(),
                    args.token_io.embed_placement, || {
                        crate::shared::memory::device::Device { library: &self.library, id: args.device }
                            .run(|| loader.model(&self.cfg, &execution))
                    }, |name, bytes| loader.allocate(embedding_device.id, name, bytes)))?),
            };
            let tp2 = if tp2_layers.is_empty() { pushed_whole = pushed; None }
                else { Some(tp2_weights(pushed)?) };
            Ok((loaded.0, (loaded.1, tp2)))
            })?
        };
        embedding.validate_residency(embedding_device.id, admitted_plan.facts.allocations.iter()
            .filter(|owner| owner.location == cuteafd_loader::placement::planner::Location::coordinator(embedding_rank as u8)
                && owner.category == cuteafd_core::memory_layout::Category::Embedding
                && owner.group == "embedding" && owner.view.is_some())
            .map(|owner| owner.bytes).sum::<u64>()).context("GLM Flash embedding admission")?;
        tracing::info!(rank = embedding_rank, device = embedding_device.id, bytes = embedding.device_bytes(),
            "GLM Flash embedding matches admission (peer-readable single residency)");
        // `--fp8-head packed`: every operand has landed (through the mount, or the committed push
        // group), so the BF16 head packs into its exact 12-bit copy and is released.
        let model = if args.head_format == head::GlmfHeadFormat::Packed {
            let weights::GlmfWeights { layers: body, norm, head } = model;
            let head = crate::shared::memory::device::Device { library: &self.library, id: args.device }.run(|| {
                let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                let _memory_format = cuteafd_ffi::memory_ledger::format("bf16-packed12");
                head.pack(&self.library, self.cfg.vocab_size, self.cfg.hidden, loader.arenas.get(&args.device))
            })?;
            weights::GlmfWeights { layers: body, norm, head }
        } else { model };
        let physical = devices.iter().map(|device| device.id).collect::<Vec<_>>();
        let receipt_devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?;
        let mut weight_census = crate::shared::memory::receipts::Census::default();
        weight_census.owners(model.allocation_receipts(&shares, receipt_devices)?);
        weight_census.owners(embedding.allocation_receipts(receipt_devices, "glm5_flash.embedding")?);
        weight_census.normalize(cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT)?;
        tracing::info!(owners = weight_census.receipts.iter().filter(|r| r.allocation.view.is_none()).count(),
            aliases = weight_census.receipts.iter().filter(|r| r.allocation.view.is_some()).count(),
            "GLM Flash retained weight arena census");
        probe("target-weights")?;
        let resident: usize = model.layers.iter().map(weights::GlmfLayer::bytes).sum();
        let peer_resident: usize = shares.iter().flatten().map(weights::GlmfLayer::bytes).sum();
        let single = model.check_single_residency(args.kda_fp8, args.head_format)?;
        // Header-only resident inventory uses the actual loader flags, not launcher defaults.
        if native_admission.is_none() && args.kda_nvfp4_gate.is_none() {
            let headers = cuteafd_loader::plan::Checkpoint::coordinator_view(&self.view, false, false)?;
            let expected = cuteafd_loader::families::glm5_flash::resident::resident_weights_placed(
                &headers, &self.cfg, layers, 1 + usize::from(peer_stream.is_some()),
                cuteafd_loader::families::glm5_flash::resident::GlmfRepresentation {
                    kda_fp8: args.kda_fp8 != fp8::KdaFp8::Off, head: args.head_format,
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
            head = model.head.name(), head_mib = mib(single.head_bf16 + single.head_fp8 + single.head_packed),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "GLM 5.3 Flash coordinator weights validated after Spark admission (one copy each; elapsed includes wait)");
        let moe = (0..layers).any(|l| !self.cfg.dense[l]);
        let reserved_units = if args.prefix_marks == prefix::PrefixMarks::Pool {
            cuteafd_loader::serving_capacity::GLMF_POOL_MARK_RESERVED_UNITS } else { 0 };
        let pool_tokens = usize::try_from(placement.pool_tokens)?;
        let max_context = crate::shared::context::pool_context("glm5_flash", cuteafd_loader::placement::planner::Context::resolve(args.max_context as u64, requested_context), pool_tokens)?;
        let step_experts = admitted_step_experts(&self.cfg, &placement, args.skip_experts, self.fp8().is_some())?;
        let early_workspaces = if args.replay_records == engine::ReplayRecords::Shared {
            let mut settings = step_settings(args, index_cache);
            settings.max_context = max_context;
            let plan = engine::StepPlan::new(&self.library, &programs, &self.cfg, &model.layers, &model.head, step_experts, settings);
            let lanes = args.prefill_waves;
            Some(engine::StepWorkspaces::allocate(&plan, args.prefill_rows, lanes)?)
        } else { None };
        let records = early_workspaces.as_ref().map(|w| w.prefill_scratch()
            .context("shared replay records need prefill lanes")).transpose()?;
        let pages = (pool_tokens + reserved_units as usize * engine::UNIT_ROWS).div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::GlmfEngine::new(&self.library, &programs, self.cfg.clone(), model, stream,
            max_context, args.prefill_rows, args.prefill_waves, args.prefill_span_rows, pages, args.slots, embedding, index_cache,
            args.kda_state, args.decode_rows, records, context_mode, args.prefix_marks == prefix::PrefixMarks::Pool,
            &execution, step_experts)?;
        engine.admitted_host = admitted_host;
        engine.max_pool_tokens_no_onboard = placement.max_pool_tokens_no_onboard;
        engine.attention_plan = attention_plan;
        engine.spark_bf16 = self.checkpoint.spark_input_is_bf16();

        tracing::info!(index_cache = ?engine.index_cache, pool_tokens, reserved_units, "GLM 5.3 Flash DSA index cache");
        engine.kda_fp32_partials = args.kda_fp32_partials;
        engine.kda_output_shard = args.kda_output_shard;
        engine.kda_prefill_expanded = args.kda_prefill_expanded;
        engine.mark_slots = mark_slots;
        engine.target_head = args.resolved_target_head();
        // The route applies to the BF16 head only; the FP8 and packed heads run their own GEMVs.
        tracing::info!(target_head = ?engine.target_head, requested = ?args.target_head,
            fp8_head = args.head_format.setting(), head = engine.weights.head.name(),
            "GLM 5.3 Flash target head");
        if let Some((device, peer_stream)) = peer_stream {
            engine.attach_peer(device, peer_stream, shares.pop().context("head-split shares")?, &execution)?;
            tracing::info!(device = args.device, split_device = device, attention = %attention.mode,
                "GLM 5.3 Flash head split over two GPUs");
        }
        if let Some(native) = &native_admission {
            let shared = native_shared.take().context("native shared owners missing after load")?;
            let manifest = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
            engine.bind_native(&manifest, &native.charges, shared)?;
        } else {
            engine.bind_core(self.core_admission(args, layers, 1 + usize::from(split_device.is_some()), attention)?)?;
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
        engine.set_graph_budget(None);
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
            let local_layers = whole_layers[0].clone();
            if !local_layers.is_empty() {
                let budget = usize::try_from(placement.expert_ranges.first().context("local expert arena admission")?.peak_bytes)?;
                let mut expert_args = args.clone();
                expert_args.local_experts = true;
                expert_args.peers = None;
                let experts = if let Some(native) = native_local.take() {
                    engine::Experts::LocalNative(std::cell::RefCell::new(Box::new(native.execution()?)))
                } else {
                    ensure!(native_admission.is_none(), "native whole expert owner missing");
                    let pushed = pushed_whole.take().map(|ranks| {
                        ensure!(ranks.len() == gpus, "GLM Flash committed whole rank count differs from admission");
                        let mut ranks = ranks.into_iter();
                        let owner = ranks.next().context("GLM Flash whole owner rank")?;
                        ensure!(ranks.all(|rank| rank.len() == 0), "GLM Flash dense-only ranks received unbound experts");
                        Ok(owner)
                    }).transpose()?;
                    self.experts_for_layers(&expert_args, &local_layers, Some(budget), pushed)?
                        .context("admitted local expert backend is missing")?
                };
                if let engine::Experts::LocalExl3(local) = &experts { local.ensure(local_layers[0], stream)?; }
                engine.install_local(local_layers, experts)?;
            }
            if spark || args.skip_experts {
                let mut expert_args = args.clone();
                expert_args.local_experts = false;
                if let Some(experts) = self.experts_for_layers(&expert_args, &spark_layers, None, None)? {
                    engine.set_experts(experts)?;
                }
            }
        }
        engine.bind_expert_homes(&placement, &spark_layers, args.skip_experts)?;
        finish_loading(&self.library, &mut engine, |engine| {
            engine.drafter = self.load_drafter(args, &programs, stream, &engine.embedding, prepared_draft.take(), draft_read,
                if args.tp2_dflash { Some(engine.draft_exchange()?) } else { None },
                crate::shared::speculation::dflash::TargetProjection::from_bf16(engine.weights.head.bf16().is_some()))?;
            engine.bind_draft_core_head()?;
            engine.attention_plan = attention_plan.validate_loaded(engine.drafter.as_ref().map(dspark::Drafter::drafts))?;
            cuteafd_bench::context::set_resolved("attention-plan-rows", &engine.attention_plan.rows().to_string());
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
            probe("workspaces")
        }, |engine| {
            engine.prime_tp2()?;
            if let Some(dspark::Drafter::RemoteDflash(draft)) = &engine.drafter {
                draft.prime(&engine.embedding, engine.draft_head())?;
            }
            probe("expert-prime")
        })?;
        engine.configure_packing(args.prefill_packing.unwrap_or_default())?;
        let result = with_startup_graphs(engine.has_tp2(), args.serving_graph_policy,
            (16, engine.drafter.is_some()), |sequences, speculation| {
                if let Some(capture) = engine.warm_decode_graphs(sequences.min(args.decode_rows), speculation)? {
                    use cuteafd_loader::placement::inventory::{GraphCensus, GraphCensusIdentity};
                    let measured = cuteafd_loader::placement::planner::serving::reconcile_startup_graphs(
                        &mut admitted_plan, "glmf.graphs", &admitted_graphs, &declared_graphs, &capture.ranks)?;
                    cuteafd_loader::placement::reserve::log_plan(&admitted_plan, "glm5_flash captured");
                    use sha2::{Digest, Sha256};
                    let hash = |path: &std::path::Path| -> Result<String> {
                        let mut file = std::fs::File::open(path)?;
                        let mut digest = Sha256::new();
                        std::io::copy(&mut file, &mut digest)?;
                        Ok(format!("{:x}", digest.finalize()))
                    };
                    let programs_sha256 = hash(&args.manifest)?;
                    let native_sha256 = hash(self.library.loaded_path())?;
                    let layout = serde_json::to_string(&serde_json::json!({
                        "family": "glm5_flash", "context": max_context, "pool": pool_tokens,
                        "sequences": sequences.min(args.decode_rows), "speculation": speculation,
                        "args": format!("{args:?}"), "decisions": admitted_plan.decisions,
                        "spark_ranks": working.spark_ranks,
                    }))?;
                    for (rank, (graph, gpu)) in measured.ranks.iter().zip(&inventory.gpus).enumerate() {
                        let identity = GraphCensusIdentity {
                            declared_graphs: declared_graphs[rank].clone(),
                            programs_sha256: programs_sha256.clone(), native_sha256: native_sha256.clone(),
                            toolchain: env!("CUTEAFD_BUILD_TOOLCHAIN").into(),
                            gpu_class: format!("{}/{}/{}", gpu.arch, gpu.sms, gpu.total_bytes),
                            driver: gpu.driver.clone(), layout: layout.clone(),
                        };
                        identity.validate().map_err(anyhow::Error::msg)?;
                        let census = GraphCensus { identity, executables: graph.executables, bytes: graph.bytes,
                            evidence: "GLM Flash synchronized startup capture after exact masked-segment prewarm; tracked device growth excluded".into(),
                            producing_commit: env!("CUTEAFD_BUILD_COMMIT").into() };
                        tracing::info!(target: "cuteafd::placement", rank, census = %serde_json::to_string(&census)?,
                            "startup graph residency census");
                    }
                } else {
                    ensure!(admitted_graphs.lifetime != cuteafd_loader::placement::Lifetime::Startup,
                        "startup graphs were admitted but the executor did not capture them");
                }
                probe("graph-capture")
            }, || {
                engine.check_admitted_graphs(&admitted_graphs)?;
                engine.check_tp2_routes()
            }, || body(&engine));
        drop(engine);
        // SAFETY: the engine that used the streams is gone.
        unsafe { self.library.cuda_stream_destroy(stream)? };
        if let Some((device, peer_stream)) = peer_stream {
            crate::shared::peer_split::on_device(&self.library, device,
                // SAFETY: as above.
                || unsafe { self.library.cuda_stream_destroy(peer_stream) })?;
        }
        result
    }

    /// The drafter `--draft` names, on `stream` (it reads only the mask token's embedding row).
    fn load_drafter<'s>(&'s self, args: &EngineArgs, programs: &'s cuteafd_ffi::programs::Programs<'s>, stream: *mut std::ffi::c_void,
        embedding: &crate::shared::token_io::TokenEmbedding<'_>, prepared: Option<crate::shared::draft::remote::Prepared<'s>>,
        read: Option<dspark::DraftRead>, exchange: Option<std::rc::Rc<crate::shared::peer_split::PeerExchange<'s>>>,
        target_projection: crate::shared::speculation::dflash::TargetProjection)
        -> Result<Option<dspark::Drafter<'s>>> {
        let Some(snapshot) = &args.draft else { return Ok(None) };
        let read = read.context("the drafter read was not started with --draft")?;
        let started = Instant::now();
        let representation = cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation
            ::from_fp8_option(args.draft_fp8);
        let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let capacity = cuteafd_loader::families::glm5::draft_representation::flash_draft_capacity(&config,
            args.draft_context_slots, args.draft_sequences)?;
        let (slots, batch) = (capacity.context_slots, capacity.max_batch_sequences);
        // Admitted standalone push group of the drafter's own GPU, after the target
        // group committed; priming (RemoteDflash::new) reads only committed weights.
        let admitted = self.draft_push(args)?.map(|draft| draft.host);
        let drafter = if let Some(exchange) = exchange {
            use crate::shared::speculation::dflash::{DflashConfig, DflashDrafter};
            ensure!(prepared.is_none(), "TP2 DFlash cannot use a remote single-home drafter");
            let cfg = DflashConfig::read(snapshot)?;
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "DFlash2 does not fit this target");
            let mask = embedding.host_rows(&[cfg.mask_token])?;
            let file = read.join_dflash()?;
            let devices = [args.device, exchange.device(1)].map(|id|
                crate::shared::memory::device::Device { library: &self.library, id });
            dspark::Drafter::Dflash2(crate::shared::experts::rtx::push::load_alone_admitted(snapshot,
                &devices, admitted, || {
                    let arenas = dspark::dflash_arenas(&self.library, snapshot, representation, &devices.map(|d| d.id))?;
                    DflashDrafter::load_tp2_with(&self.library, programs, target_projection, snapshot, file,
                        slots, batch, mask, true, representation, args.fp8_scales, args.draft_linear,
                        args.draft_kv_cache, exchange,
                        |device, name, bytes| dspark::draft_allocation(&self.library, &arenas, device, name, bytes))
                })?)
        } else if let Some(prepared) = prepared {
            use crate::shared::speculation::dflash::{DflashConfig, DflashDrafter};
            let cfg = DflashConfig::read(snapshot)?;
            ensure!(cfg.hidden == self.cfg.hidden && cfg.vocab == self.cfg.vocab_size
                && cfg.taps.iter().all(|&l| l < self.cfg.layers), "DFlash2 does not fit this target");
            let mask = embedding.host_rows(&[cfg.mask_token])?;
            let file = read.join_dflash()?;
            let device = prepared.stream.device;
            let draft = crate::shared::experts::rtx::push::load_alone_admitted(snapshot, &[device], admitted, ||
                device.own(|| {
                    let arenas = dspark::dflash_arenas(&self.library, snapshot, representation, &[device.id])?;
                    DflashDrafter::load_with(&self.library, programs, crate::shared::speculation::dflash::TargetProjection::Native, snapshot, file, prepared.stream.raw,
                        slots, batch, mask, true, representation, args.fp8_scales, args.draft_linear, args.draft_kv_cache,
                        |device, name, bytes| dspark::draft_allocation(&self.library, &arenas, device, name, bytes))
                }))?;
            dspark::Drafter::RemoteDflash(crate::shared::draft::remote_dflash::RemoteDflash::new(draft, prepared, target_projection)?)
        } else {
            let device = crate::shared::memory::device::Device { library: &self.library, id: args.device };
            crate::shared::experts::rtx::push::load_alone_admitted(snapshot, &[device], admitted, ||
                dspark::Drafter::load(&self.library, programs, target_projection, snapshot, stream, slots, batch,
                    embedding, self.cfg.hidden, self.cfg.vocab_size, self.cfg.layers, representation, args.fp8_scales,
                    args.draft_linear, args.draft_kv_cache, read))?
        };
        drafter.set_draft_head(args.draft_head);
        let mut physical = vec![args.device];
        if let Some(peer) = args.split_device { physical.push(peer); }
        if let dspark::Drafter::RemoteDflash(remote) = &drafter {
            if !physical.contains(&remote.draft.device.id) { physical.push(remote.draft.device.id); }
        }
        let mut census = crate::shared::memory::receipts::Census::default();
        census.owners(drafter.allocation_receipts(
            cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?)?);
        census.normalize(cuteafd_loader::placement::planner::PhaseOrder::TARGET_THEN_DRAFT)?;
        tracing::info!(owners = census.receipts.iter().filter(|r| r.allocation.view.is_none()).count(),
            aliases = census.receipts.iter().filter(|r| r.allocation.view.is_some()).count(),
            "GLM Flash retained draft arena census");
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, draft_head = ?args.draft_head,
            draft_linear = ?args.draft_linear, "{} drafter resident", drafter.name());
        Ok(Some(drafter))
    }

    /// The drafter's standalone push owners from its headers alone (None: no
    /// drafter, or a snapshot outside a push namespace, which mounts).
    fn draft_push(&self, args: &EngineArgs)
        -> Result<Option<cuteafd_loader::placement::families::glm5_flash::admission::DraftPush>> {
        let Some(snapshot) = &args.draft else { return Ok(None) };
        let config = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        cuteafd_loader::placement::families::glm5_flash::admission::DraftPush::for_snapshot(snapshot, &config,
            cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::from_fp8_option(args.draft_fp8))?
            .map(|push| push.for_tp(if args.tp2_dflash { 2 } else { 1 })).transpose()
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

    fn experts_for_layers<'s>(&'s self, args: &EngineArgs, selected: &[usize], budget: Option<usize>,
        pushed: Option<crate::shared::experts::rtx::push::PushedExperts<'s>>) -> Result<Option<engine::Experts<'s>>> {
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
            let experts = match pushed {
                Some(pushed) => crate::shared::experts::fp8::Fp8Experts::from_pushed(&self.library, tensors,
                    &directory, selected, 1, 0, args.expert_rows(), budget, pushed.fp8()?),
                None => crate::shared::experts::fp8::Fp8Experts::load(&self.library, tensors, &directory,
                    selected, 1, 0, args.expert_rows(), budget),
            }.context("local routed experts must fit with step and prefix-cache reservations; use --peers for Sparks, \
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
            let resident = match pushed {
                Some(pushed) => {
                    let (selections, _) = crate::families::deepseek_v4::local::LocalExperts::exl3_selections(
                        &args.native_lib, catalog, 0, selected, args.expert_rows(), budget)?
                        .context("GLM Flash whole EXL3 package")?;
                    ensure!(selections.len() == selected.len(), "committed EXL3 inventory exceeds admission");
                    let weights = pushed.exl3()?;
                    ensure!(weights.len() == selected.len() && weights.iter().zip(selected).all(|(weight, &layer)|
                        matches!(weight.layout.layer, cuteafd_loader::V41Exl3Layer::Backbone(id) if id == layer)
                            && weight.layout.world == 1 && weight.layout.rank == 0),
                        "committed GLM Flash whole EXL3 payload differs from solved inventory");
                    let local = crate::families::deepseek_v4::local::LocalExperts::from_exl3_weights(
                        &self.library, &args.native_lib, catalog, &selections, weights, args.expert_rows())?
                        .context("committed GLM Flash EXL3 backend is empty")?;
                    Some((selected.to_vec(), local))
                }
                None => None,
            };
            let loads = resident.as_ref().map_or(0, |(layers, _)| layers.len());
            return Ok(Some(engine::Experts::LocalExl3(engine::LocalExl3 {
                library: &self.library, native_lib: args.native_lib.clone(), catalog,
                resident: std::cell::RefCell::new(resident), window: selected.len(),
                layers: selected.to_vec(), max_rows: args.expert_rows(),
                budget, loads: std::cell::RefCell::new(loads),
            })));
        }
        let Some(peers) = args.peers.as_deref() else { return Ok(None) };
        if selected.is_empty() { return Ok(None); }
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let spark_world = cuteafd_transport::expert::SparkWorld::canonical(peers.len())?;
        // One link retains independent intake planes for every outstanding unit wave.
        let mut transport = crate::shared::spark_intake::SparkLink::new_waves(&self.library,
            spark_world, &peers, u32::try_from(args.expert_rows())?, cuteafd_transport::TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(),
                timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 }, self.cfg.hidden * 2,
            args.prefill_waves, u32::try_from(args.prefill_rows)?)?;
        transport.enable_terminal_ownership(self.library.clone())?;
        let shape = engine::spark_session_shape(&self.cfg, args.expert_rows(), self.checkpoint.spark_input_is_bf16())?;
        transport.bootstrap(&shape)?;
        let started = Instant::now();
        transport.warm_experts(&shape, selected[0], self.cfg.experts, cuteafd_transport::ExpertV2SourceKind::Prefill)?;
        tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "Spark expert transport warm");
        Ok(Some(engine::Experts::Spark { transport: std::cell::RefCell::new(transport) }))
    }
}

impl Opened {
    /// The checkpoint's `embed_tokens` (BF16 [vocab, hidden]).
    fn embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let name = format!("{}embed_tokens.weight", weight_layout::PREFIX);
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

fn golden(mut args: GoldenArgs) -> Result<()> {
    args.engine.prefill_packing = Some(crate::shared::prefill_packing::PackingMode::resolve(
        args.engine.prefill_packing, args.packed_check.is_some()));
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
        std::fs::write(dir.join("attention.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "mode": opened.view.mode(), "bundle_fingerprint": opened.view.fingerprint(),
            "config": opened.view.config_name(), "index": opened.view.index_name(),
            "fp8_recipe": (opened.view.mode() == cuteafd_loader::plan::checkpoint::AttentionMode::Fp8Compact)
                .then_some(cuteafd_loader::plan::checkpoint::view::FP8_RECIPE),
        }))?)?;
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
