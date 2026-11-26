//! Qwen 3.8 Flash Next (qwen4_exp) on the generic engine: weights, the PLE
//! n-gram table, the coordinator programs' layer chain, and the golden
//! comparison command.
mod media;
mod graph_spanning;
pub(crate) mod engine;
mod admission;
mod hybrid;
mod programs;
mod dual;
mod mtp_golden;
pub(crate) mod ple;
pub(crate) mod prefix;
pub(crate) mod serve;
pub(crate) mod speculate;
pub(crate) mod weights;

use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::formats::fp8_experts::Fp8ExpertTensors;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use cuteafd_loader::families::qwen4::Qwen4Config;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

/// What every Qwen 3.8 Flash Next command needs to stand up the engine.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct EngineArgs {
    #[arg(long, default_value = "bf16")]
    pub(crate) attention_precision: cuteafd_core::attention::AttentionPrecision,
    /// Checkpoint snapshot (coordinator weights, PLE table, config, tokenizer).
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Two RTX whole-layer attention owners, with optional Spark expert fallback.
    #[arg(long, env = "CUTEAFD_QWEN_SPLIT_DEVICE", conflicts_with_all = ["shared_only", "expert_window"])]
    pub split_device: Option<i32>,
    /// Pool-first expert onboarding on one or two RTX devices: auto, all
    /// (every routed-expert layer of the target model), or an exact N.
    #[arg(long = "rtx-expert-layers", value_parser = |text: &str| text.parse::<cuteafd_loader::placement::OnboardSetting>())]
    pub onboard: Option<cuteafd_loader::placement::OnboardSetting>,
    /// Attention ownership; dual Qwen supports whole layers, not head shards.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto", value_parser = ["auto", "heads", "layers", "context"])]
    pub attention_placement: String,
    /// Run only the first N layers.
    #[arg(long)]
    pub layers: Option<usize>,
    /// Served context: auto fits the admitted pool; a positive count pins it.
    #[arg(long = "max-context-tokens", default_value = "auto", value_parser = crate::shared::context::parse_tokens::<usize>)]
    pub max_context: usize,
    /// Tokens the K/V record pools hold across sequences (0: planner admission).
    #[arg(long, default_value_t = 32_768)]
    pub pool_tokens: usize,
    /// Full-attention KV records (BF16 index keys in either format).
    #[arg(long = "kv-cache", default_value_t = cuteafd_loader::families::qwen4::Qwen4KvCache::default())]
    pub kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache,
    /// Concrete serving prefix arena reservation; filled before engine loading.
    #[arg(skip)]
    pub planner_prefix_bytes: Option<u64>,
    /// Resolve the serving host quota once the final pool and context are known.
    #[arg(skip)]
    pub planner_host_prefix: Option<(crate::shared::prefix::PrefixArgs, usize)>,
    /// Actual local encoder checkpoint weights, keyed by physical CUDA ordinal.
    #[arg(skip)]
    pub planner_media_weights: Option<(i32, cuteafd_loader::placement::ResidentWeights)>,
    /// Resolved media embedding-cache quota (host), charged by the recipe.
    #[arg(skip)]
    pub planner_media_cache_bytes: u64,
    /// Sequences with GDN/PLE state (about 115 MiB each).
    #[arg(long, default_value_t = 8)]
    pub slots: usize,
    /// Per-unit rows; 0 selects the shared topology policy.
    #[arg(long, default_value_t = 0)]
    pub prefill_rows: usize,
    /// Units in flight; 0 selects the shared policy within executor capability.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u8).range(0..=4))]
    pub prefill_waves: u8,
    /// Explicit per-unit rows for diagnostic cuts, including N=1.
    #[arg(long)]
    pub prefill_units: Option<usize>,
    /// Arithmetic-route policy for already-queued short prompt packing.
    #[arg(long, value_enum, default_value_t = crate::shared::prefill_packing::PackingMode::default())]
    pub prefill_packing: crate::shared::prefill_packing::PackingMode,
    /// Hold the GDN and attention in/out projections (target and MTP layers)
    /// as E4M3 with FP32 128x128 block scales, quantized at load, INSTEAD of
    /// the checkpoint's BF16 (no BF16 copy stays resident): every step shape
    /// runs the `qwen4_*_w8_*` programs (decode: 16-row GEMV, W8A16 TMA above;
    /// prefill: W8A16, bitwise BF16 over the dequantized weights, or W8A8 with
    /// --fp8-prefill-w8a8). Default false: checkpoint BF16.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub fp8_decode: bool,
    /// With --fp8-decode: prefill programs quantize their activations too
    /// (E4M3 per row and 128-K block; the GDN in-projection stays W8A16).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, requires = "fp8_decode")]
    pub fp8_prefill_w8a8: bool,
    /// Scale rule of the FP8 copies made from BF16 weights at load: amax /
    /// 448, the smallest power of two >= it (pow2), or per block whichever of
    /// the two leaves the smaller error (best).
    #[arg(long, value_enum, default_value_t = crate::shared::fp8_linear::Fp8Scales::Amax)]
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    #[command(flatten)]
    pub l2: crate::shared::l2_prefetch::L2PrefetchArgs,
    /// Spark ranks in TP order (HOST:PORT,...) serving the routed experts.
    #[arg(long, conflicts_with_all = ["local_experts", "shared_only"])]
    pub peers: Option<String>,
    /// Run the routed experts on this GPU: EXL3 checkpoints through the
    /// coordinator `exl3-qwen4-k45/rtx-tp1` package, FP8 ones through the TP1
    /// `fp8-qwen4` package. `--experts-snapshot` names another checkpoint for
    /// the experts.
    #[arg(long, conflicts_with = "shared_only")]
    pub local_experts: bool,
    /// No routed experts: the MoE output is the shared expert alone (plumbing tests).
    #[arg(long)]
    pub shared_only: bool,
    #[arg(long)]
    pub experts_snapshot: Option<PathBuf>,
    /// TP1 FP8 package directory (default `<libdir>/fp8/fp8-qwen4/tp1`).
    #[arg(long)]
    pub fp8_package: Option<PathBuf>,
    /// Diagnostic paging: keep only N local expert layers resident. By default
    /// all routed experts stay on the GPU; checkpoints that do not fit need Sparks.
    #[arg(long, requires = "local_experts")]
    pub expert_window: Option<usize>,
    /// GPU memory (GiB) kept free of local experts for step workspaces
    /// (logits alone are 4 GiB at 4096 rows).
    #[arg(long, default_value_t = 12)]
    pub expert_reserve_gib: usize,
    /// Native MTP drafts per step (0: no MTP). Loads the MTP layer (`mtp.*`,
    /// its experts local) and verifies up to this many drafts per sequence.
    #[arg(long, default_value_t = 0)]
    pub mtp: usize,
    /// Maximum decode/verify batch rows (`auto`, `64`, or `128`).
    #[arg(long, default_value = "auto")]
    pub decode_rows: cuteafd_loader::plan::decode_rows::DecodeRowsSetting,
    /// Numerical attention recipe rows; automatic uses the actually loaded MTP width.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLAN_ROWS")]
    pub attention_plan_rows: Option<usize>,
    /// Hold lm_head as E4M3 with FP32 per-row x 128-K scales (quantized on the
    /// host at load) INSTEAD of BF16: the one head the target (prefill, verify,
    /// golden rows) and the MTP drafts share, every logits row through
    /// `qwen4_head_fp8` in 16-row spans. Default false: both share the BF16 head.
    #[arg(long, alias = "fp8-head", default_value_t = false, action = clap::ArgAction::Set)]
    pub mtp_fp8_head: bool,
    #[command(flatten)]
    pub token_io: crate::shared::token_io::TokenIoArgs,
}

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Directory with tokens.bin, optional layerNN.bin and logits.bin from
    /// python/reference/families/qwen4/golden.py, or a published shared fidelity set.
    #[arg(long)]
    pub golden: PathBuf,
    /// Compare exact-row graphed decode/verify logits and persistent state byte
    /// for byte against ungraphed eager execution across short and long contexts.
    #[arg(long)]
    pub replay_check: bool,
    /// Compare real packed and singleton target state/logits on 2..=8 members.
    #[arg(long)]
    pub packed_check: Option<usize>,
    /// One window from the published shared fidelity quick subset (otherwise all eight).
    #[arg(long, requires_all = ["nll", "save_logits"])]
    pub window: Option<String>,
    /// Prefill only the first N tokens, then feed the rest through decode
    /// steps of --step-rows rows (teacher-forced), comparing their rows.
    #[arg(long)]
    pub prefill: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub step_rows: usize,
    /// Feed each layer the golden output of the previous one (prefill only),
    /// so every layer's cosine measures that layer alone.
    #[arg(long)]
    pub teacher_force: bool,
    /// Score every prefill row's logits against the golden (mean NLL, top-1).
    #[arg(long)]
    pub nll: bool,
    /// Save engine logits even without a reference logits.bin: with --nll every prefill row,
    /// otherwise only decode/verify rows after --prefill N. Shared fidelity sets instead write
    /// scored Probe rows, per-window metrics and fidelity.json to a new directory.
    #[arg(long)]
    pub save_logits: Option<PathBuf>,
    /// After the comparison, time this many greedy single-row decode steps.
    #[arg(long, default_value_t = 0)]
    pub bench_decode: usize,
    /// Time this many more prefills of the golden prompt (up to --prefill-rows
    /// tokens) on fresh sequences, without layer downloads.
    #[arg(long, default_value_t = 0)]
    pub bench_prefill: usize,
    /// Prompt length of --bench-prefill (the golden tokens repeated), in
    /// chunks of the engine's prefill rows; default the golden prompt up to
    /// one chunk.
    #[arg(long)]
    pub bench_prefill_tokens: Option<usize>,
    /// With --step-rows k: verify each step speculatively and commit only
    /// this many of its rows (the next step starts after them), checking
    /// GDN/PLE verify-by-replay against the golden logits.
    #[arg(long)]
    pub spec_keep: Option<usize>,
    /// Compare the MTP layer (needs --mtp) with the torch reference in this
    /// directory (python/reference/families/qwen4/mtp.py), teacher forced on the
    /// golden target streams, then stop.
    #[arg(long)]
    pub mtp_oracle: Option<PathBuf>,
    /// Token I/O gate after the golden prompt, then stop: the resident
    /// embedding table against the shard, device against host greedy
    /// selection over this many decode steps, and device against host sampling.
    #[arg(long)]
    pub token_check: Option<usize>,
    /// Greedy-decode this many tokens after the golden prompt (--prefill N
    /// truncates it) plainly and with MTP speculation at depth --mtp; the
    /// outputs must match. Reports acceptance and step costs, then stops.
    #[arg(long)]
    pub spec_decode: Option<usize>,
    /// Prefix-cache restore check at each token P (comma separated): prefill the first P tokens
    /// (of --prefill, default P + --resume-span), capture their snapshot (shared units, the
    /// copied tail unit, the GDN/PLE state mark), restore it into a second sequence, continue
    /// both (prefill in --prefill-chunk rows, then --resume-decode greedy steps) and compare
    /// every layer's rows, the logits, the paged rows, the state slot and the mark round trip
    /// byte for byte (a restore must be exact). Every P runs with every --prefill-chunk.
    #[arg(long, value_delimiter = ',')]
    pub resume_at: Vec<usize>,
    /// Prefill chunk rows of --resume-at, comma separated (default the engine's prefill rows).
    #[arg(long, value_delimiter = ',')]
    pub prefill_chunk: Vec<usize>,
    /// Tokens prefilled past P by --resume-at without --prefill.
    #[arg(long, default_value_t = 1000)]
    pub resume_span: usize,
    #[arg(long, default_value_t = 4)]
    pub resume_decode: usize,
    /// With --resume-at: prefill the second sequence cold on its own units instead of
    /// restoring it (the floor: what page placement alone changes).
    #[arg(long, hide = true)]
    pub resume_cold: bool,
    /// --resume-at attempts on fresh sequences per case (all must be byte-identical).
    #[arg(long, default_value_t = 1)]
    pub resume_repeat: usize,
    /// Multi-sequence 128-row verify + GDN/PLE commit + graphed replay parity oracle (32 x 4 = 128 rows).
    #[arg(long, requires_all = ["nll", "save_logits"])]
    pub wide_batch: bool,
}

impl EngineArgs {
    fn resolved_prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        let ranks = self.peers.as_deref().map_or(0, |peers| peers.split(',').filter(|p| !p.is_empty()).count());
        let local = self.local_experts || (ranks == 0 && !self.shared_only);
        let shape = prefill_shape(if local { 0 } else { ranks }, cuteafd_loader::families::qwen4::prefill_limits(local),
            PrefillOverrides {
                unit_rows: self.prefill_units.or((self.prefill_rows > 0).then_some(self.prefill_rows)).map(|r| r as u64),
                waves: (self.prefill_waves > 0).then_some(u64::from(self.prefill_waves)),
                span_rows: self.prefill_units.and_then(|_| (self.prefill_rows > 0).then_some(self.prefill_rows as u64)),
                reserve_rows: None,
            })?;
        ensure!(shape.waves == 1 || self.peers.is_some() || self.shared_only,
            "Qwen prefill units need Spark experts or shared-only diagnostics");
        cuteafd_bench::context::ResolvedSettings::prefill(shape)
            .value("prefill-rows", shape.span_rows).value("prefill-units", shape.unit_rows).publish();
        Ok(shape)
    }

    fn prefill_shape(&self) -> Result<Option<cuteafd_loader::serving_capacity::PrefillShape>> {
        let shape = self.resolved_prefill()?;
        Ok((self.prefill_units.is_some() || shape.waves > 1).then_some(shape.units()))
    }

    /// Per-wave tables, PLE gathers and Spark workers never hold the whole span.
    fn prefill_wave_rows(&self) -> usize {
        let decode = self.resolved_decode_rows(self.slots.max(1) as u64).unwrap_or(engine::DECODE_ROWS);
        self.prefill_units.unwrap_or(if self.prefill_rows > 0 { self.prefill_rows } else { 4096 })
            .max(decode)
    }

    pub(super) fn resolved_decode_rows(&self, concurrency: u64) -> Result<usize> {
        let gpus = 1 + usize::from(self.split_device.is_some());
        let rows = cuteafd_loader::plan::resolve_decode_rows(
            "qwen4",
            cuteafd_loader::plan::decode_rows::DecodeRowsRequest {
                requested: self.decode_rows.as_option(),
                concurrency: concurrency.max(1),
                decode_lanes: 1,
                rtx_gpus: gpus,
                draft_length: (self.mtp > 0).then_some(self.mtp as u32),
            },
        )?;
        Ok(cuteafd_loader::families::qwen4::decode_capacity(usize::try_from(rows)?))
    }

}

/// The checkpoint and native library, opened on the calling thread.
pub(crate) struct Opened {
    pub checkpoint: Checkpoint,
    pub cfg: Qwen4Config,
    pub library: std::sync::Arc<NativeLibrary>,
    /// Complete routed headers for local ownership or exact Spark admission;
    /// bundled MTP experts remain coordinator-local with Spark peers.
    pub experts: Option<cuteafd_loader::OfficialV41Catalog>,
}

impl Opened {
    fn fp8(&self) -> Option<&Fp8ExpertTensors> {
        self.experts.as_ref().and_then(|c| c.fp8())
    }
}

fn required_fp8_programs(args: &EngineArgs, decode_rows: usize) -> Vec<String> {
    let mut required = Vec::new();
    let decode_cap = if decode_rows <= engine::DECODE_ROWS { "m64" } else { "m128" };
    if args.fp8_decode {
        let suffix = if args.kv_format == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 { "_kv_fp8" } else { "" };
        for cap in [decode_cap, "m4096"] {
            required.extend([format!("qwen4_gdn_w8_{cap}"),
                format!("qwen4_attn_producer_w8{suffix}_{cap}"), format!("qwen4_attn_o_w8_{cap}")]);
        }
    }
    if args.mtp_fp8_head { required.push("qwen4_head_fp8".into()); }
    required
}

fn attention_plan(args: &EngineArgs, loaded_mtp: bool)
    -> std::result::Result<cuteafd_core::coordinator_programs::AttentionReductionPlan,
        cuteafd_core::coordinator_programs::AttentionPlanError>
{
    let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
        engine::DECODE_ROWS, loaded_mtp.then_some(args.mtp), args.attention_plan_rows)?;
    cuteafd_bench::context::set_resolved("attention-plan-rows", &plan.rows().to_string());
    Ok(plan)
}

#[cfg(test)]
mod weight_representation_tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Command {
        #[command(flatten)]
        engine: EngineArgs,
    }

    fn args(extra: &[&str]) -> std::result::Result<EngineArgs, clap::Error> {
        Command::try_parse_from(["qwen", "--snapshot", "/missing/checkpoint", "--native-lib", "/missing/native.so"]
            .into_iter().chain(extra.iter().copied())).map(|c| c.engine)
    }

    #[test]
    fn remote_catalog_keeps_complete_backbone_without_mtp() -> Result<()> {
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_exl3, write_snapshot};
        let root = tempfile::tempdir()?;
        let (tensors, _) = qwen4_exl3(2, 4);
        write_snapshot(root.path(), &qwen4_config(2), &tensors, None);
        let mut args = args(&["--peers", "127.0.0.1:9870"])?;
        args.snapshot = root.path().to_owned();
        let catalog = expert_catalog(&args)?.context("remote routed catalog")?;
        assert_eq!(args.mtp, 0);
        assert_eq!(catalog.routed_experts().layers, 2);
        let manifest = catalog.exl3().context("EXL3 headers")?;
        for layer in 0..2 {
            assert!(manifest.residency(cuteafd_loader::V41Exl3Layer::Backbone(layer), 3, 0)?
                .device_arena_layout()?.1 > 0);
        }
        Ok(())
    }

    #[test]
    fn prefill_units_validate_owner_slots_and_preserve_explicit_geometry() -> Result<()> {
        assert_eq!(args(&[])?.prefill_shape()?, None);
        for owners in [vec!["--local-experts"], vec!["--local-experts", "--split-device", "1"]] {
            let mut explicit = owners;
            explicit.extend(["--prefill-rows", "4096", "--prefill-waves", "1"]);
            let parsed = args(&explicit)?;
            assert_eq!(parsed.prefill_shape()?, None);
            assert_eq!(parsed.prefill_wave_rows(), 4096);
        }
        for (rows, waves) in [(512, 1), (1024, 2), (2048, 2), (2752, 2), (4096, 2), (2048, 3), (2048, 4), (2752, 3)] {
            let rows_arg = rows.to_string();
            let waves_arg = waves.to_string();
            let parsed = args(&["--shared-only", "--prefill-units", &rows_arg, "--prefill-waves", &waves_arg])?;
            let shape = crate::commands::bench::assert_resolved_family_seam("qwen4",
                &["max-context-tokens", "pool-tokens", "slots", "mtp", "rtx-expert-layers", "attention-plan-rows", "decode-rows"], || {
                    crate::commands::bench::fixture_prefix().resolve_entries(16);
                    let cfg = Qwen4Config::from_hf(&cuteafd_loader::plan::testing::qwen4_config(48))?;
                    let inputs = admission::inputs(&parsed, &cfg, 48, false, None, Some((160, true)), 0)?;
                    Ok(inputs.prefill_shape)
                })?.unwrap();
            assert_eq!((shape.unit_rows, shape.waves), (rows, waves));
            assert_eq!(parsed.prefill_rows, 0);
        }
        for extra in [
            vec!["--prefill-waves", "2"],
            vec!["--shared-only", "--prefill-units", "511"],
            vec!["--shared-only", "--prefill-units", "8192"],
            vec!["--shared-only", "--prefill-units", "4096", "--prefill-waves", "3"],
            vec!["--shared-only", "--prefill-units", "2752", "--prefill-waves", "4"],
        ] {
            assert!(args(&extra)?.prefill_shape().is_err(), "{extra:?}");
        }
        let diagnostic = args(&["--shared-only", "--prefill-rows", "16384", "--prefill-units", "2752",
            "--prefill-waves", "2"])?;
        assert!(diagnostic.prefill_shape().is_err());
        let diagnostic = args(&["--shared-only", "--prefill-rows", "5504", "--prefill-units", "2752",
            "--prefill-waves", "2"])?;
        let shape = diagnostic.prefill_shape()?.unwrap();
        assert_eq!((shape.unit_rows, shape.waves, diagnostic.prefill_rows), (2752, 2, 5504));
        let cfg = Qwen4Config::from_hf(&cuteafd_loader::plan::testing::qwen4_config(48))?;
        let admitted = admission::inputs(&diagnostic, &cfg, 48, false, None, Some((160, true)), 0)?;
        assert_eq!((admitted.prefill_shape, admitted.prefill_rows), (Some(shape), 5504));
        let shape = args(&["--peers", "localhost:9870", "--prefill-rows", "2048", "--prefill-waves", "2"])?.prefill_shape()?.unwrap();
        assert_eq!((shape.unit_rows, shape.slot_rows()), (2048, 4096));
        let mut parsed = args(&["--peers", "localhost:9870", "--prefill-waves", "2"])?;
        let shape = parsed.prefill_shape()?.unwrap();
        assert_eq!((shape.unit_rows, shape.slot_rows()), (4096, 8192));
        parsed.prefill_rows = shape.slot_rows() as usize;
        parsed.prefill_units = Some(shape.unit_rows as usize);
        assert_eq!(parsed.prefill_shape()?, Some(shape));
        assert_eq!(parsed.prefill_wave_rows(), 4096);
        let worker = admission::spark_model(cuteafd_loader::plan::spark_executions::SparkBackend::Fp8,
            parsed.prefill_wave_rows(), shape.waves, engine::DECODE_ROWS, false)?;
        assert_eq!((worker.capacity(), worker.compiled_capacity()), (4096, 4096));
        let cfg = Qwen4Config::from_hf(&cuteafd_loader::plan::testing::qwen4_config(48))?;
        let admission = admission::inputs(&parsed, &cfg, 48, false, None, Some((160, true)), 0)?;
        assert_eq!(admission.prefill_rows, 8192);
        assert_eq!(admission.prefill_shape, Some(shape));
        assert!(required_fp8_programs(&EngineArgs { fp8_decode: true, ..parsed }, engine::DECODE_ROWS)
            .iter().all(|name| !name.contains("m8192")));
        Ok(())
    }

    #[test]
    fn six_spark_prefill_charges_shared_token_table_depth() -> Result<()> {
        let mut parsed = args(&["--peers", "a:9870,b:9870,c:9870,d:9870,e:9870,f:9870"])?;
        let shape = parsed.resolved_prefill()?;
        assert_eq!((shape.unit_rows, shape.waves, shape.span_rows), (2752, 3, 8256));
        parsed.prefill_rows = shape.span_rows as usize;
        parsed.prefill_units = Some(shape.unit_rows as usize);
        let cfg = Qwen4Config::from_hf(&cuteafd_loader::plan::testing::qwen4_config(48))?;
        let inputs = admission::inputs(&parsed, &cfg, 48, false, None, Some((160, true)), 0)?;
        assert_eq!(inputs.prefill_shape, Some(shape.units()));
        assert_eq!(inputs.prefill_rows, 8256);
        let staging = cuteafd_loader::formats::mapped_table::MappedStaging::new(
            2752 * cfg.ple_rows(), 3, 64 * cfg.ple_rows(), 160)?;
        assert_eq!(staging.slot_bytes().count(), 8);
        assert_eq!(staging.total_bytes, staging.slot_bytes().sum::<usize>() + staging.workspace_bytes);
        Ok(())
    }

    #[test]
    fn dual_selector_requires_resident_local_experts() {
        assert!(args(&["--split-device", "1"]).is_ok());
        assert!(args(&["--split-device", "1", "--peers", "localhost:1"]).is_ok());
        assert!(args(&["--local-experts", "--split-device", "1"]).is_ok());
        assert!(args(&["--local-experts", "--split-device", "1", "--expert-window", "1"]).is_err());
        assert!(args(&["--local-experts", "--split-device", "1", "--peers", "localhost:1"]).is_err());
        assert!(args(&[]).unwrap().split_device.is_none());
    }

    #[test]
    fn dual_selector_uses_shared_onboarding_and_environment_key() {
        use clap::CommandFactory;
        let command = Command::command();
        let selector = command.get_arguments().find(|arg| arg.get_id() == "split_device").unwrap();
        assert_eq!(selector.get_env(), Some(std::ffi::OsStr::new("CUTEAFD_QWEN_SPLIT_DEVICE")));
        assert!(args(&["--rtx-expert-layers", "all"]).is_ok());
        for policy in ["auto", "all", "0", "5"] {
            assert!(args(&["--local-experts", "--split-device", "1", "--rtx-expert-layers", policy]).is_ok());
        }
        for retired in ["max", "100%"] {
            assert!(args(&["--local-experts", "--split-device", "1", "--rtx-expert-layers", retired]).is_err());
        }
    }

    #[test]
    fn default_bf16_head_and_projections_require_explicit_fp8() {
        for extra in [&[][..], &["--mtp", "1"][..]] {
            let parsed = args(extra).unwrap();
            assert!(!parsed.fp8_decode && !parsed.mtp_fp8_head && !parsed.fp8_prefill_w8a8);
        }
        assert!(!args(&["--mtp-fp8-head", "false"]).unwrap().mtp_fp8_head);
        let parsed = args(&["--fp8-decode", "true", "--mtp-fp8-head", "true", "--fp8-prefill-w8a8", "true"]).unwrap();
        assert!(parsed.fp8_decode && parsed.mtp_fp8_head && parsed.fp8_prefill_w8a8);
        assert!(args(&["--fp8-head", "true"]).unwrap().mtp_fp8_head);
        // The W8A8 switch applies to FP8-only projections only.
        assert!(args(&["--fp8-prefill-w8a8", "true"]).is_err());
    }

    #[test]
    fn attention_plan_uses_loaded_mtp_and_validates_override() {
        for drafts in [1, 3, 63, 100] {
            let draft_arg = drafts.to_string();
            let parsed = args(&["--mtp", &draft_arg]).unwrap();
            let plain = attention_plan(&parsed, false).unwrap();
            assert_eq!(plain.rows(), 1);
            assert_eq!(plain.active_drafts(), None);
            let loaded = attention_plan(&parsed, true).unwrap();
            assert_eq!(loaded.rows(), (drafts + 1).min(engine::DECODE_ROWS));
            assert_eq!(loaded.active_drafts(), Some(drafts));
        }
        for rows in [1, 8, 64] {
            let row_arg = rows.to_string();
            let parsed = args(&["--mtp", "3", "--attention-plan-rows", &row_arg]).unwrap();
            for loaded in [false, true] {
                assert_eq!(attention_plan(&parsed, loaded).unwrap().rows(), rows);
            }
        }
        for row_arg in ["0", "65"] {
            let parsed = args(&["--attention-plan-rows", row_arg]).unwrap();
            assert!(attention_plan(&parsed, false).is_err());
        }
        assert!(args(&["--attention-plan-rows", "auto"]).is_err());
    }

    #[test]
    fn kv_cache_defaults_to_fp8_and_keeps_explicit_bf16() {
        use cuteafd_loader::families::qwen4::Qwen4KvCache;
        assert_eq!(args(&[]).unwrap().kv_format, Qwen4KvCache::Fp8);
        assert_eq!(args(&["--kv-cache", "bf16"]).unwrap().kv_format, Qwen4KvCache::Bf16);
        assert_eq!(args(&["--kv-cache", "fp8"]).unwrap().kv_format, Qwen4KvCache::Fp8);
        assert!(args(&["--kv-cache", "int8"]).is_err());
        assert!(args(&["--kv-format", "bf16"]).is_err());
    }

    #[test]
    fn required_fp8_producer_uses_selected_kv_format() {
        for format in ["bf16", "fp8"] {
            let parsed = args(&["--fp8-decode", "true", "--kv-cache", format]).unwrap();
            let suffix = if format == "fp8" { "_kv_fp8" } else { "" };
            for (decode_rows, decode_cap) in [(64usize, "m64"), (128, "m128")] {
                let required = required_fp8_programs(&parsed, decode_rows);
                for cap in [decode_cap.to_string(), "m4096".to_string()] {
                    assert!(required.contains(&format!("qwen4_attn_producer_w8{suffix}_{cap}")));
                    let other = if format == "fp8" { "" } else { "_kv_fp8" };
                    assert!(!required.contains(&format!("qwen4_attn_producer_w8{other}_{cap}")));
                    assert!(required.contains(&format!("qwen4_gdn_w8_{cap}")));
                    assert!(required.contains(&format!("qwen4_attn_o_w8_{cap}")));
                }
            }
        }
    }

    #[test]
    fn fp8_requests_reach_the_checkpoint_instead_of_a_refusal() {
        for extra in [&["--fp8-decode", "true"][..], &["--mtp", "1", "--mtp-fp8-head", "true"][..]] {
            let error = open(&args(extra).unwrap()).err().unwrap().to_string();
            assert!(!error.contains("unsupported"), "{error}");
        }
    }
}

fn expert_catalog(args: &EngineArgs) -> Result<Option<cuteafd_loader::OfficialV41Catalog>> {
    if !args.local_experts && args.peers.is_none() { return Ok(None); }
    let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
    let mut catalog = cuteafd_loader::read_expert_catalog(source)?;
    catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::selected("qwen4",
        if args.mtp > 0 { "mtp" } else { "off" }))?;
    ensure!(catalog.fp8().is_some() || catalog.exl3().is_some(),
        "Qwen routed experts need FP8 or EXL3 headers; {} has neither", source.display());
    Ok(Some(catalog))
}

pub(crate) fn open(args: &EngineArgs) -> Result<Opened> {
    let checkpoint = Checkpoint::coordinator(&args.snapshot, args.local_experts, args.mtp > 0)?;
    let cfg = Qwen4Config::read(&args.snapshot)?;
    cuteafd_bench::context::set_family(cfg.family_id());
    cfg.check_programs()?;
    // The expert geometry is process-wide and must be fixed before the native
    // library loads (its expert helpers size rows from it).
    let geometry = cuteafd_core::ExpertGeometry::QWEN4;
    ensure!(geometry.hidden as usize == cfg.hidden && geometry.experts as usize == cfg.experts
        && geometry.topk as usize == cfg.topk && geometry.intermediate as usize == cfg.moe_intermediate,
        "checkpoint experts do not match the Qwen 3.8 Flash Next geometry");
    cuteafd_core::set_expert_geometry(geometry).map_err(|g| anyhow::anyhow!("expert geometry already {g:?}"))?;
    let experts = expert_catalog(args)?;
    // SAFETY: the library is the cuteafd native shim built for this engine.
    let library = std::sync::Arc::new(unsafe { NativeLibrary::load(&args.native_lib) }?);
    library.cuda_set_device(args.device)?;
    Ok(Opened { checkpoint, cfg, library, experts })
}

impl Opened {
    /// Builds the engine and hands it to `body`.
    pub fn with_engine<T>(&self, args: &EngineArgs, body: impl FnOnce(&engine::Qwen4Engine<'_>) -> Result<T>)
        -> Result<T> {
        self.with_engine_media(args, Default::default(), body)
    }
    pub(super) fn with_engine_media<T>(&self, args: &EngineArgs,
        media: crate::shared::experts::rtx::dense::Records,
        body: impl FnOnce(&engine::Qwen4Engine<'_>) -> Result<T>) -> Result<T> {
        let requested = cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?;
        let selected = if args.split_device.is_some() {
            ensure!(requested.is_none() || requested == Some(cuteafd_loader::placement::AttentionPlacement::Layers),
                "Qwen dual supports whole-layer attention only");
            Some(cuteafd_loader::placement::AttentionPlacement::Layers)
        } else { requested };
        cuteafd_loader::placement::families::QWEN4.check_attention(selected,
            1 + usize::from(args.split_device.is_some()), args.split_device.is_some())?;
        let requested_context = (args.max_context > 0).then_some(args.max_context as u64);
        let mut context_args = args.clone();
        let resolved = args.resolved_prefill()?;
        let prefill_shape = args.prefill_shape()?;
        context_args.prefill_rows = usize::try_from(resolved.span_rows)?;
        context_args.prefill_waves = u8::try_from(resolved.waves)?;
        if let Some(shape) = prefill_shape {
            context_args.prefill_units = Some(shape.unit_rows as usize);
        }
        context_args.max_context = crate::shared::context::checkpoint_context(
            &args.snapshot, &args.manifest, "qwen4", args.max_context)?;
        let args = &context_args;
        let context = cuteafd_loader::placement::planner::Context::resolve(args.max_context as u64, requested_context);
        if args.pool_tokens > 0 && (args.pool_tokens as u64) < context.required() {
            return Err(cuteafd_loader::placement::PlacementError::RequestedBelowContext {
                requested: args.pool_tokens as u64, floor: context.required(),
            }.into());
        }
        if let Some(peer) = args.split_device {
            return self.with_dual_engine(args, peer, cuteafd_loader::placement::planner::Context::resolve(args.max_context as u64, requested_context), media, body);
        }
        let decode_rows = args.resolved_decode_rows(args.slots.max(1) as u64)?;
        cuteafd_bench::context::set_resolved("decode-rows", &decode_rows.to_string());
        let programs = self.library.programs()?.with_manifest(&args.manifest)?;
        programs.capacities().require_context("qwen4", args.max_context)?;
        self::programs::validate_all(&self.cfg, &programs, args.max_context)?;
        let required = required_fp8_programs(args, decode_rows);
        for name in &required {
            programs.spec(name).with_context(|| format!("--fp8-decode / --mtp-fp8-head need the FP8-only program \
                {name} (export_b12x_dsv4_aot.py qwen4 with the fork's fp8_only Qwen programs); rebuild the native \
                library or run with checkpoint BF16 (--fp8-decode false --mtp-fp8-head false)"))?;
        }
        // The family's own programs (`qwen4_*`), sampled as the runtime inventory: the image also carries
        // every other family's modules, which Qwen never launches.
        let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: "qwen4", split_family: None };
        let device = self.library.cuda_get_device()?;
        let inventory = crate::shared::inventory::RuntimeInventory::measure(&self.library, &[device], crate::shared::inventory::InventoryPhase::PreWeight, |_| {
            let (loaded, skipped) = programs.load_matching(|name| selected.contains(name))?;
            tracing::info!(loaded, skipped, "Qwen programs loaded");
            Ok(())
        })?;
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        let mut manifest_json: Option<serde_json::Value> = std::fs::read(&args.manifest).ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok());
        let with_mtp = args.mtp > 0 && layers == self.cfg.layers;
        let attention_plan = attention_plan(args, with_mtp)?;
        let programs = programs.with_attention_plan(attention_plan, |name| selected.contains(name))?;
        if let Some(manifest) = &mut manifest_json {
            if let Some(entries) = manifest["programs"].as_array_mut() {
                for entry in entries {
                    if !entry["name"].as_str().is_some_and(|name| selected.contains(name)) { continue; }
                    if let Some(bytes) = cuteafd_core::coordinator_programs::attention_plan_scratch(
                        &entry["geometry"], attention_plan)? {
                        entry["scratch_bytes_at_capacity"]["scratch"] = bytes.into();
                    }
                }
            }
        }
        let (pool_tokens, admitted_host, admitted_experts) = admission::preallocate(&self.library, args, &self.checkpoint,
            &self.cfg, self.experts.as_ref(), layers, with_mtp, manifest_json.as_ref(),
            cuteafd_loader::placement::planner::Context::resolve(args.max_context as u64, requested_context),
            &inventory)?;
        let stream_owner = std::rc::Rc::new(crate::shared::memory::device::Stream::new(
            crate::shared::memory::device::Device { library: &self.library, id: device })?);
        let stream = stream_owner.raw;
        let started = Instant::now();
        let loader = weights::Qwen4Loader { library: &self.library, checkpoint: &self.checkpoint,
            fp8_decode: args.fp8_decode, fp8_scales: args.fp8_scales, stream,
            dense: std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::mount(&self.library)), arenas: None };
        let source = self.embed_source()?;
        cuteafd_bench::context::set_resolved("mtp", &if with_mtp { args.mtp.min(engine::DECODE_ROWS - 1) } else { 0 }.to_string());
        cuteafd_bench::context::set_rtx_expert_layers(admitted_experts.placement.onboard_layers);
        let admitted_target = admission::single_weights(args, &self.checkpoint, &self.cfg, layers, with_mtp)?;
        // The coordinator's dense records push in a group of this GPU alone when
        // the checkpoint is SparkNest's; records are planned from headers first,
        // and Required planning or delivery failures refuse startup.
        let push_policy = crate::shared::experts::rtx::push::PushPolicy::for_snapshot(&args.snapshot)?;
        let device = crate::shared::memory::device::Device { library: &self.library, id: self.library.cuda_get_device()? };
        let loader = loader.with_arenas(&self.cfg, &vec![0; layers], &[(device, stream)],
            with_mtp, args.mtp_fp8_head, args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu)?;
        let planned = push_policy.plan(|| {
            let load = std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::push(&self.library));
            let planning = weights::Qwen4Loader { dense: load.clone(), ..loader.clone() };
            let model = {
                let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                planning.model(&self.cfg, layers, with_mtp, args.mtp_fp8_head)?
            };
            let ((embedding, ()), mut records) = crate::shared::experts::rtx::dense::recording(||
                crate::shared::token_io::TokenEmbedding::load_with(&self.library, source.clone(), args.token_io.embed_placement,
                    || Ok(()), |name, bytes| loader.embedding_allocation(name, bytes)))?;
            records.extend(load.sink.as_ref().context("push sink")?.take_all());
            records.extend(media.clone());
            let (landed, operands, staged, steps, largest, finishes) = records.summary();
            tracing::info!(landed, operands, staged, steps, largest_step_bytes = largest, finishes,
                "Qwen coordinator dense records planned");
            let dense = crate::shared::experts::rtx::dense::plan_records(&args.snapshot, records.clone(), &[device], 0)?;
            let push = crate::shared::experts::rtx::push::CoordinatorPush::prepare_alone(
                &args.snapshot, None, 1, None)?;
            Ok((push, embedding, model, records, dense))
        }).context("Qwen coordinator dense push planning")?;
        let (embedding, model) = match planned {
            Some((push, embedding, model, records, dense)) => {
                let split = crate::shared::memory::load_timing::Split::start();
                let (pushed, ()) = push.receive_with(vec![device], None, 0, Some(Ok(dense)), || ());
                let pushed = push_policy.committed(pushed).context("Qwen coordinator dense push delivery")?;
                tracing::info!(pushed = pushed.is_some(), operands = records.operands.len(),
                    "Qwen coordinator dense operands landed");
                split.log("qwen4 dense push");
                (embedding, model)
            }
            None => crate::shared::token_io::TokenEmbedding::load_with(&self.library, source, args.token_io.embed_placement,
                || { let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                    loader.model(&self.cfg, layers, with_mtp, args.mtp_fp8_head) },
                |name, bytes| loader.embedding_allocation(name, bytes))?,
        };
        let resident = model.bytes();
        // One resident representation per selectable weight, as the plan accounts for it.
        let (projections, head) = model.check_single_residency(args.fp8_decode, args.mtp_fp8_head)?;
        let selection = cuteafd_loader::families::qwen4::resident::Qwen4Representation {
            fp8_projections: args.fp8_decode, fp8_head: args.mtp_fp8_head };
        let plan = cuteafd_loader::families::qwen4::resident::resident_bytes(&self.cfg, layers, model.mtp.is_some(),
            selection);
        ensure!(projections == plan.projections && head == plan.head,
            "selectable weights hold {projections} + {head} B, the plan {} + {} B", plan.projections, plan.head);
        let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
        tracing::info!(layers, gib = gib(resident), elapsed_ms = started.elapsed().as_millis() as u64,
            projections = if args.fp8_decode { "fp8-only" } else { "bf16" },
            head = if args.mtp_fp8_head { "fp8-only (shared)" } else { "bf16 (shared)" },
            projection_gib = gib(projections), head_gib = gib(head),
            checkpoint_bf16_gib = gib(plan.projections_bf16 + plan.head_bf16),
            "Qwen 3.8 Flash Next coordinator weights resident (one representation per weight)");
        let ple = match self.cfg.ple_layers.first() {
            Some(&layer) if layer < layers => Some(ple::PleTable::load(&self.library, &self.checkpoint, &self.cfg,
                layer, args.prefill_wave_rows(), prefill_shape.map_or(1, |shape| shape.waves as usize),
                decode_rows)?),
            _ => None,
        };
        weights::arena::probe(&self.library, &[device.id], &model, &embedding)?;
        use cuteafd_loader::placement::ResidentWeights;
        // The physical backings actually retained (the arenas, with the device
        // embedding as a view of the target), not the per-tensor payload.
        let mut loaded_weights = ResidentWeights(weights::arena::retained_backing_bytes(&[device.id], &model, &embedding)?);
        if let Some((gpu, weights)) = args.planner_media_weights {
            if gpu == args.device { loaded_weights = loaded_weights.checked_add(weights)?; }
            else { weights.check(0, crate::shared::memory::budget::resident_weight_limit())?; }
        }
        ensure!(loaded_weights == admitted_target,
            "Qwen checkpoint weight owners differ from the pre-upload admission: {} != {} bytes",
            loaded_weights.0, admitted_target.0);
        loaded_weights.check(0, crate::shared::memory::budget::resident_weight_limit())?;
        let max_context = crate::shared::context::pool_context("qwen4", cuteafd_loader::placement::planner::Context::resolve(args.max_context as u64, requested_context), pool_tokens)?;
        let pages = pool_tokens.div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::Qwen4Engine::new_with_decode_rows(&self.library, &programs, self.cfg.clone(), model, ple, std::rc::Rc::clone(&stream_owner),
            max_context, args.prefill_rows, pages, args.slots, decode_rows, args.kv_format, embedding)?;
        engine.prefill_shape = prefill_shape;
        engine.attention_plan = attention_plan;
        engine.spark_bf16 = self.checkpoint.spark_input_is_bf16();
        engine.w8a8_prefill = args.fp8_prefill_w8a8;
        if args.shared_only {
            engine.set_experts(engine::Experts::SharedOnly);
        } else if self.experts.is_some() {
            let experts = self.load_hybrid(args, &[device], &[stream], &admitted_experts)?;
            engine.install_hybrid(experts, &admitted_experts.placement, cuteafd_loader::placement::HopSpec {
                storage: cuteafd_loader::placement::HopStorage::ReceiveSlots,
                row_bytes: self.cfg.hc_width() as u64 * 2, rows: args.prefill_rows as u64,
                lanes: 1, entry_gpu: 0, head_gpu: 0, terminal: cuteafd_loader::placement::TerminalHome::AttentionOwner })?;
        }
        engine.configure_packing(args.prefill_packing)?;
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::OTHER_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
        }
        engine.admitted_host = Some(admitted_host);
        engine.admit_host_memory(args)?;
        engine.prepare_prefill_units()?;
        // Weights and experts are resident: the load's pinned upload high-water
        // is transient (serving's small table uploads regrow only what they use).
        crate::shared::memory_report::release_load_staging(&self.library)?;
        let result = body(&engine);
        drop(engine);
        std::rc::Rc::into_inner(stream_owner).context("qwen4 lead stream still referenced")?.destroy()?;
        result
    }

    fn local_mtp_experts<'s>(&'s self, args: &EngineArgs, layer: usize,
        stream: *mut std::ffi::c_void, budget: usize) -> Result<engine::MtpExperts<'s>> {
        let catalog = self.experts.as_ref().context("--mtp needs the local draft expert catalog")?;
        let started = Instant::now();
        let mtp = if catalog.exl3().is_some() {
            let experts = crate::families::deepseek_v4::local::LocalExperts::load(&self.library,
                &args.native_lib, catalog, 1, &[], args.prefill_wave_rows(), budget, stream)?
                .context("coordinator-local MTP needs exl3-qwen4-k45/rtx-tp1/m* packages")?;
            ensure!(experts.layers().is_empty() && experts.stages() == 1,
                "local draft executor must keep exactly the MTP expert layer, no backbone layers");
            engine::MtpExperts::Exl3(std::cell::RefCell::new(experts))
        } else {
            let tensors = catalog.fp8().context("coordinator-local MTP needs FP8, NVFP4 or EXL3 experts")?;
            let draft = tensors.for_layer(layer)?;
            let directory = crate::shared::experts::fp8::package_directory(&args.native_lib, 1, draft.format());
            let experts = crate::shared::experts::fp8::Fp8Experts::load(&self.library, &draft, &directory,
                &[layer], 1, 0, args.prefill_wave_rows(), budget)?;
            ensure!(!experts.wire_input(), "coordinator-local MTP FP8 package must take BF16 rows");
            engine::MtpExperts::Fp8(experts)
        };
        tracing::info!(layer, elapsed_ms = started.elapsed().as_millis() as u64,
            "Qwen full-width MTP experts resident on head owner");
        Ok(mtp)
    }
}

impl Opened {
    /// The checkpoint's `embed_tokens` (BF16 [vocab, hidden]).
    fn embed_source(&self) -> Result<crate::shared::token_io::EmbedSource> {
        let name = format!("{}embed_tokens.weight", weights::PREFIX);
        let at = self.checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(&name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no {name}"))?;
        let tensor = &self.checkpoint.tensors[at];
        crate::shared::token_io::EmbedSource::new(&self.checkpoint.snapshot, &tensor.shard, &tensor.meta, self.cfg.hidden)
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

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || golden(args)).await?
}

fn golden(mut args: GoldenArgs) -> Result<()> {
    if let Some(members) = args.packed_check {
        ensure!((2..=8).contains(&members), "--packed-check needs 2..=8 members");
        args.engine.slots = args.engine.slots.max(2 * members);
    }
    ensure!(args.step_rows > 0, "Qwen golden --step-rows must be nonzero");
    let windows = if args.golden.join("manifest.json").try_exists()? && args.packed_check.is_some() {
        Some(fidelity_windows(&args.golden, None,
            args.engine.prefill_packing == crate::shared::prefill_packing::PackingMode::Unrestricted)?)
    } else if args.golden.join("manifest.json").try_exists()? {
        ensure!(args.nll && args.save_logits.is_some(), "shared Qwen fidelity needs --nll --save-logits DIR");
        ensure!(args.prefill.is_none() && !args.teacher_force && args.spec_keep.is_none()
            && args.mtp_oracle.is_none() && args.token_check.is_none() && args.spec_decode.is_none()
            && args.resume_at.is_empty() && args.bench_prefill == 0 && args.bench_decode == 0,
            "shared Qwen fidelity cannot be combined with legacy golden diagnostics");
        let output = args.save_logits.as_deref().unwrap();
        ensure!(!output.try_exists()? && output.parent().is_some_and(std::path::Path::is_dir),
            "Qwen fidelity output needs an existing parent and a new directory");
        Some(fidelity_windows(&args.golden, args.window.as_deref(), false)?)
    } else {
        ensure!(args.window.is_none(), "--window needs a published shared fidelity set");
        None
    };
    let mut args = args;
    if args.wide_batch {
        args.engine.slots = args.engine.slots.max(32);
    }
    let opened = open(&args.engine)?;
    opened.with_engine(&args.engine, |engine| {
        if args.replay_check {
            let bytes = std::fs::read(args.golden.join("tokens.bin"))?;
            ensure!(!bytes.is_empty() && bytes.len() % 4 == 0, "invalid replay-check token payload");
            let tokens: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
            engine.prepare_serving_workspaces()?;
            let reserves = engine.configure_graph_cache()?;
            ensure!(reserves.iter().any(|&bytes| bytes > 0), "replay-check needs a nonzero graph reserve");
            engine.warm_decode_graphs(args.engine.slots, args.engine.mtp > 0)?;
            engine.check_decode_graphs(&tokens)?;
            return Ok(());
        }
        if let Some(members) = args.packed_check {
            let tokens = if let Some(windows) = &windows {
                windows.first().context("packed check needs a fidelity window")?.tokens.clone()
            } else {
                let bytes = std::fs::read(args.golden.join("tokens.bin"))?;
                ensure!(!bytes.is_empty() && bytes.len() % 4 == 0, "invalid packed-check token payload");
                bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>()
            };
            packed_check(engine, &tokens, members, args.engine.prefill_packing)?;
            if args.engine.prefill_packing == crate::shared::prefill_packing::PackingMode::Exact {
                engine.configure_packing(crate::shared::prefill_packing::PackingMode::Unrestricted)?;
                packed_decode_interleave(engine, &tokens)?;
            }
            if args.engine.prefill_packing == crate::shared::prefill_packing::PackingMode::Unrestricted {
                let windows = windows.as_ref().context("unrestricted packed check needs the sealed official reference")?;
                let first = args.window.as_deref().unwrap_or(&windows.first().context("empty packed reference")?.id);
                let selected = crate::shared::qwen::packed_check::reference_windows(windows, first, members, true)?;
                let precision = cuteafd_bench::fidelity_precision::resolve("qwen-packed-check", Some(&args.engine.snapshot),
                    &serde_json::json!([]), "Qwen/Qwen3.8-Flash-Next", None)?;
                let mut allocator = engine::Allocator::new(engine.pages, engine.slots, &engine.cfg);
                let placements = selected.iter().map(|w| allocator.admit(w.tokens.len())).collect::<Result<Vec<_>>>()?;
                let controls = selected.iter().map(|w| allocator.admit(w.tokens.len())).collect::<Result<Vec<_>>>()?;
                crate::shared::qwen::packed_check::reference::<_, speculate::MtpSeq>(engine.library, &selected, placements, controls,
                    engine.packed_limits().context("packed Qwen requires one attention owner")?, engine.weights.mtp.is_some(), &precision,
                    |s| {
                        let head = engine.prefill_device(s.placement, s.tokens, None, None, 1)?.context("reference check needs head")?;
                        let logits = head.row_host(engine.library, 0)?;
                        if s.neural { speculate::prefill_chunk(engine, s.placement, s.start, s.tokens, s.next, s.draft)?; }
                        Ok(logits)
                    }, |s, callback| engine.prefill_packed(s, callback))?;
            }
            return Ok(());
        }
        if let Some(windows) = &windows {
            if args.wide_batch { return score_wide_batch(&args, &opened, engine, windows); }
            return score_fidelity(&args, &opened, engine, windows);
        }
        if let Some(dir) = &args.mtp_oracle {
            return mtp_golden::mtp_oracle(&args, &opened, engine, dir);
        }
        if let Some(count) = args.spec_decode {
            return mtp_golden::spec_decode(&args, &opened, engine, count, args.engine.mtp);
        }
        if let Some(steps) = args.token_check {
            return token_check(&args, &opened, engine, steps);
        }
        if !args.resume_at.is_empty() {
            return resume(&args, engine);
        }
        golden_run(&args, &opened, engine)
    })
}

fn packed_check(engine: &engine::Qwen4Engine<'_>, tokens: &[u32], members: usize,
    mode: crate::shared::prefill_packing::PackingMode) -> Result<()> {
    let prompts = crate::shared::qwen::packed_check::prompts(tokens, members)?;
    let mut allocator = engine::Allocator::new(engine.pages, engine.slots, &engine.cfg);
    let mut admit = || prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>();
    let packed = admit()?;
    let alone = admit()?;
    let limits = engine.packed_limits().context("packed Qwen requires one attention owner")?;
    crate::shared::qwen::packed_check::check::<_, speculate::MtpSeq>(engine.library, &prompts, packed, alone,
        limits, engine.weights.mtp.is_some(), mode == crate::shared::prefill_packing::PackingMode::Exact, |s| {
            let head = engine.prefill_device(s.placement, s.tokens, None, None, 1)?.context("packed check needs full model head")?;
            let logits = head.row_host(engine.library, 0)?;
            if s.neural { speculate::prefill_chunk(engine, s.placement, s.start, s.tokens, s.next, s.draft)?; }
            Ok(logits)
        }, |sequences, callback| engine.prefill_packed(sequences, callback),
        |p| prefix::packed_state(engine, p))
}

/// Overwrite the captured decode tables with packed metadata twice, compare
/// logits and durable state against the same decode steps between singleton prefills.
fn packed_decode_interleave(engine: &engine::Qwen4Engine<'_>, tokens: &[u32]) -> Result<()> {
    use crate::shared::qwen::scheduler::PrefillSeq;
    use crate::shared::decode_graph::device::{snapshots, Cycle};
    let mut allocator = engine::Allocator::new(engine.pages, engine.slots, &engine.cfg);
    // Match the singleton arithmetic routes in this diagnostic separately from
    // Exact serving's fail-closed singleton fallback. Both arms are above the
    // 24-row PLE threshold and below the 80-row EXL3 threshold (#890).
    const ROWS: usize = 29;
    const TOKENS: usize = 2 * (ROWS + 1) + 1;
    let mut admit = || Ok::<_, anyhow::Error>([allocator.admit(TOKENS)?, allocator.admit(TOKENS)?]);
    let (mut packed, mut single) = (admit()?, admit()?);
    let mut packed_draft = [speculate::MtpSeq::default(), speculate::MtpSeq::default()];
    let mut single_draft = packed_draft.clone();
    let neural = engine.weights.mtp.is_some();
    let input: [[u32; TOKENS]; 2] = std::array::from_fn(|i| std::array::from_fn(|j| tokens[(i * 97 + j) % tokens.len()]));
    let equal = |a: &[f32], b: &[f32]| a.iter().map(|x| x.to_bits()).eq(b.iter().map(|x| x.to_bits()));
    engine.prepare_serving_workspaces()?;
    let reserve = engine.configure_graph_cache()?;
    ensure!(reserve.iter().any(|&bytes| bytes > 0), "packed/decode golden needs a nonzero graph reserve");
    engine.warm_decode_graphs(2, false)?;
    let graph_hits = || snapshots().iter().filter_map(|s| s.cycles.get(&Cycle::Decode)).map(|c| c.hits).sum::<u64>();
    for start in [0, ROWS + 1] {
        let end = start + ROWS;
        let mut logits = [Vec::new(), Vec::new()];
        let mut batch = packed.iter_mut().zip(&mut packed_draft).zip(&input).map(|((placement, draft), ids)|
            PrefillSeq { placement, tokens: &ids[start..end], start, next: Some(ids[end]), draft, neural }).collect::<Vec<_>>();
        for outcome in engine.prefill_packed(&mut batch, &mut |i, head| {
            logits[i] = head.row_host(engine.library, 0)?;
            Ok(Ok(()))
        })? { outcome?; }
        drop(batch);
        // First decode the packed arm, without another prefill in between.
        let decode = |placement: &mut engine::Qwen4Placement, draft: &mut speculate::MtpSeq, ids: &[u32]| -> Result<_> {
            let head = engine.verify_device(&mut [(placement, &ids[end..end + 1])], false)?
                .context("packed/decode golden head")?;
            let logits = head.row_host(engine.library, 0)?;
            if neural { speculate::prefill_chunk(engine, placement, end, &ids[end..end + 1], Some(ids[end + 1]), draft)?; }
            Ok(logits)
        };
        let before = graph_hits();
        // Table row zero still names member 0. Decode member 1 first so a
        // missing rewrite cannot accidentally read the right table on replay.
        // With lazy exact-row capture, member 1 captures on iteration 0 and
        // member 0 replays it; on iteration 1 both replay cached graphs.
        let second = decode(&mut packed[1], &mut packed_draft[1], &input[1])?;
        let first_decoded = decode(&mut packed[0], &mut packed_draft[0], &input[0])?;
        ensure!(graph_hits() > before, "packed/decode golden did not replay decode graph at row {end}");
        let decoded = [first_decoded, second];
        for i in [1, 0] {
            let ids = &input[i];
            let head = engine.prefill_device(&mut single[i], &ids[start..end], None, None, 1)?
                .context("packed/decode singleton head")?.row_host(engine.library, 0)?;
            ensure!(equal(&logits[i], &head), "packed/decode prefill logits differ at {start}, member {i}");
            if neural { speculate::prefill_chunk(engine, &single[i], start, &ids[start..end], Some(ids[end]), &mut single_draft[i])?; }
            let control = decode(&mut single[i], &mut single_draft[i], ids)?;
            ensure!(equal(&decoded[i], &control) && prefix::packed_state(engine, &packed[i])? == prefix::packed_state(engine, &single[i])?,
                "packed/decode logits or state differ at {end}, member {i}");
        }
    }
    println!("packed/decode/packed: target and MTP state and logits byte-identical, decode graph hits {}", graph_hits());
    Ok(())
}

/// `--resume-at P,..`: [`prefix::resume_check`] for every P and every --prefill-chunk.
fn resume(args: &GoldenArgs, engine: &engine::Qwen4Engine<'_>) -> Result<()> {
    let tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let chunks = if args.prefill_chunk.is_empty() { vec![engine.prefill_rows] } else { args.prefill_chunk.clone() };
    let cases: Vec<prefix::ResumeCase> = chunks.iter().flat_map(|&chunk| args.resume_at.iter().map(move |&at| (at, chunk)))
        .map(|(at, chunk)| prefix::ResumeCase { at, n: args.prefill.unwrap_or(at + args.resume_span).min(tokens.len()),
            chunk })
        .collect();
    let passed = prefix::resume_check(engine, &tokens, &cases, args.resume_decode, args.resume_cold, args.resume_repeat)?;
    ensure!(passed, "a restored sequence differs from the straight one (see above)");
    println!("resume check: all {} cases byte-identical", cases.len());
    Ok(())
}

/// Mean NLL of `logits` rows against the next tokens, and top-1 agreements.
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

fn fidelity_windows(root: &std::path::Path, selected: Option<&str>, packed: bool) -> Result<Vec<cuteafd_bench::reference::Window>> {
    use cuteafd_bench::fidelity_dataset;
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(root.join("manifest.json"))?;
    ensure!(format!("{:x}", Sha256::digest(&bytes)) == fidelity_dataset::QWEN_MANIFEST_SHA256,
        "Qwen fidelity manifest differs from immutable publication");
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == fidelity_dataset::QWEN_CONFIG && manifest["vocab"] == 248320,
        "Qwen fidelity config/vocabulary differs");
    let reference = fidelity_dataset::load(root, &manifest)?;
    // Packed qualification needs long genuine generations from the full panel;
    // ordinary fidelity cards retain their immutable quick subset.
    let mut windows = if packed { reference.windows.clone() } else { fidelity_dataset::quick_windows(&reference)? };
    if let Some(id) = selected {
        windows.retain(|window| window.id == id);
        ensure!(windows.len() == 1, "Qwen fidelity window must belong to the sealed quick subset");
    }
    ensure!(windows.iter().all(|window| window.media.is_empty() && window.tokens.len() <= 16384
        && window.score_from > 0 && window.score_from < window.tokens.len()),
        "Qwen fidelity requires text-only bounded causal windows");
    Ok(windows)
}

fn fidelity_chunks(tokens: usize, capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    ensure!((2..=16384).contains(&tokens) && capacity > 0, "Qwen fidelity causal window/chunk extent differs");
    let count = tokens - 1;
    Ok((0..count).step_by(capacity).map(|first| first..(first + capacity).min(count)).collect())
}

fn score_fidelity(args: &GoldenArgs, opened: &Opened, engine: &engine::Qwen4Engine<'_>,
    windows: &[cuteafd_bench::reference::Window]) -> Result<()> {
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};
    use cuteafd_bench::{fidelity_dataset, fidelity_rows, reference::Fidelity};
    ensure!(engine.weights.layers.len() == opened.cfg.layers && opened.cfg.vocab_size == 248320,
        "Qwen fidelity needs every target layer and the published vocabulary");
    let root = args.save_logits.as_deref().context("shared Qwen fidelity requires --save-logits")?;
    std::fs::create_dir(root).context("Qwen fidelity output must be a new directory with an existing parent")?;
    let provenance = serde_json::json!({"prefill_rows":engine.prefill_rows,
        "prefill_shape":engine.prefill_shape.map(|shape| serde_json::json!({"unit_rows":shape.unit_rows,"waves":shape.waves})),
        "kv_cache":args.engine.kv_format.to_string(), "fp8_decode":args.engine.fp8_decode,
        "fp8_prefill_w8a8":args.engine.fp8_prefill_w8a8});
    let mut allocator = engine::Allocator::new(engine.pages, engine.slots, &opened.cfg);
    let mut records = Vec::new();
    let vocab = opened.cfg.vocab_size;
    for window in windows {
        let mut placement = allocator.admit(window.tokens.len() - 1)?;
        let dump = root.join(&window.id);
        let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true,
            prompt_ids: Some(window.tokens.clone()), score_from: Some(window.score_from),
            score_positions: Some(window.positions.iter().map(|p| p.pos).collect()), top_k: 32,
            want: window.want(), dump_rows: Some(dump.clone()), ..Default::default() });
        probe.admitted("qwen4", &window.tokens, 0);
        probe.selected_score_path("prefill");
        probe.provenance(provenance.clone());
        let consumer = Some(probe.clone());
        for chunk in fidelity_chunks(window.tokens.len(), engine.prefill_rows)? {
            engine.prefill_consumer(&mut placement, &window.tokens[chunk.clone()], None, None, chunk.len(),
                crate::shared::probe::ScoringRows { consumer: &consumer, position: chunk.start + 1 })?
                .context("Qwen fidelity needs all target logits")?;
            probe.scored_prefill_chunk(chunk.start + 1, chunk.len());
        }
        let record = probe.record();
        ensure!(record.error.is_none(), "Qwen fidelity row dump failed: {:?}", record.error);
        let mut score = window.score(&record.rows);
        ensure!(score.missing == 0 && score.non_finite == 0 && score.records.len() == window.positions.len(),
            "Qwen fidelity did not cover every finite scored row");
        fidelity_rows::score_compact(vocab, window, &dump, &mut score)?;
        println!("Qwen window {}: rows={} top-1={:.4}% mean NLL={:.8} mean KL(top1024+tail)={:.8}",
            window.id, score.positions, score.top1 * 100.0, score.nll, score.kl);
        let file = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join(format!("{}.json", window.id)))?;
        serde_json::to_writer(file, &serde_json::json!({"probe":record,"score":score}))?;
        records.extend(score.records);
        engine.drain_state()?;
        allocator.release(placement);
    }
    let score = Fidelity::from_records(records);
    println!("Qwen prefill logits: rows={} top-1 agreement {:.4}% | mean NLL engine {:.8} | mean KL(top1024+tail) {:.8}",
        score.positions, score.top1 * 100.0, score.nll, score.kl);
    let file = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join("fidelity.json"))?;
    serde_json::to_writer(file, &serde_json::json!({"schema":"cuteafd.qwen4.golden/1",
        "reference_config":fidelity_dataset::QWEN_CONFIG,"reference_revision":fidelity_dataset::QWEN_REVISION,
        "reference_manifest_sha256":fidelity_dataset::QWEN_MANIFEST_SHA256,
        "path_shape":"prefill-shaped","kl_kind":"top1024+tail","provenance":provenance,"score":score}))?;
    Ok(())
}

/// Multi-sequence 128-row verify + GDN/PLE commit + graphed replay oracle.
/// Runs 32 independent sequences (4 verify tokens each = 128 rows) solo (1x4),
/// batched 32x4 eager, and batched 32x4 graphed (capture + replay), committing
/// varied kept counts (1..=4) via `qwen4_gdn_commit_m128` + `qwen4_ple_commit_m128`
/// and verifying post-commit follow-up decode parity against solo.
fn score_wide_batch(args: &GoldenArgs, opened: &Opened, engine: &engine::Qwen4Engine<'_>,
    windows: &[cuteafd_bench::reference::Window]) -> Result<()> {
    use crate::shared::decode_graph::device::{snapshots, Cycle};
    ensure!(engine.decode_rows == 128, "Qwen4 --wide-batch requires --decode-rows 128 (got {})", engine.decode_rows);
    ensure!(engine.slots >= 32, "Qwen4 --wide-batch requires at least 32 slots (got {})", engine.slots);
    ensure!(!windows.is_empty(), "Qwen4 --wide-batch needs at least one fidelity window");
    let root = args.save_logits.as_deref().context("--wide-batch requires --save-logits")?;
    std::fs::create_dir(root).context("Qwen4 wide-batch output must be a new directory")?;
    let vocab = opened.cfg.vocab_size;

    const SEQS: usize = 32;
    const VERIFY: usize = 4;
    const TOTAL_ROWS: usize = SEQS * VERIFY;
    struct Sample { prefix: Vec<u32>, verify: Vec<u32>, kept: usize, followup: u32 }
    let mut samples = Vec::with_capacity(SEQS);
    for idx in 0..SEQS {
        let win = &windows[idx % windows.len()];
        let prefix_len = 24 + (idx * 5) % 40;
        let offset = (idx / windows.len()) * 80;
        ensure!(offset + prefix_len + VERIFY + 1 <= win.tokens.len(), "Qwen4 --wide-batch window too short");
        let prefix = win.tokens[offset..offset + prefix_len].to_vec();
        let verify = win.tokens[offset + prefix_len..offset + prefix_len + VERIFY].to_vec();
        let kept = (idx % VERIFY) + 1;
        let followup = win.tokens[offset + prefix_len + kept];
        samples.push(Sample { prefix, verify, kept, followup });
    }

    let argmax = |row: &[f32]| -> usize {
        row.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i).unwrap_or(0)
    };
    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (xd, yd) = (x as f64, y as f64);
            dot += xd * yd; na += xd * xd; nb += yd * yd;
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-12)
    };
    let compare_rows = |a: &[Vec<f32>], b: &[Vec<f32>]| -> (usize, usize, f32, f64) {
        let (mut bit_eq, mut top1_eq, mut max_abs, mut min_cos) = (0usize, 0usize, 0.0f32, 1.0f64);
        for (ra, rb) in a.iter().zip(b.iter()) {
            let mut exact = true;
            for (&x, &y) in ra.iter().zip(rb.iter()) {
                if x.to_bits() != y.to_bits() { exact = false; }
                let diff = (x - y).abs();
                if diff > max_abs { max_abs = diff; }
            }
            if exact { bit_eq += 1; }
            if argmax(ra) == argmax(rb) { top1_eq += 1; }
            let c = cosine(ra, rb);
            if c < min_cos { min_cos = c; }
        }
        (bit_eq, top1_eq, max_abs, min_cos)
    };

    engine.prepare_serving_workspaces()?;
    let reserve = engine.configure_graph_cache()?;
    ensure!(reserve.iter().any(|&bytes| bytes > 0), "Qwen4 --wide-batch needs a nonzero graph reserve");
    engine.enable_lazy_graphs_for_golden()?;
    let graph_counters = || -> (u64, u64) {
        let snaps = snapshots();
        let sum = |f: fn(&crate::shared::decode_graph::device::Counters) -> u64| -> u64 {
            snaps.iter().flat_map(|s| [Cycle::Verify, Cycle::Decode].into_iter().filter_map(|c| s.cycles.get(&c))).map(f).sum()
        };
        (sum(|c| c.captures), sum(|c| c.hits))
    };

    let mut allocator = engine::Allocator::new(engine.pages, engine.slots, &opened.cfg);

    // 1. Solo reference: each sequence individually (4-row verify + partial commit + 1-row follow-up).
    let mut solo_verify: Vec<Vec<f32>> = Vec::with_capacity(TOTAL_ROWS);
    let mut solo_followup: Vec<Vec<f32>> = Vec::with_capacity(SEQS);
    for sample in &samples {
        let mut p = allocator.admit(sample.prefix.len() + VERIFY + 1)?;
        engine.prefill_device(&mut p, &sample.prefix, None, None, 1)?;
        let start = p.len;
        let history = p.history.clone();
        let logits = engine.verify_device_ungraphed(&mut [(&mut p, &sample.verify)], true)?
            .context("Qwen4 solo verify logits")?;
        let values = logits.to_host(engine.library)?;
        ensure!(values.len() == VERIFY * vocab, "Qwen4 solo verify logits shape");
        for row in values.chunks_exact(vocab) { solo_verify.push(row.to_vec()); }
        engine.commit(&[(p.slot, 0, sample.kept)])?;
        engine.rewind(&mut p, start, history, &sample.verify[..sample.kept]);
        let follow = engine.verify_device_ungraphed(&mut [(&mut p, &[sample.followup])], false)?
            .context("Qwen4 solo follow-up logits")?;
        let follow_vals = follow.to_host(engine.library)?;
        ensure!(follow_vals.len() == vocab, "Qwen4 solo follow-up logits shape");
        solo_followup.push(follow_vals);
        engine.drain_state()?;
        allocator.release(p);
    }

    // Helper: run all 32 sequences together as one 128-row verify + 32-slot commit + per-slot 1-row follow-up.
    let mut run_batched = |graphed: bool| -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, u64, u64)> {
        let mut placements = Vec::with_capacity(SEQS);
        for sample in &samples {
            let mut p = allocator.admit(sample.prefix.len() + VERIFY + 1)?;
            engine.prefill_device(&mut p, &sample.prefix, None, None, 1)?;
            placements.push(p);
        }
        let starts: Vec<usize> = placements.iter().map(|p| p.len).collect();
        let histories: Vec<_> = placements.iter().map(|p| p.history.clone()).collect();
        let (caps_before, hits_before) = graph_counters();
        let logits = {
            let mut batch: Vec<(&mut engine::Qwen4Placement, &[u32])> = placements.iter_mut()
                .zip(&samples).map(|(p, s)| (p, s.verify.as_slice())).collect();
            if graphed {
                engine.verify_device(&mut batch, true)?.context("Qwen4 wide-batch graphed verify logits")?
            } else {
                engine.verify_device_ungraphed(&mut batch, true)?.context("Qwen4 wide-batch eager verify logits")?
            }
        };
        let values = logits.to_host(engine.library)?;
        let (caps_after, hits_after) = graph_counters();
        ensure!(values.len() == TOTAL_ROWS * vocab, "Qwen4 wide-batch verify logits shape");
        let verify_rows: Vec<Vec<f32>> = values.chunks_exact(vocab).map(|r| r.to_vec()).collect();
        let accepted: Vec<(i32, usize, usize)> = placements.iter().zip(&samples).enumerate()
            .map(|(i, (p, s))| (p.slot, i * VERIFY, s.kept)).collect();
        engine.commit(&accepted)?;
        for (((p, &start), history), s) in placements.iter_mut().zip(&starts).zip(histories).zip(&samples) {
            engine.rewind(p, start, history, &s.verify[..s.kept]);
        }
        let mut follow_rows = Vec::with_capacity(SEQS);
        for (p, s) in placements.iter_mut().zip(&samples) {
            let follow = engine.verify_device_ungraphed(&mut [(p, &[s.followup])], false)?
                .context("Qwen4 wide-batch follow-up logits")?;
            let follow_vals = follow.to_host(engine.library)?;
            ensure!(follow_vals.len() == vocab, "Qwen4 wide-batch follow-up logits shape");
            follow_rows.push(follow_vals);
        }
        engine.drain_state()?;
        for p in placements { allocator.release(p); }
        Ok((verify_rows, follow_rows, caps_after - caps_before, hits_after - hits_before))
    };

    let (eager_verify, eager_follow, eager_caps, eager_hits) = run_batched(false)?;
    let (cap_verify, cap_follow, cap_caps, cap_hits) = run_batched(true)?;
    let (rep_verify, rep_follow, rep_caps, rep_hits) = run_batched(true)?;

    let (solo_vs_eager_bit, solo_vs_eager_top1, solo_vs_eager_max, solo_vs_eager_cos) = compare_rows(&solo_verify, &eager_verify);
    let (eager_vs_rep_bit, eager_vs_rep_top1, eager_vs_rep_max, eager_vs_rep_cos) = compare_rows(&eager_verify, &rep_verify);
    let (cap_vs_rep_bit, cap_vs_rep_top1, cap_vs_rep_max, cap_vs_rep_cos) = compare_rows(&cap_verify, &rep_verify);
    let (follow_solo_vs_eager_bit, follow_solo_vs_eager_top1, follow_solo_vs_eager_max, follow_solo_vs_eager_cos) = compare_rows(&solo_followup, &eager_follow);
    let (follow_solo_vs_rep_bit, follow_solo_vs_rep_top1, follow_solo_vs_rep_max, follow_solo_vs_rep_cos) = compare_rows(&solo_followup, &rep_follow);
    let (follow_cap_vs_rep_bit, follow_cap_vs_rep_top1, follow_cap_vs_rep_max, follow_cap_vs_rep_cos) = compare_rows(&cap_follow, &rep_follow);

    let receipt = serde_json::json!({
        "schema": "cuteafd.qwen4.wide-batch-128/1",
        "decode_rows": engine.decode_rows,
        "sequences": SEQS,
        "verify_per_sequence": VERIFY,
        "total_verify_rows": TOTAL_ROWS,
        "kv_cache": args.engine.kv_format.to_string(),
        "passes": {
            "eager": { "captures": eager_caps, "hits": eager_hits },
            "graphed_capture": { "captures": cap_caps, "hits": cap_hits },
            "graphed_replay": { "captures": rep_caps, "hits": rep_hits },
        },
        "verify_parity": {
            "solo_vs_eager": { "bit_identical_rows": solo_vs_eager_bit, "top1_exact_rows": solo_vs_eager_top1, "total_rows": TOTAL_ROWS, "max_abs_diff": solo_vs_eager_max, "min_cosine": solo_vs_eager_cos },
            "eager_vs_graphed_replay": { "bit_identical_rows": eager_vs_rep_bit, "top1_exact_rows": eager_vs_rep_top1, "total_rows": TOTAL_ROWS, "max_abs_diff": eager_vs_rep_max, "min_cosine": eager_vs_rep_cos },
            "graphed_capture_vs_replay": { "bit_identical_rows": cap_vs_rep_bit, "top1_exact_rows": cap_vs_rep_top1, "total_rows": TOTAL_ROWS, "max_abs_diff": cap_vs_rep_max, "min_cosine": cap_vs_rep_cos },
        },
        "gdn_ple_commit_m128_followup_parity": {
            "solo_vs_eager": { "bit_identical_rows": follow_solo_vs_eager_bit, "top1_exact_rows": follow_solo_vs_eager_top1, "total_rows": SEQS, "max_abs_diff": follow_solo_vs_eager_max, "min_cosine": follow_solo_vs_eager_cos },
            "solo_vs_graphed_replay": { "bit_identical_rows": follow_solo_vs_rep_bit, "top1_exact_rows": follow_solo_vs_rep_top1, "total_rows": SEQS, "max_abs_diff": follow_solo_vs_rep_max, "min_cosine": follow_solo_vs_rep_cos },
            "graphed_capture_vs_replay": { "bit_identical_rows": follow_cap_vs_rep_bit, "top1_exact_rows": follow_cap_vs_rep_top1, "total_rows": SEQS, "max_abs_diff": follow_cap_vs_rep_max, "min_cosine": follow_cap_vs_rep_cos },
        },
    });
    println!("Qwen4 wide-batch 128-row oracle: {}", serde_json::to_string_pretty(&receipt)?);
    let out = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join("wide-batch-128.json"))?;
    serde_json::to_writer_pretty(out, &receipt)?;

    ensure!(cap_caps > 0 && rep_caps == 0 && rep_hits > 0 && rep_hits == cap_caps,
        "Qwen4 128-row graphed replay failed (cap=({cap_caps},{cap_hits}), rep=({rep_caps},{rep_hits}))");
    ensure!(solo_vs_eager_cos >= 0.95 && solo_vs_eager_top1 >= 115,
        "Qwen4 128-row solo vs batched verify mismatch ({solo_vs_eager_top1}/{TOTAL_ROWS}, min_cos={solo_vs_eager_cos}, max_abs={solo_vs_eager_max})");
    ensure!(eager_vs_rep_bit == TOTAL_ROWS && cap_vs_rep_bit == TOTAL_ROWS && follow_cap_vs_rep_bit == SEQS,
        "Qwen4 128-row graphed replay diverged (eager_vs_rep bit={eager_vs_rep_bit}/{TOTAL_ROWS} max={eager_vs_rep_max}, cap_vs_rep bit={cap_vs_rep_bit}/{TOTAL_ROWS})");
    ensure!(follow_solo_vs_eager_cos >= 0.95 && follow_solo_vs_rep_cos >= 0.95 && follow_solo_vs_eager_top1 >= 27 && follow_solo_vs_rep_top1 >= 27,
        "Qwen4 32-slot gdn/ple commit_m128 follow-up mismatch (eager={follow_solo_vs_eager_top1}/{SEQS} cos={follow_solo_vs_eager_cos}, rep={follow_solo_vs_rep_top1}/{SEQS} cos={follow_solo_vs_rep_cos})");
    Ok(())
}

fn golden_logits(root: &std::path::Path, dump_only: bool, rows: usize, vocab: usize)
    -> Result<Option<crate::shared::golden_scores::ReferenceRows>> {
    crate::shared::golden_scores::ReferenceRows::open(&root.join("logits.bin"), rows, vocab, dump_only)
}

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::Qwen4Engine<'_>) -> Result<()> {
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    let token_bytes = std::fs::read(args.golden.join("tokens.bin"))?;
    ensure!(!token_bytes.is_empty() && token_bytes.len().is_multiple_of(4), "Qwen golden token extent differs");
    let tokens: Vec<u32> = token_bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let reference_logits = golden_logits(&args.golden, args.save_logits.is_some(),
        tokens.len(), cfg.vocab_size)?;
    if reference_logits.is_none() { println!("Qwen engine-only logits dump; no reference NLL/KL or agreement comparison"); }
    let mut allocator = engine::Allocator::new(engine.pages, engine.slots, cfg);
    let mut placement = allocator.admit(tokens.len() + args.bench_decode + usize::from(args.bench_decode > 0))?;
    let row = cfg.hidden * 2;
    let stream_row = row * 4;
    let prefill = args.prefill.unwrap_or(tokens.len()).min(tokens.len());
    ensure!(prefill > 0, "Qwen golden requires a nonempty prefill");
    // Rows [first, first + n) of layer `layer`'s golden streams.
    let compare = |layer: usize, first: usize, streams: &[u8], worst: &mut Vec<f64>| -> Result<()> {
        let path = args.golden.join(format!("layer{layer:02}.bin"));
        let Ok(file) = std::fs::File::open(&path) else { return Ok(()) };
        let mut golden = vec![0u8; streams.len()];
        file.read_exact_at(&mut golden, (first * stream_row) as u64)
            .map_err(|e| anyhow::anyhow!("golden layer {layer} is short: {e}"))?;
        let (ours, theirs) = (bf16s(streams), bf16s(&golden));
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
        Ok(())
    };
    let mut worst = Vec::new();
    let started = Instant::now();
    let forced = |layer: usize| -> Option<Vec<u8>> {
        let file = std::fs::File::open(args.golden.join(format!("layer{layer:02}.bin"))).ok()?;
        let mut rows = vec![0u8; prefill * stream_row];
        file.read_exact_at(&mut rows, 0).ok()?;
        Some(rows)
    };
    ensure!(!args.teacher_force || prefill <= engine.prefill_rows, "teacher forcing takes one prefill chunk");
    use std::io::Write;
    let mut logits: Option<Vec<f32>> = None;
    let mut prefill_metrics = crate::shared::golden_scores::RowMetrics::default();
    let mut saved_prefill = if args.nll {
        args.save_logits.as_ref().map(|dir| -> Result<_> {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("tokens.bin"), &token_bytes[..prefill * 4])?;
            Ok(std::io::BufWriter::new(std::fs::File::create(dir.join("logits.bin"))?))
        }).transpose()?
    } else { None };
    let mut raw_prefill = if args.nll {
        std::env::var_os("CUTEAFD_DUMP_PREFILL_LOGITS").map(|path|
            std::fs::File::create(path).map(std::io::BufWriter::new)).transpose()?
    } else { None };
    let mut done = 0;
    while done < prefill {
        let n = engine.prefill_rows.min(prefill - done);
        let first = done;
        let mut on_layer = |layer, streams: &[u8]| compare(layer, first, streams, &mut worst);
        let hook = if engine.prefill_shape.is_some() {
            ensure!(!args.teacher_force, "Qwen unit golden compares logits, not teacher-forced layer streams");
            None
        } else { Some(&mut on_layer as &mut dyn FnMut(usize, &[u8]) -> Result<()>) };
        let forced = args.teacher_force.then_some(&forced as &dyn Fn(usize) -> Option<Vec<u8>>);
        if args.nll {
            let consumer = crate::shared::probe::CallbackRows::new(|position: usize, values: &[f32]| {
                let row = position.checked_sub(1).context("Qwen golden zero score position")?;
                ensure!(values.len() == cfg.vocab_size && row == prefill_metrics.rows,
                    "Qwen golden logits row order/shape differs");
                let reference = reference_logits.as_ref().map(|r| r.row(row)).transpose()?;
                prefill_metrics.push(values, reference.as_deref(), tokens.get(row + 1).copied())?;
                for writer in [&mut saved_prefill, &mut raw_prefill].into_iter().flatten() {
                    for value in values { writer.write_all(&value.to_le_bytes())?; }
                }
                logits = Some(values.to_vec());
                Ok(())
            });
            engine.prefill_consumer(&mut placement, &tokens[done..done + n], hook, forced, n,
                crate::shared::probe::ScoringRows { consumer: &consumer, position: first + 1 })?
                .context("Qwen golden needs every target layer")?;
        } else {
            logits = engine.prefill_forced(&mut placement, &tokens[done..done + n], hook, forced, 1)?;
        }
        done += n;
    }
    if args.nll { ensure!(prefill_metrics.rows == prefill, "Qwen golden prefill row coverage differs"); }
    for writer in [&mut saved_prefill, &mut raw_prefill].into_iter().flatten() { writer.flush()?; }
    let prefill_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut decode_worst = Vec::new();
    let mut decode_logits: Vec<f32> = Vec::new();
    let mut position = prefill;
    let mut spec_steps = 0usize;
    while position < tokens.len() {
        let n = args.step_rows.min(tokens.len() - position);
        let first = position;
        let rows = &tokens[position..position + n];
        let Some(keep) = args.spec_keep else {
            if let Some(logits) = engine.verify(&mut [(&mut placement, rows)],
                Some(&mut |layer, streams| compare(layer, first, streams, &mut decode_worst)))? {
                decode_logits.extend(logits);
            }
            position += n;
            continue;
        };
        // Speculative: verify n rows, keep the first `keep` (the rest are verified again next step).
        let keep = keep.clamp(1, n);
        let history = placement.history.clone();
        let logits = engine.verify_spec(&mut [(&mut placement, rows)],
            Some(&mut |layer, streams| compare(layer, first, streams, &mut decode_worst)))?;
        engine.commit(&[(placement.slot, 0, keep)])?;
        engine.rewind(&mut placement, first, history, &rows[..keep])?;
        if let Some(logits) = logits {
            decode_logits.extend_from_slice(&logits[..keep * cfg.vocab_size]);
        }
        spec_steps += 1;
        position += keep;
    }
    if args.spec_keep.is_some() {
        println!("speculative verify: {spec_steps} steps of {} rows, each committing {} (GDN/PLE verify-by-replay)",
            args.step_rows, args.spec_keep.unwrap_or(0));
    }
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
            if let Some(golden) = &reference_logits {
                let mut metrics = crate::shared::golden_scores::RowMetrics::default();
                for (row, values) in decode_logits.chunks_exact(vocab).enumerate() {
                    let reference = golden.row(prefill + row)?;
                    metrics.push(values, Some(&reference), tokens.get(prefill + row + 1).copied())?;
                }
                println!("{}", metrics.context_only_line("decode"));
            }
        }
    }
    // Reference dumps for `scripts/bench/golden-run.py --compare-logits`: with --nll every
    // prefill row, otherwise the decode/verify rows past --prefill (tokens.bin from --prefill).
    if let Some(dir) = &args.save_logits {
        std::fs::create_dir_all(dir)?;
        if args.nll {
            println!("saved {prefill} prefill rows to {}", dir.display());
        } else {
            ensure!(prefill < tokens.len(), "--save-logits without --nll saves the decode rows: pass --prefill N");
            ensure!(decode_logits.len() == (tokens.len() - prefill) * vocab,
                "saved logits cover {} of {} decode rows", decode_logits.len() / vocab, tokens.len() - prefill);
            std::fs::write(dir.join("tokens.bin"), tokens[prefill..].iter()
                .flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())?;
            std::fs::write(dir.join("logits.bin"), decode_logits.iter()
                .flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
            println!("saved {} decode rows (positions {prefill}..{}) to {}",
                decode_logits.len() / vocab, tokens.len(), dir.display());
        }
    }
    if args.bench_prefill > 0 {
        let n = args.bench_prefill_tokens.unwrap_or(prefill.min(engine.prefill_rows));
        let long: Vec<u32> = tokens.iter().copied().cycle().take(n).collect();
        let mut times = Vec::new();
        for _ in 0..args.bench_prefill {
            let mut fresh = allocator.admit(n)?;
            let started = Instant::now();
            for chunk in long.chunks(engine.prefill_rows) {
                engine.prefill(&mut fresh, chunk)?;
            }
            times.push(started.elapsed().as_secs_f64());
            allocator.release(fresh);
        }
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        println!("prefill bench: {n} tokens through {layers} layers, median {:.1} ms ({:.0} tok/s), min {:.1} ms",
            1e3 * median, n as f64 / median, 1e3 * times[0]);
    }
    if args.bench_decode > 0 {
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
        let mut token = tokens[placement.len.min(tokens.len() - 1)];
        let mut times = Vec::new();
        let mut produced = Vec::new();
        // Capture the complete decode shape and warm its kernels before timing.
        let warm = [token];
        if let Some(logits) = engine.verify(&mut [(&mut placement, &warm[..])], None)? {
            token = argmax(&logits);
        }
        let warm_graphs = engine.captured_graphs();
        *engine.profile.borrow_mut() = [0.0; 2];
        // FNV-1a over every step's logits bits (bit-identity checks between configs).
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..args.bench_decode {
            let started = Instant::now();
            let step = [token];
            let logits = engine.verify(&mut [(&mut placement, &step[..])], None)?;
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
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        println!("decode bench: {} steps through {layers} layers, median {:.3} ms (mean {:.3}, min {:.3}, max {:.2}); \
            expert GPU wait {:.1} ms, exchange {:.1} ms total; logits digest {digest:016x}; tokens {:?}; \
            graph captures warm {warm_graphs}, timed {}", times.len(),
            1e3 * times[times.len() / 2], 1e3 * mean, 1e3 * times[0], 1e3 * times[times.len() - 1], 1e3 * profile[0],
            1e3 * profile[1], &produced[..produced.len().min(16)], engine.captured_graphs() - warm_graphs);
    }
    let loads = match engine.experts() {
        Some(engine::Experts::Hybrid(experts)) => format!(", {} resident routed expert layers", experts.local_layers.len()),
        _ => String::new(),
    };
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s{loads}");
    if let Some(logits) = logits {
        // Match the decode dump for exact before/after prefill qualification.
        if !args.nll {
            if let Ok(path) = std::env::var("CUTEAFD_DUMP_PREFILL_LOGITS") {
                std::fs::write(&path, logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
        let Some(golden) = &reference_logits else { return Ok(()) };
        if args.nll {
            println!("{}", prefill_metrics.context_only_line("prefill"));
        }
        let logits = &logits[logits.len() - vocab..];
        let last = golden.row(prefill - 1)?;
        let (cosine, _) = similarity(logits, &last);
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(logits), argmax(&last));
    }
    Ok(())
}

#[cfg(test)]
mod golden_tests {
    use super::{fidelity_chunks, fidelity_windows, golden_logits};

    #[test]
    fn causal_fidelity_chunks_cover_next_token_positions_once() {
        for tokens in [513, 751, 4097, 8193, 10773] {
            for capacity in [512, 2048, 4096, 16384] {
                let chunks = fidelity_chunks(tokens, capacity).unwrap();
                let predicted = chunks.iter().flat_map(|chunk| chunk.clone().map(|row| row + 1)).collect::<Vec<_>>();
                assert_eq!(predicted, (1..tokens).collect::<Vec<_>>());
                assert!(chunks.iter().all(|chunk| !chunk.is_empty() && chunk.len() <= capacity));
            }
        }
        assert!(fidelity_chunks(1, 2048).is_err());
        assert!(fidelity_chunks(513, 0).is_err());
    }

    #[test]
    fn engine_only_logits_dump_requires_explicit_opt_in_and_rejects_bad_references() {
        let dir = tempfile::tempdir().unwrap();
        assert!(golden_logits(dir.path(), true, 2, 2).unwrap().is_none());
        assert!(golden_logits(dir.path(), false, 2, 2).is_err());
        let values = [0.0f32, 1.0, -1.0, 2.0];
        let path = dir.path().join("logits.bin");
        std::fs::write(&path, values.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();
        let reference = golden_logits(dir.path(), true, 2, 2).unwrap().unwrap();
        assert_eq!(reference.row(0).unwrap(), values[..2]);
        assert_eq!(reference.row(1).unwrap(), values[2..]);
        assert!(reference.row(2).is_err());
        assert!(golden_logits(dir.path(), true, 3, 1).is_err());
        std::fs::write(&path, [f32::NAN, 1.0].into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();
        assert!(golden_logits(dir.path(), true, 1, 2).is_err()); // refused at open (#2719)
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(golden_logits(dir.path(), true, 1, 1).is_err());
    }

    #[test]
    fn shared_fidelity_rejects_an_unpublished_manifest() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("manifest.json"), b"{}").unwrap();
        assert!(fidelity_windows(dir.path(), None, false).unwrap_err().to_string().contains("immutable publication"));
    }
}

/// `--token-check N`: prefill the golden prompt, then [`crate::shared::token_io::gate`]
/// over N greedy decode steps.
fn token_check(args: &GoldenArgs, opened: &Opened, engine: &engine::Qwen4Engine<'_>, steps: usize) -> Result<()> {
    let mut tokens: Vec<u32> = std::fs::read(args.golden.join("tokens.bin"))?
        .chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    if let Some(p) = args.prefill {
        tokens.truncate(p);
    }
    let mut allocator = engine::Allocator::new(engine.pages, engine.slots, &opened.cfg);
    let mut placement = allocator.admit(tokens.len() + steps + 1)?;
    let mut last = None;
    for chunk in tokens.chunks(engine.prefill_rows) {
        last = engine.prefill_device(&mut placement, chunk, None, None, 1)?;
    }
    let last = last.ok_or_else(|| anyhow::anyhow!("--token-check needs every layer"))?.row_host(&opened.library, 0)?;
    let first = cuteafd_core::TargetSamplingParams::greedy().select_token(&last, None, 0)? as u32;
    let result = crate::shared::token_io::gate(&opened.library, &engine.embedding, first, steps, |token| {
        engine.verify_device(&mut [(&mut placement, &[token][..])], false)?
            .ok_or_else(|| anyhow::anyhow!("decode needs every layer"))
    });
    allocator.release(placement);
    result
}
