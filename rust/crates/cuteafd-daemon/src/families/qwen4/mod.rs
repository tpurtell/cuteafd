//! Qwen 3.8 Flash Next (qwen4_exp) on the generic engine: weights, the PLE
//! n-gram table, the coordinator programs' layer chain, and the golden
//! comparison command.
mod media;
pub(crate) mod engine;
mod admission;
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
    /// Checkpoint snapshot (coordinator weights, PLE table, config, tokenizer).
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// Spark-free local TP2 with solver-owned whole-layer attention ranges.
    #[arg(long, env = "CUTEAFD_QWEN_SPLIT_DEVICE", requires = "local_experts", conflicts_with_all = ["peers", "shared_only", "expert_window"])]
    pub split_device: Option<i32>,
    /// Expert onboarding policy for an explicit local coordinator pair.
    #[arg(long = "rtx-expert-layers", requires = "split_device", value_parser = |text: &str| text.parse::<cuteafd_loader::placement::Onboard>())]
    pub onboard: Option<cuteafd_loader::placement::Onboard>,
    /// Attention ownership; dual Qwen supports whole layers, not head shards.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLACEMENT", default_value = "auto", value_parser = ["auto", "heads", "layers", "context"])]
    pub attention_placement: String,
    /// Run only the first N layers.
    #[arg(long)]
    pub layers: Option<usize>,
    /// Longest sequence; 0 selects checkpoint full, bounded by compiled support and the admitted pool.
    #[arg(long, default_value_t = 0)]
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
    /// Serving graph modes, set before weight/KV admission; diagnostics stay lazy.
    #[arg(skip)]
    pub planner_graph_modes: Option<(usize, bool)>,
    /// Resolve the serving host quota once the final pool and context are known.
    #[arg(skip)]
    pub planner_host_prefix: Option<(crate::shared::prefix::PrefixArgs, usize)>,
    /// Sequences with GDN/PLE state (about 115 MiB each).
    #[arg(long, default_value_t = 8)]
    pub slots: usize,
    #[arg(long, default_value_t = 4096)]
    pub prefill_rows: usize,
    /// Units in flight on the one Spark transport (one keeps the legacy path).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub prefill_waves: u8,
    /// Diagnostic unit cuts, including N=1, without changing the span size.
    #[arg(long)]
    pub prefill_units: Option<usize>,
    /// Admit every prefill row's logits at startup for fidelity probes.
    #[arg(long)]
    pub full_prefill_logits: bool,
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
    /// Gather and prefetch options for the mapped PLE n-gram table.
    #[command(flatten)]
    pub table: crate::shared::mapped_table::MappedTableArgs,
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
    /// Most EXL3 expert layers resident at once (the free memory decides first).
    #[arg(long, default_value_t = 48)]
    pub exl3_window: usize,
    /// GPU memory (GiB) kept free of local experts for step workspaces
    /// (logits alone are 4 GiB at 4096 rows).
    #[arg(long, default_value_t = 12)]
    pub expert_reserve_gib: usize,
    /// Native MTP drafts per step (0: no MTP). Loads the MTP layer (`mtp.*`,
    /// its experts local) and verifies up to this many drafts per sequence.
    #[arg(long, default_value_t = 0)]
    pub mtp: usize,
    /// Numerical attention recipe rows; automatic uses the actually loaded MTP width.
    #[arg(long, env = "CUTEAFD_ATTENTION_PLAN_ROWS")]
    pub attention_plan_rows: Option<usize>,
    /// Hold lm_head as E4M3 with FP32 per-row x 128-K scales (quantized on the
    /// host at load) INSTEAD of BF16: the one head the target (prefill, verify,
    /// golden rows) and the MTP drafts share, every logits row through
    /// `qwen4_head_fp8` in 16-row spans. Default false: both share the BF16 head.
    /// Default true: measured on one RTX (C1 code 196 -> 222 tok/s, KL 0.034 ->
    /// 0.036, top-1 unchanged 88.5%). --fp8-decode stays off: KL +0.012.
    #[arg(long, alias = "fp8-head", default_value_t = true, action = clap::ArgAction::Set)]
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
}

impl EngineArgs {
    fn prefill_shape(&self) -> Result<Option<cuteafd_loader::serving_capacity::PrefillShape>> {
        if self.prefill_units.is_none() && self.prefill_waves == 1 { return Ok(None); }
        ensure!(self.split_device.is_none(), "Qwen prefill units do not admit dual-owner hop slots");
        ensure!(self.peers.is_some() || self.shared_only, "Qwen prefill units need Spark experts or shared-only diagnostics");
        let shape = cuteafd_loader::serving_capacity::PrefillShape {
            unit_rows: self.prefill_units.unwrap_or(self.prefill_rows) as u64, waves: u64::from(self.prefill_waves),
        }.validate()?;
        ensure!(shape.waves <= 2, "Qwen mapped PLE gathers admit at most two prefill waves");
        ensure!(self.prefill_rows <= 8192, "Qwen prefill span exceeds 8192 rows");
        Ok(Some(shape))
    }

    /// Per-wave tables, PLE gathers and Spark workers never hold the whole span.
    fn prefill_wave_rows(&self) -> usize {
        self.prefill_units.unwrap_or(self.prefill_rows).max(engine::DECODE_ROWS)
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

fn required_fp8_programs(args: &EngineArgs) -> Vec<String> {
    let mut required = Vec::new();
    if args.fp8_decode {
        let suffix = if args.kv_format == cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 { "_kv_fp8" } else { "" };
        for cap in ["m64", "m4096"] {
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
    cuteafd_core::coordinator_programs::AttentionReductionPlan::new(
        engine::DECODE_ROWS, loaded_mtp.then_some(args.mtp), args.attention_plan_rows)
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
        for (rows, waves) in [(512, 1), (1024, 2), (2048, 2), (4096, 2)] {
            let rows_arg = rows.to_string();
            let waves_arg = waves.to_string();
            let parsed = args(&["--shared-only", "--prefill-units", &rows_arg, "--prefill-waves", &waves_arg])?;
            let shape = parsed.prefill_shape()?.unwrap();
            assert_eq!((shape.unit_rows, shape.waves), (rows, waves));
            assert_eq!(parsed.prefill_rows, 4096);
        }
        for extra in [
            vec!["--prefill-waves", "2"],
            vec!["--shared-only", "--prefill-units", "511"],
            vec!["--shared-only", "--prefill-units", "8192"],
            vec!["--shared-only", "--prefill-rows", "16384", "--prefill-units", "4096"],
            vec!["--shared-only", "--prefill-units", "2048", "--prefill-waves", "3"],
            vec!["--shared-only", "--prefill-units", "512", "--prefill-waves", "3"],
            vec!["--local-experts", "--split-device", "1", "--prefill-units", "2048"],
        ] {
            assert!(args(&extra)?.prefill_shape().is_err(), "{extra:?}");
        }
        let shape = args(&["--peers", "localhost:9870", "--prefill-rows", "2048", "--prefill-waves", "2"])?.prefill_shape()?.unwrap();
        assert_eq!((shape.unit_rows, shape.slot_rows()), (2048, 4096));
        let mut parsed = args(&["--peers", "localhost:9870", "--prefill-waves", "2"])?;
        let shape = parsed.prefill_shape()?.unwrap();
        assert_eq!((shape.unit_rows, shape.slot_rows()), (4096, 8192));
        parsed.prefill_rows = shape.slot_rows() as usize;
        parsed.prefill_units = Some(shape.unit_rows as usize);
        assert_eq!(parsed.prefill_shape()?, Some(shape));
        assert_eq!(parsed.prefill_wave_rows(), 4096);
        let worker = admission::spark_model(parsed.prefill_wave_rows(), false)?;
        assert_eq!((worker.capacity, worker.compiled_capacity), (4096, 4096));
        let cfg = Qwen4Config::from_hf(&cuteafd_loader::plan::testing::qwen4_config(48))?;
        let admission = admission::inputs(&parsed, &cfg, 48, false, None, Some((160, true)), 0)?;
        assert_eq!(admission.prefill_rows, 8192);
        assert_eq!(admission.prefill_shape, Some(shape));
        assert!(required_fp8_programs(&EngineArgs { fp8_decode: true, ..parsed })
            .iter().all(|name| !name.contains("m8192")));
        Ok(())
    }

    #[test]
    fn dual_selector_requires_resident_local_experts() {
        assert!(args(&["--split-device", "1"]).is_err());
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
        assert!(args(&["--rtx-expert-layers", "all"]).is_err());
        for policy in ["auto", "all", "max", "100%"] {
            assert!(args(&["--local-experts", "--split-device", "1", "--rtx-expert-layers", policy]).is_ok());
        }
    }

    #[test]
    fn measured_defaults_fp8_head_bf16_projections_and_options() {
        for extra in [&[][..], &["--mtp", "1"][..]] {
            let parsed = args(extra).unwrap();
            assert!(!parsed.fp8_decode && parsed.mtp_fp8_head && !parsed.fp8_prefill_w8a8);
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
            let required = required_fp8_programs(&parsed);
            let suffix = if format == "fp8" { "_kv_fp8" } else { "" };
            for cap in ["m64".to_string(), format!("m{}", parsed.prefill_rows)] {
                assert!(required.contains(&format!("qwen4_attn_producer_w8{suffix}_{cap}")));
                let other = if format == "fp8" { "" } else { "_kv_fp8" };
                assert!(!required.contains(&format!("qwen4_attn_producer_w8{other}_{cap}")));
                assert!(required.contains(&format!("qwen4_gdn_w8_{cap}")));
                assert!(required.contains(&format!("qwen4_attn_o_w8_{cap}")));
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
        let requested = cuteafd_loader::placement::attention::parse(&args.attention_placement).map_err(anyhow::Error::msg)?;
        let selected = if args.split_device.is_some() {
            ensure!(requested.is_none() || requested == Some(cuteafd_loader::placement::AttentionPlacement::Layers),
                "Qwen dual supports whole-layer attention only");
            Some(cuteafd_loader::placement::AttentionPlacement::Layers)
        } else { requested };
        cuteafd_loader::placement::families::QWEN4.check_attention(selected,
            1 + usize::from(args.split_device.is_some()), args.split_device.is_some())?;
        let automatic_context = args.max_context == 0;
        let mut context_args = args.clone();
        let prefill_shape = args.prefill_shape()?;
        if let Some(shape) = prefill_shape {
            // Explicit diagnostic cuts retain the requested whole-step span.
            if args.prefill_units.is_none() { context_args.prefill_rows = shape.slot_rows() as usize; }
            context_args.prefill_units = Some(shape.unit_rows as usize);
        }
        context_args.max_context = crate::shared::context::checkpoint_context(
            &args.snapshot, &args.manifest, "qwen4", args.max_context)?;
        let args = &context_args;
        cuteafd_loader::placement::PoolPolicy::resolve(&[], args.max_context as u64,
            (args.pool_tokens > 0).then_some(args.pool_tokens as u64), 256, false)
            .with_automatic_context(automatic_context).validate_explicit()?;
        if let Some(peer) = args.split_device {
            return self.with_dual_engine(args, peer, automatic_context, body);
        }
        let programs = self.library.programs()?.with_manifest(&args.manifest)?;
        programs.capacities().require_context("qwen4", args.max_context)?;
        let required = required_fp8_programs(args);
        for name in &required {
            programs.spec(name).with_context(|| format!("--fp8-decode / --mtp-fp8-head need the FP8-only program \
                {name} (export_b12x_dsv4_aot.py qwen4 with the fork's fp8_only Qwen programs); rebuild the native \
                library or run with checkpoint BF16 (--fp8-decode false --mtp-fp8-head false)"))?;
        }
        // The family's own programs (`qwen4_*`), sampled as the runtime inventory: the image also carries
        // every other family's modules, which Qwen never launches.
        let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: "qwen4", split_family: None };
        let device = self.library.cuda_get_device()?;
        crate::shared::inventory::RuntimeInventory::measure(&self.library, &[device], crate::shared::inventory::InventoryPhase::PreWeight, |_| {
            let (loaded, skipped) = programs.load_matching(|name| selected.contains(name))?;
            tracing::info!(loaded, skipped, "Qwen programs loaded");
            Ok(())
        })?;
        let layers = args.layers.unwrap_or(self.cfg.layers).min(self.cfg.layers);
        if let Some(peers) = args.peers.as_deref() {
            let peers = peers.split(',').map(str::parse)
                .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            let working = cuteafd_loader::placement::SparkWorkingSet {
                spark_layers: (0..layers).collect(), spark_ranks: peers.len(), prefill_lanes: 1,
                ..Default::default()
            };
            let source = args.experts_snapshot.as_deref().unwrap_or(&args.snapshot);
            crate::shared::placement::SparkAdmission::plan(source, cuteafd_core::ExpertGeometry::QWEN4,
                self.experts.as_ref(), &working, peers.len(), None,
                admission::spark_model(args.prefill_wave_rows(), self.checkpoint.spark_input_is_bf16())?)?.select(&peers)?;
        }
        let mut manifest_json: Option<serde_json::Value> = std::fs::read(&args.manifest).ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok());
        let stream = self.library.cuda_stream_create()?;
        let started = Instant::now();
        let loader = weights::Qwen4Loader { library: &self.library, checkpoint: &self.checkpoint,
            fp8_decode: args.fp8_decode, fp8_scales: args.fp8_scales, stream,
            dense: std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::mount(&self.library)) };
        let source = self.embed_source()?;
        let with_mtp = args.mtp > 0 && layers == self.cfg.layers;
        // The coordinator's dense records push in a group of this GPU alone when
        // the checkpoint is SparkNest's; records are planned from headers first,
        // and any failure loads them through the mount.
        let device = crate::shared::memory::device::Device { library: &self.library, id: self.library.cuda_get_device()? };
        let planned = (|| -> Result<_> {
            let load = std::rc::Rc::new(crate::shared::experts::rtx::dense::DenseLoad::push(&self.library));
            let planning = weights::Qwen4Loader { dense: load.clone(), ..loader.clone() };
            let model = {
                let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                planning.model(&self.cfg, layers, with_mtp, args.mtp_fp8_head)?
            };
            let records = load.sink.as_ref().context("push sink")?.take_all();
            let (landed, operands, staged, steps, largest, finishes) = records.summary();
            tracing::info!(landed, operands, staged, steps, largest_step_bytes = largest, finishes,
                "Qwen coordinator dense records planned");
            let dense = crate::shared::experts::rtx::dense::plan_records(&args.snapshot, records.clone(), &[device], 0)?;
            let push = crate::shared::experts::rtx::push::CoordinatorPush::prepare_alone(&args.snapshot, None, 1, None)?;
            Ok((push, model, records, dense))
        })().map_err(|error| tracing::warn!(%error, "Qwen coordinator dense push unavailable; mount")).ok();
        let (embedding, model) = match planned {
            Some((push, model, records, dense)) => {
                let split = crate::shared::memory::load_timing::Split::start();
                let (pushed, embedding) = push.receive_with(vec![device], None, 0, Some(Ok(dense)), ||
                    crate::shared::token_io::TokenEmbedding::load(&self.library, source, args.token_io.embed_placement,
                        || Ok(())));
                let pushed = pushed?;
                let (embedding, ()) = embedding?;
                if pushed.is_none() {
                    crate::shared::experts::rtx::dense::fill_records_by_mount(&self.library, &records, |_| stream)?;
                }
                split.log(if pushed.is_some() { "qwen4 dense push" } else { "qwen4 dense mount fill" });
                (embedding, model)
            }
            None => crate::shared::token_io::TokenEmbedding::load(&self.library, source, args.token_io.embed_placement,
                || { let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
                    loader.model(&self.cfg, layers, with_mtp, args.mtp_fp8_head) })?,
        };
        let attention_plan = attention_plan(args, model.mtp.is_some())?;
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
        tracing::info!(plan_rows = attention_plan.rows(), active_drafts = ?attention_plan.active_drafts(),
            "Qwen immutable attention reduction plan from loaded MTP");
        let resident: usize = model.layers.iter().map(weights::Qwen4Layer::bytes).sum::<usize>()
            + model.mtp.as_ref().map_or(0, weights::MtpWeights::bytes) + model.head.bytes();
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
                layer, &args.table, args.prefill_wave_rows())?),
            _ => None,
        };
        // Establish expert ownership before admission. EXL3 keeps its existing
        // lazy first-use load; reserve the exact loader plan before sizing KV.
        let mut future_expert_bytes = 0;
        let startup_admission = args.planner_graph_modes.is_some() && engine::startup_graphs_enabled(
            std::env::var("CUTEAFD_QWEN4_GRAPHS").ok().as_deref(),
            std::env::var("CUTEAFD_QWEN4_STARTUP_GRAPHS").ok().as_deref());
        let budget_admission = args.full_prefill_logits || args.pool_tokens == 0 || cuteafd_ffi::coordinator_gpu_budget().is_some() || startup_admission;
        let admitted_experts = if budget_admission {
            ensure!(args.shared_only || self.fp8().is_none() || args.expert_window.is_none(),
                "Qwen automatic KV admission does not support diagnostic --expert-window paging; use a fixed pool or Sparks");
            let experts = self.experts(args, layers, stream)?;
            if let Some(engine::Experts::LocalExl3(local)) = &experts {
                ensure!(local.window >= layers,
                    "Qwen automatic KV admission requires all EXL3 backbone experts resident; use --exl3-window at least {layers}, a fixed pool, or Sparks");
                let expected: Vec<_> = (0..local.window.min(layers)).collect();
                let plan = crate::families::deepseek_v4::local::plan(&self.library, &local.native_lib,
                    local.catalog, usize::from(local.mtp), &expected, local.max_rows, local.budget)?;
                ensure!(plan.layers == expected,
                    "Qwen automatic KV admission requires the complete requested EXL3 expert window to fit");
                future_expert_bytes = u64::try_from(plan.peak_bytes)?;
                tracing::info!(layers=?plan.layers, peak_bytes=plan.peak_bytes,
                    "Qwen planner reserved the lazy EXL3 expert window");
            }
            Some(experts)
        } else { None };
        let pool_tokens = if budget_admission {
            let source = admission::ExpertSource::selected(admitted_experts.as_ref().and_then(Option::as_ref))?;
            admission::pool_tokens(&self.library, args, &admission::inputs(args, &self.cfg, layers, model.mtp.is_some(),
                manifest_json.as_ref(), ple.as_ref().map(|p| (p.row_bytes as u64, p.fp8)),
                future_expert_bytes)?, source, automatic_context)?
        } else { args.pool_tokens };
        let max_context = crate::shared::context::pool_context("qwen4", args.max_context, automatic_context, pool_tokens, 256)?;
        let pages = pool_tokens.div_ceil(engine::PAGE_ROWS);
        let mut engine = engine::Qwen4Engine::new(&self.library, &programs, self.cfg.clone(), model, ple, stream,
            max_context, args.prefill_rows, pages, args.slots, args.kv_format, embedding)?;
        engine.prefill_shape = prefill_shape;
        engine.attention_plan = attention_plan;
        engine.spark_bf16 = self.checkpoint.spark_input_is_bf16();
        if args.planner_graph_modes.is_some() { engine.enable_startup_graphs(); }
        engine.w8a8_prefill = args.fp8_prefill_w8a8;
        if let Some(experts) = match admitted_experts { Some(experts) => experts, None => self.experts(args, layers, stream)? } {
            // Local EXL3 experts load their first window by push when the
            // checkpoint is SparkNest's (a group of this GPU alone), else lazily.
            if let engine::Experts::LocalExl3(local) = &experts {
                local.push_window(args.experts_snapshot.as_deref().unwrap_or(&args.snapshot), self.library.cuda_get_device()?)?;
            }
            engine.set_experts(experts);
        }
        if let Some(budget) = args.l2.budget(&self.library, crate::shared::l2_prefetch::OTHER_DEFAULT)? {
            engine.l2 = Some(crate::shared::l2_prefetch::L2Prefetch::new(&self.library, budget, &engine.decode_read_order())?);
        }
        engine.admit_host_memory(args)?;
        if args.full_prefill_logits { engine.prepare_scoring_prefill()?; }
        engine.prepare_prefill_units()?;
        let result = body(&engine);
        drop(engine);
        // SAFETY: the engine that used the stream is gone.
        unsafe { self.library.cuda_stream_destroy(stream)? };
        result
    }

    fn experts<'s>(&'s self, args: &EngineArgs, layers: usize, stream: *mut std::ffi::c_void)
        -> Result<Option<engine::Experts<'s>>> {
        if args.mtp > 0 && layers == self.cfg.layers
            && self.checkpoint.module_numerics("mtp.layers.0.mlp.experts.0.gate_proj")?
                == cuteafd_loader::plan::checkpoint::ModuleNumerics::W8A8 {
            tracing::warn!("Qwen MTP experts: deviation from declared W8A8, kernel pending; existing FP8 expert path retained");
        }
        if args.shared_only {
            tracing::warn!("--shared-only: routed experts are skipped (outputs do not match the model)");
            return Ok(Some(engine::Experts::SharedOnly));
        }
        if let Some(tensors) = self.fp8().filter(|_| args.local_experts) {
            let directory = args.fp8_package.clone()
                .unwrap_or_else(|| crate::shared::experts::fp8::package_directory(&args.native_lib, 1, tensors.format()));
            let (free, _) = self.library.cuda_memory_info()?;
            ensure!(args.expert_window != Some(0), "--expert-window must be at least 1");
            let mtp = args.mtp > 0 && layers == self.cfg.layers;
            let mixed_mtp = mtp && tensors.layer_format(layers)? != tensors.format();
            let resident = if args.expert_window.is_some() { 0..0 } else {
                0..layers + usize::from(mtp && !mixed_mtp)
            };
            let layer_ids: Vec<_> = resident.collect();
            let started = Instant::now();
            let experts = crate::shared::experts::fp8::Fp8Experts::load(&self.library, tensors, &directory, &layer_ids, 1, 0,
                args.prefill_rows, free.saturating_sub(args.expert_reserve_gib.saturating_mul(1 << 30)))
                .context("local routed experts must fit with step and prefix-cache reservations; use --peers for Sparks, \
                    or --expert-window N for diagnostic paging")?;
            // The backbone and draft each own exactly one weight representation.
            // Re-query free memory after the target allocation so the second
            // package admits both its weights and scratch against the remaining budget.
            let mtp_experts = if mixed_mtp {
                let draft = tensors.for_layer(layers)?;
                let directory = crate::shared::experts::fp8::package_directory(&args.native_lib, 1, draft.format());
                let (free, _) = self.library.cuda_memory_info()?;
                Some(crate::shared::experts::fp8::Fp8Experts::load(&self.library, &draft, &directory,
                    &[layers], 1, 0, args.prefill_rows,
                    free.saturating_sub(args.expert_reserve_gib.saturating_mul(1 << 30)))
                    .context("loading the separate resident Qwen MTP expert package")?)
            } else { None };
            let loads = experts.layers.len() + mtp_experts.as_ref().map_or(0, |e| e.layers.len());
            tracing::info!(layers = loads, window = ?args.expert_window, elapsed_ms = started.elapsed().as_millis() as u64,
                "Qwen routed experts resident on this GPU");
            return Ok(Some(engine::Experts::Local(engine::LocalExperts {
                library: &self.library, tensors, experts: std::cell::RefCell::new(experts), mtp_experts,
                window: args.expert_window, loads: std::cell::RefCell::new(loads),
            })));
        }
        if let Some(catalog) = self.experts.as_ref().filter(|c| args.local_experts && c.exl3().is_some()) {
            let (free, _) = self.library.cuda_memory_info()?;
            return Ok(Some(engine::Experts::LocalExl3(engine::LocalExl3 {
                library: &self.library, native_lib: args.native_lib.clone(), catalog,
                resident: std::cell::RefCell::new(None), window: args.exl3_window.max(1), layers,
                mtp: args.mtp > 0 && layers == self.cfg.layers,
                max_rows: args.prefill_rows,
                budget: free.saturating_sub(args.expert_reserve_gib << 30), loads: std::cell::RefCell::new(0),
            })));
        }
        let Some(peers) = args.peers.as_deref() else { return Ok(None) };
        let mtp = if args.mtp > 0 && layers == self.cfg.layers {
            Some(self.spark_mtp_experts(args, layers, stream)?)
        } else { None };
        let peers = peers.split(',').map(str::parse).collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
        let spark_world = cuteafd_transport::expert::SparkWorld::canonical(peers.len())?;
        let shape = args.prefill_shape()?;
        let mut transport = crate::shared::spark_intake::SparkLink::new_waves(&self.library, spark_world, &peers,
            u32::try_from(args.prefill_wave_rows())?, cuteafd_transport::TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(),
                timeout: std::time::Duration::from_secs(120), max_frame_bytes: 64 << 20 }, self.cfg.hidden * 2,
                shape.map_or(1, |s| s.waves as usize),
                u32::try_from(shape.map_or(args.prefill_rows, |s| s.unit_rows as usize).max(engine::DECODE_ROWS))?)?;
        transport.enable_terminal_ownership(self.library.clone())?;
        // Connect every rank and publish its installed expert grouping now, with
        // the widest wave this transport can carry: the draft planner prices the
        // workers before any serving step opens a session, and the sessions this
        // opens are the ones every step after it reuses.
        transport.bootstrap(&engine::spark_bootstrap_request(&self.cfg, args.prefill_wave_rows(),
            self.checkpoint.spark_input_is_bf16())?)?;
        crate::shared::memory_report::release_load_staging(&self.library);
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        Ok(Some(engine::Experts::Spark { transport: std::cell::RefCell::new(transport), runtime, mtp }))
    }

    fn spark_mtp_experts<'s>(&'s self, args: &EngineArgs, layer: usize, stream: *mut std::ffi::c_void)
        -> Result<engine::MtpExperts<'s>> {
        let (free, _) = self.library.cuda_memory_info()?;
        let budget = free.saturating_sub(args.expert_reserve_gib.saturating_mul(1 << 30));
        self.local_mtp_experts(args, layer, stream, budget)
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
    args.engine.full_prefill_logits |= args.nll;
    ensure!(args.step_rows > 0, "Qwen golden --step-rows must be nonzero");
    let windows = if args.golden.join("manifest.json").try_exists()? {
        ensure!(args.nll && args.save_logits.is_some(), "shared Qwen fidelity needs --nll --save-logits DIR");
        ensure!(args.prefill.is_none() && !args.teacher_force && args.spec_keep.is_none()
            && args.mtp_oracle.is_none() && args.token_check.is_none() && args.spec_decode.is_none()
            && args.resume_at.is_empty() && args.bench_prefill == 0 && args.bench_decode == 0,
            "shared Qwen fidelity cannot be combined with legacy golden diagnostics");
        let output = args.save_logits.as_deref().unwrap();
        ensure!(!output.try_exists()? && output.parent().is_some_and(std::path::Path::is_dir),
            "Qwen fidelity output needs an existing parent and a new directory");
        Some(fidelity_windows(&args.golden, args.window.as_deref())?)
    } else {
        ensure!(args.window.is_none(), "--window needs a published shared fidelity set");
        None
    };
    let opened = open(&args.engine)?;
    opened.with_engine(&args.engine, |engine| {
        if let Some(windows) = &windows { return score_fidelity(&args, &opened, engine, windows); }
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

fn fidelity_windows(root: &std::path::Path, selected: Option<&str>) -> Result<Vec<cuteafd_bench::reference::Window>> {
    use cuteafd_bench::fidelity_dataset;
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(root.join("manifest.json"))?;
    ensure!(format!("{:x}", Sha256::digest(&bytes)) == fidelity_dataset::QWEN_MANIFEST_SHA256,
        "Qwen fidelity manifest differs from immutable publication");
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == fidelity_dataset::QWEN_CONFIG && manifest["vocab"] == 248320,
        "Qwen fidelity config/vocabulary differs");
    let reference = fidelity_dataset::load(root, &manifest)?;
    let mut windows = fidelity_dataset::quick_windows(&reference)?;
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
        for chunk in fidelity_chunks(window.tokens.len(), engine.prefill_rows)? {
            let logits = engine.prefill_forced(&mut placement, &window.tokens[chunk.clone()], None, None, chunk.len())?
                .context("Qwen fidelity needs all target logits")?;
            ensure!(logits.len() == chunk.len() * vocab, "Qwen fidelity logits shape differs");
            for (row, logits) in logits.chunks_exact(vocab).enumerate() {
                let position = chunk.start + row + 1;
                if position >= window.score_from { probe.row(position, logits); }
            }
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

fn golden_logits(root: &std::path::Path, dump_only: bool, expected: usize) -> Result<Option<Vec<f32>>> {
    let bytes = match std::fs::read(root.join("logits.bin")) {
        Ok(bytes) => bytes,
        Err(error) if dump_only && error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("Qwen golden logits; pass --save-logits for a pure engine dump"),
    };
    ensure!(bytes.len() == expected.checked_mul(4).context("Qwen golden logits extent overflow")?,
        "Qwen golden logits shape differs");
    let values = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "Qwen golden logits are non-finite");
    Ok(Some(values))
}

fn golden_run(args: &GoldenArgs, opened: &Opened, engine: &engine::Qwen4Engine<'_>) -> Result<()> {
    let cfg = &opened.cfg;
    let layers = engine.weights.layers.len();
    let token_bytes = std::fs::read(args.golden.join("tokens.bin"))?;
    ensure!(!token_bytes.is_empty() && token_bytes.len().is_multiple_of(4), "Qwen golden token extent differs");
    let tokens: Vec<u32> = token_bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    let reference_logits = golden_logits(&args.golden, args.save_logits.is_some(),
        tokens.len().checked_mul(cfg.vocab_size).context("Qwen golden logits shape overflow")?)?;
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
    let mut logits: Option<Vec<f32>> = None;
    let mut done = 0;
    while done < prefill {
        let n = engine.prefill_rows.min(prefill - done);
        let first = done;
        let chunk = if engine.prefill_shape.is_some() {
            ensure!(!args.teacher_force, "Qwen unit golden compares logits, not teacher-forced layer streams");
            engine.prefill_forced(&mut placement, &tokens[done..done + n], None, None, if args.nll { n } else { 1 })?
        } else {
            engine.prefill_forced(&mut placement, &tokens[done..done + n],
                Some(&mut |layer, streams| compare(layer, first, streams, &mut worst)),
                args.teacher_force.then_some(&forced as &dyn Fn(usize) -> Option<Vec<u8>>), if args.nll { n } else { 1 })?
        };
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
                let (agree, next_ok, golden_next, nll, scored) = score(&decode_logits, golden, &tokens, prefill, vocab);
                let rows = decode_logits.len() / vocab;
                println!("decode logits: top-1 agreement {:.1}% over {rows} rows | next-token accuracy engine {:.1}% \
                    golden {:.1}% | mean NLL {:.4} | mean KL(golden||engine) {:.5}", 100.0 * agree as f64 / rows as f64,
                    100.0 * next_ok as f64 / scored.max(1) as f64, 100.0 * golden_next as f64 / scored.max(1) as f64,
                    nll / scored.max(1) as f64, crate::families::glm5_flash::mean_kl(&decode_logits, golden, prefill, vocab));
            }
        }
    }
    // Reference dumps for `scripts/bench/golden-run.py --compare-logits`: with --nll every
    // prefill row, otherwise the decode/verify rows past --prefill (tokens.bin from --prefill).
    if let Some(dir) = &args.save_logits {
        std::fs::create_dir_all(dir)?;
        if args.nll {
            let rows = logits.as_deref().context("--save-logits with --nll produced no logits")?;
            ensure!(rows.len() == prefill * vocab, "saved logits cover {} of {prefill} prefill rows",
                rows.len() / vocab);
            std::fs::write(dir.join("tokens.bin"), tokens[..prefill].iter()
                .flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>())?;
            std::fs::write(dir.join("logits.bin"), rows.iter()
                .flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
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
        Some(engine::Experts::Local(local)) => format!(", {} FP8 expert layer loads", local.loads.borrow()),
        Some(engine::Experts::LocalExl3(local)) => format!(", {} EXL3 expert layer loads", local.loads.borrow()),
        _ => String::new(),
    };
    println!("prefill: {prefill} tokens through {layers} layers in {prefill_seconds:.2} s{loads}");
    if let Some(logits) = logits {
        // Match the decode dump for exact before/after prefill qualification.
        if let Ok(path) = std::env::var("CUTEAFD_DUMP_PREFILL_LOGITS") {
            std::fs::write(&path, logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        let Some(golden) = &reference_logits else { return Ok(()) };
        if args.nll {
            let (agree, next_ok, golden_next, nll, scored) = score(&logits, golden, &tokens, 0, vocab);
            let (_, _, _, golden_nll, _) = score(&golden[..prefill * vocab], &golden, &tokens, 0, vocab);
            println!("prefill logits: top-1 agreement {:.1}% over {prefill} rows | next-token accuracy engine {:.1}% \
                golden {:.1}% | mean NLL engine {:.4} golden {:.4}", 100.0 * agree as f64 / prefill as f64,
                100.0 * next_ok as f64 / scored.max(1) as f64, 100.0 * golden_next as f64 / scored.max(1) as f64,
                nll / scored.max(1) as f64, golden_nll / scored.max(1) as f64);
        }
        let logits = &logits[logits.len() - vocab..];
        let last = &golden[(prefill - 1) * vocab..][..vocab];
        let (cosine, _) = similarity(logits, last);
        let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i);
        println!("last-row logits: argmax engine {} golden {} cosine {cosine:.6}", argmax(logits), argmax(last));
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
        assert!(golden_logits(dir.path(), true, 4).unwrap().is_none());
        assert!(golden_logits(dir.path(), false, 4).is_err());
        let values = [0.0f32, 1.0, -1.0, 2.0];
        let path = dir.path().join("logits.bin");
        std::fs::write(&path, values.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>()).unwrap();
        assert_eq!(golden_logits(dir.path(), true, 4).unwrap().unwrap(), values);
        assert!(golden_logits(dir.path(), true, 3).is_err());
        std::fs::write(&path, f32::NAN.to_le_bytes()).unwrap();
        assert!(golden_logits(dir.path(), true, 1).is_err());
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();
        assert!(golden_logits(dir.path(), true, 1).is_err());
    }

    #[test]
    fn shared_fidelity_rejects_an_unpublished_manifest() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("manifest.json"), b"{}").unwrap();
        assert!(fidelity_windows(dir.path(), None).unwrap_err().to_string().contains("immutable publication"));
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
