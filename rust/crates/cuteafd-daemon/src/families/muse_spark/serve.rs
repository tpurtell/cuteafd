//! Muse Spark serving entry: validated arguments, program preflight, the
//! measured pre-weight inventory and shared admission. The execution engine
//! (Backend, PrefillHooks, prefix cache, DFlash2) is not built yet, so a
//! launch stops after admission with the admitted placement and allocates
//! nothing: unsupported is a result, not a crash.
use super::startup;
use crate::shared::inventory::{InventoryPhase, RuntimeInventory};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::muse_spark::{core::CoreInventory, MuseSparkConfig};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] pub manifest: PathBuf,
    /// Coordinator GPUs: one RTX, or a head-split pair (attention heads halve).
    #[arg(long, value_delimiter = ',', default_values_t = [0, 1])] pub devices: Vec<i32>,
    /// Spark expert ranks; whole experts are partitioned across them (expert parallel).
    #[arg(long, value_delimiter = ',')] pub peers: Vec<SocketAddr>,
    /// Auto admission grows beyond the selected context floor while memory allows.
    #[arg(long)] pub pool_tokens: Option<u64>,
    /// API ceiling and compiled extent; unset takes the checkpoint's 1M with the
    /// pool deciding the admitted context.
    #[arg(long)] pub max_context: Option<u64>,
    #[arg(long, default_value = "auto")] pub rtx_expert_layers: cuteafd_loader::placement::Onboard,
    /// Complete per-rank Spark allocation ceiling, estimated until the first worker census.
    #[arg(long)] pub spark_budget_gib: Option<f64>,
    #[arg(long, default_value = "0.0.0.0:8000")] pub listen: String,
    #[arg(long, default_value_t = 32768)] pub max_output: u32,
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u8).range(1..=64))]
    pub max_sequences: u8,
    #[arg(long)] pub model_id: Option<String>,
    /// Prefill unit rows, at most 4096; unset follows the shared wave policy.
    #[arg(long)] pub prefill_rows: Option<usize>,
    /// Prefill units in flight on the one Spark transport; unset follows the shared policy.
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub prefill_waves: Option<u8>,
    /// Spark expert partial replies: a request contributing at most this many rows to a
    /// step replies BF16, wider segments NVFP4 (the shared EXL3 worker's reply cutoff).
    /// Must cover the widest verify step; the default keeps decode and verify lossless.
    #[arg(long, default_value_t = cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS)]
    pub expert_partial_bf16_max_rows: usize,
    /// Partial format of segments above the cutoff: nvfp4 (E2M1, E4M3 per 16) or fp8 (E4M3, UE8M0 per 32).
    #[arg(long, default_value = "nvfp4")]
    pub expert_partial_wide_format: cuteafd_loader::families::muse_spark::replies::WideFormat,
    #[command(flatten)] pub prefix: crate::shared::prefix::PrefixArgs,
}

impl ServeArgs {
    pub(super) fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(self.peers.len(), cuteafd_loader::plan::Family::prefill_limits(
            &cuteafd_loader::plan::families::muse_spark::MUSE_SPARK), PrefillOverrides {
            unit_rows: self.prefill_rows.map(|rows| rows as u64),
            waves: self.prefill_waves.map(u64::from), span_rows: None, reserve_rows: None,
        })?)
    }

    fn spark_budget_bytes(&self) -> u64 {
        self.spark_budget_gib.map_or(cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES,
            |gib| (gib * (1u64 << 30) as f64) as u64)
    }

    pub(super) fn validate(&self) -> Result<()> {
        cuteafd_loader::placement::families::muse_spark::executor(self.devices.len())?;
        ensure!(self.devices.iter().all(|&device| device >= 0)
            && self.devices.iter().collect::<std::collections::HashSet<_>>().len() == self.devices.len(),
            "Muse Spark coordinator devices must be distinct");
        let distinct = self.peers.iter().collect::<std::collections::HashSet<_>>().len();
        ensure!([2, 3, 4, 6].contains(&self.peers.len()) && distinct == self.peers.len(),
            "Muse Spark serves its EXL3 TP experts on 2, 3, 4 or 6 distinct Spark endpoints");
        ensure!(self.max_context.is_none_or(|context| (1..=1 << 20).contains(&context)),
            "Muse Spark context exceeds the checkpoint's 1M extent");
        ensure!(self.spark_budget_gib.is_none_or(|gib| gib.is_finite() && gib > 0.0
            && gib < u64::MAX as f64 / (1u64 << 30) as f64),
            "Muse Spark requires a finite positive --spark-budget-gib");
        self.prefill()?;
        // No drafter yet: a step appends at most the 64-row rollback step.
        cuteafd_loader::families::muse_spark::replies::Cutoff::new(self.expert_partial_bf16_max_rows,
            cuteafd_loader::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS)?;
        ensure!(self.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Muse Spark serving requires exact prefix frontiers");
        Ok(())
    }
}

pub(crate) async fn run(args: ServeArgs) -> Result<()> {
    args.validate()?;
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::MUSE_SPARK)
            .map_err(|_| anyhow::anyhow!("Muse Spark serving needs an isolated process"))?;
        let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&args.snapshot, false, false)?;
        let cfg = MuseSparkConfig::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let core = CoreInventory::from_checkpoint(&cfg, &checkpoint)?;
        let gpus = args.devices.len();
        let manifest = startup::preflight(&args.manifest, gpus)?;
        let catalog = cuteafd_loader::read_expert_catalog(&args.snapshot)?;
        // SAFETY: every native owner drops on this worker before the library.
        let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib)? });
        let inventory = RuntimeInventory::measure(&library, &args.devices, InventoryPhase::PreWeight, |_| Ok(()))?;
        let (built, placement) = startup::admit(startup::Options { cfg: &cfg, core: &core, catalog: &catalog,
            manifest: &manifest,
            measured: inventory.gpus.iter().map(|sample| startup::Measured { total_bytes: sample.total_bytes,
                free_bytes: sample.free_bytes, untracked_bytes: sample.context_bytes }).collect(),
            headroom_bytes: 2 << 30, spark_ranks: args.peers.len(), prefill: args.prefill()?,
            slots: u64::from(args.max_sequences),
            mark_slots: args.prefix.mark_rule(usize::from(args.max_sequences))
                .slots(cuteafd_loader::placement::families::muse_spark::mark_bytes(&cfg, 1) as usize) as u64,
            max_context: args.max_context,
            pool_tokens: args.pool_tokens.filter(|&tokens| tokens > 0), onboard: args.rtx_expert_layers,
            spark_budget_bytes: args.spark_budget_bytes(),
            partial_bf16_max_rows: args.expert_partial_bf16_max_rows,
            partial_wide_format: args.expert_partial_wide_format })?;
        tracing::info!(placement = %placement.summary(), rtx_expert_layers = placement.onboard_layers,
            pool_tokens = placement.pool_tokens, k6_programs = built.projections.iter().map(|r| r.programs.len()).sum::<usize>(),
            "Muse Spark admitted TP experts with coordinator-combined partials");
        anyhow::bail!("Muse Spark execution engine is not built yet: admitted {} (pool {} tokens, {} RTX expert \
            layers); nothing was allocated", placement.summary(), placement.pool_tokens, placement.onboard_layers)
    });
    worker.await.context("Muse Spark startup worker")?
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    const PEERS: &str = "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4,127.0.0.1:5,127.0.0.1:6";

    fn args(extra: &[&str]) -> Result<ServeArgs, clap::Error> {
        let mut argv = vec!["cuteafd", "serve-muse", "--snapshot", "/checkpoint", "--native-lib", "/native.so"];
        argv.extend_from_slice(extra);
        let cli = crate::cli::Cli::try_parse_from(argv)?;
        let crate::cli::Commands::ServeMuse(args) = cli.command else { panic!("Muse command") };
        Ok(args)
    }

    #[test]
    fn defaults_are_a_head_split_pair_with_automatic_context_and_onboarding() -> Result<()> {
        let defaults = args(&["--peers", PEERS])?;
        defaults.validate()?;
        assert_eq!(defaults.devices, [0, 1]);
        assert_eq!(defaults.max_context, None);
        assert_eq!(defaults.pool_tokens, None);
        assert_eq!(defaults.rtx_expert_layers, cuteafd_loader::placement::Onboard::Auto);
        assert_eq!(defaults.max_sequences, 16);
        assert_eq!(defaults.spark_budget_bytes(), cuteafd_loader::plan::DEFAULT_SPARK_BUDGET_BYTES);
        assert_eq!(defaults.expert_partial_bf16_max_rows, 64);
        assert_eq!(defaults.expert_partial_wide_format, cuteafd_loader::families::muse_spark::replies::WideFormat::Nvfp4);
        assert!(args(&["--peers", PEERS, "--expert-partial-wide-format", "fp8"]).is_ok());
        assert!(args(&["--peers", PEERS, "--expert-partial-wide-format", "bf16"]).is_err());
        args(&["--peers", PEERS, "--expert-partial-bf16-max-rows", "4096"])?.validate()?;
        let single = args(&["--peers", PEERS, "--devices", "1"])?;
        single.validate()?;
        assert_eq!(single.devices, [1]);
        for peers in ["127.0.0.1:1,127.0.0.1:2,127.0.0.1:3,127.0.0.1:4", PEERS] {
            args(&["--peers", peers])?.validate()?;
        }
        Ok(())
    }

    #[test]
    fn prefill_shape_is_the_planners() -> Result<()> {
        let shape = args(&["--peers", PEERS])?.prefill()?;
        let planned = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(6,
            cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::muse_spark::MUSE_SPARK),
            Default::default())?;
        assert_eq!(shape, planned);
        assert_eq!(args(&["--peers", PEERS, "--prefill-waves", "2"])?.prefill()?.waves, 2);
        Ok(())
    }

    #[test]
    fn unsupported_layouts_refuse_before_any_device_access() -> Result<()> {
        assert!(args(&[])?.validate().is_err(), "experts need Spark ranks");
        let nine = (1..=9).map(|n| format!("127.0.0.1:{n}")).collect::<Vec<_>>().join(",");
        for extra in [vec!["--peers", nine.as_str()], vec!["--peers", "127.0.0.1:1,127.0.0.1:1"],
            vec!["--peers", PEERS, "--devices", "0,1,2"], vec!["--peers", PEERS, "--devices", "0,0"],
            vec!["--peers", PEERS, "--max-context", "1048577"], vec!["--peers", PEERS, "--max-context", "0"],
            vec!["--peers", PEERS, "--spark-budget-gib", "0"], vec!["--peers", PEERS, "--spark-budget-gib", "NaN"],
            vec!["--peers", PEERS, "--prefill-rows", "4097"], vec!["--peers", PEERS, "--prefix-partial", "on"],
            vec!["--peers", PEERS, "--expert-partial-bf16-max-rows", "8"]] {
            assert!(args(&extra)?.validate().is_err(), "{extra:?}");
        }
        assert!(args(&["--peers", PEERS, "--max-sequences", "65"]).is_err());
        Ok(())
    }
}
