use clap::{Args, Parser, Subcommand};
use cuteafd_core::{DEFAULT_MODEL_ID, DS4_FLASH_HIDDEN_SIZE};
use std::path::PathBuf;

pub(crate) const DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS: usize = 128 * 1024;

/// Inspect original argv: clap normalizes aliases before exposing matches.
pub(crate) fn deprecated_budget_flags(args: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<&'static str> {
    let mut replacements = Vec::new();
    for arg in args {
        let replacement = match arg.as_ref().split('=').next().unwrap_or("") {
            "--rtx-budget-gib" | "--rtx-gib" => Some("--coordinator-gpu-budget-gib"),
            "--coordinator-budget-gib" => Some("--coordinator-weight-budget-gib"),
            _ => None,
        };
        if let Some(replacement) = replacement {
            if !replacements.contains(&replacement) { replacements.push(replacement); }
        }
    }
    replacements
}

#[derive(Debug, Parser)]
#[command(name = "cuteafd", about = "CUTEAFD phase0 runtime CLI")]
pub(crate) struct Cli {
    /// Buffered mapped-table gathers (engram / PLE); mmap until qualified.
    #[arg(long, global = true, env = "CUTEAFD_TABLE_BACKEND", value_parser = ["uring", "mmap", "mincore-routed"])]
    pub(crate) table_backend: Option<String>,
    #[arg(long, global = true, env = "VISION")]
    pub(crate) vision: Option<cuteafd_loader::plan::MediaMode>,
    #[arg(long, global = true, env = "AUDIO")]
    pub(crate) audio: Option<cuteafd_loader::plan::MediaMode>,
    /// Generic-family per-image LM token cap (detail=low also caps at 256).
    #[arg(long, global = true, value_parser = clap::value_parser!(u32).range(1..=16384))]
    pub(crate) max_image_tokens: Option<u32>,
    /// Remote image URL policy; inline data URLs work in every mode.
    #[arg(long, global = true)]
    pub(crate) image_url_fetch: Option<cuteafd_api::openai::media::ImageUrlFetch>,
    /// Logical GiB ceiling per coordinator GPU (weights, KV, workspaces,
    /// graphs and drafts); leaves physical GPU capacity/SM/L2 unchanged.
    #[arg(long, global = true, aliases = ["rtx-budget-gib", "rtx-gib"])]
    pub(crate) coordinator_gpu_budget_gib: Option<f64>,
    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
    /// Serve Anthropic Messages, OpenAI Responses and Realtime through an HTTP upstream (no GPUs).
    Gateway(crate::commands::gateway::GatewayArgs),
    /// Usage history: clear either tier, export metadata as CSV, or serve a synthetic demo.
    Usage(crate::commands::usage::UsageArgs),
    Doctor(DoctorArgs),
    /// Describe a checkpoint: family, placement, formats, and what this build lacks.
    Plan(PlanArgs),
    /// Report RDMA ports, link and PCIe rates, subnets and the rail plan.
    Fabric(FabricArgs),
    /// Check live Spark expert ranks for one layer against a CPU oracle.
    ExpertProbe(ExpertProbeArgs),
    /// Serve a checkpoint through the OpenAI API; the family comes from
    /// --snapshot's config.json (or --family). `serve --family ID --help` lists
    /// that family's options.
    Serve(crate::commands::family::FamilyArgs),
    /// Compare a family's coordinator layer by layer with its golden reference
    /// outputs (python/reference/families/<id>); --family or --snapshot picks it.
    Golden(crate::commands::family::FamilyArgs),
    /// Score a DeepSeek V4.1 golden prompt teacher-forced through the serving
    /// worker's scoring probe and compare its logits with golden.py's.
    #[command(hide = true)]
    V41Golden(crate::families::deepseek_v41::v41_golden::GoldenArgs),
    /// Prefill a DeepSeek V4 golden prompt and compare layers and logits.
    #[command(hide = true)]
    Dsv4Golden(crate::families::deepseek_v4::GoldenArgs),
    /// Run GLM 5.x layers through the exported programs and compare with golden.py outputs.
    #[command(hide = true)]
    GlmGolden(crate::families::glm5::GoldenArgs),
    /// Compare the MiMo V2 coordinator programs layer by layer with python/reference/families/mimo_v2/mimo_v2/golden.py outputs.
    #[command(hide = true)]
    MimoGolden(crate::families::mimo_v2::GoldenArgs),
    /// Serve MiMo V2 Flash (mimo_v2) through the OpenAI API with Spark FP8 experts.
    #[command(hide = true)]
    ServeMimo(crate::families::mimo_v2::serve::ServeArgs),
    /// Compare the GLM 5.3 Flash coordinator programs layer by layer with python/reference/families/glm5_flash/golden.py outputs.
    #[command(hide = true)]
    GlmfGolden(crate::families::glm5_flash::GoldenArgs),
    /// Compare the Qwen 3.8 Flash Next (qwen4_exp) engine layer by layer with
    /// python/reference/families/qwen4/golden.py outputs.
    #[command(hide = true)]
    Qwen4Golden(crate::families::qwen4::GoldenArgs),
    /// Serve Qwen 3.8 Flash Next (qwen4_exp) through the OpenAI API on the qwen4 engine.
    #[command(hide = true)]
    ServeQwen4(crate::families::qwen4::serve::ServeArgs),
    /// Serve GLM 5.3 Flash (glm5_next) through the OpenAI API on the glmf engine.
    #[command(hide = true)]
    ServeGlmf(crate::families::glm5_flash::serve::ServeArgs),
    /// Serve a GLM 5.x checkpoint (OpenAI-compatible API) over the glm_* programs and Spark experts.
    #[command(hide = true)]
    ServeGlm(crate::families::glm5::serve::ServeArgs),
    /// Serve a DeepSeek V4 checkpoint (OpenAI API) with Spark experts.
    #[command(hide = true)]
    ServeDsv4(crate::families::deepseek_v4::serve::ServeArgs),
    /// Serve routed experts on a Spark rank over RoCE (every family).
    #[command(alias = "expertd-native")]
    Expertd(NativeExpertDaemonArgs),
    /// Serve the official V4.1 target text path.
    #[command(hide = true)]
    ServeNative(NativeServeArgs),
    /// Benchmark a running server (its own in-server runner), publish
    /// results, or run the release smoke matrix.
    Bench(crate::commands::bench::BenchArgs),
    BenchRdma(BenchRdmaArgs),
    BenchRdmaRing(BenchRdmaRingArgs),
    TransportCapabilities(TransportCapabilitiesArgs),
}

#[derive(Debug, Args)]
pub(crate) struct ExpertProbeArgs {
    /// Checkpoint snapshot directory the Spark ranks serve.
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    /// Spark ranks in TP order, comma-separated HOST:PORT.
    #[arg(long, required_unless_present = "local")]
    pub(crate) peers: Option<String>,
    /// Run the experts on this GPU through the coordinator's resident-layer
    /// path (DeepSeek V4 `LocalExperts`) instead of the Sparks.
    #[arg(long, requires = "native_lib")]
    pub(crate) local: bool,
    /// Native library whose coordinator kernels/packages serve `--local`.
    #[arg(long)]
    pub(crate) native_lib: Option<PathBuf>,
    /// With `--local`, probe dSpark stage S (`mtp.S`) instead of `--layer`.
    #[arg(long, requires = "local")]
    pub(crate) stage: Option<usize>,
    #[arg(long, default_value_t = 3)]
    pub(crate) layer: usize,
    #[arg(long, default_value_t = 16)]
    pub(crate) rows: u32,
    /// Transport capacity; must not exceed the ranks' --capacity.
    #[arg(long, default_value_t = 4096)]
    pub(crate) capacity: u32,
    #[arg(long, default_value_t = 20260929)]
    pub(crate) seed: u64,
    /// Time this many more round trips after the checked one (median, min).
    #[arg(long, default_value_t = 0)]
    pub(crate) repeat: usize,
    /// With `--local` on an FP8 checkpoint: the package layout directory
    /// (default `<libdir>/fp8/fp8-<family>/tp<local-tp>`).
    #[arg(long)]
    pub(crate) fp8_package: Option<PathBuf>,
    /// With `--local` on an FP8 checkpoint: run every rank slice of TP degree
    /// N on this GPU in turn (a Spark package's tp2/tp4 layout) and sum the
    /// BF16 rank partials in FP32, as the coordinator does.
    #[arg(long, default_value_t = 1, requires = "local")]
    pub(crate) local_tp: usize,
    /// Compare and time coordinator intakes of the ranks' partials, e.g.
    /// `host,pinned,gpu` (bit-identical planes and sums; `--repeat` waves each).
    #[arg(long, requires = "native_lib")]
    pub(crate) intake: Option<String>,
    /// With `--intake`: run each mode's transport on its own lane thread (as
    /// GLM 5.3 prefill does) instead of inline.
    #[arg(long, requires = "intake")]
    pub(crate) intake_lane: bool,
    /// With `--local --local-tp N` on an EXL3 checkpoint: the Spark-role EXL3
    /// package root (`tp<N>-rank<R>/m<capacity>` layouts, e.g. a `--loopback`
    /// build) whose rank slices run on this GPU in turn, BF16 partials summed.
    #[arg(long, requires = "local")]
    pub(crate) exl3_package: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct FabricArgs {
    #[arg(long)]
    pub(crate) json: bool,
    /// Native library for the GPU landing probe (default `$CUTEAFD_NATIVE_LIB`;
    /// without one the probe is skipped).
    #[arg(long)]
    pub(crate) native_lib: Option<PathBuf>,
    /// CUDA device the probe lands in.
    #[arg(long, default_value_t = 0)]
    pub(crate) device: i32,
    /// Also measure GPU<->GPU peer transfers (copy engine, SM pull/push,
    /// pinned-host bounce; one-way, hops and two-way exchanges, eager and
    /// graph-captured) between the two `--p2p-devices`, idle and under
    /// host->GPU ingress (a proxy for NIC->GPU traffic on the same links).
    #[arg(long)]
    pub(crate) p2p: bool,
    #[arg(long, value_delimiter = ',', default_values_t = [0, 1])]
    pub(crate) p2p_devices: Vec<i32>,
    /// Transfer sizes in bytes (default: one 6144-wide BF16 row, 8 rows,
    /// 1 MiB and a 4096-row prefill chunk).
    #[arg(long, value_delimiter = ',', default_values_t = [12288usize, 98304, 1 << 20, 50331648])]
    pub(crate) p2p_bytes: Vec<usize>,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct PlanArgs {
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=6))]
    pub(crate) vision_replicas: u32,
    /// Vision policy (auto does not imply an implemented encoder).
    #[arg(skip = cuteafd_loader::plan::MediaMode::Auto)]
    pub(crate) vision: cuteafd_loader::plan::MediaMode,
    /// Auto enables qualified bundled audio towers, otherwise off.
    #[arg(skip = cuteafd_loader::plan::MediaMode::Auto)]
    pub(crate) audio: cuteafd_loader::plan::MediaMode,
    /// Override automatic placement of the untied token embedding.
    #[arg(long, alias = "embed-placement", value_enum)]
    pub(crate) embedding_placement: Option<crate::shared::token_io::EmbedPlacement>,
    /// Hugging Face model id (resolved under HF_HOME) or a snapshot directory.
    pub(crate) model: String,
    #[arg(long)]
    pub(crate) revision: Option<String>,
    #[arg(long)]
    pub(crate) hf_home: Option<PathBuf>,
    /// Spark ranks sharing the routed experts: 2, 3, 4 or 6, or 0 to place
    /// every routed expert on the coordinator GPU (local-only). Unset prefers
    /// qualified local experts when serving reservations fit; otherwise 4.
    #[arg(long)]
    pub(crate) spark_ranks: Option<usize>,
    /// Routed-expert weight budget per Spark rank, GiB.
    #[arg(long, default_value_t = 100.0)]
    pub(crate) spark_budget_gib: f64,
    /// Weight budget of the coordinator GPU, GiB (its own tensors, plus every
    /// routed expert with --spark-ranks 0).
    #[arg(long, alias = "coordinator-budget-gib", default_value_t = 80.0)]
    pub(crate) coordinator_weight_budget_gib: f64,
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
    /// Emit snapshot-relative paths suitable for rsync --files-from.
    #[arg(long)]
    pub(crate) files: bool,
    /// Role whose headers/files to inspect (coordinator, sparkN, vision, audio, drafter).
    #[arg(long)]
    pub(crate) role: Option<String>,
    /// Select a host from --file-layout, or the destination host for --fetch.
    #[arg(long)]
    pub(crate) host: Option<String>,
    /// JSON host-to-role map, e.g. {"worker":["spark0","vision"]}.
    #[arg(long)]
    pub(crate) file_layout: Option<PathBuf>,
    /// Maximum concurrent transfers when fetching the whole host layout.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=16))]
    pub(crate) fetch_parallel: u32,
    /// Copy this role's missing files with rdmasync or rsync; never delete.
    #[arg(long, requires = "destination")]
    pub(crate) fetch: bool,
    /// Transfer source: a local snapshot path or HOST:/snapshot. MODEL supplies
    /// the local index/config used to inventory role requirements.
    #[arg(long, requires = "fetch")]
    pub(crate) source: Option<String>,
    /// Forward the SSH agent to a source peer for peer-to-target copies.
    #[arg(long, requires = "fetch")]
    pub(crate) forward_agent: bool,
    /// Snapshot destination directory. Plain files are valid HF snapshot entries.
    #[arg(long)]
    pub(crate) destination: Option<PathBuf>,
    /// Print the transfer command and bytes without copying.
    #[arg(long, requires = "fetch")]
    pub(crate) dry_run: bool,
    /// Replace matching-size files too.
    #[arg(long, requires = "fetch")]
    pub(crate) force: bool,
    /// Serving KEY=VALUE config whose speculator and media settings to inventory.
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,
    /// Speculator to inventory (auto selects the family's release default).
    #[arg(long, default_value = "auto", value_parser = ["auto", "off", "mtp", "dflash2", "dspark"])]
    pub(crate) speculator: String,
    /// Explicitly include speculator inputs, even when the config disables them.
    #[arg(long, conflicts_with = "no_speculator")]
    pub(crate) include_speculator: bool,
    /// Inventory a slice served with speculation disabled.
    #[arg(long, conflicts_with_all = ["include_speculator", "drafter_snapshot"])]
    pub(crate) no_speculator: bool,
    /// Separately configured drafter checkpoint to include in JSON inventory.
    #[arg(long)]
    pub(crate) drafter_snapshot: Option<PathBuf>,
    /// Separately configured vision checkpoint to include in JSON inventory.
    #[arg(long)]
    pub(crate) vision_snapshot: Option<PathBuf>,
    /// Separately configured audio checkpoint to include in JSON inventory.
    #[arg(long)]
    pub(crate) audio_snapshot: Option<PathBuf>,
    /// Exit non-zero unless every part is servable.
    #[arg(long, default_value_t = false)]
    pub(crate) require_ready: bool,
    /// Lay out each device's memory: weights by group and format, KV pool,
    /// workspaces, runtime and Spark buffers.
    #[arg(long)]
    pub(crate) layout: bool,
    /// Coordinator GPUs for --layout (1 or 2; Qwen currently uses only the first).
    #[arg(long, default_value_t = 1)]
    pub(crate) rtx: usize,
    /// Attention placement: auto is the qualified family default (heads until decided).
    #[arg(long, default_value = "auto", value_parser = ["auto", "context", "layers", "heads"])]
    pub(crate) attention_placement: String,
    /// Resolved global logical GPU ceiling; PRO defaults to its CUDA total (94.97 GiB).
    #[arg(skip)]
    pub(crate) coordinator_gpu_budget_gib: Option<f64>,
    /// Explicit KV pool tokens for --layout (0 or omitted: automatic).
    #[arg(long)]
    pub(crate) pool_tokens: Option<u64>,
    /// External drafter GiB on the last GPU for --layout (DFlash).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) drafter_gib: f64,
    /// Routed backbone expert layers resident on RTX (DeepSeek).
    #[arg(long)]
    pub(crate) local_expert_layers: Option<usize>,
    /// Families on the shared placement solver (V4): RTX-resident routed
    /// layers, `auto` (pool first), `max` (experts first above a 262K pool),
    /// `N`, `N%` or `all` (the KV pool is then the output). Overrides
    /// --local-expert-layers.
    #[arg(long, value_parser = |text: &str| text.parse::<cuteafd_loader::placement::Onboard>())]
    pub(crate) rtx_expert_layers: Option<cuteafd_loader::placement::Onboard>,
    /// Removed whole-layer GPU1 placement flag.
    #[arg(long = "peer-expert-ranges", hide = true, value_parser = reject_peer_expert_ranges)]
    pub(crate) deprecated_peer_expert_ranges: bool,
    /// V4: force FP32 FFN exchange for prefill as well as decode.
    #[arg(long)]
    pub(crate) exchange_f32: bool,
    /// Compiled maximum context for table and workspace reservations (0: family/image default).
    #[arg(long, default_value_t = 0)]
    pub(crate) context_tokens: u64,
    /// Prefill workspace capacity for --layout (0: family/image default; GLM 5.3 Flash: per lane).
    #[arg(long, default_value_t = 0)]
    pub(crate) prefill_rows: u64,
    /// Include the diagnostic all-row prefill head reservation in --layout.
    #[arg(long)]
    pub(crate) full_prefill_logits: bool,
    /// Prefill lanes for --layout (GLM 5.3 Flash; 0: the family default).
    #[arg(long, default_value_t = 0)]
    pub(crate) prefill_lanes: u64,
    /// Decode and verify step rows for --layout (GLM 5.3 Flash's --decode-rows: 64, or 128 with the
    /// wide programs on one GPU): its decode workspace, token selector and replay records.
    #[arg(long, default_value_t = 64, value_parser = plan_decode_rows)]
    pub(crate) decode_rows: u64,
    /// GLM 5.3 Flash's DSA index storage for --layout (compact is single-GPU only).
    #[arg(long, value_enum, default_value = "keys")]
    pub(crate) index_cache: crate::families::glm5_flash::engine::IndexCache,
    /// GPU memory (GiB) each coordinator GPU keeps free for runtime growth in --layout (the
    /// engines' --headroom-gib).
    #[arg(long, default_value_t = 2.0)]
    pub(crate) headroom_gib: f64,
    /// Decode graph budget (MiB) for --layout (GLM 5.3 Flash's --graph-budget-mib; unset: the
    /// family's graph allowance). One GPU with Spark experts and an automatic pool keeps the budget
    /// itself, as its measured admission does; other layouts keep at least the allowance.
    #[arg(long)]
    pub(crate) graph_budget_mib: Option<u64>,
    /// GLM 5.3 Flash's replay records for --layout (`shared`: in the prefill scratch, one GPU).
    #[arg(long, value_enum, default_value = "own")]
    pub(crate) replay_records: crate::families::glm5_flash::engine::ReplayRecords,
    /// Concurrent sequences for --layout (0: family default).
    #[arg(long, default_value_t = 0)]
    pub(crate) concurrency: u64,
    /// The engine's recurrent-state slots for --layout, when they differ from the family's rule
    /// over --concurrency (GLM 5.3 Flash serves max(--slots, --max-sequences)).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) state_slots: Option<u64>,
    /// GLM 5.3 Flash: the lanes its prefix mark arena counts for --layout (the 2C + 2 floor),
    /// when they differ from --concurrency (serve-glmf counts min(--max-sequences, 64)).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) mark_lanes: Option<u64>,
    /// Prefix mark arena slots (0 disables marks).
    #[arg(long)]
    pub(crate) prefix_slots: Option<u64>,
    /// MiMo and GLM 5.3 Flash retained snapshots per bank for --layout (matches serving).
    #[arg(long, default_value_t = 20)]
    pub(crate) prefix_cache_entries: u64,
    /// MiMo and GLM 5.3 Flash device positional-mark budget for --layout, MiB.
    #[arg(long, default_value_t = 2048)]
    pub(crate) prefix_cache_mark_mib: u64,
    /// MiMo warm DFlash prefix context marks (off until qualified).
    #[arg(long)]
    pub(crate) mimo_prefix_draft: bool,
    /// MiMo target ring count for --layout (at least concurrency).
    #[arg(long = "rings", default_value_t = 16)]
    pub(crate) mimo_rings: u64,
    /// MiMo drafter batch sequences for --layout (serving raises to concurrency).
    #[arg(long, default_value_t = 4)]
    pub(crate) draft_sequences: u64,
    /// Explicit MiMo drafter context arena for --layout.
    #[arg(long)]
    pub(crate) draft_context_slots: Option<u64>,
    /// MiMo segmented decode graph storage (off by default).
    #[arg(long)]
    pub(crate) mimo_decode_graphs: bool,
    /// Physical coordinator SM count for memory-budget simulations.
    #[arg(long)]
    pub(crate) physical_sms: Option<u32>,
    /// Plan the bundled MiMo drafter without FP8 conversion.
    #[arg(long)]
    pub(crate) mimo_draft_bf16: bool,
    /// MiMo native local expert package manifest.json for exact scratch.
    #[arg(long)]
    pub(crate) mimo_expert_manifest: Option<PathBuf>,
    /// GLM 5.3 Flash's prefix marks for --layout (`pool`: in pool units, no arena, one reserved
    /// unit beside the pool).
    #[arg(long, value_enum, default_value = "arena")]
    pub(crate) prefix_marks: crate::families::glm5_flash::prefix::PrefixMarks,
    /// Native drafter stages, 0 disables the native drafter.
    #[arg(long, default_value_t = 3)]
    pub(crate) native_mtp_layers: usize,
    /// Matching image PROGRAMS.json for exact V4 Flash/Pro workspaces.
    #[arg(long)]
    pub(crate) workspace_manifest: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct DoctorArgs {
    #[arg(long, default_value = "coordinator")]
    pub(crate) role: String,
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) hf_home: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}







#[derive(Debug, Args)]
pub(crate) struct NativeExpertDaemonArgs {
    /// Load the checkpoint vision tower in this expert process.
    #[arg(long, conflicts_with = "encoder_only")]
    pub(crate) encoder: bool,
    /// Serve only vision, without reading or allocating routed experts.
    #[arg(long)]
    pub(crate) encoder_only: bool,
    #[arg(long, default_value = "0.0.0.0:9200")]
    pub(crate) encoder_listen: String,
    /// Coordinator's encoder placement SHA-256; checked in the TCP handshake.
    #[arg(long, required_if_eq_any = [("encoder", "true"), ("encoder_only", "true")])]
    pub(crate) encoder_plan_hash: Option<String>,
    /// Explicit checkpoint revision (defaults to snapshot directory name).
    #[arg(long)]
    pub(crate) encoder_revision: Option<String>,
    #[arg(long, default_value_t = 4096, value_parser = clap::value_parser!(u32).range(1..=4096))]
    pub(crate) encoder_max_tokens: u32,
    /// Load the checkpoint audio tower in this expert process.
    #[arg(long, conflicts_with = "audio_encoder_only")]
    pub(crate) audio_encoder: bool,
    /// Serve only audio, without routed experts; may also host vision.
    #[arg(long, conflicts_with = "encoder")]
    pub(crate) audio_encoder_only: bool,
    #[arg(long, default_value = "0.0.0.0:9300")]
    pub(crate) audio_encoder_listen: String,
    #[arg(long, required_if_eq_any = [("audio_encoder", "true"), ("audio_encoder_only", "true")])]
    pub(crate) audio_encoder_plan_hash: Option<String>,
    #[arg(long)]
    pub(crate) audio_encoder_revision: Option<String>,
    /// First resident backbone layer; use 20 when both RTX GPUs host the encoder.
    /// Checked against the checkpoint's layer count at startup.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..256))]
    pub(crate) first_layer: u32,
    /// Last resident backbone layer (inclusive); defaults to the model's last
    /// layer. A partial range serves a subset of layers, for bring-up and probes.
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..256))]
    pub(crate) last_layer: Option<u32>,
    /// Official local snapshot directory, including all shard headers.
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    /// Spark-role native library built with V4.1 expert AOT kernels.
    #[arg(long)]
    pub(crate) native_lib: PathBuf,
    /// Override the native EXL3 rank directory containing m1, m16 and larger capacities.
    #[arg(long)]
    pub(crate) exl3_aot_dir: Option<PathBuf>,
    /// EXL3 decode schedule: default, or gb10 for the GLM 5.3 Flash TP4 decode exports
    /// (m1-gb10, m80-gb10): the same bits, with weight words staged L2 evict-first and, at
    /// m80, 64x128 tiles at two CTAs per SM.
    #[arg(long, value_enum, default_value_t = crate::shared::experts::exl3::execution::Exl3Schedule::Default)]
    pub(crate) exl3_schedule: crate::shared::experts::exl3::execution::Exl3Schedule,
    /// Override the FP8 expert package layout directory (`fp8-<family>/tp<world>`).
    #[arg(long)]
    pub(crate) fp8_package: Option<PathBuf>,
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..6))]
    pub(crate) rank: u32,
    /// Spark tensor-parallel world; one rank supports whole Qwen EXL3 experts.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=6))]
    pub(crate) world: u32,
    /// Replicated-group tensor-parallel degree inside one group (opt-in; all-or-none with --spark-ep).
    #[arg(long, requires = "spark_ep", value_parser = parse_spark_tp)]
    pub(crate) spark_tp: Option<u8>,
    /// Number of replicated expert groups, each holding all 384 experts (opt-in; all-or-none with --spark-tp).
    #[arg(long, requires = "spark_tp", value_parser = clap::value_parser!(u8).range(1..=3))]
    pub(crate) spark_ep: Option<u8>,
    #[arg(long, default_value_t = 16)]
    pub(crate) capacity: u32,
    /// Total device bytes allowed for resident weights, loading and execution.
    #[arg(long)]
    pub(crate) device_budget_bytes: usize,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    pub(crate) max_frame_bytes: usize,
    #[arg(long, default_value = "0.0.0.0:9100")]
    pub(crate) listen: String,
}

#[derive(Debug, Args)]
pub(crate) struct BenchRdmaArgs {
    #[arg(long)]
    pub(crate) peer: Option<String>,
    #[arg(long, default_value = "auto")]
    pub(crate) mode: String,
    #[arg(long, default_value_t = 18515)]
    pub(crate) port: u16,
    #[arg(long, default_value = "4096,8192,12288,16384,32768,65536")]
    pub(crate) payload_bytes: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) duration_secs: u64,
}

#[derive(Debug, Args)]
pub(crate) struct BenchRdmaRingArgs {
    #[arg(long, default_value = "server")]
    pub(crate) mode: String,
    #[arg(long, default_value = "0.0.0.0:18525")]
    pub(crate) listen: String,
    #[arg(long)]
    pub(crate) peer: Option<String>,
    #[arg(long)]
    pub(crate) peers: Option<String>,
    #[arg(long, default_value_t = 16 * 1024)]
    pub(crate) slot_bytes: usize,
    #[arg(long, default_value_t = 8)]
    pub(crate) depth: usize,
    #[arg(long, default_value_t = 100)]
    pub(crate) warmup_iterations: usize,
    #[arg(long, default_value_t = 1000)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) window: usize,
    #[arg(long)]
    pub(crate) request_bytes: Option<usize>,
    #[arg(long)]
    pub(crate) response_bytes: Option<usize>,
    #[arg(long, default_value_t = 0)]
    pub(crate) compute_delay_us: u64,
    #[arg(long, default_value = "unspecified")]
    pub(crate) network_label: String,
    #[arg(long, default_value_t = false)]
    pub(crate) gpu_echo: bool,
    #[arg(long, default_value = "fp8")]
    pub(crate) wire_codec: String,
    #[arg(long, default_value_t = 1)]
    pub(crate) rows: usize,
    /// Hidden-width partial carried by each TP rank in the reduction benchmark.
    #[arg(long, default_value_t = DS4_FLASH_HIDDEN_SIZE)]
    pub(crate) row_width: usize,
    #[arg(long, default_value_t = 1000)]
    pub(crate) kernel_iterations: usize,
    #[arg(long, default_value_t = 0)]
    pub(crate) reduction_rank: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) reduction_world_size: usize,
    #[arg(long)]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long, default_value_t = 30_000)]
    pub(crate) timeout_ms: u64,
}




#[derive(Debug, Args)]
pub(crate) struct TransportCapabilitiesArgs {
    #[arg(long)]
    pub(crate) benchmark_jsonl: Option<PathBuf>,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}



#[cfg(test)]
mod tests {
    #[test]
    fn dsv4_golden_accepts_nll_and_requires_it_for_saved_logits() {
        use clap::Parser;
        let base = ["cuteafd", "dsv4-golden", "--snapshot", "/model", "--native-lib", "/native.so",
            "--golden", "/golden"];
        let super::Commands::Dsv4Golden(args) = super::Cli::try_parse_from(
            base.into_iter().chain(["--nll", "--save-logits", "/rows"])).unwrap().command else {
            panic!("expected DeepSeek V4 golden");
        };
        assert!(args.nll);
        assert_eq!(args.save_logits.as_deref(), Some(std::path::Path::new("/rows")));
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--save-logits", "/rows"])).is_err());
    }

    #[test]
    fn dsv4_quantizer_defaults_to_device_geometry_and_keeps_tuning_optional() {
        use clap::Parser;
        let base = ["cuteafd", "serve-dsv4", "--snapshot", "/model", "--native-lib", "/native.so"];
        let super::Commands::ServeDsv4(defaults) = super::Cli::try_parse_from(base).unwrap().command else {
            panic!("expected DeepSeek V4 serving");
        };
        assert_eq!(defaults.engine.sms, None);
        let super::Commands::ServeDsv4(limited) = super::Cli::try_parse_from(
            base.into_iter().chain(["--sms", "73"])).unwrap().command else {
            panic!("expected DeepSeek V4 serving");
        };
        assert_eq!(limited.engine.sms, Some(73));
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--sms", "-1"])).is_err());
    }

    #[test]
    fn native_host_cache_accepts_auto_and_legacy_byte_counts() {
        use clap::Parser;
        use crate::families::deepseek_v41::v41_native_serve::memory::HostBudget;
        let base = ["cuteafd", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--peers", "127.0.0.1:19441"];
        for (value, bytes) in [("auto", None), ("0", Some(0)), ("1GiB", Some(1<<30)),
            ("4294967296", Some(1u64<<32))] {
            let super::Commands::ServeNative(args) = super::Cli::try_parse_from(
                base.into_iter().chain(["--host-cache-bytes", value])).unwrap().command else {
                panic!("expected native serving");
            };
            match (args.host_cache_bytes, bytes) {
                (HostBudget::Auto, None) => (),
                (HostBudget::Bytes(actual), Some(expected)) => assert_eq!(actual, expected),
                _ => panic!("incorrect host budget mode"),
            }
        }
        assert!(super::Cli::try_parse_from(base.into_iter()
            .chain(["--host-cache-bytes", "-1"])).is_err());
    }

    #[test]
    fn native_limits_default_to_model_maximum_and_allow_smaller_launches() {
        use clap::Parser;
        let base = ["cuteafd", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--peers", "127.0.0.1:19441"];
        let super::Commands::ServeNative(args) = super::Cli::try_parse_from(base).unwrap().command else {
            panic!("expected native serving");
        };
        assert_eq!(args.max_context_tokens, 1_048_576);
        assert_eq!(args.max_output_tokens, 393_216);
        assert_eq!(args.concurrency, 16);
        assert_eq!(args.prefix_cache_entries, 20);
        assert_eq!(args.dspark_draft_limit, 5);
        assert!(!args.exl3_paired_tp4);
        let super::Commands::ServeNative(paired) = super::Cli::try_parse_from(
            base.into_iter().chain(["--exl3-paired-tp4"])).unwrap().command else {
            panic!("expected native serving");
        };
        assert!(paired.exl3_paired_tp4);
        assert!(!args.adaptive_dspark()); // Target-only remains target-only.
        for (flags, adaptive) in [
            (vec!["--dspark"], true),
            (vec!["--dspark", "--independent-decode-lanes"], true),
            (vec!["--dspark", "--dspark-fixed"], false),
        ] {
            let super::Commands::ServeNative(args) = super::Cli::try_parse_from(
                base.into_iter().chain(flags)).unwrap().command else { panic!("expected native serving"); };
            assert_eq!(args.adaptive_dspark(), adaptive);
        }
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-fixed"])).is_err());
        // The removed policies' flags are rejected rather than silently ignored.
        for removed in [["--dspark", "--dspark-adaptive"].as_slice(),
            &["--dspark", "--dspark-confidence-cutoff", "0.5"],
            &["--dspark", "--dspark-reuse-floor", "0.5"]] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(removed.iter().copied())).is_err());
        }
        for (flags, expected) in [
            (vec!["--dspark"], 5),
            (vec!["--dspark", "--rtx-gpus", "1"], 5),
            (vec!["--dspark", "--rtx-gpus", "2"], 5),
            (vec!["--dspark", "--rtx-gpus", "2", "--dspark-draft-limit", "5"], 5),
            (vec!["--dspark", "--dspark-draft-limit", "5", "--rtx-gpus", "2"], 5),
            (vec!["--dspark", "--dspark-draft-limit", "5"], 5),
            (vec!["--dspark", "--rtx-gpus", "2", "--dspark-draft-limit", "7"], 7),
        ] {
            let super::Commands::ServeNative(args) = super::Cli::try_parse_from(
                base.into_iter().chain(flags)).unwrap().command else { panic!("expected native serving"); };
            assert_eq!(args.dspark_draft_limit, expected);
        }
        for limit in ["1", "2", "3", "4", "5", "6", "7"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-draft-limit", limit])).is_ok());
        }
        for limit in ["0", "8"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--dspark-draft-limit", limit])).is_err());
        }
        for entries in ["0", "2", "24", "128"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--prefix-cache-entries", entries])).is_ok());
        }
        assert!(super::Cli::try_parse_from(base.into_iter().chain(["--prefix-cache-entries", "129"])).is_err());
        for concurrency in ["1", "2", "16"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--concurrency", concurrency, "--kv-pool-size", "1.5GiB", "--memory-reservation", "87.5%"])).is_ok());
        }
        for concurrency in ["0", "17"] {
            assert!(super::Cli::try_parse_from(base.into_iter().chain(["--concurrency", concurrency])).is_err());
        }
        for (context, output, valid) in [("256", "128", true), ("0", "128", false),
            ("1048577", "128", false), ("256", "0", false), ("256", "393217", false)] {
            let command = base.into_iter().chain(["--max-context-tokens", context, "--max-output-tokens", output]);
            assert_eq!(super::Cli::try_parse_from(command).is_ok(), valid);
        }
    }
    use super::*;

    #[test]
    fn transport_benchmark_defaults_use_flash_geometry() {
        let cli = Cli::try_parse_from(["cuteafd", "bench-rdma-ring"]).unwrap();
        let Commands::BenchRdmaRing(args) = cli.command else {
            panic!("expected bench-rdma-ring command");
        };
        assert_eq!(args.row_width, DS4_FLASH_HIDDEN_SIZE);
    }

    #[test]
    fn spark_topology_flags_are_opt_in_all_or_none_and_range_checked() {
        use clap::Parser;
        let serve = ["cuteafd", "serve-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--peers", "127.0.0.1:19441"];
        let expert = ["cuteafd", "expertd-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--device-budget-bytes", "1000", "--rank", "0", "--world", "4"];
        // Absent keys preserve the legacy launch vector exactly.
        for base in [&serve[..], &expert[..]] {
            let cli = Cli::try_parse_from(base).unwrap();
            match cli.command {
                Commands::ServeNative(args) => {
                    assert_eq!((args.spark_tp, args.spark_ep), (None, None));
                }
                Commands::Expertd(args) => {
                    assert_eq!((args.spark_tp, args.spark_ep), (None, None));
                }
                other => panic!("unexpected command {other:?}"),
            }
        }
        // Both keys parse for either process.
        let cli = Cli::try_parse_from(serve.into_iter().chain(["--spark-tp", "2", "--spark-ep", "2"]))
            .unwrap();
        let Commands::ServeNative(args) = cli.command else { panic!("serve-native") };
        assert_eq!((args.spark_tp, args.spark_ep), (Some(2), Some(2)));
        let cli = Cli::try_parse_from(expert.into_iter()
            .chain(["--spark-tp", "3", "--spark-ep", "2"])).unwrap();
        let Commands::Expertd(args) = cli.command else { panic!("expertd-native") };
        assert_eq!((args.spark_tp, args.spark_ep), (Some(3), Some(2)));
        // All-or-none and value ranges are enforced at parse time.
        for flags in [vec!["--spark-tp", "2"], vec!["--spark-ep", "2"]] {
            assert!(Cli::try_parse_from(serve.into_iter().chain(flags.clone())).is_err());
            assert!(Cli::try_parse_from(expert.into_iter().chain(flags)).is_err());
        }
        for (tp, ep) in [("1", "2"), ("5", "1"), ("2", "0"), ("2", "4")] {
            let flags = ["--spark-tp", tp, "--spark-ep", ep];
            assert!(Cli::try_parse_from(serve.into_iter().chain(flags)).is_err(), "{tp}x{ep}");
            assert!(Cli::try_parse_from(expert.into_iter().chain(flags)).is_err(), "{tp}x{ep}");
        }
        // Whole-expert Qwen TP1 must reach the worker's family validation.
        let one = ["cuteafd", "expertd-native", "--snapshot", "/model", "--native-lib", "/native.so",
            "--device-budget-bytes", "1000", "--rank", "0", "--world"];
        let cli = Cli::try_parse_from(one.into_iter().chain(["1"])).unwrap();
        let Commands::Expertd(args) = cli.command else { panic!("expertd-native") };
        assert_eq!((args.rank, args.world), (0, 1));
        for invalid in ["0", "7"] {
            assert!(Cli::try_parse_from(one.into_iter().chain([invalid])).is_err());
        }
        // The worker world range also admits the six-rank layouts.
        let cli = Cli::try_parse_from(["cuteafd", "expertd-native", "--snapshot", "/model",
            "--native-lib", "/native.so", "--device-budget-bytes", "1000",
            "--rank", "5", "--world", "6"]).unwrap();
        let Commands::Expertd(args) = cli.command else { panic!("expertd-native") };
        assert_eq!((args.rank, args.world), (5, 6));

        // Pure TP6EP1 is a real shard family on both processes. Five is not a
        // family and stays rejected even though it is inside the numeric range.
        let six_expert = ["cuteafd", "expertd-native", "--snapshot", "/model", "--native-lib",
            "/native.so", "--device-budget-bytes", "1000", "--rank", "5", "--world", "6"];
        for command in [&serve[..], &six_expert[..]] {
            let cli = Cli::try_parse_from(command.iter().copied()
                .chain(["--spark-tp", "6", "--spark-ep", "1"])).unwrap();
            match cli.command {
                Commands::ServeNative(args) => assert_eq!((args.spark_tp, args.spark_ep), (Some(6), Some(1))),
                Commands::Expertd(args) => assert_eq!((args.spark_tp, args.spark_ep), (Some(6), Some(1))),
                other => panic!("unexpected command {other:?}"),
            }
            let error = Cli::try_parse_from(command.iter().copied()
                .chain(["--spark-tp", "5", "--spark-ep", "1"]))
                .expect_err("TP5 has no shard family");
            assert!(error.to_string().contains("expected 2, 3, 4 or 6"), "{error}");
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct NativeServeArgs {
    /// Checked Spark vision replicas; omitted for local RTX fallback.
    #[arg(long)]
    pub vision_peers: Option<String>,
    #[arg(long, requires = "vision_peers")]
    pub encoder_plan_hash: Option<String>,
    #[arg(long, requires = "vision_peers")]
    pub encoder_revision: Option<String>,
    /// Embedding placement: pinned mapped RAM on <=32 GiB, GPU otherwise; explicit overrides win.
    #[arg(long, alias = "embed-placement", value_enum)]
    pub embedding_placement: Option<crate::shared::token_io::EmbedPlacement>,
    /// Force one RTX or the distributed two-RTX layout (automatic launcher selection is pending).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=2))]
    pub rtx_gpus: u32,
    /// Private startup handoff directory supplied by the release launcher.
    #[arg(long, hide = true)]
    pub placement_directory: Option<std::path::PathBuf>,
    /// Experimental TP2 attention with replicated KV; requires two RTX GPUs.
    #[arg(long)]
    pub tp2_attention: bool,
    /// Experimental TP2 query-B projection; independent of TP2 attention, requires two RTX GPUs.
    #[arg(long)]
    pub tp2_query_projection: bool,
    /// Experimental TP2 output-B projection; independent of other TP2 switches, requires two RTX GPUs.
    #[arg(long)]
    pub tp2_output_projection: bool,
    /// Experimental native dSpark routed-expert TP2; requires --dspark and two RTX GPUs.
    #[arg(long)]
    pub tp2_dspark_experts: bool,

    /// Maximum tokens per prefill step. Storage rounds up to an AOT capacity
    /// (80, 256, 1024, or 4096); all expert peers must support that capacity.
    #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u32).range(80..=4096))]
    pub prefill_batch_tokens: u32,

    /// Share of the time decoding requests keep while a prompt prefills in
    /// encoder waves. 0 (default) prefills whole prompts at admission and builds
    /// no prefill queue; positive shares are not yet qualified (v3 stage 1b).
    #[arg(long, env = "CUTEAFD_V41_DECODE_SHARE", default_value_t = 0.0, hide = true)]
    pub decode_share: f64,

    /// Total prompt plus generated tokens; compressed cache is reserved at startup.
    #[arg(long, default_value_t = cuteafd_api::openai::MAX_CONTEXT_TOKENS, value_parser = clap::value_parser!(u32).range(1..=1048576))]
    pub max_context_tokens: u32,

    /// Default and maximum generated tokens, further bounded by remaining context.
    #[arg(long, default_value_t = cuteafd_api::openai::MAX_OUTPUT_TOKENS, value_parser = clap::value_parser!(u32).range(1..=393216))]
    pub max_output_tokens: u32,

    /// Exact global KV/index byte budget (B/MB/GB/MiB/GiB), rounded down to page groups.
    #[arg(long)]
    pub kv_pool_size: Option<crate::families::deepseek_v41::v41_native_serve::memory::ByteSize>,

    /// Aggregate logical KV tokens; 0 selects planner admission. Omit to keep
    /// the existing V4.1 pool policy. Cannot be combined with --kv-pool-size.
    #[arg(long, conflicts_with = "kv_pool_size")]
    pub pool_tokens: Option<u64>,

    /// Total device occupancy ceiling (% or B/MB/GB/MiB/GiB); sizes KV after fixed allocations.
    #[arg(long)]
    pub memory_reservation: Option<crate::families::deepseek_v41::v41_native_serve::memory::Reservation>,

    /// Complete bottom-up RTX routed layers: auto fills available memory, or 0..40.
    #[arg(long, default_value = "auto")]
    pub rtx_expert_layers: crate::families::deepseek_v41::v41_native_serve::memory::LocalLayers,

    /// Use paired H128 EXL3 ownership; requires paired AOT packages on all four Spark peers.
    #[arg(long)]
    pub exl3_paired_tp4: bool,

    /// Maximum active requests, shared by both execution lanes.
    #[arg(long, default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..=16))]
    pub concurrency: u32,

    /// Buffered HTTP jobs; defaults to concurrency. At most this many additional callers wait.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=4096))]
    pub http_queue_depth: Option<u32>,

    /// Maximum wait for space in the HTTP job queue; zero rejects immediately.
    #[arg(long, default_value_t = 25000)]
    pub http_queue_wait_ms: u64,

    /// Retained completed turns, plus a separate prompt-repeat bank of this size; zero disables reuse.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(0..=128))]
    pub prefix_cache_entries: u32,

    /// Pinned snapshot memory: auto sizes logical GPU+RAM capacity above retained entries * context; 0 disables it.
    #[arg(long, default_value = "0", env = "CUTEAFD_HOST_CACHE_BYTES")]
    pub host_cache_bytes: crate::families::deepseek_v41::v41_native_serve::memory::HostBudget,
    /// Pinned allocation and registration granularity for the snapshot cache.
    #[arg(long, default_value_t = 256 << 20, env = "CUTEAFD_HOST_CACHE_CHUNK_BYTES")]
    pub host_cache_chunk_bytes: u64,
    /// When the device-to-host copy of a retained snapshot is issued.
    #[arg(long, value_enum, default_value_t = HostCacheStore::OnRetain, env = "CUTEAFD_HOST_CACHE_STORE")]
    pub host_cache_store: HostCacheStore,
    /// Longest the device-evict path waits for an in-flight store before dropping it uncached.
    /// A dropped snapshot costs its next visit a full re-prefill (minutes at long contexts), so the
    /// budget errs long: a bounded stall of the scheduler beats losing the snapshot.
    #[arg(long, default_value_t = 1000, env = "CUTEAFD_HOST_CACHE_COPY_BUDGET_MS")]
    pub host_cache_copy_budget_ms: u64,
    /// Longest a host restore waits before the request falls through to prefill.
    #[arg(long, default_value_t = 500, env = "CUTEAFD_HOST_CACHE_RESTORE_BUDGET_MS")]
    pub host_cache_restore_budget_ms: u64,
    /// Pace prefill chunks against pending host-cache stores: when the oldest in-flight
    /// store copy is older than this, each prefill chunk boundary waits on its event for at
    /// most this long again. Zero disables the guard (no holds, no hold metrics). The
    /// recommended fleet value is 500 ms.
    #[arg(long, default_value_t = 0, env = "CUTEAFD_HOST_CACHE_STORE_PACE_MS")]
    pub host_cache_store_pace_ms: u64,
    /// Snapshots shorter than this are not cached.
    #[arg(long, default_value_t = 512, env = "CUTEAFD_HOST_CACHE_MIN_TOKENS")]
    pub host_cache_min_tokens: u32,
    /// Snapshots longer than this are not cached.
    #[arg(long, default_value_t = cuteafd_api::openai::MAX_CONTEXT_TOKENS, env = "CUTEAFD_HOST_CACHE_MAX_TOKENS")]
    pub host_cache_max_tokens: u32,
    /// Which retention banks the cache serves: `prompt`, `turn`, or `prompt,turn`.
    #[arg(long, default_value = "prompt,turn", env = "CUTEAFD_HOST_CACHE_KINDS")]
    pub host_cache_kinds: String,

    /// Enable greedy RTX dSpark proposal generation and target verification.
    #[arg(long)] pub dspark: bool,
    /// Draft width (tokens proposed per request). Defaults to the drafter's
    /// trained five-token block on both layouts; 7 loads the wider block and
    /// lets the policy choose 5 or 7 each round (measured slower under
    /// concurrency, so not the default).
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u8).range(1..=7))]
    pub dspark_draft_limit: u8,
    /// Verify every available draft instead of the online bandwidth-balance
    /// length selection (the default whenever dSpark is enabled).
    #[arg(long, requires = "dspark")]
    pub dspark_fixed: bool,
    /// Compatibility spelling: decode lanes always advance independently.
    #[arg(long, hide = true)]
    pub independent_decode_lanes: bool,
    /// Let the live console at `/` show generated token text. Only viewers
    /// holding the console unlock cookie receive it (bench runs excepted).
    #[arg(long, env = "CUTEAFD_CONSOLE_TEXT", num_args = 0..=1, default_value = "true",
        default_missing_value = "true", value_parser = clap::builder::BoolishValueParser::new())]
    pub console_text: bool,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,

    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long)] pub native_lib: PathBuf,
    #[arg(long,value_delimiter=',',num_args=1..)] pub peers: Vec<std::net::SocketAddr>,
    #[arg(long,default_value="127.0.0.1:8000")] pub listen: String,
    /// Replicated-group tensor-parallel degree inside one Spark group (opt-in; all-or-none with --spark-ep).
    #[arg(long, requires = "spark_ep", value_parser = parse_spark_tp)]
    pub spark_tp: Option<u8>,
    /// Number of replicated Spark expert groups, each holding all 384 experts (opt-in; all-or-none with --spark-tp).
    #[arg(long, requires = "spark_tp", value_parser = clap::value_parser!(u8).range(1..=3))]
    pub spark_ep: Option<u8>,
}

/// Spark TP degrees with a native shard family: 2/3 replicated-group shards,
/// 4 the legacy grouped layout, and 6 the pure unreplicated `TP6EP1` slices.
/// Five has no artifact family, so it is rejected here rather than at load time.
fn parse_spark_tp(value: &str) -> Result<u8, String> {
    let value: u8 = value
        .parse()
        .map_err(|_| "expected a Spark TP degree between 2 and 6".to_string())?;
    match value {
        2 | 3 | 4 | 6 => Ok(value),
        _ => Err(format!(
            "unsupported Spark TP degree {value}; expected 2, 3, 4 or 6"
        )),
    }
}

/// `cuteafd plan --decode-rows`: GLM 5.3 Flash's decode programs' 64 rows, or the wide programs' 128.
fn plan_decode_rows(value: &str) -> Result<u64, String> {
    match value {
        "64" => Ok(64),
        "128" => Ok(128),
        _ => Err("64 or 128".to_string()),
    }
}

impl NativeServeArgs {
    pub fn adaptive_dspark(&self) -> bool {
        self.dspark && !self.dspark_fixed
    }
}

/// When the host snapshot cache copies a retained snapshot (see `cuteafd_hostcache::config::StoreMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum HostCacheStore {
    OnRetain,
    OnEvict,
}

impl NativeServeArgs {
    /// The host snapshot cache configuration these flags describe (validated by the cache).
    pub fn host_cache_config(&self) -> anyhow::Result<cuteafd_hostcache::config::Config> {
        use cuteafd_hostcache::config::{Config, Kinds, StoreMode};
        let mut kinds = Kinds { prompt: false, turn: false };
        for kind in self.host_cache_kinds.split(',').map(str::trim).filter(|k| !k.is_empty()) {
            match kind {
                "prompt" => kinds.prompt = true,
                "turn" => kinds.turn = true,
                other => anyhow::bail!("unknown host cache kind {other:?} (expected prompt or turn)"),
            }
        }
        let config = Config {
            bytes: self.host_cache_bytes.explicit_bytes(),
            chunk_bytes: self.host_cache_chunk_bytes,
            store: match self.host_cache_store {
                HostCacheStore::OnRetain => StoreMode::OnRetain,
                HostCacheStore::OnEvict => StoreMode::OnEvict,
            },
            copy_budget_ns: self.host_cache_copy_budget_ms * 1_000_000,
            restore_budget_ns: self.host_cache_restore_budget_ms * 1_000_000,
            store_pace_ns: self.host_cache_store_pace_ms * 1_000_000,
            min_tokens: self.host_cache_min_tokens,
            max_tokens: self.host_cache_max_tokens,
            kinds,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Retain a diagnostic for old launch configurations, never the old path.
pub(crate) fn reject_peer_expert_ranges(text: &str) -> Result<bool, String> {
    if text == "false" { return Ok(false); }
    Err("GPU1 whole-layer expert ranges were replaced by TP2 halves (v3 P4); drop --peer-expert-ranges".into())
}
