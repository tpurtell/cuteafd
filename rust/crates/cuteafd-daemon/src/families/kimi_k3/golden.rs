//! Cold residual-prefix diagnostics. These states never qualify logits or NLL.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, path::{Path, PathBuf}, rc::Rc, sync::Arc};
use super::{admission, device::NativePrograms, engine::Engine};
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory}, memory::device::{Device, Stream},
    peer_split::{hop::HopLink, PeerExchange, RankDevice}};
use cuteafd_core::{ExpertGeometry, memory_layout::{Basis, Category}};
use cuteafd_loader::{families::kimi_k3::config::KimiK3Config, placement::{Demand, GpuBudget, HOP_SLOTS}};

const HIDDEN: usize = 7168;
const REVISION: &str = "f831ab66814297da540d832a5235f8e904f29d06";
const SOURCE: [(&str, &str); 3] = [
    ("configuration_kimi_k3_sha256", "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae"),
    ("modeling_kimi_linear_sha256", "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"),
    ("modeling_kimi_k3_sha256", "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2"),
];

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[arg(long)] snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] manifest: PathBuf,
    /// Read-only cuteafd.kimi.layer-state/1 export, never a qualified reference panel.
    #[arg(long)] reference_state: PathBuf,
    /// Original token panel whose set and window hashes bind the residual state.
    #[arg(long)] windows: PathBuf,
    #[arg(long)] window: String,
    /// Inclusive zero-based decoder layer; inventory is truncated before admission.
    #[arg(long)] last_layer: usize,
    #[arg(long, value_delimiter = ',')] peers: Vec<SocketAddr>,
    #[arg(long, default_value_t = 0)] device: i32,
    #[arg(long, default_value_t = 1)] split_device: i32,
    #[arg(long)] check_headers: bool,
    /// Informational F32 comparison only; direct gate/up is the speed-first default.
    #[arg(long, env = "CUTEAFD_KIMI_COMPENSATED_FFN")]
    compensated_ffn: bool,
    /// Always refused: a residual prefix cannot score or qualify the model.
    #[arg(long)] nll: bool,
}

struct Reference { tokens: Vec<u32>, hidden: Vec<f32> }

impl GoldenArgs {
    fn ffn_mode(&self) -> cuteafd_ffi::KimiFfnMode {
        if self.compensated_ffn { cuteafd_ffi::KimiFfnMode::Compensated }
        else { cuteafd_ffi::KimiFfnMode::default() }
    }

    fn provenance(&self) -> Value {
        serde_json::json!({"schema": "cuteafd.kimi.run/1", "ffn_mode": self.ffn_mode().name(),
            "gate_up_programs": ([3072, 16896].map(|width| self.ffn_mode().gate_up_stem(width))),
            "window": self.window, "last_layer": self.last_layer, "diagnostic_only": true})
    }
}

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn sha(value: &Value) -> bool {
    value.as_str().is_some_and(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
}

fn reference(root: &Path, panel: &Value, window: &str, last: usize) -> Result<Reference> {
    ensure!(!window.is_empty() && window.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
        "Kimi unsafe reference window ID");
    let meta: Value = serde_json::from_slice(&std::fs::read(root.join("latest.json"))?)?;
    ensure!(meta["schema"] == "cuteafd.kimi.layer-state/1" && meta["layer"].as_u64() == Some(last as u64)
        && meta["diagnostic_prefix"].as_u64() == Some(last as u64),
        "Kimi requires a flagged diagnostic state retaining exactly --last-layer");
    let binding = &meta["binding"];
    ensure!(binding["state_dtype"] == "float32" && panel["family"] == "kimi_k3"
        && sha(&panel["set_sha256"]) && binding["set_sha256"] == panel["set_sha256"],
        "Kimi reference dtype, family or set binding differs");
    ensure!(binding["arithmetic"] == serde_json::json!({"dtype":"float32", "matmul_precision":"highest",
        "torch_matmul_tf32":false, "cudnn_tf32":false, "triton_dot_precision":"ieee",
        "kda_solve_tril_precision":"ieee"}), "Kimi reference F32 arithmetic policy differs");
    let identity = &binding["snapshot_identity"];
    ensure!(identity["snapshot_revision"] == REVISION && sha(&identity["config_sha256"])
        && sha(&identity["tokenizer_sha256"]), "Kimi reference snapshot identity differs");
    if let Some(arm) = panel.get("reference_snapshot") {
        for key in ["snapshot_revision", "config_sha256", "tokenizer_sha256", "checkpoint", "precision"] {
            if let Some(expected) = arm.get(key) {
                ensure!(identity.get(key) == Some(expected), "Kimi panel reference identity differs: {key}");
            }
        }
    }
    if let Some(expected) = panel.get("tokenizer_sha256") {
        ensure!(&identity["tokenizer_sha256"] == expected, "Kimi panel tokenizer identity differs");
    }
    let pins = &binding["source_pin"];
    ensure!(pins["revision"] == REVISION, "Kimi reference official revision differs");
    for (key, expected) in SOURCE { ensure!(pins[key] == expected, "Kimi reference source pin differs: {key}"); }
    for key in ["fla_source_sha256", "transformers_source_sha256", "compressed_tensors_source_sha256",
        "reference_adapter_sha256", "reference_runner_sha256"] {
        ensure!(sha(&pins[key]), "Kimi missing reference runtime/source digest: {key}");
    }
    let matches = panel["windows"].as_array().context("Kimi panel has no windows")?.iter()
        .filter(|w| w["id"] == window).collect::<Vec<_>>();
    ensure!(matches.len() == 1, "Kimi window absent or duplicated");
    ensure!(matches[0]["media"].as_array().is_none_or(Vec::is_empty), "Kimi diagnostic is text-only");
    let ids = matches[0]["tokens"].as_array().context("Kimi window has no tokens")?;
    let tokens = ids.iter().map(|id| id.as_u64().filter(|id| *id < 163840)
        .map(|id| id as u32).context("Kimi token outside vocabulary")).collect::<Result<Vec<_>>>()?;
    ensure!((1..=4096).contains(&tokens.len()), "Kimi cold diagnostic window must fit one 4096-row lane");
    ensure!(binding["windows"][window] == digest(&serde_json::to_vec(&tokens)?),
        "Kimi window token hash differs");
    let blob = &meta["windows"][window]["hidden"];
    let name = format!("layer{last:03}/{window}.bin");
    ensure!(blob["path"] == name && sha(&blob["sha256"]), "Kimi residual path or SHA256 differs");
    let root = root.canonicalize()?;
    let path = root.join(&name);
    ensure!(path.canonicalize()?.starts_with(&root) && !path.symlink_metadata()?.file_type().is_symlink(),
        "Kimi residual path escapes state directory");
    let expected = tokens.len().checked_mul(HIDDEN * 4).context("Kimi residual extent overflow")?;
    ensure!(blob["bytes"].as_u64() == Some(expected as u64) && path.metadata()?.len() == expected as u64,
        "Kimi reference residual shape differs");
    let bytes = std::fs::read(path)?;
    ensure!(blob["sha256"] == digest(&bytes), "Kimi residual file SHA256 differs");
    let hidden = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    ensure!(hidden.iter().all(|v| v.is_finite()), "Kimi non-finite reference residual");
    Ok(Reference { tokens, hidden })
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    ensure!(!actual.is_empty() && actual.len() == expected.len()
        && actual.iter().chain(expected).all(|v| v.is_finite()), "Kimi residual comparison shape/finite failure");
    let (mut aa, mut bb, mut ab, mut dd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        aa += a * a; bb += b * b; ab += a * b; dd += (a - b) * (a - b);
    }
    ensure!(aa > 0.0 && bb > 0.0, "Kimi zero-norm residual comparison");
    Ok((ab / (aa.sqrt() * bb.sqrt()), (dd / bb).sqrt()))
}

fn hosted_layers(value: &str) -> Result<Vec<usize>> {
    if value.is_empty() { return Ok(vec![]); }
    value.split(',').map(|s| s.parse::<usize>().context("Kimi hosted layer ID")).collect()
}

pub(crate) fn run(args: GoldenArgs) -> Result<()> {
    ensure!(!args.nll, "Kimi residual prefix is diagnostic only; NLL/KL/publication unavailable");
    tracing::info!(ffn_mode = args.ffn_mode().name(), "Kimi FFN numerics selection");
    println!("Kimi run provenance: {}", args.provenance());
    let panel: Value = serde_json::from_slice(&std::fs::read(&args.windows)?)?;
    let reference = reference(&args.reference_state, &panel, &args.window, args.last_layer)?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&args.snapshot, false, false)?;
    let full = KimiK3Config::from_hf(&checkpoint.config)?;
    full.release_geometry()?;
    let cfg = admission::prefix(&full, args.last_layer.checked_add(1).context("Kimi last-layer overflow")?)?;
    let hosted = hosted_layers(&std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?)?;
    ensure!(args.peers.len() == 6 && hosted == cfg.dense.iter().enumerate()
        .filter_map(|(layer, dense)| (!dense).then_some(layer)).collect::<Vec<_>>(),
        "Kimi diagnostic needs exactly the selected routed layers on six real ranks");
    if args.check_headers {
        println!("Kimi diagnostic reference/header binding valid: layers=0..{} rows={}; no numerical qualification",
            args.last_layer, reference.tokens.len());
        return Ok(());
    }
    native(&args, &checkpoint, &full, &cfg, reference)
}

fn native(args: &GoldenArgs, checkpoint: &cuteafd_loader::plan::Checkpoint,
    full: &KimiK3Config, cfg: &KimiK3Config, reference: Reference) -> Result<()> {
    cuteafd_core::set_expert_geometry(ExpertGeometry::KIMI_K3)
        .map_err(|_| anyhow::anyhow!("Kimi golden needs an isolated process"))?;
    // SAFETY: sealed runner supplies the DSO; all borrowing owners drop before it.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(args.native_lib.as_ref().context("--native-lib required")?)? });
    ensure!(crate::shared::peer_split::probed_device(&library, args.device, Some(args.split_device))?
        == Some(args.split_device), "Kimi TP2 requires bidirectional peer access");
    let devices = [args.device, args.split_device].map(|id| Device { library: &library, id });
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let native = devices.map(|device| NativePrograms::load(device, full, &programs, args.ffn_mode()).map(Rc::new));
    let [zero, one] = native;
    let native = [zero?, one?];
    let diagnostics = std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1");
    let inventory = RuntimeInventory::measure(&library, &[args.device, args.split_device], InventoryPhase::PreWeight, |_| {
        library.tp2_native_initialize()?;
        library.peer_exchange_initialize()?;
        library.peer_abort_initialize()?;
        if diagnostics { library.peer_diagnostic_initialize()?; }
        library.v41_compact_reducer()?.require_plane_reduction(6)?;
        Ok(())
    })?;
    let rows = reference.tokens.len();
    let lane_layouts = vec![[native[0].lane_layout(full, rows, 1_048_576)?,
        native[1].lane_layout(full, rows, 1_048_576)?]];
    let mut fixed = exchange_demands(rows, diagnostics)?;
    let selected = cuteafd_loader::placement::ProgramSet::from_names(&["kimik3"], programs.names());
    for (rank, sample) in inventory.gpus.iter().enumerate() {
        let pending = crate::shared::inventory::pending_code(&library, sample.device, rank, true,
            "kimik3", "exl3", &selected)?;
        let arch = cuteafd_loader::placement::ArchContext::for_device(&sample.arch, sample.total_bytes);
        fixed.push(code_demand(rank, pending, arch.cublas_bytes, 1)?);
    }
    let gpus = inventory.gpus.iter().map(|sample| {
        let capacity = cuteafd_ffi::coordinator_gpu_budget().map_or(sample.total_bytes, |b| b.0.min(sample.total_bytes));
        GpuBudget { capacity_bytes: capacity, headroom_bytes: 2 << 30,
            baseline: cuteafd_loader::placement::Baseline::Measured { free_bytes: sample.free_bytes } }
    }).collect::<Vec<_>>().try_into().map_err(|_| anyhow::anyhow!("Kimi GPU sample count"))?;
    let admitted = admission::admit(admission::Inputs { cfg, gpus, devices: [args.device, args.split_device],
        peer_access: true, spark_ranks: 6, attention: None, requested_pool: Some(262_144),
        state_slots: 1, prefix_marks: 1, rows, lanes: 1, lane_layouts, fixed,
        startup: vec![], pool_bounded: vec![] })?;
    let plan = admitted.execution_plan()?;
    let admission_free = inventory.gpus.iter().map(|sample| sample.free_bytes).collect::<Vec<_>>();
    let admitted_bytes = admitted.placement.items.iter().map(|items| items.iter().map(|item| item.bytes).sum::<u64>())
        .collect::<Vec<_>>();
    let pages = admitted.placement.pool_tokens as usize / cuteafd_ffi::MLA_PAGE_ROWS;
    let streams = [Rc::new(Stream::new(devices[0])?), Rc::new(Stream::new(devices[1])?)];
    let ranks = streams.each_ref().map(|s| RankDevice { device: s.device.id, stream: s.raw });
    let peer = devices[0].run(|| PeerExchange::new_abortable(&library, ranks, 1, rows * HIDDEN * 2))?;
    let mut hops = devices[0].run(|| HopLink::new(&library, ranks, plan.hops(), plan.spec()))?;
    hops.attach_abort([peer.abort_word(0)?.context("Kimi missing abort owner")?,
        peer.abort_word(1)?.context("Kimi missing abort owner")?])?;
    let engine = Engine::load_admitted(&library, streams, admitted, checkpoint, native, peer, hops,
        args.peers.clone(), cuteafd_transport::TcpTransportConfig::default(), 4096)?;
    let loaded = RuntimeInventory::measure(&library, &[args.device, args.split_device], InventoryPhase::PostLoad, |_| Ok(()))?;
    loaded.record("kimi_k3-prefix-loaded", &admission_free);
    for (rank, sample) in loaded.gpus.iter().enumerate() {
        tracing::info!(rank, device = sample.device, admitted_bytes = admitted_bytes[rank],
            measured_growth_bytes = admission_free[rank].saturating_sub(sample.free_bytes),
            "Kimi diagnostic loaded usage against admission");
    }
    let mut pool = cuteafd_engine::prefix::RefPagePool::new(pages, cuteafd_ffi::MLA_PAGE_ROWS);
    let allocated = pool.alloc(rows.div_ceil(cuteafd_ffi::MLA_PAGE_ROWS))?;
    let actual = engine.prefix_residual(&reference.tokens, &allocated, &pool)?;
    let steady = RuntimeInventory::measure(&library, &[args.device, args.split_device], InventoryPhase::PostLoad, |_| Ok(()))?;
    steady.record("kimi_k3-prefix-steady", &admission_free);
    for (rank, sample) in steady.gpus.iter().enumerate() {
        tracing::info!(rank, device = sample.device, admitted_bytes = admitted_bytes[rank],
            measured_growth_bytes = admission_free[rank].saturating_sub(sample.free_bytes),
            "Kimi diagnostic steady usage against admission");
    }
    let (cosine, relative_l2) = compare(&actual, &reference.hidden)?;
    println!("Kimi diagnostic only: layer={} rows={} cosine={cosine:.12} relative_l2={relative_l2:.12}; no NLL/KL or qualification",
        args.last_layer, rows);
    Ok(())
}

fn code_demand(rank: usize, pending: u64, cublas_bytes: u64, lanes: usize) -> Result<Demand> {
    ensure!(rank < 2 && (1..=2).contains(&lanes), "Kimi code allowance rank/lane geometry");
    // Shared allowance covers one thread-local handle per rank. Each head-rank
    // lane owns an additional private handle beside its explicitly charged workspace.
    let heads = if rank == 1 { lanes as u64 } else { 0 };
    let bytes = heads.checked_mul(cublas_bytes).and_then(|extra| pending.checked_add(extra))
        .context("Kimi future code/handle allowance overflow")?;
    Ok(Demand::new(rank as u8, Category::Workspace, "Kimi future loaded code and private head handles",
        bytes, Basis::Formula))
}

fn exchange_demands(rows: usize, diagnostics: bool) -> Result<Vec<Demand>> {
    ensure!((1..=4096).contains(&rows), "Kimi exchange row capacity");
    let recv = (rows * HIDDEN * 2).max(256);
    let progress = if diagnostics { (2 * std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>()).max(256) } else { 0 };
    // Hop receive planes are charged by the shared solver; only controls are fixed.
    let bytes = recv + 256 + 256 + (((HOP_SLOTS + 1) * 16) as usize).max(256) + progress;
    Ok((0..2).map(|rank| Demand::new(rank, Category::Workspace, "Kimi peer/abort/hop controls",
        bytes as u64, Basis::Exact)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture(root: &Path) -> Result<Value> {
        std::fs::create_dir_all(root.join("layer003"))?;
        let bytes = vec![0u8; 2 * HIDDEN * 4];
        std::fs::write(root.join("layer003/w.bin"), &bytes)?;
        let mut pins = json!({"revision":REVISION});
        for (key, value) in SOURCE { pins[key] = json!(value); }
        for key in ["fla_source_sha256", "transformers_source_sha256", "compressed_tensors_source_sha256",
            "reference_adapter_sha256", "reference_runner_sha256"] { pins[key] = json!("a".repeat(64)); }
        let panel = json!({"family":"kimi_k3", "set_sha256":"b".repeat(64), "windows":[{"id":"w","tokens":[1,2]}]});
        let state = json!({"schema":"cuteafd.kimi.layer-state/1", "layer":3, "diagnostic_prefix":3,
            "binding":{"source_pin":pins,"state_dtype":"float32","set_sha256":panel["set_sha256"],
                "arithmetic":{"dtype":"float32","matmul_precision":"highest","torch_matmul_tf32":false,
                    "cudnn_tf32":false,"triton_dot_precision":"ieee","kda_solve_tril_precision":"ieee"},
                "snapshot_identity":{"snapshot_revision":REVISION,"config_sha256":"c".repeat(64),
                    "tokenizer_sha256":"d".repeat(64)},"windows":{"w":digest(b"[1,2]")}},
            "windows":{"w":{"hidden":{"path":"layer003/w.bin","bytes":bytes.len(),"sha256":digest(&bytes)},"blocks":[]}}});
        std::fs::write(root.join("latest.json"), serde_json::to_vec(&state)?)?;
        Ok(panel)
    }

    #[test]
    fn flagged_reference_validates_native_source_tokens_extent_and_hash() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let panel = fixture(tmp.path())?;
        let loaded = reference(tmp.path(), &panel, "w", 3)?;
        assert_eq!(loaded.tokens, [1, 2]);
        assert_eq!(loaded.hidden.len(), 2 * HIDDEN);
        assert!(reference(tmp.path(), &panel, "w", 2).is_err());
        assert!(reference(tmp.path(), &panel, "../w", 3).is_err());
        let mut changed = panel.clone(); changed["windows"][0]["tokens"][0] = json!(3);
        assert!(reference(tmp.path(), &changed, "w", 3).is_err());
        let original = std::fs::read(tmp.path().join("latest.json"))?;
        for pointer in ["/diagnostic_prefix", "/binding/state_dtype", "/binding/set_sha256",
            "/binding/source_pin/revision", "/binding/source_pin/modeling_kimi_linear_sha256",
            "/binding/source_pin/reference_runner_sha256", "/binding/windows/w",
            "/binding/arithmetic/torch_matmul_tf32", "/binding/arithmetic/triton_dot_precision",
            "/binding/snapshot_identity/snapshot_revision", "/binding/snapshot_identity/config_sha256",
            "/windows/w/hidden/bytes", "/windows/w/hidden/path", "/windows/w/hidden/sha256"] {
            let mut state: Value = serde_json::from_slice(&original)?;
            *state.pointer_mut(pointer).unwrap() = Value::Null;
            std::fs::write(tmp.path().join("latest.json"), serde_json::to_vec(&state)?)?;
            assert!(reference(tmp.path(), &panel, "w", 3).is_err(), "{pointer}");
        }
        std::fs::write(tmp.path().join("latest.json"), original)?;
        let mut bytes = std::fs::read(tmp.path().join("layer003/w.bin"))?; bytes[0] = 1;
        std::fs::write(tmp.path().join("layer003/w.bin"), bytes)?;
        assert!(reference(tmp.path(), &panel, "w", 3).is_err());
        Ok(())
    }

    #[test]
    fn diagnostic_cli_requires_reference_and_layer_and_refuses_scores() -> Result<()> {
        use clap::Parser;
        let args = ["cuteafd", "kimi-golden", "--snapshot", "/checkpoint", "--windows", "/panel",
            "--reference-state", "/state", "--window", "w", "--last-layer", "3"];
        let parsed = crate::cli::Cli::try_parse_from(args)?;
        let crate::cli::Commands::KimiGolden(parsed) = parsed.command else { panic!("Kimi command"); };
        assert_eq!(parsed.last_layer, 3);
        assert_eq!(parsed.reference_state, Path::new("/state"));
        for flag in ["--last-layer", "--reference-state"] {
            let at = args.iter().position(|arg| *arg == flag).unwrap();
            assert!(crate::cli::Cli::try_parse_from(args.iter().enumerate()
                .filter_map(|(index, arg)| (!(at..at + 2).contains(&index)).then_some(*arg))).is_err());
        }
        let parsed = crate::cli::Cli::try_parse_from(args.into_iter().chain(["--nll"]))?;
        let crate::cli::Commands::KimiGolden(parsed) = parsed.command else { panic!("Kimi command"); };
        assert!(run(parsed).unwrap_err().to_string().contains("diagnostic only"));
        Ok(())
    }

    #[test]
    fn ffn_cli_and_env_opt_in_are_explicit_and_recorded() -> Result<()> {
        use clap::{CommandFactory, FromArgMatches};
        let args = ["cuteafd", "kimi-golden", "--snapshot", "/checkpoint", "--windows", "/panel",
            "--reference-state", "/state", "--window", "w", "--last-layer", "3"];
        // Remove the inherited environment from the command, without mutating
        // process-global state shared with other CPU tests.
        for (value, tail, expected) in [(None, vec![], "direct"),
            (Some("true"), vec![], "compensated"),
            (Some("false"), vec![], "direct"),
            (Some("false"), vec!["--compensated-ffn"], "compensated")] {
            let command = crate::cli::Cli::command().mut_subcommand("kimi-golden", |command|
                command.mut_arg("compensated_ffn", |arg|
                    arg.env(None::<&str>).default_value(value.unwrap_or("false"))));
            let matches = command.try_get_matches_from(args.into_iter().chain(tail))?;
            let parsed = crate::cli::Cli::from_arg_matches(&matches)?;
            let crate::cli::Commands::KimiGolden(parsed) = parsed.command else { panic!("Kimi command"); };
            assert_eq!(parsed.ffn_mode().name(), expected);
            assert_eq!(parsed.provenance()["ffn_mode"], expected);
            let suffix = if expected == "compensated" { "_comp" } else { "" };
            assert_eq!(parsed.provenance()["gate_up_programs"], serde_json::json!([
                format!("kimik3_ffn_gate_up_i3072{suffix}"), format!("kimik3_ffn_gate_up_i16896{suffix}")]));
        }
        let command = crate::cli::Cli::command();
        let sub = command.get_subcommands().find(|command| command.get_name() == "kimi-golden").unwrap();
        let arg = sub.get_arguments().find(|arg| arg.get_id() == "compensated_ffn").unwrap();
        assert_eq!(arg.get_env().unwrap(), "CUTEAFD_KIMI_COMPENSATED_FFN");
        Ok(())
    }

    #[test]
    fn terminal_prefix_path_never_applies_output_mix_or_head() {
        let source = include_str!("device.rs");
        let terminal = source.split("pub(crate) fn terminal_residual").nth(1).unwrap()
            .split("pub(crate) fn residual").next().unwrap();
        assert!(terminal.contains("8 * self.lane.rows() * 7168 * 2"));
        assert!(terminal.contains("self.programs.add.launch(prefix, delta, output"));
        assert!(!terminal.contains("MixSite::Output"));
        let source = include_str!("engine.rs");
        let terminal = source.split("pub(crate) fn prefix_residual").nth(1).unwrap()
            .split("pub(crate) fn forward").next().unwrap();
        assert!(terminal.contains("super::layers::Output::Residual"));
        assert!(!terminal.contains(".project("));
        assert!(!terminal.contains("MixSite::Output"));
    }

    #[test]
    fn selected_hosted_layers_allow_dense_only_but_reject_missing_ids() -> Result<()> {
        assert!(hosted_layers("")?.is_empty());
        assert_eq!(hosted_layers("1,2,3")?, [1, 2, 3]);
        for invalid in [",", "1,", ",1", "1,,2", "-1", "a"] {
            assert!(hosted_layers(invalid).is_err(), "{invalid}");
        }
        Ok(())
    }

    #[test]
    fn reference_identity_cannot_be_rebound_by_the_panel() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let panel = fixture(tmp.path())?;
        for (key, value) in [("snapshot_revision", "wrong"), ("config_sha256", "e"),
            ("tokenizer_sha256", "f"), ("checkpoint", "wrong"), ("precision", "bf16")] {
            let mut changed = panel.clone();
            changed["reference_snapshot"] = serde_json::json!({key:value});
            assert!(reference(tmp.path(), &changed, "w", 3).is_err(), "{key}");
        }
        let mut changed = panel.clone();
        changed["tokenizer_sha256"] = serde_json::json!("e".repeat(64));
        assert!(reference(tmp.path(), &changed, "w", 3).is_err());
        Ok(())
    }

    #[test]
    fn shared_code_allowance_and_each_private_head_are_reserved() -> Result<()> {
        use cuteafd_loader::placement::{ArchContext, ProgramSet, inventory::loaded_code_allowance};
        let arch = ArchContext::for_device("sm_120", 96 << 30);
        let programs = ProgramSet::from_names(&["kimik3"], ["kimik3_kda_m64", "other_m64"]);
        for rank in 0..2 {
            let code = loaded_code_allowance("kimik3", "exl3", true, rank as u8, arch, Some(&programs));
            assert_eq!(code.basis, Basis::Formula);
            assert_eq!(code.bytes, arch.cublas_bytes + programs.module_bytes() + arch.peer_context_bytes);
            for lanes in [1, 2] {
                let demand = code_demand(rank, code.bytes, arch.cublas_bytes, lanes)?;
                assert_eq!(demand.basis, Basis::Formula);
                assert_eq!(demand.bytes, code.bytes + if rank == 1 { lanes as u64 * arch.cublas_bytes } else { 0 });
            }
        }
        assert!(code_demand(2, 0, 0, 1).is_err());
        assert!(code_demand(1, u64::MAX, 1, 1).is_err());
        Ok(())
    }

    #[test]
    fn peer_exchange_fixed_charges_exclude_solver_owned_hop_planes() -> Result<()> {
        for rows in [1, 64, 4096] {
            for diagnostics in [false, true] {
                let demands = exchange_demands(rows, diagnostics)?;
                let progress = if diagnostics {
                    (2 * std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>()).max(256)
                } else { 0 };
                for (rank, demand) in demands.iter().enumerate() {
                    assert_eq!(demand.gpu as usize, rank);
                    assert_eq!(demand.basis, Basis::Exact);
                    assert_eq!(demand.bytes as usize, (rows * HIDDEN * 2).max(256)
                        + 256 + 256 + 256 + progress);
                }
            }
        }
        assert!(exchange_demands(0, false).is_err());
        assert!(exchange_demands(4097, false).is_err());
        Ok(())
    }

    #[test]
    fn residual_metrics_are_finite_unscored_diagnostics() -> Result<()> {
        let (cos, l2) = compare(&[1.0, 2.0], &[1.0, 2.0])?;
        assert!((cos - 1.0).abs() < 1e-14 && l2 == 0.0);
        assert!(compare(&[f32::NAN], &[1.0]).is_err());
        assert!(compare(&[1.0], &[1.0, 2.0]).is_err());
        assert!(compare(&[0.0], &[0.0]).is_err());
        Ok(())
    }
}
