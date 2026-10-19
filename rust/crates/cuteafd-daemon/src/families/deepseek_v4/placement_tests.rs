//! `planner_equals_runtime_deepseek_v4`: `cuteafd plan --layout` and
//! serve-dsv4's admission build their placement requests on separate paths
//! (planned charges vs one measured sample per GPU) and must solve to the
//! same placement. The runtime side runs serve-dsv4's own `admission::request`
//! from parsed serve arguments with a fake CUDA sample: each GPU's total
//! minus exactly what the planner says is already loaded at that point
//! (context, modules, weights, embedding, drafter).
use super::{admission, EngineArgs};
use clap::Parser;
use cuteafd_core::memory_layout::{Category, MemoryLayout};
use cuteafd_loader::placement::{solve, Baseline, Onboard, Placement};
use cuteafd_loader::plan::{layout::LayoutOptions, ExpertPlacement, PlanInput, PlanOptions};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    prefix: crate::shared::prefix::PrefixArgs,
}

struct Case<'a> {
    snapshot: &'a Path,
    manifest: &'a Path,
    rtx: usize,
    sparks: usize,
    context: u64,
    budget: u64,
    /// `None`: the family default (no flag on either side).
    onboard: Option<Onboard>,
    dspark: bool,
    exchange_f32: bool,
    peer_budget: Option<u64>,
    /// `--attention-placement` on both sides (`None`: auto).
    attention: Option<cuteafd_loader::placement::AttentionPlacement>,
}

/// One checkpoint opened and parsed once for every case of a test: the
/// planner's [`PlanInput`] (its shard headers and routed-expert catalog) and
/// serve-dsv4's own catalog, which differs only when the launch does not keep
/// the checkpoint's bundled draft. The sweep re-opened and re-validated both
/// per case; sharing them is what makes it cheap.
struct Shared {
    input: PlanInput,
    /// The runtime catalog of a launch that drops the bundled draft.
    draft_off: OnceLock<anyhow::Result<cuteafd_loader::OfficialV41Catalog>>,
}

impl Shared {
    fn open(snapshot: &Path) -> anyhow::Result<Self> {
        let input = PlanInput::open(snapshot)?;
        anyhow::ensure!(input.checkpoint().missing_shards.is_empty(),
            "incomplete V4 checkpoint {}: unreadable shards {:?}", snapshot.display(), input.checkpoint().missing_shards);
        input.expert_catalog()?;
        Ok(Self { input, draft_off: OnceLock::new() })
    }

    /// serve-dsv4's own catalog for this launch shape: the planner's, when the
    /// launch keeps the checkpoint's bundled draft (`select_draft` is then a
    /// no-op), else the stripped catalog, read and stripped once per snapshot.
    fn catalog(&self, dspark: bool) -> anyhow::Result<&cuteafd_loader::OfficialV41Catalog> {
        if dspark { return self.input.expert_catalog(); }
        match self.draft_off.get_or_init(|| {
            let mut catalog = cuteafd_loader::read_expert_catalog(&self.input.checkpoint().snapshot)?;
            catalog.select_draft(cuteafd_loader::plan::spec::DraftOwnership::selected("deepseek_v4", "off"))?;
            Ok(catalog)
        }) {
            Ok(catalog) => Ok(catalog),
            Err(error) => Err(anyhow::anyhow!("{error:#}")),
        }
    }
}

fn planned(shared: &Shared, case: &Case<'_>) -> (MemoryLayout, bool, Vec<String>) {
    let report = shared.input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(case.sparks),
        layout: Some(LayoutOptions { rtx_bytes: (0..case.rtx).map(|g| if g == 1 { case.peer_budget.unwrap_or(case.budget) } else { case.budget }).collect(), context_tokens: case.context,
            // Retained measured fixtures and the direct EngineArgs default are C8.
            concurrency: 8, workspace_manifest: Some(case.manifest.to_path_buf()), onboard: case.onboard, exchange_f32: case.exchange_f32,
            native_mtp_layers: if case.dspark { 3 } else { 0 }, headroom_bytes: 0, ..Default::default() }),
        attention_placement: case.attention, ..Default::default() }).unwrap();
    let hints = report.hints.iter().map(|h| h.what.clone()).collect();
    (report.memory_layout.unwrap(), report.placement_supported, hints)
}

/// serve-dsv4's admission over the planner's own loaded bytes as the sample.
fn runtime(shared: &Shared, case: &Case<'_>, layout: &MemoryLayout) -> anyhow::Result<Placement> {
    runtime_with_tp1(shared, case, layout, true)
}

fn runtime_with_tp1(shared: &Shared, case: &Case<'_>, layout: &MemoryLayout, tp1_available: bool)
    -> anyhow::Result<Placement> {
    runtime_with_upload_check(shared, case, layout, tp1_available, false)
}

fn runtime_with_upload_check(shared: &Shared, case: &Case<'_>, layout: &MemoryLayout, tp1_available: bool,
    check_upload: bool) -> anyhow::Result<Placement> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(case.manifest)?)?;
    let catalog = shared.catalog(case.dspark)?;
    let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(case.snapshot, 1)?;
    let family = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
    let peers = (0..case.sparks).map(|r| format!("10.0.0.{}:1970{r}", r + 1)).collect::<Vec<_>>().join(",");
    let mut argv = vec!["serve".to_string(), "--snapshot".into(), case.snapshot.display().to_string(),
        "--native-lib".into(), "/nonexistent/libcuteafd.so".into(), "--peers".into(), peers];
    if let Some(onboard) = case.onboard { argv.extend(["--rtx-expert-layers".into(), onboard.to_string()]); }
    if case.rtx == 2 { argv.extend(["--split-device".into(), "1".into()]); }
    if case.dspark { argv.push("--dspark".into()); }
    if let Some(attention) = case.attention { argv.extend(["--attention-placement".into(), attention.to_string()]); }
    let cli = Cli::try_parse_from(argv)?;
    // The selected model owns caches only for its retained dSpark stages.
    let stages = (0..).take_while(|stage| catalog.tensors().iter()
        .any(|t| t.metadata.name.starts_with(&format!("mtp.{stage}.")))).count();
    let gpus = layout.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx)
        .enumerate().map(|(gpu, device)| {
            let loaded: u64 = device.items.iter().filter(|i| matches!(i.category,
                Category::Embedding) || i.group == "context+modules")
                .map(|i| i.bytes).sum();
            let budget = if gpu == 1 { case.peer_budget.unwrap_or(case.budget) } else { case.budget };
            (budget, Baseline::Measured { free_bytes: budget - loaded })
        }).collect();
    // The planner's EXL3 arena from the same capacity manifests serve-dsv4 reads.
    let expert_workspace = layout_expert_workspace(case, &catalog)?;
    let inputs = admission::Inputs { cfg: &cfg, catalog: &catalog, manifest: &manifest, family, gpus,
        loaded_modes: None, cache_stages: stages, prefill_rows: manifest["capacities"]["prefill_rows"].as_u64().unwrap() as usize,
        decode_rows: manifest["capacities"]["decode_rows"].as_u64().unwrap() as usize,
        max_context: case.context as usize, prefix: Some(&cli.prefix), expert_workspace: tp1_available.then_some(expert_workspace),
        tp2_workspace: if case.rtx == 2 { cuteafd_loader::serving_capacity::deepseek_v4_tp2_workspace(
            &catalog, Some(case.manifest), manifest["capacities"]["prefill_rows"].as_u64().unwrap()
                .max(manifest["capacities"]["decode_rows"].as_u64().unwrap())).ok().map(|b| [b; 2]) } else { None },
        exchange_f32: case.exchange_f32 };
    let request = admission::request(&cli.engine, &inputs)?;
    assert_eq!(request.hops.terminal, cuteafd_loader::placement::TerminalHome::Head);
    let placement = solve(&request)?;
    assert_eq!(placement.residual.last(), Some(&cuteafd_loader::placement::ResidualHome::Owned(0)),
        "V4 final residual must live on the rank0 head, not on both ranks");
    cuteafd_loader::placement::LayerExecutionPlan::new(&placement, request.hops)?;
    if check_upload {
        let modes = placement.layers.iter().map(|layer| layer.mode).collect::<Vec<_>>();
        let weights = cuteafd_loader::families::deepseek_v4::resident::WeightInventory::from_headers(
            &cfg, case.rtx, case.dspark, catalog.tensors().iter().map(|t| &t.metadata))?;
        let loaded = weights.loaded_bytes(&modes, case.rtx)?;
        let measured = inputs.gpus.iter().zip(loaded).map(|(&(total, baseline), bytes)| {
            let Baseline::Measured { free_bytes } = baseline else { unreachable!() };
            Ok((total, Baseline::Measured { free_bytes: free_bytes.checked_sub(bytes)
                .ok_or_else(|| anyhow::anyhow!("fixture weight uploads exceed free memory"))? }))
        }).collect::<anyhow::Result<Vec<_>>>()?;
        let after = admission::Inputs { gpus: measured, loaded_modes: Some(&modes), ..inputs };
        let mut request = admission::request(&cli.engine, &after)?;
        request.pool.pin_selected(placement.pool_tokens);
        request.onboard = Onboard::Layers(placement.onboard_layers);
        let checked = cuteafd_loader::placement::solve_frozen(&request, &placement)?;
        assert_eq!(checked, placement, "pre-upload/frozen measured admission differs");
    }
    Ok(placement)
}

/// serve-dsv4's `local::workspace_bytes` without CUDA: the EXL3 package's
/// capacity arenas plus BF16 output rows, or the native export's scratch.
fn layout_expert_workspace(case: &Case<'_>, catalog: &cuteafd_loader::OfficialV41Catalog) -> anyhow::Result<u64> {
    let shape = *catalog.routed_experts();
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(case.manifest)?)?;
    let rows = manifest["capacities"]["prefill_rows"].as_u64().unwrap()
        .max(manifest["capacities"]["decode_rows"].as_u64().unwrap());
    match catalog.exl3() {
        Some(exl3) => {
            let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
            let family = if shape.hidden == 4096 { "dsv4f" } else { "dsv4p" };
            let root = case.manifest.parent().unwrap().join("exl3").join(format!("exl3-{family}-k{tiers}/rtx-tp1"));
            // As local::workspace_bytes: every capacity up to the first at or above the rows, once.
            let manifests = cuteafd_loader::placement::inventory::exl3_capacities(rows).into_iter()
                .map(|n| -> anyhow::Result<serde_json::Value> {
                    Ok(serde_json::from_slice(&std::fs::read(root.join(format!("m{n}/v41_exl3.json")))?)?)
                }).collect::<anyhow::Result<Vec<_>>>()?;
            Ok(cuteafd_loader::serving_capacity::exl3_workspace_bytes(&manifests, true)? + rows * shape.hidden as u64 * 2)
        }
        None => Ok(cuteafd_loader::serving_capacity::deepseek_v4_native_workspace(shape.hidden as u64,
            shape.intermediate as u64, shape.experts as u64, shape.topk as u64, rows)?),
    }
}

/// The planner's layout carries the runtime placement's items, pool and ranges.
fn assert_equal(shared: &Shared, case: &Case<'_>, label: &str) -> Placement {
    let (layout, supported, hints) = planned(shared, case);
    assert!(supported, "{label}: planner refused: {:?} {:?}", layout.notes, hints);
    let runtime = runtime(shared, case, &layout).unwrap_or_else(|e| panic!("{label}: runtime admission: {e:#}"));
    assert_eq!(layout.pool_tokens, runtime.pool_tokens, "{label}: pool");
    for (gpu, items) in runtime.items.iter().enumerate() {
        let device = &layout.devices[gpu];
        for item in items {
            assert!(device.items.contains(item), "{label}: rtx{gpu} lacks {item:?}");
        }
        let budget = if gpu == 1 { case.peer_budget.unwrap_or(case.budget) } else { case.budget };
        let reserve = budget - device.capacity_bytes;
        assert!(reserve > 0, "{label}: rtx{gpu} has no reserve");
    }
    let notes = runtime.expert_ranges.iter().enumerate().map(|(gpu, r)|
        format!("rtx{gpu}: {} local expert layers ({}..{})", r.layers, r.first, r.first + r.layers)).collect::<Vec<_>>();
    for note in notes { assert!(layout.notes.contains(&note), "{label}: planner lacks {note}"); }
    // Per-layer ownership: today's V4 is all head split (2 RTX) or all GPU0,
    // intermediate residuals replicated under the split, terminal on rank0,
    // no charged hop, and the engine accepts the placement.
    assert_eq!(runtime.residual.len(), runtime.layers.len() + 1, "{label}: residual homes");
    assert!(runtime.hops.iter().all(|h| !h.charged()), "{label}: V4 hops");
    assert!(runtime.items.iter().flatten().all(|i| i.group != "residual hops"), "{label}: hop buffers");
    let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(case.snapshot, 1).unwrap();
    super::check_modes(&runtime, case.rtx == 2, &cfg).unwrap_or_else(|e| panic!("{label}: {e:#}"));
    assert!(super::check_modes(&runtime, case.rtx != 2, &cfg).is_err(), "{label}: the other engine shape is refused");
    if let Some(t) = runtime.tp2 {
        assert!(runtime.expert_ranges.iter().all(|r| r.layers == 0));
        assert!(layout.notes.contains(&format!("rtx0/rtx1: {} TP2 expert layer halves ({}..{})", t.layers, t.first, t.first + t.layers)));
        for gpu in 0..2 {
            let bytes: u64 = layout.devices[gpu].items.iter().filter(|i| i.format == "tp2").map(|i| i.bytes).sum();
            assert_eq!(bytes, t.peak_bytes[gpu], "{label}: rtx{gpu} TP2 arena");
        }
    }
    runtime
}

fn synthetic(dir: &Path) -> PathBuf {
    cuteafd_loader::plan::testing::write_v4_snapshot(dir);
    let manifest = serde_json::json!({"capacities": {"prefill_rows": 4096, "decode_rows": 64, "max_context": 1048576},
        "programs": [{"family": "dsv4f", "name": "dsv4f_sparse_mla_decode_c128_m64", "params": {"indexed_width": 8192}},
            {"name": "dsv4f_index_topk_decode_m64", "scratch_bytes_at_capacity": {"scratch": 8653824}},
            {"name": "dsv4f_index_topk_prefill_m4096", "scratch_bytes_at_capacity": {"scratch": 558007296}}]});
    let path = dir.join("PROGRAMS.json");
    std::fs::write(&path, manifest.to_string()).unwrap();
    path
}

/// [`synthetic`] with the C4 token-split programs a context build exports.
fn synthetic_context(dir: &Path) -> PathBuf {
    let path = synthetic(dir);
    let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let programs = manifest["programs"].as_array_mut().unwrap();
    for (name, scratch) in [("dsv4f_scored_index_topk_decode_m64", 9_178_112), ("dsv4f_dsa_candidate_merge", 0),
        ("dsv4f_lse_combine2", 0), ("dsv4f_paged_staging_gather_c4", 0), ("dsv4f_paged_staging_gather_index", 0),
        ("dsv4f_sparse_mla_partial_h0_s4_m64", 4_000_000), ("dsv4f_sparse_mla_partial_h32_s4_m64", 4_000_000)] {
        programs.push(serde_json::json!({"name": name, "scratch_bytes_at_capacity": {"scratch": scratch}}));
    }
    std::fs::write(&path, manifest.to_string()).unwrap();
    path
}

/// C4 token-split admission (#58): the planner and serve-dsv4 agree under `context`, auto resolves
/// `context` where the build carries the programs, only C4 layers split, and the context
/// buffers are charged on both GPUs while the C4 records halve.
#[test]
fn planner_equals_runtime_deepseek_v4_fixture_context() {
    use cuteafd_loader::placement::{AttentionPlacement, LayerMode};
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic_context(dir.path());
    let shared = Shared::open(dir.path()).unwrap();
    let case = |attention| Case { snapshot: dir.path(), manifest: &manifest, rtx: 2, sparks: 2, context: 1_048_576,
        budget: 24 << 30, onboard: None, dspark: false, exchange_f32: false, peer_budget: None, attention };
    let heads = assert_equal(&shared, &case(Some(AttentionPlacement::Heads)), "context fixture heads");
    let auto = assert_equal(&shared, &case(None), "context fixture auto");
    let context = assert_equal(&shared, &case(Some(AttentionPlacement::Context)), "context fixture context");
    assert_eq!(auto, context, "auto is context where the build carries the programs");
    assert_ne!(auto, heads);
    assert_eq!(context.attention_placement, AttentionPlacement::Context);
    let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(dir.path(), 1).unwrap();
    for (layer, (a, &ratio)) in context.layers.iter().zip(&cfg.compress_ratios).enumerate() {
        assert_eq!(a.mode, if ratio == 4 { LayerMode::ContextSplit } else { LayerMode::HeadSplit }, "layer {layer}");
    }
    assert!(super::check_modes(&context, true, &cfg).unwrap());
    assert!(!super::check_modes(&heads, true, &cfg).unwrap());
    let bytes = |p: &Placement, gpu: usize, group: &str| p.items[gpu].iter().filter(|i| i.group == group)
        .map(|i| i.bytes).sum::<u64>();
    for gpu in 0..2 {
        for group in ["context staging", "context exchange", "context workspace"] {
            assert!(bytes(&context, gpu, group) > 0 && bytes(&heads, gpu, group) == 0, "rtx{gpu} {group}");
        }
        // Per 256-token unit: the two C4 layers' 45,888 B halve on each GPU, the C128 layer's 1,728 B
        // stays whole (the window-only layer has none).
        let per_unit = |p: &Placement| bytes(p, gpu, "records") / (p.pool_tokens / 256);
        assert_eq!(per_unit(&heads) - per_unit(&context), 45_888, "rtx{gpu} C4 records halve");
    }
    // A build without the programs refuses an explicit context and runs heads on auto.
    let plain = synthetic(&dir.path().join("plain").tap_dir());
    let mut refused = case(Some(AttentionPlacement::Context));
    refused.manifest = &plain;
    let (layout, _, _) = planned(&shared, &refused);
    assert!(runtime(&shared, &refused, &layout).is_err(), "explicit context without its programs");
}

#[test]
fn preload_equals_frozen_measured_admission_after_native_uploads() {
    use cuteafd_loader::placement::AttentionPlacement;
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic_context(dir.path());
    for dspark in [false, true] {
        if dspark { cuteafd_loader::plan::testing::write_v4_snapshot_with_dspark(dir.path(), 3); }
        let shared = Shared::open(dir.path()).unwrap();
        for rtx in [1, 2] {
            for attention in [AttentionPlacement::Heads, AttentionPlacement::Context] {
                if rtx == 1 && attention == AttentionPlacement::Context { continue; }
                for onboard in [Onboard::Auto, Onboard::Layers(2)] {
                    let case = Case { snapshot: dir.path(), manifest: &manifest, rtx, sparks: 2,
                        context: 1_048_576, budget: 32 << 30, onboard: Some(onboard), dspark,
                        exchange_f32: true, peer_budget: Some(30 << 30), attention: Some(attention) };
                    let (layout, supported, hints) = planned(&shared, &case);
                    assert!(supported, "preload fixture dspark={dspark} rtx={rtx} attention={attention:?} onboard={onboard:?}: {:?} {hints:?}", layout.notes);
                    runtime_with_upload_check(&shared, &case, &layout, true, true).unwrap();
                }
            }
        }
    }
}

trait TapDir { fn tap_dir(self) -> Self; }
impl TapDir for PathBuf {
    fn tap_dir(self) -> Self { std::fs::create_dir_all(&self).unwrap(); self }
}

/// Always runs: the fixture's four layers at Flash geometry, one and two
/// GPUs, both extents, automatic and fixed onboard.
#[test]
fn planner_equals_runtime_deepseek_v4_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic(dir.path());
    let shared = Shared::open(dir.path()).unwrap();
    for (rtx, budget) in [(1, 24u64 << 30), (2, 24 << 30)] {
        for context in [131_072, 1_048_576] {
            for onboard in [None, Some(Onboard::Auto), Some(Onboard::Layers(2)), Some(Onboard::Layers(0)), Some(Onboard::Fraction(0.5)),
                Some(Onboard::ExpertsFirst { pool_floor: 262_144 })] {
                let case = Case { snapshot: dir.path(), manifest: &manifest, rtx, sparks: 2, context, budget,
                    onboard, dspark: false, exchange_f32: false, peer_budget: None, attention: None };
                let placement = assert_equal(&shared, &case, &format!("fixture rtx{rtx} {context} {onboard:?}"));
                // The default: pool first for Flash on one or two RTX (`default_onboard`).
                let effective = onboard.unwrap_or(cuteafd_loader::placement::families::deepseek_v4::default_onboard());
                match effective {
                    Onboard::ExpertsFirst { .. } => assert!(placement.pool_tokens >= 262_144, "experts first keeps a 262K pool"),
                    Onboard::Auto => assert_eq!(placement.pool_tokens, 1 << 20, "24 GiB cards target 1M"),
                    Onboard::Layers(n) => {
                        assert_eq!(placement.onboard_layers, n);
                        assert!(placement.pool_tokens >= 1 << 20, "a fixed onboard fills the pool past the target");
                    }
                    Onboard::Fraction(_) => assert_eq!(placement.onboard_layers, 2),
                }
            }
        }
    }
}

#[test]
fn planner_equals_runtime_deepseek_v4_fixture_tp2_asym_dspark_f32() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic(dir.path());
    cuteafd_loader::plan::testing::write_v4_snapshot_with_dspark(dir.path(), 3);
    let shared = Shared::open(dir.path()).unwrap();
    for onboard in [None, Some(Onboard::Auto), Some(Onboard::Layers(2)), Some(Onboard::Fraction(0.5)),
        Some(Onboard::ExpertsFirst { pool_floor: 262_144 })] {
        for exchange_f32 in [false, true] {
            let case = Case { snapshot: dir.path(), manifest: &manifest, rtx: 2, sparks: 2,
                context: 1_048_576, budget: 32 << 30, peer_budget: Some(30 << 30),
                onboard, dspark: true, exchange_f32, attention: None };
            let p = assert_equal(&shared, &case, &format!("fixture TP2 dSpark {onboard:?} f32={exchange_f32}"));
            assert!(p.expert_ranges.iter().all(|r| r.layers == 0));
            assert!(p.expert_ranges[0].peak_bytes > 0);
            assert_eq!(p.expert_ranges[1].peak_bytes, 0);
            assert_eq!(p.tp2.unwrap().layers, p.onboard_layers);
        }
    }
}

#[test]
fn planner_equals_runtime_v4_bundled_draft_off() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic(dir.path());
    cuteafd_loader::plan::testing::write_v4_snapshot_with_dspark(dir.path(), 3);
    let shared = Shared::open(dir.path()).unwrap();
    for rtx in [1, 2] {
        let case = Case { snapshot: dir.path(), manifest: &manifest, rtx, sparks: 2,
            context: 1_048_576, budget: 32 << 30, peer_budget: None,
            onboard: Some(Onboard::Layers(0)), dspark: false, exchange_f32: false, attention: None };
        let p = assert_equal(&shared, &case, &format!("bundled draft off rtx{rtx}"));
        assert!(p.expert_ranges.iter().all(|range| range.peak_bytes == 0));
        let (layout, _, _) = planned(&shared, &case);
        assert!(!layout.devices.iter().flat_map(|device| &device.items).any(|item|
            item.group == "speculator" || item.group == "speculator_expert"));
    }
}

#[test]
fn tp2_backbone_admission_does_not_require_tp1_kernels() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic(dir.path());
    let shared = Shared::open(dir.path()).unwrap();
    for onboard in [None, Some(Onboard::Auto), Some(Onboard::Layers(0)), Some(Onboard::Layers(2)),
        Some(Onboard::ExpertsFirst { pool_floor: 262_144 })] {
        let case = Case { snapshot: dir.path(), manifest: &manifest, rtx: 2, sparks: 2,
            context: 1_048_576, budget: 24 << 30, peer_budget: None, onboard, dspark: false, exchange_f32: false,
            attention: None };
        let (layout, supported, _) = planned(&shared, &case);
        assert!(supported);
        assert_eq!(runtime_with_tp1(&shared, &case, &layout, false).unwrap(),
            runtime(&shared, &case, &layout).unwrap());
    }
}

#[test]
fn removed_peer_expert_ranges_has_a_clear_error() {
    let err = Cli::try_parse_from(["serve", "--snapshot", "fixture", "--native-lib", "none", "--peer-expert-ranges"])
        .err().expect("old flag must refuse").to_string();
    assert!(err.contains("GPU1 whole-layer expert ranges were replaced by TP2 halves"), "{err}");
}

fn snapshot(model: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HOME").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/sparknest/hf-home")).join("hub")
        .join(format!("models--{}", model.replace('/', "--"))).join("snapshots");
    selected_snapshot(&hub)
}

fn selected_snapshot(hub: &Path) -> Option<PathBuf> {
    // The published revision wins; directory iteration order is not a pin.
    if let Ok(revision) = std::fs::read_to_string(hub.parent()?.join("refs/main")) {
        let selected = hub.join(revision.trim());
        return selected.join("model.safetensors.index.json").is_file().then_some(selected);
    }
    let mut candidates: Vec<_> = std::fs::read_dir(hub).ok()?.filter_map(|e| e.ok())
        .map(|e| e.path()).filter(|p| p.join("model.safetensors.index.json").is_file()).collect();
    candidates.sort();
    candidates.into_iter().next()
}

#[test]
fn live_snapshot_selection_is_pinned_and_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let hub = dir.path().join("snapshots");
    for revision in ["z", "a"] {
        std::fs::create_dir_all(hub.join(revision)).unwrap();
        std::fs::write(hub.join(revision).join("model.safetensors.index.json"), "{}").unwrap();
    }
    assert_eq!(selected_snapshot(&hub), Some(hub.join("a")));
    std::fs::create_dir_all(dir.path().join("refs")).unwrap();
    std::fs::write(dir.path().join("refs/main"), "z\n").unwrap();
    assert_eq!(selected_snapshot(&hub), Some(hub.join("z")));
    std::fs::write(dir.path().join("refs/main"), "missing").unwrap();
    assert_eq!(selected_snapshot(&hub), None);
}

#[test]
fn equality_input_refuses_missing_or_truncated_shards() {
    for truncated in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        cuteafd_loader::plan::testing::write_v4_snapshot(dir.path());
        let shard = dir.path().join("model-00001-of-00001.safetensors");
        if truncated {
            std::fs::OpenOptions::new().write(true).open(&shard).unwrap().set_len(9).unwrap();
        } else {
            std::fs::remove_file(&shard).unwrap();
        }
        let error = match Shared::open(dir.path()) {
            Ok(_) => panic!("partial checkpoint was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("incomplete V4 checkpoint"), "{error:#}");
    }
}

fn inputs_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("CUTEAFD_V4_PLACEMENT_INPUTS").map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/cuteafd/builds/v4-placement/plans-input")))?;
    dir.is_dir().then_some(dir)
}

/// The real checkpoints (headers only) with the release image's program and
/// EXL3 package manifests: Flash and Pro EXL3 K2, natural minimum and
/// maximum, compiled extents 131,072 and 1,048,576, automatic and a fixed
/// onboard of half the routed layers (Flash max). Skips when the snapshots
/// or manifests are not on this host.
#[test]
fn planner_equals_runtime_deepseek_v4() {
    const CHILD: &str = "CUTEAFD_V4_EQUALITY_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Bound real-checkpoint/FUSE work without dropping any of the 80 cases.
        let started = std::time::Instant::now();
        let output = std::process::Command::new("timeout").args(["--kill-after=5s", "300s"])
            .arg(std::env::current_exe().unwrap())
            .args(["--exact", "families::deepseek_v4::placement_tests::planner_equals_runtime_deepseek_v4",
                "--nocapture", "--test-threads=1"])
            .env(CHILD, "1").output().unwrap();
        assert!(output.status.success(),
            "V4 equality gate failed after {:?}: status={} HOME={:?} HF_HOME={:?} CUTEAFD_V4_PLACEMENT_INPUTS={:?}\n{}\n{}",
            started.elapsed(), output.status, std::env::var_os("HOME"), std::env::var_os("HF_HOME"),
            std::env::var_os("CUTEAFD_V4_PLACEMENT_INPUTS"), String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        eprintln!("V4 equality sweep: {:?}\n{}{}", started.elapsed(),
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        return;
    }
    let (Some(flash), Some(pro), Some(inputs)) = (snapshot("deepseek-ai/DeepSeek-V4-Flash-0731"),
        snapshot("wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1"), inputs_dir()) else {
        eprintln!("planner_equals_runtime_deepseek_v4: V4 snapshots or manifests absent; skipped");
        return;
    };
    let budget = (95.5 * (1u64 << 30) as f64) as u64;
    // One open per checkpoint for all 40 of its cases: the plan and the runtime
    // admission both read its shard headers and routed-expert catalog.
    let sweep_started = std::time::Instant::now();
    eprintln!("live V4 equality: opening Flash checkpoint {}", flash.display());
    let flash = Shared::open(&flash).unwrap();
    eprintln!("live V4 equality: opening Pro checkpoint {} after {:?}", pro.display(), sweep_started.elapsed());
    let pro = Shared::open(&pro).unwrap();
    eprintln!("live V4 equality: checkpoints ready after {:?}", sweep_started.elapsed());
    let mut cases = 0;
    for context in [131_072u64, 1_048_576] {
        for (model, shared, rtx, sparks) in [("flash", &flash, 1, 2), ("flash", &flash, 2, 4),
            ("pro", &pro, 1, 4), ("pro", &pro, 2, 6)] {
            let snapshot: &Path = &shared.input.checkpoint().snapshot;
            let manifest = inputs.join(format!("{model}-{rtx}-{context}/PROGRAMS.json"));
            for onboard in [None, Some(Onboard::Auto), Some(Onboard::ExpertsFirst { pool_floor: 262_144 }),
                Some(Onboard::Layers(2)), Some(Onboard::Fraction(0.03))] {
                for asymmetric in [false, true] {
                    let case = Case { snapshot, manifest: &manifest, rtx, sparks, context, budget,
                        onboard, dspark: true, exchange_f32: asymmetric,
                        peer_budget: asymmetric.then_some(budget - (2 << 30)), attention: None };
                    eprintln!("live V4 equality: starting case {} after {:?}: {model} rtx{rtx} {context} {onboard:?} asym={asymmetric}",
                        cases + 1, sweep_started.elapsed());
                    let placement = assert_equal(shared, &case, &format!("{model} rtx{rtx} {context} {onboard:?} asym={asymmetric}"));
                    cases += 1;
                    if rtx == 2 {
                        assert!(placement.expert_ranges.iter().all(|r| r.layers == 0));
                        assert_eq!(placement.tp2.unwrap().layers, placement.onboard_layers);
                        assert!(placement.expert_ranges[0].peak_bytes > 0);
                        assert_eq!(placement.expert_ranges[1].peak_bytes, 0);
                        if onboard.is_none() || onboard == Some(Onboard::Auto) { assert_eq!(placement.pool_tokens, 2 << 20); }
                    } else { assert!(placement.tp2.is_none()); }
                }
            }
        }
    }
    assert_eq!(cases, 80);
    eprintln!("live V4 equality: {cases} cases in {:?}", sweep_started.elapsed());
}

#[test]
fn planner_equals_runtime_deepseek_v4_missing_tp2_package() {
    let (Some(pro), Some(inputs)) = (snapshot("wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1"), inputs_dir()) else {
        eprintln!("missing TP2 package test: real EXL3 snapshot/manifests absent; skipped");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let source = inputs.join("pro-2-1048576");
    let manifest = dir.path().join("PROGRAMS.json");
    std::fs::copy(source.join("PROGRAMS.json"), &manifest).unwrap();
    let package = dir.path().join("exl3/exl3-dsv4p-k23");
    std::fs::create_dir_all(&package).unwrap();
    std::os::unix::fs::symlink(source.join("exl3/exl3-dsv4p-k23/rtx-tp1"), package.join("rtx-tp1")).unwrap();
    let shared = Shared::open(&pro).unwrap();
    for onboard in [None, Some(Onboard::Auto), Some(Onboard::Layers(0))] {
        let case = Case { snapshot: &pro, manifest: &manifest, rtx: 2, sparks: 6,
            context: 1_048_576, budget: (95.5 * (1u64 << 30) as f64) as u64,
            peer_budget: None, onboard, dspark: false, exchange_f32: false, attention: None };
        let p = assert_equal(&shared, &case, &format!("missing TP2 package {onboard:?}"));
        assert_eq!(p.onboard_layers, 0);
        assert!(p.tp2.is_none());
        assert!(p.layers.iter().all(|l| l.experts == cuteafd_loader::placement::ExpertHome::Spark));
        assert!(p.expert_ranges.iter().all(|r| r.layers == 0 && r.peak_bytes == 0));
    }
}

/// A hardware inventory fixture (`shared/placement_fixtures/<card>.json`),
/// recorded from serve-dsv4's `placement inventory` line on a real launch:
/// each GPU's CUDA total and the free bytes its admission sampled after
/// context, modules and weights.
#[derive(serde::Deserialize)]
struct Fixture {
    #[serde(default)]
    historical_tp1: bool,
    model: String,
    rtx: usize,
    sparks: usize,
    context: u64,
    /// `RTX_EXPERT_LAYERS` of the launch (`None`: the default).
    onboard: Option<String>,
    manifest: String,
    driver: String,
    gpus: Vec<FixtureGpu>,
    /// The launch's admitted pool and RTX expert layers (the runtime's answer).
    pool_tokens: u64,
    onboard_layers: usize,
}

#[derive(serde::Deserialize)]
struct FixtureGpu {
    total_bytes: u64,
    admission_free_bytes: u64,
    /// Preserve the raw launch sample when its code table predates the ready measurement.
    #[serde(default)]
    reserved_code_bytes: Option<u64>,
    #[serde(default)]
    untracked_at_admission_bytes: Option<u64>,
}

impl FixtureGpu {
    fn free_with_current_code(&self, code: u64) -> u64 {
        match (self.reserved_code_bytes, self.untracked_at_admission_bytes) {
            (Some(reserved), Some(untracked)) => {
                let loaded = cuteafd_loader::placement::inventory::LoadedCode {
                    family: "dsv4", experts: "*", split: true, rank: 0,
                    bytes: code, source: "fixture replay",
                };
                let context = cuteafd_loader::placement::inventory::ArchContext::for_device(
                    "sm_120", self.total_bytes).context_bytes;
                self.admission_free_bytes.checked_add(reserved).unwrap()
                    .checked_sub(loaded.pending(untracked, context)).unwrap()
            }
            (None, None) => self.admission_free_bytes,
            _ => panic!("fixture must record both pending-code inputs"),
        }
    }
}

#[test]
fn measured_fixture_replays_pending_code_without_changing_raw_sample() {
    let gpu = FixtureGpu { total_bytes: 101_973_491_712, admission_free_bytes: 93_570_006_176,
        reserved_code_bytes: Some(967_770_976), untracked_at_admission_bytes: Some(598_715_904) };
    assert_eq!(gpu.free_with_current_code(980_070_752), gpu.admission_free_bytes);
    assert_eq!(gpu.free_with_current_code(1_007_333_728), gpu.admission_free_bytes - 27_262_976);
    assert_eq!(gpu.free_with_current_code(0), gpu.admission_free_bytes + 967_770_976);
    assert_eq!(gpu.admission_free_bytes, 93_570_006_176);
}

const MIB: u64 = 1 << 20;

/// `planner_equals_runtime_deepseek_v4` against measured inventories: the
/// runtime side solves from the free bytes a real launch sampled (not the
/// planner's own loaded bytes), and the planner, at the same CUDA total, must
/// predict that sample within 64 MiB per GPU and resolve the same placement
/// (P1 could not see its 1.45M vs 1.18M Flash max gap: it fed the planner's
/// numbers back in). Skips fixtures whose snapshot or manifest is absent.
#[test]
fn planner_equals_runtime_deepseek_v4_measured() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shared/placement_fixtures");
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if !path.file_name().unwrap().to_string_lossy().starts_with("v4-") { continue; }
        let fixture: Fixture = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        if fixture.historical_tp1 { continue; }
        let label = path.file_stem().unwrap().to_string_lossy().into_owned();
        let manifest = PathBuf::from(shellexpand(&fixture.manifest));
        let (Some(snapshot), true) = (snapshot(&fixture.model), manifest.is_file()) else {
            eprintln!("{label}: snapshot or manifest absent; skipped");
            continue;
        };
        let shared = Shared::open(&snapshot).unwrap();
        let budget = fixture.gpus[0].total_bytes;
        assert!(fixture.gpus.iter().all(|g| g.total_bytes.abs_diff(budget) < 8 * MIB), "{label}: unequal GPUs");
        let onboard = fixture.onboard.as_deref().map(|o| o.parse::<Onboard>().unwrap());
        let case = Case { snapshot: &snapshot, manifest: &manifest, rtx: fixture.rtx, sparks: fixture.sparks,
            context: fixture.context, budget, onboard, dspark: true, exchange_f32: false,
            peer_budget: fixture.gpus.get(1).map(|g| g.total_bytes), attention: None };
        let (layout, supported, hints) = planned(&shared, &case);
        assert!(supported, "{label}: planner refused: {:?} {:?}", layout.notes, hints);
        let catalog = shared.input.expert_catalog().unwrap();
        let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(&snapshot, 1).unwrap();
        let code = cuteafd_loader::placement::families::deepseek_v4::code_bytes(
            cfg.dim, cuteafd_loader::placement::families::deepseek_v4::code_experts(&catalog), fixture.rtx);
        // Replay pending-code reservation, retaining the raw launch sample.
        for (gpu, measured) in fixture.gpus.iter().enumerate() {
            let device = &layout.devices[gpu];
            let planned_loaded: u64 = device.items.iter().filter(|i| matches!(i.category,
                Category::Weights | Category::Embedding | Category::Drafter) || i.group == "context+modules")
                .map(|i| i.bytes).sum();
            let measured_loaded = measured.total_bytes - measured.free_with_current_code(code[gpu]);
            let diff = planned_loaded as i64 - measured_loaded as i64;
            eprintln!("{label} rtx{gpu}: planned loaded {planned_loaded} measured {measured_loaded} diff {diff} \
                (driver {})", fixture.driver);
            assert!(diff.unsigned_abs() <= 64 * MIB, "{label} rtx{gpu}: planner baseline {diff:+} B from the \
                measured inventory");
        }
        // The runtime's admission over the measured sample.
        let runtime = runtime_measured(&shared, &case, &fixture).unwrap_or_else(|e| panic!("{label}: runtime: {e:#}"));
        assert_eq!(runtime.pool_tokens, fixture.pool_tokens, "{label}: fixture pool vs this build's runtime");
        assert_eq!(runtime.onboard_layers, fixture.onboard_layers, "{label}: fixture layers vs this build's runtime");
        // The planner resolves the same layers and a pool within the baseline difference.
        let notes = runtime.expert_ranges.iter().enumerate().map(|(gpu, r)|
            format!("rtx{gpu}: {} local expert layers ({}..{})", r.layers, r.first, r.first + r.layers));
        for note in notes { assert!(layout.notes.contains(&note), "{label}: planner lacks {note}"); }
        let unit: u64 = layout.devices[0].items.iter().filter(|i| i.group == "records").map(|i| i.bytes).sum::<u64>()
            / (layout.pool_tokens / 256).max(1);
        let slack = (64 * MIB).div_ceil(unit.max(1)) * 256;
        assert!(layout.pool_tokens.abs_diff(runtime.pool_tokens) <= slack, "{label}: pool {} planned vs {} measured",
            layout.pool_tokens, runtime.pool_tokens);
        if fixture.rtx == 2 {
            assert!(runtime.expert_ranges.iter().all(|r| r.layers == 0));
            assert_eq!(runtime.tp2.unwrap().layers, fixture.onboard_layers);
            // Re-solve the same measured pre-allocation inventory for every
            // retained onboard policy; no deleted two-RTX TP1 placement.
            for onboard in [None, Some(Onboard::Auto),
                Some(Onboard::ExpertsFirst { pool_floor: 262_144 }),
                Some(Onboard::Layers(2)), Some(Onboard::Fraction(0.03))] {
                let variant = Case { onboard, ..case };
                let (layout, supported, hints) = planned(&shared, &variant);
                assert!(supported, "{label} {onboard:?}: {hints:?}");
                let runtime = runtime_measured(&shared, &variant, &fixture).unwrap();
                let tp2 = runtime.tp2.unwrap();
                assert!(runtime.expert_ranges.iter().all(|r| r.layers == 0));
                assert!(layout.notes.contains(&format!(
                    "rtx0/rtx1: {} TP2 expert layer halves ({}..{})",
                    tp2.layers, tp2.first, tp2.first + tp2.layers)), "{label} {onboard:?}");
                assert!(layout.pool_tokens.abs_diff(runtime.pool_tokens) <= slack,
                    "{label} {onboard:?}: pool {} planned vs {} measured",
                    layout.pool_tokens, runtime.pool_tokens);
                if onboard.is_none() || onboard == Some(Onboard::Auto) {
                    assert_eq!(runtime.pool_tokens, 2 << 20);
                }
                if let Some(Onboard::Layers(n)) = onboard {
                    assert_eq!(runtime.onboard_layers, n);
                }
                eprintln!("{label} {onboard:?}: TP2 {} layers pool {}", tp2.layers, runtime.pool_tokens);
            }
        }
        checked += 1;
    }
    eprintln!("planner_equals_runtime_deepseek_v4_measured: {checked} fixtures");
}

fn shellexpand(path: &str) -> String {
    path.replace("~", &std::env::var("HOME").unwrap_or_default())
}

/// serve-dsv4's admission over a fixture's measured free bytes.
fn runtime_measured(shared: &Shared, case: &Case<'_>, fixture: &Fixture) -> anyhow::Result<Placement> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(case.manifest)?)?;
    let catalog = shared.catalog(case.dspark)?;
    let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(case.snapshot, 1)?;
    let family = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
    let peers = (0..case.sparks).map(|r| format!("10.0.0.{}:1970{r}", r + 1)).collect::<Vec<_>>().join(",");
    let mut argv = vec!["serve".to_string(), "--snapshot".into(), case.snapshot.display().to_string(),
        "--native-lib".into(), "/nonexistent/libcuteafd.so".into(), "--peers".into(), peers, "--dspark".into()];
    if let Some(onboard) = case.onboard { argv.extend(["--rtx-expert-layers".into(), onboard.to_string()]); }
    if case.rtx == 2 { argv.extend(["--split-device".into(), "1".into()]); }
    let cli = Cli::try_parse_from(argv)?;
    let stages = (0..).take_while(|stage| catalog.tensors().iter()
        .any(|t| t.metadata.name.starts_with(&format!("mtp.{stage}.")))).count();
    let code = cuteafd_loader::placement::families::deepseek_v4::code_bytes(
        cfg.dim, cuteafd_loader::placement::families::deepseek_v4::code_experts(&catalog), fixture.rtx);
    let gpus = fixture.gpus.iter().enumerate().map(|(gpu, g)| (g.total_bytes,
        Baseline::Measured { free_bytes: g.free_with_current_code(code[gpu]) })).collect();
    let expert_workspace = layout_expert_workspace(case, &catalog)?;
    let loaded_modes = vec![if case.rtx == 2 { cuteafd_loader::placement::LayerMode::HeadSplit }
        else { cuteafd_loader::placement::LayerMode::Whole { gpu: 0,
            ffn: cuteafd_loader::placement::FfnMode::Owner } }; cfg.n_layers];
    let inputs = admission::Inputs { cfg: &cfg, catalog: &catalog, manifest: &manifest, family, gpus,
        loaded_modes: Some(&loaded_modes), cache_stages: stages, prefill_rows: manifest["capacities"]["prefill_rows"].as_u64().unwrap() as usize,
        decode_rows: manifest["capacities"]["decode_rows"].as_u64().unwrap() as usize,
        max_context: case.context as usize, prefix: Some(&cli.prefix), expert_workspace: Some(expert_workspace),
        tp2_workspace: if case.rtx == 2 { cuteafd_loader::serving_capacity::deepseek_v4_tp2_workspace(
            &catalog, Some(case.manifest), manifest["capacities"]["prefill_rows"].as_u64().unwrap()
                .max(manifest["capacities"]["decode_rows"].as_u64().unwrap())).ok().map(|b| [b; 2]) } else { None },
        exchange_f32: case.exchange_f32 };
    Ok(solve(&admission::request(&cli.engine, &inputs)?)?)
}

/// P2's two-RTX fixture predates TP2 and its deleted GPU0-only routed layout.
/// Preserve its recorded sample and pending-code arithmetic, not its TP1 answer.
#[test]
fn historical_p2_flash_max_inventory_and_pending_code() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../shared/placement_fixtures/v4-flash-max.json")).unwrap();
    assert!(fixture.historical_tp1);
    assert_eq!((fixture.rtx, fixture.sparks), (2, 4));
    for (gpu, code) in fixture.gpus.iter().zip([560_200_800, 260_349_424]) {
        assert!(gpu.admission_free_bytes < gpu.total_bytes);
        let measured_used = gpu.total_bytes - gpu.admission_free_bytes;
        let historical = cuteafd_loader::placement::inventory::LoadedCode {
            family: "dsv4", experts: "*", split: true, rank: 0,
            bytes: code, source: "historical P2 TP1 sample",
        };
        let context = 586_416_128;
        assert_eq!(historical.pending(context + 100, context), code - 100);
        assert_eq!(historical.pending(context + code, context), 0);
        assert!(measured_used > context + code);
    }
}
