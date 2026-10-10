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
use cuteafd_loader::plan::{layout::LayoutOptions, plan, ExpertPlacement, PlanOptions};
use std::path::{Path, PathBuf};

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
    peer: bool,
}

fn planned(case: &Case<'_>) -> (MemoryLayout, bool, Vec<String>) {
    let report = plan(case.snapshot, &PlanOptions { placement: ExpertPlacement::from_spark_ranks(case.sparks),
        layout: Some(LayoutOptions { rtx_bytes: vec![case.budget; case.rtx], context_tokens: case.context,
            workspace_manifest: Some(case.manifest.to_path_buf()), onboard: case.onboard, peer_expert_ranges: case.peer,
            native_mtp_layers: if case.dspark { 3 } else { 0 }, headroom_bytes: 0, ..Default::default() }),
        ..Default::default() }).unwrap();
    let hints = report.hints.iter().map(|h| h.what.clone()).collect();
    (report.memory_layout.unwrap(), report.placement_supported, hints)
}

/// serve-dsv4's admission over the planner's own loaded bytes as the sample.
fn runtime(case: &Case<'_>, layout: &MemoryLayout) -> anyhow::Result<Placement> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(case.manifest)?)?;
    let catalog = cuteafd_loader::read_expert_catalog(case.snapshot)?;
    let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(case.snapshot, 1)?;
    let family = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
    let peers = (0..case.sparks).map(|r| format!("10.0.0.{}:1970{r}", r + 1)).collect::<Vec<_>>().join(",");
    let mut argv = vec!["serve".to_string(), "--snapshot".into(), case.snapshot.display().to_string(),
        "--native-lib".into(), "/nonexistent/libcuteafd.so".into(), "--peers".into(), peers];
    if let Some(onboard) = case.onboard { argv.extend(["--rtx-expert-layers".into(), onboard.to_string()]); }
    if case.peer { argv.push("--peer-expert-ranges".into()); }
    if case.rtx == 2 { argv.extend(["--split-device".into(), "1".into()]); }
    if case.dspark { argv.push("--dspark".into()); }
    let cli = Cli::try_parse_from(argv)?;
    // serve-dsv4 owns caches for every dSpark stage the checkpoint carries.
    let stages = (0..).take_while(|stage| catalog.tensors().iter()
        .any(|t| t.metadata.name.starts_with(&format!("mtp.{stage}.")))).count();
    let gpus = layout.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx)
        .map(|device| {
            let loaded: u64 = device.items.iter().filter(|i| matches!(i.category,
                Category::Weights | Category::Embedding | Category::Drafter) || i.group == "context+modules")
                .map(|i| i.bytes).sum();
            (case.budget, Baseline::Measured { free_bytes: case.budget - loaded })
        }).collect();
    // The planner's EXL3 arena from the same capacity manifests serve-dsv4 reads.
    let expert_workspace = layout_expert_workspace(case, &catalog)?;
    let inputs = admission::Inputs { cfg: &cfg, catalog: &catalog, manifest: &manifest, family, gpus,
        cache_stages: stages, prefill_rows: manifest["capacities"]["prefill_rows"].as_u64().unwrap() as usize,
        decode_rows: manifest["capacities"]["decode_rows"].as_u64().unwrap() as usize,
        max_context: case.context as usize, prefix: Some(&cli.prefix), expert_workspace: Some(expert_workspace) };
    Ok(solve(&admission::request(&cli.engine, &inputs)?)?)
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
fn assert_equal(case: &Case<'_>, label: &str) -> Placement {
    let (layout, supported, hints) = planned(case);
    assert!(supported, "{label}: planner refused: {:?} {:?}", layout.notes, hints);
    let runtime = runtime(case, &layout).unwrap_or_else(|e| panic!("{label}: runtime admission: {e:#}"));
    assert_eq!(layout.pool_tokens, runtime.pool_tokens, "{label}: pool");
    for (gpu, items) in runtime.items.iter().enumerate() {
        let device = &layout.devices[gpu];
        for item in items {
            assert!(device.items.contains(item), "{label}: rtx{gpu} lacks {item:?}");
        }
        let reserve = case.budget - device.capacity_bytes;
        assert!(reserve > 0, "{label}: rtx{gpu} has no reserve");
    }
    let notes = runtime.expert_ranges.iter().enumerate().map(|(gpu, r)|
        format!("rtx{gpu}: {} local expert layers ({}..{})", r.layers, r.first, r.first + r.layers)).collect::<Vec<_>>();
    for note in notes { assert!(layout.notes.contains(&note), "{label}: planner lacks {note}"); }
    // Per-layer ownership: today's V4 is all head split (2 RTX) or all GPU0,
    // the residual replicated after layer 0 under the split, no charged hop,
    // and the engine accepts the placement.
    assert_eq!(runtime.residual.len(), runtime.layers.len() + 1, "{label}: residual homes");
    assert!(runtime.hops.iter().all(|h| !h.charged()), "{label}: V4 hops");
    assert!(runtime.items.iter().flatten().all(|i| i.group != "residual hops"), "{label}: hop buffers");
    super::check_modes(&runtime, case.rtx == 2).unwrap_or_else(|e| panic!("{label}: {e:#}"));
    assert!(super::check_modes(&runtime, case.rtx != 2).is_err(), "{label}: the other engine shape is refused");
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

/// Always runs: the fixture's four layers at Flash geometry, one and two
/// GPUs, both extents, automatic and fixed onboard.
#[test]
fn planner_equals_runtime_deepseek_v4_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic(dir.path());
    for (rtx, budget) in [(1, 24u64 << 30), (2, 24 << 30)] {
        for context in [131_072, 1_048_576] {
            let max = Onboard::ExpertsFirst { pool_floor: 262_144 };
            for onboard in [None, Some(Onboard::Auto), Some(max), Some(Onboard::Layers(2)), Some(Onboard::Layers(0))] {
                let case = Case { snapshot: dir.path(), manifest: &manifest, rtx, sparks: 2, context, budget,
                    onboard, dspark: false, peer: false };
                let placement = assert_equal(&case, &format!("fixture rtx{rtx} {context} {onboard:?}"));
                // The default: pool first on one RTX, experts first on two (`default_onboard`).
                let effective = onboard.unwrap_or(cuteafd_loader::placement::families::deepseek_v4::default_onboard(rtx));
                match effective {
                    Onboard::ExpertsFirst { .. } => assert!(placement.pool_tokens >= 262_144, "experts first keeps a 262K pool"),
                    Onboard::Auto => assert_eq!(placement.pool_tokens, 1 << 20, "24 GiB cards target 1M"),
                    Onboard::Layers(n) => {
                        assert_eq!(placement.onboard_layers, n);
                        assert!(placement.pool_tokens >= 1 << 20, "a fixed onboard fills the pool past the target");
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

fn snapshot(model: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HOME").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/sparknest/hf-home")).join("hub")
        .join(format!("models--{}", model.replace('/', "--"))).join("snapshots");
    std::fs::read_dir(hub).ok()?.filter_map(|e| e.ok()).map(|e| e.path())
        .find(|p| p.join("model.safetensors.index.json").is_file())
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
    let (Some(flash), Some(pro), Some(inputs)) = (snapshot("deepseek-ai/DeepSeek-V4-Flash-0731"),
        snapshot("wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1"), inputs_dir()) else {
        eprintln!("planner_equals_runtime_deepseek_v4: V4 snapshots or manifests absent; skipped");
        return;
    };
    let budget = 90u64 << 30;
    let mut auto = std::collections::BTreeMap::new();
    for context in [131_072u64, 1_048_576] {
        for (model, snapshot, rtx, sparks) in [("flash", &flash, 1, 2), ("flash", &flash, 2, 4),
            ("pro", &pro, 1, 4), ("pro", &pro, 2, 6)] {
            let manifest = inputs.join(format!("{model}-{rtx}-{context}/PROGRAMS.json"));
            // Pool first with GPU1 ranges opted in (EXL3 only; native binds to GPU0).
            let case = Case { snapshot, manifest: &manifest, rtx, sparks, context, budget,
                onboard: Some(Onboard::Auto), dspark: true, peer: true };
            let placement = assert_equal(&case, &format!("{model} rtx{rtx} {context}"));
            assert_eq!(placement.pool_tokens, 2 << 20, "{model} rtx{rtx}: pool first reaches 2M");
            if rtx == 2 { assert_eq!(placement.expert_ranges[1].layers > 0, model == "pro", "{model}: GPU1 routed layers"); }
            // The default: pool first on one RTX (the 2M pool), experts first on two (GPU0 only, a
            // pool between 262K and the target).
            let default = assert_equal(&Case { onboard: None, peer: false, ..case }, &format!("{model} rtx{rtx} {context} default"));
            assert!(default.expert_ranges.get(1).is_none_or(|r| r.layers == 0));
            if matches!(cuteafd_loader::placement::families::deepseek_v4::default_onboard(rtx), Onboard::Auto) {
                assert_eq!(default.pool_tokens, 2 << 20, "{model}: a pool-first default reaches 2M");
            } else { assert!((262_144..=2 << 20).contains(&default.pool_tokens)); }
            auto.insert((model, rtx, context), placement);
        }
    }
    // Pool first is extent-independent at these budgets.
    for model in ["flash", "pro"] {
        for rtx in [1, 2] {
            assert_eq!(auto[&(model, rtx, 131_072)].expert_ranges, auto[&(model, rtx, 1_048_576)].expert_ranges);
        }
    }
    // Experts first (v2's policy): more layers than auto, a pool above 262K.
    for (model, snapshot, sparks) in [("flash", &flash, 2), ("pro", &pro, 4)] {
        let manifest = inputs.join(format!("{model}-1-1048576/PROGRAMS.json"));
        let case = Case { snapshot, manifest: &manifest, rtx: 1, sparks, context: 1_048_576, budget,
            onboard: Some(Onboard::ExpertsFirst { pool_floor: 262_144 }), dspark: true, peer: false };
        let placement = assert_equal(&case, &format!("{model} min max"));
        assert!(placement.onboard_layers > auto[&(model, 1, 1_048_576)].onboard_layers);
        assert!((262_144..2 << 20).contains(&placement.pool_tokens));
    }
    // Fixed onboard: the pool is the output.
    for context in [131_072u64, 1_048_576] {
        let manifest = inputs.join(format!("flash-2-{context}/PROGRAMS.json"));
        let case = Case { snapshot: &flash, manifest: &manifest, rtx: 2, sparks: 4, context, budget,
            onboard: Some(Onboard::Auto), dspark: true, peer: true };
        // Native MXFP4 layers stay on GPU0: 15 fit beside a fixed onboard's
        // minimum pool; Pro EXL3 max splits half of its 61 over both GPUs.
        let placement = assert_equal(&Case { onboard: Some(Onboard::Layers(12)), ..case }, &format!("flash max {context} onboard 12"));
        assert_eq!((placement.onboard_layers, placement.expert_ranges[1].layers), (12, 0));
        assert!(placement.pool_tokens > 2 << 20, "fewer layers than auto leave a pool above the target");
        let manifest = inputs.join(format!("pro-2-{context}/PROGRAMS.json"));
        let pro_case = Case { snapshot: &pro, manifest: &manifest, rtx: 2, sparks: 6, context, budget,
            onboard: Some(Onboard::Layers(6)), dspark: true, peer: true };
        let pro_fixed = assert_equal(&pro_case, &format!("pro max {context} onboard 6"));
        assert_eq!(pro_fixed.onboard_layers, 6);
        assert!(pro_fixed.expert_ranges[1].layers > 0 && pro_fixed.pool_tokens > 2 << 20);
        let percent = Case { onboard: Some(Onboard::Fraction(0.1)), ..pro_case };
        assert_eq!(assert_equal(&percent, "pro max 10%").onboard_layers, 6);
    }
}

/// A hardware inventory fixture (`shared/placement_fixtures/<card>.json`),
/// recorded from serve-dsv4's `placement inventory` line on a real launch:
/// each GPU's CUDA total and the free bytes its admission sampled after
/// context, modules and weights.
#[derive(serde::Deserialize)]
struct Fixture {
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
        let label = path.file_stem().unwrap().to_string_lossy().into_owned();
        let manifest = PathBuf::from(shellexpand(&fixture.manifest));
        let (Some(snapshot), true) = (snapshot(&fixture.model), manifest.is_file()) else {
            eprintln!("{label}: snapshot or manifest absent; skipped");
            continue;
        };
        let budget = fixture.gpus[0].total_bytes;
        assert!(fixture.gpus.iter().all(|g| g.total_bytes.abs_diff(budget) < 8 * MIB), "{label}: unequal GPUs");
        let onboard = fixture.onboard.as_deref().map(|o| o.parse::<Onboard>().unwrap());
        let case = Case { snapshot: &snapshot, manifest: &manifest, rtx: fixture.rtx, sparks: fixture.sparks,
            context: fixture.context, budget, onboard, dspark: true, peer: false };
        let (layout, supported, hints) = planned(&case);
        assert!(supported, "{label}: planner refused: {:?} {:?}", layout.notes, hints);
        // The planner's view of the sample: everything loaded before admission.
        for (gpu, measured) in fixture.gpus.iter().enumerate() {
            let device = &layout.devices[gpu];
            let planned_loaded: u64 = device.items.iter().filter(|i| matches!(i.category,
                Category::Weights | Category::Embedding | Category::Drafter) || i.group == "context+modules")
                .map(|i| i.bytes).sum();
            let measured_loaded = measured.total_bytes - measured.admission_free_bytes;
            let diff = planned_loaded as i64 - measured_loaded as i64;
            eprintln!("{label} rtx{gpu}: planned loaded {planned_loaded} measured {measured_loaded} diff {diff} \
                (driver {})", fixture.driver);
            assert!(diff.unsigned_abs() <= 64 * MIB, "{label} rtx{gpu}: planner baseline {diff:+} B from the \
                measured inventory");
        }
        // The runtime's admission over the measured sample.
        let runtime = runtime_measured(&case, &fixture).unwrap_or_else(|e| panic!("{label}: runtime: {e:#}"));
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
        checked += 1;
    }
    eprintln!("planner_equals_runtime_deepseek_v4_measured: {checked} fixtures");
}

fn shellexpand(path: &str) -> String {
    path.replace("~", &std::env::var("HOME").unwrap_or_default())
}

/// serve-dsv4's admission over a fixture's measured free bytes.
fn runtime_measured(case: &Case<'_>, fixture: &Fixture) -> anyhow::Result<Placement> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(case.manifest)?)?;
    let catalog = cuteafd_loader::read_expert_catalog(case.snapshot)?;
    let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(case.snapshot, 1)?;
    let family = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
    let peers = (0..case.sparks).map(|r| format!("10.0.0.{}:1970{r}", r + 1)).collect::<Vec<_>>().join(",");
    let mut argv = vec!["serve".to_string(), "--snapshot".into(), case.snapshot.display().to_string(),
        "--native-lib".into(), "/nonexistent/libcuteafd.so".into(), "--peers".into(), peers, "--dspark".into()];
    if let Some(onboard) = &fixture.onboard { argv.extend(["--rtx-expert-layers".into(), onboard.clone()]); }
    if case.rtx == 2 { argv.extend(["--split-device".into(), "1".into()]); }
    let cli = Cli::try_parse_from(argv)?;
    let stages = (0..).take_while(|stage| catalog.tensors().iter()
        .any(|t| t.metadata.name.starts_with(&format!("mtp.{stage}.")))).count();
    let gpus = fixture.gpus.iter().map(|g| (g.total_bytes, Baseline::Measured { free_bytes: g.admission_free_bytes }))
        .collect();
    let expert_workspace = layout_expert_workspace(case, &catalog)?;
    let inputs = admission::Inputs { cfg: &cfg, catalog: &catalog, manifest: &manifest, family, gpus,
        cache_stages: stages, prefill_rows: manifest["capacities"]["prefill_rows"].as_u64().unwrap() as usize,
        decode_rows: manifest["capacities"]["decode_rows"].as_u64().unwrap() as usize,
        max_context: case.context as usize, prefix: Some(&cli.prefix), expert_workspace: Some(expert_workspace) };
    Ok(solve(&admission::request(&cli.engine, &inputs)?)?)
}
