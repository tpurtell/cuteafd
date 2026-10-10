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
    exchange_f32: bool,
    peer_budget: Option<u64>,
}

fn planned(case: &Case<'_>) -> (MemoryLayout, bool, Vec<String>) {
    let report = plan(case.snapshot, &PlanOptions { placement: ExpertPlacement::from_spark_ranks(case.sparks),
        layout: Some(LayoutOptions { rtx_bytes: (0..case.rtx).map(|g| if g == 1 { case.peer_budget.unwrap_or(case.budget) } else { case.budget }).collect(), context_tokens: case.context,
            workspace_manifest: Some(case.manifest.to_path_buf()), onboard: case.onboard, exchange_f32: case.exchange_f32,
            native_mtp_layers: if case.dspark { 3 } else { 0 }, headroom_bytes: 0, ..Default::default() }),
        ..Default::default() }).unwrap();
    let hints = report.hints.iter().map(|h| h.what.clone()).collect();
    (report.memory_layout.unwrap(), report.placement_supported, hints)
}

/// serve-dsv4's admission over the planner's own loaded bytes as the sample.
fn runtime(case: &Case<'_>, layout: &MemoryLayout) -> anyhow::Result<Placement> {
    runtime_with_tp1(case, layout, true)
}

fn runtime_with_tp1(case: &Case<'_>, layout: &MemoryLayout, tp1_available: bool) -> anyhow::Result<Placement> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(case.manifest)?)?;
    let catalog = cuteafd_loader::read_expert_catalog(case.snapshot)?;
    let cfg = cuteafd_loader::families::deepseek_v4::DeepseekV4Config::read(case.snapshot, 1)?;
    let family = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
    let peers = (0..case.sparks).map(|r| format!("10.0.0.{}:1970{r}", r + 1)).collect::<Vec<_>>().join(",");
    let mut argv = vec!["serve".to_string(), "--snapshot".into(), case.snapshot.display().to_string(),
        "--native-lib".into(), "/nonexistent/libcuteafd.so".into(), "--peers".into(), peers];
    if let Some(onboard) = case.onboard { argv.extend(["--rtx-expert-layers".into(), onboard.to_string()]); }
    if case.rtx == 2 { argv.extend(["--split-device".into(), "1".into()]); }
    if case.dspark { argv.push("--dspark".into()); }
    let cli = Cli::try_parse_from(argv)?;
    // serve-dsv4 owns caches for every dSpark stage the checkpoint carries.
    let stages = (0..).take_while(|stage| catalog.tensors().iter()
        .any(|t| t.metadata.name.starts_with(&format!("mtp.{stage}.")))).count();
    let gpus = layout.devices.iter().filter(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx)
        .enumerate().map(|(gpu, device)| {
            let loaded: u64 = device.items.iter().filter(|i| matches!(i.category,
                Category::Weights | Category::Embedding | Category::Drafter) || i.group == "context+modules")
                .map(|i| i.bytes).sum();
            let budget = if gpu == 1 { case.peer_budget.unwrap_or(case.budget) } else { case.budget };
            (budget, Baseline::Measured { free_bytes: budget - loaded })
        }).collect();
    // The planner's EXL3 arena from the same capacity manifests serve-dsv4 reads.
    let expert_workspace = layout_expert_workspace(case, &catalog)?;
    let inputs = admission::Inputs { cfg: &cfg, catalog: &catalog, manifest: &manifest, family, gpus,
        cache_stages: stages, prefill_rows: manifest["capacities"]["prefill_rows"].as_u64().unwrap() as usize,
        decode_rows: manifest["capacities"]["decode_rows"].as_u64().unwrap() as usize,
        max_context: case.context as usize, prefix: Some(&cli.prefix), expert_workspace: tp1_available.then_some(expert_workspace),
        tp2_workspace: if case.rtx == 2 { cuteafd_loader::serving_capacity::deepseek_v4_tp2_workspace(
            &catalog, Some(case.manifest), manifest["capacities"]["prefill_rows"].as_u64().unwrap()
                .max(manifest["capacities"]["decode_rows"].as_u64().unwrap())).ok().map(|b| [b; 2]) } else { None },
        exchange_f32: case.exchange_f32 };
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
        let budget = if gpu == 1 { case.peer_budget.unwrap_or(case.budget) } else { case.budget };
        let reserve = budget - device.capacity_bytes;
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

/// Always runs: the fixture's four layers at Flash geometry, one and two
/// GPUs, both extents, automatic and fixed onboard.
#[test]
fn planner_equals_runtime_deepseek_v4_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic(dir.path());
    for (rtx, budget) in [(1, 24u64 << 30), (2, 24 << 30)] {
        for context in [131_072, 1_048_576] {
            for onboard in [None, Some(Onboard::Auto), Some(Onboard::Layers(2)), Some(Onboard::Layers(0)), Some(Onboard::Fraction(0.5)),
                Some(Onboard::ExpertsFirst { pool_floor: 262_144 })] {
                let case = Case { snapshot: dir.path(), manifest: &manifest, rtx, sparks: 2, context, budget,
                    onboard, dspark: false, exchange_f32: false, peer_budget: None };
                let placement = assert_equal(&case, &format!("fixture rtx{rtx} {context} {onboard:?}"));
                // The default: pool first for Flash on one or two RTX (`default_onboard`).
                let effective = onboard.unwrap_or(cuteafd_loader::placement::families::deepseek_v4::default_onboard(4096, rtx));
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
    for onboard in [None, Some(Onboard::Auto), Some(Onboard::Layers(2)), Some(Onboard::Fraction(0.5)),
        Some(Onboard::ExpertsFirst { pool_floor: 262_144 })] {
        for exchange_f32 in [false, true] {
            let case = Case { snapshot: dir.path(), manifest: &manifest, rtx: 2, sparks: 2,
                context: 1_048_576, budget: 32 << 30, peer_budget: Some(30 << 30),
                onboard, dspark: true, exchange_f32 };
            let p = assert_equal(&case, &format!("fixture TP2 dSpark {onboard:?} f32={exchange_f32}"));
            assert!(p.expert_ranges.iter().all(|r| r.layers == 0));
            assert!(p.expert_ranges[0].peak_bytes > 0);
            assert_eq!(p.expert_ranges[1].peak_bytes, 0);
            assert_eq!(p.tp2.unwrap().layers, p.onboard_layers);
        }
    }
}

#[test]
fn tp2_backbone_admission_does_not_require_tp1_kernels() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = synthetic(dir.path());
    for onboard in [None, Some(Onboard::Auto), Some(Onboard::Layers(0)), Some(Onboard::Layers(2)),
        Some(Onboard::ExpertsFirst { pool_floor: 262_144 })] {
        let case = Case { snapshot: dir.path(), manifest: &manifest, rtx: 2, sparks: 2,
            context: 1_048_576, budget: 24 << 30, peer_budget: None, onboard, dspark: false, exchange_f32: false };
        let (layout, supported, _) = planned(&case);
        assert!(supported);
        assert_eq!(runtime_with_tp1(&case, &layout, false).unwrap(), runtime(&case, &layout).unwrap());
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
    let budget = (95.5 * (1u64 << 30) as f64) as u64;
    for context in [131_072u64, 1_048_576] {
        for (model, snapshot, rtx, sparks) in [("flash", &flash, 1, 2), ("flash", &flash, 2, 4),
            ("pro", &pro, 1, 4), ("pro", &pro, 2, 6)] {
            let manifest = inputs.join(format!("{model}-{rtx}-{context}/PROGRAMS.json"));
            for onboard in [None, Some(Onboard::Auto), Some(Onboard::ExpertsFirst { pool_floor: 262_144 }),
                Some(Onboard::Layers(2)), Some(Onboard::Fraction(0.03))] {
                for asymmetric in [false, true] {
                    let case = Case { snapshot, manifest: &manifest, rtx, sparks, context, budget,
                        onboard, dspark: true, exchange_f32: asymmetric,
                        peer_budget: asymmetric.then_some(budget - (2 << 30)) };
                    let placement = assert_equal(&case, &format!("{model} rtx{rtx} {context} {onboard:?} asym={asymmetric}"));
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
    for onboard in [None, Some(Onboard::Auto), Some(Onboard::Layers(0))] {
        let case = Case { snapshot: &pro, manifest: &manifest, rtx: 2, sparks: 6,
            context: 1_048_576, budget: (95.5 * (1u64 << 30) as f64) as u64,
            peer_budget: None, onboard, dspark: false, exchange_f32: false };
        let p = assert_equal(&case, &format!("missing TP2 package {onboard:?}"));
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
        let budget = fixture.gpus[0].total_bytes;
        assert!(fixture.gpus.iter().all(|g| g.total_bytes.abs_diff(budget) < 8 * MIB), "{label}: unequal GPUs");
        let onboard = fixture.onboard.as_deref().map(|o| o.parse::<Onboard>().unwrap());
        let case = Case { snapshot: &snapshot, manifest: &manifest, rtx: fixture.rtx, sparks: fixture.sparks,
            context: fixture.context, budget, onboard, dspark: true, exchange_f32: false, peer_budget: None };
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
        if fixture.rtx == 2 {
            assert!(runtime.expert_ranges.iter().all(|r| r.layers == 0));
            assert_eq!(runtime.tp2.unwrap().layers, fixture.onboard_layers);
            // Re-solve the same measured pre-allocation inventory for every
            // retained onboard policy; no deleted two-RTX TP1 placement.
            for onboard in [None, Some(Onboard::Auto),
                Some(Onboard::ExpertsFirst { pool_floor: 262_144 }),
                Some(Onboard::Layers(2)), Some(Onboard::Fraction(0.03))] {
                let variant = Case { onboard, ..case };
                let (layout, supported, hints) = planned(&variant);
                assert!(supported, "{label} {onboard:?}: {hints:?}");
                let runtime = runtime_measured(&variant, &fixture).unwrap();
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
fn runtime_measured(case: &Case<'_>, fixture: &Fixture) -> anyhow::Result<Placement> {
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(case.manifest)?)?;
    let catalog = cuteafd_loader::read_expert_catalog(case.snapshot)?;
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
    let gpus = fixture.gpus.iter().map(|g| (g.total_bytes, Baseline::Measured { free_bytes: g.admission_free_bytes }))
        .collect();
    let expert_workspace = layout_expert_workspace(case, &catalog)?;
    let inputs = admission::Inputs { cfg: &cfg, catalog: &catalog, manifest: &manifest, family, gpus,
        cache_stages: stages, prefill_rows: manifest["capacities"]["prefill_rows"].as_u64().unwrap() as usize,
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
