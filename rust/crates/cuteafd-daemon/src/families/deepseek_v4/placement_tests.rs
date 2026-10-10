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
            // As local::workspace_bytes: every capacity up to the rows, then the
            // first at or above them (4096 twice at 4096 rows).
            const CAPACITIES: [u64; 6] = [1, 16, 80, 256, 1024, 4096];
            let manifests = CAPACITIES.into_iter().filter(|&n| n <= rows)
                .chain(CAPACITIES.into_iter().find(|&n| n >= rows))
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
    if let Some(t) = runtime.tp2 {
        assert!(runtime.expert_ranges.iter().all(|r| r.layers == 0));
        assert!(layout.notes.contains(&format!("rtx0/rtx1: {} TP2 expert layer halves ({}..{})", t.layers, t.first, t.first + t.layers)));
        for gpu in 0..2 {
            let bytes: u64 = layout.devices[gpu].items.iter().filter(|i| i.format == "tp2").map(|i| i.bytes).sum();
            assert_eq!(bytes, t.peak_bytes[gpu], "{label}: rtx{gpu} TP2 arena");
        }
    }
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
                match onboard {
                    None => assert!(placement.pool_tokens >= 262_144, "experts first keeps a 262K pool"),
                    Some(Onboard::Auto) => assert_eq!(placement.pool_tokens, 1 << 20, "24 GiB cards target 1M"),
                    Some(Onboard::Layers(n)) => {
                        assert_eq!(placement.onboard_layers, n);
                        assert!(placement.pool_tokens >= 1 << 20, "a fixed onboard fills the pool past the target");
                    }
                    Some(Onboard::Fraction(_)) => assert_eq!(placement.onboard_layers, 2),
                    Some(Onboard::ExpertsFirst { .. }) => assert!(placement.pool_tokens >= context),
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
