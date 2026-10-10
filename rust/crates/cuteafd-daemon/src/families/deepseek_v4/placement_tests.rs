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
    onboard: Onboard,
    dspark: bool,
}

fn planned(case: &Case<'_>) -> (MemoryLayout, bool, Vec<String>) {
    let report = plan(case.snapshot, &PlanOptions { placement: ExpertPlacement::from_spark_ranks(case.sparks),
        layout: Some(LayoutOptions { rtx_bytes: vec![case.budget; case.rtx], context_tokens: case.context,
            workspace_manifest: Some(case.manifest.to_path_buf()), onboard: case.onboard,
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
    let onboard = case.onboard.to_string();
    let mut argv = vec!["serve".to_string(), "--snapshot".into(), case.snapshot.display().to_string(),
        "--native-lib".into(), "/nonexistent/libcuteafd.so".into(), "--peers".into(), peers,
        "--rtx-expert-layers".into(), onboard];
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
        let reserve = case.budget - device.capacity_bytes;
        assert!(reserve > 0, "{label}: rtx{gpu} has no reserve");
    }
    let notes = runtime.expert_ranges.iter().enumerate().map(|(gpu, r)|
        format!("rtx{gpu}: {} local expert layers ({}..{})", r.layers, r.first, r.first + r.layers)).collect::<Vec<_>>();
    for note in notes { assert!(layout.notes.contains(&note), "{label}: planner lacks {note}"); }
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
            for onboard in [Onboard::Auto, Onboard::Layers(2), Onboard::Layers(0)] {
                let case = Case { snapshot: dir.path(), manifest: &manifest, rtx, sparks: 2, context, budget,
                    onboard, dspark: false };
                let placement = assert_equal(&case, &format!("fixture rtx{rtx} {context} {onboard}"));
                match onboard {
                    Onboard::Auto => assert_eq!(placement.pool_tokens, 1 << 20, "24 GiB cards target 1M"),
                    Onboard::Layers(n) => {
                        assert_eq!(placement.onboard_layers, n);
                        assert!(placement.pool_tokens >= 1 << 20, "a fixed onboard fills the pool past the target");
                    }
                    Onboard::Fraction(_) | Onboard::ExpertsFirst { .. } => unreachable!(),
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
            let case = Case { snapshot, manifest: &manifest, rtx, sparks, context, budget, onboard: Onboard::Auto,
                dspark: true };
            let placement = assert_equal(&case, &format!("{model} rtx{rtx} {context}"));
            assert_eq!(placement.pool_tokens, 2 << 20, "{model} rtx{rtx}: pool first reaches 2M");
            // EXL3 runs on both GPUs; native rtx_backbone binds to GPU0 (until P4).
            if rtx == 2 { assert_eq!(placement.expert_ranges[1].layers > 0, model == "pro", "{model}: GPU1 routed layers"); }
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
            onboard: Onboard::ExpertsFirst { pool_floor: 262_144 }, dspark: true };
        let placement = assert_equal(&case, &format!("{model} min max"));
        assert!(placement.onboard_layers > auto[&(model, 1, 1_048_576)].onboard_layers);
        assert!((262_144..2 << 20).contains(&placement.pool_tokens));
    }
    // Fixed onboard: the pool is the output.
    for context in [131_072u64, 1_048_576] {
        let manifest = inputs.join(format!("flash-2-{context}/PROGRAMS.json"));
        let case = Case { snapshot: &flash, manifest: &manifest, rtx: 2, sparks: 4, context, budget,
            onboard: Onboard::Auto, dspark: true };
        // Native MXFP4 layers stay on GPU0: 15 fit beside a fixed onboard's
        // minimum pool; Pro EXL3 max splits half of its 61 over both GPUs.
        let placement = assert_equal(&Case { onboard: Onboard::Layers(12), ..case }, &format!("flash max {context} onboard 12"));
        assert_eq!((placement.onboard_layers, placement.expert_ranges[1].layers), (12, 0));
        assert!(placement.pool_tokens > 2 << 20, "fewer layers than auto leave a pool above the target");
        let manifest = inputs.join(format!("pro-2-{context}/PROGRAMS.json"));
        let pro_case = Case { snapshot: &pro, manifest: &manifest, rtx: 2, sparks: 6, context, budget,
            onboard: Onboard::Layers(6), dspark: true };
        let pro_fixed = assert_equal(&pro_case, &format!("pro max {context} onboard 6"));
        assert_eq!(pro_fixed.onboard_layers, 6);
        assert!(pro_fixed.expert_ranges[1].layers > 0 && pro_fixed.pool_tokens > 2 << 20);
        let percent = Case { onboard: Onboard::Fraction(0.1), ..pro_case };
        assert_eq!(assert_equal(&percent, "pro max 10%").onboard_layers, 6);
    }
}
