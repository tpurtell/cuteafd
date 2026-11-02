"""Qwen tower ABI, interpolation order and head72 export contracts, CPU-only."""
import ast
import ctypes
import re
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_lazy_exl3_draft_grouping_uses_the_shared_installed_window_order():
    daemon = ROOT / "rust/crates/cuteafd-daemon/src/families"
    for family, stream in (("qwen4", "self.execution_stream()"), ("glm5_flash", "self.stream")):
        source = (daemon / family / "engine.rs").read_text()
        geometry = source.split("pub(crate) fn draft_geometry(", 1)[1].split("\n    }", 1)[0]
        assert "crate::families::deepseek_v4::local::installed_window_group_rows(" in geometry
        assert f"&local.resident, || local.ensure(layer, {stream}), rows," in geometry
        assert "|experts, rows| experts.group_rows(rows)" in geometry
        assert "local.resident.borrow()" not in geometry
    # GLM Flash already installs TP1 windows during admitted onboarding; the
    # shared helper reuses these without changing their stream or load timing.
    glm = (daemon / "glm5_flash/mod.rs").read_text()
    assert glm.index("local.ensure(local_layers[0], stream)?;") < glm.index("engine.install_local(local_layers, experts)?;")


def test_qwen_bucket_projection_registry_matches_pinned_fork():
    fork = ROOT / "third_party/sparkinfer/b12x"
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs").read_text()
    def rust(name):
        return int(re.search(rf"const {name}: usize = (\d+);", engine).group(1))
    skinny = ast.parse((fork / "gemm/bf16_gemv/_skinny.py").read_text())
    threshold = next(n for n in skinny.body if isinstance(n, ast.FunctionDef) and n.name == "skinny_max_rows")
    # Execute only the pure scalar rule, never import CUDA, torch or the exporter.
    scope = {}
    exec(compile(ast.Module(body=[threshold], type_ignores=[]), "pinned skinny rule", "exec"), scope)
    scalar_rule = scope["skinny_max_rows"]
    assert scalar_rule(12800, 2560) == rust("QWEN_WIDE_SKINNY_ROWS")
    for n in (512, 1296):
        assert scalar_rule(n, 2560) == rust("QWEN_MEDIUM_SKINNY_ROWS")
    for n in (320, 324):
        assert scalar_rule(n, 10240) == rust("QWEN_SMALL_SKINNY_ROWS")
    glm = ast.parse((fork / "integration/cuteafd/_glm_kernels.py").read_text())
    canonical = next(n for n in glm.body if isinstance(n, ast.FunctionDef) and n.name == "glm_projection")
    route = next(n.value for n in canonical.body if isinstance(n, ast.Return))
    assert isinstance(route, ast.Call) and route.func.id == "RoutedBf16Projection"
    cutoff = next(keyword.value for keyword in route.keywords if keyword.arg == "max_skinny_rows")
    assert ast.literal_eval(cutoff) == rust("GLM_BF16_SKINNY_ROWS") == 0
    fp8 = (fork / "integration/cuteafd/_fp8_weights.py").read_text()
    # One canonical reduction per load at every width: the thin tile the decode programs run at
    # C1 is FP8_GEMV_ROWS, the capacity a multi-tile GEMV over the same warps/groups covers is
    # FP8_DECODE_ROWS, and the registered arithmetic crossover is that capacity (not the tile).
    assert int(re.search(r"FP8_DECODE_ROWS = (\d+)", fp8).group(1)) == rust("FP8_PROJECTION_SKINNY_ROWS")
    assert "self.inner = glm_projection(n, k)" in fp8
    fp8_ast = ast.parse(fp8)
    projection_class = next(n for n in fp8_ast.body if isinstance(n, ast.ClassDef) and n.name == "Fp8Projection")
    init = next(n for n in projection_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = {arg.arg: ast.literal_eval(default) for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults)
                if arg.arg in ("gemv_rows", "wide_rows")}
    assert defaults["gemv_rows"] == int(re.search(r"FP8_GEMV_ROWS = (\d+)", fp8).group(1))
    assert defaults["wide_rows"] == 0
    for filename, contracts in {
        "qwen4_gdn.py": ("self.in_proj = projection(p, g.hidden, self.fp8)", "self.o_proj = projection(g.hidden, v, self.fp8)"),
        "qwen4_attention.py": ("self.proj = projection(g.attn_in_width, g.hidden, self.fp8)",
                               "self.o_proj = projection(g.hidden, g.heads * g.head_dim, self.fp8)"),
        "glmf.py": ("self.proj = RoutedBf16Projection(e, h, out_dtype=cutlass.Float32)",),
    }.items():
        source = (fork / "integration/cuteafd" / filename).read_text()
        for contract in contracts:
            assert contract in source
    qwen = (fork / "integration/cuteafd/qwen4.py").read_text()
    for contract in ("if (k // 8) % 32:", "return TmaBf16Projection(n, k, out_dtype=out_dtype)",
                     "return RoutedBf16Projection(n, k, out_dtype=out_dtype)",
                     "self.kv = _projection(c + h, g.ple_dim)", "self.di = _projection(self.d_width, c)",
                     "self.up = _projection(c, r)", "self.gate_up = _projection(SHARED_ROWS, g.hidden)",
                     "self.down = _projection(g.hidden, i)", "self.fc = _projection(g.hidden, g.hidden)"):
        assert contract in qwen
    expected_registry = {
        "gdn.in_projection.bf16": "GLM_BF16_SKINNY_ROWS", "gdn.out_projection.bf16": "GLM_BF16_SKINNY_ROWS",
        "attention.in_projection.bf16": "GLM_BF16_SKINNY_ROWS", "attention.out_projection.bf16": "GLM_BF16_SKINNY_ROWS",
        "hc.down_inject": "QWEN_SMALL_SKINNY_ROWS", "head.mixer_down": "QWEN_SMALL_SKINNY_ROWS",
        "ple.kv": "QWEN_WIDE_SKINNY_ROWS", "router.scores": "QWEN_MEDIUM_SKINNY_ROWS",
        "shared.gate_up": "QWEN_MEDIUM_SKINNY_ROWS", "mtp.feedback": "QWEN_WIDE_SKINNY_ROWS",
        "gdn.projections.fp8": "FP8_PROJECTION_SKINNY_ROWS", "attention.projections.fp8": "FP8_PROJECTION_SKINNY_ROWS",
    }
    actual_registry = dict(re.findall(r'ProjectionThreshold \{ name: "([^"]+)", skinny_rows: (\w+) \}', engine))
    assert actual_registry == expected_registry
    for buckets in ("PLAIN_BUCKETS", "SPEC_BUCKETS"):
        assert f"check_bucket_thresholds({buckets}, DECODE_PROJECTION_THRESHOLDS)?" in engine
    # The buckets are the startup graph set's, shared with admission and the planner.
    graphs = (ROOT / "rust/crates/cuteafd-loader/src/serving_capacity/qwen_graphs.rs").read_text()
    assert "QWEN_PLAIN_BUCKETS as PLAIN_BUCKETS, QWEN_SPEC_BUCKETS as SPEC_BUCKETS" in engine
    assert "pub const QWEN_PLAIN_BUCKETS: &[usize] = &[1, 4, 8, 16];" in graphs
    assert "pub const QWEN_SPEC_BUCKETS: &[usize] = &[2, 4, 8, 16, 24, 32, 64];" in graphs
    assert int(re.search(r"pub\(crate\) const FP8_HEAD_ROWS: usize = (\d+);", engine).group(1)) == rust("FP8_PROJECTION_SKINNY_ROWS")
    assert "_HeadFp8(vocab, h, max_rows=FP8_DECODE_ROWS)" in qwen
    tables = {node.targets[0].id: ast.literal_eval(node.value) for node in skinny.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in ("DECODE_TMA_TILES", "SKINNY_TILES_BY_ROWS")}
    # Qwen's row-tiled shapes are pinned exactly; other families' keys (Step-5's, fork #71)
    # must not alias a Qwen [N, K].
    qwen_tiles = {(2560, 6144), (16480, 2560), (13952, 2560), (2560, 640), (10240, 320)}
    step5_tiles = {(4096, 12288), (4096, 6144), (384, 4096), (3072, 4096)}
    assert set(tables["DECODE_TMA_TILES"]) == qwen_tiles | step5_tiles
    for (n, k), entries in tables["SKINNY_TILES_BY_ROWS"].items():
        assert len({t for _, _, _, t in entries}) == 1
        assert entries[-1][0] == scalar_rule(n, k)


def test_head72_uses_the_b12x_le128_tile():
    export = module("python/tools/aot/export_b12x_vision_aot.py", "vision_export")
    assert export.tile_for_head_dim(64) == (128,128)
    assert export.tile_for_head_dim(72) == (128,64)
    assert export.tile_for_head_dim(128) == (128,64)
    for dim in (0,73,136):
        with pytest.raises(ValueError): export.tile_for_head_dim(dim)


def test_vision_pointer_count_abi_matches_c_rust():
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "qwen_vision_gate")
    assert ctypes.sizeof(gate.Block) == 96
    assert ctypes.sizeof(gate.common.Spec) == 256
    assert ctypes.sizeof(gate.Spec) == 256
    assert gate.Spec.blocks.offset == 72
    assert gate.Spec.hidden.offset == 80
    assert gate.Spec.patch_bias.offset == 120
    assert gate.Spec.norm1_bias.offset == 224


def test_qwen_patch_lut_uses_host_f64_rescale_then_f32_normalize():
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "qwen_lut_gate")
    lut = gate.normalization_lut()
    expected = np.array([np.float32((np.float32(i*(1/255))-.5)/.5) for i in range(256)])
    np.testing.assert_array_equal(lut, np.broadcast_to(expected,(3,256)))
    gh,gw,rgb = gate.common.fixture(256)
    patches = gate.common.patches(rgb,lut).reshape(gh*gw,3,2,16,16)
    np.testing.assert_array_equal(patches[:,:,0],patches[:,:,1])


def test_qwen_calibration_keeps_strict_misses_and_final_output_mean_floor():
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "qwen_calibration_gate")
    floor = dict(relative_l2=.040262, mean_cosine=.997386, worst_cosine=.870111, **{"pass":False})
    measured = dict(relative_l2=.037299, mean_cosine=.997565, worst_cosine=.939182, **{"pass":False})
    intermediate = gate.calibrated_metrics(measured,floor,final_output=False)
    assert intermediate["pass"] and not intermediate["strict_pass"]
    assert not gate.calibrated_metrics(measured,floor,final_output=True)["pass"]
    # MiMo's original policy remains unchanged and still fails this intermediate.
    assert not gate.common.calibrated_metrics(measured,floor)["pass"]
    for field,value in (("relative_l2",.042263),("mean_cosine",.997335),("worst_cosine",.869110)):
        assert not gate.calibrated_metrics(dict(measured,**{field:value}),floor,final_output=False)["pass"]
    final = dict(relative_l2=.025771,mean_cosine=.999705,worst_cosine=.995945,**{"pass":True})
    final_floor = dict(relative_l2=.027005,mean_cosine=.999679,worst_cosine=.994904,**{"pass":True})
    assert gate.calibrated_metrics(final,final_floor,final_output=True)["pass"]


def test_qwen_pointwise_launches_cover_every_element():
    source = (ROOT / "native/shared/cuda/vision_qwen.cuh").read_text()
    # Every pointwise kernel using the unchanged capped launcher must grid-stride.
    assert source.count("i+=size_t(gridDim.x)*blockDim.x") == 5
    assert "(cast_float<<<grid_for(" not in source
    for count in (4096*256+1,256*4*1152,4096*4304,4096*2560):
        stride = min((count+255)//256,4096)*256
        first_thread = (count-1)%stride
        visited = range(first_thread,count,stride)
        assert visited[-1] == count-1
    test = (ROOT / "native/tests/vision_pointwise_test.cu").read_text()
    assert "4096*256+1" in test
    for kernel in ("qwen_patch_position","qwen_residual","qwen_biased_gelu","qwen_biased_cast","qwen_cast_float"):
        assert kernel in test


@pytest.mark.parametrize("tokens", [256,1024,2048,4096])
def test_qwen_diagnostic_fixture_geometry(tokens):
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "qwen_fixture_gate")
    gh,gw,rgb = gate.fixture(tokens)
    assert gh*gw == tokens*4
    assert gh%2 == gw%2 == 0
    assert rgb.shape == (gh*16,gw*16,3)
    assert rgb.dtype == np.uint8 and rgb.flags.c_contiguous


def test_qwen_cold_replay_echo_has_validated_prefill_decode_execution():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs").read_text()
    api = (ROOT / "rust/crates/cuteafd-api/src/openai/probe.rs").read_text()
    assert serve.index("probe.spec.validate_cold_steps(") < serve.index('probe::admitted(&job.probe, geometry.family,')
    assert "probe.spec.cold_steps.iter().map(|step| step.end).collect(), points: Vec::new()" in serve
    assert "probe.spec.cold_steps.get(p.chunks)" in serve
    assert "let logits = if decode {\n                        engine.verify_device_ungraphed" in serve
    assert "} else { engine.prefill_device(&mut p.placement, chunk, 1, cache.pool())? };" in serve
    assert 'matches!(engine, "mimo_v2" | "glm5_flash" | "qwen4" | "deepseek_v41")' in api
    assert "else if logits.is_some() { PointPlan::default() } else { plan }" in serve


def test_qwen_unit_table_uploads_are_reserved_at_full_extent_before_startup_body():
    root = ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4"
    engine = (root / "engine.rs").read_text()
    family = (root / "mod.rs").read_text()
    startup = family[family.index("pub fn with_engine<T>"):family.index("    fn experts<'s>")]
    assert startup.index("engine.prepare_scoring_prefill()?") < startup.index("engine.prepare_prefill_units()?")
    assert startup.index("engine.prepare_prefill_units()?") < startup.index("let result = body(&engine);")

    prepare = engine[engine.index("    pub(crate) fn prepare_prefill_units("):engine.index("    /// Pre-capture every reachable")]
    assert "let Some(shape) = self.prefill_shape else { return Ok(()) };" in prepare
    assert "self.workspace_rows(shape.unit_rows as usize, false," in prepare
    assert "shape.slot_rows() as usize" in prepare
    assert "RetainedUploads::new(" in engine
    stage = engine[engine.index("    fn stage(&self"):engine.index("    fn stage_table")]
    assert "w.staging.borrow_mut().upload(dst, bytes, self.execution_stream())" in stage
    assert "copy_h2d_ordered" not in stage
    assert "begin_staging(" not in engine and "reserve_h2d_ordered(" not in engine


def test_packed_qwen_metadata_is_uploaded_once_and_only_viewed_inside_layers():
    root = ROOT / "rust/crates/cuteafd-daemon/src/families"
    moe = (root / "qwen4/engine.rs").read_text()
    packed = moe[moe.index("    pub(crate) fn prefill_packed("):moe.index("    pub(crate) fn mtp_prefill_rows(")]
    assert packed.count("self.stage_step_tables(w, &all)?;") == 1
    assert "run_prefill_units(&mut hooks, &[unit], 1)?" in packed
    hooks = moe[moe.index("struct PackedHooks"):moe.index("struct UnitHooks")]
    assert "stage_step_tables" not in hooks
    assert "self.w.table_view.set(Some(view))" in hooks
    assert "self.w.table_view.set(None)" in hooks
    assert "local_seq_first" in moe and "seq_first\", w.first_table()" in moe
    dense = (root / "qwen3_5/engine.rs").read_text()
    packed = dense[dense.index("    pub(crate) fn prefill_packed("):dense.index("    pub(crate) fn packed_head(")]
    assert packed.count("self.row_tables(&all)?;") == 1
    layers = packed[packed.index("for (index, layer)"):]
    assert "row_tables(" not in layers
    for field in ["slots", "seq_first", "positions", "rope_positions", "kv_slots", "page_table"]:
        assert f"planes.{field} = view(" in layers


def test_qwen_joint_host_admission_precedes_lazy_allocations_on_every_layout():
    root = ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4"
    engine = (root / "engine.rs").read_text()
    for filename in ["mod.rs", "dual.rs"]:
        startup = (root / filename).read_text()
        assert startup.count("engine.admit_host_memory(args)?;") == 1
        assert startup.index("engine.admit_host_memory(args)?;") < startup.index("engine.prepare_scoring_prefill()?")
        assert startup.index("engine.admit_host_memory(args)?;") < startup.index("let result = body(&engine);")

    admit = engine[engine.index("    pub(super) fn admit_host_memory("):engine.index("    fn future_host_workspaces(")]
    assert "!self.host_memory_admitted" in admit
    assert "prefix.resolve_host_config(layout, self.max_context, *lanes)?" in admit
    assert admit.index("self.future_host_workspaces(") < admit.index("budget::admit_host_memory(")
    assert "config.as_ref().map_or(0, |c| c.bytes)" in admit
    assert admit.index("budget::admit_host_memory(") < admit.index("self.host_prefix_config = config;")
    assert "self.host_memory_admitted = true;" in admit

    future = engine[engine.index("    fn future_host_workspaces("):engine.index("    /// Size every unit-table upload")]
    assert "for owner in 0..self.gdn_banks.len()" in future
    assert "for decode in [false, true]" in future
    assert "self.workspace_slot(owner, decode)?.borrow().is_some() { continue; }" in future
    for field in ["host.staging", "host.logits"]:
        assert field in future
    assert "self.mtp_prefill_workspace.borrow().is_none()" in future
    assert "self.mtp_prefill_rows()" in future
    assert "ple.mapped()" in future
    assert '"staging/mapped-table-worker"' in future
    assert "ple.mapped().host_workspace_bytes()" in future
    assert "HostAllocation::new" not in future

    serve = (root / "serve.rs").read_text()
    assert "args.planner_host_prefix = Some((prefix.clone(), max_sequences));" in serve
    cache = serve[serve.index("fn prefix_cache"):]
    assert "engine.host_prefix_config.clone().map(|config|" in cache
    assert ".host_config(" not in cache and ".resolve_host_config(" not in cache
    assert ".with_scoring(" not in cache


def test_qwen_graph_stats_are_published_before_ready_and_on_every_update():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs").read_text()
    family = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/serve.rs").read_text()
    startup = family[family.index("let result = opened.with_engine"):family.index("fn prefix_cache")]
    assert "scheduler::schedule(&Qwen4Backend" in startup
    assert startup.index("probe::graph_capture_stats(&mut stats);") < startup.index("ready.send(Ok(")
    publish = serve[serve.index("fn publish<E:"):serve.index("pub(crate) const MESSAGE_STARTS")]
    assert publish.index("*stats = serde_json::json!") < publish.index("probe::graph_capture_stats(&mut stats);")


def test_qwen_verify_histogram_uses_existing_timer_and_physical_rows():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs").read_text()
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs").read_text()
    timer = serve[serve.index("    let elapsed = timer.elapsed().as_secs_f64();"):]
    assert timer.index("lane.stats.record(tokens.len(), engine.verify_bucket_rows(tokens.len(), spec, diagnostic)") < timer.index("resource::observe_drafted(")
    assert '"verify": verify.snapshot()' in serve
    assert serve.count("&cache, media, preparer, &lane.stats,\n") == 2
    assert "self.startup_graphs && self.use_graphs && !diagnostic" in engine
    assert "decode_bucket(rows, spec) } else { rows }" in engine
    assert '"by_real_rows"' in serve and '"by_bucket"' in serve


def test_shared_copy_extensions_precede_grammar_and_verify_width():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs").read_text()
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs").read_text()
    trim = serve[serve.index("    let sequences: Vec<Vec<u32>>"):serve.index("    let starts: Vec<usize>")]
    assert trim.index("state.truncate_proposal") < trim.index("let spec = sequences.iter().any")
    assert "Drafts::Copy" not in serve
    copy = serve.index("resource::copy_continuation(")
    select = serve.index("resource::select_drafted(", copy)
    substitute = serve.index("*proposal = copy.tokens;", select)
    assert copy < select < substitute < serve.index("let sequences: Vec<Vec<u32>>")
    route = engine[engine.index("pub(crate) fn copy_verify_row_limit"):engine.index("pub(crate) fn verify_bucket_rows")]
    assert "self.startup_graphs && self.use_graphs && !diagnostic" in route
    assert "copy_row_limit(rows, sequences)" in route and "else { rows }" in route
    gate = engine[engine.index("// Exercise the serving trim helper"):engine.index("pub(crate) fn copy_verify_row_limit")]
    assert "crate::shared::qwen::scheduler::trim_copy_rows(&mut sequences, limit)" in gate
    assert "before_rows == 37 && input.len() == 32" in gate
    assert "self.verify_device_ungraphed" in gate and "self.verify_device(" in gate
    assert "snapshot(&buffers)? == exact_state" in gate
    assert "Qwen copy trim logits and cache/state byte-exact" in gate


def test_qwen_startup_graphs_precede_ready_and_diagnostics_bypass_capture():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/shared/qwen/scheduler.rs").read_text()
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs").read_text()
    family = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/serve.rs").read_text()
    startup = family[family.index("let result = opened.with_engine"):family.index("fn prefix_cache")]
    assert "scheduler::schedule(&Qwen4Backend" in startup
    assert startup.index("engine.warm_decode_graphs(") < startup.index("probe::graph_capture_stats(")
    assert startup.index("engine.check_decode_padding(") < startup.index("ready.send(Ok(")
    scoring = serve[serve.index("|placement, chunk|"):serve.index("|placement, chunk|") + 160]
    assert "verify_device_ungraphed" in scoring
    assert "active.iter().any(|a| a.job.probe.is_some())" in serve
    assert "if diagnostic {\n        engine.verify_device_ungraphed" in serve
    assert "logits.rows = rows;" in engine
    assert "Self::region(&w.select, rows * 4, rows * 4)" in engine
    assert "Qwen serving graph was not captured at startup" in engine
    graphed = engine[engine.index("    fn decode_graphed("):engine.index("    fn replay(")]
    assert "tables.positions.iter().take_while(|&&position| position >= 0).count()" in graphed
    assert 'if self.startup_graphs { Ok(()) } else { self.moe_front(' in graphed
    host_experts = graphed[graphed.index("            cur ^= if index") :]
    assert 'ensure!(real_rows == 0, "Qwen startup MoE rows must all be masked")' in host_experts
    assert "clear_tail(0..t)?" in host_experts
    assert "real_row_moe(real_rows, t, |real_rows|" in host_experts
    assert "self.moe_front(w, index, &layers[index], real_rows, expert_rows)?" in host_experts
    assert "self.moe_experts(w, index, real_rows, expert_rows, true)" in host_experts
    assert "}, clear_tail)?" in host_experts
    assert "self.moe_experts(w, index, t, rows, true)?" in host_experts
    assert "tail.start * self.cfg.hidden * 2" in host_experts
    assert "tail.len() * self.cfg.hidden * 2" in host_experts
    shared = (ROOT / "rust/crates/cuteafd-daemon/src/shared/decode_graph.rs").read_text()
    assert shared.index("run(real)?;") < shared.index("clear(real..bucket)?;")
    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    assert 'get QWEN_STARTUP_GRAPHS' in launcher
    assert 'on) trace_args+=(-e CUTEAFD_QWEN4_STARTUP_GRAPHS=1)' in launcher
    assert 'off) trace_args+=(-e CUTEAFD_QWEN4_STARTUP_GRAPHS=0)' in launcher


def test_qwen_startup_shapes_drain_without_serving_warmup_reachability():
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs").read_text()

    def check(source):
        start = source.index("    pub fn warm_decode_graphs(")
        end = source.index("    /// Compare real logits", start)
        warm = source[start:end]
        loop = warm[warm.index("            for &(rows, spec, geometry) in &shapes {"):]
        loop = loop[:loop.index("\n            }")]
        assert loop.index("self.step(") < loop.index("self.drain_state()?;")
        # No engine execution helper may enter startup warm-up, even indirectly.
        assert "warm_decode_graphs(" not in source[:start] + source[end:]

    check(engine)
    drain = "                self.drain_state()?;\n"
    warm_start = engine.index("    pub fn warm_decode_graphs(")
    drain_at = engine.index(drain, warm_start)
    with pytest.raises((AssertionError, ValueError)):
        check(engine[:drain_at] + engine[drain_at + len(drain):])
    with pytest.raises(AssertionError):
        check(engine + "\nfn serving_step() { self.warm_decode_graphs(1, false); }\n")


@pytest.mark.parametrize("family", ["qwen4", "mimo_v2", "glm5_flash", "glm5"])
@pytest.mark.parametrize("vision", ["off", "auto", "rtx", "spark:1"])
@pytest.mark.parametrize("quota", ["", "2621440"])
def test_media_cache_flag_only_reaches_enabled_vision(family, vision, quota):
    launcher = (ROOT / "scripts/launch/run-family.sh").read_text()
    start = launcher.index('if [[ ( $family == mimo_v2')
    block = launcher[start:launcher.index('if [[ $family == mimo_v2 ]]; then', start)]
    program = '''
family=$1; vision=$2; quota=$3
get() { printf '%s' "$quota"; }
family_args=()
''' + block + '\nif ((${#family_args[@]})); then printf "%s\\n" "${family_args[@]}"; fi\n'
    result = subprocess.run(["bash", "-c", program, "test", family, vision, quota],
                            check=True, capture_output=True, text=True)
    expected = ["--media-cache-bytes", quota] if family in ("qwen4", "mimo_v2", "glm5_flash") and vision != "off" and quota else []
    assert result.stdout.splitlines() == expected


def test_qwen_interpolation_is_multiply_then_divide_not_ratio():
    # At rectangular sizes a precomputed ratio changes taps by an FP32 ULP.
    positions = np.arange(256,dtype=np.float32)
    official = positions*np.float32(47)/np.float32(255)
    ratio = positions*np.float32(np.float32(47)/np.float32(255))
    assert np.any(official != ratio)
    source = (ROOT/"native/shared/cuda/vision_qwen.cuh").read_text()
    assert "__fdiv_rn(float(y)*47.f,float(gh-1))" in source
    assert "__fdiv_rn(float(x)*47.f,float(gw-1))" in source


def test_qwen_dead_pool_map_and_route_staging_are_not_retained():
    root = ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4"
    engine = (root / "engine.rs").read_text()
    admission = (root / "admission.rs").read_text()
    prefix = (root / "prefix.rs").read_text()
    for dead in ["pool_logical", "router_host", "map_pools("]:
        assert dead not in engine
    assert "map_pools(" not in prefix
    assert "pub router:" not in admission
    assert "host.router" not in engine


def packed_spec_fields(snapshot, family, placements=None):
    """Execute the actual ctypes packer without allocating checkpoint payloads."""
    import json
    snapshot = Path(snapshot)
    gate = module(f"python/tools/qualify/{family}/qualify-vision.py", f"storage_{family}")
    config = json.loads((snapshot / "config.json").read_text())
    prefix = "visual." if family == "mimo_v2" else "model.visual."
    with (snapshot / "vision.safetensors").open("rb") as source:
        header = json.loads(source.read(int.from_bytes(source.read(8), "little")))

    arena, cursor = {}, [0]

    class SizedBytes(bytes):
        def __new__(cls, size):
            result = super().__new__(cls, b"")
            result.size = size
            return result
        def __len__(self):
            return self.size

    # Exercise the actual elementwise packer operations on every BF16 bit
    # pattern, without materializing the towers' multi-GiB matrix payloads.
    patterns = np.arange(65536, dtype=np.uint16)

    class ShapeOnly:
        def __init__(self, shape, dtype=np.uint16, pieces=(), sample=None):
            self.shape, self.dtype = tuple(shape), np.dtype(dtype)
            self.nbytes = int(np.prod(shape)) * self.dtype.itemsize
            self.pieces = pieces
            self.sample = patterns if sample is None else sample
        def astype(self, dtype):
            return ShapeOnly(self.shape, dtype, self.pieces, self.sample.astype(dtype))
        def __lshift__(self, bits):
            return ShapeOnly(self.shape, self.dtype, self.pieces, self.sample << bits)
        def view(self, dtype):
            return ShapeOnly(self.shape, dtype, self.pieces, self.sample.view(dtype))
        def tobytes(self):
            expected = patterns.tobytes() if self.dtype == np.uint16 else (patterns.astype(np.uint32) << 16).tobytes()
            assert self.sample.tobytes() == expected, self.pieces
            cursor[0] = (cursor[0] + 255) & ~255
            offset = cursor[0]
            for name, count in self.pieces:
                length = count * self.dtype.itemsize
                arena[name] = (offset, length)
                offset += length
            assert offset == cursor[0] + self.nbytes
            cursor[0] = offset
            return SizedBytes(self.nbytes)

    tensors = {name.removeprefix(prefix): ShapeOnly(entry["shape"], pieces=((name.removeprefix(prefix), int(np.prod(entry["shape"]))),))
               for name, entry in header.items() if name.startswith(prefix)}
    concatenate = np.concatenate
    def shapes_only(values, *args, **kwargs):
        if isinstance(values[0], ShapeOnly):
            assert all(value.sample.tobytes() == values[0].sample.tobytes() for value in values)
            return ShapeOnly((sum(value.shape[0] for value in values), *values[0].shape[1:]), values[0].dtype,
                tuple(piece for value in values for piece in value.pieces), values[0].sample)
        return concatenate(values, *args, **kwargs)
    np.concatenate = shapes_only
    try:
        spec, _ = gate.pack_spec(config, tensors)
        # Evaluate the unmodified pre-change packer and fixed-array declaration
        # from this branch's base. No checkpoint allocation or CUDA imports.
        old_common_source = subprocess.check_output(["git", "-C", str(ROOT), "show",
            "34b0d33b1985441746df5e2fe02c2ae3a9a29f6f:python/tools/qualify/mimo_v2/qualify-vision.py"], text=True)
        common_ast = ast.parse(old_common_source)
        selected = [node for node in common_ast.body if isinstance(node, ast.ClassDef)
                    and node.name in ("Block", "Spec")]
        # The second packing run records old placements independently.
        new_arena = dict(arena)
        cursor[0] = 0
        arena.clear()
        scope = {"C": ctypes, "np": np, "NO_OFFSET": (1 << 64) - 1,
                 "FIELDS": gate.common.FIELDS if hasattr(gate, "common") else gate.FIELDS}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "old fixed vision ABI", "exec"), scope)
        import types
        old_common = types.SimpleNamespace(Spec=scope["Spec"], Block=scope["Block"], NO_OFFSET=(1 << 64) - 1)
        old_source = subprocess.check_output(["git", "-C", str(ROOT), "show",
            f"34b0d33b1985441746df5e2fe02c2ae3a9a29f6f:python/tools/qualify/{family}/qualify-vision.py"], text=True)
        nodes = ast.parse(old_source).body
        scope.update(common=old_common, Block=old_common.Block)
        if family != "mimo_v2":
            declaration = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == "Spec")
            exec(compile(ast.Module(body=[declaration], type_ignores=[]), "old tower ABI", "exec"), scope)
        pack = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == "pack_spec")
        exec(compile(ast.Module(body=[pack], type_ignores=[]), "old tower packer", "exec"), scope)
        fixed, _ = scope["pack_spec"](config, tensors)
        assert arena == new_arena
        # Compare every old scalar, complete live block bytes and layer offsets;
        # only storage representation (pointers vs arrays) is allowed to change.
        for name, kind in fixed._fields_:
            if name == "blocks":
                assert bytes(fixed.blocks)[:spec.depth * 96] == ctypes.string_at(spec.blocks, spec.depth * 96)
            elif name in ("norm1_bias", "norm2_bias", "q_norm", "k_norm"):
                assert list(getattr(fixed, name))[:spec.depth] == [getattr(spec, name)[i] for i in range(spec.depth)]
            elif issubclass(kind, ctypes.Array):
                assert bytes(getattr(fixed, name)) == bytes(getattr(spec, name))
            else:
                assert getattr(fixed, name) == getattr(spec, name), name
    finally:
        np.concatenate = concatenate
    # Qwen's probe admits 4096 while serving intentionally admits 1024. Compare
    # the same admitted request capacity rather than changing probe behavior.
    if family == "qwen4":
        spec.max_tokens = 1024
    remap = {}
    if placements is not None:
        # No numerical data is materialized here: BF16 matrices are copied and
        # BF16 vectors are widened identically. Each tensor's full byte extent
        # and destination are checked, including the second fused gate/up part.
        assert set(placements) == set(arena)
        for name, (offset, length) in arena.items():
            destination, expected_length = placements[name]
            assert length == expected_length, name
            remap[offset] = destination
    values = []
    layers = ("blocks", "norm1_bias", "norm2_bias", "q_norm", "k_norm")
    for name, _ in spec._fields_:
        if name in layers or name == "storage_abi":
            continue
        value = getattr(spec, name)
        if name == "norm_eps":
            import struct
            values.append(int.from_bytes(struct.pack("<f", value), "little"))
        elif name == "merger_extra":
            values.extend(value)
        else:
            if name in ("patch", "merger_norm", "merger_fc1", "merger_fc2", "patch_bias",
                        "pos_embed", "merger_norm_bias", "merger_fc1_bias", "merger_fc2_bias"):
                value = remap.get(value, value)
            values.append(value)
    # Merger extras are offsets, not geometry.
    values[25:33] = [remap.get(value, value) for value in values[25:33]]
    for i in range(spec.depth):
        block = spec.blocks[i]
        values.extend(remap.get(getattr(block, name), getattr(block, name))
                      if name not in ("window", "column_order") else getattr(block, name)
                      for name, _ in block._fields_)
    for name in layers[1:]:
        values.extend(remap.get(getattr(spec, name)[i], getattr(spec, name)[i]) for i in range(spec.depth))
    return values


@pytest.mark.parametrize("depth", [24, 27, 28, 30, 50])
def test_ctypes_storage_is_owned_at_tower_depth(depth):
    import gc
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "owned_depth")
    spec = gate.Spec(depth=depth)
    spec.blocks[depth - 1].norm1 = 1234
    spec.q_norm[depth - 1] = gate.common.NO_OFFSET
    gc.collect()
    assert spec.storage_abi == 3
    assert len(spec._layer_storage["blocks"]) == depth
    assert spec.blocks[depth - 1].norm1 == 1234
    assert spec.q_norm[depth - 1] == gate.common.NO_OFFSET
