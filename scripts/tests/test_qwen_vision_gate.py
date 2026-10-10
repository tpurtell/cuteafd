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
    glm = (fork / "integration/cuteafd/_glm_kernels.py").read_text()
    assert int(re.search(r"WIDE_SKINNY_MAX_ROWS = (\d+)", glm).group(1)) == rust("GLM_BF16_SKINNY_ROWS")
    assert "max_skinny_rows=WIDE_SKINNY_MAX_ROWS if n >= 2048 else None" in glm
    fp8 = (fork / "integration/cuteafd/_fp8_weights.py").read_text()
    assert int(re.search(r"FP8_GEMV_ROWS = (\d+)", fp8).group(1)) == rust("FP8_PROJECTION_SKINNY_ROWS")
    assert "self.inner = glm_projection(n, k)" in fp8
    fp8_ast = ast.parse(fp8)
    projection_class = next(n for n in fp8_ast.body if isinstance(n, ast.ClassDef) and n.name == "Fp8Projection")
    init = next(n for n in projection_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = {arg.arg: ast.literal_eval(default) for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults)
                if arg.arg in ("gemv_rows", "wide_rows")}
    assert defaults["gemv_rows"] == rust("FP8_PROJECTION_SKINNY_ROWS")
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


def test_head72_uses_the_b12x_le128_tile():
    export = module("python/tools/aot/export_b12x_vision_attention_aot.py", "vision_export")
    assert export.tile_for_head_dim(64) == (128,128)
    assert export.tile_for_head_dim(72) == (128,64)
    assert export.tile_for_head_dim(128) == (128,64)
    for dim in (0,73,136):
        with pytest.raises(ValueError): export.tile_for_head_dim(dim)


def test_qwen_abi_keeps_mimo_prefix_and_matches_c_rust_suffix():
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "qwen_vision_gate")
    assert ctypes.sizeof(gate.Block) == 96
    assert ctypes.sizeof(gate.common.Spec) == 2752
    assert ctypes.sizeof(gate.Spec) == 3792
    assert gate.Spec.hidden.offset == 2752
    assert gate.Spec.patch_bias.offset == 2792
    assert gate.Spec.norm1_bias.offset == 2896


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
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/serve.rs").read_text()
    api = (ROOT / "rust/crates/cuteafd-api/src/openai/probe.rs").read_text()
    assert serve.index("probe.spec.validate_cold_steps(") < serve.index('probe::admitted(&job.probe, "qwen4"')
    assert "probe.spec.cold_steps.iter().map(|step| step.end).collect(), points: Vec::new()" in serve
    assert "probe.spec.cold_steps.get(p.chunks)" in serve
    assert "let logits = if decode {\n                        engine.verify_device_ungraphed" in serve
    assert "} else { engine.prefill_device(&mut p.placement, chunk, None, None, 1)? };" in serve
    assert 'matches!(engine, "mimo_v2" | "glm5_flash" | "qwen4")' in api
    assert "else if logits.is_some() { PointPlan::default() } else { plan }" in serve


def test_qwen_graph_stats_are_published_before_ready_and_on_every_update():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/serve.rs").read_text()
    startup = serve[serve.index("let result = opened.with_engine"):serve.index("struct Active")]
    assert startup.index("probe::graph_capture_stats(&mut stats);") < startup.index("ready.send(Ok(")
    publish = serve[serve.index("fn publish("):serve.index("pub(crate) const MESSAGE_STARTS")]
    assert publish.index("*stats = serde_json::json!") < publish.index("probe::graph_capture_stats(&mut stats);")


def test_qwen_verify_histogram_uses_existing_timer_and_physical_rows():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/serve.rs").read_text()
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs").read_text()
    timer = serve[serve.index("        let elapsed = timer.elapsed().as_secs_f64();"):]
    assert timer.index("verify_stats.record(tokens.len(), engine.verify_bucket_rows(tokens.len(), spec, diagnostic)") < timer.index("cost.observe_verify(")
    assert '"verify": verify.snapshot()' in serve
    assert serve.count("&cache, media, preparer, &verify_stats);") == 2
    assert "self.startup_graphs && self.use_graphs && !diagnostic" in engine
    assert "decode_bucket(rows, spec) } else { rows }" in engine
    assert '"by_real_rows"' in serve and '"by_bucket"' in serve


def test_copy_trim_is_opt_in_and_precedes_verify_width_and_selection():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/serve.rs").read_text()
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs").read_text()
    trim = serve[serve.index("        let mut sequences: Vec<Vec<u32>>"):serve.index("        let starts: Vec<usize>")]
    assert trim.index("state.truncate_proposal") < trim.index("trim_copy_rows(&mut sequences, limit)")
    assert trim.index("trim_copy_rows(&mut sequences, limit)") < trim.index("let spec = sequences.iter().any")
    assert "if matches!(drafts, Drafts::Copy)" in trim
    route = engine[engine.index("pub(crate) fn copy_verify_row_limit"):engine.index("pub(crate) fn verify_bucket_rows")]
    assert "self.startup_graphs && self.use_graphs && !diagnostic" in route
    assert "copy_row_limit(rows, sequences)" in route and "else { rows }" in route
    gate = engine[engine.index("// Exercise the serving trim helper"):engine.index("pub(crate) fn copy_verify_row_limit")]
    assert "super::serve::trim_copy_rows(&mut sequences, limit)" in gate
    assert "before_rows == 37 && input.len() == 32" in gate
    assert "self.verify_device_ungraphed" in gate and "self.verify_device(" in gate
    assert "snapshot(&buffers)? == exact_state" in gate
    assert "Qwen copy trim logits and cache/state byte-exact" in gate


def test_qwen_startup_graphs_precede_ready_and_diagnostics_bypass_capture():
    serve = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/serve.rs").read_text()
    engine = (ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs").read_text()
    startup = serve[serve.index("let result = opened.with_engine"):serve.index("struct Active")]
    assert startup.index("engine.warm_decode_graphs(") < startup.index("probe::graph_capture_stats(")
    assert startup.index("engine.check_decode_padding(") < startup.index("ready.send(Ok(")
    scoring = serve[serve.index("|placement, chunk|"):serve.index("|placement, chunk|") + 160]
    assert "verify_device_ungraphed" in scoring
    assert "active.iter().any(|a| a.job.probe.is_some())" in serve
    assert "if diagnostic {\n            engine.verify_device_ungraphed" in serve
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
