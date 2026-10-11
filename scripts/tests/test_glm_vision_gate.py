"""CPU contracts for GLM's ABI2 tower qualification; no CUDA imports."""
import ctypes
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_gate():
    path = ROOT / "python/tools/qualify/glm5_flash/qualify-vision.py"
    spec = importlib.util.spec_from_file_location("glm_vision_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glm_abi_preserves_mimo_prefix():
    gate = load_gate()
    assert ctypes.sizeof(gate.Block) == 96
    assert gate.Spec.blocks.offset == 64
    assert gate.Spec.hidden.offset == 2752
    assert gate.Spec.patch_bias.offset == 2792
    assert gate.Spec.norm1_bias.offset == 2896
    assert ctypes.sizeof(gate.Spec) == 3792


def test_glm_patch_order_duplicates_temporal_axis_and_keeps_merge_units():
    gate = load_gate()
    rgb = np.arange(28 * 56 * 3, dtype=np.uint16).astype(np.uint8).reshape(28, 56, 3)
    lut = gate.common.normalization_lut()
    values = gate.patches(rgb, lut)
    assert values.shape == (8, 1176)
    for patch in range(8):
        unit, within = divmod(patch, 4)
        y, x = within // 2 * 14, unit * 28 + within % 2 * 14
        for channel in range(3):
            expected = lut[channel, rgb[y:y + 14, x:x + 14, channel]].reshape(-1)
            np.testing.assert_array_equal(values[patch, channel * 392:channel * 392 + 196], expected)
            np.testing.assert_array_equal(values[patch, channel * 392 + 196:(channel + 1) * 392], expected)


def test_glm_qualifier_uses_snapshot_normalization_not_fixed_defaults(tmp_path):
    import json
    gate = load_gate()
    processor = dict(patch_size=14, temporal_patch_size=2, merge_size=2,
        do_rescale=True, image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6])
    path = tmp_path / "processor_config.json"
    path.write_text(json.dumps({"image_processor": processor}))
    mean, std = np.asarray(processor["image_mean"], np.float32), np.asarray(processor["image_std"], np.float32)
    expected = (np.arange(256, dtype=np.float32)[None, :] * np.float32(1 / 255) - mean[:, None]) / std[:, None]
    np.testing.assert_array_equal(gate.normalization_lut(tmp_path), expected)
    assert not np.array_equal(expected, gate.common.normalization_lut())
    for changed in (dict(patch_size=16), dict(image_std=[0, 1, 1]), dict(do_rescale=False)):
        path.write_text(json.dumps({"image_processor": dict(processor, **changed)}))
        with pytest.raises(ValueError):
            gate.normalization_lut(tmp_path)


def test_glm_shapes_and_calibrated_floor_keep_strict_failure():
    gate = load_gate()
    for count in (256, 1024, 4096):
        gh, gw, rgb = gate.fixture(count)
        assert gh * gw == count * 4
        assert rgb.shape == (gh * 14, gw * 14, 3)
    floor = dict(relative_l2=0.035, mean_cosine=0.9996, worst_cosine=0.97, pass_=False)
    native = dict(relative_l2=0.036, mean_cosine=0.99957, worst_cosine=0.9695, **{"pass": False})
    measured = gate.common.calibrated_metrics(native, floor)
    assert measured["pass"] and not measured["strict_pass"]
    assert not gate.common.calibrated_metrics(dict(native, mean_cosine=0.99949), floor)["pass"]


def test_glm_bf16_relative_floor_preserves_strict_merger_miss():
    gate = load_gate()
    floor = dict(relative_l2=0.35, mean_cosine=0.9968, worst_cosine=0.13, **{"pass": False})
    native = dict(relative_l2=0.20, mean_cosine=0.9987, worst_cosine=0.38, **{"pass": False})
    calibrated = gate.calibrated_metrics(native, floor, "16")
    assert calibrated["pass"] and not calibrated["strict_pass"]
    merger = gate.calibrated_metrics(native, floor, "26")
    assert merger["pass"] and not merger["strict_pass"]
    for changed in (dict(relative_l2=0.353), dict(mean_cosine=0.9967), dict(worst_cosine=0.128)):
        assert not gate.calibrated_metrics(dict(native, **changed), floor, "16")["pass"]


def test_glm_pointwise_kernels_cover_capped_grid_tail():
    import re
    source = (ROOT / "native/shared/cuda/vision_glm.cuh").read_text()
    for name in ("glm_rgb_patches", "glm_bias", "glm_conv_gather"):
        body = re.search(r"__global__ void " + name + r"\(.*?(?=\n(?:__global__|//|int encode_glm))", source, re.S).group()
        assert "i+=size_t(gridDim.x)*blockDim.x" in body, name
    assert "namespace {" not in source, "included inside the shared owner's anonymous namespace"


def test_glm_g1_reads_actual_nested_processor(tmp_path):
    pytest.importorskip("PIL")
    import json
    path = ROOT / "scripts/qualify/media/preprocess-goldens.py"
    spec = importlib.util.spec_from_file_location("preprocess_goldens_glm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.glm_processor_config()
    changed = dict(config, min_image_tokens=32, max_image_tokens=1024)
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": changed,
        "video_processor": {"patch_size": 99}}))
    assert module.glm_processor_config(tmp_path) == changed
    for invalid in ({}, {"image_processor": []}, {"image_processor": {"patch_size": 16}}):
        (tmp_path / "processor_config.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            module.glm_processor_config(tmp_path)


def test_glm_memory_diagnostics_do_not_change_cuda_state():
    source = (ROOT / "rust/crates/cuteafd-ffi/src/memory_ledger.rs").read_text()
    binding = source.split("fn loaded_cuda_runtime()", 1)[1].split("fn cuda_pool_snapshot", 1)[0]
    assert "RTLD_NOLOAD" in binding and "2 | 0x4" in binding
    query = source.split("pub fn current_cuda_memory_snapshot()", 1)[1].split("pub fn device_memory", 1)[0]
    for symbol in ("cudaGetDevice", "cudaMemGetInfo", "cudaDeviceGetMemPool", "cudaMemPoolGetAttribute"):
        assert symbol in query
    for forbidden in ("cudaSetDevice", "cudaDeviceSynchronize", "cudaStreamSynchronize",
                      "cudaMemPoolTrimTo", "cudaMalloc", "cudaFree"):
        assert forbidden not in query
    assert "serde_json::Value::Null" in query
    serving = (ROOT / "rust/crates/cuteafd-daemon/src/families/glm5_flash/serve.rs").read_text()
    idle = serving.split("fn publish(", 1)[1].split("fn schedule(", 1)[0]
    assert idle.index("if active == 0 && prefilling == 0") < idle.index("current_cuda_memory_snapshot()")
    assert '"first_prefill_boundaries"' in idle
    assert "first_prefill_sample(&first_prefill_seen" in serving
    assert "media encoder preparation precedes this boundary" in serving


def test_blas_handle_diagnostics_log_only_after_successful_creation():
    source = (ROOT / "native/shared/cuda/linear.cu").read_text()
    logging = source.split("void log_blas_handle_created(", 1)[1].split("cuteafd_status_t cublas_handle", 1)[0]
    assert "pthread_getname_np" in logging
    assert "configured_workspace=runtime-default configured_workspace_bytes=unknown" in logging
    assert "cuda" not in logging and "cublas" not in logging
    blas = source.split("cuteafd_status_t cublas_handle(", 1)[1].split("\n}", 1)[0]
    lt = source.split("cublasLtHandle_t cublaslt_handle()", 1)[1].split("\n}", 1)[0]
    assert blas.index("cublasCreate(&handle)") < blas.index("return status_from_cublas(status)") < blas.index('log_blas_handle_created("cublas")')
    assert lt.index("cublasLtCreate(&handle)") < lt.index("return nullptr") < lt.index('log_blas_handle_created("cublasLt")')
    assert "static thread_local" in blas and "static thread_local" in lt
    assert "cublasSetWorkspace" not in blas + lt


def test_blas_plan_workspace_log_names_successful_allocation_and_shape_cache():
    source = (ROOT / "native/shared/cuda/linear.cu").read_text()
    plan = source.split("CublasLtM1ParityBatchedPlan* create_cublaslt_m1_parity_plan(", 1)[1].split("\n}", 1)[0]
    allocated = plan.split("  char thread_name[16]", 1)[1]
    assert plan.index("cudaMalloc(&plan->workspace, plan->workspace_bytes)") < plan.index("  char thread_name[16]")
    assert "pthread_getname_np" in allocated
    assert "rows=%zu input_dim=%zu output_dim=%zu workspace_bytes=%zu" in allocated
    assert "scope=device-shape-cache allocator=cudaMalloc" in allocated
    assert "thread_name, rows, input_dim, output_dim, plan->workspace_bytes" in allocated
    assert "cudaGetDevice" not in allocated and "cudaMemGetInfo" not in allocated
    assert "cudaMalloc(" not in allocated and "cudaFree(" not in allocated
    cache = source.split("CublasLtM1ParityBatchedPlan* cublaslt_m1_parity_plan(", 1)[1].split("\n}", 1)[0]
    for key in ("std::to_string(device)", "std::to_string(rows)", "std::to_string(input_dim)", "std::to_string(output_dim)"):
        assert key in cache


def test_glm_serving_admits_and_precreates_all_reachable_workspaces():
    root = ROOT / "rust/crates/cuteafd-daemon/src/families"
    source = (root / "glm5_flash/engine.rs").read_text()
    # Every workspace buffer comes from the loader's arithmetic, which the reserve charges: a lane's
    # own buffers over its GPU's shared temporaries.
    temporaries = source.split("fn temporaries(", 1)[1].split("fn lane(", 1)[0]
    assert "glmf_temporary_bytes(" in temporaries and "self.alloc(bytes.scratch)" in temporaries
    lane = source.split("fn lane(", 1)[1].split("/// ModelOpt NVFP4 dense MLPs", 1)[0]
    assert "glmf_lane_bytes(" in lane and "self.alloc(bytes.streams)" in lane
    reserve = source.split("pub(crate) fn workspace_reserve(", 1)[1].split("/// The prefill lanes a serving engine", 1)[0]
    assert "plan.workspace_bytes(" in reserve and "workspace_reserve_bytes(" in reserve
    prepare = source.split("pub fn prepare_serving_workspaces(", 1)[1].split("/// Capture the complete", 1)[0]
    for required in ("self.decode_workspace_of(rank)", "self.pipelined()", "self.prefill_lane_count",
                     "self.prefill_lanes_of(rank, lanes)", "drafter.prepare_workspace()", "self.synchronize()"):
        assert required in prepare
    # Scoring (`--full-prefill-logits`) precreates the same lanes: a serial prefill runs in lane 0.
    scoring = source.split("pub fn prepare_scoring_prefill(", 1)[1].split("fn step_plan(", 1)[0]
    assert "self.prefill_lanes_of(rank, lanes)" in scoring and "self.workspace(" not in scoring
    # The per-workspace runtime allowance is the loader's shared definition; where the measured
    # loaded code covers it (`placement::inventory::LOADED_CODE`) the reserve charges that instead.
    graphs = (ROOT / "rust/crates/cuteafd-loader/src/serving_capacity/glmf_graphs.rs").read_text()
    assert "WORKSPACE_RUNTIME_OVERHEAD_BYTES: u64 = 72 << 20" in graphs
    assert "measured/calibrated allowance is not exact cuBLAS allocator ownership" in source
    assert "kda.in[24896|12576,4096]" in source
    module = (root / "glm5_flash/mod.rs").read_text()
    gathering = module.split("fn admit(", 1)[1].split("pub fn with_engine", 1)[0]
    admission = gathering.index("admission::solve_with_graphs(")
    for required in ("RuntimeInventory::measure(", "pending_code(", "resident_weights(",
                     "glmf_step_workspaces(", "admission::startup_graphs("):
        assert gathering.index(required) < admission
    assert "representation, resident, router_replica_bytes, workspace, graphs" in gathering
    opening = module.split("pub fn with_engine", 1)[1]
    assert opening.index("self.admit(") < opening.index("TokenEmbedding::load(")
    assert "admitted_graphs.lifetime" in opening
    assert 'full_prefill_logits_bytes(' not in opening
    assert opening.index("engine.prepare_serving_workspaces()") < opening.index("let result = body(&engine)")
    assert opening.index("engine.prepare_scoring_prefill()") < opening.index("let result = body(&engine)")
    assert "if args.serving_graph_policy.is_some()" in opening
    # Shared DFlash only gains an explicit API; all existing users keep lazy timing.
    drafter = (root / "glm5/dflash.rs").read_text()
    prepare = drafter.split("pub(crate) fn prepare_workspace(", 1)[1].split("fn workspace(", 1)[0]
    assert "self.workspace(self.max_sequences)" in prepare
    assert "prepare_workspace()" not in drafter
