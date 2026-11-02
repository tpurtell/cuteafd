"""CPU-only contracts for Kimi/GLM core shapes and shared ABI exports."""
import ast
import json
import os
import struct

import pytest
from pathlib import Path
import runpy
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "python/tools/aot/export_b12x_dense_trellis_k6_aot.py"


def load_exporter(monkeypatch):
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", ModuleType("_pinned_sparkinfer"))
    # The exporter prepends its lib path like every standalone AOT entry point.
    monkeypatch.setattr(sys, "path", list(sys.path))
    return runpy.run_path(str(EXPORTER))


def test_local_shapes_and_sides(monkeypatch):
    shapes = load_exporter(monkeypatch)["KIMI_SHAPES"]
    assert len({name for name, *_ in shapes}) == len(shapes)
    assert {(k, n, side) for _, k, n, side in shapes} == {
        (7168, 6144, 1), (6144, 7168, 2), (7168, 3072, 1), (3072, 7168, 2),
        (7168, 3584, 1), (3584, 7168, 1), (7168, 1536, 1), (7168, 640, 1),
        (1536, 18432, 1), (1536, 9216, 1), (7168, 12288, 1),
        (12288, 7168, 1), (7168, 163840, 1),
    }
    assert all(k % 128 == n % 128 == 0 for _, k, n, _ in shapes)
    assert not any(n == 576 or k == 1792 or n == 1792 for _, k, n, _ in shapes)


def test_only_shared_native_compiler_and_both_codebooks(monkeypatch):
    module = load_exporter(monkeypatch)
    fake = ModuleType("b12x.integration.cuteafd.trellis_linear")
    fake.compile_trellis_linear_aot = lambda **params: params
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    transforms = ModuleType("b12x.integration.cuteafd.trellis_transform")
    transforms.compile_trellis_input_aot = lambda **params: params
    transforms.compile_trellis_output_aot = lambda **params: params
    monkeypatch.setitem(sys.modules, transforms.__name__, transforms)
    programs = module["programs"]()
    assert len(programs) == 46
    assert len({name for name, *_ in programs}) == len(programs)
    grouped = [item for item in programs if item[2].get("groups")]
    assert {name for name, *_ in grouped} == {
        "trellis_kda_qkvg_tp2_mcg_g4", "trellis_kda_qkvg_tp2_mul1_g4",
        "trellis_input_k7168_bf16_g4", "trellis_output_n6144_bf16_g4"}
    for name, op, params, compile_program in grouped:
        assert params["groups"] == 4 and params["max_rows"] == 4096
        assert compile_program() == params
        if op == "trellis_linear":
            assert params["rotation_count"] == 4 and params["sharded_side"] == 1 and params["bits"] == 6
        elif op == "trellis_output":
            assert params["output_dtype"] == "bf16"
    programs = [item for item in programs if not item[2].get("groups")]
    core = [item for item in programs if item[1] == "trellis_linear"]
    assert len(core) == 2 * len(module["KIMI_SHAPES"])
    for _, op, params, compile_program in core:
        assert op == "trellis_linear"
        assert params["bits"] == 6
        assert params["codebook"] in ("mcg", "mul1")
        assert params["max_rows"] == 4096
        assert params["rotation_count"] == 1
        assert compile_program() == params
    ingress = [item for item in programs if item[1] == "trellis_input"]
    egress = [item for item in programs if item[1] == "trellis_output"]
    assert len(ingress) == 6
    assert {params["features"] for _, _, params, _ in ingress} == {1536, 3072, 3584, 6144, 7168, 12288}
    assert {params["features"] for _, _, params, _ in egress} == {640, 1536, 3072, 3584, 6144, 7168, 9216, 12288, 18432, 163840}
    for name, op, params, compile_program in ingress + egress:
        assert params["max_rows"] == 4096
        assert compile_program() == params
        if op == "trellis_input":
            assert params["input_dtype"] == "bf16"
            suffix = "_rotated_bf16" if params.get("output_dtype") == "bf16" else ""
            assert name == f"trellis_input_k{params['features']}_bf16{suffix}"
        else:
            dtype = "f32" if params["features"] == 163840 else "bf16"
            assert params["output_dtype"] == dtype
            assert name == f"trellis_output_n{params['features']}_{dtype}"
    for _, op, params, compile_program in programs:
        assert compile_program() == params
    names = {name for name, *_ in programs} | {name for name, *_ in grouped}
    for policy in module["row_policies"]():
        for field in ("native_program", "native_input", "output_program"):
            assert policy[field] in names
        if "grouped" in policy:
            assert policy["shape"] == "kda_qkvg_tp2"
            assert policy["grouped"]["native_program"] == f"trellis_kda_qkvg_tp2_{policy['codebook']}_g4"
            for field in ("native_program", "native_input", "output_program"):
                assert policy["grouped"][field] in names
    assert sum("grouped" in policy for policy in module["row_policies"]()) == 2
    fixture = module["programs_for_shapes"]((("s1", 128, 256, 1), ("s2", 128, 256, 2)),
        max_rows=64, f32_outputs=(256,))
    assert {name for name, *_ in fixture} == {
        "trellis_s1_mcg", "trellis_s1_mul1", "trellis_s2_mcg", "trellis_s2_mul1",
        "trellis_input_k128_bf16", "trellis_output_n256_f32"}
    assert all(params["max_rows"] == 64 for _, _, params, _ in fixture)
    imports = [node.module for node in ast.walk(ast.parse(EXPORTER.read_text()))
               if isinstance(node, ast.ImportFrom)]
    assert "b12x.gemm.trellis_linear" not in imports
    assert "b12x.integration.cuteafd.trellis_linear" in imports


def test_comparison_group_is_opt_in():
    groups = runpy.run_path(str(ROOT / "scripts/build/compare-sm-exports.py"))["jobs"]("all")
    group = next(row for row in groups if row[0] == "kimi-k6-dense")
    assert group[1:] == ("export_b12x_dsv4_aot.py", ["--geometry", "dense_trellis_k6,kimi_split"])


def test_native_only_policy_has_no_dequant_scratch(monkeypatch):
    policies = load_exporter(monkeypatch)["row_policies"]()
    assert len(policies) == 26
    for policy in policies:
        assert not any("chunk" in key or "dequant" in key or "scratch" in key for key in policy)
        assert "wide_input" not in policy
        if "grouped" in policy:
            assert policy["grouped"]["max_rows"] == 4096
    head = next(p for p in policies if p["shape"] == "lm_head")
    assert head["output_program"].endswith("_f32")


def test_bf16_diagnostic_specialization_preserves_stems_and_default(monkeypatch):
    module = load_exporter(monkeypatch)
    fake = ModuleType("b12x.integration.cuteafd.trellis_linear")
    fake.compile_trellis_linear_aot = lambda **params: params
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    transforms = ModuleType("b12x.integration.cuteafd.trellis_transform")
    transforms.compile_trellis_input_aot = lambda **params: params
    transforms.compile_trellis_output_aot = lambda **params: params
    monkeypatch.setitem(sys.modules, transforms.__name__, transforms)
    default = module["programs"]()
    bf16 = module["programs"](input_dtype="bf16")
    assert [p[0] for p in default] == [p[0] for p in bf16]
    assert all(a[:3] == b[:3] for a, b in zip(default, module["programs"](input_dtype="f16")))
    for (_, op, params, thunk), (_, _, original, _) in zip(bf16, default):
        assert thunk() == params
        if op == "trellis_linear":
            assert params["input_dtype"] == "bf16"
            assert original.get("input_dtype", "f16") == "f16"
        elif op == "trellis_input":
            assert params["output_dtype"] == "bf16"
            assert original.get("output_dtype", "f16") == "f16"
        else:
            assert params == original


def test_glm_architecture_geometries_and_unique_policy_lookup(monkeypatch):
    module = load_exporter(monkeypatch)
    shapes = module["shapes_for"](("glm",))
    assert {(k, n, side) for _, k, n, side in shapes} == {
        (6144, 2048, 1), (6144, 640, 1), (2048, 16384, 1), (2048, 8192, 1),
        (16384, 6144, 1), (8192, 6144, 2), (6144, 12288, 1), (6144, 6144, 1),
        (12288, 6144, 1), (6144, 6144, 2), (6144, 1024, 1), (2048, 6144, 1),
        (1024, 6144, 2), (6144, 154880, 1),
    }
    policies = module["row_policies"](("kimik3", "glm"))
    keys = [(p["input_features"], p["output_features"], p["sharded_side"], p["codebook"])
            for p in policies]
    assert len(set(keys)) == len(keys)
    assert all(k % 128 == n % 128 == 0 for _, k, n, _ in shapes)
    head = next(p for p in policies if p["shape"] == "glm_lm_head")
    assert head["output_program"] == "trellis_output_n154880_f32"
    assert (6144, 576, 1) not in {item[1:] for item in shapes}


def test_adding_glm_preserves_every_kimi_compile_input_and_order(monkeypatch):
    module = load_exporter(monkeypatch)
    for name, methods in (
        ("b12x.integration.cuteafd.trellis_linear", ("compile_trellis_linear_aot",)),
        ("b12x.integration.cuteafd.trellis_transform", ("compile_trellis_input_aot", "compile_trellis_output_aot")),
    ):
        fake = ModuleType(name)
        for method in methods:
            setattr(fake, method, lambda **params: params)
        monkeypatch.setitem(sys.modules, name, fake)
    kimi = module["programs"](input_dtype="bf16")
    combined = module["programs"](families=("kimik3", "glm"), input_dtype="bf16")
    assert [p[:3] for p in combined[:len(kimi)]] == [p[:3] for p in kimi]
    assert len({p[0] for p in combined}) == len(combined)
    assert all(thunk() == params for _, _, params, thunk in combined)
    names = {item[0] for item in combined}
    for policy in module["row_policies"](("kimik3", "glm")):
        assert all(policy[field] in names for field in ("native_program", "native_input", "output_program"))
    assert not any(p[2].get("groups") for p in module["programs"](families=("glm",)))


def test_glm_static_table_matches_installed_published_checkpoint_headers(monkeypatch):
    module = load_exporter(monkeypatch)
    home = Path(os.environ.get("HF_HOME", "/mnt/sparknest/hf-home"))
    snapshot = (home / "hub/models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-Core-K6/snapshots"
                / "5a4e83a09a11f3145332ed632d28a9894ea88225")
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip("published GLM Core-K6 checkpoint is not installed")
    config = json.loads((snapshot / "config.json").read_text())
    h, heads = config["hidden_size"], config["num_attention_heads"]
    expected = {"lm_head": (config["vocab_size"], h, "whole")}
    for layer in range(config["num_hidden_layers"]):
        root = f"model.layers.{layer}"
        for leaf, n, k, split in (
            ("q_a_proj", config["q_lora_rank"], h, "whole"),
            ("kv_a_proj_with_mqa", config["kv_lora_rank"] + config["qk_rope_head_dim"], h, "whole"),
            ("q_b_proj", heads * (config["qk_nope_head_dim"] + config["qk_rope_head_dim"]), config["q_lora_rank"], "output"),
            ("o_proj", h, heads * config["v_head_dim"], "input"),
        ):
            expected[f"{root}.self_attn.{leaf}"] = (n, k, split)
        dense = layer < config["first_k_dense_replace"]
        inter = config["intermediate_size"] if dense else config["moe_intermediate_size"] * config["n_shared_experts"]
        mlp = "mlp" if dense else "mlp.shared_experts"
        for leaf, n, k, split in (("gate_proj", inter, h, "output"), ("up_proj", inter, h, "output"),
                                  ("down_proj", h, inter, "input")):
            expected[f"{root}.{mlp}.{leaf}"] = (n, k, split)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {f"{stem}.{suffix}" for stem in expected for suffix in ("trellis", "suh", "svh", "mcg")}
    assert wanted <= index.keys()
    headers = {}
    for shard in sorted({index[name] for name in wanted}):
        path = snapshot / shard
        if not path.is_file():
            pytest.skip(f"published GLM Core-K6 shard is not installed: {shard}")
        with path.open("rb") as stream:
            size, = struct.unpack("<Q", stream.read(8))
            assert 0 < size <= 64 * 1024 * 1024
            header = json.loads(stream.read(size))
        headers.update((name, header[name]) for name in wanted if index[name] == shard)
    geometries = {shape[1:] for shape in module["shapes_for"](("glm",))}
    for stem, (n, k, split) in expected.items():
        stored = ((n + 127) // 128) * 128 if stem.endswith(".kv_a_proj_with_mqa") else n
        for suffix, dtype, shape in (("trellis", "I16", [k // 16, stored // 16, 96]),
                                     ("suh", "F16", [k]), ("svh", "F16", [stored]), ("mcg", "I32", [])):
            tensor = headers[f"{stem}.{suffix}"]
            assert (tensor["dtype"], tensor["shape"]) == (dtype, shape), (stem, suffix, tensor)
        for world in ((1,) if split == "whole" else (1, 2)):
            local_k = k // world if split == "input" else k
            local_n = stored // world if split == "output" else stored
            side = 2 if split == "input" and world == 2 else 1
            assert (local_k, local_n, side) in geometries, (stem, local_k, local_n, side)
    assert len(expected) == 547


def test_muse_geometries_dtypes_and_existing_family_order(monkeypatch):
    module = load_exporter(monkeypatch)
    for name, methods in (
        ("b12x.integration.cuteafd.trellis_linear", ("compile_trellis_linear_aot",)),
        ("b12x.integration.cuteafd.trellis_transform", ("compile_trellis_input_aot", "compile_trellis_output_aot")),
    ):
        fake = ModuleType(name)
        for method in methods:
            setattr(fake, method, lambda **params: params)
        monkeypatch.setitem(sys.modules, name, fake)
    geometries = {shape[1:] for shape in module["shapes_for"](("muse",))}
    assert geometries == {(8192, 4096, 1), (8192, 8192, 1), (8192, 512, 1),
                          (8192, 1024, 1), (4096, 8192, 2), (4096, 8192, 1),
                          (8192, 2048, 1), (2048, 8192, 2), (8192, 202112, 1)}
    programs = module["programs"](families=("muse",), input_dtype="bf16")
    names = {item[0] for item in programs}
    assert {"trellis_output_n8192_bf16", "trellis_output_n8192_f32",
            "trellis_output_n202112_f32"} <= names
    policies = module["row_policies"](("muse",))
    assert len(policies) == 18
    for policy in policies:
        assert all(policy[field] in names for field in ("native_input", "native_program", "output_program"))
        expected = "f32" if policy["shape"] in ("muse_o_tp2", "muse_post_expert_tp2", "muse_lm_head") else "bf16"
        assert policy["output_program"].endswith("_" + expected)
    for family in ("kimik3", "glm", "glmf"):
        baseline = module["programs"](families=(family,), input_dtype="bf16")
        combined = module["programs"](families=(family, "muse"), input_dtype="bf16")
        assert [item[:3] for item in combined[:len(baseline)]] == [item[:3] for item in baseline]
        assert len({item[0] for item in combined}) == len(combined)


@pytest.mark.skipif(os.environ.get("CUTEAFD_MUSE_HEAD_GPU_TEST") != "1",
                    reason="opt-in SM120 Muse head byte-invariance gate")
def test_muse_f32_head_chunk_row_invariance():
    """Real Muse geometry: the complete target/drafter head, not a mock GEMM."""
    import torch
    from b12x.integration.cuteafd.muse import compile_softcap_aot
    from b12x.integration.cuteafd.trellis_linear import compile_trellis_linear_aot
    from b12x.integration.cuteafd.trellis_transform import (
        compile_trellis_input_aot, compile_trellis_output_aot,
    )

    assert torch.cuda.get_device_capability() == (12, 0), "Muse coordinator head needs SM120"
    k, stored, vocab, capacity = 8192, 202112, 202048, 64
    torch.manual_seed(923)
    x = (torch.randn((128, k), device="cuda") * .03).bfloat16()
    suh = (torch.randn(k, device="cuda") * .2).half()
    svh = (torch.randn(stored, device="cuda") * .2).half()
    payload = torch.randint(0, 256, (k * stored * 6 // 8,), dtype=torch.uint8, device="cuda")
    rotated = torch.empty((capacity, k), dtype=torch.bfloat16, device="cuda")
    pointers = torch.tensor([rotated.data_ptr()], dtype=torch.int64, device="cuda")
    scales = torch.empty(k // 128, dtype=torch.float16, device="cuda")
    partial = torch.empty((capacity, k // 1024, stored), device="cuda")
    acc = torch.empty((capacity, stored), device="cuda")
    staged = torch.empty_like(acc)
    trimmed = torch.empty((capacity, vocab), device="cuda")
    result = torch.empty((128, vocab), device="cuda")
    # Production packages have wide capacity; live head scratch remains 64 rows.
    inp = compile_trellis_input_aot(features=k, input_dtype="bf16", output_dtype="bf16", max_rows=4096)
    core = compile_trellis_linear_aot(input_features=k, output_features=stored, bits=6,
        codebook="mcg", sharded_side=1, rotation_count=1, input_dtype="bf16", max_rows=4096)
    out = compile_trellis_output_aot(features=stored, output_dtype="f32", max_rows=4096)
    softcap = compile_softcap_aot(vocab=vocab)
    ctas = torch.cuda.get_device_properties(0).multi_processor_count * 4

    def project(first, live):
        inp.launch(x[first:first + live], suh, rotated, scales, scalars=(live, ctas))
        core.launch(pointers, payload, scales, partial.flatten(), acc.flatten(), scalars=(live, 0, ctas))
        out.launch(acc, svh, staged, scalars=(live, ctas))

    # Retain pre-softcap F32 bytes too: softcap BF16 rounding must not hide drift.
    expected = torch.empty((128, stored), dtype=torch.float32, device="cpu")
    for first in (0, 64):
        project(first, capacity)
        expected[first:first + capacity].copy_(staged.cpu())
        trimmed.copy_(staged[:, :vocab])
        softcap.launch(trimmed, result[first:first + capacity], scalars=(capacity,))
    assert torch.isfinite(expected).all()
    expected_softcap = result.cpu()
    for chunk in (1, 16, 64):
        for first in range(0, 128, chunk):
            project(first, chunk)
            actual = staged[:chunk].cpu()
            assert torch.equal(actual.view(torch.int32), expected[first:first + chunk].view(torch.int32)), (
                f"Muse F32 head depends on chunk rows={chunk}, first={first}")
            trimmed[:chunk].copy_(staged[:chunk, :vocab])
            softcap.launch(trimmed, result[first:first + chunk], scalars=(chunk,))
        assert torch.equal(result.cpu().view(torch.int32), expected_softcap.view(torch.int32)), (
            f"Muse trimmed/softcapped head depends on chunk rows={chunk}")
        print(f"Muse head byte-invariant: chunk={chunk}, rows=128, second_chunk=64", flush=True)
