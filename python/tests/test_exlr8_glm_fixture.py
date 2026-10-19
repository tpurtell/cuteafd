"""Optional same-quant EXLR8 layer-3 fixtures, independent of native execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROWS = (1, 8, 64, 65, 128)
SEED = 408
EXLR8_REVISION = "a2b599e06c59f2ccb01aaa44e6f182ba21e39f80"
DENSE_REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
SOURCES = (
    "python/reference/exlr8.py",
    "python/reference/shape_invariant.py",
    "python/tests/test_exlr8_glm_fixture.py",
    "third_party/transformers/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py",
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def load_reference(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DecodedExperts:
    """Bounded CPU cache: official expert iteration needs only one expert at once."""

    def __init__(self, snapshot, manifest, oracle, torch):
        self.snapshot, self.manifest, self.oracle, self.torch = snapshot, manifest, oracle, torch
        self.current = None
        self.operands = None

    def get(self, expert, shared=False):
        identity = (int(expert), shared)
        if identity != self.current:
            selected = self.manifest["layers"]["3"]
            if shared:
                selected = selected["shared"]
            operands = {}
            for projection in ("gate", "up", "down"):
                widths = selected[projection] if shared else selected[projection][int(expert)]
                if projection == "down":
                    widths = [widths] * len(selected["gate"] if shared else selected["gate"][int(expert)])
                units = [self.oracle.expert_unit(self.snapshot, 3, int(k), int(expert), unit,
                                                projection, shared=shared)[0]
                         for unit, k in enumerate(widths)]
                import numpy as np
                # Oracle matrices are (input, output); official linear weights are transposed.
                axis = 0 if projection == "down" else 1
                operands[projection] = self.torch.from_numpy(np.concatenate(units, axis=axis).T.copy())
            self.current, self.operands = identity, operands
        return self.operands


def layer_widths(selected):
    widths = set(selected["down"])
    widths.add(selected["shared"]["down"])
    for projection in ("gate", "up"):
        widths.update(k for expert in selected[projection] for k in expert)
        widths.update(selected["shared"][projection])
    return widths


def test_decoded_experts_preserve_unit_axes_widths_and_shared_identity():
    import numpy as np
    torch = pytest.importorskip("torch")
    calls = []

    def unit(snapshot, layer, k, expert, unit, projection, shared=False):
        calls.append((k, expert, unit, projection, shared))
        shape = (128, 256) if projection == "down" else (256, 128)
        return np.full(shape, k + unit / 8, dtype=np.float32), None, "checksum"

    selected = {"layers": {"3": {
        **{p: [[2, 3]] for p in ("gate", "up")}, "down": [3],
        "shared": {"gate": [4, 4], "up": [4, 4], "down": 4}}}}
    assert layer_widths(selected["layers"]["3"]) == {2, 3, 4}
    bank = DecodedExperts(None, selected, SimpleNamespace(expert_unit=unit), torch)
    routed = bank.get(0)
    assert all(w.shape == (256, 256) and w.dtype == torch.float32 for w in routed.values())
    assert torch.all(routed["gate"][:128] == 2)
    assert torch.all(routed["gate"][128:] == 3.125)
    assert torch.all(routed["down"][:, :128] == 3)
    assert torch.all(routed["down"][:, 128:] == 3.125)
    assert bank.get(0) is routed and len(calls) == 6
    shared = bank.get(0, shared=True)
    assert len(calls) == 12 and all(c[-1] for c in calls[6:])
    assert torch.all(shared["gate"][:128] == 4)


def test_up1_width_inventory_matches_mounted_checkpoint_headers():
    configured = os.environ.get("CUTEAFD_EXLR8_TEST_SNAPSHOT")
    if not configured:
        pytest.skip("set CUTEAFD_EXLR8_TEST_SNAPSHOT for the real up1 header contract")
    snapshot = Path(configured)
    assert snapshot.name == EXLR8_REVISION
    selected = json.loads((snapshot / "manifests/exlr8-manifest-up1.json").read_text())
    oracle = load_reference("exlr8_fixture_header_oracle", "python/reference/exlr8.py")
    layer = selected["layers"]["3"]
    assert len(layer["down"]) == 256 and all(isinstance(k, int) for k in layer["down"])
    assert isinstance(layer["shared"]["down"], int)
    widths = layer_widths(layer)
    assert widths == {4}
    index, _ = oracle.header(snapshot / "experts/exlr8-layer-003.index.safetensors")
    for k in widths:
        for prefix, experts in (("", 256), ("shared.", 1)):
            assert index[f"{prefix}k{k}.index"]["shape"] == [experts, 3, 16, 2]
            assert index[f"{prefix}k{k}.checksum"]["shape"] == [experts, 3, 16]
        payload, _ = oracle.header(snapshot / f"experts/exlr8-layer-003.k{k}.safetensors")
        assert payload["__metadata__"]["k"] == str(k)
        assert payload["__metadata__"]["index_file"] == "exlr8-layer-003.index.safetensors"
        for projection in ("gate", "up", "down"):
            assert payload[f"trellis.{projection}"]["dtype"] == "U8"


def official_moe(ref, bank, config, hidden, routes, ids, routed_hidden=None):
    """Use vendor expert visitation, BF16 boundaries and index_add without reimplementing them."""
    import torch

    def weight(expert, projection):
        operands = bank.get(expert)
        value = (torch.cat((operands["gate"], operands["up"]), dim=0)
                 if projection == "gate_up" else operands[projection])
        return value.to(device=hidden.device, dtype=hidden.dtype)

    class Projection:
        def __init__(self, projection):
            self.projection = projection

        def __getitem__(self, expert):
            return weight(expert, self.projection)

    experts = SimpleNamespace(config=config, num_experts=config.num_local_experts,
                              gate_up_proj=Projection("gate_up"), down_proj=Projection("down"),
                              act_fn=torch.nn.functional.silu)
    shared = bank.get(0, shared=True)
    shared = {p: w.to(device=hidden.device, dtype=hidden.dtype) for p, w in shared.items()}
    mlp = SimpleNamespace(act_fn=torch.nn.functional.silu, **{
        p + "_proj": lambda x, p=p: torch.nn.functional.linear(x, shared[p])
        for p in ("gate", "up", "down")})
    moe = SimpleNamespace(
        gate=lambda x: (None, routes.to(hidden.dtype) if hidden.dtype == torch.float64 else routes, ids),
        experts=lambda x, i, w: ref.GlmMoeDsaExperts.forward(
            experts, x if routed_hidden is None else routed_hidden.to(x.dtype), i, w),
        shared_experts=lambda x: ref.GlmMoeDsaMLP.forward(mlp, x))
    return ref.GlmMoeDsaMoE.forward(moe, hidden)


@pytest.mark.parametrize("routed_override", [False, True])
def test_official_moe_binding_matches_pinned_module_on_cpu(monkeypatch, routed_override):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "third_party/transformers/src"))
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref

    config = ref.GlmMoeDsaConfig(hidden_size=4, moe_intermediate_size=4,
                                num_local_experts=2, n_shared_experts=1,
                                num_experts_per_tok=2, experts_implementation="eager")
    generator = torch.Generator().manual_seed(SEED)
    operands = {(e, shared): {p: torch.randn(4, 4, generator=generator) / 8
                             for p in ("gate", "up", "down")}
                for e, shared in ((0, False), (1, False), (0, True))}
    bank = SimpleNamespace(get=lambda e, shared=False: operands[int(e), shared])
    hidden = torch.randn(3, 4, generator=generator).bfloat16()
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
    routes = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    for dtype in (torch.bfloat16, torch.float64):
        module = ref.GlmMoeDsaMoE(config).to(dtype=dtype)
        with torch.no_grad():
            for expert in range(2):
                weights = bank.get(expert)
                module.experts.gate_up_proj[expert].copy_(torch.cat((weights["gate"], weights["up"])))
                module.experts.down_proj[expert].copy_(weights["down"])
            for projection, weight in bank.get(0, shared=True).items():
                getattr(module.shared_experts, projection + "_proj").weight.copy_(weight)
            routing = routes.double() if dtype == torch.float64 else routes
            monkeypatch.setattr(module.gate, "forward", lambda x: (None, routing, ids))
            original = hidden.to(dtype)
            routed = (hidden.float() + 0.25).bfloat16().float() if routed_override else None
            if routed_override:
                # The real MoE still sends original activations to gate/shared.
                forward = module.experts.forward
                monkeypatch.setattr(module.experts, "forward", lambda x, i, w: forward(routed.to(dtype), i, w))
            expected = module(original)
            vendor_forward = ref.GlmMoeDsaMoE.forward
            seen = []

            def check_inputs(moe, x):
                for name in ("gate", "shared_experts"):
                    operation = getattr(moe, name)

                    def checked(value, name=name, operation=operation):
                        torch.testing.assert_close(value, original, rtol=0, atol=0)
                        seen.append(name)
                        return operation(value)

                    setattr(moe, name, checked)
                return vendor_forward(moe, x)

            with monkeypatch.context() as patch:
                patch.setattr(ref.GlmMoeDsaMoE, "forward", check_inputs)
                actual = official_moe(ref, bank, config, original, routes, ids, routed)
            assert seen == ["gate", "shared_experts"]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if routed_override:
                all_changed = module(routed.to(dtype))
                assert not torch.equal(expected, all_changed), "shared input must stay unquantized"


@pytest.mark.parametrize("case", ["random", "zeros", "tiny", "maximum", "subnormal", "ties"])
def test_fp8_wire_numpy_matches_serving_cpu_intrinsics(case):
    import numpy as np
    torch = pytest.importorskip("torch")
    from b12x._lib.intrinsics import pow2_ceil_ue8m0_torch, _ue8m0_output_scale_torch

    hidden = torch.randn((8, 128), generator=torch.Generator().manual_seed(SEED)).bfloat16()
    if case == "zeros":
        hidden.zero_()
        hidden[:, 1::2] = -0.0
    elif case == "tiny":
        hidden.mul_(1e-8)
    elif case == "maximum":
        hidden.fill_(torch.finfo(torch.bfloat16).max)
        hidden[:, 1::2].neg_()
    elif case == "subnormal":
        hidden = torch.arange(32, dtype=torch.int32).view(torch.bfloat16).reshape(2, 32).repeat(4, 4)
    elif case == "ties":
        hidden = torch.tensor([1.0625, 1.1875, 2**-10, 3*2**-10, -0.0, 448, -448, 1.0],
                              dtype=torch.bfloat16).repeat(8, 16)
    blocks = hidden.float().reshape(8, -1, 32)
    _, scales = pow2_ceil_ue8m0_torch(blocks.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0))
    inverse = _ue8m0_output_scale_torch(scales)
    payload = (blocks * inverse[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    expected = torch.cat((payload.reshape(8, 128), scales), dim=1).numpy()
    oracle = load_reference("exlr8_cpu_wire_oracle", "python/reference/exlr8.py")
    actual = oracle.fp8_k32_wire(hidden.float().numpy())
    assert actual.tobytes() == expected.tobytes()
    decoded = payload.view(torch.float8_e4m3fn).float() * torch.exp2(scales.float() - 127)[..., None]
    if case == "maximum":
        # 256 * 2**120 = 2**128: serving FP32 reconstruction also overflows.
        assert (actual[:, 128:] == 247).all()
        assert (actual[:, :128:2] == 0x78).all()
        assert (actual[:, 1:128:2] == 0xF8).all()
        with pytest.warns(RuntimeWarning, match="overflow encountered in multiply"):
            reconstructed = oracle.fp8_k32_decode(actual)
        assert np.isinf(reconstructed).all()
    else:
        reconstructed = oracle.fp8_k32_decode(actual)
        assert np.isfinite(reconstructed).all()
    np.testing.assert_array_equal(reconstructed, decoded.reshape(8, 128).numpy())
    if case in ("zeros", "tiny", "subnormal"):
        assert (actual[:, 128:] == 105).all(), "amax floor precludes subnormal wire scales"


def save_fixture(root, manifest, name, tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    data = value.view(torch.uint8).numpy().tobytes()
    (root / (name + ".bin")).write_bytes(data)
    manifest["files"][name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                               "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def generate_outputs(ref, bank, config, router, norm, root, manifest, device):
    import torch
    with torch.inference_mode():
        residual = torch.randn((max(ROWS), config.hidden_size),
                               generator=torch.Generator().manual_seed(SEED)).bfloat16().to(device)
        hidden = norm(residual).bfloat16()
        _, routes, ids = router(hidden)
        oracle = load_reference("exlr8_fixture_wire_oracle", "python/reference/exlr8.py")
        wire = oracle.fp8_k32_wire(hidden.float().cpu().numpy())
        routed = torch.from_numpy(oracle.fp8_k32_decode(wire)).to(device)
        official = (residual + official_moe(ref, bank, config, hidden, routes, ids, routed)).float()
        fp64 = residual.double() + official_moe(ref, bank, config, hidden.double(), routes, ids, routed)
        assert torch.isfinite(official).all() and torch.isfinite(fp64).all()
        for rows in ROWS:
            for name, value in {"residual": residual, "hidden": hidden, "wire": torch.from_numpy(wire),
                                "ids": ids.int(), "routes": routes,
                                "official_output": official, "fp64_output": fp64}.items():
                save_fixture(root, manifest, f"r{rows}_{name}", value[:rows])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def test_generator_cpu_end_to_end_with_real_layer_operands(monkeypatch, tmp_path):
    configured = os.environ.get("CUTEAFD_EXLR8_TEST_SNAPSHOT")
    dense = os.environ.get("CUTEAFD_EXLR8_DENSE_SNAPSHOT")
    if not configured or not dense:
        pytest.skip("set real EXLR8 and dense snapshots for the CPU generator qualification")
    import copy
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "third_party/transformers/src"))
    monkeypatch.syspath_prepend(str(ROOT / "python/reference"))
    from transformers import AutoConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref
    from shape_invariant import install

    monkeypatch.setattr(torch.nn.functional, "linear", torch.nn.functional.linear)
    monkeypatch.setattr(torch, "einsum", torch.einsum)
    install()
    config = copy.deepcopy(AutoConfig.from_pretrained(dense))
    config._experts_implementation = "eager"
    config.num_local_experts, config.num_experts_per_tok = 2, 2
    config.n_group, config.topk_group, config.moe_intermediate_size = 1, 1, 128
    selected = json.loads((Path(configured) / "manifests/exlr8-manifest-up1.json").read_text())
    layer = selected["layers"]["3"]
    for projection in ("gate", "up"):
        layer[projection] = [choices[:1] for choices in layer[projection][:2]]
        layer["shared"][projection] = layer["shared"][projection][:1]
    layer["down"] = layer["down"][:2]
    oracle = load_reference("exlr8_cpu_generator_oracle", "python/reference/exlr8.py")
    golden = load_reference("exlr8_cpu_generator_glm", "python/reference/families/glm5/golden.py")
    reader = golden.Weights(Path(dense))
    router = ref.GlmMoeDsaTopkRouter(config)
    norm = ref.GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps)
    with torch.no_grad():
        router.weight.copy_(reader.get("model.layers.3.mlp.gate.weight", device="cpu")[:2])
        router.e_score_correction_bias.copy_(reader.get(
            "model.layers.3.mlp.gate.e_score_correction_bias", device="cpu")[:2])
        norm.weight.data = reader.get("model.layers.3.post_attention_layernorm.weight", device="cpu")
    bank = DecodedExperts(Path(configured), selected, oracle, torch)
    generate_outputs(ref, bank, config, router, norm, tmp_path, {"files": {}}, "cpu")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["files"]) == len(ROWS) * 7
    for name, entry in manifest["files"].items():
        data = (tmp_path / (name + ".bin")).read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
        rows = int(name.split("_")[0][1:])
        width = (2 if name.endswith(("_ids", "_routes")) else
                 config.hidden_size + config.hidden_size // 32 if name.endswith("_wire") else config.hidden_size)
        assert entry["shape"] == [rows, width]


def test_layer3_same_quant_fixtures(monkeypatch):
    destination = os.environ.get("CUTEAFD_EXLR8_GLM_FIXTURE_DIR")
    if not destination:
        pytest.skip("set CUTEAFD_EXLR8_GLM_FIXTURE_DIR for real-weight CUDA fixtures")
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(ROOT / "third_party/transformers/src"))
    from transformers import AutoConfig
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as ref
    from shape_invariant import install

    snapshot = Path(os.environ["CUTEAFD_GOLDEN_SNAPSHOT"])
    dense = Path(os.environ["CUTEAFD_EXLR8_DENSE_SNAPSHOT"])
    assert snapshot.name == EXLR8_REVISION and dense.name == DENSE_REVISION
    root = Path(destination).resolve()
    assert root == Path(os.environ["CUTEAFD_GOLDEN_FIXTURES"]).resolve()
    assert root.is_dir()
    oracle = load_reference("exlr8_fixture_oracle", "python/reference/exlr8.py")
    golden = load_reference("exlr8_fixture_glm", "python/reference/families/glm5/golden.py")
    selected_path = snapshot / "manifests/exlr8-manifest-up1.json"
    selected = json.loads(selected_path.read_text())
    assert selected["name"] == "up1" and selected["format"] == "exlr8-v2-manifest"
    assert sha256(Path(ref.__file__)) == sha256(ROOT / SOURCES[-1]), "official control-flow source differs"
    config = AutoConfig.from_pretrained(dense)
    config._experts_implementation = "eager"
    assert config.hidden_size == 6144 and config.moe_intermediate_size == 2048
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.nn.functional, "linear", torch.nn.functional.linear)
    monkeypatch.setattr(torch, "einsum", torch.einsum)
    install()
    manifest = {
        "schema": 2, "seed": SEED, "layer": 3, "rows": list(ROWS), "manifest": "up1",
        "activation_wire": "fp8_e4m3_ue8m0_k32_amax_floor_1e-4_routed_only",
        "checkpoint": "kindlingai/glm-5.3-exlr8-k2-k3-k4", "revision": EXLR8_REVISION,
        "dense_checkpoint": "zai-org/GLM-5.3", "dense_revision": DENSE_REVISION,
        "set_id": selected["set_id"], "error_ratio_bound": 1.01,
        "source_sha256": {p: sha256(ROOT / p) for p in SOURCES},
        "checkpoint_files_sha256": {}, "dense_tensors_sha256": {}, "files": {},
        "official_arithmetic": "official GLM MoE/Experts/MLP forward; BF16 decoded operands; routed FP8 K32 roundtrip, BF16 shared input and router; fixed M128",
        "fp64_arithmetic": "same oracle-decoded F32 physical operands; identical routed FP8 K32 roundtrip and BF16 shared input; FP64 projections, silu, routing and residual sum",
        "torch": torch.__version__, "cuda": torch.version.cuda, "allow_tf32": False,
    }
    widths = layer_widths(selected["layers"]["3"])
    paths = ["config.json", "manifests/exlr8-manifest-up1.json", "experts/exlr8-layer-003.index.safetensors"]
    paths += [f"experts/exlr8-layer-003.k{k}.safetensors" for k in sorted(widths)]
    manifest["checkpoint_files_sha256"] = {p: sha256(snapshot / p) for p in paths}

    reader = golden.Weights(dense)

    def dense_operand(name):
        # Hash the actual raw source operands, including FP8 block scales if present.
        for source in (name, name.removesuffix("weight") + "weight_scale_inv"):
            if source in reader.index:
                value = reader.raw(source)
                data = value.contiguous().view(torch.uint8).numpy().tobytes()
                manifest["dense_tensors_sha256"][source] = hashlib.sha256(data).hexdigest()
        return reader.get(name)

    with torch.inference_mode():
        router = ref.GlmMoeDsaTopkRouter(config).cuda()
        router.weight.data = dense_operand("model.layers.3.mlp.gate.weight")
        router.e_score_correction_bias.copy_(dense_operand("model.layers.3.mlp.gate.e_score_correction_bias"))
        norm = ref.GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps).cuda()
        norm.weight.data = dense_operand("model.layers.3.post_attention_layernorm.weight")
        bank = DecodedExperts(snapshot, selected, oracle, torch)
        generate_outputs(ref, bank, config, router, norm, root, manifest, "cuda")
