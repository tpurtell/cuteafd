"""Kimi native-source, causal-geometry and atomic layer-resume contracts."""
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from families.kimi_k3 import model, golden


@pytest.fixture(autouse=True)
def nvme_budget(monkeypatch):
    monkeypatch.setattr(golden.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=2 ** 40))


def test_source_identity_fails_closed(tmp_path, monkeypatch):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    source = snapshot / "modeling.py"
    source.write_text("# pinned fixture\n")
    monkeypatch.setattr(model, "REVISION", snapshot.name)
    monkeypatch.setattr(model, "SOURCE_PINS", {source.name: hashlib.sha256(source.read_bytes()).hexdigest()})
    assert model.source_identity(snapshot)["revision"] == snapshot.name
    source.write_text("# changed\n")
    with pytest.raises(ValueError, match="source differs"):
        model.source_identity(snapshot)


def test_native_shape_disagreement_stops_before_payload_read():
    weights = model.LayerWeights.__new__(model.LayerWeights)
    name = "language_model.model.layers.0.self_attn.dt_bias"
    weights.kda_num_heads = 96
    weights.index = {name: "fixture.safetensors"}
    weights.descriptor = lambda key: ("fixture.safetensors", {
        "dtype": "F32", "shape": [128], "data_offsets": [0, 512]})
    module = types.SimpleNamespace(named_parameters=lambda: [
        ("self_attn.dt_bias", types.SimpleNamespace(shape=(96,)))])
    with pytest.raises(ValueError, match=r"checkpoint \[128\], official model \[96\]"):
        weights.validate_module(module, "language_model.model.layers.0")


@pytest.mark.parametrize("bad_tail", [None, 1., -0., float("nan")])
def test_native_a_log_padding_is_bit_exact_and_audited(bad_tail):
    torch = pytest.importorskip("torch")
    weights = model.LayerWeights.__new__(model.LayerWeights)
    weights.kda_num_heads, weights.narrowed_a_log = 96, {}
    name = "language_model.model.layers.0.self_attn.A_log"
    value = torch.cat((torch.arange(96, dtype=torch.float32), torch.zeros(32)))
    if bad_tail is not None:
        value[-1] = bad_tail
        with pytest.raises(ValueError, match="not bit-exact zero"):
            weights.narrow_a_log(name, value, (96,))
        assert not weights.narrowed_a_log
    else:
        narrowed = weights.narrow_a_log(name, value, (96,))
        assert torch.equal(narrowed, value[:96])
        assert narrowed.dtype == torch.float32
        proof = weights.padding_provenance()["layers"][0]
        assert proof["tail_bitwise_zero"] is True
        assert proof["tail_sha256"] == hashlib.sha256(bytes(128)).hexdigest()


@pytest.mark.parametrize("heads,name,shape,expected", [
    (128, "language_model.model.layers.0.self_attn.A_log", (128,), (96,)),
    (96, "language_model.model.layers.0.self_attn.dt_bias", (128,), (96,)),
    (96, "language_model.model.layers.0.self_attn.A_log", (129,), (96,)),
    (96, "language_model.model.layers.0.self_attn.A_log", (128,), (95,)),
])
def test_a_log_padding_exception_is_narrow(heads, name, shape, expected):
    weights = model.LayerWeights.__new__(model.LayerWeights)
    weights.kda_num_heads = heads
    assert not weights.padded_a_log(name, shape, expected)


def test_module_loader_checks_and_narrows_only_zero_a_log():
    torch = pytest.importorskip("torch")
    weights = model.LayerWeights.__new__(model.LayerWeights)
    weights.kda_num_heads, weights.narrowed_a_log = 96, {}
    name = "language_model.model.layers.0.self_attn.A_log"
    weights.index = {name: "fixture.safetensors"}
    weights.descriptor = lambda key: ("fixture.safetensors", {
        "dtype": "F32", "shape": [128], "data_offsets": [0, 512]})
    value = torch.cat((torch.arange(96, dtype=torch.float32), torch.zeros(32)))
    weights.tensor = lambda key, device="cpu": value.clone()
    weights.weight = lambda key: value.clone()
    weights.close = lambda: None
    module = torch.nn.Module()
    module.self_attn = torch.nn.Module()
    module.self_attn.A_log = torch.nn.Parameter(torch.empty(96, device="meta"))
    weights.load_module(module, "language_model.model.layers.0")
    assert torch.equal(module.self_attn.A_log, value[:96])
    assert module.self_attn.A_log.dtype == torch.float32
    value[-1] = 1
    with pytest.raises(ValueError, match="not bit-exact zero"):
        weights.load_module(module, "language_model.model.layers.0")


def blob(root, pending, wid, data=b"\0\0\0\0"):
    path = root / ("layer%03d" % pending["layer"]) / (wid + ".bin")
    path.write_bytes(data)
    entry = {"path": str(path.relative_to(root)), "bytes": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}
    pending["windows"][wid] = {"hidden": entry, "blocks": []}
    return entry


def test_resume_advances_only_after_atomic_layer_commit(tmp_path):
    state = golden.LayerState(tmp_path, {"set": "fixture"})
    pending = state.begin(-1)
    blob(tmp_path, pending, "window")
    state.commit(pending)
    interrupted = state.begin(0)
    blob(tmp_path, interrupted, "window", b"partial")
    resumed = golden.LayerState(tmp_path, {"set": "fixture"})
    assert resumed.latest["layer"] == -1
    retried = resumed.begin(0)
    blob(tmp_path, retried, "window", b"\1\0\1\0")
    resumed.commit(retried)
    assert golden.LayerState(tmp_path, {"set": "fixture"}).latest["layer"] == 0
    assert not (tmp_path / "layer-01").exists()


def test_resume_rejects_changed_binding_or_blob(tmp_path):
    state = golden.LayerState(tmp_path, {"set": "fixture"})
    pending = state.begin(-1)
    entry = blob(tmp_path, pending, "window")
    state.commit(pending)
    with pytest.raises(ValueError, match="runtime differs"):
        golden.LayerState(tmp_path, {"set": "changed"})
    (tmp_path / entry["path"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="blob differs"):
        golden.LayerState(tmp_path, {"set": "fixture"})


def test_layer_write_reserves_nvme_headroom(tmp_path, monkeypatch):
    state = golden.LayerState(tmp_path, {"state_bytes_per_layer": 1024})
    monkeypatch.setattr(golden.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=2 ** 30))
    with pytest.raises(ValueError, match="headroom"):
        state.begin(0)


def test_resume_rejects_escaping_paths(tmp_path):
    state = golden.LayerState(tmp_path, {})
    with pytest.raises(ValueError, match="unsafe"):
        state.contained("../outside")


def test_block_residual_files_survive_old_layer_removal(tmp_path):
    torch = pytest.importorskip("torch")
    state = golden.LayerState(tmp_path, {})
    original = torch.arange(8).reshape(1, 2, 4).bfloat16()
    pending = state.begin(-1)
    state.save(pending, "window", original)
    state.commit(pending)
    pending = state.begin(0)
    state.save(pending, "window", original + 1, append_block=True)
    state.commit(pending)
    block = state.latest["windows"]["window"]["blocks"][0]
    assert golden.digest(tmp_path / block["path"]) == block["sha256"]
    assert not (tmp_path / "layer-01").exists()
    assert np.array_equal(np.fromfile(tmp_path / block["path"], dtype="<u2"),
                          original.view(torch.uint16).numpy().ravel())


def test_bounded_attention_preserves_native_math_and_prefix():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    def official(module, q, k, v, mask, scaling, dropout=0, **kwargs):
        score = torch.einsum("bhqd,bhkd->bhqk", q, k) * scaling + mask
        probability = score.softmax(-1)
        return torch.einsum("bhqk,bhkd->bhqd", probability, v).transpose(1, 2), probability
    reference = types.SimpleNamespace(eager_attention_forward=official)
    module = types.SimpleNamespace(training=False)
    bounded = model.causal_eager(reference)
    q, k, v = [torch.randn(1, 2, 640, 8, dtype=torch.float64) for _ in range(3)]
    short = bounded(module, q[:, :, :576], k[:, :, :576], v[:, :, :576], scaling=.25)[0]
    extended = bounded(module, q, k, v, scaling=.25)[0]
    assert torch.equal(short, extended[:, :576])
    mask = torch.zeros((640, 640), dtype=torch.float64).masked_fill(torch.ones(640, 640).bool().triu(1), -float("inf"))
    unchanged = official(module, q, k, v, mask[None, None], .25)[0]
    torch.testing.assert_close(extended, unchanged, atol=1e-14, rtol=1e-14)


def test_geometry_restored_on_failure():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    original = F.linear, torch.einsum
    eager = lambda *args, **kwargs: None
    reference = types.SimpleNamespace(eager_attention_forward=eager)
    with pytest.raises(RuntimeError):
        with model.geometry(reference):
            assert F.linear is not original[0]
            raise RuntimeError("interrupted")
    assert (F.linear, torch.einsum, reference.eager_attention_forward) == (*original, eager)


def test_native_mxfp4_packing_and_scale_values():
    torch = pytest.importorskip("torch")
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors.mxfp4 import MXFP4PackedCompressor
    from compressed_tensors.quantization import QuantizationScheme
    scheme = QuantizationScheme.model_validate({"targets": ["Linear"], "weights": {
        "num_bits": 4, "type": "float", "group_size": 32,
        "strategy": "group", "symmetric": True}})
    packed = torch.arange(256).to(torch.uint8).reshape(16, 16)
    scale = torch.arange(119, 135).to(torch.uint8)[:, None]
    native = MXFP4PackedCompressor.decompress(
        {"weight_packed": packed, "weight_scale": scale}, scheme)["weight"]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(16, 32).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    expected = (table[codes] * torch.exp2(scale.float() - 127)).bfloat16()
    assert native.dtype == torch.bfloat16
    assert torch.equal(native.view(torch.uint16), expected.view(torch.uint16))


def test_official_forward_gate_rejects_disagreement(tmp_path):
    native = np.array([[4., 0., -1.], [0., 4., -1.]], dtype=np.float32)
    assert golden.official_check(tmp_path, [native, native.copy()], [])["passed"]
    with pytest.raises(ValueError, match="numerical check failed"):
        golden.official_check(tmp_path, [native, native[:, ::-1].copy()], [])
    assert json.loads((tmp_path / "official-forward.json").read_text())["passed"] is False
