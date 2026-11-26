"""GLM 5.3 / GLM Flash legacy goldens keyed to an exact u32 ``tokens.bin`` (#2719)."""
import argparse
import ast
import json
import pathlib
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import read_tokens_file, write_tokens_file  # noqa: E402

GOLDENS = {"glm5": ROOT / "python/reference/families/glm5/golden.py",
           "glm5_flash": ROOT / "python/reference/families/glm5_flash/golden.py"}
# Window 0's first ids of the published GLM set, plus the last vocabulary id.
TOKENS = [2, 57149, 8275, 35, 3119, 271, 154879]


def test_tokens_file_round_trips_u32_little_endian(tmp_path):
    path = tmp_path / "tokens.bin"
    write_tokens_file(path, TOKENS)
    assert path.read_bytes() == b"".join(t.to_bytes(4, "little") for t in TOKENS)
    assert read_tokens_file(path, 154880) == TOKENS
    # The same decode trace.py --tokens-file applies (mistral_large4).
    data = path.read_bytes()
    assert [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)] == TOKENS


@pytest.mark.parametrize("payload,match", [(b"", "nonempty"), (b"\x01\x00\x00", "u32"),
                                           ((154880).to_bytes(4, "little"), "vocabulary")])
def test_tokens_file_refuses_empty_ragged_and_out_of_vocab(tmp_path, payload, match):
    path = tmp_path / "tokens.bin"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=match):
        read_tokens_file(path, 154880)


def function(family, name):
    tree = ast.parse(GOLDENS[family].read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize("family", sorted(GOLDENS))
@pytest.mark.parametrize("argv,ok", [
    (["--tokens-file", "t.bin"], True),
    (["--text", "hi"], True),
    ([], False),
    (["--tokens-file", "t.bin", "--text", "hi"], False),
    (["--tokens-file", "t.bin", "--text-file", "p.txt"], False),
    (["--tokens-file", "t.bin", "--windows", "w.json"], False),
])
def test_exactly_one_prompt_source(monkeypatch, family, argv, ok):
    scope = {"argparse": argparse, "Path": pathlib.Path, "__doc__": ""}
    exec(compile(ast.Module(body=[function(family, "main")], type_ignores=[]), "main", "exec"), scope)
    # Past validation, main imports the reference stack; a stub without it stops there.
    monkeypatch.setitem(sys.modules, "tokenizers", types.ModuleType("tokenizers"))
    monkeypatch.setattr(sys, "argv", ["golden.py", "--snapshot", "s", "--out", "o", *argv])
    with pytest.raises(ImportError if ok else SystemExit):
        scope["main"]()


class Tensor:
    def __init__(self, data): self.data = np.asarray(data)
    shape = property(lambda self: self.data.shape)
    dtype = property(lambda self: "bf16")
    def __getitem__(self, key): return Tensor(self.data[key.data if isinstance(key, Tensor) else key])
    def __eq__(self, other): return Tensor(self.data == other.data)
    def cpu(self): return Tensor(self.data.copy())
    def cuda(self): return self.cpu()
    def contiguous(self): return self.cpu()
    def float(self): return Tensor(self.data.astype(np.float32))
    def double(self): return Tensor(self.data.astype(np.float64))
    def to(self, *_args): return self
    def view(self, _dtype): return self
    def numpy(self): return self.data
    def eval(self): return self
    def triu(self, diagonal): return Tensor(np.triu(self.data, diagonal))
    def argmax(self, dim): return Tensor(self.data.argmax(axis=dim))
    def mean(self, dim=None): return Tensor(self.data.mean(axis=dim))
    def item(self): return self.data.item()
    def __int__(self): return int(self.data)
    def gather(self, dim, index): return Tensor(np.take_along_axis(self.data, index.data, axis=dim))
    def unsqueeze(self, axis): return Tensor(np.expand_dims(self.data, axis))
    def expand(self, *shape):
        return Tensor(np.broadcast_to(self.data, tuple(o if n == -1 else n for o, n in zip(self.data.shape, shape))))
    def copy_(self, other): self.data = other.data.copy()


def fake_torch():
    def log_softmax(x, dim):
        shifted = x.data - x.data.max(axis=dim, keepdims=True)
        return Tensor(shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True)))
    return SimpleNamespace(
        inference_mode=nullcontext, device=lambda _name: nullcontext(), set_default_dtype=lambda _d: None,
        bfloat16="bf16", float32="f32", int16="i16", bool="bool",
        tensor=lambda data, **_kw: Tensor(data), arange=lambda n, **_kw: Tensor(np.arange(n)),
        full=lambda shape, value, **_kw: Tensor(np.full(shape, value)),
        ones=lambda *shape, **_kw: Tensor(np.ones(shape, dtype=bool)), log_softmax=log_softmax, cuda=None,
        nn=SimpleNamespace(functional=SimpleNamespace(
            embedding=lambda ids, _w: Tensor(ids.data[..., None].astype(np.float32)),
            linear=lambda x, w: Tensor(x.data @ w.data.T))))


class Layer:
    calls = []
    def __init__(self, _config, layer_id): self.layer_id = layer_id
    def to_empty(self, **_kw): return self
    def eval(self): return self
    def named_parameters(self): return []
    def named_buffers(self): return []
    def __call__(self, h, *, attention_mask, prev_topk_indices, **_kw):
        Layer.calls.append((self.layer_id, None if prev_topk_indices is None else prev_topk_indices.data.item(),
                            attention_mask.data.copy()))
        return Tensor(h.data + 1), Tensor(np.array(10 + self.layer_id))


class Norm:
    def __init__(self, *_args): self.weight = Tensor([1.0])
    def cuda(self): return self
    def to(self, _dtype): return self
    def __call__(self, h): return h


def run(family, tmp_path, tokens_file, *, stop_after=None):
    Layer.calls = []
    vocab = 154880
    head = np.zeros((vocab, 1), dtype=np.float32)
    head[:, 0] = np.linspace(-1, 1, vocab)
    weights = SimpleNamespace(read_bytes=0, staged={}, stage_layer=lambda _prefix: None,
                              get=lambda name: Tensor(head if "lm_head" in name else [1.0]))
    storage = lambda *_args: SimpleNamespace(release=lambda: None, check=lambda _label: None)
    extents = []
    scope = dict(torch=fake_torch(), time=__import__("time"), json=json, CheckpointStorage=storage,
                 load_layer=lambda *_args: None, log_checkpoint_reads=lambda *_args: None,
                 install_dsa=lambda _ref, extent: extents.append(extent), write_tokens_file=write_tokens_file,
                 PREFIX="model.language_model.", FP32_KEYS=())
    exec(compile(ast.Module(body=[function(family, "run_tokens")], type_ignores=[]), "run_tokens", "exec"), scope)
    config = SimpleNamespace(num_hidden_layers=3, hidden_size=1, rms_norm_eps=1e-6, hc_mult=4,
                             layer_types=["dsa", "kda", "dsa"])
    ref = SimpleNamespace(GlmMoeDsaDecoderLayer=Layer, Glm5NextTextDecoderLayer=Layer,
                          GlmMoeDsaRMSNorm=Norm, Glm5NextTextRMSNorm=Norm,
                          GlmMoeDsaRotaryEmbedding=lambda **_kw: SimpleNamespace(cuda=lambda: lambda h, **_k: (h, h)))
    out = tmp_path / "out"
    out.mkdir()
    args = SimpleNamespace(out=out, snapshot=tmp_path, experts_snapshot=None, layers=None, stop_after=stop_after,
                           tokens_file=tokens_file, requant_ue8m0=False)
    tokens = read_tokens_file(tokens_file, vocab)
    if family == "glm5":
        scope["run_tokens"](args, config, ref, weights, tokens)
    else:
        scope["run_tokens"](args, config, ref, weights, weights, tokens)
    return out, extents, vocab


@pytest.mark.parametrize("family", sorted(GOLDENS))
def test_reference_is_keyed_to_the_exact_token_file(tmp_path, family):
    source = tmp_path / "fixture-tokens.bin"
    write_tokens_file(source, TOKENS)
    out, extents, vocab = run(family, tmp_path, source)
    # Byte-identical ids, never a re-tokenized prompt.
    assert (out / "tokens.bin").read_bytes() == source.read_bytes()
    logits = np.fromfile(out / "logits.bin", dtype="<f4").reshape(len(TOKENS), vocab)
    assert np.isfinite(logits).all() and (logits.max(axis=1) > logits.min(axis=1)).all()
    np.testing.assert_allclose(logits[:, -1], np.array(TOKENS, dtype=np.float32) + 3)
    assert sorted(p.name for p in out.glob("layer*.bin")) == ["layer00.bin", "layer01.bin", "layer02.bin"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tokens"] == len(TOKENS) and meta["tokens_file"] == str(source) and meta["family"] == family
    assert np.isfinite(meta["mean_nll"])
    # The official model loop threads each layer's DSA top-k into the next.
    assert [(layer, incoming) for layer, incoming, _ in Layer.calls] == [(0, None), (1, 10), (2, 11)]
    if family == "glm5":
        # One causal window through the same DSA adaptation as --windows.
        assert extents == [len(TOKENS)]
        mask = Layer.calls[0][2][0, 0]
        assert mask.shape == (len(TOKENS), len(TOKENS))
        assert np.isneginf(mask[np.triu_indices(len(TOKENS), 1)]).all()
        assert (mask[np.tril_indices(len(TOKENS))] == 0).all()


def test_glm_flash_stop_after_writes_no_logits(tmp_path):
    source = tmp_path / "fixture-tokens.bin"
    write_tokens_file(source, TOKENS)
    out, _, _ = run("glm5_flash", tmp_path, source, stop_after=0)
    assert (out / "tokens.bin").read_bytes() == source.read_bytes()
    assert not (out / "logits.bin").exists() and [p.name for p in out.glob("layer*.bin")] == ["layer00.bin"]
