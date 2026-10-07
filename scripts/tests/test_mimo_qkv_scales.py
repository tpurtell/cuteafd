"""CPU regressions for Hugh Madden's fused-QKV scale finding (issue #3)."""
import ast
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "python/reference/families/mimo_v2/mimo_v26/golden.py"


class Tensor(np.ndarray):
    """Only the CPU tensor operations used by the checkpoint dequantizer."""

    def float(self):
        return self.astype(np.float32).view(Tensor)

    def bfloat16(self):
        bits = self.float().view(np.uint32)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        return rounded.view(np.float32).view(Tensor)

    def repeat_interleave(self, repeats, dim):
        return np.repeat(self, repeats, axis=dim).view(Tensor)


def reference_weights(value, scale, q, k, v, tp):
    # Execute the actual reader without importing torch or CUDA on the host.
    tree = ast.parse(REFERENCE.read_text())
    selected = [node for node in tree.body if
                isinstance(node, ast.ClassDef) and node.name == "Weights" or
                isinstance(node, ast.FunctionDef) and node.name == "fp8_blocks_to_bf16"]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    torch = SimpleNamespace(uint8=np.dtype("uint8"), float8_e4m3fn=np.dtype("float32"),
                            cat=lambda parts, dim: np.concatenate(parts, axis=dim).view(Tensor))
    namespace = {"torch": torch}
    exec(compile(module, str(REFERENCE), "exec"), namespace)
    weights = namespace["Weights"].__new__(namespace["Weights"])
    weights.ckpt_tp = tp
    weights.qkv_shards = lambda layer: (q, k, v)
    weights.raw = lambda name: (scale if name.endswith("_scale_inv") else value).view(Tensor)
    return weights.get("model.layers.0.self_attn.qkv_proj.weight", 0)


def naive_segment_rows(q, k, v, tp):
    blocks = [(size + 127) // 128 for size in (q, k, v)]
    return np.concatenate([np.concatenate([
        s * sum(blocks) + sum(blocks[:i]) + np.arange(size) // 128
        for i, size in enumerate((q, k, v))]) for s in range(tp)])


def shard_rows(q, k, v, tp):
    rows = q + k + v
    return np.concatenate([s * ((rows + 127) // 128) + np.arange(rows) // 128 for s in range(tp)])


def deinterleave(value, q, k, v, tp):
    shards = value.reshape(tp, q + k + v, -1)
    return np.concatenate([shards[:, first:first + size].reshape(tp * size, -1)
                           for first, size in ((0, q), (q, k), (q + k, v))])


@pytest.mark.parametrize("q,k,v,tp", [(3072, 192, 128, 4), (3072, 384, 256, 4), (3072, 192, 128, 8)])
def test_reference_uses_whole_shard_grid(q, k, v, tp):
    rows = tp * (q + k + v)
    value = np.ones((rows, 128), dtype=np.float32)
    scales = np.arange(1, tp * ((q + k + v + 127) // 128) + 1, dtype=np.float32)[:, None]
    expected = deinterleave(value * scales[shard_rows(q, k, v, tp)], q, k, v, tp)
    np.testing.assert_array_equal(reference_weights(value, scales, q, k, v, tp), expected)
    naive = deinterleave(value * scales[naive_segment_rows(q, k, v, tp)], q, k, v, tp)
    changed = np.flatnonzero(np.any(naive != expected, axis=1))
    affected = (np.concatenate([tp * (q + k) + s * v + np.arange(64) for s in range(tp)])
                if k == 192 else np.array([], dtype=np.int64))
    np.testing.assert_array_equal(changed, affected)


def test_t2_poisoned_pad_detects_per_segment_rule():
    q = k = v = 64
    tp = 2
    value = np.ones((tp * (q + k + v), 128), dtype=np.float32)
    clean = np.array([1, 2, 3, 4, 0, 0], dtype=np.float32)[:, None]
    poisoned = clean.copy()
    poisoned[4:] = 1e30
    expected = reference_weights(value, clean[:4], q, k, v, tp)
    np.testing.assert_array_equal(reference_weights(value, poisoned[:4], q, k, v, tp), expected)
    naive = deinterleave(value * poisoned[naive_segment_rows(q, k, v, tp)], q, k, v, tp)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(naive, expected)
    with pytest.raises(AssertionError):
        reference_weights(value, poisoned, q, k, v, tp)


def raw_tensor(snapshot, index, name):
    with (snapshot / index[name]).open("rb") as file:
        length = struct.unpack("<Q", file.read(8))[0]
        meta = json.loads(file.read(length))[name]
        first, last = meta["data_offsets"]
        file.seek(8 + length + first)
        return meta, file.read(last - first)


def saturated_blocks(codes, row_groups):
    rows, cols = codes.shape
    assert cols % 128 == 0
    saturated = ((codes & 0x7F) == 0x7E).reshape(rows, cols // 128, 128).any(axis=2)
    return all(saturated[first:last].any(axis=0).all() for first, last in row_groups)


@pytest.mark.parametrize("model,revision,tp", [
    ("MiMo-V2.6-Flash-MOPD", "2479e2d0029eca9a34cc7e7f55a121925f81908e", 4),
    ("MiMo-V2.6-Pro-MOPD", "adea8e2c5373181e5a973fa1ecb343cb31af214b", 8),
])
def test_checkpoint_qkv_has_amax_in_every_shard_block(model, revision, tp):
    hub = Path(os.environ.get("CUTEAFD_CHECKPOINT_TEST_HUB", "/mnt/sparknest/hf-home/hub"))
    snapshot = hub / f"models--XiaomiMiMo--{model}" / "snapshots" / revision
    if not (snapshot / "model.safetensors.index.json").is_file():
        pytest.skip(f"checkpoint absent: {snapshot}")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    assert index["metadata"]["tp_size"] == tp
    config = json.loads((snapshot / "config.json").read_text())
    config = config.get("text_config", config)
    impossible_segments = []
    for layer, swa in enumerate(config["hybrid_layer_pattern"]):
        prefix = "swa_" if swa else ""
        q = config.get(prefix + "num_attention_heads", config["num_attention_heads"]) // tp * config.get(prefix + "head_dim", config["head_dim"])
        k = config.get(prefix + "num_key_value_heads", config["num_key_value_heads"]) // tp * config.get(prefix + "head_dim", config["head_dim"])
        v = config.get(prefix + "num_key_value_heads", config["num_key_value_heads"]) // tp * config.get(prefix + "v_head_dim", config["v_head_dim"])
        rows = q + k + v
        name = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        meta, raw = raw_tensor(snapshot, index["weight_map"], name)
        codes = np.frombuffer(raw, np.uint8).reshape(meta["shape"])
        assert meta["dtype"] == "F8_E4M3" and codes.shape[0] == tp * rows
        scale_meta, _ = raw_tensor(snapshot, index["weight_map"], name + "_scale_inv")
        assert scale_meta["shape"] == [tp * ((rows + 127) // 128), codes.shape[1] // 128]
        whole = [(s * rows + r, s * rows + min(r + 128, rows)) for s in range(tp) for r in range(0, rows, 128)]
        assert saturated_blocks(codes, whole), f"{model} layer {layer}: inconsistent per-shard grid"
        parts = [(s * rows + first + r, s * rows + first + min(r + 128, size))
                 for s in range(tp) for first, size in ((0, q), (q, k), (q + k, v)) for r in range(0, size, 128)]
        if not saturated_blocks(codes, parts):
            impossible_segments.append(layer)
    assert impossible_segments, "checkpoint no longer distinguishes the incorrect per-segment rule"
