"""CPU contracts for Mistral Large 4's Pixtral tower qualification; no CUDA imports."""
import ctypes
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "rust/crates/cuteafd-loader/src/families/mistral_large4/official-config.json"
PROCESSOR = ROOT / "rust/crates/cuteafd-loader/src/families/mistral_large4/official-processor-config.json"


def load_gate():
    path = ROOT / "python/tools/qualify/mistral_large4/qualify-vision.py"
    spec = importlib.util.spec_from_file_location("mistral_vision_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShapeOnly:
    """A BF16 tensor of a shape whose bytes are never materialized."""
    def __init__(self, shape):
        self.shape, self.dtype = tuple(shape), np.dtype(np.uint16)
        self.nbytes = int(np.prod(shape)) * 2

    def tobytes(self):
        return b""


def packed(monkeypatch, tmp_path):
    gate = load_gate()
    (tmp_path / "params.json").write_text(json.dumps({"vision_encoder": {
        "image_token_id": 10, "image_break_token_id": 12, "image_end_token_id": 13}}))
    shapes = {"vision_encoder.patch_conv.weight": [2048, 3, 14, 14], "vision_encoder.ln_pre.weight": [2048]}
    for i in range(30):
        p = f"vision_encoder.transformer.layers.{i}."
        for name, shape in (("attention.wq", [2048, 2048]), ("attention.wk", [2048, 2048]),
                            ("attention.wv", [2048, 2048]), ("attention.wo", [2048, 2048]),
                            ("attention_norm", [2048]), ("ffn_norm", [2048]), ("feed_forward.w1", [6144, 2048]),
                            ("feed_forward.w3", [6144, 2048]), ("feed_forward.w2", [2048, 6144])):
            shapes[p + name + ".weight"] = shape
    shapes.update({"pre_mm_projector_norm.weight": [2048], "patch_merger.merging_layer.weight": [2048, 8192],
                   "vision_language_adapter.w_in.weight": [6144, 2048],
                   "vision_language_adapter.w_out.weight": [6144, 6144]})
    tensors = {name: ShapeOnly(shape) for name, shape in shapes.items()}
    # Fused operands are concatenated by the packer; keep that shape-only too.
    concatenate = np.concatenate
    monkeypatch.setattr(gate.np, "concatenate", lambda parts: ShapeOnly(
        [sum(p.shape[0] for p in parts), *parts[0].shape[1:]]) if isinstance(parts[0], ShapeOnly)
        else concatenate(parts))
    real_rows = [np.zeros(6144, np.uint16), np.zeros(6144, np.uint16)]
    monkeypatch.setattr(gate, "marker_rows", lambda _snapshot, ids: real_rows)
    return gate, gate.pack_spec(json.loads(CONFIG.read_text()), tensors, tmp_path)


def test_mistral_pointer_count_abi_is_the_shared_base():
    gate = load_gate()
    assert gate.Spec is gate.common.Spec and gate.Ledger is gate.common.Ledger
    assert ctypes.sizeof(gate.Spec) == 256


def test_mistral_packer_matches_the_rust_tower_arena(monkeypatch, tmp_path):
    gate, (spec, _blob, _rows) = packed(monkeypatch, tmp_path)
    # Rust TowerSpec::mistral_large4 + resident_weight_bytes (cuteafd-loader vision.rs).
    assert spec.weight_bytes == 3_408_462_080
    assert (spec.abi_version, spec.reserved, spec.max_tokens, spec.output_width) == (2, 4, 55 * 55, 6144)
    assert (spec.hidden, spec.depth, spec.heads, spec.head_dim, spec.intermediate) == (2048, 30, 16, 128, 6144)
    assert spec.patch == 0 and spec.merger_extra[0] == 2048 * 588 * 2
    layer = 3 * 2048 * 2048 * 2 + 2048 * 2048 * 2 + 2 * 6144 * 2048 * 2 + 2048 * 6144 * 2 + 2 * 256 * 16
    assert spec.blocks[1].qkv - spec.blocks[0].qkv == layer
    for block in (spec.blocks[i] for i in range(spec.depth)):
        assert block.proj == block.qkv + 3 * 2048 * 2048 * 2
        assert block.gate_up == block.proj + 2048 * 2048 * 2
        assert block.qkv_bias == block.key0_bias == gate.NO_OFFSET
    assert spec.merger_extra[5] - spec.merger_extra[4] == 6144 * 2
    assert spec.inv_freq == spec.weight_bytes - 64 * 4


def test_mistral_axial_frequencies_follow_pixtral_interleave():
    base = 1 / np.power(np.float32(10000), np.arange(0, 128, 2, dtype=np.float32) / np.float32(128))
    table = np.concatenate([base[0::2], base[1::2]])
    assert table[0] == 1 and table[32] == base[1] and table[63] == base[63]


def test_mistral_normalization_reads_the_checkpoint_processor(tmp_path):
    gate = load_gate()
    (tmp_path / "processor_config.json").write_text(PROCESSOR.read_text())
    lut = gate.normalization_lut(tmp_path)
    document = json.loads(PROCESSOR.read_text())["image_processor"]
    mean, std = np.asarray(document["image_mean"], np.float32), np.asarray(document["image_std"], np.float32)
    np.testing.assert_array_equal(lut, (np.arange(256, dtype=np.float32)[None] * np.float32(1 / 255)
                                        - mean[:, None]) / std[:, None])
    document["image_processor_type"] = "Other"
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": document}))
    with pytest.raises(ValueError):
        gate.normalization_lut(tmp_path)


def test_mistral_fixture_grids_fit_the_tower_and_are_even():
    gate = load_gate()
    for name, (gh, gw) in gate.GRIDS.items():
        assert gh % 2 == 0 and gw % 2 == 0 and gh // 2 <= 55 and gw // 2 <= 55, name
        _, _, rgb = gate.fixture(name)
        assert rgb.shape == (gh * 14, gw * 14, 3)
