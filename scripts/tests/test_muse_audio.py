"""CPU guards for bundled raw-PCM span and admission contracts."""
import ast
import ctypes
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audio = load("muse_pcm_reference", "python/reference/families/muse_spark/audio.py")
gate = load("muse_pcm_qualifier", "python/tools/qualify/muse_spark/qualify-audio.py")


@pytest.mark.parametrize("samples,tokens", [(1, 1), (1919, 1), (1920, 1), (1921, 2),
    (30719, 16), (30720, 16), (30721, 17), (7_200_000, 3750)])
def test_official_span_rounding(samples, tokens):
    assert audio.token_count(samples) == tokens
    assert audio.placeholder(samples).encode() == (
        b"<|soft_audio_start|>" + b"<|audio|>" * tokens + b"<|soft_audio_end|>")


@pytest.mark.parametrize("samples", [0, -1, 7_200_001])
def test_invalid_clip_fails_before_admission(samples):
    with pytest.raises(ValueError):
        audio.placeholder(samples)
    with pytest.raises(ValueError):
        gate.scratch_bytes(samples)


@pytest.mark.parametrize("samples", [1, 1920, 1921, 30720, 30721, 7_200_000])
def test_planner_scratch_matches_runtime_arena(samples):
    # Independent allocation walk from pcm_audio_runtime.cu, including alignment.
    tokens = min(-(-samples // 1920), 16)
    rows = tokens * 1920
    maximum = tokens * 8192
    for width, patch in [(256, 16), (512, 6), (1024, 5), (2048, 4)]:
        rows //= patch
        maximum = max(maximum, rows * width)
    allocations = [maximum * 2] * 3 + [maximum * 8] * 2
    allocations += [width * 6 * 2 for width in (256, 512, 1024, 2048) for _ in range(2)]
    scratch = sum(-(-size // 256) * 256 for size in allocations)
    assert gate.scratch_bytes(samples) == scratch
    charge = gate.Ledger(weights=4096, scratch=scratch, device_allocations=2)
    assert charge.weights + charge.scratch == 4096 + scratch
    assert gate.scratch_bytes(30720) == gate.scratch_bytes(7_200_000)


def test_pcm_c_abi_layout():
    assert [ctypes.sizeof(t) for t in (gate.Block, gate.Stage, gate.Spec, gate.Ledger)] == [56, 136, 600, 32]


def test_reference_import_does_not_require_cuda_or_torch():
    tree = ast.parse((ROOT / "python/reference/families/muse_spark/audio.py").read_text())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("torch" not in ast.unparse(node) for node in imports)
