"""Standard-v2 split: one sealed table, data-driven block counts, decode-first continuation."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_split import split, standard_half, standard_ids, standard_layout


def windows(blocks):
    out = []
    for block, count in blocks.items():
        prefix = block.lower()
        for i in range(count - (block == "E")):
            out.append({"id": f"{prefix}{i:02d}", "block": block})
        if block == "E":
            out.append({"id": "legacy", "block": "E"})
    return out


def test_sealed_table_is_the_rust_table_and_hash():
    data = split()
    assert len(data["sealed"]["decode"]) == len(data["sealed"]["prefill"]) == 32
    source = (ROOT / "rust/crates/cuteafd-bench/src/fidelity_dataset.rs").read_text()
    assert 'include_str!("../../../../python/reference/fidelity_split.json")' in source
    assert "pub const STANDARD_DECODE" not in source and "pub const STANDARD_PREFILL" not in source


def test_original_64_window_allocation_reproduces_the_sealed_halves():
    w = windows({"A": 26, "B": 12, "C": 10, "D": 6, "E": 10})
    assert standard_layout(w) == {"A": 26, "B": 12, "C": 10, "D": 6, "E": 10}
    assert standard_ids(w, "decode") == split()["sealed"]["decode"]
    assert standard_ids(w, "prefill") == split()["sealed"]["prefill"]


def test_extension_continues_decode_first_past_the_sealed_range():
    assert [standard_half(f"b{i}") for i in range(12, 16)] == ["decode", "prefill", "decode", "prefill"]
    assert standard_half("a26") == "decode" and standard_half("e09") == "decode"
    w = windows({"A": 26, "B": 15, "C": 10, "D": 6, "E": 10})
    assert standard_ids(w, "decode")[-2:] == ["b12", "b14"] and standard_ids(w, "prefill")[-1] == "b13"


@pytest.mark.parametrize("mutate", ["gap", "duplicate", "misnamed", "foreign"])
def test_layout_fails_closed(mutate):
    w = windows({"A": 26, "B": 15, "C": 10, "D": 6, "E": 10})
    if mutate == "gap":
        w = [x for x in w if x["id"] != "b13"]
    elif mutate == "duplicate":
        w.append(dict(w[0]))
    elif mutate == "misnamed":
        next(x for x in w if x["id"] == "b14")["block"] = "A"
    else:
        w.append({"id": "z00", "block": "Z"})
    with pytest.raises(ValueError):
        standard_layout(w)


@pytest.mark.parametrize("path", [
    "~/.cache/cuteafd/builds/fidelity-muse-mistral/mistral/windows-v2.json",
    "~/.cache/cuteafd/builds/fidelity-muse-mistral/muse/windows-v2.json",
    "~/.cache/cuteafd/builds/kimi-k3-golden/published-effort-audit/deepseek_v41-v2_20261017_1017/windows.json",
])
def test_staged_and_published_sets_lay_out(path):
    file = pathlib.Path(path).expanduser()
    if not file.exists():
        pytest.skip("local set not present")
    panel = json.loads(file.read_text())
    counts = standard_layout(panel["windows"])
    assert sum(counts.values()) == len(panel["windows"])
    for half in ("decode", "prefill"):
        assert standard_ids(panel["windows"], half)
