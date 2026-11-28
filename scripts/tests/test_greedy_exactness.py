"""greedy-exactness.py compare and client argument parsing (no server)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("greedy_exactness", ROOT / "scripts/bench/greedy-exactness.py")
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)


def gen(path, variant=None, levels=(1, 4), ids=("p0", "p1")):
    """Write a greedy.json; variant(level, id, text) may alter the text."""
    cases = []
    for lv in levels:
        for i in ids:
            text = f"{i}-" + "abcdefgh" * 8
            if variant:
                text = variant(lv, i, text)
            cases.append({"level": lv, "id": i, "content": text, "reasoning_content": "r-" + i})
    path.write_text(json.dumps({"model": "m", "cases": cases}))
    return str(path)


def test_equal(tmp_path, capsys):
    a, b = gen(tmp_path / "a.json"), gen(tmp_path / "b.json")
    assert ge.main(["compare", a, b]) == 0
    assert "4/4" in capsys.readouterr().out


def test_mismatch_offset(tmp_path, capsys):
    a = gen(tmp_path / "a.json")
    b = gen(tmp_path / "b.json", lambda lv, i, t: t[:5] + "X" + t[6:] if (lv, i) == (4, "p1") else t)
    assert ge.main(["compare", a, b]) == 1
    out = capsys.readouterr().out
    assert "level=4 id=p1 field=content offset=5" in out
    assert "level c1: 2/2" in out and "level c4: 1/2" in out


def test_cross_level_match(tmp_path, capsys):
    # A's output differs per level; B at level 4 reproduces A's level-1 output.
    a = gen(tmp_path / "a.json", lambda lv, i, t: t + ("!" if lv == 4 else ""))
    b = gen(tmp_path / "b.json")
    assert ge.main(["compare", a, b]) == 1
    assert "B@c4 equals A at levels: [1]" in capsys.readouterr().out


def test_missing_case(tmp_path):
    a = gen(tmp_path / "a.json")
    b = gen(tmp_path / "b.json", levels=(1,))
    assert ge.main(["compare", a, b]) == 1


def test_client_args():
    args = ge.build_parser().parse_args(["client", "f.json"])
    assert args.levels == [1, 4, 10] and args.fixture == "f.json"
    assert ge.build_parser().parse_args(["client", "f.json", "--levels", "2,8"]).levels == [2, 8]
    with pytest.raises(SystemExit):
        ge.build_parser().parse_args(["client", "f.json", "--levels", "0"])
