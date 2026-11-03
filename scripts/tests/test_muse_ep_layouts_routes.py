"""ep_layouts.py on real Muse routes: --routes is read (digests verified) or refused,
never silently replaced by synthetic routes."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "python/reference/families/muse_spark/ep_layouts.py"


def write_layer(root: Path, layer: int, tokens: int = 3000, seed: int = 0, corrupt: bool = False) -> None:
    rng = np.random.default_rng(seed + layer)
    directory = root / f"layer{layer:03d}"
    directory.mkdir(parents=True)
    files = {}
    for window in ("a00", "a01"):
        ids = np.stack([rng.choice(256, 8, replace=False) for _ in range(tokens)]).astype("<i2")
        weights = rng.random((tokens, 8)).astype("<f4")
        for kind, array in (("ids", ids), ("weights", weights)):
            data = array.tobytes()
            (directory / f"{window}.{kind}.bin").write_bytes(data)
            files[f"{window}.{kind}"] = hashlib.sha256(data).hexdigest()
    if corrupt:
        files["a00.ids"] = "0" * 64
    (directory / "index.json").write_text(json.dumps({"schema": "cuteafd.muse.routes/1", "layer": layer,
        "experts": 256, "top_k": 8, "ids_dtype": "<i2", "weights_dtype": "<f4", "files": files}))


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), "--samples", "2", "--widths", "1,8", *args],
                          capture_output=True, text=True, timeout=300)


def test_real_routes_are_used_and_reported(tmp_path):
    for layer in (0, 1):
        write_layer(tmp_path, layer)
    out = run("--routes", str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert "routes: real (2 layers, EP owner staggered)" in out.stdout
    assert "synthetic" not in out.stdout
    assert "hottest expert / mean" in out.stdout and "EP balanced: max rank / mean" in out.stdout


@pytest.mark.parametrize("case", ["missing", "empty", "corrupt"])
def test_unreadable_routes_refuse_instead_of_falling_back(tmp_path, case):
    routes = tmp_path / "routes"
    if case == "empty":
        routes.mkdir()
    elif case == "corrupt":
        write_layer(routes, 0, corrupt=True)
    out = run("--routes", str(routes))
    assert out.returncode != 0
    assert "synthetic" not in out.stdout and "Decode/verify" not in out.stdout
    assert {"missing": "not a readable directory", "empty": "no layerNNN route directories",
            "corrupt": "sha256 differs"}[case] in (out.stdout + out.stderr)


def test_balanced_owner_needs_real_routes():
    out = run("--owner", "balanced")
    assert out.returncode != 0 and "--owner balanced needs --routes" in (out.stdout + out.stderr)
