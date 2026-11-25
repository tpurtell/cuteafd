from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_package_exl3_aot():
    spec = importlib.util.spec_from_file_location(
        "package_exl3_aot_verify_test",
        ROOT / "python/tools/aot/package_exl3_aot.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _base_config(spark0: str = "spark0", extra: str = "") -> str:
    return (
        "MODEL_ID=wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC\n"
        "RTX_GPUS=1\n"
        "SPARK_COUNT=1\n"
        f"SPARK_0_HOST={spark0}\n"
        "SPARK_0_LANE_A=10.55.0.1\n"
        f"{extra}"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("SPARK_0_HOST", "REPLACE-ME-spark-0"),
        ("SPARK_0_HOST", "replace-me-spark-0"),
        ("SPARK_0_HOST", "RePlAcE-Me-host"),
        ("SPARK_HOSTS", "spark0,replace-me-spark-1"),
        ("WIP_BUILD_SEED", "REPLACE-ME-seed"),
    ],
)
def test_release_load_config_refuses_placeholder_hosts_case_insensitive(
    tmp_path: Path, key: str, value: str
) -> None:
    cfg = tmp_path / "placeholder.config"
    if key == "SPARK_0_HOST":
        cfg.write_text(_base_config(spark0=value))
    elif key == "SPARK_HOSTS":
        cfg.write_text(_base_config(spark0="spark0", extra=f"SPARK_HOSTS={value}\n"))
    else:
        cfg.write_text(_base_config(spark0="spark0", extra=f"{key}={value}\n"))
    res = subprocess.run(
        ["bash", "-euc", 'source scripts/lib/release-common.sh; release_load_config "$1"', "test", str(cfg)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode != 0
    assert "placeholder host" in res.stderr and key in res.stderr and value in res.stderr


@pytest.mark.parametrize(
    "example",
    [
        "examples/configs/glm53-exl3-k4-mc-max.config",
        "examples/configs/v41-flash-strided-max.config",
        "examples/configs/mistral-large4-exl3-k4-max.config",
    ],
)
def test_unedited_example_templates_fail_before_hwlock_or_ssh(tmp_path: Path, example: str) -> None:
    # Stub ssh, docker and hwlock into a marker log; none may be called.
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    marker = tmp_path / "invoked.log"
    for tool in ("ssh", "docker"):
        script = stub_bin / tool
        script.write_text(f"#!/usr/bin/env bash\necho {tool} \"$*\" >> {marker}\nexit 99\n")
        script.chmod(0o755)
    env = {**os.environ, "PATH": f"{stub_bin}:{os.environ['PATH']}"}

    wip = subprocess.run(
        ["bash", "wip.sh", "--slot", "placeholder-test", "--role", "both", "--config", example],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert wip.returncode != 0
    assert "placeholder host" in wip.stderr
    assert not marker.exists(), f"external tool called before placeholder check: {marker.read_text()}"

    run = subprocess.run(
        ["bash", "run.sh", "--config", example, "--dry-run"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode != 0
    assert "placeholder host" in run.stderr
    assert not marker.exists(), f"external tool called before placeholder check: {marker.read_text()}"


def test_wip_refuses_placeholder_seed_option(tmp_path: Path) -> None:
    cfg = tmp_path / "valid.config"
    cfg.write_text(_base_config(spark0="spark0"))
    res = subprocess.run(
        [
            "bash",
            "wip.sh",
            "--slot",
            "placeholder-seed",
            "--role",
            "both",
            "--config",
            str(cfg),
            "--seed",
            "RePlAcE-Me-seed",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode != 0
    assert "WIP_SEED_HOST" in res.stderr and "placeholder host" in res.stderr


def test_exl3_verify_reports_missing_and_unlisted_names_and_counts_capped(tmp_path: Path) -> None:
    mod = _load_package_exl3_aot()
    pkg = tmp_path / "exl3-glm-k45"
    pkg.mkdir()
    expected_files = {f"expected/file_{i:02d}.bin": {"bytes": 0, "sha256": "0" * 64} for i in range(25)}
    manifest = {
        "schema": "cuteafd.exl3-package.v1",
        "codebook": "mul1",
        "sparkinfer_revision": "rev",
        "role": "spark",
        "files": expected_files,
        "variants": [],
    }
    (pkg / "manifest.json").write_text(json.dumps(manifest))
    # Create only 2 of the 25 expected files (23 missing) and 24 unlisted files.
    (pkg / "expected").mkdir()
    for i in range(2):
        (pkg / f"expected/file_{i:02d}.bin").write_bytes(b"")
    (pkg / "tp4-rank0").mkdir()
    for i in range(24):
        (pkg / f"tp4-rank0/stale_{i:02d}.so").write_bytes(b"")

    with pytest.raises(ValueError) as exc:
        mod.verify(pkg)
    msg = str(exc.value)
    assert "EXL3 package contains missing or unlisted files" in msg
    assert "missing (23):" in msg
    assert "expected/file_02.bin" in msg
    assert "expected/file_21.bin" in msg
    assert "expected/file_22.bin" not in msg
    assert "(+3 more)" in msg
    assert "unlisted (24):" in msg
    assert "tp4-rank0/stale_00.so" in msg
    assert "tp4-rank0/stale_19.so" in msg
    assert "tp4-rank0/stale_20.so" not in msg
    assert "(+4 more)" in msg
