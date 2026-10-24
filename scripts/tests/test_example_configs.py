"""The per-family starting configs in examples/configs/ (current serving recipes).

Each one parses through the real launcher's dry-run or argv path (CPU only:
docker, ssh, nvidia-smi and nest are stubs that print their argv), and its
speculator and pool keys match the launcher's bare default for that family, so
a user who drops them gets the same launch and the two cannot drift apart.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "configs"
STARTING = re.compile(r"^(?P<model>[a-z0-9.-]+)-(?P<profile>min|max)\.config$")
TP_EP = {"tp2ep2-native.config", "tp2ep3-native.config", "tp3ep1-native.config",
         "tp3ep2-native.config", "tp4ep1-explicit-native.config", "tp6ep1-native.config",
         "exl3-compact-tp3.config"}
# WIP qualification configs for families without a release card yet (#273, #274):
# served from ./wip.sh slots only, checked by test_serve_launch_new_families.py.
QUALIFICATION = {"kimi-k3-exl3-max.config", "kimi-k3-exl3-k6-max.config", "step5-fp8-max.config", "step5-nvfp4-min.config"}
V4_PREP = {"glm53-exlr8-min.config", "glm53-exlr8-max.config",
           "qwen38-27b-bf16-min.config", "qwen38-27b-fp8-min.config"}
PRIVATE = ("raptor", "ostrich", "dodo", "emu", "kiwi", "rhea", "moa", "10.55.", "/home/", "/mnt/",
           "sparknest")

# One stub config.json per checkpoint the starting configs name: the keys the
# launcher reads from the snapshot (family, routed layers, quantization, MTP,
# dSpark, towers). Real snapshots are not needed.
GLM5 = {"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
GLMF = {"model_type": "glm5_next", "num_hidden_layers": 2, "mlp_layer_types": ["sparse"] * 2,
        "layer_types": ["linear_attention", "deepseek_sparse_attention"], "vision_config": {"depth": 2}}
DSV4 = {"model_type": "deepseek_v4", "dspark_block_size": 5}
QWEN = {"model_type": "qwen4_exp", "vision_config": {"depth": 2},
        "text_config": {"num_hidden_layers": 2, "mtp_num_hidden_layers": 1,
                        "layer_types": ["linear_attention", "full_attention"]}}
MIMO = {"model_type": "mimo_v2", "num_hidden_layers": 2, "moe_layer_freq": [0, 1],
        "quantization_config": {"store_dtype": "mxfp4"}, "vision_config": {"depth": 2}}
NVFP4_GROUPS = {"config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}
CHECKPOINTS = {
    "deepseek-ai/DeepSeek-V4.1-Flash": {"model_type": "deepseek_v41"},
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": {"model_type": "deepseek_v41"},
    "nvidia/DeepSeek-V4.1-Flash-NVFP4": {"model_type": "deepseek_v41",
                                         "quantization_config": {"moe_quant_algo": "NVFP4"}},
    "deepseek-ai/DeepSeek-V4-Flash-0731": DSV4,
    "wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1": {**DSV4, "quantization_config": {"quant_method": "exl3"}},
    "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1": GLM5,
    "nvidia/GLM-5.3-NVFP4": GLM5,
    "zai-org/GLM-5.3-Flash": GLMF,
    "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8": GLMF,
    "nvidia/GLM-5.3-Flash-NVFP4": GLMF,
    "brandonmusic/GLM-5.3-Flash-tr3-4bpw": GLMF,
    "XiaomiMiMo/MiMo-V2.6-Flash-MOPD": {**MIMO, "hidden_size": 4096},
    "XiaomiMiMo/MiMo-V2.6-Pro-MOPD": {**MIMO, "hidden_size": 6144},
    "wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1": {**QWEN, "quantization_config": {"quant_method": "exl3"}},
    "nvidia/Qwen3.8-Flash-Next-NVFP4": {**QWEN, "quantization_config": {"quant_method": "modelopt", **NVFP4_GROUPS}},
    "kindlingai/glm-5.3-exlr8-k2-k3-k4": GLM5,
    "Qwen/Qwen3.8-27B": {"model_type": "qwen3_5", "text_config": {
        "hidden_size": 5120, "num_hidden_layers": 32,
        "layer_types": ["linear_attention"] * 31 + ["full_attention"]}},
    "Qwen/Qwen3.8-27B-FP8": {"model_type": "qwen3_5", "text_config": {
        "hidden_size": 5120, "num_hidden_layers": 32,
        "layer_types": ["linear_attention"] * 31 + ["full_attention"]}},
    "RadixArk/Qwen3.8-27B-DSpark": {"speculators_model_type": "dspark"},
    "stepfun-ai/Step-5-Preview": {"model_type": "step3p5", "num_hidden_layers": 8},
    "stepfun-ai/Step-5-Preview-NVFP4": {"model_type": "step3p5", "num_hidden_layers": 8},
    "wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6": {"model_type": "kimi_k3", "num_hidden_layers": 8},
    # Drafters the starting configs name.
    "incoai/GLM-5.3-DFlash2": {"architectures": ["DFlash2DraftModel"]},
    "incoai/GLM-5.3-Flash-DFlash2": {"architectures": ["DFlash2DraftModel"]},
}
FAMILY_DOCS = {"deepseek_v41": "deepseek_v41", "deepseek_v4": "deepseek_v4", "glm5": "glm5",
               "glm5_flash": "glm5_flash", "mimo_v2": "mimo_v2", "qwen4": "qwen4"}
# Launcher keys that select the speculator and the KV pool.
SPECULATOR_KEYS = ("SPECULATOR", "SPECULATOR_MODEL_ID", "SPECULATOR_DEPTH")
POOL_KEYS = ("POOL_TOKENS",)
# The one named exception (decided 2026-10-10): the V4.1 cards ran with
# POOL_TOKENS=auto, so the V4.1 examples pin it, while the V4.1 loader leaves the key
# empty when absent (run-family.sh's V4.1 section then resolves auto). Roadmap step S5 moves V4.1 onto
# run-family.sh, where the defaults are unified and this exception goes away.
POOL_EXCEPTIONS = {("deepseek_v41", "POOL_TOKENS"): "auto"}


def starting_configs() -> list[Path]:
    return sorted(p for p in EXAMPLES.glob("*.config") if p.name not in TP_EP | QUALIFICATION | V4_PREP)


def keys(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines()
                if re.match(r"^[A-Z_0-9]+=", line))


def family_of(path: Path) -> str:
    config = CHECKPOINTS[keys(path)["MODEL_ID"]]
    out = subprocess.run(["python3", str(ROOT / "scripts" / "lib" / "checkpoint-family.py"), "/dev/stdin"],
                         input=json.dumps(config), capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def _hub(tmp_path: Path) -> Path:
    hf = tmp_path / "hf"
    for model, config in CHECKPOINTS.items():
        root = hf / "hub" / f"models--{model.replace('/', '--')}"
        (root / "refs").mkdir(parents=True)
        (root / "refs" / "main").write_text("abc")
        (root / "snapshots" / "abc").mkdir(parents=True)
        (root / "snapshots" / "abc" / "config.json").write_text(json.dumps(config))
    draft = hf / "hub/models--RadixArk--Qwen3.8-27B-DSpark/snapshots"
    (draft / "85ef153be924f17ce4bf62726954eeaa4a73e854").symlink_to(draft / "abc")
    return hf


def _pin_snapshots(hf: Path, config: dict[str, str]) -> None:
    """Expose each pinned revision as the stub snapshot."""
    revision = config.get("MODEL_REVISION")
    if revision:
        root = hf / "hub" / f"models--{config['MODEL_ID'].replace('/', '--')}" / "snapshots"
        if not (root / revision).exists():
            (root / revision).symlink_to(root / "abc")


def _stubs(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Every tool prints its argv to stderr; the coordinator stub reports ready.
    log = '#!/usr/bin/env bash\nprintf "%s " "$(basename "$0")" "$@" >&2; echo >&2\n'
    for tool in ("docker", "ssh", "nest"):
        body = log
        if tool == "docker":
            body += 'case "$*" in *"PROGRAMS.json"*) exit 1 ;; esac\n'
        if tool == "ssh":
            body += 'case "$*" in *"docker logs"*) echo "worker ready" ;; esac\n'
        (bin_dir / tool).write_text(body)
        (bin_dir / tool).chmod(0o755)
    (bin_dir / "curl").write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' \'{"data":[{"id":"test/model"}]}\'\n')
    (bin_dir / "nvidia-smi").write_text('#!/usr/bin/env bash\ncase "$*" in *memory.total*) echo 98304 ;; '
                                        '*memory.free*) echo 97000 ;; *query-gpu=uuid*) echo GPU-00000000-0000-0000-0000-000000000000 ;; *) printf "%s\\n" 0 1 ;; esac\n')
    for tool in ("curl", "nvidia-smi"):
        (bin_dir / tool).chmod(0o755)
    from launcher_sandbox import stub_cuda_python
    stub_cuda_python(bin_dir, 98304 * 2**20)
    return bin_dir


def _family_launch(tmp_path: Path, config: Path) -> subprocess.CompletedProcess[str]:
    """run.sh --config CONFIG for a non-V4.1 family: dispatch to the real run-family.sh."""
    hf = _hub(tmp_path)
    _pin_snapshots(hf, keys(config))
    bin_dir = _stubs(tmp_path)
    # The encoder plan the coordinator image would print: Spark-first vision with Spark
    # ranks; with local experts the automatic KV target keeps the tower off (the cards).
    count = int(keys(config).get("SPARK_COUNT", sum(key.endswith("_HOST") for key in keys(config))))
    plan = {**(SPARK_PLAN if count else LOCAL_PLAN), "spark_ranks": count}
    if plan is not None:
        # The encoder plan the coordinator image would print for this layout.
        docker = bin_dir / "docker"
        described = subprocess.check_output(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"),
            str(hf / "hub" / f"models--{keys(config)['MODEL_ID'].replace('/', '--')}" / "snapshots/abc/config.json")], text=True).split()
        family, _, first, last = described
        stub = tmp_path / "deployment.py"
        stub.write_text("""import json, sys
args = sys.argv[1:]
def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
report = json.loads(REPORT)
family, first, last = FAMILY, FIRST, LAST
print(json.dumps(dict(report, version=1, family=family, executable=True, report=report,
    launch=dict(family=family, first_expert_layer=first, last_expert_layer=last),
    spark_admission="coordinator_authoritative", decode_lanes=1,
    peers=value("--deployment-peers", "").split(",")[:report["spark_ranks"]],
    speculator=value("--speculator", "off"))))
""".replace("REPORT", repr(json.dumps(plan))).replace("FAMILY", repr(family))
            .replace("FIRST", first).replace("LAST", "None" if last == "-1" else last))
        docker.write_text(docker.read_text() + f'case "$*" in *"cuteafd plan"*) exec python3 "{stub}" "$@" ;; esac\n')
    env = {**os.environ, "HF_HOME": str(hf), "HOME": str(tmp_path / "home"),
           "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(ROOT / "run.sh"), "--config", str(config)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)


def _strip(config: Path, tmp_path: Path, drop: tuple[str, ...]) -> Path:
    """The example without the named keys: what a user gets from the launcher's bare defaults."""
    lines = [line for line in config.read_text().splitlines()
             if not any(line.startswith(f"{key}=") for key in drop)]
    bare = tmp_path / f"bare-{config.name}"
    bare.write_text("\n".join(lines) + "\n")
    return bare


def _coordinator(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, result.stderr
    return next(line for line in result.stderr.splitlines() if " --snapshot " in line and " serve-" in line)


# Encoder plans for families with towers: Spark-first vision on rank 0, or off.
SPARK_PLAN = {"placement_supported": True, "fits": True, "spark_ranks": 2, "encoder_plan_hash": "ab" * 32,
              "prefill_unit_rows": 4096, "prefill_waves": 1, "prefill_span_rows": 4096,
              "encoder": {"kind": {"kind": "spark", "rank": 0}, "replicas": []},
              "audio_encoder": {"kind": {"kind": "off"}, "replicas": []}}
LOCAL_PLAN = {**SPARK_PLAN, "spark_ranks": 0, "encoder": {"kind": {"kind": "off"}, "replicas": []}}


def test_one_starting_config_per_release_card() -> None:
    names = {p.name for p in starting_configs()}
    expected = {f"{model}-{profile}.config" for model in (
        "deepseek-v41-mxfp4", "deepseek-v4-flash-mxfp4", "deepseek-v4-pro-exl3",
        "glm53-exl3", "glm53-nvfp4", "glm53-flash-fp8", "glm53-flash-exl3", "glm53-flash-nvfp4",
        "glm53-flash-tr3", "mimo-v26-flash-mxfp4", "mimo-v26-pro-mxfp4") for profile in ("min", "max")}
    # Qwen fits one RTX: the 2x RTX column is n/a, so only min exists.
    expected |= {"qwen38-exl3-min.config", "qwen38-nvfp4-min.config"}
    assert names == expected
    for path in starting_configs():
        assert STARTING.match(path.name), path.name


def test_release_smoke_matrix_covers_owned_columns_with_current_defaults() -> None:
    matrix = json.loads((ROOT / "scripts/bench/release-smoke.example.json").read_text())
    assert matrix["profile"] == "smoke"
    entries = matrix["entries"]
    assert len(entries) == 26
    assert len({entry["name"] for entry in entries}) == len(entries)
    defaults = matrix["defaults"]["set"]
    stale = {"CONCURRENCY", "MAX_CONTEXT_TOKENS", "HOST_CACHE_BYTES", "BENCH_SIMULATED",
             "TP2_ATTENTION", "TP2_QUERY_PROJECTION", "TP2_OUTPUT_PROJECTION", "TP2_DSPARK_EXPERTS"}
    assert not stale.intersection(defaults)
    models = {}
    ports = set()
    for entry in entries:
        values = {**keys(ROOT / entry["config"]), **defaults, **entry["set"]}
        assert not stale.intersection(entry["set"])
        assert values["POOL_TOKENS"] == "auto"
        if entry["family"] != "deepseek_v41":
            assert not any(re.fullmatch(r"SPARK_\d+_LANE_B", key) for key in values)
        else:
            for rank in range(len(entry["sparks"])):
                assert values[f"SPARK_{rank}_LANE_B"].startswith("10.55.1.")
        assert int(values["RTX_GPUS"]) == len(entry["gpus"])
        assert int(values["COORDINATOR_GPU"]) == entry["gpus"][0]
        assert int(values["SPARK_COUNT"]) == len(entry["sparks"])
        assert len(entry["gpus"]) == (1 if entry["name"].endswith("-min") else 2)
        for rank, host in enumerate(entry["sparks"]):
            assert values[f"SPARK_{rank}_HOST"] == host
            assert values[f"SPARK_{rank}_LANE_A"].startswith("10.55.0.")
        if entry["name"] in {
            "qwen38-exl3-min", "qwen38-exl3-max", "qwen38-nvfp4-max", "glm53f-exl3-max"
        }:
            assert not entry["sparks"]
            assert values["EXPERT_BACKEND"] == "local"
        if entry["name"] in {"glm53f-nvfp4-max", "glm53f-tr3-max", "qwen38-nvfp4-min"}:
            assert entry["sparks"] == ["ostrich", "dodo"]
            assert values["EXPERT_BACKEND"] == "spark"
        for key in ("ADDR", "EXPERT_PORT"):
            assert values[key] not in ports
            ports.add(values[key])
        models.setdefault(values["MODEL_ID"], set()).add(len(entry["gpus"]))
    assert len(models) == 13
    assert all(columns == {1, 2} for columns in models.values())


def test_v41_max_uses_six_sparks_and_keeps_tp4_reference() -> None:
    pool = ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"]
    config = keys(ROOT / "examples/configs/deepseek-v41-mxfp4-max.config")
    assert config["RTX_GPUS"] == "2" and config["SPARK_COUNT"] == "6"
    for rank in range(6):
        assert config[f"SPARK_{rank}_HOST"] == f"REPLACE-ME-spark-{rank}"
        assert config[f"SPARK_{rank}_LANE_A"] == f"192.0.2.{rank + 1}"
    for path in ("scripts/bench/kits/gate/matrix-gate.json",
                 "scripts/bench/release-smoke.example.json"):
        matrix = json.loads((ROOT / path).read_text())
        entries = {entry["name"]: entry for entry in matrix["entries"]}
        maximum = entries["v41-flash-max"]
        assert maximum["gpus"] == [0, 1] and maximum["sparks"] == pool
        assert maximum["set"]["SPARK_COUNT"] == "6"
        assert [maximum["set"][f"SPARK_{rank}_HOST"] for rank in range(6)] == pool
        if "v41-flash-ref-tp4" in entries:
            reference = entries["v41-flash-ref-tp4"]
            assert reference["gpus"] == [0, 1] and reference["sparks"] == pool[:4]
            assert reference["set"]["SPARK_COUNT"] == "4"
            for key in ("ADDR", "EXPERT_PORT"):
                assert sum(entry.get("set", {}).get(key) == reference["set"][key]
                           for entry in entries.values()) == 1
        elif "matrix-gate" in path:
            pytest.fail("gate kit must retain the TP4 reference card")


def test_v3_release_matrix_covers_shipped_quants_reference_layouts_and_real_5090() -> None:
    matrix = json.loads((ROOT / "scripts/bench/release-smoke-v3.json").read_text())
    values_of = lambda entry: {**keys(ROOT / entry["config"]), **matrix["defaults"]["set"], **entry["set"]}
    assert matrix["profile"] == "smoke"
    assert matrix["build"] == "REPLACE-ME-v3-rc"
    defaults = matrix["defaults"]["set"]
    assert all("REPLACE-ME-v3-rc" in defaults[key] for key in
               ("COORDINATOR_DOCKER_INFERENCE", "SPARK_EXPERT_DOCKER_INFERENCE"))
    entries = {entry["name"]: entry for entry in matrix["entries"]}
    assert len(entries) == len(matrix["entries"]) == 24
    columns = {}
    ports = set()
    for name, entry in entries.items():
        values = {**keys(ROOT / entry["config"]), **defaults, **entry["set"]}
        assert values["MODEL_ID"] in CHECKPOINTS
        assert entry["family"] in FAMILY_DOCS
        assert "BENCH_SIMULATED" not in values
        assert int(values["RTX_GPUS"]) == len(entry["gpus"])
        assert int(values["COORDINATOR_GPU"]) == entry["gpus"][0]
        assert int(values["SPARK_COUNT"]) == len(entry["sparks"])
        assert values["SPECULATOR"] in {"dspark", "dflash2", "mtp"}
        assert values["POOL_TOKENS"] in {"auto", "2097152"}
        for rank, host in enumerate(entry["sparks"]):
            assert host == values[f"SPARK_{rank}_HOST"] == f"REPLACE-ME-spark-{rank}"
            assert values[f"SPARK_{rank}_LANE_A"] == f"192.0.2.{rank + 1}"
        for key in ("ADDR", "EXPERT_PORT"):
            assert values[key] not in ports
            ports.add(values[key])
        if name.endswith(("-min", "-max")):
            columns.setdefault(values["MODEL_ID"], set()).add(name.rsplit("-", 1)[1])
        if name.endswith("-5090"):
            assert values["BENCH_HARDWARE_CLASS"] == "5090"
            assert len(entry["gpus"]) == 1
            assert "BENCH_SIMULATED" not in entry["set"]
    # v3 publishes one column per checkpoint, except V4.1 and GLM Flash EXL3 (both).
    assert len(columns) == 12
    assert {model: sorted(sides) for model, sides in columns.items() if len(sides) == 2} == {
        values_of(entries["v41-flash-min"])["MODEL_ID"]: ["max", "min"],
        values_of(entries["glm53f-exl3-min"])["MODEL_ID"]: ["max", "min"]}
    assert len([name for name in entries if name.endswith("-5090")]) == 8
    assert len(entries["v41-flash-reference-tp4"]["sparks"]) == 4
    assert len(entries["v41-flash-reference-tp6"]["sparks"]) == 6
    for name in ("glm53f-exl3-max", "qwen38-exl3-min"):
        assert not entries[name]["sparks"]
        assert entries[name]["set"]["EXPERT_BACKEND"] == "local"
    assert not any("reference-all-spark" in name or "reference-half" in name for name in entries)
    text = json.dumps(matrix)
    assert not any(private in text for private in PRIVATE)


@pytest.mark.parametrize("gpu_name,expected", [
    ("NVIDIA GeForce RTX 5090", 0),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 1),
])
def test_v3_5090_precheck_requires_real_hardware(tmp_path: Path, gpu_name: str, expected: int) -> None:
    matrix = json.loads((ROOT / "scripts/bench/release-smoke-v3.json").read_text())
    probe = tmp_path / "nvidia-smi"
    probe.write_text(f"#!/bin/sh\nprintf '%s\\n' '{gpu_name}'\n")
    probe.chmod(0o755)
    for entry in matrix["entries"]:
        if not entry["name"].endswith("-5090"):
            continue
        hook = entry["precheck"]
        assert hook["required"]
        result = subprocess.run(hook["argv"], env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
                                timeout=hook["timeout_s"], check=False)
        assert result.returncode == expected, entry["name"]


def test_glmf_exl3_smoke_and_gate_cards_use_the_published_core_checkpoint() -> None:
    model = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8"
    revision = "d31ebc921f1f098b4a56d37aaccb3cdef4096184"
    for filename in ("scripts/bench/release-smoke.example.json", "scripts/bench/release-smoke-v3.json",
                     "scripts/bench/kits/gate/matrix-gate.json"):
        matrix = json.loads((ROOT / filename).read_text())
        entries = [entry for entry in matrix["entries"] if entry["family"] == "glm5_flash"
                   and "exl3" in entry["name"]]
        assert entries, filename
        for entry in entries:
            effective = {**keys(ROOT / entry["config"]), **matrix.get("defaults", {}).get("set", {}),
                         **entry.get("set", {})}
            assert effective["MODEL_ID"] == model, entry["name"]
            assert effective["MODEL_REVISION"] == revision, entry["name"]


def test_published_matrix_pins_current_reports() -> None:
    matrix = json.loads((ROOT / "benchmarks/current-matrix.json").read_text())
    assert len(matrix["cells"]) == 12
    for row in matrix["cells"]:
        # The 5090 column carries our simulated card where the checkpoint fits 32 GB.
        assert row["5090"]["status"] in ("current", "na")
        if row["5090"]["status"] == "current":
            assert row["5090"]["report"].endswith("-5090")
        current = [row[c] for c in ("1rtx", "2rtx") if row[c]["status"] == "current"]
        # v3 publishes one card per checkpoint (two for V4.1 and GLM Flash EXL3).
        assert current and all(row[c]["status"] in ("current", "na") for c in ("1rtx", "2rtx"))
        for cell in current:
            assert (ROOT / cell["report"] / "report.json").is_file()
    assert "simulated-5090" not in json.dumps(matrix)


@pytest.mark.parametrize("path", starting_configs(), ids=lambda p: p.name)
def test_starting_config_has_no_private_detail_and_names_the_release_pair(path: Path) -> None:
    text = path.read_text()
    for needle in PRIVATE:
        assert needle not in text.lower(), (path.name, needle)
    values = keys(path)
    cuteafd = keys(ROOT / "cuteafd.config")
    for image in ("COORDINATOR_DOCKER_INFERENCE", "SPARK_EXPERT_DOCKER_INFERENCE"):
        assert values[image] == cuteafd[image]
    sparks = int(values["SPARK_COUNT"])
    for rank in range(sparks):
        assert values[f"SPARK_{rank}_HOST"] == f"REPLACE-ME-spark-{rank}"
        assert values[f"SPARK_{rank}_LANE_A"] == f"192.0.2.{rank + 1}"
    assert not any(key.startswith(f"SPARK_{sparks}_") for key in values)
    assert values["RTX_GPUS"] == ("1" if path.name.endswith("-min.config") else "2")
    assert f"./run.sh --config examples/configs/{path.name} --dry-run" in text


@pytest.mark.parametrize("path", starting_configs(), ids=lambda p: p.name)
def test_family_doc_and_readme_link_every_starting_config(path: Path) -> None:
    doc = (ROOT / "docs" / "models" / f"{FAMILY_DOCS[family_of(path)]}.md").read_text()
    assert f"../../examples/configs/{path.name}" in doc
    assert f"({path.name})" in (EXAMPLES / "README.md").read_text()


V41 = [p for p in starting_configs() if p.name.startswith("deepseek-v41-")]
FAMILY = [p for p in starting_configs() if not p.name.startswith("deepseek-v41-")]


@pytest.mark.parametrize("entry", [entry for filename in ("release-smoke-v3.json", "release-smoke-v4.json")
                                     for entry in json.loads((ROOT / "scripts/bench" / filename).read_text())["entries"]],
                         ids=lambda entry: entry["name"])
def test_release_smoke_entry_passes_config_validation(entry: dict, tmp_path: Path) -> None:
    matrix = json.loads((ROOT / "scripts/bench/release-smoke-v3.json").read_text())
    values = {**keys(ROOT / entry["config"]), **matrix.get("defaults", {}).get("set", {}),
              **entry.get("set", {})}
    # These publication-only keys are stripped by bench smoke's write_config.
    for key in ("BENCH_SIMULATED", "BENCH_HARDWARE_CLASS"):
        values.pop(key, None)
    path = tmp_path / "entry.config"
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    hf = _hub(tmp_path)
    snapshot = hf / "hub" / f"models--{values['MODEL_ID'].replace('/', '--')}" / "snapshots/abc/config.json"
    checkpoint = json.loads(snapshot.read_text())
    if checkpoint["model_type"] == "glm5_next":
        # The reference-half entry names 22 of this checkpoint's 45 routed layers.
        checkpoint.update(num_hidden_layers=45, mlp_layer_types=["sparse"] * 45,
                          layer_types=["linear_attention"] * 45)
        snapshot.write_text(json.dumps(checkpoint))
    _pin_snapshots(hf, values)
    result = subprocess.run(
        ["bash", "-euc", 'source scripts/lib/release-common.sh; release_load_config "$1"',
         "test", str(path)], cwd=ROOT, env={**os.environ, "HF_HOME": str(hf)},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", V41, ids=lambda p: p.name)
def test_v41_starting_config_passes_the_native_config_validation(path: Path, tmp_path: Path) -> None:
    """run.sh --dry-run's config stage: release_load_config validates every key, the
    topology and the rails before any image or host is touched."""
    hf = _hub(tmp_path)
    _pin_snapshots(hf, keys(path))
    result = subprocess.run(
        ["bash", "-euc", 'source scripts/lib/release-common.sh; [[ "$(release_config_family "$1")" == deepseek_v41 ]]; '
         'release_load_config "$1"; printf "%s\\n" "$SPARK_COUNT" "$RTX_GPUS" "$POOL_TOKENS" "$SPECULATOR"',
         "test", str(path)], cwd=ROOT, env={**os.environ, "HF_HOME": str(hf)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    count, gpus, pool, dspark = result.stdout.split()
    assert (count, gpus, pool, dspark) == (keys(path)["SPARK_COUNT"], keys(path)["RTX_GPUS"], "auto", "dspark")


@pytest.mark.parametrize("path", FAMILY, ids=lambda p: p.name)
def test_family_starting_config_launches_through_run_family(path: Path, tmp_path: Path) -> None:
    """run.sh dispatches to run-family.sh, which validates every key and prints the argv
    it would start (docker/ssh are stubs)."""
    result = _family_launch(tmp_path, path)
    launch = _coordinator(result)
    values = keys(path)
    sparks = int(values["SPARK_COUNT"])
    assert ("--peers " in launch) == (sparks > 0)
    assert result.stderr.count("cuteafd expertd-native") == sparks
    assert "--pool-tokens 0" in launch
    expected = {"dspark": "--dspark", "dflash2": "--draft", "mtp": "--mtp"}[values["SPECULATOR"]]
    assert expected in [flag.split()[0] for flag in _speculation(launch)], launch
    if "SERVED_MODEL_ID" in values:
        assert f"--model-id {values['SERVED_MODEL_ID']}" in launch


def _speculation(launch: str) -> list[str]:
    """The speculator part of a coordinator argv: every --draft*/--mtp/--dspark flag
    with its value."""
    tokens, picked = launch.split(), []
    for index, token in enumerate(tokens):
        if token in ("--mtp", "--dspark") or token.startswith("--draft"):
            value = tokens[index + 1] if index + 1 < len(tokens) else ""
            picked.append(token if value.startswith("--") else f"{token} {value}")
    return picked


@pytest.mark.parametrize("path", FAMILY, ids=lambda p: p.name)
def test_family_starting_config_matches_the_launcher_defaults(path: Path, tmp_path: Path) -> None:
    """Dropping the speculator and pool keys gives the same launch: the example and
    run-family.sh's bare defaults cannot drift."""
    keep = _coordinator(_family_launch(tmp_path / "example", path))
    bare = _strip(path, tmp_path, SPECULATOR_KEYS + POOL_KEYS)
    default = _coordinator(_family_launch(tmp_path / "bare", bare))
    # Separate sandbox roots now remain visible at their original container paths.
    assert _speculation(keep.replace(str(tmp_path / "example"), "<root>")) == _speculation(
        default.replace(str(tmp_path / "bare"), "<root>"))
    assert _speculation(keep), keep
    assert re.findall(r"--pool-tokens \S+", keep) == re.findall(r"--pool-tokens \S+", default)


@pytest.mark.parametrize("path", V41, ids=lambda p: p.name)
def test_v41_starting_config_matches_the_launcher_defaults(path: Path, tmp_path: Path) -> None:
    """V4.1 runs through the release loader (run-family.sh's V4.1 section): its speculator key matches the loader
    default, and POOL_TOKENS is the one named exception (see POOL_EXCEPTIONS)."""
    values = keys(path)
    result = subprocess.run(
        ["bash", "-euc", 'source scripts/lib/release-common.sh; release_load_config "$1"; '
         'printf "%s=%s\\n" SPECULATOR "$SPECULATOR" POOL_TOKENS "$POOL_TOKENS"',
         "test", str(_strip(path, tmp_path, SPECULATOR_KEYS + POOL_KEYS))],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    defaults = dict(line.split("=", 1) for line in result.stdout.splitlines())
    for key in SPECULATOR_KEYS + POOL_KEYS:
        if key not in values:
            continue
        if ("deepseek_v41", key) in POOL_EXCEPTIONS:
            assert values[key] == POOL_EXCEPTIONS[("deepseek_v41", key)]
            assert defaults[key] == "", "V4.1 now defaults POOL_TOKENS; drop the exception"
        else:
            assert values[key] == defaults[key], key


def test_tp_ep_examples_are_left_alone() -> None:
    assert TP_EP <= {p.name for p in EXAMPLES.glob("*.config")}


def test_qualification_configs_exist_and_are_listed() -> None:
    assert QUALIFICATION <= {p.name for p in EXAMPLES.glob("*.config")}
    readme = (EXAMPLES / "README.md").read_text()
    for name in QUALIFICATION:
        assert f"({name})" in readme


@pytest.mark.parametrize("config,count", [
    ({"model_type": "deepseek_v41", "text_config": {"num_hidden_layers": 40}}, 40),
    ({"model_type": "deepseek_v4", "n_layers": 61}, 61),
    ({"model_type": "qwen4_exp", "text_config": {"num_hidden_layers": 48}}, 48),
    ({"model_type": "glm_moe_dsa", "num_hidden_layers": 78, "first_k_dense_replace": 3}, 75),
    ({"model_type": "glm5_next", "num_hidden_layers": 4,
      "layer_types": ["linear_attention"] * 4,
      "mlp_layer_types": ["sparse", "dense", "sparse", "dense"]}, 2),
    ({"model_type": "mimo_v2", "num_hidden_layers": 5, "moe_layer_freq": [0, 1, 0, 1, 1]}, 3),
    ({"model_type": "mimo_v2_flash", "num_hidden_layers": 48}, 47),
])
@pytest.mark.parametrize("value", ["auto", "max", "all", "50%", "100%", "0"])
def test_checkpoint_onboarding_count_and_rounding(tmp_path, config, count, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"),
                             "--onboard", value, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    expected = {"all": str(count), "100%": str(count), "50%": str(int(count * 0.5 + 0.5))}.get(value, value)
    assert result.stdout.strip() == expected
    bounded = subprocess.run(["bash", "-c", 'source "$1"; release_onboard "$2" test/model pinned',
                              "test", str(ROOT / "scripts/lib/release-common.sh"), value],
                             env={**os.environ, "HF_HOME": str(tmp_path / "hf")},
                             capture_output=True, text=True)
    # With no cached checkpoint, validate grammar without inventing a family limit.
    assert bounded.returncode == 0, bounded.stderr
    assert bounded.stdout.strip() == value


@pytest.mark.parametrize("value", ["-1", "101%", "NaN%", "inf%", "half", "5x", ""])
def test_checkpoint_onboarding_rejects_invalid_grammar(value):
    result = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"),
                             "--onboard", value], capture_output=True, text=True)
    assert result.returncode != 0
    assert "RTX_EXPERT_LAYERS must be" in result.stderr


@pytest.mark.parametrize("name", sorted(V4_PREP))
def test_v4_candidate_config_contract(name: str, tmp_path: Path) -> None:
    path = EXAMPLES / name
    values = keys(path)
    text = path.read_text()
    assert f"./run.sh --config examples/configs/{name}" in text and "--dry-run" in text
    assert re.fullmatch("[0-9a-f]{40}", values["MODEL_REVISION"])
    assert not any(needle in text.lower() for needle in PRIVATE)
    assert "EXLR8_DENSE_FROM" not in values
    if name.startswith("glm53"):
        assert values["EXLR8_MANIFEST"] == "up1"
        assert values["MODEL_ID"] == "kindlingai/glm-5.3-exlr8-k2-k3-k4"
        assert values["RTX_EXPERT_LAYERS"] == "0"
        assert values["SPARK_COUNT"] == ("4" if "-min" in name else "6")
        assert values["ATTENTION_PLACEMENT"] == ("heads" if "-min" in name else "context")
        # Native-dense launcher integration is owned by #659/#663, not this prep.
        # The smoke-entry loader test above still validates every config key.
    else:
        assert values["SPARK_COUNT"] == "0" and values["RTX_GPUS"] == "1"
        assert values["SPECULATOR"] == "dspark"
        from test_run_family_dispatch import _family_launch_result, _snapshot
        _snapshot(tmp_path / "hf", "RadixArk/Qwen3.8-27B-DSpark",
                  {"speculators_model_type": "dspark"}, "85ef153be924f17ce4bf62726954eeaa4a73e854")
        launch_values = {k: v for k, v in values.items() if k not in ("MODEL_ID", "MODEL_REVISION")}
        result = _family_launch_result(tmp_path, CHECKPOINTS[values["MODEL_ID"]], values["MODEL_ID"],
            "".join(f"{k}={v}\n" for k, v in launch_values.items()),
            extra_args=("--wip", "dense-test"), encoder_plan=LOCAL_PLAN)
        launch = _coordinator(result)
        assert "serve-qwen35" in launch and "--draft " in launch
        assert "--peers " not in launch


def test_v4_matrix_card_set_and_retired_pins() -> None:
    v3 = json.loads((ROOT / "scripts/bench/release-smoke-v3.json").read_text())
    v4 = json.loads((ROOT / "scripts/bench/release-smoke-v4.json").read_text())
    entries = {entry["name"]: entry for entry in v4["entries"]}
    assert len(entries) == len(v4["entries"])
    for entry in v3["entries"]:
        if entry["name"].endswith("-5090") and entry["name"] != "qwen38-nvfp4-5090":
            assert entries[entry["name"]] == entry
    assert entries["qwen38-exl3-5090"]["set"]["SPARK_COUNT"] == "1"
    for name, gpus, sparks in [
        ("v41-flash-max", 2, 6), ("v41-flash-reference-tp4", 1, 4),
        ("v4-flash-reference-tp4", 1, 4), ("glm53-exl3-max", 2, 6),
        ("glm53-exlr8-min", 1, 4), ("glm53-exlr8-max", 2, 6),
        ("kimi-k3-max", 2, 6), ("step5-fp8-max", 2, 6),
        ("step5-nvfp4-min", 1, 4), ("qwen38-27b-bf16-min", 1, 0),
        ("qwen38-27b-fp8-min", 1, 0),
    ]:
        assert len(entries[name]["gpus"]) == gpus
        assert len(entries[name]["sparks"]) == sparks
        assert entries[name]["set"]["SPARK_COUNT"] == str(sparks)
    assert not any(name.startswith("qwen38-nvfp4") for name in entries)
    retired = v4["retired"]
    assert len(retired) == 14
    assert "glm53-exl3-max" not in {entry["name"] for entry in retired}
    assert {"qwen38-nvfp4-" + size for size in ("min", "max", "5090")} <= {e["name"] for e in retired}
    assert retired == json.loads((ROOT / "benchmarks/current-matrix.json").read_text())["retired"]
    reports = [(path, json.loads(path.read_text())) for path in (ROOT / "benchmarks").glob("*/*/report.json")]
    for entry in retired:
        assert (ROOT / entry["config"]).is_file()
        report = json.loads((ROOT / entry["report"] / "report.json").read_text())
        assert report["profile"] == "smoke" and report["baseline"]
        build = report["server"]["build"]
        expected = build["release"] or f"v3 dev {report['created'][:10]}"
        assert entry["release"] == expected and entry["release_note"]
        assert "nearest" not in entry["release_note"].lower()
        if entry["name"] == "glm53f-exl3-v1":
            candidates = [d for _, d in reports if d["profile"] == "smoke" and
                          d["server"]["model"] == "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1"]
        else:
            candidates = [d for p, d in reports if d["profile"] == "smoke" and p.parent.name.endswith("-" + entry["name"])]
        assert report["created"] == max(d["created"] for d in candidates)
