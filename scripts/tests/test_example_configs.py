"""The per-family starting configs in examples/configs/ (current serving recipes).

Each one parses through the real launcher's dry-run or argv path (CPU only:
docker, ssh, nvidia-smi and nest are stubs that print their argv), and its
speculator and pool keys match the launcher's bare default for that family, so
a user who drops them gets the same launch and the two cannot drift apart.
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "configs"
STARTING = re.compile(r"^(?P<model>[a-z0-9.-]+)-(?P<profile>min|max)(?:-(?:fp8|nvfp4)-reply)?\.config$")
# WIP qualification configs for families without a release card yet (#273, #274):
# served from ./wip.sh slots only, checked by test_serve_launch_new_families.py.
ML4_QUALIFICATION = {f"mistral-large4-exl3-k4-{size}.config" for size in ("min", "max")}
NEW_PUBLICATIONS = {f"{model}-{size}.config" for model in
                    ("mistral-large4-exl3-k4", "glm53-flash-exl3-k4-mc",
                     "glm53-exl3-k4-mc", "step5-exl3-k4-mc") for size in ("min", "max")}
QUALIFICATION = {"minimax-m31-flash-nvfp4-min.config", "kimi-k3-exl3-max.config", "kimi-k3-exl3-k6-min.config", "kimi-k3-exl3-k6-max.config", "step5-fp8-max.config", "step5-nvfp4-min.config",
                 "muse-spark-exl3-min.config", "muse-spark-exl3-max.config",
                 "step5-exl3-k6core-min.config", "step5-exl3-k6core-max.config"} | ML4_QUALIFICATION | NEW_PUBLICATIONS
V4_PREP = {"qwen35-9b-bf16-min.config", "glm53-exlr8-min.config", "glm53-exlr8-max.config",
           "qwen38-27b-bf16-min.config", "qwen38-27b-fp8-min.config",
           "glm53f-exl3-k4-core-k6-min.config", "glm53f-exl3-k4-core-k6-max.config"}
FP8_EXAMPLES = {"glm53-flash-exl3-k4-mc-fp8-max.config"}
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
    "wrldsuksgo2mars/DeepSeek-V4.1-Flash-1017-StridedPLE": {"model_type": "deepseek_v41"},
    "nvidia/DeepSeek-V4.1-Flash-NVFP4": {"model_type": "deepseek_v41",
                                         "quantization_config": {"moe_quant_algo": "NVFP4"}},
    "deepseek-ai/DeepSeek-V4-Flash-0731": DSV4,
    "wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1": {**DSV4, "quantization_config": {"quant_method": "exl3"}},
    "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1": GLM5,
    "wrldsuksgo2mars/GLM-5.3-EXL3-K4-Core-K6": {**GLM5, "quantization_config": {
        "quant_method": "exl3", "bits": 4.0, "core_bits": 6.0, "head_bits": 6.0, "mtp_bits": 4.0}},
    "nvidia/GLM-5.3-NVFP4": GLM5,
    "zai-org/GLM-5.3-Flash": GLMF,
    "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8": GLMF,
    "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-Core-K6": {**GLMF, "quantization_config": {
        "quant_method": "exl3", "bits": 4.0, "core_bits": 6.0, "head_bits": 6.0, "mtp_bits": 4.0}},
    "nvidia/GLM-5.3-Flash-NVFP4": GLMF,
    "XiaomiMiMo/MiMo-V2.6-Flash-MOPD": {**MIMO, "hidden_size": 4096},
    "XiaomiMiMo/MiMo-V2.6-Pro-MOPD": {**MIMO, "hidden_size": 6144},
    "wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1": {**QWEN, "quantization_config": {"quant_method": "exl3"}},
    "nvidia/Qwen3.8-Flash-Next-NVFP4": {**QWEN, "quantization_config": {"quant_method": "modelopt", **NVFP4_GROUPS}},
    "kindlingai/glm-5.3-exlr8-k2-k3-k4": GLM5,
    "kindlingai/glm-5.3-flash-exlr8-k2-k3-k4": GLMF,
    "Qwen/Qwen3.5-9B": {"model_type": "qwen3_5", "text_config": {
        "hidden_size": 4096, "num_hidden_layers": 32,
        "layer_types": ["linear_attention"] * 31 + ["full_attention"]}},
    "Qwen/Qwen3.8-27B": {"model_type": "qwen3_5", "text_config": {
        "hidden_size": 5120, "num_hidden_layers": 32,
        "layer_types": ["linear_attention"] * 31 + ["full_attention"]}},
    "Qwen/Qwen3.8-27B-FP8": {"model_type": "qwen3_5", "text_config": {
        "hidden_size": 5120, "num_hidden_layers": 32,
        "layer_types": ["linear_attention"] * 31 + ["full_attention"]}},
    "RadixArk/Qwen3.8-27B-DSpark": {"speculators_model_type": "dspark"},
    "stepfun-ai/Step-5-Preview": {"model_type": "step3p5", "num_hidden_layers": 8},
    "stepfun-ai/Step-5-Preview-NVFP4": {"model_type": "step3p5", "num_hidden_layers": 8},
    "wrldsuksgo2mars/Step-5-Preview-EXL3-K4-Core-K6": {"model_type": "step3p5", "num_hidden_layers": 8},
    "wrldsuksgo2mars/Kimi-K3-EXL3-K2-Core-K6": {"model_type": "kimi_k3", "num_hidden_layers": 8},
    "wrldsuksgo2mars/Mistral-Large-4-EXL3-K4": {"model_type": "mistral_large4", "text_config": {
        "model_type": "mistral_large4_text", "num_hidden_layers": 4, "first_k_dense_replace": 1}},
    # Drafters the starting configs name.
    "incoai/GLM-5.3-DFlash2": {"architectures": ["DFlash2DraftModel"]},
    "incoai/GLM-5.3-Flash-DFlash2": {"architectures": ["DFlash2DraftModel"]},
}
FAMILY_DOCS = {"deepseek_v41": "deepseek_v41", "deepseek_v4": "deepseek_v4", "glm5": "glm5",
               "glm5_flash": "glm5_flash", "mimo_v2": "mimo_v2", "qwen4": "qwen4",
               "kimi_k3": "kimi_k3", "qwen3_5": "qwen3_5", "step5": "step5"}
# Launcher keys that select the speculator and the KV pool.
SPECULATOR_KEYS = ("SPECULATOR", "SPECULATOR_MODEL_ID", "SPECULATOR_DEPTH")
POOL_KEYS = ("POOL_TOKENS",)
# The one named exception (decided 2026-10-10): the V4.1 cards ran with
# POOL_TOKENS=auto, so the V4.1 examples pin it, while the V4.1 loader leaves the key
# empty when absent (run-family.sh's V4.1 section then resolves auto). Roadmap step S5 moves V4.1 onto
# run-family.sh, where the defaults are unified and this exception goes away.
POOL_EXCEPTIONS = {("deepseek_v41", "POOL_TOKENS"): "auto"}


def starting_configs() -> list[Path]:
    return sorted(p for p in EXAMPLES.glob("*.config") if p.name not in QUALIFICATION | V4_PREP | FP8_EXAMPLES)


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
        (root / "refs" / "main").write_text("a" * 40)
        (root / "snapshots" / "abc").mkdir(parents=True)
        (root / "snapshots" / ("a" * 40)).symlink_to(root / "snapshots" / "abc")
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
    (bin_dir / "curl").chmod(0o755)
    from launcher_sandbox import fake_nvidia_smi, stub_cuda_python
    fake_nvidia_smi(bin_dir)
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
if family == "glm5_flash":
    # Launch coverage uses a whole-drafter plan, not a synthetic memory-fit decision.
    report["tp2_dflash"] = value("--tp2-dflash", "false") == "true"
    requested = value("--replay-records", "own")
    report["replay_records"] = ("shared" if int(value("--rtx", "1")) == 1
        and int(value("--pool-tokens", "0")) == 0 and report["spark_ranks"] > 0 else "own") if requested == "auto" else requested
selected = value("--speculator", "off")
if family == "qwen4" and selected == "auto":
    # Both Qwen4 starting checkpoints qualify; dense Qwen is a separate family.
    selected = "mtp" if report["spark_ranks"] == 0 else "off"
print(json.dumps(dict(report, spark_expert_world=report["spark_ranks"],
    spark_roles=[dict(rank=i, experts=True, layers=[], encoder=report.get("encoder") if i == 0 else None,
                      audio_encoder=None, drafter=None) for i in range(report["spark_ranks"])], version=1, family=family, executable=True, report=report,
    launch=dict(family=family, first_expert_layer=first, last_expert_layer=last),
    spark_admission="coordinator_authoritative", decode_lanes=1,
    peers=value("--deployment-peers", "").split(",")[:report["spark_ranks"]],
    speculator=selected, prefill_rows=int(value("--prefill-rows", "4096")) or 4096,
    # Match DeviceLayout's allocation ceiling, independently of host-inclusive capacity.
    memory_layout=dict(devices=[dict(kind="rtx", index=i, items=[dict(category="weights", bytes=8 << 30)])
                                for i in range(int(value("--rtx", "1")))] +
                               [dict(kind="spark", index=i, capacity_bytes=int(value("--spark-memory-cap", str(108 << 30))),
                                     worker_budget_bytes=int(value("--spark-memory-cap", str(108 << 30))),
                                     items=[], kv_tokens=0) for i in range(report["spark_ranks"])]))))
""".replace("REPORT", repr(json.dumps(plan))).replace("FAMILY", repr(family))
            .replace("FIRST", first).replace("LAST", "None" if last == "-1" else last))
        docker.write_text(docker.read_text() + f'case "$*" in *"cuteafd plan"*) exec python3 "{stub}" "$@" ;; esac\n')
    from launcher_sandbox import fake_gpus
    env = {**os.environ, "HF_HOME": str(hf), "HOME": str(tmp_path / "home"), "FAKE_GPUS": json.dumps(fake_gpus(2)),
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
              "prefill_reserve_rows": 8192,
              "encoder": {"kind": {"kind": "spark", "rank": 0}, "replicas": []},
              "audio_encoder": {"kind": {"kind": "off"}, "replicas": []}}
LOCAL_PLAN = {**SPARK_PLAN, "spark_ranks": 0, "encoder": {"kind": {"kind": "off"}, "replicas": []}}


def test_one_starting_config_per_release_card() -> None:
    names = {p.name for p in starting_configs()}
    expected = {f"{model}-{profile}.config" for model in (
        "deepseek-v41-mxfp4", "v41-flash-strided", "deepseek-v4-flash-mxfp4", "deepseek-v4-pro-exl3",
        "glm53-exl3", "glm53-exl3-k6core", "glm53-nvfp4", "glm53-flash-fp8", "glm53-flash-exl3", "glm53-flash-exl3-k6", "glm53-flash-nvfp4",
        "glm53-flash-exlr8-k4", "mimo-v26-flash-mxfp4", "mimo-v26-pro-mxfp4") for profile in ("min", "max")}
    # Qwen fits one RTX: the 2x RTX column is n/a, so only min exists.
    expected |= {"qwen38-exl3-min.config", "qwen38-nvfp4-min.config"}
    expected |= {f"deepseek-v41-mxfp4-max-{codec}-reply.config" for codec in ("fp8", "nvfp4")}
    assert names == expected
    for path in starting_configs():
        assert STARTING.match(path.name), path.name


def test_release_smoke_matrix_covers_owned_columns_with_current_defaults() -> None:
    matrix = json.loads((ROOT / "scripts/bench/release-smoke.example.json").read_text())
    assert matrix["profile"] == "smoke"
    entries = matrix["entries"]
    assert len(entries) == 28
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
        assert values["COORDINATOR_GPUS"] == ",".join(map(str, entry["gpus"]))
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
        if entry["name"] in {"glm53f-nvfp4-max", "qwen38-nvfp4-min"}:
            assert entry["sparks"] == ["ostrich", "dodo"]
            assert values["EXPERT_BACKEND"] == "spark"
        for key in ("ADDR", "EXPERT_PORT"):
            assert values[key] not in ports
            ports.add(values[key])
        models.setdefault(values["MODEL_ID"], set()).add(len(entry["gpus"]))
    assert len(models) == 13
    assert all(columns == {1, 2} for columns in models.values())


def test_glm_k6_core_cards_preserve_native_checkpoint_and_topology() -> None:
    matrix = json.loads((ROOT / "scripts/bench/release-smoke.example.json").read_text())
    entries = {entry["name"]: entry for entry in matrix["entries"]}
    for size, gpus, sparks in (("min", [1], 4), ("max", [0, 1], 6)):
        name = f"glm53-exl3-k6core-{size}"
        entry = entries[name]
        values = keys(ROOT / entry["config"])
        assert values["MODEL_ID"] == "wrldsuksgo2mars/GLM-5.3-EXL3-K4-Core-K6"
        assert values["MODEL_REVISION"] == "5a4e83a09a11f3145332ed632d28a9894ea88225"
        assert values["SPECULATOR"] == "dflash2"
        assert values["SPECULATOR_MODEL_ID"] == "incoai/GLM-5.3-DFlash2"
        assert values["POOL_TOKENS"] == "auto"
        assert "RTX_EXPERT_LAYERS" not in values
        assert entry["gpus"] == gpus and len(entry["sparks"]) == sparks
        baseline = entries[f"glm53-exl3-{size}"]
        assert entry["sparks"] == baseline["sparks"]
        assert entry["family"] == baseline["family"] == "glm5"


def test_v4_flash_max_keeps_all_experts_on_the_rtx_pair() -> None:
    config = keys(EXAMPLES / "deepseek-v4-flash-mxfp4-max.config")
    assert config["RTX_GPUS"] == "2" and config["SPARK_COUNT"] == "0"
    assert config["EXPERT_BACKEND"] == "auto"
    assert not any(re.fullmatch(r"SPARK_\d+_(HOST|LANE_A|LANE_B)", key) for key in config)
    for path in ("scripts/bench/release-smoke.example.json", "scripts/bench/kits/gate/matrix-gate.json"):
        matrix = json.loads((ROOT / path).read_text())
        entry = next(entry for entry in matrix["entries"] if entry["name"] == "v4-flash-max")
        assert entry["gpus"] == [0, 1] and entry["sparks"] == []
        assert entry["set"]["SPARK_COUNT"] == "0"
        assert not any(re.fullmatch(r"SPARK_\d+_(HOST|LANE_A|LANE_B)", key) for key in entry["set"])



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
    assert len(entries) == len(matrix["entries"]) == 22
    columns = {}
    ports = set()
    for name, entry in entries.items():
        values = {**keys(ROOT / entry["config"]), **defaults, **entry["set"]}
        assert values["MODEL_ID"] in CHECKPOINTS
        assert entry["family"] in FAMILY_DOCS
        assert "BENCH_SIMULATED" not in values
        assert int(values["RTX_GPUS"]) == len(entry["gpus"])
        assert values["COORDINATOR_GPUS"] == ",".join(map(str, entry["gpus"]))
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
    assert len(columns) == 11
    assert {model: sorted(sides) for model, sides in columns.items() if len(sides) == 2} == {
        values_of(entries["v41-flash-min"])["MODEL_ID"]: ["max", "min"],
        values_of(entries["glm53f-exl3-min"])["MODEL_ID"]: ["max", "min"]}
    assert len([name for name in entries if name.endswith("-5090")]) == 7
    assert len(entries["v41-flash-reference-tp4"]["sparks"]) == 4
    assert len(entries["v41-flash-reference-tp6"]["sparks"]) == 6
    for name in ("glm53f-exl3-max", "qwen38-exl3-min"):
        assert not entries[name]["sparks"]
        assert entries[name]["set"]["EXPERT_BACKEND"] == "local"
    assert not any("reference-all-spark" in name or "reference-half" in name for name in entries)
    text = json.dumps(matrix)
    assert not any(private in text for private in PRIVATE)

    retired = {entry["name"]: entry for entry in matrix["retired"]}
    assert set(retired) == {"glm53f-tr3-min", "glm53f-tr3-5090"}
    for entry in retired.values():
        assert not (ROOT / entry["config"]).exists()
        report = json.loads((ROOT / entry["report"] / "report.json").read_text())
        assert entry["release"] == report["server"]["build"]["release"] == "v3.0.0-rc4"


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
    readme = (ROOT / "README.md").read_text()
    results = readme.split("<!-- results:begin -->", 1)[1].split("<!-- results:end -->", 1)[0]
    identities = set()
    assert matrix["cells"]
    for row in matrix["cells"]:
        identity = row["family"], row["checkpoint"]
        assert all(identity) and identity not in identities
        identities.add(identity)
        assert set(row) == {"family", "checkpoint", "5090", "1rtx", "2rtx"}
        current = []
        for column in ("5090", "1rtx", "2rtx"):
            cell = row[column]
            assert cell["status"] in {"current", "held", "pending", "na"}
            if cell["status"] != "current":
                assert cell.get("reason", "").strip() and "report" not in cell
                continue
            assert not cell.get("reason")
            directory = Path(cell["report"])
            assert not directory.is_absolute() and ".." not in directory.parts
            assert len(directory.parts) == 3 and directory.parts[:2] == ("benchmarks", row["family"])
            report = json.loads((ROOT / directory / "report.json").read_text())
            server = report["server"]
            snapshot = server.get("configuration", {}).get("snapshot") or ""
            repo = next((part.removeprefix("models--").replace("--", "/")
                         for part in Path(snapshot).parts if part.startswith("models--")), server["model"])
            assert (server["family"], repo) == identity
            assert report["profile"] in {"smoke", "share"}
            settings = {s["name"]: s.get("value") for s in server["configuration"].get("settings", [])}
            simulated = settings.get("simulated") == "5090"
            small_column = simulated or settings.get("hardware.class") == "5090"
            gpus = server["hardware"]["gpus"]
            used = [gpu for gpu in gpus if gpu.get("used")] or gpus
            if column == "5090":
                assert small_column and len(used) == 1
                assert simulated or all("RTX 5090" in gpu["name"] for gpu in used)
            else:
                assert not small_column and len(used) == int(column[0])
            assert report.get("baseline") or column == "5090" and settings.get("qualification.no-fit")
            assert f'href="{directory.as_posix()}/card.svg"' in results
            assert (ROOT / directory / "card.svg").is_file()
            current.append(cell)
        if not current:
            assert row["checkpoint"] in results, "Unmeasured checkpoints remain visible"
            for column in ("5090", "1rtx", "2rtx"):
                assert html.escape(row[column]["reason"]).replace("&#x27;", "&#39;") in results
    retired_names = set()
    for entry in matrix["retired"]:
        assert entry["name"] and entry["name"] not in retired_names
        retired_names.add(entry["name"])
        assert_archived_config(entry)
        report = json.loads((ROOT / entry["report"] / "report.json").read_text())
        assert report["profile"] == "smoke" and report.get("baseline")
        assert_retired_release(entry, report)
        row = next(line for line in results.splitlines() if f'href="{entry["report"]}/card.svg"' in line)
        assert f'>{entry["release"]}</td>' in row
        build = report["server"]["build"]
        label = build["release"] or build["commit"][:12] + ("+dirty" if build.get("dirty") else "")
        assert f'build {label}; commit {build["commit"]};' in row
        assert f'href="{entry["report"]}/card.svg"' in results
    assert "simulated-5090" not in json.dumps(matrix)


def assert_archived_config(entry: dict) -> None:
    path = ROOT / entry["config"]
    if path.name in {"glm53-flash-tr3-min.config", "glm53-flash-tr3-max.config"}:
        assert not path.exists(), "Retired checkpoint recipes must not remain launchable"
    else:
        assert path.is_file()


def assert_retired_release(entry: dict, report: dict) -> None:
    label = entry["release"]
    build = report["server"]["build"]
    final = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", label)
    if final:
        tag = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", f"refs/tags/{label}^{{commit}}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        actual = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)(?:-rc\d+)?", build["release"])
        assert actual and actual.group(1, 2) == final.group(1, 2)
        assert tag == build["commit"] or re.fullmatch(re.escape(label) + r"-rc\d+", build["release"])
    else:
        assert re.fullmatch(r"v\d+ dev \d{4}-\d{2}-\d{2}", label)
        assert not build["release"] and label.endswith(report["created"][:10])
    assert entry["release_note"].strip()


def test_v5_smoke_card_inventory() -> None:
    smoke = json.loads((ROOT / "scripts/bench/release-smoke-v5.json").read_text())
    names, ports, cards = set(), set(), set()
    for entry in smoke["entries"]:
        name = entry["name"]
        assert name not in names
        names.add(name)
        assert_archived_config(entry)
        values = {**keys(ROOT / entry["config"]), **smoke["defaults"]["set"], **entry["set"]}
        assert re.fullmatch(r"[a-f0-9]{40}", values["MODEL_REVISION"])
        family = entry["family"]
        if family == "deepseek_v41":
            assert "EXPERT_BACKEND" not in values
        else:
            assert not any(re.fullmatch(r"SPARK_\d+_LANE_B", key) for key in values)
        if family == "qwen3_5":
            assert "EXPERT_PORT" not in values
        card_ports = [values["ADDR"].rsplit(":", 1)[1]]
        if "EXPERT_PORT" in values:
            card_ports.append(values["EXPERT_PORT"])
        for port in card_ports:
            assert port not in ports
            ports.add(port)
        assert len(entry["gpus"]) == int(values["RTX_GPUS"])
        assert values["COORDINATOR_GPUS"] == ",".join(map(str, entry["gpus"]))
        assert len(entry["sparks"]) == int(values["SPARK_COUNT"])
        for rank, host in enumerate(entry["sparks"]):
            assert host == values[f"SPARK_{rank}_HOST"] == f"REPLACE-ME-spark-{rank}"
        assert "nvfp4" not in name and "nospec" not in name and "reference" not in name
        assert entry.get("baseline", True), "All cards retain baseline quick quality"
        assert values.get("CORE_VIEW", "original") == "original", "Release cards use the checkpoint's default core"
        column = "5090" if name.endswith("-5090") else f"{len(entry['gpus'])}rtx"
        identity = (entry["family"], values["MODEL_ID"], column)
        assert identity not in cards
        cards.add(identity)
        if column == "5090":
            assert values["GPU_MEMORY_CAP"] == "31.8GiB"
            assert values["BENCH_HARDWARE_CLASS"] == values["BENCH_SIMULATED"] == "5090"
            assert len(entry["sparks"]) == 4
            assert entry["precheck"] == {
                "argv": ["sh", "-c", "nvidia-smi --query-gpu=name --format=csv,noheader --id=1 | grep -q 'RTX PRO 6000'"],
                "timeout_s": 30,
                "required": True,
            }
    assert len(cards) == 37
    assert len({(family, model) for family, model, _ in cards}) == 19
    assert sum(column == "5090" for _, _, column in cards) == 7
    assert not names.intersection(entry["name"] for entry in smoke["retired"])


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


V41 = [p for p in starting_configs() if CHECKPOINTS[keys(p)["MODEL_ID"]]["model_type"] == "deepseek_v41"]
FAMILY = [p for p in starting_configs() if p not in V41]


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


@pytest.mark.parametrize("name", sorted(ML4_QUALIFICATION))
def test_ml4_qualification_config_preserves_native_draft_defaults(name: str, tmp_path: Path) -> None:
    from test_run_family_dispatch import _family_launch_result

    path = EXAMPLES / name
    values = keys(path)
    minimum = name.endswith("-min.config")
    assert values["MODEL_ID"] == "wrldsuksgo2mars/Mistral-Large-4-EXL3-K4"
    assert values["MODEL_REVISION"] == "9c354cee78cd6feddb057549b8edde6e3b319822"
    assert values["SPECULATOR"] == "dspark" and values["POOL_TOKENS"] == "auto"
    assert not {"SPECULATOR_MODEL_ID", "SPECULATOR_DEPTH", "RTX_EXPERT_LAYERS"}.intersection(values)
    assert values["SPARK_COUNT"] == ("5" if minimum else "6")
    assert values["RTX_GPUS"] == ("1" if minimum else "2")
    assert f"./run.sh --config examples/configs/{name} --dry-run" in path.read_text()
    assert not any(needle in path.read_text().lower() for needle in PRIVATE)
    for rank in range(int(values["SPARK_COUNT"])):
        assert values[f"SPARK_{rank}_HOST"] == f"REPLACE-ME-spark-{rank}"
        assert values[f"SPARK_{rank}_LANE_A"] == f"192.0.2.{rank + 1}"
    checkpoint = {"model_type": "mistral_large4", "text_config": {
        "model_type": "mistral_large4_text", "num_hidden_layers": 4, "first_k_dense_replace": 1}}
    launches = []
    for label, drop in (("example", ()), ("bare", SPECULATOR_KEYS + POOL_KEYS)):
        config = "".join(f"{k}={v}\n" for k, v in values.items()
                         if k not in ("MODEL_ID", "MODEL_REVISION", *drop))
        result = _family_launch_result(tmp_path / label, checkpoint, values["MODEL_ID"], config,
                                      extra_args=("--wip", "ml4-test"))
        launch = _coordinator(result)
        assert "serve-glm" in launch and "--dspark" in launch and "--draft " not in launch
        assert "--pool-tokens 0" in launch
        assert result.stderr.count("cuteafd expertd-native") == int(values["SPARK_COUNT"])
        launches.append(_speculation(launch))
    assert launches[0] == launches[1]


@pytest.mark.parametrize("name", sorted(NEW_PUBLICATIONS))
def test_new_publications_preserve_pinned_recipe_layouts(name: str) -> None:
    path = EXAMPLES / name
    values = keys(path)
    ml4 = name.startswith("mistral-")
    if ml4:
        expected = dict(values)
        minimum = name.endswith("-min.config")
        assert values["SPARK_COUNT"] == ("5" if minimum else "6")
        assert values["RTX_GPUS"] == ("1" if minimum else "2")
        assert values["SPECULATOR"] == "dspark" and values["POOL_TOKENS"] == "auto"
    elif name.startswith("glm53-exl3-k4-mc-"):
        expected = keys(EXAMPLES / name.replace("exl3-k4-mc-", "exl3-"))
    elif name.startswith("step5-exl3-k4-mc-"):
        expected = keys(EXAMPLES / name.replace("exl3-k4-mc-", "exl3-k6core-"))
    else:
        expected = keys(EXAMPLES / name.replace("exl3-k4-mc-", "exl3-k6-"))
    publication = next((model, revision, doc) for stem, model, revision, doc in [
        ("mistral-", "Mistral-Large-4-EXL3-K4", "9c354cee78cd6feddb057549b8edde6e3b319822", "mistral_large4"),
        ("glm53-flash-", "GLM-5.3-Flash-EXL3-K4-MC", "91a8dbc9c12e790c65aa1a38c6aede6a3e23a77d", "glm5_flash"),
        ("glm53-exl3-k4-mc-", "GLM-5.3-EXL3-K4-MC", "c39486376dee84e5faa4904e014e07d7b6e76405", "glm5"),
        ("step5-exl3-k4-mc-", "Step-5-Preview-EXL3-K4-MC", "4483a4bd874e8c360aec486ea72bd31d1ad7a247", "step5"),
    ] if name.startswith(stem))
    model, revision, family_doc = publication
    expected.update(MODEL_ID="wrldsuksgo2mars/" + model, MODEL_REVISION=revision,
                    CORE_VIEW="original")
    assert values == expected
    assert not any(needle in path.read_text().lower() for needle in PRIVATE)
    assert re.search(rf"\./run\.sh --config examples/configs/{re.escape(name)} (?:--wip SLOT )?--dry-run", path.read_text())
    doc = ROOT / "docs/models" / (family_doc + ".md")
    assert f"../../examples/configs/{name}" in doc.read_text()
    assert f"({name})" in (EXAMPLES / "README.md").read_text()
    if not ml4:
        assert all(view in path.read_text() for view in ("k8-compact", "k6-compact"))
    matrix = json.loads((ROOT / "benchmarks/current-matrix.json").read_text())
    assert name not in json.dumps(matrix), "Unbenchmarked publications are not release cards"


def test_ml4_smoke_entries_use_published_examples_and_minimum_sparks() -> None:
    matrix = json.loads((ROOT / "scripts/bench/kits/gate/matrix-ml4.json").read_text())
    entries = {entry["name"]: entry for entry in matrix["entries"]}
    for size, gpus, sparks in (("min", [1], 5), ("max", [0, 1], 6)):
        entry = entries[f"ml4-exl3-{size}"]
        assert entry["config"] == f"examples/configs/mistral-large4-exl3-k4-{size}.config"
        assert entry["family"] == "mistral_large4" and entry["gpus"] == gpus
        assert len(entry["sparks"]) == sparks and entry["set"]["SPARK_COUNT"] == str(sparks)
        assert not any(k.startswith(f"SPARK_{sparks}_") for k in entry["set"])
        assert keys(ROOT / entry["config"])["SPECULATOR"] == "dspark"


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
@pytest.mark.parametrize("value", ["auto", "all", "0"])
def test_checkpoint_onboarding_accepts_auto_all_and_counts(tmp_path, config, count, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"),
                             "--onboard", value, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # `all` stays unresolved; the engine resolves it from the checkpoint itself.
    assert result.stdout.strip() == value
    # The checkpoint's own routed-layer count is a valid exact pin; one more is
    # refused with that count.
    exact = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"),
                            "--onboard", str(count), str(path)], capture_output=True, text=True)
    assert exact.returncode == 0, exact.stderr
    assert exact.stdout.strip() == str(count)
    over = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"),
                           "--onboard", str(count + 1), str(path)], capture_output=True, text=True)
    assert over.returncode != 0
    assert f"checkpoint has {count} routed layers" in over.stderr
    bounded = subprocess.run(["bash", "-c", 'source "$1"; release_onboard "$2" test/model pinned',
                              "test", str(ROOT / "scripts/lib/release-common.sh"), value],
                             env={**os.environ, "HF_HOME": str(tmp_path / "hf")},
                             capture_output=True, text=True)
    # With no cached checkpoint, validate grammar without inventing a family limit.
    assert bounded.returncode == 0, bounded.stderr
    assert bounded.stdout.strip() == value


@pytest.mark.parametrize("value", ["-1", "max", "50%", "101%", "NaN%", "inf%", "half", "5x", ""])
def test_checkpoint_onboarding_rejects_invalid_grammar(value):
    result = subprocess.run(["python3", str(ROOT / "scripts/lib/checkpoint-family.py"),
                             "--onboard", value], capture_output=True, text=True)
    assert result.returncode != 0
    assert "RTX_EXPERT_LAYERS must be" in result.stderr


@pytest.mark.parametrize("name", sorted(FP8_EXAMPLES))
def test_mc_fp8_example_declares_only_activation_extras(name: str, tmp_path: Path) -> None:
    values = keys(EXAMPLES / name)
    assert values["CORE_VIEW"] == "fp8-compact"
    # MLA/FFN W8A8 are the default; the example opts KDA in as well.
    assert values["FP8_PREFILL"] == "mla,ffn,kda-in,kda-o,experts"
    assert "GLM5_FLASH_FFN_FP8_PREFILL" not in values
    assert values["RTX_GPUS"] == "2" and values["SPARK_COUNT"] == "0"
    assert "GLM5_FLASH_FP8_HEAD" not in values and "GLM5_FLASH_KDA_FP8" not in values
    assert re.fullmatch("[0-9a-f]{40}", values["MODEL_REVISION"])
    from test_run_family_dispatch import _family_launch_result, _snapshot
    _snapshot(tmp_path / "hf", values["SPECULATOR_MODEL_ID"], {"model_type": "glmf_dflash"}, "draft-revision")
    result = _family_launch_result(tmp_path, GLMF, values["MODEL_ID"],
        "".join(f"{k}={v}\n" for k, v in values.items() if k not in ("MODEL_ID", "MODEL_REVISION")),
        extra_args=("--wip", "mc-fp8-test"), encoder_plan=LOCAL_PLAN)
    launch = _coordinator(result)
    assert "--core-view fp8-compact" in launch
    assert "--attention-fp8-prefill all" in launch and "--ffn-fp8-prefill" not in launch


@pytest.mark.parametrize("name", sorted(V4_PREP))
def test_v4_candidate_config_contract(name: str, tmp_path: Path) -> None:
    path = EXAMPLES / name
    values = keys(path)
    text = path.read_text()
    assert f"./run.sh --config examples/configs/{name}" in text and "--dry-run" in text
    if name.startswith("glm53f-exl3-k4-core-k6"):
        assert values["MODEL_ID"] == "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-Core-K6"
        assert values["SPECULATOR"] == "dflash2"
        assert values["SPECULATOR_MODEL_ID"] == "incoai/GLM-5.3-Flash-DFlash2"
        assert values["SPARK_COUNT"] == ("2" if "-min" in name else "0")
        assert values["RTX_GPUS"] == ("1" if "-min" in name else "2")
        assert values["POOL_TOKENS"] == "auto"
        assert not any(needle in text.lower() for needle in PRIVATE)
        return
    assert re.fullmatch("[0-9a-f]{40}", values["MODEL_REVISION"])
    assert not any(needle in text.lower() for needle in PRIVATE)
    assert "EXLR8_DENSE_FROM" not in values
    if name.startswith("glm53"):
        assert values["EXLR8_MANIFEST"] == "up1"
        assert values["MODEL_ID"] == "kindlingai/glm-5.3-exlr8-k2-k3-k4"
        assert "RTX_EXPERT_LAYERS" not in values
        assert values["SPARK_COUNT"] == ("4" if "-min" in name else "6")
        assert values["ATTENTION_PLACEMENT"] == ("heads" if "-min" in name else "context")
        # Native-dense launcher integration is owned by #659/#663, not this prep.
        # The smoke-entry loader test above still validates every config key.
    else:
        assert values["SPARK_COUNT"] == "0" and values["RTX_GPUS"] == "1"
        speculator = "off" if name.startswith("qwen35-9b") else "dspark"
        assert values["SPECULATOR"] == speculator
        from test_run_family_dispatch import _family_launch_result, _snapshot
        _snapshot(tmp_path / "hf", "RadixArk/Qwen3.8-27B-DSpark",
                  {"speculators_model_type": "dspark"}, "85ef153be924f17ce4bf62726954eeaa4a73e854")
        launch_values = {k: v for k, v in values.items() if k not in ("MODEL_ID", "MODEL_REVISION")}
        result = _family_launch_result(tmp_path, CHECKPOINTS[values["MODEL_ID"]], values["MODEL_ID"],
            "".join(f"{k}={v}\n" for k, v in launch_values.items()),
            extra_args=("--wip", "dense-test"), encoder_plan=LOCAL_PLAN)
        launch = _coordinator(result)
        assert "serve-qwen35" in launch
        assert ("--draft " in launch) == (speculator == "dspark")
        assert "--peers " not in launch


def test_v4_matrix_card_set_and_retired_pins() -> None:
    v3 = json.loads((ROOT / "scripts/bench/release-smoke-v3.json").read_text())
    v4 = json.loads((ROOT / "scripts/bench/release-smoke-v4.json").read_text())
    entries = {entry["name"]: entry for entry in v4["entries"]}
    assert len(entries) == len(v4["entries"])
    for entry in v3["entries"]:
        if entry["name"].endswith("-5090") and entry["name"] != "qwen38-nvfp4-5090":
            candidate = {**entries[entry["name"]], "set": dict(entries[entry["name"]]["set"])}
            candidate["set"].pop("SPARK_EXPERT_DOCKER_INFERENCE", None)
            assert candidate == entry
    assert entries["qwen38-exl3-5090"]["set"]["SPARK_COUNT"] == "1"
    for name, gpus, sparks in [
        ("v41-flash-max", 2, 6), ("v41-flash-reference-tp4", 1, 4),
        ("v4-flash-reference-tp4", 1, 4), ("glm53-exl3-max", 2, 6),
        ("glm53-exlr8-min", 1, 4), ("glm53-exlr8-max", 2, 6), ("glm53-exlr8-a8-max", 2, 6),
        ("kimi-k3-min", 1, 6), ("kimi-k3-max", 2, 6), ("step5-fp8-max", 2, 6),
        ("step5-nvfp4-min", 1, 4), ("step5-exl3-k6core-min", 1, 3), ("step5-exl3-k6core-max", 2, 6),
        ("qwen38-27b-bf16-min", 1, 0),
        ("qwen38-27b-fp8-min", 1, 0), ("qwen35-9b-bf16-min", 1, 0),
        ("qwen38-27b-bf16-min-nospec", 1, 0), ("qwen38-27b-fp8-min-nospec", 1, 0),
    ]:
        assert len(entries[name]["gpus"]) == gpus
        assert len(entries[name]["sparks"]) == sparks
        assert entries[name]["set"]["SPARK_COUNT"] == str(sparks)
    a8 = entries["glm53-exlr8-a8-max"]
    assert a8 == {**entries["glm53-exlr8-max"], "name": "glm53-exlr8-a8-max", "set": {
        **entries["glm53-exlr8-max"]["set"], "EXLR8_PREFILL_ACTIVATIONS": "a8",
        "ADDR": a8["set"]["ADDR"], "EXPERT_PORT": a8["set"]["EXPERT_PORT"]}}
    dense = [entry for entry in v4["entries"] if entry["family"] == "qwen3_5"]
    assert {entry["name"]: entry["set"]["SPECULATOR"] for entry in dense} == {
        "qwen35-9b-bf16-min": "off", "qwen38-27b-bf16-min": "dspark",
        "qwen38-27b-fp8-min": "dspark", "qwen38-27b-bf16-min-nospec": "off",
        "qwen38-27b-fp8-min-nospec": "off"}
    for quant in ("bf16", "fp8"):
        base = entries[f"qwen38-27b-{quant}-min"]
        nospec = entries[f"{base['name']}-nospec"]
        assert nospec == {**base, "name": f"{base['name']}-nospec", "set": {
            **base["set"], "SPECULATOR": "off", "ADDR": nospec["set"]["ADDR"]}}
    assert "SPARK_EXPERT_DOCKER_INFERENCE" not in v4["defaults"]["set"]
    for entry in v4["entries"]:
        assert ("SPARK_EXPERT_DOCKER_INFERENCE" in entry["set"]) == bool(entry["sparks"])
    for entry in dense:
        assert "EXPERT_PORT" not in entry["set"]
        assert sum(other.get("set", {}).get("ADDR") == entry["set"]["ADDR"]
                   for other in v4["entries"]) == 1
    assert not any(name.startswith("qwen38-nvfp4") for name in entries)
    ml4_retired = [entry for entry in v4["retired"] if entry["name"] in {"ml4-exl3-min", "ml4-exl3-max"}]
    assert {entry["name"] for entry in ml4_retired} == {"ml4-exl3-min", "ml4-exl3-max"}
    for entry in ml4_retired:
        assert entry["release_note"] == "Core-K6 checkpoint retired 2026-11-12; replaced by Mistral-Large-4-EXL3-K4"
        assert not (ROOT / entry["config"]).exists()
        assert entry["release"] == "v5.0.0-rc1"
        assert (ROOT / entry["report"] / "report.json").is_file()
    retired = [entry for entry in v4["retired"] if entry not in ml4_retired]
    assert len(retired) == 21
    assert "glm53-exl3-max" not in {entry["name"] for entry in retired}
    assert {"qwen38-nvfp4-" + size for size in ("min", "max", "5090")} <= {e["name"] for e in retired}
    published = {entry["name"]: entry for entry in
                 json.loads((ROOT / "benchmarks/current-matrix.json").read_text())["retired"]}
    assert all(published.get(entry["name"], {}).get("report") == entry["report"] for entry in retired)
    v5 = json.loads((ROOT / "scripts/bench/release-smoke-v5.json").read_text())
    assert {name: entry["report"] for name, entry in published.items()} == {
        entry["name"]: entry["report"] for entry in v5["retired"]
    }
    assert set(published) - {entry["name"] for entry in retired} == {
        "v41-flash-reference-tp4", "glm53f-exl3-5090", "v4-flash-reference-tp4",
        "glm53-exlr8-a8-max", "qwen38-27b-bf16-min-nospec", "qwen38-27b-fp8-min-nospec",
        "glm53-exl3-min", "glm53-exl3-max", "glm53-exl3-k6core-min",
        "glm53-exl3-k6core-max", "step5-exl3-k6core-min", "step5-exl3-k6core-max",
    }
    reports = [(path, json.loads(path.read_text())) for path in (ROOT / "benchmarks").glob("*/*/report.json")]
    for entry in retired:
        assert_archived_config(entry)
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


@pytest.mark.parametrize("filename", ["scripts/bench/release-smoke-v4.json",
                                    "scripts/bench/kits/gate/matrix-gate.json"])
def test_glmf_exlr8_k4_cards_preserve_qualified_native_recipes(filename: str) -> None:
    matrix = json.loads((ROOT / filename).read_text())
    entries = {entry["name"]: entry for entry in matrix["entries"]}
    pins = keys(ROOT / "cuteafd.config")
    for size, gpus, sparks in (("min", [1], 1), ("max", [0, 1], 0)):
        entry = entries[f"glmf-exlr8-k4-{size}"]
        config = keys(ROOT / entry["config"])
        values = {**config, **matrix.get("defaults", {}).get("set", {}), **entry["set"]}
        assert config["MODEL_ID"] == "kindlingai/glm-5.3-flash-exlr8-k2-k3-k4"
        assert config["MODEL_REVISION"] == "867580abd28d142bb4c56218bb81a752b7dea988"
        assert config["SERVED_MODEL_ID"] == "zai-org/GLM-5.3-Flash"
        assert config["EXLR8_MANIFEST"] == "k4"
        assert config["EXLR8_PREFILL_ACTIVATIONS"] == "a16"
        assert config["SPECULATOR"] == "dflash2"
        assert config["SPECULATOR_MODEL_ID"] == "incoai/GLM-5.3-Flash-DFlash2"
        assert config["POOL_TOKENS"] == config["ATTENTION_PLACEMENT"] == "auto"
        assert "VISION" not in config
        assert values.get("VISION", "auto") == "auto"
        assert config["AUDIO"] == "off"
        assert config["EXPERT_BACKEND"] == ("spark" if sparks else "local")
        for key in ("RTX_EXPERT_LAYERS", "EXLR8_DENSE_FROM"):
            assert key not in values
        for image in ("COORDINATOR_DOCKER_INFERENCE", "SPARK_EXPERT_DOCKER_INFERENCE"):
            assert config[image] == pins[image]
        assert entry["family"] == "glm5_flash"
        assert entry["gpus"] == gpus and len(entry["sparks"]) == sparks
        assert values["RTX_GPUS"] == str(len(gpus))
        assert values["COORDINATOR_GPUS"] == ",".join(map(str, gpus))
        assert values["SPARK_COUNT"] == str(sparks)
        for rank, host in enumerate(entry["sparks"]):
            assert values[f"SPARK_{rank}_HOST"] == host
        for key in ("ADDR", "EXPERT_PORT"):
            assert sum(other.get("set", {}).get(key) == entry["set"][key]
                       for other in entries.values()) == 1
    if "matrix-gate" in filename:
        assert entries["glmf-exlr8-k4-max"]["prefill_lengths"] == [65536]


@pytest.mark.parametrize("family,model,revision,sparks", [
    ("glm5", "GLM-5.3-EXL3-K4-MC", "c39486376dee84e5faa4904e014e07d7b6e76405", 4),
    ("step5", "Step-5-Preview-EXL3-K4-MC", "4483a4bd874e8c360aec486ea72bd31d1ad7a247", 3),
])
def test_glm53_step5_mc_rollout_pins_original_and_separates_comparisons(
    family: str, model: str, revision: str, sparks: int,
) -> None:
    repo = "wrldsuksgo2mars/" + model
    release = json.loads((ROOT / "scripts/bench/release-smoke-v5.json").read_text())
    comparisons = json.loads((ROOT / "scripts/bench/release-smoke-v51-mc-core-comparison.json").read_text())
    published = json.loads((ROOT / "benchmarks/current-matrix.json").read_text())
    entries = [entry for entry in release["entries"] if entry["set"].get("MODEL_ID") == repo]
    assert len(entries) == 2
    assert {tuple(entry["gpus"]) for entry in entries} == {(1,), (0, 1)}
    retired_configs = {entry["config"] for entry in published["retired"]}
    for entry in entries:
        assert entry["config"] not in retired_configs
        config = keys(ROOT / entry["config"])
        assert config["MODEL_ID"] == repo and config["MODEL_REVISION"] == revision
        assert config["CORE_VIEW"] == "original"
        values = {**config, **entry["set"]}
        assert values["MODEL_REVISION"] == revision
        assert values["CORE_VIEW"] == "original"
        assert len(entry["sparks"]) == (sparks if len(entry["gpus"]) == 1 else 6)
        assert values["SPECULATOR"] == ("dflash2" if family == "glm5" else "mtp")
    compact = [entry for entry in comparisons["entries"] if entry["family"] == family]
    assert {entry["set"]["CORE_VIEW"] for entry in compact} == {"k8-compact", "k6-compact"}
    for entry in compact:
        assert entry["config"] not in retired_configs
        assert keys(ROOT / entry["config"])["MODEL_ID"] == repo
        assert entry["set"]["MODEL_ID"] == repo
        assert entry["set"]["MODEL_REVISION"] == revision
        assert entry["gpus"] == [0, 1] and len(entry["sparks"]) == 6
        assert entry["name"] not in {entry["name"] for entry in release["entries"]}
    unsupported = [entry for entry in comparisons["unsupported"] if entry["family"] == family]
    assert len(unsupported) == 1
    assert unsupported[0]["core_view"] == "fp8-compact"
    assert unsupported[0]["status"] == "na" and unsupported[0]["reason"]
    rows = [row for row in published["cells"] if row["checkpoint"] == repo]
    assert len(rows) == 1
    assert all(rows[0][column]["status"] in {"pending", "current"} for column in ("1rtx", "2rtx"))
    retired_models = {"wrldsuksgo2mars/" + name for name in (
        "GLM-5.3-EXL3-K4-v1", "GLM-5.3-EXL3-K4-Core-K6", "Step-5-Preview-EXL3-K4-Core-K6",
    )}
    assert not retired_models.intersection(row["checkpoint"] for row in published["cells"])
    assert not retired_models.intersection(entry["set"].get("MODEL_ID") for entry in release["entries"])
    for page in ("docs/models/" + family + ".md", "docs/mc-checkpoints.md", "docs/options.md"):
        assert model in (ROOT / page).read_text()
        assert revision in (ROOT / page).read_text()
