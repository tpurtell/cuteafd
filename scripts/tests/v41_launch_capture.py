"""Run a launcher end to end against process-boundary stubs and record what it starts.

The DeepSeek V4.1 equivalence gate (#82) compares the coordinator and Spark
container argv, mounts and environment of the launcher before and after V4.1
moved into scripts/launch/run-family.sh. docker, nvidia-smi, curl, ss and nest
are stubs; ssh runs the remote script locally (with that host's HOME), so the
Spark `docker run` calls are recorded exactly as each host would issue them.
Only `docker run -d` (a serving container) is recorded.

  python3 scripts/tests/v41_launch_capture.py --write   # refresh the goldens

The goldens in scripts/tests/fixtures/v41-launch/ were first recorded from the
pre-#82 run.sh (V4.1's private path, 0fe78133); test_v41_shared_launcher.py
compares today's launcher with them, minus the intended additions it names.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "scripts" / "tests" / "fixtures" / "v41-launch"
REVISIONS = {"deepseek-ai/DeepSeek-V4.1-Flash": "dba1be0a40aa45a94ad051997016db3960a90277",
             "nvidia/DeepSeek-V4.1-Flash-NVFP4": "3431dde3247c13b5957f682b1e3c6fcae2566079"}
CHECKPOINTS = {"deepseek-ai/DeepSeek-V4.1-Flash": {"model_type": "deepseek_v41"},
               "nvidia/DeepSeek-V4.1-Flash-NVFP4": {"model_type": "deepseek_v41",
                                                    "quantization_config": {"moe_quant_algo": "NVFP4"}}}
GPU_UUIDS = ("GPU-00000000-0000-0000-0000-000000000000", "GPU-11111111-1111-1111-1111-111111111111")
GPU_PCI = ("00000000:01:00.0", "00000000:02:00.0")
HOSTS = ("ostrich", "dodo", "emu", "kiwi", "rhea", "moa")
# The gate kit's two V4.1 cards (scripts/bench/kits/gate/matrix-gate.json), as
# wip-cards.py generates their configs for a WIP arm.
KIT_CARDS = ("v41-flash-max", "v41-flash-min")
EXAMPLES = ("deepseek-v41-mxfp4-max", "deepseek-v41-mxfp4-min", "deepseek-v41-nvfp4-max", "deepseek-v41-nvfp4-min")
CASES = EXAMPLES + KIT_CARDS
WIP_SLOT = "gate"

# Every stub logs its argv as one JSON line; behavior is keyed on the arguments.
STUB = r'''#!/usr/bin/env python3
import json, os, sys, subprocess
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
log = os.environ["CAPTURE_LOG"]
host = os.environ.get("CAPTURE_HOST", "raptor")
def record(kind, value):
    with open(log, "a") as stream:
        stream.write(json.dumps([kind, host, value]) + "\n")
if tool == "ssh":
    # OpenSSH: options, host, then one command string run by the remote shell.
    rest = list(args)
    while rest and rest[0].startswith("-"):
        rest = rest[2:] if rest[0] in ("-o", "-F", "-p", "-i", "-l") else rest[1:]
    target, command = rest[0], " ".join(rest[1:])
    home = Path(os.environ["CAPTURE_ROOT"]) / "hosts" / target
    env = dict(os.environ, HOME=str(home), CAPTURE_HOST=target)
    env.pop("HF_HOME", None)
    stdin = None if "bash -s" in command or "python3 -" in command or sys.stdin.isatty() else subprocess.DEVNULL
    sys.exit(subprocess.run(["bash", "-c", command], env=env, stdin=stdin).returncode)
if tool == "nvidia-smi":
    joined = " ".join(args)
    if "topo" in joined:
        print("\tGPU0\tGPU1\nGPU0\tX\tOK\nGPU1\tOK\tX")
    elif "query-compute-apps" in joined:
        pass
    elif "index,uuid,pci.bus_id,memory.total,memory.free" in joined:
        for i in range(2):
            print(f"{i}, {UUIDS[i]}, {PCI[i]}, 97887, 97000")
    elif "index,uuid,pci.bus_id" in joined:
        selector = next(a.split("=", 1)[1] for a in args if a.startswith("--id="))
        i = PCI.index(selector) if selector in PCI else int(selector)
        print(f"{i}, {UUIDS[i]}, {PCI[i]}")
    elif "memory.total" in joined:
        print(97887)
    elif "memory.free" in joined:
        print(97000)
    else:
        print("0\n1")
    sys.exit(0)
if tool == "curl":
    print(json.dumps({"object": "list", "data": [{"id": "deepseek-ai/DeepSeek-V4.1-Flash"}, {"id": "test/model"}]}))
    sys.exit(0)
if tool in ("ss", "nest"):
    sys.exit(0)
if tool == "id":
    print({"-u": "1000", "-g": "1000", "-G": "1000 44", "-un": "tj"}[args[0]])
    sys.exit(0)
if tool == "hostname":
    print("raptor")
    sys.exit(0)
if tool == "timeout":
    # Readiness probes a worker port: report it open.
    if any("/dev/tcp" in a for a in args):
        sys.exit(0)
    sys.exit(subprocess.run(args[1:]).returncode)
if tool == "docker":
    if args[:2] == ["run", "-d"]:
        record("run", args)
        print("container-id")
        sys.exit(0)
    if args and args[0] == "run" and "plan" in args:
        record("plan", args)
        ranks = os.environ["CAPTURE_SPARKS"]
        explicit = "--spark-ranks" in args
        if explicit:
            ranks = args[args.index("--spark-ranks") + 1]
        rows = args[args.index("--prefill-rows") + 1] if "--prefill-rows" in args else "0"
        print(json.dumps({"spark_ranks": int(ranks), "prefill_rows": int(rows) or 2048,
                          "encoder": {"kind": {"kind": "spark", "rank": 0}, "replicas": []},
                          "encoder_plan_hash": "ab" * 32, "placement_supported": True, "fits": True}))
        sys.exit(0)
    if args and args[0] == "run" and "--entrypoint" in args and "python3" in args:
        # The NVFP4 role check reads the image's TP manifest.
        sys.exit(0)
    if args and args[0] == "run" and "serve-native" in args and "--help" in args:
        print("--console-secret-file")
        sys.exit(0)
    if args and args[0] == "run":
        record("other-run", args)
        sys.exit(0)
    if args and args[0] == "exec" and "python3" in args:
        sys.stdin.read()
        print("tp2;tp3;tp6")
        sys.exit(0)
    if args and args[0] == "exec" and "cat" in args and any(a.endswith("plan.json") for a in args):
        print(Path(os.environ["CAPTURE_PLAN"]).read_text())
        sys.exit(0)
    if args and args[0] == "exec":
        sys.exit(0)
    if args and args[0] in ("inspect", "container") or args[:2] == ["container", "inspect"]:
        joined = " ".join(args)
        if args[:2] == ["container", "inspect"] and "-wip" in args[-1]:
            sys.exit(0)
        if "State.Status" in joined:
            print("running")
            sys.exit(0)
        if "Destination" in joined:
            sys.exit(1)
        sys.exit(1)
    if args[:2] == ["image", "inspect"]:
        joined = " ".join(args)
        if "{{.Id}}" in joined:
            print("sha256:" + "0" * 64)
        elif "spark_tp_roles" in joined:
            print("tp2;tp3;tp6")
        elif "sparkinfer.revision" in joined:
            print(os.environ["CAPTURE_SPARKINFER"])
        elif "image.revision" in joined:
            print("engine-revision")
        elif "console-gate" in joined:
            print("1")
        sys.exit(0)
    if args and args[0] == "cp":
        src, dst = args[1], args[2]
        dst = Path(dst)
        if src.endswith("/.cuteafd-wip/."):
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("cuteafd", "libcuteafd_native.so", "PROGRAMS.json"):
                (dst / name).write_text(name)
        elif src.endswith("release-entrypoint.sh"):
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "release-entrypoint.sh").write_text("#!/bin/sh\n")
        elif src.endswith("third_party/sparkinfer"):
            (dst / "sparkinfer").mkdir(parents=True, exist_ok=True)
        elif src.endswith("sparkinfer.lock.json"):
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "sparkinfer.lock.json").write_text("{}")
        elif src.endswith("verify-sparkinfer-source.py"):
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "verify-sparkinfer-source.py").write_text("print('ok')\n")
        sys.exit(0)
    if args and args[0] in ("info", "top", "pull", "rm", "stop", "logs", "ps"):
        sys.exit(0)
    sys.exit(0)
sys.exit(0)
'''


def stub_text() -> str:
    return STUB.replace("UUIDS[i]", f"{list(GPU_UUIDS)!r}[i]").replace("PCI[i]", f"{list(GPU_PCI)!r}[i]") \
        .replace("PCI.index(selector) if selector in PCI", f"{list(GPU_PCI)!r}.index(selector) if selector in {list(GPU_PCI)!r}")


def kit_config(name: str, directory: Path) -> Path:
    """The base.config wip-cards.py generates for a gate-kit V4.1 card on a WIP arm."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("wip_cards", ROOT / "scripts" / "bench" / "wip-cards.py")
    cards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cards)
    matrix = json.loads((ROOT / "scripts/bench/kits/gate/matrix-gate.json").read_text())
    entry = next(e for e in matrix["entries"] if e["name"] == name)
    entry = {**entry, "config": str(ROOT / entry["config"])}
    key = directory / "api-key"
    key.write_text("k" * 64 + "\n")
    key.chmod(0o600)
    arm = {"instance": "cap", "slot": WIP_SLOT}
    job = cards.generate(entry, "cand", arm, directory / "state", key, {}, set(), None, 1)
    config = Path(job["entry"]["config"])
    return config


def example_config(name: str, directory: Path) -> Path:
    """An example config with its documentation hosts replaced by the pool's."""
    text = (ROOT / "examples" / "configs" / f"{name}.config").read_text()
    for rank, host in enumerate(HOSTS):
        text = text.replace(f"REPLACE-ME-spark-{rank}", host).replace(f"192.0.2.{rank + 1}", f"10.55.0.{rank + 1}")
    config = directory / f"{name}.config"
    config.write_text(text)
    return config


def normalize(value: str, root: Path) -> str:
    return value.replace(str(root), "<root>")


def capture(launcher_root: Path, case: str) -> dict:
    """Run LAUNCHER_ROOT/run.sh for one case; return the recorded serving containers."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
        root = Path(temporary)
        (root / "bin").mkdir()
        for tool in ("docker", "ssh", "nvidia-smi", "curl", "ss", "nest", "id", "hostname", "timeout"):
            path = root / "bin" / tool
            path.write_text(stub_text())
            path.chmod(0o755)
        home = root / "hosts" / "raptor"
        home.mkdir(parents=True)
        for host in HOSTS:
            (root / "hosts" / host).mkdir(parents=True, exist_ok=True)
        hub = root / "hf" / "hub"
        for model, config in CHECKPOINTS.items():
            snapshot = hub / f"models--{model.replace('/', '--')}" / "snapshots" / REVISIONS[model]
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(json.dumps(config))
            (hub / f"models--{model.replace('/', '--')}" / "refs").mkdir()
            (hub / f"models--{model.replace('/', '--')}" / "refs" / "main").write_text(REVISIONS[model])
        for host in HOSTS + ("raptor",):
            cache = root / "hosts" / host / ".cache" / "huggingface"
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "hub").symlink_to(hub)
        # The same launcher tree at a fixed path inside the root, so paths compare.
        repo = root / "repo"
        shutil.copytree(launcher_root, repo, symlinks=True,
                        ignore=shutil.ignore_patterns(".git", "rust", "native", "third_party", "python", "docs",
                                                      "benchmarks", "assets", "quantization", "set", "target"))
        (repo / "rust/crates/cuteafd-core/src").mkdir(parents=True)
        shutil.copy(launcher_root / "rust/crates/cuteafd-core/src/coordinator_programs.rs",
                    repo / "rust/crates/cuteafd-core/src/coordinator_programs.rs")
        (repo / "third_party/sparkinfer").mkdir(parents=True)
        lock = json.loads((launcher_root / "third_party/sparkinfer.lock.json").read_text())
        (repo / "third_party/sparkinfer.lock.json").write_text(json.dumps(lock))
        (repo / "scripts/build").mkdir(parents=True, exist_ok=True)
        (repo / "scripts/build/verify-sparkinfer-source.py").write_text(
            f"import sys\nprint({lock['revision']!r}) if '--print-revision' in sys.argv else None\n")
        directory = root / "config"
        directory.mkdir()
        config = kit_config(case, directory) if case in KIT_CARDS else example_config(case, directory)
        text = config.read_text()
        sparks = next(int(line.split("=", 1)[1]) for line in text.splitlines() if line.startswith("SPARK_COUNT="))
        rtx = next(int(line.split("=", 1)[1]) for line in text.splitlines() if line.startswith("RTX_GPUS="))
        (root / "plan.json").write_text(json.dumps({
            "version": 1, "rtx_gpus": rtx, "nonce": "n", "rtx_expert_layers": 20 if rtx == 2 else 5,
            "spark_first_layer": 20 if rtx == 2 else 5}))
        log = root / "log.jsonl"
        env = {"PATH": f"{root / 'bin'}:/usr/bin:/bin", "HOME": str(home), "HF_HOME": str(root / "hf"),
               "CAPTURE_LOG": str(log), "CAPTURE_ROOT": str(root), "CAPTURE_PLAN": str(root / "plan.json"),
               "CAPTURE_SPARKS": str(sparks), "CAPTURE_SPARKINFER": lock["revision"],
               "CUTEAFD_RELEASE_READY_TIMEOUT_SECONDS": "30", "TMPDIR": str(root),
               "LANG": "C.UTF-8", "USER": "tj", "LOGNAME": "tj"}
        args = ["--config", str(config), "--restart"]
        if case in KIT_CARDS:
            args += ["--wip", WIP_SLOT]
        result = subprocess.run(["bash", str(repo / "run.sh"), *args], cwd=repo, env=env,
                                capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError(f"{case}: launcher exited {result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
        runs, plans = [], []
        for line in log.read_text().splitlines():
            kind, host, value = json.loads(line)
            if kind == "run":
                runs.append({"host": host, "argv": [normalize(v, root) for v in value]})
            elif kind == "plan":
                plans.append([normalize(v, root) for v in value])
        # Spark workers start in parallel: order by host, coordinator first.
        runs.sort(key=lambda run: (run["host"] != "raptor", run["host"]))
        return {"case": case, "containers": runs, "plans": plans}


def parse_run(argv: list[str]) -> dict:
    """A `docker run -d` argv as {options, env, mounts, image, command}, order-free where Docker is."""
    options, env, mounts = [], {}, []
    flags_with_value = {"--name", "--restart", "--gpus", "--network", "--ipc", "--ulimit", "--device",
                        "--security-opt", "--user", "--group-add", "--entrypoint", "--stop-signal",
                        "--stop-timeout", "--label", "--mount"}
    index = 2
    while index < len(argv):
        token = argv[index]
        if token == "-e":
            key, _, value = argv[index + 1].partition("=")
            env[key] = value
            index += 2
        elif token == "-v":
            mounts.append(argv[index + 1])
            index += 2
        elif token == "--mount":
            mounts.append("mount:" + argv[index + 1])
            index += 2
        elif token in flags_with_value:
            options.append(f"{token} {argv[index + 1]}")
            index += 2
        elif token.startswith("-"):
            options.append(token)
            index += 1
        else:
            break
    return {"options": sorted(options), "env": env, "mounts": sorted(mounts), "image": argv[index],
            "command": argv[index + 1:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launcher", type=Path, default=ROOT, help="checkout whose run.sh to record")
    parser.add_argument("--write", action="store_true", help="write the goldens")
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    for case in args.case or CASES:
        result = capture(args.launcher.resolve(), case)
        text = json.dumps(result, indent=1) + "\n"
        if args.write:
            FIXTURES.mkdir(parents=True, exist_ok=True)
            (FIXTURES / f"{case}.json").write_text(text)
            print(f"wrote {case}: {len(result['containers'])} containers")
        else:
            sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
