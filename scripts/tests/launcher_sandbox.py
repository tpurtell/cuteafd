"""Session-owned, immutable launcher inputs; writable files never share inodes."""
from pathlib import Path
import json
import shutil
import tempfile

_ROOT = None
_TEMPLATES = {}

LAUNCHER_SHARED_INPUTS = tuple(json.loads(
    (Path(__file__).resolve().parents[2] / "scripts/build/spark-package-ranges.json").read_text()
)["shared_inputs"])


def configure(root):
    global _ROOT
    _ROOT = root
    _TEMPLATES.clear()


def _template_dir():
    """A fresh directory per template: a copy that fails partway leaves its
    own debris, never a name the next template would reuse."""
    return Path(tempfile.mkdtemp(prefix="template-", dir=_ROOT))


def copy_input(source, destination):
    """Link a read-only input from a private session copy, not the checkout."""
    source, destination = Path(source), Path(destination)
    if destination.is_dir():
        destination /= source.name
    if source.name == "release-common.sh":
        root = source.resolve().parents[2]
        copy_launcher_shared_inputs(root, destination.parent.parent.parent)
        # run-family.sh sources the per-family key registries and the attention
        # validation beside release-common.
        for shared in ("family-keys.sh", "attention-placement.sh"):
            staged = destination.parent / shared
            if not staged.exists():
                copy_input(root / "scripts" / "lib" / shared, staged)
    if _ROOT is None:
        return shutil.copy(source, destination)
    key = source.resolve()
    if key not in _TEMPLATES:
        cached = _template_dir() / source.name
        shutil.copy(source, cached)
        _TEMPLATES[key] = cached
    # A shared dependency may already have been staged before an explicit copy.
    if destination.exists():
        destination.unlink()
    destination.hardlink_to(_TEMPLATES[key])
    return str(destination)


def copy_launcher_shared_inputs(source_root, destination_root):
    """Stage the same relative Python/data paths used by the real source tree."""
    for relative in LAUNCHER_SHARED_INPUTS:
        source, target = Path(source_root) / relative, Path(destination_root) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            copy_input(source, target)


def fake_deployment_memory_layout(rtx, ranks, spark_memory_cap=None):
    """Typed device records shared by process-boundary planner stubs.

    Self-contained: stubs inline it with inspect.getsource. The Spark cap is a
    launcher memory size (B, MB, GB, MiB or GiB; bare bytes)."""
    import re
    from decimal import Decimal
    budget = 108 << 30
    if spark_memory_cap:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]{1,6})?)(B|MB|GB|MiB|GiB)?", str(spark_memory_cap))
        scale = {None: 1, "B": 1, "MB": 10**6, "GB": 10**9, "MiB": 2**20, "GiB": 2**30}[match.group(2)]
        budget = int(Decimal(match.group(1)) * scale)
    return {"devices": [
        {"kind": "rtx", "index": i, "capacity_bytes": 96 << 30,
         "worker_budget_bytes": 0, "items": [{"category": "weights", "bytes": 8 << 30}],
         "kv_tokens": 0} for i in range(int(rtx))
    ] + [
        {"kind": "spark", "index": i, "capacity_bytes": budget,
         "worker_budget_bytes": budget, "items": [], "kv_tokens": 0}
        for i in range(int(ranks))
    ]}


def fake_resolved_settings(args, family, sparks):
    """Canned launch-ABI response; real resolver/fit parity uses the CPU binary."""
    from pathlib import Path
    def value(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default
    config = {}
    path = value("--config")
    if path:
        for index, arg in enumerate(args[:-1]):
            if arg in ("-v", "--volume"):
                parts = args[index + 1].split(":")
                if len(parts) > 1 and parts[1] == path:
                    path = parts[0]
        if Path(path).is_file():
            config = dict(line.split("=", 1) for line in Path(path).read_text().splitlines()
                          if "=" in line and not line.startswith("#"))
    def resolved(flag, key, auto):
        explicit = value(flag)
        configured = config.get(key)
        raw = explicit if explicit is not None else configured
        return {"value": auto if raw in (None, "", "auto") else raw,
                "provenance": "explicit" if explicit else "config" if configured not in (None, "", "auto") else "auto"}
    tp = int(value("--rtx", "1"))
    shared = tp == 1 and int(value("--pool-tokens", "0") or 0) == 0 and int(sparks) > 0
    result = {"coordinator_weight_budget_gib": {
        "value": int(value("--gpu-memory-cap", str(96 << 30))) / 2**30, "provenance": "auto"}}
    if family == "kimi_k3":
        qualified = (config.get("MODEL_ID") == "wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC"
            and value("--core-view", "original") in ("original", "k6-compact") and tp in (1, 2)
            and value("--kv-cache", "fp8") == "fp8")
        precision = resolved("--draft-fp8", "SPECULATOR_FP8", value("--speculator") == "dflash2" or qualified)
        raw = precision["value"]
        precision["value"] = raw if isinstance(raw, bool) else raw in ("on", "true")
        result["draft_fp8"] = precision
    if family == "qwen4":
        result["native_mtp_layers"] = {"value": int(value("--native-mtp-layers", config.get("SPECULATOR_DEPTH") or "3")), "provenance": "auto"}
    if family == "mimo_v2":
        precision = resolved("--draft-fp8", "SPECULATOR_FP8", True)
        raw = precision["value"]
        precision["value"] = raw if isinstance(raw, bool) else raw in ("on", "true")
        result["draft_fp8"] = precision
        result["native_mtp_layers"] = {"value": int(config.get("SPECULATOR_DEPTH") or 1) if value("--speculator") == "mtp" else 0, "provenance": "auto"}
    if family == "glm5_flash":
        result.update(index_cache=resolved("--index-cache", "GLM5_FLASH_INDEX_CACHE", "compact" if tp == 1 else "keys"),
            draft_linear=resolved("--draft-linear", "GLM5_FLASH_DRAFT_LINEAR",
                "w8a8" if value("--speculator", "off") != "off" and config.get("SPECULATOR_FP8") != "off" else "w8a16"),
            replay_records=resolved("--replay-records", "GLM5_FLASH_REPLAY_RECORDS", "shared" if shared else "own"))
        if result["index_cache"]["value"] not in ("keys", "compact"):
            raise SystemExit("GLM5_FLASH_INDEX_CACHE must be keys or compact")
        if result["draft_linear"]["value"] not in ("w8a16", "wide", "w8a8"):
            raise SystemExit("GLM5_FLASH_DRAFT_LINEAR must be w8a16, wide or w8a8")
        if result["replay_records"]["value"] not in ("own", "shared"):
            raise SystemExit("GLM5_FLASH_REPLAY_RECORDS must be own or shared")
        if result["replay_records"]["value"] == "shared" and tp != 1:
            raise SystemExit("GLM5_FLASH_REPLAY_RECORDS=shared keeps the records in one GPU's prefill scratch; serve it without a head split")
    return result


def stub_cuda_capacity(common, total_bytes):
    """Launcher captures fake the host; driver/UUID behavior has its own unit tests."""
    common = Path(common)
    source = common.read_text()
    common.unlink()  # Cached inputs may be hardlinked into other captures.
    common.write_text(source + '\nrelease_cuda_total_bytes() { printf "%s\\n" '
                      + str(int(total_bytes)) + '; }\n')


def stub_cuda_python(bin_dir, total_bytes):
    """Intercept only the driver query in captures that launch the live checkout."""
    import sys
    wrapper = Path(bin_dir) / 'python3'
    wrapper.write_text('#!' + sys.executable + '\n'
        'import subprocess, sys\n'
        'args = sys.argv[1:]\n'
        'source = sys.stdin.read() if args and args[0] == "-" else None\n'
        'if source is not None and "cuDeviceTotalMem_v2" in source:\n'
        f'    print({int(total_bytes)})\n'
        'else:\n'
        f'    raise SystemExit(subprocess.run([{sys.executable!r}, *args], input=source, text=True).returncode)\n')
    wrapper.chmod(0o755)


def copy_tree(source, destination, **kwargs):
    """Build the tree once per worker; give each capture its own directory entries.

    Source registries, launch scripts and verifier are edited by capture tests,
    so those use real copies. All other files are read-only inputs.
    """
    if _ROOT is None:
        return shutil.copytree(source, destination, **kwargs)
    key = ('tree', Path(source).resolve())
    if key not in _TEMPLATES:
        cached = _template_dir() / "tree"
        shutil.copytree(source, cached, **kwargs)
        _TEMPLATES[key] = cached

    def clone(src, dst):
        path = Path(src)
        if path.suffix == '.sh' or path.name == 'verify-sparkinfer-source.py':
            return shutil.copy2(src, dst)
        Path(dst).hardlink_to(src)
        return dst

    return shutil.copytree(_TEMPLATES[key], destination, symlinks=True, copy_function=clone)


FAKE_GPU_UUIDS = tuple(f"GPU-{d * 8}-{d * 4}-{d * 4}-{d * 4}-{d * 12}" for d in "0123")
FAKE_GPU_PCI = tuple(f"00000000:{bus}:00.0" for bus in ("01", "02", "41", "81"))

FAKE_NVIDIA_SMI = r'''#!/usr/bin/env python3
"""A host's nvidia-smi for launcher tests: the queries the launchers make, keyed on argv."""
import json, os, re, sys
gpus = json.loads(os.environ["FAKE_GPUS"])
args = sys.argv[1:]
joined = " ".join(args)
if os.environ.get("FAKE_NVIDIA_SMI_LOG"):
    with open(os.environ["FAKE_NVIDIA_SMI_LOG"], "a") as stream:
        stream.write(joined + "\n")
selector = next((a.split("=", 1)[1] for a in args if a.startswith("--id=")), None)
if selector is None and "-i" in args:
    selector = args[args.index("-i") + 1]
if selector is not None:
    def same(gpu):
        if selector.isdigit():
            return gpu["index"] == int(selector)
        if selector.startswith("GPU-"):
            return gpu["uuid"] == selector
        return gpu["pci"].lower().split(":", 1)[1] == selector.lower().split(":", 1)[1]
    gpus = [gpu for gpu in gpus if same(gpu)]
    if not gpus:
        sys.exit("No devices were found")
fields = {"index": "index", "uuid": "uuid", "pci.bus_id": "pci", "memory.total": "total",
          "memory.free": "free", "name": "name"}
if "topo" in args:
    names = ["GPU%d" % gpu["index"] for gpu in gpus]
    print("\t" + "\t".join(names))
    for gpu in gpus:
        print("GPU%d\t" % gpu["index"] + "\t".join("X" if other is gpu else "OK" if gpu.get("p2p", True) and other.get("p2p", True) else "NS"
                                                     for other in gpus))
elif "--query-compute-apps" in joined:
    wanted = re.search(r"--query-compute-apps=(\S+)", joined).group(1).split(",")
    for gpu in gpus:
        for pid, mib in gpu.get("apps", []):
            row = {"gpu_uuid": gpu["uuid"], "pid": pid, "used_memory": mib, "used_gpu_memory": mib}
            print(", ".join(str(row[field]) for field in wanted))
elif "--query-gpu" in joined:
    wanted = re.search(r"--query-gpu=(\S+)", joined).group(1).split(",")
    for gpu in gpus:
        print(", ".join(str(gpu[fields[field]]) for field in wanted))
elif "-L" in args:
    for gpu in gpus:
        print("GPU %d: %s (UUID: %s)" % (gpu["index"], gpu["name"], gpu["uuid"]))
else:
    print("CUDA Version: 13.0")
'''


def fake_gpus(count=2, total_mib=98304, free_mib=97000, busy=(), apps=()):
    """FAKE_GPUS for `fake_nvidia_smi`: COUNT RTX GPUs; BUSY indices report 2 GiB free;
    APPS are (index, pid, MiB) compute processes."""
    return [{"index": i, "uuid": FAKE_GPU_UUIDS[i], "pci": FAKE_GPU_PCI[i], "total": total_mib,
             "free": 2048 if i in busy else free_mib, "name": "NVIDIA RTX PRO 6000 Blackwell",
             "apps": [[pid, mib] for index, pid, mib in apps if index == i]} for i in range(count)]


def fake_nvidia_smi(bin_dir):
    """Write the shared nvidia-smi stub; callers set FAKE_GPUS (json of `fake_gpus`)."""
    import sys
    hostname = Path(bin_dir) / "hostname"
    hostname.write_text("#!/bin/sh\nprintf 'coordinator-test\\n'\n")
    hostname.chmod(0o755)
    path = Path(bin_dir) / "nvidia-smi"
    path.write_text("#!" + sys.executable + "\n" + FAKE_NVIDIA_SMI)
    path.chmod(0o755)
    return path


class SimCluster:
    """Canonical script-test environment, customizable once per suite."""
    def __init__(self, gpus=2, sparks=6, host_gib=184, gpu_bytes=101_973_491_712):
        self.gpus, self.sparks, self.host_gib, self.gpu_bytes = gpus, sparks, host_gib, gpu_bytes

    def env(self):
        return {"CUTEAFD_TEST_HOST": json.dumps({"host_ram": [self.host_gib << 30] * 2,
            "gpu_count": self.gpus, "gpu_bytes": self.gpu_bytes,
            "spark_count": self.sparks, "spark_bytes": 121 << 30})}

    def devices(self):
        return fake_gpus(self.gpus, total_mib=self.gpu_bytes >> 20)


def seed_exl3_index(hf, model, revision="fixture", codebook="mcg"):
    root = Path(hf) / "hub" / ("models--" + model.replace("/", "--"))
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    stem = "model.layers.1.mlp.experts.0.gate_proj"
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            f"{stem}.trellis": "model-00001-of-00001.safetensors",
            f"{stem}.{codebook}": "model-00001-of-00001.safetensors",
        }
    }))
    if not (root / "refs/main").is_file():
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "refs/main").write_text(revision)
    return snapshot


def seed_checkpoint_metadata(hf):
    """Build-scope tests consume a shared miniature hub, never installed models."""
    import copy
    from test_example_configs import CHECKPOINTS, DSV4, GLM5, GLMF, MIMO, QWEN
    checkpoints = copy.deepcopy(CHECKPOINTS)
    aliases = {
        "wrldsuksgo2mars/DeepSeek-V4-Pro-EXL3-K2-Core-K6": {**DSV4, "quantization_config": {"quant_method": "exl3"}},
        "wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC": GLM5,
        "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC": GLMF,
        "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC-MUL1": GLMF,
        "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-MC": GLMF,
        "wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC": {"model_type": "step5_vl", "text_config": {
            "model_type": "step5", "num_hidden_layers": 92, "moe_layer_list": list(range(3, 91))}},
        "wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-MC": {"model_type": "muse_spark", "text_config": {
            "num_hidden_layers": 62, "mlp_layer_types": ["sparse"] * 62}},
        "wrldsuksgo2mars/Muse-Spark-Open-EXL3-K4-Core-K6": {"model_type": "muse_spark", "text_config": {
            "num_hidden_layers": 62, "mlp_layer_types": ["sparse"] * 62}},
        "wrldsuksgo2mars/Kimi-K3-EXL3-K2": {"model_type": "kimi_k3", "text_config": {
            "num_hidden_layers": 93, "first_k_dense_replace": 1}},
        "wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC": {"model_type": "kimi_k31", "text_config": {
            "model_type": "kimi_linear", "num_hidden_layers": 93, "first_k_dense_replace": 1}},
        "MiniMaxAI/MiniMax-M3.1-Flash": {"model_type": "minimax_m3_vl", "text_config": {
            "num_hidden_layers": 3, "moe_layer_freq": [0, 1, 1]}},
        "thinkingmachines/Inkling-2-Small-MXFP8": {"model_type": "inkling2", "text_config": {
            "num_hidden_layers": 48}, "quantization_config": {"quant_method": "mxfp8"}},
        "thinkingmachines/Inkling-2-Small": {"model_type": "inkling2", "text_config": {
            "num_hidden_layers": 48}, "quantization_config": {"quant_method": "nvfp4"}},
        "thinkingmachines/Inkling-2-Mini": {"model_type": "inkling2_loop", "text_config": {
            "num_hidden_layers": 24}, "quantization_config": {"quant_method": "nvfp4"}},
        "wrldsuksgo2mars/Inkling-2-Small-EXL3-K4-MC": {"model_type": "inkling2", "text_config": {
            "num_hidden_layers": 48}, "quantization_config": {"quant_method": "exl3"}},
        "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC": {"model_type": "hy_v4_flash", "text_config": {
            "num_hidden_layers": 42, "mlp_layer_types": ["dense"] + ["sparse"] * 41},
            "quantization_config": {"quant_method": "exl3"}},
        "wrldsuksgo2mars/Hy4-EXL3-K4-MC": {"model_type": "hy_v4_vl", "text_config": {
            "num_hidden_layers": 78, "mlp_layer_types": ["dense"] + ["sparse"] * 77},
            "quantization_config": {"quant_method": "exl3"}},
    }
    checkpoints.update(copy.deepcopy(aliases))
    for name, config in checkpoints.items():
        config = copy.deepcopy(config)
        if "NVFP4" in name:
            config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        if "EXL3" in name:
            config["quantization_config"] = {"quant_method": "exl3"}
        if config.get("model_type") == "qwen3_5":
            wide = "3.8" in name
            text = config["text_config"]
            text.update(hidden_size=5120 if wide else 4096, intermediate_size=17408 if wide else 12288,
                        num_hidden_layers=64 if wide else 32, num_attention_heads=24 if wide else 16,
                        num_key_value_heads=4, linear_num_key_heads=16, linear_num_value_heads=48 if wide else 32)
            text["layer_types"] = ["linear_attention"] * (text["num_hidden_layers"] - 1) + ["full_attention"]
        root = Path(hf) / "hub" / ("models--" + name.replace("/", "--"))
        snapshot = root / "snapshots/fixture"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text(json.dumps(config))
        (root / "refs").mkdir(exist_ok=True)
        (root / "refs/main").write_text("fixture")
        if "EXL3" in name or (config.get("quantization_config") or {}).get("quant_method") == "exl3":
            codebook = "mul1" if (config.get("model_type") in ("hy_v4_flash", "hy_v4_vl") or "MUL1" in name) else "mcg"
            seed_exl3_index(hf, name, revision="fixture", codebook=codebook)
    repo = Path(__file__).resolve().parents[2]
    for path in (repo / "examples/configs").glob("*.config"):
        values = dict(line.split("=", 1) for line in path.read_text().splitlines()
                      if line and not line.startswith("#") and "=" in line)
        model, revision = values.get("MODEL_ID"), values.get("MODEL_REVISION")
        if model in checkpoints and revision:
            root = Path(hf) / "hub" / ("models--" + model.replace("/", "--")) / "snapshots"
            if not (root / revision).exists():
                (root / revision).symlink_to("fixture", target_is_directory=True)
