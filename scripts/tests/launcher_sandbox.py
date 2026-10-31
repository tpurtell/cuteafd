"""Session-owned, immutable launcher inputs; writable files never share inodes."""
from pathlib import Path
import shutil

_ROOT = None
_TEMPLATES = {}

LAUNCHER_SHARED_INPUTS = (
    "python/tools/lib/spark_packages.py",
    "scripts/build/spark-package-ranges.json",
)


def configure(root):
    global _ROOT
    _ROOT = root
    _TEMPLATES.clear()


def copy_input(source, destination):
    """Link a read-only input from a private session copy, not the checkout."""
    source, destination = Path(source), Path(destination)
    if destination.is_dir():
        destination /= source.name
    if source.name == "release-common.sh":
        copy_launcher_shared_inputs(source.resolve().parents[2], destination.parent.parent.parent)
    if _ROOT is None:
        return shutil.copy(source, destination)
    key = source.resolve()
    if key not in _TEMPLATES:
        cached = _ROOT / str(len(_TEMPLATES))
        shutil.copy(source, cached)
        _TEMPLATES[key] = cached
    destination.hardlink_to(_TEMPLATES[key])
    return str(destination)


def copy_launcher_shared_inputs(source_root, destination_root):
    """Stage the same relative Python/data paths used by the real source tree."""
    for relative in LAUNCHER_SHARED_INPUTS:
        source, target = Path(source_root) / relative, Path(destination_root) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            copy_input(source, target)


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
        cached = _ROOT / str(len(_TEMPLATES))
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
    path = Path(bin_dir) / "nvidia-smi"
    path.write_text("#!" + sys.executable + "\n" + FAKE_NVIDIA_SMI)
    path.chmod(0o755)
    return path
