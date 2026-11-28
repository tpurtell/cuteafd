"""programs-check.sh must load a CUDA-linked native library in a CPU-only build container.

AOT exports run with no CUDA driver installed, but libcuteafd_native.so links
libcuda.so.1, so `cuteafd programs-check` failed to load it there (v7.0.0-rc1 Spark
leg: `libcuda.so.1: cannot open shared object file`). The helper resolves the CUDA
toolkit's stub when no real driver is present.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/build/programs-check.sh"
STUB = Path("/usr/local/cuda/lib64/stubs/libcuda.so")


def driver_installed() -> bool:
    result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    return "libcuda.so.1 " in result.stdout


def build_fixture(tmp: Path) -> tuple[Path, Path]:
    """A native library that links libcuda.so.1 and a fake `cuteafd` that dlopens it."""
    lib_src = tmp / "native.c"
    lib_src.write_text("int cuInit(unsigned int flags);\nint native_probe(void) { return (int)(long)&cuInit; }\n")
    native = tmp / "libcuteafd_native.so"
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(native), str(lib_src),
                    f"-L{STUB.parent}", "-lcuda"], check=True)
    bin_src = tmp / "fake.c"
    bin_src.write_text(
        "#include <dlfcn.h>\n#include <stdio.h>\n#include <string.h>\n"
        "int main(int argc, char **argv) {\n"
        "  const char *lib = 0;\n"
        "  for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], \"--native-lib\")) lib = argv[i + 1];\n"
        "  if (!lib || strcmp(argv[1], \"programs-check\")) return 3;\n"
        "  if (!dlopen(lib, RTLD_NOW)) { fprintf(stderr, \"%s\\n\", dlerror()); return 1; }\n"
        "  puts(\"programs-check: loaded\");\n  return 0;\n}\n")
    binary = tmp / "cuteafd"
    subprocess.run(["cc", "-o", str(binary), str(bin_src), "-ldl"], check=True)
    return binary, native


@pytest.mark.skipif(not STUB.exists() or shutil.which("cc") is None,
                    reason="needs the CUDA toolkit stub and a C compiler")
@pytest.mark.skipif(driver_installed(), reason="a real CUDA driver is installed; the stub path is not exercised")
def test_helper_loads_cuda_linked_library_without_a_driver(tmp_path):
    binary, native = build_fixture(tmp_path)
    manifest = tmp_path / "PROGRAMS.json"
    manifest.write_text("{}")
    # Without the helper the library cannot load: this is the rc1 failure.
    direct = subprocess.run([str(binary), "programs-check", "--native-lib", str(native)],
                            capture_output=True, text=True)
    assert direct.returncode == 1 and "libcuda.so.1" in direct.stderr
    result = subprocess.run([str(HELPER), str(binary), str(native), str(manifest), "spark", "glm;glmf"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "programs-check: loaded" in result.stdout


def test_helper_rejects_unknown_role(tmp_path):
    result = subprocess.run([str(HELPER), "/bin/true", "lib.so", "m.json", "gpu", "glm"],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "role must be coordinator or spark" in result.stderr
