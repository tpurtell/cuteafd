"""CPU-only coverage of the shared vision module registry and owner lifetime."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vision_owner_registry", ROOT / "scripts/build/vision-owner-registry.py")
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)


def header(path, name, *, pointers=8, scalar="float"):
    symbol = "cuteafd_" + name
    args = [symbol + "_Kernel_Module_t *module"] + [f"void *p{i}" for i in range(pointers)]
    args += [scalar + " value", "cudaStream_t stream"]
    path.write_text(f"""
#pragma once
#include <cuda_runtime.h>
struct {symbol}_Kernel_Module_t {{ cudaLibrary_t module; }};
extern "C" void _mlir_{symbol}_cuda_init(void **);
extern "C" void _mlir_{symbol}_cuda_load_to_device(void **);
extern "C" void _mlir_{symbol}_entry(void **args, int32_t num_args);
static inline int32_t cute_dsl_{symbol}_wrapper({', '.join(args)}) {{ return 0; }}
""")


def entry(directory, dim=64, programs=None, **kw):
    return dict(head_dim=dim, windowed=False, pointwise_group="", directory=str(directory),
                programs=programs or [f"vision_attention_d{dim}"], **kw)


def test_registry_validates_and_binds_both_export_abis(tmp_path):
    for dim in [64, 72]:
        header(tmp_path / f"vision_attention_d{dim}.h", f"vision_attention_d{dim}")
    header(tmp_path / "vision_attention_d96_full.h", "vision_attention_d96_full", pointers=6)
    header(tmp_path / "vision_axial_rotary.h", "vision_axial_rotary", pointers=6, scalar="int32_t")
    entries = [entry(tmp_path), entry(tmp_path, 72), entry(tmp_path, 96,
        programs=["vision_attention_d96_full", "vision_axial_rotary"])]
    text = registry.registry(entries)
    assert '_mlir_cuteafd_vision_attention_d64_entry, 8, "f"' in text
    assert '_mlir_cuteafd_vision_attention_d96_full_entry, 6, "f"' in text
    assert '_mlir_cuteafd_vision_axial_rotary_entry, 6, "i"' in text
    with pytest.raises(ValueError, match="duplicate"):
        registry.registry(entries + entries[:1])
    with pytest.raises(ValueError, match="varlen attention"):
        registry.registry([entry(tmp_path, programs=["vision_axial_rotary"])])
    with pytest.raises(ValueError, match="invalid vision program"):
        registry.registry([entry(tmp_path, programs=["../escape"])])


def test_cmake_entry_helper_preserves_legacy_export_command(tmp_path):
    cmake = shutil.which("cmake")
    assert cmake, "CMake is required for the native owner CPU gate"
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.24)
project(vision_owner NONE)
set(CUTEAFD_ENABLE_CUDA ON)
set(CUTEAFD_CUDA_ARCHITECTURES 120)
set(Python3_EXECUTABLE python3)
add_custom_target(cuteafd_verify_sparkinfer_source)
include(\"{ROOT}/native/cmake/shared/vision_attention.cmake\")
""")
    result = subprocess.run([cmake, "-S", str(source), "-B", str(tmp_path / "build")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    entries = json.loads((tmp_path / "build/vision_attention/entries.json").read_text())
    assert [(e["head_dim"], e["windowed"], e["pointwise_group"], e["programs"]) for e in entries] == [
        (64, False, "", ["vision_attention_d64"]), (72, False, "", ["vision_attention_d72"])]
    command = (tmp_path / "build/CMakeFiles/cuteafd_vision_attention_export.dir/build.make").read_text()
    exports = [line for line in command.splitlines() if "-E env" in line and "export_b12x_vision_aot.py" in line]
    assert len(exports) == 1
    assert exports[0].endswith("--output-dir " + str(tmp_path / "build/vision_attention") + " --capability 120")


def test_native_table_lifetimes_failures_and_dispatch_without_cuda(tmp_path):
    compiler = shutil.which("c++")
    assert compiler, "a C++ compiler is required for the native owner CPU gate"
    (tmp_path / "cuda_runtime.h").write_text("""
#pragma once
#include <stdint.h>
using cudaLibrary_t=void*;using cudaStream_t=void*;
constexpr int cudaErrorInvalidValue=1,cudaErrorInvalidDevice=2,cudaErrorMemoryAllocation=3,
 cudaErrorInitializationError=4,cudaErrorInvalidResourceHandle=5;
extern int device,unloads,fail_unload;
inline int cudaGetDevice(int* d) {*d=device;return 0;}
inline int cudaLibraryUnload(void*) {++unloads;return fail_unload;}
""")
    for dim in [64, 72]:
        header(tmp_path / f"vision_attention_d{dim}.h", f"vision_attention_d{dim}")
    header(tmp_path / "vision_attention_d96_full.h", "vision_attention_d96_full", pointers=6)
    header(tmp_path / "vision_axial_rotary.h", "vision_axial_rotary", pointers=6, scalar="int32_t")
    modern = entry(tmp_path, 96, programs=["vision_attention_d96_full", "vision_axial_rotary"])
    modern["pointwise_group"] = "d96_f32"
    (tmp_path / "vision_owner_entries.h").write_text(registry.registry([entry(tmp_path), entry(tmp_path, 72), modern]))
    fixture = """
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include "cuteafd_vision_attention_internal.h"
int device=0,unloads=0,fail_unload=0,init[2]{},loads[2]{},calls[2]{},fail=0;
void initialize(void** a,int i) {++init[i];**static_cast<void***>(a[0])=reinterpret_cast<void*>(uintptr_t(i+1));*static_cast<int*>(a[1])=0;}
void load(void** a,int i) {++loads[i];*static_cast<int*>(a[2])=fail;}
void launch(void** a,int n,int i) {assert(n==11);assert(*static_cast<float*>(a[8])==0.5f);++calls[i];*static_cast<int*>(a[10])=0;}
#define FIX(D,I) \
extern "C" void _mlir_cuteafd_vision_attention_d##D##_cuda_init(void** a){initialize(a,I);} \
extern "C" void _mlir_cuteafd_vision_attention_d##D##_cuda_load_to_device(void** a){load(a,I);} \
extern "C" void _mlir_cuteafd_vision_attention_d##D##_entry(void** a,int n){launch(a,n,I);}
FIX(64,0)
FIX(72,1)
extern "C" void _mlir_cuteafd_vision_attention_d96_full_cuda_init(void** a){initialize(a,0);}
extern "C" void _mlir_cuteafd_vision_attention_d96_full_cuda_load_to_device(void** a){load(a,0);}
extern "C" void _mlir_cuteafd_vision_attention_d96_full_entry(void** a,int n){
 assert(n==9);assert(*static_cast<float*>(a[6])==.5f);*static_cast<int*>(a[8])=0;}
extern "C" void _mlir_cuteafd_vision_axial_rotary_cuda_init(void** a){initialize(a,0);}
extern "C" void _mlir_cuteafd_vision_axial_rotary_cuda_load_to_device(void** a){load(a,0);}
extern "C" void _mlir_cuteafd_vision_axial_rotary_entry(void** a,int n){
 assert(n==9);assert(*static_cast<int32_t*>(a[6])==17);*static_cast<int*>(a[8])=0;}
int main(){
 void *a=nullptr,*b=nullptr,*c=nullptr;
 assert(cuteafd_vision_attention_create(64,&a)==0);
 assert(cuteafd_vision_attention_create(64,&b)==0);
 assert(init[0]==1 && loads[0]==1);
 device=1;assert(cuteafd_vision_attention_create(64,&c)==0);assert(loads[0]==2);
 assert(cuteafd_vision_attention_launch(a,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,.5f,nullptr)==cudaErrorInvalidDevice);
 assert(cuteafd_vision_attention_launch(c,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,.5f,nullptr)==0);
 assert(calls[0]==1);
 assert(cuteafd_vision_attention_destroy(a)==0 && unloads==0);
 assert(cuteafd_vision_attention_destroy(b)==0 && unloads==0);
 assert(cuteafd_vision_attention_destroy(c)==0 && unloads==1);
 fail=19;assert(cuteafd_vision_attention_create(72,&a)==19 && a==nullptr);assert(unloads==2);
 fail=0;assert(cuteafd_vision_attention_create(72,&a)==0);assert(init[1]==2);
 fail_unload=23;assert(cuteafd_vision_attention_destroy(a)==23);
 fail_unload=0;assert(cuteafd_vision_attention_create(72,&a)==0);assert(init[1]==2);
 assert(cuteafd_vision_attention_destroy(a)==0);
 assert(cuteafd_vision_attention_create_entry(64,1,"",&a)==cudaErrorInvalidValue && a==nullptr);
 assert(cuteafd_vision_attention_create_entry(64,0,"missing",&a)==cudaErrorInvalidValue);
 assert(cuteafd_vision_attention_create(96,&a)==cudaErrorInvalidValue);
 assert(cuteafd_vision_attention_create_entry(96,0,"d96_f32",&a)==0);
 assert(cuteafd_vision_attention_launch(a,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,.5f,nullptr)==0);
 void* ptrs[6]{};uint64_t scalar=17;
 assert(cuteafd_vision_pointwise_launch(a,"vision_axial_rotary",ptrs,&scalar,nullptr)==0);
 assert(cuteafd_vision_pointwise_launch(a,"missing",ptrs,&scalar,nullptr)==cudaErrorInvalidValue);
 assert(cuteafd_vision_pointwise_launch(a,"vision_axial_rotary",ptrs,nullptr,nullptr)==cudaErrorInvalidValue);
 assert(cuteafd_vision_attention_destroy(a)==0);
 device=32;assert(cuteafd_vision_attention_create(64,&a)==cudaErrorInvalidDevice);
 assert(cuteafd_vision_attention_destroy(nullptr)==0);
}
"""
    (tmp_path / "fixture.cc").write_text(fixture)
    executable = tmp_path / "test-owner"
    result = subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(tmp_path), "-I",
        str(ROOT / "native/shared/include"), str(ROOT / "native/shared/src/vision_attention_aot.cc"),
        str(tmp_path / "fixture.cc"), "-o", str(executable)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(executable)], check=True, timeout=10)
