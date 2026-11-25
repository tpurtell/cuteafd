"""CPU-only coverage of the shared vision module registry and owner lifetime."""
import importlib.util
import re
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("vision_owner_registry", ROOT / "scripts/build/vision-owner-registry.py")
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)


def header(path, name, *, pointers=8, scalars=("float",)):
    symbol = "cuteafd_" + name
    args = [symbol + "_Kernel_Module_t *module"] + [f"void *p{i}" for i in range(pointers)]
    args += [f"{kind} value{i}" for i, kind in enumerate(scalars)] + ["cudaStream_t stream"]
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
        header(tmp_path / f"vision_attention_d{dim}.h", f"vision_attention_d{dim}",
               scalars=("float", "int32_t"))
    header(tmp_path / "vision_attention_d96_full.h", "vision_attention_d96_full", pointers=6,
           scalars=("float", "int32_t"))
    header(tmp_path / "vision_axial_rotary.h", "vision_axial_rotary", pointers=6, scalars=("int32_t",))
    entries = [entry(tmp_path), entry(tmp_path, 72), entry(tmp_path, 96,
        programs=["vision_attention_d96_full", "vision_axial_rotary"])]
    text = registry.registry(entries)
    assert '_mlir_cuteafd_vision_attention_d64_entry, 8, "fi"' in text
    assert '_mlir_cuteafd_vision_attention_d96_full_entry, 6, "fi"' in text
    assert '_mlir_cuteafd_vision_axial_rotary_entry, 6, "i"' in text
    header(tmp_path / "vision_dynamic_attention.h", "vision_dynamic_attention", pointers=6,
           scalars=("int32_t","float","int32_t"))
    dynamic=registry.registry([entry(tmp_path,128,programs=["vision_dynamic_attention"])])
    assert '_mlir_cuteafd_vision_dynamic_attention_entry, 6, "ifi"' in dynamic
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
    assert [(e["head_dim"], e["windowed"], e["pointwise_group"]) for e in entries] == [
        (64, False, ""), (72, False, ""), (96, False, "muse"), (96, True, "muse"), (128, False, "pixtral"),
        (64, False, "pe_inkling2"), (64, False, "pe_inkling2_mini"), (96, False, "pe_step5"),
        (128, False, "tower_kimi"), (80, False, "tower_minimax"), (72, False, "tower_hy4f"), (96, False, "tower_hy4")]
    assert entries[0]["programs"] == ["vision_attention_d64"]
    assert entries[1]["programs"] == ["vision_attention_d72"]
    exporter = ROOT / "python/tools/aot/export_b12x_vision_aot.py"
    inventory = subprocess.check_output(["python3", str(exporter), "--list-transformer-programs"], text=True).strip().split(";")
    assert len(inventory) == 14
    for item, mode in zip(entries[2:4], ("full", "windows"), strict=True):
        assert item["programs"][0] == f"vision_attention_d96_{mode}"
        assert [p["name"] for p in item["programs"][1:]] == [
            "vision_axial_rotary_d96_f32", "vision_position_gather_h1536", *inventory]
    assert entries[2]["programs"][1:] == entries[3]["programs"][1:]
    rules = (tmp_path / "build/CMakeFiles/cuteafd_vision_attention_export.dir/build.make").read_text()
    exports = [line for line in rules.splitlines() if "-E env" in line and "export_b12x_vision_aot.py" in line]
    assert len(exports) == 18, exports  # Step reuses Muse; inkling2_mini reuses inkling2; each generic tower adds two groups
    legacy = [line for line in exports if "--group" not in line]
    assert len(legacy) == 1
    assert legacy[0].endswith("--output-dir " + str(tmp_path / "build/vision_attention") + " --capability 120")
    full = next(line for line in exports if "/d96_full " in line)
    windows = next(line for line in exports if "/d96_windows " in line)
    assert "--windowed" not in full
    assert windows.endswith("--group attention --head-dim 96 --windowed")
    pixtral = entries[4]
    assert pixtral["programs"][0] == "vision_attention_d128_full"
    assert [p["name"] for p in pixtral["programs"][1:]] == [
        "vision_axial_rotary_d128_bf16", "vision_rms_bf16_gamma_h2048"]
    for item, kind, dim in zip(entries[5:8], ("inkling2", "inkling2_mini", "step5"), (64, 64, 96), strict=True):
        inventory = subprocess.check_output(["python3", str(exporter), "--list-perception-programs", "--perception", kind], text=True).strip().split(";")
        assert item["programs"][0] == f"vision_attention_d{dim}_full"
        assert [p["name"] for p in item["programs"][1:]] == inventory
        assert len(inventory) == (14 if kind in ("inkling2", "inkling2_mini") else 20)
    inkling_export = next(line for line in exports if "--group perception --perception inkling2 " in line or line.endswith("--group perception --perception inkling2"))
    assert "--omit-program vision_gelu_h4096" in inkling_export
    assert entries[5]["programs"][-1]["directory"].endswith("pointwise_muse")
    assert "vision_pe_patch_projection_h4096_o4096" in [p["name"] for p in entries[5]["programs"][1:]]
    assert "vision_projection_h4096_o2304" in [p["name"] for p in entries[6]["programs"][1:]]
    assert "vision_pe_patch_projection_h6144_o4096" in [p["name"] for p in entries[7]["programs"][1:]]
    step = entries[7]
    assert step["directory"] == entries[2]["directory"]
    borrowed = [p for p in step["programs"][1:] if p["directory"].endswith("pointwise_muse")]
    assert [p["name"] for p in borrowed] == [p["name"] for p in entries[2]["programs"][1:] if p in borrowed]
    assert len(borrowed) == 4
    step_export = next(line for line in exports if "--group perception --perception step5" in line)
    assert step_export.count("--omit-program") == 4
    for program in borrowed:
        assert f'--omit-program {program["name"]}' in step_export
        assert not any(Path(p).name == program["name"] + ".o" and "pointwise_pe_step5" in p
                       for p in rules.split())
    for item, kind, heads, dim in zip(entries[8:], ("kimi", "minimax", "hy4f", "hy4"), (12, 16, 16, 16), (128, 80, 72, 96), strict=True):
        inventory = subprocess.check_output(["python3", str(exporter), "--list-tower-programs", "--tower", kind], text=True).strip().split(";")
        assert item["programs"][0] == f"vision_dynamic_attention_h{heads}_d{dim}"
        assert [p["name"] for p in item["programs"][1:]] == inventory[1:]
        borrowed = [p for p in item["programs"][1:] if not p["directory"].endswith(f"pointwise_tower_{kind}")]
        expected_borrowed = (
            ["vision_projection_h4096_o4096", "vision_gelu_h4096"] if kind == "kimi" else
            ["vision_patches_raster_p16_t1", "vision_residual_h1536", "vision_gelu_h6144",
             "vision_biased_projection_h1536_o1536", "vision_biased_projection_h6144_o6144"] if kind == "hy4" else []
        )
        assert [p["name"] for p in borrowed] == expected_borrowed
        pointwise = next(line for line in exports if f"/pointwise_tower_{kind} " in line)
        assert f"--omit-program {inventory[0]}" in pointwise
        for program in borrowed:
            assert f"--omit-program {program['name']}" in pointwise
    legacy_rule = rules.split("vision_attention/vision_attention_d64.h:", 1)[1].split("vision_attention/vision_attention_d64.o:", 1)[0]
    assert "--group attention" not in legacy_rule
    assert "--group pointwise" not in legacy_rule


def test_cmake_rejects_borrowing_an_undeclared_program(tmp_path):
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
cuteafd_vision_owner_entry(bad HEAD_DIM 64 POINTWISE_GROUP invalid
  PROGRAMS vision_attention_d64_full ARGS --group attention --head-dim 64
  POINTWISE_PROGRAMS vision_missing POINTWISE_ARGS --group transformer
  BORROW_POINTWISE_GROUP muse BORROW_POINTWISE_PROGRAMS vision_missing)
""")
    result = subprocess.run(["cmake", "-S", str(source), "-B", str(tmp_path / "build")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "does not export vision_missing" in result.stderr


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
inline int cudaDeviceGetAttribute(int* v,int,int) {*v=50331648;return 0;}
constexpr int cudaDevAttrL2CacheSize=104;
inline int cudaLibraryUnload(void*) {++unloads;return fail_unload;}
""")
    for dim in [64, 72]:
        header(tmp_path / f"vision_attention_d{dim}.h", f"vision_attention_d{dim}",
               scalars=("float", "int32_t"))
    header(tmp_path / "vision_attention_d96_full.h", "vision_attention_d96_full", pointers=6,
           scalars=("float", "int32_t"))
    header(tmp_path / "vision_axial_rotary.h", "vision_axial_rotary", pointers=6, scalars=("int32_t",))
    modern = entry(tmp_path, 96, programs=["vision_attention_d96_full", "vision_axial_rotary"])
    modern["pointwise_group"] = "d96_f32"
    exporter = ROOT / "python/tools/aot/export_b12x_vision_aot.py"
    generic_entries = []
    generic_names = []
    for kind, dim in (("kimi", 128), ("minimax", 80)):
        names = subprocess.check_output(["python3", str(exporter), "--list-tower-programs", "--tower", kind], text=True).strip().split(";")
        generic_names.append(names)
        for index, name in enumerate(names):
            header(tmp_path / (name + ".h"), name, pointers=6 if index == 0 else 4,
                   scalars=("int32_t", "float", "int32_t") if index == 0 else ("int32_t",))
        item = entry(tmp_path, dim, programs=names)
        item["pointwise_group"] = "tower_" + kind
        generic_entries.append(item)
    (tmp_path / "vision_owner_entries.h").write_text(registry.registry([entry(tmp_path), entry(tmp_path, 72), modern, *generic_entries]))
    fixture = """
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include "cuteafd_vision_attention_internal.h"
int device=0,unloads=0,fail_unload=0,init[2]{},loads[2]{},calls[2]{},fail=0;
void initialize(void** a,int i) {++init[i];**static_cast<void***>(a[0])=reinterpret_cast<void*>(uintptr_t(i+1));*static_cast<int*>(a[1])=0;}
void load(void** a,int i) {++loads[i];*static_cast<int*>(a[2])=fail;}
void launch(void** a,int n,int i) {assert(n==12);assert(*static_cast<float*>(a[8])==0.5f);
 assert(*static_cast<int32_t*>(a[9])==50331648);++calls[i];*static_cast<int*>(a[11])=0;}
#define FIX(D,I) \
extern "C" void _mlir_cuteafd_vision_attention_d##D##_cuda_init(void** a){initialize(a,I);} \
extern "C" void _mlir_cuteafd_vision_attention_d##D##_cuda_load_to_device(void** a){load(a,I);} \
extern "C" void _mlir_cuteafd_vision_attention_d##D##_entry(void** a,int n){launch(a,n,I);}
FIX(64,0)
FIX(72,1)
extern "C" void _mlir_cuteafd_vision_attention_d96_full_cuda_init(void** a){initialize(a,0);}
extern "C" void _mlir_cuteafd_vision_attention_d96_full_cuda_load_to_device(void** a){load(a,0);}
extern "C" void _mlir_cuteafd_vision_attention_d96_full_entry(void** a,int n){
 assert(n==10);assert(*static_cast<float*>(a[6])==.5f);
 assert(*static_cast<int32_t*>(a[7])==50331648);*static_cast<int*>(a[9])=0;}
extern "C" void _mlir_cuteafd_vision_axial_rotary_cuda_init(void** a){initialize(a,0);}
extern "C" void _mlir_cuteafd_vision_axial_rotary_cuda_load_to_device(void** a){load(a,0);}
extern "C" void _mlir_cuteafd_vision_axial_rotary_entry(void** a,int n){
 assert(n==9);assert(*static_cast<int32_t*>(a[6])==17);*static_cast<int*>(a[8])=0;}
GENERIC_PROGRAMS
int main(){
 GENERIC_CHECKS
 void *a=nullptr,*b=nullptr,*c=nullptr;
 assert(cuteafd_vision_attention_create_entry(64,0,"",&a)==0);
 assert(cuteafd_vision_attention_create_entry(64,0,"",&b)==0);
 assert(init[0]==1 && loads[0]==1);
 device=1;assert(cuteafd_vision_attention_create_entry(64,0,"",&c)==0);assert(loads[0]==2);
 assert(cuteafd_vision_attention_launch(a,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,.5f,nullptr)==cudaErrorInvalidDevice);
 assert(cuteafd_vision_attention_launch(c,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,.5f,nullptr)==0);
 assert(calls[0]==1);
 assert(cuteafd_vision_attention_destroy(a)==0 && unloads==0);
 assert(cuteafd_vision_attention_destroy(b)==0 && unloads==0);
 assert(cuteafd_vision_attention_destroy(c)==0 && unloads==1);
 fail=19;assert(cuteafd_vision_attention_create_entry(72,0,"",&a)==19 && a==nullptr);assert(unloads==2);
 fail=0;assert(cuteafd_vision_attention_create_entry(72,0,"",&a)==0);assert(init[1]==2);
 fail_unload=23;assert(cuteafd_vision_attention_destroy(a)==23);
 fail_unload=0;assert(cuteafd_vision_attention_create_entry(72,0,"",&a)==0);assert(init[1]==2);
 assert(cuteafd_vision_attention_destroy(a)==0);
 assert(cuteafd_vision_attention_create_entry(64,1,"",&a)==cudaErrorInvalidValue && a==nullptr);
 assert(cuteafd_vision_attention_create_entry(64,0,"missing",&a)==cudaErrorInvalidValue);
 assert(cuteafd_vision_attention_create_entry(96,0,"",&a)==cudaErrorInvalidValue);
 assert(cuteafd_vision_attention_create_entry(96,0,"d96_f32",&a)==0);
 assert(cuteafd_vision_attention_launch(a,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,.5f,nullptr)==0);
 void* ptrs[6]{};uint64_t scalar=17;
 assert(cuteafd_vision_pointwise_launch(a,"vision_axial_rotary",ptrs,&scalar,nullptr)==0);
 assert(cuteafd_vision_pointwise_launch(a,"missing",ptrs,&scalar,nullptr)==cudaErrorInvalidValue);
 assert(cuteafd_vision_pointwise_launch(a,"vision_axial_rotary",ptrs,nullptr,nullptr)==cudaErrorInvalidValue);
 assert(cuteafd_vision_attention_destroy(a)==0);
 device=32;assert(cuteafd_vision_attention_create_entry(64,0,"",&a)==cudaErrorInvalidDevice);
 assert(cuteafd_vision_attention_destroy(nullptr)==0);
}
"""
    definitions, checks, seen = [], [], set()
    for kind, dim, names in zip(("kimi", "minimax"), (128, 80), generic_names, strict=True):
        for index, name in enumerate(names):
            if name in seen:
                continue
            seen.add(name)
            validation = ('assert(n==11);assert(*static_cast<int32_t*>(a[6])==16);'
                          'assert(*static_cast<float*>(a[7])==.5f);'
                          'assert(*static_cast<int32_t*>(a[8])==50331648);*static_cast<int*>(a[10])=0;' if index == 0 else
                          'assert(n==7);*static_cast<int*>(a[6])=0;')
            definitions.append(f'''extern "C" void _mlir_cuteafd_{name}_cuda_init(void** a){{initialize(a,0);}}
extern "C" void _mlir_cuteafd_{name}_cuda_load_to_device(void** a){{load(a,0);}}
extern "C" void _mlir_cuteafd_{name}_entry(void** a,int n){{{validation}}}''')
        launch_checks = '\n'.join(f'assert(cuteafd_vision_pointwise_launch(g,"{name}",ptrs,&scalar,nullptr)==0);' for name in names[1:])
        checks.append(f'''{{void* g=nullptr;assert(cuteafd_vision_attention_create_entry({dim},0,"tower_{kind}",&g)==0);
void* ptrs[6]{{}};uint64_t scalar=16;
assert(cuteafd_vision_pointwise_launch(g,"{names[0]}",ptrs,&scalar,nullptr)==cudaErrorInvalidValue);
assert(cuteafd_vision_dynamic_attention_launch(g,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,16,.5f,nullptr)==0);
assert(cuteafd_vision_dynamic_attention_launch(g,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,.5f,nullptr)==cudaErrorInvalidValue);
assert(cuteafd_vision_attention_launch(g,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,.5f,nullptr)==cudaErrorInvalidValue);
{launch_checks}
assert(cuteafd_vision_attention_destroy(g)==0);}}
unloads=0;init[0]=0;loads[0]=0;''')
    fixture = fixture.replace("GENERIC_PROGRAMS", '\n'.join(definitions)).replace("GENERIC_CHECKS", '\n'.join(checks))
    (tmp_path / "fixture.cc").write_text(fixture)
    executable = tmp_path / "test-owner"
    result = subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(tmp_path), "-I",
        str(ROOT / "native/shared/include"), str(ROOT / "native/shared/src/vision_attention_aot.cc"),
        str(tmp_path / "fixture.cc"), "-o", str(executable)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(executable)], check=True, timeout=10)


def test_inactive_generic_tower_inventory_is_unique_and_has_dynamic_attention_first():
    path=ROOT / "python/tools/aot/vision_tower_programs.py"
    spec=importlib.util.spec_from_file_location("generic_tower_inventory",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for kind,count in [("kimi",18),("minimax",16)]:
        names=module.names(kind)
        assert len(names)==count and len(set(names))==count
        assert names[0] == ("vision_dynamic_attention_h12_d128" if kind == "kimi" else "vision_dynamic_attention_h16_d80")
        assert not any("65536" in name or "66792" in name for name in names)
        assert ("vision_vendor_rotary_raster_t1" in names) == (kind=="kimi")
        assert "vision_rotary_raster_t1" not in names
        assert not any("cu130" in name for name in names)
    # Every tower group links into one native library: unborrowed export names
    # must be disjoint across groups (Hy4 explicitly borrows shared objects via CMake).
    hy4_borrowed = {
        "vision_patches_raster_p16_t1", "vision_residual_h1536", "vision_gelu_h6144",
        "vision_biased_projection_h1536_o1536", "vision_biased_projection_h6144_o6144",
    }
    groups={kind:set(module.names(kind)) for kind in ("kimi","minimax","hy4f","hy4")}
    owned={kind:(names - hy4_borrowed if kind == "hy4" else names) for kind, names in groups.items()}
    for a in owned:
        for b in owned:
            assert a>=b or not owned[a]&owned[b], f"{a}/{b} share exports {sorted(owned[a]&owned[b])}"
    hy4=(ROOT / "native/shared/cuda/vision_hy4.cuh").read_text()
    # Literal names (or the literal prefix of a composed one) must resolve to a Hy4 export.
    launched=set(re.findall(r'generic_program\(o,"(vision_\w+)"',hy4))
    assert "vision_patches_raster_p16_t1" in launched
    for kind in ("hy4f", "hy4"):
        assert all(any(name==prefix or name.startswith(prefix) for name in groups[kind]) for prefix in launched), (kind, launched)
    native=(ROOT / "native/shared/cuda/vision_runtime.cu").read_text()
    assert "if(s.tower)return false" in native
    assert "return vision_required_impl(s,out,valid_tower);" in native
    assert "return vision_create_impl(s,device,admitted,tags,out,valid_tower);" in native
    seam = native.split('extern "C" int32_t cuteafd_internal_vision_tower_oracle', 1)[1].split('extern "C"', 1)[0]
    assert "vision_required_impl(spec,ledger,valid_generic_tower)" in seam
    assert "vision_create_impl(spec,device,admitted,tags,owner,valid_generic_tower)" in seam
    assert "scratch_generic_tower" not in seam and "encode_generic_tower" not in seam
    header = (ROOT / "native/shared/include/cuteafd_vision.h").read_text()
    assert "cuteafd_internal_vision_tower_oracle" not in header
    cmake = (ROOT / "native/CMakeLists.txt").read_text()
    assert "add_library(cuteafd_vision_tower_oracle" not in cmake
    assert "cuteafd_internal_vision_tower_teacher" not in header
    assert "using TowerTeacherInput = const uint16_t* (*)(void*, int32_t, int32_t, int32_t);" in native
    teacher = native.split('extern "C" int32_t cuteafd_internal_vision_tower_teacher', 1)[1].split('extern "C"', 1)[0]
    assert "if(!o->spec.tower)return cudaErrorInvalidValue;" in teacher
    assert "rc=drain(o,0);if(rc)return rc;" in teacher
    encode = (ROOT / "native/shared/cuda/vision_tower_encode.cuh").read_text()
    assert "if(!o->teacher_input)return 0;" in encode
    assert "cudaMemcpyHostToDevice,o->stream" in encode
    assert "return drain(o,rc);" in encode
    assert "cuteafd_vision_dynamic_attention_launch(o->attention" in encode
    assert 'generic_program(o,"vision_dynamic_attention' not in encode
    assert "generic_teacher_boundary(o,0,b.x,n,h)" in encode
    assert "if(layer+1<s.depth)GENERIC_RUN(generic_teacher_boundary(o,int(layer)+1,b.x,n,h));" in encode
