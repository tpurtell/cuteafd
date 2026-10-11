"""Execute the real native bridges against stubbed CUDA/AOT launch entries."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import shutil
import subprocess
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = Path.home() / ".cache/cuteafd/builds/plat1-sm120/native-host-tests"

CUDA = r"""
#pragma once
using cudaLibrary_t = void*;
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorMemoryAllocation=2, cudaErrorInvalidDevice=101;
constexpr int cudaErrorInvalidResourceHandle=400, cudaErrorNoKernelImageForDevice=209;
constexpr int cudaDevAttrComputeCapabilityMajor=1, cudaDevAttrComputeCapabilityMinor=2,
              cudaDevAttrMultiProcessorCount=3;
inline int device=2, major=12, minor=0, recorded_grid=0, recorded_cap=0, attribute_error=0, override_sms=0;
inline int cudaGetDevice(int* out) { *out=device; return 0; }
inline int cudaDeviceGetAttribute(int* out, int attr, int dev) {
  if (attribute_error) return attribute_error;
  *out=attr==1 ? major : attr==2 ? minor : override_sms ? override_sms : dev==2 ? 170 : 188; return 0;
}
inline int cudaLibraryUnload(void*) { return 0; }
inline void init(void** args) { **static_cast<void***>(args[0])=reinterpret_cast<void*>(1); }
inline void load(void**) {}
inline void noop(void**, int) {}
inline void quant(void** args, int) { recorded_grid=*static_cast<int*>(args[5]); }
inline void grouped_quant(void** args, int) { recorded_grid=*static_cast<int*>(args[8]); }
inline void expert_launch(void** args, int) { recorded_cap=*static_cast<int*>(args[50]); }
"""

FP8_HEADER = r"""
#pragma once
#define CUTEAFD_V41_FP8_SMS 188
static const uint32_t grids[] = {752};
#define CUTEAFD_V41_FP8_VARIANTS \
 {{1,1,32,32,64,0,16,32,16},{init,load,quant},{init,load,noop},{init,load,noop},grids,0,1,1,0}, \
 {{1,1,256,32,64,0,16,32,16},{init,load,grouped_quant},{init,load,noop},{init,load,grouped_quant},grids,0,1,8,0}
#define CUTEAFD_V41_HC_PROJECT_MODULE {init,load,noop}
"""

FP8_MAIN = r"""
#include <cassert>
#include "SOURCE"
extern "C" int32_t cuteafd_v41_fp8_grouped_output(const uint16_t*,uint16_t*,int32_t,void*) { return 0; }
extern "C" int32_t cuteafd_v41_fp8_reduce_splits(const float*,uint16_t*,int32_t,int32_t,int32_t,void*) { return 0; }
extern "C" int32_t cuteafd_v41_fp8_initialize_storage(void*,uint64_t,float*,void*) { return 0; }
int main() {
 void* h=nullptr;
 minor=1;
 assert(cuteafd_v41_fp8_matrix_initialize(1,32,32,&h)==cudaErrorInvalidDevice && h==nullptr);
 minor=0;
 for (int dev : {2,5}) {
  device=dev;
  for (int k : {32,256}) {
   assert(cuteafd_v41_fp8_matrix_initialize(1,k,32,&h)==0);
   assert(handle(h)->device_sms==(dev==2 ? 170 : 188));
   cuteafd_v41_fp8_info_t info;
   assert(cuteafd_v41_fp8_matrix_info(1,k,32,&info)==0 && info.scratch_bytes==64);
   assert(cuteafd_v41_fp8_launch(h,reinterpret_cast<uint16_t*>(0x10000),
     reinterpret_cast<uint8_t*>(0x20000),reinterpret_cast<uint8_t*>(0x30000),
     reinterpret_cast<void*>(0x40000),64,reinterpret_cast<float*>(0x50000),
     reinterpret_cast<uint16_t*>(0x60000),1,nullptr)==0);
   assert(recorded_grid==(dev==2 ? 680 : 752));
  }
 }
 device=2;
 assert(cuteafd_v41_fp8_launch(h,reinterpret_cast<uint16_t*>(0x10000),
   reinterpret_cast<uint8_t*>(0x20000),reinterpret_cast<uint8_t*>(0x30000),
   reinterpret_cast<void*>(0x40000),64,reinterpret_cast<float*>(0x50000),
   reinterpret_cast<uint16_t*>(0x60000),1,nullptr)==cudaErrorInvalidDevice);
}
"""

EXPERT_HEADER = r"""
#pragma once
#define CUTEAFD_V41_SMS 188
#define CUTEAFD_V41_CC_MINOR 0
#define CUTEAFD_V41_VARIANTS {{2,0,1,5120,2304,2304,3,1,64,1,1,1,1,188,1},init,load,expert_launch,{}}
"""

INPUT_QUANT_HEADER = r"""
#pragma once
static const uint32_t cuteafd_v41_input_quant_grids[] = {752};
#define _mlir_cuteafd_v41_expert_input_quant_cuda_init ::init
#define _mlir_cuteafd_v41_expert_input_quant_cuda_load_to_device ::load
#define CUTEAFD_V41_INPUT_QUANT_ENTRY quant
"""

EXPERT_MAIN = r"""
#include <cassert>
#include "SOURCE"
extern "C" int32_t cuteafd_initialize_scratch_storage_async(void*,uint64_t,uint64_t,uint64_t,uint32_t,void*) { return 0; }
int main() {
 void* h=nullptr;
 minor=1;
 assert(cuteafd_v41_expert_initialize(1,&h)==cudaErrorInvalidDevice && h==nullptr);
 assert(cuteafd_v41_expert_input_quant_initialize(&h)==cudaErrorInvalidDevice && h==nullptr);
 minor=0;
 assert(cuteafd_v41_expert_initialize(1,&h)==0);
 assert(by_handle(h)->device_sms==170 && by_handle(h)->info.scratch_bytes==64);
 cuteafd_expert_launch_t args{};
 for (auto& p : args.tensors) p=reinterpret_cast<void*>(0x10000);
 args.num_tokens=1; args.scatter_rows=3; args.max_rows=1;
 args.rows_padded=1; args.max_tasks=1; args.max_phys_tiles=1;
 for (int cap : {188,170,94,1}) {
  args.max_active_clusters=cap;
  assert(cuteafd_v41_expert_launch(h,&args)==0);
  assert(recorded_cap==std::min(cap,170));
 }
 device=5;
 assert(cuteafd_v41_expert_launch(h,&args)==cudaErrorInvalidDevice);
 for (int dev : {2,5}) {
  device=dev;
  assert(cuteafd_v41_expert_input_quant_initialize(&h)==0);
  assert(cuteafd_v41_expert_input_quantize_async(h,reinterpret_cast<uint16_t*>(0x10000),
    reinterpret_cast<uint8_t*>(0x20000),1,nullptr)==0);
  assert(recorded_grid==(dev==2 ? 680 : 752));
 }
 device=2;
 assert(cuteafd_v41_expert_input_quantize_async(h,reinterpret_cast<uint16_t*>(0x10000),
   reinterpret_cast<uint8_t*>(0x20000),1,nullptr)==cudaErrorInvalidDevice);
}
"""


def test_fused_diagnostic_load_fails_before_cuda_initialization_on_sm170():
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("native bridge qualification needs a C++ compiler")
    subprocess.run(["python3", str(ROOT / "scripts/build/assert-build-filesystem.py"), str(BUILD_ROOT)], check=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temp:
        directory = Path(temp)
        (directory / "cuda_runtime.h").write_text(CUDA)
        (directory / "dsv4_programs.h").write_text(r'''
#pragma once
#include "cuda_runtime.h"
inline int initializes=0;
inline void counted_init(void** args) { ++initializes; init(args); }
#define CUTEAFD_DSV4_CC_MINOR 0
#define CUTEAFD_DSV4_PROGRAMS \
 {"flash",counted_init,load,noop,1,"i",188}, \
 {"pro",counted_init,load,noop,1,"i",188}, \
 {"glm",counted_init,load,noop,1,"i",188}, \
 {"glmf",counted_init,load,noop,1,"i",188}, \
 {"m16",counted_init,load,noop,1,"i",11}, \
 {"ordinary",counted_init,load,noop,1,"i",0}
''')
        (directory / "main.cc").write_text(r'''
#include <cassert>
#include "SOURCE"
int main() {
 for (uint32_t i=0; i<4; ++i) {
  assert(cuteafd_dsv4_program_load(i)==CUTEAFD_DSV4_ERROR_INSUFFICIENT_RESIDENT_SMS);
  assert(initializes==0 && !programs[i].library && !programs[i].loaded[2]);
  void* pointers[]={reinterpret_cast<void*>(0x10000)}; uint64_t scalars[]={1};
  assert(cuteafd_dsv4_program_launch(i,pointers,scalars,nullptr)==cudaErrorInvalidResourceHandle);
 }
 attribute_error=999;
 assert(cuteafd_dsv4_program_load(0)==999 && initializes==0);
 attribute_error=0; override_sms=188;
 for (uint32_t i=0; i<4; ++i) {
  assert(cuteafd_dsv4_program_load(i)==0);
  assert(cuteafd_dsv4_program_load(i)==0);
 }
 assert(initializes==4);
 override_sms=170;
 assert(cuteafd_dsv4_program_load(0)==CUTEAFD_DSV4_ERROR_INSUFFICIENT_RESIDENT_SMS);
 assert(cuteafd_dsv4_program_load(4)==0 && cuteafd_dsv4_program_load(5)==0);
 override_sms=0; minor=1;
 assert(cuteafd_dsv4_program_load(5)==cudaErrorNoKernelImageForDevice);
}
'''.replace("SOURCE", str(ROOT / "native/shared/src/dsv4_programs.cc")))
        subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(directory),
                        "-I", str(ROOT / "native/shared/include"),
                        str(directory / "main.cc"), "-o", str(directory / "test")],
                       check=True, timeout=750)
        subprocess.run([str(directory / "test")], check=True, timeout=30)


@pytest.mark.parametrize("blocks", [1, 2])
def test_exl3_generated_bridge_uses_runtime_sms_and_propagates_errors(monkeypatch, blocks):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("native bridge qualification needs a C++ compiler")
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", ModuleType("_pinned_sparkinfer"))
    spec = importlib.util.spec_from_file_location(
        "sm_test_exl3_export", ROOT / "python/tools/aot/export_b12x_exl3_aot.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    subprocess.run(["python3", str(ROOT / "scripts/build/assert-build-filesystem.py"), str(BUILD_ROOT)], check=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temp:
        directory = Path(temp)
        (directory / "cuda_runtime.h").write_text(CUDA)
        objects = []
        for role in ("core", "sum"):
            label = f"v41_exl3_{role}"
            (directory / f"{label}.h").write_text(f'''
#pragma once
#include "cuda_runtime.h"
struct cuteafd_{label}_Kernel_Module_t {{ cudaLibrary_t module; }};
inline void _mlir_cuteafd_{label}_cuda_init(void** args) {{
  struct Init {{ cudaLibrary_t** library; cudaError_t* status; }};
  auto* init = reinterpret_cast<Init*>(args);
  **init->library = reinterpret_cast<void*>(1); *init->status = 0;
}}
inline void _mlir_cuteafd_{label}_cuda_load_to_device(void**) {{}}
inline int wrapper_{role}(cuteafd_{label}_Kernel_Module_t*, void*, int32_t, int32_t grid, cudaStream_t) {{ recorded_grid=grid; return 0; }}
''')
            objects.append(dict(label=label, wrapper=f"wrapper_{role}", parameters=[
                f"cuteafd_{label}_Kernel_Module_t *module", "void *input",
                "int32_t active_m", "int32_t grid_x", "cudaStream_t stream"]))
        exporter.write_bridge(directory, dict(blocks_per_sm=blocks, sms=188, compute=[12, 0],
            capacity=16, hidden=128, intermediate=128, experts=4, top_k=2,
            bits=[3, 4], output_dtype="bf16", objects=objects))
        (directory / "main.cc").write_text(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "v41_exl3_bridge.cc"
int main() {
 for (int dev : {2,5}) {
  device=dev; void* handle=nullptr;
  assert(cuteafd_exl3_create(&handle)==0);
  void* pointers[]={reinterpret_cast<void*>(0x10000)};
  for (int requested : {1,94,170,188,376}) {
   int32_t scalars[]={16,requested};
   const int expected=std::min(requested,(dev==2 ? 170 : 188)*BLOCKS);
   assert(cuteafd_exl3_core(handle,pointers,scalars,nullptr)==0);
   assert(recorded_grid==expected && scalars[1]==requested);
   assert(cuteafd_exl3_sum(handle,pointers,scalars,nullptr)==0);
   assert(recorded_grid==expected);
  }
  device=dev==2 ? 5 : 2;
  int32_t scalars[]={1,1};
  assert(cuteafd_exl3_core(handle,pointers,scalars,nullptr)==cudaErrorInvalidDevice);
  cuteafd_exl3_destroy(handle);
 }
 void* handle=reinterpret_cast<void*>(1);
 attribute_error=999;
 assert(cuteafd_exl3_create(&handle)==999 && handle==nullptr);
 attribute_error=0; override_sms=189;
 assert(cuteafd_exl3_create(&handle)==cudaErrorInvalidDevice && handle==nullptr);
 override_sms=0; minor=1;
 assert(cuteafd_exl3_create(&handle)==cudaErrorInvalidDevice && handle==nullptr);
}
'''.replace("BLOCKS", str(blocks)))
        subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(directory),
                        str(directory / "main.cc"), "-o", str(directory / "test")],
                       check=True, timeout=750)
        subprocess.run([str(directory / "test")], check=True, timeout=30)


@pytest.mark.parametrize("family", ["fp8", "expert"])
def test_capability_only_admission_and_live_launch_caps(family):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("native bridge qualification needs a C++ compiler")
    subprocess.run(["python3", str(ROOT / "scripts/build/assert-build-filesystem.py"), str(BUILD_ROOT)], check=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temp:
        directory = Path(temp)
        (directory / "cuda_runtime.h").write_text(CUDA)
        if family == "fp8":
            (directory / "v41_fp8_variants.h").write_text(FP8_HEADER)
            source = ROOT / "native/families/deepseek_v41/src/v41_fp8.cc"
            main = FP8_MAIN
        else:
            (directory / "v41_expert_variants.h").write_text(EXPERT_HEADER)
            (directory / "v41_input_quant_dispatch.h").write_text(INPUT_QUANT_HEADER)
            source = ROOT / "native/shared/src/v41_experts.cc"
            main = EXPERT_MAIN
        (directory / "main.cc").write_text(main.replace("SOURCE", str(source)))
        # A single small translation unit in a private temp dir: no build.lock, so a
        # long compile elsewhere can't time this test out.
        subprocess.run([compiler, "-std=c++17", "-pthread", "-I", str(directory),
                        "-I", str(ROOT / "native/shared/include"),
                        "-I", str(ROOT / "native/families/deepseek_v41/include"),
                        str(directory / "main.cc"), "-o", str(directory / "test")],
                       check=True, timeout=750)
        subprocess.run([str(directory / "test")], check=True, timeout=30)
