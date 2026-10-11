// Generic launcher for the exported DeepSeek V4 coordinator programs. The
// exporter generates dsv4_programs.h: one {name, init, load, entry, pointer
// count, scalar kinds, minimum resident SMs} row per program.
#include "cuteafd_dsv4_programs.h"
#include "dsv4_programs.h"
#include <cuda_runtime.h>
#include <cstring>
#include <mutex>

namespace {
using ModuleFn = void (*)(void**);
using LaunchFn = void (*)(void**, int32_t);
constexpr int kMaxDevices = 8;
constexpr int kMaxArguments = 64;
struct Program {
  const char* name;
  ModuleFn initialize;
  ModuleFn load;
  LaunchFn launch;
  uint32_t pointers;
  const char* kinds;
  uint32_t minimum_sms;
  cudaLibrary_t library = nullptr;
  bool loaded[kMaxDevices] = {};
};
Program programs[] = {CUTEAFD_DSV4_PROGRAMS};
constexpr uint32_t kCount = sizeof(programs) / sizeof(programs[0]);
std::mutex load_mutex;

Program* at(uint32_t index) { return index < kCount ? &programs[index] : nullptr; }
}  // namespace

extern "C" uint32_t cuteafd_dsv4_program_count(void) { return kCount; }

extern "C" int32_t cuteafd_dsv4_program_info(uint32_t index, cuteafd_dsv4_program_info_t* out) {
  auto* program = at(index);
  if (!program || !out) return cudaErrorInvalidValue;
  std::memset(out, 0, sizeof(*out));
  std::strncpy(out->name, program->name, sizeof(out->name) - 1);
  out->pointers = program->pointers;
  out->scalars = static_cast<uint32_t>(std::strlen(program->kinds));
  std::strncpy(out->scalar_kinds, program->kinds, sizeof(out->scalar_kinds) - 1);
  return cudaSuccess;
}

extern "C" int32_t cuteafd_dsv4_program_load(uint32_t index) {
  auto* program = at(index);
  if (!program) return cudaErrorInvalidValue;
  int device = -1, major = 0, minor = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device < 0 || device >= kMaxDevices) return cudaErrorInvalidDevice;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (status != cudaSuccess) return status;
  if (major != 12 || minor != CUTEAFD_DSV4_CC_MINOR) return cudaErrorNoKernelImageForDevice;
  if (program->minimum_sms) {
    int sms = 0;
    status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return status;
    if (sms <= 0 || static_cast<uint32_t>(sms) < program->minimum_sms)
      return CUTEAFD_DSV4_ERROR_INSUFFICIENT_RESIDENT_SMS;
  }
  std::lock_guard<std::mutex> lock(load_mutex);
  if (program->loaded[device]) return cudaSuccess;
  auto* library = &program->library;
  if (!program->library) {
    void* init_args[] = {&library, &status};
    program->initialize(init_args);
    if (status != cudaSuccess) {
      if (program->library) cudaLibraryUnload(program->library);
      program->library = nullptr;
      return status;
    }
  }
  int32_t device_id = device;
  void* load_args[] = {&library, &device_id, &status};
  program->load(load_args);
  if (status != cudaSuccess) return status;
  program->loaded[device] = true;
  return cudaSuccess;
}

extern "C" int32_t cuteafd_dsv4_program_launch(uint32_t index, void* const* pointers,
    const uint64_t* scalars, void* stream) {
  auto* program = at(index);
  const uint32_t scalar_count = program ? static_cast<uint32_t>(std::strlen(program->kinds)) : 0;
  if (!program || !pointers || (scalar_count && !scalars) ||
      program->pointers + scalar_count + 2 > kMaxArguments)
    return cudaErrorInvalidValue;
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device < 0 || device >= kMaxDevices || !program->loaded[device]) return cudaErrorInvalidResourceHandle;
  void* pointer_values[kMaxArguments];
  int32_t i32[kMaxArguments];
  int64_t i64[kMaxArguments];
  float f32[kMaxArguments];
  void* arguments[kMaxArguments];
  uint32_t count = 0;
  for (uint32_t i = 0; i < program->pointers; ++i) {
    pointer_values[i] = pointers[i];
    arguments[count++] = &pointer_values[i];
  }
  for (uint32_t i = 0; i < scalar_count; ++i) {
    switch (program->kinds[i]) {
      case 'i': i32[i] = static_cast<int32_t>(scalars[i]); arguments[count++] = &i32[i]; break;
      case 'l': i64[i] = static_cast<int64_t>(scalars[i]); arguments[count++] = &i64[i]; break;
      case 'f': {
        const uint32_t bits = static_cast<uint32_t>(scalars[i]);
        std::memcpy(&f32[i], &bits, sizeof(bits));
        arguments[count++] = &f32[i];
        break;
      }
      default: return cudaErrorInvalidValue;
    }
  }
  int32_t result = 0;
  arguments[count++] = &stream;
  arguments[count++] = &result;
  program->launch(arguments, static_cast<int32_t>(count));
  return result;
}
