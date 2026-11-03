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
  ModuleFn initialize_sm188 = nullptr;
  ModuleFn load_sm188 = nullptr;
  LaunchFn launch_sm188 = nullptr;
  cudaLibrary_t library = nullptr;
  cudaLibrary_t library_sm188 = nullptr;
  // Zero is unloaded; 1/2 retain the selected 170/188-SM plan before capture.
  uint8_t loaded[kMaxDevices] = {};
  int32_t sm_count[kMaxDevices] = {};

  bool runtime_sms() const { return std::strchr(kinds, 's') != nullptr; }
  uint32_t public_scalars() const {
    return static_cast<uint32_t>(std::strlen(kinds)) - (runtime_sms() ? 1 : 0);
  }
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
  out->scalars = program->public_scalars();
  std::memcpy(out->scalar_kinds, program->kinds, out->scalars);
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
  int sms = 0;
  if (program->minimum_sms || program->launch_sm188 || program->runtime_sms()) {
    status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return status;
    if (sms <= 0) return cudaErrorInvalidDevice;
    if (static_cast<uint32_t>(sms) < program->minimum_sms)
      return CUTEAFD_DSV4_ERROR_INSUFFICIENT_RESIDENT_SMS;
  }
  const bool use_sm188 = program->launch_sm188 && sms >= 188;
  std::lock_guard<std::mutex> lock(load_mutex);
  if (program->loaded[device]) return cudaSuccess;
  auto* storage = use_sm188 ? &program->library_sm188 : &program->library;
  auto* library = storage;
  if (!*storage) {
    void* init_args[] = {&library, &status};
    (use_sm188 ? program->initialize_sm188 : program->initialize)(init_args);
    if (status != cudaSuccess) {
      if (*storage) cudaLibraryUnload(*storage);
      *storage = nullptr;
      return status;
    }
  }
  int32_t device_id = device;
  void* load_args[] = {&library, &device_id, &status};
  (use_sm188 ? program->load_sm188 : program->load)(load_args);
  if (status != cudaSuccess) return status;
  program->sm_count[device] = sms;
  program->loaded[device] = use_sm188 ? 2 : 1;
  return cudaSuccess;
}

extern "C" int32_t cuteafd_dsv4_program_launch(uint32_t index, void* const* pointers,
    const uint64_t* scalars, void* stream) {
  auto* program = at(index);
  const uint32_t scalar_count = program ? static_cast<uint32_t>(std::strlen(program->kinds)) : 0;
  if (!program || !pointers || (program->public_scalars() && !scalars) ||
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
      case 's': i32[i] = program->sm_count[device]; arguments[count++] = &i32[i]; break;
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
  (program->loaded[device] == 2 ? program->launch_sm188 : program->launch)(
      arguments, static_cast<int32_t>(count));
  return result;
}
