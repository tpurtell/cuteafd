#pragma once
// Runs the statement once per compiled routed-expert hidden size, with
// `kHidden` bound as a constant so row/column indexing keeps constant-divisor
// code generation. Unsupported sizes return cudaErrorInvalidValue.
#include <cstdint>
#include <cuda_runtime.h>

#define CUTEAFD_HIDDEN_CASE(value, ...) \
  case value: { constexpr uint64_t kHidden = value; (void)kHidden; __VA_ARGS__; } break;
#define CUTEAFD_WITH_HIDDEN(hidden, ...) do { switch (hidden) { \
  CUTEAFD_HIDDEN_CASE(2048, __VA_ARGS__) \
  CUTEAFD_HIDDEN_CASE(2560, __VA_ARGS__) \
  CUTEAFD_HIDDEN_CASE(3072, __VA_ARGS__) \
  CUTEAFD_HIDDEN_CASE(3584, __VA_ARGS__) \
  CUTEAFD_HIDDEN_CASE(4096, __VA_ARGS__) \
  CUTEAFD_HIDDEN_CASE(5120, __VA_ARGS__) \
  CUTEAFD_HIDDEN_CASE(6144, __VA_ARGS__) \
  CUTEAFD_HIDDEN_CASE(7168, __VA_ARGS__) \
  default: return cudaErrorInvalidValue; } } while (0)
