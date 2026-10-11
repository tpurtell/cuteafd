#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* DeepSeek V4 coordinator programs exported from b12x.integration.cuteafd
 * (python/tools/aot/export_b12x_dsv4_aot.py). Each program takes its documented
 * pointers, then its scalars, then a stream; dsv4_programs.json in the image
 * carries the pointer names, shapes and scratch sizes. */
typedef struct {
  char name[64];
  uint32_t pointers;
  uint32_t scalars;
  /* One character per scalar: 'i' int32, 'l' int64, 'f' float32. */
  char scalar_kinds[16];
} cuteafd_dsv4_program_info_t;

/* A fused diagnostic's spin-barrier CTA group cannot be resident on this card.
 * This is an engine contract error, not a CUDA runtime status. */
#define CUTEAFD_DSV4_ERROR_INSUFFICIENT_RESIDENT_SMS (-12001)

uint32_t cuteafd_dsv4_program_count(void);
int32_t cuteafd_dsv4_program_info(uint32_t index, cuteafd_dsv4_program_info_t* out);
/* Loads the program's kernels on the current device; idempotent per device. */
int32_t cuteafd_dsv4_program_load(uint32_t index);
/* Scalars travel as 64-bit slots: integers by value, float32 as its bits in
 * the low word. Returns the program's status. */
int32_t cuteafd_dsv4_program_launch(uint32_t index, void* const* pointers,
    const uint64_t* scalars, void* stream);

#ifdef __cplusplus
}
#endif
