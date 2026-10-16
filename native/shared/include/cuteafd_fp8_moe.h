// Exact FP8 routed-expert package library (python/tools/aot/package_fp8_moe_aot.py):
// one libcuteafd_fp8moe.so per layout (tp1, tp2, tp4) of an fp8-<geometry>
// package, carrying one b12x fp8_moe program per capacity. Weights are the
// checkpoint's E4M3 experts with FP32 128x128 block scales; nothing is
// re-quantized. Output is the BF16 [rows, H] route sum of the layout's
// intermediate slice (the Spark rank partial; the whole layer at tp1).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// words (16): [0] packed contract (all bytes required):
//   bits 0..7: ABI version 5 (old unversioned contracts are rejected)
//   bits 8..15: storage 1=FP8 block128, 2=MXFP4, 4=NVFP4 W4A4, 5=BF16
//   bits 16..23: activation 1=pre-SiLU clamp + BF16 SiLU/product rounding,
//                          2=post-SiLU clamp + FP32 product/final BF16 rounding
//   bits 24..31: projection 1=BF16 gate/up and down outputs
// Both activations clamp up to [-limit, limit]; limit zero disables clamping.
// [1] hidden, [2] slice, [3] experts, [4] top-k,
// [5] intermediate, [6] tp, [7] input dtype (7 = FP8 E4M3 + UE8M0 K32 wire
// rows: Spark packages; 1 = BF16 rows: coordinator packages), [8] SwiGLU limit (FP32 bits, 0 = none), [9] capacity count n,
// [10 .. 10+n) capacities.
int32_t cuteafd_fp8moe_info(uint32_t* words, uint32_t count);
int32_t cuteafd_fp8moe_scratch_bytes(uint32_t capacity, uint64_t* bytes);
// Static installed traffic contract for capacity/form (same fallback as launch).
// rows=0 queries the final segment. contract=1: exact_grouping; contract=2:
// conservative_route_upper_bound (group_rows=1, one slice per routed occurrence).
int32_t cuteafd_fp8moe_weight_read_group(uint32_t capacity, uint32_t form, uint32_t rows,
                                       uint32_t* group_rows, uint32_t* contract);
// Loads every program on the current device.
int32_t cuteafd_fp8moe_create(void** context);
// pointers (11): x (wire rows [rows, H + H/32] or BF16 [rows, H]), ids i32 [rows, k], weights f32
// [rows, k], w1 e4m3 [E, I, H], s1 f32 [E, I/128, H/128], w3, s3 (same),
// w2 e4m3 [E, H, I], s2 f32 [E, H/128, I/128], out bf16 [rows, H], scratch.
// Native BF16: w1/w3 [E,I,H] and w2 [E,H,I] are contiguous BF16 matrices;
// s1/s3/s2 are unused (may be null), with zero scale storage. BF16/NVFP4
// require BF16 input. Validate the static contract before device allocation.
int32_t cuteafd_fp8moe_launch(void* context, uint32_t capacity, void* const* pointers, int32_t rows,
                              void* stream);
void cuteafd_fp8moe_destroy(void* context);
// options: the large-row form of FP8 programs. 0 the package default (wire rows
// W8A8: block-scaled E4M3 x E4M3 gate/up; BF16 rows W8A16, exact), 1 W8A16 (the
// former programs), 2 W8A8 (BF16 rows quantized to wire rows first). Packages
// built before the forms lack this symbol.
int32_t cuteafd_fp8moe_set_options(void* context, uint32_t options);

#ifdef __cplusplus
}
#endif
