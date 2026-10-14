# Kernel duel #100 — single-pass wide W8A8

Branch `duel/grok-100` in `/home/tj/Developer/cuteafd-grok-k100`. Local commit only.

## Diff summary

`native/shared/cuda/fp8_gemv.cu`: when `w8a8 && rows > 64`, one 2D grid `(m_tiles, n_ctas)`, splits = 1, existing packed weights, new `gemv_w8a8_wide_kernel<16>` using the same `mma.m16n8k32` math as `gemv_w8a8_kernel<16>`. Direct BF16 or FP32 store, no reduce. Rows ≤ 64 and every W8A16 path keep the old kernels and the old host loop. Wide W8A8 scratch is `max(legacy split-K arena, single-pass activation tile)`. SM count still comes from `cudaDevAttrMultiProcessorCount` (188 on this card). ptxas: the wide kernel uses 254 registers and does not spill, the same register file as `gemv_w8a8_kernel<16>`.

`draft_fp8_scratch_bytes` mirrors that size. The Flash inventory test asserts 43,253,760 for the FC at both 188 and 170 SMs.

Harness `native/tests/fp8_w8a8_wide_selftest.cu` is a standalone nvcc binary, not a CMake target. The baseline `.so` is `git show HEAD:native/shared/cuda/fp8_gemv.cu` before this change.

## Correctness

GPU0, RTX PRO 6000, 188 SMs, UUID `GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0`. Reference is an FP64 sum of the dequantized E4M3 products (device quant, checked bit-exact against the packed weight bytes and against the single-pass activation workspace). `rel>|1e-3|` ignores outputs with `|ref| ≤ 1e-3`, where a 1e-6 floor makes raw rel meaningless. Cosine is over the whole matrix.

Narrow paths, both shapes (FC 20480×4096 and KV 4096×2048), modes 0, 1, and 2, M ∈ {1, 8, 16, 64}, FP32 and BF16: bit-identical to the baseline `.so`. Modes 0 and 1 at M=2048 are also bit-identical. Launch counter stays 0 on those routes. Ternary operands (the existing mode-2 vs mode-0 contract) are bit-exact at M ∈ {64, 128, 2048} on both shapes.

| shape | M | fp32 max abs | fp32 rel, \|ref\|>1e-3 | bf16 max abs | cosine | vs old fp32 max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fc | 65 | 4.37e-6 | 5.06e-4 | 1.56e-2 | 1.00000000 | 4.29e-6 |
| fc | 128 | 4.56e-6 | 5.55e-4 | 1.56e-2 | 1.00000000 | 4.29e-6 |
| fc | 129 | 3.43e-6 | 4.75e-4 | 1.56e-2 | 1.00000000 | 1.46e-1 |
| fc | 1024 | 4.25e-6 | 8.87e-4 | 1.56e-2 | 1.00000000 | 4.29e-6 |
| fc | 2048 | 4.60e-6 | 1.03e-3 | 1.56e-2 | 1.00000000 | 4.77e-6 |
| kv | 65 | 6.82e-7 | 1.08e-4 | 7.81e-3 | 1.00000000 | 7.15e-7 |
| kv | 128 | 7.48e-7 | 1.11e-4 | 7.80e-3 | 1.00000000 | 9.54e-7 |
| kv | 129 | 8.11e-7 | 1.34e-4 | 7.81e-3 | 1.00000000 | 5.91e-2 |
| kv | 1024 | 7.76e-7 | 1.68e-4 | 7.81e-3 | 1.00000000 | 9.54e-7 |
| kv | 2048 | 8.61e-7 | 1.88e-4 | 7.81e-3 | 1.00000000 | 9.54e-7 |

BF16 output is the device rounding of the FP32 output on every wide case. Gate: fp32 rel on `|ref|>1e-3` under 2e-2, bf16 abs under 5e-2, cosine at least 0.999. Pass.

M=129 versus the old route is the one larger gap. The old loop's last row is a 1-row W8A16 tail; the new path quantizes that row as W8A8. The new value still matches the W8A8 FP64 reference (abs under 1e-6).

E4M3 host decode matched the device table for all 256 codes. Weight constructor and activation cvt agreed on 4096 random floats.

## Performance

Same process, one stream, one warmup, median of 51 CUDA events. Old kernel counts are structural: 16 passes of 128 rows, and at 188 SMs both shapes split K (FC 12-way, KV 16-way), so each pass is pack + gemm + reduce.

| workload | old ms | new ms | speedup | old kernels | new kernels |
| --- | ---: | ---: | ---: | ---: | ---: |
| fc M=2048 | 1.1117 | 0.8086 | 1.375 | 48 | 2 |
| kv M=2048 | 0.2425 | 0.1154 | 2.101 | 48 | 2 |
| 1 fc + 5 kv | 2.6745 | 1.6904 | 1.582 | 288 | 12 |

## Stop-bar verdict

Fail. The wave is 1.58×, short of 2×. Kernel count fell 288 → 12; FC latency only 1.37×. The wide kernel holds 254 registers, so occupancy stays at the old 16-group kernel's level, and one CTA still reads the full K. No further tuning.

Thermal: no in-run `/v1/thermal` window (no server). `nvidia-smi -i 0` immediately after the process: 50 °C, 2505 MHz, 195 W, power limit 545 W, software power cap active, hardware thermal slowdown not active. Both arms ran back to back in that process.

## Wall time

- Reading the kernel, issue #100, and the design note: the previous session.
- Coding the kernel, the scratch mirror, and the harness: that session plus this one.
- Build and mechanics: loader tests 21 s (3 passed). nvcc of the two `.so` files and the harness, a few container starts, well under 5 minutes of compile.
- GPU: one GPU1 attempt died on a byte-swapped UUID before any kernel. The GPU0 command's wall time was 1750 s, almost all of it the fair-queue wait. On-device work (bit checks, FP64 reference ~0.9 s, 51-iteration timing) was about a minute inside the container.
- Analysis and this report: after that log.
