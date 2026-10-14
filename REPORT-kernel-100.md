# Kernel duel #100 — single-pass wide W8A8

Branch `duel/grok-100` in `/home/tj/Developer/cuteafd-grok-k100`. Local commit only.

## Diff summary

`native/shared/cuda/fp8_gemv.cu`: when `w8a8 && rows > 64`, one 2D grid `(m_tiles, n_ctas)`, splits = 1, existing packed weights, new `gemv_w8a8_wide_kernel<16>` using the same `mma.m16n8k32` math as `gemv_w8a8_kernel<16>`. Direct BF16 or FP32 store, no reduce. Rows ≤ 64 and every W8A16 path keep the old kernels and the old host loop. Wide W8A8 scratch is `max(legacy split-K arena, single-pass activation tile)`. SM count still comes from `cudaDevAttrMultiProcessorCount` (188 on this card). ptxas: the wide kernel uses 254 registers and does not spill, the same register file as `gemv_w8a8_kernel<16>`.

`draft_fp8_scratch_bytes` mirrors that size. The Flash inventory test asserts 43,253,760 for the FC at both 188 and 170 SMs.

Harness `native/tests/fp8_w8a8_wide_selftest.cu` is a standalone nvcc binary, not a CMake target. The baseline `.so` is `git show HEAD:native/shared/cuda/fp8_gemv.cu` before this change.

## Correctness

GPU0, RTX PRO 6000, 188 SMs, UUID `GPU-fe5b6dd0-a77c-c8fb-6360-e1b9d9918ac0`. One run, not repeated. Reference is an FP64 sum of the dequantized E4M3 products (device quant, checked bit-exact against the packed weight bytes and against the single-pass activation workspace).

Narrow paths, both shapes (FC 20480×4096 and KV 4096×2048), modes 0, 1, and 2, M ∈ {1, 8, 16, 64}, FP32 and BF16: bit-identical to the baseline `.so`. Modes 0 and 1 at M=2048 are also bit-identical. Launch counter stays 0 on those routes. Ternary operands are bit-exact at M ∈ {64, 128, 2048} on both shapes.

Raw max rel divides by `max(|ref|, 1e-6)`. `rel_gt1e-3` is the max of `|got-ref|/|ref|` over elements with `|ref| > 1e-3`. `vs_old` is the new FP32 output against the baseline split-K library.

| shape | M | max abs | raw max rel | rel_gt1e-3 | vs_old abs | vs_old rel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fc | 65 | 4.366e-6 | 7.271e-2 | 5.060e-4 | 4.292e-6 | 3.175e-2 |
| fc | 128 | 4.561e-6 | 1.633e-1 | 5.550e-4 | 4.292e-6 | 6.457e-2 |
| fc | 129 | 3.432e-6 | 8.886e-2 | 4.754e-4 | 1.464e-1 | 3.544e+1 |
| fc | 1024 | 4.254e-6 | 2.419e-1 | 8.870e-4 | 4.292e-6 | 2.220e-1 |
| fc | 2048 | 4.604e-6 | 3.418e-1 | 1.029e-3 | 4.768e-6 | 2.988e-1 |
| kv | 65 | 6.818e-7 | 2.616e-3 | 1.081e-4 | 7.153e-7 | 2.981e-3 |
| kv | 128 | 7.481e-7 | 7.700e-3 | 1.105e-4 | 9.537e-7 | 9.179e-3 |
| kv | 129 | 8.108e-7 | 4.795e-3 | 1.344e-4 | 5.908e-2 | 4.510e+2 |
| kv | 1024 | 7.764e-7 | 6.505e-2 | 1.679e-4 | 9.537e-7 | 5.888e-2 |
| kv | 2048 | 8.605e-7 | 6.228e-2 | 1.884e-4 | 9.537e-7 | 5.025e-2 |

At M=129 the old route's last pass is a single row, and a pass of at most 8 rows stays on W8A16. The new wide path uses W8A8 for every row, including that tail. Only those outlier elements move (vs_old abs 1.46e-1 on FC, 5.91e-2 on KV). Cosine against the old route stays ~1.0 (0.99999751 FC, 0.99999766 KV). Against the FP64 W8A8 reference the same rows are ordinary: abs under 1e-6.

The binary that produced this table failed its own check (`correctness FAIL`). That check was raw `fp32.max_rel > 2e-2` or `bf.max_rel > 5e-2`, plus non-finite outputs, a BF16 store that is not the rounding of the FP32 output, or cosine under 0.999. Raw max rel is unusable here: the denominator `max(|ref|, 1e-6)` turns a 4.6e-6 absolute error on a near-zero output into a reported rel of 0.34. The harness source now gates on `fp32.max_rel_big > 2e-2` or `bf.max_rel_big > 5e-2` or `fp32.max_abs > 1e-2` or `bf.max_abs > 5e-2`, with the same finite, rounding, and cosine checks. The logged `rel_gt1e-3` values are at most 1.029e-3 and the FP32 abs values are at most 4.604e-6, so that criterion passes. The GPU binary was not rebuilt or re-run after the criterion change.

E4M3 host decode matched the device table for all 256 codes. Weight constructor and activation cvt agreed on 4096 random floats.

## Performance

Same process, one stream, one warmup, median of 51 CUDA events. Old kernel counts are structural: 16 passes of 128 rows, and at 188 SMs both shapes split K (FC 12-way, KV 16-way), so each pass is pack + gemm + reduce.

| workload | old ms | new ms | speedup | old kernels | new kernels |
| --- | ---: | ---: | ---: | ---: | ---: |
| fc M=2048 | 1.1117 | 0.8086 | 1.375 | 48 | 2 |
| kv M=2048 | 0.2425 | 0.1154 | 2.101 | 48 | 2 |
| 1 fc + 5 kv | 2.6745 | 1.6904 | 1.582 | 288 | 12 |

## Stop-bar verdict

Fail. The wave is 2.6745 → 1.6904 ms, 1.582×, which is under 2×. Per the brief: report and stop. Kernel count fell fc 48→2, kv 48→2, wave 288→12. FC latency only moved 1.375×. The wide kernel holds 254 registers, so occupancy stays at the old 16-group kernel's level, and one CTA still reads the full K. No further tuning, and the GPU was not re-run.

Thermal: no in-run `/v1/thermal` window (no server). `nvidia-smi -i 0` immediately after the process: 50 °C, 2505 MHz, 195 W, power limit 545 W, software power cap active, hardware thermal slowdown not active. Both arms ran back to back in that process.

## Wall time

- Reading the kernel, issue #100, and the design note: the previous session.
- Coding the kernel, the scratch mirror, and the harness: that session plus this one.
- Build and mechanics: loader tests 21 s (3 passed). nvcc of the two `.so` files and the harness, a few container starts, well under 5 minutes of compile.
- GPU: one GPU1 attempt died on a byte-swapped UUID before any kernel. The GPU0 command's wall time was 1750 s, almost all of it the fair-queue wait. On-device work (bit checks, FP64 reference ~0.9 s, 51-iteration timing) was about a minute inside the container.
- Analysis and this report: after that log.
