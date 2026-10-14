# #210 — FP8 projection GEMV/TMA switch at M=17

Analysis of the pinned fork. No kernel change, no GPU run.

* cuteafd `work/grok-210` base `b91c26068f7604d3142b783595b1aabdec79b9f6` (`origin/work/p0` at the branch point).
* `third_party/sparkinfer` pin `91b104db6868b60f24a0b1c06e54cfe431da9791`.
* Line numbers are that pin. Paths are under `third_party/sparkinfer/` unless noted.

## Dispatch

`Fp8Projection.__call__` (`b12x/integration/cuteafd/_fp8_weights.py:172`) is the shared decode switch. With `prefill_rows is None`:

* `limit = min(fp8_rows, max_gemv_rows)`.
* `rows <= limit` and `rows <= gemv_rows` (16): narrow `MmaFp8Gemv`.
* `rows <= limit` and a `wide_rows` tile exists: wide `MmaFp8Gemv`.
* otherwise: `TmaFp8Gemm` (W8A16). The `else` is line 195.

`FP8_GEMV_ROWS = 16` (`_fp8_weights.py:199`). Prefill does not use this branch. It picks `BlockFp8Projection` (W8A8) when `fp8_rows & prefill_mask` is set, else the same TMA class, and the choice does not depend on M.

The Rust threshold registry (`rust/crates/cuteafd-daemon/src/families/glm5_flash/engine.rs:1406`) refuses a decode bucket that straddles one of these caps, so one captured launch is one kernel. It does not make a row computed at M=1 match the same row computed at M=32.

## The two K orders

Dequant is the same per element: `cvt` E4M3 to FP32, `mul.rn.f32` by the scale, round to BF16. GEMV widens 4 values (`_e4m3x4_scaled_bf16x2x2`, `_fp8_weights.py:544`). TMA widens 8 (`_e4m3x8_scaled_bf16`, `:278`). The scale granule is one FP32 per 128 K (TMA reads it at `k_tile // 2`, `:384`). What differs is the order of the FP32 add.

**GEMV** (`MmaFp8Gemv.kernel`, `:645`):

* Warps split K. Warp `w` owns `[w * K/warps, (w+1) * K/warps)` (`:658`, `k_per_warp` at `:611`).
* Inside the warp, K walks 128-wide blocks, two 64-wide chunks, then four `m16n8k16` BF16→FP32 MMAs (`bf16_mma_m16n8k16_f32`, `:719`).
* Partials fold in warp-id order: `total = 0; for src in 0..warps: total += partial[src]` (`:736`).

Every production config uses at least 2 warps (`gemv_warps`, `:264`, is 8 only at K=6144 and 4 otherwise; the Flash table's minimum is 2). The fold order is therefore not the sequential K order.

Narrow and wide GEMVs with the **same** `(warps, groups)` use the same per-row K order. `m_tiles = ceil(max_rows / 16)` (`:606`) only adds accumulator fragments; tile 0's MMA sequence does not read the other tiles' registers. Different `(warps, groups)` would change the K split. `_w8` (`glmf.py:290`) forces `wide_warps=8, wide_groups=2` whenever `wide_rows` is set. Today the only `_w8` wide consumer is half-head `o`, whose narrow config is also `(8, 2)`, and half-head `in` is replaced with an explicit `(8, 2)` / `(8, 2)` (`glmf.py:684`). `mimo_w8` (`mimo_attention.py:253`) does not override the wide tile, so both tiles keep the narrow pair.

Compiling **one** GEMV at `max_rows=64` and launching it at M=1 is not the thin kernel. `m_tiles` is a constructor constant, `frags = m_tiles * groups * 4` (`:615`), and the K loop iterates every tile (`:695`). The register file is the fat one.

**TMA** (`TmaFp8Gemm.kernel`, `:490`):

* Warps are stacked on M. `tiled_mma` atom is `(num_compute_warps, 1, 1)` (`:373`). Decode uses `compute_warps=4`, so `tile_m = 64` (`:360`). No K split.
* K walks `k_tile = 0 .. K/64 - 1` (`tile_k = 64`), then `cute.gemm` of `m16n8k16` into one FP32 accumulator (`:510`).
* `tile_n = decode_tile_n(n)` (`:202`): 128 if N≥16384, 64 if N≥4096 and N%64==0, else 32. Stages are 4 at tile 128 and 6 otherwise (`:158`).

For every shape in the table, `tile_m` is 64, so a decode TMA launch at M=1 and at M=17 has the same grid (`ceil(M/64) = 1` through M=64) and the same K loop. Weight traffic is the E4M3 matrix once. The published 17-row TMA times are the right stand-in for "TMA at M=1".

**Dual-copy large path** is not `TmaFp8Gemm`. `RoutedFp8Projection` (`:57`) and `projection()` (`:100`) use `TmaBf16Projection` above 16. `_Fp8Switch` (`glmf.py:151`) uses `glm_projection` (`_glm_kernels.py:53`), which is `RoutedBf16Projection(..., max_skinny_rows=0)`: one BF16 TMA order at every M≥1. That TMA walks K the same way (`Bf16PrefillKernel` consumer loop `_prefill.py:359`, inner `cute.gemm` in `_warp_gemm_direct` `:79`, `compensated=False` from `prefill_config`). GEMV versus that TMA is the same class of order change. It is not the kernel #173 measured.

## Who crosses 16

Serving does not launch M=17. It launches the bucket, and the kernel branches on the live `rows` scalar.

| Family | Buckets that reach the projection | fp8_rows the kernel sees |
| --- | --- | --- |
| GLM Flash | plain `[1,4,8,16,32,64]`, spec `[2,4,8,16,32,64]` (`serving_capacity/glmf_graphs.rs:7`). Wide `--decode-rows 128` appends the verify budget (`:19`): 128 on 188 SMs, 127 on 170, 99 on 132. | KDA: 16 full heads, 32 half heads (`glm5_flash/engine.rs:4187`). MLA and dense/shared FFN: 16 on decode (`fp8_scalars`, `:4204`). |
| GLM 5.3 | short context `[1..16, 20, 24, 28, 32, 40, 48, 56, 64]`, long tables `[1,2,4,8,16,32,64]` (`serving_capacity.rs:298`). A 17-token verify pads to 20. | Baked `Int32(FP8_GEMV_ROWS)` inside the decode class. The Rust m64 scalar list is just `rows`. |
| Qwen 3.8 | plain `[1,4,8,16]` (C16 decode stays on GEMV), spec `[2,4,8,16,24,32,64]` (`qwen_graphs.rs:10`). | Baked 16 in `_GdnW8Decode`, `_ProducerW8Decode`, `_AttnOW8Decode`. |
| MiMo | every decode/verify shape `1..=64` is captured (`mimo_v2/serve.rs:273`). | `FP8_DECODE_ROWS = 32` (`mimo_v2/engine.rs:180`), passed at runtime. |

16 concurrent sequences times a verify width of 64 is one 64-row launch on every family. Half-head / MiMo stay on GEMV at 17 and 32. They take TMA at 64 (and at the 128-wide Flash budget).

## Consumers

Shapes are `(N, K)` of `out = x @ W^T`, W stored `[N, K]`. "Crosses" is whether decode or verify launches a row count on the other side of the cap. "Orders" is whether those two launches reduce K differently.

Warp pairs come from `FP8_GEMV_CONFIG` (`glmf.py:241`) or, if absent, `(gemv_warps(K), 4)`.

### Certain at 17 — GEMV ≤16, TMA above, warps > 1, serving crosses

| Consumer | Construction | N, K | Scales | GEMV | Cap |
| --- | --- | --- | --- | --- | --- |
| GLMF KDA in, full | `_w8` via `_KdaW8` `glmf.py:682` | 24896, 4096 | row, K-major | 8, 2 | 16 |
| GLMF KDA o, full | `_KdaW8` `:689` | 4096, 8192 | row, K-major | 8, 2 | 16 |
| GLMF MLA qkv_a | `_MlaProducerW8` `:1011` | 2048, 4096 | block | 8, 4 | 16 |
| GLMF MLA q_b, full | `:1012` | 16384, 1536 | block | 2, 2 | 16 |
| GLMF MLA q_b, half heads | same, heads 32 | 8192, 1536 | block | 4, 4 | 16 |
| GLMF MLA o, full | `_GlmfOutputW8` `glmf.py:1509` | 4096, 16384 | block | 4, 2 | 16 |
| GLMF MLA o, half | same, heads 32 | 4096, 8192 | block | 8, 2 | 16 |
| GLMF MLA output rows | `GlmfMlaOutputRows` `_glmf_mla_output_rows.py:45`, GEMV iff `total_rows ≤ 16` (`:58`) | 4096, 16384 | block | 4, 2 | 16 |
| GLMF dense gate_up, full | `_GlmfFfnW8` `glmf.py:1149` | 24576, 4096 | block | 8, 4 | 16 |
| GLMF dense down, full | `:1150` | 4096, 12288 | block | 8, 2 | 16 |
| GLMF dense gate_up, half | `ffn_on` halves `inter` when `layer.split` (`engine.rs:4245`) | 12288, 4096 | block | 4, 4 | 16 |
| GLMF dense down, half | same | 4096, 6144 | block | 8, 4 | 16 |
| GLMF shared gate_up, full | `_GlmfFfnW8`, `moe_inter` 2048 | 4096, 4096 | block | 8, 2 | 16 |
| GLMF shared down, full | same | 4096, 2048 | block | 2, 2 | 16 |
| GLMF shared gate_up, half | `inter/2` | 2048, 4096 | block | 8, 4 | 16 |
| GLMF shared down, half | `inter/2` | 4096, 1024 | block | 4, 4 | 16 |
| GLM 5.3 qkv_a | `_w8_projection` `glm_attention.py:303`, decode bakes 16 at `:361`. Decode scales are the 128×128 grid; row scales are prefill only (`:304`) | 2624, 6144 | block | 8, 4 | 16 |
| GLM 5.3 q_b | `:316` | 16384, 2048 | block | 4, 4 | 16 |
| GLM 5.3 o | `_OutputW8` `:437`, bake `:461` | 6144, 16384 | block | 4, 4 | 16 |
| GLM 5.3 index wq | `_IndexProducerW8` `:380`, bake `:416`. `wk` is BF16 `glm_projection`, not this switch | 4096, 2048 | block | 4, 4 | 16 |
| GLM 5.3 dense gate_up | `glm_ffn.py:163`, bake `:194` | 24576, 6144 | block | 8, 4 | 16 |
| GLM 5.3 dense down | `:164` | 6144, 12288 | block | 4, 4 | 16 |
| GLM 5.3 shared gate_up | same class, `moe_inter` 2048 | 4096, 6144 | block | 8, 4 | 16 |
| GLM 5.3 shared down | same | 6144, 2048 | block | 4, 4 | 16 |
| Qwen GDN in | `qwen4_gdn.py:224`, bake `:373` | 16480, 2560 | block | 4, 4 | 16 |
| Qwen GDN o | `:226` | 2560, 6144 | block | 8, 4 | 16 |
| Qwen attn in | `qwen4_attention.py:241`, bake `:277` | 13952, 2560 | block | 4, 4 | 16 |
| Qwen attn o | `:661`, bake `:683` | 2560, 6144 | block | 8, 4 | 16 |

`FP8_GEMV_CONFIG` keys are `(N, K)` only, so a second consumer of the same shape inherits the pair. Shared gate_up half is `(2048, 4096)`, the qkv_a entry, so it runs `(8, 4)` rather than the default `(4, 4)`. Half-head MLA o is `(4096, 8192)`, the full-head KDA o entry, so it runs `(8, 2)`.

Four pairs compile to one kernel. The test still names both consumers:

* full MLA o and MLA output rows, `_w8(4096, 16384)`
* MLA qkv_a and shared gate_up half, `_w8(2048, 4096)`
* Qwen GDN o and attn o, `Fp8Projection(2560, 6144)`
* MiMo Pro o and Pro dense down, `mimo_w8(6144, 16384)`

MLA output rows adds one behavior the bare projection does not: `fp8_rows` follows global `total_rows` (`_glmf_mla_output_rows.py:58`), so one owned row inside a 17-row step still takes TMA. At the widths in the test, passing the cap and passing 0 above the cap select the same kernel.

#173 already measured the full-head KDA W8/S16 failure at 17, on the outputs and on the recurrent and conv state. The other rows of this table are the same branch and the same two kernels, unmeasured.

1× RTX runs full KDA heads (switch at 16). 2× RTX runs the half-head shapes. Both are production columns.

### Maybe at 17, certain above 32 — wide GEMV shares the K split

Same `(warps, groups)` on the 16-row tile and the 32-row tile, by the constructors below. M=1, 16, 17 and 32 should match. M=64 takes TMA and should not. Unproven until the test; the "should match" is the K-loop inspection, not a GPU result.

| Consumer | Construction | N, K | Scales | GEMV, both tiles | Cap |
| --- | --- | --- | --- | --- | --- |
| GLMF KDA in, half heads | explicit `Fp8Projection` `glmf.py:687` (replaces `_w8`, which would have picked `(4, 4)` and a wide `(8, 2)`) | 12576, 4096 | row, K-major | 8, 2 | 32 |
| GLMF KDA o, half | `_w8(..., wide_rows=32)` `:689`. Config `(4096, 4096)` is already `(8, 2)` | 4096, 4096 | row, K-major | 8, 2 | 32 |
| GLMF output rows | `GlmfOutputRows` `_glmf_output_rows.py:18`. GEMV iff `total_rows ≤ 32` (`:43`) | 4096, 8192 | row, K-major | 8, 2 | 32 |
| GLMF output shard | `GlmfOutputShard` `_glmf_output_shard.py:46`. Always passes `fp8_rows=32` (`:66`) | 2048, 8192 | row, K-major | 8, 2 | 32 |
| MiMo Flash qkv full | `mimo_w8` `mimo_attention.py:281` | 13568, 4096 | row | 2, 2 | 32 |
| MiMo Flash qkv SWA | same | 14848, 4096 | row | 2, 2 | 32 |
| MiMo Flash o | `:556` | 4096, 8192 | row | 4, 4 | 32 |
| MiMo Flash dense gate_up | `mimo_ffn.py:112` | 32768, 4096 | row | 8, 2 | 32 |
| MiMo Flash dense down | `:113` | 4096, 16384 | row | 4, 4 | 32 |
| MiMo V2.6 Pro qkv (full and SWA share the width) | `mimo_w8` | 27648, 6144 | row | 8, 4 | 32 |
| MiMo Pro o | same | 6144, 16384 | row | 4, 4 | 32 |
| MiMo Pro dense gate_up | `mimo_ffn.py` | 32768, 6144 | row | 8, 4 | 32 |
| MiMo Pro dense down | same | 6144, 16384 | row | 4, 4 | 32 |

### Maybe — dual copy, not the measured kernel

Production `--fp8-decode` checkpoints take `fp8_only` (`Fp8Projection`). `fp8=True` is still compilable. The large path is BF16 TMA (`glm_projection` / `TmaBf16Projection`), so a bit difference at 17 is the same order change against a different GEMM. The existing `test_fp8_projection_decode_matches_routed` compares each M to the BF16-copy route **at that same M**. Both sides can switch together and still move a row's bits between M=1 and M=17.

The test covers two shapes, not every BF16 projection:

| Consumer | Construction | N, K | Scales | GEMV | Large path |
| --- | --- | --- | --- | --- | --- |
| `RoutedFp8Projection` GLM q_b | `projection()` / `RoutedFp8Projection` `_fp8_weights.py:57`. The shape the same-M test already compiles | 16384, 2048 | block | 4, 4 | `TmaBf16Projection` above 16 |
| GLMF KDA in, dual copy | `_Fp8Switch` `glmf.py:575`, `row_scales=True`, no `wide_rows` (half heads do **not** get the W8 wide tile on this path) | 24896, 4096 | row | 8, 2 | `glm_projection` (BF16 TMA at every M) above 16 |

Do not fold this path into the W8 fix. If the representative case fails, it is a follow-up on `_Fp8Switch` / `RoutedFp8Projection`.

### Safe for this defect

* **Prefill `Fp8Projection`.** W8A8 vs W8A16 is `fp8_rows & prefill_mask`, not M (`_fp8_weights.py:174`).
* **Qwen GDN prefill in-projection.** `P = 16480` is not a multiple of 128, so prefill uses `_PrefillW8A16` (`qwen4_gdn.py:179`), which always calls TMA. Decode still switches; that row is in the certain table.
* **LM heads.** `_HeadFp8` (`glmf.py:1577`) and `compile_mimo_head_fp8_aot` are `MmaFp8Gemv` only, `max_rows=16`. There is no second kernel. (Launching M>16 on that object does not write the extra rows; `m_tiles` is 1. That is a capacity limit.)
* **V4 / V4.1.** No `Fp8Projection`, `MmaFp8Gemv`, `TmaFp8Gemm`, or `RoutedFp8Projection` anywhere in the pinned `b12x/` tree outside the GLM / MiMo / Qwen files named above.
* **Qwen shared expert and the other `qwen4.py` linears.** BF16 `_projection` (`qwen4.py:107` and the shared gate_up/down call sites). Not `Fp8Projection`.
* **GLM `wk` and `BatchedFp8Gemm`** (kv_b absorb / W_UV). Not an M switch.
* **Routed MoE.** `_fp8_moe_kernels.py`: decode is `GroupedFp8Gemv` (an `MmaFp8Gemv` per expert), prefill is `GroupedFp8Gemm` (a `TmaFp8Gemm` per 64-row tile). The header says rows inside one group are invariant because each row reads only its own input. That is a different pipeline from the dense M=16 switch, and it is not in this test.
* **GLM tensor-scale FFN** (`_compile_glm_ffn_tensor_w8`) is prefill only.

## Test

`scripts/qualify/test_fp8_projection_row_invariance.py`

Drop it in as `third_party/sparkinfer/tests/integration/test_fp8_projection_row_invariance.py` (the `from ..conftest import require_b12x` import is the fork's). It is not on the host suite path (`pytest scripts/tests`). A stray collection from `scripts/qualify` hits the import guard and skips.

One pytest, parametrized by every certain and maybe consumer above and by M in `{16, 17, 32, 64}`:

* Build the production object (`_w8`, `mimo_w8`, `Fp8Projection`, `GlmfOutputRows` / `GlmfOutputShard`, the half-head in-proj constructor, `RoutedFp8Projection`, `_Fp8Switch`).
* Before the compare, require that a compiled GEMV has `warps > 1`, and that a wide GEMV, when one is compiled, uses the same `(warps, groups)`. The 16/32 cap is not frozen. A same-order GEMV through 64 and a TMA route at every M both reach the bit compare. A mixed switch stays red.
* Same E4M3 weight and scales at every M. Dual-copy also builds `bf16(w * s)` and feeds it to the large path, so both routes are the same product.
* Row 0 is one BF16 vector. Other rows are a finite poison (not NaN, not a copy of row 0).
* Output buffer filled with `int16` `0x7FFF` before each launch. Decode `qscratch` is `Int64(0)`, as in `test_fp8_projection_decode_matches_routed`.
* `fp8_rows` is the projection's own cap (`max_gemv_rows`, or `max_fp8_rows` for the switch). On this pin that cap selects the same kernel as the baked scalar and as the output-row wrapper (a constant 16 and a 0-above-16 agree once `rows > 16`; a constant 32 and a 0-above-32 agree once `rows > 32`). The pytest does not execute the Rust scalar, the baked `Int32(FP8_GEMV_ROWS)`, or those wrappers. Raising only `wide_rows` makes this file go green while those callers still pass 16 or 0 and serving stays on TMA.
* Compare row 0 with `view(torch.int16)`. Assert the sentinel was overwritten on row 0 and on every other live row.

No xfail. 43 cases, 39 compiles (the four shared shapes above), 172 launches. On pin `91b104db` the expected colours are:

| Tag | M=16 | M=17 | M=32 | M=64 |
| --- | --- | --- | --- | --- |
| `s16`, `copy` | pass | fail | fail | fail |
| `s32` | pass | pass | pass | fail |

A green `s32` row at 17 and 32 is the evidence that two GEMV tiles with one `(warps, groups)` match. A red row there means the two-specialization fix below does not hold the bits. Not run.

## Fix

Change the decode branch of `Fp8Projection` once. Do not add another per-consumer threshold, and do not special-case KDA.

For every M in the verify range (1..64, and 128 on the wide Flash program), use one K order:

1. Keep today's thin GEMV for `rows ≤ 16`, with today's `(warps, groups)`.
2. Add a wide GEMV through the verify cap with **those same** `(warps, groups)`. `_w8`'s hardcoded `wide_warps=8, wide_groups=2` must not apply to a shape whose narrow pair is different. The half-head in-proj override `(8, 2)` stays: it is a warp config, so head-split partials keep the full-head fold. It is not a route.
3. Stop calling decode `TmaFp8Gemm` inside that range.

The fat tile's `max_rows` is that program's compile capacity, not one global 64. A program compiled at 64 rows cannot launch 128. `--decode-rows 128` is a separate compile: `verify_budget` (`glmf_graphs.rs:19`) appends 128 on 188 SMs, 127 on 170, and 99 on 132 after the 64-row bucket. This pytest stops at 64 and does not launch the `_m128` program.

### Raising `wide_rows` is not the route

`limit = min(fp8_rows, max_gemv_rows)` (`_fp8_weights.py:183`). `_Fp8Switch` clamps the same way (`glmf.py:218`). A fatter GEMV never runs while the scalar stays at 16, or at 0:

* GLM Flash full-head KDA and the non-KDA projections pass Rust `FP8_ROWS` (`engine.rs:88`, value 16) from `kda_w8` (`:4187`) and `fp8_scalars` (`:4208`). Half-head KDA passes 32.
* GLM 5.3 decode bakes `Int32(FP8_GEMV_ROWS)` in the kernel: `_ProducerW8Decode` `glm_attention.py:361`, `_IndexProducerW8Decode` `:416`, `_OutputW8Decode` `:461`, `_FfnW8Decode` `glm_ffn.py:194`. Changing the Rust scalar does not move these.
* Qwen decode bakes the same constant: `_GdnW8Decode` `qwen4_gdn.py:373`, `_ProducerW8Decode` `qwen4_attention.py:277`, `_AttnOW8Decode` `:683`.
* MiMo decode passes runtime `FP8_DECODE_ROWS` (`mimo_v2/engine.rs:180`, value 32) from `w8_scalars` (`:2561`) and the o_proj launch (`:2537`).
* `GlmfOutputRows` (`_glmf_output_rows.py:41`) passes 32 only while `total_rows <= 32`, else 0. `GlmfOutputShard` (`_glmf_output_shard.py:66`) always passes 32. `GlmfMlaOutputRows` (`_glmf_mla_output_rows.py:57`) passes `FP8_ROWS` only while `total_rows <= 16`, else 0.

Each of those scalars has to become that program's capacity, together with `wide_rows`. The half-head `(8, 2)` warp override stays.

`check_bucket_thresholds` (`decode_graph.rs:36`) rejects a threshold that sits strictly between two buckets (`previous < threshold && threshold < bucket`). A threshold equal to 64 does not straddle `[1, 4, 8, 16, 32, 64]`, and it does not straddle the wide speculative set that appends 128 after 64: `64 < 64` is false. The set that fails is one that jumps over 64. `[32, 72]` and `[..., 32, 72, 127]` already do, for `glmf.decode_capacity[m64|m128]` (`engine.rs:5305`). Register each program's crossover at that program's own capacity: 64 on the `_m64` programs, the verify budget on the `_m128` programs. Leave the old 16/32 entries behind only while the kernel still switches there. Raising the registry while the scalar is still 16 would claim one kernel across a launch that still switches.

C1 stays on the thin GEMV (no change at M=1) only if the new test shows the thin and wide tiles match. That is the reason `s32` is in the test at M=17 and M=32.

Fallback, already the route #173 proved for KDA: `TmaFp8Gemm` at every M, including 1. Use it only if the GEMV-everywhere proof fails.

Leave prefill, the LM head, and the dual-copy classes alone in that patch.

### C1 of forcing TMA at M=1

No new measurement. Figures are the comments already in the pin, RTX PRO 6000 at 325 W, L2-cold. GEMV times are M=1. TMA times are the 17-row column of `decode_tile_n` (`_fp8_weights.py:202`); the grid does not change between M=1 and M=17.

| Projection | GEMV M=1 | TMA at 17 (chosen tile) | Ratio |
| --- | --- | --- | --- |
| GLM o `6144×16384`, 4 warps | 68 µs (`gemv_warps`, `:264`) | 125 µs (tile 64) | 1.84× |
| GLM q_a\|kv_a `2624×6144`, 8 warps | 14 µs | 37 µs (tile 32) | 2.64× |
| GLM dense down `6144×12288`, 4 warps | 53 µs | 94 µs (tile 64) | 1.77× |

Flash has tuned GEMV times and no paired TMA (`FP8_GEMV_CONFIG`, `glmf.py:235`): KDA in `24896×4096` 62.7 µs, KDA o `4096×8192` 22.5 µs, both at M=1. The same weight bytes move through either kernel, so the 1.8–2.6× band is the right ceiling, not a measured step. The two KDA projections are ~85 µs of GEMV; a ~2× TMA tax is on the order of +85 µs per KDA layer. Attention, routed MoE, and the KDA recurrence are unchanged. This is not a token-delta measurement.

`decode_tile_n` also records the rejected tile-128 TMA, which is slower than the chosen tile (o 188 µs vs 125, q_a\|kv_a 74 vs 37, down 145 vs 94, at 17 rows).

## Gates

* CPU: `python -m py_compile` on the test, and a loose import that executes the case-table checks (43 cases, s16=28, s32=13, copy=2). Both passed again after the structural assert stopped freezing the 16/32 cap.
* GPU pytest: not run. The file skips without SM120/SM121.
* No defaults changed. No submodule pointer change. No image. Not merged.
