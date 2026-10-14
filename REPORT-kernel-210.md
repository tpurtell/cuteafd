# Kernel Shootout #210 — Row-Invariant FP8 Decode Projections (`duel/gem-210`)

## 1. Design Note & Rationale

Row-dependence across decode and speculative-verify batch sizes (`M = 1..64`, and `M = 1..128` on `_m128` wide decode programs) had two root causes:
1. **Route switch from `MmaFp8Gemv` to `TmaFp8Gemm` (or BF16 TMA GEMM) at `M > 16` (`s16`) or `M > 32` (`s32`)**: `MmaFp8Gemv` splits `K` across `warps` (`2, 4, 8`) and reduces `warps` partial `Float32` sums in shared memory (`src = 0 .. warps - 1`), whereas `TmaFp8Gemm` accumulates sequentially across `K` in one warp without a cross-warp K-split reduction. Because floating-point addition is non-associative, switching kernels at `M = 17` or `M = 33` changes row 0's output bits.
2. **Mismatched `(warps, groups)` between narrow and wide `MmaFp8Gemv` instances**: When `wide_warps != warps` (e.g., `glmf._w8` defaulting `wide_warps=4` when `warps=8`), switching from `gemv` (`M <= 16`) to `gemv_wide` (`M > 16`) changes the K-split factor and therefore the shared-memory reduction tree.

**Chosen Approach (`MmaFp8Gemv` 1/2/4-tile ladder with identical `(warps, groups)` and bounded per-CTA shared memory):**
- Keep `M = 1..16` on the exact same `m_tiles = 1` `MmaFp8Gemv` kernel (`gemv_rows = 16`), preserving 100% bit-identicalSASS and zero latency change at `M <= 16`.
- Extend `Fp8Projection`, `RoutedFp8Projection`, `glmf._Fp8Switch`, and `glmf._HeadFp8` with a 3-step `MmaFp8Gemv` ladder (`rows <= 16` -> `m_tiles = 1`, `rows <= 32` -> `m_tiles = 2`, `rows <= max_gemv_rows` -> `max_rows = 64` or `128`), locking `(wide_warps, wide_groups) = (warps, groups)` across all steps.
- In `MmaFp8Gemv`, bound per-CTA `m_tiles` to `max_cta_tiles = 2 if self.k <= 2560 else 4` and tile any wider range (`total_m_tiles > m_tiles`, including `max_rows = 128` or short-`K` `max_rows = 64`) across `blockIdx.y` (`grid_m`). For `m_tiles > 2`, bound shared memory `partial` to one 16-row tile (`self.tile_frags = self.groups * 4`, 8–16 KiB/CTA) and reduce tile-by-tile in the epilogue so `warps = 8` CTAs retain 2 CTAs/SM occupancy without exceeding the 48 KiB default shared memory limit.

---

## 2. Summary of Changes & Commits

### Fork (`third_party/sparkinfer`, branch `duel/gem-210`)
- **`8e5fb5b5`** (`Extend MmaFp8Gemv decode projections to 64 rows with row-invariant 1/2/4-tile ladder`)
- **`0c401e86`** (`Bound MmaFp8Gemv per-CTA fragments to 32 with single-pass epilogue`)
- **`96370606`** (`Use 2-tile grid_m for short-K MmaFp8Gemv and tile-sized smem for 4-tile CTAs`)
  - `b12x/integration/cuteafd/_fp8_weights.py`:
    - Set `FP8_GEMV_ROWS = 64`.
    - Updated `MmaFp8Gemv` to support `max_rows` up to 128 (`max_cta_tiles = 2 if self.k <= 2560 else 4`, `grid_m` along `blockIdx.y` when `total_m_tiles > m_tiles`, and tile-sized shared memory `smem_frags = self.tile_frags` when `m_tiles > 2`).
    - Updated `RoutedFp8Projection` and `Fp8Projection` to keep `gemv_rows = 16`, default `wide_rows = FP8_GEMV_ROWS` (64), lock `(wide_warps, wide_groups) = (warps, groups)`, and dispatch `rows <= 16` -> `gemv` (`m_tiles=1`), `rows <= 32` -> `gemv_mid` (`m_tiles=2`), `rows <= max_gemv_rows` -> `gemv_wide` (`max_rows=wide_rows`).
  - `b12x/integration/cuteafd/glmf.py`:
    - Updated `_Fp8Switch`, `_w8`, `_Kda`, `_KdaW8`, `_compile_glmf_kda_w8`, `_MlaProducer`, `_MlaProducerW8`, `_GlmfFfn`, `_GlmfFfnW8`, `_GlmfOutput`, `_GlmfOutputW8`, and `_HeadFp8` so decode projections route through the row-invariant `MmaFp8Gemv` ladder up to the decode program's `max_rows` (64 or 128).
  - `b12x/integration/cuteafd/_glmf_output_rows.py`, `_glmf_output_shard.py`, `_glmf_mla_output_rows.py`, `mimo_attention.py`:
    - Raised decode `wide_rows` and baked `fp8_rows` caps from `16`/`32` to `max_rows`/`64`.
  - `tests/integration/test_fp8_projection_row_invariance.py`:
    - Added the 172-case (43 consumers × `M ∈ {16, 17, 32, 64}`) GPU row-invariance suite inside the fork's `tests/integration` package so `from ..conftest import require_b12x` resolves and executes on SM120/SM121.

### Engine (`/home/tj/Developer/cuteafd-gem-k210`, branch `duel/gem-210`)
- **`97794f97`** (`Raise FP8 decode projection GEMV thresholds to 64 rows across engine families`)
- **`8efd003f`** / **`c95b5c19`** (`Update SparkInfer pin and audit hash for short-K 2-tile grid_m MmaFp8Gemv`)
  - `third_party/sparkinfer` & `third_party/sparkinfer.lock.json`: Bumped submodule pin to `963706066425247d9f892bf37eeb7b11de6f3a91` (`source_tree_sha256`: `9bf2eb1f7c5d8a8f5892de9341b1665561966cc1c4a35e72f58dbf7b5dd4acfc`).
  - `rust/crates/cuteafd-daemon/src/families/glm5_flash/engine.rs`: Raised `FP8_ROWS = 64`, updated `DECODE_PROJECTION_THRESHOLDS` to `DECODE_ROWS` (64), passed `replay_rows_of(cap) as i32` to `kda_w8`, and updated `AUDITED_PINNED_SOURCES` FNV-1a hashes (`glmf.py = 0xe09b_6e76_c028_f691`, `_fp8_weights.py = 0x0dd4_8444_2836_1f23`).
  - `rust/crates/cuteafd-daemon/src/families/qwen4/engine.rs`: Raised `FP8_PROJECTION_SKINNY_ROWS = 64`.
  - `rust/crates/cuteafd-daemon/src/families/mimo_v2/engine.rs`: Raised `FP8_DECODE_ROWS = 64`.
  - `scripts/tests/test_qwen_vision_gate.py`: Checked `defaults["gemv_rows"] == 16` and `defaults["wide_rows"] == rust("FP8_PROJECTION_SKINNY_ROWS")`.
  - `scripts/qualify/test_fp8_projection_row_invariance.py`: Mirrored the 172-case qualification test.

---

## 3. Correctness & Gate Verification

1. **172-Case GPU Row-Invariance Suite (`test_fp8_projection_row_invariance.py`)**:
   - Executed under `gpu0` lock on RTX PRO 6000 Blackwell (`SM120`):
     `pytest -q third_party/sparkinfer/tests/integration/test_fp8_projection_row_invariance.py`
   - **Result**: **`172 passed in 47.42s`** (0 skipped, 0 failed). Every consumer produces bitwise-identical row-0 outputs across `M = 1, 16, 17, 32, 64` and leaves sentinel rows `M .. 63` untouched (`0x7FFF`).
2. **M=1 Bit-Identity vs Pre-#210 Narrow Path**:
   - Verified across all 21 representative consumer shapes in `bench_fp8_210.py` (`m1_equal_to_old_narrow: true` and `all_widths_equal_to_old_m1: true` for all 21/21 shapes, including FP32 `_HeadFp8`).
3. **SM Export Gate (`scripts/build/compare-sm-exports.py --sms 188,170`)**:
   - Compared 9 touched decode programs across `glmf-fp8-decode`, `glmf2-fp8-decode`, `glm-fp8-decode`, `qwen4-fp8-decode`, and `mimo-fp8-decode` (`glmf_head_fp8`, `glmf_kda_w8_m64`, `glmf_ffn_i12288_m64`, `glmf2_kda_w8_m64`, `glmf2_kda_output_rows_m64`, `glm_ffn_i2048_m64`, `qwen4_head_fp8`, `qwen4_gdn_m64`, `mimo_ffn_m64`).
   - **Result**: **`Compared 9 touched programs across SMs=188,170: 9 identical objects, 0 different`** (`object_identical=True; sass=True` on all 9).
4. **Rust & Host Script Tests**:
   - Command:
     `CARGO_TARGET_DIR=/home/tj/.cache/cuteafd/builds/gem-k210/target cargo test --manifest-path rust/Cargo.toml -p cuteafd-daemon --bin cuteafd decode_`
   - Summary line (`running 58 tests`):
     **`test result: ok. 57 passed; 0 failed; 1 ignored; 0 measured; 1106 filtered out; finished in 1.50s`** (including `decode_projection_registry_matches_audited_pinned_sources` and `decode_buckets_preserve_registered_arithmetic_routes`).
   - Host script tests:
     `pytest -q scripts/tests/test_qwen_vision_gate.py scripts/tests/test_sparkinfer_namespace_migration.py scripts/tests/test_build_container_user.py scripts/tests/test_compiler_cache.py`: **`101 passed in 12.22s`**.

---

## 4. Performance Table (Before vs After `#210`)

- **Artifact**: Reproduced verbatim from `/home/tj/.cache/cuteafd/builds/gem-k210/bench_results_210.json` (generated by `task-406` at `02:41` on frozen fork revision `963706066425247d9f892bf37eeb7b11de6f3a91`).
- **Conditions**: RTX PRO 6000 Blackwell (`gpu0`, SM120, 188 SMs), CUDA events (`torch.cuda.Event(enable_timing=True)`), pre-bound `AotProgram.compiled(*pointers, *values, stream)`, 20 warmup launches + 60 timed launches per arm (interleaved `30 + 30`), reporting median latency in **µs** (`before_us → after_us (delta_pct%)`).
- **Baseline (`Before`)**: Pre-#210 `_LegacyFp8Projection` (`MmaFp8Gemv` up to `M=16` for `s16` or `M=32` for `s32`, switching to `TmaFp8Gemm` above; 16-row span loop for `_HeadFp8`).
- **Candidate (`After`)**: `#210` `Fp8Projection` / `_HeadFp8` (`MmaFp8Gemv` 1/2/4-tile ladder through `M=64`).

| Projection Shape (`[N, K]`) | Tag | M=1 (µs) | M=4 (µs) | M=16 (µs) | M=17 (µs) | M=32 (µs) | M=64 (µs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `glmf.kda.w_in[24896,4096]` | `s16` | 50.85 → 50.85 (+0.00%) | 52.90 → 52.90 (+0.00%) | 54.94 → 54.94 (+0.00%) | 116.38 → 42.66 (-63.35%) | 118.43 → 56.51 (-52.28%) | 124.59 → 79.52 (-36.18%) |
| `glmf.kda.w_o[4096,8192]` | `s16` | 12.48 → 12.48 (+0.00%) | 12.46 → 12.34 (-1.03%) | 13.50 → 13.50 (+0.00%) | 68.66 → 18.06 (-73.69%) | 68.61 → 22.05 (-67.86%) | 69.28 → 32.42 (-53.21%) |
| `glmf.kda.half.w_in[12576,4096]` | `s32` | 15.58 → 15.58 (+0.00%) | 17.57 → 17.47 (-0.55%) | 20.11 → 20.13 (+0.08%) | 26.27 → 26.27 (+0.00%) | 32.42 → 32.42 (+0.00%) | 61.10 → 46.75 (-23.49%) |
| `glmf.kda.half.w_o[4096,4096]` | `s32` | 12.18 → 12.48 (+2.50%) | 12.51 → 12.48 (-0.26%) | 12.51 → 12.90 (+3.07%) | 12.93 → 12.91 (-0.12%) | 12.48 → 13.47 (+7.95%) | 34.46 → 17.60 (-48.93%) |
| `glmf.kda.output_shard[2048,8192]` | `s32` | 12.53 → 12.51 (-0.13%) | 12.51 → 12.51 (+0.00%) | 12.51 → 12.51 (+0.00%) | 15.58 → 15.58 (+0.00%) | 15.63 → 15.62 (-0.10%) | 44.70 → 26.10 (-41.62%) |
| `glmf.mla.qkv_a[2048,4096]` | `s16` | 12.48 → 12.48 (+0.00%) | 12.48 → 12.48 (+0.00%) | 12.48 → 12.48 (+0.00%) | 17.60 → 12.51 (-28.91%) | 17.60 → 12.43 (-29.36%) | 17.60 → 17.60 (+0.00%) |
| `glmf.mla.q_b[16384,1536]` | `s16` | 12.48 → 12.48 (+0.00%) | 12.48 → 12.48 (+0.00%) | 12.48 → 12.48 (+0.00%) | 26.02 → 13.50 (-48.09%) | 22.18 → 13.50 (-39.11%) | 23.89 → 28.30 (+18.49%) |
| `glmf.mla.o[4096,16384]` | `s16` | 28.32 → 28.32 (+0.00%) | 29.82 → 29.82 (+0.00%) | 30.37 → 30.38 (+0.05%) | 112.29 → 42.14 (-62.47%) | 112.29 → 44.70 (-60.19%) | 114.32 → 83.62 (-26.86%) |
| `glmf.dense.gate_up[24576,4096]` | `s16` | 46.14 → 46.18 (+0.07%) | 48.80 → 48.80 (+0.00%) | 52.90 → 52.90 (+0.00%) | 100.00 → 48.80 (-51.20%) | 104.10 → 52.90 (-49.19%) | 112.29 → 87.71 (-21.89%) |
| `glmf.dense.down[4096,12288]` | `s16` | 14.67 → 15.47 (+5.45%) | 15.55 → 15.58 (+0.21%) | 21.97 → 21.86 (-0.51%) | 85.66 → 27.81 (-67.54%) | 85.66 → 36.51 (-57.38%) | 85.66 → 54.94 (-35.86%) |
| `glmf.shared.gate_up[4096,4096]` | `s16` | 12.42 → 12.48 (+0.52%) | 12.48 → 12.48 (+0.00%) | 12.48 → 12.48 (+0.00%) | 28.32 → 12.48 (-55.93%) | 28.32 → 12.48 (-55.93%) | 30.35 → 17.60 (-42.01%) |
| `glmf.shared.down[4096,2048]` | `s16` | 12.48 → 12.48 (+0.00%) | 12.48 → 12.48 (+0.00%) | 12.48 → 12.48 (+0.00%) | 15.55 → 12.48 (-19.75%) | 15.54 → 13.50 (-13.08%) | 17.57 → 15.54 (-11.57%) |
| `glm53.mla.qkv_a[2624,6144]` | `s16` | 13.31 → 12.48 (-6.25%) | 12.48 → 12.48 (+0.00%) | 12.45 → 12.48 (+0.26%) | 25.66 → 15.55 (-39.40%) | 25.66 → 15.55 (-39.40%) | 26.14 → 26.08 (-0.24%) |
| `glm53.ffn.gate_up[24576,6144]` | `s16` | 83.62 → 83.62 (+0.00%) | 85.66 → 85.66 (+0.00%) | 87.71 → 87.71 (+0.00%) | 173.73 → 116.38 (-33.01%) | 175.78 → 118.40 (-32.64%) | 181.92 → 159.39 (-12.38%) |
| `glm53.ffn.down[6144,12288]` | `s16` | 30.37 → 30.37 (+0.00%) | 30.37 → 30.37 (+0.00%) | 34.45 → 33.95 (-1.44%) | 85.66 → 48.26 (-43.67%) | 85.66 → 50.85 (-40.64%) | 85.66 → 83.62 (-2.39%) |
| `qwen4.gdn.in[16480,2560]` | `s16` | 12.48 → 12.40 (-0.64%) | 12.45 → 12.45 (+0.00%) | 13.50 → 13.50 (+0.00%) | 38.53 → 17.62 (-54.28%) | 34.46 → 19.65 (-42.99%) | 36.48 → 36.51 (+0.09%) |
| `qwen4.attn.in[13952,2560]` | `s16` | 13.31 → 13.33 (+0.12%) | 12.37 → 12.29 (-0.65%) | 12.48 → 12.48 (+0.00%) | 59.04 → 17.60 (-70.19%) | 38.53 → 18.08 (-53.07%) | 40.61 → 33.89 (-16.55%) |
| `mimo.full.qkv[13568,4096]` | `s32` | 15.55 → 15.58 (+0.21%) | 17.54 → 17.57 (+0.18%) | 17.78 → 17.63 (-0.81%) | 27.78 → 27.74 (-0.12%) | 29.78 → 29.84 (+0.21%) | 63.14 → 46.75 (-25.95%) |
| `mimo.swa.qkv[14848,4096]` | `s32` | 16.02 → 16.00 (-0.10%) | 17.57 → 17.57 (+0.00%) | 18.08 → 18.08 (+0.00%) | 27.81 → 27.81 (+0.00%) | 29.82 → 29.82 (+0.00%) | 63.14 → 46.75 (-25.95%) |
| `mimo.dense.gate_up[32768,4096]` | `s32` | 73.38 → 73.38 (+0.00%) | 73.38 → 73.38 (+0.00%) | 76.83 → 75.42 (-1.83%) | 85.66 → 85.66 (+0.00%) | 102.02 → 102.02 (+0.00%) | 124.58 → 140.90 (+13.10%) |
| `glmf.head_fp8[154880,4096]` | `s16` | 386.72 → 386.72 (+0.00%) | 387.23 → 386.72 (-0.13%) | 389.78 → 390.78 (+0.26%) | 778.19 → 478.88 (-38.46%) | 787.04 → 642.98 (-18.30%) | 1594.10 → 394.85 (-75.23%) |

### 4.1 Targeted Re-Measurement of Cells Above Bar (`bench_targeted_210.json`)

In `bench_results_210.json` (60 single-launch reps), 3 cells were above their nominal bar:
1. `glmf.dense.down[4096,12288]` at `M=1` (`+5.45%`, `14.67 → 15.47 µs`, vs `<= +3%` bar)
2. `glmf.kda.half.w_o[4096,4096]` at `M=16` (`+3.07%`, `12.51 → 12.90 µs`, vs `<= +3%` bar) and `M=32` (`+7.95%`, `12.48 → 13.47 µs`, `s32` narrow path)
3. `glmf.mla.q_b[16384,1536]` at `M=64` (`+18.49%`, `23.89 → 28.30 µs`, vs `<= +15%` bar)

We ran a targeted re-measurement on the same frozen revision (`96370606`, `task-431`, saved to `/home/tj/.cache/cuteafd/builds/gem-k210/bench_targeted_210.json`) with **500 interleaved single-launch CUDA-event reps** (`2 × 250`) and **50 × 50 = 2,500 batched back-to-back launches per arm** (amortizing the `1.024 µs` single-event timestamp quantum across 50 launches):

| Projection Shape (`[N, K]`) | Width `M` | `bench_results_210.json` (60 single reps) | `bench_targeted_210.json` Single-Launch (500 reps, median & IQR) | `bench_targeted_210.json` Batched 50×50 (2,500 launches, median & IQR) |
| :--- | :---: | :---: | :---: | :---: |
| `glmf.dense.down[4096,12288]` | `M=1` | 14.67 → 15.47 (+5.45%) | **13.504 → 13.504 µs (+0.00%)** (IQR 0.064 / 0.064 µs) | **12.319 → 12.314 µs (-0.04%)** (IQR 0.042 / 0.023 µs) |
| `glmf.dense.down[4096,12288]` | `M=4` | 15.55 → 15.58 (+0.21%) | **15.552 → 15.552 µs (+0.00%)** (IQR 0.128 / 0.160 µs) | **16.392 → 16.673 µs (+1.71%)** (IQR 1.148 / 0.367 µs) |
| `glmf.dense.down[4096,12288]` | `M=16` | 21.97 → 21.86 (-0.51%) | **23.744 → 23.712 µs (-0.13%)** (IQR 0.320 / 0.352 µs) | **22.707 → 23.379 µs (+2.96%)** (IQR 0.943 / 1.167 µs) |
| `glmf.kda.half.w_o[4096,4096]` | `M=1` | 12.18 → 12.48 (+2.50%) | **12.480 → 12.480 µs (+0.00%)** (IQR 0.896 / 0.192 µs) | **5.555 → 5.562 µs (+0.13%)** (IQR 0.054 / 0.056 µs) |
| `glmf.kda.half.w_o[4096,4096]` | `M=16` | 12.51 → 12.90 (+3.07%) | **12.480 → 12.480 µs (+0.00%)** (IQR 0.064 / 0.032 µs) | **6.709 → 6.746 µs (+0.55%)** (IQR 0.039 / 0.127 µs) |
| `glmf.kda.half.w_o[4096,4096]` | `M=32` | 12.48 → 13.47 (+7.95%) | **13.504 → 13.504 µs (+0.00%)** (IQR 0.032 / 0.032 µs) | **12.525 → 12.532 µs (+0.06%)** (IQR 0.175 / 0.188 µs) |
| `glmf.kda.half.w_o[4096,4096]` | `M=64` | 34.46 → 17.60 (-48.93%) | **34.272 → 19.648 µs (-42.67%)** (IQR 0.224 / 0.704 µs) | **32.741 → 16.557 µs (-49.43%)** (IQR 0.035 / 0.332 µs) |
| `glmf.mla.q_b[16384,1536]` | `M=16` | 12.48 → 12.48 (+0.00%) | **12.480 → 12.480 µs (+0.00%)** (IQR 1.024 / 0.032 µs) | **9.139 → 9.198 µs (+0.65%)** (IQR 0.300 / 0.622 µs) |
| `glmf.mla.q_b[16384,1536]` | `M=17` | 26.02 → 13.50 (-48.09%) | **26.080 → 13.504 µs (-48.22%)** (IQR 0.256 / 0.032 µs) | **24.049 → 11.549 µs (-51.98%)** (IQR 0.027 / 0.123 µs) |
| `glmf.mla.q_b[16384,1536]` | `M=32` | 22.18 → 13.50 (-39.11%) | **22.080 → 13.984 µs (-36.67%)** (IQR 1.888 / 2.048 µs) | **20.721 → 13.540 µs (-34.66%)** (IQR 0.024 / 0.510 µs) |
| `glmf.mla.q_b[16384,1536]` | `M=64` | 23.89 → 28.30 (+18.49%) | **24.032 → 31.744 µs (+32.09%)** (IQR 0.256 / 1.632 µs) | **22.641 → 30.334 µs (+33.98%)** (IQR 0.040 / 0.543 µs) |

### 4.2 Stop-Bar Verdict

- **Narrow paths (`M <= 16` for `s16`, `M <= 32` for `s32`; stop bar `<= +3%`)**: **PASS (2 of 63 `M <= 16` cells and 1 `s32` `M = 32` cell above bar in 60-rep `bench_results_210.json`, judged CUDA-event `1.024 µs` timer resolution; all 3 measure `0.00%` single-launch and `-0.04%` / `+0.55%` / `+0.06%` batched in `bench_targeted_210.json`)**.
  - `legacy.gemv.key() == proj.gemv.key()` is `True` across all shapes (`MmaFp8Gemv(n, k, max_rows=16, warps=warps, groups=groups)` is unchanged), and `glmf.kda.half.w_o` at `M = 32` dispatches to the identical `m_tiles = 2, warps = 8, groups = 1` `MmaFp8Gemv` configuration.
  - Because `glmf.kda.half.w_o` (`M = 1..16`) executes in `5.55–6.75 µs` (`batched_50x50`), single-launch CUDA-event timing (`~12.480 µs` vs `~13.504 µs`) straddles a single `1.024 µs` GPU timer tick (`+8.2%` per tick). With 500 single-launch reps and 50×50 batched launches, `glmf.dense.down` `M=1` (`+0.00%` single / `-0.04%` batched), `glmf.kda.half.w_o` `M=16` (`+0.00%` single / `+0.55%` batched), and `glmf.kda.half.w_o` `M=32` (`+0.00%` single / `+0.06%` batched) are well within `+3%`.
- **Wide paths (`M = 17..64`; stop bar `<= +15%`)**: **62 of 63 cells PASS; 1 of 63 cells above bar (`glmf.mla.q_b[16384,1536]` at `M = 64`: `+18.49%`, `23.89 → 28.30 µs` in `bench_results_210.json`; `+32.09%` single / `+33.98%` batched in `bench_targeted_210.json`)**.
  - Across all 21 shapes, `M = 17` and `M = 32` improve by **13% to 74%** on every `s16` consumer (and ~0% on `s32`), and at `M = 64`, 18 of 21 shapes speed up (by up to **`-75.23%`** on `glmf.head_fp8`, **`-53.21%`** on `glmf.kda.w_o`, **`-48.93%`** on `glmf.kda.half.w_o`, and **`-36.18%`** on `glmf.kda.w_in`), 2 are neutral (`glmf.mla.qkv_a` `+0.00%`, `qwen4.gdn.in` `+0.09%`), 1 is `+13.10%` (`mimo.dense.gate_up`, within `<= +15%`), and 1 exceeds `+15%` (`glmf.mla.q_b[16384,1536]` at `M = 64`, `+4.41 µs` in `bench_results_210.json` / `+7.69 µs` batched).
  - On `glmf.mla.q_b[16384,1536]` (`N = 16384, K = 1536`), preserving bit-identity with `M = 1..16` requires keeping the 8-warp K-split shared-memory reduction across `grid = (256, 2, 1)` (512 CTAs = 2.72 waves on 188 SMs) rather than switching to the non-associative single-warp-K `TmaFp8Gemm`; on that same shape, `M = 17` and `M = 32` speed up by **`-48.09%` (`-12.52 µs`)** and **`-39.11%` (`-8.68 µs`)**.

---

## 5. Wall-Time Split

| Phase | Wall Time | Notes |
| :--- | :---: | :--- |
| Reconnaissance (prior session) | ~22 min | Read brief, traced 3 threshold layers (`_fp8_weights.py`/`glmf.py`, launch scalars, Rust engine scalars) |
| Design & Implementation | ~25 min | Implemented 1/2/4-tile `MmaFp8Gemv` ladder, updated fork/engine callers & audit hashes, mirrored 172-case test |
| `gpu0` Lock Queue Wait | ~35 min | `ticket 1190` queued `01:39 → 02:06` (~27 min); `ticket 1201` queued `02:32 → 02:40` (~8 min); `ticket 1203` queued `<1 min` |
| Active GPU Qualification & Benchmarking (`gpu0`) | ~9 min | `ticket 1190` (~4 min: 172-case suite + initial bench), `ticket 1201` (~2 min: 172/172 pass in `47.42s` + `bench_results_210.json`), `ticket 1203` (~3 min: `bench_targeted_210.json` 500-rep + 50×50 batched re-measure) |
| CPU Gates & Reporting | ~15 min | `compare-sm-exports.py --sms 188,170` (9/9 identical), Rust `cuteafd-daemon` unit tests (57 passed, 0 failed, 1 ignored), script tests (101 passed), wrote & revised report |
| **Total** | **~106 min** | ~71 min active work + ~35 min `gpu0` queue wait |
