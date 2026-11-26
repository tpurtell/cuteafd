# Inkling 2 Mini

Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, `Inkling2LoopForConditionalGeneration`,
`model_type: inkling2_loop`) is CuteAFD's first looped transformer family (`inkling2_mini`),
serving TP1 locally on **1× RTX PRO 6000 (0 Sparks)** ([#2701](https://github.com/tpurtell/cuteafd/issues/2701)).

Automatic KV sizing uses the shared 2M-token target on every card. Placement is
fixed at that target (all 24 physical routed-expert layers resident on GPU 0) and
the global paged KV pool grows into remaining admitted memory, within native
encoding limits. See [memory policy](../options.md#memory-and-pools).

## Architecture

- **Looped backbone:** 24 physical layers executed across **48 visits**:
  - Prelude: physical layers `0..6` (visits `0..6`).
  - Looped core: physical layers `6..18` executed across **3 passes** (visits `6..18`, `18..30`, `30..42`).
    At visit 5 (end of prelude), the pre-core residual state and block history are saved;
    before core passes 1 and 2 (visits 18 and 30), the residual stream restarts from that
    pre-core state while retaining the previous pass's 2-block summary (`h_after_L11`, `h_after_L17`)
    and injecting gated cross-pass messages through per-layer rank-64 controllers (`attn`, `k`, `v`, `mlp`).
  - Coda: physical layers `18..24` (visits `42..48`).
- **48 distinct cache slots:** every visit owns its own KV and short-convolution state:
  - **8 global paged KV slots** (`[5, 11, 17, 23, 29, 35, 41, 47]`), FP8 (default) or BF16.
  - **40 local sliding-window-1024 slots**, BF16 rings.
  - **48 contiguous FP32 short-convolution lag slots** (width `6656 = 2304 + 2304 + 1024 + 1024`, 3 lags per slot).
- **Attention:** GQA 24/8 × 128 (`H=2304`, group size 3), relative-position bias (`d_rel=16`,
  local window 1024, global extent 1024), per-token per-head attention sinks, sigmoid output gate,
  Q/K L2 clip (`22.627 = 2*sqrt(128)`), R L2 clip (`8.0`), global log-scaling (`floor=8192`, `alpha=0.1`),
  and short causal K/V convolutions (static on local layers; static + dynamic per-token taps on global layers).
- **MoE:** 128 routed experts, top-8, intermediate 1152, no shared expert; sigmoid + F32 bias
  selection, log-sigmoid softmax weights × `8.0`, interleaved `w13` gate/up with SiTU softcaps
  (`gate=4`, `up=25`), route weight applied before `w2` input quantization.
  - Physical layers `0..20`: native **NVFP4 W4A4** (packed E2M1, E4M3 per-16 scales, F32 per-expert amax
    and outer scale denominator; per-token amax activations with `fouroversix_mae` using exact `div.rn`).
  - Physical layers `20..24`: native **MXFP8 W8A8** (E4M3 weights, UE8M0 scales per 32-wide K group).
- **Head:** untied BF16 `embed`/`unembed` (`[201024, 2304]`), logits trimmed to `200058`, F32 unembed
  projection with softcap `30.0`.
- **Bundled MTP drafter (`mtp.safetensors`):** 3-layer recurrent dSpark/Markov drafter (`block_size=8`,
  `markov_rank=256`, target visit taps `[5, 23, 41, 48]`), own packed arena, history draft policy.
- **Bundled vision tower:** 23-layer PE-Lang ViT (`1024` hidden, 16 heads, patch 20) + 2×2 merge
  projector (`proj_out=[2304, 4096]`), own packed arena, batched images and prefix-hash spans
  through the shared media path. `VISION=auto` admits the tower when the KV pool still meets
  `KV_MIN_TOKENS`; explicit `VISION=off` leaves it unloaded.

## Authoritative Precision Contract (`f1940f40eb`)

The engine matches the checkpoint's `hf_quant_config.json` parameter groups and official
`transformers@d8f10a82` F32 gate/blend islands (authoritative reference oracle commit **`f1940f40eb`**):

- **BF16 compute:** embedding and layer state; dense projections (`wq_du`, `wr_du`, `wk_dv`, `wv_dv`,
  `gate_attn`, `wo_ud`); QK and PV GEMMs; loop controller `down_proj` GEMM and its rank-64 result.
- **F32 compute:** declared `attn_res` queries (`attn_queries`, `mlp_queries`, `output_query`),
  `middle_cycle_cross_pass_*queries`, and `mlp.gate.bias`; RMS, L2-clip, and log-scale internals;
  residual block and cross-pass message retrieval/mixing; loop controller `forget`/`gain` GEMMs plus
  `tanh` and blend; static and dynamic convolution weights, state, and reductions; router logits,
  scores, and gamma; `attn_sink_proj` sinks; combined main + sink attention softmax (sink dropped,
  probabilities cast to BF16 for PV); output-gate sigmoid and multiply (cast back to BF16).
- **Experts:** native NVFP4 W4A4 (`fouroversix_mae` with exact `div.rn`, bit-exact against oracle
  commit `6bcb2a8da0`/`ece18287ff`) and native MXFP8 W8A8 with IEEE F32 math, BF16 FC1 and SiTU
  epilogues, F32 expert combine, BF16 output.
  - **NVFP4 epilogue scale (exact vendor association):**
    `r = rcp_rn(1536.0 * denom)` (hoisted once per CTA via IEEE `div.rn.f32(1.0, 1536.0 * denom)`),
    `factor = (x_amax * w_amax) * r` (literal multiply order per row, no compiler reassociation),
    `out = bf16(acc * factor)` — matching pinned vendor vLLM (`WoosukKwon/vllm @ d664cd1e` /
    `vllm-project/vllm @ a0c1f162`, `ops/moe.py:347-353` and `ops/activation.py:155-171`) and
    SGLang (`sgl-project/sglang @ ec5e7ff5`, `ops/inkling2/moe.py:147-150` and `nvfp4.py:256-257`).
- **Head:** BF16 stored `unembed.weight` × F32 projection and logits, F32 softcap `30.0`.

## Measured SM120 MoE Row Curves

Isolated SM120 (RTX PRO 6000, `capacity=4096`, `--route auto --graph --check --check-prefix-rows 16`, TP1)
kernel row curves — NVFP4 measured on merged `#431` (`429ececda5`, literal vendor epilogue scale association),
MXFP8 unchanged since `#430` (`306d4902be`) — strictly monotonic in `us/row`, zero `>10%` jumps, and
`prefix_comparison.bitwise_equal=true` (`rows=16`, `differing_rows=0`, `max_abs_diff=0.0`) on every wide shape
(`17..4096`):

| rows | NVFP4 W4A4 (`bf16`, `429ececd`) us | us/row | GB/s | TFLOPS | MXFP8 W8A8 (`wire`) us | us/row | GB/s | TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40.96 | 40.960 | 874.8 | 3.1 | 55.30 | 55.296 | 1188.0 | 2.3 |
| 8 | 172.03 | 21.504 | 1352.2 | 5.9 | 276.48 | 34.560 | 1542.5 | 3.7 |
| 16 | 265.22 | 16.576 | 1424.9 | 7.7 | 440.32 | 27.520 | 1573.5 | 4.6 |
| 17 | 264.19 | 15.541 | 1437.9 | 8.2 | 432.16 | 25.421 | 1611.5 | 5.0 |
| 64 | 389.12 | 6.080 | 1456.1 | 21.0 | 656.42 | 10.256 | 1582.5 | 12.4 |
| 256 | 411.65 | 1.608 | 1392.7 | 79.2 | 675.84 | 2.640 | 1555.2 | 48.3 |
| 1024 | 495.62 | 0.484 | 1156.8 | 263.2 | 717.82 | 0.701 | 1464.2 | 181.7 |
| 2048 | 752.64 | 0.368 | 761.7 | 346.7 | 978.98 | 0.478 | 1073.6 | 266.5 |
| 2752 | 830.46 | 0.302 | 690.3 | 422.2 | 1183.71 | 0.430 | 887.9 | 296.2 |
| 4096 | 1162.24 | 0.284 | 493.3 | 449.0 | 1626.14 | 0.397 | 646.4 | 320.9 |

## Checkpoints

| Checkpoint | Experts | Core |
|---|---|---|
| `thinkingmachines/Inkling-2-Mini` @ `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` | layers `0..20` native NVFP4 W4A4 (`fouroversix_mae`), layers `20..24` native MXFP8 W8A8 | BF16 + official F32 islands |

## Serving and Qualification

- Public serve command: `cuteafd serve --family inkling2_mini` (`serve-inkling2-mini`) or
  `./run.sh --config examples/configs/inkling2-mini-1x-rtx.config --wip SLOT`.
- Public golden command: `cuteafd golden --family inkling2_mini` (`inkling2-mini-golden`) via
  `scripts/bench/golden-run.py --family inkling2_mini`.
- **Fidelity gate (3–6 bpw tier):** Mini's effective routed-expert bpw is `4.67–5.13` (20 NVFP4 + 4 MXFP8),
  measured against published reference set `wrldsuksgo2mars/cuteafd-fidelity` @
  `eabfaec64c7e96206ac1dffbbe51ad6deb73f635` (`inkling2-mini-v1_20261124_own-text-native`, manifest
  SHA-256 `a7276a3e4f10653614bfe8b591acc3b32878d8f88ff1bdf5d9cf22f189ade449`, 64 windows, pooled NLL `0.79864`):
  pooled generated-row **KL ≤ 0.05 nats** and **top-1 ≥ 0.93** (standard tier = 64 decode windows;
  quick tier = 8 windows). Byte-exact gates apply to `N=1` vs `N` width invariance, graph vs eager parity,
  exact 48-slot prefix restore, and same-input QDQ kernel oracles (`div.rn` tie vector + exact-decode negative control).
- Monolithic 48-visit exact-row CUDA decode/verify graphs (`inkling2_mini.step`, keyed by `(rows, mode, taps)`) bounded by the shared
  `#2697` soft-cap working-set reserve scaled to the resolved key space (`Decode 1..=clamp(CONCURRENCY, 1, 64)` plus `Verify 1..=64` with `dspark`, `8,306,688` B/key, `10 MiB` capture/step growth; `CUTEAFD_INKLING2_MINI_GRAPHS=1|0`).
- Exact 48-slot prefix cache (prompt-end and turn-end marks, host offload).
