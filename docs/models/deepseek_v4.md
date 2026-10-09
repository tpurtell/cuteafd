# DeepSeek V4 (Flash / Pro)

Re-hosted on CuteAFD's generic per-layer engine rather than the legacy
`real_full` path. Release cards measure each checkpoint against CuteAFD's
own history, not an external-engine parity campaign.

## Supported checkpoints / quants

- `deepseek-ai/DeepSeek-V4-Flash-0731` — FP8 128x128-block coordinator
  weights, native MXFP4 routed experts.
- `wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1` — official
  FP8-named dense tensors with HF-named EXL3 K2 routed experts (~96 GiB per
  Spark rank at TP4).

## Engineering summary

- Attention: compressed MLA with an alternating 4/128 ratio schedule, an
  indexer compressor on each ratio-4 layer, no CED KV sharing; FP8 blocks
  are 128x128 (V4.1 uses 32x32).
- Router: hash routing (`ffn.gate.tid2eid`) on the first `num_hash_layers`,
  sqrtsoftplus noaux_tc scoring on the rest.
- Routed experts run on 2, 3, 4 or 6 Spark ranks over `expertd-native`:
  native MXFP4 for V4 Flash (hidden 4096 / intermediate 2048 / 256 experts),
  EXL3 K2–K4 for V4 Pro (hidden 7168 / intermediate 3072 / 384 experts).
  mHC adds an `hc_head_{fn,base,scale}` output head beyond V4.1's mixing.
- Speculator: three-stage dSpark drafter at this family's width, reusing
  V4.1's structure with family-specific geometry.
- KV format: FP8 128x128 block scales (UE8M0) on coordinator weights.
- RTX/Spark layouts: head split is the default on 2 RTX (measured decode
  -10% Flash / -12% Pro, prefill neutral); TP6 across all six Sparks is an
  option for V4 Pro when it divides the model's width better than TP4.
- Prefix cache: merged — 256-token units covering the C4, index and C128
  pages of one index, per-layer SWA-window marks, dSpark rings included so
  drafts stay warm, plus the compressors' FP32 rolling state.

## Known limits

<!-- release-v2-limits -->
- **Memory vs. kernel support:** the planner reports 1,048,576 checkpoint context tokens, but rc1's compiled index extent is 131,072. The frozen V4 engine derives its C128 row stride and context from that manifest, and the scheduler rejects prompts at/above 131,072 and clamps generation there. Flash and Pro maximum requested 1M at launch; their effective context is 131,072, corrected in cards with the requested value retained as provenance. Their C8/fidelity measurements remain valid: the runtime geometry matches the compiled extent. Pro minimum's 600 s timeout is retained; its 131,072-token correction is separate. rc2 raises the compiled extent to 1M to meet the 256K agentic-context floor. Pro's 31.8 GiB no-fit is a coordinator-memory rejection even with six Sparks, not evidence of a missing TP kernel.
- **Memory-placement limit:** automatic KV sizing follows resident routed-expert placement on GPU0, rather than reserving the 2M pool first. Saved rc1 planner estimates are Flash 903,168/720,384 tokens on one/two RTX and Pro 731,648/489,984; all fall below the 1M checkpoint context, so default usable context is pool-limited. Corrected runtime cards separately report Flash 774,400/581,376 and Pro maximum 1,387,264 tokens; planner estimates are not measured capacities. In the two-RTX plans, GPU1 uses only 13.10 GiB (Flash) or 24.01 GiB (Pro) of about 89-90 GiB, with no resident routed layers there. Planned fix: V4.1's pool-first 2M rule and spreading resident experts across both GPUs. This is a memory-placement limit, not a compiled-kernel extent limit.
- V4 Pro EXL3 K2 measures KL 0.059-0.061 against a 0.06 gate; at the threshold on min. rc1 maximum passes at KL 0.0596 / top-1 92.6%; minimum fails at KL 0.0605 / top-1 92.0%. The same checkpoint differs in placement, coordinator-resident layer count and reduction order. This threshold-straddling quant result is not established as a regression; the minimum card stays FAIL and thresholds are unchanged.
- V4 Flash fidelity varies between unchanged cold launches (historical top-1 96.552% vs 96.937%); deterministic index top-k does not remove expert-reduction/verify nondeterminism.
- `wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1` is not supported on a single 32 GB card with all six available Sparks; needs a larger coordinator card or a different checkpoint/placement. rc1 planner: coordinator weights need 46.0 GiB, over the 32 GiB coordinator budget; rtx0 full memory layout needs 62112178443 bytes, budget 29724160841 bytes, shortfall 32388017602 bytes. See the linked planner-only 5090 cell; no performance was measured.

- V4 Pro's EXL3 prefill is Spark-compute bound; TP6 raises decode but
  coordinator intake of partial rows is the current prefill bottleneck on
  some layouts (see `PLAN.md` Spark-side reduction notes).
- V4 Pro's rc1 quick golden fidelity check passes on the measured maximum
  layout; it does not establish batch invariance. Speculative and plain
  greedy outputs, and C1/C4 greedy outputs, can differ; verify rounding
  remains open.
- V4 Flash prompt and turn-end prefix restores are byte-exact against their
  own snapshots; cold prefill can differ through arrival-ordered Spark FP32
  atomic reductions. Deterministic prefill and verify are deferred.
- V4 Flash's native TP2 layout is qualified with legacy expert requests;
  the opt-in device exchange does not support that wire geometry.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v2 | 2026-10-09 | Unified runtime SM120 launch sizing and probed GPU/RDMA paths; launch-admitted full-prefill fidelity scoring; shared media/benchmark admission and C8 release cards. | Pending rc1 exports |
| v1 | 2026-10-04 | Qualify native Flash TP2 on legacy expert requests; honor explicit local-expert placement; compressed-cache and drafter admission; exact turn-end restore check. | <a href="../../benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/report.svg"><img src="../../benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-1rtx-2spark-v4-flash-min/card.svg" width="360" alt="DeepSeek-V4-Flash-0731 (mxfp4-g32) (min)"></a> <a href="../../benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max/report.svg"><img src="../../benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-flash-0731-2rtx-4spark-v4-flash-max/card.svg" width="360" alt="DeepSeek-V4-Flash-0731 (mxfp4-g32) (max)"></a> <a href="../../benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min/report.svg"><img src="../../benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark-v4-pro-exl3-min/card.svg" width="360" alt="DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) (min)"></a> <a href="../../benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/report.svg"><img src="../../benchmarks/deepseek_v4/2026-10-04-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark-v4-pro-exl3-max/card.svg" width="360" alt="DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) (max)"></a> |
| v0 | 2026-10-02 | First release | <a href="../../benchmarks/deepseek_v4/2026-10-02-smoke-deepseek-v4-flash-0731-1rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v4/2026-10-02-smoke-deepseek-v4-flash-0731-1rtx-4spark/card.svg" width="360" alt="DeepSeek-V4-Flash-0731 (mxfp4-g32) (min)"></a> <a href="../../benchmarks/deepseek_v4/2026-10-02-smoke-deepseek-v4-flash-0731-2rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v4/2026-10-02-smoke-deepseek-v4-flash-0731-2rtx-4spark/card.svg" width="360" alt="DeepSeek-V4-Flash-0731 (mxfp4-g32) (max)"></a> <a href="../../benchmarks/deepseek_v4/2026-10-02-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark/report.svg"><img src="../../benchmarks/deepseek_v4/2026-10-02-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-1rtx-4spark/card.svg" width="360" alt="DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) (min)"></a> <a href="../../benchmarks/deepseek_v4/2026-10-02-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark/report.svg"><img src="../../benchmarks/deepseek_v4/2026-10-02-smoke-deepseek-v4-pro-0813-exl3-k2-calibrated-v1-2rtx-6spark/card.svg" width="360" alt="DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1 (exl3-k2) (max)"></a> |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
