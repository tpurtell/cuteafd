# Qwen 3.8 Flash Next

`qwen4_exp`: Gated DeltaNet (GDN) linear attention with a full-attention
layer every fourth, a PLE n-gram memory table, and fused expert tensors.

## Supported checkpoints / quants

- `Qwen/Qwen3.8-Flash-Next` official release — FP8 128x128-block routed
  experts (`qwen4:fp8`).
- EXL3 K4.25 PLE publications of the same checkpoint (`qwen4:exl3-k45`).
- NVIDIA ModelOpt NVFP4 — routed experts run W4A16 (`qwen4:nvfp4`).

## Engineering summary

- Attention: Gated DeltaNet linear recurrence on most layers, full GQA with
  an indexer every fourth layer.
- Shared expert with a sigmoid gate; hyper-connections (low-rank) mix the
  residual stream alongside the router.
- PLE n-gram memory table: a mapped table gathering 16 rows of 160 per token
  from pinned host RAM (or GPU), with bounded prefetch.
- Routed experts: 512 experts, top-10, hidden 2560 / intermediate 640, SiLU
  unclamped, stored as one fused `[experts, ...]` tensor per projection per
  layer; EXL3 K4/K5, FP8 128x128 blocks, or ModelOpt NVFP4 group-16; a local
  (RTX-resident, TP1) expert path is supported.
- Speculator: native MTP (full attention, 512 experts, and a hyper-connection
  feedback path). The launcher defaults to MTP3 with resident local EXL3
  experts; `SPECULATOR=off` disables it and `SPECULATOR_DEPTH` overrides the
  depth. The policy adapts the number of drafts to concurrency and acceptance.
- RTX/Spark layouts: qualified EXL3 fits on one RTX with resident local
  experts; the official FP8 expert package (~173 GB) does not fit one RTX.
  Spark EXL3 supports whole-expert TP1 and TP4 using uneven intermediate
  slices for the 640-wide experts. The qualified small-RTX layout keeps
  backbone experts on one Spark and the MTP layer's experts on the RTX.
- Prefix cache: merged — 256-row units over the full-attention layers, a
  combined GDN-state + PLE mark, with n-gram history recomputed from token
  ids rather than cached.

## Default precision (single residency)

Every weight has one resident format. Precision is chosen by measurement:
FP8 converts at load into the only copy where it is faster and the golden
stays within ~0.005 nat KL/NLL; drafters run FP8 when delivered tok/s is
higher and the speculation-lossless gate passes. Measured 2026-10-03,
then-current one-RTX reference, one warm launch per arm, `CONCURRENCY=4`, code tok/s
(C4 aggregate), golden 512 tokens.

| Arm | C1 | C4 | 8K prefill | KL · top-1 · NLL |
| --- | ---: | ---: | ---: | --- |
| checkpoint (BF16 projections, head) | 196 | 439 | 6,146 | 0.034 · 88.5% · 3.297 |
| **FP8 head (default)** | 222 | 441 | 6,100 | 0.036 · 88.5% · 3.300 |
| FP8 head + FP8 GDN/attention projections | 247 | 489 | 6,177 | 0.046 · 86.7% · 3.360 |

Qwen 3.8 Flash Next EXL3 K4.25, 1 RTX, MTP 3. FP8 projections fail the KL gate (+0.012) and stay opt-in (`QWEN_FP8_DECODE=on`); `QWEN_FP8_HEAD=off` keeps BF16. The target and MTP share the same head; enabling MTP adds no second vocabulary-head copy.

## Default speculation and placement

`EXPERT_BACKEND=auto` prefers resident local EXL3 experts when the planner
admits the weights, MTP, serving reservations and requested KV pool. After
local admission, an unset `SPECULATOR` selects native MTP at depth 3. Explicit
`EXPERT_BACKEND=local` uses the same speculation default. `MTP=0` retains the
legacy opt-out; explicit depth settings retain their meaning.

The default is selected by delivered C1/C4 code and reasoning-on agentic
tok/s, with C16 and 8K prefill measured alongside. Native MTP also prefills
its own attention state. Release smoke uses
the current speculation-lossless rule: a greedy flip is informational when
plain decode repeats every token and row exactly and verify rows already
differed before the flip. A sudden state change still fails. Golden NLL and
prefix-cache restore checks must pass.

## EXL3 reference configurations

The natural minimum is a 32 GB RTX + one Spark; the maximum is one RTX
PRO 6000 with resident local experts. The minimum card uses an RTX PRO
6000 at 325 W with a **32 GiB logical device ceiling**, not a physical 5090.
It validates capacity, not 5090 throughput. Both cards load
`wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1`, revision
`888306bd3996d6317758c07df50622829259ad17`.

Common launcher settings: `CONCURRENCY=4`, `MAX_CONTEXT_TOKENS=16384`,
`MAX_OUTPUT_TOKENS=4096`, `POOL_TOKENS=32768`, `PREFIX_CACHE_ENTRIES=20`,
`SPECULATOR=mtp`, `SPECULATOR_DEPTH=3`, `COPY_DRAFTS=off`. Keep the default
BF16 projections, shared single-copy FP8 head, and mapped host PLE table.

- Minimum: `RTX_GPUS=1`, `COORDINATOR_GPU_BUDGET_GIB=32`,
  `EXPERT_BACKEND=spark`, `SPARK_COUNT=1`, `SPARK_HOSTS=moa`,
  `SPARK_DEVICE_BUDGET_BYTES=110000000000`. All backbone experts are on moa;
  only the one MTP expert layer stays on the coordinator.
- Maximum: `RTX_GPUS=1`, `EXPERT_BACKEND=local`, `SPARK_COUNT=0`; omit the
  coordinator budget. Backbone and MTP experts stay resident on the RTX.

`cuteafd plan MODEL --layout --rtx 1 --rtx-budget-gib 32 --spark-ranks 1`
models the ceiling. The same launcher budget is enforced before weight,
KV, workspace and graph allocations and includes physical/untracked device
usage; it does not reserve a dummy allocation or alter SM/L2 geometry.
Mapped host PLE backing is not charged as device memory. Explicit KV pools
are refused on shortfall, not silently reduced.

The new minimum, speculation-off control and local maximum pass Release
smoke. Their teacher-forced rows are byte-identical at all 512 golden
positions; this smoke set is not a new precision qualification. Prompt-end
and turn-end prefix restores pass. MTP3 wins the matched minimum C1/C4 code
and reasoning-on code-agent measurements, so the reference retains MTP3.
Readiness includes Spark loading in the launcher, whereas the card's
readiness field measures only the coordinator process.

## Known limits

<!-- release-v2-limits -->
- **Memory vs. kernel support:** the measured checkpoints support 262,144 context tokens in the planner; this image's compiled index extent is 131,072. Corrected cards request 131,072 explicitly. Local experts fit one PRO card, but the planner separately reports the two-coordinator cache geometry as unsupported; a second RTX is not a supported split.
- No two-GPU split: the 2x RTX column is n/a rather than a duplicated one-GPU measurement. The minimum is local on one PRO card; small-card simulations may need Spark experts.

- FP8 experts have no Spark TP layout yet: 640 is not evenly divisible the
  way the FP8 MoE kernel currently tiles larger TP degrees. Spark supports
  EXL3 and NVFP4; the official FP8 checkpoint is outside the v1 smoke matrix.
- Spark workers serve backbone expert layers only. Explicit `SPECULATOR=mtp`
  keeps `mtp.layers.0.mlp.experts` resident on the coordinator; it does not
  copy backbone experts there. Spark MTP3 is qualified for EXL3 TP1. An unset
  Spark speculator remains off; use the explicit minimum settings above.
- The historical local EXL3 C1 difference includes the change from v0
  decode-only FP8 projections with dual residency to BF16 projections.
  Matched BF16 ABAB passes parity. Single-copy FP8 projections remain
  opt-in because their golden NLL/KL misses the precision gate.
- The automatic MTP default is qualified for local EXL3. Other expert formats
  retain explicit speculation settings.
- Local NVFP4 supports explicit `SPECULATOR=mtp`, `SPECULATOR_DEPTH=3`.
  Routed layers use the NVFP4 TP1 package; the MTP layer uses the FP8 TP1
  package. Both stay resident, with one copy per layer and separately
  admitted prefill scratch. This fixes the RC1 minimum-card launch failure.
  The automatic MTP default remains limited to EXL3.
- MTP verify and plain decode can differ at low-margin greedy positions,
  and C1/C4 outputs can differ. The current gate accepts proven verify
  rounding; byte-identical speculation and batch invariance are open. The
  new minimum passes its speculation probe byte-exactly; the local maximum
  passes via the existing near-tie rule. This is not a universal lossless proof.
- Tool calls parse, reasoning is retained, and short reasoning-on code-agent
  sessions succeed. Chat-template tool-result re-render still differs from
  the original token prefix and is informational, not byte-exact.
- There is no two-GPU coordinator head split; a two-RTX request serves from
  the first GPU. Spark expert placement is slower than qualified local EXL3
  at the reference configurations.
- One-RTX NVFP4 local experts must fit resident weight and serving
  reservations. Implicit paging was removed; `--expert-window` explicitly
  enables the slower paging fallback. The local automatic MTP default is
  enabled for EXL3; NVFP4 uses explicit MTP settings.
- NVFP4 decode/verify uses W4A16; native W4A4 for these small-row shapes is
  deferred.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |
| v2 | 2026-10-09 | Bundled vision auto admission with calibrated image token cap; precreated serving graphs and bounded verify rows; unified SM120/small-card admission; resident local one-RTX reference and explicit no-two-GPU-split column. | Pending rc1 exports |
| v1 | 2026-10-05 | Automatic resident local EXL3 placement and native MTP3 with a shared FP8 head; generic coordinator device-budget admission; qualified 32 GiB RTX + one Spark EXL3 minimum with coordinator-local MTP experts and resident one-RTX maximum; resident mixed NVFP4/FP8 local experts for NVFP4 MTP3, with explicit paging fallback. | <a href="../../benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-32gib-1rtx-1spark-mtp3/report.svg"><img src="../../benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-32gib-1rtx-1spark-mtp3/card.svg" width="360" alt="Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) (min)"></a> <a href="../../benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-1rtx-local-mtp3/report.svg"><img src="../../benchmarks/qwen4/2026-10-05-smoke-qwen38-exl3-1rtx-local-mtp3/card.svg" width="360" alt="Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) (max)"></a> <a href="../../benchmarks/qwen4/2026-10-05-smoke-qwen3-8-flash-next-nvfp4-1rtx-rc2/report.svg"><img src="../../benchmarks/qwen4/2026-10-05-smoke-qwen3-8-flash-next-nvfp4-1rtx-rc2/card.svg" width="360" alt="Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) (min)"></a> <a href="../../benchmarks/qwen4/2026-10-04-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark-qwen38-nvfp4-max/report.svg"><img src="../../benchmarks/qwen4/2026-10-04-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark-qwen38-nvfp4-max/card.svg" width="360" alt="Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) (max)"></a> |
| v0 | 2026-10-02 | First release | <a href="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx/report.svg"><img src="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx/card.svg" width="360" alt="Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) (min)"></a> <a href="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-4spark/report.svg"><img src="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-exl3-k4-25-ple-fp8-v1-1rtx-4spark/card.svg" width="360" alt="Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1 (exl3-k4+exl3-k5) (max)"></a> <a href="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark/report.svg"><img src="../../benchmarks/qwen4/2026-10-02-smoke-qwen3-8-flash-next-nvfp4-1rtx-4spark/card.svg" width="360" alt="Qwen3.8-Flash-Next-NVFP4 (nvfp4-g16) (min)"></a> |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
